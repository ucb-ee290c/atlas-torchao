from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler
from modelutils import load_llama, load_smolvla
from atlas_quant import hw_bf16_to_e4m3, hw_e4m3_to_float, scale_exp_for
from torch.fx.passes.graph_drawer import FxGraphDrawer

E8M0_BIAS = 127
FP8_MAX = 448.0          # change to 240.0 if exponent 15 turns out to be outside the MXU operand range

@dataclass
class AtlasQConfig(AOBaseConfig):
    input_scale_exp: int = 0                 
    
class AtlasQuantLinear(nn.Module):
    """FP8 E4M3 linear for Atlas.
    real weight     = weight (fp8) * 2^weight_scale_exp
    real activation = fp8(x / 2^input_scale_exp) * 2^input_scale_exp
    """
    def __init__(self, linear: nn.Linear, config: AtlasQConfig):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.quantize_activations = config.quantize_activations 
        w = linear.weight.detach()

        weight_scale_exp = scale_exp_for(w)                                    # int32
        weight_fp8 = hw_bf16_to_e4m3(w, E8M0_BIAS - weight_scale_exp)          # raw bytes

        # named buffers -> appear in the manifest as <module>.weight, <module>.weight_scale_exp, ...
        self.register_buffer("weight", weight_fp8.view(torch.float8_e4m3fn))  # float8_e4m3fn
        self.register_buffer("weight_scale_exp", weight_scale_exp)
        self.register_buffer("input_scale_exp",
                             torch.tensor(config.input_scale_exp, dtype=torch.int32, device=w.device))
        self.bias = linear.bias

    def forward(self, x):
        # activation: BF16 -> FP8 with a fixed scale (vpack on the chip)
        sx = torch.exp2(self.input_scale_exp.float())
        x_fp8 = (x / sx).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        x_bf16 = x_fp8.to(torch.bfloat16) * sx

        # weight: FP8 -> BF16 with its scale
        w_bf16 = self.weight.to(torch.bfloat16) * torch.exp2(self.weight_scale_exp.float())

        return F.linear(x_bf16, w_bf16, self.bias)

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"fp8_e4m3, weight_scale_exp={int(self.weight_scale_exp)}, "
                f"input_scale_exp={int(self.input_scale_exp)}")
        
@register_quantize_module_handler(AtlasQConfig)
def _atlas_handler(module: nn.Linear, config: AtlasQConfig, **kwargs) -> nn.Module:
    return AtlasQuantLinear(module, config)

def skip_lm_head(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and "lm_head" not in fqn
