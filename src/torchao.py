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
from model2MLIR import m2m

E8M0_BIAS = 127

@dataclass
class AtlasQConfig(AOBaseConfig):
    x_scale_exp: int = 0                 
    w_scale_granularity: str = "tensor"  
    
class AtlasQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: AtlasQConfig):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.bias = linear.bias
        w = linear.weight.detach()

        w_scale_exp = scale_exp_for(w)
        w_fp8 = hw_bf16_to_e4m3(w, E8M0_BIAS - w_scale_exp)

        self.register_buffer("w_fp8", w_fp8.view(torch.float8_e4m3fn))
        self.register_buffer("w_scale_exp", w_scale_exp)
        self.register_buffer("x_scale_exp",
                             torch.tensor(config.x_scale_exp, dtype=torch.int32, device=w.device))

    @property
    def weight(self):
        """Dequantized weight, for code that reads .weight (e.g. .weight.dtype). Not stored."""
        return self.w_fp8.to(torch.bfloat16) * torch.exp2(self.w_scale_exp.float())

    def forward(self, x):
        x_scale_factor = torch.exp2(self.x_scale_exp.float())
        x_fp8 = (x / x_scale_factor).clamp(-448, 448).to(torch.float8_e4m3fn)
        x_bf16 = x_fp8.to(torch.bfloat16) * x_scale_factor
        return F.linear(x_bf16, self.weight, self.bias)

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, fp8_e4m3"

@register_quantize_module_handler(AtlasQConfig)
def _atlas_handler(module: nn.Linear, config: AtlasQConfig, **kwargs) -> nn.Module:
    return AtlasQuantLinear(module, config)

def skip_lm_head(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and "lm_head" not in fqn
