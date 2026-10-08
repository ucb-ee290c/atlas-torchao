from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchao.core.config import AOBaseConfig
from torchao.quantization import quantize_
from torchao.quantization.transform_module import register_quantize_module_handler
from utils import load_llama, load_model
from quant import hw_bf16_to_e4m3, hw_e4m3_to_float, scale_exp_for
from torch.fx.passes.graph_drawer import FxGraphDrawer

E8M0_BIAS = 127

@dataclass
class AtlasQConfig(AOBaseConfig):
    x_scale_exp: int = 0                 # activation scale exponent: 2^0 = 1 (matches seli 127)
    w_scale_granularity: str = "tensor"  # how weight scales are chosen; "tile" (32x32) possible later
    
class AtlasQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, config: AtlasQConfig):
        super().__init__()
        w = linear.weight.detach()

        # weight: real value = fp8 * 2^w_scale_exp  (computed from this layer's weights)
        w_scale_exp = scale_exp_for(w)
        w_fp8 = hw_bf16_to_e4m3(w, E8M0_BIAS - w_scale_exp)   # converter multiplies -> pass -exp

        self.register_buffer("w_fp8", w_fp8.view(torch.float8_e4m3fn))
        self.register_buffer("w_scale_exp", w_scale_exp)
        self.register_buffer("x_scale_exp",
                             torch.tensor(config.x_scale_exp, dtype=torch.int32, device=w.device))
        self.bias = linear.bias

    def forward(self, x):
        # activation: real value = fp8 * 2^x_scale_exp
        x_scale_factor = torch.exp2(self.x_scale_exp.float())
        x_fp8 = (x / x_scale_factor).clamp(-448, 448).to(torch.float8_e4m3fn)
        x_bf16 = x_fp8.to(torch.bfloat16) * x_scale_factor

        # weight: real value = fp8 * 2^w_scale_exp
        w_scale_factor = torch.exp2(self.w_scale_exp.float())
        w_bf16 = self.w_fp8.to(torch.bfloat16) * w_scale_factor

        return F.linear(x_bf16, w_bf16, self.bias)


@register_quantize_module_handler(AtlasQConfig)
def _atlas_handler(module: nn.Linear, config: AtlasQConfig, **kwargs) -> nn.Module:
    return AtlasQuantLinear(module, config)

def skip_lm_head(module: nn.Module, fqn: str) -> bool:
    return isinstance(module, nn.Linear) and "lm_head" not in fqn

def save_for_merlin(model: nn.Module, path: str):
    package = {}
    for name, m in model.named_modules():
        if isinstance(m, AtlasQuantLinear):
            package[name] = {
                "w_fp8": m.w_fp8.cpu(),
                "w_scale_exp": int(m.w_scale_exp),
                "bias": None if m.bias is None else m.bias.detach().cpu(),
            }
    torch.save(package, path)


if __name__ == "__main__":
    MODEL_PATH = "/rscratch/agarcia/TinyLlama-1.1B-Chat-v1.0"
    model = load_llama(MODEL_PATH, seqlen=2048)
     
    with torch.no_grad():
        quantize_(model, AtlasQConfig(), filter_fn=skip_lm_head)
    model.eval()
    device = next(model.parameters()).device

    # 1) One quantized layer: small, readable graph
    x = torch.randn(1, 8, 2048, dtype=torch.bfloat16, device=device)

    # 2) One MLP block: the first milestone
    mlp = model.model.layers[0].mlp
    from torch_mlir import fx
    from torch_mlir.compiler_utils import OutputType
    m = fx.export_and_import(mlp, x, output_type=OutputType.LINALG_ON_TENSORS)
    open("mlp_layer0_linalg.mlir", "w").write(str(m))

    """ 
    with torch.no_grad():
        ep_mlp = torch.export.export(mlp, (x,))
    with open("mlp_layer0_graph.txt", "w") as f:
        f.write(str(ep_mlp.graph))
    print("MLP graph:", len(ep_mlp.graph.nodes), "nodes -> mlp_layer0_graph.txt")

    drawer = FxGraphDrawer(ep_mlp.graph_module, "mlp_layer0")
    drawer.get_dot_graph().write_svg("mlp_layer0_graph.svg")
    """