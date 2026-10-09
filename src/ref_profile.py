"""The reference author's FP8 profile as nn.Module replacements.

Per call, both operands of every matmul are cast with the reference rule (ref_numerics.quantize,
a port of hw_float_ops.quantize_bf16_to_e4m3): a dynamic per-tensor power-of-two scale
s = floor(log2(amax_bf16 / 256)) and the vpack E4M3 encoder (quant_linear.py:153-154). Then:

  reference   F.linear on the dequantized operands in the layer dtype (NEW: the reference has no
              FP8 path without a functional model, quant_linear.py:173-175; this is the closest
              framework-accumulation analogue), BF16 bias added afterwards.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import ref_numerics as rn


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    """smolVLA model_patcher.py:520-525."""
    parent = root
    *path, last = name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, last, new)


class RefQuantLinear(nn.Module):
    """nn.Linear under the reference profile.

    The weight and bias are shared with the original layer (quant_linear.py:110-111), so
    `layer.weight.dtype` still reports the model dtype to code that casts activations by it.
    The weight's codes and scale are computed once: the reference recomputes them every call
    (quant_linear.py:154) with the same deterministic result.
    """

    def __init__(self, linear: nn.Linear, name: str = ""):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.layer_name = name
        with torch.no_grad():
            w_q, s_w = rn.quantize(self.weight.detach())
        self.register_buffer("weight_codes", w_q, persistent=False)
        self.register_buffer("weight_scale_exp", s_w, persistent=False)

    def _dequant_weight(self) -> torch.Tensor:
        return rn.dequantize(self.weight_codes, self.weight_scale_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.weight.dtype
        # Operate in the weight dtype; some activations arrive as float32 (quant_linear.py:133-137).
        x = x.to(dtype)
        x_q, s_x = rn.quantize(x)                                   # quant_linear.py:153
        x_dq = rn.dequantize(x_q, s_x).to(dtype)                    # exact in BF16 / FP32
        y = F.linear(x_dq, self._dequant_weight().to(dtype))        # NEW: framework accumulation
        if self.bias is not None:
            y = y + self.bias                                       # BF16 bias after (quant_linear.py:171-172)
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"weight_scale_exp={int(self.weight_scale_exp)}, profile=reference")


def quantize_reference(model: nn.Module, skip=()) -> dict:
    """Replace every nn.Linear (any dtype) whose fqn contains none of `skip`.

    The reference selects `type(module) is nn.Linear` (smolVLA model_patcher.py:496-499) and
    quantizes fp32 layers too (model_patcher.py:83-87; casts at quant_linear.py:136-137,170 and
    hw_float_ops.py:89), so unlike collab_v0 there is no bf16-only filter.
    """
    planned = [n for n, m in model.named_modules()
               if type(m) is nn.Linear and not any(s in n for s in skip)]
    modules = dict(model.named_modules())
    for n in planned:
        _set_module(model, n, RefQuantLinear(modules[n], name=n))
    return {"planned": len(planned), "quantized": len(planned), "untouched": []}
