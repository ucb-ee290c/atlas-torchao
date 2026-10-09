"""The reference author's FP8 profile as nn.Module replacements.

Per call, both operands of every matmul are cast with the reference rule (ref_numerics.quantize,
a port of hw_float_ops.quantize_bf16_to_e4m3): a dynamic per-tensor power-of-two scale
s = floor(log2(amax_bf16 / 256)) and the vpack E4M3 encoder (quant_linear.py:153-154). Then:

  reference   F.linear on the dequantized operands in the layer dtype (NEW: the reference has no
              FP8 path without a functional model, quant_linear.py:173-175; this is the closest
              framework-accumulation analogue), BF16 bias added afterwards.
  reference_hw the reference functional-model path (quant_linear.py:145-172): the normalized
              E4M3 operands go through an MXU accumulation model (accum_backends: --mxu ipt|sa,
              implementation ref_cpu|torch), the BF16 result is rescaled by 2^(sx+sw) and cast to
              the layer dtype, and the BF16 bias is added.
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

    def __init__(self, linear: nn.Linear, name: str = "", fm=None):
        super().__init__()
        self.fm = fm
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.layer_name = name
        with torch.no_grad():
            w_q, s_w = rn.quantize(self.weight.detach())
        self.register_buffer("weight_codes", w_q, persistent=False)
        self.register_buffer("weight_scale_exp", s_w, persistent=False)
        if fm is not None:
            # Persistent normalized weight: exact E4M3 values at scale 1 (quant_linear.py:161-162).
            # Kept as one tensor so the reference models' weight cache stays valid across calls.
            self.register_buffer("weight_norm", rn.e4m3_unpack(w_q), persistent=False)

    def _dequant_weight(self) -> torch.Tensor:
        return rn.dequantize(self.weight_codes, self.weight_scale_exp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.weight.dtype
        # Operate in the weight dtype; some activations arrive as float32 (quant_linear.py:133-137).
        x = x.to(dtype)
        x_q, s_x = rn.quantize(x)                                   # quant_linear.py:153
        if self.fm is not None:
            x_norm = rn.e4m3_unpack(x_q)                            # quant_linear.py:161
            y_unscaled = self.fm(x_norm, self.weight_norm, None, static=True)   # quant_linear.py:166
            # x 2^(sx+sw) in float32 (exact), then the layer dtype (quant_linear.py:169-170)
            scale = torch.exp2((s_x + self.weight_scale_exp).float())
            y = (y_unscaled * scale).to(dtype=dtype, device=x.device)
            if self.bias is not None:
                y = y + self.bias                                   # quant_linear.py:171-172
            return y
        x_dq = rn.dequantize(x_q, s_x).to(dtype)                    # exact in BF16 / FP32
        y = F.linear(x_dq, self._dequant_weight().to(dtype))        # NEW: framework accumulation
        if self.bias is not None:
            y = y + self.bias                                       # BF16 bias after (quant_linear.py:171-172)
        return y

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"weight_scale_exp={int(self.weight_scale_exp)}, "
                f"profile={'reference' if self.fm is None else 'reference_hw'}")


def quantize_reference(model: nn.Module, skip=(), fm_factory=None) -> dict:
    """Replace every nn.Linear (any dtype) whose fqn contains none of `skip`.

    With `fm_factory` (reference_hw), each layer gets its own accumulator instance, as the
    reference builds one functional model per layer (model_patcher.py:213-214); the reference
    CPU models cache one encoded weight per instance.

    The reference selects `type(module) is nn.Linear` (smolVLA model_patcher.py:496-499) and
    quantizes fp32 layers too (model_patcher.py:83-87; casts at quant_linear.py:136-137,170 and
    hw_float_ops.py:89), so unlike collab_v0 there is no bf16-only filter.
    """
    planned = [n for n, m in model.named_modules()
               if type(m) is nn.Linear and not any(s in n for s in skip)]
    modules = dict(model.named_modules())
    for n in planned:
        _set_module(model, n, RefQuantLinear(modules[n], name=n,
                                                   fm=fm_factory() if fm_factory else None))
    return {"planned": len(planned), "quantized": len(planned), "untouched": []}
