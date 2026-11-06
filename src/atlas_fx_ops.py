"""Custom operators that carry one quantized Linear as two opaque FX nodes (fx_export --native-fp8).

    atlas::quantize_act(x, mode, reduce, rule, static_exp, code_dtype) -> (codes, exp)
    atlas::qlinear(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format) -> y

Importing this module registers both operators (torch.library). A saved FX bundle
(`model.pt2`) names them as `atlas.quantize_act.default` / `atlas.qlinear.default`, so
`import atlas_fx_ops` MUST run before `torch.export.load` on such a bundle; `fx_export.verify_saved`
does that when the manifest says `native_fp8`. Meta/fake implementations let `torch.export` keep each
operator as one node instead of tracing into its body.

The eager bodies add no numerics. They call the code that the reference graph traces:
`ref_numerics` (`e4m3_vpack`, `dequantize`), `algos.granularity.scale_exp` (the policy's token / tensor
rule, `ref` or `nc`) and `torch.nn.functional.linear` in the layer dtype with the bias added afterwards
(`ref_profile.RefQuantLinear.forward`).

Weight format `atlas_e4m3`: E4M3 with exponent field 15 finite (max 448), 0x7F / 0xFF reserved and never
produced, exponent-field-0 codes decoded as signed zero (`ref_numerics.e4m3_unpack`). Codes may be passed
as uint8 or as torch.float8_e4m3fn (the same bytes); the body views either as the bytes. torch's
float8_e4m3fn decodes the 14 exponent-0 non-zero codes (0x01-0x07, 0x81-0x87) as subnormals where
`ref_numerics` gives zero, and 0x7F / 0xFF as NaN. The encoder never emits any of these 16 codes, and
`fx_export` refuses to bitcast a tensor that contains one.

Output rescale. The policy's `output_rescale` is y = (x_norm . w_norm) * 2^(sx+sw) in float32, then the
layer dtype, bias after. Here the power-of-two exponents are applied to the operands before the
contraction (`rn.dequantize`), as eager `ref_conv` does. For per-token / per-tensor activation exponents
and per-channel / per-tensor weight exponents the two orders are bit-identical (a power-of-two scale
commutes with fp32 accumulation and with the one BF16 output rounding while no value leaves the BF16
range); tests/test_fx_native.py measures this.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

import ref_numerics as rn
from algos.granularity import scale_exp

WEIGHT_FORMAT = "atlas_e4m3"
CODE_DTYPES = {"uint8": torch.uint8, "float8_e4m3fn": torch.float8_e4m3fn}
MODES = ("dynamic", "static")
REDUCES = ("tensor", "token")
RULES = ("ref", "nc")


def _exp_shape(x: torch.Tensor, mode: str, reduce: str):
    return [] if mode == "static" or reduce == "tensor" else list(x.shape[:-1]) + [1]


def _check_act(mode, reduce, rule, static_exp, code_dtype):
    if mode not in MODES or reduce not in REDUCES or rule not in RULES or code_dtype not in CODE_DTYPES:
        raise ValueError(f"atlas::quantize_act: unsupported mode={mode!r} reduce={reduce!r} "
                         f"rule={rule!r} code_dtype={code_dtype!r}")
    if (mode == "static") != (static_exp is not None):
        raise ValueError("atlas::quantize_act: static_exp is required exactly when mode='static'")


@torch.library.custom_op("atlas::quantize_act", mutates_args=())
def quantize_act(x: torch.Tensor, mode: str, reduce: str, rule: str, static_exp: Optional[int],
                 code_dtype: str) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 activation -> (E4M3 codes, int32 power-of-two exponents), real = e4m3(code) * 2^exp.

    mode='dynamic': exponent from the activation itself, `scale_exp(x, dims, rule)` with dims=(-1,)
    for reduce='token' (one exponent per row, shape [..., 1]) or None for reduce='tensor' (shape []);
    rule 'ref' = floor(log2(amax)) - 8, 'nc' = the no-clip rule (algos/granularity.py).
    mode='static': the fixed exponent `static_exp` (shape []). code_dtype selects uint8 or float8_e4m3fn
    for the returned codes (the same bytes)."""
    _check_act(mode, reduce, rule, static_exp, code_dtype)
    xb = x.to(torch.bfloat16)
    if mode == "static":
        s = torch.tensor(int(static_exp), dtype=torch.int32, device=x.device)
    else:
        s = scale_exp(xb, (-1,) if reduce == "token" else None, rule)
    codes = rn.e4m3_vpack(xb, s)
    if code_dtype == "float8_e4m3fn":
        codes = codes.view(torch.float8_e4m3fn)
    return codes, s.to(torch.int32)


@quantize_act.register_fake
def _(x, mode, reduce, rule, static_exp, code_dtype):
    _check_act(mode, reduce, rule, static_exp, code_dtype)
    return (torch.empty(x.shape, dtype=CODE_DTYPES[code_dtype], device=x.device),
            torch.empty(_exp_shape(x, mode, reduce), dtype=torch.int32, device=x.device))


@torch.library.custom_op("atlas::qlinear", mutates_args=())
def qlinear(x_codes: torch.Tensor, x_exp: torch.Tensor, w_codes: torch.Tensor, w_exp: torch.Tensor,
            bias: Optional[torch.Tensor], out_dtype: str, weight_format: str) -> torch.Tensor:
    """FP8 x FP8 linear on E4M3 codes with power-of-two exponents; one Linear = one node.

    y = linear(e4m3(x_codes) * 2^x_exp, e4m3(w_codes) * 2^w_exp) in `out_dtype`, then + bias (the layer
    dtype), exactly `RefQuantLinear.forward` (ref_conv). x_exp is [..., 1] or []; w_exp is [out, 1] or [];
    w_codes is [out, in]. Per-block exponents are not supported by this operator."""
    if weight_format != WEIGHT_FORMAT:
        raise ValueError(f"atlas::qlinear: weight_format {weight_format!r}, expected {WEIGHT_FORMAT!r}")
    dtype = getattr(torch, out_dtype)
    x_dq = rn.dequantize(x_codes, x_exp).to(dtype)
    w_dq = rn.dequantize(w_codes, w_exp).to(dtype)
    y = F.linear(x_dq, w_dq)
    return y if bias is None else y + bias


@qlinear.register_fake
def _(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format):
    if weight_format != WEIGHT_FORMAT:
        raise ValueError(f"atlas::qlinear: weight_format {weight_format!r}, expected {WEIGHT_FORMAT!r}")
    return torch.empty(list(x_codes.shape[:-1]) + [w_codes.shape[0]], dtype=getattr(torch, out_dtype),
                       device=x_codes.device)


OPS = (torch.ops.atlas.quantize_act.default, torch.ops.atlas.qlinear.default)
