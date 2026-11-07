"""Custom operators that carry one quantized Linear as two opaque FX nodes (fx_export --native-fp8).

    atlas::quantize_act(x, mode, reduce, rule, static_exp, code_dtype, tag="", rows=32) -> (codes, exp)
    atlas::qlinear(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format) -> y
    atlas::qconv2d(x_cols_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format, B, oh, ow) -> y
    atlas::qk_matmul(q_codes, q_exp, k_codes, k_exp) -> scores          (float32, before the head_dim scaling)
    atlas::pv_matmul(p_codes, p_exp, vt_codes, v_exp) -> out            (float32; vt = V^T)

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
from algos.granularity import row_group_scale_exp, scale_exp

WEIGHT_FORMAT = "atlas_e4m3"
CODE_DTYPES = {"uint8": torch.uint8, "float8_e4m3fn": torch.float8_e4m3fn}
MODES = ("dynamic", "static")
REDUCES = ("tensor", "token", "head", "block", "head_block")   # block, head_block: NEW (tile-aware)
RULES = ("ref", "nc")


def _exp_shape(x: torch.Tensor, mode: str, reduce: str):
    if mode == "static" or reduce == "tensor":
        return []
    if reduce in ("token", "block", "head_block"):          # block kinds: the per-token table, group values repeated
        return list(x.shape[:-1]) + [1]
    return list(x.shape[:-2]) + [1, 1]


def _check_act(mode, reduce, rule, static_exp, code_dtype, rows=32):
    if int(rows) < 1:
        raise ValueError(f"atlas::quantize_act: rows={rows}; must be >= 1")
    if mode not in MODES or reduce not in REDUCES or rule not in RULES or code_dtype not in CODE_DTYPES:
        raise ValueError(f"atlas::quantize_act: unsupported mode={mode!r} reduce={reduce!r} "
                         f"rule={rule!r} code_dtype={code_dtype!r}")
    if (mode == "static") != (static_exp is not None):
        raise ValueError("atlas::quantize_act: static_exp is required exactly when mode='static'")


@torch.library.custom_op("atlas::quantize_act", mutates_args=())
def quantize_act(x: torch.Tensor, mode: str, reduce: str, rule: str, static_exp: Optional[int],
                 code_dtype: str, tag: str = "", rows: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """BF16 activation -> (E4M3 codes, int32 power-of-two exponents), real = e4m3(code) * 2^exp.

    mode='dynamic': exponent from the activation itself, `scale_exp(x, dims, rule)` with dims=(-1,)
    for reduce='token' (one exponent per row, shape [..., 1]), (-2, -1) for reduce='head' (one per
    batch x head matrix, shape [..., 1, 1]: the attention operands' reference_dims) or None for
    reduce='tensor' (shape []);
    reduce='block' (NEW, tile-aware): one exponent per `rows` consecutive rows (default 32) over the full
    K, the rows being the flattened leading dims of x ([B, T, K] -> B*T, row-major; the last group may be
    short), returned as the per-token table [..., 1] with the group value repeated, so qlinear is unchanged;
    reduce='head_block' (NEW): the same per `rows` rows of each (batch, head) matrix [..., S, D] along S,
    shape [..., S, 1] (attention q and p);
    rule 'ref' = floor(log2(amax)) - 8, 'nc' = the no-clip rule (algos/granularity.py).
    mode='static': the fixed exponent `static_exp` (shape []). code_dtype selects uint8 or float8_e4m3fn
    for the returned codes (the same bytes). `tag` is a label only (the attention operands carry
    "<policy site>:<q|k|p|v>"); it does not change the result. `rows` (NEW, last, default 32) is read only
    by reduce='block' / 'head_block'; bundles saved before it existed call the operator without it."""
    _check_act(mode, reduce, rule, static_exp, code_dtype, rows)
    xb = x.to(torch.bfloat16)
    if mode == "static":
        s = torch.tensor(int(static_exp), dtype=torch.int32, device=x.device)
    else:
        if reduce == "block":
            s = row_group_scale_exp(xb.reshape(-1, xb.shape[-1]), int(rows), rule).reshape(*xb.shape[:-1], 1)
        elif reduce == "head_block":
            s = row_group_scale_exp(xb, int(rows), rule)
        else:
            s = scale_exp(xb, {"token": (-1,), "head": (-2, -1), "tensor": None}[reduce], rule)
    codes = rn.e4m3_vpack(xb, s)
    if code_dtype == "float8_e4m3fn":
        codes = codes.view(torch.float8_e4m3fn)
    return codes, s.to(torch.int32)


@quantize_act.register_fake
def _(x, mode, reduce, rule, static_exp, code_dtype, tag="", rows=32):
    _check_act(mode, reduce, rule, static_exp, code_dtype, rows)
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


@torch.library.custom_op("atlas::qconv2d", mutates_args=())
def qconv2d(x_codes: torch.Tensor, x_exp: torch.Tensor, w_codes: torch.Tensor, w_exp: torch.Tensor,
            bias: Optional[torch.Tensor], out_dtype: str, weight_format: str, batch: int, out_h: int,
            out_w: int) -> torch.Tensor:
    """Conv2d as FP8 x FP8 matmul on im2col'd codes: x_codes [B*oh*ow, in*kh*kw] (the unfolded input,
    quantized per row), w_codes [out, in*kh*kw] (the export's flattened weight). Same contraction as
    `qlinear`, then `RefQuantConv2d.forward`'s tail: [B*oh*ow, out] -> [B, out, oh, ow], bias added in the
    layer dtype. The im2col (F.unfold, permute, reshape) stays a visible node before quantize_act."""
    if weight_format != WEIGHT_FORMAT:
        raise ValueError(f"atlas::qconv2d: weight_format {weight_format!r}, expected {WEIGHT_FORMAT!r}")
    dtype = getattr(torch, out_dtype)
    out_channels = w_codes.shape[0]
    y = F.linear(rn.dequantize(x_codes, x_exp).to(dtype), rn.dequantize(w_codes, w_exp).to(dtype))
    y = y.reshape(batch, out_h * out_w, out_channels).permute(0, 2, 1).reshape(batch, out_channels, out_h, out_w)
    return y if bias is None else y + bias.view(1, -1, 1, 1).to(dtype=dtype, device=y.device)


@qconv2d.register_fake
def _(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, weight_format, batch, out_h, out_w):
    if weight_format != WEIGHT_FORMAT:
        raise ValueError(f"atlas::qconv2d: weight_format {weight_format!r}, expected {WEIGHT_FORMAT!r}")
    return torch.empty([batch, w_codes.shape[0], out_h, out_w], dtype=getattr(torch, out_dtype),
                       device=x_codes.device)


@torch.library.custom_op("atlas::qk_matmul", mutates_args=())
def qk_matmul(q_codes: torch.Tensor, q_exp: torch.Tensor, k_codes: torch.Tensor,
              k_exp: torch.Tensor) -> torch.Tensor:
    """Attention scores Q.K^T on E4M3 codes, float32 out: q [..., Sq, D], k [..., Skv, D], exponents
    [..., 1, 1] (per head), [..., Sq, 1] (per token) or []. `ref_numerics.dq_matmul_nt`, the body of
    ref_attention._mm. The head_dim scaling, mask and softmax are separate BF16/float32 nodes."""
    return rn.dq_matmul_nt(q_codes, q_exp, k_codes, k_exp)


@qk_matmul.register_fake
def _(q_codes, q_exp, k_codes, k_exp):
    return _matmul_nt_fake(q_codes, k_codes)


@torch.library.custom_op("atlas::pv_matmul", mutates_args=())
def pv_matmul(p_codes: torch.Tensor, p_exp: torch.Tensor, vt_codes: torch.Tensor,
              v_exp: torch.Tensor, pop_shift: int = 0) -> torch.Tensor:
    """Attention output P.V on E4M3 codes, float32 out: p [..., Sq, Skv], vt = V^T [..., Dv, Skv] (the
    transposed view `ref_attention` hands to `_mm`), v_exp [..., 1, 1]. Same body as qk_matmul.
    pop_shift (NEW, last, default 0; docs/algos/pack-fold.md): the output is multiplied by 2^pop_shift, i.e. the PV pop
    rescale constant is 2^(sp + sv + pop_shift). Exact in float32; `fx_export` passes it only when it is non-zero, so every
    other node (and every bundle saved before it existed) keeps its old argument list."""
    y = rn.dq_matmul_nt(p_codes, p_exp, vt_codes, v_exp)
    return y if not pop_shift else y * (2.0 ** int(pop_shift))


@pv_matmul.register_fake
def _(p_codes, p_exp, vt_codes, v_exp, pop_shift=0):
    return _matmul_nt_fake(p_codes, vt_codes)


def _matmul_nt_fake(a, b):
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return torch.empty(list(batch) + [a.shape[-2], b.shape[-2]], dtype=torch.float32, device=a.device)


OPS = (torch.ops.atlas.quantize_act.default, torch.ops.atlas.qlinear.default, torch.ops.atlas.qconv2d.default,
       torch.ops.atlas.qk_matmul.default, torch.ops.atlas.pv_matmul.default)
