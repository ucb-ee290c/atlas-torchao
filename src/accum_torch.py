"""Device-agnostic torch ports of the reference author's Atlas MXU accumulation models.

Two functional models, both bit-exact ports of the reference sources at pi0-quant 8bc0f6a:

    sa_linear   mxu0, the systolic array (func_models_sa/systolic_c: systolic_array_model.h,
                sa_linear.h == systolic_array_linear.h, systolic_array_rtl_linear.py)
    ipt_linear  mxu1, the inner-product trees, with two variants
                  "numba"  funct_models_ipt/ipt_numba (_numba_kernels.py, ipt_rtl_linear.py),
                           the reference author's choice and the default
                  "c"      funct_models_ipt/ipt_c (inner_product_trees_model.h, ipt_linear.h)

Call contract (src/accum_backends.py on reference-profile; pi0_inout/functional_models.py:7-12):

    y = sa_linear(x, w, b=None, *, rows=32)       y = ipt_linear(x, w, b=None, *, vec_len=32)

x is [*, K] with w [N, K] -> y [*, N], or batched x [*B, M, K] with w [*B, N, K] -> y [*B, M, N].
x, w and b are float32 values (encoded with the reference encoder, see float_to_e4m3_bytes) or
raw E4M3 bytes (torch.uint8, used as-is, as systolic_array_rtl_linear.py:273-277 does).
y is float32 on x.device holding the decoded model output (ipt_rtl_linear.py:86-90).
The *_bits variants return the raw uint16 output words as int32.

Why the GPU cannot diverge from the reference (NEW design; re-verify on CUDA, see below):
  - Every rounding the reference performs is emulated in integer tensor arithmetic on the bit
    fields (int32/int64 shifts, masks, compares, where). No floating-point add, multiply,
    matmul, bmm, einsum, conv or float reduction is used for the emulated accumulation, so there
    are no tensor cores, no TF32, no reduced-precision reductions and nothing an FMA could
    contract: each rounding step (product, align, add, normalise, round, clamp) is its own
    materialised integer tensor op.
  - Products are exact table lookups (the reference's own 65536-entry LUT); SA adds run in a
    Python loop over K in index order, one E4M3ProdAddBF16 per element, exactly as the C model.
  - The only reduction is IPT's int64 sum of aligned products within a tile. Integer addition
    is associative and the reference wraps modulo 2^intWidth either per add (C) or once (numba),
    which are the same modular sum, so the result is independent of reduction order.
  - The only float use is an exact int -> float64 conversion to read a bit length from the
    exponent field (values < 2^53, so exact on any IEEE device; no arithmetic on it).
  - Ops used (gather by advanced indexing, where, shifts, amax/sum over int) are deterministic
    on CPU and CUDA; torch.use_deterministic_algorithms(True) is compatible (tested on CPU).
  - The TF32 / reduced-precision flags do not touch this path. enforce_exact_settings() sets
    them off anyway and numerics_report() records them for result JSONs.
To confirm on the CUDA machine (cannot be checked on this CPU-only host): the CUDA tests in
tests/test_accum_torch.py (CUDA == CPU bits, run-to-run determinism), the full suite with
--device cuda semantics via scripts/bench_accum.py --check, and the torch.compile path
(Triton codegen of int32 shifts by tensor amounts: shift counts here are always in [0, 31]
for int32 and [0, 63] for int64, which is defined behaviour in both C++ and PTX).

Comments cite the reference file:line each behaviour comes from. NEW marks choices with no
reference source (vectorisation, chunking, the compile flag, argument handling).
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch

__all__ = [
    "sa_linear", "ipt_linear", "sa_linear_bits", "ipt_linear_bits",
    "float_to_e4m3_bytes", "decode_output_bits", "e4m3_mul_to_prod_lut",
    "sa_prod_add_bf16", "output_conv_stage", "OUT_BF16", "OUT_E4M3",
    "enforce_exact_settings", "numerics_report",
]

# OutputFmtSel (fp_formats.h:62-66)
OUT_BF16 = 0
OUT_E4M3 = 1

# Product format E5M7, bias 13 (fp_formats.h:49; converters.h:19-22)
_PROD_BIAS = 13
# BF16 bias minus product bias (systolic_array_model.h:95)
_SA_EXP_BIAS_OFFSET = 114
# IPT exponent sentinel for "no exponent": -(1 << (expWorkWidth - 2)), expWorkWidth = 8 + 4
# (inner_product_trees_model.h:122; params_and_requests.h:95-97; _numba_kernels.py:252)
_SENTINEL = -1024

# NEW: elements per [*, M, Nc, vec_len] product block in ipt_linear (memory bound, not numerics)
_DEFAULT_CHUNK_ELEMS = 1 << 24


def _fmt_value(sel) -> int:
    """Accept an int or an OutputFmtSel enum (ipt_rtl_linear.py:87 does the same)."""
    return int(getattr(sel, "value", sel))


# ---------------------------------------------------------------------------------------------
# Shared pieces: product LUT, output stage, encoder, decoder
# ---------------------------------------------------------------------------------------------

_LUT_CACHE: dict = {}


def e4m3_mul_to_prod_lut(device=None) -> torch.Tensor:
    """int32[65536] indexed by (a << 8) | b: the 13-bit E5M7 product of two E4M3 bytes.

    converters.h:55-94 (_build_e4m3_mul_to_prod_lut), identical to the Python LUT that
    ipt_numba uses (_numba_kernels.py:18-21; ipt_numba/converters.py):
      - either operand with exponent field 0 (zero or subnormal) gives a signed zero (:73-77)
      - exponent field 15 is an ordinary finite exponent, no NaN/Inf handling (:60-71)
      - the 4x4-bit significand product is exact; normalise by one bit if bit 7 is set (:79-86)
      - exponent field = a_exp + b_exp + need_shift - 1, in [1, 30] (:86)
    """
    key = str(torch.device(device) if device is not None else torch.device("cpu"))
    lut = _LUT_CACHE.get(key)
    if lut is None:
        i = torch.arange(65536, dtype=torch.int32)
        a, b = i >> 8, i & 0xFF
        a_sign, a_exp, a_man = (a >> 7) & 1, (a >> 3) & 0xF, a & 0x7
        b_sign, b_exp, b_man = (b >> 7) & 1, (b >> 3) & 0xF, b & 0x7
        out_sign = a_sign ^ b_sign
        prod_sig = (8 | a_man) * (8 | b_man)
        need_shift = (prod_sig >> 7) & 1
        out_man = torch.where(need_shift == 1, prod_sig & 0x7F, (prod_sig & 0x3F) << 1)
        out_exp = a_exp + b_exp + need_shift - 1
        val = (out_sign << 12) | ((out_exp & 0x1F) << 7) | (out_man & 0x7F)
        zero = (a_exp == 0) | (b_exp == 0)
        lut = torch.where(zero, out_sign << 12, val).to(torch.int32).to(device)
        _LUT_CACHE[key] = lut
    return lut


def _sanitize_bf16(bits: torch.Tensor) -> torch.Tensor:
    """fp_formats.h:319-338: BF16 subnormal -> +0, NaN -> +0, +-Inf -> +-max finite (0x7F7F)."""
    exp = (bits >> 7) & 0xFF
    frac = bits & 0x7F
    sat = torch.where((bits & 0x8000) != 0, 0xFF7F, 0x7F7F)
    out = torch.where((exp == 0) & (frac != 0), 0, bits)
    return torch.where(exp == 0xFF, torch.where(frac != 0, 0, sat), out)


def _bf16_scale_to_e4m3(bits: torch.Tensor, scale_exp) -> torch.Tensor:
    """converters.h:261-300 (== _numba_kernels.py:192-219)."""
    sign = (bits >> 15) & 1
    exp = (bits >> 7) & 0xFF
    frac = bits & 0x7F
    sat = torch.where(sign == 1, 0xFE, 0x7E)                              # fp_formats.h:74-75
    scaled = exp - 127 + scale_exp                                         # :279
    mant8 = 0x80 | frac                                                    # :280
    # round_right_shift4_rne (fp_formats.h:304-310)
    trunc = (mant8 >> 4) & 0xF
    guard = (mant8 >> 3) & 1
    sticky = ((mant8 & 0x7) != 0).to(bits.dtype)
    rn = trunc + (guard & (sticky | (trunc & 1)))
    carry = rn == 16                                                       # :284-293
    final = torch.where(carry, scaled + 1, scaled)
    mant = torch.where(carry, 0, (rn - 8) & 0x7)
    enc = (sign << 7) | (((final + 7) & 0xF) << 3) | mant                  # fp_formats.h:267-272
    out = torch.where(final > 8, sat, torch.where(final >= -6, enc, 0))    # :295-299
    out = torch.where(exp == 0, 0, out)                                    # :269-270
    return torch.where(exp == 0xFF, torch.where(frac != 0, 0, sat), out)   # :272-277


def output_conv_stage(bits: torch.Tensor, out_fmt_sel=OUT_BF16, scale_exp=0) -> torch.Tensor:
    """converters.h:308-316 (== _numba_kernels.py:222-227): sanitize, then BF16 or E4M3 byte."""
    s = _sanitize_bf16(bits)
    if _fmt_value(out_fmt_sel) == OUT_BF16:
        return s
    return _bf16_scale_to_e4m3(s, scale_exp) & 0xFF


def float_to_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """The functional models' input encoder, float -> E4M3 byte (uint8), on x.device.

    Port of shim_float_to_e4m3 (ipt_c/ipt_rtl_linear_c.py:240-287), which SA uses too
    (systolic_array_rtl_linear.py:23-27, 279-280). It equals ipt_numba's
    _float_to_e4m3_byte_scalar (ipt_numba/ipt_rtl_linear.py:43-72) on every float32 input
    (tests/test_accum_torch.py checks both). Done on the float32 bit pattern, so no log2:
      - NaN -> 0 (:245); |v| >= 496 or Inf -> 0x7E / 0xFE by the sign of v (:247-253)
      - |v| < 2^-6 (including -0) -> 0, no subnormal outputs (:255-258)
      - exponent = floor(log2|v|) = the float32 exponent, exact since |v| >= 2^-6 is normal (:260)
      - 3-bit fraction by round-half-even of (|v| / 2^exp - 1) * 8 (:264-277), i.e. RNE of the
        23-bit float32 fraction to 3 bits; a carry to 8 bumps the exponent (:279)
      - exponent 8 encodes as field 15, so (464, 496) gives 0x7F = 480 (:281-285)
    """
    xf = x.detach().to(torch.float32)
    u = xf.view(torch.int32)
    sign = ((u >> 31) & 1).to(torch.int32)          # v < 0 for every non-NaN, non-zero input
    exp_f = (u >> 23) & 0xFF
    m23 = u & 0x7FFFFF
    is_nan = (exp_f == 0xFF) & (m23 != 0)
    sat_in = (exp_f == 0xFF) | (exp_f >= 127 + 8) & ((exp_f > 127 + 8) | (m23 >= 0x780000))  # >= 496
    small = exp_f < 127 - 6                          # |v| < 2^-6, including zeros and subnormals
    e = exp_f - 127
    q = m23 >> 20
    r = m23 & 0xFFFFF
    up = ((r > 0x80000) | ((r == 0x80000) & ((q & 1) == 1))).to(torch.int32)
    frac = q + up
    carry = frac >= 8
    frac = torch.where(carry, 0, frac)
    e = torch.where(carry, e + 1, e)
    sat = torch.where(sign == 1, 0xFE, 0x7E)
    enc = (sign << 7) | (((e + 7) & 0xF) << 3) | (frac & 0x7)
    out = torch.where(e > 8, sat, enc)               # :281
    out = torch.where(small, 0, out)
    out = torch.where(sat_in, sat, out)
    out = torch.where(is_nan, 0, out)
    return out.to(torch.uint8)


def _e4m3_output_lut(device) -> torch.Tensor:
    """decode_model_output_bits' E4M3 table (ipt_numba/ipt_rtl_linear.py:14-21): normal values
    only; exponent field 0 and 15 decode to 0. The ipt_c table (ipt_rtl_linear_c.py:208-229)
    also decodes subnormals, but the output stage never emits one (converters.h:297-299)."""
    i = torch.arange(256, dtype=torch.int32)
    sign, exp, frac = (i >> 7) & 1, (i >> 3) & 0xF, i & 0x7
    val = (1.0 + frac.double() / 8.0) * torch.exp2((exp - 7).double())
    val = torch.where((exp > 0) & (exp < 0xF), val, torch.zeros_like(val))
    return torch.where(sign == 1, -val, val).to(torch.float32).to(device)


def decode_output_bits(bits: torch.Tensor, out_fmt_sel=OUT_BF16) -> torch.Tensor:
    """decode_model_output_bits (ipt_numba/ipt_rtl_linear.py:86-90), used by SA too
    (systolic_array_rtl_linear.py:28, 290). BF16: bits << 16 viewed as float32."""
    if _fmt_value(out_fmt_sel) == OUT_BF16:
        return ((bits.to(torch.int32) & 0xFFFF) << 16).view(torch.float32)
    return _e4m3_output_lut(bits.device)[(bits & 0xFF).long()]


def _bit_length(v: torch.Tensor) -> torch.Tensor:
    """bit_length of non-negative integers < 2^53, via the exact float64 exponent (NEW helper).
    Returns 0 for 0."""
    d = v.to(torch.float64).view(torch.int64)
    bl = ((d >> 52) & 0x7FF) - 1022
    return torch.where(v == 0, 0, bl).to(v.dtype)


def enforce_exact_settings() -> dict:
    """NEW: switch off every reduced-precision float matmul/conv mode and return the report.
    The integer emulation does not depend on these; this is belt and braces for runs that
    record them. Note it changes them process-wide (they also govern the model's own
    non-emulated float layers)."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    rep = numerics_report()
    assert not rep["cuda_matmul_allow_tf32"] and not rep["cudnn_allow_tf32"]
    return rep


def numerics_report() -> dict:
    """NEW: facts about how this process computes, for result JSONs."""
    m = torch.backends.cuda.matmul
    return {
        "accum_arithmetic": "integer bit-field emulation (no float matmul/reduction)",
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_matmul_allow_tf32": bool(m.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_fp16_reduced_precision_reduction":
            bool(m.allow_fp16_reduced_precision_reduction),
        "cuda_matmul_allow_bf16_reduced_precision_reduction":
            bool(m.allow_bf16_reduced_precision_reduction),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "compile_default": _compile_default(),
    }


# ---------------------------------------------------------------------------------------------
# Operand handling (NEW: argument plumbing around the reference tiling loops)
# ---------------------------------------------------------------------------------------------

def _as_bytes(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.uint8:                     # raw bytes bypass (systolic_array_rtl_linear.py:273-277)
        return t
    return float_to_e4m3_bytes(t)


def _prepare(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor]):
    """Flatten to x [B, M, K], w [B, N, K], b [B, N] or None (int32 bytes on x.device).

    2-D w: x [*, K] is flattened to rows (ipt_rtl_linear.py:215-219), y is [*, N].
    Batched w [*B, N, K] with x [*B, M, K]: y [*B, M, N] (accum_backends.py contract, NEW).
    """
    device = x.device
    K = x.shape[-1]
    if w.shape[-1] != K:
        raise ValueError(f"in_features mismatch: x {tuple(x.shape)} vs w {tuple(w.shape)}")
    xb = _as_bytes(x).to(device=device, dtype=torch.int32)
    wb = _as_bytes(w).to(device=device, dtype=torch.int32)
    N = w.shape[-2]
    if w.dim() == 2:
        out_shape = (*x.shape[:-1], N)
        xb = xb.reshape(1, math.prod(x.shape[:-1]), K)
        wb = wb.reshape(1, N, K)
    else:
        lead = w.shape[:-2]
        if x.dim() != w.dim() or x.shape[:-2] != lead:
            raise ValueError(f"batched form needs x [*B, M, K] with w [*B, N, K]; "
                             f"got x {tuple(x.shape)}, w {tuple(w.shape)}")
        out_shape = (*lead, x.shape[-2], N)
        xb = xb.reshape(-1, x.shape[-2], K)
        wb = wb.reshape(-1, N, K)
    bb = None
    if b is not None:
        bb = _as_bytes(b).to(device=device, dtype=torch.int32)
        bb = bb.reshape(-1, N).expand(wb.shape[0], N) if bb.dim() > 1 else bb.reshape(1, N).expand(wb.shape[0], N)
    return xb, wb, bb, out_shape


def _n_chunks(N: int, per_col: int, chunk_elems: int):
    step = max(1, chunk_elems // max(1, per_col))
    return [(n0, min(N, n0 + step)) for n0 in range(0, N, step)]


# ---------------------------------------------------------------------------------------------
# SA (mxu0): systolic array
# ---------------------------------------------------------------------------------------------

def sa_prod_add_bf16(p: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """E4M3ProdAddBF16: 13-bit E5M7 product + BF16 addend -> BF16, elementwise on int32 tensors.

    Line-for-line port of sa_e4m3_prod_add_bf16 (systolic_array_model.h:97-220), which
    translates E4M3ProdAddBF16.scala. Not an IEEE add: the result is the RNE sum except that
    an exponent <= 0 after normalisation flushes to +0 (:197, :205-209), overflow clamps to
    max finite 0x7F7F/0xFF7F (:198-204), an exact cancellation is +0 (:217-218), a zero
    product passes the addend through unchanged (signed zeros kept, :118-119), and a zero
    (exponent-0) addend returns the product widened to BF16 with its sign (:120-121).
    """
    # 1. Unpack product (:103-108)
    p_sign = (p >> 12) & 1
    p_exp5 = (p >> 7) & 0x1F
    p_man7 = p & 0x7F
    p_zero = p_exp5 == 0
    p_exp8 = (p_exp5 + _SA_EXP_BIAS_OFFSET) & 0xFF
    # 2. Unpack addend (:111-115)
    a_sign = (a >> 15) & 1
    a_exp8 = (a >> 7) & 0xFF
    a_man7 = a & 0x7F
    a_zero = a_exp8 == 0
    # 4. Extend to 11 bits, GRS = 3 (:124-125); sig8 = 0x80 | man7 for non-zero operands
    p_sig11 = (0x80 | p_man7) << 3
    a_sig11 = (0x80 | a_man7) << 3
    # 5. Compare and swap (:128-134)
    p_gt_a = (p_exp8 > a_exp8) | ((p_exp8 == a_exp8) & (p_sig11 >= a_sig11))
    e_max = torch.where(p_gt_a, p_exp8, a_exp8)
    e_min = torch.where(p_gt_a, a_exp8, p_exp8)
    s_max = torch.where(p_gt_a, p_sign, a_sign)
    s_min = torch.where(p_gt_a, a_sign, p_sign)
    sig_max = torch.where(p_gt_a, p_sig11, a_sig11)
    sig_min = torch.where(p_gt_a, a_sig11, p_sig11)
    shift = torch.clamp(e_max - e_min, max=11)                                     # :136-137
    # 6. Shift-right-jam (:140-143)
    shifted = sig_min >> shift
    sticky = (sig_min != (shifted << shift)).to(p.dtype)
    aligned = (shifted & 0x7FF) | sticky
    # 7. 12-bit add or subtract (:146-151)
    sig_sum = torch.where(s_max == s_min, sig_max + aligned, sig_max - aligned) & 0xFFF
    cancel = sig_sum == 0
    # 8. Carry normalisation (:154-157)
    carry = (sig_sum >> 11) & 1
    san = torch.where(carry == 1, sig_sum >> 1, sig_sum & 0x7FF)
    sticky2 = torch.where(carry == 1, sticky | (sig_sum & 1), sticky)
    e_an = e_max + carry
    # 9. Leading-zero normalisation (:161-182)
    lzc = 11 - _bit_length(san)
    need_ls = (carry == 0) & (((san >> 10) & 1) == 0)
    lsa = torch.where(need_ls, torch.minimum(lzc, e_an), 0)
    sig_norm = torch.where(need_ls, (san << lsa) & 0x7FF, san)
    e_norm = e_an - lsa
    # 10. Round to nearest even (:185-194)
    g = (sig_norm >> 2) & 1
    r = (sig_norm >> 1) & 1
    s = (sig_norm & 1) | sticky2
    m7 = (sig_norm >> 3) & 0x7F
    m_round = m7 + (g & (r | s | (m7 & 1)))
    r_ovf = (m_round >> 7) & 1
    m_out = torch.where(r_ovf == 1, 0, m_round & 0x7F)
    e_round = e_norm + r_ovf
    # 11. Overflow / underflow clamp (:197-214)
    overflow = e_round > 254
    underflow = e_norm <= 0
    e_out = torch.where(overflow, 254, torch.where(underflow, 0, e_round & 0xFF))
    m_out = torch.where(overflow, 0x7F, torch.where(underflow, 0, m_out))
    # 12. Output mux (:217-219), then the zero fast paths (:118-121, checked first in C)
    res = torch.where(cancel, 0, (s_max << 15) | (e_out << 7) | m_out)
    res = torch.where(a_zero, (p_sign << 15) | (p_exp8 << 7) | p_man7, res)
    return torch.where(p_zero, a, res)


def _sa_e4m3_to_bf16(byte: torch.Tensor) -> torch.Tensor:
    """sa_e4m3_to_bf16 (systolic_array_model.h:69-81): bias byte -> BF16; exponent field 0
    (zero and subnormals) -> signed zero; exponent field 15 is finite."""
    sign = (byte >> 7) & 1
    exp4 = (byte >> 3) & 0xF
    mant3 = byte & 0x7
    val = (sign << 15) | ((exp4 + 120) << 7) | ((mant3 << 4) & 0x7F)
    return torch.where(exp4 == 0, sign << 15, val)


def _sa_step(acc: torch.Tensor, xk: torch.Tensor, wk: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """One PE row of a compute (systolic_array_model.h:464-476) for every (batch, row, column):
    product of activation k and weight k, then E4M3ProdAddBF16 into the column's addend.
    xk [B, M], wk [B, N] int32 bytes; vectorised over [B, M, N] (NEW)."""
    idx = (xk[:, :, None] << 8) | wk[:, None, :]                  # (act << 8) | weight (converters.h:111)
    return sa_prod_add_bf16(lut[idx], acc)                        # sa_pe_compute (:229-233)


def _sa_step_flat(acc: torch.Tensor, ops: torch.Tensor, xi: torch.Tensor, wi: torch.Tensor,
                  dx: torch.Tensor, dw: torch.Tensor, lut: torch.Tensor):
    """_sa_step on a shape-stable signature, for torch.compile (NEW).

    acc     int32 [L], the [B, M, Nc] addends flattened
    ops     int32 [S], xT [B, K, M] flattened followed by wT [B, K, Nc] flattened
    xi, wi  int32 [L] (int64 if S >= 2^31), where element (b, m, n) reads its activation and
            weight in ops at step k
    dx, dw  0-dim, the advance of those positions per step (M and Nc)
    Returns (new acc, xi + dx, wi + dw). Every tensor is fresh (not a view), 1-D or 0-dim, and
    every per-shape or per-step integer is tensor data, so the only symbolic sizes are L and S:
    no rank, stride, storage-offset, dtype or Python-int guard changes with the layer shape or
    k, and one graph serves every call (a second only for L == 1). Values are _sa_step's."""
    p = lut[(ops[xi] << 8) | ops[wi]]
    return sa_prod_add_bf16(p, acc), xi + dx, wi + dw


_sa_step_compiled = None


def _get_sa_step_compiled():
    """NEW: torch.compile of one flat step, so the ~60 element-wise ops of the adder fuse into
    one kernel. The K loop stays in Python: rows are applied strictly in index order."""
    global _sa_step_compiled
    if _sa_step_compiled is None:
        _sa_step_compiled = torch.compile(_sa_step_flat, dynamic=True)
    return _sa_step_compiled


def _compile_default() -> bool:
    return os.environ.get("ATLAS_ACCUM_COMPILE", "0") not in ("", "0")


def sa_linear_bits(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None, *,
                   rows: int = 32, scale_exp: int = 0, out_fmt_sel=OUT_BF16,
                   compile: Optional[bool] = None,
                   chunk_elems: int = _DEFAULT_CHUNK_ELEMS) -> torch.Tensor:
    """Raw output words (int32 holding uint16) of sa_linear_call (sa_linear.h:67-225).

    Tiling, as in sa_linear.h:
      - K is split into tiles of `rows` (:79-81, 122-128); the tail tile is zero-padded
        (:141-143, 185-187). A zero activation gives a zero product, which passes the addend
        through (systolic_array_model.h:118-119), so padded rows are exact no-ops and the
        torch loop runs over the real K only.
      - tile 0 starts from the E4M3 bias widened to BF16 when a bias is given (SA_UseBias,
        :161-167; systolic_array_model.h:448-450), else from +0 (SA_UseZero, :168-172).
      - each tile's last-row value goes through OutputConvStage (systolic_array_model.h:479-483)
        and is fed back as the next tile's psum (SA_UsePsum, :173-177, 204-205). With OutE4M3
        that feedback is the E4M3 byte read as BF16 bits, exactly as the C model does.
      - output columns are independent; `cols` grouping (:99-103) cannot change values.
      - in_features == 0 runs no tile and returns zeros (:81, 105-106, 211-217).
    """
    compile = _compile_default() if compile is None else compile
    xb, wb, bb, out_shape = _prepare(x, w, b)
    B, M, K = xb.shape
    N = wb.shape[1]
    device = xb.device
    sel = _fmt_value(out_fmt_sel)
    out = torch.zeros(B, M, N, dtype=torch.int32, device=device)
    if K == 0 or M == 0 or N == 0:
        return out.reshape(out_shape)
    lut = e4m3_mul_to_prod_lut(device)
    xT = xb.transpose(1, 2).contiguous()                          # [B, K, M]
    for n0, n1 in _n_chunks(N, B * M * 4, chunk_elems):         # NEW: bounds temporaries
        Nc = n1 - n0
        wT = wb[:, n0:n1].transpose(1, 2).contiguous()           # [B, K, Nc]
        if bb is not None:
            acc = _sa_e4m3_to_bf16(bb[:, None, n0:n1]).expand(B, M, Nc).contiguous()
        else:
            acc = torch.zeros(B, M, Nc, dtype=torch.int32, device=device)
        if compile:
            # NEW: flat, shape-stable operands (see _sa_step_flat); acc is kept int32 and 1-D
            # so its dtype and rank never change between tiles (output_conv_stage widens).
            step = _get_sa_step_compiled()
            ops = torch.cat([xT.reshape(-1), wT.reshape(-1)])
            torch._dynamo.mark_static(lut, 0)
            idt = torch.int32 if ops.numel() < 2 ** 31 else torch.int64
            i = torch.arange(B * M * Nc, dtype=idt, device=device)
            n, m, bi = i % Nc, (i // Nc) % M, i // (Nc * M)
            xi = bi * (K * M) + m                                 # xT[b, 0, m]
            wi = xT.numel() + bi * (K * Nc) + n                   # wT[b, 0, n]
            dx = torch.tensor(M, dtype=idt, device=device)        # 0-dim: no size symbol
            dw = torch.tensor(Nc, dtype=idt, device=device)
            del i, n, m, bi
            acc = acc.reshape(-1).clone()                         # fresh, not a view
            for k0 in range(0, K, rows):
                for _ in range(k0, min(K, k0 + rows)):            # rows in index order (:464)
                    acc, xi, wi = step(acc, ops, xi, wi, dx, dw, lut)
                acc = output_conv_stage(acc, sel, scale_exp).to(torch.int32)
            acc = acc.reshape(B, M, Nc)
        else:
            for k0 in range(0, K, rows):
                k1 = min(K, k0 + rows)
                for k in range(k0, k1):                           # rows in index order (:464)
                    acc = _sa_step(acc, xT[:, k], wT[:, k], lut)
                acc = output_conv_stage(acc, sel, scale_exp)
        out[:, :, n0:n1] = acc
    return out.reshape(out_shape)


def sa_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None, *,
              rows: int = 32, scale_exp: int = 0, out_fmt_sel=OUT_BF16,
              compile: Optional[bool] = None) -> torch.Tensor:
    """SARTLLinearFunction(rows, cols).__call__ (systolic_array_rtl_linear.py:257-291) in torch."""
    bits = sa_linear_bits(x, w, b, rows=rows, scale_exp=scale_exp, out_fmt_sel=out_fmt_sel,
                          compile=compile)
    return decode_output_bits(bits, out_fmt_sel)


# ---------------------------------------------------------------------------------------------
# IPT (mxu1): inner-product trees
# ---------------------------------------------------------------------------------------------

def _ipt_consts(vec_len: int, variant: str):
    if variant == "numba":
        # _numba_kernels.py:252-254 and :26 hard-code anchorHeadroom 7 and intWidth 32
        # (InnerProductTreeParams.intWidth = 8 + headroom + 17, ipt_numba/params_and_requests.py:81-86),
        # whatever vec_len the wrapper is given.
        return 7, 32
    if variant == "c":
        headroom = (vec_len + 1).bit_length() + 1                 # params_and_requests.h:84-86
        return headroom, 8 + headroom + 15                        # :88-93
    raise ValueError(f"unknown ipt variant {variant!r}; choose 'numba' or 'c'")


def _wrap(v: torch.Tensor, width: int) -> torch.Tensor:
    """wrap_signed (fp_formats.h:154-163; _numba_kernels.py:29-35) on int64."""
    m = (1 << width) - 1
    v = v & m
    return torch.where(v >= (1 << (width - 1)), v - (1 << width), v)


def _ieee_to_aligned(bits: torch.Tensor, anchor: torch.Tensor, mant_bits: int, exp_width: int,
                     ieee_bias: int, sign_bit: int, int_width: int) -> torch.Tensor:
    """ieee_to_aligned_int (converters.h:131-183; _numba_kernels.py:54-105), int64.

    Note the reference quirk kept here: a subnormal input uses unbiased exponent
    0 - bias, not 1 - bias (converters.h:147-148; _numba_kernels.py:66-67, 93-94).
    """
    sign = (bits >> sign_bit) & 1
    exp_field = (bits >> mant_bits) & ((1 << exp_width) - 1)
    frac = bits & ((1 << mant_bits) - 1)
    zero = (exp_field == 0) & (frac == 0)                                            # :144-145
    unb = exp_field - ieee_bias                                                      # :147
    full_sig = torch.where(exp_field != 0, (1 << mant_bits) | frac, frac)           # :148
    sr = anchor - unb - (int_width - 1 - mant_bits)                                  # :150
    right = full_sig >> sr.clamp(0, 62)
    left = (full_sig << (-sr).clamp(0, 62)) & 0xFFFFFFFF      # uint32 shift in C (:165)
    shifted = torch.where(sr >= int_width, 0,
                          torch.where(sr >= 0, right, torch.where(sr > -int_width, left, 0)))  # :155-170
    mag = shifted & ((1 << int_width) - 1)                                           # :173-175
    val = _wrap(torch.where(sign == 1, -mag, mag), int_width)                        # :180-182
    return torch.where(zero, 0, val)


def _aligned_to_bf16_once(total: torch.Tensor, anchor: torch.Tensor, int_width: int) -> torch.Tensor:
    """ipt_numba _aligned_to_bf16 (_numba_kernels.py:156-167) + _f64_to_bf16_rne (:108-153):
    total * 2^(anchor - (int_width - 1)) rounded once, RNE, to BF16; biased exponent >= 255
    -> +-0x7F7F (:148-149); biased exponent <= 0 -> +0 (:150-151); total 0 -> +0 (:164-165).
    The value of a 32-bit integer is exact in float64, so the integer RNE here is identical."""
    sign = (total < 0).to(torch.int64)
    mag = total.abs()
    bl = _bit_length(mag)
    sh = (bl - 8).clamp(min=0)
    top = mag >> sh
    rem = mag - (top << sh)
    half = torch.where(sh > 0, torch.ones_like(sh) << (sh - 1).clamp(min=0), 0)
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((top & 1) == 1)))
    top = top + up.to(torch.int64)
    top = torch.where(sh == 0, mag << (8 - bl).clamp(min=0), top)
    rcarry = top >= 256                                                              # :143-145
    top = torch.where(rcarry, top >> 1, top)
    unb = bl - 1 + anchor - (int_width - 1) + rcarry.to(torch.int64)
    biased = unb + 127
    res = (sign << 15) | (biased << 7) | (top & 0x7F)
    res = torch.where(biased >= 0xFF, torch.where(sign == 1, 0xFF7F, 0x7F7F), res)
    res = torch.where(biased <= 0, 0, res)
    return torch.where(mag == 0, 0, res)


def _aligned_to_bf16_c(total: torch.Tensor, anchor: torch.Tensor, int_width: int) -> torch.Tensor:
    """ipt_c aligned_int_to_bf16 (converters.h:244-254): f = ldexpf((float)int, anchor -
    (int_width - 1)), then f32_to_bf16_bits_rne (fp_formats.h:279-293). Two roundings: int ->
    float32 (24-bit RNE), then float32 -> BF16 RNE; ldexpf itself rounds (RNE) only when the
    result is float32-subnormal and gives +-Inf above FLT_MAX. Done in integers here."""
    sign = (total < 0).to(torch.int64)
    mag = total.abs()
    bl = _bit_length(mag)
    # (float)int_in: RNE to 24 significant bits
    sh = (bl - 24).clamp(min=0)
    q = mag >> sh
    rem = mag - (q << sh)
    half = torch.where(sh > 0, torch.ones_like(sh) << (sh - 1).clamp(min=0), 0)
    q = q + ((sh > 0) & ((rem > half) | ((rem == half) & ((q & 1) == 1)))).to(torch.int64)
    q = torch.where(sh == 0, mag << (24 - bl).clamp(min=0), q)
    c = q >= (1 << 24)
    q = torch.where(c, q >> 1, q)
    E = bl - 1 + c.to(torch.int64) + anchor - (int_width - 1)       # unbiased float32 exponent
    # ldexpf: normal, subnormal (RNE at 2^-149), or overflow to Inf
    u_norm = (sign << 31) | ((E + 127) << 23) | (q - (1 << 23))
    d = (-126 - E).clamp(min=1, max=40)
    qs = q >> d
    rs = q - (qs << d)
    hs = torch.ones_like(d) << (d - 1)
    qs = qs + ((rs > hs) | ((rs == hs) & ((qs & 1) == 1))).to(torch.int64)
    u_sub = (sign << 31) | qs                                          # qs == 2^23 is min normal
    u_inf = (sign << 31) | 0x7F800000
    u = torch.where(E > 127, u_inf, torch.where(E >= -126, u_norm, u_sub))
    # f32_to_bf16_bits_rne (fp_formats.h:279-293); Inf maps to its upper half (:284-287)
    rounded = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    res = torch.where(E > 127, u >> 16, rounded) & 0xFFFF
    return torch.where(mag == 0, 0, res)                               # converters.h:249-250


def _ipt_tile(xt: torch.Tensor, wt: torch.Tensor, lut: torch.Tensor, addend: torch.Tensor,
              addend_kind: int, headroom: int, int_width: int, variant: str,
              sel: int, scale_exp) -> torch.Tensor:
    """One compute_now for every (batch, row, lane): aat_compute_lane
    (inner_product_trees_model.h:144-213; _numba_kernels.py:258-315), vectorised (NEW).

    xt [B, M, V], wt [B, Nc, V] int32 bytes (zero-padded to vec_len, ipt_linear.h:128-131,
    173-176); addend [B, M, Nc] (bias byte or psum word); addend_kind 0=UseAct (none),
    1=UseBias, 2=UsePsum (fp_formats.h:55-60).
    """
    prod = lut[(xt[:, :, None, :] << 8) | wt[:, None, :, :]].to(torch.int64)       # [B, M, Nc, V]
    exp5 = (prod >> 7) & 0x1F
    nz = exp5 != 0
    unb = exp5 - _PROD_BIAS
    # Stage 0: max product exponent, sentinel for zero products (:166-172, :137-141)
    max_pe = torch.where(nz, unb, _SENTINEL).amax(dim=-1)
    # Addend exponent (:174-187; _numba_kernels.py:279-291)
    if addend_kind == 1:
        bef = (addend >> 3) & 0xF
        bz = (addend >> 3) & 0x1F          # includes the sign bit: -0 bias counts (:179-181)
        addend_exp = torch.where(bz != 0, bef - 7, _SENTINEL)
    elif addend_kind == 2:
        pef = (addend >> 7) & 0xFF
        pfrc = addend & 0x7F
        addend_exp = torch.where((pef == 0) & (pfrc == 0), _SENTINEL, pef - 127)
    else:
        addend_exp = torch.full_like(max_pe, _SENTINEL)
    anchor = torch.maximum(max_pe, addend_exp) + headroom                           # :190-191
    # e4m3_prod_to_aligned_int (converters.h:199-236; _numba_kernels.py:38-51). For a
    # non-zero product rshift >= headroom > 0. The clamp at 63 only matters where the C/numba
    # shift would be >= 64 (undefined behaviour there); the shifted value is 0 either way
    # mathematically, which is what the clamp gives.
    rshift = (anchor[..., None] - unb).clamp(0, 63)
    sig_wide = (0x80 | (prod & 0x7F)) << (int_width - 8)
    mag = (sig_wide >> rshift) & ((1 << int_width) - 1)
    vals = torch.where(nz, torch.where(((prod >> 12) & 1) == 1, -mag, mag), 0)
    # Accumulate (:194-198): wrapping each add (C) or once at the end (numba, :298-310) is the
    # same modular sum, so one int64 sum then a wrap is exact for any order.
    prod_sum = vals.sum(dim=-1)
    if addend_kind == 1:
        addend_int = _ieee_to_aligned(addend, anchor, 3, 4, 7, 7, int_width)          # E4M3 bias
    elif addend_kind == 2:
        addend_int = _ieee_to_aligned(addend & 0xFFFF, anchor, 7, 8, 127, 15, int_width)  # BF16 psum
    else:
        addend_int = torch.zeros_like(prod_sum)
    total = _wrap(prod_sum + addend_int, int_width)                                 # :210
    if variant == "numba":
        bf16 = _aligned_to_bf16_once(total, anchor, int_width)
    else:
        bf16 = _aligned_to_bf16_c(total, anchor, int_width)
    return output_conv_stage(bf16, sel, scale_exp)                                  # :211-212


def ipt_linear_bits(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None, *,
                    vec_len: int = 32, variant: str = "numba", scale_exp: int = 0,
                    out_fmt_sel=OUT_BF16,
                    chunk_elems: int = _DEFAULT_CHUNK_ELEMS) -> torch.Tensor:
    """Raw output words (int32 holding uint16) of the IPT linear call.

    variant "numba": IPTLinearRTLFunction (ipt_numba/ipt_rtl_linear.py:208-299) with
    compute_lanes_batch (_numba_kernels.py:234-317). variant "c": ipt_linear_call
    (ipt_c/ipt_linear.h:57-223). Same tiling in both:
      - K tiles of vec_len, tail zero-padded (ipt_linear.h:107-131; ipt_rtl_linear.py:232-237);
        padded zero products have exponent field 0 and contribute nothing.
      - tile 0 uses the E4M3 bias (UseBias) if given, else no addend (UseAct); later tiles use
        the previous tile's output word as BF16 psum (UsePsum) (ipt_linear.h:147-165, 199-200;
        ipt_rtl_linear.py:266-292). With OutE4M3 that feedback is the E4M3 byte read as BF16.
      - lanes are independent, so num_lanes and output tiling cannot change values.
    """
    xb, wb, bb, out_shape = _prepare(x, w, b)
    B, M, K = xb.shape
    N = wb.shape[1]
    device = xb.device
    sel = _fmt_value(out_fmt_sel)
    headroom, int_width = _ipt_consts(vec_len, variant)
    out = torch.zeros(B, M, N, dtype=torch.int32, device=device)
    if K == 0 or M == 0 or N == 0:
        return out.reshape(out_shape)
    lut = e4m3_mul_to_prod_lut(device)
    n_tiles = (K + vec_len - 1) // vec_len
    pad = n_tiles * vec_len - K
    if pad:
        xb = torch.nn.functional.pad(xb, (0, pad))
        wb = torch.nn.functional.pad(wb, (0, pad))
    for n0, n1 in _n_chunks(N, B * M * vec_len * 3, chunk_elems):    # NEW: bounds memory
        acc = None
        for t in range(n_tiles):
            xt = xb[:, :, t * vec_len:(t + 1) * vec_len]
            wt = wb[:, n0:n1, t * vec_len:(t + 1) * vec_len]
            if t == 0 and bb is not None:
                addend, kind = bb[:, None, n0:n1].to(torch.int64).expand(B, M, n1 - n0), 1
            elif t == 0:
                addend, kind = torch.zeros(B, M, n1 - n0, dtype=torch.int64, device=device), 0
            else:
                addend, kind = acc, 2
            acc = _ipt_tile(xt, wt, lut, addend, kind, headroom, int_width, variant,
                            sel, scale_exp).to(torch.int64)
        out[:, :, n0:n1] = acc.to(torch.int32)
    return out.reshape(out_shape)


def ipt_linear(x: torch.Tensor, w: torch.Tensor, b: Optional[torch.Tensor] = None, *,
               vec_len: int = 32, variant: str = "numba", scale_exp: int = 0,
               out_fmt_sel=OUT_BF16) -> torch.Tensor:
    """IPTLinearRTLFunction(vec_len).__call__ (ipt_numba/ipt_rtl_linear.py:208-299), or with
    variant="c" CIPTLinearRTLFunction (ipt_c/ipt_rtl_linear_c.py:621-639), in torch."""
    bits = ipt_linear_bits(x, w, b, vec_len=vec_len, variant=variant, scale_exp=scale_exp,
                           out_fmt_sel=out_fmt_sel)
    return decode_output_bits(bits, out_fmt_sel)
