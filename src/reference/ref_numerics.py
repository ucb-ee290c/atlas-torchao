"""The reference author's operand cast and scale rule, vectorized (device-agnostic torch).

The reference quantizes each matmul operand per call with
    x_fp8, s = quantize_bf16_to_e4m3(x)            (pi0_inout/hw_float_ops.py:64-103)
    real value = e4m3(x_fp8) * 2^s
whose encoder loops over elements in Python (hw_float_ops.py:97-98), too slow for whole models.
These ports are bit-equal to it (tests/test_ref_numerics.py checks every BF16 input at 47 shifts
against funct_models_vector/fp8_e4m3.py and the scale rule against compute_po2_scale_exp); the
reference functions stay as the test oracles.
"""

from __future__ import annotations

import torch

E4M3_MAX_POS = 0x7E          # fp8_e4m3.py:15-16
E4M3_MAX_NEG = 0xFE
PO2_MAX_EXP = 8              # log2 of _E4M3_FP8_MAX_PO2 = 256 (hw_float_ops.py:31)


def bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """BF16 bit patterns of x (cast to BF16 first with torch's RNE, as x.bfloat16() in the
    reference: hw_float_ops.py:89) as int32 in [0, 65535]."""
    return x.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def e4m3_vpack(x: torch.Tensor, exp_shift) -> torch.Tensor:
    """Port of bf16_to_e4m3_byte(bf16, exp_shift) (fp8_e4m3.py:38-86), i.e. FP8Pack.scala:57-95.

    Divides by 2^exp_shift, rounds to nearest even, saturates to 0x7E/0xFE, clamps a result of
    0x7F/0xFF (480) to 0x7E/0xFE (fp8_e4m3.py:84-85; FP8Pack.scala:88-91), flushes results below
    2^-6 to zero, maps BF16 zero/subnormal/NaN to 0 and +-Inf to +-0x7E. `exp_shift` is an int or an
    integer tensor broadcastable against x. Returns uint8 bytes.
    """
    b = bf16_bits(x)
    sign = (b >> 15) & 1
    exp_bf = (b >> 7) & 0xFF
    mant_bf = b & 0x7F
    if not torch.is_tensor(exp_shift):
        exp_shift = torch.tensor(int(exp_shift), dtype=torch.int32, device=x.device)
    exp_adj = (exp_bf - 127) - exp_shift.to(device=x.device, dtype=torch.int32)

    mant8 = 0x80 | mant_bf                                  # _round_right_shift_4_rne (fp8_e4m3.py:19-31)
    trunc = mant8 >> 4
    guard = (mant8 >> 3) & 1
    sticky = (mant8 & 7) != 0
    inc = (guard == 1) & (sticky | ((trunc & 1) == 1))
    rounded = trunc + inc.to(torch.int32)
    carry = rounded == 16
    final_exp = exp_adj + carry.to(torch.int32)
    mant = torch.where(carry, torch.zeros_like(rounded), (rounded - 8) & 7)

    sat = torch.where(sign == 1, E4M3_MAX_NEG, E4M3_MAX_POS)
    packed = (sign << 7) | (((final_exp + 7) & 0xF) << 3) | mant
    packed = torch.where((packed & 0x7F) == 0x7F, sat, packed)           # reserved-NaN clamp
    out = torch.where(final_exp > 8, sat, torch.where(final_exp < -6, torch.zeros_like(packed), packed))
    out = torch.where((exp_bf == 0xFF) & (mant_bf == 0), sat, out)       # +-Inf -> max finite
    out = torch.where((exp_bf == 0) | ((exp_bf == 0xFF) & (mant_bf != 0)), torch.zeros_like(out), out)
    return out.to(torch.uint8)


def e4m3_unpack(q: torch.Tensor, exp_shift=0) -> torch.Tensor:
    """Port of e4m3_byte_to_bf16 (fp8_e4m3.py:89-121, FP8Unpack.scala:69-100), returned as float32.

    Exponent 15 is finite (only 0x7F/0xFF are reserved and decode to signed zero, as do
    subnormals); BF16 overflow saturates to the max finite BF16 and underflow gives signed zero.
    On the encoder's outputs (never 0x7F, never subnormal) this equals the reference's
    `bytes.view(torch.float8_e4m3fn).to(torch.float32)` (quant_linear.py:161-162).
    """
    if q.dtype == torch.float8_e4m3fn:
        q = q.view(torch.uint8)
    q = q.to(torch.int32) & 0xFF
    sign = (q >> 7) & 1
    exp = (q >> 3) & 0xF
    mant = q & 7
    if not torch.is_tensor(exp_shift):
        exp_shift = torch.tensor(int(exp_shift), dtype=torch.int32, device=q.device)
    e = (exp - 7) + exp_shift.to(device=q.device, dtype=torch.int32) + 127
    bits = (sign << 15) | ((e & 0xFF) << 7) | (mant << 4)
    bits = torch.where(e >= 255, (sign << 15) | 0x7F7F, bits)
    zero = (exp == 0) | ((exp == 0xF) & (mant == 7)) | (e <= 0)
    bits = torch.where(zero, sign << 15, bits)
    return (bits.contiguous() << 16).view(torch.float32)          # int32 shift wraps: the sign lands in bit 31


def po2_scale_exp(x: torch.Tensor, dims=None) -> torch.Tensor:
    """Port of compute_po2_scale_exp (hw_float_ops.py:38-57): s = floor(log2(amax / 256)).

    amax is taken in BF16 after nan_to_num(nan=0, posinf=0) of |x| (hw_float_ops.py:54), and s = 0
    when amax is 0. Computed as floor(log2(amax)) - 8 from the exponent of the (exact) float32
    amax, which is what log2 gives for a BF16 value (its mantissa is at most 1 + 127/128, so
    log2 never rounds across an integer). `dims=None` reduces over the whole tensor (the
    reference); a tuple of dims gives one scale per remaining index (kept as size-1 dims), used
    for the per-(batch, head) attention scales. Returns int32.
    """
    a = x.to(torch.bfloat16).abs().nan_to_num(nan=0.0, posinf=0.0)
    if dims is None:
        amax = a.max().float()
    else:
        amax = a.amax(dim=dims, keepdim=True).float()
    _, ex = torch.frexp(amax)                            # amax = m * 2^ex, m in [0.5, 1)
    s = (ex.to(torch.int32) - 1) - PO2_MAX_EXP
    return torch.where(amax == 0, torch.zeros_like(s), s)


def quantize(x: torch.Tensor, dims=None):
    """Port of quantize_bf16_to_e4m3 (hw_float_ops.py:64-103): (uint8 bytes, int32 scale exponent).

    x is cast to BF16 first, as the reference does (hw_float_ops.py:89), so an fp32 layer is
    quantized from its BF16 rounding. Real value = e4m3(bytes) * 2^s.
    """
    xb = x.to(torch.bfloat16)
    s = po2_scale_exp(xb, dims)
    return e4m3_vpack(xb, s), s


def dequantize(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """e4m3(q) * 2^s as float32 (exact: the product of an E4M3 value and a power of two)."""
    return e4m3_unpack(q) * torch.exp2(s.float())


def dq_matmul_nt(a_q: torch.Tensor, a_s: torch.Tensor, b_q: torch.Tensor, b_s: torch.Tensor) -> torch.Tensor:
    """a [..., M, K] times b [..., N, K]^T on E4M3 operands with power-of-two exponents: both are decoded
    (`dequantize`, float32, exact) and multiplied with torch.matmul in float32. The reference attention's
    framework-accumulation matmul (ref_attention._mm); the atlas.qk_matmul / atlas.pv_matmul operators
    call it so the graph and the eager reference share one body."""
    a = dequantize(a_q, a_s)
    b = dequantize(b_q, b_s)
    return torch.matmul(a, b.transpose(-1, -2))
