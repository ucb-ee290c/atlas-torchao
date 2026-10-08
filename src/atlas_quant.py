import torch


def hw_bf16_to_e4m3(x: torch.Tensor, scale_e8m0: torch.Tensor) -> torch.Tensor:
    """Bit-exact BF16ScaleToE4M3 (MXU pop path = npu_model RtlNumerics.acc_to_fp8).

    Computes x * 2^(scale_e8m0 - 127) and rounds to E4M3, returning raw bytes (uint8).
    To DIVIDE by 2^s (normal quantization), pass scale_e8m0 = 127 - s.
    scale_e8m0 must broadcast against x.

    Behavior: round-to-nearest-even; |value| above the E4M3 range saturates to 448
    (0x7E/0xFE); a value rounding to 480 encodes 0x7F/0xFF (matches the RTL);
    results below 2^-6 flush to 0; zero/subnormal/NaN inputs -> 0; +/-Inf saturates.

    Not vpack: the VPU's vpack.bf16.fp8 DIVIDES by the scale and saturates to 448.
    """
    b = x.to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign, e, f = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F

    mant8 = 0x80 | f                                                      # hidden bit + 7 fraction bits
    trunc = mant8 >> 4                                                    # keep hidden bit + 3 bits
    inc = ((mant8 >> 3) & 1) & (((mant8 & 7) != 0) | (trunc & 1)).int()  # round to nearest even
    rounded = trunc + inc
    carry = rounded == 16                                                 # mantissa overflowed

    final_exp = (e - 127) + (scale_e8m0.to(torch.int32) - 127) + carry.int()
    mant = torch.where(carry, 0, (rounded - 8) & 7)

    normal = (sign << 7) | (((final_exp + 7) & 0xF) << 3) | mant
    sat = (sign << 7) | 0x7E
    out = torch.where(final_exp > 8, sat, torch.where(final_exp >= -6, normal, 0))
    out = torch.where((e == 0xFF) & (f == 0), sat, out)                  # +/-Inf saturates
    out = torch.where((e == 0) | ((e == 0xFF) & (f != 0)), 0, out)       # zero/subnormal/NaN -> 0
    return out.to(torch.uint8)


def hw_e4m3_to_float(q: torch.Tensor) -> torch.Tensor:
    """Bit-exact e4m3ToBf16 / RtlNumerics.from_fp8 at unit scale.

    Accepts raw bytes (uint8) or a torch.float8_e4m3fn tensor (its bits are reinterpreted,
    not value-converted). Subnormals and 0x7F/0xFF flush to signed zero. Returns float32.
    """
    if q.dtype == torch.float8_e4m3fn:
        q = q.view(torch.uint8)                      # reinterpret bits, don't convert values
    q = q.to(torch.int32)
    exp, mant = (q >> 3) & 0xF, q & 7
    val = (1 + mant / 8.0) * torch.exp2((exp - 7).float())
    val = torch.where((q >> 7) & 1 == 1, -val, val)
    flush = (exp == 0) | ((exp == 0xF) & (mant == 7))
    return torch.where(flush, torch.copysign(torch.zeros_like(val), val), val)


def scale_exp_for(t: torch.Tensor) -> torch.Tensor:
    """Pick a power-of-two scale exponent s so max|t| / 2^s <= 448 (the E4M3 max).

    Real value = fp8 value * 2^s. Uses ceil so the scaled max never exceeds 448
    (which also avoids the 480 -> 0x7F case). Clamped so the E8M0 byte stays in 0..254.
    """
    amax = t.detach().abs().max().float().clamp_min(2.0**-126)
    return torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127).to(torch.int32)