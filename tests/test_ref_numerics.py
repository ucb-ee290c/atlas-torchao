import math

import pytest
import torch

from refutil import need_refs

import reference.ref_loader as ref_loader
import reference.ref_numerics as rn
from collab.atlas_quant import hw_bf16_to_e4m3

SHIFTS = range(-23, 24)            # 47 shifts


@pytest.fixture(scope="module")
def fp8():
    need_refs()
    return ref_loader.load_file("pi0-quant", "funct_models_vector/fp8_e4m3.py")


@pytest.fixture(scope="module")
def hw():
    need_refs()
    return ref_loader.import_ref("pi0_inout.hw_float_ops")


def _all_bf16():
    return torch.arange(1 << 16, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)


def test_ref_encoder_exhaustive(fp8):
    """Every BF16 bit pattern at 47 shifts == bf16_to_e4m3_byte (fp8_e4m3.py:38-86)."""
    x = _all_bf16()
    bits = list(range(1 << 16))
    for s in SHIFTS:
        ref = torch.tensor([fp8.bf16_to_e4m3_byte(b, s) for b in bits], dtype=torch.uint8)
        assert torch.equal(rn.e4m3_vpack(x, s), ref), f"shift {s}"


def test_ref_encoder_tensor_shift_broadcasts(fp8):
    x = _all_bf16()[::97].reshape(-1, 1).expand(-1, 5)
    s = torch.tensor([[-3, 0, 2, 7, 12]], dtype=torch.int32)
    out = rn.e4m3_vpack(x, s)
    for j in range(5):
        assert torch.equal(out[:, j], rn.e4m3_vpack(x[:, j], int(s[0, j])))


def test_decode_256(fp8):
    """Exponent 15 is finite; 0x7F/0xFF and subnormals decode to signed zero (fp8_e4m3.py:89-121)."""
    q = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    for s in (-9, 0, 5):
        ref = torch.tensor([fp8.e4m3_byte_to_bf16(i, s) for i in range(256)], dtype=torch.int32)
        ref = (ref << 16).view(torch.float32)
        got = rn.e4m3_unpack(q, s)
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert rn.e4m3_unpack(torch.tensor([0x7E], dtype=torch.uint8)).item() == 448.0
    assert rn.e4m3_unpack(torch.tensor([0x78], dtype=torch.uint8)).item() == 256.0
    assert rn.e4m3_unpack(torch.tensor([0x7F], dtype=torch.uint8)).item() == 0.0
    # On every byte the encoder can produce, the reference's torch view decode agrees.
    enc = torch.tensor([i for i in range(256) if (i & 0x7F) != 0x7F and ((i >> 3) & 0xF) != 0 or i in (0,)],
                       dtype=torch.uint8)
    assert torch.equal(rn.e4m3_unpack(enc), enc.view(torch.float8_e4m3fn).float())


def test_scale_rule(hw):
    """po2_scale_exp == compute_po2_scale_exp (hw_float_ops.py:38-57) for every BF16 amax,
    zeros, NaN, +-Inf and mixtures."""
    x = _all_bf16()
    got = rn.po2_scale_exp(x.reshape(-1, 1), dims=(1,)).reshape(-1)
    ref = torch.tensor([hw.compute_po2_scale_exp(v.reshape(1)) for v in x], dtype=torch.int32)
    assert torch.equal(got, ref)
    cases = [torch.zeros(4), torch.tensor([float("nan"), 3.0]), torch.tensor([float("inf"), -5.0]),
             torch.tensor([float("-inf"), 0.25]), torch.tensor([255.9, -256.0]), torch.tensor([257.0]),
             torch.tensor([float("nan"), float("inf")]), torch.tensor([1e-40, 0.0])]
    for c in cases:
        assert int(rn.po2_scale_exp(c)) == hw.compute_po2_scale_exp(c), c


def test_quantize_tensor(hw):
    """Bytes and scale == quantize_bf16_to_e4m3 (hw_float_ops.py:64-103), including fp32 input."""
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.bfloat16, torch.float32):
        for scale in (1e-4, 0.03, 1.0, 300.0, 7e4):
            x = (torch.randn(13, 29, generator=g) * scale * torch.exp2(torch.randn(13, 29, generator=g) * 3)).to(dtype)
            x[0, 0] = 0.0
            q_ref, s_ref = hw.quantize_bf16_to_e4m3(x)
            q, s = rn.quantize(x)
            assert int(s) == s_ref and torch.equal(q, q_ref)


def test_per_head_scales_match_per_slice_reference(hw):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(2, 3, 5, 8, generator=g).bfloat16() * torch.tensor([1e-3, 1.0, 50.0]).view(1, 3, 1, 1).bfloat16()
    q, s = rn.quantize(x, dims=(-2, -1))
    assert s.shape == (2, 3, 1, 1)
    for b in range(2):
        for h in range(3):
            q_ref, s_ref = hw.quantize_bf16_to_e4m3(x[b, h])
            assert int(s[b, h]) == s_ref and torch.equal(q[b, h], q_ref)


def test_pop_codec_delta(fp8):
    """The collaborator's pop-path codec (atlas_quant.hw_bf16_to_e4m3) differs from vpack only where
    the pop path emits 0x7F/0xFF (a result rounding to 480), 30 inputs per shift, and is equal
    after clamping those to 0x7E/0xFE (plan section 2.3; OutputConvStage has no clamp)."""
    x = _all_bf16()
    for s in SHIFTS:
        pop = hw_bf16_to_e4m3(x, torch.tensor(127 - s))
        ref = rn.e4m3_vpack(x, s)
        diff = pop != ref
        assert int(diff.sum()) == 30, s
        assert set(pop[diff].tolist()) <= {0x7F, 0xFF} and set(ref[diff].tolist()) <= {0x7E, 0xFE}
        clamped = torch.where((pop & 0x7F) == 0x7F, pop - 1, pop)
        assert torch.equal(clamped, ref)


def test_scale_byte_is_clamped_like_the_rtl():
    """E8M0 byte 255 means +127 (VectorEngine.scala:247 clamps to [-128, 127]); shifts below -127 are not
    encodable in a byte. The pack and unpack clamp the shift the same way (found by the Verilator run)."""
    import reference.ref_numerics as rn
    x = torch.tensor([3.0, 100.0, 1.0e-3]).to(torch.bfloat16)
    assert torch.equal(rn.e4m3_vpack(x, 128), rn.e4m3_vpack(x, 127))
    assert torch.equal(rn.e4m3_vpack(x, -200), rn.e4m3_vpack(x, -127))
    q = torch.tensor([0x38, 0x77, 0xB8], dtype=torch.uint8)
    assert torch.equal(rn.e4m3_unpack(q, 128), rn.e4m3_unpack(q, 127))
    assert torch.equal(rn.e4m3_unpack(q, -200), rn.e4m3_unpack(q, -127))
