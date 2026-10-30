"""ChipVPU against the reference VectorRTLFunctions (the oracle), bit for bit.

Unary ops: every BF16 pattern. Binary ops: a structured set (zeros, subnormals, infinities, extreme
exponents, ties) crossed with itself plus random pairs. Row reductions: random rows with
cancellations, ties and mixed magnitudes. NaN inputs are checked separately: the reference
adapter's own NaN conversion (vector_rtl_forward.py:66-73) is reproduced, and results that are
NaN are compared as "both NaN".
"""
import pytest
import torch

import chip_ops as co
import ref_loader
from refutil import need_refs


@pytest.fixture(scope="module")
def vrf():
    need_refs()
    m = ref_loader.import_ref("funct_models_vector.vector_rtl_forward")
    return m.VectorRTLFunctions(num_lanes=16)


@pytest.fixture(scope="module")
def vpu():
    return co.ChipVPU()


def all_bf16():
    return co.bf16_bits_to_f32(torch.arange(65536, dtype=torch.int32))


def bits(x):
    return co.f32_to_bf16_bits(x)


def same(a, b):
    """Bitwise BF16 equality, NaN == NaN."""
    ba, bb = bits(a), bits(b)
    nan_a = torch.isnan(a.float())
    nan_b = torch.isnan(b.float())
    return bool(((ba == bb) | (nan_a & nan_b)).all())


def special_values():
    v = [0.0, -0.0, 1.0, -1.0, 2.0, 0.5, 3.0, 1.5, 1.0078125, -1.0078125, 255.0, 256.0, 448.0,
         1e-38, -1e-38, 1e-40, -1e-40, 9.2e-41, 3.3895e38, -3.3895e38, float("inf"), float("-inf"),
         65504.0, 1e-3, 7.0, 0.1, -0.1, 1e30, -1e30, 2.0**-126, 2.0**-133, 2.0**127]
    return torch.tensor(v, dtype=torch.float32).bfloat16().float()


@pytest.mark.parametrize("op", ["rcp", "sqrt", "sin", "cos", "log2", "tanh", "exp", "exp2",
                                "square", "cube", "relu"])
def test_unary_exhaustive(vrf, vpu, op):
    x = all_bf16()
    want = getattr(vrf, op)(x)
    got = getattr(vpu, op)(x)
    assert same(got, want), op


def test_unary_float32_inputs_round_like_reference(vrf, vpu):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(4096, generator=g) * 10                 # genuine float32 values (RNE to BF16)
    for op in ("exp", "rcp", "tanh"):
        assert same(getattr(vpu, op)(x), getattr(vrf, op)(x))
        assert getattr(vpu, op)(x).dtype == torch.float32


def _pairs(n_rand=20000, seed=0):
    s = special_values()
    a = s.repeat_interleave(len(s))
    b = s.repeat(len(s))
    g = torch.Generator().manual_seed(seed)
    ra = (torch.randn(n_rand, generator=g) * torch.exp2(torch.randint(-30, 30, (n_rand,), generator=g).float()))
    rb = (torch.randn(n_rand, generator=g) * torch.exp2(torch.randint(-30, 30, (n_rand,), generator=g).float()))
    close = ra * (1 + torch.randn(n_rand, generator=g) * 2.0**-8)          # near-cancellation
    ra, rb, close = ra.bfloat16().float(), rb.bfloat16().float(), close.bfloat16().float()
    return torch.cat([a, ra, ra, -ra]), torch.cat([b, rb, close, close])


@pytest.mark.parametrize("op", ["add", "sub", "mul", "pairwise_max", "pairwise_min"])
def test_binary(vrf, vpu, op):
    a, b = _pairs()
    a, b = a.bfloat16(), b.bfloat16()
    assert same(getattr(vpu, op)(a, b), getattr(vrf, op)(a, b)), op


def test_binary_all_patterns_against_fixed_operands(vrf, vpu):
    """Every BF16 pattern on one side, a few operands on the other (both orders)."""
    x = all_bf16()
    for c in (1.0, -1.5, 2.0**-120, 3.0e38):
        y = torch.full_like(x, c)
        for op in ("add", "sub", "mul"):
            assert same(getattr(vpu, op)(x, y), getattr(vrf, op)(x, y)), (op, c)
            assert same(getattr(vpu, op)(y, x), getattr(vrf, op)(y, x)), (op, c)


@pytest.mark.parametrize("op", ["rsum", "rmax", "rmin"])
def test_row_reduce(vrf, vpu, op):
    g = torch.Generator().manual_seed(1)
    rows = 300
    x = torch.randn(rows, 96, generator=g) * torch.exp2(torch.randint(-20, 20, (rows, 96), generator=g).float())
    x[:50] = x[:50].abs()                                     # same-sign rows
    x[50:100, 1::2] = -x[50:100, 0::2]                        # exact cancellations
    x[100:120] = 0.0
    x[120:130, :5] = -0.0
    x[130:140] = 1.0 + torch.arange(96).float() * 2.0**-7     # rounding ties after widening
    x = x.bfloat16()
    want = getattr(vrf, op)(x)
    got = getattr(vpu, op)(x)
    assert got.shape == want.shape == (rows, 3)
    assert same(got, want)


def test_reference_decompositions_run_on_chipvpu(vrf, vpu):
    """The reference composite sequences (vpu_decomp.py) give identical bits on both units."""
    dec = ref_loader.load_file("smolVLA-quant", "smolvla_inout/vpu_decomp.py")
    g = torch.Generator().manual_seed(2)
    x = (torch.randn(3, 70, generator=g) * 3).bfloat16()
    w = torch.randn(70, generator=g).bfloat16()
    b = torch.randn(70, generator=g).bfloat16()
    assert same(dec._vpu_softmax(vpu, x, -1), dec._vpu_softmax(vrf, x, -1))
    assert same(dec._vpu_layer_norm(vpu, x, w, b, 1e-5), dec._vpu_layer_norm(vrf, x, w, b, 1e-5))
    assert same(dec._vpu_gelu_tanh(vpu, x), dec._vpu_gelu_tanh(vrf, x))
    assert same(dec._vpu_silu(vpu, x), dec._vpu_silu(vrf, x))
    assert same(dec._vpu_rsqrt(vpu, x.abs()), dec._vpu_rsqrt(vrf, x.abs()))


def test_counter_and_device_dtype(vpu):
    seen = []
    v = co.ChipVPU(counter=lambda i, n: seen.append((i, n)))
    x = torch.randn(4, 64, dtype=torch.float32)
    y = v.rsum(v.mul(x, x))
    assert y.dtype == torch.float32 and y.shape == (4, 2)
    assert seen == [("vmul.bf16", 256), ("vredsum.row.bf16", 256)]


def test_nan_inputs_follow_the_reference_conversion(vrf, vpu):
    """NaN patterns (exp 0xFF, mantissa != 0) through every unary op."""
    x = co.bf16_bits_to_f32(torch.tensor([0x7FC0, 0xFFC0, 0x7F81, 0xFF81], dtype=torch.int32))
    for op in ("rcp", "exp", "tanh", "sqrt"):
        assert same(getattr(vpu, op)(x), getattr(vrf, op)(x))


def test_vadd_vsub_emit_the_canonical_nan(vpu):
    """The RTL returns +qNaN 0x7FC0 for any NaN out of vadd/vsub (hardfloat, then the upper-16-bit
    slice), whatever the input NaN's sign or payload (found by the Verilator run)."""
    nan_in = co.bf16_bits_to_f32(torch.tensor([0xFFE1, 0x7FC1, 0x7F81], dtype=torch.int32))
    one = torch.ones(3)
    inf = torch.full((3,), float("inf"))
    for out in (vpu.add(nan_in, one), vpu.sub(nan_in, one), vpu.add(inf, -inf), vpu.sub(inf, inf)):
        assert (co.f32_to_bf16_bits(out)[:3] == 0x7FC0).all()
