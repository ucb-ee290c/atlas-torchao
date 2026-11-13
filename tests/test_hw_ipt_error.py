"""Synthetic checks for src/scripts/hw_ipt_error.py: float64 oracle vs the bit-exact IPT model."""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "scripts"))
import torch
import mxu.accum_torch as at
import reference.ref_numerics as rn
import hw_ipt_error as h


def _bytes(vals):
    return at.float_to_e4m3_bytes(torch.as_tensor(vals, dtype=torch.float32))


def test_bf16_rne_f64_matches_torch():
    v = torch.randn(4096, dtype=torch.float64) * torch.exp2(torch.randint(-20, 20, (4096,)).double())
    assert torch.equal(h.bf16_rne_f64(v), v.float().to(torch.bfloat16).double())
    tie = torch.tensor([1.0 + 2 ** -8, 1.0 + 3 * 2 ** -8, -(1.0 + 2 ** -8)], dtype=torch.float64)
    assert h.bf16_rne_f64(tie).tolist() == [1.0, 1.015625, -1.0]


def test_single_tile_exact_friendly():
    x = _bytes([[1.0, 2.0, 0.5, 4.0] * 8])           # one 32-wide tile, small integers/powers of two
    w = _bytes([[1.0, 1.0, 2.0, 0.5] * 8])
    p = h.products(x, w)
    a = p.sum(-1)
    y = h.decode(at.ipt_linear_bits(x, w))
    assert a.item() == 8 * (1 + 2 + 1 + 2)
    assert torch.equal(y, a)


def test_b_equals_c_same_exponent_products():
    # every product has the same exponent -> no bit falls below the 17 guard bits, psum exact
    torch.manual_seed(0)
    M, N, K = 4, 3, 96
    vals = torch.tensor([1.0, 1.125, 1.25, 1.5, 1.75])
    x = _bytes(vals[torch.randint(0, 5, (M, K))] * (torch.randint(0, 2, (M, K)) * 2 - 1))
    w = _bytes(vals[torch.randint(0, 5, (N, K))] * (torch.randint(0, 2, (N, K)) * 2 - 1))
    p, a, b, c, d, ps = h.run_chains(x, w, with_sa=False)
    assert torch.equal(b, c)
    assert torch.allclose(c, a, rtol=2 ** -6)         # three BF16 roundings, still close to exact


def test_b_equals_c_on_random_real_like_data():
    torch.manual_seed(1)
    x = rn.quantize(torch.randn(8, 256))[0]
    w = rn.quantize(torch.randn(5, 256), dims=(-1,))[0]
    p, a, b, c, d, ps = h.run_chains(x, w, with_sa=False)
    assert torch.equal(b, c)                          # truncation never flips a BF16 rounding here
    assert (b - a).norm() / a.norm() < 2 ** -6


def test_truncation_can_occur_with_wide_spread():
    # 1 large product and many tiny ones 2^-30 below: ipt drops them, the tile-exact model keeps them
    x = _bytes([[1.0] + [2.0 ** -6] * 31])
    w = _bytes([[256.0] + [2.0 ** -6] * 31])
    p = h.products(x, w)
    exact = p.sum(-1).item()
    ipt = h.decode(at.ipt_linear_bits(x, w)).item()
    assert exact == 256 + 31 * 2.0 ** -12
    assert ipt == 256.0                               # tiny terms truncated, then RNE to BF16
    b, _ = h.tile_exact_bf16(p)
    assert b.item() == 256.0                          # the BF16 rounding hides it as well


def test_tile_stats_spread_zero_for_equal_products():
    x = _bytes([[1.0] * 32]); w = _bytes([[2.0] * 32])
    p = h.products(x, w)
    _, ps = h.tile_exact_bf16(p)
    sp, tr, ul = h.tile_stats(p, ps)
    assert sp.item() == 0.0 and tr.item() == 0.0
