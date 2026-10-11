"""vpu-numerics: --chip-vadd {trunc,rn,rne_hw} and --chip-vadd-sites {residual,data,all}.

The default leaves the chip profile bit-identical (also checked across trees by test_algo_base.py);
the round-to-nearest sequence equals correct rounding off ties; the sites select the adds they name.
"""
import copy

import pytest
import torch

import algos
import chip_coverage as cc
import chip_numerics as cn
import chip_ops
from refutil import need_refs
from test_ladder import FAST, tiny_llama


def _bf16_pairs(n, seed, spread):
    """BF16 pairs (a, b) with |b|/|a| spanning 2^-spread .. 2^spread, both signs."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, generator=g) * torch.exp2(torch.randint(-20, 21, (n,), generator=g).float())
    k = torch.randint(-spread, spread + 1, (n,), generator=g).float()
    b = a * torch.exp2(k) * torch.randn(n, generator=g)
    return a.bfloat16().float(), b.bfloat16().float()


def _all_bf16(lo=-30, hi=30):
    bits = torch.arange(65536, dtype=torch.int32)
    x = chip_ops.bf16_bits_to_f32(bits)
    e = torch.floor(torch.log2(x.abs().clamp_min(1e-38)))
    keep = torch.isfinite(x) & (x != 0) & (e >= lo) & (e <= hi)
    return x[keep]


def _reference(a, b, sub):
    """Exact sum in float64 (exact for these operands), the correctly rounded BF16 (RNE), the tie
    mask, and the BF16 neighbour away from zero."""
    ex = a.double() - b.double() if sub else a.double() + b.double()
    m, sg = ex.abs(), ex.sign()
    e = torch.floor(torch.log2(m.clamp_min(2.0 ** -126)))
    u = torch.exp2(e - 7)                       # BF16 ulp in [2^e, 2^(e+1))
    q = torch.floor(m / u)
    lo, hi = q * u, (q + 1) * u
    rem = m - lo
    tie = rem == u / 2
    up = (rem > u / 2) | (tie & ((q % 2) == 1))
    return ex, sg * torch.where(up, hi, lo), tie, sg * hi


@pytest.mark.parametrize("sub", [False, True])
def test_rn_equals_correct_rounding_off_ties(sub):
    v = chip_ops.ChipVPU()
    # every BF16 value with exponent in [-30, 30] against random partners at many magnitude ratios,
    # plus a random population
    xa = _all_bf16()
    g = torch.Generator().manual_seed(3)
    parts = []
    for k in (-12, -6, -2, 0, 2, 6, 12):
        xb = (xa[torch.randperm(xa.numel(), generator=g)] * 2.0 ** k).bfloat16().float()
        parts.append((xa, xb))
    parts.append(_bf16_pairs(400_000, 7, 10))
    for a, b in parts:
        out = cn.vadd_rn(v, a, b, sub=sub).double()
        ex, rne, tie, away = _reference(a, b, sub)
        ok = ~tie & (ex != 0)
        bad = ok & (out != rne)
        assert int(bad.sum()) == 0, (a[bad][:4], b[bad][:4], out[bad][:4], rne[bad][:4])
        assert torch.equal(out[tie], away[tie])                   # ties go away from zero


def test_truncation_is_what_rn_fixes():
    v = chip_ops.ChipVPU()
    a, b = _bf16_pairs(200_000, 11, 4)
    ex, rne, tie, _ = _reference(a, b, False)
    t = v.add(a, b).double()
    assert (t.abs() <= (a + b).double().abs()).all()         # the FP32 sum, truncated toward zero
    assert float(((t != rne) & ~tie).double().mean()) > 0.2  # far from RNE


def test_rne_hw_is_rne_of_the_fp32_sum():
    v = chip_ops.ChipVPU()
    a, b = _bf16_pairs(200_000, 5, 8)
    for sub in (False, True):
        s32 = (a - b) if sub else (a + b)
        assert torch.equal(cn.vadd_rne_hw(v, a, b, sub), s32.bfloat16().float())


def test_instruction_counts():
    seen = {}
    v = chip_ops.ChipVPU(counter=lambda i, n: seen.__setitem__(i, seen.get(i, 0) + n))
    a, b = _bf16_pairs(64, 1, 3)
    cn.vadd_rn(v, a, b)
    assert sum(seen.values()) == cn.RN_OPS_PER_ADD * 64
    seen.clear()
    cn.vadd_rne_hw(v, a, b)
    assert seen == {"vadd.bf16": 64}


def _chip(model, vadd=None, sites="residual"):
    opts = dict(FAST, chip_mxu="framework")
    if vadd is not None:
        opts.update(chip_vadd=vadd, chip_vadd_sites=sites)
    return algos.quantize(model, "chip", ["lm_head"], opts, algo=None)


def _ids():
    return torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))


def test_default_and_explicit_trunc_are_the_profile_unchanged():
    need_refs()
    m0, m1 = tiny_llama(), tiny_llama()
    r0, r1 = _chip(m0), _chip(m1, "trunc")
    with torch.no_grad():
        y0, y1 = m0(_ids()).logits, m1(_ids()).logits
    assert torch.equal(y0.view(torch.int16), y1.view(torch.int16))
    assert "vadd" not in r0["profile"] and "vadd" not in r1["profile"]
    assert r0["ladder"] == r1["ladder"]


@pytest.mark.parametrize("sites,expect", [("residual", "residual"), ("data", "data"), ("all", "all")])
def test_site_scoping(sites, expect):
    need_refs()
    layers = 2
    m = tiny_llama(layers)
    rep = _chip(m, "rn", sites)
    with torch.no_grad():
        m(_ids())
    rec = rep["profile"]["vadd"]
    assert rec["mode"] == "rn" and rec["sites"] == sites and rec["instructions_per_add"] == 24
    n = rec["adds_routed"]
    if expect == "residual":
        assert n == 2 * layers                                # two residual adds per decoder layer
    elif expect == "data":
        # residual adds and the RoPE combines (q and k) per layer; not the mask add, not the norm eps
        assert n == 4 * layers
    else:
        assert n > 4 * layers                                 # decompositions' vadds too
    assert rec["extra_lane_ops"] > 0
    assert any("--chip-vadd rn" in o for o in rep["ladder"]["off_ladder_reasons"])
    assert not rep["ladder"]["on_ladder"]


def test_rn_changes_and_rne_hw_is_labelled():
    need_refs()
    base, rn, hw = tiny_llama(), tiny_llama(), tiny_llama()
    _chip(base)
    _chip(rn, "rn")
    rep = _chip(hw, "rne_hw")
    with torch.no_grad():
        yb, yr, yh = base(_ids()).logits, rn(_ids()).logits, hw(_ids()).logits
    assert not torch.equal(yb, yr) and not torch.equal(yb, yh)
    assert "pending hardware ruling" in rep["profile"]["vadd"]["status"]
    assert any("pending hardware ruling" in o for o in rep["ladder"]["off_ladder_reasons"])


def test_mask_and_constant_adds_are_not_routed():
    pol = cn.VaddPolicy("rn", "data")
    x = torch.randn(4, 8)
    assert pol.wants([], "add", x, torch.randn(4, 8))
    assert not pol.wants([], "add", x, torch.full((4, 8), -3.4e38))      # additive mask
    assert not pol.wants([], "add", x, torch.full((4, 8), 1e-6))         # a constant (eps)
    pol = cn.VaddPolicy("rn", "residual")
    assert not pol.wants([("vn_norm", "")], "add", x, torch.randn(4, 8))
    assert not pol.wants([], "add", x, torch.randn(1, 8))                # broadcast: not a stream add


def test_mtdit_emulator_accepts_the_options():
    need_refs()
    emu = cc.make_mtdit_op_emulator(vadd="rn", vadd_sites="data")
    assert emu.router.vadd is not None and emu.router.vadd.mode == "rn"
    assert cc.make_mtdit_op_emulator().router.vadd is None


def test_mtdit_emulator_env(monkeypatch):
    need_refs()
    monkeypatch.setenv("ATLAS_CHIP_VADD", "rne_hw")
    monkeypatch.setenv("ATLAS_CHIP_VADD_SITES", "all")
    r = cc.make_mtdit_op_emulator().router
    assert r.vadd.mode == "rne_hw" and isinstance(r.vpu, cn.NumericsVPU)
    # multitask-dit-quant's chip_full adapter builds the router directly
    r2 = cc.ChipRouter(cc.Coverage(), mxu="ipt", mxu_impl="ref_cpu")
    assert r2.vadd.mode == "rne_hw"


def test_chip_profile_ignores_the_environment(monkeypatch):
    need_refs()
    monkeypatch.setenv("ATLAS_CHIP_VADD", "rn")
    m0, m1 = tiny_llama(), tiny_llama()
    monkeypatch.delenv("ATLAS_CHIP_VADD")
    _chip(m0)
    monkeypatch.setenv("ATLAS_CHIP_VADD", "rn")
    rep = _chip(m1)
    with torch.no_grad():
        y0, y1 = m0(_ids()).logits, m1(_ids()).logits
    assert torch.equal(y0.view(torch.int16), y1.view(torch.int16)) and "vadd" not in rep["profile"]


def test_kl_meter_zero_against_itself_and_positive_otherwise():
    from eval import KLMeter

    m = tiny_llama()
    ids = _ids()
    k = KLMeter(m)
    with torch.no_grad():
        k(ids, m(ids, use_cache=False))
    s = k.summary()
    assert abs(s["kl_vs_bf16"]) < 1e-6 and s["top1_agree_vs_bf16"] == 1.0 and s["tokens"] == 16
    other = tiny_llama()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.05 * torch.randn_like(p))
        k2 = KLMeter(m)
        k2(ids, other(ids, use_cache=False))
    assert k2.summary()["kl_vs_bf16"] > 0
