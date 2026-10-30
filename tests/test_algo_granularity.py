"""granularity: per-channel / per-token / block32 power-of-two scales (src/algos/granularity.py)."""
import copy
import math

import pytest
import torch
import torch.nn as nn

import algos
import ref_numerics as rn
from algos.granularity import block_scale_exp, scale_exp
from refutil import need_refs
from test_ladder import FAST, ConvNet, tiny_llama

ALL_MODES = ["collab_v0", "ref_linear", "ref_attn", "ref_conv", "reference_hw", "chip"]
IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))


def _out(y):
    return getattr(y, "logits", y)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_defaults_equal_none(mode):
    """w=tensor, a=tensor, rule=ref, attn=head is the reference profile, bit for bit."""
    need_refs()
    for build, inp, skip in ((tiny_llama, IDS, ["lm_head"]),
                             (lambda: ConvNet().to(torch.bfloat16),
                              torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16), [])):
        torch.manual_seed(0)
        m0 = build()
        m1 = copy.deepcopy(m0)
        algos.quantize(m0, mode, skip, dict(FAST), algo=None)
        rep = algos.quantize(m1, mode, skip, dict(FAST), algo=algos.make("granularity"))
        with torch.no_grad():
            y0, y1 = _out(m0(inp)), _out(m1(inp))
        assert torch.equal(y0.view(torch.int16), y1.view(torch.int16)), mode
        assert rep["algo"]["numerics"]["departs_from_reference_rule"] is False


def test_nc_rule_is_ceil_log2_amax_over_448():
    """rule=nc == ceil(log2(amax/448)) for every positive finite BF16 amax in a wide range
    (all 2^16 bit patterns with exponent field 100..160), including exact 448 * 2^k boundaries."""
    bits = torch.arange(0, 1 << 15, dtype=torch.int32)
    vals = (bits << 16).view(torch.float32)
    e = (bits >> 7) & 0xFF
    vals = vals[(e >= 100) & (e <= 160)]
    x = vals.to(torch.bfloat16).reshape(-1, 1)
    got = scale_exp(x, (-1,), "nc").reshape(-1)
    want = torch.tensor([math.ceil(math.log2(float(v) / 448.0)) for v in vals.tolist()], dtype=torch.int32)
    assert torch.equal(got, want)
    # ... and the reference rule is floor(log2(amax)) - 8 (unchanged)
    assert torch.equal(scale_exp(x, (-1,), "ref"), rn.po2_scale_exp(x, (-1,)))
    for k in (-3, 0, 5):                                   # boundaries: 448*2^k fits, no increment
        t = torch.tensor([[448.0 * 2 ** k]]).to(torch.bfloat16)
        assert int(scale_exp(t, (-1,), "nc")) == k


def test_nc_never_clips_and_ref_does():
    x = torch.tensor([[460.0, 1.0, -3.0], [500.0, 2.0, 0.5]]).to(torch.bfloat16)
    for rule, clipped in (("ref", True), ("nc", False)):
        s = scale_exp(x, (-1,), rule)
        dq = rn.dequantize(rn.e4m3_vpack(x, s), s)
        assert bool((dq[1, 0] < 500 * 0.95)) is clipped
    sb = block_scale_exp(torch.randn(4, 64), "nc")
    assert sb.shape == (4, 2, 1)


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_output_rescale_exact_on_mxu_model(mxu):
    """reference_hw with w=channel, a=token: the MXU model sees only E4M3 codes, and the output is
    model(codes) * 2^(sx[m] + sw[n]) exactly (then BF16), i.e. the scales are a pure output rescale."""
    need_refs()
    import accum_torch as at

    torch.manual_seed(0)
    lin = nn.Linear(96, 40, bias=False).to(torch.bfloat16)
    x = (torch.randn(2, 5, 96) * torch.exp(torch.randn(2, 5, 1) * 3)).to(torch.bfloat16)
    m = nn.Sequential(copy.deepcopy(lin))
    algos.quantize(m, "reference_hw", options={**FAST, "mxu": mxu},
                   algo=algos.make("granularity:w=channel,a=token,rule=nc"))
    with torch.no_grad():
        y = m(x)
    sw = scale_exp(lin.weight, (-1,), "nc")                       # [40, 1]
    sx = scale_exp(x, (-1,), "nc")                                # [2, 5, 1]
    xq, wq = rn.e4m3_vpack(x, sx), rn.e4m3_vpack(lin.weight.detach(), sw)
    fm = at.ipt_linear if mxu == "ipt" else (lambda a, b, c: at.sa_linear(a, b, c, compile=False))
    y_codes = fm(rn.e4m3_unpack(xq), rn.e4m3_unpack(wq), None)
    want = (y_codes * torch.exp2((sx + sw.reshape(-1)).float())).to(torch.bfloat16)
    assert torch.equal(y.view(torch.int16), want.view(torch.int16))


def test_framework_path_matches_dequantized_matmul():
    torch.manual_seed(0)
    lin = nn.Linear(64, 24, bias=False).to(torch.bfloat16)
    x = torch.randn(7, 64).to(torch.bfloat16)
    for spec, wdims, xdims in (("w=channel,a=token", (-1,), (-1,)), ("w=channel,a=tensor", (-1,), None)):
        m = nn.Sequential(copy.deepcopy(lin))
        algos.quantize(m, "ref_linear", algo=algos.make("granularity:" + spec))
        sw, sx = scale_exp(lin.weight, wdims, "ref"), scale_exp(x, xdims, "ref")
        wd = rn.dequantize(rn.e4m3_vpack(lin.weight.detach(), sw), sw).to(torch.bfloat16)
        xd = rn.dequantize(rn.e4m3_vpack(x, sx), sx).to(torch.bfloat16)
        with torch.no_grad():
            assert torch.equal(m(x).view(torch.int16), torch.nn.functional.linear(xd, wd).view(torch.int16))


def test_block32_framework_only():
    need_refs()
    torch.manual_seed(0)
    base = nn.Sequential(nn.Linear(64, 16), nn.ReLU(), nn.Linear(16, 8)).to(torch.bfloat16)  # K=16: tail fallback
    x = torch.randn(3, 64).to(torch.bfloat16)
    ref = base(x)
    m = copy.deepcopy(base)
    rep = algos.quantize(m, "ref_linear", algo=algos.make("granularity:w=block32,a=block32,rule=nc"))
    with torch.no_grad():
        y = m(x)
    assert ((y.float() - ref.float()).norm() / ref.float().norm()) < 0.1
    g = rep["algo"]["granularity"]
    assert g["weight"]["linear:per_block(block=32)"] == 1 and g["weight"]["linear:per_channel(block_tail)"] == 1
    assert g["activation"]["linear:per_block(block=32)"] == 1
    for mode in ("reference_hw", "chip"):
        with pytest.raises(NotImplementedError):
            algos.quantize(copy.deepcopy(base), mode, options=dict(FAST),
                           algo=algos.make("granularity:w=block32,a=tensor"))
        m2 = copy.deepcopy(base)
        algos.quantize(m2, mode, options=dict(FAST), algo=algos.make("granularity:w=tensor,a=block32"))
        with pytest.raises(NotImplementedError), torch.no_grad():
            m2(x)


@pytest.mark.parametrize("mode", ["ref_attn", "reference_hw", "chip"])
def test_token_channel_run_on_mxu_steps(mode):
    need_refs()
    m0, m1 = tiny_llama(), tiny_llama()
    algos.quantize(m0, mode, ["lm_head"], dict(FAST), algo=None)
    rep = algos.quantize(m1, mode, ["lm_head"], dict(FAST),
                         algo=algos.make("granularity:w=channel,a=token,rule=nc,attn=token"))
    with torch.no_grad():
        y0, y1 = _out(m0(IDS)).float(), _out(m1(IDS)).float()
    assert torch.isfinite(y1).all() and ((y1 - y0).norm() / y0.norm()) < 0.1
    g = rep["algo"]["granularity"]
    assert g["weight"]["linear:per_channel"] == 7 and g["activation"]["linear:per_token"] == 7
    assert g["activation"]["attention.q:per_token"] == 1 and g["activation"]["attention.v:per_head"] == 1
    assert rep["algo"]["numerics"]["departs_from_reference_rule"] is True


def test_chip_coverage_counts_extra_vmuls():
    """chip: one vmul per activation element (2^-sx before vpack) for every per-token cast, plus one
    per output element for a linear with both a row and a column factor; amax + vpack counted by the cast."""
    need_refs()
    model = tiny_llama()
    rep = algos.quantize(model, "chip", ["lm_head"], dict(FAST),
                         algo=algos.make("granularity:w=channel,a=token,rule=nc,attn=token"))
    lin = {n: m for n, m in model.named_modules() if type(m).__name__ == "RefQuantLinear"}
    seen = []
    hs = [m.register_forward_pre_hook(lambda mod, a: seen.append((a[0].numel(), a[0].numel() // a[0].shape[-1]
                                                                   * mod.out_features)))
          for m in lin.values()]
    with torch.no_grad():
        model(IDS)
    for h in hs:
        h.remove()
    rows = {r["op"]: r for r in rep["coverage"]["ops"] if r["unit"] == "VPU"}
    r = rows["granularity_rescale"]
    lin_vmul = sum(a + b for a, b in seen)
    # the attention q and p per-token casts add their element counts (pre-vpack vmul only)
    B, H, S, D = 1, 4, 16, 16
    attn_vmul = B * H * S * D + B * H * S * S
    assert r["instructions"] == {"vmul.bf16": lin_vmul + attn_vmul}
    cast = rows["fp8_operand_cast"]["instructions"]
    assert cast["vredmax.row.bf16"] == cast["vpack.bf16.fp8"] > 0
    assert "exp2" not in rows                           # the exponent table is not a vexp2 per output
    assert any("granularity_rescale" in d["op"] for d in rep["coverage"]["new_decisions"])


def test_bad_params():
    for bad in ("granularity:w=row", "granularity:a=channel", "granularity:rule=ceil", "granularity:x=1"):
        with pytest.raises(ValueError):
            algos.make(bad)


# -- NEW a=block (tokens per block, full K) -----------------------------------------------------------

def _x(m=37, k=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(m, k, generator=g) * torch.exp(2 * torch.randn(m, 1, generator=g))).to(torch.bfloat16)


def test_block_rows1_equals_per_token():
    x = _x()
    got = algos.make("granularity:a=block,rows=1").act_scale("l", x, "linear")
    want = algos.make("granularity:a=token,rule=nc").act_scale("l", x, "linear")
    assert torch.equal(got.exp, want.exp) and got.exp.shape == (37, 1)


def test_block_rows_ge_tokens_equals_per_tensor():
    x = _x()
    want = algos.make("granularity:a=tensor,rule=nc").act_scale("l", x, "linear")
    for rows in (37, 64, 4096):
        got = algos.make(f"granularity:a=block,rows={rows}").act_scale("l", x, "linear")
        assert torch.equal(got.exp, want.exp.expand(37, 1))


def test_block_short_last_block_and_never_clips():
    x = _x(70, 32)                                      # 32 + 32 + 6 tokens
    s = algos.make("granularity:a=block").act_scale("l", x, "linear").exp.reshape(-1)
    for lo, hi in ((0, 32), (32, 64), (64, 70)):
        want = scale_exp(x[lo:hi], None, "nc")
        assert (s[lo:hi] == want).all()
    amax = x.abs().float().amax(-1)
    assert (amax <= 448.0 * torch.exp2(s.float())).all()


def test_block_default_rows_and_scope():
    g = algos.make("granularity:a=block")
    assert g.rows == 32
    x = _x(8, 32)
    assert g.act_scale("c", x, "conv2d") is None                       # Linear inputs only (rule=ref: reference)
    assert g.act_scale("a", x, "attention.q", dims=(-2, -1)) is None   # attention keeps the per-head rule
    assert g.record()["numerics"]["departs_from_reference_rule"] is True
    with pytest.raises(ValueError):
        algos.make("granularity:a=block,rows=0")


def test_block_runs_on_mxu_steps_and_chip_cost_below_per_token():
    need_refs()
    import hw_scale

    for mode in ("reference_hw", "chip"):
        m0, m1 = tiny_llama(), tiny_llama()
        algos.quantize(m0, mode, ["lm_head"], dict(FAST), algo=None)
        rep = algos.quantize(m1, mode, ["lm_head"], dict(FAST), algo=algos.make("granularity:a=block"))
        with torch.no_grad():
            y0, y1 = _out(m0(IDS)).float(), _out(m1(IDS)).float()
        assert torch.isfinite(y1).all() and ((y1 - y0).norm() / y0.norm()) < 0.1
        assert rep["algo"]["granularity"]["activation"]["linear:per_token_block"] == 7
    rows = {r["op"]: r for r in rep["coverage"]["ops"]}
    assert rows["dyn_scale_chain"]["instructions"]["vstore"] > 0 and "dyn_scale_chain_scalar" in rows
    assert "dyn_scale_chain" in rep["coverage"]["new_decisions"][-1]["op"] or any(
        "dyn_scale_chain" in d["op"] for d in rep["coverage"]["new_decisions"])
    assert rep["coverage"]["dynamic_scale_chain_NEW"]["block"]["serialized_cycles"] > 0
    # cost: a TinyLlama-sized layer input (128 tokens x K=2048)
    blk, tok = hw_scale.dynamic_scale_cost(128, 2048, "block"), hw_scale.dynamic_scale_cost(128, 2048, "token")
    assert blk["instructions"] < tok["instructions"]
    assert hw_scale.dynamic_scale_cost(128, 2048, "block_vec")["cycles"] < tok["cycles"]
