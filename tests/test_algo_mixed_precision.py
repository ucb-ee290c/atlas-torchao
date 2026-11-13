"""mixed_precision: ranking-file selection, exact-name checks, the FP8x3 residual passes, chip
coverage of both opt-out routes, and --algo none / k=0 left bit-identical."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import algos
import reference.ref_numerics as rn
from algos.mixed_precision import check_exact, select
from refutil import need_refs
from test_ladder import FAST, tiny_llama

REPO = Path(__file__).resolve().parent.parent
IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
DOWN0 = "model.layers.0.mlp.down_proj"


def _bits(t):
    return t.view(torch.int16)


def _rank_file(tmp_path, layers, name="rank.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"meta": {"test": True}, "layers": layers}))
    return p


# -- selection ---------------------------------------------------------------------------------

def test_select_k_budget_and_ties():
    L = {"a": {"fisher_out": 3.0, "macs": 10}, "b": {"fisher_out": 2.0, "macs": 80},
         "c": {"fisher_out": 2.0, "macs": 5}, "d": {"fisher_out": 1.0, "macs": 5}}
    assert select(L, k=2) == ["a", "b"]                         # ties broken by name
    assert select(L, mac_budget=0.2) == ["a", "c", "d"]         # b does not fit; greedy goes on
    with pytest.raises(ValueError):
        select(L)
    with pytest.raises(ValueError):
        select(L, score="nope", k=1)


def test_exact_names_and_substring_collision():
    model = tiny_llama(layers=12)
    check_exact(model, ["model.layers.1.mlp.down_proj", "model.layers.11.mlp.down_proj"])
    with pytest.raises(ValueError, match="not a Linear"):
        check_exact(model, ["layers.1.mlp"])
    net = nn.ModuleDict({"a": nn.Linear(4, 4), "xa": nn.Linear(4, 4)})
    with pytest.raises(ValueError, match="substring"):
        check_exact(net, ["a"])


def test_bf16_selection_from_rank_file_and_record(tmp_path):
    need_refs()
    model = tiny_llama(layers=2)
    names = [n for n, m in model.named_modules() if type(m) is nn.Linear and "lm_head" not in n]
    L = {n: {"fisher_out": float(i), "hawq_w": 0.0, "iso_rel_rmse": 0.0,
             "macs": dict(model.named_modules())[n].weight.numel()} for i, n in enumerate(names)}
    rf = _rank_file(tmp_path, L)
    algo = algos.make(f"mixed_precision:rank_file={rf},k=2")
    rep = algos.quantize(model, "ref_linear", ["lm_head"], dict(FAST), algo=algo)
    top = sorted(names, key=lambda n: -L[n]["fisher_out"])[:2]
    assert rep["algo"]["bf16_opt_out"] == top
    mods = dict(model.named_modules())
    assert all(type(mods[n]) is nn.Linear for n in top)
    assert sum(type(mods[n]).__name__ == "RefQuantLinear" for n in names) == len(names) - 2
    sel = rep["algo"]["selection"]
    assert sel["units"] == top and sel["ranking"]["sha256"] and sel["ranking"]["meta"] == {"test": True}
    want = sum(L[n]["macs"] for n in top) / sum(v["macs"] for v in L.values())
    assert sel["mac_share_unquantized"] == pytest.approx(want)


def test_k0_and_none_bit_identical(tmp_path):
    """k=0 selects nothing and must leave every reference step exactly as --algo none."""
    need_refs()
    rf = _rank_file(tmp_path, {DOWN0: {"fisher_out": 1.0, "macs": 1}})
    for mode in ("ref_linear", "reference_hw", "chip"):
        torch.manual_seed(0)
        m0 = tiny_llama()
        m1 = copy.deepcopy(m0)
        algos.quantize(m0, mode, ["lm_head"], dict(FAST), algo=None)
        algos.quantize(m1, mode, ["lm_head"], dict(FAST), algo=algos.make(f"mixed_precision:rank_file={rf},k=0"))
        with torch.no_grad():
            assert torch.equal(_bits(m0(IDS).logits), _bits(m1(IDS).logits)), mode


# -- ranking -----------------------------------------------------------------------------------

def test_ranking_reproducible():
    need_refs()
    sys.path.insert(0, str(REPO / "scripts"))
    import rank_layers

    g = torch.Generator().manual_seed(7)
    batches = [torch.randint(0, 128, (1, 24), generator=g) for _ in range(2)]
    rec = {"source": "synthetic", "seed": 7}
    a = rank_layers.rank(tiny_llama(layers=2), batches, rec)
    b = rank_layers.rank(tiny_llama(layers=2), batches, rec)
    assert a["layers"] == b["layers"]
    some = next(iter(a["layers"].values()))
    assert {"fisher_out", "hawq_w", "iso_rel_rmse", "macs"} <= set(some)
    assert a["meta"]["calibration"] == rec and a["meta"]["profile"] == "ref_linear"
    assert all(v["fisher_out"] >= 0 and v["hawq_w"] >= 0 for v in a["layers"].values())
    assert a["meta"]["device"] == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_ranking_on_cuda():
    """The model on CUDA, the calibration batches on the CPU (as calib.wikitext2_train returns
    them): every layer is scored, with finite scores, and the record names the device."""
    need_refs()
    sys.path.insert(0, str(REPO / "scripts"))
    import rank_layers

    g = torch.Generator().manual_seed(7)
    batches = [torch.randint(0, 128, (1, 24), generator=g) for _ in range(2)]
    cpu = rank_layers.rank(tiny_llama(layers=2), batches, {"seed": 7})
    gpu = rank_layers.rank(tiny_llama(layers=2).to("cuda"), batches, {"seed": 7})
    assert gpu["meta"]["device"].startswith("cuda")
    assert gpu["layers"].keys() == cpu["layers"].keys()
    for v in gpu["layers"].values():
        assert all(torch.isfinite(torch.tensor(v[k])) for k in ("fisher_out", "hawq_w", "iso_rel_rmse"))


# -- FP8x3 -------------------------------------------------------------------------------------

def _q(t):
    q, s = rn.quantize(t)
    return rn.dequantize(q, s).to(t.dtype)


def _fp8x3_definition(x, w):
    """x_hi.w_hi + x_hi.w_lo + x_lo.w_hi, each part the reference cast, BF16 sums (framework)."""
    w_hi = _q(w)
    w_lo = _q((w.float() - w_hi.float()).to(w.dtype))
    x_hi = _q(x)
    x_lo = _q(x - x_hi)
    return F.linear(x_hi, w_hi) + F.linear(x_hi, w_lo) + F.linear(x_lo, w_hi)


def _one_layer(w, keep_as):
    lin = nn.Linear(w.shape[1], w.shape[0], bias=False).to(torch.bfloat16)
    with torch.no_grad():
        lin.weight.copy_(w)
    net = nn.Sequential(lin)
    spec = f"mixed_precision:units=0,keep_as={keep_as}" if keep_as else None
    rep = algos.quantize(net, "ref_linear", (), dict(FAST), algo=algos.make(spec) if spec else None)
    return net, rep


def _outlier_data():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(32, 64, generator=g)
    x[:, :3] *= 60.0                                     # outlier channels
    w = torch.randn(48, 64, generator=g) * 0.05
    w[:2] *= 40.0                                         # outlier rows
    return x.bfloat16(), w.bfloat16()


def test_fp8x3_equals_definition():
    need_refs()
    x, w = _outlier_data()
    net, rep = _one_layer(w, "fp8x3")
    assert rep["algo"]["bf16_opt_out"] == [] and net[0].residual == "wx"
    with torch.no_grad():
        assert torch.equal(_bits(net(x)), _bits(_fp8x3_definition(x, w)))


def test_fp8x3_beats_single_fp8_on_outliers():
    need_refs()
    x, w = _outlier_data()
    ref = x.double() @ w.double().T
    err = {}
    for keep in (None, "fp8x2_w", "fp8x3"):
        net, _ = _one_layer(w, keep)
        with torch.no_grad():
            y = net(x).double()
        err[keep] = ((y - ref).norm() / ref.norm()).item()
    assert err["fp8x3"] < 0.3 * err[None], err
    assert err["fp8x2_w"] <= err[None], err


def test_fp8x3_counts_three_mxu_passes_on_chip():
    need_refs()

    def mxu_linear(spec):
        model = tiny_llama()
        rep = algos.quantize(model, "chip", ["lm_head"], dict(FAST), algo=algos.make(spec) if spec else None)
        with torch.no_grad():
            model(IDS)
        rows = {(o["unit"], o["op"]): o for o in rep["coverage"]["ops"]}
        return rows[("MXU", "linear")]["work"], rep

    base, _ = mxu_linear(None)
    x3, rep = mxu_linear(f"mixed_precision:units={DOWN0},keep_as=fp8x3")
    m = tiny_llama()
    down = dict(m.named_modules())[DOWN0].weight.numel() * IDS.shape[1] * 2      # 2*M*N*K FLOPs
    assert x3 - base == 2 * down
    assert rep["algo"]["selection"]["extra_mxu_passes"] == 2
    assert "residual_passes" in rep["algo"]["module_hooks"]


def test_fp8x3_refused_for_conv_and_collab_records_note():
    need_refs()
    net = nn.Sequential(nn.Conv2d(3, 4, 3)).to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        algos.quantize(net, "ref_conv", (), dict(FAST), algo=algos.make("mixed_precision:units=0,keep_as=fp8x3"))
    model = tiny_llama()
    rep = algos.quantize(model, "collab_v0", ["lm_head"], dict(FAST),
                         algo=algos.make(f"mixed_precision:units={DOWN0},keep_as=fp8x3"))
    assert rep["algo"]["module_hooks_applied"] is False and "module_hooks_note" in rep["algo"]


# -- FP8x3 with per-token / per-channel scales (granularity) -------------------------------------

def _token_channel_x3():
    """granularity:w=channel,a=token,rule=nc whose every Linear also takes the fp8x3 passes."""
    from algos.granularity import Granularity

    class TokenChannelX3(Granularity):
        name = "token_channel_x3"

        def residual_passes(self, name, kind):
            return "wx" if kind == "linear" else None

    return TokenChannelX3(w="channel", a="token", rule="nc")


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_fp8x3_with_token_channel_scales_on_mxu_model(mxu):
    """Each residual pass is rescaled by the [..., out] exponent table 2^(sx[m] + sw[n]) of
    algos.base.output_scale: x_hi.w_hi with both tables, x_hi.w_lo with the per-token row only,
    x_lo.w_hi with the per-channel column only, summed in BF16."""
    need_refs()
    import mxu.accum_torch as at
    from algos.granularity import scale_exp

    torch.manual_seed(0)
    lin = nn.Linear(96, 40, bias=False).to(torch.bfloat16)
    x = (torch.randn(2, 5, 96) * torch.exp(torch.randn(2, 5, 1) * 3)).to(torch.bfloat16)
    m = nn.Sequential(copy.deepcopy(lin))
    rep = algos.quantize(m, "reference_hw", options={**FAST, "mxu": mxu}, algo=_token_channel_x3())
    with torch.no_grad():
        y = m(x)
    fm = at.ipt_linear if mxu == "ipt" else (lambda a, b, c: at.sa_linear(a, b, c, compile=False))
    w = lin.weight.detach()
    sw = scale_exp(w, (-1,), "nc")
    sx = scale_exp(x, (-1,), "nc")
    wq, xq = rn.e4m3_vpack(w, sw), rn.e4m3_vpack(x, sx)
    w_lo = (w.float() - rn.dequantize(wq, sw).float()).to(w.dtype)
    wlq, swl = rn.quantize(w_lo)
    x_lo = x - rn.dequantize(xq, sx).to(x.dtype)
    xlq, sxl = rn.quantize(x_lo)

    def p(a_q, sa, b_q, sb):
        sb = sb.reshape(-1) if sb.dim() == 2 else sb
        return (fm(rn.e4m3_unpack(a_q), rn.e4m3_unpack(b_q), None) * torch.exp2((sa + sb).float())).to(torch.bfloat16)

    want = p(xq, sx, wq, sw) + p(xq, sx, wlq, swl) + p(xlq, sxl, wq, sw)
    assert torch.equal(_bits(y), _bits(want))
    g = rep["algo"]["granularity"]
    assert g["weight"]["linear:per_channel"] == 1 and g["weight"]["linear:residual_wx"] == 1


def test_fp8x3_with_token_channel_scales_on_chip_counts_no_vexp2():
    """chip: the residual passes' exponent tables are built under the chip guard like the single
    pass (no vexp2 over every output element), and the per-token cast's vmuls are counted."""
    need_refs()
    model = tiny_llama()
    rep = algos.quantize(model, "chip", ["lm_head"], dict(FAST), algo=_token_channel_x3())
    with torch.no_grad():
        y = model(IDS).logits
    assert torch.isfinite(y.float()).all()
    ops = rep["coverage"]["ops"]
    # before output_scale, every pass built its [..., out] table outside the guard: counted as
    # one exp2 per output element (and an add per table) on the scalar unit
    assert not [r for r in ops if r["op"] == "exp2"]
    rows = {r["op"]: r for r in ops if r["unit"] == "VPU"}
    assert rows["granularity_rescale"]["instructions"]["vmul.bf16"] > 0
