"""Stacked algorithms (`--algo A+B+...`, docs/algos/stacking.md): the parser, the fixed order,
calibration on the transformed model, the one-Scale-per-decision rule, and bit identity of the
plain paths."""
import copy

import pytest
import torch

import algos
import reference.ref_numerics as rn
from algos.base import Algo
from algos.granularity import scale_exp
from algos.stack import Stack, StackConflict
from refutil import need_refs
from test_ladder import FAST, tiny_llama

IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
ATLAS_V1 = "smoothquant:alpha=0.5,po2=true,gain=false+gptq_awq:method=gptq,actorder=true" \
           "+granularity:w=channel,a=token,rule=nc"


def _bits(t):
    return t.contiguous().view(torch.int16)


def _batches(n=2, T=12, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]


def _run(m, ids):
    return m(ids, use_cache=False)


def _calib(batches=None):
    b = batches or _batches()
    return lambda: (b, _run, {"source": "test", "n_samples": len(b), "seed": 5})


# -- parser -------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec,want", [
    (None, ("none", {})),
    ("none", ("none", {})),
    ("granularity:w=channel,a=token,rule=nc", ("granularity", {"w": "channel", "a": "token", "rule": "nc"})),
    ("smoothquant:alpha=0.5,po2=true,sites=qkv+fc1", ("smoothquant", {"alpha": 0.5, "po2": True, "sites": "qkv+fc1"})),
    ("gptq_awq:damp=0.01,block=64", ("gptq_awq", {"damp": 0.01, "block": 64})),
])
def test_single_spec_unchanged(spec, want):
    assert algos.parse(spec) == want
    assert algos.parse_stack(spec) == [want]
    a = algos.make(spec)
    assert (a is None) if want[0] == "none" else (type(a).__name__ != "Stack" and a.name == want[0])


def test_comma_continuation_and_plus_inside_values():
    assert algos.parse("example_identity:online=down,o,impl=fp32") == (
        "example_identity", {"online": "down,o", "impl": "fp32"})
    assert algos.split("smoothquant:sites=qkv+fc1+down,alpha=0.5+granularity:w=channel") == [
        "smoothquant:sites=qkv+fc1+down,alpha=0.5", "granularity:w=channel"]
    with pytest.raises(ValueError):
        algos.parse("granularity:channel")                  # no k=v before the bare token
    with pytest.raises(ValueError):
        algos.make("none+granularity")
    with pytest.raises(ValueError):
        algos.make("granularity+nosuchalgo:x=1")           # '+nosuchalgo' joins w's value -> bad value


# -- order ---------------------------------------------------------------------------------------

def test_fixed_order_whatever_the_spec_order():
    a = algos.make("mixed_precision:units=x+granularity:w=channel+gptq_awq:method=gptq+smoothquant:alpha=0.5")
    assert [m.name for m in a.members] == ["smoothquant", "gptq_awq", "granularity", "mixed_precision"]
    assert a.delegations and a.delegations[0]["pinned_by"] == "gptq_awq"
    assert a.delegations[0]["grid_from"] == ["granularity"]


def test_calibrate_and_prepare_interleave_in_order():
    """Each member calibrates on the model the earlier members' prepare left."""
    log = []

    def member(name, calib):
        class M(Algo):
            needs_calibration = calib

            def calibrate(self, model, batches, run):
                log.append((name, "calibrate", float(model[0].weight.detach().sum())))

            def prepare(self, model, mode):
                log.append((name, "prepare", None))
                with torch.no_grad():
                    model[0].weight.mul_(2.0)              # a fold the next member must see
                return {"ok": name}
        m = M()
        m.name = name
        return m

    net = torch.nn.Sequential(torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16))
    w0 = float(net[0].weight.detach().sum())
    st = Stack([member("granularity", True), member("smoothquant", True), member("gptq_awq", False)], spec="t")
    rep = algos.quantize(net, "ref_linear", (), dict(FAST), algo=st,
                         calib=lambda: ([torch.zeros(1, 8)], lambda m, b: m(b), {"source": "t"}))
    assert [(n, k) for n, k, _ in log] == [("smoothquant", "calibrate"), ("smoothquant", "prepare"),
                                           ("gptq_awq", "prepare"),
                                           ("granularity", "calibrate"), ("granularity", "prepare")]
    assert log[0][2] == pytest.approx(w0) and log[3][2] == pytest.approx(4 * w0, rel=1e-2)
    r = rep["algo"]
    assert r["name"] == "stack" and r["order"] == ["smoothquant", "gptq_awq", "granularity"]
    assert list(r["prepare"]) == ["smoothquant", "gptq_awq", "granularity"]
    assert set(r["calibration"]) == {"smoothquant", "granularity"}
    assert [m["stage"] for m in r["members"]] == ["transform", "transform", "scale"]


# -- conflicts -------------------------------------------------------------------------------------

def test_two_act_scales_for_one_layer_raise():
    need_refs()
    m = tiny_llama()
    algo = algos.make("granularity:a=token+static_calib:method=absmax")
    algos.quantize(m, "ref_linear", ["lm_head"], dict(FAST), algo=algo, calib=_calib())
    with pytest.raises(StackConflict, match="act_scale"), torch.no_grad():
        m(IDS)


def test_two_weight_scales_raise_at_quantize():
    need_refs()
    with pytest.raises(StackConflict, match="weight_scale"):
        algos.quantize(tiny_llama(), "ref_linear", ["lm_head"], dict(FAST),
                       algo=algos.make("granularity:w=channel+static_calib:weight=ceil448"), calib=_calib())


def test_double_fold_and_duplicates_refused():
    with pytest.raises(StackConflict, match="fold"):
        algos.make("smoothquant:alpha=0.5+gptq_awq:method=awq_gptq")
    algos.make("smoothquant:alpha=off+gptq_awq:method=awq_gptq")      # gain only: no smoothing fold
    with pytest.raises(StackConflict, match="more than once"):
        algos.make("granularity:w=channel+granularity:a=token")


# -- bit identity ----------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ref_linear", "ref_attn", "reference_hw", "chip"])
def test_none_and_neutral_stack_bit_identical(mode):
    """`none` is no algorithm at all, and a stack of neutral members (example_identity and the
    reference-default granularity) takes the algorithm path and still equals it bit for bit."""
    need_refs()
    torch.manual_seed(0)
    m0 = tiny_llama()
    m1, m2 = copy.deepcopy(m0), copy.deepcopy(m0)
    assert algos.make("none") is None
    algos.quantize(m0, mode, ["lm_head"], dict(FAST), algo=None)
    algos.quantize(m1, mode, ["lm_head"], dict(FAST), algo=algos.make("none"))
    rep = algos.quantize(m2, mode, ["lm_head"], dict(FAST), algo=algos.make("granularity+example_identity"))
    with torch.no_grad():
        y0, y1, y2 = m0(IDS).logits, m1(IDS).logits, m2(IDS).logits
    assert torch.equal(_bits(y0), _bits(y1)) and torch.equal(_bits(y0), _bits(y2))
    assert rep["algo"]["order"] == ["example_identity", "granularity"]


# -- the integrated stack ----------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ref_linear", "chip"])
def test_smoothing_gptq_per_channel_stack(mode):
    """smoothquant (folded) + gptq on granularity's per-channel grid + per-token activations:
    GPTQ codes are pinned at granularity's [out, 1] exponents of the smoothed weight, and the
    output stays close to the reference profile."""
    need_refs()
    torch.manual_seed(0)
    m0 = tiny_llama()
    m1 = copy.deepcopy(m0)
    algos.quantize(m0, mode, ["lm_head"], dict(FAST), algo=None)
    algo = algos.make(ATLAS_V1)
    rep = algos.quantize(m1, mode, ["lm_head"], dict(FAST), algo=algo, calib=_calib())
    with torch.no_grad():
        y0, y1 = m0(IDS).logits.float(), m1(IDS).logits.float()
    assert torch.isfinite(y1).all() and ((y1 - y0).norm() / y0.norm()) < 0.1
    r = rep["algo"]
    assert r["order"] == ["smoothquant", "gptq_awq", "granularity"]
    assert r["granularity"]["weight"]["linear:per_channel"] == 7
    assert r["granularity"]["activation"]["linear:per_token"] == 7
    gq = next(x for x in r["members"] if x["name"] == "gptq_awq")
    assert gq["weights"] == {"gptq": 7}
    lin = next(mod for n, mod in m1.named_modules() if n.endswith("layers.0.mlp.down_proj"))
    s = lin.weight_scale_exp
    assert s.shape == (lin.out_features, 1)
    # pinned at granularity's rule on the (smoothed) weight GPTQ received; every code representable
    assert torch.equal(s, scale_exp(lin.weight.detach(), (-1,), "nc"))
    dq = rn.dequantize(lin.weight_codes, s)
    assert torch.equal(rn.dequantize(rn.e4m3_vpack(dq, s), s), dq)
