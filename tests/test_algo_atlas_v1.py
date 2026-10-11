"""Named recipes atlas_v1, atlas_v1_awq and atlas_v1_static (docs/algos/atlas_v1.md): expansion,
recorded chip defaults, explicit refusals, and one quantized forward per recipe."""
import copy

import pytest
import torch

import algos
from refutil import need_refs
from test_ladder import FAST, tiny_llama

IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
ORDERS = {"atlas_v1": ["smoothquant", "gptq_awq", "granularity"],
          "atlas_v1_awq": ["gptq_awq", "granularity"],
          "atlas_v1_static": ["smoothquant", "gptq_awq"]}


def _calib(n=2, T=12):
    g = torch.Generator().manual_seed(5)
    b = [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]
    return lambda: (b, lambda m, ids: m(ids, use_cache=False), {"source": "test", "n_samples": n, "seed": 5})


@pytest.mark.parametrize("name", list(ORDERS))
def test_recipe_expands_to_its_documented_stack(name):
    a = algos.make(name)
    assert [m.name for m in a.members] == ORDERS[name]
    assert a.recipe["name"] == name and a.recipe["spec"] == algos.RECIPES[name]["spec"]
    assert a.recipe["chip_defaults"] == {"chip_vadd": "rn", "chip_vadd_sites": "residual"}
    sq = next((m for m in a.members if m.name == "smoothquant"), None)
    if sq is not None:
        assert sq.partial is True                      # policies: unfolded blocks are recorded
    rec = algos.none_record(name, "bf16")
    assert rec["recipe"]["name"] == name and rec["applied"] is False


def test_recipe_components():
    v1, awq, st = (algos.make(n) for n in ORDERS)
    sq = v1.members[0]
    assert sq.alpha == 0.5 and sq.po2 and not sq.gain and not sq.static
    assert v1.members[1].method == "gptq" and v1.members[1].actorder
    g = v1.members[2]
    assert (g.w, g.a, g.rule) == ("channel", "token", "nc")
    assert awq.members[0].method == "awq_gptq"
    assert st.members[0].static and st.members[0].gain and st.members[1].method == "gptq"
    assert not any(m.name == "granularity" for m in st.members)          # per-token is a run-time scale
    assert not any(m.name == "mixed_precision" for n in ORDERS for m in algos.make(n).members)


def test_opt_in_and_refusals():
    a = algos.make("atlas_v1+mixed_precision:units=model.layers.0.mlp.down_proj")
    assert [m.name for m in a.members][-1] == "mixed_precision" and a.recipe["name"] == "atlas_v1"
    with pytest.raises(ValueError, match="no parameters"):
        algos.make("atlas_v1:alpha=0.3")
    with pytest.raises(ValueError, match="more than one recipe"):
        algos.make("atlas_v1+atlas_v1_static")
    with pytest.raises(ValueError, match="listed more than once"):
        algos.make("atlas_v1+granularity:w=channel")


def test_env_router_refuses_an_algorithm(monkeypatch):
    need_refs()
    import chip_coverage as cc

    monkeypatch.setenv("ATLAS_ALGO", "atlas_v1")
    with pytest.raises(NotImplementedError, match="eval_policy.py"):
        cc.make_mtdit_op_emulator()
    monkeypatch.setenv("ATLAS_ALGO", "none")
    cc.make_mtdit_op_emulator()


@pytest.mark.parametrize("name", list(ORDERS))
@pytest.mark.parametrize("mode", ["ref_conv", "chip"])
def test_recipe_runs_and_records_chip_defaults(name, mode):
    need_refs()
    m0 = tiny_llama(layers=2)
    m1 = copy.deepcopy(m0)
    opts = {**FAST, "chip_mxu": "framework"}
    algos.quantize(m0, mode, ["lm_head"], dict(opts), algo=None)
    rep = algos.quantize(m1, mode, ["lm_head"], dict(opts), algo=algos.make(name), calib=_calib())
    with torch.no_grad():
        y0, y1 = m0(IDS).logits.float(), m1(IDS).logits.float()
    assert torch.isfinite(y1).all() and ((y1 - y0).norm() / y0.norm()) < 0.1
    r = rep["algo"]
    assert r["order"] == ORDERS[name] and r["recipe"]["name"] == name
    if mode == "chip":
        assert r["recipe"]["chip_options_applied"] == {"chip_vadd": "rn", "chip_vadd_sites": "residual"}
        assert any("--chip-vadd rn" in s for s in rep["ladder"]["off_ladder_reasons"])
    else:
        assert r["recipe"]["chip_options_applied"] == {}
    if "smoothquant" in ORDERS[name]:
        assert r["prepare"]["smoothquant"]["not_folded_blocks"] == []


def test_explicit_chip_vadd_wins():
    need_refs()
    m = tiny_llama()
    rep = algos.quantize(m, "chip", ["lm_head"], {**FAST, "chip_mxu": "framework", "chip_vadd": "trunc"},
                         algo=algos.make("atlas_v1_awq"), calib=_calib())
    assert rep["algo"]["recipe"]["chip_options_applied"] == {"chip_vadd_sites": "residual"}
    assert not any("--chip-vadd" in s for s in rep["ladder"]["off_ladder_reasons"])


@pytest.mark.parametrize("name", list(ORDERS))
def test_recipe_on_pi0_records_unfolded_blocks(name):
    """Pi0 (and SmolVLA / DiT) carry blocks smoothquant does not fold (SigLIP): the recipe's
    partial=true folds the supported decoder layers and lists the rest in the record."""
    need_refs()
    import pi0_tiny
    why = pi0_tiny.available()
    if why:
        pytest.skip(why)
    m, cfg = pi0_tiny.build("float32")
    obs, nz = pi0_tiny.observation(cfg), pi0_tiny.noise(cfg)
    run = lambda mm, b: mm.sample_actions("cpu", obs, noise=nz, num_steps=2)     # noqa: E731
    rep = algos.quantize(m, "ref_linear", [], dict(FAST), algo=algos.make(name),
                         calib=lambda: ([0], run, {"source": "test", "n_samples": 1, "seed": 0}))
    with torch.no_grad():
        assert torch.isfinite(run(m, 0)).all()
    if "smoothquant" in ORDERS[name]:
        assert rep["algo"]["prepare"]["smoothquant"]["not_folded_blocks"]
