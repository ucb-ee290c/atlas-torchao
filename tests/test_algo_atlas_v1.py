"""Named recipes atlas_v1, atlas_v1_awq and atlas_v1_static (docs/algos/atlas_v1.md): expansion,
recorded chip defaults, explicit refusals, and one quantized forward per recipe."""
import copy
from types import SimpleNamespace

import pytest
import torch

import algos
from refutil import need_refs
from test_ladder import FAST, tiny_llama

IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
ORDERS = {"atlas_v1": ["gptq_awq", "granularity"],
          "atlas_v1_awq": ["gptq_awq", "granularity"],
          "atlas_v1_sq": ["smoothquant", "gptq_awq", "granularity"],
          "atlas_v1_static": ["smoothquant", "gptq_awq"]}

DIT_PATCH = "observation_encoder.vision_encoder.model.vision_model.embeddings.patch_embedding"


class TinyDiT(torch.nn.Module):
    """The exact policy identity/path plus a sibling that substring skips would also exclude."""

    def __init__(self, policy_type="multi_task_dit"):
        super().__init__()
        self.config = SimpleNamespace(type=policy_type)
        parent = self
        for name in DIT_PATCH.split(".")[:-1]:
            child = torch.nn.Module()
            parent.add_module(name, child)
            parent = child
        parent.patch_embedding = torch.nn.Conv2d(3, 4, 2, stride=2)
        parent.patch_embedding_extra = torch.nn.Conv2d(3, 4, 2, stride=2)
        self.head = torch.nn.Linear(4, 3)
        self.to(torch.bfloat16).eval()

    def forward(self, x):
        emb = self.get_submodule(DIT_PATCH.rsplit(".", 1)[0])
        return self.head((emb.patch_embedding(x) + emb.patch_embedding_extra(x)).mean((2, 3)))


def dit_calib():
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(42))
    return lambda: ([x], lambda m, b: m(b), {"source": "test", "n_samples": 2, "seed": 42})


@pytest.mark.parametrize("policy_type,recipe,expected", [
    ("multi_task_dit", "atlas_v1", [DIT_PATCH]),
    ("smolvla", "atlas_v1", []),
    ("pi0", "atlas_v1", []),
    ("multi_task_dit", "atlas_v1_awq", []),
])
def test_exact_dit_exception_and_historical_alias(policy_type, recipe, expected):
    need_refs()
    m = TinyDiT(policy_type)
    w0 = m.get_submodule(DIT_PATCH).weight.detach().clone()
    a = algos.make(recipe)
    rep = algos.quantize(m, "ref_conv", algo=a, calib=dit_calib())
    assert rep["algo"]["bf16_opt_out"] == expected
    assert rep["algo"]["bf16_opt_out_exact"] == expected
    assert (type(m.get_submodule(DIT_PATCH)) is torch.nn.Conv2d) == bool(expected)
    assert type(m.get_submodule(DIT_PATCH + "_extra")).__name__ == "RefQuantConv2d"
    if expected:
        assert torch.equal(m.get_submodule(DIT_PATCH).weight, w0)
        assert all(DIT_PATCH not in member.layers for member in a.members if hasattr(member, "layers"))
        assert rep["algo"]["recipe"]["new_decisions"]


@pytest.mark.parametrize("missing", [False, True])
def test_dit_exception_refuses_changed_structure_before_calibration(missing):
    m = TinyDiT()
    emb = m.get_submodule(DIT_PATCH.rsplit(".", 1)[0])
    if missing:
        del emb.patch_embedding
    else:
        emb.patch_embedding = torch.nn.Identity()
    def unexpected_calib():
        pytest.fail("the exact exception must be validated before calibration")
    with pytest.raises(ValueError, match="expected exact BF16 Conv2d"):
        algos.quantize(m, "ref_conv", algo=algos.make("atlas_v1"), calib=unexpected_calib)


def test_canonical_non_dit_matches_historical_awq_bit_for_bit():
    need_refs()
    canonical = tiny_llama(layers=2)
    historical = copy.deepcopy(canonical)
    for m, name in ((canonical, "atlas_v1"), (historical, "atlas_v1_awq")):
        rep = algos.quantize(m, "ref_conv", ["lm_head"], dict(FAST),
                             algo=algos.make(name), calib=_calib())
        assert rep["algo"]["bf16_opt_out"] == []
    with torch.no_grad():
        assert torch.equal(canonical(IDS).logits, historical(IDS).logits)


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
    v1, awq, historical, st = (algos.make(n) for n in ORDERS)
    sq = historical.members[0]
    assert sq.alpha == 0.5 and sq.po2 and not sq.gain and not sq.static
    assert v1.members[0].method == "awq_gptq" and v1.members[0].actorder
    assert historical.members[1].method == "gptq"
    g = v1.members[1]
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
