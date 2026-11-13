"""Named recipes atlas_v1_sq, atlas_v1_awq, atlas_v1_static and the alias atlas_v1
(docs/algos/atlas_v1.md): immutable expansions, alias records, recorded chip defaults, explicit
refusals, and one quantized forward per recipe."""
import copy

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

# The recipes, verbatim (docs/algos/atlas_v1.md). A recipe name never changes meaning: a new stack
# needs a new name, never an edit here.
FROZEN = {
    "atlas_v1_sq": "smoothquant:alpha=0.5,po2=true,gain=false,partial=true"
                   "+gptq_awq:method=gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
    "atlas_v1_awq": "gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
    "atlas_v1_static": "smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true"
                       "+gptq_awq:method=gptq,actorder=true",
    # tile-aware recipes (NEW; docs/algos/tile-aware.md, tests/test_algo_tile.py)
    "atlas_v1_awq_tile": "gptq_awq:method=awq_gptq,actorder=true"
                         "+granularity:w=panel,a=block,rows=32,rule=nc,attn=head",
    "atlas_v1_tile": "gptq_awq:method=awq_gptq,actorder=true"
                     "+granularity:w=panel,a=block,rows=32,rule=nc,attn=head+keep_bf16:rule=input_conv",
    "atlas_v1_awq_tile_static": "gptq_awq:method=awq_gptq,actorder=true"
                                "+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib",
}
CHIP = {"chip_vadd": "rn", "chip_vadd_sites": "residual"}

DIT_PATCH = "observation_encoder.vision_encoder.model.vision_model.embeddings.patch_embedding"


class TinyDiT(torch.nn.Module):
    """A vision-tower path ending in an RGB patch embedding, plus a similarly named sibling
    convolution that reads the patch features (4 channels), which substring skips would also hit."""

    def __init__(self):
        super().__init__()
        parent = self
        for name in DIT_PATCH.split(".")[:-1]:
            child = torch.nn.Module()
            parent.add_module(name, child)
            parent = child
        parent.patch_embedding = torch.nn.Conv2d(3, 4, 2, stride=2)
        parent.patch_embedding_extra = torch.nn.Conv2d(4, 4, 1)
        self.head = torch.nn.Linear(4, 3)
        self.to(torch.bfloat16).eval()

    def forward(self, x):
        emb = self.get_submodule(DIT_PATCH.rsplit(".", 1)[0])
        return self.head(emb.patch_embedding_extra(emb.patch_embedding(x)).mean((2, 3)))


def dit_calib():
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16,
                    generator=torch.Generator().manual_seed(42))
    return lambda: ([x], lambda m, b: m(b), {"source": "test", "n_samples": 2, "seed": 42})


def test_recipes_are_frozen():
    assert set(algos.RECIPES) == set(FROZEN)
    for name, spec in FROZEN.items():
        assert algos.RECIPES[name]["spec"] == spec
        assert algos.RECIPES[name]["chip"] == CHIP
        assert not set(algos.RECIPES[name]) - {"spec", "chip", "activation_scales", "new_decisions"}


def test_alias_resolves_to_one_recipe():
    assert algos.ALIASES == {"atlas_v1": "atlas_v1_awq"}
    assert algos.resolve("atlas_v1") == ("atlas_v1", "atlas_v1_awq")
    assert algos.resolve("atlas_v1_sq") == (None, "atlas_v1_sq")
    assert algos.expand("atlas_v1") == FROZEN["atlas_v1_awq"]
    with pytest.raises(ValueError, match="unknown recipe"):
        algos.resolve("atlas_v2")


@pytest.mark.parametrize("spec,alias,name", [
    ("atlas_v1", "atlas_v1", "atlas_v1_awq"),
    ("atlas_v1_awq", None, "atlas_v1_awq"),
    ("atlas_v1+mixed_precision:units=a.b", "atlas_v1", "atlas_v1_awq"),
    ("atlas_v1_sq", None, "atlas_v1_sq"),
])
def test_records_carry_alias_resolved_and_expanded(spec, alias, name):
    expanded = algos.expand(spec)
    assert expanded.startswith(FROZEN[name])
    for rec in (algos.make(spec).recipe, algos.none_record(spec, "bf16")["recipe"]):
        assert rec["requested"] == spec.partition("+mixed")[0]
        assert rec["alias"] == alias and rec["name"] == rec["resolved"] == name
        assert rec["spec"] == FROZEN[name] and rec["expanded"] == expanded


@pytest.mark.parametrize("recipe", list(ORDERS))
def test_no_recipe_keeps_a_layer_in_bf16(recipe):
    """The input-conv exception is opt-in (keep_bf16): no recipe applies it."""
    need_refs()
    m = TinyDiT()
    rep = algos.quantize(m, "ref_conv", algo=algos.make(recipe), calib=dit_calib())
    assert rep["algo"]["bf16_opt_out"] == [] and rep["algo"]["bf16_opt_out_exact"] == []
    assert type(m.get_submodule(DIT_PATCH)).__name__ == "RefQuantConv2d"


def test_alias_matches_its_recipe_bit_for_bit():
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
    resolved = algos.resolve(name)[1]
    assert [m.name for m in a.members] == ORDERS[name]
    assert a.recipe["name"] == resolved and a.recipe["spec"] == algos.RECIPES[resolved]["spec"]
    assert a.recipe["chip_defaults"] == CHIP
    sq = next((m for m in a.members if m.name == "smoothquant"), None)
    if sq is not None:
        assert sq.partial is True                      # policies: unfolded blocks are recorded
    rec = algos.none_record(name, "bf16")
    assert rec["recipe"]["name"] == resolved and rec["applied"] is False


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
    assert [m.name for m in a.members][-1] == "mixed_precision" and a.recipe["alias"] == "atlas_v1"
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
    assert r["order"] == ORDERS[name] and r["recipe"]["name"] == algos.resolve(name)[1]
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
