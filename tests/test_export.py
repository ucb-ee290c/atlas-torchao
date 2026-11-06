"""Export of a quantized model and its reload (docs/algos/export.md; src/exporter/quant_export.py,
src/exporter/load_quantized.py): the loaded model is bit-identical to the in-memory quantized one, the
files hold no float weight of a quantized layer, and the manifest's digests match the files."""
import copy
import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file

import algos
import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export
from test_ladder import FAST, ConvNet, tiny_llama

# The atlas_v1 recipes written out (docs/algos/atlas_v1.md); on a branch with algos.RECIPES the
# recipe names are used instead, and their expansion is checked against these.
RECIPE_SPECS = {
    "atlas_v1": "gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
    "atlas_v1_sq": "smoothquant:alpha=0.5,po2=true,gain=false,partial=true"
                "+gptq_awq:method=gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
    "atlas_v1_awq": "gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
    "atlas_v1_static": "smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true"
                       "+gptq_awq:method=gptq,actorder=true",
}
HAVE_RECIPES = hasattr(algos, "RECIPES")
SPECS = ["none"] + [n if HAVE_RECIPES else s for n, s in RECIPE_SPECS.items()]
IDS = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))


def _calib(n=2, T=12):
    g = torch.Generator().manual_seed(5)
    b = [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]
    return b, lambda: (b, lambda m, ids: m(ids, use_cache=False), {"source": "test", "n_samples": n, "seed": 5})


def _quantize_export(tmp_path, spec, mode="ref_conv", opts=None, layers=2, device="cpu"):
    opts = dict(opts or FAST)
    m = tiny_llama(layers).to(device)
    algo = algos.make(spec)
    batches, calib = _calib()
    if device != "cpu":
        b = [x.to(device) for x in batches]
        calib = lambda: (b, lambda mm, ids: mm(ids, use_cache=False), {"source": "test", "n_samples": 2, "seed": 5})  # noqa: E731
    rep = algos.quantize(m, mode, ["lm_head"], opts, algo=algo, calib=calib)
    d = tmp_path / "export"
    man = quant_export.export(m, d, mode=mode, report=rep, algo=algo, algo_spec=spec, options=opts,
                              skip=["lm_head"], model_record={"kind": "causal_lm", "id": "tiny"},
                              calib_batches=batches, calib_record={"source": "test"}, device=device)
    return m, d, man, opts


def _logits(m, ids):
    with torch.no_grad():
        return m(ids.to(next(m.parameters()).device), use_cache=False).logits


def _reload(d, mode, opts, layers=2, device="cpu"):
    m, rep, man = load_quantized.load_quantized(None, d, device, build=lambda: tiny_llama(layers).to(device),
                                                mode=mode, options=opts)
    return m


@pytest.mark.parametrize("spec", SPECS)
def test_round_trip_bit_identical(tmp_path, spec):
    m0, d, man, opts = _quantize_export(tmp_path, spec)
    y0 = _logits(m0, IDS)
    m1 = _reload(d, "ref_conv", opts)
    assert torch.equal(y0, _logits(m1, IDS))
    # ref_linear and ref_attn share the weights: the same export rebuilds them, and ref_linear
    # there equals ref_linear quantized in memory
    m2 = _reload(d, "ref_linear", opts)
    m3, *_ = _quantize_export(tmp_path / "lin", spec, mode="ref_linear")
    assert torch.equal(_logits(m2, IDS), _logits(m3, IDS))


@pytest.mark.parametrize("spec", [SPECS[0], SPECS[1], RECIPE_SPECS["atlas_v1_static"]
                                 if not HAVE_RECIPES else "atlas_v1_static"])
def test_round_trip_chip_framework(tmp_path, spec):
    opts = {**FAST, "chip_mxu": "framework"}
    m0, d, man, _ = _quantize_export(tmp_path, spec, mode="chip", opts=opts)
    y0 = _logits(m0, IDS)
    assert torch.equal(y0, _logits(_reload(d, "chip", opts), IDS))


def test_round_trip_reference_hw_and_options(tmp_path):
    """reference_hw (MXU ipt model) from a ref_conv export, against reference_hw in memory."""
    spec = SPECS[1]
    _, d, _, opts = _quantize_export(tmp_path, spec)
    m_hw, *_ = _quantize_export(tmp_path / "hw", spec, mode="reference_hw")
    assert torch.equal(_logits(m_hw, IDS), _logits(_reload(d, "reference_hw", opts), IDS))


@pytest.mark.parametrize("spec,mode", [
    ("granularity:w=block32,a=block32,rule=nc,attn=token", "ref_conv"),
    ("static_calib:per_step=true,p_exp=-8", "ref_conv"),
    ("granularity:w=channel,a=token,rule=nc+mixed_precision:units=model.layers.0.mlp.down_proj,keep_as=fp8x3",
     "reference_hw"),
    ("mixed_precision:units=model.layers.0.mlp.down_proj;model.layers.1.self_attn.o_proj,keep_as=bf16", "ref_conv"),
])
def test_round_trip_other_algorithms(tmp_path, spec, mode):
    m0, d, man, opts = _quantize_export(tmp_path, spec, mode=mode)
    y0, y0b = _logits(m0, IDS), _logits(m0, IDS)          # per_step: the second call uses step 1
    m1 = _reload(d, mode, opts)
    assert torch.equal(y0, _logits(m1, IDS)) and torch.equal(y0b, _logits(m1, IDS))
    pol = json.loads((d / "policy.json").read_text())
    if "keep_as=bf16" in spec:
        assert man["counts"]["host"] == 2
        st = load_file(str(d / "model.safetensors"))
        assert st["model.layers.0.mlp.down_proj.weight"].dtype == torch.bfloat16
        assert pol["sites"]["model.layers.0.mlp.down_proj"]["status"] == "host"
    if "fp8x3" in spec:
        assert pol["sites"]["model.layers.0.mlp.down_proj"]["residual_passes"]["kind"] == "wx"


def test_conv2d_round_trip(tmp_path):
    torch.manual_seed(0)
    m0 = ConvNet().to(torch.bfloat16).eval()
    fresh = copy.deepcopy(m0)
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    rep = algos.quantize(m0, "ref_conv", [], dict(FAST), algo=algo)
    quant_export.export(m0, tmp_path, mode="ref_conv", report=rep, algo=algo, algo_spec="granularity",
                        options=dict(FAST))
    x = torch.randn(2, 3, 8, 8, dtype=torch.bfloat16)
    m1, *_ = load_quantized.load_quantized(None, tmp_path, build=lambda: copy.deepcopy(fresh),
                                           options=dict(FAST))
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x))
    st = load_file(str(tmp_path / "model.safetensors"))
    assert "conv.weight" not in st and st["conv.weight_codes"].shape == (8, 27)


@pytest.mark.parametrize("mode", ["ref_conv", "reference_hw", "chip"])
def test_input_conv_bf16_exception_round_trip(tmp_path, mode):
    from test_algo_atlas_v1 import DIT_PATCH, TinyDiT, dit_calib

    torch.manual_seed(17)
    m0 = TinyDiT()
    fresh = copy.deepcopy(m0)
    opts = {**FAST, "chip_mxu": "framework"}
    spec = "atlas_v1+keep_bf16:rule=input_conv"
    a = algos.make(spec)
    rep = algos.quantize(m0, mode, options=opts, algo=a, calib=dit_calib())
    man = quant_export.export(m0, tmp_path, mode=mode, report=rep, algo=a,
                              algo_spec=spec, options=opts)
    pol = json.loads((tmp_path / "policy.json").read_text())
    reason = rep["algo"]["bf16_opt_out_reasons"][DIT_PATCH]
    assert pol["bf16_opt_out"] == [{"site": DIT_PATCH, "reason": reason, "match": "exact"}]
    assert pol["sites"][DIT_PATCH]["reason"] == reason and reason.startswith("keep_bf16 rule=input_conv")
    assert man["counts"]["host"] == 1
    rec = man["recipe"]
    assert (rec["algo"], rec["alias"], rec["resolved"]) == (spec, "atlas_v1", "atlas_v1_awq")
    assert rec["expanded"] == algos.RECIPES["atlas_v1_awq"]["spec"] + "+keep_bf16:rule=input_conv"
    assert rec["recipe"]["alias"] == "atlas_v1" and rec["recipe"]["expanded"] == rec["expanded"]
    assert rec["algorithm_record"]["bf16_opt_out_exact"] == [DIT_PATCH]
    st = load_file(str(tmp_path / "model.safetensors"))
    assert st[DIT_PATCH + ".weight"].dtype == torch.bfloat16
    assert DIT_PATCH + ".weight_codes" not in st
    assert DIT_PATCH + "_extra.weight_codes" in st
    m1, loaded, _ = load_quantized.load_quantized(None, tmp_path,
                         build=lambda: copy.deepcopy(fresh), mode=mode, options=opts)
    assert loaded["algo"]["bf16_opt_out_exact"] == [DIT_PATCH]
    assert type(m1.get_submodule(DIT_PATCH)) is torch.nn.Conv2d
    assert type(m1.get_submodule(DIT_PATCH + "_extra")).__name__ == "RefQuantConv2d"
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
    with torch.no_grad():
        assert torch.equal(m0(x), m1(x))


def test_recipe_export_records_no_bf16_exception(tmp_path):
    from test_algo_atlas_v1 import TinyDiT, dit_calib

    m0 = TinyDiT()
    a = algos.make("atlas_v1")
    rep = algos.quantize(m0, "ref_conv", options=dict(FAST), algo=a, calib=dit_calib())
    man = quant_export.export(m0, tmp_path, mode="ref_conv", report=rep, algo=a,
                              algo_spec="atlas_v1", options=dict(FAST))
    assert json.loads((tmp_path / "policy.json").read_text())["bf16_opt_out"] == []
    assert man["counts"]["host"] == 0
    assert (man["recipe"]["alias"], man["recipe"]["resolved"]) == ("atlas_v1", "atlas_v1_awq")


@pytest.mark.parametrize("spec", SPECS)
def test_export_contents(tmp_path, spec):
    m0, d, man, _ = _quantize_export(tmp_path, spec)
    st = load_file(str(d / "model.safetensors"))
    pol = json.loads((d / "policy.json").read_text())
    q = quant_export.quantized_modules(m0)
    assert q and len(q) == man["counts"]["quantized"] - len(quant_export.attention_sites(m0))
    for n, mod in q.items():
        assert f"{n}.weight" not in st                        # no float weight of a quantized layer
        codes = st[f"{n}.weight_codes"]
        assert codes.dtype == torch.uint8
        assert not ((codes == 0x7F) | (codes == 0xFF)).any()
        assert torch.equal(codes, mod.weight_codes.cpu())
        exp = st[f"{n}.weight_scale_exp"]
        assert exp.dtype == torch.int32 and torch.equal(exp, mod.weight_scale_exp.cpu().to(torch.int32))
        assert pol["sites"][n]["status"] == "quantized"
    # the BF16 tensors are the post-fold ones (smoothquant / AWQ change norms and producer rows)
    sd = m0.state_dict()
    for k, v in st.items():
        if not k.endswith(quant_export.QUANT_SUFFIXES):
            assert torch.equal(v, sd[k].cpu()), k
    expanded = str(man["recipe"]["expanded"])
    if "smoothquant" in expanded or "awq" in expanded:
        fresh = tiny_llama(2).state_dict()
        assert [k for k in st if k.endswith("layernorm.weight") and not torch.equal(st[k], fresh[k])]
    gran = {pol["sites"][n]["weight"]["granularity"] for n in q}
    if "granularity:w=channel" in str(man["recipe"]["expanded"]):
        assert gran == {"per_channel"}
    if "static=true" in str(man["recipe"]["expanded"]):
        assert all(pol["sites"][n]["activation"]["mode"] == "static" for n in q)
        assert all(f"{n}.input_scale_exp" in st for n in q)
    assert man["recipe"]["algo"] == spec


@pytest.mark.parametrize("spec", [SPECS[0], SPECS[1]])
def test_manifest_digests_and_fields(tmp_path, spec):
    _, d, man, _ = _quantize_export(tmp_path, spec)
    for f in ("model.safetensors", "policy.json"):
        assert man["files"][f]["sha256"] == hashlib.sha256((d / f).read_bytes()).hexdigest()
    assert man["policy_sha256"] == man["files"]["policy.json"]["sha256"]
    disk = json.loads((d / "manifest.json").read_text())
    assert disk["files"] == man["files"]
    assert disk["m2m_manifest_v2_aligned"] == "partial" and disk["m2m_manifest_v2_differences"]
    ids = [s["site_id"] for s in disk["sites"]]
    assert len(ids) == len(set(ids)) and all(s["status"] in ("quantized", "host", "skipped") for s in disk["sites"])
    assert disk["calibration"]["data_sha256"] if spec != "none" else True
    assert disk["new_decisions"] and disk["pending_hardware_rulings"]
    assert {"atlas_torchao_commit", "atlas_torchao_dirty", "versions"} <= set(disk["provenance"])
    # a changed file is refused on load
    (d / "policy.json").write_text((d / "policy.json").read_text() + " ")
    with pytest.raises(ValueError, match="sha256"):
        load_quantized.read_export(d)


def test_recipe_specs_match_the_registry():
    if not HAVE_RECIPES:
        pytest.skip("no named recipes on this branch")
    for n, s in RECIPE_SPECS.items():
        assert algos.expand(n) == s


def test_export_refuses_what_it_cannot_express():
    from algos.base import Algo

    class Online(Algo):
        name = "online_x"

        def online_transform(self, name, kind):
            return lambda x: x

    class Scaler(Algo):
        name = "unknown_scaler"

        def act_scale(self, name, x, kind, dims=None):
            return None

    with pytest.raises(NotImplementedError, match="online transform"):
        quant_export.check_exportable(Online())
    with pytest.raises(NotImplementedError, match="descriptor"):
        quant_export.check_exportable(Scaler())
    assert quant_export.pick_mode(["bf16", "collab_v0", "ref_linear", "ref_conv", "chip"]) == "ref_conv"
    assert quant_export.pick_mode(["bf16", "chip"]) == "chip"
    assert quant_export.pick_mode(["bf16", "collab_v0"]) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("spec", SPECS)
def test_round_trip_cuda(tmp_path, spec):
    m0, d, man, opts = _quantize_export(tmp_path, spec, device="cuda")
    y0 = _logits(m0, IDS)
    m1 = _reload(d, "ref_conv", opts, device="cuda")
    assert torch.equal(y0, _logits(m1, IDS))


# ---------------------------------------------------------------------------------------------
# Policies: SmolVLA's SmolVLMWithExpertModel on tiny Llama stacks, and a tiny random Pi0
# ---------------------------------------------------------------------------------------------

def test_smolvla_round_trip(tmp_path):
    from test_algo_smoothquant import _smol_batch, _smol_run, _smolvla

    m0 = _smolvla().to(torch.bfloat16)
    fresh = copy.deepcopy(m0)
    batches = [tuple(t.to(torch.bfloat16) for t in _smol_batch(s)) for s in (11, 12)]
    calib = lambda: (batches, _smol_run, {"source": "test", "n_samples": 2, "seed": 11})  # noqa: E731
    spec = SPECS[1]
    algo = algos.make(spec)
    rep = algos.quantize(m0, "ref_conv", [], dict(FAST), algo=algo, calib=calib)
    quant_export.export(m0, tmp_path, mode="ref_conv", report=rep, algo=algo, algo_spec=spec,
                        options=dict(FAST), model_record={"kind": "lerobot_policy", "id": "tiny-smolvla"})
    m1, *_ = load_quantized.load_quantized(None, tmp_path, build=lambda: copy.deepcopy(fresh), options=dict(FAST))
    b = tuple(t.to(torch.bfloat16) for t in _smol_batch(3))
    with torch.no_grad():
        for a, c in zip(_smol_run(m0, b), _smol_run(m1, b)):
            assert torch.equal(a, c)
    pol = json.loads((tmp_path / "policy.json").read_text())
    assert any(s["kind"] == "attention" for s in pol["sites"].values())


def test_pi0_tiny_round_trip(tmp_path):
    from refutil import need_refs

    need_refs()
    import pi0_tiny

    why = pi0_tiny.available()
    if why:
        pytest.skip(why)
    m0, cfg = pi0_tiny.build("bfloat16")
    fresh = copy.deepcopy(m0)
    obs, nz = pi0_tiny.observation(cfg), pi0_tiny.noise(cfg)
    run = lambda m, b: m.sample_actions("cpu", obs, noise=nz, num_steps=2)     # noqa: E731
    calib = lambda: ([0], run, {"source": "test", "n_samples": 1, "seed": 0})  # noqa: E731
    spec = SPECS[1]
    algo = algos.make(spec)
    rep = algos.quantize(m0, "ref_conv", ["lm_head"], dict(FAST), algo=algo, calib=calib)
    quant_export.export(m0, tmp_path, mode="ref_conv", report=rep, algo=algo, algo_spec=spec,
                        options=dict(FAST), skip=["lm_head"], model_record={"kind": "pi0", "id": "tiny-pi0"})
    m1, *_ = load_quantized.load_quantized(None, tmp_path, build=lambda: copy.deepcopy(fresh), options=dict(FAST))
    with torch.no_grad():
        assert torch.equal(run(m0, 0), run(m1, 0))
    pol = json.loads((tmp_path / "policy.json").read_text())
    assert any(s["kind"] == "conv2d" for s in pol["sites"].values())
