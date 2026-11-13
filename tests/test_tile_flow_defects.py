"""Defects of the tile-aware flow (docs/algos/tile-aware.md "Limits"): uncalibrated Linear under a static recipe,
the dynamic-recipe pack-admission report, make_bundle_manifest on a Pi0-shaped manifest, and the declared-vs-unexpected
classification of verify_bundle.py."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
import exporter.fx_export as fx_export
import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export
import exporter.tile_native as tn
from refutil import need_refs
from test_export import _calib, _quantize_export, _reload
from test_ladder import FAST, tiny_llama

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts" / "tile_flow"
STATIC = "atlas_v1_awq_tile_static"
INPUTS = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
          (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _llama_with_unused_head():
    m = tiny_llama(1)
    m.unused_head = nn.Linear(64, 64, bias=False).to(torch.bfloat16)      # never called: no calibration data
    return m


def _export_with_unused_head(tmp_path, spec, **kw):
    m = _llama_with_unused_head()
    algo = algos.make(spec)
    batches, calib = _calib()
    rep = algos.quantize(m, "ref_conv", ["lm_head"], dict(FAST), algo=algo, calib=calib)
    d = tmp_path / "export"
    quant_export.export(m, d, mode="ref_conv", report=rep, algo=algo, algo_spec=spec, options=dict(FAST),
                        skip=["lm_head"], model_record={"kind": "causal_lm", "id": "tiny"},
                        calib_batches=batches, calib_record={"source": "test"}, **kw)
    return m, d


# -- 1. uncalibrated Linear under a static recipe -------------------------------------------------------

def test_uncalibrated_linear_is_declared_not_tile_native_and_numerics_unchanged(tmp_path):
    need_refs()
    m0, d = _export_with_unused_head(tmp_path, STATIC)
    policy = json.loads((d / "policy.json").read_text())
    site = policy["sites"]["unused_head"]
    assert site["activation"]["uncalibrated_fallback"] is True and site["activation"]["label"] == "uncalibrated_fallback"
    assert site["activation"]["source"] == "reference" and site["activation"]["mode"] == "dynamic"   # numerics: unchanged
    assert site["tile_native"]["tile_native"] is False and site["tile_native"]["uncalibrated_fallback"] is True
    assert "uncalibrated_fallback" in " ".join(site["tile_native"]["reasons"])
    fb = policy["uncalibrated_fallback"]
    assert fb["static_recipe"] is True and fb["count"] == 1 and fb["sites"] == ["unused_head"] and fb["declared"] is False
    tile = policy["tile_native"]
    assert tile["all_tile_native"] is False and list(tile["not_tile_native"]) == ["unused_head"]
    assert tile["uncalibrated_fallback_sites"] == ["unused_head"] and tile["uncalibrated_fallback_count"] == 1
    # calibrated sites keep their static rule and stay tile-native
    others = [s for n, s in policy["sites"].items() if s["kind"] == "linear" and n != "unused_head"]
    assert others and all(s["activation"]["mode"] == "static" and s["tile_native"]["tile_native"] for s in others)
    # replay: same dynamic rule, bit-identical logits
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    m1, _, _ = load_quantized.load_quantized(None, d, "cpu", build=_llama_with_unused_head, mode="ref_conv",
                                             options=dict(FAST))
    with torch.no_grad():
        assert torch.equal(m0(ids, use_cache=False).logits, m1(ids, use_cache=False).logits)


def test_uncalibrated_declaration_flag_is_recorded(tmp_path):
    need_refs()
    _, d = _export_with_unused_head(tmp_path, STATIC, allow_uncalibrated_fallback=True)
    fb = json.loads((d / "policy.json").read_text())["uncalibrated_fallback"]
    assert fb["declared"] is True and fb["count"] == 1


@pytest.mark.parametrize("spec", ["atlas_v1_awq_tile", "atlas_v1_awq"])
def test_dynamic_recipes_are_not_flagged_as_uncalibrated(tmp_path, spec):
    need_refs()
    _, d = _export_with_unused_head(tmp_path, spec)
    policy = json.loads((d / "policy.json").read_text())
    assert policy["uncalibrated_fallback"]["static_recipe"] is False and policy["uncalibrated_fallback"]["count"] == 0
    assert "uncalibrated_fallback" not in policy["sites"]["unused_head"]["activation"]


def test_cli_exposes_the_declaration_flag():
    import argparse

    ap = argparse.ArgumentParser()
    quant_export.add_args(ap)
    assert ap.parse_args([]).allow_uncalibrated_fallback is False
    assert ap.parse_args(["--allow-uncalibrated-fallback"]).allow_uncalibrated_fallback is True


def test_fallback_covers_only_the_kinds_a_static_member_calibrates():
    sq = algos.make("smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true")
    assert quant_export.act_rule(sq, "x", "linear").get("uncalibrated_fallback") is True
    assert quant_export.act_rule(sq, "x", "conv2d") == quant_export.REFERENCE_ACT      # SQ never calibrates a conv
    sc = algos.make("static_calib")
    assert all(quant_export.act_rule(sc, "x", k).get("uncalibrated_fallback") is True for k in ("linear", "conv2d"))
    assert quant_export.act_rule(sc, "x", "attention.q") == quant_export.REFERENCE_ACT   # q / k / v dynamic by design
    assert "uncalibrated_fallback" not in quant_export.act_rule(algos.make("atlas_v1_awq_tile"), "x", "linear")
    assert quant_export.act_rule(algos.make("atlas_v1_awq"), "x", "linear")["source"] == "granularity"


def test_activation_rule_status_of_the_fallback_rule():
    st = tn.activation_rule_status(dict(quant_export.UNCALIBRATED_FALLBACK_ACT))
    assert st["tile_native"] is False and st["uncalibrated_fallback"] is True and st["dynamic"] is True
    assert tn.activation_rule_status(dict(quant_export.REFERENCE_ACT))["tile_native"] is True    # unchanged


# -- 2. dynamic recipes: pack admission is not a pass -----------------------------------------------------

def test_dynamic_recipe_pack_admission_reports_not_checked(tmp_path):
    need_refs()
    _, d, _, _ = _quantize_export(tmp_path, "atlas_v1_awq_tile", layers=1)
    adm = json.loads((d / "policy.json").read_text())["compiler_pack_scale_admission"]
    assert adm["status"] == "not_checked_dynamic"
    assert adm["sites_checked"] is None and adm["sites_outside"] is None and adm["offending_sites"] is None
    assert "graph_inspection.json" in adm["see"] and adm["admitted_range"] == [122, 128]


def test_pack_admission_report_statuses():
    assert tn.pack_admission_report({}, {"w": (100, 130)})["status"] == "not_checked_dynamic"
    rep = tn.pack_admission_report({"a": [-8]}, {})
    assert rep["status"] == "checked_static_sites" and rep["sites_checked"] == 1 and rep["sites_outside"] == 1


# -- 3. make_bundle_manifest on a Pi0-shaped manifest ---------------------------------------------------------

def _bundle_dir(tmp_path, model):
    d = tmp_path / "bundle"
    (d / "export").mkdir(parents=True)
    (d / "fx").mkdir()
    man = {"model": model, "recipe": {"algo": "x", "expanded": "x", "stack": [{"name": "granularity",
                                                                         "params": {"rule": "nc", "rows": 32}}]},
           "calibration": {"source": "test"}, "device": None,
           "provenance": {"atlas_torchao_commit": "c", "atlas_torchao_branch": "b", "atlas_torchao_dirty": False,
                          "versions": {}, "device": "cpu"},
           "counts": {"quantized": 0}, "files": {"model.safetensors": {"sha256": "0", "bytes": 0},
                                                 "policy.json": {"sha256": "0", "bytes": 0}}}
    pol = {"sites": {}, "tile_native": {"sites_checked": 0, "tile_native_sites": 0, "all_tile_native": True},
           "compiler_pack_scale_admission": {"status": "not_checked_dynamic", "sites_checked": None,
                                             "sites_outside": None}}
    (d / "export" / "manifest.json").write_text(json.dumps(man))
    (d / "export" / "policy.json").write_text(json.dumps(pol))
    (d / "fx" / "fx_manifest.json").write_text(json.dumps({"provenance": {"capture_source": {"commit": "c", "dirty": ""}},
                                                           "native_fp8": True}))
    for n in ("model.pt2", "graph_inspection.json", "buffer_inventory.json", "policy.json", "graph.txt",
              "verification.json"):
        (d / "fx" / n).write_text("{}")
    (d / "verify_bundle.json").write_text(json.dumps({
        "ok": True, "failed": [], "checks": {"a": {"ok": True}},
        "graph": {"call_function_targets_atlas": {}, "allowed_non_atlas_contractions": []}}))
    return d


@pytest.mark.parametrize("model,want", [
    ({"kind": "pi0", "cli": "src/eval_pi0.py", "config": "pi0_droid", "checkpoint": {"dir": "x"}}, "pi0_droid"),
    ({"kind": "causal_lm", "id": "tiny", "checkpoint": {"dir": "x"}}, "tiny"),
    ({"kind": "other", "name": "named"}, "named"),
    ({"kind": "bare"}, None)])
def test_make_bundle_manifest_model_id_fallback(tmp_path, model, want):
    mbm = _script("make_bundle_manifest")
    d = _bundle_dir(tmp_path, model)
    mbm.main(d)                                                            # the Pi0 shape used to KeyError 'id'
    out = json.loads((d / "bundle_manifest.json").read_text())
    assert out["model"]["id"] == want
    assert out["compiler_pack_scale_admission"]["status"] == "not_checked_dynamic"
    assert out["compiler_pack_scale_admission"]["policy_static_sites_checked"] is None


# -- 4. verify_bundle: declared vs unexpected ------------------------------------------------------------------

def _c(target, operands, module=None, node="n"):
    return {"node": node, "target": target, "module": module,
            "operands": [{"name": n, "dtype": dt, "shape": sh} for n, dt, sh in operands]}


BF16 = "torch.bfloat16"
POLICY = {"sites": {
    "enc.layers.0.self_attn": {"kind": "attention", "status": "quantized"},
    "enc.patch": {"kind": "Conv2d", "status": "host"},
    "enc.layers.0.fc": {"kind": "linear", "status": "quantized"},
    "lm_head": {"kind": "Linear", "status": "skipped"}}}
TARGETS = {"p_patch_w": "model.enc.patch.weight", "p_fc_w": "model.enc.layers.0.fc.weight", "p_x": "model.other.weight"}


def _classify(cs, **kw):
    vb = _script("verify_bundle")
    return vb.classify_contractions(cs, POLICY, TARGETS, **kw)


def test_declared_bf16_by_policy_conv_is_allowed():
    conv = _c("aten.conv2d.default", [("img", BF16, [1, 3, 8, 8]), ("p_patch_w", BF16, [4, 3, 2, 2])])
    allowed, unexpected = _classify([conv])
    assert not unexpected and allowed[0]["declared_as"] == "bf16_by_policy" and "enc.patch" in allowed[0]["why"]


def test_plain_sdpa_outside_policy_attention_is_allowed_and_counted():
    sdpa = _c("aten.scaled_dot_product_attention.default", [("q", BF16, [1, 8, 4, 8])] * 3,
              module="model.noise_predictor.blocks.0.attn")
    allowed, unexpected = _classify([sdpa])
    assert not unexpected and allowed[0]["declared_as"] == "unquantized_attention"
    allowed, unexpected = _classify([sdpa], allow_unquantized_attention=False)       # strict mode
    assert unexpected and not allowed


def test_legacy_lm_head_and_rope_rules_still_hold():
    lm = _c("aten.linear.default", [("h", BF16, [1, 4, 8]), ("p_lm_head_weight", BF16, [8, 8])])
    rope = _c("aten.matmul.default", [("inv", "torch.float32", [4, 1]), ("pos", "torch.float32", [1, 8])])
    allowed, unexpected = _classify([lm, rope])
    assert not unexpected and [a["declared_as"] for a in allowed] == ["bf16_by_policy", "rope_outer_product"]


@pytest.mark.parametrize("name,contraction", [
    ("bf16 mm on activations", _c("aten.mm.default", [("a", BF16, [4, 8]), ("b", BF16, [8, 4])])),
    ("bf16 matmul on activations", _c("aten.matmul.default", [("a", BF16, [1, 4, 8]), ("b", BF16, [1, 8, 4])])),
    ("bf16 Linear of a quantized site", _c("aten.linear.default", [("x", BF16, [1, 4, 8]), ("p_fc_w", BF16, [8, 8])])),
    ("Linear with an unknown weight", _c("aten.linear.default", [("x", BF16, [1, 4, 8]), ("p_x", BF16, [8, 8])])),
    ("SDPA inside a policy attention site", _c("aten.scaled_dot_product_attention.default", [("q", BF16, [1, 8, 4, 8])] * 3,
                                               module="model.enc.layers.0.self_attn")),
    ("SDPA without module info", _c("aten.scaled_dot_product_attention.default", [("q", BF16, [1, 8, 4, 8])] * 3)),
    ("SDPA in float32", _c("aten.scaled_dot_product_attention.default", [("q", "torch.float32", [1, 8, 4, 8])] * 3,
                           module="model.free.attn")),
])
def test_genuinely_unexpected_contraction_still_fails(name, contraction):
    allowed, unexpected = _classify([contraction])
    assert unexpected == [contraction] and not allowed, name


def _capture(tmp_path, wrapper_cls=None):
    need_refs()
    tmp_path.mkdir(parents=True, exist_ok=True)
    eager, d, _, opts = _quantize_export(tmp_path, STATIC, layers=1)
    loaded = _reload(d, "ref_conv", opts, layers=1)
    policy = json.loads((d / "policy.json").read_text())
    wrapped = (wrapper_cls or fx_export.CausalLogits)(loaded).eval()
    fx_export.capture_fx(wrapped, INPUTS, tmp_path / "fx", policy=policy, fresh_process=True, native_fp8="uint8")
    return d, tmp_path / "fx"


class _ExtraMatmul(fx_export.CausalLogits):
    def forward(self, ids):
        y = super().forward(ids)
        return y @ y.transpose(-1, -2)                                          # a BF16 matmul no policy site covers


def test_verify_bundle_end_to_end_pass_unexpected_and_static_check(tmp_path):
    vb = _script("verify_bundle")
    d, fx = _capture(tmp_path / "good")
    res = vb.run(d, fx, tmp_path / "v.json")
    # the tiny random model has one exponent per site, so "panel not collapsed" cannot hold; nothing else may fail
    assert set(res["failed"]) <= {"panel_not_collapsed_to_per_tensor"}, res["failed"]
    assert res["graph"]["unquantized_attention"]["count"] == 0
    assert res["checks"]["static_recipe_fully_static"]["ok"] and res["static_recipe_check"]["static_recipe"] is True
    # a static bundle that carries an undeclared uncalibrated Linear fails; declared (export or CLI flag) passes
    pol = json.loads((d / "policy.json").read_text())
    site = next(n for n, s in pol["sites"].items() if s["kind"] == "linear")
    pol["sites"][site]["activation"] = dict(quant_export.UNCALIBRATED_FALLBACK_ACT)
    pol["uncalibrated_fallback"] = {"static_recipe": True, "count": 1, "sites": [site], "declared": False}
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    (tampered / "model.safetensors").symlink_to(d / "model.safetensors")
    (tampered / "policy.json").write_text(json.dumps(pol))
    bad = vb.run(tampered, fx, tmp_path / "v2.json")
    assert "static_recipe_fully_static" in bad["failed"] and bad["static_recipe_check"]["uncalibrated_fallback_count"] == 1
    assert vb.run(tampered, fx, tmp_path / "v3.json", allow_fallback=True)["checks"]["static_recipe_fully_static"]["ok"]
    pol["uncalibrated_fallback"]["declared"] = True
    (tampered / "policy.json").write_text(json.dumps(pol))
    assert vb.run(tampered, fx, tmp_path / "v4.json")["checks"]["static_recipe_fully_static"]["ok"]
    # a genuinely unexpected BF16 contraction in the graph fails the check
    d2, fx2 = _capture(tmp_path / "bad", _ExtraMatmul)
    res2 = vb.run(d2, fx2, tmp_path / "v5.json")
    assert "no_unexpected_non_atlas_contraction" in res2["failed"]
    assert res2["checks"]["no_unexpected_non_atlas_contraction"]["detail"][0]["target"].startswith(("aten.matmul", "aten.bmm"))
