"""verify_bundle.py on bundles where one policy site is not one graph node (SmolVLA): a quantized layer the captured
boundary never calls (lm_head), a module called as module.forward() (no scope on the graph nodes) so one attention site holds
many qk/pv pairs, a Conv2d next to the Linears, and the wrapper attribute 'policy.' instead of 'model.'.
The positive cases build on a real tiny capture; the real SmolVLA bundle is used when present. Negative cases show the
relaxation does not accept an undeclared or inconsistent bundle (docs/algos/tile-aware.md#verify)."""
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from test_tile_flow_defects import _capture, _script

REAL = Path("/scratch2/lorenhung/atlas-fp8/exports-tile/smolvla/atlas_v1_awq_tile_static")
LEGACY_OK = {"panel_not_collapsed_to_per_tensor"}        # tiny random weights carry one exponent per site


# -- helpers (no capture needed) -------------------------------------------------------------------------------------------

def test_scope_helpers_handle_both_wrapper_names_and_scope_less_nodes():
    vb = _script("verify_bundle")
    sites = ["model.vlm_with_expert", "model.vlm_with_expert.vlm.vision.layers.0.self_attn", "layers.1.self_attn"]
    assert vb._owner_site("policy.model.vlm_with_expert.vlm.vision.layers.0.self_attn", sites) == sites[1]   # longest wins
    assert vb._owner_site("policy.model.vlm_with_expert", sites) == "model.vlm_with_expert"
    assert vb._owner_site("model.layers.1.self_attn", sites) == "layers.1.self_attn"                      # LM wrapper
    assert vb._owner_site("", sites) is None and vb._owner_site(None, sites) is None                      # module.forward()
    assert vb._owner_site("policy.model.other", sites) is None
    bufs = ["policy.model.a.weight_codes", "policy.model.a.weight_scale_exp", "model.layers.0.q.weight_codes"]
    stems = vb._wrapper_stems(bufs, "weight_codes")
    assert "model.a" in stems and "layers.0.q" in stems and "policy.model.a" in stems and "model.b" not in stems


def test_host_conv_under_the_policy_wrapper_is_declared_bf16_by_policy():
    """keep_bf16 SmolVLA bundle: the BF16 patch-embedding Conv2d has parameter target 'policy.model.<...>.weight' and the
    host site 'model.<...>'. It is a declared bf16_by_policy contraction (stripping only 'model.' never matched it); a
    Conv2d whose site is quantized, or that no site covers, stays unexpected."""
    vb = _script("verify_bundle")
    site = "model.vlm_with_expert.vlm.model.vision_model.embeddings.patch_embedding"
    policy = {"sites": {site: {"status": "host", "kind": "Conv2d"},
                        "model.action_in_proj": {"status": "quantized", "kind": "linear"}}}
    conv = {"node": "conv2d", "target": "aten.conv2d.padding", "module": "policy." + site,
            "operands": [{"name": "to", "dtype": "torch.bfloat16", "shape": [1, 3, 512, 512]},
                         {"name": "p_w", "dtype": "torch.bfloat16", "shape": [768, 3, 16, 16]},
                         {"name": "p_b", "dtype": "torch.bfloat16", "shape": [768]}]}
    for prefix in ("policy.", ""):                                     # SmolVLA wrapper and the LM wrapper 'model.'
        allowed, unexpected = vb.classify_contractions([conv], policy, {"p_w": f"{prefix}{site}.weight"})
        assert not unexpected and [a["declared_as"] for a in allowed] == ["bf16_by_policy"], (prefix, unexpected)
    for target in ("policy.model.action_in_proj.weight", "policy.model.other.weight"):
        allowed, unexpected = vb.classify_contractions([conv], policy, {"p_w": target})
        assert not allowed and len(unexpected) == 1


# -- a real tiny capture, with a never-called quantized site grafted on ----------------------------------------------------

@pytest.fixture(scope="module")
def good(tmp_path_factory):
    d, fx = _capture(tmp_path_factory.mktemp("cap"))
    return d, fx


def _graft_unused(tmp, good, declare, legacy=True, fallback_flag=False):
    """Copy of the bundle whose policy has one more quantized Linear (reference rule, no calibration) that the graph never
    calls. declare: list it in graph_inspection.unused_quantized_modules. legacy: no `uncalibrated_fallback` field."""
    d, fx = good
    d2, fx2 = tmp / "export", tmp / "fx"
    d2.mkdir(parents=True)
    shutil.copytree(fx, fx2)
    pol = json.loads((d / "policy.json").read_text())
    src = next(n for n, s in pol["sites"].items() if s["kind"] == "linear" and s["status"] == "quantized")
    new = "unused_head"
    site = json.loads(json.dumps(pol["sites"][src]))
    site["activation"] = {"mode": "dynamic", "source": "reference", "rule": "ref"}
    pol["sites"][new] = site
    pol.pop("uncalibrated_fallback", None)
    if not legacy:
        pol["uncalibrated_fallback"] = {"static_recipe": True, "count": 1, "sites": [new], "declared": False}
    (d2 / "policy.json").write_text(json.dumps(pol))
    tensors = load_file(str(d / "model.safetensors"))
    w = site["weight"]
    tensors[f"{new}.weight_codes"], tensors[f"{new}.weight_scale_exp"] = tensors[w["codes"]].clone(), tensors[w["scale_exp"]].clone()
    site["weight"] = {**w, "codes": f"{new}.weight_codes", "scale_exp": f"{new}.weight_scale_exp"}
    (d2 / "policy.json").write_text(json.dumps(pol))
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(d2 / "model.safetensors"))
    gi = json.loads((fx2 / "graph_inspection.json").read_text())
    gi["unused_quantized_modules"] = [{"site": new, "module": f"model.{new}"}] if declare else []
    (fx2 / "graph_inspection.json").write_text(json.dumps(gi))
    return d2, fx2


def test_never_called_fallback_site_is_exempt_only_when_graph_and_bundle_agree(tmp_path, good):
    vb = _script("verify_bundle")
    d, fx = _graft_unused(tmp_path / "ok", good, declare=True)
    res = vb.run(d, fx, tmp_path / "ok.json")
    assert set(res["failed"]) <= LEGACY_OK, res["failed"]
    assert res["graph_inventory"]["unused_in_graph_derived"] == ["unused_head"]
    assert res["static_recipe_check"]["fallback_sites_never_called_in_graph"] == ["unused_head"]
    assert res["static_recipe_check"]["fallback_sites_live_in_graph"] == 0
    assert res["static_recipe_check"]["declared"] is False          # still not a declared fallback: it is unused, not accepted


def test_never_called_site_the_bundle_does_not_declare_still_fails(tmp_path, good):
    vb = _script("verify_bundle")
    d, fx = _graft_unused(tmp_path / "undeclared", good, declare=False)
    res = vb.run(d, fx, tmp_path / "undeclared.json")
    # a site silently missing from the graph is the failure these checks exist for
    assert {"graph_qlinear_count_equals_quantized_linears", "static_recipe_fully_static"} <= set(res["failed"]), res["failed"]
    assert res["checks"]["graph_qlinear_count_equals_quantized_linears"]["detail"]["policy_sites_not_in_graph"] == ["unused_head"]


def test_declared_unused_site_that_is_actually_in_the_graph_fails(tmp_path, good):
    """Declaring a live site unused must not hide it: unused is derived from the lifted buffers, then compared."""
    vb = _script("verify_bundle")
    d, fx = good
    d2, fx2 = tmp_path / "export", tmp_path / "fx"
    d2.mkdir(parents=True)
    shutil.copytree(fx, fx2)
    shutil.copy(d / "policy.json", d2 / "policy.json")
    shutil.copy(d / "model.safetensors", d2 / "model.safetensors")
    pol = json.loads((d / "policy.json").read_text())
    live = next(n for n, s in pol["sites"].items() if s["kind"] == "linear" and s["status"] == "quantized")
    gi = json.loads((fx2 / "graph_inspection.json").read_text())
    gi["unused_quantized_modules"] = [{"site": live, "module": live}]
    (fx2 / "graph_inspection.json").write_text(json.dumps(gi))
    res = vb.run(d2, fx2, tmp_path / "v.json")
    assert "graph_qlinear_count_equals_quantized_linears" in res["failed"]


def test_attention_pair_counts_are_cross_checked_against_the_graph(tmp_path, good):
    vb = _script("verify_bundle")
    d, fx = good
    base = vb.run(d, fx, tmp_path / "base.json")
    assert base["checks"]["graph_qk_pv_counts_equal_attention_sites"]["ok"]
    assert base["graph_attention"]["scope_less_pairs"] == 0 and base["graph_attention"]["scoped_pairs"] >= 1
    # the bundle claims more pairs for a site than the graph holds
    fx2 = tmp_path / "fx"
    shutil.copytree(fx, fx2)
    gi = json.loads((fx2 / "graph_inspection.json").read_text())
    gi["native_fp8_attention_modules"][0]["pairs"] += 1
    (fx2 / "graph_inspection.json").write_text(json.dumps(gi))
    res = vb.run(d, fx2, tmp_path / "v.json")
    assert "graph_qk_pv_counts_equal_attention_sites" in res["failed"]
    # a policy attention site with no node in the graph (neither scoped nor declared scope-less) fails
    pol = json.loads((d / "policy.json").read_text())
    att = next(n for n, s in pol["sites"].items() if s["kind"] == "attention")
    pol["sites"]["ghost_attn"] = json.loads(json.dumps(pol["sites"][att]))
    d3 = tmp_path / "export3"
    d3.mkdir()
    (d3 / "policy.json").write_text(json.dumps(pol))
    (d3 / "model.safetensors").symlink_to(d / "model.safetensors")
    res3 = vb.run(d3, fx, tmp_path / "v3.json")
    assert "graph_qk_pv_counts_equal_attention_sites" in res3["failed"]


# -- the real SmolVLA bundle -------------------------------------------------------------------------------------------------

@pytest.mark.skipif(not (REAL / "fx" / "model.pt2").exists(), reason="SmolVLA bundle not present")
def test_real_smolvla_bundle_passes_with_derived_counts(tmp_path):
    vb = _script("verify_bundle")
    res = vb.run(REAL / "export", REAL / "fx", tmp_path / "v.json")
    assert res["ok"] and len(res["checks"]) == 25, res["failed"]
    g = res["graph"]["call_function_targets_atlas"]
    assert (g["atlas.qlinear.default"], g["atlas.qconv2d.default"], g["atlas.qk_matmul.default"],
            g["atlas.pv_matmul.default"], g["atlas.quantize_act.default"]) == (302, 1, 36, 36, 447)
    assert res["graph_inventory"]["unused_in_graph_derived"] == ["model.vlm_with_expert.vlm.lm_head"]
    ga = res["graph_attention"]
    assert (ga["policy_sites"], ga["scoped_pairs"], ga["scope_less_pairs"]) == (13, 12, 24)
    assert res["static_recipe_check"]["fallback_sites_live_in_graph"] == 0
    assert res["static_recipe_check"]["dynamic_rule_linear_conv_activations_in_graph"] == 0
