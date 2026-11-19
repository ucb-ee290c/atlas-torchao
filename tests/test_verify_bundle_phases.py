"""verify_bundle.py on the DiT per-phase bundles (encode, step): attention sites the graph does not hold are derived from the graph
(no quantize_act tagged '<site>:<operand>') and must equal the declared graph_inspection.unused_attention_modules; the qlinear node count
per site is derived from the graph and must equal the declared `calls`. The positive cases run the real bundles (skipped when absent),
the negative cases tamper a copy of one bundle's inspection and must fail the matching check
(docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest)."""
import json
import os
import shutil
from pathlib import Path

import pytest

from test_tile_flow_defects import _script

PHASES = Path("/scratch2/lorenhung/atlas-fp8/exports-tile/multitask-dit/atlas_v1_awq_tile_static_attn_phases")
needs_phases = pytest.mark.skipif(not (PHASES / "encode" / "fx" / "model.pt2").exists() and not (PHASES / "step" / "fx" / "model.pt2").exists(),
                                reason="needs the DiT phase bundles")


def _tampered(tmp_path, phase, edit):
    """A copy of <phase>/fx with graph_inspection.json edited by `edit(dict)`; big files are symlinks."""
    src = PHASES / phase / "fx"
    dst = tmp_path / phase / "fx"
    dst.mkdir(parents=True)
    for f in src.iterdir():
        if f.name == "graph_inspection.json":
            gi = json.loads(f.read_text())
            edit(gi)
            (dst / f.name).write_text(json.dumps(gi))
        elif f.stat().st_size > 5_000_000:
            os.symlink(f, dst / f.name)
        else:
            shutil.copy(f, dst / f.name)
    return dst


def _run(vb, phase, fx, out):
    return vb.run(PHASES / phase / "export", fx, out)


@pytest.fixture(scope="module")
def vb():
    return _script("verify_bundle")


@needs_phases
@pytest.mark.parametrize("phase,absent_prefix", [("encode", "noise_predictor"), ("step", "observation_encoder")])
def test_phase_bundle_verifies_with_absent_sites_derived_from_the_graph(vb, tmp_path, phase, absent_prefix):
    if not (PHASES / phase / "fx" / "model.pt2").exists():
        pytest.skip(f"no {phase} bundle")
    res = _run(vb, phase, PHASES / phase / "fx", tmp_path / "v.json")
    assert res["ok"], res["failed"]
    ga = res["graph_attention"]
    assert ga["absent_sites_derived"] == ga["absent_sites_declared"] and ga["absent_sites_derived"]
    assert all(absent_prefix in s for s in ga["absent_sites_derived"])
    assert ga["sites_in_graph"] + len(ga["absent_sites_derived"]) == ga["policy_sites"] == 30
    assert res["checks"]["graph_qk_pv_counts_equal_attention_sites"]["ok"]


@needs_phases
def test_declaring_a_present_site_as_absent_fails(vb, tmp_path):
    phase = "step" if (PHASES / "step" / "fx" / "model.pt2").exists() else "encode"
    pres = json.loads((PHASES / phase / "fx" / "graph_inspection.json").read_text())["native_fp8_attention_modules"][0]

    def edit(gi):
        gi["unused_attention_modules"].append({"site": pres["site"], "module": pres["module"]})
    res = _run(vb, phase, _tampered(tmp_path, phase, edit), tmp_path / "v.json")
    assert not res["ok"] and "graph_qk_pv_counts_equal_attention_sites" in res["failed"]


@needs_phases
def test_forgetting_to_declare_an_absent_site_fails(vb, tmp_path):
    phase = "step" if (PHASES / "step" / "fx" / "model.pt2").exists() else "encode"

    def edit(gi):
        gi["unused_attention_modules"] = gi["unused_attention_modules"][1:]
    res = _run(vb, phase, _tampered(tmp_path, phase, edit), tmp_path / "v.json")
    assert not res["ok"] and "graph_qk_pv_counts_equal_attention_sites" in res["failed"]


@needs_phases
def test_a_declared_call_count_that_disagrees_with_the_graph_fails(vb, tmp_path):
    phase = "step" if (PHASES / "step" / "fx" / "model.pt2").exists() else "encode"

    def edit(gi):
        gi["weight_site_bindings"][0]["calls"] = 2
    res = _run(vb, phase, _tampered(tmp_path, phase, edit), tmp_path / "v.json")
    assert not res["ok"] and "graph_qlinear_count_equals_quantized_linears" in res["failed"]
    assert res["checks"]["graph_qlinear_count_equals_quantized_linears"]["detail"]["sites_called_more_than_once"] == {}


# -- hoisted layout (encode, modulate, step) ------------------------------------------------------------------------------
HOISTED = Path(os.environ.get("ATLAS_DIT_HOISTED_PHASES",
                              "/scratch2/lorenhung/atlas-fp8/exports-tile/multitask-dit/atlas_v1_awq_tile_static_hoisted_phases"))


@pytest.mark.parametrize("phase", ["modulate", "step"])
def test_hoisted_phase_bundle_verifies(vb, tmp_path, phase):
    """modulate holds only time_mlp and the adaLN Linears (each called calls_per_site times) and no attention site; the
    hoisted step holds the rest of the noise predictor, each once."""
    if not (HOISTED / phase / "fx" / "model.pt2").exists():
        pytest.skip(f"no hoisted {phase} bundle")
    lm = json.loads((HOISTED / "loop_manifest.json").read_text())
    res = vb.run(HOISTED / phase / "export", HOISTED / phase / "fx", tmp_path / "v.json")
    assert res["ok"], res["failed"]
    gi = json.loads((HOISTED / phase / "fx" / "graph_inspection.json").read_text())
    calls = {r["site"]: r["calls"] for r in gi["weight_site_bindings"]}
    hoisted = set(lm["graphs"]["modulate"]["hoisted_sites"])
    if phase == "modulate":
        assert set(calls) == hoisted and set(calls.values()) == {lm["graphs"]["modulate"]["calls_per_site"]}
        assert res["graph_attention"]["sites_in_graph"] == 0
    else:
        assert not set(calls) & hoisted and set(calls.values()) == {1}
        assert all(s.startswith("noise_predictor.") for s in calls)
