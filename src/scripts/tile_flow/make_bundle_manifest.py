"""Write <recipe_dir>/bundle_manifest.json: one file a consumer reads first (calibration, commit, recipe, rule, rows,
file digests, verification and quality pointers). Every value is read from the files below it, nothing is typed in.
usage: make_bundle_manifest.py RECIPE_DIR    (RECIPE_DIR has export/, fx/, verify_bundle.json, quality.json, ...)"""
import hashlib, json, sys
from pathlib import Path


def model_record(man):
    """{"id", "checkpoint"} of a bundle manifest. LM / policy exports record model.id; the Pi0 export records
    model.config (no id) and others a name: the first present of id / config / name, else null (never a crash)."""
    m = man.get("model") or {}
    return {"id": next((m[k] for k in ("id", "config", "name") if m.get(k)), None), "checkpoint": m.get("checkpoint")}


def main(recipe_dir):
    d = Path(recipe_dir)
    ex, fx = d / "export", d / "fx"
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest() if Path(p).stat().st_size < 3e9 else None  # noqa: E731
    man = json.loads((ex / "manifest.json").read_text())
    pol = json.loads((ex / "policy.json").read_text())
    fxm = json.loads((fx / "fx_manifest.json").read_text())
    ver = json.loads((d / "verify_bundle.json").read_text())
    gran = next(m for m in man["recipe"]["stack"] if m["name"] == "granularity")["params"]
    out = {
        "schema": "atlas_torchao.tile_bundle_manifest.v1 (descriptive, not an existing consumer schema)",
        "model": model_record(man),
        "recipe": {"algo": man["recipe"]["algo"], "expanded": man["recipe"]["expanded"], "granularity_params": gran,
                   "weight_rule": gran["rule"], "rows": gran["rows"],
                   "weights": "one exponent per group of 32 output channels over all K (per_panel(32))",
                   "activations": sorted({(s["activation"]["mode"], s["activation"].get("reduce") or s["activation"].get("source"))
                                          for s in pol["sites"].values() if s.get("kind") == "linear" and s.get("status") == "quantized"}),
                   "activations_note": "see policy.json sites[*].activation; attention operands in sites[*].operands"},
        "calibration": man["calibration"],
        "producing_commit": man["provenance"]["atlas_torchao_commit"],
        "producing_branch": man["provenance"]["atlas_torchao_branch"],
        "producing_tree_dirty": man["provenance"]["atlas_torchao_dirty"],
        "fx_capture_commit": fxm["provenance"]["capture_source"]["commit"],
        "fx_capture_tree_dirty_porcelain": fxm["provenance"]["capture_source"]["dirty"],
        "versions": man["provenance"]["versions"], "device": man["provenance"]["device"],
        "counts": man["counts"],
        "files": {"export/model.safetensors": man["files"]["model.safetensors"], "export/policy.json": man["files"]["policy.json"],
                  "export/manifest.json": {"sha256": sha(ex / "manifest.json"), "bytes": (ex / "manifest.json").stat().st_size},
                  **{f"fx/{n}": {"sha256": sha(fx / n), "bytes": (fx / n).stat().st_size}
                     for n in ("model.pt2", "graph_inspection.json", "buffer_inventory.json", "fx_manifest.json", "policy.json",
                               "graph.txt", "verification.json")}},
        "fx": {"seqlen": fxm["provenance"].get("seqlen"), "batch": 1, "native_fp8": fxm.get("native_fp8"),
               "graph_atlas_ops": ver["graph"]["call_function_targets_atlas"],
               "non_atlas_contractions": [(c["target"], c["why"]) for c in ver["graph"]["allowed_non_atlas_contractions"]]},
        "independent_verification": {"file": "verify_bundle.json", "ok": ver["ok"], "failed": ver["failed"],
                                     "checks_passed": sum(1 for c in ver["checks"].values() if c["ok"]), "checks": len(ver["checks"])},
        "tile_native": {k: pol["tile_native"][k] for k in ("sites_checked", "tile_native_sites", "all_tile_native")},
        "uncalibrated_fallback": pol.get("uncalibrated_fallback", "absent (export predates the field)"),
        "unquantized_attention": ver["graph"].get("unquantized_attention", "absent (verify_bundle predates the field)"),
        "compiler_pack_scale_admission": {"admitted_range": [122, 128],
            "status": pol["compiler_pack_scale_admission"].get(
                "status", "checked_static_sites" if pol["compiler_pack_scale_admission"].get("sites_checked") else
                "not_checked_dynamic (legacy export: sites_checked 0 means not checked)"),
            "policy_static_sites_checked": pol["compiler_pack_scale_admission"].get("sites_checked"),
            "policy_static_sites_outside": pol["compiler_pack_scale_admission"].get("sites_outside"),
            "see": ["policy.json compiler_pack_scale_admission", "fx/graph_inspection.json compiler_pack_scale_admission"]},
        "hardware_equivalence": "not claimed: numerics are the framework's exact float32 accumulation unless quality.json "
                                "accumulation_models says otherwise (sa / ipt isolated layers only); no RTL run",
    }
    if "pack_fold_summary" in pol:                      # src/pack_fold.py (docs/algos/pack-fold.md), opt-in
        ps = pol["pack_fold_summary"]
        out["pack_fold"] = {k: ps[k] for k in ("static_sites", "outside_before", "outside_after", "sites_folded", "folded_by_class",
                                               "outside_after_by_class", "outside_after_by_cost", "exact")}
        out["pack_fold"]["source_export"] = man.get("pack_fold", {}).get("source_export")
    if (d / "quality.json").exists():
        q = json.loads((d / "quality.json").read_text())
        out["quality"] = {"file": "quality.json", "final_logits": q["final_logits"], "batch": q["batch"]}
    if (d / "ppl_w16x512.json").exists():
        out.setdefault("quality", {})["ppl_file"] = "ppl_w16x512.json"
        out["quality"]["ppl"] = json.loads((d / "ppl_w16x512.json").read_text())["ppl"]
    if (d / "loop_manifest.json").exists():       # per-phase bundles: the host loop around this graph (make_loop_manifest.py)
        lm = json.loads((d / "loop_manifest.json").read_text())
        phase = fxm["provenance"].get("phase")
        graph = (lm.get("graphs") or {}).get(phase) or {}
        out["loop_manifest"] = {"file": "loop_manifest.json", "sha256": sha(d / "loop_manifest.json"), "model": lm["model"],
                                "denoising_steps": lm["denoising_steps"], "phase": phase,
                                "calls_per_action_chunk": graph.get("calls_per_action_chunk"),
                                "recorded_in_fx_manifest": fxm["provenance"].get("loop_manifest"),
                                "matches_fx_manifest": (fxm["provenance"].get("loop_manifest") or {}).get("sha256")
                                                       == sha(d / "loop_manifest.json")}
    (d / "bundle_manifest.json").write_text(json.dumps(out, indent=1))
    print("wrote", d / "bundle_manifest.json")


if __name__ == "__main__":
    main(sys.argv[1])
