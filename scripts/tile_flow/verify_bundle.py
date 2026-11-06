"""Independent checks of a tile-aware export + native FP8 FX bundle. Reads the files directly (safetensors, policy.json,
model.pt2 graph); does not trust graph_inspection.json except where it is cross-checked.
usage: verify_bundle.py EXPORT_DIR FX_DIR OUT_JSON   (run with PYTHONPATH=<worktree>/src)"""
import collections
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import atlas_fx_ops  # noqa: F401  registers atlas.* ops

export, fx, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
policy = json.loads((export / "policy.json").read_text())
res = {"export_dir": str(export), "fx_dir": str(fx), "checks": {}}
fails = []


def check(name, ok, detail=None):
    res["checks"][name] = {"ok": bool(ok), **({"detail": detail} if detail is not None else {})}
    if not ok:
        fails.append(name)


# ---- 1. weight tensors straight from safetensors ---------------------------------------------------------------
sites = {n: s for n, s in policy["sites"].items() if s.get("status") == "quantized" and s.get("kind") in ("linear", "conv2d")}
bad = collections.defaultdict(list)
distinct_per_site = {}
groups_per_site = {}
n_codes = n_reserved = n_exp15 = 0
with safe_open(str(export / "model.safetensors"), "pt") as f:
    for n, s in sites.items():
        w = s["weight"]
        c, e = f.get_tensor(w["codes"]), f.get_tensor(w["scale_exp"])
        if c.dtype != torch.uint8:
            bad["codes_dtype"].append(n)
        if e.dtype != torch.int32:
            bad["exp_dtype"].append(n)
        out_f = c.shape[0]
        if tuple(e.shape) != (out_f, 1):
            bad["exp_shape"].append(n)
            continue
        ci = c.to(torch.int32)
        n_codes += ci.numel()
        n_reserved += int(((ci == 0x7F) | (ci == 0xFF)).sum())
        n_exp15 += int((((ci >> 3) & 0xF) == 15).sum())
        ev = e.reshape(-1)
        ng = -(-out_f // 32)
        groups_ok = all(int(ev[g * 32:(g + 1) * 32].min()) == int(ev[g * 32:(g + 1) * 32].max()) for g in range(ng))
        if not groups_ok:
            bad["not_constant_per_32_group"].append(n)
        distinct_per_site[n] = int(ev.unique().numel())
        groups_per_site[n] = ng
        if w.get("granularity", "").startswith("per_panel") is False:
            bad["granularity_label"].append(n)
    names = set(f.keys())
    float_weights = [n for n in sites if f"{n}.weight" in names]
check("weight_codes_uint8", not bad["codes_dtype"], bad["codes_dtype"][:3])
check("weight_scale_exp_int32", not bad["exp_dtype"], bad["exp_dtype"][:3])
check("weight_scale_exp_shape_out_by_1", not bad["exp_shape"], bad["exp_shape"][:3])
check("weight_scale_exp_constant_per_32_channel_group", not bad["not_constant_per_32_group"], bad["not_constant_per_32_group"][:3])
check("weight_granularity_label_per_panel", not bad["granularity_label"], bad["granularity_label"][:3])
check("no_0x7F_0xFF_in_weight_codes", n_reserved == 0, n_reserved)
check("no_float_weight_stored_for_quantized_layers", not float_weights, float_weights[:3])
multi = sum(1 for n in sites if groups_per_site.get(n, 1) > 1 and distinct_per_site.get(n, 1) > 1)
multi_possible = sum(1 for n in sites if groups_per_site.get(n, 1) > 1)
res["weights"] = {"quantized_sites": len(sites), "codes_total": n_codes, "reserved_codes": n_reserved,
                  "exp15_code_fraction": n_exp15 / max(n_codes, 1),
                  "sites_with_more_than_one_group": multi_possible,
                  "of_which_with_more_than_one_distinct_exponent": multi,
                  "exponents_stored_per_site_sum": sum(int(policy["sites"][n]["weight"]["scale_exp_shape"][0]) for n in sites),
                  "distinct_exponents_needed_sum": sum(distinct_per_site.values()),
                  "panel_groups_sum": sum(groups_per_site.values())}
check("panel_not_collapsed_to_per_tensor", multi_possible == 0 or multi > 0, res["weights"])

# ---- 2. policy activation census ---------------------------------------------------------------------------------
census = collections.Counter()
for n, s in policy["sites"].items():
    if s.get("status") != "quantized":
        continue
    if s["kind"] == "attention":
        for op, r in s["operands"].items():
            census[("attention." + op, r.get("mode"), r.get("reduce") or r.get("source"), r.get("rule"))] += 1
    else:
        a = s["activation"]
        census[(s["kind"], a.get("mode"), a.get("reduce") or a.get("source"), a.get("rule"))] += 1
res["policy_activation_census"] = [{"site_kind": k[0], "mode": k[1], "reduce_or_source": k[2], "rule": k[3], "count": v}
                                   for k, v in sorted(census.items(), key=str)]
tn = policy["tile_native"]
check("policy_all_sites_tile_native", tn["all_tile_native"] and tn["sites_checked"] == tn["tile_native_sites"], tn["sites_checked"])
check("policy_reference_rule_sites_zero_for_linear",
      not any(k[0] in ("linear", "conv2d") and k[2] == "reference" for k in census), None)

# ---- 3. the graph itself -------------------------------------------------------------------------------------------
prog = torch.export.load(fx / "model.pt2")
targets = collections.Counter(str(n.target) for n in prog.graph.nodes if n.op == "call_function")
CONTRACT = ("aten.linear", "aten.matmul", "aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.einsum",
            "aten.conv", "aten._convolution", "aten.dot", "aten.mv", "aten.scaled_dot_product", "aten._scaled_dot")
non_atlas = []
for n in prog.graph.nodes:
    if n.op != "call_function" or not any(str(n.target).startswith(f"{p}") or p in str(n.target) for p in CONTRACT):
        continue
    ops = []
    for a in n.args:
        v = getattr(a, "meta", {}).get("val") if hasattr(a, "meta") else None
        ops.append({"name": getattr(a, "name", str(a)), "dtype": str(getattr(v, "dtype", None)),
                    "shape": list(getattr(v, "shape", []))})
    non_atlas.append({"node": n.name, "target": str(n.target), "operands": ops})
res["graph"] = {"call_function_targets_atlas": {k: v for k, v in targets.items() if "atlas" in k},
                "nodes_total": sum(targets.values()), "non_atlas_contractions": non_atlas}
n_q = len(sites)
n_attn = sum(1 for s in policy["sites"].values() if s["kind"] == "attention" and s["status"] == "quantized")
cnt = lambda t: targets.get(f"atlas.{t}.default", 0)  # noqa: E731
check("graph_qlinear_count_equals_quantized_linears", cnt("qlinear") == n_q, {"qlinear": cnt("qlinear"), "policy": n_q})
check("graph_qk_pv_counts_equal_attention_sites", cnt("qk_matmul") == n_attn and cnt("pv_matmul") == n_attn,
      {"qk": cnt("qk_matmul"), "pv": cnt("pv_matmul"), "policy": n_attn})
check("graph_quantize_act_count", cnt("quantize_act") == n_q + 4 * n_attn, cnt("quantize_act"))
# every non-atlas contraction must be lm_head (bf16 by policy) or the fp32 RoPE inv_freq x position matmul
allowed = []
unexpected = []
for c in non_atlas:
    dts = {o["dtype"] for o in c["operands"] if o["dtype"] != "None"}
    if c["target"].endswith("linear.default") and any("lm_head" in o["name"] for o in c["operands"]):
        allowed.append({**c, "why": "lm_head, bf16 by policy (skipped)"})
    elif c["target"].endswith("matmul.default") and dts == {"torch.float32"} and \
            any(o["shape"][-1:] == [1] for o in c["operands"]):
        allowed.append({**c, "why": "float32 RoPE inv_freq[:,None] x position_ids[None,:] outer product, no weight"})
    else:
        unexpected.append(c)
res["graph"]["allowed_non_atlas_contractions"] = allowed
check("no_unexpected_non_atlas_contraction", not unexpected, unexpected)
bf16_policy = [s for s, v in policy["sites"].items() if v.get("status") in ("host", "skipped")]
res["graph"]["bf16_by_policy_sites"] = bf16_policy

# buffers retained as uint8 / int32 in the loaded program
bufs = dict(prog.named_buffers())
badbuf = [k for k in bufs if k.endswith("weight_codes") and bufs[k].dtype != torch.uint8] + \
         [k for k in bufs if k.endswith("weight_scale_exp") and bufs[k].dtype != torch.int32]
nb = sum(1 for k in bufs if k.endswith("weight_codes"))
check("graph_weight_buffers_uint8_int32", not badbuf and nb == n_q, {"weight_code_buffers": nb, "bad": badbuf[:3]})
qmods = {"model." + n for n in sites}
fl = [k for k in dict(prog.named_parameters()) if k.endswith(".weight") and k[:-7] in qmods]
check("graph_has_no_float_weight_param_for_quantized_linears", not fl, fl[:3])

# ---- 4. graph_inspection + reports ------------------------------------------------------------------------------------
gi = json.loads((fx / "graph_inspection.json").read_text())
t = gi["tile_native"]
check("inspection_all_tile_native", t["all_tile_native"] and t["sites_checked"] == t["tile_native_sites"] and not t["not_tile_native"],
      {"sites_checked": t["sites_checked"]})
rows = gi["weight_site_bindings"]
check("inspection_all_weight_sites_native", len(rows) == n_q and all(r["native"] for r in rows), len(rows))
check("inspection_nothing_emulated", gi["still_emulated"]["emulated_modules"] == {}, gi["still_emulated"]["emulated_modules"])
check("inspection_bf16_by_policy_is_declared_only",
      [r["site"] for r in gi["bf16_by_policy"]] == bf16_policy, [r["site"] for r in gi["bf16_by_policy"]])
check("inspection_native_attention_modules", len(gi["native_fp8_attention_modules"]) == n_attn, len(gi["native_fp8_attention_modules"]))
fx_stats = t["fixture_activation_stats"]
res["fixtures"] = {"sites": len(fx_stats),
                   "reserved_codes_total": sum(v.get("reserved_codes", 0) for v in fx_stats.values()),
                   "all_groups_constant": all(v.get("groups_constant", True) for v in fx_stats.values()),
                   "token_split_all_32_aligned": [k for k, v in fx_stats.items() if v.get("token_split_all_32_aligned") is False][:3]}
check("fixtures_no_reserved_codes_and_groups_constant",
      res["fixtures"]["reserved_codes_total"] == 0 and res["fixtures"]["all_groups_constant"], res["fixtures"])
adm = gi["compiler_pack_scale_admission"]
res["pack_admission_static_and_fixture"] = {"static_sites_checked": adm["sites_checked"], "static_sites_outside": adm["sites_outside"],
                                            "dynamic_fixture": {k: v for k, v in adm["dynamic_activation_scales"].items() if k != "codes"}}
ver = json.loads((fx / "verification.json").read_text())
fresh = json.loads((fx / "fresh_process_verification.json").read_text())
check("capture_verification_passed", ver["status"] == "passed", ver["status"])
check("fresh_process_verification_passed", fresh.get("status") == "passed", fresh.get("status"))
res["fresh_process"] = {k: fresh.get(k) for k in ("status", "atlas_qlinear_nodes", "atlas_qk_matmul_nodes", "atlas_pv_matmul_nodes",
                                                  "model_replay_modules_imported")}
res["ok"] = not fails
res["failed"] = fails
out.write_text(json.dumps(res, indent=1))
print(json.dumps({"ok": res["ok"], "failed": fails, "weights": res["weights"], "graph_atlas": res["graph"]["call_function_targets_atlas"]}, indent=1))
sys.exit(0 if res["ok"] else 1)
