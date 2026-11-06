"""Independent checks of a tile-aware export + native FP8 FX bundle. Reads the files directly (safetensors, policy.json,
model.pt2 graph); does not trust graph_inspection.json except where it is cross-checked.
usage: verify_bundle.py EXPORT_DIR FX_DIR OUT_JSON [--allow-uncalibrated-fallback] [--forbid-unquantized-attention]
       (run with PYTHONPATH=<worktree>/src)

Declared vs unexpected contractions (check no_unexpected_non_atlas_contraction). A contraction outside the atlas.*
ops is DECLARED, and reported in graph.allowed_non_atlas_contractions, when it is
  - a Linear / Conv2d whose weight is a policy site with status host or skipped (BF16 by policy; the lm_head and the
    input patch-embedding conv), or the legacy lm_head operand-name rule,
  - the float32 RoPE inv_freq x position outer product (no weight),
  - a plain BF16 scaled_dot_product_attention whose module is NOT in or around any policy attention site (the
    ALLOWED_PLAIN_OPS allow-list). These are counted in graph.unquantized_attention, never hidden.
Everything else (a BF16 matmul / mm / bmm / Linear with a quantized or unknown weight, an SDPA inside a policy
attention site, an SDPA without module info or not in BF16) is UNEXPECTED and fails the check."""
import argparse
import collections
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open

import atlas_fx_ops  # noqa: F401  registers atlas.* ops

CONTRACT = ("aten.linear", "aten.matmul", "aten.mm", "aten.bmm", "aten.addmm", "aten.baddbmm", "aten.einsum",
            "aten.conv", "aten._convolution", "aten.dot", "aten.mv", "aten.scaled_dot_product", "aten._scaled_dot")
WEIGHTED = ("aten.linear", "aten.conv", "aten._convolution")          # args[1] is a weight parameter
# explicit allow-list of plain (non-atlas) contractions accepted when no policy site covers them
ALLOWED_PLAIN_OPS = {"aten.scaled_dot_product_attention": "plain BF16 scaled_dot_product_attention in a block "
                                                          "without a policy attention site (unquantized attention)"}


def _module_candidates(module):
    """The module path as recorded (wrapper-prefixed) and with the wrapper's leading 'model.' removed."""
    return {module, module[len("model."):]} if module and module.startswith("model.") else {module}


def _overlaps(module, site):
    """True when `module` is `site`, inside it, or encloses it (a module path prefix relation)."""
    return any(m == site or m.startswith(site + ".") or site.startswith(m + ".") for m in _module_candidates(module))


def classify_contractions(contractions, policy, weight_targets, allow_unquantized_attention=True):
    """Split non-atlas contraction nodes into (allowed, unexpected).
    contractions: [{node, target, operands: [{name, dtype, shape}], module: innermost nn module path | None}]
    weight_targets: {placeholder name: parameter target 'model.<site>.weight'}.
    Allowed entries carry `why` (and `declared_as`: bf16_by_policy | rope_outer_product | unquantized_attention)."""
    host = {n for n, s in policy["sites"].items() if s.get("status") in ("host", "skipped")}
    attn_sites = [n for n, s in policy["sites"].items() if s.get("kind") == "attention"]
    allowed, unexpected = [], []
    for c in contractions:
        dts = {o["dtype"] for o in c["operands"] if o["dtype"] != "None"}
        tgt = c["target"]
        weight = c["operands"][1]["name"] if len(c["operands"]) > 1 else None
        ptarget = weight_targets.get(weight, "")
        site = ptarget[:-len(".weight")] if ptarget.endswith(".weight") else None
        site_host = next((m for m in _module_candidates(site) if m in host), None) if site else None
        if tgt.endswith("linear.default") and any("lm_head" in o["name"] for o in c["operands"]):
            allowed.append({**c, "why": "lm_head, bf16 by policy (skipped)", "declared_as": "bf16_by_policy"})
        elif tgt.endswith("matmul.default") and dts == {"torch.float32"} and \
                any(o["shape"][-1:] == [1] for o in c["operands"]):
            allowed.append({**c, "why": "float32 RoPE inv_freq[:,None] x position_ids[None,:] outer product, no weight",
                            "declared_as": "rope_outer_product"})
        elif any(tgt.startswith(w) for w in WEIGHTED) and site_host:
            allowed.append({**c, "why": f"{site_host}: bf16 by policy ({policy['sites'][site_host]['status']})",
                            "declared_as": "bf16_by_policy"})
        elif tgt.startswith("aten.scaled_dot_product_attention") and allow_unquantized_attention \
                and tgt.split(".default")[0] in ALLOWED_PLAIN_OPS and c.get("module") \
                and not any(_overlaps(c["module"], a) for a in attn_sites) \
                and {o["dtype"] for o in c["operands"][:3]} == {"torch.bfloat16"}:
            allowed.append({**c, "why": ALLOWED_PLAIN_OPS[tgt.split(".default")[0]] + f" [{c['module']}]",
                            "declared_as": "unquantized_attention"})
        else:
            unexpected.append(c)
    return allowed, unexpected


def _innermost_module(node):
    stack = node.meta.get("nn_module_stack") or {}
    return list(stack.values())[-1][0] if stack else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("export")
    ap.add_argument("fx")
    ap.add_argument("out")
    ap.add_argument("--allow-uncalibrated-fallback", action="store_true",
                    help="accept Linear sites of a static recipe that kept the dynamic rule (also accepted when the "
                         "export declared them)")
    ap.add_argument("--forbid-unquantized-attention", action="store_true",
                    help="treat plain BF16 attention outside policy sites as unexpected instead of declared")
    args = ap.parse_args(argv)
    export, fx, out = Path(args.export), Path(args.fx), Path(args.out)
    return run(export, fx, out, args.allow_uncalibrated_fallback, not args.forbid_unquantized_attention)


def run(export, fx, out, allow_fallback=False, allow_unquantized_attention=True):
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
    census_plain = collections.Counter()      # census without declared-fallback sites (owned by static_recipe_fully_static)
    for n, s in policy["sites"].items():
        if s.get("status") != "quantized":
            continue
        if s["kind"] == "attention":
            for op, r in s["operands"].items():
                census[("attention." + op, r.get("mode"), r.get("reduce") or r.get("source"), r.get("rule"))] += 1
        else:
            a = s["activation"]
            census[(s["kind"], a.get("mode"), a.get("reduce") or a.get("source"), a.get("rule"))] += 1
            if not a.get("uncalibrated_fallback"):
                census_plain[(s["kind"], a.get("mode"), a.get("reduce") or a.get("source"), a.get("rule"))] += 1
    res["policy_activation_census"] = [{"site_kind": k[0], "mode": k[1], "reduce_or_source": k[2], "rule": k[3], "count": v}
                                       for k, v in sorted(census.items(), key=str)]
    tn = policy["tile_native"]
    # static recipe: no Linear / Conv2d may stay on the dynamic reference rule unless the site is declared
    fb_rec = policy.get("uncalibrated_fallback")
    lin_quant = {n: s for n, s in policy["sites"].items() if s.get("status") == "quantized" and s["kind"] in ("linear", "conv2d")}
    static_recipe = bool(fb_rec["static_recipe"]) if fb_rec else any(
        s["activation"].get("mode") in ("static", "static_per_step") for s in lin_quant.values())
    if fb_rec:
        fb_sites = sorted(fb_rec["sites"])
    else:                                   # export predates the field: reference-rule Linear sites of a static recipe
        fb_sites = sorted(n for n, s in lin_quant.items() if s["activation"].get("source") == "reference") if static_recipe else []
    declared = bool(fb_rec and fb_rec.get("declared")) or allow_fallback
    res["static_recipe_check"] = {"static_recipe": static_recipe, "uncalibrated_fallback_count": len(fb_sites),
                                  "sites": fb_sites[:10], "declared": declared,
                                  "export_records_field": fb_rec is not None}
    check("static_recipe_fully_static", not static_recipe or not fb_sites or declared, res["static_recipe_check"])
    # declared fallback sites are the only not-tile-native sites a bundle may carry
    tile_exempt = set(fb_sites) if declared else set()
    check("policy_all_sites_tile_native",
          set(tn["not_tile_native"]) <= tile_exempt and tn["sites_checked"] == tn["tile_native_sites"] + len(tn["not_tile_native"]),
          {"sites_checked": tn["sites_checked"], "declared_fallback_exempt": len(set(tn["not_tile_native"]) & tile_exempt)})
    check("policy_reference_rule_sites_zero_for_linear",
          not any(k[0] in ("linear", "conv2d") and k[2] == "reference" for k in census_plain), None)

    # ---- 3. the graph itself -------------------------------------------------------------------------------------------
    prog = torch.export.load(fx / "model.pt2")
    targets = collections.Counter(str(n.target) for n in prog.graph.nodes if n.op == "call_function")
    by_name = {n.name: n for n in prog.graph.nodes}
    weight_targets = {sp.arg.name: sp.target for sp in prog.graph_signature.input_specs
                      if sp.kind.name in ("PARAMETER", "BUFFER") and hasattr(sp.arg, "name") and isinstance(sp.target, str)}
    non_atlas = []
    for n in prog.graph.nodes:
        if n.op != "call_function" or not any(str(n.target).startswith(f"{p}") or p in str(n.target) for p in CONTRACT):
            continue
        ops = []
        for a in n.args:
            v = getattr(a, "meta", {}).get("val") if hasattr(a, "meta") else None
            ops.append({"name": getattr(a, "name", str(a)), "dtype": str(getattr(v, "dtype", None)),
                        "shape": list(getattr(v, "shape", []))})
        non_atlas.append({"node": n.name, "target": str(n.target), "operands": ops, "module": _innermost_module(n)})
    res["graph"] = {"call_function_targets_atlas": {k: v for k, v in targets.items() if "atlas" in k},
                    "nodes_total": sum(targets.values()), "non_atlas_contractions": non_atlas}
    n_q = len(sites)
    n_attn = sum(1 for s in policy["sites"].values() if s["kind"] == "attention" and s["status"] == "quantized")
    cnt = lambda t: targets.get(f"atlas.{t}.default", 0)  # noqa: E731
    check("graph_qlinear_count_equals_quantized_linears", cnt("qlinear") == n_q, {"qlinear": cnt("qlinear"), "policy": n_q})
    check("graph_qk_pv_counts_equal_attention_sites", cnt("qk_matmul") == n_attn and cnt("pv_matmul") == n_attn,
          {"qk": cnt("qk_matmul"), "pv": cnt("pv_matmul"), "policy": n_attn})
    check("graph_quantize_act_count", cnt("quantize_act") == n_q + 4 * n_attn, cnt("quantize_act"))
    # every non-atlas contraction must be declared (see the module docstring) or the check fails
    allowed, unexpected = classify_contractions(non_atlas, policy, weight_targets, allow_unquantized_attention)
    res["graph"]["allowed_non_atlas_contractions"] = allowed
    unq = [c for c in allowed if c["declared_as"] == "unquantized_attention"]
    res["graph"]["unquantized_attention"] = {
        "count": len(unq), "policy_attention_sites": n_attn, "allow_list": sorted(ALLOWED_PLAIN_OPS),
        "modules": [c["module"] for c in unq],
        "note": "plain BF16 scaled_dot_product_attention in blocks without a policy attention site; "
                "these attentions are NOT FP8-quantized" if unq else "none"}
    res["graph"]["declared_bf16_by_policy_contractions"] = sum(1 for c in allowed if c["declared_as"] == "bf16_by_policy")
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
    fresh = json.loads((fx / "fresh_process_verification.json").read_text())
    placeholder_of = {sp.target: sp.arg.name for sp in prog.graph_signature.input_specs
                      if sp.kind.name == "PARAMETER" and hasattr(sp.arg, "name")}
    if fl and fresh.get("unused_float_weights_not_materialized") is True:
        # non-strict export keeps unused quantized layers' weights as meta placeholders: tolerated only when each is
        # meta (no payload) and no node consumes it
        sdict = prog.state_dict
        fl_bad = [k for k in fl if k not in sdict or not sdict[k].is_meta
                  or k not in placeholder_of or by_name[placeholder_of[k]].users]
        res["graph"]["unused_meta_placeholder_float_weights"] = len(fl) - len(fl_bad)
    else:
        fl_bad = fl
    check("graph_has_no_float_weight_param_for_quantized_linears", not fl_bad, fl_bad[:3])

    # ---- 4. graph_inspection + reports ------------------------------------------------------------------------------------
    gi = json.loads((fx / "graph_inspection.json").read_text())
    t = gi["tile_native"]
    check("inspection_all_tile_native",
          set(t["not_tile_native"]) <= tile_exempt and t["sites_checked"] == t["tile_native_sites"] + len(t["not_tile_native"]),
          {"sites_checked": t["sites_checked"], "declared_fallback_exempt": len(set(t["not_tile_native"]) & tile_exempt)})
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
    dyn_adm = adm.get("dynamic_activation_scales")
    res["pack_admission_static_and_fixture"] = {
        "status": adm.get("status", "legacy export: no status field" if "status" not in adm else None),
        "static_sites_checked": adm.get("sites_checked"), "static_sites_outside": adm.get("sites_outside"),
        "dynamic_fixture": {k: v for k, v in dyn_adm.items() if k != "codes"} if isinstance(dyn_adm, dict) else dyn_adm}
    ver = json.loads((fx / "verification.json").read_text())
    check("capture_verification_passed", ver["status"] == "passed", ver["status"])
    check("fresh_process_verification_passed", fresh.get("status") == "passed", fresh.get("status"))
    res["fresh_process"] = {k: fresh.get(k) for k in ("status", "atlas_qlinear_nodes", "atlas_qk_matmul_nodes", "atlas_pv_matmul_nodes",
                                                      "model_replay_modules_imported")}
    res["ok"] = not fails
    res["failed"] = fails
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"ok": res["ok"], "failed": fails, "weights": res["weights"], "graph_atlas": res["graph"]["call_function_targets_atlas"],
                      "unquantized_attention": res["graph"]["unquantized_attention"]["count"],
                      "uncalibrated_fallback": len(fb_sites)}, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(0 if main()["ok"] else 1)
