"""Independent checks of a tile-aware export + native FP8 FX bundle. Reads the files directly (safetensors, policy.json,
model.pt2 graph); does not trust graph_inspection.json except where it is cross-checked.
usage: verify_bundle.py EXPORT_DIR FX_DIR OUT_JSON [--allow-uncalibrated-fallback] [--forbid-unquantized-attention]
                        [--require-pack-range]
       (run with PYTHONPATH=<worktree>/src)

Declared vs unexpected contractions (check no_unexpected_non_atlas_contraction). A contraction outside the atlas.*
ops is DECLARED, and reported in graph.allowed_non_atlas_contractions, when it is
  - a Linear / Conv2d whose weight is a policy site with status host or skipped (BF16 by policy; the lm_head and the
    input patch-embedding conv), or the legacy lm_head operand-name rule,
  - the float32 RoPE inv_freq x position outer product (no weight),
  - a plain BF16 scaled_dot_product_attention whose module is NOT in or around any policy attention site (the
    ALLOWED_PLAIN_OPS allow-list). These are counted in graph.unquantized_attention, never hidden.
Pack plan (informational): `pack_plan` in the result recomputes the per-site pack-scale plan from policy.json and
fx/graph.txt (src/pack_plan.py, docs/algos/pack-plan.md) and compares it with the recorded `pack_plan_summary`; it never fails
a check. With --require-pack-range the check `pack_range_plan_present` fails when a static activation site is outside the
admitted E8M0 range 122..128 and its recorded plan (fx/policy.json, else export/policy.json) is missing or infeasible.
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
import pack_plan

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


def _scope_candidates(module):
    """Module-path spellings of a graph scope: as recorded, minus a leading wrapper attribute 'policy.', minus a leading
    wrapper 'model.' (the wrapper name differs between bundles: 'model' for LMs / DiT / Pi0, 'policy' for SmolVLA)."""
    if not module:
        return set()
    out, pending = set(), [module]
    while pending:
        m = pending.pop()
        if m in out:
            continue
        out.add(m)
        for p in ("policy.", "model."):
            if m.startswith(p):
                pending.append(m[len(p):])
    return out


def _owner_site(module, sites):
    """Longest policy site that is `module` or encloses it; None when the node carries no scope (a module called as
    `module.forward(...)` is not on the nn_module_stack) or no site covers it."""
    best = None
    for m in _scope_candidates(module):
        for s in sites:
            if (m == s or m.startswith(s + ".")) and (best is None or len(s) > len(best)):
                best = s
    return best


def _wrapper_stems(names, suffix):
    """{stem with and without its first path component} -> original name, for buffers 'x.<site>.<suffix>'."""
    out = {}
    for k in names:
        if k.endswith("." + suffix):
            stem = k[:-len(suffix) - 1]
            out[stem] = k
            if "." in stem:
                out[stem.split(".", 1)[1]] = k
    return out


def _act_consumers(graph):
    """{quantize_act node name: consumer class} through the getitem (and the visible V^T transpose): qlinear | qconv2d |
    qk_matmul | pv_matmul | other. Used to check that every quantize_act feeds exactly one atlas contraction."""
    out = {}
    for n in graph.nodes:
        if n.op != "call_function" or str(n.target) != "atlas.quantize_act.default":
            continue
        kinds, pending = set(), [u for g in n.users for u in g.users]
        while pending:
            u = pending.pop()
            t = str(u.target)
            if t == "aten.transpose.int":
                pending.extend(u.users)
            else:
                kinds.add(t.replace("atlas.", "").replace(".default", "") if t.startswith("atlas.") else "other")
        out[n.name] = kinds.pop() if len(kinds) == 1 else "other"
    return out


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
    ap.add_argument("--require-pack-range", action="store_true",
                    help="fail when a static activation site is outside the compiler's admitted pack range (E8M0 122..128) "
                         "and has no recorded pack_plan (default: the pack plan is informational only)")
    args = ap.parse_args(argv)
    export, fx, out = Path(args.export), Path(args.fx), Path(args.out)
    return run(export, fx, out, args.allow_uncalibrated_fallback, not args.forbid_unquantized_attention,
               args.require_pack_range)


def pack_plan_report(policy, export, fx):
    """INFORMATIONAL pack plan: recomputed from the export policy and the FX graph text, next to what the bundle recorded.
    Returns (report, recorded plan {site key: record or None})."""
    g, s = fx / "graph.txt", fx / "graph_signature.txt"
    graph = g.read_text() if g.exists() else None
    plan = pack_plan.plan_from_files(policy, graph, s.read_text() if s.exists() else None)
    recomputed = pack_plan.summarize(plan, policy)
    fx_pol = json.loads((fx / "policy.json").read_text()) if (fx / "policy.json").exists() else {}
    gi = json.loads((fx / "graph_inspection.json").read_text()) if (fx / "graph_inspection.json").exists() else {}
    recorded_fx = pack_plan.recorded_plan(fx_pol) if fx_pol else {}
    recorded_ex = pack_plan.recorded_plan(policy)
    # the plan a compiler owner consumes: fx/policy.json (producer classes from the graph), else the export's own
    recorded = {k: recorded_fx.get(k) or recorded_ex.get(k) for k in plan}
    keys = ("sites_outside", "outside_by_class", "outside_by_cost", "free_fold_sites")
    rec_sum = {"export_policy": policy.get("pack_plan_summary"), "fx_policy": fx_pol.get("pack_plan_summary"),
               "graph_inspection": gi.get("pack_plan_summary")}
    return {"informational": True, "graph_available": graph is not None, "recomputed": recomputed,
            "recorded_summaries_present": {k: v is not None for k, v in rec_sum.items()},
            "recorded_matches_recomputed": {k: (None if v is None else all(v.get(f) == recomputed.get(f) for f in keys))
                                            for k, v in rec_sum.items()},
            "sites_with_recorded_plan": sum(1 for r in recorded.values() if r is not None)}, recorded


def pack_fold_report(policy, plan_report):
    """INFORMATIONAL (never a failing check): the pack fold recorded in the export policy (src/pack_fold.py) next to the sites
    that are STILL outside the admitted range according to the recomputed plan, by producer class and plan cost."""
    rec = plan_report["recomputed"]
    out = {"informational": True, "folded": "pack_fold_summary" in policy,
           "sites_still_out_of_range": rec.get("sites_outside"), "static_sites": rec.get("static_sites"),
           "still_out_of_range_by_class": rec.get("outside_by_class"), "still_out_of_range_by_cost": rec.get("outside_by_cost"),
           "admitted_range": rec.get("admitted_range")}
    s = policy.get("pack_fold_summary")
    if s:
        # pack_plan does not know the fold guards: a site still outside whose plan says cost "free" was REFUSED (pack_fold.reason), its free
        # placement is not valid (shared attention site, dynamic consumer, ...); the compiler's vmul fallback applies
        out["planned_free_but_refused"] = (rec.get("outside_by_cost") or {}).get("free", 0)
        out.update({"sites_folded": s["sites_folded"], "outside_before_fold": s["outside_before"],
                    "outside_after_fold_recorded": s["outside_after"], "folded_by_class": s["folded_by_class"],
                    "recorded_matches_recomputed": s["outside_after"] == rec.get("sites_outside"),
                    "pv_pop_shift_sites": sorted(n for n, x in policy["sites"].items() if x.get("pv_pop_shift"))})
    return out


def run(export, fx, out, allow_fallback=False, allow_unquantized_attention=True, require_pack_range=False):
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

    # ---- 1b. graph facts: which policy sites the graph really holds -------------------------------------------------
    # A policy site is not always one graph node: a quantized layer the captured boundary never calls has no node and no
    # lifted buffers (SmolVLA lm_head), and a module called several times (or as module.forward(), SmolVLA's text/expert
    # attention) has one policy site but several qk/pv pairs. Expected counts are therefore derived from the graph's own
    # inventory (lifted buffers, scopes, operand wiring) and cross-checked with the bundle's declarations.
    prog = torch.export.load(fx / "model.pt2")
    gi = json.loads((fx / "graph_inspection.json").read_text())
    bufs = dict(prog.named_buffers())
    code_stems = _wrapper_stems(bufs, "weight_codes")
    exp_stems = _wrapper_stems(bufs, "weight_scale_exp")
    in_graph = {n for n in sites if n in code_stems}
    unused_derived = sorted(set(sites) - in_graph)
    unused_declared = sorted(r["site"] for r in gi.get("unused_quantized_modules", []))
    unused_agree = unused_derived == unused_declared
    used = {n: s for n, s in sites.items() if n in in_graph}
    n_lin_used = sum(1 for s in used.values() if s["kind"] == "linear")
    n_conv_used = sum(1 for s in used.values() if s["kind"] == "conv2d")
    res["graph_inventory"] = {"policy_quantized_linear": sum(1 for s in sites.values() if s["kind"] == "linear"),
                              "policy_quantized_conv2d": sum(1 for s in sites.values() if s["kind"] == "conv2d"),
                              "sites_in_graph_by_buffers": len(in_graph), "unused_in_graph_derived": unused_derived,
                              "unused_declared_by_bundle": unused_declared, "derived_equals_declared": unused_agree}

    attn_policy = sorted(n for n, s in policy["sites"].items() if s["kind"] == "attention" and s["status"] == "quantized")
    qk_nodes = [n for n in prog.graph.nodes if n.op == "call_function" and str(n.target) == "atlas.qk_matmul.default"]
    pv_nodes = [n for n in prog.graph.nodes if n.op == "call_function" and str(n.target) == "atlas.pv_matmul.default"]
    scoped = {k: collections.Counter() for k in ("qk", "pv")}
    unscoped = collections.Counter()
    for kind, nodes in (("qk", qk_nodes), ("pv", pv_nodes)):
        for n in nodes:
            owner = _owner_site(_innermost_module(n), attn_policy)
            if owner is None:
                unscoped[kind] += 1
            else:
                scoped[kind][owner] += 1
    # sites that own no scoped node are called as module.forward(): their pairs are the scope-less ones, and their number is
    # the exporter's tag-bound count (graph_inspection native_fp8_attention_modules), which must equal the scope-less nodes
    scopeless_sites = [a for a in attn_policy if scoped["qk"][a] == 0 and scoped["pv"][a] == 0]
    rec_pairs = {r["site"]: r["pairs"] for r in gi.get("native_fp8_attention_modules", [])}
    scopeless_expected = sum(rec_pairs.get(a, 0) for a in scopeless_sites)
    per_site_ok = all(scoped["qk"][a] == scoped["pv"][a] and (scoped["qk"][a] == rec_pairs.get(a, scoped["qk"][a]))
                      for a in attn_policy if a not in scopeless_sites)
    attn_ok = (per_site_ok and unscoped["qk"] == unscoped["pv"] == scopeless_expected
               and all(rec_pairs.get(a, 0) >= 1 for a in scopeless_sites)
               and (bool(scopeless_sites) or unscoped["qk"] == 0))
    pairs_expected = sum(scoped["qk"].values()) + scopeless_expected
    res["graph_attention"] = {"policy_sites": len(attn_policy), "scoped_pairs": sum(scoped["qk"].values()),
                              "scope_less_pairs": unscoped["qk"], "scope_less_sites": scopeless_sites,
                              "scope_less_pairs_declared": scopeless_expected,
                              "pairs_by_site": {a: (scoped["qk"][a] or rec_pairs.get(a, 0)) for a in attn_policy}}
    consumers = _act_consumers(prog.graph)
    cons_count = collections.Counter(consumers.values())
    # a Linear / Conv2d activation on the dynamic rule is visible in the graph as quantize_act mode 'dynamic' feeding qlinear / qconv2d
    dyn_linear_acts = sum(1 for n in prog.graph.nodes if n.op == "call_function" and str(n.target) == "atlas.quantize_act.default"
                          and n.args[1] == "dynamic" and consumers.get(n.name) in ("qlinear", "qconv2d"))

    # ---- 2. policy activation census ---------------------------------------------------------------------------------
    census = collections.Counter()
    census_plain = collections.Counter()      # census without declared-fallback sites (owned by static_recipe_fully_static)
    # a fallback site the graph never calls (derived from the lifted buffers AND declared by the bundle) runs no rule at all
    def _unused_fallback(names):
        return {n for n in names if n in unused_derived and unused_agree}
    for n, s in policy["sites"].items():
        if s.get("status") != "quantized":
            continue
        if s["kind"] == "attention":
            for op, r in s["operands"].items():
                census[("attention." + op, r.get("mode"), r.get("reduce") or r.get("source"), r.get("rule"))] += 1
        else:
            a = s["activation"]
            census[(s["kind"], a.get("mode"), a.get("reduce") or a.get("source"), a.get("rule"))] += 1
            if not a.get("uncalibrated_fallback") and not (n in unused_derived and unused_agree and a.get("source") == "reference"):
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
    exempt_unused = sorted(_unused_fallback(fb_sites))
    live_fb = [n for n in fb_sites if n not in exempt_unused]
    res["static_recipe_check"] = {"static_recipe": static_recipe, "uncalibrated_fallback_count": len(fb_sites),
                                  "sites": fb_sites[:10], "declared": declared,
                                  "export_records_field": fb_rec is not None,
                                  "fallback_sites_never_called_in_graph": exempt_unused,
                                  "fallback_sites_live_in_graph": len(live_fb),
                                  "dynamic_rule_linear_conv_activations_in_graph": dyn_linear_acts}
    check("static_recipe_fully_static",
          not static_recipe or declared or (not live_fb and dyn_linear_acts == 0), res["static_recipe_check"])
    # declared fallback sites are the only not-tile-native sites a bundle may carry
    tile_exempt = (set(fb_sites) if declared else set()) | set(exempt_unused)
    check("policy_all_sites_tile_native",
          set(tn["not_tile_native"]) <= tile_exempt and tn["sites_checked"] == tn["tile_native_sites"] + len(tn["not_tile_native"]),
          {"sites_checked": tn["sites_checked"], "declared_fallback_exempt": len(set(tn["not_tile_native"]) & tile_exempt)})
    check("policy_reference_rule_sites_zero_for_linear",
          not any(k[0] in ("linear", "conv2d") and k[2] == "reference" for k in census_plain), None)

    # ---- 3. the graph itself -------------------------------------------------------------------------------------------
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
    n_q = len(used)                      # quantized Linear + Conv2d sites the graph holds (policy sites minus never-called ones)
    n_attn = len(attn_policy)
    cnt = lambda t: targets.get(f"atlas.{t}.default", 0)  # noqa: E731
    check("graph_qlinear_count_equals_quantized_linears",
          cnt("qlinear") == n_lin_used and cnt("qconv2d") == n_conv_used and unused_agree,
          {"qlinear": cnt("qlinear"), "policy_linear_in_graph": n_lin_used, "qconv2d": cnt("qconv2d"),
           "policy_conv2d_in_graph": n_conv_used, "policy_sites_not_in_graph": unused_derived,
           "declared_unused": unused_declared})
    check("graph_qk_pv_counts_equal_attention_sites",
          cnt("qk_matmul") == pairs_expected and cnt("pv_matmul") == pairs_expected and attn_ok,
          {"qk": cnt("qk_matmul"), "pv": cnt("pv_matmul"), "expected_pairs": pairs_expected, "policy_sites": n_attn,
           **{k: v for k, v in res["graph_attention"].items() if k != "pairs_by_site"}})
    act_expected = n_q + 4 * pairs_expected
    check("graph_quantize_act_count",
          cnt("quantize_act") == act_expected and set(cons_count) <= {"qlinear", "qconv2d", "qk_matmul", "pv_matmul"}
          and cons_count["qlinear"] == cnt("qlinear") and cons_count["qconv2d"] == cnt("qconv2d")
          and cons_count["qk_matmul"] == 2 * cnt("qk_matmul") and cons_count["pv_matmul"] == 2 * cnt("pv_matmul"),
          {"quantize_act": cnt("quantize_act"), "expected": act_expected, "consumers": dict(cons_count)})
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
    badbuf = [k for k in bufs if k.endswith("weight_codes") and bufs[k].dtype != torch.uint8] + \
             [k for k in bufs if k.endswith("weight_scale_exp") and bufs[k].dtype != torch.int32]
    nb = sum(1 for k in bufs if k.endswith("weight_codes"))
    buffer_sites_ok = {n for n in sites if n in code_stems} == {n for n in sites if n in exp_stems} == set(used) \
        and nb == n_q == sum(1 for k in bufs if k.endswith("weight_scale_exp"))
    check("graph_weight_buffers_uint8_int32", not badbuf and buffer_sites_ok,
          {"weight_code_buffers": nb, "expected": n_q, "bad": badbuf[:3], "policy_sites_without_buffers": unused_derived[:3]})
    fl = [k for k in dict(prog.named_parameters()) if k.endswith(".weight") and any(c in sites for c in (k[:-7], k[:-7].split(".", 1)[-1]))]
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
    t = gi["tile_native"]
    check("inspection_all_tile_native",
          set(t["not_tile_native"]) <= tile_exempt and t["sites_checked"] == t["tile_native_sites"] + len(t["not_tile_native"]),
          {"sites_checked": t["sites_checked"], "declared_fallback_exempt": len(set(t["not_tile_native"]) & tile_exempt)})
    rows = gi["weight_site_bindings"]
    check("inspection_all_weight_sites_native",
          len(rows) == n_q and {r["site"] for r in rows} == set(used) and all(r["native"] for r in rows),
          {"rows": len(rows), "expected": n_q})
    check("inspection_nothing_emulated", gi["still_emulated"]["emulated_modules"] == {}, gi["still_emulated"]["emulated_modules"])
    check("inspection_bf16_by_policy_is_declared_only",
          [r["site"] for r in gi["bf16_by_policy"]] == bf16_policy, [r["site"] for r in gi["bf16_by_policy"]])
    check("inspection_native_attention_modules",
          len(gi["native_fp8_attention_modules"]) == n_attn
          and {r["site"] for r in gi["native_fp8_attention_modules"]} == set(attn_policy),
          len(gi["native_fp8_attention_modules"]))
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
    res["pack_plan"], recorded_plan = pack_plan_report(policy, export, fx)      # informational, never a failing check
    res["pack_fold"] = pack_fold_report(policy, res["pack_plan"])               # informational: sites still out of range
    if require_pack_range:
        exps = pack_plan.static_sites(policy)
        lo, hi = pack_plan.PACK_ADMISSION
        outside = {k: r for k, r in recorded_plan.items()
                   if (r is not None and not r.get("in_range")) or (r is None and not lo <= 127 + exps[k]["exp"] <= hi)}
        missing = pack_plan.require_pack_range(outside)
        check("pack_range_plan_present", not missing,
              {"static_sites": len(recorded_plan), "sites_outside_without_plan": len(missing), "examples": missing[:5]})
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
                      "uncalibrated_fallback": len(fb_sites),
                      "pack_range_sites_still_outside": res["pack_fold"]["sites_still_out_of_range"],
                      "pack_fold_applied": res["pack_fold"]["folded"]}, indent=1))
    return res


if __name__ == "__main__":
    sys.exit(0 if main()["ok"] else 1)
