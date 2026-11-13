"""Pack-scale plan (REPORTING ONLY): for every static activation site, the E8M0 pack code, whether the compiler admits it,
the exact power-of-two fold that would bring it into the admitted range, which producer would absorb the fold and what
that costs. Nothing here changes a weight, an exponent, the FX graph or any export byte; the plan is written next to the
existing `compiler_pack_scale_admission` report (docs/algos/pack-plan.md, research: docs/algos/pack-range-options.md 4.A, 6).

A static activation exponent s means real = e4m3(byte) * 2^s, pack code c = 127 + s. Folding 2^k into the producer and packing
at c' = c + k gives the same bytes (`pack(2^k x, s + k) == pack(x, s)`), so the consumer's dequant exponent stays s.

Producer classes (spec item 3), derived from the FX graph text (the same text fx/graph.txt holds), else 'unknown':
norm_affine, layer_norm, norm_modulated, attn_p, attn_out_pv, attn_out_sdpa, gated_product, x_sigmoid_product,
bare_activation, linear_to_linear, modulation_add, raw_input, other:<op>, unknown.
"""
from __future__ import annotations

import collections
import copy
import re

from exporter.tile_native import E8M0_BIAS, PACK_ADMISSION

SCHEMA = "atlas_torchao.pack_plan.v1"
UNIT_CODE = 127                       # the compiler's unit pack code, default fold target (pack-range-options.md 4.A)
MAX_ABS_K = 20                        # spec item 2: a fold larger than this is not planned

# cost values (what the fold adds on the device), in the order summaries print them
COST_FREE = "free"                                  # an existing constant absorbs 2^k: no added instruction
COST_VMUL_ROW_BLOCK = "vmul_per_row_block"          # attention p: one vmul on the broadcast recip(sum) per (head, 32-row block)
COST_VMUL_TILE = "vmul_per_tile"                    # one vmul.bf16 per 32x32 tile before the pack (compiler fallback 4.F)
COST_ACT_REWRITE = "act_constant_rewrite"           # rewrite the activation's internal constants (algebraic, NOT run)
COST_HOST = "host"                                  # the host scales the raw model input before the BF16 conversion
COST_INFEASIBLE = "infeasible"                      # |k| above MAX_ABS_K: no plan
COST_NONE = "none_in_range"                         # already admitted: nothing to fold
COSTS = (COST_FREE, COST_VMUL_ROW_BLOCK, COST_VMUL_TILE, COST_ACT_REWRITE, COST_HOST, COST_INFEASIBLE)

# class -> (placement, cost when the producer has no other consumer, what to scale)
CLASS_PLAN = {
    "norm_affine": ("norm_gamma", COST_FREE, "scale the norm gamma (or the parameter-only (1 + w)) by 2^k"),
    "layer_norm": ("norm_gamma_beta", COST_FREE, "scale the LayerNorm gamma and beta by 2^k"),
    "gated_product": ("weight_exp:up_proj", COST_FREE,
                      "add k to every weight_scale_exp of the product's linear operand (the up_proj pop constant 2^(sx+sw) exists)"),
    "attn_out_pv": ("pv_rescale", COST_FREE, "fold 2^k into the PV BF16-pop rescale constant 2^(sp+sv)"),
    "linear_to_linear": ("weight_exp:producer_linear", COST_FREE,
                         "add k to the producing Linear's weight_scale_exp (only when nothing else reads its output)"),
    "attn_p": ("softmax_multiplier", COST_VMUL_ROW_BLOCK,
               "p is normalized, recip(sum) is data: one vmul of the broadcast recip(sum) by 2^k per (head, 32-row block), "
               "reused by every p tile of the block"),
    "bare_activation": ("act_constant", COST_ACT_REWRITE,
                        "rewrite the activation's internal constants and scale its producer (algebraic, not run); "
                        "fallback: one vmul per tile"),
    "x_sigmoid_product": ("act_constant", COST_ACT_REWRITE,
                          "rewrite the sigmoid / product constants and scale the producer (algebraic, not run); "
                          "fallback: one vmul per tile"),
    "attn_out_sdpa": ("vmul", COST_VMUL_TILE, "SDPA output has no constant to absorb 2^k: one vmul per tile"),
    "modulation_add": ("vmul", COST_VMUL_TILE, "the add of a conditioning output has no constant: one vmul per tile"),
    "norm_modulated": ("vmul", COST_VMUL_TILE, "the norm multiplier depends on a conditioning output: one vmul per tile"),
    "raw_input": ("host", COST_HOST, "the host scales the raw input before the f32 to BF16 conversion"),
    "unknown": ("vmul", COST_VMUL_TILE,
                "producer not derived (no FX graph at this stage): the compiler-side vmul fallback always applies"),
}
OTHER_PLAN = ("vmul", COST_VMUL_TILE, "producer class not recognized: one vmul per tile (compiler-side fallback)")
# classes whose free placement is lost when the producer has a second consumer (the scaled value would leak)
LEAK_SENSITIVE = {"norm_affine", "layer_norm", "gated_product", "attn_out_pv", "linear_to_linear",
                  "bare_activation", "x_sigmoid_product"}


# ---------------------------------------------------------------------------------------------
# The fold itself
# ---------------------------------------------------------------------------------------------

def e8m0_pack_code(exp: int) -> int:
    return E8M0_BIAS + int(exp)


def fold_for_code(code: int, lo: int = PACK_ADMISSION[0], hi: int = PACK_ADMISSION[1], target="nearest",
                  max_abs_k: int = MAX_ABS_K) -> dict:
    """Fold that brings pack code `code` into lo..hi. target: "nearest" (default; the in-range code with the smallest |k|,
    which is never larger than the |k| to the unit code 127) or an int code (clamped into lo..hi). Always reports k to the
    unit code 127 too. k == 0 and target_code == code when already admitted."""
    code = int(code)
    in_range = lo <= code <= hi
    k127 = UNIT_CODE - code
    if in_range:
        return {"in_range": True, "k": 0, "target_code": code, "k_to_unit_127": k127, "feasible": True}
    tgt = min(max(code, lo), hi) if target == "nearest" else min(max(int(target), lo), hi)
    k = tgt - code
    return {"in_range": False, "k": k, "target_code": tgt, "k_to_unit_127": k127, "feasible": abs(k) <= max_abs_k}


def fold_exactness(x, s: int, k: int) -> dict:
    """Spec item 7 helper (NOT wired to the exports, it needs real activations): for a BF16 tensor x, check
    e4m3_vpack(x, s) == e4m3_vpack(bf16(x * 2^k), s + k) byte for byte and that bf16(x * 2^k) == x * 2^k in float32."""
    import torch

    import reference.ref_numerics as rn

    xb = x.to(torch.bfloat16)
    xs = (xb.to(torch.float32) * (2.0 ** k))
    xk = xs.to(torch.bfloat16)
    a = rn.e4m3_vpack(xb, torch.tensor(int(s), dtype=torch.int32))
    b = rn.e4m3_vpack(xk, torch.tensor(int(s) + int(k), dtype=torch.int32))
    return {"elements": int(xb.numel()), "differing_bytes": int((a != b).sum()),
            "scaling_exact_in_bf16": bool((xk.to(torch.float32) == xs).all())}


def plan_record(exp: int, producer: dict | None = None, *, target="nearest", lo: int = PACK_ADMISSION[0],
                hi: int = PACK_ADMISSION[1], max_abs_k: int = MAX_ABS_K, shared_input: dict | None = None) -> dict:
    """The `pack_plan` record of one static activation site. producer: {"class", "other_consumers", "source", "node"?} or
    None (class 'unknown'). Reporting only."""
    code = e8m0_pack_code(exp)
    fold = fold_for_code(code, lo, hi, target, max_abs_k)
    cls = (producer or {}).get("class") or "unknown"
    others = list((producer or {}).get("other_consumers") or [])
    placement, cost, how = CLASS_PLAN.get(cls, OTHER_PLAN)
    if cls != "raw_input" and cls in LEAK_SENSITIVE and others:
        placement, cost = "vmul", COST_VMUL_TILE
        how = ("the producer has other consumers (" + ", ".join(others) + "): scaling it in place would leak, "
               "so one vmul per tile before the pack")
    if fold["in_range"]:
        status, cost = "in_range", COST_NONE
        placement, how = None, None
    elif not fold["feasible"]:
        status, cost = "infeasible", COST_INFEASIBLE
    else:
        status = "fold_planned"
    rec = {"schema": SCHEMA, "exp": int(exp), "e8m0_code": code, "admitted_range": [lo, hi],
           "in_range": fold["in_range"], "status": status,
           "fold": None if fold["in_range"] else {
               "k": fold["k"], "target_code": fold["target_code"], "k_to_unit_127": fold["k_to_unit_127"],
               "exact": "algebraic: pack(2^k x, s + k) == pack(x, s); the dequant / pop exponent stays s"},
           "producer": {"class": cls, "source": (producer or {}).get("source", "none"),
                        "other_consumers": others, **({"node": producer["node"]} if (producer or {}).get("node") else {}),
                        **({"mixed_classes": producer["mixed_classes"]} if (producer or {}).get("mixed_classes") else {})},
           "placement": placement, "cost": cost, "fold_free": cost == COST_FREE, "how": how}
    if cls == "attn_p":
        rec["note"] = "attention p needs a vmul on recip(sum): p is normalized and recip(sum) is data (pack-range-options.md 4.A)"
    if shared_input:
        rec["shared_input"] = shared_input
    return rec


# ---------------------------------------------------------------------------------------------
# Policy side: which sites are static, where records live
# ---------------------------------------------------------------------------------------------

def static_sites(policy: dict) -> dict:
    """{key: {"exp", "kind", "site", "operand"}} for every static activation: Linear / Conv2d `site` name, attention
    `site:operand`. Same keys as quant_export's static_act (the compiler_pack_scale_admission report)."""
    out = {}
    for name, s in policy.get("sites", {}).items():
        if s.get("status") != "quantized":
            continue
        if s.get("kind") == "attention":
            for op, r in (s.get("operands") or {}).items():
                if r.get("mode") == "static" and r.get("exp") is not None:
                    out[f"{name}:{op}"] = {"exp": int(r["exp"]), "kind": "attention", "site": name, "operand": op}
        else:
            a = s.get("activation") or {}
            if a.get("mode") == "static" and a.get("exp") is not None:
                out[name] = {"exp": int(a["exp"]), "kind": s.get("kind"), "site": name, "operand": None}
    return out


def dynamic_sites(policy: dict) -> dict:
    """Spec item 6: activations whose exponent is data dependent, so no `scale_code` immediate can express them."""
    out = {}
    for name, s in policy.get("sites", {}).items():
        if s.get("status") != "quantized":
            continue
        if s.get("kind") == "attention":
            for op, r in (s.get("operands") or {}).items():
                if r.get("mode") != "static":
                    out[f"{name}:{op}"] = "not_expressible_as_scale_code"
        else:
            a = s.get("activation") or {}
            if a.get("mode") != "static":
                out[name] = "not_expressible_as_scale_code"
    return out


def build_plan(policy: dict, producers: dict | None = None, shared: dict | None = None, target="nearest") -> dict:
    """{key: pack_plan record} for every static site of `policy`. producers / shared: {key: ...} from the FX graph
    (producers_from_graph); sites without one get class 'unknown' (attention p: 'attn_p' by structure)."""
    producers, shared = producers or {}, shared or {}
    plan = {}
    for key, st in static_sites(policy).items():
        prod = producers.get(key)
        if prod is None and st["operand"] == "p":
            prod = {"class": "attn_p", "source": "policy_operand", "other_consumers": []}
        plan[key] = plan_record(st["exp"], prod, target=target, shared_input=shared.get(key))
    return plan


def summarize(plan: dict, policy: dict | None = None, lo: int = PACK_ADMISSION[0], hi: int = PACK_ADMISSION[1]) -> dict:
    """`pack_plan_summary`: counts per producer class and per cost, over the sites outside lo..hi (and per class over all
    static sites), the shared-input groups with differing codes, the dynamic sites no scale code can express, and the pop
    exponent bound sx + sw (spec item 9, informational)."""
    if not plan:
        return {"schema": SCHEMA, "informational": True, "status": "not_checked_dynamic", "admitted_range": [lo, hi],
                "static_sites": None, "sites_outside": None, "free_fold_sites": None, "outside_by_class": None,
                "outside_by_cost": None,
                "note": "NOT CHECKED: this export has no static activation exponent, so there is nothing to plan; dynamic "
                        "exponents cannot be expressed as a scale code (see dynamic_sites_not_expressible_as_scale_code)",
                **({"dynamic_sites_not_expressible_as_scale_code": _dyn_summary(policy)} if policy is not None else {})}
    outside = {k: r for k, r in plan.items() if not r["in_range"]}
    by_class = collections.Counter(r["producer"]["class"] for r in outside.values())
    by_class_all = collections.Counter(r["producer"]["class"] for r in plan.values())
    by_cost = collections.Counter(r["cost"] for r in outside.values())
    ks = [r["fold"]["k"] for r in outside.values()]
    codes = [r["e8m0_code"] for r in plan.values()]
    summ = {
        "schema": SCHEMA, "informational": True, "status": "planned_static_sites", "admitted_range": [lo, hi],
        "default_target": "nearest in-range code",
        "static_sites": len(plan), "sites_outside": len(outside), "sites_in_range": len(plan) - len(outside),
        "code_range": [min(codes), max(codes)] if codes else None,
        "k_range": [min(ks), max(ks)] if ks else None,
        "max_abs_k": max((abs(k) for k in ks), default=0),
        "outside_by_class": dict(sorted(by_class.items())),
        "static_by_class": dict(sorted(by_class_all.items())),
        "outside_by_cost": {c: by_cost[c] for c in COSTS if by_cost.get(c)},
        "free_fold_sites": by_cost.get(COST_FREE, 0),
        "vmul_sites": by_cost.get(COST_VMUL_ROW_BLOCK, 0) + by_cost.get(COST_VMUL_TILE, 0),
        "infeasible_sites": by_cost.get(COST_INFEASIBLE, 0),
        "outside_unknown_producer": by_class.get("unknown", 0),
        "producer_source": dict(collections.Counter(r["producer"]["source"] for r in plan.values())),
        "producer_classes": ("derived from the FX graph" if all(r["producer"]["source"] == "fx_graph" for r in plan.values()) else
                             "NOT derived for sites without source fx_graph (no FX graph at export time; cost is then the "
                             "vmul fallback, not the free fold): see fx/policy.json pack_plan_summary"),
        "shared_input_groups": _shared_summary(plan),
        "pack_shift_attribute": "not needed while the plan is only recorded; a baked fold needs an explicit quantize_act "
                                "pack_shift separate from the dequant exponent (atlas_fx_ops.quantize_act, spec item 10)",
        "note": "INFORMATIONAL: no exponent, weight or graph was changed; compiler_pack_scale_admission is unchanged. "
                "A site is 'free' when an existing constant absorbs 2^k; attention p needs a vmul on recip(sum)."}
    if policy is not None:
        summ["dynamic_sites_not_expressible_as_scale_code"] = _dyn_summary(policy)
        summ["pop_exponent_bound"] = pop_exponent_bound(policy)
    return summ


def _dyn_summary(policy: dict) -> dict:
    dyn = dynamic_sites(policy)
    return {"count": len(dyn), "sites": sorted(dyn)[:20], "more": max(0, len(dyn) - 20)}


def _shared_summary(plan: dict) -> dict:
    groups = {}
    for r in plan.values():
        g = r.get("shared_input")
        if g and g["size"] > 1:
            groups[g["id"]] = g
    differing = [g for g in groups.values() if not g["identical"]]
    return {"multi_consumer_groups": len(groups), "groups_with_differing_codes": len(differing),
            "max_code_spread": max((max(g["codes"]) - min(g["codes"]) for g in differing), default=0)}


def pop_exponent_bound(policy: dict) -> dict:
    """Bound on the pop exponent sx + sw over the static Linear / Conv2d sites (activation exponent plus the weight exponent
    range recorded in policy.json). Spec item 9: -126 <= sx + sw <= 127 (BF16 constant)."""
    lo = hi = None
    n = 0
    for name, st in static_sites(policy).items():
        if st["kind"] == "attention":
            continue
        rng = ((policy["sites"][name].get("weight") or {}).get("e8m0_code_range"))
        if not rng:
            continue
        a, b = st["exp"] + rng["min"] - E8M0_BIAS, st["exp"] + rng["max"] - E8M0_BIAS
        lo = a if lo is None else min(lo, a)
        hi = b if hi is None else max(hi, b)
        n += 1
    return {"sites": n, "min": lo, "max": hi, "ok": None if lo is None else (-126 <= lo and hi <= 127),
            "required": [-126, 127]}


def require_pack_range(plan: dict) -> list:
    """Sites that are outside the admitted range and have no usable plan (record absent or infeasible): what
    `verify_bundle --require-pack-range` fails on. `plan` maps every static site to its record or None."""
    return sorted(k for k, r in plan.items()
                  if r is None or (not r.get("in_range") and (r.get("status") != "fold_planned" or not r.get("fold"))))


def attach(policy: dict, plan: dict, summary: dict | None = None) -> dict:
    """Write `pack_plan` (Linear / Conv2d: site["pack_plan"] = record; attention: site["pack_plans"] = {operand: record})
    and the top-level `pack_plan_summary` into `policy` IN PLACE and return it. Existing keys are untouched."""
    sites = static_sites(policy)
    for key, rec in plan.items():
        st = sites[key]
        site = policy["sites"][st["site"]]
        if st["operand"]:
            site.setdefault("pack_plans", {})[st["operand"]] = rec
        else:
            site["pack_plan"] = rec
    policy["pack_plan_summary"] = summary if summary is not None else summarize(plan, policy)
    return policy


def recorded_plan(policy: dict) -> dict:
    """{key: record or None} for every static site: the `pack_plan` records already present in `policy`."""
    out = {}
    for key, st in static_sites(policy).items():
        s = policy["sites"][st["site"]]
        out[key] = (s.get("pack_plans") or {}).get(st["operand"]) if st["operand"] else s.get("pack_plan")
    return out


def with_plan(policy: dict, producers: dict | None = None, shared: dict | None = None, target="nearest") -> dict:
    """A deep copy of `policy` carrying the pack plan (policy itself is not modified)."""
    pol = copy.deepcopy(policy)
    plan = build_plan(pol, producers, shared, target)
    return attach(pol, plan)


# ---------------------------------------------------------------------------------------------
# FX graph side: producer class of every packed tensor (port of docs/algos/pack-range-options/fx_pack_census.py)
# ---------------------------------------------------------------------------------------------

PASS = {"aten.to.dtype", "aten.to.dtype_layout", "aten.view.default", "aten.reshape.default", "aten.contiguous.default",
        "aten.transpose.int", "aten.expand.default", "aten.unsqueeze.default", "aten.permute.default",
        "aten.clone.default", "aten._unsafe_view.default", "aten.flatten.using_ints", "aten.squeeze.dim",
        "aten.slice.Tensor", "aten.select.int", "aten.detach.default", "aten.alias.default", "aten.t.default",
        "operator.getitem", "aten.squeeze.dims", "aten.unflatten.int", "aten.narrow.default", "aten.cat.default"}
_LINE = re.compile(r'^\s*%(\S+) : \[num_users=(\d+)\] = call_function\[target=([^\]]+)\]\(args = \((.*)\), kwargs = (.*)\)\s*$')
_SIG = re.compile(r"^(\S+): BUFFER target='([^']+)'")                              # fx/graph_signature.txt, one line per input
_SIG_INLINE = re.compile(r"TensorArgument\(name='([^']+)'\), target='([^']+)'")    # str(ExportGraphSignature) on one line


def parse_graph_text(text: str):
    """str(program.graph) / fx/graph.txt -> ({name: {t, args, refs}}, [names in order])."""
    nodes, order = {}, []
    for ln in text.splitlines():
        m = _LINE.match(ln)
        if not m:
            continue
        name, _, tgt, args, _kw = m.groups()
        tgt = tgt.replace("torch.ops.", "")
        if tgt.startswith("atlas."):
            tgt = tgt.replace(".default", "")
        nodes[name] = dict(t=tgt, args=args, refs=re.findall(r"%([\w\.]+)", args))
        order.append(name)
    return nodes, order


def buffer_targets(signature_text: str) -> dict:
    """str(program.graph_signature) / fx/graph_signature.txt -> {buffer placeholder name: target}."""
    out = {m.group(1): m.group(2) for m in map(_SIG.match, signature_text.splitlines()) if m}
    out.update({a: b for a, b in _SIG_INLINE.findall(signature_text) if a.startswith("b_")})
    return out


def _real(nodes, n):
    for _ in range(50):
        if n in nodes and nodes[n]["t"] in PASS and nodes[n]["refs"]:
            n = nodes[n]["refs"][0]
        else:
            break
    return n


def _kind(nodes, r):
    rn_ = _real(nodes, r)
    if rn_ not in nodes:
        return "param" if r.startswith(("p_", "b_")) else "input"
    t = nodes[rn_]["t"]
    return t.split(".")[1] if t.startswith("atlas.") else t.replace("aten.", "").split(".")[0]


def _desc(nodes, n):
    if n not in nodes:
        return "input/param:" + n
    d = nodes[n]
    t = d["t"]
    if t in ("aten.mul.Tensor", "aten.add.Tensor"):
        ks = [_kind(nodes, r) for r in d["refs"]] + (["const"] if len(d["refs"]) == 1 else [])
        return ("mul(" if t == "aten.mul.Tensor" else "add(") + ",".join(ks) + ")"
    return t


def _param_only(nodes, n, depth=0):
    if n not in nodes:
        return n.startswith(("p_", "b_"))
    if depth > 8:
        return False
    return all(_param_only(nodes, r, depth + 1) for r in nodes[n]["refs"]) if nodes[n]["refs"] else True


def _category(nodes, src, pd):
    if pd == "mul(mul,add)":
        add_operand = [r for r in nodes[src]["refs"] if _kind(nodes, r) == "add"]
        return "norm_affine" if add_operand and _param_only(nodes, _real(nodes, add_operand[0])) else "norm_modulated"
    if pd == "mul(param,mul)":
        return "norm_affine"
    if pd == "aten.layer_norm.default":
        return "layer_norm"
    if pd == "aten.softmax.int":
        return "attn_p"
    if pd == "atlas.pv_matmul":
        return "attn_out_pv"
    if pd == "aten.scaled_dot_product_attention.default":
        return "attn_out_sdpa"
    if pd in ("mul(silu,qlinear)", "mul(gelu,qlinear)"):
        return "gated_product"
    if pd == "mul(qlinear,sigmoid)":
        return "x_sigmoid_product"
    if pd in ("aten.gelu.default", "aten.silu.default"):
        return "bare_activation"
    if pd == "atlas.qlinear":
        return "linear_to_linear"
    if pd.startswith("add("):
        return "modulation_add"
    if pd.startswith("input/param:") or pd == "aten.im2col.default":
        return "raw_input"
    return "other:" + pd


def _leaves(nodes, users, n):
    out = []
    for u in users[n]:
        t = nodes[u]["t"]
        if t in PASS or t.startswith("aten._assert"):
            out += _leaves(nodes, users, u)
        else:
            out.append(t)
    return out


def _match_site(path: str, names, suffix=""):
    """Policy site name for a graph-side path ('policy.model.x.y' for site 'model.x.y'): equal, else the longest site that is a
    dotted suffix of the path. None when none matches."""
    if suffix and path.endswith(suffix):
        path = path[:-len(suffix)]
    if path in names:
        return path
    best = [n for n in names if path.endswith("." + n)]
    return max(best, key=len) if best else None


def producers_from_graph(graph_text: str, signature_text: str, policy: dict) -> tuple[dict, dict]:
    """Walk the FX graph back from every static atlas.quantize_act. Returns (producers, shared): producers {site key:
    {"class", "other_consumers", "source": "fx_graph", "node", ["mixed_classes"]}} and shared {site key: shared-input
    group}; keys as in static_sites. A site packed at several nodes (a module called more than once) takes the class of the
    first node and lists the others under mixed_classes when they differ. Sites the graph does not show are absent."""
    nodes, order = parse_graph_text(graph_text)
    targets = buffer_targets(signature_text)
    wanted = static_sites(policy)
    site_names = {st["site"] for st in wanted.values() if st["kind"] != "attention"}
    attn_names = {st["site"] for st in wanted.values() if st["kind"] == "attention"}
    users = collections.defaultdict(list)
    for n in order:
        for r in nodes[n]["refs"]:
            users[r].append(n)
    per_key, groups = collections.defaultdict(list), collections.defaultdict(list)
    for n in order:
        if nodes[n]["t"] != "atlas.quantize_act":
            continue
        a = [x.strip() for x in nodes[n]["args"].split(",")]
        if len(a) < 5 or a[1] != "static":
            continue
        key = None
        if len(a) > 6 and ":" in a[6]:                                    # attention operand tag "<site>:<op>"
            path, _, op = a[6].rpartition(":")
            site = _match_site(path, attn_names)
            key = f"{site}:{op}" if site else None
        else:
            for gi in users[n]:
                for u in users[gi]:
                    if nodes[u]["t"] in ("atlas.qlinear", "atlas.qconv2d"):
                        w = next((r for r in nodes[u]["refs"] if r.endswith("weight_codes")), None)
                        tgt = targets.get(w)
                        if tgt is not None:
                            key = _match_site(tgt, site_names, ".weight_codes")
        if key is None or key not in wanted:
            continue
        src = _real(nodes, nodes[n]["refs"][0])
        cls = _category(nodes, src, _desc(nodes, src))
        extra = sorted({t for t in _leaves(nodes, users, src)
                        if t not in ("atlas.quantize_act", "aten._assert_tensor_metadata.default")})
        per_key[key].append({"class": cls, "other_consumers": extra, "node": n, "code": E8M0_BIAS + int(a[4]), "src": src})
        groups[src].append(key)
    producers, shared = {}, {}
    for key, rows in per_key.items():
        first = rows[0]
        classes = sorted({r["class"] for r in rows})
        p = {"class": first["class"], "other_consumers": first["other_consumers"], "source": "fx_graph", "node": first["node"]}
        if len(classes) > 1:
            p["mixed_classes"] = classes
        producers[key] = p
    for src, keys in groups.items():
        codes = sorted({E8M0_BIAS + wanted[k]["exp"] for k in keys if k in wanted})
        if len(keys) < 2 and len(codes) < 2:
            continue
        g = {"id": src, "size": len(keys), "codes": codes, "identical": len(codes) == 1}
        for k in keys:
            shared[k] = g
    return producers, shared


def plan_from_files(policy: dict, graph_text: str | None = None, signature_text: str | None = None, target="nearest") -> dict:
    """Plan of `policy` with producers read from an FX graph text pair (None: producers 'unknown'). Read-only helper for
    verify_bundle and the census script."""
    producers, shared = ({}, {}) if graph_text is None else producers_from_graph(graph_text, signature_text or "", policy)
    return build_plan(policy, producers, shared, target)
