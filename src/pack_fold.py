"""Pack-range fold (mitigation A of docs/algos/pack-range-options.md), an OPT-IN exporter pass.

A static activation exponent s means real = e4m3(byte) * 2^s and the pack code is c = 127 + s. The compiler admits
c in 122..128. Packing 2^k * x at exponent s + k gives byte-identical FP8 codes to packing x at s, so for every static
site that sits behind a producer which already multiplies by a constant, the pass
  1. scales the producer by 2^k exactly:
       norm_affine      (RMSNorm gamma `mul(param, mul)`; Gemma x * (1 + w) when float32 and exact)
                                                                gamma             <- gamma * 2^k      (BF16, exact)
       layer_norm       (weight and bias)                       both scaled by 2^k (exact)
       gated_product    (act(gate) * up)                        up_proj weight_scale_exp += k  (bias * 2^k)
       linear_to_linear (a Linear feeding only a Linear)        producer weight_scale_exp += k (bias * 2^k)
       attn_out_pv      (P.V feeding o_proj)                    policy attention site `pv_pop_shift` += k, i.e. the
                                                                PV pop rescale constant 2^(sp+sv+k) (atlas.pv_matmul
                                                                `pop_shift`, ref_attention `pop_shift`)
  2. sets the static input exponent of every consumer of that tensor to s + k (policy activation.exp / e8m0_code and the
     `input_scale_exp` tensor), so the pack code lands in 122..128, and
  3. subtracts k from the consumer's weight_scale_exp (codes untouched) so the pop / dequant constant 2^(sx + sw) is the
     same number as before. The FX graph is unchanged (the static_exp literal and the int32 buffers differ; only the
     pv_matmul node of a folded attention site gains one trailing integer argument).
Nothing is re-calibrated and no FP8 code changes. Classes that are not free (attention p: a vmul on recip(sum); activation
constant rewrites; one vmul per tile; host) are NOT touched and stay reported as out of range with their plan.

All sites that read one producer must agree on k: the group's k is the smallest-|k| shift that brings every member into
122..128 (a member that is already in range stays in range). Docs: docs/algos/pack-fold.md.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import re
from pathlib import Path

import torch

import pack_plan as pp
import tile_native as tn

SCHEMA = "atlas_torchao.pack_fold.v1"
FREE_CLASSES = ("norm_affine", "layer_norm", "gated_product", "attn_out_pv", "linear_to_linear")
BF16_MAX = float(torch.finfo(torch.bfloat16).max)
BF16_MIN_NORMAL = float(torch.finfo(torch.bfloat16).tiny)
E8M0_CODE_LIMITS = (1, 254)               # a weight exponent code outside this would alias / clamp (VectorEngine.scala e8m0 clamp)
_PARAM = re.compile(r"^(\S+): PARAMETER target='([^']+)'")
_PARAM_INLINE = re.compile(r"TensorArgument\(name='([^']+)'\), target='([^']+)'")


# ---------------------------------------------------------------------------------------------
# Graph side: who produces each packed tensor, and which weights / tensors the fold has to touch
# ---------------------------------------------------------------------------------------------

def param_targets(signature_text: str) -> dict:
    """{parameter placeholder name: target} from fx/graph_signature.txt (either text form)."""
    out = {m.group(1): m.group(2) for m in map(_PARAM.match, signature_text.splitlines()) if m}
    out.update({a: b for a, b in _PARAM_INLINE.findall(signature_text) if a.startswith("p_")})
    return out


def _match_tensor(target: str, names) -> str | None:
    """The stored tensor named by a graph-side target ('model.model.layers.0.x.weight' for 'model.layers.0.x.weight')."""
    if target in names:
        return target
    best = [n for n in names if target.endswith("." + n)]
    return max(best, key=len) if best else None


def _users(nodes, order):
    users = collections.defaultdict(list)
    for n in order:
        for r in nodes[n]["refs"]:
            users[r].append(n)
    return users


def _qlinear_site(nodes, buffers, node, site_names):
    w = next((r for r in nodes[node]["refs"] if r.endswith("weight_codes")), None)
    tgt = buffers.get(w)
    return pp._match_site(tgt, site_names, ".weight_codes") if tgt is not None else None


def graph_info(graph_text: str, signature_text: str, policy: dict, tensor_names=()) -> dict:
    """Walk the FX graph text back from every static atlas.quantize_act. Returns {site key: record} (keys as pack_plan.static_sites)
    with: class, other_consumers, node (the quantize_act), nodes (how many packs of this key), src (producer node), consumer_site
    (the Linear / Conv2d that reads the pack, or None), group_acts (every quantize_act that reads `src`: [(mode, key|None)]),
    producer {kind: norm_gamma|weight_exp|pv_shift|unsupported, ...}."""
    nodes, order = pp.parse_graph_text(graph_text)
    buffers, params = pp.buffer_targets(signature_text), param_targets(signature_text)
    wanted = pp.static_sites(policy)
    qsites = {n for n, s in policy["sites"].items() if s.get("status") == "quantized" and s.get("kind") in ("linear", "conv2d")}
    attn_names = {st["site"] for st in wanted.values() if st["kind"] == "attention"}
    users = _users(nodes, order)
    acts_of_src = collections.defaultdict(list)
    per_key = collections.defaultdict(list)
    site_calls = collections.Counter()                  # qlinear / qconv2d nodes per weight site (a module called twice shows 2)
    for n in order:
        if nodes[n]["t"] in ("atlas.qlinear", "atlas.qconv2d"):
            s_ = _qlinear_site(nodes, buffers, n, qsites)
            if s_:
                site_calls[s_] += 1
    for n in order:
        if nodes[n]["t"] != "atlas.quantize_act":
            continue
        a = [x.strip() for x in nodes[n]["args"].split(",")]
        src = pp._real(nodes, nodes[n]["refs"][0])
        mode = a[1] if len(a) > 1 else "?"
        key, consumers = None, set()
        if mode == "static" and len(a) > 6 and ":" in a[6]:
            path, _, op = a[6].rpartition(":")
            site = pp._match_site(path, attn_names)
            key = f"{site}:{op}" if site else None
        elif mode == "static":
            for gi in users[n]:
                for u in users[gi]:
                    if nodes[u]["t"] in ("atlas.qlinear", "atlas.qconv2d"):
                        s = _qlinear_site(nodes, buffers, u, qsites)
                        if s:
                            consumers.add(s)
                            key = s
        acts_of_src[src].append((mode, key if key in wanted else None))
        if key in wanted:
            per_key[key].append({"node": n, "src": src, "consumers": consumers, "a": a})
    info = {}
    for key, rows in per_key.items():
        first = rows[0]
        src = first["src"]
        pd = pp._desc(nodes, src)
        cls = pp._category(nodes, src, pd)
        extra = sorted({t for t in pp._leaves(nodes, users, src)
                        if t not in ("atlas.quantize_act", "aten._assert_tensor_metadata.default")})
        info[key] = {"class": cls, "other_consumers": extra, "node": first["node"], "nodes": len(rows), "src": src,
                     "desc": pd, "site_calls": dict(site_calls), "consumer_site": next(iter(first["consumers"])) if len(first["consumers"]) == 1 else None,
                     "consumer_sites": sorted(first["consumers"]), "group_acts": acts_of_src[src],
                     "producer": _producer(nodes, users, buffers, params, src, cls, pd, qsites, attn_names, tensor_names)}
        info[key]["producer"] = _single_user(nodes, users, info[key]["producer"], params)
    return info


def _producer(nodes, users, buffers, params, src, cls, pd, qsites, attn_names, tensor_names) -> dict:
    def stored(ph):
        tgt = params.get(ph)
        return (_match_tensor(tgt, tensor_names) if tgt and tensor_names else None), tgt

    if cls == "norm_affine":
        if pd == "mul(mul,add)":                       # Gemma style x * (1 + w): the multiplier is a parameter-only add
            addn = next((pp._real(nodes, r) for r in nodes[src]["refs"] if pp._kind(nodes, r) == "add"), None)
            ph = pp._real(nodes, nodes[addn]["refs"][0]) if addn in nodes and nodes[addn]["refs"] else None
            tensor, tgt = stored(ph)
            if tensor is None or not nodes[addn]["args"].rstrip().endswith(("1.0", ", 1")):
                return {"kind": "unsupported", "why": f"(1 + w) parameter {ph!r} is not a stored tensor / not add(param, 1)"}
            return {"kind": "norm_one_plus", "tensor": tensor, "param_target": tgt}
        ph = next((pp._real(nodes, r) for r in nodes[src]["refs"] if pp._kind(nodes, r) == "param"), None)
        tensor, tgt = stored(ph)
        if tensor is None:
            return {"kind": "unsupported", "why": f"gamma parameter {ph!r} (target {tgt!r}) is not a stored tensor"}
        return {"kind": "norm_gamma", "tensor": tensor, "param_target": tgt}
    if cls == "layer_norm":
        refs = nodes[src]["refs"]
        if len(refs) < 3:
            return {"kind": "unsupported", "why": "LayerNorm without weight and bias arguments"}
        (wt, wtgt), (bt, _) = stored(pp._real(nodes, refs[1])), stored(pp._real(nodes, refs[2]))
        if wt is None or bt is None:
            return {"kind": "unsupported", "why": "LayerNorm weight / bias are not stored tensors"}
        return {"kind": "layer_norm", "tensor": wt, "bias_tensor": bt, "param_target": wtgt}
    if cls in ("gated_product", "linear_to_linear"):
        if cls == "gated_product":
            ref = next((r for r in nodes[src]["refs"] if pp._kind(nodes, r) == "qlinear"), None)
            prod = pp._real(nodes, ref) if ref else None
        else:
            prod = src
        if prod not in nodes or nodes[prod]["t"] != "atlas.qlinear":
            return {"kind": "unsupported", "why": "the Linear operand of the product is not an atlas.qlinear node"}
        site = _qlinear_site(nodes, buffers, prod, qsites)
        outs = sorted({t for t in pp._leaves(nodes, users, prod)
                       if t not in ("aten._assert_tensor_metadata.default",)})
        return {"kind": "weight_exp", "site": site, "node": prod, "qlinear_consumers": outs}
    if cls == "attn_out_pv":
        p_act = pp._real(nodes, nodes[src]["refs"][0])
        a = [x.strip() for x in nodes[p_act]["args"].split(",")] if p_act in nodes else []
        site = pp._match_site(a[6].rpartition(":")[0], attn_names) if len(a) > 6 and ":" in a[6] else None
        return {"kind": "pv_shift", "site": site, "node": src}
    return {"kind": "unsupported", "why": f"producer class {cls!r} is not free"}


def _single_user(nodes, users, prod: dict, params: dict) -> dict:
    """A parameter placeholder read by more than one node (shared / tied gamma, a module called twice) cannot be scaled in place."""
    ph = [p for p, tgt in params.items() if tgt == prod.get("param_target")] if prod.get("param_target") else []
    if prod["kind"] in ("norm_gamma", "norm_one_plus", "layer_norm") and any(
            len([u for u in users[p] if not u.startswith("_assert")]) != 1 for p in ph):
        return {"kind": "unsupported", "why": "the parameter placeholder is read by more than one node (shared or called twice)"}
    return prod


def producers_for_plan(info: dict) -> dict:
    """pack_plan.build_plan `producers` argument from graph_info."""
    return {k: {"class": r["class"], "other_consumers": r["other_consumers"], "source": "fx_graph", "node": r["node"]}
            for k, r in info.items()}


# ---------------------------------------------------------------------------------------------
# Planning: one k per producer group
# ---------------------------------------------------------------------------------------------

def group_k(codes, lo=pp.PACK_ADMISSION[0], hi=pp.PACK_ADMISSION[1]):
    """Smallest-|k| shift that puts every code of the group into lo..hi, or None when the group spans more than the range.
    0 when all are in range. For a single site this is exactly pack_plan.fold_for_code(target="nearest")."""
    a, b = lo - min(codes), hi - max(codes)          # k must satisfy a <= k <= b
    if a > b:
        return None
    return 0 if a <= 0 <= b else (a if a > 0 else b)


def plan_folds(policy: dict, info: dict, *, max_abs_k: int = pp.MAX_ABS_K) -> dict:
    """{src: group} for every producer that feeds at least one static site; a group is
    {src, class, keys, codes, k, applied (planned), reason, producer, consumer_sites}. `reason` explains a group that is left
    alone (in_range, class_not_free, other_consumers, ...). Pure function of the policy and the graph record."""
    lo, hi = pp.PACK_ADMISSION
    sites = pp.static_sites(policy)
    groups: dict = {}
    for key, r in info.items():
        g = groups.setdefault(r["src"], {"src": r["src"], "class": r["class"], "keys": [], "producer": r["producer"],
                                         "other_consumers": r["other_consumers"], "group_acts": r["group_acts"], "info": {}})
        g["keys"].append(key)
        g["info"][key] = r
    for g in groups.values():
        keys = sorted(g["keys"])
        g["keys"] = keys
        g["codes"] = {k: tn.E8M0_BIAS + sites[k]["exp"] for k in keys}
        k = group_k(list(g["codes"].values()))
        g["k"] = k if k is not None else 0
        g["planned_k_to_nearest"] = {kk: pp.fold_for_code(c)["k"] for kk, c in g["codes"].items()}
        g["reason"] = _group_reason(policy, g, k, max_abs_k)
        g["applied"] = False
    return groups


def _group_reason(policy, g, k, max_abs_k) -> str | None:
    infos = g["info"].values()
    lo, hi = pp.PACK_ADMISSION
    if all(lo <= c <= hi for c in g["codes"].values()):
        return "in_range"
    if g["class"] not in FREE_CLASSES:
        return f"class_not_free:{g['class']}"
    if g["producer"]["kind"] == "unsupported":
        return "producer_unsupported: " + g["producer"]["why"]
    if g["other_consumers"]:
        return "other_consumers:" + ",".join(g["other_consumers"])
    if k is None:
        return "group_spread: the codes " + str(sorted(set(g["codes"].values()))) + " do not fit one 7-code window"
    if abs(k) > max_abs_k:
        return f"infeasible_k:{k}"
    if any(r["nodes"] != 1 for r in infos):
        return "multi_call_site: a site is packed at more than one node"
    if any(r["consumer_site"] is None for r in infos):
        return "consumer_not_a_single_linear"
    calls = next(iter(infos))["site_calls"]
    multi = sorted(s_ for s_ in {r["consumer_site"] for r in infos} | ({g["producer"]["site"]} if g["producer"].get("site") else set())
                   if s_ in calls and calls[s_] != 1)
    if multi:
        return "multi_call_site: weights read by more than one node: " + ",".join(multi)
    # every quantize_act that reads the producer must be one of this group's static Linear sites
    bad = [m for m, key in g["group_acts"] if m != "static" or key not in g["keys"]]
    if bad:
        return "non_static_or_attention_consumer"
    for r in infos:
        s = policy["sites"][r["consumer_site"]]
        if s.get("residual_passes") or s.get("kind") not in ("linear", "conv2d"):
            return f"consumer_has_residual_passes:{r['consumer_site']}"
    prod = g["producer"]
    if prod["kind"] == "weight_exp":
        ps = policy["sites"].get(prod["site"] or "", {})
        if not prod["site"] or ps.get("status") != "quantized":
            return "producer_site_unresolved"
        if ps.get("residual_passes"):
            return f"producer_has_residual_passes:{prod['site']}"
        if g["class"] == "gated_product" and prod["qlinear_consumers"] != ["aten.mul.Tensor"]:
            return "producer_output_has_other_consumers:" + ",".join(prod["qlinear_consumers"])
    if prod["kind"] == "pv_shift" and not prod["site"]:
        return "attention_site_unresolved"
    return None


# ---------------------------------------------------------------------------------------------
# Applying
# ---------------------------------------------------------------------------------------------

def _scale_pow2(t: torch.Tensor, k: int):
    """(t * 2^k in t's dtype, exact?, stats). Exact means the float32 product survives the cast back to t.dtype unchanged, is finite
    and no non-zero result falls below the BF16 minimum normal."""
    t32 = t.to(torch.float32)
    s32 = t32 * (2.0 ** k)
    out = s32.to(t.dtype)
    nz = s32[s32 != 0].abs()
    exact = (bool(torch.isfinite(s32).all()) and bool((out.to(torch.float32) == s32).all())
             and (not nz.numel() or float(nz.min()) >= BF16_MIN_NORMAL))        # a BF16 subnormal result counts as underflow
    stats = {"elements": t.numel(), "min_abs_nonzero": float(nz.min()) if nz.numel() else None,
             "max_abs": float(s32.abs().max()) if s32.numel() else 0.0}
    return out, exact, stats


def _scale_one_plus(w: torch.Tensor, k: int):
    """w' with fl32(w' + 1) == 2^k * fl32(w + 1) bit for bit (the graph forms 1 + w in float32), or (None, stats) when some
    element has no such w' in the parameter dtype. Only float32 parameters qualify."""
    if w.dtype != torch.float32:
        return None, {"why": f"parameter dtype {w.dtype}: 2^k (1 + w) - 1 is not exact outside float32"}
    w64 = w.double()
    new64 = (w64 + 1.0) * (2.0 ** k) - 1.0
    new = new64.to(torch.float32)
    target = (w + 1.0) * (2.0 ** k)
    # what has to hold is the identity of the multiplier the graph forms, fl32(w' + 1) == 2^k fl32(w + 1); w' itself need not be exact
    exact = bool(torch.isfinite(new).all()) and bool(((new + 1.0) == target).all())
    mult = target.abs()
    nz = mult[mult != 0]
    return (new if exact else None), {"elements": w.numel(), "min_abs_nonzero": float(nz.min()) if nz.numel() else None,
                                      "max_abs": float(mult.max()) if mult.numel() else 0.0}


def _weight_guard(exp: torch.Tensor, delta: int):
    """New weight exponents must stay an admissible E8M0 code and keep the dequantized weights normal in BF16."""
    new = exp.to(torch.int64) + delta
    codes = new + tn.E8M0_BIAS
    if int(codes.min()) < E8M0_CODE_LIMITS[0] or int(codes.max()) > E8M0_CODE_LIMITS[1]:
        return False, f"weight exponent code would leave {E8M0_CODE_LIMITS}: {int(codes.min())}..{int(codes.max())}"
    if float(2.0 ** (int(new.min()) - 6)) < BF16_MIN_NORMAL or float(448.0 * 2.0 ** int(new.max())) > BF16_MAX:
        return False, "dequantized weights would leave the BF16 normal range"
    return True, None


def apply_folds(tensors: dict, policy: dict, groups: dict, *, only_free=True) -> tuple[dict, dict, dict]:
    """Apply every plannable group. Returns (tensors, policy, report); inputs are not modified (tensors are copied only where
    they change). A group whose exactness guard fails (BF16 range, weight code range, bias) is not applied and says why."""
    tensors = dict(tensors)
    policy = copy.deepcopy(policy)
    report = {"groups": [], "bf16_range": {"gamma_sites": 0, "min_abs_nonzero_after": None, "max_abs_after": None,
                                          "bf16_min_normal": BF16_MIN_NORMAL, "bf16_max": BF16_MAX, "failures": []}}
    rng = report["bf16_range"]
    aliased = set(policy.get("aliases") or {}) | set((policy.get("aliases") or {}).values())
    param_edited, producer_used = set(), set()            # stored tensors a producer edit has already claimed
    for src, g in sorted(groups.items()):
        row = {"src": src, "class": g["class"], "keys": g["keys"], "k": g["k"], "reason": g["reason"], "applied": False}
        report["groups"].append(row)
        if g["reason"] is not None:
            continue
        k = g["k"]
        deltas = collections.defaultdict(int)             # weight_scale_exp tensor name -> delta
        edits = {}                                        # tensor name -> new tensor
        stats = []                                        # BF16 range stats, merged only when the group is applied
        why = None
        prod = g["producer"]
        claimed = [prod[f] for f in ("tensor", "bias_tensor") if prod.get(f)]
        if prod["kind"] == "weight_exp":
            claimed.append(policy["sites"][prod["site"]]["weight"]["scale_exp"])
        touched = claimed + [policy["sites"][g["info"][key]["consumer_site"]]["weight"]["scale_exp"] for key in g["keys"]]
        if any(t_ in aliased for t_ in touched):
            why = "aliased_tensor: " + ",".join(sorted(t_ for t_ in touched if t_ in aliased))
        elif any(t_ in param_edited | producer_used for t_ in claimed):
            why = "shared_producer_tensor: " + ",".join(sorted(t_ for t_ in claimed if t_ in param_edited | producer_used))
        if why is not None:
            row["reason"] = g["reason"] = why
            continue
        if prod["kind"] == "norm_gamma":
            new, exact, st = _scale_pow2(tensors[prod["tensor"]], k)
            if not exact:
                why = f"bf16_range: {prod['tensor']} * 2^{k} is not finite / not exact in {tensors[prod['tensor']].dtype}"
                rng["failures"].append({"tensor": prod["tensor"], "k": k, **st})
            else:
                edits[prod["tensor"]] = new
                stats.append(st)
        elif prod["kind"] == "norm_one_plus":
            new, st = _scale_one_plus(tensors[prod["tensor"]], k)
            if new is None:
                why = f"bf16_range: {prod['tensor']}: (1 + w) * 2^{k} is not representable exactly ({st.get('why', 'float32 mismatch')})"
                rng["failures"].append({"tensor": prod["tensor"], "k": k, **st})
            else:
                edits[prod["tensor"]] = new
                stats.append(st)
        elif prod["kind"] == "layer_norm":
            for tname in (prod["tensor"], prod["bias_tensor"]):
                new, exact, st = _scale_pow2(tensors[tname], k)
                if not exact:
                    why = f"bf16_range: {tname} * 2^{k} is not finite / not exact in {tensors[tname].dtype}"
                    rng["failures"].append({"tensor": tname, "k": k, **st})
                    break
                edits[tname] = new
                stats.append(st)
        elif prod["kind"] == "weight_exp":
            ps = policy["sites"][prod["site"]]
            deltas[ps["weight"]["scale_exp"]] += k
            if ps.get("bias"):
                bn = ps["bias"]["tensor"]
                new, exact, st = _scale_pow2(tensors[bn], k)
                if not exact:
                    why = f"bf16_range: bias {bn} * 2^{k} is not exact"
                else:
                    edits[bn] = new
        for key in g["keys"]:
            cs = policy["sites"][g["info"][key]["consumer_site"]]
            deltas[cs["weight"]["scale_exp"]] -= k
        if why is None:
            for name, d in deltas.items():
                if d == 0:
                    continue
                ok, msg = _weight_guard(tensors[name], d)
                if not ok:
                    why = f"{name}: {msg}"
                    break
        if why is not None:
            row["reason"] = why
            g["reason"] = why
            continue
        for name, new in edits.items():
            tensors[name] = new
        if stats:
            rng["gamma_sites"] += 1
            for st in stats:
                rng["min_abs_nonzero_after"] = _min_opt(rng["min_abs_nonzero_after"], st["min_abs_nonzero"])
                rng["max_abs_after"] = _max_opt(rng["max_abs_after"], st["max_abs"])
        (producer_used if prod["kind"] == "weight_exp" else param_edited).update(claimed)
        for name, d in deltas.items():
            if d:
                tensors[name] = (tensors[name].to(torch.int64) + d).to(tensors[name].dtype)
        # policy: weight code ranges follow the exponent deltas
        for s in policy["sites"].values():
            w = s.get("weight") or {}
            if w.get("scale_exp") in deltas and deltas[w["scale_exp"]] and "e8m0_code_range" in w:
                d = deltas[w["scale_exp"]]
                w["e8m0_code_range"] = {"min": w["e8m0_code_range"]["min"] + d, "max": w["e8m0_code_range"]["max"] + d}
        if prod["kind"] == "pv_shift":
            site = policy["sites"][prod["site"]]
            site["pv_pop_shift"] = int(site.get("pv_pop_shift", 0)) + k
        for key in g["keys"]:
            cs = policy["sites"][g["info"][key]["consumer_site"]]
            act = cs["activation"]
            act["exp"] = int(act["exp"]) + k
            act["e8m0_code"] = tn.E8M0_BIAS + act["exp"]
            act["description"] = f"static exponent {act['exp']}"
            if act.get("tensor"):
                tensors[act["tensor"]] = (tensors[act["tensor"]].to(torch.int64) + k).to(tensors[act["tensor"]].dtype)
        g["applied"] = True
        row["applied"] = True
        row["reason"] = None
    return tensors, policy, report


def _min_opt(a, b):
    return b if a is None else a if b is None else min(a, b)


def _max_opt(a, b):
    return b if a is None else a if b is None else max(a, b)


def _placement(g) -> dict:
    prod = g["producer"]
    out = {"kind": prod["kind"]}
    for f in ("tensor", "bias_tensor", "site"):
        if prod.get(f):
            out[f] = prod[f]
    return out


def annotate(policy: dict, groups: dict, info: dict, before: dict, *, enabled=True) -> dict:
    """Write `pack_fold` {k, producer, applied, ...} on every static site (Linear / Conv2d: site["pack_fold"]; attention operand:
    site["pack_folds"][operand]), refresh `pack_plan` records and `pack_plan_summary` from the folded exponents (classes from the
    graph), recompute `compiler_pack_scale_admission` and add `pack_fold_summary`. Mutates and returns `policy`."""
    sites_now = pp.static_sites(policy)
    for g in groups.values():
        for key in g["keys"]:
            st = sites_now[key]
            c_before = tn.E8M0_BIAS + before[key]["exp"]
            rec = {"schema": SCHEMA, "k": g["k"] if g["applied"] else g["planned_k_to_nearest"].get(key, 0),
                   "producer": g["class"], "placement": _placement(g), "applied": g["applied"],
                   "reason": g["reason"], "exp_before": before[key]["exp"], "exp_after": st["exp"],
                   "e8m0_code_before": c_before, "e8m0_code_after": tn.E8M0_BIAS + st["exp"],
                   "consumer_weight_exp_delta": -g["k"] if g["applied"] else 0,
                   "group_size": len(g["keys"]), "group_k": g["k"]}
            if g["reason"] == "in_range":
                rec["k"] = 0
            site = policy["sites"][st["site"]]
            if st["operand"]:
                site.setdefault("pack_folds", {})[st["operand"]] = rec
            else:
                site["pack_fold"] = rec
    plan = pp.build_plan(policy, producers_for_plan(info), _shared(groups))
    pp.attach(policy, plan)
    policy["compiler_pack_scale_admission"] = rebuild_admission(policy)
    policy["pack_fold_summary"] = summarize(policy, groups, before, plan, enabled=enabled)
    return policy


def _shared(groups: dict) -> dict:
    out = {}
    for g in groups.values():
        codes = sorted({c + (g["k"] if g["applied"] else 0) for c in g["codes"].values()})
        if len(g["keys"]) > 1:
            for key in g["keys"]:
                out[key] = {"id": g["src"], "size": len(g["keys"]), "codes": codes, "identical": len(codes) == 1}
    return out


def rebuild_admission(policy: dict) -> dict:
    """compiler_pack_scale_admission as quant_export builds it, from the (folded) policy."""
    static_act, weight_ranges = {}, {}
    for name, s in policy["sites"].items():
        if s.get("status") != "quantized":
            continue
        if s.get("kind") == "attention":
            for op, r in (s.get("operands") or {}).items():
                if r.get("mode") == "static":
                    static_act[f"{name}:{op}"] = [int(r["exp"])]
            continue
        a = s.get("activation") or {}
        if a.get("mode") == "static":
            static_act[name] = [int(a["exp"])]
        elif a.get("mode") == "static_per_step":
            static_act[name] = [int(e) for e in a["exps"]]
        rng = (s.get("weight") or {}).get("e8m0_code_range")
        if rng:
            weight_ranges[name] = (rng["min"], rng["max"])
    return tn.pack_admission_report(static_act, weight_ranges)


def summarize(policy, groups, before, plan, *, enabled=True) -> dict:
    lo, hi = pp.PACK_ADMISSION
    keys = sorted(before)
    out_before = [k for k in keys if not lo <= tn.E8M0_BIAS + before[k]["exp"] <= hi]
    now = pp.static_sites(policy)
    out_after = [k for k in keys if not lo <= tn.E8M0_BIAS + now[k]["exp"] <= hi]
    cls_of = {k: g["class"] for g in groups.values() for k in g["keys"]}
    folded = [k for g in groups.values() if g["applied"] for k in g["keys"]]
    left = {k: {"class": cls_of[k], "e8m0_code": tn.E8M0_BIAS + now[k]["exp"],
                "plan": {"k": plan[k]["fold"]["k"], "placement": plan[k]["placement"], "cost": plan[k]["cost"]},
                "reason": next(g["reason"] for g in groups.values() if k in g["keys"])} for k in out_after}
    return {"schema": SCHEMA, "enabled": enabled, "admitted_range": [lo, hi], "static_sites": len(keys),
            "outside_before": len(out_before), "outside_after": len(out_after),
            "in_range_before": len(keys) - len(out_before), "in_range_after": len(keys) - len(out_after),
            "sites_folded": len(folded), "groups_folded": sum(1 for g in groups.values() if g["applied"]),
            "folded_by_class": dict(sorted(collections.Counter(cls_of[k] for k in folded).items())),
            "outside_before_by_class": dict(sorted(collections.Counter(cls_of[k] for k in out_before).items())),
            "outside_after_by_class": dict(sorted(collections.Counter(cls_of[k] for k in out_after).items())),
            "outside_after_by_reason": dict(sorted(collections.Counter(v["reason"].split(":")[0] for v in left.values()).items())),
            "outside_after_by_cost": dict(sorted(collections.Counter(v["plan"]["cost"] for v in left.values()).items())),
            "outside_after_sites": dict(sorted(left.items())),
            "exact": "pack(2^k x, s + k) == pack(x, s) byte for byte; consumer weight_scale_exp -= k keeps 2^(sx + sw); "
                     "no calibration, no FP8 code changed",
            "pop_exponent_bound": pp.pop_exponent_bound(policy)}


# ---------------------------------------------------------------------------------------------
# Export directory transform
# ---------------------------------------------------------------------------------------------

def fold_export_dir(src, dst, graph_text: str, signature_text: str) -> dict:
    """Post-hoc, exact transform of an export directory (model.safetensors, policy.json, manifest.json) into `dst`: the folded
    tensors / policy and a manifest with recomputed hashes. `graph_text` / `signature_text` are fx/graph.txt and
    fx/graph_signature.txt of a native-FP8 capture of `src`. Returns the pack_fold_summary."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    import load_quantized
    import quant_export

    src, dst = Path(src), Path(dst)
    tensors, policy, manifest = load_quantized.read_export(src, verify=True)
    if "pack_fold_summary" in policy:
        raise ValueError(f"{src}: already folded (policy.json has pack_fold_summary)")
    with safe_open(str(src / "model.safetensors"), "pt") as f:
        meta = f.metadata()
    info = graph_info(graph_text, signature_text, policy, set(tensors))
    before = pp.static_sites(policy)
    groups = plan_folds(policy, info)
    new_tensors, new_policy, report = apply_folds(tensors, policy, groups)
    annotate(new_policy, groups, info, before)
    new_policy["pack_fold_summary"]["bf16_range"] = report["bf16_range"]
    new_policy["pack_fold_summary"]["groups"] = report["groups"]
    dst.mkdir(parents=True, exist_ok=True)
    changed = [k for k in tensors if not torch.equal(tensors[k], new_tensors[k])]
    save_file({k: v.contiguous() for k, v in new_tensors.items()}, str(dst / "model.safetensors"), metadata=meta)
    pf = dst / "policy.json"
    pf.write_text(json.dumps(new_policy, indent=2, sort_keys=False, allow_nan=False) + "\n")
    man = copy.deepcopy(manifest)
    man["files"] = {"model.safetensors": {"sha256": quant_export.sha256_file(dst / "model.safetensors"),
                                           "bytes": (dst / "model.safetensors").stat().st_size},
                    "policy.json": {"sha256": quant_export.sha256_file(pf), "bytes": pf.stat().st_size}}
    man["policy_sha256"] = man["files"]["policy.json"]["sha256"]
    man["pack_fold"] = {"schema": SCHEMA, "source_export": str(src),
                        "source_model_safetensors_sha256": manifest["files"]["model.safetensors"]["sha256"],
                        "source_policy_sha256": manifest["policy_sha256"], "tensors_changed": len(changed),
                        "summary": {k: v for k, v in new_policy["pack_fold_summary"].items()
                                    if k not in ("outside_after_sites", "groups", "pop_exponent_bound")}}
    (dst / "manifest.json").write_text(json.dumps(man, indent=2, allow_nan=False) + "\n")
    return new_policy["pack_fold_summary"]


def fold_export_from_fx_dir(src, dst, fx_dir) -> dict:
    fx = Path(fx_dir)
    return fold_export_dir(src, dst, (fx / "graph.txt").read_text(), (fx / "graph_signature.txt").read_text())


def pop_exponent_unchanged(before_policy: dict, after_policy: dict, tensors_before: dict, tensors_after: dict) -> bool:
    """Every static Linear's pop exponent sx + sw (per output row) is the same before and after the fold."""
    for name, st in pp.static_sites(before_policy).items():
        if st["kind"] == "attention":
            continue
        b, a = before_policy["sites"][name], after_policy["sites"][name]
        wb = tensors_before[b["weight"]["scale_exp"]].to(torch.int64) + st["exp"]
        wa = tensors_after[a["weight"]["scale_exp"]].to(torch.int64) + pp.static_sites(after_policy)[name]["exp"]
        if not torch.equal(wb, wa):
            return False
    return True


def capture_graph_text(model, policy: dict, seqlen: int = 8):
    """(graph.txt, graph_signature.txt) of a native-FP8 capture of a causal LM (in process, temporary directory). Only the
    producer classes are read from it, so a short sequence is enough."""
    import tempfile

    import fx_export

    vocab = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if not vocab:
        raise RuntimeError("--pack-fold: cannot capture the FX graph in process for a model without config.vocab_size; "
                           "capture it with src/fx_export.py and pass --pack-fold-graph <fx dir>")
    ids = torch.arange(seqlen, dtype=torch.long).reshape(1, seqlen) % vocab
    cases = [(ids,), ((ids + seqlen) % vocab,)]
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        fx = Path(d) / "fx"
        fx_export.capture_fx(fx_export.CausalLogits(model).eval(), cases, fx, policy=policy, fresh_process=False,
                             native_fp8="uint8")
        return (fx / "graph.txt").read_text(), (fx / "graph_signature.txt").read_text()


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Fold the pack exponents of an export into 122..128 (post-hoc, exact).")
    ap.add_argument("--export-dir", required=True, help="unfolded export (model.safetensors, policy.json, manifest.json)")
    ap.add_argument("--fx-dir", required=True, help="fx/ directory of a native-FP8 capture of that export (graph.txt, graph_signature.txt)")
    ap.add_argument("--out-dir", required=True, help="where the folded export is written (must not exist)")
    args = ap.parse_args(argv)
    if Path(args.out_dir).exists():
        ap.error(f"{args.out_dir} exists")
    s = fold_export_from_fx_dir(args.export_dir, args.out_dir, args.fx_dir)
    print(json.dumps({k: v for k, v in s.items() if k not in ("outside_after_sites", "groups", "pop_exponent_bound")}, indent=1))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
