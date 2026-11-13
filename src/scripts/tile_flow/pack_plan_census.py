"""Pack-plan census of existing tile static exports, READ-ONLY on the exports: recomputes the per-site pack plan from each
bundle's export/policy.json and fx/graph.txt (+ fx/graph_signature.txt) with src/exporter/pack_plan.py and writes a JSON and a markdown
table to the given output paths (never into the exports root). Compares with the research census
(docs/algos/tile-aware.md#pack-scale-admission-range) and prints every mismatch.
usage: PYTHONPATH=src python pack_plan_census.py EXPORTS_ROOT OUT_JSON OUT_MD [OUT_MD ...]"""
import json
import sys
from pathlib import Path

import exporter.pack_plan as pack_plan

BUNDLE = "atlas_v1_awq_tile_static"
MODELS = ("tinyllama", "smolvla", "multitask-dit", "pi0")
# research census (tile-aware.md#pack-scale-admission-range): static sites, outside 122..128, attention p, no-added-instruction (free) sites
RESEARCH = {"tinyllama": dict(sites=176, outside=114, p=22, free=92), "smolvla": dict(sites=316, outside=244, p=13, free=221),
            "multitask-dit": dict(sites=203, outside=88, p=24, free=24), "pi0": dict(sites=483, outside=127, p=63, free=61)}
RESEARCH_CLASS = {   # producer class -> outside sites (T, S, D, P), research census table (norm = RMSNorm + LayerNorm rows)
    "norm_affine": (59, 145, 0, 24), "layer_norm": (0, 0, 0, 3), "attn_p": (22, 13, 24, 63), "attn_out_pv": (22, 44, 24, 27),
    "attn_out_sdpa": (0, 0, 6, 0), "gated_product": (11, 24, 0, 7), "x_sigmoid_product": (0, 0, 17, 0),
    "bare_activation": (0, 6, 13, 1), "linear_to_linear": (0, 9, 0, 0), "modulation_add": (0, 1, 1, 0),
    "raw_input": (0, 2, 1, 2), "other:aten.sin.default": (0, 0, 1, 0), "other:aten.index.Tensor": (0, 0, 1, 0)}


def census(root):
    out = {}
    for m in MODELS:
        b = Path(root) / m / BUNDLE
        policy = json.loads((b / "export" / "policy.json").read_text())
        plan = pack_plan.plan_from_files(policy, (b / "fx" / "graph.txt").read_text(),
                                         (b / "fx" / "graph_signature.txt").read_text())
        out[m] = {"summary": pack_plan.summarize(plan, policy), "policy_sites_outside": policy["compiler_pack_scale_admission"]["sites_outside"],
                  "policy_sites_checked": policy["compiler_pack_scale_admission"]["sites_checked"]}
    return out


def mismatches(out):
    bad = []
    for i, m in enumerate(MODELS):
        s, r = out[m]["summary"], RESEARCH[m]
        got = dict(sites=s["static_sites"], outside=s["sites_outside"], p=s["outside_by_class"].get("attn_p", 0),
                   free=s["free_fold_sites"])
        bad += [f"{m} {k}: plan {got[k]} vs research {r[k]}" for k in r if got[k] != r[k]]
        if (out[m]["policy_sites_checked"], out[m]["policy_sites_outside"]) != (s["static_sites"], s["sites_outside"]):
            bad.append(f"{m}: plan sites {s['static_sites']}/{s['sites_outside']} vs compiler_pack_scale_admission "
                       f"{out[m]['policy_sites_checked']}/{out[m]['policy_sites_outside']}")
        for c, v in RESEARCH_CLASS.items():
            if s["outside_by_class"].get(c, 0) != v[i]:
                bad.append(f"{m} class {c}: plan {s['outside_by_class'].get(c, 0)} vs research {v[i]}")
    return bad


def markdown(out, bad):
    cols = ["static sites", "outside 122..128", "free fold (no added instruction)", "vmul per p row block",
            "vmul per tile", "act-constant rewrite (unverified)", "host", "unknown producer", "code range", "k range (nearest in-range)",
            "shared-input groups differing (of multi)"]
    rows = {m: out[m]["summary"] for m in MODELS}
    cost = lambda s, c: s["outside_by_cost"].get(c, 0)  # noqa: E731
    vals = {"static sites": lambda s: s["static_sites"], "outside 122..128": lambda s: s["sites_outside"],
            "free fold (no added instruction)": lambda s: s["free_fold_sites"],
            "vmul per p row block": lambda s: cost(s, "vmul_per_row_block"), "vmul per tile": lambda s: cost(s, "vmul_per_tile"),
            "act-constant rewrite (unverified)": lambda s: cost(s, "act_constant_rewrite"), "host": lambda s: cost(s, "host"),
            "unknown producer": lambda s: s["outside_unknown_producer"],
            "code range": lambda s: "%d..%d" % tuple(s["code_range"]), "k range (nearest in-range)": lambda s: "%d..%d" % tuple(s["k_range"]),
            "shared-input groups differing (of multi)": lambda s: "%d of %d" % (s["shared_input_groups"]["groups_with_differing_codes"],
                                                                                 s["shared_input_groups"]["multi_consumer_groups"])}
    L = ["| | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    L += [f"| {c} | " + " | ".join(str(vals[c](rows[m])) for m in MODELS) + " |" for c in cols]
    classes = sorted({c for m in MODELS for c in rows[m]["outside_by_class"]})
    L += ["", "Outside-range sites by producer class:", "", "| producer class | " + " | ".join(MODELS) + " |", "|---|" + "---|" * len(MODELS)]
    L += [f"| {c} | " + " | ".join(str(rows[m]["outside_by_class"].get(c, "-")) for m in MODELS) + " |" for c in classes]
    L += ["", "Mismatches against the research census: " + ("none" if not bad else ""), ""] + [f"- {b}" for b in bad]
    return "\n".join(L) + "\n"


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    root, out_json, outs_md = a[0], Path(a[1]), [Path(x) for x in a[2:]]
    if Path(root).resolve() in [Path(x).resolve().parent for x in [out_json, *outs_md]]:
        raise SystemExit("refusing to write into the exports root")
    out = census(root)
    bad = mismatches(out)
    out_json.write_text(json.dumps({"exports_root": str(root), "mismatches_vs_research": bad, "models": out}, indent=1))
    md = markdown(out, bad)
    for p in outs_md:
        p.write_text(md)
    print(md)
    return not bad


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
