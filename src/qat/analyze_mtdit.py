"""Contrasts and seed statistics for the Multi-Task DiT hardware-recipe QAT study (from eval_mtdit_hw.py's JSON).

TL;DR: reads the per-frame squared errors vs bf16 that eval_mtdit_hw.py stores and prints markdown for
RESULTS.md: per-model gap vs A and vs B, QAT seed mean +- std (C, D), and the five design contrasts
(A->B, A->C, B->D, C vs D, D vs E). Each contrast is a paired gap recovered: 1 - sum_f err_X(f) / sum_f err_Y(f)
over test frames (err = squared error of the chunk vs the original bf16 policy, summed over draws). The
SE is paired per frame and the CI comes from a frame bootstrap. A seed group (C, D) enters a contrast
"pooled": its per-frame error is the mean over its seeds.

  python src/qat/analyze_mtdit.py src/out/qat_mtdit/eval_mtdit_hw.json > src/out/qat_mtdit/contrasts.md
"""
import json
import math
import sys

import torch

E = json.loads(open(sys.argv[1]).read())
recs, frames = E["per_frame"]["records"], E["per_frame"]["frames"]
episodes = dict(zip(frames, E["per_frame"]["episodes"]))
R = E["results"]
N_BOOT = 4000
keyf = str if isinstance(next(iter(recs["A"])), str) else int


def fsum(name, key="sq"):
    return torch.tensor([sum(recs[name][keyf(f)][key]) for f in frames], dtype=torch.float64)


def group(prefix):
    return sorted(n for n in recs if n.startswith(prefix))


def err(spec, key="sq"):
    """per-frame error of a model name or of a pooled seed group ("C_*")."""
    if spec.endswith("*"):
        ns = group(spec[:-1])
        return torch.stack([fsum(n, key) for n in ns]).mean(0)
    return fsum(spec, key)


def paired(base, model, seed=0):
    d = base - model
    gap = float(d.sum() / base.sum())
    se = float(d.std() * math.sqrt(len(d)) / base.sum())
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(d), (N_BOOT, len(d)), generator=g)
    boots = 1.0 - model[idx].sum(1) / base[idx].sum(1)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return gap, se, lo, hi, float((d > 0).double().mean())


def rel_l2(spec):
    return math.sqrt(float(err(spec).sum() / err("bf16", "ref2").sum()))


def pc(x):
    return f"{100 * x:+.1f}%"


out = []
P = out.append
P(f"Test: {E['meta']['frames']} frames x {E['meta']['draws']} draws, {len(set(episodes.values()))} episodes. "
  "Gap = 1 - err(model)/err(reference), squared error vs the original bf16 policy, paired per frame.\n")

P("## Seed mean +- std (gap recovered, chunk)\n")
P("| group | seeds | rel_l2 per seed | gap vs A per seed | mean +- std vs A | gap vs B per seed | mean +- std vs B |")
P("|---|---|---|---|---|---|---|")
for g in ("C_", "D_"):
    ns = group(g)
    if not ns:
        continue
    ga = [paired(err("A"), err(n))[0] for n in ns]
    gb = [paired(err("B"), err(n))[0] for n in ns]
    sd = lambda v: (sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")  # noqa: E731
    P(f"| {g[:-1]} | {', '.join(ns)} | {', '.join(f'{rel_l2(n):.5f}' for n in ns)} | "
      f"{', '.join(pc(x) for x in ga)} | {100 * sum(ga) / len(ga):.1f}% +- {100 * sd(ga):.1f} | "
      f"{', '.join(pc(x) for x in gb)} | {100 * sum(gb) / len(gb):.1f}% +- {100 * sd(gb):.1f} |")

P("\n## Design contrasts (chunk; C and D pooled over seeds; per-seed rows below)\n")
P("| contrast | meaning | reference -> model | rel_l2 ref -> model | gap recovered +- SE | 95% CI | frames improved "
  "| denoiser gap |")
P("|---|---|---|---|---|---|---|---|")
C = [("A->B", "AWQ+GPTQ gain", "A", "B"), ("A->C", "QAT alone", "A", "C_*"),
     ("B->D", "QAT on top of what ships (headline)", "B", "D_*"),
     ("C vs D", "QAT with vs without AWQ+GPTQ underneath", "C_*", "D_*"),
     ("D vs E", "training through FP8 vs plain fine-tuning", "E_noquant", "D_*"),
     ("B->E", "plain fine-tuning (control) on top of B", "B", "E_noquant")]
for lab, mean, ref, mod in C:
    if any((s.endswith("*") and not group(s[:-1])) or (not s.endswith("*") and s not in recs) for s in (ref, mod)):
        continue
    g, se, lo, hi, imp = paired(err(ref), err(mod))
    has_v = bool(recs[ref.rstrip("*") if not ref.endswith("*") else group(ref[:-1])[0]][keyf(frames[0])]["vel"])
    vg = paired(err(ref, "vel"), err(mod, "vel"), seed=1)[0] if has_v else None
    P(f"| {lab} | {mean} | {ref} -> {mod} | {rel_l2(ref):.5f} -> {rel_l2(mod):.5f} | {pc(g)} +- {100 * se:.1f} "
      f"| [{pc(lo)}, {pc(hi)}] | {100 * imp:.0f}% | {pc(vg) if vg is not None else '—'} |")
P("\nPer seed:\n")
P("| contrast | model | gap +- SE | 95% CI |")
P("|---|---|---|---|")
for lab, ref, g_ in (("A->C", "A", "C_"), ("B->D", "B", "D_"), ("D vs E", "E_noquant", "D_")):
    if ref not in recs:
        continue
    for n in group(g_):
        g, se, lo, hi, _ = paired(err(ref), err(n))
        P(f"| {lab} | {n} | {pc(g)} +- {100 * se:.1f} | [{pc(lo)}, {pc(hi)}] |")

# per-dim: gap vs A for each model, plus B->D pooled per dim
P("\n## Per action dim (chunk)\n")
dims = len(R["A"]["chunk"]["rel_l2_per_dim"])


def sq_dim(spec):
    ns = group(spec[:-1]) if spec.endswith("*") else [spec]
    return torch.stack([torch.tensor([[sum(x[k] for x in recs[n][keyf(f)]["sq_dim"]) for k in range(dims)]
                                      for f in frames], dtype=torch.float64).sum(0) for n in ns]).mean(0)


P("| row | " + " | ".join(f"a{k}" for k in range(dims)) + " |")
P("|---|" + "---|" * dims)
for n in ("A", "B"):
    P(f"| {n} rel_l2 | " + " | ".join(f"{x:.4f}" for x in R[n]["chunk"]["rel_l2_per_dim"]) + " |")
for lab, ref, mod in (("A->B", "A", "B"), ("A->C (pooled)", "A", "C_*"), ("B->D (pooled)", "B", "D_*"),
                      ("B->E", "B", "E_noquant")):
    if (mod.endswith("*") and not group(mod[:-1])) or (not mod.endswith("*") and mod not in recs):
        continue
    v = 1 - sq_dim(mod) / sq_dim(ref)
    P(f"| gap {lab} | " + " | ".join(pc(x) for x in v.tolist()) + " |")

P("\n## Per test episode (chunk gap; frames per episode in the header row)\n")
eps = sorted(set(episodes.values()))
P("| contrast | " + " | ".join(f"ep {e}" for e in eps) + " |")
P("|---|" + "---|" * len(eps))
P("| frames | " + " | ".join(str(sum(1 for f in frames if episodes[f] == e)) for e in eps) + " |")
for lab, ref, mod in (("A->B", "A", "B"), ("A->C", "A", "C_*"), ("B->D", "B", "D_*"), ("B->E", "B", "E_noquant")):
    if (mod.endswith("*") and not group(mod[:-1])) or (not mod.endswith("*") and mod not in recs):
        continue
    rb, rm = err(ref), err(mod)
    row = []
    for e in eps:
        m = torch.tensor([episodes[f] == e for f in frames])
        row.append(pc(float(1 - rm[m].sum() / rb[m].sum())))
    P(f"| {lab} | " + " | ".join(row) + " |")

# concentration: how much of A's / B's error the top frames carry (explains the noise)
P("\n## Error concentration\n")
for n in ("A", "B"):
    s = err(n).sort(descending=True).values
    P(f"- {n}: the top 1% / 5% / 10% of frames carry {100 * float(s[:max(1, len(s) // 100)].sum() / s.sum()):.0f}% / "
      f"{100 * float(s[:max(1, len(s) // 20)].sum() / s.sum()):.0f}% / "
      f"{100 * float(s[:max(1, len(s) // 10)].sum() / s.sum()):.0f}% of its squared error.")
print("\n".join(out))
