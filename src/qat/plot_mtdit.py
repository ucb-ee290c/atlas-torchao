"""Multi-Task DiT hardware-recipe QAT plots (src/out/qat_mtdit), adapted from plot_progress_libero.py.

TL;DR: val curves and linear training loss from each run's qat_log.json (falls back to train.log), and,
from eval_mtdit_hw.py's JSON, a "gap recovered vs A" bar chart plus a per-dimension chart.

  python src/qat/plot_mtdit.py src/out/qat_mtdit [src/out/qat_mtdit/eval_mtdit_hw.json]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

Q = Path(sys.argv[1] if len(sys.argv) > 1 else "src/out/qat_mtdit")
EVAL = Path(sys.argv[2]) if len(sys.argv) > 2 else Q / "eval_mtdit_hw.json"
OUT = Q / "plots"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#7d7c78"
DIMS = ["a0", "a1", "a2", "a3", "a4", "a5", "a6"]   # TRI-ML/PretrainFinetune names its 7 action dims a0..a6
# (run dir, label, color, line style, marker): one hue per recipe, seeds told apart by marker
RUNS = [("C_s0", "C: RTN→QAT seed 0", BLUE, "-", "o"), ("C_s1", "C: RTN→QAT seed 1", BLUE, "-", "s"),
        ("C_s2", "C: RTN→QAT seed 2", BLUE, "-", "^"),
        ("D_s0", "D: AWQ+GPTQ→QAT seed 0", ORANGE, "-", "o"), ("D_s1", "D: AWQ+GPTQ→QAT seed 1", ORANGE, "-", "s"),
        ("D_s2", "D: AWQ+GPTQ→QAT seed 2", ORANGE, "-", "^"),
        ("E_noquant", "E: no-quant control (D_s0 settings)", GRAY, "--", "o")]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "legend.frameon": False, "text.color": INK})


def load(run):
    d = Q / run
    if (d / "qat_log.json").exists():
        L = json.loads((d / "qat_log.json").read_text())
        if L.get("steps"):
            return [(v["step"], v.get("gap_recovered", 0.0), v.get("gap_recovered_se") or 0) for v in L["val"]], \
                   [s["loss"] for s in L["steps"]]
    if not (d / "train.log").exists():
        return None
    txt = (d / "train.log").read_text()
    val = [(0, 0.0, 0.0)] + [(int(a), float(b) / 100, float(c) / 100) for a, b, c in
                             re.findall(r"val step (\d+): rel_l2 [\d.]+, gap recovered (-?[\d.]+)% \+- ([\d.]+)", txt)]
    loss = [float(x) for x in re.findall(r"step \d+/\d+ loss ([\d.e+-]+)", txt)]
    return (val, loss) if loss else None


data = {r: load(r) for r, *_ in RUNS}
runs = [r for r in RUNS if data[r[0]]]

# 1. val gap recovered over training (one panel per start point: the --init runs' val is relative to AWQ+GPTQ)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
for ax, fam, title in [(axes[0], ("C_",), "C: start A (val gap vs A)"),
                       (axes[1], ("D_", "E_"), "D, E: start B (val gap vs B)")]:
    ax.axhline(0, color=INK2, lw=1)
    for run, lab, c, ls, mk in runs:
        if not run.startswith(fam):
            continue
        v = np.array(data[run][0])
        ax.fill_between(v[:, 0], 100 * (v[:, 1] - v[:, 2]), 100 * (v[:, 1] + v[:, 2]), color=c, alpha=0.06, lw=0)
        ax.plot(v[:, 0], 100 * v[:, 1], ls, color=c, lw=2, marker=mk, ms=6, label=lab)
    ax.set_xlabel("optimizer step")
    ax.set_title(title, loc="left", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
axes[0].set_ylabel("val gap recovered (% of squared error)")
fig.suptitle("Val (128 frames, 4 episodes; bands ±1 SE): for watching training only, claims come from eval_mtdit_hw",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "val_gap_curves.png", dpi=160)
plt.close(fig)

# 2. training loss, linear axis: raw steps (faint) + 50-step mean
fig, ax = plt.subplots(figsize=(10, 5))
k = 50
for run, lab, c, ls, mk in runs:
    l = np.array(data[run][1])
    if len(l) < k:
        continue
    ax.scatter(np.arange(1, len(l) + 1), l, s=5, color=c, alpha=0.10, lw=0)
    sm = np.convolve(l, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, len(l) + 1), sm, ls, color=c, lw=2,
            label=f"{lab}: {l[:k].mean():.4f} → {l[-k:].mean():.4f} ({100 * (l[-k:].mean() / l[:k].mean() - 1):+.0f}%)")
allv = np.concatenate([np.array(data[r][1]) for r, *_ in runs]) if runs else np.array([1.0])
ax.set_ylim(0, np.quantile(allv, 0.99) * 1.1)
ax.set_xlabel("optimizer step")
ax.set_ylabel("training loss (teacher-forced epsilon MSE)")
ax.set_title("Training loss: dots = each step, line = 50-step mean (the control trains without FP8 casts)",
             loc="left", fontsize=11)
ax.legend(loc="upper right", fontsize=8.5, title="first-50 mean → last-50 mean", title_fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "train_loss_linear.png", dpi=160)
plt.close(fig)
print("wrote val_gap_curves.png, train_loss_linear.png")

if not EVAL.exists():
    sys.exit(0)
E = json.loads(EVAL.read_text())
R, meta = E["results"], E["meta"]
n_chunks = meta["frames"] * meta["draws"]


def color_of(name):
    return {"E_noquant": GRAY, "B": AQUA, "A": INK2}.get(name, ORANGE if name.startswith("D_") else BLUE)


# 3. gap recovered vs default FP8 (ref_conv), every checkpoint, ±1 SE and the bootstrap CI
BASE = meta["baseline"]
names = [n for n in R if n not in ("bf16", BASE)]
fig, ax = plt.subplots(figsize=(10, 0.55 * len(names) + 1.8))
y = np.arange(len(names))[::-1]
for yi, n in zip(y, names):
    c = R[n]["chunk"]
    g, se, (lo, hi) = 100 * c["gap_recovered"], 100 * c["se"], c["ci95"]
    ax.barh(yi, g, 0.6, color=color_of(n), alpha=0.9)
    ax.plot([100 * lo, 100 * hi], [yi, yi], color=INK2, lw=1)
    ax.errorbar(g, yi, xerr=se, fmt="none", ecolor=INK, capsize=4, lw=1.6)
    ax.text(max(g, 100 * hi, 0) + 1.5, yi, f"{g:.1f}% ± {se:.1f}  (rel_l2 {c['rel_l2']:.4f})",
            va="center", fontsize=9)
ax.axvline(0, color=INK, lw=1.2)
ax.set_yticks(y, names)
ax.grid(axis="y", visible=False)
xs = [100 * R[n]["chunk"]["ci95"][1] for n in names] + [100 * R[n]["chunk"]["gap_recovered"] for n in names]
ax.set_xlim(min(0, min(100 * R[n]["chunk"]["ci95"][0] for n in names)) - 3, max(xs) + 32)
ax.set_xlabel(f"gap recovered vs {BASE} = share of {BASE}'s squared error vs bf16 removed (%)")
ax.set_title(f"Held-out test, {meta['frames']} frames × {meta['draws']} draws = {n_chunks} chunks; "
             f"{BASE} rel_l2 {R[BASE]['chunk']['rel_l2']:.4f}\n"
             "black bar = ±1 SE (paired per frame), gray line = 95% frame-bootstrap CI",
             loc="left", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "gap_recovered_test.png", dpi=160)
plt.close(fig)

# 4. per-dimension gap recovered: QAT seed mean, AWQ+GPTQ, AWQ+GPTQ->QAT seed mean, control
groups = [("B: AWQ+GPTQ", [n for n in names if n == "B"], AQUA),
          ("C: RTN→QAT (mean of seeds)", [n for n in names if n.startswith("C_")], BLUE),
          ("D: AWQ+GPTQ→QAT (mean)", [n for n in names if n.startswith("D_")], ORANGE),
          ("E: no-quant control", [n for n in names if n == "E_noquant"], GRAY)]
groups = [g for g in groups if g[1]]
ndim = len(R[BASE]["chunk"]["rel_l2_per_dim"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 2]})
a1.bar(np.arange(ndim), R[BASE]["chunk"]["rel_l2_per_dim"], 0.6, color=GRAY)
a1.set_xticks(np.arange(ndim), DIMS[:ndim])
a1.set_ylabel("rel_l2 vs bf16")
a1.set_title(f"{BASE} error per action dim", loc="left", fontsize=11)
w = 0.8 / len(groups)
for j, (lab, ns, c) in enumerate(groups):
    vals = 100 * np.mean([R[n]["chunk"]["gap_per_dim"] for n in ns], axis=0)
    a2.bar(np.arange(ndim) - 0.4 + w * (j + 0.5), vals, w * 0.92, color=c, label=lab)
a2.axhline(0, color=INK, lw=1)
a2.set_xticks(np.arange(ndim), DIMS[:ndim])
a2.set_ylabel("gap recovered, squared error (%)")
a2.set_title(f"Gap recovered vs {BASE} per action dim", loc="left", fontsize=11)
a2.legend(fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(OUT / "gap_per_dim_test.png", dpi=160)
plt.close(fig)
print("wrote gap_recovered_test.png, gap_per_dim_test.png")
