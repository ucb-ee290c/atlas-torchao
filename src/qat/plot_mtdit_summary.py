"""One self-contained summary figure for the Multi-Task DiT hardware-recipe QAT study.

TL;DR: a forest plot of the design contrasts (A->B, A->C, B->D, C vs D, D vs E, B->E) on the final held-out
eval, in two panels with their own x-axes: the free-running action chunk and the teacher-forced denoiser
output. Dot = paired gap recovered, thick bar = +-1 SE, thin line = 95% frame-bootstrap CI.

  python src/qat/plot_mtdit_summary.py src/out/qat_mtdit/eval_mtdit_hw_s0.json src/out/qat_mtdit/plots/summary.png
"""
import json
import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

E = json.loads(open(sys.argv[1]).read())
OUT = sys.argv[2]
recs, frames = E["per_frame"]["records"], E["per_frame"]["frames"]
keyf = str if isinstance(next(iter(recs["A"])), str) else int
meta = E["meta"]
n_eps = len(set(E["per_frame"]["episodes"]))


def group(p):
    return sorted(n for n in recs if n.startswith(p))


def err(spec, key):
    ns = group(spec[:-1]) if spec.endswith("*") else [spec]
    return torch.stack([torch.tensor([sum(recs[n][keyf(f)][key]) for f in frames], dtype=torch.float64)
                        for n in ns]).mean(0)


def paired(base, model, seed=0, nb=4000):
    d = base - model
    gap = float(d.sum() / base.sum())
    se = float(d.std() * math.sqrt(len(d)) / base.sum())
    idx = torch.randint(0, len(d), (nb, len(d)), generator=torch.Generator().manual_seed(seed))
    boots = 1.0 - model[idx].sum(1) / base[idx].sum(1)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return 100 * gap, 100 * se, 100 * lo, 100 * hi


def label(s):
    return s.replace("_*", "")


ROWS = [("A → B", "AWQ+GPTQ (what ships) vs RTN", "A", "B"),
        ("A → C", "QAT alone, from RTN", "A", "C_*"),
        ("B → D", "QAT on top of what ships", "B", "D_*"),
        ("C vs D", "QAT with vs without AWQ+GPTQ under it", "C_*", "D_*"),
        ("D vs E", "QAT through FP8 vs plain fine-tuning", "E_noquant", "D_*"),
        ("B → E", "control: plain fine-tuning on B", "B", "E_noquant")]
HEADLINE = "B → D"
seeds = {g: len(group(g)) for g in ("C_", "D_")}

INK, INK2, MUTED, GRID, SURF, BLUE, BAND = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb", "#2a78d6", "#eef3fb"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "xtick.color": INK2, "ytick.color": INK2})

fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True, gridspec_kw={"wspace": 0.08})
fig.patch.set_facecolor(SURF)
ys = list(range(len(ROWS)))[::-1]
panels = [("sq", "Free-running action chunk (100-step DDPM sampling)", (-110, 75)),
          ("vel", "Teacher-forced denoiser output (ε at 10 DDPM steps)", (-3, 22))]
for ax, (key, title, xl) in zip(axes, panels):
    ax.set_facecolor(SURF)
    for y, (lab, _, ref, mod) in zip(ys, ROWS):
        if lab == HEADLINE:
            ax.axhspan(y - 0.45, y + 0.45, color=BAND, zorder=0, lw=0)
        g, se, lo, hi = paired(err(ref, key), err(mod, key), seed=1 if key == "vel" else 0)
        ax.plot([max(lo, xl[0]), min(hi, xl[1])], [y, y], color=BLUE, lw=1.2, alpha=0.55, solid_capstyle="round",
                zorder=2)
        if lo < xl[0]:
            ax.annotate("", xy=(xl[0], y), xytext=(xl[0] + 6, y),
                        arrowprops=dict(arrowstyle="-|>", color=BLUE, alpha=0.55, lw=1.2))
        ax.plot([g - se, g + se], [y, y], color=BLUE, lw=5, solid_capstyle="round", zorder=3)
        ax.plot(g, y, "o", ms=9, color=BLUE, mec=SURF, mew=2, zorder=4)
        ax.text(g, y + 0.27, f"{g:+.1f}% ± {se:.1f}", ha="center", va="bottom", fontsize=9,
                color=INK, fontweight="bold" if lab == HEADLINE else "normal")
    ax.axvline(0, color=INK2, lw=1, zorder=1)
    ax.set_xlim(*xl)
    ax.set_ylim(-0.6, len(ROWS) - 0.35)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8)
    ax.set_xlabel("gap recovered (%)  ←  worse   |   better  →", color=INK2)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))

axes[0].set_yticks(ys)
axes[0].set_yticklabels([f"{lab}\n{desc}" for lab, desc, _, _ in ROWS], fontsize=10)
for t, (lab, *_r) in zip(axes[0].get_yticklabels(), ROWS):
    t.set_color(INK); t.set_linespacing(1.4)
    if lab == HEADLINE:
        t.set_fontweight("bold")

seed_txt = ", ".join(f"{g[0]}: {n} seed{'s' if n > 1 else ''}" for g, n in seeds.items())
fig.suptitle("Multi-Task DiT, FP8 hardware recipe: AWQ+GPTQ closes the most gap; QAT on top adds a small,\n"
             "precise gain on the denoiser output that the noisy chunk metric can't resolve",
             x=0.01, ha="left", fontsize=12.5, color=INK, fontweight="bold", y=0.995)
fig.text(0.01, 0.885,
         f"Held-out test: {meta['frames']} frames × {meta['draws']} noise draws from {n_eps} episodes. "
         "Gap recovered = 1 − err(model)/err(reference), squared error vs the original bf16 policy, paired per frame.\n"
         "A = RTN (per-32-row panel + static activations), B = shipped AWQ+GPTQ recipe, C = A + QAT, D = B + QAT, "
         f"E = B + plain bf16 fine-tuning (same settings as D). {seed_txt}.",
         ha="left", va="top", fontsize=9, color=INK2)
fig.text(0.01, 0.015,
         "Dot = gap recovered, thick bar = ±1 SE (paired per frame), thin line = 95% frame-bootstrap CI "
         "(arrow = extends off the axis).\nPanels have separate x-scales. Chunk error is heavy-tailed "
         "(top 5% of frames carry ~2/3 of it), so its intervals are wide.",
         ha="left", va="bottom", fontsize=8.5, color=MUTED)
fig.subplots_adjust(left=0.25, right=0.985, top=0.76, bottom=0.16)
fig.savefig(OUT, dpi=170, facecolor=SURF)
print("wrote", OUT)
