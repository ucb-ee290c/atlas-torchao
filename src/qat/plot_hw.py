"""Presentation plots for the main experiment (QAT under the FX recipe atlas_v1_awq_tile_static).

TL;DR: reads src/out/qat_hw/eval_hw.json and each run's qat_log.json and writes self-contained PNGs
(the title states the takeaway, the subtitle defines the metric) to src/out/qat_hw/plots/.
Usage: python src/qat/plot_hw.py [--dir src/out/qat_hw]
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COL = {"A": "#7a7a7a", "B": "#d9822b", "C": "#86a9e0", "D": "#1f4fb4", "E": "#a99bc2"}
LABEL = {
    "A": "A  plain FP8\n(round-to-nearest)",
    "B": "B  AWQ+GPTQ\n(what FX ships)",
    "C": "C  QAT\nstarting from A",
    "D": "D  QAT\nstarting from B",
    "E": "E  control: D's training\nwith FP8 turned off",
}
SEEDS = ["s0", "s1", "s2"]
DIMS = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
FOOT = ("SmolVLA-LIBERO (lerobot/smolvla_libero) | FP8 numerics = ref_conv + atlas_v1_awq_tile_static (the FX export) | "
        "500 held-out test frames, 32 episodes")

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 14})


def models(arm):
    return [f"{arm}_{s}" for s in SEEDS] if arm in "CD" else [arm]


def finish(fig, path, title, subtitle):
    fig.suptitle(title, x=0.01, ha="left", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10.5, color="#333")
    fig.subplots_adjust(top=0.80, bottom=0.16)
    fig.text(0.01, 0.01, FOOT, ha="left", fontsize=8, color="#777")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def p1_setup(out):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.axis("off")
    rows = [("A", "Plain FP8", "no", "no", "the starting point: what FP8 costs with no tricks"),
            ("B", "AWQ+GPTQ", "yes", "no", "what the FX exporter ships today"),
            ("C", "QAT from A", "no", "yes (3 seeds)", "does training alone help?"),
            ("D", "QAT from B", "yes", "yes (3 seeds)", "does training help ON TOP of what ships?"),
            ("E", "Control", "yes", "same training, FP8 off", "is D's gain just extra fine-tuning?")]
    hdr = ["", "model", "AWQ+GPTQ", "QAT", "question it answers"]
    xs = [0.0, 0.06, 0.22, 0.36, 0.62]
    for x, h in zip(xs, hdr):
        ax.text(x, 0.93, h, fontweight="bold", fontsize=11.5, transform=ax.transAxes)
    for i, r in enumerate(rows):
        y = 0.78 - i * 0.16
        ax.add_patch(plt.Rectangle((0.0, y - 0.045), 0.04, 0.11, color=COL[r[0]], transform=ax.transAxes))
        ax.text(0.02, y + 0.01, r[0], ha="center", va="center", color="white", fontweight="bold",
                transform=ax.transAxes)
        for x, t in zip(xs[1:], r[1:]):
            ax.text(x, y, t, va="center", fontsize=11, transform=ax.transAxes,
                    fontweight="bold" if r[0] == "D" and x == xs[-1] else "normal")
    finish(fig, out / "01_setup.png", "Experiment: five versions of the same FP8 model",
           "All five run the exact FP8 numerics the FX exporter ships (bit-exact). They differ only in the weights.\n"
           "QAT = a short fine-tune (500 steps) where the model sees its own FP8 rounding and learns to match the "
           "original bf16 model.")


def p2_error(r, out):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    arms = ["A", "B", "C", "D", "E"]
    a = r["A"]["vel"]["mse"]
    for i, arm in enumerate(arms):
        v = [r[m]["vel"]["mse"] for m in models(arm)]
        m = np.mean(v)
        ax.bar(i, m, width=0.6, color=COL[arm])
        if len(v) > 1:
            ax.scatter([i] * 3, v, color="black", s=18, zorder=3, label="individual seed" if arm == "C" else None)
        lab = f"{m * 1e3:.1f}" + ("" if arm == "A" else f"\n−{100 * (1 - m / a):.0f}% vs A")
        ax.text(i, m + 0.0004, lab, ha="center", va="bottom", fontsize=10.5,
                fontweight="bold" if arm == "D" else "normal")
    ax.axhline(a, color=COL["A"], ls="--", lw=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels([LABEL[k] for k in arms], fontsize=10)
    ax.set_ylabel("error vs original bf16 model\n(velocity MSE ×10⁻³, lower is better)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x * 1e3:.0f}"))
    ax.set_ylim(0, a * 1.25)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    finish(fig, out / "02_error_per_model.png",
           "QAT on top of AWQ+GPTQ (D) gives the lowest FP8 error",
           "Bar height = how far each FP8 model's denoising velocity is from the original bf16 model's (averaged over "
           "500 test frames).\nThe bf16 model scores 0. C and D are the mean of 3 training seeds (dots).")


def p3_vs_shipped(r, out):
    fig, ax = plt.subplots(figsize=(11, 5.4))
    arms = ["C", "D", "E"]
    for i, arm in enumerate(arms):
        ms = models(arm)
        g = [100 * r[m]["vs"]["B"]["vel"]["gap_recovered"] for m in ms]
        se = [100 * r[m]["vs"]["B"]["vel"]["se"] for m in ms]
        mean = np.mean(g)
        ax.bar(i, mean, width=0.6, color=COL[arm])
        off = np.linspace(-0.15, 0.15, len(g)) if len(g) > 1 else [0]
        ax.errorbar(i + np.array(off), g, yerr=se, fmt="o", color="black", ms=5, capsize=3, lw=1,
                    label="one trained model ± standard error" if arm == "C" else None)
        ax.text(i + 0.33, mean, f"{mean:+.1f}%", va="center", fontsize=13, fontweight="bold", color=COL[arm])
    ax.axhline(0, color=COL["B"], lw=2)
    ax.text(-0.45, -0.25, "0 = same as the shipped recipe (B)", ha="left", va="top", color=COL["B"], fontsize=10)
    ax.set_xticks(range(3))
    ax.set_xticklabels([LABEL[k] for k in arms], fontsize=10)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylabel("% of B's remaining FP8 error removed\n(higher is better)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    finish(fig, out / "03_gain_over_shipped.png",
           "QAT removes ~11% more FP8 error than the shipped recipe; plain fine-tuning removes ~0%",
           "Each bar: how much of B's error (vs bf16) the model removes, paired frame by frame on 500 test frames "
           "(velocity metric).\nD beats B in all 3 seeds; the control E, trained identically but without FP8, does "
           "not. So the gain comes from training through FP8.")


def p4_dims(r, out):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    arms = ["B", "C", "D", "E"]
    w = 0.2
    x = np.arange(len(DIMS))
    for j, arm in enumerate(arms):
        g = np.mean([[100 * v for v in r[m]["chunk"]["gap_per_dim"]] for m in models(arm)], axis=0)
        ax.bar(x + (j - 1.5) * w, g, width=w * 0.9, color=COL[arm], label=LABEL[arm].replace("\n", " "))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_xlabel("action dimension (end-effector motion + gripper)")
    ax.set_ylabel("% of plain-FP8 (A) error removed\n(higher is better)")
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper left")
    ax.set_ylim(0, 40)
    finish(fig, out / "04_per_action_dim.png",
           "QAT on top of AWQ+GPTQ (D, dark blue) helps every action dimension by ≥17%",
           "For each of the 7 output actions: how much of plain FP8's error (vs bf16) is removed, over the 50-step action "
           "chunks of 500 test frames.\nC and D bars are the mean of 3 seeds. Orange (AWQ+GPTQ alone) is ~0% on roll and yaw (wrist rotation). Per-dimension numbers use the\nnoisy action-chunk metric, so read the pattern, not exact values.")


def p5_training(qdir, out):
    fig, ax = plt.subplots(figsize=(11, 5.0))
    k = 50
    for arm in ["C", "D"]:
        bins = []
        for m in models(arm):
            y = np.array([s["loss"] for s in json.loads((qdir / m / "qat_log.json").read_text())["steps"]]) * 1e3
            bins.append(y[:len(y) // k * k].reshape(-1, k).mean(1))
        bins = np.array(bins)
        x = (np.arange(bins.shape[1]) + 0.5) * k
        mean = bins.mean(0)
        ax.fill_between(x, bins.min(0), bins.max(0), color=COL[arm], alpha=0.2, lw=0)
        ax.plot(x, mean, "o-", color=COL[arm], lw=2.5, ms=5, label=LABEL[arm].replace("\n", " "))
        drop = 100 * (1 - mean[-2:].mean() / mean[:2].mean())
        ax.text(x[-1] + 10, mean[-1] + (0.6 if arm == "C" else -0.6), f"{mean[-1]:.1f} (−{drop:.0f}% vs start)",
                va="bottom" if arm == "C" else "top", fontsize=10,
                color=COL[arm], fontweight="bold")
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 30)
    ax.set_xlabel("optimizer step (each step = 8 training frames; 500 steps total)")
    ax.set_ylabel("training loss: FP8 student vs bf16 teacher\n(velocity MSE ×10⁻³, lower is better)")
    ax.legend(frameon=False, loc="lower left")
    finish(fig, out / "05_training_loss.png",
           "Training loss falls ~15-17% over 500 steps; D (on top of AWQ+GPTQ) stays below C throughout",
           "Each point = average loss over 50 steps, mean of 3 seeds (shaded = seed range). Per-step loss is very noisy "
           "(random frame, noise and\ntime step each step), so it is averaged. The control E is not shown: with FP8 off "
           "its loss measures a different thing.")


def p6_chunk(r, out):
    fig, ax = plt.subplots(figsize=(11, 5.2))
    names = ["B"] + models("C") + models("D") + ["E"]
    for i, m in enumerate(names):
        c = r[m]["chunk"]
        g, lo, hi = 100 * c["gap_recovered"], 100 * c["ci95"][0], 100 * c["ci95"][1]
        arm = m[0]
        ax.plot([lo, hi], [i, i], color=COL[arm], lw=3, solid_capstyle="round")
        ax.plot(g, i, "o", color="black", ms=6)
        ax.text(hi + 1.5, i, f"{g:.0f}%  [{lo:.0f}, {hi:.0f}]", va="center", fontsize=9.5)
    ax.axvline(0, color=COL["A"], ls="--")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("_s", " seed ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("% of plain-FP8 (A) error removed on the final action chunk (dot = estimate, bar = 95% CI)")
    ax.set_xlim(-25, 70)
    finish(fig, out / "06_action_chunk_noisy.png",
           "On the final action chunk every model beats plain FP8, but the noise is too large to rank them",
           "Error of the full 50-step action chunk vs bf16 (random noise draw included). A few test episodes have "
           "huge swings,\nso every 95% interval spans ~40 points. That is why the claims use the velocity metric "
           "(intervals ~±3 points).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="src/out/qat_hw")
    a = ap.parse_args()
    qdir = Path(a.dir)
    out = qdir / "plots"
    out.mkdir(exist_ok=True)
    r = json.loads((qdir / "eval_hw.json").read_text())["results"]
    p1_setup(out)
    p2_error(r, out)
    p3_vs_shipped(r, out)
    p4_dims(r, out)
    p5_training(qdir, out)
    p6_chunk(r, out)


if __name__ == "__main__":
    main()
