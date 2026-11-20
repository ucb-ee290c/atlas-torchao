"""Interim QAT plots from qat_log.json / train.log (val curves, training loss, seed-0 test per joint)."""
import json, re, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

Q = Path(sys.argv[1] if len(sys.argv) > 1 else "src/out/qat")
OUT = Q / "plots"; OUT.mkdir(exist_ok=True)
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RUNS = [("smolvla_vel", "QAT seed 0 (32-frame val)", "#2a78d6", "-"),
        ("smolvla_vel_s1", "QAT seed 1", "#eb6834", "-"),
        ("smolvla_vel_s2", "QAT seed 2", "#1baf7a", "-"),
        ("smolvla_noquant", "No-quant control", "#7d7c78", "--")]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "legend.frameon": False})

def load(run):
    d = Q / run
    if (d / "qat_log.json").exists():
        L = json.loads((d / "qat_log.json").read_text())
        return [(v["step"], v.get("gap_recovered", 0.0), v.get("gap_recovered_se") or 0) for v in L["val"]], \
               [s["loss"] for s in L["steps"]], L
    txt = (d / "train.log").read_text()
    val = [(0, 0.0, 0.0)] + [(int(a), float(b) / 100, float(c) / 100) for a, b, c in
                             re.findall(r"val step (\d+): rel_l2 [\d.]+, gap recovered (-?[\d.]+)% \+- ([\d.]+)", txt)]
    loss = [float(x) for x in re.findall(r"step \d+/\d+ loss ([\d.e+-]+)", txt)]
    return val, loss, None

data = {r: load(r) for r, *_ in RUNS}

# 1. val gap recovered over training
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.axhline(0, color=INK2, lw=1)
for run, lab, c, ls in RUNS:
    v = np.array(data[run][0])
    if run != "smolvla_vel":
        ax.fill_between(v[:, 0], 100 * (v[:, 1] - v[:, 2]), 100 * (v[:, 1] + v[:, 2]), color=c, alpha=0.10, lw=0)
    ax.plot(v[:, 0], 100 * v[:, 1], ls, color=c, lw=2, marker="o", ms=5, label=lab, alpha=0.55 if run == "smolvla_vel" else 1)
ax.set_xlabel("optimizer step"); ax.set_ylabel("gap recovered on val (%)")
ax.set_title("Val: share of the FP8 (ref_conv) squared error removed, vs bf16\n"
             "bands = ±1 SE over 128 val frames; seed 0 used 32 frames (no band, very noisy)",
             loc="left", color=INK, fontsize=11)
ax.legend(loc="lower right", ncol=2, fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "val_gap_curves.png", dpi=160); plt.close(fig)

# 2. training loss (rolling mean)
fig, ax = plt.subplots(figsize=(8, 4.6))
for run, lab, c, ls in RUNS:
    l = np.array(data[run][1])
    if len(l) < 25: continue
    k = 25; sm = np.convolve(l, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, len(l) + 1), sm, ls, color=c, lw=2, label=lab.replace(" (32-frame val)", ""))
ax.set_yscale("log"); ax.set_xlabel("optimizer step"); ax.set_ylabel("KD velocity MSE (25-step mean, log)")
ax.set_title("Training loss: QAT learns (−28%); the control drifts away from its teacher", loc="left", color=INK)
ax.legend(loc="center right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "train_loss.png", dpi=160); plt.close(fig)

# 3. seed-0 test per joint: ref_conv error and QAT error (two panels, same unit)
L0 = data["smolvla_vel"][2]
t = L0["test"]
r0, q0 = np.array(t["ref_conv"]["rel_l2_per_dim"]), np.array(t["qat_ref_conv"]["rel_l2_per_dim"])
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
x = np.arange(len(r0)); w = 0.38
a1.bar(x - w / 2 - 0.01, r0, w, color="#7d7c78", label="ref_conv FP8")
a1.bar(x + w / 2 + 0.01, q0, w, color="#2a78d6", label="QAT seed 0")
a1.set_xticks(x, [f"j{i}" if i < 5 else "grip" for i in x]); a1.set_ylabel("rel_l2 vs bf16")
a1.set_title("Test error per action dim (200 frames)", loc="left", color=INK); a1.legend(fontsize=9)
g = 100 * (1 - (q0 / r0) ** 2)   # same definition as the headline: on squared error
a2.bar(x, g, 0.6, color="#2a78d6"); a2.axhline(0, color=INK2, lw=1)
for i, v in enumerate(g):
    a2.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
a2.set_xticks(x, [f"j{i}" if i < 5 else "grip" for i in x]); a2.set_ylabel("gap recovered, squared error (%)")
tot = t["qat_ref_conv"]
a2.set_title(f"Gap recovered per dim (overall {100*tot['gap_recovered']:.1f}% ± {100*tot['gap_recovered_se']:.1f})",
             loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "test_seed0_per_dim.png", dpi=160); plt.close(fig)

# 4. latest val reading per run with ±1 SE (and the seed-0 test point)
fig, ax = plt.subplots(figsize=(8, 3.8))
rows = []
for run, lab, c, _ in RUNS:
    s, g_, se = data[run][0][-1]
    rows.append((f"{lab} — val step {s}", 100 * g_, 100 * se, c))
rows.insert(0, ("QAT seed 0 — TEST (200 frames)", 100 * tot["gap_recovered"], 100 * tot["gap_recovered_se"], "#2a78d6"))
for i, (lab, m, se, c) in enumerate(rows[::-1]):
    ax.errorbar(m, i, xerr=se, fmt="o", color=c, ms=8, capsize=4, lw=2)
    ax.text(m, i + 0.22, f"{m:.0f}%", ha="center", fontsize=9, color=INK)
ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]]); ax.axvline(0, color=INK2, lw=1)
ax.set_xlabel("gap recovered (% of squared error), ±1 SE"); ax.grid(axis="y", visible=False)
ax.set_title("Where each run stands (interim)", loc="left", color=INK, pad=16)
fig.tight_layout(); fig.savefig(OUT / "latest_summary.png", dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in OUT.glob("*.png")))

# 5. QAT training loss on a linear axis: raw steps (faint) + 50-step mean, first/last-50 means marked
fig, ax = plt.subplots(figsize=(9, 5))
k = 50
for run, lab, c, _ in RUNS[:3]:
    l = np.array(data[run][1])
    if len(l) < k: continue
    ax.scatter(np.arange(1, len(l) + 1), l, s=6, color=c, alpha=0.18, lw=0)
    sm = np.convolve(l, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, len(l) + 1), sm, color=c, lw=2.2,
            label=f"{lab.replace(' (32-frame val)', '')}: {l[:k].mean():.4f} → {l[-k:].mean():.4f} "
                  f"({100 * (l[-k:].mean() / l[:k].mean() - 1):+.0f}%)" + ("" if len(l) == 500 else f", step {len(l)}"))
ax.set_ylim(0, 0.03)
ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss (KD velocity MSE)")
ax.set_title("QAT training loss: dots = each step, line = 50-step mean", loc="left", color=INK)
ax.legend(loc="upper right", fontsize=9, title="first-50 mean → last-50 mean", title_fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "qat_train_loss_linear.png", dpi=160); plt.close(fig)
print("wrote qat_train_loss_linear.png")

# 6. The gap picture: bf16 = 0, default FP8 (ref_conv) = 100% of the gap, QAT = what is left
REC, LEFT = "#2a78d6", "#7d7c78"
rq, rr = t["qat_ref_conv"]["sq_err"], t["ref_conv"]["sq_err"]
left = 100 * rq / rr
fig = plt.figure(figsize=(10, 6.4))
gs = fig.add_gridspec(2, 1, height_ratios=[1.1, 2.2], hspace=0.55)
a = fig.add_subplot(gs[0])
a.barh(1, 100, 0.55, color=LEFT)
a.barh(0, left, 0.55, color=LEFT)
a.barh(0, 100 - left, 0.55, left=left, color=REC, alpha=0.25, edgecolor=REC, hatch="//", lw=1.2)
a.text(50, 1, "default FP8 (ref_conv): 100%% of the gap  ·  rel_l2 %.4f" % t["ref_conv"]["rel_l2"],
       ha="center", va="center", color="white", fontsize=10)
a.text(left / 2, 0, "QAT seed 0: %.0f%% left  ·  rel_l2 %.4f" % (left, t["qat_ref_conv"]["rel_l2"]),
       ha="center", va="center", color="white", fontsize=10)
a.text(left + (100 - left) / 2, 0, "%.0f%%\nrecovered" % (100 - left), ha="center", va="center", color=INK,
       fontsize=10, fontweight="bold")
a.axvline(0, color=INK, lw=1.5)
a.set_yticks([]); a.set_xlim(0, 100); a.grid(False); a.spines["left"].set_visible(False)
a.set_xlabel("squared error vs bf16, as %% of default FP8's  (± %.0f%% SE on the recovered share)"
             % (100 * t["qat_ref_conv"]["gap_recovered_se"]))
a.set_title("How much of the FP8 error QAT removes — held-out test, 200 frames", loc="left", color=INK)

b = fig.add_subplot(gs[1])
names = [f"joint {i}" if i < 5 else "gripper" for i in range(len(r0))]
lf = 100 * (q0 / r0) ** 2
y = np.arange(len(r0))[::-1]
for yi, n, l_, rr0 in zip(y, names, lf, r0):
    b.barh(yi, l_, 0.6, color=LEFT)
    b.barh(yi, 100 - l_, 0.6, left=l_, color=REC, alpha=0.25, edgecolor=REC, hatch="//", lw=1.2)
    b.text(101, yi, f"{100 - l_:.0f}% recovered", va="center", fontsize=9, color=INK)
b.set_yticks(y, [f"{n}  (FP8 rel_l2 {r:.3f})" for n, r in zip(names, r0)])
b.set_xlim(0, 100); b.grid(axis="y", visible=False); b.axvline(0, color=INK, lw=1.5)
b.set_xlabel("squared error left after QAT, % of default FP8's (per action dimension)")
b.set_title("Per action dimension (gray = error still left, hatched = removed by QAT)", loc="left", color=INK, fontsize=11)
fig.subplots_adjust(left=0.27, right=0.86, top=0.93, bottom=0.08)
fig.savefig(OUT / "gap_recovered_test.png", dpi=160); plt.close(fig)
print("wrote gap_recovered_test.png")
