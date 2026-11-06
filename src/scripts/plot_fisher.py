"""Plot a profile_fisher_model JSON (python -m reference.ref_profile --out ...).

    python src/scripts/plot_fisher.py src/out/fisher/tinyllama/tinyllama_n16.json [out.png]
    python src/scripts/plot_fisher.py src/out/fisher/smolvla/smolvla_n32.json --summary   # ONE figure, 6 panels

Needs matplotlib + numpy only. Panels: layer x module heatmap (log colour), cumulative share of
the layers ranked most-sensitive first, weight- vs activation-sensitivity scatter, top-15 layers."""
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

TOWER = None          # set from --tower in __main__
MODS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BLUE, GRAY = "#2a78d6", "#b9b8b2"


ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "fc1", "fc2"]


def _layer_grid(L):
    """(grid [n_layers, n_modules], module names) from keys ...layers.<i>.(self_attn|mlp).<module>;
    modules are whatever the report contains, in a canonical order. None if there are no such keys."""
    cells = {}
    for n, r in L.items():
        m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$", n)
        if m:
            if r["fisher"] > 0:                  # zero = not in the loss graph; log colour cannot show it
                cells[(int(m.group(1)), m.group(2))] = r["fisher"]
    if not cells:
        return None
    mods = sorted({o for _, o in cells}, key=lambda o: ORDER.index(o) if o in ORDER else len(ORDER))
    grid = np.full((1 + max(i for i, _ in cells), len(mods)), np.nan)
    for (i, o), v in cells.items():
        grid[i, mods.index(o)] = v
    return grid, mods


def _title_data(meta):
    return (f"{meta['n']} x {meta['seqlen']} tokens" if meta.get("seqlen") not in (None, "n/a")
            else f"{meta['n']} frames x {meta.get('draws', 1)} draws") + f", seed {meta['seed']}"


def _save(fig, dst, src):
    """Save with a footer naming the run (report file + model + data), so a PNG says where it came from."""
    d = json.loads(Path(src).read_text())["meta"]
    tower = f" | tower: {TOWER}" if TOWER else ""
    fig.text(0.005, 0.004, f"run: {Path(src).name} | {Path(str(d['model'])).name} | {_title_data(d)}{tower}",
             fontsize=7, color="0.4", ha="left", va="bottom")
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dst, dpi=150)


def _short(n):
    """Readable label: model.layers.3.mlp.down_proj / ...text_model.layers.3.mlp.down_proj -> L3.down_proj."""
    m = re.search(r"layers\.(\d+)\.(?:self_attn|mlp)\.(\w+)$", n)
    return f"L{m.group(1)}.{m.group(2)}" if m else ".".join(n.split(".")[-2:])


def _load(src, tower=None):
    """(meta, linear entries, attention entries): attention keys carry kind == "attn". `tower` keeps
    only entries whose "tower" field matches (VLA reports: vision | language | expert | action_head)."""
    d = json.loads(Path(src).read_text())
    keep = {n: r for n, r in d["layers"].items() if tower is None or r.get("tower") == tower}
    if tower is not None and not keep:
        raise SystemExit(f"no entries with tower={tower!r} in {src} (only VLA reports carry a tower field)")
    lin = {n: r for n, r in keep.items() if r.get("kind") != "attn"}
    attn = {n: r for n, r in keep.items() if r.get("kind") == "attn"}
    return d["meta"], lin, attn


def main(src, dst):
    meta, L, _ = _load(src, TOWER)
    names = list(L)
    total = sum(r["fisher"] for r in L.values())

    fig, ax = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle(f"Fisher-delta sensitivity - {Path(meta['model']).name} "
                 f"({_title_data(meta)})" + (f" - {TOWER}" if TOWER else ""), fontsize=14)

    # 1. heatmap
    a = ax[0, 0]
    lg = _layer_grid(L)
    if lg is None:
        a.axis("off")
        a.set_title("no layers.<i> entries for a heatmap")
    else:
        grid, mods = lg
        im = a.imshow(grid, aspect="auto", cmap="Blues", norm=LogNorm(np.nanmin(grid), np.nanmax(grid)))
        a.set_xticks(range(len(mods)), [m.replace("_proj", "") for m in mods])
        a.set_yticks(range(0, grid.shape[0], 2))
        a.set_ylabel("layer")
        a.set_title("fisher per layer (log colour)")
        fig.colorbar(im, ax=a, label="fisher")

    # 2. cumulative share
    srt = sorted(names, key=lambda n: -L[n]["fisher"])
    cum = np.cumsum([L[n]["fisher"] for n in srt]) / total
    k = np.arange(1, len(srt) + 1)
    a = ax[0, 1]
    a.plot(k, cum, color=BLUE, lw=2)
    for kk in (1, 3, 10, 30):
        a.plot(kk, cum[kk - 1], "o", color=BLUE, mec="w", mew=1.5, ms=8)
        a.annotate(f"top {kk}: {cum[kk-1]:.0%}", (kk, cum[kk - 1]), (6, -16), textcoords="offset points")
    a.set_xscale("log")
    a.set_ylim(0, 1.02)
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.set_xlabel("layers kept, most sensitive first")
    a.set_ylabel("cumulative share of total fisher")
    a.set_title("how concentrated is it?")
    a.grid(alpha=.3)

    # 3. weights vs activations
    a = ax[1, 0]
    w = np.array([L[n]["fisher_w"] for n in names])
    x = np.array([L[n]["fisher_x"] for n in names])
    dn = np.array([n.endswith("down_proj") for n in names])
    ok = (w > 0) & (x > 0)
    names, w, x, dn = [n for n, k in zip(names, ok) if k], w[ok], x[ok], dn[ok]
    a.scatter(w[~dn], x[~dn], s=40, c=GRAY, edgecolors="w", linewidths=1.2, label="other layers")
    a.scatter(w[dn], x[dn], s=40, c=BLUE, edgecolors="w", linewidths=1.2, label="down_proj")
    lim = [min(w.min(), x.min()) * .7, max(w.max(), x.max()) * 1.4]
    a.plot(lim, lim, "--", color="gray", lw=1)
    for i in np.argsort(-(w + x))[:3]:
        a.annotate(_short(names[i]), (w[i], x[i]), (7, 3),
                   textcoords="offset points")
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="fisher_w (weights)",
          ylabel="fisher_x (activations)", title="above the dashed line: activations dominate")
    a.legend(frameon=False)
    a.grid(alpha=.3)

    # 4. top 15, stacked weight / activation
    top = srt[:15][::-1]
    a = ax[1, 1]
    wt = np.array([L[n]["fisher_w"] for n in top])
    xt = np.array([L[n]["fisher_x"] for n in top])
    a.barh(range(len(top)), wt, color=BLUE, height=.7, label="weights")
    a.barh(range(len(top)), xt, left=wt, color="#eb6834", height=.7, label="activations")
    a.set_yticks(range(len(top)), [_short(n) for n in top])
    a.set_xscale("log")
    a.set_xlabel("fisher (log)")
    a.set_title("top 15 layers")
    a.legend(frameon=False, loc="lower right")
    a.grid(alpha=.3, axis="x")

    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


def scatter_normalized(src, dst):
    """fisher_w vs fisher_x, each axis divided by its own total over layers (share of that term)."""
    _, L, _ = _load(src, TOWER)
    names = list(L)
    w = np.array([L[n]["fisher_w"] for n in names])
    x = np.array([L[n]["fisher_x"] for n in names])
    w, x = w / w.sum(), x / x.sum()
    dn = np.array([n.endswith("down_proj") for n in names])
    ok = (w > 0) & (x > 0)                       # zero-score entries cannot sit on log axes
    if not ok.all():
        print(f"scatter: {int((~ok).sum())} zero-score entries not drawn")
    names, w, x, dn = [n for n, k in zip(names, ok) if k], w[ok], x[ok], dn[ok]

    fig, a = plt.subplots(figsize=(7.5, 7))
    a.scatter(w[~dn], x[~dn], s=45, c=GRAY, edgecolors="w", linewidths=1.2, label="other layers")
    a.scatter(w[dn], x[dn], s=45, c=BLUE, edgecolors="w", linewidths=1.2, label="down_proj")
    lim = [min(w.min(), x.min()) * .8, max(w.max(), x.max()) * 1.4]
    a.plot(lim, lim, "--", color="gray", lw=1)
    for i in np.argsort(-(w + x))[:4]:
        a.annotate(_short(names[i]), (w[i], x[i]), (7, 3),
                   textcoords="offset points")
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
          xlabel="weight fisher, share of total  (fisher_w / sum fisher_w)",
          ylabel="activation fisher, share of total  (fisher_x / sum fisher_x)",
          title="Normalized sensitivity per layer\nabove the dashed line: activations matter more than weights")
    a.set_aspect("equal")                      # log ticks keep matplotlib's default 10^x labels
    a.legend(frameon=False, loc="upper left")
    a.grid(alpha=.3)
    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


def heatmap_combined(src, dst):
    """layer x module heatmap of fisher = fisher_w + fisher_x; cells print the % of the model total."""
    meta, L, _ = _load(src, TOWER)
    total = sum(r["fisher"] for r in L.values())
    lg = _layer_grid(L)
    if lg is None:
        print("heatmap skipped: no layers.<i> entries", "for tower " + str(TOWER) if TOWER else "")
        return
    grid, MODS = lg
    nl = grid.shape[0]

    fig, a = plt.subplots(figsize=(1.6 + 1.1 * len(MODS), 0.35 * nl + 3))
    norm = LogNorm(np.nanmin(grid), np.nanmax(grid))
    im = a.imshow(grid, aspect="auto", cmap="Blues", norm=norm)
    for i in range(nl):
        for j in range(len(MODS)):
            v = grid[i, j]
            if np.isnan(v):
                continue
            dark = norm(v) > 0.55
            a.text(j, i, f"{100 * v / total:.1f}", ha="center", va="center", fontsize=8,
                   color="white" if dark else "#333")
    a.set_xticks(range(len(MODS)), [m.replace("_proj", "") for m in MODS])
    a.set_yticks(range(nl))
    a.set_ylabel("layer")
    a.xaxis.tick_top()
    a.set_title(f"Combined fisher (weights + activations) per layer, {Path(meta['model']).name}\n"
                f"cell text = % of {'tower' if TOWER else 'model'} total; colour is log scale", pad=34)
    fig.colorbar(im, ax=a, label="fisher = fisher_w + fisher_x", fraction=.04, pad=.03)
    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


def tower_summary(src, dst):
    """Share of the model's total fisher per tower, split into linear layers and attention operands."""
    d = json.loads(Path(src).read_text())
    tot = sum(r["fisher"] for r in d["layers"].values())
    towers = sorted({r.get("tower", "?") for r in d["layers"].values()})
    lin = np.array([sum(r["fisher"] for r in d["layers"].values() if r.get("tower") == t and r.get("kind") != "attn")
                    for t in towers]) / tot
    att = np.array([sum(r["fisher"] for r in d["layers"].values() if r.get("tower") == t and r.get("kind") == "attn")
                    for t in towers]) / tot
    fig, a = plt.subplots(figsize=(8, 1.2 + 0.9 * len(towers)))
    y = np.arange(len(towers))
    a.barh(y, lin, color=BLUE, height=.6, label="linear layers")
    a.barh(y, att, left=lin, color="#eb6834", height=.6, label="attention q/k/p/v")
    for i, (l, t) in enumerate(zip(lin, att)):
        a.text(l + t + .005, i, f"{100 * (l + t):.1f}%", va="center")
    n_lin = [sum(1 for r in d["layers"].values() if r.get("tower") == t and r.get("kind") != "attn") for t in towers]
    n_att = [sum(1 for r in d["layers"].values() if r.get("tower") == t and r.get("kind") == "attn") for t in towers]
    a.set_yticks(y, [f"{t}\n{nl} linear" + (f" + {na} attn" if na else "") for t, nl, na in zip(towers, n_lin, n_att)])
    a.invert_yaxis()
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.set_xlabel("share of total fisher")
    a.set_title(f"Where the sensitivity sits, by tower - {Path(d['meta']['model']).name}")
    a.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


TOWER_COLORS = {"vision": "#2a78d6", "language": "#eb6834", "expert": "#2e9e6b", "action_head": "#8a5fd0",
                "joint_attn": "#c9a227"}


def summary_figure(src, dst):
    """One figure, six panels, for any report: tower shares, normalized w-vs-x scatter, cumulative
    concentration, fisher per layer index, attention operands, top-15 entries."""
    d = json.loads(Path(src).read_text())
    L = d["layers"]
    tw = lambda r: r.get("tower", "model")
    col = lambda t: TOWER_COLORS.get(t, BLUE)
    total = sum(r["fisher"] for r in L.values())
    lin = {n: r for n, r in L.items() if r.get("kind") != "attn"}
    att = {n: r for n, r in L.items() if r.get("kind") == "attn"}
    towers = sorted({tw(r) for r in L.values()})

    fig, ax = plt.subplots(2, 3, figsize=(19, 10.5))
    a = ax[0, 0]                                                          # A: share by tower
    y = np.arange(len(towers))
    sl = np.array([sum(r["fisher"] for r in lin.values() if tw(r) == t) for t in towers]) / total
    sa = np.array([sum(r["fisher"] for r in att.values() if tw(r) == t) for t in towers]) / total
    a.barh(y, sl, color="#4d4d4d", height=.6, label="linear layers")        # neutral: tower colors mean towers
    a.barh(y, sa, left=sl, color="#b3b3b3", height=.6, label="attention q/k/p/v")
    for i, (l, t) in enumerate(zip(sl, sa)):
        a.text(l + t + .005, i, f"{100 * (l + t):.1f}%", va="center")
    count = lambda t, group: sum(1 for r in group.values() if tw(r) == t)
    a.set_yticks(y, [f"{t}\n{count(t, lin)} linear" + (f" + {count(t, att)} attn" if att else "") for t in towers])
    a.invert_yaxis()
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.set(xlabel="share of total fisher", xlim=(0, 1.15 * (sl + sa).max()),
          title="A. where the sensitivity sits, by tower\n(entries per tower under each name)")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)

    a = ax[0, 1]                                                          # B: weights vs activations
    names = [n for n, r in lin.items() if r["fisher_w"] > 0 and r["fisher_x"] > 0]
    w = np.array([lin[n]["fisher_w"] for n in names]) / sum(r["fisher_w"] for r in lin.values())
    x = np.array([lin[n]["fisher_x"] for n in names]) / sum(r["fisher_x"] for r in lin.values())
    for t in towers:
        m = np.array([tw(lin[n]) == t for n in names])
        if m.any():
            a.scatter(w[m], x[m], s=30, c=col(t), edgecolors="w", linewidths=.8, label=t)
    lim = [min(w.min(), x.min()) * .8, max(w.max(), x.max()) * 1.4]
    a.plot(lim, lim, "--", color="gray", lw=1)
    for i in np.argsort(-(w + x))[:4]:
        a.annotate(_short(names[i]), (w[i], x[i]), (6, 3), textcoords="offset points", fontsize=8)
    a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, xlabel="weight fisher (share of total)",
          ylabel="activation fisher (share of total)",
          title="B. weights vs activations (linear layers)\nabove dashed line: activations matter more")
    a.set_aspect("equal")
    a.legend(frameon=False, loc="upper left", fontsize=8)
    a.grid(alpha=.3)

    a = ax[0, 2]                                                          # C: concentration
    f = np.sort([r["fisher"] for r in L.values()])[::-1] / total
    c = np.cumsum(f)
    a.plot(np.arange(1, len(f) + 1), c, color=BLUE, lw=2)
    for k in (10, 50):
        if k <= len(f):
            a.plot(k, c[k - 1], "o", color="#eb6834")
            a.annotate(f"top {k}: {c[k - 1]:.0%}", (k, c[k - 1]), (8, -14), textcoords="offset points")
    a.set(xscale="log", xlabel=f"entries ranked most sensitive first (of {len(f)})", ylabel="cumulative share",
          title="C. how concentrated is it?")
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.grid(alpha=.3)

    a = ax[1, 0]                                                          # D: fisher per layer index
    drawn = False
    for t in towers:
        per = {}
        for n, r in lin.items():
            m = re.search(r"layers\.(\d+)\.", n)
            if tw(r) == t and m:
                per[int(m.group(1))] = per.get(int(m.group(1)), 0) + r["fisher"]
        if per:
            xs = sorted(per)
            a.plot(xs, [per[i] / total for i in xs], "o-", color=col(t), label=t, ms=4)
            drawn = True
    a.set(xlabel="layer index", ylabel="share of total fisher (linear layers, summed per layer)",
          title="D. sensitivity by depth")
    a.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    a.grid(alpha=.3)
    if drawn:
        a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]                                                          # E: attention operands
    if att:
        groups = sorted({tw(r) for r in att.values()})
        wd = .8 / len(groups)
        for gi, t in enumerate(groups):
            vals = [sum(r["fisher"] for n, r in att.items() if tw(r) == t and n.endswith("." + o)) / total
                    for o in OPS]
            a.bar(np.arange(len(OPS)) + gi * wd, vals, wd, color=col(t), label=f"{t} attention")
        a.set_xticks(np.arange(len(OPS)) + wd * (len(groups) - 1) / 2, OPS)
        a.set(ylabel="share of total fisher", title="E. attention operands (q, k, softmax p, v)")
        a.yaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
        a.legend(frameon=False, fontsize=8, loc="upper left")
    else:
        a.text(.5, .5, "no attention entries", ha="center", transform=a.transAxes)
        a.set_title("E. attention operands")

    a = ax[1, 2]                                                          # F: top 15
    top = sorted(L.items(), key=lambda kv: -kv[1]["fisher"])[:15]
    ab = {"action_head": "act", "vision": "vis", "language": "lang", "expert": "exp", "joint_attn": "joint"}
    labels = [f"{ab.get(tw(r), tw(r))}  {_short(n)}" + ("  (attn)" if r.get("kind") == "attn" else "") for n, r in top]
    a.barh(np.arange(len(top)), [r["fisher"] / total for _, r in top], color=[col(tw(r)) for _, r in top], height=.7)
    a.set_yticks(np.arange(len(top)), labels, fontsize=8)
    a.invert_yaxis()
    a.xaxis.set_major_formatter(lambda v, _: f"{v:.1%}")
    a.set(xlabel="share of total fisher", title="F. top 15 entries")

    fig.suptitle(f"Fisher-delta sensitivity - {Path(str(d['meta']['model'])).name}, {_title_data(d['meta'])}", fontsize=14)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, dst, src)
    print("wrote", dst)


OPS = ["q", "k", "p", "v"]


def _attn_grid(attn):
    """{(layer, op): fisher} -> [n_layers, 4] array in OPS order."""
    cells = {}
    for n, r in attn.items():
        m = re.search(r"layers\.(\d+)\..*\.([qkpv])$", n) or re.search(r"\.L(\d+)\.([qkpv])$", n)
        if m and r["fisher"] > 0:
            cells[(int(m.group(1)), m.group(2))] = r["fisher"]
    nl = 1 + max(i for i, _ in cells)
    grid = np.full((nl, len(OPS)), np.nan)
    for (i, o), v in cells.items():
        grid[i, OPS.index(o)] = v
    return grid


def attn_heatmap(src, dst):
    """layer x {q,k,p,v} heatmap of the attention-operand fisher; cells print % of the attention total."""
    meta, _, attn = _load(src, TOWER)
    grid = _attn_grid(attn)
    total = np.nansum(grid)
    fig, a = plt.subplots(figsize=(5.5, 10))
    norm = LogNorm(np.nanmin(grid), np.nanmax(grid))
    im = a.imshow(grid, aspect="auto", cmap="Blues", norm=norm)
    for i in range(grid.shape[0]):
        for j in range(len(OPS)):
            v = grid[i, j]
            if not np.isnan(v):
                a.text(j, i, f"{100 * v / total:.1f}", ha="center", va="center", fontsize=8,
                       color="white" if norm(v) > .55 else "#333")
    a.set_xticks(range(len(OPS)), OPS)
    a.set_yticks(range(grid.shape[0]))
    a.set_ylabel("layer")
    a.xaxis.tick_top()
    a.set_title(f"Attention-operand fisher per layer, {Path(meta['model']).name}\n"
                f"cell text = % of {TOWER + ' ' if TOWER else ''}attention total; colour is log scale", pad=34, fontsize=10)
    fig.colorbar(im, ax=a, label="fisher", fraction=.08, pad=.04)
    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


def attn_scatter(src, dst):
    """Per layer: k vs q (the Q.K^T operands) and v vs p (the P.V operands), each axis as a share of its own total."""
    meta, _, attn = _load(src, TOWER)
    g = _attn_grid(attn)
    g = g / np.nansum(g, axis=0)                                   # share of each operand's total
    col = {o: g[:, i] for i, o in enumerate(OPS)}
    fig, ax = plt.subplots(1, 2, figsize=(14, 6.8))
    for a, (xo, yo) in zip(ax, [("q", "k"), ("p", "v")]):
        x, y = col[xo], col[yo]
        a.scatter(x, y, s=45, c=BLUE, edgecolors="w", linewidths=1.2)
        lim = [np.nanmin([x, y]) * .8, np.nanmax([x, y]) * 1.4]
        a.plot(lim, lim, "--", color="gray", lw=1)
        for i in [j for j in np.argsort(-np.nan_to_num(x + y))[:4] if not np.isnan(x[j] + y[j])]:
            a.annotate(f"L{i}", (x[i], y[i]), (6, 3), textcoords="offset points")
        a.set(xscale="log", yscale="log", xlim=lim, ylim=lim, aspect="equal",
              xlabel=f"{xo} fisher, share of total", ylabel=f"{yo} fisher, share of total",
              title=f"{yo} vs {xo} per layer")
        a.grid(alpha=.3)
    fig.suptitle(f"Attention-operand sensitivity, {Path(meta['model']).name} (each dot is a layer)")
    fig.tight_layout()
    _save(fig, dst, src)
    print("wrote", dst)


def plot_all(src, dst):
    main(src, str(dst))
    scatter_normalized(src, str(dst.with_name(dst.stem + "_scatter.png")))
    heatmap_combined(src, str(dst.with_name(dst.stem + "_heatmap.png")))
    if _load(src, TOWER)[2]:                                       # attention entries present
        attn_heatmap(src, str(dst.with_name(dst.stem + "_attn_heatmap.png")))
        attn_scatter(src, str(dst.with_name(dst.stem + "_attn_scatter.png")))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?", default=None, help="base PNG path (default: <src dir>/plots/<src stem>/<src stem>.png)")
    ap.add_argument("--summary", action="store_true", help="only ONE figure with six panels (no per-plot PNGs)")
    ap.add_argument("--tower", default=None, help="VLA reports: plot only vision|language|expert|action_head")
    a = ap.parse_args()
    src = a.src
    base = Path(a.dst) if a.dst else Path(src).parent / "plots" / Path(src).stem / (Path(src).stem + ".png")
    base.parent.mkdir(parents=True, exist_ok=True)
    if a.summary:
        summary_figure(src, str(base.with_name(base.stem + "_summary.png")))
        raise SystemExit(0)
    towers = sorted({r["tower"] for r in json.loads(Path(src).read_text())["layers"].values() if "tower" in r})
    if a.tower or not towers:                                       # one tower, or an LLM report
        TOWER = a.tower
        plot_all(src, base.with_name(f"{base.stem}_{a.tower}{base.suffix}") if a.tower else base)
    else:                                                           # VLA: layer indices repeat per tower
        tower_summary(src, str(base.with_name(base.stem + "_towers.png")))
        for t in towers:
            TOWER = t
            if _layer_grid(_load(src, t)[1]) is None:
                print(f"{t}: no layers.<i> entries, skipped")
                continue
            plot_all(src, base.with_name(f"{base.stem}_{t}{base.suffix}"))
