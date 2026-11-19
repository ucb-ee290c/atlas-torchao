"""Weights-only check of gptq_awq:producer_aware on the real Pi0 MLP pairs (CPU, no calibration run).

For each Gemma MLP (up_proj -> down_proj) of the stock Pi0 checkpoint:
  * s_j (the AWQ scale of the recorded tile export) is recovered from the export's down_proj columns
    (median_i |dq_down[i, j] / W_down[i, j]|), and the norm group's s_in from the up_proj columns, so the
    folded up_proj the export held, R = bf16(W_up * s_in), is known exactly (the s_in group is not searched here);
  * calibration activations are not recoverable without a Pi0 forward pass, so the search runs on a PROXY:
    the consumer Hessian is diag(sx^2) with sx = s^(1/alpha_recorded) (AWQ's s = sx^alpha), the producer
    Hessian is the identity. The proxy is validated by checking that the OLD search on it reproduces the
    recorded alpha and s.
Variants: old = awq_search without the producer term; new = with it; each searched and measured on the panel grid
(w=panel rows=32 rule=nc, the tile export) and on a per-channel grid (rows=1, same rule; the control).
Measured (RTN casts, no GPTQ): rows of the producer whose effective row s_j * Q(R_j / s_j) differs from R_j by
more than 10% (relative L2), all-zero producer rows, and the composite ||Dd @ Du - Wd @ R|| / ||Wd @ R||.
usage: awq_producer_validate.py OUT.json LAYER_KEY[,LAYER_KEY...] [fixed:A_OLD,A_NEW | truth]   (LM0, E1, ... : LM = paligemma language model, E = gemma_expert)"""
import json
import math
import sys
import time

import torch
from safetensors import safe_open

sys.path.insert(0, __file__.rsplit("/src/scripts/", 1)[0] + "/src")
import reference.ref_numerics as rn
from algos import gptq_awq as ga
from algos.granularity import panel_scale_exp

STOCK = "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors"
EXPORT = "/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/atlas_v1_awq_tile_static/export"
GRIDS = {"panel": lambda W: panel_scale_exp(W, 32, "nc"), "channel": lambda W: panel_scale_exp(W, 1, "nc")}


def base(key):
    kind, i = ("LM" if key.startswith("LM") else "E"), int(key.lstrip("LME"))
    return ("paligemma_with_expert.paligemma.model.language_model" if kind == "LM" else "paligemma_with_expert.gemma_expert.model") + f".layers.{i}.mlp."


def recover(Wu, Wd, Du, Dd):
    m = (Dd != 0) & (Wd != 0)
    r = torch.where(m, (Dd / torch.where(Wd != 0, Wd, torch.ones_like(Wd))).abs(), torch.full_like(Wd, float("nan")))
    s = torch.nanmedian(r, dim=0).values                              # per intermediate channel j
    s = torch.where(torch.isfinite(s) & (s > 0), s, torch.ones_like(s))
    m = (Du != 0) & (Wu != 0)
    r = torch.where(m, (Du * s[:, None] / torch.where(Wu != 0, Wu, torch.ones_like(Wu))).abs(), torch.full_like(Wu, float("nan")))
    s_in = torch.nanmedian(r, dim=0).values                           # per input channel i
    return s, torch.where(torch.isfinite(s_in) & (s_in > 0), s_in, torch.ones_like(s_in))


def measure(R, Wd, s, gp, gc):
    """producer on grid gp, consumer on grid gc, same s."""
    Rf = R.float()
    Rs = (Rf / s[:, None]).to(R.dtype).float()
    Du = ga.qdq(Rs, GRIDS[gp](Rs))
    eff = Du * s[:, None]
    row = (eff - Rf).norm(dim=1) / Rf.norm(dim=1).clamp(min=1e-30)
    Wds = (Wd.float() * s).to(Wd.dtype).float()
    Dd = ga.qdq(Wds, GRIDS[gc](Wds))
    P0 = Wd.float() @ Rf
    comp = float((Dd @ Du - P0).norm() / P0.norm())
    # same, but only the producer quantized (consumer exact): isolates the producer's share
    comp_p = float(((Wds @ Du) - P0).norm() / P0.norm())
    Dc = ga.qdq(Rs, GRIDS["channel"](Rs))                               # same s, per-channel exponents
    vs_chan = (Du - Dc).norm(dim=1) / Dc.norm(dim=1).clamp(min=1e-30)
    return {"rows_panel_cast_vs_channel_cast_gt10pct": int((vs_chan > 0.1).sum()),
            "damaged_rows_gt10pct": int((row > 0.1).sum()), "zero_rows": int((Du.abs().amax(1) == 0).sum()),
            "median_row_err": float(row.median()), "composite_rel": comp, "composite_producer_only": comp_p}


def run(key, stock, ex, out, fixed=None):
    b = base(key)
    dq = lambda s: rn.dequantize(ex.get_tensor(b + s + ".weight_codes"), ex.get_tensor(b + s + ".weight_scale_exp"))
    Wu, Wd = stock.get_tensor(b + "up_proj.weight"), stock.get_tensor(b + "down_proj.weight")
    s, s_in = recover(Wu.float(), Wd.float(), dq("up_proj"), dq("down_proj"))
    man = json.load(open(EXPORT + "/manifest.json"))["recipe"]["algorithm_record"]["prepare"]["gptq_awq"]
    alpha_rec = man["alpha_picks"][b + "up_proj->down_proj"]["weight"]
    R = (Wu.float() * s_in[None, :]).to(Wu.dtype)
    ls = torch.log(s.double())
    res = {"key": key, "alpha_recorded": alpha_rec, "I": Wu.shape[0], "K": Wu.shape[1],
           "s_log2_range": float((ls.max() - ls.min()) / math.log(2))}
    # proxy sx: s = sx^alpha / sqrt(max*min)  =>  log sx = log s / alpha + const
    if alpha_rec > 0:
        lsx = ls / alpha_rec
        lsx = lsx - lsx.mean()
        sx = torch.exp(lsx.clamp(-60, 60)).float()
    else:
        sx = torch.ones_like(s)
    Hd = (sx.double() ** 2).float()                                   # diagonal consumer Hessian proxy
    ncols = Wd.shape[0]
    variants = {"recorded_export_s": (s, None)}
    t = time.time()
    if fixed:                       # reuse alphas of an earlier full search (same s = sx^alpha / sqrt(max*min))
        for lab, a in zip(("old_search_panel", "new_search_panel"), fixed):
            sv = sx.pow(a).clamp(min=1e-4)
            variants[lab] = (sv / (sv.max() * sv.min()).sqrt(), a)
    for gname in (() if fixed else ("panel", "channel")):
        cp = ga.producer_row_weights(R, None, Wd, Hd)
        for lab, prod in (("old", None), ("new", {"W": R, "H": None, "c": cp, "exp_fn": GRIDS[gname]})):
            sv, a, loss = ga.awq_search([Wd], Hd, sx, 20, GRIDS[gname], producer=prod)
            variants[f"{lab}_search_{gname}"] = (sv, a)
            print(key, gname, lab, "alpha", a, "t %.0fs" % (time.time() - t), flush=True)
    res["variants"] = {}
    for lab, (sv, a) in variants.items():
        r = {"alpha": a}
        for g in ("panel", "channel"):
            r[g] = measure(R, Wd, sv, g, g)
        res["variants"][lab] = r
        print(key, lab, "alpha", a, "panel", r["panel"], "channel", r["channel"], flush=True)
    out[key] = res


def export_truth(key):
    """What the recorded export itself holds (RTN + GPTQ of the real run): damaged producer rows and composite against
    the norm-folded stock pair R (mlp_composite.py compares against the stock up_proj without s_in)."""
    stock = safe_open(STOCK, "pt")
    ex = safe_open(EXPORT + "/model.safetensors", "pt")
    b = base(key)
    dq = lambda s: rn.dequantize(ex.get_tensor(b + s + ".weight_codes"), ex.get_tensor(b + s + ".weight_scale_exp"))
    Wu, Wd = stock.get_tensor(b + "up_proj.weight"), stock.get_tensor(b + "down_proj.weight")
    Du, Dd = dq("up_proj"), dq("down_proj")
    s, s_in = recover(Wu.float(), Wd.float(), Du, Dd)
    R = (Wu.float() * s_in[None, :]).to(Wu.dtype).float()
    row = ((Du * s[:, None]) - R).norm(dim=1) / R.norm(dim=1).clamp(min=1e-30)
    P0 = Wd.float() @ R
    P0s = Wd.float() @ Wu.float()
    return {"damaged_rows_gt10pct": int((row > 0.1).sum()), "composite_vs_normfolded": float((Dd @ Du - P0).norm() / P0.norm()),
            "composite_vs_stock_up (mlp_composite.py)": float((Dd @ Du - P0s).norm() / P0s.norm()),
            "s_in_log2_range": float(torch.log2(s_in).max() - torch.log2(s_in).min())}


if __name__ == "__main__":
    stock = safe_open(STOCK, "pt")
    ex = safe_open(EXPORT + "/model.safetensors", "pt")
    out = {}
    if len(sys.argv) > 3 and sys.argv[3] == 'truth':
        print(json.dumps({k: export_truth(k) for k in sys.argv[2].split(',')}, indent=1)); sys.exit()
    for k in sys.argv[2].split(","):
        run(k, stock, ex, out, [float(x) for x in sys.argv[3].split(":")[1].split(",")] if len(sys.argv) > 3 else None)
        json.dump(out, open(sys.argv[1], "w"), indent=1)
