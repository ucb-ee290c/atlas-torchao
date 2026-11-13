"""Gemma layer 0 up_proj: the rows the recorded panel export damages (panel export vs the per-channel control export, same
alpha / s_j, rows differing > 10%, as g0_dissect.py) and what a cast of those rows looks like under different s (RTN, panel grid):
binades below the amax of their 32-row panel (folded basis) and the fraction of their entries flushed to zero.
usage: awq_producer_g0_rows.py ALPHA_NEW   (s for the 'new' column = sx^ALPHA_NEW, sx = s_recorded^(1/0.8); alpha_old = 0.8)"""
import sys
import torch
from safetensors import safe_open
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/scripts/tile_diag")
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/src")
import awq_producer_validate as v
import ref_numerics as rn
from algos import gptq_awq as ga

CTRL = "/scratch2/lorenhung/atlas-fp8/exports-tile-diag/pi0/awq_channel_cpu/export/model.safetensors"
b = v.base("LM0")
stock, T, C = safe_open(v.STOCK, "pt"), safe_open(v.EXPORT + "/model.safetensors", "pt"), safe_open(CTRL, "pt")
dq = lambda f, s: rn.dequantize(f.get_tensor(b + s + ".weight_codes"), f.get_tensor(b + s + ".weight_scale_exp"))
Wu, Wd = stock.get_tensor(b + "up_proj.weight"), stock.get_tensor(b + "down_proj.weight")
Dt, Dc = dq(T, "up_proj"), dq(C, "up_proj")
s, s_in = v.recover(Wu.float(), Wd.float(), Dt, dq(T, "down_proj"))
err = (Dt - Dc).norm(dim=1) / Dc.norm(dim=1).clamp(min=1e-30)
bad = err > 0.1
print("rows where panel export differs from per-channel export by >10%:", int(bad.sum()), "of", Wu.shape[0],
      "; all-zero in panel export:", int((Dt.norm(dim=1) == 0).sum()))
sx = torch.exp((torch.log(s.double()) / 0.8).clamp(-60, 60) - torch.log(s.double()).div(0.8).mean()).float()
R = (Wu.float() * s_in[None, :]).to(Wu.dtype)
print("median log2 s: damaged rows %.2f, all rows %.2f" % (float(torch.log2(s[bad]).median()), float(torch.log2(s).median())))
for lab, a in (("recorded alpha 0.8", 0.8), ("alpha %.2f" % float(sys.argv[1]), float(sys.argv[1]))):
    sv = sx.pow(a).clamp(min=1e-4)
    sv = sv / (sv.max() * sv.min()).sqrt()
    Rs = (R.float() / sv[:, None]).to(R.dtype).float()
    e = v.GRIDS["panel"](Rs)
    Q = ga.qdq(Rs, e)
    ra = Rs.abs().amax(1)
    pa = ra[: ra.shape[0] // 32 * 32].reshape(-1, 32).amax(1).repeat_interleave(32)
    below = torch.log2(pa / ra.clamp(min=1e-30))[bad]
    flushed = ((Q == 0) & (Rs != 0)).float().mean(1)[bad]
    rowerr = ((Q - Rs).norm(dim=1) / Rs.norm(dim=1))[bad]
    print("%-20s damaged rows: binades below panel amax median %.1f max %.1f | flushed-entry fraction mean %.3f max %.3f | RTN row err median %.3f, >10%%: %d"
          % (lab, float(below.median()), float(below.max()), float(flushed.mean()), float(flushed.max()), float(rowerr.median()), int((rowerr > 0.1).sum())))
