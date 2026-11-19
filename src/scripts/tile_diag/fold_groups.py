"""AWQ up_proj -> down_proj fold groups of a Pi0 export: recover the per-j scale s_j by two independent robust (median)
estimators, one from the up row j (dq_up_j * s_j ~= W_up_j) and one from the down column j (dq_down_j / s_j ~= W_down_j),
then report effective-weight errors vs stock for up and down, agreement of the two estimators, the log2 range of s,
and (for up) the log2 range of row amax inside 32-row groups of the dequantized (quantized-basis) weight.
usage: fold_groups.py OUT.json EXPORT.safetensors:label ..."""
import sys, json, torch
sys.path.insert(0, __file__.rsplit("/src/scripts/", 1)[0] + "/src")
import reference.ref_numerics as rn
from safetensors import safe_open
stock = safe_open("/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors", "pt")
out = sys.argv[1]; res = {}
def colmedian(r, mask):  # median over dim 0 of r where mask, per column; nan if none
    r = torch.where(mask, r, torch.full_like(r, float("nan")))
    return torch.nanmedian(r, dim=0).values
for a in sys.argv[2:]:
    p, _, lab = a.partition(":"); f = safe_open(p, "pt"); R = {}
    bases = sorted(k[:-len("up_proj.weight_codes")] for k in f.keys() if k.endswith("mlp.up_proj.weight_codes"))
    for b in bases:
        def dq(s): return rn.dequantize(f.get_tensor(s + ".weight_codes"), f.get_tensor(s + ".weight_scale_exp"))
        Du, Dd = dq(b + "up_proj"), dq(b + "down_proj")
        Wu, Wd = stock.get_tensor(b + "up_proj.weight").float(), stock.get_tensor(b + "down_proj.weight").float()
        mu = (Du != 0) & (Wu != 0); md = (Dd != 0) & (Wd != 0)
        s_up = colmedian((Wu / Du.where(Du != 0, torch.ones_like(Du))).abs().T, mu.T.contiguous()).T if False else None
        # per up row j: median over columns i of |Wu_ji / Du_ji|
        r = torch.where(mu, (Wu / torch.where(Du != 0, Du, torch.ones_like(Du))).abs(), torch.full_like(Wu, float("nan")))
        s_u = torch.nanmedian(r, dim=1).values
        # per down column j: median over rows i of |Dd_ij / Wd_ij|
        r = torch.where(md, (Dd / torch.where(Wd != 0, Wd, torch.ones_like(Wd))).abs(), torch.full_like(Wd, float("nan")))
        s_d = torch.nanmedian(r, dim=0).values
        both = torch.isfinite(s_u) & torch.isfinite(s_d)
        lg = (torch.log2(s_u[both]) - torch.log2(s_d[both]))
        s = torch.where(torch.isfinite(s_d), s_d, s_u)
        s = torch.where(torch.isfinite(s), s, torch.ones_like(s))
        eu = float((Du * s[:, None] - Wu).norm() / Wu.norm()); ed = float((Dd / s[None, :] - Wd).norm() / Wd.norm())
        ra = Du.abs().amax(1); n = ra.shape[0] // 32 * 32; g = ra[:n].reshape(-1, 32)
        sp = g.amax(1) / g.amin(1).clamp(min=1e-30)
        ls = torch.log2(s)
        R[b.replace("paligemma_with_expert.", "")] = dict(
            err_up=eu, err_down=ed, s_log2_range=float(ls.max() - ls.min()), s_log2_p5_p95=[float(torch.quantile(ls, .05)), float(torch.quantile(ls, .95))],
            est_agree_median_abs_log2=float(lg.abs().median()), est_agree_p95_log2=float(lg.abs().quantile(.95)), n_both=int(both.sum()), n=int(s.numel()),
            up_dq_rowamax_spread_median=float(sp.median()), up_dq_rowamax_spread_p95=float(sp.quantile(.95)),
            up_zero_frac=float(((Du == 0) & (Wu != 0)).float().sum() / (Wu != 0).sum()), down_zero_frac=float(((Dd == 0) & (Wd != 0)).float().sum() / (Wd != 0).sum()))
        print(lab, b.replace("paligemma_with_expert.", ""), "up %.4f down %.4f srange %.1f agree %.3f" % (eu, ed, R[b.replace("paligemma_with_expert.", "")]["s_log2_range"], R[b.replace("paligemma_with_expert.", "")]["est_agree_median_abs_log2"]), flush=True)
    res[lab] = R
json.dump(res, open(out, "w"), indent=1)
