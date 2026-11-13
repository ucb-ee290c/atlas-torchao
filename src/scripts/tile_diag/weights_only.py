"""Weights-only comparison of two Pi0 exports against the stock checkpoint (CPU).
usage: weights_only.py OUT.json EXPORT_A.safetensors[:label] EXPORT_B.safetensors[:label] ...
Per quantized Linear: rel Frobenius error of the dequantized weight (e4m3(code) * 2^exp) vs the stock weight, after
undoing the AWQ fold on the up_proj -> down_proj groups (per-up-row scalar s_j fitted by least squares; the up
effective row is dq_up * s_j and the down effective column is dq_down / s_j). Also: mean/max of max/min row-amax
within each 32-row group of the stock weight and of the dequantized weight, fraction of codes that decode to zero
where the stock weight is nonzero, and the stock-weight fraction below the smallest e4m3 normal at that exponent."""
import sys, json, re, torch
sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
from safetensors import safe_open

STOCK = "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors"
out = sys.argv[1]
exports = []
for a in sys.argv[2:]:
    p, _, lab = a.partition(":")
    exports.append((lab or p, safe_open(p, "pt")))
stock = safe_open(STOCK, "pt")


def spread(a, g=32):
    ra = a.abs().amax(1)
    n = (ra.shape[0] // g) * g
    if n == 0:
        return float("nan"), float("nan")
    r = ra[:n].reshape(-1, g)
    sp = r.amax(1) / r.amin(1).clamp(min=1e-30)
    return float(sp.mean()), float(sp.max())


def dq(f, site):
    q = f.get_tensor(site + ".weight_codes")
    e = f.get_tensor(site + ".weight_scale_exp")
    return rn.dequantize(q, e), q, e


sites = sorted(k[:-len(".weight_codes")] for k in exports[0][1].keys() if k.endswith(".weight_codes"))
res = {}
for lab, f in exports:
    cache = {}
    R = {}
    for site in sites:
        if site + ".weight_codes" not in set():
            pass
    keys = set(f.keys())
    for site in sites:
        if site + ".weight_codes" not in keys:
            continue
        W = stock.get_tensor(site + ".weight").float()
        D, q, e = dq(f, site)
        if W.ndim != 2:
            continue
        eff = D
        m = re.match(r"(.*)\.mlp\.(up_proj|down_proj)$", site)
        fold = None
        if m:
            base = m.group(1) + ".mlp."
            Du, _, _ = dq(f, base + "up_proj"); Dd, _, _ = dq(f, base + "down_proj")
            Wu = stock.get_tensor(base + "up_proj.weight").float(); Wd = stock.get_tensor(base + "down_proj.weight").float()
            # dq_up_j ~= Wu_j / s_j  =>  1/s_j = <Du_j,Wu_j>/<Wu_j,Wu_j>? use Du_j*s_j ~= Wu_j  => s_j = <Du_j,Wu_j>/<Du_j,Du_j>
            s = (Du * Wu).sum(1) / (Du * Du).sum(1).clamp(min=1e-30)
            fold = s
            eff = Du * s[:, None] if m.group(2) == "up_proj" else Dd / s[None, :]
        nz = W != 0
        zero_frac = float(((q & 0x7F) == 0)[nz].float().mean())
        err = float((eff - W).norm() / W.norm())
        # dequantized weight in the quantized basis, row-amax spread (what the exponent grid actually sees)
        s_stock = spread(W)
        s_dq = spread(D)
        R[site] = dict(shape=list(W.shape), err=err, zero_frac=zero_frac, stock_spread_mean=s_stock[0], stock_spread_max=s_stock[1],
                       dq_spread_mean=s_dq[0], dq_spread_max=s_dq[1],
                       fold_s_log2_range=(float(torch.log2(fold.abs().clamp(min=1e-30)).max() - torch.log2(fold.abs().clamp(min=1e-30)).min()) if fold is not None else None),
                       n_exp=int(torch.unique(e).numel()), exp_min=int(e.min()), exp_max=int(e.max()))
        print(lab, site.replace("paligemma_with_expert.", ""), shape := tuple(W.shape), "err %.5f" % err, "zero %.4f" % zero_frac, flush=True)
    res[lab] = R
json.dump(res, open(out, "w"), indent=1)
