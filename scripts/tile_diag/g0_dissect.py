"""Per-row / per-column comparison of one AWQ up/down pair between a panel export and a per-channel export that chose the same alpha
(same s_j, same calibration): the dequantized up rows (and down columns) approximate W/s_j (W*s_j) in both, so they are compared directly.
usage: g0_dissect.py LAYER_PREFIX PANEL.safetensors CHANNEL.safetensors"""
import sys, torch
sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
from safetensors import safe_open
b, T, C = sys.argv[1], safe_open(sys.argv[2], "pt"), safe_open(sys.argv[3], "pt")
st = safe_open("/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors", "pt")
dq = lambda f, s: rn.dequantize(f.get_tensor(s + ".weight_codes"), f.get_tensor(s + ".weight_scale_exp"))
for nm in ("up_proj", "down_proj"):
    s = b + nm
    Dt, Dc, W = dq(T, s), dq(C, s), st.get_tensor(s + ".weight").float()
    ax = 1 if nm == "up_proj" else 0   # reduce over the dim that holds one intermediate channel's weights
    n = lambda x: x.norm(dim=ax)
    err_t_vs_c = n(Dt - Dc) / n(Dc).clamp(min=1e-30)
    zero_rows = (n(Dt) == 0)
    print(nm, W.shape, "units = %s" % ("rows (intermediate channel j)" if ax == 1 else "columns (intermediate channel j)"))
    print("  units whose tile weights differ from channel weights by >10%%: %.4f, >50%%: %.4f, ==100%% (tile all-zero): %.4f; channel all-zero: %.4f" % (
        float((err_t_vs_c > .1).float().mean()), float((err_t_vs_c > .5).float().mean()), float(zero_rows.float().mean()), float((n(Dc) == 0).float().mean())))
    amax_c = Dc.abs().amax(ax)
    if nm == "up_proj":
        g = amax_c[: amax_c.shape[0] // 32 * 32].reshape(-1, 32)
        sp = g.amax(1) / g.amin(1).clamp(min=1e-30)
        print("  channel-export dq row-amax spread inside 32-row groups (max/min): median %.1f p90 %.0f p99 %.0f max %.0f; groups with spread>2^8: %.3f" % (
            float(sp.median()), float(sp.quantile(.9)), float(sp.quantile(.99)), float(sp.max()), float((sp > 256).float().mean())))
        e = T.get_tensor(s + ".weight_scale_exp").flatten(); ec = C.get_tensor(s + ".weight_scale_exp").flatten()
        print("  exponents tile min/max %d/%d, channel min/max %d/%d; channel exponents spread within 32-row groups: max %d binades" % (
            e.min(), e.max(), ec.min(), ec.max(), int((ec[: len(ec)//32*32].reshape(-1, 32).amax(1) - ec[: len(ec)//32*32].reshape(-1, 32).amin(1)).max())))
    else:
        print("  down: channel-export exps min/max %d/%d, tile %d/%d" % (C.get_tensor(s+'.weight_scale_exp').min(), C.get_tensor(s+'.weight_scale_exp').max(), T.get_tensor(s+'.weight_scale_exp').min(), T.get_tensor(s+'.weight_scale_exp').max()))
    # contribution to the product error: energy of (Dt-Dc) in units of the channel product weights
    print("  zero frac among stock nonzero: tile %.4f channel %.4f" % (float(((Dt == 0) & (W != 0)).sum() / (W != 0).sum()), float(((Dc == 0) & (W != 0)).sum() / (W != 0).sum())))
