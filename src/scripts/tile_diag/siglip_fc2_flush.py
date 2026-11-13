"""Why the SigLIP mlp.fc2 layers of the Pi0 GPTQ exports decode mostly to zero (CPU, weights + optional captured inputs).
usage: siglip_fc2_flush.py OUT.json [FC2_INPUTS.pt]   (FC2_INPUTS.pt from siglip_fc2_capture.py)
Per SigLIP fc2 layer (stock weight [1152, 4304], out x in):
  stock stats : per-row amax / median|w|, share of rows with a huge outlier, log2(|w| / row amax) quantiles, flush share at the exponent
  export      : zero share of the codes, fully zero columns (input channels), error of the decoded export vs stock
  options     : plain RTN on the panel grid (same nc exponent as the export), RTN per row (not tile-native), static_calib's mse rule per tensor,
                mse per panel (own generalisation: algos.static_calib.search on each 32-row panel), BF16 (stock weight)
With FC2_INPUTS.pt: fc2 input channels with exactly zero energy (diag H == 0, the test algos.gptq_awq.gptq uses to zero a weight column),
  and the layer-output rel error ||X (Wd - W)^T|| / ||X W^T|| on those real inputs and on Gaussian inputs."""
import sys, json, math, torch
sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
from algos.granularity import panel_scale_exp, scale_exp
from algos.static_calib import bf16_abs_hist, sq_err, search, weight_exponent
from safetensors import safe_open

STOCK = "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors"
EXPORT = "/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq/export/model.safetensors"
SITE = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.%d.mlp.fc2"
out = sys.argv[1]
X = torch.load(sys.argv[2]) if len(sys.argv) > 2 else None
OLD = "/scratch2/lorenhung/atlas-fp8/exports-gpu/pi0_atlas_v1_awq/model.safetensors"      # per-channel nc, awq_gptq
f, g, go = safe_open(STOCK, "pt"), safe_open(EXPORT, "pt"), safe_open(OLD, "pt")
torch.manual_seed(0)
G = torch.randn(1024, 4304)


def rel(a, b):
    return float((a - b).norm() / b.norm())


def cast(W, s):                      # decoded E4M3 x 2^s weight, s [out,1] or scalar
    s = torch.as_tensor(s, dtype=torch.int32)
    q = rn.e4m3_vpack(W, s)
    return rn.dequantize(q, s.expand(W.shape[0], 1) if s.dim() < 2 else s), q


def flushed(q, W):
    return float((((q & 0x7F) == 0) & (W != 0)).float().sum() / (W != 0).float().sum())


def mse_panel(W, rows=32):
    base = panel_scale_exp(W, rows, "nc").reshape(-1)[::rows]
    out_s = []
    for i, s0 in enumerate(base.tolist()):
        P = W[i * rows:(i + 1) * rows]
        out_s.append(search(bf16_abs_hist(P), int(rn.po2_scale_exp(P)), sq_err, -3, 2))
    return torch.tensor(out_s, dtype=torch.int32).repeat_interleave(rows).reshape(-1, 1)


def stats(W):
    A = W.abs().float(); ra = A.amax(1); med = A.median(1).values
    ratio = ra / med
    lg = torch.log2(A / ra[:, None]).flatten()
    qs = torch.tensor([0.001, 0.01, 0.1, 0.5, 0.9])
    # column concentration of the row maxima
    am = A.argmax(1); top = torch.bincount(am, minlength=A.shape[1]).float()
    cmax = A.amax(0)
    return dict(rowamax_min=float(ra.min()), rowamax_med=float(ra.median()), rowamax_max=float(ra.max()),
                rowmedian_med=float(med.median()), ratio_q=[float(v) for v in torch.quantile(ratio, torch.tensor([0, .5, .9, .99, 1.]))],
                rows_ratio_gt_1e3=float((ratio > 1e3).float().mean()), rows_ratio_gt_100=float((ratio > 100).float().mean()),
                rows_amax_gt_20x_p99=float((ra > 20 * torch.quantile(A, 0.99, dim=1)).float().mean()),
                log2_w_over_rowamax_q=dict(zip(["p0.1", "p1", "p10", "p50", "p90"], [float(v) for v in torch.quantile(lg[torch.randperm(lg.numel())[:2_000_000]], qs)])),
                share_log2_below_minus14=float((lg < -14).float().mean()),
                distinct_argmax_cols=int((top > 0).sum()), top1_col_share_of_rowmax=float(top.max() / A.shape[0]),
                col_amax_max_over_median=float(cmax.max() / cmax.median()))


res = {}
for L in range(27):
    W = f.get_tensor(SITE % L + ".weight").float()
    q = g.get_tensor(SITE % L + ".weight_codes"); e = g.get_tensor(SITE % L + ".weight_scale_exp")
    D = rn.dequantize(q, e)
    dead_cols = ((q & 0x7F) == 0).all(0)
    s_panel = panel_scale_exp(W, 32, "nc"); s_row = scale_exp(W, (-1,), "nc")
    R = dict(shape=list(W.shape), export_exp_range=[int(e.min()), int(e.max())], export_matches_panel_nc=bool(torch.equal(s_panel.reshape(-1), e.reshape(-1))),
             export_zero_share_codes=float(((q & 0x7F) == 0).float().mean()), export_flushed_fraction=flushed(q, W),
             export_dead_cols=int(dead_cols.sum()), export_err=rel(D, W))
    Dp, qp = cast(W, s_panel); Dr, qr = cast(W, s_row)
    s_t = weight_exponent(W, "mse"); Dt, qt = cast(W, torch.tensor(s_t))
    s_mp = mse_panel(W); Dm, qm = cast(W, s_mp)
    Dfix = torch.where(dead_cols[None, :], Dp, D)       # export with only its all-zero columns re-cast by RTN on the same panel grid
    opts = {"export": D, "rtn_panel_nc": Dp, "rtn_row_nc": Dr, "mse_tensor": Dt, "mse_panel": Dm, "export_deadcols_rtn": Dfix}
    qo = go.get_tensor(SITE % L + ".weight_codes"); dead_old = ((qo & 0x7F) == 0).all(0)
    R["old_export"] = dict(zero_share_codes=float(((qo & 0x7F) == 0).float().mean()), dead_cols=int(dead_old.sum()),
                           dead_cols_in_both=int((dead_old & dead_cols).sum()), err=rel(rn.dequantize(qo, go.get_tensor(SITE % L + ".weight_scale_exp")), W))
    R["panel_exp_range"] = [int(s_panel.min()), int(s_panel.max())]; R["row_exp_range"] = [int(s_row.min()), int(s_row.max())]
    R["panel_vs_row_exp_diff_mean"] = float((s_panel - s_row).float().mean()); R["mse_tensor_exp"] = s_t
    R["mse_panel_exp_range"] = [int(s_mp.min()), int(s_mp.max())]
    R["flushed"] = {"rtn_panel_nc": flushed(qp, W), "rtn_row_nc": flushed(qr, W), "mse_tensor": flushed(qt, W), "mse_panel": flushed(qm, W)}
    R["weight_err"] = {k: rel(v, W) for k, v in opts.items()}
    R["gauss_out_err"] = {k: rel(G @ v.T, G @ W.T) for k, v in opts.items()}
    if X is not None:
        x = X[L].float()
        xe = (x ** 2).sum(0)
        R["input"] = dict(tokens=int(x.shape[0]), dead_channels_diagH0=int((xe == 0).sum()), amax=float(x.abs().max()),
                          chan_amax_max_over_median=float(x.abs().amax(0).max() / x.abs().amax(0).median().clamp(min=1e-30)),
                          chan_amax_top5=[float(v) for v in x.abs().amax(0).topk(5).values],
                          export_dead_cols_that_are_dead_in_this_frame=int((dead_cols & (xe == 0)).sum()),
                          export_dead_cols_alive_in_this_frame=int((dead_cols & (xe > 0)).sum()),
                          energy_share_in_export_dead_cols=float(xe[dead_cols].sum() / xe.sum()))
        h = x.shape[0] // 2
        R["input"]["alive_channels_image0"] = int((x[:h].abs().amax(0) > 0).sum()); R["input"]["alive_channels_image1"] = int((x[h:].abs().amax(0) > 0).sum())
        R["input"]["alive_channels_union"] = int((x.abs().amax(0) > 0).sum()); R["input"]["export_alive_channels"] = int((~dead_cols).sum())
        R["input"]["zero_share"] = float((x == 0).float().mean())
        R["real_out_err"] = {k: rel(x @ v.T, x @ W.T) for k, v in opts.items()}
    if L in (5, 23, 17):
        R["stock_stats"] = stats(W)
    res[L] = R
    print(L, "exp_zero %.4f dead_cols %d err %.4f | rtn_panel %.4f rtn_row %.4f mse_t %.4f mse_p %.4f | flush rtn_panel %.5f"
          % (R["export_zero_share_codes"], R["export_dead_cols"], R["export_err"], R["weight_err"]["rtn_panel_nc"], R["weight_err"]["rtn_row_nc"],
             R["weight_err"]["mse_tensor"], R["weight_err"]["mse_panel"], R["flushed"]["rtn_panel_nc"]) +
          ((" | in dead(diagH0) %d, real out err export %.4f rtn_panel %.4f" % (R["input"]["dead_channels_diagH0"], R["real_out_err"]["export"], R["real_out_err"]["rtn_panel_nc"])) if X is not None else ""), flush=True)
json.dump(res, open(out, "w"), indent=1)
