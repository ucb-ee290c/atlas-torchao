"""Real-tensor check of gptq_awq:dead_columns on Pi0 SigLIP layer L mlp.fc2 (CPU, stock weight + captured fc2 inputs).
usage: gptq_dead_columns_validate.py OUT.json [LAYER=23]   (FC2_INPUTS.pt from siglip_fc2_capture.py)
H = x^T x of the captured fc2 inputs (float32, as algos.gptq_awq.calibrate), GPTQ at the panel-nc exponents (32 rows) of the stock weight,
block=128, damp=0.01, actorder=True, as the export. Two calibration sets: the whole captured frame (in-sample) and image 0 only (image 1 held out)."""
import json, sys, time, torch
sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
import tile_native as tn
from algos import gptq_awq as ga
from algos.granularity import panel_scale_exp
from safetensors import safe_open

STOCK = "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors"
EXPORT = "/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq/export/model.safetensors"
SITE = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.%d.mlp.fc2"
FC2_IN = "/scratch2/lorenhung/atlas-fp8/tmp_fc2/fc2_in.pt"
out = sys.argv[1]
L = int(sys.argv[2]) if len(sys.argv) > 2 else 23
rel = lambda a, b: float((a - b).norm() / b.norm())
W = safe_open(STOCK, "pt").get_tensor(SITE % L + ".weight").float()
s = panel_scale_exp(W, 32, "nc")
X = torch.load(FC2_IN)[L].float()
G = torch.randn(1024, W.shape[1], generator=torch.Generator().manual_seed(0))
q_ex = safe_open(EXPORT, "pt").get_tensor(SITE % L + ".weight_codes")
export_zero_cols = int(((q_ex & 0x7F) == 0).all(0).sum())
plain = ga.qdq(W, s)
h = X.shape[0] // 2
res = {"layer": L, "shape": list(W.shape), "tokens": int(X.shape[0]), "export_zero_columns": export_zero_cols,
       "panel_exp_range": [int(s.min()), int(s.max())], "exp_constant_per_32": tn.check_constant_groups(s, 32).ok}


def metrics(Q, Xe_list):
    codes = rn.e4m3_vpack(Q, s)
    zc = int((Q == 0).all(0).sum())
    r = {"zero_columns": zc, "weight_err": rel(Q, W), "gauss_out_err": rel(G @ Q.T, G @ W.T),
         "reserved_codes": tn.reserved_code_count(codes), "on_grid": bool(torch.equal(ga.qdq(Q, s), Q))}
    for k, x in Xe_list.items():
        r[k] = rel(x @ Q.T, x @ W.T)
    return r


for name, Xc, evals in (("calib_full_frame", X, {"real_out_err_calib_frame": X}),
                        ("calib_image0", X[:h], {"real_out_err_image0_calib": X[:h], "real_out_err_image1_heldout": X[h:]})):
    H = Xc.T @ Xc
    R = {"dead_columns_in_H": int((torch.diag(H) == 0).sum()), "rtn_same_exponents": metrics(plain, evals)}
    for mode in ("zero", "rtn"):
        t0 = time.time()
        info = {}
        Q = ga.gptq(W, H, s, 128, 0.01, True, info=info, **({} if mode == "zero" else {"dead_columns": "rtn"}))
        R[mode] = metrics(Q, evals) | {"dead": info["dead"], "damp": info["damp"], "seconds": round(time.time() - t0, 1)}
        if mode == "zero":
            Qz = Q
        else:
            live = torch.diag(H) != 0
            R["live_columns_bit_identical_rtn_vs_zero"] = bool(torch.equal(Q[:, live], Qz[:, live]))
            R["dead_columns_equal_plain_rtn"] = bool(torch.equal(Q[:, ~live], plain[:, ~live]))
    res[name] = R
    print(name, json.dumps(R, indent=1), flush=True)
json.dump(res, open(out, "w"), indent=1)
