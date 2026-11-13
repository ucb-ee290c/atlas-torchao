"""SigLIP mlp.fc2 layers of Pi0 exports against the stock weight, on the captured real inputs (CPU).
usage: fc2_export_compare.py OUT.json NAME=export/model.safetensors [NAME=...]   (inputs: tmp_fc2/fc2_in.pt from siglip_fc2_capture.py)
Per layer 17..25 and export: columns (input channels) whose decoded weight is all zero, share of decoded zeros among nonzero stock weights,
weight rel Frobenius error, and the layer-output rel error ||X (Wd - W)^T|| / ||X W^T|| on the captured frame (512 tokens = 2 images x 256),
on image 0 and image 1 separately, and on Gaussian inputs. The captured frame (DROID episode 252 frame 30) is an evaluated episode, so it is
not in any export's calibration set: these are held-out inputs for the exports."""
import json, sys, torch
sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
from safetensors import safe_open

STOCK = "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors"
SITE = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.%d.mlp.fc2"
FC2_IN = "/scratch2/lorenhung/atlas-fp8/tmp_fc2/fc2_in.pt"
out = sys.argv[1]
exports = [a.split("=", 1) for a in sys.argv[2:]]
rel = lambda a, b: float((a - b).norm() / b.norm())
fs = safe_open(STOCK, "pt")
files = {n: safe_open(p, "pt") for n, p in exports}
X = torch.load(FC2_IN)
G = torch.randn(1024, 4304, generator=torch.Generator().manual_seed(0))
res = {}
for L in range(17, 26):
    W = fs.get_tensor(SITE % L + ".weight").float()
    x = X[L].float()
    xs = {"frame": x, "image0": x[:256], "image1": x[256:], "gauss": G}
    R = {"input_dead_channels_frame": int(((x * x).sum(0) == 0).sum())}
    for n, f in files.items():
        q = f.get_tensor(SITE % L + ".weight_codes"); s = f.get_tensor(SITE % L + ".weight_scale_exp")
        D = rn.dequantize(q, s)
        nz = W != 0
        R[n] = {"zero_columns": int((D == 0).all(0).sum()), "decoded_zero_share_of_nonzero": float(((D == 0) & nz).float().sum() / nz.float().sum()),
                "weight_err": rel(D, W), **{"out_err_" + k: rel(v @ D.T, v @ W.T) for k, v in xs.items()}}
    res[L] = R
    print(L, json.dumps(R), flush=True)
json.dump(res, open(out, "w"), indent=1)
