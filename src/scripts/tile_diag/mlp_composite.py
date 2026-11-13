"""Basis-independent error of the AWQ-folded MLP pair: ||Dd @ Du - Wd @ Wu||_F / ||Wd @ Wu||_F, where Du, Dd are the
dequantized up_proj / down_proj of an export and Wu, Wd the stock weights. The per-intermediate-channel AWQ scale s_j
cancels exactly in this product (Dd[:,j]/s_j * s_j Du[j,:]), so no estimate of s is needed. It ignores the gate
nonlinearity (a proxy for the pair's error, not the layer's output error).
usage: mlp_composite.py OUT.json EXPORT.safetensors:label ..."""
import sys, json, torch
sys.path.insert(0, __file__.rsplit("/src/scripts/", 1)[0] + "/src")
import reference.ref_numerics as rn
from safetensors import safe_open
stock = safe_open("/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris/model.safetensors", "pt")
res = {}
for a in sys.argv[2:]:
    p, _, lab = a.partition(":"); f = safe_open(p, "pt"); R = {}
    for k in sorted(f.keys()):
        if not k.endswith("mlp.up_proj.weight_codes"):
            continue
        b = k[:-len("up_proj.weight_codes")]
        dq = lambda s: rn.dequantize(f.get_tensor(s + ".weight_codes"), f.get_tensor(s + ".weight_scale_exp"))
        P0 = stock.get_tensor(b + "down_proj.weight").float() @ stock.get_tensor(b + "up_proj.weight").float()
        P1 = dq(b + "down_proj") @ dq(b + "up_proj")
        R[b.replace("paligemma_with_expert.", "")] = float((P1 - P0).norm() / P0.norm())
        print(lab, list(R.items())[-1], flush=True)
    res[lab] = R
json.dump(res, open(sys.argv[1], "w"), indent=1)
