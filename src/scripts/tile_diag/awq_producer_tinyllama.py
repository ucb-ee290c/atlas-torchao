"""TinyLlama layers 0..N-1: REAL calibration (WikiText-2 train windows) of gptq_awq method=awq on the panel grid,
producer_aware off vs on, and the per-channel control (granularity w=channel, producer_aware off). Reports per layer
the alpha picked for the up_proj -> down_proj group, producer rows whose effective row differs > 10% from the stock
(norm-folded) row, and the MLP output relative L2 error of RTN-cast gate/up/down on held-out windows (the layer
input is the stock residual after attention).
usage: awq_producer_tinyllama.py OUT.json [N_LAYERS=2] [N_CALIB=8] [SEQLEN=256]"""
import copy
import json
import sys

import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0] + "/src")
import algos
from algos import calib as algo_calib
from algos import gptq_awq as ga
from algos.granularity import panel_scale_exp

PATH = "/scratch/agarcia05/TinyLlama-1.1B-Chat-v1.0"
out, NL = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2
NC, SL = (int(sys.argv[3]) if len(sys.argv) > 3 else 8), (int(sys.argv[4]) if len(sys.argv) > 4 else 256)
cfg = AutoConfig.from_pretrained(PATH)
cfg.num_hidden_layers = NL
model = AutoModelForCausalLM.from_pretrained(PATH, config=cfg, torch_dtype=torch.bfloat16).eval()
calib, rec = algo_calib.wikitext2_train(PATH, NC, SL, 1234)
held, _ = algo_calib.wikitext2_train(PATH, 4, SL, 99)
run = lambda m, ids: m(ids, use_cache=False)            # noqa: E731
GR = {"panel": lambda W: panel_scale_exp(W, 32, "nc"), "channel": lambda W: panel_scale_exp(W, 1, "nc")}


def prep(spec):
    st = algos.make(spec)
    member = st.members[0]
    m = copy.deepcopy(model)
    member.calibrate(m, calib, run)
    member.prepare(m, "ref_linear")
    return m, member


def mlp_err(m_folded, grid):
    """held-out MLP output error of the folded model's layers' MLPs, weights cast on `grid`, vs the stock MLP."""
    res = {}
    for i in range(NL):
        cap = {}
        h = model.model.layers[i].post_attention_layernorm.register_forward_pre_hook(lambda mod, a: cap.setdefault("h", []).append(a[0]))
        for b in held:
            run(model, b)
        h.remove()
        L0, L1 = model.model.layers[i], m_folded.model.layers[i]
        g = GR[grid]
        q = lambda w: ga.qdq(w.float(), g(w.float()))        # noqa: E731
        err_n = err_d = 0.0
        for hs in cap["h"]:
            with torch.no_grad():
                ref = L0.mlp(L0.post_attention_layernorm(hs)).float()
                x = L1.post_attention_layernorm(hs).float()
                y = (torch.nn.functional.silu(x @ q(L1.mlp.gate_proj.weight).T) * (x @ q(L1.mlp.up_proj.weight).T)) @ q(L1.mlp.down_proj.weight).T
            err_n += float(((y - ref) ** 2).sum()); err_d += float((ref ** 2).sum())
        res[i] = (err_n / err_d) ** 0.5
    return res


def rows_damaged(m_folded, member, grid, i):
    p = f"model.layers.{i}.mlp."
    s_mid = member.awq_scale[p + "down_proj"].float()
    s_in = member.awq_scale[p + "up_proj"].float()
    Wu = model.model.layers[i].mlp.up_proj.weight.float()
    R = (Wu * s_in[None, :]).to(torch.bfloat16).float()                  # norm-folded stock up_proj
    Fu = m_folded.model.layers[i].mlp.up_proj.weight.float()
    D = ga.qdq(Fu, GR[grid](Fu)) * s_mid[:, None]
    row = (D - R).norm(dim=1) / R.norm(dim=1).clamp(min=1e-30)
    return int((row > 0.1).sum()), int((D.abs().amax(1) == 0).sum()), float(row.median())


res = {"calib": rec, "n_layers": NL, "heldout": {"n": 4, "seed": 99, "seqlen": SL}}
for lab, spec, grid in (("panel_old", "gptq_awq:method=awq+granularity:w=panel,rows=32,rule=nc", "panel"),
                        ("panel_new", "gptq_awq:method=awq,producer_aware=true+granularity:w=panel,rows=32,rule=nc", "panel"),
                        ("channel_old_control", "gptq_awq:method=awq+granularity:w=channel,rule=nc", "channel")):
    m, member = prep(spec)
    e = mlp_err(m, grid)
    R = {}
    for i in range(NL):
        dm, zr, med = rows_damaged(m, member, grid, i)
        R[i] = {"alpha": member.alpha_picks[f"model.layers.{i}.mlp.up_proj->down_proj"]["weight"], "damaged_rows_gt10pct": dm,
                "zero_rows": zr, "median_row_err": med, "mlp_out_rel_err_heldout": e[i]}
    res[lab] = R
    print(lab, json.dumps(R), flush=True)
    json.dump(res, open(out, "w"), indent=1)
