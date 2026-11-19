"""Quality of exported FP8 bundles against the BF16 model on one fixed token batch.

per-layer relative L2   ||y_q - y_ref|| / ||y_ref|| of every quantized Linear output (cumulative: the layer inside the full
                        quantized forward vs the same layer inside the clean BF16 forward, same tokens)
final logits            same measure on the logits, plus top-1 agreement and mean KL(p_ref || p_q)
accumulation models     for first / middle / last layer of each projection class only: the quantized module alone, fed the
                        input it saw in the framework run, executed under the MXU model (sa = MXU0, ipt = MXU1);
                        compared with the BF16 layer output and with the framework (exact fp32 accumulation) output.
usage: quality.py MODEL_DIR TEXT_SEQLEN BATCH LABEL=EXPORT_DIR[=OUT_JSON] ...   (PYTHONPATH=<worktree>/src)"""
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import exporter.load_quantized as load_quantized
from loaders.modelutils import load_llama

model_dir, T, B = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
jobs = [a.split("=") for a in sys.argv[4:]]
ISO_ROWS = 64                       # rows of the batch used for the isolated MXU-model runs (two 32-token blocks)
CLASSES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

t0 = time.time()
from datasets import load_dataset
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids_all = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids
ids = ids_all[:, : B * T].reshape(B, T)
print("batch", tuple(ids.shape), "first ids", ids[0, :6].tolist(), flush=True)


def rel_l2(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


def hook_outputs(model, names, store, keep_inputs=None):
    hs = []
    mods = dict(model.named_modules())
    for n in names:
        def h(m, inp, out, n=n):
            store[n] = out.detach().to(torch.bfloat16 if out.dtype == torch.bfloat16 else out.dtype).clone()
            if keep_inputs is not None and n in keep_inputs:
                keep_inputs[n] = inp[0].detach().clone()
        hs.append(mods[n].register_forward_hook(h))
    return hs


# ---- clean reference ---------------------------------------------------------------------------------------------
clean = load_llama(model_dir, seqlen=T, device="cpu")
lin_names = [n for n, m in clean.named_modules() if isinstance(m, torch.nn.Linear) and n != "lm_head"
             and re.search(r"(q|k|v|o|gate|up|down)_proj$", n)]
ref_out = {}
hs = hook_outputs(clean, lin_names, ref_out)
with torch.no_grad():
    ref_logits = clean(ids, use_cache=False).logits.float()
for h in hs:
    h.remove()
del clean
print(f"clean pass done {time.time() - t0:.0f}s, {len(lin_names)} linears", flush=True)
L = max(int(re.search(r"layers\.(\d+)", n).group(1)) for n in lin_names) + 1
sel_layers = sorted({0, (L - 1) // 2, L - 1})
iso_names = [n for n in lin_names if int(re.search(r"layers\.(\d+)", n).group(1)) in sel_layers]


def ref_probs(lg):
    return F.log_softmax(lg, -1)


for label, export, *outp in jobs:
    t1 = time.time()
    manifest = json.loads((Path(export) / "manifest.json").read_text())
    folds = {g["group"] for g in ((manifest["recipe"]["algorithm_record"].get("prepare", {}).get("gptq_awq") or {})
                                  .get("awq_folded") or [])}
    model, rep, _ = load_quantized.load_quantized(model_dir, export, "cpu", mode="ref_conv", seqlen=T)
    q_out, q_in = {}, {n: None for n in iso_names}
    hs = hook_outputs(model, lin_names, q_out, q_in)
    with torch.no_grad():
        q_logits = model(ids, use_cache=False).logits.float()
    for h in hs:
        h.remove()
    print(f"[{label}] framework pass {time.time() - t1:.0f}s", flush=True)

    # up_proj rows can carry a folded 1/s (AWQ up_proj -> down_proj fold): output = y_ref / s_j per channel.
    # Recover s_j as the per-channel rms ratio and report both the raw and the scale-recovered error.
    rows, recovered = {}, {}
    for n in lin_names:
        yq, yr = q_out[n].float(), ref_out[n].float()
        r = {"rel_l2": rel_l2(yq, yr)}
        li = int(re.search(r"layers\.(\d+)", n).group(1))
        if n.endswith("up_proj") and f"model.layers.{li}.mlp.up_proj->down_proj" in folds:
            ratio = (yq.reshape(-1, yq.shape[-1]).pow(2).mean(0).sqrt() / yr.reshape(-1, yr.shape[-1]).pow(2).mean(0).sqrt().clamp_min(1e-30))
            recovered[n] = ratio
            r["rel_l2_scale_recovered"] = rel_l2(yq / ratio, yr)
            r["awq_row_fold"] = True
        rows[n] = r
    cls = {}
    for c in CLASSES:
        v = [rows[n]["rel_l2"] for n in lin_names if n.endswith(c)]
        v2 = [rows[n].get("rel_l2_scale_recovered", rows[n]["rel_l2"]) for n in lin_names if n.endswith(c)]
        cls[c] = {"layers": len(v), "mean": sum(v) / len(v), "max": max(v), "min": min(v),
                  "mean_scale_recovered": sum(v2) / len(v2)}
    lp, rp = ref_probs(q_logits), ref_probs(ref_logits)
    final = {"rel_l2": rel_l2(q_logits, ref_logits),
             "top1_agree": float((q_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
             "kl_ref_to_quant_mean_per_token": float((rp.exp() * (rp - lp)).sum(-1).mean())}

    # ---- accumulation models, isolated modules ---------------------------------------------------------------------
    acc = {}
    for mxu in ("sa", "ipt"):
        t2 = time.time()
        m2, _, _ = load_quantized.load_quantized(model_dir, export, "cpu", mode="reference_hw", seqlen=T,
                                                 options={"mxu": mxu, "mxu_impl": "torch", "crosscheck": "off"})
        mods = dict(m2.named_modules())
        for n in iso_names:
            x = q_in[n][0, :ISO_ROWS]
            with torch.no_grad():
                y = mods[n](x).float()
            yr = ref_out[n][0, :ISO_ROWS].float()
            yfw = q_out[n][0, :ISO_ROWS].float()
            if n in recovered:
                y = y / recovered[n]
                yfw = yfw / recovered[n]
            acc.setdefault(n, {})[mxu] = {"rel_l2_vs_bf16": rel_l2(y, yr), "rel_l2_vs_framework": rel_l2(y, yfw),
                                          "framework_rel_l2_vs_bf16_same_rows": rel_l2(yfw, yr)}
        del m2
        print(f"[{label}] {mxu} isolated runs {time.time() - t2:.0f}s", flush=True)

    result = {"label": label, "export": export, "model": model_dir, "seconds": time.time() - t1,
              "batch": {"source": "Salesforce/wikitext wikitext-2-raw-v1 test, slow tokenizer, first B*T tokens",
                        "batch_size": B, "seqlen": T, "tokens": B * T, "first_ids": ids[0, :8].tolist()},
              "accumulation_of_per_layer_and_logits": "exact float32 accumulation (ref_conv framework mode); not the MXU model",
              "recipe": manifest["recipe"]["algo"], "manifest_commit": manifest["provenance"]["atlas_torchao_commit"],
              "final_logits": final, "per_class": cls, "per_layer": rows,
              "per_layer_note": "cumulative: layer output inside the full quantized forward vs the same layer inside the "
                                "BF16 forward on the same tokens. up_proj rows folded by AWQ (up_proj->down_proj) are "
                                "scaled by 1/s_j; rel_l2_scale_recovered divides the per-channel rms ratio out first "
                                "(an approximation that absorbs part of the error), rel_l2 is the raw figure.",
              "accumulation_models": {"layers_selected": sel_layers, "rows": ISO_ROWS,
                                      "note": "one module alone under reference_hw (MXU model), input = the input it saw in "
                                              "the framework run, first rows of sequence 0; sa = MXU0, ipt = MXU1; "
                                              "not the whole model; no RTL check", "per_layer": acc}}
    out = Path(outp[0] if outp else Path(export).parent / "quality.json")
    out.write_text(json.dumps(result, indent=1))
    print(f"[{label}] final logits rel_l2 {final['rel_l2']:.4e} top1 {final['top1_agree']:.4f}; wrote {out}", flush=True)
    del model, q_out, q_in
