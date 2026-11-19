"""Real-text E8M0 codes (127 + nc block exponent) of every Linear activation block of a dynamic a=block export, against 122..128.
usage: dyn_admission.py MODEL_DIR EXPORT_DIR OUT_JSON SEQLEN BATCH"""
import json, sys, torch
import exporter.load_quantized as load_quantized
from algos.granularity import row_group_scale_exp
model_dir, export, out, T, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : B * T].reshape(B, T)
model, _, _ = load_quantized.load_quantized(model_dir, export, "cpu", mode="ref_conv", seqlen=T)
rec = {}
def mk(n):
    def pre(m, inp):
        x = inp[0].detach()
        s = row_group_scale_exp(x.reshape(-1, x.shape[-1]), 32, "nc")[::32, 0] + 127
        rec[n] = s
    return pre
hs = [m.register_forward_pre_hook(mk(n)) for n, m in model.named_modules() if type(m).__name__ == "RefQuantLinear"]
with torch.no_grad():
    model(ids, use_cache=False)
sites = {}
tot = outb = 0
for n, s in rec.items():
    o = int(((s < 122) | (s > 128)).sum())
    sites[n] = {"min": int(s.min()), "max": int(s.max()), "blocks": int(s.numel()), "blocks_outside": o}
    tot += s.numel(); outb += o
res = {"export": export, "tokens": B * T, "blocks_per_site": 16, "sites": len(sites),
       "sites_with_any_block_outside_122_128": sum(1 for v in sites.values() if v["blocks_outside"]),
       "sites_all_blocks_inside": sum(1 for v in sites.values() if not v["blocks_outside"]),
       "blocks_total": tot, "blocks_outside": outb,
       "global_min_code": min(v["min"] for v in sites.values()), "global_max_code": max(v["max"] for v in sites.values()),
       "by_class": {}, "per_site": sites}
for c in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]:
    v = [x for n, x in sites.items() if n.endswith(c)]
    res["by_class"][c] = {"sites": len(v), "sites_with_outside": sum(1 for x in v if x["blocks_outside"]),
                          "min": min(x["min"] for x in v), "max": max(x["max"] for x in v)}
json.dump(res, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_site"})
