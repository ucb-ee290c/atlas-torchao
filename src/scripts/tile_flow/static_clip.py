"""Static-recipe activation saturation on real text: for every Linear input, the fraction of values with |x| above the
E4M3 maximum at the stored static exponent (448 * 2^s, so the cast saturates) and the fraction flushed to zero (below 2^(s-6)).
usage: static_clip.py MODEL_DIR EXPORT_DIR OUT_JSON SEQLEN BATCH"""
import json, sys, torch
import exporter.load_quantized as load_quantized
model_dir, export, out, T, B = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
from datasets import load_dataset
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
ids = tok("\n\n".join(test["text"]), return_tensors="pt").input_ids[:, : B * T].reshape(B, T)
model, _, man = load_quantized.load_quantized(model_dir, export, "cpu", mode="ref_conv", seqlen=T)
pol = json.load(open(export + "/policy.json"))
rec = {}
def mk(n, s):
    def pre(m, inp):
        x = inp[0].detach().float().abs()
        rec[n] = {"static_exp": s, "frac_saturated": float((x > 448.0 * 2.0 ** s).float().mean()),
                  "frac_above_max_or_equal_gt_448x2s_count": int((x > 448.0 * 2.0 ** s).sum()),
                  "frac_flushed_to_zero": float(((x < 2.0 ** (s - 6)) & (x > 0)).float().mean()), "amax": float(x.max()),
                  "amax_over_static_range": float(x.max() / (448.0 * 2.0 ** s))}
    return pre
for n, m in model.named_modules():
    if type(m).__name__ == "RefQuantLinear":
        a = pol["sites"][m.layer_name]["activation"]
        assert a["mode"] == "static", (n, a)
        m.register_forward_pre_hook(mk(n, int(a["exp"])))
with torch.no_grad():
    model(ids, use_cache=False)
tot_sat = sum(v["frac_above_max_or_equal_gt_448x2s_count"] for v in rec.values())
res = {"tokens": B * T, "sites": len(rec), "sites_with_any_saturated_value": sum(1 for v in rec.values() if v["frac_above_max_or_equal_gt_448x2s_count"]),
       "values_saturated_total": tot_sat, "max_frac_saturated_site": max(rec.items(), key=lambda kv: kv[1]["frac_saturated"])[0],
       "max_frac_saturated": max(v["frac_saturated"] for v in rec.values()),
       "max_amax_over_static_range": max(v["amax_over_static_range"] for v in rec.values()),
       "mean_frac_flushed_to_zero": sum(v["frac_flushed_to_zero"] for v in rec.values()) / len(rec), "per_site": rec}
json.dump(res, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k != "per_site"})
