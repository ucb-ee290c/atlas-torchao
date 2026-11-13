"""Re-quantize the model in memory (same recipe, calibration, seed) and compare with the exported bundle:
codes/exponents bytes and logits of the saved native FX graph. usage: inmem_check.py MODEL RECIPE EXPORT FX CALIB_N SEQLEN_CAL FXSEQ OUT"""
import json, sys, torch
from safetensors import safe_open
import algos, atlas_fx_ops, fx_export  # noqa
from algos import calib as C
from modelutils import load_llama
model_id, recipe, export, fx, n, cal_seq, fx_seq, out = sys.argv[1:9]
n, cal_seq, fx_seq = int(n), int(cal_seq), int(fx_seq)
m = load_llama(model_id, seqlen=cal_seq, device="cpu")
batches, rec = C.wikitext2_train(model_id, n, cal_seq, 1234, use_fast=False)
algo = algos.make(recipe)
algos.quantize(m, "ref_conv", skip=["lm_head"], options={"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"},
               algo=algo, calib=lambda: (batches, lambda mm, ids: mm(ids, use_cache=False), rec))
res = {"recipe": recipe, "codes_equal": {}, "exp_equal": {}}
with safe_open(export + "/model.safetensors", "pt") as f:
    for name, mod in m.named_modules():
        if type(mod).__name__ == "RefQuantLinear":
            res["codes_equal"][name] = bool(torch.equal(f.get_tensor(name + ".weight_codes"), mod.weight_codes.cpu()))
            res["exp_equal"][name] = bool(torch.equal(f.get_tensor(name + ".weight_scale_exp"), mod.weight_scale_exp.cpu()))
prog = torch.export.load(fx + "/model.pt2").module()
fix = None
from safetensors.torch import load_file
fix = load_file(fx + "/fixtures.safetensors")
res["fixture_keys"] = list(fix.keys())[:6]
ids = [torch.randint(m.config.vocab_size, (1, fx_seq), generator=torch.Generator().manual_seed(100 + i)) for i in range(3)]
eq = []
with torch.no_grad():
    for x in ids:
        a = m(x, use_cache=False).logits
        b = prog(x)
        eq.append(bool(torch.equal(a, b)))
        res.setdefault("max_abs_diff", []).append(float((a.float() - b.float()).abs().max()))
res["logits_equal_in_memory_vs_saved_native_graph"] = eq
res["all_codes_equal"] = all(res["codes_equal"].values()) and len(res["codes_equal"]) > 0
res["all_exp_equal"] = all(res["exp_equal"].values())
res["n_sites"] = len(res["codes_equal"])
json.dump({k: v for k, v in res.items() if k not in ("codes_equal", "exp_equal")}, open(out, "w"), indent=1)
print({k: v for k, v in res.items() if k not in ("codes_equal", "exp_equal")})
