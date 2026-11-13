import sys, json, torch
sys.path.insert(0, "src")
import atlas_fx_ops  # noqa
from safetensors.torch import load_file
B = "/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq"
n = int(sys.argv[1]); torch.set_num_threads(n)
m = torch.export.load(B + "/fx/model.pt2").module()
fx = load_file(B + "/fx/fixtures.safetensors"); par = json.load(open(B + "/fx/parity.json"))
res = {}
with torch.no_grad():
    for i, c in enumerate(par["cases"]):
        out = m(*[fx[k] for k in c["inputs"]]); want = fx[c["outputs"][0]]
        torch.save(out, f"{sys.argv[2]}/out_t{n}_c{i}.pt")
        res[i] = {"max_abs_vs_fixture": float((out - want).abs().max()), "n_diff": int((out != want).sum()), "n": out.numel()}
print(json.dumps({"threads": n, "torch_threads": torch.get_num_threads(), "cases": res}), flush=True)
