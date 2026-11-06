"""Layer-9 / layer-17 Gemma q-input drift of the stock Pi0 model carrying dequantized export weights, measured as the
peer session did (/scratch2/lorenhung/atlas-bringup/work/asis-layers/tile/pi0_capture_tile.py: DROID episode 252 frame 30,
flow noise seed 0, hooks on the first call, no activation quantization), rel L2 against the stock capture
(.../asis-layers/pi0/capture/orig/gem.L<l>.q_in.npy). Forward is stopped right after the layer-17 q_proj input is seen.
usage: hybrid_drift.py BASE.safetensors [--swap OTHER.safetensors PREFIX[,PREFIX...]]...
  every quantized weight comes from BASE, except tensors whose name contains one of the PREFIXes: they come from OTHER.
  (swap both members of a pair: mlp.up_proj and mlp.down_proj of the same layer)."""
import os, sys, json, time, importlib.util, argparse
from types import SimpleNamespace
import numpy as np, torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
ap = argparse.ArgumentParser(); ap.add_argument("base"); ap.add_argument("--swap", nargs=2, action="append", default=[], metavar=("OTHER", "PREFIXES"))
ap.add_argument("--out", default=None); a = ap.parse_args()
CAPSCRIPT = "/scratch2/lorenhung/atlas-bringup/work/representative/capture_scripts/capture_pi0.py"
ORIG = "/scratch2/lorenhung/atlas-bringup/work/asis-layers/pi0/capture/orig/"
from safetensors import safe_open

sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/atlas-torchao-tile/src")
import ref_numerics as rn
spec = importlib.util.spec_from_file_location("cap_pi0_old", CAPSCRIPT); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
import safetensors.torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
cfg = SimpleNamespace(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", pi05=False, dtype="bfloat16", action_dim=32,
                      action_horizon=10, max_token_len=100, pytorch_compile_mode=None)
model = PI0Pytorch(cfg); safetensors.torch.load_model(model, str(old.CKPT / "model.safetensors"), device="cpu"); model.eval()
obs, ometa = old.droid_obs(); mobs, smeta = old.build_model_obs(obs, "cpu")
files = {"base": safe_open(a.base, "pt")}
swaps = []
for i, (other, pre) in enumerate(a.swap):
    files[f"s{i}"] = safe_open(other, "pt"); swaps.append((f"s{i}", pre.split(",")))
params = dict(model.named_parameters()); n = 0; nswap = 0
with torch.no_grad():
    for k in files["base"].keys():
        if not k.endswith(".weight_codes"):
            continue
        site = k[:-len(".weight_codes")]
        src = "base"
        for tag, pres in swaps:
            if any(p in site for p in pres):
                src = tag; nswap += 1
        f = files[src]
        d = rn.dequantize(f.get_tensor(site + ".weight_codes"), f.get_tensor(site + ".weight_scale_exp"))
        p = params[site + ".weight"]; p.copy_(d.reshape(p.shape).to(p.dtype)); n += 1
print("loaded", n, "dequantized weights; swapped", nswap, flush=True)
lm = model.paligemma_with_expert.paligemma.model.language_model
cap = {}
class Stop(Exception): pass
def hook(l):
    def pre(m, args):
        if f"L{l}" in cap: return
        x = args[0]; x = x[0] if x.dim() == 3 else x
        cap[f"L{l}"] = x.detach().to(torch.bfloat16).float().numpy()
        if l == 17: raise Stop()
    return pre
for l in (0, 9, 17): lm.layers[l].self_attn.q_proj.register_forward_pre_hook(hook(l))
noise = torch.randn((1, 10, 32), generator=torch.Generator().manual_seed(0), dtype=torch.float32)
t = time.time()
try:
    with torch.no_grad(): model.sample_actions("cpu", mobs, noise=noise, num_steps=1)
except Stop: pass
print("forward", round(time.time() - t), "s", flush=True)
def words(w):  # uint16 bf16 words -> float32
    return (w.astype(np.uint32) << 16).view(np.float32)
res = {}
for l in (0, 9, 17):
    ref = words(np.load(ORIG + f"gem.L{l}.q_in.npy")); x = cap[f"L{l}"]
    res[f"L{l}"] = float(np.linalg.norm(x - ref) / np.linalg.norm(ref)); print(f"gem L{l} q_in rel L2 vs stock", res[f"L{l}"], flush=True)
if a.out: json.dump(dict(base=a.base, swaps=a.swap, n_swapped=nswap, drift=res), open(a.out, "w"), indent=1)
