"""Capture the SigLIP mlp.fc2 inputs of the stock Pi0 model on one real DROID frame (CPU, no quantization).
usage: siglip_fc2_capture.py OUT.pt
Saves {layer: bf16 [tokens, 4304]} for every SigLIP encoder layer. The forward is stopped once layer 26's fc2 input is seen.
Frame/observation builder is the one hybrid_drift.py uses (DROID episode 252 frame 30)."""
import os, sys, importlib.util
from types import SimpleNamespace
import torch, safetensors.torch
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
CAPSCRIPT = "/scratch2/lorenhung/atlas-bringup/work/representative/capture_scripts/capture_pi0.py"
spec = importlib.util.spec_from_file_location("cap_pi0_old", CAPSCRIPT); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
cfg = SimpleNamespace(paligemma_variant="gemma_2b", action_expert_variant="gemma_300m", pi05=False, dtype="bfloat16", action_dim=32,
                      action_horizon=10, max_token_len=100, pytorch_compile_mode=None)
model = PI0Pytorch(cfg); safetensors.torch.load_model(model, str(old.CKPT / "model.safetensors"), device="cpu"); model.eval()
obs, ometa = old.droid_obs(); mobs, smeta = old.build_model_obs(obs, "cpu")
vt = model.paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers
cap = {}
class Stop(Exception): pass
def mk(l):
    def pre(m, args):
        if l in cap: return                      # first call only (the vision tower runs once per image)
        x = args[0].detach(); cap[l] = (cap.get(l, []) + [x.reshape(-1, x.shape[-1]).to(torch.bfloat16).clone()])
        if l == len(vt) - 1: raise Stop()
    return pre
# one call per image: collect all images' tokens by not stopping at first call; stop after the last layer of the last image
calls = {l: [] for l in range(len(vt))}
def mk2(l):
    def pre(m, args):
        x = args[0].detach(); calls[l].append(x.reshape(-1, x.shape[-1]).to(torch.bfloat16).clone())
        if l == len(vt) - 1 and len(calls[l]) >= int(os.environ.get("NIMG", "2")): raise Stop()
    return pre
for l in range(len(vt)): vt[l].mlp.fc2.register_forward_pre_hook(mk2(l))
noise = torch.randn((1, 10, 32), generator=torch.Generator().manual_seed(0), dtype=torch.float32)
try:
    with torch.no_grad(): model.sample_actions("cpu", mobs, noise=noise, num_steps=1)
except Stop: pass
out = {l: torch.cat(v) for l, v in calls.items()}
print({l: tuple(v.shape) for l, v in out.items() if l in (0, 5, 23, 26)}, flush=True)
torch.save(out, sys.argv[1])
