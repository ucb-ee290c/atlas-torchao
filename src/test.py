# repro_fp8_lost.py  — run in the atlas312 env
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import m2m


class FP8Linear(nn.Module):
    """Minimal: FP8 weight buffer + FP8 activation cast, then a matmul."""
    def __init__(self):
        super().__init__()
        w = torch.randn(32, 32, dtype=torch.bfloat16) * 0.1
        self.register_buffer("w_fp8", w.to(torch.float8_e4m3fn))     # weight stored as FP8

    def forward(self, x):
        x_q = x.clamp(-448, 448).to(torch.float8_e4m3fn)             # activation -> FP8
        return F.linear(x_q.to(torch.bfloat16), self.w_fp8.to(torch.bfloat16))


model = FP8Linear().eval()
x = torch.randn(4, 32, dtype=torch.bfloat16)
os.makedirs("out/repro", exist_ok=True)

r = m2m.convert(model, (x,), backend="fx_importer",
                weights_path="out/repro/weights.safetensors")
assert r.ok, r.diagnostics
open("out/repro/fp8.mlir", "w").write(r.mlir_text)

# --- what to check ---
sig = r.mlir_text.split("function_type = ")[1].split("}>")[0]
print("function signature:", sig)
print("f8E4M3FN occurrences in MLIR:", r.mlir_text.count("f8E4M3FN"))
print("truncf-to-f8 present:", "to f8E4M3FN" in r.mlir_text or "-> f8E4M3FN" in r.mlir_text)

from safetensors import safe_open
with safe_open("out/repro/weights.safetensors", "pt") as f:
    for k in f.keys():
        print("weights file:", k, f.get_slice(k).get_dtype(), f.get_slice(k).get_shape())