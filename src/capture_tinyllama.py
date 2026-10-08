import os
import torch
import torch.nn as nn
import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig
from torchao.quantization import quantize_
from atlas_torchao import AtlasQConfig, AtlasQuantLinear

torch.manual_seed(0)

# a tiny "MLP block": 64 -> 128 -> 64 (all multiples of 32, like Atlas tiles)
model = nn.Sequential(
    nn.Linear(64, 128, bias=False),
    nn.SiLU(),
    nn.Linear(128, 64, bias=False),
).to(torch.bfloat16).eval()
x = torch.randn(4, 64, dtype=torch.bfloat16)

os.makedirs("out/tiny", exist_ok=True)

# --- A. plain BF16 model, no quantization (baseline) ---
r0 = m2m.convert(model, (x,), backend="fx_importer",
                 weights_path="out/tiny/bf16_weights.safetensors")
assert r0.ok, r0.diagnostics
open("out/tiny/bf16.mlir", "w").write(r0.mlir_text)

# --- B. same model, quantized with AtlasQConfig ---
original = m2m.capture_frontend_snapshot(model, (x,), stage="original")
with torch.no_grad():
    quantize_(model, AtlasQConfig(), filter_fn=lambda m, fqn: isinstance(m, nn.Linear))
print("quantized layers:", sum(isinstance(m, AtlasQuantLinear) for m in model.modules()))  # 2

r1 = m2m.convert(model, (x,), backend="fx_importer",
                 weights_path="out/tiny/fp8_weights.safetensors",
                 capture_trace=True,
                 quantization=QuantizationConfig(scheme="atlas_fp8_e4m3"),
                 quantization_preapplied=True,
                 original_frontend_snapshot=original)
assert r1.ok, r1.diagnostics
open("out/tiny/fp8.mlir", "w").write(r1.mlir_text)

print("--- FP8 MLIR ---")
print(r1.mlir_text)