import os
import torch
import torch.nn as nn
from datautils import get_wikitext2
import m2m
from m2m.capture.torchao_pipeline import QuantizationConfig
from torchao.quantization import quantize_
from atlas_torchao import AtlasQConfig, AtlasQuantLinear, skip_lm_head
from modelutils import load_llama


TINY_LLAMA = "/rscratch/agarcia/TinyLlama-1.1B-Chat-v1.0"
model = load_llama(TINY_LLAMA, seqlen=512)

trainloader, testenc = get_wikitext2(nsamples=4, seed=0, seqlen=512, model=TINY_LLAMA)

model = load_llama(TINY_LLAMA, seqlen=512)   
dev = next(model.parameters()).device
ids = trainloader[0][0].to(dev)

r0 = m2m.convert(model, (ids,), backend="torch_mlir", allow_fallback=False)
print(r0.path_taken, r0.ok, r0.diagnostics)
assert r0.ok, r0.diagnostics
open("out/tiny/bf16.mlir", "w").write(r0.mlir_text)


original = m2m.capture_frontend_snapshot(model, (ids,), stage="original")
with torch.no_grad():
    quantize_(model, AtlasQConfig(), filter_fn=skip_lm_head)
print("quantized layers:", sum(isinstance(m, AtlasQuantLinear) for m in model.modules()))  # 2

r1 = m2m.convert(model, (ids,), backend="torch_mlir",
                 weights_path="out/tiny/fp8_weights.safetensors",
                 capture_trace=True,
                 quantization=QuantizationConfig(scheme="atlas_fp8_e4m3"),
                 quantization_preapplied=True,
                 original_frontend_snapshot=original)

assert r1.ok, r1.diagnostics
open("out/tiny/fp8.mlir", "w").write(r1.mlir_text)
