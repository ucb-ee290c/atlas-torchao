import torch

x = torch.randn(4, 32).to(torch.float8_e4m3fn)
w = torch.randn(32, 32).to(torch.float8_e4m3fn)
one = torch.tensor(1.0)

class M(torch.nn.Module):
    def forward(self, x, w):
        return torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16)

ep = torch.export.export(M(), (x, w))
print(ep.graph)        # look for aten._scaled_mm with float8 inputs and a bf16 output

import os, m2m

os.makedirs("out/scaled_mm", exist_ok=True)
for backend in ("fx_importer", "torch_mlir"):
    r = m2m.convert(M(), (x, w), backend=backend,
                    weights_path=f"out/scaled_mm/{backend}_weights.safetensors")
    print(f"\n=== {backend}: ok={r.ok} path={r.path_taken}")
    if not r.ok:
        print("\n".join(r.diagnostics)[:2000]); continue
    open(f"out/scaled_mm/{backend}.mlir", "w").write(r.mlir_text)
    t = r.mlir_text
    print("f8E4M3FN count:", t.count("f8E4M3FN"))
    print("linalg.matmul:", t.count("linalg.matmul"), " func.call (opaque):", t.count("func.call"))