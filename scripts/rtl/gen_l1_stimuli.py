"""Stimuli for the L1 RTL run (AtlasTorchaoStimTest.scala): tiles from atlas-torchao's flow.

Groups (each case: 32 activation bytes, 32 weight bytes, a BF16 partial, an E8M0 scale byte):
  chain    real TinyLlama export weight codes against quantized activations, every K tile of a row,
           the partial being our model's previous-tile output (a real K chain)
  scales   all 256 E8M0 scale bytes with random tiles and edge-case partials
  pop      BF16 partials around the FP8 saturation edge (448..496, both signs) and the 2^-6 flush edge
  extreme  max-magnitude and all-equal codes, one huge product among tiny ones, subnormal and NaN codes,
           BF16 subnormal / huge / NaN partials
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accum_torch as at, ref_numerics as rn
import npu_model_crosscheck as xc


def case(a, w, partial, scale, group):
    return {"a": [int(v) for v in a], "w": [int(v) for v in w], "partial": int(partial), "scale": int(scale), "group": group}


def ours_ipt(a, w, partial, lut, h, iw):
    return int(at._ipt_tile(torch.tensor(a, dtype=torch.int32)[None, None], torch.tensor(w, dtype=torch.int32)[None, None],
                            lut, torch.tensor([[[partial]]], dtype=torch.int64), 2, h, iw, "numba", at.OUT_BF16, 0).flatten()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--export", default=xc.DEFAULT_EXPORT)
    args = ap.parse_args()
    g = torch.Generator().manual_seed(11)
    lut = at.e4m3_mul_to_prod_lut(); h, iw = at._ipt_consts(32, "numba")
    cases = []
    # chain: real weight codes
    from safetensors import safe_open
    with safe_open(args.export, "pt") as f:
        names = [k for k in f.keys() if f.get_slice(k).get_dtype() == "U8" and len(f.get_slice(k).get_shape()) == 2]
        for name, rows_n, rows_m in ((names[len(names) // 2], 16, 4), (names[-1], 8, 1)):
            w = f.get_tensor(name)[:rows_n]
            K = w.shape[1]
            xf = torch.randn(rows_m, K, generator=g)
            x = rn.e4m3_vpack(xf.to(torch.bfloat16), rn.po2_scale_exp(xf, dims=-1)).to(torch.uint8)
            for m in range(rows_m):
                for n in range(rows_n):
                    psum = 0
                    for t in range(0, K, 32):
                        a, ws = x[m, t:t + 32].tolist(), w[n, t:t + 32].tolist()
                        cases.append(case(a, ws, psum, 127, "chain"))
                        psum = ours_ipt(a, ws, psum, lut, h, iw)
    # scales
    edges = [0, 0x8000, 0x0001, 0x007F, 0x3F80, 0xBF80, 0x43E0, 0x43F0, 0x7F7F, 0xFF7F, 0x7F80, 0x7FC0]
    for s in range(256):
        for k in range(4):
            partial = edges[(s + k) % len(edges)] if k < 2 else int(xc.rand_bf16_bits(g, (1,), normal_only=False)[0])
            cases.append(case(xc.rand_e4m3(g, (32,)).tolist(), xc.rand_e4m3(g, (32,)).tolist(), partial, s, "scales"))
    # pop: saturation and flush edges
    zero = [0] * 32
    for sign in (0, 0x8000):
        for bits in range(0x43C0, 0x4400):                       # 384 .. 512
            for s in (127, 128, 126):
                cases.append(case(zero, zero, bits | sign, s, "pop"))
        for bits in range(0x3C40, 0x3CC0):                       # around 2^-6
            for s in (127, 130):
                cases.append(case(zero, zero, bits | sign, s, "pop"))
    # extreme
    codes = [0x00, 0x08, 0x01, 0x7E, 0x7F, 0xFE, 0xFF, 0x80, 0x38, 0xB8, 0x77]
    for ca in codes:
        for cw in codes:
            for p in (0, 0x3F80, 0x7F7F, 0x0040):
                cases.append(case([ca] * 32, [cw] * 32, p, 127, "extreme"))
    for _ in range(600):
        a = xc.rand_e4m3(g, (32,)).tolist(); w = xc.rand_e4m3(g, (32,)).tolist()
        i = int(torch.randint(0, 32, (1,), generator=g)); a[i] = 0x7E; w[i] = 0x7E                    # one huge product
        small = [c for c in range(256) if (c >> 3) & 15 in (1, 2)]
        for j in range(32):
            if j != i:
                a[j] = small[int(torch.randint(0, len(small), (1,), generator=g))]
        cases.append(case(a, w, int(xc.rand_bf16_bits(g, (1,), normal_only=False)[0]), 127, "extreme"))
    Path(args.out).write_text(json.dumps({"cases": cases}))
    from collections import Counter
    print(len(cases), dict(Counter(c["group"] for c in cases)))


if __name__ == "__main__":
    main()
