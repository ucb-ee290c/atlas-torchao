"""Differential check of atlas-torchao's Atlas numerics against npu-model's RTL-fitted numerics.

npu-model (`npu_model/configs/rtl_math.py`, selected by `--numerics rtl`) is independent of
pi0-quant: its integer datapaths follow E4M3Mul, E4M3ProdAddBF16 and AnchorAccumulationTree, and
its unary BF16 tables are exhaustive outputs of the RTL. This script feeds the same bits to both
sides and counts disagreements. Every comparison is bit for bit (NaN compares as "both NaN").

    python scripts/npu_model_crosscheck.py [--out result.json] [--npu-model DIR]

Needs torch, the pi0-quant submodule (third_party) and an npu-model checkout (default
$NPU_MODEL or /scratch2/lorenhung/atlas-npu-rtl/src/npu-model). Only rtl_math.py is imported, by
path, so npu-model's own Python requirement does not apply.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import accum_torch as at  # noqa: E402
import chip_ops as co  # noqa: E402
import ref_numerics as rn  # noqa: E402

DEFAULT_NPU_MODEL = "/scratch2/lorenhung/atlas-npu-rtl/src/npu-model"
DEFAULT_EXPORT = "/scratch2/lorenhung/atlas-fp8/handoff/atlas_v1/export_smoke/tinyllama/atlas_v1/model.safetensors"
UNARY_NAMES = {"relu": "relu", "rcp": "rcp", "sqrt": "sqrt", "sin": "sin", "cos": "cos",
               "log": "log", "tanh": "tanh", "exp": "exp", "exp2": "exp2", "square": "square",
               "cube": "cube"}
SHIFTS = (-127, -126, -12, -8, -3, -1, 0, 1, 3, 8, 12, 126, 127, 128)   # scale shifts e - 127 of the E8M0 bytes 0..255 (ends: byte 0 and byte 255, which the RTL clamps to +127)


def load_rtl_math(npu_model: str):
    path = Path(npu_model) / "npu_model" / "configs" / "rtl_math.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("npu_rtl_math", path)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)          # the unary tables are read next to this file
    spec.loader.exec_module(mod)
    return mod


def _bits16(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF


def _isnan_bits(b: torch.Tensor) -> torch.Tensor:
    return ((b >> 7) & 0xFF == 0xFF) & ((b & 0x7F) != 0)


def _result(name, a: torch.Tensor, b: torch.Tensor, note: str = "", nan_equal=True) -> dict:
    """Compare two integer tensors of raw words."""
    a, b = a.to(torch.int64).flatten(), b.to(torch.int64).flatten()
    bad = a != b
    if nan_equal:
        bad &= ~(_isnan_bits(a.to(torch.int32)) & _isnan_bits(b.to(torch.int32)))
    idx = bad.nonzero().flatten()[:5].tolist()
    return {"name": name, "cases": int(a.numel()), "mismatches": int(bad.sum()), "note": note,
            "examples": [{"index": i, "ours": int(a[i]), "npu_model": int(b[i])} for i in idx]}


def rand_e4m3(g, shape) -> torch.Tensor:
    return torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)


def rand_bf16_bits(g, shape, normal_only=True) -> torch.Tensor:
    """Random BF16 words: normals (and +0) by default, since a BF16 psum is never subnormal here."""
    sign = torch.randint(0, 2, shape, generator=g, dtype=torch.int32)
    exp = torch.randint(100 if normal_only else 0, 150, shape, generator=g, dtype=torch.int32)
    mant = torch.randint(0, 128, shape, generator=g, dtype=torch.int32)
    out = (sign << 15) | (exp << 7) | mant
    zero = torch.randint(0, 8, shape, generator=g, dtype=torch.int32) == 0     # some exact zeros
    return torch.where(zero, torch.zeros_like(out), out)


def as_bf16(bits: torch.Tensor) -> torch.Tensor:
    return bits.to(torch.int32).to(torch.uint16).view(torch.bfloat16)


# ---------------------------------------------------------------------------------------------
# Matrix unit
# ---------------------------------------------------------------------------------------------

def check_ipt_tile(R, n_cases=300, lanes=32, seed=0) -> list:
    """One IPT compute: 32 products plus a BF16 psum, then a single RNE. Both psum kinds."""
    g = torch.Generator().manual_seed(seed)
    lut = at.e4m3_mul_to_prod_lut()
    headroom, int_width = at._ipt_consts(32, "numba")
    ours, theirs = [], []
    ours0, theirs0 = [], []
    for _ in range(n_cases):
        a = rand_e4m3(g, (32,))
        w = rand_e4m3(g, (lanes, 32))
        psum = rand_bf16_bits(g, (lanes,))
        ref = R.ipt_row(a, w, as_bf16(psum))
        theirs.append(_bits16(ref))
        xt = a.to(torch.int32)[None, None, :]
        wt = w.to(torch.int32)[None, :, :]
        got = at._ipt_tile(xt, wt, lut, psum.to(torch.int64)[None, None, :], 2, headroom, int_width,
                           "numba", at.OUT_BF16, 0)
        ours.append(got.flatten())
        ref0 = R.ipt_row(a, w, as_bf16(torch.zeros(lanes, dtype=torch.int32)))
        theirs0.append(_bits16(ref0))
        got0 = at._ipt_tile(xt, wt, lut, torch.zeros(1, 1, lanes, dtype=torch.int64), 0, headroom,
                            int_width, "numba", at.OUT_BF16, 0)
        ours0.append(got0.flatten())
    return [_result("ipt_tile_with_psum", torch.cat(ours), torch.cat(theirs),
                    f"{n_cases} tiles x {lanes} lanes, uniform E4M3 bytes incl. 0x7F/0xFF, BF16 psum"),
            _result("ipt_tile_no_psum", torch.cat(ours0), torch.cat(theirs0),
                    f"{n_cases} tiles x {lanes} lanes, psum 0")]


def check_ipt_chain(R, seed=1) -> list:
    """K-tile chaining: tile 0 from zero, later tiles take the previous BF16 word as psum."""
    out = []
    for K, M, N in ((100, 3, 6), (256, 3, 6), (4416, 2, 4)):
        g = torch.Generator().manual_seed(seed + K)
        x = rand_e4m3(g, (M, K))
        w = rand_e4m3(g, (N, K))
        ours = at.ipt_linear_bits(x, w)                        # [M, N] raw words
        theirs = torch.zeros(M, N, dtype=torch.int32)
        for m in range(M):
            acc = torch.zeros(N, dtype=torch.int32)
            for t in range(0, K, 32):
                a = torch.nn.functional.pad(x[m, t:t + 32], (0, max(0, t + 32 - K)))
                ws = torch.nn.functional.pad(w[:, t:t + 32], (0, max(0, t + 32 - K)))
                acc = _bits16(R.ipt_row(a, ws, as_bf16(acc)))
            theirs[m] = acc
        out.append(_result(f"ipt_chain_K{K}", ours, theirs, f"{M}x{N} outputs, {-(-K // 32)} K tiles"))
    return out


def check_sa_chain(R, seed=2) -> list:
    """SA: RNE after every add, sequential down K from +0."""
    out = []
    for K, M, N in ((32, 4, 8), (100, 3, 6), (4416, 2, 3)):
        g = torch.Generator().manual_seed(seed + K)
        x = rand_e4m3(g, (M, K))
        w = rand_e4m3(g, (N, K))
        ours = at.sa_linear_bits(x, w, compile=False)
        xs = x[:, None, :].expand(M, N, K).reshape(M * N, K)
        ws = w[None, :, :].expand(M, N, K).reshape(M * N, K)
        acc = as_bf16(torch.zeros(M * N, dtype=torch.int32))
        for k in range(K):
            acc = R.sa_fma(xs[:, k].contiguous(), ws[:, k].contiguous(), acc)
        out.append(_result(f"sa_chain_K{K}", ours.reshape(-1), _bits16(acc),
                           f"{M}x{N} outputs, K={K} sequential FMAs"))
    return out


def check_real_weights(R, export=None, seed=4) -> list:
    """Real E4M3 weight codes from an atlas_v1 export against both matrix units, K chained in full.

    Takes the first, a middle and the last 2-D uint8 matrix of a model.safetensors, 16 output rows
    each, and four random activation rows quantized with ref_numerics' pack (per-row po2 scale)."""
    from safetensors import safe_open
    export = export or DEFAULT_EXPORT
    if not Path(export).exists():
        return []
    g = torch.Generator().manual_seed(seed)
    out = []
    with safe_open(export, "pt") as f:
        names = [k for k in f.keys() if f.get_slice(k).get_dtype() == "U8" and len(f.get_slice(k).get_shape()) == 2]
        for name in (names[0], names[len(names) // 2], names[-1]):
            w = f.get_tensor(name)[:16].contiguous()
            K = w.shape[1]
            xf = torch.randn(4, K, generator=g)
            x = rn.e4m3_vpack(xf.to(torch.bfloat16), rn.po2_scale_exp(xf, dims=-1)).to(torch.uint8)
            ours = at.ipt_linear_bits(x, w)
            theirs = torch.zeros(4, 16, dtype=torch.int32)
            for m in range(4):
                acc = torch.zeros(16, dtype=torch.int32)
                for t in range(0, K, 32):
                    acc = _bits16(R.ipt_row(x[m, t:t + 32], w[:, t:t + 32], as_bf16(acc)))
                theirs[m] = acc
            out.append(_result(f"real_ipt:{name}", ours, theirs,
                               f"4x16 outputs, K={K} ({-(-K // 32)} tiles), real export codes"))
            ours = at.sa_linear_bits(x, w, compile=False)
            xs = x[:, None, :].expand(4, 16, K).reshape(64, K)
            ws = w[None, :, :].expand(4, 16, K).reshape(64, K)
            acc = as_bf16(torch.zeros(64, dtype=torch.int32))
            for k in range(K):
                acc = R.sa_fma(xs[:, k].contiguous(), ws[:, k].contiguous(), acc)
            out.append(_result(f"real_sa:{name}", ours.reshape(-1), _bits16(acc),
                               f"4x16 outputs, K={K} sequential FMAs, real export codes"))
    return out


# ---------------------------------------------------------------------------------------------
# FP8 conversions
# ---------------------------------------------------------------------------------------------

def check_fp8(R) -> list:
    out = []
    allbits = torch.arange(65536, dtype=torch.int32)
    values = as_bf16(allbits)
    ours_pack, theirs_pack, ours_pop, theirs_pop = [], [], [], []
    for s in SHIFTS:
        ours_pack.append(rn.e4m3_vpack(values, s).to(torch.int32))
        theirs_pack.append(R.pack_row(values, s + 127, mxu=False).to(torch.int32))
        ours_pop.append(at._bf16_scale_to_e4m3(allbits.clone(), max(-127, min(127, s))).to(torch.int32))   # byte -> shift clamp, as the RTL
        theirs_pop.append(R.pack_row(values, s + 127, mxu=True).to(torch.int32))
    out.append(_result("vpack_all_bf16", torch.cat(ours_pack), torch.cat(theirs_pack),
                       f"ref_numerics.e4m3_vpack vs pack_row(mxu=False): 65536 BF16 x {len(SHIFTS)} scales"))
    out.append(_result("mxu_pop_all_bf16", torch.cat(ours_pop), torch.cat(theirs_pop),
                       f"accum_torch._bf16_scale_to_e4m3 vs pack_row(mxu=True): 65536 BF16 x {len(SHIFTS)} scales"))
    codes = torch.arange(256, dtype=torch.uint8)
    ours_un, theirs_un = [], []
    for s in SHIFTS:
        ours_un.append(_bits16(rn.e4m3_unpack(codes, s).to(torch.bfloat16)))
        theirs_un.append(_bits16(R.unpack_row(codes, s + 127)))
    out.append(_result("vunpack_all_codes", torch.cat(ours_un), torch.cat(theirs_un),
                       f"ref_numerics.e4m3_unpack vs unpack_row: 256 codes x {len(SHIFTS)} scales"))
    return out


# ---------------------------------------------------------------------------------------------
# Vector unit
# ---------------------------------------------------------------------------------------------

def check_unary(R) -> list:
    out = []
    allv = as_bf16(torch.arange(65536, dtype=torch.int32))
    for ours_name, npu_name in UNARY_NAMES.items():
        ours = co.unary_table(ours_name)
        theirs = _bits16(R.unary(npu_name, allv))
        out.append(_result(f"unary_{ours_name}", ours, theirs,
                           "chip_ops table (pi0-quant lane box) vs RTL-generated table, all 65536 BF16"))
    return out


def check_binary(R, n=200000, seed=3) -> list:
    g = torch.Generator().manual_seed(seed)
    a = as_bf16(torch.randint(0, 65536, (n,), generator=g, dtype=torch.int32))
    b = as_bf16(torch.randint(0, 65536, (n,), generator=g, dtype=torch.int32))
    keep = ~(torch.isnan(a.float()) | torch.isnan(b.float()) | torch.isinf(a.float()) | torch.isinf(b.float()))
    a, b = a[keep], b[keep]
    vpu = co.ChipVPU()
    out = []
    ours = _bits16(vpu.add(a.float(), b.float()).to(torch.bfloat16))
    theirs = _bits16(R._truncated_bf16(a.float() + b.float()))
    out.append(_result("vadd_truncate", ours, theirs, f"{a.numel()} random finite BF16 pairs"))
    for name, fn in (("vmax", lambda x, y: vpu.pairwise_max(x, y)), ("vmin", lambda x, y: vpu.pairwise_min(x, y))):
        ours = _bits16(fn(a.float(), b.float()).to(torch.bfloat16))
        theirs = _bits16(R.minmax(a, b, maximum=(name == "vmax")))
        out.append(_result(f"{name}_pairwise", ours, theirs, f"{a.numel()} random finite BF16 pairs"))
    return out


def check_rtl_fixture(npu_model: str) -> list:
    """Replay npu-model's Verilator-captured arithmetic fixture (tests/rtl/arithmetic.json: E4M3FMA,
    AnchorAccumulationTree and the converters, simulated from the Chisel RTL) through OUR models.
    This compares against real RTL output, not against another Python model."""
    path = Path(npu_model) / "tests" / "rtl" / "arithmetic.json"
    if not path.exists():
        return []
    cases = json.loads(path.read_text())
    lut = at.e4m3_mul_to_prod_lut()
    headroom, int_width = at._ipt_consts(32, "numba")
    fma_o, fma_r, ipt_o, ipt_r, q_o, q_r, dq_o, dq_r = ([] for _ in range(8))
    for c in cases:
        a = torch.tensor(c["a"], dtype=torch.int32)
        w = torch.tensor(c["w"], dtype=torch.int32)
        partial = torch.tensor([c["partial"]], dtype=torch.int64)
        ipt_o.append(int(at._ipt_tile(a[None, None, :], w[None, None, :], lut, partial[None, None, :], 2,
                                      headroom, int_width, "numba", at.OUT_BF16, 0).flatten()[0]))
        ipt_r.append(c["ipt"])
        acc = partial.to(torch.int32)[None, None, :]                      # [B=1, M=1, N=1]
        fma_o.append(int(at._sa_step(acc, a[:1][None, :], w[:1][None, :], lut).flatten()[0]))
        fma_r.append(c["fma"])
        q_o.append(int(at._bf16_scale_to_e4m3(partial.to(torch.int32), c["scale"] - 127).flatten()[0]))
        q_r.append(c["quant"])
        dq_o.append(int(_bits16(rn.e4m3_unpack(a[:1].to(torch.uint8), 0).to(torch.bfloat16)).flatten()[0]))
        dq_r.append(c["dequant"])
    t = lambda v: torch.tensor(v, dtype=torch.int64)  # noqa: E731
    n = f"{len(cases)} Verilator cases"
    return [_result("rtl_fixture_ipt", t(ipt_o), t(ipt_r), f"AnchorAccumulationTree, {n}", nan_equal=False),
            _result("rtl_fixture_sa_fma", t(fma_o), t(fma_r), f"E4M3FMA, {n}", nan_equal=False),
            _result("rtl_fixture_mxu_pop", t(q_o), t(q_r), f"BF16ScaleToE4M3 (pop), {n}, scale byte as captured",
                    nan_equal=False),
            _result("rtl_fixture_unpack", t(dq_o), t(dq_r), f"FP8 dequant, {n}", nan_equal=False)]


def run(npu_model: str) -> dict:
    R = load_rtl_math(npu_model)
    sections = {"rtl_fixture_replay": check_rtl_fixture(npu_model),
                "matrix_unit": check_ipt_tile(R) + check_ipt_chain(R) + check_sa_chain(R),
                "real_weights": check_real_weights(R),
                "fp8_conversions": check_fp8(R),
                "vector_unit": check_unary(R) + check_binary(R)}
    flat = [r for v in sections.values() for r in v]
    return {"npu_model": npu_model, "sections": sections,
            "total_cases": sum(r["cases"] for r in flat),
            "total_mismatches": sum(r["mismatches"] for r in flat)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--npu-model", default=os.environ.get("NPU_MODEL", DEFAULT_NPU_MODEL))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    res = run(args.npu_model)
    for sec, rows in res["sections"].items():
        print(f"== {sec}")
        for r in rows:
            flag = "OK  " if r["mismatches"] == 0 else "DIFF"
            print(f"  {flag} {r['name']:<22} {r['mismatches']:>8} / {r['cases']:<9} {r['note']}")
    print(f"total mismatches {res['total_mismatches']} / {res['total_cases']}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))
    return 0 if res["total_mismatches"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
