"""Throughput of the torch MXU accumulation ports (src/accum_torch.py), in MAC/s.

    python scripts/bench_accum.py                       # CPU, eager, default shapes
    python scripts/bench_accum.py --device cuda --compile --check
    PI0_QUANT_DIR=third_party/pi0-quant python scripts/bench_accum.py --ref

--check   re-run each case and compare bits run-to-run; on a non-CPU device also compare
          against the same call on CPU (the CUDA re-verification the docs ask for).
--ref     also time the reference CPU models (systolic_c, ipt_c, ipt_numba) on the same bytes.
--compile torch.compile the SA step (Triton on CUDA, C++ on CPU).

Prints one JSON object per case and a summary line. MAC = M * N * K per call.

Measured on the CPU dev host (2026-10-09, torch 2.11 CPU, shared 160-core host at load ~300,
so treat as rough; --threads matters: the default 80 threads collapse under contention):
  model / shape M x K x N          threads  MAC/s
  torch sa, eager   16x2048x256      1      4.0e6
  torch sa, compile 16x2048x256      1      1.5e7
  torch sa, compile 64x2048x512      1      3.3e7
  torch sa, eager   64x2048x512      8      1.9e6
  torch ipt (numba) 16x2048x256      1      1.5e7
  torch ipt (numba) 64x2048x512      8      7.9e6
  reference systolic_c 16x2048x256   1      4.1e7   (torch port bit-exact on this call)
  reference ipt_c      16x2048x256   1      3.7e7   (torch port bit-exact on this call)
The CPU path is for testing; throughput is the GPU's job.

Expected scaling (NEW estimate, to be replaced by a measurement on the GPU host):
  SA is a K-step loop of ~60 element-wise int32 ops over an [M, N] state. Eager mode is memory
  bound: ~60 passes of 4 bytes over M*N per step, so on an L40S (~860 GB/s) roughly
  860e9 / (60 * 2 * 4) ~ 1.8e9 MAC/s once M*N is large (>= 1e6) and per-launch overhead
  (~60 launches/step, ~5 us each) is hidden. --compile fuses a step into one kernel
  (read state + gather, write state: ~12 bytes/element), ~1e10-5e10 MAC/s at large M*N, then
  bounded by launch rate (one launch per K step) for small M*N.
  IPT does ~15 int64 element-wise ops over [M, N, 32] per tile, no K-serial dependency inside
  a tile, so it is ~3-5x cheaper per MAC than eager SA and scales with memory bandwidth.
  For TinyLlama decode-like shapes (M = tokens per call) use large M to amortise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import accum_torch as at  # noqa: E402

DEFAULT_SHAPES = ["16x2048x256", "64x2048x2048", "8x5632x512"]   # M x K x N


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def _time(fn, dev, reps):
    fn()                      # warm-up (and compile)
    _sync(dev)
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    _sync(dev)
    return (time.perf_counter() - t0) / reps, out


def _ref_models():
    ref = os.environ.get("PI0_QUANT_DIR", str(ROOT / "third_party" / "pi0-quant"))
    if ref not in sys.path:
        sys.path.append(ref)
    import importlib
    out = {}
    try:
        sa = importlib.import_module("func_models_sa.systolic_c.systolic_array_rtl_linear")
        out["systolic_c"] = ("sa", sa.SARTLLinearFunction(rows=32, cols=32))
    except Exception as e:  # noqa: BLE001
        print(f"# systolic_c unavailable: {e!r}", file=sys.stderr)
    try:
        ic = importlib.import_module("funct_models_ipt.ipt_c.ipt_rtl_linear_c")
        out["ipt_c"] = ("ipt_c", ic.CIPTLinearRTLFunction(vec_len=32, num_lanes=32))
    except Exception as e:  # noqa: BLE001
        print(f"# ipt_c unavailable: {e!r}", file=sys.stderr)
    try:
        inb = importlib.import_module("funct_models_ipt.ipt_numba.ipt_rtl_linear")
        out["ipt_numba"] = ("ipt", inb.IPTLinearRTLFunction(vec_len=32, num_lanes=32))
    except Exception as e:  # noqa: BLE001
        print(f"# ipt_numba unavailable: {e!r}", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shapes", nargs="+", default=DEFAULT_SHAPES, help="MxKxN")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ref", action="store_true")
    ap.add_argument("--models", nargs="+", default=["sa", "ipt", "ipt_c"])
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    report = at.numerics_report()
    print(json.dumps({"device": str(dev), "threads": torch.get_num_threads(), **report}))
    refs = _ref_models() if args.ref else {}
    bad = 0
    for shp in args.shapes:
        M, K, N = (int(v) for v in shp.split("x"))
        g = torch.Generator().manual_seed(M * 31 + K * 7 + N)
        # finite normal E4M3 values and zeros, as the vpack encoder produces them
        x = torch.randint(0, 256, (M, K), generator=g, dtype=torch.int32)
        w = torch.randint(0, 256, (N, K), generator=g, dtype=torch.int32)
        fix = lambda t: torch.where(((t >> 3) & 0xF) == 0, 0, torch.where((t & 0x7F) == 0x7F, t - 1, t)).to(torch.uint8)
        xb, wb = fix(x).to(dev), fix(w).to(dev)
        macs = M * N * K
        for name in args.models:
            if name == "sa":
                fn = lambda: at.sa_linear_bits(xb, wb, compile=args.compile)
            elif name == "ipt":
                fn = lambda: at.ipt_linear_bits(xb, wb)
            elif name == "ipt_c":
                fn = lambda: at.ipt_linear_bits(xb, wb, variant="c")
            else:
                raise SystemExit(f"unknown model {name}")
            dt, out = _time(fn, dev, args.reps)
            rec = {"model": f"torch_{name}", "M": M, "K": K, "N": N, "device": str(dev),
                   "compile": bool(args.compile and name == "sa"), "s_per_call": round(dt, 4),
                   "mac_per_s": macs / dt}
            if args.check:
                same = torch.equal(fn(), out)
                rec["repeat_identical"] = same
                bad += not same
                if dev.type != "cpu":
                    if name == "sa":
                        cpu = at.sa_linear_bits(xb.cpu(), wb.cpu())
                    else:
                        cpu = at.ipt_linear_bits(xb.cpu(), wb.cpu(), variant="c" if name == "ipt_c" else "numba")
                    rec["equals_cpu"] = torch.equal(out.cpu(), cpu)
                    bad += not rec["equals_cpu"]
            print(json.dumps(rec), flush=True)
        for rname, (kind, model) in refs.items():
            xflt = _bytes_to_float(fix(x))
            wflt = _bytes_to_float(fix(w))
            dt, y = _time(lambda: model(xflt, wflt, None), torch.device("cpu"), args.reps)
            rec = {"model": rname, "M": M, "K": K, "N": N, "device": "cpu", "s_per_call": round(dt, 4),
                   "mac_per_s": macs / dt}
            if args.check:
                if kind == "sa":
                    mine = at.sa_linear(xflt, wflt)
                else:
                    mine = at.ipt_linear(xflt, wflt, variant="c" if kind == "ipt_c" else "numba")
                rec["torch_port_bitexact"] = torch.equal(mine.view(torch.int32), y.view(torch.int32))
                bad += not rec["torch_port_bitexact"]
            print(json.dumps(rec), flush=True)
    print(json.dumps({"summary": "ok" if bad == 0 else f"{bad} check(s) FAILED"}))
    return 1 if bad else 0


def _bytes_to_float(b):
    b = b.to(torch.int32)
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
    v = (1 + m.double() / 8) * torch.exp2((e - 7).double())
    v = torch.where(e == 0, torch.zeros_like(v), v)
    return torch.where(s == 1, -v, v).float()


if __name__ == "__main__":
    raise SystemExit(main())
