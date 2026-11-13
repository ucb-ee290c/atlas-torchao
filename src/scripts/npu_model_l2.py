"""L2: cycles of a quantized linear on npu-model (`--numerics rtl`), with TinyLlama layer shapes and real
export weight codes, compared with the analytic hw_cost estimate and with accum_torch.ipt_linear_bits.

npu-model needs Python >= 3.13 (PEP 696 type-parameter defaults in npu_model/isa.py:91; it fails to import under
3.12 with a SyntaxError) and torch. The parent process (the atlas-lerobot 3.12 env) prepares data, spawns the
simulator under a 3.14 interpreter (--py314), and does the analytic and bit-for-bit comparisons. The same file is
the child (`--child`). The simulator checkout is only read; no file is written inside it.

    /scratch2/lorenhung/envs/atlas-lerobot/bin/python -I src/scripts/npu_model_l2.py --py314 <venv314>/bin/python

Kernel (generated here, MXU1 = inner-product tree, weight [N,K] tiles pushed WITHOUT vtrpose.xlu): per output
strip n, for k in 0..Kt-1: vload A tile(s), vload W tile (burst of `burst` tiles per DMA), vmatpush.weight,
vmatmul / vmatmul.acc into acc0 (and acc1 for a second 32-token block that reuses the pushed weight), then pop BF16
and DMA the tile out. Every dependency is covered by an explicit `delay` (the shipped kernels do the same: the
model has no scoreboard), so the compute part is serialized by construction; the DMA of the next burst overlaps it.
A tiles are DMA'd once into VMEM before the strip loop (setup cost, reported separately from the per-strip slope).
Only two accumulators exist per MXU, so at most 2 token blocks (64 tokens) reuse a pushed weight tile in the
simulator; larger block counts are analytic only.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

NPU_MODEL = os.environ.get("NPU_MODEL", "/scratch2/lorenhung/atlas-npu-rtl/src/npu-model")
EXPORT = ("/scratch2/lorenhung/atlas-fp8/handoff/atlas_v1/export_smoke/tinyllama/atlas_v1/model.safetensors")
OUT = Path("/scratch2/lorenhung/atlas-fp8/results/hw-verify/l2")
TILE = 32
VA = 0x800                          # VMEM word address (4 B words) of the A region, as the shipped kernels use 0x800


def li(r, v):
    return f"li x{r}, {v}"


# ---------------------------------------------------------------------------------------------------------
# kernel generator (text assembly, assembled by npu_model.util.converter.load_asm in the child)
# ---------------------------------------------------------------------------------------------------------
def gen_asm(mb: int, Nt: int, Kt: int, burst: int, dram_w: int, dram_a: int, dram_c: int) -> str:
    assert mb in (1, 2) and Kt % burst == 0
    a_words = mb * Kt * 256
    vw = [VA + a_words, VA + a_words + burst * 256]       # two W burst buffers
    vc = vw[1] + burst * 256                              # C tile staging: mb x (two 1 KiB halves)
    L = []
    L += [li(5, burst * 1024), li(6, 2048), li(9, 4096), li(30, 1), li(2, Nt), li(1, 0)]
    L += ["dma.config x30, ch0", "dma.wait ch0", "dma.config x30, ch1", "dma.wait ch1",
          "dma.config x30, ch2", "dma.wait ch2"]
    # A preload: mb*Kt tiles contiguous in DRAM, 4 KiB per transfer
    ntiles = mb * Kt
    off = 0
    while off < ntiles:
        n = min(4, ntiles - off)
        L += [li(7, VA + off * 256), li(8, dram_a + off * 1024), li(10, n * 1024),
              "dma.load x7, x8, x10, ch2", "dma.wait ch2"]
        off += n
    L += [li(3, dram_w), li(11, dram_c), li(12, dram_c + Nt * 2048)]
    L += ["loop_n:"]
    G = Kt // burst
    L += [li(7, vw[0]), "dma.load x7, x3, x5, ch0", "add x3, x3, x5"]
    for g in range(G):
        cur, nxt = g % 2, (g + 1) % 2
        if g + 1 < G:
            L += [li(7, vw[nxt]), f"dma.load x7, x3, x5, ch{nxt}", "add x3, x3, x5"]
        L += [f"dma.wait ch{cur}"]
        for j in range(burst):
            k = g * burst + j
            L += [li(31, VA + k * 256), "vload m0, 0(x31)", "delay 34"]
            if mb == 2:
                L += [li(31, VA + (Kt + k) * 256), "vload m1, 0(x31)", "delay 34"]
            L += [li(31, vw[cur] + j * 256), "vload m2, 0(x31)", "delay 34",
                  "vmatpush.weight.mxu1 w0, m2", "delay 32"]
            op = "vmatmul.mxu1" if k == 0 else "vmatmul.acc.mxu1"
            L += [f"{op} acc0, m0, w0", "delay 35"]
            if mb == 2:
                L += [f"{op} acc1, m1, w0", "delay 35"]
    pops = [("acc0", 4, 11)] + ([("acc1", 6, 12)] if mb == 2 else [])
    for acc, m, ptr in pops:
        L += [f"vmatpop.bf16.acc.mxu1 m{m}, {acc}", "delay 32"]
        L += [li(31, vc), f"vstore m{m}, 0(x31)", "delay 34", li(31, vc + 256), f"vstore m{m + 1}, 0(x31)", "delay 34",
              li(7, vc), f"dma.store x{ptr}, x7, x6, ch0", "dma.wait ch0", f"add x{ptr}, x{ptr}, x6"]
    # the strip body can exceed a conditional branch's +-2048 word range (Kt=176), so loop with bge + jal
    L += ["addi x1, x1, 1", "bge x1, x2, done_n", "addi x0, x0, 0", "jal x0, loop_n", "addi x0, x0, 0",
          "done_n:", "addi x0, x0, 0", "addi x0, x0, 0"]
    return "\n".join(L) + "\n"


def tile_matrix(mat: np.ndarray) -> np.ndarray:
    R, C = mat.shape
    return np.ascontiguousarray(mat.reshape(R // TILE, TILE, C // TILE, TILE).transpose(0, 2, 1, 3)).reshape(-1)


# ---------------------------------------------------------------------------------------------------------
# child: runs under python >= 3.13 with torch + npu_model
# ---------------------------------------------------------------------------------------------------------
def child(spec_path: str):
    from dataclasses import replace
    spec = json.loads(Path(spec_path).read_text())
    sys.path.insert(0, NPU_MODEL)
    os.chdir(NPU_MODEL)                      # configs.programs loads asm through a relative path
    import torch
    from npu_model.configs.hardware import DefaultHardwareConfig
    from npu_model.logging import LoggerConfig
    from npu_model.simulation import Simulation
    from npu_model.software.program import InstantiableProgram
    from npu_model.util.converter import load_asm

    d = Path(spec["dir"])
    res = []
    for c in spec["cases"]:
        asm = d / f"{c['name']}.S"
        asm.write_text(gen_asm(c["mb"], c["Nt"], c["Kt"], c["burst"], c["dram_w"], c["dram_a"], c["dram_c"]))
        w = torch.from_numpy(np.load(d / f"{c['name']}.w.npy"))
        a = torch.from_numpy(np.load(d / f"{c['name']}.a.npy"))
        prog = InstantiableProgram(load_asm(asm))
        prog.memory_regions = [(c["dram_w"], w), (c["dram_a"], a)]
        hw = DefaultHardwareConfig()
        hw.arch_state_config = replace(hw.arch_state_config, numerics="rtl")
        sim = Simulation(hardware_config=hw, logger_config=LoggerConfig(filename=str(d / "trace_unused.json")),
                         program=prog, verbose=False)
        t0 = time.time()
        sim.run(max_cycles=c["max_cycles"])
        st = sim.get_stats()
        finished = bool(sim.core.is_finished())
        out_bytes = c["mb"] * c["Nt"] * 2048
        out = sim.core.arch_state.read_dram(prog.dram_base + c["dram_c"], out_bytes).clone().numpy()
        np.save(d / f"{c['name']}.c.npy", out)
        res.append(dict(name=c["name"], cycles=st.cycles, finished=finished, wall_s=time.time() - t0,
                        instructions=st.total_instructions,
                        busy={k: v.busy_cycles for k, v in st.exu_stats.items()}))
        print(json.dumps(res[-1]), flush=True)
        with open(Path(spec_path).with_suffix(".jsonl"), "a") as fh:      # incremental: a later crash loses nothing
            fh.write(json.dumps(res[-1]) + "\n")
        sim.close()
        try:
            os.remove(d / "trace_unused.json")
        except OSError:
            pass


# ---------------------------------------------------------------------------------------------------------
# parent
# ---------------------------------------------------------------------------------------------------------
def load_codes(export: str, name: str, n_rows: int, k: int) -> np.ndarray:
    from safetensors import safe_open
    with safe_open(export, "numpy") as f:
        return np.ascontiguousarray(f.get_tensor(name)[:n_rows, :k])


def make_case(c: dict, d: Path, seed: int):
    import torch
    import reference.ref_numerics as rn
    Nt, Kt, mb = c["Nt"], c["Kt"], c["mb"]
    N, K, M = Nt * TILE, Kt * TILE, mb * TILE
    w = load_codes(EXPORT, c["tensor"], N, K)                                   # [N, K] uint8 codes, as exported
    g = torch.Generator().manual_seed(seed)
    xf = torch.randn(M, K, generator=g)
    x = rn.e4m3_vpack(xf.to(torch.bfloat16), rn.po2_scale_exp(xf, dims=-1)).to(torch.uint8).numpy()
    wt = np.concatenate([tile_matrix(w[n * TILE:(n + 1) * TILE]) for n in range(Nt)])       # tile (n,k) contiguous
    at = np.concatenate([tile_matrix(x[m * TILE:(m + 1) * TILE]) for m in range(mb)])        # tile (m,k) contiguous
    np.save(d / f"{c['name']}.w.npy", wt)
    np.save(d / f"{c['name']}.a.npy", at)
    c["dram_w"] = 0
    c["dram_a"] = wt.size
    c["dram_c"] = wt.size + at.size
    return w, x


def decode_c(raw: np.ndarray, mb: int, Nt: int, M: int, N: int) -> np.ndarray:
    """raw C: per (m,n) tile two 32x16 BF16 halves (cols 0-15 then 16-31), tiles in (m, n) order -> [M, N] uint16."""
    u = raw.view(np.uint16).reshape(mb, Nt, 2, TILE, 16)
    return np.ascontiguousarray(u.transpose(0, 3, 1, 2, 4)).reshape(M, N)


def analytic(c: dict):
    import chip.hw_params as hp
    import chip.hw_cost as hc
    burst = c["burst"]
    dma_tile = hp.dma_load_cycles(burst * 1024) / burst            # DERIVED from the measured curve
    compute_tile = 34 * (c["mb"] + 1) + 32 + 35 * c["mb"]            # my kernel's serialized delays per k step
    return dict(dma_cycles_per_tile=dma_tile, kernel_compute_cycles_per_tile=compute_tile,
                overlapped_cycles_per_tile=max(dma_tile, compute_tile),
                hw_cost_mxu_cycles_per_tile_block=hp.MXU_ISSUE_GAP * c["mb"])


def parent(args):
    import torch
    import mxu.accum_torch as at
    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / "work"
    d.mkdir(exist_ok=True)
    layers = {   # name: (tensor, N tiles of the real layer, K tiles)
        "q_proj_2048x2048": ("model.layers.0.self_attn.q_proj.weight_codes", 64, 64),
        "k_proj_256x2048": ("model.layers.0.self_attn.k_proj.weight_codes", 8, 64),
        "gate_proj_5632x2048": ("model.layers.0.mlp.gate_proj.weight_codes", 176, 64),
        "down_proj_2048x5632": ("model.layers.0.mlp.down_proj.weight_codes", 64, 176),
    }
    cases, meta = [], {}
    for lname, (tensor, Nfull, Kt) in layers.items():
        for mb in (1, 2):
            for nrun in (1, 2):
                c = dict(name=f"{lname}_mb{mb}_n{nrun}", layer=lname, tensor=tensor, mb=mb, Nt=nrun, Kt=Kt, burst=4,
                         Nfull=Nfull, max_cycles=args.max_cycles)
                meta[c["name"]] = make_case(c, d, seed=11 + Kt + mb)
                cases.append(c)
        if args.full:                         # the whole layer (all N tiles); mb=2 only where it is cheap
            for mb in ((1, 2) if lname in ("k_proj_256x2048", "q_proj_2048x2048") else (1,)):
                c = dict(name=f"{lname}_mb{mb}_full", layer=lname, tensor=tensor, mb=mb, Nt=Nfull, Kt=Kt, burst=4,
                         Nfull=Nfull, max_cycles=args.max_cycles)
                meta[c["name"]] = make_case(c, d, seed=11 + Kt + mb)
                cases.append(c)
    if args.extra_burst1:                     # 1 KiB transfers instead of 4 KiB bursts, one shape
        c = dict(name="q_proj_2048x2048_mb1_n1_burst1", layer="q_proj_2048x2048", tensor=layers["q_proj_2048x2048"][0],
                 mb=1, Nt=1, Kt=64, burst=1, Nfull=64, max_cycles=args.max_cycles)
        meta[c["name"]] = make_case(c, d, seed=11 + 64 + 1)
        cases.append(c)
    if args.only:
        cases = [c for c in cases if any(s in c["name"] for s in args.only)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    t0 = time.time()
    procs = []
    for lname in dict.fromkeys(c["layer"] for c in cases):       # one simulator process per layer, in parallel
        spec = d / f"spec_{lname}.json"
        spec.with_suffix(".jsonl").unlink(missing_ok=True)
        spec.write_text(json.dumps(dict(dir=str(d), cases=[c for c in cases if c["layer"] == lname])))
        procs.append(subprocess.Popen([args.py314, "-I", str(Path(__file__).resolve()), "--child", str(spec)], env=env))
    for pr in procs:
        pr.wait()
    wall = time.time() - t0
    child_res = {}
    for jl in d.glob("spec_*.jsonl"):
        for line in jl.read_text().splitlines():
            r = json.loads(line)
            child_res[r["name"]] = r
    missing = [c["name"] for c in cases if c["name"] not in child_res]
    if missing:
        raise SystemExit(f"simulator children did not finish: {missing}")
    results = []
    for c in cases:
        r = child_res[c["name"]]
        w, x = meta[c["name"]]
        M, N = c["mb"] * TILE, c["Nt"] * TILE
        raw = np.load(d / f"{c['name']}.c.npy")
        sim_words = decode_c(raw, c["mb"], c["Nt"], M, N)
        ref = at.ipt_linear_bits(torch.from_numpy(x), torch.from_numpy(w)).numpy().astype(np.uint16)
        mism = int((sim_words != ref).sum())
        a = analytic(c)
        results.append(dict(c, **{k: v for k, v in r.items() if k != "name"}, analytic=a,
                            bit_compare=dict(elements=int(ref.size), mismatches=mism,
                                             reference="accum_torch.ipt_linear_bits (numba IPT), same A/W code bytes")))
    # per-strip slope (nrun 2 - nrun 1) and setup, per layer and mb
    summary = {}
    for lname, (tensor, Nfull, Kt) in layers.items():
        for mb in (1, 2):
            a1 = next((r for r in results if r["name"] == f"{lname}_mb{mb}_n1"), None)
            a2 = next((r for r in results if r["name"] == f"{lname}_mb{mb}_n2"), None)
            if not (a1 and a2) or not (a1["finished"] and a2["finished"]):
                continue
            strip = a2["cycles"] - a1["cycles"]
            setup = a1["cycles"] - strip
            an = a1["analytic"]
            full = next((r for r in results if r["name"] == f"{lname}_mb{mb}_full"), None)
            summary[f"{lname}_mb{mb}"] = dict(
                Kt=Kt, N_tiles=Nfull, tokens=mb * TILE,
                sim_cycles_per_strip=strip, sim_cycles_per_weight_tile=strip / Kt,
                setup_cycles_A_preload=setup,
                analytic_dma_cycles_per_tile=an["dma_cycles_per_tile"],
                analytic_overlapped_cycles_per_tile=an["overlapped_cycles_per_tile"],
                sim_over_analytic_dma=strip / Kt / an["dma_cycles_per_tile"],
                layer_cycles_extrapolated=setup + Nfull * strip,
                layer_cycles_extrapolated_label="EXTRAPOLATED: setup + N_tiles x per-strip slope (A preload + strips)",
                layer_seconds_extrapolated=(setup + Nfull * strip) / 500e6,
                layer_cycles_full_sim=(full["cycles"] if full and full["finished"] else None),
                layer_full_sim_cycles_per_weight_tile=((full["cycles"] - setup) / (Nfull * Kt)
                                                       if full and full["finished"] else None),
                layer_full_sim_over_analytic_dma_total=(full["cycles"] / (Nfull * Kt * an["dma_cycles_per_tile"])
                                                        if full and full["finished"] else None))
    out = dict(npu_model=NPU_MODEL, numerics="rtl", mxu="mxu1 (IPT)", child_wall_s=wall, cases=results, summary=summary,
               total_mismatches=sum(r["bit_compare"]["mismatches"] for r in results),
               total_compared=sum(r["bit_compare"]["elements"] for r in results))
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))
    print("bit compare:", out["total_mismatches"], "mismatches of", out["total_compared"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", default=None)
    ap.add_argument("--py314", default=None, help="python >= 3.13 with torch+numpy (npu-model needs it)")
    ap.add_argument("--max-cycles", type=int, default=20_000_000)
    ap.add_argument("--extra-burst1", action="store_true")
    ap.add_argument("--full", action="store_true", help="also simulate each whole layer (about 20k simulated cycles/s)")
    ap.add_argument("--only", nargs="*", default=None, help="substrings: run only matching cases")
    ap.add_argument("--out", default=str(OUT.parent / "l2_npu_model.json"))
    args = ap.parse_args()
    if args.child:
        child(args.child)
    else:
        if not args.py314:
            ap.error("--py314 is required")
        parent(args)


if __name__ == "__main__":
    main()
