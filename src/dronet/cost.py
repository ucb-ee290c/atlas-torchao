"""First-principles cost model of one DroNet frame (batch 1, 1x200x200) on the Atlas NPU.

EVERYTHING HERE THAT IS A TIME IS AN ESTIMATE, NOT A MEASUREMENT. Nothing was run on hardware or in
the cycle model. What is exact: the MAC / byte / element counts (`layer_table`, `nonconv_table`;
cross-checked against torch's FlopCounterMode in tests/test_dronet_cost.py). What is derived: cycles
from sourced Atlas constants (`chip.hw_params`, which cites the RTL; the sources are re-listed in
`SOURCES`). What is assumed: the scheduling / layout parameters of `SCENARIOS`, each named and
labelled ASSUMED.

Method (same idea as npu_model/npu_performance/speed_of_light.py): every resource that limits
throughput is charged for the work it must do, per layer:

  MXU     one 32x32x32 FP8 tile matmul holds the MXU for 32 cycles (mxu.py:25 PORT_CYCLES), so
          1024 MAC/cycle per MXU, two MXUs; padding to 32 in M, K, N is charged exactly.
  POP     one accumulator pop per output tile, 32 cycles per MXU (spec 04 'Pop, then pop').
  VLOAD   one 1 KiB tensor register per 35 cycles on ONE shared path (lsu.py:26, spec 04).
  VSTORE  likewise, independent path.
  VPU     one 1024-element BF16 tile op per ~65 cycles, two issue slots, binary ops and row
          reductions take both (vpu.py:40-56).
  DMA     DRAM<->VMEM at the measured simulation rate (hw_params: ~1.34 B/cycle load).

`high` takes the max over resources per layer (perfect overlap, the speed-of-light bound of the
chosen schedule); `low` takes the sum (no overlap). `nominal` overlaps like `high` but with the
less favourable assumptions. The spread between the scenarios is the honest uncertainty; the
absolute scale is further multiplied by the clock (a separate axis, `CLOCKS_HZ`).
"""
from __future__ import annotations

from dataclasses import dataclass

import chip.hw_params as hp

TILE = hp.TILE
BN_FIELDS = 2          # scale, shift per channel
# source file:line for every hardware number used (RTL = ucb-bar/atlas-npu @ eee27e2; NM = npu_model @ b97b2f5)
SOURCES = {
    "fp8_max_448": "atlas-npu src/main/scala/atlas/vector/laneBoxes/FP8Pack.scala:39-40 (E4M3_MAX_POS = 0x7e; hw_params.py cites 40-41, off by one at eee27e2)",
    "vmem_1p5MiB_6_banks": "atlas-npu src/main/scala/atlas/scalar/ScalarCore.scala:50-51 (VMEM_SIZE 0x180000, 6 banks)",
    "vmem_1MiB_spec": "npu_model/npu_spec/02_system_parameters/README.md (VMEM_SIZE 1 MiB; stale vs RTL)",
    "vmem_banks_3_5_768KiB": "atlas-backend-kit README.md:63-64,78 (window at 0x200C0000, 'banks 3-5'; 768 KiB cap at :123)",
    "mxu_array_32x32": "atlas-npu src/main/scala/atlas/common/MxuParams.scala:19 (arrayRows 32); spec 02 MXU_COUNT 2",
    "mxu_port_cycles_32": "npu_model/npu_model/hardware/mxu.py:25 (PORT_CYCLES = 32); npu_performance/roofline.py:156-168",
    "mxu_latency_95_35": "npu_model/npu_model/hardware/mxu.py:18-19; docs/rtl-timing.md:51 (spec 02 says 96/35)",
    "mxu_acc_gap": "npu_model/npu_spec/04_functional_units/README.md 'MXU buffer rules' (matmul->acc: SA 64, IPT 32)",
    "vpu_16_lanes": "atlas-npu src/main/scala/atlas/common/VpuParams.scala:10 (numLanes = 16); spec 02 VPU_LANES_BF16",
    "vpu_latency_66_2slots": "npu_model/npu_model/hardware/vpu.py:40-56",
    "vload_vstore_35": "npu_model/npu_model/hardware/lsu.py:23-28 ('vload': 35, 'vstore': 35); spec 05:115 says 34",
    "dma_rate": "npu_model/calibration/dma/measurements/ee290sim_vcs.json (perf_dma_load_4k 3056 cyc); docs/rtl-timing.md:178-190 "
                "(about 23 cycles per 32 B beat) -- measured in VCS sim with DRAMSim2 DDR3, NOT silicon",
    "clock_500MHz": "npu_model/docs/rtl-timing.md:165-166 ('500 MHz core'); calibration json frequencyGHz 0.5 (sim/design clock)",
    "clock_100_30MHz": "atlas-backend-kit README.md:74 (run-chip baud 384615 'at 100 MHz', --baud 115200 'at 30 MHz')",
    "perf_mm_dual_93pct": "atlas-npu baremetal/assembly/perf_mm_dual_128x128x128.S:12 (measured 1099 cyc, 93% util on both MXUs)",
    "perf_vpu_binary_83pct": "atlas-npu baremetal/assembly/perf_vpu_binary.S:10-12 (460 cyc vs 384 theoretical, 83.4%)",
    "two_acc_weight_reuse": "src/scripts/npu_model_l2.py:19-20 (only 2 accumulators per MXU -> a pushed weight tile serves 2 token blocks)",
}

CLOCKS_HZ = {"500 MHz (design / RTL timing model)": 500e6, "100 MHz (backend-kit chip bench)": 100e6,
             "30 MHz (backend-kit fallback)": 30e6}


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


def pad32(n: int) -> int:
    return cdiv(n, TILE) * TILE


def tiles(rows: int, cols: int) -> int:
    """32x32 tiles covering a rows x cols tensor (one row per pixel, one column per channel)."""
    return cdiv(rows, TILE) * cdiv(cols, TILE)


def _out(n: int, k: int, s: int, p: int) -> int:
    return (n + 2 * p - k) // s + 1


# ----------------------------------------------------------------------------------------------
# Part 1: exact counts
# ----------------------------------------------------------------------------------------------
def layer_table(h: int = 200, w: int = 200) -> list[dict]:
    """Per FP8-contraction layer (10 convs + the 2 heads), batch 1. All counts are exact.

    M = output pixels, K = Cin*kh*kw, N = Cout (conv as im2col matmul; K is NOT padded in `macs`).
    """
    rows = []

    def conv(name, cin, cout, k, s, p, hin, win, stage):
        ho, wo = _out(hin, k, s, p), _out(win, k, s, p)
        M, K, N = ho * wo, cin * k * k, cout
        Mb, Kt, Nt = cdiv(M, TILE), cdiv(K, TILE), cdiv(N, TILE)
        rows.append(dict(
            name=name, kind="conv", stage=stage, k=k, stride=s, cin=cin, cout=cout,
            in_hw=(hin, win), out_hw=(ho, wo), M=M, K=K, N=N, macs=M * K * N,
            w_params=K * N, bias_params=N,
            w_fp8_bytes=K * N, w_bf16_bytes=2 * K * N, w_fp8_bytes_padded=pad32(K) * pad32(N),
            in_elems=cin * hin * win, out_elems=cout * ho * wo,
            in_fp8=cin * hin * win, in_bf16=2 * cin * hin * win,
            out_fp8=cout * ho * wo, out_bf16=2 * cout * ho * wo,
            im2col_elems=M * K, im2col_fp8=M * K, im2col_bf16=2 * M * K,
            im2col_fp8_padded=M * pad32(K),
            Mb=Mb, Kt=Kt, Nt=Nt, matmuls=Mb * Kt * Nt, out_tiles=Mb * Nt,
            mac_util_padding=M * K * N / (Mb * Kt * Nt * TILE ** 3),
        ))
        return ho, wo

    # F.pad (1,2,1,2): 200 -> 203; conv1 5x5/2 valid on the padded frame
    hp_, wp_ = h + 3, w + 3
    ho, wo = conv("conv1", 1, 32, 5, 2, 0, hp_, wp_, "stem")
    ho, wo = _out(ho, 3, 2, 0), _out(wo, 3, 2, 0)              # maxpool 3x3/2 -> 49x49
    c = 32
    for i, cout in enumerate((32, 64, 128), 1):
        s0 = f"block{i}"
        a = conv(f"{s0}.conv_a", c, cout, 3, 2, 1, ho, wo, s0)
        conv(f"{s0}.conv_b", cout, cout, 3, 1, 1, *a, s0)
        conv(f"{s0}.skip", c, cout, 1, 2, 0, ho, wo, s0)
        ho, wo = a
        c = cout
    feat = c * ho * wo
    for head in ("steer", "collision"):
        rows.append(dict(
            name=head, kind="linear", stage="head", k=1, stride=1, cin=feat, cout=1,
            in_hw=(1, feat), out_hw=(1, 1), M=1, K=feat, N=1, macs=feat, w_params=feat, bias_params=1,
            w_fp8_bytes=feat, w_bf16_bytes=2 * feat, w_fp8_bytes_padded=pad32(feat) * pad32(1),
            in_elems=feat, out_elems=1, in_fp8=feat, in_bf16=2 * feat, out_fp8=1, out_bf16=2,
            im2col_elems=feat, im2col_fp8=0, im2col_bf16=0, im2col_fp8_padded=0,
            Mb=1, Kt=cdiv(feat, TILE), Nt=1, matmuls=cdiv(feat, TILE), out_tiles=1,
            mac_util_padding=feat / (cdiv(feat, TILE) * TILE ** 3),
        ))
    return rows


def hand_macs() -> int:
    return sum(r["macs"] for r in layer_table())


def nonconv_table(h: int = 200, w: int = 200) -> list[dict]:
    """Element counts of every non-contraction op (batch 1). `unit`: VPU = vector work, data =
    data movement only, host = no Atlas instruction (chip profile: op_unit_map.py)."""
    L = {r["name"]: r for r in layer_table(h, w)}
    ops = [dict(op="pad(1,2,1,2)", where="input", elems=(h + 3) * (w + 3), unit="data", note="TF SAME for 5x5/2")]
    c1 = L["conv1"]
    ops.append(dict(op="conv1 bias + rescale", where="conv1", elems=c1["out_elems"], unit="VPU", note="2 binary ops"))
    pool_hw = (_out(c1["out_hw"][0], 3, 2, 0), _out(c1["out_hw"][1], 3, 2, 0))
    ops.append(dict(op="maxpool3x3/2", where="stem", elems=32 * pool_hw[0] * pool_hw[1], unit="host",
                    note="max_pool2d_with_indices: no mapped Atlas instruction (chip coverage); a vmaximum "
                         "decomposition (8 binary ops per output tile) is possible but NOT implemented in the router"))
    for i in (1, 2, 3):
        a, b, s = L[f"block{i}.conv_a"], L[f"block{i}.conv_b"], L[f"block{i}.skip"]
        ops += [
            dict(op="affine_a (x*scale+shift)", where=f"block{i}", elems=a["in_elems"], unit="VPU", note="vmul + vadd"),
            dict(op="relu (after affine_a)", where=f"block{i}", elems=a["in_elems"], unit="VPU", note="vrelu"),
            dict(op="conv_a bias + rescale", where=f"block{i}", elems=a["out_elems"], unit="VPU", note="2 binary"),
            dict(op="affine_b", where=f"block{i}", elems=a["out_elems"], unit="VPU", note="vmul + vadd"),
            dict(op="relu (after affine_b)", where=f"block{i}", elems=a["out_elems"], unit="VPU", note="vrelu"),
            dict(op="conv_b bias + rescale", where=f"block{i}", elems=b["out_elems"], unit="VPU", note="2 binary"),
            dict(op="skip bias + rescale", where=f"block{i}", elems=s["out_elems"], unit="VPU", note="2 binary"),
            dict(op="residual add", where=f"block{i}", elems=b["out_elems"], unit="VPU", note="vadd"),
        ]
    feat = L["steer"]["K"]
    ops += [dict(op="relu (flatten)", where="head", elems=feat, unit="VPU", note="vrelu"),
            dict(op="head bias + rescale", where="head", elems=2, unit="VPU", note="2 binary, 2 heads"),
            dict(op="cat", where="head", elems=2, unit="data", note="DMA into one buffer"),
            dict(op="sigmoid(collision)", where="host", elems=1, unit="host", note="by design, collision_probability()")]
    return ops


def totals() -> dict:
    t = layer_table()
    mac = sum(r["macs"] for r in t)
    return dict(
        macs=mac, flops=2 * mac, macs_fp8_contractions=mac, mac_share_fp8=1.0,
        w_params=sum(r["w_params"] for r in t), bias_params=sum(r["bias_params"] for r in t),
        w_fp8_bytes=sum(r["w_fp8_bytes"] for r in t), w_bf16_bytes=sum(r["w_bf16_bytes"] for r in t),
        w_fp8_bytes_padded=sum(r["w_fp8_bytes_padded"] for r in t),
        matmuls=sum(r["matmuls"] for r in t),
        matmuls_fused_heads=sum(r["matmuls"] for r in t if r["kind"] == "conv") + t[-1]["matmuls"],
        padded_macs=sum(r["matmuls"] for r in t) * TILE ** 3,
    )


# ----------------------------------------------------------------------------------------------
# Part 2: memory liveness (does it fit VMEM?)
# ----------------------------------------------------------------------------------------------
def liveness(act_dtype_bytes: int = 2, materialize_im2col: bool = True) -> list[dict]:
    """Bytes live at each contraction, no tiling. Inter-layer tensors are BF16 (`act_dtype_bytes`=2:
    pop gives BF16 and the BN/residual math is BF16); each FP8 operand (the cast of the layer input) and,
    if `materialize_im2col`, the FP8 im2col matrix are live too. Weights are NOT included (see
    `memory_summary`). A residual block keeps its raw input x alive until the add."""
    L = {r["name"]: r for r in layer_table()}
    rows = []

    def add(name, parts):
        rows.append(dict(stage=name, parts=parts, bytes=sum(parts.values())))

    c1 = L["conv1"]
    add("conv1", {"input fp8 (203x203 padded)": 203 * 203, "im2col fp8 [10000x32]": c1["M"] * pad32(c1["K"]) if materialize_im2col else 0,
                  "conv1 out bf16": c1["out_bf16"]})
    pool_elems = 32 * 49 * 49
    add("maxpool", {"conv1 out bf16": c1["out_bf16"], "pool out bf16": 2 * pool_elems})
    for i in (1, 2, 3):
        a, b, s = L[f"block{i}.conv_a"], L[f"block{i}.conv_b"], L[f"block{i}.skip"]
        x_bf16 = a["in_bf16"]
        add(f"block{i}.conv_a", {"x bf16 (kept for skip+add)": x_bf16, "operand fp8": a["in_fp8"],
                                 "im2col fp8": a["M"] * a["K"] if materialize_im2col else 0, "out bf16": a["out_bf16"]})
        add(f"block{i}.conv_b", {"x bf16": x_bf16, "operand fp8": b["in_fp8"],
                                 "im2col fp8": b["M"] * b["K"] if materialize_im2col else 0,
                                 "conv_a relu'd (bf16, freed after cast)": 0, "out bf16": b["out_bf16"]})
        add(f"block{i}.skip", {"x bf16": x_bf16, "operand fp8": s["in_fp8"], "conv_b out bf16": b["out_bf16"],
                               "skip out bf16": s["out_bf16"]})
    h = L["steer"]
    add("heads", {"x fp8 (6272)": h["in_fp8"], "x bf16": h["in_bf16"]})
    return rows


def memory_summary() -> dict:
    t = totals()
    live = liveness()
    peak = max(live, key=lambda r: r["bytes"])
    live_noim = max(liveness(materialize_im2col=False), key=lambda r: r["bytes"])
    heads_padded = sum(r["w_fp8_bytes_padded"] for r in layer_table() if r["kind"] == "linear")
    heads_fused_padded = pad32(layer_table()[-1]["K"]) * pad32(2)
    conv_w_padded = t["w_fp8_bytes_padded"] - heads_padded
    win, full = 768 * 1024, hp.VMEM_BYTES
    resident = [(r["stage"], r["bytes"] + conv_w_padded) for r in live]
    return dict(
        conv_weights_fp8_padded=conv_w_padded,
        stages_over_window_with_conv_weights=[st for st, b in resident if b > win],
        stages_over_full_with_conv_weights=[st for st, b in resident if b > full],
        peak_stage_with_conv_weights=max(b for _, b in resident),
        weights_fp8_unpadded=t["w_fp8_bytes"], weights_fp8_padded=t["w_fp8_bytes_padded"],
        heads_padded_separate=heads_padded, heads_padded_fused=heads_fused_padded,
        peak_activation_bytes=peak["bytes"], peak_stage=peak["stage"], peak_parts=peak["parts"],
        peak_activation_bytes_no_im2col=live_noim["bytes"], peak_stage_no_im2col=live_noim["stage"],
        vmem_full=hp.VMEM_BYTES, vmem_window=768 * 1024,
        peak_plus_weights=peak["bytes"] + t["w_fp8_bytes_padded"],
        fits_full_untiled=peak["bytes"] + t["w_fp8_bytes_padded"] <= hp.VMEM_BYTES,
        fits_window_untiled=peak["bytes"] + t["w_fp8_bytes_padded"] <= 768 * 1024,
        weights_alone_fit_window=t["w_fp8_bytes_padded"] <= 768 * 1024,
        per_stage=[(r["stage"], r["bytes"]) for r in live],
    )


# ----------------------------------------------------------------------------------------------
# Part 3: cycle model
# ----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """Every field is an ASSUMPTION unless its comment cites a source."""
    name: str
    combine: str                 # "max" (resources overlap) or "sum" (they do not)
    n_mxu: int                   # MXUs used; 1 = MXU1 (IPT) only, 2 = both (spec 02 MXU_COUNT 2)
    mxu_sched_eff: float         # fraction of the 32-cycle tile rate sustained; 0.93 = measured, perf_mm_dual_128x128x128.S:12
    weight_reuse: str            # "once" (tile loaded once per layer), "pair" (2 accumulators, l2 script), "none"
    act_share_across_n: bool     # one activation vload feeds the n-tiles of both MXUs
    vstore_mregs_per_tile: float  # 1 = FP8 tile, 2 = BF16 tile spilled to VMEM
    vpu_binary_cycles: float     # 65 theoretical (66-1), 78 measured-ish (perf_vpu_binary.S: 460/6 ops)
    vpu_unary_pair: float        # relu/pack: 1 = no slot pairing, 2 = two slots busy
    dynamic_scale: bool          # amax chain (vredmax.row, vredmin.row + cross-tile max/min) before every FP8 cast
    fold_affine: bool            # bn_b affine folded into conv_a bias/rescale (graph-level rewrite, not done today)
    cast_after_im2col: bool      # cast the im2col'd (9x duplicated) matrix, as the reference profile does
    fuse_heads: bool             # two heads as one N=2 matmul
    maxpool: str                 # "vpu" (8 vmaximum / out tile, not implemented in the router) or "host"
    host_cycles_per_elem: float  # host maxpool cost per input element (UNVERIFIED: Shuttle throughput unknown)
    relayout: str                # stride-2 operand layout: "free" (parity-split producer), "scalar" (row copies), "dma" (im2col via DMA)
    relayout_B_per_cycle_scalar: float = 2.0   # S1 issues 1 instr/cycle: lw+sw per 4 B -> 2 B/cycle (ASSUMED)
    input_bytes: int = 40_000    # DMA'd per frame: 40000 raw FP8 | 320000 host-built im2col [10000,32] FP8
    weights_dma_each_frame: bool = False   # weights resident across frames unless the 768 KiB window forces reload
    fill_per_layer: int = (hp.MXU_MATMUL_LATENCY["ipt"] + hp.MXU_PUSH_POP_LATENCY + hp.VSTORE_CYCLES
                           + 2 * (hp.VPU_TILE_ELEMENTWISE_CYCLES + 1))  # dependent-chain drain at each layer boundary (DERIVED)


SCENARIOS = {
    "high": Scenario(
        "high (speed of light of an optimised schedule)", "max", n_mxu=2, mxu_sched_eff=0.93, weight_reuse="once",
        act_share_across_n=True, vstore_mregs_per_tile=1.0, vpu_binary_cycles=65.0, vpu_unary_pair=2.0,
        dynamic_scale=False, fold_affine=True, cast_after_im2col=False, fuse_heads=True, maxpool="vpu",
        host_cycles_per_elem=0.0, relayout="free", input_bytes=40_000),
    "nominal": Scenario(
        "nominal (plausible compiler output)", "max", n_mxu=1, mxu_sched_eff=0.75, weight_reuse="pair",
        act_share_across_n=False, vstore_mregs_per_tile=2.0, vpu_binary_cycles=78.0, vpu_unary_pair=1.0,
        dynamic_scale=False, fold_affine=False, cast_after_im2col=False, fuse_heads=False, maxpool="vpu",
        host_cycles_per_elem=0.0, relayout="scalar", input_bytes=320_000),
    "low": Scenario(
        "low (naive: reference numerics, no overlap, host maxpool)", "sum", n_mxu=1, mxu_sched_eff=0.5,
        weight_reuse="none", act_share_across_n=False, vstore_mregs_per_tile=2.0, vpu_binary_cycles=78.0,
        vpu_unary_pair=1.0, dynamic_scale=True, fold_affine=False, cast_after_im2col=True, fuse_heads=False,
        maxpool="host", host_cycles_per_elem=4.0, relayout="dma", input_bytes=320_000, weights_dma_each_frame=True),
}

DMA_LOAD_B_PER_CYCLE = hp.DMA_LOAD_BYTES_PER_CYCLE                       # about 1.34 (4 KiB loads, measured in sim)
DMA_STORE_B_PER_CYCLE = (4 * hp.KIB) / hp.DMA_STORE_CYCLES[4 * hp.KIB]    # about 1.45


def _vpu_tile_ops(sc: Scenario) -> dict[str, dict]:
    """VPU tile-op counts per stage: {stage: {"bin": n, "un": n, "pack": n, "rowred": n}} (tiles x ops)."""
    L = {r["name"]: r for r in layer_table()}
    st: dict[str, dict] = {}

    def add(stage, kind, n):
        d = st.setdefault(stage, dict(bin=0, un=0, pack=0, rowred=0))
        d[kind] += n

    def cast(stage, n_tiles):
        add(stage, "pack", n_tiles)
        if sc.dynamic_scale:
            add(stage, "rowred", 2 * n_tiles)     # vredmax.row + vredmin.row
            add(stage, "bin", 2 * n_tiles)        # fold the per-tile extrema (vmaximum / vminimum)

    c1 = L["conv1"]
    add("conv1", "bin", 2 * c1["out_tiles"])
    pool_tiles = tiles(49 * 49, 32)
    if sc.maxpool == "vpu":
        add("maxpool", "bin", 8 * pool_tiles)
    for i in (1, 2, 3):
        a, b, s = L[f"block{i}.conv_a"], L[f"block{i}.conv_b"], L[f"block{i}.skip"]
        tx = tiles(a["in_hw"][0] * a["in_hw"][1], a["cin"])
        tg = a["out_tiles"]
        stg = f"block{i}"
        add(stg, "bin", 2 * tx)                    # affine_a
        add(stg, "un", tx)                         # relu
        cast_a = tiles(a["M"], a["K"]) if sc.cast_after_im2col else tx
        cast(stg, cast_a)                          # operand of conv_a
        cast_s = tiles(s["M"], s["K"]) if sc.cast_after_im2col else tx
        cast(stg, cast_s)                          # raw x, operand of skip
        add(stg, "bin", 2 * tg)                    # conv_a rescale + bias
        if not sc.fold_affine:
            add(stg, "bin", 2 * tg)                # affine_b
        add(stg, "un", tg)                         # relu
        cast_b = tiles(b["M"], b["K"]) if sc.cast_after_im2col else tg
        cast(stg, cast_b)                          # operand of conv_b
        add(stg, "bin", 2 * b["out_tiles"] + 2 * s["out_tiles"] + b["out_tiles"])   # rescale+bias x2, residual add
    feat = L["steer"]["K"]
    ft = cdiv(feat, TILE * TILE)
    add("head", "un", ft)
    cast("head", ft)
    add("head", "bin", 2 * (1 if sc.fuse_heads else 2))
    return st


def _stage_of(name: str) -> str:
    return name.split(".")[0] if name.startswith("block") else ("conv1" if name == "conv1" else "head")


def _unit_cycles(sc: Scenario) -> dict[str, float]:
    return dict(bin=sc.vpu_binary_cycles, un=hp.VPU_TILE_ELEMENTWISE_CYCLES / sc.vpu_unary_pair,
                pack=hp.VPU_TILE_ELEMENTWISE_CYCLES, rowred=hp.VREDMAX_ROW_CYCLES - 1)


def layer_cycles(sc: Scenario) -> list[dict]:
    """Per contraction layer: cycles charged to each resource, and the layer total under `sc.combine`."""
    L = layer_table()
    if sc.fuse_heads:
        h0, h1 = L[-2], L[-1]
        L = L[:-2] + [dict(h0, name="heads(fused N=2)", N=2, macs=h0["macs"] + h1["macs"])]
    vpu = _vpu_tile_ops(sc)
    uc = _unit_cycles(sc)
    stage_cyc = {s: sum(uc[k] * n for k, n in d.items()) for s, d in vpu.items()}
    stage_layers: dict[str, list] = {}
    for r in L:
        stage_layers.setdefault(_stage_of(r["name"]), []).append(r)
    out = []
    for r in L:
        n_eff = max(1, min(sc.n_mxu, r["out_tiles"]))
        mxu = r["matmuls"] * hp.MXU_ISSUE_GAP / (n_eff * sc.mxu_sched_eff)
        pop = r["out_tiles"] * hp.MXU_ISSUE_GAP / n_eff
        n_act = r["Mb"] * r["Kt"] * (cdiv(r["Nt"], 2) if sc.act_share_across_n and sc.n_mxu > 1 else r["Nt"])
        n_w = {"once": r["Kt"] * r["Nt"], "pair": cdiv(r["matmuls"], 2), "none": r["matmuls"]}[sc.weight_reuse]
        vload = (n_act + n_w) * hp.VSTORE_CYCLES
        vstore = r["out_tiles"] * sc.vstore_mregs_per_tile * hp.VSTORE_CYCLES
        # VPU stage cost is shared by the stage's layers in proportion to their output tiles
        sl = stage_layers[_stage_of(r["name"])]
        share = r["out_tiles"] / sum(x["out_tiles"] for x in sl)
        vpu_c = stage_cyc.get(_stage_of(r["name"]), 0.0) * share
        # stride-2 operand re-layout (pixels are not contiguous for vload, spec 06:294)
        relayout = 0.0
        if r["stride"] == 2 and r["kind"] == "conv" and r["name"] != "conv1":   # conv1 operand: see Scenario.input_bytes
            if sc.relayout == "scalar":
                relayout = r["in_fp8"] / sc.relayout_B_per_cycle_scalar
            elif sc.relayout == "dma":
                relayout = (r["M"] * r["K"]) / DMA_LOAD_B_PER_CYCLE      # gather im2col rows through DMA
        wdma = (r["w_fp8_bytes"] / DMA_LOAD_B_PER_CYCLE) if sc.weights_dma_each_frame else 0.0
        res = dict(mxu=mxu, pop=pop, vload=vload, vstore=vstore, vpu=vpu_c, relayout=relayout, wdma=wdma)
        body = max(res.values()) if sc.combine == "max" else sum(res.values())
        bound = max(res, key=res.get)
        out.append(dict(name=r["name"], matmuls=r["matmuls"], resources=res, bound=bound,
                        cycles=body + sc.fill_per_layer))
    return out


def frame_estimate(sc: Scenario) -> dict:
    """Cycles per frame for one scenario: compute path, DMA path, host path, and the two totals."""
    layers = layer_cycles(sc)
    compute = sum(l["cycles"] for l in layers)
    vpu_stages = _vpu_tile_ops(sc)
    uc = _unit_cycles(sc)
    pool_vpu = sum(uc[k] * n for k, n in vpu_stages.get("maxpool", {}).items())
    extra = 0.0
    host = 0.0
    if sc.maxpool == "vpu":
        extra += pool_vpu
    else:
        c1 = layer_table()[0]
        host = (32 * 100 * 100) * sc.host_cycles_per_elem
        host += c1["out_bf16"] / DMA_STORE_B_PER_CYCLE + 2 * 32 * 49 * 49 / DMA_LOAD_B_PER_CYCLE   # VMEM->DRAM and back
    dma_in = sc.input_bytes / DMA_LOAD_B_PER_CYCLE
    dma_out = 4 / DMA_STORE_B_PER_CYCLE + hp.DMA_STORE_CYCLES[1 * hp.KIB]    # two BF16 logits, one small transfer
    compute_all = compute + extra + host
    if sc.combine == "sum":
        throughput = latency = compute_all + dma_in + dma_out
    else:
        latency = dma_in + compute_all + dma_out
        throughput = max(compute_all, dma_in + dma_out)       # next frame's input DMA hidden behind this frame
    totals_ = {k: sum(l["resources"][k] for l in layers) for k in layers[0]["resources"]}
    totals_["maxpool_vpu"] = extra
    totals_["host_maxpool"] = host
    totals_["dma_frame_io"] = dma_in + dma_out
    return dict(scenario=sc.name, resource_totals=totals_, compute_cycles=compute, maxpool_vpu_cycles=extra, host_cycles=host,
                dma_in_cycles=dma_in, dma_out_cycles=dma_out, cycles_throughput=throughput,
                cycles_latency=latency, layers=layers,
                fps={k: dict(throughput=v / throughput, latency=v / latency) for k, v in CLOCKS_HZ.items()})


def estimate() -> dict:
    return {k: frame_estimate(sc) for k, sc in SCENARIOS.items()}


# ----------------------------------------------------------------------------------------------
# Cross-check against torch
# ----------------------------------------------------------------------------------------------
def flopcounter_macs(model=None) -> int:
    """MACs from torch.utils.flop_counter.FlopCounterMode (it counts 2 FLOPs per MAC; divided here)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    from dronet.model import DroNet

    m = (model or DroNet()).eval().float()
    x = torch.zeros(1, 1, 200, 200)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    return fc.get_total_flops() // 2
