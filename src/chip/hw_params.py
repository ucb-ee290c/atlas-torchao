"""Atlas hardware constants shared by the cost model, the shape report and the chip profile.

Where the frozen spec (npu-model/npu_spec) and the RTL disagree, the RTL wins; each such value says
so. Sources are in the atlas-npu checkout (RTL = ucb-bar/atlas-npu @ eee27e2, `src/main/scala/...`;
DOC = npu-model/docs/rtl-timing.md, calibrated against Verilator traces; MEAS = a measurement of the
RTL in simulation). Derived values are marked DERIVED. Audit: handoff/HW_AWARE_AUDIT.md.
"""

from __future__ import annotations

KIB, MIB = 1024, 1024 * 1024

# --- Arithmetic formats (RTL) -------------------------------------------------------------------
FP8_MAX = 448.0                      # FP8Pack.scala:40-41 (0x7E); the spec-adjacent Merlin value 240 is wrong
FP8_MIN_NORMAL_EXP = -6              # FP8Pack.scala:97-99: results below 2^-6 flush to zero, no subnormals
FP8_MUL_ACCEPTS_UP_TO = 480.0        # E4M3Mul.scala:34-43: exponent 15 is finite, 0x7F multiplies as 480
SCALE_REGS = 32                      # ScalarISA.scala:158
SCALE_BIAS = 127                     # E8M0, FPUtils.scala:147
VPU_LANES = 16                       # VpuParams.scala:6-12
TILE = 32                            # MxuParams.scala: 32x32 array, 32x32 accumulators

# --- Memory (RTL wins over spec 02:33, which says 1 MiB VMEM) ----------------------------------
VMEM_BYTES = 0x18_0000               # 1.5 MiB, six 256 KiB banks (ScalarCore.scala:49-51, VmemParams.scala:217-259)
MREG_COUNT = 64                      # MregParams.scala:83-90
MREG_BYTES = 32 * 32                 # one tile row-major, 32 rows x 32 B
MREG_TOTAL_BYTES = MREG_COUNT * MREG_BYTES            # 64 KiB
WEIGHT_SLOT_BYTES = TILE * TILE      # 1 KiB of E4M3 (MxuParams.scala:165-185)
WEIGHT_SLOTS_PER_MXU = 2
ACC_BUFFER_BYTES = TILE * TILE * 2   # 32x32 BF16, two per MXU

# --- DMA (RTL structure, VCS measurement of the RTL for the rate; spec 02:90 says 2 B/cycle) ----
DMA_CHANNELS = 8                     # DmaParams.scala:36-47
DMA_BEAT_BYTES = 32
DMA_MAX_TRANSFER_BYTES = 4 * KIB
CLOCK_HZ = 500e6                     # DOC rtl-timing.md:165-166; calibration json frequencyGHz 0.5
# MEAS calibration/dma/measurements/ee290sim_vcs.json (EE290SimConfig, refresh off): cycles per transfer
DMA_LOAD_CYCLES = {32: 55, 1 * KIB: 797, 4 * KIB: 3056}
DMA_STORE_CYCLES = {1 * KIB: 724, 4 * KIB: 2820}
DMA_LOAD_BYTES_PER_CYCLE = (4 * KIB) / DMA_LOAD_CYCLES[4 * KIB]   # DERIVED, about 1.34 B/cycle on 4 KiB loads
DMA_SPEC_BYTES_PER_CYCLE = 2.0       # spec 02:90; unattained, listed for comparison only

# --- Matrix units (software picks by instruction funct7 LSB: Instructions.scala:99-113) --------
# MXU0 = systolic array (SA, RNE after every add), MXU1 = inner-product tree (IPT, one RNE per K tile)
MXU_MATMUL_LATENCY = {"sa": 95, "ipt": 35}       # DOC rtl-timing.md:51 (rows T+63..T+94); spec 02:41 says 96
MXU_ISSUE_GAP = 32                   # cycles between matmuls on one MXU: one 32x32x32 tile per 32 cycles
MXU_PUSH_POP_LATENCY = 33
MXU_POP_AFTER_ACC_GAP = {"sa": 64, "ipt": 32}    # spec 04:174-180

# --- Instruction latencies used by the dynamic-scale chain (spec 04 classes, DOC) --------------
VREDMAX_ROW_CYCLES = 34              # spec 04
VSTORE_CYCLES = 35
SELD_CYCLES = 4                      # spec 04:57
VTRPOSE_CYCLES = 66                  # DOC rtl-timing.md:102-103
VPU_TILE_ELEMENTWISE_CYCLES = 65     # a BF16 32x32 tile is two mregs of 16 lanes (spec 04:153)
VPU_COLUMN_REDUCE_CYCLES = 130       # spec 04:65,134


def dma_load_cycles(nbytes: int) -> float:
    """DERIVED: cycles to load `nbytes` from DRAM at the measured 4 KiB rate plus the 55-cycle floor."""
    transfers = -(-nbytes // DMA_MAX_TRANSFER_BYTES)
    return transfers * (DMA_LOAD_CYCLES[4 * KIB]) if nbytes >= DMA_MAX_TRANSFER_BYTES else \
        DMA_LOAD_CYCLES[32] + (nbytes - 32) * (DMA_LOAD_CYCLES[1 * KIB] - DMA_LOAD_CYCLES[32]) / (KIB - 32)


def mxu_tile_cycles(tokens: int = TILE) -> int:
    """DERIVED: MXU cycles per resident weight tile and token block. The sequencer always iterates all
    32 activation rows (InnerProductTreesSequencer.scala:191-193), so fewer than 32 tokens cost the same.
    The DMA rate above is not scaled by the 8 channels: the TileLink port streams about 23 cycles per
    32 B beat whatever the channel count (found by the cost worker's read, MEAS ee290sim_vcs.json)."""
    return MXU_ISSUE_GAP
