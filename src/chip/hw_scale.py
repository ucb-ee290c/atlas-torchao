"""Dynamic-scale derivation on the Atlas scalar core (RV32I only) and its cost (closes hardware question H2).

The RTL has no exponent-extract instruction. The only route from data to a scale register is
    vredmax.row / vredmin.row (34 cycles) -> vstore (35) -> scalar lhu of the BF16 amax pattern
    -> integer ops -> sb of the E8M0 byte to VMEM -> seld (4 cycles).
The scalar core is RV32I: no MUL, no FP. `scale_exp_rv32` is that integer step, written as a short
RV32I program (`PROGRAMS`), executed by a tiny interpreter that counts instructions. It is bit-equal
to the model's scale rules on all 65,536 BF16 patterns (tests/test_hw_scale.py):

    rule "ref"  ref_numerics.po2_scale_exp          s = floor(log2 amax) - 8
    rule "nc"   algos.granularity.scale_exp (nc)    s = ceil(log2(amax / 448)) = ref + (amax > 1.75 * 2^floor(log2 amax))

Derivation (BF16 pattern b = sign:1 | e:8 | m:7, amax = |value|):
  - E8M0 byte = s + 127. With e the biased exponent field, floor(log2 amax) = e - 127, so
    ref byte = e - 8, and nc byte = e - 8 + bump, bump = (m > 96) because 448 * 2^(E-8) = 1.75 * 2^E and
    1.75 = 1 + 96/128 (m = 96 is exactly 448 after scaling: fits). bump = (m + 31) >> 7, no compare.
  - amax = 0, +-Inf, NaN: the reference does nan_to_num(nan=0, posinf=0) of |x|, so s = 0, byte 127.
  - BF16 subnormals and tiny normals give byte < 0: clamped to byte 0 (s = -127, the smallest E8M0
    scale). The reference rule has no clamp, so for amax below about 2^-119 the model's s is lower
    than -127; the chip value is clip(model s, -127, 127). The upper clamp never fires (max byte 247).
  - The sign bit is dropped for free by the 8-bit mask, so a signed amax pattern works too.

Cost helpers (`dynamic_scale_cost`) model the full chain per 32-token block / per token; every
latency is serialized (no overlap is claimed). The scalar CPI of 1 and the layout assumptions are
DERIVED, not measured, and listed in COST_ASSUMPTIONS.
"""

from __future__ import annotations

import math

from chip.hw_params import (SCALE_BIAS, SELD_CYCLES, TILE, VPU_TILE_ELEMENTWISE_CYCLES, VREDMAX_ROW_CYCLES,
                       VSTORE_CYCLES, VTRPOSE_CYCLES)

MASK32 = 0xFFFFFFFF

# --- the RV32I sequences: input pattern in a0, E8M0 byte out in t0 -----------------------------
# (mnemonic, operands...). Labels are strings. Only RV32I instructions: no mul, no F/M extension.
_NC_BUMP = [("addi", "t1", "t1", 31), ("srli", "t1", "t1", 7), ("add", "t0", "t0", "t1")]
PROGRAMS = {
    "nc": [
        ("srli", "t0", "a0", 7),            # sign:e
        ("andi", "t0", "t0", 255),          # e (sign dropped)
        ("addi", "t2", "t0", -255),
        ("beq", "t2", "zero", "special"),   # Inf / NaN
        ("andi", "t1", "a0", 127),          # m
        *_NC_BUMP[:2],                      # bump = (m + 31) >> 7  (1 iff m > 96)
        _NC_BUMP[2],                        # byte = e + bump
        ("addi", "t0", "t0", -8),           # byte = e - 8 + bump
        ("bge", "t0", "zero", "done"),
        ("slli", "t1", "a0", 17),           # byte < 0: amax == 0 ?
        ("beq", "t1", "zero", "special"),
        ("addi", "t0", "zero", 0),          # subnormal / tiny: clamp to byte 0
        ("jal", "zero", "done"),
        "special",
        ("addi", "t0", "zero", SCALE_BIAS),  # amax 0 / Inf / NaN: s = 0
        "done",
    ],
    "ref": [
        ("srli", "t0", "a0", 7),
        ("andi", "t0", "t0", 255),
        ("addi", "t2", "t0", -255),
        ("beq", "t2", "zero", "special"),
        ("addi", "t0", "t0", -8),
        ("bge", "t0", "zero", "done"),
        ("slli", "t1", "a0", 17),
        ("beq", "t1", "zero", "special"),
        ("addi", "t0", "zero", 0),
        ("jal", "zero", "done"),
        "special",
        ("addi", "t0", "zero", SCALE_BIAS),
        "done",
    ],
}
RV32I_ALLOWED = {"add", "addi", "andi", "srli", "slli", "xori", "or", "slt", "sltiu", "beq", "bne", "bge",
                 "blt", "jal", "lhu", "sb", "sh", "sub"}
# load / store around the sequence in the real chain
LHU, SB = ("lhu", "a0", "a1", 0), ("sb", "t0", "a2", 0)


def _reg(r):
    return 0 if r == "zero" else r


def run_program(prog, bits: int, with_memory_ops: bool = False):
    """Interpret `prog` on a 16-bit pattern. Returns (t0 byte, executed instruction count)."""
    flat, labels = [], {}
    for x in prog:
        if isinstance(x, str):
            labels[x] = len(flat)
        else:
            assert x[0] in RV32I_ALLOWED, f"{x[0]} is not an RV32I instruction"
            flat.append(x)
    regs = {"a0": bits & 0xFFFF}
    get = lambda r: 0 if r == "zero" else regs.get(r, 0) & MASK32
    sx = lambda v: v - (1 << 32) if v & 0x80000000 else v
    pc = n = 0
    while pc < len(flat):
        op, *a = flat[pc]
        n += 1
        nxt = pc + 1
        if op == "addi":
            regs[a[0]] = (get(a[1]) + a[2]) & MASK32
        elif op == "add":
            regs[a[0]] = (get(a[1]) + get(a[2])) & MASK32
        elif op == "andi":
            regs[a[0]] = get(a[1]) & (a[2] & MASK32)
        elif op == "srli":
            regs[a[0]] = get(a[1]) >> a[2]
        elif op == "slli":
            regs[a[0]] = (get(a[1]) << a[2]) & MASK32
        elif op == "beq":
            nxt = labels[a[2]] if get(a[0]) == get(a[1]) else nxt
        elif op == "bge":
            nxt = labels[a[2]] if sx(get(a[0])) >= sx(get(a[1])) else nxt
        elif op == "jal":
            nxt = labels[a[1]]
        else:
            raise NotImplementedError(op)
        pc = nxt
    return sx(get("t0")), n + (2 if with_memory_ops else 0)


def scale_exp_rv32_counted(amax_bf16_bits: int, rule: str = "nc"):
    """(E8M0 exponent s = byte - 127, scalar instructions executed by the integer sequence)."""
    byte, n = run_program(PROGRAMS[rule], amax_bf16_bits)
    return byte - SCALE_BIAS, n


def scale_exp_rv32(amax_bf16_bits: int, rule: str = "nc") -> int:
    """E8M0 scale exponent s (real value = e4m3(code) * 2^s) from the 16-bit BF16 amax pattern, using
    only RV32I integer operations. The byte the chip stores is s + 127. rule 'nc' (default) never clips."""
    return scale_exp_rv32_counted(amax_bf16_bits, rule)[0]


def program_counts(rule: str = "nc") -> dict:
    """Static length, and executed-count extremes over all 65,536 patterns, of the sequence alone
    (excluding the lhu before and the sb after it, which the chain counts separately)."""
    ns = [run_program(PROGRAMS[rule], b)[1] for b in range(1 << 16)]
    prog = [x for x in PROGRAMS[rule] if isinstance(x, tuple)]
    normal = run_program(PROGRAMS[rule], 0x3F80)[1]       # 1.0: finite, non-clamped path
    return {"static": len(prog), "normal_path": normal, "min": min(ns), "max": max(ns)}


SEQ_NORMAL = {r: program_counts(r)["normal_path"] for r in PROGRAMS}
SEQ_MAX = {r: program_counts(r)["max"] for r in PROGRAMS}


# --- cost of the whole dynamic-scale chain ------------------------------------------------------
COST_ASSUMPTIONS = (
    "Counts are instructions issued and serialized cycles (sum of latencies, no overlap claimed). "
    "A BF16 32x32 tile is two 16-lane mregs: an elementwise tile op is 2 VPU instructions and "
    f"{VPU_TILE_ELEMENTWISE_CYCLES} cycles; vredmax.row / vredmin.row / vtrpose / vstore are one "
    "instruction each at their hw_params latency (the reduction is treated as one instruction per tile; "
    "DERIVED). Scalar core: 1 instruction per cycle, lhu/sb included (DERIVED; not in the RTL timing doc). "
    "amax = max(vredmax, -vredmin): a vmul by -1 and a vmax per tile. The cross-row reduction of a "
    "tile's 32 row maxima is a vtrpose.xlu plus one more vredmax.row. The per-token factor vector is "
    "assumed to be broadcast across the tile columns by one vload (a LOWER BOUND: if the VPU has no "
    "such broadcast, building the 32x32 factor tile costs up to 32x more sh per token block).")


def _tiles(k: int) -> int:
    return -(-k // TILE)


def dynamic_scale_cost(rows: int, k: int, mode: str, rule: str = "nc") -> dict:
    """Cost of deriving the dynamic activation scales of a [rows, k] Linear input.

    mode 'block': one scale per 32-token block over the full K (block32 of docs/algos/granularity.md).
          Per K tile: vredmax.row, vredmin.row, vmul(-1), vmax, vtrpose.xlu, vredmax.row, vstore,
          scalar lhu; across the K/32 tile amaxes a scalar integer max (BF16 amax patterns of
          non-negative values order as integers: 2 instructions per extra tile); then once per block the
          exponent sequence, sb, seld. No pre-vpack vmul: the vpack reads the scale register.
    mode 'block_vec': as 'block', but the max across K tiles is taken on the row-amax vectors in the VPU
          (t-1 elementwise vmax) and the cross-row reduction, vstore, lhu, sequence, sb, seld happen once
          per block (cheaper variant; not what granularity a=block counts by default).
    mode 'token': one scale per token. Per K tile: vredmax.row, vredmin.row, vmul(-1), vmax; across
          tiles a vmax of the row-amax vectors; one vstore per block; then per token lhu, the
          exponent sequence, the BF16 bits of 2^-s (addi/sub/slli), sb of the byte is NOT needed (no
          seld: vpack uses a fixed scale), one sh into the factor vector; then one vload and, per K
          tile, a vmul of the tile by the factor tile (2 instructions).
    Returns {"vpu": {instr: n}, "scalar": {"rv32i": n}, "instructions": total, "cycles": serialized,
             "cycle_parts": {...}, "blocks": n}.
    """
    t = _tiles(k)
    blocks = -(-rows // TILE)
    seq = SEQ_NORMAL[rule]
    ew = 2                                                 # instructions per elementwise tile op
    vpu: dict = {}
    scalar = 0
    cyc = {}

    def add(d, key, n):
        d[key] = d.get(key, 0) + n

    if mode == "block":
        add(vpu, "vredmax.row.bf16", 2 * t * blocks)       # row max, then cross-row max
        add(vpu, "vredmin.row.bf16", t * blocks)
        add(vpu, "vmul.bf16", ew * t * blocks)             # -vmin
        add(vpu, "vmax.bf16", ew * t * blocks)
        add(vpu, "vtrpose.xlu", t * blocks)
        add(vpu, "vstore", t * blocks)
        scalar = blocks * (t * 1 + (t - 1) * 2 + seq + 1)  # lhu per tile, max across tiles, sequence, sb
        seld = blocks
        cyc["vredmax.row"] = 2 * VREDMAX_ROW_CYCLES * t * blocks
        cyc["vredmin.row"] = VREDMAX_ROW_CYCLES * t * blocks
        cyc["elementwise"] = 2 * VPU_TILE_ELEMENTWISE_CYCLES * t * blocks
        cyc["vtrpose"] = VTRPOSE_CYCLES * t * blocks
        cyc["vstore"] = VSTORE_CYCLES * t * blocks
        cyc["scalar"] = scalar
        cyc["seld"] = SELD_CYCLES * seld
        cyc["pre_vpack_vmul"] = 0
    elif mode == "block_vec":
        # variant: the cross-K max is done on the row-amax vectors in the VPU (t-1 elementwise vmax), then
        # ONE cross-row reduction (vtrpose + vredmax.row), vstore, lhu, sequence, sb, seld per block
        add(vpu, "vredmax.row.bf16", (t + 1) * blocks)
        add(vpu, "vredmin.row.bf16", t * blocks)
        add(vpu, "vmul.bf16", ew * t * blocks)
        add(vpu, "vmax.bf16", ew * (t + t - 1) * blocks)
        add(vpu, "vtrpose.xlu", blocks)
        add(vpu, "vstore", blocks)
        scalar = blocks * (1 + seq + 1)
        seld = blocks
        cyc["vredmax.row"] = VREDMAX_ROW_CYCLES * (t + 1) * blocks
        cyc["vredmin.row"] = VREDMAX_ROW_CYCLES * t * blocks
        cyc["elementwise"] = VPU_TILE_ELEMENTWISE_CYCLES * (t + t + t - 1) * blocks
        cyc["vtrpose"] = VTRPOSE_CYCLES * blocks
        cyc["vstore"] = VSTORE_CYCLES * blocks
        cyc["scalar"] = scalar
        cyc["seld"] = SELD_CYCLES * seld
        cyc["pre_vpack_vmul"] = 0
    elif mode == "token":
        add(vpu, "vredmax.row.bf16", t * blocks)
        add(vpu, "vredmin.row.bf16", t * blocks)
        add(vpu, "vmul.bf16", ew * t * blocks)             # -vmin
        add(vpu, "vmax.bf16", ew * t * blocks + ew * (t - 1) * blocks)   # combine + across K tiles
        add(vpu, "vstore", blocks)
        add(vpu, "vload", blocks)                          # the broadcast factor vector (lower bound)
        add(vpu, "vmul.bf16", ew * t * blocks)             # x * 2^-s per K tile (pre-vpack)
        per_token = 1 + seq + 3 + 1                        # lhu, sequence, addi/sub/slli, sh
        scalar = rows * per_token
        seld = 0
        cyc["vredmax.row"] = VREDMAX_ROW_CYCLES * t * blocks
        cyc["vredmin.row"] = VREDMAX_ROW_CYCLES * t * blocks
        cyc["elementwise"] = VPU_TILE_ELEMENTWISE_CYCLES * (2 * t + (t - 1)) * blocks
        cyc["vstore"] = VSTORE_CYCLES * blocks
        cyc["scalar"] = scalar
        cyc["pre_vpack_vmul"] = VPU_TILE_ELEMENTWISE_CYCLES * t * blocks
        cyc["vload"] = VSTORE_CYCLES * blocks              # vload taken at the vstore latency (DERIVED)
    else:
        raise ValueError(mode)
    vpu_n = sum(vpu.values())
    return {"vpu": vpu, "scalar": {"rv32i": scalar}, "seld": seld, "blocks": blocks,
            "instructions": vpu_n + scalar + seld, "cycles": sum(cyc.values()), "cycle_parts": cyc,
            "sequence_instructions": seq, "assumptions": COST_ASSUMPTIONS}
