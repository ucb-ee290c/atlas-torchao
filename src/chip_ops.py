"""Atlas VPU emulation in device-agnostic torch, bit-exact to the reference VPU functional model.

`ChipVPU` is a drop-in replacement for the reference author's `VectorRTLFunctions`
(funct_models_vector/vector_rtl_forward.py:92-337): same method names, same shapes, same BF16
conversion of the inputs, same dtype/device of the outputs. The reference adapter loops over
16-lane chunks in Python on the CPU; `ChipVPU` computes whole tensors at once on any device, so the
reference decompositions (smolvla_inout/vpu_decomp.py: softmax, LayerNorm, GELU, SiLU, rsqrt and the
staged row reductions) run unchanged on it, and so do the ones `chip_coverage` adds.

How each instruction is reproduced (the reference stays the oracle, tests/test_chip_ops.py):

  unary (vrecip, vsqrt, vsin, vcos, vlog2, vtanh, vexp, vexp2, vsquare, vcube, vrelu, vmov)
      Every lane box is a pure function of one 16-bit lane, so the reference lane box is evaluated
      once on all 65536 BF16 patterns (VectorEngineModel.execute, vector_engine_model.py:252-281)
      and the result is a 65536-entry table indexed on the device. Exhaustively equal by
      construction; the test re-checks it through the VectorRTLFunctions adapter.
  vadd / vsub
      AddSubSumVec: BF16 lanes widened to FP32, one FP32 RNE add, raw upper 16 bits
      (add_sub_sum_vec.py:281-288; RTL AddSubSumVec.scala:133 fNFromRecFN(8, 24, v)(31, 16)).
  vmul
      MulRec: exact product rounded once to BF16 (bf16_utils.py:186-207), NaN -> 0x7FC0
      (fp_formats.py:155-167).
  vmaximum / vminimum
      sign-magnitude total order, b on ties (bf16_utils.py:60-81; VectorParam.scala:98-110).
  vredsum.row (32 lanes = a BF16 register pair)
      ReduSumRec: FP32 adds over adjacent pairs, 5 levels, one RNE to BF16 at the end
      (vector_engine_model.py:318-330; RTL SumRedu.scala:83-159, VectorEngine.scala:292-299).
  vredmax.row / vredmin.row
      RowMax/RowMin on each 16-lane half, then one compare (vector_engine_model.py:331-340);
      the result is the extreme of the sign-magnitude order, whatever the tree shape.

Inputs are converted with the reference adapter's RNE (torch_float_to_bf16_bits,
vector_rtl_forward.py:66-73), including its NaN behaviour; outputs are BF16 values in the input's
dtype, on the input's device. FP32 arithmetic here is single IEEE operations (add, multiply), never
a fused or reassociated reduction, so no TF32 / FMA contraction can enter.

Every call reports (instruction, element count) to an optional counter, which `chip_coverage` uses
for the per-op instruction mix and element-op totals.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

import torch

import ref_loader

# instruction mnemonics: examples/atlas/.../atlas_isa_green_card.md section 3
INSTR = {
    "add": "vadd.bf16", "sub": "vsub.bf16", "mul": "vmul.bf16",
    "pairmax": "vmaximum.bf16", "pairmin": "vminimum.bf16",
    "rcp": "vrecip.bf16", "sqrt": "vsqrt.bf16", "sin": "vsin.bf16", "cos": "vcos.bf16",
    "log": "vlog2.bf16", "tanh": "vtanh.bf16", "exp": "vexp.bf16", "exp2": "vexp2.bf16",
    "square": "vsquare.bf16", "cube": "vcube.bf16", "relu": "vrelu.bf16", "mov": "vmov",
    "rsum": "vredsum.row.bf16", "rmax": "vredmax.row.bf16", "rmin": "vredmin.row.bf16",
    "fp8pack": "vpack.bf16.fp8",
}
UNARY_OPS = ("rcp", "sqrt", "sin", "cos", "log", "tanh", "exp", "exp2", "square", "cube", "relu", "mov")

_lut_lock = threading.Lock()
_luts: dict[str, torch.Tensor] = {}          # op -> int32[65536] on the CPU
_lut_dev: dict[tuple, torch.Tensor] = {}     # (op, device) -> table on that device


# ---------------------------------------------------------------------------
# BF16 bit helpers
# ---------------------------------------------------------------------------

def f32_to_bf16_bits(x: torch.Tensor) -> torch.Tensor:
    """Port of torch_float_to_bf16_bits (vector_rtl_forward.py:66-73): RNE on the FP32 bits with no
    NaN special case (as the reference adapter does). Returns int32 in [0, 65535]."""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u >> 16) & 0xFFFF
    return (((u + (0x7FFF + (upper & 1))) >> 16) & 0xFFFF).to(torch.int32)


def f32_to_bf16_bits_rne(x: torch.Tensor) -> torch.Tensor:
    """Port of fp_formats.f32_to_bf16_bits_rne (fp_formats.py:155-167): as above, but NaN -> 0x7FC0
    and infinities pass through. Used where the reference model rounds an FP32 result."""
    u = x.float().contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    upper = (u >> 16) & 0xFFFF
    r = ((u + (0x7FFF + (upper & 1))) >> 16) & 0xFFFF
    is_nan = ((u >> 23) & 0xFF == 0xFF) & ((u & 0x7FFFFF) != 0)
    r = torch.where(is_nan, torch.full_like(r, 0x7FC0), r)
    return r.to(torch.int32)


def bf16_bits_to_f32(bits: torch.Tensor) -> torch.Tensor:
    """BF16 bits (int, low 16 bits) -> float32 (exact)."""
    return ((bits.to(torch.int32) & 0xFFFF) << 16).contiguous().view(torch.float32)


def ordered_key(bits: torch.Tensor) -> torch.Tensor:
    """bf16_ordered_key (bf16_utils.py:60-71): sign-magnitude -> total order."""
    return torch.where((bits & 0x8000) != 0, (~bits) & 0xFFFF, bits ^ 0x8000)


def key_to_bits(key: torch.Tensor) -> torch.Tensor:
    """Inverse of ordered_key."""
    return torch.where((key & 0x8000) != 0, key ^ 0x8000, (~key) & 0xFFFF)


# ---------------------------------------------------------------------------
# Unary lane-box tables, built from the reference lane boxes
# ---------------------------------------------------------------------------

def _engine():
    vem = ref_loader.import_ref("funct_models_vector.vector_engine_model")
    vp = ref_loader.import_ref("funct_models_vector.vector_params")
    return vem.VectorEngineModel(vp.VectorParams())       # num_lanes=16 (run_eval.py:528)


def unary_table(op: str) -> torch.Tensor:
    """int32[65536]: the reference lane box `op` on every BF16 pattern (CPU tensor)."""
    with _lut_lock:
        if op not in _luts:
            if op not in UNARY_OPS:
                raise ValueError(f"not a unary VPU op: {op!r}")
            eng = _engine()
            n = eng.p.num_lanes
            out: list[int] = []
            for i in range(0, 65536, n):
                out.extend(eng.execute(op, a_vec=list(range(i, i + n))))
            _luts[op] = torch.tensor(out, dtype=torch.int32) & 0xFFFF
        return _luts[op]


def _table_on(op: str, device) -> torch.Tensor:
    key = (op, str(device))
    t = _lut_dev.get(key)
    if t is None:
        t = unary_table(op).to(device=device, dtype=torch.int64)
        _lut_dev[key] = t
    return t


# ---------------------------------------------------------------------------
# The vector unit
# ---------------------------------------------------------------------------

class ChipVPU:
    """Vectorized torch stand-in for VectorRTLFunctions (same interface, same results).

    counter: optional callable(instruction_mnemonic, n_elements) called once per instruction
    issue group (every element of the tensor counts as one lane-op).
    """

    def __init__(self, num_lanes: int = 16, counter: Optional[Callable[[str, int], None]] = None):
        if num_lanes != 16:
            # The lane count of the reference adapter (run_eval.py:528) and of the RTL
            # (VectorParams.num_lanes=16); results do not depend on it except for row-reduce width.
            raise ValueError("ChipVPU models the 16-lane VPU (2 x 16 = 32-element row reductions)")
        self.num_lanes = num_lanes
        self.counter = counter

    def _count(self, op: str, n: int) -> None:
        if self.counter is not None:
            self.counter(INSTR[op], int(n))

    # -- conversions ------------------------------------------------------------------------
    @staticmethod
    def _bits(a: torch.Tensor) -> torch.Tensor:
        return f32_to_bf16_bits(a)

    @staticmethod
    def _out(bits: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return bf16_bits_to_f32(bits).to(like.dtype)

    # -- binary pointwise --------------------------------------------------------------------
    def _check(self, op, a, b):
        if a.shape != b.shape:
            raise ValueError(f"{op}: shape mismatch a={tuple(a.shape)} b={tuple(b.shape)}")

    def _addsub(self, op: str, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        self._check(op, a, b)
        self._count(op, a.numel())
        a32 = bf16_bits_to_f32(self._bits(a))
        b32 = bf16_bits_to_f32(self._bits(b))
        s = a32 + b32 if op == "add" else a32 - b32                 # one FP32 RNE add
        bits = (s.contiguous().view(torch.int32) >> 16) & 0xFFFF     # raw upper half, no 2nd rounding
        return self._out(bits, a)

    def add(self, a, b):
        return self._addsub("add", a, b)

    def sub(self, a, b):
        return self._addsub("sub", a, b)

    def mul(self, a, b):
        self._check("mul", a, b)
        self._count("mul", a.numel())
        p = bf16_bits_to_f32(self._bits(a)) * bf16_bits_to_f32(self._bits(b))   # exact or FP32-rounded
        return self._out(f32_to_bf16_bits_rne(p), a)

    def _pair(self, op, a, b):
        self._check(op, a, b)
        self._count(op, a.numel())
        ka, kb = ordered_key(self._bits(a)), ordered_key(self._bits(b))
        pick_a = (ka > kb) if op == "pairmax" else (ka < kb)                 # b on tie
        return self._out(key_to_bits(torch.where(pick_a, ka, kb)), a)

    def pairwise_max(self, a, b):
        return self._pair("pairmax", a, b)

    def pairwise_min(self, a, b):
        return self._pair("pairmin", a, b)

    # -- unary pointwise ---------------------------------------------------------------------
    def _unary(self, op: str, a: torch.Tensor) -> torch.Tensor:
        self._count(op, a.numel())
        table = _table_on(op, a.device)
        bits = table[self._bits(a).to(torch.int64)]
        return self._out(bits.to(torch.int32), a)

    def relu(self, a):
        return self._unary("relu", a)

    def rcp(self, a):
        return self._unary("rcp", a)

    def sqrt(self, a):
        return self._unary("sqrt", a)

    def sin(self, a):
        return self._unary("sin", a)

    def cos(self, a):
        return self._unary("cos", a)

    def log2(self, a):
        return self._unary("log", a)

    def tanh(self, a):
        return self._unary("tanh", a)

    def exp(self, a):
        return self._unary("exp", a)

    def exp2(self, a):
        return self._unary("exp2", a)

    def square(self, a):
        return self._unary("square", a)

    def cube(self, a):
        return self._unary("cube", a)

    def mov(self, a):
        return self._unary("mov", a)

    # -- row reductions (a BF16 register pair: 2 x num_lanes = 32 elements per row) ----------
    def _rows(self, op: str, a: torch.Tensor) -> torch.Tensor:
        width = 2 * self.num_lanes
        if a.shape[-1] % width != 0:
            raise ValueError(f"{op}: last dim {a.shape[-1]} must be a multiple of {width}")
        self._count(op, a.numel())
        return self._bits(a).reshape(*a.shape[:-1], a.shape[-1] // width, width)

    def rsum(self, a):
        x = bf16_bits_to_f32(self._rows("rsum", a))
        while x.shape[-1] > 1:                                       # adjacent pairs, FP32 RNE
            x = x[..., 0::2] + x[..., 1::2]
        return self._out(f32_to_bf16_bits_rne(x[..., 0]), a)

    def rmax(self, a):
        k = ordered_key(self._rows("rmax", a))
        return self._out(key_to_bits(k.amax(dim=-1)), a)

    def rmin(self, a):
        k = ordered_key(self._rows("rmin", a))
        return self._out(key_to_bits(k.amin(dim=-1)), a)


def warm(ops=UNARY_OPS) -> dict:
    """Build the unary tables now (about 3 s on a CPU); returns {op: n_entries}."""
    return {op: int(unary_table(op).numel()) for op in ops}


def lut_provenance() -> dict:
    """sha256 of each table built in this process, for result JSONs."""
    import hashlib

    return {op: hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16] for op, t in sorted(_luts.items())}
