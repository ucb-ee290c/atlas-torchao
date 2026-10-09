"""MXU accumulation backends for the reference_hw profile, selected by name: `--mxu` x implementation.

    mxu   ipt   inner-product tree (the reference author's choice: smolVLA run_eval.py:50 runs ipt_numba)
          sa    per-step systolic array (functional_models.py:103-108, systolic_c)
    impl  ref_cpu    the reference author's CPU functional models, used as-is (the oracles)
          ref_cpu_c  ipt only: the C build of the IPT model (ipt_c), for cross-checking
          torch      device-agnostic torch ports in src/accum_torch.py (GPU-capable)

Every backend is a callable with the reference functional-model call contract
(pi0_inout/functional_models.py:7-12, 79-108; quant_linear.py:166):

    y = fm(x, w, b=None)

Contract that src/accum_torch.py must implement, with these exact signatures:

    def ipt_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None, *,
                   vec_len: int = 32) -> torch.Tensor
    def sa_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None, *,
                  rows: int = 32) -> torch.Tensor

  x    float32, any device. Every element is 0 or an exact finite normal E4M3 value, i.e. the
       vpack encoder's bytes decoded with `.view(torch.float8_e4m3fn).to(torch.float32)`
       (quant_linear.py:161-162). It never holds 0x7F/0xFF (480) or E4M3 subnormals. The
       reference models re-encode it with their own encoder, which is idempotent on such values.
  w    float32, same value domain. Either 2-D [N, K], with x [*, K] -> y [*, N] (the reference:
       x is flattened to rows, ipt_rtl_linear.py:215-219), or batched [*B, N, K] with
       x [*B, M, K] -> y [*B, M, N], equal to applying the 2-D form per batch index (used by the
       per-head attention path to avoid a Python loop over heads).
  b    None in this repo: the reference profile adds a BF16 bias after the rescale
       (quant_linear.py:164-172). An implementation may raise NotImplementedError for a tensor.
  y    float32 on x.device; every element is the BF16 value the reference returns with
       OutBF16 output (decode_model_output_bits, ipt_rtl_linear.py:86-90; the C models'
       uint16 bits << 16). Must not depend on lane count, output tiling or chunking.

  Bit-exact oracles:
    ipt  funct_models_ipt.ipt_numba.ipt_rtl_linear.IPTLinearRTLFunction(vec_len=32, num_lanes=32)
         (_numba_kernels.py:234-317). Its aligned-integer -> BF16 step is one RNE rounding
         (_numba_kernels.py:156-167), as in the RTL: atlas-npu 0079c05 FPUtils.scala:228-236
         uses INToRecFN with round_near_even. ipt_c instead goes through (float)int and then
         float -> BF16 (ipt_c/converters.h:244-254), a double rounding that can differ when the
         32-bit total is >= 2^24 and lands on a BF16 tie after the first rounding. The ipt_numba
         semantics are the target; a case where ipt_c differs is a finding, not a port bug.
    sa   func_models_sa.systolic_c.systolic_array_rtl_linear.SARTLLinearFunction(rows=32, cols=32):
         one E4M3ProdAddBF16 per K element (systolic_array_model.h:97-233, 431-476); K tiles chain
         through the BF16 psum (systolic_array_linear.h:159-206); a tail tile is zero-padded.
"""

from __future__ import annotations

from typing import Callable

import torch

import ref_loader

MXUS = ("ipt", "sa")
IMPLS = ("ref_cpu", "ref_cpu_c", "torch")
DEFAULT_MXU = "ipt"      # the reference author's choice (smolVLA-quant experiments/run_eval.py:50)

Accumulator = Callable[..., torch.Tensor]


class _RefCPU:
    """Wraps one reference functional-model instance; returns float32 on the input's device.

    The reference models cache the encoded weight keyed on (data_ptr, shape, stride, dtype,
    _version) (ipt_rtl_linear.py:123-145, ipt_rtl_linear_c.py:492-499, 546-567). A temporary that
    is freed and re-allocated at the same address with the same shape would hit that cache with
    different contents, which can happen to the per-head V^T operand in the reference attention
    (smolVLA model_patcher.py:696). So `static=False` (the default) drops the cache before every
    call; the linear layers pass `static=True` for their persistent normalized weights.
    """

    def __init__(self, mxu: str, impl: str):
        self.mxu, self.impl = mxu, impl
        if mxu == "ipt" and impl == "ref_cpu":
            m = ref_loader.import_ref("funct_models_ipt.ipt_numba.ipt_rtl_linear")
            self.fn = m.IPTLinearRTLFunction(vec_len=32, num_lanes=32)   # ipt_rtl_linear.py:105-110
        elif mxu == "ipt" and impl == "ref_cpu_c":
            m = ref_loader.import_ref("funct_models_ipt.ipt_c.ipt_rtl_linear_c")
            # 32 lanes as ipt_numba (ipt_rtl_linear.py:107-108); the C default is 16
            # (ipt_rtl_linear_c.py:529-530). Lanes are independent, so the count cannot change
            # results (tests/test_accum_backends.py checks 16 == 32).
            self.fn = m.CIPTLinearRTLFunction(vec_len=32, num_lanes=32)
        elif mxu == "sa" and impl == "ref_cpu":
            m = ref_loader.import_ref("func_models_sa.systolic_c.systolic_array_rtl_linear")
            self.fn = m.SARTLLinearFunction(rows=32, cols=32)             # functional_models.py:103-105
        else:
            raise ValueError(f"no {impl!r} implementation for mxu {mxu!r}")

    def _drop_cache(self):
        for attr in ("_prepared_cache", "_w_cache_key", "_b_cache_key"):
            if hasattr(self.fn, attr):
                setattr(self.fn, attr, None)

    def __call__(self, x, w, b=None, static: bool = False):
        if w.dim() > 2:                                   # batched form: one reference call per index
            lead = w.shape[:-2]
            xs, ws = x.reshape(-1, *x.shape[-2:]), w.reshape(-1, *w.shape[-2:])
            ys = [self(xi, wi, b) for xi, wi in zip(xs, ws)]
            return torch.stack(ys).reshape(*lead, x.shape[-2], w.shape[-2])
        if not static:
            self._drop_cache()
        y = self.fn(x.detach().float().cpu(), w.detach().float().cpu(),
                    None if b is None else b.detach().float().cpu())
        return y.to(device=x.device, dtype=torch.float32)


class _Torch:
    """src/accum_torch.py. ipt follows ipt_numba (variant="numba", the reference author's run and
    the RTL's single rounding); sa can torch.compile its step (compile=None reads
    ATLAS_ACCUM_COMPILE). Pure integer bit-field arithmetic: TF32 and reduction-order settings
    cannot change its results."""

    def __init__(self, mxu: str, compile=None):
        try:
            import accum_torch
        except ImportError as e:
            raise RuntimeError("the torch accumulation implementation (src/accum_torch.py) is not "
                               "available in this checkout; use --mxu-impl ref_cpu") from e
        self.mxu, self.compile = mxu, compile
        self._at = accum_torch

    def __call__(self, x, w, b=None, static: bool = False):
        if self.mxu == "ipt":
            return self._at.ipt_linear(x, w, b, variant="numba")
        return self._at.sa_linear(x, w, b, compile=self.compile)


def get_accumulator(mxu: str = DEFAULT_MXU, impl: str = "ref_cpu", compile=None) -> Accumulator:
    """A callable fm(x, w, b=None, static=False) for the given MXU model and implementation.
    `compile` only affects the torch SA port."""
    if mxu not in MXUS:
        raise ValueError(f"unknown mxu {mxu!r}; choose from {MXUS}")
    if impl not in IMPLS:
        raise ValueError(f"unknown implementation {impl!r}; choose from {IMPLS}")
    if impl == "torch":
        return _Torch(mxu, compile)
    return _RefCPU(mxu, impl)


def describe(mxu: str, impl: str) -> dict:
    """What ran, for result JSONs."""
    oracle = {("ipt", "ref_cpu"): "funct_models_ipt.ipt_numba IPTLinearRTLFunction(32, 32)",
              ("ipt", "ref_cpu_c"): "funct_models_ipt.ipt_c CIPTLinearRTLFunction(32, 32)",
              ("sa", "ref_cpu"): "func_models_sa.systolic_c SARTLLinearFunction(32, 32)"}
    return {"mxu": mxu, "impl": impl,
            "model": oracle.get((mxu, impl), "accum_torch.ipt_linear(variant='numba')" if mxu == "ipt"
                                else "accum_torch.sa_linear")}


def crosschecked(mxu: str, fm, sample_frac: float = 0.01, seed: int = 0, time_budget_s: float = 600.0):
    """Wrap a torch accumulator so a seeded sample of real calls is recomputed by the reference
    CPU model and compared bit for bit (src/accum_crosscheck.py). One instance is shared by every
    layer so the time budget is global. The reference is always called with its weight cache
    dropped: the operands it receives are fresh CPU copies, whose addresses can repeat."""
    from accum_crosscheck import CrossChecked

    ref = get_accumulator(mxu, "ref_cpu")
    return CrossChecked(fm, lambda x, w, b=None: ref(x, w, b, static=False), mxu=mxu, impl="torch",
                        ref_impl="ref_cpu", sample_frac=sample_frac, seed=seed,
                        time_budget_s=time_budget_s)
