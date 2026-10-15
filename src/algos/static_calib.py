"""static_calib: static power-of-two activation scales fixed offline by calibration (NEW track;
docs/algos/static-calib.md).

Every Linear / Conv2d activation gets one exponent per layer chosen from calibration data, in
place of the reference's per-call dynamic rule floor(log2(amax / 256)) (ref_numerics.po2_scale_exp,
a port of pi0-quant hw_float_ops.py:38-57). On the chip this is one `seli` per operand (green card
l.315; atlas-npu ScalarCore.scala:462-464) instead of a VPU amax reduction plus a VMEM round trip
into the scale register (`seld`), and a static cast is counted as vpack only (chip_profile.
count_algo_cast with dynamic=False).

Calibration statistic: the exact histogram of |x| over all 2^15 BF16 magnitude bit patterns, per
layer. Operands are cast to BF16 before the vpack encoder (hw_float_ops.py:89), so the histogram
loses nothing and every candidate exponent's error is evaluated exactly (NEW).

Parameters (`--algo static_calib:k=v,...`):
  method=absmax   absmax: the reference rule applied to the calibration-set amax
                  (FP8 formats paper, arXiv 2209.05433 Sec. 2 / 4.2 "max calibration")
                  mse: the exponent minimising the exact histogram squared error under the vpack
                  cast, searched in [s_absmax - 8, s_absmax + 3] (Kuzmin et al. arXiv 2208.09225
                  Sec. 3.1, 5.2 / App. E: per-tensor exponent bias by MSE)
                  percentile: the reference rule applied to the p-th percentile of |x|
                  (Wu et al. arXiv 2004.09602 Sec. 3.4)
                  kl: TensorRT-style entropy calibration adapted to the po2 FP8 grid (NEW
                  adaptation; Wu et al. Sec. 3.4): source bins merged per FP8 level, each level's
                  mass spread over its non-empty bins
                  (percentile and kl clip LLM massive activations, Sun et al. arXiv 2402.17762
                  Sec. 2.1; kept to reproduce that negative result, see the track doc)
  p=99.99         percentile for method=percentile
  headroom=0      binades added to every calibrated activation exponent (NEW: guards saturation
                  on inputs hotter than the calibration set)
  weight=ref      ref: the reference per-tensor rule (weight_scale returns None);
                  ceil448: the collaborator's ceil(log2(amax / 448)) (atlas_quant.py:53-60);
                  mse: exact histogram MSE under vpack in [s_ref - 3, s_ref + 2]
  p_exp=-8        attention.p exponent (static). Softmax outputs lie in [0, 1], so floor(log2(1/256))
                  = -8 is the reference rule's value at the largest possible amax and never
                  saturates (NEW). p_exp=dynamic keeps the reference per-(batch, head) rule.
  per_step=false  true: one exponent per (layer, call index within one calibration forward), for
                  diffusion / flow models whose layers run once per denoising step (DiT, Pi0).
                  At run time the call index is the layer's call count modulo its calls per
                  forward. Only for method=absmax; layers whose call count differs between
                  calibration batches fall back to the per-layer exponent (NEW).
attention.q / k / v keep the reference's dynamic per-(batch, head) scales (act_scale returns None).
Layers never reached during calibration also keep the dynamic rule (counted in the record).
"""

from __future__ import annotations

import math
from collections import Counter

import torch

import ref_numerics as rn
from algos.base import Algo, Scale, observe_inputs

NBINS = 1 << 15                      # BF16 magnitudes 0x0000 .. 0x7FFF
METHODS = ("absmax", "mse", "percentile", "kl")
WEIGHT_RULES = ("ref", "ceil448", "mse")


# ---------------------------------------------------------------------------------------------
# Histogram statistics (exact for BF16 operands)
# ---------------------------------------------------------------------------------------------

def magnitudes() -> torch.Tensor:
    """float32 value of every BF16 magnitude bit pattern 0..0x7FFF (Inf/NaN patterns included)."""
    return (torch.arange(NBINS, dtype=torch.int32) << 16).view(torch.float32)


def bf16_abs_hist(x: torch.Tensor) -> torch.Tensor:
    """Counts of |x| per BF16 magnitude bit pattern (x cast to BF16 first, as the reference does)."""
    b = rn.bf16_bits(x.detach()).flatten() & 0x7FFF
    return torch.bincount(b.cpu(), minlength=NBINS).to(torch.float64)


def _finite_support(hist):
    v = magnitudes()
    ok = torch.isfinite(v) & (hist > 0)
    return v[ok], hist[ok]


def amax_of(hist) -> float:
    v, _ = _finite_support(hist)
    return float(v.max()) if v.numel() else 0.0


def ref_exp(amax: float) -> int:
    """floor(log2(amax / 256)) of a BF16 magnitude, 0 for 0 (rn.po2_scale_exp on a scalar)."""
    if amax <= 0 or not math.isfinite(amax):
        return 0
    return math.frexp(amax)[1] - 1 - rn.PO2_MAX_EXP


def vpack_roundtrip(v: torch.Tensor, s: int) -> torch.Tensor:
    s_t = torch.tensor(int(s), dtype=torch.int32)
    return rn.e4m3_unpack(rn.e4m3_vpack(v, s_t), s_t)


def sq_err(hist, s: int) -> float:
    v, c = _finite_support(hist)
    q = vpack_roundtrip(v, s).double()
    return float((c * (q - v.double()) ** 2).sum())


def kl_div(hist, s: int) -> float:
    v, c = _finite_support(hist)
    P = c / c.sum()
    q = vpack_roundtrip(v, s)
    _, inv = torch.unique(q, return_inverse=True)
    mass = torch.zeros(int(inv.max()) + 1, dtype=torch.float64).index_add_(0, inv, P)
    n = torch.zeros_like(mass).index_add_(0, inv, torch.ones_like(P))
    Q = mass[inv] / n[inv]
    return float((P * (P / Q).log()).sum())


def percentile_mag(hist, p: float) -> float:
    v, c = _finite_support(hist)
    if not v.numel():
        return 0.0
    cum = torch.cumsum(c, 0)
    i = int(torch.searchsorted(cum, torch.tensor(p / 100.0 * float(cum[-1]), dtype=torch.float64)))
    return float(v[min(i, v.numel() - 1)])


def search(hist, s0: int, fn, lo: int, hi: int) -> int:
    """argmin of fn(hist, s) over s in [s0 + lo, s0 + hi]; ties go to the larger exponent
    (less saturation)."""
    best, best_e = None, None
    for s in range(s0 + lo, s0 + hi + 1):
        e = fn(hist, s)
        if best_e is None or e <= best_e:
            best, best_e = s, e
    return best


def act_exponent(hist, method: str, p: float = 99.99) -> int:
    s_abs = ref_exp(amax_of(hist))
    if method == "absmax":
        return s_abs
    if method == "mse":
        return search(hist, s_abs, sq_err, -8, 3)
    if method == "percentile":
        return ref_exp(percentile_mag(hist, p))
    if method == "kl":
        return search(hist, s_abs, kl_div, -8, 3)
    raise ValueError(f"static_calib: method must be one of {METHODS}, got {method!r}")


def weight_exponent(w: torch.Tensor, rule: str) -> int | None:
    if rule == "ref":
        return None
    if rule == "ceil448":
        from atlas_quant import scale_exp_for          # the collaborator (atlas_quant.py:53-60)
        return int(scale_exp_for(w))
    if rule == "mse":
        return search(bf16_abs_hist(w), int(rn.po2_scale_exp(w)), sq_err, -3, 2)
    raise ValueError(f"static_calib: weight must be one of {WEIGHT_RULES}, got {rule!r}")


# ---------------------------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------------------------

class StaticCalib(Algo):
    name = "static_calib"
    needs_calibration = True

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        self.method = str(p.get("method", "absmax"))
        self.p = float(p.get("p", 99.99))
        self.headroom = int(p.get("headroom", 0))
        self.weight_rule = str(p.get("weight", "ref"))
        pe = p.get("p_exp", -8)
        self.p_exp = None if str(pe) == "dynamic" else int(pe)
        self.per_step = bool(p.get("per_step", False))
        if self.method not in METHODS:
            raise ValueError(f"static_calib: method must be one of {METHODS}, got {self.method!r}")
        if self.weight_rule not in WEIGHT_RULES:
            raise ValueError(f"static_calib: weight must be one of {WEIGHT_RULES}, got {self.weight_rule!r}")
        if self.per_step and self.method != "absmax":
            raise ValueError("static_calib: per_step=true supports method=absmax only")
        if not hasattr(self, "exps"):
            self.exps, self.step_exps, self.calls = {}, {}, Counter()

    # -- (f) calibration ------------------------------------------------------------------------
    def calibrate(self, model, batches, run):
        """One BF16 |x| histogram per Linear / Conv2d input over the calibration set (and, with
        per_step, the amax per call index within one forward)."""
        self.exps, self.step_exps, self.calls = {}, {}, Counter()
        per_batch_calls: list[Counter] = []
        counts: Counter = Counter()

        def reduce(prev, x):
            st = prev or {"hist": torch.zeros(NBINS, dtype=torch.float64), "steps": {}}
            st["hist"] += bf16_abs_hist(x)
            return st

        def reduce_step(prev, x, name):
            st = reduce(prev, x)
            k = counts[name]
            counts[name] += 1
            a = float(x.detach().to(torch.bfloat16).abs().nan_to_num(nan=0.0, posinf=0.0).max())
            st["steps"][k] = max(st["steps"].get(k, 0.0), a)
            return st

        def run_counted(m, b):
            counts.clear()
            run(m, b)
            per_batch_calls.append(Counter(counts))

        if self.per_step:
            stats = _observe_named(model, batches, run_counted, reduce_step)
        else:
            stats = observe_inputs(model, batches, run, reduce=reduce)

        for n, st in stats.items():
            self.exps[n] = act_exponent(st["hist"], self.method, self.p) + self.headroom
        fallback = []
        if self.per_step:
            for n, st in stats.items():
                ns = {c[n] for c in per_batch_calls}
                if len(ns) == 1 and next(iter(ns)) > 1:
                    k_n = next(iter(ns))
                    self.step_exps[n] = [ref_exp(st["steps"][k]) + self.headroom for k in range(k_n)]
                elif len(ns) > 1:
                    fallback.append(n)
        abs_exps = {n: ref_exp(amax_of(st["hist"])) + self.headroom for n, st in stats.items()}
        return {"method": self.method, "p": self.p if self.method == "percentile" else None,
                "headroom": self.headroom, "layers": len(self.exps),
                "exp_minus_absmax": dict(sorted(Counter(self.exps[n] - abs_exps[n] for n in self.exps).items())),
                "exponents": dict(self.exps),
                "per_step_layers": {n: len(v) for n, v in self.step_exps.items()},
                "per_step_fallback": fallback}

    # -- module hooks -------------------------------------------------------------------------
    def weight_scale(self, name, w, kind):
        s = weight_exponent(w, self.weight_rule)
        if s is None:
            return None
        return Scale(torch.tensor(s, dtype=torch.int32, device=w.device), f"per_tensor_{self.weight_rule}",
                     dynamic=False)

    def act_scale(self, name, x, kind, dims=None):
        if kind == "attention.p":
            if self.p_exp is None:
                return None
            return Scale(torch.tensor(self.p_exp, dtype=torch.int32, device=x.device), "static", dynamic=False)
        if kind.startswith("attention."):
            return None                                       # q / k / v: dynamic per (batch, head)
        if name not in self.exps:
            return None                                       # never seen in calibration: dynamic
        steps = self.step_exps.get(name)
        if steps:
            k = self.calls[name] % len(steps)
            self.calls[name] += 1
            return Scale(torch.tensor(steps[k], dtype=torch.int32, device=x.device), "static_per_step",
                         dynamic=False)
        return Scale(torch.tensor(self.exps[name], dtype=torch.int32, device=x.device), "static", dynamic=False)


@torch.no_grad()
def _observe_named(model, batches, run, reduce):
    """observe_inputs with the layer name passed to reduce(prev, x, name) (per-step counting)."""
    import torch.nn as nn

    state, handles = {}, []
    for n, m in model.named_modules():
        if n and isinstance(m, (nn.Linear, nn.Conv2d)):
            def hook(mod, args, n=n):
                if args and torch.is_tensor(args[0]):
                    state[n] = reduce(state.get(n), args[0], n)
            handles.append(m.register_forward_pre_hook(hook))
    try:
        for b in batches:
            run(model, b)
    finally:
        for h in handles:
            h.remove()
    return state


ALGO = StaticCalib
