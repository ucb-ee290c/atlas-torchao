"""Hook API for FP8 quantization algorithms (NEW; docs/algos/README.md).

An algorithm is a subclass of `Algo` that overrides some of the hooks below. Every hook's default
is a no-op, and with `--algo none` no `Algo` object exists at all: the quantized modules see
`algo is None` and call exactly the code of l40s-run-2.1 (tests/test_algo_base.py checks this bit
for bit on every ladder step).

Model-level hooks (applied by algos.quantize to every mode except bf16):
  bf16_layers(model, mode)            (e) fqns kept in BF16; passed to the profile as extra skip
                                      entries, so the chip coverage report counts them as host
                                      ("kept off the chip on request") and the algo record lists them
  calibrate(model, batches, run)      (f) on the unquantized model; run(model, batch) is one forward
  prepare(model, mode)                (d, folded) in-place transforms folded into weights / norms
                                      (e.g. smoothing, rotations) before any layer is quantized

Module-level hooks (called by the reference-numerics modules, i.e. ref_linear .. chip_rope_vpu:
RefQuantLinear, RefQuantConv2d and the reference attention routine; collab_v0 and torchao
receive the model-level hooks only):
  transform_weight(name, w, kind)     (a) the weight (2-D [out, in]; Conv2d flattened) right
                                      before it is cast to FP8; returns the tensor to quantize
  weight_scale(name, w, kind)         (b) None = the reference per-tensor rule, or a Scale
                                      (per_tensor, per_channel [out, 1], per_block along in)
  act_scale(name, x, kind, dims)      (c) None = dynamic per-tensor (the reference rule), or a
                                      Scale (static from calibration, per-token, per-block, ...);
                                      `dims` are the reference's reduction dims (None = the whole
                                      tensor; attention operands (-2, -1), i.e. per (batch, head))
  online_transform(name, kind)        (d, online) None, or a callable x -> x' applied at run time
                                      before the activation cast (its weight-side counterpart is
                                      applied in transform_weight)
  residual_passes(name, kind)         (g, proposed on algo/mixed-precision) None, or "w" / "wx":
                                      extra FP8 MXU passes on the residual of the weight ("w":
                                      x_hi.w_hi + x_hi.w_lo) or of both operands ("wx": + x_lo.w_hi),
                                      each residual cast with the reference rule (Linear only)

`kind` is "linear", "conv2d" or "attention.q|k|p|v" (attention operands; `name` is the site).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import torch


@dataclass
class Scale:
    """A power-of-two scale choice: real value = e4m3(code) * 2^exp.

    exp          int32 tensor, broadcastable against the operand (per_block: against the operand
                 viewed as [..., n_blocks, block])
    granularity  descriptor recorded in the results: per_tensor | per_channel | per_token |
                 per_block | static | any other short string the track documents
    block        block length along the last (reduction) dim for per_block, else 0
    dynamic      True if computed from the operand at run time (counts an amax on the VPU)
    """
    exp: torch.Tensor
    granularity: str = "per_tensor"
    block: int = 0
    dynamic: bool = True

    def describe(self) -> str:
        return f"{self.granularity}(block={self.block})" if self.block else self.granularity


class Algo:
    """Base algorithm: every hook is a no-op. Subclasses set `name` and override hooks."""

    name = "base"
    needs_calibration = False
    # Hooks a subclass actually overrides that act inside the reference modules; recorded so a
    # result under collab_v0 / torchao says they did not apply.
    module_hooks = ("transform_weight", "weight_scale", "act_scale", "online_transform", "residual_passes")

    def __init__(self, **params):
        self.params = dict(params)
        self.stats = {"weight": Counter(), "activation": Counter(), "online_transforms": Counter()}
        self.calibration = None          # {source, n_samples, seed, ...} once calibrated
        self._seen = set()

    def note(self, category, descriptor, name) -> None:
        """Count `name` once under stats[category][descriptor] (layers / sites, not calls)."""
        if (category, descriptor, name) not in self._seen:
            self._seen.add((category, descriptor, name))
            self.stats[category][descriptor] += 1

    # -- model level -------------------------------------------------------------------------
    def bf16_layers(self, model, mode) -> list:
        return []

    def calibrate(self, model, batches, run) -> dict | None:
        return None

    def prepare(self, model, mode) -> dict | None:
        return None

    # -- module level --------------------------------------------------------------------------
    def transform_weight(self, name, w, kind):
        return w

    def weight_scale(self, name, w, kind) -> Scale | None:
        return None

    def act_scale(self, name, x, kind, dims=None) -> Scale | None:
        return None

    def online_transform(self, name, kind):
        return None

    def residual_passes(self, name, kind) -> str | None:
        return None

    # -- bookkeeping ---------------------------------------------------------------------------
    def overridden_module_hooks(self) -> list:
        return [h for h in self.module_hooks if getattr(type(self), h) is not getattr(Algo, h)]

    def record(self) -> dict:
        return {"name": self.name, "params": dict(self.params), "calibration": self.calibration,
                # live Counters: activation entries fill in during the evaluation's forwards
                "granularity": dict(self.stats)}


# ---------------------------------------------------------------------------------------------
# The casts the reference modules call. With algo None they are exactly ref_numerics.quantize /
# dequantize (looked up at call time, so the chip profile's instrumented rn.quantize is used).
# ---------------------------------------------------------------------------------------------

def _rn():
    import ref_numerics as rn
    return rn


def _guard():
    """The chip router's guard (operand casts are not model ops); a no-op outside `chip`."""
    import contextlib
    import sys

    cc = sys.modules.get("chip_coverage")                  # loaded only by the chip profiles
    return cc.guard() if cc is not None else contextlib.nullcontext()


def _count_cast(x, dynamic):
    import sys

    chip_profile = sys.modules.get("chip_profile")       # loaded only by the chip profiles
    if chip_profile is not None:
        chip_profile.count_algo_cast(x.numel(), dynamic)


def cast_with(x: torch.Tensor, scale: Scale):
    """BF16 -> E4M3 bytes with an explicit Scale (vpack encoder, as ref_numerics.quantize)."""
    rn = _rn()
    with _guard():
        xb = x.to(torch.bfloat16)
        if scale.block:
            q = rn.e4m3_vpack(xb.unflatten(-1, (-1, scale.block)), scale.exp).flatten(-2)
        else:
            q = rn.e4m3_vpack(xb, scale.exp)
    _count_cast(x, scale.dynamic)
    return q, scale.exp


def dequant(q, exp, block=0):
    rn = _rn()
    if not block:
        return rn.dequantize(q, exp)
    with _guard():
        return (rn.e4m3_unpack(q).unflatten(-1, (-1, block)) * torch.exp2(exp.float())).flatten(-2)


def weight_cast(algo, name, w, kind):
    """(codes, exp, block) of a weight. algo None: rn.quantize(w) (the reference rule)."""
    if algo is None:
        q, s = _rn().quantize(w)
        return q, s, 0
    w = algo.transform_weight(name, w, kind)
    sc = algo.weight_scale(name, w, kind)
    if sc is None:
        q, s = _rn().quantize(w)
        algo.note("weight", f"{kind}:per_tensor", name)
        return q, s, 0
    algo.note("weight", f"{kind}:{sc.describe()}", name)
    q, s = cast_with(w, sc)
    return q, s, sc.block


def act_cast(algo, name, x, kind, dims=None):
    """(codes, exp, block) of an activation. algo None: rn.quantize(x, dims)."""
    if algo is None:
        q, s = _rn().quantize(x, dims)
        return q, s, 0
    f = algo.online_transform(name, kind)
    if f is not None:
        x = f(x)
        algo.note("online_transforms", kind, name)
    with _guard():                       # scale selection is part of the cast (counted by cast_with)
        sc = algo.act_scale(name, x, kind, dims)
    if sc is None:
        q, s = _rn().quantize(x, dims)
        algo.note("activation", f"{kind}:dynamic " + ("per_tensor" if dims is None else "per_head"), name)
        return q, s, 0
    algo.note("activation", f"{kind}:{sc.describe()}", name)
    q, s = cast_with(x, sc)
    return q, s, sc.block


def output_exp(s_x, s_w):
    """The 2^(sx+sw) exponent on the [..., out] matmul output: a per-channel weight exponent
    [out, 1] becomes [out]. Per-tensor exponents are returned unchanged."""
    if s_w.dim() == 2 and s_w.shape[-1] == 1:
        s_w = s_w.reshape(-1)
    return s_x + s_w


def output_scale(s_x, s_w):
    """2^(sx+sw) for the BF16 output rescale (quant_linear.py:169-170).

    Per-tensor exponents on both sides: computed exactly as the reference modules always did (so
    the chip router counts it as before). A per-row / per-column exponent table (NEW, e.g. the
    granularity track): the [..., out] table is built under the chip guard, because on Atlas the
    factors come from the scale derivation and are applied as vmul.bf16 on the output, not as a
    vexp2 over every output element; the algorithm that chose such scales counts those vmuls
    itself (chip_profile.count_algo_vpu)."""
    if s_x.numel() == 1 and s_w.numel() == 1:
        return torch.exp2(output_exp(s_x, s_w).float())
    with _guard():
        return torch.exp2(output_exp(s_x, s_w).float())


# ---------------------------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------------------------

@torch.no_grad()
def observe_inputs(model, batches, run, select=None, reduce=None):
    """Run `batches` through `model` and reduce each selected module's input activation.

    select(name, module) -> bool (default: nn.Linear and nn.Conv2d); reduce(prev, x) -> state
    (default: running BF16 amax as float). Returns {fqn: state}."""
    import torch.nn as nn

    select = select or (lambda n, m: isinstance(m, (nn.Linear, nn.Conv2d)))
    reduce = reduce or (lambda prev, x: torch.maximum(prev, x.detach().to(torch.bfloat16).abs().amax().float())
                        if prev is not None else x.detach().to(torch.bfloat16).abs().amax().float())
    state, handles = {}, []
    for n, m in model.named_modules():
        if n and select(n, m):
            def hook(mod, args, n=n):
                if args and torch.is_tensor(args[0]):
                    state[n] = reduce(state.get(n), args[0])
            handles.append(m.register_forward_pre_hook(hook))
    try:
        for b in batches:
            run(model, b)
    finally:
        for h in handles:
            h.remove()
    return state
