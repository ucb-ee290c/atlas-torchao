"""Hardware-recipe numerics for SmolVLA QAT: freeze a recipe's FP8 decisions and replay them.

TL;DR: a recipe such as `atlas_v1_awq_tile_static` (the FX export's recipe) decides, once, every
weight code and exponent (AWQ folds, GPTQ rounding, per-32-row panel exponents) and every static
activation exponent (static_calib), keeping the reference dynamic rule only where it says so
(attention q/k/v per (b, h), uncalibrated layers). `bake()` runs the real recipe through
`algos.quantize`, records those decisions, and writes:

  checkpoint/          a plain bf16 lerobot policy whose quantized weights ARE the recipe's
                       dequantized codes (exact in bf16) and whose norms carry the AWQ folds
  checkpoint/hw_state.pt   {"weight": {fqn: (exp, block)}, "act": {(fqn, kind): exp | None}, ...}

`Pinned(state)` is an `algos.base.Algo` that hands those decisions back (no calibration, no GPTQ):
`algos.quantize(policy_from_checkpoint, "ref_conv", algo=Pinned(state))` reproduces the recipe's
model bit for bit (bake() checks this and refuses otherwise). QAT trains through exactly these casts
(smolvla_hw.py) and its trained checkpoint, quantized with the same Pinned state, is what deploys:
the same exponents, codes = the vpack cast of the trained weights.
"""
from __future__ import annotations

import torch

from algos.base import Algo, Scale


def _key(name, kind):
    return f"{name}|{kind}"


class Pinned(Algo):
    """Replays a frozen recipe: pinned weight exponents (and blocks), static activation exponents,
    None (= the reference dynamic rule) where the recipe used it. Weights are taken as given."""

    name = "pinned"
    needs_calibration = False

    def __init__(self, state=None, **params):
        super().__init__(**params)
        if state is not None:
            self.state = state

    def weight_scale(self, name, w, kind):
        rec = self.state["weight"].get(name)
        if rec is None:
            raise KeyError(f"{name}: no pinned weight exponent (layer not in the recipe's model)")
        exp, block = rec
        return Scale(exp.to(w.device), "pinned", block=block, dynamic=False)

    def act_scale(self, name, x, kind, dims=None):
        k = _key(name, kind)
        if k not in self.state["act"]:
            raise KeyError(f"{k}: activation never seen when the recipe was recorded")
        e = self.state["act"][k]
        if e is None:
            return None
        return Scale(torch.tensor(e, dtype=torch.int32, device=x.device), "pinned_static", dynamic=False)


class Recorder:
    """Wraps a real recipe's act_scale / online_transform during forwards and records the decisions."""

    def __init__(self, algo):
        self.algo, self.act, self.bad = algo, {}, []
        self._act, self._ot = algo.act_scale, algo.online_transform

    def __enter__(self):
        def act_scale(name, x, kind, dims=None):
            sc = self._act(name, x, kind, dims)
            k = _key(name, kind)
            if sc is None:
                v = None
            elif sc.dynamic or sc.block or sc.exp.numel() != 1:
                self.bad.append(f"{k}: {sc.describe()} dynamic={sc.dynamic} shape={tuple(sc.exp.shape)}")
                v = "unsupported"
            else:
                v = int(sc.exp.reshape(()))
            if k in self.act and self.act[k] != v:
                self.bad.append(f"{k}: exponent changes between calls ({self.act[k]} vs {v})")
            self.act[k] = v
            return sc

        def online_transform(name, kind):
            f = self._ot(name, kind)
            if f is not None:
                self.bad.append(f"{_key(name, kind)}: online transform (cannot be pinned)")
            return f

        self.algo.act_scale, self.algo.online_transform = act_scale, online_transform
        return self

    def __exit__(self, *exc):
        self.algo.act_scale, self.algo.online_transform = self._act, self._ot
        return False


def weight_state(model):
    """{fqn: (exp int32 tensor on cpu, block)} for every RefQuantLinear / RefQuantConv2d."""
    from reference.ref_attention import RefQuantConv2d
    from reference.ref_profile import RefQuantLinear

    out = {}
    for n, m in model.named_modules():
        if isinstance(m, (RefQuantLinear, RefQuantConv2d)):
            if getattr(m, "residual", None):
                raise ValueError(f"{n}: residual passes cannot be pinned")
            out[n] = (m.weight_scale_exp.detach().to(torch.int32).cpu().clone(), int(m.weight_block or 0))
    return out


@torch.no_grad()
def baked_weights(model):
    """{fqn.weight: dequantized codes in the weight's dtype and shape} (exact; checked)."""
    from algos import base as algo_hooks
    from reference.ref_attention import RefQuantConv2d
    from reference.ref_profile import RefQuantLinear

    out = {}
    for n, m in model.named_modules():
        if isinstance(m, (RefQuantLinear, RefQuantConv2d)):
            wd = algo_hooks.dequant(m.weight_codes, m.weight_scale_exp, m.weight_block).float()
            wb = wd.reshape(m.weight.shape).to(m.weight.dtype)
            if not torch.equal(wb.float().reshape(wd.shape), wd):
                raise RuntimeError(f"{n}: dequantized codes are not exact in {m.weight.dtype}")
            out[f"{n}.weight"] = wb.cpu()
    return out
