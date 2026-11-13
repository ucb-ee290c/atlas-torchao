"""Two-term FP8 activations (opt-in, eval time): x ~ dq(Q(x)) + dq(Q(x - dq(Q(x)))).

Each term is its own per-token power-of-two cast, as the layer's own activation rule makes it
(algos.base.act_cast with the layer's algorithm). The pair represents a BF16 activation to about
8 mantissa bits, which removes the activation half of the FP8 layer error (docs/algos/two-term.md).
On Atlas the second MXU pass reuses the resident weight tile (two accumulators, two weight slots), so
weights are loaded once; only MXU passes and a little VPU work double (hw_cost.two_term_layer_cost).

A wrapped RefQuantLinear (reference, reference_hw, chip, chip_rope_vpu) computes
    x1 = dq(Q(x))                    (on the FP8 grid, so F re-quantizes it exactly)
    r  = x - x1                      (BF16 vsub, truncating like the VPU)
    y  = F(x1) + F_nobias(r)         (F = the layer's existing forward; the bias is in the first term only;
                                      the sum is one BF16 vadd)
Weights are never two-term. Attention matmuls (unless the attn choices are given, see ref_attention two_term) and Conv2d are not touched.

Hooks: eval.py / eval_policy.py call `apply(model, mode, option)` after the quantized model is built.
The forward is replaced on the module instance, so module names, hooks and state are unchanged.
"""

from __future__ import annotations

import os

import torch

import chip.chip_ops as chip_ops
import reference.ref_numerics as rn
from algos import base as algo_hooks

CHOICES = ("off", "all", "down", "attnq", "attn", "all+attn", "all+attnq")
ATTN_TWO_TERM = {"attnq": "q", "attn": "qp", "all+attn": "qp", "all+attnq": "q"}     # option -> ref_attention two_term
LINEAR_CHOICES = ("all", "down", "all+attn", "all+attnq")
_VPU = chip_ops.ChipVPU()                 # bit-exact vadd/vsub (truncating upper-16-bit slice), uncounted


def add_args(ap) -> None:
    g = ap.add_argument_group("hardware-aware two-term activations (NEW, opt-in)")
    g.add_argument("--act-two-term", choices=CHOICES, default="off",
                   help="off (default): unchanged. all: every quantized Linear runs F(x1)+F(x-x1) with x1 the "
                        "FP8 round trip of x (bias once, BF16 vsub/vadd); down: down_proj layers only. "
                        "attnq: attention only, S = fm(Q1,K)+fm(Q2,K); attn: attention only, Q and P two-term "
                        "(O = fm(P1,V)+fm(P2,V) too); all+attn: all Linears plus attention Q and P. K, V, "
                        "weights stay single-term; Conv2d is never touched")


def _router():
    """The open chip router (chip profile, inside a model call), else None."""
    cp = __import__("sys").modules.get("chip.chip_profile")
    return cp._RUNNING[-1] if cp is not None and cp._RUNNING else None


def vsub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """BF16 a - b as the VPU does it. Under the chip router: its VPU (counted) with the active vadd mode."""
    return _vop(a, b, True)


def vadd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _vop(a, b, False)


def _vop(a, b, sub):
    r = _router()
    if r is None:
        if os.environ.get("ATLAS_TWO_TERM_ADD", "trunc") == "rne":
            # round-to-nearest-even BF16 add: what an RNE vadd (the requested RTL change, or the 24-instruction
            # sequence) gives; the default models the truncating vadd the RTL has today
            return ((a.float() - b.float()) if sub else (a.float() + b.float())).to(a.dtype)
        return _VPU.sub(a, b) if sub else _VPU.add(a, b)
    import chip.chip_coverage as cc
    import chip.chip_numerics as chip_numerics

    with cc.guard():                       # explicit VPU call, not re-dispatched through the router
        if r.vadd is not None:
            r.vadd.note(a.numel()) if r.vadd.active else None
        return chip_numerics.add(r.vadd.mode if r.vadd is not None else "trunc", r.vpu, a, b, sub=sub)


def act_roundtrip(layer, x: torch.Tensor) -> torch.Tensor:
    """x1 = dq(Q(x)) with the layer's own activation cast (BF16 tensor, exact)."""
    q, s, blk = algo_hooks.act_cast(layer.algo, layer.layer_name, x, "linear")
    xd = algo_hooks.dequant(q, s, blk) if blk else rn.dequantize(q, s)
    return xd.to(x.dtype)


def two_term_forward(layer, orig, x: torch.Tensor) -> torch.Tensor:
    dtype = layer.weight.dtype
    x = x.to(dtype)
    x1 = act_roundtrip(layer, x)
    r = vsub(x, x1)
    y1 = orig(x1)                          # includes the bias
    bias, layer.bias = layer.bias, None
    try:
        y2 = orig(r)
    finally:
        layer.bias = bias
    return vadd(y1, y2)


def wrap_layer(layer) -> None:
    if getattr(layer, "_two_term_orig", None) is not None:
        return
    if layer.algo is not None and layer.algo.online_transform(layer.layer_name, "linear") is not None:
        raise NotImplementedError(f"{layer.layer_name}: an online activation transform does not commute with "
                                  "the two-term split")
    orig = layer.forward
    layer._two_term_orig = orig
    layer.forward = lambda x, _l=layer, _o=orig: two_term_forward(_l, _o, x)


def unwrap(model) -> int:
    n = 0
    for m in model.modules():
        o = getattr(m, "_two_term_orig", None)
        if o is not None:
            del m.forward
            m._two_term_orig = None
            n += 1
    return n


def apply(model, mode: str, option: str = "off") -> dict | None:
    """Wrap the quantized Linears of `model` (RefQuantLinear: reference / reference_hw / chip profiles).
    Returns the record for the result JSON, or None when the option is off or `mode` is bf16."""
    if option not in CHOICES:
        raise ValueError(f"--act-two-term {option!r}: choose from {CHOICES}")
    if option == "off" or mode == "bf16":
        return None
    attn_sites = []
    if option in ATTN_TWO_TERM:
        for name, m in model.named_modules():
            cfg = getattr(m, "_atlas_ref_attn", None)
            if isinstance(cfg, dict):
                if cfg.get("algo") is not None and cfg["algo"].online_transform(cfg.get("site", name),
                                                                                "attention.q") is not None:
                    raise NotImplementedError(f"{name}: an online attention transform does not commute with "
                                              "the two-term split")
                cfg["two_term"] = ATTN_TWO_TERM[option]
                attn_sites.append(name)
        if not attn_sites and option not in ("all+attn", "all+attnq"):
            raise RuntimeError(f"--act-two-term {option}: no patched attention site in mode {mode!r} "
                               "(profile without --ops attention?)")
    wrapped, skipped = [], []
    for name, m in model.named_modules():
        if option not in LINEAR_CHOICES:
            break
        if type(m).__name__ != "RefQuantLinear":
            continue
        if option == "down" and not name.endswith("down_proj"):
            continue
        if m.weight.dtype != torch.bfloat16:
            skipped.append(name)               # the VPU subtract is a BF16 op; fp32 layers stay single-term
            continue
        wrap_layer(m)
        wrapped.append(name)
    if not wrapped and option in LINEAR_CHOICES:
        raise RuntimeError(f"--act-two-term {option}: no quantized Linear found to wrap in mode {mode!r} "
                           "(profile without Linear quantization?)")
    return {"act_two_term": option, "mode": mode, "wrapped_layers": len(wrapped), "skipped_non_bf16": skipped,
            "attention_sites": len(attn_sites),
            "attention_two_term": ATTN_TWO_TERM.get(option),
            "semantics": "Linear: y = F(x1) + F_nobias(x - x1), x1 = dq(Q(x)); attention: S = fm(Q1,K)+fm(Q2,K) "
                         "[and O = fm(P1,V)+fm(P2,V) for attn/all+attn], x1 = dq(Q(x)) per (b,h), x2 = Q(x - x1); "
                         "BF16 vsub/vadd (truncating; chip: router VPU, active vadd mode); bias once; K, V, "
                         "weights single-term; conv untouched"}
