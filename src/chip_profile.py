"""The `chip` profile: the whole model as it would run on Atlas.

    chip = reference_hw (every nn.Linear, and conv / attention when the reference profile covers
           them, through the MXU accumulation model)
         + every other op routed by chip_coverage.ChipRouter onto the VPU / scalar core / data
           movement, or listed as host fallback with a reason.

`quantize_chip(model, skip, options)` is called by quant_modes.quantize for mode "chip". It returns
the reference_hw report plus a `coverage` dict that the router keeps up to date in place, so the
result JSONs written by eval.py / eval_policy.py (which store this report) carry the final
coverage; the coverage table is also printed after the first pass and at exit.

The router is active while any entry point of the model runs: forward, and for LeRobot policies
predict_action_chunk / select_action / _generate_actions / sample_actions (eval_policy.py calls
these, not forward). Pre/post-processing pipelines run outside the policy and are not emulated.

Options (argparse or dict): mxu, mxu_impl (reference profiles), chip_rope ("vpu": RoPE tables
computed on the VPU as the reference does for SmolVLA, quant_vector.py:170-171; "host": tables
from *RotaryEmbedding modules precomputed on the host), chip_strict (raise on an unmapped float op).
"""

from __future__ import annotations

import atexit
import functools

import torch
import torch.nn as nn

import chip_coverage as cc
import chip_ops
import ref_numerics as rn

ENTRY_POINTS = ("forward", "predict_action_chunk", "select_action", "_generate_actions",
                "sample_actions", "conditional_sample")
_ALL: list = []


def add_args(ap) -> None:
    g = ap.add_argument_group("chip profile")
    g.add_argument("--chip-rope", choices=("vpu", "host"), default="vpu",
                   help="RoPE tables: vpu (computed on the VPU, the reference treatment) or host "
                        "(*RotaryEmbedding tables precomputed on the host)")
    g.add_argument("--chip-strict", action="store_true", help="raise on an unmapped float op")
    g.add_argument("--chip-mxu", choices=("model", "framework"), default="model",
                   help="model: MXU accumulation through the reference_hw model (--mxu/--mxu-impl); "
                        "framework: FP8 operands, float32 matmul rounded to BF16 (NEW, for fast CPU smokes)")


def _opt(options, key, default):
    if isinstance(options, dict):
        return options.get(key, default)
    return getattr(options, key, default) if options is not None else default


def framework_fm(x, w, b=None, static: bool = False):
    """NEW stand-in for the MXU model (--chip-mxu framework): the exact E4M3 operands multiplied in
    float32 and rounded once to BF16 (the MXU's output format, GC 1: 32 x 32 BF16 accumulators).
    It is not the hardware accumulation order; it exists so whole-model CPU smokes finish. The
    reference profile's own F.linear stand-in (ref_profile.py, NEW) is the analogue followed."""
    y = torch.matmul(x.float(), w.float().transpose(-1, -2))
    return y.to(torch.bfloat16).float()


class _GuardedFM:
    """An MXU accumulation model call: guarded, counted as 2*M*N*K FLOPs."""

    def __init__(self, fm, router, op):
        self.fm, self.router, self.op = fm, router, op

    def __call__(self, x, w, b=None, static: bool = False):
        with cc.guard():
            y = self.fm(x, w, b, static=static) if _takes_static(self.fm) else self.fm(x, w, b)
        if self.router.active:
            self.router.count_mxu(self.op, 2 * y.numel() * x.shape[-1])
        return y

    def __getattr__(self, k):
        return getattr(self.fm, k)


def _takes_static(fm) -> bool:
    import inspect
    try:
        return "static" in inspect.signature(fm.__call__).parameters
    except (TypeError, ValueError):
        return False


class _Session:
    """Enter/exit the router around the model's entry points (re-entrant)."""

    def __init__(self, router):
        self.router, self.depth, self.saved, self.calls = router, 0, {}, 0

    def enter(self):
        if self.depth == 0:
            r = self.router
            orig_q = rn.quantize

            def quantize(x, dims=None):
                with cc.guard():
                    out = orig_q(x, dims)
                n = x.numel()
                r.cov.add("VPU", "fp8_operand_cast", 3 * n,
                          {"vredmax.row.bf16": n, "vredmin.row.bf16": n, "vpack.bf16.fp8": n},
                          reason="activation amax (rmax/rmin chain, hw_float_ops.py:9-12) and vpack "
                                 "(FP8Pack); bit-equal to ref_numerics.quantize")
                return out

            def guarded(f):
                @functools.wraps(f)
                def g(*a, **k):
                    with cc.guard():
                        return f(*a, **k)
                return g

            self.saved = {"quantize": rn.quantize, "e4m3_unpack": rn.e4m3_unpack, "dequantize": rn.dequantize}
            rn.quantize, rn.e4m3_unpack, rn.dequantize = quantize, guarded(rn.e4m3_unpack), guarded(rn.dequantize)
            try:
                import ref_attention
                self.saved["_mm"] = ref_attention._mm
                ref_attention._mm = guarded(ref_attention._mm)
            except ImportError:
                pass
            r.active = True
            r.__enter__()
        self.depth += 1

    def exit(self):
        self.depth -= 1
        if self.depth == 0:
            r = self.router
            r.__exit__(None, None, None)
            r.active = False
            rn.quantize = self.saved["quantize"]
            rn.e4m3_unpack = self.saved["e4m3_unpack"]
            rn.dequantize = self.saved["dequantize"]
            if "_mm" in self.saved:
                import ref_attention
                ref_attention._mm = self.saved["_mm"]
            r.cov.refresh()
            self.calls += 1
            if self.calls == 1:
                print(cc.format_report(r.cov.live), flush=True)


def _wrap_entry(obj, name, session):
    orig = getattr(obj, name, None)
    if orig is None or getattr(orig, "_chip_wrapped", False):
        return False

    @functools.wraps(orig)
    def run(*a, **k):
        session.enter()
        try:
            return orig(*a, **k)
        finally:
            session.exit()

    run._chip_wrapped = True
    setattr(obj, name, run)
    return True


def _set_eager(model) -> list:
    """Attention through its eager path, so scores, mask and softmax are separate ops."""
    changed = []
    for name, m in model.named_modules():
        cfg = getattr(m, "config", None)
        impl = getattr(cfg, "_attn_implementation", None)
        if impl is not None and impl not in ("eager", "atlas_ref"):
            try:
                cfg._attn_implementation = "eager"
                changed.append(name or "<root>")
            except Exception:
                pass
    return changed


def quantize_chip(model: nn.Module, skip=(), options=None) -> dict:
    import quant_modes

    mxu_kind = _opt(options, "chip_mxu", "model") or "model"
    rep = quant_modes.quantize(model, "reference_hw" if mxu_kind == "model" else "reference", skip, options)
    opts = quant_modes.profile_options(options or {})
    rope = _opt(options, "chip_rope", "vpu") or "vpu"
    cov = cc.Coverage(type(model).__name__)
    router = cc.ChipRouter(cov, opts["mxu"], opts["mxu_impl"], rope=rope,
                           strict=bool(_opt(options, "chip_strict", False)))
    router.active = False
    if mxu_kind == "framework":
        router.fm = framework_fm
    chip_ops.warm()

    for name, m in model.named_modules():
        fm = getattr(m, "fm", None)
        if fm is not None and not isinstance(fm, _GuardedFM):
            m.fm = _GuardedFM(fm, router, "conv2d" if "Conv" in type(m).__name__ else "linear")
        cfg = getattr(m, "_atlas_ref_attn", None)
        if isinstance(cfg, dict):
            for k in ("fm_qk", "fm_av"):
                if cfg.get(k) is not None and not isinstance(cfg[k], _GuardedFM):
                    cfg[k] = _GuardedFM(cfg[k], router, "attention")

    def ctx_hooks(mod, kind, detail):
        mod.register_forward_pre_hook(lambda *_: router.push(kind, detail))
        mod.register_forward_hook(lambda *_: router.pop())

    n_skip = 0
    for name, m in model.named_modules():
        s = next((s for s in skip if s and s in name), None)
        if s is not None and name:
            ctx_hooks(m, "skip", s)
            n_skip += 1
        elif rope == "host" and type(m).__name__.endswith("RotaryEmbedding"):
            ctx_hooks(m, "rope_host", type(m).__name__)
        elif type(m).__name__ in ("RefQuantLinear", "RefQuantConv2d"):
            ctx_hooks(m, "mxu_module", type(m).__name__)

    eager = _set_eager(model)
    session = _Session(router)
    entries = [n for n in ENTRY_POINTS if _wrap_entry(model, n, session)]
    _ALL.append(cov)

    rep["profile"] = {**rep.get("profile", {}), "name": "chip",
                      "base": "reference_hw" if mxu_kind == "model" else "reference (framework accumulation)",
                      "mxu": mxu_kind,
                      "vpu": "chip_ops.ChipVPU (bit-exact to VectorRTLFunctions, 16 lanes)",
                      "rope": rope, "entry_points": entries, "eager_attention_set": eager,
                      "skipped_to_host": list(skip), "skip_modules": n_skip,
                      "vpu_lut_sha256_16": chip_ops.lut_provenance()}
    import ref_loader
    rep["reference_sources"] = ref_loader.provenance()
    rep["coverage"] = cov.refresh()
    return rep


@atexit.register
def _print_all():
    for cov in _ALL:
        if cov.rows:
            print(cc.format_report(cov.refresh()), flush=True)
