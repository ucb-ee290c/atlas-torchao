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
computed on the VPU as the reference does for SmolVLA, quant_vector.py:170-171, also the
chip_rope_vpu profile; "host", the default (NEW): tables
from *RotaryEmbedding modules, and the sinusoidal timestep table of flow-matching policies
(`create_sinusoidal_pos_embedding` in the model's module, NEW), precomputed on the host),
chip_strict (raise on an unmapped float op).
"""

from __future__ import annotations

import atexit
import functools

import torch
import torch.nn as nn

import chip_coverage as cc
import chip_numerics
import chip_ops
import hw_scale
import ref_numerics as rn

ENTRY_POINTS = ("forward", "predict_action_chunk", "select_action", "_generate_actions",
                "sample_actions", "conditional_sample")
_ALL: list = []
_RUNNING: list = []      # routers whose session is open (innermost last)


def count_algo_cast(n: int, dynamic: bool) -> None:
    """An FP8 operand cast with an algorithm's own scale (src/algos/base.cast_with), counted like
    the reference cast: vpack, plus the amax chain when the scale is computed at run time."""
    if not _RUNNING:
        return
    instr = {"vpack.bf16.fp8": n}
    if dynamic:
        instr.update({"vredmax.row.bf16": n, "vredmin.row.bf16": n})
    _RUNNING[-1].cov.add("VPU", "fp8_operand_cast", sum(instr.values()), instr,
                         reason="algorithm operand cast (src/algos): vpack with the algorithm's scale"
                                + ("; amax at run time" if dynamic else "; scale fixed ahead of time"))


class _ChainCoverage(cc.Coverage):
    """Coverage that also carries the NEW dynamic-scale chain totals (count_dynamic_scale_chain)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.chain: dict = {}

    def report(self) -> dict:
        rep = super().report()
        if self.chain:
            rep["dynamic_scale_chain_NEW"] = {
                mode: {key: (dict(v) if isinstance(v, dict) else v) for key, v in tot.items()}
                for mode, tot in self.chain.items()}
            rep["dynamic_scale_chain_NEW_note"] = hw_scale.COST_ASSUMPTIONS
        return rep


def count_dynamic_scale_chain(rows: int, k: int, mode: str, rule: str = "nc") -> None:
    """NEW, additive to count_algo_cast (whose vpack + vredmax.row + vredmin.row counts stay as they were):
    the real on-chip chain that derives a dynamic scale (hw_scale.dynamic_scale_cost): vector amax,
    vstore, scalar RV32I exponent sequence, sb, seld, and for mode 'token' the pre-vpack 2^-s factor
    path. mode 'block' = one scale per 32-token block over the full K (granularity a=block); 'token' =
    one per token. Rows: VPU `dyn_scale_chain` and scalar `dyn_scale_chain_scalar` (instructions);
    serialized cycles go in the report's `dynamic_scale_chain_NEW` section (no overlap claimed)."""
    if not _RUNNING:
        return
    cov = _RUNNING[-1].cov
    c = hw_scale.dynamic_scale_cost(rows, k, mode, rule)
    cov.add("VPU", "dyn_scale_chain", sum(c["vpu"].values()), c["vpu"],
            reason=f"NEW: real dynamic-scale chain, mode={mode} (vredmax/vredmin, combine, vstore"
                   + (", pre-vpack vmul" if mode == "token" else "") + "); H2 closed by hw_scale",
            new={"choice": "vredmax.row/vredmin.row -> vstore -> scalar lhu + RV32I exponent -> sb -> seld",
                 "analogue": "RTL has no exponent-extract instruction (handoff/HW_AWARE_AUDIT.md section 2)",
                 "reason": "honest cost of dynamic scales; additive to the 3-instruction amax+vpack count"})
    sc = {"rv32i": c["scalar"]["rv32i"], "seld": c["seld"]}
    cov.add("scalar", "dyn_scale_chain_scalar", sum(sc.values()), sc,
            reason="NEW: scalar-core RV32I instructions (lhu, exponent sequence, sb) and seld of the chain")
    tot = cov.chain.setdefault(mode, {"scales_blocks": 0, "vpu": {}, "rv32i": 0, "seld": 0,
                                      "instructions": 0, "serialized_cycles": 0})
    tot["scales_blocks"] += c["blocks"]
    for key, v in c["vpu"].items():
        tot["vpu"][key] = tot["vpu"].get(key, 0) + v
    tot["rv32i"] += c["scalar"]["rv32i"]
    tot["seld"] += c["seld"]
    tot["instructions"] += c["instructions"]
    tot["serialized_cycles"] += c["cycles"]


def count_algo_vpu(op: str, instrs: dict, reason: str, new: dict | None = None) -> None:
    """Extra VPU instructions an algorithm's numerics need on the chip (src/algos), e.g. the
    per-token pre-vpack vmul of the granularity track. Counted on its own VPU row `op`; `new`
    ({choice, analogue, reason}) lists it under the report's new_decisions."""
    if not _RUNNING or not instrs:
        return
    _RUNNING[-1].cov.add("VPU", op, sum(instrs.values()), instrs, reason=reason, new=new)


def add_args(ap) -> None:
    g = ap.add_argument_group("chip profile")
    g.add_argument("--chip-rope", choices=("vpu", "host"), default=None,
                   help="RoPE and timestep tables for `chip`: host (default, NEW: *RotaryEmbedding and "
                        "sinusoidal timestep tables are compile-time host constants) or vpu (computed "
                        "on the VPU, the reference treatment; also the chip_rope_vpu profile)")
    g.add_argument("--chip-strict", action="store_true", help="raise on an unmapped float op")
    g.add_argument("--chip-native", nargs="*", default=[],
                   help="aten ops (overload-packet names) run on the host instead, for ablations; "
                        "they are reported as host fallback")
    g.add_argument("--chip-vadd", choices=chip_numerics.MODES, default=None,
                   help="vadd/vsub numerics: trunc (the hardware as built: FP32 add, upper-16-bit slice; "
                        "default, unchanged, unless an --algo recipe such as atlas_v1_awq sets rn), rn (NEW: round-to-nearest from 24 VPU instructions, "
                        "chip_numerics.vadd_rn), rne_hw (model of an RNE narrowing in the adder; pending "
                        "a hardware ruling). rn and rne_hw are recorded off the ladder")
    g.add_argument("--chip-vadd-sites", choices=chip_numerics.SITES, default="residual",
                   help="where --chip-vadd applies: residual (same-shape adds of computed tensors outside "
                        "norm/attention/MLP/matmul modules), data (every add of two computed tensors "
                        "except additive masks), all (every vadd/vsub, decompositions included)")
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
            _RUNNING.append(r)
        self.depth += 1

    def exit(self):
        self.depth -= 1
        if self.depth == 0:
            r = self.router
            r.__exit__(None, None, None)
            r.active = False
            _RUNNING.remove(r)
            rn.quantize = self.saved["quantize"]
            rn.e4m3_unpack = self.saved["e4m3_unpack"]
            rn.dequantize = self.saved["dequantize"]
            if "_mm" in self.saved:
                import ref_attention
                ref_attention._mm = self.saved["_mm"]
            if r.vadd is not None and getattr(r, "vadd_record", None) is not None:
                r.vadd_record.update(r.vadd.record())
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


TABLE_FUNCTIONS = ("create_sinusoidal_pos_embedding",)


def _host_tables(model, router) -> list:
    """--chip-rope host: position/schedule-only table functions of the model's module run on the
    host (NEW, op_unit_map.TABLE_HOST). The module attribute is wrapped once per process and pushes
    the context only for a router that is running, so other models sharing the module (the bf16
    baseline) are unaffected."""
    import sys

    mod = sys.modules.get(type(model).__module__)
    done = []
    for fname in TABLE_FUNCTIONS:
        fn = getattr(mod, fname, None) if mod is not None else None
        if fn is None:
            continue
        routers = getattr(fn, "_chip_routers", None)
        if routers is None:
            orig, routers = fn, []

            @functools.wraps(orig)
            def run(*a, _orig=orig, _routers=routers, _name=fname, **k):
                live = [r for r in _routers if r.active]
                for r in live:
                    r.push("rope_host", _name)
                try:
                    return _orig(*a, **k)
                finally:
                    for r in live:
                        r.pop()

            run._chip_routers = routers
            setattr(mod, fname, run)
        routers.append(router)
        done.append(f"{mod.__name__}.{fname}")
    return done


def quantize_chip(model: nn.Module, skip=(), options=None) -> dict:
    import quant_modes

    mxu_kind = _opt(options, "chip_mxu", "model") or "model"
    rep = quant_modes.quantize(model, "reference_hw" if mxu_kind == "model" else "reference", skip, options)
    opts = quant_modes.profile_options(options or {})
    rope = _opt(options, "chip_rope", "host") or "host"
    cov = _ChainCoverage(type(model).__name__)
    vadd_mode = _opt(options, "chip_vadd", "trunc") or "trunc"
    # explicit (never the environment): the CLI flags define the chip profile
    vadd = chip_numerics.VaddPolicy(vadd_mode, _opt(options, "chip_vadd_sites", "residual") or "residual")
    vadd = vadd if vadd.active else None
    router = cc.ChipRouter(cov, opts["mxu"], opts["mxu_impl"], rope=rope,
                           strict=bool(_opt(options, "chip_strict", False)),
                           native_ops=_opt(options, "chip_native", None) or (),
                           vadd=vadd if vadd is not None else chip_numerics.VaddPolicy("trunc"))
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
                if cfg.get(k) is None and mxu_kind == "framework":
                    # the reference attention routine (ref_attention) with no MXU model: its score
                    # matmuls run guarded, so give them the framework stand-in to keep them counted
                    cfg[k] = framework_fm
                if cfg.get(k) is not None and not isinstance(cfg[k], _GuardedFM):
                    cfg[k] = _GuardedFM(cfg[k], router, "attention")

    def ctx_hooks(mod, kind, detail):
        mod.register_forward_pre_hook(lambda *_: router.push(kind, detail))
        mod.register_forward_hook(lambda *_: router.pop())

    n_skip = 0
    for name, m in model.named_modules():
        s = next((s for s in skip if s and s in name), None)
        if name in (_opt(options, "_algo_bf16_exact", None) or ()):
            s = name
        if s is not None and name:
            opt_out = s in (_opt(options, "_algo_bf16", None) or ())
            ctx_hooks(m, "skip", cc.BF16_OPT_OUT + s if opt_out else s)
            n_skip += 1
        elif rope == "host" and type(m).__name__.endswith("RotaryEmbedding"):
            ctx_hooks(m, "rope_host", type(m).__name__)
        elif type(m).__name__ in ("RefQuantLinear", "RefQuantConv2d"):
            ctx_hooks(m, "mxu_module", type(m).__name__)

    if vadd is not None and vadd.sites == "residual":
        # module contexts that exclude an add from the residual site (chip_numerics.VaddPolicy)
        for name, m in model.named_modules():
            k = chip_numerics.module_kind(m)
            if k is not None:
                ctx_hooks(m, k, type(m).__name__)

    if any(s.get("variant") == "pi0" for s in rep.get("attention_sites", [])):
        # Pi0: every attention route (Gemma eager, SigLIP sdpa) is already the reference routine,
        # whose softmax and mask are separate ops; switching SigLIP to eager would change its
        # softmax precision away from the reference's SDPA route (pi0-quant model_patcher.py:787).
        eager = []
    else:
        eager = _set_eager(model)
    tables = _host_tables(model, router) if rope == "host" else []
    session = _Session(router)
    entries = [n for n in ENTRY_POINTS if _wrap_entry(model, n, session)]
    _ALL.append(cov)

    rep["profile"] = {**rep.get("profile", {}), "name": "chip",
                      "base": "reference_hw" if mxu_kind == "model" else "reference (framework accumulation)",
                      "mxu": mxu_kind,
                      "vpu": "chip_ops.ChipVPU (bit-exact to VectorRTLFunctions, 16 lanes)",
                      "rope": rope, "host_tables": tables, "entry_points": entries, "eager_attention_set": eager,
                      "skipped_to_host": list(dict.fromkeys(list(skip) + list(
                          _opt(options, "_algo_bf16_exact", None) or ()))), "skip_modules": n_skip,
                      "vpu_lut_sha256_16": chip_ops.lut_provenance()}
    if vadd is not None:
        router.vadd_record = vadd.record()
        rep["profile"]["vadd"] = router.vadd_record
    import ref_loader
    rep["reference_sources"] = ref_loader.provenance()
    rep["coverage"] = cov.refresh()
    return rep


@atexit.register
def _print_all():
    for cov in _ALL:
        if cov.rows:
            print(cc.format_report(cov.refresh()), flush=True)
