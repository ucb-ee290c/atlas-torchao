"""Quantization profiles shared by eval.py (LLM perplexity) and eval_policy.py (policy actions).

  bf16          no quantization (the reference)
  collab_v0     the collaborator's AtlasQConfig: Atlas FP8 E4M3 rules, power-of-two per-tensor
                scales (ceil/448), fixed activation scale 2^0. Alias: atlas
  collab_v0_hw  as collab_v0, activations through the hardware cast (|x/2^s| < 2^-6 flushed to
                zero). Alias: atlas_hw
  reference     the reference author's operand cast: dynamic per-tensor po2 scales
                (floor(log2(amax/256))) and the vpack encoder for activations and weights, every
                nn.Linear including fp32 ones; F.linear on the dequantized operands (NEW), BF16
                bias after (src/ref_profile.py)
  reference_hw  as reference, with the matmul through an MXU accumulation model: --mxu ipt
                (default, the reference author's choice) or sa, implementation --mxu-impl
                ref_cpu (the reference CPU models) or torch (src/accum_torch.py)
  torchao       stock TorchAO Float8DynamicActivationFloat8WeightConfig, per-tensor
                (native FP8 matmul; needs compute capability >= 8.9, e.g. L40S / H100)

The collab_v0 profiles keep exactly the behaviour of atlas-eval (daf8e78) under both their new
names and their old ones, so the collaborator's numbers reproduce
(tests/test_profiles.py::test_collab_v0_golden).

Coverage ladder (NEW): each step adds exactly one source of error over the previous one, so the
difference between two consecutive steps on the same windows / frames isolates that source.

  1 bf16           baseline
  2 collab_v0      Linear FP8 the collaborator's way (fixed 2^0 activation scale, torch cast)
  3 ref_linear     reference numerics (dynamic po2 floor/256 scales, vpack cast, BF16 bias after
                   the rescale) on nn.Linear only; framework accumulation
  4 ref_attn       + attention Q.K^T and P.V, per-head FP8
  5 ref_conv       + Conv2d (alias: reference); equals ref_attn for a model without Conv2d
  6 reference_hw   + Atlas MXU accumulation model (--mxu ipt|sa), with the cross-check
  7 chip           + every other op on the VPU / scalar core, RoPE and timestep tables as host
                   constants (--chip-rope host, the default)
  8 chip_rope_vpu  chip with the tables computed on the VPU (the reference author's treatment)

Every report carries `ladder` (ladder_record): the step, what it adds, and exactly which op
classes were quantized and how matmuls, vector ops and tables were emulated. An explicit --ops
or --chip-rope that changes a step's definition is recorded as `on_ladder: false`.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Profile:
    name: str
    summary: str
    source: str
    aliases: tuple = field(default=())
    impl: str = ""        # the implementation profile it configures ("" = itself)
    ops: tuple = ()       # reference profiles: the op classes it quantizes unless --ops is given
    fixed: tuple = ()     # ((option, value), ...) the profile sets regardless of the CLI
    step: int = 0         # position on the coverage ladder (0 = off the ladder)
    adds: str = ""        # what the step adds over the previous one

    @property
    def base(self) -> str:
        return self.impl or self.name


OPS = ("linear", "conv2d", "attention")

PROFILES = {
    p.name: p for p in (
        Profile("bf16", "no quantization", "-", step=1, adds="baseline"),
        Profile("collab_v0", "AtlasQConfig, ceil/448 scales, fixed activation scale 2^0, torch fp8 "
                "activation cast, F.linear", "the collaborator (atlas_torchao.py, atlas_quant.py)",
                aliases=("atlas",), step=2, adds="Linear FP8, the collaborator's cast and scales"),
        Profile("collab_v0_hw", "collab_v0 with the hardware activation cast", "the collaborator "
                "(atlas_torchao.py:57-61)", aliases=("atlas_hw",)),
        Profile("ref_linear", "reference cast (dynamic po2 floor/256, vpack) on nn.Linear only, "
                "F.linear on dequantized values (NEW), BF16 bias after", "pi0-quant "
                "hw_float_ops.py:38-57,64-103; quant_linear.py:153-172", impl="reference",
                ops=("linear",), step=3, adds="reference numerics on Linear (framework accumulation)"),
        Profile("ref_attn", "ref_linear + attention Q.K^T / P.V per-head FP8", "model_patcher.py:624-723",
                impl="reference", ops=("linear", "attention"), step=4,
                adds="attention Q.K^T and P.V per-head FP8"),
        Profile("ref_conv", "ref_attn + Conv2d (im2col, per-tensor casts): the reference profile",
                "model_patcher.py:282-375", aliases=("reference",), impl="reference", ops=OPS, step=5,
                adds="Conv2d FP8 (none for a model without Conv2d)"),
        Profile("reference_hw", "reference cast, MXU accumulation model (--mxu ipt|sa), output "
                "x 2^(sx+sw), BF16 bias after", "pi0-quant quant_linear.py:145-172; "
                "functional_models.py:79-108", ops=OPS, step=6,
                adds="Atlas MXU accumulation (--mxu, cross-checked)"),
        Profile("torchao", "Float8DynamicActivationFloat8WeightConfig, per-tensor", "torchao 0.18"),
        Profile("chip", "reference_hw + every other op on the Atlas VPU / scalar core, or listed as "
                "host fallback; RoPE / timestep tables host constants by default (src/chip_profile.py)",
                "see src/op_unit_map.py", ops=OPS, step=7,
                adds="VPU / scalar-core vector ops, lane-exact; position/timestep tables host constants"),
        Profile("chip_rope_vpu", "chip with RoPE / timestep tables computed on the VPU (--chip-rope vpu)",
                "the reference author's treatment, quant_vector.py:170-171", impl="chip", ops=OPS,
                fixed=(("chip_rope", "vpu"),), step=8,
                adds="RoPE / timestep sin-cos tables on the VPU instead of host constants"),
    )
}
LADDER = tuple(sorted((p.name for p in PROFILES.values() if p.step), key=lambda n: PROFILES[n].step))
# Steps that run an MXU accumulation model: slow, so they are run on a subset of windows / frames.
SUBSET_PROFILES = ("reference_hw", "chip", "chip_rope_vpu")
FAST_PREDECESSOR = "ref_conv"     # the last fast step, re-run on a subset by --subset-compare
ALIASES = {a: p.name for p in PROFILES.values() for a in p.aliases}

QUANT_MODES = tuple(n for n in (*PROFILES, *ALIASES) if n != "bf16")
ALL_MODES = ("bf16", *QUANT_MODES)


def canonical(mode: str) -> str:
    """The profile name behind a mode or alias (atlas -> collab_v0)."""
    name = ALIASES.get(mode, mode)
    if name not in PROFILES:
        raise ValueError(f"unknown mode {mode!r}; choose from {ALL_MODES}")
    return name


def require_mode_supported(mode: str, device: str) -> None:
    """Raise RuntimeError with a clear message if `mode` cannot run on `device`."""
    if canonical(mode) != "torchao":
        return
    try:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor  # noqa: F401
    except ImportError as e:
        import torchao

        raise RuntimeError(f"mode 'torchao' needs torchao.quantization."
                           f"Float8DynamicActivationFloat8WeightConfig and PerTensor, which torchao "
                           f"{torchao.__version__} does not provide ({e})") from e
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("mode 'torchao' needs a CUDA GPU with compute capability >= 8.9 "
                           f"(native FP8 matmul); device is {device!r}")
    idx = torch.device(device).index
    cap = torch.cuda.get_device_capability(idx if idx is not None else torch.cuda.current_device())
    if cap < (8, 9):
        raise RuntimeError(f"mode 'torchao' needs compute capability >= 8.9 (L40S/H100), "
                           f"found {cap[0]}.{cap[1]} on {torch.cuda.get_device_name(idx)}")


def _quantize_torchao_style(model: nn.Module, name: str, skip) -> dict:
    """collab_v0 / collab_v0_hw / torchao: torchao quantize_ over the bf16 nn.Linear layers."""
    from torchao.quantization import quantize_

    def keep(module, fqn):
        # only bf16 layers: a deliberately-float32 layer (e.g. SmolVLA's action_out_proj) stays as is
        return (isinstance(module, nn.Linear) and module.weight.dtype == torch.bfloat16
                and not any(s in fqn for s in skip))

    planned = [fqn for fqn, m in model.named_modules() if keep(m, fqn)]
    before = {fqn: type(m) for fqn, m in model.named_modules() if fqn in set(planned)}
    if name in ("collab_v0", "collab_v0_hw"):
        from atlas_torchao import AtlasQConfig

        config = AtlasQConfig(hw_activation_cast=(name == "collab_v0_hw"))
    else:
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor

        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    with torch.no_grad():
        quantize_(model, config, filter_fn=keep)
    modules = dict(model.named_modules())
    changed = [
        fqn for fqn in planned
        if type(modules[fqn]) is not before[fqn]
        or type(getattr(modules[fqn], "weight", None)) not in (torch.Tensor, nn.Parameter)
    ]
    return {"planned": len(planned), "quantized": len(changed),
            "untouched": sorted(set(planned) - set(changed))}


def add_profile_args(ap) -> None:
    """Options of the reference profiles, shared by the eval CLIs."""
    from accum_backends import DEFAULT_MXU, IMPLS, MXUS

    g = ap.add_argument_group("reference profiles")
    g.add_argument("--mxu", choices=MXUS, default=DEFAULT_MXU,
                   help="reference_hw accumulation model: ipt (inner-product tree, the reference "
                        "author's choice) or sa (per-step systolic array)")
    g.add_argument("--mxu-impl", choices=IMPLS, default="torch",
                   help="torch: bit-exact device-agnostic port (default); ref_cpu: the reference "
                        "CPU models (the oracles; slow)")
    g.add_argument("--accum-compile", action="store_true",
                   help="torch.compile the SA step (recommended on GPU for --mxu sa)")
    g.add_argument("--crosscheck", choices=("on", "off"), default="on",
                   help="reference_hw with --mxu-impl torch: recompute a seeded sample of real "
                        "calls with the reference CPU model and compare bits; any mismatch marks "
                        "the result invalid (default on)")
    g.add_argument("--crosscheck-frac", type=float, default=0.01,
                   help="fraction of calls checked beyond the first call of each shape")
    g.add_argument("--crosscheck-budget", type=float, default=600.0,
                   help="seconds of reference CPU time the cross-check may spend")
    __import__("chip_profile").add_args(ap)
    g.add_argument("--ops", default=None,
                   help="comma-separated ops the reference profiles cover: linear, conv2d, attention "
                        "(smolVLA run_eval.py:7-18). Default: the profile's own (ref_linear: linear; "
                        "ref_attn: linear,attention; the others all). Given explicitly it overrides "
                        "every profile and the result is recorded as off the ladder")


def profile_options(options) -> dict:
    """The profile options in `options` (an argparse namespace or a dict), with defaults."""
    from accum_backends import DEFAULT_MXU

    get = options.get if isinstance(options, dict) else (lambda k, d=None: getattr(options, k, d))
    ops = get("ops", None) or get("_profile_ops", None) or ",".join(OPS)
    ops = tuple(o.strip() for o in (ops.split(",") if isinstance(ops, str) else ops) if o.strip())
    bad = set(ops) - set(OPS)
    if bad:
        raise ValueError(f"unknown ops {sorted(bad)}; choose from {OPS}")
    return {"mxu": get("mxu", DEFAULT_MXU) or DEFAULT_MXU,
            "mxu_impl": get("mxu_impl", "torch") or "torch", "ops": ops,
            "accum_compile": bool(get("accum_compile", False)) or None,
            "crosscheck": (get("crosscheck", "on") or "on") == "on",
            "crosscheck_frac": float(get("crosscheck_frac", 0.01)),
            "crosscheck_budget": float(get("crosscheck_budget", 600.0))}


def finalize_report(rep):
    """Call after the evaluation, before writing the JSON: turns the live cross-check into its
    summary and marks the result invalid if any checked bit differed (or nothing was checked)."""
    if not isinstance(rep, dict):
        return rep
    cc = rep.get("crosscheck")
    if cc is not None and hasattr(cc, "summary"):
        rep["crosscheck"] = cc = cc.summary()
    if isinstance(cc, dict) and not cc.get("ok", False):
        rep["valid"] = False
        rep.setdefault("invalid_reasons", []).append(
            f"crosscheck: {cc.get('bit_mismatches')} bit mismatches, {cc.get('calls_checked')} calls "
            f"checked, {len(cc.get('errors', []))} errors")
        print("WARNING: reference_hw cross-check failed; result marked invalid:",
              rep["invalid_reasons"][-1], flush=True)
    elif isinstance(cc, dict):
        rep.setdefault("valid", True)
    return rep


def _options_dict(options) -> dict:
    if options is None:
        return {}
    return dict(options) if isinstance(options, dict) else dict(vars(options))


def resolved_options(mode: str, options=None) -> dict:
    """`options` as a dict with the profile's own settings applied: its op classes (unless --ops
    was given) under `_profile_ops`, and its fixed options (chip_rope_vpu: chip_rope=vpu)."""
    prof = PROFILES[canonical(mode)]
    opts = _options_dict(options)
    if prof.ops:
        opts["_profile_ops"] = ",".join(prof.ops)
    opts.update(dict(prof.fixed))
    if prof.base == "chip" and not opts.get("chip_rope"):
        opts["chip_rope"] = "host"           # NEW default: tables are compile-time host constants
    return opts


def quantize(model: nn.Module, mode: str, skip=(), options=None) -> dict:
    """Quantize `model` in place under profile `mode` (fqn substrings in `skip` are kept).

    `options` (argparse namespace or dict) carries the reference-profile options
    (add_profile_args); other profiles ignore it. Ladder profiles (ref_linear, ref_attn, ref_conv,
    chip_rope_vpu) configure their implementation profile through resolved_options.

    Returns {"planned", "quantized", "untouched", "ladder"}: a layer counts as quantized only if it
    was actually replaced (TorchAO's float8 path skips some shapes without raising); `ladder` is
    ladder_record(mode, report).
    """
    opts = resolved_options(mode, options)
    rep = _quantize_profile(model, PROFILES[canonical(mode)].base, skip, opts)
    rep["ladder"] = ladder_record(mode, rep, opts)
    return rep


def _count(rep, key):
    v = rep.get(key)
    if isinstance(v, dict):
        return int(v.get("quantized", 0))
    return len(v) if isinstance(v, list) else 0


def ladder_record(mode: str, rep: dict, options=None) -> dict:
    """What `mode` quantized and emulated, for the result JSON: the ladder step, what it adds over
    the previous step, the op classes actually quantized (with counts), and how matmuls, vector ops
    and position / timestep tables ran. `equivalent_to` names the previous step when the op class
    this step adds found nothing to quantize (TinyLlama has no Conv2d: ref_conv == ref_attn)."""
    name = canonical(mode)
    prof = PROFILES[name]
    opts = _options_dict(options)
    base = prof.base
    ops = profile_options(opts)["ops"] if base in ("reference", "reference_hw", "chip") else ()
    rec = {"mode": mode, "profile": name, "step": prof.step or None, "adds": prof.adds or None,
           "previous": LADDER[prof.step - 2] if prof.step > 1 else None}
    if base == "bf16":
        rec.update(operand_cast=None, quantized_op_classes={}, accumulation="framework",
                   vector_ops="framework", rope_tables="framework")
        return {**rec, "on_ladder": True, "off_ladder_reasons": []}
    if base in ("collab_v0", "collab_v0_hw", "torchao"):
        cast = {"collab_v0": "collab: ceil(log2(amax/448)) weight scale, fixed 2^0 activation scale, "
                             "torch fp8 cast",
                "collab_v0_hw": "collab with the hardware activation cast",
                "torchao": "torchao float8 per-tensor dynamic"}[base]
        rec.update(operand_cast=cast, quantized_op_classes={"linear": int(rep.get("quantized", 0))},
                   accumulation="framework" if base != "torchao" else "native fp8 matmul",
                   vector_ops="framework", rope_tables="framework")
        return {**rec, "on_ladder": bool(prof.step), "off_ladder_reasons": []}
    classes = {}
    if "linear" in ops:
        classes["linear"] = int(rep.get("quantized", 0))
    if "attention" in ops:
        classes["attention"] = _count(rep, "attention_sites")
    if "conv2d" in ops:
        classes["conv2d"] = _count(rep, "conv2d")
    chip_mxu = opts.get("chip_mxu") or "model"
    if base == "reference" or (base == "chip" and chip_mxu == "framework"):
        acc = "framework" if base == "reference" else "framework (FP8 operands, float32 matmul, one BF16 rounding)"
    else:
        po = profile_options(opts)
        acc = f"mxu {po['mxu']} ({po['mxu_impl']})"
    rec.update(operand_cast="reference: dynamic po2 floor(log2(amax/256)), vpack; BF16 bias after rescale",
               quantized_op_classes=classes, accumulation=acc,
               vector_ops="vpu (ChipVPU, lane-exact) + scalar core" if base == "chip" else "framework",
               rope_tables=(opts.get("chip_rope") or "host") if base == "chip" else "framework")
    off = []
    if prof.ops and tuple(sorted(ops)) != tuple(sorted(prof.ops)):
        off.append(f"--ops {','.join(ops)} instead of {','.join(prof.ops)}")
    if base == "chip" and name == "chip" and rec["rope_tables"] != "host":
        off.append(f"--chip-rope {rec['rope_tables']} (chip_rope_vpu is that step)")
    if base == "chip" and chip_mxu == "framework":
        off.append("--chip-mxu framework (no MXU accumulation model)")
    added = {"ref_attn": "attention", "ref_conv": "conv2d"}.get(name)
    if added and not off and classes.get(added, 0) == 0:
        rec["equivalent_to"] = rec["previous"]
    return {**rec, "on_ladder": bool(prof.step) and not off, "off_ladder_reasons": off}


def add_subset_compare_arg(ap) -> None:
    ap.add_argument("--subset-compare", choices=("on", "off"), default="on",
                    help="when a subset profile (reference_hw, chip, chip_rope_vpu) is requested, also "
                         "evaluate bf16 and the preceding fast step (ref_conv) on the identical "
                         "windows / frames, so the ladder deltas compare like for like (default on)")


def subset_compare_modes(modes, enabled=True) -> tuple:
    """(modes, added): with any subset profile in `modes`, bf16 and ref_conv (or its alias
    `reference`) are added if missing, ahead of the first subset profile. Every mode of one run
    sees the same windows / frames, so this makes the subset steps' deltas like-for-like."""
    modes = list(modes)
    if not enabled or not any(canonical(m) in SUBSET_PROFILES for m in modes):
        return modes, []
    have = {canonical(m) for m in modes}
    need = [m for m in ("bf16", FAST_PREDECESSOR) if m not in have]
    first = min(i for i, m in enumerate(modes) if canonical(m) in SUBSET_PROFILES)
    return modes[:first] + need + modes[first:], need


def ladder_deltas(metrics: dict, ladder: dict | None = None) -> list:
    """Consecutive-step deltas over the modes of ONE run (so on identical windows / frames).
    `metrics`: {mode: {metric: value}}; `ladder`: {mode: ladder_record}, whose off-ladder modes
    are skipped. Missing steps are skipped too: a delta is between the nearest steps present,
    named in `from`/`to`, and `steps` > 1 shows a gap (the delta then mixes several causes)."""
    present = {}
    for mode, m in metrics.items():
        try:
            name = canonical(mode)
        except ValueError:
            continue
        rec = (ladder or {}).get(mode)
        if rec is not None and not rec.get("on_ladder", True):
            continue
        if PROFILES[name].step and name not in present:
            present[name] = (mode, m)
    out, prev = [], None
    for name in LADDER:
        if name not in present:
            continue
        if prev is not None:
            (pm, pv), (cm, cv) = present[prev], present[name]
            out.append({"from": pm, "to": cm, "steps": PROFILES[name].step - PROFILES[prev].step,
                        "adds": PROFILES[name].adds,
                        "delta": {k: cv[k] - pv[k] for k in cv
                                  if isinstance(cv.get(k), (int, float)) and isinstance(pv.get(k), (int, float))}})
        prev = name
    return out


def _quantize_profile(model: nn.Module, name: str, skip=(), options=None) -> dict:
    """The implementation profiles: bf16, collab_v0(_hw), reference, reference_hw, chip, torchao."""
    if name == "chip":
        return __import__("chip_profile").quantize_chip(model, skip, options)
    if name == "bf16":
        return {"planned": 0, "quantized": 0, "untouched": []}
    if name in ("reference", "reference_hw"):
        import ref_loader
        from ref_profile import quantize_reference

        opts = profile_options(options or {})
        factory = fm0 = None
        if name == "reference_hw":
            from accum_backends import describe, get_accumulator

            from accum_backends import crosschecked

            fm0 = get_accumulator(opts["mxu"], opts["mxu_impl"], compile=opts["accum_compile"])
            if opts["mxu_impl"] == "torch":
                # stateless: one instance serves every layer
                if opts["crosscheck"]:
                    fm0 = crosschecked(opts["mxu"], fm0, opts["crosscheck_frac"],
                                       time_budget_s=opts["crosscheck_budget"])
                factory = lambda: fm0  # noqa: E731
            else:
                factory = lambda: get_accumulator(opts["mxu"], opts["mxu_impl"])  # noqa: E731
        from ref_attention import patch_attention, quantize_conv2d

        if "linear" in opts["ops"]:
            rep = quantize_reference(model, skip, fm_factory=factory)
        else:
            rep = {"planned": 0, "quantized": 0, "untouched": []}
        if "conv2d" in opts["ops"]:
            rep["conv2d"] = quantize_conv2d(model, skip, fm_factory=factory)
        if "attention" in opts["ops"]:
            rep["attention_sites"] = patch_attention(model, fm_factory=factory, skip=skip)
        rep["profile"] = {"name": name, "ops": list(opts["ops"]),
                          **({"accumulation": describe(opts["mxu"], opts["mxu_impl"])}
                             if name == "reference_hw" else {})}
        if name == "reference_hw" and hasattr(fm0, "summary"):
            rep["crosscheck"] = fm0                  # replaced by its summary in finalize_report
        rep["reference_sources"] = ref_loader.provenance()
        return rep
    return _quantize_torchao_style(model, name, skip)
