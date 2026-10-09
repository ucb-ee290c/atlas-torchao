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


PROFILES = {
    p.name: p for p in (
        Profile("bf16", "no quantization", "-"),
        Profile("collab_v0", "AtlasQConfig, ceil/448 scales, fixed activation scale 2^0, torch fp8 "
                "activation cast, F.linear", "the collaborator (atlas_torchao.py, atlas_quant.py)",
                aliases=("atlas",)),
        Profile("collab_v0_hw", "collab_v0 with the hardware activation cast", "the collaborator "
                "(atlas_torchao.py:57-61)", aliases=("atlas_hw",)),
        Profile("reference", "dynamic po2 scales + vpack cast for x and W, F.linear on dequantized "
                "values (NEW), BF16 bias after", "pi0-quant hw_float_ops.py:38-57,64-103; "
                "quant_linear.py:153-172"),
        Profile("reference_hw", "reference cast, MXU accumulation model (--mxu ipt|sa), output "
                "x 2^(sx+sw), BF16 bias after", "pi0-quant quant_linear.py:145-172; "
                "functional_models.py:79-108"),
        Profile("torchao", "Float8DynamicActivationFloat8WeightConfig, per-tensor", "torchao 0.18"),
        Profile("chip", "reference_hw + every other op on the Atlas VPU / scalar core, or listed as "
                "host fallback (src/chip_profile.py)", "see src/op_unit_map.py"),
    )
}
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
    g.add_argument("--mxu-impl", choices=IMPLS, default="ref_cpu",
                   help="ref_cpu: the reference CPU models (oracles); torch: GPU-capable port")
    __import__("chip_profile").add_args(ap)


def profile_options(options) -> dict:
    """The profile options in `options` (an argparse namespace or a dict), with defaults."""
    from accum_backends import DEFAULT_MXU

    get = options.get if isinstance(options, dict) else (lambda k, d=None: getattr(options, k, d))
    return {"mxu": get("mxu", DEFAULT_MXU) or DEFAULT_MXU,
            "mxu_impl": get("mxu_impl", "ref_cpu") or "ref_cpu"}


def quantize(model: nn.Module, mode: str, skip=(), options=None) -> dict:
    """Quantize `model` in place under profile `mode` (fqn substrings in `skip` are kept).

    `options` (argparse namespace or dict) carries the reference-profile options
    (add_profile_args); other profiles ignore it.

    Returns {"planned", "quantized", "untouched"}: a layer counts as quantized only if it was
    actually replaced (TorchAO's float8 path skips some shapes without raising).
    """
    name = canonical(mode)
    if name == "chip":
        return __import__("chip_profile").quantize_chip(model, skip, options)
    if name == "bf16":
        return {"planned": 0, "quantized": 0, "untouched": []}
    if name in ("reference", "reference_hw"):
        import ref_loader
        from ref_profile import quantize_reference

        opts = profile_options(options or {})
        factory = None
        if name == "reference_hw":
            from accum_backends import describe, get_accumulator

            get_accumulator(opts["mxu"], opts["mxu_impl"])          # fail before replacing anything
            factory = lambda: get_accumulator(opts["mxu"], opts["mxu_impl"])  # noqa: E731
        rep = quantize_reference(model, skip, fm_factory=factory)
        rep["profile"] = {"name": name,
                          **({"accumulation": describe(opts["mxu"], opts["mxu_impl"])}
                             if name == "reference_hw" else {})}
        rep["reference_sources"] = ref_loader.provenance()
        return rep
    return _quantize_torchao_style(model, name, skip)
