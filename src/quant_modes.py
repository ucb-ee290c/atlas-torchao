"""Quantization modes shared by eval.py (LLM perplexity) and eval_policy.py (policy actions).

  bf16        no quantization (the reference)
  atlas       AtlasQConfig: Atlas FP8 E4M3 rules, power-of-two per-tensor scales
  atlas_hw    as atlas, activations through the hardware cast (|x/2^s| < 2^-6 flushed to zero)
  torchao     stock TorchAO Float8DynamicActivationFloat8WeightConfig, per-tensor
              (native FP8 matmul; needs compute capability >= 8.9, e.g. L40 / H100)
"""

import torch
import torch.nn as nn

QUANT_MODES = ("atlas", "atlas_hw", "torchao")
ALL_MODES = ("bf16", *QUANT_MODES)


def require_mode_supported(mode: str, device: str) -> None:
    """Raise RuntimeError with a clear message if `mode` cannot run on `device`."""
    if mode != "torchao":
        return
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("mode 'torchao' needs a CUDA GPU with compute capability >= 8.9 "
                           f"(native FP8 matmul); device is {device!r}")
    idx = torch.device(device).index
    cap = torch.cuda.get_device_capability(idx if idx is not None else torch.cuda.current_device())
    if cap < (8, 9):
        raise RuntimeError(f"mode 'torchao' needs compute capability >= 8.9 (L40/H100), "
                           f"found {cap[0]}.{cap[1]} on {torch.cuda.get_device_name(idx)}")


def quantize(model: nn.Module, mode: str, skip=()) -> dict:
    """Quantize the nn.Linear layers of `model` in place (fqn substrings in `skip` are kept).

    Returns {"planned", "quantized", "untouched"}: a layer counts as quantized only if quantize_
    actually changed it (TorchAO's float8 path skips some shapes without raising).
    """
    from torchao.quantization import quantize_

    def keep(module, fqn):
        # only bf16 layers: a deliberately-float32 layer (e.g. SmolVLA's action_out_proj) stays as is
        return (isinstance(module, nn.Linear) and module.weight.dtype == torch.bfloat16
                and not any(s in fqn for s in skip))

    if mode == "bf16":
        return {"planned": 0, "quantized": 0, "untouched": []}
    planned = [fqn for fqn, m in model.named_modules() if keep(m, fqn)]
    before = {fqn: type(m) for fqn, m in model.named_modules() if fqn in set(planned)}
    if mode in ("atlas", "atlas_hw"):
        from atlas_torchao import AtlasQConfig

        config = AtlasQConfig(hw_activation_cast=(mode == "atlas_hw"))
    elif mode == "torchao":
        from torchao.quantization import Float8DynamicActivationFloat8WeightConfig, PerTensor

        config = Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    else:
        raise ValueError(f"unknown mode {mode!r}; choose from {ALL_MODES}")
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
