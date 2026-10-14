"""NEW bounded diagnostic traces, separate from clean inference timing.

Population moments describe every finite element. Quantiles and paired errors
describe only the explicitly recorded, deterministic sample; they are not gates.
No quantizer, model forward, or global RNG state is changed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import random

import torch

SCHEMA = "atlas_torchao.layer_trace.v1"


def _key(record):
    return (record["sample_id"], record["seed"], record["site"],
            record["boundary"], record["call_index"], record["stage"])


def _quantiles(values):
    if not values.numel():
        return None
    q = torch.tensor([0.5, 0.9, 0.95, 0.99], dtype=torch.float64)
    return dict(zip(("p50", "p90", "p95", "p99"), torch.quantile(values, q).tolist()))


@torch.no_grad()
def tensor_summary(value, *, sample_seed, max_samples=4096):
    """Exact population counts/moments and a bounded sample with explicit indices.

    This observer synchronizes/transfers values and allocates temporaries. Its
    cost and peak memory must never be included in clean performance results.
    """
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError("layer traces require a floating tensor")
    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    flat = value.detach().reshape(-1)
    n = flat.numel()
    finite = torch.isfinite(flat)
    count = int(finite.sum())
    clean = flat[finite].double()
    if count:
        variance, mean = torch.var_mean(clean, correction=0)
        population = {"mean": float(mean), "std": float(variance.sqrt()),
                      "rms": float(clean.square().mean().sqrt()),
                      "min": float(clean.min()), "max": float(clean.max()),
                      "max_abs": float(clean.abs().max())}
    else:
        population = dict.fromkeys(("mean", "std", "rms", "min", "max", "max_abs"))
    indices = sorted(random.Random(sample_seed).sample(range(n), min(n, max_samples)))
    selected = flat[torch.tensor(indices, device=flat.device, dtype=torch.long)].double().cpu()
    sampled_finite = selected[torch.isfinite(selected)]
    values = [float(x) if bool(torch.isfinite(x)) else None for x in selected]
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "tensor_bytes": value.numel() * value.element_size(),
            "population_elements": n, "finite_elements": count,
            "nonfinite_elements": n - count, "zero_elements": int((flat == 0).sum()),
            "population_finite": population,
            "sample": {"seed": sample_seed, "indices": indices, "values": values,
                       "elements": len(indices), "max_elements": max_samples,
                       "quantiles_are_sample_estimates": len(indices) < n,
                       "finite_abs_quantiles": _quantiles(sampled_finite.abs())}}


class LayerTrace:
    """Observe explicit sites; begin_observation resets per-site call indices.

    Call indices are recorded, not guessed to be flow/denoising iterations. The
    caller must supply an explicit stage and observation identity. Persist the
    resulting records next to the exact input/model/export manifests.
    """

    def __init__(self, model, sites, *, max_samples=4096, max_records=20000):
        if model.training:
            raise ValueError("diagnostic trace requires an evaluation-mode model")
        sites = list(sites)
        if not sites or len(set(sites)) != len(sites):
            raise ValueError("provide distinct exact module sites")
        if max_samples < 1 or max_records < 1:
            raise ValueError("trace budgets must be positive")
        modules = dict(model.named_modules())
        missing = set(sites) - set(modules)
        if missing:
            raise ValueError(f"trace sites missing: {sorted(missing)}")
        self.modules = {site: modules[site] for site in sites}
        self.max_samples, self.max_records = max_samples, max_records
        self.records, self.handles = [], []
        self.context, self.calls, self.pending = None, {}, {}

    def begin_observation(self, sample_id, *, mode, seed, stage="policy"):
        if self.pending:
            raise RuntimeError("cannot change observation during a traced forward")
        if not isinstance(sample_id, str) or not sample_id or not mode or not stage:
            raise ValueError("record an explicit sample, mode and stage")
        self.context = {"sample_id": sample_id, "mode": mode, "seed": int(seed), "stage": stage}
        self.calls = {}

    def _record(self, site, boundary, call_index, value):
        if self.context is None:
            raise RuntimeError("begin_observation is required before model execution")
        if len(self.records) >= self.max_records:
            raise RuntimeError("layer trace record budget exhausted; retain partial records")
        identity = {**self.context, "site": site, "boundary": boundary, "call_index": call_index}
        # Excluding mode ensures identical coordinates for paired precision runs.
        seed = int.from_bytes(hashlib.sha256(json.dumps(_key(identity)).encode()).digest()[:8], "little")
        # Do not let observer arithmetic contaminate the chip router's counters.
        try:
            from chip_coverage import guard
        except ImportError:
            guard = contextlib.nullcontext
        with guard():
            summary = tensor_summary(value, sample_seed=seed, max_samples=self.max_samples)
        self.records.append({"schema": SCHEMA, **identity, **summary,
                             "measurement_class": "instrumented_framework_diagnostic",
                             "sampling_step": None})

    def __enter__(self):
        if self.handles:
            raise RuntimeError("trace already active")
        for site, module in self.modules.items():
            def before(_module, args, name=site):
                if name in self.pending:
                    raise RuntimeError("recursive selected-module execution is unsupported")
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError("selected module must have a tensor positional input")
                index = self.calls.get(name, 0)
                self.calls[name] = index + 1
                self.pending[name] = index
                self._record(name, "input", index, args[0])

            def after(_module, _args, output, name=site):
                index = self.pending.pop(name)
                self._record(name, "output", index, output)

            self.handles.extend((module.register_forward_pre_hook(before),
                                 module.register_forward_hook(after)))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.pending.clear()


def compare_samples(reference, candidate):
    """Pair exact observation/site/call/coordinates; never invent missing matches."""
    def keyed(records):
        if not records:
            raise ValueError("empty traces cannot establish paired measurements")
        result = {}
        for record in records:
            sample = record["sample"]
            if (len(sample["indices"]) != len(sample["values"])
                    or len(set(sample["indices"])) != len(sample["indices"])
                    or any(index < 0 or index >= record["population_elements"] for index in sample["indices"])):
                raise ValueError("invalid trace sample coordinates or values")
            key = _key(record)
            if key in result:
                raise ValueError(f"duplicate trace key: {key}")
            result[key] = record
        return result
    ref, cand = keyed(reference), keyed(candidate)
    if set(ref) != set(cand):
        raise ValueError("trace coverage differs; missing matches are not zero error")
    result = []
    for key, r in ref.items():
        c = cand[key]
        if (r["shape"] != c["shape"] or r["population_elements"] != c["population_elements"]
                or r["sample"]["indices"] != c["sample"]["indices"]):
            raise ValueError("paired trace shape or sample coordinates differ")
        rv, cv = r["sample"]["values"], c["sample"]["values"]
        if any(x is None for x in rv + cv):
            raise ValueError("nonfinite sampled values cannot yield finite error metrics")
        error = torch.tensor(cv, dtype=torch.float64) - torch.tensor(rv, dtype=torch.float64)
        metrics = None if not error.numel() else {
            "rmse": float(error.square().mean().sqrt()), "bias": float(error.mean()),
            "mae": float(error.abs().mean()), "std": float(error.std(correction=0)),
            "max_abs": float(error.abs().max()), "abs_quantiles": _quantiles(error.abs())}
        result.append({"sample_id": key[0], "seed": key[1], "site": key[2],
                       "boundary": key[3], "call_index": key[4], "stage": key[5],
                       "reference_mode": r["mode"], "candidate_mode": c["mode"],
                       "sample_elements": error.numel(), "sample_metrics": metrics,
                       "comparison_scope": "paired whole-policy boundary samples; not clean-input local replay",
                       "full_tensor_error_measured": len(rv) == r["population_elements"]})
    return result
