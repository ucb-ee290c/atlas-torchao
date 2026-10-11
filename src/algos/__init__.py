"""FP8 quantization algorithms selectable with `--algo NAME[:k=v,...]` (NEW; docs/algos/README.md).

`--algo` is orthogonal to the ladder modes: every mode except bf16 runs with the algorithm's hooks
(src/algos/base.py), and `--algo none` (the default) leaves every mode exactly as l40s-run-2.1.

Registration is one line per algorithm in REGISTRY: name -> module. The module defines `ALGO`, an
`algos.base.Algo` subclass constructed with the parsed k=v params.
"""

from __future__ import annotations

import contextlib
import importlib

REGISTRY = {
    "none": None,
    "example_identity": "algos.example_identity",
    "granularity": "algos.granularity",
}

_CURRENT = []


def parse(spec: str | None) -> tuple[str, dict]:
    """'name:k=v,k2=v2' -> (name, {k: v}); values are int, float, true/false or strings."""
    spec = (spec or "none").strip()
    name, _, rest = spec.partition(":")
    params = {}
    for item in filter(None, (p.strip() for p in rest.split(","))):
        k, eq, v = item.partition("=")
        if not eq:
            raise ValueError(f"--algo {spec!r}: parameter {item!r} is not k=v")
        params[k.strip()] = _value(v.strip())
    if name not in REGISTRY:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(REGISTRY)}")
    if name == "none" and params:
        raise ValueError("--algo none takes no parameters")
    return name, params


def _value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def make(spec: str | None):
    """The Algo instance for `spec`, or None for `none`."""
    name, params = parse(spec)
    if REGISTRY[name] is None:
        return None
    algo = importlib.import_module(REGISTRY[name]).ALGO(**params)
    algo.name = name
    return algo


def current():
    """The algorithm whose quantization is in progress (read by the reference modules)."""
    return _CURRENT[-1] if _CURRENT else None


@contextlib.contextmanager
def active(algo):
    _CURRENT.append(algo)
    try:
        yield algo
    finally:
        _CURRENT.pop()


def add_args(ap, calib_sources=("auto",)) -> None:
    g = ap.add_argument_group("quantization algorithm (docs/algos/README.md)")
    g.add_argument("--algo", default="none",
                   help="NAME[:k=v,...] from src/algos (default none: the profiles unchanged); "
                        f"registered: {', '.join(sorted(REGISTRY))}")
    g.add_argument("--calib-n", type=int, default=None,
                   help="calibration samples (windows / frames / observations); default: the algorithm's")
    g.add_argument("--calib-seed", type=int, default=1234,
                   help="seed of the calibration sample (never the eval data, see docs/algos)")
    g.add_argument("--calib-source", choices=calib_sources, default=calib_sources[0],
                   help="calibration data source (model-specific; never overlaps the eval data)")


def none_record(spec, mode=None) -> dict:
    name, params = parse(spec)
    rec = {"name": name, "params": params, "calibration": None}
    if name != "none" and mode == "bf16":
        rec.update(applied=False, reason="bf16 baseline: no algorithm applied")
    return rec


def entry_record(layers, mode, spec) -> dict:
    """The `algo` record of one mode's result entry (popped from the quantize report if set)."""
    if isinstance(layers, dict) and "algo" in layers:
        return layers.pop("algo")
    return none_record(spec, mode)


def quantize(model, mode, skip=(), options=None, algo=None, calib=None) -> dict:
    """quant_modes.quantize with the algorithm's hooks. algo None: exactly quant_modes.quantize.

    calib: zero-argument callable returning (batches, run, record) for (f): `batches` an iterable
    of calibration inputs, run(model, batch) one forward, record {source, n_samples, seed, ...}.
    It is called only if the algorithm needs calibration."""
    import quant_modes

    if algo is None:
        return quant_modes.quantize(model, mode, skip, options)
    from algos.base import Algo

    Algo.__init__(algo, **algo.params)          # fresh per-mode stats (the instance is reused)
    skip = list(skip or ())
    bf16 = list(algo.bf16_layers(model, mode) or [])
    if algo.needs_calibration:
        if calib is None:
            raise RuntimeError(f"algorithm {algo.name!r} needs calibration data and none was provided")
        batches, run, crec = calib()
        out = algo.calibrate(model, batches, run)
        algo.calibration = {**crec, **({"result": out} if out else {})}
    prep = algo.prepare(model, mode)
    if bf16:
        # the chip profile labels these skips as BF16 opt-outs in its coverage report
        options = {**quant_modes._options_dict(options), "_algo_bf16": tuple(bf16)}
    with active(algo):
        rep = quant_modes.quantize(model, mode, skip + bf16, options)
    rec = algo.record()
    base = quant_modes.PROFILES[quant_modes.canonical(mode)].base
    hooks = algo.overridden_module_hooks()
    rec.update(applied=True, bf16_opt_out=bf16, prepare=prep,
               module_hooks=hooks, module_hooks_applied=base in ("reference", "reference_hw", "chip"))
    if hooks and not rec["module_hooks_applied"]:
        rec["module_hooks_note"] = f"{mode}: module hooks {hooks} act only in the reference modules"
    rep["algo"] = rec
    return rep
