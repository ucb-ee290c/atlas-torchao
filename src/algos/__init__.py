"""FP8 quantization algorithms selectable with `--algo NAME[:k=v,...]` (NEW; docs/algos/README.md).

`--algo` is orthogonal to the ladder modes: every mode except bf16 runs with the algorithm's hooks
(src/algos/base.py), and `--algo none` (the default) leaves every mode exactly as l40s-run-2.1.

Registration is one line per algorithm in REGISTRY: name -> module. The module defines `ALGO`, an
`algos.base.Algo` subclass constructed with the parsed k=v params.

Stacks (NEW; docs/algos/stacking.md): `--algo A:k=v+B:k=v` joins several algorithms with `+` into
an `algos.stack.Stack`. A `+` starts a new algorithm only when the text after it names a registered
algorithm, so a value such as smoothquant's `sites=qkv+fc1` keeps its `+`. Likewise a parameter
token without `=` continues the previous value: `rotation:online=down,o,impl=fp32` gives
online="down,o". A single spec parses and behaves exactly as before.
"""

from __future__ import annotations

import contextlib
import importlib

REGISTRY = {
    "none": None,
    "example_identity": "algos.example_identity",
    "granularity": "algos.granularity",
    "mixed_precision": "algos.mixed_precision",
    "static_calib": "algos.static_calib",
    "smoothquant": "algos.smoothquant",
    "gptq_awq": "algos.gptq_awq",
}

# Named recipes (NEW; docs/algos/atlas_v1.md): `--algo NAME` expands to a documented stack, and
# NAME+other:k=v appends opt-in members (e.g. atlas_v1+mixed_precision:...). `chip` options apply
# to the chip profiles unless the command line sets them (--chip-vadd). In the SQ/static recipes,
# smoothquant runs with partial=true so that a policy's SigLIP / CLIP / DiT blocks are left unfolded and
# listed in the record (prepare: unsupported_blocks) instead of refusing the model; on a pure
# decoder stack (TinyLlama) partial=true folds exactly what partial=false folds.
RECIPES = {
    "atlas_v1": {
        "spec": "gptq_awq:method=awq_gptq,actorder=true"
                "+granularity:w=channel,a=token,rule=nc",
        "chip": {"chip_vadd": "rn", "chip_vadd_sites": "residual"},
        "activation_scales": "dynamic per token (run-time amax on chip)",
        "new_decisions": ["NEW: promote run-5.x AWQ+GPTQ+granularity with residual vadd rn",
                          "NEW: recipe declares conditional multi_task_dit exact patch_embedding "
                          "Conv2d BF16 exception (bf16_opt_out_exact records application); "
                          "run-5.2 ref_conv evidence only, chip result pending"],
        "bf16_by_policy": {"multi_task_dit": [
            "observation_encoder.vision_encoder.model.vision_model.embeddings.patch_embedding"]},
    },
    # Historical run-5.x atlas_v1, retained under a distinct name for replay (NEW name).
    "atlas_v1_sq": {
        "spec": "smoothquant:alpha=0.5,po2=true,gain=false,partial=true"
                "+gptq_awq:method=gptq,actorder=true"
                "+granularity:w=channel,a=token,rule=nc",
        "chip": {"chip_vadd": "rn", "chip_vadd_sites": "residual"},
        "activation_scales": "dynamic per token (run-time amax on chip)",
    },
    "atlas_v1_awq": {
        # Historical compatibility alias: no automatic DiT exception, matching run 5.3.
        "spec": "gptq_awq:method=awq_gptq,actorder=true"
                "+granularity:w=channel,a=token,rule=nc",
        "chip": {"chip_vadd": "rn", "chip_vadd_sites": "residual"},
        "activation_scales": "dynamic per token (run-time amax on chip)",
    },
    "atlas_v1_static": {
        "spec": "smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true"
                "+gptq_awq:method=gptq,actorder=true",
        "chip": {"chip_vadd": "rn", "chip_vadd_sites": "residual"},
        "activation_scales": "static per tensor for every calibrated Linear input (no run-time amax "
                             "there); attention operands keep the reference dynamic per-(batch, head) rule",
    },
}

_CURRENT = []


def split(spec: str | None) -> list:
    """'A:k=v+B:k=v' -> ['A:k=v', 'B:k=v']. A '+' separates algorithms only when the text after it
    starts with a registered name (followed by ':' or the end of that part); otherwise it belongs
    to the current value (e.g. smoothquant sites=qkv+fc1)."""
    spec = (spec or "none").strip()
    parts = []
    for tok in spec.split("+"):
        head = tok.strip().partition(":")[0].strip()
        if parts and head not in REGISTRY and head not in RECIPES:
            parts[-1] += "+" + tok
        else:
            parts.append(tok.strip())
    return parts


def parse(spec: str | None) -> tuple[str, dict]:
    """'name:k=v,k2=v2' -> (name, {k: v}); values are int, float, true/false or strings.

    A token without '=' continues the previous value with a comma ('online=down,o' -> 'down,o');
    such a joined value stays a string."""
    spec = (spec or "none").strip()
    name, _, rest = spec.partition(":")
    name = name.strip()
    raw, last = {}, None
    for item in filter(None, (p.strip() for p in rest.split(","))):
        k, eq, v = item.partition("=")
        if not eq:
            if last is None:
                raise ValueError(f"--algo {spec!r}: parameter {item!r} is not k=v")
            raw[last] = (raw[last][0] + "," + item, True)
            continue
        last = k.strip()
        raw[last] = (v.strip(), False)
    params = {k: (v if joined else _value(v)) for k, (v, joined) in raw.items()}
    if name not in REGISTRY:
        raise ValueError(f"unknown algorithm {name!r}; choose from {sorted(REGISTRY)}")
    if name == "none" and params:
        raise ValueError("--algo none takes no parameters")
    return name, params


def recipe_of(spec: str | None):
    """(recipe name, its dict) if `spec` uses a named recipe, else (None, None). Recipes take no
    parameters, and at most one recipe may appear in a spec."""
    names = [p.strip() for p in split(spec) if p.partition(":")[0].strip() in RECIPES]
    if not names:
        return None, None
    if len(names) > 1:
        raise ValueError(f"--algo {spec!r}: more than one recipe ({names})")
    if ":" in names[0]:
        raise ValueError(f"--algo {spec!r}: recipe {names[0].partition(':')[0]!r} takes no parameters; "
                         "write its stack out (docs/algos/atlas_v1.md) to change one")
    return names[0], RECIPES[names[0]]


def expand(spec: str | None) -> str:
    """`spec` with a named recipe replaced by its stack."""
    parts = split(spec)
    return "+".join(RECIPES[p.strip()]["spec"] if p.strip() in RECIPES else p for p in parts)


def parse_stack(spec: str | None) -> list:
    """[(name, params), ...] of every algorithm in `spec` (recipes expanded), in the order written."""
    recipe_of(spec)                                  # validates recipe use
    out = [parse(p) for p in split(expand(spec))]
    if len(out) > 1 and any(n == "none" for n, _ in out):
        raise ValueError(f"--algo {spec!r}: 'none' cannot be stacked")
    return out


def _value(v: str):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


def _make_one(name, params):
    algo = importlib.import_module(REGISTRY[name]).ALGO(**params)
    algo.name = name
    return algo


def make(spec: str | None):
    """The Algo instance for `spec`, an algos.stack.Stack for 'A+B+...', or None for `none`."""
    parsed = parse_stack(spec)
    if len(parsed) == 1:
        name, params = parsed[0]
        return None if REGISTRY[name] is None else _make_one(name, params)
    from algos.stack import Stack

    st = Stack([_make_one(n, p) for n, p in parsed], spec=(spec or "").strip())
    name, rcp = recipe_of(spec)
    if name is not None:
        st.recipe = {"name": name, "spec": rcp["spec"], "expanded": expand(spec),
                     "chip_defaults": dict(rcp["chip"]), "activation_scales": rcp["activation_scales"],
                     "new_decisions": list(rcp.get("new_decisions", [])),
                     "bf16_by_policy": dict(rcp.get("bf16_by_policy", {})),
                     "doc": "docs/algos/atlas_v1.md"}
    return st


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
                   help="NAME[:k=v,...] from src/algos, or several joined by '+' "
                        "(docs/algos/stacking.md; default none: the profiles unchanged); "
                        f"registered: {', '.join(sorted(REGISTRY))}; recipes "
                        f"(docs/algos/atlas_v1.md): {', '.join(RECIPES)}")
    g.add_argument("--calib-n", type=int, default=None,
                   help="calibration samples (windows / frames / observations); default: the algorithm's")
    g.add_argument("--calib-seed", type=int, default=1234,
                   help="seed of the calibration sample (never the eval data, see docs/algos)")
    g.add_argument("--calib-source", choices=calib_sources, default=calib_sources[0],
                   help="calibration data source (model-specific; never overlaps the eval data)")


def none_record(spec, mode=None) -> dict:
    parsed = parse_stack(spec)
    if len(parsed) == 1:
        name, params = parsed[0]
        rec = {"name": name, "params": params, "calibration": None}
    else:
        from algos.stack import order

        name = "stack"
        by = dict(parsed)
        rec = {"name": name, "spec": str(spec).strip(), "order": order([n for n, _ in parsed]),
               "params": by, "calibration": None}
        rname, rcp = recipe_of(spec)
        if rname is not None:
            rec["recipe"] = {"name": rname, "spec": rcp["spec"], "expanded": expand(spec)}
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

    members = getattr(algo, "members", None)     # an algos.stack.Stack
    if members is None:
        Algo.__init__(algo, **algo.params)      # fresh per-mode stats (the instance is reused)
    else:
        algo.reset()
    skip = list(skip or ())
    recipe = getattr(algo, "recipe", None)
    chip_applied = {}
    if recipe and quant_modes.PROFILES[quant_modes.canonical(mode)].base == "chip":
        # the recipe's chip defaults, unless the command line set them (an explicit --chip-vadd wins)
        opts = quant_modes._options_dict(options)
        chip_applied = {k: v for k, v in recipe["chip_defaults"].items() if opts.get(k) is None}
        options = {**opts, **chip_applied}
    bf16 = list(algo.bf16_layers(model, mode) or [])
    # NEW: resolve the canonical recipe's exact model exception before calibration. This
    # Conv2d is outside GPTQ/AWQ's Linear calibration/folds, and must never reach cast hooks.
    exact_bf16 = []
    policy_type = getattr(getattr(model, "config", None), "type", None)
    for name in (recipe or {}).get("bf16_by_policy", {}).get(policy_type, []):
        import torch.nn as nn

        mod = dict(model.named_modules()).get(name)
        if type(mod) is not nn.Conv2d:
            raise ValueError(f"recipe {recipe['name']}: expected exact BF16 Conv2d {name!r}")
        exact_bf16.append(name)
    bf16 = list(dict.fromkeys(bf16 + exact_bf16))
    if algo.needs_calibration and calib is None:
        raise RuntimeError(f"algorithm {algo.name!r} needs calibration data and none was provided")
    loaded = []

    def calibrate(a):
        if not loaded:
            loaded.append(calib())
        batches, run, crec = loaded[0]
        out = a.calibrate(model, batches, run)
        a.calibration = {**crec, **({"result": out} if out else {})}

    if members is None:
        if algo.needs_calibration:
            calibrate(algo)
        prep = algo.prepare(model, mode)
    else:
        # per member in the stack's fixed order: calibrate on the model as the earlier members'
        # prepare left it, then prepare (NEW; docs/algos/stacking.md)
        prep = {}
        for a in members:
            if a.needs_calibration:
                calibrate(a)
            prep[a.name] = a.prepare(model, mode)
    if bf16:
        # the chip profile labels these skips as BF16 opt-outs in its coverage report
        options = {**quant_modes._options_dict(options), "_algo_bf16": tuple(bf16),
                   "_algo_bf16_exact": tuple(exact_bf16)}
    with active(algo):
        rep = quant_modes.quantize(model, mode, skip + [n for n in bf16 if n not in exact_bf16], options)
    rec = algo.record()
    base = quant_modes.PROFILES[quant_modes.canonical(mode)].base
    hooks = algo.overridden_module_hooks()
    if recipe:
        rec["recipe"] = {**recipe, "chip_options_applied": chip_applied}
    rec.update(applied=True, bf16_opt_out=bf16, prepare=prep,
               bf16_opt_out_exact=exact_bf16,
               module_hooks=hooks, module_hooks_applied=base in ("reference", "reference_hw", "chip"))
    if hooks and not rec["module_hooks_applied"]:
        rec["module_hooks_note"] = f"{mode}: module hooks {hooks} act only in the reference modules"
    rep["algo"] = rec
    return rep
