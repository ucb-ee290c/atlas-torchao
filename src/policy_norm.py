"""Normalization protocol for LeRobot policies: stats selection, a hard guard and a probe.

LeRobot's (un)normalizer passes a tensor through unchanged when its stats have no entry for the
feature's key (lerobot 0.6.1, processor/normalize_processor.py:330-331: `key not in
self._tensor_stats` returns the tensor). lerobot/smolvla_base's saved processors declare
STATE / ACTION MEAN_STD but store stats only under `so100{,-blue,-red}.buffer.action.*`, so with
the checkpoint's processors SmolVLA silently receives raw joint degrees as state and its actions
are never unnormalized. Everything here exists so that cannot happen silently again.

Protocols (`--norm-stats`):
  checkpoint  the processors exactly as saved with the checkpoint (the historical default).
  dataset     NEW: a zero-shot evaluation protocol for a pretraining checkpoint, mirroring what a
              fine-tune would start from. The normalizer and unnormalizer stats are replaced by the
              evaluated dataset's `LeRobotDatasetMetadata(...).stats` through the same override
              mechanism lerobot_train uses for a fine-tune from a pretrained checkpoint
              (lerobot/scripts/lerobot_train.py:355-377: `normalizer_processor` /
              `unnormalizer_processor` overrides of features, norm_map and stats on
              make_pre_post_processors). The stats' keys follow the observation rename map, as
              lerobot's rollout path does (lerobot/rollout/context.py:461, `rename_stats`), since
              the normalizer runs after the rename step.

The guard (`check`, `require`) runs after the processors are built, for every protocol: every
feature the policy config normalizes (its input / output features under its normalization_mapping,
NEW: not the processors' own saved features, which can omit one) must be covered by the matching
step under the same mode, with finite stats under the exact key the step looks up and a positive
std (MEAN_STD) or range (MIN_MAX, QUANTILES, QUANTILE10); and, the other way round, every feature
a step normalizes must be declared by the config under the same mode (a config IDENTITY is a
mismatch). The probe then checks one real frame, on every key the config requires or a step
transforms, and one action chunk against the mode's formula (in float64) and requires finite
outputs; a tiny but positive scale is judged there, by what it actually produces.
`--allow-unnormalized` (NEW) skips the refusal to reproduce the historical behaviour for
comparison; results then say `normalization: legacy-unverified`.
"""

from __future__ import annotations

import copy
import hashlib
import json

import torch

PROTOCOLS = ("checkpoint", "dataset")
VERIFIED = "verified"
LEGACY = "legacy-unverified"
# Stats each mode reads, and the pair whose difference is its scale (normalize_processor.py:347-417)
REQUIRED = {"MEAN_STD": (("mean", "std"), "std"),
            "MIN_MAX": (("min", "max"), ("min", "max")),
            "QUANTILES": (("q01", "q99"), ("q01", "q99")),
            "QUANTILE10": (("q10", "q90"), ("q10", "q90"))}
ROUND_TRIP_RTOL = 1e-4   # NEW: float32 (x * s + m - m) / (s + 1e-8) stays well inside this
FORMULA_ULPS = 8         # NEW: probe outputs match the formula to this many ulps of their dtype
ROLES = (("pre", "NormalizerProcessorStep"), ("post", "UnnormalizerProcessorStep"))
# NEW: where a transform sits in the ordered chain, and what the chain digest covers per transform
POSITION_FIELDS = ("role", "step", "step_name", "key")
CHAIN_FIELDS = (*POSITION_FIELDS, "mode", "eps", "sha256")


class NormalizationError(RuntimeError):
    pass


def add_args(ap) -> None:
    g = ap.add_argument_group("normalization (src/policy_norm.py)")
    g.add_argument("--norm-stats", choices=PROTOCOLS, default="checkpoint",
                   help="checkpoint: the processors' saved stats; dataset (NEW): override the "
                        "normalizer / unnormalizer stats with the evaluated dataset's stats, as a "
                        "lerobot fine-tune from this checkpoint would start")
    g.add_argument("--allow-unnormalized", action="store_true",
                   help="NEW: do not refuse when a normalized feature has no stats (the historical "
                        "silent pass-through); results are marked normalization: legacy-unverified")


def check_args(ap, args) -> None:
    if args.allow_unnormalized and args.norm_stats != "checkpoint":
        ap.error("--allow-unnormalized reproduces the historical checkpoint behaviour; it needs "
                 "--norm-stats checkpoint")


# ---------------------------------------------------------------------------------------------
# Building the processors
# ---------------------------------------------------------------------------------------------

def dataset_stats(dataset, rename_map=None):
    """(stats renamed onto the policy's keys, source record) for `dataset`."""
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.processor import rename_stats

    meta = LeRobotDatasetMetadata(dataset)
    root = getattr(meta, "root", None)
    source = {"kind": "dataset", "repo_id": dataset, "revision": getattr(meta, "revision", None),
              # the resolved snapshot (a commit sha for a hub cache), when the root is one
              "snapshot": root.name if root is not None and root.parent.name == "snapshots" else None,
              "rename_map": dict(rename_map or {})}
    return rename_stats(meta.stats, rename_map or {}), source


def overrides(cfg, stats):
    """The normalizer / unnormalizer overrides lerobot_train passes for a fine-tune from a
    pretrained checkpoint (lerobot_train.py:355-377): features, norm_map and stats."""
    pre = {"normalizer_processor": {"features": {**cfg.input_features, **cfg.output_features},
                                    "norm_map": cfg.normalization_mapping,
                                    "stats": stats}}
    post = {"unnormalizer_processor": {"features": cfg.output_features,
                                       "norm_map": cfg.normalization_mapping,
                                       "stats": stats}}
    return pre, post


def merge_overrides(base: dict, extra: dict) -> dict:
    out = {k: dict(v) for k, v in base.items()}
    for k, v in extra.items():
        out.setdefault(k, {}).update(v)
    return out


# ---------------------------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------------------------

def _indexed_steps(pipeline, cls_name):
    """[(position in the pipeline, step)] of the pipeline's `cls_name` steps, in order."""
    return [(i, s) for i, s in enumerate(getattr(pipeline, "steps", [])) if type(s).__name__ == cls_name]


def _steps(pipeline, cls_name):
    return [s for _, s in _indexed_steps(pipeline, cls_name)]


def _mode_name(mode) -> str:
    return getattr(mode, "value", str(mode))


def normalized_keys(step) -> dict:
    """{key the step looks up: mode name} for every feature the step does not leave IDENTITY.

    Observations are looked up under their feature key (subject to normalize_observation_keys);
    the action always under "action" (normalize_processor.py:288-298)."""
    from lerobot.configs import FeatureType
    from lerobot.utils.constants import ACTION

    out = {}
    for key, feat in step.features.items():
        mode = _mode_name(step.norm_map.get(feat.type, "IDENTITY"))
        if mode == "IDENTITY":
            continue
        if feat.type == FeatureType.ACTION:
            out[ACTION] = mode
        elif step.normalize_observation_keys is None or key in step.normalize_observation_keys:
            out[key] = mode
    return out


def declared_modes(cfg) -> dict:
    """{role: {key: mode}} the policy config declares for the normalizer ("pre") and unnormalizer
    ("post"), IDENTITY included, from the config rather than the processors' saved features (NEW).

    The config's feature keys are the policy-side names, i.e. already after the observation rename
    map, which is what the normalizer looks up since it runs after the rename step."""
    from lerobot.configs import FeatureType
    from lerobot.utils.constants import ACTION

    mapping = {getattr(t, "value", t): _mode_name(m) for t, m in cfg.normalization_mapping.items()}
    pre, post = {}, {}
    for key, feat in {**cfg.input_features, **cfg.output_features}.items():
        mode = mapping.get(feat.type.value, "IDENTITY")
        if feat.type == FeatureType.ACTION:
            pre[ACTION] = post[ACTION] = mode
        else:
            pre[key] = mode
    return {"pre": pre, "post": post}


def required_keys(cfg) -> dict:
    """{role: {key: mode}} every normalizer ("pre") and unnormalizer ("post") must cover: the
    config's declared features that are not IDENTITY."""
    return {role: {k: m for k, m in keys.items() if m != "IDENTITY"}
            for role, keys in declared_modes(cfg).items()}


def _float_stats(step) -> dict:
    from lerobot.processor.converters import to_tensor

    return {k: {n: to_tensor(v, dtype=torch.float32, device="cpu") for n, v in sub.items()}
            for k, sub in (step.stats or {}).items() if isinstance(sub, dict)}


def _canonical_stats(stats: dict) -> dict:
    """Stat names sorted, each as its shape and its float32 values as exact decimal float64."""
    return {n: {"shape": list(t.shape), "values": [float(x) for x in t.double().flatten()]}
            for n, t in sorted(stats.items())}


def _sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def stats_sha256(stats: dict) -> str:
    """sha256 of one key's stats, canonically serialized (NEW)."""
    return _sha256(_canonical_stats(stats))


def transform_sha256(mode: str, eps: float, stats: dict) -> str:
    """sha256 of one key's whole transform: the mode, the step's eps and the stats (NEW). lerobot's
    (un)normalizer has no clipping or other parameter (normalize_processor.py:347-417)."""
    return _sha256({"mode": mode, "eps": float(eps), "stats": _canonical_stats(stats)})


def check_step(step, role) -> tuple[dict, list]:
    """({key: guard record}, problems) for one (un)normalizer step."""
    stats = _float_stats(step)
    keys, problems = {}, []
    for key, mode in sorted(normalized_keys(step).items()):
        rec = {"mode": mode, "eps": float(step.eps)}
        keys[key] = rec
        problem = _check_key(rec, role, key, mode, stats)
        if problem is not None:
            problems.append(problem)
    return keys, problems


def _check_key(rec, role, key, mode, stats) -> str | None:
    """The problem with `key`'s stats under `mode`, or None; fills `rec` (missing stat names, or
    the stats / transform sha256 and smallest scale of usable stats)."""
    if mode not in REQUIRED:
        return f"{role}: {key!r} uses unsupported mode {mode}"
    need = REQUIRED[mode][0]
    sub = stats.get(key)
    if sub is None:
        rec["missing"] = list(need)
        return f"{role}: no stats for {key!r} ({mode}); stored keys: {sorted(stats)[:8]}"
    missing = [n for n in need if n not in sub]
    if missing:
        rec["missing"] = missing
        return f"{role}: {key!r} ({mode}) lacks {missing}"
    non_finite = [n for n in need if not torch.isfinite(sub[n]).all()]
    if non_finite:
        return f"{role}: {key!r} ({mode}) has non-finite {non_finite}"
    problem = _scale_problem(role, key, mode, sub)
    if problem is not None:
        return problem
    rec["sha256"] = stats_sha256(sub)
    rec["transform_sha256"] = transform_sha256(mode, rec["eps"], sub)
    rec["min_scale"] = float(_width(mode, sub).min())
    return None


def _width(mode, sub):
    scale = REQUIRED[mode][1]
    return sub[scale] if isinstance(scale, str) else sub[scale[1]] - sub[scale[0]]


def _scale_problem(role, key, mode, sub) -> str | None:
    """A zero or negative std / range. A tiny positive one is left to the probe: whether it
    overflows depends on the inputs, and the probe checks real outputs for finiteness."""
    width = _width(mode, sub)
    if (width > 0).all():
        return None
    what = "std" if isinstance(REQUIRED[mode][1], str) else "range"
    return (f"{role}: {key!r} ({mode}) has a non-positive {what} in "
            f"{int((width <= 0).sum())} of {width.numel()} entries")


def _coverage_problems(role, declared: dict, covered: dict) -> list:
    """Both directions, per key: each key the config declares non-IDENTITY must be covered by a
    step under that mode, and each key a step transforms must be declared by the config under the
    step's mode (NEW: a config IDENTITY or undeclared key that a step normalizes is refused)."""
    problems = []
    for key in sorted(set(declared) | set(covered)):
        problem = _coverage_problem(role, key, declared.get(key), covered.get(key, {}).get("mode"))
        if problem is not None:
            problems.append(problem)
    return problems


def _coverage_problem(role, key, config_mode, step_mode) -> str | None:
    if step_mode is None:
        if config_mode in (None, "IDENTITY"):
            return None
        return f"{role}: the config normalizes {key!r} ({config_mode}) but no step covers it"
    if config_mode is None:
        return f"{role}: a step normalizes {key!r} ({step_mode}) but the config does not declare it"
    if step_mode != config_mode:
        return f"{role}: {key!r} is {step_mode} in the processor, {config_mode} in the config"
    return None


def check(preprocess, postprocess, cfg) -> dict:
    """The guard's findings for built pipelines against the policy config `cfg`: the ordered list
    of transforms (NEW: one entry per step and key, so several steps cannot hide behind one key),
    per role and key the mode, eps, stats / transform sha256, and any problem. Every protocol goes
    through it."""
    declared = declared_modes(cfg)
    rec = {"required": required_keys(cfg), "declared": declared, "transforms": []}
    problems = []
    for (role, cls), pipe in zip(ROLES, (preprocess, postprocess)):
        transforms, role_problems = _check_role(role, cls, pipe)
        rec["transforms"] += transforms
        rec[role] = _per_key(transforms)
        problems += role_problems + _coverage_problems(role, declared[role], rec[role])
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def _check_role(role, cls, pipe) -> tuple[list, list]:
    """(ordered transforms, problems) of one pipeline's `cls` steps."""
    steps = _indexed_steps(pipe, cls)
    if not steps:
        return [], [f"{role}: the pipeline has no {cls}"]
    transforms, problems = [], []
    for index, step in steps:
        keys, step_problems = check_step(step, role)
        transforms += [{"role": role, "step": index, "step_name": cls, "key": k, **r}
                       for k, r in keys.items()]
        problems += step_problems
    return transforms, problems + _duplicate_problems(role, transforms)


def _duplicate_problems(role, transforms) -> list:
    """A key (un)normalized by more than one step: lerobot would compose them, which neither the
    per-key record nor the probe's formula describes, so it is refused (NEW)."""
    steps_by_key = {}
    for t in transforms:
        steps_by_key.setdefault(t["key"], []).append(t["step"])
    return [f"{role}: {key!r} is normalized by {len(steps)} steps (pipeline positions {steps}); "
            "one step per key is supported"
            for key, steps in sorted(steps_by_key.items()) if len(steps) > 1]


def _per_key(transforms) -> dict:
    """{key: record} of one role, the first step's for a key several steps normalize (refused)."""
    out = {}
    for t in transforms:
        out.setdefault(t["key"], {k: v for k, v in t.items() if k not in POSITION_FIELDS})
    return out


def combined_sha256(guard: dict) -> str | None:
    """The historical stats-only digest over every role and key's stats sha256; None when any key
    lacks it."""
    per = {f"{r}:{k}": v.get("sha256") for r, _ in ROLES for k, v in sorted(guard.get(r, {}).items())}
    if not per or any(v is None for v in per.values()):
        return None
    return hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()


def transform_chain(guard: dict) -> list:
    """The ordered transforms as recorded and hashed: step position and name, key, mode, eps and
    stats sha256 (NEW)."""
    return [{f: t.get(f) for f in CHAIN_FIELDS} for t in guard.get("transforms", [])]


def chain_sha256(guard: dict) -> str | None:
    """Digest of the ordered transform chain (NEW: replaces a per-key digest, which an extra step
    normalizing an already covered key left unchanged); None when any transform lacks stats."""
    chain = transform_chain(guard)
    if not chain or any(t["sha256"] is None for t in chain):
        return None
    return _sha256(chain)


def require(guard: dict, protocol: str, allow_unnormalized: bool) -> str:
    """The run's normalization status; raises NormalizationError when the guard failed and the
    escape hatch is not set."""
    if allow_unnormalized:
        print(_legacy_warning(guard["problems"]), flush=True)
        return LEGACY
    if guard["problems"]:
        raise NormalizationError(
            f"--norm-stats {protocol}: the processors would silently leave normalized features "
            "unnormalized:\n  " + "\n  ".join(guard["problems"])
            + "\nUse --norm-stats dataset to normalize with the evaluated dataset's stats "
              "(or --allow-unnormalized to reproduce the historical, unverified behaviour).")
    return VERIFIED


def _legacy_warning(problems: list) -> str:
    detail = f" ({len(problems)} guard problem(s): " + "; ".join(problems) + ")" if problems else ""
    return ("WARNING: --allow-unnormalized: normalization is NOT verified; results are marked "
            f"normalization: {LEGACY}" + detail)


# ---------------------------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------------------------

def probe(preprocess, postprocess, sample, to_batch, required, rename_map=None, action_shape=None,
          seed=0) -> dict:
    """Push one real frame through `preprocess` and one action chunk through `postprocess`, then
    back through the preprocessor's normalizer (NEW).

    Every observation `required` (check()'s rec["required"]) normalizes, and every one a
    normalizer step actually transforms whatever the config says (NEW), must be in the frame, and
    its output must be finite and match the mode's formula on the step's stats; the unnormalized
    actions likewise, and the action round trip must be the identity within ROUND_TRIP_RTOL. An
    identity transform (mean 0, std 1) passes: what is checked is the formula, not movement.
    Returns the record; rec["ok"] says whether all of it held."""
    from lerobot.utils.constants import ACTION

    pre_steps = _steps(preprocess, "NormalizerProcessorStep")
    obs_modes = {k: m for k, m in _probe_modes(required["pre"], pre_steps).items() if k != ACTION}
    observations = _probe_observations(sample, to_batch(sample, preprocess), obs_modes, pre_steps,
                                       rename_map)
    rec = {"observations": observations, "action_round_trip": None}
    ok = all(o["ok"] for o in observations.values())
    if action_shape is not None and pre_steps:
        round_trip = _probe_action_round_trip(pre_steps[-1], postprocess, action_shape, seed)
        rec["action_round_trip"] = round_trip
        ok = ok and round_trip["ok"]
    rec["ok"] = bool(ok)
    return rec


def _probe_modes(required: dict, steps) -> dict:
    """{key: mode} to probe: the config's required keys, overridden by the mode of the last step
    that transforms each key, so that every transformed key is probed under its real mode."""
    modes = dict(required)
    for step in steps:
        modes.update(normalized_keys(step))
    return modes


def _covering(steps, key):
    """(the last of `steps` that normalizes `key`, its float32 stats for `key`), or (None, None)."""
    for step in reversed(steps):
        if key in normalized_keys(step):
            return step, _float_stats(step).get(key)
    return None, None


def _usable(stats, mode) -> bool:
    return stats is not None and mode in REQUIRED and all(n in stats for n in REQUIRED[mode][0])


def _affine(mode, stats, eps, inverse):
    """(offset, denom, scale, shift) of lerobot's transform (normalize_processor.py:347-417):
    normalize is scale * (x - offset) / denom + shift, unnormalize its inverse. MEAN_STD adds eps
    to the std when normalizing only."""
    lo, hi = (stats[n].double() for n in REQUIRED[mode][0])
    if mode == "MEAN_STD":
        return lo, hi if inverse else hi + eps, 1.0, 0.0
    return lo, hi - lo, 2.0, -1.0


def formula_check(x, y, mode, stats, eps, inverse=False) -> dict:
    """Compare a processor's output `y` with the formula applied to its input `x` in float64,
    within FORMULA_ULPS units of `y`'s dtype on the magnitudes involved (NEW)."""
    offset, denom, scale, shift = _affine(mode, stats, eps, inverse)
    x = x.double()
    if inverse:
        want = (x - shift) / scale * denom + offset
        size = (x.abs() + abs(shift)) / scale * denom.abs() + offset.abs()
    else:
        want = scale * (x - offset) / denom + shift
        size = scale * (x.abs() + offset.abs()) / denom.abs() + abs(shift)
    err = (y.double() - want).abs()
    bound = FORMULA_ULPS * torch.finfo(y.dtype).eps * (want.abs() + size) + 1e-12
    return {"finite": bool(torch.isfinite(y).all()), "max_formula_err": float(err.max()),
            "matches": bool((err <= bound).all())}


def _probe_observations(sample, batch, modes, steps, rename_map) -> dict:
    """{key: record} of each required observation, raw (`sample`) against preprocessed (`batch`);
    a required key missing from either is a failure."""
    inv = {v: k for k, v in (rename_map or {}).items()}
    out_rec = {}
    for key, mode in sorted(modes.items()):
        src = inv.get(key, key)
        if src not in sample or key not in batch:
            out_rec[key] = {"mode": mode, "present": False, "ok": False}
            continue
        out_rec[key] = _probe_observation(torch.as_tensor(sample[src]), batch[key], mode,
                                          *_covering(steps, key))
    return out_rec


def _probe_observation(raw, out, mode, step, stats) -> dict:
    raw, out = raw.float().cpu(), out.detach().cpu()
    rec = {"mode": mode, "present": True, "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
           "normalized_mean": float(out.float().mean()), "normalized_std": float(out.float().std())}
    if out.numel() != raw.numel():
        return {**rec, "ok": False, "reason": f"shape {tuple(out.shape)} vs raw {tuple(raw.shape)}"}
    if not _usable(stats, mode):
        return {**rec, "ok": False, "reason": "no usable stats"}
    found = formula_check(raw, out.reshape(raw.shape), mode, _image_stats(stats, raw), step.eps)
    return {**rec, **found, "ok": found["finite"] and found["matches"]}


def _image_stats(stats, raw):
    """Flat (C,) stats as (C, 1, 1) over a (C, H, W) image, as lerobot reshapes visual stats."""
    if raw.ndim != 3:
        return stats
    return {n: t.view(-1, 1, 1) if t.ndim == 1 and t.numel() == raw.shape[0] else t
            for n, t in stats.items()}


def _probe_action_round_trip(normalizer, postprocess, action_shape, seed) -> dict:
    """Unnormalize a seeded model-space action chunk with `postprocess` and check it against the
    formula, then normalize it back with a copy of the preprocessor's `normalizer` step."""
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.converters import create_transition

    g = torch.Generator().manual_seed(seed)
    chunk = torch.randn((1, 4, *action_shape), generator=g)  # NEW: a model-space chunk
    unnorm = postprocess(chunk.clone()).cpu()
    step = copy.deepcopy(normalizer)  # keeps the run's step free of the probe's dtype change
    back = step(create_transition(action=unnorm.clone().float()))[TransitionKey.ACTION].float().cpu()
    err = float((back - chunk).abs().max())
    rec = {"max_abs_err": err, "rtol": ROUND_TRIP_RTOL,
           "identity": err <= ROUND_TRIP_RTOL * max(float(chunk.abs().max()), 1.0),
           "finite": bool(torch.isfinite(unnorm).all() and torch.isfinite(back).all()),
           "unnormalized_mean": float(unnorm.float().mean()),
           "unnormalized_std": float(unnorm.float().std())}
    rec.update(_unnormalize_check(postprocess, chunk, unnorm))
    rec["ok"] = rec["identity"] and rec["finite"] and rec["matches"]
    return rec


def _unnormalize_check(postprocess, chunk, unnorm) -> dict:
    """The unnormalizer's output against the formula; an action the config leaves IDENTITY (no
    covering step) must come back unchanged."""
    from lerobot.utils.constants import ACTION

    step, stats = _covering(_steps(postprocess, "UnnormalizerProcessorStep"), ACTION)
    if step is None:
        return {"modes": [], "matches": bool(torch.equal(unnorm.float(), chunk))}
    mode = normalized_keys(step)[ACTION]
    if not _usable(stats, mode):
        return {"modes": [mode], "matches": False, "reason": "no usable stats"}
    found = formula_check(chunk, unnorm, mode, stats, step.eps, inverse=True)
    return {"modes": [mode], "matches": found["matches"], "max_formula_err": found["max_formula_err"]}


# ---------------------------------------------------------------------------------------------
# Records and exports
# ---------------------------------------------------------------------------------------------

RECORD_VERSION = 3   # 2: the transform (mode, eps) beside the stats; NEW in 3: the ordered chain


def export_record(norm: dict) -> dict:
    """What an export carries in manifest.json and policy.json. `stats_sha256*` are kept as in
    record version 1; `transform` adds the mode and eps per key, and `transforms` the ordered chain
    that `transform_sha256` digests (record version 3)."""
    guard = norm.get("guard") or {}
    per_role = {r: guard.get(r, {}) for r, _ in ROLES}
    return {"record_version": RECORD_VERSION,
            "protocol": norm["protocol"], "status": norm["status"],
            "stats_source": norm.get("stats_source"),
            "stats_sha256": norm.get("stats_sha256"),
            "stats_sha256_per_key": {r: {k: v.get("sha256") for k, v in keys.items()}
                                     for r, keys in per_role.items()},
            "transform_sha256": norm.get("transform_sha256"),
            "transform": {r: {k: {"mode": v.get("mode"), "eps": v.get("eps"),
                                  "sha256": v.get("transform_sha256")} for k, v in keys.items()}
                          for r, keys in per_role.items()},
            "transforms": transform_chain(guard)}


LEGACY_EXPORT = {"protocol": "checkpoint", "status": LEGACY, "stats_sha256": None}


def check_export(manifest: dict, norm: dict) -> None:
    """--from-export: the export's normalization must be the one requested (NEW). An export written
    before this protocol existed carries none and counts as checkpoint / legacy-unverified; a
    verified export must carry the transform chain digest (record version 3) and match it."""
    have = manifest.get("normalization") or LEGACY_EXPORT
    _check_export_protocol(have, norm)
    _check_export_stats(have, norm)
    if have["status"] != LEGACY:
        _check_export_transform(have, norm)


def _check_export_transform(have: dict, norm: dict) -> None:
    version = have.get("record_version", 1)
    if not have.get("transform_sha256") or version < RECORD_VERSION:
        raise NormalizationError(
            "--from-export: the export does not record its ordered normalization transforms "
            f"(record version {version}, need {RECORD_VERSION}); re-export it")
    if norm.get("transform_sha256") and have["transform_sha256"] != norm["transform_sha256"]:
        raise NormalizationError(
            f"--from-export: the export's normalization transform (sha256 "
            f"{have['transform_sha256'][:12]}, modes {_modes(have)}) differs from this run's "
            f"({norm['transform_sha256'][:12]}); the stats source was {have.get('stats_source')}")


def _check_export_protocol(have: dict, norm: dict) -> None:
    want = (norm["protocol"], norm["status"])
    got = (have.get("protocol"), have.get("status"))
    if got != want:
        raise NormalizationError(
            f"--from-export: the export was made with normalization {got[0]} / {got[1]}, this run "
            f"requests {want[0]} / {want[1]}; re-export with --norm-stats {want[0]} or request "
            f"--norm-stats {got[0]}" + (" --allow-unnormalized" if got[1] == LEGACY else ""))


def _check_export_stats(have: dict, norm: dict) -> None:
    if have.get("stats_sha256") and norm.get("stats_sha256") and have["stats_sha256"] != norm["stats_sha256"]:
        raise NormalizationError(
            f"--from-export: the export's normalization stats (sha256 {have['stats_sha256'][:12]}) "
            f"differ from this run's ({norm['stats_sha256'][:12]}); the stats source was "
            f"{have.get('stats_source')}")


def _modes(record: dict) -> dict:
    return {f"{r}:{k}": v.get("mode") for r, keys in (record.get("transform") or {}).items()
            for k, v in keys.items()}
