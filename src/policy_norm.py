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
feature whose normalization mode is not IDENTITY must have finite stats under the exact key the
step looks up, with a positive std (MEAN_STD) or range (MIN_MAX, QUANTILES, QUANTILE10).
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

def _steps(pipeline, cls_name):
    return [s for s in getattr(pipeline, "steps", []) if type(s).__name__ == cls_name]


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


def _float_stats(step) -> dict:
    from lerobot.processor.converters import to_tensor

    return {k: {n: to_tensor(v, dtype=torch.float32, device="cpu") for n, v in sub.items()}
            for k, sub in (step.stats or {}).items() if isinstance(sub, dict)}


def stats_sha256(stats: dict) -> str:
    """sha256 of one key's stats, canonically serialized: stat names sorted, each as its shape and
    its float32 values written as exact decimal float64 (NEW)."""
    canon = {n: {"shape": list(t.shape), "values": [float(x) for x in t.double().flatten()]}
             for n, t in sorted(stats.items())}
    return hashlib.sha256(json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_step(step, role) -> tuple[dict, list]:
    """({key: guard record}, problems) for one (un)normalizer step."""
    stats = _float_stats(step)
    keys, problems = {}, []
    for key, mode in sorted(normalized_keys(step).items()):
        rec = {"mode": mode}
        keys[key] = rec
        problem = _check_key(rec, role, key, mode, stats)
        if problem is not None:
            problems.append(problem)
    return keys, problems


def _check_key(rec, role, key, mode, stats) -> str | None:
    """The problem with `key`'s stats under `mode`, or None; fills `rec` (missing stat names, or
    the sha256 and smallest scale of usable stats)."""
    need, scale = REQUIRED.get(mode, (None, None))
    if need is None:
        return f"{role}: {key!r} uses unsupported mode {mode}"
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
    is_std = isinstance(scale, str)
    width = sub[scale] if is_std else sub[scale[1]] - sub[scale[0]]
    if not (width > 0).all():
        return (f"{role}: {key!r} ({mode}) has a non-positive {'std' if is_std else 'range'} in "
                f"{int((width <= 0).sum())} of {width.numel()} entries")
    rec["sha256"] = stats_sha256(sub)
    rec["min_scale"] = float(width.min())
    return None


def check(preprocess, postprocess) -> dict:
    """The guard's findings for built pipelines: per role and key, the mode, the stats sha256 and
    any problem. Every protocol goes through it."""
    rec, problems = {}, []
    for role, pipe, cls in (("pre", preprocess, "NormalizerProcessorStep"),
                            ("post", postprocess, "UnnormalizerProcessorStep")):
        steps = _steps(pipe, cls)
        if not steps:
            problems.append(f"{role}: the pipeline has no {cls}")
        rec[role] = {}
        for step in steps:
            keys, step_problems = check_step(step, role)
            rec[role].update(keys)
            problems += step_problems
    rec["problems"] = problems
    rec["ok"] = not problems
    return rec


def combined_sha256(guard: dict) -> str | None:
    per = {f"{r}:{k}": v.get("sha256") for r in ("pre", "post") for k, v in sorted(guard.get(r, {}).items())}
    if not per or any(v is None for v in per.values()):
        return None
    return hashlib.sha256(json.dumps(per, sort_keys=True).encode()).hexdigest()


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

def probe(preprocess, postprocess, sample, to_batch, rename_map=None, action_shape=None, seed=0) -> dict:
    """Push one real frame through `preprocess` and one action chunk through `postprocess`, then
    back through the preprocessor's normalizer (NEW).

    Each normalized observation must change; the action round trip must be the identity within
    ROUND_TRIP_RTOL. Returns the record; rec["ok"] says whether both held."""
    from lerobot.utils.constants import ACTION

    pre_steps = _steps(preprocess, "NormalizerProcessorStep")
    norm_keys = {k: m for s in pre_steps for k, m in normalized_keys(s).items() if k != ACTION}
    observations = _probe_observations(sample, to_batch(sample, preprocess), norm_keys, rename_map)
    rec = {"observations": observations, "action_round_trip": None}
    ok = all(o["changed"] for o in observations.values() if o["present"])
    if action_shape is not None and pre_steps:
        round_trip = _probe_action_round_trip(pre_steps[0], postprocess, action_shape, seed)
        rec["action_round_trip"] = round_trip
        ok = ok and round_trip["identity"] and (round_trip["unnormalize_changed"] or not round_trip["modes"])
    rec["ok"] = bool(ok)
    return rec


def _probe_observations(sample, batch, norm_keys, rename_map) -> dict:
    """{key: record} of each normalized observation, raw (`sample`) against preprocessed (`batch`)."""
    inv = {v: k for k, v in (rename_map or {}).items()}
    out_rec = {}
    for key, mode in sorted(norm_keys.items()):
        src = inv.get(key, key)
        if src not in sample or key not in batch:
            out_rec[key] = {"mode": mode, "present": False}
            continue
        raw = torch.as_tensor(sample[src]).float().cpu()
        out = batch[key].float().cpu().reshape(raw.shape) if batch[key].numel() == raw.numel() \
            else batch[key].float().cpu()
        changed = out.shape != raw.shape or not torch.allclose(out, raw, rtol=1e-3, atol=1e-3)
        out_rec[key] = {"mode": mode, "present": True, "changed": changed,
                        "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
                        "normalized_mean": float(out.mean()), "normalized_std": float(out.std())}
    return out_rec


def _probe_action_round_trip(normalizer, postprocess, action_shape, seed) -> dict:
    """Unnormalize a seeded model-space action chunk with `postprocess`, then normalize it back
    with a copy of the preprocessor's `normalizer` step."""
    from lerobot.lerobot_types import TransitionKey
    from lerobot.processor.converters import create_transition
    from lerobot.utils.constants import ACTION

    g = torch.Generator().manual_seed(seed)
    chunk = torch.randn((1, 4, *action_shape), generator=g)  # NEW: a model-space chunk
    unnorm = postprocess(chunk.clone()).float().cpu()
    step = copy.deepcopy(normalizer)  # keeps the run's step free of the probe's dtype change
    back = step(create_transition(action=unnorm.clone()))[TransitionKey.ACTION].float().cpu()
    err = float((back - chunk).abs().max())
    scale = float(chunk.abs().max())
    post_modes = {m for s in _steps(postprocess, "UnnormalizerProcessorStep")
                  for k, m in normalized_keys(s).items() if k == ACTION}
    return {"max_abs_err": err, "rtol": ROUND_TRIP_RTOL,
            "identity": err <= ROUND_TRIP_RTOL * max(scale, 1.0),
            "unnormalize_changed": not torch.allclose(unnorm, chunk), "modes": sorted(post_modes),
            "unnormalized_mean": float(unnorm.mean()), "unnormalized_std": float(unnorm.std())}


# ---------------------------------------------------------------------------------------------
# Records and exports
# ---------------------------------------------------------------------------------------------

def export_record(norm: dict) -> dict:
    """What an export carries in manifest.json and policy.json."""
    guard = norm.get("guard") or {}
    return {"protocol": norm["protocol"], "status": norm["status"],
            "stats_source": norm.get("stats_source"),
            "stats_sha256": norm.get("stats_sha256"),
            "stats_sha256_per_key": {r: {k: v.get("sha256") for k, v in guard.get(r, {}).items()}
                                     for r in ("pre", "post")}}


def check_export(manifest: dict, norm: dict) -> None:
    """--from-export: the export's normalization must be the one requested (NEW). An export written
    before this protocol existed carries none and counts as checkpoint / legacy-unverified."""
    have = manifest.get("normalization") or {"protocol": "checkpoint", "status": LEGACY,
                                             "stats_sha256": None}
    want = (norm["protocol"], norm["status"])
    got = (have.get("protocol"), have.get("status"))
    if got != want:
        raise NormalizationError(
            f"--from-export: the export was made with normalization {got[0]} / {got[1]}, this run "
            f"requests {want[0]} / {want[1]}; re-export with --norm-stats {want[0]} or request "
            f"--norm-stats {got[0]}" + (" --allow-unnormalized" if got[1] == LEGACY else ""))
    if have.get("stats_sha256") and norm.get("stats_sha256") and have["stats_sha256"] != norm["stats_sha256"]:
        raise NormalizationError(
            f"--from-export: the export's normalization stats (sha256 {have['stats_sha256'][:12]}) "
            f"differ from this run's ({norm['stats_sha256'][:12]}); the stats source was "
            f"{have.get('stats_source')}")
