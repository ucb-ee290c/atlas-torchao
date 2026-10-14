"""NEW: frozen native-action preflight, export replay recorder and offline analysis.

Preflight reads configuration and parquet metadata only, never model weights or videos.
Recording reuses the existing policy/export inference implementation without fitting.
Physical units and controller semantics remain unknown unless explicitly supplied.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

SCHEMA = "atlas.action_characterization.v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_export_config(exported, config_path, policy_id):
    """Names may alias; exact configuration and later weight hashes establish identity."""
    model = exported.get("model") or {}
    checkpoint = model.get("checkpoint") or {}
    if model.get("kind") != "lerobot_policy":
        raise ValueError("export must identify a LeRobot policy")
    if checkpoint.get("config_json_sha256") != digest(config_path):
        raise ValueError("export checkpoint configuration fingerprint differs or is missing")
    files = checkpoint.get("files")
    if not files or any(not row.get("sha256") for row in files.values()):
        raise ValueError("export original checkpoint weight fingerprints required")
    return {"requested_policy_id": policy_id, "exported_model_id": model.get("id"),
            "configuration_sha256": digest(config_path),
            "equivalence": "configuration verified; exact original weights checked before inference"}


def command_record(command, cwd=None):
    try:
        value = subprocess.check_output(command, cwd=cwd, text=True, stderr=subprocess.STDOUT).strip()
        return {"value": value, "unknown_reason": None}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"value": None, "unknown_reason": type(error).__name__ + ": " + str(error)}


def validate_checkpoint_weights(checkpoint, exported):
    actual = checkpoint.get("files")
    expected = (exported.get("model", {}).get("checkpoint") or {}).get("files")
    if not actual or not expected or any(not row.get("sha256") for row in [*actual.values(), *expected.values()]):
        raise ValueError("original checkpoint weight fingerprints required for paired baseline")
    if {name: row["sha256"] for name, row in actual.items()} != {
            name: row["sha256"] for name, row in expected.items()}:
        raise ValueError("BF16 checkpoint weights differ from export provenance")


def runtime_identity(device):
    """Fresh read-only execution identity; unavailable fields carry an explicit reason."""
    import ref_loader
    import torch

    root = Path(__file__).resolve().parent.parent
    props = torch.cuda.get_device_properties(torch.device(device))
    uuid = getattr(props, "uuid", None)
    source_status = command_record(["git", "status", "--porcelain"], cwd=root)
    return {"host": platform.node(), "platform": platform.platform(), "python": sys.version,
            "python_executable": sys.executable, "cuda_runtime": torch.version.cuda,
            "selected_device": str(device), "selected_device_name": props.name,
            "selected_device_uuid": {"value": str(uuid) if uuid else None,
                                     "unknown_reason": None if uuid else "Torch properties expose no UUID"},
            "gpu_inventory": command_record(["nvidia-smi", "--query-gpu=uuid,name,driver_version,pci.bus_id",
                                              "--format=csv,noheader"]),
            "source_status": source_status,
            "source_dirty": bool(source_status["value"]) if source_status["value"] is not None else None,
            "reference_submodules": {name: {"expected": expected, "actual": ref_loader.head(name),
                "status": command_record(["git", "status", "--porcelain"], cwd=ref_loader.repo_path(name))}
                for name, expected in ref_loader.PINS.items()},
            "environment": {key: os.environ.get(key) for key in (
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "HF_HOME", "HF_HUB_CACHE",
                "HF_DATASETS_CACHE", "LEROBOT_HOME", "ATLAS_REF_ALLOW_UNPINNED", "OPENPI_DIR",
                "SMOLVLA_QUANT_ROOT", "PI0_QUANT_ROOT", "ATLAS_REF_ROOT")}}


def calibration_frames(record, episodes):
    """Resolve export's flat calibration indices to actual episode/frame IDs."""
    lengths = {int(row["episode_index"]): int(row["length"]) for row in episodes}
    pool = record.get("episodes")
    indices = record.get("frame_indices")
    if not pool or not indices or not record.get("data_sha256"):
        raise ValueError("export lacks exact calibration episodes/frame indices/data digest")
    boundaries, total = [], 0
    # algo_calib.lerobot_frames sorts the episode pool before LeRobotDataset construction.
    for ep in sorted(pool):
        if ep not in lengths:
            raise ValueError("calibration episode missing from dataset metadata")
        boundaries.append((int(ep), total, total + lengths[ep]))
        total += lengths[ep]
    out = []
    for index in indices:
        if index < 0 or index >= total:
            raise ValueError("calibration frame outside metadata bounds")
        ep, start, _ = next(row for row in boundaries if row[1] <= index < row[2])
        out.append({"episode_id": ep, "frame_index": int(index - start)})
    return out


def balanced_plan(episodes, *, count, per_episode, seed, excluded=()):
    if count <= 0 or per_episode <= 0:
        raise ValueError("episode and frame counts must be positive")
    eligible = [row for row in episodes if int(row["episode_index"]) not in set(excluded)
                and int(row["length"]) >= per_episode]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    if len(eligible) < count:
        raise ValueError("insufficient eligible episodes after calibration exclusion")
    chosen = sorted(eligible[:count], key=lambda row: int(row["episode_index"]))
    frames, offset = [], 0
    for row in chosen:
        ep, length = int(row["episode_index"]), int(row["length"])
        for k in range(per_episode):
            # One random frame per equal temporal stratum; no invented task-phase labels.
            frame = rng.randrange(k * length // per_episode, (k + 1) * length // per_episode)
            frames.append({"episode_id": ep, "frame_index": frame,
                           "source_index": int(row["dataset_from_index"]) + frame,
                           "dataset_local_index": offset + frame,
                           "sampler_seed": seed + int(row["dataset_from_index"]) + frame})
        offset += length
    return frames


def preflight(args):
    import pandas as pd

    root, config_path = Path(args.dataset_root), Path(args.model_config)
    config = json.loads(config_path.read_text())
    info = json.loads((root / "meta/info.json").read_text())
    if config.get("type") != "smolvla":
        raise ValueError("first recorder supports SmolVLA only; other policy adapters need validation")
    dim = config["output_features"]["action"]["shape"][0]
    if info["features"]["action"]["shape"] != [dim]:
        raise ValueError("checkpoint and dataset action dimensions differ")
    names = info["features"]["action"].get("names")
    if not isinstance(names, list) or len(names) != dim:
        raise ValueError("native action channel names required")
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    matching = tasks[tasks["task_index"] == args.task_id]
    if len(matching) != 1:
        raise ValueError("task ID does not resolve uniquely")
    task_text = str(matching.index[0])
    episode_files = sorted((root / "meta/episodes").rglob("*.parquet"))
    if not episode_files:
        raise ValueError("episode metadata unavailable")
    all_episodes = pd.concat([pd.read_parquet(path) for path in episode_files]).to_dict("records")
    eligible = [row for row in all_episodes if task_text in list(row["tasks"])]
    actual_export = None
    if args.export_dir:
        path = Path(args.export_dir) / "manifest.json"
        exported = json.loads(path.read_text())
        identity = validate_export_config(exported, config_path, args.policy_id)
        source = exported.get("provenance", {}).get("lerobot_dataset")
        if source != args.dataset_id:
            raise ValueError("export calibration dataset identity differs or is missing")
        calibration = {"status": "existing_export", "record": exported.get("calibration") or {},
            "data_sha256_verified": False,
            "disjointness_status": "provisional until exact preprocessed calibration data digest is verified"}
        calibration["frames"] = calibration_frames(calibration["record"], all_episodes)
        actual_export = {"manifest_sha256": digest(path), "manifest": exported,
                         "directory_hint": str(Path(args.export_dir).resolve()), "model_identity": identity}
    else:
        calibration = {"status": "planned_unfitted", "frames": [],
                       "note": "No fitting authorized; bind an existing export before recording."}
    excluded = {row["episode_id"] for row in calibration["frames"]}
    frames = balanced_plan(eligible, count=args.episodes, per_episode=args.frames_per_episode,
                           seed=args.seed, excluded=excluded)
    eval_episodes = sorted({row["episode_id"] for row in frames})
    if not actual_export:
        remaining = [row for row in all_episodes if int(row["episode_index"]) not in eval_episodes]
        # Planned provenance only; this does not run a calibration forward or change weights.
        pool = [(int(row["episode_index"]), k) for row in remaining for k in range(int(row["length"]))]
        if len(pool) < args.calibration_frames:
            raise ValueError("insufficient disjoint proposed calibration frames")
        selected = random.Random(args.calibration_seed).sample(pool, args.calibration_frames)
        calibration["frames"] = [dict(zip(("episode_id", "frame_index"), pair)) for pair in selected]
    calib_episodes = sorted({row["episode_id"] for row in calibration["frames"]})
    if set(eval_episodes) & set(calib_episodes):
        raise ValueError("evaluation and provisional calibration episodes overlap")
    assets = [root / "meta/info.json", root / "meta/stats.json", root / "meta/tasks.parquet", *episode_files]
    # Source frame/task verification reads parquet scalar columns, never image/video payloads.
    source_files = sorted({root / info["data_path"].format(chunk_index=int(row["data/chunk_index"]),
                    file_index=int(row["data/file_index"])) for row in eligible
                    if int(row["episode_index"]) in eval_episodes})
    scalars = pd.concat([pd.read_parquet(path, columns=["episode_index", "frame_index", "index", "task_index"])
                         for path in source_files]).set_index("index")
    for frame in frames:
        row = scalars.loc[frame["source_index"]]
        if (int(row["episode_index"]), int(row["frame_index"]), int(row["task_index"])) != (
                frame["episode_id"], frame["frame_index"], args.task_id):
            raise ValueError("planned frame IDs/task disagree with source parquet")
        frame["task_id"] = args.task_id
    payload = {"schema": SCHEMA, "new_decisions": ["NEW: balanced temporal strata and frozen action recorder"],
        "policy_id": args.policy_id, "dataset_id": args.dataset_id,
        "model_config_sha256": digest(config_path), "model_config": config,
        "model_metadata_sha256": {path.name: digest(path) for path in sorted(config_path.parent.glob("*.json"))},
        "model_root_hint": str(config_path.parent.resolve()), "dataset_root_hint": str(root.resolve()),
        "dataset_assets_sha256": {str(path.relative_to(root)): digest(path) for path in [*assets, *source_files]},
        "task": {"id": args.task_id, "text": task_text}, "seed": args.seed,
        "sampling": {"episodes": eval_episodes, "frames_per_episode": args.frames_per_episode,
                     "chunks": len(frames), "training_holdout_established": False,
                     "method": "one seeded frame per equal temporal stratum; chunks may overlap"},
        "input_fingerprint_scope": {"metadata_and_selected_parquet": "sha256 verified",
            "video_bytes": "not fingerprinted during metadata-only preflight",
            "decoded_preprocessed_observations": "frozen by per-frame input_sha256 during recording"},
        "action_contract": {"channels": names, "dimension": dim, "returned_horizon": config["chunk_size"],
            "normalization": config.get("normalization_mapping"), "dataset_fps": info["fps"],
            "units": ["unknown"] * dim, "coordinates": "native joint/gripper channels",
            "physical_controller_mapping_validated": False, "executed_prefix": None,
            "flow_steps": config["num_steps"]},
        "calibration": calibration, "calibration_episode_ids": calib_episodes,
        "calibration_disjointness_status": ("provisional_pending_exact_export_data_digest"
                                            if actual_export else "planned_unfitted_only"),
        "export": actual_export, "observations": frames}
    payload["content_sha256"] = canonical_digest(payload)
    write_new(args.out, payload)
    return payload


def validate_manifest(manifest, root, model_root):
    expected = manifest.get("content_sha256")
    if canonical_digest({k: v for k, v in manifest.items() if k != "content_sha256"}) != expected:
        raise ValueError("input manifest content digest mismatch")
    if digest(Path(model_root) / "config.json") != manifest["model_config_sha256"]:
        raise ValueError("model configuration fingerprint mismatch")
    for name, expected_hash in manifest["model_metadata_sha256"].items():
        if digest(Path(model_root) / name) != expected_hash:
            raise ValueError("model metadata fingerprint mismatch: " + name)
    for name, expected_hash in manifest["dataset_assets_sha256"].items():
        if digest(Path(root) / name) != expected_hash:
            raise ValueError("dataset asset fingerprint mismatch: " + name)


def tensor_digest(batch):
    import torch

    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(str((str(t.dtype), tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            h.update(repr(value).encode())
    visit(batch)
    return h.hexdigest()


def trace_selection(observations, sites, *, per_episode=1, samples=256, max_records=20000):
    """Select the first frozen observations, never guess task phase or solver step."""
    if not sites:
        return set()
    if len(set(sites)) != len(sites) or min(per_episode, samples, max_records) < 1:
        raise ValueError("trace sites must be distinct and trace budgets positive")
    counts, selected = {}, set()
    for row in observations:
        episode = row["episode_id"]
        if counts.get(episode, 0) < per_episode:
            selected.add((episode, row["frame_index"], row["sampler_seed"]))
            counts[episode] = counts.get(episode, 0) + 1
    return selected


def traced_inference(model, run, sites, *, sample_id, mode, seed, samples, max_records, stream):
    """Hooks exist only during this diagnostic call; persist partial records on failure."""
    from characterize_layers import LayerTrace

    if max_records < 1:
        raise RuntimeError("layer trace record budget exhausted; partial trace retained")
    trace = LayerTrace(model, sites, max_samples=samples, max_records=max_records)
    try:
        with trace:
            trace.begin_observation(sample_id, mode=mode, seed=seed, stage="policy")
            result = run()
    finally:
        for row in trace.records:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        stream.flush()
    return result, trace.records


def verify_calibration_data(record, dataset, preprocess_sample, evaluation_episodes, expected_frames):
    """Rebuild data only, never fit or execute a model, before claiming disjointness.

    Dataset must be constructed with the original sorted calibration episode pool
    and checkpoint delta timestamps. Digest equality binds the original flat
    indices to their actual preprocessed observations, including images/state/IDs.
    """
    from quant_export import digest_batches

    expected_digest = record.get("data_sha256")
    indices = record.get("frame_indices")
    if not expected_digest or not indices:
        raise ValueError("exact calibration data digest and frame indices required")
    batches, actual_frames = [], []
    for index in indices:
        sample = dataset[index]
        actual_frames.append({"episode_id": int(sample["episode_index"]),
                              "frame_index": int(sample["frame_index"])})
        batches.append(preprocess_sample(sample))
    actual_digest = digest_batches(batches)
    if actual_digest is None or actual_digest != expected_digest:
        raise ValueError("exact preprocessed calibration data digest differs; disjointness unverified")
    if actual_frames != expected_frames:
        raise ValueError("verified calibration source IDs differ from provisional frame mapping")
    episodes = sorted({frame["episode_id"] for frame in actual_frames})
    if set(episodes) & set(evaluation_episodes):
        raise ValueError("verified calibration episodes overlap evaluation")
    return {"data_sha256_verified": True, "expected_data_sha256": expected_digest,
            "actual_data_sha256": actual_digest, "actual_episode_ids": episodes,
            "actual_frames": actual_frames, "samples": len(indices),
            "model_forward_executed": False, "fitting_executed": False}


def record(args):
    import eval_policy as ep
    import load_quantized
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.factory import make_pre_post_processors

    manifest = json.loads(Path(args.manifest).read_text())
    root = Path(args.dataset_root or manifest["dataset_root_hint"])
    model_root = Path(args.model_root or manifest["model_root_hint"])
    validate_manifest(manifest, root, model_root)
    if manifest["calibration"]["status"] != "existing_export" or not manifest["export"]:
        raise ValueError("recording requires preflight bound to an existing export; no recalibration")
    if digest(Path(args.export_dir) / "manifest.json") != manifest["export"]["manifest_sha256"]:
        raise ValueError("export fingerprint differs from frozen input manifest")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("GPU recording unavailable; CPU full-model inference is deliberately unsupported")
    sites = args.trace_sites or []
    selected_traces = trace_selection(manifest["observations"], sites,
        per_episode=args.trace_frames_per_episode, samples=args.trace_samples, max_records=args.trace_max_records)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=False)
    base, cfg = ep.load_policy(str(model_root), args.device, torch.bfloat16)
    candidate, _, exported = load_quantized.load_quantized(None, args.export_dir, args.device,
        build=lambda: ep.load_policy(str(model_root), args.device, torch.bfloat16)[0], mode="ref_conv")
    checkpoint = __import__("quant_export").checkpoint_record(str(model_root))
    validate_checkpoint_weights(checkpoint, exported)
    if sites:
        from characterize_layers import LayerTrace
        # Validate exact site presence on both models before executing either model.
        for model in (base, candidate):
            LayerTrace(model, sites, max_samples=args.trace_samples, max_records=args.trace_max_records)
    metadata = LeRobotDatasetMetadata(manifest["dataset_id"], root=root)
    have, want = sorted(metadata.camera_keys), sorted(cfg.image_features)
    rename = {} if set(want) & set(have) else dict(zip(have, want))
    preprocess, postprocess = make_pre_post_processors(cfg, pretrained_path=str(model_root),
        preprocessor_overrides={"device_processor": {"device": args.device, "float_dtype": "bfloat16"},
                                "rename_observations_processor": {"rename_map": rename}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    calibration_record = manifest["calibration"]["record"]
    calibration_dataset = LeRobotDataset(manifest["dataset_id"], root=root,
        episodes=sorted(calibration_record["episodes"]),
        delta_timestamps=resolve_delta_timestamps(cfg, metadata))
    try:
        calibration_verification = verify_calibration_data(calibration_record, calibration_dataset,
            lambda sample: ep.to_batch(sample, preprocess), manifest["sampling"]["episodes"],
            manifest["calibration"]["frames"])
    except Exception as error:
        write_new(output / "calibration_verification.json", {"data_sha256_verified": False,
            "expected_data_sha256": calibration_record.get("data_sha256"),
            "failure": type(error).__name__ + ": " + str(error),
            "evaluation_forward_executed": False, "fitting_executed": False})
        raise
    write_new(output / "calibration_verification.json", calibration_verification)
    del calibration_dataset
    dataset = LeRobotDataset(manifest["dataset_id"], root=root, episodes=manifest["sampling"]["episodes"],
                            delta_timestamps=resolve_delta_timestamps(cfg, metadata))
    header = {"schema": SCHEMA, "input_manifest_sha256": digest(args.manifest),
              "input_manifest": manifest, "export_manifest_sha256": digest(Path(args.export_dir) / "manifest.json"),
              "checkpoint": checkpoint, "camera_rename_map": rename,
              "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                            cwd=Path(__file__).resolve().parent, text=True).strip(),
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(args.device),
              "calibration_verification": calibration_verification,
              "runtime_identity": runtime_identity(args.device),
              "environment": {name: importlib.metadata.version(name) for name in
                              ("torch", "torchao", "transformers", "lerobot", "numpy", "pandas", "safetensors")},
              "source_sha256": {str(path.relative_to(Path(__file__).resolve().parent)): digest(path)
                                for path in sorted(Path(__file__).resolve().parent.rglob("*.py"))},
              "determinism_state": ep.determinism_state(), "timing_eligible": False,
              "candidate_mode": "ref_conv", "candidate_recipe": exported.get("recipe"),
              "layer_trace": {"enabled": bool(sites), "exact_sites": sites,
                  "samples_per_boundary": args.trace_samples, "record_budget_per_mode": args.trace_max_records,
                  "frames_per_episode": args.trace_frames_per_episode, "selected_observations": len(selected_traces),
                  "diagnostic_only": True, "timing_eligible": False,
                  "error_scope": "paired deterministic samples; population moments are not full-tensor error",
                  "call_index_is_sampling_step": False},
              "new_decisions": ["NEW: raw paired native-action recorder; diagnostic-only"]}
    write_new(output / "header.json", header)
    trace_counts = {"bf16": 0, "candidate": 0}
    trace_paths = {}
    with contextlib.ExitStack() as stack, torch.no_grad():
        stream = stack.enter_context((output / "actions.jsonl").open("x"))
        trace_streams = {}
        if sites:
            for name in ("bf16", "candidate", "paired_samples"):
                trace_paths[name] = output / ("layer_trace_" + name + ".jsonl")
                trace_streams[name] = stack.enter_context(trace_paths[name].open("x"))
        for frame in manifest["observations"]:
            sample = dataset[frame["dataset_local_index"]]
            actual = tuple(int(sample[key]) for key in ("episode_index", "frame_index", "index", "task_index"))
            expected = tuple(frame[key] for key in ("episode_id", "frame_index", "source_index", "task_id"))
            if actual != expected:
                raise ValueError("runtime sample IDs differ from frozen plan")
            batch = ep.to_batch(sample, preprocess)
            entry = {**frame, "input_sha256": tensor_digest(batch),
                     "observation_state": sample["observation.state"].float().tolist()}
            frame_traces = {}
            for name, policy in (("bf16", base), ("candidate", candidate)):
                policy.reset()
                def run():
                    return ep.predict_chunk(policy, dict(batch), frame["sampler_seed"], args.device)
                if (frame["episode_id"], frame["frame_index"], frame["sampler_seed"]) in selected_traces:
                    sample_id = (f"episode:{frame['episode_id']}/frame:{frame['frame_index']}"
                                 f"/source:{frame['source_index']}")
                    chunk, frame_traces[name] = traced_inference(policy, run, sites, sample_id=sample_id,
                        mode=name, seed=frame["sampler_seed"], samples=args.trace_samples,
                        max_records=args.trace_max_records - trace_counts[name], stream=trace_streams[name])
                    trace_counts[name] += len(frame_traces[name])
                else:
                    chunk = run()
                if tensor_digest(batch) != entry["input_sha256"]:
                    raise ValueError("inference mutated the shared paired input")
                native = chunk.detach().clone()
                decoded = postprocess(chunk.detach().clone()).float().cpu()[0]
                h, d = decoded.shape
                contract = manifest["action_contract"]
                if (h, d) != (contract["returned_horizon"], contract["dimension"]):
                    raise ValueError("runtime native action shape differs from frozen contract")
                if not torch.isfinite(native).all() or not torch.isfinite(decoded).all():
                    raise ValueError("non-finite action output")
                entry[name + "_model"] = native.float().cpu()[0].tolist()
                entry[name + "_dataset"] = decoded.tolist()
            if frame_traces:
                from characterize_layers import compare_samples
                for row in compare_samples(frame_traces["bf16"], frame_traces["candidate"]):
                    trace_streams["paired_samples"].write(json.dumps(row, allow_nan=False) + "\n")
                trace_streams["paired_samples"].flush()
            offset = list(cfg.action_delta_indices).index(0)
            target = sample["action"].float()[offset:offset + h]
            all_pad = sample.get("action_is_pad", torch.zeros(sample["action"].shape[0], dtype=torch.bool))
            pad = all_pad[offset:offset + h]
            full_target = torch.zeros((h, d))
            valid = torch.zeros(h, dtype=torch.bool)
            full_target[:len(target)] = target
            valid[:len(target)] = ~pad
            entry.update(target=full_target.tolist(), valid_target_mask=valid.tolist(), returned_horizon=h,
                         target_available_horizon=len(target))
            stream.write(json.dumps(entry, allow_nan=False) + "\n")
            stream.flush()
    write_new(output / "completion.json", {"complete": True, "chunks": len(manifest["observations"]),
              "actions_sha256": digest(output / "actions.jsonl"), "header_sha256": digest(output / "header.json"),
              "calibration_data_sha256_verified": True,
              "calibration_verification_sha256": digest(output / "calibration_verification.json"),
              "layer_trace": {"files_sha256": {path.name: digest(path) for path in trace_paths.values()},
                              "records_per_mode": trace_counts, "diagnostic_only": True, "timing_eligible": False},
              "reference_provenance": __import__("ref_loader").provenance()})


def statistics(values):
    import numpy as np

    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not x.size:
        return {"count": 0, "bias": None, "rmse": None, "mae": None, "std": None, "variance": None,
                "max_abs": None, "absolute_quantiles": None}
    if not np.isfinite(x).all():
        raise ValueError("non-finite measurement")
    try:
        with np.errstate(over="raise", invalid="raise"):
            result = {"count": int(x.size), "bias": float(x.mean()), "rmse": float(np.sqrt(np.mean(x * x))),
                      "mae": float(np.abs(x).mean()), "std": float(x.std()), "variance": float(x.var()),
                      "max_abs": float(np.abs(x).max()),
                      "absolute_quantiles": {str(q): float(np.quantile(np.abs(x), q)) for q in (.5, .9, .95, .99)}}
    except FloatingPointError as error:
        raise ValueError("non-finite derived measurement (numeric overflow)") from error
    if not all(np.isfinite(value) for key, value in result.items() if key not in ("absolute_quantiles", "count")):
        raise ValueError("non-finite derived measurement")
    return result


def summarize(error, mask):
    import numpy as np

    error, mask = np.asarray(error, dtype=np.float64), np.asarray(mask, dtype=bool)
    n, h, d = error.shape
    if mask.shape != (n, h):
        raise ValueError("mask shape differs from action axes")
    return {"pooled": statistics(error[mask]),
            "per_component": [statistics(error[:, :, k][mask]) for k in range(d)],
            "per_horizon_component": [[statistics(error[:, t, k][mask[:, t]]) for k in range(d)] for t in range(h)]}


def validated_arrays(records, contract, *, physical=False):
    """Validate native axes and finite values once for analysis and chart sources."""
    import numpy as np

    if physical and (not contract.get("physical_controller_mapping_validated")
                     or "unknown" in contract.get("units", ["unknown"])):
        raise ValueError("physical analysis requires validated units and controller mapping")
    if not records:
        raise ValueError("no action records")
    h, d = contract["returned_horizon"], contract["dimension"]
    if type(h) is not int or type(d) is not int or h <= 0 or d <= 0:
        raise ValueError("native horizon and dimension must be positive integers")
    channels = contract["channels"]
    if (not isinstance(channels, list) or len(channels) != d
            or any(not isinstance(x, str) or not x.strip() for x in channels)
            or len(set(channels)) != d):
        raise ValueError("channel names differ from native dimension")
    units = contract.get("units", ["unknown"] * d)
    if not isinstance(units, list) or len(units) != d or any(not isinstance(x, str) or not x.strip() for x in units):
        raise ValueError("native component units must match dimension")
    ids = [(row["episode_id"], row["frame_index"], row["sampler_seed"]) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate observation/seed records")
    arrays = {}
    for key in ("bf16_model", "candidate_model", "bf16_dataset", "candidate_dataset", "target"):
        arrays[key] = np.asarray([row[key] for row in records], dtype=np.float64)
        if arrays[key].shape != (len(records), h, d) or not np.isfinite(arrays[key]).all():
            raise ValueError("invalid shape or non-finite values: " + key)
    mask = np.asarray([row["valid_target_mask"] for row in records])
    if mask.dtype != np.bool_ or mask.shape != (len(records), h):
        raise ValueError("valid-target mask must be boolean [chunks,horizon]")
    return arrays, mask


def measurement_errors(arrays, mask):
    """Return each numerical comparison with its exact full/valid denominator."""
    import numpy as np

    full = np.ones_like(mask)
    measurements = {}
    for space in ("model", "dataset"):
        diff = arrays["candidate_" + space] - arrays["bf16_" + space]
        measurements[space + "_full_generated_fidelity"] = (diff, full)
        measurements[space + "_valid_target_fidelity"] = (diff, mask)
    for name in ("bf16", "candidate"):
        measurements[name + "_valid_target_error"] = (arrays[name + "_dataset"] - arrays["target"], mask)
    return measurements


def descriptive_distribution(values):
    """Distribution of nonnegative chunk RMSE, not error bias or RMS-of-RMSE."""
    import numpy as np

    summary = statistics(values)
    return {"count": summary["count"], "mean": summary["bias"], "std": summary["std"],
            "variance": summary["variance"], "min": float(np.min(values)) if len(values) else None,
            "max": summary["max_abs"], "quantiles": summary["absolute_quantiles"],
            "variance_denominator": "population (ddof=0); descriptive, not a confidence interval"}


def chunk_components(error, mask):
    """Per-chunk component RMSE and its descriptive between-chunk distribution."""
    values = [[statistics(chunk[:, k][valid])["rmse"] for k in range(error.shape[2])]
              for chunk, valid in zip(error, mask)]
    return {"rmse": values, "valid_steps": [int(x.sum()) for x in mask],
            "per_component_rmse_distribution": [
                descriptive_distribution([row[k] for row in values if row[k] is not None])
                for k in range(error.shape[2])],
            "distribution_scope": "one value per chunk, descriptive; overlapping chunks are not independent"}


def analyze_records(records, contract, *, physical=False):
    import numpy as np

    arrays, mask = validated_arrays(records, contract, physical=physical)
    full = np.ones_like(mask)
    out = {"schema": SCHEMA, "contract": contract, "chunks": len(records),
           "episode_clusters": len({row["episode_id"] for row in records}),
           "uncertainty": "descriptive only; no scalar-step independence or confidence interval claimed",
           "pooled_scope": "native-coordinate aggregate; components may have incomparable units; not physical RMSE",
           "task_grouping": {"missing_task_id_chunks": sum(row.get("task_id") is None for row in records),
                             "missing_id_label": "unknown (null), never inferred from action values",
                             "scope": "recorded task IDs; names/duplicate human labels do not establish task success"},
           "valid_target_steps": int(mask.sum()), "generated_steps": int(full.sum())}
    measurements = measurement_errors(arrays, mask)
    chunk_ids = [{key: row.get(key) for key in ("task_id", "episode_id", "frame_index", "sampler_seed")}
                 for row in records]
    out["per_chunk_components"] = {"observation_ids": chunk_ids, "measurements": {}}
    for name, (error, valid) in measurements.items():
        out[name] = summarize(error, valid)
        out["per_chunk_components"]["measurements"][name] = chunk_components(error, valid)
    out["groups"] = {"tasks": [], "episodes": []}
    for axis, keys in (("tasks", ("task_id",)), ("episodes", ("task_id", "episode_id"))):
        groups = {}
        for index, row in enumerate(records):
            identity = tuple(row.get(key) for key in keys)
            groups.setdefault(identity, []).append(index)
        for identity, indices in groups.items():
            group = {**dict(zip(keys, identity)), "chunks": len(indices),
                     "episode_clusters": len({records[i]["episode_id"] for i in indices}),
                     "generated_steps": int(full[indices].sum()), "valid_target_steps": int(mask[indices].sum()),
                     "measurements": {}}
            for name, (error, valid) in measurements.items():
                group["measurements"][name] = {**summarize(error[indices], valid[indices]),
                    "per_chunk_components": chunk_components(error[indices], valid[indices])}
            out["groups"][axis].append(group)
    return out


def load_verified_recording(record_dir):
    """Read the same hash/identity-verified recording for analysis and reporting.

    This proves internal evidence binding, not authenticity of supplied completion files.
    No source dataset, model or GPU imports are required for offline use.
    """
    root = Path(record_dir)
    completion_bytes = (root / "completion.json").read_bytes()
    complete = json.loads(completion_bytes)
    action_bytes = (root / "actions.jsonl").read_bytes()
    if complete.get("complete") is not True or hashlib.sha256(action_bytes).hexdigest() != complete["actions_sha256"]:
        raise ValueError("recording incomplete or action payload digest differs")
    header_bytes = (root / "header.json").read_bytes()
    if hashlib.sha256(header_bytes).hexdigest() != complete["header_sha256"]:
        raise ValueError("record header fingerprint differs")
    for filename, expected_hash in complete.get("layer_trace", {}).get("files_sha256", {}).items():
        if Path(filename).name != filename or filename in (".", ".."):
            raise ValueError("layer trace filename must be a recording-local basename")
        if digest(root / filename) != expected_hash:
            raise ValueError("layer trace payload fingerprint differs: " + filename)
    header = json.loads(header_bytes)
    manifest = header["input_manifest"]
    if ("content_sha256" in manifest and canonical_digest(
            {key: value for key, value in manifest.items() if key != "content_sha256"}) != manifest["content_sha256"]):
        raise ValueError("embedded input manifest content digest differs")
    if (complete.get("calibration_data_sha256_verified") is not True
            or header.get("calibration_verification", {}).get("data_sha256_verified") is not True):
        raise ValueError("exact calibration data digest verification is required")
    verification_bytes = (root / "calibration_verification.json").read_bytes()
    if hashlib.sha256(verification_bytes).hexdigest() != complete.get("calibration_verification_sha256"):
        raise ValueError("calibration verification fingerprint differs")
    verification = json.loads(verification_bytes)
    expected_calibration_digest = header["input_manifest"]["calibration"]["record"]["data_sha256"]
    if (verification != header["calibration_verification"]
            or verification.get("actual_data_sha256") != expected_calibration_digest
            or verification.get("expected_data_sha256") != expected_calibration_digest):
        raise ValueError("calibration digest proof differs from original export provenance")
    rows = [json.loads(line) for line in action_bytes.splitlines()]
    if type(complete["chunks"]) is not int or len(rows) != complete["chunks"]:
        raise ValueError("recorded chunk count differs from completion marker")
    expected = header["input_manifest"]["observations"]
    keys = ("episode_id", "frame_index", "source_index", "sampler_seed", "task_id")
    if [tuple(row[key] for key in keys) for row in rows] != [tuple(row[key] for key in keys) for row in expected]:
        raise ValueError("recorded observation IDs differ from frozen manifest")
    validated_arrays(rows, header["input_manifest"]["action_contract"])
    complete["verified_completion_sha256"] = hashlib.sha256(completion_bytes).hexdigest()
    return rows, header, complete, verification


def analyze(args):
    rows, header, complete, verification = load_verified_recording(args.record_dir)
    result = analyze_records(rows, header["input_manifest"]["action_contract"], physical=args.physical)
    result["provenance"] = {"header_sha256": complete["header_sha256"],
                            "actions_sha256": complete["actions_sha256"],
                            "input_manifest_sha256": header["input_manifest_sha256"]}
    result["layer_trace"] = complete.get("layer_trace")
    result["calibration_verification"] = verification
    write_new(args.out, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight")
    for name in ("model-config", "dataset-root", "policy-id", "dataset-id", "out"):
        pre.add_argument("--" + name, required=True)
    pre.add_argument("--task-id", type=int, required=True)
    pre.add_argument("--episodes", type=int, default=10)
    pre.add_argument("--frames-per-episode", type=int, default=10)
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--calibration-frames", type=int, default=32)
    pre.add_argument("--calibration-seed", type=int, default=1234)
    pre.add_argument("--export-dir")
    rec = commands.add_parser("record")
    for name in ("manifest", "export-dir", "out-dir"):
        rec.add_argument("--" + name, required=True)
    rec.add_argument("--dataset-root")
    rec.add_argument("--model-root")
    rec.add_argument("--device", default="cuda")
    rec.add_argument("--trace-sites", nargs="+", help="optional distinct exact module FQNs; no substring matching")
    rec.add_argument("--trace-samples", type=int, default=256)
    rec.add_argument("--trace-frames-per-episode", type=int, default=1)
    rec.add_argument("--trace-max-records", type=int, default=20000,
                     help="hard record cap per mode across all traced observations; partial traces retained on failure")
    ana = commands.add_parser("analyze")
    ana.add_argument("--record-dir", required=True)
    ana.add_argument("--out", required=True)
    ana.add_argument("--physical", action="store_true")
    args = parser.parse_args()
    try:
        result = {"preflight": preflight, "record": record, "analyze": analyze}[args.command](args)
    except (ValueError, FileExistsError, FileNotFoundError) as error:
        parser.error(str(error))
    if result is not None:
        print(json.dumps({key: result[key] for key in ("schema", "content_sha256", "chunks") if key in result}))


if __name__ == "__main__":
    main()
