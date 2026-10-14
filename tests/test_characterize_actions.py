"""NEW characterization contracts: native axes, masks, IDs and frozen metadata."""
import argparse
import copy
import json

import characterize_actions as ca
import numpy as np
import pytest


def contract(h=3, d=2):
    return {"returned_horizon": h, "dimension": d, "channels": [f"joint_{k}" for k in range(d)],
            "units": ["unknown"] * d, "physical_controller_mapping_validated": False}


def actions(h=3, d=2):
    base = np.zeros((h, d))
    candidate = np.arange(h * d).reshape(h, d).astype(float)
    return [{"episode_id": 7, "frame_index": 11, "sampler_seed": 42,
             "bf16_model": base.tolist(), "candidate_model": candidate.tolist(),
             "bf16_dataset": base.tolist(), "candidate_dataset": candidate.tolist(),
             "target": base.tolist(), "valid_target_mask": [True] + [False] * (h - 1)}]


def test_native_shapes_masks_and_independent_target_error():
    out = ca.analyze_records(actions(), contract())
    assert out["generated_steps"] == 3 and out["valid_target_steps"] == 1
    assert out["dataset_full_generated_fidelity"]["pooled"]["count"] == 6
    assert out["dataset_valid_target_fidelity"]["pooled"]["count"] == 2
    assert out["dataset_valid_target_fidelity"]["pooled"]["rmse"] == pytest.approx(2 ** -.5)
    assert out["bf16_valid_target_error"]["pooled"]["rmse"] == 0
    assert out["candidate_valid_target_error"]["per_horizon_component"][2][1]["count"] == 0
    assert out["episode_clusters"] == 1 and "descriptive" in out["uncertainty"]


@pytest.mark.parametrize("h,d", [(50, 6), (16, 7), (50, 8)])
def test_no_universal_seven_by_fifty(h, d):
    out = ca.analyze_records(actions(h, d), contract(h, d))
    assert len(out["model_full_generated_fidelity"]["per_horizon_component"]) == h
    assert len(out["model_full_generated_fidelity"]["per_component"]) == d


def test_statistics_bias_std_quantiles_empty_population():
    out = ca.statistics([-1, 1, 3])
    assert out["bias"] == 1 and out["mae"] == pytest.approx(5 / 3)
    assert out["std"] == pytest.approx(np.std([-1, 1, 3]))
    assert out["absolute_quantiles"]["0.5"] == 1
    assert ca.statistics([])["rmse"] is None


def test_unknown_physical_units_duplicate_ids_nonfinite_and_mask_refused():
    rows = actions()
    with pytest.raises(ValueError, match="validated units"):
        ca.analyze_records(rows, contract(), physical=True)
    with pytest.raises(ValueError, match="duplicate"):
        ca.analyze_records(rows + rows, contract())
    bad = copy.deepcopy(rows)
    bad[0]["candidate_dataset"][0][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        ca.analyze_records(bad, contract())
    bad = copy.deepcopy(rows)
    bad[0]["valid_target_mask"] = [1, 0, 0]
    with pytest.raises(ValueError, match="boolean"):
        ca.analyze_records(bad, contract())


def test_balanced_plan_deterministic_disjoint_and_insufficient_refused():
    episodes = [dict(episode_index=k, length=20, dataset_from_index=k * 20) for k in range(6)]
    kwargs = dict(count=3, per_episode=4, seed=0, excluded=[0, 1])
    plan = ca.balanced_plan(episodes, **kwargs)
    assert plan == ca.balanced_plan(episodes, **kwargs)
    assert len(plan) == 12
    assert not {0, 1} & {row["episode_id"] for row in plan}
    for ep in {row["episode_id"] for row in plan}:
        frames = [row["frame_index"] for row in plan if row["episode_id"] == ep]
        assert [frame // 5 for frame in frames] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="insufficient"):
        ca.balanced_plan(episodes, count=6, per_episode=4, seed=0, excluded=[0])


def test_actual_calibration_indices_resolve_source_episode():
    episodes = [dict(episode_index=2, length=3), dict(episode_index=9, length=5)]
    rec = dict(episodes=[9, 2], frame_indices=[0, 2, 3, 7], data_sha256="a")
    assert ca.calibration_frames(rec, episodes) == [
        dict(episode_id=2, frame_index=0), dict(episode_id=2, frame_index=2),
        dict(episode_id=9, frame_index=0), dict(episode_id=9, frame_index=4)]
    with pytest.raises(ValueError, match="exact calibration"):
        ca.calibration_frames({}, episodes)


@pytest.fixture
def fixture_preflight(tmp_path):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    root = tmp_path / "data"
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    model = tmp_path / "model"
    model.mkdir()
    config = dict(type="smolvla", output_features={"action": {"shape": [2]}},
                  chunk_size=3, num_steps=10, normalization_mapping={"ACTION": "MEAN_STD"})
    (model / "config.json").write_text(json.dumps(config))
    info = dict(features={"action": {"shape": [2], "names": ["joint", "gripper"]}}, fps=30,
                data_path="data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text("{}")
    pd.DataFrame({"task_index": [0]}, index=["pick"]).to_parquet(root / "meta/tasks.parquet")
    pd.DataFrame([dict(episode_index=k, length=20, dataset_from_index=k * 20,
                      tasks=["pick"], **{"data/chunk_index": 0, "data/file_index": 0})
                  for k in range(6)]).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame([dict(episode_index=k, frame_index=j, index=k * 20 + j, task_index=0)
                  for k in range(6) for j in range(20)]).to_parquet(root / "data/chunk-000/file-000.parquet")
    return argparse.Namespace(dataset_root=str(root), model_config=str(model / "config.json"),
        task_id=0, policy_id="test/policy", dataset_id="test/data", episodes=2,
        frames_per_episode=3, seed=0, calibration_frames=4, calibration_seed=1234,
        export_dir=None, out=str(tmp_path / "manifest.json"))


def export_model(args, identity=None):
    return {"id": identity or args.policy_id, "kind": "lerobot_policy", "checkpoint": {
        "config_json_sha256": ca.digest(args.model_config),
        "files": {"model.safetensors": {"sha256": "a" * 64}}}}


def test_metadata_preflight_does_not_import_models_and_detects_tampering(fixture_preflight):
    args = fixture_preflight
    manifest = ca.preflight(args)
    assert manifest["sampling"]["chunks"] == 6
    assert manifest["calibration"]["status"] == "planned_unfitted"
    assert not set(manifest["sampling"]["episodes"]) & set(manifest["calibration_episode_ids"])
    ca.validate_manifest(manifest, args.dataset_root, __import__("pathlib").Path(args.model_config).parent)
    bad = copy.deepcopy(manifest)
    bad["observations"][0]["frame_index"] += 1
    with pytest.raises(ValueError, match="content digest"):
        ca.validate_manifest(bad, args.dataset_root, __import__("pathlib").Path(args.model_config).parent)
    with pytest.raises(FileExistsError):
        ca.preflight(args)


def test_bound_export_selects_episodes_outside_actual_calibration(fixture_preflight, tmp_path):
    args = fixture_preflight
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps({"model": export_model(args),
        "provenance": {"lerobot_dataset": args.dataset_id}, "calibration": {
        "episodes": [0, 1, 2], "frame_indices": [0, 21, 42], "data_sha256": "a"}}))
    args.export_dir = str(export)
    manifest = ca.preflight(args)
    assert set(manifest["sampling"]["episodes"]) <= {3, 4, 5}
    assert manifest["calibration"]["status"] == "existing_export"


def test_complete_record_integrity_required(tmp_path):
    ca.write_new(tmp_path / "actions.jsonl", {})
    ca.write_new(tmp_path / "completion.json", {"complete": True, "actions_sha256": "wrong"})
    with pytest.raises(ValueError, match="digest"):
        ca.analyze(argparse.Namespace(record_dir=str(tmp_path), physical=False, out=str(tmp_path / "summary.json")))


def test_tensor_input_digest_includes_scalar_and_float_dtype():
    import torch
    batch = {"index": torch.tensor(1), "state": torch.tensor([1.0], dtype=torch.bfloat16)}
    assert ca.tensor_digest(batch) == ca.tensor_digest(copy.deepcopy(batch))
    changed = {**batch, "index": torch.tensor(2)}
    assert ca.tensor_digest(batch) != ca.tensor_digest(changed)


def test_dataset_tampering_refused(fixture_preflight):
    from pathlib import Path
    args = fixture_preflight
    manifest = ca.preflight(args)
    (Path(args.dataset_root) / "meta/stats.json").write_text('{"modified": true}')
    with pytest.raises(ValueError, match="dataset asset fingerprint"):
        ca.validate_manifest(manifest, args.dataset_root, Path(args.model_config).parent)


def test_missing_task_and_unsupported_policy_refused(fixture_preflight):
    from pathlib import Path
    args = fixture_preflight
    args.task_id = 999
    with pytest.raises(ValueError, match="task ID"):
        ca.preflight(args)
    args.task_id = 0
    config = json.loads(Path(args.model_config).read_text())
    config["type"] = "multi_task_dit"
    Path(args.model_config).write_text(json.dumps(config))
    with pytest.raises(ValueError, match="supports SmolVLA only"):
        ca.preflight(args)


def test_export_without_actual_calibration_refused(fixture_preflight, tmp_path):
    args = fixture_preflight
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps({"model": export_model(args),
        "provenance": {"lerobot_dataset": args.dataset_id}, "calibration": {}}))
    args.export_dir = str(export)
    with pytest.raises(ValueError, match="exact calibration"):
        ca.preflight(args)


def test_persisted_complete_analysis_retains_provenance(tmp_path):
    rows = actions()
    manifest = {"observations": [{key: row.get(key) for key in
                  ("episode_id", "frame_index", "source_index", "sampler_seed", "task_id")}
                 for row in rows], "action_contract": contract(),
                 "calibration": {"record": {"data_sha256": "a" * 64}}}
    rows[0].update(source_index=None, task_id=None)
    verification = {"data_sha256_verified": True, "actual_data_sha256": "a" * 64,
                    "expected_data_sha256": "a" * 64}
    ca.write_new(tmp_path / "calibration_verification.json", verification)
    header = {"input_manifest": manifest, "input_manifest_sha256": "frozen",
              "calibration_verification": verification}
    ca.write_new(tmp_path / "header.json", header)
    (tmp_path / "actions.jsonl").write_text(json.dumps(rows[0]) + "\n")
    ca.write_new(tmp_path / "completion.json", {"complete": True, "chunks": 1,
        "actions_sha256": ca.digest(tmp_path / "actions.jsonl"),
        "header_sha256": ca.digest(tmp_path / "header.json"), "calibration_data_sha256_verified": True,
        "calibration_verification_sha256": ca.digest(tmp_path / "calibration_verification.json")})
    out = ca.analyze(argparse.Namespace(record_dir=str(tmp_path), physical=False, out=str(tmp_path / "summary.json")))
    assert out["provenance"]["input_manifest_sha256"] == "frozen"
    assert json.loads((tmp_path / "summary.json").read_text())["chunks"] == 1


def test_absolute_model_alias_requires_exact_config_and_weight_provenance(fixture_preflight):
    args = fixture_preflight
    exported = {"model": export_model(args, "/gpu/cache/pinned/smolvla")}
    identity = ca.validate_export_config(exported, args.model_config, args.policy_id)
    assert identity["exported_model_id"] == "/gpu/cache/pinned/smolvla"
    exported["model"]["checkpoint"]["config_json_sha256"] = "wrong"
    with pytest.raises(ValueError, match="configuration fingerprint"):
        ca.validate_export_config(exported, args.model_config, args.policy_id)
    exported = {"model": export_model(args)}
    exported["model"]["checkpoint"]["files"]["model.safetensors"]["sha256"] = None
    with pytest.raises(ValueError, match="weight fingerprints"):
        ca.validate_export_config(exported, args.model_config, args.policy_id)


def test_unavailable_runtime_command_retains_reason(monkeypatch):
    def absent(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi unavailable")
    monkeypatch.setattr(ca.subprocess, "check_output", absent)
    out = ca.command_record(["nvidia-smi"])
    assert out["value"] is None and "FileNotFoundError" in out["unknown_reason"]


def test_preflight_states_video_hash_boundary(fixture_preflight):
    manifest = ca.preflight(fixture_preflight)
    assert "not fingerprinted" in manifest["input_fingerprint_scope"]["video_bytes"]
    assert "per-frame input_sha256" in manifest["input_fingerprint_scope"]["decoded_preprocessed_observations"]


def test_actual_original_weights_must_match_even_for_valid_alias(fixture_preflight):
    exported = {"model": export_model(fixture_preflight, "/gpu/policy/alias")}
    original = copy.deepcopy(exported["model"]["checkpoint"])
    ca.validate_checkpoint_weights(original, exported)
    original["files"]["model.safetensors"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="weights differ"):
        ca.validate_checkpoint_weights(original, exported)
    original["files"]["model.safetensors"]["sha256"] = None
    with pytest.raises(ValueError, match="fingerprints required"):
        ca.validate_checkpoint_weights(original, exported)


def test_optional_trace_selection_first_frozen_frame_per_episode():
    observations = [{"episode_id": ep, "frame_index": frame, "sampler_seed": frame + 100}
                    for ep in (2, 9) for frame in (7, 21, 42)]
    assert ca.trace_selection(observations, []) == set()
    assert ca.trace_selection(observations, ["exact.site"]) == {(2, 7, 107), (9, 7, 107)}
    assert len(ca.trace_selection(observations, ["exact.site"], per_episode=2)) == 4
    with pytest.raises(ValueError, match="distinct"):
        ca.trace_selection(observations, ["site", "site"])
    with pytest.raises(ValueError, match="positive"):
        ca.trace_selection(observations, ["site"], samples=0)


def test_optional_trace_cpu_pairing_keeps_samples_bounded_and_hooks_scoped():
    import io

    import torch
    from characterize_layers import compare_samples
    model = torch.nn.Sequential(torch.nn.Linear(4, 3)).eval()
    x = torch.arange(8).float().reshape(2, 4)
    expected = model(x).detach().clone()
    traces = {}
    for mode in ("bf16", "candidate"):
        stream = io.StringIO()
        result, traces[mode] = ca.traced_inference(model, lambda: model(x), ["0"],
            sample_id="episode:2/frame:7/source:47", mode=mode, seed=107,
            samples=2, max_records=20, stream=stream)
        assert torch.equal(result, expected)
        assert len(stream.getvalue().splitlines()) == 2
        assert all(row["sample"]["elements"] <= 2 for row in traces[mode])
        assert all(row["sampling_step"] is None for row in traces[mode])
        assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks
    paired = compare_samples(traces["bf16"], traces["candidate"])
    assert all(row["sample_metrics"]["rmse"] == 0 for row in paired)
    assert all(not row["full_tensor_error_measured"] for row in paired)
    # An unselected inference installs no observer and emits no additional records.
    model(x)
    assert len(traces["bf16"]) == 2


def test_optional_trace_budget_preserves_partial_stream_and_removes_hooks():
    import io

    import torch
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="budget"):
        ca.traced_inference(model, lambda: model(torch.ones(4)), ["0"],
            sample_id="episode:2/frame:7", mode="bf16", seed=0,
            samples=2, max_records=1, stream=stream)
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(rows) == 1 and rows[0]["boundary"] == "input"
    assert not model[0]._forward_hooks and not model[0]._forward_pre_hooks


def test_optional_trace_missing_exact_site_refuses_before_forward():
    import io

    import torch
    model = torch.nn.Sequential(torch.nn.Identity()).eval()
    with pytest.raises(ValueError, match="missing"):
        ca.traced_inference(model, lambda: pytest.fail("must not run"), ["0.not_a_substring"],
            sample_id="frame", mode="bf16", seed=0, samples=2, max_records=10, stream=io.StringIO())


def calibration_sample(episode, frame, value=None):
    import torch
    return {"episode_index": torch.tensor(episode), "frame_index": torch.tensor(frame),
            "state": torch.tensor([float(episode), float(frame) if value is None else value])}


def calibration_batch(sample):
    # Match to_batch's scalar unsqueeze before using the actual export digest implementation.
    return {key: value.unsqueeze(0) for key, value in sample.items()}


def test_exact_calibration_digest_verified_without_model_forward():
    from quant_export import digest_batches
    dataset = [calibration_sample(0, 0), calibration_sample(1, 0), calibration_sample(1, 1)]
    record = {"frame_indices": [1, 2], "data_sha256": digest_batches([calibration_batch(dataset[i]) for i in (1, 2)])}
    frames = [{"episode_id": 1, "frame_index": 0}, {"episode_id": 1, "frame_index": 1}]
    verified = ca.verify_calibration_data(record, dataset, calibration_batch, [0], frames)
    assert verified["data_sha256_verified"] is True
    assert verified["actual_episode_ids"] == [1]
    assert not verified["model_forward_executed"] and not verified["fitting_executed"]
    changed = copy.deepcopy(dataset)
    changed[2] = calibration_sample(1, 1, value=99.0)
    with pytest.raises(ValueError, match="digest differs"):
        ca.verify_calibration_data(record, changed, calibration_batch, [0], frames)


def test_changed_episode_lengths_cannot_prove_calibration_disjointness():
    from quant_export import digest_batches
    old = [calibration_sample(ep, frame) for ep, n in ((0, 3), (1, 5)) for frame in range(n)]
    new = [calibration_sample(ep, frame) for ep, n in ((0, 5), (1, 3)) for frame in range(n)]
    record = {"episodes": [0, 1], "frame_indices": [4],
              "data_sha256": digest_batches([calibration_batch(old[4])])}
    provisional = ca.calibration_frames(record, [{"episode_index": 0, "length": 5},
                                                 {"episode_index": 1, "length": 3}])
    assert provisional == [{"episode_id": 0, "frame_index": 4}]
    assert int(old[4]["episode_index"]) == 1  # Actual original calibration was in prospective eval episode 1.
    with pytest.raises(ValueError, match="digest differs; disjointness unverified"):
        ca.verify_calibration_data(record, new, calibration_batch, [1], provisional)


def test_verified_calibration_still_refuses_overlapping_or_misidentified_frames():
    from quant_export import digest_batches
    dataset = [calibration_sample(2, 8)]
    record = {"frame_indices": [0], "data_sha256": digest_batches([calibration_batch(dataset[0])])}
    with pytest.raises(ValueError, match="overlap evaluation"):
        ca.verify_calibration_data(record, dataset, calibration_batch, [2], [{"episode_id": 2, "frame_index": 8}])
    with pytest.raises(ValueError, match="source IDs differ"):
        ca.verify_calibration_data(record, dataset, calibration_batch, [3], [{"episode_id": 2, "frame_index": 9}])


def test_bound_preflight_disjointness_explicitly_provisional(fixture_preflight, tmp_path):
    args = fixture_preflight
    export = tmp_path / "export"
    export.mkdir()
    (export / "manifest.json").write_text(json.dumps({"model": export_model(args),
        "provenance": {"lerobot_dataset": args.dataset_id}, "calibration": {
        "episodes": [0, 1, 2], "frame_indices": [0, 21, 42], "data_sha256": "a" * 64}}))
    args.export_dir = str(export)
    manifest = ca.preflight(args)
    assert manifest["calibration"]["data_sha256_verified"] is False
    assert manifest["calibration_disjointness_status"] == "provisional_pending_exact_export_data_digest"
