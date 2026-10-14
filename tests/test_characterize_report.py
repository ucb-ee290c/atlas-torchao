"""Synthetic offline reporting contracts; these fixtures are not model measurements."""
import argparse
import builtins
import copy
import csv
import json
import xml.etree.ElementTree as ET

import characterize_actions as ca
import characterize_report as report
import numpy as np
import pytest


def contract():
    return {"returned_horizon": 3, "dimension": 2, "channels": ["native_joint", "gripper"],
            "units": ["unknown", "unknown"], "physical_controller_mapping_validated": False}


def rows():
    errors = [np.array([[-1, 0], [1, 2], [3, 4]]), np.full((3, 2), 2), np.full((3, 2), 6)]
    result = []
    for k, error in enumerate(errors):
        result.append({"episode_id": 7 if k < 2 else 9, "task_id": 1 if k < 2 else 2,
            "frame_index": 10 + k, "source_index": 40 + k, "sampler_seed": 100 + k,
            "bf16_model": np.zeros((3, 2)).tolist(), "candidate_model": (error * 2).tolist(),
            "bf16_dataset": np.zeros((3, 2)).tolist(), "candidate_dataset": error.tolist(),
            "target": np.ones((3, 2)).tolist(), "valid_target_mask": [True, True, False]})
    return result


def completed(path, *, samples=None, native=None):
    path.mkdir()
    samples = rows() if samples is None else samples
    manifest = {"observations": [{key: row[key] for key in
                ("episode_id", "frame_index", "source_index", "sampler_seed", "task_id")} for row in samples],
                "action_contract": native or contract(), "calibration": {"record": {"data_sha256": "a" * 64}},
                "scope": "synthetic test fixture, not model evidence"}
    manifest["content_sha256"] = ca.canonical_digest(manifest)
    verification = {"data_sha256_verified": True, "actual_data_sha256": "a" * 64,
                    "expected_data_sha256": "a" * 64}
    ca.write_new(path / "calibration_verification.json", verification)
    ca.write_new(path / "header.json", {"input_manifest": manifest, "input_manifest_sha256": "f" * 64,
        "calibration_verification": verification, "source_commit": "synthetic", "candidate_mode": "synthetic"})
    with (path / "actions.jsonl").open("x") as stream:
        for row in samples:
            stream.write(json.dumps(row) + "\n")
    completion = {"complete": True, "chunks": len(samples),
        "actions_sha256": ca.digest(path / "actions.jsonl"), "header_sha256": ca.digest(path / "header.json"),
        "calibration_data_sha256_verified": True,
        "calibration_verification_sha256": ca.digest(path / "calibration_verification.json")}
    ca.write_new(path / "completion.json", completion)
    return path


def test_grouped_variance_per_chunk_rmse_and_exact_mask_denominators():
    summary = ca.analyze_records(rows(), contract())
    assert summary["generated_steps"] == 9 and summary["valid_target_steps"] == 6
    assert summary["dataset_full_generated_fidelity"]["pooled"]["count"] == 18
    assert summary["dataset_valid_target_fidelity"]["pooled"]["count"] == 12
    first = summary["groups"]["tasks"][0]
    assert first["task_id"] == 1 and first["chunks"] == 2
    error = np.array([-1, 1, 3, 2, 2, 2])
    measured = first["measurements"]["dataset_full_generated_fidelity"]["per_component"][0]
    assert measured["rmse"] == pytest.approx(np.sqrt(np.mean(error ** 2)))
    assert measured["variance"] == pytest.approx(np.var(error))
    assert measured["std"] == pytest.approx(np.std(error))
    chunks = summary["per_chunk_components"]["measurements"]["dataset_full_generated_fidelity"]
    expected = [np.sqrt(11 / 3), 2, 6]
    assert [x[0] for x in chunks["rmse"]] == pytest.approx(expected)
    distribution = chunks["per_component_rmse_distribution"][0]
    assert distribution["mean"] == pytest.approx(np.mean(expected))
    assert distribution["variance"] == pytest.approx(np.var(expected))
    assert "bias" not in distribution and "rmse" not in distribution
    assert summary["model_full_generated_fidelity"]["per_component"][0]["rmse"] == pytest.approx(
        2 * summary["dataset_full_generated_fidelity"]["per_component"][0]["rmse"])
    assert len(summary["groups"]["episodes"]) == 2
    assert "incomparable units" in summary["pooled_scope"]


def test_report_csv_chart_sources_and_provenance_bind_verified_records(tmp_path):
    source = completed(tmp_path / "record")
    output = tmp_path / "report"
    result = report.report(source, output, ecdf_points=3)
    assert result["complete"] and len(result["charts"]) == 18
    for filename, expected in result["files_sha256"].items():
        assert ca.digest(output / filename) == expected
    assert result["provenance"]["actions_sha256"] == ca.digest(source / "actions.jsonl")
    assert result["provenance"]["completion_sha256"] == ca.digest(source / "completion.json")
    assert result["scope"]["acceptance_thresholds"] is None
    assert not result["scope"]["physical_unit_conversion"] and not result["scope"]["task_success_measured"]
    with (output / "metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    component = next(x for x in metrics if x["group"] == "all" and x["point"] == "component"
                     and x["component_index"] == "0" and x["comparison"] == "dataset_valid_target_fidelity")
    assert component["count"] == "6" and component["units"] == "unknown"
    with (output / "chart_data.csv").open() as stream:
        data = list(csv.DictReader(stream))
    actual = np.sort(np.abs(np.asarray([x["candidate_dataset"] for x in rows()])[:, :, 0]).reshape(-1))
    ecdf = [x for x in data if x["comparison"] == "dataset_full_generated_fidelity"
            and x["chart"] == "absolute_error_ecdf" and x["component_index"] == "0"]
    assert len(ecdf) <= 3
    for point in ecdf:
        assert float(point["cdf"]) == pytest.approx(np.searchsorted(actual, float(point["value"]), side="right") / 9)
    for chart in result["charts"]:
        root = ET.parse(output / chart["file"]).getroot()
        assert chart["chart_source_sha256"] == ca.digest(output / "chart_data.csv")
        assert chart["chart_source_sha256"] in root.find("{http://www.w3.org/2000/svg}metadata").text


def test_empty_valid_target_population_is_missing_not_zero(tmp_path):
    samples = rows()
    for row in samples:
        row["valid_target_mask"] = [False] * 3
    output = tmp_path / "report"
    report.report(completed(tmp_path / "record", samples=samples), output)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["dataset_valid_target_fidelity"]["pooled"]["count"] == 0
    assert summary["dataset_valid_target_fidelity"]["pooled"]["rmse"] is None
    chunks = summary["per_chunk_components"]["measurements"]["dataset_valid_target_fidelity"]
    assert chunks["rmse"] == [[None, None]] * 3
    assert chunks["per_component_rmse_distribution"][0]["count"] == 0
    assert "no valid targets" in (output / "dataset_valid_target_fidelity__component_rmse.svg").read_text()
    assert "#dddddd" in (output / "dataset_valid_target_fidelity__horizon_component_rmse.svg").read_text()


def test_no_overwrite_of_record_or_existing_report(tmp_path):
    source = completed(tmp_path / "record")
    before = {p.name: ca.digest(p) for p in source.iterdir()}
    with pytest.raises(FileExistsError):
        report.report(source, source)
    output = tmp_path / "report"
    report.report(source, output)
    hashes = {p.name: ca.digest(p) for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        report.report(source, output)
    assert hashes == {p.name: ca.digest(p) for p in output.iterdir()}
    assert before == {p.name: ca.digest(p) for p in source.iterdir()}


def test_labels_are_data_not_svg_or_spreadsheet_code(tmp_path):
    native = contract()
    native["channels"] = ['=SUM(1,2)<script>&"\x01', "joint,with\nnewline"]
    output = tmp_path / "report"
    report.report(completed(tmp_path / "record", native=native), output)
    for path in output.glob("*.svg"):
        root = ET.parse(path).getroot()
        assert not root.findall(".//{http://www.w3.org/2000/svg}script")
    with (output / "metrics.csv").open() as stream:
        metrics = list(csv.DictReader(stream))
    assert any(row["channel"].startswith("'=SUM") for row in metrics)
    assert any(row["channel"] == "joint,with\nnewline" for row in metrics)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["contract"]["channels"] == native["channels"]


@pytest.mark.parametrize("value", ["=1+1", "  =1+1", "\n=1+1", "\t +1+1", "\r@formula", " -2", "+3"])
def test_csv_formula_escaping_including_whitespace(value):
    assert report.csv_text(value) == "'" + value
    assert report.csv_text("native_joint") == "native_joint"


@pytest.mark.parametrize("fault", ["payload", "duplicate", "nonfinite", "ids", "calibration", "manifest"])
def test_same_validation_guards_as_analyze_refuse_before_report_creation(tmp_path, fault):
    samples = rows()
    if fault == "duplicate":
        samples[1] = copy.deepcopy(samples[0])
    if fault == "nonfinite":
        samples[0]["candidate_model"][0][0] = float("nan")
    source = completed(tmp_path / "record", samples=samples)
    if fault == "payload":
        with (source / "actions.jsonl").open("a") as stream:
            stream.write("{}\n")
    elif fault in ("ids", "manifest"):
        header = json.loads((source / "header.json").read_text())
        header["input_manifest"]["observations"][0]["frame_index"] = 999
        if fault == "ids":
            content = header["input_manifest"]
            content["content_sha256"] = ca.canonical_digest({k: v for k, v in content.items() if k != "content_sha256"})
        (source / "header.json").write_text(json.dumps(header))
        completion = json.loads((source / "completion.json").read_text())
        completion["header_sha256"] = ca.digest(source / "header.json")
        (source / "completion.json").write_text(json.dumps(completion))
    elif fault == "calibration":
        (source / "calibration_verification.json").write_text("{}")
    for runner in (lambda: report.report(source, tmp_path / "report"),
                   lambda: ca.analyze(argparse.Namespace(record_dir=source, physical=False, out=tmp_path / "summary"))):
        with pytest.raises(ValueError):
            runner()
    assert not (tmp_path / "report").exists() and not (tmp_path / "summary").exists()


@pytest.mark.parametrize("mutation", ["bool_horizon", "unit_length", "duplicate_channels", "empty_channel",
                                      "string_channels", "string_units"])
def test_malformed_contract_refused(mutation):
    native = contract()
    if mutation == "bool_horizon":
        native["returned_horizon"] = True
    elif mutation == "unit_length":
        native["units"] = ["unknown"]
    elif mutation == "duplicate_channels":
        native["channels"] = ["same", "same"]
    elif mutation == "empty_channel":
        native["channels"][0] = ""
    elif mutation == "string_channels":
        native["channels"] = "ab"
    else:
        native["units"] = "xy"
    with pytest.raises(ValueError):
        ca.analyze_records(rows(), native)


def test_missing_task_id_kept_unknown_and_human_labels_do_not_merge_ids():
    samples = rows()
    samples[0].pop("task_id")
    samples[1]["task_name"] = samples[2]["task_name"] = "same human label"
    result = ca.analyze_records(samples, contract())
    assert result["task_grouping"]["missing_task_id_chunks"] == 1
    assert [group["task_id"] for group in result["groups"]["tasks"]] == [None, 1, 2]
    assert result["per_chunk_components"]["observation_ids"][0]["task_id"] is None


def test_derived_metric_overflow_refused_and_no_model_imports(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="derived measurement"):
        ca.statistics([1e308, -1e308])
    source = completed(tmp_path / "record")
    original = builtins.__import__
    def restricted(name, *args, **kwargs):
        if name.startswith(("torch", "lerobot", "ref_loader", "load_quantized")):
            pytest.fail("offline reporting must not import model/GPU implementation: " + name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", restricted)
    assert report.report(source, tmp_path / "report")["complete"]


def test_loader_hashes_exact_bytes_it_parses_despite_later_file_change(tmp_path, monkeypatch):
    from pathlib import Path
    source = completed(tmp_path / "record")
    original = Path.read_bytes
    def changing_read(path):
        value = original(path)
        if path == source / "actions.jsonl":
            changed = rows()
            changed[0]["candidate_dataset"][0][0] = 999
            path.write_text("".join(json.dumps(row) + "\n" for row in changed))
        return value
    monkeypatch.setattr(Path, "read_bytes", changing_read)
    parsed, _, _, _ = ca.load_verified_recording(source)
    assert parsed[0]["candidate_dataset"][0][0] == -1
    # A subsequent load sees the changed payload and refuses its old completion hash.
    with pytest.raises(ValueError, match="digest differs"):
        ca.load_verified_recording(source)


@pytest.mark.parametrize("points", [True, 0, 1, 4097, 2.5])
def test_invalid_chart_sampling_refused(points, tmp_path):
    with pytest.raises(ValueError):
        report.report(tmp_path / "absent", tmp_path / "report", ecdf_points=points)
