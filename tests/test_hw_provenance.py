"""The hardware-verification sidecar (src/scripts/hw_provenance.py): hashes, counts and scope."""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))

import hw_provenance as hp  # noqa: E402


@pytest.fixture
def export(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"weights")
    (d / "manifest.json").write_text("{}")
    return d


def write(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj))
    return p


def test_record_ties_results_commits_and_export(tmp_path, export):
    l0 = write(tmp_path, "l0.json", {"total_cases": 100, "total_mismatches": 2})
    l1 = write(tmp_path, "l1.json", {"total_comparisons": 50, "total_mismatches": 0})
    rec = hp.build_record(export, [l0, l1], commits={"rtl": "a" * 40, "npu_model": "b" * 40, "atlas_torchao": "c" * 40},
                          now=datetime(2026, 10, 29, tzinfo=timezone.utc))
    assert rec["export"]["model_safetensors_sha256"] == hashlib.sha256(b"weights").hexdigest()
    assert rec["commits"] == {"rtl": "a" * 40, "npu_model": "b" * 40, "atlas_torchao": "c" * 40}
    assert [(r["file"], r["comparisons"], r["mismatches"]) for r in rec["results"]] == [("l0.json", 100, 2), ("l1.json", 50, 0)]
    assert rec["results"][1]["sha256"] == hashlib.sha256(l1.read_bytes()).hexdigest()
    assert "Not a full-SoC" in rec["scope"]


def test_an_unrecognized_result_is_refused(tmp_path, export):
    with pytest.raises(ValueError):
        hp.build_record(export, [write(tmp_path, "x.json", {"nothing": 1})], commits={})


def test_a_changed_result_changes_its_digest(tmp_path, export):
    p = write(tmp_path, "l1.json", {"total_comparisons": 5, "total_mismatches": 0})
    first = hp.summarize_result(p)["sha256"]
    write(tmp_path, "l1.json", {"total_comparisons": 5, "total_mismatches": 1})
    assert hp.summarize_result(p)["sha256"] != first
