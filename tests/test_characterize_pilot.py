"""Pinned pilot shell/control tests use synthetic stubs, never a model or GPU."""
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import characterize_actions as ca
import pytest


def export_fixture(path):
    path.mkdir()
    (path / "policy.json").write_text("{}")
    (path / "model.safetensors").write_bytes(b"synthetic test, not tensor payload")
    manifest = {"files": {name: {"sha256": ca.digest(path / name), "bytes": (path / name).stat().st_size}
                           for name in ("policy.json", "model.safetensors")}}
    ca.write_new(path / "manifest.json", manifest)
    return ca.digest(path / "manifest.json")


def test_pin_verification_never_loads_tensors_and_checks_every_payload(tmp_path):
    root = tmp_path / "export"
    pin = export_fixture(root)
    result = ca.verify_export_pin(root, pin)
    assert result["manifest_sha256"] == pin and len(result["payloads"]) == 2
    assert not result["fitting_executed"] and not result["model_forward_executed"]
    with pytest.raises(ValueError, match="artifact pin"):
        ca.verify_export_pin(root, "0" * 64)
    (root / "policy.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="payload digest"):
        ca.verify_export_pin(root, pin)


@pytest.mark.parametrize("mutation", ["missing", "path", "short_hash", "size"])
def test_pinned_manifest_still_requires_safe_complete_payloads(tmp_path, mutation):
    root = tmp_path / "export"
    export_fixture(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if mutation == "missing":
        del manifest["files"]["model.safetensors"]
    elif mutation == "path":
        manifest["files"]["../outside"] = copy.deepcopy(manifest["files"]["policy.json"])
    elif mutation == "short_hash":
        manifest["files"]["policy.json"]["sha256"] = "abc"
    else:
        manifest["files"]["policy.json"]["bytes"] += 1
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        ca.verify_export_pin(root, ca.digest(root / "manifest.json"))


STUB = '''import hashlib,json,os,sys
from pathlib import Path
def canonical_digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def write_new(p,x):
 with Path(p).open('x') as f: json.dump(x,f)
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_verified_recording(p): return [],json.loads((Path(p)/'header.json').read_text()),{},{}
if __name__ == '__main__':
 a=sys.argv[1:]; cmd=a.pop(0)
 def arg(k): return a[a.index(k)+1]
 def put(p,x):
  with Path(p).open('x') as f: json.dump(x,f)
 if cmd=='verify-export-pin': put(arg('--out'),{'synthetic':True})
 elif cmd=='preflight':
  m={'calibration':{'status':'existing_export'},'export':{'manifest_sha256':os.environ['EXPORT_MANIFEST_SHA256']},
   'sampling':{'chunks':100,'episodes':list(range(10)),'frames_per_episode':10},
   'observations':[{}]*100,'action_contract':{'returned_horizon':50,'dimension':6},'task':{'id':0}}
  m['content_sha256']=canonical_digest(m); put(arg('--out'),m)
 elif cmd=='record':
  Path(arg('--out-dir')).mkdir(); put(Path(arg('--out-dir'))/'completion.json',{'synthetic':True})
  put(Path(arg('--out-dir'))/'header.json',{'source_commit':os.environ['PILOT_EXPECTED_COMMIT'],
   'input_manifest_sha256':digest(arg('--manifest')),'export_manifest_sha256':os.environ['EXPORT_MANIFEST_SHA256']})
 elif cmd=='analyze': put(arg('--out'),{'synthetic':True})
'''
REPORT_STUB = '''import json,sys
from pathlib import Path
a=sys.argv[1:]; out=Path(a[a.index('--out-dir')+1]); out.mkdir()
with (out/'report_manifest.json').open('x') as f: json.dump({'synthetic':True},f)
'''


@pytest.fixture
def pilot(tmp_path):
    source = tmp_path / "source"
    (source / "scripts").mkdir(parents=True)
    (source / "src").mkdir()
    shutil.copyfile(Path(__file__).resolve().parents[1] / "scripts/run_characterization_pilot.sh",
                    source / "scripts/run_characterization_pilot.sh")
    (source / "src/characterize_actions.py").write_text(STUB)
    (source / "src/characterize_report.py").write_text(REPORT_STUB)
    (source / ".gitignore").write_text("__pycache__/\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Verification", "-c", "user.email=verification@example.invalid",
        "commit", "-qm", "test fixture")
    commit = git("rev-parse", "HEAD")
    git("checkout", "--detach", "-q", commit)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    root = tmp_path / "export"
    pin = export_fixture(root)
    env = {**os.environ, "MODEL_CONFIG": str(model / "config.json"), "DATASET_ROOT": str(dataset),
           "EXPORT_DIR": str(root), "EXPORT_MANIFEST_SHA256": pin, "PILOT_DIR": str(tmp_path / "pilot"),
           "PILOT_EXPECTED_COMMIT": commit, "PILOT_PYTHON": sys.executable}
    def run(*args):
        return subprocess.run(["bash", str(source / "scripts/run_characterization_pilot.sh"), *args],
                              env=env, text=True, capture_output=True)
    return source, env, run


def test_preflight_resume_records_reports_and_never_overwrites(pilot):
    _, env, run = pilot
    root = Path(env["PILOT_DIR"])
    first = run("--preflight-only")
    assert first.returncode == 0, first.stderr
    assert "CHARACTERIZATION_STARTED" in first.stdout and "PILOT_PREFLIGHT_DONE" in first.stdout
    frozen = ca.digest(root / "input_manifest.json")
    second = run("--record-only", "--device", "cuda:0")
    assert second.returncode == 0, second.stderr
    assert "CHARACTERIZATION_PILOT_DONE" in second.stdout
    assert (root / "report/report_manifest.json").exists()
    assert ca.digest(root / "input_manifest.json") == frozen
    assert run("--record-only").returncode != 0
    assert run("--report-only").returncode != 0
    assert run().returncode != 0


def test_default_run_includes_latest_report(pilot):
    _, env, run = pilot
    result = run()
    assert result.returncode == 0, result.stderr
    assert (Path(env["PILOT_DIR"]) / "report/report_manifest.json").exists()


@pytest.mark.parametrize("argument", ["--manifest", "--manifest=/tmp/other", "--export-dir", "--out-dir",
                                         "--dataset-root=/tmp/other", "--model-root", "-h"])
def test_forwarded_options_cannot_override_frozen_job_paths(pilot, argument):
    _, env, run = pilot
    result = run(argument, "/tmp/other")
    assert result.returncode == 2 and "Only device/trace options" in result.stderr
    assert not Path(env["PILOT_DIR"]).exists()


@pytest.mark.parametrize("field", ["input_manifest_sha256", "export_manifest_sha256", "source_commit"])
def test_report_only_refuses_recording_from_other_frozen_job(pilot, field):
    _, env, run = pilot
    root = Path(env["PILOT_DIR"])
    assert run("--preflight-only").returncode == 0
    paired = root / "paired"
    paired.mkdir()
    header = {"input_manifest_sha256": ca.digest(root / "input_manifest.json"),
              "export_manifest_sha256": env["EXPORT_MANIFEST_SHA256"], "source_commit": env["PILOT_EXPECTED_COMMIT"]}
    header[field] = "wrong"
    ca.write_new(paired / "header.json", header)
    result = run("--report-only")
    assert result.returncode != 0 and "PILOT_STAGE_FAILED" in result.stderr
    assert not (root / "report").exists() and not (root / "summary.json").exists()


@pytest.mark.parametrize("fault", ["wrong_source", "dirty_source", "relative_path", "wrong_frozen_contract"])
def test_starter_refuses_before_stub_inference(pilot, fault):
    source, env, run = pilot
    root = Path(env["PILOT_DIR"])
    mode = []
    if fault == "wrong_source":
        env["PILOT_EXPECTED_COMMIT"] = "0" * 40
    elif fault == "dirty_source":
        (source / "unexpected.txt").write_text("dirty")
    elif fault == "relative_path":
        env["MODEL_CONFIG"] = "relative/config.json"
    else:
        assert run("--preflight-only").returncode == 0
        manifest = json.loads((root / "input_manifest.json").read_text())
        manifest["action_contract"]["dimension"] = 7
        manifest["content_sha256"] = ca.canonical_digest({k: v for k, v in manifest.items() if k != "content_sha256"})
        (root / "input_manifest.json").write_text(json.dumps(manifest))
        mode = ["--record-only"]
    assert run(*mode).returncode != 0
    assert not (root / "paired").exists()
