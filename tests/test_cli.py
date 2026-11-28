"""src/atlas.py: the `atlas` CLI is a translation into the existing commands, which keep working unchanged.

- translate(): each subcommand's mapping, defaults, family detection and usage errors (pure, no model).
- The translated argv is accepted by the real legacy parser, run through atlas.run() exactly as the CLI runs it (the
  parse is intercepted before any model loads), and gives the same namespace as the documented legacy command.
- Every existing entry point still parses (--help) and the protected imports still resolve.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import atlas

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TILE = SRC / "scripts" / "tile_flow"


def tr(*argv):
    target, legacy, dry = atlas.translate(list(argv))
    return target, legacy


@pytest.fixture
def llm_dir(tmp_path):
    d = tmp_path / "tiny"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}))
    return str(d)


@pytest.fixture
def policy_dir(tmp_path):
    d = tmp_path / "pol"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"type": "smolvla", "n_obs_steps": 1}))
    return str(d)


# ------------------------------------------------------------------------------------------------ translate()

def test_export_llm_defaults(llm_dir):
    target, legacy = tr("export", llm_dir, "/x/export", "--device", "cpu")
    assert target == ("path", str(SRC / "eval.py"))
    assert legacy == ["--model", llm_dir, "--modes", "ref_conv", "--export-only", "--algo", "atlas_v1_awq_tile_static",
                      "--out", "/x/export_cli.json", "--export-dir", "/x/export", "--device", "cpu"]


def test_export_default_recipe_is_explicit_not_the_alias():
    import algos

    assert atlas.DEFAULT_ALGO in algos.RECIPES and atlas.DEFAULT_ALGO not in algos.ALIASES


def test_export_explicit_options_replace_the_defaults(llm_dir):
    _, legacy = tr("export", llm_dir, "/x/e/", "--algo", "none", "--modes", "ref_attn", "--out", "r.json",
                   "--export-only")
    assert legacy == ["--model", llm_dir, "--export-dir", "/x/e/", "--algo", "none", "--modes", "ref_attn",
                      "--out", "r.json", "--export-only"]
    _, legacy = tr("export", llm_dir, "/x/e", "--algo=gptq", "--modes=ref_conv")
    assert "atlas_v1_awq_tile_static" not in legacy and legacy.count("--modes") == 0
    assert legacy[legacy.index("--out") + 1] == "/x/e_cli.json"       # a trailing / does not change the record path


def test_export_policy_by_dataset():
    target, legacy = tr("export", "lerobot/smolvla_base", "e", "--dataset", "lerobot/svla_so100_pickplace",
                        "--norm-stats", "dataset")
    assert target == ("path", str(SRC / "eval_policy.py"))
    assert legacy[:2] == ["--policy", "lerobot/smolvla_base"]
    assert legacy[-4:] == ["--dataset", "lerobot/svla_so100_pickplace", "--norm-stats", "dataset"]
    assert "--norm-stats" not in legacy[:-4]          # no numerics choice is added


def test_export_dronet_by_npz():
    target, legacy = tr("export", "w.npz", "e", "--calib-cache", "v.npz")
    assert target == ("path", str(SRC / "eval_dronet.py"))
    assert legacy[:2] == ["--weights", "w.npz"] and legacy[-2:] == ["--calib-cache", "v.npz"]


def test_eval_is_a_plain_passthrough(llm_dir):
    target, legacy = tr("eval", llm_dir, "--from-export", "e", "--modes", "bf16", "ref_conv", "--out", "r.json")
    assert target == ("path", str(SRC / "eval.py"))
    assert legacy == ["--model", llm_dir, "--from-export", "e", "--modes", "bf16", "ref_conv", "--out", "r.json"]


def test_family_detection(llm_dir, policy_dir):
    assert atlas.detect_family(llm_dir, []) == "llm"
    assert atlas.detect_family(policy_dir, ["--dataset", "d"]) == "policy"
    assert atlas.detect_family(policy_dir, ["--dataset=d"]) == "policy"
    assert atlas.detect_family("/any/weights.npz", []) == "dronet"
    with pytest.raises(atlas.UsageError, match="LeRobot policy .*--dataset"):
        atlas.detect_family(policy_dir, [])
    with pytest.raises(atlas.UsageError, match="--family"):
        atlas.detect_family("/does/not/exist", [])


def test_family_detection_reads_the_hf_cache_offline(tmp_path, monkeypatch):
    snap = tmp_path / "hub" / "models--org--m" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "hub" / "models--org--m" / "refs").mkdir()
    (tmp_path / "hub" / "models--org--m" / "refs" / "main").write_text("abc")
    import huggingface_hub.constants as hc

    monkeypatch.setattr(hc, "HF_HUB_CACHE", str(tmp_path / "hub"))
    assert atlas.detect_family("org/m", []) == "llm"


def test_explicit_family_wins(policy_dir):
    target, legacy = tr("eval", policy_dir, "--family", "llm")
    assert target[1].endswith("eval.py") and legacy == ["--model", policy_dir]
    with pytest.raises(atlas.UsageError, match="--family must be one of"):
        tr("eval", policy_dir, "--family", "pi0")


def test_positionals_come_first(llm_dir):
    with pytest.raises(atlas.UsageError, match="missing MODEL EXPORT_DIR"):
        tr("export", "--device", "cpu", llm_dir, "e")
    with pytest.raises(atlas.UsageError, match="missing EXPORT_DIR"):
        tr("export", llm_dir, "--device", "cpu")
    with pytest.raises(atlas.UsageError, match="unknown subcommand"):
        tr("quantize", llm_dir)


def test_fx():
    assert tr("fx", "e", "fx", "--seqlen", "64") == (
        ("module", "exporter.fx_export"), ["--export-dir", "e", "--out-dir", "fx", "--native-fp8", "--seqlen", "64"])
    assert tr("fx", "e", "fx", "--no-native-fp8")[1] == ["--export-dir", "e", "--out-dir", "fx"]
    assert tr("fx", "e", "fx", "--native-fp8")[1] == ["--export-dir", "e", "--out-dir", "fx", "--native-fp8"]
    assert tr("fx", "--verify-dir", "fx", "--device", "cpu")[1] == ["--verify-dir", "fx", "--device", "cpu"]
    with pytest.raises(atlas.UsageError):
        tr("fx", "e", "--verify-dir", "fx")


def test_verify():
    target, legacy = tr("verify", "e", "fx/", "--require-pack-range")
    assert target == ("path", str(TILE / "verify_bundle.py"))
    assert legacy == ["e", "fx/", "fx_verify_bundle.json", "--require-pack-range"]
    assert tr("verify", "e", "fx", "o.json")[1] == ["e", "fx", "o.json"]


def test_dry_run_and_help(llm_dir, capsys):
    assert atlas.main(["export", llm_dir, "e", "--dry-run"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("python ") and "eval.py --model" in out and "--dry-run" not in out
    assert atlas.main(["export", "--help"]) == 0 and "usage: atlas export" in capsys.readouterr().out
    assert atlas.main(["--help"]) == 0 and "atlas export MODEL EXPORT_DIR" in capsys.readouterr().out
    assert atlas.main([]) == 2
    assert atlas.main(["export", llm_dir]) == 2 and "missing EXPORT_DIR" in capsys.readouterr().err


# --------------------------------------------------------------- the legacy parsers accept the translated argv

class _Parsed(Exception):
    pass


_PARSE_ARGS = argparse.ArgumentParser.parse_args


def run_legacy(monkeypatch, target, legacy):
    """atlas.run(target, legacy) as the CLI runs it, stopped at the legacy parser's parse_args (nothing past argument
    parsing runs); returns (its namespace as a dict, sys.argv[0] the command saw)."""
    seen = {}

    def capture(self, args=None, namespace=None):
        seen["ns"] = _PARSE_ARGS(self, args, namespace)
        seen["argv0"] = sys.argv[0]
        raise _Parsed

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(_Parsed):
        atlas.run(target, legacy)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", _PARSE_ARGS)
    return vars(seen["ns"]), seen["argv0"]


def parsed_by_legacy(monkeypatch, argv):
    """`atlas <argv>`: translated, then run as run_legacy does."""
    target, legacy, _ = atlas.translate(list(argv))
    return run_legacy(monkeypatch, target, legacy)


def test_llm_export_namespace_equals_the_guide_command(monkeypatch, llm_dir, tmp_path):
    smoke = str(tmp_path / "smoke")
    new, argv0 = parsed_by_legacy(monkeypatch, ["export", llm_dir, smoke + "/export", "--seqlen", "128",
                                                "--calib-n", "4", "--device", "cpu"])
    assert argv0 == str(SRC / "eval.py")
    old, _ = run_legacy(monkeypatch, ("path", str(SRC / "eval.py")),
                     ["--model", llm_dir, "--seqlen", "128", "--calib-n", "4", "--modes", "ref_conv", "--export-only",
                      "--algo", "atlas_v1_awq_tile_static", "--device", "cpu", "--export-dir", smoke + "/export",
                      "--out", smoke + "/export_cli.json"])
    assert new == old
    assert new["export_only"] and new["modes"] == ["ref_conv"] and new["model"] == llm_dir


def test_policy_export_namespace_equals_the_guide_command(monkeypatch, tmp_path):
    e = str(tmp_path / "smolvla" / "export")
    new, _ = parsed_by_legacy(monkeypatch, ["export", "lerobot/smolvla_base", e, "--dataset",
                                            "lerobot/svla_so100_pickplace", "--norm-stats", "dataset", "--seed", "0",
                                            "--device", "cpu"])
    old, _ = run_legacy(monkeypatch, ("path", str(SRC / "eval_policy.py")),
                     ["--policy", "lerobot/smolvla_base", "--dataset", "lerobot/svla_so100_pickplace", "--norm-stats",
                      "dataset", "--seed", "0", "--modes", "ref_conv", "--export-only", "--algo",
                      "atlas_v1_awq_tile_static", "--device", "cpu", "--export-dir", e, "--out", e + "_cli.json"])
    assert new == old and new["policy"] == "lerobot/smolvla_base"


def test_dronet_export_parses(monkeypatch, tmp_path):
    new, _ = parsed_by_legacy(monkeypatch, ["export", str(tmp_path / "w.npz"), str(tmp_path / "e"),
                                            "--calib-cache", "v.npz", "--algo", "static_calib"])
    assert new["weights"].endswith("w.npz") and new["export_only"] and new["algo"] == "static_calib"


def test_fx_and_verify_parse(monkeypatch):
    new, argv0 = parsed_by_legacy(monkeypatch, ["fx", "e", "fx", "--seqlen", "64", "--device", "cpu"])
    assert new["export_dir"] == "e" and new["out_dir"] == "fx" and new["native_fp8"] and new["seqlen"] == 64
    assert argv0.endswith(os.path.join("exporter", "fx_export.py"))          # as `python -m exporter.fx_export`
    new, _ = parsed_by_legacy(monkeypatch, ["verify", "e", "fx", "--allow-uncalibrated-fallback"])
    assert new == {"export": "e", "fx": "fx", "out": "fx_verify_bundle.json", "allow_uncalibrated_fallback": True,
                   "forbid_unquantized_attention": False, "require_pack_range": False}


def test_run_restores_sys_state(monkeypatch):
    argv, path, main = sys.argv[:], sys.path[:], sys.modules.get("__main__")
    parsed_by_legacy(monkeypatch, ["verify", "e", "fx"])
    assert sys.argv == argv and sys.path == path and sys.modules.get("__main__") is main


# ------------------------------------------------------------------------- the existing entry points still work

LEGACY_HELP = [
    ["src/eval.py"], ["src/eval_policy.py"], ["src/eval_dronet.py"], ["-m", "exporter.fx_export"],
    ["src/scripts/tile_flow/verify_bundle.py"], ["src/scripts/tile_flow/make_loop_manifest.py"],
    ["src/scripts/tile_flow/policy_fx_capture_phases.py"], ["-m", "atlas"],
]


@pytest.mark.parametrize("cmd", LEGACY_HELP, ids=lambda c: c[-1])
def test_entry_point_parses(cmd):
    env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, *cmd, "--help"], cwd=ROOT, env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "usage:" in r.stdout or "atlas export MODEL EXPORT_DIR" in r.stdout


def test_python_dash_m_atlas_dry_run(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "-m", "atlas", "export", str(tmp_path / "w.npz"), "e", "--dry-run"],
                       cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split()[1].endswith("eval_dronet.py") and "--weights" in r.stdout


def test_protected_imports():
    import chip_coverage  # noqa: F401  (multitask-dit-quant: load_op_emulator("chip_coverage"))
    from exporter.load_quantized import load_quantized

    assert callable(load_quantized)
