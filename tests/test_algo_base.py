"""algo/base: --algo none is bit-identical to l40s-run-2.1 on every ladder step, and the hook
plumbing (registry, spec parsing, records, granularities, BF16 opt-out, calibration) works."""
import copy
import io
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import algos
import quant_modes as qm
from algos.base import Algo, Scale
from refutil import need_refs

REPO = Path(__file__).resolve().parent.parent
BASE_TAG = "l40s-run-2.1"
PROBE = Path(__file__).with_name("algo_probe.py")


def _bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


def _assert_same(a, b, label):
    assert a.keys() == b.keys(), label
    for k in a:
        ra, rb = a[k], b[k]
        assert torch.equal(_bits(ra["out"]), _bits(rb["out"])), f"{label}: output differs for {k}"
        assert ra["state"].keys() == rb["state"].keys(), f"{label}: state keys differ for {k}"
        for s in ra["state"]:
            assert ra["state"][s].dtype == rb["state"][s].dtype, f"{label}: {k} {s} dtype"
            assert torch.equal(_bits(ra["state"][s]), _bits(rb["state"][s])), f"{label}: {k} {s} differs"
        assert ra["ladder"] == rb["ladder"], f"{label}: ladder record differs for {k}"


def _export_base(tmp):
    try:
        blob = subprocess.check_output(["git", "-C", str(REPO), "archive", BASE_TAG, "src"],
                                       stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"tag {BASE_TAG} not available in {REPO}")
    with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
        tf.extractall(tmp, filter="data") if sys.version_info >= (3, 12) else tf.extractall(tmp)
    (tmp / "third_party").symlink_to(REPO / "third_party")
    return tmp / "src"


def _probe(src, out, spec=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.Popen([sys.executable, str(PROBE), str(src), str(out), *([spec] if spec else [])],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_none_and_identity_bit_identical_to_base_on_every_ladder_step(tmp_path):
    """Every ladder step (and collab_v0_hw) on a tiny Llama and a Conv2d net: the l40s-run-2.1
    tree, the current tree through quant_modes.quantize, through algos.quantize with --algo none,
    and with example_identity (every hook, explicit-scale path) give identical outputs, state dicts
    (persistent and non-persistent buffers) and ladder records."""
    need_refs()
    base_src = _export_base(tmp_path / "base")
    runs = {"base": (base_src, None), "head": (REPO / "src", None),
            "none": (REPO / "src", "none"), "identity": (REPO / "src", "example_identity")}
    procs = {k: _probe(src, tmp_path / f"{k}.pt", spec) for k, (src, spec) in runs.items()}
    for k, p in procs.items():
        log, _ = p.communicate(timeout=1800)
        assert p.returncode == 0, f"{k} probe failed:\n{log[-4000:]}"
    res = {k: torch.load(tmp_path / f"{k}.pt", weights_only=False) for k in runs}
    assert len(res["base"]) == 2 * (len(qm.LADDER) + 1)
    for k in ("head", "none", "identity"):
        _assert_same(res["base"], res[k], k)
    for key, r in res["none"].items():
        assert r["algo"] == "null", key                    # none adds nothing to the report
    for key, r in res["identity"].items():
        if not key.endswith("/bf16"):
            assert '"name": "example_identity"' in r["algo"], key


def test_parse_and_registry():
    assert algos.parse(None) == ("none", {})
    assert algos.parse("example_identity:calib=true,bf16=down_proj,n=3,x=0.5") == (
        "example_identity", {"calib": True, "bf16": "down_proj", "n": 3, "x": 0.5})
    assert algos.make("none") is None
    a = algos.make("example_identity")
    assert isinstance(a, Algo) and a.name == "example_identity"
    for bad in ("nope", "none:k=1", "example_identity:k"):
        with pytest.raises(ValueError):
            algos.parse(bad)
    for name, mod in algos.REGISTRY.items():
        assert mod is None or mod == f"algos.{name}"


def test_entry_records():
    assert algos.entry_record(None, "bf16", "none") == {"name": "none", "params": {}, "calibration": None}
    rec = algos.entry_record(None, "bf16", "example_identity:calib=true")
    assert rec["applied"] is False
    layers = {"algo": {"name": "x"}}
    assert algos.entry_record(layers, "ref_linear", "x") == {"name": "x"} and "algo" not in layers


class _Granular(Algo):
    """Per-channel weights, static per-tensor activations, an online x2 / weight /2 pair (exact)."""

    def __init__(self, **p):
        super().__init__(**p)
        self.static = {}

    def transform_weight(self, name, w, kind):
        return w / 2

    def weight_scale(self, name, w, kind):
        import ref_numerics as rn
        return Scale(rn.po2_scale_exp(w, (1,)), "per_channel")

    def act_scale(self, name, x, kind, dims=None):
        return Scale(torch.tensor(self.static.get(name, 0), dtype=torch.int32), "static", dynamic=False)

    def online_transform(self, name, kind):
        return lambda x: x * 2


def test_granularity_and_online_hooks_reference_paths():
    need_refs()
    torch.manual_seed(0)
    lin = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 8)).to(torch.bfloat16)
    x = torch.randn(4, 64).to(torch.bfloat16)
    ref = lin(x)
    for mode in ("ref_linear", "reference_hw"):
        m = copy.deepcopy(lin)
        algo = _Granular()
        rep = algos.quantize(m, mode, options={"crosscheck": "off", "mxu_impl": "torch"}, algo=algo)
        y = m(x)
        assert torch.isfinite(y.float()).all()
        assert ((y.float() - ref.float()).norm() / ref.float().norm()) < 0.1
        g = rep["algo"]["granularity"]
        assert g["weight"] == {"linear:per_channel": 2}
        assert g["activation"] == {"linear:static": 2} and g["online_transforms"] == {"linear": 2}
        assert rep["algo"]["module_hooks_applied"] is True


def test_per_block_framework_path_and_refusal_under_mxu():
    need_refs()
    import ref_numerics as rn

    class Block(Algo):
        def weight_scale(self, name, w, kind):
            wb = w.unflatten(-1, (-1, 32))
            return Scale(rn.po2_scale_exp(wb, (-1,)), "per_block", block=32)

    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(64, 16)).to(torch.bfloat16)
    x = torch.randn(3, 64).to(torch.bfloat16)
    ref = m(x)
    rep = algos.quantize(m, "ref_linear", algo=Block())
    assert rep["algo"]["granularity"]["weight"] == {"linear:per_block(block=32)": 1}
    assert ((m(x).float() - ref.float()).norm() / ref.float().norm()) < 0.1
    m2 = nn.Sequential(nn.Linear(64, 16)).to(torch.bfloat16)
    with pytest.raises(NotImplementedError):
        algos.quantize(m2, "reference_hw", options={"crosscheck": "off"}, algo=Block())


def test_bf16_opt_out_counted_as_host_in_chip_coverage():
    need_refs()
    from test_ladder import tiny_llama

    model = tiny_llama()
    algo = algos.make("example_identity:bf16=layers.0.mlp.down_proj")
    rep = algos.quantize(model, "chip", skip=["lm_head"],
                         options={"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}, algo=algo)
    assert rep["algo"]["bf16_opt_out"] == ["model.layers.0.mlp.down_proj"]
    assert type(dict(model.named_modules())["model.layers.0.mlp.down_proj"]) is nn.Linear
    with torch.no_grad():
        model(torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1)))
    rows = {h["op"]: h for h in rep["coverage"]["host_fallback"]}
    assert "mm@bf16_opt_out" in rows and "bf16_layers" in rows["mm@bf16_opt_out"]["reason"]
    assert "--skip lm_head" in rows["mm"]["reason"]


def test_calibration_hook_and_record():
    need_refs()
    from test_ladder import tiny_llama

    model = tiny_llama()
    algo = algos.make("example_identity:calib=true")
    batches = [torch.randint(0, 128, (1, 8), generator=torch.Generator().manual_seed(i)) for i in range(2)]
    calls = []

    def calib():
        calls.append(1)
        return batches, (lambda m, b: m(b)), {"source": "test", "n_samples": 2, "seed": 0}

    rep = algos.quantize(model, "ref_linear", skip=["lm_head"], algo=algo, calib=calib)
    c = rep["algo"]["calibration"]
    assert calls and c["source"] == "test" and c["n_samples"] == 2 and c["seed"] == 0
    assert c["result"]["layers_observed"] == 8        # 7 Linear in the one layer + lm_head
    with pytest.raises(RuntimeError):
        algos.quantize(tiny_llama(), "ref_linear", algo=algos.make("example_identity:calib=true"))


def test_collab_v0_gets_model_level_hooks_only():
    m = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)).to(torch.bfloat16)
    rep = algos.quantize(m, "collab_v0", algo=algos.make("example_identity:bf16=1"))
    assert rep["algo"]["bf16_opt_out"] == ["1"] and type(m[1]) is nn.Linear and type(m[0]) is not nn.Linear
    assert rep["algo"]["module_hooks_applied"] is False and "module_hooks_note" in rep["algo"]


def test_cli_flags_on_every_eval():
    import eval_pi0
    import eval_policy

    a = eval_policy.build_parser().parse_args(["--policy", "p", "--dataset", "d", "--out", "o"])
    assert a.algo == "none" and a.calib_source == "non_eval_episodes"
    b = eval_pi0.build_parser().parse_args(["--out", "o", "--algo", "example_identity:calib=true"])
    assert b.algo == "example_identity:calib=true" and b.calib_source == "auto"
    src = (REPO / "src" / "eval.py").read_text()
    assert "algos.add_args(ap" in src and "algos.quantize(" in src
