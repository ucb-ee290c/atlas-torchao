"""native_fp8 as an export mode (docs/algos/export.md#when-the-export-is-written).

native_fp8 quantizes with ref_conv's operand casts (the same weight codes and exponents); only the matmul runs on the
FP8 tensor cores. So `--modes bf16 native_fp8 --export-dir` writes the same model.safetensors tensors as the ref_conv
export of the same model and recipe, and the same policy.json / manifest.json apart from the fields that name the
mode. ref_conv wins when both modes are in --modes.

The export comparison drives src/eval.py on a tiny Llama (as tests/test_ladder.py does) and needs a CUDA GPU with FP8
tensor cores (compute capability >= 8.9); it skips on CPU. Tensors are compared, not file bytes: safetensors writes
its __metadata__ keys in a per-process order.
"""
import json
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

import exporter.quant_export as quant_export
from test_export import SPECS
from test_ladder import tiny_llama

gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="native_fp8 needs a CUDA GPU with FP8 tensor cores")

# manifest.json fields that differ between two exports of the same mode: the time, the digest of model.safetensors (its
# __metadata__ order varies per process), argv (it holds the mode and the output path) and the algorithms' timings
VOLATILE = ("created_utc", "files/model.safetensors/sha256", "provenance/argv/")
# the fields that name or describe the exported mode; nothing else may differ
MODE_POLICY = {"exported_mode", "profile", "accumulate/exported_mode_accumulation"}
MODE_MANIFEST = ("exported_mode", "ladder/", "files/policy.json/", "policy_sha256",
                 "recipe/algorithm_record/prepare/smoothquant/mode")


def test_pick_mode_native_fp8():
    """native_fp8 alone is exported; ref_conv wins when both are present, in either order."""
    assert quant_export.pick_mode(["bf16", "native_fp8"]) == "native_fp8"
    assert quant_export.pick_mode(["bf16", "native_fp8", "ref_conv"]) == "ref_conv"
    assert quant_export.pick_mode(["bf16", "ref_conv", "native_fp8"]) == "ref_conv"
    assert quant_export.pick_mode(["bf16", "torchao"]) is None


def _export(monkeypatch, tmp_path, name, mode, spec):
    """src/eval.py --modes bf16 <mode> --export-dir <tmp>/<name> --export-only on a tiny Llama (cuda)."""
    import algos.calib as algo_calib
    import eval as ev
    import loaders.datautils as datautils
    import loaders.modelutils as modelutils

    def load(path, seqlen=None, device="cpu"):
        m = tiny_llama(2).to(device)
        m.seqlen = seqlen
        return m

    g = torch.Generator().manual_seed(5)
    batches = [torch.randint(0, 128, (1, 16), generator=g) for _ in range(4)]
    ids = torch.randint(0, 128, (1, 64), generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(modelutils, "load_llama", load)
    monkeypatch.setattr(datautils, "get_wikitext2", lambda **k: (None, SimpleNamespace(input_ids=ids)))
    monkeypatch.setattr(algo_calib, "wikitext2_train",
                        lambda *a, **k: (batches, {"source": "test", "n_samples": len(batches), "seed": 5}))
    d = tmp_path / name
    monkeypatch.setattr(sys, "argv", ["eval.py", "--model", "x", "--seqlen", "16", "--modes", "bf16", mode,
                                      "--crosscheck", "off", "--device", "cuda", "--algo", spec,
                                      "--export-dir", str(d), "--export-only"])
    ev.main()
    return d


def _read(d):
    with safe_open(str(d / "model.safetensors"), "pt") as f:
        meta = f.metadata()
        tensors = {k: f.get_tensor(k) for k in f.keys()}
    return (tensors, meta, json.loads((d / "policy.json").read_text()),
            json.loads((d / "manifest.json").read_text()))


def _same_tensors(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert torch.equal(a[k].reshape(-1).view(torch.uint8), b[k].reshape(-1).view(torch.uint8)), k


def _flat(x, path=""):
    if isinstance(x, dict) and x:
        for k, v in x.items():
            yield from _flat(v, f"{path}{k}/")
    elif isinstance(x, list) and x:
        for i, v in enumerate(x):
            yield from _flat(v, f"{path}{i}/")
    else:
        yield path.rstrip("/"), x


def _diff(a, b):
    """Paths (a/b/c) whose leaf values differ, without the algorithms' wall-clock timings."""
    fa, fb = dict(_flat(a)), dict(_flat(b))
    return {p for p in fa.keys() | fb.keys() if fa.get(p, KeyError) != fb.get(p, KeyError) and "/timing_s/" not in p}


def _outside(paths, allowed):
    return {p for p in paths if not any(p == a or (a.endswith("/") and p.startswith(a)) for a in allowed)}


@gpu
@pytest.mark.parametrize("spec", [SPECS[0], SPECS[1], SPECS[4]], ids=["none", "atlas_v1", "atlas_v1_static"])
def test_native_fp8_export_equals_ref_conv_export(monkeypatch, tmp_path, spec):
    ref = _read(_export(monkeypatch, tmp_path, "ref_conv", "ref_conv", spec))
    ctl = _read(_export(monkeypatch, tmp_path, "ref_conv_again", "ref_conv", spec))
    nat = _read(_export(monkeypatch, tmp_path, "native_fp8", "native_fp8", spec))

    # control: two ref_conv exports agree, so a native_fp8 difference means something
    _same_tensors(ref[0], ctl[0])
    assert ref[1] == ctl[1] and ref[2] == ctl[2]
    assert _outside(_diff(ref[3], ctl[3]), VOLATILE) == set()

    tensors, meta, policy, manifest = nat
    r_tensors, r_meta, r_policy, r_manifest = ref
    # model.safetensors: byte-identical tensors; the metadata differs only in the mode name
    _same_tensors(tensors, r_tensors)
    assert meta == {**r_meta, "exported_mode": "native_fp8"} and r_meta["exported_mode"] == "ref_conv"

    # policy.json: the mode name, its profile and the exported mode's accumulation record, nothing else
    assert _diff(policy, r_policy) == MODE_POLICY
    assert (policy["exported_mode"], policy["profile"]) == ("native_fp8", "native_fp8")
    assert (r_policy["exported_mode"], r_policy["profile"]) == ("ref_conv", "ref_conv")
    assert r_policy["accumulate"]["exported_mode_accumulation"] == "framework"
    assert policy["accumulate"]["exported_mode_accumulation"].startswith("native fp8: torch._scaled_mm")

    # manifest.json: the mode name, the mode's ladder record, the digest of the policy.json that names it, and the mode
    # SmoothQuant recorded (static recipe); recipe, calibration, sites and counts are the same
    assert _outside(_diff(manifest, r_manifest), VOLATILE + MODE_MANIFEST) == set()
    assert manifest["exported_mode"] == "native_fp8" and r_manifest["exported_mode"] == "ref_conv"
    assert (manifest["ladder"]["mode"], manifest["ladder"]["profile"]) == ("native_fp8", "native_fp8")
    assert manifest["ladder"]["accumulation"] == policy["accumulate"]["exported_mode_accumulation"]
