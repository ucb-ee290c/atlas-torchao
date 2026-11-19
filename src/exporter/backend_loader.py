"""Saved native-FP8 FX bundle -> Atlas backend capture (Merlin's m2m capture worker, atlas_fp8 adapter).

This file is a *loader* for `merlin.targetgen._m2m_capture_worker --loader`. It runs inside the backend's
compiler interpreter, not in this repo's environment, and is loaded by path, so it imports nothing from
this package: the adapter registers the source custom ops itself (`import atlas_fx_ops`, the pinned
atlas-torchao flat module on PYTHONPATH). Importing `exporter.atlas_fx_ops` here as well would define the
`atlas` op namespace twice.

What it does (no tensor is converted, re-quantized or re-exported):
  1. checks the selected export receipts (manifest.json / policy.json SHA-256 frozen by the caller);
  2. loads the saved native program `<bundle>/<fx>/model.pt2` (atlas ops registered first);
  3. selects one fixture case of the bundle (`fixtures.safetensors`, `parity.json`) as the example inputs;
  4. hands the saved program to the backend's `atlas_mlir_native.capture.prepare_model`, which runs the
     installed `atlas_fp8` adapter on it (saved-program branch: graph identity, lifted-state authentication
     against the export tensors, native call census) and returns the ExportedProgram the worker converts.

Environment (all required unless marked):
  ATLAS_FX_BUNDLE                   bundle root holding export/ and the FX directory
  ATLAS_FX_DIR                      FX directory relative to the bundle (default "fx")
  ATLAS_EXPORT_DIR                  export directory (default <bundle>/export)
  ATLAS_EXPECTED_CONTRACT_SHA256    SHA-256 of export/manifest.json from the source receipt
  ATLAS_EXPECTED_POLICY_SHA256      SHA-256 of export/policy.json from the source receipt
  ATLAS_FX_CASE                     fixture case index (default 0)
  ATLAS_EXPORT_PREFIX               state-name prefix of the capture wrapper (default: derived, see below)

`write_boundary(bundle, fx, case, out)` (also `python backend_loader.py --write-boundary ...`) writes the
selected case as `inputs.safetensors` + `source.json` in the form the unchanged_fx packager
(`prepare_live_execution.py --boundary-source/--inputs`) authenticates, and the case's expected outputs as
`expected.npz` (raw words, BF16 as uint16) for the word-exact comparison after execution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

ATLAS_OPS_MODULE = "atlas_fx_ops"


def _sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _raw(tensor):
    return tensor.detach().contiguous().reshape(-1).view(__import__("torch").uint8).numpy().tobytes()


def tensor_record(tensor):
    raw = _raw(tensor)
    return {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "sha256": hashlib.sha256(raw).hexdigest()}


def select_case(fx_dir, case=0):
    """(input tensors, expected output tensors) of one fixture case, in the bundle's recorded order."""
    from safetensors.torch import load_file

    fx_dir = Path(fx_dir)
    manifest = json.loads((fx_dir / "fx_manifest.json").read_text())
    for name in ("fixtures.safetensors", "parity.json", "model.pt2"):
        record = manifest["files"].get(name)
        if record is None or (fx_dir / name).stat().st_size != record["bytes"]:
            raise ValueError(f"FX bundle file differs from its manifest: {name}")
    if _sha_file(fx_dir / "fixtures.safetensors") != manifest["files"]["fixtures.safetensors"]["sha256"]:
        raise ValueError("fixtures.safetensors differs from fx_manifest.json")
    parity = json.loads((fx_dir / "parity.json").read_text())
    selected = parity["cases"][case]
    fixtures = load_file(str(fx_dir / "fixtures.safetensors"))
    return [fixtures[k] for k in selected["inputs"]], [fixtures[k] for k in selected["outputs"]]


def export_prefix(program, tensors):
    """The wrapper prefix that maps every lifted state name onto an export tensor name.

    CausalLogits holds the model as `.model` ("model."), SmolVLAFlowLoss the policy as `.policy` ("policy."),
    phase wrappers may use others. Exactly one prefix of the first state name must map it onto the export.
    Meta placeholders (the float weights of quantized layers, never materialized) are not export tensors; the
    adapter checks them against the policy's weight shape and dtype instead."""
    names = [n for n, v in program.state_dict.items() if v.device.type != "meta"]
    names += [n for n in program.constants if n in {str(s.target) for s in program.graph_signature.input_specs
                                                      if str(s.kind).endswith("BUFFER")}]
    if not names:
        return ""
    first = names[0]
    parts = first.split(".")
    candidates = [".".join(parts[:i]) + "." if i else "" for i in range(len(parts))]
    matches = [p for p in candidates if first[len(p):] in tensors and all(
        n.startswith(p) and n[len(p):] in tensors for n in names)]
    if len(matches) != 1:
        raise ValueError(f"no unique wrapper prefix maps the saved program state onto the export: {matches}")
    return matches[0]


def get_model_and_inputs():
    from importlib import import_module

    import torch
    from atlas_mlir_native.capture import load_export, prepare_model

    bundle = Path(os.environ["ATLAS_FX_BUNDLE"]).absolute()
    fx_dir = bundle / os.environ.get("ATLAS_FX_DIR", "fx")
    export = Path(os.environ.get("ATLAS_EXPORT_DIR") or bundle / "export").absolute()
    expected_contract = os.environ["ATLAS_EXPECTED_CONTRACT_SHA256"]
    expected_policy = os.environ["ATLAS_EXPECTED_POLICY_SHA256"]
    for path, expected in ((export / "manifest.json", expected_contract), (export / "policy.json", expected_policy)):
        if _sha_file(path) != expected:
            raise ValueError(f"selected source receipt digest differs: {path}")
    manifest = json.loads((fx_dir / "fx_manifest.json").read_text())
    if not manifest.get("native_fp8"):
        raise ValueError("bundle is not a native FP8 FX capture (fx_manifest.json has no native_fp8 record)")
    program_path = fx_dir / "model.pt2"
    program_sha = _sha_file(program_path)
    if program_sha != manifest["files"]["model.pt2"]["sha256"]:
        raise ValueError("model.pt2 differs from fx_manifest.json")
    inputs, _ = select_case(fx_dir, int(os.environ.get("ATLAS_FX_CASE", "0")))
    program = load_export(program_path, ops_module=ATLAS_OPS_MODULE)
    tensors, _, _ = import_module("load_quantized").read_export(export, verify=True)
    prefix = os.environ.get("ATLAS_EXPORT_PREFIX")
    if prefix is None:
        prefix = export_prefix(program, tensors)
    program._atlas_saved_program_path = program_path
    program._atlas_saved_program_sha256 = program_sha
    return prepare_model(program, tuple(inputs), export / "manifest.json", export / "policy.json",
                         export_prefix=prefix, expected_contract_sha256=expected_contract,
                         expected_policy_sha256=expected_policy)


def normalize_capture(module, manifest):
    """Worker hook: the backend's own Atlas source normalization (atlas_mlir_native.capture)."""
    from atlas_mlir_native.capture import normalize_capture as normalize

    return normalize(module, manifest)


def write_boundary(bundle, fx="fx", case=0, out=None, kind="saved_fx"):
    """Write inputs.safetensors, source.json and expected.npz for one fixture case of a saved FX bundle."""
    import numpy as np
    import torch
    from safetensors.torch import save_file

    fx_dir = Path(bundle) / fx
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    inputs, outputs = select_case(fx_dir, case)
    keys = [f"input_{i}" for i in range(len(inputs))]
    save_file({k: v.contiguous() for k, v in zip(keys, inputs)}, str(out / "inputs.safetensors"))
    manifest = json.loads((fx_dir / "fx_manifest.json").read_text())
    source = {"schema": "atlas_torchao.saved_fx_boundary.v1", "kind": kind, "case": case,
              "bundle": str(Path(bundle).absolute()), "fx": fx,
              "model_sha256": manifest["files"]["model.pt2"]["sha256"],
              "input_keys": keys, "tensors": {k: tensor_record(v) for k, v in zip(keys, inputs)},
              "outputs": {f"output_{i}": tensor_record(v) for i, v in enumerate(outputs)}}
    (out / "source.json").write_text(json.dumps(source, indent=2) + "\n")
    words = {f"output_{i}": (v.contiguous().view(torch.uint16).numpy() if v.dtype == torch.bfloat16 else v.numpy())
             for i, v in enumerate(outputs)}
    np.savez(out / "expected.npz", **words)
    return source


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write-boundary", required=True, type=Path, help="output directory")
    ap.add_argument("--bundle", required=True, type=Path)
    ap.add_argument("--fx", default="fx")
    ap.add_argument("--case", type=int, default=0)
    a = ap.parse_args(argv)
    source = write_boundary(a.bundle, a.fx, a.case, a.write_boundary)
    print(json.dumps({"inputs": source["tensors"], "outputs": source["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
