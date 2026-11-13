"""Per-phase FX capture for the Multi-Task DiT (and Pi0, see below): the encoder and one denoising step as two separate
graphs, with the host owning the loop (docs/algos/phase-graphs.md, docs/algos/dit-pi0-full-graph-plan.md items 2, 3, 5).

  dit encode:  DiTEncode  (state, images, input_ids, attention_mask)           -> conditioning_vec
               = observation_encoder.encode(batch)   (CLIP towers, patch-embedding Conv2d)
  dit step:    DiTStep    (conditioning_vec, noisy_actions, timestep)          -> predicted noise
               = noise_predictor(noisy_actions, timestep, conditioning_vec=cond)   (one call, no scheduler, no RNG)

The existing export is reused (read only) and the capture cases are the existing single-step bundle's fixture inputs, so the
composition (encode, then step on its output) can be compared byte for byte with that bundle's recorded outputs.
usage (PYTHONPATH=<worktree>/src, atlas-lerobot env):
  policy_fx_capture_phases.py dit encode|step --export-dir EXPORT --single-step-bundle BUNDLE --loop-manifest LM --out DIR
DIR gets fx/ (the bundle), loop_manifest.json (a copy, same sha256 as LM, recorded in fx_manifest.json provenance) and an
`export` symlink to EXPORT so verify_bundle.py / make_bundle_manifest.py can read it."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import fx_export
import load_quantized
import quant_export

OBS_NAMES = ["state", "images", "input_ids", "attention_mask"]


def _batch(state, images, input_ids, attention_mask):
    from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
    return {OBS_STATE: state, OBS_IMAGES: images, OBS_LANGUAGE_TOKENS: input_ids, OBS_LANGUAGE_ATTENTION_MASK: attention_mask}


class DiTEncode(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.model = policy   # attribute name 'model' = the path prefix verify_bundle.py expects

    def forward(self, state, images, input_ids, attention_mask):
        return self.model.observation_encoder.encode(_batch(state, images, input_ids, attention_mask))


class DiTStep(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.model = policy

    def forward(self, conditioning_vec, noisy_actions, timestep):
        return self.model.noise_predictor(noisy_actions, timestep, conditioning_vec=conditioning_vec)


def load_dit(export_dir, device="cpu"):
    """The quantized policy exactly as policy_fx_capture.dit_setup builds it, without the dataset (only the policy is needed)."""
    import eval_policy
    manifest = json.loads((Path(export_dir) / "manifest.json").read_text())
    ident = manifest["model"]["id"]
    args = eval_policy.build_parser().parse_args(
        ["--policy", ident, "--dataset", manifest["provenance"]["lerobot_dataset"], "--seed",
         str(manifest["provenance"]["eval_seed"]), "--n-frames", "2", "--modes", "ref_conv", "--export-only",
         "--algo", manifest["recipe"]["algo"], "--device", device, "--export-dir", "unused", "--out", "unused"])
    policy, cfg = eval_policy.load_policy(ident, device, torch.bfloat16)
    load_quantized.load_quantized(ident, export_dir, device, build=lambda: policy, mode="ref_conv", options=args)
    return policy, cfg


def fixture_cases(single_step_bundle, device="cpu"):
    """[(state, images, input_ids, attention_mask, noisy_actions, timestep), ...] and the recorded noise predictions."""
    from safetensors.torch import load_file
    fx = Path(single_step_bundle) / "fx"
    tensors = load_file(str(fx / "fixtures.safetensors"), device=device)
    parity = json.loads((fx / "parity.json").read_text())
    cases = [tuple(tensors[k] for k in c["inputs"]) for c in parity["cases"]]
    outputs = [tuple(tensors[k] for k in c["outputs"]) for c in parity["cases"]]
    return cases, outputs


def phase_cases(phase, policy, full_cases):
    """The phase boundary's capture cases from the full-boundary cases (encode first for the step phase)."""
    if phase == "encode":
        return [c[:4] for c in full_cases]
    with torch.no_grad():
        return [(DiTEncode(policy).eval()(*c[:4]), c[4], c[5]) for c in full_cases]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit",))
    ap.add_argument("phase", choices=("encode", "step"))
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--single-step-bundle", required=True)
    ap.add_argument("--loop-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--non-strict", action="store_true")
    ap.add_argument("--dry", action="store_true", help="run the cases eagerly, no capture")
    a = ap.parse_args(argv)
    export_dir, out = Path(a.export_dir), Path(a.out)
    _, policy_json, manifest = load_quantized.read_export(export_dir)
    policy, cfg = load_dit(export_dir, a.device)
    full_cases, _ = fixture_cases(a.single_step_bundle, a.device)
    boundary = (DiTEncode if a.phase == "encode" else DiTStep)(policy).eval()
    cases = phase_cases(a.phase, policy, full_cases)
    if a.dry:
        with torch.no_grad():
            for c in cases:
                o = boundary(*c)
                print("dry ok", a.phase, [tuple(x.shape) for x in c], "->", tuple(o.shape), o.dtype, flush=True)
        return 0
    lm_path = Path(a.loop_manifest)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loop_manifest.json").write_bytes(lm_path.read_bytes())
    link = out / "export"
    if not link.exists():
        os.symlink(export_dir.resolve(), link)
    here = Path(__file__).resolve()
    lm = json.loads(lm_path.read_text())
    import dataclasses
    report = fx_export.capture_fx(
        boundary, cases, out / "fx", policy=policy_json, strict=not a.non_strict, native_fp8="uint8",
        provenance={"export_dir": str(export_dir.resolve()),
                    "export_manifest_sha256": quant_export.sha256_file(export_dir / "manifest.json"),
                    "export_provenance": manifest.get("provenance", {}), "export_recipe": manifest.get("recipe", {}),
                    "architecture_locator": (manifest.get("model") or {}).get("id"),
                    "boundary": f"dit_{a.phase}", "phase": a.phase, "boundary_driver": str(here),
                    "boundary_driver_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
                    "boundary_note": "per-phase boundary; the host owns the denoising loop (loop_manifest)",
                    "loop_manifest": {"file": "../loop_manifest.json", "copy_in_bundle_dir": "loop_manifest.json",
                                      "sha256": hashlib.sha256(lm_path.read_bytes()).hexdigest(),
                                      "denoising_steps": lm["denoising_steps"],
                                      "calls_per_action_chunk": lm["graphs"][a.phase]["calls_per_action_chunk"]},
                    "boundary_inputs": list(lm["graphs"][a.phase]["inputs"]),
                    "capture_cases_from": str(Path(a.single_step_bundle) / "fx" / "fixtures.safetensors"),
                    "config_sha256": hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
                                                    ).hexdigest(),
                    "capture_source": fx_export._capture_source(), "seed": 0})
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
