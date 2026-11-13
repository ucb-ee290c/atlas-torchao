"""Per-phase FX capture for Pi0 (openpi PyTorch): the prefix (SigLIP + PaliGemma prefix producing the KV cache) and one action-expert
denoising step reading that cache, as two graphs with the host owning the Euler loop (docs/algos/phase-graphs.md).

  prefix: Pi0Prefix (img x3, image_mask x3, tokenized_prompt, tokenized_prompt_mask) -> (prefix_pad_masks, k_0, v_0, ..., k_17, v_17)
          = _preprocess_observation + embed_prefix + paligemma_with_expert.forward(inputs_embeds=[prefix, None], use_cache=True)
          Cameras are NOT stacked by default: openpi's embed_prefix runs the SigLIP tower once per camera, so every SigLIP module is
          called 3 times (fx_export counts the calls and expects 3 nodes per site). --stack-cameras reproduces the earlier one-tower
          deviation (pi0_fx._embed_prefix_one_tower_call).
  step:   Pi0Step (state, prefix_pad_masks, x_t, time, k_0, v_0, ..., k_17, v_17) -> v_t
          = denoise_step with the K/V rebuilt into a DynamicCache (read only: use_cache=False in denoise_step)
The existing export is reused (read only); capture cases are the existing single-step bundle's fixture inputs.
usage (PYTHONPATH=<worktree>/src, atlas-pi0 env):
  pi0_phase_capture.py prefix|step --export-dir EXPORT --single-step-bundle BUNDLE --loop-manifest LM --out DIR
        --openpi-dir OPENPI --checkpoint-dir CKPT [--stack-cameras] [--dry]"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _split_kv(flat):
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


class Pi0Prefix(torch.nn.Module):
    def __init__(self, model, stack_cameras=False):
        super().__init__()
        self.model = model
        self.stack_cameras = stack_cameras

    def forward(self, img0, img1, img2, m0, m1, m2, tokens, token_mask):
        from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
        model = self.model
        obs = SimpleNamespace(images=dict(zip(KEYS, (img0, img1, img2))), image_masks=dict(zip(KEYS, (m0, m1, m2))),
                              state=torch.zeros(img0.shape[0], model.config.action_dim, device=img0.device),
                              tokenized_prompt=tokens, tokenized_prompt_mask=token_mask,
                              token_ar_mask=torch.zeros_like(token_mask), token_loss_mask=torch.zeros_like(token_mask))
        images, img_masks, lang_tokens, lang_masks, _ = model._preprocess_observation(obs, train=False)
        if self.stack_cameras:
            from pi0_fx import _embed_prefix_one_tower_call
            embs, pad, att = _embed_prefix_one_tower_call(model, images, img_masks, lang_tokens, lang_masks)
        else:
            embs, pad, att = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        att_2d = make_att_2d_masks(pad, att)
        position_ids = torch.cumsum(pad, dim=1) - 1
        att_4d = model._prepare_attention_masks_4d(att_2d)
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001
        _, cache = model.paligemma_with_expert.forward(attention_mask=att_4d, position_ids=position_ids, past_key_values=None,
                                                       inputs_embeds=[embs, None], use_cache=True)
        flat = []
        for layer in range(len(cache)):
            k, v = cache[layer]
            flat += [k, v]
        return (pad, *flat)


class Pi0Step(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state, prefix_pad_masks, x_t, time, *kv):
        from transformers import DynamicCache
        cache = DynamicCache()
        for layer, (k, v) in enumerate(_split_kv(kv)):
            cache.update(k, v, layer)
        return self.model.denoise_step(state, prefix_pad_masks, cache, x_t, time)


def load_pi0(export_dir, openpi_dir, checkpoint_dir, tokenizer_path, device="cpu"):
    import eval_pi0
    import fx_export
    import load_quantized
    import pi0_model

    # same driver-level workaround as the single-step bundle: SigLIP's patch embedding exports as aten.conv2d.padding
    fx_export.CONTRACTIONS[torch.ops.aten.conv2d.padding] = (0, 1)
    manifest = json.loads((Path(export_dir) / "manifest.json").read_text())
    args = eval_pi0.build_parser().parse_args(
        ["--obs", "droid", "--n-obs", "2", "--seed", "0", "--modes", "ref_conv", "--algo", manifest["recipe"]["algo"],
         "--device", device, "--checkpoint-dir", checkpoint_dir, "--openpi-dir", openpi_dir, "--tokenizer-path", tokenizer_path,
         "--export-dir", "unused", "--export-only", "--out", "unused"])
    rt = pi0_model.runtime(pi0_model.openpi_dir(openpi_dir))
    cfg = pi0_model.model_config(rt, args.config)
    model = pi0_model.load_model(rt, args.config, checkpoint_dir, device)
    load_quantized.load_quantized(None, export_dir, device, build=lambda: model, mode="ref_conv", options=args)
    return model, cfg


def fixture_cases(single_step_bundle, device="cpu"):
    """[(img0, img1, img2, m0, m1, m2, tokens, token_mask, state, noise), ...] and the recorded one-step outputs."""
    from safetensors.torch import load_file
    fx = Path(single_step_bundle) / "fx"
    tensors = load_file(str(fx / "fixtures.safetensors"), device=device)
    parity = json.loads((fx / "parity.json").read_text())
    return ([tuple(tensors[k] for k in c["inputs"]) for c in parity["cases"]],
            [tuple(tensors[k] for k in c["outputs"]) for c in parity["cases"]])


def phase_cases(phase, model, full_cases, stack_cameras):
    """Capture cases of the phase boundary. Step cases take the eager prefix outputs and time = 1.0, the first Euler time, so
    that step(prefix(obs), state, noise, 1.0) is exactly the one-step bundle's computation (x_1 = noise - v)."""
    if phase == "prefix":
        return [c[:8] for c in full_cases]
    prefix = Pi0Prefix(model, stack_cameras).eval()
    cases = []
    with torch.no_grad():
        for c in full_cases:
            pad, *kv = prefix(*c[:8])
            cases.append((c[8], pad, c[9], torch.ones(c[9].shape[0], dtype=torch.float32), *kv))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("prefix", "step"))
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--single-step-bundle", required=True)
    ap.add_argument("--loop-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--openpi-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--tokenizer-path", default="/scratch2/lorenhung/models/big_vision/paligemma_tokenizer.model")
    ap.add_argument("--stack-cameras", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(a.single_step_bundle)))            # pi0_fx.py (only for --stack-cameras)
    import fx_export
    import load_quantized
    import quant_export

    export_dir, out = Path(a.export_dir), Path(a.out)
    _, policy_json, manifest = load_quantized.read_export(export_dir)
    model, cfg = load_pi0(export_dir, a.openpi_dir, a.checkpoint_dir, a.tokenizer_path, a.device)
    full_cases, _ = fixture_cases(a.single_step_bundle, a.device)
    boundary = (Pi0Prefix(model, a.stack_cameras) if a.phase == "prefix" else Pi0Step(model)).eval()
    cases = phase_cases(a.phase, model, full_cases, a.stack_cameras)
    if a.dry:
        with torch.no_grad():
            for c in cases:
                o = boundary(*c)
                o = o if isinstance(o, tuple) else (o,)
                print("dry ok", a.phase, len(c), "inputs ->", len(o), "outputs", [tuple(x.shape) for x in o[:3]], flush=True)
        return 0
    lm_path = Path(a.loop_manifest)
    out.mkdir(parents=True, exist_ok=True)
    (out / "loop_manifest.json").write_bytes(lm_path.read_bytes())
    if not (out / "export").exists():
        os.symlink(export_dir.resolve(), out / "export")
    here = Path(__file__).resolve()
    lm = json.loads(lm_path.read_text())
    report = fx_export.capture_fx(
        boundary, cases, out / "fx", policy=policy_json, strict=False, native_fp8="uint8",
        provenance={"export_dir": str(export_dir.resolve()),
                    "export_manifest_sha256": quant_export.sha256_file(export_dir / "manifest.json"),
                    "export_provenance": manifest.get("provenance", {}), "export_recipe": manifest.get("recipe", {}),
                    "architecture_locator": None, "boundary": f"pi0_{a.phase}", "phase": a.phase,
                    "boundary_driver": str(here), "boundary_driver_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
                    "boundary_note": "per-phase boundary; the host owns the Euler loop (loop_manifest)",
                    "stack_cameras": a.stack_cameras,
                    "loop_manifest": {"file": "../loop_manifest.json", "sha256": hashlib.sha256(lm_path.read_bytes()).hexdigest(),
                                      "denoising_steps": lm["denoising_steps"],
                                      "calls_per_action_chunk": lm["graphs"][a.phase]["calls_per_action_chunk"]},
                    "boundary_inputs": list(lm["graphs"][a.phase]["inputs"]),
                    "contraction_overload_added_in_driver": "aten.conv2d.padding",
                    "capture_cases_from": str(Path(a.single_step_bundle) / "fx" / "fixtures.safetensors"),
                    "openpi_dir": a.openpi_dir, "checkpoint_dir": a.checkpoint_dir,
                    "config_sha256": hashlib.sha256(json.dumps({k: str(v) for k, v in vars(cfg).items()}, sort_keys=True
                                                               ).encode()).hexdigest() if hasattr(cfg, "__dict__") else None,
                    "capture_source": fx_export._capture_source(), "seed": 0})
    print(json.dumps(report, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
