"""Per-phase FX capture for the Multi-Task DiT (and Pi0, see below): the encoder and one denoising step as two separate
graphs, with the host owning the loop (docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).

  dit encode:  DiTEncode  (state, images, input_ids, attention_mask)           -> conditioning_vec
               = observation_encoder.encode(batch)   (CLIP towers, patch-embedding Conv2d)
  dit step:    DiTStep    (conditioning_vec, noisy_actions, timestep)          -> predicted noise
               = noise_predictor(noisy_actions, timestep, conditioning_vec=cond)   (one call, no scheduler, no RNG)
Hoisted layout (loop manifest with a `modulate` graph, the default of make_loop_manifest.py):
  dit modulate: DiTModulate (conditioning_vec, timesteps [S])                 -> modulation [layers, S, B, 6 * hidden]
               = time_mlp + every block's adaLN_modulation for all S timesteps, steps_per_call timesteps per call
  dit step:    DiTStepHoisted (noisy_actions, modulation [layers, B, 6 * hidden]) -> predicted noise
               = noise_predictor without time_mlp / adaLN_modulation; the host passes modulation[:, i] at step i

The existing export is reused (read only) and the capture cases are the existing single-step bundle's fixture inputs, so the
composition (encode, then step on its output) can be compared byte for byte with that bundle's recorded outputs.
usage (PYTHONPATH=<worktree>/src, atlas-lerobot env):
  policy_fx_capture_phases.py dit encode|modulate|step --export-dir EXPORT --single-step-bundle BUNDLE --loop-manifest LM --out DIR
DIR gets fx/ (the bundle), loop_manifest.json (a copy, same sha256 as LM, recorded in fx_manifest.json provenance) and an
`export` symlink to EXPORT so verify_bundle.py / make_bundle_manifest.py can read it."""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

import exporter.fx_export as fx_export
import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export

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


# ------------------------------------------------------------------------------------------------------------------
# Hoisted layout: the adaLN modulation (time_mlp + every block's SiLU + Linear) depends only on the timestep and on
# conditioning_vec, never on the noisy actions, so all steps' modulations are computed once per action chunk
# (`modulate`) and the step graph reads its step's rows (docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).
# ------------------------------------------------------------------------------------------------------------------

def hoisted_paths(noise_predictor, prefix="noise_predictor."):
    """Module paths of the Linear layers the hoist moves from `step` to `modulate`: time_mlp's Linears and every
    block's adaLN_modulation Linear (nn.Linear or its quantized replacement: both have in_features)."""
    return sorted(prefix + name for name, m in noise_predictor.named_modules()
                  if (name.startswith("time_mlp.") or ".adaLN_modulation." in name) and hasattr(m, "in_features"))


def _row_spanning(rule):
    """True when an activation rule's exponent can depend on more than one row of the reduced input (dynamic per
    tensor or per block of rows, or the reference per-tensor rule); static and per-token rules are per row."""
    if not rule:
        return False
    if rule.get("source") == "reference":
        return True
    if rule.get("mode") == "static":
        return False
    if rule.get("mode") != "dynamic":
        return True                                                  # unknown: fail closed
    return rule.get("reduce") != "token"


def check_hoist_rules(policy_json, paths, steps_per_call):
    """Refuse a modulate graph that batches several steps into one call of a site whose activation exponent spans rows:
    one step's input then shares a scale group with other steps' rows, which changes its codes (per 32-row block under
    a=block). With one step per call the grouping is the per-step one. Returns the row-spanning sites."""
    bad = [p for p in paths if _row_spanning((policy_json["sites"].get(p) or {}).get("activation"))]
    if bad and steps_per_call != 1:
        raise ValueError(f"{len(bad)} hoisted site(s) have a dynamic activation scale spanning rows "
                         f"(e.g. {bad[0]}: {policy_json['sites'][bad[0]].get('activation')}); {steps_per_call} steps per "
                         "call would put several steps in one scale group and change the result. Use steps_per_call=1")
    return bad


def modulation_rows(noise_predictor, conditioning_vec, timesteps):
    """[num_layers, n * B, 6 * hidden] adaLN outputs for `timesteps` (int64 [n]) and conditioning_vec [B, C], rows
    step-major (row i * B + b = step i, batch b). The arithmetic of DiffusionTransformer.forward and TransformerBlock.forward
    up to the adaLN call: time_mlp(timestep), cat([time features, conditioning_vec], -1), adaLN_modulation."""
    n, batch = timesteps.shape[0], conditioning_vec.shape[0]
    t_rows = timesteps.repeat_interleave(batch)
    cond_rows = conditioning_vec.repeat(n, 1)
    cond_features = torch.cat([noise_predictor.time_mlp(t_rows), cond_rows], dim=-1)
    return torch.stack([block.adaLN_modulation(cond_features) for block in noise_predictor.transformer_blocks])


class DiTModulate(torch.nn.Module):
    """(conditioning_vec [B, C], timesteps int64 [S]) -> modulation [num_layers, S, B, 6 * hidden]: every step's adaLN
    output, computed in calls of `steps_per_call` steps (each hoisted Linear is called ceil(S / steps_per_call) times with
    steps_per_call * B rows; steps_per_call = S is one call per site)."""

    def __init__(self, policy, steps_per_call):
        super().__init__()
        self.model = policy
        self.steps_per_call = int(steps_per_call)
        if self.steps_per_call < 1:
            raise ValueError("steps_per_call must be >= 1")

    def forward(self, conditioning_vec, timesteps):
        npred, k = self.model.noise_predictor, self.steps_per_call
        parts = [modulation_rows(npred, conditioning_vec, timesteps[i:i + k]) for i in range(0, timesteps.shape[0], k)]
        rows = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return rows.reshape(rows.shape[0], timesteps.shape[0], conditioning_vec.shape[0], rows.shape[-1])


def block_with_modulation(block, x, modulation):
    """TransformerBlock.forward (lerobot multi_task_dit) with the adaLN output given instead of computed."""
    from lerobot.policies.multi_task_dit.modeling_multi_task_dit import modulate

    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=1)
    attn_input = modulate(block.norm1(x), shift_msa.unsqueeze(1), scale_msa.unsqueeze(1))
    if block.use_rope:
        attn_out = block.attn(attn_input)
    else:
        attn_out, _ = block.multihead_attn(attn_input, attn_input, attn_input)
    x = x + gate_msa.unsqueeze(1) * attn_out
    mlp_input = modulate(block.norm2(x), shift_mlp.unsqueeze(1), scale_mlp.unsqueeze(1))
    mlp_out = block.mlp(mlp_input)
    return x + gate_mlp.unsqueeze(1) * mlp_out


class DiTStepHoisted(torch.nn.Module):
    """(noisy_actions [B, horizon, action_dim], modulation [num_layers, B, 6 * hidden]) -> predicted noise:
    DiffusionTransformer.forward without time_mlp and adaLN_modulation (their output is the `modulation` input)."""

    def __init__(self, policy):
        super().__init__()
        self.model = policy

    def forward(self, noisy_actions, modulation):
        npred = self.model.noise_predictor
        _, seq_len, _ = noisy_actions.shape
        hidden_seq = npred.input_proj(noisy_actions)
        if npred.pos_embedding is not None:
            hidden_seq = hidden_seq + npred.pos_embedding[:, :seq_len, :]
        for i, block in enumerate(npred.transformer_blocks):
            hidden_seq = block_with_modulation(block, hidden_seq, modulation[i])
        return npred.output_proj(hidden_seq)


def hoisted_host_loop(step, modulation, x, rows, draw_noise, ddpm_step):
    """The hoisted loop body shared by the eager check and the composer: step i reads modulation[:, i] (a copy, so the
    graph input is contiguous); `rows` are the manifest's per_step rows; draw_noise(eps) -> z or None; ddpm_step(row, x,
    eps, z) -> x. Returns (x, step calls)."""
    calls = 0
    for i, row in enumerate(rows):
        eps = step(x, modulation[:, i].contiguous())
        calls += 1
        x = ddpm_step(row, x, eps, draw_noise(row, eps))
    return x, calls


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


def manifest_timesteps(lm):
    """The scheduler timesteps of a loop manifest as the int64 tensor the modulate graph takes."""
    return torch.tensor(lm["scheduler"]["timesteps"], dtype=torch.long)


def phase_cases(phase, policy, full_cases, lm=None):
    """The phase boundary's capture cases from the full-boundary cases (encode first for the later phases). Hoisted
    layout (the manifest has a modulate graph): modulate takes (conditioning_vec, timesteps); step takes (noisy_actions,
    modulation rows of the case's timestep), so step(modulate(encode(obs))) is the single-step bundle's computation."""
    if phase == "encode":
        return [c[:4] for c in full_cases]
    with torch.no_grad():
        conds = [DiTEncode(policy).eval()(*c[:4]) for c in full_cases]
        if lm is None or "modulate" not in lm["graphs"]:
            return [(cond, c[4], c[5]) for cond, c in zip(conds, full_cases)]
        ts = manifest_timesteps(lm)
        if phase == "modulate":
            return [(cond, ts.clone()) for cond in conds]
        modulate = DiTModulate(policy, 1).eval()
        cases = []
        for cond, c in zip(conds, full_cases):
            t = c[5].reshape(-1)
            if not bool((t == t[0]).all()):
                raise ValueError("a capture case's batch holds different timesteps; the hoisted step takes one timestep")
            cases.append((c[4], modulate(cond, t[:1])[:, 0].contiguous()))
        return cases


def boundary_for(phase, policy, lm):
    if phase == "encode":
        return DiTEncode(policy)
    if "modulate" not in lm["graphs"]:
        if phase == "modulate":
            raise SystemExit("the loop manifest has no modulate graph (two_phase layout)")
        return DiTStep(policy)
    if phase == "modulate":
        return DiTModulate(policy, lm["graphs"]["modulate"]["steps_per_call"])
    return DiTStepHoisted(policy)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit",))
    ap.add_argument("phase", choices=("encode", "modulate", "step"))
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
    lm_path = Path(a.loop_manifest)
    lm = json.loads(lm_path.read_text())
    boundary = boundary_for(a.phase, policy, lm).eval()
    if a.phase == "modulate":
        check_hoist_rules(policy_json, lm["graphs"]["modulate"]["hoisted_sites"], lm["graphs"]["modulate"]["steps_per_call"])
    cases = phase_cases(a.phase, policy, full_cases, lm)
    if a.dry:
        with torch.no_grad():
            for c in cases:
                o = boundary(*c)
                print("dry ok", a.phase, [tuple(x.shape) for x in c], "->", tuple(o.shape), o.dtype, flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "loop_manifest.json").write_bytes(lm_path.read_bytes())
    link = out / "export"
    if not link.exists():
        os.symlink(export_dir.resolve(), link)
    here = Path(__file__).resolve()
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
                    "layout": lm.get("layout", "two_phase"),
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
