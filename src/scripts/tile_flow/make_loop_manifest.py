"""Write loop_manifest.json: what the host has to do around the per-phase FX graphs of the Multi-Task DiT and of Pi0.
Every number is read from a config, a source file, a scheduler object or a fixture of an existing bundle; none is typed in
(docs/algos/phase-graphs.md). The manifest says how many denoising steps the real model takes, the scheduler / Euler constants,
which inputs change per call, which operations stay on the host and how often each graph is called.

usage (run with PYTHONPATH=<worktree>/src):
  make_loop_manifest.py dit --single-step-bundle DIR --out PHASES_DIR/loop_manifest.json      # atlas-lerobot env
  make_loop_manifest.py pi0 --single-step-bundle DIR --checkpoint-dir CKPT --openpi-dir OPENPI --out PATH   # atlas-pi0 env
`--single-step-bundle` is the existing one-step bundle (read only): its bundle_manifest.json names the checkpoint snapshot, its
fx/fixtures.safetensors gives the observation tensor shapes, dtypes and the camera masks of the capture cases."""
import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

SCHEMA = "atlas_torchao.loop_manifest.v1"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _fixture_specs(fx_dir, case=0, with_values=()):
    """{input_j: {shape, dtype}} of the capture case (and the values of the named inputs) from fx/fixtures.safetensors."""
    from safetensors import safe_open

    out, values = {}, {}
    with safe_open(str(Path(fx_dir) / "fixtures.safetensors"), "pt") as f:
        for key in sorted(f.keys()):
            if key.startswith(f"case_{case}."):
                s = f.get_slice(key)
                out[key.split(".", 1)[1]] = {"shape": list(s.get_shape()), "dtype": s.get_dtype()}
        for key in with_values:
            values[key] = f.get_tensor(f"case_{case}.{key}")
    return out, values


def _line_range(path, needle, span=1):
    """'file:first-last' of the first line containing `needle` (the citation for a number)."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines, 1):
        if needle in line:
            return f"{path}:{i}" if span == 1 else f"{path}:{i}-{i + span - 1}"
    raise LookupError(f"{needle!r} not found in {path}")


# ------------------------------------------------------------------------------------------------------------------
# Multi-Task DiT
# ------------------------------------------------------------------------------------------------------------------

def ddpm_schedule(scheduler):
    """Per-call constants of DDPMScheduler.step for the timesteps set on `scheduler`, taken from its own tensors with the
    arithmetic of diffusers/schedulers/scheduling_ddpm.py step(): float32 values, as the scheduler computes them."""
    rows = []
    for t in [int(v) for v in scheduler.timesteps]:
        prev = int(scheduler.previous_timestep(t))
        a_t = scheduler.alphas_cumprod[t]
        a_prev = scheduler.alphas_cumprod[prev] if prev >= 0 else scheduler.one
        b_t, b_prev = 1 - a_t, 1 - a_prev
        cur_alpha = a_t / a_prev
        cur_beta = 1 - cur_alpha
        rows.append({
            "t": t, "previous_t": prev,
            "alpha_prod_t": float(a_t), "alpha_prod_t_prev": float(a_prev),
            "sqrt_beta_prod_t": float(b_t ** 0.5), "sqrt_alpha_prod_t": float(a_t ** 0.5),
            "pred_original_sample_coeff": float((a_prev ** 0.5 * cur_beta) / b_t),
            "current_sample_coeff": float(cur_alpha ** 0.5 * b_prev / b_t),
            "variance_noise_draw": t > 0,
            "variance_std": float(scheduler._get_variance(t) ** 0.5) if t > 0 else 0.0})
    return rows


def dit_manifest(single_step_bundle):
    from types import SimpleNamespace

    import diffusers
    import torch
    import lerobot.policies.multi_task_dit.modeling_multi_task_dit as mt

    bundle = Path(single_step_bundle)
    bm = json.loads((bundle / "bundle_manifest.json").read_text())
    snap = Path(bm["model"]["checkpoint"]["snapshot"])
    cfg_path = snap / "config.json"
    if _sha(cfg_path) != bm["model"]["checkpoint"]["config_json_sha256"]:
        raise SystemExit(f"{cfg_path} differs from the bundle's recorded config_json_sha256")
    cfg = json.loads(cfg_path.read_text())
    action_dim = cfg["output_features"]["action"]["shape"][0]
    # the policy's own objective builds the scheduler and the step count (modeling_multi_task_dit.py DiffusionObjective)
    if cfg["objective"] != "diffusion":
        raise SystemExit(f"objective {cfg['objective']!r}: this manifest covers the diffusion objective")
    objective = mt.DiffusionObjective(SimpleNamespace(**cfg), action_dim, cfg["horizon"])
    sched = objective.noise_scheduler
    n = int(objective.num_inference_steps)
    sched.set_timesteps(n)
    table = ddpm_schedule(sched)
    src = Path(mt.__file__)
    inputs, _ = _fixture_specs(bundle / "fx")
    names = ["state", "images", "input_ids", "attention_mask", "noisy_actions", "timestep"]
    spec = {names[i]: inputs[f"input_{i}"] for i in range(len(names))}
    out_spec = inputs["output_0"]
    # conditioning vector: ObservationEncoder._setup_vector_output = (cams * vision_dim + state_dim + text_dim) * n_obs_steps
    from transformers import CLIPVisionConfig

    vision_dim = CLIPVisionConfig.from_pretrained(cfg["vision_encoder_name"]).hidden_size
    cams = sum(1 for f in cfg["input_features"].values() if f["type"] == "VISUAL")
    state_dim = cfg["input_features"]["observation.state"]["shape"][0]
    cond_dim = (cams * vision_dim + state_dim + cfg["hidden_dim"]) * cfg["n_obs_steps"]
    cond = {"shape": [spec["state"]["shape"][0], cond_dim], "dtype": spec["state"]["dtype"],
            "derivation": f"({cams} cameras x {vision_dim} CLIP CLS + {state_dim} state + {cfg['hidden_dim']} text) x "
                          f"{cfg['n_obs_steps']} obs steps = {cond_dim}; ObservationEncoder._setup_vector_output",
            "check": "equals the encode graph output shape (verified at equivalence time, see phase-graphs.md)"}
    start, end = cfg["n_obs_steps"] - 1, cfg["n_obs_steps"] - 1 + cfg["n_action_steps"]
    return {
        "schema": SCHEMA, "model": "multi_task_dit", "objective": "diffusion (DDPM)",
        "sources": {
            "policy_config": str(cfg_path), "policy_config_sha256": _sha(cfg_path),
            "step_count": f"num_inference_steps={cfg['num_inference_steps']!r} in config.json, so DiffusionObjective falls back to "
                          f"noise_scheduler.config.num_train_timesteps ({cfg['num_train_timesteps']}): "
                          + _line_range(src, "self.num_inference_steps = (", 5),
            "loop": _line_range(src, "def conditional_sample", 20),
            "chunk_slice": _line_range(src, "start = n_obs_steps - 1", 3),
            "diffusers_version": diffusers.__version__, "diffusers_scheduler": "DDPMScheduler.step (scheduling_ddpm.py)",
            "lerobot_file": str(src), "torch": torch.__version__,
            "single_step_bundle": str(bundle), "capture_input_shapes": "that bundle's fx/fixtures.safetensors case_0"},
        "denoising_steps": n,
        "scheduler": {
            "class": type(sched).__name__,
            "config": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                       for k, v in dict(sched.config).items() if not k.startswith("_")},
            "timesteps": [r["t"] for r in table],
            "per_step": table,
            "step_arithmetic": [
                "prev_t = previous_timestep(t); a_t = alphas_cumprod[t]; a_prev = alphas_cumprod[prev_t] (1 if prev_t < 0)",
                "x0 = (x_t - sqrt_beta_prod_t * eps) / sqrt_alpha_prod_t                       (prediction_type epsilon)",
                f"x0 = clamp(x0, -{cfg['clip_sample_range']}, {cfg['clip_sample_range']})            (clip_sample {cfg['clip_sample']})",
                "x_prev = pred_original_sample_coeff * x0 + current_sample_coeff * x_t",
                "if t > 0: x_prev += variance_std * z, z ~ N(0, I) drawn on the host with the shape and dtype of eps"],
            "variance_noise_draws": sum(1 for r in table if r["variance_noise_draw"]),
            "initial_noise": {"draws": 1, "shape": [1, cfg["horizon"], action_dim], "dtype": spec["noisy_actions"]["dtype"],
                              "distribution": "standard normal (torch.randn in conditional_sample)"}},
        "graphs": {
            "encode": {"path": "encode/fx", "calls_per_action_chunk": 1,
                       "boundary": "observation_encoder.encode(batch): resize + center crop, CLIP vision and text towers, "
                                   "projection, concatenation; no timestep, no noise",
                       "inputs": {k: spec[k] for k in names[:4]}, "outputs": {"conditioning_vec": cond}},
            "step": {"path": "step/fx", "calls_per_action_chunk": n,
                     "boundary": "one noise_predictor(noisy_actions, timestep, conditioning_vec=cond) call",
                     "inputs": {"conditioning_vec": cond, "noisy_actions": spec["noisy_actions"],
                                "timestep": spec["timestep"]},
                     "outputs": {"predicted_noise": out_spec}}},
        "per_call_inputs": {
            "encode": "once per action chunk: state / images (n_obs_steps = %d frames each) and tokenized prompt" % cfg["n_obs_steps"],
            "step": {"changes_every_call": ["noisy_actions (previous scheduler output)", "timestep (timesteps[i], int64 shape [1])"],
                     "constant_over_the_loop": ["conditioning_vec (encode output)"]}},
        "loop": ["cond = encode(state, images, input_ids, attention_mask)       # 1 call",
                 "x = randn(1, horizon, action_dim)                              # host",
                 f"for t in scheduler.timesteps:                                 # {n} iterations, host",
                 "    eps = step(cond, x, full([1], t, int64))                   # {n} calls".replace("{n}", str(n)),
                 "    x = scheduler_step(eps, x, t)                              # host, includes clip and variance noise",
                 f"actions = x[:, {start}:{end}]                                   # host (n_obs_steps - 1 : + n_action_steps)"],
        "host_operations": [
            {"op": "observation queues", "detail": f"n_obs_steps={cfg['n_obs_steps']} frames of state and images stacked by populate_queues"},
            {"op": "tokenization", "detail": f"CLIP tokenizer, max length {cfg['tokenizer_max_length']}, padding {cfg['tokenizer_padding']}"},
            {"op": "input normalization", "detail": f"mapping {cfg['normalization_mapping']} (policy_preprocessor.json)"},
            {"op": "initial noise draw", "detail": "randn, 1 draw per action chunk"},
            {"op": "timestep tensor", "detail": "torch.full([batch], t, dtype=int64) from scheduler.timesteps"},
            {"op": "scheduler arithmetic", "detail": "DDPM step with the per_step table, epsilon prediction, x0 clip to "
                                                     f"[-{cfg['clip_sample_range']}, {cfg['clip_sample_range']}]"},
            {"op": "variance noise draws", "detail": f"{sum(1 for r in table if r['variance_noise_draw'])} draws (every step with t > 0)"},
            {"op": "action chunk slice", "detail": f"actions[:, {start}:{end}] of horizon {cfg['horizon']}"},
            {"op": "output unnormalization", "detail": "policy_postprocessor.json unnormalizer (MIN_MAX)"}],
        "not_provided": ["unrolled 100-step graph (not sensible: 99 variance-noise inputs and host scheduler arithmetic)",
                         "random-crop augmentation (training only; eval uses center crop inside the encode graph)"]}


# ------------------------------------------------------------------------------------------------------------------
# Pi0
# ------------------------------------------------------------------------------------------------------------------

def sample_actions_default_steps(pi0_pytorch_py):
    """num_steps default of PI0Pytorch.sample_actions, read from the source with ast (no openpi import needed)."""
    tree = ast.parse(Path(pi0_pytorch_py).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "sample_actions":
            names = [a.arg for a in node.args.args]
            defaults = dict(zip(names[len(names) - len(node.args.defaults):], [ast.literal_eval(d) for d in node.args.defaults]))
            return defaults["num_steps"], node.lineno
    raise LookupError("sample_actions not found")


def euler_times(num_steps):
    """The float32 times of PI0Pytorch.sample_actions: dt = float32(-1/num_steps); time = float32(1); while time >= -dt/2: step; time += dt
    (pi0_pytorch.py sample_actions, same tensor dtypes and the same recurrence)."""
    import torch

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32)
    time = torch.tensor(1.0, dtype=torch.float32)
    times = []
    while time >= -dt / 2:
        times.append(float(time))
        time += dt
    return float(dt), times


def pi0_manifest(single_step_bundle, checkpoint_dir, openpi_dir, stack_cameras=False):
    import torch

    bundle, openpi = Path(single_step_bundle), Path(openpi_dir)
    cfg_path = Path(checkpoint_dir) / "config.json"
    ckpt = json.loads(cfg_path.read_text())
    pi0_py = openpi / "src/openpi/models_pytorch/pi0_pytorch.py"
    default_steps, def_line = sample_actions_default_steps(pi0_py)
    shim = Path(__file__).resolve().parents[2] / "src" / "pi0_model.py"
    shim_steps = {int(m) for m in re.findall(r"num_steps\s*=\s*(\d+)", shim.read_text())} | \
                 {int(m) for m in re.findall(r"predict\([^)]*,\s*(\d+)\)", (bundle / "pi0_fx.py").read_text())}
    if shim_steps != {default_steps}:
        raise SystemExit(f"step counts disagree: sample_actions default {default_steps}, shim / driver {sorted(shim_steps)}")
    dt, times = euler_times(default_steps)
    sys.path.insert(0, str(openpi / "src"))
    import openpi.models.gemma as gemma
    from transformers.models.auto import CONFIG_MAPPING

    vlm, expert = gemma.get_config(ckpt["paligemma_variant"]), gemma.get_config(ckpt["action_expert_variant"])
    vision = CONFIG_MAPPING["paligemma"]().vision_config
    tokens_per_image = (vision.image_size // vision.patch_size) ** 2
    inputs, values = _fixture_specs(bundle / "fx", with_values=("input_3", "input_4", "input_5"))
    keys_assign = [n for n in ast.walk(ast.parse((bundle / "pi0_fx.py").read_text()))
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "KEYS"]
    cameras = list(ast.literal_eval(keys_assign[0].value))
    prompt_len = inputs["input_6"]["shape"][1]
    image_shape = inputs["input_0"]["shape"]
    if image_shape[2] // vision.patch_size != image_shape[2] / vision.patch_size or image_shape[2] != vision.image_size:
        raise SystemExit(f"capture image size {image_shape} differs from the SigLIP config {vision.image_size}")
    prefix_len = len(cameras) * tokens_per_image + prompt_len
    if vlm.depth != expert.depth or vlm.num_kv_heads != expert.num_kv_heads or vlm.head_dim != expert.head_dim:
        raise SystemExit("PaliGemma and action expert KV geometry differ; the expert would not read the prefix cache")
    kv_dtype = ckpt["precision"]
    kv = {"layers": vlm.depth, "tensors": 2 * vlm.depth, "per_tensor_shape": [1, vlm.num_kv_heads, prefix_len, vlm.head_dim],
          "dtype": kv_dtype,
          "derivation": f"gemma {ckpt['paligemma_variant']}: depth {vlm.depth}, kv heads {vlm.num_kv_heads}, head_dim {vlm.head_dim} "
                        f"(openpi gemma.get_config); prefix = {len(cameras)} cameras x {tokens_per_image} SigLIP tokens "
                        f"((image {vision.image_size} / patch {vision.patch_size})^2) + {prompt_len} prompt tokens",
          "check": "tensor shapes of the prefix graph outputs (verified at equivalence time)"}
    horizon, adim = ckpt["action_horizon"], ckpt["action_dim"]
    state, noise = inputs["input_8"], inputs["input_9"]
    return {
        "schema": SCHEMA, "model": "pi0", "objective": "flow matching, Euler",
        "sources": {"checkpoint_config": str(cfg_path), "checkpoint_config_sha256": _sha(cfg_path),
                    "step_count": f"num_steps default {default_steps}: {pi0_py}:{def_line} (def sample_actions); the reference shim passes "
                                  f"the same value: {_line_range(shim, 'self.num_steps =')}, "
                                  f"{_line_range(bundle / 'pi0_fx.py', 'pi0_model.predict(')}",
                    "loop": f"{pi0_py}: sample_actions (dt = -1/num_steps, float32 tensors, while time >= -dt/2)",
                    "kv_geometry": "openpi/src/openpi/models/gemma.py get_config", "torch": torch.__version__,
                    "single_step_bundle": str(bundle), "capture_input_shapes": "that bundle's fx/fixtures.safetensors case_0"},
        "denoising_steps": default_steps,
        "euler": {"dt": dt, "dt_dtype": "float32 tensor", "start_time": 1.0,
                  "loop_condition": "time >= -dt / 2 (float32)", "times": times, "times_dtype": "float32",
                  "update": "x_t = x_t + dt * v_t ; time += dt (float32)", "count": len(times)},
        "model": {"action_dim": adim, "action_horizon": horizon, "paligemma_variant": ckpt["paligemma_variant"],
                  "action_expert_variant": ckpt["action_expert_variant"], "precision": ckpt["precision"]},
        "cameras": {"order": cameras, "tokens_per_image": tokens_per_image,
                    "siglip_calls_per_prefix_call": 1 if stack_cameras else len(cameras),
                    "camera_handling": ("stacked on the batch axis: one SigLIP call for all cameras (pi0_fx._embed_prefix_one_tower_call)"
                                        if stack_cameras else "openpi embed_prefix unmodified: one SigLIP call per camera, so each "
                                        "SigLIP quantized module is called %d times in the prefix graph" % len(cameras)),
                    "image_masks_in_capture_case_0": {c: bool(values[f"input_{3 + i}"].item()) for i, c in enumerate(cameras)}},
        "graphs": {
            "prefix": {"path": "prefix/fx", "calls_per_action_chunk": 1,
                       "boundary": "embed_prefix (SigLIP, embedding) + PaliGemma language model with use_cache=True",
                       "inputs": {**{f"image_{c}": inputs[f"input_{i}"] for i, c in enumerate(cameras)},
                                  **{f"image_mask_{c}": inputs[f"input_{3 + i}"] for i, c in enumerate(cameras)},
                                  "tokenized_prompt": inputs["input_6"], "tokenized_prompt_mask": inputs["input_7"]},
                       "outputs": {"prefix_pad_masks": {"shape": [1, prefix_len], "dtype": "BOOL"},
                                   "kv_cache": kv}},
            "step": {"path": "step/fx", "calls_per_action_chunk": default_steps,
                     "boundary": "denoise_step: embed_suffix + action expert reading the prefix K/V + action_out_proj",
                     "inputs": {"state": state, "prefix_pad_masks": {"shape": [1, prefix_len], "dtype": "BOOL"},
                                "x_t": noise, "time": {"shape": [1], "dtype": "F32"}, "kv_cache": kv},
                     "outputs": {"v_t": noise}}},
        "per_call_inputs": {
            "prefix": "once per action chunk: 3 camera images, camera masks, tokenized prompt",
            "step": {"changes_every_call": ["x_t (previous Euler output)", "time (euler.times[i], float32, expanded to batch)"],
                     "constant_over_the_loop": ["state", "prefix_pad_masks", f"kv_cache ({kv['tensors']} tensors, prefix pass output; "
                                                "the expert never writes it: use_cache=False in denoise_step)"]}},
        "loop": ["kv, prefix_pad_masks = prefix(images, masks, tokens)         # 1 call",
                 "x = normal(1, horizon, action_dim) float32                  # host",
                 "time = float32(1.0); dt = float32(-1/num_steps)             # host",
                 f"while time >= -dt/2:                                       # {len(times)} iterations, host",
                 "    v = step(state, prefix_pad_masks, kv, x, time.expand(1))  # step calls",
                 "    x = x + dt * v ; time += dt                              # host, float32"],
        "host_operations": [
            {"op": "image preprocessing", "detail": "client resize_with_pad to 224x224 and _preprocess_observation (eval: identity at 224x224)"},
            {"op": "tokenization and state normalization", "detail": "reference policy shim (Pi0PyTorchPolicy.infer)"},
            {"op": "noise draw", "detail": f"torch.normal float32 [1, {horizon}, {adim}], 1 draw per chunk (PI0Pytorch.sample_noise)"},
            {"op": "time recurrence and loop condition", "detail": "the euler.times list; float32"},
            {"op": "Euler update", "detail": "x_t + dt * v_t on the host (the step graph returns v_t)"},
            {"op": "output postprocessing", "detail": "unnormalize, joint deltas to absolute, first 8 dims (reference policy shim)"}],
        "not_provided": ["unrolled 10-step graph (plan item 6)"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit", "pi0"))
    ap.add_argument("--single-step-bundle", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--openpi-dir")
    ap.add_argument("--stack-cameras", action="store_true", help="pi0: record the one-tower-call deviation instead of one SigLIP call per camera")
    a = ap.parse_args(argv)
    if a.model == "dit":
        manifest = dit_manifest(a.single_step_bundle)
    else:
        if not a.checkpoint_dir or not a.openpi_dir:
            ap.error("pi0 needs --checkpoint-dir and --openpi-dir")
        manifest = pi0_manifest(a.single_step_bundle, a.checkpoint_dir, a.openpi_dir, a.stack_cameras)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1, allow_nan=False) + "\n")
    print(json.dumps({"out": str(out), "sha256": _sha(out), "denoising_steps": manifest["denoising_steps"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
