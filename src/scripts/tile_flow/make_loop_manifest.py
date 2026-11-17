"""Write loop_manifest.json: what the host has to do around the per-phase FX graphs of the Multi-Task DiT, of Pi0 and of
SmolVLA.
Every number is read from a config, a source file, a scheduler object or a fixture of an existing bundle; none is typed in
(docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest). The manifest says how many denoising steps the real model takes, the scheduler / Euler constants,
which inputs change per call, which operations stay on the host and how often each graph is called.

usage (run with PYTHONPATH=<worktree>/src):
  make_loop_manifest.py dit --single-step-bundle DIR --out PHASES_DIR/loop_manifest.json      # atlas-lerobot env
  make_loop_manifest.py pi0 --single-step-bundle DIR --checkpoint-dir CKPT --openpi-dir OPENPI --out PATH   # atlas-pi0 env
  make_loop_manifest.py smolvla --policy lerobot/smolvla_libero [--n-images 2] [--host-rope] --out PATH        # atlas-lerobot env
`--single-step-bundle` is the existing one-step bundle (read only): its bundle_manifest.json names the checkpoint snapshot, its
fx/fixtures.safetensors gives the observation tensor shapes, dtypes and the camera masks of the capture cases. SmolVLA has no
one-step bundle: its numbers come from the policy's config.json, the VLM's config.json and lerobot's euler_integrate itself
(probed), and the phase graphs are checked against them when composed (compose_phases.py smolvla)."""
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
            "check": "equals the encode graph output shape (verified at equivalence time, see tile-aware.md#per-phase-graphs-and-loop-manifest)"}
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
    shim = Path(__file__).resolve().parents[3] / "src" / "loaders" / "pi0_model.py"
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


# ------------------------------------------------------------------------------------------------------------------
# SmolVLA
# ------------------------------------------------------------------------------------------------------------------

def euler_integrate_grid(num_steps):
    """(times, dt, record) of lerobot's euler_integrate(denoise_fn, noise, num_steps) with its defaults (NOISE_AT_ONE, no
    time_grid, no RTC), read by running it: `times` are the float32 values of the time tensors it passes to denoise_fn
    (torch.full((batch,), 1.0 + step * dt, dtype=float32), the product taken in Python floats), `dt` the Python float
    -1.0 / num_steps it multiplies v_t by. Also checked here, with a probe velocity, that its update x_t + dt * v_t on a
    float32 x_t equals x_t + float32(dt) * v_t in float32 (PyTorch's wrapped-number promotion), so a host may hold dt as
    a float32 scalar."""
    import torch
    from lerobot.policies.common.flow_matching import euler_integrate

    seen, xs, vs = [], [], []
    g = torch.Generator().manual_seed(0)

    def denoise(x_t, time):
        seen.append(time.clone())
        xs.append(x_t.clone())
        vs.append(torch.randn(x_t.shape, generator=g, dtype=torch.float32) * 3)
        return vs[-1]

    noise = torch.randn(2, 5, 3, generator=torch.Generator().manual_seed(1))
    out = euler_integrate(denoise, noise, num_steps)
    if len(seen) != num_steps or any(t.dtype != torch.float32 or t.shape != (2,) or t[0] != t[1] for t in seen):
        raise SystemExit("euler_integrate did not pass one float32 time per batch row per step")
    dt = -1.0 / num_steps
    expected = [torch.full((2,), 1.0 + step * dt, dtype=torch.float32) for step in range(num_steps)]
    if not all(torch.equal(a, b) for a, b in zip(seen, expected)):
        raise SystemExit("euler_integrate's time grid is not 1.0 + step * (-1 / num_steps)")
    dt32 = torch.tensor(dt, dtype=torch.float32)
    steps = xs[1:] + [out]
    python_float = all(torch.equal(x + dt * v, nxt) for x, v, nxt in zip(xs, vs, steps))
    float32 = all(torch.equal(x + dt32 * v, nxt) for x, v, nxt in zip(xs, vs, steps))
    if not python_float:
        raise SystemExit("euler_integrate's update is not x_t + dt * v_t with the Python float dt")
    times = [float(t[0]) for t in seen]
    return times, dt, {"update_reproduced_with_python_float_dt": python_float,
                       "update_reproduced_with_float32_dt": float32, "dt_float32": float(dt32)}


def _hf_file(repo_id, name):
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id, name, local_files_only=True))


def smolvla_manifest(policy_id, n_images, host_rope=False):
    import torch
    import lerobot.policies.common.flow_matching as fm
    import lerobot.policies.smolvla.modeling_smolvla as ms
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    cfg_path = _hf_file(policy_id, "config.json")
    cfg = json.loads(cfg_path.read_text())
    if cfg["type"] != "smolvla":
        raise SystemExit(f"{policy_id} is a {cfg['type']!r} policy")
    if cfg.get("add_image_special_tokens") or cfg.get("prefix_length", 0) > 0 or not cfg.get("use_cache", True):
        raise SystemExit("this manifest covers add_image_special_tokens=False, prefix_length=0, use_cache=True")
    if (cfg.get("rtc_config") or {}).get("enabled"):
        raise SystemExit("RTC guidance is enabled in the config; the host loop here has none")
    vlm_path = _hf_file(cfg["vlm_model_name"], "config.json")
    vlm = json.loads(vlm_path.read_text())
    text, vision = vlm["text_config"], vlm["vision_config"]
    cameras = sorted(k for k, f in cfg["input_features"].items() if f["type"] == "VISUAL")
    if n_images != len(cameras) + cfg.get("empty_cameras", 0):
        raise SystemExit(f"--n-images {n_images}: the policy has cameras {cameras} (+{cfg.get('empty_cameras', 0)} empty)")
    side = vision["image_size"] // vision["patch_size"]
    tokens_per_image = side * side // vlm["scale_factor"] ** 2
    lang, chunk, adim = cfg["tokenizer_max_length"], cfg["chunk_size"], cfg["max_action_dim"]
    prefix_len = n_images * tokens_per_image + lang + 1
    layers, kv_heads, head_dim = cfg["num_vlm_layers"], text["num_key_value_heads"], text["head_dim"]
    every = cfg["self_attn_every_n_layers"]
    num_steps = cfg["num_steps"]
    times, dt, update = euler_integrate_grid(num_steps)
    src, swe_src, fm_src = Path(ms.__file__), Path(swe.__file__), Path(fm.__file__)
    kv_shape = [1, prefix_len, kv_heads, head_dim]
    roles = ["self_attention" if (every > 0 and i % every == 0) or "cross" not in cfg["attention_mode"] else
             "cross_attention" for i in range(layers)]
    kv = {"layers": layers, "tensors": 2 * layers, "names": [f"{t}_{i}" for i in range(layers) for t in "kv"],
          "per_tensor_shape": kv_shape, "dtype": "BF16",
          "layout": "[batch, prefix tokens, kv heads, head_dim]: the k_proj / v_proj output layout; lerobot's DynamicCache "
                    "holds the same values transposed, [batch, kv heads, prefix tokens, head_dim]",
          "content": "post-RoPE keys and values of every VLM text layer over the prefix (images, language, state)",
          "layer_roles": roles,
          "role_use_in_step": {
              "self_attention": "the expert's suffix K / V are appended along the token axis and the chunk attends to "
                                "[prefix; suffix] (forward_attn_layer)",
              "cross_attention": "the expert projects the prefix K / V with its own k_proj / v_proj and the chunk attends "
                                 "to the prefix only (forward_cross_attn_layer)"},
          "derivation": f"{layers} VLM layers (num_vlm_layers), {kv_heads} kv heads and head_dim {head_dim} "
                        f"({cfg['vlm_model_name']} text_config); prefix = {n_images} cameras x {tokens_per_image} tokens "
                        f"((image {vision['image_size']} / patch {vision['patch_size']})^2 / scale_factor "
                        f"{vlm['scale_factor']}^2) + {lang} language tokens (tokenizer_max_length) + 1 state token",
          "check": "tensor shapes of the prefix graph outputs and step graph inputs (compose_phases.py smolvla)"}
    rope = {"cos_sin_shape_prefix": [1, prefix_len, 1, head_dim // 2], "cos_sin_shape_step": [1, chunk, 1, head_dim // 2],
            "dtype": "F32", "angles": "positions / 10000 ** ((2 / head_dim) * arange(head_dim / 2)) "
                                      "(smolvlm_with_expert.apply_rope, float32)"}
    prefix_positions = ({"prefix_cos": {"shape": rope["cos_sin_shape_prefix"], "dtype": "F32"},
                         "prefix_sin": {"shape": rope["cos_sin_shape_prefix"], "dtype": "F32"}} if host_rope else
                        {"position_ids": {"shape": [1, prefix_len], "dtype": "I64"}})
    step_positions = ({f"{name}_{f}": {"shape": rope["cos_sin_shape_step"], "dtype": "F32"}
                       for name in ("suffix", "expert") for f in ("cos", "sin")} if host_rope else
                      {"position_ids": {"shape": [1, chunk], "dtype": "I64"}})
    noise = {"shape": [1, chunk, adim], "dtype": "F32", "distribution": "standard normal (sample_noise, float32)"}
    return {
        "schema": SCHEMA, "model": "smolvla", "objective": "flow matching, Euler (noise at t = 1)",
        "sources": {"policy_config": str(cfg_path), "policy_config_sha256": _sha(cfg_path),
                    "vlm_config": str(vlm_path), "vlm_config_sha256": _sha(vlm_path),
                    "step_count": f"num_steps {num_steps} in config.json; "
                                  + _line_range(src, "num_steps = self.config.num_steps"),
                    "loop": _line_range(src, "return euler_integrate(") + " -> " + _line_range(fm_src, "def euler_integrate"),
                    "time_grid": _line_range(fm_src, "uniform_dt = ") + ", " + _line_range(fm_src, "def uniform_time"),
                    "step": _line_range(src, "def denoise_step"), "prefix": _line_range(src, "def sample_actions"),
                    "cache_read": _line_range(swe_src, "def forward_cross_attn_layer"),
                    "cache_append": _line_range(swe_src, "key_states, value_states = past_key_values.update("),
                    "torch": torch.__version__},
        "denoising_steps": num_steps,
        "euler": {"times": times, "times_dtype": "float32",
                  "time_formula": "1.0 + step * dt in Python floats, then torch.full((batch,), ., dtype=float32), "
                                  "step = 0 .. num_steps - 1 (euler_integrate, time_grid None)",
                  "dt": dt, "dt_dtype": "Python float (float64 scalar) multiplied into the float32 v_t; the product is "
                                        "float32 (wrapped-number promotion)",
                  **update, "update": "x_t = x_t + dt * v_t (float32)", "count": len(times),
                  "probe": "make_loop_manifest.euler_integrate_grid runs lerobot's euler_integrate with a recording "
                           "denoise_fn and checks these values and the update"},
        "policy": {"id": policy_id, "chunk_size": chunk, "max_action_dim": adim, "tokens_per_image": tokens_per_image,
                   "head_dim": head_dim, "prefix_len": prefix_len, "action_dim": cfg["output_features"]["action"]["shape"][0],
                  "max_state_dim": cfg["max_state_dim"], "tokenizer_max_length": lang, "num_vlm_layers": layers,
                  "self_attn_every_n_layers": every, "attention_mode": cfg["attention_mode"],
                  "vlm_model_name": cfg["vlm_model_name"], "image_size": cfg["resize_imgs_with_padding"],
                  "cameras": cameras, "host_rope": host_rope},
        "graphs": {
            "prefix": {"path": "prefix bundle fx/", "calls_per_action_chunk": 1,
                       "boundary": "SmolVLAPrefix: embed_prefix (vision tower + connector per camera, language "
                                   "embedding, state_proj) and every VLM text layer up to its K / V",
                       "inputs": {**{f"image_{i}": {"shape": [1, 3, *cfg["resize_imgs_with_padding"][::-1]],
                                                    "dtype": "BF16"} for i in range(n_images)},
                                  "lang_tokens": {"shape": [1, lang], "dtype": "I64"},
                                  "state": {"shape": [1, cfg["max_state_dim"]], "dtype": "BF16"},
                                  "attention_mask": {"shape": [1, prefix_len, prefix_len], "dtype": "BF16",
                                                     "form": "additive: 0 kept, BF16 minimum masked"},
                                  **prefix_positions},
                       "outputs": {"kv_cache": kv}},
            "step": {"path": "denoise_step bundle fx/", "calls_per_action_chunk": num_steps,
                     "boundary": "SmolVLADenoiseStep: embed_suffix (action / time MLP) + the action expert reading the "
                                 "prefix K / V + action_out_proj",
                     "inputs": {"x_t": {"shape": noise["shape"], "dtype": noise["dtype"]},
                                "time": {"shape": [1], "dtype": "F32"},
                                "attention_mask": {"shape": [1, chunk, prefix_len + chunk], "dtype": "BF16",
                                                   "form": "additive: 0 kept, BF16 minimum masked"},
                                **step_positions, "kv_cache": kv},
                     "outputs": {"v_t": {"shape": [1, chunk, adim], "dtype": "F32"}}}},
        "per_call_inputs": {
            "prefix": f"once per action chunk: {n_images} camera images, tokenized instruction, state",
            "step": {"changes_every_call": ["x_t (previous Euler output; the noise at the first call)",
                                            "time (euler.times[i], float32, expanded to batch)"],
                     "constant_over_the_loop": ["attention_mask and positions (they depend only on the prefix pad mask)",
                                                f"kv_cache ({kv['tensors']} tensors, the prefix graph outputs; the step "
                                                "graph never writes them)"]}},
        "host_inputs": {
            "prefix_pad_masks": f"[1, {prefix_len}] bool: per camera {tokens_per_image} copies of its image mask, the "
                                "tokenizer attention mask, 1 (state); block mask 1 on the state token only "
                                "(embed_prefix; fx_export.smolvla_prefix_masks)",
            "prefix attention_mask / position_ids": "make_att_2d_masks(pad, block) as where(keep, 0, bf16_min); "
                                                    "cumsum(pad) - 1 (fx_export.smolvla_prefix_inputs)",
            "step attention_mask": f"keep = [prefix_pad_masks broadcast over the {chunk} queries | causal over the chunk]"
                                   " (denoise_step; fx_export.smolvla_step_host_inputs)",
            "step position_ids": "sum(prefix_pad_masks) + arange(chunk); the cross-attention layers use arange(chunk) "
                                 "(position_ids minus its row minimum)",
            **({"rope_tables": rope} if host_rope else {}),
            "language_padding": "padded language tokens stay padded (keep_self_for_padded off): their prefix rows are "
                                "fully masked, which the additive form keeps uniform, as the boolean form does"},
        "initial_noise": noise | {"draws": 1},
        "loop": ["k_0, v_0, ... = prefix(images, lang_tokens, state, mask, positions)      # 1 call",
                 "x = normal(1, chunk, max_action_dim) float32                             # host",
                 "mask, positions = step host inputs (from prefix_pad_masks)                # host, once",
                 f"for t in euler.times:                                                   # {len(times)} iterations",
                 "    v = step(x, full([1], t, float32), mask, positions, k_0, v_0, ...)    # step calls",
                 "    x = x + dt * v                                                        # host, float32",
                 f"actions = x[:, :, :{cfg['output_features']['action']['shape'][0]}]        # host, unpad"],
        "host_operations": [
            {"op": "preprocessing", "detail": "policy_preprocessor.json: camera rename, state normalization (MEAN_STD), "
                                              "tokenization (padding to max_length), device / dtype"},
            {"op": "image preparation", "detail": f"resize_with_pad to {cfg['resize_imgs_with_padding']}, [0, 1] -> "
                                                  "[-1, 1] (SmolVLAPolicy.prepare_images)"},
            {"op": "state padding", "detail": f"pad to max_state_dim {cfg['max_state_dim']} (prepare_state)"},
            {"op": "masks and positions", "detail": "see host_inputs"},
            {"op": "noise draw", "detail": "1 draw per chunk"},
            {"op": "Euler update", "detail": "x_t + dt * v_t on the host (the step graph returns v_t)"},
            {"op": "output postprocessing", "detail": "unpad to the action dim, unnormalize (policy_postprocessor.json)"}],
        "not_provided": ["unrolled 10-step graph", "RTC guidance (rtc_config is null)"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit", "pi0", "smolvla"))
    ap.add_argument("--single-step-bundle")
    ap.add_argument("--policy", default="lerobot/smolvla_libero", help="smolvla: policy repo id (local HF cache)")
    ap.add_argument("--n-images", type=int, default=2, help="smolvla: cameras at the prefix boundary")
    ap.add_argument("--host-rope", action="store_true", help="smolvla: the phase graphs take host RoPE tables")
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--openpi-dir")
    ap.add_argument("--stack-cameras", action="store_true", help="pi0: record the one-tower-call deviation instead of one SigLIP call per camera")
    a = ap.parse_args(argv)
    if a.model != "smolvla" and not a.single_step_bundle:
        ap.error(f"{a.model} needs --single-step-bundle")
    if a.model == "dit":
        manifest = dit_manifest(a.single_step_bundle)
    elif a.model == "smolvla":
        manifest = smolvla_manifest(a.policy, a.n_images, a.host_rope)
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
