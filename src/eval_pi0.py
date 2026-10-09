"""Pi0 (openpi PI0Pytorch) action accuracy before and after FP8 quantization.

The reference author's pi0-quant (third_party/pi0-quant @8bc0f6a) is the Pi0 reference profile:
model, checkpoint, config, observation pipelines, noise and metrics follow it (src/pi0_model.py
has the loader sources). Every mode is compared with the unquantized baseline `bf16` (openpi's
own mixed precision: bfloat16 weights; norms, SigLIP patch / position embeddings and the action
head in float32, gemma_pytorch.py:71-82, pi0_pytorch.py:100-109) on the same frames and noise.

Observation sources (--obs), each replicating one reference script exactly:
  dummy      experiments/run_output_eval.py: --n-obs (default 4) observations from
             _make_dummy_obs (:87-95) with np.random.default_rng(seed) (:439-440), through the
             reference policy shim Pi0PyTorchPolicy.infer (serve_quant.py:395-495: state
             normalized, actions unnormalized, joint deltas made absolute, first 8 dims kept).
  mxu_dummy  experiments/run_eval_mxu_io.py: torch.manual_seed(0) then --n-obs model-level
             observations from its _make_dummy_obs(config, device) (:114-133, :793-801),
             straight into model.sample_actions (:579, :652); actions are the first 8 dims of the
             model's (normalized, delta) output.
  sim        run_eval_mxu_io.py --obs-dir/--obs-file: sim-evals obs_*.npz files through its
             _load_real_obs (:136-256), model-level as mxu_dummy. The files are not public; pass
             your own.
  droid      NEW, ground truth: lerobot/droid_1.0.1 frames with recorded joint-position action
             chunks (pi0_model.droid_observations), through the policy shim as dummy.

Flow-matching noise (--noise), always explicit (passed to sample_actions), never the global RNG:
  reference  (default) the reference's own draws, reproduced bit for bit with a per-frame
             torch.Generator on the model's device: policy-level sources draw from one stream
             seeded once per pass (run_output_eval.py:176, :216, sample_noise = torch.normal,
             pi0_pytorch.py:173-180), so frame i takes draw i of a generator seeded `seed`;
             model-level sources reseed per frame (run_eval_mxu_io.py:579, :652), so frame i
             takes the first draw of a generator seeded seed + i (seed 0 = the reference).
  per_frame  pi0_model.frame_noise(seed + i): drawn on the CPU, so identical across devices.

Per mode:
  headline         (NEW) the delta-space numbers below: agreement with bf16 and, for droid, MSE /
                   MAE vs the recorded deltas. The reference's action metric is on absolute joint
                   positions (state added back, serve_quant.py:488-492), where the robot's pose
                   dominates the norm and hides the quantization error;
  vs_bf16_delta,   as vs_bf16 / vs_ground_truth on joint deltas (pred - state, gt - state on the
  vs_ground_truth_delta  7 joint dims; gripper absolute). Model-level sources are deltas already;
  vs_bf16          rel_l2, cosine, max_abs, per-dimension rel_l2 (eval_policy.agreement), plus
                   per-dimension RMSE / max_abs / cosine (NEW), and the reference's action metric
                   (run_output_eval.py:251-295: per-observation and overall RMSE, ref RMS, rel
                   RMSE, written as action_rmse.csv by the reference function itself);
  vs_ground_truth  (--obs droid) MSE / MAE overall and per dimension vs the recorded chunks;
  rmse             (--rmse) per-layer isolated and cumulative relative RMSE of every quantized
                   Linear / Conv2d against a clean pass on the same frame and noise, by component
                   vision / language / action_expert / action_head (run_eval_mxu_io.py:493-679,
                   CSVs as its :37-40; src/layer_rmse.py, components model_patcher.py:92-109);
  determinism      each mode runs twice with the same noise; must reproduce bit for bit;
  patch_coverage   every Linear / Conv2d / attention site, patched and called or why not;
  ladder           the coverage-ladder record (quant_modes.ladder_record); the result's
                   ladder_deltas difference consecutive steps on these same frames, and a subset
                   profile (reference_hw, chip, chip_rope_vpu) brings in bf16 and ref_conv unless
                   --subset-compare off; chip tables (RoPE, timestep) default to --chip-rope host;
  layers           the profile report: for reference_hw / chip the MXU cross-check
                   (layers.crosscheck, layers.valid), for chip the unit coverage
                   (layers.coverage), as eval.py / eval_policy.py store them.

The reference's policy shim always runs 10 flow steps (serve_quant.py:479: num_steps=10), so
run_output_eval's --steps has no effect there (its README: "label only"); here --steps reaches
sample_actions for every source, and anything but 10 is flagged reduced_steps.

Example (L40S):
  OPENPI_DIR=.../openpi python src/eval_pi0.py --checkpoint-dir .../pi0_droid_jointpos_polaris \\
      --tokenizer-path .../paligemma_tokenizer.model --modes bf16 reference reference_hw torchao \\
      --obs dummy --device cuda --out out/pi0_dummy.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

import pi0_model
from eval_policy import agreement, determinism_state, error_vs_gt, make_deterministic, repeat_agreement
from pi0_model import CallCounter, inventory, patch_coverage
from quant_modes import (ALL_MODES, add_profile_args, add_subset_compare_arg, finalize_report, ladder_deltas,
                         ladder_record, quantize, require_mode_supported, subset_compare_modes)

REFERENCE_STEPS = 10      # serve_quant.py:479, run_output_eval.py:356, run_eval_mxu_io.py:707
ACTION_DIMS = 8           # serve_quant.py:494-495: 7 joints + gripper
OBS_LEVEL = {"dummy": "policy", "droid": "policy", "mxu_dummy": "model", "sim": "model"}

# Pi0-specific decisions with no source in pi0-quant (marked NEW in code, here and in the JSON).
NEW_DECISIONS = [
    {"decision": "flow-matching noise passed explicitly to sample_actions from a per-frame "
                 "torch.Generator on the model device (--noise reference: the reference's draws "
                 "reproduced bit for bit; per_frame: CPU draws seeded seed + frame)",
     "reference": "global RNG: torch.manual_seed(seed) once per pass (run_output_eval.py:176, "
                  ":216) or torch.manual_seed(i) per frame (run_eval_mxu_io.py:579, :652); "
                  "sample_noise draws (pi0_pytorch.py:173-180)",
     "reason": "a frame's noise must not depend on the frames, modes or profiles that ran before "
               "it (determinism check, --rmse clean pass)"},
    {"decision": "--obs droid: lerobot/droid_1.0.1 (joint-position state and actions), first data "
                 "and video file, RandomState(seed) episode shuffle, one frame per episode, episodes "
                 "without a prompt skipped; images resized with padding to 224x224 on the client "
                 "side as openpi's DROID client does (examples/droid/main.py:120-123)",
     "reference": "run_eval.py:183-298 reads lerobot/droid_100 and stretches images to 224x224 "
                  "(run_eval.py:247)",
     "reason": "droid_100's state/actions are cartesian (meta/stats.json), not the joint-position "
               "space of pi0_droid_jointpos_polaris; droid_1.0.1 has the matching recorded actions. "
               "DROID 1.0.1 is this checkpoint's training mixture (openpi polaris_config.py:116-142, "
               "weight 0.9) and has one split, so no episode is guaranteed held out"},
    {"decision": "every mode on a freshly loaded model; the baseline is a separate bf16 run",
     "reference": "baseline and patched pass on one model object, unpatched first "
                  "(run_output_eval.py:172-236)",
     "reason": "the profiles replace modules in place and are not undone; equivalent, since the "
               "unpatched model is deterministic under fixed noise"},
    {"decision": "headline metrics in joint-delta space (pred - state, gt - state on the joint dims): "
                 "rel_l2, cosine, per dimension, MSE / MAE vs the recorded deltas",
     "reference": "action RMSE on absolute joint positions after AbsoluteActions "
                  "(run_output_eval.py:251-295, serve_quant.py:488-492)",
     "reason": "the state added back dominates the absolute norm, so absolute rel-L2 / rel RMSE "
               "understate the quantization error; the reference metric is kept as is"},
    {"decision": "agreement metrics beyond the reference RMSE: rel_l2, cosine, max_abs and "
                 "per-dimension rel_l2 / RMSE / max_abs / cosine",
     "reference": "overall and per-observation RMSE and rel RMSE only (run_output_eval.py:251-295)",
     "reason": "a single channel (e.g. the gripper) can dominate a pooled RMSE"},
    {"decision": "lm_head (PaliGemma and expert) left unpatched by default (--skip lm_head)",
     "reference": "patch_model replaces every nn.Linear (model_patcher.py:246-288)",
     "reason": "never called in Pi0 inference (run_eval.py:20-22): identical results, ~3 GB less "
               "memory under reference_hw"},
    {"decision": "one pair of score-matmul accumulators per attention module",
     "reference": "one pair per patch function (model_patcher.py:699-700, 902-903)",
     "reason": "the MXU models here keep no state for non-weight operands, so it is equivalent"},
    {"decision": "chip: SigLIP attention stays on the SDPA route (not switched to eager)",
     "reference": "SigLIP runs through patch_attn_sdpa under transformers 4.53.2 (vision config "
                  "attn_implementation 'sdpa')",
     "reason": "the eager route has a float32 softmax (model_patcher.py:1134), the SDPA route a "
               "BF16 one (787); keep the reference's route"},
    {"decision": "checkpoint converted with openpi's converter at its default precision (bfloat16)",
     "reference": "run_eval.py:855 loads pi0_droid_jointpos_safetensors; its precision is not "
                  "recorded",
     "reason": "converter default (convert_jax_model_to_pytorch.py:562); float32 layers of the "
               "model hold BF16-representable values"},
]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--openpi-dir", default=os.environ.get("OPENPI_DIR"),
                    help=f"openpi source tree (default $OPENPI_DIR), pinned at {pi0_model.OPENPI_PIN[:12]}")
    ap.add_argument("--checkpoint-dir", default=os.environ.get("PI0_CHECKPOINT"),
                    help="converted PyTorch checkpoint dir with model.safetensors (default $PI0_CHECKPOINT)")
    ap.add_argument("--config", default=pi0_model.DEFAULT_CONFIG,
                    help="openpi config name, resolved by the reference table (serve_quant.py:93-149)")
    ap.add_argument("--norm-stats-dir", default=None,
                    help="dir with norm_stats.json (default <checkpoint-dir>/assets/droid)")
    ap.add_argument("--tokenizer-path", default=os.environ.get("PALIGEMMA_TOKENIZER"),
                    help=f"paligemma_tokenizer.model ({pi0_model.TOKENIZER_SOURCE})")
    ap.add_argument("--obs", choices=tuple(OBS_LEVEL), default="dummy",
                    help="dummy: run_output_eval.py observations; mxu_dummy: run_eval_mxu_io.py "
                         "observations; sim: run_eval_mxu_io.py --obs-dir files; droid: real frames "
                         "with recorded actions (lerobot/droid_1.0.1)")
    ap.add_argument("--obs-dir", default=None, help="--obs sim: directory of obs_*.npz")
    ap.add_argument("--obs-file", default=None, help="--obs sim: one obs file (as the reference's)")
    ap.add_argument("--n-obs", type=int, default=4, help="observations (both reference scripts: 4)")
    ap.add_argument("--frame-idx", type=int, default=0, help="--obs droid: frame within each episode")
    ap.add_argument("--steps", type=int, default=REFERENCE_STEPS,
                    help="flow-matching steps (the reference uses 10; fewer is flagged reduced_steps)")
    ap.add_argument("--seed", type=int, default=0, help="observation and noise seed (reference: 0)")
    ap.add_argument("--noise", choices=("reference", "per_frame"), default="reference",
                    help="reference: the reference's draws, reproduced with explicit generators; "
                         "per_frame: CPU draws seeded seed + frame (device-independent)")
    ap.add_argument("--modes", nargs="+", default=["bf16", "reference"], choices=list(ALL_MODES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
    ap.add_argument("--skip", nargs="*", default=list(pi0_model.DEFAULT_SKIP),
                    help="fqn substrings left unquantized (default: lm_head, never called)")
    ap.add_argument("--rmse", action="store_true",
                    help="per-layer and cumulative relative RMSE against a clean pass (reference CSVs)")
    ap.add_argument("--rmse-dir", default=None, help="default: <out stem>_rmse next to --out")
    ap.add_argument("--no-determinism-check", dest="determinism_check", action="store_false")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True), cuDNN deterministic, cuBLAS config")
    ap.add_argument("--no-hash", action="store_true", help="do not hash model.safetensors")
    ap.add_argument("--out", required=True)
    add_profile_args(ap)
    add_subset_compare_arg(ap)
    return ap


# ---------------------------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------------------------

def _normal(shape, g, device):
    # PI0Pytorch.sample_noise (pi0_pytorch.py:173-180), with an explicit generator
    return torch.normal(mean=0.0, std=1.0, size=shape, generator=g, dtype=torch.float32, device=device)


def frame_noise(kind: str, level: str, seed: int, frame: int, shape, device) -> torch.Tensor:
    """The noise of frame `frame` (module docstring, --noise)."""
    if kind == "per_frame":
        return pi0_model.frame_noise(seed + frame, shape[1], shape[2], device)
    device = torch.device(device)
    g = torch.Generator(device=device)
    if level == "policy":       # one stream per pass: run_output_eval.py:176, :216
        g.manual_seed(seed)
        draws = frame + 1
    else:                        # reseeded per frame: run_eval_mxu_io.py:579, :652
        g.manual_seed(seed + frame)
        draws = 1
    for _ in range(draws):
        x = _normal(shape, g, device)
    return x


def noise_digest(noises) -> str:
    h = hashlib.sha256()
    for n in noises:
        h.update(n.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------

def mxu_dummy_observations(n: int, cfg, device, seed: int = 0) -> list[dict]:
    """run_eval_mxu_io.py:793-801: torch.manual_seed(0), then n of its _make_dummy_obs, drawn by
    the reference function on `device` (the CUDA and CPU streams differ, as in the reference).
    The global RNG state is restored afterwards (NEW)."""
    import ref_loader

    rio = ref_loader.load_file("pi0-quant", "experiments/run_eval_mxu_io.py")
    dev = torch.device(device)
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        obs = [rio._make_dummy_obs(cfg, dev) for _ in range(n)]
    return [{"obs": o, "gt": None, "meta": {"source": "mxu_dummy", "index": i, "generated_on": str(dev)}}
            for i, o in enumerate(obs)]


def sim_observations(obs_dir, obs_file, cfg, checkpoint_dir, norm_stats_dir, tokenizer_path,
                     device) -> list[dict]:
    """run_eval_mxu_io.py --obs-dir/--obs-file: its _load_real_obs (:136-256) itself. That function
    finds the tokenizer only under ~/Desktop or ~/.cache/openpi/big_vision (:163-173), so it runs
    with HOME pointing at a temporary dir that links --tokenizer-path there (NEW)."""
    import ref_loader

    rio = ref_loader.load_file("pi0-quant", "experiments/run_eval_mxu_io.py")
    old = os.environ.get("HOME")
    with tempfile.TemporaryDirectory() as home:
        d = Path(home) / ".cache" / "openpi" / "big_vision"
        d.mkdir(parents=True)
        (d / "paligemma_tokenizer.model").symlink_to(Path(tokenizer_path).resolve())
        os.environ["HOME"] = home
        try:
            obs = rio._load_real_obs(obs_dir, cfg, checkpoint_dir, torch.device(device),
                                     obs_file=obs_file, norm_stats_dir=norm_stats_dir)
        finally:
            if old is None:
                os.environ.pop("HOME", None)
            else:
                os.environ["HOME"] = old
    return [{"obs": o, "gt": None, "meta": {"source": "sim", "file": getattr(o, "_source", None)}}
            for o in obs]


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------

def _t(xs):
    return [torch.as_tensor(np.asarray(x, dtype=np.float32)) for x in xs]


def per_dim(preds, ref) -> dict:
    """NEW: per action dimension RMSE, max |diff| and cosine over all frames and horizon steps."""
    p, r = torch.cat(_t(preds)).double(), torch.cat(_t(ref)).double()
    d = p - r
    cos = (p * r).sum(0) / (p.norm(dim=0) * r.norm(dim=0)).clamp_min(1e-300)
    return {"rmse_per_dim": d.pow(2).mean(0).sqrt().tolist(), "max_abs_per_dim": d.abs().amax(0).tolist(),
            "cosine_per_dim": cos.tolist()}


def vs_bf16(preds, ref, csv_path) -> dict:
    out = {**agreement(_t(preds), _t(ref)), **per_dim(preds, ref)}
    out["reference_action_rmse"] = reference_action_rmse(ref, preds, csv_path)
    return out


JOINT_DIMS = 7   # AbsoluteActions(make_bool_mask(7, -1)): joints absolute, gripper not (serve_quant.py:488-492)


def frame_state(obs) -> np.ndarray | None:
    """The raw robot state (7 joints + gripper) a policy-level frame was given, as the shim builds
    it (serve_quant.py:424-427); None for model-level frames."""
    if not isinstance(obs, dict):
        return None
    return np.concatenate([np.asarray(obs["observation/joint_position"], np.float32).flatten(),
                           np.asarray(obs["observation/gripper_position"], np.float32).flatten()])


def to_delta(chunks, frames, level):
    """NEW: actions as joint deltas from the frame's state (pred - state on the joint dims, the
    inverse of the shim's AbsoluteActions; the gripper stays absolute, as the model emits it).
    Model-level outputs are already deltas (normalized) and are returned unchanged."""
    if level != "policy":
        return [np.asarray(c, np.float32) for c in chunks]
    out = []
    for c, fr in zip(chunks, frames):
        c = np.array(c, dtype=np.float32, copy=True)
        c[:, :JOINT_DIMS] -= frame_state(fr["obs"])[:JOINT_DIMS]
        out.append(c)
    return out


def headline(entry) -> dict:
    """NEW: the delta-space numbers first. On absolute joint positions the robot's pose dominates
    the norm, so the reference's absolute-space rel RMSE understates the quantization error."""
    h = {}
    if "vs_bf16_delta" in entry:
        d = entry["vs_bf16_delta"]
        h.update(delta_rel_l2=d["rel_l2"], delta_cosine=d["cosine"], delta_max_abs=d["max_abs"],
                 reference_rel_rmse_absolute=entry["vs_bf16"]["reference_action_rmse"]["rel_rmse"])
    if "vs_ground_truth_delta" in entry:
        h.update(gt_delta_mse=entry["vs_ground_truth_delta"]["mse"],
                 gt_delta_mae=entry["vs_ground_truth_delta"]["mae"])
    return h


def vs_ground_truth(preds, gts) -> dict:
    """MSE / MAE against the recorded chunks (eval_policy.error_vs_gt, no padding: every chunk is
    complete), plus per dimension (NEW)."""
    p, g = _t(preds), _t(gts)
    out = error_vs_gt(p, g, [torch.zeros(x.shape[0], dtype=torch.bool) for x in p])
    d = torch.cat(p).double() - torch.cat(g).double()
    out.update(mse_per_dim=d.pow(2).mean(0).tolist(), mae_per_dim=d.abs().mean(0).tolist())
    return out


def reference_action_rmse(base, quant, csv_path) -> dict:
    """The reference's action metric, by the reference function (run_output_eval.py:251-295)."""
    import ref_loader

    roe = ref_loader.load_file("pi0-quant", "experiments/run_output_eval.py")
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rmse, ref_rms = roe._write_action_rmse(csv_path, _t(base), _t(quant))
    return {"overall_rmse": rmse, "overall_ref_rms": ref_rms,
            "rel_rmse": rmse / ref_rms if ref_rms > 0 else float("nan"), "csv": str(csv_path)}


# ---------------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------------

def model_actions(model, obs, noise, steps, device) -> np.ndarray:
    """Model-level frame (run_eval_mxu_io.py:652-653): the first ACTION_DIMS of (horizon, 32)."""
    with torch.no_grad():
        a = model.sample_actions(str(device), obs, noise=noise, num_steps=steps)
    return a[0, :, :ACTION_DIMS].float().cpu().numpy()


def make_runner(model, policy, level, steps, device):
    if level == "policy":
        if policy is None:
            raise ValueError("policy-level observations need the policy shim (--tokenizer-path)")
        return lambda obs, noise: pi0_model.predict(policy, obs, noise, steps)
    return lambda obs, noise: model_actions(model, obs, noise, steps, device)


def run_frames(run, frames, noises, device, before_frame=None):
    preds, times = [], []
    for i, (fr, noise) in enumerate(zip(frames, noises)):
        if before_frame is not None:
            before_frame(fr["obs"], noise)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        a = run(fr["obs"], noise)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        preds.append(np.asarray(a, dtype=np.float32))
        print(f"  frame {i + 1}/{len(frames)}: {times[-1]:.1f}s", flush=True)
    return preds, times


def evaluate(args, frames, build, level, horizon, action_dim, out: Path) -> dict:
    """Run every mode of args.modes (bf16 first) on `frames`.

    build(mode) -> (model, inventory, layers report or None, policy or None): a freshly loaded
    model with `mode` applied, and the policy shim around it for policy-level frames."""
    shape = (1, horizon, action_dim)
    noises = [frame_noise(args.noise, level, args.seed, i, shape, args.device) for i in range(len(frames))]
    rmse_dir = Path(args.rmse_dir) if args.rmse_dir else out.with_name(out.stem + "_rmse")
    act_dir = out.with_name(out.stem + "_actions")
    modes = {}
    ref = ref_delta = clean = clean_run = None
    order = ["bf16", *[m for m in args.modes if m != "bf16"]]      # bf16 first: the baseline
    # the frames are always a sample, so a subset profile gets ref_conv on the same frames
    order, added = subset_compare_modes(order, getattr(args, "subset_compare", "on") == "on")
    for mode in order:
        print(f"== {mode}", flush=True)
        model, inv, layers, policy = build(mode)
        eq = (layers or {}).get("ladder", {}).get("equivalent_to")
        if eq in modes:
            # the op class this step adds is absent from the model: same model, same numbers
            modes[mode] = {**modes[eq], "layers": finalize_report(layers),
                           "ladder": {**layers["ladder"], "duplicate_of": eq},
                           "added_by_subset_compare": mode in added}
            print(f"{mode}: identical to {eq} (nothing added to quantize); not re-run", flush=True)
            del model, policy
            continue
        run = make_runner(model, policy, level, args.steps, args.device)
        counter = CallCounter(model, [f for f, i in inv.items() if i["kind"] in ("linear", "conv2d")])
        rm = before = None
        if args.rmse and mode != "bf16":
            import layer_rmse

            if clean is None:
                cm, _, _, cp = build("bf16")
                clean, clean_run = cm, make_runner(cm, cp, level, args.steps, args.device)
            rm = layer_rmse.LayerRMSE(clean, model, default_component="UNKNOWN")

            def before(obs, noise):
                rm.capture(lambda: clean_run(obs, noise))
        preds, times = run_frames(run, frames, noises, args.device, before_frame=before)
        counter.close()
        entry = {"patch_coverage": patch_coverage(inv, model, mode, layers, counter.counts, args.skip),
                 "latency_s": {"median": sorted(times)[len(times) // 2], "mean": sum(times) / len(times)},
                 "actions": [p.tolist() for p in preds]}
        if rm is not None:
            rm.close()
            entry["rmse"] = {**rm.summary(), "csv": rm.write_csvs(rmse_dir / mode)}
        entry["deterministic"] = None
        if args.determinism_check:
            again, _ = run_frames(run, frames, noises, args.device)
            entry["determinism"] = repeat_agreement(_t(preds), _t(again))
            entry["deterministic"] = entry["determinism"]["deterministic"]
            if not entry["deterministic"]:
                print(f"WARNING: mode {mode} is not deterministic: {entry['determinism']}", flush=True)
        # after every pass, so the cross-check summary covers the repeat as well
        entry["layers"] = finalize_report(layers)
        entry["ladder"] = layers["ladder"] if layers is not None else ladder_record("bf16", {}, args)
        entry["added_by_subset_compare"] = mode in added
        entry["valid"] = bool((layers or {}).get("valid", True))
        if layers is not None and layers.get("quantized", 1) == 0 and not layers.get("attention_sites"):
            print(f"WARNING: mode {mode} quantized no layers; its numbers are the baseline's", flush=True)
        delta = to_delta(preds, frames, level)
        if frames[0]["gt"] is not None:
            gts = [f["gt"] for f in frames]
            entry["vs_ground_truth"] = vs_ground_truth(preds, gts)
            entry["vs_ground_truth_delta"] = vs_ground_truth(delta, to_delta(gts, frames, level))
        if mode == "bf16":
            ref, ref_delta = preds, delta
        else:
            entry["vs_bf16"] = vs_bf16(preds, ref, act_dir / mode / "action_rmse.csv")
            entry["vs_bf16_delta"] = {**agreement(_t(delta), _t(ref_delta)), **per_dim(delta, ref_delta)}
        entry["headline"] = headline(entry)
        modes[mode] = entry
        brief = {k: v for k, v in entry.items() if k not in ("actions", "patch_coverage", "layers", "rmse")}
        print(mode, "headline:", json.dumps(entry["headline"]), flush=True)
        print(mode, json.dumps(brief, default=str)[:600], flush=True)
        print(mode, "patch coverage:", json.dumps(entry["patch_coverage"]["summary"]), flush=True)
        if isinstance((layers or {}).get("coverage"), dict):
            import chip_coverage

            print(chip_coverage.format_report(layers["coverage"]), flush=True)
        del model, policy, run
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
    return {"noise": {"kind": args.noise, "level": level, "sha256": noise_digest(noises),
                      "device": str(noises[0].device)},
            "modes": modes, "valid": all(e["valid"] for e in modes.values()),
            "subset_compare": {"enabled": bool(added) or None, "added_modes": added},
            "ladder_deltas": ladder_deltas({m: ladder_metrics(e) for m, e in modes.items()},
                                           {m: e["ladder"] for m, e in modes.items()})}


def ladder_metrics(entry) -> dict:
    """The numbers ladder_deltas differences between consecutive steps: the delta-space headline,
    the absolute-space agreement and the reference's rel RMSE (bf16 is 0 / 1 against itself)."""
    m = {k: v for k, v in entry["headline"].items() if isinstance(v, (int, float))}
    if "vs_bf16" in entry:
        m.update(bf16_rel_l2=entry["vs_bf16"]["rel_l2"], bf16_cosine=entry["vs_bf16"]["cosine"])
    else:
        m.update(bf16_rel_l2=0.0, bf16_cosine=1.0, delta_rel_l2=0.0, delta_cosine=1.0,
                 reference_rel_rmse_absolute=0.0)
    return m


def provenance(args, openpi) -> dict:
    import subprocess

    import torchao
    import transformers

    import ref_loader

    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, cwd=Path(__file__).parent).strip()
        except Exception:
            return None

    def version(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    out = {"args": vars(args), "atlas_torchao_commit": sh("git", "rev-parse", "HEAD"),
           "atlas_torchao_dirty": bool(sh("git", "status", "--porcelain")),
           "reference_pins": ref_loader.PINS, "openpi": pi0_model.openpi_info(openpi) if openpi else None,
           "config": args.config,
           "torch": torch.__version__, "torchao": torchao.__version__,
           "transformers": transformers.__version__, "numpy": np.__version__,
           "sentencepiece": version("sentencepiece"), "python": platform.python_version(),
           "host": platform.node(), "cpu": platform.processor() or platform.machine(),
           "device": str(args.device), "num_threads": torch.get_num_threads(), "gpu": gpu,
           "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())) if gpu else None}
    if args.checkpoint_dir:
        out["checkpoint"] = pi0_model.checkpoint_info(args.checkpoint_dir, hash_weights=not args.no_hash)
    ns = Path(args.norm_stats_dir or "") / "norm_stats.json"
    if args.norm_stats_dir and ns.is_file():
        out["norm_stats"] = {"path": str(ns), "sha256": pi0_model.sha256_file(ns, cache=False)}
    if args.tokenizer_path and Path(args.tokenizer_path).is_file():
        out["tokenizer"] = {"path": str(args.tokenizer_path), "source": pi0_model.TOKENIZER_SOURCE,
                            "sha256": pi0_model.sha256_file(args.tokenizer_path, cache=False)}
    return out


def load_frames(args, rt, cfg) -> list[dict]:
    if args.obs == "dummy":
        return pi0_model.dummy_observations(args.n_obs, args.seed)
    if args.obs == "mxu_dummy":
        return mxu_dummy_observations(args.n_obs, cfg, args.device, args.seed)
    if args.obs == "sim":
        return sim_observations(args.obs_dir, args.obs_file, cfg, args.checkpoint_dir,
                                args.norm_stats_dir, args.tokenizer_path, args.device)
    return client_resize(pi0_model.droid_observations(args.n_obs, cfg.action_horizon, args.frame_idx,
                                                      args.seed))


def client_resize(frames, size=224) -> list[dict]:
    """Camera images resized with padding to 224x224 on the client side, as openpi's DROID client
    does before sending (openpi examples/droid/main.py:120-123, openpi_client image_tools
    .resize_with_pad) and as the reference pre-resizes its sim observations (run_eval_mxu_io.py:
    189-190). Without it the 180x320 DROID frames reach the stubbed resize_with_pad_torch, which
    drops the batch dimension (preprocessing_pytorch.py:146 then fails)."""
    from openpi_client import image_tools

    for fr in frames:
        for key in pi0_model.DROID_CAMERAS:
            fr["obs"][key] = image_tools.resize_with_pad(fr["obs"][key], size, size)
        fr["meta"]["client_resize"] = f"openpi_client.image_tools.resize_with_pad {size}x{size}"
    return frames


def check_args(ap, args) -> None:
    for m in args.modes:
        try:
            require_mode_supported(m, args.device)
        except RuntimeError as e:
            ap.error(str(e))
    if not args.checkpoint_dir:
        ap.error("--checkpoint-dir is required (or $PI0_CHECKPOINT)")
    if OBS_LEVEL[args.obs] == "policy" or args.obs == "sim":
        if not args.tokenizer_path:
            ap.error(f"--obs {args.obs} needs --tokenizer-path (or $PALIGEMMA_TOKENIZER)")
    if args.obs == "sim" and not args.obs_dir:
        ap.error("--obs sim needs --obs-dir")
    if "bf16" not in args.modes:
        args.modes = ["bf16", *args.modes]        # every FP8 mode is compared against it
    else:
        args.modes = ["bf16", *[m for m in args.modes if m != "bf16"]]


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    check_args(ap, args)
    args.norm_stats_dir = args.norm_stats_dir or str(Path(args.checkpoint_dir) / "assets" / "droid")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.deterministic:
        make_deterministic()

    openpi = pi0_model.openpi_dir(args.openpi_dir)
    rt = pi0_model.runtime(openpi)
    cfg = pi0_model.model_config(rt, args.config)
    level = OBS_LEVEL[args.obs]
    frames = load_frames(args, rt, cfg)
    if args.steps != REFERENCE_STEPS:
        print(f"NOTE: {args.steps} flow steps instead of the reference's {REFERENCE_STEPS} "
              "(reduced_steps: true in the JSON)", flush=True)

    def build(mode):
        model = pi0_model.load_model(rt, args.config, args.checkpoint_dir, args.device)
        inv = inventory(model)
        layers = quantize(model, mode, args.skip, options=args) if mode != "bf16" else None
        policy = (pi0_model.make_policy(rt, model, args.device, args.config, args.norm_stats_dir,
                                        args.tokenizer_path) if level == "policy" else None)
        return model, inv, layers, policy

    out = Path(args.out)
    result = {"model": "pi0", "provenance": provenance(args, openpi), "determinism_state": determinism_state(),
              "data": data_record(args, frames, level, cfg.action_horizon), "new_decisions": NEW_DECISIONS}
    result.update(evaluate(args, frames, build, level, cfg.action_horizon, cfg.action_dim, out))
    write_result(args, result, out)


def data_record(args, frames, level, horizon) -> dict:
    return {"obs": args.obs, "level": level, "frames": [f["meta"] for f in frames], "n_obs": len(frames),
            "steps": args.steps, "reduced_steps": args.steps != REFERENCE_STEPS, "seed": args.seed,
            "action_horizon": horizon, "action_dims_reported": ACTION_DIMS,
            "action_space": ("robot units: absolute joint positions + gripper (policy shim)"
                             if level == "policy" else "model output: normalized joint deltas + gripper"),
            "ground_truth": frames[0]["gt"] is not None}


def write_result(args, result, out: Path) -> None:
    import ref_loader

    if args.rmse:
        import layer_rmse

        result["reference_provenance"] = layer_rmse.ref_provenance()
    result["reference_sources"] = ref_loader.provenance()
    if not result["valid"]:
        print("WARNING: at least one mode is invalid (cross-check failed); see modes.*.layers", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    print("wrote", out)


if __name__ == "__main__":
    main()
