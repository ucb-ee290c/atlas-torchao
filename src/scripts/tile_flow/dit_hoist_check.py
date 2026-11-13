"""Eager check of the hoisted Multi-Task DiT loop on real frames (docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).

For an export (quantized policy rebuilt with load_quantized, reference emulation, CPU):
  rows:    the modulate rows for all scheduler timesteps computed in calls of k timesteps, for each --steps-per-call k,
           against k = 1 (each step's rows alone, the shapes of the per-step graph): differing elements per hoisted site.
  frames:  for --n-frames frames of the evaluation protocol (eval_policy make_data / processors, same seeds per frame),
           policy._generate_actions (stock: encode, conditional_sample with the 100 noise-predictor calls, slice) against
           the hoisted loop (encode, DiTModulate once, DiTStepHoisted per step, the same scheduler.step and RNG draws),
           byte for byte, for each k that passes the rows check (or every k with --all-k).
  bytes:   quantized weight bytes (E4M3 codes + int32 exponents + BF16 bias) read per step and per chunk, before (every
           noise-predictor site per step) and after (hoisted sites once per modulate call, the rest per step).
usage (PYTHONPATH=<worktree>/src, atlas-lerobot env, CUDA_VISIBLE_DEVICES=""):
  dit_hoist_check.py --export-dir EXPORT --out OUT.json [--n-frames 4] [--n-test-episodes 32] [--steps-per-call 1 25 100]"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy_fx_capture_phases as cap  # noqa: E402


def _layer(policy, path):
    return policy.get_submodule(path)


def weight_bytes(layer):
    """Bytes a call streams for one quantized Linear: codes (1 B each), exponents and bias as stored."""
    n = 0
    for name in ("weight_codes", "weight_scale_exp", "bias"):
        t = getattr(layer, name, None)
        if t is not None:
            n += t.numel() * t.element_size()
    return n


def byte_table(policy, steps, steps_per_call):
    npred = [p for p, m in policy.named_modules() if p.startswith("noise_predictor.") and type(m).__name__ == "RefQuantLinear"]
    hoisted = set(cap.hoisted_paths(policy.noise_predictor))
    per = {p: weight_bytes(_layer(policy, p)) for p in npred}
    before_step = sum(per.values())
    after_step = sum(v for p, v in per.items() if p not in hoisted)
    hoisted_bytes = sum(v for p, v in per.items() if p in hoisted)
    calls = -(-steps // steps_per_call)
    mib = 1 << 20
    return {"steps": steps, "steps_per_call": steps_per_call, "modulate_calls_per_site": calls,
            "noise_predictor_sites": len(npred), "hoisted_sites": sorted(hoisted),
            "before": {"bytes_per_step": before_step, "bytes_per_chunk": before_step * steps,
                       "MiB_per_step": before_step / mib, "MiB_per_chunk": before_step * steps / mib},
            "after": {"step_bytes_per_step": after_step, "modulate_bytes_per_chunk": hoisted_bytes * calls,
                      "bytes_per_chunk": after_step * steps + hoisted_bytes * calls,
                      "MiB_per_step": after_step / mib,
                      "MiB_per_chunk": (after_step * steps + hoisted_bytes * calls) / mib},
            "hoisted_bytes_once": hoisted_bytes, "hoisted_MiB_once": hoisted_bytes / mib,
            "note": "quantized Linear payloads of the noise predictor (E4M3 codes, int32 exponents, BF16 bias); encode not "
                    "included; read from the rebuilt policy's buffers"}


def rows_check(policy, cond, timesteps, ks):
    """{k: {site: differing elements}} of the modulation computed k timesteps per call against one per call."""
    out = {}
    with torch.no_grad():
        ref = cap.DiTModulate(policy, 1)(cond, timesteps)
        for k in ks:
            got = cap.DiTModulate(policy, k)(cond, timesteps)
            out[k] = {"differing_elements": int((got != ref).sum()), "elements": got.numel(),
                      "per_layer": [int((got[i] != ref[i]).sum()) for i in range(got.shape[0])],
                      "max_abs": float((got.float() - ref.float()).abs().max())}
    return out


def hoisted_generate(policy, batch, steps_per_call):
    """policy._generate_actions with the hoisted loop: same encode, same initial noise and scheduler.step draws."""
    import lerobot.policies.multi_task_dit.modeling_multi_task_dit as mt  # noqa: F401

    obj, cfg = policy.objective, policy.config
    cond = policy.observation_encoder.encode(batch)
    sched = obj.noise_scheduler
    sched.set_timesteps(obj.num_inference_steps)
    dtype = next(policy.noise_predictor.parameters()).dtype
    x = torch.randn(size=(cond.shape[0], obj.horizon, obj.action_dim), dtype=dtype, device=cond.device)
    mod = cap.DiTModulate(policy, steps_per_call)(cond, sched.timesteps.to(torch.long))
    step = cap.DiTStepHoisted(policy)
    for i, t in enumerate(sched.timesteps):
        x = sched.step(step(x, mod[:, i].contiguous()), t, x).prev_sample
    start = cfg.n_obs_steps - 1
    return x[:, start:start + cfg.n_action_steps]


def write_cases(policy, manifest, batches, timesteps, out, seed, frames):
    """Capture cases in the single-step bundle layout: inputs (state, images, input_ids, attention_mask, noisy_actions,
    timestep), output the stock noise_predictor's prediction; case j uses frame j and a timestep spread over the schedule.
    Each case records its frame and the sampling seed the evaluation uses for it (seed + frame, eval_policy.run_mode)."""
    import hashlib

    from huggingface_hub import snapshot_download
    from lerobot.utils.constants import OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
    from safetensors.torch import save_file

    (out / "fx").mkdir(parents=True, exist_ok=True)
    tensors, parity = {}, {"cases": []}
    g = torch.Generator().manual_seed(seed)
    obj = policy.objective
    with torch.no_grad():
        for j, b in enumerate(batches):
            noisy = torch.randn(1, obj.horizon, obj.action_dim, generator=g).to(torch.bfloat16)
            t = timesteps[(j * 37) % len(timesteps)].reshape(1).clone()
            ins = [b[OBS_STATE], b[OBS_IMAGES], b[OBS_LANGUAGE_TOKENS], b[OBS_LANGUAGE_ATTENTION_MASK], noisy, t]
            cond = policy.observation_encoder.encode(cap._batch(*ins[:4]))
            pred = policy.noise_predictor(noisy, t, conditioning_vec=cond)
            keys = [f"case_{j}.input_{i}" for i in range(len(ins))]
            tensors.update({k: v.detach().cpu().contiguous().clone() for k, v in zip(keys, ins)})
            tensors[f"case_{j}.output_0"] = pred.contiguous()
            parity["cases"].append({"inputs": keys, "outputs": [f"case_{j}.output_0"], "frame": frames[j],
                                    "seed": seed + frames[j]})
    save_file(tensors, str(out / "fx" / "fixtures.safetensors"))
    (out / "fx" / "parity.json").write_text(json.dumps(parity, indent=1))
    snap = Path(snapshot_download(manifest["model"]["id"], local_files_only=True))
    bm = {"note": "capture cases only (dit_hoist_check.py --write-cases): no single-step graph",
          "model": {"id": manifest["model"]["id"], "checkpoint": {
              "snapshot": str(snap), "config_json_sha256": hashlib.sha256((snap / "config.json").read_bytes()).hexdigest()}}}
    (out / "bundle_manifest.json").write_text(json.dumps(bm, indent=1))
    return {"dir": str(out), "cases": len(batches)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-frames", type=int, default=4, help="frame selection of the evaluation protocol (eval_policy make_data)")
    ap.add_argument("--max-frames", type=int, default=None, help="check only the first N frames of that selection")
    ap.add_argument("--n-test-episodes", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps-per-call", type=int, nargs="+", default=[1, 2, 4, 5, 10, 20, 25, 50, 100])
    ap.add_argument("--all-k", action="store_true", help="run the frames check for every k, not only the row-exact ones")
    ap.add_argument("--rows-only", action="store_true")
    ap.add_argument("--write-cases", metavar="DIR",
                    help="also write DIR/fx/{fixtures.safetensors,parity.json} and DIR/bundle_manifest.json: capture cases in "
                         "the single-step bundle layout (obs, noisy actions, timestep -> stock noise prediction) for "
                         "make_loop_manifest.py / policy_fx_capture_phases.py when no captured single-step bundle exists")
    a = ap.parse_args(argv)
    import eval_policy

    t0 = time.time()
    export_dir = Path(a.export_dir)
    manifest = json.loads((export_dir / "manifest.json").read_text())
    policy, cfg = cap.load_dit(export_dir)
    policy_json = json.loads((export_dir / "policy.json").read_text())
    res = {"export_dir": str(export_dir), "recipe": manifest["recipe"].get("algo"),
           "export_argv": manifest["provenance"].get("argv"), "torch_threads": torch.get_num_threads(),
           "device": "cpu"}
    sched = policy.objective.noise_scheduler
    sched.set_timesteps(policy.objective.num_inference_steps)
    timesteps = sched.timesteps.to(torch.long)
    spanning = cap.check_hoist_rules(policy_json, [p for p in cap.hoisted_paths(policy.noise_predictor)], 1)
    res["row_spanning_hoisted_sites"] = spanning
    ks = [k for k in a.steps_per_call if k == 1 or not spanning]
    res["steps_per_call_refused"] = [k for k in a.steps_per_call if k not in ks]
    res["bytes"] = {k: byte_table(policy, len(timesteps), k) for k in ks}

    args = eval_policy.build_parser().parse_args(
        ["--policy", manifest["model"]["id"], "--dataset", manifest["provenance"]["lerobot_dataset"], "--seed", str(a.seed),
         "--n-test-episodes", str(a.n_test_episodes), "--n-frames", str(a.n_frames), "--modes", "ref_conv",
         "--norm-stats", "dataset", "--device", "cpu", "--out", "unused"])
    rename = eval_policy.camera_rename_map(cfg, args.dataset, args.rename_map)
    ds, _, test_eps, idx = eval_policy.make_data(cfg, args.dataset, args.n_test_episodes, args.n_frames, args.seed)
    norm, norm_stats = eval_policy.setup_normalization(args, cfg, rename, ds, idx)
    preprocess, _ = eval_policy.build_processors(cfg, args, rename, norm_stats)
    if a.max_frames is not None:
        idx = idx[:a.max_frames]
    res["data"] = {"dataset": args.dataset, "test_episodes": test_eps, "selection_n_frames": a.n_frames, "frames": idx,
                   "normalization": norm["status"]}
    batches = []
    for i in idx:
        b = eval_policy.to_batch(ds[i], preprocess)
        batches.append(policy._prepare_batch({k: v for k, v in b.items() if k != "action"}))
    with torch.no_grad():
        cond0 = policy.observation_encoder.encode(batches[0])
        res["rows"] = rows_check(policy, cond0, timesteps, ks)
    if a.write_cases:
        res["cases_written"] = write_cases(policy, manifest, batches, timesteps, Path(a.write_cases), a.seed, idx)
    exact_k = [k for k in ks if res["rows"][k]["differing_elements"] == 0]
    res["row_exact_steps_per_call"] = exact_k
    Path(a.out).write_text(json.dumps(res, indent=1))
    if a.rows_only:
        return 0
    frames = []
    for j, (i, batch) in enumerate(zip(idx, batches)):
        row = {"frame": i}
        with torch.no_grad():
            policy.reset()
            with eval_policy.seeded_rng(a.seed + i, "cpu"):
                stock = policy._generate_actions(batch)
            for k in (ks if a.all_k else exact_k):
                with eval_policy.seeded_rng(a.seed + i, "cpu"):
                    got = hoisted_generate(policy, batch, k)
                row[f"k{k}"] = {"bytes_equal": bool(torch.equal(got, stock) and got.dtype == stock.dtype),
                                "differing_elements": int((got != stock).sum()), "elements": got.numel(),
                                "max_abs": float((got.float() - stock.float()).abs().max())}
        frames.append(row)
        res["frames"] = frames
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps(row), flush=True)
    res["all_frames_bytes_equal_for_row_exact_k"] = all(f[f"k{k}"]["bytes_equal"] for f in frames for k in exact_k)
    res["seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("row_exact_steps_per_call", "all_frames_bytes_equal_for_row_exact_k", "seconds")}))
    return 0 if res["all_frames_bytes_equal_for_row_exact_k"] else 1


if __name__ == "__main__":
    sys.exit(main())
