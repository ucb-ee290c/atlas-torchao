"""Per-phase FX capture for SmolVLA (LeRobot) on real LIBERO frames: the prefix pass (cameras, instruction and state -> the
post-RoPE K / V of every VLM text layer) and one denoising step reading them (x_t, t -> v_t), as two graphs with the host owning
the Euler loop (loop_manifest.json from make_loop_manifest.py smolvla; docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).

  prefix:       SmolVLAPrefix (image_0, image_1, lang_tokens, state, mask, positions) -> (k_0, v_0, ..., k_15, v_15)
  denoise_step: SmolVLADenoiseStep (x_t, time, mask, positions, k_0, v_0, ..., k_15, v_15) -> v_t
  (positions: the position ids, or with --host-rope the host RoPE tables; fx_export.smolvla_prefix_inputs /
  smolvla_step_host_inputs compute them and the masks on the host. With --host-text-embeddings the prefix takes
  lerobot's embed_language_tokens(lang_tokens) instead of lang_tokens; with --host-time-embedding the step takes the
  flow-timestep embedding right after time, fx_export.smolvla_time_inputs)

Capture cases: --frames LIBERO frames of the protocol the export was made with (eval_policy.py --dataset lerobot/libero
--n-test-episodes 32 --n-frames 256 --seed 0: test episodes by seed, frames by seed + 1), evenly spaced over those 256,
preprocessed by the policy's own processors with the dataset's normalization stats; the noise of a frame is the one
eval_policy draws for it (randn with seed + frame index). The step case of frame i takes x_t and t of that frame's eager
trajectory at Euler step round(i * (num_steps - 1) / (frames - 1)), so the step cases span the time grid. The dataset is read
from its local snapshot (lerobot's hub cache), so no Hub access is needed (eval_policy.make_data resolves the dataset
revision on the Hub and fails with HF_HUB_OFFLINE=1).

The bundle (--out) holds export/ (hard links), fx/ (capture_fx), loop_manifest.json (copy) and host_cases.safetensors /
host_cases.json: the policy-level observation of every case (images, image masks, tokens, language mask, state, noise),
from which compose_phases.py smolvla rebuilds the host inputs and the eager reference.

usage (PYTHONPATH=<worktree>/src; OMP_NUM_THREADS=1, the thread count the fixtures replay with):
  smolvla_phase_capture.py prefix|denoise_step --export-dir EXPORT --loop-manifest LM --out BUNDLE [--frames 4] [--host-rope]
        [--dry]"""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

DATASET, N_TEST_EPISODES, N_FRAMES, SEED = "lerobot/libero", 32, 256, 0
OBS_KEYS = ("lang_tokens", "lang_masks", "state", "noise")


def dataset_root(repo_id):
    """The local snapshot of `repo_id` in lerobot's hub cache: the commit its refs/<CODEBASE_VERSION> names, the revision
    LeRobotDatasetMetadata resolves on the Hub for this lerobot version."""
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.utils.constants import HF_LEROBOT_HUB_CACHE

    repo = HF_LEROBOT_HUB_CACHE / f"datasets--{repo_id.replace('/', '--')}"
    ref = (repo / "refs" / CODEBASE_VERSION).read_text().strip()
    root = repo / "snapshots" / ref
    if not (root / "meta" / "info.json").is_file():
        raise SystemExit(f"no local snapshot of {repo_id} at {root}")
    return root


def load_policy(export_dir, device="cpu"):
    """(lerobot policy rebuilt as the export's reference modules (ref_conv), its config, policy.json, manifest)."""
    import eval_policy
    import exporter.load_quantized as load_quantized

    _, policy_json, manifest = load_quantized.read_export(export_dir)
    model_id = manifest["model"]["id"]
    policy, cfg = eval_policy.load_policy(model_id, device, torch.bfloat16)
    load_quantized.load_quantized(model_id, export_dir, device, build=lambda: policy, mode="ref_conv", options={})
    return policy, cfg, policy_json, manifest


def load_bf16_policy(model_id, device="cpu"):
    import eval_policy

    return eval_policy.load_policy(model_id, device, torch.bfloat16)


def libero_frames(cfg, n_frames, root=None):
    """(dataset, rename map, normalization stats, frame indices, record): the eval_policy protocol's 256 frames, of which
    n_frames evenly spaced ones are returned."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.processor import rename_stats

    root = Path(root) if root else dataset_root(DATASET)
    meta = LeRobotDatasetMetadata(DATASET, root=root)
    have, want = sorted(meta.camera_keys), sorted(cfg.image_features)
    rename = {} if set(want) & set(have) else dict(zip(have, want))         # eval_policy.camera_rename_map
    episodes = list(range(meta.total_episodes))
    random.Random(SEED).shuffle(episodes)                                    # eval_policy.make_data
    test_eps = sorted(episodes[:N_TEST_EPISODES])
    ds = LeRobotDataset(DATASET, root=root, episodes=test_eps, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    idx = sorted(random.Random(SEED + 1).sample(range(len(ds)), min(N_FRAMES, len(ds))))
    picks = [idx[round(i * (len(idx) - 1) / max(1, n_frames - 1))] for i in range(n_frames)]
    record = {"dataset": DATASET, "root": str(root), "protocol": f"eval_policy.make_data(seed={SEED}, n_test_episodes="
              f"{N_TEST_EPISODES}, n_frames={N_FRAMES}), {n_frames} of them evenly spaced", "test_episodes": test_eps,
              "frames": picks, "rename_map": rename}
    return ds, rename, rename_stats(meta.stats, rename), picks, record


def observations(policy, cfg, model_id, n_frames, root=None, device="cpu"):
    """[{images, img_masks, lang_tokens, lang_masks, state, noise, meta}] of n_frames LIBERO frames, prepared as
    SmolVLAPolicy._get_action_chunk prepares them for sample_actions."""
    import eval_policy

    ds, rename, stats, picks, record = libero_frames(cfg, n_frames, root)
    pre, _ = eval_policy.build_processors(cfg, SimpleNamespace(device=device, dtype="bfloat16", policy=model_id), rename,
                                          stats)
    out = []
    for frame in picks:
        sample = ds[frame]
        batch = eval_policy.to_batch(sample, pre)
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        noise = torch.randn((1, cfg.chunk_size, cfg.max_action_dim),
                            generator=torch.Generator().manual_seed(SEED + frame)).to(device)   # eval_policy.noise_kwargs
        lang_masks = batch["observation.language.attention_mask"]
        out.append({"images": [i.contiguous() for i in images], "img_masks": img_masks,
                    "lang_tokens": batch["observation.language.tokens"].contiguous(), "lang_masks": lang_masks.contiguous(),
                    "state": state.contiguous(), "noise": noise,
                    "meta": {"frame": frame, "episode": int(sample["episode_index"]), "task": sample["task"].strip(),
                             "valid_language_tokens": int(lang_masks.sum()), "language_tokens": int(lang_masks.shape[1])}})
    return out, record


def save_host_cases(path, obs, record, steps=None):
    from safetensors.torch import save_file

    tensors = {}
    for i, o in enumerate(obs):
        for j, (image, mask) in enumerate(zip(o["images"], o["img_masks"])):
            tensors[f"case_{i}.image_{j}"] = image.cpu().contiguous()
            tensors[f"case_{i}.img_mask_{j}"] = mask.cpu().contiguous()
        for key in OBS_KEYS:
            tensors[f"case_{i}.{key}"] = o[key].cpu().contiguous()
    save_file(tensors, str(path / "host_cases.safetensors"))
    meta = {"schema": "atlas_torchao.smolvla_host_cases.v1", "cases": len(obs), "n_images": len(obs[0]["images"]),
            "observations": [o["meta"] for o in obs], "frames": record,
            "step_case_euler_steps": steps,
            "keys": "case_<i>.image_<j>, case_<i>.img_mask_<j>, case_<i>." + ", case_<i>.".join(OBS_KEYS)}
    (path / "host_cases.json").write_text(json.dumps(meta, indent=1) + "\n")


def load_host_cases(path):
    """[(images, img_masks, lang_tokens, lang_masks, state, noise)], host_cases.json."""
    from safetensors.torch import load_file

    meta = json.loads((Path(path) / "host_cases.json").read_text())
    t = load_file(str(Path(path) / "host_cases.safetensors"))
    cases = []
    for i in range(meta["cases"]):
        n = meta["n_images"]
        cases.append(([t[f"case_{i}.image_{j}"] for j in range(n)], [t[f"case_{i}.img_mask_{j}"] for j in range(n)],
                      *(t[f"case_{i}.{k}"] for k in OBS_KEYS)))
    return cases, meta


def euler_step_of_case(i, n_cases, num_steps):
    return round(i * (num_steps - 1) / max(1, n_cases - 1))


def trajectory_points(policy, prefix_cases, pads, kvs, noises, times, dt, step_indices, host_rope,
                      host_time_embedding=False):
    """[(x_t, time)] at the given Euler step of each case, from the eager SmolVLADenoiseStep loop."""
    import exporter.fx_export as fx_export

    geometry = fx_export.smolvla_geometry(policy)
    step = fx_export.SmolVLADenoiseStep(policy, host_rope=host_rope, host_time_embedding=host_time_embedding).eval()
    points = []
    try:
        with torch.no_grad():
            for pad, kv, noise, k in zip(pads, kvs, noises, step_indices):
                host = fx_export.smolvla_step_host_inputs(geometry, pad, host_rope)
                x = noise
                for t in times[:k]:
                    tt = torch.full((x.shape[0],), t, dtype=torch.float32)
                    x = x + dt * step(*fx_export.smolvla_step_inputs(geometry, x, tt, host, kv, host_time_embedding))
                points.append((x.contiguous(), torch.full((x.shape[0],), times[k], dtype=torch.float32)))
    finally:
        step.restore()
    return points


def spec_of(tensor):
    return {"shape": list(tensor.shape), "dtype": {torch.bfloat16: "BF16", torch.float32: "F32", torch.int64: "I64",
                                                   torch.bool: "BOOL"}[tensor.dtype]}


def check_against_manifest(lm, phase, case, outputs=None):
    """The case's inputs (and the prefix outputs) against the loop manifest's input / output specs, in order."""
    graph = lm["graphs"]["prefix" if phase == "prefix" else "step"]
    want = []
    for name, spec in graph["inputs"].items():
        if name == "kv_cache":
            want += [{"shape": spec["per_tensor_shape"], "dtype": spec["dtype"]}] * spec["tensors"]
        else:
            want.append({"shape": spec["shape"], "dtype": spec["dtype"]})
    got = [spec_of(t) for t in case]
    if got != want:
        raise SystemExit(f"{phase} case inputs differ from the loop manifest: {got} vs {want}")
    if outputs is not None:
        kv = graph["outputs"]["kv_cache"]
        if [spec_of(t) for t in outputs] != [{"shape": kv["per_tensor_shape"], "dtype": kv["dtype"]}] * kv["tensors"]:
            raise SystemExit("prefix outputs differ from the loop manifest's kv_cache")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("prefix", "denoise_step"))
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--loop-manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--host-rope", action="store_true")
    ap.add_argument("--host-text-embeddings", action="store_true", help="prefix: host language-token embeddings")
    ap.add_argument("--host-time-embedding", action="store_true", help="denoise_step: host flow-timestep embedding")
    ap.add_argument("--dataset-root", help="local snapshot of lerobot/libero (default: lerobot's hub cache)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    import exporter.fx_export as fx_export
    import exporter.quant_export as quant_export

    export_dir, out, lm_path = Path(a.export_dir), Path(a.out), Path(a.loop_manifest)
    lm = json.loads(lm_path.read_text())
    flags = {"host_rope": a.host_rope, "host_text_embeddings": a.host_text_embeddings,
             "host_time_embedding": a.host_time_embedding}
    if lm.get("model") != "smolvla" or any(lm["policy"].get(k, False) != v for k, v in flags.items()):
        raise SystemExit(f"{lm_path} is not a SmolVLA loop manifest with {flags}")
    policy, cfg, policy_json, manifest = load_policy(export_dir, a.device)
    obs, record = observations(policy, cfg, manifest["model"]["id"], a.frames, a.dataset_root, a.device)
    n_images = len(obs[0]["images"])
    geometry = fx_export.smolvla_geometry(policy)
    embed = policy.model.vlm_with_expert.embed_language_tokens if a.host_text_embeddings else None   # lerobot's own
    prefix_cases, pads = [], []
    for o in obs:
        case, pad = fx_export.smolvla_prefix_inputs(geometry, o["images"], o["img_masks"], o["lang_tokens"],
                                                    o["lang_masks"], o["state"], a.host_rope, embed)
        check_against_manifest(lm, "prefix", case)
        prefix_cases.append(case)
        pads.append(pad)
    steps = None
    if a.phase == "prefix":
        cases = prefix_cases
        boundary = fx_export.SmolVLAPrefix(policy, n_images, host_rope=a.host_rope,
                                           host_text_embeddings=a.host_text_embeddings).eval()
    else:
        kvs = fx_export.smolvla_prefix_outputs(policy, n_images, prefix_cases, a.host_rope, a.host_text_embeddings)
        for kv in kvs:
            check_against_manifest(lm, "prefix", prefix_cases[0], kv)
        steps = [euler_step_of_case(i, len(obs), lm["denoising_steps"]) for i in range(len(obs))]
        points = trajectory_points(policy, prefix_cases, pads, kvs, [o["noise"] for o in obs], lm["euler"]["times"],
                                   lm["euler"]["dt"], steps, a.host_rope, a.host_time_embedding)
        cases = [fx_export.smolvla_step_inputs(geometry, x, t, fx_export.smolvla_step_host_inputs(geometry, pad, a.host_rope),
                                               tuple(k.contiguous() for k in kv), a.host_time_embedding)
                 for (x, t), pad, kv in zip(points, pads, kvs)]
        for case in cases:
            check_against_manifest(lm, "denoise_step", case)
        boundary = fx_export.SmolVLADenoiseStep(policy, host_rope=a.host_rope,
                                                host_time_embedding=a.host_time_embedding).eval()
    try:
        if a.dry:
            with torch.no_grad():
                for c in cases:
                    o = boundary(*c)
                    o = o if isinstance(o, tuple) else (o,)
                    print("dry ok", a.phase, len(c), "inputs ->", len(o), "outputs", [tuple(x.shape) for x in o[:2]],
                          flush=True)
            return 0
        out.mkdir(parents=True, exist_ok=True)
        (out / "loop_manifest.json").write_bytes(lm_path.read_bytes())
        save_host_cases(out, obs, record, steps)
        fx_export.link_export(export_dir, out / "export")      # real files: the backend refuses symlinked export paths
        here = Path(__file__).resolve()
        report = fx_export.capture_fx(
            boundary, cases, out / "fx", policy=policy_json, strict=True, native_fp8="uint8",
            provenance={"export_dir": str(export_dir.resolve()),
                        "export_manifest_sha256": quant_export.sha256_file(export_dir / "manifest.json"),
                        "export_provenance": manifest.get("provenance", {}), "export_recipe": manifest.get("recipe", {}),
                        "architecture_locator": manifest["model"]["id"], "boundary": f"smolvla_{a.phase}",
                        "phase": a.phase, **flags, "n_images": n_images,
                        "boundary_driver": str(here), "boundary_driver_sha256": hashlib.sha256(here.read_bytes()).hexdigest(),
                        "boundary_note": "per-phase boundary; the host owns the Euler loop (loop_manifest)",
                        "loop_manifest": {"file": "../loop_manifest.json",
                                          "sha256": hashlib.sha256(lm_path.read_bytes()).hexdigest(),
                                          "denoising_steps": lm["denoising_steps"],
                                          "calls_per_action_chunk": lm["graphs"]["prefix" if a.phase == "prefix" else
                                                                                 "step"]["calls_per_action_chunk"]},
                        "capture_cases": {**record, "observations": [o["meta"] for o in obs],
                                          "step_case_euler_steps": steps, "host_cases": "../host_cases.safetensors"},
                        "config_sha256": hashlib.sha256(json.dumps(__import__("dataclasses").asdict(cfg), sort_keys=True,
                                                                   default=str).encode()).hexdigest(),
                        "capture_source": fx_export._capture_source()})
        print(json.dumps({k: v for k, v in report.items() if k not in ("provenance",)}, indent=1, default=str)[:3000])
    finally:
        boundary.restore()
    return 0


if __name__ == "__main__":
    sys.exit(main())
