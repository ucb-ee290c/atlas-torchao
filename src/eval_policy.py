"""Offline action accuracy of a LeRobot policy before and after FP8 quantization.

For each held-out frame the policy predicts one action chunk. Every mode sees the same frames and
the same per-frame RNG seed, so the diffusion / flow-matching noise is identical and differences
come from the numerics alone. Reported per mode:
  - error against the dataset's recorded actions (MSE, MAE; padded steps masked out), and
  - agreement with the unquantized (bf16) prediction (relative L2, cosine) for the FP8 modes, and
  - with --rmse, per-layer (isolated) and cumulative relative RMSE of every quantized layer against
    a clean pass on the same frames and noise (src/layer_rmse.py; the reference author's
    StatsTracker / ReferenceStore and CSV layout).

Modes:
  bf16        the checkpoint cast to bfloat16 (the reference)
  atlas       AtlasQConfig: Atlas FP8 E4M3 rules, power-of-two per-tensor scales
  atlas_hw    as atlas, with activations through the hardware cast (subnormals flushed)
  torchao     stock TorchAO Float8DynamicActivationFloat8WeightConfig, per-tensor
              (native FP8; needs compute capability >= 8.9, e.g. L40 / H100)

Example:
  python src/eval_policy.py --policy lerobot/smolvla_base \
      --dataset lerobot/svla_so100_pickplace --modes bf16 atlas atlas_hw torchao \
      --n-frames 200 --out out/smolvla_eval.json
"""

import argparse
import json
import platform
import random
import subprocess
import time
from pathlib import Path

import torch

from quant_modes import ALL_MODES, quantize, require_mode_supported



def load_policy(path, device, dtype):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    policy = policy.to(device=device, dtype=dtype).eval()
    # lerobot's SmolVLA integrates the flow in float32: the action expert output is cast to float32
    # before action_out_proj and x_t stays float32, so the action in/out projections and the
    # action-time MLP must stay float32 (bf16 there fails with a dtype mismatch). quantize() only
    # touches bf16 layers, so these remain as shipped.
    inner = getattr(policy, "model", None)
    for name in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        if hasattr(inner, name):
            getattr(inner, name).float()
    # Multi-Task DiT builds its timestep embedding in float32 (long * float32 frequencies); round it to
    # the policy dtype so the bf16 time MLP accepts it.
    for m in policy.modules():
        if type(m).__name__ == "SinusoidalPosEmb":
            m.register_forward_hook(lambda mod, inp, out: out.to(dtype))
    return policy, cfg


def camera_rename_map(cfg, dataset, override=None):
    """Map the dataset's camera keys onto the checkpoint's (e.g. top/wrist -> camera1/camera2).

    The saved pipeline expects the checkpoint's keys. If the dataset already uses them, nothing is
    renamed; otherwise cameras are paired in sorted order (use --rename-map to choose explicitly).
    """
    if override:
        return override
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    have = sorted(LeRobotDatasetMetadata(dataset).camera_keys)
    want = sorted(cfg.image_features)
    if set(want) & set(have):
        return {}
    return dict(zip(have, want))


def make_data(policy_cfg, dataset, n_test_episodes, n_frames, seed):
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(dataset)
    episodes = list(range(meta.total_episodes))
    random.Random(seed).shuffle(episodes)
    test_eps = sorted(episodes[:n_test_episodes])  # held out by episode, never by frame
    ds = LeRobotDataset(dataset, episodes=test_eps,
                        delta_timestamps=resolve_delta_timestamps(policy_cfg, meta))
    idx = sorted(random.Random(seed + 1).sample(range(len(ds)), min(n_frames, len(ds))))
    return ds, meta, test_eps, idx


def to_batch(sample, preprocess):
    batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else [v]) for k, v in sample.items()}
    return preprocess(batch)


def predict_chunk(policy, batch, seed, device):
    """One action chunk for a preprocessed offline batch.

    Policies with an observation history (n_obs_steps > 1, e.g. Multi-Task DiT) normally fill it from
    an online queue that predict_action_chunk reads; offline the history is already in the batch (the
    dataset was built with the config's observation delta timestamps), so call the model directly.
    """
    if getattr(policy.config, "n_obs_steps", 1) > 1 and hasattr(policy, "_generate_actions"):
        b = {k: v for k, v in batch.items() if k != "action"}
        return policy._generate_actions(policy._prepare_batch(b))
    return policy.predict_action_chunk(batch, **noise_kwargs(policy, seed, device))


def noise_kwargs(policy, seed, device):
    """Explicit initial noise for policies that take it (SmolVLA). Others sample from the global RNG, which run_mode seeds per frame."""
    import inspect

    if "noise" not in inspect.signature(policy.predict_action_chunk).parameters:
        return {}
    cfg = policy.config
    g = torch.Generator().manual_seed(seed)
    shape = (1, cfg.chunk_size, cfg.max_action_dim)
    return {"noise": torch.randn(shape, generator=g).to(device=device)}  # float32, as lerobot samples


@torch.no_grad()
def run_mode(policy, ds, idx, preprocess, postprocess, seed, device, before_frame=None):
    """before_frame(batch, frame_seed), if given, runs first on each frame (NEW; --rmse clean pass)."""
    preds, gts, pads, times = [], [], [], []
    for i in idx:
        sample = ds[i]
        batch = to_batch(sample, preprocess)
        if before_frame is not None:
            before_frame(batch, seed + i)
        policy.reset()
        torch.manual_seed(seed + i)  # same noise for every mode on this frame
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = predict_chunk(policy, batch, seed + i, device)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        chunk = postprocess(chunk).float().cpu()[0]  # (T, action_dim), dataset units
        # The chunk starts at the current step; the sample's action window starts at its first delta
        # index (e.g. -1 for a 2-step observation history), so skip to the delta-0 entry.
        off = list(policy.config.action_delta_indices).index(0)
        gt = sample["action"].float()[off:]
        pad = sample.get("action_is_pad", torch.zeros(sample["action"].shape[0], dtype=torch.bool))[off:]
        t = min(chunk.shape[0], gt.shape[0])
        preds.append(chunk[:t])
        gts.append(gt[:t])
        pads.append(pad[:t])
    return preds, gts, pads, times


def error_vs_gt(preds, gts, pads):
    se, ae, n = 0.0, 0.0, 0
    for p, g, pad in zip(preds, gts, pads):
        m = ~pad
        d = (p[m] - g[m])
        se += float((d * d).sum())
        ae += float(d.abs().sum())
        n += d.numel()
    return {"mse": se / n, "mae": ae / n, "elements": n}


def agreement(preds, ref):
    p = torch.cat([x.flatten() for x in preds])
    r = torch.cat([x.flatten() for x in ref])
    return {"rel_l2": float((p - r).norm() / r.norm()),
            "cosine": float(torch.nn.functional.cosine_similarity(p, r, dim=0)),
            "max_abs": float((p - r).abs().max())}


def provenance(args):
    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, cwd=Path(__file__).parent).strip()
        except Exception:
            return None

    import lerobot
    import torchao

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    cap = ".".join(map(str, torch.cuda.get_device_capability())) if gpu else None
    return {"args": vars(args), "atlas_torchao_commit": sh("git", "rev-parse", "HEAD"),
            "atlas_torchao_dirty": bool(sh("git", "status", "--porcelain")),
            "torch": torch.__version__, "torchao": torchao.__version__,
            "lerobot": getattr(lerobot, "__version__", None), "python": platform.python_version(),
            "host": platform.node(), "gpu": gpu, "compute_capability": cap}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--dataset", required=True, help="LeRobot dataset repo id")
    ap.add_argument("--modes", nargs="+", default=["bf16", "atlas", "atlas_hw"],
                    choices=list(ALL_MODES))
    ap.add_argument("--n-test-episodes", type=int, default=5)
    ap.add_argument("--n-frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip", nargs="*", default=[], help="fqn substrings left unquantized")
    ap.add_argument("--rename-map", type=json.loads, default=None,
                    help='JSON {dataset_key: policy_key}; default pairs cameras in sorted order')
    ap.add_argument("--rmse", action="store_true",
                    help="per-layer (isolated) and cumulative relative RMSE of every quantized layer "
                         "against a clean pass, written as the reference author's CSVs")
    ap.add_argument("--rmse-dir", default=None,
                    help="where the --rmse CSVs go (<dir>/<mode>/); default: <out stem>_rmse next to --out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    for m in args.modes:
        try:
            require_mode_supported(m, args.device)  # fail early, before any model download
        except RuntimeError as e:
            ap.error(str(e))
    if "bf16" not in args.modes:
        args.modes = ["bf16", *args.modes]  # every FP8 mode is compared against it

    from lerobot.policies.factory import make_pre_post_processors

    result = {"provenance": provenance(args), "modes": {}}
    ref = None
    clean = None
    rmse_dir = Path(args.rmse_dir) if args.rmse_dir else Path(args.out).with_name(Path(args.out).stem + "_rmse")
    for mode in args.modes:
        policy, cfg = load_policy(args.policy, args.device, torch.bfloat16)
        rename = camera_rename_map(cfg, args.dataset, args.rename_map)
        # The saved pipelines pin device='cuda'; override it, and cast float inputs to the policy dtype.
        preprocess, postprocess = make_pre_post_processors(
            cfg, pretrained_path=args.policy,
            preprocessor_overrides={
                "device_processor": {"device": args.device, "float_dtype": "bfloat16"},
                "rename_observations_processor": {"rename_map": rename}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        if ref is None:
            ds, meta, test_eps, idx = make_data(cfg, args.dataset, args.n_test_episodes,
                                                args.n_frames, args.seed)
            result["data"] = {"dataset": args.dataset, "test_episodes": test_eps,
                              "frames": len(idx), "fps": meta.fps}
        layers = quantize(policy, mode, args.skip) if mode != "bf16" else None
        rm = before = None
        if args.rmse and mode != "bf16":
            import layer_rmse

            if clean is None:
                clean, _ = load_policy(args.policy, args.device, torch.bfloat16)
            rm = layer_rmse.LayerRMSE(clean, policy)

            def before(batch, frame_seed):
                def run_clean():
                    clean.reset()
                    torch.manual_seed(frame_seed)  # the same noise as the quantized pass
                    predict_chunk(clean, dict(batch), frame_seed, args.device)
                rm.capture(run_clean)
        preds, gts, pads, times = run_mode(policy, ds, idx, preprocess, postprocess,
                                           args.seed, args.device, before_frame=before)
        entry = {"layers": layers, "vs_ground_truth": error_vs_gt(preds, gts, pads),
                 "latency_s": {"median": sorted(times)[len(times) // 2], "mean": sum(times) / len(times)}}
        if rm is not None:
            rm.close()
            entry["rmse"] = {**rm.summary(), "csv": rm.write_csvs(rmse_dir / mode)}
        if mode == "bf16":
            ref = preds
        else:
            entry["vs_bf16"] = agreement(preds, ref)
        result["modes"][mode] = entry
        print(mode, json.dumps(entry, default=str)[:400], flush=True)
        del policy
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    if args.rmse:
        import layer_rmse

        result["reference_provenance"] = layer_rmse.ref_provenance()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
