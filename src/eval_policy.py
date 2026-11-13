"""Offline action accuracy of a LeRobot policy before and after FP8 quantization.

For each held-out frame the policy predicts one action chunk. Every mode sees the same frames and
the same per-frame RNG seed, so the diffusion / flow-matching noise is identical and differences
come from the numerics alone. Each mode is run twice and must reproduce itself bit for bit; a mode
that does not is marked "deterministic": false. Reported per mode:
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

Normalization (src/loaders/policy_norm.py): --norm-stats checkpoint (default) uses the processors as saved;
--norm-stats dataset (NEW) overrides their stats with the evaluated dataset's, as a lerobot fine-tune
from the checkpoint would start. Either way a guard refuses processors that would leave a normalized
feature unnormalized (lerobot/smolvla_base's saved processors do: SmolVLA results made before this
protocol saw raw joint degrees and their ground-truth errors are invalid). --allow-unnormalized
reproduces that historical behaviour, recorded as normalization: legacy-unverified.

--dtype float32 loads the baseline in float32, as the reference author serves SmolVLA
(smolVLA-quant smolvla_inout/serve_smolvla.py:68-69).

Example:
  python src/eval_policy.py --policy lerobot/smolvla_base \
      --dataset lerobot/svla_so100_pickplace --modes bf16 atlas atlas_hw torchao \
      --norm-stats dataset --n-frames 200 --out out/smolvla_eval.json
"""

import argparse
import contextlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import torch

import algos
import loaders.ckpt_keys as ckpt_keys
import loaders.policy_norm as policy_norm
from quant_modes import (ALL_MODES, add_profile_args, add_subset_compare_arg, finalize_report, ladder_deltas,
                         ladder_record, quantize, require_mode_supported, subset_compare_modes)



def load_policy(path, device, dtype):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
    # NEW: a non-strict load leaves keys a newer library nests one level deeper (e.g. the DiT's
    # fine-tuned CLIP) at the constructor's weights; repair the nesting or refuse (src/loaders/ckpt_keys.py).
    key_repair = ckpt_keys.repair_checkpoint_keys(policy, path)
    policy = policy.to(device=device, dtype=dtype).eval()
    policy.checkpoint_key_repair = key_repair
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


def load_config(path, device):
    """The checkpoint's policy config, without its weights."""
    import lerobot.policies.factory  # noqa: F401  (registers the policy config classes)
    from lerobot.configs.policies import PreTrainedConfig

    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device = device
    return cfg


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


def build_processors(cfg, args, rename, norm_stats=None):
    """The saved pre/post pipelines, with the device / dtype / rename overrides and, for
    --norm-stats dataset, lerobot_train's normalizer / unnormalizer stats overrides."""
    from lerobot.policies.factory import make_pre_post_processors

    # The saved pipelines pin device='cuda'; override it, and cast float inputs to the policy dtype.
    pre = {"device_processor": {"device": args.device, "float_dtype": args.dtype},
           "rename_observations_processor": {"rename_map": rename}}
    post = {"device_processor": {"device": "cpu"}}
    if norm_stats is not None:
        npre, npost = policy_norm.overrides(cfg, norm_stats)
        pre, post = policy_norm.merge_overrides(pre, npre), policy_norm.merge_overrides(post, npost)
    return make_pre_post_processors(cfg, pretrained_path=args.policy,
                                    preprocessor_overrides=pre, postprocessor_overrides=post)


def setup_normalization(args, cfg, rename, ds, idx):
    """Build the processors once, guard them, probe them on one evaluated frame and return
    (norm record, dataset stats or None). Raises policy_norm.NormalizationError on refusal."""
    norm_stats = None
    if args.norm_stats == "dataset":
        norm_stats, source = policy_norm.dataset_stats(args.dataset, rename)
    else:
        source = {"kind": "checkpoint", "id": args.policy}
    preprocess, postprocess = build_processors(cfg, args, rename, norm_stats)
    guard = policy_norm.check(preprocess, postprocess, cfg)
    status = policy_norm.require(guard, args.norm_stats, args.allow_unnormalized)
    act = cfg.output_features.get("action")
    probe = policy_norm.probe(preprocess, postprocess, ds[idx[0]], to_batch, guard["required"], rename,
                              tuple(act.shape) if act is not None else None, seed=args.seed)
    if not probe["ok"] and status == policy_norm.VERIFIED:
        raise policy_norm.NormalizationError(f"normalization probe failed: {json.dumps(probe)}")
    norm = {"protocol": args.norm_stats, "status": status, "stats_source": source,
            "stats_sha256": policy_norm.combined_sha256(guard),
            "transform_sha256": policy_norm.chain_sha256(guard),
            "guard": guard, "probe": probe,
            "new": ("--norm-stats dataset is a NEW zero-shot evaluation protocol for a pretraining "
                    "checkpoint (the stats a fine-tune would start from); the guard, the probe and "
                    "--allow-unnormalized are NEW")}
    print(f"normalization: {status} (--norm-stats {args.norm_stats}, transform sha256 "
          f"{norm['transform_sha256']}, probe ok {probe['ok']})", flush=True)
    return norm, norm_stats


def require_guarded_normalization(mode, cfg, preprocess, postprocess, norm):
    """Every mode rebuilds its processors; they must normalize exactly as the guarded ones did."""
    guard = policy_norm.check(preprocess, postprocess, cfg)
    same = policy_norm.chain_sha256(guard) == norm["transform_sha256"]
    if not same or guard["problems"] != norm["guard"]["problems"]:
        raise RuntimeError(f"{mode}: the processors' normalization differs from the guarded one")


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


@contextlib.contextmanager
def seeded_rng(seed, device):
    """Seed the global RNGs for the body only, then restore them (NEW).

    lerobot's Multi-Task DiT takes no generator: its DDPM sampling draws the initial sample
    (modeling_multi_task_dit.py:701) and one variance-noise tensor per denoising step
    (diffusers DDPMScheduler.step, scheduling_ddpm.py:543-548) from the global RNG. Seeding right
    here, immediately before sampling, makes those draws depend only on `seed`, not on whatever
    consumed the global RNG earlier (data loading, a clean --rmse pass, a profile); forking keeps
    the sampling from shifting anything that runs after it.
    """
    dev = torch.device(device)
    cuda = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)  # seeds CPU and every CUDA device
        yield


def predict_chunk(policy, batch, seed, device):
    """One action chunk for a preprocessed offline batch, sampled under `seed`.

    Policies with an observation history (n_obs_steps > 1, e.g. Multi-Task DiT) normally fill it from
    an online queue that predict_action_chunk reads; offline the history is already in the batch (the
    dataset was built with the config's observation delta timestamps), so call the model directly.
    """
    with seeded_rng(seed, device):
        if getattr(policy.config, "n_obs_steps", 1) > 1 and hasattr(policy, "_generate_actions"):
            b = {k: v for k, v in batch.items() if k != "action"}
            return policy._generate_actions(policy._prepare_batch(b))
        return policy.predict_action_chunk(batch, **noise_kwargs(policy, seed, device))


def noise_kwargs(policy, seed, device):
    """Explicit initial noise for policies that take it (SmolVLA). Others sample from the global RNG, which predict_chunk seeds."""
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
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        chunk = predict_chunk(policy, batch, seed + i, device)  # same noise for every mode on this frame
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
    # NEW: per action dimension, since one flipped binary channel (e.g. a gripper) can dominate rel_l2
    pd, rd = torch.cat(list(preds)), torch.cat(list(ref))
    return {"rel_l2": float((p - r).norm() / r.norm()),
            "cosine": float(torch.nn.functional.cosine_similarity(p, r, dim=0)),
            "max_abs": float((p - r).abs().max()),
            "rel_l2_per_dim": ((pd - rd).norm(dim=0) / rd.norm(dim=0)).tolist()}


def repeat_agreement(first, second):
    """Determinism check (NEW): a second run of a mode with the same seeds must be bitwise identical."""
    same = [torch.equal(a.view(torch.int8) if a.is_floating_point() else a,
                        b.view(torch.int8) if b.is_floating_point() else b)
            for a, b in zip(first, second)]
    p = torch.cat([x.flatten() for x in first]).double()
    q = torch.cat([x.flatten() for x in second]).double()
    d = (p - q).nan_to_num(nan=float("inf")) if not all(same) else torch.zeros_like(p)
    return {"deterministic": all(same) and len(first) == len(second),
            "rel_l2": float(d.norm() / p.norm()), "max_abs": float(d.abs().max()),
            "frames_differing": sum(not s for s in same)}


def determinism_state():
    """The settings that decide whether a repeat can differ (NEW; recorded in the JSON)."""
    def get(f):
        try:
            return f()
        except Exception:
            return None

    return {"use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_allow_tf32": get(lambda: torch.backends.cudnn.allow_tf32),
            "cuda_matmul_allow_tf32": get(lambda: torch.backends.cuda.matmul.allow_tf32),
            "cuda_matmul_allow_bf16_reduced_precision_reduction":
                get(lambda: torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            # CPU reductions are split by thread, so the thread count changes bf16 results
            "num_threads": torch.get_num_threads()}


def make_deterministic():
    """--deterministic (NEW): deterministic kernels, failing loudly on ops that have none."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # required by cuBLAS for determinism
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def torch_dtype(name):
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


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


def build_parser():
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
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16",
                    help="policy dtype for every mode (the baseline included); the reference author "
                         "loads SmolVLA in float32 (serve_smolvla.py:68-69)")
    ap.add_argument("--no-determinism-check", dest="determinism_check", action="store_false",
                    help="run each mode once instead of twice (halves the time; determinism unchecked)")
    ap.add_argument("--deterministic", action="store_true",
                    help="torch.use_deterministic_algorithms(True), cuDNN deterministic, cuBLAS "
                         "workspace config")
    ap.add_argument("--out", required=True)
    policy_norm.add_args(ap)
    add_profile_args(ap)
    add_subset_compare_arg(ap)
    algos.add_args(ap, calib_sources=("non_eval_episodes",))
    __import__("exporter.quant_export", fromlist=['_']).add_args(ap)
    return ap


def _calib_loader(cache, args, cfg, preprocess, norm=None):
    """(f) calibration frames from the episodes make_data does not hold out, loaded once per run;
    each is run under its own seed (calib_seed + i) as run_mode does for the evaluated frames."""
    def calib():
        from algos import calib as algo_calib

        if not cache:
            cache.append(algo_calib.lerobot_frames(
                cfg, args.dataset, args.n_test_episodes, args.seed,
                args.calib_n or algo_calib.DEFAULT_N["policy"], args.calib_seed,
                lambda sample: to_batch(sample, preprocess)))
            if norm is not None:   # the evaluated frames' processors, so the same normalization
                cache[0][1]["normalization"] = {k: norm[k] for k in ("protocol", "status", "stats_sha256",
                                                                      "transform_sha256")}
        batches, rec = cache[0]
        seeds = {id(b): args.calib_seed + i for i, b in enumerate(batches)}

        def run(policy, batch):
            policy.reset()
            return predict_chunk(policy, dict(batch), seeds[id(batch)], args.device)
        return batches, run, rec
    return calib


def main():
    ap = build_parser()
    args = ap.parse_args()
    try:
        algo = algos.make(args.algo)
    except (ValueError, ImportError) as e:
        ap.error(str(e))
    if args.deterministic:
        make_deterministic()
    dtype = torch_dtype(args.dtype)
    policy_norm.check_args(ap, args)
    for m in args.modes:
        try:
            require_mode_supported(m, args.device)  # fail early, before any model download
        except RuntimeError as e:
            ap.error(str(e))
    if "bf16" not in args.modes:
        args.modes = ["bf16", *args.modes]  # every FP8 mode is compared against it
    # the frames are always a sample, so a subset profile gets ref_conv on the same frames
    args.modes, added = subset_compare_modes(args.modes, args.subset_compare == "on")
    import exporter.quant_export as quant_export

    export_mode = quant_export.check_args(ap, args)
    if args.export_only:
        args.modes, added, args.rmse = [export_mode], [], False   # quantize, export, evaluate nothing

    cfg0 = load_config(args.policy, args.device)
    rename = camera_rename_map(cfg0, args.dataset, args.rename_map)
    ds, meta, test_eps, idx = make_data(cfg0, args.dataset, args.n_test_episodes, args.n_frames, args.seed)
    try:
        norm, norm_stats = setup_normalization(args, cfg0, rename, ds, idx)
        if args.from_export:
            policy_norm.check_export(json.loads((Path(args.from_export) / "manifest.json").read_text()), norm)
    except policy_norm.NormalizationError as e:
        raise SystemExit(f"error: {e}")
    result = {"normalization": norm["status"], "normalization_protocol": norm,
              "provenance": provenance(args), "determinism_state": determinism_state(), "modes": {},
              "subset_compare": {"enabled": bool(added) or None, "added_modes": added}}
    ref = None
    clean = None
    calib_cache = []
    rmse_dir = Path(args.rmse_dir) if args.rmse_dir else Path(args.out).with_name(Path(args.out).stem + "_rmse")
    for mode in args.modes:
        policy, cfg = load_policy(args.policy, args.device, dtype)
        result["provenance"].setdefault("checkpoint_key_repair", policy.checkpoint_key_repair)
        preprocess, postprocess = build_processors(cfg, args, rename, norm_stats)
        require_guarded_normalization(mode, cfg, preprocess, postprocess, norm)
        if not args.export_only:
            result["data"] = {"dataset": args.dataset, "test_episodes": test_eps,
                              "frames": len(idx), "fps": meta.fps}
        calib = _calib_loader(calib_cache, args, cfg, preprocess, norm)
        if args.from_export and mode != "bf16":
            import exporter.load_quantized as load_quantized

            # no calibration; every exported tensor overwrites the checkpoint's (docs/algos/export.md)
            _, layers, man = load_quantized.load_quantized(args.policy, args.from_export, args.device,
                                                           build=lambda: policy, mode=mode, options=args)
            result["from_export"] = {"dir": args.from_export, "files": man["files"],
                                     "exported_mode": man["exported_mode"], "algo": man["recipe"]["algo"]}
        else:
            layers = (algos.quantize(policy, mode, args.skip, options=args, algo=algo, calib=calib)
                      if mode != "bf16" else None)
        if mode == export_mode and "export" not in result:
            man = quant_export.export(
                policy, args.export_dir, mode=mode, report=layers, algo=algo, algo_spec=args.algo,
                options=args, skip=args.skip, device=args.device,
                model_record={"kind": "lerobot_policy", "cli": "src/eval_policy.py", "id": args.policy,
                              "dtype": args.dtype, "checkpoint": quant_export.checkpoint_record(args.policy)},
                calib_batches=calib_cache[0][0] if calib_cache else None,
                calib_record=calib_cache[0][1] if calib_cache else None,
                normalization=policy_norm.export_record(norm),
                allow_uncalibrated_fallback=getattr(args, "allow_uncalibrated_fallback", False),
                extra_provenance={"lerobot_dataset": args.dataset, "n_test_episodes": args.n_test_episodes,
                                  "eval_seed": args.seed})
            result["export"] = {"dir": args.export_dir, "mode": mode, "files": man["files"]}
            if args.export_only:
                break
        tt_rec = None
        if getattr(args, "act_two_term", "off") != "off":
            import hw_two_term

            tt_rec = hw_two_term.apply(policy, mode, args.act_two_term)
        algo_rec = algos.entry_record(layers, mode, args.algo)
        eq = layers["ladder"].get("equivalent_to") if layers is not None else None
        if eq in result["modes"]:
            # the op class this step adds is absent from the model: same model, same numbers
            result["modes"][mode] = {**result["modes"][eq], "layers": finalize_report(layers),
                                     "ladder": {**layers["ladder"], "duplicate_of": eq}, "algo": algo_rec,
                                     "added_by_subset_compare": mode in added}
            print(f"{mode}: identical to {eq} (nothing added to quantize); not re-run", flush=True)
            del policy
            continue
        rm = before = None
        if args.rmse and mode != "bf16":
            import layer_rmse

            if clean is None:
                clean, _ = load_policy(args.policy, args.device, dtype)
            rm = layer_rmse.LayerRMSE(clean, policy)

            def before(batch, frame_seed):
                def run_clean():
                    clean.reset()
                    predict_chunk(clean, dict(batch), frame_seed, args.device)  # same noise
                rm.capture(run_clean)
        preds, gts, pads, times = run_mode(policy, ds, idx, preprocess, postprocess,
                                           args.seed, args.device, before_frame=before)
        entry = {"normalization": norm["status"], "layers": finalize_report(layers),
                 **({"hw_two_term": tt_rec} if tt_rec else {}),
                 "ladder": layers["ladder"] if layers is not None else ladder_record("bf16", {}, args),
                 "algo": algo_rec,
                 "added_by_subset_compare": mode in added,
                 "vs_ground_truth": error_vs_gt(preds, gts, pads),
                 "latency_s": {"median": sorted(times)[len(times) // 2], "mean": sum(times) / len(times)}}
        if rm is not None:
            rm.close()
            entry["rmse"] = {**rm.summary(), "csv": rm.write_csvs(rmse_dir / mode)}
        if layers is not None and layers["quantized"] == 0:
            print(f"WARNING: mode {mode} quantized no layers (dtype {args.dtype}); its numbers are the "
                  "baseline's", flush=True)
        entry["deterministic"] = None
        if args.determinism_check:
            again, *_ = run_mode(policy, ds, idx, preprocess, postprocess, args.seed, args.device)
            entry["determinism"] = repeat_agreement(preds, again)
            entry["deterministic"] = entry["determinism"]["deterministic"]
            if not entry["deterministic"]:
                print(f"WARNING: mode {mode} is not deterministic: a repeat with the same seeds gave "
                      f"rel_l2 {entry['determinism']['rel_l2']:.3e} on "
                      f"{entry['determinism']['frames_differing']} frame(s)", flush=True)
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
    result["ladder_deltas"] = ladder_deltas(
        {m: {**{f"gt_{k}": v for k, v in e["vs_ground_truth"].items() if k != "elements"},
             **({f"bf16_{k}": e["vs_bf16"][k] for k in ("rel_l2", "cosine")} if "vs_bf16" in e
                else {"bf16_rel_l2": 0.0, "bf16_cosine": 1.0})}
         for m, e in result["modes"].items()},
        {m: e["ladder"] for m, e in result["modes"].items()})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=str))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
