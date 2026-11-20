"""DroNet (obstacle avoidance): FP8 ladder eval and quantized export.

    python src/eval_dronet.py --weights W.npz --test-cache TEST.npz --calib-cache VAL.npz \
        --modes fp32 bf16 ref_conv --algo none --out out/dronet_ladder.json
    python src/eval_dronet.py --weights W.npz --calib-cache VAL.npz --modes ref_conv --algo static_calib \
        --export-only --export-dir /new/dir
    python -m exporter.fx_export --export-dir /new/dir --out-dir /new/fx --boundary dronet \
        --frames TEST.npz --native-fp8

The model is the published uzh-rpg DroNet (src/dronet). Boundary: frame (1, 1, 200, 200) in [0, 1] to
(steering, collision logit). `fp32` is an extra unquantized baseline; `bf16` is the repo's usual baseline.
Any other mode is a quant_modes ladder step (`ref_linear`, `ref_conv`, `reference_hw`, `chip`, ...).
Collision metrics are against the dataset labels at threshold 0.5; agreement and error columns are against
the `bf16` mode of the same run (same frames).

Caches are made by `python -m dronet.data build` (uint8 `frames` (N, 200, 200), `labels`).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import algos  # noqa: E402
import quant_modes  # noqa: E402
from dronet import model as dronet_model  # noqa: E402

PROFILE_OPTIONS = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}


def load_cache(path, n=0, seed=0):
    z = np.load(path)
    images, labels = z["frames"], (z["labels"] if "labels" in z.files else None)
    idx = np.arange(len(images))
    if n and n < len(images):
        idx = np.sort(np.random.default_rng(seed).choice(len(images), n, replace=False))
    return images[idx], (labels[idx] if labels is not None else None), idx


def to_input(images, dtype, device):
    return torch.from_numpy(images).float().div(255)[:, None].to(dtype=dtype, device=device)


def calibration(args, dtype):
    """(f) calibration frames: a validation cache that is disjoint from the evaluated test frames."""
    if not args.calib_cache:
        return None
    images, _, idx = load_cache(args.calib_cache, args.calib_n, args.calib_seed)
    batches = [to_input(images[i:i + 1], dtype, args.device) for i in range(len(images))]
    record = {"source": str(Path(args.calib_cache).resolve()), "n_samples": len(batches),
              "seed": args.calib_seed, "frame_indices": idx.tolist()}
    return lambda: (batches, lambda m, b: m(b), record)


def build(args, mode):
    dtype = torch.float32 if mode == "fp32" else torch.bfloat16
    model = dronet_model.load_keras_npz(args.weights, dtype, fused_heads=args.fused_heads).to(args.device)
    report = None
    if mode not in ("fp32", "bf16"):
        algo = algos.make(args.algo)
        calib = calibration(args, dtype)
        report = algos.quantize(model, mode, [], dict(PROFILE_OPTIONS), algo=algo, calib=calib)
        report["_algo"], report["_calib"] = algo, calib
    return model.eval(), dtype, report


def export(args, model, mode, report, dtype):
    import exporter.quant_export as quant_export

    calib = report["_calib"]
    batches, _, record = calib() if calib else (None, None, None)
    return quant_export.export(
        model, args.export_dir, mode=mode, report=report, algo=report["_algo"], algo_spec=args.algo,
        options=dict(PROFILE_OPTIONS), skip=[], device=args.device,
        model_record={"kind": "dronet", "cli": "src/eval_dronet.py",
                      "source": "uzh-rpg/rpg_public_dronet best_weights.h5 (http://rpg.ifi.uzh.ch/data/dronet_model.zip)",
                      "weights_npz": str(Path(args.weights).resolve()), "fused_heads": bool(args.fused_heads)},
        calib_batches=batches, calib_record=record)


@torch.no_grad()
def run(model, images, dtype, args):
    out = []
    for i in range(0, len(images), args.batch_size):
        out.append(model(to_input(images[i:i + args.batch_size], dtype, args.device)).float().cpu())
    return torch.cat(out).numpy()          # (N, 2): steering, collision logit


def summarize(logits, labels, base):
    prob = 1 / (1 + np.exp(-logits[:, 1].astype(np.float64)))
    row = {"steer_mean": float(logits[:, 0].mean()), "steer_std": float(logits[:, 0].std()),
           "collision_rate_pred": float((prob >= 0.5).mean())}
    if labels is not None:
        from dronet import data as dronet_data

        row["collision_vs_labels"] = dronet_data.metrics(prob, labels)
    if base is not None:
        d = logits - base
        row["vs_bf16"] = {
            "steer_max_abs": float(np.abs(d[:, 0]).max()), "steer_mean_abs": float(np.abs(d[:, 0]).mean()),
            "logit_max_abs": float(np.abs(d[:, 1]).max()), "logit_mean_abs": float(np.abs(d[:, 1]).mean()),
            "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(base), 1e-12)),
            "collision_decision_agreement": float(((logits[:, 1] >= 0) == (base[:, 1] >= 0)).mean())}
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="npz from dronet.convert_keras")
    ap.add_argument("--test-cache")
    ap.add_argument("--calib-cache")
    ap.add_argument("--calib-n", type=int, default=64)
    ap.add_argument("--calib-seed", type=int, default=0)
    ap.add_argument("--n-frames", type=int, default=0, help="seeded subset of the test cache (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--modes", nargs="+", default=["fp32", "bf16", "ref_conv"])
    ap.add_argument("--algo", default="none")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--export-dir")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--fused-heads", action="store_true",
                    help="one Linear(6272, 2) head (the tile-aware export's verifier rejects single-output weights); "
                         "the two heads then share one weight exponent")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    if args.export_only and not args.export_dir:
        ap.error("--export-only needs --export-dir")
    if not args.export_only and not args.test_cache:
        ap.error("--test-cache is required unless --export-only")

    quantized = [m for m in args.modes if m not in ("fp32", "bf16")]
    export_mode = None
    if args.export_dir:
        from exporter import quant_export

        export_mode = quant_export.pick_mode(args.modes)
        if export_mode is None:
            ap.error("--export-dir needs a reference-numerics mode in --modes (e.g. ref_conv)")
    result = {"cli": "src/eval_dronet.py", "algo": args.algo, "weights": str(Path(args.weights).resolve()),
              "modes": {}}
    if not args.export_only:
        images, labels, idx = load_cache(args.test_cache, args.n_frames, args.seed)
        result["data"] = {"test_cache": str(Path(args.test_cache).resolve()), "n_frames": int(len(images)),
                          "seed": args.seed, "subset": bool(args.n_frames)}
    base = None
    order = (["bf16"] if "bf16" not in args.modes and not args.export_only else []) + list(args.modes)
    for mode in order:
        if args.export_only and mode != export_mode:
            continue
        model, dtype, report = build(args, mode)
        if mode == export_mode:
            result["export"] = export(args, model, mode, report, dtype)["counts"]
        if args.export_only:
            continue
        logits = run(model, images, dtype, args)
        if mode == "bf16":
            base = logits
        row = summarize(logits, labels, base if mode != "bf16" else None)
        if report is not None:
            row["quantized"] = {k: report.get(k) for k in ("planned", "quantized", "untouched", "conv2d")}
            row["ladder"] = report.get("ladder")
        result["modes"][mode] = row
        print(f"{mode}: {json.dumps({k: v for k, v in row.items() if k != 'ladder'})}", flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
