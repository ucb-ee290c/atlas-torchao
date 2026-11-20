"""Bake a PTQ algorithm (default AWQ + GPTQ) into a plain bf16 SmolVLA checkpoint, as a QAT start point.

TL;DR: runs the repo's `algos.quantize(policy, "ref_conv", algo=gptq_awq:...)` with calibration frames
from the QAT TRAIN episodes only (never val / test), then writes every Linear's dequantized GPTQ weight
(exact in bf16: an E4M3 value times 2^s) and the AWQ-folded norms / producers into a normal checkpoint.
Loading that checkpoint and running plain `ref_conv` (--algo none) must reproduce the PTQ model bit for
bit; the script checks this on val frames and refuses to save otherwise.

That check can fail only through the weight exponent: gptq_awq pins s0 = po2_scale_exp(w before GPTQ),
while plain ref_conv re-derives it from the baked weight. If GPTQ lowered a tensor's amax below
256 * 2^s0 the derived exponent drops by one and the top codes would saturate. Such layers are listed.

Usage (one GPU):
  python src/qat/bake_ptq.py --policy /rscratch/agarcia/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --norm-stats dataset --out-dir src/out/qat/ptq_awq_gptq --algo gptq_awq:method=awq_gptq,actorder=true
"""
import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smolvla as q  # noqa: E402  (puts src/ on sys.path)

import algos  # noqa: E402
import eval_policy as ep  # noqa: E402
import exporter.quant_export as quant_export  # noqa: E402
import quant_modes  # noqa: E402
import reference.ref_numerics as rn  # noqa: E402
from loaders import policy_norm  # noqa: E402
from reference.ref_profile import RefQuantLinear  # noqa: E402


def main():
    ap = q.build_parser()
    ap.add_argument("--algo", default="gptq_awq:method=awq_gptq,actorder=true")
    ap.add_argument("--calib-n", type=int, default=64, help="calibration frames from the TRAIN episodes")
    ap.add_argument("--calib-seed", type=int, default=3000)
    ap.add_argument("--check-n", type=int, default=8, help="val frames for the bake parity check")
    args = ap.parse_args()
    policy_norm.check_args(ap, args)
    dev = args.device
    dtype = ep.torch_dtype(args.dtype)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ep_args = argparse.Namespace(**{**vars(args), "seed": args.eval_seed, "policy": args.policy})

    cfg0 = ep.load_config(args.policy, dev)
    rename = ep.camera_rename_map(cfg0, args.dataset, args.rename_map)
    meta, test_eps, val_eps, train_eps = q.split_episodes(args.dataset, args.n_test_episodes, args.eval_seed,
                                                          args.n_val_episodes)
    ds_test, _, _, idx_test = ep.make_data(cfg0, args.dataset, args.n_test_episodes, args.n_frames, args.eval_seed)
    norm, norm_stats = ep.setup_normalization(ep_args, cfg0, rename, ds_test, idx_test)
    ds_cal, idx_cal = q.episode_frames(cfg0, args.dataset, meta, train_eps, args.calib_n, args.calib_seed)
    ds_val, idx_val = q.episode_frames(cfg0, args.dataset, meta, val_eps, args.val_n, args.val_seed)
    idx_chk = idx_val[:args.check_n]

    policy, cfg = ep.load_policy(args.policy, dev, dtype)
    preprocess, postprocess = ep.build_processors(cfg, ep_args, rename, norm_stats)
    ep.require_guarded_normalization("bake", cfg, preprocess, postprocess, norm)

    batches = [ep.to_batch(ds_cal[i], preprocess) for i in idx_cal]
    seeds = {id(b): args.calib_seed + j for j, b in enumerate(batches)}
    crec = {"source": f"{args.dataset} QAT train episodes", "episodes": train_eps, "n_samples": len(idx_cal),
            "seed": args.calib_seed, "frame_indices": idx_cal}

    def calib():
        def run(pol, batch):
            pol.reset()
            return ep.predict_chunk(pol, dict(batch), seeds[id(batch)], dev)
        return batches, run, crec

    print(f"calibration: {len(batches)} train frames ({time.time() - t0:.0f}s)", flush=True)
    algo = algos.make(args.algo)
    rep = algos.quantize(policy, q.MODE, args.skip, options={}, algo=algo, calib=calib)
    print(f"{args.algo} applied ({time.time() - t0:.0f}s): {json.dumps(rep['algo'].get('weights'))}", flush=True)

    # -- bake: the PTQ model's state dict with each Linear weight replaced by its dequantized codes
    sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    exp_shift = {}
    n_lin = 0
    with torch.no_grad():
        for name, m in policy.named_modules():
            if not isinstance(m, RefQuantLinear):
                continue
            n_lin += 1
            wd = m._dequant_weight().float().reshape(m.weight.shape)
            wb = wd.to(dtype)
            if not torch.equal(wb.float(), wd):
                raise SystemExit(f"{name}: dequantized weight is not exact in {dtype}")
            s_dyn = rn.po2_scale_exp(wb)
            if not torch.equal(s_dyn.reshape(()).cpu(), m.weight_scale_exp.reshape(()).cpu()):
                exp_shift[name] = (int(m.weight_scale_exp.reshape(())), int(s_dyn.reshape(())))
            sd[f"{name}.weight"] = wb.cpu()
    print(f"baked {n_lin} Linear weights; exponent changes under the dynamic rule: {len(exp_shift)}", flush=True)

    # PTQ model predictions on the check frames (before it is freed)
    ptq_preds = ep.run_mode(policy, ds_val, idx_chk, preprocess, postprocess, args.val_seed, dev)[0]
    del policy
    torch.cuda.empty_cache()

    clean, _ = ep.load_policy(args.policy, "cpu", dtype)
    missing, unexpected = clean.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    ckpt = out / "checkpoint"
    clean.save_pretrained(ckpt)
    src = Path(args.policy)
    if not src.is_dir():                                     # a Hub id: its local snapshot
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.policy))
    for f in src.glob("policy_*processor*"):
        shutil.copy2(f, ckpt / f.name)
    del clean

    # -- check: the baked checkpoint under plain ref_conv == the PTQ model, bit for bit
    baked, _ = ep.load_policy(str(ckpt), dev, dtype)
    quant_modes.quantize(baked, q.MODE, args.skip, options={})
    baked_preds = ep.run_mode(baked, ds_val, idx_chk, preprocess, postprocess, args.val_seed, dev)[0]
    check = ep.repeat_agreement(ptq_preds, baked_preds)
    check["frames"] = len(idx_chk)
    print(f"bake check (baked + ref_conv vs {args.algo} model): {json.dumps(check)}", flush=True)
    rec = {"algo": args.algo, "record": quant_export.jsonable(rep["algo"]),
           "calibration": crec, "linear_baked": n_lin, "exp_shift": exp_shift, "bake_check": check,
           "episodes": {"test": test_eps, "val": val_eps, "train": train_eps}, "seconds": time.time() - t0}
    (out / "bake.json").write_text(json.dumps(rec, indent=2, default=str))
    if not check["deterministic"]:
        shutil.rmtree(ckpt)
        raise SystemExit(f"bake check failed (exponent shifts: {exp_shift}); checkpoint removed")
    print("wrote", ckpt, flush=True)


if __name__ == "__main__":
    random.seed(0)
    main()
