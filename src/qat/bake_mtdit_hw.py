"""Bake a hardware recipe into a Multi-Task DiT QAT start point (bake_hw.py + the keep_bf16 sites).

TL;DR: bake_hw.py unchanged, except that (1) it imports mtdit_hw instead of smolvla (the same split /
parser helpers), (2) it records the recipe's exact BF16 sites (keep_bf16:rule=input_conv: the CLIP patch
embedding Conv2d) in hw_state.pt["bf16_exact"], and (3) the reload check uses mtdit_hw.PinnedKeep.
Why: bake_hw.py reloads with plain hw_recipe.Pinned, which has no bf16_exact, so algos.quantize would
quantize the kept conv and Pinned.weight_scale raises "no pinned weight exponent". Everything else
(eval_policy's calibration protocol, Recorder, baked weights, the bit-exact reload check that removes
the checkpoint on failure) is bake_hw.py's.

Arms (src/qat/HW_EXPERIMENT.md, with the DiT export's keep_bf16 member):
  A  --recipe "granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib+keep_bf16:rule=input_conv"
  B  --recipe "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv"     (what the DiT export ships)
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mtdit_hw as q  # noqa: E402  (puts src/ on sys.path)

import algos  # noqa: E402
import eval_policy as ep  # noqa: E402
import exporter.quant_export as quant_export  # noqa: E402
import hw_recipe as hw  # noqa: E402
from loaders import policy_norm  # noqa: E402


def main():
    ap = q.build_parser()
    ap.add_argument("--recipe", default="atlas_v1_awq_tile_static+keep_bf16:rule=input_conv")
    ap.add_argument("--calib-n", type=int, default=None, help="default: the algorithm's (eval_policy's)")
    ap.add_argument("--calib-seed", type=int, default=1234, help="eval_policy's default")
    ap.add_argument("--check-n", type=int, default=8, help="val frames for the bit-exact reload check")
    args = ap.parse_args()
    policy_norm.check_args(ap, args)
    dev, dtype = args.device, ep.torch_dtype(args.dtype)
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
    ds_val, idx_val = q.episode_frames(cfg0, args.dataset, meta, val_eps, args.val_n, args.val_seed)
    idx_chk = idx_val[:args.check_n]

    policy, cfg = ep.load_policy(args.policy, dev, dtype)
    preprocess, postprocess = ep.build_processors(cfg, ep_args, rename, norm_stats)
    ep.require_guarded_normalization("bake_hw", cfg, preprocess, postprocess, norm)
    calib = ep._calib_loader([], ep_args, cfg, preprocess, norm)       # eval_policy's own protocol

    algo = algos.make(args.recipe)
    rep = algos.quantize(policy, q.MODE, args.skip, options={}, algo=algo, calib=calib)
    arec = quant_export.jsonable(rep["algo"])
    print(f"{args.recipe} applied ({time.time() - t0:.0f}s)", flush=True)

    with hw.Recorder(algo) as recd:
        ref_preds = ep.run_mode(policy, ds_val, idx_chk, preprocess, postprocess, args.val_seed, dev)[0]
    if recd.bad:
        raise SystemExit("decisions that cannot be pinned:\n  " + "\n  ".join(recd.bad[:20]))
    exact = list(rep["algo"].get("bf16_opt_out_exact") or [])
    if list(rep["algo"].get("bf16_opt_out") or []) != exact:
        raise SystemExit(f"non-exact BF16 opt-outs cannot be pinned: {rep['algo'].get('bf16_opt_out')}")
    state = {"recipe": args.recipe, "weight": hw.weight_state(policy), "act": recd.act,
             "skip": list(args.skip), "mode": q.MODE, "bf16_exact": exact}
    print(f"keep_bf16 sites (exact): {exact}", flush=True)
    n_static = sum(v is not None for v in recd.act.values())
    print(f"recorded {len(state['weight'])} weight sites, {len(recd.act)} activation sites "
          f"({n_static} static, {len(recd.act) - n_static} reference-dynamic)", flush=True)

    sd = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    sd.update(hw.baked_weights(policy))
    del policy
    torch.cuda.empty_cache()

    clean, _ = ep.load_policy(args.policy, "cpu", dtype)
    missing, unexpected = clean.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state_dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    ckpt = out / "checkpoint"
    clean.save_pretrained(ckpt)
    del clean
    src = Path(args.policy)
    if not src.is_dir():
        from huggingface_hub import snapshot_download

        src = Path(snapshot_download(args.policy))
    for f in src.glob("policy_*processor*"):
        shutil.copy2(f, ckpt / f.name)
    torch.save(state, ckpt / "hw_state.pt")

    # -- reload check: checkpoint + Pinned == the recipe's model, bit for bit
    rel, _ = ep.load_policy(str(ckpt), dev, dtype)
    algos.quantize(rel, q.MODE, args.skip, options={}, algo=q.pinned(torch.load(ckpt / "hw_state.pt")))
    rel_preds = ep.run_mode(rel, ds_val, idx_chk, preprocess, postprocess, args.val_seed, dev)[0]
    check = ep.repeat_agreement(ref_preds, rel_preds)
    check["frames"] = len(idx_chk)
    print(f"reload check (checkpoint + Pinned vs {args.recipe} model): {json.dumps(check)}", flush=True)
    rec = {"recipe": args.recipe, "algo_record": arec, "reload_check": check, "n_weight_sites": len(state["weight"]),
           "n_act_sites": len(recd.act), "n_static_act": n_static, "bf16_exact": exact,
           "episodes": {"test": test_eps, "val": val_eps, "train": train_eps},
           "calibration": "eval_policy._calib_loader (non-test episodes, may include the 4 val episodes)",
           "seconds": time.time() - t0}
    (out / "bake.json").write_text(json.dumps(rec, indent=2, default=str))
    if not check["deterministic"]:
        shutil.rmtree(ckpt)
        raise SystemExit("reload check failed; checkpoint removed")
    print("wrote", ckpt, flush=True)


if __name__ == "__main__":
    main()
