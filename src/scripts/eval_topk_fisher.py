"""Keep the top-k Fisher linear layers in bf16, everything else FP8; compare with all-bf16 and all-FP8.

    python src/scripts/eval_topk_fisher.py --fisher src/out/fisher/tinyllama/tinyllama_n128.json \\
        --ks 10 5 3 --device cuda:1
    python src/scripts/eval_topk_fisher.py --fisher ... --attn-fp8        # q/k/p/v also FP8

Recovery = (L_allFP8 - L_config) / (L_allFP8 - L_bf16) on the mean loss: 0% = no better than all-FP8,
100% = as good as all-bf16; "+-" is the bootstrap standard error over the evaluation items. Random-k baselines
(same k, random linear layers) show what a ranking has to beat. Optional --measured takes a
validate_*_actual.json (single-layer-quantized loss changes) as a second ranking.

Works for an LLM report and a VLA report; the kind is read from the report (seqlen "n/a" = VLA):
  llm  metric: mean next-token NLL on held-out WikiText-2 windows (none overlaps a scored window).
  vla  metric: the policy's flow-matching loss on held-out frames from the calibration episodes (frames the
       scores did not use), each (frame, draw) under a fixed noise / time seed so every config sees the same
       draws. Needs `lerobot` and the report's policy / dataset.

    python src/scripts/eval_topk_fisher.py --fisher src/out/fisher/smolvla/smolvla_n8.json --norm-stats dataset \\
        --ks 10 5 3 --device cuda:1

Output: <fisher dir>/validation/topk_<fisher stem>.json (rankings, per-item losses, summary)."""
import argparse
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def load_llm(a, meta):
    """-> SimpleNamespace(model, items, loss(model, item) -> float, info)."""
    import torch.nn.functional as F

    from algos import calib
    from loaders.modelutils import load_llama

    model_path, seqlen = meta["model"], meta["seqlen"]
    used = meta["calibration"]["window_starts"]
    cand, rec = calib.wikitext2_train(model_path, 4 * a.n + 40, seqlen, a.seed, use_fast=True)
    keep = [i for i, s in enumerate(rec["window_starts"]) if all(abs(s - u) >= seqlen for u in used)][:a.n]
    if len(keep) < a.n:
        raise SystemExit(f"only {len(keep)} held-out windows available, wanted {a.n}")
    model = load_llama(model_path, seqlen=seqlen, device=a.device)
    dev = next(model.parameters()).device

    def loss(m, ids):
        ids = ids.to(dev)
        return F.cross_entropy(m(ids, use_cache=False).logits[0, :-1].float(), ids[0, 1:], reduction="mean").item()

    return SimpleNamespace(model=model, items=[cand[i] for i in keep], loss=loss, label="windows",
                           info={"seqlen": seqlen, "window_starts": [rec["window_starts"][i] for i in keep]})


def load_vla(a, meta):
    import eval_policy as ep
    from algos import calib

    a.device = a.device or "cuda"                           # lerobot's device step rejects None
    ns = SimpleNamespace(**vars(a))
    ns.dataset, ns.policy = meta["dataset"], meta["model"]   # what eval_policy's helpers read off args
    ns.seed = a.eval_seed                                   # the name eval_policy's helpers read
    dtype = ep.torch_dtype(a.dtype)
    cfg0 = ep.load_config(meta["model"], a.device)
    rename = ep.camera_rename_map(cfg0, meta["dataset"], a.rename_map)
    ds, _, _, idx = ep.make_data(cfg0, meta["dataset"], a.n_test_episodes, 1, a.eval_seed)
    norm, norm_stats = ep.setup_normalization(ns, cfg0, rename, ds, idx)
    policy, cfg = ep.load_policy(meta["model"], a.device, dtype)
    preprocess, _ = ep.build_processors(cfg, ns, rename, norm_stats)
    used = set(meta["calibration"]["frame_indices"])
    frames, rec = calib.lerobot_frames(cfg, meta["dataset"], a.n_test_episodes, a.eval_seed, 3 * a.n + 8, a.seed,
                                       lambda s: ep.to_batch(s, preprocess))
    fresh = [(b, i) for b, i in zip(frames, rec["frame_indices"]) if i not in used][:a.n]
    if len(fresh) < a.n:
        raise SystemExit(f"only {len(fresh)} unscored frames available, wanted {a.n}")
    draws = a.draws
    items = [(b, a.seed * 1000 + 17 * k + d) for k, (b, _) in enumerate(fresh) for d in range(draws)]

    ref = {}                                                # item -> the bf16 action chunk

    def loss(m, item):
        batch, seed = item
        if a.vla_metric == "flow_loss":                     # one noisy training step (insensitive: see docs)
            with ep.seeded_rng(seed, a.device):
                out = m.forward(dict(batch))
            return float((out[0] if isinstance(out, (tuple, list)) else out).float())
        # chunk: relative L2 distance of the generated action chunk from the bf16 chunk, same initial noise.
        # The first call per item is the all-bf16 baseline (main() runs it first), which becomes the reference.
        m.reset()
        chunk = ep.predict_chunk(m, dict(batch), seed, a.device).float().cpu()
        key = id(batch), seed
        if key not in ref:
            ref[key] = chunk
            return 0.0
        return float((chunk - ref[key]).norm() / ref[key].norm())

    return SimpleNamespace(model=policy, items=items, loss=loss, label="frame-draws",
                           info={"dataset": meta["dataset"], "frame_indices": [i for _, i in fresh],
                                 "draws": draws, "normalization": norm["status"]})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fisher", required=True, help="report JSON from profile_fisher_llm.py / _vla.py")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 5, 3])
    ap.add_argument("--n", type=int, default=None,
                    help="held-out windows (llm, default 64) or frames (vla, default 32)")
    ap.add_argument("--seed", type=int, default=777, help="held-out sampling seed")
    ap.add_argument("--random", type=int, default=0,
                    help="random layer sets per random-k, as a chance baseline (default 0: none, top-k only)")
    ap.add_argument("--random-ks", type=int, nargs="+", default=None,
                    help="k values that get random baselines (default: only the largest --ks, the hardest "
                         "random baseline; pass all of --ks for one per k)")
    ap.add_argument("--measured", default=None, help="validate_*_actual.json: adds a measured-top-k ranking")
    ap.add_argument("--attn-fp8", action="store_true", help="also run with attention q/k/p/v in FP8")
    ap.add_argument("--exclude", nargs="*", default=None,
                    help="fqn substrings left in bf16 always (default: lm_head for llm, none for vla)")
    ap.add_argument("--boot", type=int, default=500, help="bootstrap resamples for the +- error")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None, help="default: <fisher dir>/validation/topk_<fisher stem>.json")
    vla = ap.add_argument_group("vla only")
    vla.add_argument("--draws", type=int, default=1, help="noise draws per frame (more frames beat more draws)")
    vla.add_argument("--n-test-episodes", type=int, default=5, help="episodes held out (as eval_policy)")
    vla.add_argument("--eval-seed", type=int, default=0, help="eval_policy's --seed: fixes the held-out split")
    vla.add_argument("--vla-metric", choices=["chunk", "flow_loss"], default="chunk",
                     help="chunk: action-chunk distance from the bf16 chunk (default); flow_loss: training loss")
    vla.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    vla.add_argument("--rename-map", type=json.loads, default=None)
    import loaders.policy_norm as policy_norm
    policy_norm.add_args(ap)
    a = ap.parse_args()

    import numpy as np
    import torch

    import reference.ref_attention as ra
    from reference.ref_profile import RefQuantLinear, quantize_reference

    fisher = Path(a.fisher)
    out = Path(a.out) if a.out else fisher.parent / "validation" / f"topk_{fisher.stem}.json"
    sc = json.loads(fisher.read_text())
    meta = sc["meta"]
    kind = "vla" if meta.get("seqlen") in (None, "n/a") else "llm"
    if a.exclude is None:
        a.exclude = ["lm_head"] if kind == "llm" else []
    if a.n is None:
        a.n = 64 if kind == "llm" else 32
    a.random_ks = a.random_ks or [max(a.ks)]
    print(f"report kind: {kind}, n = {a.n}, random baselines: {a.random_ks if a.random else 'none'}", flush=True)

    lin = {n: r["fisher"] for n, r in sc["layers"].items() if r.get("kind") != "attn"}
    rankings = {"fisher": sorted(lin, key=lambda n: -lin[n])}
    if a.measured:
        gt = json.loads(Path(a.measured).read_text())["delta"]
        rankings["measured"] = sorted(gt, key=lambda n: -np.mean(gt[n]))

    task = (load_llm if kind == "llm" else load_vla)(a, meta)
    model = task.model
    print(f"{len(task.items)} held-out {task.label}", flush=True)
    quantize_reference(model, skip=tuple(a.exclude))
    layers = {n: m for n, m in model.named_modules() if isinstance(m, RefQuantLinear)}
    unknown = [n for r in rankings.values() for n in r[:max(a.ks)] if n not in layers]
    if unknown:
        raise SystemExit(f"ranked layers not in the model (excluded or renamed?): {unknown[:3]}")

    def set_keep(keep_bf16):                    # clean pass-through = bf16; the rest stay on the FP8 path
        for n, m in layers.items():
            m.profile_fisher(n in keep_bf16, record=False)

    @torch.no_grad()
    def losses():
        return np.array([task.loss(model, it) for it in task.items])

    rng = random.Random(0)
    configs = [("fp8_all", set())]
    for k in a.ks:
        for rname, r in rankings.items():
            configs.append((f"{rname}_top{k}", set(r[:k])))
    for k in a.random_ks:
        for j in range(a.random):
            configs.append((f"random{k}_{j}", set(rng.sample(list(layers), k))))

    t0 = time.time()
    set_keep(set(layers))
    base = losses()
    print(f"bf16 loss {base.mean():.5f}", flush=True)
    result = {"bf16": base.tolist()}

    def run(variant):
        result[variant] = {}
        for name, S in configs:
            set_keep(S)
            result[variant][name] = losses().tolist()
            m = np.mean(result[variant][name])
            print(f"[{variant}] {name:16s} loss {m:.5f}  (+{m - base.mean():.5f} vs bf16)  {time.time() - t0:.0f}s",
                  flush=True)
            write(done=False)

    def recovery(cfg, fp8, boot_rng):
        """(fp8 - cfg) / (fp8 - bf16) on the mean, and its bootstrap standard error over items."""
        c, f = np.array(cfg), np.array(fp8)
        point = (f.mean() - c.mean()) / (f.mean() - base.mean())
        idx = boot_rng.integers(0, len(c), (a.boot, len(c)))
        bs = (f[idx].mean(1) - c[idx].mean(1)) / (f[idx].mean(1) - base[idx].mean(1))
        return float(point), float(np.std(bs))

    def summarize():
        s, boot_rng = {}, np.random.default_rng(0)
        for variant in [v for v in ("linear_fp8", "linear_attn_fp8") if v in result]:
            fp8 = result[variant].get("fp8_all")
            if fp8 is None:
                continue
            s[variant] = {"fp8_all_loss": float(np.mean(fp8)), "bf16_loss": float(base.mean())}
            for name, v in result[variant].items():
                if name != "fp8_all" and not name.startswith("random"):
                    r, se = recovery(v, fp8, boot_rng)
                    s[variant][name] = {"loss": float(np.mean(v)), "recovery": r, "recovery_se": se}
            for k in a.random_ks:
                rs = [recovery(v, fp8, boot_rng)[0] for n, v in result[variant].items()
                      if n.startswith(f"random{k}_")]
                if rs:
                    s[variant][f"random{k}"] = {"recovery_mean": float(np.mean(rs)),
                                                "recovery_min": float(min(rs)), "recovery_max": float(max(rs))}
        return s

    def write(done=True):
        payload = {"meta": {"kind": kind, "fisher": str(fisher), "model": meta["model"], "n_items": len(task.items),
                            "eval_seed": a.seed, **task.info, "ks": a.ks, "random_draws": a.random, "random_ks": a.random_ks,
                            "excluded": a.exclude,
                            "metric": ("mean next-token NLL per window" if kind == "llm"
                                       else ("action-chunk relative L2 from the bf16 chunk per (frame, draw)" if a.vla_metric == "chunk"
                                        else "flow-matching loss per (frame, draw)"))
                                      + "; recovery = (fp8 - cfg) / (fp8 - bf16)"},
                   "rankings": {k: v[:max(a.ks)] for k, v in rankings.items()},
                   "per_item_loss": result, "summary": summarize() if done else None}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=1))

    run("linear_fp8")
    if a.attn_fp8:
        print(len(ra.patch_attention(model)), "attention sites -> FP8 q/k/p/v", flush=True)
        run("linear_attn_fp8")
    write(done=True)

    print(f"\nbf16 loss {base.mean():.5f}")
    for variant, s in summarize().items():
        print(f"\n[{variant}]  all-FP8 loss {s['fp8_all_loss']:.5f}  (+{s['fp8_all_loss'] - s['bf16_loss']:.5f})")
        for name, v in s.items():
            if isinstance(v, dict) and "recovery_se" in v:
                print(f"  {name:16s} loss {v['loss']:.5f}  recovery {100 * v['recovery']:5.1f}% +- {100 * v['recovery_se']:.1f}%")
            elif isinstance(v, dict):
                print(f"  {name:16s} mean {100 * v['recovery_mean']:5.1f}%  range {100 * v['recovery_min']:.1f}..{100 * v['recovery_max']:.1f}%")
    print("wrote", out)


if __name__ == "__main__":
    main()
