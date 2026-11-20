"""Evaluate SmolVLA under a HARDWARE RECIPE (pinned FP8 decisions) on one large held-out test set.

TL;DR: eval_qat.py with one change: every --ckpt NAME=DIR is a bake_hw.py or smolvla_hw.py checkpoint
and is quantized with hw_recipe.Pinned from its own hw_state.pt, i.e. exactly the casts it deploys with
(the FX export's recipe). "% gap recovered" is paired per frame against --baseline (the first --ckpt by
default); every extra --compare NAME gets its own paired gap columns. bf16 (--policy) is the reference.

  torchrun --standalone --nproc_per_node 4 src/qat/eval_hw.py --policy lerobot/smolvla_libero \
      --dataset lerobot/libero --norm-stats checkpoint --n-test-episodes 32 \
      --ckpt A=src/out/qat_hw/A_rtn/checkpoint --ckpt B=src/out/qat_hw/B_awq_gptq/checkpoint \
      --ckpt C_s0=... --baseline A --compare B --n-frames 500 --draws 1 --out src/out/qat_hw/eval_hw.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_policy as ep  # noqa: E402
import loaders.policy_norm as policy_norm  # noqa: E402
import algos  # noqa: E402
import hw_recipe as hw  # noqa: E402
import smolvla as q  # noqa: E402  (qat/smolvla.py: teacher_trace, student_loss, dist helpers, prefetch)

MODE = "ref_conv"
BASE = None          # set from --baseline
DRAW_STRIDE = 1_000_003            # draw d uses noise seeds eval_seed + d * DRAW_STRIDE + frame index


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--policy", required=True, help="the ORIGINAL checkpoint (bf16 reference, ref_conv baseline)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=DIR", help="QAT checkpoint (repeatable)")
    ap.add_argument("--ckpt-bf16", action="store_true", help="also evaluate each checkpoint unquantized")
    ap.add_argument("--baseline", default=None, help="NAME of the --ckpt every gap is paired against (default: first)")
    ap.add_argument("--compare", action="append", default=[], help="extra baseline NAME(s) with their own gap columns")
    ap.add_argument("--n-test-episodes", type=int, default=5)
    ap.add_argument("--eval-seed", type=int, default=0, help="eval_policy's --seed (fixes the test episodes)")
    ap.add_argument("--n-frames", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=2, help="noise draws per frame")
    ap.add_argument("--no-vel", action="store_true", help="skip the teacher-forced velocity metric")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--rename-map", type=json.loads, default=None)
    ap.add_argument("--tokenizer-name", default=None)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", required=True, help="JSON; a .md table is written next to it")
    policy_norm.add_args(ap)
    return ap


# ---------------------------------------------------------------------------------------------
# Per-frame measurements (this rank's frames)
# ---------------------------------------------------------------------------------------------

def gt_of(policy, sample, t):
    """The recorded action window from the delta-0 step, as eval_policy.run_mode slices it."""
    off = list(policy.config.action_delta_indices).index(0)
    gt = sample["action"].float()[off:]
    pad = sample.get("action_is_pad", torch.zeros(sample["action"].shape[0], dtype=torch.bool))[off:]
    return gt[:t], pad[:t]


@torch.no_grad()
def measure(policy, ds, idx, preprocess, postprocess, traces, seed_base, dev, adim, teacher):
    """One sampling pass per frame (noise seed seed_base + frame, as run_mode): the free-running action
    chunk (dataset units) and, if traces is given, the teacher-forced velocity error. At each Euler
    step the model is also called on the teacher's x_t (denoise_step crops the KV cache back to the
    prefix after every call, so the extra call leaves the free-running path unchanged; checked
    against eval_policy.run_mode by the caller). Returns (preds, gts, pads, vel)."""
    preds, gts, pads, vel = [], [], [], []
    for i in idx:
        seed = seed_base + i
        sample = ds[i]
        batch = q.make_batch(sample, preprocess, teacher, seed, dev)
        errs = []
        if traces is not None:
            _, xs, vs = traces[(i, seed_base)]

            def both(orig, k, **kw):
                v_f = orig(**{**kw, "x_t": xs[k]})
                errs.append(float(F.mse_loss(v_f[..., :adim].float(), vs[k][..., :adim].float())))
                return orig(**kw)
            ctx = q.intercept_denoise(policy.model, both)
        else:
            ctx = contextlib.nullcontext()
        with ctx, ep.seeded_rng(seed, dev):
            chunk = q.run_chunk(policy, batch)
        chunk = postprocess(chunk).float().cpu()[0]
        gt, pad = gt_of(policy, sample, chunk.shape[0])
        t = min(chunk.shape[0], gt.shape[0])
        preds.append(chunk[:t])
        gts.append(gt)
        pads.append(pad)
        vel.append(sum(errs) / len(errs) if errs else 0.0)
    return preds, gts, pads, vel


def frame_records(name, preds_by_draw, ref_by_draw, gt_by_draw, idx, vel_by_draw):
    """{frame: {...}} for one model: per draw sq err vs bf16 (total and per dim), gt errors, vel."""
    recs = {}
    for n, i in enumerate(idx):
        r = {"sq": [], "sq_dim": [], "ref2": [], "ref2_dim": [], "gt_se": [], "gt_ae": [], "gt_n": [], "vel": []}
        for d in range(len(preds_by_draw)):
            p, ref = preds_by_draw[d][n].double(), ref_by_draw[d][n].double()
            g, pad = gt_by_draw[d][0][n].double(), gt_by_draw[d][1][n]
            r["sq"].append(float((p - ref).pow(2).sum()))
            r["sq_dim"].append((p - ref).pow(2).sum(0).tolist())
            r["ref2"].append(float(ref.pow(2).sum()))
            r["ref2_dim"].append(ref.pow(2).sum(0).tolist())
            m = ~pad
            e = p[m] - g[m]
            r["gt_se"].append(float(e.pow(2).sum()))
            r["gt_ae"].append(float(e.abs().sum()))
            r["gt_n"].append(int(e.numel()))
            if vel_by_draw is not None:
                r["vel"].append(vel_by_draw[d][n])
        recs[i] = r
    return name, recs


# ---------------------------------------------------------------------------------------------
# Statistics (rank 0)
# ---------------------------------------------------------------------------------------------

def frame_sums(recs, frames, key):
    return torch.tensor([sum(recs[f][key]) for f in frames], dtype=torch.float64)


def paired_gap(base, model, n_boot, seed=0):
    """1 - sum(model) / sum(base) over frames, paired SE (delta method on the sum) and a frame
    bootstrap 95% CI."""
    if float(base.sum()) == 0.0:
        return {"gap_recovered": None}
    d = base - model
    gap = float(d.sum() / base.sum())
    se = float(d.std() * math.sqrt(len(d)) / base.sum()) if len(d) > 1 else None
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(d), (n_boot, len(d)), generator=g)
    boots = 1.0 - model[idx].sum(1) / base[idx].sum(1)
    lo, hi = torch.quantile(boots, torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist()
    return {"gap_recovered": gap, "se": se, "ci95": [lo, hi],
            "frames_improved": float((d > 0).double().mean())}


def summarize(all_recs, names, frames, episodes, n_boot, has_vel, base_name, compare=()):
    base = all_recs[base_name]
    res = {}
    ep_ids = sorted(set(episodes.values()))
    for name in names:
        r = all_recs[name]
        sq, ref2 = frame_sums(r, frames, "sq"), frame_sums(r, frames, "ref2")
        se_gt = frame_sums(r, frames, "gt_se")
        ae_gt = frame_sums(r, frames, "gt_ae")
        n_gt = frame_sums(r, frames, "gt_n")
        per_frame_rel = (sq / ref2.clamp_min(1e-30)).sqrt()
        rec = {"chunk": {"rel_l2": float((sq.sum() / ref2.sum()).sqrt()),
                         "median_frame_rel_l2": float(per_frame_rel.median()),
                         "sq_err": float(sq.sum())},
               "gt": {"mse": float(se_gt.sum() / n_gt.sum()), "mae": float(ae_gt.sum() / n_gt.sum())}}
        dims = len(r[frames[0]]["sq_dim"][0])
        sq_dim = torch.tensor([[sum(x[k] for x in r[f]["sq_dim"]) for k in range(dims)] for f in frames],
                              dtype=torch.float64)
        ref_dim = torch.tensor([[sum(x[k] for x in r[f]["ref2_dim"]) for k in range(dims)] for f in frames],
                               dtype=torch.float64)
        rec["chunk"]["rel_l2_per_dim"] = (sq_dim.sum(0) / ref_dim.sum(0)).sqrt().tolist()
        if has_vel:
            rec["vel"] = {"mse": float(frame_sums(r, frames, "vel").sum() / (len(frames) * len(r[frames[0]]["vel"])))}
        for cb in compare:
            if name in (cb, "bf16"):
                continue
            cb_sq = frame_sums(all_recs[cb], frames, "sq")
            rec.setdefault("vs", {})[cb] = {"chunk": paired_gap(cb_sq, sq, n_boot)}
            if has_vel:
                rec["vs"][cb]["vel"] = paired_gap(frame_sums(all_recs[cb], frames, "vel"),
                                                  frame_sums(r, frames, "vel"), n_boot, seed=1)
        if name not in (base_name, "bf16"):
            b_sq = frame_sums(base, frames, "sq")
            rec["chunk"].update(paired_gap(b_sq, sq, n_boot))
            b_dim = torch.tensor([[sum(x[k] for x in base[f]["sq_dim"]) for k in range(dims)] for f in frames],
                                 dtype=torch.float64)
            rec["chunk"]["gap_per_dim"] = (1 - sq_dim.sum(0) / b_dim.sum(0)).tolist()
            rec["chunk"]["gap_per_episode"] = {}
            for e in ep_ids:
                m = torch.tensor([episodes[f] == e for f in frames])
                rec["chunk"]["gap_per_episode"][str(e)] = {"frames": int(m.sum()),
                                                           "gap_recovered": float(1 - sq[m].sum() / b_sq[m].sum())}
            if has_vel:
                b_v, v = frame_sums(base, frames, "vel"), frame_sums(r, frames, "vel")
                rec["vel"].update(paired_gap(b_v, v, n_boot, seed=1))
                rec["vel"]["gap_per_episode"] = {
                    str(e): float(1 - v[torch.tensor([episodes[f] == e for f in frames])].sum()
                                  / b_v[torch.tensor([episodes[f] == e for f in frames])].sum()) for e in ep_ids}
        res[name] = rec
    return res


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}%"


def markdown(res, meta):
    B = meta["baseline"]
    lines = [f"# SmolVLA QAT under the hardware recipe — held-out test (gap vs {B})", "",
             f"Test episodes {meta['test_episodes']}, {meta['frames']} frames × {meta['draws']} noise draws "
             f"= {meta['frames'] * meta['draws']} chunks per model. Reference: the original bf16 policy. "
             f"Gap recovered = 1 − err(model) / err({B}), paired per frame (± SE; 95% bootstrap CI). "
             f"Every model is quantized with its own pinned recipe (hw_state.pt).", "",
             "| model | chunk rel_l2 | chunk gap recovered | 95% CI | frames improved | median frame rel_l2 "
             "| vel MSE | vel gap recovered | GT MAE |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        c, v = r["chunk"], r.get("vel", {})
        gap = f"{pct(c.get('gap_recovered'))} ± {pct(c.get('se'))}" if c.get("gap_recovered") is not None else "—"
        ci = f"[{pct(c['ci95'][0])}, {pct(c['ci95'][1])}]" if "ci95" in c else "—"
        vg = f"{pct(v.get('gap_recovered'))} ± {pct(v.get('se'))}" if v.get("gap_recovered") is not None else "—"
        lines.append(f"| {name} | {c['rel_l2']:.5f} | {gap} | {ci} | {pct(c.get('frames_improved'))} "
                     f"| {c['median_frame_rel_l2']:.5f} | {v.get('mse', float('nan')):.3e} | {vg} "
                     f"| {r['gt']['mae']:.4f} |")
    lines += ["", "## Gap recovered per test episode (chunk)", ""]
    eps = meta["test_episodes"]
    lines += ["| model | " + " | ".join(f"ep {e}" for e in eps) + " |", "|---|" + "---|" * len(eps)]
    for name, r in res.items():
        g = r["chunk"].get("gap_per_episode")
        if g:
            lines.append(f"| {name} | " + " | ".join(pct(g[str(e)]["gap_recovered"]) if str(e) in g else "—"
                                                     for e in eps) + " |")
    lines += ["", "## Gap recovered per action dimension (chunk)", ""]
    dims = len(res[B]["chunk"]["rel_l2_per_dim"])
    lines += ["| model | " + " | ".join(f"dim {k}" for k in range(dims)) + " |", "|---|" + "---|" * dims]
    lines.append(f"| {B} rel_l2 | " + " | ".join(f"{x:.4f}" for x in res[B]["chunk"]["rel_l2_per_dim"]) + " |")
    for name, r in res.items():
        g = r["chunk"].get("gap_per_dim")
        if g:
            lines.append(f"| {name} | " + " | ".join(pct(x) for x in g) + " |")
    for cb in meta.get("compare", []):
        lines += ["", f"## Gap recovered vs {cb} (chunk and velocity, paired)", "",
                  "| model | chunk gap | 95% CI | frames improved | vel gap |", "|---|---|---|---|---|"]
        for name, r in res.items():
            v = r.get("vs", {}).get(cb)
            if not v:
                continue
            c, vv = v["chunk"], v.get("vel", {})
            lines.append(f"| {name} | {pct(c.get('gap_recovered'))} ± {pct(c.get('se'))} | "
                         f"[{pct(c['ci95'][0])}, {pct(c['ci95'][1])}] | {pct(c.get('frames_improved'))} | "
                         f"{pct(vv.get('gap_recovered'))} ± {pct(vv.get('se'))} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------------

def main():
    ap = build_parser()
    args = ap.parse_args()
    policy_norm.check_args(ap, args)
    rank, world, dev = q.dist_setup(args.device)
    args.device = dev
    main0 = rank == 0
    say = print if main0 else (lambda *a, **k: None)
    t_start = time.perf_counter()
    dtype = ep.torch_dtype(args.dtype)
    ckpts = []
    for spec in args.ckpt:
        name, _, path = spec.partition("=")
        if not path or not Path(path).is_dir():
            raise SystemExit(f"--ckpt {spec!r}: expected NAME=DIR with an existing directory")
        if not (Path(path) / "hw_state.pt").exists():
            raise SystemExit(f"--ckpt {spec!r}: no hw_state.pt (not a bake_hw / smolvla_hw checkpoint)")
        ckpts.append((name, path))
    names_ = [n for n, _ in ckpts]
    base_name = args.baseline or names_[0]
    if base_name not in names_ or any(c not in names_ for c in args.compare):
        raise SystemExit(f"--baseline / --compare must name a --ckpt ({names_})")
    ep_args = argparse.Namespace(**{**vars(args), "seed": args.eval_seed})

    cfg0 = ep.load_config(args.policy, dev)
    rename = ep.camera_rename_map(cfg0, args.dataset, args.rename_map)
    ds_test, _, test_eps, idx = ep.make_data(cfg0, args.dataset, args.n_test_episodes, args.n_frames, args.eval_seed)
    try:
        norm, norm_stats = ep.setup_normalization(ep_args, cfg0, rename, ds_test, idx)
    except policy_norm.NormalizationError as e:
        raise SystemExit(f"error: {e}")
    mine = q.share(idx, rank, world)
    ds = q.prefetch(ds_test, mine, args.workers)
    episodes_local = {i: int(ds[i]["episode_index"]) for i in mine}
    say(f"{world} GPU(s); test episodes {test_eps}, {len(idx)} frames x {args.draws} draws; "
        f"frames decoded ({time.perf_counter() - t_start:.0f}s)", flush=True)

    teacher, cfg = ep.load_policy(args.policy, dev, dtype)
    teacher.requires_grad_(False)
    preprocess, postprocess = ep.build_processors(cfg, ep_args, rename, norm_stats)
    ep.require_guarded_normalization("eval_qat", cfg, preprocess, postprocess, norm)
    adim = cfg.action_feature.shape[0]
    bases = [args.eval_seed + d * DRAW_STRIDE for d in range(args.draws)]
    has_vel = not args.no_vel

    # reference: bf16 chunks (+ gt) and the teacher's full traces for the velocity metric
    ref_by_draw, gt_by_draw, traces = [], [], {}
    for b in bases:
        preds, gts, pads = [], [], []
        for i in mine:
            batch = q.make_batch(ds[i], preprocess, teacher, b + i, dev)
            tr = q.teacher_trace(teacher, batch, b + i, dev)
            if has_vel:
                traces[(i, b)] = tr
            chunk = postprocess(tr[0]).float().cpu()[0]
            gt, pad = gt_of(teacher, ds[i], chunk.shape[0])
            t = min(chunk.shape[0], gt.shape[0])
            preds.append(chunk[:t])
            gts.append(gt)
            pads.append(pad)
        ref_by_draw.append(preds)
        gt_by_draw.append((gts, pads))
    # consistency: these chunks are eval_policy.run_mode's, bit for bit (first frames of this rank)
    chk = mine[:2]
    rm = ep.run_mode(teacher, ds, chk, preprocess, postprocess, bases[0], dev)[0]
    check = {"bf16": ep.repeat_agreement(rm, ref_by_draw[0][:len(chk)])}
    local = dict([frame_records("bf16", ref_by_draw, ref_by_draw, gt_by_draw, mine,
                                [[0.0] * len(mine) for _ in bases] if has_vel else None)])
    say(f"bf16 reference done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    def evaluate(name, path, quantize):
        policy, _ = ep.load_policy(path, dev, dtype)
        if quantize:
            st = torch.load(Path(path) / "hw_state.pt")
            algos.quantize(policy, MODE, st["skip"], options={}, algo=hw.Pinned(st))
        preds, vel = [], []
        for b in bases:
            p, _, _, v = measure(policy, ds, mine, preprocess, postprocess, traces if has_vel else None,
                                 b, dev, adim, teacher)
            preds.append(p)
            vel.append(v)
        rm = ep.run_mode(policy, ds, chk, preprocess, postprocess, bases[0], dev)[0]
        check[name] = ep.repeat_agreement(rm, preds[0][:len(chk)])
        local.update(dict([frame_records(name, preds, ref_by_draw, gt_by_draw, mine, vel if has_vel else None)]))
        del policy
        torch.cuda.empty_cache()
        say(f"{name} done ({time.perf_counter() - t_start:.0f}s)", flush=True)

    for name, path in ckpts:
        evaluate(name, path, True)
        if args.ckpt_bf16:
            evaluate(f"{name}@bf16", path, False)

    # gather every rank's frames
    payload = {"recs": local, "episodes": episodes_local, "check": check}
    if world > 1:
        import torch.distributed as dist

        parts = [None] * world
        dist.all_gather_object(parts, payload)
    else:
        parts = [payload]
    if main0:
        names = list(local)
        all_recs = {n: {} for n in names}
        episodes = {}
        for p in parts:
            for n in names:
                all_recs[n].update(p["recs"][n])
            episodes.update(p["episodes"])
        frames = sorted(all_recs[base_name])
        checks = {n: all(p["check"][n]["deterministic"] for p in parts if n in p["check"]) for n in names}
        if not all(checks.values()):
            print("WARNING: the one-pass chunk differs from eval_policy.run_mode for", checks, flush=True)
        res = summarize(all_recs, names, frames, episodes, args.bootstrap, has_vel, base_name, args.compare)
        meta = {"baseline": base_name, "compare": args.compare, "test_episodes": test_eps, "frames": len(frames), "draws": args.draws, "noise_bases": bases,
                "checkpoints": dict(ckpts), "policy": args.policy, "dataset": args.dataset,
                "normalization": norm["status"], "world": world, "matches_eval_policy_run_mode": checks,
                "seconds": round(time.perf_counter() - t_start), "frames_per_episode":
                    {str(e): sum(1 for f in frames if episodes[f] == e) for e in sorted(set(episodes.values()))}}
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"meta": meta, "results": res,
                                   "per_frame": {"frames": frames, "episodes": [episodes[f] for f in frames],
                                                 "records": all_recs}}, indent=1))
        md = markdown(res, meta)
        out.with_suffix(".md").write_text(md)
        print(md)
        print("wrote", out, "and", out.with_suffix(".md"))
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    random.seed(0)
    main()
