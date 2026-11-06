"""Fisher-delta profile of a LeRobot VLA (SmolVLA) under the reference FP8 profile.

    python src/scripts/profile_fisher_vla.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \\
        --norm-stats dataset --n 8 --draws 4 --out src/out/fisher/smolvla/smolvla_n8.json
    python src/scripts/plot_fisher.py src/out/fisher/smolvla/smolvla_n8.json --tower language

Needs the environment that has `lerobot`. UNTESTED: written against eval_policy.py's loaders; the loss
call and the anchor names below are the parts to check on a first run (see --help).

Loss: the policy's flow-matching loss on a calibration frame (episodes eval_policy holds out are never
used). The loss samples noise and a time t, so every (frame, draw) pair is its own job, run under its
own seed; scores are averaged over jobs (not the gradients, which would square a mean).

Entries carry "tower": vision | language | expert | action_head | joint_attn, so towers can be compared or
plotted separately. SmolVLA's attention runs through one shared function; smolvla_attention_split.py assigns
each call to language (the VLM's own attention), expert (its cross-attention) or joint_attn (one attention
over both) from the call order."""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

TOWERS = [("lm_expert", "expert"), ("vision_model", "vision"), ("text_model", "language"),
          ("action_in_proj", "action_head"), ("action_out_proj", "action_head"),
          ("action_time_mlp", "action_head"), ("state_proj", "action_head"),
          ("connector", "vision"), ("vlm_with_expert", "language")]      # first match wins (layer_rmse.py)
# Where gradients enter the graph: one parameter per input path (module-name suffixes).
ANCHOR_SUFFIXES = ("vision_model.embeddings.patch_embedding", "text_model.embed_tokens",
                   "state_proj", "action_in_proj")


def tower_of(name):
    return next((t for key, t in TOWERS if key in name), "other")


def find_anchors(policy):
    mods = dict(policy.named_modules())
    out, missing = [], []
    for suf in ANCHOR_SUFFIXES:
        hit = [m for n, m in mods.items() if n.endswith(suf) and hasattr(m, "weight")]
        (out.extend(m.weight for m in hit) if hit else missing.append(suf))
    if missing:
        print(f"WARNING: no module ends with {missing}; layers fed only through them get zero scores. "
              f"Check names: {[n for n, _ in policy.named_modules() if 'embed' in n or 'proj' in n][:12]} ...")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", required=True, help="HF repo id or local checkpoint dir")
    ap.add_argument("--dataset", required=True, help="LeRobot dataset repo id")
    ap.add_argument("--n-test-episodes", type=int, default=5, help="episodes held out (as eval_policy)")
    ap.add_argument("--eval-seed", type=int, default=0, help="eval_policy's --seed: fixes the held-out split")
    ap.add_argument("--n", type=int, default=8, help="calibration frames")
    ap.add_argument("--calib-seed", type=int, default=1234)
    ap.add_argument("--draws", type=int, default=4, help="noise / time draws per frame")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--rename-map", type=__import__("json").loads, default=None)
    ap.add_argument("--include", default=None, help="regex over layer fqns, e.g. 'lm_expert' or 'text_model'")
    ap.add_argument("--exclude", nargs="*", default=[], help="fqn substrings left unquantized and unprofiled")
    ap.add_argument("--no-attn", action="store_true")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None,
                    help="report JSON (default: src/out/fisher/<policy>/<policy>_n<N>.json)")
    import loaders.policy_norm as policy_norm
    policy_norm.add_args(ap)
    a = ap.parse_args()
    if a.out is None:
        tag = re.split(r"[-_]", Path(a.policy).name)[0].lower()
        a.out = str(REPO / "src" / "out" / "fisher" / tag / f"{tag}_n{a.n}.json")
    a.seed = a.eval_seed                      # the name eval_policy's helpers read

    import torch

    import eval_policy as ep
    import reference.ref_attention as ra
    from algos import calib
    from reference.ref_profile import profile_fisher_model, quantize_reference, write_report

    dtype = ep.torch_dtype(a.dtype)
    cfg0 = ep.load_config(a.policy, a.device)
    rename = ep.camera_rename_map(cfg0, a.dataset, a.rename_map)
    ds, _, _, idx = ep.make_data(cfg0, a.dataset, a.n_test_episodes, 1, a.eval_seed)
    norm, norm_stats = ep.setup_normalization(a, cfg0, rename, ds, idx)
    policy, cfg = ep.load_policy(a.policy, a.device, dtype)
    preprocess, _ = ep.build_processors(cfg, a, rename, norm_stats)
    frames, rec = calib.lerobot_frames(cfg, a.dataset, a.n_test_episodes, a.eval_seed, a.n, a.calib_seed,
                                       lambda s: ep.to_batch(s, preprocess))
    jobs = [(b, a.calib_seed * 1000 + 17 * i + d) for i, b in enumerate(frames) for d in range(a.draws)]

    print(quantize_reference(policy, skip=tuple(a.exclude)))
    if not a.no_attn:
        print(len(ra.patch_attention(policy)), "attention sites routed through ref_attention")

    def loss_fn(m, job):
        batch, seed = job
        with ep.seeded_rng(seed, a.device):               # noise / t drawn here, then restored
            out = m.forward(dict(batch))                  # lerobot: (loss, info); check on first run
        loss = out[0] if isinstance(out, (tuple, list)) else out
        return loss * batch["action"].numel()

    anchors = lambda m: find_anchors(m)
    res = profile_fisher_model(policy, jobs, loss_fn, include=a.include, exclude=tuple(a.exclude),
                               anchors=anchors, attention=not a.no_attn)
    for n, r in res.items():
        r["tower"] = tower_of(n)
    from smolvla_attention_split import split_joint_attention
    split = split_joint_attention(res, getattr(policy.config, "self_attn_every_n_layers", 2))
    if split:
        print(f"attention calls by site: {split}")
    bad = [n for n, r in res.items() if not 0 < r["fisher"] < float("inf")]
    print(f"{len(res)} entries; zero / non-finite: {len(bad)} {bad[:8]}")
    tot = sum(r["fisher"] for r in res.values())
    for t in sorted({r["tower"] for r in res.values()}):
        s = sum(r["fisher"] for r in res.values() if r["tower"] == t)
        print(f"  {t:12s} {100 * s / tot:5.1f}% of total fisher")
    for n, r in sorted(res.items(), key=lambda kv: -kv[1]["fisher"])[:a.top]:
        print(f"{n:80s} fisher={r['fisher']:.3e}" + (" (attn)" if r.get("kind") == "attn" else ""))
    write_report(res, {"model": a.policy, "dataset": a.dataset, "n": a.n, "draws": a.draws,
                       "seqlen": "n/a", "seed": a.calib_seed, "include": a.include, "calibration": rec,
                       "loss": "policy flow-matching loss x action numel (sum convention)",
                       "normalization": norm["status"],
                       "attention_split": "joint / vlm / expert by call order (src/scripts/smolvla_attention_split.py)"
                       if split else None}, a.out)


if __name__ == "__main__":
    main()
