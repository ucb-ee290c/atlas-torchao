"""Calibration data for the (f) calibrate hook (NEW; docs/algos/README.md "Calibration rules").

Calibration data never overlaps the evaluation data:
  TinyLlama        WikiText-2 *train* split (the evaluation reads the test split)
  SmolVLA / DiT    frames from episodes outside the held-out test episodes of eval_policy.make_data
  Pi0              DROID frames from episodes disjoint from the evaluated ones, or dummy
                   observations drawn with a different seed

Each loader returns (batches, record) with record = {source, n_samples, seed, ...}; the eval CLIs
pair it with a run(model, batch) forward and hand both to algos.quantize, which stores the record
under the result's `algo.calibration`.
"""

from __future__ import annotations

import random

DEFAULT_N = {"llm": 16, "policy": 32, "pi0": 8}


def wikitext2_train(model_id, n, seqlen, seed, use_fast=False, tokenizer=None):
    """n random windows of `seqlen` tokens from the WikiText-2 train split, joined as the
    collaborator's loader joins it (datautils.get_wikitext2: "\\n\\n".join of every line), with a
    local RNG (datautils seeds the global `random`). use_fast follows the eval protocol (default:
    slow tokenizer; mxq: fast)."""
    from datasets import load_dataset

    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=use_fast)
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    starts = [rng.randint(0, ids.shape[1] - seqlen - 1) for _ in range(n)]
    batches = [ids[:, i:i + seqlen] for i in starts]
    return batches, {"source": "Salesforce/wikitext wikitext-2-raw-v1 train", "n_samples": n,
                     "seed": seed, "seqlen": seqlen, "tokenizer_fast": use_fast, "window_starts": starts}


def lerobot_frames(policy_cfg, dataset, n_test_episodes, eval_seed, n, seed, to_batch):
    """n frames from the episodes eval_policy.make_data does NOT hold out (same shuffle, same
    eval_seed), sampled with `seed`. to_batch(sample) -> the policy's preprocessed batch."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(dataset)
    episodes = list(range(meta.total_episodes))
    random.Random(eval_seed).shuffle(episodes)                 # eval_policy.make_data
    test_eps = set(episodes[:n_test_episodes])
    calib_eps = sorted(episodes[n_test_episodes:])
    assert not test_eps & set(calib_eps)
    if not calib_eps:
        raise ValueError(f"{dataset}: no episodes left for calibration after {n_test_episodes} test episodes")
    ds = LeRobotDataset(dataset, episodes=calib_eps, delta_timestamps=resolve_delta_timestamps(policy_cfg, meta))
    idx = sorted(random.Random(seed).sample(range(len(ds)), min(n, len(ds))))
    batches = [to_batch(ds[i]) for i in idx]
    return batches, {"source": f"{dataset} episodes outside the {n_test_episodes} test episodes",
                     "n_samples": len(idx), "seed": seed, "episodes": calib_eps,
                     "excluded_test_episodes": sorted(test_eps), "frame_indices": idx}


def pi0_droid(n_eval, n, horizon, frame_idx, eval_seed, eval_frames=()):
    """n DROID frames from the episodes after the evaluated ones in pi0_model.droid_observations'
    seeded order (so disjoint from them by construction; checked against `eval_frames`)."""
    import pi0_model

    frames = pi0_model.droid_observations(n_eval + n, horizon, frame_idx, eval_seed)[n_eval:]
    used = {f["meta"]["episode"] for f in eval_frames}
    got = [f["meta"]["episode"] for f in frames]
    if used & set(got):
        raise ValueError(f"calibration episodes overlap the evaluated ones: {sorted(used & set(got))}")
    return frames, {"source": f"{pi0_model.DROID_REPO} episodes disjoint from the evaluated ones",
                    "n_samples": len(frames), "seed": eval_seed, "episodes": got,
                    "excluded_eval_episodes": sorted(used), "frame_index": frame_idx}


def pi0_dummy(n, seed, eval_seed=None, model_level=False, cfg=None, device="cpu"):
    """n dummy observations (the reference's generators) with `seed`, which must differ from the
    evaluation's seed when the evaluation is itself on dummy observations."""
    if eval_seed is not None and seed == eval_seed:
        raise ValueError(f"calibration seed {seed} equals the evaluation seed: the dummy observations "
                         "would be the evaluated ones")
    if model_level:
        import eval_pi0

        frames = eval_pi0.mxu_dummy_observations(n, cfg, device, seed)
        src = "mxu_dummy (run_eval_mxu_io.py _make_dummy_obs)"
    else:
        import pi0_model

        frames = pi0_model.dummy_observations(n, seed)
        src = "dummy (run_output_eval.py _make_dummy_obs)"
    return frames, {"source": src, "n_samples": n, "seed": seed}
