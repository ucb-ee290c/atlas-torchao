# Top-k Fisher eval: keep the most sensitive layers in bf16

**TL;DR:** `src/scripts/eval_topk_fisher.py` tests whether a Fisher ranking is useful. It keeps the top-k
ranked linear layers in bf16, leaves every other layer in FP8, and measures how much of the FP8 loss
increase that wins back. It works on LLM reports (metric: next-token NLL) and VLA reports (metric: how far the
generated action chunk moves from the bf16 chunk). The LLM path is validated; the VLA path has a first 6-frame
result (see Status).

See also `docs/fisher_profile.md` for how the Fisher scores are produced.

## The flow, step by step

Same steps for both model kinds; only steps 2 and 4 differ.

1. **Read the Fisher report.** Take its linear-layer scores and sort them (highest first). That is the
   `fisher` ranking. Its metadata says which model, which data, and which items were already used for scoring.
2. **Pick held-out items** the scores did not use.
   - LLM: WikiText-2 windows that do not overlap a scored window.
   - VLA: robot frames (images + joint state + instruction) that are not in the report's scored frame list. Each
     frame is run with `--draws` different initial-noise seeds; one (frame, draw) is one item.
3. **Load the model once and put every linear layer on the FP8 path** (`quantize_reference`).
4. **Define "how bad is this config" for one item.**
   - LLM: mean next-token cross-entropy of the window (lower = better).
   - VLA (default `--vla-metric chunk`): run the policy's full action generation
     (`eval_policy.predict_chunk`) from a fixed initial-noise tensor, and take the relative L2 distance
     `||chunk - bf16_chunk|| / ||bf16_chunk||`. The first pass is all-bf16; its chunk is stored as the reference,
     so bf16 scores 0 and the number grows as quantization moves the actions further away.
5. **Score the configs, one after another, on the same items** (the model is not reloaded; each config just
   switches which layers are bf16): all-bf16, all-FP8, Fisher top-k for each k, optionally measured top-k, and
   random k-layer sets. The bf16 switch is `profile_fisher(True, record=False)`; the FP8 switch is
   `profile_fisher(False)`.
6. **Turn the scores into recovery** and a bootstrap error, print the table, write the JSON.

```
recovery = (loss_fp8_all - loss_config) / (loss_fp8_all - loss_bf16)       # bf16 = 0 for the VLA chunk metric
```

## The question it answers

The Fisher profile ranks layers by how much quantization error each one adds to the loss. This script
checks that ranking by the thing you care about: if you protect the top-k layers, does the model get close to
all-bf16 quality?

For a set of evaluation items it computes the mean loss for:

| Config | Meaning |
|---|---|
| `bf16` | every layer in bf16 (reference) |
| `fp8_all` | every linear layer FP8 |
| `fisher_top<k>` | top-k layers by Fisher score in bf16, the rest FP8 |
| `measured_top<k>` | same with a *measured* ranking (optional, `--measured`) |
| `random<k>_<j>` | k random linear layers in bf16 (baseline, `--random` draws per k) |

Then:

```
recovery = (loss_fp8_all - loss_config) / (loss_fp8_all - loss_bf16)
```

0% means the config is no better than all-FP8, 100% means it matches all-bf16, negative means worse than
all-FP8. The `+-` is the bootstrap standard error over the evaluation items (`--boot` resamples, default
500). It reflects which items were drawn only, not the model, data or which random layer sets were chosen.

## How a layer is held in bf16

Every `nn.Linear` is replaced by `RefQuantLinear` (`quantize_reference`). For each config the script calls
`profile_fisher(enable, record=False)` on every layer:

- layers in the keep set: profile mode, which returns the clean (unquantized) output = bf16;
- all other layers: normal FP8 forward.

No weights are changed between configs, so one loaded model serves every config.

With `--attn-fp8` the whole set of configs is repeated a second time with attention q/k/p/v also routed
through the FP8 attention path (`ref_attention.patch_attention`). Attention operands are never individually
protected: they are either all FP8 or all bf16.

## Choosing the evaluation items

The items must not be the ones the scores were computed on, or the check is circular. The script reads the
Fisher report's metadata to avoid them. The kind is read from the report: `seqlen: "n/a"` means VLA, otherwise
LLM.

| | LLM | VLA |
|---|---|---|
| Model / data | `meta.model`, WikiText-2 train | `meta.model` (policy) + `meta.dataset` |
| Items | `--n` windows of `seqlen` tokens, sampled with `--seed` (default 777) | `--n` frames x `--draws` noise/time draws |
| Excluded | any window within `seqlen` tokens of a scored window | frames whose index is in `meta.calibration.frame_indices` |
| Metric per item | mean next-token cross-entropy | `chunk` (default): relative L2 of the generated action chunk from the bf16 chunk. `flow_loss`: the policy's flow-matching training loss |
| Fixed randomness | none needed | each (frame, draw) gets its own seed: the initial action noise (`chunk`) or the noise and time `t` (`flow_loss`) is the same for every config |
| Default exclude | `lm_head` stays bf16 | none |

**Why the VLA default is the chunk distance and not the training loss.** The flow-matching loss is one noisy
denoising step against a random target. A tiny numerical change moves it up or down by a few thousandths
whichever way it falls: on 2 frames FP8 changed it by -0.0028 on one frame and +0.0068 on the other (mean
+0.002), so keeping 3 layers in bf16 scored *worse* than all-FP8 and recovery read -308%. The loss is
reproducible under a fixed seed and the bf16 pass-through matches the unquantized model exactly, so nothing is
broken; the metric is just too noisy for an effect this small. The chunk distance is zero or positive and grows
with the quantization damage, so the ratio is stable. `--vla-metric flow_loss` is kept for comparison.

The chunk distance measures how far FP8 moves the actions, not whether they get closer to the recorded
actions. `eval_policy.py` reports the error against ground truth separately; this script does not.

## Running it

Use `atlas-torchao` from the repo root. `lerobot` is needed for VLA.

```
# LLM
python src/scripts/eval_topk_fisher.py --fisher src/out/fisher/tinyllama/tinyllama_n128.json \
    --ks 10 5 3 --device cuda:1
python src/scripts/eval_topk_fisher.py --fisher ... --attn-fp8

# VLA
python src/scripts/eval_topk_fisher.py --fisher src/out/fisher/smolvla/smolvla_n8.json \
    --norm-stats dataset --ks 10 5 3 --device cuda:1
```

| Flag | Default | Meaning |
|---|---|---|
| `--fisher` | required | report JSON from `profile_fisher_llm.py` / `profile_fisher_vla.py` |
| `--ks` | 10 5 3 | how many top layers to keep in bf16 |
| `--n` | 64 (LLM), 32 (VLA) | held-out windows (LLM) or frames (VLA); the `+-` shrinks ~ 1/sqrt(n) |
| `--seed` | 777 | held-out sampling seed |
| `--random` | 0 | random layer sets per random-k, a chance baseline. Off by default: the run is just bf16, all-FP8 and the top-k configs. `--random 3` adds 3 sets |
| `--random-ks` | largest of `--ks` | which k values get random baselines. Random-10 is the strictest baseline (most layers kept), so beating it implies beating the smaller ones; pass all of `--ks` for one per k |
| `--measured` | none | `validate_*_actual.json`: adds a measured top-k ranking |
| `--attn-fp8` | off | also run with attention in FP8 |
| `--exclude` | `lm_head` (LLM), none (VLA) | fqn substrings kept in bf16 in every config |
| `--boot` | 500 | bootstrap resamples |
| `--device` | auto (LLM), `cuda` (VLA) | |
| `--out` | `<fisher dir>/validation/topk_<fisher stem>.json` | |

VLA only: `--vla-metric chunk|flow_loss` (default `chunk`), `--draws` (default 1; more frames beat more draws), `--n-test-episodes` (5), `--eval-seed` (0), `--dtype`,
`--rename-map`, and the normalization flags (`--norm-stats dataset`). These must match what the profile run
used, so the same held-out episodes are excluded.

Shell wrappers with these defaults: `bash src/scripts/run_topk_fisher_llm.sh` and
`bash src/scripts/run_topk_fisher_vla.sh` (env overrides `GPU`, `N`, `KS`, `RANDOM_DRAWS`, `FISHER`, `OUT`,
`PYTHON`; extra arguments pass through). Each also writes a `.log` next to the JSON.

Choosing `--n`: with 16 windows the LLM recovery had a `+-` of about 9 points; the defaults aim for about 4 to
6 (64 windows, 32 frames). Fisher scoring itself is stable at n=16 (n=16 vs n=128: Spearman 0.992, same top
10), so spend the budget on the eval items and on `--random` draws, not on more scoring windows.

## Output

The terminal prints each config as it finishes, then a summary:

```
[linear_fp8]  all-FP8 loss 2.08241  (+0.01634)
  fisher_top3      loss 2.07187  recovery  64.5% +- 6.3%
  random3          mean -28.8%  range -28.8..-28.8%
```

The JSON holds `meta` (kind, data, items, seed, metric), `rankings` (the top-max(k) names of each ranking),
`per_item_loss` (every config, every item, for each variant) and `summary` (the recovery table). It is
rewritten after every config, so a crash keeps the finished configs, and `summary` is `null` until the run
completes.

## Reading the result

- Two rows whose `+-` ranges overlap can't be told apart. Compare the Fisher row with the random-k range for
  the same k first: a ranking that does not beat random has not shown anything.
- The percentage depends on how large the FP8 loss gap is. If `fp8_all` is barely above `bf16`, recovery is a
  ratio of two small, noisy numbers and the bootstrap error will be large (or undefined if the gap is zero).
- Per-item losses are in the JSON if you want paired comparisons or your own plots.

## Status and limits

- **LLM:** validated on TinyLlama / WikiText-2. With 16 windows, Fisher top-10 / 5 / 3 in bf16 recovered
  65% / 63% / 57% of the FP8 loss increase (linear layers FP8), against -1% to 19% (mean over 5 draws) for random sets of the
  same size. See `docs/fisher_profile.md` for the full table.
- **VLA (chunk metric, first result):** 6 frames x 1 draw, `smolvla_n8.json` scores. All-FP8 moves the action
  chunk by 0.108 (relative L2 from bf16). Keeping Fisher top-10 / 5 / 3 in bf16 gives 0.059 / 0.071 / 0.096,
  i.e. recovery 45% / 34% / 11%. Random 10-layer sets gave 1% / 36% / 2% and one random 5-layer set -2%. So
  Fisher top-10 beats two of three random sets and is matched by the best one: suggestive, not established.
  That run hit a time limit after 8 of 15 configs, so the remaining random draws, the `+-` errors and the
  summary are missing. Rerun with more frames and a longer time limit before relying on it.
- VLA held-out frames come from the same calibration episodes as the scored ones, just different frame
  indices. Neighbouring frames are correlated, so this is a weaker held-out check than the LLM one.
- Only linear layers are individually protected; attention operands move together (`--attn-fp8`).
- With few items the bootstrap error is large; with `--n 2` it says almost nothing.
