# Fisher-delta sensitivity profile (LLM and VLA)

**TL;DR:** One forward + backward pass per calibration sample scores every linear layer and every attention
operand (q, k, p, v) by `0.5 * sum(grad^2 * delta^2)`, where `delta` is the FP8 rounding error. The score is
a cheap proxy for "how much does the loss move if this layer goes to FP8", so you can pick which layers to
keep in bf16 without quantizing them one at a time. Validated on TinyLlama/WikiText as a coarse top-k flag
(keeping its top-10 in bf16 recovers ~65% of the FP8 loss increase); the tail ranking and the VLA scores are
not validated.

## What is computed

For a linear `y = x W^T`, with `g = dL/dy`:

| Term | Gradient | Perturbation | Score |
|---|---|---|---|
| weights (`fisher_w`) | `gW = g^T x` | `dW = dequant(W_q) - W` | `0.5 * sum(gW^2 * dW^2)` |
| input activations (`fisher_x`) | `gx = g W` | `dx = dequant(quant(x)) - x` | `0.5 * sum(gx^2 * dx^2)` |

`fisher = fisher_w + fisher_x`. The 0.5 is the second-order Taylor term, `0.5 * d^T F d` with `F` the diagonal
empirical Fisher (`g^2`). The first-order term `g * delta` is **not** in the profiler; it has zero mean over
random-sign errors but is not zero for a fixed quantizer, so the score can differ from the true loss change.

Attention operands q, k, p (softmax output) and v get the same formula with the gradient of the operand
tensor. Under GQA, k and v are hooked after expansion across heads, so they are counted per query head and
cross terms are dropped.

Scores are per sample, averaged over the calibration batches (`fisher_report(n_batches)`).

## How it works (`src/reference/ref_profile.py`)

- `RefQuantLinear.profile_fisher(enable, record)` switches a layer to **profile mode**. In profile mode the
  forward returns the *clean* (unquantized) output, because the FP8 cast is not differentiable and any layer
  left on the FP8 path would block gradients to everything upstream.
- `record=True` accumulates the score in a backward hook on the output; `record=False` keeps the clean
  pass-through but records nothing (layers outside the selection).
- `profile_fisher_model(model, batches, loss_fn, include=None, exclude=(), anchors=None, attention=True)` does
  the whole run: select layers (`include` is a regex over fqn, `exclude` is substrings), put every other
  `RefQuantLinear` in clean pass-through, enable gradient anchors, run `loss_fn` + `backward()` per batch,
  collect `{fqn: {macs, fisher_w, fisher_x, fisher}}` plus attention entries `{"kind": "attn", "fisher"}`,
  then restore `requires_grad` and zero grads.
- **Gradient anchors**: the model has frozen weights, so a parameter at each input path is given
  `requires_grad` to put activations in the graph (LLM: the input embedding; SmolVLA: vision
  `patch_embedding`, text `embed_tokens`, `state_proj`, `action_in_proj`).
- Attention is profiled in `src/reference/ref_attention.py`: `_cast` records the operand error in a hook
  when profiling is on, and `new_forward()` / `attn_fisher()` reset and read the accumulators.
- `write_report(res, meta, out)` writes `{"meta": ..., "layers": ...}` JSON.

## Running it

Use the env that has the dependencies (`atlas-torchao`; `lerobot` is needed for the VLA). Run from the repo
root. Plots need matplotlib (the `atlas-npu` env has it).

### LLM (TinyLlama + WikiText-2)

```
CUDA_VISIBLE_DEVICES=1 python src/scripts/profile_fisher_llm.py --n 16 --out src/out/fisher_tinyllama.json
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | TinyLlama-1.1B-Chat | local path or HF id |
| `--seqlen` | 512 | window length |
| `--n` | 16 | WikiText-2 train windows |
| `--seed` | 1234 | window sampling |
| `--include` | none | regex over layer fqns |
| `--exclude` | `lm_head` | substrings left out |
| `--no-attn` | off | skip q/k/p/v |
| `--top` | 10 | rows printed |

Loss is summed cross-entropy (`reduction="sum"`), so scores are per window, not per token.

### VLA (SmolVLA via lerobot)

```
python src/scripts/profile_fisher_vla.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
    --norm-stats dataset --n 8 --draws 4 --out src/out/fisher_smolvla.json
```

- Reuses `eval_policy`'s loaders and held-out episode split (`--n-test-episodes`, `--eval-seed`), so
  calibration frames never come from held-out episodes.
- The flow-matching loss samples noise and a time `t`, so each (frame, draw) pair is its own job under its
  own seed (`--n` frames x `--draws`). Scores are averaged over jobs (not the gradients, which would square a
  mean).
- Loss is `policy.forward(batch)[0] * action.numel()`.
- Each entry carries `tower`: `vision | language | expert | action_head`.
- Attention is called 24 times per forward (8 self + 8 x 2 cross); entries are tagged `.L{i}`.
- Some entries are exactly zero because the action loss never uses them: the last language layer's
  q/o/MLP, `lm_head`, and attention at L22.

### Plots

```
python src/scripts/plot_fisher.py src/out/fisher_tinyllama.json
python src/scripts/plot_fisher.py src/out/fisher_smolvla.json --tower language
```

Linear and attention data are always plotted separately. Scatter plots are log scale (10^x tick labels), each
axis normalized by its own total, zero entries dropped. Heatmaps are per layer, one per tower for the VLA.

## What was validated (TinyLlama, WikiText-2)

| Check | Result |
|---|---|
| Stability, n=16 vs n=128 | Spearman 0.992; top-10 linear layers identical (10/10) |
| Per-layer ground truth (quantize one layer, measure loss change, 32 held-out windows) | too noisy to rank the tail: split-half reliability 0.09; Spearman with Fisher +0.13; top-10 overlap 5/10 |
| End-to-end top-k (16 held-out windows) | see below |

End-to-end: keep the Fisher top-k linear layers in bf16, the rest FP8, and compare with all-bf16.
Recovery = `(L_allFP8 - L_config) / (L_allFP8 - L_bf16)` on per-window NLL; `+-` is the bootstrap standard
error over the 16 windows. Baseline perplexity 8.662 (bf16), 8.794 (all-FP8 linear).

| Config (linear FP8) | k=10 | k=5 | k=3 |
|---|---|---|---|
| Fisher top-k in bf16 | 65% +- 9% | 63% +- 9% | 57% +- 11% |
| Measured (single-layer) top-k | 67% | 62% | 52% |
| Random k (5 draws, mean) | -1% | 19% | 12% |

With attention also FP8 (all-FP8 ppl 8.810): Fisher 42% / 36% / 27%; measured 29% / 26% / 6%; random
negative on average.

**Conclusion.** Use it as a coarse flag for which few layers to protect. It matches the measured top-k and
beats random. It does not rank the tail reliably, because the ground truth there is itself noise-level.

## Limits

- Only TinyLlama / WikiText is validated; other models and the VLA scores are not.
- The VLA run has been done once; `pi0` is untested.
- The second-order diagonal score ignores cross terms and the first-order term.
- Norms and other VPU ops are not profiled.
- Scores come from the clean (unquantized) forward, so they do not include the effect of error from earlier
  quantized layers.
