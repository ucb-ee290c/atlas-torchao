# QAT for SmolVLA (`ref_conv` FP8)

## TL;DR

`smolvla.py` fine-tunes SmolVLA so it loses less accuracy under the `ref_conv` FP8 profile. The forward
pass uses the real `ref_conv` code with the casts made differentiable (straight-through estimator, STE).
A bit-exact parity check against `ref_conv` runs before training. The loss distils the original bf16
policy, and the output is a plain lerobot checkpoint for the unchanged `eval_policy.py`.
**Status: written and byte-compiled, never run.** Next step: `--parity-only`.

## Files

| File | What it is |
|---|---|
| `smolvla.py` | The QAT script (data split, parity check, training, save, test) |
| `view_smolvla.py` | Checkpoint viewer: parameter counts by prefix, tensor list, module tree |
| `model.txt` | `print(policy)` of `/rscratch/agarcia/smolvla_base` (module names and shapes) |

## Why QAT, and what carries over from TinyLlama

FP8 post-training quantization moves SmolVLA's action chunks away from the bf16 ones. Fisher top-k
bf16 protection recovered only part of that: 37% at 16% of the weights, 62% at 33%, because the
error is spread over many layers. QAT tries the other route: train the weights so they hold up
under rounding.

Lessons from the TinyLlama QAT runs (`atlas-torchao-tile-aware/scripts/qat_tinyllama.py`):

- **The training forward must be bit-identical to the eval forward.** An fp32 + autocast forward
  gave no gain. So the model stays in bf16 as `load_policy` loads it, the optimizer keeps fp32
  master weights, and the masters are rounded back into the model after every step.
- **Scale the lr to the gap.** On TinyLlama, 1e-5 diverged and 3e-6 worked; the default here is 3e-6.
- **Use a paired val with at least 16 samples, and quote held-out test numbers.** The train-split
  val read low on TinyLlama.

## How it works

1. **Student = the real `ref_conv` model, made differentiable.**
   `quant_modes.quantize(policy, "ref_conv")` picks and patches the layers exactly as the eval does.
   - Every `RefQuantLinear` / `RefQuantConv2d` is swapped for `QATLinear` / `QATConv2d`. These have
     the same forward, but the weight is cast on every call (the originals cache the codes once),
     and the cast has an STE gradient: identity, zero where the value saturated (|x| > 448·2^s).
   - The attention patches stay. Inside `qat_numerics()`, `ref_attention._cast` is swapped for an
     STE version, so both attention paths keep their exact masks and softmax rules: text/expert
     (`eager_attention_forward`) and vision. The FP8 Q/K/P/V values are the same, and the
     matmuls stay in fp32 as in the eval.
2. **Parity check.** On a few val frames, the real `ref_conv` model and the QAT student (no
   training) both go through `eval_policy.run_mode`. They must match bit for bit, or the
   script stops (`--allow-parity-fail` overrides).
3. **Training forward = the eval forward.** `run_chunk` is the body of `predict_action_chunk`
   without `no_grad`: a prefix pass with KV cache, then 10 Euler denoise steps. `denoise_step` is
   intercepted to record the teacher and to apply teacher forcing to the student.
4. **Loss: distil the frozen bf16 teacher (no action labels).**
   - `vel` (default): at one random Euler step, the student denoises the teacher's `x_t`; MSE
     against the teacher's velocity. Cost: prefix pass + one step.
   - `vel-all`: the same at all 10 steps.
   - `chunk`: the student integrates freely from the same noise; MSE of the final chunks. This is
     exactly what the eval's agreement metric measures.
5. **Optimizer.** AdamW on fp32 masters, 8 frames per optimizer step (`--accum`), warmup + cosine
   schedule, gradient clip at 1. Embeddings and `lm_head` are frozen; `--train-only REGEX`
   restricts training further (e.g. `'lm_expert|action_'` for expert-only). The log records whether
   any weight scale exponent changed.
6. **Save.** The trained weights go into a freshly loaded clean policy (no QAT modules or patches),
   saved to `<out-dir>/checkpoint` with the processor files copied in. `qat_log.json` goes next to it.

## Data (`lerobot/svla_so100_pickplace`: 50 episodes, 19,631 frames, 30 fps)

Split by episode, so neighbouring frames never leak between splits:

| Split | Episodes | Frames | Used for |
|---|---|---|---|
| test | the 5 `eval_policy` holds out (same shuffle, `--eval-seed`) | `--n-frames` 200 | `--test` only |
| val | 4 whole episodes from the rest | `--val-n` 32 | progress, % gap recovered |
| train | the remaining ~41 | `--train-n` 2,000 | training, with fresh noise every visit |

The recorded actions are never used. FP8 barely changes accuracy against them (0.695 vs 0.692), so
the signal is in matching the bf16 policy.

## Metric

Relative L2 of the action chunk (dataset units) against the **original** bf16 policy, on the same
frames and noise. % gap recovered = 1 − err / err(step 0), with a paired per-frame SE.

Note: `eval_policy.py --modes bf16 ref_conv` on the QAT checkpoint compares against that
checkpoint's own bf16, which is no longer the original model. Use `--test` (or a compare script) for
the gap number, and `eval_policy` for the standard accuracy-vs-recorded-actions report.

## Running (in order)

```bash
# 1. parity check + step-0 val only
python src/qat/smolvla.py --policy /rscratch/agarcia/smolvla_base \
    --dataset lerobot/svla_so100_pickplace --norm-stats dataset \
    --out-dir src/out/qat/smolvla_parity --parity-only

# 2. a short smoke run (do the loss, gradients and timing look sane?)
python src/qat/smolvla.py ... --out-dir src/out/qat/smolvla_smoke --max-steps 5 --val-every 5

# 3. the real run, with the no-quant control
python src/qat/smolvla.py ... --out-dir src/out/qat/smolvla_vel --loss vel --lr 3e-6 --epochs 2 --test
python src/qat/smolvla.py ... --out-dir src/out/qat/smolvla_noquant --no-quant --test
```

## Open questions / check on the first run

- **Parity.** Expected bit-exact, but not yet verified: the patch conv's `F.unfold`, and the fp32
  attention matmul on the STE values against `dq_matmul_nt`.
- **Gradients through the KV cache.** In `vel` mode the vision and text towers get gradients only
  through the prefix KV cache. If their gradients are zero, lerobot's cache detaches somewhere.
- **Speed.** Each step decodes video and runs a full teacher trace. If that's too slow, cache the
  teacher traces or decode the frames once.
- **lr / loss choice.** Untested on this model. Sweep lr around 1e-6 to 3e-6, then compare `vel` with `chunk`.
