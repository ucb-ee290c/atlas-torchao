# Main experiment: QAT of SmolVLA under the FX export's hardware recipe

**TL;DR:** We measure how much QAT improves `lerobot/smolvla_libero`, with and without AWQ+GPTQ,
under the exact FP8 numerics the FX exporter ships: `ref_conv` + `atlas_v1_awq_tile_static`. That
means per-32-row panel weight exponents, static activation exponents, attention P at exponent −8 and
Q/K/V reference-dynamic. Every training forward equals the recipe's model bit for bit, and every
trained model deploys with the same pinned exponents.

## Design (2×2 + control), `lerobot/smolvla_libero` on `lerobot/libero`

| | no training | QAT (3 seeds) |
|---|---|---|
| **round-to-nearest**: `granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib` | **A** | **C** |
| **AWQ+GPTQ**: `atlas_v1_awq_tile_static` (what the FX export ships) | **B** | **D** |
| control: D's training with every cast off (`--no-quant`), evaluated with B's pinned recipe | | **E** |

- A → B: the AWQ+GPTQ gain.
- A → C: QAT alone.
- **B → D: QAT on top of what ships (the headline).**
- C vs D: QAT with vs without AWQ+GPTQ underneath.
- D vs E: training through FP8 vs plain fine-tuning.

## Pipeline (files in `src/qat/`; `source src/qat/hw_env.sh` first: caches on /rscratch)

1. `bake_hw.py --recipe R --out-dir DIR` runs the real recipe through `algos.quantize`, using
   eval_policy's calibration protocol (the protocol the FX export uses). It records every decision
   (`hw_recipe.Recorder`, `weight_state`) and writes `DIR/checkpoint` (bf16 weights = the recipe's
   dequantized codes, plus the AWQ folds) and `hw_state.pt`. It then checks that
   **checkpoint + `hw_recipe.Pinned` == the recipe's model, bit for bit**; on failure the checkpoint is
   removed.
2. `smolvla_hw.py --init DIR/checkpoint` is `smolvla.py` (same split, teacher, loss, optimizer,
   torchrun) with:
   - the student = the checkpoint quantized with `Pinned`;
   - Linear / Conv2d / attention casts wrapped in an STE (`fq_act`, `fq_weight`, `_qat_cast`);
   - weights re-cast every step at their **pinned** exponents (saturation masked in the backward pass;
     `w_sat` is logged).

   Parity must hold before training. The saved checkpoint carries `hw_state.pt`.
3. `eval_hw.py` is `eval_qat.py` with every model quantized by its own pinned recipe. Gap recovered is
   paired against `--baseline` (A); `--compare B` adds gaps vs B.

## Hyperparameters (unchanged from the ref_conv study)
`--loss vel --lr 3e-6 --epochs 2 --accum 8`, with embed_tokens and lm_head frozen; 2,000 train
frames; 128 val frames. `--n-test-episodes 32 --eval-seed 0` (the FX capture protocol). Seeds 0/1/2
with train-seeds 1234/1235/1236; the control uses seed 0 / 1234.

## Caveats
- Calibration follows eval_policy: frames from the non-test episodes. These can include the 4 val
  episodes, which affects val readings only, never test.
- Static activation exponents are fixed at calibration and are not re-calibrated after QAT. That is
  intended: deployment keeps the same pinned state.
- Gap recovered is on squared error.
- `smolvla_libero` was fine-tuned on LIBERO, so the test episodes may have been in its training data.
  That's fine for fidelity to bf16.
