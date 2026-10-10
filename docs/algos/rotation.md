# `--algo rotation`: Hadamard rotations of matmul inputs

`--algo rotation` applies QuaRot / SpinQuant-style Hadamard rotations under the Atlas FP8 numerics:
- an online Hadamard on the inputs of `down_proj` / `o_proj`,
- the per-head H_hd folded into `v_proj` / `o_proj`,
- and optionally the residual-stream rotation R1 with the RMSNorm fold.

Code: `src/algos/rotation.py`. Tests: `tests/test_algo_rotation.py`.

**Status: weak keep (CPU screening only).**
- Under dynamic or calibrated per-tensor po2 scales, the online Hadamard lowers the worst-layer error: TinyLlama L2 down_proj rel RMSE goes from .051 to .033.
- It moves perplexity only from +0.84% to +0.47%, which is within the noise of 4 windows.
- R1 makes perplexity worse (+1.4%).
- Under the collaborator's fixed 2^0 scale with flush-to-zero, every rotation is catastrophic. The module hooks do not reach collab_v0_hw anyway.

## Sources
- **QuaRot, arXiv 2404.00456.**
  - Sec. 3.4 and Sec. 4 stage 1a: fold the RMSNorm scale into the next linears, then rotate x → xQ. This is valid only "so long as no re-scaling happens in the RMSNorm block".
  - Stage 1b: an online Hadamard before down_proj, computed "in full precision".
  - Stage 1c, Eq. 8–10: per-head H_dh folded offline into W_v / W_out, plus an online cross-head H_nh before o_proj.
  - Sec. 3.1: non-power-of-two sizes are built as H_2^n ⊗ H_m.
  - App. A.10: overhead.
  - App. A.5: Hadamard rotations beat random orthogonal ones.
- **SpinQuant, arXiv 2405.16406.**
  - Sec. 3.1–3.2: R1 and R2 are mergeable; R4 is the online down_proj Hadamard.
  - Sec. 4.2: the online Hadamard gives "marginal benefit" at 8-bit activations.
  - Learned rotations (Cayley SGD) are **not** implemented.
- **Paley construction I** (order q+1 for a prime q ≡ 3 mod 4) supplies H_m for non-power-of-two factors, e.g. 5632 = 44·128 and 96 = 12·8. NEW: standard math, with the matrices checked in the tests.
- **Operand casts** are the reference vpack and the po2 rule through `algos.base.cast_with` (`ref_numerics.py`, a port of pi0-quant `hw_float_ops.py:38-57,64-103`).

## Parameters (`--algo rotation:k=v,...`)

| Parameter | Default | Meaning |
|---|---|---|
| `online` | `down,o` | Layers with an online Hadamard on their input (`*.down_proj`, `*.o_proj`), or `none`. |
| `impl` | `fp32` | `fp32`: orthonormal H in float32, the upper bound. No Atlas unit runs it, so it is counted as host work.<br>`vpu`: ±1 butterflies across 32-wide feature tiles, with truncating vadd/vsub. NEW.<br>`mxu2`: two FP8 matmuls against ±1. NEW. |
| `fold_head` | `true` | Fold the per-head H_hd into v_proj rows and o_proj columns, in `prepare`. |
| `r1` | `false` | RMSNorm fold plus residual rotation Q, in `prepare`. Llama only. |
| `seed` | `0` | R1 signs. 0 is the plain Sylvester H. Any other value gives Q = D·H, with the random signs on the input side. Output-side signs (H·D) cannot change a sign-symmetric cast. |
| `static` | `false` | Static po2 activation scales, floor(log2(amax/256)), calibrated on the **rotated** inputs. |
| `vpu_max` | `32` | Largest butterfly for `impl=vpu`. |

## How it maps to Atlas
- **Operands.**
  - The transform uses only ±1, which is exact in E4M3 (0x38 / 0xB8).
  - The factor 2^-k, with k = floor(log2 √m) and m the mixing size, belongs to the E8M0 scale of the following vpack / `vmatpop.fp8`, so it costs nothing.
  - The non-po2 remainder c = √m / 2^k is folded into the next weight (W → W·Mᵀ/c) by `transform_weight`, before that weight's FP8 cast.
  - 1/√m itself is never an operand. It is not a power of two for odd log2 m, and for m = 5632 it is below the E4M3 normal floor 2^-6.
- **`impl=mxu2`.**
  - MXU operands are FP8 only, so casting x to FP8 and then rotating commits the outlier error before the rotation. The screen measured that as worse than no rotation.
  - Instead, the BF16 input is split into hi = vpack(x, s) and lo = vpack(x − hi, s − 4). Both are accumulated through `vmatmul.acc` against H and an exact ±2^-4 copy of H.
  - The result is modelled as exact sums followed by one BF16 rounding. This is NEW and approximates the accumulator.
  - Estimated cost on TinyLlama down_proj: +10.8% MXU tile-matmuls for each term, plus one relayout.
- **`impl=vpu`.**
  - There is no intra-register permute, so only partners at stride ≥ 32 features can be mixed, using vadd/vsub between register pairs.
  - Each add is an FP32 add truncated to its upper 16 bits (`chip_ops.py` header; AddSubSumVec.scala:133).
  - TinyLlama shapes: down_proj is I11⊗H16⊗I32 (4 stages), and o_proj is H32 over heads ⊗ I64 (5 stages).
  - The truncation costs rel RMSE of about 0.8–1.0% on its own.
- **Extra work.**
  - The `algo` record carries `online_cost` per layer (VPU element ops, MXU MACs and tile-matmuls per 32 tokens, the mxu share of the layer, relayouts) and `online_cost_total`.
  - Under `chip`, the transform adds rows `rotation.online_hadamard` (MXU, VPU or host) and `rotation.online_hadamard.split` (VPU) to the coverage report.
  - The transform's own tensor ops run under the router guard, so they are not double-counted as model ops.
- **Needs a hardware ruling.**
  - VPU and MXU issue rates and VMEM traffic. No throughput figures exist.
  - `vtrpose.xlu` for the relayout between Kronecker factors.
  - Whether dynamic scales are intended (reference_profile Q2). Rotations need a scale chosen after the rotation and are incompatible with the fixed 2^0 scale.

## Calibration ordering
`algos.quantize` runs `calibrate` before `prepare`, and `observe_inputs` would see the inputs *before* the online transform. So `Rotation.calibrate`:
1. runs the folds first (`prepare` then returns the same record without re-folding);
2. observes each online layer's input after its own transform;
3. for `mxu2`, also observes the input before the transform, for the static scale of the split.

Data: `--calib-source wikitext2_train` (default 16 windows).

## Refusals
`r1=true` raises `RotationRefused` with the reason for:
- a LayerNorm or GroupNorm (mean-centering breaks the invariance, QuaRot Sec. 3.4);
- an adaptive RMSNorm (openpi `GemmaRMSNorm` with `cond_dim` / `dense`);
- adaLN or modulation blocks (DiT);
- any model that is not a Llama decoder with `LlamaRMSNorm`. That includes Gemma's (1 + w) norm, which is not implemented.

Other limits:
- The per-head fold is skipped, with a reason in the record, when head dims differ across attention modules or have no Hadamard matrix.
- Layers whose width has no full Hadamard mix only the power-of-two part. This is noted in `prepare.notes`.

## Screening results (CPU, TinyLlama, 4 test windows × 2048, Linear only)
bf16 PPL 9.2938. These are SCREENING numbers from the phase-1 standalone script, not this module.

| Rotation | Reference cast (dynamic) | Static | Fixed 2^0 + FTZ |
|---|---|---|---|
| none | +0.84% | +0.85% | +77.5% |
| R1 | +1.40% | +1.31% | +274% |
| online fp32 | +0.47% | +0.58% | ×1541 |
| R1 + online VPU | +1.33% | +1.44% | ×707 |
| R1 + online mxu2 | +1.08% | +0.84% | ×1921 |

Full write-up, scripts and diagnostics: `out/artifacts/handoff/atlas-fp8/algos/rotation.md` and `rotation/` in the merlin tracking worktree.

## GPU runs wanted
- **TinyLlama, 165 windows.** `rotation` and `rotation:impl=mxu2`, each with and without `static=true`, at ref_linear, ref_attn, reference_hw (ipt) and chip. Add `rotation:r1=true` at ref_linear only.
- **SmolVLA, DiT, Pi0.** Online on down_proj / o_proj at ref_linear and reference_hw. R1 is refused there.
