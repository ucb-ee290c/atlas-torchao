# static_calib: static calibrated power-of-two activation scales

`--algo static_calib[:k=v,...]` fixes each Linear / Conv2d activation exponent offline, from
calibration data. It replaces the reference's per-call dynamic rule `floor(log2(amax / 256))`
(pi0-quant `hw_float_ops.py:38-57`; `ref_numerics.po2_scale_exp`).

On the chip, a static scale is one `seli` per operand, with no run-time amax. It applies through
the `act_scale` hook (and optionally `weight_scale`) on the reference steps `ref_linear` through
`chip`.

**Status:** CPU screening only. GPU runs are pending.

## Why

The collaborator's chip-cast flow is `collab_v0_hw`:
- a fixed `2^0` activation scale (`atlas_torchao.py:17,54-64`);
- the hardware cast flushes `|x| < 2^-6` to zero.

TinyLlama activations are small: 88 of 154 Linear inputs have amax < 8, and the early `o_proj`
inputs have amax 0.28–0.48. With `2^0`, most of their mass is flushed, and that is the +84%.
Any amax-anchored exponent puts the flush threshold near amax·2^-14.

## Sources

| Choice | Source |
|---|---|
| Max calibration; saturate overflows; a per-tensor exponent bias equals a po2 scale | NVIDIA, "FP8 Formats for Deep Learning", arXiv 2209.05433, Sec. 2, 3.2, 4.2, 4.3 |
| Max, entropy (KL) and percentile calibration; 99.9% clips too aggressively | Wu et al., arXiv 2004.09602, Sec. 3.4, 4.2 / Table 5 |
| MSE-optimal exponent bias per tensor | Kuzmin et al., "FP8 Quantization: The Power of the Exponent", arXiv 2208.09225, Sec. 3.1, 5.2, App. E |
| Massive activations, which make percentile and KL clip | Sun et al., arXiv 2402.17762, Sec. 2.1 |
| Exponent rule `floor(log2(v/256))`, BF16 amax | pi0-quant `hw_float_ops.py:38-57` (amax intended from VPU rmax: `:44-51`) |
| Weight rule `ceil448` | the collaborator, `atlas_quant.py:53-60` |
| vpack cast (flush, saturation, 480 clamp) | `ref_numerics.e4m3_vpack` = `fp8_e4m3.py:38-86` = `FP8Pack.scala:57-95` |
| Exact BF16-magnitude histogram as the calibration statistic | **NEW**: operands are cast to BF16 before vpack (`hw_float_ops.py:89`), so a 2^15-bin histogram is lossless and every candidate's error is exact |
| KL on the po2 FP8 grid | **NEW** adaptation of TensorRT entropy calibration: source bins merged per FP8 level, mass spread over non-empty bins |
| `attention.p` exponent −8 | **NEW**: softmax ∈ [0, 1], so −8 is the reference rule at the largest possible amax (1.0 maps to 256) and never saturates. The dynamic rule gives −9 when max P < 1, so this costs at most one binade at the bottom. |
| `headroom` | **NEW**: guards saturation on inputs hotter than the calibration set |
| `per_step` | **NEW**: diffusion / flow layers run once per denoising step with step-dependent ranges |

## Parameters

| Key | Default | Meaning |
|---|---|---|
| `method` | `absmax` | `absmax`: the reference rule on the calibration-set amax.<br>`mse`: argmin of the exact vpack squared error over `[s_absmax-8, s_absmax+3]`; ties go to the larger exponent.<br>`percentile`: the reference rule on the `p`-th percentile of \|x\|.<br>`kl`: entropy, as above. |
| `p` | `99.99` | Percentile for `method=percentile`. |
| `headroom` | `0` | Binades added to every calibrated activation exponent. |
| `weight` | `ref` | `ref`: the reference per-tensor rule (hook returns None).<br>`ceil448`: the collaborator's rule.<br>`mse`: exact vpack MSE over `[s_ref-3, s_ref+2]`. |
| `p_exp` | `-8` | Static `attention.p` exponent. `dynamic` keeps the per-(batch, head) rule. |
| `per_step` | `false` | One exponent per (layer, call index within one calibration forward). See below. `absmax` only. |

Hook behaviour:
- `attention.q/k/v` keep the dynamic per-(batch, head) scales.
- A layer never reached during calibration keeps the dynamic rule. It is counted under
  `algo.granularity.activation` as `linear:dynamic per_tensor`.
- The record `algo.calibration.result` holds the exponent of every layer, the
  `exp_minus_absmax` histogram, and the per-step layers and fallbacks.

### per_step (diffusion / flow models)

The usual case is a DiT denoising loop or Pi0 flow steps. There, a layer is called several times
per forward, once per step, and its activation range changes with the step.

**Calibration.** For each layer, `calibrate` records the amax per call index within each
calibration forward.

**Run time.** The step index is `call_count % calls_per_forward`.

**Fallback.** If a layer's call count differs between calibration batches, it falls back to its
per-layer exponent and is listed in `per_step_fallback`.

**Assumption.** The evaluation must invoke the model the same way the calibration `run` does:
same entry point and same step count. The modulo counter does not see sample boundaries.
It is still static on the chip: one `seli` per (layer, step), from a table fixed at compile time.

## Hardware cost: on-chip amax vs static (ISA green card, atlas-npu RTL @0079c054)

**Scale registers.**
- `e0-e31` hold one E8M0 exponent each (green card l.70).
- They are written only by `seli` (`e[rd] = imm`, l.315) or `seld` (`e[rd] = VMEM[x[rs1]+imm]`,
  l.314). RTL: `atlas/scalar/ScalarCore.scala:462-464` (write enable = `is_seli` or a SELD
  load response; data = `imm[7:0]` or the loaded byte).
- No instruction moves a vector register into a scale register.
- `vpack.bf16.fp8` and `vunpack.fp8.bf16` read `e[es1]` (l.358-359).

**Dynamic per-tensor scale.** This is what the reference rule implies on chip (reference_profile
Q2):
1. **|x|.** There is no `vabs`. Either take `vredmax` and `vredmin` separately, as chip_profile
   counts it, or `vmul` by −1 then `vmaximum`.
2. **Reduce** each 32×32 tile pair with `vredmax.row.bf16` (l.353) and `vredmax.bf16` (l.350).
   RTL decode: `Instructions.scala:128,133`. Then a `vmaximum` tree across every tile of the
   tensor.
3. **Barrier.** No tile can be packed until the whole tensor is reduced. The activation is read
   twice, or kept resident in VMEM / registers.
4. **Move to a scale register.** `vstore` the reduced tile to VMEM. `lhu` the BF16 bits. On the
   RV32I core (no FP), `srli 7` and `andi 0xFF` extract the exponent, then `addi` forms the E8M0
   byte for `floor(log2(amax)) - 8`. Then `sb` to VMEM and `seld` into `e[k]`.

**Static.** One `seli e[k], imm` per operand (per step with `per_step`), with no reduction,
barrier or VMEM traffic. The `2^(sx+sw)` output rescale also becomes a compile-time constant.

No throughput or latency figures exist anywhere, so neither path is costed in cycles. The chip
profile counts the difference: `chip_profile.count_algo_cast` books a static cast as
`vpack.bf16.fp8` only, and a dynamic one adds `vredmax.row` and `vredmin.row`.

**Hardware rulings wanted.**
- Q2: is dynamic intended at all?
- Is a compile-time E8M0 table per layer, or per step, the intended scale encoding?
  `software-spec.yaml` has `scale_encoding: unknown`.

## CPU screening (TinyLlama-1.1B)

**SCREENING:**
- **Calibration:** 8 × 2048 WikiText-2 train tokens.
- **Evaluation:** the first 4 × 2048 test windows; bf16 9.2938.
- Weights use the reference rule.

Scripts and full tables: `merlin-wt-tracka/out/artifacts/handoff/atlas-fp8/algos/static-calib/`.

| Activation exponent | vpack cast Δppl | torch cast Δppl |
|---|---|---|
| fixed 2^0 (collab_v0_hw / collab_v0) | +77.51% | +0.56% |
| dynamic per call (ref_linear) | +0.84% | +0.44% |
| static absmax | +0.87% | +0.50% |
| static mse | +0.84% | +0.53% |
| static p99.999 / p99.99 / p99.9 | +11 550% / +8 880% / +45 441% | not run |
| static kl | +37 476% | not run |

**Percentile and KL clip the massive activations.** In `layers.{2,7,21}.mlp.down_proj`, amax is
about 190 while p99.999 is about 0.25. Isolated rel RMSE there reaches .99.

**The weight rule does not matter at this sample size:** ceil448, ref and mse are all within 0.25
points.

**The gap left over is the flush, not the scale choice:** about +0.85% under vpack against about
+0.5% under the torch cast.

## Relation to other tracks

The smoothquant track's folded po2 "gain" reaches the same static effect by folding the exponent
into the producer. This track keeps it as an `act_scale` with `granularity = static`, and records
it separately. The two are to be compared, and possibly merged, after the GPU grid.

## Usage

    python src/eval.py --model <TinyLlama> --modes bf16 ref_linear ref_attn reference_hw chip \
        --algo static_calib:method=absmax --calib-n 16
    python src/eval.py ... --algo static_calib:method=mse,headroom=1
    python src/eval_policy.py ... --algo static_calib:per_step=true     # DiT
