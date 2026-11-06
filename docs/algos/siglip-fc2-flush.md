# Pi0 SigLIP `mlp.fc2`: why 83 to 97 percent of the decoded weights are zero (2026-11-06, CPU, weights plus one real frame)

Question from `pi0-gptq-panel-noawq.md`: SigLIP encoder layers 17 to 25 `mlp.fc2` (and `state_proj`) have weight-only relative error 0.85 to 0.98 and
`flushed_fraction` 0.9706 for layer 23, in both the new GPTQ-panel bundle and the old per-channel `pi0_atlas_v1_awq`.
Every number below comes from `scripts/tile_diag/siglip_fc2_flush.py` (all 27 layers; raw results
`/scratch2/lorenhung/atlas-fp8/tmp_fc2/siglip_fc2_flush.json`) and `scripts/tile_diag/siglip_fc2_capture.py` (captured fc2 inputs, same dir). Exports were read only.
Input data: stock Pi0 model in BF16 on CPU, DROID episode 252 frame 30 (the frame `hybrid_drift.py` uses), 2 camera images x 256 tokens = 512 tokens per layer.
One frame only; the export calibration used 8 other frames.

## Answer in four lines

1. The zeros are not E4M3 flush. They are whole **input-channel columns** (K axis) of fc2 set to 0 by GPTQ: `algos/gptq_awq.py:107-109`
   (`dead = diag(H) == 0; H[dead, dead] = 1; W[:, dead] = 0`). Layer 23 export: 4177 of 4304 columns are entirely zero codes (0.9705 of the codes; the policy
   `flushed_fraction` counts exponent-field-0 codes, so it counts these). Re-casting the same stock weight with plain RTN on the same panel `nc` exponents flushes **0.035 %** of layer 23.
2. The cause is the activation side: in layers 15 to 26 the fc2 input (GELU of fc1, BF16) is exactly zero for most channels. Layer 23, this frame: 98.81 % of the fc2 input
   entries are exactly 0, only 75 of 4304 channels are ever nonzero over 512 tokens, amax 63.5. There are no massive outlier weights.
3. On real inputs the damage is not visible for this frame: layer 23 fc2 output relative error 0.0244 (export) vs 0.0254 (RTN on the same exponents); layers 17 to 25 export 0.0028 to 0.052
   (RTN 0.014 to 0.030). The weight-only metric counts columns that carry no input. It is real only for inputs that excite a column the calibration set never excited.
4. Both exports use GPTQ (`atlas_v1_awq` = `method=awq_gptq`, new bundle `method=gptq`), so the premise "not caused by GPTQ" does not hold: both zero the same columns
   (layer 23: old 4172 zero columns, new 4177, 4172 in both). The `mse` weight rule does nothing here (no flush to fix).

## Q1 Weights are well behaved (stock fc2, `[1152 out, 4304 in]`, BF16 stock file)

| quantity | layer 23 | layer 5 (control) |
|---|---|---|
| per-row amax min / median / max | 0.0879 / 0.1338 / 0.2930 | 0.1270 / 0.1934 / 0.6602 |
| per-row median abs w (median over rows) | 0.0182 | 0.0317 |
| amax / median per row, min / median / p90 / p99 / max | 5.5 / 7.4 / 9.3 / 11.8 / 25.2 | 3.8 / 6.1 / 7.6 / 13.6 / 20.3 |
| rows with amax/median > 100 or > 1000 | 0 / 0 | 0 / 0 |
| rows with amax > 20 x their 99th percentile | 0 | 0 |
| log2(abs w / row amax): p0.1 / p1 / p10 / p50 / p90 | -12.4 / -9.1 / -5.5 / -2.9 / -1.6 | -11.7 / -8.4 / -5.1 / -2.7 / -1.3 |
| share of weights with log2(abs w / row amax) < -14 | 0.034 % | 0.021 % |
| distinct columns holding a row's max (of 4304) | 238 (top column holds the max of 3.0 % of rows) | 860 (2.4 %) |
| column amax, max / median | 3.1 | 3.6 |
| export exponents (panel, 32 rows, `nc`) | -11 .. -10 | -10 .. -9 |
| per-row `nc` exponent range, panel-minus-row mean | -12 .. -10, +0.21 | -11 .. -9, +0.99 |

The `nc` rule gives amax / 2^s in (224, 448], so E4M3 flushes (final exponent < -6) a weight below about 2^(s-6), i.e. about 2^-14 to 2^-15 of the group amax, not 2^-6 of it.
At layer 23 only 0.034 % of weights are that small. Plain RTN on the export's own panel exponents: flushed 0.0350 % (panel `nc`), 0.0293 % (per-row `nc`), weight error 0.0266 (layer 23) and
0.0265 (layer 5). The same recast over all 27 layers: flush 0.027 to 0.056 % in every layer except layer 1 (8.05 %), weight error 0.0264 to 0.0266 in every layer.
Layer 1 is a genuine small-weight flush (panel exps -10..-8, RTN error still 0.0266). So the weights show no outlier problem in layer 23, and the control is the same.

## Q2 Cause: sparse, massive fc2 inputs, then GPTQ's dead-column rule

fc2 input (GELU tanh output in BF16), 512 real tokens, `siglip_fc2_flush.json` `input` block. BF16 `gelu_tanh` is exactly 0 for a pre-activation at or below -5.0625 (measured), which is why zeros appear.

| layer | exact-zero share of fc2 input | alive channels img0 / img1 / union (of 4304) | amax | export alive channels (8 calib frames) |
|---|---|---|---|---|
| 5 | 1.7 % | 4304 / 4304 / 4304 | 11.2 | 4304 |
| 14 | 4.2 % | 4301 / 4303 / 4304 | | 4303 |
| 15 | 57.6 % | 3021 / 3181 / 3264 | | 3689 |
| 16 | 88.8 % | 1064 / 1268 / 1330 | | 1801 |
| 17 | 95.9 % | 369 / 385 / 414 | 138.0 | 743 |
| 20 | 99.4 % | 65 / 67 / 68 | 42.0 | 121 |
| 23 | 98.8 % | 72 / 72 / 75 | 63.5 | 127 |
| 25 | 99.2 % | 318 / 301 / 367 | 180.0 | 736 |
| 26 | 93.9 % | 2692 / 2338 / 2904 | 524.0 | 3623 |

Top-5 channel amax at layer 23: 63.5, 56.0, 53.0, 44.8, 29.6 (activations in the few live channels are large; everything else is exactly 0). The export's zero columns
(layer 23: 4177) are a subset of this frame's dead channels (4229): 0 export-dead channels are alive in this frame at layer 23. Across layers 15 to 26 the export zero-column count
rises exactly where the zero share of the input rises (layer 15: 615 columns, 16: 2503, 17: 3561 ... 23: 4177, 25: 3568, 26: 681).
Not an input-scaling artifact: there is no AWQ here (the new bundle has none; the old one has none on fc2, which has no gate/up pair to fold), and the weights are not outlier-heavy.
Not checked: whether the activation exponent static `input_scale_exp` (layer 23: -1) is a problem; not part of this question.

`state_proj` (1024 x 32): 75.0 % zero codes, 24 of 32 columns entirely zero. DROID has 8 real state dims padded to 32, so 24 dead input columns are exactly the padding. Harmless, same mechanism.

Old per-channel export, same layers (`exports-gpu/pi0_atlas_v1_awq`): fc2 weight error layer 23 0.9783 (new 0.9788), dead columns 4172 (new 4177), 4172 in both.
Across layers 15 to 26 the old export's zero-column set is a subset of the new one's (columns in both = old count) and the new one has 5 to 214 more (layer 16: 2289 vs 2503, layer 17: 3461 vs 3561).

## Q3 Is the damage real for outputs? Relative error ||X (Wd - W)^T|| / ||X W^T||

Real = captured inputs of this frame, Gaussian = 1024 x 4304 standard normal rows (seed 0). RTN = plain E4M3 on the same panel `nc` exponents as the export.

| layer | weight err export | real out err export | real out err RTN | Gaussian out err export | Gaussian RTN |
|---|---|---|---|---|---|
| 5 | 0.0504 | 0.0110 | 0.0134 | 0.0503 | 0.0265 |
| 17 | 0.8458 | 0.0028 | 0.0173 | 0.8454 | 0.0266 |
| 18 | 0.9287 | 0.0094 | 0.0264 | 0.9290 | 0.0265 |
| 19 | 0.9574 | 0.0068 | 0.0290 | 0.9578 | 0.0265 |
| 20 | 0.9768 | 0.0521 | 0.0267 | 0.9770 | 0.0266 |
| 21 | 0.9757 | 0.0228 | 0.0267 | 0.9756 | 0.0266 |
| 22 | 0.9780 | 0.0246 | 0.0257 | 0.9781 | 0.0265 |
| 23 | 0.9788 | 0.0244 | 0.0254 | 0.9784 | 0.0266 |
| 24 | 0.9528 | 0.0100 | 0.0199 | 0.9526 | 0.0266 |
| 25 | 0.8674 | 0.0043 | 0.0139 | 0.8676 | 0.0267 |

On this frame the export is as good as RTN (better in most layers: GPTQ fits the live columns), except layer 20 (0.0521 vs 0.0267): there 1 export-dead column is alive
in this frame with 3.2e-4 of the input energy. Elsewhere the dead-in-export but alive-in-this-frame channel counts are 8 (layer 17), 6 (layer 25), 0 (layer 23), with input energy share 8e-16, 6.6e-14, 0 (other layers: see `export_dead_cols_alive_in_this_frame` in the json).
Exposure: the alive set differs between frames (layer 23: 72 channels per image, 75 over two images, 127 over the 8 calibration frames), so a new frame can excite a zeroed
column. A zeroed column's whole contribution to every output channel is lost. Not measured: actions or any frame beyond this one; the Gaussian column is the worst case (all columns excited).

## Q4 Options (weights-only on layers 17 to 25, plus real/Gaussian output error)

Layers 17..25 fc2, weight rel error, from `siglip_fc2_flush.json` (all four RTN-type rules, each layer, agree to 4 digits):

| option | layer 23 weight err | flush at layer 23 | layers 17-25 weight err | cost / tile-native |
|---|---|---|---|---|
| current (GPTQ, panel `nc`) | 0.9788 | 97.06 % (zero columns) | 0.846 .. 0.979 | as exported; fragile on new frames |
| RTN, panel `nc` (same exponents) | 0.0266 | 0.035 % | 0.0265 .. 0.0267 | tile-native; loses GPTQ on these 9 layers (Gaussian err 0.0266) |
| RTN, per-row `nc` (not tile-native) | 0.0266 | 0.029 % | same | no gain over the panel |
| `static_calib` `weight=mse` (per tensor, as shipped) | 0.0266, exponent -10 | 0.060 % (layer 17: 0.177 %) | same | no change: the weights have no flush problem to fix |
| `mse` per 32-row panel (own generalisation, `algos.static_calib.search` per panel) | 0.0266 | 0.034 % | same | no change |
| BF16 (stock weight) | 0 (not quantized) | n/a | 0 | host/VPU work, fc2 17-25 = 10.9 % of SigLIP linear MACs (9 x 1152 x 4304 of 27 layers x (4 x 1152^2 + 2 x 1152 x 4304)); the MXU has no BF16 operand mode |
| export with only its zero columns re-cast by RTN (proxy for a GPTQ that does not zero dead columns) | not computed (Gaussian out err 0.0514, real out err 0.0244) | 0 zero columns | Gaussian out err 0.033 .. 0.051 | not available in the kit: needs a change in `gptq()` |

Notes.
- How to get BF16 with existing options: `keep_bf16` only implements `rule=input_conv` (patch embedding), so fc2 would use `mixed_precision:units=<fqn>;...,keep_as=bf16` (exact fqns, `check_exact` rejects substring clashes). Not run.
- `method=awq` in `gptq_awq` performs RTN on layers without an AWQ group, but applies to the whole model (no per-layer selection), so it removes GPTQ from Gemma too.
- AWQ-like column scaling for fc2 would need the factor in the producer of fc2's input, which is `GELU(fc1(x))`. GELU is not positively homogeneous, so a per-channel factor cannot be folded into fc1 rows/bias; the kit folds only the gate*up -> down pair and norm -> consumers (`gptq_awq.py:27`). Not foldable exactly, and it would not remove the zeros (they come from dead inputs).
- A smaller `rows=` is not hardware native and would not matter: RTN at per-row exponents flushes as little as the panel.

## Recommendation

1. Do not "fix the flush" with `mse`, a smaller `rows`, or per-row exponents: they change the weight error by under 1e-4 and the flush by under 0.2 %.
2. The weights-only error and `flushed_fraction` in the SigLIP fc2 sites of GPTQ exports should be read as "dead columns zeroed", not as quantization damage. Add the real-input output error (as here) to the weights-only report so the metric is not misread.
3. For robustness to frames that excite channels dead in the 8 calibration frames, the smallest change is in `gptq()`: quantize the dead columns with RTN on the pinned exponent instead of zeroing them (my re-cast of the export's zero columns is the weights-only proxy: real error unchanged, Gaussian 0.033 .. 0.051, no zero columns). If no code change is wanted, BF16 for fc2 17..25 (10.9 % of SigLIP linear MACs) or RTN for those layers are the two available options; RTN keeps tile-native FP8 at weight error 0.0266 and, on this frame, real output error 0.014 to 0.030 vs 0.0028 to 0.052 for the GPTQ export.
4. Before settling, measure on more frames: the alive-channel set of layer 17..25 fc2 over, say, 32 DROID frames versus the export's zero columns, and the action drift with fc2 17..25 as RTN, BF16, and as exported. Not done here.
