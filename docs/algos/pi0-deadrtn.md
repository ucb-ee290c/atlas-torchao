# Pi0 bundle with the two recommended numerics defaults: panel GPTQ, AWQ off, `dead_columns=rtn` (2026-11-06, CPU)

Bundle: `/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq_deadrtn/` (`export/`, `fx/`, `verify_bundle.json`, `bundle_manifest.json`,
`weights_only.json`, `drift.json`, `results/`, the run scripts `run_*.sh`, logs, and the capture drivers copied unchanged). Every number below comes from a
command run for this note; the files named are where the raw output is. The defaults were chosen by the requester as recommendations to be reviewed, not by a
quality measurement of the combination.

Defaults applied: (1) panel GPTQ with AWQ off (`tile-pi0-awq-diag.md`: AWQ + panel damaged Gemma `up_proj` rows), (2) `dead_columns=rtn`
(`gptq-dead-columns.md`; the cause of the SigLIP fc2 zero columns, `siglip-fc2-flush.md`).

## Recipe and export

`gptq_awq:method=gptq,actorder=true,dead_columns=rtn+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib+keep_bf16:rule=input_conv`
(`recipe.algo` in `bundle_manifest.json`). The previous bundle `gptq_panel_static_noawq` has the same string without `dead_columns=rtn`.

Command (`run_export.sh`): `eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --algo <recipe> --export-dir export --export-only --device cpu` with the atlas-pi0 python
(torch 2.7.1), `taskset -a -c 0-15 nice -n 10`, 5 threads. Calibration: 8 DROID frames, seed 0, episodes 203, 257, 170, 326, 85, 136, 282, 9 (the same list as the previous
bundle's `bundle_manifest.json`; the 32 evaluated episodes excluded). Export wall time 34.5 min (14:56:52 to 15:31:25, shared host). 483 quantized sites, 1 host, 2 skipped.
Code at the export: HEAD `4ddedee` (contains `7f5d9a3`, the `dead_columns` option); `producing_tree_dirty: true` in the manifest because `scripts/tile_diag/fc2_export_compare.py` was untracked then.
The previous bundle's weights came from commit `45edd84` (dirty tree, `exports-tile-diag/pi0/panel_gptq_noawq_cpu`).
No existing export directory was written to.

## Zero columns (weights only, `export/model.safetensors` codes, a column counts when every code in it decodes to zero)

| group | previous (GPTQ panel, no AWQ) | new (+ `dead_columns=rtn`) |
|---|---|---|
| SigLIP fc2 layers 17 to 25 | 3561, 4022, 4162, 4183, 4183, 4192, 4177, 4126, 3568 (sum 36174) | 0 in all nine |
| all SigLIP linears | 40392 (15 tensors) | 305 (1 tensor: layer 1 `mlp.fc2`) |
| Gemma LM | 635 (32 tensors; LM layer 0 q/k/v 198 each) | 0 |
| Gemma expert | 10 (2 tensors) | 0 |
| other (`state_proj` etc.) | 24 (1 tensor) | 0 |

The 305 remaining columns of SigLIP layer 1 fc2 are not GPTQ's doing: plain RTN on the panel `nc` exponents of the stock weight gives exactly 305 all-zero columns there
(largest stock |w| in those columns 1.46e-05, flushed by the grid). Stock has no all-zero column in that tensor.
`state_proj`: decoded zero share of the weight 0.0001 (previous: 0.75, `weights_only.json`).

## Weight error vs stock (`scripts/tile_diag/weights_only.py`, 420 Linears, `weights_only.json`, `results/weights_only_summary.txt`)

| | previous | new |
|---|---|---|
| median | 0.03381 | 0.03337 |
| mean | 0.06045 | 0.03554 |
| max | 0.97883 (SigLIP 23 fc2) | 0.11443 (SigLIP 2 k_proj) |
| layers with err > 0.1 | 15 | 2 (SigLIP 2 k_proj 0.11443, q_proj 0.11201) |

Worst 10 of the new export: SigLIP 2 k_proj 0.11443, 2 q_proj 0.11201, 3 k_proj 0.06959, 0 k_proj 0.06681, 3 q_proj 0.06663, 0 q_proj 0.05613, 0 fc1 0.05584,
5 k_proj 0.05426, 2 v_proj 0.05418, 23 fc2 0.05235. Previous worst 10 were fc2 17 to 25 (0.846 to 0.979) and `state_proj` (0.839). New fc2 17 to 25: 0.0293, 0.0365, 0.0339, 0.0428, 0.0475, 0.0393, 0.0523, 0.0339, 0.0348;
`state_proj` 0.0366. (The AWQ-fold undo of the script is a no-op fit here for a non-AWQ export; as in `pi0-gptq-panel-noawq.md`.)

## Gemma drift (`scripts/tile_diag/hybrid_drift.py export/model.safetensors`, DROID ep 252 frame 30, noise seed 0, stock model with dequantized export weights, `drift.json`)

| weights | L0 | L9 | L17 |
|---|---|---|---|
| AWQ + panel (`atlas_v1_awq_tile_static`) | | 0.1273 | 0.0852 |
| GPTQ panel, no AWQ (previous bundle; numbers from `pi0-gptq-panel-noawq.md`, not re-run) | 0.00668 | 0.0237 | 0.0207 |
| old per-channel (`pi0_atlas_v1_awq`, same source) | | 0.0252 | 0.0223 |
| new (this bundle) | 0.006734 | **0.024373** | **0.022152** |

The new export's Gemma drift is 3% (L9) and 7% (L17) above the previous no-AWQ export's, and slightly below the old per-channel export's at L9 and equal at L17 (0.0222 vs 0.0223).
Why: not isolated. Facts: `dead_columns=rtn` cannot move live columns (`gptq-dead-columns.md`: bit-identical live columns for a given H), yet in the 223 non-SigLIP tensors that had no zero column in
the previous export, all 223 differ in codes (median 18.7% of codes, range 0.1% to 32.7%); activation `input_scale_exp` differs in 2 of 483 sites (LM layer 14 down_proj, SigLIP layer 16 fc2), exponents of weights are equal.
So the two exports are not bit-reproducible from each other for reasons that are not the dead-column option (candidates: a different H from numerically different calibration activations on the shared host or a
different thread count, the code between `45edd84` and `4ddedee`); no repeat export with the same recipe was run to measure that run-to-run spread, so the 3% / 7% cannot be assigned to the option or to noise.
Also, the previous export zeroed 635 Gemma LM columns (and 10 expert columns) that are now kept; the sign of that effect on drift was not measured.

## SigLIP fc2 layers 17 to 25 on real inputs (`scripts/tile_diag/fc2_export_compare.py`, `results/fc2_compare.json`)

Real input = the captured fc2 inputs of DROID episode 252 frame 30 (`tmp_fc2/fc2_in.pt`, 512 tokens = 2 images x 256). Episode 252 is one of the 32 evaluated episodes, so these inputs were not
calibration data for either export (the exports calibrated on 8 other episodes). Output error = `||X (Wd - W)^T|| / ||X W^T||`, Wd the decoded export weight, W the stock weight.

| layer | zero cols prev | zero cols new | weight err prev | weight err new | out err frame prev | out err frame new | image0 prev / new | image1 prev / new |
|---|---|---|---|---|---|---|---|---|
| 17 | 3561 | 0 | 0.8458 | 0.0293 | 0.0028 | 0.0028 | 0.0031 / 0.0031 | 0.0025 / 0.0025 |
| 18 | 4022 | 0 | 0.9287 | 0.0365 | 0.0094 | 0.0094 | 0.0095 / 0.0095 | 0.0094 / 0.0094 |
| 19 | 4162 | 0 | 0.9574 | 0.0339 | 0.0068 | 0.0068 | 0.0071 / 0.0071 | 0.0065 / 0.0064 |
| 20 | 4183 | 0 | 0.9768 | 0.0428 | 0.0521 | 0.0499 | 0.0239 / 0.0238 | 0.0718 / 0.0685 |
| 21 | 4183 | 0 | 0.9757 | 0.0475 | 0.0228 | 0.0228 | 0.0224 / 0.0224 | 0.0233 / 0.0233 |
| 22 | 4192 | 0 | 0.9780 | 0.0393 | 0.0246 | 0.0246 | 0.0241 / 0.0241 | 0.0250 / 0.0250 |
| 23 | 4177 | 0 | 0.9788 | 0.0523 | 0.0244 | 0.0244 | 0.0243 / 0.0243 | 0.0245 / 0.0245 |
| 24 | 4126 | 0 | 0.9528 | 0.0339 | 0.0100 | 0.0100 | 0.0103 / 0.0103 | 0.0096 / 0.0096 |
| 25 | 3568 | 0 | 0.8674 | 0.0348 | 0.0043 | 0.0043 | 0.0040 / 0.0040 | 0.0045 / 0.0045 |

Reading: on this frame the output error is the same for both exports to three decimals except layer 20 (0.0521 to 0.0499; image 1 0.0718 to 0.0685). The frame has 3890 to 4236 input channels that are exactly
zero, and the columns the previous export zeroed carry (almost) no energy here, so zeroing them did not hurt this frame; the weight-error table above (0.85 to 0.98 to 0.03 to 0.05) and the Gaussian-input errors
(`out_err_gauss` 0.85 to 0.98 previous, 0.029 to 0.052 new, in `fc2_compare.json`) show the columns are zeroed in the previous weights, but this frame cannot show a benefit from restoring them.
Layer 20's improvement is the only place a difference is visible, and it comes from image 1.

### Held-out image check, layers 17 to 25 (`scripts/tile_diag/gptq_dead_columns_validate.py L`, `results/heldout_l*.json`, `results/heldout_table.txt`)

This is the check of `gptq-dead-columns.md` repeated for all nine layers (stock weight, `H` from image 0 only, GPTQ `zero` vs `rtn` on the panel `nc` grid, error on image 0 (calibration) and image 1 (held out);
one image pair, not the export's calibration):

| layer | dead cols in H | image 0: RTN / GPTQ zero / GPTQ rtn | held-out image 1: RTN / GPTQ zero / GPTQ rtn |
|---|---|---|---|
| 17 | 3935 | 0.0173 / 0.0016 / 0.0016 | 0.0173 / 0.0035 / 0.0029 |
| 18 | 4138 | 0.0264 / 0.0039 / 0.0039 | 0.0265 / 0.0121 / 0.0120 |
| 19 | 4214 | 0.0290 / 0.0058 / 0.0058 | 0.0290 / 0.0079 / 0.0079 |
| 20 | 4239 | 0.0266 / 0.0178 / 0.0178 | 0.0267 / 0.0502 / 0.0452 |
| 21 | 4239 | 0.0267 / 0.0220 / 0.0220 | 0.0267 / 0.0248 / 0.0248 |
| 22 | 4235 | 0.0256 / 0.0160 / 0.0160 | 0.0258 / 0.0484 / 0.0242 |
| 23 | 4232 | 0.0253 / 0.0164 / 0.0164 | 0.0255 / 0.1156 / 0.0226 |
| 24 | 4232 | 0.0198 / 0.0080 / 0.0080 | 0.0201 / 0.0229 / 0.0131 |
| 25 | 3986 | 0.0140 / 0.0026 / 0.0026 | 0.0138 / 0.0242 / 0.0051 |

GPTQ `rtn` is never worse than `zero` on held-out image 1; it is clearly better in layers 22 to 25 (1.7x to 5.1x lower held-out error; layer 23 0.1156 to 0.0226, layer 25 0.0242 to 0.0051), equal in 18, 19, 21, small in 17 and 20.
Layer 20 is worse than plain RTN on image 1 under both (0.0502 and 0.0452 against 0.0267).

## FX capture and verification

Capture: the drivers `policy_fx_capture.py` and `pi0_fx.py` copied unchanged from the previous bundle (`cmp`-identical to `atlas_v1_awq_tile_static`'s), `policy_fx_capture.py pi0 export fx --non-strict`, atlas-pi0 python,
same CPU prefix, 15:37:26 to 16:05:19 (about 28 min; the previous one took 54 to 55 min under heavier load). Fresh-process replay `passed`, 2 cases, outputs byte identical, 420 `qlinear`, 63 `qk_matmul`, 63 `pv_matmul`, 672 `quantize_act`, 0 `qconv2d`, code and scale bytes preserved.

`verify_bundle.py export fx verify_bundle.json` (atlas-pi0 python, 23.5 s): **25 of 25 checks pass**, `failed: []`, `unquantized_attention` 0, `uncalibrated_fallback` 0, reserved codes 0 over 2709909504 weight codes.
`make_bundle_manifest.py .` wrote `bundle_manifest.json` (counts: quantized 483, host 1, skipped 2).

Pack-scale admission (`export/policy.json` `compiler_pack_scale_admission`, admitted 122..128, activation codes, informational): **125 of 483 checked sites outside** (previous bundle: 124). Attention `p` operands 63 of 63 (code 119), Linear activation inputs 62 (previous 61).
Codes outside the range, number of sites: 119 (63), 120 (3), 121 (54), 129 (3), 130 (1), 135 (1) (previous: 129 had 2). Weight exponent codes 115..121 (recorded only). The extra site is one more code-129 Linear input (`input_scale_exp` differs from the previous export at 2 sites).

## Action error, quantized weights in the stock model vs stock (`eval_pi0.py --obs droid --n-obs N --seed 0 --modes bf16 ref_conv --from-export <export> --no-determinism-check --device cpu`)

`ref_conv` is the reference numerics (E4M3 weights as exported, static activation exponents from `policy.json`, E4M3 attention operands, conv in BF16) built from the export, so this includes activation and attention quantization, not only the weights.
The observations are the first N of the seed-0 shuffle of DROID `lerobot/droid_1.0.1` (one frame 0 per episode, full policy path with the 10-step sampler and fixed noise); they are disjoint from the 8 calibration episodes by construction (calibration takes positions 32 to 39).
Metric: `delta_rel_l2` = relative L2 of (action minus state) against the stock BF16 output (the metric `eval_pi0.py` headlines); `reference_rel_rmse_absolute` = the reference script's rel RMSE in absolute action space (the state dominates there, so it is much smaller).
Stock BF16 actions are identical between the runs of the two exports (`actions`, compared exactly).

| N obs | weights | `delta_rel_l2` | `delta_cosine` | `delta_max_abs` | abs-space rel RMSE | per-frame abs rel L2 | MSE vs recorded delta |
|---|---|---|---|---|---|---|---|
| 4 | previous export (GPTQ panel, no AWQ) | 0.0564 | 0.99868 | 0.0288 | 0.00545 | 0.0051, 0.0048, 0.0053, 0.0064 | 0.00708 (stock 0.00747) |
| 4 | new export (+ dead_columns=rtn) | 0.0594 | 0.99834 | 0.0346 | 0.00574 | 0.0056, 0.0056, 0.0059, 0.0058 | 0.00725 (stock 0.00747) |

## Not measured

- No closed-loop rollout, no success rate, no quality number against ground truth beyond the MSE/MAE vs recorded deltas printed above, which are dominated by the policy's own error (stock BF16 delta MSE 0.00747 on 4 frames).
- No repeat of any export with the same recipe, so the run-to-run spread of GPTQ on this host is unknown; the 3% / 7% Gemma drift difference and the action-error difference between the previous and the new export are not attributed to either the option or noise.
- No export with `dead_columns=rtn` and AWQ on; no export with the default (`zero`) from today's code, so "previous vs new" also differs by the code between `45edd84` and `4ddedee` and the run itself.
- Drift is measured on one frame (episode 252 frame 30), Gemma q inputs at layers 0, 9, 17 only; the SigLIP and the action expert are not in the drift metric.
- fc2 real-input errors use one frame (2 images) outside the calibration set; the export's own calibration `H` was not reproduced, so the in-sample comparison is only in the (single image 0) check above, not for the export.
- Chip numerics (`chip` mode), the FX graph's numerical output against the stock model, hardware execution, 10-step graph (the graph is a one-step sampler), LIBERO, other recipes.
- Pack-scale admission is informational; no activation exponent was changed to fit 122..128.
