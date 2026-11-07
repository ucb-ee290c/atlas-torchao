# Pi0 native FP8 FX bundle with GPTQ on the panel grid, AWQ off (2026-11-06, CPU)

Bundle: `/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq/` (`export/`, `fx/`, `verify_bundle.json`, `bundle_manifest.json`;
logs and driver copies beside them). Purpose: avoid the AWQ-on-panel regression of `atlas_v1_awq_tile_static`
(`tile-pi0-awq-diag.md`). Every number below was produced by a command run for this note. Quality (actions, rel L2 of outputs) was not measured.

## Recipe

`gptq_awq:method=gptq,actorder=true+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib+keep_bf16:rule=input_conv`
(recorded as `recipe.algo` in `bundle_manifest.json`). Compared with `atlas_v1_awq_tile_static+keep_bf16:rule=input_conv` the only change is
`method=gptq` instead of `method=awq_gptq`. Calibration: 8 DROID frames, seed 0 (same episodes as the earlier export).
Weights are the already exported `exports-tile-diag/pi0/panel_gptq_noawq_cpu/export/` copied unchanged (cmp-identical `model.safetensors`, sha256 `ba37b7eb...`).
Not re-exported. The export manifest records producing commit `45edd84` with `atlas_torchao_dirty: true`; the FX capture ran from checkout `79628ee`
(`fx_capture_commit` in the bundle manifest).

## Capture and verification

Capture: `policy_fx_capture.py pi0 export fx --non-strict` with the atlas-pi0 python (torch 2.7.1), the drivers copied unchanged from the previous
Pi0 bundle; 2 cases, fresh-process replay `passed`, outputs byte identical, 420 `qlinear`, 63 `qk_matmul`, 63 `pv_matmul`, 672 `quantize_act`, 0 `qconv2d`,
6509 graph nodes. Wall time was about 55 minutes on the shared, heavily loaded host (the previous bundle took 1549 s).

`verify_bundle.py export fx verify_bundle.json` with the atlas-pi0 python: **25 of 25 checks pass**, no failures, `unquantized_attention` 0,
`uncalibrated_fallback` 0. Declared exceptions, as before: 1 contraction outside the atlas ops, the SigLIP patch embedding
(`aten.conv2d.padding`, float32 weight `[1152,3,14,14]`, BF16-by-policy via `keep_bf16:rule=input_conv`), and 420 unused float weight meta placeholders
(non-strict export). `make_bundle_manifest.py` needed no fallback this time (commit `09ef1df` already has the Pi0-safe `model_record`).

## Drift (`scripts/tile_diag/hybrid_drift.py export/model.safetensors`, DROID ep 252 frame 30, noise seed 0, stock model with dequantized export weights)

| weights | L9 | L17 |
|---|---|---|
| old `pi0_atlas_v1_awq` (GPU, channel) | 0.0252 | 0.0223 |
| previous tile export (AWQ, panel) | 0.1273 | 0.0852 |
| this bundle (GPTQ, panel, no AWQ) | **0.0237** (0.023687) | **0.0207** (0.020661) |

L0 0.00668. The bundle's own weights reproduce the diag export's numbers exactly (`drift.json`).

## Weights-only error vs stock (`scripts/tile_diag/weights_only.py`, 420 Linears; old = `exports-gpu/pi0_atlas_v1_awq` per-channel)

Rel Frobenius error of the dequantized weight against the stock weight (AWQ fold on up/down fitted and undone by the script; for this export that fit is
a least-squares scalar per row, not a recorded fold). Median 0.0338 (this) vs 0.0340 (old); mean 0.0605 vs 0.0983; layers with err > 0.1: 15 vs 19.
Old worst: Gemma expert L0 down_proj 6.918, expert L3 down_proj 5.057, Gemma LM L0 down_proj 3.221 (the AWQ-folded pairs); none of these is in this export's worst 10.

Worst 10 of this export (old value for the same site in brackets, `weights_only.json`, `worst10.txt`):

| err | site | old |
|---|---|---|
| 0.97883 | SigLIP encoder layer 23 mlp.fc2 | 0.97826 |
| 0.97802 | SigLIP 22 fc2 | 0.97723 |
| 0.97679 | SigLIP 20 fc2 | 0.97631 |
| 0.97566 | SigLIP 21 fc2 | 0.97451 |
| 0.95742 | SigLIP 19 fc2 | 0.95653 |
| 0.95280 | SigLIP 24 fc2 | 0.95152 |
| 0.92872 | SigLIP 18 fc2 | 0.92724 |
| 0.86737 | SigLIP 25 fc2 | 0.86220 |
| 0.84577 | SigLIP 17 fc2 | 0.83698 |
| 0.83949 | `state_proj` (1024x32) | 0.83949 |

Finding not caused by this recipe and not diagnosed here: both exports have SigLIP fc2 layers 17 to 25 with 83 to 97 percent of nonzero stock weights decoding to
zero (weight exponent range -11..-10 for layer 23), and `state_proj` 75 percent. The export policy itself reports `flushed_fraction` 0.9706 for
SigLIP layer 23 fc2. The drift metric does not see these layers (it measures Gemma q inputs). Cause and effect on actions are open.

## Pack-scale admission (policy.json `compiler_pack_scale_admission`, admitted 122..128)

124 of 483 checked sites are outside the range (previous tile export: 127 of 483). By class: 63 of 63 attention `p` operands (code 119 for all 63),
61 of 420 Linear activation inputs. Codes outside the range, number of sites: 119 (63), 120 (3), 121 (54), 129 (2), 130 (1), 135 (1); previous export: 117 (1), 118 (2),
119 (63), 120 (6), 121 (54), 133 (1). Counts by part are in `policy.json`. Weight exponent codes are recorded only (115..121 here, 110..129 previously); the range applies to activation packs.
