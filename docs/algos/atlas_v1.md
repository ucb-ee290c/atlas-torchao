# Atlas FP8 v1: AWQ+GPTQ, granularity and rounded residual adds

`--algo atlas_v1` (or `atlas_v1_sq`, `atlas_v1_awq`, `atlas_v1_static`) expands to a documented stack of the
merged algorithms (`algos.RECIPES` in `src/algos/__init__.py`; stacking rules in
[stacking.md](stacking.md)). The recipe itself is **NEW**: it combines tracks that each cite their
own sources, but no source defines this combination. Each component keeps the numerics of its
track, and the reference author's numerics (`src/ref_numerics.py`) stay the source of truth for
every cast.

There are two variants because the hardware ruling on run-time activation scales is open:
`atlas_v1` needs an amax on chip at run time (per token); `atlas_v1_static` needs none for the
Linear inputs.

## The recipes

| Recipe | Expands to | Chip default |
|---|---|---|
| `atlas_v1` (selected, NEW decision) | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc`; exact DiT patch embedding in BF16 | `--chip-vadd rn --chip-vadd-sites residual` |
| `atlas_v1_awq` (historical compatibility alias) | the same AWQ+GPTQ stack, with no automatic DiT BF16 exception | the same |
| `atlas_v1_sq` (NEW replay name for run-5.x `atlas_v1`) | `smoothquant:alpha=0.5,po2=true,gain=false,partial=true` + `gptq_awq:method=gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | the same |
| `atlas_v1_static` (no on-chip amax for Linear inputs) | `smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true` + `gptq_awq:method=gptq,actorder=true` | the same |

- **Chip default.** Under a chip profile, the recipe sets `--chip-vadd rn --chip-vadd-sites
  residual` unless the command line passes `--chip-vadd` (an explicit value, `trunc` included,
  wins). `--chip-vadd` defaults to unset on this branch so that the difference is visible; unset
  still means `trunc`. The record lists what was applied in
  `algo.recipe.chip_options_applied`, and the ladder marks the step off-ladder as for any
  `--chip-vadd rn` run.
- **Choice (NEW).** Run-5.x AWQ+GPTQ has the lowest measured chip errors on TinyLlama,
  SmolVLA and Pi0. The supplied table marks each chip improvement over `none` outside
  the reported noise floor, subject to the protocol/unit qualifications below. This is a
  ranking of measured errors; no DiT chip recipe measurement was supplied.
- **Historical smoothing.** SmoothQuant smoothing and AWQ both fold per-channel
  factors into the same norms (`input_layernorm`, `post_attention_layernorm`) and producers
  (`up_proj` rows). The stack refuses that combination (stacking.md). `atlas_v1_sq` preserves
  the previous smoothing plus GPTQ recipe. Historical run tags keep their original meanings.
- **Why no granularity in `atlas_v1_static`.** A per-token scale is a run-time scale. The static
  recipe keeps the per-tensor weight exponent that GPTQ pins, and the activation exponent of every
  calibrated Linear input is static (smoothquant `static=true`, computed from the post-fold
  calibration maximum by the reference rule). Attention operands keep the reference dynamic
  per-(batch, head) rule, so `atlas_v1_static` still needs an amax there (open).
- **Weight grid.** In `atlas_v1` and `atlas_v1_awq`, granularity chooses the per-channel weight
  exponent and GPTQ quantizes onto it and pins it (the weight-grid delegation in stacking.md).
- **DiT exception (NEW).** Only canonical `atlas_v1`, on `model.config.type == "multi_task_dit"`,
  preserves exactly `observation_encoder.vision_encoder.model.vision_model.embeddings.patch_embedding`
  as its existing BF16 Conv2d. Identity/type validation happens before calibration; GPTQ/AWQ
  collect and transform Linear weights only, and this Conv2d never reaches quantization hooks.
  A missing or changed module type is refused. Similarly named siblings and other policy types
  are unaffected. The result records `bf16_opt_out_exact`; export records `match: exact`, the
  BF16 weight and host status, and load replays that exact exclusion without calibration.
  BF16 has no MXU operand path: placement on the VPU or host remains pending R4.
- **Other mixed precision.** fp8x3 residual passes need ruling R4 (whether passes can share the MXU
  accumulator). Opt in with `+`, for example
  `--algo atlas_v1+mixed_precision:rank_file=PATH,k=4` (ranking from `scripts/rank_layers.py`;
  causal LM only).
- Recipes take no parameters (`atlas_v1:alpha=0.3` is refused). To change a component, write the
  stack out.

## Run-5.x evidence and limits

Historical recipe labels below are the labels at the cited tags. AWQ means historical
`atlas_v1_awq` (the canonical v1 stack); SQ means historical `atlas_v1` (now `atlas_v1_sq`).

| Model / metric | BF16 R / C | none R / C | SQ R / C | AWQ R / C | static R / C | 2σ R / C |
|---|---|---|---|---|---|---|
| TinyLlama perplexity | 7.9745 / 8.0560 | 8.1067 / 8.2559 | 8.0662 / 8.1921 | 8.0660 / 8.1444 | 8.0731 / 8.1461 | 0.077 / 0.244 pt |
| SmolVLA rel L2 | baseline | 0.1519 / 0.1908 | 0.1243 / 0.1618 | 0.1271 / 0.1459 | 0.1251 / 0.1796 | 0.0095 / 0.020 |
| Pi0 absolute rel L2 | baseline | 0.0120 / 0.0144 | 0.0081 / 0.0100 | 0.0066 / 0.0076 | 0.0106 / 0.0082 | 0.0056 / 0.0027 |

TinyLlama: R full 166 windows, C 4 mxq windows; calibration 16 WikiText-2 train windows.
SmolVLA: R 200 frames, C 32 frames; calibration 32 frames outside evaluation episodes.
These rows used `l40s-run-5` (`0447807`). Pi0: 5 DROID observations, seed 0; calibration
8 observations outside evaluated episodes. `none` used run 5; SQ/static used run 5.1
(`46b0b1d`); AWQ used run 5.3 (`a30779b`). AWQ raised one Cholesky dampening to 0.1,
which is recorded. Pi0 delta-space rel L2 R/C was 0.106/0.127 (none), 0.071/0.088 (SQ),
0.058/0.067 (AWQ), 0.093/0.072 (static). The supplied table labels all Pi0 reference
improvements within its reported threshold and all Pi0 chip improvements outside it.
AWQ versus SQ/static head-to-head differences are not claimed statistically significant.

Noise-floor transfer is provisional. Run-3 TinyLlama used 40 mxq windows, versus the
current 166 full-split reference windows and 4 chip windows; run-3 Pi0 used 32 observations,
versus the current 5. The supplied TinyLlama changes such as -0.51 and -1.38 are changes
in BF16-normalized relative degradation, in percentage points, rather than raw perplexity
differences. The reported floors use the ambiguous unit `pt`; their unit must be reconciled
before comparing them with these changes. The raw AWQ-minus-none perplexity differences
are -0.0407 on R and -0.1115 on C. We retain the supplied floors without claiming fresh
statistical significance from them.

All repeats were identical, all layers were reached, all values were finite, and chip
cross-checks had zero mismatches. This is output fidelity evidence, not closed-loop task success.

### DiT after checkpoint repair (run 5.2, `4ab3889`)

The repaired checkpoint remapped 395 CLIP keys. Earlier DiT measurements are superseded.
At 200 frames from episodes 136/363/380/464/474, BF16 GT MSE was 0.01822.

| Historical recipe, ref_conv only | rel L2 vs BF16 | vs none | GT MSE | BF16 opt-outs |
|---|---|---|---|---|
| none | 0.1572 | — | 0.01890 | 0 |
| SQ / AWQ, bit identical | 0.1394 | -0.0178, within noise | 0.02025 | 0 |
| static | 0.1514 | -0.0058, within noise | 0.01955 | 0 |
| SQ + patch embedding BF16 | 0.0883 | -0.0689, outside 2σ | 0.01828 | 1 |

The 2σ threshold is 0.0559 rel L2, from repaired multitask-dit-quant Set A
(`3baa5d6`), a separate harness; it is an empirical comparison threshold, not a paired
significance test. SQ folded nothing and AWQ recorded `awq_folded: []`, making their
unexcluded outputs bit identical. Applying the measured patch exception to AWQ is NEW;
the canonical DiT combination still needs a full checkpoint evaluation, fresh export and
round trip. No canonical DiT chip result is available.

The other harness found Set A rel L2 0.1768 ± 0.0279. Conv only (0.1427) and no conv
(0.1348) do not support the old conv-dominance claim. SA still concentrates error in all
six adaLN layers (local rel RMSE 0.178–0.202); IPT's worst local layer is 0.048.
The 32-frame IPT reference_hw output error was 0.1586; SA was 0.2853 on 14 frames.
These use different frame counts and are not a controlled paired SA/IPT comparison.

### Export evidence

Run 5.3 exported all 12 historical model/recipe combinations. Each round trip was bit
identical and every manifest digest matched. Sizes: TinyLlama 1176.0/1174.5 MiB,
SmolVLA 477.3/475.9 MiB, DiT 264.0/263.3 MiB, Pi0 4099.7/4095.1 MiB
(dynamic/static). Every historical export had **zero** BF16 opt-outs, including DiT.
They retain historical labels and must not be relabeled as the new canonical DiT export.
Pi0 export checks used 32 observations and the export's own `--n-obs`, because calibration
follows evaluated observations. Release exports must use new directories and cite the release tag.

## Historical run-4 component evidence (TinyLlama)

Historical Δ retains the reported `pt` label; its normalization/unit needs reconciliation.
R is `ref_conv` on the full split
(noise floor 0.076); C is `chip` on 4 windows.

| Component in the recipes | Run-4 variant | R | C (chip) |
|---|---|---|---|
| gptq_awq method=gptq (atlas_v1, atlas_v1_static) | `gptq_awq:method=gptq,actorder=true` | -0.21 | -0.35 |
| gptq_awq method=awq_gptq (atlas_v1_awq) | `gptq_awq:method=awq_gptq,actorder=true` | -0.45 | -1.00 |
| granularity (atlas_v1, atlas_v1_awq) | `granularity:w=channel,a=token,rule=nc` | -0.36 | -0.76 |
| vadd rn (all three, chip only) | `--chip-vadd rn --chip-vadd-sites residual` | n/a | -0.51 |
| smoothquant static (atlas_v1_static) | `smoothquant:alpha=0.5,po2=true,gain=true,static=true` | -0.37 | -0.47 (collab_v0_hw: +84% -> +0.64%) |
| smoothquant smoothing only (atlas_v1) | not measured alone in run 4; nearest: `sc_absmax` / `sq_gain` | -0.14 | -0.18 |

Excluded on that evidence:

| Variant | R | C | Why excluded |
|---|---|---|---|
| mixed_precision k=4 / k=1 | -0.52 / -0.38 | -0.82 / -0.55 | Needs the BF16 fallback ruling; opt-in only. |
| rotation (default) | -0.13 | +0.02 | Within noise on chip; not merged. |
| static_calib method=mse | +1.14 | +0.55 | Worse. |

These are single-component deltas and need not add up. Run 5.x above measures the stacks;
component labels here retain their historical meanings.

## Per model

| CLI | Model | What the recipe does there |
|---|---|---|
| `src/eval.py` | TinyLlama | Canonical AWQ folds supported groups, GPTQ rounds onto the per-channel grid; no BF16 exception. SQ/static use `partial=true`. |
| `src/eval_policy.py` | SmolVLA, Multi-Task DiT | Canonical AWQ records folded/refused groups; GPTQ and granularity act on Linear sites. Only Multi-Task DiT gets the exact canonical patch exception above. |
| `src/eval_pi0.py` | Pi0 | Canonical AWQ folds supported PaliGemma/expert groups; SigLIP groups may be refused and recorded. No automatic BF16 exception. |
| multitask-dit-quant `experiments/run_eval.py` with `ATLAS_TORCHAO_SRC` | Multi-Task DiT | That repo quantizes the DiT itself and imports only the chip router, so no `--algo` hook can act there. `ATLAS_ALGO=<recipe>` raises `NotImplementedError` with this explanation rather than running without it. The recipe's vadd component alone is `ATLAS_CHIP_VADD=rn ATLAS_CHIP_VADD_SITES=residual`. For the full recipe on the DiT, use `src/eval_policy.py --policy <Multi-Task DiT checkpoint>`. |

Calibration follows docs/algos/README.md: WikiText-2 train for TinyLlama, frames from episodes
outside the evaluated ones for the policies, and DROID or dummy observations for Pi0.

## Pending hardware rulings

| Ruling | Affects | Status |
|---|---|---|
| Run-time vs static activation scales (static-calib Q2: is a dynamic amax on chip intended at all?) | `atlas_v1` (per-token amax) vs `atlas_v1_static` | Open. This is why there are two recipes. |
| H2: how a per-row exponent is derived from `vredmax.row` (no floor / exponent-extract instruction) | `atlas_v1`, `atlas_v1_awq` per-token scales; per-tensor dynamic scales too | Open. Counted on chip as the `granularity_rescale` VPU row (NEW). |
| vadd rounding: `trunc` (as built) vs `rn` (24-instruction sequence) vs an RNE adder | Chip default of all recipes | Open. `rn` is NEW and off-ladder; `rne_hw` models the hardware change. |
| Operand exponent 15: 448 (RTL: exponent 15 finite) vs 240 (spec `exponent_range: [1,14]`) | Every FP8 cast; `rule=nc` targets 448 | Open. The reference author's numerics use 448. |
| MXU scale granularity / MX32 | per-channel weights, per-token activations | Open. |
| R4: can residual passes share the MXU accumulator? BF16 fallback on VPU or host | canonical DiT exception; `+mixed_precision` | Open. |
| SA accumulation precision for long reductions | SA alternative; IPT remains default | Open. |
| Off-chip memory size and bandwidth | exported model residency and transfer | Open. |

## Verification commands (GPU)

Run from the repository root with `PYTHONPATH=$PWD/src`. `$TL` is the TinyLlama checkpoint, `$NC`
the chip window count of run 4 (so its noise floor applies), and `$P` the Pi0 environment's python.
Each recipe is compared with `--algo none` on the same protocol.

```bash
for R in none atlas_v1 atlas_v1_sq atlas_v1_static; do
  # TinyLlama: R (reference, full split) and C (chip, run-4 windows; --kl works with mxq)
  python src/eval.py --model $TL --modes bf16 ref_conv --algo $R --out out/v1_R_tl_$R.json
  python src/eval.py --model $TL --protocol mxq --nsamples $NC --seed 0 --modes bf16 chip --kl \
      --algo $R --out out/v1_C_tl_$R.json
  # SmolVLA
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 200 --seed 0 --modes bf16 ref_conv --algo $R --out out/v1_R_smolvla_$R.json
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 32 --seed 0 --modes chip --algo $R --out out/v1_C_smolvla_$R.json
  # Multi-Task DiT (full recipe through this repo)
  python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 \
      --dataset TRI-ML/PretrainFinetune --n-frames 200 --seed 0 --modes bf16 ref_conv \
      --algo $R --out out/v1_R_dit_$R.json
  # Pi0 (atlas-pi0 environment)
  $P src/eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes bf16 ref_conv chip --algo $R \
      --out out/v1_pi0_$R.json
done
# Multi-Task DiT in multitask-dit-quant: the recipes' vadd component only (no --algo hooks there)
ATLAS_TORCHAO_SRC=<this checkout>/src ATLAS_CHIP_VADD=rn ATLAS_CHIP_VADD_SITES=residual \
  python experiments/run_eval.py --label v1_dit_vadd_rn --modes bf16 chip --n-episodes 2 \
  --n-frames 32 --seed 0 --rmse-frames 0 --device cuda
```

The chip runs of `--algo none` use `--chip-vadd trunc` (the default); every recipe's chip run
uses `rn`. To separate the vadd effect from the algorithms, add
`--algo none --chip-vadd rn --chip-vadd-sites residual` (run 4's `vadd_rn`).

## Export (GPU)

Each recipe can be shipped as files: `--export-dir DIR --export-only` quantizes the `ref_conv`
model with the recipe (calibration, folds, GPTQ), writes `model.safetensors`, `policy.json` and
`manifest.json`, and stops ([export.md](export.md)). The calibration flags are those of the
verification commands above, so the exported weights are the evaluated ones. `--from-export DIR`
re-evaluates an export without calibration.

```bash
E=~/atlas-fp8/exports/atlas-v1.0
for R in atlas_v1 atlas_v1_static; do
  python src/eval.py --model $TL --modes ref_conv --algo $R \
      --export-dir $E/tinyllama/$R --export-only --out $E/tinyllama/$R.run.json
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 200 --seed 0 --modes ref_conv --algo $R \
      --export-dir $E/smolvla/$R --export-only --out $E/smolvla/$R.run.json
  python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 \
      --dataset TRI-ML/PretrainFinetune --n-frames 200 --seed 0 --modes ref_conv --algo $R \
      --export-dir $E/multitask_dit/$R --export-only --out $E/multitask_dit/$R.run.json
  $P src/eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --algo $R \
      --export-dir $E/pi0/$R --export-only --out $E/pi0/$R.run.json
done
```
