# Atlas FP8 recipes: `atlas_v1`, `atlas_v1_awq`, `atlas_v1_static`

`--algo atlas_v1` (or `atlas_v1_awq`, `atlas_v1_static`) expands to a documented stack of the
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
| `atlas_v1` (dynamic scales) | `smoothquant:alpha=0.5,po2=true,gain=false,partial=true` + `gptq_awq:method=gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | `--chip-vadd rn --chip-vadd-sites residual` |
| `atlas_v1_awq` (GPU comparison) | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | the same |
| `atlas_v1_static` (no on-chip amax for Linear inputs) | `smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true` + `gptq_awq:method=gptq,actorder=true` | the same |

- **Chip default.** Under a chip profile, the recipe sets `--chip-vadd rn --chip-vadd-sites
  residual` unless the command line passes `--chip-vadd` (an explicit value, `trunc` included,
  wins). `--chip-vadd` defaults to unset on this branch so that the difference is visible; unset
  still means `trunc`. The record lists what was applied in
  `algo.recipe.chip_options_applied`, and the ladder marks the step off-ladder as for any
  `--chip-vadd rn` run.
- **Why gptq and not awq_gptq in `atlas_v1`.** SmoothQuant smoothing and AWQ both fold per-channel
  factors into the same norms (`input_layernorm`, `post_attention_layernorm`) and producers
  (`up_proj` rows). The stack refuses that combination (stacking.md), so `atlas_v1` uses GPTQ
  alone after smoothing. `atlas_v1_awq` drops smoothing and keeps AWQ+GPTQ, so the GPU run can
  compare the two.
- **Why no granularity in `atlas_v1_static`.** A per-token scale is a run-time scale. The static
  recipe keeps the per-tensor weight exponent that GPTQ pins, and the activation exponent of every
  calibrated Linear input is static (smoothquant `static=true`, computed from the post-fold
  calibration maximum by the reference rule). Attention operands keep the reference dynamic
  per-(batch, head) rule, so `atlas_v1_static` still needs an amax there (open).
- **Weight grid.** In `atlas_v1` and `atlas_v1_awq`, granularity chooses the per-channel weight
  exponent and GPTQ quantizes onto it and pins it (the weight-grid delegation in stacking.md).
- **Mixed precision is not in any recipe.** Its BF16 fallback has no MXU path (the MXU has no BF16
  operand mode), and its fp8x3 residual passes need ruling R4 (whether passes can share the MXU
  accumulator). Opt in with `+`, for example
  `--algo atlas_v1+mixed_precision:rank_file=PATH,k=4` (ranking from `scripts/rank_layers.py`;
  causal LM only).
- Recipes take no parameters (`atlas_v1:alpha=0.3` is refused). To change a component, write the
  stack out.

## Run-4 evidence (TinyLlama)

Δ is the change in perplexity points relative to no algorithm. R is `ref_conv` on the full split
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

The stacked recipes themselves have not been measured. The effects above are single-algorithm
deltas and need not add up; the GPU run below measures the stacks.

## Per model

| CLI | Model | What the recipe does there |
|---|---|---|
| `src/eval.py` | TinyLlama | All components. `partial=true` folds every decoder layer, as `partial=false` would; `prepare.smoothquant.not_folded_blocks` is `[]`. |
| `src/eval_policy.py` | SmolVLA, Multi-Task DiT | Smoothquant folds the supported decoder layers (SmolVLA's VLM + expert) and lists every SigLIP / DiT block it left unfolded in `prepare.smoothquant.not_folded_blocks`. GPTQ and granularity apply to every Linear; AWQ records each refused fold group. |
| `src/eval_pi0.py` | Pi0 | As for SmolVLA (PaliGemma + expert folded, SigLIP listed). |
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
| R4: can residual passes share the MXU accumulator? BF16 fallback | `+mixed_precision` opt-in | Open. |

## Verification commands (GPU)

Run from the repository root with `PYTHONPATH=$PWD/src`. `$TL` is the TinyLlama checkpoint, `$NC`
the chip window count of run 4 (so its noise floor applies), and `$P` the Pi0 environment's python.
Each recipe is compared with `--algo none` on the same protocol.

```bash
for R in none atlas_v1 atlas_v1_awq atlas_v1_static; do
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
E=~/atlas-fp8/exports/atlas_v1
for R in atlas_v1 atlas_v1_awq atlas_v1_static; do
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
