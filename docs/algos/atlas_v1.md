# Atlas FP8 v1 recipes: `atlas_v1_awq`, `atlas_v1_sq`, `atlas_v1_static`

> **Read first.**
> - **SmolVLA results measured before `--norm-stats dataset` are on un-normalized inputs.** The
>   checkpoint's saved processors never normalized state or unnormalized actions (see
>   [the normalization protocol](#lerobot-normalization-protocol---norm-stats)). Their
>   ground-truth metrics (MSE / MAE) are invalid, and their FP8-vs-BF16 fidelity (rel L2,
>   cosine) is measured off-distribution. Every SmolVLA number on this page and in
>   [release/](release/README.md) up to and including v1.0 is of that kind.
> - **The v1.0 AWQ-vs-SmoothQuant choice was within noise on SmolVLA and Pi0** (chip margins
>   0.0159 vs a 0.020 floor and 0.0024 vs a 0.0027 floor), the TinyLlama margin depends on the
>   unresolved `pt` unit, and **the Pi0 rows mixed commits** (none at run 5, SmoothQuant and
>   static at run 5.1, AWQ at run 5.3). `l40s-run-6` re-measures all recipes at one commit.
> - **`atlas_v1` changed meaning across tags** (table below). Cite a recipe by its explicit
>   name; results and exports record the alias, the resolved name and the expanded stack.

`--algo atlas_v1_awq` (or `atlas_v1_sq`, `atlas_v1_static`, or the alias `atlas_v1`) expands to
a documented stack of the merged algorithms (`algos.RECIPES` and `algos.ALIASES` in
`src/algos/__init__.py`; stacking rules in [stacking.md](stacking.md)). The recipe itself is **NEW**: it combines tracks that each cite their
own sources, but no source defines this combination. Each component keeps the numerics of its
track, and the reference author's numerics (`src/ref_numerics.py`) stay the source of truth for
every cast.

There are dynamic and static variants because the hardware ruling on run-time activation scales
is open: `atlas_v1_awq` and `atlas_v1_sq` need an amax on chip at run time (per token);
`atlas_v1_static` needs none for the Linear inputs.

## The recipes

Each recipe name denotes exactly one stack and never changes meaning (NEW rule, from v1.1): a
different stack gets a new name. `tests/test_algo_atlas_v1.py` pins every spec verbatim. No
recipe keeps any layer in BF16.

| Recipe | Expands to | Chip default |
|---|---|---|
| `atlas_v1_awq` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | `--chip-vadd rn --chip-vadd-sites residual` |
| `atlas_v1_sq` | `smoothquant:alpha=0.5,po2=true,gain=false,partial=true` + `gptq_awq:method=gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | the same |
| `atlas_v1_static` (no on-chip amax for Linear inputs) | `smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true` + `gptq_awq:method=gptq,actorder=true` | the same |

| Alias | Resolves to (v1.1) |
|---|---|
| `atlas_v1` | `atlas_v1_awq` (unchanged target until run 6 re-picks the recipe; NEW) |

Every result entry (`algo.recipe`) and every export manifest (`recipe`) records the requested
token, the `alias` (null when an explicit name was given), the `resolved` recipe, the recipe's
own spec and the fully `expanded` spec, including opt-in members.

### Recipe names by tag

| Tag / commit | `atlas_v1` meant | `atlas_v1_sq` | `atlas_v1_awq` | `atlas_v1_static` |
|---|---|---|---|---|
| `l40s-run-5` (`0447807`), `l40s-run-5.1` (`46b0b1d`), `l40s-run-5.2` (`4ab3889`), `l40s-run-5.3` (`a30779b`), local `atlas-v1-handoff` (`aa679e2`) | SmoothQuant+GPTQ+granularity (today's `atlas_v1_sq`) | did not exist | as today | as today |
| `5ff028e` (local tag `atlas-v1.0`, never published), `fx-gpu-check-1` and main at `2d4b187` / merge `54e1855`, `characterization-gpu-pilot-1-2026-10-14` (`a99a062`) | AWQ+GPTQ+granularity **plus** a BF16 exception for the Multi-Task DiT CLIP patch embedding, keyed on the policy type `multi_task_dit` | as today | as today (no exception) | as today |
| v1.1 (this branch) | alias of `atlas_v1_awq`, no exception | as today | as today | as today |

All chip defaults (`rn`, `residual`) are the same at every tag. A result labelled `atlas_v1`
without a recorded resolved name must be read against its source tag in this table.

- **Chip default.** Under a chip profile, the recipe sets `--chip-vadd rn --chip-vadd-sites
  residual` unless the command line passes `--chip-vadd` (an explicit value, `trunc` included,
  wins). `--chip-vadd` defaults to unset on this branch so that the difference is visible; unset
  still means `trunc`. The record lists what was applied in
  `algo.recipe.chip_options_applied`, and the ladder marks the step off-ladder as for any
  `--chip-vadd rn` run.
- **Choice (NEW).** Run-5.x AWQ+GPTQ has the lowest measured chip errors on TinyLlama,
  SmolVLA and Pi0. The supplied table marks each chip improvement over `none` outside
  the reported noise floor, subject to the protocol/unit qualifications below. This is a
  ranking of measured errors, within noise of SmoothQuant on SmolVLA and Pi0 (see the note at
  the top); no DiT chip recipe measurement was supplied.
- **Historical smoothing.** SmoothQuant smoothing and AWQ both fold per-channel
  factors into the same norms (`input_layernorm`, `post_attention_layernorm`) and producers
  (`up_proj` rows). The stack refuses that combination (stacking.md). `atlas_v1_sq` preserves
  the previous smoothing plus GPTQ recipe. Historical run tags keep their original meanings.
- **Why no granularity in `atlas_v1_static`.** A per-token scale is a run-time scale. The static
  recipe keeps the per-tensor weight exponent that GPTQ pins, and the activation exponent of every
  calibrated Linear input is static (smoothquant `static=true`, computed from the post-fold
  calibration maximum by the reference rule). Attention operands keep the reference dynamic
  per-(batch, head) rule, so `atlas_v1_static` still needs an amax there (open).
- **Weight grid.** In `atlas_v1_awq` and `atlas_v1_sq`, granularity chooses the per-channel weight
  exponent and GPTQ quantizes onto it and pins it (the weight-grid delegation in stacking.md).
- **BF16 input-convolution exception (opt-in, NEW).** v1.0 `atlas_v1` kept the Multi-Task DiT
  CLIP patch embedding in BF16, selected by the policy type string and a module path. v1.1
  removes it from every recipe. The general replacement is the structural rule
  [`keep_bf16:rule=input_conv`](keep-bf16.md): every `Conv2d` that reads image pixels directly
  (`in_channels == 3`), kept by exact fqn, validated before calibration, with the rule's own reason
  in the result and export. Use `--algo atlas_v1_awq+keep_bf16:rule=input_conv`. Run 6 measures
  it on the DiT. BF16 has no MXU operand path: placement on the VPU or host remains pending R4.
- **Other mixed precision.** fp8x3 residual passes need ruling R4 (whether passes can share the MXU
  accumulator). Opt in with `+`, for example
  `--algo atlas_v1_awq+mixed_precision:rank_file=PATH,k=4` (ranking from `scripts/rank_layers.py`;
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

**SmolVLA rows above predate the normalization protocol (next section) and are invalid as task
metrics.** They are model-space
fidelity on unnormalized inputs: SmolVLA received raw joint degrees as state and its actions were
never unnormalized. Their rel L2 / cosine against BF16 compare two quantizations of that same
unnormalized computation; their ground-truth MSE / MAE (on the order of 2000) are invalid and
must not be cited. The TinyLlama, Pi0 and DiT rows are unaffected.

## LeRobot normalization protocol (`--norm-stats`)

`lerobot/smolvla_base`'s saved processors declare STATE / ACTION `MEAN_STD` but store stats only
under `so100{,-blue,-red}.buffer.action.{mean,std}`. LeRobot 0.6.1 passes a tensor through
unchanged when its key has no stats (`lerobot/processor/normalize_processor.py:330-331`), so with
the checkpoint's processors nothing is normalized and nothing fails. The `so100` stats do not match
`svla_so100_pickplace` and are not aliased.

`src/eval_policy.py` (and `src/policy_norm.py`) therefore:

- `--norm-stats checkpoint` (default): the processors as saved.
- `--norm-stats dataset` (NEW): a zero-shot evaluation protocol for a pretraining checkpoint,
  mirroring what a fine-tune would start from. The normalizer and unnormalizer stats are replaced
  with the evaluated dataset's `LeRobotDatasetMetadata(...).stats` through the overrides
  `lerobot_train` passes when it fine-tunes from a pretrained checkpoint
  (`lerobot/scripts/lerobot_train.py:355-377`: features, `norm_map` and `stats` of
  `normalizer_processor` / `unnormalizer_processor`). The stats' camera keys follow the rename map,
  as in LeRobot's rollout path (`rename_stats`, `lerobot/rollout/context.py:461`).
- A guard (NEW), for every protocol, right after the processors are built: every feature whose
  mode is not `IDENTITY` must have finite stats under the exact key the step looks up (the action
  under `action`) in both the normalizer and the unnormalizer, with a positive std (`MEAN_STD`) or
  range (`MIN_MAX`, `QUANTILES`, `QUANTILE10`). Otherwise the run refuses, names the keys and
  suggests `--norm-stats dataset`. SmolVLA is refused with `checkpoint`; the Multi-Task DiT
  (literal `MIN_MAX` state / action and `MEAN_STD` image stats) passes with `checkpoint`.
- A probe (NEW): one evaluated frame through the preprocessor (each normalized observation must
  change) and one action chunk through the postprocessor and back through the normalizer (must be
  the identity within 1e-4 relative).
- `--allow-unnormalized` (NEW, `checkpoint` only) reproduces the historical behaviour for
  comparison; the result JSON says `"normalization": "legacy-unverified"` at the top and in every
  mode.

Every result records `normalization` (status) and `normalization_protocol`: the protocol, the stats
source (dataset repo, revision and resolved snapshot, or the checkpoint id), the sha256 of each
key's canonically serialized stats and their combined digest, the guard and the probe.
Calibration frames go through the same processors and record the same digest. Exports carry the
protocol and the digest, and `--from-export` refuses an export made under another protocol or with
other stats ([export.md](export.md)).

### DiT after checkpoint repair (run 5.2, `4ab3889`)

The repaired checkpoint remapped 395 CLIP keys. Earlier DiT measurements are superseded.
At 200 frames from episodes 136/363/380/464/474, BF16 GT MSE was 0.01822.

| Historical recipe, ref_conv only | rel L2 vs BF16 | vs none | GT MSE | BF16 opt-outs |
|---|---|---|---|---|
| none | 0.1572 | — | 0.01890 | 0 |
| SQ / AWQ, bit identical | 0.1394 | -0.0178, within noise | 0.02025 | 0 |
| static | 0.1514 | -0.0058, within noise | 0.01955 | 0 |
| SQ + patch embedding BF16 | 0.0883 | -0.0689, outside 2σ | 0.01828 | 1 |

Labels: "SQ" is `atlas_v1` at `4ab3889` (today's `atlas_v1_sq`); "AWQ" is `atlas_v1_awq`.

The 2σ threshold is 0.0559 rel L2, from repaired multitask-dit-quant Set A
(`3baa5d6`), a separate harness; it is an empirical comparison threshold, not a paired
significance test. SQ folded nothing and AWQ recorded `awq_folded: []`, making their
unexcluded outputs bit identical. Applying the measured patch exception to AWQ was NEW in v1.0
and is an opt-in rule in v1.1 (`keep_bf16:rule=input_conv`).
The v1.0 full-checkpoint export at `5ff028ee58b28606873278236a379dcc09339b82`
passed an eight-frame round trip: all 96 captured tensors were byte-identical, action
statistics matched, and outputs were finite and deterministic. This verifies export replay;
it does not add a new 200-frame accuracy measurement. No DiT chip recipe result is available.

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
Their `atlas_v1` label means `atlas_v1_sq` (run 5.3 tag); they are not the v1.0 DiT export.
Pi0 export checks used 32 observations and the export's own `--n-obs`, because calibration
follows evaluated observations. Release exports must use new directories and cite the release tag.

The v1.0 DiT release export (v1.0 `atlas_v1`, with the patch exception) has 204 total sites: 179 quantized Linear, 24 quantized
attention, one BF16 Conv2d and zero skipped sites. Its manifest SHA256 is
`5cbb07ef1e892af7bd5acb56accd8b9e94c1bcc8bded1e8ce05e1454ed59b629`.
The final GPU release checks report 75 passed, zero failures and five main-environment
skips whose cases passed in the Pi0 environment. Both compilation checks passed.
See the [release contents](release/RELEASE_CONTENTS.md), [complete results](release/RESULTS_SUMMARY.md)
and [verification record](release/release_verification.json). The `atlas-v1.0` tag exists
only in a local checkout and was never published; cite commit `5ff028e`. The release was merged
to `main` by pull request #1 (merge `54e1855`).

## Historical run-4 component evidence (TinyLlama)

Historical Δ retains the reported `pt` label; its normalization/unit needs reconciliation.
R is `ref_conv` on the full split
(noise floor 0.076); C is `chip` on 4 windows.

| Component in the recipes | Run-4 variant | R | C (chip) |
|---|---|---|---|
| gptq_awq method=gptq (atlas_v1_sq, atlas_v1_static) | `gptq_awq:method=gptq,actorder=true` | -0.21 | -0.35 |
| gptq_awq method=awq_gptq (atlas_v1_awq) | `gptq_awq:method=awq_gptq,actorder=true` | -0.45 | -1.00 |
| granularity (atlas_v1_sq, atlas_v1_awq) | `granularity:w=channel,a=token,rule=nc` | -0.36 | -0.76 |
| vadd rn (all three, chip only) | `--chip-vadd rn --chip-vadd-sites residual` | n/a | -0.51 |
| smoothquant static (atlas_v1_static) | `smoothquant:alpha=0.5,po2=true,gain=true,static=true` | -0.37 | -0.47 (collab_v0_hw: +84% -> +0.64%) |
| smoothquant smoothing only (atlas_v1_sq) | not measured alone in run 4; nearest: `sc_absmax` / `sq_gain` | -0.14 | -0.18 |

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
| `src/eval_policy.py` | SmolVLA, Multi-Task DiT | AWQ records folded/refused groups; GPTQ and granularity act on Linear sites. No BF16 exception unless `+keep_bf16:rule=input_conv` is given. |
| `src/eval_pi0.py` | Pi0 | AWQ folds supported PaliGemma/expert groups; SigLIP groups may be refused and recorded. No BF16 exception unless requested. |
| multitask-dit-quant `experiments/run_eval.py` with `ATLAS_TORCHAO_SRC` | Multi-Task DiT | That repo quantizes the DiT itself and imports only the chip router, so no `--algo` hook can act there. `ATLAS_ALGO=<recipe>` raises `NotImplementedError` with this explanation rather than running without it. The recipe's vadd component alone is `ATLAS_CHIP_VADD=rn ATLAS_CHIP_VADD_SITES=residual`. For the full recipe on the DiT, use `src/eval_policy.py --policy <Multi-Task DiT checkpoint>`. |

Calibration follows docs/algos/README.md: WikiText-2 train for TinyLlama, frames from episodes
outside the evaluated ones for the policies, and DROID or dummy observations for Pi0.

## Matrix unit choice: the inner-product tree for every linear layer

Run every Linear on the inner-product tree (MXU1, `--mxu ipt`, the default) and treat the systolic array (MXU0, `--mxu sa`) as an ablation. The RTL gives the reasons (audit: `handoff/HW_AWARE_AUDIT.md`):

- **Rounding.** The IPT aligns all 32 products and the incoming BF16 partial sum to one anchor in a 32-bit integer (17 guard bits below the largest product, shifted-out bits truncated), sums them exactly and rounds once per 32-wide K tile (`AnchorAccumulationTree.scala`). The SA's elaborated PE (`CustomFMA`, `SystolicArrayParams.scala:75`) rounds to BF16 after every one of the 32 adds in a tile. For K = 4416 (138 tiles) that is 138 roundings against 4416, which is the measured +3.8 pt TinyLlama cost and the DiT adaLN degradation of the SA in earlier runs.
- **Latency.** A matmul takes 35 cycles on MXU1 and 95 on MXU0 (`rtl-timing.md:51`; spec 02 says 96, the RTL-calibrated 95 is used). The issue gap is 32 cycles on both, so throughput per MXU is the same; the IPT is the lower-latency path.
- **Selection.** Software picks the unit per instruction (`Instructions.scala:99-113`), so this is a scheduling and recipe decision, not a hardware ruling. Whether the taped-out tile contains both units should still be confirmed with the hardware team; the standalone elaboration stub (`chipyard/top/AtlasCluster.scala`) instantiates only the IPT.
- **Evidence.** The IPT numerics were checked bit for bit against the RTL under Verilator on 14,780 tiles from this flow (`scripts/rtl/`, `results/hw-verify/l1.json`).

## Pending hardware rulings

| Ruling | Affects | Status |
|---|---|---|
| Run-time vs static activation scales (static-calib Q2: is a dynamic amax on chip intended at all?) | `atlas_v1_awq` / `atlas_v1_sq` (per-token amax) vs `atlas_v1_static` | Open. This is why there are two recipes. |
| H2: how a per-row exponent is derived from `vredmax.row` (no floor / exponent-extract instruction) | `atlas_v1_awq`, `atlas_v1_sq` per-token scales; per-tensor dynamic scales too | Open. Counted on chip as the `granularity_rescale` VPU row (NEW). |
| vadd rounding: `trunc` (as built) vs `rn` (24-instruction sequence) vs an RNE adder | Chip default of all recipes | Settled for the RTL as built: it truncates (`AddSubSumVec.scala:133`). Open as a request: RNE narrowing there would replace the 24-instruction `rn` sequence with one instruction (`rne_hw`). |
| Operand exponent 15: 448 (RTL: exponent 15 finite) vs 240 (spec `exponent_range: [1,14]`) | Every FP8 cast; `rule=nc` targets 448 | Settled by the RTL: 448 (`FP8Pack.scala:40-41`, `E4M3Mul.scala:34-43`). The Merlin software spec's 240 is wrong. |
| MXU scale granularity / MX32 | per-channel weights, per-token activations | Settled by the RTL: the matmul reads no scale; one E8M0 per 32x32 tile applies only at `vmatpop.fp8` (multiply), `vpack` (divide) and `vunpack` (multiply). Per-channel and per-token scales are vector-unit work. |
| R4: can residual passes share the MXU accumulator? BF16 fallback on VPU or host | `+keep_bf16`; `+mixed_precision` | Open. |
| SA accumulation precision for long reductions | SA alternative; IPT remains default | Resolved as a software choice: the instruction's funct7 LSB selects the unit (`Instructions.scala:99-113`), see "Matrix unit choice". |
| Off-chip memory size and bandwidth | exported model residency and transfer | Open. |

## Verification commands (GPU)

Run from the repository root with `PYTHONPATH=$PWD/src`. `$TL` is the TinyLlama checkpoint, `$NC`
the chip window count of run 4 (so its noise floor applies), and `$P` the Pi0 environment's python.
Each recipe is compared with `--algo none` on the same protocol.

```bash
for R in none atlas_v1_awq atlas_v1_sq atlas_v1_static; do
  # TinyLlama: R (reference, full split) and C (chip, run-4 windows; --kl works with mxq)
  python src/eval.py --model $TL --modes bf16 ref_conv --algo $R --out out/v1_R_tl_$R.json
  python src/eval.py --model $TL --protocol mxq --nsamples $NC --seed 0 --modes bf16 chip --kl \
      --algo $R --out out/v1_C_tl_$R.json
  # SmolVLA (its checkpoint stats are refused by the guard; see the normalization protocol)
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 200 --seed 0 --modes bf16 ref_conv --algo $R --norm-stats dataset \
      --out out/v1_R_smolvla_$R.json
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 32 --seed 0 --modes chip --algo $R --norm-stats dataset --out out/v1_C_smolvla_$R.json
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
E=~/atlas-fp8/exports/<tag>
for R in atlas_v1_awq atlas_v1_sq atlas_v1_static; do
  python src/eval.py --model $TL --modes ref_conv --algo $R \
      --export-dir $E/tinyllama/$R --export-only --out $E/tinyllama/$R.run.json
  python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
      --n-frames 200 --seed 0 --modes ref_conv --algo $R --norm-stats dataset \
      --export-dir $E/smolvla/$R --export-only --out $E/smolvla/$R.run.json
  python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 \
      --dataset TRI-ML/PretrainFinetune --n-frames 200 --seed 0 --modes ref_conv --algo $R \
      --export-dir $E/multitask_dit/$R --export-only --out $E/multitask_dit/$R.run.json
  $P src/eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --algo $R \
      --export-dir $E/pi0/$R --export-only --out $E/pi0/$R.run.json
done
```
