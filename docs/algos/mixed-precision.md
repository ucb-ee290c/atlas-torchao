# `--algo mixed_precision`: sensitivity-driven mixed precision

This algorithm keeps the few layers that are most sensitive to FP8 out of single-pass FP8. The layer
choice comes from a ranking file built offline on the calibration split. A chosen layer is either
opted out to BF16 (`keep_as=bf16`), or run as two or three FP8 passes on the MXU over its operand
residuals (`keep_as=fp8x3` / `fp8x2_w`).

## Usage

```
python src/scripts/rank_layers.py --model $TINYLLAMA --seqlen 512 --calib-n 4 --calib-seed 1234 --device cuda \
    --out rank.json
python src/eval.py --model $TINYLLAMA --modes bf16 ref_linear chip \
    --algo mixed_precision:rank_file=rank.json,k=1                  # or mac_budget=0.05
python src/eval.py ... --algo mixed_precision:rank_file=rank.json,k=4,keep_as=fp8x3
python src/eval.py ... --algo "mixed_precision:units=model.layers.2.mlp.down_proj"
```

| Parameter | Meaning |
|---|---|
| `rank_file` | JSON from `src/scripts/rank_layers.py`. Required unless `units` is given. |
| `score` | `fisher_out` (default), `hawq_w` or `iso_rel_rmse`. |
| `k` / `mac_budget` | Top k layers, or the top layers fitting into a share of the ranked layers' MACs (greedy in score order). Give exactly one. |
| `units` | Explicit fqns separated by `;` (no ranking). |
| `keep_as` | `bf16` (default), `fp8x3` or `fp8x2_w`. |

Each result's `algo.selection` records:

- the units;
- the ranking provenance (path, sha256 and the file's `meta`: model, profile, calibration windows,
  commit, versions);
- `mac_share_selected` and `mac_share_unquantized`;
- `extra_mxu_passes`.

Every fqn must name a Linear or Conv2d, and must not be a substring of any other module's name.
Profiles match `skip` entries by substring, so `layers.1.` would also catch `layers.11.`. Such a
selection is refused.

## Scores (`algos.mixed_precision.score_layers`)

| Score | Definition | Source |
|---|---|---|
| `fisher_out` | ½ Σ_t (∂L/∂y_t · e_t)². e_t is the layer's local FP8 output error on the clean input, under the profile's cast. | NEW: an empirical-Fisher / Gauss-Newton estimate of ΔL that covers the activation cast as well as the weight cast. |
| `hawq_w` | Tr(F_i)/n_i · ‖Q(W_i) − W_i‖², with Tr(F_i) = Σ_t ‖g_t‖² ‖x_t‖² (per-token Fisher trace of a Linear). | HAWQ-V2, arXiv 1911.03852, Sec. 2.3, Eq. 2.11. Replacing the Hessian trace with the Fisher trace is NEW. |
| `iso_rel_rmse` | Isolated relative RMSE. | `src/layer_rmse.py` (the reference author's StatsTracker, smolvla_inout/stats_tracker.py:181-237). |

L is the summed token NLL on WikiText-2 **train** windows from `algos.calib.wikitext2_train`, so it
never touches the evaluation's test split.

Screening on TinyLlama with ref_linear: 4 test windows × 512, mixed-precision.md in the handoff.

- `fisher_out` top-1 is `layers.2.mlp.down_proj`. Keeping it in BF16 closes about 72% of the ppl
  gap at 1.2% of linear MACs.
- iso rel-RMSE top-1 closes 18%.
- Leave-one-out at this sample size is noise.

**Hook order.** `algos.quantize` calls `bf16_layers` before `calibrate`, so this algorithm reads a
ranking computed earlier instead of computing one in the run. The proposed base option, which was
not made here, is `Algo.calibrate_first = True`: an algorithm sets it to have `calibrate` run before
`bf16_layers`. This is additive, and every existing algorithm keeps the current order.

## `keep_as`: what "not FP8" can mean on Atlas

The MXU takes E4M3 operands only (ISA green card 77/10-13); there is no BF16-operand mode.

**`bf16`.** The layer goes into `skip` and is listed in `algo.bf16_opt_out`. The chip coverage
report counts it as host work on `<op>@bf16_opt_out` rows. Running it on the VPU instead would cost
an estimated 2–3.6 lane-ops per MAC; that estimate is NEW (handoff mixed-precision.md "Fit to
Atlas"). The VPU route is only accurate if every partial sum is reduced by `vredsum.row` trees.
A `vadd` chain truncates (AddSubSumVec.scala:133) and comes out *less* accurate than FP8 in
screening. Hardware rulings R1–R3 apply.

**`fp8x3`** computes x_hi·w_hi + x_hi·w_lo + x_lo·w_hi:

- w_lo = Q(w − w_hi) and x_lo = Q(x − x_hi). Each part uses the reference cast
  (`ref_numerics.quantize`, its own per-tensor po2 scale).
- Each pass goes through the step's matmul path: framework accumulation on the framework steps, and
  the MXU model (`--mxu`) under reference_hw and chip.
- Each pass is rescaled by 2^(s_x+s_w), and the passes are summed in the layer dtype.
- Source: the precision refinement of Markidis et al., arXiv 1803.04014, Sec. V, applied with FP8
  parts. Two FP8 parts in place of one BF16 operand is NEW.

`fp8x2_w` uses the weight residual only (two passes).

**Chip coverage of `fp8x3`:**

- Each pass is an MXU call, so the MXU `linear` row grows by 2× the layer's FLOPs (tested).
- The activation residual is a VPU `sub` plus an `fp8_operand_cast`. The rescales and sums are
  VPU `mul` / `add` rows, and `add` is the truncating `vadd`.
- Summing the passes in the MXU accumulator instead (`vmatpush.acc.bf16`, with differing scales)
  is hardware ruling **R4**.

`fp8x3` and `fp8x2_w` are Linear-only: Conv2d is refused. They act only on the reference steps
(ref_linear … chip). Under collab_v0 and torchao the record notes that the module hook did not
apply.

## Proposed algo/base change: hook (g) `residual_passes`

This change is on `algo/mixed-precision` only, kept to two files for integration.

- **`src/algos/base.py`:** `Algo.residual_passes(name, kind) -> None | "w" | "wx"`, a no-op by
  default, added to `module_hooks`.
- **`src/reference/ref_profile.py`:** `RefQuantLinear` reads the hook once at construction. If the hook
  returns non-None, it builds the residual weight buffers and routes forward through
  `_forward_residual`.
  - With `--algo none` (`algo is None`) or any algorithm that does not override the hook, the
    existing code paths are untouched.
  - tests/test_algo_base.py (bit-identity to the baseline commit fcfd7d8) and the example_identity test pass.
- **Conflicts with the granularity branch:** none. It does not touch `algos.base.output_scale` or
  `chip_profile.count_algo_vpu`, the two points the granularity branch changed. The coverage
  accounting comes from the existing `_GuardedFM` MXU counting and VPU dispatch.

## Tests (`tests/test_algo_mixed_precision.py`)

- k / budget selection with deterministic tie-breaks; exact-name and substring-collision refusal.
- The rank file drives `bf16_opt_out`, and the record carries provenance and the MAC share.
- k=0 is bit-identical to `--algo none` on ref_linear, reference_hw and chip.
- The ranking is reproducible for fixed calibration batches.
- FP8x3 equals its definition bit for bit, and beats single FP8 (and FP8x2_w does not lose) on
  synthetic outliers.
- Chip MXU work grows by exactly 2× the layer FLOPs under FP8x3.
- Conv2d is refused, and collab_v0 records the module-hook note.
