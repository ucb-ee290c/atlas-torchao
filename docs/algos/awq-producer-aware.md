# AWQ on a panel grid: scoring the producer (`gptq_awq:producer_aware=true`) (NEW, default off)

Everything here is **NEW**; no paper defines it. Measured on CPU, 2026-11-06, branch `tile-awq-producer`. Numbers marked *proxy* come from a
weights-only stand-in for the calibration data (stated below), not from a calibration run.

## The gap

`gptq_awq` folds AWQ's `1/s_j` into the producer row `j` of an `up_proj -> down_proj` pair and `s_j` into the matching `down_proj` column.
`awq_search` picked `s` from the consumers only (`Ws = [mods[c].weight ...]` in `prepare`): the loss is `tr(dW D H D dW^T)` of `down_proj`.
Under `w=panel` one exponent covers 32 output channels, so a row with a large `s_j` (AWQ-salient) becomes the smallest row of its panel in the
folded basis and is flushed or coarsely rounded; per-channel exponents absorb this. The search never saw it
(`docs/algos/tile-pi0-awq-diag.md`).

## Design chosen: producer error on the producer's own grid in the search loss

For each alpha the search now also computes, for the group's producer rows, `e = s * (Q(W_p / s) - W_p / s)` where `Q` is the cast on the
producer's weight grid (`weight_grid` of a stacked `granularity:w=panel`, else the per-tensor rule) applied to what the folded checkpoint holds
(`bf16(W_p / s)`), and adds `sum_j c_j e_j H_p e_j^T` to the consumer loss (`src/algos/gptq_awq.py`, `awq_search(..., producer=)`).
`H_p` is the Hessian of the producer's input (already collected), rescaled `D H D` with `D = diag(1/s_in)` when the norm group was folded first (it scales
the `up_proj` columns before this pair is searched; `transform_weight` does the same for GPTQ; `test_producer_hessian_follows_the_norm_group_fold`). `c_j = ||W_down[:, j]||^2 * g_j`, `g_j = H_down[j, j] / (W_p[j] H_p W_p[j]^T)`
(`producer_row_weights`): the channel's second moment at the down input relative to the producer's alone, which carries the gate
(`down_in = act(gate) * up`) without a new calibration pass. Same units as the consumer loss (squared output error summed over tokens). Assumptions:
errors of different producer rows are uncorrelated (diagonal in `j`), and the gate is independent of the row's error. Only `up_proj -> down_proj`
groups get the term (a norm producer is a 1-D BF16 weight). `awq_loss=cast` plus `producer_aware=true` raises. Default off: `producer=None`
leaves `awq_search` bit-identical (`test_producer_none_is_the_unchanged_search`, `test_default_off_prepare_is_bit_identical`), and the record
gains `producer_aware: true` only when on. A larger `s` now costs the producer rows' panel error, so the search backs off alpha where that error
outweighs the consumer gain.

Why not the permutation (sort intermediate channels so a panel holds similar `s_j`): it is lossless and also fixes a layer whose salient rows
are scattered, but (1) it reorders `gate_proj`/`up_proj` rows and `down_proj` columns in the exported tensors, so every consumer of stock
channel order breaks (`scripts/tile_diag/*` and `weights_only.py` compare to the stock checkpoint row by row; any compiler or FX check
against stock layout would need the permutation recorded and honored); I did not verify what the compiler/FX contract requires, so that risk is
open; (2) it needs a rule that also keeps `s` spread inside a panel small, while the search term needs none; (3) it does nothing for a panel
whose `s_j` all differ widely. It is complementary and not implemented here.

## Use

`--algo "gptq_awq:method=awq_gptq,actorder=true,producer_aware=true+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib"`
(no new named recipe: recipe names are pinned by tests). Cost per alpha: one more cast of the producer `[I, K]` and, with a dense `H_p`, one
`[I, K] x [K, K]` product (Gemma 2B LM: 16384 x 2048 x 2048, about 69 GFLOP per alpha per layer); not timed on a real calibration.

## Validation

Tests: `tests/test_awq_producer_aware.py` (10 tests: 9 passed with the reference submodules present; without them the end-to-end test skips via `need_refs`).
Synthetic pair (I=64, K=48, three channels with 10^4x the activation magnitude): old search alpha 0.85 leaves 3 rows with 24-30% error on the
panel grid (0 on a per-channel grid); new search alpha 0.55 leaves 0.

### Pi0 Gemma MLP pairs, weights only (`scripts/tile_diag/awq_producer_validate.py`, `awq_producer_g0_rows.py`)

Stock Pi0 weights; `s_j` of the recorded tile export (`exports-tile/pi0/atlas_v1_awq_tile_static`) recovered from the export's `down_proj` columns
(median of `|dq_down[i, j] / W_down[i, j]|`), the norm group's `s_in` from the `up_proj` columns (not searched here). Real activations are not
recoverable without a Pi0 forward pass, so the search uses a **proxy**: `H_down = diag(sx^2)` with `sx = s^(1/alpha_recorded)`, `H_p = I` on the norm-folded
`R` (the LM0 `s_in` is nearly flat, median 2^5.61, so the `s_in` rescale above would change this proxy little).
Casts are RTN (`qdq`), no GPTQ. "Damaged" = producer row with `s_j * Q(R_j / s_j)` more than 10% (relative L2) from the norm-folded stock row `R_j`.
`comp` = `||Dd Du - Wd R|| / ||Wd R||`. Panel = `rows=32, rule=nc`; control = per-channel exponents, same rule. Raw: `exports-tile-diag/pi0/results_producer/`.

| layer | search | alpha | damaged rows, panel | comp, panel | comp, per-channel (same s) |
|---|---|---|---|---|---|
| LM0 | recorded export s | 0.8 | 2 | 0.3076 | 0.1317 |
| LM0 | old (proxy) | 0.9 | 4 (1 all-zero) | 0.6047 | 0.2502 |
| LM0 | new (proxy) | 0.45 | 0 | 0.0277 | 0.0252 |
| E1 | recorded export s | 0.15 | 0 | 0.0227 | 0.0227 |
| E1 | old (proxy) | 0.3 | 104 | 0.0491 | 0.0285 |
| E1 | new (proxy) | 0.1 | 0 | 0.0227 | 0.0227 |
| E2 | recorded export s | 0.8 | 1 | 0.0310 | 0.0262 |
| E2 | old (proxy) | 0.3 | 0 | 0.0247 | 0.0247 |
| E2 | new (proxy) | 0.0 | 0 | 0.0247 | 0.0247 |
| LM11 | recorded export s | 0.8 | 0 | 0.0370 | not run |
| LM11 | old / new (proxy) | 0.0 / 0.0 | 0 / 0 | 0.0253 / 0.0253 | not run |

LM = PaliGemma language model, E = Gemma expert; layers chosen from `docs/algos/tile-pi0-awq-diag.md` and `exports-tile-diag/pi0/results/comp.json`
(LM0 worst, then E2, E1, LM11).

Reading, and what it does not show:
* The proxy search does not reproduce the recorded alpha (LM0 0.9 vs 0.8, E1 0.3 vs 0.15, E2 0.3 vs 0.8, LM11 0.0 vs 0.8), so "old (proxy)" is not
  the old export. What the rows show is the mechanism: where the old objective picks an alpha whose `s` damages panel rows (LM0, E1), the
  producer term lowers alpha (0.9 -> 0.45, 0.3 -> 0.1) and the damaged rows and the composite error go to the per-channel level; where nothing is damaged
  (E2, LM11) it changes nothing that matters.
* The 53 rows of the diag doc reproduce from the exports (panel export vs per-channel control export, rows differing >10%: 53 of 16384). Under the
  recorded `s` these rows sit a median 6.5 and a max 12.7 binades below their 32-row panel amax, with up to 86% of their entries flushed to zero. At
  alpha 0.45 (`s = sx^0.45`, normalized as the search does) they sit at most 7.0 binades below, and at most 2.2% of their entries flush.
  Only 2 of the 53 are >10% from stock under RTN with the recorded `s`; the rest of the damage in the real export comes from the GPTQ stage
  (not emulated: needs the real `H`). So RTN counts above understate the real export's damaged-row count.
* `scripts/tile_diag/mlp_composite.py` compares `Dd @ Du` to stock `Wd @ Wu` with no `s_in`. For LM0 the median `s_in` is `2^5.6` (49x), and the
  reported 41.58 is of that order; against the norm-folded stock pair (recovered `s_in`), the real tile export's composite is 1.29.
  The composite of this table uses the norm-folded pair.
* Not done: a calibrated run on the real Pi0 `H` (no Pi0 forward here), so the alpha the new search picks on real activations, and the L9
  drift of 0.1273, are untested. The check that settles it is one Pi0 export with `producer_aware=true` and `hybrid_drift.py` on L9 / L17.

### TinyLlama layers 0-1, real calibration (`scripts/tile_diag/awq_producer_tinyllama.py`)

8 WikiText-2 train windows of 256 tokens (seed 1234) for AWQ; MLP output relative L2 error on 4 held-out windows (seed 99), RTN casts of
gate/up/down, stock residual as the layer input. Only 2 layers are loaded (layer 0 and 1 depend on nothing else).

| layer | panel, old: alpha / damaged rows / MLP err | panel, new: alpha / damaged rows / MLP err | per-channel control: alpha / MLP err |
|---|---|---|---|
| 0 | 0.70 / 0 / 0.02342 | 0.90 / 0 / 0.02323 | 0.70 / 0.02342 |
| 1 | 0.65 / 0 / 0.03146 | 0.35 / 0 / 0.03159 | 0.65 / 0.03146 |

TinyLlama shows no panel damage at these layers, so this is a no-regression check (errors within 0.4% of each other), not a demonstration. A
full TinyLlama export (about 20 min) was not run.
