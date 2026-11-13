# Pack plan: per-site pack-scale fold plan (reporting only)

Facts only. This implements the exporter side of `docs/algos/pack-range-options.md` (mitigation 4.A, check spec section 6). It
**records** a plan; it changes no weight, no exponent, no FX graph and no exported byte (bit-identical, tested). The compiler's
admission range 122..128 (`tile_native.PACK_ADMISSION`) and the existing `compiler_pack_scale_admission` report, including its
`not_checked_dynamic` status for exports with no static activation exponent, are unchanged.

The opt-in pass that applies the free folds (changes weights, exponents and the policy; exact) is [pack-fold.md](pack-fold.md); this plan stays reporting only.

## What is recorded

A static activation exponent `s` means `real = e4m3(byte) * 2^s`; the pack code is `c = 127 + s`. Multiplying the value by `2^k` at its
producer and packing at `c + k` gives the same bytes (`pack(2^k x, s + k) == pack(x, s)`), so the consumer's dequant / pop exponent
stays `s`. The plan says which `k`, who absorbs it and what it costs.

Where (code: `src/pack_plan.py`):

| file | key | content |
|---|---|---|
| `export/policy.json` (from `quant_export.build_policy`) | `sites[<linear>].pack_plan`, `sites[<attention>].pack_plans.<operand>`, top-level `pack_plan_summary` | the record below; producer class is `unknown` (no graph at export time) except attention p (`attn_p`, by structure) |
| `fx/policy.json` (from `fx_export.capture_fx`) | same keys | the same records with the producer class, placement and other consumers read from the captured graph |
| `fx/graph_inspection.json` | `pack_plan_summary`, `pack_plan` (`{site key: record}`; attention key `site:p`) | same as `fx/policy.json` |

A record (`pack_plan`):

| field | meaning |
|---|---|
| `exp`, `e8m0_code` | static exponent and `127 + exp` |
| `admitted_range`, `in_range` | `[122, 128]` from `tile_native.PACK_ADMISSION`; `122 <= code <= 128` |
| `status` | `in_range`, `fold_planned`, or `infeasible` (`abs(k) > 20`) |
| `fold` | `null` when in range; else `k`, `target_code`, `k_to_unit_127` and the exactness statement. Target is the **nearest in-range code** (its `abs(k)` is never above the `abs(k)` to the unit code 127, which is also reported as `k_to_unit_127`); `pack_plan.fold_for_code(code, target=127)` gives the unit-127 fold |
| `producer` | `class`, `source` (`fx_graph`, `policy_operand`, `none`), `other_consumers` (leaf ops other than `quantize_act`), `node`, `mixed_classes` when one site is packed at several graph nodes with different classes (the first node's class is used) |
| `placement`, `how` | where the `2^k` goes: `norm_gamma`, `norm_gamma_beta`, `weight_exp:up_proj`, `weight_exp:producer_linear`, `pv_rescale`, `softmax_multiplier`, `act_constant`, `vmul`, `host` |
| `cost`, `fold_free` | `free` (an existing constant absorbs `2^k`, no added instruction), `vmul_per_row_block` (attention p), `vmul_per_tile`, `act_constant_rewrite` (algebraic, not run; fallback is `vmul_per_tile`), `host`, `infeasible`, `none_in_range`. `fold_free` is `cost == free` |
| `note` | attention p only: needs a vmul on `recip(sum)` (p is normalized and `recip(sum)` is data) |
| `shared_input` | when several `quantize_act` read one source tensor: group size, codes, `identical` |

Producer classes (derived from the graph by walking back through `to/view/reshape/contiguous/transpose/expand/slice/getitem`, the same
rules as `docs/algos/pack-range-options/fx_pack_census.py`): `norm_affine`, `layer_norm`, `norm_modulated`, `attn_p`, `attn_out_pv`,
`attn_out_sdpa`, `gated_product`, `x_sigmoid_product`, `bare_activation`, `linear_to_linear`, `modulation_add`, `raw_input`,
`other:<op>`, `unknown`. A class that normally folds for free (norm, gated product, PV, linear to linear, activation) falls back to
`vmul_per_tile` when the producer has another consumer (the scaled value would leak). `raw_input` stays `host`.

`pack_plan_summary`: `static_sites`, `sites_outside`, `outside_by_class`, `static_by_class`, `outside_by_cost`, `free_fold_sites`, `vmul_sites`,
`infeasible_sites`, `outside_unknown_producer`, `code_range`, `k_range`, `shared_input_groups` (multi-consumer groups, groups with differing codes,
max spread), `dynamic_sites_not_expressible_as_scale_code`, `pop_exponent_bound` (`sx + sw` against -126..127). With no static
activation the summary has `status: not_checked_dynamic` and null counts, never zeros.

## How a compiler owner consumes it

1. Read `fx/policy.json` `sites[*].pack_plan` / `pack_plans.p` (or `fx/graph_inspection.json` `pack_plan`). Skip records with `in_range: true`.
2. Lower the pack at `fold.target_code` (admitted today) instead of `e8m0_code`; keep the consumer's pop / dequant exponent at `exp`.
3. Apply the `2^fold.k` where `placement` says:
   - `free` classes: fold it into the existing constant (gamma, `up_proj` `weight_scale_exp` rows, the PV rescale, the producing Linear's `weight_scale_exp`);
     no instruction is added. This edits device weights, so the baked form needs a `pack_shift` attribute on `quantize_act` (spec item 10, not done).
   - `vmul_per_row_block` (attention p): one `vmul` of the broadcast `recip(sum)` by `2^k` per (head, 32-row block), reused by every p tile of the block.
   - `vmul_per_tile`: insert one `vmul.bf16` by `2^k` before the pack (compiler-side fallback, research 4.F). It is also correct for every other class.
   - `act_constant_rewrite`: rewrite the activation's internal constants (not run); otherwise use `vmul_per_tile`.
   - `host`: scale the raw input before the f32 to BF16 conversion.
4. `producer.other_consumers` non-empty means the producer cannot be scaled in place; the record already says `vmul`.
5. `shared_input.identical == false`: the same tensor is packed at two codes; each pack folds separately.
6. `verify_bundle.py` prints the same report (recomputed from `policy.json` and `fx/graph.txt`) as `pack_plan` in its JSON, informational; add
   `--require-pack-range` to fail (`pack_range_plan_present`) when a static site is outside 122..128 and has no usable recorded plan (record absent or infeasible).
   Bundles exported before this change carry no records and fail that flag by design.

`scripts/tile_flow/pack_plan_census.py` recomputes the plan from existing bundles, read-only, and writes a table (`docs/algos/pack-plan-census.md`).

## Spec items (`pack-range-options.md` section 6)

| # | item | status |
|---|---|---|
| 1 | `code = 127 + exp`, admission constants from one named constant | implemented (`tile_native.PACK_ADMISSION`, `E8M0_BIAS`) |
| 2 | `in_range`, `k`, hard-fail on `abs(k) > 20` or BF16 overflow of `amax * 2^k` | `in_range`, `k`, `infeasible` status at `abs(k) > 20` implemented; it fails only through `--require-pack-range`; the overflow check needs activation data: left |
| 3 | producer class from the graph, `placement` | implemented (FX graph; `unknown` without a graph) |
| 4 | `other_consumers` | implemented; non-empty switches the free placement to `vmul` |
| 5 | shared-input groups with differing codes | implemented (`shared_input`, summary `shared_input_groups`); the per-pack distinct-code list is the group's `codes` |
| 6 | dynamic sites as `not_expressible_as_scale_code` | listed in the summary; the real-text fraction outside from a calibration pass: left |
| 7 | differential fold test on each site's real input | helper `pack_plan.fold_exactness(x, s, k)` implemented and unit-tested on synthetic data; not wired to the exports (needs hooked activations): left |
| 8 | clamp report (flush fraction, saturated count) | left |
| 9 | weights informational, pop exponent `sx + sw` in -126..127 | `pop_exponent_bound` in the summary (from the recorded weight code range); the 0x7F / 0xFF failure already exists in `verify_export`; no weight admission check |
| 10 | `pack_shift` attribute in the FX op | left (not needed while the plan is only recorded) |

## Census from the existing exports

Recomputed read-only from `export/policy.json` and `fx/graph.txt` of the four `atlas_v1_awq_tile_static` bundles
(`scripts/tile_flow/pack_plan_census.py`); the script compares static sites, sites outside, attention p, free-fold sites, and the outside count of every producer class in the research table (section 3 and 4.A; including raw input, sin and index); no mismatch.

| | tinyllama | smolvla | multitask-dit | pi0 |
|---|---|---|---|---|
| static sites | 176 | 316 | 203 | 483 |
| outside 122..128 | 114 | 244 | 88 | 127 |
| free fold (no added instruction) | 92 | 221 | 24 | 61 |
| vmul per p row block | 22 | 13 | 24 | 63 |
| vmul per tile | 0 | 2 | 9 | 0 |
| act-constant rewrite (unverified) | 0 | 6 | 30 | 1 |
| host | 0 | 2 | 1 | 2 |
| unknown producer | 0 | 0 | 0 | 0 |
| code range | 117..126 | 117..125 | 119..126 | 117..133 |
| k range (nearest in-range) | 1..5 | 1..5 | 1..3 | -5..5 |
| shared-input groups differing (of multi) | 0 of 44 | 2 of 76 | 0 of 24 | 0 of 99 |

Outside-range sites by producer class:

| producer class | tinyllama | smolvla | multitask-dit | pi0 |
|---|---|---|---|---|
| attn_out_pv | 22 | 44 | 24 | 27 |
| attn_out_sdpa | - | - | 6 | - |
| attn_p | 22 | 13 | 24 | 63 |
| bare_activation | - | 6 | 13 | 1 |
| gated_product | 11 | 24 | - | 7 |
| layer_norm | - | - | - | 3 |
| linear_to_linear | - | 9 | - | - |
| modulation_add | - | 1 | 1 | - |
| norm_affine | 59 | 145 | - | 24 |
| other:aten.index.Tensor | - | - | 1 | - |
| other:aten.sin.default | - | - | 1 | - |
| raw_input | - | 2 | 1 | 2 |
| x_sigmoid_product | - | - | 17 | - |

Mismatches against the research census: none


Not measured here: any hardware or RTL run of a fold, the quality of a fold (it is bit-exact by construction for the pack; the activation-constant
rewrites are algebraic and were not run), and the 10 spec items marked left.
