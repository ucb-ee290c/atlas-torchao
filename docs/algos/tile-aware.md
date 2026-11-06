# Tile-aware FP8 scales: `w=panel`, `a=block`, `attn=block` (NEW)

Everything here is **NEW**: no source defines these scale layouts for Atlas. The numerics of each cast stay
those of the reference author's encoder (`src/ref_numerics.py`); only the choice of exponent changes.
No quality result exists yet for the three recipes below (no full-model run has been made).

## What and why

The Atlas matrix unit (MXU, matrix multiply unit) multiplies **unscaled** E4M3 operands and accumulates in
BF16 across all K tiles (32 wide). A power-of-two scale (one E8M0 exponent per 32 x 32 tile) is applied only at
three instructions: `vmatpop.fp8` (multiply the popped output tile), `vpack` (divide on the way into FP8) and
`vunpack` (multiply on the way out). The consequence, used throughout this page:

> Only exponents **constant along K** are hardware-native.

| Operand | Hardware-native | Not hardware-native |
|---|---|---|
| Weight `[out, K]` | one exponent per group of 32 output channels over all K (`w=panel`), or one per tensor | per output channel (`w=channel`), per 32-element block along K (`w=block32`) |
| Activation `[M, K]` | one exponent per group of 32 tokens over all K (`a=block`), or one static per tensor | per token (`a=token`), per 32-element block along K (`a=block32`) |
| Attention | per head for q, k, v and static `-8` for p (already constant along the reduction); q and p optionally per 32 rows (`attn=block`) | per row for q, p (`attn=token`) |

An output tile pops with the combined scale `2^(sx[block] + sw[panel])`. It is free when the pop feeds an FP8
consumer (folded into `vmatpop.fp8`); a BF16 pop has no scale slot, so it needs one `vmul.bf16` by a tile
constant. Per-K-block scales (the repository's `block32`) cannot be realized (hardware ruling **H3**, below)
and are labeled so everywhere.

Terminology. "Panel" in this repository's option name `w=panel` means a **32-channel group (option name
`panel`; the atlas-kernels "panel" means a launch of up to 2048 columns sharing one exponent)**. The two are
unrelated: the option never groups more than `rows` (default 32) channels. The option name is not changed.

## Exact semantics

Both options live in `granularity` (`src/algos/granularity.py`); `rows` (default 32) is shared by `w=panel`,
`a=block` and `attn=block`; `rule=ref|nc` is the existing scale rule.

- **`w=panel`.** For a weight `[out, K]`, channels `32g .. 32g+31` form group `g`. The group's exponent is
  `rule(amax over all K and all 32 channels)` (`ref`: `floor(log2(amax)) - 8`; `nc`: the never-clip rule
  `ceil(log2(amax / 448))`). It is returned as `Scale.exp` of shape `[out, 1]` holding the group value repeated
  for every member channel, so the export schema, `output_scale` and `atlas::qlinear` need no shape change. The
  group exponent is `>=` each member channel's own exponent. A final partial group (`out % 32 != 0`) forms its
  own group, and the label is `per_panel(32,partial_tail)` instead of `per_panel(32)`. Conv2d weights are
  flattened to 2-D first, as everywhere else.
- **`a=block`** (activation, from `origin/hw-aware`, extended). One **never-clip (`nc`) exponent per `rows`
  consecutive tokens over the full K**, for Linear inputs; the rule is `nc` regardless of `rule=` (recorded as
  `departs_from_reference_rule`). A Conv2d input under `a=block` keeps the per-tensor rule of `a=tensor`. The
  exponent is returned per token `[..., 1]` (the group value repeated), so the cast and the output rescale are
  those of `a=token`. Stats key `linear:per_token_block`.
- **Token grouping rule.** A tensor `[B, T, K]` is the 2-D matmul operand `[M = B*T, K]` a compiler would tile in
  32-row tiles. Tokens are grouped along the **flattened row-major (batch-major) axis**: tokens `0..31`,
  `32..63`, ... A group may therefore straddle two batch elements when `T % 32 != 0`. The last group is
  partial when `M % 32 != 0`; decode (`M = 1`) is one group of one real row. Padding rows are **zeros**: they
  add nothing to a group's amax (so no padding is materialized in the emulation) and encode to the `+0` code
  `0x00` at any exponent (`tile_native.padding_encodes_to_plus_zero` checks this on the encoder).
- **`attn=block`.** q and p of each attention site get one `nc` exponent per `rows` rows of each
  `(batch, head)` matrix `[S, D]` (groups along `S`, constant along the reduction `D` of Q.K and `Skv` of P.V);
  k and v keep one exponent per `(batch, head)`. Shape `[B, H, S, 1]` for q and p. Default `attn=head` is
  unchanged. Label `per_row_block`.
- **`defer`** (`a=defer`, `attn=defer`): `granularity` gives no activation Scale and leaves it to a stacked member
  (`static_calib`) or the reference rule. This is what lets `static_calib` coexist with `w=panel`
  (without it the stack raises `StackConflict`). Default off.
- **`block32`** (along K) is kept for the screening runs and is **NOT hardware-realizable (H3)**. Its docstring,
  the export (`tile_native: false`, reason "ALONG K ... H3") and the FX inspection all say so.

## The three recipes

Immutable names (`src/algos/__init__.py`, pinned in `tests/test_algo_atlas_v1.py`/`tests/test_algo_tile.py`). All use the
`atlas_v1` chip defaults (`--chip-vadd rn --chip-vadd-sites residual`).

| Recipe | Expands to | Activation scales |
|---|---|---|
| `atlas_v1_awq_tile` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=panel,a=block,rows=32,rule=nc,attn=head` | dynamic per 32-token block over the full K (run-time amax per block); attention per `(batch, head)`, `nc` |
| `atlas_v1_tile` | the same + `keep_bf16:rule=input_conv` | the same; every `Conv2d` with 3 input channels stays BF16 |
| `atlas_v1_awq_tile_static` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc` + `static_calib` | static per tensor for every calibrated Linear / Conv2d input (no run-time amax; avoids **H2**); attention p static `-8`, q, k, v per `(batch, head)` (reference dynamic rule) |

`atlas_v1_awq_tile` mirrors `atlas_v1_awq` (no BF16 exception). `atlas_v1_tile` adds the input-convolution
exception through the v1.1 member `keep_bf16` (the DiT patch embedding that v1.0 `atlas_v1` kept in BF16, keyed
there on the policy type `multi_task_dit`). `keep_bf16:rule=input_conv` is structural: it also selects the
SigLIP patch embedding of Pi0 / SmolVLA (see [keep-bf16.md](keep-bf16.md)). This is the one recipe that carries a BF16
exception, an exception to the v1.1 "no recipe keeps a layer in BF16" rule, noted in `algos/__init__.py`.

### GPTQ and AWQ on the panel grid

`gptq_awq` is stacked before `granularity` and is delegated its weight grid (stacking.md): `weight_grid` is
`granularity.weight_scale`, whose answer for `w=panel` is the `[out, 1]` panel table. (1) AWQ's search quantizes
`W * diag(s)` with the exponent re-derived by that same function (`exp_fn`), so its loss is evaluated on the
panel grid, not a per-tensor grid. (2) GPTQ pins the panel exponents computed on the weight it receives, passes
`s.reshape(-1)` (one per row) to `qdq`, and `weight_scale` hands the same pinned table back, so the final codes
and exponents are the ones GPTQ optimized for. (3) The `nc` no-clip guarantee holds for the weight GPTQ receives;
GPTQ's error feedback can push a value past `448 * 2^s` afterwards, where the cast saturates (as for `w=channel`).
Tested in `tests/test_algo_tile.py` (spy on `awq_search` / `gptq`, constancy after GPTQ). The opt-in
`awq_loss=cast` path of `gptq_awq` evaluates its activations with a per-token cast; with `a=block` that loss is
an approximation of the run-time cast (not used by the tile recipes).

## Contracts and verifier (`src/tile_native.py`)

- `check_constant_groups(exp, rows, layout)` returns a structured `GroupCheck` (ok, groups, partial last
  group, violating groups with their values). Layouts: `flat` (weights `[out, 1]`, tokens `[..., T, 1]`
  flattened) and `per_matrix` (`[..., S, 1]`, attention).
- `quant_export` writes into `policy.json`: per site `weight.granularity` (`per_panel(32)`, `per_channel`,
  `per_block(32)`, `per_tensor`), `weight.panel_rows`, `weight.code_report` and `weight.e8m0_code_range`,
  `activation.tile_label` (`per_block(32 rows)`), `tile_native` {tile_native, reasons, checks}; and top-level
  `tile_native` (summary and the sites that are not tile-native, with reasons) and
  `compiler_pack_scale_admission`.
- `load_quantized` fails loudly (`TileContractError`) if a `per_panel`-labeled weight varies within a group, or
  any weight carries `0x7F` / `0xFF`.
- `fx_export` adds to `graph_inspection.json`: `tile_native` (per site, `native_fp8`, the not-tile-native list
  and reasons, `fixture_activation_stats` from the capture cases) and `compiler_pack_scale_admission`.
- **Informational checks, nothing is clamped or changed.** Every static activation exponent and every weight
  exponent is recorded as its E8M0 code `127 + s`; activation sites outside `122..128` (the compiler's `lowerPack`
  admission today) are listed. Static attention `p = -8` is code 119, so every attention site of
  `atlas_v1_awq_tile_static` is listed. Dynamic exponents are data dependent: only the capture-case fixtures
  are reported. The flushed (exponent-field-0) code fraction is reported per group of 32 channels for weights and, for
  block-scaled activation sites, per group of 32 rows (min / max / mean over groups and calls; other activation sites
  report a mean per site); only `0x7F` / `0xFF` in weight codes fails. Dynamic block sites report the token
  split (32-aligned or the partial last block) and the padding check.

## FX ops

`atlas::quantize_act(x, mode, reduce, rule, static_exp, code_dtype, tag="", rows=32)`: `reduce="block"`
(exponent per `rows` flattened rows, shape `[..., 1]` with repeated values) and `reduce="head_block"` (attention
q, p: per `rows` rows of each matrix, `[..., S, 1]`) are NEW; `rows` is the last argument with a default, and
`fx_export` passes it only for the block reductions, so every other node keeps its old argument list. `qlinear`,
`qconv2d`, `qk_matmul` and `pv_matmul` are unchanged. A bundle saved before `rows` existed still imports and
replays (`tests/test_algo_tile.py::test_bundle_saved_with_the_old_quantize_act_schema_still_verifies`, which saves it in a
subprocess with a frozen copy of the old operator module, `tests/old_schema/`).

The FX emulation decodes both operands with their exponents and then multiplies. For exponents constant along K
this is **bit-identical** to the pop-scale dataflow (matmul of the unscaled E4M3 values, one multiply by
`2^(sx[block] + sw[panel])`) **in float32 accumulation, while no value overflows or underflows**: a power-of-two scale
commutes with exact float32 products and with the accumulation order. The hardware accumulates in BF16; that
path is not tested here, so the statement is about the dataflow, not about the BF16 rounding. Tested, together
with the negative case (a scale varying along K is not reproducible by any single pop scale).

## Costs (`chip_profile.count_tile_output_rescale`)

Under `w=panel` with `a=block`, static or per-tensor activations, the per-element pre-`vpack` `vmul`
(`granularity_rescale`) is not counted, and neither is the per-element output vmul. The output rescale is
counted as folded into `vmatpop.fp8` when the consumer is FP8 (0 instructions), otherwise as one `vmul.bf16` by
a tile constant per 32 x 32 output tile (VPU row `tile_output_rescale`). **Assumption:** the emulation cannot
see the consumer of each pop, so `TILE_OUTPUT_CONSUMER = "bf16"` (conservative) is counted unless a caller
sets `"fp8"`. The per-token / per-channel accounting (`granularity_rescale`) is unchanged and applies to
`a=token`. The `a=block` dynamic chain (`dyn_scale_chain`) is unchanged.

`hw_cost.rescale_plan` (`docs/algos/hw-report.md`) classifies each linear's output consumer from the traced layer
graph and says a pop scale is free only when the consumer is an FP8 matmul operand with a static, K-constant scale
and the weight scale is uniform per 32-channel group; `chip_profile.TILE_OUTPUT_CONSUMER = "bf16"` never grants
the free pop, which agrees with that classification on the conservative side but does not use it per layer.

Arithmetic note for the admission check (a question, not a decision): under the reference rule an exponent is
`floor(log2(amax)) - 8`, so the E8M0 code is `119 + floor(log2(amax))`; the range `122..128` then admits only
`amax` in `[8, 1024)` (the `nc` rule shifts a bound by at most one), and a static attention `p = -8` is code 119,
always outside it. Most activation sites of the tiny test models fall at codes 114 to 121.

## Hardware rulings involved (not decided here)

- **H1**: is a per-row / per-column output vmul the intended rescale path, or is a scale applied in the pop path?
  The panel / block layout needs only the per-tile pop scale of `vmatpop.fp8` plus, for BF16 pops, one tile-constant
  `vmul`.
- **H2**: deriving the exponent of a run-time amax. `granularity.md` records it as closed in software
  (`src/hw_scale.py`, RV32I scalar sequence), while `PENDING_RULINGS` in `src/quant_export.py` still lists it as
  open; the two are not reconciled here. The static recipe needs no run-time amax at all.
- **H3**: per-K-block scales inside the reduction. Tile-aware layouts are chosen so that none is needed;
  `block32` along K stays labeled not realizable.

## Commands

```bash
# export a tile recipe (see export.md for the full flags; the eval.py line needs the model and its
# calibration data, the fx_export line was run on a tiny random-weight export on CPU)
python src/eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --algo atlas_v1_awq_tile --modes ref_conv \
  --export-dir /new/dir --export-only
python src/fx_export.py --export-dir /new/dir --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --out-dir /new/fx \
  --device cpu --seqlen 8 --native-fp8
# tests
python -m pytest tests/test_algo_tile.py tests/test_algo_granularity.py tests/test_algo_atlas_v1.py -q
```

## Limits

- No quality measurement exists (no full-model run); the recipes are mechanically tested on tiny models only.
- `rows` other than 32 are emulated but do not match the 32 x 32 hardware tile.
- `a=block` forces `nc` and applies to Linear inputs only; Conv2d inputs get the per-tensor rule.
- On chip under `reference_hw` / `chip`, panel / block scales run through the existing MXU model with the
  exponent table applied after the matmul, as for `w=channel`; the pop-scale dataflow is equivalent (tested) but
  the BF16 accumulation order is the model's, not the RTL's.
- The consumer of each pop (FP8 or BF16) is an assumption in the cost accounting (above).
- Attention under `reference_hw` / `chip` accumulation models use per-row / per-block exponent tables like
  `attn=token`; no RTL check was made.
