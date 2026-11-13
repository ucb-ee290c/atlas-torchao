# Tile-aware FP8 quantization

This page covers the tile-aware FP8 flow for Atlas: the scale layouts and why they are needed, the recipes, how to take a
model through AWQ / GPTQ calibration to a native FP8 FX graph, the opt-in options added with it, what is handed to the
compiler, and the measured results. Related pages: [export.md](export.md) (export files), [fx-export.md](fx-export.md) (FX
graph), [granularity.md](granularity.md), [stacking.md](stacking.md), [gptq-awq.md](gptq-awq.md), [keep-bf16.md](keep-bf16.md).

Status as of 2026-11-06. Unless a section says otherwise (simulator and bring-up harness results), numbers are from the
framework's software emulation on CPU, with exact float32 accumulation unless an MXU model (`sa`, `ipt`) is named. No RTL
run was made by this flow, and the compilers team has not reviewed or
accepted the bundles. Bundles, exports and raw result files lived on the build host and are not part of the repository.
A result that a later measurement replaced is marked as superseded ([Superseded results](#superseded-results)).

## Terms

| Term | Meaning |
|---|---|
| K, amax | K is the reduction (input) axis of a matmul: a weight is `[out, K]`, an activation `[tokens, K]`. amax is the maximum absolute value over the elements an exponent covers. |
| q, k, v, p | The attention operands: queries, keys, values, and the softmax probabilities `p` that multiply v (the two attention matmuls are Q.K and P.V). SDPA is PyTorch's fused `scaled_dot_product_attention`. |
| MXU | Matrix unit. Multiplies unscaled E4M3 operands and accumulates in BF16 across K tiles of 32. Two variants are modelled: MXU0, a systolic array (`sa`), and MXU1, an inner-product tree (`ipt`). |
| VPU | Vector unit: element-wise BF16 work (`vmul`, `vadd`, `vpack`, `vunpack`). |
| DMA, VMEM, HBM | DMA moves data between off-chip memory (HBM) and the on-chip vector memory (VMEM, 1.5 MiB). |
| kit | The hand-written Atlas kernels (`atlas-kernels`), used as the reference for how the hardware is driven today. |
| E4M3 | The 8-bit floating-point format of weights and activations. Maximum 448 (RTL; one hardware spec says 240). Codes `0x7F` / `0xFF` are reserved. |
| E8M0 | An 8-bit power-of-two scale code. An exponent `s` is stored as code `127 + s`. |
| pack, pop, unpack | The only three places the hardware applies a scale: `vpack` divides a BF16 value by `2^s` on its way into FP8, `vmatpop.fp8` multiplies a popped 32 x 32 output tile, `vunpack` multiplies on the way out of FP8. `vmatpop.bf16` has no scale operand. |
| AWQ | Activation-aware weight quantization (Lin et al. 2024): per-input-channel scales `s` folded into the producer (a norm or `up_proj` rows) and into the consumer's columns. |
| GPTQ | Second-order weight rounding with error feedback (Frantar et al. 2023). `gptq_awq:method=awq_gptq` runs AWQ then GPTQ; `method=gptq` runs GPTQ only. |
| panel (`w=panel`) | One weight exponent per group of 32 output channels over all of K. Unrelated to the atlas-kernels "panel" (a launch of up to 2048 columns sharing one exponent); the option name is kept. |
| block (`a=block`, `attn=block`) | One activation exponent per 32 consecutive token rows over all of K. |
| static, dynamic | A static exponent is calibrated offline and stored. A dynamic exponent is derived at run time from the amax of the data. |
| `ref`, `nc` rule | Exponent rules: `ref` is `floor(log2(amax)) - 8`; `nc` (never clip) is `ceil(log2(amax / 448))`. |
| site | One quantized Linear, Conv2d or attention module in `policy.json`. |
| bundle | An `export/` directory (weights, `policy.json`, `manifest.json`) plus the `fx/` directory of its captured graph. |
| boundary | The part of a model that is captured as one graph (for example the full causal-LM forward, or one sampler step). |
| strict, non-strict | `torch.export` modes: strict traces with TorchDynamo; non-strict runs the Python code and records the ops, which needs fewer code changes but can keep unused buffers. |

## The tile rule

The MXU multiplies unscaled E4M3 operands and accumulates in BF16 across all K tiles (32 wide). A power-of-two scale (one
E8M0 exponent per 32 x 32 tile) can be applied only at pack, pop or unpack. Consequently:

> Only exponents that are **constant along K** are hardware-native.

| Operand | Hardware-native | Not hardware-native |
|---|---|---|
| Weight `[out, K]` | one exponent per 32 output channels over all K (`w=panel`), or one per tensor | per output channel (`w=channel`), per 32-element block along K (`w=block32`) |
| Activation `[M, K]` | one exponent per 32 tokens over all K (`a=block`), or one static exponent per tensor | per token (`a=token`), per 32-element block along K (`a=block32`) |
| Attention | per head for q, k, v and static `-8` for p (already constant along the reduction); q and p optionally per 32 rows (`attn=block`) | per row for q and p (`attn=token`) |

An output tile pops with the combined scale `2^(sx[block] + sw[panel])`. This is free when the pop feeds an FP8 consumer
(folded into `vmatpop.fp8`); a BF16 pop needs one `vmul.bf16` by a tile constant. The numerics of each cast are those of the
reference encoder (`src/reference/ref_numerics.py`); only the choice of exponent changes.

### Exact semantics

All options live in `granularity` (`src/algos/granularity.py`). `rows` (default 32) is shared by `w=panel`, `a=block` and
`attn=block`; `rule=ref|nc` is the existing scale rule.

- **`w=panel`.** Channels `32g .. 32g+31` form group `g`, whose exponent is `rule(amax over all K and the 32 channels)`. It is
  stored as `weight_scale_exp` of shape `[out, 1]` with the group value repeated per channel, so the export schema and
  `atlas::qlinear` are unchanged. A final partial group (`out % 32 != 0`) gets the label `per_panel(32,partial_tail)`.
  Conv2d weights are flattened to 2-D first.
- **`a=block`.** One `nc` exponent per `rows` consecutive tokens over the full K, for Linear inputs; the rule is `nc`
  regardless of `rule=`. A Conv2d input keeps the per-tensor rule. The exponent is returned per token (group value repeated).
- **Token grouping.** A tensor `[B, T, K]` is the matmul operand `[M = B*T, K]`. Tokens are grouped along the flattened
  row-major (batch-major) axis, so a group may straddle two batch elements when `T % 32 != 0`. The last group is partial when
  `M % 32 != 0`; decode (`M = 1`) is one group of one row. Padding rows are zeros: they add nothing to the amax and encode to
  the `+0` code `0x00` at any exponent (`tile_native.padding_encodes_to_plus_zero`).
- **`attn=block`.** q and p get one `nc` exponent per `rows` rows of each `(batch, head)` matrix; k and v keep one exponent per
  `(batch, head)`. The default is `attn=head`. No named recipe uses `attn=block`.
- **`defer`** (`a=defer`, `attn=defer`): `granularity` sets no activation scale and leaves it to a stacked member
  (`static_calib`) or the reference rule. This is what lets `static_calib` stack with `w=panel`.
- **`block32`** (along K) is kept for screening and is labelled **not hardware-realizable** (ruling H3) in its docstring, the
  export (`tile_native: false`) and the FX inspection.

### Hardware rulings

These are not decided by this flow:

- **H1**: is a per-row or per-column output `vmul` the intended rescale path, or is the scale applied in the pop path? The
  panel / block layout needs only the per-tile pop scale of `vmatpop.fp8`, plus one tile-constant `vmul` for BF16 pops.
- **H2**: deriving an exponent from a run-time amax. [granularity.md](granularity.md) records it as closed in software
  (`src/chip/hw_scale.py`), while `PENDING_RULINGS` in `src/exporter/quant_export.py` still lists it as open; the two are not
  reconciled. The static recipe needs no run-time amax.
- **H3**: per-K-block scales inside the reduction. The tile layouts need none; `block32` stays labelled not realizable.

### Contracts and checks in the code

- `src/exporter/tile_native.py` checks group constancy (`check_constant_groups`). `src/exporter/quant_export.py` writes per
  site `weight.granularity` (`per_panel(32)`, `per_channel`, `per_block(32)`, `per_tensor`), `code_report`, `e8m0_code_range`,
  `activation.tile_label` and `tile_native`, plus top-level `tile_native` and `compiler_pack_scale_admission`.
  `src/exporter/load_quantized.py` raises `TileContractError` if a `per_panel` weight varies within a group or any weight
  carries `0x7F` / `0xFF`. `src/exporter/fx_export.py` adds the same records to `graph_inspection.json`. Nothing is clamped.
- FX op `atlas::quantize_act(x, mode, reduce, rule, static_exp, code_dtype, tag="", rows=32)` gains `reduce="block"` and
  `"head_block"`; `rows` is passed only for those, so older bundles still load (`tests/old_schema/`).
- For exponents constant along K, decoding both operands then multiplying is bit-identical to the pop-scale dataflow in
  float32 accumulation while nothing overflows or underflows (tested, with a K-varying negative case). BF16 accumulation is
  not covered by that statement.
- Cost accounting (`src/chip/chip_profile.py`): the output rescale counts as free when the consumer is FP8, otherwise one
  `vmul.bf16` per 32 x 32 output tile. The emulation cannot see each pop's consumer, so `TILE_OUTPUT_CONSUMER = "bf16"`
  (conservative) is assumed unless a caller sets `"fp8"` (per-layer classification: [Hardware cost report](#hardware-cost-report)).

## Recipes

Recipe names are immutable (`src/algos/__init__.py`, pinned in `tests/test_algo_atlas_v1.py` and `tests/test_algo_tile.py`).
All use the `atlas_v1` chip defaults (`--chip-vadd rn --chip-vadd-sites residual`).

| Recipe | Expands to | Activations | Attention | Use when |
|---|---|---|---|---|
| `atlas_v1_awq_tile_static` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc` + `static_calib` | one static exponent per tensor, no run-time amax | p static `-8`; q, k, v dynamic per (batch, head), reference rule | Primary. A compiler-team report (not verified here) says every exponent must be a compile-time constant. |
| `atlas_v1_awq_tile` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=panel,a=block,rows=32,rule=nc,attn=head` | dynamic per 32-token block over all K (`nc`) | q, k, v, p dynamic per (batch, head), `nc` | Scales must follow the data and a run-time amax is acceptable (H2 open). |
| `atlas_v1_tile` | `atlas_v1_awq_tile` + `keep_bf16:rule=input_conv` | as `atlas_v1_awq_tile` | as `atlas_v1_awq_tile` | Models with an input convolution: every `Conv2d` with 3 input channels stays BF16. |
| `atlas_v1_awq` | `gptq_awq:method=awq_gptq,actorder=true` + `granularity:w=channel,a=token,rule=nc` | dynamic per token | per head | Not tile-native (needs per-row vector rescales); the comparison baseline. |

Pi0 candidate (no named recipe; AWQ off and `dead_columns=rtn`, see [Pi0 results](#pi0-awq-and-panel-damage)):

```
gptq_awq:method=gptq,actorder=true,dead_columns=rtn+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib+keep_bf16:rule=input_conv
```

- `atlas_v1_awq_tile_static` has no `keep_bf16` member: Multi-Task DiT and Pi0 were exported with
  `atlas_v1_awq_tile_static+keep_bf16:rule=input_conv`, SmolVLA without it (its patch embedding is quantized, 1 `qconv2d`).
- `atlas_v1` is now an alias of `atlas_v1_awq` with no DiT BF16 exception (a meaning change from the `hw-aware` merge).
- The two TinyLlama tile exports have byte-identical weights; the static one adds 154 `<fqn>.input_scale_exp` tensors.

### GPTQ and AWQ on the panel grid

`gptq_awq` is stacked before `granularity` and delegated its weight grid ([stacking.md](stacking.md)). (1) AWQ's search
quantizes `W * diag(s)` with the exponent re-derived by `granularity.weight_scale`, so its loss is evaluated on the panel grid.
(2) GPTQ pins the panel exponents of the weight it receives, and `weight_scale` hands the same table back, so the final codes
and exponents are the ones GPTQ optimized for. (3) The `nc` no-clip guarantee holds for the weight GPTQ receives; error
feedback can push a value past `448 * 2^s`, where the cast saturates (spy test in `tests/test_algo_tile.py`). AWQ's search
scores only the consumer; see [Producer-aware AWQ](#producer-aware-awq) for the consequence.

## From a model to a native FP8 FX graph

The commands below were run on CPU (TinyLlama-1.1B-Chat, torch 2.11.0 CPU, transformers 5.5.4, torchao 0.18.0; Pi0 used a
separate environment with torch 2.7.1). Wall times are from a shared 16-core host and are only indicative.

### Setup

```bash
export PYTHONPATH=src
export CAP="taskset -a -c 0-15 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"   # optional, caps a shared host
export MODEL=/path/to/TinyLlama-1.1B-Chat-v1.0          # any Hugging Face Llama-family directory
export OUT=/path/to/exports/tinyllama/atlas_v1_awq_tile_static   # one directory per model and recipe
git status --short        # must be empty: the export and the capture record the commit and a dirty flag
sha256sum $MODEL/model.safetensors   # TinyLlama: 6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933
```

### Smoke test (about 1 minute)

A 2-layer random Llama with spread row scales, so that different 32-channel groups get different exponents (with an
unspread init every exponent is 115 and the "not collapsed to per-tensor" check fails on purpose):

```bash
export TINY=/tmp/tiny-llama TOUT=/path/to/exports/smoke-tiny-llama/atlas_v1_awq_tile_static
$CAP python src/scripts/tile_flow/make_tiny_llama.py $TINY $MODEL
$CAP python src/eval.py --model $TINY --seqlen 128 --calib-n 4 --modes ref_conv --export-only \
    --algo atlas_v1_awq_tile_static --device cpu --export-dir $TOUT/export --out $TOUT/export_cli.json   # 19 s
$CAP python -m exporter.fx_export --export-dir $TOUT/export --model $TINY --out-dir $TOUT/fx \
    --device cpu --seqlen 64 --native-fp8                                                                 # 22 s
$CAP python src/scripts/tile_flow/verify_bundle.py $TOUT/export $TOUT/fx $TOUT/verify_bundle.json
```

### Calibrate and export

```bash
# causal LM; --calib-n and --calib-seed default to 16 and 1234
$CAP python -u src/eval.py --model $MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --device cpu \
    --export-dir $OUT/export --out $OUT/../atlas_v1_awq_tile_static_export_cli.json
```

Measured: `atlas_v1_awq_tile_static` 1542 s, `atlas_v1_awq_tile` 1146 s. The export directory must not exist. Other model
types use their own CLI with the same `--algo` / `--export-only` / `--export-dir` flags: `src/eval_policy.py` (LeRobot
policies, needs `--dataset` and `--norm-stats dataset`) and `src/eval_pi0.py`. With `--calib-n N`, state N next to any result.

### Capture the FX graph

```bash
$CAP python -m exporter.fx_export --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx --device cpu --seqlen 64 --native-fp8
```

326 s for TinyLlama. `--seqlen` is fixed in the graph (batch 1); use a multiple of 32 so every 32-token block of the dynamic
recipe is full. `--model` supplies the architecture config only. The capture already replays the saved graph in a fresh
process (`fresh_process_verification.json`); `python -m exporter.fx_export --verify-dir <fx dir> --device cpu` reruns that
check and rewrites the file. Load `model.pt2` only after importing `exporter.atlas_fx_ops`.

### Verify

```bash
$CAP python src/scripts/tile_flow/verify_bundle.py $OUT/export $OUT/fx $OUT/verify_bundle.json   # 25 checks, exit 0 = all pass
#   --allow-uncalibrated-fallback   accept undeclared uncalibrated Linear sites of a static recipe
#   --forbid-unquantized-attention  plain BF16 attention outside policy sites becomes a failure
#   --require-pack-range            fail when a static site is outside 122..128 with no usable pack plan
# tiny model only: re-calibrate in memory and compare codes, exponents and logits with the bundle
$CAP python src/scripts/tile_flow/inmem_check.py $TINY atlas_v1_awq_tile_static $TOUT/export $TOUT/fx 4 128 64 $TOUT/inmem_check.json
python src/scripts/tile_flow/make_bundle_manifest.py $OUT                                         # writes bundle_manifest.json
```

`verify_bundle.py` reads `model.safetensors`, `policy.json` and `model.pt2` directly. What is checked:

| # | Check | Enforced by |
|---|---|---|
| 1 | `model.safetensors` and `policy.json` sha256 equal `manifest.json` | `load_quantized` (raises) |
| 2 | every quantized Linear has uint8 `weight_codes` and int32 `[out, 1]` `weight_scale_exp`; no float weight stored | `verify_bundle.py` |
| 3 | no `0x7F` / `0xFF` weight code | `verify_bundle.py`; `load_quantized` raises `TileContractError` |
| 4 | `weight_scale_exp` constant within every 32-channel group and not collapsed to one value per tensor | `verify_bundle.py` (own implementation) |
| 5 | `tile_native.all_tile_native` is true and nothing is still emulated | `verify_bundle.py` |
| 6 | one `qlinear` / `qconv2d` node per call of each quantized site, and qk / pv pairs per attention site, derived from the graph | `verify_bundle.py` |
| 7 | the only non-atlas contractions are declared ones: BF16-by-policy Linear / Conv2d, the float32 RoPE outer product, plain BF16 attention in blocks with no policy attention site (`graph.unquantized_attention`) | `verify_bundle.py` |
| 8 | weight buffers in the loaded program are uint8 / int32; unused meta placeholders of a non-strict export are tolerated only when unconsumed | `verify_bundle.py` |
| 8b | a static recipe has no undeclared uncalibrated Linear / Conv2d (`static_recipe_fully_static`) | `verify_bundle.py` |
| 9 | eager replay, captured graph and reloaded graph byte-identical on two inputs; fresh-process replay passes | `fx_export` |
| 10 | in-memory model equals the bundle | `inmem_check.py`; for large models `load_quantized` re-derives every code |
| 11 | dynamic block exponents constant within each 32-row group in the capture cases | `graph_inspection.json` `fixture_activation_stats` |

The verifier counts sites from the graph, not from the policy: a site the boundary never calls (SmolVLA `vlm.lm_head`) is
exempt only when it matches the capture's `unused_quantized_modules`, and scope-less attention nodes (a module called as
`module.forward()`) are attributed to the policy site that holds them. Before this change it failed 7 of 25 checks on the
SmolVLA bundle; all 7 were verifier assumptions, not bundle defects.

### Bundle contents

| Location | Files |
|---|---|
| `export/` | `model.safetensors` (`<fqn>.weight_codes` uint8 `[out, K]`, `<fqn>.weight_scale_exp` int32 `[out, 1]`, static recipe: `<fqn>.input_scale_exp`; norms, embeddings and BF16 `lm_head` in their own dtype), `policy.json` (per-site rules, `tile_native`, `compiler_pack_scale_admission`, `pack_plan`), `manifest.json` (sha256 of both, commit and dirty flag, library versions, checkpoint sha256, expanded recipe, calibration provenance) |
| `fx/` | `model.pt2`, `graph.txt`, `graph_signature.txt`, `graph_inspection.json`, `buffer_inventory.json`, `fixtures.safetensors`, `parity.json`, `policy.json` (copy with graph-derived pack plan), `fx_manifest.json`, `verification.json`, `fresh_process_verification.json` |
| bundle root | `verify_bundle.json`, `bundle_manifest.json`; optionally `quality.json`, `ppl_w16x512.json`, `loop_manifest.json` |

### Quality measurement (optional)

```bash
# relative L2 vs BF16 on the first 2 x 256 WikiText-2 test tokens, plus sa / ipt on first / middle / last layers
$CAP python -u src/scripts/tile_flow/quality.py $MODEL 256 2 tile_static=$OUT/export=$OUT/quality.json
# perplexity on the first 16 windows of 512 WikiText-2 test tokens
$CAP python -u src/eval.py --model $MODEL --seqlen 512 --max-windows 16 --device cpu --from-export $OUT/export \
    --modes bf16 ref_conv --out $OUT/ppl_w16x512.json --mxu ipt --crosscheck off
$CAP python src/scripts/tile_flow/dyn_admission.py $MODEL $OUT/export $OUT/dynamic_pack_admission_real_text.json 256 2   # dynamic recipe
$CAP python src/scripts/tile_flow/static_clip.py $MODEL $OUT/export $OUT/static_activation_saturation_real_text.json 256 2  # static recipe
```

### Other models

```bash
# SmolVLA (flow-loss boundary): export 1266 s, capture 403 s
export SV=/path/to/exports/smolvla/atlas_v1_awq_tile_static
$CAP python -u src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --norm-stats dataset \
    --n-frames 2 --seed 0 --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --device cpu --export-dir $SV/export
$CAP python -u -m exporter.fx_export --export-dir $SV/export --out-dir $SV/fx --device cpu --native-fp8 --boundary smolvla_flow_loss --n-images 1
# Multi-Task DiT (export 385 s)
python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune \
    --seed 0 --n-frames 200 --modes ref_conv --export-only --algo "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv" ...
# Pi0 (torch 2.7.1 environment; export 2374 s with AWQ, about 35 min for the candidate recipe)
python -u src/eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --algo "<recipe>" --export-dir export --export-only \
    --device cpu --openpi-dir ... --checkpoint-dir ... --tokenizer-path ...
```

`fx_export --boundary` knows only `causal_lm` and `smolvla_flow_loss`. The single-step DiT and Pi0 graphs were captured by a
driver outside the repository (its sha256 is in each `fx_manifest.json`); the per-phase graphs use drivers in
`src/scripts/tile_flow/` ([Per-phase graphs and loop manifest](#per-phase-graphs-and-loop-manifest)). Pi0 capture is non-strict
under torch 2.7.1.

## Options

All options below are off by default and leave a default export unchanged.

### Pack fold

`--pack-fold` (export CLIs) or the post-hoc pass `python -m exporter.pack_fold` (`src/exporter/pack_fold.py`). The compiler's
`lowerPack` admits activation pack codes 122..128 only ([Pack-scale admission range](#pack-scale-admission-range)). Packing
`2^k * x` at exponent `s + k` gives byte-identical FP8 codes to packing `x` at `s`, so for an out-of-range static site the
pass (1) multiplies the producer of the packed tensor by `2^k` exactly, (2) sets the consumer's static input exponent to
`s + k`, and (3) subtracts `k` from the consumer's `weight_scale_exp` (codes untouched), so the pop constant `2^(sx + sw)` is
unchanged. No calibration runs and no FP8 byte changes. `k` is the smallest shift that puts every site of a producer group into
range (target: the nearest in-range code).

Folded (no added instruction): RMSNorm gamma and LayerNorm weight / bias; Gemma `x * (1 + w)` via `w' = 2^k (1 + w) - 1`
(float32 only, checked exact per element); `act(gate) * up` via `up_proj`'s `weight_scale_exp`; a Linear feeding a Linear; and
P.V into `o_proj` via a new trailing `pop_shift` argument of `atlas.pv_matmul`. Reported but not touched: attention p (one
`vmul` on `recip(sum)` per head and 32-row block), bare GELU / SiLU and `x * sigmoid` (activation-constant rewrite, not run),
SDPA output, modulation add, sin and index (one `vmul` per tile), raw model inputs (host). Never folded: a producer with another consumer, a consumer that is not a single static Linear / Conv2d, an attention site with
more than one `pv_matmul` node, `|k| > 20`, or a failed exactness guard. Refusals are recorded in `policy.json`
(`pack_fold.reason`, `pack_fold_summary`); the export CLI keeps the unfolded export in `<export-dir>.unfolded`.

| Bundle | Static sites | Outside before | Folded | Outside after | End-to-end check |
|---|---|---|---|---|---|
| TinyLlama | 176 | 114 | 92 | 22 (all attention p) | 16,384,000 logits, 0 differing; checker 25 of 25 |
| SmolVLA | 316 | 244 | 181 | 63 | eager and FX outputs bit-identical; checker 25 of 25 |
| Multi-Task DiT (first bundle) | 203 | 88 | 24 | 64 | counts only |
| Pi0 (`atlas_v1_awq_tile_static`) | 483 | 127 | 61 | 66 | counts only |

SmolVLA's shared text attention site has 24 `pv_matmul` nodes, so a per-site `pop_shift` cannot be applied to it (the first
version did, and outputs differed by up to 81.7); such sites are now refused. The nearest in-range target puts 147 of the 176
TinyLlama sites on code 122, the edge of the range; system-level pack tests exist only at codes 126 and 128. DiT and Pi0
folded bundles were not captured.

```bash
python -m exporter.pack_fold --export-dir <E>/export --fx-dir <E>/fx --out-dir <F>/export
python -m exporter.fx_export --export-dir <F>/export --out-dir <F>/fx --native-fp8 --seqlen 64
python src/scripts/tile_flow/verify_bundle.py <F>/export <F>/fx <F>/verify_bundle.json
python src/scripts/tile_flow/compare_fx_logits.py <E>/fx <F>/fx <F>/compare_fx_logits.json --random 6
```

For non-causal-LM models, `--pack-fold` inside an export CLI needs `--pack-fold-graph <fx dir>` from a capture of the unfolded
export. Tests: `tests/test_pack_fold.py`.

### Pack plan

Reporting only; changes no byte (tested). For every static activation site, `policy.json` (`pack_plan`, `pack_plans.p`,
`pack_plan_summary`) records the code `127 + exp`, whether it is in 122..128, the fold `k` to the nearest in-range code, the
producer class, where the `2^k` would go and its `cost` (`free`, `vmul_per_row_block`, `vmul_per_tile`, `act_constant_rewrite`,
`host`, `infeasible`). The producer class is read from the graph, so it is `unknown` in `export/policy.json` and filled in
`fx/policy.json` (`src/exporter/pack_plan.py`; `src/scripts/tile_flow/pack_plan_census.py` recomputes it from bundles). The
plan does not know the fold's guards: `verify_bundle.json` counts `pack_fold.planned_free_but_refused` (SmolVLA: 40).

### Dead columns

`gptq_awq:...,dead_columns=zero|rtn`. By default (`zero`) GPTQ zeroes every weight column whose calibration Hessian diagonal is
zero (`dead = diag(H) == 0`). When calibration never excites an input channel, that channel's weights are lost
([SigLIP fc2 zero columns](#siglip-fc2-zero-columns)). With `rtn`, dead columns are cast to nearest on the same pinned exponents
and take no part in error feedback; live columns are bit-identical to `zero`. The default is bit-identical to the original
routine. Tests: `tests/test_gptq_dead_columns.py` (13 tests). After a run, count all-zero weight columns per tensor with
`src/scripts/tile_diag/fc2_export_compare.py`.

### Producer-aware AWQ

`gptq_awq:...,producer_aware=true`, default off and bit-identical when off. AWQ folds `1/s_j` into producer row `j` of an
`up_proj -> down_proj` pair, but the search scores only `down_proj`. Under `w=panel`, a row with a large `s_j` becomes the
smallest row of its 32-row panel and is flushed ([Pi0 AWQ and panel damage](#pi0-awq-and-panel-damage)). The option adds the
producer rows' quantization error on their own grid, weighted by the channel's second moment at the `down_proj` input, to the
AWQ loss, so the search lowers alpha where that error outweighs the consumer gain. It combines with neither `awq_loss=cast`
(raises) nor norm producers (1-D weights).

```
--algo "gptq_awq:method=awq_gptq,actorder=true,producer_aware=true+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib"
```

Cost: about 69 GFLOP per alpha per layer for Gemma 2B (not timed). Evidence is a weights-only proxy, not a calibrated export
([Pi0 AWQ and panel damage](#pi0-awq-and-panel-damage)). Tests: `tests/test_awq_producer_aware.py`.

### K order

`k_order:rule=asc|desc|rand|none` (`seed=N` for `rand`), stacked last, for example `atlas_v1_awq+k_order:rule=asc`. The
hardware accumulates K tiles in issue order with one round-to-BF16 per tile, so the K order of a Llama MLP `down_proj` is a free
offline choice: the rows of `gate_proj` / `up_proj` and the columns of `down_proj` are permuted together. FP8 values, flush and
clip are unchanged; only the accumulation order changes. `asc` puts the least important channel (`mean|x_k| * ||W[:,k]||`)
first. Screen on TinyLlama `down_proj` layers 0, 7, 14, 21: `asc` changes the error of the bit-exact chain by -2% to -60%
(`ipt`) and -10% to -57% (`sa`); `desc` makes it worse; `rand` varies by about +-20% between seeds. It refuses per-block
activation scales (`a=block*`), so it does not stack on `atlas_v1_awq_tile`; it was screened on the per-channel recipe only.
Code: `src/algos/k_order.py`, `src/scripts/k_order_screen.py`.

### Two-term activations

`--act-two-term {off,all,down,attnq,attn,all+attn}` (eval time; `src/chip/hw_two_term.py`). Each activation is represented as
two FP8 terms, `x ~ dq(Q(x)) + dq(Q(x - dq(Q(x))))`, each with its own per-token scale; the layer runs twice on the same weight
tile and the two outputs are added in BF16. On TinyLlama layers the error splits into weight-only 0.0183, activation-only 0.0181,
both 0.0258; two-term activations bring the mean layer error from 0.0258 to 0.0183. The attention variants make Q (and P)
two-term. Cost (`hw_cost.two_term_layer_cost`): the second MXU pass costs time only above 11.9 blocks (about 380 tokens) per
resident weight tile; for attention it is free only for Q in a decode step whose KV cache streams from HBM (prefill: Q +50%
MXU, Q and P +321%). It uses the per-token cast and is not part of any tile recipe.

### Keep BF16

`keep_bf16:rule=input_conv` keeps every convolution that reads image pixels directly in BF16 ([keep-bf16.md](keep-bf16.md)).

### Hardware cost report

`python -I src/scripts/hw_report.py [--export model.safetensors] [--T 1 32 128 2048] [--mxu ipt|sa]` (model
`src/chip/hw_cost.py`, constants `src/chip/hw_params.py`, tests `tests/test_hw_cost.py`). It reports per-layer shapes, DMA
and MXU cycles, flush bounds, and where a pop scale is free. The report predates the tile recipes and was run on the per-channel
TinyLlama `atlas_v1` export; its findings motivate the tile layout:

- One full weight pass (968,884,224 B, 4 KiB DMA bursts) is 722,878,464 cycles (1.446 s at 500 MHz), 764 cycles per 1 KiB
  tile. Counting weight DMA only, a layer becomes compute-bound at 24 blocks of 32 tokens (737 tokens); with output stores,
  at 77 blocks (about 2.5k tokens) for K=2048 and 32 blocks (about 1k tokens) for K=5632.
- The exported `[N, K]` layout feeds the MXU weight push with no transpose.
- A pop scale is free only if the output feeds an FP8 matmul directly, the producer's activation scale is static, the weight
  scale is constant within each 32-channel group, and the consumer's scale is static and constant along its K. In a Llama
  layer only `v_proj -> P.V` and `P.V -> o_proj` are such edges. As exported (per-token, per-channel) none of the 154 linears
  qualifies.
- An `npu_model` simulator kernel (`src/scripts/npu_model_l2.py`) matched the `ipt` model bit for bit (0 mismatches of
  504,832 BF16 words).

### Per-phase graphs and loop manifest

DiT and Pi0 run a sampling loop (DiT 100 DDPM steps, Pi0 10 Euler steps) that one graph cannot hold. No boundary spec exists
from the compilers team; their op inventory captures the whole inference unrolled and non-strict, and their device route is a
hand-written host graph that owns the loop, cache and stop conditions. The default taken here is one graph per phase plus a
host-owned loop described by `loop_manifest.json`; an unrolled whole-inference graph was not built.

| | Multi-Task DiT | Pi0 |
|---|---|---|
| phases and calls per action chunk | `encode` x1, `step` x100 (outputs the noise prediction) | `prefix` x1, `step` x10 (outputs `v_t` [1, 10, 32]) |
| constant over the loop | `conditioning_vec` [1, 4160] bf16 | `state`, `prefix_pad_masks`, 36 K/V tensors [1, 1, 868, 256] bf16 |
| host-owned | noise draws (99 of 100 steps draw variance noise), DDPM arithmetic with clipping, action slice, normalization | noise draw, float32 time recurrence (`dt` = -0.10000000149011612), Euler update, preprocessing |
| `qlinear` / `qk_matmul` / `pv_matmul` / `quantize_act` | encode 145 / 24 / 24 / 241; step 34 / 6 / 6 / 58 | prefix 615 / 99 / 99 / 1,011; step 131 / 18 / 18 / 203 |
| checker | 25 of 25 each | 25 of 25 each |

The manifest records the step count, the scheduler table (DiT) or float32 times (Pi0), per-call inputs and shapes and the host
operations, each read from the checkpoint or the code. Build it with `src/scripts/tile_flow/make_loop_manifest.py dit|pi0
--single-step-bundle DIR --out PATH`; capture the phases with `policy_fx_capture_phases.py` (DiT, strict, torch 2.11) or
`pi0_phase_capture.py` (Pi0, non-strict, torch 2.7.1); compare with `compose_phases.py` (`--eager-loop` for the full loop).

`capture_fx` now counts calls per module: a module called N times yields N node sets (each with its own `quantize_act`) on one
weight buffer pair, a module never called is recorded as absent (`unused_quantized_modules`, `unused_attention_modules`). The
Pi0 `prefix` therefore calls SigLIP three times (one per camera) instead of stacking the cameras into one call as the
single-step bundle does. The phase bundles of a model hold disjoint quantized sites; the union is the model. Tests:
`tests/test_loop_manifest.py`, `tests/test_fx_calls.py`, `tests/test_verify_bundle_phases.py`, `tests/test_dit_hoist.py`. The
DiT's adaLN modulation now has its own phase ([DiT modulation phase](#dit-modulation-phase)); the rows below are the two-graph
layout.

Equivalence (in-process, reference emulation):

- DiT: `step(encode(obs))` is byte-identical to the single-step bundle; the 100-step host loop equals eager
  `conditional_sample`, max abs difference 0.0.
- Pi0: one step through the phases is byte-identical to the single-step graph (which also shows, on real weights, that stacked
  cameras equal three SigLIP calls). The 10-step host loop equals eager `sample_actions(num_steps=10)`: 0 of 320 elements
  differ on both cases (one process, 8 threads, one noise draw per case). This is a consistency check, not an accuracy check.
- The Pi0 phase bundles were captured from the `gptq_panel_static_noawq` export (GPTQ on the panel grid, AWQ off, without
  `dead_columns=rtn`).

Fixture reproducibility: the recorded `fixtures.safetensors` of the single-step DiT and Pi0 bundles do not reproduce
bit-exactly on the build host (DiT case 0 max abs 0.0234; Pi0 0.036 and 0.045), although host name and torch build match.
DiT results depend on CPU kernel dispatch; the Pi0 output changes with the thread count (1, 4 and 8 threads give 0.036 to
0.056 against the fixture). Root cause not found: use same-process comparisons with a fixed thread count.

Strict Pi0 export under torch 2.7.1 (pinned by openpi) is blocked by six issues in the driver, openpi and `fx_export`; the
compilers' own exports are non-strict, so this is not blocking.

#### DiT modulation phase

In the Multi-Task DiT, `time_mlp` and every block's `adaLN_modulation` (SiLU, then Linear 4416 -> 3072 with bias) read only
the timestep and `conditioning_vec`, never the noisy actions (lerobot `DiffusionTransformer.forward`,
`TransformerBlock.forward`). The timesteps come from the scheduler and `conditioning_vec` from `encode`, so all 100 steps'
modulations are known before the loop. The hoisted layout, now the default of `make_loop_manifest.py dit`
(`--layout two_phase` keeps the old one), has three graphs:

| graph | calls per chunk | inputs | output |
|---|---|---|---|
| `encode` | 1 | observation | `conditioning_vec` [1, 4160] |
| `modulate` | 1 | `conditioning_vec`, `timesteps` int64 [100] | `modulation` [6, 100, 1, 3072] |
| `step` | 100 | `noisy_actions`, `modulation[:, i]` [6, 1, 3072] | predicted noise |

`modulation[l, i]` is block `l`'s adaLN output at the i-th scheduler timestep (the six chunks shift, scale and gate for
attention and MLP, in lerobot's order). The step graph is the noise predictor without `time_mlp` and `adaLN_modulation`;
the host passes the step's slice. `modulate` calls each of its 8 Linears in calls of `steps_per_call` timesteps (default 25,
so 4 calls of 25 rows); the manifest records `steps_per_call`, `calls_per_site` and the hoisted sites. The three bundles hold
disjoint quantized sites; their union is the model.

Weight bytes read from the captured graphs (`phase_sizes.py`; E4M3 codes, int32 exponents and BF16 bias of the noise
predictor's quantized Linears, `atlas_v1_awq_tile_static+keep_bf16:rule=input_conv`):

| | per step | modulate per chunk | noise predictor per chunk |
|---|---|---|---|
| two graphs (`step` holds `time_mlp` and adaLN) | 96.2 MiB (34 `qlinear`) | | 9,615 MiB |
| hoisted | 18.2 MiB (26 `qlinear`) | 311.9 MiB (8 sites x 4 calls; 78.0 MiB once) | 2,129 MiB |

Exactness conditions (the hoisted loop must give the step-by-step result byte for byte):

- Static or per-token activation scales on the hoisted sites: each row is quantized on its own, so a step's rows get the
  codes the per-step graph gives them. A dynamic scale that spans rows (`a=block` per 32 rows, per tensor) would put several
  steps in one scale group and change the codes, so the capture refuses `steps_per_call` > 1 for such a recipe;
  `--steps-per-call 1` keeps the per-step grouping and is exact, but then reads the adaLN weights 100 times per chunk unless
  the compiler batches the calls.
- Row count of the reference emulation's matmul: the hardware computes each output row on its own, but the CPU BF16 matmul
  behind `atlas::qlinear` is not row-count invariant. On the real checkpoint (8 threads), 2 to 25 rows per call give the
  one-row results exactly; 50 and 100 rows differ in 57 of 1,843,200 adaLN outputs (max 0.0078), which changes the chunk
  (84 to 97 of 112 action elements, max 0.098). Hence 25 by default.
- The adaLN Linear is not split into its time and observation halves: that would change the accumulation and the rounding.
  Computing the observation half once per chunk is a possible follow-up, not an exact one.

Checks: `tests/test_dit_hoist.py` (tiny lerobot DiT: the hoisted step equals the stock noise predictor; the hoisted 100-step
loop equals `conditional_sample` for 1, 25 and 100 steps per call; the captured modulate and step graphs reproduce the eager
loop; the dynamic recipe is refused when batched and exact with one step per call). On the real checkpoint,
`dit_hoist_check.py` compares `policy._generate_actions` with the hoisted loop on frames of the evaluation protocol (CPU,
8 threads, exports calibrated on 8 frames): 0 differing elements with 1 and 25 steps per call under
`atlas_v1_awq_tile_static` and `atlas_v1_awq` (per-token scales), and with 1 step per call under `atlas_v1_awq_tile`
(batching refused); 4 frames per recipe. The hoisted phase bundles (static recipe) pass
`verify_bundle.py` (25 of 25 each); one step through them equals the stock noise prediction on 4 cases (timesteps 99, 62,
25, 88), and the 100-step host loop over the three graphs equals eager `conditional_sample`, max abs difference 0.0.

```bash
python src/scripts/tile_flow/make_loop_manifest.py dit --single-step-bundle B --out P/loop_manifest.json   # hoisted, 25
for ph in encode modulate step; do
  python src/scripts/tile_flow/policy_fx_capture_phases.py dit $ph --export-dir E --single-step-bundle B \
      --loop-manifest P/loop_manifest.json --out P/$ph
done
python src/scripts/tile_flow/compose_phases.py dit P B P/compose.json --eager-loop E
python src/scripts/tile_flow/phase_sizes.py P --out P/sizes.json
python src/scripts/tile_flow/dit_hoist_check.py --export-dir E --out check.json [--write-cases B]   # B without a graph
```

Without a captured single-step bundle, `dit_hoist_check.py --write-cases B` writes capture cases (real frames, stock noise
prediction) in that layout; `compose_phases.py` then compares one step against those recorded outputs instead of a graph.

Other models: SmolVLA's action expert runs `k_proj` and `v_proj` of each cross-attention layer (8 of 16) over the 177 cached
prefix K/V rows at every one of its 10 steps (`smolvlm_with_expert.py` `forward_cross_attn_layer`), although that input is
the same for all steps; computing them once in a prefix phase would be exact (same inputs, same row count). There is no
SmolVLA prefix / step capture yet (`fx_export --boundary smolvla_flow_loss` is a training forward), so this needs the phase
boundaries, manifest, composer and tests first. Pi0's step re-projects nothing from the prefix (the cache holds the
projected, rotated K/V); its only per-step invariant is `state_proj(state)` (32 -> 1024, read from lerobot's port; openpi's
source was not on the build host).

## Compiler handoff

### What a bundle contains

Contractions are `atlas::qlinear`, `atlas::qconv2d`, `atlas::qk_matmul` and `atlas::pv_matmul`, each fed by
`atlas::quantize_act`. Weights are E4M3 codes (uint8) with an int32 exponent per 32 output channels over all K; activation
exponents are per 32-token block or one static value per site. Every scale is constant along K, so the pop constant is
`2^(sx + sw)`, applied once at `vmatpop`. BF16 by policy: `lm_head` and patch embeddings selected by `keep_bf16`; the float32
RoPE outer product and BF16 / float32 softmax, norm and residual nodes are ordinary ops. Independent check: `verify_bundle.py`
(25 checks).

| Model | Bundle (recipe) | Boundary | `qlinear` | `qconv2d` | qk / pv pairs | `quantize_act` | Checker |
|---|---|---|---|---|---|---|---|
| TinyLlama | `atlas_v1_awq_tile_static`, `atlas_v1_awq_tile` | full causal LM forward, batch 1, seqlen 64 | 154 | 0 | 22 | 242 | 25 of 25 |
| TinyLlama, folded | `atlas_v1_awq_tile_static_packfold` | as TinyLlama | 154 | 0 | 22 | 242 | 25 of 25 |
| SmolVLA | `atlas_v1_awq_tile_static` | flow-loss forward, 1 image | 302 | 1 | 36 | 447 | 25 of 25 |
| Multi-Task DiT | `atlas_v1_awq_tile_static_attn` | encoder plus one noise-predictor call | 179 | 0 | 30 (24 CLIP, 6 DiT) | 299 | 25 of 25 |
| Multi-Task DiT phases | `..._attn_phases/{encode,step}` + `loop_manifest.json` | `encode` once, `step` 100 times | see [Per-phase graphs and loop manifest](#per-phase-graphs-and-loop-manifest) | | | | 25 of 25 each |
| Multi-Task DiT hoisted phases | `atlas_v1_awq_tile_static` (calibration 8 frames), `{encode,modulate,step}` + `loop_manifest.json` | `encode` once, `modulate` once, `step` 100 times | 145 / 32 (8 sites x 4) / 26 | 0 | 24 / 0 / 6 | 241 / 32 / 50 | 25 of 25 each |
| Pi0 | `gptq_panel_static_noawq_deadrtn` (candidate) | one sampler step, non-strict | 420 | 0 | 63 | 672 | 25 of 25 |
| Pi0 phases | `gptq_panel_static_noawq_phases/{prefix,step}` + `loop_manifest.json` | `prefix` once, `step` 10 times | see [Per-phase graphs and loop manifest](#per-phase-graphs-and-loop-manifest) | | | | 25 of 25 each |

### Pack-scale admission range

The compiler's `lowerPack` rejects activation pack codes outside 122..128 ("observed E8M0 scale codes 122..128 only"); no
file read for this analysis states a hardware reason. The RTL defines every code 0..255 as a shift, and the `npu_model` software model passes an
exhaustive pack test over all 256 codes. Weights are never packed and are unaffected. Static sites outside the range (code =
`127 + exp`; attention p at static `-8` is code 119, always outside):

| Bundle | Static sites | Outside | Linear inputs outside | Attention p outside | Code range |
|---|---|---|---|---|---|
| TinyLlama `atlas_v1_awq_tile_static` | 176 | 114 | 92 of 154 | 22 of 22 | 117..126 |
| SmolVLA `atlas_v1_awq_tile_static` | 316 | 244 | 231 of 303 | 13 of 13 | 117..125 |
| Multi-Task DiT `atlas_v1_awq_tile_static` (first bundle) | 203 | 88 | 64 of 179 | 24 of 24 | 119..126 |
| Multi-Task DiT `atlas_v1_awq_tile_static_attn` | 209 | 94 | 64 | 30 of 30 | |
| Pi0 `atlas_v1_awq_tile_static` | 483 | 127 (1 above 128) | 64 of 420 | 63 of 63 | 117..133 |
| Pi0 `gptq_panel_static_noawq` | 483 | 124 | 61 of 420 | 63 of 63 | |
| Pi0 `gptq_panel_static_noawq_deadrtn` | 483 | 125 | 62 of 420 | 63 of 63 | |

Dynamic exponents are not in these counts. Attention q, k, v are dynamic in every recipe: on TinyLlama real text 1060 of 1408 q,
48 of 1408 k and 1408 of 1408 v exponents (layer, batch, head) are outside the range. Under the dynamic `atlas_v1_awq_tile`
recipe, 2080 of 2464 blocks and 149 of 154 Linear sites have a block outside. For dynamic recipes `policy.json` reports
`compiler_pack_scale_admission.status: not_checked_dynamic`; only capture-case fixtures are reported in `graph_inspection.json`.
A dynamic exponent cannot be expressed at all today: `scale_code` is a compile-time attribute.

Options, in the recommended order:

| Option | Where | Exact | Sites fixed (TinyLlama / SmolVLA / DiT / Pi0) |
|---|---|---|---|
| A. Fold `2^k` into the producer ([Pack fold](#pack-fold)) | exporter | yes (0 differing bytes over 116,654,080 elements at 92 TinyLlama sites) | no added instruction: 92 / 221 / 24 / 61 estimated; implemented fold: 92 / 181 / 24 / 61 |
| F. Compiler `vmul.bf16` by `2^k` per tile, then pack at an in-range code | compiler | yes | all; universal fallback |
| D. Widen the admission (the census needs 117..133) | compiler | if the RTL honours the shift | all; needs RTL runs at the new endpoints |
| G. Use the MXU pop as the pack (kit style, `--lower-atlas-guarded-pack`) | compiler | yes | 0 today: it requires a provably finite source, and every packed tensor appears to pass a multiply, exp or add after the matmul (inferred from the graphs, not run) |
| B. Clamp the exponent into range | exporter | no: changes quantization | all; stop-gap only |
| H. Calibrate attention q, k, v statically | exporter and flow | n/a | none of the counts above, but required for any compile-time q, k, v scale |

Clamping (B) on TinyLlama: perplexity 12.8756 with all sites clamped, 12.8609 Linear only, 12.9027 p only, against 12.8980
unclamped; at the 92 affected Linear inputs 4.356% of nonzero values flush instead of 0.917%, and 60.1% of nonzero attention
p entries instead of 29.1%. Folding attention p (option A) costs one `vmul` on the broadcast `recip(sum)` per (head, 32-row
block), because the flow's p is normalized and `recip(sum)` is data (inferred, not run).

### What the compilers team needs to confirm

Nothing below is decided.

1. **Loop ownership.** Per-phase graphs plus a host-owned loop are delivered; an unrolled whole-inference graph is not built.
2. **`pop_shift`.** `atlas.pv_matmul` gains a trailing `pop_shift` (default 0), emitted only by folded bundles. That the
   kit's existing BF16-pop `vmul` constant `2^(sp+sv)` can absorb the extra `2^k` is inferred, not run on RTL.
3. **Pack range.** Widen 122..128, accept exporter folds, or use the `vmul` fallback. Unfolded bundles contain out-of-range sites.
   A compiler that pins `sw` to the stored value instead of using the product `sx + sw` would mis-handle a folded bundle.
4. **Attention q, k, v stay dynamic** per (batch, head); only p is static. Dynamic scales are not compile-time constants.
5. **Partial tiles.** DiT attention has 24 tokens (24 x 64 Q / K / V, 24 x 24 P per head); Linear weights whose output size is not
   a multiple of 32 carry `per_panel(32,partial_tail)` (35 in SmolVLA). `tile_native` checks exponent constancy, not shape.
6. **Non-strict Pi0 export** keeps unused lifted buffers (`model.pt2` of 4.1 to 4.3 GB); ignore inputs no node reads
   (`graph_inspection.json` `unused_quantized_modules`). The strict DiT phase bundles are 155 MB (step) and 184 MB (encode).
7. **Recorded fixtures** of the older single-step DiT and Pi0 bundles are not bit-exact references (see above).

## Results

### TinyLlama quality

Calibration: 16 WikiText-2 train windows of 2048 tokens, seed 1234. Perplexity: WikiText-2 test, 16 windows of 512 tokens,
exact float32 accumulation. Relative L2, top-1 agreement and KL: final logits against BF16 on the first 2 x 256 test tokens.

| Export | Perplexity | Logits rel. L2 | Top-1 agreement | Logit KL | Weight exponents | Codes with exponent field 15 |
|---|---|---|---|---|---|---|
| BF16 | 12.7558 | | | | | |
| `atlas_v1_awq_tile_static` | 12.8980 | 0.0536 | 0.951 | 0.0128 | 12,320 | 0.031% |
| `atlas_v1_awq_tile` | 12.8287 | 0.0522 | 0.934 | 0.0121 | 12,320 | 0.031% |
| `atlas_v1_awq` (older per-channel export, GPU-calibrated, code not recorded) | 12.8451 | 0.0579 | 0.936 | 0.0185 | 394,240 | 0.706% |
| tile static, GPTQ only (AWQ off) | 12.9008 | 0.0646 | 0.922 | | | |

- Static versus dynamic is controlled (same commit, identical weights). AWQ on versus off is controlled. The tile exports
  versus the older `atlas_v1_awq` export are confounded (calibration host and code differ, and 6 of 66 AWQ fold groups picked a
  different alpha): dynamic differs from it by 0.016 perplexity and both tile recipes are within 0.07, which shows no
  measurable loss, not a recipe win. No per-channel static control was run, so there is no granularity-only number.
- Panel scales leave far fewer weight codes in the exponent-field-15 range (above 240) but flush more small weights to zero
  (0.067% versus 0.025%). 150 of 154 sites carry more than one distinct exponent; no reserved code in 968,884,224 weight codes.
- Exponent table: 12,320 groups versus 394,240 rows; stored bytes are identical today (1,576,960 B); one E8M0 byte per group
  would be 12,320 B.
- MXU models on nine isolated layers (layers 0, 10, 21 of each projection class, static export, relative L2 against BF16):
  exact float32 0.034 to 0.095, `ipt` 0.035 to 0.096, `sa` 0.047 to 0.117.
- Static activations on the 2 x 256-token test batch: 6 of 201,850,880 Linear-input values exceed `448 * 2^s` (3 of 154
  sites); the mean over sites of the fraction of all elements flushed to zero is 1.15%.

### TinyLlama ablation

Removing AWQ at panel granularity with static activations changes perplexity from 12.8980 to 12.9008 and logits relative L2
from 0.0536 to 0.0646 (one calibration set, 16 windows, no confidence interval). TinyLlama does not show the Pi0 panel damage
(weights-only scan against the per-channel export): the AWQ fold scale range is at most 4.87 binades (Pi0: 19), the
within-panel row-amax spread of the folded `up_proj` is at most 6.0x at the median and 9.9x at p95, the composite MLP error has
median 0.0449 (tile) against 0.0450 (per-channel), and alpha-matched layers agree within -0.5% to +0.7%. Layer 5 is large in
both exports alike, an artifact of the weight-space check, not panel damage.

### SmolVLA, Multi-Task DiT and Pi0 graphs

- **SmolVLA** (32 calibration frames, seed 1234): 317 quantized sites, recorded as all tile-native by the export (made before
  the uncalibrated-fallback fix); capture and fresh-process replay passed. The one dynamic Linear site, `vlm.lm_head`, has no
  calibration data and is never called in the flow-loss boundary; under the current rules it counts as an uncalibrated
  fallback and not tile-native. No quality number.
- **Multi-Task DiT.** The first bundle left the DiT's own six `RoPEAttention` blocks as plain BF16 attention. They are now
  attention sites (`src/reference/ref_attention.py`, variant `vision`), which needed a re-export: `atlas_v1_awq_tile_static_attn`
  has 209 of 209 sites tile-native. The CLIP sites use variant `text` while the reference DiT port uses `vision`; not decided.
- **Pi0** (8 DROID calibration frames, seed 0): the `sample_actions` graph holds 27 SigLIP, 18 PaliGemma and 18 action-expert
  attention sites. The SigLIP patch embedding (float32 `[1152,3,14,14]`) is kept by policy; the two `lm_head` are skipped.

### Pi0 AWQ and panel damage

Drift metric: relative L2 of the Gemma layer-L `q_proj` input of the stock Pi0 model carrying dequantized export weights,
against the stock model (one DROID frame, noise seed 0).

| Weights | L9 | L17 |
|---|---|---|
| older per-channel `atlas_v1_awq` (GPU) | 0.0252 | 0.0223 |
| per-channel control, CPU, same commit and calibration | 0.0240 | 0.0231 |
| `atlas_v1_awq_tile_static` (AWQ + panel) | 0.1273 | 0.0852 |
| tile, with only Gemma layer-0 `up_proj` taken from the control | 0.0230 | 0.0228 |
| tile, with only Gemma layer-0 `down_proj` taken from the control | 0.1270 | 0.0849 |
| GPTQ on the panel grid, AWQ off (`gptq_panel_static_noawq`) | 0.0237 | 0.0207 |
| candidate, AWQ off + `dead_columns=rtn` (`gptq_panel_static_noawq_deadrtn`) | 0.0244 | 0.0222 |

Cause: in the Gemma layer-0 `up_proj -> down_proj` pair AWQ picked alpha 0.8 with `s` spanning 19 binades. The 53 rows with
the largest `s_j` sit about 12 binades below their 32-row panel amax in the folded basis and are flushed or badly rounded under
the shared exponent. Ruled out: missing norm fold, activations, calibration frames, shape bugs. The
`atlas_v1_awq_tile_static` Pi0 bundle is therefore superseded for quality.

Fix 1, **AWQ off** (GPTQ on the panel grid), brings drift back to the per-channel level (table); this is the candidate.
Fix 2, **[producer-aware AWQ](#producer-aware-awq)**, weights-only proxy on Pi0 Gemma layer 0 (stock weights, `s` recovered from
the export, proxy Hessians, round-to-nearest casts): damaged producer rows 4 to 0, composite error 0.6047 to 0.0277, alpha
0.9 to 0.45. The proxy search does not reproduce the recorded alphas, so this shows the mechanism only. On TinyLlama layers 0
and 1 with real calibration there is no regression (MLP error -0.7% and +0.4%). Not validated end to end: the check that
settles it is one Pi0 export with `producer_aware=true` and the L9 / L17 drift.

### SigLIP fc2 zero columns

Both the per-channel and the panel Pi0 exports show SigLIP `mlp.fc2` layers 17 to 25 with weight-only relative error 0.85 to
0.98. This is not E4M3 flush. In layer 23, 4177 of 4304 input columns are entirely zero because GPTQ zeroes every column whose
calibration Hessian diagonal is zero, and the fc2 input (BF16 GELU of fc1) is exactly zero for most channels: 98.81% of
entries, only 75 of 4304 channels nonzero over 512 tokens of one frame. Round-to-nearest on the same exponents flushes 0.035%
and gives weight error 0.0266, so the `mse` rule, smaller `rows` or per-row exponents change nothing. The alive channel set
varies by frame (72 per image, 127 over the 8 calibration frames), so a new frame can hit a zeroed column. `state_proj` loses 24
of 32 columns the same way; they are the padding of an 8-dimensional state and harmless.

`dead_columns=rtn` on layer 23 alone, calibrated on one image and tested on another: held-out output error 0.0226 against
0.1156 for `zero` (round-to-nearest 0.0255); the calibration-image error is identical for both.

### Pi0 candidate bundle

`gptq_panel_static_noawq_deadrtn` (recipe in [Recipes](#recipes), same 8 calibration frames as the previous no-AWQ bundle):

| | Previous (`gptq_panel_static_noawq`) | Candidate (`+ dead_columns=rtn`) |
|---|---|---|
| all-zero weight columns, SigLIP | 40,392 | 305 (all in layer 1 `fc2`; plain round-to-nearest gives the same 305) |
| all-zero columns, SigLIP fc2 layers 17 to 25 | 36174 | 0 |
| all-zero columns, Gemma LM / expert / `state_proj` | 635 / 10 / 24 | 0 / 0 / 0 |
| weight error against stock, mean / median / max | 0.06045 / 0.03381 / 0.97883 | 0.03554 / 0.03337 / 0.11443 |
| layers with weight error above 0.1 | 15 | 2 |
| Gemma drift L9 / L17 | 0.0237 / 0.0207 | 0.0244 / 0.0222 |
| action error `delta_rel_l2` against stock, 4 / 8 held-out observations | 0.0564 / 0.0574 | 0.0594 / 0.0539 |

- On held-out images, `rtn` is never worse than `zero` in fc2 layers 17 to 25 and 1.7x to 5.1x lower in layers 22 to 25.
- The drift is 3% (L9) and 7% (L17) higher than the previous bundle. The cause is not isolated: the two exports are not
  bit-reproducible from each other (all 223 non-SigLIP tensors without zero columns differ in codes; 2 of 483 activation
  exponents differ), and no repeat export was run to measure run-to-run spread.
- The action error changes sign between 4 and 8 observations, so the two are not separable. Both stay within cosine 0.9983 to
  0.9987 of stock; quantization adds at most 2.7% to the stock policy's own MSE against recorded actions.
- Not measured: closed-loop rollouts, run-to-run spread, the FX graph's output against stock.

### Hardware and cycle findings

These were measured with the separate hardware bring-up harness (`npu_model` with RTL numerics), which reads only
`export/model.safetensors`, never the FX graph.

- **Cycles, TinyLlama**, 110 plain-projection launches (layers 0 and 10), 110 of 110 golden-exact. Unmodified program plus host
  correction: 459,785 cycles (32x2048x256) and 4,501,837 (`down`, 32x5632x1024). Per-row on-device rescale variants: +2.7% to
  +2.9% (+1.1% to +1.2% on `down`). Tile-aware scale at pop (one 4-byte scalar per 32 output columns): 460,540 to 460,622
  cycles (+0.16% to +0.18%) and 4,504,708 to 4,504,748 on `down` (about +0.06%). The folded bundle's 20 projection launches
  were accepted and exact. Not exercised: `pop_shift` on `pv_matmul` (no PV launch).
- **Pi0 candidate**, 75 launches golden-exact, 8 of 8 RTL replays whole-arena exact. Device error against BF16 is large and
  does not depend on the recipe: Gemma `down` layers 1 to 11 are 12.5% to 37.1% off (`atlas_v1_awq`: 11.9% to 37.3%), SigLIP
  `fc2` layers 0 to 3 12.3% to 45.4% (`atlas_v1_awq`: 12.6% to 45.3%). Exact FP8 math on the same codes and activations is only
  1.2% to 2.7% (Gemma `down`) and 0.7% to 1.7% (SigLIP `fc2`) off, so the gap is in device accumulation (suspected: BF16
  K-ordered accumulation on long K; not isolated). Gemma `down` layer 7: 83% of non-zero activations flush to zero in the cast.

### Test suite

Full suite on the merged branch (pack fold and phase graphs): 1041 passed, 46 skipped, 0 failed, 32 minutes 35 seconds.
Existing recipes are bit-identical before and after the tile work on a tiny Llama (12 logit hashes); a default export with
the pack-fold code is byte-identical to the base commit (49 tensors, `policy.json`, graph files). Command:
`python -m pytest tests -q -p no:cacheprovider --deselect tests/test_algo_cuda.py`.

### Superseded results

| Earlier statement | Replaced by |
|---|---|
| `verify_bundle.py` has 24 checks; DiT passed 23 of 24, Pi0 22 of 24, SmolVLA failed 7 of 25 | 25 checks; declared exceptions are separated from unexpected contractions, counts come from the graph; every bundle passes 25 of 25 |
| Earlier full-suite runs (787, 870, 889 passed) | 1041 passed, 46 skipped, 0 failed |
| DiT bundle with 24 qk / pv pairs, 275 `quantize_act`, 6 unquantized attention blocks, 88 of 203 sites outside 122..128 | `atlas_v1_awq_tile_static_attn`: 30 pairs, 299 `quantize_act`, 0 unquantized, 94 of 209 outside |
| Pi0 `atlas_v1_awq_tile_static` as the Pi0 bundle | superseded for quality by the AWQ-off bundles; candidate `gptq_panel_static_noawq_deadrtn` |
| `dead_columns=rtn` validated on one layer only, no Pi0 re-export | full Pi0 re-export (candidate bundle) |
| SmolVLA free folds 221 (research estimate) | 181 folded; 32 shared-attention P.V sites and 8 VLM `v_proj` sites are refused |
| Pack-range analysis folds to unit code 127 | implementation folds to the nearest in-range code |
| Pi0 10-step host loop against eager "not finished" | done: 0 of 320 elements differ on both cases |
| Camera stacking checked only on a tiny Pi0 and a 2-layer real-width SigLIP | also shown on real weights through the phase composition |
| Uncalibrated Linear under a static recipe marked tile-native; dynamic pack admission reported 0 sites | declared as `uncalibrated_fallback` and not tile-native; dynamic status `not_checked_dynamic` |
| "No Pi0 quality number" | action error against stock exists for the AWQ-off bundles (reference emulation, 4 and 8 observations); still none for SmolVLA and DiT |

## Limits

- **No hardware equivalence is claimed.** Quality numbers use exact float32 accumulation; `sa` / `ipt` were run only on nine
  isolated layers on 64 rows. On chip profiles, panel / block scales run through the existing MXU model with the exponent table
  applied after the matmul; the BF16 accumulation order is the model's, not the RTL's.
- **Quality coverage.** TinyLlama perplexity only (16 windows); a Pi0 action-error proxy; none for SmolVLA or DiT.
- **The static recipe is not fully static.** Attention q, k, v keep a dynamic per-(batch, head) exponent; only p is static.
- **Static activations saturate silently** on unseen data with a larger range.
- **Uncalibrated Linear in a static recipe** (for example SmolVLA `vlm.lm_head`) keeps the dynamic per-tensor reference rule.
  It is declared in `policy.json` (`activation.uncalibrated_fallback: true`, `tile_native: false`, top-level
  `uncalibrated_fallback`), the export warns, and `verify_bundle.py` fails `static_recipe_fully_static` unless the export or the
  verifier is run with `--allow-uncalibrated-fallback`. Older exports lack the field; the verifier then derives the count.
- **Fixed graph shape.** Batch 1 and one `--seqlen`; another length needs a new capture. Native FP8 covers Linear, Conv2d
  and the two attention matmuls only; `lm_head` stays BF16.
- **AWQ folds.** Scales are folded into `input_layernorm`, `post_attention_layernorm` and `up_proj` rows. Under grouped-query
  attention the `v_proj -> o_proj` fold is not implemented. Because `up_proj`'s output is scaled by `1/s`, the "local" error of
  `src/layer_rmse.py` is invalid for folded layers (smoke model: 1.01, 1.27 and 8.80 local against 0.07, 0.95 and 0.10
  cumulative for gate, up and down).
- **`a=block` and `rows`.** `a=block` applies to Linear inputs only (Conv2d inputs use the per-tensor rule); `rows` other than
  32 are emulated but do not match the hardware tile.
- **Attention model.** The flow quantizes normalized p; the kit quantizes the unnormalized `exp` times 256. The cost model
  assumes a BF16 consumer at every pop. The FP8 range is 448 (RTL) versus 240 in one hardware spec.
- **Reproducibility.** `model.pt2` is tied to the torch version it was captured with; recorded fixtures of the single-step DiT
  and Pi0 bundles do not reproduce bit-exactly; the run-to-run spread of a Pi0 GPTQ export is unmeasured.
- **CPU calibration** of TinyLlama takes 19 to 26 minutes on 16 shared cores; larger models need a GPU or a reduced `--calib-n`.

## Open questions

1. Compiler: pack range (widen, accept folds, or `vmul` fallback), `pop_shift`, folded bundles as default, host-owned loop
   versus an unrolled graph, and the fold target code (nearest in-range, mostly 122, versus 126..128 where system tests exist).
2. Pi0: ship with AWQ off and `dead_columns=rtn` (current candidate), or wait for producer-aware AWQ to be validated.
3. Quality numbers for SmolVLA and DiT; closed-loop benefit of any Pi0 variant; root cause of the unreproducible fixtures.
4. Hardware: the 12% to 45% device error against BF16 on long-K Pi0 layers that exact FP8 math does not show, and the Gemma
   `down` layer 7 activation flush (static exponent or BF16 opt-out); rulings H1, H2, H3; the pop saturation hole; 448 versus 240.
5. Static calibration of attention q, k, v, without which they have no compile-time scale code.
6. MXU0 versus MXU1 for projections, `nc` versus reference rule, bias handling, `lm_head`, the BF16 opt-out list, `panel` naming,
   and the CLIP attention variant of the DiT.

## Code map

| Area | Files |
|---|---|
| scale layouts and recipes | `src/algos/granularity.py`, `src/algos/__init__.py`, `src/algos/gptq_awq.py` (`dead_columns`, `producer_aware`), `src/algos/k_order.py` |
| tile contract, export, replay | `src/exporter/tile_native.py`, `src/exporter/quant_export.py`, `src/exporter/load_quantized.py`, `src/exporter/fx_export.py`, `src/exporter/atlas_fx_ops.py` |
| pack range | `src/exporter/pack_plan.py`, `src/exporter/pack_fold.py`, `src/scripts/tile_flow/pack_plan_census.py` |
| attention | `src/reference/ref_attention.py` |
| hardware model | `src/chip/chip_profile.py`, `src/chip/hw_params.py`, `src/chip/hw_cost.py`, `src/chip/hw_scale.py`, `src/chip/hw_two_term.py` |
| scripts | `src/scripts/tile_flow/` (`verify_bundle`, `inmem_check`, `quality`, `dyn_admission`, `static_clip`, `compare_fx_logits`, `make_bundle_manifest`, `make_tiny_llama`, `pack_plan_census`, `make_loop_manifest`, `policy_fx_capture_phases`, `pi0_phase_capture`, `compose_phases`, `pi0_thread_repro`), `src/scripts/tile_diag/` (Pi0 drift, weights-only error, fc2 and AWQ diagnostics), `src/scripts/hw_report.py`, `src/scripts/npu_model_l2.py`, `src/scripts/k_order_screen.py` |
| tests | `tests/test_algo_tile.py`, `tests/test_tile_flow_defects.py`, `tests/test_pack_plan.py`, `tests/test_pack_fold.py`, `tests/test_gptq_dead_columns.py`, `tests/test_awq_producer_aware.py`, `tests/test_dit_rope_attention.py`, `tests/test_fx_calls.py`, `tests/test_loop_manifest.py`, `tests/test_verify_bundle_phases.py`, `tests/test_verify_bundle_smolvla.py`, `tests/test_hw_cost.py`, `tests/test_algo_k_order.py`, `tests/test_hw_two_term.py` |

Sources: Atlas RTL (`ucb-bar/atlas-npu`) and `npu_model` (`vmatpop.fp8`, `vpack`, `vunpack`), `atlas-mlir` `lowerPack`,
`atlas-kernels`; AWQ (Lin et al. 2024); GPTQ (Frantar et al. 2023); OCP microscaling (MX) formats, which scale per K block and
are therefore not realizable here.
