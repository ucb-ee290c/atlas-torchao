# How to take a model through AWQ / GPTQ calibration to a native FP8 FX graph

Everything below was run on CPU on branch `tile-aware` (commit `53ab820`), TinyLlama-1.1B-Chat, 2026-11-06. The
numbers in the tables are from those runs; they are measurements of one model, not guarantees for another.
Background: [tile-aware.md](tile-aware.md) (the scale layouts), [export.md](export.md) (the export files),
[fx-export.md](fx-export.md) (the graph). The helper scripts live in `scripts/tile_flow/`.

## 0. Setup

```bash
cd /scratch2/lorenhung/atlas-fp8/atlas-torchao-tile            # the worktree; src/ on PYTHONPATH
export PY=/scratch2/lorenhung/envs/atlas-lerobot/bin/python    # torch 2.11.0 CPU, transformers 5.5.4, torchao 0.18.0
export PYTHONPATH=src
export CAP="taskset -a -c 0-15 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"   # shared host: cap every heavy command
export MODEL=/scratch/agarcia05/TinyLlama-1.1B-Chat-v1.0       # local dir; any HF Llama-family dir works
export OUT=/scratch2/lorenhung/atlas-fp8/exports-tile/tinyllama/atlas_v1_awq_tile_static   # one dir per model x recipe
git status --short        # must be empty: the export and the FX capture record the commit and a dirty flag
```

Check the checkpoint before calibrating (the manifest stores this sha256): `sha256sum $MODEL/model.safetensors`
(TinyLlama: `6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933`).

## 0.5 Smoke test first (about 1 minute)

A 2-layer random Llama with spread row scales (so different 32-channel groups get different exponents) goes through a tile
recipe; `verify_bundle.py` checks the result and `inmem_check.py` (section 4) compares it with an in-memory re-quantization.

```bash
export TINY=/tmp/tiny-llama; export TOUT=/scratch2/lorenhung/atlas-fp8/exports-tile/smoke-tiny-llama/atlas_v1_awq_tile_static
$CAP $PY scripts/tile_flow/make_tiny_llama.py $TINY $MODEL
$CAP $PY src/eval.py --model $TINY --seqlen 128 --calib-n 4 --modes ref_conv --export-only --algo atlas_v1_awq_tile_static \
    --device cpu --export-dir $TOUT/export --out $TOUT/export_cli.json                                  # 19 s
$CAP $PY src/fx_export.py --export-dir $TOUT/export --model $TINY --out-dir $TOUT/fx --device cpu --seqlen 64 --native-fp8   # 22 s
$CAP $PY scripts/tile_flow/verify_bundle.py $TOUT/export $TOUT/fx $TOUT/verify_bundle.json
```

With the unspread default random init every weight exponent is 115, the constancy check passes trivially, and `verify_bundle.py`
fails its "not collapsed to per-tensor" check (on purpose: it proves nothing about grouping).

## 1. Choose a recipe

| Recipe | Weights | Activations | Attention | Use when |
|---|---|---|---|---|
| `atlas_v1_awq_tile_static` | one exponent per 32 output channels (`per_panel(32)`), rule `nc` | one **static** exponent per tensor (calibrated, no run-time amax) | p static -8; q, k, v dynamic per (batch, head), reference rule | **Primary.** A compiler-team report (unverified here, `tile-aware-checkin.md`) says every exponent must be a compile-time constant, which favors static scales; q, k, v are still dynamic |
| `atlas_v1_awq_tile` | same | **dynamic** per 32-token block over all K (`nc`) | q, k, v, p dynamic per (batch, head), `nc` | You need scales that follow the data and accept a run-time amax (ruling H2 open) |
| `atlas_v1_tile` | same | same as `atlas_v1_awq_tile` | same | Models with an input convolution (SigLIP / DiT patch embedding): keeps every `Conv2d` with 3 input channels in BF16 |
| `atlas_v1_awq` | one exponent per output channel | one exponent per token | per head | Not tile-native (needs per-row vector rescales); the comparison baseline |

All three tile recipes run `gptq_awq:method=awq_gptq,actorder=true` first, so AWQ and GPTQ both see the 32-channel grid
(tile-aware.md, "GPTQ and AWQ on the panel grid"). The two TinyLlama tile exports have byte-identical weight tensors
(checked tensor by tensor); `atlas_v1_awq_tile_static` adds 154 `<fqn>.input_scale_exp` tensors.

## 2. Calibrate (AWQ, then GPTQ) and export

```bash
# causal LM (TinyLlama). --calib-n and --calib-seed default to 16 and 1234: the same windows as the old tl_E0 export.
$CAP $PY -u src/eval.py --model $MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --device cpu \
    --export-dir $OUT/export --out $OUT/../atlas_v1_awq_tile_static_export_cli.json
```

Measured (16 cores shared, 2 jobs at once): `atlas_v1_awq_tile_static` 1542 s, `atlas_v1_awq_tile` 1146 s. The export
directory must not exist. Other model types use their own CLI with the same `--algo` / `--export-only` / `--export-dir`:
`src/eval_policy.py` (LeRobot policies, needs `--dataset` and `--norm-stats dataset`) and `src/eval_pi0.py`.
Pass `--calib-n N` to calibrate on fewer samples; state N next to any result.

## 3. Capture the native FP8 FX graph

```bash
$CAP $PY src/fx_export.py --export-dir $OUT/export --model $MODEL --out-dir $OUT/fx --device cpu --seqlen 64 --native-fp8
```

326 s for TinyLlama. `--seqlen` is fixed in the graph (batch 1). Use a multiple of 32 so the 32-token blocks of the
dynamic recipe are all full; 8 gave one partial block in the smoke test. `--model` supplies the architecture config only.
The capture already replays the saved graph in a fresh process (`fresh_process_verification.json`). The standalone command
(`$CAP $PY src/fx_export.py --verify-dir <fx dir> --device cpu`) was run only on the smoke bundle (3.7 s, passed); it rewrites
`fresh_process_verification.json` in that directory, and was not run on the TinyLlama bundles.

## 4. Verify (checklist)

```bash
$CAP $PY scripts/tile_flow/verify_bundle.py $OUT/export $OUT/fx $OUT/verify_bundle.json     # 25 checks, exit 0 = all pass
# optional flags: --allow-uncalibrated-fallback (accept undeclared uncalibrated Linear sites of a static recipe),
#                 --forbid-unquantized-attention (plain BF16 attention outside policy sites becomes unexpected)
# tiny model only (re-calibrates in memory with the same recipe / calibration / seed, then compares codes, exponents and
# logits with the bundle); run on the section 0.5 smoke model. args: MODEL RECIPE EXPORT FX CALIB_N CALIB_SEQLEN FX_SEQLEN OUT
$CAP $PY scripts/tile_flow/inmem_check.py $TINY atlas_v1_awq_tile_static $TOUT/export $TOUT/fx 4 128 64 $TOUT/inmem_check.json
python3 scripts/tile_flow/make_bundle_manifest.py $OUT                                    # writes $OUT/bundle_manifest.json
```

| # | Check | Where it is enforced |
|---|---|---|
| 1 | `model.safetensors` and `policy.json` sha256 equal `manifest.json` | `load_quantized` (raises) |
| 2 | every quantized Linear has `weight_codes` uint8 and `weight_scale_exp` int32 `[out, 1]`; no float weight stored | `verify_bundle.py` reads the safetensors directly |
| 3 | no code `0x7F` / `0xFF` in any weight | `verify_bundle.py`; `load_quantized` raises `TileContractError` |
| 4 | `weight_scale_exp` constant within every group of 32 output channels, and not collapsed to one value per tensor | `verify_bundle.py` (own implementation, not `tile_native`) |
| 5 | `graph_inspection.json` `tile_native.all_tile_native` is true; `still_emulated.emulated_modules` is empty | `verify_bundle.py` |
| 6 | graph holds one `atlas.qlinear` / `atlas.qconv2d` per quantized Linear / Conv2d site the graph calls, and qk / pv pairs equal to the per-site pair count derived from the graph (scope-less sites cross-checked with `graph_inspection.json`); a policy site the boundary never calls (SmolVLA `lm_head`) is exempt only when the lifted buffers and the bundle's `unused_quantized_modules` agree (`docs/algos/smolvla-verify.md`) | `verify_bundle.py` reads `model.pt2` |
| 7 | the only non-atlas contractions are declared: BF16-by-policy Linear / Conv2d (`host` / `skipped` sites such as `lm_head` and the patch-embedding conv), the float32 RoPE `inv_freq x position_ids` outer product, and plain BF16 `scaled_dot_product_attention` in blocks without a policy attention site (reported as `graph.unquantized_attention`); anything else fails | `verify_bundle.py` (operand dtypes and `nn_module_stack` from the graph) |
| 8 | weight buffers in the loaded program are uint8 / int32; no float weight parameter for a quantized layer (unused meta placeholders of a non-strict export are tolerated when the fresh-process report says `unused_float_weights_not_materialized: true`, they are meta and unconsumed) | `verify_bundle.py` |
| 8b | a static recipe has no undeclared uncalibrated Linear / Conv2d (`policy.json` `uncalibrated_fallback`; `static_recipe_fully_static`); a fallback site the graph never calls is reported, not failed, and the graph must hold no dynamic-rule quantize_act feeding a qlinear / qconv2d | `verify_bundle.py` |
| 9 | eager replay, captured graph and reloaded graph are byte identical on two inputs; fresh-process replay passes | `fx_export.py` (`verification.json`, `fresh_process_verification.json`) |
| 10 | in-memory model equals the exported bundle (codes, exponents, logits) | `inmem_check.py`; for a model too big to re-calibrate, `load_quantized` re-derives every code bit for bit |
| 11 | dynamic block exponents are constant within each 32-row group in the capture cases | `graph_inspection.json` `fixture_activation_stats` (`groups_constant`) |

## 5. Expected files

`$OUT/export/` (written by `eval.py`):

| File | Contents |
|---|---|
| `model.safetensors` | `<fqn>.weight_codes` uint8 `[out, K]`; `<fqn>.weight_scale_exp` int32 `[out, 1]` (the group value repeated for the 32 channels of a group); static recipe: `<fqn>.input_scale_exp` int32; every other tensor in its own dtype (norms, embeddings, BF16 `lm_head`) |
| `policy.json` | per site: weight granularity `per_panel(32)`, `panel_rows`, `e8m0_code_range`, `code_report`; activation rule (`static` + `exp`, or `dynamic` + `reduce: block`); attention operand rules; `tile_native`; `compiler_pack_scale_admission` |
| `manifest.json` | sha256 of the two files above, producing commit and dirty flag, library versions, checkpoint sha256, recipe (requested, resolved, expanded, per-member parameters incl. `rule` and `rows`), calibration (source, `n_samples`, `seed`, `seqlen`, `window_starts`, `data_sha256`), site counts |

`$OUT/fx/` (written by `fx_export.py`): `model.pt2` (the ExportedProgram; load with `import atlas_fx_ops` first),
`graph.txt`, `graph_signature.txt`, `graph_inspection.json` (per-site `tile_native`, native bindings, `bf16_by_policy`,
`still_emulated`, `fixture_activation_stats`, `compiler_pack_scale_admission`), `buffer_inventory.json`,
`fixtures.safetensors`, `parity.json`, `policy.json` (copy), `fx_manifest.json`, `verification.json`,
`fresh_process_verification.json`.

`$OUT/` (written by the verification scripts): `verify_bundle.json`, `bundle_manifest.json`, and for quality
`quality.json`, `ppl_w16x512.json` (section 6).

`weight_scale_exp` is `[out, 1]` int32 in both the tile and the per-channel recipes: the file size does not change. What
changes is the number of distinct values a consumer needs: one per 32 channels (TinyLlama: 12,320 groups) instead of one
per channel (394,240 rows). A compacted layout is not written by the exporter.

## 6. Quality numbers (optional but recommended before handing over)

```bash
# one fixed batch: wikitext-2 test, first 2 x 256 tokens. Per-layer and final-logits relative L2 vs BF16 with exact fp32
# accumulation; first / middle / last layer of each projection class again under the MXU models sa (MXU0) and ipt (MXU1).
$CAP $PY -u scripts/tile_flow/quality.py $MODEL 256 2 tile_static=$OUT/export=$OUT/quality.json
# wikitext-2 test perplexity on the first 16 windows of 512 tokens (bf16 baseline and the export, same windows)
$CAP $PY -u src/eval.py --model $MODEL --seqlen 512 --max-windows 16 --device cpu --from-export $OUT/export \
    --modes bf16 ref_conv --out $OUT/ppl_w16x512.json --mxu ipt --crosscheck off
# real-text E8M0 codes of the dynamic block recipe against the compiler's 122..128 (dynamic recipe only)
$CAP $PY scripts/tile_flow/dyn_admission.py $MODEL $OUT/export $OUT/dynamic_pack_admission_real_text.json 256 2
# static recipe: fraction of activation values that saturate / flush at the stored static exponents
$CAP $PY scripts/tile_flow/static_clip.py $MODEL $OUT/export $OUT/static_activation_saturation_real_text.json 256 2
```

TinyLlama results (batch 2 x 256 tokens; perplexity 16 x 512 tokens, bf16 = 12.7558):

| Export | Perplexity | Final-logits rel. L2 | Top-1 agreement | Weight exp. needed (groups) | Weight codes with exponent field 15 |
|---|---|---|---|---|---|
| `atlas_v1_awq_tile_static` | 12.8980 | 0.0536 | 0.951 | 12,320 | 0.031% |
| `atlas_v1_awq_tile` | 12.8287 | 0.0522 | 0.934 | 12,320 | 0.031% |
| `atlas_v1_awq` (old `tl_E0`, per channel, GPU-calibrated, older code) | 12.8451 | 0.0579 | 0.936 | 394,240 | 0.706% |

The `tl_E0` row differs in calibration provenance as well as in recipe (GPU box, commit not recorded), so its gap to the
tile rows is not a pure recipe effect. All accuracy figures use the framework's exact float32 accumulation unless a column says
`sa` or `ipt`; the isolated-layer runs under `sa` / `ipt` are in `quality.json` under `accumulation_models`.

## 7. Known limitations (read before relying on a bundle)

- **No hardware equivalence is claimed.** Quality numbers are exact float32 accumulation. `sa` / `ipt` were run only on isolated
  first / middle / last layers of each class, on 64 rows, not the whole model, and no RTL was run.
- **Compiler pack admission.** The compiler's `lowerPack` accepts activation E8M0 codes 122..128 today. Static recipe (policy):
  114 of 176 activation sites are outside (all 22 attention `p` sites at code 119, and 92 of 154 Linear inputs). Dynamic recipe (real text,
  `dyn_admission.py`): 149 of 154 Linear sites have at least one 32-token block outside. Nothing is clamped.
- **The static recipe is not fully static.** Attention q, k, v keep a dynamic per-(batch, head) exponent (reference rule); only `p` is static.
- **Static activations saturate on unseen data.** On the 2 x 256-token test batch 6 of 201,850,880 Linear-input values saturate (3 of 154 sites); the mean over sites of the fraction flushed to zero is 1.15% (`static_clip.py`). Unseen data with a larger range saturates silently.
- **Graph shape is fixed.** Batch 1, one `--seqlen`; dynamic block grouping runs along the flattened `B*T` axis, so another length needs a
  new capture. Last block is partial when `B*T % 32 != 0` (zero padded; the padding encodes to +0).
- **`lm_head` is BF16** (skipped by every recipe); the graph also has the float32 RoPE outer product and BF16 / float32 softmax, norm and
  residual nodes as ordinary ops. "Native FP8" covers Linear, Conv2d and the two attention matmuls only.
- **AWQ folds.** Scales are folded into `input_layernorm`, `post_attention_layernorm` and `up_proj` rows. Under grouped-query attention
  the `v_proj -> o_proj` fold is not implemented (`o_proj` stays unscaled). Consequence: `up_proj`'s output is scaled by 1/s per channel, so a
  layer output compared with the BF16 model is not comparable for `up_proj`; `quality.py` reports raw and scale-recovered figures for it.
  The existing `--rmse` of `eval.py` (`src/layer_rmse.py`, older than the tile work) has a "local" error that applies the clean layer to the quantized
  layer's input. On the smoke model with `atlas_v1_awq_tile` that local error (mean relative RMSE) is 1.01 for `gate_proj`, 1.27 for `up_proj` and 8.80 for
  `down_proj`, against 0.07, 0.95 and 0.10 cumulative, and the same with `atlas_v1_awq`; treat it as invalid for folded layers. Measured on the
  tiny model only.
- **FP8 range 448** (RTL source); a hardware spec says 240. Panel scales leave far fewer weight codes in the exponent-field-15 range (256 to 448, above 240) than per-channel scales (0.031% vs 0.706%
  of weight codes) but flush more small weights to zero (0.067% vs 0.025%).
- **CPU calibration time** 19 to 26 minutes for TinyLlama on 16 shared cores. Larger models need a GPU or a reduced `--calib-n` (state it).
- `model.pt2` is tied to the torch version it was captured with (2.11.0 CPU); the manifest records it.
- **Uncalibrated Linear in the static recipe.** A Linear that calibration never reached keeps the reference dynamic per-tensor rule (SmolVLA: the unused VLM `lm_head`). It is declared in `policy.json` (`activation.uncalibrated_fallback: true`, `tile_native: false`, top-level `uncalibrated_fallback.count` / `sites` / `declared`), the export warns, and `verify_bundle.py` fails `static_recipe_fully_static` unless the export was run with `--allow-uncalibrated-fallback` (or the verifier with the same flag). Exports made before this change lack the field.
- **Dynamic recipes: pack admission is not checked in `policy.json`.** `compiler_pack_scale_admission.status` is `not_checked_dynamic` (counts null); read `graph_inspection.json` `compiler_pack_scale_admission.dynamic_activation_scales` (capture-case fixtures) and `dyn_admission.py`.
- `verify_bundle.py` classifies non-atlas contractions as declared or unexpected using the policy and the graph's module stack, so DiT / Pi0 bundles no longer fail for their BF16-by-policy convs or the DiT's unquantized `RoPEAttention` SDPA nodes; those are listed, not hidden (`graph.unquantized_attention`). A policy boundary's quality still has to be read from `graph_inspection.json` (section 8).

## 8. Other models

SmolVLA (`lerobot/smolvla_base`, flow-loss boundary), static recipe, run on CPU with the same prefix:

```bash
export SV=/scratch2/lorenhung/atlas-fp8/exports-tile/smolvla/atlas_v1_awq_tile_static
$CAP $PY -u src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --norm-stats dataset \
    --n-frames 2 --seed 0 --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --device cpu \
    --export-dir $SV/export --out $SV/../atlas_v1_awq_tile_static_export_cli.json         # 1266 s
$CAP $PY -u src/fx_export.py --export-dir $SV/export --out-dir $SV/fx --device cpu --native-fp8 \
    --boundary smolvla_flow_loss --n-images 1                                               # 403 s
```

Calibration is the 32 frames of the earlier `sv_E0` export (`data_sha256 b6e8b00d...`, seed 1234, normalization `dataset`, verified).
Result: 317 quantized sites, all tile-native; graph has 302 `atlas.qlinear`, 1 `atlas.qconv2d`, 36 `atlas.qk_matmul`, 36 `atlas.pv_matmul`;
nothing emulated; `bf16_by_policy` empty; capture and fresh-process replay passed. 35 Linear weights have an output size that is not a
multiple of 32 and carry the label `per_panel(32,partial_tail)`. No quality number was measured for SmolVLA.

### Multi-Task DiT and Pi0 (2026-11-06, CPU, commit `7aadbdc`)

Both were run with `--algo atlas_v1_awq_tile_static+keep_bf16:rule=input_conv`. The named recipe `atlas_v1_awq_tile_static` has no `keep_bf16`
member (only `atlas_v1_tile` does), so without the appended member the patch-embedding Conv2d is quantized (SmolVLA above: 1 `qconv2d`). Bundles:
`exports-tile/multitask-dit/atlas_v1_awq_tile_static/` and `exports-tile/pi0/atlas_v1_awq_tile_static/` (each with `export/`, `fx/`,
`verify_bundle.json`, `bundle_manifest.json`, and the capture driver `policy_fx_capture.py` + `pi0_fx.py`).

| | Multi-Task DiT | Pi0 |
|---|---|---|
| export command | `eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --seed 0 --n-frames 200 --modes ref_conv --export-only` (atlas-lerobot env) | `eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --export-only` (atlas-pi0 env, torch 2.7.1) |
| export wall time | 385 s | 2374 s |
| calibration | 32 frames, seed 1234, test episodes excluded (defaults) | 8 DROID frames (default `--calib-n`), seed 0, episodes disjoint from the 32 evaluated ones |
| sites in `policy.json` | 179 Linear + 24 attention quantized, 1 Conv2d host | 420 Linear + 63 attention quantized, 1 Conv2d host, 2 skipped (`lm_head` x2) |
| graph | 179 `qlinear`, 24 `qk_matmul`, 24 `pv_matmul`, 275 `quantize_act`, 0 `qconv2d` | 420 `qlinear`, 63 `qk_matmul`, 63 `pv_matmul`, 672 `quantize_act`, 0 `qconv2d` |
| BF16 by policy | `...clip vision_model.embeddings.patch_embedding` (bf16 `[768,3,16,16]`) | SigLIP `...vision_tower.vision_model.embeddings.patch_embedding` (**float32** `[1152,3,14,14]`, selected by the same rule) and the two skipped `lm_head` (not in the graph) |
| `mode: dynamic` Linear sites | 0 | 0 |
| static activation codes outside 122..128 | 88 of 203 (64 of 179 Linear inputs, 24 of 24 `p`) | 127 of 483 (64 of 420 Linear inputs, 63 of 63 `p`) |
| `verify_bundle.py` | 23 of 24 pass | 22 of 24 pass |
| capture | strict, torch 2.11, 266 s | **non-strict**, torch 2.7.1, 1549 s |

Boundaries are not in `fx_export.py` (`--boundary` has only `causal_lm` and `smolvla_flow_loss`); the driver `policy_fx_capture.py` mirrors the SmolVLA branch.
DiT: `observation_encoder.encode(batch)` + one `noise_predictor` call (state, images, tokens, noisy actions, timestep in; predicted noise out).
Pi0: `PI0Pytorch.sample_actions(num_steps=1)` with explicit noise (SigLIP, PaliGemma prefix with KV cache, one expert denoise step). Not the flow-loss
forward: its joint prefix + suffix path runs attention through the PaliGemma layers only, so the 18 action-expert attention sites had no matmul
in that graph (capture refused it). The driver also stacks the 3 cameras into one SigLIP call (`fx_export` refuses a quantized module called 3 times)
and adds `aten.conv2d.padding` to `fx_export.CONTRACTIONS` at run time (SigLIP's patch embedding exports with that overload, so the BF16-by-policy
report could not find its contraction). `src/` was not modified. Neither graph is the 10-step sampler or the DiT's 100-step loop.

`verify_bundle.py` on these two bundles failed (table row above, as first run) before the declared / unexpected classification was added: `no_unexpected_non_atlas_contraction` failed on the patch-embedding conv (declared BF16 / float32
by policy) and, for the DiT, on the 6 `scaled_dot_product_attention` nodes of the DiT's own `RoPEAttention` blocks: they are functional BF16 attention with
no policy site and are not quantized (the 24 attention sites are the CLIP text and vision towers). Pi0 only:
`graph_has_no_float_weight_param_for_quantized_linears` fails because the non-strict export keeps 420 float weight parameters as meta placeholders
(no storage, no users in the graph; `fresh_process_verification.json`: `unused_float_weights_not_materialized: true`).
After the change the same verifier, run read-only on the same bundles, passes 25 of 25 checks for both (DiT: 7 declared contractions, 1 BF16-by-policy conv and 6 unquantized attention; Pi0: 1 declared conv and 420 tolerated meta placeholders), and still 25 of 25 for TinyLlama `atlas_v1_awq_tile` and `atlas_v1_awq_tile_static` (the extra check is `static_recipe_fully_static`). The bundles' own `policy.json` files were written before `uncalibrated_fallback` existed; their count is derived by the verifier (0).
Pi0 and DiT quality was not measured (no comparable old number; the old DiT and Pi0 exports were never produced).

## 9. Optional steps added after the first bundles (2026-11-06)

All are off by default; none changes a default export (checked byte for byte against the base commit for the pack fold).

### 9.1 `dead_columns=rtn` (GPTQ, any model with ReLU/GELU-gated inputs)
GPTQ zeroes every weight column whose calibration Hessian diagonal is zero. When calibration never excites an input channel, that channel's weights
are lost for good (Pi0 SigLIP `fc2`: 3.5k to 4.2k of 4,304 columns per layer in layers 17 to 25). `dead_columns=rtn` rounds those columns to nearest on
the same exponents and gives no error feedback; live columns are bit-identical to the default. Put it in the `gptq_awq` stage options. The Pi0 candidate recipe:

```
gptq_awq:method=gptq,actorder=true,dead_columns=rtn+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib+keep_bf16:rule=input_conv
```
Pi0 export (atlas-pi0 env, about 35 min on a loaded CPU host):
`python -u src/eval_pi0.py --obs droid --n-obs 32 --seed 0 --modes ref_conv --algo "$R" --export-dir export --export-only --device cpu --openpi-dir ... --checkpoint-dir ... --tokenizer-path ...`
with `R` set to the recipe string above. Why AWQ is off for Pi0: AWQ with panel exponents damages salient producer rows (`tile-pi0-awq-diag.md`). Result and the exact bundle: `pi0-deadrtn.md`.
Check afterwards: count all-zero weight columns per tensor (`scripts/tile_diag/fc2_export_compare.py`).

### 9.2 Pack-range fold (`--pack-fold`)
The compiler's `lowerPack` accepts activation codes 122..128. The fold moves out-of-range static exponents into range by an exact power-of-two change in the producer
(RMSNorm/LayerNorm/`(1+w)` weights, `up_proj`, `pv_pop_shift` for P.V into `o_proj`) and compensates in the consumer, so FP8 bytes and the pop constant are unchanged.
Post-hoc on an existing export, no re-calibration:
```bash
python src/pack_fold.py --export-dir <E>/export --fx-dir <E>/fx --out-dir <F>/export
python src/fx_export.py --export-dir <F>/export --out-dir <F>/fx --native-fp8 --seqlen 64       # re-capture
python scripts/tile_flow/verify_bundle.py <F>/export <F>/fx <F>/verify_bundle.json
python scripts/tile_flow/compare_fx_logits.py <E>/fx <F>/fx <F>/compare_fx_logits.json --random 6
python scripts/tile_flow/make_bundle_manifest.py <F>
```
TinyLlama: 114 of 176 sites outside the range becomes 22 (attention `p`), 0 of 16.4M logits differ. Needs the compiler importer to accept the new trailing `pop_shift`
argument of `atlas.pv_matmul`. Refused sites (shared producers, BF16 `(1+w)`, dynamic consumers, shared attention sites) are listed per site in `policy.json` (`pack_fold`). Details: `pack-fold.md`.

### 9.3 Per-phase graphs for iterative policies (DiT, Pi0)
Diffusion and flow policies loop; a single graph cannot hold the loop. The default taken: one graph per phase and a host-owned loop described by `loop_manifest.json`
(DiT: `encode` once, `step` 100 times; Pi0: `prefix` once, `step` 10 times). Build the manifest with `scripts/tile_flow/make_loop_manifest.py dit|pi0 --single-step-bundle DIR --out PATH`,
capture the phases with `scripts/tile_flow/policy_fx_capture_phases.py` (DiT) or `scripts/tile_flow/pi0_phase_capture.py` (Pi0), verify each phase with `verify_bundle.py`,
and compare the composition with the single-step graph using `scripts/tile_flow/compose_phases.py`. A quantized module that a graph calls N times is now allowed (N node sets, each with its own
`quantize_act`); one that is never called is recorded as absent. Exact commands and expected numbers: `phase-graphs.md`. The Pi0 10-step loop against eager is not finished.
