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
$CAP $PY scripts/tile_flow/verify_bundle.py $OUT/export $OUT/fx $OUT/verify_bundle.json     # 8 s, 24 checks, exit 0 = all pass
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
| 6 | graph holds exactly one `atlas.qlinear` per quantized Linear, one `atlas.qk_matmul` and one `atlas.pv_matmul` per attention site | `verify_bundle.py` reads `model.pt2` |
| 7 | the only non-atlas contractions are `lm_head` (BF16 by policy) and the float32 RoPE `inv_freq x position_ids` outer product | `verify_bundle.py` (operand dtypes from the graph) |
| 8 | weight buffers in the loaded program are uint8 / int32; no float weight parameter for a quantized layer | `verify_bundle.py` |
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
- **Uncalibrated Linear in the static recipe.** A Linear that calibration never reached keeps the reference dynamic per-tensor rule (SmolVLA: the unused VLM `lm_head`, `policy.json` activation `source: reference`), so the policy is not 100% static. Count `mode: dynamic` Linear sites in `policy.json` before calling a bundle static.
- `verify_bundle.py` is written for causal LMs (it names `lm_head` and the RoPE outer product as the allowed non-atlas contractions); for a policy boundary read `graph_inspection.json` directly (section 8).

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

Multi-Task DiT and Pi0 were not run. Present on this machine: the DiT checkpoint (`TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000`, 954 MB) and
its dataset snapshot (97 MB) in the Hugging Face cache; the Pi0 checkpoint (`/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris`, 7.0 GB)
and a separate `atlas-pi0` environment (torch 2.7.1). The DiT conv exception needs `atlas_v1_tile`; DiT and Pi0 action sampling are not FX boundaries
(fx-export.md).
