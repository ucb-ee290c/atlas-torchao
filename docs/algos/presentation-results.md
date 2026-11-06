# Tile-aware quantization: results summary for presentation

Status date: 2026-11-06. Branch `tile-aware` (PR https://github.com/ucb-ee290c/atlas-torchao/pull/5). Every number below names its source. Items marked PENDING were still being produced when this was written; replace them or delete them before presenting.

## 1. What changed

| | Before (`atlas_v1_awq`) | Tile-aware |
|---|---|---|
| Weight exponent | one per output channel | one per 32 output channels, constant over all of K |
| Activation exponent | one per token, run-time | one per 32-token block, or one static value per site |
| Applied by hardware | not natively: needs extra vector multiplies around the matrix multiply | once, at pop: weight and activation exponents fold into one scale |
| Exponent entries (TinyLlama) | 394,240 | 12,320 |

Why: the matrix unit multiplies unscaled E4M3 and accumulates in BF16 across K tiles; a power-of-two scale can only be applied when a tile is popped, packed or unpacked, so only exponents constant along K are hardware-native. (`docs/algos/tile-aware.md`)

## 2. Deliverables

- Flow: `w=panel`, `a=block`, `attn=block`; recipes `atlas_v1_awq_tile`, `atlas_v1_tile`, `atlas_v1_awq_tile_static`.
- Verifier for the tile contract (export and replay) and an independent bundle checker (`scripts/tile_flow/verify_bundle.py`, 25 checks).
- How-to from any model to the FX graph: `docs/algos/howto-model-to-fx-graph.md`; artifact page https://claude.ai/artifact/CXCiae1VpYf6UsqsoD7QJp (private).
- Verified native FP8 FX bundles under `/scratch2/lorenhung/atlas-fp8/exports-tile/`: TinyLlama (2 recipes), SmolVLA, Multi-Task DiT, Pi0.

## 3. Verified results

### Native FP8 graph (source: each bundle's `verify_bundle.json` and `fx/graph_inspection.json`)

| Model | qlinear | qconv2d | qk / pv matmul each | Dynamic Linear sites | Checker |
|---|---|---|---|---|---|
| TinyLlama (static and dynamic) | 154 | 0 | 22 | 0 / 154 | 25 of 25 |
| SmolVLA (static) | 302 | 1 | 36 | 1 (`vlm.lm_head`, never called in the captured boundary) | 25 of 25 after the verifier was made correct for SmolVLA (`docs/algos/smolvla-verify.md`) |
| Multi-Task DiT (static) | 179 | 0 | 24 | 0 | 25 of 25 after the verifier fix |
| Multi-Task DiT with own attention quantized (`atlas_v1_awq_tile_static_attn`) | 179 | 0 | 30 (24 CLIP + 6 DiT) | 0 | 25 of 25, `unquantized_attention` 0 |
| Pi0 (static) | 420 | 0 | 63 | 0 | 25 of 25 after the verifier fix |

All contractions are native FP8 operations except declared exceptions: `lm_head`, patch-embedding convolutions kept BF16 by `keep_bf16:rule=input_conv`, and the RoPE outer product. DiT's own six attention blocks were plain BF16 in this bundle (see 5).

### Quality, TinyLlama (sources: `exports-tile/tinyllama/*/ppl_w16x512.json`, `quality.json`)

| Export | Perplexity (BF16 12.756) | Final-logits relative L2 | Logit KL | Top-1 agreement |
|---|---|---|---|---|
| Previous `atlas_v1_awq` (tl_E0, GPU-calibrated, unrecorded code) | 12.845 | 0.0579 | 0.0185 | 0.936 |
| Tile static | 12.898 | 0.0536 | 0.0128 | 0.951 |
| Tile dynamic | 12.829 | 0.0522 | 0.0121 | 0.934 |
| Tile static, GPTQ only (AWQ off) | 12.901 | pending | pending | pending |

The previous export has a different calibration provenance, so the comparison is not controlled. Controlled for AWQ: turning AWQ off at panel granularity with static activations changes perplexity from 12.8980 to 12.9008 (one calibration set, 16 windows, no confidence interval), so on TinyLlama the tile result barely depends on AWQ. Static versus dynamic is controlled (same commit, identical weights). Granularity alone has no controlled number (the per-channel static control was not run; each export takes about 45 minutes on the loaded host). Tile versus the previous export stays confounded (`docs/algos/tinyllama-ablation.md`).

Accumulation model on nine isolated layers (static export): relative L2 against BF16 is 0.034 to 0.095 for exact float32, 0.035 to 0.096 for MXU1 (`ipt`), 0.047 to 0.117 for MXU0 (`sa`). No RTL run.

### Metadata size (source: `tile_vs_nontile_comparison.json`)
12,320 groups versus 394,240 rows. Stored bytes are identical today (int32 `[out,1]`, 1,576,960 B); compacted to one E8M0 byte per group it would be 12,320 B.

## 4. Findings

1. **AWQ with panel exponents damages salient rows** (Pi0). Gemma layer-9 q-input relative L2: 0.1273 (tile export) versus 0.0252 (previous export). Swapping the one Gemma layer-0 `up_proj` tensor from a per-channel export gives 0.0230. About 53 of 16384 rows are damaged. Ruled out: missing norm fold, activations, calibration frames, shape bugs. (`docs/algos/tile-pi0-awq-diag.md`, commit 57298dd)
   - Mitigation, tile-native: GPTQ on the panel grid with AWQ off gives layer 9 at 0.0237 and layer 17 at 0.0207 (old per-channel export 0.0252 and 0.0223). Full bundle `exports-tile/pi0/gptq_panel_static_noawq/`: 420 `qlinear`, 63 attention pairs, checker 25 of 25, fresh-process replay passed, 124 of 483 static sites outside 122..128 (`docs/algos/pi0-gptq-panel-noawq.md`, commit a349620). Drift only; no end-to-end action quality.
   - Fix: producer-aware AWQ search (option `producer_aware=true`, default off, bit-identical when off; `docs/algos/awq-producer-aware.md`). Adds the producer's panel-grid quantization error to the AWQ loss. Weights-only proxy on Pi0 Gemma layer 0: damaged producer rows 4 (old search) to 0 (new), composite error 0.605 to 0.028. 10 new tests (9 pass, 1 skips without submodules). NOT validated end to end: no Pi0 export with the option and no layer-9 drift measurement; the proxy search does not reproduce the recorded alphas.
   - TinyLlama does not show the damage (weights-only scan, `docs/algos/tinyllama-ablation.md`): AWQ fold scale range at most 4.87 binades against 19 for Pi0, within-panel row-amax spread of folded `up_proj` at most 6.0x median and 9.9x at the 95th percentile (Pi0 rows sat about 12 binades below their panel maximum), composite error median 0.0449 tile versus 0.0450 per-channel. The producer-aware search showed no regression on TinyLlama layers 0 and 1 (MLP error -0.7% and +0.4%).
2. **SigLIP `mlp.fc2` weights: the apparent 97% flush is dead columns zeroed by GPTQ, not E4M3 flush** (Pi0, both the old per-channel and the new export; `docs/algos/siglip-fc2-flush.md`, commit a6cd07e). In layer 23, 4177 of 4304 input columns are entirely zero (0.9706 of the codes) because `src/algos/gptq_awq.py:107-109` zeroes any column whose calibration Hessian diagonal is zero, and the GELU output into those channels was exactly zero on the calibration frames. Round-to-nearest on the same exponents flushes 0.035% and gives weight error 0.0266, so there are no outlier weights and the `mse` rule, smaller `rows` and per-row exponents change nothing. On one real DROID frame the layer 23 output error is 0.0244 for the export (RTN 0.0254); on Gaussian inputs that excite every column it is 0.85 to 0.98 versus 0.0266. The alive channel set varies by frame (72 per image, 127 over the 8 calibration frames), so a new frame can hit a zeroed column. Actions not measured, one frame only. Fix implemented as an opt-in option `dead_columns=rtn` (default `zero`, bit-identical when off; `docs/algos/gptq-dead-columns.md`, commit 7f5d9a3, 13 new tests, 115 passed across the six affected test files): dead columns are cast to nearest on the same exponents and give no error feedback, live columns are bit-identical to `zero`. Validated on SigLIP layer 23 `fc2` only (one frame; the export's own 8-frame Hessian was not reproduced): weight error 0.0411 against 0.9894, Gaussian output error 0.0407 against 0.9891, and on a held-out image the output error is 0.0226 against 0.1156 for `zero`. Not validated: no Pi0 re-export, no action quality. Alternative: keep layers 17 to 25 `fc2` in BF16 (10.9% of SigLIP linear multiply-accumulates) with `mixed_precision:units=...,keep_as=bf16`. The same mechanism zeroes 24 of 32 `state_proj` columns (padding), which is harmless.
3. **Compiler pack range 122..128.** Static activation sites outside the range: TinyLlama 114 of 176, SmolVLA 244 of 316, DiT 88 of 203, Pi0 127 of 483 (`docs/algos/pack-range-options.md`, `pack-plan-census.md`). A power-of-two fold into the producer is exact: 0 differing FP8 bytes over 116,654,080 elements at the 92 out-of-range TinyLlama Linear inputs. Clamping instead changes quantization (TinyLlama perplexity 12.8756 clamp all, 12.8609 Linear only, 12.9027 `p` only, 12.8980 unclamped). Sites where the fold costs no added instruction: 92 / 221 / 24 / 61. Attention `p` needs one multiply per head and 32-row block. Exporter now records a per-site plan (`pack_plan` in `policy.json`, informational; `--require-pack-range` to enforce). Not run on RTL or hardware.
4. **Reporting defects fixed** (commit 09ef1df): uncalibrated Linears under a static recipe are declared and not marked tile-native; dynamic admission reads `not_checked_dynamic`; declared-versus-unexpected contractions in the verifier; Pi0 manifest crash fixed.
5. **Hardware findings from the peer harness** (not measured by us): variants `c` (one scalar per 32-column tile) +0.02% to +0.18% cycles, `cb` (one scalar per 32x32 output tile) +0.23% (`/scratch2/lorenhung/atlas-bringup/work/asis-layers/tile/results.md`).

## 5. Limits

- DiT and Pi0 graphs are partial: DiT is one noise-predictor call; Pi0 is a one-step sampler captured non-strict under torch 2.7.1 with three cameras stacked into one SigLIP call (bitwise equal to three calls on a tiny Pi0 and a real-width two-layer SigLIP; real checkpoint not run). No Pi0 or DiT boundary spec exists from the compilers team. (`docs/algos/dit-pi0-full-graph-plan.md`)
- DiT's own attention: the six `RoPEAttention` blocks are now attention sites (7 new tests pass; 77 passed and 16 skipped with the tile and native tests). Re-exported bundle `exports-tile/multitask-dit/atlas_v1_awq_tile_static_attn/`: 30 qk and 30 pv pairs, 209 of 209 sites tile-native, checker 25 of 25, strict capture and fresh-process replay passed (`docs/algos/dit-own-attention.md`). Static sites outside 122..128: 94 of 209 (the 6 new `p` sites at code 119 added 6). Attention matrices have 24 tokens, a partial tile; the compiler must confirm tail-tile handling. No DiT quality number, dropout not modelled.
- No quality numbers for SmolVLA, DiT or Pi0.
- No hardware equivalence claim: exact float32 accumulation plus `sa`/`ipt` models on nine isolated TinyLlama layers; no RTL run by us.
- Full suite on the merged branch before the last two changes (`dead_columns` option, SmolVLA verifier): 870 passed, 35 skipped, 10 deselected, 0 failed. Rerun on the final branch (commit 154e08c, includes the `dead_columns` option and the SmolVLA verifier): 889 passed, 35 skipped, 10 deselected, 0 failed, in 22 minutes 19 seconds.

## 6. Decisions needed (full list in `/scratch2/lorenhung/atlas-fp8/tile-aware-checkin.md`)

1. Compiler owner: widen the pack range, or accept exporter folds (recommended order: exact folds, compiler expansion as fallback, widening after RTL tests).
2. Pi0: ship with AWQ off now, or wait for the producer-aware AWQ fix.
3. DiT and Pi0 full graphs: about three days of work for the blocking items; confirm what boundary the compilers team wants.
4. Matrix unit for projections (MXU0 versus MXU1), `nc` versus reference rule, bias handling, `lm_head`, BF16 opt-out list, `panel` naming.
5. Hardware team: rescale path (H1), exponent derivation on chip (H2), K-varying scales (H3), pop saturation hole, FP8 maximum 448 versus 240.
