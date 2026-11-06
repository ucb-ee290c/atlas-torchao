# TinyLlama ablation: AWQ, granularity, activation mode (2026-11-06, CPU)

Every number below comes from a command run on this branch or from a result file named next to it. TinyLlama-1.1B-Chat only; one model, one calibration
set (16 WikiText-2 train windows of 2048 tokens, seed 1234, slow tokenizer, `data_sha256 07428620...`), so these are measurements, not general claims.
Perplexity: WikiText-2 test, 16 windows of 512 tokens, exact fp32 accumulation (`ref_conv`). Logits rel. L2: final logits against BF16 on the fixed 2 x 256-token batch
(`scripts/tile_flow/quality.py`). Commands follow `howto-model-to-fx-graph.md`.

## Part 2. Results table

| Export | Weights | AWQ | Activations | Perplexity (BF16 12.7558) | Logits rel. L2 | Top-1 agree | Source |
|---|---|---|---|---|---|---|---|
| BF16 | | | | 12.7558 | 0 | 1 | `tile_static/ppl_w16x512.json` |
| tile static, AWQ + GPTQ | panel(32) | on | static | 12.8980 | 0.0536 | 0.951 | `exports-tile/tinyllama/atlas_v1_awq_tile_static/` |
| tile static, GPTQ only (this work) | panel(32) | off | static | 12.9008 | 0.0646 | 0.922 | `exports-tile-diag/tinyllama/gptq_panel_static_noawq/` |
| per-channel static, AWQ | channel | on | static | not run (see below) | | | |
| tile dynamic, AWQ + GPTQ | panel(32) | on | dynamic per 32-token block | 12.8287 | 0.0522 | 0.934 | `exports-tile/tinyllama/atlas_v1_awq_tile/` |
| `tl_E0` (`atlas_v1_awq`) | channel | on | dynamic per token | 12.8451 | 0.0579 | 0.936 | `exports-tile/tinyllama/atlas_v1_awq_baseline_tl_E0/` |

Reading (controlled AWQ comparison): removing AWQ changes perplexity from 12.8980 to 12.9008 (+0.0028) and logits rel. L2 from 0.0536 to 0.0646. One calibration set and one 16-window perplexity sample; no confidence interval was computed, so small differences are not resolved.

Recipe of the new export (same string as the Pi0 diagnostic, `tile-pi0-awq-diag.md`, without the Pi0-only `keep_bf16`):
`gptq_awq:method=gptq,actorder=true+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib`, exported with
`src/eval.py --modes ref_conv --export-only --device cpu`, default calibration (16, seed 1234, seqlen 2048). It is the tile static recipe with
`method=awq_gptq` replaced by `method=gptq`; nothing else differs.

### Which comparisons are controlled

- **Tile static AWQ+GPTQ vs tile static GPTQ-only: controlled for AWQ** (same granularity, same static activations, same calibration windows, same CPU host class).
  The tile static run was made at commit 53ab820; the GPTQ-only export was started at 12:34 with the tree at 79628ee. `git diff 53ab820 79628ee -- src` touches `eval.py`, `eval_pi0.py`,
  `eval_policy.py`, `quant_export.py`, `tile_native.py` (declared uncalibrated fallback, dynamic pack admission reporting, Pi0-safe manifest); no file under `src/algos/` changed.
  Caveats: (1) the tile static export was not rerun at 79628ee, so "same code" rests on that diff, not on a bit-identical rerun; (2) other sessions merged changes into this
  worktree while the run was going (`src/ref_attention.py`, `src/quant_export.py`, `src/fx_export.py`, `src/pack_plan.py`, merges at 12:51 and 13:09); modules already imported at 12:34 are
  unaffected, but a lazily imported changed module, and the perplexity / quality runs started later, would see the newer files. I did not diff numerics for those.
- **Tile static vs tile dynamic: controlled for activation mode.** Same commit, same calibration, byte-identical weight tensors (howto section 1, checked tensor by tensor).
  Attention differs too (static `p`, dynamic per-block q/p), so "activation mode" here means the whole policy, not only the Linear inputs.
- **Tile vs `tl_E0`: confounded.** Granularity, activation scale (per token vs per block or static), calibration host (GPU vs CPU), code version (not recorded for `tl_E0`) and
  the AWQ alpha picks all differ; 6 of 66 fold groups picked a different alpha (Part 1). Do not read that gap as a granularity effect.
- **Granularity alone (per-channel, static activations, same calibration): not run.** The recipe would be
  `gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=defer,attn=defer,rule=nc+static_calib` (it parses; not exported). I dropped it for time: the
  GPTQ-only export alone took over 40 minutes on the loaded host (load average 140 to 270). So no controlled granularity number exists here.


## Part 1. Is there Pi0-style AWQ row damage on TinyLlama? (weights only, CPU)

Compared: tile static export (`atlas_v1_awq_tile_static`, panel, commit 53ab820) against the per-channel `tl_E0` export
(`exports-ready/tinyllama/tl_E0/model.safetensors`, GPU-calibrated, code not recorded), both against the stock BF16 checkpoint.
Scripts: copies of `scripts/tile_diag/{mlp_composite,fold_groups}.py` plus one new `rows.py`, in
`/scratch2/lorenhung/atlas-fp8/exports-tile-diag/tinyllama/results/scripts/`; outputs `part1_*.json` next to them.
`weights_only.py` was started and stopped (too slow on the loaded host); no number from it is used.

One adaptation was required. On TinyLlama the AWQ input-side scale is folded into the RMSNorm weight and multiplied into the columns of
`gate_proj`/`up_proj` (and q/k/v), which the Pi0 scripts do not undo. Without the undo the up_proj error is 0.13 to 10 for both exports
alike. The copies multiply each dequantized consumer weight by `c = norm_export / norm_stock` (the norm weights are stored in BF16, so
`c` carries BF16 rounding).

**AWQ alpha is not identical across the two exports.** Of 66 fold groups, 6 differ: post_attention_layernorm L0 (0.4 vs 0.7), L13 (0.75 vs 0.35), L21 (0.95 vs 0.6);
input_layernorm L13 (0.5 vs 0.3); up_proj->down_proj L12 (0.45 vs 0.8), L14 (0.25 vs 0.85). The other 60 are equal, including up->down in 20 of 22 layers.
Same data, same seed, but a different device, code and (for panel) grid in the search, so the alphas can legitimately differ.
Layers 0, 12, 13, 14, 21 are therefore not matched comparisons; 17 layers are.

### Basis-independent composite, `||Dd Du - Wd Wu|| / ||Wd Wu||` (gate nonlinearity ignored)

| | tile static | per-channel tl_E0 |
|---|---|---|
| median over 22 layers (excluding L5) | 0.0449 | 0.0450 |
| ratio tile / channel, 17 alpha-matched layers | 0.995 (L11) to 1.007 (L6) excluding L5; L5 1.040 | |
| ratio tile / channel, mismatched layers | L0 0.560, L12 0.981, L13 1.000, L14 1.004, L21 0.999 | |

Per layer (tile / channel): L0 0.0505 / 0.0902, L11 0.1095 / 0.1100, L12 0.0741 / 0.0756, L10 0.0650 / 0.0646, L6 0.0792 / 0.0786,
L5 1.656 / 1.592. All other layers are 0.038 to 0.049 for both exports.

L5 is large for both exports at an identical level, so it is not panel damage. It is most likely an artifact of this weight-space check,
not a quantization error: the undo multiplies each column's error by `c`, which spans 0.020 to 51 in L5's post-attention norm and
the metric is not activation weighted. The activation-based cumulative per-layer errors already in the two `quality.json` files do not single L5 out
(down_proj output rel. L2 0.128 tile static / 0.124 per-channel; scale-recovered up_proj 0.0996 / 0.0946). The L5 cause was not isolated.

### Fold scale range and within-panel row spread (tile static export, up_proj in the quantized basis)

| measure | TinyLlama | Pi0 (docs/algos/tile-pi0-awq-diag.md) |
|---|---|---|
| log2 range of the up->down fold scale s over the 5632 channels, max over layers | 4.87 binades (L8); 14 of 22 layers below 2 | 19 binades (Gemma L0) |
| median within-32-row-panel max/min row-amax ratio of folded up_proj, max over layers | 6.0 (L0) | rows about 12 binades below their panel amax (53 rows) |
| p95 of the same ratio, max over layers | 9.9 (L0), next 9.6 (L21), 8.2 (L11) | |

The same quantity for the per-channel export (what a panel would see in that basis) is 10.9 median / 17.6 p95 at L0 and at most 8.2 p95 (L11) elsewhere.

### Up_proj rows, panel vs per-channel (`rows.py`)

Effective row = dequantized row x column undo x s_j, with s_j taken from the down_proj column (median |Dd/Wd|, independent of how the up rows were grouped).
Row error = `||eff_j - Wu_j|| / ||Wu_j||`.

| layer | alpha (up->down) tile / channel | rows with error > 10%: tile / channel | rows where tile and channel effective rows differ by > 10% of ||Wu_j|| |
|---|---|---|---|
| 0 | 0.7 / 0.7 (post-attn norm 0.4 / 0.7) | 0.04% / 39.0% | 49.2% |
| 5 | 0.6 / 0.6 | 99.8% / 99.9% | 99.8% |
| 6 | 0.45 / 0.45 | 21.0% / 21.4% | 17.5% |
| 10 | 0.85 / 0.85 | 11.2% / 10.9% | 8.8% |
| 11 | 0.3 / 0.3 | 39.1% / 40.5% | 23.9% |
| 12 | 0.45 / 0.8 | 17.7% / 17.7% | 36.1% |
| 4 | 0.95 / 0.95 | 0.02% / 0.04% | 0.09% |
| every other layer | | 0.00% / 0.00% | 0 to 0.04% |

Where a layer has many bad rows (L5, 6, 10, 11, 12) the per-channel export has as many, so these are the undo artifact above, not panel grouping. The only layer where the
two exports differ materially is L0, and there the tile export is the better one (alpha for the post-attention norm differs there, so L0 is not a matched pair).
The 1% of rows with the largest s_j (the Pi0 mechanism) have a mean error of 0.031 to 0.077 on the tile export (L5 3.06 and L11 0.207 aside, same artifact on both exports; L0 channel 0.127 against tile 0.045), i.e. no sign of the large-s rows being flushed under a shared panel exponent.

**Finding (Part 1).** TinyLlama does not show the Pi0 layer-0 pattern. At alpha-matched layers the panel and per-channel composite errors agree within -0.5% to +0.7% (L5 +4%
under the artifact above), the AWQ fold range is at most 4.9 binades (Pi0: 19), and the within-panel row-amax spread in the folded basis is at most about 10x (Pi0: about 12 binades = 4096x).
This is a weights-only result; the composite ignores activations and the gate nonlinearity.

## Reproduce

Part 1 (about 3 minutes each, 2 threads): `rows.py`, `fold_groups.py`, `mlp_composite.py` from
`/scratch2/lorenhung/atlas-fp8/exports-tile-diag/tinyllama/results/scripts/` (run from that directory; they import `ref_numerics` from this worktree's `src/`).
Part 2: the export command above with `--export-dir /scratch2/lorenhung/atlas-fp8/exports-tile-diag/tinyllama/gptq_panel_static_noawq/export`, then the two commands of howto section 6
(`quality.py ... 256 2 label=export=out.json`, `src/eval.py --seqlen 512 --max-windows 16 --from-export ... --modes ref_conv`).
Raw Part 1 outputs: `.../results/part1_composite.json`, `part1_fold_groups.json`, `part1_up_rows.json`.
