# Pi0 tile export: Gemma layer-9 q-input drift, isolation (2026-11-06, CPU)

Measured with `scripts/tile_diag/*` (results in `/scratch2/lorenhung/atlas-fp8/exports-tile-diag/pi0/results/`). Drift = rel L2 of the Gemma layer-L
q_proj input of the stock Pi0 model carrying dequantized export weights (no activation quantization) against the stock capture (DROID ep 252
frame 30, noise seed 0), `hybrid_drift.py`.

| weights loaded | L9 | L17 |
|---|---|---|
| old `pi0_atlas_v1_awq` (GPU, commit 54e1855, channel) | 0.0252 | 0.0223 |
| tile static export (CPU, 7aadbdc, panel) | 0.1273 | 0.0852 |
| control: `atlas_v1_awq` channel, CPU, same commit/calibration | 0.0240 | 0.0231 |
| tile, Gemma layer 0 up_proj taken from the control | 0.0230 | 0.0228 |
| tile, Gemma layer 0 down_proj taken from the control | 0.1270 | 0.0849 |
| old, Gemma layer 0 mlp taken from the tile export | 0.1282 | 0.0858 |
| panel, GPTQ only (`gptq_awq:method=gptq`, no AWQ), CPU | 0.0237 | 0.0207 |

Cause: Gemma layer 0 AWQ up_proj -> down_proj pair, alpha 0.8, s spanning 19 binades. up_proj rows are divided by s_j; the 53 rows with the largest s_j
(median log2 s 2.9 against -8.4 for all rows) sit about 12 binades below the amax of their 32-row panel in the folded basis and are flushed or
badly rounded under the shared panel exponent (per-channel exponents absorb this). Those are exactly the salient channels AWQ amplifies on the
down_proj side. `awq_search` (src/algos/gptq_awq.py, `Ws = [mods[c].weight for c in cons]` in `prepare`) scores only the consumers (down_proj);
the producer's quantization on the panel grid is not in the loss.
