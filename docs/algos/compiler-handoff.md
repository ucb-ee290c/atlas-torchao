# Compiler handoff index (tile-aware native FP8 FX graphs)

Status: prepared 2026-11-06 for the compilers team. Nothing here has been reviewed or accepted by them. Branch `tile-aware`, PR https://github.com/ucb-ee290c/atlas-torchao/pull/5.

## What a bundle is
Each bundle directory has `export/` (model.safetensors, policy.json, manifest.json) and `fx/` (`model.pt2` ExportedProgram, `graph.txt`, `graph_inspection.json`, `policy.json`, `fixtures.safetensors`, replay manifest). Contractions are `atlas::qlinear`, `atlas::qconv2d`, `atlas::qk_matmul`, `atlas::pv_matmul`, each fed by `atlas::quantize_act`. Weights are E4M3 codes (uint8) with an int32 exponent per 32 output channels over all K; activation exponents are per 32-token block or one static value per site. Every scale is constant along K, so the pop constant is `2^(sx+sw)` and is applied once at `vmatpop`. Contract: `docs/algos/tile-aware.md`. Independent check: `python scripts/tile_flow/verify_bundle.py EXPORT FX OUT.json` (25 checks).

## Bundles (root `/scratch2/lorenhung/atlas-fp8/exports-tile/`)
| Model | Directory | Boundary | Verifier | Notes |
|---|---|---|---|---|
| TinyLlama | `tinyllama/atlas_v1_awq_tile_static`, `tinyllama/atlas_v1_awq_tile` | full causal LM forward, batch 1, seqlen 64 | 25 of 25 | 154 qlinear, 22 qk and pv; `lm_head` BF16 |
| SmolVLA | `smolvla/atlas_v1_awq_tile_static` | flow-loss forward | 25 of 25 | 302 qlinear, 1 qconv2d, 36 qk and pv |
| Multi-Task DiT | `multitask-dit/atlas_v1_awq_tile_static_attn` | encoder plus ONE noise-predictor call | 25 of 25 | 179 qlinear, 30 qk and pv (24 CLIP, 6 DiT, 24-token attention is a partial tile) |
| Multi-Task DiT phases | `multitask-dit/atlas_v1_awq_tile_static_attn_phases/{encode,step}` + `loop_manifest.json` | `encode` once, `step` 100 times | 25 of 25 each | host owns the DDPM update, noise, clipping |
| Pi0 | `pi0/gptq_panel_static_noawq_deadrtn` | one sampler step (non-strict, torch 2.7.1) | 25 of 25 | 420 qlinear, 63 qk and pv; candidate recipe |
| Pi0 phases | `pi0/gptq_panel_static_noawq_phases/{prefix,step}` + `loop_manifest.json` | `prefix` once, `step` 10 times | 25 of 25 each | host owns the Euler loop and K/V lifetime |
| TinyLlama, folded | `/scratch2/lorenhung/atlas-fp8/exports-tile-fold/tinyllama/atlas_v1_awq_tile_static_packfold` | as TinyLlama | 25 of 25 | opt-in exact pack fold, needs `pop_shift` |

## Equivalence evidence (all in-process, reference emulation, no RTL)
- DiT: `step(encode(obs))` equals the single-step bundle; the 100-step host loop equals eager `conditional_sample`, max abs 0.0.
- Pi0: composed phases equal the single-step graph; the 10-step host loop equals eager `sample_actions(num_steps=10)`, 0 of 320 elements differ on both cases (one process, 8 threads, a consistency check, not accuracy).
- TinyLlama fold: 16.4M logits, 0 differ between folded and unfolded.

## What the compilers team needs to confirm (nothing below is decided)
1. **Loop ownership.** Delivered: per-phase graphs plus a host-owned loop (`loop_manifest.json`). An unrolled whole-inference graph is NOT built; say if you need one.
2. **`pop_shift`.** `atlas.pv_matmul` gains a trailing `pop_shift` (default 0), emitted only by folded bundles. Unfolded bundles do not use it.
3. **Pack range.** `lowerPack` admits E8M0 codes 122..128. Static sites outside: TinyLlama 114 of 176, SmolVLA 244 of 316, DiT 88 of 203, Pi0 125 of 483. Unfolded bundles contain them; folding (opt-in) reduces TinyLlama to 22 (all attention P). Widen the range, accept folds, or use the vmul fallback.
4. **Attention q, k, v remain dynamic** per (batch, head); only P is static. Dynamic scales are not compile-time constants.
5. **Partial tiles.** DiT attention has 24 tokens; Linear weights with output size not a multiple of 32 carry the label `per_panel(32,partial_tail)`.
6. **Non-strict Pi0 export** keeps unused lifted buffers (4.1 to 4.3 GB `model.pt2`); ignore inputs that no node reads (`graph_inspection.json`, `unused_quantized_modules`).
7. **Recorded fixtures.** `fixtures.safetensors` of the older single-step DiT and Pi0 bundles do not reproduce bit-exactly on the build host (DiT case 0 max abs 0.0234; Pi0 0.04 to 0.06, and Pi0 output varies with thread count). Do not use them as bit-exact references; `phase-graphs.md` section "Fixture reproducibility finding".

Not run: RTL, hardware, closed-loop policy rollouts.
