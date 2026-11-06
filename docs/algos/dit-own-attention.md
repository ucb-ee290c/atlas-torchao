# Multi-Task DiT: its own attention blocks as attention sites

Branch `tile-dit-attn`, on the tile-aware tip `79628ee`. Plan: `docs/algos/dit-pi0-full-graph-plan.md` section 3 and item 1.

## What changed

`src/ref_attention.py`: `patch_attention` has a third loop. Every module whose class name is `RoPEAttention` and that has `qkv_proj` and `rope`
gets `_atlas_ref_attn = {"fm_qk", "fm_av", "variant": "vision", **algo cfg}` (so `quant_export.attention_sites` and `fx_export.native_attention`
discover it with no other registry) and an instance `forward` (`_dit_rope_attention`). The forward is the lerobot one (fused `qkv_proj`, reshape to
`[B, H, T, D]`, plain BF16 RoPE on q and k, `out_proj`) with the `F.scaled_dot_product_attention` call replaced by
`ref_attention(q, k, v, self.scale, None, ...)`: no mask, not causal, scale `head_dim ** -0.5`. Q and K are quantized after the rotation.
Variant `vision` (scores to BF16, softmax in float32, P to BF16), the variant the reference `mtdit_inout` uses for the DiT SDPA.
Dropout is not modelled (the original passes `dropout_p = 0` in eval). No global monkeypatch, no hook, no `fx_export.py` change.

`native_attention` is all-or-nothing (`fx_export.py`): a model whose `RoPEAttention` modules carry `_atlas_ref_attn` and whose policy has no sites for them
returns `None` for the whole model, so all 24 CLIP sites fall back to the decode path. The bundle therefore has to be re-exported (new policy sites,
new static `p` exponents from calibration), not patched.

## Tests (`tests/test_dit_rope_attention.py`, 7 tests)

Real lerobot `RoPEAttention` (hidden 128, 2 heads, head_dim 64, T = 24), two blocks.
* site discovered with kind `RoPEAttention`, variant `vision`, instance `forward`; an unpatched instance is untouched;
* equivalence: with `ref_attention._cast` / `_mm` replaced by identity and a float matmul, the patched forward matches the original forward
  (BF16 model, 3 seeds, max abs error below 4 % of the output's max; float32 module, below 3 %). What remains is the vision routine's BF16 rounding of scores and P;
* the casts are applied to q, k, p, v in that order, on `[B,H,T,64]` / `[B,H,T,T]`, and the quantized output differs from BF16 within 25 % of its max;
* with the tile recipe's `attn=block` rule, `act_cast` is called for `attention.q|k|p|v` under the module paths and the returned scales are per (batch, head) at T = 24;
* a native FX capture of the two-block model passes, outputs identical to the decode graph, 2 `atlas.qk_matmul` + 2 `atlas.pv_matmul`, 4 `quantize_act` per module,
  no `scaled_dot_product_attention` node, nothing emulated.

## Re-export and re-capture (CPU, atlas-lerobot env, torch 2.11)

Bundle: `/scratch2/lorenhung/atlas-fp8/exports-tile/multitask-dit/atlas_v1_awq_tile_static_attn/` (`export/`, `fx/`, `verify_bundle.json`, `bundle_manifest.json`,
`policy_fx_capture.py` copied unchanged from the sibling `atlas_v1_awq_tile_static`, `export.log`, `capture.log`, `tests.log`). The sibling was not touched.

Export: `src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --seed 0 --n-frames 200 --modes ref_conv --export-only
--algo "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv"` (the sibling's arguments; calibration defaults, 32 frames, calib seed 1234). It started on the working tree that
holds the code of `38ef5ff` before that commit existed; the manifest records `atlas_torchao_commit 6b1b358`, dirty false (read when the manifest was written; `src/` is identical in the two commits).
Wall time was about 30 minutes on a host with load average above 300 (the sibling took 385 s); that figure is contention, not a cost of the change.
Capture: `policy_fx_capture.py dit EXPORT OUT`, strict, same boundary (`DiTDenoiseStep`), 2 cases; status passed, fresh-process replay passed, outputs byte identical.

| | sibling `atlas_v1_awq_tile_static` | `atlas_v1_awq_tile_static_attn` |
|---|---|---|
| quantized sites in `policy.json` | 179 Linear + 24 attention, 1 Conv2d host | 179 Linear + 30 attention, 1 Conv2d host (209 quantized) |
| attention sites by variant | 24 `text` | 24 `text` (CLIP) + 6 `vision` (`noise_predictor.transformer_blocks.N.attn`) |
| `atlas.qk_matmul` / `atlas.pv_matmul` | 24 / 24 | 30 / 30 (fresh-process replay: 30 / 30) |
| `atlas.qlinear` / `quantize_act` | 179 / 275 | 179 / 299 (275 + 6 x 4) |
| `unquantized_attention` (verifier) | 6 (howto-model-to-fx-graph.md; the sibling's stored `verify_bundle.json` predates the verifier fix and was not re-run here) | 0 |
| `verify_bundle.py` | see previous row | 25 of 25, `ok: true`, `uncalibrated_fallback` 0 |
| tile-native sites | | 209 of 209 |
| `still_emulated.emulated_modules` | | empty; 30 modules in `native_fp8_attention_modules`, 1 pair each |

The sibling column is copied from `howto-model-to-fx-graph.md`, not re-measured. The 24 CLIP sites stayed native after the re-export (`native_attention` did not fall back), as the 30 pairs show.

### Pack-scale admission (policy.json `compiler_pack_scale_admission`, admitted range 122..128)

209 static sites checked, 94 outside (was 88 of 203): 64 Linear inputs (22 under `noise_predictor`, same 64 as before) and all 30 `p` operands, of which the 6 new DiT `p`
are all code 119 (static exponent -8), like the 24 CLIP `p`. Linear inputs sit at 119 to 121. So the 6 new sites add 6 to the sites outside the admitted range; the new q, k, v operands
are dynamic and are not part of this check. This is the same issue as in `pack-range-options.md`, now with 6 more `p` sites.

### Tests run (after the export finished)

`tests/test_algo_tile.py`, `tests/test_fx_native.py`, `tests/test_dit_rope_attention.py` together: 77 passed, 16 skipped (skip reasons not inspected), 174.9 s. The full suite was not run.

### Not measured

* No quality number (no comparable eval; the sibling had none either). BF16 vs FP8 DiT attention output was not compared on real frames.
* T = 24 is not a multiple of 32, so each DiT attention matrix is a partial tile (Q, K, V of 24 x 64 and P of 24 x 24 per head). `tile_native.attention_site_status` checks exponent
  constancy only, so tail-tile handling has to be confirmed by the compiler side. With `attn=block` and rows 32 each (batch, head) matrix is one partial block (unit test checks scale shapes per (b, h)).
* Dropout is not modelled in the patched forward (p = 0 in eval).
