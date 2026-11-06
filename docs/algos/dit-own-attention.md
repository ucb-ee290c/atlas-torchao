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

## Re-export and re-capture

Not completed when this commit was written. The export was started on the committed code's working tree (before the code commit; the export manifest will
record the tree as dirty at `79628ee`) into `exports-tile/multitask-dit/atlas_v1_awq_tile_static_attn/export` with
`src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --seed 0 --n-frames 200 --modes ref_conv --export-only --algo "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv"`
(same arguments as the sibling `atlas_v1_awq_tile_static`; calibration defaults: 32 frames, calib seed 1234). After 30 minutes of wall time on a host with load
average above 300 it had not finished. The capture (`policy_fx_capture.py dit EXPORT OUT`), `scripts/tile_flow/verify_bundle.py` and `make_bundle_manifest.py`
were not run, so there are no graph counts, no pack-scale admission numbers and no verifier result for the new bundle yet. Nothing here measures them.

Not measured, from reading: T = 24 is not a multiple of 32, so each DiT attention matrix is a partial tile (Q, K, V of 24 x 64 and P of 24 x 24 per head);
`tile_native.attention_site_status` checks exponent constancy only, so tail-tile handling has to be confirmed by the compiler side.
