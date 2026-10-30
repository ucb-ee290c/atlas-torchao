# Two-term FP8 activations (`--act-two-term`)

Opt-in, eval time, default off. The named recipes (`atlas_v1_*`) are unchanged.

## Why

On real TinyLlama layers the FP8 layer error splits evenly: weight-only 0.0183, activation-only 0.0181, both 0.0258
(`results/hw-aware/two_term_screen.py`). Representing the activation as two FP8 terms,
`x ~ dq(Q(x)) + dq(Q(x - dq(Q(x))))`, each with its own per-token power-of-two scale, reproduces a BF16
activation to about 8 mantissa bits and removes the activation error: mean layer error 0.0258 -> 0.0183.

Two-term weights would give the same gain but double the weight DMA, which is the bound on Atlas at decode and
small prefill; they are not implemented.

## Semantics

`--act-two-term {off,all,down}` (registered in `quant_modes.add_profile_args`; `down` wraps only `*down_proj`).
After the quantized model is built (normal path and `--from-export`; `eval.py`, `eval_policy.py`) every
`RefQuantLinear` (profiles reference, reference_hw, chip, chip_rope_vpu) with a BF16 weight computes

    x1 = dq(Q(x))              the layer's own activation cast (algos.base.act_cast: granularity a=token, rule nc, ...)
    r  = x - x1                BF16 vsub, truncating (ChipVPU.sub)
    y  = F(x1) + F_nobias(r)   F = the layer's existing forward; one BF16 vadd (truncating)

x1 lies on the FP8 grid, so F re-quantizes it exactly (dequantized values; a row whose amax lands on a boundary
may pick a different exponent but dequantizes identically). The bias is added once: the layer's bias is
temporarily set to None for the second call, so it needs no subtraction. Rank > 2 inputs work (per-token over the
last dim). In the chip profile the vsub and vadd run on the router's VPU (counted in the coverage) with the active
`--chip-vadd` mode. FP32 layers (policy action MLPs) are left single-term and listed under `skipped_non_bf16`.
An algorithm with an online activation transform raises (it does not commute with the split).

Not touched: attention matmuls (QK^T, PV), Conv2d, weights. The forward is replaced on the module instance, so
module names, hooks, the cross-check and exports are unchanged. The result JSON carries
`hw_two_term[mode]` (eval.py) or `modes[mode].hw_two_term` (eval_policy.py).

## Hardware cost (`hw_cost.two_term_layer_cost`)

The same weight tile serves both passes (two accumulators, two weight slots per MXU). The MXU takes 32 cycles per
resident tile and token block per pass, while a 1 KiB tile costs about 764 cycles of DMA. The second pass starts
to cost time at 2 x 32 x blocks > 764, i.e. above 11.9 blocks (about 380 tokens). Extra VPU work per activation
tile (formed once, shared by all N tiles): vunpack, dequant vmul, vsub, vredmax, vredmin, pre-vpack vmul, vpack;
per output tile: pass-2 rescale vmul and the final vadd. The VPU part is INFERRED (no simulation of the sequence).

## Attention operands (`--act-two-term attnq | attn | all+attn`)

Choices: `attnq` = attention only, Q two-term; `attn` = attention only, Q and P two-term; `all+attn` = every
quantized Linear (as `all`) plus Q and P two-term in attention. `off` stays the default and is bit-identical.
The patched attention routine (`ref_attention.ref_attention(two_term=None|"q"|"qp")`, set per site by
`hw_two_term.apply` through the site's `_atlas_ref_attn["two_term"]`) then computes, per (batch, head), with the
same helpers and the active vadd mode as the Linear version:

    Q1 = dq(Q(Q))   (exactly the single-term cast),  Q2 = Q(Q - Q1) with the residual a BF16 truncating vsub
    S  = fm(Q1, K) + fm(Q2, K)     each pass carries its own 2^(s+sk) rescale; the two partials are rounded to BF16
                                   and summed by one BF16 vadd; scaling, mask and softmax follow as before
    O  = fm(P1, V) + fm(P2, V)     (qp only; P is formed in BF16 before its cast, as the stack does)

K and V are the resident operands and stay single-term; the boolean (and additive) mask handling is unchanged. Under
the chip profile the vsub/vadd run on the router VPU (counted). Consequence: with Q two-term the scores reach the
softmax as a BF16 sum (the single-term path keeps the float32 matmul output). Pi0, SmolVLA text/expert and vision
variants all go through the same function. An algorithm with an online attention transform raises.

Check (TinyLlama, 256 tokens, reference profile, error of the attention output O vs exact BF16 attention; results in
`results/hw-aware/two_term/attn_wrapped_check.json`, script next to it): O error L0 .0365 -> .0316, L10 .0925 -> .0638,
L21 .0624 -> .0444 (single -> qp); after an exact o_proj L10 .0934 -> .0603 (screen predicted .0927 -> .0583), Q-only
.0647.

### Hardware cost (`hw_cost.attention_two_term_cost`)

K and V are read from VMEM (the resident tile of the matmul); the second term streams a second Q or P against the same
tile, so matrix passes double and no extra weight DMA occurs. Extra VPU work per streamed tile and per output tile as
for the Linear layers (INFERRED). TinyLlama (32 heads, 4 KV heads, D=64), Skv = 2048, full square, softmax and casts
not modelled:

| case | variant | MXU cycles 1 -> 2 | extra VPU tile instrs | overhead |
|---|---|---|---|---|
| prefill 2048 tokens, KV in VMEM | q | 16.8M -> 25.2M | 290.8k | +50% (MXU) |
| prefill 2048 tokens, KV in VMEM | qp | 16.8M -> 33.6M | 1.22M (70.7M cycles) | +321% (VPU bound by the P tiles) |
| decode 1 token, KV in VMEM | q | 262k -> 393k | 4.5k | +50% (MXU) |
| decode 1 token, KV streamed from HBM (782k cycles DMA) | q | same | same | 0%, DMA bound |
| decode 1 token, KV streamed from HBM | qp | 262k -> 524k | 19.0k (1.10M cycles) | +41% (VPU) |

It is free only where the weight-style DMA bound hides the second pass: a decode step whose KV cache streams from HBM,
Q-only. With the cache resident in VMEM, or at prefill, the matrix passes double (Q: +50% of attention MXU time) and the
P term is VPU-bound because every P tile (Sq/32 x Skv/32 per head) needs the residual and recast sequence. Per-(b, h)
exponents also force the residual to be held until its row reductions finish. The cost is for a full square (no causal
tile skipping), and the baseline softmax/cast VPU work is in neither column, so the real overhead is smaller where that
work already bounds the layer.
