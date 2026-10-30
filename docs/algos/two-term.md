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
