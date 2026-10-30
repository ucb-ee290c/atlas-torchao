# k_order: function-preserving K permutation of Llama MLP down_proj (NEW, opt-in)

Not in any recipe (the `atlas_v1_*` recipes are immutable). Stack it last:

    --algo gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc+k_order:rule=asc

(the same as `atlas_v1_awq+k_order:rule=asc`). Parameters: `rule=asc|desc|rand|none` (required),
`seed=N` (rand only).

## Why
The hardware accumulates K tiles in the order software issues them, one RNE to BF16 per 32-wide tile with
the BF16 partial sum carried on (results/hw-aware/hw_ipt_error.json). The K order is therefore a free offline choice
wherever a permutation folds away. For down_proj of a Llama MLP (input `act(gate(x)) * up(x)`) permute the rows of
gate_proj and up_proj and the columns of down_proj with the same pi. q/k/v/gate/up read the residual stream and o_proj
is tied to the head structure by GQA, so they are not touched.

## What it does
- Runs on the already-quantized model, in place, when the algo record is built (algos.quantize), after every other
  member. It registers rank 60 in `algos.stack.ORDER` on import; stack.py itself is unchanged.
- Moves `weight_codes`, `weight_norm`, residual-pass tensors, the shared `weight` and (rows) `weight_scale_exp` and bias
  with their rows / columns. No value changes: a row maximum is invariant to column order, per-token and per-tensor
  activation scales (max over K) are invariant, so FP8 values, flush and clip are identical; only the accumulation order
  changes. Exports simply hold the permuted tensors.
- Importance of K-channel k: `mean|x_k| * ||W[:,k]||_2` (x = down_proj input over the calibration tokens). mean|x_k| is
  collected by k_order's own pass in `calibrate` on the unquantized model (gptq_awq's s_X is private to that member and
  keyed differently). `asc` puts the least important channel first, `desc` the most important, `rand` is a seeded
  control, `none` does not touch the model. Without calibration statistics (`apply(model)` called directly) the weight
  column norm alone is used.
- Refused with a clear error: models without Llama MLPs (any `down_proj` outside a gate/up/down + `act_fn` module),
  per-block weight scales along K on down_proj, per-block activation scales (`granularity a=block*`), online transforms on
  these layers.
- The algo record carries `rule`, `seed`, `applied_permutation` (per layer: K, hash, importance source; overall hash).

## Screen (scripts/k_order_screen.py, results/hw-aware/k_order_screen.json)
TinyLlama-1.1B-Chat down_proj, layers 0/7/14/21, 64 WikiText-2 test tokens x 16 random outputs, importance from 2048
train tokens, real FP8 codes. Relative L2 error of the bit-exact chain against the float64 exact sum of the same FP8
operands (which is identical for every order):

| layer | FP8 input | IPT none | IPT asc | IPT desc | IPT rand (3) | SA none | SA asc | SA desc | SA rand |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.0371 | 0.0139 | 0.0137 (-2%) | 0.0181 (+30%) | 0.0154 (+11%) | 0.0676 | 0.0528 (-22%) | 0.0732 (+8%) | 0.0651 (-4%) |
| 7 | 0.0220 | 0.0092 | 0.0061 (-34%) | 0.0186 (+103%) | 0.0119 (+30%) | 0.0323 | 0.0277 (-14%) | 0.0408 (+26%) | 0.0355 (+10%) |
| 14 | 0.0369 | 0.0156 | 0.0117 (-25%) | 0.0174 (+12%) | 0.0147 (-6%) | 0.0754 | 0.0676 (-10%) | 0.0944 (+25%) | 0.0746 (-1%) |
| 21 | 0.0263 | 0.0167 | 0.0068 (-60%) | 0.0202 (+21%) | 0.0148 (-12%) | 0.0740 | 0.0315 (-57%) | 0.1125 (+52%) | 0.0725 (-2%) |

Caveats: one sample of 64 x 16 outputs per layer; rand varies by about +-20% between seeds, so differences inside that
band (layer 0 IPT asc) are not distinguishable from a lucky order. Layers 0 and 14 are the least reliable.
