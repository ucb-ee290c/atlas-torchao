# Stacking algorithms (`--algo A+B+...`)

`--algo` takes one algorithm (`NAME[:k=v,...]`) or several joined by `+`:

```
--algo smoothquant:alpha=0.5,po2=true,gain=false+gptq_awq:method=gptq,actorder=true+granularity:w=channel,a=token,rule=nc
```

Everything on this page is **NEW**: neither the reference author's code nor the papers behind the
individual tracks define how their methods compose. The rules below are this repository's choice,
made so that every member sees exactly what it would see alone, on the model the earlier members
left behind, and so that no decision is ever taken twice silently.

Code: `src/algos/stack.py` (`Stack`), the parser and `algos.quantize` in `src/algos/__init__.py`.
Tests: `tests/test_algo_stack.py`.

## Parsing

- A `+` starts a new algorithm only when the text after it begins with a registered name
  (`smoothquant`, `gptq_awq`, `granularity`, `static_calib`, `mixed_precision`, ...). Otherwise
  it belongs to the current value, so `smoothquant:sites=qkv+fc1` keeps its `+`.
- A parameter token without `=` continues the previous value with a comma:
  `rotation:online=down,o,impl=fp32` gives `online="down,o"`. A joined value stays a string.
- A single spec parses exactly as before and builds the same single `Algo` object; its result
  record is unchanged.
- `none` cannot be stacked, and the same algorithm may not appear twice.

## Order

Members run in a fixed order, whatever order the spec lists them in:

| Stage | Members, in order | Why first / last |
|---|---|---|
| transform | `example_identity`, `smoothquant`, `gptq_awq` | Folds change the weights and activations every later decision is taken on. GPTQ rounds the weights that smoothing left. |
| scale | `granularity`, `static_calib` | Scale choices are taken on the transformed tensors. |
| exclusion | `mixed_precision` | Layer exclusions and residual passes act on the final layer set. |

A registered track not in this table runs after the transforms and before the scale choices.

Model-level hooks in `algos.quantize`:

1. `bf16_layers` of every member (union, in order).
2. For each member in order: `calibrate` (if it needs calibration) on the model as the earlier
   members' `prepare` left it, then its own `prepare`. The calibration data is loaded once and
   shared.

## Module hooks

| Hook | Rule |
|---|---|
| `transform_weight` | Chained in order. |
| `online_transform` | Composed in order. |
| `act_scale` | At most one member may return a `Scale` for a given (layer, kind). Two answers raise `StackConflict` at that layer's first cast. |
| `residual_passes` | The same rule. |
| `weight_scale` | The same rule, with one explicit delegation (below). |

**Weight-grid delegation.** `gptq_awq` rounds weights onto a grid it does not itself choose. When a
later member chooses the weight scale (for example `granularity:w=channel`), the stack hands that
member's `weight_scale` to `gptq_awq` as its grid (`GptqAwq.weight_grid`): GPTQ quantizes each
column onto the per-row exponents, AWQ's search evaluates Q with the same rule, and the exponent is
pinned exactly as `gptq_awq` pins its per-tensor exponent alone. Its pinned `Scale` (with the
chooser's granularity descriptor) then answers for both members, and the result lists it under
`algo.delegations`. A per-block grid (`w=block32`) is refused.

## Refused combinations

- `smoothquant` smoothing (`alpha` not `off`) with `gptq_awq:method=awq|awq_gptq`. Both fold
  per-channel factors into the same norms and producers. The composition is exact in real
  arithmetic, but the second search would optimise against already-smoothed statistics, and no
  run measured it. Stack smoothing with `method=gptq`, or use `smoothquant:alpha=off` (gain only).
- Two scale choices for the same decision, for example `granularity:a=token` with `static_calib`
  (both choose activation scales), `granularity:w=channel` with `static_calib:weight=ceil448`, or
  `smoothquant:static=true` with `granularity:a=token`.

## Result record

A stacked run records under `algo` (per mode):

- `name: "stack"`, `spec` (as given), `order`, `params` per member
- `members`: each member's own record (without the cast counts) plus its `stage`
- `calibration` per member, `prepare` per member, `delegations`
- `granularity`: the cast counts of the whole stack (each cast is counted once)
- `applied`, `bf16_opt_out`, `module_hooks`, `module_hooks_applied` as for a single algorithm

`--chip-vadd` stays a separate flag; it is not part of `--algo`.
