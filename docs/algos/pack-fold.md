# Pack fold: static activation pack exponents into the compiler's admitted range, exactly (opt-in)

Facts only. Every number below was printed by a command listed in "Commands". Statements about hardware or compiler behaviour that
were not run are marked **[inference]**. This implements mitigation A of [pack-range-options.md](pack-range-options.md) (4.A) for the
producer classes that need no added instruction. Default exports are unchanged: the pass runs only with `--pack-fold` (or
`python src/pack_fold.py`).

## What it does

A static activation exponent `s` means `real = e4m3(byte) * 2^s`; the E8M0 pack code is `c = 127 + s`; the compiler's `lowerPack`
admits `122 <= c <= 128` (`tile_native.PACK_ADMISSION`). Packing `2^k * x` at `s + k` gives byte-identical FP8 codes to packing
`x` at `s` (BF16 scaling by a power of two is exact; the pack depends only on `x * 2^-s`). So for a site `s` outside the range the pass

1. multiplies the **producer** of the packed tensor by `2^k` exactly,
2. sets the static input exponent of every consumer of that tensor to `s + k` (policy `activation.exp`, `activation.e8m0_code`, the
   `<site>.input_scale_exp` tensor), so the pack code is in range, and
3. subtracts `k` from the consumer's `weight_scale_exp` (the E4M3 weight codes are not touched), so the pop / dequant constant
   `2^(sx + sw)` of the consumer is the same number as before.

The FX graph keeps its structure and node count. What differs in a folded bundle: the `static_exp` literal of each folded
`atlas.quantize_act`, the int32 `weight_scale_exp` buffers, the gamma / bias parameters, and one trailing integer argument
(`pop_shift`) on the `atlas.pv_matmul` of a folded attention site. No calibration runs and no FP8 byte changes.

`k` is the smallest-`|k|` shift that puts every site of a producer group into `122..128` (`pack_fold.group_k`; for one site it is
`pack_plan.fold_for_code(code)["k"]`, the default "nearest in-range code"). All sites that read one producer tensor (q/k/v after one
norm, gate/up after one norm) share one `k`; a member already in range moves with its group. A group whose codes do not fit one
7-code window is left alone (reason `group_spread`).

## What is folded, what is not

| producer class (`pack_plan`) | free of added instruction | what is scaled by `2^k` | status |
|---|---|---|---|
| `norm_affine`, RMSNorm `gamma * x` | yes | the stored `gamma` tensor (BF16 / FP32, checked exact) | implemented |
| `norm_affine`, Gemma `x * (1 + w)` | yes | `w' = 2^k (1 + w) - 1`, accepted only when float32 and `fl32(w' + 1) == 2^k fl32(w + 1)` for every element | implemented, refused for BF16 `w` |
| `layer_norm` | yes | LayerNorm weight and bias | implemented |
| `gated_product`, `act(gate) * up` | yes | every `weight_scale_exp` of `up_proj` `+= k` (bias `* 2^k` if present) | implemented |
| `linear_to_linear` | yes | the producing Linear's `weight_scale_exp += k`, only when nothing else reads its output | implemented, see Limits |
| `attn_out_pv` (P.V into o_proj) | yes | the PV pop rescale constant: policy attention site `pv_pop_shift += k`; `atlas.pv_matmul(..., pop_shift)`; `ref_attention` | implemented |
| `attn_p` | **no**: one `vmul` on `recip(sum)` per (head, 32-row block) | not touched | reported out of range with its plan |
| `bare_activation`, `x_sigmoid_product` | no: activation-constant rewrite (not run) | not touched | reported |
| `attn_out_sdpa`, `modulation_add`, `norm_modulated`, sin / index | no: one `vmul` per tile | not touched | reported |
| `raw_input` | no: host scales the input | not touched | reported |

Never folded (reason recorded in `pack_fold.reason`): the producer has another consumer (`other_consumers:...`), a consumer is not a
single static Linear / Conv2d (`non_static_or_attention_consumer`, `consumer_not_a_single_linear`), a site is packed at more than one node,
a consumer or the producer has residual passes, `|k| > 20`, or an exactness guard fails (below).

## Exactness guards (applied before anything is written, per group)

* gamma / bias / LayerNorm parameters: `param * 2^k` must be finite, equal to its float32 product after the cast back, and no non-zero
  result may fall below the BF16 minimum normal (a BF16 subnormal result counts as underflow); otherwise the whole group stays as it
  was and the failure is listed in `pack_fold_summary.bf16_range.failures`.
* weight exponents: the new E8M0 code must stay in `1..254` and the dequantized weights in the BF16 normal range.
* The consumer's pop exponent `sx + sw` is unchanged by construction; `pack_fold.pop_exponent_unchanged` checks it.
* The pack input `2^k x` must not overflow BF16: not checkable offline for activations (their range is data); it is covered only by
  the bit-identical comparison on the cases listed under Validation.

## Where it lives

* `src/pack_fold.py`: `graph_info` (producer, shared-input groups and the tensors to touch, from `fx/graph.txt` + `fx/graph_signature.txt`),
  `plan_folds`, `apply_folds`, `annotate` (policy records), `fold_export_dir` (post-hoc, exact transform of an export directory with
  recomputed manifest hashes), CLI `python src/pack_fold.py --export-dir E --fx-dir FX --out-dir OUT`.
* Exporter flag `--pack-fold` (`quant_export.add_args`, honoured by `eval.py`, `eval_policy.py`, `eval_pi0.py` through `options`): after the
  export is written, the graph is captured in process (causal LMs) or read from `--pack-fold-graph <fx dir>`, the export in `--export-dir`
  is replaced by its folded twin and the unfolded export is kept in `<export-dir>.unfolded`.
* `atlas.pv_matmul(..., pop_shift=0)` (new trailing argument, default 0; `fx_export` passes it only when non-zero, so every other node and every
  bundle saved earlier keeps its argument list), `ref_attention._mm(..., pop_shift=0)`, `algos.base.attn_pv_shift`, `PolicyAlgo.pv_pop_shift`.
* `scripts/tile_flow/verify_bundle.py`: `pack_fold` report block (informational, not a counted check): sites still out of range by class and cost,
  folded sites, whether the recorded summary equals the recomputed one. `scripts/tile_flow/compare_fx_logits.py`: logits of two FX bundles.

## policy.json fields (only in a folded export)

* per static Linear / Conv2d `sites[n].pack_fold`, per attention operand `sites[n].pack_folds[operand]`:
  `{k, producer (class), placement, applied, reason, exp_before, exp_after, e8m0_code_before, e8m0_code_after, consumer_weight_exp_delta, group_size, group_k}`.
  For a site that is not folded `k` is the planned `k` to the nearest in-range code (`0` when already in range) and `reason` says why.
* `sites[attention].pv_pop_shift`: integer `k` of the PV pop rescale (`2^(sp + sv + k)`); absent means 0.
* top level `pack_fold_summary`: counts before / after, `folded_by_class`, `outside_after_by_class` / `_by_reason` / `_by_cost`,
  `outside_after_sites` (each with class, code, plan `k` / placement / cost, reason), `bf16_range`, per-group results.
* `compiler_pack_scale_admission`, `pack_plan` and `pack_plan_summary` are recomputed from the folded exponents (classes from the graph).

## How a compiler owner reads the result

1. Pack code of every folded site: `policy.json sites[n].activation.e8m0_code` (122..128); the immediate for `lowerPack` is that value.
2. The consumer's pop constant is `2^(sx + sw)` with `sx = activation.exp` and `sw` the (new) `weight_scale_exp`: the product is the one of the unfolded
   bundle, so the pop code the lowering computes (`254 - c` for an FP8 pop, or the BF16 constant) does not change.
3. Producers carry the `2^k`: gamma / bias tensors in `model.safetensors`, `up_proj` `weight_scale_exp`, and for attention outputs the `pop_shift`
   argument of `atlas.pv_matmul` (the PV BF16-pop rescale constant `2^(sp + sv + k)`).
4. Sites still outside 122..128: `pack_fold_summary.outside_after_sites` and `verify_bundle.json pack_fold`; each has its plan (attention p: one `vmul` of
   `recip(sum)` by `2^k` per (head, 32-row block), as in pack-range-options.md 4.C1).
5. `pack_fold.applied == false` with `reason: in_range` means nothing was needed.

RESULTS_PLACEHOLDER
