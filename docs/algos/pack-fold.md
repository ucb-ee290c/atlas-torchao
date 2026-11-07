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

## Validation

### Tiny random Llama (tests/test_pack_fold.py, 72 tests, all passed together with test_pack_plan.py, test_tile_flow_defects.py and test_algo_tile.py: 184 passed, 0 skipped)

Static recipe `atlas_v1_awq_tile_static`, 2 layers: 16 static sites, 16 outside, after the fold 2 outside (the two attention p sites; 14 folded:
10 norm_affine, 2 gated_product, 2 attn_out_pv). Eager logits of the folded and unfolded exports are `torch.equal`; the native-FP8 FX captures of both give
`torch.equal` outputs; `verify_bundle.run` on the folded bundle fails the same checks as on the unfolded one (none) and reports the `pack_fold` block;
an export made with `--pack-fold` (in-process capture) replaces the export and keeps `<dir>.unfolded`. The mechanism is also tested on the real operators
(`atlas.quantize_act`, `qlinear`, `pv_matmul`) for several `(s, k)`, including a negative control that shows the comparison is not vacuous (leaving the
consumer weight exponent alone, or not moving the pack exponent with the value, changes the output). Default export: no `pack_fold*` / `pv_pop_shift` keys,
no `<dir>.unfolded`, and the FX graph of the unfolded export has no `pop_shift` argument.

### Real TinyLlama (post-hoc fold of the existing export, no re-calibration)

Folded bundle: `/scratch2/lorenhung/atlas-fp8/exports-tile-fold/tinyllama/atlas_v1_awq_tile_static_packfold/` (`export/` folded export, `fx/` new capture
of the folded export at `--seqlen 64`, `verify_bundle.json`, `compare_fx_logits.json`, `bundle_manifest.json`). Source (read only):
`/scratch2/lorenhung/atlas-fp8/exports-tile/tinyllama/atlas_v1_awq_tile_static`.

| | before | after the fold |
|---|---|---|
| static activation sites | 176 | 176 |
| outside 122..128 | 114 | 22 |
| in range | 62 | 154 |
| sites folded | | 92 (59 norm_affine, 11 gated_product, 22 attn_out_pv) in 57 producer groups (24 gamma tensors, 11 up_proj, 22 PV) |
| still outside, by class | attn_p 22, attn_out_pv 22, gated_product 11, norm_affine 59 | attn_p 22 (reason `class_not_free`, cost `vmul_per_row_block`, plan k = 3) |
| pack codes (code: sites) | 117:6 118:18 119:51 120:23 121:16 122:55 123:3 124:1 125:1 126:2 | 119:22 122:147 123:3 124:1 125:1 126:2 |
| `k` over the 176 sites (k: sites) | | 0:84 1:16 2:23 3:29 4:18 5:6 |
| pop exponent `sx + sw` bound over the 154 Linears | -23..-9 | -23..-9 |

Fact worth knowing: the nearest in-range choice puts 147 of the 176 sites on code 122, the lowest admitted code. The research doc records general-value system
pack tests only at codes 126 and 128 and the RTL-exact `resnet_pool` launch at 127 (pack-range-options.md 4.A); no evidence at 122 was gathered here.
Choosing a different target code changes only `k`; it is not implemented (default kept as asked).

BF16 range guard (24 gamma tensors scaled): smallest non-zero `|2^k gamma|` 2.3e-4, largest 3.73, BF16 minimum normal 1.18e-38, maximum 3.39e38; 0 failures,
0 groups refused.

FX bundle: `capture_fx` passed (decode-graph outputs identical to native graph, same-process and fresh-process reload byte identical: 154 qlinear, 22 qk_matmul,
22 pv_matmul nodes). `verify_bundle.py`: 25 of 25 checks pass (the unfolded bundle: 25 of 25, same check names). Graph: 242 `quantize_act`, 154 `qlinear`,
22 `qk_matmul`, 22 `pv_matmul`, the same counts as the unfolded bundle; the 22 `pv_matmul` nodes carry the extra argument `pop_shift`.
The out-of-range line is an informational report block (`verify_bundle.json` `pack_fold`: 22 sites still out of range, all attn_p, recorded summary equals the
recomputed one), not a counted check, so the count stays 25.

Logits, unfolded versus folded FX bundle (`scripts/tile_flow/compare_fx_logits.py`): 8 inputs of shape [1, 64] (the 2 capture-case inputs and 6 random token
sequences, seed 0), 16,384,000 logits compared: **0 differing logits, max abs difference 0.0, byte identical**. This is the FX emulation with the
framework's float32 accumulation; no BF16-accumulation or hardware run was made.

Not re-measured here: the byte identity of the FP8 codes on real activations (0 of 116,654,080 differing, measured in pack-range-options.md section 1);
the end-to-end logits comparison covers it indirectly (a differing code would change a logit; the negative control above shows a wrong fold does).

### Other bundles

| bundle | static sites | outside before | folded | outside after | outside after, by class | end to end |
|---|---|---|---|---|---|---|
| TinyLlama | 176 | 114 | 92 | 22 | attn_p 22 | yes (above) |
| SmolVLA | 316 | 244 | 181 (145 norm_affine, 24 gated_product, 12 attn_out_pv) | 63 | attn_out_pv 32, attn_p 13, linear_to_linear 9, bare_activation 6, raw_input 2, modulation_add 1 | yes (below) |
| Multi-Task DiT | 203 | 88 | 24 (attn_out_pv) | 64 | attn_p 24, x_sigmoid_product 17, bare_activation 13, attn_out_sdpa 6, modulation_add 1, index 1, sin 1, raw_input 1 | **no**, counts only |
| Pi0 | 483 | 127 | 61 (27 attn_out_pv, 24 norm_affine (1 + w), 7 gated_product, 3 layer_norm) | 66 | attn_p 63, raw_input 2, bare_activation 1 | **no**, counts only |

SmolVLA, end to end (post-hoc fold of the existing export, fresh `fx_export.py --boundary smolvla_flow_loss --native-fp8 --seqlen 8` capture of the
folded export into `/scratch2/lorenhung/atlas-fp8/exports-tile-fold/smolvla/atlas_v1_awq_tile_static_packfold/`): capture passed (302 qlinear, 1 qconv2d,
36 qk_matmul, 36 pv_matmul nodes, fresh-process reload byte identical), `verify_bundle.py` 25 of 25 (unfolded: 25 of 25), `pack_fold` report 63 sites still out of range,
recorded equals recomputed. The flow-loss outputs of the two bundles on the same 2 capture cases (2 x 1600 values): eager outputs stored in the two `fixtures.safetensors`
are `torch.equal`, and the two `model.pt2` graphs give 0 differing values, max abs difference 0.0 (`compare_fx_outputs.json`). 65 gamma tensors scaled, smallest
`|2^k gamma|` 3.0e-3, largest 5.5, 0 BF16-range failures.

A bug this end-to-end run found (fixed, regression test `test_attention_site_with_two_pv_nodes_is_not_folded`): `pv_pop_shift` is stored per attention SITE, and
SmolVLA's shared text attention site has 24 `atlas.pv_matmul` nodes (module called 24 times). The first version folded the groups of that site, so the shift
reached all 24 nodes and the outputs differed (all 1600 values, max abs difference 81.7 in the first case). A fresh-process bisect over the folded classes showed
`gated_product` and `norm_affine` exact and `attn_out_pv` not. The pass now refuses a P.V fold when the attention site has more than one `pv_matmul` node
(`attention_site_called_more_than_once`): 24 groups, 32 sites in SmolVLA. TinyLlama, DiT and Pi0 have one `pv_matmul` per site (counted from the graphs), so their counts do not change.
A per-node shift would need the shift to live on the node, not the policy site; not implemented.

Differences from the research estimate (92 / 221 / 24 / 61 free): SmolVLA 181 instead of 221 (-32 shared-attention P.V sites above, -8 below); TinyLlama, DiT and Pi0 equal the estimate.
The 8 SmolVLA `linear_to_linear` sites that the research counted as free are the VLM `v_proj` outputs; the same tensor is also packed by a dynamic attention-v
`quantize_act` in the same graph, so scaling the producer would change the attention numerics (the dynamic v exponent follows the scale and the PV output is then `2^k` too
large unless `pop_shift = -k` is also set). The pass refuses them (`non_static_or_attention_consumer`); the 9th has a non-pack consumer (`expand_as`).
Pi0: all 24 Gemma `(1 + w)` norms fold (the parameters are float32 and `fl32(w' + 1) == 2^k fl32(w + 1)` held for every element), the 3 LayerNorms fold, 0 BF16-range
failures (smallest effective multiplier 1.06e-6, largest 30.1); a BF16 `(1 + w)` parameter would be refused. DiT: its 24 free sites are all P.V outputs (no norm or gated product).

DiT and Pi0 were not validated end to end. DiT: re-capturing even the UNFOLDED existing export with the current code fails before the fold is involved
(`KeyError: export policy has no quantized site 'noise_predictor.transformer_blocks.0.attn'`; the export predates the DiT attention sites that `_dit_rope_attention`
now routes through `ref_attention`), and a new DiT export is a long run. Pi0's capture takes about 1,549 s. Their P.V, LayerNorm and `(1 + w)` folds are therefore covered
by the synthetic tests and the SmolVLA / TinyLlama runs of the same code paths, not by a bundle of their own.

## Limits and open items

* Not free, not touched, still out of range: attention p (all bundles; needs one `vmul` of `recip(sum)` by `2^k` per (head, 32-row block) or the compiler's
  `vmul` fallback), bare GELU / SiLU and `x * sigmoid` (activation-constant rewrite, not run), SDPA output / modulation add / sin / index (one `vmul` per tile),
  raw model inputs (host). The pass does not invent folds for them.
* The target is the nearest in-range code, which is 122 for 147 of 176 TinyLlama sites (see above). The code of the pack is therefore at the edge of the
  admitted range; if system evidence at a code nearer 127 is wanted, only `k` changes (not implemented).
* `pop_shift` is a new trailing argument of `atlas.pv_matmul` (default 0). A compiler or importer that reads `pv_matmul` nodes must accept it; unfolded
  bundles and every bundle saved earlier are unaffected. The BF16-pop `vmul` that applies `2^(sp+sv)` already exists in the kit's attention (pack-range-options.md
  2.3); that the same constant can absorb `2^k` is **[inference]**, not run on RTL.
* The consumer's `weight_scale_exp` changes by `-k` (codes untouched) so that `sx + sw` is constant. A compiler that derives the pop constant from the weight
  exponent and the activation exponent separately gets the same product; one that pins `sw` to a stored value would not. The weight exponent code stays within
  `1..254` (checked per group).
* Producers read by another node, parameters read by more than one node, aliased tensors, weights read by more than one graph node, residual-pass Linears and
  sites packed at several nodes are refused.
* The offline guard cannot see activation magnitudes: `2^k x` overflowing BF16 is excluded only by the calibrated range (`|x| <= 448 * 2^s`, so
  `|2^k x| <= 448 * 2^(s+k)`, with `s + k <= 1`), not measured per element.
* `--pack-fold` inside the export CLIs needs the FX graph: causal LMs are captured in process; for other models pass `--pack-fold-graph <fx dir>` from a capture
  of that model's unfolded export, or use the post-hoc CLI `python src/pack_fold.py` (the path used for every number above). Only causal-LM (TinyLlama) and SmolVLA folded
  bundles were exercised end to end here; the export CLI flag itself was exercised only on the tiny Llama.
* The eval result JSON files (`--out`) list `vars(args)`, so they gain two keys (`pack_fold`, `pack_fold_graph`); the export files (`policy.json`, `manifest.json`,
  `model.safetensors`) of a default run are not affected.

## Commands

```bash
# all with PYTHONPATH=src, env /scratch2/lorenhung/envs/atlas-lerobot/bin/python, taskset -a -c 0-15 nice -n 10, OMP/MKL 4 threads
python src/pack_fold.py --export-dir <tinyllama>/export --fx-dir <tinyllama>/fx --out-dir <fold>/export        # 5.9 s
python src/fx_export.py --export-dir <fold>/export --out-dir <fold>/fx --native-fp8 --seqlen 64               # about 7 min
python scripts/tile_flow/verify_bundle.py <fold>/export <fold>/fx <fold>/verify_bundle.json                   # 10.7 s, 25 of 25
python scripts/tile_flow/compare_fx_logits.py <tinyllama>/fx <fold>/fx <fold>/compare_fx_logits.json --random 6
python scripts/tile_flow/make_bundle_manifest.py <fold>
# SmolVLA: same pack_fold.py call, then
python src/fx_export.py --export-dir <fold-smolvla>/export --out-dir <fold-smolvla>/fx --boundary smolvla_flow_loss --native-fp8 --seqlen 8   # about 9 min
python scripts/tile_flow/verify_bundle.py ... ; python scripts/tile_flow/compare_fx_logits.py <smolvla>/fx <fold-smolvla>/fx <fold-smolvla>/compare_fx_outputs.json
python -m pytest tests/test_pack_fold.py tests/test_pack_plan.py tests/test_tile_flow_defects.py tests/test_algo_tile.py -q   # 184 passed, 225 s
```
