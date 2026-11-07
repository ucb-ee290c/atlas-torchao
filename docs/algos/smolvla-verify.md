# SmolVLA bundle and verify_bundle.py

Bundle: `/scratch2/lorenhung/atlas-fp8/exports-tile/smolvla/atlas_v1_awq_tile_static/`. Before the fix `verify_bundle.py` failed 7 of 25
checks on it; after the fix it passes 25 of 25 with and without `--allow-uncalibrated-fallback`. All 7 failures were verifier assumptions
(one policy site = one graph node, wrapper named `model`, Conv2d counted as a Linear). No bundle defect was found. Numbers below are from
`verify_bundle.json` (SmolVLA) and from the commands in the last section.

## Cause per failed check

| Check | Value that failed | Cause | Verdict |
|---|---|---|---|
| `graph_qlinear_count_equals_quantized_linears` | qlinear 302 vs 304 | the verifier compared `qlinear` with every quantized Linear **and Conv2d** site (303 Linear + 1 Conv2d = 304). The graph holds 302 `qlinear` + 1 `qconv2d`; the 303rd Linear, `model.vlm_with_expert.vlm.lm_head`, is never called in the flow-loss boundary, so it has no node and no lifted buffer. Only one site is missing from the graph, not two. | verifier |
| `graph_qk_pv_counts_equal_attention_sites` | qk 36, pv 36 vs 13 | the policy has 13 attention sites: 12 SigLIP `...encoder.layers.N.self_attn` (1 pair each, scoped to the site) and `model.vlm_with_expert`, the text / expert stack, which is called as `module.forward()`: its 24 qk and 24 pv nodes carry an empty `nn_module_stack` and belong to that one site (`graph_inspection.json` `native_fp8_attention_modules`: pairs 24). 12 + 24 = 36. | verifier |
| `graph_quantize_act_count` | 447 vs `n_q + 4 * n_attn` | formula used 304 + 4 x 13 = 356. Real: 302 qlinear + 1 qconv2d + 4 x 36 = 447. Consumers of the 447 nodes in the graph: qlinear 302, qconv2d 1, qk_matmul 72, pv_matmul 72 (V goes through the visible transpose). | verifier |
| `graph_weight_buffers_uint8_int32` | `weight_code_buffers` 303, `bad` [] | `bad` lists dtype errors only; the check also required `nb == n_q`, 303 vs 304. `lm_head` has no lifted buffer (`buffer_inventory.json`: 606 entries = 303 codes + 303 exponents, none for `lm_head`). Dtypes are all uint8 / int32. | verifier |
| `inspection_all_weight_sites_native` | 303 | `weight_site_bindings` has 303 rows (302 Linear + patch-embedding Conv2d), the check expected 304. | verifier |
| `static_recipe_fully_static` | 1 fallback site, `lm_head`, declared false | the export predates `uncalibrated_fallback`, so the verifier derives fallback sites from `source == "reference"`: `lm_head` has `dynamic / reference / ref` because it had no calibration data (never called). The graph holds 0 dynamic-rule `quantize_act` feeding a qlinear / qconv2d. | verifier (the site is real in the policy but unreachable from the graph) |
| `policy_reference_rule_sites_zero_for_linear` | census has 1 Linear on `reference` | the same `lm_head`. | verifier |

Two further latent verifier defects found while reading the code (neither failed on this bundle):

- `graph_has_no_float_weight_param_for_quantized_linears` matched parameters as `"model." + site`. SmolVLA's wrapper attribute is `policy`, so the
  test could not match anything. It now matches the site with the wrapper prefix stripped (the first path component, or `policy.` / `model.`).
- The Conv2d was folded into the Linear count (first row). Linear and Conv2d are now counted separately.

## What the verifier derives now (scripts/tile_flow/verify_bundle.py)

- Sites in graph: a policy quantized Linear / Conv2d is in the graph when its `weight_codes` buffer is lifted in `model.pt2`. The rest are
  `unused_derived`. They are exempt only if `unused_derived` equals the bundle's `graph_inspection.json` `unused_quantized_modules` (set
  equality; that list comes from the capture's eager forward hooks). A site missing from the graph but not declared fails
  (`graph_qlinear_count...`, `graph_weight_buffers...`, `static_recipe_fully_static`); a live site declared unused also fails.
- A fallback site that is exempt this way is reported under `static_recipe_check.fallback_sites_never_called_in_graph`; `declared` stays false.
  `static_recipe_fully_static` additionally requires `dynamic_rule_linear_conv_activations_in_graph == 0`, read from the graph (the quantize_act
  mode feeding each qlinear / qconv2d), which the policy-only version could not see.
- Attention: each qk / pv node is mapped to the longest policy attention site containing its scope (wrapper prefixes `policy.` and `model.`
  stripped). Scoped pairs must be equal for qk and pv per site and equal to the pair count in `graph_inspection.json`. Scope-less nodes
  (module called as `module.forward()`) must belong to policy sites with no scoped node, and their number must equal the sum of those sites'
  recorded pairs. Every policy attention site must have at least one pair. This scope-less count is only cross-checked against the exporter's
  tag-bound record (the tags are not stored in `model.pt2`), so it is weaker than the scoped case; the structure of each pair (operands from
  own quantize_act, P from a qk output) is checked by `fx_export.py`, not here.
- `quantize_act` count is `qlinear + qconv2d + 4 x pairs` and every node's consumers must be qlinear / qconv2d / qk / pv in the matching numbers.
- Unchanged: one qlinear per Linear site in the graph (a Linear called twice still fails, as before); no change to the contraction classifier.

Result keys added: `graph_inventory`, `graph_attention`, and the three `static_recipe_check` fields above. The number of checks stays 25.

## Is `lm_head` a bundle problem?

The export ships `lm_head` as quantized codes with a reference-rule activation, but nothing in the flow-loss boundary calls it
(`unused_quantized_modules` = `lm_head` only, `unused_parameters` []). In `lerobot/policies/smolvla` `lm_head` appears only in the parameter-freezing name lists (`smolvlm_with_expert.py` lines 170 and 190), not in a forward. It costs
export file size, not graph correctness. If a boundary that calls `lm_head` is ever captured, the policy has no static activation exponent for
it, and the verifier fails the bundle (live fallback) unless it is declared.

## Evidence (commands run)

`python scripts/tile_flow/verify_bundle.py <bundle>/export <bundle>/fx <out>` with the atlas-lerobot python (atlas-pi0 python for Pi0):

| Bundle | Result |
|---|---|
| SmolVLA (default flags, written to the bundle's `verify_bundle.json`) | 25 of 25, qlinear 302, qconv2d 1, qk 36, pv 36, quantize_act 447 |
| SmolVLA `--allow-uncalibrated-fallback` | 25 of 25, `declared` true |
| TinyLlama static | 25 of 25 (qlinear 154, qk / pv 22, quantize_act 242) |
| Multitask DiT static | 25 of 25 (qlinear 179, qk / pv 24, `unquantized_attention` 6) |
| Multitask DiT static attn | 25 of 25 (qk / pv 30) |
| Pi0 static and Pi0 GPTQ panel no-AWQ | 25 of 25 each (qlinear 420, qk / pv 63, quantize_act 672) |

Tests: `tests/test_verify_bundle_smolvla.py` (helpers; real tiny capture with a never-called fallback site grafted on: passes when declared,
fails when undeclared, fails when a live site is declared unused; attention pair mismatch and a ghost attention site fail; the real SmolVLA
bundle when present), plus `tests/test_tile_flow_defects.py` and `tests/test_pack_plan.py`.
