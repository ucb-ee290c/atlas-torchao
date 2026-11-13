# Pack scale range: options to bring every activation pack scale into the compiler's admitted range

Facts only. Every number below comes from a command listed in section 8 or from a `file:line` that was read. A statement that is an
inference (not measured or read) is marked **[inference]**. Scope: the four `atlas_v1_awq_tile_static` bundles under
`/scratch2/lorenhung/atlas-fp8/exports-tile/{tinyllama,smolvla,multitask-dit,pi0}/`; quality numbers are TinyLlama only.
Nothing in `src/` was changed.

Conventions. A static activation exponent `s` means real value = `e4m3(byte) * 2^s` (`src/ref_numerics.py:4-6`, `:34-68`), i.e. the
pack divides the BF16 value by `2^s`. E8M0 *pack code* `c = 127 + s` (`policy.json` `e8m0_code`). Admitted by the compiler: `122 <= c <= 128`,
i.e. `s` in -5..+1. The MXU pop multiplies, so the pop code that realizes the same quantization is `254 - c` (`lib/AtlasGuardedPack.cpp:95`).

## 1. Summary

1. Moving a pack exponent while moving the value with it is **bit-exact**: `pack(2^k * x, s + k) == pack(x, s)` byte for byte
   (BF16 scaling by a power of two is exact; the pack output depends only on `x * 2^-s`). Measured on all 92 out-of-range TinyLlama
   Linear inputs (116,654,080 elements): 0 differing bytes. The consumer's dequant / pop exponent stays at the original `s`.
   Only *clamping the exponent while leaving the value* changes the quantization.
2. All 114 / 244 / 88 / 127 out-of-range static sites (TinyLlama / SmolVLA / Multi-Task DiT / Pi0) can be fixed exactly by (1). 92 / 221 / 24 / 61 of
   them sit behind an op that already multiplies by a constant (norm gamma, `up_proj` pop rescale, PV rescale), so the fold costs no instruction.
   The attention p sites (22 / 13 / 24 / 63) need one `vmul` on the broadcast `recip(sum)` per (head, 32-row block) because the flow's p is normalized
   (the kit's free `* 256` belongs to the unnormalized design, which changes numerics). The rest need one `vmul.bf16` per 32x32 tile, an activation-constant rewrite, or a host-side scale (table in 4.A).
3. The production kit never uses `vpack` for activations (only `resnet_pool` does, at code 127). It clamps to +-448*2^s, pushes BF16 into an
   accumulator and pops FP8 with `seli e, 127 - s`. That path has no range check in the compiler (`AtlasVirtualToMachine.cpp:773-783`;
   scale constants admit 0..255, `AtlasOps.cpp:246-249`). The compiler's own `--lower-atlas-guarded-pack` rewrites a pack into that path, but only when it can prove the
   source finite; a `vmul` between the matmul and the pack already defeats it (measured, 4.G).
4. Dynamic scales cannot be expressed at all: `scale_code` is a compile-time attribute (`AtlasOps.cpp:293-294`) and the FP8 readout requires a
   `virtual_scale_constant` (`AtlasVirtualToMachine.cpp:773-777`). Attention q, k, v in the "static" recipe are dynamic: on TinyLlama real text 1060/1408 q,
   48/1408 k and 1408/1408 v (layer, batch, head) exponents are outside 122..128 (4.H).
5. Clamping all static exponents into 122..128 on TinyLlama gives perplexity 12.8756 (unclamped static export 12.8980, BF16 12.7558, 16 windows x 512; Linear-only clamp 12.8609, p-only clamp 12.9027)
   but flushes 4.36% instead of 0.92% of the nonzero values at the affected Linear inputs and 60% instead of 29% of attention p entries (4.B). It changes the quantization.
6. Weights are never packed; unaffected (4.E).

## 2. Evidence read

### 2.1 Compiler admission
- `atlas-mlir/lib/AtlasVirtualToMachine.cpp:566-575` `lowerPack`: rejects `scale_code < 122 || > 128` ("virtual-to-machine pack admits observed E8M0 scale codes 122..128 only"), else `seli e3, code` + `vpack.bf16.fp8` + a 32-row scalar relayout loop (`:577-628`).
- Same range check also at: `tools/atlas_virtual_evaluator.py:251-252`, `tools/compile_atlas_linalg_tile.py:361-364`, `python/atlas_mlir_native/pack_lane.py:24`, `binding.py:77-79,178,559,640,2224,2632,2640,2893,2907`, `reference.py:473,503`, `graph_precision.py:137`.
- Dialect verifiers accept 0..255 for both `pack_fp8` and `scale_constant` (`lib/AtlasOps.cpp:246-249`, `:293-294`; `test/test_virtual_mxu.py:145`).
- "observed" has no stated hardware reason in any file read: `docs/hardware-status.md:99` says only "Limited ... Only powers of two and zero checked"; `docs/native-merlin-binding.md:199` "122-128 are accepted".
- `VMATPOP.FP8` is listed "Untested" at `docs/hardware-status.md:52` although the kit attention and rmsnorm launches that use it are in `atlas-kernels/coverage/RTL-RESULTS.md:8-9` (Exact = yes, standalone AtlasCore Verilator vs npu_model).

### 2.2 RTL semantics (`atlas-npu.git` @ `0079c0541111197741a231c002e3843fa6f545b2`, read with `git show`)
- `src/main/scala/atlas/vector/VectorEngine.scala:53-59` `e8m0ToScaleExpClamped`: `shift = clamp(code - 127, -128, 127)`. No special code: code 0 is shift -127, code 255 is +128 clamped to +127 (aliases 254; the compiler says the same at `AtlasGuardedPack.cpp:94`). `src/ref_numerics.py:25-31` records the same.
- `FP8Pack.scala` (read lines 1-100): `expAdjusted = unbiased_exp - expShift`; round-nearest-even on a 4-bit significand; `> 8` saturates to 0x7E/0xFE; `< -6` flushes to 0 (no E4M3 subnormals); BF16 zero/subnormal/NaN go to 0; Inf to max. `src/ref_numerics.py:e4m3_vpack` is the bit-equal port, so the quantization numbers here (flush below `2^(s-6)`) have the RTL flush semantics.
- MXU pop uses `FPUtils.bf16ToE4m3(bf16, scaleE8M0)` with the same E8M0 clamp (`mxu/FPUtils.scala:141-157`). `docs/rtl-timing.md:107` (npu-model): "VPU pack divides by the scale; MXU pop multiplies"; the pop converter differs at the rounded 0x7F encoding, hence the kit's clamp-before-pop (`atlas-kernels/kernels/epilogue/atlas_epilogue.py:231`: "Clip before MXU pop: its RTL rounding can otherwise encode 480 as NaN").
- `vmatpop.bf16.acc` has no scale operand (`npu_spec/06_instruction_set/README.md:370` vs `:368` for `.fp8`), the compiler lowers it with `scale_reg=0` (`AtlasVirtualToMachine.cpp:767-771`). A BF16 pop therefore needs a `vmul.bf16` by a tile constant to apply `2^(sx+sw)`.
- `vmul.bf16` rounds to nearest even; `vadd/vsub` chop (`docs/compiler-pipeline.md:151`). Both commute with power-of-two scaling absent under/overflow.
- Lane-level RTL test of `FP8Pack` covers shifts 0, +2, +10, -10 (`src/test/scala/atlas/vector/RVFP8PackTest.scala:234-295`), i.e. codes 127, 129, 137, 117. Full-system test: `test_vpu_e8m0_pack_reference.py:158-196` (cosim of the selected core, env-gated, codes 128 and 126; not run here). `atlas-npu-rtl/.../gen_vpu_pack.py:30,33` uses 0x80 for pack and 0x81 for unpack.
- Software model only (not RTL): `atlas-mlir/test/test_guarded_pack_exactness.py:60-88` checks all 65,280 finite BF16 words x all 256 codes, for the model's pack and for clamp + MXU pop at code `254 - c`, against an independent oracle. **Run here** against `atlas-bringup/npu_model` @ `47b8599`: both tests pass (23.6 s; `ATLAS_PACK_MODEL_ROOT=... python -m unittest discover -s test -p test_guarded_pack_exactness.py` with `PYTHONPATH=python`). No receipt file for this test exists in the repository.

### 2.3 What the kit does
- Every activation quantization in the kit is `clip(+-448*2^s)`, `vmatpush.acc.bf16`, `seli e, 127 - s`, `vmatpop.fp8.acc`: attention q/k/v/o (`atlas_attention_v2_program.py:455-462,477,586`), epilogue FP8 output (`atlas_epilogue.py:231-239`), rmsnorm output via that epilogue (`atlas_rmsnorm_postpass.py:103`, default exponent -8, allowed -126..119 at `:32`). `vpack.bf16.fp8` appears only at `atlas_resnet_pool.py:154`, with `e0 = 127` (`:124`).
- Attention p: `exp(s - max)` (unnormalized, <= 1) `* 256` (`vmul`, `atlas_attention_v2_program.py:550-551`), pop at unit code `e3 = 127`; the denominator is the sum of the *packed* p codes (matmul against FP8 ones, `:532-533,555`), and the output is `PV * recip(sum) * 2^v_exp` (`:560-563`). The p scale cancels, so `p_exponent = -8` is just the `* 256`.
- The RTL-verified attention case (`lib/rtl_cases.py:109`, seed 7, `lib/rtl_cases.py:15`) uses q/k/v exponents -7/-6/-6 (recomputed: `python kit_exps.py`, same draw order as `kernels/attention/reference.py:56-66`), i.e. pop codes 134/133/133 and pack-equivalent codes 120/121/121, all below 122. It ran Exact on RTL (`RTL-RESULTS.md:8`). The o exponent was not recomputed.

## 3. Census (measured, `policy.json` and `fx/graph.txt`)

| | TinyLlama | SmolVLA | Multi-Task DiT | Pi0 |
|---|---|---|---|---|
| static activation sites (policy.json) | 176 | 316 | 203 | 483 |
| of which attention p | 22 | 13 | 24 | 63 |
| outside 122..128 (matches `compiler_pack_scale_admission.sites_outside`) | 114 | 244 | 88 | 127 |
| below 122 / above 128 | 114 / 0 | 244 / 0 | 88 / 0 | 126 / 1 (code 133) |
| Linear/Conv inputs outside | 92/154 | 231/303 | 64/179 | 64/420 |
| attention p outside (static -8, code 119) | 22/22 | 13/13 | 24/24 | 63/63 |
| code range of all static sites | 117..126 | 117..125 | 119..126 | 117..133 |
| `\|k\|` to the nearest in-range code, max | 5 | 5 | 3 | 5 |
| `k = 127 - c` to unit code 127, range | 1..10 | 2..10 | 1..8 | -6..10 |
| source tensors packed / distinct (tensor, code) packs outside | 110 / 79 | 227 / 181 | 155 / 88 | 321 / 110 |
| shared-input groups with differing codes | 0 of 44 | 2 of 76 (spread 1) | 0 of 24 | 0 of 99 |
| `mode: dynamic` Linear | 0 | 1 (`vlm.lm_head`) | 0 | 0 |
| weight e8m0 code range (stored, not packed) | 114..123 | 115..124 | 113..119 | 110..129 |
| pop exponent `sx + sw` range (bound from per-site ranges) | -23..-9 | -21..-6 | -21..-9 | -21..-1 |

Dynamic attention q/k/v (not in the 122..128 counts above): `graph_inspection.json` fixture codes outside 122..128: 47/66, 39/39, 72/72, 132/189 sites. TinyLlama real text,
2 x 256 tokens (`pack_range_probe.py`): q 1060/1408 outside (codes 118..122), k 48/1408 (120..123; 4 KV heads repeated by GQA expansion), v 1408/1408 (116..121). The dynamic
`a=block` export has 2080/2464 blocks and 149/154 Linear sites with a block outside (`tinyllama/atlas_v1_awq_tile/dynamic_pack_admission_real_text.json`, from `dyn_admission.py`).

Producers of the packed tensor, out-of-range sites per class (`fx_pack_census.py`; totals equal the policy counts above):

| producer of the packed tensor | TinyLlama | SmolVLA | Multi-Task DiT | Pi0 |
|---|---|---|---|---|
| RMSNorm gamma or `(1+w)` multiplier (`mul(param,mul)` / `mul(mul,add)`) | 59 | 145 | - | 24 |
| LayerNorm | 0 | 0 | 0 | 3 |
| attention p (softmax) | 22 | 13 | 24 | 63 |
| attention output, `pv_matmul` | 22 | 44 | 24 | 27 |
| attention output, SDPA | - | - | 6 | - |
| gated product `act(a) * b` (SiLU/GELU) | 11 | 24 | - | 7 |
| `x * sigmoid(.)` product (DiT vision/text encoder MLP) | - | - | 17 | - |
| bare GELU / SiLU | - | 6 | 13 | 1 |
| linear output into a linear | - | 9 | - | 0 |
| modulation add | - | 1 | 1 | - |
| sin / index (DiT time embedding, text projection) | - | - | 2 | - |
| raw model input (image patches, state, noise, noisy actions) | - | 2 | 1 | 2 |

Non-pack consumers of a producer (so a scaling of the producer would leak): SmolVLA 1 linear output also feeds `expand_as`, 8 `empty_like` results (all in range) also feed `copy_`;
Pi0 raw `noise` also feeds an `add`, 1 linear output feeds `expand_as` (in range). Every other packed tensor has only pack consumers (`fx_pack_census.py`).

## 4. Mitigations

"Sites fixed" are policy.json static sites out of range (TinyLlama / SmolVLA / DiT / Pi0), totals 114 / 244 / 88 / 127.

### 4.A Power-of-two fold into the producer (exporter plan, lowered at unit code 127)

Choose a code `c'` in 122..128 (admitted today) with `k = c' - c`; 127 is the default used below. Evidence per code: 127 is the code of the RTL-exact `resnet_pool` launch (`RTL-RESULTS.md:14`; its data are unpacked FP8 values, so that pack is a round trip, not general BF16) and of the compiler's unit-pack policy (`binding.py:75-76`, input domain "seven powers of two and zero"); the general-value system-level pack test uses 126 and 128 (`test_vpu_e8m0_pack_reference.py:185`, not run here). Pick 126 or 128 if general-value system evidence at the chosen code is wanted. Multiply the BF16 value by `2^k` at its producer, pack at `c'`.
Exact because (i) BF16 scaling by `2^k` is exact for normal results (`|k| <= 10` here; measured exact at all 92 sites), (ii) `vmul` rounds RNE and `vadd` chops, both scale-equivariant, (iii) the pack output depends only on `x * 2^-s`.
The FP8 bytes are unchanged, so the pop / dequant exponent stays `s` and the FX graph numerics are unchanged.

| producer class | where the `2^k` goes | cost | sites fixed (T / S / D / P) |
|---|---|---|---|
| RMSNorm / LayerNorm / `(1+w)` | gamma (and beta for LayerNorm) scaled by `2^k`; for `(1+w)` the product `2^k(1+w)` is formed in fp32 as the graph does | 0 | 59 / 145 / 0 / 27 |
| `act(a) * b` | `b`'s producer: add `k` to every `weight_scale_exp` of `up_proj` (codes unchanged, pop constant `2^(sx+sw)` already exists) | 0 | 11 / 24 / 0 / 7 |
| attention p | **[inference]** `p = exp(s - max) * recip(sum)` is normalized and `recip(sum)` is data, so no existing constant absorbs `2^k`: one `vmul` on the `recip(sum)` broadcast per (head, 32-row block), reused by every p tile of that block (the kit's `* 256` is free only in its unnormalized design, 4.C2). p <= 1 so `p * 2^8` <= 256 cannot overflow | 1 vmul per row block, amortized | 22 / 13 / 24 / 63 |
| attention output (`pv_matmul`) | the PV BF16-pop rescale constant `2^(sp+sv)` (kit: `recip(sum) * 2^v_exp`, `atlas_attention_v2_program.py:560-563`) | 0 | 22 / 44 / 24 / 27 |
| linear output into a linear | producer `weight_scale_exp += k`, only if the output has no other consumer | 0 (8 of 9 SmolVLA) | 0 / 8 / 0 / 0 |
| bare GELU/SiLU, `x * sigmoid(c x)` | **[inference]** rewrite the activation's internal constants (tanh-GELU final `0.5`, SiLU first `-1`, sigmoid `1.702`) and scale the producer by `2^k`; equivariance is algebraic, not run | 0 if confirmed | 0 / 6 / 30 / 1 |
| linear->linear with a second consumer, modulation add, SDPA out, sin, index | one `vmul.bf16` per 32x32 tile before the pack | 1 VPU op / tile | 0 / 2 / 9 / 0 |
| raw model input | host scales before the f32 to BF16 conversion | host | 0 / 2 / 1 / 2 |
| total | | | 114 / 244 / 88 / 127 (no added instruction: 92 / 221 / 24 / 61) |

Census checks (`fx_pack_census.py`): all 181 Pi0 `mul(mul,add)` sites have a parameter-only `add` operand (Gemma-style `1 + w`, not a conditioning output), so the 24 out-of-range ones are parameter folds; every LayerNorm site (49 + 96 + 109) carries weight and bias arguments, including the 3 out-of-range Pi0 ones.

Where it must live: **exporter** records the plan; **flow** applies the edits or leaves the bundle untouched; the producer constants are device weights, so baking them into `model.safetensors`
changes the graph (a packed tensor is `2^k x` its logical value), which requires the FX op to carry a pack shift separate from the dequant exponent (today `quantize_act` returns one exponent used for both: `atlas_fx_ops.py:71-92`).
Cheaper first step: leave the export and graph untouched, add the plan to `policy.json`, and let the lowering apply it.
Cost: low for the free classes (the AWQ fold already rewrites norm gamma and `up_proj` rows, `howto-model-to-fx-graph.md` section 7). Risk: low numerics (bit-identical, tested), medium bookkeeping (the 2 SmolVLA groups with differing codes need two packs, spread 1 so both fold to 127; producers with non-pack consumers must be excluded).

### 4.B Clamp the static exponents into 122..128 (`s' = clamp(s, -5, 1)`), value unchanged

Changes the quantization. Measured on TinyLlama real text (2 x 256 wikitext-2 test tokens, hooks on the original export; `pack_range_probe.py`, derived from `scripts/tile_flow/static_clip.py`):

| | original exponent | clamped | note |
|---|---|---|---|
| Linear inputs changed | - | 92 of 154 | all raised (all out-of-range codes are below 122 in TinyLlama), so no new saturation |
| saturated values at those 92 sites (byte 0x7E, i.e. `x / 2^s >= 432`, includes values that round to 448) | 85 | 0 | `static_clip.py`'s stricter `x > 448 * 2^s` count over all 154 sites was 6 values at 3 sites (recorded in `static_activation_saturation_real_text.json`); that file's flush figure (mean 1.15% of *all* elements, zeros included) uses another denominator than the nonzero-element fraction below |
| flushed (nonzero to 0), fraction of 116,614,478 nonzero | 0.917% | 4.356% | by `k`: 1.0 to 1.9%, 1.8 to 5.5%, 0.2 to 1.5%, 0.7 to 7.4%, 0.4 to 12.4% for k = 1, 2, 3, 4, 5 |
| per-site decoded relative L2 vs the BF16 input, mean / max | 0.0266 / 0.0276 | 0.0270 / 0.0383 | worst `layers.0.self_attn.o_proj` (k = 5): flush 1.2% to 34.1% |
| attention p (22 layers), flushed fraction of nonzero p, mean / max | 29.1% / 47.9% (s=-8) | 60.1% / 79.6% (s=-5) | threshold `2^(s-6)` |
| p probability mass lost | 7.5e-4 mean (1.1e-3 max) | 8.7e-3 (1.3e-2) | |
| `p @ v` relative L2 vs fp32 `p @ v` (v unquantized), mean / max | 0.0205 / 0.0244 | 0.0296 / 0.0500 | |
| perplexity, 16 windows x 512 (bf16 12.7558) | 12.8980 (rerun reproduces the recorded `ppl_w16x512.json` value to every printed digit) | all: 12.8756; p only: 12.9027; Linear only: 12.8609 | `ref_conv`, exact fp32 accumulation, 16 windows, same windows as `ppl_w16x512.json` |

Reading: the eval is deterministic (rerun identical), so these differences are not run-to-run noise, but the eval JSON has no per-window losses, so the spread over windows is not measured. As measured, clamping the Linear exponents lowered perplexity by 0.037, clamping p raised it by 0.005, clamping both lowered it by 0.022, against a 0.142 gap between the unclamped export and BF16. The per-value damage (flush, error) is concentrated in a few sites and in p; perplexity did not register it.
Where: exporter only. Cost: none. Risk: quality (unmeasured on SmolVLA, DiT, Pi0; for Pi0 one site is above 128 and clamping *down* would saturate: not measured).

### 4.C Attention p

| option | FP8 bytes vs today | PV pop | status |
|---|---|---|---|
| C1. p * 2^8 (one `vmul` on the `recip(sum)` broadcast per row block), pack at 127 | identical to static -8 / code 119 (probe check, TinyLlama real text, `fold_p_x256_unit_bytes_identical`: 22 of 22 layers) | unchanged | = 4.A |
| C2. kit style: unnormalized `exp(s-max) * 256`, pack at 127, divide by the sum of packed codes | different quantization | output `PV * recip(sum) * 2^v_exp` | emulated on TinyLlama real text (unquantized V): `p @ v` relative L2 0.0122 mean / 0.0171 max vs 0.0205 / 0.0244 for static -8; flushed 24.1% vs 29.1%. Emulation uses `p~ = bf16(p / rowmax(p))`, not the kit's BF16 `vexp` pipeline. Changes the FX graph (row sum, divide) |
| C3. PV pop scale | n/a | **not constrained** by `lowerPack`: BF16 pop has no scale and is followed by a `vmul` with any power-of-two BF16 constant; an FP8 readout takes any code 0..255 (`AtlasOps.cpp:246-249`, lowering has no check `:773-783`) | read |

`-8` is already the largest power-of-two scale for p <= 1 (`256 <= 448`); a larger `s` flushes more (4.B).

### 4.D Widen the compiler admission

RTL source: no special code; every code 0..255 is a defined shift (2.2). Evidence for other codes: lane-level RTL at shifts 0, +2, +10, -10 (codes 127, 129, 137, 117); full-system pack tests on general values only at 126 and 128 (127 only in the `resnet_pool` RTL row, FP8-representable data); npu_model software model exhaustive over all 256 codes (run here, passes); the kit's pop path on RTL at pack-equivalent codes 120, 121 (different converter). A range that covers the measured census is 117..133 (static) and 116..137 including dynamic v codes and the lane-level endpoints.

Exact edit list for a compiler owner (all found by `grep -rn 122`):
- `lib/AtlasVirtualToMachine.cpp:568-570` (bounds and message); `tools/atlas_virtual_evaluator.py:251-252`; `tools/compile_atlas_linalg_tile.py:361-364`.
- `python/atlas_mlir_native/pack_lane.py:11,24`; `binding.py:77-79,178,559,640,2224,2632,2640,2893,2907`; `reference.py:35-36,473,503`; `graph_precision.py:137`. The policy strings `...-122-128-v1` (`binding.py:77-78`, `pack_lane.py:11`, `reference.py:35-36`) embed the range, so renaming them changes the target identity.
- Tests that assert the rejection or loop over 122..128: `test_virtual_mxu.py:89-95`, `test_native_binding.py:1448,1518,6059`, `test_graph_precision.py:110-120`, `test_guarded_pack_exactness.py:43`.
- Docs: `dialect-reference.md:48`, `compiler-pipeline.md:154,298`, `native-merlin-binding.md:156,199`, `pack-relayout-loop.md:160`.
- Add RTL runs of `test_vpu_e8m0_pack_reference.py` at the new endpoints (for example 117 and 133).

Where: compiler. Cost: low code, but needs RTL evidence for the new codes. Risk: the 122..128 range is "observed", not derived; **[inference]** a failure at a new code would be silent wrong bytes, not a compile error.

### 4.E Weights

Weights are stored FP8 bytes plus a per-32-channel-panel exponent and never pass through pack: in the compiler they are `virtual_input_fp8` DMA inputs (`test/examples/virtual_fp8_two_layer_mlp.mlir:9-11`) consumed by `mxu_load_weight`, whose signature is state, FP8, `unit` with no scale operand (`docs/dialect-reference.md:51`, `AtlasOps.cpp:251-256`); in the kit they are loaded as FP8 bytes. The weight exponent only enters the pop constant `2^(sx+sw)`. Measured weight code ranges 110..129 (table above) are therefore not subject to 122..128, and no weight carries 0x7F/0xFF (`reserved_codes` = 0 at every site, all four models). Sites affected: 0.

### 4.F Compiler-side expansion (new)

For an out-of-range code `c`, lower `pack(x, c)` as `vmul.bf16 x, splat(2^(c'-c))` then `pack` at `c'`. Exact (same argument as 4.A), no flow change, all 114 / 244 / 88 / 127 sites. Cost: one `vmul.bf16` per 32x32 tile (plus a splat, hoistable) next to a pack whose scalar relayout loop already takes about 29 cycles a row (`pack-relayout-loop.md:65,134`). Where: compiler (`lowerPack`). Risk: low. It is the universal fallback for the classes that 4.A cannot fold for free.

### 4.G MXU pop as the pack (new; what the kit and `--lower-atlas-guarded-pack` do)

Measured with `atlas-mlir/build/bin/atlas-opt` on `virtual_fp8_two_layer_mlp.mlir` with `scale_code = 119`:
- `--lower-atlas-virtual-to-machine --insert-atlas-dma-waits`: error "pack admits observed E8M0 scale codes 122..128 only".
- `--atlas-compile=guarded-pack=true`: rc 0; the pack became `max/min` clamp splats (bits 0xBFE0 / 0x3FE0 = -+1.75 = 448 * 2^-8), `virtual_mxu_load_acc_bf16`, `virtual_scale_constant` code 135 (`254 - 119`) and `virtual_mxu_readout_fp8`; no `bf16_to_fp8` pack remains.
- The same program with a `vpu_binary mul` inserted between the ReLU and the pack: rejected again, with or without guarded-pack.
`provenFinite` (`AtlasGuardedPack.cpp:15-38`) accepts splats, matmul results, BF16 readouts of reset or loaded accumulators, `mov`/`relu`, and `min`/`max` of finite operands. **[inference]** every packed tensor in the four graphs passes through at least one multiply, exp or add after the matmul (norm, gated product, softmax, residual or modulation add), so none is covered today.
**[inference]** Extending the compiler to treat a producer as finite (or to accept an exporter-declared finiteness bound, which the clamp-before-pop then makes safe) would route all sites through a path with no range limit. Where: compiler. Cost: medium. Risk: medium (pop converter differs from pack at 0x7F; clamp required).

### 4.H Not a range problem but a blocker: attention q, k, v are dynamic (new)

Counts in section 3. A dynamic exponent has no `scale_code` immediate. Making q, k, v static (calibrated) puts them into the same fold analysis: **[inference]** q, k and v are linear in their projection outputs (RoPE is linear), so `k`-folds go into the projection `weight_scale_exp` at no cost and the score rescale `2^(sq+sk)` is unchanged. Where: exporter (`static_calib` extension) and flow.

## 5. Summary per mitigation and recommendation ordering

| mitigation | sites fixed (T / S / D / P) | exactness | where | cost | risk |
|---|---|---|---|---|---|
| A. fold into producer, pack at 127 | 114 / 244 / 88 / 127 (no added instruction: 92 / 221 / 24 / 61) | lossless (bytes identical; bit test on 92 TinyLlama sites) | exporter plan + flow/lowering | 0 for free classes, 1 vmul/tile otherwise | low; producers with other consumers excluded |
| B. clamp exponent | 114 / 244 / 88 / 127 (changes numerics at every one) | changes quantization (flush 0.9% to 4.4%, p flush 29% to 60%) | exporter | 0 | quality unmeasured outside TinyLlama; perplexity 12.8756 with all clamped (unclamped 12.8980) |
| C1. p * 2^8 then pack at 127 | 22 / 13 / 24 / 63 | lossless (bytes identical, 22 of 22 layers) | flow (vmul on `recip(sum)`) | 1 vmul per (head, 32-row block) | low |
| C2. kit-style unnormalized p | 22 / 13 / 24 / 63 | changes quantization (emulated: lower PV error) | flow + FX graph | row-sum matmul + divide | medium |
| D. widen admission | all (needs 117..133) | lossless if the RTL honors the shift (model: all 256 codes pass) | compiler | one line plus tests/docs | RTL evidence only at 126..128 (system) and +-10 (lane) |
| E. weights | 0 affected | n/a | none | 0 | none |
| F. compiler vmul + pack at in-range code | 114 / 244 / 88 / 127 | lossless | compiler (`lowerPack`) | 1 vmul/tile | low |
| G. MXU pop pack (guarded) | 0 today (no packed tensor is provably finite) | lossless where applicable (model: all 256 codes) | compiler | medium | medium (0x7F handling, clamp) |
| H. static attention q/k/v | 0 of the counts above (they are dynamic) | n/a | exporter + flow | calibration | quality |

Ordering:

1. **4.A for the free classes plus 4.F as the universal fallback.** Exact; no change to the admitted range, no new hardware evidence (every pack stays inside the admitted 122..128); the exporter only records a plan, FX numerics stay bit-identical. Covers every out-of-range site: no added instruction for 92 / 221 / 24 / 61, one `vmul` per p row block for 22 / 13 / 24 / 63, activation-constant rewrite **[inference]** for 0 / 6 / 30 / 1, one `vmul` per tile for 0 / 2 / 9 / 0, host for 0 / 2 / 1 / 2.
2. **4.D widen the compiler**, once the two RTL runs at the new endpoints pass; this removes the need for any fold but trades exactness-by-construction for hardware evidence.
3. **4.G** extend `provenFinite` or add a declared-bound opt-in; it is the production kit mechanism and has no range limit.
4. **4.B clamp** only as a stop-gap: it is the only option that changes numerics, and its cost is concentrated in p and in k >= 4 sites.
5. Separately, **4.H**: calibrate attention q, k, v statically, otherwise they remain uncompilable whatever the range.

## 6. Exporter-side check spec

Run after `quant_export` and again on the FX bundle. Inputs: `policy.json`, `fx/graph.txt`, fixtures or calibration activations. Output: a `pack_lowering` block per static site in `policy.json` and in `graph_inspection.json`, and a failing exit for the hard checks.

For every site with `activation.mode == "static"` and every attention operand `p` with `mode == "static"`:
1. `code = 127 + exp`. Admission constants (`122`, `128`) come from one named constant, not a literal.
2. `in_range = 122 <= code <= 128`. If not: `k = c' - code` with `c' = 127`; hard-fail if `|k| > 20` or if `amax * 2^k` exceeds the BF16 maximum (never happened: `k` in -6..10).
3. Producer class from the graph by walking back through `to/view/reshape/contiguous/transpose/expand/slice/getitem` (`fx_pack_census.py` does this): one of `norm_affine, layer_norm, attn_p, attn_out_pv, attn_out_sdpa, gated_product, x_sigmoid_product, bare_activation, linear_to_linear, modulation_add, raw_input, other:<op>`. Record `placement` (`norm_gamma`, `weight_exp:<producer site>`, `softmax_multiplier`, `pv_rescale`, `act_constant`, `vmul`, `host`).
4. `other_consumers`: leaves of the producer's transitive users other than `quantize_act`; non-empty means the producer cannot be scaled in place (fail the free placement, fall back to `vmul`).
5. Shared-input groups (several `quantize_act` on one source tensor): codes identical, else record the distinct packs (SmolVLA: 2 groups, spread 1).
6. Dynamic sites (`mode: dynamic` Linear, and attention q/k/v with `source: reference`): list them as `not_expressible_as_scale_code` (SmolVLA `lm_head`; all attention q/k/v). Real-text codes: report the fraction outside from a calibration pass.
7. Differential fold test, per out-of-range site on the site's actual input `x` (a hooked replay of the fixture or calibration cases; `fx/fixtures.safetensors` holds only the graph's boundary inputs): `e4m3_vpack(x_bf16, s) == e4m3_vpack(bf16(x * 2^k), s + k)` for all elements and `bf16(x * 2^k) == x * 2^k` in float32. Hard-fail on any difference (TinyLlama: 0 of 116,654,080).
8. Clamp report (informational): per site flush fraction and saturated count at `s` versus at `clamp(s)`.
9. Weights: informational range only (no admission); keep the existing 0x7F/0xFF failure. Pop exponent `sx + sw` must satisfy `-126 <= sx + sw <= 127` (BF16 constant) or `0 <= 127 + sx + sw <= 254` (FP8 pop code): measured -23..-1.
10. Schema: today `quantize_act` returns one exponent used for both the pack and the dequant (`atlas_fx_ops.py:71-92`); a baked fold needs an explicit `pack_shift` attribute. Not required if the plan is only recorded.

## 7. Not measured / not verified

- Quality of any clamp or fold on SmolVLA, Multi-Task DiT, Pi0; only TinyLlama 16 x 512 perplexity and 2 x 256 token per-site statistics.
- No RTL or hardware run of any option. 4.A's exactness is algebraic plus the bit test on the encoder (`ref_numerics.e4m3_vpack`); the activation-constant rewrites (SiLU, GELU, `x * sigmoid`) were not run.
- The kit attention o exponent, and the kit's BF16 `vexp` pipeline in 4.C2 (emulated with `bf16(p / rowmax(p))`).
- Provenance of the TinyLlama runs: repository HEAD 3384385 (parent 29714ee) with uncommitted edits by other sessions in the working tree at run time (`src/eval.py`, `src/quant_export.py`, `src/tile_native.py`, `src/eval_pi0.py`, `src/eval_policy.py`); the unclamped rerun reproduced the recorded perplexity exactly, which is the only check that those edits did not move the numbers.
- Graph node counts exceed policy site counts for SmolVLA (339 static `quantize_act` nodes against 316 sites); all tables use policy sites (distinct consumer weight or p tag), which match `compiler_pack_scale_admission`.
- `docs/algos/tile-aware.md` was edited by another session while this was written; not read again, not touched.

## 8. Commands

```bash
# census (policy.json and fx/graph.txt, read-only)
python3 docs/algos/pack-range-options/fx_pack_census.py
# real-text probe on the TinyLlama static export: fold exactness, clamp flush/saturation, p options, dynamic q/k/v codes
export PYTHONPATH=src PY=/scratch2/lorenhung/envs/atlas-lerobot/bin/python
CAP="taskset -a -c 0-15 nice -n 10 env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8"
$CAP $PY -u docs/algos/pack-range-options/pack_range_probe.py /scratch/agarcia05/TinyLlama-1.1B-Chat-v1.0 \
  /scratch2/lorenhung/atlas-fp8/exports-tile/tinyllama/atlas_v1_awq_tile_static/export OUT.json 256 2      # result: pack-range-options/probe_tinyllama.json
# clamped perplexity: policy.json copies with exp = clamp(exp, -5, 1) (Linear and/or p), manifest sha updated, model.safetensors symlinked
$CAP $PY -u src/eval.py --model /scratch/agarcia05/TinyLlama-1.1B-Chat-v1.0 --seqlen 512 --max-windows 16 --device cpu \
  --from-export CLAMPED_DIR --modes ref_conv --out OUT.json --mxu ipt --crosscheck off
# compiler
atlas-mlir/build/bin/atlas-opt P.mlir --lower-atlas-virtual-to-machine --insert-atlas-dma-waits   # rejects 119
atlas-mlir/build/bin/atlas-opt P.mlir --atlas-compile=guarded-pack=true                            # accepts 119 for relu(matmul), rejects with a mul
# software-model exhaustive pack test, all 256 codes
cd atlas-mlir && PYTHONPATH=python ATLAS_PACK_MODEL_ROOT=../npu_model python -m unittest discover -s test -p test_guarded_pack_exactness.py
```
