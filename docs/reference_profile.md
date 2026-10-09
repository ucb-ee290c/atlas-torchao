# The reference FP8 profile

This page describes the profiles that follow the reference author's Atlas FP8 numerics
(`third_party/pi0-quant` at 8bc0f6a, `third_party/smolVLA-quant` at 3373632). It gives the source
of every numerics decision. **NEW** marks a decision with no source, with the reason for it.
Reference numerics win unless the RTL clearly contradicts them; the RTL cited is atlas-npu 0079c05.

## Profiles

| Profile | Operand cast and scale | Accumulation | Other ops | Source |
|---|---|---|---|---|
| `bf16` | none | framework | framework | - |
| `collab_v0` (alias `atlas`) | ceil(log2(amax/448)) weight scale, fixed 2^0 activation scale, torch fp8 activation cast, pop-path weight codec | `F.linear` | framework | the collaborator: atlas_torchao.py, atlas_quant.py; byte-identical to daf8e78 (tests/test_profiles.py) |
| `collab_v0_hw` (alias `atlas_hw`) | as `collab_v0`, hardware activation cast | `F.linear` | framework | atlas_torchao.py:57-61 |
| `reference` | reference rule for x and W, dynamic per call | `F.linear` on dequantized values (NEW) | framework | hw_float_ops.py:38-57,64-103; quant_linear.py:153-172 |
| `reference_hw` | as `reference` | MXU model, `--mxu ipt` (default) or `sa` | framework | quant_linear.py:145-172; functional_models.py:79-108 |
| `chip` | as `reference_hw` | as `reference_hw` | VPU (bit-exact to `VectorRTLFunctions`), scalar core, or reported host fallback | src/chip_profile.py, src/op_unit_map.py |
| `torchao` | torchao per-tensor float8, dynamic | native FP8 matmul (CC >= 8.9) | framework | torchao 0.18 |

`reference`, `reference_hw` and `chip` cover the ops chosen with `--ops` (default
`linear,conv2d,attention`, as in smolVLA run_eval.py:7-18).

Linear layers: every `nn.Linear` is covered, fp32 ones included (model_patcher.py:83-87, 496-499). The Llama
eval skips `lm_head`, which was the collaborator's choice (eval.py:66).

`collab_v0` keeps its bf16-only layer filter.

## Numerics decisions

1. **Scale rule.** `s = floor(log2(amax_bf16 / 256))`.
   - `nan_to_num` is applied to `|x|` first, and an all-zero tensor gets `s = 0` (hw_float_ops.py:31, 54-57).
   - The port reads the exponent from the float32 amax. This is an implementation detail; it equals the reference for every BF16 amax (tests/test_ref_numerics.py).
2. **Activation scale.** The scale is dynamic, per call, and taken over the whole tensor, the same rule as for weights (quant_linear.py:153-154).
3. **Cast for every role: vpack / FP8Pack** (fp8_e4m3.py:38-86; FP8Pack.scala:57-95).
   - Divide by 2^s, then round to nearest even.
   - Saturate to 0x7E, and clamp 480 (0x7F) to 0x7E.
   - Flush results below 2^-6 to zero.
   - BF16 subnormals and NaN become 0; ±Inf becomes ±0x7E.
   - The collaborator's pop-path codec differs only by emitting 0x7F on 30 inputs per shift. It is equal after the clamp.
4. **Decode.** Exponent 15 is finite. Only 0x7F/0xFF are reserved; they decode to signed zero, as do subnormals (fp8_e4m3.py:89-121; FPUtils.scala:120-138).
5. **Bias.** It stays BF16 and is added after the rescale (quant_linear.py:164-172).
6. **Rescale.** The MXU's BF16 output is multiplied by 2^(sx+sw) in float32, which is exact, then cast to the layer dtype (quant_linear.py:169-170).
7. **fp32 layers** are cast through BF16 (quant_linear.py:136-137; hw_float_ops.py:89).
8. **MXU models.** The torch ports in src/accum_torch.py use pure integer bit-field arithmetic. They are bit-exact to:
   - `ipt`: ipt_numba, i.e. `IPTLinearRTLFunction(32, 32)` in _numba_kernels.py:234-317. The 32-bit aligned total is rounded to BF16 once, as the RTL does: FPUtils.scala:228-236 uses `INToRecFN` with `round_near_even`. This is the reference author's run (smolVLA run_eval.py:50).
   - `sa`: systolic_c, i.e. `SARTLLinearFunction(32, 32)`. Each K element goes through one E4M3ProdAddBF16 (systolic_array_model.h:97-233).
   - The reference C models (`--mxu-impl ref_cpu`) are the oracles. With the torch implementation, `--crosscheck` (on by default) recomputes a seeded sample of the run's own calls with them. Any differing bit marks the result `valid: false`.
   - IPT lane count does not change results; 16 and 32 lanes are tested equal.
9. **Attention** is a per-(batch, head) port of model_patcher.py:624-723.
   - GQA is expanded first. Q, K, P and V each get their own scale per head.
   - Scores: MM, then ×2^(sq+sk), then ×head_dim^-0.5.
   - Two variants follow the reference's two paths:
     - **text:** scores and softmax stay float32 and the mask is boolean (675-684).
     - **vision:** scores are bf16, the mask is additive, and softmax is float32 (827-836).
   - Llama uses the text variant with its additive eager mask added (NEW: the closest analogue; modeling_llama.py:213-214).
   - Under `reference`, the score matmuls use dequantized operands (NEW).
10. **Conv2d** is a port of QuantConv2d (model_patcher.py:282-375).
    - Path: im2col, per-tensor casts, MM, rescale, then bias.
    - Layers with `groups != 1` or string padding stay unquantized and are reported.
11. **No E8M0 clamp on s.** The reference does not clamp either; see R2.

## Findings (with evidence)

- **F-IPT: ipt_c and ipt_numba are different models.**
  - ipt_c converts with `(float)int` and then to BF16 (ipt_c/converters.h:244-254). That is a double rounding, and it uses a 30-bit sum.
  - ipt_numba rounds once, matching the RTL.
  - A crafted tile with aligned total 2^24 + 2^16 + 1 gives 65792 in ipt_numba and 65536 in ipt_c (tests/test_accum_backends.py::test_ipt_c_double_rounds_where_numba_and_rtl_round_once).
  - The plan named ipt_c as the IPT oracle. This repo uses ipt_numba.
- **F-cache: the reference models can reuse a stale weight.**
  - They cache the encoded weight keyed on `(data_ptr, shape, stride, dtype, _version)` (ipt_rtl_linear.py:123-145; ipt_rtl_linear_c.py:492-499).
  - A temporary that is freed and re-allocated at the same address with the same shape hits that cache with stale contents. The per-head `v_norm.T.contiguous()` in the reference attention (model_patcher.py:696) is exposed to this.
  - This repo drops the cache for every non-weight operand.
- **F-vision-attn: the reference vision patch may not run under SDPA.**
  - The reference patches the module-level `eager_attention_forward` (model_patcher.py:868), which only takes effect when the vision config uses `eager`.
  - lerobot does not set an attention implementation for SmolVLM.
  - This repo registers a transformers attention key instead, so the routine runs whatever implementation was configured. Whether the reference's vision patch fired in the reference author's runs is not verified here (transformers 4.53.2 there, 5.5.4 here).

## Open questions

For the Atlas hardware team:

- **Q1.** Exponent 15: the spec says [1,14] (software-spec.yaml:18), but the RTL treats it as finite. Should the spec be widened?
- **Q2.** Are dynamic per-tensor activation scales, computed by VPU rmax, intended? The spec has `scale_encoding: unknown`.
- **Q3.** Which feeds the next matmul: `vmatpop.fp8` (480 becomes 0x7F) or `vpack` (480 becomes 0x7E)?
- **Q4.** Is the bias added in BF16 after the rescale, or preloaded into the accumulator (green card:386-389)?
- **Q5.** Which MXU runs which ops: SA (mxu0) or IPT (mxu1)?

For the reference author:

- **R1.** The two encoders and three exponent-15 decoders disagree. Which one is canonical?
- **R2.** There is no E8M0 clamp on s.
- **R3.** Softmax runs in bf16 in pi0 and in fp32 in smolVLA. Which eval set is intended: DROID or svla_so100?
- **R4.** The `QuantLinear` docstring promises format-flag quantization, but the code passes inputs through unquantized.
- **R5.** The weight-cache key can alias freed temporaries (F-cache).
- **R6.** ipt_c double-rounds and uses a 30-bit sum (F-IPT). Should it match ipt_numba and the RTL?

## Must be re-verified on the L40S (CUDA)

This host has no GPU. Everything above was tested with CPU torch, using the same ops. On the L40S, re-check the following:

1. **Bit-exactness on CUDA.** Run `pytest tests/test_accum_torch.py` and `ATLAS_TEST_COMPILE=1 pytest tests/test_accum_torch.py -k compile`.
2. **The benchmark.** Run `scripts/bench_accum.py --device cuda --compile --check --ref`. Its `--check` compares CUDA bits against CPU bits.
3. **The cross-check in every reference_hw/chip result JSON:** `layers.<mode>.crosscheck.ok` must be true with `bit_mismatches == 0`.
4. **BF16 rounding.** The vpack port and the scale rule use `.to(torch.bfloat16)` (RNE) and integer ops, so they should be device-independent. Spot-check them by running tests/test_ref_numerics.py with CUDA tensors.
5. **Determinism of the non-MXU float ops** (softmax, norms and the `reference` F.linear) across two runs.
   - The MXU ports are integer-only, so TF32 and reduction order cannot affect them.
   - `accum_torch.enforce_exact_settings()` is not called, so the rest of the model keeps PyTorch defaults: TF32 is off for float32 matmul but on for cuDNN float32 convolutions.

CPU microbenchmark (from scripts/bench_accum.py; shared 160-core host under load, so rough figures):

| Model | M × K × N | MAC/s |
|---|---|---|
| torch SA, eager | 16×2048×256 | 4.0e6 |
| torch SA, compiled | 64×2048×512 | 3.3e7 |
| torch IPT | 16×2048×256 | 1.5e7 |
| reference systolic_c | 16×2048×256 | 4.1e7 |
| reference ipt_c | 16×2048×256 | 3.7e7 |

The L40S estimates are not measured: SA eager about 2e9 MAC/s, SA compiled 1e10–5e10. TinyLlama is about 0.97 GMAC per token.
