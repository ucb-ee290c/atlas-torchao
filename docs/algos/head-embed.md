# Embedding and LM-head options (`--tie`, `--lm-head`, `--embedding`)

NEW. These options make a causal LM's input embedding and LM head smaller in an export. They were added for
TinyLlama, whose untied BF16 `embed_tokens` and `lm_head` ([32000, 2048], 131,072,000 bytes each) make the
export about 1.23 GB, more than a 1 GiB on-board DDR. The code is in `src/exporter/head_embed.py` and the
tests are in `tests/test_head_embed.py`.

The defaults change nothing. With `--tie none --lm-head bf16 --embedding bf16`, the export has the same
policy, tensors and manifest content as an export written without these options. `policy.json` gains no key
and `new_decisions` gains no entry.

| Option | Values | Effect |
|---|---|---|
| `--tie` | `none` (default), `lm_head_from_embedding`, `embedding_from_lm_head` | Ties the two to one tensor before quantization. TinyLlama is trained untied, so either direction changes the model. |
| `--lm-head` | `bf16` (default), `fp8`, `nvfp4` | `bf16`: the head stays an ordinary BF16 Linear outside quantization (`skip=["lm_head"]`, as before). `fp8`: the skip is removed, and the recipe quantizes the head like every other Linear (GPTQ / AWQ, weight grid, calibrated static activation exponent). `nvfp4`: weight-only NVFP4. |
| `--embedding` | `bf16` (default), `fp8`, `nvfp4` | Storage only. Each lookup gathers rows of the codes and decodes them to BF16. |

With a tie, `--lm-head` and `--embedding` must name the same precision, because one stored tensor serves
both:

- **Tied BF16:** the tensor is stored once. `policy.json` `aliases` names the other parameter.
- **Tied FP8 or NVFP4:** the head's buffers hold the codes. The embedding site (`tied_to: lm_head`) gathers
  rows from those buffers, so no second copy is stored.

Example: tied FP8 TinyLlama with today's recipe.

    python src/eval.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --modes ref_conv --export-only \
        --algo atlas_v1_awq_tile_static --device cpu --export-dir OUT/export \
        --tie lm_head_from_embedding --lm-head fp8 --embedding fp8

## Order of operations (`src/eval.py`)

1. `head_embed.prepare` applies the tie to the BF16 model.
2. `algos.quantize` runs with `skip_for(options)`: `["lm_head"]`, or `[]` with `--lm-head fp8`.
   Calibration, folds and GPTQ see the tied model.
3. `head_embed.finish` installs the NVFP4 head (`NVFP4Linear`) and the quantized embedding
   (`QuantEmbedding`).

The embedding is quantized after calibration, so the decoder layers are calibrated on the BF16 embedding.
With `--tie none` or `lm_head_from_embedding`, the decoder-layer tensors are therefore those of the default
export, byte for byte. The test checks this on a small Llama. With `embedding_from_lm_head`, the embedding
values themselves change, so every layer is re-calibrated.

In a `bf16` mode, the options still apply, which isolates their effect on top of BF16. For `--lm-head fp8` in a
`bf16` mode there is no recipe, so the head uses the reference rule: a per-tensor weight exponent and a dynamic
activation exponent.

## Formats

**FP8 embedding (untied):**

- Codes: E4M3 codes from `ref_numerics.e4m3_vpack`, the Atlas operand encoder.
- Scales: one power-of-two exponent per row, int32 `[V, 1]`, using the never-clip rule
  `granularity.scale_exp(..., "nc")`.
- Row value: `e4m3(code) * 2^exp`, which is exact in BF16.

**FP8 embedding (tied):** the head's codes and its exponent table `[V, 1]`. Under `w=panel`, one exponent is
repeated over each block of 32 rows.

**NVFP4:**

- Codes: E2M1 codes, two per byte, element `2i` in the low nibble. Rounding matches torchao's
  `f32_to_f4_unpacked`, and a test compares the two.
- Block scales: one E4M3 block scale per 16 elements along the hidden dimension, set to
  `e4m3(clamp(block_amax / 6 / tensor_scale, 2^-6, 448))` with RNE rounding.
- Tensor scale: one FP32 tensor scale, rounded up to a power of two, `2^ceil(log2(amax / (448 * 6)))`.
  This is NEW; torchao 0.18 uses `amax / (448 * 6)`. With a power-of-two tensor scale, every decoded value
  `e2m1 * e4m3 * 2^k` is exact in BF16, since `e2m1 * e4m3` has at most 6 significant bits.
- The largest block scale then lies in (224, 448] instead of being exactly 448.
- MXFP4 is not offered.

**NVFP4 head:** weight-only. The activation stays BF16, and the head computes
`F.linear(x, decode(codes))` in BF16, the same BF16 Linear as the unquantized head on an exactly decoded
weight. The Atlas matrix unit takes E4M3 operands, and an NVFP4 value is not an E4M3 value in general, so the
NVFP4 head cannot be an `atlas.qlinear`. It is a storage format that must be decoded before a BF16 product.

## Export, reload and FX capture

Each quantized embedding and each NVFP4 head becomes a `policy.json` site with `status: quantized`:

| kind | Tensors | Notes |
|---|---|---|
| `embedding` | `weight_codes`, `weight_scale_exp` (FP8), or `weight_codes`, `weight_block_scales`, `weight_tensor_scale` (NVFP4) | A tied site has `tied_to: lm_head` and names the head's tensors. |
| `nvfp4_linear` | `weight_codes`, `weight_block_scales`, `weight_tensor_scale` | |

`policy.json` and `manifest.json` carry a `head_embed` record of the options and the tie digests. The manifest
adds the head_embed entries to `new_decisions`. An FP8 head is an ordinary quantized `linear` site.

`load_quantized.apply_export` installs the storage modules before its state-dict check, and after the head is
quantized it binds a tied embedding to the head. The reload is bit-identical to the in-memory model for every
option, which the test checks.

`fx_export` captures the storage modules as ordinary graph nodes:

- The embedding row gather and its decode.
- The NVFP4 decode followed by `aten.linear`.
- The decode tables, held as graph constants.

`graph_inspection.json` lists these modules under `quantized_storage_sites`. Each one must read its codes
from a lifted buffer (a tied embedding reads the head's) and must hold no float weight. Their decode nodes are
storage decoding, not an emulated matmul, so they are exempt from the native-FP8 rule that no contraction may
be fed by a decode. With `--native-fp8`, an FP8 head is an `atlas.qlinear` like the other Linears. The tied
embedding is the only other reader allowed on that head's code buffer.

## Sizes

`src/scripts/export_sizes.py EXPORT_DIR` reads the safetensors header and splits the stored bytes into
embedding, LM head and the rest.
