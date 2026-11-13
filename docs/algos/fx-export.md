# FX graph export and verification

`src/fx_export.py` captures a model reconstructed from the existing quantized export,
saves a PyTorch `ExportedProgram` as `model.pt2`, and verifies it after reload. The
program contains an FX computation graph, lifted tensor state and fixed input guards.
No calibration or quantizer search runs during this export.

This is **NEW reference-emulation graph export**, separate from the immutable release
tag. The graph decodes uint8 E4M3 codes and applies int32 power-of-two exponents, then
uses the reference framework's BF16/FP32 contractions and vector operations. It is not
native-FP8 MLIR, an Atlas executable or hardware qualification. Chip-only choices such
as the `rn` residual-add sequence remain recorded in the policy but are not executed
by the `ref_conv` graph.

## Causal language model command

Run from this checkout with the environment used for the original model. The initial
CLI captures batch 1, fixed sequence length, tokens to logits, with caching disabled.
The output directory must not exist; existing exports and output directories are never
overwritten.

```bash
python src/fx_export.py \
  --export-dir /absolute/path/to/verified/tinyllama/export \
  --model /absolute/path/to/tinyllama/config \
  --out-dir /absolute/path/to/new/fx-directory \
  --device cuda --seqlen 8 --seed 0
```

`--model` selects architecture configuration only; it defaults to the source export's
model identifier. `load_quantized` checks the original payload hashes and rebuilds
`ref_conv` replay. Two different fixed token inputs are checked. Strict capture is the
default; `--non-strict` explicitly selects the alternate tracing mode, without an
automatic fallback. Unsupported operations and parity failures stop the check and
leave a diagnostic `verification.json`.

To replay a saved graph independently:

```bash
python src/fx_export.py --verify-dir /absolute/path/to/fx-directory --device cuda
```

The fresh-process check imports PyTorch and SafeTensors but not the original model
architecture, `load_quantized`, `ref_profile` or `ref_attention`. Use the capture device
and a compatible PyTorch version; cross-version or cross-device portability is not
established by this check.

## Native-FP8 option (`--native-fp8`)

The default graph above decodes every quantized Linear, Conv2d and attention operand into aten ops. With
`--native-fp8` the quantized contractions become opaque nodes from `src/atlas_fx_ops.py`, fed directly by the
weight code and exponent buffers:

| Site | Nodes in the graph |
| --- | --- |
| Linear | `atlas.quantize_act` then `atlas.qlinear(x_codes, x_exp, w_codes, w_exp, bias, out_dtype, "atlas_e4m3")` |
| Conv2d | visible im2col (`aten.im2col`, permute, reshape), `atlas.quantize_act` per patch row, `atlas.qconv2d(..., B, oh, ow)` on the `[out, in*kh*kw]` weight codes; geometry is checked against `policy.json` |
| Attention | per Q, K, P, V one `atlas.quantize_act` (per head `reduce=head`, or per token), `atlas.qk_matmul`, then the head_dim scaling, mask, softmax and the BF16 cast as ordinary nodes, then `atlas.pv_matmul` on `V^T` |

```
%quantize_act = atlas.quantize_act(%x_bf16, dynamic, token, nc, None, uint8)   -> (codes, exp)
%qlinear      = atlas.qlinear(%codes, %exp, %<fqn>_weight_codes, %<fqn>_weight_scale_exp, %bias, bfloat16, atlas_e4m3)
```

No `convert`, `exp2`, `mul`, bit operation, `frexp` or float weight remains in those scopes, and `inspect_graph`
fails closed otherwise. The activation rule is read from `policy.json` (reference, dynamic token / tensor /
per-head with rule `ref` or `nc`, or static). The eager bodies call the reference code (`ref_numerics`,
`algos.granularity.scale_exp`, `F.linear` with the bias after, `ref_numerics.dq_matmul_nt`, which
`ref_attention._mm` also calls), so outputs are byte-identical to eager replay and to the decode graph;
`capture_fx` captures the decode graph in memory as well and compares both on every input case. Capture
swaps `ref_attention._cast` / `_mm` and each layer's `forward` inside a restoring context only.

```bash
python src/fx_export.py --export-dir EXPORT --model CONFIG --out-dir NEW_DIR --device cpu --seqlen 8 --native-fp8
python src/fx_export.py --verify-dir NEW_DIR --device cpu       # imports atlas_fx_ops itself
python src/fx_export.py --export-dir SMOLVLA_EXPORT --out-dir NEW_DIR --device cpu --native-fp8 \
    --boundary smolvla_flow_loss --n-images 1
```

- **Loading a saved native bundle.** `model.pt2` names the `atlas.*` operators. Run `import atlas_fx_ops` (from
  `src/`) before `torch.export.load`, or the load fails with an unknown-operator error. `verify_saved` does
  this when `fx_manifest.json` has a `native_fp8` record and checks the operator counts against
  `graph_inspection.json`; tests/test_fx_native.py checks that a bare load fails and the verifier passes.
- **Storage dtype.** `--native-storage uint8` (default) holds the export bytes as uint8 and the operators carry
  the format as the `weight_format="atlas_e4m3"` attribute. `--native-storage float8_e4m3fn` holds the same
  bytes as a `torch.float8_e4m3fn` bitcast. The two decoders do not agree on every code:
  `torch.float8_e4m3fn` reads the 14 exponent-0 non-zero codes (0x01-0x07, 0x81-0x87) as subnormals where
  `ref_numerics.e4m3_unpack` gives zero (flush to zero), and 0x7F/0xFF as NaN. The other 240 of the 254 valid
  codes (including exponent 15 and the 448 maximum) are bit-identical. The encoder never emits those 16 codes,
  and the float8 option refuses any layer that contains one. `uint8` is the default because the proof is
  "every valid code", not "every code the encoder emits".
- **Inventory.** `buffer_inventory.json` records the held dtype and the sha256 of the bytes; for float8 storage
  the hash equals the hash of the original uint8 codes (checked at capture).
- **Tile-native report (NEW).** `graph_inspection.json` also has `tile_native` (per site: `tile_native`, reasons, whether
  it is carried natively; the list of sites whose exponents are not constant along K at 32-granularity; the
  observed dynamic exponents, token splits and code hygiene of the capture cases under `fixture_activation_stats`)
  and `compiler_pack_scale_admission` (informational). `quantize_act` has `reduce="block"` / `"head_block"` and a
  trailing `rows=32` argument with a default, so bundles saved earlier still load. See [tile-aware.md](tile-aware.md).
- **What stays emulated.** `graph_inspection.json` lists under `still_emulated` any quantized module that
  could not be carried natively (per-block weight or activation scales, residual passes, MXU models, an
  activation rule `quantize_act` does not express, a Conv2d whose geometry differs from the policy). Attention
  is native for all sites or for none: one unsupported operand rule keeps every attention module on the
  decode path. `native_fp8_attention_modules` gives, per attention module, the operand rules, the native
  node names and `remaining_nodes_in_module_by_target`: what is left in the module (RoPE, repeat_kv,
  masks, softmax, casts) is ordinary BF16 / float32 nodes, not FP8 emulation.
- **BF16 by policy.** `bf16_by_policy` (written for every capture, native or not) lists the sites the policy
  keeps in BF16 (status `host` or `skipped`, e.g. TinyLlama `lm_head`, or the DiT patch-embedding Conv2d
  exception): the module, the reason, the contraction node and the real BF16 weight. Each is an ordinary
  `aten.linear` / `aten.conv2d` with no `atlas.*` operator in its scope. This is separate from
  `still_emulated`: BF16 by policy is intended, an emulated site is FP8 that is not yet native.
- **Boundary for SmolVLA.** `--boundary smolvla_flow_loss` captures `VLAFlowMatching.forward` (the
  flow-matching loss for given images, language tokens, state, actions, noise and time): the vision tower
  including its patch-embedding Conv2d, the connector, all text/expert layers and the action projections, as a
  tensor-in / tensor-out forward with no sampling loop, no KV cache and no random draw. The 10-step action
  sampler is a different boundary and is not captured. The 12 vision attention modules and the single
  text/expert attention site (`model.vlm_with_expert`, invoked once per layer) are native.
- **Output rescale.** The operators scale the operands by their power-of-two exponents before the contraction,
  as eager `ref_conv` does. The policy's wording (normalized operands, then `2^(sx+sw)` in float32, then the
  layer dtype) gives the same bytes for per-token x per-channel exponents;
  `test_rescale_after_contraction_equals_operand_scaling` measures it for the Linear. Attention scores are
  float32 matmuls on the decoded operands, as in `ref_attention`.

## Files

| File | Contents |
| --- | --- |
| `model.pt2` | Serialized PyTorch ExportedProgram with its FX graph and tensor state |
| `graph.txt`, `graph_signature.txt` | Readable operations and lifted input/output signatures |
| `buffer_inventory.json` | Exact code/exponent buffer names, dtypes, shapes and byte hashes |
| `graph_inspection.json` | Actual contraction dependencies, live-scale witnesses and scope limits |
| `fixtures.safetensors`, `parity.json` | Fixed inputs, eager expected outputs and tensor structure |
| `policy.json` | Original per-site quantization and execution policy |
| `fx_manifest.json` | Artifact hashes, byte sizes, PyTorch version and source provenance |
| `verification.json` | Capture, same-process reload and fresh-process verification results |
| `fresh_process_verification.json`, `fresh_process.log` | Independent reload outcome and log |

## Required checks

- Eager replay, the captured graph and the saved/reloaded graph must produce
  byte-identical finite outputs on both inputs.
- All selected weight code and scale buffers must retain exact bytes, uint8/int32
  dtypes and shapes. Nonpersistent buffers are checked through
  `ExportedProgram.named_buffers()`, including constants rather than only `state_dict`.
- Each selected Linear/Conv2d contraction must consume both its code and exponent
  buffers. Float placeholder weights must have no graph users. During capture only,
  their redundant storage becomes same-shape/dtype metadata-only tensors; original
  parameters are restored even after failure. Saved/reloaded placeholders must remain
  metadata-only, with no duplicate floating-point payload.
- Dynamic Linear/Conv2d activation rules must retain input-value-dependent amax,
  exponent derivation and, for `nc`, comparison operations. Shape ancestry alone is
  insufficient: deriving a scale from `ones_like(input)` is rejected.
- All selected quantized Linear/Conv2d modules must be accounted for. Mutable
  `static_per_step` policies and matrix-unit emulators/residual passes are refused by
  this initial path.
- All eight required input artifacts must match their manifest before replay. Empty
  inventories, fewer than two distinct input cases and empty parity cases are refused.
  Existing output directories are refused.

Tests also change code/exponent buffers, poison unused float placeholders, vary
activation amplitudes and inject capture failures. Functional attention is exercised
by full-boundary output parity, but per-site structural attention binding is not yet
certified and is explicitly labelled in the inspection report.

## Policy model boundary and compiler handoff

The Python API accepts an explicit tensor-input/tensor-output module boundary. The
initial CLI does not export SmolVLA, Pi0 or DiT action sampling. DiT's ordinary
`forward` computes training loss; its action path includes observation preparation,
encoding, random noise and scheduler state. A denoising component taking observations,
noisy actions and timestep is a different boundary and must be labelled as such.

A small synthetic DiT regression can check preservation of the exact BF16 patch
exception without establishing full-checkpoint action-graph export. Real-model GPU
capture is separate evidence from the synthetic CPU tests.

model2MLIR lowering follows this graph check (step 3, not part of the native-FP8 option). Its native-FP8-to-FP32 projection and the
448-versus-240 target-contract conflict remain separate blockers to claiming native
FP8 or hardware compatibility; they do not change the reference-emulation scope here.
