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

The default graph above decodes every quantized Linear into aten ops. With `--native-fp8`, each
eligible Linear is instead two opaque nodes from `src/atlas_fx_ops.py`:

```
%quantize_act = atlas.quantize_act(%x_bf16, dynamic, token, nc, None, uint8)   -> (codes, exp)
%qlinear      = atlas.qlinear(%codes, %exp, %<fqn>_weight_codes, %<fqn>_weight_scale_exp, %bias, bfloat16, atlas_e4m3)
```

The weight code and exponent buffers are consumed directly; no `convert`, `exp2`, `mul` or float
weight remains in that layer's scope. The activation rule is read from `policy.json` (reference per
tensor, dynamic token/tensor with rule `ref` or `nc`, or a static exponent). The eager bodies call the
reference code (`ref_numerics`, `algos.granularity.scale_exp`, `F.linear`, bias after), so outputs are
byte-identical to eager replay and to the decode graph; `capture_fx` also captures the decode graph
in memory and compares both on every input case.

```bash
python src/fx_export.py --export-dir EXPORT --model CONFIG --out-dir NEW_DIR --device cpu --seqlen 8 --native-fp8
python src/fx_export.py --verify-dir NEW_DIR --device cpu       # imports atlas_fx_ops itself
```

- **Loading a saved native bundle.** `model.pt2` names `atlas.quantize_act.default` and
  `atlas.qlinear.default`. Run `import atlas_fx_ops` (from `src/`) before `torch.export.load`, or the
  load fails with an unknown-operator error. `verify_saved` does this when `fx_manifest.json` has a
  `native_fp8` record; tests/test_fx_native.py checks that a bare load fails and the verifier passes.
- **Storage dtype.** `--native-storage uint8` (default) holds the export bytes as uint8 and the
  operators carry the format as the `weight_format="atlas_e4m3"` attribute. `--native-storage
  float8_e4m3fn` holds the same bytes as a `torch.float8_e4m3fn` bitcast. The two decoders do not agree
  on every code: `torch.float8_e4m3fn` reads the 14 exponent-0 non-zero codes (0x01-0x07, 0x81-0x87)
  as subnormals where `ref_numerics.e4m3_unpack` gives zero (flush to zero), and 0x7F/0xFF as NaN. The
  other 240 of the 254 valid codes (including exponent 15 and the 448 maximum) are bit-identical. The
  encoder never emits those 16 codes, and the float8 option refuses any layer that contains one.
  `uint8` is the default because the proof is "every valid code", not "every code the encoder emits".
- **Inventory.** `buffer_inventory.json` records the held dtype and the sha256 of the bytes; for
  float8 storage the hash equals the hash of the original uint8 codes (checked at capture).
- **Scope.** Linear layers with a per-tensor or per-channel weight exponent and a reference, token,
  tensor or static activation rule. Per-block scales, residual passes, MXU models, Conv2d and the
  attention Q.K / P.V operands stay on the decode path. `graph_inspection.json` lists them under
  `still_emulated` (modules, their contraction nodes, the amax/frexp/gt scale nodes, node counts).
- **Output rescale.** The operator scales the operands by their power-of-two exponents before the
  contraction, as eager `ref_conv` does. The policy's wording (normalized operands, then
  `2^(sx+sw)` in float32, then the layer dtype) gives the same bytes for per-token x per-channel
  exponents; `test_rescale_after_contraction_equals_operand_scaling` measures it.

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
