# FX graph export and verification

`src/exporter/fx_export.py` captures a model reconstructed from the existing quantized export,
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
PYTHONPATH=src python -m exporter.fx_export \
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
PYTHONPATH=src python -m exporter.fx_export --verify-dir /absolute/path/to/fx-directory --device cuda
```

The fresh-process check imports PyTorch and SafeTensors but not the original model
architecture, `load_quantized`, `ref_profile` or `ref_attention`. Use the capture device
and a compatible PyTorch version; cross-version or cross-device portability is not
established by this check.

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

model2MLIR lowering follows this graph check. Its native-FP8-to-FP32 projection and the
448-versus-240 target-contract conflict remain separate blockers to claiming native
FP8 or hardware compatibility; they do not change the reference-emulation scope here.
