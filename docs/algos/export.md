# Exporting a quantized model (`--export-dir`)

`--export-dir DIR` writes a quantized model as files. It works on `src/eval.py`, `src/eval_policy.py`
and `src/eval_pi0.py`. `src/load_quantized.py` rebuilds the model from those files without
calibration and without the float weights of the quantized layers. Code: `src/quant_export.py`,
`src/load_quantized.py`. Tests: `tests/test_export.py`.

The format is **NEW**. No source defines a file format for these numerics. Every choice that
follows is listed in the manifest under `new_decisions`.

## When the export is written

The CLI exports the first mode in `--modes` that uses the reference numerics: `ref_linear`,
`ref_attn`, `ref_conv` / `reference`, `reference_hw`, `chip` or `chip_rope_vpu`. If `ref_conv` is
present, it wins. All of these modes share the same weights.

The export is written right after `algos.quantize`: calibration, folds and algorithm transforms
have all been applied, and evaluation has not started. Evaluation then runs as usual.

- `--export-only` quantizes only the exported mode, writes the export and stops. It skips `bf16`,
  evaluation, `--rmse` and `--kl`.
- `--from-export DIR` builds every quantized mode from an export instead of quantizing. Use it to
  re-evaluate an export under the same protocol. It cannot be combined with `--export-dir`.

`collab_v0`, `collab_v0_hw` and `torchao` cannot be exported. Their weights are not E4M3 codes with
power-of-two exponents.

## Files

One directory per model × recipe.

### `model.safetensors`

| Tensor | dtype | What it is |
|---|---|---|
| `<fqn>.weight_codes` | uint8 | E4M3 codes of each quantized Linear / Conv2d, as `ref_numerics.e4m3_vpack` encodes them. A Conv2d weight is flattened to `[out, in*kh*kw]`. Codes 0x7F / 0xFF never occur. |
| `<fqn>.weight_scale_exp` | int32 | Power-of-two exponents, E8M0-style: real = e4m3(code) · 2^exp. The shape follows the granularity: `[]` per tensor, `[out, 1]` per channel, `[out, K/32, 1]` per block. |
| `<fqn>.weight_lo_codes`, `<fqn>.weight_lo_scale_exp` | uint8 / int32 | Residual weight, only for `mixed_precision keep_as=fp8x3|fp8x2_w`. |
| `<fqn>.input_scale_exp` | int32 | Static activation exponent, for example `smoothquant static=true` (`atlas_v1_static`). |
| every other state-dict entry | its own dtype | The model as it stands after the folds. This covers norm weights and producer rows that SmoothQuant / AWQ changed, biases, embeddings, `lm_head`, BF16 opt-out layers, and RoPE `inv_freq` (eval.py's loader makes it persistent). |

The float weight of a quantized layer is not stored. Tied tensors are stored once, and
`policy.json` lists the other names under `aliases`.

### `policy.json` (`atlas_torchao.execution_policy.v1`, NEW)

This is the per-site execution policy. The loader executes it, so the bit-identical round trip in
the tests also checks the policy. It is meant to become the m2m adapter's policy file.

- **`operand`**: E4M3 with exponent 15 finite, finite max 448, reserved codes 0x7F / 0xFF, flush
  to +0 below 2^-6, saturation, RNE, and a BF16 cast before encoding. The field names follow
  Merlin's `merlin.quantization_adapter_contract.v1` `operand` block.
- **`scale`**: the E8M0-style exponent encoding, the reference rule `floor(log2(amax/256))` and the
  NEW `nc` rule `ceil(log2(amax/448))`.
- **`accumulate`**: MXU, the IPT model assumed by default (`--mxu ipt`), BF16 readout, and the
  accumulation the exported mode used.
- **`sites`**: one entry per site, keyed by fqn.
  - **Linear / Conv2d**: `weight` (tensor names, granularity, block) and `weight_dtype`, the layer
    dtype the output is cast to. Also `activation`, which is one of:
    - the reference dynamic rule;
    - `dynamic` with `reduce` = `tensor | token | reference_dims | block32_or_token` and
      `rule` = `ref | nc`;
    - `static` with `exp`;
    - `static_per_step` with `exps`.
    The entry also gives `output_rescale` (`2^(s_x+s_w)` in float32, then the layer dtype, bias
    after), `residual_passes`, and for a Conv2d the im2col geometry.
  - **Attention**: one rule per operand `q`, `k`, `p`, `v`.
  - **BF16 opt-outs**: status `host` and the reason. NEW: canonical `atlas_v1` on
    `multi_task_dit` keeps the exact CLIP patch embedding Conv2d in BF16. Its top-level
    `bf16_opt_out` row has `match: exact`; the loader excludes only that site, preserving
    similarly named siblings. Historical exports without `match` retain substring replay.
    The canonical full-checkpoint release export passed an eight-frame, 96-tensor
    byte-identical round trip at `5ff028e`; all historical run-5.3 DiT exports had zero
    BF16 opt-outs. See the [verification record](release/release_verification.json).
  - **`--skip` layers**: status `skipped`.
- **`runtime`**:
  - `chip_options`: the recipe's chip defaults, such as `chip_vadd rn` on residual sites, plus any
    chip option given on the command line;
  - requirements such as RoPE / timestep tables as host constants, and a run-time amax for
    dynamic sites.
- **`pending_hardware_rulings`**: the open rulings listed below.

### `manifest.json` (`atlas_torchao.export_manifest.v1`, NEW)

- **`files`**: sha256 and byte size of `model.safetensors` and `policy.json`. `policy_sha256`
  repeats the policy's digest. The loader refuses an export whose files do not match.
- **`provenance`**: atlas-torchao commit, branch and dirty flag; torch, torchao, transformers,
  safetensors, lerobot and numpy versions; host, device, GPU and argv. For Pi0 it also records the
  openpi pin, norm stats and tokenizer digests.
- **`model`**: the CLI, model id and checkpoint record. That record holds the sha256 of each
  `*.safetensors` of a local checkpoint or cached HF snapshot. For Pi0 it is
  `pi0_model.checkpoint_info`.
- **`recipe`**: the `--algo` string, its expanded stack with parameters, the recipe record
  (`algos.RECIPES` on `atlas_v1`), and the full algorithm record from `algos.quantize`, including
  per-member calibration and prepare records.
- **`calibration`**: the calibration loader's record (source, n, seed, windows / episodes /
  frames). `data_sha256` is a digest over the calibration batches as the CLI held them.
- **`sites`** and **`counts`**: the site census.
- **`new_decisions`** and **`pending_hardware_rulings`**.
- **`m2m_manifest_v2_aligned: "partial"`**, with `m2m_manifest_v2_matching_fields` and
  `m2m_manifest_v2_differences`.

## Loader

```python
from load_quantized import load_quantized
model, report, manifest = load_quantized(model_id, export_dir, device)                 # causal LM
model, report, manifest = load_quantized(None, export_dir, device, build=make_policy)   # policy / Pi0
model, report, manifest = load_quantized(None, export_dir, device, mode="chip", options={"chip_mxu": "framework"})
```

The loader works in six steps:

1. Read the three files and check their sha256 against the manifest.
2. Build the architecture.
   - A causal LM is built from its config alone, with the same rope scaling, eager attention and
     persistent buffers as `modelutils.load_llama`. No checkpoint weight is read.
   - A LeRobot policy or Pi0 has no weight-free constructor here, so `build=` (the CLI's own
     loader) builds the module tree from the checkpoint. The next step then overwrites every
     tensor, and any state-dict entry the export does not cover raises. No original value
     survives.
3. Assign every exported tensor in its stored dtype. Give each quantized layer a placeholder
   weight `dequant(codes)` in its layer dtype, so that `weight.dtype` / shape introspection works.
   The quantized forward never reads that weight.
4. Run `quant_modes.quantize` under `PolicyAlgo`, an `algos.base.Algo` whose hooks read the policy.
   Weight exponents come from the export. Activation scales come from the per-site rules, using
   the same functions as the algorithms (`granularity.scale_exp`, `block_scale_exp`,
   `ref_numerics.po2_scale_exp`).
5. Check that the re-derived codes equal the exported ones, bit for bit. Install the residual codes.
6. Optionally rebuild a different reference mode than the exported one, if the export's op classes
   cover it. For example, `ref_linear`, `reference_hw` or `chip` from a `ref_conv` export.

## Tests (`tests/test_export.py`)

- **Bit-identical round trip on a tiny Llama.** The logits of the reloaded model equal the
  in-memory quantized model for `none`, `atlas_v1`, `atlas_v1_sq`, `atlas_v1_awq` and `atlas_v1_static` (the
  recipe names where `algos.RECIPES` exists, otherwise the written-out stacks):
  - under `ref_conv`;
  - under `ref_linear` rebuilt from the same export;
  - under `reference_hw`;
  - under `chip --chip-mxu framework`.
- **Other algorithms.** The same round trip for granularity block32 with per-token attention,
  static_calib `per_step` with a static `p` exponent, mixed_precision `fp8x3` (residual passes) and
  `bf16` (opt-outs), and a Conv2d model.
- **File contents.** No float weight is stored for a quantized layer. Codes are uint8 and never
  0x7F / 0xFF. Exponents equal the modules' `weight_scale_exp`. The BF16 tensors are the post-fold
  ones.
- **Manifest.** The sha256 values match the files, and a modified file is refused.
- **CUDA.** A round trip (skipped without CUDA).
- **SmolVLA.** The real `SmolVLMWithExpertModel` forward on tiny Llama stacks, with `atlas_v1`.
- **Pi0.** A tiny random `PI0Pytorch` (`tests/pi0_tiny.py`), with `atlas_v1`. Needs the atlas-pi0
  environment and openpi.

Full-checkpoint Multi-Task DiT release verification used `eval_policy.py`: eight frames,
96 byte-identical captured tensors, exact action statistics, deterministic repeats, finite
outputs and matching payload hashes. Its 204 total policy sites comprise 203 quantized
sites and one exact BF16 Conv2d exception. There is no tiny constructor for this model;
this full-checkpoint verification is separate from the small-model unit tests.

The bundle does not contain a serialized FX graph, `ExportedProgram`, ONNX model, MLIR
module or device binary. Graph capture and compiler lowering are separate work.

## Relation to the m2m adapter

`policy.json` holds what an m2m quantization adapter (`m2m.quantization_adapters` entry point,
`pyproject.toml`, not yet written) needs to execute a site. `manifest.json` is aligned with
model2MLIR's `m2m.quantization_manifest.v2` (`m2m/capture/external_quantization.py`,
`validate_manifest`) only in part.

These fields match:
- `policy_sha256`;
- a nonempty `sites` list with a unique `site_id` per site;
- a `status` in `quantized | host | skipped`;
- JSON without NaN.

These fields differ:
- `schema`;
- no `adapter_id`;
- no `contract_sha256`, because no `merlin.quantization_adapter_contract.v1` was read;
- `preserved` / `execution_route` are not used;
- the extra top-level fields.

The adapter would read a Merlin adapter contract and check it against `policy.operand`. It would
then emit the v2 manifest with these fields filled.

## NEW decisions (also in `manifest.new_decisions`)

- The three-file format, the tensor names and both schemas.
- The export is taken from the first reference mode, with `ref_conv` preferred, before evaluation.
- Exponents keep their in-memory shapes, and static activation exponents are stored as
  `<fqn>.input_scale_exp`.
- The loader's placeholder weight is `dequant(codes)`.
- Every non-quantized tensor is stored in its own dtype. That is not always BF16: smoothquant's
  promoted (1+w) norms and SmolVLA's FP32 action projections stay FP32.
- BF16 opt-outs have m2m status `host`, and `--skip` layers have status `skipped`.
- The calibration data digest.

## Pending hardware rulings (also in `policy.json` and `manifest.json`)

- **Q2**: run-time vs static activation scales. This is why `atlas_v1` and `atlas_v1_static` both
  exist.
- **H2**: how a per-row exponent is derived from `vredmax.row`.
- **vadd rounding**: `trunc`, `rn` or an RNE adder.
- **448 vs 240**: the operand range. This export uses 448, the reference author's numerics.
- **R4**: residual passes sharing the MXU accumulator, and the BF16 fallback.
