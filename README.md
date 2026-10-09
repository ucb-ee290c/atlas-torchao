# atlas-torchao

## Setup

    export HF_HOME=/rscratch/$USER/hf_cache
    conda env create -f environment.yml
    conda activate atlas
    git submodule update --init
    pip install -e model2MLIR --no-deps     # deps (xdsl, structlog, PyYAML) are in requirements.txt

### Notes
- Python 3.12 is required: model2MLIR uses Python 3.11+ features (enum.StrEnum).
- Install model2MLIR with --no-deps; its optional extras can upgrade torch/torchao.
- Create conda envs on /scratch or /rscratch, not your home directory.

## Reference numerics (third_party)

The reference author's quantization code is pinned as two submodules and loaded from there
(`src/ref_loader.py` checks the pins and records the sha256 of every file it loads):

    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    uv pip install numba            # or: pip install -e .[reference]; gcc is needed for the C models

Do not use `--recursive`: smolVLA-quant's nested `npu_model` submodule is not needed.

## The `chip` profile: the whole model as Atlas would run it

`--modes chip` = `reference_hw` (MXU accumulation model for the matmuls) + every other op of the
forward pass routed to the Atlas unit that would execute it (`src/chip_profile.py`,
`src/chip_coverage.py`, `src/chip_ops.py`, `src/op_unit_map.py`):

| unit | what runs there | how it is emulated |
|---|---|---|
| MXU | nn.Linear (reference_hw), attention Q.K^T / P.V and stray mm/bmm, convolutions (im2col) | FP8 E4M3 operands with the reference po2 cast, `--mxu ipt\|sa`, `--mxu-impl`; `--chip-mxu framework` swaps in an FP8-operand float32 matmul rounded to BF16 (NEW, for CPU smokes) |
| VPU | elementwise arithmetic, exp/sin/cos/tanh/log2/rcp/sqrt, RMSNorm/LayerNorm, softmax, SiLU/GELU/sigmoid, reductions, rescale and bias adds after the MXU, additive masks | `ChipVPU`: vectorized torch, bit-exact to the reference `VectorRTLFunctions` (tests/test_chip_ops.py: unary ops on all 65536 BF16 inputs); composites are the reference sequences in `vpu_decomp.py` |
| scalar | integer / bool index arithmetic, comparisons, cumsum of masks | exact (RV32I ALU, ScalarALU.scala:26-36); integer mul/div is a software routine (NEW) |
| data | views, casts, concat, gather/embedding, copies, constants | exact, run natively |
| host | anything with no Atlas instruction (fused SDPA, image resize, bucketize, RNG, float compares, pow with other exponents, select between two computed tensors) and modules kept off-chip with `--skip` | torch on the host, **listed with a reason** |
| eval | perplexity loss | listed separately |

Every result JSON written with `chip` carries `layers.chip.coverage` (eval.py) or
`modes.chip.layers.coverage` (eval_policy.py): calls and work per unit (MXU FLOPs = 2MNK, VPU
lane-element ops, output elements otherwise), the host-fallback list with reasons, the
`new_decisions` list (op, choice, analogue, reason) and `silent_bf16_ops: 0`. The same table is
printed after the first pass and at exit. Options: `--chip-rope vpu|host` (RoPE tables on the VPU,
the reference treatment, or precomputed on the host), `--chip-strict` (raise on an unmapped op).

    python src/eval.py --model /path/TinyLlama-1.1B-Chat-v1.0 --seqlen 128 --max-windows 2 \
        --modes bf16 chip --chip-mxu framework --device cpu --out out/tinyllama_chip.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
        --modes bf16 chip --chip-mxu framework --n-test-episodes 1 --n-frames 2 --device cpu \
        --out out/smolvla_chip.json

### NEW decisions (no reference or ISA source; the analogue followed is cited)

| op | choice | analogue |
|---|---|---|
| `add`/`sub` with alpha, in-place forms | vmul by alpha, write back in place | vpu_decomp.py:232-246 (the reference drops alpha and returns a new tensor) |
| `rsub` | vsub with the operands swapped | vpu_decomp.py:241-246 |
| `sigmoid` (QuickGELU) | vsub, vexp, vadd, vrecip | the sigmoid part of the reference SiLU, vpu_decomp.py:188-191 |
| `log` | vlog2 then vmul by ln 2 | vlog2.bf16 (green card 57/4C) |
| `abs` | vmaximum(x, 0 - x) | neg as vpu_decomp.py:263-264, vmaximum 57/06 |
| `pow` exponents 0.5, -1, -0.5, 1; `c ** x` | vsqrt, vrecip, vsqrt+vrecip, none; vexp2(x * log2 c) | _pow_dispatch, vpu_decomp.py:199-208 (other exponents stay host, as there) |
| `cumsum` (float) | sequential vadd scan | in TARGET_OPS (quant_vector.py:182) with no dispatch entry |
| multi-dim `sum`/`mean`/`amax`, `amin`, `max`/`min` | staged row reductions per dim; rmin pads with +inf | _reduce_with_dim, quant_vector.py:220-235; _vpu_rmax_all, vpu_decomp.py:53-72 |
| `where`/`masked_fill` with a constant branch | vadd of a host-precomputed additive mask (fill rounded to BF16); two computed branches are host | Llama's additive eager mask; no select instruction |
| `_fused_rms_norm`, `_safe_softmax` | the Llama RMSNorm primitives; the softmax sequence | quant_vector.py:150-188, vpu_decomp.py:132-158 |
| `mm`/`addmm` outside nn.Linear | per-tensor cast, MXU, vmul rescale, vadd bias | quant_linear.py:145-172 |
| `bmm`/`baddbmm` (attention when not patched) | per-matrix cast (per batch, head), MXU, vmul rescale; K = 1 is a vmul | model_patcher.py:667-699 |
| `exp2` of the integer scale exponent in RefQuantLinear | scalar-core exponent construction (exact) | 2^(sx+sw) in quant_linear.py:168 is a Python scalar |
| `--chip-mxu framework` | FP8 operands, float32 matmul, one BF16 rounding | ref_profile.py `reference` F.linear stand-in |
| float64 0-d scalar arithmetic | host | schedule constants; Atlas has no FP64 |

The model-level op -> unit map (with the aten ops each model operation becomes) is `MODEL_OPS` in
`src/op_unit_map.py`; the aten-level map with instruction sequences and citations is `ATEN`.

### Multi-Task DiT

multitask-dit-quant's op router accepts an external emulator. With this repo's `src` on
`sys.path`:

    from mtdit_inout.op_emulation import load_op_emulator
    em = load_op_emulator("chip_coverage")          # chip_coverage.make_mtdit_op_emulator()

`em.classify(func, args, kwargs)` returns "VPU" / "scalar" / None (None for data movement, which
that router counts as movement, and for matmuls, which its FP8 patches own: a matmul reaching the
router is host); `em.emulate(func, args, kwargs)` runs the op on ChipVPU with the chip profile's
handlers. Ops the emulator cannot run are classified None, i.e. host fallback in that report.
