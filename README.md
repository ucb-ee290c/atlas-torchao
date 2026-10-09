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

## Profiles

`--modes` accepts `bf16`, `collab_v0` (alias `atlas`), `collab_v0_hw` (alias `atlas_hw`),
`reference`, `reference_hw`, `chip` and `torchao`. `chip` is the headline: the whole model as it
would run on Atlas (MXU model + VPU + scalar core, host fallbacks reported). `reference_hw` is the
MXU-only comparison. Every numerics decision and its source, the findings and the open questions are
in [docs/reference_profile.md](docs/reference_profile.md).

Reference-profile options: `--mxu ipt|sa` (default `ipt`, the ipt_numba model, which is the reference
author's choice), `--mxu-impl torch|ref_cpu` (default `torch`), `--accum-compile` (compile the SA
step), `--ops linear,conv2d,attention`, and `--crosscheck on|off` (default on). With the cross-check on,
a seeded sample of real calls is recomputed by the reference CPU model, and any bit mismatch sets
`valid: false` in the result JSON.

## L40S run matrix

Setup, once per machine:

    git clone -b reference-profile git@github.com:ucb-ee290c/atlas-torchao.git && cd atlas-torchao
    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    pip install numba                                        # gcc must be on PATH for the C oracles
    export PYTHONPATH=src HF_HOME=/path/to/hf_cache TL=/path/to/TinyLlama-1.1B-Chat-v1.0
    mkdir -p out

Gates (stop if any fail):

    python -m pytest -q tests
    ATLAS_TEST_COMPILE=1 python -m pytest -q tests/test_accum_torch.py
    python scripts/bench_accum.py --device cuda --compile --check --ref | tee out/bench_accum.txt

Size the slow jobs from the measured throughput, so that each job fits in about 1–2 h:

    N = floor(seconds * MAC_per_s / (2048 * 0.97e9))

Use the ipt or SA compiled MAC/s from `out/bench_accum.txt` for 2048-token TinyLlama windows.
Start with N = 4 if in doubt.

TinyLlama (WikiText-2):

    # fast profiles, full split; the collaborator's protocol for comparison
    python src/eval.py --model $TL --modes bf16 collab_v0 collab_v0_hw reference torchao --out out/tl_fast_default.json
    python src/eval.py --model $TL --protocol mxq --nsamples 0 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/tl_fast_mxq.json
    # MXU models, seeded windows
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes reference_hw --mxu ipt --rmse --out out/tl_refhw_ipt.json
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes reference_hw --mxu sa --accum-compile --rmse --out out/tl_refhw_sa.json
    # whole chip (headline)
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes chip --mxu ipt --out out/tl_chip_ipt.json

SmolVLA (lerobot/svla_so100_pickplace; set F, the frame count for the slow jobs, from the
benchmark: one frame is roughly one forward of a few hundred tokens times the flow steps):

    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames 200 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/smolvla_fast.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames $F --modes reference_hw --mxu ipt --rmse --out out/smolvla_refhw_ipt.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames $F --modes chip --mxu ipt --out out/smolvla_chip_ipt.json

Policy-eval options (eval_policy.py):

- Determinism check: each mode runs twice by default and the result JSON records `determinism` and
  `deterministic` per mode. `--no-determinism-check` runs each mode once (halves the time, determinism
  unchecked). `--deterministic` turns on deterministic kernels (`torch.use_deterministic_algorithms`,
  cuDNN deterministic, cuBLAS workspace config). Compare runs only at equal thread counts: CPU
  reductions are split by thread, so the thread count changes bf16 results.
- `--dtype {bfloat16,float32}` (default bfloat16) sets the policy dtype for every mode, the baseline
  included; the reference author loads SmolVLA in float32. The baseline mode label stays `bf16` even
  with `--dtype float32` (it is the mode name used throughout the result JSON and the `vs_bf16`
  keys, so it is documented rather than renamed); read it as "the unquantized baseline in --dtype".

Multi-Task DiT. The dedicated DiT profile lives in multitask-dit-quant (see its README:
git@github.com:lorenshung/multitask-dit-quant.git). Here there is no DiT-specific profile: the
generic rules apply, and its custom attention is left unpatched (see `attention_sites` in the JSON).

    python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --n-frames 200 --modes bf16 collab_v0 collab_v0_hw reference --out out/dit_fast.json

Check every reference_hw and chip result before quoting it:

- `layers.<mode>.valid` is true;
- `crosscheck.ok` is true and `bit_mismatches` is 0;
- `reference_sources` records the pinned commits.

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
| exp inside `_softmax`, `silu`, `sigmoid` | `vmaximum(x, -128)` before `vexp` | the vexp model (bit-exact to ExpLane.scala) wraps below x = -336: exp(-3.3e38) = 1, so an additive min/-inf mask did not mask (TinyLlama chip PPL 942 -> 9.5 on the first window); analogue: ExpLane.scala:47 overflow pre-check |
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

### Findings for the Atlas hardware team

- `vexp.bf16` (reference FPEX model, bit-exact to ExpLane.scala) is correct only for x >= -336; for
  more negative inputs the Q(m,n) conversion wraps (exp(-1000) = 2.6e10, exp(-3.3e38) = 1.0). Masked
  softmax with the framework's min/-inf mask is wrong unless software clamps first. Is a saturating
  underflow (return +0) intended in the RTL?
- `vsin`/`vcos` take a Q(9,12) argument (x 2/pi), i.e. |x| < ~400 rad, and BF16 angles have an ulp of
  0.5 rad at 64-128: RoPE angles for positions beyond a few hundred are not representable on the VPU.
  Should RoPE tables be precomputed on the host (`--chip-rope host`) in deployment?
- `vadd`/`vsub` truncate the FP32 sum to BF16 (raw upper half, AddSubSumVec.scala:133) while
  `vmul` and `vredsum.row` round to nearest even: intended?
- The scalar core has no multiply/divide and no FP (ScalarALU.scala:26-36): float schedule
  constants (flow-matching dt, DDPM coefficients) and integer index multiplies need host
  precomputation or software routines.
- No select/compare instruction on the VPU: masks must be additive and precomputed on the host.
