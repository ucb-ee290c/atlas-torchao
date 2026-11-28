# The `chip` profile: the whole model as Atlas would run it

`--modes chip` = `reference_hw` (MXU accumulation model for the matmuls) + every other op of the
forward pass routed to the Atlas unit that would execute it (`src/chip/chip_profile.py`,
`src/chip/chip_coverage.py`, `src/chip/chip_ops.py`, `src/chip/op_unit_map.py`):

| unit | what runs there | how it is emulated |
|---|---|---|
| MXU | nn.Linear (reference_hw), attention Q.K^T / P.V and stray mm/bmm, convolutions (im2col) | FP8 E4M3 operands with the reference po2 cast, `--mxu ipt\|sa`, `--mxu-impl`; `--chip-mxu framework` swaps in an FP8-operand float32 matmul rounded to BF16 (NEW, for CPU smokes) |
| VPU | elementwise arithmetic, exp/sin/cos/tanh/log2/rcp/sqrt, RMSNorm/LayerNorm, softmax, SiLU/GELU/sigmoid, reductions, rescale and bias adds after the MXU, additive masks | `ChipVPU`: vectorized torch, bit-exact to the reference `VectorRTLFunctions` (tests/test_chip_ops.py: unary ops on all 65536 BF16 inputs); composites are the reference sequences in `vpu_decomp.py` |
| scalar | integer / bool index arithmetic, comparisons, cumsum of masks; arithmetic whose tensor operands are all one-element float32 (DDPM scheduler coefficients, Pi0's `-dt / 2`) | exact (RV32I ALU, ScalarALU.scala:26-36); integer mul/div is a software routine (NEW); one-element float32 arithmetic is a software float32 routine, run natively (NEW) |
| data | views, casts, concat, gather/embedding, copies, constants | exact, run natively |
| host | anything with no Atlas instruction (fused SDPA, image resize, bucketize, RNG, float compares, pow with other exponents, select between two computed tensors) and modules kept off-chip with `--skip` | torch on the host, **listed with a reason** |
| eval | perplexity loss | listed separately |

Every result JSON written with `chip` carries `layers.chip.coverage` (eval.py) or
`modes.chip.layers.coverage` (eval_policy.py): calls and work per unit (MXU FLOPs = 2MNK, VPU
lane-element ops, output elements otherwise), the host-fallback list with reasons, the
`new_decisions` list (op, choice, analogue, reason) and `silent_bf16_ops: 0`. The same table is
printed after the first pass and at exit. Options: `--chip-rope host|vpu` (default `host`, NEW:
RoPE tables and the sinusoidal timestep table are compile-time host constants; `vpu` computes them on
the VPU, the reference treatment, and is the `chip_rope_vpu` ladder step), `--chip-strict` (raise on
an unmapped op), `--chip-vadd trunc|rn|rne_hw` and `--chip-vadd-sites residual|data|all` (vadd/vsub
numerics: default `trunc` is the hardware as built; `rn` and `rne_hw` are off-ladder options, see
docs/algos/vpu-numerics.md).

    python src/eval.py --model /path/TinyLlama-1.1B-Chat-v1.0 --seqlen 128 --max-windows 2 \
        --modes bf16 chip --chip-mxu framework --device cpu --out out/tinyllama_chip.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
        --modes bf16 chip --chip-mxu framework --n-test-episodes 1 --n-frames 2 --device cpu \
        --norm-stats dataset --out out/smolvla_chip.json

## Pi0

Every aten op of openpi's PI0Pytorch (`sample_actions`, pi0=True) has a unit: `MODEL_OPS` rows
`"pi0"` in `src/chip/op_unit_map.py`, cited to pi0-quant's `patch_vector_ops` (quant_vector.py:136-171,
360-433) and vpu_decomp.py, which this repo follows op for op (GELU-tanh, RMSNorm with `1 + w` as
vsquare / staged rsum / vsqrt+vrecip / vadd / vmul, softmax, SiLU, embedding scale, RoPE apply,
Euler step). On a tiny random-weight Pi0 (tests/pi0_tiny.py, tests/test_chip_pi0.py)
`silent_bf16_ops` is 0 and the only host fallbacks are the float `le` of the block-causal mask
build, the loop test `ge` and the loop time update `add_`; the loop bound `-dt / 2` is one-element
float32 arithmetic on the scalar core (plus the RoPE and timestep tables with
`--chip-rope host`, now the default for every model). Pi0 shows why: its prefix positions reach
~880 and its timestep angles 1571 rad, where the VPU's sin/cos (and BF16 angles) are wrong by up
to 2.0. The reference puts them on the VPU; that treatment is the `chip_rope_vpu` step.
The masked logits (-2.38e38) are zeroed by the vexp floor (-128).

## NEW decisions (no reference or ISA source; the analogue followed is cited)

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
| arithmetic whose tensor operands are all one-element float32 (Python-scalar operands may join; any multi-element operand keeps the op on the VPU) | scalar core, software float32, run natively | multitask-dit-quant mtdit_inout/chip_full.py:58-65, op_emulation.py:177-179,209-217. diffusers' DDPM `alphas_cumprod` is float32: in BF16 0.9994 and 0.9983 both round to 1.0, so at t = 1 `beta_prod_t = current_beta_t = 0` and the step's div is vmul(0, vrcp(0)) = NaN, which spread to the whole sample |
| 0-d float `add_`/`sub_` (Pi0 flow `time += dt`) | host, float32 | pi0-quant TARGET_OPS has no `add_` (quant_vector.py:136-171), so the reference keeps it float32; the loop compare is host anyway. A BF16 vadd would truncate the timestep each step (1.0, 0.898, 0.797, ...) |
| `where(mask, c1, c2)` with two constant branches (Pi0 `torch.where(att, 0, -2.38e38)`) | data: the precomputed additive mask | the where/masked_fill decision above; openpi pi0_pytorch.py:160 |
| sinusoidal timestep table (`create_sinusoidal_pos_embedding`, Pi0 / SmolVLA) under `--chip-rope host` | host table, like RoPE | angles reach 2π/4e-3 = 1571 rad: outside vsin/vcos's range and 8 rad per BF16 ulp; the table depends only on the step schedule |
| `--chip-rope host` the default of `chip` for every model (`chip_rope_vpu` keeps the reference treatment) | RoPE and timestep sin/cos tables are compile-time host constants | on a deployment, position- and schedule-only tables are constants; vsin/vcos take \|x\| < ~400 rad and BF16 angles have an 8 rad ulp near 1000, so on-VPU tables are wrong for long positions (Pi0 findings); the reference computes them on the VPU (quant_vector.py:170-171) |

The model-level op -> unit map (with the aten ops each model operation becomes) is `MODEL_OPS` in
`src/chip/op_unit_map.py`; the aten-level map with instruction sequences and citations is `ATEN`.

## Multi-Task DiT

multitask-dit-quant's op router accepts an external emulator. With this repo's `src` on
`sys.path`:

    from mtdit_inout.op_emulation import load_op_emulator
    em = load_op_emulator("chip_coverage")          # chip_coverage.make_mtdit_op_emulator()

`em.classify(func, args, kwargs)` returns "VPU" / "scalar" / None (None for data movement, which
that router counts as movement, and for matmuls, which its FP8 patches own: a matmul reaching the
router is host); `em.emulate(func, args, kwargs)` runs the op on ChipVPU with the chip profile's
handlers. Ops the emulator cannot run are classified None, i.e. host fallback in that report.

## Findings for the Atlas hardware team

- `vexp.bf16` (reference FPEX model, bit-exact to ExpLane.scala) is correct only for x >= -336; for
  more negative inputs the Q(m,n) conversion wraps (exp(-1000) = 2.6e10, exp(-3.3e38) = 1.0). Masked
  softmax with the framework's min/-inf mask is wrong unless software clamps first. Is a saturating
  underflow (return +0) intended in the RTL?
- `vsin`/`vcos` take a Q(9,12) argument (x 2/pi), i.e. |x| < ~400 rad, and BF16 angles have an ulp of
  0.5 rad at 64-128: RoPE angles for positions beyond a few hundred are not representable on the VPU.
  This repo now precomputes them on the host by default (`chip`); `chip_rope_vpu` measures the
  on-VPU alternative. Is that the intended deployment?
  Pi0 is worse: prefix positions to ~880 and flow-timestep angles to 1571 rad (max |sin - sin_ref|
  up to 2.0 on the VPU; 406 of 1024 timestep-embedding values off by more than 0.1 at t = 1).
- `vexp.bf16` wraps already at x = -336 (sigmoid via the reference SiLU sequence comes out 0 for
  x = 336), not only below it.
- `vadd`/`vsub` truncate the FP32 sum to BF16 (raw upper half, AddSubSumVec.scala:133) while
  `vmul` and `vredsum.row` round to nearest even: intended?
- The scalar core has no multiply/divide and no FP (ScalarALU.scala:26-36): float schedule
  constants (flow-matching dt, DDPM coefficients) and integer index multiplies need host
  precomputation or software routines (this repo assumes a software float32 routine for
  one-element float32 arithmetic: the DDPM coefficients in BF16 make the step NaN).
- No select/compare instruction on the VPU: masks must be additive and precomputed on the host.
