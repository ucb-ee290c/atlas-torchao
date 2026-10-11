# vpu-numerics: vadd/vsub numerics for the `chip` profile

`--chip-vadd {trunc,rn,rne_hw}` and `--chip-vadd-sites {residual,data,all}` (`src/chip_numerics.py`,
wired through `src/chip_profile.py`, `src/chip_coverage.py` and `src/quant_modes.py`). The flags are
chip-profile options, not `--algo` quantization hooks: they change how the VPU adds, not how operands
are cast. The default, `trunc`, is the chip profile exactly as on `l40s-run-2.1`
(`tests/test_algo_base.py`, `tests/test_algo_vpu_numerics.py`).

## Why

The Atlas vadd/vsub widen BF16 to FP32, add once with RNE, and keep the raw upper 16 bits:

- `AddSubSumVec.scala:49` `roundingMode := 0.U(3.W)  // May change this later` (FP32 RNE add);
- `AddSubSumVec.scala:133` `fNFromRecFN(BF16T.expWidth, computeSigWidth, v)(31,16)` (no second
  rounding: the BF16 result is the FP32 sum truncated toward zero);
- reference model `funct_models_vector/lane_boxes/add_sub_sum_vec.py:281-288`; emulated bit-exactly by
  `chip_ops.ChipVPU.add/sub`.

Against correct rounding, vadd differs on 27-50% of lanes and is biased by about -0.4 ulp
(-2.3e-3 relative at a magnitude ratio of 0.1). On a residual stream the bias accumulates layer
after layer.

CPU screening (TinyLlama seqlen 512 x 2 windows and 2048 x 1, SmolVLA 2-4 frames; small samples,
scripts and outputs in `out/artifacts/handoff/atlas-fp8/algos/vpu-numerics/` of the merlin tree)
reverted one VPU op class at a time to framework BF16. With RoPE tables on the host, the residual
adds carry the whole VPU cost on TinyLlama; softmax, RMSNorm/LayerNorm, SiLU/GELU and the RoPE
apply were within noise:

| TinyLlama | KL(bf16 ‖ mode) | ppl |
|---|---|---|
| 512 x 2: chip / all VPU sites native / residual add native | .0257 / .0206 / .0188 | 8.421 / 8.375 / 8.313 |
| 512 x 2: chip + RN residual add (screening sequence) | .0193 | 8.415 |
| 2048 x 1: chip / all native / chip + RN residual add (bf16 5.5434) | .0221 / .0212 / .0190 | 5.6474 / 5.6300 / 5.6132 |

SmolVLA, 4 frames, rel-L2 vs bf16: all native .133, chip .182, chip + RN residual add .141. The
single-class reverts on 2 frames were noise-dominated (.145 to .197) and do not attribute SmolVLA.
(The screening used a 19-op version of the sequence; the shipped 24-op version adds the m2 gate below,
which only changes lanes whose operands are more than 16 bits apart.)

## Modes

| `--chip-vadd` | What | Instructions per add | Ladder |
|---|---|---|---|
| `trunc` (default) | the hardware as built | 1 | unchanged |
| `rn` | NEW, ISA only: `chip_numerics.vadd_rn`, round to nearest (ties away from zero) | 24 | off ladder: "NEW round-to-nearest vadd sequence" |
| `rne_hw` | model of a hardware change: RNE narrowing in place of the upper-16-bit slice | 1 | off ladder: **pending hardware ruling** |

`vadd_rn` (instructions from green card section 3 only: vadd, vsub, vmul, vmaximum, vminimum, vrelu):

1. `s = vadd(a, b)`: the truncated sum.
2. Fast2Sum remainders for both operand orders (Dekker's error-free transformation; with
   |big| >= |small|, `big - s` is exact and `(big - s) + small` is the exact remainder in vadd's FP32
   adder): `r1 = vadd(vsub(a, s), b)`, `r2 = vadd(vsub(b, s), a)` (for vsub: `(a - s) - b` and
   `a - (b + s)`).
3. There is no select instruction, so blend with an exact 0/1 mask: `|x| = vmaximum(x, 0 - x)`,
   `m = vminimum(vrelu(vmul(|a| - |b|, 2^120)), 1)`, `r = m*r1 + (1 - m)*r2`.
4. `rr = r + r`. When the FP32 add itself rounded up in magnitude (operands more than 16 bits apart),
   `r` has the sign opposite to `s` and is far below half an ulp: the answer is `s`. Gate:
   `m2 = vminimum(vrelu(vmul(vmul(rr, s), 2^120)), 1)`.
5. `out = vadd(s, vmul(m2, rr))`: truncation of `s + 2r` lands on `s`'s neighbour away from zero
   exactly when |r| >= ulp(s)/2.

`tests/test_algo_vpu_numerics.py` checks it equals correct rounding off ties (ties: away from zero)
for every BF16 value with exponent in [-30, 30] against partners at magnitude ratios 2^-12 .. 2^12,
and on 4e5 random pairs, for vadd and vsub; and that `rne_hw` is the RNE BF16 of the FP32 sum.

## Sites

| `--chip-vadd-sites` | Which vadd/vsub |
|---|---|
| `residual` (default) | through the router's add/sub handler, two computed (non-constant) operands of the same shape, outside `*Norm*`, `*Attention*`, `*MLP` and FP8-matmul module contexts, not an additive mask (an operand <= -1e30). The residual adds; also same-shape stream updates (flow-matching `x_t + dt*v_t`), and SmolVLA's `apply_rope` combines (a module-level function with no module to hook) |
| `data` | every add/sub through the handler with two computed operands (broadcast allowed), except additive masks: adds the RoPE apply and similar |
| `all` | every vadd/vsub the VPU issues, including inside the softmax / norm / sigmoid / GELU / cumsum decompositions and the bias add after an FP8 matmul (`chip_numerics.NumericsVPU`) |

On a two-layer tiny Llama, `residual` routes 2 adds per layer and `data` 4 (the two RoPE combines
added); the mask add and the norm epsilon add are never routed except under `all`.

**csum** (`vredsum.bf16`, column sum) narrows its FP32 accumulator with the same upper-16-bit slice
(`vector_engine_model.py:393-420`; `ColAddVec.scala:77` RNE accumulate, sliced by the engine). No
decomposition of the chip profile issues csum: row reductions use `vredsum.row`, an FP32 tree with
one RNE at the end (`vector_engine_model.py:318-330`, `SumRedu.scala:83-159`). So there is no csum
path to switch; a sequence that adopts csum should take the same policy (under `rne_hw` the
narrowing would be RNE too). `cumsum` (a chain of vadds) follows `--chip-vadd` under `all`.

## Records and accounting

- `rep["profile"]["vadd"]`: `mode`, `sites`, `instructions_per_add`, `adds_routed`, `extra_lane_ops`
  (23 per element for `rn`), `status` ("NEW ..." or the pending-ruling text). Refreshed at the end of
  every forward.
- The sequence's instructions are issued on the emulated VPU, so the coverage report counts them in
  the instruction mix and work of the op that issued them (e.g. `add`: `vadd.bf16`, `vsub.bf16`,
  `vmul.bf16`, `vmaximum.bf16`, `vminimum.bf16`, `vrelu.bf16`).
- `ladder.off_ladder_reasons` names the flag for `rn` and `rne_hw`.

Cost: residual adds are 0.4% of TinyLlama's VPU lane-ops at 2048 tokens (22 layers x 2 x 2048 per
token, of about 22 M), so `rn --chip-vadd-sites residual` adds about +9% VPU lane-ops (23 extra per
element); `rne_hw` adds nothing.

## Other VPU findings (not implemented; screening showed no gain)

Exhaustive lane-box accuracy over all 65536 BF16 inputs (reference lane boxes via `chip_ops` tables):

| Lane box | Max error (ulp) | Correctly rounded | Note |
|---|---|---|---|
| vexp | 1.24 | 99.2% | FPEX LUT; wraps below -336, hence the chip profile's floor `vmaximum(x, -128)` (`ExpLane.scala:40-47`) |
| vexp2 | 0.53 | 99.9% | |
| vrecip | 0.99 | 59% | 7-bit-address LUT (`Rcp.scala:32`); truncating, mean 0.42 ulp |
| vsqrt | 1.0 | 49% | LUT (`Sqrt.scala:32`); truncating |
| vlog2 | 1.0 | 50% | LUT (`Log.scala:31`) |
| vsquare / vcube | 0.97 / 0.99 | 57% / 55% | truncating |
| vtanh | 1.19 for \|x\| >= 2^-6 | 64% | fixed-point absolute error below 2^-6 (harmless inside 1 + tanh) |
| vsin | up to 177 | 36% | fixed point; the RoPE-table issue (`--chip-rope host`) |

Screened and not adopted (TinyLlama 512 x 2, KL; chip .0257): softmax with `vexp2(z * log2 e)` .0253;
a Newton step on the softmax reciprocal .0261; rsqrt plus a Newton step .0241; SiLU as
`x * 0.5 * (1 + tanh(x/2))` .0268. All within the run-to-run spread of single-class reverts.

## Use

    python src/eval.py --model ... --modes bf16 chip --chip-vadd rn [--chip-vadd-sites residual] --kl
    python src/eval_policy.py ... --modes chip --chip-vadd rn
    python src/eval_pi0.py ... --modes chip --chip-vadd rne_hw

`--kl` (src/eval.py, default protocol): token-mean KL(p_bf16 ‖ p_mode) and top-1 agreement against a
clean bf16 copy of the model on the same windows, written under `kl` in the result JSON. Perplexity
stays the primary metric; KL is less noisy on a few windows.

### Multi-Task DiT (multitask-dit-quant)

That repo builds the router itself (`mtdit_inout/chip_full.py`: `cc.ChipRouter(self.cov, mxu=, mxu_impl=)`,
with this `src` on `ATLAS_TORCHAO_SRC`), and its `load_op_emulator("chip_coverage")` calls
`chip_coverage.make_mtdit_op_emulator()`. Both read `ATLAS_CHIP_VADD` / `ATLAS_CHIP_VADD_SITES` when no
explicit policy is passed (unset: unchanged), so no edit there is needed:

    ATLAS_TORCHAO_SRC=.../atlas-torchao/src ATLAS_CHIP_VADD=rn python experiments/run_eval.py --modes chip ...

With no module context hooks in that path, `residual` there means every same-shape add of two
computed tensors outside masks. The chip profile itself never reads the environment (its CLI flags
define it). The cleaner long-term hook is to pass `vadd=chip_numerics.VaddPolicy(...)` at
`chip_full.py`'s `ChipRouter(...)` call.

## Sources

- Atlas RTL @0079c054 (`generators/atlas-npu/src/main/scala/atlas/vector/laneBoxes/`): AddSubSumVec.scala
  49, 133-134; ColAddVec.scala 54, 77; MulRec.scala 117-124; Rcp.scala 32; Sqrt.scala 32; Log.scala 31;
  ExpLane.scala 40-47.
- Reference VPU model (pi0-quant @8bc0f6a `funct_models_vector/`): vector_engine_model.py 318-330
  (row sum), 343-437 (csum); lane_boxes/add_sub_sum_vec.py 281-288; lane_boxes/col_add_vec.py.
- Green card: `examples/atlas/phase1/contracts/hwbringup_atlas_v0/isa_include/atlas_isa_green_card.md`
  section 3 (no select, compare, FMA or FP32 register state).
- Fast2Sum: T. J. Dekker, "A floating-point technique for extending the available precision",
  Numerische Mathematik 18 (1971) (not re-read for this note; the property used is checked exhaustively
  by the tests above).
