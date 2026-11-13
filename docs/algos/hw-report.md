# Hardware cost report: per-layer shape, DMA, MXU and flush accounting

Source of truth for the numbers: `src/hw_cost.py` (model), `scripts/hw_report.py` (the report),
`scripts/npu_model_l2.py` (simulator check), `tests/test_hw_cost.py`. Every hardware constant comes from
`src/hw_params.py`; where the frozen spec and the RTL disagree the RTL wins. Results:
`results/hw-verify/hw_report_tinyllama.json` and `results/hw-verify/l2_npu_model.json`.

Labels: **MEASURED** read from the export (shapes, code histograms, scale exponents) or from a simulator run.
**DERIVED** arithmetic on `hw_params` constants. **INFERRED** reasoning over RTL text that no run confirmed.
**NOT MEASURED** needs captured activations.

Run: `PYTHONPATH=$PWD/src python -I scripts/hw_report.py [--export model.safetensors] [--T 1 32 128 2048]
[--mxu ipt|sa]` (stdout: per-type table; JSON to `results/hw-verify/`).

## Input

TinyLlama-1.1B-Chat atlas_v1 export (`export_smoke/tinyllama/atlas_v1/model.safetensors`): 154 quantized
2-D linears (7 types x 22 layers; `lm_head` is skipped and stays BF16), stored as `<site>.weight_codes`
(uint8 E4M3, shape [N, K]) and `<site>.weight_scale_exp` (int32 [N, 1], one exponent per output channel).
`policy.json` gives the per-site rules (activation scale: dynamic per token on all 154; attention operands
q, k, p, v: dynamic per (batch, head)).

## Aggregate (MEASURED shapes, DERIVED cycles)

| Quantity | Value |
|---|---|
| Quantized matrices | 154, all N and K multiples of 32: **padding waste 0 for every layer** |
| FP8 weights | 968,884,224 B = **924.0 MiB**, 946,176 32x32 tiles |
| One full weight pass, 4 KiB DMA bursts (DERIVED) | 722,878,464 cycles = **1.446 s at 500 MHz** |
| Same, weights left as stored (32 B pieces; DERIVED **upper bound**, serialized one-in-flight; pipelined cost not measured) | <= 1,665,269,760 cycles = 3.331 s |
| Weight zero codes (upper bound on flushed mass) | 305,850 of 968,884,224 = **0.0316 %** |
| Subnormal-exponent codes (0x01-0x07, 0x81-0x87) / reserved (0x7F, 0xFF) | 0 / 0 |

DMA rate is `hw_params.DMA_LOAD_CYCLES[4 KiB]` = 3056 cycles per 4 KiB (1.34 B/cycle, a VCS measurement). It is
not scaled by the 8 channels: the memory port streams about 23 cycles per 32 B beat whatever the channel count
(`npu-model/docs/rtl-timing.md`), and the simulator run below, with two transfers in flight, still lands at
about 751 cycles per tile: **764 cycles per 1 KiB weight tile** for every layer.
MXU cost is 32 cycles per tile per 32-token block (`MXU_ISSUE_GAP`) plus the pipeline fill once (35 IPT, 95 SA).

### Per layer type

| type | count | N x K | tiles/layer | MiB/layer | DMA cyc/layer | MXU cyc/layer (one 32-token block, IPT) | compute-bound at | fits VMEM (1.5 MiB) | zero-code frac (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| q_proj | 22 | 2048 x 2048 | 4096 | 4.00 | 3,129,344 | 131,107 | 24 blocks (737 tokens) | no | 0.0617 % |
| k_proj | 22 | 256 x 2048 | 512 | 0.50 | 391,168 | 16,419 | 24 blocks (737 tokens) | yes | 0.0897 % |
| v_proj | 22 | 256 x 2048 | 512 | 0.50 | 391,168 | 16,419 | 24 blocks (737 tokens) | yes | 0.0220 % |
| o_proj | 22 | 2048 x 2048 | 4096 | 4.00 | 3,129,344 | 131,107 | 24 blocks (737 tokens) | no | 0.0258 % |
| gate_proj | 22 | 5632 x 2048 | 11264 | 11.00 | 8,605,696 | 360,483 | 24 blocks (737 tokens) | no | 0.0234 % |
| up_proj | 22 | 5632 x 2048 | 11264 | 11.00 | 8,605,696 | 360,483 | 24 blocks (737 tokens) | no | 0.0228 % |
| down_proj | 22 | 2048 x 5632 | 11264 | 11.00 | 8,605,696 | 360,483 | 24 blocks (737 tokens) | no | 0.0374 % |

**The weight-DMA compute-bound threshold is the same for every layer type** (it is DMA per tile over MXU per tile:
764 / 32 = 23.875 blocks, so 24 blocks; 764 tokens continuous, 737 as the smallest token count that reaches
24 blocks). Layer type changes only the absolute cycles. Tokens here means rows pushed through one resident
weight tile, i.e. prefill length or decode batch x speculation depth. A single decode token (T=1) uses 1/24
of the MXU time of the DMA at best (`mxu_over_dma_T` = 0.04).

**This threshold counts weight DMA only (DERIVED); two more terms move it.** (1) Each 32-token block also stores
one BF16 output tile (2 KiB, `DMA_STORE_CYCLES[4 KiB]`/2 = 1410 cycles) per output tile on the same port, against
Kt x 32 MXU cycles per output tile. Including the stores the threshold becomes
b >= Kt x 764 / (Kt x 32 - 1410): **77 blocks (about 2.5k tokens) at K=2048 (q, k, v, o, gate, up) and 32 blocks
(about 1k tokens) at K=5632 (down_proj)**, so with outputs charged it depends on K
(`hw_cost.compute_bound_blocks_with_stores`). (2) VMEM: the FP8 activations of T tokens need T x K bytes, so
at K=2048 the whole 1.5 MiB holds 768 tokens (24 blocks, nothing else resident) and at K=5632 only 279 tokens
(`vmem_activation_token_cap`); a 2048-wide BF16 output stops fitting beyond 384 tokens. Whether any schedule
reaches either threshold is open.

MXU cost does not drop below 32 tokens: the IPT sequencer iterates `tileRows` = 32 rows for every matmul
(`InnerProductTreesSequencer.scala:191-193`), so T=1 costs the same MXU cycles as T=32. (Flag for
`hw_params.mxu_tile_cycles`: it returns `tokens` below 32; `hw_cost.mxu_cycles` does not use it.)

### Whole-model weight pass vs tokens (DERIVED, IPT)

| T | MXU cycles / pass | MXU seconds | DMA cycles / pass | bound |
|---|---|---|---|---|
| 1 | 30,283,022 | 0.061 | 722,878,464 | DMA |
| 32 | 30,283,022 | 0.061 | 722,878,464 | DMA |
| 128 | 121,115,918 | 0.242 | 722,878,464 | DMA |
| 2048 | 1,937,773,838 | 3.876 | 722,878,464 | compute |

Caveat (INFERRED): reusing a pushed weight tile across more than 2 blocks needs more accumulators than the
hardware has (two per MXU), so the software has to pop and re-push BF16 partial sums or re-DMA the tile; the
compute-bound row for T >= 737 is a bound on the MXU/DMA ratio, not a schedule. The simulator run below can
reuse a pushed tile for 2 blocks.

## Weight flush mass (upper bound)

`zero_code_frac` counts weight elements whose E4M3 code is 0x00 or 0x80. The encoder flushes values below 2^-6
to code 0x00 and never emits 0x80, but a code of zero also arises from a true zero, so **this is an upper
bound on flushed mass (flushed values plus exact zeros), not the flushed mass.** Overall 0.0316 %; per type
0.022 % to 0.090 %; the worst single matrix is `layers.0.self_attn.k_proj` at 0.716 %; 4 of 154 matrices exceed
0.1 %. No subnormal or reserved codes occur. Per-channel exponents put the largest weight of each row near
the top of the range, so little mass falls under 2^-6. (The smoothing and GPTQ passes changed the weights, so
the original checkpoint's zero count cannot correct this bound.)

**Activation flush mass: NOT MEASURED.** It needs captured activations after the scale is applied.

## Weight layout: does the export feed the weight push without a transpose?

- Stored (MEASURED): `weight_codes` is [N, K] uint8 row-major, row n = output channel n; the shape equals the
  site's `weight_shape` [N, K] for all 154 sites; `weight_scale_exp` is [N, 1].
- RTL (settles it, lane orientation): `InnerProductTreesSequencer.scala:424-428` writes mreg row r into weight
  lane `laneIdx = r` (`WeightBuffers.scala:62-67` stores it as `slot(lane)`);
  `InnerProductTrees.scala:60-64` gives output column n the lane-n vector;
  `AnchorAccumulationTree.scala:53` multiplies weight element j with activation element j. The systolic array
  agrees: `SystolicArrayTop.scala:116-117` wires `weights(i)(j) = slot(j)(i)` (lane j = output column j, i = K).
  So an mreg whose row n holds `W[n0+n, k0:k0+32]` is exactly the push image: **the [N, K] export feeds the
  push with no transpose** on either MXU.
- Corroboration (MEASURED in the simulator): `npu-model`'s shipped matmul stores B as [K, N] and runs
  `vtrpose.xlu` (66 cycles) before each push ("MXU weights are streamed as columns"). Our L2 kernel pushes the
  untransposed [N, K] tile and matches `accum_torch.ipt_linear_bits` bit for bit (below).
- INFERRED: how a DMA'd tile becomes an mreg (row r = 32 consecutive bytes) is read off `vload` usage in the
  shipped kernels and confirmed only by that simulator run, not by the RTL read.
- DMA caveat: the export is row-major over the full K, so a tile is 32 pieces of 32 B at stride K, not one
  contiguous 1 KiB. Headline cycles assume an offline re-layout to tile-major (a pure permutation, no
  transpose) so the DMA streams 4 KiB bursts. As stored (32 B transfers) the same pass is **at most** 3.33 s
  (2.3x): that is an upper bound with one transfer in flight at the isolated 55-cycle latency; the pipelined
  cost with several transfers outstanding is not measured (the port rate suggests about 736 cycles per tile)
  and the per-transfer command overhead is unknown. Whether the DMA has a strided mode is not settled here.

## Pop scale: where it is free

**Correction to the audit** (`handoff/HW_AWARE_AUDIT.md` section 5, item 3, "QKV into attention, MLP up into
down"). The audit names the wrong sites for these models. The classification below is derived by running one
HF `LlamaDecoderLayer` (TinyLlama dimensions, eager attention, meta device, no weights) under a
`TorchFunctionMode` and following each linear's output through layout-only ops (`view`, `transpose`,
`reshape`, `contiguous`, `expand`, indexing) to the first value-changing op (`hw_cost.trace_layer_graph`,
`classify_linears`; pinned by `tests/test_hw_cost.py`).

| Producer | First non-layout consumer (from the graph) | Class |
|---|---|---|
| q_proj | `mul`, `neg`, `cat` (RoPE) | VPU op |
| k_proj | `mul`, `neg`, `cat` (RoPE) | VPU op |
| v_proj | `matmul` arg 1 (P.V), after view/transpose/index/expand/reshape (GQA `repeat_kv`) | FP8 matmul operand |
| P.V (attention matmul) | `linear` arg 0 (o_proj) after transpose/contiguous/reshape | FP8 matmul operand |
| QK^T | `mul` (1/sqrt(d)), then softmax | VPU op |
| o_proj | residual `add` | residual |
| gate_proj | `silu` | VPU op |
| up_proj | `mul` with SiLU(gate) | VPU op |
| down_proj | residual `add` | residual |

TinyLlama's MLP is SwiGLU: `SiLU(gate) * up` runs in BF16 on the VPU before `down_proj`, so up_proj does not
feed down_proj as an FP8 operand. Q and K go through RoPE on the VPU before QK^T. **Only v_proj (into P.V)
and the P.V output (into o_proj) are direct FP8-matmul-to-FP8-matmul edges.**

RTL semantics (what the pop can carry): `vmatpop.fp8` reads one scale-file register (`ScalarCore.scala:184-187`,
`scaleE8M0 := scaleRF.io.regs(vs1(4,0))`; the field defaults to 0 for every other MXU op, line 179) and
`OutputConvStage.scala:120-173` applies that one signed exponent to the whole 32x32 tile; `vmatpop.bf16`
reads no scale. So the pop scale is one 2^e per 32 tokens x 32 output channels.

A pop scale is free only if all of these hold (`hw_cost.rescale_plan`):

1. the output feeds an FP8 matmul operand directly (only layout movement between);
2. the producer's own activation scale sx is **static** (a per-token dynamic sx is one exponent per row, which a
   per-tile scale cannot carry);
3. the weight scale sw is **constant within each 32-output-channel group** (the pop carries one value per tile);
4. the consumer's operand scale is static and constant along the consumer's K axis (the accumulator holds BF16
   partial sums across K tiles, so a scale that varied along K would need a rescale per K tile).

**As exported, none of the 154 linears qualifies.** All activation scales are dynamic per token, and the
per-output-channel weight exponents are almost never uniform within a 32-channel group (MEASURED, fraction of
groups with a single exponent, mean over the 22 layers): q 0.0 %, k 0.0 %, up 0.0 %, v 0.6 %, down 1.7 %,
gate 4.6 %, o 10.9 %. The attention operands (including V and P) are dynamic per head in every recipe, so
condition 4 fails too. It becomes available for v_proj -> P.V and P.V -> o_proj only with (a) a static
activation scale for v_proj's input and for the attention operands (the dynamic per-head rule is the opposite
choice), (b) a group-of-32 weight scale for v_proj (a re-quantization; its accuracy cost is not measured), and
(c) a static o_proj input scale. V as the right operand of P.V contracts over tokens, so the MXU weight side
needs V^T: `vtrpose.xlu` per V tile (`hw_params.VTRPOSE_CYCLES` = 66; for T tokens that is ceil(T/32) x 8 tiles
per layer) or swapped operand roles. P.V output heads are 64 channels = 2 groups, so a per-head scale is
group-uniform by construction.

Fallback (everywhere else): `vmatpop.bf16` (writes two mregs) followed by `vmul` with 2^(sx+sw): one vmul if sx
is static or per-head (a per-column tile built once), **two** if sx is per-token (row factor and column
factor), plus a `vpack` when the consumer is FP8. Costs use `VPU_TILE_ELEMENTWISE_CYCLES` = 65 per 32x32 tile
for each vmul and, ASSUMED, for the vpack (hw_params has no vpack latency).

Extra VPU cycles per forward over 22 layers (DERIVED; excludes the dynamic-scale chain, softmax, RoPE, norms):

| T | as exported (per-token sx, per-channel sw) | best case (static sx, group-32 sw, static consumer; free on v_proj and P.V only) |
|---|---|---|
| 1 or 32 | 1,796,080 cycles (3.6 ms) | 789,360 (1.6 ms) |
| 128 | 7,184,320 (14.4 ms) | 3,157,440 (6.3 ms) |
| 2048 | 114,949,120 (0.230 s) | 50,519,040 (0.101 s) |

The best case still costs 44 % of the as-exported rescale work, because q, k, o, gate, up and down all
feed VPU ops or residuals and need the vmul regardless; the free pop removes only the v_proj and P.V terms
and the per-token second vmul. gate needs its real-valued output before SiLU (a nonlinearity). up's consumer
is the `mul`, which is linear in up, but its per-channel factor would then sit on down_proj's K axis: a `vpack`
carries one scale per tile and the BF16 accumulator cannot vary a scale along K, so only an offline fold into
down_proj's weight columns could absorb it (a recipe change, not measured here).

## L2: cycles on npu-model

**Setup.** `npu-model` cannot be imported under the existing Python 3.12 env: `npu_model/isa.py:91` uses PEP 696
type-parameter defaults (`class IType[RD: (...) = ScalarReg, ...]`), a `SyntaxError` before 3.13 (`rtl_math.py`
alone imports fine, which is why the L0 cross-check works). `uv venv --python 3.14` downloaded CPython 3.14.8
and `uv pip install numpy tabulate torch` (CPU wheel index) completed without a permission refusal. Run:
`python -I scripts/npu_model_l2.py --py314 <venv>/bin/python --full --extra-burst1`. The simulator checkout is
only read (no `.pyc`, no output inside it). The shipped kernels have fixed 64-cube shapes, so
`scripts/npu_model_l2.py` generates its own assembly: MXU1 (inner-product tree, `--numerics rtl`), weight tiles
pushed as stored ([N, K] 32x32 tiles, **no `vtrpose.xlu`**), 4 KiB DMA bursts with the next burst prefetched,
explicit `delay`s for every dependency (the model has no scoreboard, as in the shipped kernels). One or two
32-token blocks share a pushed weight tile (two accumulators per MXU is the hardware limit). Real export weight
codes (layer 0) with random activations quantized by `ref_numerics.e4m3_vpack`.

**Bit-for-bit (MEASURED): 0 mismatches of 504,832 BF16 output words** (23 runs, K up to 5632 chained over 176
tiles) against `accum_torch.ipt_linear_bits` (numba IPT) on the same A and W code bytes. This also shows the
untransposed [N, K] tile is the correct push image in the simulator.

**Cycles (MEASURED in the simulator; whole layers, 32 tokens, including the one-time activation preload and the
output stores):**

| layer (TinyLlama) | N x K | weight tiles | simulated cycles, 32 tokens | ms at 500 MHz | simulated, 64 tokens (2 blocks share a pushed tile) | hw_cost weight-DMA bound (DERIVED) |
|---|---|---|---|---|---|---|
| k_proj / v_proj | 256 x 2048 | 512 | 450,392 | 0.90 | 513,532 | 391,168 |
| q_proj / o_proj | 2048 x 2048 | 4096 | 3,262,936 | 6.53 | 3,428,164 | 3,129,344 |
| down_proj | 2048 x 5632 | 11264 | 8,732,229 | 17.46 | extrapolated 8,982,368 | 8,605,696 |
| gate_proj / up_proj | 5632 x 2048 | 11264 | 8,888,024 | 17.78 | extrapolated 9,257,254 | 8,605,696 |

Per weight tile: 785 cycles (K=2048 strips of 64 tiles), 763 (K=5632 strips of 176) against 764 derived from
`hw_params`. The strip cost fits a fixed 2.1k cycles per strip (the first burst is not hidden: no prefetch across
strips) plus **751 cycles per tile**, 1.7 % under the hw_params 764. This is not independent evidence of the DMA
rate: npu-model's DMA backend is a curve fitted to probe runs on the same VCS target that
`hw_params.DMA_LOAD_CYCLES` reads (a lone 4 KiB load costs about 3,038 cycles in the activation preload against
3,056 measured).
Kernel overheads put full layers 1.5-4.3 % above the weight-DMA bound for q, gate and down (the activation preload, strip starts and stores); k_proj is 15 % above because its 48.6k-cycle preload is large next to 512 tiles. Extrapolating
setup + strips from a 1- and 2-strip run reproduces the full-layer runs to 0.01 % (the full runs are the
measured numbers). The simulated kernel is DMA-bound: the MXU1 is busy 68 cycles per tile of the 785 (its
serialized compute is 135 (1 block) to 204 (2 blocks) cycles per tile, hidden under the next DMA), which is the same conclusion as the
analytic table: at 32 tokens the weight DMA, not the MXU, sets the layer time. Two token blocks sharing a tile
cost 5 % more cycles for twice the tokens.

Whole model at 32 tokens, from the simulated layers (v = k, o = q, up = gate by shape; **extrapolated across
layers**): 746.6M cycles = **1.493 s at 500 MHz**, against the hw_cost 722.9M (1.446 s): the simulator is 3.3 %
slower, the difference being the activation preload, the output stores and strip starts that the weight-only
estimate leaves out. The close agreement mostly shows that both read the same DMA calibration; what L2 adds on
its own is the bit-exact output, the untransposed push image, and the size of the kernel overheads (1.5-4.3 %,
15 % for k_proj).

1 KiB transfers (`--extra-burst1`) with one transfer prefetched ahead cost 778 cycles per tile in the simulator,
below the isolated-transfer 797 in `hw_params.DMA_LOAD_CYCLES[1 KiB]`: overlapping two loads hides part of the
lead-off latency, so 797 is an upper figure once a software pipeline is in place.

Not simulated: more than 2 token blocks per pushed tile (the second pop/push-acc schedule), the SA unit (MXU0),
and a schedule that overlaps strips. Cycle counts are a property of this kernel's explicit `delay`s plus the
simulator's DMA model; they are not RTL measurements.

## What is measured, derived, inferred

| Item | Status |
|---|---|
| N, K, bytes, tiles, padding, code histograms, group-uniform fractions | MEASURED (export) |
| DMA cycles, MXU cycles, seconds at 500 MHz, reuse threshold, vmul costs | DERIVED (hw_params) |
| DMA rate | hw_params, a VCS measurement of one channel; npu-model's DMA model is calibrated to it |
| Weight push hidden behind matmul by two weight slots; reuse beyond 2 blocks | INFERRED |
| No-transpose weight push | RTL read (lane orientation), simulator run (vload layout) |
| Pop-scale applicability | graph MEASURED (meta-device trace), RTL read, costs DERIVED |
| Weight flush mass | upper bound only |
| Activation flush mass | NOT MEASURED |
