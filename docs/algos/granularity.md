# `granularity`: per-channel, per-token and 32-element-block FP8 scales

`--algo granularity:w=tensor|channel|block32,a=tensor|token|block32|block,rows=32,rule=ref|nc,attn=head|token`.
Defaults: `w=tensor,a=tensor,rule=ref,attn=head` (`a=block` and `rows` are **NEW** opt-ins; the named
`atlas_v1_*` recipes do not use them and expand exactly as before). These give the reference profile bit for bit
(`tests/test_algo_granularity.py::test_defaults_equal_none`).

Everything in this doc is supported by CPU screening on TinyLlama only (4 windows). None of it is a
final result.

## Idea and sources

- The reference scales every matmul operand per tensor (pi0-quant `quant_linear.py:153-154`) with
  `s = floor(log2(amax_bf16 / 256))` (`hw_float_ops.py:31,38-57`).
- `w=channel` and `a=token` apply the same rule along `[out, 1]` and `[..., T, 1]`.
- `block32` applies it per 32-element block along K, as in OCP Microscaling v1.0.
  - Blocks: sec. 5.1 and Table 1 in sec. 5.2, k = 32.
  - Scales: E8M0 (sec. 5.4.1).
  - Conversion: sec. 6.3. Its rule `floor(log2 amax) - emax_elem` is the reference rule, the same
    as MX paper arXiv 2310.10537 Algorithm 1 and torchao 0.18 `mx_formats` FLOOR.
- Atlas's vpack flushes subnormals to zero (`FP8Pack.scala:57-95`). OCP sec. 5.3.1 requires them, so
  `block32` is MX-like, not OCP-compliant.
- `rule=nc` is **NEW** and departs from the reference rule. It raises the exponent by one where the
  maximum would clip (`amax > 448 * 2^s`), which equals `ceil(log2(amax / 448))`. The same rule
  appears as:
  - the collaborator's weight rule (`atlas_quant.py`);
  - torchao mx `RCEIL` (`prototype/mx_formats/config.py:41-66`);
  - microscaling-quant `ocp_no_clip` (`mxq/scale_factor.py:6-24`).

  Every result records `algo.numerics.departs_from_reference_rule = true` and a note.
- Why `nc` exists: with power-of-two scales, E4M3 rounding does not depend on the scale anywhere in
  the normal range. So a finer granularity changes only two things:
  - which elements are flushed to zero;
  - which block maxima the reference rule clips (`amax` in `(448, 512) * 2^s`).

  Finer granularity clips more maxima. Screening, as change in PPL vs bf16:

  | Granularity | `rule=ref` | `rule=nc` |
  |---|---|---|
  | per-tensor (reference) | +0.84% | – |
  | per-channel W | +1.03% | – |
  | MX32 | +0.98% | +0.59% |
  | per-token A + per-channel W | +0.45% | +0.38% |

  Scripts and data are in `out/artifacts/handoff/atlas-fp8/algos/granularity/` (merlin-wt-tracka).

## Where each scale acts on Atlas

- **Per-channel weights and per-token activations.** These scales commute with the K reduction, so
  they are an exact power-of-two rescale of the MXU's BF16 output: `2^(sx[m] + sw[n])`. This is
  `quant_linear.py:169-170` generalised from scalars to a table (`algos.base.output_scale`).
  - They run on every reference step, including `reference_hw` and `chip`.
  - `tests/test_algo_granularity.py::test_output_rescale_exact_on_mxu_model` checks bit-exactness on the IPT and SA
    models.
  - On `chip`, the coverage report has a VPU row `granularity_rescale`, listed under
    `new_decisions`. It counts:
    - one `vmul.bf16` per activation element: the 2^-sx multiply before `vpack`, which takes one E8M0
      per instruction (green card 57/44);
    - one more `vmul.bf16` per output element when the layer has both a row factor and a column factor.
  - The row amax (`vredmax.row` / `vredmin.row`) and `vpack` are counted by the cast itself.
  - The output exponent table is not counted as a `vexp2` per output element.
- **`block32`.** These scales sit inside the K reduction. They work only on the framework steps
  (`ref_linear` .. `ref_conv`); under `reference_hw` and `chip`, algo/base raises
  `NotImplementedError` (hardware ruling H3).
  - A K that is not a multiple of 32 falls back to the per-token / per-channel scale for that
    operand. It is recorded as `per_token(block_tail)` / `per_channel(block_tail)`.
  - A software path that would be bit-equal to an MXU scaling each K tile was screened on the
    bit-exact IPT model. It pops the BF16 partial sum per 32-wide K tile, multiplies it by
    `2^(e_{t-1} - e_t)` and pushes it back (`vmatpush.acc.bf16`).
  - It is not implemented here:
    - the reference accumulation-model contract has no partial-sum input, so it would only work on
      the torch ports;
    - it needs ruling H3.
- **Attention (`attn=token`).** Per-row scales apply to the left operands `q` and `p` only, where
  rows are output rows. `k` and `v` keep the per-(batch, head) scale: a per-row `v` scale would lie
  along the P·V reduction.

## `a=block`: one scale per block of tokens over the full K (NEW)

`a=block,rows=B` (default `B=32`) gives one power-of-two activation scale per `B` consecutive tokens,
taken over the full K, never-clip (`nc`) rule regardless of `rule=`. Why full K: the accumulator holds
BF16 partial sums across K tiles, so an activation scale must be constant across K for a token block
(otherwise each K tile needs a rescale before accumulation, about 65 VPU cycles against 32 MXU
cycles). One 32-token block is exactly one `vpack` scale (one E8M0 per 32x32 tile).

- Scope: Linear inputs only. Attention operands keep the per-(batch, head) rule, other kinds (Conv2d)
  fall back to the `a=tensor` behaviour. Tokens are the flattened leading dims of the input (the matmul
  M dimension); a count not divisible by `B` leaves a short last block (its scale uses only its rows).
- Equivalences (tests): `rows=1` equals `a=token,rule=nc`; `rows >= tokens` equals `a=tensor,rule=nc`.
- Runs on `reference_hw` and `chip` (it commutes with the K reduction; contrast `block32` along K, which
  does not). The exponent is returned per token, so the cast and the output rescale are those of
  `a=token`. Results record `departs_from_reference_rule = true`; stats key `linear:per_token_block`.
- Chip cost: `dyn_scale_chain` rows, below. No pre-vpack `vmul`, because `vpack` reads the block's scale register.

## Dynamic scale chain (NEW, closes H2; `src/hw_scale.py`, `chip_profile.count_dynamic_scale_chain`)

The real route from data to a scale register: `vredmax.row`/`vredmin.row` (34 cycles) -> `vstore` (35) ->
scalar `lhu` of the BF16 amax -> RV32I integer ops -> `sb` of the E8M0 byte -> `seld` (4). The integer step
(`scale_exp_rv32`) is a pure RV32I sequence (srli/andi/addi/slli/add/beq/bge/jal; no multiply, no float):
E8M0 byte = e - 8 + bump with bump = (m + 31) >> 7 (1 iff the BF16 mantissa m > 96, i.e. amax above
1.75 * 2^E, which would exceed 448 after scaling); amax 0, Inf and NaN give s = 0 (the reference's
`nan_to_num`); tiny amax clamps to byte 0. `tests/test_hw_scale.py` proves it equal to
`po2_scale_exp` (`rule=ref`) and to `rule=nc` on all 65,536 BF16 patterns (the model has no clamp, so
equality holds against `max(s, -127)`; the clamp fires only below about 2^-119). BF16 rounding is monotone,
so amax after BF16 rounding equals the BF16 of the fp32 amax: model and chip agree exactly.

The coverage report keeps the old 3-instruction `fp8_operand_cast` count and adds, for `a=token` and `a=block`,
VPU row `dyn_scale_chain` and scalar row `dyn_scale_chain_scalar` plus a `dynamic_scale_chain_NEW` section with
serialized cycles (sum of latencies, no overlap claimed; scalar CPI 1 and the layout are DERIVED
assumptions, listed in the report). Example, a layer input of 128 tokens x K=2048 (TinyLlama hidden size),
instructions / serialized cycles: per-token 4,480 / 85,908; block32 (per-tile chain, scalar cross-K max) 3,112 /
86,068; block32 with the K max taken in the VPU (`block_vec`) 2,104 / 67,672. Block32 saves instructions in all
variants and cycles only in the vector variant; the per-token factor vector is costed as a lower bound
(a single `vload` broadcast).

## Hardware rulings needed

| # | Question |
|---|---|
| H1 | Is the per-row / per-column output vmul the intended rescale path, or is a per-lane scale planned in the pop path? `PopAccBF16` has no scale; `MxuCmd.scaleE8M0` is one per instruction, `MxuBundles.scala:31,47`. |
| H2 | **Closed in software (NEW, `src/hw_scale.py`).** The VPU has no exponent-extract instruction; the exponent is derived on the RV32I scalar core from the BF16 amax pattern in 10 instructions on the normal path (15 static, 14 worst; 12 and 16 with `lhu` and `sb`): see "Dynamic scale chain". Hardware can still shorten it (an exponent-extract instruction or the unimplemented scale CSR window). |
| H3 | Per-block scales inside the reduction: is a pop/vmul/push of the partial sum per K tile acceptable, or will the compute request get an exponent input? RTL `ComputeReq` has no scale field, `MxuBundles.scala:60-67`. |
| H4 | Is the scale encoding E8M0 only? `software-spec.yaml` lists `scale_encoding` and `block_size` as unknown. |
| H5 | Should FP8Pack keep subnormals? OCP MX requires them. |

## Parameters and records

- No calibration is used. All activation scales are dynamic; weight scales are computed once and
  recorded as `dynamic=False`.
- `algo.granularity` counts the scales per layer and site, e.g. `linear:per_token`,
  `attention.q:per_token`, `linear:per_block(block=32)`.
- `algo.numerics` records the rule and whether it departs from the reference.
