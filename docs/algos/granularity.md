# `granularity`: per-channel, per-token and 32-element-block FP8 scales

`--algo granularity:w=tensor|channel|block32,a=tensor|token|block32,rule=ref|nc,attn=head|token`.
Defaults: `w=tensor,a=tensor,rule=ref,attn=head`. These give the reference profile bit for bit
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

## Hardware rulings needed

| # | Question |
|---|---|
| H1 | Is the per-row / per-column output vmul the intended rescale path, or is a per-lane scale planned in the pop path? `PopAccBF16` has no scale; `MxuCmd.scaleE8M0` is one per instruction, `MxuBundles.scala:31,47`. |
| H2 | How is a per-row exponent derived from `vredmax.row` on chip? The VPU has no floor or exponent-extract instruction. This is open for per-tensor scales too. |
| H3 | Per-block scales inside the reduction: is a pop/vmul/push of the partial sum per K tile acceptable, or will the compute request get an exponent input? RTL `ComputeReq` has no scale field, `MxuBundles.scala:60-67`. |
| H4 | Is the scale encoding E8M0 only? `software-spec.yaml` lists `scale_encoding` and `block_size` as unknown. |
| H5 | Should FP8Pack keep subnormals? OCP MX requires them. |

## Parameters and records

- No calibration is used. All activation scales are dynamic; weight scales are computed once and
  recorded as `dynamic=False`.
- `algo.granularity` counts the scales per layer and site, e.g. `linear:per_token`,
  `attention.q:per_token`, `linear:per_block(block=32)`.
- `algo.numerics` records the rule and whether it departs from the reference.
