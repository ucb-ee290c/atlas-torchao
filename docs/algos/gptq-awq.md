# `gptq_awq`: GPTQ and AWQ weights under the Atlas E4M3 cast

`--algo gptq_awq[:method=gptq|awq|awq_gptq,actorder=true,damp=0.01,block=128,grid=20]`
(`src/algos/gptq_awq.py`, tests in `tests/test_algo_gptq_awq.py`).

Only Linear weights change. Every activation, every attention operand and every Conv2d weight keeps
the reference rule. Nothing changes at run time: the same E4M3 codes and the same per-tensor E8M0
exponent go to the MXU, just with different code values. With `--algo none` the profiles stay
bit-identical (`tests/test_algo_base.py`).

| method | What it does |
|---|---|
| `gptq` (default) | GPTQ with the Atlas cast as its quantizer, act-order on |
| `awq` | AWQ's search-based per-channel scaling, folded into the producers, then the reference cast |
| `awq_gptq` | AWQ folds, then GPTQ on the folded weights, with the Hessian transformed by diag(1/s) |

## GPTQ

GPTQ comes from Frantar et al., *GPTQ: Accurate Post-Training Quantization for Generative
Pre-trained Transformers*, arXiv 2210.17323.

**Algorithm.** The weight is quantized one column at a time. Each column's error is fed into the
columns not yet quantized through the inverse Hessian H = 2XXᵀ of ||WX − ŴX||² (Sec. 3, Eq. 2). The
implementation follows Algorithm 1:

- Lazy batch updates in blocks of `block` columns (Sec. 4 Step 2, B = 128).
- The upper Cholesky factor of H⁻¹ (Step 3).
- Dampening of `damp` × mean diag(H) (Step 3: 1%).
- Dead columns (diag(H) = 0) are zeroed, as torchao 0.18 does (`prototype/gptq/api.py:392-394`).

**Quantizer.** Eq. 2's `quant(w)` "rounds w to the nearest value on the quantization grid", so the
grid is free to choose. Here it is the Atlas vpack cast at the layer's exponent s₀:
`e4m3_unpack(e4m3_vpack(w, s₀)) · 2^s₀`.

- The cast rounds to nearest even, flushes results below 2^-6 to zero, saturates at 448 and treats
  exponent 15 as finite.
- Sources: `src/reference/ref_numerics.py:27-59`, a bit-exact port of pi0-quant `fp8_e4m3.py:38-86` and the
  RTL `FP8Pack.scala:57-95`.
- The error fed back is therefore the true FP8 error, never a uniform-integer rounding error.
- vpack first rounds its input to BF16, as the hardware path does.

**Pinned exponent.** s₀ is the reference rule floor(log2(amax/256)) (`ref_numerics.po2_scale_exp`,
pi0-quant `hw_float_ops.py:38-57`). It is computed on the weight GPTQ receives and **pinned**:
`weight_scale` returns `Scale(s₀, "per_tensor", dynamic=False)`, so `cast_with` reproduces the
GPTQ codes bit for bit.

- `transform_weight` checks this and raises otherwise.
- Pinning is needed because GPTQ can lower amax below 256 · 2^s₀. The dynamic rule would then pick
  s₀ − 1, and codes in [224, 256) · 2^s₀ would saturate at 448. `test_pinned_exponent_never_saturates`
  covers this.
- On Atlas, the weight is shipped as pre-packed codes plus that exponent. It needs a ruling only if
  deployment re-derives weight scales on chip from BF16 weights.

**Act-order.** Columns are processed by descending diag(H), then put back in the original order.
This is the "reorder" variant that arXiv 2306.00978 Sec. 5.1 benchmarks as GPTQ-R. With a per-tensor
scale it breaks no groups (NEW for this setting).

**Calibration (f):**

- Forward pre-hooks on every `nn.Linear` accumulate H = Σxxᵀ in float32 on the activation's device,
  plus the mean |x| per channel for AWQ.
- Linears that receive the very same input tensor object share one H: q/k/v, and gate/up. Sharing
  is detected by object identity through weak references.
- The data is the eval CLIs' calibration set, which never overlaps the evaluation data (README
  "Calibration rules"). TinyLlama uses WikiText-2 train with `--calib-n 16` by default.
- The same calibration set in a later mode of the same run is not run again; its record says
  `reused`. GPTQ results are cached per (layer, parameters, weight bytes).
- A Linear the calibration never reaches keeps the reference cast and is recorded as `rtn: not
  reached`.

**Calibration size.** Use at least as many calibration tokens as the widest Linear input. TinyLlama
`down_proj` has 5632 inputs, so the default 16 windows × 2048 tokens is enough. With fewer tokens H
is singular, only the dampening makes it invertible, and GPTQ fits the calibration set. The record
lists such layers in `calibration.result.rank_deficient_hessians`.

CPU smoke of `src/eval.py --protocol mxq --modes bf16 ref_linear` (bf16 compute):

| Calibration | Test windows | `--algo none` ref_linear | `gptq_awq` ref_linear |
|---|---|---|---|
| 16 × 2048 tokens | window 164 (1 × 2048) | +1.25% | +1.18% |
| 4 × 512 tokens (rank-deficient) | 2 × 512 | +2.18% | +2.96% |

**One-pass Hessians (NEW).** The paper quantizes layer by layer on the outputs of the
already-quantized layers. Here every H comes from one pass of the clean model, which is cheaper. A
sequential variant is future work if the GPU runs show a gap.

## AWQ

AWQ comes from Lin et al., *AWQ: Activation-aware Weight Quantization*, arXiv 2306.00978.

**Search.** Per input channel, s = s_X^α, where s_X is the mean |x| (Sec. 3.2, Eq. 5). α is searched
on `grid` points in [0, 1) (Sec. 5.1: 20). s is normalized by sqrt(max · min), as torchao 0.18 does
in `prototype/awq/core.py:71-74`. The search minimizes Eq. 4, L(s) = ||Q(W·diag(s)) diag(s)⁻¹X − WX||:

- Q is the reference cast, with its exponent re-derived on W·diag(s) as rounded to the parameter
  dtype.
- The loss is evaluated exactly from the calibration Hessian as tr(ΔW D H D ΔWᵀ), with
  D = diag(1/s) (NEW: no activations are stored).
- AWQ's weight-clipping search is not used: with power-of-two scales, clipping is a choice of
  exponent, which is another track's subject.

**Folds.** Sec. 3.2 says s⁻¹ "can usually be fused into the previous operator". The fold map for
Llama and Gemma decoder layers is NEW:

| Producer (gets 1/s) | Consumers (columns × s) |
|---|---|
| `input_layernorm` γ | `self_attn.{q,k,v}_proj` |
| `post_attention_layernorm` γ (`pre_feedforward_layernorm` on Gemma 2/3) | `mlp.{gate,up}_proj` |
| `mlp.up_proj` rows (and bias) | `mlp.down_proj` |

**Verification and refusal.** Every fold is checked on the calibration data before it is used. A
group that fails is left unscaled, and the `prepare` record (`awq_unscaled`) gives the reason. The
checks:

- The norm's output tensor must reach exactly the expected Linears. For example, a SmolVLA
  cross-attention `k_proj` fed from the VLM fails this check.
- The folded norm must reproduce out / s within 2^-6 relative (NEW tolerance: BF16 rounding of γ).
  Both γ·n(x) and Gemma's (1 + γ)·n(x) forms are tried.
- The MLP must satisfy `down_in == act_fn(gate_out) · up_out`.

**`o_proj` is always unscaled.** Its producer is the attention. Under GQA, `v_proj` has fewer output
rows than `o_proj` has inputs; a fold would need one scale tied across each KV group, which is not
implemented.

**Model coverage:**

- TinyLlama: fully covered and tested.
- SmolVLA, DiT and Pi0: run through the same verified map. Groups that fail are refused and recorded,
  not folded. These models have not yet been run with AWQ.
- GPTQ needs no folds and applies to every Linear of all four models.

**AWQ vs SmoothQuant.** AWQ searches a weight-quantization loss over s_X^α. SmoothQuant uses a
closed-form max|X|^α / max|W|^(1−α) to move activation outliers into the weights. This track keeps
AWQ in its search form.

**Expected gain.** The paper's argument (Sec. 3.1, Eq. 3) needs a step Δ that does not change while
w·s grows. The E4M3 step grows with |w|, so the gain largely cancels, except at the flush and
saturation edges and through the shared exponent. The screen shows this: flat on 152 of 154 TinyLlama
layers. The two `down_proj` exceptions come from a smaller activation range, which is SmoothQuant's
mechanism.

## Under a stack: a chosen weight grid (NEW)

Stacked after an algorithm that chooses the weight scale (for example `granularity:w=channel`),
`gptq_awq` quantizes onto that algorithm's grid instead of the per-tensor reference exponent. A
per-row exponent `[out, 1]` quantizes each column per row (GPTQ's own per-row quantizer), the
exponent is pinned as above with the chooser's granularity descriptor, and AWQ's search evaluates
Q with the same rule. A per-block grid raises `NotImplementedError`. Alone, nothing changes. See
[stacking.md](stacking.md).

## torchao 0.18 prototypes (not usable for Atlas FP8)

- **GPTQ.** `torchao/prototype/gptq/api.py` accepts only `Int4WeightOnlyConfig`,
  `Int8WeightOnlyConfig` (per-row) or NVFP4 as `base_config` (`:88-91`, `:361-378`). It asserts
  `device.type == "cuda"` (`:390`). There is no float8 path, and its quantizer is not the vpack cast.
- **AWQ.** `torchao/prototype/awq/api.py:104-111` requires the quantized weight to implement
  `SupportsActivationPreScaling`. In 0.18 only the Int4/Int8 tensors do; Float8Tensor does not.
- Both are cited only for their defaults: `percdamp=0.01` (`gptq/api.py:93`) and the AWQ grid
  normalization (`awq/core.py:71-74`).

## Screening (CPU, TinyLlama-1.1B, small samples; not final)

**Setup:**

- Calibration: 16 WikiText-2 train windows × 2048 tokens.
- Perplexity: 4 mxq test windows {164, 148, 147, 135}, seqlen 2048, `ref_linear`, fp32 compute.
- Layer-wise error: rel RMSE on 1 test window, geomean over 154 Linears.
- Produced by the phase-1 screen scripts, not by this module.

| method | Weight-only rel RMSE | W8A8 rel RMSE | PPL | vs bf16 |
|---|---|---|---|---|
| bf16 | – | – | 8.0575 | – |
| reference (RTN) | .0192 | .0274 | 8.1421 | +1.05% |
| GPTQ, natural order | .0140 | .0242 | 8.1340 | +0.95% |
| `gptq` (act-order) | .0131 | .0239 | 8.1089 | +0.64% |
| `awq` | .0188 | .0270 | 8.1374 | +0.99% |
| `awq_gptq` | .0130 | .0238 | 8.1043 | +0.58% |

On every one of the 4 windows, `gptq` and `awq_gptq` beat RTN. Their ranking against each other
needs the 165-window GPU run.

**Offline cost:**

- Hessian capture: one forward pass over the calibration set.
- GPTQ: about 2–11 s per TinyLlama matrix on a loaded CPU (seconds in total on a GPU).
- AWQ: `grid` Hessian-form loss evaluations per group.

## Records

The `algo` entry adds:

- `weights`: a count per outcome (`gptq`, `gptq+awq`, `awq`, `rtn`, Conv2d left on the reference
  cast).
- `prepare.awq_folded` and `prepare.awq_unscaled`, each with α or the reason.
- `calibration.result`: Hessian counts, shared inputs, verified and refused fold groups.
- `granularity.weight`: `linear:per_tensor` for every layer.
