# smoothquant: folded outlier migration and a folded power-of-two gain

`--algo smoothquant[:k=v,...]` (src/algos/smoothquant.py) applies per-channel smoothing and a
per-group power-of-two gain. Both are folded offline into the ops that produce each linear's input.
Neither adds run-time work.

The gain is what rescues the collaborator's fixed 2^0 activation scale. In the CPU screen,
smoothing alone did not.

## Sources

- **SmoothQuant** (Xiao et al., arXiv 2211.10438v7):
  - Eq. (3), Sec. 4: Y = (X diag(s)^-1)(diag(s) W).
  - Eq. (4), Sec. 4: s_j = max|X_j|^α / max|W_j|^(1−α).
  - Sec. 4: s is fused into the previous op.
  - Sec. 5.1: α = 0.5 is the default; the paper calibrates on 512 Pile sentences.
  - Tables 6–7: Llama uses α = 0.8–0.85.
- **Weight-scale rule** for the gain exponent: ceil(log2(amax / 448)), from atlas_quant.py:172-179.
- **Reference scale rule** for `static=true`: floor(log2(amax / 256)) (ref_numerics.po2_scale_exp,
  pi0-quant hw_float_ops.py:38-57).

## NEW decisions

| Decision | Reason |
|---|---|
| Smooth the inputs of down_proj (fold into the up_proj rows) and o_proj (fold into the v_proj rows), as well as the paper's QKV and FC1 | Under the fixed scale, the worst Atlas layers are early mlp.down_proj and self_attn.o_proj (BRIEF baselines). Both inputs are linear in the producer: act(gate)·up, and softmax·V. |
| `po2=true` rounds s to 2^round(log2 s) | The fold only changes exponents, so it is exact in BF16. This matches Atlas's E8M0 scales. `tests/test_algo_smoothquant.py::test_po2_fold_exact_in_bf16` checks that the BF16 forward is bit-identical. |
| `gain=true` folds g = 2^-ceil(log2(amax_post/448)) into each group's producer | Under a fixed 2^0 activation scale, the E4M3 normal range is [2^-6, 448]. The early layers' inputs sit almost entirely below 2^-6: TinyLlama layer-0 down_proj input has amax 0.57, and 52% of its channels never exceed 2^-6. Smoothing balances channels against each other but leaves the overall magnitude arbitrary; the gain sets the magnitude. |
| The gain is recorded as a folded transform (`algo.prepare.gain`), not as an activation `Scale` | It must stay distinct from a static-scale algorithm, so the two can be compared or merged. |
| s is tied across the query heads that share a kv head (GQA/MQA) | The o_proj input channel h·hd+d reads V channel kv(h)·hd+d (HF repeat_kv). |
| The 1e-5 floor on max\|X_j\| and max\|W_j\| | Keeps dead channels finite. |
| Gemma's (1+w) RMSNorm is folded as w' = (1+w)/f − 1 | This is not exact in BF16 even for po2 f. A BF16 weight is promoted to FP32 (`gemma_norm=fp32`) or refused (`gemma_norm=refuse`). The remaining FP32 error is recorded as `norm_fold_max_rel_err`. openpi already keeps these norms in FP32 (gemma_pytorch.py:71-82). |

## Fit to Atlas

There is no run-time cost: every fold changes BF16/FP32 parameters only.

With `gain=true`, the collaborator's profile (fixed 2^0 activation scale, no VPU amax) effectively
gets a calibrated static scale. This bears on open question Q2 (dynamic rmax scales, in
docs/reference_profile.md): wherever every linear input has a foldable producer, dynamic rmax is
not needed.

No hardware ruling is needed.

## Structures (folding map)

| Structure | qkv | fc1 | down | o |
|---|---|---|---|---|
| Standalone HF decoder stack (TinyLlama, any `*.layers` of Llama/Gemma decoder layers) | input_layernorm → q,k,v columns | post_attention_layernorm → gate,up columns | up_proj rows → down_proj columns | v_proj rows → o_proj columns; GQA tie |
| SmolVLA `SmolVLMWithExpertModel` (smolvlm_with_expert.py) | VLM: q,k,v. Expert cross layers: q only (k/v read the VLM's K/V) | both | both | Joint layers (:262-264): V tied across VLM and expert. Cross layers: VLM V is folded and the expert's v_proj columns undo it (:365-369); expert V is folded on its own |
| Pi0 `PaliGemmaWithExpertModel` (openpi gemma_pytorch.py:157-223) | both | both | both, GeGLU (modeling_gemma.py:125) | V tied across PaliGemma and the expert in every layer; MQA (1 kv head) ties all heads |

**Not implemented, and refused:**
- SigLIP/CLIP encoder layers (LayerNorm with bias; fc2 after GELU cannot be folded).
- DiT blocks (adaLN modulation).
- Adaptive RMSNorm (pi05).
- Any norm type not listed in the code.

For these, `prepare` raises `NotImplementedError` naming the block types. Pass `partial=true` to fold
only the supported decoder layers; the record then lists every unfolded block under
`not_folded_blocks`. SmolVLA and Pi0 currently need `partial=true`, because their vision towers are
SigLIP.

Planned folds (phase-1 design, handoff smoothquant.md):
- **SigLIP/CLIP:** γ and β divided by s; v rows and v bias.
- **DiT:** adaLN shift rows divided by s; scale rows W/s with b' = (1+b)/s − 1.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `alpha` | `0.5` | Migration strength. `off` disables smoothing, leaving the gain only. |
| `po2` | `true` | Power-of-two s. |
| `gain` | `true` | Fold the po2 gain. |
| `gain_target` | `448` | |
| `static` | `false` | Static per-tensor activation exponent on the reference steps (`act_scale`, recorded as `static_per_tensor(smoothquant)`). |
| `sites` | `qkv+fc1+down+o` | |
| `gemma_norm` | `fp32` | `fp32` or `refuse`. |
| `partial` | `false` | |

Calibration statistic: the per-channel BF16 amax of every nn.Linear input, gathered on the
unquantized model from the calibration source in docs/algos/README.md.

Ladder steps:
- `prepare` (the folds and the gain) applies on every quantized step, including collab_v0,
  collab_v0_hw and torchao.
- `static` acts only on ref_linear … chip_rope_vpu.
- On a dynamic per-tensor po2 step, the gain shifts the dynamic exponent and leaves the FP8 codes
  unchanged, apart from range effects in the producer.

## Results

**CPU screening on TinyLlama** (handoff smoothquant.md): 4 test windows × 2048 tokens, calibration on
32 × 512 WikiText-2 train windows. Rel-ppl change vs clean; screening only, not final:

| Smoothing | collab_v0_hw (fixed 2^0) | + folded gain | ref_linear |
|---|---|---|---|
| none | +96.3% | +0.95% | +0.67% |
| α=0.5, po2 | +280% | +0.69% | +0.62% |
| α=0.85, po2 | +7.5% | +1.03% | +1.14% |

Under dynamic or static scales, smoothing alone stays within screening noise. α = 1 fails (+24%): the
layer-7 down_proj input has a massive-activation channel, and migrating it fully pushes those weight
columns out of E4M3 range.

## GPU runs wanted

- **Models:** TinyLlama (165 windows), SmolVLA, DiT, Pi0.
- **Ladder steps:** collab_v0_hw, ref_linear, ref_attn, reference_hw, chip.
- **Algorithm settings:** `alpha=off` (gain only), `alpha=0.5`, `alpha=0.85`, each with `po2=true`, and `static=true` on the reference steps.
