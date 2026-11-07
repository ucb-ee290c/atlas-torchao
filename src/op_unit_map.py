"""Which Atlas unit executes each aten op, in which number format, and on whose authority.

Units (examples/atlas/.../isa_include/atlas_isa_green_card.md, "GC" below):

  MXU     FP8 E4M3 operands, BF16 accumulation (vmatmul / vmatmul.acc, GC 3: 77/10-77/13). Emulated
          by the reference_hw accumulation model (accum_backends: ipt | sa).
  VPU     16 BF16 lanes per register, register pairs of 32 (GC 1, GC 3: opcode 57; VectorParams
          num_lanes=16). Emulated by chip_ops.ChipVPU, bit-exact to the reference VPU model.
  scalar  RV32I integer ALU only: add/sub/and/or/xor/shifts/slt (ScalarALU.scala:26-36,
          Instructions.scala:24-44). No multiply/divide (no M extension) and no floating point.
          Integer ops are exact, so they run natively in 32-bit-compatible integer arithmetic.
          Arithmetic whose tensor operands are all one-element float32 (schedule coefficients such
          as the DDPM alphas/betas) is a software float32 routine here, run natively (NEW).
  data    data movement with no arithmetic: DMA, vload/vstore, vmov, vtrpose.xlu, scalar address
          computation (GC 3: 07/0-1, 57/40, 6B/00, 7B). Bit-preserving, run natively; a cast to BF16 is
          the RNE rounding the reference adapter applies on load (vector_rtl_forward.py:66-73).
  host    no Atlas instruction (or kept off-chip on request): runs in torch on the host, listed with
          a reason in every coverage report.
  eval    evaluation-only ops (the perplexity loss): not part of inference; listed separately.

Status of each mapping:
  REF  the reference author's code decides it (smolVLA-quant quant_vector.py TARGET_OPS and
       vpu_decomp.py; pi0-quant vpu_decomp.py where it fixes broadcasting).
  ISA  not in the reference vector path, but a single instruction in the green card / RTL does it.
  NEW  a decomposition or treatment with no reference source; the reason is given.
  HOST no instruction; host fallback with the reason given.

`ATEN` maps an aten overload-packet name to an Entry. Integer/bool tensors are routed by
`INT_SCALAR` / `DATA` first (chip_coverage.classify), so the float entries below describe float data.
`MODEL_OPS` is the model-level view (which model operation becomes which aten ops on which unit),
from the op census of TinyLlama, SmolVLA and Multi-Task DiT (tests/test_op_unit_map.py keeps the two
consistent).
"""

from __future__ import annotations

from dataclasses import dataclass, field

QV = "smolVLA-quant smolvla_inout/quant_vector.py"
VD = "smolVLA-quant smolvla_inout/vpu_decomp.py"
VD_PI0 = "pi0-quant pi0_inout/vpu_decomp.py"
QV_PI0 = "pi0-quant pi0_inout/quant_vector.py"
OP = "openpi@215abfb models_pytorch"          # Physical-Intelligence/openpi, the Pi0 model source
GC = "atlas_isa_green_card.md 3"
VRF = "funct_models_vector/vector_rtl_forward.py"


@dataclass(frozen=True)
class Entry:
    unit: str
    instrs: str
    fmt: str
    source: str
    status: str
    reason: str = ""


BF16 = "BF16 lanes (inputs RNE to BF16, result BF16)"
FP8 = "FP8 E4M3 operands (per-tensor po2 scale), BF16 accumulate"


def _v(instrs, source, status="REF", reason=""):
    return Entry("VPU", instrs, BF16, source, status, reason)


ATEN: dict[str, Entry] = {
    # ---- elementwise arithmetic (TARGET_OPS quant_vector.py:150-176; table vpu_decomp.py:229-277)
    "add": _v("vadd.bf16 (alpha != 1: vmul.bf16 first)", f"{QV}:152-154; {VD}:232-239; {VD_PI0}:218-233",
              reason="alpha is applied with vmul (NEW: the reference drops it)"),
    "add_": _v("vadd.bf16", f"{QV}:154; {VD}:234-235", reason="written back in place (the reference returns a new tensor)"),
    "sub": _v("vsub.bf16", f"{QV}:155-156; {VD}:241-246"),
    "sub_": _v("vsub.bf16", f"{VD}:241-242", "NEW", "in-place form of sub"),
    "rsub": _v("vsub.bf16 (operands swapped)", f"{VD}:241-246", "NEW", "c - x is sub with the operands swapped"),
    "mul": _v("vmul.bf16", f"{QV}:158-159; {VD}:247-252"),
    "mul_": _v("vmul.bf16", f"{QV}:160; {VD}:249-250", reason="written back in place"),
    "div": _v("vrecip.bf16, vmul.bf16", f"{QV}:164; {VD}:253-255"),
    "div_": _v("vrecip.bf16, vmul.bf16", f"{VD}:253-255", "NEW", "in-place form of div"),
    "reciprocal": _v("vrecip.bf16", f"{QV}:165; {VD}:259-260"),
    "neg": _v("vsub.bf16 (0 - x)", f"{QV}:176; {VD}:263-264"),
    "sqrt": _v("vsqrt.bf16", f"{QV}:167; {VD}:261-262"),
    "rsqrt": _v("vsqrt.bf16, vrecip.bf16", f"{QV}:168; {VD}:79-81"),
    "pow": _v("x^2 vsquare.bf16, x^3 vcube.bf16 (REF); x^0.5 vsqrt, x^-1 vrecip, x^-0.5 vsqrt+vrecip, "
              "x^1 none, c^x vexp2(x*log2 c) (NEW)", f"{QV}:162; {VD}:199-208",
              reason="other exponents: host (the reference falls back to torch, vpu_decomp.py:205-208)"),
    "exp": _v("vexp.bf16", f"{QV}:174; {VD}:271-272"),
    "exp2": _v("vexp2.bf16", f"{QV}:175; {VD}:273-274"),
    "log2": _v("vlog2.bf16", f"{QV}:173; {VD}:275-276"),
    "log": _v("vlog2.bf16, vmul.bf16 (x ln2)", f"{GC} vlog2.bf16 57/4C", "NEW", "natural log = log2 x ln 2"),
    "sin": _v("vsin.bf16", f"{QV}:170; {VD}:265-266"),
    "cos": _v("vcos.bf16", f"{QV}:171; {VD}:267-268"),
    "tanh": _v("vtanh.bf16", f"{QV}:172; {VD}:269-270"),
    "relu": _v("vrelu.bf16", f"{GC} 57/48; {VRF}:125-126", "ISA"),
    "abs": _v("vsub.bf16, vmaximum.bf16 (max(x, 0 - x))", f"{GC} 57/06", "NEW", "no vabs instruction"),
    "maximum": _v("vmaximum.bf16", f"{GC} 57/06; {VRF}:115-116", "ISA"),
    "minimum": _v("vminimum.bf16", f"{GC} 57/04; {VRF}:118-119", "ISA"),
    "clamp": _v("vmaximum.bf16, vminimum.bf16", f"{GC} 57/04, 57/06", "ISA"),
    "clamp_min": _v("vmaximum.bf16", f"{GC} 57/06", "ISA"),
    "clamp_max": _v("vminimum.bf16", f"{GC} 57/04", "ISA"),
    "clamp_": _v("vmaximum.bf16, vminimum.bf16", f"{GC} 57/04, 57/06", "ISA"),
    # ---- activations / fused composites (quant_vector.py:183-187, 375-396; vpu_decomp.py:79-192)
    "silu": _v("vsub, vmaximum (exp floor), vexp, vadd, vrecip, vmul", f"{QV}:187,390-392; {VD}:181-192",
               "NEW", "reference sequence plus vmaximum(x, -128) before vexp: the vexp model wraps "
               "below -336 (exp(-1000) = 2.6e10); analogue: ExpLane.scala:47 overflow pre-check"),
    "silu_": _v("vsub, vmaximum, vexp, vadd, vrecip, vmul", f"{VD}:181-192", "NEW", "in-place form of silu"),
    "sigmoid": _v("vsub, vmaximum (exp floor), vexp, vadd, vrecip", f"{VD}:187-191", "NEW",
                  "the sigmoid part of the reference SiLU sequence (vpu_decomp.py:188-191), with the exp floor"),
    "gelu": _v("vmul x3, vmul, vadd, vmul, vtanh, vadd, vmul, vmul", f"{QV}:186,386-388; {VD}:161-178",
               reason="tanh form for every `approximate` (the reference ignores it, quant_vector.py:386-388)"),
    "_softmax": _v("vredmax.row (staged), vsub, vmaximum (exp floor), vexp, vredsum.row (staged), vrecip, vmul",
                   f"{QV}:185,381-384; {VD}:132-158", "NEW",
                   "reference sequence plus vmaximum(x - max, -128) before vexp: the vexp model returns "
                   "exp(-3.3e38) = 1, so an additive -inf/min mask would not mask; analogue: "
                   "ExpLane.scala:47 overflow pre-check"),
    "_safe_softmax": _v("as _softmax", f"{VD}:132-158", "NEW", "same op without fully-masked-row NaNs in torch"),
    "native_layer_norm": _v("vredsum.row, vmul, vsub, vmul, vredsum.row, vmul, vadd, vsqrt, vrecip, vmul, vmul, vadd",
                            f"{QV}:184,376-379; {VD}:84-129"),
    "_fused_rms_norm": _v("vsquare, vredsum.row, vmul, vadd, vsqrt, vrecip, vmul, vmul",
                          f"{QV}:162,181,168 (Llama RMSNorm primitives)", "NEW",
                          "the fused kernel decomposed into the primitives the reference intercepts for LlamaRMSNorm"),
    # ---- reductions (staged 32-wide row reductions, vpu_decomp.py:33-72; quant_vector.py:220-235, 398-427)
    "sum": _v("vredsum.row (staged)", f"{QV}:179-180,418-427; {VD}:33-50",
              reason="several dims reduced one after another (NEW: the reference takes only the first)"),
    "mean": _v("vredsum.row (staged), vmul (1/N)", f"{QV}:181,399-407; {VD}:33-50"),
    "amax": _v("vredmax.row (staged)", f"{QV}:178,409-416; {VD}:53-72"),
    "amin": _v("vredmin.row (staged)", f"{GC} 57/24; {VD}:53-72", "ISA"),
    "max": _v("vredmax.row (staged) / vmaximum.bf16", f"{QV}:409-416; {GC} 57/06", "ISA",
              "max(x) and max(x, y); max.dim (with indices) is host"),
    "min": _v("vredmin.row (staged) / vminimum.bf16", f"{GC} 57/04, 57/24", "ISA",
              "min(x) and min(x, y); min.dim (with indices) is host"),
    "cumsum": _v("vadd.bf16 per step (sequential scan)", f"{QV}:182", "NEW",
                 "in TARGET_OPS but absent from the reference dispatch table (falls back to torch); scanned with vadd"),
    # ---- masking (attention masks; Atlas has no compare or select instruction)
    "where": _v("vadd.bf16 of a host-precomputed additive mask", f"{GC} 57/00", "NEW",
                "where(mask, x, c) with constant c becomes x + (mask ? 0 : bf16(c)); a select of two "
                "computed tensors is host"),
    "masked_fill": _v("vadd.bf16 of a host-precomputed additive mask", f"{GC} 57/00", "NEW",
                      "as where; the mask is input-independent"),
    "masked_fill_": _v("vadd.bf16 of a host-precomputed additive mask", f"{GC} 57/00", "NEW", "as where"),
    # ---- matmuls
    "mm": Entry("MXU", "vpack.bf16.fp8 x2, vmatmul, vmatpop.bf16, vmul.bf16 (x 2^(sa+sb))", FP8,
                "pi0-quant quant_linear.py:145-172 (per-tensor scales)", "NEW",
                "a matmul outside nn.Linear (not replaced by reference_hw) cast like a linear layer"),
    "addmm": Entry("MXU", "as mm, then vadd.bf16 (bias)", FP8, "pi0-quant quant_linear.py:145-172", "NEW",
                   "as mm; bias added on the VPU after the rescale (quant_linear.py:171-172)"),
    "bmm": Entry("MXU", "per-matrix vpack.bf16.fp8, vmatmul, vmatpop.bf16, vmul.bf16 (x 2^(sa+sb))", FP8,
                 "smolVLA-quant model_patcher.py:667-699 (per-(batch, head) scales)", "NEW",
                 "attention score matmuls taken at the aten level when the model's attention function is not "
                 "patched; K = 1 (an outer product) is a vmul.bf16"),
    "baddbmm": Entry("MXU", "as bmm, then vadd.bf16", FP8, "smolVLA-quant model_patcher.py:667-699", "NEW", "as bmm"),
    "linear": Entry("MXU", "vpack.bf16.fp8, vmatmul(.acc), vmatpop.bf16", FP8,
                    "reference_hw RefQuantLinear (pi0-quant quant_linear.py:145-172)", "REF"),
    "conv2d": Entry("MXU", "im2col, vpack.bf16.fp8, vmatmul, vmatpop.bf16", FP8,
                    "reference_hw RefQuantConv2d (smolVLA-quant model_patcher.py:345-375)", "REF"),
    "attention": Entry("MXU", "per-(b,h) vpack.bf16.fp8, vmatmul, vmatpop.bf16", FP8,
                       "reference_hw ref_attention (smolVLA-quant model_patcher.py:624-723, 798-866)", "REF"),
    "fp8_operand_cast": Entry("VPU", "vredmax.row, vredmin.row (amax), vpack.bf16.fp8", "BF16 -> FP8 E4M3",
                              "pi0-quant hw_float_ops.py:9-12,38-103 (bit-equal ref_numerics.quantize)", "REF"),
    "convolution": Entry("MXU", "im2col (data), vpack, vmatmul, vmul (rescale), vadd (bias)", FP8,
                         "smolVLA-quant model_patcher.py:345-375 (QuantConv2d)", "REF",
                         "groups != 1 is host (model_patcher.py:339-343)"),
}

# Integer / bool ops on the scalar core (exact). Multiply/divide have no RV32I instruction: a software
# routine of shifts and adds does them exactly (NEW).
INT_SCALAR = {
    "add", "add_", "sub", "sub_", "rsub", "neg", "eq", "ne", "lt", "le", "gt", "ge", "bitwise_and",
    "bitwise_or", "bitwise_xor", "bitwise_not", "logical_and", "logical_or", "logical_not", "logical_xor",
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "bitwise_left_shift", "bitwise_right_shift",
    "cumsum", "sum", "all", "any", "min", "max", "amax", "amin", "maximum", "minimum", "clamp", "clamp_min",
    "clamp_max", "where", "masked_fill", "masked_fill_", "abs", "sign", "argmax", "argmin",
    "mul", "mul_", "div", "floor_divide", "remainder", "fmod", "prod",
}
INT_SOFTWARE = {"mul", "mul_", "div", "floor_divide", "remainder", "fmod", "prod"}

DATA = {
    "view", "_unsafe_view", "reshape", "_reshape_alias", "view_as", "expand", "expand_as", "permute", "transpose",
    "t", "unsqueeze", "squeeze", "select", "slice", "narrow", "as_strided", "contiguous", "clone", "flatten",
    "unflatten", "movedim", "alias", "detach", "lift_fresh", "lift_fresh_copy", "_to_copy", "to", "copy_",
    "copy", "cat", "stack", "split", "split_with_sizes", "chunk", "unbind", "embedding", "index",
    "index_select", "gather", "scatter", "index_put", "index_put_", "slice_scatter", "select_scatter",
    "repeat", "repeat_interleave", "flip", "roll", "constant_pad_nd", "pad", "arange", "linspace", "full",
    "full_like", "zeros", "zeros_like", "ones", "ones_like", "empty", "empty_like", "empty_strided",
    "new_empty", "new_zeros", "new_ones", "new_full", "new_empty_strided", "fill_", "fill", "zero_",
    "scalar_tensor", "_local_scalar_dense", "resize_", "set_", "squeeze_", "unsqueeze_", "t_",
    "transpose_", "masked_select", "nonzero", "tril", "triu", "diagonal", "unfold", "im2col", "pixel_shuffle",
    "pixel_unshuffle", "channel_shuffle",
}
DATA_NOTES = {
    "_to_copy": "dtype cast: BF16 RNE on load (vector_rtl_forward.py:66-73); exact widening otherwise",
    "embedding": "row gather by DMA (GC 3: dma.load), addresses on the scalar core",
    "index": "gather by DMA", "index_select": "gather by DMA", "gather": "gather by DMA",
    "index_put_": "scatter by DMA / vstore", "copy_": "vstore / vmov", "cat": "DMA into one buffer",
    "transpose": "vtrpose.xlu (GC 3: 6B/00) or strided DMA", "permute": "vtrpose.xlu / strided DMA",
    "full": "vli.all (GC 3: 5F/0) constant", "_local_scalar_dense": "host readback of a scalar (control flow)",
    "tril": "constant mask", "triu": "constant mask",
}

EVAL_ONLY = {"_log_softmax", "nll_loss_forward", "nll_loss2d_forward", "nll_loss", "cross_entropy_loss",
             "mse_loss", "l1_loss", "smooth_l1_loss"}

# Float ops with no instruction. Reasons for the ones the census meets; anything else gets a generic one.
HOST_REASONS = {
    "_scaled_dot_product_flash_attention_for_cpu": "fused SDPA kernel; the chip profile switches attention to "
                                                   "eager so its parts can be emulated",
    "_scaled_dot_product_efficient_attention": "fused SDPA kernel (see above)",
    "_scaled_dot_product_flash_attention": "fused SDPA kernel (see above)",
    "_scaled_dot_product_cudnn_attention": "fused SDPA kernel (see above)",
    "upsample_bilinear2d": "image resize (interpolation): no gather-lerp instruction; input preprocessing",
    "_upsample_bilinear2d_aa": "image resize with antialiasing: input preprocessing",
    "upsample_bicubic2d": "image resize: input preprocessing",
    "bucketize": "patch-position bucketing (compares): input-independent, precomputed on the host",
    "randn": "random noise is drawn on the host and DMA'd in",
    "normal": "random noise is drawn on the host", "normal_": "random noise is drawn on the host",
    "rand": "random numbers are drawn on the host", "randint": "random numbers are drawn on the host",
    "randn_like": "random noise is drawn on the host", "uniform_": "random numbers are drawn on the host",
    "floor": "no rounding-to-integer instruction", "ceil": "no rounding-to-integer instruction",
    "round": "no rounding-to-integer instruction", "trunc": "no rounding-to-integer instruction",
    "erf": "no erf instruction", "erfinv": "no erf instruction",
    "eq": "float compare: no FP compare instruction", "ne": "float compare: no FP compare instruction",
    "lt": "float compare: no FP compare instruction", "le": "float compare: no FP compare instruction",
    "gt": "float compare: no FP compare instruction", "ge": "float compare: no FP compare instruction",
    "isnan": "float classify: no instruction", "isinf": "float classify: no instruction",
    "sort": "no sort instruction", "topk": "no sort instruction", "argmax": "no arg-reduction instruction",
    "argmin": "no arg-reduction instruction", "var": "not decomposed (no instance in the models)",
    "var_mean": "not decomposed (no instance in the models)",
    "native_group_norm": "not decomposed (no instance in the models)",
    "reciprocal_int": "reciprocal of an integer shape constant (computed once on the host)",
    "float64": "float64 scalar arithmetic on schedule constants",
}

# Host / data treatments that are NEW decisions (no ATEN entry of their own). The router passes the
# reason string; Coverage lists the row under `new_decisions` with this choice and analogue.
LOOP_SCALAR = ("loop-carried 0-d float scalar updated in place (flow-matching `time += dt`): kept on the "
               "host with the loop comparison it feeds (NEW)")
MASK_BUILD = ("where(mask, c1, c2) with two constant branches: the additive attention mask itself, "
              "built once from the boolean mask (NEW)")
MTDIT = "multitask-dit-quant mtdit_inout"
ONE_ELEMENT_FP32 = "one-element fp32 scheduler arithmetic on the scalar core (NEW)"
NEW_TREATMENTS = {
    ONE_ELEMENT_FP32: {
        "choice": "scalar core, software float32 (run natively), not the BF16 VPU",
        "analogue": f"{MTDIT}/chip_full.py:58-65 and op_emulation.py:177-179,209-217 (one-element fp32 "
                    "operands stay on the scalar core in fp32)",
        "reason": "diffusers' DDPM alphas_cumprod is float32; in BF16 0.9994 and 0.9983 both round to 1.0, so "
                  "at t = 1 beta_prod_t = current_beta_t = 0 and div becomes vmul(0, vrcp(0)) = 0 * inf = NaN, "
                  "which spreads to the sample. Python-scalar operands may join; any multi-element tensor "
                  "operand keeps the op on the VPU"},
    LOOP_SCALAR: {"choice": "host (float32), not vadd",
                  "analogue": f"{QV_PI0}:136-171 (TARGET_OPS has no add_, so the reference keeps it in "
                              f"float32); {OP}/pi0_pytorch.py:407,419",
                  "reason": "the loop test `time >= -dt/2` is a float compare (host) and the scalar core has "
                            "no FP; a BF16 vadd would truncate the timestep every step (1.0, 0.898, 0.797, ... "
                            "instead of 1.0, 0.9, 0.8, ...) and shift every time embedding"},
    MASK_BUILD: {"choice": "data (precomputed constant mask, DMA'd in)",
                 "analogue": "where/masked_fill decision (additive host-precomputed masks); "
                             f"{OP}/pi0_pytorch.py:160",
                 "reason": "a select between two constants is mask construction, not arithmetic on data"},
}
# Position- or schedule-only tables computed on the host under --chip-rope host (chip_profile hooks).
TABLE_HOST = {
    "create_sinusoidal_pos_embedding": {
        "choice": "host table (float64 as in the model), --chip-rope host only",
        "analogue": f"--chip-rope host for RoPE tables; {OP}/pi0_pytorch.py:25-42 (also lerobot SmolVLA)",
        "reason": "sinusoidal timestep table (schedule-only), precomputed on the host (--chip-rope host): its "
                  "angles reach 2 pi / 4e-3 = 1571 rad, beyond vsin/vcos's Q(9,12) range (~400 rad), and "
                  "BF16 angles have an ulp of 8 rad at 1024"},
}

# Model-level view: (model, operation) -> (aten ops seen, unit, format, citation, status / reason).
# The census that produced the "aten" column: tests/test_op_unit_map.py::test_model_ops_cover_census
# (TinyLlama seqlen 128; SmolVLA one frame; Multi-Task DiT one frame).
MODEL_OPS = [
    # TinyLlama (transformers 5.5.4 LlamaForCausalLM, eager attention)
    ("llama", "token embedding", "embedding", "data", "BF16 rows", "DMA gather", "ISA"),
    ("llama", "RMSNorm", "_to_copy, pow(2), mean, add(eps), rsqrt, mul, mul(weight)", "VPU", BF16,
     f"{QV}:150-188 via {VD} (pow->vsquare, mean->rsum+mul, rsqrt->sqrt+rcp)", "REF"),
    ("llama", "RoPE tables (inv_freq x positions, cos, sin)", "bmm(K=1), cat, cos, sin, mul", "VPU", BF16,
     f"{QV}:170-171 (sin/cos on the VPU as for SmolVLA); outer product as vmul (NEW); --chip-rope host "
     "precomputes the table", "REF/NEW"),
    ("llama", "RoPE apply (x cos + rotate_half(x) sin)", "slice, neg, cat, mul, add", "VPU", BF16, f"{VD}:229-264", "REF"),
    ("llama", "q/k/v/o, gate/up/down projections", "Linear", "MXU", FP8, "reference_hw RefQuantLinear", "REF"),
    ("llama", "attention scores Q.K^T, P.V", "bmm", "MXU", FP8,
     "per-(b,h) scales as smolVLA-quant model_patcher.py:667-699 (aten-level; ref_attention when merged)", "NEW"),
    ("llama", "attention scaling", "mul", "VPU", BF16, f"{QV}:158", "REF"),
    ("llama", "causal mask", "where(bool, 0, min) [mask build], add", "VPU", BF16,
     "mask built once from positions (data); vadd on the VPU", "NEW"),
    ("llama", "softmax", "_to_copy(fp32), _softmax", "VPU", BF16, f"{QV}:185; {VD}:132-158", "REF"),
    ("llama", "SiLU and SwiGLU gating", "silu, mul", "VPU", BF16, f"{QV}:187; {VD}:181-192", "REF"),
    ("llama", "residual adds", "add", "VPU", BF16, f"{QV}:152-154", "REF"),
    ("llama", "lm_head", "Linear (skipped by eval.py)", "host / MXU", "BF16 on host when --skip lm_head",
     "eval.py skip=['lm_head']: host fallback by request; MXU otherwise", "HOST"),
    ("llama", "perplexity loss", "_log_softmax, nll_loss", "eval", "float32", "evaluation only", "-"),
    # SmolVLA (lerobot 0.6.1)
    ("smolvla", "image resize / pad", "upsample_bilinear2d, constant_pad_nd", "host", "-",
     "input preprocessing (resize_with_pad)", "HOST"),
    ("smolvla", "SigLIP patch embedding (conv)", "convolution", "MXU", FP8,
     "smolVLA-quant model_patcher.py:345-375", "REF"),
    ("smolvla", "vision position ids", "bucketize, clamp, arange, index", "host / data", "-",
     "input-independent patch grid", "HOST"),
    ("smolvla", "vision LayerNorm", "native_layer_norm", "VPU", BF16, f"{QV}:184; {VD}:84-129", "REF"),
    ("smolvla", "vision GELU (tanh)", "gelu", "VPU", BF16, f"{QV}:186; {VD}:161-178", "REF"),
    ("smolvla", "vision attention", "bmm, mul, _softmax (eager)", "MXU + VPU", f"{FP8}; {BF16}",
     "smolVLA-quant model_patcher.py:798-866", "REF"),
    ("smolvla", "pixel shuffle / connector", "view, permute, reshape, Linear", "data + MXU", FP8,
     "reshape is data movement; projection is a Linear", "REF"),
    ("smolvla", "SmolLM / expert RMSNorm", "pow, mean, add, rsqrt, mul", "VPU", BF16, f"{QV}:150-188", "REF"),
    ("smolvla", "RoPE (apply_rope)", "arange, pow.Scalar, div, sin, cos, mul, sub, add, copy_", "VPU", BF16,
     f"{QV}:170-171 (sin/cos); timescale c^x as vexp2(x log2 c) (NEW)", "REF/NEW"),
    ("smolvla", "text/expert attention", "bmm, mul, where(mask), _softmax", "MXU + VPU", f"{FP8}; {BF16}",
     "smolVLA-quant model_patcher.py:624-723", "REF"),
    ("smolvla", "SiLU MLP", "silu, mul", "VPU", BF16, f"{QV}:187", "REF"),
    ("smolvla", "timestep sinusoidal embedding", "linspace, pow.Scalar(f64), reciprocal, mul, sin, cos", "VPU",
     BF16, f"{QV}:170-171; constants in float64 are rounded to BF16 like every VPU input", "REF/NEW"),
    ("smolvla", "action time MLP (Linear, SiLU)", "addmm(fp32), silu", "MXU + VPU", f"{FP8}; {BF16}",
     f"quant_linear.py:133-137 (fp32 layers cast to BF16); {QV}:187", "REF"),
    ("smolvla", "flow-matching Euler step x + dt v", "mul, add", "VPU", BF16, f"{QV}:152,158", "REF"),
    ("smolvla", "initial noise", "randn", "host", "float32", "RNG on the host", "HOST"),
    # Multi-Task DiT (lerobot 0.6.1 multi_task_dit; CLIP towers from transformers)
    ("dit", "CLIP patch embedding (conv)", "convolution", "MXU", FP8, "model_patcher.py:345-375", "REF"),
    ("dit", "CLIP LayerNorm", "native_layer_norm", "VPU", BF16, f"{QV}:184; {VD}:84-129", "REF"),
    ("dit", "CLIP QuickGELU x sigmoid(1.702 x)", "mul, sigmoid, mul", "VPU", BF16,
     f"{VD}:187-191 (sigmoid from the SiLU sequence)", "NEW"),
    ("dit", "CLIP attention", "bmm, mul, add(mask), _softmax", "MXU + VPU", f"{FP8}; {BF16}",
     "model_patcher.py:798-866 analogue", "NEW"),
    ("dit", "DiT adaLN modulation x (1 + scale) + shift, gate", "silu, Linear, add, mul", "VPU + MXU",
     f"{BF16}; {FP8}", f"{QV}:152-160,187", "REF"),
    ("dit", "DiT RoPE", "sin/cos tables, mul, add, cat", "VPU", BF16, f"{QV}:170-171", "REF"),
    ("dit", "timestep sinusoidal embedding", "arange, mul, exp, sin, cos, cat", "VPU", BF16, f"{QV}:170-174", "REF"),
    ("dit", "DiT MLP (GELU)", "gelu, Linear", "VPU + MXU", f"{BF16}; {FP8}", f"{QV}:186", "REF"),
    ("dit", "DDPM step arithmetic on the sample", "mul, sub, add, div, clamp, randn", "VPU (+ host RNG)", BF16,
     f"{QV}:150-176; clamp via vmaximum/vminimum (ISA)", "REF/ISA"),
    ("dit", "DDPM step coefficients (alphas_cumprod, betas: one-element fp32)", "rsub, pow, div, mul, clamp",
     "scalar", "float32", f"{MTDIT}/chip_full.py:58-65; in BF16 1 - alpha_prod_t cancels to 0 and 0 / 0 is NaN",
     "NEW"),
    # Pi0 (openpi PI0Pytorch, pi0=True; transformers 4.53.2 + openpi transformers_replace). Reference:
    # pi0-quant patch_vector_ops (quant_vector.py:136-171 TARGET_OPS, 360-433 dispatch); the vector
    # sequences are vpu_decomp.py (identical to smolVLA-quant's apart from broadcasting, 218-225).
    ("pi0", "image layout / resize", "permute (resize only for non-224 inputs: upsample_bilinear2d, clamp)",
     "data / host", "-", f"{OP}/preprocessing_pytorch.py:41-50; resize as SmolVLA input preprocessing", "HOST"),
    ("pi0", "SigLIP patch embedding (conv, float32 weights)", "convolution", "MXU", FP8,
     f"pi0-quant model_patcher.py:159,493 (QuantConv2d); fp32 cast through BF16 (quant_linear.py:136-137)", "REF"),
    ("pi0", "SigLIP position embedding", "embedding, add", "data + VPU", BF16,
     f"{OP}/transformers_replace/models/siglip/modeling_siglip.py:280; {QV_PI0}:138", "REF"),
    ("pi0", "SigLIP LayerNorm", "native_layer_norm", "VPU", BF16, f"{QV_PI0}:167,364-369; {VD_PI0}:84-129", "REF"),
    ("pi0", "SigLIP / Gemma MLP GELU (tanh)", "gelu(approximate='tanh'), mul (GeGLU gate x up)", "VPU", BF16,
     f"{QV_PI0}:169,377-379; {VD_PI0}:161-178; {QV_PI0}:143", "REF"),
    ("pi0", "SigLIP attention (SDPA in the model)", "reference routine: bmm (MXU), _softmax", "MXU + VPU",
     f"{FP8}; {BF16}", "pi0-quant model_patcher.py:738-823 (ref_attention variant pi0)", "REF"),
    ("pi0", "multimodal projector, q/k/v/o, gate/up/down, action head linears", "Linear", "MXU", FP8,
     "reference_hw RefQuantLinear (pi0-quant quant_linear.py:145-172)", "REF"),
    ("pi0", "token embedding x sqrt(width)", "embedding, mul", "data + VPU", BF16,
     f"{OP}/pi0_pytorch.py:217; {QV_PI0}:143 (the constant rounded to BF16 like every VPU input)", "REF"),
    ("pi0", "Gemma RMSNorm x rsqrt(mean(x^2) + eps) (1 + w)", "pow(2), mean, add, rsqrt, mul, add(1 + w), mul",
     "VPU", BF16, f"{OP}/transformers_replace/models/gemma/modeling_gemma.py:68-80; {QV_PI0}:146,165,152 via "
     f"{VD_PI0}:199-202 (vsquare), {QV_PI0}:390-399 (mean), {VD_PI0}:79-81 (rsqrt), 232-233 (1 + w as vadd)", "REF"),
    ("pi0", "RoPE tables (inv_freq x positions, cos, sin)", "bmm(K=1), cat, cos, sin, mul", "VPU / host", BF16,
     f"{QV_PI0}:154-155 (the reference puts sin/cos on the VPU); K=1 bmm as vmul (NEW); --chip-rope host "
     "precomputes them (Pi0 positions reach ~880 rad, beyond vsin/vcos's range)", "REF/NEW"),
    ("pi0", "RoPE apply (q cos + rotate_half(q) sin)", "slice, neg, cat, mul, add", "VPU", BF16,
     f"{OP}/transformers_replace/models/gemma/modeling_gemma.py:163-193; {VD_PI0}:232-260", "REF"),
    ("pi0", "prefix / block-causal masks", "cumsum, le, mul(bool), bitwise_and, where(bool, 0, -2.38e38)",
     "scalar / host / data", "-", f"{OP}/pi0_pytorch.py:78-80,160; float cumsum on the VPU, float `le` host "
     "(no FP compare), constant select = mask build (NEW)", "NEW"),
    ("pi0", "Gemma attention (eager, joint prefix + suffix)", "reference routine: bmm (MXU), add(mask), _softmax",
     "MXU + VPU", f"{FP8}; {BF16}", "pi0-quant model_patcher.py:911-1005; mask add vadd; softmax with the vexp "
     "floor (NEW, masked logits -2.38e38 -> exp = 0)", "REF/NEW"),
    ("pi0", "residual adds", "add", "VPU", BF16, f"{OP}/transformers_replace/models/gemma/modeling_gemma.py:226; "
     f"{QV_PI0}:138", "REF"),
    ("pi0", "state / action projections, action-time MLP (SiLU)", "Linear, cat, silu", "MXU + VPU",
     f"{FP8}; {BF16}", f"{OP}/pi0_pytorch.py:250,278-284; {QV_PI0}:170,381-383 (SiLU with the vexp floor, NEW)",
     "REF"),
    ("pi0", "sinusoidal time embedding", "linspace, pow.Scalar(f64), reciprocal, mul, sin, cos, cat", "VPU / host",
     BF16, f"{OP}/pi0_pytorch.py:25-42; {QV_PI0}:149,143,154-155 (float64 inputs rounded to BF16); c^x as "
     "vexp2 (NEW; the reference keeps pow.Scalar in float64); --chip-rope host precomputes the table (NEW)",
     "REF/NEW"),
    ("pi0", "flow-matching Euler step x + dt v", "mul, add", "VPU", BF16, f"{OP}/pi0_pytorch.py:418; {QV_PI0}:138,143",
     "REF"),
    ("pi0", "flow time update and loop test", "add_ (0-d), neg, div, ge, _local_scalar_dense", "host + scalar",
     "float32", f"{OP}/pi0_pytorch.py:402-419; -dt/2 in float32 on the scalar core (NEW one-element fp32 rule; "
     f"the reference puts it on the VPU, {QV_PI0}:148,160; the loop count is unchanged), time += dt and the "
     "compare on the host (NEW, as the reference: add_ is not in its TARGET_OPS)", "NEW"),
    ("pi0", "initial noise", "normal (randn)", "host", "float32", "RNG on the host", "HOST"),
]
