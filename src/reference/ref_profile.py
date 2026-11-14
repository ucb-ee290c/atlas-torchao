"""The reference author's FP8 profile as nn.Module replacements.

Per call, both operands of every matmul are cast with the reference rule (ref_numerics.quantize,
a port of hw_float_ops.quantize_bf16_to_e4m3): a dynamic per-tensor power-of-two scale
s = floor(log2(amax_bf16 / 256)) and the vpack E4M3 encoder (quant_linear.py:153-154). Then:

  reference   F.linear on the dequantized operands in the layer dtype (NEW: the reference has no
              FP8 path without a functional model, quant_linear.py:173-175; this is the closest
              framework-accumulation analogue), BF16 bias added afterwards.
  reference_hw the reference functional-model path (quant_linear.py:145-172): the normalized
              E4M3 operands go through an MXU accumulation model (accum_backends: --mxu ipt|sa,
              implementation ref_cpu|torch), the BF16 result is rescaled by 2^(sx+sw) and cast to
              the layer dtype, and the BF16 bias is added.
"""

from __future__ import annotations

import re
from contextlib import ExitStack, contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

import algos
import reference.native_fp8 as native_fp8
import reference.ref_numerics as rn
from algos import base as algo_hooks


def _set_module(root: nn.Module, name: str, new: nn.Module) -> None:
    """smolVLA model_patcher.py:520-525."""
    parent = root
    *path, last = name.split(".")
    for part in path:
        parent = getattr(parent, part)
    setattr(parent, last, new)


class RefQuantLinear(nn.Module):
    """nn.Linear under the reference profile.

    The weight and bias are shared with the original layer (quant_linear.py:110-111), so
    `layer.weight.dtype` still reports the model dtype to code that casts activations by it.
    The weight's codes and scale are computed once: the reference recomputes them every call
    (quant_linear.py:154) with the same deterministic result.
    """

    def __init__(self, linear: nn.Linear, name: str = "", fm=None):
        super().__init__()
        self.fm = fm
        self.weight = linear.weight
        self.bias = linear.bias
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.layer_name = name
        self.algo = algos.current()              # None under --algo none (src/algos)
        self._profile_fisher = None
        self._dw = None
        with torch.no_grad():
            w_q, s_w, self.weight_block = algo_hooks.weight_cast(self.algo, name, self.weight.detach(), "linear")
        self.register_buffer("weight_codes", w_q, persistent=False)
        self.register_buffer("weight_scale_exp", s_w, persistent=False)
        self.native = native_fp8.is_native(fm)   # NEW native_fp8: fm is the FP8 kernel, fed codes
        if fm is not None and self.weight_block:
            raise NotImplementedError(f"{name}: per-block weight scales under an MXU accumulation model "
                                      "(or native_fp8) need per-block partial sums; not defined in algo/base")
        if fm is not None and not self.native:
            # Persistent normalized weight: exact E4M3 values at scale 1 (quant_linear.py:161-162).
            # Kept as one tensor so the reference models' weight cache stays valid across calls.
            self.register_buffer("weight_norm", rn.e4m3_unpack(w_q), persistent=False)
        self.residual = self.algo.residual_passes(name, "linear") if self.algo is not None else None
        if self.residual:
            self._init_residual(fm)

    def _init_residual(self, fm) -> None:
        """(g) residual_passes: the weight residual w - dequant(w_q), cast with the reference rule
        (its own per-tensor po2 scale). NEW, after the precision refinement of Markidis et al.
        (arXiv 1803.04014, Sec. V) with FP8 parts; docs/algos/mixed-precision.md."""
        if self.residual not in ("w", "wx"):
            raise ValueError(f"{self.layer_name}: residual_passes must be None, 'w' or 'wx', got {self.residual!r}")
        if self.weight_block:
            raise NotImplementedError(f"{self.layer_name}: residual passes with per-block weight scales")
        with torch.no_grad():
            w = self.weight.detach()
            w_r = (w.float() - self._dequant_weight().float()).to(w.dtype)
            q2, s2 = rn.quantize(w_r)
        self.register_buffer("weight_lo_codes", q2, persistent=False)
        self.register_buffer("weight_lo_scale_exp", s2, persistent=False)
        if fm is not None and not native_fp8.is_native(fm):
            self.register_buffer("weight_lo_norm", rn.e4m3_unpack(q2), persistent=False)
        self.algo.note("weight", f"linear:residual_{self.residual}", self.layer_name)

    def _dequant_weight(self) -> torch.Tensor:
        if self.weight_block:
            return algo_hooks.dequant(self.weight_codes, self.weight_scale_exp, self.weight_block)
        return rn.dequantize(self.weight_codes, self.weight_scale_exp)
    
    def _dequant_act(self, x: torch.Tensor) -> torch.Tensor:
        if self.algo is not None:
            x_q, s_x, blk = algo_hooks.act_cast(self.algo, self.layer_name, x, "linear")
            return algo_hooks.dequant(x_q, s_x, blk) if blk else rn.dequantize(x_q, s_x)
        x_q, s_x = rn.quantize(x)
        return rn.dequantize(x_q, s_x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._profile_fisher is not None:
            return self._forward_profiled(x)
        return self._forward_quant(x)

    def profile_fisher(self, enable: bool = True, record: bool = True) -> None:
        """Profile mode: forward returns the clean output (the FP8 cast is not differentiable, so a
        layer left on the quantized path blocks gradients to everything upstream of it). record=False
        keeps that clean pass-through but accumulates nothing (layers outside the profiled set)."""
        if enable:
            self._profile_fisher = torch.zeros(2, dtype=torch.float64, device=self.weight.device)
            self._dw = (self._dequant_weight().float() - self.weight.float().reshape(self.weight_codes.shape)
                        if record else None)
        else:
            self._profile_fisher, self._dw = None, None

    def fisher_report(self, n_batches: int = 1) -> dict:
        """Accumulated sensitivity per batch: weights, input activations, and their sum."""
        if self._profile_fisher is None:
            raise RuntimeError(f"{self.layer_name}: fisher_report called without profile_fisher(True)")
        fw, fx = (self._profile_fisher.double() / max(n_batches, 1)).tolist()
        return {"macs": self.weight.numel(), "fisher_w": fw, "fisher_x": fx, "fisher": fw + fx}

    def _forward_profiled(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.weight.dtype)
        y = F.linear(x, self.weight, self.bias)
  
        if y.requires_grad and self._dw is not None:
            with torch.no_grad():
                xd = x.detach()
                dx = (self._dequant_act(xd).float() - xd.float()).flatten(0, -2)

            xf = xd.float().flatten(0, -2)

            def fisher(grad):
                g = grad.float().flatten(0, -2)                     # [T, out]
                gW = g.T @ xf                                       # [out, in]
                gx = g   @ self.weight.detach().float()               # [T, in]
                self._profile_fisher[0] += 0.5 * (gW.pow(2) * self._dw.pow(2)).sum()
                self._profile_fisher[1] += 0.5 * (gx.pow(2) * dx.pow(2)).sum()

            y.register_hook(fisher)
            
        return y
        
    def _forward_quant(self, x: torch.Tensor) -> torch.Tensor:
        dtype = self.weight.dtype
        # Operate in the weight dtype; some activations arrive as float32 (quant_linear.py:133-137).
        x = x.to(dtype)
        if self.algo is not None:
            return self._forward_algo(x, dtype)
        x_q, s_x = rn.quantize(x)                                   # quant_linear.py:153
        if native_fp8.is_native(self.fm):
            y = self.fm.linear(x_q, s_x, self.weight_codes, self.weight_scale_exp, dtype, self.layer_name)
            return y if self.bias is None else y + self.bias
        if self.fm is not None:
            x_norm = rn.e4m3_unpack(x_q)                            # quant_linear.py:161
            y_unscaled = self.fm(x_norm, self.weight_norm, None, static=True)   # quant_linear.py:166
            # x 2^(sx+sw) in float32 (exact), then the layer dtype (quant_linear.py:169-170)
            scale = torch.exp2((s_x + self.weight_scale_exp).float())
            y = (y_unscaled * scale).to(dtype=dtype, device=x.device)
            if self.bias is not None:
                y = y + self.bias                                   # quant_linear.py:171-172
            return y
        x_dq = rn.dequantize(x_q, s_x).to(dtype)                    # exact in BF16 / FP32
        y = F.linear(x_dq, self._dequant_weight().to(dtype))        # NEW: framework accumulation
        if self.bias is not None:
            y = y + self.bias                                       # BF16 bias after (quant_linear.py:171-172)
        return y

    def _forward_algo(self, x, dtype):
        """forward with the algorithm's activation hooks (c, d online); the same steps as forward."""
        x_q, s_x, blk = algo_hooks.act_cast(self.algo, self.layer_name, x, "linear")
        if self.residual:
            return self._forward_residual(x, x_q, s_x, blk, dtype)
        if native_fp8.is_native(self.fm):
            if blk:
                raise NotImplementedError(f"{self.layer_name}: per-K-block activation scales under native_fp8")
            y = self.fm.linear(x_q, s_x, self.weight_codes, self.weight_scale_exp, dtype, self.layer_name)
        elif self.fm is not None:
            if blk:
                raise NotImplementedError(f"{self.layer_name}: per-block activation scales under an "
                                          "MXU accumulation model are not defined in algo/base")
            y_unscaled = self.fm(rn.e4m3_unpack(x_q), self.weight_norm, None, static=True)
            scale = algo_hooks.output_scale(s_x, self.weight_scale_exp)
            y = (y_unscaled * scale).to(dtype=dtype, device=x.device)
        else:
            x_dq = (algo_hooks.dequant(x_q, s_x, blk) if blk else rn.dequantize(x_q, s_x)).to(dtype)
            y = F.linear(x_dq, self._dequant_weight().to(dtype))
        if self.bias is not None:
            y = y + self.bias
        return y

    def _pass(self, x_q, s_x, w_q, w_norm, s_w, dtype, device):
        """One FP8 matmul pass: MXU model (+ x 2^(sx+sw)) or framework accumulation, as forward.

        The rescale goes through algo_hooks.output_scale, as _forward_algo does, so a per-token /
        per-channel exponent table (e.g. granularity) broadcasts to [..., out] and its table is
        built under the chip guard (the algorithm counts the extra vmuls; the multiply itself is
        counted by the router like any other pass)."""
        if native_fp8.is_native(self.fm):
            return self.fm.linear(x_q, s_x, w_q, s_w, dtype, self.layer_name)
        if self.fm is not None:
            y = self.fm(rn.e4m3_unpack(x_q), w_norm, None, static=True)
            return (y * algo_hooks.output_scale(s_x, s_w)).to(dtype=dtype, device=device)
        return F.linear(rn.dequantize(x_q, s_x).to(dtype), rn.dequantize(w_q, s_w).to(dtype))

    def _forward_residual(self, x, x_q, s_x, blk, dtype):
        """x_hi.w_hi + x_hi.w_lo (+ x_lo.w_hi for "wx"): two or three FP8 passes, each rescaled
        and summed in the layer dtype (on the chip: vmul / vadd on the VPU; whether the passes
        could share the MXU accumulator is hardware ruling R4)."""
        if blk:
            raise NotImplementedError(f"{self.layer_name}: residual passes with per-block activation scales")
        wn = getattr(self, "weight_norm", None)
        wln = getattr(self, "weight_lo_norm", None)
        y = self._pass(x_q, s_x, self.weight_codes, wn, self.weight_scale_exp, dtype, x.device)
        y = y + self._pass(x_q, s_x, self.weight_lo_codes, wln, self.weight_lo_scale_exp, dtype, x.device)
        if self.residual == "wx":
            x_r = x - rn.dequantize(x_q, s_x).to(dtype)               # activation residual (VPU vsub)
            xr_q, s_xr = rn.quantize(x_r)                             # reference cast of the residual
            y = y + self._pass(xr_q, s_xr, self.weight_codes, wn, self.weight_scale_exp, dtype, x.device)
        if self.bias is not None:
            y = y + self.bias
        return y

    def extra_repr(self) -> str:
        s_w = self.weight_scale_exp
        s_w = int(s_w) if s_w.numel() == 1 else f"{tuple(s_w.shape)}"
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"weight_scale_exp={s_w}, "
                f"profile={'reference' if self.fm is None else 'native_fp8' if native_fp8.is_native(self.fm) else 'reference_hw'}")


def quantize_reference(model: nn.Module, skip=(), fm_factory=None) -> dict:
    """Replace every nn.Linear (any dtype) whose fqn contains none of `skip`.

    With `fm_factory` (reference_hw), each layer gets its own accumulator instance, as the
    reference builds one functional model per layer (model_patcher.py:213-214); the reference
    CPU models cache one encoded weight per instance.

    The reference selects `type(module) is nn.Linear` (smolVLA model_patcher.py:496-499) and
    quantizes fp32 layers too (model_patcher.py:83-87; casts at quant_linear.py:136-137,170 and
    hw_float_ops.py:89), so unlike collab_v0 there is no bf16-only filter.
    """
    planned = [n for n, m in model.named_modules()
               if type(m) is nn.Linear and not any(s in n for s in skip)]
    modules = dict(model.named_modules())
    for n in planned:
        _set_module(model, n, RefQuantLinear(modules[n], name=n,
                                                   fm=fm_factory() if fm_factory else None))
    return {"planned": len(planned), "quantized": len(planned), "untouched": []}

def _select_layers(model: nn.Module, include=None, exclude=()) -> dict:
    """{fqn: RefQuantLinear} whose fqn matches regex `include` (None = all) and has no `exclude` substring."""
    pat = re.compile(include) if include else None
    return {n: m for n, m in model.named_modules()
            if isinstance(m, RefQuantLinear)
            and (pat is None or pat.search(n))
            and not any(s in n for s in exclude)}


@contextmanager
def _grad_anchored(model: nn.Module, anchors):
    """Freeze every parameter except `anchors`, so activations carry grad but weights do not.
    Restores the original requires_grad flags and clears grads on exit."""
    saved = {p: p.requires_grad for p in model.parameters()}
    model.requires_grad_(False)
    for p in anchors:
        p.requires_grad_(True)
    try:
        yield
    finally:
        model.zero_grad(set_to_none=True)
        for p, r in saved.items():
            p.requires_grad_(r)


@contextmanager
def _profiling(layers, record=True):
    layers = list(layers)
    for m in layers:
        m.profile_fisher(True, record)
    try:
        yield
    finally:
        for m in layers:
            m.profile_fisher(False)


def profile_fisher_model(model: nn.Module, batches, loss_fn, include=None, exclude=(), anchors=None,
                         attention=True) -> dict:
    """{fqn: {macs, fisher_w, fisher_x, fisher}} averaged over `batches`.

    attention  also profile the q / k / p / v operands of every reference-attention call (the model
               must have been routed through reference.ref_attention.patch_attention); entries are
               keyed "<site>.q|k|p|v" with {"kind": "attn", "fisher"}.

    model    already quantized by quantize_reference; every RefQuantLinear returns its clean output
             while profiling (selected or not), so the gradients are those of the unquantized model.
    loss_fn  loss_fn(model, batch) -> scalar loss (summed over tokens / elements).
    include  regex over layer fqns (None = every RefQuantLinear); exclude: substrings to skip.
    anchors  callable(model) -> params that get requires_grad so every profiled layer's output
             is in the graph; default is the input embedding (LLM). A VLA needs one anchor per
             input path, e.g. vision patch embed, token embed, state / action projections.
    """
    import reference.ref_attention as ra

    layers = _select_layers(model, include, exclude)
    if not layers:
        raise ValueError(f"no RefQuantLinear matches include={include!r} exclude={exclude!r}")
    anchors = anchors or (lambda m: [m.get_input_embeddings().weight])
    nb = max(len(batches), 1)
    with ExitStack() as st:
        st.enter_context(_grad_anchored(model, anchors(model)))
        chosen = set(layers.values())
        others = [m for m in model.modules() if isinstance(m, RefQuantLinear) and m not in chosen]
        st.enter_context(_profiling(others, record=False))     # not profiled, but must not block gradients
        st.enter_context(_profiling(layers.values()))
        acc = st.enter_context(ra.attn_fisher()) if attention else {}
        for b in batches:
            ra.new_forward()
            loss_fn(model, b).backward()
            model.zero_grad(set_to_none=True)
        res = {n: m.fisher_report(len(batches)) for n, m in layers.items()}
        res.update({k: {"kind": "attn", "fisher": v.item() / nb} for k, v in acc.items()})
    return res


def write_report(res: dict, meta: dict, out: str) -> None:
    """{"meta", "layers"} JSON as src/scripts/plot_fisher.py reads it."""
    import json
    from pathlib import Path

    meta = {**meta, "torch": torch.__version__, "loss": "summed (see script)",
            "score": "0.5 * sum(grad^2 * delta^2), averaged over batches; fisher = fisher_w + fisher_x"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"meta": meta, "layers": res}, indent=1))
    print("wrote", out)
