"""Reference-profile attention score matmuls (Q.K^T and P.V) and Conv2d.

Attention follows the reference author's per-head routine (smolVLA-quant
smolvla_inout/model_patcher.py:624-723 for the VLM text and expert layers, 798-866 for the vision
encoder):

  for each (batch b, head h), with GQA expanded first (640-646):
    q, k -> per-(b, h) po2 scale + vpack cast (669-674)
    scores = MM(q, k) * 2^(sq+sk) * head_dim^-0.5                    (675-677)
    mask; softmax                                                    (680-684)
    P -> bf16 -> per-(b, h) cast; V -> per-(b, h) cast               (684-694)
    out = MM(P, V) * 2^(sp+sv) -> bf16                               (696-699)

where MM is the reference_hw accumulation model or, under `reference`, a matmul on the
dequantized operands (NEW, as for the linear layers). The two variants differ where the
reference's own two paths differ:

  text    scores stay float32 through the mask and softmax, the mask is boolean
          (where(mask, s, finfo(float32).min)) (675-684). Used for SmolVLA text/expert layers and,
          as the closest analogue, for Llama-family LMs (NEW: Llama's eager mask is additive, so it
          is added instead, as LlamaAttention's eager path does: modeling_llama.py:213-214).
  vision  scores are cast to bf16, the additive mask is added, softmax runs in float32 and P is
          cast to bf16 (827-836).

(B, H) are processed in one batched call instead of the reference's Python loop over heads; the
per-head scales make that equivalent (tests/test_ref_attention.py compares against the
reference closure).
"""

from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import ref_numerics as rn

ATTN_KEY = "atlas_ref"


def _mm(fm, a_q, a_s, b_q, b_s, dtype_out=torch.float32):
    """a [..., M, K] times b [..., N, K]^T on cast operands; result * 2^(sa+sb) in float32."""
    if fm is not None:
        y = fm(rn.e4m3_unpack(a_q), rn.e4m3_unpack(b_q), None)
        return y * torch.exp2((a_s + b_s).float())
    # NEW: reference profile without an MXU model: matmul on the dequantized operands in float32.
    a = rn.dequantize(a_q, a_s)
    b = rn.dequantize(b_q, b_s)
    return torch.matmul(a, b.transpose(-1, -2))


def ref_attention(q, k, v, scaling, mask=None, fm_qk=None, fm_av=None, variant="text"):
    """q [B, H, Sq, D], k [B, H, Skv, D], v [B, H, Skv, Dv] (GQA already expanded).

    mask: text variant - bool [B, Sq, Skv] or [B, 1|H, Sq, Skv] (True = attend), or an additive float
    mask of the same shapes; vision variant - additive float [B, 1|H, Sq, Skv] or None.
    Returns bf16 [B, H, Sq, Dv].
    """
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)    # 649-651, 815-817
    hd = (-2, -1)
    q_q, s_q = rn.quantize(q, hd)
    k_q, s_k = rn.quantize(k, hd)
    scores = _mm(fm_qk, q_q, s_q, k_q, s_k) * scaling                              # 675-677
    if variant == "vision":
        scores = scores.to(torch.bfloat16)                                         # 827-829
        if mask is not None:
            scores = scores + mask                                                 # 831-833
        p = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)               # 836
    else:
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[:, None]
            if mask.dtype == torch.bool:
                scores = torch.where(mask, scores, torch.finfo(torch.float32).min)  # 680-681
            else:
                scores = scores + mask                                             # NEW: additive mask
        p = torch.softmax(scores, dim=-1).to(torch.bfloat16)                       # 684
    p_q, s_p = rn.quantize(p, hd)                                                  # 689
    v_q, s_v = rn.quantize(v, hd)                                                  # 690
    v_t_q = v_q.transpose(-1, -2)                                                  # v_norm.T (696)
    out = _mm(fm_av, p_q, s_p, v_t_q, s_v)                                         # 696
    return out.to(torch.bfloat16)                                                  # 696-698


def _expand_kv(x, n_rep):
    """[B, Hkv, S, D] -> [B, Hkv * n_rep, S, D] (each kv head repeated n_rep times, as
    model_patcher.py:641-646 and transformers' repeat_kv)."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


# ---------------------------------------------------------------------------
# Hooks into the models
# ---------------------------------------------------------------------------

def _hf_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    """transformers attention-interface function (Llama, SmolVLM vision).

    query [B, H, Sq, D], key/value [B, Hkv, S, D]; returns ([B, Sq, H, D], None) like
    eager_attention_forward. Modules without reference settings fall back to sdpa.
    """
    cfg = getattr(module, "_atlas_ref_attn", None)
    if cfg is None:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        return ALL_ATTENTION_FUNCTIONS["sdpa"](module, query, key, value, attention_mask,
                                               scaling=scaling, dropout=dropout, **kwargs)
    n_rep = query.shape[1] // key.shape[1]
    k, v = _expand_kv(key, n_rep), _expand_kv(value, n_rep)
    scaling = scaling if scaling is not None else query.shape[-1] ** -0.5
    out = ref_attention(query, k, v, scaling, attention_mask, cfg["fm_qk"], cfg["fm_av"], cfg["variant"])
    return out.to(query.dtype).transpose(1, 2).contiguous(), None                  # 852-853


_registered = False


def register_hf_attention() -> None:
    """Register ATTN_KEY with transformers' attention and mask interfaces (eager-style masks)."""
    global _registered
    if _registered:
        return
    from transformers import AttentionInterface
    from transformers.masking_utils import AttentionMaskInterface, eager_mask

    AttentionInterface.register(ATTN_KEY, _hf_attention)
    AttentionMaskInterface.register(ATTN_KEY, eager_mask)
    _registered = True


def _smolvla_text_attention(self_inner, attention_mask, batch_size, head_dim,
                            query_states, key_states, value_states):
    """SmolVLMWithExpertModel.eager_attention_forward replacement (model_patcher.py:624-723)."""
    cfg = self_inner._atlas_ref_attn
    n_heads, n_kv = self_inner.num_attention_heads, self_inner.num_key_value_heads
    q = query_states.transpose(1, 2)                                               # [B, H, Sq, D]
    k = _expand_kv(key_states.transpose(1, 2), n_heads // n_kv)
    v = _expand_kv(value_states.transpose(1, 2), n_heads // n_kv)
    out = ref_attention(q, k, v, head_dim ** -0.5, attention_mask, cfg["fm_qk"], cfg["fm_av"], "text")
    dv = out.shape[-1]
    return out.permute(0, 2, 1, 3).reshape(batch_size, -1, n_heads * dv)          # 723


def patch_attention(model: nn.Module, fm_factory=None) -> list:
    """Route the model's attention score matmuls through the reference routine.

    Returns the list of patched sites (recorded in the result JSON). Sites:
      - SmolVLA text + expert layers: SmolVLMWithExpertModel.eager_attention_forward, patched on the
        instance (model_patcher.py:573-729), text variant;
      - modules driven by a transformers config with an attention-interface key: the config is
        switched to ATTN_KEY (eager-style masks). SmolVLM vision layers use the vision variant;
        everything else (Llama-family decoders) the text variant.
    Models with neither (e.g. custom attention modules) are left unchanged.
    """
    new_fm = (lambda: fm_factory()) if fm_factory else (lambda: None)
    sites = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SmolVLMWithExpertModel" and hasattr(mod, "eager_attention_forward"):
            mod._atlas_ref_attn = {"fm_qk": new_fm(), "fm_av": new_fm(), "variant": "text"}
            mod.eager_attention_forward = types.MethodType(_smolvla_text_attention, mod)
            sites.append({"site": name, "kind": "smolvla_text_expert", "variant": "text"})
    smolvla = bool(sites)
    configs = {}
    for name, mod in model.named_modules():
        cfg = getattr(mod, "config", None)
        if cfg is None or not hasattr(cfg, "_attn_implementation"):
            continue
        if not (hasattr(mod, "q_proj") and hasattr(mod, "k_proj") and hasattr(mod, "v_proj")):
            continue
        variant = "vision" if "vision" in type(mod).__name__.lower() else "text"
        if smolvla and variant == "text":
            continue          # SmolVLA text/expert attention runs through eager_attention_forward
        mod._atlas_ref_attn = {"fm_qk": new_fm(), "fm_av": new_fm(), "variant": variant}
        configs[id(cfg)] = cfg
        sites.append({"site": name, "kind": type(mod).__name__, "variant": variant})
    if configs:
        register_hf_attention()
        for cfg in configs.values():
            cfg._attn_implementation = ATTN_KEY
    return sites


# ---------------------------------------------------------------------------
# Conv2d
# ---------------------------------------------------------------------------

class RefQuantConv2d(nn.Module):
    """nn.Conv2d (groups=1) under the reference profiles: port of QuantConv2d's functional-model
    path (model_patcher.py:345-375): im2col, per-tensor cast of the unfolded input and the flattened
    weight, MM, x 2^(sx+sw), cast to the weight dtype, bias added after. Under `reference` the MM is
    a matmul on the dequantized operands (NEW)."""

    def __init__(self, conv: nn.Conv2d, name: str = "", fm=None):
        super().__init__()
        self.weight, self.bias = conv.weight, conv.bias
        self.kernel_size, self.stride = conv.kernel_size, conv.stride
        self.padding, self.dilation = conv.padding, conv.dilation
        if isinstance(self.padding, str):                                          # 330-337
            if self.padding != "valid":
                raise ValueError(f"padding={self.padding!r} not supported")
            self.padding = (0, 0)
        self.out_channels = conv.out_channels
        self.layer_name, self.fm = name, fm
        with torch.no_grad():
            w_q, s_w = rn.quantize(self.weight.detach().reshape(self.out_channels, -1))   # 357, 362
        self.register_buffer("weight_codes", w_q, persistent=False)
        self.register_buffer("weight_scale_exp", s_w, persistent=False)
        if fm is not None:
            self.register_buffer("weight_norm", rn.e4m3_unpack(w_q), persistent=False)

    def forward(self, x):
        w = self.weight
        B, _, H, W = x.shape
        oh = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        ow = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        cols = F.unfold(x, kernel_size=self.kernel_size, stride=self.stride,
                        padding=self.padding, dilation=self.dilation)               # 354-355
        x_flat = cols.permute(0, 2, 1).reshape(B * oh * ow, -1).to(w.dtype)         # 356
        x_q, s_x = rn.quantize(x_flat)                                              # 361
        if self.fm is not None:
            y = self.fm(rn.e4m3_unpack(x_q), self.weight_norm, None, static=True)    # 369
            y = (y * torch.exp2((s_x + self.weight_scale_exp).float())).to(dtype=w.dtype, device=x.device)
        else:
            y = F.linear(rn.dequantize(x_q, s_x).to(w.dtype),
                         rn.dequantize(self.weight_codes, self.weight_scale_exp).to(w.dtype))   # NEW
        y = y.reshape(B, oh * ow, self.out_channels).permute(0, 2, 1).reshape(B, self.out_channels, oh, ow)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1).to(dtype=w.dtype, device=x.device)  # 374-375
        return y


def quantize_conv2d(model: nn.Module, skip=(), fm_factory=None) -> dict:
    """Replace every nn.Conv2d (model_patcher.py:508-511) that the reference path supports:
    groups == 1 (339-343) and integer or 'valid' padding (330-337); others are reported."""
    from ref_profile import _set_module

    done, untouched = [], []
    for n, m in list(model.named_modules()):
        if type(m) is not nn.Conv2d or any(s in n for s in skip):
            continue
        if m.groups != 1 or (isinstance(m.padding, str) and m.padding != "valid") or m.padding_mode != "zeros":
            untouched.append(n)
            continue
        _set_module(model, n, RefQuantConv2d(m, n, fm_factory() if fm_factory else None))
        done.append(n)
    return {"quantized": len(done), "untouched": untouched}
