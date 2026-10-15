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
  pi0     Pi0 (pi0-quant model_patcher.py:738-823 SDPA, 911-1005 Gemma eager): scores cast to bf16,
          mask added, softmax in the dtype of the result (float32 with Pi0's float32 additive mask,
          bf16 for the unmasked SigLIP SDPA call), P cast to bf16. See patch_pi0_attention.

(B, H) are processed in one batched call instead of the reference's Python loop over heads; the
per-head scales make that equivalent (tests/test_ref_attention.py compares against the
reference closure).
"""

from __future__ import annotations

import threading
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import algos
import ref_numerics as rn
from algos import base as algo_hooks

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


def _cast(algo, site, operand, x, dims):
    """rn.quantize(x, dims) (per-(b, h) scales); with an algorithm, its act_cast hook."""
    if algo is None:
        return rn.quantize(x, dims)
    q, s, blk = algo_hooks.act_cast(algo, site, x, "attention." + operand, dims)
    if blk:
        raise NotImplementedError(f"{site}: per-block scales on attention operands are not defined "
                                  "in algo/base")
    return q, s


def ref_attention(q, k, v, scaling, mask=None, fm_qk=None, fm_av=None, variant="text", algo=None, site=""):
    """q [B, H, Sq, D], k [B, H, Skv, D], v [B, H, Skv, Dv] (GQA already expanded).

    mask: text variant - bool [B, Sq, Skv] or [B, 1|H, Sq, Skv] (True = attend), or an additive float
    mask of the same shapes; vision variant - additive float [B, 1|H, Sq, Skv] or None.
    Returns bf16 [B, H, Sq, Dv]. `algo` (src/algos; None under --algo none) gets the operand casts
    as act_cast(site, x, "attention.q|k|p|v"); with no hook they are the per-(b, h) casts below.
    """
    q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)    # 649-651, 815-817
    hd = (-2, -1)
    q_q, s_q = _cast(algo, site, "q", q, hd)
    k_q, s_k = _cast(algo, site, "k", k, hd)
    scores = _mm(fm_qk, q_q, s_q, k_q, s_k) * scaling                              # 675-677
    if variant == "vision":
        scores = scores.to(torch.bfloat16)                                         # 827-829
        if mask is not None:
            scores = scores + mask                                                 # 831-833
        p = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)               # 836
    elif variant == "pi0":
        # pi0-quant model_patcher.py: Gemma eager 957-965, SigLIP SDPA 772-787. The scores are
        # rounded to BF16, the mask is added (Pi0's additive mask is float32, pi0_pytorch.py:160,
        # so the sum and the softmax are float32; without a mask the softmax runs in BF16) and P is
        # cast to BF16 by the quantizer (hw_float_ops.py:89).
        scores = scores.to(torch.bfloat16)                                         # 772-774, 957-959
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[:, None]
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(~mask, float("-inf"))                  # 724-725
            else:
                scores = scores + mask[..., :scores.shape[-2], :scores.shape[-1]]  # 727, 963
        p = torch.softmax(scores, dim=-1).to(torch.bfloat16)                       # 787, 965
    else:
        if mask is not None:
            if mask.dim() == 3:
                mask = mask[:, None]
            if mask.dtype == torch.bool:
                scores = torch.where(mask, scores, torch.finfo(torch.float32).min)  # 680-681
            else:
                scores = scores + mask                                             # NEW: additive mask
        p = torch.softmax(scores, dim=-1).to(torch.bfloat16)                       # 684
    p_q, s_p = _cast(algo, site, "p", p, hd)                                       # 689
    v_q, s_v = _cast(algo, site, "v", v, hd)                                       # 690
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
    out = ref_attention(query, k, v, scaling, attention_mask, cfg["fm_qk"], cfg["fm_av"], cfg["variant"],
                        cfg.get("algo"), cfg.get("site", ""))
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
    out = ref_attention(q, k, v, head_dim ** -0.5, attention_mask, cfg["fm_qk"], cfg["fm_av"], "text",
                        cfg.get("algo"), cfg.get("site", ""))
    dv = out.shape[-1]
    return out.permute(0, 2, 1, 3).reshape(batch_size, -1, n_heads * dv)          # 723


def _algo_cfg(site) -> dict:
    """{"algo", "site"} for a patched site while an algorithm is active (src/algos), else {}."""
    algo = algos.current()
    return {"algo": algo, "site": site} if algo is not None else {}


def patch_attention(model: nn.Module, fm_factory=None, skip=()) -> list:
    """Route the model's attention score matmuls through the reference routine.

    Returns the list of patched sites (recorded in the result JSON). Sites:
      - SmolVLA text + expert layers: SmolVLMWithExpertModel.eager_attention_forward, patched on the
        instance (model_patcher.py:573-729), text variant;
      - modules driven by a transformers config with an attention-interface key: the config is
        switched to ATTN_KEY (eager-style masks). SmolVLM vision layers use the vision variant;
        everything else (Llama-family decoders) the text variant.
    Models with neither (e.g. custom attention modules) are left unchanged.
    """
    if any(type(m).__name__ == "PaliGemmaWithExpertModel" for m in model.modules()):
        return patch_pi0_attention(model, fm_factory, skip)
    new_fm = (lambda: fm_factory()) if fm_factory else (lambda: None)
    sites = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "SmolVLMWithExpertModel" and hasattr(mod, "eager_attention_forward"):
            mod._atlas_ref_attn = {"fm_qk": new_fm(), "fm_av": new_fm(), "variant": "text", **_algo_cfg(name)}
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
        mod._atlas_ref_attn = {"fm_qk": new_fm(), "fm_av": new_fm(), "variant": variant, **_algo_cfg(name)}
        configs[id(cfg)] = cfg
        sites.append({"site": name, "kind": type(mod).__name__, "variant": variant})
    if configs:
        register_hf_attention()
        for cfg in configs.values():
            cfg._attn_implementation = ATTN_KEY
    return sites


# ---------------------------------------------------------------------------
# Pi0 (openpi PI0Pytorch): the three routes of pi0-quant model_patcher.py
# ---------------------------------------------------------------------------

# Every patched Pi0 attention module carries `_atlas_ref_attn`; the three module-level functions
# below are installed once per process and fall through to the saved originals for any other
# module (an unquantized baseline or the clean --rmse copy share the same transformers modules).
_PI0_ORIG: dict = {}
_pi0_current = threading.local()


def _pi0_cfg(module):
    cfg = getattr(module, "_atlas_ref_attn", None)
    return cfg if isinstance(cfg, dict) and cfg.get("variant") == "pi0" else None


def _pi0_gemma_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """modeling_gemma.eager_attention_forward replacement (pi0-quant model_patcher.py:911-1005).

    Called for the language model, the action expert and the joint co-attention layers
    (gemma_pytorch.py:200-207 calls the module-level function directly)."""
    cfg = _pi0_cfg(module)
    if cfg is None:
        return _PI0_ORIG["gemma"](module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    n_rep = module.num_key_value_groups                                           # 925-926 repeat_kv
    k, v = _expand_kv(key, n_rep), _expand_kv(value, n_rep)
    out = ref_attention(query, k, v, scaling, attention_mask, cfg["fm_qk"], cfg["fm_av"], "pi0",
                        cfg.get("algo"), cfg.get("site", ""))
    cfg["calls"]["gemma_eager"] = cfg["calls"].get("gemma_eager", 0) + 1
    return out.to(query.dtype).transpose(1, 2).contiguous(), None                  # 982-1005


def _pi0_siglip_eager(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    """modeling_siglip.eager_attention_forward replacement (model_patcher.py:1087-1165): fires
    only when the vision config uses eager attention (the chip profile sets it). Its softmax is
    float32 then cast to bf16 (1134): the `vision` variant."""
    cfg = _pi0_cfg(module)
    if cfg is None:
        return _PI0_ORIG["siglip"](module, query, key, value, attention_mask, scaling, dropout, **kwargs)
    out = ref_attention(query, key, value, scaling, attention_mask, cfg["fm_qk"], cfg["fm_av"], "vision",
                        cfg.get("algo"), cfg.get("site", ""))
    cfg["calls"]["siglip_eager"] = cfg["calls"].get("siglip_eager", 0) + 1
    return out.to(query.dtype).transpose(1, 2).contiguous(), None                  # 1151-1165


def _pi0_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kwargs):
    """F.scaled_dot_product_attention replacement (model_patcher.py:738-823): active while a
    patched Pi0 self_attn module runs (forward pre/post hooks, 703-719). In Pi0 this is the
    SigLIP vision tower, whose config uses sdpa (transformers 4.53.2 default)."""
    mod = getattr(_pi0_current, "module", None)
    cfg = _pi0_cfg(mod) if mod is not None else None
    if cfg is None or kwargs:
        return _PI0_ORIG["sdpa"](query, key, value, attn_mask, dropout_p, is_causal, scale=scale, **kwargs)
    scale = scale if scale is not None else query.size(-1) ** -0.5                 # 746-747
    mask = attn_mask
    if mask is None and is_causal:                                                 # 728-734
        sq, skv = query.shape[-2], key.shape[-2]
        mask = torch.ones(sq, skv, dtype=torch.bool, device=query.device).tril()[None, None]
    out = ref_attention(query, key, value, scale, mask, cfg["fm_qk"], cfg["fm_av"], "pi0",
                        cfg.get("algo"), cfg.get("site", ""))
    cfg["calls"]["sdpa"] = cfg["calls"].get("sdpa", 0) + 1
    return out.to(query.dtype)                                                     # 804


def _install_pi0_routes() -> None:
    if _PI0_ORIG:
        return
    from transformers.models.gemma import modeling_gemma
    from transformers.models.siglip import modeling_siglip

    _PI0_ORIG.update(gemma=modeling_gemma.eager_attention_forward,
                     siglip=modeling_siglip.eager_attention_forward,
                     sdpa=F.scaled_dot_product_attention)
    modeling_gemma.eager_attention_forward = _pi0_gemma_eager                      # 1007
    modeling_siglip.eager_attention_forward = _pi0_siglip_eager                    # 1167
    F.scaled_dot_product_attention = _pi0_sdpa                                     # 825


def pi0_component(path: str) -> str:
    """pi0-quant model_patcher.py:92-126 (_COMPONENT_RULES, first match wins)."""
    for sub, comp in PI0_COMPONENT_RULES:
        if sub in path:
            return comp
    return "unknown"


PI0_COMPONENT_RULES = (                                                            # model_patcher.py:92-109
    ("action_in_proj", "action_head"), ("action_out_proj", "action_head"),
    ("action_time_mlp_in", "action_head"), ("action_time_mlp_out", "action_head"),
    ("state_proj", "action_head"), ("time_mlp_in", "action_head"), ("time_mlp_out", "action_head"),
    ("gemma_expert", "action_expert"), ("vision_tower", "vision"), ("language_model", "language"),
    ("paligemma", "language"),
)


def patch_pi0_attention(model: nn.Module, fm_factory=None, skip=()) -> list:
    """Pi0's attention score matmuls through the reference routine, as pi0-quant run_eval.py:774-796
    does with patch_attn_sdpa + patch_attn_eager + patch_attn_siglip_eager (ops `attention`, all
    groups): every module named `*.self_attn` (model_patcher.py:703-707, 893-900) gets its own
    score-matmul accumulators. The reference shares one pair per route (699-700, 902-903); the
    MXU models are stateless for non-weight operands here (ref_profile F-cache), so this is
    equivalent. Each returned site carries a live `calls` dict {route: count}."""
    _install_pi0_routes()
    new_fm = (lambda: fm_factory()) if fm_factory else (lambda: None)
    sites = []
    for name, mod in model.named_modules():
        if name.split(".")[-1] != "self_attn" or any(x and x in name for x in skip):
            continue                                # --skip leaves a site unpatched (NEW for Pi0)
        calls: dict = {}
        mod._atlas_ref_attn = {"fm_qk": new_fm(), "fm_av": new_fm(), "variant": "pi0", "calls": calls,
                               **_algo_cfg(name)}

        def pre(m, *_):
            _pi0_current.module = m

        def post(m, *_):
            _pi0_current.module = None

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        sites.append({"site": name, "kind": type(mod).__name__, "variant": "pi0",
                      "component": pi0_component(name), "calls": calls})
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
        self.algo = algos.current()              # None under --algo none (src/algos)
        with torch.no_grad():
            w_q, s_w, self.weight_block = algo_hooks.weight_cast(
                self.algo, name, self.weight.detach().reshape(self.out_channels, -1), "conv2d")   # 357, 362
        if self.weight_block and fm is not None:
            raise NotImplementedError(f"{name}: per-block weight scales under an MXU accumulation model "
                                      "are not defined in algo/base")
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
        if self.algo is not None:
            y = self._mm_algo(x_flat, w)
        elif self.fm is not None:
            x_q, s_x = rn.quantize(x_flat)                                          # 361
            y = self.fm(rn.e4m3_unpack(x_q), self.weight_norm, None, static=True)    # 369
            y = (y * torch.exp2((s_x + self.weight_scale_exp).float())).to(dtype=w.dtype, device=x.device)
        else:
            x_q, s_x = rn.quantize(x_flat)                                          # 361
            y = F.linear(rn.dequantize(x_q, s_x).to(w.dtype),
                         rn.dequantize(self.weight_codes, self.weight_scale_exp).to(w.dtype))   # NEW
        y = y.reshape(B, oh * ow, self.out_channels).permute(0, 2, 1).reshape(B, self.out_channels, oh, ow)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1).to(dtype=w.dtype, device=x.device)  # 374-375
        return y

    def _mm_algo(self, x_flat, w):
        """The matmul of forward with the algorithm's activation hooks (c, d online)."""
        x_q, s_x, blk = algo_hooks.act_cast(self.algo, self.layer_name, x_flat, "conv2d")
        if self.fm is not None:
            if blk:
                raise NotImplementedError(f"{self.layer_name}: per-block activation scales under an "
                                          "MXU accumulation model are not defined in algo/base")
            y = self.fm(rn.e4m3_unpack(x_q), self.weight_norm, None, static=True)
            return (y * algo_hooks.output_scale(s_x, self.weight_scale_exp)).to(
                dtype=w.dtype, device=x_flat.device)
        w_dq = (algo_hooks.dequant(self.weight_codes, self.weight_scale_exp, self.weight_block)
                if self.weight_block else rn.dequantize(self.weight_codes, self.weight_scale_exp))
        x_dq = algo_hooks.dequant(x_q, s_x, blk) if blk else rn.dequantize(x_q, s_x)
        return F.linear(x_dq.to(w.dtype), w_dq.to(w.dtype))


def quantize_conv2d(model: nn.Module, skip=(), fm_factory=None, exact_skip=()) -> dict:
    """Replace every nn.Conv2d (model_patcher.py:508-511) that the reference path supports:
    groups == 1 (339-343) and integer or 'valid' padding (330-337); others are reported."""
    from ref_profile import _set_module

    done, untouched = [], []
    for n, m in list(model.named_modules()):
        if type(m) is not nn.Conv2d or n in exact_skip or any(s in n for s in skip):
            continue
        if m.groups != 1 or (isinstance(m.padding, str) and m.padding != "valid") or m.padding_mode != "zeros":
            untouched.append(n)
            continue
        _set_module(model, n, RefQuantConv2d(m, n, fm_factory() if fm_factory else None))
        done.append(n)
    return {"quantized": len(done), "untouched": untouched}
