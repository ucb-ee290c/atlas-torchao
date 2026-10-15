"""smoothquant: SmoothQuant outlier migration folded into the producing ops, plus a folded po2 gain.

Source: Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large
Language Models", arXiv 2211.10438v7.
  Eq. (3) (Sec. 4)  Y = (X diag(s)^-1) (diag(s) W): divide the activation channels by s and
                    multiply the matching weight input columns by s; the product is unchanged.
  Eq. (4) (Sec. 4)  s_j = max|X_j|^alpha / max|W_j|^(1 - alpha), activation maxima from calibration.
  Sec. 4            s is fused offline into the op that produces X (no run-time cost).
  Sec. 4 / Fig. 6   the paper smooths the QKV and FC1 inputs. Smoothing the down_proj and o_proj
                    inputs as well is NEW here: those are Atlas's worst layers under the fixed
                    activation scale (handoff/atlas-fp8/algos/smoothquant.md).

NEW (reasons in docs/algos/smoothquant.md):
  po2=true      s is rounded to 2^round(log2 s); a power-of-two fold is exact in BF16 / FP32
                (only exponents change) and matches the E8M0 scale world of Atlas.
  gain=true     after smoothing, a per-group power-of-two gain g = 2^-ceil(log2(amax_post / 448))
                (the collaborator's weight-scale rule, atlas_quant.py:172-179, applied to the
                calibrated activation maximum) is folded into the same producer. Under the
                collaborator's fixed 2^0 activation scale (atlas_torchao.py:54-64) this gives the
                activation a calibrated static scale at no run-time cost. It is recorded as a folded
                transform (prepare record "gain"), not as an activation Scale, so it stays distinct
                from a static-scale algorithm.
  static=true   (reference steps only) act_scale returns a static per-tensor exponent from the
                post-fold calibration maximum by the reference rule floor(log2(amax / 256))
                (ref_numerics.po2_scale_exp = pi0-quant hw_float_ops.py:38-57).

Fold sites, per decoder layer (every one an exact identity in real arithmetic):
  qkv   input_layernorm  -> q/k/v_proj columns        (shared s over the consumers' columns)
  fc1   post_attention_layernorm -> gate/up columns
  down  up_proj output rows -> down_proj columns       (down input SiLU/GELU(gate) * up is linear in up)
  o     v_proj output rows -> o_proj columns           (o input = softmax . V is linear in V; s is
                                                        tied over the query heads sharing a kv head)
Norms: plain RMSNorm (weight * x) and Gemma's (1 + weight) * x. The Gemma fold w' = (1+w)/f - 1 is
not exact in BF16 even for po2 f, so a BF16 Gemma norm weight is promoted to FP32 (gemma_norm=fp32,
the default) or refused (gemma_norm=refuse). Adaptive norms (a `dense` modulation) are refused.

Structures: a standalone decoder stack (e.g. LlamaForCausalLM), SmolVLA (SmolVLMWithExpertModel:
joint layers tie the VLM's and the expert's V; cross-attention layers feed the VLM's V into the
expert's v_proj, whose columns undo the VLM fold) and Pi0 (PaliGemmaWithExpertModel: every layer is
joint, so V is tied across PaliGemma and the expert). Any other transformer block (SigLIP / CLIP
encoder layers, DiT blocks, ...) is not implemented: prepare raises NotImplementedError unless
partial=true, which folds what is supported and lists every unfolded block in the record. With
partial=true and no supported layer at all, prepare folds nothing and records folded_groups: 0 and
a warning (NEW); static=true still gives every Linear its static exponent from calibration.

Parameters (`--algo smoothquant:k=v,...`):
  alpha=0.5     migration strength; `off` disables smoothing (gain only)
  po2=true      power-of-two s
  gain=true     fold the po2 gain
  gain_target=448
  static=false  static activation exponent on the reference steps
  sites=qkv+fc1+down+o
  gemma_norm=fp32 | refuse
  partial=false
"""

from __future__ import annotations

import math
import warnings as _warnings
from collections import Counter

import torch
import torch.nn as nn

from algos.base import Algo, Scale, observe_inputs

SITES = ("qkv", "fc1", "down", "o")
_BLOCK_SUFFIXES = ("EncoderLayer", "DecoderLayer", "Block", "TransformerBlock")
_PLAIN_RMSNORMS = ("LlamaRMSNorm", "MistralRMSNorm", "Qwen2RMSNorm")


# ---------------------------------------------------------------------------------------------
# Structure discovery
# ---------------------------------------------------------------------------------------------

def _is_decoder_layer(m) -> bool:
    a, mlp = getattr(m, "self_attn", None), getattr(m, "mlp", None)
    return (a is not None and mlp is not None and hasattr(m, "input_layernorm")
            and hasattr(m, "post_attention_layernorm")
            and all(isinstance(getattr(a, k, None), nn.Linear) for k in ("q_proj", "k_proj", "v_proj", "o_proj"))
            and all(isinstance(getattr(mlp, k, None), nn.Linear) for k in ("gate_proj", "up_proj", "down_proj")))


def _head_dim(layer) -> int:
    hd = getattr(layer.self_attn, "head_dim", None)
    if not hd:
        raise NotImplementedError(f"{type(layer.self_attn).__name__}: no head_dim attribute")
    return int(hd)


class _Layer:
    """One decoder layer and how its tensors are consumed."""

    def __init__(self, fqn, layer, qkv_consumers=("q_proj", "k_proj", "v_proj")):
        self.fqn, self.m = fqn, layer
        self.qkv_consumers = tuple(qkv_consumers)

    def lin(self, short):
        part = "self_attn" if short.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) else "mlp"
        return f"{self.fqn}.{part}.{short}", getattr(getattr(self.m, part), short)


def _discover(model):
    """(standalone stacks, smolvla containers, pi0 containers, all covered layer fqns)."""
    containers_s, containers_p = [], []
    for n, m in model.named_modules():
        if type(m).__name__ == "SmolVLMWithExpertModel":
            containers_s.append((n, m))
        elif type(m).__name__ == "PaliGemmaWithExpertModel":
            containers_p.append((n, m))
    inside = [p for p, _ in containers_s + containers_p]
    standalone = []
    for n, m in model.named_modules():
        layers = getattr(m, "layers", None)
        if not isinstance(layers, nn.ModuleList) or not len(layers) or not all(_is_decoder_layer(l) for l in layers):
            continue
        if any(c == "" or n == c or n.startswith(c + ".") for c in inside):
            continue                                   # handled by its container's adapter
        standalone.append((n, m))
    return standalone, containers_s, containers_p


def _fqn(base, rel):
    return f"{base}.{rel}" if base else rel


# ---------------------------------------------------------------------------------------------
# Groups: one producer tensor whose channels are divided by f, consumers whose columns are * f
# ---------------------------------------------------------------------------------------------

class _Group:
    def __init__(self, site, name, C):
        self.site, self.name, self.C = site, name, C
        self.primary = []      # (fqn, linear, idx[in_features] -> group channel); define s and g
        self.undo = []         # consumers that only compensate (SmolVLA cross-attention v_proj)
        self.norms = []        # (fqn, norm)          channels = group channels
        self.rows = []         # (fqn, linear)        output rows = group channels


def _ident(n, device=None):
    return torch.arange(n, device=device)


def _o_index(n_q, n_kv, hd):
    """o_proj input channel h*hd + d -> kv channel kv(h)*hd + d (HF repeat_kv: kv(h) = h // groups)."""
    if n_q % n_kv:
        raise NotImplementedError(f"{n_q} query heads not a multiple of {n_kv} kv heads")
    h = torch.arange(n_q).repeat_interleave(hd)
    d = torch.arange(hd).repeat(n_q)
    return (h // (n_q // n_kv)) * hd + d


def _layer_groups(L: _Layer, sites, tag):
    gs = []
    a, mlp = L.m.self_attn, L.m.mlp
    hidden = L.m.input_layernorm.weight.numel()
    if "qkv" in sites:
        g = _Group("qkv", f"{L.fqn}:qkv{tag}", hidden)
        g.norms.append((f"{L.fqn}.input_layernorm", L.m.input_layernorm))
        for k in L.qkv_consumers:
            f, lin = L.lin(k)
            g.primary.append((f, lin, _ident(hidden)))
        gs.append(g)
    if "fc1" in sites:
        g = _Group("fc1", f"{L.fqn}:fc1", hidden)
        g.norms.append((f"{L.fqn}.post_attention_layernorm", L.m.post_attention_layernorm))
        for k in ("gate_proj", "up_proj"):
            f, lin = L.lin(k)
            g.primary.append((f, lin, _ident(hidden)))
        gs.append(g)
    if "down" in sites:
        inter = mlp.up_proj.out_features
        g = _Group("down", f"{L.fqn}:down", inter)
        g.rows.append(L.lin("up_proj"))
        f, lin = L.lin("down_proj")
        g.primary.append((f, lin, _ident(inter)))
        gs.append(g)
    return gs


def _o_group(name, members, extra_undo=()):
    """members: _Layers whose V is mixed into every member's o_proj input (tied s)."""
    hd = _head_dim(members[0].m)
    n_kv = members[0].m.self_attn.v_proj.out_features // hd
    for L in members:
        if _head_dim(L.m) != hd or L.m.self_attn.v_proj.out_features != n_kv * hd:
            raise NotImplementedError(f"{name}: tied layers disagree on kv heads / head_dim")
    g = _Group("o", name, n_kv * hd)
    for L in members:
        g.rows.append(L.lin("v_proj"))
        f, o = L.lin("o_proj")
        g.primary.append((f, o, _o_index(o.in_features // hd, n_kv, hd)))
    for f, lin in extra_undo:
        if lin.in_features != g.C:
            raise NotImplementedError(f"{f}: input width {lin.in_features} != V width {g.C}")
        g.undo.append((f, lin, _ident(g.C)))
    return g


def build_groups(model, sites, partial=False):
    """All fold groups of `model` and a record of what was not folded (raises on unsupported)."""
    standalone, smol, pi0 = _discover(model)
    groups, covered, notes = [], set(), []

    for base, stack in standalone:
        for i, layer in enumerate(stack.layers):
            L = _Layer(_fqn(base, f"layers.{i}"), layer)
            covered.add(L.fqn)
            groups += _layer_groups(L, sites, "")
            if "o" in sites:
                groups.append(_o_group(f"{L.fqn}:o", [L]))
        notes.append(f"standalone:{base or '<root>'}")

    for base, m in smol:
        # smolvlm_with_expert.py:396-408 (layer pairing), :436-443 (self vs cross per layer index)
        tm = m.get_vlm_model().text_model
        vbase, ebase = _fqn(base, _rel(m, tm)), _fqn(base, _rel(m, m.lm_expert))
        vl, el = m.get_model_layers([tm, m.lm_expert])
        cross_mode = "cross" in m.attention_mode
        n_every = m.self_attn_every_n_layers
        eidx = {id(l): j for j, l in enumerate(m.lm_expert.layers)}
        for i, (v, e) in enumerate(zip(vl, el)):
            V = _Layer(f"{vbase}.layers.{i}", v)
            covered.add(V.fqn)
            groups += _layer_groups(V, sites, "")
            is_cross = cross_mode and not (n_every > 0 and i % n_every == 0)
            E = None
            if e is not None:
                j = eidx[id(e)]
                hidden_e = e.input_layernorm.weight.numel()
                if is_cross:
                    if e.self_attn.v_proj.in_features == hidden_e and e.self_attn.v_proj.in_features != v.self_attn.v_proj.out_features:
                        raise NotImplementedError(f"layer {i}: cross-attention expected but expert v_proj reads its own hidden state")
                    E = _Layer(f"{ebase}.layers.{j}", e, qkv_consumers=("q_proj",))
                else:
                    E = _Layer(f"{ebase}.layers.{j}", e)
                if E.fqn not in covered:
                    covered.add(E.fqn)
                    groups += _layer_groups(E, sites, "")
            if "o" in sites:
                if e is None:
                    groups.append(_o_group(f"{V.fqn}:o", [V]))
                elif is_cross:
                    # VLM o fold: its V also feeds the expert's v_proj (smolvlm_with_expert.py:365-369)
                    groups.append(_o_group(f"{V.fqn}:o", [V], extra_undo=[E.lin("v_proj")]))
                    groups.append(_o_group(f"{E.fqn}:o", [E]))
                else:
                    # joint attention: V of both is concatenated (smolvlm_with_expert.py:262-264)
                    groups.append(_o_group(f"{V.fqn}+{E.fqn}:o", [V, E]))
        notes.append(f"smolvla:{base or '<root>'}")

    for base, m in pi0:
        # openpi gemma_pytorch.py:157-223: every layer concatenates both experts' Q/K/V
        pm, em = m.paligemma.language_model, m.gemma_expert.model
        pb, eb = _fqn(base, _rel(m, pm)), _fqn(base, _rel(m, em))
        if len(pm.layers) != len(em.layers):
            raise NotImplementedError("Pi0: PaliGemma and expert layer counts differ")
        for i, (p, e) in enumerate(zip(pm.layers, em.layers)):
            P, E = _Layer(f"{pb}.layers.{i}", p), _Layer(f"{eb}.layers.{i}", e)
            covered |= {P.fqn, E.fqn}
            groups += _layer_groups(P, sites, "") + _layer_groups(E, sites, "")
            if "o" in sites:
                groups.append(_o_group(f"{P.fqn}+{E.fqn}:o", [P, E]))
        notes.append(f"pi0:{base or '<root>'}")

    unsupported = []
    for n, mod in model.named_modules():
        if n in covered or not type(mod).__name__.endswith(_BLOCK_SUFFIXES):
            continue
        if any(n.startswith(c + ".") for c in covered):
            continue
        if any(isinstance(x, nn.Linear) for x in mod.modules()):
            unsupported.append(f"{n} ({type(mod).__name__})")
    if unsupported and not partial:
        kinds = sorted({u.split(" (")[1][:-1] for u in unsupported})
        raise NotImplementedError(
            f"smoothquant: folding is not implemented for {len(unsupported)} block(s) of type {kinds} "
            f"(first: {unsupported[0]}); pass partial=true to fold only the supported decoder layers "
            f"and record the rest as not folded")
    warnings = []
    if not groups:
        if not partial:
            raise NotImplementedError("smoothquant: no supported decoder layers found in the model")
        # NEW: partial=true on a model with nothing foldable (e.g. a DiT) is a recorded no-op, so a
        # stack that starts with smoothquant still runs; static=true still scales every Linear.
        warnings.append(f"smoothquant: partial=true and no supported decoder layers; nothing folded "
                        f"({len(unsupported)} unsupported block(s) listed in not_folded_blocks)")
        _warnings.warn(warnings[-1], stacklevel=2)
    return groups, {"structures": notes, "not_folded_blocks": unsupported,
                    "folded_groups": len(groups), "warnings": warnings}


def _rel(parent, child):
    for n, m in parent.named_modules():
        if m is child:
            return n
    raise RuntimeError("submodule not found")


# ---------------------------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------------------------

def _channel_amax(prev, x):
    a = x.detach().to(torch.bfloat16).abs().reshape(-1, x.shape[-1]).amax(0).float()
    return a if prev is None else torch.maximum(prev, a)


def _scatter_max(vals, idx, C):
    out = torch.zeros(C, dtype=torch.float64)
    return out.scatter_reduce(0, idx.cpu(), vals.double().cpu(), "amax", include_self=True)


def _norm_kind(fqn, norm):
    if getattr(norm, "dense", None) is not None:
        raise NotImplementedError(f"{fqn}: adaptive norm ({type(norm).__name__} with dense modulation)")
    t = type(norm).__name__
    if "Gemma" in t and "RMSNorm" in t:
        return "one_plus"
    if t in _PLAIN_RMSNORMS:
        return "plain"
    raise NotImplementedError(f"{fqn}: norm type {t} is not supported for folding")


class SmoothQuant(Algo):
    name = "smoothquant"
    needs_calibration = True

    def __init__(self, **params):
        super().__init__(**params)
        p = self.params
        a = p.get("alpha", 0.5)
        self.alpha = None if str(a).lower() in ("off", "none") else float(a)
        self.po2 = bool(p.get("po2", True))
        self.gain = bool(p.get("gain", True))
        self.gain_target = float(p.get("gain_target", 448.0))
        self.static = bool(p.get("static", False))
        self.sites = tuple(str(p.get("sites", "+".join(SITES))).split("+"))
        bad = [s for s in self.sites if s not in SITES]
        if bad:
            raise ValueError(f"smoothquant: unknown sites {bad}; choose from {SITES}")
        self.gemma_norm = str(p.get("gemma_norm", "fp32"))
        if self.gemma_norm not in ("fp32", "refuse"):
            raise ValueError("smoothquant: gemma_norm must be fp32 or refuse")
        self.partial = bool(p.get("partial", False))
        if not hasattr(self, "_amax"):
            self._amax = None
        self._static_exp = {}

    # -- (f) -----------------------------------------------------------------------------------
    def calibrate(self, model, batches, run):
        self._amax = observe_inputs(model, batches, run,
                                    select=lambda n, m: isinstance(m, nn.Linear), reduce=_channel_amax)
        return {"layers_observed": len(self._amax), "statistic": "per-channel BF16 amax of linear inputs"}

    # -- (d, folded) ---------------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, model, mode):
        if self._amax is None:
            raise RuntimeError("smoothquant: prepare before calibrate")
        groups, rec = build_groups(model, self.sites, self.partial)
        plans = [self._plan(g) for g in groups]           # every factor from the original weights
        promoted, inexact, max_rel = [], Counter(), 0.0
        divisor = {}
        for g, (f, gexp) in zip(groups, plans):
            for fq, norm in g.norms:
                kind = _norm_kind(fq, norm)
                if kind == "one_plus" and norm.weight.dtype != torch.float32:
                    if self.gemma_norm == "refuse":
                        raise NotImplementedError(
                            f"{fq}: (1+w) RMSNorm weight is {norm.weight.dtype}; the fold (1+w)/f - 1 is "
                            "not exact there (gemma_norm=fp32 keeps it in FP32)")
                    norm.weight.data = norm.weight.data.float()
                    promoted.append(fq)
                n_bad, rel = _fold_norm(norm, f, kind)
                inexact[g.site] += n_bad
                max_rel = max(max_rel, rel)
            for fq, lin in g.rows:
                inexact[g.site] += _scale_rows(lin, 1.0 / f)
            for fq, lin, idx in g.primary + g.undo:
                inexact[g.site] += _scale_cols(lin, f[idx.to(f.device)])
                if fq in divisor:
                    raise RuntimeError(f"{fq}: input folded twice")
                divisor[fq] = f[idx.to(f.device)]
        # post-fold calibration maxima: what the static scale (and the gain) were derived from
        self._static_exp = {}
        if self.static:
            import ref_numerics as rn
            # On the CPU in float64 whatever device the model ran on: the calibration maxima follow
            # the model's device, the fold factors may not, and a CUDA run must give the CPU exponents.
            for fq, a in self._amax.items():
                a = a.double().cpu()
                post = (a / divisor[fq].double().cpu()).max() if fq in divisor else a.max()
                self._static_exp[fq] = rn.po2_scale_exp(post.float().reshape(()))
        sites = Counter(g.site for g in groups)
        gains = Counter(int(e) for _, e in plans if e is not None)
        return {**rec, "groups": dict(sites),
                "smoothing": {"alpha": self.alpha, "po2": self.po2, "sites": list(self.sites)},
                "gain": ({"target": self.gain_target, "exp_histogram": dict(sorted(gains.items()))}
                         if self.gain else None),
                "inexact_params": dict(inexact), "norm_fold_max_rel_err": max_rel,
                "norm_weights_promoted_fp32": promoted,
                "static_act_scales": len(self._static_exp) if self.static else 0,
                "mode": mode}

    def _plan(self, g):
        """(factor f[C] float64 on the weights' device, gain exponent or None)."""
        dev = g.primary[0][1].weight.device
        xmax = torch.zeros(g.C, dtype=torch.float64)
        wmax = torch.zeros(g.C, dtype=torch.float64)
        for fq, lin, idx in g.primary:
            if fq not in self._amax:
                raise RuntimeError(f"smoothquant: no calibration statistics for {fq}")
            xmax = torch.maximum(xmax, _scatter_max(self._amax[fq], idx, g.C))
            wmax = torch.maximum(wmax, _scatter_max(lin.weight.detach().abs().amax(0).float(), idx, g.C))
        if self.alpha is None:
            s = torch.ones(g.C, dtype=torch.float64)
        else:
            # Eq. (4). The 1e-5 floor keeps dead channels finite (NEW: not in the paper text)
            s = xmax.clamp_min(1e-5).pow(self.alpha) / wmax.clamp_min(1e-5).pow(1 - self.alpha)
            if self.po2:
                s = torch.exp2(torch.round(torch.log2(s)))
        e = None
        if self.gain:
            post = (xmax / s).max().item()
            e = math.ceil(math.log2(post / self.gain_target)) if post > 0 else 0
            s = s * 2.0 ** e                     # dividing X by 2^e multiplies it by the gain 2^-e
        return s.to(dev), e

    # -- (c) -----------------------------------------------------------------------------------
    def act_scale(self, name, x, kind, dims=None):
        if not self.static or kind != "linear" or name not in self._static_exp:
            return None
        return Scale(self._static_exp[name].to(x.device), "static_per_tensor(smoothquant)", dynamic=False)


def _check(new, want):
    return int((new.double() != want).sum().item())


def _scale_cols(lin, f):
    w = lin.weight.data
    want = w.double() * f.double()[None, :]
    lin.weight.data = want.to(w.dtype)
    return _check(lin.weight.data, want)


def _scale_rows(lin, r):
    w = lin.weight.data
    want = w.double() * r.double()[:, None]
    lin.weight.data = want.to(w.dtype)
    n = _check(lin.weight.data, want)
    if lin.bias is not None:
        bw = lin.bias.data.double() * r.double()
        lin.bias.data = bw.to(lin.bias.dtype)
        n += _check(lin.bias.data, bw)
    return n


def _fold_norm(norm, f, kind):
    """Divide the norm's output channels by f. Returns (inexact params, max rel err of the scale)."""
    w = norm.weight.data
    if kind == "plain":
        want = w.double() / f.double()
        norm.weight.data = want.to(w.dtype)
        return _check(norm.weight.data, want), 0.0
    target = (1.0 + w.double()) / f.double()                 # the multiplier the fold must produce
    norm.weight.data = (target - 1.0).to(w.dtype)
    got = 1.0 + norm.weight.data.double()
    err = ((got - target).abs() / target.abs().clamp_min(1e-30)).max().item()
    return int((got != target).sum().item()), err


ALGO = SmoothQuant
