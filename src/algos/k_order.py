"""k_order: permute the K axis of every Llama MLP down_proj, function-preserving (NEW; docs/algos/k-order.md).

Opt-in only: no recipe includes it. The hardware accumulates K tiles in the order software issues
them (BF16 partial sum carried from tile to tile), so the order of the K channels is a free offline
choice wherever a permutation folds away. For down_proj of a Llama MLP (input act(gate) * up):
rows of gate_proj and up_proj and columns of down_proj get the same permutation pi.

Parameters (--algo k_order:k=v,...):
  rule=asc|desc|rand|none   (required) order of K by channel importance imp_k = mean|x_k| * ||W[:, k]||_2
                            (x = down_proj input over the calibration tokens); asc = least important
                            channel first, desc = most important first, rand = seeded random control,
                            none = identity (the model is not touched)
  seed=N                    rand only (default 0); layer i uses seed + i

It runs on the already-quantized model, in place, after every other member (it registers its rank
in algos.stack.ORDER on import). A permutation of rows / columns of quantized tensors moves codes
and per-row exponents with their rows and changes no value: a row maximum does not depend on its
column order, per-token activation scales (max over K) and per-tensor scales are invariant, so FP8
values, flush and clip are identical and only the accumulation order changes. Combinations that
would not stay exact are refused: per-block scales along K on down_proj (weight or activation) and
online transforms on these layers. Models without the Llama MLP structure raise.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn

from algos.base import Algo

RULES = ("asc", "desc", "rand", "none")
_LINEARS = ("gate_proj", "up_proj", "down_proj")

# run last in a stack: after weight transforms, scale choices and exclusions (algos/stack.py has
# no entry for it; unknown names would otherwise rank before the scale choices)
from algos import stack as _stack  # noqa: E402

_stack.ORDER.setdefault("k_order", 60)
_stack.STAGE.setdefault(60, "permutation")


def perm_hash(perm: torch.Tensor) -> str:
    return hashlib.sha256(perm.detach().cpu().to(torch.int32).contiguous().numpy().tobytes()).hexdigest()[:16]


def find_mlps(model) -> dict:
    """{mlp fqn: (gate, up, down)} of every Llama-style MLP. Raises on any other structure."""
    mlps, down_fqns = {}, []
    for n, m in model.named_modules():
        if n.rsplit(".", 1)[-1] == "down_proj":
            down_fqns.append(n)
        kids = {c: getattr(m, c, None) for c in _LINEARS}
        if all(isinstance(v, nn.Module) and hasattr(v, "weight") for v in kids.values()) and n:
            g, u, d = (kids[c] for c in _LINEARS)
            if not hasattr(m, "act_fn"):
                raise ValueError(f"k_order: {n} has gate/up/down_proj but no act_fn; only the Llama MLP "
                                 "down(act_fn(gate(x)) * up(x)) is supported")
            if not (g.weight.shape == u.weight.shape and d.weight.shape == g.weight.shape[::-1]):
                raise ValueError(f"k_order: {n}: gate {tuple(g.weight.shape)}, up {tuple(u.weight.shape)} "
                                 f"and down {tuple(d.weight.shape)} are not a Llama MLP")
            mlps[n] = (g, u, d)
    covered = {f"{n}.down_proj" for n in mlps}
    stray = [n for n in down_fqns if n not in covered]
    if stray or not mlps:
        raise ValueError("k_order: unsupported structure, expected Llama MLPs (gate_proj, up_proj, "
                         f"down_proj under one module with act_fn); unmatched down_proj: {stray[:3]}, "
                         f"MLPs found: {len(mlps)}")
    return mlps


def _check_down(name, down):
    """Refuse a down_proj whose scales or transforms are not invariant to the K order."""
    if getattr(down, "weight_block", 0):
        raise ValueError(f"k_order: {name}: per-block weight scales along K are not invariant to a K permutation")
    algo = getattr(down, "algo", None)
    for m in (getattr(algo, "members", None) or ([algo] if algo is not None else [])):
        if str(getattr(m, "a", "")).startswith("block"):
            raise ValueError(f"k_order: {name}: per-block activation scales ({m.name}:a={m.a}) are not "
                             "invariant to a K permutation")
        if type(m).online_transform is not Algo.online_transform and m.online_transform(name, "linear") is not None:
            raise ValueError(f"k_order: {name}: online transform of {m.name} acts on the K axis")


def _weight_values(m) -> torch.Tensor:
    return (m._dequant_weight() if hasattr(m, "weight_codes") else m.weight).detach().float()


def _permute_linear(m, perm, dim):
    """Permute rows (dim 0) or columns (dim 1) of a Linear or RefQuantLinear in place: the shared
    weight, the codes, normalized weights and (rows only) exponents and bias follow."""
    n_out = m.weight.shape[0]
    m.weight.data = m.weight.data.index_select(dim, perm.to(m.weight.device)).contiguous()
    for buf in ("weight_codes", "weight_norm", "weight_lo_codes", "weight_lo_norm"):
        b = getattr(m, buf, None)
        if b is not None:
            setattr(m, buf, b.index_select(dim, perm.to(b.device)).contiguous())
    if dim == 0:
        for buf in ("weight_scale_exp", "weight_lo_scale_exp"):
            b = getattr(m, buf, None)
            if b is not None and b.dim() == 2 and b.shape[0] == n_out and n_out > 1:
                setattr(m, buf, b.index_select(0, perm.to(b.device)).contiguous())
        if getattr(m, "bias", None) is not None:
            m.bias.data = m.bias.data.index_select(0, perm.to(m.bias.device)).contiguous()


def order_from_importance(imp: torch.Tensor, rule: str, seed: int = 0) -> torch.Tensor:
    """perm with new[j] = old[perm[j]]; ties broken by channel index (stable)."""
    K = imp.numel()
    if rule == "none":
        return torch.arange(K)
    if rule == "rand":
        return torch.randperm(K, generator=torch.Generator().manual_seed(seed))
    if rule == "asc":
        return torch.argsort(imp.double(), stable=True)
    if rule == "desc":
        return torch.argsort(-imp.double(), stable=True)
    raise ValueError(f"k_order: unknown rule {rule!r}; choose from {RULES}")


class KOrder(Algo):
    name = "k_order"
    needs_calibration = True

    def __init__(self, **params):
        unknown = set(params) - {"rule", "seed"}
        if unknown:
            raise ValueError(f"k_order: unknown parameters {sorted(unknown)}")
        if params.get("rule") not in RULES:
            raise ValueError(f"k_order: rule= is required, one of {RULES}")
        if not isinstance(params.get("seed", 0), int):
            raise ValueError("k_order: seed must be an integer")
        super().__init__(**params)
        self.act_mean = None            # {down_proj fqn: mean |x_k| [K]} from calibrate
        self.applied = None             # record of what apply() did
        self._model = None

    @property
    def rule(self):
        return self.params["rule"]

    # -- model level -------------------------------------------------------------------------
    def calibrate(self, model, batches, run):
        """mean |x_k| of every down_proj input over the calibration tokens (unquantized model)."""
        mlps = find_mlps(model)
        sums, cnt, handles = {}, {}, []

        def hook(mod, args, n):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).double()
            sums[n] = sums.get(n, 0) + x.abs().sum(0)
            cnt[n] = cnt.get(n, 0) + x.shape[0]

        for n in mlps:
            handles.append(mlps[n][2].register_forward_pre_hook(lambda mod, a, n=f"{n}.down_proj": hook(mod, a, n)))
        try:
            for b in batches:
                run(model, b)
        finally:
            for h in handles:
                h.remove()
        self.act_mean = {n: (s / cnt[n]).float() for n, s in sums.items()}
        return {"layers": len(self.act_mean), "tokens": max(cnt.values(), default=0)}

    def prepare(self, model, mode):
        find_mlps(model)                # refuse an unsupported structure before quantizing
        self._model, self.applied = model, None
        return {"deferred": "permutation is applied to the quantized model when the record is built"}

    # -- the permutation ---------------------------------------------------------------------
    def apply(self, model, act_mean=None) -> dict:
        """Permute every MLP of `model` in place (idempotent per call of prepare). act_mean:
        {down_proj fqn: [K]}; without it the importance is the weight column norm alone."""
        act_mean = act_mean if act_mean is not None else self.act_mean
        mlps = find_mlps(model)
        layers, h = {}, hashlib.sha256()
        for i, (n, (g, u, d)) in enumerate(mlps.items()):
            dn = f"{n}.down_proj"
            _check_down(dn, d)
            K = d.weight.shape[1]
            cn = _weight_values(d).norm(dim=0)
            a = act_mean.get(dn) if act_mean else None
            imp = cn if a is None else a.to(cn.device) * cn
            perm = order_from_importance(imp, self.rule, self.params.get("seed", 0) + i)
            if self.rule != "none":
                _permute_linear(g, perm, 0)
                _permute_linear(u, perm, 0)
                _permute_linear(d, perm, 1)
            ph = perm_hash(perm)
            h.update(ph.encode())
            layers[dn] = {"K": K, "hash": ph, "importance": "mean|x|*colnorm" if a is not None else "colnorm"}
        self.applied = {"layers": layers, "n_layers": len(layers), "hash": h.hexdigest()[:16]}
        return self.applied

    def record(self) -> dict:
        if self.applied is None and self._model is not None:
            self.apply(self._model)
        self._model = None
        rec = super().record()
        rec.update(rule=self.rule, seed=self.params.get("seed", 0), applied_permutation=self.applied,
                   importance="mean|x_k| over calibration tokens (own pass on the unquantized model) "
                              "x L2 norm of the weight column (quantized values)")
        return rec


ALGO = KOrder
