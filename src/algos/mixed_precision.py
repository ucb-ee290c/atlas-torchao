"""mixed_precision: keep the most FP8-sensitive layers out of single-pass FP8 (docs/algos/mixed-precision.md).

The layers are chosen from a ranking file built OFFLINE on the calibration split by
scripts/rank_layers.py (algos.quantize calls bf16_layers before calibrate, so a ranking computed
inside the run could not select them; see the doc for the proposed base option). Scores:

  fisher_out    NEW: 0.5 * sum_t (dL/dy_t . e_t)^2, e_t the layer's local FP8 output error on the
                clean input (empirical-Fisher / Gauss-Newton estimate of the loss increase; it
                covers the activation cast as well as the weight cast)
  hawq_w        HAWQ-V2 (arXiv 1911.03852) Eq. 2.11, Tr(H_i)/n_i * ||Q(W_i) - W_i||^2, with Tr(H_i)
                replaced by the per-token empirical Fisher trace sum_t ||g_t||^2 ||x_t||^2 (NEW)
  iso_rel_rmse  the reference author's isolated relative RMSE (src/layer_rmse.py, StatsTracker)

Parameters (--algo mixed_precision:k=v,...):
  rank_file=PATH   the ranking JSON (required unless units= is given)
  score=NAME       fisher_out (default) | hawq_w | iso_rel_rmse
  k=N              keep the top N layers, or
  mac_budget=F     keep the top layers whose MACs fit in the share F of the ranked layers' MACs
                   (greedy in score order; layers that do not fit are passed over)
  units=A;B        explicit fqns instead of a ranking (';'-separated)
  keep_as=MODE     bf16 (default): opt out of FP8 (host / VPU work in the chip report; the MXU
                   has no BF16-operand mode) | fp8x3: three FP8 MXU passes x_hi.w_hi + x_hi.w_lo +
                   x_lo.w_hi (residual_passes "wx", after Markidis et al., arXiv 1803.04014,
                   Sec. V) | fp8x2_w: the weight residual only ("w")
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch.nn as nn

from algos.base import Algo

SCORES = ("fisher_out", "hawq_w", "iso_rel_rmse")
KEEP_AS = {"bf16": None, "fp8x3": "wx", "fp8x2_w": "w"}
MEASURED = (nn.Linear, nn.Conv2d)


def load_ranking(path) -> dict:
    p = Path(path)
    raw = p.read_bytes()
    data = json.loads(raw)
    if "layers" not in data:
        raise ValueError(f"{p}: not a ranking file (no 'layers')")
    return {"path": str(p), "sha256": hashlib.sha256(raw).hexdigest(), **data}


def select(layers: dict, score="fisher_out", k=None, mac_budget=None) -> list:
    """The selected fqns, in score order. layers: {fqn: {score: float, "macs": int}}."""
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}, got {score!r}")
    if (k is None) == (mac_budget is None):
        raise ValueError("give exactly one of k= or mac_budget=")
    order = sorted(layers, key=lambda n: (-float(layers[n][score]), n))      # ties: by name
    if k is not None:
        return order[:int(k)]
    total = sum(int(v["macs"]) for v in layers.values())
    out, used = [], 0
    for n in order:
        m = int(layers[n]["macs"])
        if used + m <= float(mac_budget) * total:
            out.append(n)
            used += m
    return out


def check_exact(model, fqns) -> None:
    """Every fqn names a Linear / Conv2d of `model`, and as a skip substring (quant_modes /
    chip_profile match `s in name`) it selects that module only (layers.1. vs layers.11.)."""
    names = dict(model.named_modules())
    for s in fqns:
        if not isinstance(names.get(s), MEASURED):
            raise ValueError(f"{s!r} is not a Linear / Conv2d of the model")
        hits = [n for n in names if s in n and n != s and not n.startswith(s + ".")]
        if hits:
            raise ValueError(f"{s!r} would also skip {hits[:3]} (substring match); refusing")


def macs_of(model, fqns) -> dict:
    """MACs per token / output position (in x out; Conv2d: in*kh*kw x out)."""
    mods = dict(model.named_modules())
    return {n: int(mods[n].weight.numel()) for n in fqns}


class MixedPrecision(Algo):
    name = "mixed_precision"

    def __init__(self, **params):
        super().__init__(**params)
        keep = self.params.get("keep_as", "bf16")
        if keep not in KEEP_AS:
            raise ValueError(f"keep_as must be one of {sorted(KEEP_AS)}, got {keep!r}")
        self.selected: list = []
        self.selection: dict = {}

    def _select(self, model) -> list:
        p = self.params
        units = p.get("units")
        if units is not None:
            fqns, prov = [u for u in str(units).split(";") if u], {"source": "units="}
            total = None
        else:
            if "rank_file" not in p:
                raise ValueError("mixed_precision needs rank_file= (scripts/rank_layers.py) or units=")
            rk = load_ranking(p["rank_file"])
            fqns = select(rk["layers"], p.get("score", "fisher_out"), p.get("k"), p.get("mac_budget"))
            prov = {"source": "rank_file", "path": rk["path"], "sha256": rk["sha256"], "meta": rk.get("meta", {})}
            total = sum(int(v["macs"]) for v in rk["layers"].values())
        check_exact(model, fqns)
        keep = p.get("keep_as", "bf16")
        mods = dict(model.named_modules())
        if KEEP_AS[keep] and any(not isinstance(mods[n], nn.Linear) for n in fqns):
            raise NotImplementedError(f"keep_as={keep} is defined for Linear layers only; "
                                      "select Conv2d units with keep_as=bf16")
        macs = macs_of(model, fqns)
        if total is None:
            total = sum(int(m.weight.numel()) for n, m in model.named_modules()
                        if isinstance(m, MEASURED) and "lm_head" not in n)
        self.selection = {"units": fqns, "keep_as": keep, "score": p.get("score", "fisher_out"),
                          "k": p.get("k"), "mac_budget": p.get("mac_budget"), "ranking": prov,
                          "macs_selected": sum(macs.values()), "macs_ranked_total": total,
                          "mac_share_selected": sum(macs.values()) / total if total else None,
                          "mac_share_unquantized": (sum(macs.values()) / total if total else None)
                          if keep == "bf16" else 0.0,
                          "extra_mxu_passes": {"bf16": 0, "fp8x3": 2, "fp8x2_w": 1}[keep]}
        return fqns

    def bf16_layers(self, model, mode):
        self.selected = self._select(model)
        return list(self.selected) if KEEP_AS[self.params.get("keep_as", "bf16")] is None else []

    def residual_passes(self, name, kind):
        r = KEEP_AS[self.params.get("keep_as", "bf16")]
        if r and name in self.selected:
            if kind != "linear":
                raise NotImplementedError(f"{name}: keep_as residual passes are defined for Linear only")
            return r
        return None

    def record(self) -> dict:
        rec = super().record()
        rec["selection"] = dict(self.selection)
        return rec


ALGO = MixedPrecision


# ---------------------------------------------------------------------------------------------
# Offline scoring (scripts/rank_layers.py)
# ---------------------------------------------------------------------------------------------

def score_layers(clean, quant, batches, loss_fn, names, iso=True) -> dict:
    """{fqn: {macs, fisher_out, hawq_w, fisher_trace_w, dW2, iso_rel_rmse, cum_rel_rmse}} on `batches`.

    clean: the unquantized model; quant: a deep copy quantized under a reference profile (its
    module `fqn` is the FP8 layer); loss_fn(model, batch) -> scalar loss of one batch (summed over
    tokens / elements). Scores are averaged over batches."""
    import torch

    cm, qm = dict(clean.named_modules()), dict(quant.named_modules())
    fo = {n: 0.0 for n in names}
    tw = {n: 0.0 for n in names}

    def mk(n):
        def hook(mod, inp, out):
            x = inp[0].detach()
            with torch.no_grad():
                e = qm[n](x).float() - out.detach().float()          # local FP8 output error
                xn2 = x.float().pow(2).sum(-1)

            def ghook(g):
                g32 = g.float()
                fo[n] += 0.5 * (g32 * e).sum(-1).pow(2).sum().item()
                tw[n] += (g32.pow(2).sum(-1) * xn2).sum().item()
            if out.requires_grad:
                out.register_hook(ghook)
        return hook

    req = {id(p): p.requires_grad for p in clean.parameters()}
    for p in clean.parameters():
        p.requires_grad_(False)
    emb = clean.get_input_embeddings()
    emb.weight.requires_grad_(True)              # activations require grad; weights do not
    handles = [cm[n].register_forward_hook(mk(n)) for n in names]
    try:
        for b in batches:
            loss_fn(clean, b).backward()
            emb.weight.grad = None
    finally:
        for h in handles:
            h.remove()
        for p in clean.parameters():
            p.requires_grad_(req[id(p)])
    nb = max(len(batches), 1)
    res = {}
    for n in names:
        W = cm[n].weight
        dw = (qm[n]._dequant_weight().float() - W.float().reshape(qm[n].weight_codes.shape)).pow(2).sum().item()
        res[n] = {"macs": int(W.numel()), "fisher_out": fo[n] / nb, "fisher_trace_w": tw[n] / nb,
                  "dW2": dw, "hawq_w": tw[n] / nb / W.numel() * dw}
    if iso:
        import layer_rmse

        rm = layer_rmse.LayerRMSE(clean, quant, fqns=names, default_component="LANGUAGE")
        try:
            for b in batches:
                rm.capture(lambda: loss_fn(clean, b, no_grad=True))
                with torch.no_grad():
                    loss_fn(quant, b, no_grad=True)
        finally:
            rm.close()
        acc = {}
        for c in rm.tracker.calls:
            d = acc.setdefault(c["name"], {"iso": [], "cum": []})
            d["iso"].append(c["rmse"] / c["ref_rms"] if c["ref_rms"] > 0 else float("nan"))
            cr = c.get("cumulative_ref_rms", float("nan"))
            d["cum"].append(c.get("cumulative_rmse", float("nan")) / cr if cr and cr > 0 else float("nan"))
        for n in names:
            res[n]["iso_rel_rmse"] = sum(acc[n]["iso"]) / len(acc[n]["iso"])
            res[n]["cum_rel_rmse"] = sum(acc[n]["cum"]) / len(acc[n]["cum"])
    return res
