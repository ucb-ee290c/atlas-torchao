"""gptq_awq: GPTQ and AWQ weights under the exact Atlas E4M3 cast (docs/algos/gptq-awq.md).

GPTQ (Frantar et al., arXiv 2210.17323): every Linear weight is quantized column by column and each
column's error is fed into the columns not yet quantized through the inverse Hessian of the
layer-wise objective ||WX - W^X||^2, H = 2XX^T (Sec. 3, Eq. 2; Sec. 4 Steps 1-3; Algorithm 1:
lazy blocks of B columns, Cholesky form of H^-1, dampening lambda = damp * mean diag(H)). Eq. 2's
quant() "rounds w to the nearest value on the quantization grid"; here that grid is the Atlas
vpack cast at the layer's per-tensor exponent s0 (ref_numerics.e4m3_vpack: RNE, flush < 2^-6,
saturation 448, exponent 15 finite; a port of pi0-quant fp8_e4m3.py:38-86 and FP8Pack.scala:57-95),
so the error that is fed back is the true FP8 error, never a uniform-integer rounding error.
s0 is the reference rule floor(log2(amax / 256)) (ref_numerics.po2_scale_exp, hw_float_ops.py:38-57)
of the weight GPTQ receives, and it is PINNED: weight_scale returns it as a static Scale, so the
installed codes are exactly the GPTQ codes. (Re-deriving it from the GPTQ weight could pick s0 - 1
when GPTQ lowers amax below 256 * 2^s0, which would saturate codes in [224, 256) * 2^s0 at 448.)
actorder (descending diag(H)) is the "reorder" variant benchmarked as GPTQ-R in arXiv 2306.00978
Sec. 5.1; with a per-tensor scale it breaks no groups (NEW for this setting).

AWQ (Lin et al., arXiv 2306.00978): per input channel s = s_X^alpha, s_X = mean |x| (Sec. 3.2,
Eq. 5), alpha on a `grid`-point grid in [0, 1) (Sec. 5.1: 20), normalized by sqrt(max * min)
(torchao 0.18 prototype/awq/core.py:71-74), minimizing Eq. 4 L(s) = ||Q(W diag(s)) diag(s)^-1 X - WX||.
Q is the reference weight cast with its exponent re-derived on W diag(s). The loss is evaluated
exactly from the calibration Hessian, tr(dW D H D dW^T) with D = diag(1/s) (NEW: no activations
are stored). 1/s is folded into the producer of the group's input (Sec. 3.2: "fused into the
previous operator"), s into the consumers. Fold map (Llama / Gemma decoder layers; NEW):
  input_layernorm            -> self_attn.{q,k,v}_proj
  post_attention_layernorm   -> mlp.{gate,up}_proj     (pre_feedforward_layernorm on Gemma 2/3)
  mlp.up_proj (rows)         -> mlp.down_proj          (down input = act(gate) * up, checked)
Each fold is verified on the calibration data before it is used: the norm's output must reach
exactly the expected Linears (the same tensor), the folded norm must reproduce out / s, and the
MLP must satisfy down_in == act_fn(gate_out) * up_out. A group that fails, and every o_proj (its
producer is attention; under GQA v_proj has fewer rows than o_proj has inputs), is left unscaled
and recorded with the reason.

Parameters (--algo gptq_awq:k=v,...):
  method=gptq|awq|awq_gptq   default gptq
  actorder=true              GPTQ column order by descending diag(H)
  damp=0.01                  GPTQ dampening (Sec. 4 Step 3: 1%)
  block=128                  GPTQ lazy-batch block (Sec. 4 Step 2: B = 128)
  grid=20                    AWQ alpha grid (Sec. 5.1)
  hessians=cpu|device        where the calibration Hessians are kept (NEW, default cpu). Each
                             batch's x^T x is computed on the activation's device and summed into
                             a float32 CPU accumulator; GPTQ / AWQ move one layer's H to the
                             weight's device at a time. With `device` every H stays on the GPU
                             (all layers at once: about 42 GB for Pi0, more than an L40S holds).
Only Linear weights are changed; Conv2d weights and every activation keep the reference rule.

Weight grid under a stack (NEW; docs/algos/stacking.md): stacked after an algorithm that chooses
the weight scale (e.g. granularity:w=channel), `weight_grid` is that algorithm's weight_scale. GPTQ
then quantizes onto ITS grid (a per-channel exponent [out, 1] quantizes each column per row), and
pins that exponent exactly as it pins s0 alone, with the chooser's granularity descriptor; AWQ's
search evaluates Q with the same rule. A per-block grid is refused (NotImplementedError): GPTQ's
column order would have to respect the blocks, which is not implemented here.
"""

from __future__ import annotations

import hashlib
import time
import weakref

import torch
import torch.nn as nn

import ref_numerics as rn
from algos.base import Algo, Scale

METHODS = ("gptq", "awq", "awq_gptq")
# producer suffix -> consumer suffixes (relative to the decoder layer); NEW, see the docstring
NORM_GROUPS = {
    "input_layernorm": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "post_attention_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
    "pre_feedforward_layernorm": ("mlp.gate_proj", "mlp.up_proj"),
}
SUBSAMPLE_TOKENS = 512        # NEW (awq_loss=cast): calibration tokens kept per fold group
FOLD_TOL = 2.0 ** -6          # NEW: relative tolerance of the fold self-check (BF16 gamma rounding)


# ---------------------------------------------------------------------------------------------
# The quantizer and GPTQ
# ---------------------------------------------------------------------------------------------

def qdq(w: torch.Tensor, s) -> torch.Tensor:
    """The Atlas cast at a fixed exponent: e4m3(vpack(w, s)) * 2^s, float32 (exact). `s` is a
    scalar or broadcastable against `w` (a per-row [out, 1] exponent)."""
    s = torch.as_tensor(s, dtype=torch.int32, device=w.device)
    return rn.e4m3_unpack(rn.e4m3_vpack(w, s)) * torch.exp2(s.float())


DAMP_RETRIES = 3              # NEW: on a failed Cholesky, retry with 10x then 100x the dampening


@torch.no_grad()
def gptq(W: torch.Tensor, H: torch.Tensor, s, block=128, damp=0.01, actorder=True, info=None) -> torch.Tensor:
    """GPTQ Algorithm 1 (arXiv 2210.17323) with quant() = qdq(., s). W [out, in], H [in, in], s a
    scalar exponent or one per output row ([out, 1], as GPTQ's own per-row quantizer).
    Returns the dequantized weight (float32); every entry lies on the E4M3 x 2^s grid.

    If the dampened H cannot be factored (rounding in the float32 calibration sum can leave a
    large, badly scaled H slightly indefinite), the factorisation is retried with 10x and then 100x
    the dampening (NEW, as common GPTQ implementations do); `info["damp"]` records the value
    used. The first attempt is exactly the paper's step, so a layer that factors is unchanged."""
    s = torch.as_tensor(s, dtype=torch.int32, device=W.device)
    s = s.reshape(-1) if s.dim() == 2 else s                 # one exponent per entry of a column
    W = W.detach().clone().float()
    H = H.detach().clone().double().to(W.device)
    K = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    eye = torch.eye(K, dtype=H.dtype, device=H.device)
    mean_diag = torch.mean(torch.diag(H))
    Hinv = None
    for attempt in range(DAMP_RETRIES):
        d = damp * 10 ** attempt
        L, err = torch.linalg.cholesky_ex(H + d * mean_diag * eye)
        if int(err) == 0:
            U, err = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
            if int(err) == 0:
                Hinv = U.float()
                break
    if Hinv is None:
        raise RuntimeError(f"gptq: H is not positive-definite even with damp={d} "
                           f"(K={K}, mean diag {float(mean_diag):.3e})")
    if info is not None:
        info["damp"] = d
    del eye
    Q = torch.zeros_like(W)
    for i1 in range(0, K, block):
        i2 = min(i1 + block, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = qdq(w, s)
            Q[:, i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


def awq_search(Ws, H, sx, grid=20, exp_fn=None, producer=None):
    """(s, alpha, loss) minimizing sum over the group of tr(dW D H D dW^T), D = diag(1/s), with
    dW = Q(W diag(s)) - W diag(s) and Q the reference cast (exponent re-derived on W diag(s)), or
    the cast at exp_fn(W diag(s)) when a stacked algorithm chooses the weight grid.

    producer (NEW, gptq_awq:producer_aware=true; default None leaves the loss untouched) is a dict
    {"W": [I, K] producer rows, "H": [K, K] Hessian of the producer's input, "c": [I] row weights,
    "exp_fn": the producer's weight-grid rule} and adds the quantization error of the producer
    rows AFTER the fold, W_p[j] / s_j cast on the producer's own grid (a 32-row panel under
    w=panel), seen through the pair: sum_j c_j e_j H_p e_j^T, e_j = s_j (Q(W_p[j] / s_j) - W_p[j] / s_j).
    A large s_j shrinks row j inside its panel, so this term grows with the spread of s."""
    exp_fn = exp_fn or rn.po2_scale_exp
    H = H.float().to(Ws[0].device)       # [K, K], or a 1-D diagonal (NEW, used by the weights-only proxy)
    if producer is not None:
        p_exp = producer.get("exp_fn") or rn.po2_scale_exp
        Wp = producer["W"]
        Hp = producer["H"]
        Hp = None if Hp is None else Hp.float().to(Wp.device)     # None: identity (weight-only proxy)
        cp = producer["c"].float().to(Wp.device)
    best = (float("inf"), None, None)
    for i in range(grid):
        a = i / grid
        s = sx.float().pow(a).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        loss = 0.0
        for W in Ws:
            Wsc = (W.float() * s).to(W.dtype).float()        # what the folded checkpoint holds
            dW = (qdq(Wsc, exp_fn(Wsc)) - Wsc) / s
            loss += float(((dW @ H) * dW).sum() if H.dim() == 2 else ((dW * dW) * H).sum())   # 1-D H: diagonal
        if producer is not None:
            sp = s.to(Wp.device).unsqueeze(1)
            Wpc = (Wp.float() / sp).to(Wp.dtype).float()     # what the folded producer holds
            e = (qdq(Wpc, p_exp(Wpc)) - Wpc) * sp
            loss += float((((e if Hp is None else e @ Hp) * e).sum(1) * cp).sum())
        if loss < best[0]:
            best = (loss, s, a)
    return best[1], best[2], best[0]


def producer_row_weights(Wp, Hp, Wc, Hc_diag):
    """NEW (producer_aware): c_j for the producer term of awq_search, in the units of the consumer
    loss (squared output error summed over calibration tokens). Row j of the producer feeds the
    consumer input channel j; its error e_j x reaches the consumer output through ||Wc[:, j]||^2 and,
    for an MLP (down input = act(gate) * up), is gated by act(gate_j): c_j = ||Wc[:, j]||^2 * g_j with
    g_j = E[down_in_j^2] / E[up_j^2] = Hc_jj / (Wp[j] Hp Wp[j]^T), the second moment of the pair's
    channel relative to the producer's alone (independence of the gate and of the error assumed;
    no extra calibration pass). Dead channels get 0."""
    Wp = Wp.float()
    up2 = ((Wp if Hp is None else Wp @ Hp.float().to(Wp.device)) * Wp).sum(1)
    g = torch.where(up2 > 0, Hc_diag.float().to(Wp.device) / up2.clamp(min=1e-30), torch.zeros_like(up2))
    return (Wc.float() ** 2).sum(0).to(Wp.device) * g


def act_cast_nc(x: torch.Tensor) -> torch.Tensor:
    """NEW: the stack's run-time per-token activation cast (granularity a=token, rule=nc), dequantized:
    x_hat = dq(vpack(bf16(x), s_token)). x is [T, K]; float32 result. Reuses algos.granularity."""
    from algos.granularity import scale_exp

    xb = x.to(torch.bfloat16)
    exp = scale_exp(xb, (-1,), "nc")
    return rn.dequantize(rn.e4m3_vpack(xb, exp), exp).float()


# the activation cast used by awq_loss=cast and calib=chip (tests replace it, e.g. by the identity)
ACT_CAST = act_cast_nc


@torch.no_grad()
def awq_search_cast(Ws, X, sx, grid=20, exp_fn=None, cast=None, weight_H=None):
    """NEW (awq_loss=cast): (s, alpha, loss, weight_only_alpha) minimizing the layer OUTPUT error through
    the run-time casts, sum over the group's Linears of || W X^T - dq(Q_w(W s)) dq(Q_a(X / s))^T ||^2,
    X [T, K] the calibration-token subsample of the group's shared input, float32 accumulation. Same
    alpha grid and normalization of s as awq_search. If weight_H is given, the weight-only loss of
    awq_search is evaluated too and the alpha it picks is returned (for the report)."""
    exp_fn = exp_fn or rn.po2_scale_exp
    cast = cast or ACT_CAST
    X = X.float()
    refs = [X @ W.float().T for W in Ws]
    best = (float("inf"), None, None)
    wbest = (float("inf"), None)
    for i in range(grid):
        a = i / grid
        s = sx.float().pow(a).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        xh = cast((X / s.to(X.device)).to(torch.bfloat16))          # the producer emits x / s in BF16
        loss, wloss = 0.0, 0.0
        for W, ref in zip(Ws, refs):
            Wsc = (W.float() * s).to(W.dtype).float()
            Wq = qdq(Wsc, exp_fn(Wsc))
            loss += float(((xh @ Wq.T - ref) ** 2).sum())
            if weight_H is not None:
                dW = (Wq - Wsc) / s
                wloss += float(((dW @ weight_H) * dW).sum())
        if loss < best[0]:
            best = (loss, s, a)
        if wloss < wbest[0]:
            wbest = (wloss, a)
    return best[1], best[2], best[0], wbest[1]


def _fingerprint(w: torch.Tensor) -> str:
    t = w.detach().contiguous().cpu().reshape(-1)
    return (hashlib.blake2b(t.view(torch.uint8).numpy().tobytes(), digest_size=16).hexdigest()
            + str(tuple(w.shape)) + str(w.dtype))


# ---------------------------------------------------------------------------------------------
# The algorithm
# ---------------------------------------------------------------------------------------------

class GptqAwq(Algo):
    name = "gptq_awq"
    needs_calibration = True

    def __init__(self, **params):
        super().__init__(**params)
        self.method = str(params.get("method", "gptq"))
        if self.method not in METHODS:
            raise ValueError(f"gptq_awq: method={self.method!r}; choose from {METHODS}")
        unknown = set(params) - {"method", "actorder", "damp", "block", "grid", "hessians", "awq_loss", "calib", "target",
                                 "producer_aware"}
        if unknown:
            raise ValueError(f"gptq_awq: unknown parameters {sorted(unknown)}")
        self.actorder = bool(params.get("actorder", True))
        self.damp = float(params.get("damp", 0.01))
        self.block = int(params.get("block", 128))
        self.grid = int(params.get("grid", 20))
        self.hessians = str(params.get("hessians", "cpu"))
        if self.hessians not in ("cpu", "device"):
            raise ValueError(f"gptq_awq: hessians={self.hessians!r}; choose cpu or device")
        # NEW opt-in (default off): hardware-aware calibration, see docs/algos/gptq-awq.md
        self.awq_loss = str(params.get("awq_loss", "weight"))
        self.calib = str(params.get("calib", "bf16"))
        self.target = str(params.get("target", "bf16"))
        for k, v, ok in (("awq_loss", self.awq_loss, ("weight", "cast")), ("calib", self.calib, ("bf16", "chip")),
                         ("target", self.target, ("bf16", "fp"))):
            if v not in ok:
                raise ValueError(f"gptq_awq: {k}={v!r}; choose from {ok}")
        if self.target == "fp" and self.calib != "chip":
            raise ValueError("gptq_awq: target=fp needs calib=chip")
        # NEW opt-in (default off): the AWQ search also sees the producer's own quantization error
        # on its weight grid (docs/algos/awq-producer-aware.md)
        self.producer_aware = _as_bool(params.get("producer_aware", False))
        if self.producer_aware and self.awq_loss == "cast":
            raise ValueError("gptq_awq: producer_aware=true is implemented for awq_loss=weight only")
        self.subsample = {}                         # owner fqn -> [T, K] CPU token subsample
        self.Hhat, self.Cxx = {}, {}                # calib=chip: sum x_hat x_hat^T, sum x x_hat^T
        self.alpha_picks = {}                       # fold group -> {"weight": a, "cast": a}
        self.timing = {}                            # stage -> seconds (calibration, awq, chip_pass, gptq)
        self._batches = self._run = None
        self._calib_key = None
        self.H, self.sx, self.owner = {}, {}, {}     # owner: fqn -> fqn whose H it shares
        self.fold_checks = {}                       # norm / mlp fqn -> verified consumer info
        self.awq_scale = {}                         # consumer fqn -> s (folded for this mode)
        self.s0 = {}                                # fqn -> pinned exponent
        self._gptq_cache = {}
        self.damp_raised = {}                       # fqn -> dampening a failed Cholesky retried with
        self.layers = {}                            # fqn -> what was done (per mode)
        self.s0_granularity = {}                    # fqn -> descriptor of the pinned exponent
        # weight_grid(name, w, kind) -> Scale | None: the weight-scale choice of an algorithm stacked
        # after this one (algos.stack); None = the reference per-tensor rule
        self.weight_grid = None

    @property
    def uses_gptq(self):
        return self.method in ("gptq", "awq_gptq")

    @property
    def uses_awq(self):
        return self.method in ("awq", "awq_gptq")

    # -- (f) calibration -------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, model, batches, run):
        """H = sum x x^T (float32, on the activation's device) per distinct Linear input, mean |x|
        per channel (AWQ s_X), and the fold self-check data. Runs once per calibration set: the
        eval CLIs hand every mode the same batches, and every mode starts from the same weights."""
        key = (id(batches), len(batches), tuple(n for n, _ in model.named_modules()))
        if key == self._calib_key:
            return {"reused": True, **self._calib_summary()}
        self._calib_key = key
        t_start = time.time()
        self._batches, self._run = batches, run
        self.subsample = {}
        self.Hhat, self.Cxx = {}, {}
        sub_gen = torch.Generator().manual_seed(0)
        quota = -(-SUBSAMPLE_TOKENS // max(1, len(batches)))
        self.H, self.sx, self.owner, self.fold_checks = {}, {}, {}, {}
        self._gptq_cache = {}
        n_tok = {}
        # id(tensor) -> (weakref, fqn) within one batch; an entry counts only while its weakref
        # still resolves to the same object, so two modules share an input only if they received
        # the very same tensor object (a recycled id of a freed tensor never matches)
        seen = {}                                   # Linear inputs -> first (owner) Linear
        norm_out = {}                               # norm outputs -> norm fqn
        norm_in, consumers, mlp_io = {}, {}, {}
        first = {"on": True}
        linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
        norms = {n: m for n, m in model.named_modules()
                 if n.rsplit(".", 1)[-1] in NORM_GROUPS and isinstance(getattr(m, "weight", None), torch.Tensor)
                 and m.weight.dim() == 1}
        handles = []

        def lin_pre(mod, args, n):
            if not args or not torch.is_tensor(args[0]):
                return
            x = args[0]
            k = id(x)
            if first["on"] and _alive(norm_out, k, x):
                consumers.setdefault(norm_out[k][1], set()).add(n)
            if _alive(seen, k, x) and n not in self.H and seen[k][1] != n:
                self.owner[n] = seen[k][1]
                return
            if n in self.owner:
                return
            if not _alive(seen, k, x):
                seen[k] = (weakref.ref(x), n)
            x2 = x.detach().reshape(-1, x.shape[-1]).float()
            if n not in self.H:
                hdev = "cpu" if self.hessians == "cpu" else x2.device
                self.H[n] = torch.zeros(x2.shape[1], x2.shape[1], device=hdev)
                self.sx[n] = torch.zeros(x2.shape[1], device=x2.device)
                n_tok[n] = 0
            if self.H[n].device == x2.device:
                self.H[n].addmm_(x2.T, x2)
            else:
                self.H[n] += (x2.T @ x2).to(self.H[n].device)
            self.sx[n] += x2.abs().sum(0)
            if self.awq_loss == "cast":
                take = min(quota, x2.shape[0])
                idx = torch.randperm(x2.shape[0], generator=sub_gen)[:take].to(x2.device)
                self.subsample.setdefault(n, []).append(x2[idx].cpu())
            n_tok[n] += x2.shape[0]
            if first["on"] and n.endswith("mlp.down_proj"):
                mlp_io.setdefault(n[: -len(".down_proj")], {})["down_in"] = x.detach()[..., :]

        def lin_out(mod, args, out, n):
            if first["on"] and n.endswith(("mlp.gate_proj", "mlp.up_proj")) and torch.is_tensor(out):
                mlp_io.setdefault(n.rsplit(".", 1)[0], {})[n.rsplit(".", 1)[1]] = out.detach()

        def norm_hook(mod, args, out, n):
            if torch.is_tensor(out):
                norm_out[id(out)] = (weakref.ref(out), n)
                if first["on"] and n not in norm_in and args and torch.is_tensor(args[0]):
                    x = args[0].detach()
                    norm_in[n] = (x.reshape(-1, x.shape[-1])[:16].clone(), out.reshape(-1, out.shape[-1])[:16].clone())

        for n, m in linears.items():
            handles.append(m.register_forward_pre_hook(lambda mod, a, n=n: lin_pre(mod, a, n)))
            handles.append(m.register_forward_hook(lambda mod, a, o, n=n: lin_out(mod, a, o, n)))
        for n, m in norms.items():
            handles.append(m.register_forward_hook(lambda mod, a, o, n=n: norm_hook(mod, a, o, n)))
        try:
            for i, b in enumerate(batches):
                seen.clear()
                norm_out.clear()
                run(model, b)
                if i == 0:
                    self._check_mlps(model, mlp_io)
                    mlp_io.clear()
                first["on"] = False
        finally:
            for h in handles:
                h.remove()
        for n in self.sx:
            self.sx[n] /= max(n_tok[n], 1)
        self.subsample = {n: torch.cat(v)[:SUBSAMPLE_TOKENS] for n, v in self.subsample.items()}
        self.timing["calibration"] = time.time() - t_start
        self.n_tok = n_tok
        self._check_norms(model, norms, norm_in, consumers)
        return self._calib_summary()

    def _calib_summary(self):
        under = sorted(n for n, t in getattr(self, "n_tok", {}).items() if t < self.H[n].shape[0])
        return {"linears_with_hessian": len(self.H), "shared_inputs": len(self.owner),
                "hessians": self.hessians,
                # fewer calibration tokens than input channels: H is singular and only the
                # dampening makes it invertible; GPTQ then fits the calibration set (NEW warning)
                "rank_deficient_hessians": under,
                "foldable_groups": sorted(k for k, v in self.fold_checks.items() if v["ok"]),
                "refused_groups": {k: v["reason"] for k, v in self.fold_checks.items() if not v["ok"]}}

    def _check_mlps(self, model, mlp_io):
        mods = dict(model.named_modules())
        for mlp_name, io in mlp_io.items():
            key = mlp_name + ".up_proj->down_proj"
            mlp = mods.get(mlp_name)
            act = getattr(mlp, "act_fn", None)
            if not {"gate_proj", "up_proj", "down_in"} <= set(io) or act is None:
                self.fold_checks[key] = {"ok": False, "reason": "MLP is not act_fn(gate) * up -> down"}
                continue
            ref = (act(io["gate_proj"].float()) * io["up_proj"].float())
            got = io["down_in"].float()
            ok = ref.shape == got.shape and float((ref - got).abs().max()) <= FOLD_TOL * float(got.abs().max() + 1e-30)
            self.fold_checks[key] = {"ok": ok, "producer": mlp_name + ".up_proj", "rows": True,
                                     "consumers": [mlp_name + ".down_proj"],
                                     "reason": None if ok else "down_proj input != act_fn(gate) * up"}

    @torch.no_grad()
    def _check_norms(self, model, norms, norm_in, consumers):
        for n, m in norms.items():
            layer = n.rsplit(".", 1)[0]
            want = {f"{layer}.{c}" for c in NORM_GROUPS[n.rsplit(".", 1)[-1]]}
            got = consumers.get(n, set())
            if not got:
                continue                              # feeds no Linear (e.g. Gemma 2 post-attention norm)
            if got != want:
                self.fold_checks[n] = {"ok": False, "reason": f"output reaches {sorted(got)}, expected {sorted(want)}"}
                continue
            form = self._norm_form(m, *norm_in.get(n, (None, None)))
            self.fold_checks[n] = {"ok": form is not None, "producer": n, "rows": False, "form": form,
                                   "consumers": sorted(want),
                                   "reason": None if form else "folded norm does not reproduce out / s"}

    @staticmethod
    def _norm_form(m, x, y):
        """'scale' (out = n(x) * w) or 'offset' (Gemma: out = n(x) * (1 + w)), verified on calibration
        input with a random s; None if neither reproduces out / s."""
        if x is None:
            return None
        g = torch.Generator().manual_seed(0)
        s = (torch.rand(m.weight.shape, generator=g) + 0.5).to(m.weight.device)
        w0 = m.weight.detach().clone()
        try:
            for form in ("scale", "offset"):
                m.weight.copy_(_fold_norm_weight(w0, s, form))
                y1 = m(x).float()
                ref = y.float() / s
                if float((y1 - ref).abs().max()) <= FOLD_TOL * float(ref.abs().max() + 1e-30):
                    return form
            return None
        except Exception:                                  # noqa: BLE001 (e.g. a norm needing more inputs)
            return None
        finally:
            m.weight.copy_(w0)

    # -- (d) AWQ folds ---------------------------------------------------------------------------
    @torch.no_grad()
    def prepare(self, model, mode):
        self.awq_scale = {}
        self.layers = {}
        self.alpha_picks = {}
        if not self.uses_awq:
            self._chip_pass(model)
            return None
        mods = dict(model.named_modules())
        folded, refused = [], {}
        # norm groups first, then the up_proj -> down_proj folds (the order the screen used)
        for key, chk in sorted(self.fold_checks.items(), key=lambda kv: (kv[1].get("rows", False), kv[0])):
            if not chk["ok"]:
                refused[key] = chk["reason"]
                continue
            cons = chk["consumers"]
            H = self.H.get(self.owner.get(cons[0], cons[0]))
            sx = self.sx.get(self.owner.get(cons[0], cons[0]))
            if H is None:
                refused[key] = "no calibration statistics"
                continue
            Ws = [mods[c].weight.detach() for c in cons]
            t_awq = time.time()
            own = self.owner.get(cons[0], cons[0])
            if self.awq_loss == "cast" and own in self.subsample:
                s, alpha, _, a_w = awq_search_cast(Ws, self.subsample[own].to(Ws[0].device), sx, self.grid,
                                                   self._grid_exp_fn(cons[0]), weight_H=H.float().to(Ws[0].device))
                self.alpha_picks[key] = {"weight": a_w, "cast": alpha}
            else:
                producer = self._producer_term(mods, chk, cons, Ws, H) if self.producer_aware else None
                extra = {} if producer is None else {"producer": producer}      # default call unchanged
                s, alpha, _ = awq_search(Ws, H, sx, self.grid, self._grid_exp_fn(cons[0]), **extra)
                self.alpha_picks[key] = {"weight": alpha}
            self.timing["awq"] = self.timing.get("awq", 0.0) + time.time() - t_awq
            prod = mods[chk["producer"]]
            if chk["rows"]:
                prod.weight.copy_((prod.weight.float() / s.unsqueeze(1).to(prod.weight.device)).to(prod.weight.dtype))
                if getattr(prod, "bias", None) is not None:
                    prod.bias.copy_((prod.bias.float() / s.to(prod.bias.device)).to(prod.bias.dtype))
            else:
                prod.weight.copy_(_fold_norm_weight(prod.weight.detach(), s, chk["form"]))
            for c in cons:
                w = mods[c].weight
                w.copy_((w.float() * s.to(w.device)).to(w.dtype))
                self.awq_scale[c] = s
            folded.append({"group": key, "alpha": alpha})
        for n, m in mods.items():
            if n.endswith("self_attn.o_proj") and isinstance(m, nn.Linear):
                attn = mods[n.rsplit(".", 1)[0]]
                cfg = getattr(attn, "config", None)
                gqa = cfg is not None and getattr(cfg, "num_key_value_heads", None) not in (
                    None, getattr(cfg, "num_attention_heads", None))
                refused[n] = ("unscaled: producer is attention; under GQA v_proj has fewer output rows "
                              "than o_proj has inputs" if gqa else "unscaled: v_proj -> o_proj fold not implemented")
        self._chip_pass(model)
        return {"method": self.method, "awq_folded": folded, "awq_unscaled": refused,
                "alpha_picks": dict(self.alpha_picks)}

    def _producer_term(self, mods, chk, cons, Ws, Hc):
        """NEW (producer_aware): the producer-row term of awq_search for an up_proj -> down_proj
        group; None for a norm group (its producer is a 1-D weight, cast to BF16, not an FP8 operand)."""
        if not chk["rows"]:
            return None
        pname = chk["producer"]
        own_p = self.owner.get(pname, pname)
        Hp = self.H.get(own_p)
        if Hp is None:
            return None
        Wp = mods[pname].weight.detach()
        c = producer_row_weights(Wp, Hp, Ws[0], torch.diag(Hc.float()))
        return {"W": Wp, "H": Hp, "c": c, "exp_fn": self._grid_exp_fn(pname)}

    @torch.no_grad()
    def _chip_pass(self, model):
        """NEW (calib=chip): second calibration pass on the (folded) model: every Linear input that owns
        a Hessian is cast like the MXU operand, x_hat = dq(Q_a(x)); H_hat = sum x_hat x_hat^T and, for
        target=fp, C = sum x x_hat^T (x: the folded model's own input). Statistics come from this
        model's calibration batches."""
        self.Hhat, self.Cxx = {}, {}
        if self.calib != "chip" or self._batches is None:
            return
        t0 = time.time()
        linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear) and n in self.H}
        handles = []

        def pre(mod, args, n):
            if not args or not torch.is_tensor(args[0]):
                return
            x2 = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            xh = ACT_CAST(x2)
            if n not in self.Hhat:
                hdev = "cpu" if self.hessians == "cpu" else x2.device
                self.Hhat[n] = torch.zeros(x2.shape[1], x2.shape[1], device=hdev)
                if self.target == "fp":
                    self.Cxx[n] = torch.zeros(x2.shape[1], x2.shape[1], device=hdev)
            dev = self.Hhat[n].device
            self.Hhat[n] += (xh.T @ xh).to(dev)
            if self.target == "fp":
                self.Cxx[n] += (x2.T @ xh).to(dev)

        for n, m in linears.items():
            handles.append(m.register_forward_pre_hook(lambda mod, a, n=n: pre(mod, a, n)))
        try:
            for b in self._batches:
                self._run(model, b)
        finally:
            for h in handles:
                h.remove()
        self.timing["chip_pass"] = time.time() - t0

    # -- (a, b) weights ----------------------------------------------------------------------------
    def _grid(self, name, w, kind):
        """(exponent, descriptor) of the grid `w` is quantized onto: the stacked chooser's, else
        the reference per-tensor rule."""
        sc = self.weight_grid(name, w, kind) if self.weight_grid is not None else None
        if sc is None:
            return rn.po2_scale_exp(w), "per_tensor"
        if sc.block:
            raise NotImplementedError(f"{name}: gptq_awq on a per-block weight grid ({sc.describe()}) "
                                      "is not implemented")
        return sc.exp.to(torch.int32), sc.granularity

    def _grid_exp_fn(self, name):
        if self.weight_grid is None:
            return None
        return lambda W: self._grid(name, W, "linear")[0].to(W.device)

    def transform_weight(self, name, w, kind):
        if kind != "linear":
            self.layers[name] = f"{kind}: reference cast (GPTQ applies to Linear only)"
            return w
        self.s0.pop(name, None)
        self.s0_granularity.pop(name, None)
        if not self.uses_gptq:
            self.layers[name] = "awq" if name in self.awq_scale else "rtn"
            return w
        own = self.owner.get(name, name)
        H = self.H.get(own)
        if H is None:
            self.layers[name] = "rtn: not reached by the calibration data"
            return w
        D = self.awq_scale.get(name)
        chip = self.calib == "chip" and own in self.Hhat
        fp = chip and self.target == "fp" and own in self.Cxx
        if fp:      # NEW target=fp: W' = W C H_hat^-1 minimizes ||W x - W' x_hat||^2 (damped as GPTQ)
            Hh = self.Hhat[own].double()
            Hd = Hh + self.damp * torch.mean(torch.diag(Hh)) * torch.eye(Hh.shape[0], dtype=Hh.dtype)
            WC = w.double().cpu() @ self.Cxx[own].double()
            w = torch.linalg.solve(Hd, WC.T).T.to(device=w.device, dtype=w.dtype)
        s0, gran = self._grid(name, w, kind)
        key = (name, self.method, self.actorder, self.damp, self.block, _fingerprint(w),
               _fingerprint(s0.contiguous()), chip, fp)
        if key not in self._gptq_cache:
            t_gptq = time.time()
            dev = w.device                  # one layer's H moves to the weight's device (gptq())
            if chip:                         # H_hat is measured on the folded model: no 1/s rescale
                Hn = self.Hhat[own]
            elif D is not None:              # rescaled in float64: AWQ's 1/s spans many binades
                Di = (1.0 / D.double()).to(H.device)
                Hn = H.double() * Di.unsqueeze(0) * Di.unsqueeze(1)
            else:
                Hn = H
            ginfo = {}
            wq = gptq(w, Hn, s0.to(dev), self.block, self.damp, self.actorder, info=ginfo)
            if ginfo.get("damp", self.damp) != self.damp:
                self.damp_raised[name] = ginfo["damp"]
            self._gptq_cache[key] = wq.to(device=w.device, dtype=w.dtype)
            self.timing["gptq"] = self.timing.get("gptq", 0.0) + time.time() - t_gptq
        out = self._gptq_cache[key]
        if not torch.equal(qdq(out, s0), out.float()):
            raise RuntimeError(f"{name}: GPTQ weight is not representable at the pinned exponent")
        self.s0[name] = s0                  # pinned: weight_scale hands it back as a static Scale
        self.s0_granularity[name] = gran
        self.layers[name] = "gptq" + ("+awq" if D is not None else "")
        return out

    def weight_scale(self, name, w, kind):
        if kind != "linear" or name not in self.s0:
            return None
        return Scale(self.s0[name], self.s0_granularity.get(name, "per_tensor"), dynamic=False)

    def record(self) -> dict:
        rec = super().record()
        counts = {}
        for v in self.layers.values():
            counts[v] = counts.get(v, 0) + 1
        rec["weights"] = counts
        rec["damp_raised"] = dict(self.damp_raised)
        if self.producer_aware:
            rec["producer_aware"] = True
        rec["hw_aware"] = {"awq_loss": self.awq_loss, "calib": self.calib, "target": self.target,
                           "alpha_picks": dict(self.alpha_picks),
                           "timing_s": {k: round(v, 2) for k, v in self.timing.items()}}
        return rec


def _as_bool(v) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on") if isinstance(v, str) else bool(v)


def _alive(table, k, x) -> bool:
    return k in table and table[k][0]() is x


def _fold_norm_weight(w, s, form):
    s = s.to(w.device)
    if form == "scale":
        return (w.float() / s).to(w.dtype)
    return ((1.0 + w.float()) / s - 1.0).to(w.dtype)


ALGO = GptqAwq
