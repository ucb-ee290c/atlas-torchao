"""rotation: Hadamard rotations of matmul inputs (QuaRot / SpinQuant style) for Atlas FP8.

Sources (all opened):
  QuaRot, arXiv 2404.00456: Sec. 3.4 / Sec. 4 stage 1a (fold the RMSNorm scale into the next
    linears, rotate the residual stream x -> xQ with a randomized Hadamard; valid "so long as no
    re-scaling happens in the RMSNorm block"), stage 1b (online Hadamard before down_proj), stage
    1c Eq. 8-10 (per-head H_dh folded offline into W_v / W_out, online cross-head H_nh before
    o_proj), Sec. 3.1 (non-power-of-two sizes as H_2^n (x) H_m with a known H_m).
  SpinQuant, arXiv 2405.16406: Sec. 3.1-3.2 (R1 residual and R2 per-head rotations are mergeable;
    R4 is a fixed online Hadamard before down_proj). Learning R1/R2 (Cayley SGD) is not done here.
  Reference numerics: the operand cast is the reference vpack / po2 rule through algos.base
    (ref_numerics.py, a port of pi0-quant hw_float_ops.py:38-57,64-103).

What it does (parameters as `--algo rotation:k=v,...`):
  online=down,o   layers whose input gets an online Hadamard: down -> *.down_proj, o -> *.o_proj
                  (comma list, or "none"). (d, online) + its weight-side fold in transform_weight.
  impl=fp32       how the online transform runs:
                    fp32  orthonormal H in float32, then BF16 (the upper bound: QuaRot computes it
                          "in full precision"; no Atlas unit does this, counted as host work)
                    vpu   the Atlas VPU: +-1 butterflies across 32-wide feature tiles only (there
                          is no intra-register permute), every add/sub an FP32 add truncated to
                          its upper 16 bits (vadd/vsub: chip_ops.py header, AddSubSumVec.scala:133).
                          NEW
                    mxu2  an extra MXU matmul against +-1 (exact in E4M3). MXU operands are FP8,
                          so the BF16 input is split into two FP8 operands, hi = vpack(x, s) and
                          lo = vpack(x - hi, s - 4), accumulated in one accumulator (vmatmul.acc
                          against H and an exact +-2^-4 copy of H). NEW
                  For vpu / mxu2 the +-1 transform is followed by 2^-k (k = floor(log2 sqrt(m)),
                  m the mixing size): on the chip that power of two is part of the E8M0 scale of
                  the following vpack / vmatpop.fp8 (free). The non-power-of-two remainder
                  c = sqrt(m) / 2^k is folded into the next weight offline (W -> W M^T / c, before
                  its FP8 cast). 1/sqrt(m) itself is not used as an operand: it is not a power of
                  two for odd log2 m, and for m = 5632 it is below the E4M3 normal floor 2^-6.
  fold_head=true  fold the per-head H_hd into v_proj (output rows) and o_proj (input columns)
                  (QuaRot stage 1c; SpinQuant R2). Lossless; done in `prepare`.
  r1=false        also fold the RMSNorm weights and rotate the residual stream by Q = D H (QuaRot
                  stage 1a; SpinQuant R1). Llama only; refused (RotationRefused) where the norm
                  cannot be folded: LayerNorm, adaptive RMSNorm (openpi GemmaRMSNorm with
                  cond_dim), adaLN modulation, or any non-Llama architecture.
  seed=0          R1 signs: 0 = plain Sylvester H, else Q = D H with random signs D (input side;
                  an output-side H D only flips signs and cannot change a sign-symmetric cast).
  static=false    true: one static po2 scale per quantized Linear input, floor(log2(amax/256))
                  (the reference rule, ref_numerics.po2_scale_exp) on the calibration amax. The
                  amax of an online layer is taken AFTER its transform (calibrate runs `prepare`
                  first and observes the transformed input). mxu2 also gets a static scale for
                  its pre-transform split.
  vpu_max=32      largest butterfly size for impl=vpu.

Every quantized online layer records its estimated extra work (per token and per 32 tokens) under
`online_cost` in the algo record, and under the chip profile the transform's work is added to the
coverage report (MXU flops, VPU element ops, or host work for fp32).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from algos.base import Algo, Scale, _guard, cast_with, dequant

TILE = 32                     # MXU / VPU tile edge (green card: 32 x 32 BF16 accumulator, m pairs)
_LO_SHIFT = 4                 # lo operand at scale 2^(s - 4): the E4M3 mantissa step is 2^-3


class RotationRefused(ValueError):
    """The requested rotation cannot be folded into this model."""


# ------------------------------------------------------------------------------------------------
# Hadamard matrices (+-1 entries, float64)
# ------------------------------------------------------------------------------------------------

def sylvester(n: int) -> torch.Tensor:
    if n < 1 or n & (n - 1):
        raise ValueError(f"Sylvester Hadamard needs a power of two, got {n}")
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def _is_prime(q: int) -> bool:
    return q > 1 and all(q % p for p in range(2, int(q ** 0.5) + 1))


def paley1(q: int) -> torch.Tensor:
    """Paley construction I: a Hadamard matrix of order q + 1 for a prime q = 3 (mod 4)."""
    if not (_is_prime(q) and q % 4 == 3):
        raise ValueError(f"Paley I needs a prime q = 3 mod 4, got {q}")
    sq = {(i * i) % q for i in range(1, q)}
    chi = [0] + [1 if a in sq else -1 for a in range(1, q)]
    Q = torch.tensor([[chi[(j - i) % q] for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Q
    return torch.eye(q + 1, dtype=torch.float64) + S


def hadamard_factors(n: int) -> tuple[list[int], bool]:
    """Kronecker factor sizes of a Hadamard transform over n: ([m, 2^a] or [2^a], full).

    n = 2^a * m with m = 1 or a Paley-I order (q + 1, q prime = 3 mod 4) gives a full Hadamard
    (QuaRot Sec. 3.1). Otherwise only the power-of-two part mixes: [r, 2^a] with r an identity
    factor (full = False)."""
    a = (n & -n).bit_length() - 1
    for k in range(a, -1, -1):
        m = n >> k
        if m == 1:
            return [1 << k], True
        if _is_prime(m - 1) and (m - 1) % 4 == 3:
            return ([m, 1 << k] if k else [m]), True
    return [n >> a, 1 << a], False


def pm1(m: int) -> torch.Tensor:
    return sylvester(m) if m & (m - 1) == 0 else paley1(m - 1)


class Kron:
    """x[..., n] -> (A_1 (x) A_2 (x) ...) x along the last dim, without forming the dense matrix.
    `mats[i]` is an (m_i x m_i) float64 matrix or None (identity)."""

    def __init__(self, shape, mats):
        self.shape = tuple(int(s) for s in shape)
        self.mats = list(mats)
        self.n = math.prod(self.shape)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        y = x.reshape(*lead, *self.shape)
        for i, m in enumerate(self.mats):
            if m is None:
                continue
            ax = len(lead) + i
            y = torch.movedim(torch.movedim(y, ax, -1) @ m.to(device=y.device, dtype=y.dtype).T, -1, ax)
        return y.reshape(*lead, self.n)

    def scaled(self, f) -> "Kron":
        return Kron(self.shape, [None if m is None else f(m) for m in self.mats])


def bf16_trunc(v: torch.Tensor) -> torch.Tensor:
    """FP32 -> BF16 keeping the upper 16 bits: the result of vadd.bf16 / vsub.bf16."""
    return (v.float().contiguous().view(torch.int32) & -65536).view(torch.float32)


def vpu_butterfly(x: torch.Tensor, shape, axis: int) -> torch.Tensor:
    """Unnormalized +-1 Walsh-Hadamard butterfly along `axis` of x.reshape(shape) as the VPU runs
    it: per stage a + b and a - b, each an FP32 add of BF16 operands truncated to BF16."""
    lead = x.shape[:-1]
    y = x.to(torch.bfloat16).float().reshape(*lead, *shape)
    ax = len(lead) + axis
    y = torch.movedim(y, ax, -1).contiguous()
    n = y.shape[-1]
    h = 1
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a, b = y[..., 0, :], y[..., 1, :]
        y = torch.stack([bf16_trunc(a + b), bf16_trunc(a - b)], dim=-2).reshape(*y.shape[:-3], n)
        h *= 2
    return torch.movedim(y, -1, ax).reshape(*lead, -1)


# ------------------------------------------------------------------------------------------------
# Per-layer online transform specs
# ------------------------------------------------------------------------------------------------

class OnlineSpec:
    """The online transform of one layer: u = c * M x with M orthogonal, and its weight fold
    W -> W M^T / c (so W x = W' u exactly).

    impl fp32: u = M x in float32, c = 1.
    impl vpu:  u = 2^-k * H+-1 x on the butterfly axis (the rest identity), c = sqrt(P) / 2^k.
    impl mxu2: u = 2^-k * H+-1 (hi + lo), c = sqrt(m) / 2^k with m the mixing size."""

    def __init__(self, name, n, impl, shape, mats_pm1, mix_axis=None, full=True, note=""):
        self.name, self.n, self.impl, self.full, self.note = name, n, impl, full, note
        self.kron_pm1 = Kron(shape, mats_pm1)
        self.mix = math.prod(m.shape[0] for m in mats_pm1 if m is not None)
        self.kron_orth = self.kron_pm1.scaled(lambda m: m / math.sqrt(m.shape[0]))
        self.mix_axis = mix_axis
        if impl == "fp32":
            self.k, self.c = 0, 1.0
        else:
            self.k = int(math.floor(math.log2(math.sqrt(self.mix))))
            self.c = math.sqrt(self.mix) / 2 ** self.k

    def fold_weight(self, w: torch.Tensor) -> torch.Tensor:
        """W M^T / c, computed in float64 (rows of W transformed by M)."""
        return (self.kron_orth.apply(w.double()) / self.c).to(w.dtype)

    def cost(self, out_features: int) -> dict:
        """Estimated extra work per token and per 32 tokens (one 32-row block), NEW."""
        n = self.n
        layer_tiles = math.ceil(n / TILE) * math.ceil(out_features / TILE)
        rec = {"impl": self.impl, "mixing_size": self.mix, "full_hadamard": self.full,
               "factors": [m.shape[0] for m in self.kron_pm1.mats if m is not None],
               "layer_mxu_tile_matmuls_per_32_tokens": layer_tiles}
        if self.impl == "vpu":
            stages = int(math.log2(self.mix))
            rec.update(vpu_element_ops_per_token=n * stages,
                       vpu_instructions_per_32_tokens=(n // TILE) * stages,   # one per 32x32 BF16 tile
                       scale_po2_into_vpack=self.k, weight_fold_c=self.c)
        elif self.impl == "mxu2":
            sizes = rec["factors"]
            macs = sum(n * m for m in sizes)
            tiles = sum((n // m) * math.ceil(m / TILE) ** 2 for m in sizes)
            rec.update(mxu_macs_per_token=2 * macs, mxu_tile_matmuls_per_32_tokens=2 * tiles,
                       mxu_extra_vs_layer=2 * tiles / layer_tiles, relayouts=max(len(sizes) - 1, 0),
                       vpu_element_ops_per_token=2 * n,          # vunpack(hi) + vsub for lo
                       extra_vpack_per_token=n, scale_po2_into_vmatpop=self.k, weight_fold_c=self.c)
        else:
            rec.update(host_flops_per_token=2 * sum(n * m for m in rec["factors"]),
                       note="ideal float32 transform; no Atlas unit runs it (upper bound)")
        return rec


def _down_spec(name, n, impl, vpu_max):
    if impl == "vpu":
        tiles = n // TILE
        if n % TILE or tiles < 2:
            return None, f"{name}: {n} features are not 2+ tiles of {TILE}; no VPU transform"
        P = min(tiles & -tiles, vpu_max)
        if P < 2:
            return None, f"{name}: {tiles} tiles have no power-of-two factor; no VPU transform"
        r = tiles // P
        return OnlineSpec(name, n, impl, (r, P, TILE), [None, sylvester(P), None], mix_axis=1,
                          full=False), ""
    sizes, full = hadamard_factors(n)
    if not full:
        mats = [None, sylvester(sizes[1])]
        note = f"{name}: {n} is not 2^a * (Paley order); mixing only {sizes[1]}"
    else:
        mats, note = [pm1(m) for m in sizes], ""
    return OnlineSpec(name, n, impl, sizes, mats, full=full, note=note), note


def _o_spec(name, n, head_dim, impl, vpu_max):
    heads = n // head_dim
    if impl == "vpu":
        q = head_dim // TILE if head_dim % TILE == 0 else 0
        if q:
            P = min(heads & -heads, vpu_max)
            r = heads // P
            if P < 2:
                return None, f"{name}: {heads} heads have no power-of-two factor; no VPU transform"
            return OnlineSpec(name, n, impl, (r, P, head_dim), [None, sylvester(P), None], mix_axis=1,
                              full=False), ""
        return _down_spec(name, n, impl, vpu_max)       # heads narrower than a tile: tile groups
    sizes, full = hadamard_factors(heads)
    if heads < 2:
        return None, f"{name}: one head; nothing to mix online"
    if not full:
        mats = [None, sylvester(sizes[1])]
        note = f"{name}: {heads} heads are not 2^a * (Paley order); mixing only {sizes[1]}"
    else:
        mats, note = [pm1(m) for m in sizes], ""
    return OnlineSpec(name, n, impl, (*sizes, head_dim), [*mats, None], full=full, note=note), note


# ------------------------------------------------------------------------------------------------
# Offline folds (prepare)
# ------------------------------------------------------------------------------------------------

def _attention_modules(model):
    """(fqn, module, head_dim) of every attention block with v_proj / o_proj Linear children."""
    out = []
    for n, m in model.named_modules():
        v, o = getattr(m, "v_proj", None), getattr(m, "o_proj", None)
        if isinstance(v, nn.Linear) and isinstance(o, nn.Linear):
            hd = getattr(m, "head_dim", None)
            if hd is None and hasattr(getattr(model, "config", None), "num_attention_heads"):
                hd = o.in_features // model.config.num_attention_heads
            out.append((n, m, int(hd) if hd else None))
    return out


@torch.no_grad()
def fold_head_hadamard(model) -> dict:
    """QuaRot stage 1c (Eq. 8-9) / SpinQuant R2: v_proj rows <- H_hd per head, o_proj columns
    <- H_hd^T per head, with one H_hd for the whole model (attention may mix values of several
    v_proj modules, e.g. an action expert, so they must all use the same rotation)."""
    atts = _attention_modules(model)
    hds = {hd for _, _, hd in atts}
    if not atts:
        return {"folded": 0, "skipped": "no attention module with v_proj / o_proj"}
    if len(hds) != 1 or None in hds:
        return {"folded": 0, "skipped": f"head dims {sorted(map(str, hds))} differ or are unknown"}
    hd = hds.pop()
    sizes, full = hadamard_factors(hd)
    if not full:
        return {"folded": 0, "skipped": f"head_dim {hd} has no Hadamard matrix here"}
    H = Kron(sizes, [pm1(m) / math.sqrt(m) for m in sizes])
    for _, m, _ in atts:
        v, o = m.v_proj, m.o_proj
        if v.out_features % hd or o.in_features % hd:
            raise RotationRefused(f"v_proj / o_proj widths {v.out_features} / {o.in_features} are not "
                                  f"multiples of head_dim {hd}")
        # v' = H v per head: rows of W_v (and the bias) grouped by head
        Wv = v.weight.double().reshape(-1, hd, v.in_features).transpose(1, 2)        # [kv, in, hd]
        v.weight.data = H.apply(Wv).transpose(1, 2).reshape(v.out_features, -1).to(v.weight.dtype)
        if v.bias is not None:
            v.bias.data = H.apply(v.bias.double().reshape(-1, hd)).reshape(-1).to(v.bias.dtype)
        # o_proj consumes H v per head: W_o' = W_o (I (x) H^T) on its columns
        Wo = o.weight.double().reshape(o.out_features, -1, hd)
        o.weight.data = H.apply(Wo).reshape(o.out_features, -1).to(o.weight.dtype)
    return {"folded": len(atts), "head_dim": hd, "factors": sizes}


def r1_refusal(model) -> str | None:
    """Why the residual rotation cannot be folded into `model`, or None (Llama only)."""
    for n, m in model.named_modules():
        cls = type(m).__name__
        low = cls.lower()
        if isinstance(m, (nn.LayerNorm, nn.GroupNorm)) or low.endswith("layernorm"):
            return (f"{n or 'model'} is a {cls}: mean-centering breaks the rotation invariance, which "
                    f"holds only for RMSNorm without re-scaling (QuaRot Sec. 3.4)")
        if "rmsnorm" in low and (getattr(m, "dense", None) is not None or getattr(m, "cond_dim", None)):
            return (f"{n} is an adaptive {cls} (per-token scale/shift from a condition): its scale "
                    f"cannot be folded into static weights")
        if "adaln" in low or "modulat" in low or "film" in low:
            return f"{n} is an adaLN / modulation block ({cls}): its scale cannot be folded"
    mt = getattr(getattr(model, "config", None), "model_type", None)
    if mt != "llama":
        return (f"r1 is implemented for Llama only (model_type {mt!r}); Gemma's (1 + w) RMSNorm, "
                f"tied embeddings and other layouts are not handled")
    inner = getattr(model, "model", None)
    if inner is None or not hasattr(inner, "layers") or not hasattr(inner, "embed_tokens"):
        return "not a Llama decoder layout (model.model.layers / embed_tokens)"
    for layer in inner.layers:
        for nm in ("input_layernorm", "post_attention_layernorm"):
            if type(getattr(layer, nm, None)).__name__ != "LlamaRMSNorm":
                return f"{nm} is not a LlamaRMSNorm"
    return None


def rotation_q(n: int, seed: int) -> torch.Tensor:
    sizes, full = hadamard_factors(n)
    if not full:
        raise RotationRefused(f"hidden size {n} has no Hadamard matrix here")
    H = torch.ones(1, 1, dtype=torch.float64)
    for m in sizes:
        H = torch.kron(H, pm1(m))
    H = H / math.sqrt(n)
    if seed:
        g = torch.Generator().manual_seed(int(seed))
        d = torch.randint(0, 2, (n,), generator=g).double() * 2 - 1
        H = d[:, None] * H
    return H


@torch.no_grad()
def fold_r1(model, seed: int) -> dict:
    """QuaRot stage 1a: RMSNorm weights into the next linears, then x -> xQ:
    W_in <- W_in Q (q/k/v, gate/up, lm_head), W_out <- Q^T W_out (o_proj, down_proj), E <- E Q."""
    why = r1_refusal(model)
    if why:
        raise RotationRefused(f"rotation: r1=true refused: {why}")
    inner = model.model
    d = model.config.hidden_size
    Q = rotation_q(d, seed)

    def set_w(lin, W):
        lin.weight.data = W.to(lin.weight.dtype)

    emb = inner.embed_tokens
    head = getattr(model, "lm_head", None)
    untied = False
    if head is not None and head.weight.data_ptr() == emb.weight.data_ptr():
        head.weight = nn.Parameter(head.weight.detach().clone())       # untie before rotating
        untied = True
    emb.weight.data = (emb.weight.double() @ Q).to(emb.weight.dtype)
    for layer in inner.layers:
        a, mlp = layer.self_attn, layer.mlp
        g1 = layer.input_layernorm.weight.double()
        g2 = layer.post_attention_layernorm.weight.double()
        for lin in (a.q_proj, a.k_proj, a.v_proj):
            set_w(lin, (lin.weight.double() * g1[None, :]) @ Q)
        for lin in (mlp.gate_proj, mlp.up_proj):
            set_w(lin, (lin.weight.double() * g2[None, :]) @ Q)
        layer.input_layernorm.weight.data.fill_(1.0)
        layer.post_attention_layernorm.weight.data.fill_(1.0)
        for lin in (a.o_proj, mlp.down_proj):
            set_w(lin, Q.T @ lin.weight.double())
            if lin.bias is not None:
                lin.bias.data = (lin.bias.double() @ Q).to(lin.bias.dtype)
    if head is not None:
        gf = inner.norm.weight.double()
        set_w(head, (head.weight.double() * gf[None, :]) @ Q)
        inner.norm.weight.data.fill_(1.0)
    return {"r1": True, "hidden": d, "seed": int(seed), "untied_lm_head": untied}


# ------------------------------------------------------------------------------------------------
# The algorithm
# ------------------------------------------------------------------------------------------------

_SUFFIX = {"down": "down_proj", "o": "o_proj"}
IMPLS = ("fp32", "vpu", "mxu2")


class Rotation(Algo):
    name = "rotation"

    def __init__(self, **params):
        super().__init__(**params)
        impl = str(self.params.get("impl", "fp32"))
        if impl not in IMPLS:
            raise ValueError(f"rotation: impl={impl!r}; choose from {IMPLS}")
        online = str(self.params.get("online", "down,o"))
        targets = [] if online == "none" else [t.strip() for t in online.split(",") if t.strip()]
        bad = set(targets) - set(_SUFFIX)
        if bad:
            raise ValueError(f"rotation: online={online!r}; use a comma list of {sorted(_SUFFIX)} or none")
        self.impl, self.targets = impl, targets
        self.fold_head = bool(self.params.get("fold_head", True))
        self.r1 = bool(self.params.get("r1", False))
        self.seed = int(self.params.get("seed", 0))
        self.static = bool(self.params.get("static", False))
        self.vpu_max = int(self.params.get("vpu_max", 32))
        self._reset()

    def _reset(self):
        self.specs, self.notes, self.costs = {}, [], {}
        self.static_exp, self.static_pre = {}, {}
        self._fns = {}

    @property
    def needs_calibration(self):
        return self.static

    # -- model level -------------------------------------------------------------------------
    def _prepare_once(self, model) -> dict:
        prev = getattr(model, "_rotation_prepared", None)
        if prev is not None:
            return prev
        self._reset()
        rec = {"impl": self.impl, "online": self.targets}
        if self.r1:
            rec["r1"] = fold_r1(model, self.seed)
        if self.fold_head and "o" in self.targets:
            rec["head_fold"] = fold_head_hadamard(model)
        heads = {f"{n}.o_proj" if n else "o_proj": hd for n, _, hd in _attention_modules(model)}
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            leaf = n.rsplit(".", 1)[-1]
            for t in self.targets:
                if leaf != _SUFFIX[t]:
                    continue
                if t == "o":
                    hd = heads.get(n)
                    if not hd:
                        self.notes.append(f"{n}: head_dim unknown; no online transform")
                        continue
                    spec, note = _o_spec(n, m.in_features, hd, self.impl, self.vpu_max)
                else:
                    spec, note = _down_spec(n, m.in_features, self.impl, self.vpu_max)
                if note:
                    self.notes.append(note)
                if spec is not None:
                    self.specs[n] = spec
                    self.costs[n] = spec.cost(m.out_features)
        rec.update(online_layers=len(self.specs), notes=list(self.notes))
        model._rotation_prepared = rec
        return rec

    def prepare(self, model, mode):
        return self._prepare_once(model)

    @torch.no_grad()
    def calibrate(self, model, batches, run):
        """Static scales on ROTATED inputs: the folds run first (prepare is then a no-op), and each
        online layer's input is observed after its own transform (algos.base.observe_inputs would
        see the input before it)."""
        self._prepare_once(model)
        amax, pre, handles = {}, {}, []
        for n, m in model.named_modules():
            if not isinstance(m, nn.Linear):
                continue
            spec = self.specs.get(n)

            def hook(mod, args, n=n, spec=spec):
                x = args[0].detach().to(torch.bfloat16)
                if spec is not None:
                    pre[n] = max(pre.get(n, 0.0), x.float().abs().amax().item())
                    x = self._transform(spec, x, calib=True)
                amax[n] = max(amax.get(n, 0.0), x.float().abs().amax().item())
            handles.append(m.register_forward_pre_hook(hook))
        try:
            for b in batches:
                run(model, b)
        finally:
            for h in handles:
                h.remove()
        from ref_numerics import po2_scale_exp
        self.static_exp = {n: po2_scale_exp(torch.tensor(a)) for n, a in amax.items()}
        self.static_pre = {n: po2_scale_exp(torch.tensor(a)) for n, a in pre.items()}
        return {"layers": len(amax), "online_layers_observed_after_transform": len(pre),
                "rule": "floor(log2(amax / 256)) on the calibration amax (ref_numerics.po2_scale_exp)"}

    # -- module level --------------------------------------------------------------------------
    def transform_weight(self, name, w, kind):
        spec = self.specs.get(name) if kind == "linear" else None
        return w if spec is None else spec.fold_weight(w)

    def act_scale(self, name, x, kind, dims=None):
        if not self.static or kind != "linear" or name not in self.static_exp:
            return None
        return Scale(self.static_exp[name].to(x.device), "static", dynamic=False)

    def online_transform(self, name, kind):
        spec = self.specs.get(name) if kind == "linear" else None
        if spec is None:
            return None
        f = self._fns.get(name)
        if f is None:
            f = self._fns[name] = (lambda x, spec=spec: self._run(spec, x))
        return f

    # -- the transforms ------------------------------------------------------------------------
    def _run(self, spec, x):
        with _guard():                       # the transform's own tensor ops are not model ops
            u = self._transform(spec, x)
        _count_online(spec, x)
        return u

    def _transform(self, spec, x, calib=False):
        dt = x.dtype
        if spec.impl == "fp32":
            return spec.kron_orth.apply(x.float()).to(torch.bfloat16).to(dt)
        if spec.impl == "vpu":
            lead = spec.kron_pm1.shape
            u = vpu_butterfly(x, lead, spec.mix_axis) * 2.0 ** -spec.k        # exact po2
            return u.to(torch.bfloat16).to(dt)
        # mxu2
        if calib:
            xq = x.float()
        else:
            xq = self.two_term(spec.name, x)
        u = spec.kron_pm1.apply(xq)                        # +-1 operands; one BF16 rounding below
        return (u.to(torch.bfloat16).float() * 2.0 ** -spec.k).to(torch.bfloat16).to(dt)

    def two_term(self, name, x):
        """hi + lo, two FP8 operands of the same BF16 input (the mxu2 split), as float32."""
        import ref_numerics as rn
        xb = x.to(torch.bfloat16)
        if self.static and name in self.static_pre:
            s = self.static_pre[name].to(x.device)
            dyn = False
        else:
            s = rn.po2_scale_exp(xb)
            dyn = True
        q_hi, _ = cast_with(xb, Scale(s, "mxu2_hi", dynamic=dyn))
        hi = dequant(q_hi, s)
        q_lo, _ = cast_with((xb.float() - hi).to(torch.bfloat16), Scale(s - _LO_SHIFT, "mxu2_lo", dynamic=False))
        return hi + dequant(q_lo, s - _LO_SHIFT)

    # -- bookkeeping ---------------------------------------------------------------------------
    def record(self) -> dict:
        rec = super().record()
        rec["online_cost"] = dict(self.costs)
        tot = {}
        for c in self.costs.values():
            for k in ("vpu_element_ops_per_token", "mxu_macs_per_token", "host_flops_per_token",
                      "vpu_instructions_per_32_tokens", "mxu_tile_matmuls_per_32_tokens"):
                if k in c:
                    tot[k] = tot.get(k, 0) + c[k]
        rec["online_cost_total"] = tot
        return rec


def _count_online(spec, x) -> None:
    """Add the transform's work to the chip coverage report (only while a chip session runs)."""
    import sys

    cp = sys.modules.get("chip_profile")
    if cp is None or not cp._RUNNING:
        return
    tokens = x.numel() // spec.n
    c = spec.cost(1)
    cov = cp._RUNNING[-1].cov
    op = "rotation.online_hadamard"
    if spec.impl == "vpu":
        e = c["vpu_element_ops_per_token"] * tokens
        cov.add("VPU", op, e, {"vadd.bf16": e // 2, "vsub.bf16": e - e // 2},
                reason="online Hadamard butterfly across 32-wide tiles (src/algos/rotation.py); "
                       "the 2^-k goes into the next vpack scale")
    elif spec.impl == "mxu2":
        cov.add("MXU", op, 2 * c["mxu_macs_per_token"] * tokens,
                reason="online Hadamard as two FP8 matmuls (hi, lo) against +-1 (src/algos/rotation.py)")
        e = spec.n * tokens
        cov.add("VPU", op + ".split", 2 * e, {"vunpack.fp8.bf16": e, "vsub.bf16": e},
                reason="lo = x - hi for the two-term FP8 split (src/algos/rotation.py)")
    else:
        cov.add("host", op, c["host_flops_per_token"] * tokens,
                reason="ideal float32 online Hadamard (upper bound; no Atlas unit runs it)")


ALGO = Rotation
