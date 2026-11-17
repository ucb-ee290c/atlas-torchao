"""Embedding and LM-head options of a causal LM (NEW; docs/algos/head-embed.md).

Three options, all off by default (the default export is byte-identical to one written without them):

  --tie none | lm_head_from_embedding | embedding_from_lm_head
        Tie the input embedding and the LM head to one tensor before quantization (TinyLlama is
        trained untied, so either direction changes the model). The BF16 tensor is then stored once
        (quant_export records the other name as an alias).
  --lm-head bf16 | fp8 | nvfp4
        bf16: the head stays an ordinary BF16 Linear, outside quantization (`skip=["lm_head"]`, today's
              behaviour).
        fp8:  the head joins quantization as one more quantized Linear (RefQuantLinear): the recipe's
              weight rule, calibration and activation rule apply to it exactly as to the other layers.
        nvfp4: weight-only NVFP4 storage (NVFP4Linear): the activation stays BF16, the weight is decoded
              to BF16 (exact, see below) and the matmul is the same BF16 Linear as the bf16 head.
  --embedding bf16 | fp8 | nvfp4
        Storage only (QuantEmbedding): a row lookup over the codes, then per-row (fp8) or per-block
        (nvfp4) decoding to BF16. No arithmetic changes downstream.

With a tie, --lm-head and --embedding must name the same precision: one stored tensor serves both. A
tied quantized pair keeps the head's codes (RefQuantLinear for fp8, NVFP4Linear for nvfp4) and the
embedding looks its rows up in the head's buffers (no second copy).

Formats:
  fp8 embedding (untied)  E4M3 codes of ref_numerics.e4m3_vpack (the Atlas operand encoder), one
                          power-of-two exponent per row (granularity.scale_exp, never-clip `nc` rule).
                          Row value = e4m3(code) * 2^exp, exact in BF16.
  fp8 embedding (tied)    the head's codes and its per-row exponent table ([vocab, 1]; a per_panel
                          recipe repeats one exponent per 32 rows).
  nvfp4                   E2M1 codes, two per byte (low nibble first), one E4M3 block scale per 16
                          elements along the hidden dimension, and one FP32 tensor scale. NEW: the
                          tensor scale is rounded up to a power of two, 2^ceil(log2(amax / (448 * 6))),
                          instead of torchao's amax / (448 * 6), so every decoded value
                          e2m1 * e4m3 * 2^k is exact in BF16 (e2m1 * e4m3 needs at most 6 significant
                          bits). Block scale = e4m3(clamp(block_amax / 6 / tensor_scale, 2^-6, 448)) (RNE),
                          codes = e2m1_rne(clamp(x / (block_scale * tensor_scale), -6, 6)), as torchao
                          0.18 nvfp4_quantize apart from the tensor scale. MXFP4 is not offered.

Order (src/eval.py): `prepare` (tie, BF16) -> algos.quantize with `skip_for(...)` -> `finish`
(QuantEmbedding / NVFP4Linear installed). The embedding is therefore quantized after calibration: the
decoder layers are calibrated on the BF16 embedding, so with --tie none or lm_head_from_embedding the
decoder-layer codes are the default export's, byte for byte.
"""

from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F

TIES = ("none", "lm_head_from_embedding", "embedding_from_lm_head")
PRECISIONS = ("bf16", "fp8", "nvfp4")
DEFAULTS = {"tie": "none", "lm_head": "bf16", "embedding": "bf16"}
STORAGE_KINDS = ("embedding", "nvfp4_linear")         # policy site kinds written by this module
NVFP4_BLOCK = 16
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
E2M1_MAX, E4M3_MAX, E4M3_TINY = 6.0, 448.0, 2.0 ** -6

NEW_DECISIONS = [
    "NEW (head_embed): --tie ties the input embedding and the LM head before quantization; the tied BF16 "
    "tensor is stored once (alias).",
    "NEW (head_embed): --lm-head fp8 removes the default skip of lm_head, so the head is quantized by the "
    "recipe like every other Linear.",
    "NEW (head_embed): the quantized embedding is storage only: a row lookup over the codes, then decode to "
    "BF16 (fp8: per-row power-of-two exponent; nvfp4: 16-element E4M3 block scales and a tensor scale).",
    "NEW (head_embed): the embedding is quantized after calibration; the decoder layers are calibrated on "
    "the BF16 embedding.",
    "NEW (head_embed): NVFP4 tensor scale rounded up to a power of two so decoded values are exact in BF16; "
    "the NVFP4 head is weight-only (BF16 activations, BF16 Linear on the decoded weight).",
    "NEW (head_embed): a tied quantized pair stores one set of codes: the head's; the embedding gathers rows "
    "of the head's code and scale buffers.",
]


# ---------------------------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------------------------

def add_args(ap) -> None:
    g = ap.add_argument_group("embedding and LM head (docs/algos/head-embed.md)")
    g.add_argument("--tie", choices=TIES, default="none",
                   help="tie the input embedding and the LM head before quantization: lm_head_from_embedding "
                        "(the head reuses the embedding) or embedding_from_lm_head (the reverse); default none")
    g.add_argument("--lm-head", choices=PRECISIONS, default="bf16",
                   help="LM head: bf16 (default, kept out of quantization), fp8 (quantized like the other "
                        "Linears by the recipe) or nvfp4 (weight-only NVFP4, BF16 activations)")
    g.add_argument("--embedding", choices=PRECISIONS, default="bf16",
                   help="input embedding storage: bf16 (default), fp8 (E4M3, per-row power-of-two exponent) "
                        "or nvfp4 (E2M1, 16-element E4M3 block scales); a row lookup then a decode to BF16")


def options(args) -> dict:
    get = args.get if isinstance(args, dict) else (lambda k, d=None: getattr(args, k, d))
    opts = {k: (get(k, d) or d) for k, d in DEFAULTS.items()}
    check(opts)
    return opts


def check(opts) -> None:
    if opts["tie"] not in TIES:
        raise ValueError(f"--tie {opts['tie']!r}; choose from {TIES}")
    for k in ("lm_head", "embedding"):
        if opts[k] not in PRECISIONS:
            raise ValueError(f"--{k.replace('_', '-')} {opts[k]!r}; choose from {PRECISIONS}")
    if opts["tie"] != "none" and opts["lm_head"] != opts["embedding"]:
        raise ValueError(f"--tie {opts['tie']} stores one tensor for both, so --lm-head ({opts['lm_head']}) and "
                         f"--embedding ({opts['embedding']}) must name the same precision")


def is_default(opts) -> bool:
    return all(opts.get(k, d) == d for k, d in DEFAULTS.items())


def label(opts) -> str:
    """Short name of an option set, e.g. tied_lm_head_from_embedding_fp8 or head_nvfp4_emb_nvfp4."""
    if is_default(opts):
        return "default"
    if opts["tie"] != "none":
        return f"tied_{opts['tie']}_{opts['lm_head']}"
    return f"head_{opts['lm_head']}_emb_{opts['embedding']}"


def skip_for(opts, skip=("lm_head",)) -> list:
    """The quantization skip list: the default skip without lm_head when the head is quantized in FP8."""
    skip = list(skip)
    return [s for s in skip if s != "lm_head"] if opts["lm_head"] == "fp8" else skip


# ---------------------------------------------------------------------------------------------
# FP8 rows
# ---------------------------------------------------------------------------------------------

def fp8_rows(w: torch.Tensor):
    """(codes uint8 [N, K], exponents int32 [N, 1]) of w: the never-clip rule per row, the vpack encoder."""
    import reference.ref_numerics as rn
    from algos.granularity import scale_exp

    wb = w.detach().to(torch.bfloat16)
    s = scale_exp(wb, (1,), "nc").to(torch.int32)
    return rn.e4m3_vpack(wb, s), s


def fp8_decode(codes: torch.Tensor, exp: torch.Tensor) -> torch.Tensor:
    import reference.ref_numerics as rn

    return rn.dequantize(codes, exp)


# ---------------------------------------------------------------------------------------------
# NVFP4
# ---------------------------------------------------------------------------------------------

def e2m1_table(device=None) -> torch.Tensor:
    mags = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=device)
    return torch.cat([mags, -mags])


def e4m3_table(device=None) -> torch.Tensor:
    """FP32 value of every E4M3 byte as torch.float8_e4m3fn decodes it (subnormals included; 0x7F / 0xFF NaN)."""
    return torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8).view(torch.float8_e4m3fn).float()


def e2m1_pair_table(device=None) -> torch.Tensor:
    """[256, 2] FP32: the two E2M1 values of a packed byte (low nibble first). Decoding a packed byte by one table
    gather keeps the graph free of uint8 shift / and / stack, which the backend importer leaves opaque."""
    b = torch.arange(256, dtype=torch.int64, device=device)
    t = e2m1_table(device)
    return torch.stack([t[b & 15], t[b >> 4]], -1)


def atlas_e4m3_table(device=None) -> torch.Tensor:
    """[256] FP32: ref_numerics.e4m3_unpack of every byte (the Atlas decoding: exponent-0 codes and 0x7F / 0xFF are
    zero). `table[code] * 2^exp` equals ref_numerics.dequantize(code, exp) bit for bit."""
    import reference.ref_numerics as rn

    return rn.e4m3_unpack(torch.arange(256, dtype=torch.int32, device=device).to(torch.uint8))


def e2m1_rne(v: torch.Tensor) -> torch.Tensor:
    """E2M1 codes (uint8 0..15) of v (|v| <= 6) rounded to nearest, ties to the even code; sign bit 8 only for a
    negative value, -0 included (torchao f32_to_f4_unpacked; tests compare)."""
    a = v.float().abs()
    # (threshold, tie goes up?) between consecutive magnitudes; a tie goes to the even code
    steps = ((0.25, False), (0.75, True), (1.25, False), (1.75, True), (2.5, False), (3.5, True), (5.0, False))
    code = torch.zeros_like(a, dtype=torch.int32)
    for t, up in steps:
        code += (a >= t if up else a > t).to(torch.int32)
    sign = torch.signbit(v.float())
    return (code | (sign.to(torch.int32) << 3)).to(torch.uint8)


def nvfp4_tensor_scale(w: torch.Tensor) -> torch.Tensor:
    """2^ceil(log2(amax / (448 * 6))) in FP32 (NEW: torchao uses amax / (448 * 6)); 1.0 for a zero tensor."""
    amax = w.detach().float().abs().amax()
    one = torch.ones((), dtype=torch.float32, device=w.device)
    if float(amax) == 0.0:
        return one
    m, e = torch.frexp(amax / (E4M3_MAX * E2M1_MAX))         # r = m * 2^e, m in [0.5, 1)
    k = e - (m == 0.5).to(e.dtype)                            # ceil(log2(r)); an exact power of two stays
    return torch.ldexp(one, k.to(torch.float32))


def pack_codes(codes: torch.Tensor) -> torch.Tensor:
    c = codes.contiguous()
    return (c[..., 0::2] | (c[..., 1::2] << 4)).to(torch.uint8)


def unpack_codes(packed: torch.Tensor) -> torch.Tensor:
    lo, hi = packed & 0xF, packed >> 4
    return torch.stack([lo, hi], -1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)


def nvfp4_quantize(w: torch.Tensor):
    """(packed codes uint8 [N, K/2], block-scale bytes uint8 [N, K/16], tensor scale float32 []) of w [N, K]."""
    n, k = w.shape
    if k % NVFP4_BLOCK:
        raise ValueError(f"nvfp4: the hidden size {k} is not a multiple of {NVFP4_BLOCK}")
    x = w.detach().to(torch.bfloat16).float()                   # the BF16 values are what is quantized
    ts = nvfp4_tensor_scale(x)
    blocks = x.reshape(n, k // NVFP4_BLOCK, NVFP4_BLOCK)
    bs = torch.clamp(blocks.abs().amax(-1) / E2M1_MAX / ts, min=E4M3_TINY, max=E4M3_MAX).to(torch.float8_e4m3fn)
    scaled = torch.clamp(blocks / (bs.float() * ts).unsqueeze(-1), -E2M1_MAX, E2M1_MAX)
    codes = e2m1_rne(scaled).reshape(n, k)
    return pack_codes(codes), bs.view(torch.uint8), ts


def nvfp4_decode(packed, scale_bytes, tensor_scale, pair_lut=None, e4m3_lut=None) -> torch.Tensor:
    """FP32 values [..., K] of packed rows [..., K/2] with their block-scale bytes [..., K/16] (exact). Table gathers
    only (e2m1_pair_table, e4m3_table): no bit operations in a captured graph."""
    pair_lut = e2m1_pair_table(packed.device) if pair_lut is None else pair_lut
    e4m3_lut = e4m3_table(packed.device) if e4m3_lut is None else e4m3_lut
    vals = pair_lut[packed.long()].reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    scales = e4m3_lut[scale_bytes.long()] * tensor_scale
    return (vals.reshape(*scales.shape, NVFP4_BLOCK) * scales.unsqueeze(-1)).reshape(vals.shape)


# ---------------------------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------------------------

class _Tables(nn.Module):
    def _register_tables(self):
        # graph constants (fx_export holds non-persistent, non-quantization buffers as constants)
        self.register_buffer("e2m1_pair_lut", e2m1_pair_table(), persistent=False)
        self.register_buffer("e4m3_lut", e4m3_table(), persistent=False)
        self.register_buffer("e4m3_atlas_lut", atlas_e4m3_table(), persistent=False)

    def _fp8_rows(self, codes, exp):
        """atlas_e4m3_table[code] * 2^exp: ref_numerics.dequantize as one table gather and an exp2 (no bit ops)."""
        return self.e4m3_atlas_lut[codes.long()] * torch.exp2(exp.float())


class NVFP4Linear(_Tables):
    """Weight-only NVFP4 Linear (no bias): y = F.linear(x, decode(codes)) in the layer dtype, the same BF16 Linear
    as the unquantized head on an exactly decoded weight. Buffers (non-persistent; quant_export writes them):
    weight_codes uint8 [N, K/2], weight_block_scales uint8 [N, K/16] (E4M3 bits), weight_tensor_scale float32 []."""

    fmt = "nvfp4"

    def __init__(self, linear: nn.Linear, name: str = "lm_head", tensors=None):
        super().__init__()
        if linear is not None and linear.bias is not None:
            raise NotImplementedError(f"{name}: nvfp4 head with a bias")
        self.layer_name = name
        self.in_features, self.out_features = (linear.in_features, linear.out_features) if linear is not None \
            else (tensors[0].shape[1] * 2, tensors[0].shape[0])
        self.dtype = linear.weight.dtype if linear is not None else torch.bfloat16
        codes, scales, ts = tensors if tensors is not None else nvfp4_quantize(linear.weight)
        self.register_buffer("weight_codes", codes, persistent=False)
        self.register_buffer("weight_block_scales", scales, persistent=False)
        self.register_buffer("weight_tensor_scale", ts, persistent=False)
        self._register_tables()

    def decoded_weight(self) -> torch.Tensor:
        return nvfp4_decode(self.weight_codes, self.weight_block_scales, self.weight_tensor_scale,
                            self.e2m1_pair_lut, self.e4m3_lut).to(self.dtype)

    def rows(self, ids):
        """The decoded rows `ids` (a tied embedding's lookup)."""
        return nvfp4_decode(F.embedding(ids, self.weight_codes), F.embedding(ids, self.weight_block_scales),
                            self.weight_tensor_scale, self.e2m1_pair_lut, self.e4m3_lut).to(self.dtype)

    def forward(self, x):
        return F.linear(x.to(self.dtype), self.decoded_weight())

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, nvfp4 weight-only"


class QuantEmbedding(_Tables):
    """Quantized-storage embedding: rows of the codes looked up by token id, then decoded to the model dtype.

    fmt fp8:   weight_codes uint8 [V, D], weight_scale_exp int32 [V, 1]
    fmt nvfp4: weight_codes uint8 [V, D/2], weight_block_scales uint8 [V, D/16], weight_tensor_scale float32 []
    tied_to:   the path of the head whose buffers hold the codes (a tied pair); this module then holds none,
               and `bind(model)` resolves the head (after the head's own replacement)."""

    def __init__(self, num_embeddings, embedding_dim, fmt, dtype=torch.bfloat16, tensors=None, tied_to=None,
                 padding_idx=None):
        super().__init__()
        if fmt not in ("fp8", "nvfp4"):
            raise ValueError(f"QuantEmbedding: fmt {fmt!r}")
        self.num_embeddings, self.embedding_dim, self.fmt, self.dtype = num_embeddings, embedding_dim, fmt, dtype
        self.padding_idx = padding_idx
        self.tied_to = tied_to
        object.__setattr__(self, "_tied", None)          # not a submodule: the head stays registered once
        if tied_to is None:
            if fmt == "fp8":
                codes, exp = tensors
                self.register_buffer("weight_codes", codes, persistent=False)
                self.register_buffer("weight_scale_exp", exp, persistent=False)
            else:
                codes, scales, ts = tensors
                self.register_buffer("weight_codes", codes, persistent=False)
                self.register_buffer("weight_block_scales", scales, persistent=False)
                self.register_buffer("weight_tensor_scale", ts, persistent=False)
        self._register_tables()

    @classmethod
    def from_embedding(cls, emb: nn.Embedding, fmt):
        w = emb.weight.detach()
        tensors = fp8_rows(w) if fmt == "fp8" else nvfp4_quantize(w)
        return cls(emb.num_embeddings, emb.embedding_dim, fmt, w.dtype, tensors, padding_idx=emb.padding_idx)

    def bind(self, model) -> None:
        if self.tied_to is not None:
            object.__setattr__(self, "_tied", model.get_submodule(self.tied_to))

    def source(self):
        if self.tied_to is None:
            return self
        if self._tied is None:
            raise RuntimeError(f"QuantEmbedding tied to {self.tied_to!r} is not bound (head_embed.bind_ties)")
        return self._tied

    def forward(self, ids):
        src = self.source()
        if self.fmt == "fp8":
            exp = src.weight_scale_exp            # [V, 1] per row; a tied head may hold one per-tensor exponent
            exp = F.embedding(ids, exp) if exp.dim() == 2 and exp.shape[0] == self.num_embeddings else exp
            rows = self._fp8_rows(F.embedding(ids, src.weight_codes), exp)
        else:
            rows = nvfp4_decode(F.embedding(ids, src.weight_codes), F.embedding(ids, src.weight_block_scales),
                                src.weight_tensor_scale, self.e2m1_pair_lut, self.e4m3_lut)
        return rows.to(self.dtype)

    def decoded_weight(self) -> torch.Tensor:
        return self.forward(torch.arange(self.num_embeddings, device=self.source().weight_codes.device))

    def extra_repr(self):
        return (f"{self.num_embeddings}, {self.embedding_dim}, fmt={self.fmt}"
                + (f", tied_to={self.tied_to}" if self.tied_to else ""))


def storage_modules(model) -> dict:
    """{fqn: module} of the QuantEmbedding / NVFP4Linear modules of `model`."""
    return {n: m for n, m in model.named_modules() if isinstance(m, (QuantEmbedding, NVFP4Linear))}


def bind_ties(model) -> None:
    for m in model.modules():
        if isinstance(m, QuantEmbedding):
            m.bind(model)


# ---------------------------------------------------------------------------------------------
# Applying the options
# ---------------------------------------------------------------------------------------------

def _path_of(model, module) -> str:
    return next(n for n, m in model.named_modules() if m is module)


def _set(model, path, new):
    parent, _, last = path.rpartition(".")
    setattr(model.get_submodule(parent) if parent else model, last, new)


def _sha(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.no_grad()
def prepare(model, opts) -> dict:
    """Before quantization: the tie (BF16). Returns the record of what changed."""
    check(opts)
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    rec = {"options": dict(opts), "embedding": _path_of(model, emb), "lm_head": _path_of(model, head),
           "embedding_shape": list(emb.weight.shape), "lm_head_shape": list(head.weight.shape)}
    if opts["tie"] == "none":
        return rec
    if tuple(emb.weight.shape) != tuple(head.weight.shape):
        raise ValueError(f"--tie: embedding {tuple(emb.weight.shape)} and head {tuple(head.weight.shape)} differ")
    rec["already_tied"] = emb.weight is head.weight
    rec["sha256_before"] = {"embedding": _sha(emb.weight), "lm_head": _sha(head.weight)}
    if opts["tie"] == "lm_head_from_embedding":
        head.weight = emb.weight
    else:
        emb.weight = head.weight
    rec["sha256_tied"] = _sha(emb.weight)
    return rec


@torch.no_grad()
def finish(model, opts, rec=None) -> dict:
    """After quantization: install the NVFP4 head and the quantized embedding. An fp8 head must already be a
    RefQuantLinear (quantized by algos.quantize without the lm_head skip, or by `quantize_head_reference`)."""
    rec = dict(rec or {})
    emb, head = model.get_input_embeddings(), model.get_output_embeddings()
    epath, hpath = _path_of(model, emb), _path_of(model, head)
    if opts["lm_head"] == "fp8" and type(head).__name__ != "RefQuantLinear":
        raise RuntimeError(f"--lm-head fp8: {hpath} is {type(head).__name__}, not a quantized Linear")
    if opts["lm_head"] == "nvfp4":
        head = NVFP4Linear(head, name=hpath)
        _set(model, hpath, head)
    if opts["embedding"] != "bf16":
        if opts["tie"] != "none":
            new = QuantEmbedding(emb.num_embeddings, emb.embedding_dim, opts["embedding"], emb.weight.dtype,
                                 tied_to=hpath, padding_idx=emb.padding_idx)
        else:
            new = QuantEmbedding.from_embedding(emb, opts["embedding"])
        _set(model, epath, new)
        bind_ties(model)
    rec["installed"] = {n: type(m).__name__ for n, m in storage_modules(model).items()}
    return rec


@torch.no_grad()
def quantize_head_reference(model) -> None:
    """--lm-head fp8 outside a quantized mode (the bf16 screening row): the head as a RefQuantLinear with no
    algorithm (reference per-tensor weight and dynamic activation rule)."""
    from reference.ref_profile import RefQuantLinear

    head = model.get_output_embeddings()
    hpath = _path_of(model, head)
    _set(model, hpath, RefQuantLinear(head, name=hpath))


# ---------------------------------------------------------------------------------------------
# Export / reload
# ---------------------------------------------------------------------------------------------

def policy_sites(model, tensors_out: dict) -> dict:
    """{site: policy record} of the storage modules; fills `tensors_out` with their tensors."""
    sites = {}
    for n, m in storage_modules(model).items():
        if isinstance(m, NVFP4Linear):
            names = {"codes": f"{n}.weight_codes", "block_scales": f"{n}.weight_block_scales",
                     "tensor_scale": f"{n}.weight_tensor_scale"}
            tensors_out[names["codes"]] = m.weight_codes.detach().cpu().contiguous()
            tensors_out[names["block_scales"]] = m.weight_block_scales.detach().cpu().contiguous()
            tensors_out[names["tensor_scale"]] = m.weight_tensor_scale.detach().cpu().contiguous()
            sites[n] = {"kind": "nvfp4_linear", "status": "quantized", "format": "nvfp4",
                        "weight_shape": [m.out_features, m.in_features], "weight_dtype": str(m.dtype).replace("torch.", ""),
                        "bias": None,
                        "weight": {**names, "codes_shape": list(m.weight_codes.shape), "block": NVFP4_BLOCK,
                                   "packing": "two E2M1 codes per byte, element 2i in the low nibble",
                                   "block_scale_dtype": "f8E4M3 bits (torch float8_e4m3fn decoding)",
                                   "tensor_scale_rule": "float32, a power of two (NEW)"},
                        "activation": {"mode": "none", "dtype": "bf16",
                                       "description": "weight-only: the input stays in the layer dtype"},
                        "compute": "F.linear(x, decode(weight)) in the layer dtype; decode = e2m1(code) * "
                                   "e4m3(block scale) * tensor scale, exact in BF16"}
            continue
        site = {"kind": "embedding", "status": "quantized", "format": m.fmt,
                "weight_shape": [m.num_embeddings, m.embedding_dim], "weight_dtype": str(m.dtype).replace("torch.", ""),
                "lookup": "rows of the codes (and of the per-row / per-block scales) gathered by token id, then "
                          "decoded to the layer dtype; storage only"}
        src = m.source()
        prefix = m.tied_to if m.tied_to is not None else n
        if m.fmt == "fp8":
            names = {"codes": f"{prefix}.weight_codes", "scale_exp": f"{prefix}.weight_scale_exp"}
            site["weight"] = {**names, "granularity": "per_row" if m.tied_to is None else "the head's per-row table",
                              "decode": "e4m3(code) * 2^exp (ref_numerics.dequantize), exact in BF16"}
        else:
            names = {"codes": f"{prefix}.weight_codes", "block_scales": f"{prefix}.weight_block_scales",
                     "tensor_scale": f"{prefix}.weight_tensor_scale"}
            site["weight"] = {**names, "block": NVFP4_BLOCK, "decode": "e2m1(code) * e4m3(block scale) * tensor scale"}
        if m.tied_to is not None:
            site["tied_to"] = m.tied_to
            site["weight"]["stored_with"] = m.tied_to
        else:
            tensors_out[names["codes"]] = src.weight_codes.detach().cpu().contiguous()
            if m.fmt == "fp8":
                tensors_out[names["scale_exp"]] = src.weight_scale_exp.detach().cpu().contiguous()
            else:
                tensors_out[names["block_scales"]] = src.weight_block_scales.detach().cpu().contiguous()
                tensors_out[names["tensor_scale"]] = src.weight_tensor_scale.detach().cpu().contiguous()
        sites[n] = site
    return sites


def install_from_export(model, policy, tensors, device) -> dict:
    """Replace the embedding / NVFP4 head of a fresh model with the export's storage modules (before the
    load's state-dict check); returns {site: kind}. A tied embedding is bound by `bind_ties` after the head's
    own replacement."""
    done = {}
    for n, s in policy["sites"].items():
        if s.get("status") != "quantized" or s.get("kind") not in STORAGE_KINDS:
            continue
        w = s["weight"]
        dtype = getattr(torch, s.get("weight_dtype", "bfloat16"))
        if s["kind"] == "nvfp4_linear":
            ts = (tensors[w["codes"]].to(device), tensors[w["block_scales"]].to(device),
                  tensors[w["tensor_scale"]].to(device))
            new = NVFP4Linear(None, name=n, tensors=ts)
            new.dtype = dtype
        else:
            v, d = s["weight_shape"]
            tied = s.get("tied_to")
            if tied is not None:
                ts = None
            elif s["format"] == "fp8":
                ts = (tensors[w["codes"]].to(device), tensors[w["scale_exp"]].to(device))
            else:
                ts = (tensors[w["codes"]].to(device), tensors[w["block_scales"]].to(device),
                      tensors[w["tensor_scale"]].to(device))
            new = QuantEmbedding(v, d, s["format"], dtype, ts, tied_to=tied)
        _set(model, n, new.to(device))
        done[n] = s["kind"]
    return done


def export_record(opts, rec) -> dict:
    return {"options": dict(opts), "label": label(opts), **{k: v for k, v in (rec or {}).items() if k != "options"}}
