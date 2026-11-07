"""granularity: finer power-of-two scale granularity and MX-style 32-element blocks (docs/algos/granularity.md).

Parameters (`--algo granularity:w=...,a=...,rule=...,attn=...`):

  w     tensor | channel | block32 | panel
                                       weight scale over the whole tensor (the reference,
                                       quant_linear.py:154), per output channel ([out, 1]), per
                                       32-element block along K (OCP MX v1.0 sec. 5.2 Table 1, k=32;
                                       NOT hardware-realizable, ruling H3), or (NEW, tile-aware) panel:
                                       one exponent per group of `rows` (default 32) consecutive output
                                       channels over ALL of K, returned as the [out, 1] table of the
                                       repeated group value. A final short group is its own group.
  a     tensor | token | block32 | block | defer
                                       activation scale per tensor (the reference, quant_linear.py:153),
                                       per token (one row of the [..., T, K] operand), per block along
                                       K (block32: NOT hardware-realizable, ruling H3), per group of
                                       `rows` tokens over the full K (block, NEW, see below), or defer
                                       (NEW): granularity gives no activation Scale at all and leaves it
                                       to a stacked member (static_calib) or the reference rule
  rule  ref | nc                       ref: s = floor(log2(amax_bf16)) - 8 (the reference author's rule,
                                       hw_float_ops.py:31,38-57 = OCP MX sec. 6.3 = MX paper
                                       arXiv 2310.10537 Alg. 1); nc (NEW): s + 1 where amax > 448 * 2^s,
                                       i.e. ceil(log2(amax / 448)), so no block maximum is clipped
  a=block (NEW) with rows=B (default 32): one never-clip (nc) activation scale per B consecutive tokens over
                                       the FULL K, Linear inputs only (a Conv2d input keeps the per-tensor
                                       nc scale; attention operands keep the per-(batch, head) rule unless
                                       attn=block); the last short block is allowed. Tokens are the
                                       flattened row-major leading dims [B, T] -> B*T (docs/algos/tile-aware.md)
  attn  head | token | block | defer   attention operands: the reference per-(batch, head) scale, per row
                                       for the left operands q and p (k and v stay per head), block
                                       (NEW): q and p per `rows` rows of each (batch, head) matrix (k, v
                                       per head), or defer (NEW): no attention Scale from granularity
  rows  32                             group size of w=panel, a=block and attn=block (NEW)

Where the scales act on Atlas:
  - per-channel weight and per-token activation scales commute with the K reduction, so they are
    an exact power-of-two rescale of the MXU's BF16 output, 2^(sx[m] + sw[n]) (quant_linear.py:169-170
    generalised; algos.base.output_scale). They run on every reference step, reference_hw and chip
    included. On chip the per-token cast also needs one vmul.bf16 by 2^-sx per element before vpack
    (vpack takes one E8M0 per instruction, atlas_isa_green_card.md:358), and a rescale with both a
    row and a column factor needs one more vmul.bf16 per output element; both are counted on the
    VPU row "granularity_rescale" (chip_profile.count_algo_vpu). How a per-row exponent is derived
    from vredmax.row on chip is open (hardware ruling H2 in docs/algos/granularity.md).
  - panel / block scales (NEW, tile-aware) are constant along K, so they are exactly the scales the MXU
    pop can apply (vmatpop.fp8 multiplies a 32 x 32 output tile by one 2^(sx[block] + sw[panel])); under
    them no pre-vpack per-element vmul is counted (chip_profile.count_tile_output_rescale).
  - block32 scales sit inside the K reduction. They are defined only under framework accumulation;
    algo/base raises NotImplementedError under reference_hw / chip (hardware ruling H3).
    A K that is not a multiple of 32 falls back to the per-token / per-channel scale for that
    operand (recorded as "per_token(block_tail)" / "per_channel(block_tail)").

Why `nc` (NEW): with power-of-two scales, E4M3 rounding is invariant under the scale in the normal
range, so a finer granularity changes only which elements vpack flushes to zero and which block
maxima the `ref` rule clips to 448 (amax in (448, 512) * 2^s). The finer the granularity, the more
maxima are clipped (CPU screening, TinyLlama 4 windows: per-channel W +1.03% PPL and MX32 +0.98% under
`ref`, vs +0.84% per tensor; per-token+per-channel +0.45% `ref` / +0.38% `nc`; MX32 +0.59% `nc`). The
same rule is the collaborator's weight rule (atlas_quant.py), torchao mx RCEIL
(prototype/mx_formats/config.py:41-66) and microscaling-quant scale_factor.ocp_no_clip
(mxq/scale_factor.py:6-24). Every result with rule=nc records that it departs from the reference rule.
"""

from __future__ import annotations

import torch

import ref_numerics as rn
from algos.base import Algo, Scale

E4M3_MAX = 448.0                     # FP8Pack saturation value (fp8_e4m3.py:15-16)
BLOCK = 32                           # OCP MX v1.0 sec. 5.2 Table 1; = MXU K tile (MxuParams.scala:19)
W_CHOICES = ("tensor", "channel", "block32", "panel")
A_CHOICES = ("tensor", "token", "block32", "block", "defer")
TOKEN_BLOCK = 32                     # default rows per block of a=block (one vpack scale covers 32 tokens)
RULES = ("ref", "nc")
ATTN = ("head", "token", "block", "defer")

NC_NOTE = ("rule=nc departs from the reference author's scale rule floor(log2(amax/256)) "
           "(pi0-quant hw_float_ops.py:31,38-57): the exponent is raised by one where the block "
           "maximum would clip (amax > 448*2^s), i.e. ceil(log2(amax/448)). NEW; evidence: CPU "
           "screening in docs/algos/granularity.md (finer granularity clips more maxima under the "
           "reference rule).")


def _amax(x: torch.Tensor, dims):
    """BF16 amax as po2_scale_exp takes it (hw_float_ops.py:54), keepdim, float32."""
    a = x.to(torch.bfloat16).abs().nan_to_num(nan=0.0, posinf=0.0)
    return (a.max() if dims is None else a.amax(dim=dims, keepdim=True)).float()


def scale_exp(x: torch.Tensor, dims, rule: str) -> torch.Tensor:
    """Reference exponent over `dims` (None = whole tensor), plus the `nc` increment."""
    s = rn.po2_scale_exp(x, dims)
    if rule == "nc":
        s = s + (_amax(x, dims) > E4M3_MAX * torch.exp2(s.float())).to(torch.int32)
    return s


def block_scale_exp(x: torch.Tensor, rule: str, block: int = BLOCK) -> torch.Tensor:
    """[..., K/block, 1] exponents for blocks of `block` along the last dim (K % block == 0).
    NOT hardware-realizable: the scale varies along K (ruling H3)."""
    return scale_exp(x.to(torch.bfloat16).unflatten(-1, (-1, block)), (-1,), rule)


def row_group_scale_exp(x: torch.Tensor, rows: int, rule: str) -> torch.Tensor:
    """[..., M, 1] exponents, constant within each group of `rows` consecutive entries of axis -2, one
    value per group from the group's amax over the full last dim (K) and over its rows. x is [..., M, K];
    groups form independently for every leading index. The last group may be short: it is padded with
    zeros, which never change an amax. Used for weight panels (x = W [out, K]: groups of output
    channels), token blocks (x = the flattened [M, K] operand) and attention row blocks (x = [B, H, S, D])."""
    xb = x.to(torch.bfloat16)
    *lead, m, k = xb.shape
    ng = -(-m // rows)
    if ng * rows != m:
        xb = torch.cat([xb, xb.new_zeros(*lead, ng * rows - m, k)], dim=-2)
    s = scale_exp(xb.reshape(*lead, ng, rows, k), (-2, -1), rule).reshape(*lead, ng)
    return s.repeat_interleave(rows, dim=-1)[..., :m].unsqueeze(-1)


def panel_scale_exp(w: torch.Tensor, rows: int, rule: str) -> torch.Tensor:
    """[out, 1] int32 exponents of weight panels: one per group of `rows` consecutive output channels over
    all of K (ref | nc rule on the group's amax), the group value repeated for each member channel."""
    return row_group_scale_exp(w, rows, rule)


class Granularity(Algo):
    name = "granularity"

    def __init__(self, **params):
        super().__init__(**params)
        self.w = str(params.get("w", "tensor"))
        self.a = str(params.get("a", "tensor"))
        self.rule = str(params.get("rule", "ref"))
        self.attn = str(params.get("attn", "head"))
        self.rows = int(params.get("rows", TOKEN_BLOCK))
        if self.rows < 1:
            raise ValueError(f"granularity: rows={self.rows}; must be >= 1")
        for k, v, ok in (("w", self.w, W_CHOICES), ("a", self.a, A_CHOICES), ("rule", self.rule, RULES),
                         ("attn", self.attn, ATTN)):
            if v not in ok:
                raise ValueError(f"granularity: {k}={v!r}; choose from {ok}")
        unknown = set(params) - {"w", "a", "rule", "attn", "rows"}
        if unknown:
            raise ValueError(f"granularity: unknown parameters {sorted(unknown)}")
        self._out = {}                 # layer name -> out features with a per-column factor

    # -- weights (offline) -------------------------------------------------------------------------
    def weight_scale(self, name, w, kind):
        if self.w == "tensor":
            return None if self.rule == "ref" else Scale(scale_exp(w, None, "nc"), "per_tensor", dynamic=False)
        if self.w == "block32" and w.shape[-1] % BLOCK == 0:
            return Scale(block_scale_exp(w, self.rule), "per_block", block=BLOCK, dynamic=False)
        self._out[name] = w.shape[0]
        if self.w == "panel":
            return Scale(panel_scale_exp(w, self.rows, self.rule), self.panel_label(w.shape[0]), dynamic=False)
        g = "per_channel" if self.w == "channel" else "per_channel(block_tail)"
        return Scale(scale_exp(w, (-1,), self.rule), g, dynamic=False)

    def panel_label(self, out: int) -> str:
        """per_panel(32); a final short group (out % rows != 0) is labeled per_panel(32,partial_tail)."""
        return f"per_panel({self.rows}" + (",partial_tail)" if out % self.rows else ")")

    # -- activations (at run time) -------------------------------------------------------------------
    def act_scale(self, name, x, kind, dims=None):
        if kind.startswith("attention."):
            if self.attn == "defer":
                return None
            if self.attn == "token" and kind in ("attention.q", "attention.p"):
                return self._token(name, x, kind, out_cols=0, label="per_token")
            if self.attn == "block" and kind in ("attention.q", "attention.p"):
                return self._attn_block(x)
            if self.rule == "nc" or self.attn == "block":      # attn=block: k, v per head, never-clip
                return Scale(scale_exp(x, dims, "nc"), "per_tensor" if dims is None else "per_head")
            return None
        if self.w == "panel" and self.a not in ("token", "block32"):
            _count_tile_output(x, self._out.get(name, 0))       # pop scale per output tile (NEW costing)
        if self.a == "defer":
            return None
        if self.a == "block" and kind == "linear":
            return self._token_block(x, self._out.get(name, 0))
        if self.a == "tensor" or self.a == "block":      # a=block applies to Linear inputs only (NEW)
            return None if self.rule == "ref" else Scale(scale_exp(x, None, "nc"), "per_tensor")
        if self.a == "block32" and x.shape[-1] % BLOCK == 0:
            return Scale(block_scale_exp(x, self.rule), "per_block", block=BLOCK)
        label = "per_token" if self.a == "token" else "per_token(block_tail)"
        return self._token(name, x, kind, out_cols=self._out.get(name, 0), label=label)

    def _token_block(self, x, out_cols=0):
        """NEW a=block: one never-clip power-of-two scale per `rows` consecutive tokens over the FULL K
        (the MXU accumulator holds BF16 partial sums across K tiles, so a token block's activation scale
        must be constant across K). Tokens are the flattened leading dims (the matmul M dimension), batch
        major: [B, T, K] groups along B*T, so a group may straddle two batch elements when T % rows != 0.
        The last block may be short (decode M = 1: one group of one row); zero padding does not change
        its amax. Returned as a per-token exponent [..., 1] so the output rescale and the cast are those
        of a=token."""
        k = x.shape[-1]
        m = x.numel() // max(1, k)
        s = row_group_scale_exp(x.reshape(m, k), self.rows, "nc").reshape(*x.shape[:-1], 1)
        _count_block_cast(m, k)
        return Scale(s, "per_token_block")

    def _attn_block(self, x):
        """NEW attn=block: the left operands q and p of Q.K^T and P.V get one never-clip exponent per
        `rows` rows of each (batch, head) matrix [S, D] (groups along S, constant along the reduction),
        returned as [..., S, 1]. k and v stay per head (constant along their reduction already)."""
        s = row_group_scale_exp(x, self.rows, "nc")
        _count_block_cast(x.numel() // max(1, x.shape[-1]), x.shape[-1])
        return Scale(s, "per_row_block")

    def _token(self, name, x, kind, out_cols, label):
        _count_token_cast(x, out_cols)
        return Scale(scale_exp(x, (-1,), self.rule), label)

    def record(self) -> dict:
        rec = super().record()
        rec["numerics"] = {
            "scale_rule": "reference floor(log2(amax/256))" if self.rule == "ref"
            else "NEW nc: ceil(log2(amax/448))",
            "act_block_rows": self.rows if self.a == "block" else None,
            "act_block_rule": "NEW nc (always) on Linear inputs" if self.a == "block" else None,
            "departs_from_reference_rule": self.rule == "nc" or self.a == "block",
            "note": NC_NOTE if self.rule == "nc" or self.a == "block" else None,
            "per_block_on_mxu_models": "not defined (hardware ruling H3); framework steps only",
            "tile_native": self.tile_status(),
        }
        return rec

    def tile_status(self) -> dict:
        """Which of this configuration's scale choices the MXU pop can apply (docs/algos/tile-aware.md).
        per_tensor / panel / block / head are constant along K; channel, token and block32 are not."""
        w = {"tensor": True, "panel": True, "channel": False, "block32": False}[self.w]
        a = {"tensor": True, "block": True, "defer": True, "token": False, "block32": False}[self.a]
        t = {"head": True, "block": True, "defer": True, "token": False}[self.attn]
        why = []
        if not w:
            why.append(f"w={self.w}: " + ("one exponent per channel, not per 32-channel group" if self.w == "channel"
                                          else "varies along K (H3, not hardware-realizable)"))
        if not a:
            why.append(f"a={self.a}: " + ("one exponent per token, not per 32-token block" if self.a == "token"
                                          else "varies along K (H3, not hardware-realizable)"))
        if not t:
            why.append("attn=token: q / p exponent per row")
        return {"weights": w, "activations": a, "attention": t, "all": w and a and t, "reasons": why,
                "rows": self.rows}


def _count_block_cast(rows: int, k: int) -> None:
    """Chip coverage of a=block: the real dynamic-scale chain, one scale per 32-token block (NEW). No
    pre-vpack vmul: the vpack reads the block's scale register."""
    import sys

    cp = sys.modules.get("chip_profile")
    if cp is not None and hasattr(cp, "count_dynamic_scale_chain"):
        cp.count_dynamic_scale_chain(rows, k, "block")


def _count_tile_output(x: torch.Tensor, out_cols: int) -> None:
    """Chip coverage under w=panel with a block / static / tensor activation: no pre-vpack per-element
    vmul and no per-element output vmul; the combined pop scale 2^(sx[block] + sw[panel]) is folded into
    vmatpop.fp8 when the consumer is FP8 and costs one vmul.bf16 by a tile constant per output tile
    otherwise (chip_profile.count_tile_output_rescale; assumption documented in docs/algos/tile-aware.md)."""
    import sys

    cp = sys.modules.get("chip_profile")
    if cp is not None and hasattr(cp, "count_tile_output_rescale") and out_cols:
        cp.count_tile_output_rescale(x.numel() // max(1, x.shape[-1]), out_cols)


def _count_token_cast(x: torch.Tensor, out_cols: int) -> None:
    """Chip coverage: the per-token cast's vmul by 2^-sx before vpack (one per element), and, when
    the weight also has a per-column scale, the second output-rescale vmul (one per output
    element). The row amax (vredmax.row / vredmin.row) and the vpack are counted by the cast itself
    (algos.base.cast_with -> chip_profile.count_algo_cast)."""
    import sys

    cp = sys.modules.get("chip_profile")
    if cp is None or not hasattr(cp, "count_algo_vpu"):
        return
    n = x.numel()
    rows = n // max(1, x.shape[-1])
    if hasattr(cp, "count_dynamic_scale_chain"):
        cp.count_dynamic_scale_chain(rows, x.shape[-1], "token")      # NEW, additive: the real chain
    instrs = {"vmul.bf16": n + rows * out_cols}
    cp.count_algo_vpu(
        "granularity_rescale", instrs,
        reason="per-token scale: x * 2^-sx[m] before vpack (one E8M0 per vpack instruction, green card "
               "57/44) and, with per-channel weights, the second output factor 2^sw[n]",
        new={"choice": "vmul.bf16 with a per-row / per-column power-of-two factor",
             "analogue": "quant_linear.py:169-170 rescale 2^(sx+sw), generalised from scalars",
             "reason": "per-token / per-channel granularity (src/algos/granularity.py); the per-row "
                       "exponent derivation from vredmax.row is pending a hardware ruling (H2)"})


ALGO = Granularity
