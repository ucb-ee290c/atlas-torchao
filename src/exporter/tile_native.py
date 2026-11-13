"""Tile-native contracts: which FP8 scale layouts the Atlas matrix unit can apply without per-K-block
rescaling (NEW; docs/algos/tile-aware.md).

Hardware facts this module encodes (ruling H3, see docs/algos/tile-aware.md):

  - The MXU multiplies UNSCALED E4M3 operands and accumulates in BF16 across all K tiles (32 wide).
  - A power-of-two exponent (E8M0) is applied only at vmatpop.fp8 (multiply), vpack (divide) and
    vunpack (multiply). So only exponents that are CONSTANT ALONG K are hardware-native:
    a weight exponent per group of 32 output channels, an activation exponent per group of 32 tokens
    (or one static per-tensor exponent), and for attention one exponent per head (or per 32-row
    block of the left operands q and p).
  - A per-K-block scale (the repository's `block32`) is not realizable.

Everything here is a pure check on exported exponent / code tensors; none of it changes a number.

Grouping rule for tokens (the "token axis"). A Linear input [B, T, K] is the 2-D matmul operand
[M = B*T, K] a compiler tiles in 32-row tiles: tokens are grouped along the flattened row-major
(batch-major) axis, so a group may span two batch elements when T % 32 != 0. The last group is partial
when M % 32 != 0 (decode, M = 1: a single group of one real row). Padding rows are zeros: they add
nothing to a group's amax and encode to the +0 code (0x00) at any exponent.
An attention operand [B, H, S, D] is grouped per (batch, head) matrix along S ("per_matrix" layout).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

GROUP = 32                                   # MXU tile edge (MxuParams.scala:19); rows of one panel / block
E8M0_BIAS = 127                              # E8M0 code = 127 + exponent
PACK_ADMISSION = (122, 128)                  # E8M0 codes the compiler's lowerPack accepts today (atlas-mlir
                                             # lib/AtlasVirtualToMachine.cpp, reported by the coordinator);
                                             # informational, nothing is clamped here
RESERVED_CODES = (0x7F, 0xFF)

# Granularity labels as recorded in policy.json. TILE_LABELS are constant along K by construction.
PER_PANEL = "per_panel({rows})"
PER_BLOCK_ROWS = "per_block({rows} rows)"


def panel_label(rows: int = GROUP) -> str:
    return PER_PANEL.format(rows=rows)


def block_rows_label(rows: int = GROUP) -> str:
    return PER_BLOCK_ROWS.format(rows=rows)


@dataclass
class GroupCheck:
    """Result of check_constant_groups."""
    ok: bool
    layout: str                              # "flat" | "per_matrix" | "scalar"
    rows: int
    n_items: int                             # length of the grouped axis (channels or tokens), per matrix
    n_groups: int
    partial_last: int = 0                    # real rows in the last group when it is short, else 0
    violations: list = field(default_factory=list)   # [{"group": g, "values": [..]}] (first few)
    n_violations: int = 0
    reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def check_constant_groups(exp: torch.Tensor, rows: int = GROUP, layout: str = "flat",
                          max_listed: int = 4) -> GroupCheck:
    """Is `exp` constant within every group of `rows` consecutive entries of its grouped axis?

    layout "flat":       the grouped axis is the exponent tensor flattened row-major after dropping a
                         trailing singleton: a weight exponent [out, 1] (axis = output channel), an
                         activation exponent [..., T, 1] (axis = flattened B*T tokens). A scalar
                         exponent is one group and constant by definition.
    layout "per_matrix": exp is [..., S, 1]; groups lie along S independently for every leading index
                         (attention q / p of every batch and head).

    The last group may be short (partial_last = its length); a short group is its own group."""
    if rows < 1:
        raise ValueError(f"rows={rows}; must be >= 1")
    e = exp.detach().cpu().to(torch.int64)
    if e.dim() == 0:                                   # a per-tensor exponent: one group, constant by definition
        return GroupCheck(True, "scalar", rows, 1, 1)
    if layout == "per_matrix":
        if e.dim() < 2 or e.shape[-1] != 1:
            return GroupCheck(False, layout, rows, 0, 0, reason=f"per_matrix needs [..., S, 1], got {tuple(e.shape)}")
        mat = e.reshape(-1, e.shape[-2])
    elif layout == "flat":
        mat = e.reshape(1, -1)
    else:
        raise ValueError(f"layout {layout!r}; choose flat, per_matrix or scalar")
    n = mat.shape[1]
    ng = -(-n // rows)
    pad = ng * rows - n
    ref = mat[:, ::rows]                                        # first entry of every group
    first = ref.repeat_interleave(rows, dim=1)[:, :n]
    bad = (mat != first)
    bad_groups = torch.zeros(mat.shape[0], ng, dtype=torch.int64).index_add_(
        1, torch.arange(n) // rows, bad.to(torch.int64)) > 0
    nbad = int(bad_groups.sum())
    listed = []
    for r, g in bad_groups.nonzero()[:max_listed].tolist():
        lo = g * rows
        listed.append({"matrix": r, "group": g, "values": sorted(set(mat[r, lo:lo + rows].tolist()))})
    return GroupCheck(nbad == 0, layout, rows, n, ng, partial_last=(rows - pad) if pad else 0,
                      violations=listed, n_violations=nbad,
                      reason=None if nbad == 0 else
                      f"exponent varies within {nbad} group(s) of {rows} along the grouped axis")


def group_reference(values: torch.Tensor, rows: int = GROUP) -> torch.Tensor:
    """Per-group first value repeated back to the length of `values` (1-D). Test helper."""
    return values[::rows].repeat_interleave(rows)[: values.numel()]


# ---------------------------------------------------------------------------------------------
# Site classification (labels, as data)
# ---------------------------------------------------------------------------------------------

def weight_site_status(label: str, exp: torch.Tensor, rows: int = GROUP) -> dict:
    """{"tile_native": bool, "reasons": [...], "check": GroupCheck dict | None} for one weight."""
    if label == "per_tensor":
        return {"tile_native": True, "reasons": [], "check": check_constant_groups(exp, rows).to_dict()}
    if label.startswith("per_panel"):
        chk = check_constant_groups(exp, rows)
        return {"tile_native": chk.ok, "reasons": [] if chk.ok else [f"weight {label}: {chk.reason}"],
                "check": chk.to_dict()}
    if label.startswith("per_block"):
        return {"tile_native": False, "check": None,
                "reasons": [f"weight {label}: one exponent per 32-element block ALONG K; the MXU accumulates "
                            "BF16 across K tiles with no per-tile scale (ruling H3)"]}
    if label.startswith("per_channel"):
        return {"tile_native": False, "check": None,
                "reasons": [f"weight {label}: one exponent per output channel; a 32 x 32 output tile pops "
                            "with one scale, so channels in one panel must share it (use w=panel)"]}
    return {"tile_native": False, "check": None, "reasons": [f"weight granularity {label!r} is not classified"]}


def activation_rule_status(rule: dict, kind: str = "linear") -> dict:
    """Tile-native status of a policy activation rule (quant_export rule dict)."""
    mode = rule.get("mode")
    if rule.get("uncalibrated_fallback"):
        # a Linear / Conv2d of a static recipe that saw no calibration data: dynamic per-tensor reference
        # rule (run-time amax). Numerically unchanged, but not static: never counted as tile-native.
        return {"tile_native": False, "dynamic": True, "uncalibrated_fallback": True,
                "reasons": ["activation uncalibrated_fallback: no calibration data in a static recipe, so the "
                            "site keeps the dynamic per-tensor reference rule (needs a run-time amax)"]}
    if rule.get("source") == "reference" or mode == "dynamic" and rule.get("reduce") in ("tensor", "reference_dims"):
        return {"tile_native": True, "reasons": [], "dynamic": mode != "static"}
    if mode == "static":
        return {"tile_native": True, "reasons": [], "dynamic": False}
    if mode == "static_per_step":
        return {"tile_native": True, "reasons": [], "dynamic": False}
    red = rule.get("reduce")
    if red in ("block", "head_block"):
        return {"tile_native": True, "reasons": [], "dynamic": True}
    if red == "token":
        return {"tile_native": False, "dynamic": True,
                "reasons": [f"activation {rule.get('label', 'per_token')}: one exponent per token; tokens of one "
                            "32-row tile would need different pop scales (use a=block)"]}
    if red == "block32_or_token":
        return {"tile_native": False, "dynamic": True,
                "reasons": ["activation per_block(32) along K: scale varies along the reduction (ruling H3)"]}
    return {"tile_native": False, "dynamic": True, "reasons": [f"activation reduction {red!r} is not classified"]}


# ---------------------------------------------------------------------------------------------
# E8M0 codes, pack admission, code hygiene
# ---------------------------------------------------------------------------------------------

def e8m0_code(exp) -> torch.Tensor:
    """E8M0 byte of a power-of-two exponent: 127 + s (int32; not clamped, may leave 0..255)."""
    return torch.as_tensor(exp).to(torch.int32) + E8M0_BIAS


def code_range(exp) -> dict:
    c = e8m0_code(exp)
    return {"min": int(c.min()), "max": int(c.max())}


def pack_admission_flags(codes_by_site: dict, lo: int = PACK_ADMISSION[0], hi: int = PACK_ADMISSION[1]) -> dict:
    """INFORMATIONAL (compiler lowerPack admission, nothing clamped). codes_by_site: {site: (min, max)} of
    ACTIVATION E8M0 codes. Returns counts and the offending sites (outside lo..hi)."""
    off = {s: {"min": a, "max": b} for s, (a, b) in codes_by_site.items() if a < lo or b > hi}
    return {"admitted_range": [lo, hi], "sites_checked": len(codes_by_site), "sites_outside": len(off),
            "offending_sites": dict(sorted(off.items())),
            "note": "informational: the compiler's lowerPack rejects activation E8M0 codes outside "
                    f"{lo}..{hi} today (atlas-mlir lib/AtlasVirtualToMachine.cpp); no exponent is clamped here"}


def reserved_code_count(codes: torch.Tensor) -> int:
    c = codes.detach().to(torch.int32) & 0xFF
    return int(((c == RESERVED_CODES[0]) | (c == RESERVED_CODES[1])).sum())


def flushed_fraction(codes: torch.Tensor) -> float:
    """Fraction of codes whose exponent field is 0 (the encoder flushes below 2^-6 to +0, so these are 0x00)."""
    c = codes.detach().to(torch.int32) & 0xFF
    return float((((c >> 3) & 0xF) == 0).float().mean()) if c.numel() else 0.0


def weight_code_report(codes: torch.Tensor, rows: int | None = None, name: str = "",
                       fail_on_reserved: bool = True) -> dict:
    """0x7F / 0xFF count (raises ValueError when fail_on_reserved and any) and the exponent-field-0 fraction
    of a [out, K] weight code tensor, overall and per group of `rows` output channels (summary: min / max /
    mean over groups and the worst groups)."""
    n_res = reserved_code_count(codes)
    if n_res and fail_on_reserved:
        raise ValueError(f"{name or 'weight'}: {n_res} reserved E4M3 code(s) 0x7F / 0xFF in the weight codes "
                         "(the encoder never emits them; atlas_e4m3 treats them as reserved)")
    rep = {"reserved_codes": n_res, "flushed_fraction": flushed_fraction(codes)}
    if rows and codes.dim() == 2 and codes.shape[0] >= 1:
        c = codes.detach().to(torch.int32) & 0xFF
        flushed = (((c >> 3) & 0xF) == 0).to(torch.float32).mean(dim=1)         # per channel
        ng = -(-c.shape[0] // rows)
        per = torch.stack([flushed[g * rows:(g + 1) * rows].mean() for g in range(ng)])
        worst = torch.argsort(per, descending=True)[:3].tolist()
        rep["flushed_fraction_per_group"] = {"rows": rows, "groups": ng, "min": float(per.min()),
                                             "max": float(per.max()), "mean": float(per.mean()),
                                             "worst_groups": [{"group": g, "fraction": float(per[g])} for g in worst]}
    return rep


def row_group_flush_summary(codes: torch.Tensor, rows: int = GROUP, per_matrix: bool = False) -> dict:
    """Exponent-field-0 fraction per group of `rows` rows of an activation code tensor [..., M, K] (flattened
    to [M, K] unless per_matrix, where every leading index forms its own matrices): {rows, groups, min, max, mean}
    over groups (a short last group counts as one group)."""
    c = (codes.detach().to(torch.int32) & 0xFF)
    flushed = (((c >> 3) & 0xF) == 0).to(torch.float32)
    mat = flushed.reshape(-1, c.shape[-2], c.shape[-1]) if per_matrix else flushed.reshape(1, -1, c.shape[-1])
    m = mat.shape[1]
    per = torch.cat([mat[:, g:g + rows].mean(dim=(1, 2)) for g in range(0, m, rows)])
    return {"rows": rows, "groups": int(per.numel()), "min": float(per.min()), "max": float(per.max()),
            "mean": float(per.mean())}


def token_split_report(n_tokens: int, rows: int = GROUP) -> dict:
    """How n_tokens (flattened B*T) split into groups of `rows`: aligned when n_tokens % rows == 0, else the
    last group is partial (its real rows listed; the padding rows are zeros)."""
    full, rem = divmod(int(n_tokens), rows)
    return {"tokens": int(n_tokens), "rows": rows, "groups": full + (1 if rem else 0),
            "aligned_32": rem == 0, "partial_last_group_rows": rem}


def padding_encodes_to_plus_zero(k: int = GROUP, rows: int = GROUP, exp: int = 0) -> bool:
    """The zero padding rows of a partial block encode to the +0 code 0x00 at any exponent (vpack of BF16
    +0 / -0). A property of the encoder (ref_numerics.e4m3_vpack), checked, not assumed."""
    import reference.ref_numerics as rn

    z = torch.zeros(rows, k, dtype=torch.bfloat16)
    return bool((rn.e4m3_vpack(z, torch.tensor(int(exp), dtype=torch.int32)) == 0).all()
                and (rn.e4m3_vpack(-z, torch.tensor(int(exp), dtype=torch.int32)) == 0).all())


# ---------------------------------------------------------------------------------------------
# Site-level and policy-level reports (used by quant_export and fx_export)
# ---------------------------------------------------------------------------------------------

def linear_site_status(weight_label: str, exp: torch.Tensor, panel_rows: int, rule: dict) -> dict:
    """Tile-native status of a Linear / Conv2d site: its weight exponent table and activation rule."""
    w = weight_site_status(weight_label, exp, panel_rows or GROUP)
    a = activation_rule_status(rule)
    return {"tile_native": bool(w["tile_native"] and a["tile_native"]),
            "reasons": list(w["reasons"]) + list(a["reasons"]), "weight": w, "activation": a,
            **({"uncalibrated_fallback": True} if a.get("uncalibrated_fallback") else {})}


def attention_site_status(operands: dict) -> dict:
    """Status of an attention site from its four operand rules (k, v and the reference per-head rule are
    constant along the reduction; q / p per token are not)."""
    per = {op: activation_rule_status(rule, "attention." + op) for op, rule in operands.items()}
    reasons = [f"attention {op}: " + r for op, st in per.items() for r in st["reasons"]]
    return {"tile_native": all(st["tile_native"] for st in per.values()), "reasons": reasons, "operands": per}


def tile_summary(site_status: dict) -> dict:
    """{site: status dict} -> counts and the list of sites that are not tile-native, with reasons."""
    bad = {n: st["reasons"] for n, st in site_status.items() if not st["tile_native"]}
    fallback = sorted(n for n, st in site_status.items() if st.get("uncalibrated_fallback"))
    return {"sites_checked": len(site_status), "tile_native_sites": len(site_status) - len(bad),
            "all_tile_native": not bad, "not_tile_native": dict(sorted(bad.items())),
            "uncalibrated_fallback_sites": fallback,
            "uncalibrated_fallback_count": len(fallback),
            "group_rows": GROUP,
            "rules": "exponents must be constant along K: weight per 32-channel group, activation per "
                     "32-token block or static per tensor, attention per head or per 32-row block "
                     "(docs/algos/tile-aware.md; ruling H3)"}


def pack_admission_report(static_act: dict, weight_ranges: dict) -> dict:
    """INFORMATIONAL compiler-pack admission (key `compiler_pack_scale_admission`; nothing is clamped).
    static_act: {site_or_operand: [exp, ...]} static activation exponents (per-step lists allowed);
    weight_ranges: {site: (min_code, max_code)} E8M0 codes of the weight exponents. Dynamic activation
    exponents are data-dependent: they are reported from FX fixtures only (graph_inspection.json).

    Without any static activation exponent (dynamic recipes) nothing was checked: the report says
    status "not_checked_dynamic" with null counts, never zero counts that read as a pass."""
    act_codes = {k: (min(v) + E8M0_BIAS, max(v) + E8M0_BIAS) for k, v in static_act.items() if v}
    if act_codes:
        rep = {"status": "checked_static_sites", **pack_admission_flags(act_codes)}
        rep["activation_static_codes"] = {k: {"min": a, "max": b} for k, (a, b) in sorted(act_codes.items())}
    else:
        rep = {"status": "not_checked_dynamic", "admitted_range": list(PACK_ADMISSION),
               "sites_checked": None, "sites_outside": None, "offending_sites": None,
               "note": "NOT CHECKED: this export has no static activation exponent, so there is nothing to "
                       "check here; the dynamic activation exponents are data dependent and are reported "
                       "from the FX capture-case fixtures only (see)",
               "see": "fx/graph_inspection.json compiler_pack_scale_admission.dynamic_activation_scales"}
    if weight_ranges:
        rep["weight_codes"] = {"sites": len(weight_ranges),
                               "min": min(a for a, _ in weight_ranges.values()),
                               "max": max(b for _, b in weight_ranges.values()),
                               "note": "weight exponent codes recorded only; the admission range applies to "
                                       "activation packs"}
    rep["dynamic_activation_scales"] = "not statically known; see graph_inspection.json fixtures"
    return rep


class TileContractError(ValueError):
    """A tile-labeled export violates exponent constancy, or carries reserved E4M3 codes."""


def verify_export(policy: dict, tensors: dict) -> dict:
    """Replay-time check of an export (load_quantized.apply_export). For every quantized Linear / Conv2d
    site whose weight granularity is labeled per_panel(...): the stored exponent must be constant within
    each group of `rows` output channels (TileContractError otherwise). Reserved codes 0x7F / 0xFF in any
    quantized weight raise TileContractError. Returns {"panel_sites_checked": n, "sites_checked": n}."""
    panels = checked = 0
    for name, site in policy["sites"].items():
        if site.get("status") != "quantized" or site.get("kind") not in ("linear", "conv2d"):
            continue
        w = site["weight"]
        codes = tensors[w["codes"]]
        checked += 1
        if reserved_code_count(codes):
            raise TileContractError(f"{name}: {reserved_code_count(codes)} reserved E4M3 code(s) 0x7F / 0xFF "
                                    "in the exported weight codes")
        if str(w.get("granularity", "")).startswith("per_panel"):
            rows = int(w.get("panel_rows") or GROUP)
            chk = check_constant_groups(tensors[w["scale_exp"]], rows)
            panels += 1
            if not chk.ok:
                raise TileContractError(f"{name}: weight labeled {w['granularity']} but {chk.reason}: "
                                        f"{chk.violations[:2]}")
    return {"panel_sites_checked": panels, "sites_checked": checked}
