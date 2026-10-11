"""VPU add numerics for the `chip` profile (--chip-vadd, --chip-vadd-sites).

The Atlas vadd/vsub widen BF16 to FP32, add once with RNE, and keep the raw upper 16 bits
(AddSubSumVec.scala:49 roundingMode := 0, "May change this later"; :133
fNFromRecFN(8, 24, v)(31, 16); reference model add_sub_sum_vec.py:281-288). The BF16 result is the
FP32 sum truncated toward zero: about 0.4 ulp of bias per add, which accumulates along a residual
stream. Two remedies, selectable per run; the default (`trunc`) is the chip profile unchanged.

  rn      NEW, ISA only: round-to-nearest from truncating adds (`vadd_rn`), 24 VPU instructions per
          add (vadd/vsub, vmaximum, vminimum, vrelu, vmul only; green card section 3).
  rne_hw  a model of a hardware change (one RNE narrowing in place of the upper-16-bit slice at
          AddSubSumVec.scala:133): one vadd per add. PENDING A HARDWARE RULING; results with it are
          recorded off the ladder.

Sites (`--chip-vadd-sites`):

  residual  vadd/vsub through the router's add/sub handler with two computed (non-constant) operands
            of the same shape, outside Norm, Attention, MLP and FP8-matmul module contexts, and not an
            additive mask (an operand at or below -1e30). These are the residual-stream adds (and
            same-shape stream updates such as a flow-matching x_t + dt*v_t). Module-level functions
            have no module to hook: SmolVLA's apply_rope combines are counted here too.
  data      every add/sub through that handler with two computed operands, broadcasting allowed,
            except additive masks (exact either way: x + 0 or x + -big).
  all       every vadd/vsub the VPU issues, including those inside the softmax/norm/sigmoid/GELU/
            cumsum decompositions and the bias add after an FP8 matmul.

csum (vredsum.bf16, the column/sublane sum) narrows its FP32 column accumulator with the same
upper-16-bit slice (vector_engine_model.py:393-420; ColAddVec.scala). No decomposition of the chip
profile issues csum (row reductions use vredsum.row, an FP32 tree with one RNE,
vector_engine_model.py:318-330), so there is nothing to switch; a sequence that adopts csum should
take the same policy (with rne_hw, the narrowing would be RNE too).
"""

from __future__ import annotations

import torch

import chip_ops

MODES = ("trunc", "rn", "rne_hw")
SITES = ("residual", "data", "all")
RN_OPS_PER_ADD = 24
MASK_FLOOR = -1e30
HW_RULING = ("pending hardware ruling: vadd/vsub RNE narrowing instead of the upper-16-bit slice "
             "(AddSubSumVec.scala:133)")


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

def vabs(v, x):
    """|x| = vmaximum(x, 0 - x): 2 instructions."""
    return v.pairwise_max(x, v.sub(torch.zeros_like(x), x))


def vadd_rn(v, a, b, sub: bool = False):
    """a + b (a - b with sub) rounded to the nearest BF16, ties away from zero (NEW).

    s = vadd(a, b) is the FP32 sum (one RNE) truncated to BF16. Fast2Sum (Dekker) recovers the
    remainder r = (a+b) - s in vadd's FP32 adder when the operand of larger magnitude comes first:
    r = (big - s) + small. With no select instruction, both orders are formed and blended with an
    exact 0/1 mask m = vminimum(vrelu(vmul(|a| - |b|, 2^120)), 1). r comes back truncated to BF16;
    its leading bits decide the rounding: s + 2r truncates to s's neighbour away from zero when
    |r| >= ulp(s)/2. When the FP32 add itself rounded up in magnitude (operands more than 16 bits
    apart), r has the opposite sign of s and is far below ulp(s)/2: the answer is s, so 2r is
    gated by a second exact mask m2 = vminimum(vrelu(vmul(vmul(2r, s), 2^120)), 1).
    24 instructions: vadd; 2 x vabs (4); vsub, vmul, vrelu, vminimum (m); 2 x 2 (remainders);
    vmul, vsub, vmul, vadd (blend); vadd (2r); vmul, vmul, vrelu, vminimum (m2); vmul, vadd."""
    one = torch.ones_like(a)
    big = torch.full_like(a, 2.0 ** 120)
    if sub:
        s = v.sub(a, b)
        r1 = v.sub(v.sub(a, s), b)          # big = a:      (a - s) - b
        r2 = v.sub(a, v.add(b, s))          # big = -b:     a - (b + s)
    else:
        s = v.add(a, b)
        r1 = v.add(v.sub(a, s), b)          # big = a
        r2 = v.add(v.sub(b, s), a)          # big = b
    d = v.sub(vabs(v, a), vabs(v, b))
    m = v.pairwise_min(v.relu(v.mul(d, big)), one)
    r = v.add(v.mul(m, r1), v.mul(v.sub(one, m), r2))
    rr = v.add(r, r)
    m2 = v.pairwise_min(v.relu(v.mul(v.mul(rr, s), big)), one)
    return v.add(s, v.mul(m2, rr))


def vadd_rne_hw(v, a, b, sub: bool = False):
    """The hardware-ruling model: as ChipVPU.add/sub (BF16 inputs, one FP32 RNE add) but narrowed to
    BF16 with RNE instead of the upper-16-bit slice. Counted as one vadd/vsub."""
    v._check("sub" if sub else "add", a, b)
    v._count("sub" if sub else "add", a.numel())
    a32 = chip_ops.bf16_bits_to_f32(v._bits(a))
    b32 = chip_ops.bf16_bits_to_f32(v._bits(b))
    s = a32 - b32 if sub else a32 + b32
    return v._out(chip_ops.f32_to_bf16_bits_rne(s), a)


def add(mode: str, v, a, b, sub: bool = False):
    if mode == "rn":
        return vadd_rn(v, a, b, sub)
    if mode == "rne_hw":
        return vadd_rne_hw(v, a, b, sub)
    return v.sub(a, b) if sub else v.add(a, b)


# ---------------------------------------------------------------------------
# Policy (one per router)
# ---------------------------------------------------------------------------

CTX_KINDS = ("vn_norm", "vn_attn", "vn_mlp")


def module_kind(m) -> str | None:
    c = type(m).__name__
    if "Norm" in c:
        return "vn_norm"
    if "Attention" in c:
        return "vn_attn"
    if c.endswith("MLP") or c.endswith("Mlp"):
        return "vn_mlp"
    return None


def _computed(t) -> bool:
    if not isinstance(t, torch.Tensor) or t.numel() <= 1:
        return False
    f = t.reshape(-1)
    return not bool((f == f[0]).all())


def _is_mask(t) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point() and t.numel() > 0 \
        and float(t.amin()) <= MASK_FLOOR


class VaddPolicy:
    """Which vadd/vsub take the selected numerics. `ctx` is the router's context stack."""

    def __init__(self, mode: str = "trunc", sites: str = "residual"):
        if mode not in MODES:
            raise ValueError(f"--chip-vadd {mode!r}: choose from {MODES}")
        if sites not in SITES:
            raise ValueError(f"--chip-vadd-sites {sites!r}: choose from {SITES}")
        self.mode, self.sites = mode, sites
        self.applied = 0          # adds routed through the policy (handler calls / VPU adds)
        self.extra_lane_ops = 0   # lane-ops issued beyond one vadd per element (rn: 23 per element)

    def note(self, n: int) -> None:
        self.applied += 1
        self.extra_lane_ops += (RN_OPS_PER_ADD - 1) * int(n) if self.mode == "rn" else 0

    @property
    def active(self) -> bool:
        return self.mode != "trunc"

    def wants(self, ctx, name, a, b) -> bool:
        """For the router's add/sub handler (sites residual / data; `all` is handled on the VPU)."""
        if not self.active or self.sites == "all":
            return False
        if not (_computed(a) and _computed(b)) or _is_mask(a) or _is_mask(b):
            return False
        if self.sites == "data":
            return True
        kinds = {k for k, _ in ctx}
        if kinds & {"mxu_module", *CTX_KINDS}:
            return False
        return tuple(a.shape) == tuple(b.shape)

    def record(self) -> dict:
        rec = {"mode": self.mode, "sites": self.sites,
               "instructions_per_add": {"trunc": 1, "rn": RN_OPS_PER_ADD, "rne_hw": 1}[self.mode],
               "adds_routed": self.applied, "extra_lane_ops": self.extra_lane_ops,
               "note": "the sequences' instructions are counted in the coverage instruction mix of "
                       "the op that issued them"}
        if self.mode == "rne_hw":
            rec["status"] = HW_RULING
        elif self.mode == "rn":
            rec["status"] = "NEW (ISA-only sequence, chip_numerics.vadd_rn)"
        return rec


def policy_from_env() -> VaddPolicy:
    """ATLAS_CHIP_VADD / ATLAS_CHIP_VADD_SITES (unset: trunc, i.e. unchanged). For routers built
    outside the chip profile's CLI (multitask-dit-quant: mtdit_inout/chip_full.py builds
    ChipRouter(cov, mxu=, mxu_impl=) and chip_coverage.make_mtdit_op_emulator)."""
    import os
    algo = (os.environ.get("ATLAS_ALGO") or "none").strip()
    if algo != "none":
        # NEW: an --algo / recipe cannot act through this path (docs/algos/atlas_v1.md); refuse it
        # rather than run the profile without it
        raise NotImplementedError(
            f"ATLAS_ALGO={algo!r}: a router built from the environment (multitask-dit-quant "
            "run_eval.py) has no algorithm hooks: that repo quantizes the DiT itself. Run the "
            "algorithm on Multi-Task DiT with atlas-torchao src/eval_policy.py --policy <checkpoint> "
            f"--algo {algo}, or set only ATLAS_CHIP_VADD=rn ATLAS_CHIP_VADD_SITES=residual for a "
            "recipe's vadd component")
    return VaddPolicy(os.environ.get("ATLAS_CHIP_VADD") or "trunc",
                      os.environ.get("ATLAS_CHIP_VADD_SITES") or "residual")


class NumericsVPU(chip_ops.ChipVPU):
    """ChipVPU whose vadd/vsub take the policy's numerics everywhere (--chip-vadd-sites all)."""

    def __init__(self, policy: "VaddPolicy", counter=None):
        super().__init__(counter=counter)
        self._policy = policy
        self._mode = policy.mode
        self._depth = 0

    def _route(self, a, b, sub):
        if self._depth:                       # the instructions of a sequence itself
            return super().sub(a, b) if sub else super().add(a, b)
        self._policy.note(a.numel())
        self._depth += 1
        try:
            return add(self._mode, self, a, b, sub)
        finally:
            self._depth -= 1

    def add(self, a, b):
        return self._route(a, b, False)

    def sub(self, a, b):
        return self._route(a, b, True)
