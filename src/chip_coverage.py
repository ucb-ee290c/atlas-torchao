"""The `chip` runtime: every aten op of a forward pass routed to the Atlas unit that would run it.

`ChipRouter` is a TorchDispatchMode, the mechanism of the reference author's VectorQuantMode
(smolVLA-quant quant_vector.py:242-441). Each op is classified (op_unit_map) and then

  VPU     emulated on chip_ops.ChipVPU with the reference decompositions (vpu_decomp.py) or the
          NEW ones below (each listed in `new_decisions` of the report),
  MXU     matmuls the reference_hw profile did not replace (attention scores, stray mm/bmm,
          convolutions) cast to FP8 and accumulated by the same MXU model,
  scalar  integer/bool ops: exact, run natively; arithmetic whose tensor operands are all
          one-element float32 (schedule coefficients): software float32, run natively,
  data    movement and casts: exact, run natively,
  host    no Atlas instruction: run natively and listed with the reason,
  eval    loss ops of the perplexity evaluation: listed separately.

Ops issued by the FP8 matmul path itself (operand cast, accumulation model) run under a guard, as
the reference's `_in_quant_guard` (_dispatch_guards.py; quant_linear.py:151-156); the matmul is
counted on the MXU with 2*M*N*K FLOPs. The rescale multiply and bias add after the matmul are
outside the guard and go to the VPU, as in the reference (quant_linear.py:166-172).

Nothing is left silently in plain BF16: an op with no mapping is host fallback with the reason
"unmapped", and `Coverage.report()` lists every host-fallback op.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

import chip_numerics
import chip_ops
import op_unit_map as M
import ref_loader
import ref_numerics as rn

UNITS = ("MXU", "VPU", "scalar", "data", "host", "eval")
_local = threading.local()


def _guard_depth() -> int:
    return getattr(_local, "guard", 0)


BF16_OPT_OUT = "bf16_opt_out:"     # skip-context detail prefix of an algorithm's BF16 layers


class guard:
    """Ops inside are part of an MXU operation (not routed, not counted)."""

    def __enter__(self):
        _local.guard = _guard_depth() + 1

    def __exit__(self, *exc):
        _local.guard = _guard_depth() - 1


def _decomp():
    return ref_loader.load_file("smolVLA-quant", "smolvla_inout/vpu_decomp.py")


# ---------------------------------------------------------------------------
# Coverage accounting
# ---------------------------------------------------------------------------

class Coverage:
    """Calls and work per (unit, op). Work: FLOPs (2MNK) for MXU, lane-element ops for the VPU
    (sum over issued instructions), output elements for scalar / data / host / eval."""

    def __init__(self, model_name: str = ""):
        self.model = model_name
        self.rows = defaultdict(lambda: {"calls": 0, "work": 0, "instructions": defaultdict(int)})
        self.reasons: dict[tuple, str] = {}
        self.new_rows: dict[tuple, dict] = {}
        self.live: dict = {}

    def add(self, unit, op, work, instrs=None, reason=None, new=None):
        """new: a NEW decision taken for this row that has no ATEN entry of its own (a host or data
        treatment), as {"choice", "analogue", "reason"}; listed in `new_decisions`."""
        if new:
            self.new_rows[(unit, op)] = new
        r = self.rows[(unit, op)]
        r["calls"] += 1
        r["work"] += int(work)
        for k, v in (instrs or {}).items():
            r["instructions"][k] += v
        if reason:
            self.reasons[(unit, op)] = reason

    def report(self) -> dict:
        units = {u: {"calls": 0, "work": 0} for u in UNITS}
        for (u, _), r in self.rows.items():
            units[u]["calls"] += r["calls"]
            units[u]["work"] += r["work"]
        compute = ("MXU", "VPU", "scalar", "host")
        tot_calls = sum(units[u]["calls"] for u in compute) or 1
        tot_work = sum(units[u]["work"] for u in compute) or 1
        for u in compute:
            units[u]["share_calls"] = round(units[u]["calls"] / tot_calls, 6)
            units[u]["share_work"] = round(units[u]["work"] / tot_work, 6)
        ops = []
        for (u, op), r in sorted(self.rows.items(), key=lambda kv: (kv[0][0], -kv[1]["work"])):
            e = M.ATEN.get(op)
            row = {"op": op, "unit": u, "calls": r["calls"], "work": r["work"],
                   "instructions": dict(r["instructions"])}
            if u in ("VPU", "MXU") and e is not None:
                row.update(source=e.source, status=e.status)
            if (u, op) in self.reasons:
                row["reason"] = self.reasons[(u, op)]
            ops.append(row)
        host = [{"op": r["op"], "calls": r["calls"], "work": r["work"], "reason": r.get("reason", "")}
                for r in ops if r["unit"] == "host"]
        new = sorted({op for (u, op) in self.rows if u in ("VPU", "MXU")
                      and op in M.ATEN and M.ATEN[op].status == "NEW"})
        new_decisions = [{"op": op, "choice": M.ATEN[op].instrs, "analogue": M.ATEN[op].source,
                          "reason": M.ATEN[op].reason} for op in new]
        grouped: dict = {}
        for (u, op), d in sorted(self.new_rows.items()):        # one entry per decision
            grouped.setdefault(d["choice"] + d["reason"], (d, []))[1].append(f"{op} ({u})")
        new_decisions += [{"op": ", ".join(ops), **d} for d, ops in grouped.values()]
        partial_new = sorted({op for (u, op) in self.rows if u == "VPU" and op in M.ATEN
                              and M.ATEN[op].status == "REF" and M.ATEN[op].reason})
        return {
            "model": self.model,
            "units": units,
            "work_note": "MXU work = 2*M*N*K FLOPs; VPU work = lane-element ops summed over issued "
                         "instructions; scalar/data/host/eval work = output elements. Shares are over "
                         "MXU+VPU+scalar+host (data movement and eval-only ops excluded).",
            "host_fallback": host,
            "eval_only": [{"op": r["op"], "calls": r["calls"]} for r in ops if r["unit"] == "eval"],
            "new_decisions": new_decisions,
            "reference_deviations": [{"op": op, "note": M.ATEN[op].reason} for op in partial_new],
            "ops": ops,
            "silent_bf16_ops": 0,
        }

    def refresh(self) -> dict:
        self.live.clear()
        self.live.update(self.report())
        return self.live


def format_report(rep: dict) -> str:
    u = rep["units"]
    lines = [f"[chip coverage] {rep.get('model', '')}"]
    lines.append(f"  {'unit':<7}{'calls':>10}{'work':>18}{'share(calls)':>14}{'share(work)':>13}")
    for name in UNITS:
        x = u[name]
        lines.append(f"  {name:<7}{x['calls']:>10}{x['work']:>18}"
                     f"{x.get('share_calls', float('nan')):>14.4f}{x.get('share_work', float('nan')):>13.6f}")
    if rep["host_fallback"]:
        lines.append("  host fallback:")
        for h in rep["host_fallback"]:
            lines.append(f"    {h['op']:<45} calls={h['calls']:<6} work={h['work']:<12} {h['reason']}")
    if rep["new_decisions"]:
        lines.append("  NEW decisions used: " + ", ".join(d["op"] for d in rep["new_decisions"]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _numel(out) -> int:
    if isinstance(out, torch.Tensor):
        return out.numel()
    if isinstance(out, (tuple, list)):
        return sum(_numel(o) for o in out)
    return 1


def _host_work(name, args, out) -> int:
    """FLOPs for a matmul run on the host, output elements otherwise."""
    if name in ("mm", "bmm", "addmm", "baddbmm"):
        a = args[0] if name in ("mm", "bmm") else args[1]
        return 2 * _numel(out) * a.shape[-1]
    return _numel(out)


def _tensors(args):
    for a in args:
        if isinstance(a, torch.Tensor):
            yield a
        elif isinstance(a, (list, tuple)):
            yield from (t for t in a if isinstance(t, torch.Tensor))


def _is_one_element_fp32(tensors, floats) -> bool:
    """Every tensor operand holds one element and every float one is float32.

    Python scalars may join (`1 - alpha_prod_t` is rsub(tensor, 1)): a literal never promotes a
    float32 tensor, so the op still computes in float32. One-element integer/bool tensors may join
    too (they promote to float32). Any operand with more than one element (a sample, an activation)
    keeps the op on the VPU, and a BF16/FP16 or float64 one-element operand keeps its own treatment."""
    return bool(floats) and all(t.numel() == 1 for t in tensors) \
        and all(t.dtype == torch.float32 for t in floats)


def _vpu_handler(func, name, unit=None):
    """The chip handler of an op (by overload, then by packet name), optionally only on `unit`."""
    h = _VPU_HANDLERS.get(func) or _VPU_HANDLERS.get(name)
    return h if h is not None and unit in (None, h.unit) else None


def _is_const(t) -> bool:
    if not isinstance(t, torch.Tensor):
        return True
    if t.numel() <= 1:
        return True
    f = t.reshape(-1)
    return bool((f == f[0]).all())


def _full(like: torch.Tensor, value, shape=None) -> torch.Tensor:
    return torch.full(shape if shape is not None else like.shape, float(value), dtype=torch.float32,
                      device=like.device)


# ---------------------------------------------------------------------------
# The router
# ---------------------------------------------------------------------------

class ChipRouter(TorchDispatchMode):
    """Route a forward pass onto the Atlas units. One instance per model."""

    def __init__(self, coverage: Coverage, mxu: str = "ipt", mxu_impl: str = "ref_cpu",
                 rope: str = "vpu", strict: bool = False, native_ops=(), vadd="env"):
        super().__init__()
        self.native_ops = set(native_ops)
        self.cov = coverage
        self._instr = defaultdict(int)
        # --chip-vadd / --chip-vadd-sites (chip_numerics). None or mode "trunc": the profile unchanged.
        # "env" (the default, for routers built outside the chip profile, e.g. multitask-dit-quant's
        # chip_full adapter): ATLAS_CHIP_VADD / ATLAS_CHIP_VADD_SITES, unset = unchanged.
        if isinstance(vadd, str) and vadd == "env":
            vadd = chip_numerics.policy_from_env()
        self.vadd = vadd if vadd is not None and vadd.active else None
        if self.vadd is not None and self.vadd.sites == "all":
            self.vpu = chip_numerics.NumericsVPU(self.vadd, counter=self._count)
        else:
            self.vpu = chip_ops.ChipVPU(counter=self._count)
        self.dec = _decomp()
        self.rope = rope
        self.strict = strict
        from accum_backends import get_accumulator
        self.fm = get_accumulator(mxu, mxu_impl)
        self.ctx: list[tuple[str, str]] = []     # (kind, detail) from module hooks
        self.mxu_flops = 0

    # -- accounting ----------------------------------------------------------------------
    def _count(self, instr, n):
        self._instr[instr] += n

    def count_mxu(self, op, flops, extra=None):
        self.cov.add("MXU", op, flops, extra)

    # -- module contexts -----------------------------------------------------------------
    def push(self, kind, detail=""):
        self.ctx.append((kind, detail))

    def pop(self):
        self.ctx.pop()

    def _context(self):
        for kind, detail in reversed(self.ctx):
            if kind in ("skip", "rope_host"):
                return kind, detail
        return (self.ctx[-1] if self.ctx else (None, ""))

    # -- dispatch ------------------------------------------------------------------------
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _guard_depth():
            return func(*args, **kwargs)
        name = func.overloadpacket.__name__
        kind, detail = self._context()
        if kind in ("skip", "rope_host"):
            out = func(*args, **kwargs)
            if name in M.DATA:
                self.cov.add("data", name, _numel(out))
            else:
                new, row = None, name
                if kind == "skip" and detail.startswith(BF16_OPT_OUT):
                    # NEW: a layer an algorithm keeps in BF16 (src/algos bf16_layers): no BF16-operand
                    # MXU mode exists, so it is host work, on its own row
                    why, row = "kept in BF16 by the quantization algorithm (--algo, bf16_layers)", f"{name}@bf16_opt_out"
                elif kind == "skip":
                    why = f"kept off the chip on request (--skip {detail})"
                elif detail in M.TABLE_HOST:
                    why, new, row = M.TABLE_HOST[detail]["reason"], M.TABLE_HOST[detail], f"{name}@{detail}"
                else:
                    why = "RoPE table (position-only), precomputed on the host (--chip-rope host)"
                self.cov.add("host", row, _host_work(name, args, out), reason=why, new=new)
            return out
        unit, handler, reason = self.classify(func, name, args, kwargs, kind)
        new = M.NEW_TREATMENTS.get(reason) if unit in ("scalar", "host", "data") else None
        self._instr.clear()
        out = handler.fn(self, func, name, args, kwargs) if handler else func(*args, **kwargs)
        if unit == "VPU":
            work = sum(self._instr.values())
            self.cov.add("VPU", name, work, dict(self._instr), reason)
        elif unit == "MXU":
            vpu_work = sum(self._instr.values())
            if vpu_work:
                self.cov.add("VPU", f"{name}.rescale_bias", vpu_work, dict(self._instr),
                             reason="VPU side of an MXU op: x 2^(sa+sb) rescale and bias add")
        else:
            self.cov.add(unit, name, _numel(out), reason=reason, new=new)
        return out

    def classify(self, func, name, args, kwargs, kind):
        ts = list(_tensors(list(args) + list(kwargs.values())))
        floats = [t for t in ts if t.is_floating_point()]
        if name in self.native_ops:
            return "host", None, "forced to the host for an ablation (--chip-native)"
        if name in M.EVAL_ONLY:
            return "eval", None, "evaluation-only loss (not part of inference)"
        if name in M.DATA:
            return "data", None, M.DATA_NOTES.get(name)
        if not floats:
            if name in M.INT_SCALAR or name in ("arange",):
                why = ("RV32I has no multiply/divide: software shift-add routine, exact (NEW)"
                       if name in M.INT_SOFTWARE else None)
                return "scalar", None, why
            if name == "reciprocal":
                return "host", None, M.HOST_REASONS["reciprocal_int"]
            return "host", None, M.HOST_REASONS.get(name, f"unmapped integer op {name}")
        if any(t.dtype == torch.float64 for t in floats) and all(t.numel() <= 1 for t in floats):
            return "host", None, M.HOST_REASONS["float64"]
        if name in ("add_", "sub_") and isinstance(args[0], torch.Tensor) and args[0].dim() == 0 \
                and all(t.dim() == 0 for t in floats):
            # a loop-carried 0-d float scalar updated in place (flow-matching `time += dt`)
            return "host", None, M.LOOP_SCALAR
        if name == "where" and func._overloadname in ("self", "ScalarOther", "ScalarSelf", "Scalar") \
                and _is_const(args[1]) and _is_const(args[2]):
            # where(mask, c1, c2): building the additive mask itself from a boolean mask
            return "data", None, M.MASK_BUILD
        if kind == "mxu_module" and name == "exp2":
            # 2^(sx+sw) from integer scale exponents: an exponent field written by the scalar core
            return "scalar", None, "2^k from an integer scale exponent: exponent-field construction (exact)"
        if _is_one_element_fp32(ts, floats) and _vpu_handler(func, name, unit="VPU") is not None:
            # schedule arithmetic on one-element fp32 coefficients (diffusers DDPM alphas/betas): BF16
            # lanes would round 1 - alpha_prod_t to 0 and the step would divide 0 by 0. NEW, as
            # multitask-dit-quant chip_full.py:58-65. After the loop-scalar and exponent rules above,
            # which keep their own (also float32) treatment of 0-d operands.
            return "scalar", None, M.ONE_ELEMENT_FP32
        h = _vpu_handler(func, name)
        if h is not None:
            ok, why = h.applies(self, func, name, args, kwargs)
            if ok:
                return h.unit, h, None
            return "host", None, why
        if name.startswith("_scaled_dot_product"):
            return "host", None, M.HOST_REASONS.get(name, "fused SDPA kernel")
        reason = M.HOST_REASONS.get(name, f"unmapped float op {name}: no Atlas instruction mapped")
        if self.strict and name not in M.HOST_REASONS:
            raise RuntimeError(f"chip: unmapped float op {func}")
        return "host", None, reason

    # -- MXU emulation for matmuls outside reference modules -----------------------------
    def mxu_matmul(self, a, b, per_matrix: bool):
        """a [..., M, K] @ b [..., K, N] with FP8 operands (reference cast) on the MXU model."""
        out_dtype = torch.result_type(a, b)
        K = a.shape[-1]
        if K == 1:                                     # outer product: one vmul per element
            aa, bb = torch.broadcast_tensors(a, b)      # [..., M, 1] * [..., 1, N]
            return self.vpu.mul(aa.contiguous().float(), bb.contiguous().float()).to(out_dtype), 0
        dims = (-2, -1) if per_matrix else None
        bt = b.transpose(-1, -2)
        with guard():
            aq, sa = rn.quantize(a, dims)
            bq, sb = rn.quantize(bt, dims)
            if per_matrix and a.dim() > 2:
                lead = torch.broadcast_shapes(a.shape[:-2], bt.shape[:-2])
                xa = rn.e4m3_unpack(aq).expand(*lead, *a.shape[-2:]).contiguous()
                xb = rn.e4m3_unpack(bq).expand(*lead, *bt.shape[-2:]).contiguous()
                y = self.fm(xa, xb, None)
                scale = torch.exp2((sa + sb).float()).expand(*lead, 1, 1)
            else:
                y = self.fm(rn.e4m3_unpack(aq), rn.e4m3_unpack(bq), None)
                scale = torch.exp2((sa + sb).float())
        flops = 2 * y.numel() * K
        y = self.vpu.mul(y, scale.expand_as(y).contiguous())       # vmul by 2^(sa+sb) (quant_linear.py:169)
        return y.to(out_dtype), flops


# ---------------------------------------------------------------------------
# VPU / MXU handlers
# ---------------------------------------------------------------------------

class _H:
    unit = "VPU"

    def __init__(self, fn, applies=None, unit="VPU"):
        self.fn, self._applies, self.unit = fn, applies, unit

    def applies(self, r, func, name, args, kwargs):
        return (True, None) if self._applies is None else self._applies(r, func, name, args, kwargs)



def _as_f(x, like):
    """Python scalar or tensor -> float32 tensor on like's device.

    A 0-dim CPU tensor follows like's device, as PyTorch lets a 0-dim CPU tensor take part in
    an op on another device (e.g. a diffusers scheduler coefficient times a CUDA sample). Any
    other device mix is left alone, so the op fails exactly where PyTorch itself would."""
    if isinstance(x, torch.Tensor):
        if x.dim() == 0 and x.device.type == "cpu" and x.device != like.device:
            x = x.to(like.device)
        return x.float()
    return torch.tensor(float(x), dtype=torch.float32, device=like.device)


def _operand_device(a, b):
    """The device a binary op computes on: the device of an operand that is not a 0-dim CPU
    tensor (PyTorch's rule), else the first operand's."""
    for x in (a, b):
        if isinstance(x, torch.Tensor) and not (x.dim() == 0 and x.device.type == "cpu"):
            return x
    return next(x for x in (a, b) if isinstance(x, torch.Tensor))


def _bcast2(a, b):
    """Two-sided broadcast (pi0-quant vpu_decomp.py:218-225 _bcast_both)."""
    a, b = torch.broadcast_tensors(a, b)
    return a.contiguous(), b.contiguous()


def _result(out, func, name, args, dtype):
    out = out.to(dtype)
    if name.endswith("_"):                              # in place: write back
        args[0].copy_(out)
        return args[0]
    return out


def _dtype_of(func, args, kwargs):
    xs = [a for a in args if isinstance(a, (torch.Tensor, int, float, bool))]
    try:
        return torch.result_type(*xs[:2]) if len(xs) >= 2 else xs[0].dtype
    except Exception:
        return next(a for a in args if isinstance(a, torch.Tensor)).dtype


def _binary(op):
    def fn(r, func, name, args, kwargs):
        a, b = args[0], args[1]
        alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
        dt = _dtype_of(func, args, kwargs)
        if name == "rsub":
            a, b = b, a
        like = _operand_device(a, b)
        fa, fb = _as_f(a, like), _as_f(b, like)
        if alpha != 1:
            fb = r.vpu.mul(*_bcast2(fb, _as_f(alpha, fb)))
        if op in ("add", "sub") and r.vadd is not None and r.vadd.wants(r.ctx, name, fa, fb):
            # --chip-vadd rn / rne_hw on this site (chip_numerics; instructions counted as issued)
            xa, xb = _bcast2(fa, fb)
            r.vadd.note(xa.numel())
            out = chip_numerics.add(r.vadd.mode, r.vpu, xa, xb, sub=(op == "sub"))
            return _result(out, func, name, args, dt)
        fa, fb = _bcast2(fa, fb)
        if op == "div":
            out = r.vpu.mul(fa, r.vpu.rcp(fb))                  # vpu_decomp.py:253-255
        else:
            out = getattr(r.vpu, op)(fa, fb)
        return _result(out, func, name, args, dt)
    return fn


def _unary(method):
    def fn(r, func, name, args, kwargs):
        x = args[0]
        out = method(r, x.float())
        return _result(out, func, name, args, x.dtype)
    return fn


def _neg(r, x):
    return r.vpu.sub(torch.zeros_like(x), x)                    # vpu_decomp.py:263-264


def _sigmoid(r, x):                                              # vpu_decomp.py:188-191
    e = _safe_exp(r, r.vpu.sub(torch.zeros_like(x), x))
    return r.vpu.rcp(r.vpu.add(torch.full_like(e, 1.0), e))


def _log(r, x):
    return r.vpu.mul(r.vpu.log2(x), torch.full_like(x, math.log(2.0)))


def _abs(r, x):
    return r.vpu.pairwise_max(x, _neg(r, x))


def _pow_applies(r, func, name, args, kwargs):
    if func is torch.ops.aten.pow.Tensor_Scalar:
        return (float(args[1]) in (1.0, 2.0, 3.0, 0.5, -1.0, -0.5)), \
            f"pow with exponent {args[1]}: no VPU sequence (host, as vpu_decomp.py:205-208)"
    if func is torch.ops.aten.pow.Scalar:
        base = float(args[0])
        return base > 0, "pow with a non-positive scalar base"
    return False, "pow(Tensor, Tensor): host"


def _pow(r, func, name, args, kwargs):
    if func is torch.ops.aten.pow.Tensor_Scalar:
        x, e = args[0], float(args[1])
        f = x.float()
        out = {2.0: lambda: r.vpu.square(f), 3.0: lambda: r.vpu.cube(f),     # vpu_decomp.py:199-204
               1.0: lambda: f, 0.5: lambda: r.vpu.sqrt(f), -1.0: lambda: r.vpu.rcp(f),
               -0.5: lambda: r.vpu.rcp(r.vpu.sqrt(f))}[e]()
        return out.to(_dtype_of(func, args, kwargs))
    base, e = float(args[0]), args[1]                                         # c^x = 2^(x log2 c)
    f = e.float()
    out = r.vpu.exp2(r.vpu.mul(f, torch.full_like(f, math.log2(base))))
    return out.to(e.dtype)


def _clamp(r, func, name, args, kwargs):
    x = args[0]
    lo = kwargs.get("min", args[1] if len(args) > 1 else None)
    hi = kwargs.get("max", args[2] if len(args) > 2 else None)
    if name == "clamp_max":
        lo, hi = None, args[1]
    f = x.float()
    if lo is not None:
        f = r.vpu.pairwise_max(*_bcast2(f, _as_f(lo, f)))
    if hi is not None:
        f = r.vpu.pairwise_min(*_bcast2(f, _as_f(hi, f)))
    return _result(f, func, name, args, x.dtype)


def _maxmin2(op):
    def fn(r, func, name, args, kwargs):
        dt = _dtype_of(func, args, kwargs)
        return getattr(r.vpu, op)(*_bcast2(args[0].float(), args[1].float())).to(dt)
    return fn


def _reduce_last(r, x, kind):
    d = r.dec
    if kind == "sum":
        return d._vpu_rsum_all(r.vpu, x)
    if kind == "max":
        return d._vpu_rmax_all(r.vpu, x)
    # min: as _vpu_rmax_all with +inf padding (vpu_decomp.py:53-72 analogue)
    while x.shape[-1] > 1:
        k = x.shape[-1]
        if k % 32:
            pad = torch.full((*x.shape[:-1], 32 - k % 32), float("inf"), dtype=x.dtype, device=x.device)
            x = torch.cat([x, pad], -1)
        x = r.vpu.rmin(x)
    return x


def _reduce(r, x, dims, keepdim, kind):
    """Staged row reductions over `dims` (each moved last, as quant_vector.py:220-235)."""
    f = x.float()
    if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        dims = list(range(x.dim()))
    if isinstance(dims, int):
        dims = [dims]
    dims = sorted({d % max(x.dim(), 1) for d in dims}) if x.dim() else []
    if not dims:
        return f
    for d in dims:
        f = _reduce_last(r, f.transpose(d, -1).contiguous(), kind).transpose(d, -1)
    if not keepdim:
        for d in reversed(dims):
            f = f.squeeze(d)
    return f.contiguous()


def _sum(r, func, name, args, kwargs):
    x = args[0]
    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    dt = kwargs.get("dtype") or x.dtype
    return _reduce(r, x, dims, keep, "sum").to(dt)


def _mean(r, func, name, args, kwargs):
    x = args[0]
    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    s = _reduce(r, x, dims, keep, "sum")
    dd = range(x.dim()) if not dims else ([dims] if isinstance(dims, int) else dims)
    n = 1
    for d in dd:
        n *= x.shape[d]
    out = r.vpu.mul(s, torch.full_like(s, 1.0 / n))                 # quant_vector.py:404-407
    return out.to(kwargs.get("dtype") or x.dtype)


def _ext(kind):
    def fn(r, func, name, args, kwargs):
        x = args[0]
        dims = args[1] if len(args) > 1 else kwargs.get("dim")
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        return _reduce(r, x, dims, keep, kind).to(x.dtype)
    return fn


def _maxmin_applies(r, func, name, args, kwargs):
    return func.overloadpacket in (torch.ops.aten.max, torch.ops.aten.min) and \
        func._overloadname in ("default", "other"), f"{name}.{func._overloadname} returns indices: host"


def _maxmin(r, func, name, args, kwargs):
    if func._overloadname == "other":
        return _maxmin2("pairwise_max" if name == "max" else "pairwise_min")(r, func, name, args, kwargs)
    return _reduce(r, args[0], None, False, "max" if name == "max" else "min").to(args[0].dtype)


# vexp.bf16 is exact only down to x = -336: below that the reference FPEX model (bit-exact to
# ExpLane.scala) wraps in its Q(m,n) conversion and returns large or unit values (exp(-1000) =
# 2.6e10, exp(-3.3e38) = 1). Every exp of a value that can be very negative (softmax after the
# additive mask, SiLU/sigmoid of large inputs) is therefore preceded by vmaximum(x, EXP_FLOOR);
# exp(EXP_FLOOR) is exactly 0 in the model (zero for -336 <= x <= -93). NEW (one vmaximum per
# element), the analogue being the RTL's own overflow pre-check (expFPIsInf, ExpLane.scala:47).
EXP_FLOOR = -128.0


def _safe_exp(r, x):
    return r.vpu.exp(r.vpu.pairwise_max(x, torch.full_like(x, EXP_FLOOR)))


def _softmax(r, func, name, args, kwargs):
    """The reference sequence (vpu_decomp.py:132-158) with the exp floor above."""
    x, dim = args[0], args[1]
    half_to_float = len(args) > 2 and args[2]
    f = x.float()
    d = dim % f.dim()
    t = f.transpose(d, -1).contiguous() if d != f.dim() - 1 else f
    x_max = r.dec._vpu_rmax_all(r.vpu, t)
    e = _safe_exp(r, r.vpu.sub(t, x_max.expand_as(t).contiguous()))
    inv = r.vpu.rcp(r.dec._vpu_rsum_all(r.vpu, e))
    out = r.vpu.mul(e, inv.expand_as(e).contiguous())
    if d != f.dim() - 1:
        out = out.transpose(d, -1).contiguous()
    return out.to(torch.float32 if half_to_float else x.dtype)


def _layer_norm(r, func, name, args, kwargs):
    x, shape, w, b, eps = args[:5]
    n = len(shape)
    lead = x.shape[: x.dim() - n]
    xf = x.float().reshape(*lead, -1)
    wf = (w if w is not None else torch.ones(shape, device=x.device)).float().reshape(-1)
    bf = b.float().reshape(-1) if b is not None else None
    y = r.dec._vpu_layer_norm(r.vpu, xf, wf, bf, eps)                  # vpu_decomp.py:84-129
    y = y.reshape(x.shape).to(x.dtype)
    stat = x.new_zeros((*lead, *([1] * n)), dtype=torch.float32)
    return y, stat, stat


def _rms_norm(r, func, name, args, kwargs):
    x, shape, w, eps = args[0], args[1], args[2] if len(args) > 2 else None, args[3] if len(args) > 3 else None
    n = len(shape)
    xf = x.float().reshape(*x.shape[: x.dim() - n], -1)
    eps = eps if eps is not None else torch.finfo(x.dtype).eps
    var = _mean(r, None, "mean", (r.vpu.square(xf), [-1], True), {})
    inv = r.dec._vpu_rsqrt(r.vpu, r.vpu.add(var, torch.full_like(var, eps)))
    y = r.vpu.mul(xf, inv.expand_as(xf).contiguous())
    if w is not None:
        y = r.vpu.mul(y, w.float().reshape(-1).expand_as(y).contiguous())
    return y.reshape(x.shape).to(x.dtype), inv


def _gelu(r, func, name, args, kwargs):
    x = args[0]
    return r.dec._vpu_gelu_tanh(r.vpu, x.float()).to(x.dtype)       # vpu_decomp.py:161-178


def _silu(r, func, name, args, kwargs):
    """x * sigmoid(x): the reference SiLU sequence (vpu_decomp.py:181-192) with the exp floor."""
    x = args[0]
    f = x.float()
    return _result(r.vpu.mul(f, _sigmoid(r, f)), func, name, args, x.dtype)


def _cumsum(r, func, name, args, kwargs):
    x, dim = args[0], args[1]
    f = x.float().transpose(dim, -1).contiguous()
    outs = [f[..., :1]]
    for i in range(1, f.shape[-1]):
        outs.append(r.vpu.add(outs[-1].contiguous(), f[..., i:i + 1].contiguous()))
    return torch.cat(outs, -1).transpose(dim, -1).contiguous().to(kwargs.get("dtype") or x.dtype)


def _where_applies(r, func, name, args, kwargs):
    if name == "where":
        if func._overloadname not in ("self", "ScalarOther", "ScalarSelf", "Scalar"):
            return False, "where(cond) (index form): host"
        x, y = args[1], args[2]
        if _is_const(x) or _is_const(y):
            return True, None
        return False, "select between two computed tensors: no select instruction"
    return True, None


def _where(r, func, name, args, kwargs):
    if name == "where":
        cond, x, y = args[0], args[1], args[2]
        dt = torch.result_type(x, y)
        like = next(t for t in (x, y, cond) if isinstance(t, torch.Tensor))
        if _is_const(x) and _is_const(y):            # a mask being built: constant data
            return torch.where(cond, x, y)
        if _is_const(y):
            data, keep, fill = x, cond, y
        else:
            data, keep, fill = y, ~cond, x
    else:                                              # masked_fill(x, mask, value)
        data, keep, fill = args[0], ~args[1], args[2]
        dt = data.dtype
    fillv = float(fill.reshape(-1)[0]) if isinstance(fill, torch.Tensor) else float(fill)
    fill_b = torch.tensor(fillv, dtype=torch.float32).bfloat16().float().item()
    add = torch.where(keep, torch.zeros((), device=data.device), torch.tensor(fill_b, device=data.device))
    a, m = _bcast2(data.float(), add.float())
    out = r.vpu.add(a, m)                       # host-precomputed additive mask, one vadd
    return _result(out, func, name, args, dt)


def _mm_like(r, func, name, args, kwargs):
    if name in ("mm", "bmm"):
        a, b = args[0], args[1]
        bias, beta, alpha = None, 1, 1
    else:                                               # addmm / baddbmm(bias, a, b, beta, alpha)
        bias, a, b = args[0], args[1], args[2]
        beta, alpha = kwargs.get("beta", 1), kwargs.get("alpha", 1)
    y, flops = r.mxu_matmul(a, b, per_matrix=(name in ("bmm", "baddbmm")))
    yf = y.float()
    if alpha != 1:
        yf = r.vpu.mul(yf, torch.full_like(yf, alpha))
    if bias is not None and beta != 0:
        bf = bias.float()
        if beta != 1:
            bf = r.vpu.mul(bf, torch.full_like(bf, beta))
        yf = r.vpu.add(*_bcast2(yf, bf))
    if flops:
        r.count_mxu(name, flops)
    return yf.to(y.dtype)


def _conv_applies(r, func, name, args, kwargs):
    groups = args[8] if len(args) > 8 else 1
    transposed = args[6] if len(args) > 6 else False
    return (groups == 1 and not transposed and args[1].dim() == 4), \
        "grouped / transposed / non-2D convolution: host (model_patcher.py:339-343)"


def _conv(r, func, name, args, kwargs):
    """im2col + FP8 matmul (QuantConv2d functional path, model_patcher.py:345-375)."""
    import torch.nn.functional as F
    x, w, b = args[:3]
    stride, padding, dilation = ([int(v) for v in a] * (2 if len(a) == 1 else 1) for a in args[3:6])
    B, _, H, W = x.shape
    O, _, kh, kw = w.shape
    oh = (H + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (W + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
    xf = cols.permute(0, 2, 1).reshape(B * oh * ow, -1).to(w.dtype)
    y, flops = r.mxu_matmul(xf, w.reshape(O, -1).t(), per_matrix=False)
    yf = y.float().reshape(B, oh * ow, O).permute(0, 2, 1).reshape(B, O, oh, ow)
    if b is not None:
        yf = r.vpu.add(*_bcast2(yf.contiguous(), b.float().view(1, -1, 1, 1)))
    r.count_mxu(name, flops)
    return yf.to(x.dtype)


A = torch.ops.aten
_VPU_HANDLERS: dict = {
    "add": _H(_binary("add")), "add_": _H(_binary("add")), "sub": _H(_binary("sub")),
    "sub_": _H(_binary("sub")), "rsub": _H(_binary("sub")), "mul": _H(_binary("mul")),
    "mul_": _H(_binary("mul")), "div": _H(_binary("div"),
                                         lambda r, f, n, a, k: (k.get("rounding_mode") is None, "div with rounding mode: host")),
    "div_": _H(_binary("div")),
    "reciprocal": _H(_unary(lambda r, x: r.vpu.rcp(x))), "neg": _H(_unary(_neg)),
    "sqrt": _H(_unary(lambda r, x: r.vpu.sqrt(x))),
    "rsqrt": _H(_unary(lambda r, x: r.dec._vpu_rsqrt(r.vpu, x))),
    "exp": _H(_unary(lambda r, x: r.vpu.exp(x))), "exp2": _H(_unary(lambda r, x: r.vpu.exp2(x))),
    "log2": _H(_unary(lambda r, x: r.vpu.log2(x))), "log": _H(_unary(_log)),
    "sin": _H(_unary(lambda r, x: r.vpu.sin(x))), "cos": _H(_unary(lambda r, x: r.vpu.cos(x))),
    "tanh": _H(_unary(lambda r, x: r.vpu.tanh(x))), "relu": _H(_unary(lambda r, x: r.vpu.relu(x))),
    "abs": _H(_unary(_abs)), "sigmoid": _H(_unary(_sigmoid)),
    "pow": _H(_pow, _pow_applies),
    "clamp": _H(_clamp), "clamp_": _H(_clamp), "clamp_min": _H(_clamp), "clamp_max": _H(_clamp),
    "maximum": _H(_maxmin2("pairwise_max")), "minimum": _H(_maxmin2("pairwise_min")),
    "silu": _H(_silu), "silu_": _H(_silu), "gelu": _H(_gelu), "_softmax": _H(_softmax),
    "_safe_softmax": _H(_softmax), "native_layer_norm": _H(_layer_norm), "_fused_rms_norm": _H(_rms_norm),
    "sum": _H(_sum), "mean": _H(_mean), "amax": _H(_ext("max")), "amin": _H(_ext("min")),
    "max": _H(_maxmin, _maxmin_applies), "min": _H(_maxmin, _maxmin_applies),
    "cumsum": _H(_cumsum),
    "where": _H(_where, _where_applies), "masked_fill": _H(_where), "masked_fill_": _H(_where),
    "mm": _H(_mm_like, unit="MXU"), "bmm": _H(_mm_like, unit="MXU"), "addmm": _H(_mm_like, unit="MXU"),
    "baddbmm": _H(_mm_like, unit="MXU"), "convolution": _H(_conv, _conv_applies, unit="MXU"),
}


def unmapped_reference_ops() -> list:
    """Reference TARGET_OPS packets with no handler here (should be empty)."""
    import types as _t
    import importlib
    import sys
    ref_loader.require()
    if "_chip_ref_smolvla_inout" not in sys.modules:
        pkg = _t.ModuleType("_chip_ref_smolvla_inout")
        pkg.__path__ = [str(ref_loader.repo_path("smolVLA-quant") / "smolvla_inout")]
        sys.modules["_chip_ref_smolvla_inout"] = pkg
    qv = importlib.import_module("_chip_ref_smolvla_inout.quant_vector")
    return sorted({f.overloadpacket.__name__ for f in qv.TARGET_OPS} - set(_VPU_HANDLERS))


# ---------------------------------------------------------------------------
# Plug-in for multitask-dit-quant (mtdit_inout/op_emulation.py emulator protocol)
# ---------------------------------------------------------------------------

class MtditEmulator:
    """Emulator object for multitask-dit-quant's OpRouter (load_op_emulator("chip_coverage")).

    classify(func, args, kwargs) -> "VPU" | "scalar" | None. Data movement returns None (that
    router counts it as movement); matmuls return None: that repo's FP8 patches take every matmul
    inside its mxu_section, so a matmul reaching its router is not on the MXU and must show as host.
    emulate(func, args, kwargs) runs the op on ChipVPU with the same handlers as the chip profile.
    """

    name = "atlas_chip"
    emulates = True

    def __init__(self, coverage: Coverage | None = None, vadd: str = "trunc", vadd_sites: str = "residual"):
        # vadd / vadd_sites: --chip-vadd numerics (chip_numerics). There are no module context hooks
        # here, so "residual" means every same-shape add of two computed tensors outside masks.
        self.router = ChipRouter(coverage or Coverage("multitask_dit"), mxu_impl="ref_cpu",
                                 vadd=chip_numerics.VaddPolicy(vadd, vadd_sites))
        self.router.active = False
        chip_ops.warm()

    def classify(self, func, args, kwargs=None):
        name = func.overloadpacket.__name__
        unit, _, _ = self.router.classify(func, name, args, kwargs or {}, None)
        return unit if unit in ("VPU", "scalar") else None

    def emulate(self, func, args, kwargs=None):
        kwargs = kwargs or {}
        name = func.overloadpacket.__name__
        unit, h, _ = self.router.classify(func, name, args, kwargs, None)
        if unit == "VPU" and h is not None:
            return h.fn(self.router, func, name, args, kwargs)
        return func(*args, **kwargs)


def make_mtdit_op_emulator(**kw) -> MtditEmulator:
    """kw: vadd, vadd_sites (MtditEmulator). Without them, the environment variables
    ATLAS_CHIP_VADD / ATLAS_CHIP_VADD_SITES are read, so a caller that only names this factory
    (multitask-dit-quant's load_op_emulator) can select --chip-vadd numerics; unset = unchanged."""
    env = chip_numerics.policy_from_env()
    kw.setdefault("vadd", env.mode)
    kw.setdefault("vadd_sites", env.sites)
    return MtditEmulator(**kw)
