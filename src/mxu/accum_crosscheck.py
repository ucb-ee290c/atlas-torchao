"""In-run cross-check of an accumulation backend against the reference CPU functional model.

During a reference_hw eval with the torch backend (src/mxu/accum_torch.py), wrap the accumulator so
that a seeded sample of the real layer calls is also computed by the reference CPU model, and
record the bit-level comparison in the result JSON. This makes "bit-exact Atlas arithmetic" a
property checked in the run that produced the numbers, not only in the unit tests. (All NEW: the
reference has no such mode.)

    from mxu.accum_backends import get_accumulator
    from mxu.accum_crosscheck import CrossChecked

    fm = CrossChecked(get_accumulator("sa", "torch"), get_accumulator("sa", "ref_cpu"),
                      impl="torch", ref_impl="ref_cpu", mxu="sa",
                      sample_frac=0.01, seed=0, time_budget_s=600)
    y = fm(x, w, b, static=True)          # same call contract as the wrapped accumulator
    ...
    result["crosscheck"] = fm.summary()   # JSON-serialisable; summary()["ok"] must be True

Which calls are checked:
  - the first call of every distinct (x.shape, w.shape, bias?) signature, while budget remains;
  - otherwise each call with probability `sample_frac`, from a private random.Random(seed), so
    the same run checks the same calls;
  - batched calls (w of rank > 2, e.g. per-head attention) compare at most `max_batch_items`
    seeded leading indices, since the reference runs those one index at a time;
  - once the reference time spent reaches `time_budget_s`, no further calls are checked and
    `budget_exhausted` is set.

A mismatch is never silent: it is logged and raised as a CrossCheckMismatch warning (or an
exception with strict=True), and summary()["ok"] is False. A check that itself fails (the
reference raised) is counted in `errors` and also makes ok False.

summary() fields:
  enabled            bool, always True for this wrapper
  mxu, impl, ref_impl  labels passed in (e.g. "sa", "torch", "ref_cpu")
  seed, sample_frac, time_budget_s, max_batch_items   the sampling policy
  calls_total        calls made through the wrapper
  calls_checked      calls compared against the reference
  shapes_seen        distinct call signatures seen
  shapes_checked     distinct call signatures compared at least once
  shapes_unchecked   list of signatures never compared (budget ran out first)
  elements_compared  output elements compared
  bit_mismatches     output elements whose float32 bit patterns differ (+0 vs -0 counts)
  mismatched_calls   up to 20 records {call, signature, mismatches, elements, first_index,
                     got_bits, ref_bits} for the calls that differed
  errors             reference failures during checking (records, up to 20)
  ref_time_s         wall time spent in the reference model
  budget_exhausted   bool
  ok                 calls_checked > 0 and bit_mismatches == 0 and no errors
  numerics           accum_torch.numerics_report() at summary time (device-independence facts)
"""

from __future__ import annotations

import inspect
import logging
import random
import time
import warnings
from typing import Callable, Optional

import torch

log = logging.getLogger(__name__)


class CrossCheckMismatch(UserWarning):
    """The backend under test and the reference model disagree on at least one output bit."""


class CrossCheckError(RuntimeError):
    """Raised instead of the warning when strict=True."""


def compare_bits(got: torch.Tensor, ref: torch.Tensor) -> dict:
    """Bitwise comparison of two float32 outputs (NEW). Returns elements, mismatches and, if
    any differ, the first differing flat index with both bit patterns."""
    got = got.detach().to("cpu", torch.float32).contiguous()
    ref = ref.detach().to("cpu", torch.float32).contiguous()
    if got.shape != ref.shape:
        return {"elements": int(ref.numel()), "mismatches": int(ref.numel()),
                "shape_mismatch": [list(got.shape), list(ref.shape)]}
    ne = got.view(torch.int32) != ref.view(torch.int32)
    out = {"elements": int(ne.numel()), "mismatches": int(ne.sum())}
    if out["mismatches"]:
        i = int(ne.flatten().nonzero()[0])
        out.update(first_index=i,
                   got_bits=f"{int(got.view(torch.int32).flatten()[i]) & 0xFFFFFFFF:#010x}",
                   ref_bits=f"{int(ref.view(torch.int32).flatten()[i]) & 0xFFFFFFFF:#010x}")
    return out


def crosscheck(fn: Callable, ref_fn: Callable, x: torch.Tensor, w: torch.Tensor,
               b: Optional[torch.Tensor] = None, y: Optional[torch.Tensor] = None) -> dict:
    """One-shot check: run fn (unless its output y is given) and ref_fn on the same operands,
    return compare_bits(...) plus the reference wall time."""
    if y is None:
        y = fn(x, w, b)
    t0 = time.perf_counter()
    r = ref_fn(x.detach().cpu(), w.detach().cpu(), None if b is None else b.detach().cpu())
    res = compare_bits(y, r)
    res["ref_time_s"] = time.perf_counter() - t0
    return res


class CrossChecked:
    """Wraps fm(x, w, b=None, static=False) -> y; see the module docstring."""

    def __init__(self, fn: Callable, ref_fn: Callable, *, mxu: str = "", impl: str = "torch",
                 ref_impl: str = "ref_cpu", sample_frac: float = 0.0, seed: int = 0,
                 time_budget_s: float = 600.0, max_batch_items: int = 4, strict: bool = False):
        self.fn, self.ref_fn = fn, ref_fn
        self.mxu, self.impl, self.ref_impl = mxu, impl, ref_impl
        self.sample_frac, self.seed = float(sample_frac), int(seed)
        self.time_budget_s, self.max_batch_items = float(time_budget_s), int(max_batch_items)
        self.strict = strict
        self._fn_static = self._takes_static(fn)
        self._ref_static = self._takes_static(ref_fn)
        self._rng = random.Random(self.seed)
        self.calls_total = 0
        self.calls_checked = 0
        self.elements_compared = 0
        self.bit_mismatches = 0
        self.ref_time_s = 0.0
        self.budget_exhausted = False
        self.shapes_seen: dict = {}
        self.shapes_checked: set = set()
        self.mismatched_calls: list = []
        self.errors: list = []

    @staticmethod
    def _sig(x, w, b):
        return f"x{list(x.shape)}·w{list(w.shape)}" + ("+b" if b is not None else "")

    def _want(self, sig) -> bool:
        if self.ref_time_s >= self.time_budget_s:
            self.budget_exhausted = True
            return False
        first = sig not in self.shapes_checked
        draw = self._rng.random()           # drawn on every call, so sampling is reproducible
        return first or draw < self.sample_frac

    @staticmethod
    def _takes_static(fn) -> bool:
        try:
            return "static" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            return False

    def _call_ref(self, x, w, b, static):
        if self._ref_static:
            return self.ref_fn(x, w, b, static=static)
        return self.ref_fn(x, w, b)

    def __call__(self, x, w, b=None, static: bool = False):
        y = self.fn(x, w, b, static=static) if self._fn_static else self.fn(x, w, b)
        self.calls_total += 1
        sig = self._sig(x, w, b)
        self.shapes_seen[sig] = self.shapes_seen.get(sig, 0) + 1
        if not self._want(sig):
            return y
        try:
            xs, ws, ys = x, w, y
            if w.dim() > 2:                  # batched form: a seeded subset of leading indices
                lead = w.shape[:-2]
                n = int(torch.tensor(lead).prod())
                pick = sorted(self._rng.sample(range(n), min(n, self.max_batch_items)))
                xs = x.reshape(n, *x.shape[-2:])[pick]
                ws = w.reshape(n, *w.shape[-2:])[pick]
                ys = y.reshape(n, *y.shape[-2:])[pick]
            t0 = time.perf_counter()
            r = self._call_ref(xs.detach().cpu(), ws.detach().cpu(),
                               None if b is None else b.detach().cpu(), static)
            self.ref_time_s += time.perf_counter() - t0
            res = compare_bits(ys, r)
        except Exception as e:  # a failed check must be visible, not swallowed
            self.errors.append({"call": self.calls_total, "signature": sig, "error": repr(e)})
            if len(self.errors) > 20:
                self.errors = self.errors[:20]
            log.error("crosscheck: reference failed on call %d (%s): %r", self.calls_total, sig, e)
            if self.strict:
                raise
            return y
        self.calls_checked += 1
        self.shapes_checked.add(sig)
        self.elements_compared += res["elements"]
        if res["mismatches"]:
            self.bit_mismatches += res["mismatches"]
            rec = {"call": self.calls_total, "signature": sig, **res}
            if len(self.mismatched_calls) < 20:
                self.mismatched_calls.append(rec)
            msg = (f"crosscheck: {self.impl} vs {self.ref_impl} ({self.mxu}) differ on "
                   f"{res['mismatches']}/{res['elements']} output elements at call "
                   f"{self.calls_total} {sig}")
            log.error(msg)
            if self.strict:
                raise CrossCheckError(msg)
            warnings.warn(msg, CrossCheckMismatch, stacklevel=2)
        return y

    def summary(self) -> dict:
        try:
            import mxu.accum_torch as accum_torch
            numerics = accum_torch.numerics_report()
        except Exception as e:  # pragma: no cover
            numerics = {"error": repr(e)}
        return {
            "enabled": True, "mxu": self.mxu, "impl": self.impl, "ref_impl": self.ref_impl,
            "seed": self.seed, "sample_frac": self.sample_frac,
            "time_budget_s": self.time_budget_s, "max_batch_items": self.max_batch_items,
            "calls_total": self.calls_total, "calls_checked": self.calls_checked,
            "shapes_seen": len(self.shapes_seen), "shapes_checked": len(self.shapes_checked),
            "shapes_unchecked": sorted(set(self.shapes_seen) - self.shapes_checked),
            "elements_compared": self.elements_compared, "bit_mismatches": self.bit_mismatches,
            "mismatched_calls": list(self.mismatched_calls), "errors": list(self.errors),
            "ref_time_s": round(self.ref_time_s, 3), "budget_exhausted": self.budget_exhausted,
            "ok": self.calls_checked > 0 and self.bit_mismatches == 0 and not self.errors,
            "numerics": numerics,
        }
