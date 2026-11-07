"""Stacked algorithms: `--algo A:k=v+B:k=v+...` (NEW; docs/algos/stacking.md).

A Stack is an `algos.base.Algo` whose hooks delegate to its members. Nothing here is from a paper
or from the reference author's code: the composition rules are this repository's (NEW), chosen so
that every member sees exactly what it would see alone on the model the earlier members left.

Order. Members run in a FIXED order whatever order the spec lists them in (ORDER below):
  1. weight / fold transforms: smoothquant, then gptq_awq (GPTQ rounds the weights smoothing left)
  2. scale choices: granularity, static_calib
  3. layer exclusions: mixed_precision, keep_bf16
Model level (algos.quantize): bf16_layers and bf16_exact of every member (union; one reason per
exact site, the first member's), then per member in that order
calibrate (if it needs calibration) on the model as transformed by the earlier members' prepare,
then its own prepare.

Module hooks:
  transform_weight   chained in order (each member transforms the previous member's output)
  online_transform   composed in order
  act_scale          at most one member may return a Scale for a given (layer, kind); two
                     non-None answers raise StackConflict instead of silently picking one
  residual_passes    the same rule
  weight_scale       the same rule, with one explicit delegation: a member that rounds weights onto
                     a grid it does not choose (gptq_awq, attribute `weight_grid`) is given the
                     LATER members' weight_scale as its grid and pins it; its answer then stands for
                     both (recorded under `delegations`).

Refused combinations (StackConflict at construction):
  - the same algorithm twice
  - smoothquant smoothing (alpha != off) with gptq_awq method=awq|awq_gptq: both fold per-channel
    factors into the same norms / producers; the composition is exact in real arithmetic but the
    second search would optimise against already-smoothed statistics, and run 4 never measured it.
"""

from __future__ import annotations

from algos.base import Algo

# fixed hook order (NEW). Unknown names (a registered track not listed here) go after the
# transforms and before the scale choices, in spec order.
ORDER = {
    "example_identity": 0,
    "smoothquant": 10,
    "gptq_awq": 20,
    "granularity": 30,
    "static_calib": 40,
    "mixed_precision": 50,
    "keep_bf16": 55,
}
_UNKNOWN_RANK = 25
STAGE = {0: "transform", 10: "transform", 20: "transform", 25: "transform",
         30: "scale", 40: "scale", 50: "exclusion", 55: "exclusion"}


class StackConflict(ValueError):
    """Two stacked algorithms claim the same decision, or a combination is refused."""


def order(names) -> list:
    return sorted(names, key=lambda n: (ORDER.get(n, _UNKNOWN_RANK), list(names).index(n)))


class Stack(Algo):
    name = "stack"

    def __init__(self, members=None, spec=None, **params):
        super().__init__(**params)
        if members is None:                 # re-initialised by algos.quantize: keep the members
            return
        names = [m.name for m in members]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise StackConflict(f"--algo {spec!r}: {dup} listed more than once")
        by = {m.name: m for m in members}
        sq, gq = by.get("smoothquant"), by.get("gptq_awq")
        if (sq is not None and gq is not None and getattr(sq, "alpha", None) is not None
                and getattr(gq, "uses_awq", False)):
            raise StackConflict(
                f"--algo {spec!r}: smoothquant smoothing (alpha={sq.alpha}) and gptq_awq "
                f"method={gq.method} both fold per-channel factors into the same norms; stack "
                "smoothquant with method=gptq, or use smoothquant:alpha=off")
        self.members = [by[n] for n in order(names)]
        self.spec = spec
        self.needs_calibration = any(m.needs_calibration for m in self.members)
        self.delegations = []
        self._wire_weight_grid()

    # -- construction ----------------------------------------------------------------------------
    def _wire_weight_grid(self):
        for i, m in enumerate(self.members):
            if not hasattr(m, "weight_grid"):
                continue
            later = [o for o in self.members[i + 1:] if _overrides(o, "weight_scale")]
            if not later:
                continue
            m.weight_grid = lambda name, w, kind, later=later: _one("weight_scale", later, name, w, kind)
            self.delegations.append({"hook": "weight_scale", "grid_from": [o.name for o in later],
                                     "pinned_by": m.name,
                                     "rule": "NEW: the pinning member quantizes onto the chooser's grid "
                                             "and its pinned Scale is used (docs/algos/stacking.md)"})

    def reset(self):
        """Fresh per-mode stats for the stack and every member (algos.quantize)."""
        Algo.__init__(self, **self.params)
        for m in self.members:
            Algo.__init__(m, **m.params)

    # -- model level -------------------------------------------------------------------------------
    def bf16_layers(self, model, mode):
        out = []
        for m in self.members:
            for n in m.bf16_layers(model, mode) or []:
                if n not in out:
                    out.append(n)
        return out

    def bf16_exact(self, model, mode):
        out = {}
        for m in self.members:
            for n, why in (m.bf16_exact(model, mode) or {}).items():
                out.setdefault(n, why)
        return out

    # calibrate / prepare are interleaved per member by algos.quantize (see _run_members there)

    # -- module level ------------------------------------------------------------------------------
    def transform_weight(self, name, w, kind):
        for m in self.members:
            if _overrides(m, "transform_weight"):
                w = m.transform_weight(name, w, kind)
        return w

    def weight_scale(self, name, w, kind):
        for d in self.delegations:
            pin = next(m for m in self.members if m.name == d["pinned_by"])
            sc = pin.weight_scale(name, w, kind)
            if sc is not None:
                return sc                    # the chooser's grid, pinned (see _wire_weight_grid)
        return _one("weight_scale", self.members, name, w, kind)

    def act_scale(self, name, x, kind, dims=None):
        return _one("act_scale", self.members, name, x, kind, dims)

    def online_transform(self, name, kind):
        fs = [f for f in (m.online_transform(name, kind) for m in self.members
                          if _overrides(m, "online_transform")) if f is not None]
        if not fs:
            return None
        if len(fs) == 1:
            return fs[0]

        def composed(x, fs=fs):
            for f in fs:
                x = f(x)
            return x
        return composed

    def residual_passes(self, name, kind):
        return _one("residual_passes", self.members, name, kind)

    # -- bookkeeping -------------------------------------------------------------------------------
    def overridden_module_hooks(self) -> list:
        hooks = []
        for m in self.members:
            for h in m.overridden_module_hooks():
                if h not in hooks:
                    hooks.append(h)
        return hooks

    def record(self) -> dict:
        members = []
        for m in self.members:
            r = m.record()
            r.pop("granularity", None)       # the casts are counted once, on the stack
            r["stage"] = STAGE.get(ORDER.get(m.name, _UNKNOWN_RANK), "transform")
            members.append(r)
        return {"name": "stack", "spec": self.spec, "order": [m.name for m in self.members],
                "params": {m.name: dict(m.params) for m in self.members},
                "calibration": {m.name: m.calibration for m in self.members if m.calibration is not None} or None,
                "members": members, "delegations": list(self.delegations),
                "rules": "NEW (docs/algos/stacking.md): fixed order transforms -> scale choices -> "
                         "exclusions; one Scale per (layer, kind) or StackConflict",
                "granularity": dict(self.stats)}


def _overrides(algo, hook) -> bool:
    return getattr(type(algo), hook) is not getattr(Algo, hook)


def _one(hook, members, name, *args):
    """The single non-None answer of `hook` among `members`, None if none answers, StackConflict
    if two do."""
    got = []
    for m in members:
        if not _overrides(m, hook):
            continue
        r = getattr(m, hook)(name, *args)
        if r is not None:
            got.append((m.name, r))
    if len(got) > 1:
        what = ", ".join(f"{n} -> {getattr(r, 'granularity', r)}" for n, r in got)
        raise StackConflict(f"{name}: stacked algorithms both set {hook} ({what}); drop one of them "
                            "or change its parameters so that it leaves this decision to the other")
    return got[0][1] if got else None
