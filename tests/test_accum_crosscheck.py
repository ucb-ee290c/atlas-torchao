"""src/mxu/accum_crosscheck.py: the in-run bit-level cross-check of an accumulation backend."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mxu.accum_crosscheck as xc  # noqa: E402
import mxu.accum_torch as at  # noqa: E402
from test_accum_torch import bytes_to_float, needs_ref, numba_ref, rand_bytes  # noqa: E402,F401


def _xy(seed, batched=False):
    g = torch.Generator().manual_seed(seed)
    if batched:
        return (bytes_to_float(rand_bytes((2, 3, 4, 40), g, "normal")),
                bytes_to_float(rand_bytes((2, 3, 5, 40), g, "normal")))
    return bytes_to_float(rand_bytes((4, 40), g, "normal")), bytes_to_float(rand_bytes((6, 40), g, "normal"))


def test_crosscheck_agreeing_backends_report_ok():
    fm = xc.CrossChecked(at.sa_linear, at.sa_linear, mxu="sa", sample_frac=0.0, seed=1)
    for s in range(3):
        fm(*_xy(s))                       # same signature: only the first is checked
    fm(*_xy(5, batched=True))             # new signature: checked on a sample of heads
    s = fm.summary()
    assert s["ok"] and s["calls_total"] == 4 and s["calls_checked"] == 2
    assert s["shapes_seen"] == 2 and s["shapes_checked"] == 2 and s["bit_mismatches"] == 0
    assert s["elements_compared"] == 4 * 6 + 4 * 4 * 5


def test_crosscheck_mismatch_is_loud():
    def off_by_one_bit(x, w, b=None):
        y = at.sa_linear(x, w, b)
        y.view(torch.int32)[0, 0] ^= 1 << 16
        return y
    fm = xc.CrossChecked(off_by_one_bit, at.sa_linear, mxu="sa")
    with pytest.warns(xc.CrossCheckMismatch):
        fm(*_xy(0))
    s = fm.summary()
    assert not s["ok"] and s["bit_mismatches"] == 1 and s["mismatched_calls"][0]["first_index"] == 0
    strict = xc.CrossChecked(off_by_one_bit, at.sa_linear, strict=True)
    with pytest.raises(xc.CrossCheckError):
        strict(*_xy(0))


def test_crosscheck_reference_failure_is_recorded():
    def broken(x, w, b=None):
        raise RuntimeError("reference unavailable")
    fm = xc.CrossChecked(at.ipt_linear, broken)
    fm(*_xy(0))
    s = fm.summary()
    assert not s["ok"] and len(s["errors"]) == 1


def test_crosscheck_sampling_is_seeded_and_budgeted():
    def run(seed, budget):
        fm = xc.CrossChecked(at.ipt_linear, at.ipt_linear, sample_frac=0.5, seed=seed,
                             time_budget_s=budget)
        x, w = _xy(0)
        for _ in range(20):
            fm(x, w)
        return fm.summary()
    a, b = run(7, 1e9), run(7, 1e9)
    assert a["calls_checked"] == b["calls_checked"] and 1 < a["calls_checked"] < 20
    z = run(7, 0.0)
    assert z["calls_checked"] == 0 and z["budget_exhausted"] and not z["ok"]
    assert z["shapes_unchecked"]


@needs_ref
def test_crosscheck_against_reference_ipt_numba(numba_ref):
    _, lin = numba_ref
    ref = lin.IPTLinearRTLFunction(vec_len=32, num_lanes=32)
    fm = xc.CrossChecked(at.ipt_linear, lambda x, w, b=None: ref(x, w, b), mxu="ipt",
                         ref_impl="ipt_numba")
    fm(*_xy(3))
    s = fm.summary()
    assert s["ok"] and s["calls_checked"] == 1
