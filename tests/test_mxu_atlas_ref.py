"""Bit-for-bit check of the MXU ports (src/mxu/accum_torch.py: sa_linear_bits = MXU0, ipt_linear_bits =
MXU1) against the Atlas backend's own arithmetic library, atlas_ref.arithmetic.matmul_codes(unit="mxu0" |
"mxu1"), on raw E4M3 codes, including K that is not a multiple of 32.

atlas_ref is not part of this repo: set ATLAS_REF_DIR to the directory that contains the atlas_ref package
(merlin examples/atlas/rotary-host/reference); without it the comparison tests skip.

Known difference (docs/algos/export.md#allowed-mxu-units): when a MXU1 tile's integer total has fewer than 8
significant bits (heavy cancellation), atlas_ref (merlin 5503777, MODEL_PIN npu-model 47b8599d) leaves the BF16
significand unnormalized. npu-model fixed the same behaviour in 19caa5c ("normalize IPT sums that cancel below
eight bits", vectors observed on silicon and in RTL simulation); the port normalizes. So the random operands
here avoid exact cancellation, and test_mxu1_cancellation_* check the port on such cases without atlas_ref.
"""
import os
import sys

import numpy as np
import pytest
import torch

from mxu.accum_torch import _ipt_tile, e4m3_mul_to_prod_lut, ipt_linear_bits, sa_linear_bits

PORTS = {"mxu0": sa_linear_bits, "mxu1": lambda a, b: ipt_linear_bits(a, b, variant="numba")}


@pytest.fixture(scope="module")
def matmul_codes():
    d = os.environ.get("ATLAS_REF_DIR")
    if not d or not os.path.isdir(os.path.join(d, "atlas_ref")):
        pytest.skip("ATLAS_REF_DIR (directory holding the atlas_ref package) is not set")
    sys.path.insert(0, d)
    try:
        from atlas_ref.arithmetic import matmul_codes as mc
    finally:
        sys.path.remove(d)
    return mc


def _codes(g, kind, shape):
    c = g.integers(0, 256, shape, dtype=np.uint8)
    if kind == "encoder":            # what the reference cast emits: no 0x7F/0xFF, no subnormal codes, +0 only
        bad = ((c & 0x7F) == 0x7F) | (((c >> 3) & 0xF) == 0)
        c = np.where(bad, 0, c)
    elif kind == "zeros":            # mostly +-0 and exponent-0 codes
        z = g.choice(np.array([0x00, 0x80, 0x01, 0x81], dtype=np.uint8), shape)
        c = np.where(g.random(shape) < 0.8, z, c)
    elif kind == "wide":             # exponents 1, 2, 14, 15 only: long alignment shifts, 0x7F/0xFF included
        e = g.choice(np.array([1, 2, 14, 15], dtype=np.uint8), shape)
        c = (c & 0x87) | (e << 3)
    return np.ascontiguousarray(c.astype(np.uint8))


@pytest.mark.parametrize("unit", ["mxu0", "mxu1"])
@pytest.mark.parametrize("K", [1, 7, 31, 32, 33, 64, 77, 100, 720])
@pytest.mark.parametrize("kind", ["full", "encoder", "zeros", "wide"])
def test_port_equals_atlas_ref(matmul_codes, unit, K, kind):
    g = np.random.default_rng(1000 * K + len(kind))
    a, b = _codes(g, kind, (6, K)), _codes(g, kind, (10, K))          # x [M, K], w [N, K]
    ref = matmul_codes(a, b.T, unit=unit).astype(np.int64)
    got = PORTS[unit](torch.from_numpy(a), torch.from_numpy(b)).numpy().astype(np.int64) & 0xFFFF
    assert np.array_equal(ref, got), f"{int((ref != got).sum())} of {ref.size} words differ"


# npu-model 19caa5c tests/test_rtl_ipt_cancellation.py: two 32-term passes of a SmolVLA q.k^T, each seeded with the
# previous pass's BF16 partial; expected words as observed on silicon and in RTL simulation.
A46 = [219, 180, 224, 235, 100, 91, 110, 113, 241, 96, 200, 104, 68, 231, 240, 110, 225, 81, 234, 233, 103, 105, 100,
       222, 236, 229, 241, 109, 236, 97, 84, 104]
W46 = [93, 191, 104, 220, 112, 105, 219, 198, 236, 75, 95, 233, 222, 234, 236, 109, 105, 217, 91, 108, 233, 110, 84,
       227, 89, 241, 84, 49, 91, 81, 79, 178]
A47 = [213, 238, 234, 193, 228, 196, 218, 111, 230, 116, 109, 112, 233, 79, 102, 220, 101, 112, 233, 178, 74, 76, 205,
       233, 232, 91, 230, 104, 236, 220, 236, 102]
W47 = [93, 224, 225, 84, 234, 231, 240, 212, 229, 193, 114, 104, 218, 240, 96, 104, 223, 217, 82, 83, 100, 220, 232,
       106, 233, 227, 232, 225, 76, 234, 221, 235]


@pytest.mark.parametrize("a,w,partial,expected", [(A46, W46, 0xC717, 0xBDC0), (A47, W47, 0xC702, 0x3E00)])
def test_mxu1_cancellation_matches_silicon_vectors(a, w, partial, expected):
    """One 32-lane MXU1 instruction continuing from a BF16 partial (UsePsum), as in npu-model's test."""
    out = _ipt_tile(torch.tensor([[a]], dtype=torch.int32), torch.tensor([[w]], dtype=torch.int32),
                    e4m3_mul_to_prod_lut(), torch.tensor([[[partial]]], dtype=torch.int64), 2, 7, 32, "numba", 0, 0)
    assert int(out[0, 0, 0]) & 0xFFFF == expected


def test_mxu1_small_total_is_exact():
    """Six products cancel exactly, the seventh (0x11 * 0x3e = 1.125 * 2^-5 * 1.75 = 1.96875 * 2^-5) is all
    that is left: 63 units of the tile's integer grid (6 significant bits). The exact BF16 word is 0x3d7c.
    (atlas_ref 5503777 and npu-model 47b8599d return 0x3d3f here, npu-model 19caa5c returns 0x3d7c; see the module
    docstring.)"""
    a = np.frombuffer(bytes.fromhex("6ac7bc6ac7bc11"), dtype=np.uint8).reshape(1, 7).copy()
    b = np.frombuffer(bytes.fromhex("fa794b7af9cb3e"), dtype=np.uint8).reshape(1, 7).copy()
    got = int(PORTS["mxu1"](torch.from_numpy(a), torch.from_numpy(b))[0, 0]) & 0xFFFF
    assert got == 0x3d7c
    assert torch.tensor([got << 16], dtype=torch.int32).view(torch.float32).item() == 1.96875 * 2.0 ** -5
    # MXU0 rounds the BF16 partial after every MAC, so the large products do not cancel exactly there: the
    # rounding residue (64.0, word 0x4280) is what is left. atlas_ref's mxu0 gives the same word.
    assert int(PORTS["mxu0"](torch.from_numpy(a), torch.from_numpy(b))[0, 0]) & 0xFFFF == 0x4280
