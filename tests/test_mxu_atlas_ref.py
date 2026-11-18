"""Bit-for-bit check of the MXU ports (src/mxu/accum_torch.py: sa_linear_bits = MXU0, ipt_linear_bits =
MXU1) against the Atlas backend's own arithmetic library, atlas_ref.arithmetic.matmul_codes(unit="mxu0" |
"mxu1"), on raw E4M3 codes, including K that is not a multiple of 32.

atlas_ref is not part of this repo: set ATLAS_REF_DIR to the directory that contains the atlas_ref package
(merlin examples/atlas/rotary-host/reference); without it the comparison tests skip.

Known difference (docs/algos/export.md#allowed-mxu-units): atlas_ref's MXU1 tile does not normalize a tile
total with fewer than 8 significant bits (heavy cancellation), so the random operands here avoid exact
cancellation; test_mxu1_small_total_is_exact checks the port's value on such a case from first principles.
"""
import os
import sys

import numpy as np
import pytest
import torch

from mxu.accum_torch import ipt_linear_bits, sa_linear_bits

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


def test_mxu1_small_total_is_exact():
    """Six products cancel exactly, the seventh (0x11 * 0x3e = 1.125 * 2^-5 * 1.75 = 1.96875 * 2^-5) is all
    that is left: 63 units of the tile's integer grid (6 significant bits). The exact BF16 word is 0x3d7c.
    (atlas_ref 5503777 returns 0x3d3f here, the unnormalized significand; see the module docstring.)"""
    a = np.frombuffer(bytes.fromhex("6ac7bc6ac7bc11"), dtype=np.uint8).reshape(1, 7).copy()
    b = np.frombuffer(bytes.fromhex("fa794b7af9cb3e"), dtype=np.uint8).reshape(1, 7).copy()
    got = int(PORTS["mxu1"](torch.from_numpy(a), torch.from_numpy(b))[0, 0]) & 0xFFFF
    assert got == 0x3d7c
    assert torch.tensor([got << 16], dtype=torch.int32).view(torch.float32).item() == 1.96875 * 2.0 ** -5
    # MXU0 rounds the BF16 partial after every MAC, so the large products do not cancel exactly there: the
    # rounding residue (64.0, word 0x4280) is what is left. atlas_ref's mxu0 gives the same word.
    assert int(PORTS["mxu0"](torch.from_numpy(a), torch.from_numpy(b))[0, 0]) & 0xFFFF == 0x4280
