"""src/scripts/tile_flow/hw_fixtures.py contractions against the Atlas backend's atlas_ref, bit for bit.

atlas_ref is pinned to npu-model 47b8599, whose MXU1 leaves a tile total under 8 significant bits unnormalized;
hw_fixtures' default MXU1 normalizes it (npu-model 19caa5c, silicon / RTL), so the bit-for-bit comparison uses
variant "model47" and the short-total case checks both.

atlas_ref (rotary-host/reference/atlas_ref) lives outside this repo: set ATLAS_REF_ROOT to the directory that
holds atlas_ref/ (the tests skip otherwise). The model47 edge case below needs no reference tree.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "src" / "scripts" / "tile_flow"))

import mxu.accum_torch as at  # noqa: E402

ROOT = os.environ.get("ATLAS_REF_ROOT")


def test_model47_short_total():
    """A tile whose total cancels to fewer than 8 significant bits: x = [128, 128, a], w = [128, -128, 0.25].
    numba normalizes (0x3D80 = 2^-4); npu-model 47b8599 / atlas_ref keep the unshifted low bits (0x3DC0)."""
    x = torch.tensor([[0x70, 0x70, 0x28]], dtype=torch.uint8)
    w = torch.tensor([[0x70, 0xF0, 0x28]], dtype=torch.uint8)
    assert int(at.ipt_linear_bits(x, w, variant="numba")[0, 0]) == 0x3D80
    assert int(at.ipt_linear_bits(x, w, variant="model47")[0, 0]) == 0x3DC0
    assert int(at.sa_linear_bits(x, w)[0, 0]) == 0x3D80
    x = torch.tensor([[0x70, 0x70, 0x30]], dtype=torch.uint8)     # 8 significant bits: all agree
    assert int(at.ipt_linear_bits(x, w, variant="numba")[0, 0]) == int(at.ipt_linear_bits(x, w, variant="model47")[0, 0])


def _codes(rng, shape, cancel=False):
    c = rng.integers(0, 256, size=shape, dtype=np.int64)
    e, m = (c >> 3) & 15, c & 7
    c = np.where(((e == 0) & (m != 0)) | ((e == 15) & (m == 7)), 0, c)
    if cancel:   # pairs of equal magnitude, opposite sign, plus small codes: drives tile totals toward zero
        half = shape[-1] // 2
        c[..., half:2 * half] = c[..., :half] ^ 0x80
        small = rng.integers(0x08, 0x30, size=shape, dtype=np.int64)
        c = np.where(rng.random(shape) < 0.15, small, c)
    return c.astype(np.uint8)


@pytest.mark.skipif(not ROOT, reason="ATLAS_REF_ROOT not set")
@pytest.mark.parametrize("unit", ["mxu1", "mxu0"])
@pytest.mark.parametrize("compile", [False, True])
def test_contraction_matches_atlas_ref(unit, compile):
    sys.path.insert(0, ROOT)
    from atlas_ref import contractions as ref
    import hw_fixtures as hw
    rng = np.random.default_rng(1234)
    checked = 0
    for K in (1, 5, 31, 32, 33, 64, 100, 129):
        for cancel in (False, True):
            M, N = 6, 9
            a = _codes(rng, (M, K), cancel)
            b = _codes(rng, (N, K), cancel)
            b[:, : min(K, 3)] |= 0x78 & b[:, : min(K, 3)] | 0x70      # exponent-15 / large codes
            b = np.where(((b >> 3) & 15 == 15) & (b & 7 == 7), 0x77, b).astype(np.uint8)
            a[0] = 0                                                   # a zero row
            ax = rng.integers(-12, 3, size=(M, 1)).astype(np.int64)
            bx = rng.integers(-12, 3, size=(N, 1)).astype(np.int64)
            bias = (torch.randn(N) * 4).to(torch.bfloat16)
            for use_bias in (False, True):
                bias_np = bias.view(torch.int16).numpy().view(np.uint16) if use_bias else None
                want = ref.contraction_nt(a, ax, b, bx, unit=unit, bias_bf16=bias_np).output_bf16
                got = hw.contraction_nt(torch.from_numpy(a), torch.from_numpy(ax), torch.from_numpy(b),
                                        torch.from_numpy(bx), unit, bias if use_bias else None, compile, "model47")
                np.testing.assert_array_equal(got.numpy().astype(np.uint16), want)
                checked += want.size
    # short tile totals (fewer than 8 significant bits), in the first tile and carried into a second one
    a = np.zeros((3, 40), dtype=np.uint8)
    b = np.zeros((2, 40), dtype=np.uint8)
    a[:, :3] = [[0x70, 0x70, 0x28], [0x70, 0x70, 0x20], [0x70, 0x70, 0x30]]
    b[:, :3] = [[0x70, 0xF0, 0x28], [0x70, 0xF0, 0x20]]
    a[:, 35], b[:, 35] = 0x38, 0x38
    for k in (3, 40):
        want = ref.contraction_nt(a[:, :k], np.zeros((3, 1), np.int64), b[:, :k], np.zeros((2, 1), np.int64),
                                  unit=unit).output_bf16
        args = (torch.from_numpy(a[:, :k].copy()), torch.zeros(3, 1, dtype=torch.int64),
                torch.from_numpy(b[:, :k].copy()), torch.zeros(2, 1, dtype=torch.int64), unit, None, compile)
        np.testing.assert_array_equal(hw.contraction_nt(*args, "model47").numpy().astype(np.uint16), want)
        # default: short totals normalized (npu-model 19caa5c); atlas_ref (pin 47b8599) differs only there
        got = hw.contraction_nt(*args).numpy().astype(np.uint16)
        if unit == "mxu1":
            if k == 3:
                assert int(got[0, 0]) == 0x3D80 and int(want[0, 0]) == 0x3DC0
        else:
            np.testing.assert_array_equal(got, want)
    # batched, broadcast lead dims, per-head exponents (the attention operands)
    a = _codes(rng, (2, 4, 7, 40), True)
    b = _codes(rng, (2, 1, 5, 40), True)
    ax = rng.integers(-10, 0, size=(2, 4, 1, 1)).astype(np.int64)
    bx = rng.integers(-10, 0, size=(2, 1, 1, 1)).astype(np.int64)
    want = ref.contraction_nt(a, ax, b, bx, unit=unit).output_bf16
    got = hw.contraction_nt(torch.from_numpy(a), torch.from_numpy(ax), torch.from_numpy(b),
                            torch.from_numpy(bx), unit, None, compile, "model47")
    np.testing.assert_array_equal(got.numpy().astype(np.uint16), want)
    assert checked > 0
