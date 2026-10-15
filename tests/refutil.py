"""Shared helpers for the reference-numerics tests."""
import pytest
import torch

import ref_loader


def need_refs():
    bad = ref_loader.check_pins()
    if bad:
        pytest.skip(f"reference submodules not initialised at their pins: {bad}")
    try:
        import numba  # noqa: F401
    except ImportError:
        pytest.skip("numba not installed (pip install -e .[reference])")


def e4m3_values(shape, gen, spread=3.0, zero_frac=0.05, scale=1.0):
    """Random float32 tensors holding exact normal E4M3 values (and some zeros), as the vpack
    encoder produces them: decoded with .view(float8_e4m3fn).float()."""
    x = torch.randn(shape, generator=gen) * torch.exp2(torch.randn(shape, generator=gen) * spread) * scale
    x = x.clamp(-448, 448)
    q = x.to(torch.float8_e4m3fn).float()
    q = torch.where(q.abs() < 2.0**-6, torch.zeros_like(q), q)       # no subnormals
    q = torch.where(torch.rand(shape, generator=gen) < zero_frac, torch.zeros_like(q), q)
    return q


def with_ties(x, gen, frac=0.2):
    """Repeat values within each row so equal magnitudes (and exact cancellations) are common."""
    idx = torch.randint(0, x.shape[-1], x.shape, generator=gen)
    rep = torch.gather(x, -1, idx)
    sign = torch.where(torch.rand(x.shape, generator=gen) < 0.5, -1.0, 1.0)
    return torch.where(torch.rand(x.shape, generator=gen) < frac, rep * sign, x)
