import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from atlas_quant import hw_bf16_to_e4m3, hw_e4m3_to_float, scale_exp_for  # noqa: E402
from atlas_torchao import AtlasQConfig, AtlasQuantLinear  # noqa: E402

BIAS = torch.tensor(127)  # unit scale


def _all_bf16():
    bits = torch.arange(0, 1 << 16, dtype=torch.int32).to(torch.int16)
    x = bits.view(torch.bfloat16)
    return x[torch.isfinite(x)]


def _hw_roundtrip(x):
    return hw_e4m3_to_float(hw_bf16_to_e4m3(x, BIAS))


def test_hw_cast_matches_torch_in_normal_range():
    x = _all_bf16()
    a = x.abs().float()
    x = x[(a >= 2.0**-6) & (a <= 448.0)]
    assert x.numel() > 1000
    ref = x.to(torch.float8_e4m3fn).float()
    assert torch.equal(_hw_roundtrip(x), ref)


def test_hw_cast_flushes_below_min_normal():
    x = _all_bf16()
    # Values within half an E4M3 step of 2^-6 round up to 2^-6 (mantissa carry); everything
    # below that flushes to zero.
    a = x.abs().float()
    below = x[a < 2.0**-6 * (1 - 1 / 32)]
    assert torch.count_nonzero(_hw_roundtrip(below)) == 0
    near = x[(a >= 2.0**-6 * (1 - 1 / 32)) & (a < 2.0**-6)]
    assert (_hw_roundtrip(near).abs() == 2.0**-6).all()


def test_scale_exp_keeps_weights_in_range():
    torch.manual_seed(0)
    for scale in (1e-4, 1e-2, 1.0, 50.0, 3e4):
        w = (torch.randn(64, 64) * scale).to(torch.bfloat16)
        s = scale_exp_for(w)
        assert (w.abs().float().max() / torch.exp2(s.float())) <= 448.0
    assert (torch.zeros(4, 4).abs().max() / torch.exp2(scale_exp_for(torch.zeros(4, 4)).float())) <= 448.0


def _linear():
    torch.manual_seed(0)
    return nn.Linear(64, 32, bias=True).to(torch.bfloat16)


def test_hw_activation_path_runs_and_is_finite():
    lin = _linear()
    q = AtlasQuantLinear(lin, AtlasQConfig(input_scale_exp=0, hw_activation_cast=True))
    y = q(torch.randn(5, 64).to(torch.bfloat16))
    assert y.shape == (5, 32) and torch.isfinite(y.float()).all()


def test_hw_and_torch_activation_differ_only_via_tiny_values():
    lin = _linear()
    s = 2
    q_hw = AtlasQuantLinear(lin, AtlasQConfig(input_scale_exp=s, hw_activation_cast=True))
    q_t = AtlasQuantLinear(lin, AtlasQConfig(input_scale_exp=s, hw_activation_cast=False))
    x = (torch.randn(8, 64) * 2**s).to(torch.bfloat16)
    # inputs all in the normal range: outputs identical
    x_big = torch.where((x.float() / 2**s).abs() < 2.0**-6, torch.full_like(x, 2.0**s), x)
    assert torch.equal(q_hw(x_big), q_t(x_big))
    # inputs with tiny values: hw path equals the torch path with those values zeroed
    x_tiny = x_big.clone()
    x_tiny[:, :8] = (2.0**-8 * 2**s)
    x_zeroed = x_tiny.clone()
    x_zeroed[:, :8] = 0
    assert torch.equal(q_hw(x_tiny), q_t(x_zeroed))
    assert not torch.equal(q_t(x_tiny), q_hw(x_tiny))


def test_weight_attribute_reports_original_dtype_but_buffer_is_fp8():
    q = AtlasQuantLinear(_linear(), AtlasQConfig())
    assert q.weight.dtype == torch.bfloat16 and q.weight.shape == (32, 64)
    assert q.state_dict()["weight"].dtype == torch.float8_e4m3fn
    assert dict(q.named_buffers())["weight"].dtype == torch.float8_e4m3fn
