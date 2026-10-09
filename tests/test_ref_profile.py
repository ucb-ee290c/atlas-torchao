import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from refutil import need_refs

import quant_modes
import ref_loader
from ref_profile import RefQuantLinear


@pytest.fixture(scope="module")
def hw():
    need_refs()
    return ref_loader.import_ref("pi0_inout.hw_float_ops")


def _oracle_dequant(hw, t):
    q, s = hw.quantize_bf16_to_e4m3(t)
    return q.view(torch.float8_e4m3fn).float() * 2.0 ** s


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_reference_linear_matches_oracle_cast(hw, dtype):
    torch.manual_seed(0)
    lin = nn.Linear(48, 24).to(dtype)
    x = (torch.randn(2, 5, 48) * 7).to(torch.float32)            # float32 activations into the layer
    y = RefQuantLinear(lin)(x)
    xd = _oracle_dequant(hw, x.to(dtype)).to(dtype)
    wd = _oracle_dequant(hw, lin.weight.detach()).to(dtype)
    ref = F.linear(xd, wd) + lin.bias
    assert y.dtype == dtype and torch.equal(y, ref)


def test_quantize_reference_covers_fp32_layers_and_skips():
    m = nn.Sequential()
    m.add_module("a", nn.Linear(8, 8).bfloat16())
    m.add_module("b", nn.Linear(8, 8).float())
    m.add_module("lm_head", nn.Linear(8, 8).bfloat16())
    rep = quant_modes.quantize(m, "reference", skip=["lm_head"])
    assert rep == {"planned": 2, "quantized": 2, "untouched": []}
    assert isinstance(m.a, RefQuantLinear) and isinstance(m.b, RefQuantLinear)
    assert type(m.lm_head) is nn.Linear
    assert m.b.weight.dtype == torch.float32                      # callers casting by weight dtype still work
