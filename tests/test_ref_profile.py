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
    assert (rep["planned"], rep["quantized"], rep["untouched"]) == (2, 2, [])
    assert isinstance(m.a, RefQuantLinear) and isinstance(m.b, RefQuantLinear)
    assert type(m.lm_head) is nn.Linear
    assert m.b.weight.dtype == torch.float32                      # callers casting by weight dtype still work


def _raw_ref_fm(mxu):
    if mxu == "ipt":
        return ref_loader.import_ref("funct_models_ipt.ipt_numba.ipt_rtl_linear").IPTLinearRTLFunction()
    return ref_loader.import_ref("func_models_sa.systolic_c.systolic_array_rtl_linear").SARTLLinearFunction()


@pytest.mark.parametrize("impl", ["ref_cpu", "torch"])
@pytest.mark.parametrize("mxu", ["ipt", "sa"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_linear_parity(hw, mxu, dtype, impl):
    """reference_hw == the reference QuantLinear(functional_model=...) bit for bit
    (smolvla_inout quant_linear.py:133-175), K with a tail tile, bias, 3-D float32 input."""
    QuantLinear = ref_loader.import_ref("smolvla_inout.quant_linear").QuantLinear
    import accum_backends as ab
    from ref_profile import quantize_reference

    torch.manual_seed(3)
    lin = nn.Linear(70, 33).to(dtype)
    x = torch.randn(2, 4, 70) * torch.exp2(torch.randn(2, 4, 70) * 2)
    ref_layer = QuantLinear(lin, functional_model=_raw_ref_fm(mxu))
    m = nn.Sequential()
    m.add_module("lin", lin)
    quantize_reference(m, fm_factory=lambda: ab.get_accumulator(mxu, impl))
    with torch.no_grad():
        want = ref_layer(x)
        got = m.lin(x)
        got_again = m.lin(x)                    # weight cache reused
    assert got.dtype == want.dtype == dtype
    assert torch.equal(got, want) and torch.equal(got_again, want)


def test_fm_idempotent():
    """Every encoder inside the functional models maps each E4M3 value the vpack encoder can
    emit to itself, as quant_linear.py:158-160 assumes: ipt_numba's _float_to_e4m3_byte_scalar
    (ipt_rtl_linear.py:43-78), the C shim_float_to_e4m3 (ipt_rtl_linear_c.py:240-287, also used
    by systolic_c, systolic_array_rtl_linear.py:201,279) and the numba port of it."""
    need_refs()
    import ref_numerics as rn

    codes = torch.tensor([b for b in range(256) if ((b >> 3) & 0xF) != 0 and (b & 0x7F) != 0x7F] + [0],
                         dtype=torch.uint8)
    vals = rn.e4m3_unpack(codes)
    numba_lin = ref_loader.import_ref("funct_models_ipt.ipt_numba.ipt_rtl_linear")
    c_lin = ref_loader.import_ref("funct_models_ipt.ipt_c.ipt_rtl_linear_c")
    assert torch.equal(numba_lin.float_to_e4m3_bytes(vals), codes)
    assert torch.equal(torch.from_numpy(c_lin.float_to_e4m3_bytes_c(vals.numpy())), codes)
    assert torch.equal(torch.from_numpy(c_lin.float_to_e4m3_bytes_numba(vals.numpy())), codes)
