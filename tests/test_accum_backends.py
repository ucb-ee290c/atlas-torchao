import pytest
import torch

from refutil import e4m3_values, need_refs, with_ties

import accum_backends as ab


@pytest.fixture(scope="module", autouse=True)
def _refs():
    need_refs()


@pytest.mark.parametrize("K", [32, 77, 160])
def test_ipt_numba_and_c_agree_on_random(K):
    g = torch.Generator().manual_seed(K)
    x = with_ties(e4m3_values((9, K), g), g)
    w = with_ties(e4m3_values((37, K), g, spread=1.0, scale=0.05), g)
    a = ab.get_accumulator("ipt", "ref_cpu")(x, w)
    b = ab.get_accumulator("ipt", "ref_cpu_c")(x, w)
    assert torch.equal(a, b)


def test_ipt_lane_count_does_not_change_results():
    """Plan section 3: ipt_numba uses 32 lanes, ipt_c defaults to 16 (ipt_rtl_linear_c.py:529-530)."""
    g = torch.Generator().manual_seed(1)
    x, w = e4m3_values((5, 100), g), e4m3_values((40, 100), g, scale=0.1)
    m = ab.ref_loader.import_ref("funct_models_ipt.ipt_c.ipt_rtl_linear_c")
    y16 = m.CIPTLinearRTLFunction(num_lanes=16)(x, w, None)
    y32 = m.CIPTLinearRTLFunction(num_lanes=32)(x, w, None)
    assert torch.equal(y16, y32)
    assert torch.equal(y32, ab.get_accumulator("ipt", "ref_cpu")(x, w))


def test_ipt_c_double_rounds_where_numba_and_rtl_round_once():
    """One 32-element tile whose aligned total is 2^24 + 2^16 + 1 (anchor 23): one RNE rounding
    (ipt_numba _numba_kernels.py:156-167; RTL FPUtils.scala:228-236 INToRecFN) gives
    2^16 * (1 + 2^-7); ipt_c's (float)int then BF16 (converters.h:244-254) gives 2^16."""
    x = torch.zeros(1, 32)
    w = torch.zeros(1, 32)
    x[0, :3] = torch.tensor([256.0, 16.0, 2.0**-4])
    w[0, :3] = torch.tensor([256.0, 16.0, 2.0**-4])
    numba = ab.get_accumulator("ipt", "ref_cpu")(x, w)
    c = ab.get_accumulator("ipt", "ref_cpu_c")(x, w)
    assert numba.item() == 2.0**16 * (1 + 2.0**-7)
    assert c.item() == 2.0**16


def test_batched_form_equals_per_index_calls():
    g = torch.Generator().manual_seed(2)
    x, w = e4m3_values((2, 3, 6, 40), g), e4m3_values((2, 3, 5, 40), g, scale=0.1)
    for mxu in ab.MXUS:
        fm = ab.get_accumulator(mxu, "ref_cpu")
        y = fm(x, w)
        assert y.shape == (2, 3, 6, 5)
        for i in range(2):
            for j in range(3):
                assert torch.equal(y[i, j], fm(x[i, j], w[i, j]))


def test_both_mxus_track_fp64():
    g = torch.Generator().manual_seed(3)
    x, w = e4m3_values((8, 512), g, spread=1.0), e4m3_values((16, 512), g, spread=1.0, scale=0.05)
    exact = x.double() @ w.double().T
    for mxu in ab.MXUS:
        y = ab.get_accumulator(mxu, "ref_cpu")(x, w).double()
        assert ((y - exact).norm() / exact.norm()) < 5e-2   # per-step BF16 rounding (SA) loses more than the tree


def test_unknown_names_raise():
    with pytest.raises(ValueError):
        ab.get_accumulator("tpu", "ref_cpu")
    with pytest.raises(ValueError):
        ab.get_accumulator("sa", "ref_cpu_c")
