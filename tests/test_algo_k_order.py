"""k_order (docs/algos/k-order.md): a function-preserving K permutation of Llama MLP down_proj."""
import copy

import pytest
import torch

import mxu.accum_torch as at
import algos
import reference.ref_numerics as rn
from algos import k_order as ko
from refutil import need_refs
from test_algo_atlas_v1 import IDS, _calib
from test_ladder import FAST, tiny_llama

STACK = "gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc"


def _f32(layers=2):
    return tiny_llama(layers=layers).float()


def _logits(m):
    with torch.no_grad():
        return m(IDS).logits


@pytest.mark.parametrize("rule", ["asc", "desc", "rand"])
def test_function_preserved_float(rule):
    m = _f32()
    ref = _logits(m)
    a = algos.make(f"k_order:rule={rule}")
    rec = a.apply(m)
    got = _logits(m)
    assert not torch.equal(got, ref) or rule == "none"      # summation order changed
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-5)
    assert rec["n_layers"] == 2


def test_rule_none_is_exact():
    m = _f32()
    ref = _logits(m)
    w = copy.deepcopy(m.state_dict())
    algos.make("k_order:rule=none").apply(m)
    assert torch.equal(_logits(m), ref)
    assert all(torch.equal(w[k], v) for k, v in m.state_dict().items())


def test_mlp_tensors_follow_one_permutation():
    m = _f32(1)
    mlp = m.model.layers[0].mlp
    g0, u0, d0 = (getattr(mlp, c).weight.detach().clone() for c in ("gate_proj", "up_proj", "down_proj"))
    algos.make("k_order:rule=rand,seed=3").apply(m)
    perm = ko.order_from_importance(torch.zeros(g0.shape[0]), "rand", 3)
    assert torch.equal(mlp.gate_proj.weight, g0[perm]) and torch.equal(mlp.up_proj.weight, u0[perm])
    assert torch.equal(mlp.down_proj.weight, d0[:, perm])
    # attention and gate/up inputs (hidden dim) untouched
    assert mlp.gate_proj.weight.shape == g0.shape


def test_orders_are_by_importance():
    imp = torch.tensor([3., 1., 2., 1.])
    assert ko.order_from_importance(imp, "asc").tolist() == [1, 3, 2, 0]
    assert ko.order_from_importance(imp, "desc").tolist() == [0, 2, 1, 3]
    assert ko.order_from_importance(imp, "none").tolist() == [0, 1, 2, 3]
    assert ko.order_from_importance(imp, "rand", 1).tolist() == ko.order_from_importance(imp, "rand", 1).tolist()


def test_ipt_chain_changes_exact_sum_does_not():
    g = torch.Generator().manual_seed(0)
    K = 256
    x = (torch.randn(8, K, generator=g) * torch.exp2(torch.randn(K, generator=g) * 2)).bfloat16()
    w = (torch.randn(4, K, generator=g) * torch.exp2(torch.randn(K, generator=g))).bfloat16()
    xq, sx = rn.quantize(x)
    wq, sw = rn.quantize(w)
    perm = torch.randperm(K, generator=g)

    def exact(a, b):
        return (rn.e4m3_unpack(a).double() @ rn.e4m3_unpack(b).double().T)

    assert torch.equal(exact(xq, wq), exact(xq[:, perm], wq[:, perm]))      # float64 sum of FP8 products is exact
    a = at.ipt_linear_bits(xq, wq)
    b = at.ipt_linear_bits(xq[:, perm], wq[:, perm])
    assert not torch.equal(a, b)
    assert torch.equal(at.ipt_linear_bits(xq, wq), a)           # deterministic


def test_quantized_values_and_exponents_unchanged():
    need_refs()
    m0, m1 = tiny_llama(layers=2), tiny_llama(layers=2)
    a = algos.make("atlas_v1_awq+k_order:rule=desc")
    r0 = algos.quantize(m0, "ref_conv", ["lm_head"], dict(FAST), algo=algos.make("atlas_v1_awq"), calib=_calib())
    r1 = algos.quantize(m1, "ref_conv", ["lm_head"], dict(FAST), algo=a, calib=_calib())
    # record populated, k_order last
    rec = r1["algo"]
    assert rec["order"][-1] == "k_order" and rec["members"][-1]["stage"] == "permutation"
    ap = rec["members"][-1]["applied_permutation"]
    assert ap["n_layers"] == 2 and len(ap["hash"]) == 16 and rec["members"][-1]["rule"] == "desc"
    assert "k_order" in rec["params"]
    for li in range(2):
        d0, d1 = m0.model.layers[li].mlp.down_proj, m1.model.layers[li].mlp.down_proj
        g0, g1 = m0.model.layers[li].mlp.gate_proj, m1.model.layers[li].mlp.gate_proj
        # same multiset of columns / rows: sorted row contents and exponents identical
        assert torch.equal(d0.weight_scale_exp, d1.weight_scale_exp)           # down rows unmoved
        v0, v1 = d0._dequant_weight().float(), d1._dequant_weight().float()
        assert torch.equal(v0.sort(dim=1).values, v1.sort(dim=1).values)
        assert torch.equal(v0.abs().amax(1), v1.abs().amax(1))                 # row max invariant
        assert torch.equal(g0.weight_scale_exp.sort(0).values, g1.weight_scale_exp.sort(0).values)
        # gate rows moved with their exponents: dequantized row sets agree
        w0, w1 = g0._dequant_weight().float(), g1._dequant_weight().float()
        assert torch.equal(w0.sum(1).sort().values, w1.sum(1).sort().values)
    torch.testing.assert_close(_logits(m1).float(), _logits(m0).float(), rtol=0.1, atol=0.1)


def test_unsupported_structure_raises():
    class Odd(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.down_proj = torch.nn.Linear(4, 4)

    with pytest.raises(ValueError, match="unsupported structure"):
        algos.make("k_order:rule=asc").apply(Odd())
    with pytest.raises(ValueError, match="rule= is required"):
        algos.make("k_order")
    with pytest.raises(ValueError, match="unknown parameters"):
        algos.make("k_order:rule=asc,x=1")


def test_stack_order_puts_it_last():
    s = algos.make("k_order:rule=asc+" + STACK)
    assert [m.name for m in s.members] == ["gptq_awq", "granularity", "k_order"]
