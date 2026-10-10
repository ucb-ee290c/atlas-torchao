"""algo rotation: folds and online transforms are lossless without a cast, the +-1 / power-of-two
operands are exact in E4M3, the mxu2 two-term split matches its definition, refusals are clear,
and the algorithm runs on the reference and chip steps with its extra work recorded."""
import copy
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import algos
import ref_numerics as rn
from algos import rotation as rot
from refutil import need_refs
from test_ladder import FAST, tiny_llama

IDS = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))


# ------------------------------------------------------------------------------ Hadamard matrices
@pytest.mark.parametrize("m", [2, 4, 12, 20, 24, 44, 64])
def test_hadamard_orders(m):
    H = rot.pm1(m)
    assert torch.equal(H @ H.T, m * torch.eye(m, dtype=torch.float64))
    assert set(H.unique().tolist()) == {-1.0, 1.0}


def test_factorisation():
    assert rot.hadamard_factors(5632) == ([44, 128], True)          # TinyLlama intermediate
    assert rot.hadamard_factors(2048) == ([2048], True)
    assert rot.hadamard_factors(96) == ([12, 8], True)
    assert rot.hadamard_factors(15) == ([15, 1], False)              # 15 heads: nothing to mix
    assert rot.hadamard_factors(2560) == ([20, 128], True)           # SmolVLA-size MLP
    assert rot.hadamard_factors(30) == ([15, 2], False)


# ------------------------------------------------------------------------------ exactness
def test_pm1_and_po2_exact_in_e4m3():
    """+-1 operands and the 2^-k / 2^-4 factors survive the vpack encoder exactly; 1/sqrt(n) for an
    odd log2 n does not (and for n = 5632 it flushes), which is why it is never an operand."""
    H = rot.pm1(44).to(torch.bfloat16)
    assert torch.equal(rn.e4m3_unpack(rn.e4m3_vpack(H, 0)), H.float())
    for k in range(0, 7):
        v = torch.tensor([2.0 ** -k, -(2.0 ** -k)], dtype=torch.bfloat16)
        assert torch.equal(rn.e4m3_unpack(rn.e4m3_vpack(v, 0)), v.float())
    inv = torch.tensor([1 / math.sqrt(2048)], dtype=torch.bfloat16)
    assert not torch.equal(rn.e4m3_unpack(rn.e4m3_vpack(inv, 0)), inv.float())
    assert rn.e4m3_unpack(rn.e4m3_vpack(torch.tensor([1 / math.sqrt(5632)]), 0)).item() == 0.0


@pytest.mark.parametrize("impl", rot.IMPLS)
def test_spec_fold_is_exact(impl):
    """W' u == W x for u = c M x (the exact transform each impl approximates)."""
    torch.manual_seed(0)
    for spec in (rot._down_spec("d", 5632, impl, 32)[0], rot._o_spec("o", 2048, 64, impl, 32)[0]):
        W = torch.randn(48, spec.n, dtype=torch.float64)
        x = torch.randn(5, spec.n, dtype=torch.float64)
        u = spec.c * spec.kron_orth.apply(x)
        Wf = spec.fold_weight(W)
        assert torch.allclose(u @ Wf.T, x @ W.T, rtol=1e-9, atol=1e-9)
        # the +-1 form times 2^-k is that exact transform (fp32 applies M itself)
        if impl != "fp32":
            assert torch.allclose(spec.kron_pm1.apply(x) * 2.0 ** -spec.k, u, rtol=1e-12, atol=1e-12)


def test_vpu_butterfly_matches_pm1_up_to_truncation():
    torch.manual_seed(0)
    spec = rot._down_spec("d", 5632, "vpu", 32)[0]
    assert spec.kron_pm1.shape == (11, 16, 32) and spec.k == 2 and spec.c == 1.0
    x = torch.randn(64, 5632).to(torch.bfloat16)
    y = rot.vpu_butterfly(x, spec.kron_pm1.shape, spec.mix_axis)
    ref = spec.kron_pm1.apply(x.double())
    err = ((y.double() - ref).norm() / ref.norm()).item()
    assert 1e-4 < err < 0.012                                        # 4 truncating stages
    # every output is a BF16 value whose low 16 bits are zero (truncated, not rounded)
    assert torch.equal(y.contiguous().view(torch.int32) & 0xFFFF, torch.zeros_like(y, dtype=torch.int32))


def test_mxu2_two_term_split():
    """hi + lo equals its definition, is far closer to x than one cast, and the +-1 matmul of
    hi + lo equals the sum of the two accumulations (vmatmul.acc)."""
    torch.manual_seed(0)
    a = algos.make("rotation:impl=mxu2")
    x = (torch.randn(32, 256) * torch.logspace(-2, 1, 256)).to(torch.bfloat16)
    s = rn.po2_scale_exp(x)
    hi = rn.dequantize(rn.e4m3_vpack(x, s), s)
    lo = rn.dequantize(rn.e4m3_vpack((x.float() - hi).to(torch.bfloat16), s - 4), s - 4)
    two = a.two_term("l", x)
    assert torch.equal(two, hi + lo)
    e1 = ((hi - x.float()).norm() / x.float().norm()).item()
    e2 = ((two - x.float()).norm() / x.float().norm()).item()
    assert e2 < e1 / 6
    H = rot.Kron((256,), [rot.sylvester(256)])
    hi, lo = hi.double(), lo.double()                               # exact sums of E4M3 values
    assert torch.equal(H.apply(hi) + H.apply(lo), H.apply(hi + lo))


# ------------------------------------------------------------------------------ lossless folds
def _float_forward(model, algo, ids):
    """The model with every online layer computing F.linear(f(x), W') in float, no FP8 cast."""
    handles = []
    for n, m in model.named_modules():
        spec = algo.specs.get(n)
        if spec is None:
            continue
        Wf = spec.fold_weight(m.weight.detach().double())

        def fwd(mod, args, out, spec=spec, Wf=Wf):
            x = args[0].double()
            return F.linear(spec.c * spec.kron_orth.apply(x), Wf).to(out.dtype)
        handles.append(m.register_forward_hook(fwd))
    try:
        with torch.no_grad():
            return model(ids).logits
    finally:
        for h in handles:
            h.remove()


@pytest.mark.parametrize("spec", ["rotation", "rotation:impl=vpu", "rotation:impl=mxu2",
                                  "rotation:r1=true", "rotation:r1=true,seed=3,impl=mxu2"])
def test_lossless_without_cast(spec):
    m0 = tiny_llama(2).float()
    m1 = copy.deepcopy(m0)
    a = algos.make(spec)
    rec = a.prepare(m1, "ref_linear")
    assert rec["online_layers"] >= 2
    with torch.no_grad():
        y0 = m0(IDS).logits
    y1 = _float_forward(m1, a, IDS)
    assert torch.allclose(y1, y0, rtol=1e-4, atol=1e-4), (y1 - y0).abs().max()


def test_r1_seed_is_input_side():
    """Q = D H: a different seed changes the rotated embedding magnitudes (H D would not)."""
    q0, q1 = rot.rotation_q(64, 0), rot.rotation_q(64, 1)
    x = torch.zeros(1, 64, dtype=torch.float64)
    x[0, :3] = torch.tensor([3.0, 1.0, 0.5])
    assert not torch.equal((x @ q0).abs(), (x @ q1).abs())
    assert torch.allclose(q1 @ q1.T, torch.eye(64, dtype=torch.float64))


# ------------------------------------------------------------------------------ refusals
class _Cfg:
    def __init__(self, mt):
        self.model_type = mt


class _LNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg("llama")
        self.norm = nn.LayerNorm(8)


class GemmaRMSNorm(nn.Module):                     # the openpi adaptive form (cond_dim / dense)
    def __init__(self):
        super().__init__()
        self.cond_dim = 4
        self.dense = nn.Linear(4, 24)


class _AdaptiveModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg("gemma")
        self.norm = GemmaRMSNorm()


class AdaLNModulation(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 8)


class _AdaLNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _Cfg("dit")
        self.mod = AdaLNModulation()


@pytest.mark.parametrize("model,needle", [(_LNModel, "LayerNorm"), (_AdaptiveModel, "adaptive"),
                                          (_AdaLNModel, "adaLN"), (lambda: nn.Linear(4, 4), "Llama only")])
def test_r1_refusals(model, needle):
    with pytest.raises(rot.RotationRefused, match=needle):
        algos.make("rotation:r1=true").prepare(model(), "ref_linear")


def test_bad_params():
    with pytest.raises(ValueError, match="impl"):
        algos.make("rotation:impl=gpu")
    with pytest.raises(ValueError, match="online"):
        algos.make("rotation:online=up")


# ------------------------------------------------------------------------------ end to end
@pytest.mark.parametrize("mode", ["ref_linear", "reference_hw", "chip"])
@pytest.mark.parametrize("spec", ["rotation", "rotation:impl=vpu", "rotation:impl=mxu2,static=true,r1=true"])
def test_quantize_end_to_end(mode, spec):
    need_refs()
    m0 = tiny_llama()
    m1 = copy.deepcopy(m0)
    a = algos.make(spec)
    calib = lambda: ([torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(7))],  # noqa: E731
                     lambda mm, b: mm(b), {"source": "test", "n_samples": 1, "seed": 7})
    rep = algos.quantize(m1, mode, ["lm_head"], dict(FAST), algo=a, calib=calib)
    with torch.no_grad():
        y0, y1 = m0(IDS).logits.float(), m1(IDS).logits.float()
    assert torch.isfinite(y1).all()
    assert ((y1 - y0).norm() / y0.norm()).item() < 0.1
    rec = rep["algo"]
    assert rec["module_hooks_applied"] and rec["online_cost"]
    assert rec["granularity"]["online_transforms"]
    if "static" in spec:
        assert rec["calibration"]["result"]["online_layers_observed_after_transform"] >= 1
        assert any(k.endswith("static") for k in rec["granularity"]["activation"])
    if mode == "chip":
        ops = [o["op"] for o in rep["coverage"]["ops"] if o["op"].startswith("rotation.")]
        assert "rotation.online_hadamard" in ops


def test_static_observes_rotated_input():
    """The calibration amax of an online layer is the amax AFTER its transform."""
    m = tiny_llama()
    a = algos.make("rotation:static=true,online=down")
    batch = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(7))
    seen = {}
    name = "model.layers.0.mlp.down_proj"
    a.calibrate(m, [batch], lambda mm, b: mm(b))
    h = dict(m.named_modules())[name].register_forward_pre_hook(
        lambda mod, args: seen.setdefault("x", args[0].detach().to(torch.bfloat16)))
    with torch.no_grad():
        m(batch)
    h.remove()
    u = a._transform(a.specs[name], seen["x"], calib=True)
    assert torch.equal(a.static_exp[name], rn.po2_scale_exp(u))
    assert m._rotation_prepared is a.prepare(m, "ref_linear")       # prepare is not re-run
