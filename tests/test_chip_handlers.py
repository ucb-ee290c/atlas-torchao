"""Op handlers of the chip router: semantics of torch, numerics of the VPU sequences."""
import torch

import chip_coverage as cc

A = torch.ops.aten


def router():
    return cc.ChipRouter(cc.Coverage(), mxu_impl="ref_cpu")


def run(r, func, *args, **kwargs):
    name = func.overloadpacket.__name__
    unit, h, why = r.classify(func, name, args, kwargs, None)
    assert h is not None, (name, unit, why)
    return h.fn(r, func, name, args, kwargs)


def close(a, b, rtol=2e-2, atol=1e-2):
    return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)


def test_add_alpha_and_inplace():
    r = router()
    x, y = torch.randn(4, 8).bfloat16(), torch.randn(4, 8).bfloat16()
    out = run(r, A.add.Tensor, x, y, alpha=2)
    assert close(out, x.float() + 2 * y.float())
    z = x.clone()
    ret = run(r, A.add_.Tensor, z, y)
    assert ret is z and close(z, x.float() + y.float())


def test_broadcast_both_sides_and_dtype_promotion():
    r = router()
    a = torch.randn(1, 5)
    b = torch.randn(3, 1).bfloat16()
    out = run(r, A.mul.Tensor, a, b)
    assert out.shape == (3, 5) and out.dtype == torch.float32
    assert close(out, a * b.float())


def test_reductions():
    r = router()
    x = torch.randn(2, 3, 70)
    assert close(run(r, A.mean.dim, x, [-1], True), x.mean(-1, keepdim=True))
    assert close(run(r, A.mean.dim, x, [1, 2]), x.mean((1, 2)), atol=3e-2)
    assert close(run(r, A.sum.dim_IntList, x, [0]), x.sum(0), atol=5e-2)
    assert torch.equal(run(r, A.amax.default, x, [-1]), x.bfloat16().float().amax(-1))
    assert torch.equal(run(r, A.amin.default, x, [-1]), x.bfloat16().float().amin(-1))


def test_activations_and_norms():
    r = router()
    x = torch.randn(4, 64).bfloat16()
    assert close(run(r, A.sigmoid.default, x), torch.sigmoid(x.float()))
    assert close(run(r, A.silu.default, x), torch.nn.functional.silu(x.float()), atol=3e-2)
    assert close(run(r, A.gelu.default, x, approximate="tanh"),
                 torch.nn.functional.gelu(x.float(), approximate="tanh"), atol=3e-2)
    w, b = torch.ones(64).bfloat16(), torch.zeros(64).bfloat16()
    y, _, _ = run(r, A.native_layer_norm.default, x, [64], w, b, 1e-5)
    assert close(y, torch.nn.functional.layer_norm(x.float(), [64]), atol=5e-2)
    s = run(r, A._softmax.default, x.float(), -1, False)
    assert close(s, torch.softmax(x.float(), -1), atol=1e-2)


def test_pow_clamp_abs_log():
    r = router()
    x = torch.rand(4, 32) + 0.5
    assert close(run(r, A.pow.Tensor_Scalar, x, 2), x ** 2)
    assert close(run(r, A.pow.Tensor_Scalar, x, -0.5), x ** -0.5)
    # c^x = 2^(x log2 c): x log2 c is a BF16 product (ulp 2^-4 near 16), so compare exponents
    xb = x.bfloat16().float()
    got = run(r, A.pow.Scalar, 10000.0, xb)
    assert ((got.log2() - (10000.0 ** xb).log2()).abs() < 0.1).all()
    assert close(run(r, A.clamp.default, x, 0.7, 1.2), x.clamp(0.7, 1.2))
    assert close(run(r, A.abs.default, x - 1), (x - 1).abs())
    assert close(run(r, A.log.default, x), x.log(), atol=2e-2)


def test_matmuls_on_mxu_and_conv():
    r = router()
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2, 16, 32, generator=g), torch.randn(2, 32, 8, generator=g)
    y = run(r, A.bmm.default, a, b)
    assert y.shape == (2, 16, 8) and close(y, a @ b, rtol=0.15, atol=0.6)
    assert any(o["unit"] == "MXU" for o in r.cov.report()["ops"])
    x = torch.randn(1, 3, 8, 8, generator=g)
    w = torch.randn(4, 3, 2, 2, generator=g)
    bias = torch.randn(4, generator=g)
    out = run(r, A.convolution.default, x, w, bias, [2, 2], [0, 0], [1, 1], False, [0, 0], 1)
    ref = torch.nn.functional.conv2d(x, w, bias, stride=2)
    assert out.shape == ref.shape and close(out, ref, rtol=0.15, atol=0.6)
    out1 = run(r, A.convolution.default, x, w, None, [2], [0], [1], False, [0], 1)
    assert out1.shape == ref.shape
    # K = 1: an outer product on the VPU
    o = run(r, A.bmm.default, torch.arange(4.0).reshape(1, 4, 1), torch.ones(1, 1, 3))
    assert torch.equal(o, torch.arange(4.0).reshape(1, 4, 1).expand(1, 4, 3))
