"""Every aten op the three models execute has a unit (or an explicit host reason)."""
import torch

import chip_coverage as cc
import op_unit_map as M

# Op census (aten overload packets) of one pass: TinyLlama seqlen 128 (eager), SmolVLA one frame,
# Multi-Task DiT one frame (CLIP towers eager). Taken with a TorchDispatchMode on the unpatched models.
CENSUS = {
    "llama": {"_local_scalar_dense", "_log_softmax", "_softmax", "_to_copy", "_unsafe_view", "add", "alias",
              "all", "arange", "bmm", "cat", "clone", "constant_pad_nd", "cos", "cumsum", "embedding", "eq",
              "expand", "le", "lift_fresh", "mean", "mm", "mul", "ne", "neg", "nll_loss_forward", "pow",
              "rsqrt", "scalar_tensor", "select", "silu", "sin", "slice", "sub", "t", "transpose",
              "unsqueeze", "view", "where"},
    "smolvla": {"_local_scalar_dense", "_scaled_dot_product_flash_attention_for_cpu", "_softmax", "_to_copy",
                "_unsafe_view", "add", "add_", "addmm", "alias", "all", "arange", "bitwise_and", "bmm",
                "bucketize", "cat", "clamp", "clone", "constant_pad_nd", "convolution", "copy_", "cos", "cumsum",
                "div", "embedding", "empty_like", "expand", "full", "gelu", "index", "index_put_", "le",
                "lift_fresh", "linspace", "mean", "min", "mm", "mul", "mul_", "native_layer_norm", "ones",
                "permute", "pow", "randn", "reciprocal", "rsqrt", "scalar_tensor", "select", "silu", "sin",
                "slice", "split", "sub", "sum", "t", "transpose", "unsqueeze", "upsample_bilinear2d", "view",
                "where"},
    "dit": {"_local_scalar_dense", "_scaled_dot_product_flash_attention_for_cpu", "_softmax", "_to_copy",
            "_unsafe_view", "_upsample_bilinear2d_aa", "add", "addmm", "alias", "arange", "argmax",
            "bitwise_and", "bmm", "cat", "clamp", "clone", "convolution", "cos", "div", "embedding", "eq",
            "exp", "expand", "full", "ge", "gelu", "gt", "index", "le", "lift_fresh", "mul",
            "native_layer_norm", "neg", "new_ones", "nonzero", "permute", "pow", "randn", "rsub",
            "scalar_tensor", "select", "sigmoid", "silu", "sin", "slice", "split", "stack", "sub", "t",
            "transpose", "unbind", "unsqueeze", "view", "where"},
}


def known(name):
    return (name in M.DATA or name in M.EVAL_ONLY or name in M.INT_SCALAR or name in M.HOST_REASONS
            or name in cc._VPU_HANDLERS)


def test_census_ops_are_mapped():
    missing = {m: sorted(n for n in ops if not known(n)) for m, ops in CENSUS.items()}
    assert all(not v for v in missing.values()), missing


def test_float_compute_ops_have_entries():
    for name in cc._VPU_HANDLERS:
        assert name in M.ATEN, name
        e = M.ATEN[name]
        assert e.unit in ("VPU", "MXU") and e.source and e.status in ("REF", "ISA", "NEW")
        if e.status == "NEW":
            assert e.reason, name


def test_classify_examples():
    r = cc.ChipRouter(cc.Coverage(), mxu_impl="ref_cpu")
    A = torch.ops.aten
    x = torch.randn(4, 8)
    i = torch.arange(4)
    assert r.classify(A.add.Tensor, "add", (x, x), {}, None)[0] == "VPU"
    assert r.classify(A.add.Tensor, "add", (i, i), {}, None)[0] == "scalar"
    assert r.classify(A.view.default, "view", (x, [32]), {}, None)[0] == "data"
    assert r.classify(A.erf.default, "erf", (x,), {}, None)[0] == "host"
    assert r.classify(A.pow.Tensor_Scalar, "pow", (x, 2), {}, None)[0] == "VPU"
    assert r.classify(A.pow.Tensor_Scalar, "pow", (x, 1.7), {}, None)[0] == "host"
    assert r.classify(A.where.self, "where", (x > 0, x, x * 2), {}, None)[0] == "host"
    assert r.classify(A.where.self, "where", (x > 0, x, torch.tensor(-1e9)), {}, None)[0] == "VPU"
    assert r.classify(A.bmm.default, "bmm", (x[None], x.t()[None]), {}, None)[0] == "MXU"
    assert r.classify(A.max.dim, "max", (x, 1), {}, None)[0] == "host"


def test_where_is_additive_mask():
    r = cc.ChipRouter(cc.Coverage(), mxu_impl="ref_cpu")
    x = torch.randn(3, 5).bfloat16()
    m = torch.rand(3, 5) > 0.5
    out = cc._where(r, torch.ops.aten.where.self, "where", (m, x, torch.tensor(-1e30)), {})
    assert torch.equal(out[m], x[m])
    assert (out[~m] < -1e29).all()


def test_mtdit_emulator_protocol():
    em = cc.make_mtdit_op_emulator()
    A = torch.ops.aten
    x = torch.randn(2, 64)
    assert em.classify(A.gelu.default, (x,), {}) == "VPU"
    assert em.classify(A.view.default, (x, [128]), {}) is None
    assert em.classify(A.mm.default, (x, x.t()), {}) is None
    y = em.emulate(A.gelu.default, (x,), {})
    assert y.shape == x.shape and y.dtype == x.dtype
