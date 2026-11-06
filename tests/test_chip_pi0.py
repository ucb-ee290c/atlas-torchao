"""The chip profile on Pi0 (openpi PI0Pytorch): op map, the Pi0 vector sequences bit for bit against
the reference author's pi0-quant decompositions on the reference VectorRTLFunctions, the masks and
RoPE decisions, and coverage on a tiny random-weight Pi0 (that part needs the openpi environment,
see tests/pi0_tiny.py; skipped elsewhere).
"""
import math

import pytest
import torch

import chip.chip_coverage as cc
import chip.chip_ops as co
import chip.op_unit_map as M
import reference.ref_loader as ref_loader
from refutil import need_refs

A = torch.ops.aten

# aten overload packets of PI0Pytorch.sample_actions (tests/pi0_tiny.py census, 2 flow steps, eager
# SigLIP attention included). Every one must have a unit or an explicit host reason.
PI0_CENSUS = {
    "_local_scalar_dense", "_scaled_dot_product_flash_attention_for_cpu", "_softmax", "_to_copy",
    "_unsafe_view", "add", "add_", "addmm", "arange", "bitwise_and", "bmm", "cat", "clone", "convolution",
    "cos", "cumsum", "div", "embedding", "expand", "ge", "gelu", "le", "lift_fresh", "linspace", "mean", "mm",
    "mul", "native_layer_norm", "neg", "ones", "permute", "pow", "reciprocal", "rsqrt", "scalar_tensor", "silu",
    "sin", "slice", "sub", "sum", "t", "transpose", "unsqueeze", "view", "where", "normal",
}


def known(name):
    return (name in M.DATA or name in M.EVAL_ONLY or name in M.INT_SCALAR or name in M.HOST_REASONS
            or name in cc._VPU_HANDLERS or name.startswith("_scaled_dot_product"))


def test_pi0_census_ops_are_mapped():
    assert sorted(n for n in PI0_CENSUS if not known(n)) == []
    pi0_rows = [r for r in M.MODEL_OPS if r[0] == "pi0"]
    assert len(pi0_rows) >= 15
    for row in pi0_rows:
        assert row[5] and row[6] in ("REF", "NEW", "ISA", "HOST", "REF/NEW", "REF/ISA"), row


# ---------------------------------------------------------------------------------------------
# Bit-exact against the reference decompositions (pi0-quant pi0_inout/vpu_decomp.py)
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def vrf():
    need_refs()
    return ref_loader.import_ref("funct_models_vector.vector_rtl_forward").VectorRTLFunctions(num_lanes=16)


@pytest.fixture(scope="module")
def dec():
    need_refs()
    return ref_loader.load_file("pi0-quant", "pi0_inout/vpu_decomp.py")


def router():
    return cc.ChipRouter(cc.Coverage("pi0"), mxu_impl="ref_cpu")


def run(r, func, *args, **kwargs):
    name = func.overloadpacket.__name__
    unit, h, why = r.classify(func, name, args, kwargs, None)
    assert h is not None, (name, unit, why)
    return h.fn(r, func, name, args, kwargs)


def all_bf16():
    return co.bf16_bits_to_f32(torch.arange(65536, dtype=torch.int32)).bfloat16()


def same(a, b):
    ba, bb = co.f32_to_bf16_bits(a.float()), co.f32_to_bf16_bits(b.float())
    nan = torch.isnan(a.float()) & torch.isnan(b.float())
    return bool(((ba == bb) | nan).all())


def test_gelu_tanh_exhaustive(vrf, dec):
    """Gemma / SigLIP gelu(approximate='tanh') on bf16 activations: pi0-quant quant_vector.py:377-379
    -> vpu_decomp.py:161-178, on every BF16 pattern."""
    x = all_bf16()
    got = run(router(), A.gelu.default, x, approximate="tanh")
    want = dec._vpu_gelu_tanh(vrf, x).to(x.dtype)
    assert got.dtype == torch.bfloat16 and same(got, want)


def test_rsqrt_exhaustive(vrf, dec):
    """quant_vector.py:385-387 -> vpu_decomp.py:79-81 (vsqrt then vrecip), float32 input as in
    GemmaRMSNorm."""
    x = all_bf16().float()
    assert same(run(router(), A.rsqrt.default, x), dec._vpu_rsqrt(vrf, x))


@pytest.mark.parametrize("e,instr", [(2, "square"), (3, "cube")])
def test_pow_square_cube_exhaustive(vrf, dec, e, instr):
    """torch.square in GemmaRMSNorm dispatches pow.Tensor_Scalar(2): vpu_decomp.py:199-204."""
    x = all_bf16().float()
    assert same(run(router(), A.pow.Tensor_Scalar, x, e), dec._pow_dispatch(vrf, x, e))


def test_silu_exhaustive(vrf, dec):
    """Action-time MLP SiLU: the reference sequence (vpu_decomp.py:181-192) on every BF16 pattern,
    except where the reference's vexp input is -336 or below (x >= 336), where the model wraps (at
    -336 already: sigmoid comes out 0) and the exp floor (NEW) gives silu(x) = x instead."""
    x = all_bf16().float()
    got = run(router(), A.silu.default, x)
    want = dec._vpu_silu(vrf, x).float()
    keep = ~(x >= 336)
    assert same(got[keep], want[keep])
    big = x >= 336
    assert torch.equal(got[big & torch.isfinite(x)], x[big & torch.isfinite(x)])


def test_gemma_rmsnorm_is_the_reference_sequence(vrf, dec):
    """GemmaRMSNorm (openpi transformers_replace modeling_gemma.py:66-81, weight scaled as 1 + w) run
    under the router equals the reference author's op-by-op route: pow(2) -> vsquare, mean ->
    staged rsum and x 1/N (quant_vector.py:390-399), + eps, rsqrt -> vsqrt+vrecip, x * inv,
    1 + w -> vadd, then vmul."""
    g = torch.Generator().manual_seed(0)
    x = (torch.randn(3, 5, 64, generator=g) * 2).bfloat16()
    w = (torch.randn(64, generator=g) * 0.1).float()
    eps = 1e-6
    with router():
        var = torch.mean(torch.square(x.float()), dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + eps)
        got = (normed * (1.0 + w.float())).to(x.dtype)
    xf = x.float()
    sq = vrf.square(xf)
    v = vrf.mul(dec._vpu_rsum_all(vrf, sq), torch.full((3, 5, 1), 1.0 / 64))
    inv = dec._vpu_rsqrt(vrf, vrf.add(v, torch.full_like(v, eps)))
    n = vrf.mul(xf, inv.expand_as(xf).contiguous())
    w1 = vrf.add(w, torch.full_like(w, 1.0))
    want = vrf.mul(n, w1.expand_as(n).contiguous()).bfloat16()
    assert same(got, want)


def test_embedding_scale_and_gated_residual(vrf):
    """lang_emb * sqrt(width) (pi0_pytorch.py:217) and x + y (modeling_gemma.py:226) are single
    vmul / vadd instructions with the constant rounded to BF16."""
    g = torch.Generator().manual_seed(1)
    e = torch.randn(2, 7, 64, generator=g).bfloat16()
    with router():
        got = e * math.sqrt(2048)
        res = e + got
    c = torch.full(e.shape, math.sqrt(2048))
    assert same(got, vrf.mul(e.float(), c))
    assert same(res, vrf.add(e.float(), vrf.mul(e.float(), c)))


# ---------------------------------------------------------------------------------------------
# Masks, the vexp floor and the flow-time loop
# ---------------------------------------------------------------------------------------------

def _pi0_mask(att_2d):
    """PI0Pytorch._prepare_attention_masks_4d (pi0_pytorch.py:158-160)."""
    return torch.where(att_2d[:, None, :, :], 0.0, -2.3819763e38)


def test_pi0_mask_masks_through_the_vexp_floor():
    """Prefix-LM mask with padding (make_att_2d_masks, pi0_pytorch.py:52-81): masked logits are
    -2.38e38 after the vadd; vexp alone would return exp(-2.38e38 - max) = 1 (the model wraps below
    -336), the floor makes them exactly 0. A fully masked (padding) query row stays uniform, as in
    torch, without NaN."""
    torch.manual_seed(0)
    pad = torch.tensor([[1] * 10 + [0] * 3 + [1] * 5], dtype=torch.bool)       # prefix, pad, suffix
    ar = torch.tensor([[0] * 13 + [1] + [1] + [0] * 3], dtype=torch.int32)
    cs = torch.cumsum(ar, 1)
    att = (cs[:, None, :] <= cs[:, :, None]) & (pad[:, None, :] & pad[:, :, None])
    scores = (torch.randn(1, 8, 18, 18) * 4).bfloat16()
    r = router()
    with r:
        mask = _pi0_mask(att)
        p = torch.softmax((scores + mask).float(), dim=-1)
    want = torch.softmax(scores.float() + mask, dim=-1)
    allowed = att[:, None].expand_as(p)
    assert (p[~allowed & allowed.any(-1, keepdim=True)] == 0).all()
    assert not torch.isnan(p).any()
    assert torch.allclose(p, want, atol=2e-2)
    rep = r.cov.report()
    units = {(o["unit"], o["op"]) for o in rep["ops"]}
    assert ("data", "where") in units and ("VPU", "add") in units and ("VPU", "_softmax") in units
    assert any(d["op"] == "where (data)" for d in rep["new_decisions"])
    assert r.vpu.exp(torch.tensor([-2.3819763e38])).item() == 1.0                # why the floor is needed


def test_flow_time_loop_stays_float32():
    """sample_actions' `time += dt` / `while time >= -dt / 2` (pi0_pytorch.py:402-419): the 0-d
    in-place update stays on the host in float32, as in the reference (no add_ in pi0-quant's
    TARGET_OPS); a BF16 vadd would have drifted to 0.0884 by the last step."""
    r = router()
    seen = []
    with r:
        dt = torch.tensor(-1.0 / 10, dtype=torch.float32)
        time = torch.tensor(1.0, dtype=torch.float32)
        while time >= -dt / 2:
            seen.append(time.item())
            time += dt
    ref, t = [], torch.tensor(1.0)
    while t >= 0.05:
        ref.append(t.item())
        t += torch.tensor(-0.1)
    assert seen == ref and len(seen) == 10
    host = {h["op"]: h["reason"] for h in r.cov.report()["host_fallback"]}
    assert host["add_"] == M.LOOP_SCALAR and "ge" in host
    x = torch.randn(4)
    assert r.classify(A.add_.Tensor, "add_", (x, x), {}, None)[0] == "VPU"      # only 0-d scalars


def test_vpu_rope_and_time_tables_out_of_range():
    """Why --chip-rope host matters for Pi0: Pi0's prefix positions reach ~880 and its timestep
    angles 2 pi / 4e-3 = 1571 rad; the VPU's sin/cos (Q(9,12) argument, reference lane boxes) and
    BF16 angles are far off there."""
    v = co.ChipVPU()
    x = torch.tensor([100.0, 600.0, 880.0, 1571.0])
    err = (v.sin(x) - torch.sin(x.bfloat16().float())).abs()
    assert err[0] < 0.05 and (err[1:] > 0.04).all()


# ---------------------------------------------------------------------------------------------
# Tiny random-weight Pi0 (openpi environment only)
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny():
    need_refs()
    import pi0_tiny
    why = pi0_tiny.available()
    if why:
        pytest.skip(why)
    return pi0_tiny


@pytest.mark.parametrize("rope", ["vpu", "host"])
def test_tiny_pi0_coverage(tiny, rope):
    import quant_modes

    model, cfg = tiny.build()
    opts = {"mxu": "ipt", "mxu_impl": "torch", "chip_mxu": "framework", "chip_rope": rope, "crosscheck": "off"}
    rep = quant_modes.quantize(model, "chip", ["lm_head"], opts)
    with torch.no_grad():
        out = model.sample_actions("cpu", tiny.observation(cfg), noise=tiny.noise(cfg), num_steps=2)
    assert torch.isfinite(out).all()
    cov = rep["coverage"]
    assert cov["silent_bf16_ops"] == 0
    host = {h["op"] for h in cov["host_fallback"]}
    base = {"le", "ge", "add_"}                       # float mask compare, loop test, loop time update
    if rope == "vpu":
        assert host == base
    else:
        assert base <= host
        extra = host - base
        assert {"sin", "cos", "bmm"} <= extra
        assert all(op in ("sin", "cos", "bmm", "mul") or op.endswith("@create_sinusoidal_pos_embedding")
                   for op in extra), extra
        vpu_ops = {o["op"] for o in cov["ops"] if o["unit"] == "VPU"}
        assert not {"sin", "cos"} & vpu_ops                       # no table left on the VPU
    units = cov["units"]
    assert units["MXU"]["calls"] > 0 and units["VPU"]["calls"] > 0
    assert not any(o["op"].startswith("_scaled_dot_product") for o in cov["ops"])
