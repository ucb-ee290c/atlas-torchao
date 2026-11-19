"""native_fp8: ref_conv's codes and exponents on the FP8 tensor cores (src/reference/native_fp8.py).

GPU tests (compute capability >= 8.9). They check that
  - the encoder never emits the 16 codes Atlas and torch decode differently, so the codes can be viewed as
    float8_e4m3fn bytes;
  - native_fp8 quantizes to exactly ref_conv's weight codes / exponents and activation codes / exponents;
  - per layer, native_fp8 matches a float64 oracle on the same codes to float32 accumulation error, and differs
    from ref_conv only by accumulation order (the measured differences are printed with -s);
  - the kernel gate counts at least one torch._scaled_mm call per quantized site and fails a site that ran
    without one;
  - share_native (ref_conv's model switched in place) equals a fresh native_fp8 quantization.
"""

import copy

import pytest
import torch
import torch.nn as nn

import algos
import quant_modes
import reference.native_fp8 as nf
import reference.ref_numerics as rn
from algos import base as algo_hooks
from reference.ref_attention import RefQuantConv2d, ref_attention
from reference.ref_profile import RefQuantLinear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason="native_fp8 needs a CUDA GPU with FP8 tensor cores")

DEV = "cuda"
OPTS = {"crosscheck": "off"}
RECIPES = ("none",
           "granularity:w=panel,a=block,rows=32,rule=nc,attn=head",       # the tile-aware dynamic scales
           "granularity:w=channel,a=token,rule=nc",                        # the per-channel AWQ recipe's scales
           "granularity:w=panel,a=block,rows=32,rule=nc,attn=block")


class Net(nn.Module):
    """Linear layers with aligned and unaligned K / N, a float32 layer and a Conv2d."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 4, stride=4)
        self.a = nn.Linear(64, 96)                  # aligned
        self.b = nn.Linear(96, 40, bias=False)      # N % 16 != 0
        self.c = nn.Linear(40, 72)                  # K % 16 != 0 and N % 16 != 0
        self.f32 = nn.Linear(72, 48)                # stays float32

    def forward(self, img, x):
        y = self.conv(img).flatten(2).transpose(1, 2)      # [B, 16, 16]
        x = x + y.sum(-1, keepdim=True)
        h = self.c(self.b(self.a(x)))
        return self.f32(h.float())


def _net(seed=0):
    torch.manual_seed(seed)
    net = Net()
    with torch.no_grad():
        for m in (net.a, net.b, net.c):
            m.weight.mul_(torch.logspace(-2, 1, m.weight.shape[0])[:, None])   # spread row scales
    net = net.to(DEV, torch.bfloat16)
    net.f32.float()
    return net


def _inputs(seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    img = torch.randn(2, 3, 16, 16, device=DEV, generator=g).to(torch.bfloat16)
    x = (torch.randn(2, 16, 64, device=DEV, generator=g) * torch.logspace(-3, 2, 16, device=DEV)[:, None]).to(torch.bfloat16)
    return img, x


def _quantize(net, mode, spec):
    algo = algos.make(spec)
    if algo is None:
        return quant_modes.quantize(net, mode, (), OPTS), None
    return algos.quantize(net, mode, (), options=OPTS, algo=algo), algo


class _Recorder:
    """Records every activation cast (algo_hooks.act_cast / rn.quantize as called by the reference modules)."""

    def __init__(self, monkeypatch):
        self.casts = []
        orig_act, orig_q = algo_hooks.act_cast, rn.quantize

        def act_cast(*a, **k):
            out = orig_act(*a, **k)
            self.casts.append((out[0].clone(), out[1].clone()))
            return out

        def quantize(*a, **k):
            out = orig_q(*a, **k)
            self.casts.append((out[0].clone(), out[1].clone()))
            return out

        monkeypatch.setattr(algo_hooks, "act_cast", act_cast)
        monkeypatch.setattr(rn, "quantize", quantize)


def test_encoder_never_emits_the_forbidden_codes():
    bits = torch.arange(65536, dtype=torch.int32, device=DEV).to(torch.int16).view(torch.bfloat16)
    for shift in range(-40, 41, 1):
        codes = rn.e4m3_vpack(bits, shift)
        assert not bool(nf.forbidden_mask(codes).any()), f"shift {shift}"
    # the decodings agree on every other code
    allowed = torch.arange(256, dtype=torch.int32, device=DEV).to(torch.uint8)
    allowed = allowed[~nf.forbidden_mask(allowed)]
    assert allowed.numel() == 240
    assert torch.equal(rn.e4m3_unpack(allowed), allowed.view(torch.float8_e4m3fn).float())


@pytest.mark.parametrize("spec", RECIPES)
def test_same_codes_and_exponents_as_ref_conv(spec, monkeypatch):
    img, x = _inputs()
    reps, outs, casts, nets = {}, {}, {}, {}
    for mode in ("ref_conv", "native_fp8"):
        net = _net()
        reps[mode], _ = _quantize(net, mode, spec)
        rec = _Recorder(monkeypatch)
        with torch.no_grad():
            outs[mode] = net(img, x)
        monkeypatch.undo()
        casts[mode], nets[mode] = rec.casts, net
        quant_modes.finalize_report(reps[mode])
    assert reps["ref_conv"]["weight_codes_sha256"] == reps["native_fp8"]["weight_codes_sha256"]
    for n, m in nets["ref_conv"].named_modules():
        if isinstance(m, (RefQuantLinear, RefQuantConv2d)):
            m2 = dict(nets["native_fp8"].named_modules())[n]
            assert torch.equal(m.weight_codes, m2.weight_codes) and torch.equal(m.weight_scale_exp, m2.weight_scale_exp)
    assert len(casts["ref_conv"]) == len(casts["native_fp8"]) > 0
    for (q1, s1), (q2, s2) in zip(casts["ref_conv"], casts["native_fp8"]):
        assert torch.equal(q1, q2) and torch.equal(s1, s2)
    gate = reps["native_fp8"]["native_fp8"]
    assert gate["ok"], gate["problems"]
    assert gate["linear_conv_sites_called"] == 5 and gate["min_kernel_calls_per_called_site"] >= 1
    assert gate["padded_calls"] >= 2                     # b (N=40) and c (K=40, N=72)
    assert gate["forbidden_codes"] == 0
    assert reps["native_fp8"]["ladder"]["accumulation"].startswith("native fp8")
    # end-to-end: accumulation order only
    d = (outs["native_fp8"].float() - outs["ref_conv"].float()).abs().max() / outs["ref_conv"].float().abs().max()
    assert d < 2e-2, d


@pytest.mark.parametrize("spec", RECIPES[:3])
@pytest.mark.parametrize("shape", [(64, 96), (96, 40), (40, 72), (2048, 5632), (5632, 2048)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_per_layer_against_oracle_and_ref_conv(spec, shape, dtype):
    """Same codes; native vs a float64 oracle (tight) and vs ref_conv (accumulation order)."""
    k, n = shape
    torch.manual_seed(0)
    lin = nn.Linear(k, n).to(DEV, dtype)
    with torch.no_grad():
        lin.weight.mul_(torch.logspace(-2, 1, n, device=DEV, dtype=dtype)[:, None])
    x = (torch.randn(3, 37, k, device=DEV) * torch.logspace(-3, 2, 37, device=DEV)[:, None]).to(dtype)
    algo = algos.make(spec)
    native = nf.NativeFP8(compare_calls=1)
    with algos.active(algo):
        layer = RefQuantLinear(lin, "site", fm=native)
        ref_layer = RefQuantLinear(lin, "site", fm=None)
    assert torch.equal(layer.weight_codes, ref_layer.weight_codes)
    with torch.no_grad():
        y = layer(x)
        y_ref = ref_layer(x)
        x_q, s_x, _ = algo_hooks.act_cast(algo, "site", x.to(dtype), "linear")
        oracle = (rn.dequantize(x_q, s_x).double() @ rn.dequantize(layer.weight_codes, layer.weight_scale_exp).double().T)
    y_nb, y_rb = (y - lin.bias).float(), (y_ref - lin.bias).float()       # compare the matmul part
    # native vs float64 oracle: one float32 accumulation (K terms) and one rounding to dtype
    eps = 2.0 ** -8 if dtype == torch.bfloat16 else 2.0 ** -23
    tol = oracle.abs().max().item() * (k * 2.0 ** -24 + eps)
    assert (y_nb.double() - oracle).abs().max().item() <= tol + 1e-30
    stats = nf.compare_outputs(y_nb.to(dtype), y_rb.to(dtype))
    rec = native.compare["site"][0]
    print(f"\n{spec} {shape} {dtype}: native vs ref_conv max_abs {rec['max_abs']:.3e} "
          f"max_rel_to_amax {rec['max_rel_to_amax']:.3e} words_differing {rec['words_differing']}/{rec['elements']}")
    assert stats["max_rel_to_amax"] <= 2 * (k * 2.0 ** -24 + eps)
    gate = native.summary({"site": 1})
    assert gate["ok"] and gate["kernel_calls"] == 1 and gate["scaled_mm_fp8_calls_seen"] == 1


def test_attention_with_pop_shift_and_heads():
    torch.manual_seed(0)
    q = torch.randn(2, 3, 33, 40, device=DEV, dtype=torch.bfloat16)            # D % 16 != 0, S % 16 != 0
    k = torch.randn(2, 3, 50, 40, device=DEV, dtype=torch.bfloat16) * 4
    v = torch.randn(2, 3, 50, 24, device=DEV, dtype=torch.bfloat16)
    native = nf.NativeFP8(compare_calls=4)
    for variant in ("text", "vision", "pi0"):
        o_ref = ref_attention(q, k, v, 40 ** -0.5, None, None, None, variant, site="s")
        o_nat = ref_attention(q, k, v, 40 ** -0.5, None, native, native, variant, site="s")
        assert (o_nat.float() - o_ref.float()).abs().max() <= 2e-2 * o_ref.float().abs().max()
    import reference.ref_attention as ra

    a_q, a_s = rn.quantize(q, (-2, -1))
    b_q, b_s = rn.quantize(k, (-2, -1))
    y = ra._mm(native, a_q, a_s, b_q, b_s, role="pv", pop_shift=-3)
    y_ref = ra._mm(None, a_q, a_s, b_q, b_s, role="pv", pop_shift=-3)
    assert (y - y_ref).abs().max() <= 1e-5 * y_ref.abs().max()
    assert native.kernel_calls_by_role["qk"] == 3 * 6 and native.kernel_calls_by_role["pv"] == 3 * 6 + 6
    assert native.summary()["ok"]


def test_gate_fails_a_site_that_skips_the_kernel():
    net = _net()
    rep = quant_modes.quantize(net, "native_fp8", (), OPTS)
    net.b.fm = None                                    # this site now runs the emulated matmul
    with torch.no_grad():
        net(*_inputs())
    quant_modes.finalize_report(rep)
    assert rep["valid"] is False
    assert any("without a kernel call" in p for p in rep["native_fp8"]["problems"])


@pytest.mark.parametrize("spec", RECIPES[:2])
def test_share_native_equals_fresh_native(spec):
    img, x = _inputs()
    fresh = _net()
    rep_f, _ = _quantize(fresh, "native_fp8", spec)
    with torch.no_grad():
        y_fresh = fresh(img, x)
    shared = _net()
    rep_r, _ = _quantize(shared, "ref_conv", spec)
    with torch.no_grad():
        shared(img, x)
    rep_r.pop("algo", None)
    rep_s = quant_modes.share_native(shared, rep_r, OPTS)
    with torch.no_grad():
        y_shared = shared(img, x)
    quant_modes.finalize_report(rep_s)
    assert torch.equal(y_fresh, y_shared)
    assert rep_s["shared_quantization"]["weight_codes_sha256_equal"]
    assert rep_s["weight_codes_sha256"] == rep_f["weight_codes_sha256"]
    assert rep_s["native_fp8"]["ok"] and rep_s["ladder"]["profile"] == "native_fp8"
    assert quant_modes.can_share_native(["bf16", "ref_conv", "native_fp8"], 2)
    assert not quant_modes.can_share_native(["bf16", "native_fp8"], 1)
    assert not quant_modes.can_share_native(["bf16", "ref_conv", "native_fp8"], 3)


def test_block32_is_refused():
    lin = nn.Linear(64, 32).to(DEV, torch.bfloat16)
    with algos.active(algos.make("granularity:w=block32")):
        with pytest.raises(NotImplementedError):
            RefQuantLinear(lin, "x", fm=nf.NativeFP8())
