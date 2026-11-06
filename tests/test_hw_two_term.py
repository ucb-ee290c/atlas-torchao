"""Two-term FP8 activations (hw_two_term) and its cost model (hw_cost.two_term_layer_cost)."""
import pytest
import torch
import torch.nn as nn

import algos
import chip_ops
import hw_cost
import hw_params as hp
import hw_two_term as tt
import quant_modes
import ref_numerics as rn
from algos.granularity import scale_exp

VPU = chip_ops.ChipVPU()


def qdq(x, dims=(-1,)):
    xb = x.to(torch.bfloat16)
    s = scale_exp(xb, dims, "nc")
    return rn.e4m3_unpack(rn.e4m3_vpack(xb, s), s).to(torch.bfloat16), s


def heavy(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * torch.exp2(torch.randn(*shape, generator=g) * 2)).to(torch.bfloat16)


def tiny(bias=True, seed=0):
    torch.manual_seed(seed)
    m = nn.Sequential()
    m.add_module("up", nn.Linear(64, 96, bias=bias))
    m.add_module("down_proj", nn.Linear(96, 48, bias=bias))
    return m.to(torch.bfloat16)


def quantized(mode="reference", bias=True, spec=None, **opts):
    m = tiny(bias)
    algo = algos.make(spec) if spec else None
    algos.quantize(m, mode, options=opts, algo=algo)
    return m


def test_idempotence_bit_for_bit():
    x = heavy(64, 256)
    x1, s = qdq(x)
    x2, s2 = qdq(x1)
    assert torch.equal(x1.view(torch.int16), x2.view(torch.int16))          # the dequantized values
    same = (s == s2).squeeze(-1)                                             # codes too, wherever the row scale agrees
    q1 = rn.e4m3_vpack(x.to(torch.bfloat16), s)
    q2 = rn.e4m3_vpack(x1, s2)
    assert same.float().mean() > 0.9
    assert torch.equal(q1[same], q2[same])


def test_two_term_error_drops_and_act_error_is_tiny():
    torch.manual_seed(1)
    x, w = heavy(128, 512, seed=2), heavy(96, 512, seed=3)
    wq, _ = qdq(w)
    f = lambda a, b: a.double() @ b.double().T
    ref = f(x, w)
    rel = lambda y: float((y - ref).norm() / ref.norm())
    x1, _ = qdq(x)
    r = VPU.sub(x, x1)
    x2, _ = qdq(r)
    xtt = x1.float() + x2.float()
    one, two = rel(f(x1, wq)), rel(f(xtt, wq))
    assert two < one
    assert rel(f(xtt, w)) < 0.02 * rel(f(x1, w))                             # activation-only error ~ 0 (<2% of one-term)
    assert rel(f(xtt, w)) < 3e-3


def hand(layer_q, x):
    """F(x1) + F(r) by hand on a RefQuantLinear (bias in the first term only)."""
    x = x.to(torch.bfloat16)
    x1, _ = qdq(x)
    r = VPU.sub(x, x1)
    y1 = layer_q(x1)
    b, layer_q.bias = layer_q.bias, None
    y2 = layer_q(r)
    layer_q.bias = b
    return VPU.add(y1, y2)


@pytest.mark.parametrize("spec", [None, "granularity:w=channel,a=token,rule=nc"])
def test_reference_wrapped_equals_by_hand(spec):
    m = quantized(spec=spec)
    x = heavy(5, 64, seed=4)
    up = m.up
    if spec is None:
        # without an algorithm the cast is per tensor; build the expected value with that cast
        def hand_pt(x):
            q, s = rn.quantize(x)
            x1 = rn.dequantize(q, s).to(torch.bfloat16)
            r = VPU.sub(x, x1)
            y1 = up(x1)
            b, up.bias = up.bias, None
            y2 = up(r)
            up.bias = b
            return VPU.add(y1, y2)
        want = hand_pt(x)
    else:
        want = hand(up, x)
    rec = tt.apply(m, "reference", "all")
    assert rec["wrapped_layers"] == 2
    got = up(x)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
    assert not torch.equal(got, up._two_term_orig(x))


def test_bias_added_once():
    m = quantized(spec="granularity:w=channel,a=token,rule=nc")
    with torch.no_grad():
        m.up.bias.copy_(torch.full_like(m.up.bias, 8.0))
    tt.apply(m, "reference", "all")
    x = heavy(3, 64, seed=5)
    y = m.up(x)
    with torch.no_grad():
        m.up.bias.zero_()
    y0 = m.up(x)
    d = (y.float() - y0.float())
    assert torch.allclose(d, torch.full_like(d, 8.0), atol=0.1)             # once (twice would give 16)
    assert m.up.bias is not None


def test_default_off_is_bit_identical():
    x = heavy(5, 64, seed=6)
    a = quantized(spec="granularity:w=channel,a=token,rule=nc")
    b = quantized(spec="granularity:w=channel,a=token,rule=nc")
    assert tt.apply(b, "reference", "off") is None
    assert tt.apply(b, "bf16", "all") is None
    assert not any(getattr(mm, "_two_term_orig", None) for mm in b.modules())
    assert torch.equal(a(x).view(torch.int16), b(x).view(torch.int16))


def test_down_only_unwrap_and_rank3():
    m = quantized(spec="granularity:w=channel,a=token,rule=nc")
    rec = tt.apply(m, "reference", "down")
    assert rec["wrapped_layers"] == 1 and not getattr(m.up, "_two_term_orig", None)
    x = heavy(2, 3, 96, seed=7)
    y3 = m.down_proj(x)
    y2 = m.down_proj(x.reshape(6, 96)).reshape(2, 3, 48)
    assert torch.equal(y3.view(torch.int16), y2.view(torch.int16))
    assert tt.unwrap(m) == 1


def test_reference_hw_matches_by_hand():
    m = quantized("reference_hw", spec="granularity:w=channel,a=token,rule=nc", crosscheck="on", crosscheck_frac=1.0)
    x = heavy(5, 64, seed=8)
    want = hand(m.up, x)
    tt.apply(m, "reference_hw", "all")
    assert torch.equal(m.up(x).view(torch.int16), want.view(torch.int16))
    cc = m.up.fm.summary() if hasattr(m.up.fm, "summary") else {"ok": True}
    assert cc.get("ok", True) and not cc.get("bit_mismatches")


def test_chip_smoke():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=1, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, attn_implementation="eager", use_cache=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    algos.quantize(model, "chip", ["lm_head"], {"chip_mxu": "framework"},
                   algo=algos.make("granularity:w=channel,a=token,rule=nc"))
    with torch.no_grad():
        base = model(ids, labels=ids).loss.item()
        rec = tt.apply(model, "chip", "all")
        two = model(ids, labels=ids).loss.item()
    assert rec["wrapped_layers"] == 7
    assert torch.isfinite(torch.tensor(two)) and abs(two - base) < 0.5


# ---- cost model ----------------------------------------------------------------------------------
def test_cost_free_up_to_twelve_blocks():
    assert 11.9 < hw_cost.two_term_free_blocks() < 12.0
    for N, K in ((2048, 2048), (5632, 2048)):
        for t in (1, 32, 128):
            r = hw_cost.two_term_layer_cost(N, K, t)
            assert r["cycles_2"] == r["cycles_1"] and r["bound_2"] == "dma"
            assert r["vpu_extra_cycles"] < 0.05 * r["dma_cycles"]
        assert hw_cost.two_term_layer_cost(N, K, 11 * 32)["overhead"] == 0
        assert hw_cost.two_term_layer_cost(N, K, 13 * 32)["overhead"] > 0
        assert hw_cost.two_term_layer_cost(N, K, 64 * 32)["overhead"] == pytest.approx(1.0, abs=0.02)


def test_cost_dma_and_instruction_counts():
    r = hw_cost.two_term_layer_cost(2048, 2048, 32)
    assert r["tiles"] == 64 * 64
    assert r["dma_cycles"] == pytest.approx(4096 * 764, rel=0.01)
    assert r["mxu_cycles_2"] - hp.MXU_MATMUL_LATENCY["ipt"] == 2 * (r["mxu_cycles_1"] - hp.MXU_MATMUL_LATENCY["ipt"])
    assert r["vpu_extra_tile_instrs"] == 64 * 7 + 64 * 2


# ---- attention two-term (--act-two-term attnq | attn | all+attn) -----------------------------------
def _qkv(seed=0, B=2, H=3, Sq=6, Skv=9, D=32, heavy_q=True):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, Sq, D, generator=g) * (torch.exp2(torch.randn(B, H, Sq, D, generator=g) * 2) if heavy_q else 1)
    k = torch.randn(B, H, Skv, D, generator=g)
    v = torch.randn(B, H, Skv, D, generator=g)
    mask = torch.rand(B, Sq, Skv, generator=g) > 0.3
    mask[..., 0] = True
    return q.bfloat16(), k.bfloat16(), v.bfloat16(), mask


def _bf16_attn(q, k, v, mask, scale):
    s = q.double() @ k.double().transpose(-1, -2) * scale
    s = s.masked_fill(~mask[:, None], float("-inf"))
    return torch.softmax(s, -1) @ v.double()


def test_attention_default_off_bit_identical():
    import ref_attention as ra
    q, k, v, mask = _qkv()
    a = ra.ref_attention(q, k, v, 0.2, mask)
    b = ra.ref_attention(q, k, v, 0.2, mask, two_term=None)
    assert torch.equal(a, b)
    m = quantized("reference")
    assert tt.apply(m, "reference", "off") is None


def test_attention_two_pass_by_hand():
    import ref_attention as ra
    q, k, v, mask = _qkv()
    sc = 0.2
    got = ra.ref_attention(q, k, v, sc, mask, two_term="qp")
    hd = (-2, -1)
    q1q, sq = rn.quantize(q, hd)
    q1 = rn.dequantize(q1q, sq).to(torch.bfloat16)
    q2q, sq2 = rn.quantize(VPU.sub(q, q1), hd)
    kq, sk = rn.quantize(k, hd)
    mm = lambda aq, asx: torch.matmul(rn.dequantize(aq, asx), rn.dequantize(kq, sk).transpose(-1, -2))
    s = VPU.add(mm(q1q, sq).to(torch.bfloat16), mm(q2q, sq2).to(torch.bfloat16)).float() * sc
    s = torch.where(mask[:, None], s, torch.finfo(torch.float32).min)
    p = torch.softmax(s, -1).to(torch.bfloat16)
    p1q, sp = rn.quantize(p, hd)
    p1 = rn.dequantize(p1q, sp).to(torch.bfloat16)
    p2q, sp2 = rn.quantize(VPU.sub(p, p1), hd)
    vq, sv = rn.quantize(v, hd)
    vd = rn.dequantize(vq, sv)
    o = VPU.add(torch.matmul(rn.dequantize(p1q, sp), vd).to(torch.bfloat16),
                torch.matmul(rn.dequantize(p2q, sp2), vd).to(torch.bfloat16))
    assert torch.equal(got, o)
    # attnq: Q only, P single-term
    gq = ra.ref_attention(q, k, v, sc, mask, two_term="q")
    pq = torch.softmax(s, -1).to(torch.bfloat16)
    pqq, spq = rn.quantize(pq, hd)
    assert torch.equal(gq, torch.matmul(rn.dequantize(pqq, spq), vd).to(torch.bfloat16))


@pytest.mark.parametrize("two", ["q", "qp"])
def test_attention_two_term_error_lower_heavy_q(two):
    import ref_attention as ra
    q, k, v, mask = _qkv(seed=3)
    want = _bf16_attn(q, k, v, mask, 0.2)
    e = lambda t: float((ra.ref_attention(q, k, v, 0.2, mask, two_term=t).double() - want).norm() / want.norm())
    assert e(two) < e(None)
    # additive-mask (Llama eager) path too
    add = torch.zeros(mask.shape).masked_fill(~mask, float("-inf"))[:, None]
    assert torch.isfinite(ra.ref_attention(q, k, v, 0.2, add, two_term=two).float()).all()


def test_attention_apply_choices_and_llama_hook():
    from test_ladder import tiny_llama
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    base = None
    for opt, n_lin in (("attnq", 0), ("attn", 0), ("all+attn", 14)):
        model = tiny_llama(layers=2)
        from ref_attention import patch_attention
        sites = patch_attention(model)
        assert len(sites) == 2
        algos.quantize(model, "reference", ["lm_head"], {}, algo=algos.make("granularity:w=channel,a=token,rule=nc"))
        with torch.no_grad():
            base = model(ids).logits.float()
            rec = tt.apply(model, "reference", opt)
            two = model(ids).logits.float()
        assert rec["attention_sites"] == 2 and rec["wrapped_layers"] == n_lin
        assert rec["attention_two_term"] == tt.ATTN_TWO_TERM[opt]
        assert torch.isfinite(two).all() and not torch.equal(two, base)
    with pytest.raises(ValueError):
        tt.apply(model, "reference", "bogus")


def test_attention_chip_smoke():
    from test_ladder import tiny_llama
    from ref_attention import patch_attention
    ids = torch.randint(0, 128, (1, 24), generator=torch.Generator().manual_seed(0))
    model = tiny_llama(layers=1)
    algos.quantize(model, "chip", ["lm_head"], {"chip_mxu": "framework"},
                   algo=algos.make("granularity:w=channel,a=token,rule=nc"))
    with torch.no_grad():
        base = model(ids).logits.float()
        rec = tt.apply(model, "chip", "attn")
        two = model(ids).logits.float()
    assert torch.isfinite(two).all()
    assert rec["attention_sites"] == 1


def test_attention_cost_model():
    a = hw_cost.attention_two_term_cost(2048, 2048, 64, 32, 4, "q", "vmem")
    assert a["mxu_cycles_2"] > a["mxu_cycles_1"] and a["dma_cycles"] == 0 and a["overhead"] > 0.45
    d = hw_cost.attention_two_term_cost(1, 2048, 64, 32, 4, "q", "hbm")
    assert d["overhead"] == 0 and d["bound_2"] == "dma"          # decode against an HBM-streamed KV cache: free for Q
    qp = hw_cost.attention_two_term_cost(1, 2048, 64, 32, 4, "qp", "hbm")
    assert qp["vpu_extra_tile_instrs"] > d["vpu_extra_tile_instrs"]
