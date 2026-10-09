import pytest
import torch
import torch.nn as nn

from refutil import need_refs

import accum_backends as ab
import ref_loader
from ref_attention import RefQuantConv2d, patch_attention, ref_attention


@pytest.fixture(scope="module")
def mp():
    need_refs()
    return ref_loader.import_ref("smolvla_inout.model_patcher")


class SmolVLMWithExpertModel(nn.Module):           # name matters: the patcher matches it
    num_attention_heads, num_key_value_heads = 4, 2

    def eager_attention_forward(self, *a):          # never called: replaced by both patchers
        raise AssertionError


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm_with_expert = SmolVLMWithExpertModel()


def _inputs(gen, B=2, Sq=5, Skv=7, H=4, Hkv=2, D=40):
    q = torch.randn(B, Sq, H, D, generator=gen) * 3
    k = torch.randn(B, Skv, Hkv, D, generator=gen)
    v = torch.randn(B, Skv, Hkv, D, generator=gen) * 0.5
    mask = torch.rand(B, Sq, Skv, generator=gen) > 0.3
    mask[..., 0] = True
    return q.bfloat16(), k.bfloat16(), v.bfloat16(), mask


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_attention_head(mp, mxu):
    """Batched per-head routine == the reference closure (smolVLA model_patcher.py:624-723)."""
    g = torch.Generator().manual_seed(0)
    q, k, v, mask = _inputs(g)
    ref_model = _Holder()
    mp.patch_attn_smolvla(ref_model, active_groups={mp.QuantGroup.TRANSFORMER},
                          functional_model_factory=lambda i, o: ab.get_accumulator(mxu, "ref_cpu"))
    want = ref_model.vlm_with_expert.eager_attention_forward(mask, 2, 40, q, k, v)

    ours = _Holder()
    sites = patch_attention(ours, fm_factory=lambda: ab.get_accumulator(mxu, "ref_cpu"))
    assert sites[0]["kind"] == "smolvla_text_expert"
    got = ours.vlm_with_expert.eager_attention_forward(mask, 2, 40, q, k, v)
    assert got.dtype == want.dtype and torch.equal(got, want)


class SmolVLMVisionAttention(nn.Module):           # name matters for the reference vision patcher
    pass


class _VisionHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_model = nn.Module()
        self.vision_model.self_attn = SmolVLMVisionAttention()


def test_vision_attention_head(mp):
    """Vision variant == the reference Idefics3/SmolVLM closure (model_patcher.py:798-866)."""
    from transformers.models.smolvlm import modeling_smolvlm

    g = torch.Generator().manual_seed(1)
    B, H, S, D = 1, 3, 9, 32
    q, k, v = (torch.randn(B, H, S, D, generator=g).bfloat16() for _ in range(3))
    mask = torch.zeros(B, 1, S, S, dtype=torch.bfloat16)
    mask[..., -2:] = torch.finfo(torch.bfloat16).min
    model = _VisionHolder()
    mp.patch_attn_idefics3_eager(model, active_groups={mp.QuantGroup.VISION},
                                 functional_model_factory=lambda i, o: ab.get_accumulator("ipt", "ref_cpu"))
    try:
        want, _ = modeling_smolvlm.eager_attention_forward(model.vision_model.self_attn, q, k, v, mask, D ** -0.5)
    finally:
        mp.unpatch_attn_idefics3_eager()
    fm = ab.get_accumulator("ipt", "ref_cpu")
    got = ref_attention(q, k, v, D ** -0.5, mask, fm, fm, "vision").transpose(1, 2)
    assert torch.equal(got, want)


def test_torch_fallback_reference_matmul_close_to_float():
    """`reference` (no MXU model) attention stays close to plain float attention."""
    g = torch.Generator().manual_seed(2)
    q, k, v, mask = _inputs(g, H=2, Hkv=2)
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))
    got = ref_attention(q, k, v, 40 ** -0.5, mask).float()
    s = (q.float() @ k.float().transpose(-1, -2)) * 40 ** -0.5
    s = torch.where(mask[:, None], s, torch.finfo(torch.float32).min)
    want = torch.softmax(s, -1) @ v.float()
    assert ((got - want).norm() / want.norm()) < 0.1


def test_llama_attention_runs_through_reference(mp):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=64, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, attn_implementation="eager")
    torch.manual_seed(0)
    m = LlamaForCausalLM(cfg).bfloat16().eval()
    ids = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        base = m(ids).logits.float()
        sites = patch_attention(m, fm_factory=None)
        assert len(sites) == 2 and all(s["variant"] == "text" for s in sites)
        out = m(ids).logits.float()
    assert not torch.equal(out, base)
    assert ((out - base).norm() / base.norm()) < 0.2
    # causal: changing a later token must not change earlier logits
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 1) % 64
    with torch.no_grad():
        out2 = m(ids2).logits.float()
    assert torch.equal(out[0, :-1], out2[0, :-1])


@pytest.mark.parametrize("mxu", [None, "sa"])
def test_conv2d_matches_reference(mp, mxu):
    """RefQuantConv2d == the reference QuantConv2d functional-model path (model_patcher.py:345-375)."""
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, kernel_size=4, stride=4, padding=0).bfloat16()
    x = torch.randn(2, 3, 16, 12).bfloat16()
    fm = ab.get_accumulator(mxu, "ref_cpu") if mxu else None
    got = RefQuantConv2d(conv, "c", fm)(x)
    if mxu is None:
        assert ((got.float() - conv(x).float()).norm() / conv(x).float().norm()) < 0.1
        return
    QC = mp.QuantConv2d
    QF = ref_loader.import_ref("smolvla_inout.quant_types").QuantFormat
    ref = QC(conv, QF.BFLOAT16, QF.BFLOAT16, mp.Component.VISION, "c",
             functional_model=ab.get_accumulator(mxu, "ref_cpu"))
    assert torch.equal(got, ref(x))
