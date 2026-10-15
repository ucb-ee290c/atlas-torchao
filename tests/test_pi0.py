"""Pi0 (openpi PI0Pytorch) under the reference profiles, on a tiny random-weight Pi0.

Needs an openpi checkout ($OPENPI_DIR) and transformers 4.53.2 with openpi's transformers_replace
files (the atlas-pi0 environment, see README); skipped otherwise. The tiny model keeps Pi0's
structure (SigLIP tower, PaliGemma Gemma, Gemma action expert, action head, 8 heads with one KV
head) with small widths.
"""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from refutil import need_refs

import accum_backends as ab
import ref_loader


def _openpi():
    d = os.environ.get("OPENPI_DIR")
    if not d or not os.path.isdir(os.path.join(d, "src", "openpi", "models_pytorch")):
        pytest.skip("OPENPI_DIR not set to an openpi checkout")
    try:
        from transformers.models.siglip import check
        if not check.check_whether_transformers_replace_is_installed_correctly():
            raise ImportError
    except ImportError:
        pytest.skip("openpi transformers_replace not installed (transformers 4.53.2 env)")
    return d


@pytest.fixture(scope="module")
def rt():
    need_refs()
    import pi0_model

    return pi0_model.runtime(_openpi())


def tiny_pi0(rt, seed=0):
    """PI0Pytorch with tiny Gemma variants and a tiny SigLIP (projection_dim tied to the text width)."""
    import sys

    from transformers import PaliGemmaConfig, SiglipVisionConfig

    gemma = sys.modules["openpi.models.gemma"]                 # the reference's stub (_jax_stubs.py)
    from openpi.models_pytorch import gemma_pytorch, pi0_pytorch

    tiny = {"tiny_vlm": gemma.GemmaConfig(width=64, depth=2, mlp_dim=96, num_heads=8, num_kv_heads=1, head_dim=16),
            "tiny_exp": gemma.GemmaConfig(width=32, depth=2, mlp_dim=48, num_heads=8, num_kv_heads=1, head_dim=16)}

    class Mapping(dict):
        def __getitem__(self, k):
            if k == "paligemma":
                vis = SiglipVisionConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                 patch_size=56, image_size=224)
                return lambda: PaliGemmaConfig(vision_config=vis)
            return orig[k]

    orig = gemma_pytorch.CONFIG_MAPPING
    get = gemma.get_config
    gemma.get_config = lambda v: tiny.get(v) or get(v)
    gemma_pytorch.CONFIG_MAPPING = Mapping()
    # gemma_pytorch.py:38 forces projection_dim = 2048 (gemma_2b's width); tie it to the tiny width
    SiglipVisionConfig.projection_dim = property(lambda self: 64, lambda self, v: None)
    try:
        torch.manual_seed(seed)
        cfg = SimpleNamespace(paligemma_variant="tiny_vlm", action_expert_variant="tiny_exp", pi05=False,
                              dtype="bfloat16", action_dim=32, action_horizon=4, max_token_len=8,
                              pytorch_compile_mode=None)
        m = pi0_pytorch.PI0Pytorch(cfg)
    finally:
        gemma.get_config, gemma_pytorch.CONFIG_MAPPING = get, orig
        del SiglipVisionConfig.projection_dim
    with torch.no_grad():                                      # visible weights, not the init zeros
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m.eval()


def tiny_obs(seed=0, max_tok=8):
    g = torch.Generator().manual_seed(seed)
    img = lambda: torch.rand(1, 3, 224, 224, generator=g) * 2 - 1      # noqa: E731
    return SimpleNamespace(
        images={"base_0_rgb": img(), "left_wrist_0_rgb": img(), "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224)},
        image_masks={"base_0_rgb": torch.ones(1, dtype=torch.bool), "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
                     "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool)},
        state=torch.randn(1, 32, generator=g),
        tokenized_prompt=torch.randint(1, 100, (1, max_tok), generator=g),
        tokenized_prompt_mask=torch.tensor([[True] * 5 + [False] * (max_tok - 5)]),
        token_ar_mask=torch.zeros(1, max_tok, dtype=torch.bool),
        token_loss_mask=torch.zeros(1, max_tok, dtype=torch.bool))


def sample(m, seed=0, steps=2):
    import pi0_model

    noise = pi0_model.frame_noise(seed, 4, 32, "cpu")
    return m.sample_actions("cpu", tiny_obs(seed), noise=noise, num_steps=steps)


def test_patch_coverage_reference_hw(rt):
    """Every Linear (but lm_head), the patch conv and every self_attn is patched and called;
    attention fires on the reference's routes (Gemma eager, SigLIP sdpa)."""
    import pi0_model as eval_pi0
    from quant_modes import quantize

    m = tiny_pi0(rt)
    inv = eval_pi0.inventory(m)
    rep = quantize(m, "reference_hw", ("lm_head",), {"mxu": "ipt", "mxu_impl": "torch", "crosscheck": "off"})
    counter = eval_pi0.CallCounter(m, [f for f, i in inv.items() if i["kind"] in ("linear", "conv2d")])
    with torch.no_grad():
        sample(m)
    counter.close()
    cov = eval_pi0.patch_coverage(inv, m, "reference_hw", rep, counter.counts, ("lm_head",))
    s = cov["summary"]
    n_lin = sum(1 for i in inv.values() if i["kind"] == "linear")
    assert s["linear"]["patched"] == n_lin - 2 == s["linear"]["patched_and_called"]
    assert s["conv2d"]["patched"] == 1 == s["conv2d"]["patched_and_called"]
    assert s["attention"]["patched"] == s["attention"]["sites"] == s["attention"]["patched_and_called"] == 5
    reasons = {r["site"]: r["reason"] for r in cov["not_patched"]}
    assert {k for k in reasons if "lm_head" in k} == {"paligemma_with_expert.paligemma.lm_head",
                                                       "paligemma_with_expert.gemma_expert.lm_head"}
    assert all(r["kind"] in ("linear", "embedding") for r in cov["not_patched"])
    routes = {r["site"]: r["routes"] for r in cov["sites"] if r["kind"] == "attention"}
    for site, rr in routes.items():
        assert set(rr) == ({"sdpa"} if "vision_tower" in site else {"gemma_eager"}), (site, rr)
    comps = {r["component"] for r in cov["sites"] if r["patched"]}
    assert comps == {"vision", "language", "action_expert", "action_head"}
    # fp32 action head layers are quantized by the reference profiles (model_patcher.py:83-87)
    assert all(r["patched"] for r in cov["sites"] if r["component"] == "action_head")


@pytest.fixture(scope="module")
def mp():
    need_refs()
    return ref_loader.import_ref("pi0_inout.model_patcher")


class _Attn(nn.Module):
    num_key_value_groups = 8
    training = False


class _Holder(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert = nn.Module()
        self.paligemma_with_expert.gemma_expert.self_attn = _Attn()


def _qkv(g, B=1, H=8, Hkv=1, Sq=5, Skv=9, D=16):
    q = (torch.randn(B, H, Sq, D, generator=g) * 2).bfloat16()
    k = torch.randn(B, Hkv, Skv, D, generator=g).bfloat16()
    v = (torch.randn(B, Hkv, Skv, D, generator=g) * 0.5).bfloat16()
    allowed = torch.rand(B, 1, Sq, Skv, generator=g) > 0.3
    allowed[..., 0] = True
    mask = torch.where(allowed, 0.0, -2.3819763e38)            # pi0_pytorch.py:157-160 (float32)
    return q, k, v, mask


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_gemma_attention_bitexact(rt, mp, mxu):
    """Pi0 Gemma route == the reference patch_attn_eager closure (model_patcher.py:911-1005)."""
    from transformers.models.gemma import modeling_gemma

    import ref_attention

    g = torch.Generator().manual_seed(0)
    q, k, v, mask = _qkv(g)
    ours = _Holder()
    ref_attention.patch_pi0_attention(ours, fm_factory=lambda: ab.get_accumulator(mxu, "torch"))
    got, _ = modeling_gemma.eager_attention_forward(ours.paligemma_with_expert.gemma_expert.self_attn,
                                                    q, k, v, mask, 0.25)
    theirs = _Holder()
    saved = modeling_gemma.eager_attention_forward
    mp.patch_attn_eager(theirs, active_groups={mp.QuantGroup.ACTION_EXPERT},
                        functional_model_factory=lambda i, o: ab.get_accumulator(mxu, "ref_cpu"))
    try:
        want, _ = modeling_gemma.eager_attention_forward(theirs.paligemma_with_expert.gemma_expert.self_attn,
                                                         q, k, v, mask, 0.25)
    finally:
        mp.unpatch_attn_eager()
        modeling_gemma.eager_attention_forward = saved
    assert got.dtype == want.dtype and torch.equal(got, want)
    plain, _ = ref_attention._PI0_ORIG["gemma"](_Attn(), q, k, v, mask, 0.25)
    assert not torch.equal(got, plain)                          # the FP8 routine actually ran


class _VisionHolder(nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = nn.Module()
        self.paligemma_with_expert.vision_tower = nn.Module()
        self.paligemma_with_expert.vision_tower.self_attn = nn.Identity()


def test_siglip_sdpa_bitexact(rt, mp):
    """Pi0 SigLIP SDPA route == the reference patch_attn_sdpa closure (model_patcher.py:738-823),
    BF16 softmax, no mask."""
    import torch.nn.functional as F

    import ref_attention

    g = torch.Generator().manual_seed(1)
    q, k, v = ((torch.randn(1, 2, 7, 16, generator=g) * s).bfloat16() for s in (2, 1, 0.5))
    ours = _VisionHolder()
    ref_attention.patch_pi0_attention(ours, fm_factory=lambda: ab.get_accumulator("ipt", "torch"))
    attn = ours.paligemma_with_expert.vision_tower.self_attn
    attn.register_forward_hook(lambda *_: None)
    ref_attention._pi0_current.module = attn
    try:
        got = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=0.25)
    finally:
        ref_attention._pi0_current.module = None
    theirs = _VisionHolder()
    saved = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = ref_attention._PI0_ORIG["sdpa"]
    handles = mp.patch_attn_sdpa(theirs, active_groups={mp.QuantGroup.VISION},
                                 functional_model_factory=lambda i, o: ab.get_accumulator("ipt", "ref_cpu"))
    try:
        mp._attn_component_local.component = mp.Component.VISION
        want = F.scaled_dot_product_attention(q, k, v, None, 0.0, False, scale=0.25)
    finally:
        mp._attn_component_local.component = None
        for h in handles:
            h.remove()
        F.scaled_dot_product_attention = saved
    assert torch.equal(got, want)
    assert not torch.equal(got, ref_attention._PI0_ORIG["sdpa"](q, k, v, scale=0.25))


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_pi0_linear_bitexact(rt, mp, mxu):
    """A Pi0 action-head layer (float32, quantized through BF16) and a bf16 projection under
    reference_hw == the reference QuantLinear with its functional model (quant_linear.py:133-175)."""
    from ref_profile import RefQuantLinear

    m = tiny_pi0(rt)
    QF = ref_loader.import_ref("pi0_inout.quant_types").QuantFormat
    QL = ref_loader.import_ref("pi0_inout.quant_linear").QuantLinear
    g = torch.Generator().manual_seed(2)
    for name in ("action_in_proj", "paligemma_with_expert.gemma_expert.model.layers.0.mlp.down_proj"):
        lin = dict(m.named_modules())[name]
        x = (torch.randn(3, lin.in_features, generator=g) * 3).to(lin.weight.dtype)
        ours = RefQuantLinear(lin, name, ab.get_accumulator(mxu, "torch"))(x)
        ref = QL(lin, QF.BFLOAT16, QF.BFLOAT16, mp.Component.ACTION_HEAD, name,
                 functional_model=ab.get_accumulator(mxu, "ref_cpu"))(x)
        assert ours.dtype == ref.dtype and torch.equal(ours, ref), name


@pytest.mark.parametrize("mode", ["reference", "reference_hw"])
def test_determinism_and_effect(rt, mode):
    """Same frame and noise twice -> bit-identical actions; quantization changes them, a little."""
    from quant_modes import quantize

    base = tiny_pi0(rt)
    with torch.no_grad():
        b = sample(base)
    m = tiny_pi0(rt)
    quantize(m, mode, ("lm_head",), {"mxu": "ipt", "mxu_impl": "torch", "crosscheck": "off"})
    with torch.no_grad():
        a1, a2 = sample(m), sample(m)
    assert torch.equal(a1, a2)
    assert not torch.equal(a1, b)
    assert ((a1 - b).norm() / b.norm()) < 0.5


def test_component_rules_match_reference(mp):
    """Our Pi0 component tables == pi0-quant's _infer_component (model_patcher.py:112-126)."""
    import layer_rmse
    import ref_attention

    paths = ["action_in_proj", "action_time_mlp_out", "state_proj",
             "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
             "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.0.mlp.fc1",
             "paligemma_with_expert.paligemma.model.language_model.layers.3.mlp.up_proj",
             "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"]
    for p in paths:
        want = mp._infer_component(p).value
        assert ref_attention.pi0_component(p) == want
        assert layer_rmse.component_for("paligemma_with_expert." + p if "paligemma_with_expert" not in p
                                        else p).value == want or "paligemma_with_expert" not in p
