"""A tiny LeRobot SmolVLA policy for CPU tests: lerobot's own VLAFlowMatching (embed_prefix, embed_suffix,
sample_actions, denoise_step) around the tiny SmolVLMWithExpertModel of test_algo_smoothquant._smolvla, with a tiny
SmolVLM vision tower and connector (pixel shuffle) so that images go through embed_image. Same structure as
lerobot/smolvla_libero: cross_attn every second layer, float32 action projections, use_cache on, no image special
tokens, prefix_length 0."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

IMAGE, PATCH, SCALE = 32, 8, 2          # 16 patches per image -> 4 tokens per image after the pixel shuffle
CHUNK, ACTION_DIM, STATE_DIM, LANG = 3, 8, 8, 5


def tiny_policy(n_layers=4, every=2, num_steps=10):
    pytest.importorskip("lerobot.policies.smolvla.modeling_smolvla")
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching
    from test_algo_smoothquant import _smolvla
    from transformers import SmolVLMConfig, SmolVLMVisionConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMConnector, SmolVLMVisionTransformer

    inner = _smolvla(n_layers, every)
    text_cfg = inner.vlm.model.text_model.config
    vision_cfg = SmolVLMVisionConfig(hidden_size=32, intermediate_size=48, num_hidden_layers=1, num_attention_heads=2,
                                     image_size=IMAGE, patch_size=PATCH, num_channels=3)
    vision_cfg._attn_implementation = "eager"
    torch.manual_seed(1)
    vlm_config = SmolVLMConfig(vision_config=vision_cfg.to_dict(), text_config=text_cfg.to_dict(), scale_factor=SCALE)
    vlm_config.vision_config._attn_implementation = "eager"
    inner.vlm.model.vision_model = SmolVLMVisionTransformer(vlm_config.vision_config)
    inner.vlm.model.connector = SmolVLMConnector(vlm_config)
    inner.vlm.config = SimpleNamespace(text_config=text_cfg, vision_config=vlm_config.vision_config, scale_factor=SCALE)
    inner.expert_hidden_size = inner.lm_expert.config.hidden_size

    model = VLAFlowMatching.__new__(VLAFlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(chunk_size=CHUNK, max_action_dim=ACTION_DIM, max_state_dim=STATE_DIM,
                                   min_period=4e-3, max_period=4.0, num_steps=num_steps, use_cache=True,
                                   prefix_length=0, add_image_special_tokens=False, rtc_config=None)
    model.vlm_with_expert = inner
    hidden, expert = text_cfg.hidden_size, inner.expert_hidden_size
    model.state_proj = nn.Linear(STATE_DIM, hidden)
    model.action_in_proj = nn.Linear(ACTION_DIM, expert)
    model.action_out_proj = nn.Linear(expert, ACTION_DIM)
    model.action_time_mlp_in = nn.Linear(2 * expert, expert)
    model.action_time_mlp_out = nn.Linear(expert, expert)
    model.add_image_special_tokens, model.prefix_length, model.rtc_processor = False, 0, None
    policy = nn.Module()
    policy.model = model
    policy.to(torch.bfloat16)
    for name in ("action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"):
        getattr(model, name).float()                     # as eval_policy.load_policy leaves them
    policy.eval()
    model.vlm_with_expert.eval()
    return policy


def observation(seed, n_images=2, valid_tokens=LANG):
    """(images, img_masks, lang_tokens, lang_masks, state, noise): a policy-level observation as sample_actions takes it
    (images in [-1, 1]; the language mask keeps the first valid_tokens tokens)."""
    g = torch.Generator().manual_seed(seed)
    images = tuple((torch.rand(1, 3, IMAGE, IMAGE, generator=g) * 2 - 1).to(torch.bfloat16) for _ in range(n_images))
    lang_masks = torch.zeros(1, LANG, dtype=torch.bool)
    lang_masks[:, :valid_tokens] = True
    return (images, [torch.ones(1, dtype=torch.bool)] * n_images, torch.randint(0, 8, (1, LANG), generator=g),
            lang_masks, torch.randn(1, STATE_DIM, generator=g).to(torch.bfloat16),
            torch.randn(1, CHUNK, ACTION_DIM, generator=g))


def quantize(policy, spec="granularity:w=channel,a=token"):
    import algos
    import exporter.quant_export as quant_export
    from test_ladder import FAST

    algo = algos.make(spec)
    report = algos.quantize(policy, "ref_conv", [], options=dict(FAST), algo=algo)
    return quant_export.build_policy(policy, "ref_conv", report, algo, dict(FAST), [],
                                     report["algo"].get("bf16_opt_out", []), {}, {})

