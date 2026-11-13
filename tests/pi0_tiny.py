"""A tiny random-weight openpi PI0Pytorch for op census and chip-coverage tests.

The model is openpi's own PI0Pytorch (pi0=True, not pi05), built through the reference author's
loader path (pi0-quant pi0_inout/serve_quant.py:55-66: openpi source tree from $OPENPI_DIR, the
reference JAX stubs injected), with every block type of the real model but small widths:

  PaliGemma  SigLIP: conv patch embedding (14x14, 224x224 images), LayerNorm, gelu-tanh MLP,
             attention; multimodal projector; Gemma: embedding x sqrt(width), RMSNorm (1 + w),
             RoPE, prefix attention, GeGLU (gelu-tanh) MLP
  expert     Gemma (same blocks), suffix (state + action) attention over the prefix KV cache
  head       state_proj, action_in_proj, sinusoidal time embedding, action_time_mlp (SiLU),
             action_out_proj, Euler integration over the flow steps

openpi hard-codes 8 attention heads (gemma_pytorch.py: reshape to 1 * 8 * head_dim) and the
vision projector width equals the text width, so those are kept. Needs openpi's PyTorch Pi0
environment (transformers 4.53.2 with openpi's transformers_replace files): `available()` says why
not when it is missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

OPENPI_CANDIDATES = (os.environ.get("OPENPI_DIR", ""),)

TINY = {
    "tiny_vlm": dict(width=64, depth=2, mlp_dim=128, num_heads=8, num_kv_heads=1, head_dim=16),
    "tiny_expert": dict(width=48, depth=2, mlp_dim=96, num_heads=8, num_kv_heads=1, head_dim=16),
}
VISION = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2)


def openpi_dir():
    for c in OPENPI_CANDIDATES:
        if c and (Path(c) / "src/openpi/models_pytorch/pi0_pytorch.py").is_file():
            return Path(c)
    return None


def available() -> str | None:
    """None if a tiny Pi0 can be built here, else the reason it cannot."""
    if openpi_dir() is None:
        return "openpi source tree not found (set OPENPI_DIR)"
    try:
        from transformers.models.siglip import check
        if not check.check_whether_transformers_replace_is_installed_correctly():
            return "openpi transformers_replace not installed in this environment"
    except ImportError:
        return "openpi transformers_replace not installed in this environment"
    return None


def _runtime():
    import loaders.pi0_model as pi0_model

    return pi0_model.runtime(openpi_dir())          # serve_quant with the reference JAX stubs


def build(dtype: str = "bfloat16", action_horizon: int = 4, seed: int = 0):
    """A random-weight PI0Pytorch (eval mode, eager sample_actions)."""
    _runtime()
    gemma = sys.modules["openpi.models.gemma"]
    gp = __import__("openpi.models_pytorch.gemma_pytorch", fromlist=["x"])
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    real_get, real_pg = gemma.get_config, gp.PaliGemmaForConditionalGeneration

    def get_config(variant):
        if variant in TINY:
            return gemma.GemmaConfig(**TINY[variant])
        return real_get(variant)

    def paligemma(config):
        for k, v in VISION.items():
            setattr(config.vision_config, k, v)
        config.vision_config.projection_dim = config.text_config.hidden_size
        return real_pg(config=config)

    cfg = SimpleNamespace(paligemma_variant="tiny_vlm", action_expert_variant="tiny_expert", pi05=False,
                          dtype=dtype, action_dim=32, action_horizon=action_horizon, max_token_len=16,
                          pytorch_compile_mode=None)
    torch.manual_seed(seed)
    gemma.get_config, gp.PaliGemmaForConditionalGeneration = get_config, paligemma
    try:
        model = PI0Pytorch(cfg)
    finally:
        gemma.get_config, gp.PaliGemmaForConditionalGeneration = real_get, real_pg
    with torch.no_grad():                    # RMSNorm weights are zero-initialised: give them values
        for n, p in model.named_parameters():
            if n.endswith("layernorm.weight") or n.endswith("norm.weight"):
                p.normal_(0.0, 0.1)
    return model.eval(), cfg


def observation(cfg, seed: int = 0, n_tokens: int = 10):
    """A DROID-shaped observation as the reference policy shim builds it (serve_quant.py:420-483):
    two camera images in [-1, 1], one zero image with a false mask, state padded to 32, a prompt
    of n_tokens followed by padding."""
    g = torch.Generator().manual_seed(seed)
    img = lambda: torch.rand(1, 3, 224, 224, generator=g) * 2 - 1           # noqa: E731
    images = {"base_0_rgb": img(), "left_wrist_0_rgb": img(), "right_wrist_0_rgb": torch.zeros(1, 3, 224, 224)}
    masks = {"base_0_rgb": torch.ones(1, dtype=torch.bool), "left_wrist_0_rgb": torch.ones(1, dtype=torch.bool),
             "right_wrist_0_rgb": torch.zeros(1, dtype=torch.bool)}
    state = torch.zeros(1, 32)
    state[0, :8] = torch.randn(8, generator=g)
    L = cfg.max_token_len
    toks = torch.zeros(1, L, dtype=torch.int64)
    toks[0, :n_tokens] = torch.randint(1, 1000, (n_tokens,), generator=g)
    tmask = torch.zeros(1, L, dtype=torch.bool)
    tmask[0, :n_tokens] = True
    return SimpleNamespace(images=images, image_masks=masks, state=state, tokenized_prompt=toks,
                           tokenized_prompt_mask=tmask, token_ar_mask=torch.zeros(1, L, dtype=torch.bool),
                           token_loss_mask=torch.zeros(1, L, dtype=torch.bool))


def noise(cfg, seed: int = 0):
    g = torch.Generator().manual_seed(seed + 1)
    return torch.randn(1, cfg.action_horizon, cfg.action_dim, generator=g)


def census(model, obs, nz, num_steps: int = 2) -> dict:
    """{overload name: {"calls", "dtypes"}} of every aten op sample_actions dispatches."""
    from collections import defaultdict
    from torch.utils._python_dispatch import TorchDispatchMode

    seen = defaultdict(lambda: {"calls": 0, "dtypes": set()})

    class Rec(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            row = seen[str(func.name()).replace("aten::", "")]
            row["calls"] += 1
            for a in list(args) + list(kwargs.values()):
                if isinstance(a, torch.Tensor):
                    row["dtypes"].add(str(a.dtype).replace("torch.", ""))
            return func(*args, **kwargs)

    with torch.no_grad(), Rec():
        model.sample_actions("cpu", obs, noise=nz, num_steps=num_steps)
    return {k: {"calls": v["calls"], "dtypes": sorted(v["dtypes"])} for k, v in sorted(seen.items())}
