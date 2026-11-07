"""Multi-Task DiT's own attention blocks (lerobot RoPEAttention) as native FP8 attention sites
(docs/algos/dit-own-attention.md): instance-forward replacement in src/ref_attention.py."""
import copy
import json

import pytest
import torch
import torch.nn as nn

import algos
import atlas_fx_ops  # noqa: F401
import fx_export
import quant_export
import ref_attention as ra
from algos import base as algo_hooks
from test_ladder import FAST

mt = pytest.importorskip("lerobot.policies.multi_task_dit.modeling_multi_task_dit")

HIDDEN, HEADS, T = 128, 2, 24                      # head_dim 64 and T = 24 as in the DiT (T is not a multiple of 32)
SPEC = "granularity:w=channel,a=token,rule=nc"


class Tiny(nn.Module):
    def __init__(self, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(mt.RoPEAttention(HIDDEN, HEADS) for _ in range(layers))

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)
        return x


def _tiny(layers=2):
    torch.manual_seed(0)
    return Tiny(layers).to(torch.bfloat16).eval()


def _x(seed=1, batch=2):
    return torch.randn(batch, T, HIDDEN, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _identity_cast(algo, site, operand, x, dims):
    return x, torch.zeros(1)


def _float_mm(fm, a, a_s, b, b_s, dtype_out=torch.float32, role=None):
    return a.float() @ b.float().transpose(-1, -2)


def test_site_is_discovered_with_vision_variant():
    model = _tiny(3)
    sites = ra.patch_attention(model)
    assert [s["site"] for s in sites] == ["blocks.0", "blocks.1", "blocks.2"]
    assert all(s["kind"] == "RoPEAttention" and s["variant"] == "vision" for s in sites)
    assert all(model.blocks[i]._atlas_ref_attn["variant"] == "vision" for i in range(3))
    assert "forward" in model.blocks[0].__dict__            # instance replacement, no global patch
    assert mt.RoPEAttention.forward is not model.blocks[0].forward.__func__


def test_unpatched_model_is_unchanged_by_the_patch_of_another_instance():
    patched, other = _tiny(1), _tiny(1)
    ra.patch_attention(patched)
    assert "forward" not in other.blocks[0].__dict__ and not hasattr(other.blocks[0], "_atlas_ref_attn")


def test_patched_forward_with_identity_quantization_matches_original_forward(monkeypatch):
    """The only difference left is the attention arithmetic: BF16 scores / float32 softmax / BF16 P
    against the original BF16 scaled_dot_product_attention (no mask, not causal, scale head_dim**-0.5)."""
    original, patched = _tiny(), _tiny()
    ra.patch_attention(patched)
    monkeypatch.setattr(ra, "_cast", _identity_cast)
    monkeypatch.setattr(ra, "_mm", _float_mm)
    for seed in (1, 2, 3):
        x = _x(seed)
        with torch.no_grad():
            a, b = original(x).float(), patched(x).float()
        assert a.shape == b.shape == x.shape
        err = (a - b).abs().max().item()
        assert err < 0.04 * a.abs().max().item(), err        # BF16 rounding of the score / P path only


def test_patched_attention_core_equals_original_sdpa_in_float32(monkeypatch):
    """Same check on the module's attention output with a float32 module: identity quantization makes
    the vision routine the same function as SDPA up to the BF16 rounding of the scores and P."""
    original = mt.RoPEAttention(HIDDEN, HEADS).eval()
    patched = copy.deepcopy(original)
    ra.patch_attention(nn.Sequential(patched))
    monkeypatch.setattr(ra, "_cast", _identity_cast)
    monkeypatch.setattr(ra, "_mm", _float_mm)
    x = torch.randn(2, T, HIDDEN, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        a, b = original(x), patched(x)
    assert b.dtype == a.dtype
    assert (a - b).abs().max().item() < 0.03 * a.abs().max().item()


def test_quantization_is_applied_to_q_k_p_v_after_rope(monkeypatch):
    model = _tiny(1)
    ra.patch_attention(model)
    seen = []
    cast = ra._cast

    def spy(algo, site, operand, x, dims):
        seen.append((operand, tuple(x.shape), dims))
        return cast(algo, site, operand, x, dims)
    monkeypatch.setattr(ra, "_cast", spy)
    with torch.no_grad():
        out = model(_x())
    assert [s[0] for s in seen] == ["q", "k", "p", "v"]
    assert seen[0][1] == seen[1][1] == seen[3][1] == (2, HEADS, T, 64) and seen[2][1] == (2, HEADS, T, T)
    assert out.shape == (2, T, HIDDEN)
    bf16 = _tiny(1)
    with torch.no_grad():
        ref = bf16(_x())
    assert not torch.equal(out, ref) and (out - ref).abs().max().item() < 0.25 * ref.abs().max().item()


def test_tile_native_attention_scales_are_applied():
    """With the tile recipe's attn=block rule every operand goes through act_cast under the module path,
    and the scales it returns are per (batch, head) for the partial T = 24 tile."""
    model = _tiny(2)
    algo = algos.make("gptq_awq:method=awq_gptq,actorder=true+granularity:w=panel,a=block,attn=block,rule=nc")
    report = algos.quantize(model, "ref_conv", [], dict(FAST), algo=algo,
                            calib=lambda: ([_x(5)], lambda m, x: m(x), {"source": "test", "n_samples": 1, "seed": 5}))
    assert [s["site"] for s in report["attention_sites"]] == ["blocks.0", "blocks.1"]
    calls = []
    original = algo_hooks.act_cast

    def spy(algo_, site, x, kind, *dims):
        q, s, blk = original(algo_, site, x, kind, *dims)
        if kind.startswith("attention."):
            calls.append((site, kind, tuple(s.shape)))
        return q, s, blk
    algo_hooks.act_cast, restore = spy, original
    try:
        with torch.no_grad():
            model(_x())
    finally:
        algo_hooks.act_cast = restore
    assert [c[:2] for c in calls] == [("blocks.0", f"attention.{o}") for o in "qkpv"] + \
        [("blocks.1", f"attention.{o}") for o in "qkpv"]
    # T = 24 < 32 rows: one partial block per (batch, head) matrix, so one scale per (b, h)
    assert all(shape[:2] == (2, HEADS) for _, _, shape in calls)


def test_native_capture_has_one_qk_pv_pair_per_block(tmp_path):
    model = _tiny(2)
    report = algos.quantize(model, "ref_conv", [], dict(FAST), algo=algos.make(SPEC))
    policy = quant_export.build_policy(model, "ref_conv", report, algos.make(SPEC), dict(FAST), [], [], {}, {})
    assert {n for n, s in policy["sites"].items() if s["kind"] == "attention"} == {"blocks.0", "blocks.1"}
    inputs = [(_x(1),), (_x(2),)]
    rep = fx_export.capture_fx(model, inputs, tmp_path / "fx", policy=policy, fresh_process=False, native_fp8="uint8")
    assert rep["status"] == "passed" and rep["native_fp8_detail"]["outputs_identical_to_decode_graph"]
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert [m["module"] for m in inspection["native_fp8_attention_modules"]] == ["blocks.0", "blocks.1"]
    assert all(m["pairs"] == 1 and len(m["quantize_act"]) == 4 for m in inspection["native_fp8_attention_modules"])
    assert inspection["still_emulated"]["emulated_modules"] == {}
    targets = [str(n.target) for n in torch.export.load(tmp_path / "fx" / "model.pt2").graph.nodes]
    assert targets.count("atlas.qk_matmul.default") == 2 and targets.count("atlas.pv_matmul.default") == 2
    assert not any("scaled_dot_product_attention" in t for t in targets)
