"""keep_bf16:rule=input_conv (docs/algos/keep-bf16.md): a structural, opt-in BF16 exception for
convolutions that read image pixels directly."""
import os
from pathlib import Path

import pytest
import torch

import algos
from refutil import need_refs
from test_algo_atlas_v1 import DIT_PATCH, TinyDiT, dit_calib
from test_ladder import FAST, tiny_llama

RULE = "keep_bf16:rule=input_conv"


class TwoStems(torch.nn.Module):
    """Unrelated names: an RGB stem, a grayscale stem (channels=1) and a deeper 3x3 conv."""

    def __init__(self):
        super().__init__()
        self.rgb = torch.nn.Conv2d(3, 8, 3)
        self.gray = torch.nn.Conv2d(1, 8, 3)
        self.body = torch.nn.Conv2d(8, 8, 1)
        self.dw = torch.nn.Conv2d(3, 3, 1, groups=3)


def test_selects_the_rgb_input_conv_only():
    a = algos.make(RULE)
    assert list(a.select(TwoStems())) == ["rgb"]
    assert list(algos.make(RULE + ",channels=1").select(TwoStems())) == ["gray"]
    sel = a.select(TinyDiT())
    assert list(sel) == [DIT_PATCH]                       # the sibling reads 4 channels
    assert "rule=input_conv" in sel[DIT_PATCH] and "dit" not in sel[DIT_PATCH].lower()


def test_selects_nothing_on_tinyllama():
    a = algos.make(RULE)
    assert a.select(tiny_llama(layers=2)) == {}


def test_parameters_are_explicit():
    with pytest.raises(ValueError, match="rule= is required"):
        algos.make("keep_bf16")
    with pytest.raises(ValueError, match="rule= is required"):
        algos.make("keep_bf16:rule=first_conv")
    with pytest.raises(ValueError, match="unknown parameters"):
        algos.make("keep_bf16:rule=input_conv,units=x")


def test_off_by_default():
    # off unless named: the only recipe that names it is the tile-aware atlas_v1_tile (docs/algos/tile-aware.md)
    assert [n for n, r in algos.RECIPES.items() if "keep_bf16" in r["spec"]] == ["atlas_v1_tile"]
    assert algos.make("none") is None


@pytest.mark.parametrize("spec", ["atlas_v1_awq+" + RULE, "atlas_v1+" + RULE, RULE])
def test_applies_exactly_and_records_its_reason(spec):
    need_refs()
    m = TinyDiT()
    w0 = m.get_submodule(DIT_PATCH).weight.detach().clone()
    a = algos.make(spec)
    rep = algos.quantize(m, "ref_conv", algo=a, calib=dit_calib())
    r = rep["algo"]
    assert r["bf16_opt_out"] == r["bf16_opt_out_exact"] == [DIT_PATCH]
    assert r["bf16_opt_out_reasons"][DIT_PATCH].startswith("keep_bf16 rule=input_conv")
    assert type(m.get_submodule(DIT_PATCH)) is torch.nn.Conv2d
    assert torch.equal(m.get_submodule(DIT_PATCH).weight, w0)
    assert type(m.get_submodule(DIT_PATCH + "_extra")).__name__ == "RefQuantConv2d"
    member = next(x for x in r["members"] if x["name"] == "keep_bf16") if "members" in r else r
    assert member["selected"] == [DIT_PATCH] and member["rule"] == "input_conv"
    if "+" in spec:
        assert r["recipe"]["name"] == "atlas_v1_awq"


def test_tinyllama_unchanged_by_the_rule():
    need_refs()
    import copy

    m0 = tiny_llama(layers=2)
    m1 = copy.deepcopy(m0)
    from test_algo_atlas_v1 import IDS, _calib

    algos.quantize(m0, "ref_conv", ["lm_head"], dict(FAST), algo=algos.make("atlas_v1_awq"), calib=_calib())
    rep = algos.quantize(m1, "ref_conv", ["lm_head"], dict(FAST), algo=algos.make("atlas_v1_awq+" + RULE),
                         calib=_calib())
    assert rep["algo"]["bf16_opt_out_exact"] == []
    with torch.no_grad():
        assert torch.equal(m0(IDS).logits, m1(IDS).logits)


DIT = "TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000"


def _dit_cached():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return any((hub / ("models--" + DIT.replace("/", "--"))).glob("snapshots/*/model.safetensors"))


@pytest.mark.slow
@pytest.mark.skipif(not _dit_cached(), reason="Multi-Task DiT checkpoint not in HF_HOME")
def test_selects_the_dit_patch_embedding(monkeypatch):
    pytest.importorskip("lerobot.policies.factory")          # eval_policy.load_policy
    pytest.importorskip("lerobot.policies.multi_task_dit")   # the policy class it resolves
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    import eval_policy as ep

    policy, _ = ep.load_policy(DIT, "cpu", torch.float32)
    assert list(algos.make(RULE).select(policy)) == [DIT_PATCH]
