"""ckpt_keys: repairing checkpoint keys nested one module level deeper by the current model."""

import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ckpt_keys  # noqa: E402


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.norm = nn.LayerNorm(4)


class Wrapped(nn.Module):
    """A wrapper that adds one level (`inner_model`), as newer CLIP wrappers do."""

    def __init__(self):
        super().__init__()
        self.inner_model = Inner()


class New(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Module()
        self.enc.model = Wrapped()
        self.head = nn.Linear(4, 2)


def old_checkpoint(new):
    """The state dict an older library would have saved: no `inner_model` level, other values."""
    torch.manual_seed(1)
    sd = {}
    for k, v in new.state_dict().items():
        sd[k.replace("enc.model.inner_model.", "enc.model.")] = torch.randn_like(v)
    return sd


def test_one_inserted_segment_is_repaired_and_loaded():
    torch.manual_seed(0)
    model = New()
    ckpt = old_checkpoint(model)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    assert len(missing) == len(unexpected) == 4  # the silent failure being repaired
    rec = ckpt_keys.repair_state_dict(model, ckpt)
    assert rec == {"n_remapped": 4, "rules": [
        {"from_prefix": "enc.model", "to_prefix": "enc.model.inner_model", "n_keys": 4}]}
    assert torch.equal(model.enc.model.inner_model.fc.weight, ckpt["enc.model.fc.weight"])
    assert torch.equal(model.head.weight, ckpt["head.weight"])


def test_matching_checkpoint_is_a_no_op():
    model = New()
    assert ckpt_keys.repair_state_dict(model, model.state_dict()) == {"n_remapped": 0, "rules": []}


def test_missing_key_without_partner_raises():
    model = New()
    ckpt = old_checkpoint(model)
    del ckpt["enc.model.fc.bias"]
    with pytest.raises(ckpt_keys.CheckpointKeyError, match="enc.model.inner_model.fc.bias"):
        ckpt_keys.repair_state_dict(model, ckpt)


def test_inconsistent_segment_under_a_prefix_raises():
    """Two keys under the same prefix would need different inserted segments: no bijection."""
    model = New()
    ckpt = old_checkpoint(model)
    ckpt["enc.model.extra.weight"] = torch.zeros(1)  # under enc.model but maps onto nothing
    with pytest.raises(ckpt_keys.CheckpointKeyError, match="unexpected unmatched"):
        ckpt_keys.repair_state_dict(model, ckpt)


def test_unexpected_only_raises():
    model = New()
    ckpt = dict(model.state_dict())
    ckpt["stray.weight"] = torch.zeros(1)
    with pytest.raises(ckpt_keys.CheckpointKeyError, match="stray.weight"):
        ckpt_keys.repair_state_dict(model, ckpt)


def test_shape_mismatch_raises():
    model = New()
    ckpt = old_checkpoint(model)
    ckpt["enc.model.fc.weight"] = torch.zeros(5, 5)
    with pytest.raises(ckpt_keys.CheckpointKeyError, match="shape"):
        ckpt_keys.repair_state_dict(model, ckpt)


def test_tied_weights_stored_once_are_not_missing():
    model = New()
    model.extra = nn.Linear(4, 2)
    model.extra.weight = model.head.weight  # tied
    ckpt = {k: v.clone() for k, v in model.state_dict().items() if k != "extra.weight"}
    assert ckpt_keys.repair_state_dict(model, ckpt) == {"n_remapped": 0, "rules": []}


def test_repair_checkpoint_keys_reads_a_local_dir(tmp_path):
    from safetensors.torch import save_file

    model = New()
    ckpt = old_checkpoint(model)
    save_file(ckpt, str(tmp_path / "model.safetensors"))
    rec = ckpt_keys.repair_checkpoint_keys(model, str(tmp_path))
    assert rec["n_remapped"] == 4 and rec["new"] is True
    assert torch.equal(model.enc.model.inner_model.norm.bias, ckpt["enc.model.norm.bias"])


DIT = "TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000"


def _dit_cached():
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return any((hub / ("models--" + DIT.replace("/", "--"))).glob("snapshots/*/model.safetensors"))


@pytest.mark.slow
@pytest.mark.skipif(not _dit_cached(), reason="Multi-Task DiT checkpoint not in HF_HOME")
def test_dit_checkpoint_loads_completely(monkeypatch):
    pytest.importorskip("lerobot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    from safetensors import safe_open

    import eval_policy as ep

    policy, _ = ep.load_policy(DIT, "cpu", torch.float32)
    rec = policy.checkpoint_key_repair
    assert rec["n_remapped"] == 395
    file = ckpt_keys.resolve_checkpoint_file(DIT)
    with safe_open(file, framework="pt") as f:
        keys = list(f.keys())
        ref = f.get_tensor("observation_encoder.vision_encoder.model.encoder.layers.0.self_attn.q_proj.weight")
    renamed = set()
    for k in keys:
        for r in rec["rules"]:
            if k.startswith(r["from_prefix"] + "."):
                k = r["to_prefix"] + k[len(r["from_prefix"]):]
                break
        renamed.add(k)
    missing, unexpected = ckpt_keys.key_diff(policy.state_dict(), renamed)
    assert not missing and not unexpected
    got = policy.state_dict()["observation_encoder.vision_encoder.model.vision_model.encoder.layers.0.self_attn.q_proj.weight"]
    assert torch.equal(got, ref.to(got.dtype))
