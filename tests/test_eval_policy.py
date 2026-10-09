"""eval_policy.py: seeded sampling, the repeat determinism check, and --dtype (CPU, no checkpoints)."""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import eval_policy as ep  # noqa: E402


class FakeDiT(nn.Module):
    """Mimics Multi-Task DiT's sampling: an observation history (n_obs_steps > 1) and noise drawn
    from the global RNG inside _generate_actions (modeling_multi_task_dit.py:701-714)."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(n_obs_steps=2)
        self.lin = nn.Linear(3, 3)

    def _prepare_batch(self, batch):
        return batch

    def _generate_actions(self, batch):
        x = torch.randn(1, 4, 3)
        for _ in range(3):  # per-step variance noise, as DDPMScheduler.step draws it
            x = self.lin(x) + 0.1 * torch.randn_like(x)
        return x


def test_seeded_sampling_ignores_prior_global_rng_use():
    pol = FakeDiT()
    batch = {"observation.state": torch.zeros(1, 2, 3)}
    a = ep.predict_chunk(pol, batch, seed=5, device="cpu")
    torch.rand(1000)  # e.g. a profile or data loader consuming the global RNG
    b = ep.predict_chunk(pol, batch, seed=5, device="cpu")
    assert torch.equal(a, b)
    assert not torch.equal(a, ep.predict_chunk(pol, batch, seed=6, device="cpu"))


def test_seeded_sampling_leaves_global_rng_untouched():
    pol = FakeDiT()
    torch.manual_seed(123)
    want = torch.rand(4)
    torch.manual_seed(123)
    ep.predict_chunk(pol, {"observation.state": torch.zeros(1, 2, 3)}, seed=5, device="cpu")
    assert torch.equal(torch.rand(4), want)


def test_seeded_sampling_matches_previous_harness_seeding():
    """Same draws as the old torch.manual_seed(seed + i) before the call, so earlier numbers hold."""
    pol = FakeDiT()
    batch = {"observation.state": torch.zeros(1, 2, 3)}
    torch.manual_seed(9)
    old = pol._generate_actions(batch)
    assert torch.equal(ep.predict_chunk(pol, batch, seed=9, device="cpu"), old)


def test_repeat_agreement():
    p = [torch.randn(4, 3), torch.randn(4, 3)]
    same = ep.repeat_agreement(p, [x.clone() for x in p])
    assert same["deterministic"] is True and same["rel_l2"] == 0.0 and same["max_abs"] == 0.0
    q = [p[0].clone(), p[1] + 1e-3]
    diff = ep.repeat_agreement(p, q)
    assert diff["deterministic"] is False and diff["rel_l2"] > 0
    assert diff["frames_differing"] == 1


def test_repeat_agreement_nan_is_not_deterministic():
    p = [torch.tensor([float("nan"), 1.0])]
    assert ep.repeat_agreement(p, [p[0].clone()])["deterministic"] is True  # bitwise identical
    assert ep.repeat_agreement(p, [torch.tensor([0.0, 1.0])])["deterministic"] is False


def test_determinism_state_fields():
    s = ep.determinism_state()
    for k in ("use_deterministic_algorithms", "deterministic_warn_only", "cudnn_deterministic",
              "cudnn_benchmark", "cublas_workspace_config", "float32_matmul_precision"):
        assert k in s
    assert s["use_deterministic_algorithms"] == torch.are_deterministic_algorithms_enabled()


def test_agreement_per_dim():
    r = [torch.ones(2, 3)]
    p = [torch.ones(2, 3)]
    p[0][:, 2] = -1.0
    a = ep.agreement(p, r)
    assert a["rel_l2_per_dim"][:2] == [0.0, 0.0] and a["rel_l2_per_dim"][2] == pytest.approx(2.0)


def test_dtype_flag():
    ap = ep.build_parser()
    base = ["--policy", "p", "--dataset", "d", "--out", "o.json"]
    assert ap.parse_args(base).dtype == "bfloat16"
    args = ap.parse_args(base + ["--dtype", "float32"])
    assert ep.torch_dtype(args.dtype) is torch.float32
    with pytest.raises(SystemExit):
        ap.parse_args(base + ["--dtype", "float16"])
    assert ap.parse_args(base).determinism_check is True
    assert ap.parse_args(base + ["--no-determinism-check"]).determinism_check is False
