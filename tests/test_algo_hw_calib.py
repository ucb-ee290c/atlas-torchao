"""gptq_awq hardware-aware calibration (NEW opt-ins awq_loss=cast, calib=chip, target=fp; off by default)."""
import copy

import pytest
import torch

import algos
import reference.ref_numerics as rn
from algos import gptq_awq as ga
from test_ladder import tiny_llama


def _batches(n=3, T=12, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]


def _run(m, ids):
    return m(ids, use_cache=False)


def _flushed(xh, x):
    return int(((xh == 0) & (x != 0)).sum())


def test_cast_loss_prefers_alpha_that_cuts_flushed_activations():
    """One outlier channel makes every token's amax large, so the other channels flush under the
    per-token cast; the output-error loss picks an alpha > 0 that shrinks the outlier, and flushes less
    than alpha = 0."""
    g = torch.Generator().manual_seed(0)
    T, K, N = 256, 64, 32
    X = torch.randn(T, K, generator=g)
    X[:, 0] *= 600.0                                         # outlier channel
    X = X.to(torch.bfloat16).float()
    W = (torch.randn(N, K, generator=g) * 0.05).to(torch.bfloat16)
    sx = X.abs().mean(0)
    s, alpha, loss, _ = ga.awq_search_cast([W], X, sx, grid=10)
    assert alpha > 0
    f0 = _flushed(ga.act_cast_nc(X.to(torch.bfloat16)), X)
    f1 = _flushed(ga.act_cast_nc((X / s).to(torch.bfloat16)), X)
    assert f1 < f0
    _, a0_loss = ga.awq_search_cast([W], X, sx, grid=1)[1:3]   # grid=1 -> alpha 0 only
    assert loss < a0_loss


def test_identity_cast_reproduces_the_bf16_hessian(monkeypatch):
    monkeypatch.setattr(ga, "ACT_CAST", lambda x: x.float())
    batches = _batches()
    model = tiny_llama(1).float()
    algo = algos.make("gptq_awq:method=gptq,calib=chip")
    algo.calibrate(model, batches, _run)
    algo.prepare(model, "ref_linear")
    assert set(algo.Hhat) == set(algo.H) and algo.H
    for n in algo.H:
        assert torch.allclose(algo.Hhat[n], algo.H[n], rtol=1e-5, atol=1e-5), n


def test_chip_pass_runs_on_the_folded_model_and_target_fp_is_finite():
    batches = _batches()
    model = tiny_llama(1).float()
    algo = algos.make("gptq_awq:method=awq_gptq,grid=4,awq_loss=cast,calib=chip,target=fp")
    algo.calibrate(model, batches, _run)
    prep = algo.prepare(model, "ref_linear")
    assert prep["alpha_picks"] and all("cast" in v for v in prep["alpha_picks"].values())
    assert set(algo.Cxx) == set(algo.Hhat)
    mods = dict(model.named_modules())
    for n in algo.H:
        w = algo.transform_weight(n, mods[n].weight.detach(), "linear")
        assert torch.isfinite(w.float()).all()
    for n in algo.s0:                                        # pinned exponent still exact
        assert torch.equal(ga.qdq(mods[n].weight.new_zeros(1), algo.s0[n]), torch.zeros(1))


def test_defaults_are_off_and_validated():
    a = algos.make("gptq_awq:method=awq_gptq")
    assert (a.awq_loss, a.calib, a.target) == ("weight", "bf16", "bf16")
    a.calibrate(tiny_llama(1).float(), _batches(), _run)
    assert not a.subsample                                    # no subsample kept by default
    for bad in ("gptq_awq:awq_loss=x", "gptq_awq:calib=x", "gptq_awq:target=fp"):
        with pytest.raises(ValueError):
            algos.make(bad)
