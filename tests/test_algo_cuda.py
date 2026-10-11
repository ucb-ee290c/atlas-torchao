"""CUDA vs CPU for every merged algorithm (and a stack): calibrate + prepare + one forward of a tiny
Llama on each device gives the same scale exponents and E4M3 codes where the algorithm is
deterministic. Skipped without CUDA (follows test_algo_smoothquant.py::test_static_on_cuda_matches_cpu).

GPTQ is not bit-deterministic across devices (Cholesky and the error feedback accumulate in a
different order), so for gptq_awq the pinned exponents must match exactly and the codes in at
least CODE_AGREE of the entries (NEW tolerance)."""
import pytest
import torch

import algos
from algos import base as algo_hooks
from refutil import need_refs
from test_ladder import FAST, tiny_llama

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

CODE_AGREE = 0.99
SPECS = {
    "granularity": ("granularity:w=channel,a=token,rule=nc,attn=token", True),
    "mixed_precision": ("mixed_precision:units=model.layers.0.mlp.down_proj,keep_as=fp8x3", True),
    "static_calib": ("static_calib:method=mse,weight=mse", True),
    "smoothquant": ("smoothquant:alpha=0.5,po2=true,gain=true,static=true", True),
    "gptq": ("gptq_awq:method=gptq,actorder=true", False),
    "awq_gptq": ("gptq_awq:method=awq_gptq,actorder=true", False),
    "stack_dynamic": ("smoothquant:alpha=0.5,po2=true,gain=false+gptq_awq:method=gptq,actorder=true"
                      "+granularity:w=channel,a=token,rule=nc", False),
    "stack_static": ("smoothquant:alpha=0.5,po2=true,gain=true,static=true"
                     "+gptq_awq:method=gptq,actorder=true", False),
}


def _batches(dev, n=2, T=12, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, T), generator=g).to(dev) for _ in range(n)]


def _run_on(dev, spec, monkeypatch):
    """(weights {fqn: (codes, exp)}, activation exponents in call order, logits) on `dev`."""
    model = tiny_llama(layers=2).to(dev)
    batches = _batches(dev)
    acts = []
    real = algo_hooks.cast_with

    def spy(x, scale):
        acts.append(scale.exp.detach().cpu().clone())
        return real(x, scale)

    algo = algos.make(spec)
    rep = algos.quantize(model, "ref_linear", ["lm_head"], dict(FAST), algo=algo,
                         calib=lambda: (batches, lambda m, ids: m(ids, use_cache=False),
                                        {"source": "test", "n_samples": len(batches), "seed": 5}))
    monkeypatch.setattr(algo_hooks, "cast_with", spy)
    with torch.no_grad():
        y = model(batches[0], use_cache=False).logits
    monkeypatch.setattr(algo_hooks, "cast_with", real)
    w = {n: (m.weight_codes.cpu(), m.weight_scale_exp.cpu()) for n, m in model.named_modules()
         if hasattr(m, "weight_codes")}
    return w, acts, y.float().cpu(), rep["algo"]


@pytest.mark.parametrize("key", list(SPECS))
def test_cuda_matches_cpu(key, monkeypatch):
    need_refs()
    spec, exact = SPECS[key]
    w_c, a_c, y_c, r_c = _run_on("cpu", spec, monkeypatch)
    w_g, a_g, y_g, r_g = _run_on("cuda", spec, monkeypatch)
    assert torch.isfinite(y_g).all()
    assert w_c.keys() == w_g.keys() and w_c
    for n in w_c:
        assert torch.equal(w_c[n][1], w_g[n][1]), f"{n}: weight exponent differs"
        if exact:
            assert torch.equal(w_c[n][0], w_g[n][0]), f"{n}: codes differ"
        else:
            agree = (w_c[n][0] == w_g[n][0]).float().mean().item()
            assert agree >= CODE_AGREE, f"{n}: only {agree:.4f} of the codes agree"
    assert len(a_c) == len(a_g)
    if exact:
        for i, (a, b) in enumerate(zip(a_c, a_g)):
            assert torch.equal(a, b), f"activation cast {i}: exponent differs"
    assert r_c["granularity"] == r_g["granularity"]
