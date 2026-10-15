"""gptq_awq: GPTQ with the Atlas E4M3 quantizer and a pinned exponent, AWQ folds and their refusals,
and calibration that never touches the evaluation data (docs/algos/gptq-awq.md)."""
import copy

import pytest
import torch

import algos
import ref_numerics as rn
from algos import base as algo_hooks
from algos import gptq_awq as ga
from refutil import need_refs
from test_ladder import FAST, ConvNet, tiny_llama


def _batches(n=3, T=12, seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, T), generator=g) for _ in range(n)]


def _run(m, ids):
    return m(ids, use_cache=False)


def _calib(batches, seen=None):
    def run(m, ids):
        if seen is not None:
            seen.append(ids)
        return _run(m, ids)
    return lambda: (batches, run, {"source": "test", "n_samples": len(batches), "seed": 5})


def _spd(K, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * K, K, generator=g)
    X[:, 0] *= 20                                          # a dominant channel: real error feedback
    return X.T @ X


# --------------------------------------------------------------------------------------------
# GPTQ core
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("actorder", [False, True])
def test_identity_hessian_equals_rtn_bit_for_bit(actorder):
    """H = I (and any diagonal H): no cross-column feedback, so GPTQ is round-to-nearest with the
    reference cast at the same exponent, bit for bit."""
    g = torch.Generator().manual_seed(1)
    W = (torch.randn(24, 40, generator=g) * 0.05).to(torch.bfloat16).float()
    s = rn.po2_scale_exp(W)
    rtn = rn.dequantize(*rn.quantize(W))
    for H in (torch.eye(40), torch.diag(torch.rand(40, generator=g) + 0.1)):
        Q = ga.gptq(W, H, s, block=16, actorder=actorder)
        assert torch.equal(Q, rtn)


def test_error_feedback_uses_the_e4m3_grid():
    """Every GPTQ entry lies on the E4M3 x 2^s grid (the quantizer inside the loop is the vpack
    cast, not a uniform grid), and GPTQ lowers the layer-wise error ||(W^ - W) X|| below RTN."""
    g = torch.Generator().manual_seed(2)
    W = (torch.randn(32, 48, generator=g) * 0.05).to(torch.bfloat16).float()
    H = _spd(48)
    s = rn.po2_scale_exp(W)
    Q = ga.gptq(W, H, s, block=16)
    assert torch.equal(ga.qdq(Q, s), Q)
    rtn = ga.qdq(W, s)
    err = lambda A: float((((A - W) @ H) * (A - W)).sum())     # noqa: E731
    assert err(Q) < err(rtn)


def test_actorder_permutation_round_trips():
    """actorder = GPTQ on the columns permuted by descending diag(H), returned in the original order."""
    g = torch.Generator().manual_seed(3)
    W = (torch.randn(16, 32, generator=g) * 0.05).to(torch.bfloat16).float()
    H = _spd(32, seed=4)
    s = rn.po2_scale_exp(W)
    perm = torch.argsort(torch.diag(H), descending=True)
    manual = ga.gptq(W[:, perm], H[perm][:, perm], s, block=8, actorder=False)[:, torch.argsort(perm)]
    assert torch.equal(ga.gptq(W, H, s, block=8, actorder=True), manual)


def test_pinned_exponent_never_saturates():
    """The installed codes reproduce the GPTQ weight exactly at the original exponent s0. The dynamic
    rule on a GPTQ weight whose amax fell below 256 * 2^s0 would pick s0 - 1 and saturate 240 -> 448."""
    algo = algos.make("gptq_awq")
    s0 = -4
    w0 = torch.zeros(4, 8)
    w0[0, 0] = 300 * 2.0 ** s0                             # original amax: rule gives s0
    assert int(rn.po2_scale_exp(w0)) == s0
    out = torch.zeros(4, 8)
    out[0, 0] = 240 * 2.0 ** s0                            # what GPTQ might return: amax < 256 * 2^s0
    out[1, 1] = 1 * 2.0 ** s0
    assert int(rn.po2_scale_exp(out)) == s0 - 1
    algo.H = {"x": torch.eye(8)}
    algo._gptq_cache = {}
    orig = ga.gptq
    try:
        ga.gptq = lambda *a, **k: out.clone()
        q, s, block = algo_hooks.weight_cast(algo, "x", w0, "linear")
    finally:
        ga.gptq = orig
    assert int(s) == s0 and block == 0
    assert torch.equal(rn.dequantize(q, s), out)                       # exact, nothing saturated
    q_dyn, s_dyn = rn.quantize(out)
    assert float(rn.dequantize(q_dyn, s_dyn)[0, 0]) == 448 * 2.0 ** (s0 - 1) != float(out[0, 0])


# --------------------------------------------------------------------------------------------
# On a model
# --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["ref_linear", "reference_hw"])
def test_gptq_installs_its_codes_at_the_reference_exponent(mode):
    need_refs()
    batches = _batches()
    model = tiny_llama(2)
    clean = copy.deepcopy(model)
    algo = algos.make("gptq_awq:block=16")
    rep = algos.quantize(model, mode, ["lm_head"], dict(FAST), algo=algo, calib=_calib(batches))
    rec = rep["algo"]
    assert rec["module_hooks_applied"] and rec["weights"] == {"gptq": 14}
    assert rec["granularity"]["weight"] == {"linear:per_tensor": 14}
    mods, cmods = dict(model.named_modules()), dict(clean.named_modules())
    for n in algo.s0:
        w = cmods[n].weight.detach()
        H = algo.H[algo.owner.get(n, n)]
        s0 = rn.po2_scale_exp(w)
        expect = ga.gptq(w, H, s0, block=16)
        assert torch.equal(mods[n].weight_scale_exp, s0), n
        assert torch.equal(mods[n].weight_codes, rn.e4m3_vpack(expect, s0)), n
    # q/k/v and gate/up share one Hessian each
    assert set(algo.owner) == {f"model.layers.{i}.{p}" for i in range(2)
                               for p in ("self_attn.k_proj", "self_attn.v_proj", "mlp.up_proj")}
    with torch.no_grad():
        assert torch.isfinite(model(batches[0]).logits.float()).all()


def test_gptq_skips_conv2d_and_collab_gets_no_module_hooks():
    need_refs()
    net = ConvNet().to(torch.bfloat16)
    xs = [torch.randn(2, 3, 8, 8).to(torch.bfloat16) for _ in range(2)]
    algo = algos.make("gptq_awq")
    rep = algos.quantize(net, "ref_conv", [], dict(FAST), algo=algo,
                         calib=lambda: (xs, lambda m, x: m(x), {"source": "test", "n_samples": 2, "seed": 0}))
    assert rep["algo"]["weights"] == {"gptq": 1, "conv2d: reference cast (GPTQ applies to Linear only)": 1}
    rep = algos.quantize(tiny_llama(), "collab_v0", ["lm_head"], dict(FAST), algo=algos.make("gptq_awq"),
                         calib=_calib(_batches()))
    assert rep["algo"]["module_hooks_applied"] is False


def test_awq_folds_are_exact_and_o_proj_is_refused():
    """In float32 the folds change no output: norm gamma / s with the consumers' columns * s, and
    up_proj rows / s with down_proj columns * s. o_proj (GQA here) is recorded as unscaled."""
    batches = _batches()
    model = tiny_llama(2).float()
    clean = copy.deepcopy(model)
    algo = algos.make("gptq_awq:method=awq,grid=5")
    algo.calibrate(model, batches, _run)
    prep = algo.prepare(model, "ref_linear")
    groups = {f["group"] for f in prep["awq_folded"]}
    assert groups == {f"model.layers.{i}.{g}" for i in range(2)
                      for g in ("input_layernorm", "post_attention_layernorm", "mlp.up_proj->down_proj")}
    for i in range(2):
        assert "GQA" in prep["awq_unscaled"][f"model.layers.{i}.self_attn.o_proj"]
    changed = sum(not torch.equal(a, b) for a, b in zip(model.parameters(), clean.parameters()))
    assert changed > 0
    with torch.no_grad():
        for ids in batches:
            a, b = model(ids).logits, clean(ids).logits
            assert torch.allclose(a, b, rtol=1e-4, atol=1e-4)


def test_awq_refuses_a_fold_it_cannot_verify():
    """A k_proj that sees a copy of the norm output (not the same tensor) and a down_proj whose input
    is not act(gate) * up are refused with a reason, and nothing is folded for them."""
    batches = _batches()
    model = tiny_llama(1).float()
    layer = model.model.layers[0]
    layer.self_attn.k_proj.register_forward_pre_hook(lambda m, a: (a[0].clone(),))
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, a: (a[0] * 1.5,))
    clean = copy.deepcopy(model)
    algo = algos.make("gptq_awq:method=awq,grid=4")
    algo.calibrate(model, batches, _run)
    prep = algo.prepare(model, "ref_linear")
    un = prep["awq_unscaled"]
    assert "expected" in un["model.layers.0.input_layernorm"]
    assert "act_fn" in un["model.layers.0.mlp.up_proj->down_proj"]
    assert [f["group"] for f in prep["awq_folded"]] == ["model.layers.0.post_attention_layernorm"]
    assert torch.equal(layer.input_layernorm.weight, clean.model.layers[0].input_layernorm.weight)
    assert torch.equal(layer.mlp.down_proj.weight, clean.model.layers[0].mlp.down_proj.weight)


@pytest.mark.parametrize("method", ["awq", "awq_gptq"])
def test_awq_methods_quantize_end_to_end(method):
    need_refs()
    batches = _batches()
    model = tiny_llama(1)
    rep = algos.quantize(model, "ref_linear", ["lm_head"], dict(FAST), algo=algos.make(f"gptq_awq:method={method},grid=4"),
                         calib=_calib(batches))
    w = rep["algo"]["weights"]
    if method == "awq":
        assert w == {"awq": 6, "rtn": 1} and rep["algo"]["granularity"]["weight"] == {"linear:per_tensor": 7}
    else:
        assert w == {"gptq+awq": 6, "gptq": 1}
    with torch.no_grad():
        assert torch.isfinite(model(batches[0]).logits.float()).all()


# --------------------------------------------------------------------------------------------
# Calibration
# --------------------------------------------------------------------------------------------

def test_calibration_only_sees_calibration_batches_and_is_required():
    need_refs()
    batches = _batches(n=2)
    seen = []
    model = tiny_llama(1)
    calls = []
    model.register_forward_pre_hook(lambda m, a, k=None: calls.append(1))
    algo = algos.make("gptq_awq")
    rep = algos.quantize(model, "ref_linear", ["lm_head"], dict(FAST), algo=algo, calib=_calib(batches, seen))
    assert len(seen) == 2 and all(s is b for s, b in zip(seen, batches)) and len(calls) == 2
    assert rep["algo"]["calibration"]["source"] == "test"
    # the same calibration set in the next mode is not run again
    model2 = tiny_llama(1)
    rep2 = algos.quantize(model2, "ref_attn", ["lm_head"], dict(FAST), algo=algo, calib=_calib(batches, seen))
    assert len(seen) == 2 and rep2["algo"]["calibration"]["result"]["reused"]
    with pytest.raises(RuntimeError, match="needs calibration"):
        algos.quantize(tiny_llama(1), "ref_linear", ["lm_head"], dict(FAST), algo=algos.make("gptq_awq"))


def test_llm_calibration_reads_the_train_split(monkeypatch):
    """eval.py's calibration loader (algos.calib.wikitext2_train) asks for the WikiText-2 train split;
    the evaluation reads the test split."""
    import datasets

    from algos import calib

    asked = []

    def fake_load(name, cfg, split):
        asked.append(split)
        return {"text": ["a b c d e f g h i j k l m n o p"] * 4}

    class Tok:
        def __call__(self, text, return_tensors):
            class E:
                input_ids = torch.arange(64).unsqueeze(0)
            return E()

    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    batches, rec = calib.wikitext2_train("unused", 2, 8, 1234, tokenizer=Tok())
    assert asked == ["train"] and "train" in rec["source"] and len(batches) == 2


def test_bad_parameters_are_refused():
    with pytest.raises(ValueError, match="method"):
        algos.make("gptq_awq:method=smoothquant")
    with pytest.raises(ValueError, match="unknown"):
        algos.make("gptq_awq:alpha=0.5")


def test_hessians_param():
    """hessians= is validated, defaults to cpu and is recorded in the calibration summary."""
    import algos

    assert algos.make("gptq_awq:method=gptq").hessians == "cpu"
    assert algos.make("gptq_awq:method=gptq,hessians=device").hessians == "device"
    with pytest.raises(ValueError, match="hessians"):
        algos.make("gptq_awq:method=gptq,hessians=gpu")


def test_gptq_retries_dampening_on_indefinite_hessian():
    """A slightly indefinite H (as float32 rounding can leave a large calibration sum) fails the
    paper's 1% dampening; gptq() retries with 10x, records it, and still lands on the grid. A
    positive-definite H is factored on the first attempt, unchanged."""
    torch.manual_seed(0)
    K = 32
    Qm, _ = torch.linalg.qr(torch.randn(K, K, dtype=torch.float64))
    eig = torch.rand(K, dtype=torch.float64) + 0.5
    eig[0] = -0.05 * eig.mean()                     # below -1% of the mean diagonal
    H = (Qm * eig) @ Qm.T
    W = torch.randn(8, K)
    s = rn.po2_scale_exp(W.abs().amax())
    info = {}
    Q = ga.gptq(W, H, s, block=8, info=info)
    assert info["damp"] == pytest.approx(0.1)
    assert torch.equal(ga.qdq(Q, s), Q)
    good = {}
    ga.gptq(W, (Qm * eig.abs()) @ Qm.T, s, block=8, info=good)
    assert good["damp"] == 0.01


def test_gptq_refuses_hopeless_hessian():
    H = -torch.eye(16, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="not positive-definite"):
        ga.gptq(torch.randn(4, 16), H, 0, block=4)
