"""gptq_awq:producer_aware=true: the AWQ search also scores the producer's quantization error on the
producer's own weight grid (docs/algos/tile-aware.md#producer-aware-awq). Default off is bit-identical."""
import copy

import pytest
import torch

import algos
import reference.ref_numerics as rn
from algos import gptq_awq as ga
from algos.granularity import panel_scale_exp
from refutil import need_refs
from test_algo_gptq_awq import FAST, _batches, _calib, _run
from test_ladder import tiny_llama

HOT = [5, 6, 7]
PANEL = lambda W: panel_scale_exp(W, 32, "nc")           # noqa: E731  (w=panel, rows=32, rule=nc)


def _pair(seed=0, I=64, K=48, O=40):
    """up_proj [I, K] -> down_proj [O, I]; three intermediate channels carry ~10^4x the activation
    magnitude of the rest, so AWQ wants a large s_j for them."""
    g = torch.Generator().manual_seed(seed)
    Wu = (torch.randn(I, K, generator=g) * 0.05).to(torch.bfloat16)
    Wd = (torch.randn(O, I, generator=g) * 0.05).to(torch.bfloat16)
    X = torch.randn(512, K, generator=g)
    sx = (torch.rand(I, generator=g) + 0.5) * 0.01
    sx[HOT] = 100.0
    Hd = torch.diag(sx ** 2) * 512 / 3
    return Wu, Wd, X.T @ X, Hd, sx


def _row_err(Wu, s):
    """Relative error of the producer's effective rows s_j * Q(Wu_j / s_j) on the panel grid."""
    Wpc = (Wu.float() / s[:, None]).to(Wu.dtype).float()
    eff = ga.qdq(Wpc, PANEL(Wpc)) * s[:, None]
    return (eff - Wu.float()).norm(dim=1) / Wu.float().norm(dim=1)


def test_old_search_damages_salient_rows_new_search_does_not():
    Wu, Wd, Hp, Hd, sx = _pair()
    s_old, a_old, _ = ga.awq_search([Wd], Hd, sx, 20, PANEL)
    c = ga.producer_row_weights(Wu, Hp, Wd, torch.diag(Hd))
    prod = {"W": Wu, "H": Hp, "c": c, "exp_fn": PANEL}
    s_new, a_new, _ = ga.awq_search([Wd], Hd, sx, 20, PANEL, producer=prod)
    r_old, r_new = _row_err(Wu, s_old), _row_err(Wu, s_new)
    assert int((r_old > 0.1).sum()) >= len(HOT) and all(float(r_old[j]) > 0.1 for j in HOT)
    assert int((r_new > 0.1).sum()) == 0
    assert a_new < a_old                       # the producer term moves the search to a smaller alpha
    # the old rows are fine on a per-channel grid: it is the shared panel exponent that breaks them
    Wpc = (Wu.float() / s_old[:, None]).to(Wu.dtype).float()
    chan = ga.qdq(Wpc, torch.stack([rn.po2_scale_exp(r) for r in Wpc]).reshape(-1, 1)) * s_old[:, None]
    assert int((((chan - Wu.float()).norm(dim=1) / Wu.float().norm(dim=1)) > 0.1).sum()) == 0


def test_producer_none_is_the_unchanged_search():
    """With producer=None (the default) awq_search is bit-identical to its pre-option body."""
    Wu, Wd, Hp, Hd, sx = _pair(seed=3)

    def old(Ws, H, sx, grid, exp_fn):
        H = H.float().to(Ws[0].device)
        best = (float("inf"), None, None)
        for i in range(grid):
            a = i / grid
            s = sx.float().pow(a).clamp(min=1e-4)
            s = s / (s.max() * s.min()).sqrt()
            loss = 0.0
            for W in Ws:
                Wsc = (W.float() * s).to(W.dtype).float()
                dW = (ga.qdq(Wsc, exp_fn(Wsc)) - Wsc) / s
                loss += float(((dW @ H) * dW).sum())
            if loss < best[0]:
                best = (loss, s, a)
        return best[1], best[2], best[0]

    for exp_fn in (None, PANEL):
        s0, a0, l0 = old([Wd], Hd, sx, 20, exp_fn or rn.po2_scale_exp)
        s1, a1, l1 = ga.awq_search([Wd], Hd, sx, 20, exp_fn)
        assert torch.equal(s0, s1) and a0 == a1 and l0 == l1


def test_row_weights_zero_for_dead_channels_and_scale_with_the_consumer():
    Wu, Wd, Hp, Hd, sx = _pair()
    Wu = Wu.clone()
    Wu[9] = 0
    c = ga.producer_row_weights(Wu, Hp, Wd, torch.diag(Hd))
    assert c.shape == (Wu.shape[0],) and float(c[9]) == 0.0 and bool((c[:9] > 0).all())
    c2 = ga.producer_row_weights(Wu, Hp, Wd, 4 * torch.diag(Hd))
    assert torch.allclose(c2, 4 * c)


def test_option_parsing_and_guards():
    assert algos.make("gptq_awq:method=awq").producer_aware is False
    assert algos.make("gptq_awq:method=awq,producer_aware=true").producer_aware is True
    assert algos.make("gptq_awq:method=awq,producer_aware=false").producer_aware is False
    with pytest.raises(ValueError, match="awq_loss=weight only"):
        algos.make("gptq_awq:method=awq,producer_aware=true,awq_loss=cast")
    with pytest.raises(ValueError, match="unknown parameters"):
        algos.make("gptq_awq:method=awq,producer_awar=true")
    assert "producer_aware" not in algos.make("gptq_awq:method=awq").record()
    assert algos.make("gptq_awq:method=awq,producer_aware=true").record()["producer_aware"] is True


def _prepared(spec, seed_model=True):
    batches = _batches()
    model = tiny_llama(1).float()
    algo = algos.make(spec)
    algo = getattr(algo, "members", [algo])[0]       # a stack's gptq_awq member (algos.quantize interleaves hooks)
    algo.calibrate(model, batches, _run)
    prep = algo.prepare(model, "ref_linear")
    return model, algo, prep


@pytest.mark.parametrize("stack", ["", "+granularity:w=panel,rows=32,rule=nc"])
def test_default_off_prepare_is_bit_identical(stack):
    ma, _, pa = _prepared("gptq_awq:method=awq,grid=6" + stack)
    mb, _, pb = _prepared("gptq_awq:method=awq,grid=6,producer_aware=false" + stack)
    for (n, a), (_, b) in zip(ma.state_dict().items(), mb.state_dict().items()):
        assert torch.equal(a, b), n
    assert pa == pb


def test_on_changes_only_the_up_down_groups_and_keeps_the_fold_lossless():
    spec = "gptq_awq:method=awq,grid=6%s+granularity:w=panel,rows=32,rule=nc"
    off, algo_off, _ = _prepared(spec % "")
    on, algo, _ = _prepared(spec % ",producer_aware=true")
    layer = "model.layers.0."
    # norm groups are untouched by the option
    assert torch.equal(on.state_dict()[layer + "post_attention_layernorm.weight"],
                       off.state_dict()[layer + "post_attention_layernorm.weight"])
    picks = lambda a: a.alpha_picks[layer + "post_attention_layernorm"]          # noqa: E731
    assert picks(algo) == picks(algo_off)
    # the pair is still a pure rescale: up[j] / s_j and down[:, j] * s_j cancel in the composite
    # (up's input columns also carry the norm group's s_in, which the composite keeps)
    sd = tiny_llama(1).float().state_dict()
    s_in = algo.awq_scale[layer + "mlp.up_proj"].float()
    got = on.state_dict()
    comp = got[layer + "mlp.down_proj.weight"] @ got[layer + "mlp.up_proj.weight"]
    want = (sd[layer + "mlp.down_proj.weight"] @ sd[layer + "mlp.up_proj.weight"]) * s_in[None, :]
    assert torch.allclose(comp, want, rtol=1e-3, atol=1e-6)


def test_awq_gptq_end_to_end_with_producer_aware_on_panel_grid():
    need_refs()
    batches = _batches()
    model = tiny_llama(1)
    algo = algos.make("gptq_awq:method=awq_gptq,grid=4,producer_aware=true+granularity:w=panel,rows=32,rule=nc")
    rep = algos.quantize(model, "ref_linear", ["lm_head"], dict(FAST), algo=algo, calib=_calib(batches))
    member = next(m for m in rep["algo"]["members"] if m["name"] == "gptq_awq")
    assert member["weights"] == {"gptq+awq": 6, "gptq": 1}
    assert member["producer_aware"] is True
    with torch.no_grad():
        assert torch.isfinite(model(batches[0]).logits.float()).all()


def test_diagonal_h_and_identity_producer_hessian_match_the_dense_forms():
    """The weights-only proxy used by src/scripts/tile_diag/awq_producer_validate.py: a 1-D H is the diagonal of the
    dense H, and a producer H of None is the identity."""
    Wu, Wd, Hp, Hd, sx = _pair(seed=1)
    dense = torch.diag(sx ** 2) * 512 / 3
    c = ga.producer_row_weights(Wu, None, Wd, sx ** 2 * 512 / 3)
    c_dense = ga.producer_row_weights(Wu, torch.eye(Wu.shape[1]), Wd, torch.diag(dense))
    assert torch.allclose(c, c_dense, rtol=1e-6)
    a = ga.awq_search([Wd], dense, sx, 8, PANEL, producer={"W": Wu, "H": torch.eye(Wu.shape[1]), "c": c, "exp_fn": PANEL})
    b = ga.awq_search([Wd], sx ** 2 * 512 / 3, sx, 8, PANEL, producer={"W": Wu, "H": None, "c": c, "exp_fn": PANEL})
    assert a[1] == b[1] and torch.equal(a[0], b[0]) and abs(a[2] - b[2]) <= 1e-4 * abs(a[2])


def test_producer_hessian_follows_the_norm_group_fold():
    """prepare folds the norm group first (up_proj columns * s_in, input x / s_in): the producer term must use
    H' = D H D with the folded weight, which equals the unfolded weight against the unfolded H."""
    Wu, Wd, Hp, Hd, sx = _pair(seed=2)
    g = torch.Generator().manual_seed(7)
    s_in = torch.exp2(torch.randn(Wu.shape[1], generator=g) * 2)             # a strongly non-flat s_in
    Wf = (Wu.float() * s_in).float()                                         # float weights: the fold is exact
    Di = 1.0 / s_in.double()
    Hf = (Hp.double() * Di[None, :] * Di[:, None]).float()
    c0 = ga.producer_row_weights(Wu.float(), Hp, Wd, torch.diag(Hd))
    c1 = ga.producer_row_weights(Wf, Hf, Wd, torch.diag(Hd))
    assert torch.allclose(c0, c1, rtol=1e-3)
    wrong = ga.producer_row_weights(Wf, Hp, Wd, torch.diag(Hd))              # the unfolded H against the folded weight
    assert not torch.allclose(c0, wrong, rtol=1e-2)

    class _Lin:
        def __init__(self, w): self.weight = w
    algo = algos.make("gptq_awq:method=awq,producer_aware=true")
    algo.H = {"up": Hp.clone()}
    algo.owner = {}
    algo.awq_scale = {"up": s_in}
    chk = {"rows": True, "producer": "up"}
    term = algo._producer_term({"up": _Lin(Wf)}, chk, ["down"], [Wd], Hd)
    assert torch.allclose(term["H"], Hf, rtol=1e-4, atol=0) and torch.allclose(term["c"], c1)
    algo.awq_scale = {}                                                       # no norm fold: H untouched
    assert torch.equal(algo._producer_term({"up": _Lin(Wu)}, chk, ["down"], [Wd], Hd)["H"], Hp)
