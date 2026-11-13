"""gptq_awq:dead_columns=rtn: input channels with exactly zero calibration input are cast by plain
round-to-nearest on the pinned grid instead of being zeroed (docs/algos/gptq-dead-columns.md).
The default (`zero`) is the original code path, bit for bit."""
import pytest
import torch

import algos
import reference.ref_numerics as rn
import exporter.tile_native as tn
from algos import gptq_awq as ga
from algos.base import Scale
from algos.granularity import panel_scale_exp

O, K, ALIVE = 64, 96, 24          # two 32-row panels; 24 of 96 input channels carry calibration signal


def _old_gptq(W, H, s, block=128, damp=0.01, actorder=True, info=None):
    """gptq() as it was before the option (src/algos/gptq_awq.py at the parent commit): frozen copy."""
    s = torch.as_tensor(s, dtype=torch.int32, device=W.device)
    s = s.reshape(-1) if s.dim() == 2 else s
    W = W.detach().clone().float()
    H = H.detach().clone().double().to(W.device)
    K_ = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    perm = None
    if actorder:
        perm = torch.argsort(torch.diag(H), descending=True)
        W, H = W[:, perm], H[perm][:, perm]
    eye = torch.eye(K_, dtype=H.dtype, device=H.device)
    mean_diag = torch.mean(torch.diag(H))
    Hinv = None
    for attempt in range(ga.DAMP_RETRIES):
        d = damp * 10 ** attempt
        L, err = torch.linalg.cholesky_ex(H + d * mean_diag * eye)
        if int(err) == 0:
            U, err = torch.linalg.cholesky_ex(torch.cholesky_inverse(L), upper=True)
            if int(err) == 0:
                Hinv = U.float()
                break
    if info is not None:
        info["damp"] = d
    Q = torch.zeros_like(W)
    for i1 in range(0, K_, block):
        i2 = min(i1 + block, K_)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            q = ga.qdq(w, s)
            Q[:, i1 + i] = q
            err = (w - q) / Hinv1[i, i]
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    if perm is not None:
        Q = Q[:, torch.argsort(perm)]
    return Q


def _layer(seed=0):
    """W [O, K] (BF16 values), H from calibration inputs whose channels outside `alive` are exactly 0."""
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(O, K, generator=g) * 0.05).to(torch.bfloat16).float()
    alive = torch.randperm(K, generator=g)[:ALIVE].sort().values
    X = torch.zeros(4 * K, K)
    X[:, alive] = torch.randn(4 * K, ALIVE, generator=g)
    X[:, alive[0]] *= 20                                   # a dominant channel: real error feedback
    dead = torch.ones(K, dtype=torch.bool)
    dead[alive] = False
    return W, X.T @ X, dead, panel_scale_exp(W, 32, "nc")


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("actorder", [False, True])
@pytest.mark.parametrize("block", [16, 128])
def test_default_is_bit_identical_to_the_original_routine(actorder, block):
    W, H, dead, s = _layer()
    ref = _old_gptq(W, H, s, block=block, actorder=actorder)
    assert torch.equal(ga.gptq(W, H, s, block=block, actorder=actorder), ref)
    assert torch.equal(ga.gptq(W, H, s, block=block, actorder=actorder, dead_columns="zero"), ref)
    assert float(ref[:, dead].abs().max()) == 0.0          # the behaviour being kept: dead columns are zero


def test_default_algo_path_and_record_are_unchanged():
    a = algos.make("gptq_awq:method=gptq")
    assert a.dead_columns == "zero" and "dead_columns" not in a.record()
    assert "producer_aware" not in a.record()
    W, H, dead, s = _layer(1)
    a.H["lin"] = H
    a.weight_grid = lambda name, w, kind: Scale(s, "panel", dynamic=False)
    out = a.transform_weight("lin", W.to(torch.bfloat16), "linear")
    assert torch.equal(out.float(), _old_gptq(W, H, s, block=a.block, damp=a.damp, actorder=a.actorder))


@pytest.mark.parametrize("actorder", [False, True])
def test_rtn_dead_columns_are_plain_rtn_and_live_columns_unchanged(actorder):
    W, H, dead, s = _layer(2)
    zero = ga.gptq(W, H, s, block=16, actorder=actorder)
    info = {}
    rtn = ga.gptq(W, H, s, block=16, actorder=actorder, dead_columns="rtn", info=info)
    plain = ga.qdq(W, s)                                   # plain round-to-nearest on the same pinned grid
    assert info["dead"] == int(dead.sum()) == K - ALIVE
    assert torch.equal(rtn[:, dead], plain[:, dead])       # dead columns = RTN, not zero
    assert float(rtn[:, dead].abs().max()) > 0
    # Live columns: dead columns feed nothing back (their error is not propagated, and H has no
    # cross terms for them), so the live columns are bit-identical to the default
    assert torch.equal(rtn[:, ~dead], zero[:, ~dead])
    assert not torch.equal(rtn[:, ~dead], plain[:, ~dead])  # and GPTQ still did something to them
    # every entry is on the pinned grid
    assert torch.equal(ga.qdq(rtn, s), rtn)


def test_rtn_weight_error_on_a_gaussian_probe_is_near_round_to_nearest():
    W, H, dead, s = _layer(3)
    G = torch.randn(2048, K, generator=torch.Generator().manual_seed(9))
    plain = ga.qdq(W, s)
    zero = ga.gptq(W, H, s, block=16)
    rtn = ga.gptq(W, H, s, block=16, dead_columns="rtn")
    e_plain, e_zero, e_rtn = _rel(plain, W), _rel(zero, W), _rel(rtn, W)
    assert e_zero > 0.8 and e_plain < 0.05                 # the bug: 3/4 of the columns are gone
    assert e_rtn < 1.5 * e_plain                           # GPTQ error only on the live columns
    assert _rel(G @ rtn.T, G @ W.T) < 1.5 * _rel(G @ plain.T, G @ W.T)
    assert _rel(G @ zero.T, G @ W.T) > 0.8
    # on the calibration data itself GPTQ does not get worse than RTN for the live part
    live = ~dead
    Hl = H[live][:, live]
    d_rtn, d_plain = (rtn - W)[:, live], (plain - W)[:, live]
    assert float(((d_rtn @ Hl) * d_rtn).sum()) <= float(((d_plain @ Hl) * d_plain).sum())


def test_rtn_with_no_dead_column_is_the_default():
    W, H, dead, s = _layer(4)
    H = H + torch.eye(K)                                   # every channel alive
    assert torch.equal(ga.gptq(W, H, s, block=16), ga.gptq(W, H, s, block=16, dead_columns="rtn"))


def test_scalar_exponent_dead_columns():
    W, H, dead, _ = _layer(5)
    s = rn.po2_scale_exp(W)
    rtn = ga.gptq(W, H, s, block=16, dead_columns="rtn")
    assert torch.equal(rtn[:, dead], ga.qdq(W, s)[:, dead])


@pytest.mark.parametrize("dead_columns", ["zero", "rtn"])
def test_exponent_grid_is_unchanged_and_codes_keep_the_tile_contract(dead_columns):
    W, H, dead, s = _layer(6)
    Wb = W.to(torch.bfloat16)
    a = algos.make(f"gptq_awq:method=gptq,dead_columns={dead_columns}")
    a.H["lin"] = H
    a.weight_grid = lambda name, w, kind: Scale(panel_scale_exp(w, 32, "nc"), "panel", dynamic=False)
    out = a.transform_weight("lin", Wb, "linear")
    sc = a.weight_scale("lin", out, "linear")
    assert torch.equal(sc.exp, s) and not sc.dynamic       # pinned exponent = panel nc of the input weight
    assert torch.equal(a.s0["lin"], panel_scale_exp(Wb.float(), 32, "nc"))
    assert tn.check_constant_groups(sc.exp, 32).ok         # constant per 32-row panel
    codes = rn.e4m3_vpack(out.float(), sc.exp)
    assert tn.reserved_code_count(codes) == 0              # no 0x7F / 0xFF
    assert torch.equal(rn.e4m3_unpack(codes) * torch.exp2(sc.exp.float()), out.float())   # codes decode to the weight
    live_codes_zero = ((codes & 0x7F) == 0)
    if dead_columns == "zero":
        assert bool(live_codes_zero[:, dead].all())
    else:
        assert float(live_codes_zero[:, dead].float().mean()) < 0.01    # dead columns carry real codes
    # the grid does not depend on the option: same exponent as the default run
    b = algos.make("gptq_awq:method=gptq")
    b.H["lin"] = H
    b.weight_grid = a.weight_grid
    b.transform_weight("lin", Wb, "linear")
    assert torch.equal(b.s0["lin"], a.s0["lin"])


def test_option_parsing_and_record():
    assert algos.make("gptq_awq:method=gptq").dead_columns == "zero"
    a = algos.make("gptq_awq:method=awq_gptq,actorder=true,dead_columns=rtn")
    assert a.dead_columns == "rtn" and a.record()["dead_columns"] == "rtn" and a.record()["params"]["dead_columns"] == "rtn"
    s = algos.make("gptq_awq:method=gptq,dead_columns=rtn+granularity:w=panel,a=block,rule=nc")   # inside a stack
    assert s.members[0].dead_columns == "rtn"
    assert algos.make("gptq_awq:method=gptq,dead_columns=zero").dead_columns == "zero"
    for bad in ("gptq_awq:dead_columns=drop", "gptq_awq:dead_columns=true", "gptq_awq:dead_column=rtn"):
        with pytest.raises(ValueError):
            algos.make(bad)
    with pytest.raises(ValueError):
        ga.gptq(torch.zeros(2, 2), torch.eye(2), 0, dead_columns="drop")
