"""Tile-aware FP8 scales (docs/algos/tile-aware.md): w=panel, a=block, attn=block, the tile_native
verifier, the three atlas_v1_*tile* recipes, GPTQ / AWQ on a panel grid, the hardware-equivalence of
pop-scaled matmul, and the native FX graph."""
import copy
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
import exporter.fx_export as fx_export
import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export
import reference.ref_numerics as rn
import exporter.tile_native as tn
from algos import base as algo_hooks
from algos import granularity as gr
from algos import gptq_awq as ga
from refutil import need_refs
from test_export import _calib, _quantize_export, _reload
from test_ladder import FAST, tiny_llama

TILE_RECIPES = ("atlas_v1_awq_tile", "atlas_v1_tile", "atlas_v1_awq_tile_static")
# attn=block is an option, not part of a recipe: q and p per 32-row block of each (batch, head) matrix
ATTN_BLOCK = "gptq_awq:method=awq_gptq,actorder=true+granularity:w=panel,a=block,attn=block,rule=nc"
EXPORT_SPECS = TILE_RECIPES + (ATTN_BLOCK,)
ROWS16 = ATTN_BLOCK.replace("rule=nc", "rows=16,rule=nc")       # a non-default group size travels in the nodes
OLD_RECIPES = ("atlas_v1_awq", "atlas_v1_sq", "atlas_v1_static", "atlas_v1")
INPUTS = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
          (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _g(spec):
    return algos.make(spec)


def _channel_scaled(shape, gen):
    """A weight whose channels span many binades (so a channel's own exponent is below its group's)."""
    w = torch.randn(shape, generator=gen)
    return (w * torch.exp2(torch.randint(-4, 5, (shape[0], 1), generator=gen).float())).to(torch.bfloat16)


def _group_scaled(shape, gen, rows=32, steps=(-4, 0, 4, 2)):
    """Rows scaled by one power of two per group of `rows` (cycled from `steps`): group exponents differ."""
    s = torch.tensor([steps[i // rows % len(steps)] for i in range(shape[0])], dtype=torch.float32).reshape(-1, 1)
    return (torch.randn(shape, generator=gen) * torch.exp2(s)).to(torch.bfloat16)


# -- (a) panel exponents ------------------------------------------------------------------------------

@pytest.mark.parametrize("rule", ["ref", "nc"])
@pytest.mark.parametrize("out", [64, 70, 32, 5])
def test_panel_exponent_is_group_rule_of_amax_and_dominates_members(rule, out):
    w = _channel_scaled((out, 48), torch.Generator().manual_seed(out))
    sc = _g(f"granularity:w=panel,rule={rule}").weight_scale("l", w, "linear")
    e = sc.exp
    assert e.shape == (out, 1) and e.dtype == torch.int32 and not sc.dynamic
    chk = tn.check_constant_groups(e, 32)
    assert chk.ok and chk.n_groups == -(-out // 32) and chk.partial_last == (out % 32)
    own = gr.scale_exp(w, (-1,), rule)                                    # each channel's own exponent
    for g0 in range(0, out, 32):
        grp = w[g0:g0 + 32]
        want = gr.scale_exp(grp, None, rule)                              # rule(amax of the whole group)
        assert (e[g0:g0 + 32] == want).all()
        assert (e[g0:g0 + 32] >= own[g0:g0 + 32]).all()
    assert sc.granularity == ("per_panel(32)" if out % 32 == 0 else "per_panel(32,partial_tail)")


def test_panel_rows_parameter_and_label():
    w = _channel_scaled((40, 16), torch.Generator().manual_seed(0))
    sc = _g("granularity:w=panel,rows=8,rule=nc").weight_scale("l", w, "linear")
    assert tn.check_constant_groups(sc.exp, 8).ok and sc.granularity == "per_panel(8)"
    assert not tn.check_constant_groups(sc.exp, 32).ok or len(set(sc.exp.flatten().tolist())) == 1
    with pytest.raises(ValueError, match="rows"):
        _g("granularity:w=panel,rows=0")


def test_panel_cast_codes_are_the_codes_of_the_group_exponent():
    w = _channel_scaled((64, 32), torch.Generator().manual_seed(3))
    algo = _g("granularity:w=panel,rule=nc")
    q, s, blk = algo_hooks.weight_cast(algo, "l", w, "linear")
    assert blk == 0 and s.shape == (64, 1)
    assert torch.equal(q, rn.e4m3_vpack(w, s))
    assert algo.stats["weight"]["linear:per_panel(32)"] == 1


# -- (b) block exponents -------------------------------------------------------------------------------

def _token_exp_nc(x2d):
    return gr.scale_exp(x2d, (-1,), "nc")


@pytest.mark.parametrize("shape", [(2, 40, 16), (1, 1, 16), (3, 32, 16), (1, 70, 16), (80, 16)])
def test_block_exponent_groups_flattened_tokens_with_partial_last(shape):
    gen = torch.Generator().manual_seed(sum(shape))
    x = (torch.randn(shape, generator=gen) * torch.exp2(torch.randint(-5, 6, shape[:-1] + (1,), generator=gen).float())
         ).to(torch.bfloat16)
    sc = _g("granularity:a=block").act_scale("l", x, "linear")
    e = sc.exp
    assert e.shape == (*shape[:-1], 1) and sc.granularity == "per_token_block" and sc.dynamic
    flat = x.reshape(-1, shape[-1])
    m = flat.shape[0]
    chk = tn.check_constant_groups(e, 32)
    assert chk.ok and chk.partial_last == (m % 32)
    ef = e.reshape(-1)
    for g0 in range(0, m, 32):
        grp = flat[g0:g0 + 32]
        assert (ef[g0:g0 + 32] == gr.scale_exp(grp, None, "nc")).all()          # group amax, nc rule
        assert (ef[g0:g0 + 32] >= _token_exp_nc(grp)).all()                      # never below a member's own
    # zero padding does not change the amax: explicit padding to a full block gives the same exponent
    last = flat[(m // 32) * 32:] if m % 32 else flat[-32:]
    padded = torch.cat([last, last.new_zeros(32 - last.shape[0], shape[-1])])
    assert gr.scale_exp(padded, None, "nc") == ef[-1]


def test_block_group_may_straddle_batch_elements():
    """[B=2, T=40, K]: tokens 32..63 form one group although token 40 starts the second batch element."""
    x = torch.zeros(2, 40, 8, dtype=torch.bfloat16)
    x[0, 35, 0] = 1.0                                                     # group 1 (tokens 32..63), batch 0
    x[1, 3, 0] = 2.0 ** 6                                                 # flattened token 43: same group
    e = _g("granularity:a=block").act_scale("l", x, "linear").exp.reshape(-1)
    assert len(set(e[32:64].tolist())) == 1 and e[32] == gr.scale_exp(x.reshape(-1, 8)[32:64], None, "nc")
    x2 = x.clone()
    x2[1, 3, 0] = 0.0
    without = _g("granularity:a=block").act_scale("l", x2, "linear").exp.reshape(-1)
    assert e[32] > without[32] and torch.equal(e[:32], without[:32])         # batch 1's token raised the shared group


def test_block_decode_single_token_equals_per_token():
    x = torch.randn(1, 1, 64).to(torch.bfloat16)
    assert torch.equal(_g("granularity:a=block").act_scale("l", x, "linear").exp.reshape(-1),
                       _token_exp_nc(x.reshape(1, -1)).reshape(-1))


def test_attn_block_groups_rows_per_batch_head_matrix():
    gen = torch.Generator().manual_seed(9)
    q = (torch.randn(2, 3, 70, 8, generator=gen) * torch.exp2(torch.randint(-4, 5, (2, 3, 70, 1), generator=gen).float())
         ).to(torch.bfloat16)
    algo = _g("granularity:attn=block,rule=nc")
    sq = algo.act_scale("s", q, "attention.q", (-2, -1))
    assert sq.exp.shape == (2, 3, 70, 1) and tn.check_constant_groups(sq.exp, 32, "per_matrix").ok
    assert tn.check_constant_groups(sq.exp, 32, "per_matrix").partial_last == 70 % 32
    for b in range(2):
        for h in range(3):
            assert (sq.exp[b, h, :32, 0] == gr.scale_exp(q[b, h, :32], None, "nc")).all()
    # k and v stay one exponent per head; attn=block with rule=ref still gives them the never-clip rule
    sk = algo.act_scale("s", q, "attention.k", (-2, -1))
    assert sk.exp.shape == (2, 3, 1, 1) and sk.granularity == "per_head"
    assert _g("granularity:attn=head,rule=ref").act_scale("s", q, "attention.q", (-2, -1)) is None


def test_defer_returns_no_activation_scale():
    x = torch.randn(2, 4, 8).to(torch.bfloat16)
    a = _g("granularity:w=panel,a=defer,attn=defer,rule=nc")
    assert a.act_scale("l", x, "linear") is None and a.act_scale("s", x, "attention.p", (-2, -1)) is None
    assert gr.A_CHOICES[:3] == ("tensor", "token", "block32") and "defer" not in ("tensor", "token")


# -- verifier -----------------------------------------------------------------------------------------

def test_check_constant_groups_flags_a_varying_group():
    e = torch.zeros(70, 1, dtype=torch.int32)
    assert tn.check_constant_groups(e).ok
    e[40] = 3                                                             # inside group 1 (rows 32..63)
    chk = tn.check_constant_groups(e)
    assert not chk.ok and chk.n_violations == 1 and chk.violations[0]["group"] == 1
    assert chk.violations[0]["values"] == [0, 3] and "varies" in chk.reason
    assert tn.check_constant_groups(torch.tensor(5)).layout == "scalar"
    assert json.dumps(chk.to_dict())
    bad = torch.zeros(2, 70, 1, dtype=torch.int32)
    bad[1, 33] = 1
    assert not tn.check_constant_groups(bad, 32, "per_matrix").ok


def test_site_status_names_the_reason_for_non_tile_recipes():
    e = torch.arange(64, dtype=torch.int32).reshape(64, 1)
    assert not tn.weight_site_status("per_channel", e)["tile_native"]
    assert "ALONG K" in tn.weight_site_status("per_block(32)", e)["reasons"][0]
    assert tn.activation_rule_status({"mode": "dynamic", "reduce": "token", "label": "per_token"})["reasons"]
    assert "H3" in tn.activation_rule_status({"mode": "dynamic", "reduce": "block32_or_token"})["reasons"][0]
    assert tn.activation_rule_status({"mode": "static", "exp": -3})["tile_native"]


def test_pack_admission_is_informational_and_flags_outside_122_128():
    rep = tn.pack_admission_report({"a": [-6, -6], "b": [0], "p": [-8], "c": [1, 10], "edge": [-5, 1]}, {"w": (100, 130)})
    assert rep["sites_outside"] == 3 and set(rep["offending_sites"]) == {"a", "p", "c"}       # -5 .. 1 is 122 .. 128
    assert rep["activation_static_codes"]["b"] == {"min": 127, "max": 127}     # in range
    assert rep["activation_static_codes"]["p"]["min"] == 119                    # static attention p = -8
    assert rep["weight_codes"]["min"] == 100


def test_weight_code_report_fails_loudly_only_on_reserved_codes():
    codes = torch.randint(0, 0x7F, (64, 16), dtype=torch.int32).to(torch.uint8)
    rep = tn.weight_code_report(codes, 32, "w")
    assert rep["reserved_codes"] == 0 and rep["flushed_fraction_per_group"]["groups"] == 2
    codes[3, 3] = 0x7F
    with pytest.raises(ValueError, match="reserved"):
        tn.weight_code_report(codes, 32, "w")
    assert tn.weight_code_report(codes, 32, "w", fail_on_reserved=False)["reserved_codes"] == 1
    assert tn.padding_encodes_to_plus_zero(16, 32, -7)
    assert tn.token_split_report(70)["partial_last_group_rows"] == 6 and tn.token_split_report(64)["aligned_32"]


# -- (c) hardware equivalence --------------------------------------------------------------------------

def test_pop_scaled_unscaled_matmul_equals_dequantize_then_matmul():
    """The Atlas dataflow: matmul on UNSCALED E4M3 values accumulating in float32 (BF16 in hardware; exact
    float32 here), then one multiply by 2^(sx[block] + sw[panel]) per output, is bit-identical to decoding
    both operands with their exponents first and then multiplying (what the FX operators do)."""
    gen = torch.Generator().manual_seed(21)
    M, K, N = 70, 96, 70
    x = _group_scaled((M, K), gen, steps=(-6, 3, 0))
    w = _group_scaled((N, K), gen)
    algo = _g("granularity:w=panel,a=block,rule=nc")
    qx, sx, _ = algo_hooks.act_cast(algo, "l", x, "linear")
    qw, sw, _ = algo_hooks.weight_cast(algo, "l", w, "linear")
    assert len(set(sx.reshape(-1).tolist())) > 1 and len(set(sw.reshape(-1).tolist())) > 1
    y_pop = (rn.e4m3_unpack(qx) @ rn.e4m3_unpack(qw).T) * torch.exp2((sx + sw.reshape(1, -1)).float())
    y_dq = rn.dequantize(qx, sx) @ rn.dequantize(qw, sw).T
    assert torch.equal(y_pop, y_dq)


def test_scale_varying_along_k_is_not_pop_realizable():
    """block32 (a scale per K block): no single pop scale reproduces dequantize-then-matmul."""
    gen = torch.Generator().manual_seed(4)
    M, K, N = 8, 64, 8
    x = torch.randn(M, K, generator=gen)
    x[:, 32:] *= 2.0 ** 12                                                # K block 1 is 4096x larger
    x, w = x.to(torch.bfloat16), torch.randn(N, K, generator=gen).to(torch.bfloat16)
    algo = _g("granularity:w=tensor,a=block32,rule=nc")
    qx, sx, blk = algo_hooks.act_cast(algo, "l", x, "linear")
    assert blk == 32 and sx.shape == (M, 2, 1) and len(set(sx.reshape(-1).tolist())) > 1
    qw, sw = rn.quantize(w)
    y_dq = algo_hooks.dequant(qx, sx, blk) @ rn.dequantize(qw, sw).T
    xn = rn.e4m3_unpack(qx)
    wn = rn.e4m3_unpack(qw)
    for s in sorted(set(sx.reshape(-1).tolist())):                        # any one constant pop scale
        y_pop = (xn @ wn.T) * torch.exp2(torch.tensor(float(s) + float(sw)))
        assert not torch.equal(y_pop, y_dq)
        assert float((y_pop - y_dq).abs().max()) > 1e-3 * float(y_dq.abs().max())


# -- (d) GPTQ + AWQ on the panel grid -------------------------------------------------------------------

def _spread_rows(model):
    """Make the 32-row groups of every Linear differ by binades, so a per-tensor grid would be visibly wrong."""
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.Linear) and m.out_features >= 64:
                m.weight[32:64] *= 2.0 ** 5


def test_gptq_awq_pin_and_search_use_the_panel_grid(monkeypatch):
    need_refs()
    calls = {"awq": [], "gptq": []}
    real_awq, real_gptq = ga.awq_search, ga.gptq

    def spy_awq(Ws, H, sx, grid=20, exp_fn=None):
        s, alpha, loss = real_awq(Ws, H, sx, grid, exp_fn)
        assert exp_fn is not None, "AWQ search fell back to the per-tensor grid"
        for W in Ws:
            Wsc = (W.float() * s).to(W.dtype).float()                     # what the search quantizes
            e = exp_fn(Wsc)
            calls["awq"].append((e.shape == (W.shape[0], 1), tn.check_constant_groups(e).ok,
                                 torch.equal(e, gr.panel_scale_exp(Wsc, 32, "nc")),
                                 W.shape[0], len(set(e.reshape(-1).tolist()))))
        return s, alpha, loss

    def spy_gptq(W, H, s, *a, **k):
        calls["gptq"].append((W.detach().clone(), s.detach().clone()))
        return real_gptq(W, H, s, *a, **k)

    monkeypatch.setattr(ga, "awq_search", spy_awq)
    monkeypatch.setattr(ga, "gptq", spy_gptq)
    m = tiny_llama(layers=2)
    _spread_rows(m)
    algo = _g("atlas_v1_awq_tile")
    algos.quantize(m, "ref_conv", ["lm_head"], dict(FAST), algo=algo, calib=_calib()[1])
    assert calls["awq"], "no AWQ group was searched"
    assert all(shape and const and equal for shape, const, equal, *_ in calls["awq"])
    assert any(out >= 64 and distinct > 1 for *_, out, distinct in calls["awq"])     # not a per-tensor grid
    assert len(calls["gptq"]) == 14
    pinned = {}
    for W, s in calls["gptq"]:
        # GPTQ received the panel exponents of the (AWQ-scaled) weight it was handed, [out, 1] repeated per group
        want = gr.panel_scale_exp(W, 32, "nc")
        assert torch.equal(s.reshape(-1, 1), want) and tn.check_constant_groups(s).ok
        pinned[W.shape] = pinned.get(W.shape, 0) + 1
    qlin = [(n, mod) for n, mod in m.named_modules() if type(mod).__name__ == "RefQuantLinear"]
    assert len(qlin) == 14
    for n, mod in qlin:
        e = mod.weight_scale_exp
        assert e.shape == (mod.out_features, 1) and tn.check_constant_groups(e).ok          # constancy after GPTQ
        assert torch.equal(e, algo.members[0].s0[n].reshape(-1, 1).to(torch.int32))         # pinned, not re-derived
        # the final codes sit on the panel grid: decoding at the exported exponent reproduces the GPTQ weight
        dq = rn.dequantize(mod.weight_codes, e)
        assert torch.equal(ga.qdq(dq, e), dq)
        assert tn.weight_code_report(mod.weight_codes, 32)["reserved_codes"] == 0
    assert any(mod.out_features >= 64 and len(set(mod.weight_scale_exp.reshape(-1).tolist())) > 1 for _, mod in qlin)


# -- (e) existing recipes are unchanged -----------------------------------------------------------------

def test_existing_recipes_and_defaults_are_untouched():
    pinned = {
        "atlas_v1_awq": "gptq_awq:method=awq_gptq,actorder=true+granularity:w=channel,a=token,rule=nc",
        "atlas_v1_static": "smoothquant:alpha=0.5,po2=true,gain=true,static=true,partial=true"
                           "+gptq_awq:method=gptq,actorder=true",
    }
    for name, spec in pinned.items():
        assert algos.RECIPES[name]["spec"] == spec
    assert algos.expand("atlas_v1") == pinned["atlas_v1_awq"]
    g = gr.Granularity()
    assert (g.w, g.a, g.attn, g.rule) == ("tensor", "tensor", "head", "ref")
    w = _channel_scaled((70, 16), torch.Generator().manual_seed(2))
    sc = _g("granularity:w=channel,a=token,rule=nc").weight_scale("l", w, "linear")
    assert torch.equal(sc.exp, gr.scale_exp(w, (-1,), "nc")) and sc.granularity == "per_channel"
    x = torch.randn(2, 5, 16).to(torch.bfloat16)
    st = _g("granularity:w=channel,a=token,rule=nc").act_scale("l", x, "linear")
    assert st.granularity == "per_token" and torch.equal(st.exp, gr.scale_exp(x, (-1,), "nc"))
    # the new weight / activation options are off unless named
    assert _g("granularity:w=channel,a=token").tile_status()["all"] is False


def test_tile_recipes_expand_as_specified():
    st = {n: [(m.name, m.params) for m in algos.make(n).members] for n in TILE_RECIPES}
    tile = ("granularity", dict(w="panel", a="block", rows=32, rule="nc", attn="head"))
    gptq = ("gptq_awq", dict(method="awq_gptq", actorder=True))
    assert st["atlas_v1_awq_tile"] == [gptq, tile]
    assert st["atlas_v1_tile"] == [gptq, tile, ("keep_bf16", dict(rule="input_conv"))]
    assert st["atlas_v1_awq_tile_static"] == [
        gptq, ("granularity", dict(w="panel", a="defer", attn="defer", rows=32, rule="nc")), ("static_calib", {})]
    for n in TILE_RECIPES:
        assert algos.RECIPES[n]["chip"] == {"chip_vadd": "rn", "chip_vadd_sites": "residual"}
        assert algos.make(n).recipe["name"] == n
        assert algos.make(n).members[1].tile_status()["all"] is True


def test_atlas_v1_tile_keeps_the_input_conv_bf16_and_awq_tile_does_not():
    from test_algo_atlas_v1 import DIT_PATCH, TinyDiT, dit_calib

    need_refs()
    for name, kept in (("atlas_v1_tile", [DIT_PATCH]), ("atlas_v1_awq_tile", []), ("atlas_v1_awq_tile_static", [])):
        m = TinyDiT()
        rep = algos.quantize(m, "ref_conv", algo=algos.make(name), calib=dit_calib())
        assert rep["algo"]["bf16_opt_out_exact"] == kept
        assert (type(m.get_submodule(DIT_PATCH)).__name__ == "RefQuantConv2d") == (not kept)
        assert type(m.get_submodule(DIT_PATCH.replace("patch_embedding", "patch_embedding_extra"))).__name__ \
            == "RefQuantConv2d"


# -- export / replay -----------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", EXPORT_SPECS)
def test_export_records_tile_native_and_replays_bit_identically(tmp_path, spec):
    need_refs()
    m0, d, man, opts = _quantize_export(tmp_path, spec, layers=1)
    policy = json.loads((d / "policy.json").read_text())
    assert policy["tile_native"]["all_tile_native"] is True and policy["tile_native"]["not_tile_native"] == {}
    assert policy["tile_native"]["padding_encodes_to_plus_zero"] is True
    lin = [s for s in policy["sites"].values() if s["kind"] == "linear"]
    assert len(lin) == 7 and all(s["weight"]["granularity"].startswith("per_panel(32") for s in lin)
    assert all(s["weight"]["panel_rows"] == 32 and s["tile_native"]["tile_native"] for s in lin)
    assert all(s["weight"]["code_report"]["reserved_codes"] == 0 for s in lin)
    assert all(s["weight"]["e8m0_code_range"]["min"] <= s["weight"]["e8m0_code_range"]["max"] for s in lin)
    adm = policy["compiler_pack_scale_admission"]
    assert adm["admitted_range"] == [122, 128] and "note" in adm
    if spec.endswith("static"):
        assert all(s["activation"]["mode"] == "static" and "e8m0_code" in s["activation"] for s in lin)
        attn = next(s for s in policy["sites"].values() if s["kind"] == "attention")
        assert attn["operands"]["p"]["exp"] == -8 and attn["operands"]["p"]["e8m0_code"] == 119
        assert any(k.endswith(":p") for k in adm["offending_sites"])           # code 119 is outside 122..128
        assert attn["operands"]["q"]["source"] == "reference"
    else:
        assert all(s["activation"]["reduce"] == "block" and s["activation"]["rows"] == 32
                   and s["activation"]["tile_label"] == "per_block(32 rows)" for s in lin)
    if spec == ATTN_BLOCK:
        attn = next(s for s in policy["sites"].values() if s["kind"] == "attention")
        for op in "qp":
            assert attn["operands"][op]["reduce"] == "head_block" and attn["operands"][op]["rows"] == 32
            assert attn["operands"][op]["tile_label"] == "per_block(32 rows)"
        for op in "kv":
            assert attn["operands"][op]["reduce"] == "reference_dims" and attn["operands"][op]["rule"] == "nc"
    ids = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
    m1 = _reload(d, "ref_conv", opts, layers=1)
    with torch.no_grad():
        assert torch.equal(m0(ids, use_cache=False).logits, m1(ids, use_cache=False).logits)


def test_replay_fails_loudly_when_a_panel_labeled_export_varies_within_a_group(tmp_path):
    need_refs()
    from safetensors.torch import load_file, save_file

    _, d, man, opts = _quantize_export(tmp_path, "atlas_v1_awq_tile", layers=1)
    tensors = load_file(str(d / "model.safetensors"))
    key = next(k for k in tensors if k.endswith("q_proj.weight_scale_exp"))
    tensors[key] = tensors[key].clone()
    tensors[key][5] += 1                                                 # channel 5 leaves its group's exponent
    save_file(tensors, str(d / "model.safetensors"))
    with pytest.raises(tn.TileContractError, match="per_panel"):
        load_quantized.load_quantized(None, d, "cpu", build=lambda: tiny_llama(1), mode="ref_conv",
                                      options=opts, verify=False)


def test_token_recipe_export_is_not_tile_native_with_reasons(tmp_path):
    need_refs()
    _, d, *_ = _quantize_export(tmp_path, "atlas_v1_awq", layers=1)
    policy = json.loads((d / "policy.json").read_text())
    nt = policy["tile_native"]
    assert nt["all_tile_native"] is False and nt["tile_native_sites"] < nt["sites_checked"]
    some = next(iter(nt["not_tile_native"].values()))
    assert any("per_channel" in r for r in some) and any("per_token" in r for r in some)


def test_block32_export_is_labeled_not_hardware_realizable(tmp_path):
    need_refs()
    _, d, *_ = _quantize_export(tmp_path, "granularity:w=block32,a=block32,rule=nc", layers=1)
    policy = json.loads((d / "policy.json").read_text())
    reasons = " ".join(r for rs in policy["tile_native"]["not_tile_native"].values() for r in rs)
    assert "ALONG K" in reasons and "H3" in reasons


# -- (f) native FX graph ----------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", EXPORT_SPECS + (ROWS16,))
def test_tile_recipe_native_fx_graph(tmp_path, spec):
    need_refs()
    eager, directory, _, options = _quantize_export(tmp_path, spec, layers=1)
    loaded = _reload(directory, "ref_conv", options, layers=1)
    policy = json.loads((directory / "policy.json").read_text())
    wrapped = fx_export.CausalLogits(loaded).eval()
    with torch.no_grad():
        for (ids,) in INPUTS:
            assert torch.equal(eager(ids, use_cache=False).logits, wrapped(ids))
    report = fx_export.capture_fx(wrapped, INPUTS, tmp_path / "fx", policy=policy, fresh_process=True,
                                  native_fp8="uint8")
    assert report["status"] == "passed" and report["weight_sites_verified"] == 7
    fresh = report["fresh_process_reload"]
    assert fresh["atlas_qlinear_nodes"] == 7 and fresh["atlas_qk_matmul_nodes"] == 1 \
        and fresh["atlas_pv_matmul_nodes"] == 1
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert all(row["native"] for row in inspection["weight_site_bindings"])
    assert inspection["still_emulated"]["emulated_modules"] == {}
    tile = inspection["tile_native"]
    assert tile["all_tile_native"] is True and tile["not_tile_native"] == {}
    assert len(tile["sites"]) == 8 and all(r["tile_native"] and r["native_fp8"] for r in tile["sites"].values())
    stats = tile["fixture_activation_stats"]
    assert all(v["reserved_codes"] == 0 for v in stats.values())
    adm = inspection["compiler_pack_scale_admission"]
    assert adm["dynamic_activation_scales"]["source"].startswith("capture-case")
    if spec.endswith("static"):
        assert any(k.endswith(":p") for k in adm["offending_sites"])
    else:
        # 8 tokens: one partial block of 8 rows, zero padding encodes to +0
        lin = [v for k, v in stats.items() if ":" not in k]
        assert lin and all(v["reduce"] == "block" and v["groups_constant"] for v in lin)
        assert all(not v["token_split_all_32_aligned"] and v["token_splits"][0]["partial_last_group_rows"] == 8
                   and v["token_splits"][0]["padding_encodes_to_plus_zero"] for v in lin)
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    blocks = [n for n in program.graph.nodes if n.target == torch.ops.atlas.quantize_act.default
              and n.args[2] in ("block", "head_block")]
    # export drops trailing arguments equal to their defaults: rows == 32 is implied, rows != 32 is carried
    want = 16 if spec == ROWS16 else 32
    assert all((n.args[7] if len(n.args) > 7 else 32) == want for n in blocks)
    assert all(len(n.args) == 8 for n in blocks) == (want != 32) or not blocks
    if spec in (ATTN_BLOCK, ROWS16):
        assert sorted(n.args[6].rpartition(":")[2] for n in blocks if n.args[2] == "head_block") == ["p", "q"]
        for label in [k for k in stats if k.endswith((":q", ":p"))]:
            assert stats[label]["reduce"] == "head_block" and stats[label]["groups_constant"]
            assert stats[label]["token_splits"][0]["partial_last_group_rows"] in (1, 8)
        assert len(blocks) == 7 + 2
    saved = program.module()
    with torch.no_grad():
        for (ids,) in INPUTS:
            assert torch.equal(saved(ids), eager(ids, use_cache=False).logits)


def test_token_recipe_native_graph_reports_not_tile_native(tmp_path):
    need_refs()
    eager, directory, _, options = _quantize_export(tmp_path, "atlas_v1_awq", layers=1)
    loaded = _reload(directory, "ref_conv", options, layers=1)
    policy = json.loads((directory / "policy.json").read_text())
    fx_export.capture_fx(fx_export.CausalLogits(loaded).eval(), INPUTS, tmp_path / "fx", policy=policy,
                         fresh_process=False, native_fp8="uint8")
    tile = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())["tile_native"]
    assert tile["all_tile_native"] is False
    assert len(tile["not_tile_native"]) == 7 and tile["tile_native_sites"] == 1       # 7 Linears; attention is per head
    assert all("per_channel" in " ".join(r) and "per_token" in " ".join(r) for r in tile["not_tile_native"].values())
    assert all(v["native_fp8"] for v in tile["sites"].values())


def test_quantize_act_block_matches_algorithm_and_fake_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    x = (torch.randn(2, 40, 16) * torch.exp2(torch.randint(-5, 6, (2, 40, 1)).float())).to(torch.bfloat16)
    codes, s = torch.ops.atlas.quantize_act(x, "dynamic", "block", "nc", None, "uint8", "", 32)
    sc = _g("granularity:a=block").act_scale("l", x, "linear")
    assert torch.equal(s, sc.exp) and torch.equal(codes, rn.e4m3_vpack(x, s)) and s.shape == (2, 40, 1)
    c2, s2 = torch.ops.atlas.quantize_act(x, "dynamic", "block", "nc", None, "uint8")      # rows default 32
    assert torch.equal(s2, s)
    c3, s3 = torch.ops.atlas.quantize_act(x, "dynamic", "block", "nc", None, "uint8", "", 8)
    assert tn.check_constant_groups(s3, 8).ok and not torch.equal(s3, s)
    q = x.reshape(1, 2, 40, 16)
    _, sh = torch.ops.atlas.quantize_act(q, "dynamic", "head_block", "nc", None, "uint8", "", 32)
    assert sh.shape == (1, 2, 40, 1) and tn.check_constant_groups(sh, 32, "per_matrix").ok
    with FakeTensorMode():
        f = torch.empty(2, 40, 16, dtype=torch.bfloat16)
        _, sf = torch.ops.atlas.quantize_act(f, "dynamic", "block", "nc", None, "uint8", "", 32)
        assert list(sf.shape) == [2, 40, 1]


def test_native_rule_carries_rows_and_keeps_the_four_tuple():
    site = {"activation": {"mode": "dynamic", "reduce": "block", "rule": "nc", "rows": 16, "label": "per_token_block"}}
    r = fx_export.native_rule(site, 64)
    mode, reduce, rule, exp = r
    assert (mode, reduce, rule, exp, r.rows) == ("dynamic", "block", "nc", None, 16)
    assert fx_export.native_rule({"activation": {"mode": "dynamic", "reduce": "token", "rule": "nc"}}, 64).rows == 32
    assert fx_export.native_operand_rule({"mode": "dynamic", "reduce": "head_block", "rule": "nc", "rows": 32}) \
        == ("dynamic", "head_block", "nc", None)


# -- old saved bundles still import and replay -------------------------------------------------------------

def test_bundle_saved_with_the_old_quantize_act_schema_still_verifies(tmp_path):
    """A native bundle written by the code before `rows` existed (a frozen copy of the old operator module
    registers the old schema in a separate process, because a custom-op schema registers once per process) is
    loaded and replayed with the new schema."""
    old_dir = Path(__file__).parent / "old_schema"
    src = Path(__file__).parent.parent / "src"
    script = textwrap.dedent("""
        import sys
        sys.path[:0] = [sys.argv[1], sys.argv[2]]
        import exporter, reference.ref_numerics
        sys.modules["ref_numerics"] = reference.ref_numerics      # the frozen copy predates the package layout
        import atlas_fx_ops                                        # the frozen copy (old schema)
        sys.modules["exporter.atlas_fx_ops"] = exporter.atlas_fx_ops = atlas_fx_ops  # what fx_export imports
        import torch, algos
        import exporter.fx_export as fx_export
        import exporter.quant_export as quant_export
        schema = str(torch.ops.atlas.quantize_act.default._schema)
        assert "rows" not in schema, schema
        FAST = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}
        torch.manual_seed(13)
        module = torch.nn.Sequential(torch.nn.Linear(8, 4)).to(torch.bfloat16).eval()
        algo = algos.make("granularity:w=channel,a=token,rule=nc")
        report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
        policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
        first = torch.randn(2, 3, 8, dtype=torch.bfloat16)
        cases = [(first,), (first * 2,)]
        fx_export.capture_fx(module, cases, sys.argv[3], policy=policy, fresh_process=False, native_fp8="uint8")
        print("SAVED")
    """)
    out = tmp_path / "old_bundle"
    run = subprocess.run([sys.executable, "-c", script, str(old_dir), str(src), str(out)],
                         capture_output=True, text=True)
    assert "SAVED" in run.stdout, run.stdout + run.stderr
    program = torch.export.load(out / "model.pt2")
    acts = [n for n in program.graph.nodes if n.target == torch.ops.atlas.quantize_act.default]
    assert acts and all(len(n.args) <= 7 for n in acts)                  # saved without `rows`
    assert "rows" in str(torch.ops.atlas.quantize_act.default._schema)   # this process has the new schema
    fx_export.verify_saved(out)                                          # imports atlas_fx_ops, replays fixtures


# -- costs ---------------------------------------------------------------------------------------------

def _chip_ops(spec, tokens=40):
    model = tiny_llama()
    rep = algos.quantize(model, "chip", ["lm_head"], {"chip_mxu": "framework"}, algo=algos.make(spec), calib=_calib()[1]
                         if "gptq" in spec else None)
    with torch.no_grad():
        model(torch.randint(0, 128, (1, tokens), generator=torch.Generator().manual_seed(0)))
    return {o["op"]: o for o in rep["coverage"]["ops"]}


def test_panel_block_cost_has_no_per_element_rescale_and_old_accounting_is_unchanged():
    need_refs()
    old = _chip_ops("granularity:w=channel,a=token,rule=nc")
    assert old["granularity_rescale"]["instructions"]["vmul.bf16"] > 0           # per-element accounting kept
    assert "tile_output_rescale" not in old
    new = _chip_ops("granularity:w=panel,a=block,rule=nc")
    assert "granularity_rescale" not in new                                       # no pre-vpack per-element vmul
    # BF16 consumer (the conservative default): one vmul.bf16 per 32 x 32 output tile
    tiles = new["tile_output_rescale"]["instructions"]["vmul.bf16"]
    assert tiles > 0 and "tile constant" in new["tile_output_rescale"]["reason"]
    assert tiles < old["granularity_rescale"]["instructions"]["vmul.bf16"] / 100
    assert "dyn_scale_chain" in new                                               # dynamic block chain still counted
    # one Linear 40 tokens x 64 out: 2 x 2 tiles
    import chip.chip_profile as cp

    prior = cp.TILE_OUTPUT_CONSUMER
    try:
        cp.TILE_OUTPUT_CONSUMER = "fp8"                                           # folded into vmatpop.fp8
        assert "tile_output_rescale" not in _chip_ops("granularity:w=panel,a=block,rule=nc")
    finally:
        cp.TILE_OUTPUT_CONSUMER = prior
    with pytest.raises(ValueError, match="consumer"):
        cp.count_tile_output_rescale(32, 32, "int8")
