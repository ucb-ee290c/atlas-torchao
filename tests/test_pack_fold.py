"""Pack fold (src/pack_fold.py, docs/algos/pack-fold.md): exactness on synthetic layers and a synthetic FX graph, k agreement
across shared-input groups, the BF16 range guard, negative cases (non-free classes, second consumers), the off-by-default
contract, and an end-to-end tiny Llama (export, fold, FX capture, verify_bundle, bit-identical logits)."""
import copy
import importlib.util
import json
import types
from pathlib import Path

import pytest
import torch

import atlas_fx_ops  # noqa: F401
import fx_export
import pack_fold as pf
import pack_plan as pp
import quant_export
import tile_native as tn
from refutil import need_refs
from test_export import _quantize_export, _reload
from test_ladder import FAST  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "tile_flow"
STATIC = "atlas_v1_awq_tile_static"
INPUTS = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
          (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- 1. k of a producer group ---------------------------------------------------------------------------------

@pytest.mark.parametrize("codes,k", [([118], 4), ([119], 3), ([121], 1), ([127], 0), ([122, 128], 0), ([118, 117], 5),
                                     ([118, 121], 4), ([121, 123], 1), ([126, 130], -2), ([130], -2), ([116, 124], None),
                                     ([115, 125], None)])
def test_group_k_is_the_smallest_shift_that_puts_every_member_in_range(codes, k):
    assert pf.group_k(codes) == k
    if k is not None:
        assert all(122 <= c + k <= 128 for c in codes)
        assert all(not all(122 <= c + j <= 128 for c in codes) for j in range(-abs(k) + 1, abs(k)) if j != k)


@pytest.mark.parametrize("code", list(range(110, 135)))
def test_single_site_k_equals_the_pack_plan_nearest_choice(code):
    assert pf.group_k([code]) == pp.fold_for_code(code)["k"]


# -- 2. a synthetic FX graph with every producer class --------------------------------------------------------

def _n(name, tgt, args, users=1):
    return f"    %{name} : [num_users={users}] = call_function[target={tgt}](args = ({args}), kwargs = {{}})\n"


def _pack_chain(i, src, exp, qlinear_weight, tag="", users_src=1):
    s = f", {tag}" if tag else ""
    return (_n(f"qa{i}", "torch.ops.atlas.quantize_act.default", f"%{src}, static, tensor, ref, {exp}, uint8{s}", 2)
            + _n(f"ga{i}", "operator.getitem", f"%qa{i}, 0") + _n(f"gb{i}", "operator.getitem", f"%qa{i}, 1")
            + (_n(f"ql{i}", "torch.ops.atlas.qlinear.default",
                  f"%ga{i}, %gb{i}, %b_model_model_{qlinear_weight}_weight_codes, %b_model_model_{qlinear_weight}_weight_scale_exp, None, bfloat16")
               if qlinear_weight else ""))


def _graph(second_consumer=False, gamma_param="p_model_model_norm_weight"):
    g = _n("rs", "torch.ops.aten.rsqrt.default", "%var") + _n("m0", "torch.ops.aten.mul.Tensor", "%x, %rs")
    g += _n("m1", "torch.ops.aten.mul.Tensor", f"%{gamma_param}, %m0", 3 if second_consumer else 2)
    g += _n("n1", "torch.ops.aten.to.dtype", "%m1, torch.bfloat16")
    g += _pack_chain(0, "n1", -9, "a_q_proj") + _pack_chain(1, "n1", -10, "a_k_proj")     # shared input, codes 118 / 117
    g += _pack_chain(2, "x", -5, "a_gate_proj") + _pack_chain(3, "x", -5, "a_up_proj")     # raw input, in range
    g += _n("silu", "torch.ops.aten.silu.default", "%ql2") + _n("m2", "torch.ops.aten.mul.Tensor", "%silu, %ql3")
    g += _n("n2", "torch.ops.aten.to.dtype", "%m2, torch.bfloat16")
    g += _pack_chain(4, "n2", -7, "a_down_proj")                                           # gated product, code 120
    g += _n("sm", "torch.ops.aten.softmax.int", "%scores, -1")
    g += _pack_chain(5, "sm", -8, None, tag="model.a.attn:p")                              # attention p, code 119
    g += _n("pv", "torch.ops.atlas.pv_matmul.default", "%ga5, %gb5, %vt, %vs")
    g += _pack_chain(6, "pv", -6, "a_o_proj")                                              # P.V output, code 121
    if second_consumer:
        g += _n("leak", "torch.ops.aten.add.Tensor", "%m1, %x")
    return g


_SITES = ("q_proj", "k_proj", "gate_proj", "up_proj", "down_proj", "o_proj")
_SIG = ("p_model_model_norm_weight: PARAMETER target='model.model.norm.weight'\n"
        + "".join(f"b_model_model_a_{n}_weight_codes: BUFFER target='model.model.a.{n}.weight_codes' persistent=False\n" for n in _SITES))
_EXPS = {"q_proj": -9, "k_proj": -10, "gate_proj": -5, "up_proj": -5, "down_proj": -7, "o_proj": -6}


def _policy_and_tensors(gamma=None):
    sites, t = {}, {}
    for n, e in _EXPS.items():
        name = f"a.{n}"
        t[f"{name}.weight_codes"] = torch.randint(0, 120, (64, 32), dtype=torch.uint8)
        t[f"{name}.weight_scale_exp"] = torch.full((64, 1), -13, dtype=torch.int32)
        t[f"{name}.input_scale_exp"] = torch.tensor(e, dtype=torch.int32)
        sites[name] = {"kind": "linear", "status": "quantized", "bias": None,
                       "weight": {"codes": f"{name}.weight_codes", "scale_exp": f"{name}.weight_scale_exp",
                                  "e8m0_code_range": {"min": 114, "max": 114}},
                       "activation": {"mode": "static", "exp": e, "label": "static", "tensor": f"{name}.input_scale_exp",
                                      "e8m0_code": 127 + e, "description": f"static exponent {e}"}}
    sites["model.a.attn"] = {"kind": "attention", "status": "quantized",
                             "operands": {"p": {"mode": "static", "exp": -8, "label": "static", "e8m0_code": 119},
                                          **{o: {"mode": "dynamic", "source": "reference"} for o in "qkv"}}}
    t["model.norm.weight"] = gamma if gamma is not None else (torch.rand(32) + 0.5).to(torch.bfloat16)
    return {"sites": sites}, t


def _fold(policy, tensors, **kw):
    info = pf.graph_info(_graph(**kw), _SIG, policy, set(tensors))
    groups = pf.plan_folds(policy, info)
    new_t, new_p, rep = pf.apply_folds(tensors, policy, groups)
    before = pp.static_sites(policy)
    pf.annotate(new_p, groups, info, before)
    return info, groups, new_t, new_p, rep


def test_graph_info_finds_producers_and_groups():
    policy, t = _policy_and_tensors()
    info = pf.graph_info(_graph(), _SIG, policy, set(t))
    cls = {k: r["class"] for k, r in info.items()}
    assert cls == {"a.q_proj": "norm_affine", "a.k_proj": "norm_affine", "a.gate_proj": "raw_input", "a.up_proj": "raw_input",
                   "a.down_proj": "gated_product", "a.o_proj": "attn_out_pv", "model.a.attn:p": "attn_p"}
    assert info["a.q_proj"]["producer"] == {"kind": "norm_gamma", "tensor": "model.norm.weight",
                                            "param_target": "model.model.norm.weight"}
    assert info["a.q_proj"]["src"] == info["a.k_proj"]["src"] and info["a.q_proj"]["consumer_site"] == "a.q_proj"
    assert info["a.down_proj"]["producer"]["site"] == "a.up_proj"           # the non-activated operand of silu(.) * .
    assert info["a.o_proj"]["producer"] == {"kind": "pv_shift", "site": "model.a.attn", "node": "pv"}


def test_synthetic_fold_moves_exactly_the_free_classes():
    policy, t = _policy_and_tensors()
    info, groups, new_t, new_p, rep = _fold(policy, t)
    new_exp = {k: v["exp"] for k, v in pp.static_sites(new_p).items()}
    # shared norm group: codes 118 / 117 -> one k = 5; gated product: 120 -> k = 2; P.V output: 121 -> k = 1
    assert new_exp["a.q_proj"] == -4 and new_exp["a.k_proj"] == -5
    assert new_exp["a.down_proj"] == -5 and new_exp["a.o_proj"] == -5
    assert new_exp["a.gate_proj"] == new_exp["a.up_proj"] == -5                      # already in range: untouched
    assert new_exp["model.a.attn:p"] == -8                                            # not free: untouched
    assert all(r["in_range"] for k, r in pp.recorded_plan(new_p).items() if k != "model.a.attn:p")
    # the producers
    assert torch.equal(new_t["model.norm.weight"].float(), t["model.norm.weight"].float() * 32)
    assert new_t["model.norm.weight"].dtype == torch.bfloat16
    assert torch.equal(new_t["a.up_proj.weight_scale_exp"], t["a.up_proj.weight_scale_exp"] + 2)
    assert new_p["sites"]["model.a.attn"]["pv_pop_shift"] == 1
    # the consumers keep sx + sw (pop exponent), codes are never touched
    for n in _SITES:
        assert torch.equal(new_t[f"a.{n}.weight_codes"], t[f"a.{n}.weight_codes"])
    assert pf.pop_exponent_unchanged(policy, new_p, t, new_t) is False              # up_proj also carries the producer shift (+2)
    for n in ("q_proj", "k_proj", "down_proj", "o_proj"):
        k = new_exp[f"a.{n}"] - _EXPS[n]
        assert torch.equal(new_t[f"a.{n}.weight_scale_exp"], t[f"a.{n}.weight_scale_exp"] - k)
        assert new_t[f"a.{n}.input_scale_exp"].item() == new_exp[f"a.{n}"]
        assert new_p["sites"][f"a.{n}"]["weight"]["e8m0_code_range"] == {"min": 114 - k, "max": 114 - k}
        assert new_p["sites"][f"a.{n}"]["activation"]["e8m0_code"] == 127 + new_exp[f"a.{n}"]
    # records and summary
    f = new_p["sites"]["a.q_proj"]["pack_fold"]
    assert f["applied"] is True and f["k"] == 5 and f["producer"] == "norm_affine" and f["exp_before"] == -9
    assert new_p["sites"]["a.gate_proj"]["pack_fold"] == {**new_p["sites"]["a.gate_proj"]["pack_fold"], "applied": False,
                                                          "reason": "in_range", "k": 0}
    s = new_p["pack_fold_summary"]
    assert (s["outside_before"], s["outside_after"], s["sites_folded"]) == (5, 1, 4)
    assert s["outside_after_by_class"] == {"attn_p": 1} and s["outside_after_by_cost"] == {"vmul_per_row_block": 1}
    assert s["outside_after_sites"]["model.a.attn:p"]["plan"]["k"] == 3
    assert new_p["compiler_pack_scale_admission"]["sites_outside"] == 1
    assert list(new_p["compiler_pack_scale_admission"]["offending_sites"]) == ["model.a.attn:p"]


def test_attention_p_and_other_non_free_classes_keep_their_exponent_and_plan():
    policy, t = _policy_and_tensors()
    _, groups, new_t, new_p, _ = _fold(policy, t)
    rec = new_p["sites"]["model.a.attn"]["pack_folds"]["p"]
    assert rec["applied"] is False and rec["producer"] == "attn_p" and rec["reason"] == "class_not_free:attn_p"
    assert rec["k"] == 3 and rec["exp_after"] == rec["exp_before"] == -8
    plan = new_p["sites"]["model.a.attn"]["pack_plans"]["p"]
    assert plan["status"] == "fold_planned" and plan["cost"] == "vmul_per_row_block"
    # a class that is not free (here: a bare input) is never given a fold
    g = next(g for g in groups.values() if g["class"] == "raw_input")
    assert g["applied"] is False
    for cls in ("bare_activation", "x_sigmoid_product", "attn_out_sdpa", "modulation_add", "norm_modulated", "raw_input"):
        assert cls not in pf.FREE_CLASSES


def test_second_consumer_of_the_norm_output_blocks_the_fold_and_changes_nothing():
    policy, t = _policy_and_tensors()
    info, groups, new_t, new_p, _ = _fold(policy, t, second_consumer=True)
    g = groups[info["a.q_proj"]["src"]]
    assert g["applied"] is False and g["reason"].startswith("other_consumers:")
    assert torch.equal(new_t["model.norm.weight"].float(), t["model.norm.weight"].float())
    assert new_p["sites"]["a.q_proj"]["activation"]["exp"] == -9 and new_p["sites"]["a.k_proj"]["activation"]["exp"] == -10
    assert torch.equal(new_t["a.q_proj.weight_scale_exp"], t["a.q_proj.weight_scale_exp"])
    assert new_p["pack_fold_summary"]["outside_after_by_class"]["norm_affine"] == 2


def test_shared_input_group_that_does_not_fit_one_window_is_left_alone():
    policy, t = _policy_and_tensors()
    policy["sites"]["a.k_proj"]["activation"]["exp"] = -14                      # codes 118 / 113: spread 5 fits, 113 vs 120 does not
    policy["sites"]["a.q_proj"]["activation"]["exp"] = -5 + 4                   # code 126
    info, groups, new_t, new_p, _ = _fold(policy, t)
    g = groups[info["a.q_proj"]["src"]]
    assert g["reason"].startswith("group_spread") and g["applied"] is False
    assert torch.equal(new_t["model.norm.weight"].float(), t["model.norm.weight"].float())


def test_a_member_already_in_range_moves_with_its_group():
    policy, t = _policy_and_tensors()
    policy["sites"]["a.k_proj"]["activation"]["exp"] = -5                       # code 122, in range; q_proj code 118
    info, groups, new_t, new_p, _ = _fold(policy, t)
    g = groups[info["a.q_proj"]["src"]]
    assert g["applied"] and g["k"] == 4
    assert new_p["sites"]["a.q_proj"]["activation"]["exp"] == -5 and new_p["sites"]["a.k_proj"]["activation"]["exp"] == -1
    assert all(122 <= 127 + v["exp"] <= 128 for k, v in pp.static_sites(new_p).items() if k in ("a.q_proj", "a.k_proj"))


# -- 3. the BF16 range guard ----------------------------------------------------------------------------------

def test_scale_pow2_is_exact_and_flags_overflow_and_underflow():
    g = torch.tensor([0.5, 1.0, 3.0, 0.0, -2.0], dtype=torch.bfloat16)
    out, exact, st = pf._scale_pow2(g, 5)
    assert exact and torch.equal(out.float(), g.float() * 32) and st["max_abs"] == 96.0
    big = torch.tensor([2.0 ** 126], dtype=torch.bfloat16)
    assert pf._scale_pow2(big, 5)[1] is False                                   # overflow: inf
    tiny = torch.tensor([2.0 ** -126, 1.0], dtype=torch.bfloat16)
    assert pf._scale_pow2(tiny, -1)[1] is False                                 # BF16 subnormal result counts as underflow
    assert pf._scale_pow2(tiny, 3)[1] is True


def test_bf16_guard_refuses_the_whole_group_and_reports_it():
    gamma = (torch.rand(32) + 0.5).to(torch.bfloat16)
    gamma[3] = 2.0 ** 126                                                       # x 2^5 overflows BF16
    policy, t = _policy_and_tensors(gamma)
    info, groups, new_t, new_p, rep = _fold(policy, t)
    g = groups[info["a.q_proj"]["src"]]
    assert g["applied"] is False and g["reason"].startswith("bf16_range")
    assert rep["bf16_range"]["failures"] and rep["bf16_range"]["failures"][0]["tensor"] == "model.norm.weight"
    assert torch.equal(new_t["model.norm.weight"].float(), gamma.float())       # nothing of the group was touched
    assert torch.equal(new_t["a.q_proj.weight_scale_exp"], t["a.q_proj.weight_scale_exp"])
    assert new_p["sites"]["a.q_proj"]["activation"]["exp"] == -9
    assert new_p["sites"]["a.q_proj"]["pack_fold"]["reason"].startswith("bf16_range")


def test_weight_exponent_guard():
    ok, _ = pf._weight_guard(torch.full((4, 1), -13, dtype=torch.int32), -5)
    assert ok
    ok, why = pf._weight_guard(torch.full((4, 1), -125, dtype=torch.int32), -5)
    assert not ok and "leave" in why
    ok, why = pf._weight_guard(torch.full((4, 1), 118, dtype=torch.int32), 5)
    assert not ok


def test_gamma_range_is_reported():
    policy, t = _policy_and_tensors()
    _, _, _, new_p, rep = _fold(policy, t)
    r = rep["bf16_range"]
    assert r["gamma_sites"] == 1 and r["failures"] == [] and r["max_abs_after"] <= 1.5 * 32
    assert r["min_abs_nonzero_after"] >= 0.5 * 32 and r["bf16_max"] > 3e38


# -- 4. exactness of the mechanism on real operators ----------------------------------------------------------

def _bf(shape, seed, scale=1.0):
    return (torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * scale).to(torch.bfloat16)


@pytest.mark.parametrize("s,k", [(-9, 4), (-10, 5), (-8, 3), (-5, 1), (-9, 3)])
def test_norm_fold_gives_bit_identical_codes_and_output(s, k):
    x, gamma = _bf((1, 32, 64), 1, 3.0), (torch.rand(64) + 0.5).to(torch.bfloat16)
    wc = torch.randint(0, 120, (96, 64), dtype=torch.uint8)
    wc[wc == 0x7F] = 0
    we = torch.full((96, 1), -13, dtype=torch.int32)

    def run(g, exp, w_exp):
        c, e = torch.ops.atlas.quantize_act(g * x, "static", "tensor", "ref", exp, "uint8")
        return c, torch.ops.atlas.qlinear(c, e, wc, w_exp, None, "bfloat16", "atlas_e4m3")

    c0, y0 = run(gamma, s, we)
    c1, y1 = run(gamma * (2.0 ** k), s + k, we - k)
    assert torch.equal(c0, c1) and torch.equal(y0, y1)


def test_up_proj_weight_exponent_fold_scales_the_gated_product_exactly():
    x = _bf((1, 16, 64), 2)
    wc = torch.randint(0, 120, (96, 64), dtype=torch.uint8)
    we = torch.full((96, 1), -13, dtype=torch.int32)
    gate = _bf((1, 16, 96), 3)
    c, e = torch.ops.atlas.quantize_act(x, "static", "tensor", "ref", -5, "uint8")
    up0 = torch.ops.atlas.qlinear(c, e, wc, we, None, "bfloat16", "atlas_e4m3")
    up1 = torch.ops.atlas.qlinear(c, e, wc, we + 2, None, "bfloat16", "atlas_e4m3")
    p0, p1 = torch.nn.functional.silu(gate) * up0, torch.nn.functional.silu(gate) * up1
    assert torch.equal(p1.float(), p0.float() * 4)
    c0, _ = torch.ops.atlas.quantize_act(p0, "static", "tensor", "ref", -7, "uint8")
    c1, _ = torch.ops.atlas.quantize_act(p1, "static", "tensor", "ref", -5, "uint8")
    assert torch.equal(c0, c1)


def test_pv_pop_shift_scales_the_output_exactly_and_is_default_off():
    p, v = _bf((1, 2, 8, 8), 4).abs().clamp(max=1.0), _bf((1, 2, 8, 16), 5)
    pc, pe = torch.ops.atlas.quantize_act(p, "static", "tensor", "ref", -8, "uint8")
    vc, ve = torch.ops.atlas.quantize_act(v, "dynamic", "head", "ref", None, "uint8")
    vt = vc.transpose(-1, -2)
    y0 = torch.ops.atlas.pv_matmul(pc, pe, vt, ve)
    assert torch.equal(torch.ops.atlas.pv_matmul(pc, pe, vt, ve, 0), y0)
    assert torch.equal(torch.ops.atlas.pv_matmul(pc, pe, vt, ve, 4), y0 * 16)
    assert torch.equal(torch.ops.atlas.pv_matmul(pc, pe, vt, ve, -3), y0 / 8)


# -- 5. end to end: tiny Llama --------------------------------------------------------------------------------

def _tensors(path):
    from safetensors.torch import load_file
    return load_file(str(path))


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    need_refs()
    root = tmp_path_factory.mktemp("fold")
    m, d, man, opts = _quantize_export(root / "ex", STATIC, layers=2)
    policy = json.loads((d / "policy.json").read_text())
    loaded = _reload(d, "ref_conv", opts, layers=2)
    fx_export.capture_fx(fx_export.CausalLogits(loaded).eval(), INPUTS, root / "fx0", policy=policy, fresh_process=True,
                         native_fp8="uint8")
    summary = pf.fold_export_from_fx_dir(d, root / "exf", root / "fx0")
    fpol = json.loads((root / "exf" / "policy.json").read_text())
    folded = _reload(root / "exf", "ref_conv", opts, layers=2)
    fx_export.capture_fx(fx_export.CausalLogits(folded).eval(), INPUTS, root / "fx1", policy=fpol, fresh_process=True,
                         native_fp8="uint8")
    return types.SimpleNamespace(root=root, d=d, opts=opts, policy=policy, fpol=fpol, loaded=loaded, folded=folded, summary=summary)


def test_default_export_has_none_of_the_fold_fields(tiny):
    assert "pack_fold_summary" not in tiny.policy
    assert not any("pack_fold" in s or "pack_folds" in s or "pv_pop_shift" in s for s in tiny.policy["sites"].values())
    g0 = (tiny.root / "fx0" / "graph.txt").read_text()
    assert [ln for ln in g0.splitlines() if "pv_matmul.default" in ln and ", kwargs" in ln
            and len(ln.split("args = (")[1].split("), kwargs")[0].split(",")) != 4] == []      # no pop_shift argument


def test_fold_lands_every_free_site_in_range_and_leaves_p_alone(tiny):
    s = tiny.summary
    assert s["outside_before"] == 16 and s["outside_after"] == 2 and s["in_range_after"] == 14
    assert s["outside_after_by_class"] == {"attn_p": 2} and s["folded_by_class"] == {"attn_out_pv": 2, "gated_product": 2, "norm_affine": 10}
    now = pp.static_sites(tiny.fpol)
    assert all(122 <= 127 + v["exp"] <= 128 for k, v in now.items() if not k.endswith(":p"))
    assert all(v["exp"] == -8 for k, v in now.items() if k.endswith(":p"))
    assert tiny.fpol["compiler_pack_scale_admission"]["sites_outside"] == 2


def test_fold_changes_only_gamma_weight_exponents_and_input_exponents(tiny):
    a, b = _tensors(tiny.d / "model.safetensors"), _tensors(tiny.root / "exf" / "model.safetensors")
    assert set(a) == set(b)
    changed = sorted(k for k in a if not torch.equal(a[k], b[k]))
    assert changed and all(k.endswith(("input_layernorm.weight", "post_attention_layernorm.weight", ".weight_scale_exp",
                                       ".input_scale_exp")) for k in changed)
    assert not any(k.endswith("weight_codes") for k in changed)                 # no FP8 code is touched
    for k in a:
        if k.endswith("weight_codes"):
            assert torch.equal(a[k], b[k])


def test_pop_exponent_of_every_consumer_is_unchanged(tiny):
    a, b = _tensors(tiny.d / "model.safetensors"), _tensors(tiny.root / "exf" / "model.safetensors")
    before, after = pp.static_sites(tiny.policy), pp.static_sites(tiny.fpol)
    moved = 0
    for name, st in before.items():
        if st["kind"] == "attention":
            continue
        pw = tiny.policy["sites"][name]["weight"]["scale_exp"]
        pop_a, pop_b = a[pw].to(torch.int64) + st["exp"], b[pw].to(torch.int64) + after[name]["exp"]
        if name.endswith("up_proj"):                                            # also the producer of the gated product
            assert torch.equal(pop_b - pop_a, torch.full_like(pop_a, tiny.fpol["sites"]["model.layers.0.mlp.down_proj"]["pack_fold"]["group_k"]
                                                              if name.startswith("model.layers.0") else
                                                              tiny.fpol["sites"]["model.layers.1.mlp.down_proj"]["pack_fold"]["group_k"]))
        else:
            assert torch.equal(pop_a, pop_b), name
        moved += st["exp"] != after[name]["exp"]
    assert moved > 0


def test_eager_logits_are_bit_identical(tiny):
    ids = torch.arange(24).reshape(1, 24)
    with torch.no_grad():
        a, b = tiny.loaded(ids, use_cache=False).logits, tiny.folded(ids, use_cache=False).logits
    assert torch.equal(a, b)


def test_fx_graph_outputs_are_bit_identical_and_only_pv_gains_an_argument(tiny):
    ids = (torch.arange(5, 13, dtype=torch.long).reshape(1, 8),)
    outs = []
    for fx in ("fx0", "fx1"):
        prog = torch.export.load(tiny.root / fx / "model.pt2")
        outs.append(prog.module()(*ids))
    assert torch.equal(outs[0], outs[1])
    g0, g1 = (tiny.root / "fx0" / "graph.txt").read_text(), (tiny.root / "fx1" / "graph.txt").read_text()
    assert len(g0.splitlines()) == len(g1.splitlines())
    pv = [ln for ln in g1.splitlines() if "pv_matmul.default" in ln]
    assert pv and all(ln.split("args = (")[1].split("), kwargs")[0].split(",")[-1].strip() in ("5", "4", "3", "2", "1") for ln in pv)


def test_verify_bundle_passes_the_same_checks_and_reports_remaining_sites(tiny, tmp_path):
    vb = _script("verify_bundle")
    r0 = vb.run(tiny.d, tiny.root / "fx0", tmp_path / "v0.json")
    r1 = vb.run(tiny.root / "exf", tiny.root / "fx1", tmp_path / "v1.json")
    assert r1["failed"] == r0["failed"] and set(r1["checks"]) == set(r0["checks"])
    assert sum(c["ok"] for c in r1["checks"].values()) == sum(c["ok"] for c in r0["checks"].values())
    f = r1["pack_fold"]
    assert f["folded"] is True and f["sites_still_out_of_range"] == 2 and f["still_out_of_range_by_class"] == {"attn_p": 2}
    assert f["recorded_matches_recomputed"] is True and f["sites_folded"] == 14 and len(f["pv_pop_shift_sites"]) == 2
    assert r0["pack_fold"]["folded"] is False and r0["pack_fold"]["sites_still_out_of_range"] == 16


def test_manifest_hashes_are_recomputed_and_a_second_fold_is_refused(tiny):
    import load_quantized
    load_quantized.read_export(tiny.root / "exf", verify=True)                  # sha256 of both files match the manifest
    man = json.loads((tiny.root / "exf" / "manifest.json").read_text())
    assert man["pack_fold"]["source_policy_sha256"] != man["policy_sha256"] and man["pack_fold"]["tensors_changed"] > 0
    fx = tiny.root / "fx0"
    with pytest.raises(ValueError, match="already folded"):
        pf.fold_export_from_fx_dir(tiny.root / "exf", tiny.root / "exf2", fx)


def test_exporter_flag_folds_in_place_and_keeps_the_unfolded_export(tmp_path):
    need_refs()
    import algos
    from test_export import _calib, tiny_llama
    opts = dict(FAST)
    m = tiny_llama(1)
    algo = algos.make(STATIC)
    batches, calib = _calib()
    rep = algos.quantize(m, "ref_conv", ["lm_head"], opts, algo=algo, calib=calib)
    d = tmp_path / "export"
    man = quant_export.export(m, d, mode="ref_conv", report=rep, algo=algo, algo_spec=STATIC,
                              options=types.SimpleNamespace(**opts, pack_fold=True), skip=["lm_head"],
                              model_record={"kind": "causal_lm", "id": "tiny"}, calib_batches=batches,
                              calib_record={"source": "test"}, device="cpu")
    pol = json.loads((d / "policy.json").read_text())
    un = json.loads((tmp_path / "export.unfolded" / "policy.json").read_text())
    assert pol["pack_fold_summary"]["outside_after"] < pol["pack_fold_summary"]["outside_before"] and "pack_fold_summary" not in un
    assert man["pack_fold"]["tensors_changed"] > 0
    assert pp.static_sites(un)["model.layers.0.self_attn.o_proj"]["exp"] != pp.static_sites(pol)["model.layers.0.self_attn.o_proj"]["exp"]


def test_options_without_the_flag_do_not_fold(tmp_path):
    need_refs()
    _, d, _, _ = _quantize_export(tmp_path, STATIC, layers=1)
    assert not (tmp_path / "export.unfolded").exists()
    assert "pack_fold_summary" not in json.loads((d / "policy.json").read_text())


# -- 6. LayerNorm and Gemma-style (1 + w) producers -----------------------------------------------------------

def _norm_graph(kind):
    if kind == "ln":
        g = _n("ln", "torch.ops.aten.layer_norm.default", "%x, [32], %p_model_model_ln_weight, %p_model_model_ln_bias, 1e-06")
    else:
        g = _n("m0", "torch.ops.aten.mul.Tensor", "%x, %rs")
        g += _n("w32", "torch.ops.aten.to.dtype", "%p_model_model_ln_weight, torch.float32")
        g += _n("onep", "torch.ops.aten.add.Tensor", "%w32, 1.0")
        g += _n("ln", "torch.ops.aten.mul.Tensor", "%m0, %onep")
    g += _n("n1", "torch.ops.aten.to.dtype", "%ln, torch.bfloat16")
    return g + _pack_chain(0, "n1", -9, "a_q_proj") + _pack_chain(1, "n1", -9, "a_k_proj")


_LN_SIG = ("p_model_model_ln_weight: PARAMETER target='model.model.ln.weight'\n"
           "p_model_model_ln_bias: PARAMETER target='model.model.ln.bias'\n"
           + "".join(f"b_model_model_a_{n}_weight_codes: BUFFER target='model.model.a.{n}.weight_codes' persistent=False\n"
                     for n in ("q_proj", "k_proj")))


def _norm_case(kind, dtype):
    policy, t = _policy_and_tensors()
    policy["sites"] = {k: v for k, v in policy["sites"].items() if k in ("a.q_proj", "a.k_proj")}
    t = {k: v for k, v in t.items() if k.startswith(("a.q_proj", "a.k_proj"))}
    policy["sites"]["a.k_proj"]["activation"]["exp"] = -9                       # one shared code, 118: k = 4
    gen = torch.Generator().manual_seed(7)
    t["ln.weight"] = (torch.rand(32, generator=gen) - 0.5).to(dtype) if kind == "onep" else (torch.rand(32, generator=gen) + 0.5).to(dtype)
    t["ln.bias"] = (torch.rand(32, generator=gen) - 0.5).to(dtype)
    info = pf.graph_info(_norm_graph(kind), _LN_SIG, policy, set(t))
    groups = pf.plan_folds(policy, info)
    return policy, t, info, groups, pf.apply_folds(t, policy, groups)


def test_layer_norm_weight_and_bias_are_both_scaled():
    policy, t, info, groups, (nt, npol, rep) = _norm_case("ln", torch.bfloat16)
    assert info["a.q_proj"]["class"] == "layer_norm" and info["a.q_proj"]["producer"]["bias_tensor"] == "ln.bias"
    assert all(g["applied"] for g in groups.values()) and next(iter(groups.values()))["k"] == 4
    assert torch.equal(nt["ln.weight"].float(), t["ln.weight"].float() * 16) and torch.equal(nt["ln.bias"].float(), t["ln.bias"].float() * 16)
    assert npol["sites"]["a.q_proj"]["activation"]["exp"] == -5 == npol["sites"]["a.k_proj"]["activation"]["exp"]


def test_gemma_one_plus_w_fold_is_exact_in_float32_and_refused_in_bf16():
    policy, t, info, groups, (nt, npol, rep) = _norm_case("onep", torch.float32)
    assert info["a.q_proj"]["producer"]["kind"] == "norm_one_plus"
    g = next(iter(groups.values()))
    assert g["applied"] and g["k"] == 4
    assert torch.equal(nt["ln.weight"] + 1.0, (t["ln.weight"] + 1.0) * 16)                       # the multiplier the graph forms
    policy, t, info, groups, (nt, npol, rep) = _norm_case("onep", torch.bfloat16)
    g = next(iter(groups.values()))
    assert not g["applied"] and "float32" in g["reason"]
    assert torch.equal(nt["ln.weight"].float(), t["ln.weight"].float()) and npol["sites"]["a.q_proj"]["activation"]["exp"] == -9


# -- 7. shared producers, aliases, weights read twice ---------------------------------------------------------

def test_gamma_placeholder_read_by_two_nodes_is_not_scaled():
    policy, t = _policy_and_tensors()
    g = _graph() + _n("m_twice", "torch.ops.aten.mul.Tensor", "%p_model_model_norm_weight, %m0")
    info = pf.graph_info(g, _SIG, policy, set(t))
    groups = pf.plan_folds(policy, info)
    grp = groups[info["a.q_proj"]["src"]]
    assert grp["reason"].startswith("producer_unsupported") and "more than one node" in grp["reason"]
    new_t, new_p, _ = pf.apply_folds(t, policy, groups)
    assert torch.equal(new_t["model.norm.weight"].float(), t["model.norm.weight"].float())


def test_aliased_tensor_is_not_edited():
    policy, t = _policy_and_tensors()
    policy["aliases"] = {"model.tied.weight": "model.norm.weight"}
    info = pf.graph_info(_graph(), _SIG, policy, set(t))
    groups = pf.plan_folds(policy, info)
    new_t, new_p, rep = pf.apply_folds(t, policy, groups)
    assert groups[info["a.q_proj"]["src"]]["reason"].startswith("aliased_tensor")
    assert torch.equal(new_t["model.norm.weight"].float(), t["model.norm.weight"].float())
    assert new_p["sites"]["a.q_proj"]["activation"]["exp"] == -9
    assert groups[info["a.down_proj"]["src"]]["applied"]                         # an unrelated group still folds


def test_weights_read_by_a_second_node_block_the_group():
    policy, t = _policy_and_tensors()
    g = _graph() + _n("ql_again", "torch.ops.atlas.qlinear.default",
                      "%ga0, %gb0, %b_model_model_a_q_proj_weight_codes, %b_model_model_a_q_proj_weight_scale_exp, None, bfloat16")
    info = pf.graph_info(g, _SIG, policy, set(t))
    grp = pf.plan_folds(policy, info)[info["a.q_proj"]["src"]]
    assert grp["reason"].startswith("multi_call_site") and "a.q_proj" in grp["reason"]


def test_failed_group_adds_nothing_to_the_range_report():
    gamma = (torch.rand(32) + 0.5).to(torch.bfloat16)
    policy, t = _policy_and_tensors(gamma)
    t["a.q_proj.weight_scale_exp"] = torch.full((64, 1), -125, dtype=torch.int32)   # weight exponent guard fails, gamma is exact
    _, groups, new_t, new_p, rep = _fold(policy, t)
    assert rep["bf16_range"]["gamma_sites"] == 0 and rep["bf16_range"]["max_abs_after"] is None
    assert torch.equal(new_t["model.norm.weight"].float(), gamma.float())


def test_the_comparison_is_not_vacuous_a_fold_without_the_weight_compensation_changes_the_output():
    x, gamma = _bf((1, 32, 64), 1, 3.0), (torch.rand(64) + 0.5).to(torch.bfloat16)
    wc = torch.randint(0, 120, (96, 64), dtype=torch.uint8)
    we = torch.full((96, 1), -13, dtype=torch.int32)
    s, k = -9, 4

    def run(g, exp, w_exp):
        c, e = torch.ops.atlas.quantize_act(g * x, "static", "tensor", "ref", exp, "uint8")
        return torch.ops.atlas.qlinear(c, e, wc, w_exp, None, "bfloat16", "atlas_e4m3")

    y0 = run(gamma, s, we)
    assert not torch.equal(run(gamma * (2.0 ** k), s + k, we), y0)                  # consumer weight exponent left alone
    assert torch.equal(run(gamma * (2.0 ** k), s + k, we - k), y0)
    assert not torch.equal(run(gamma * (2.0 ** k), s, we), y0)                      # pack exponent not moved with the value


def test_attention_site_with_two_pv_nodes_is_not_folded():
    """pv_pop_shift is stored per attention SITE: a site whose module is called twice has two P.V nodes and the shift would reach both
    (SmolVLA's shared text attention has 24). The group is refused and nothing of it changes."""
    policy, t = _policy_and_tensors()
    g = _graph() + _n("pv_again", "torch.ops.atlas.pv_matmul.default", "%ga5, %gb5, %vt, %vs")
    info = pf.graph_info(g, _SIG, policy, set(t))
    groups = pf.plan_folds(policy, info)
    grp = groups[info["a.o_proj"]["src"]]
    assert grp["reason"].startswith("attention_site_called_more_than_once") and "2 atlas.pv_matmul" in grp["reason"]
    new_t, new_p, _ = pf.apply_folds(t, policy, groups)
    assert "pv_pop_shift" not in new_p["sites"]["model.a.attn"] and new_p["sites"]["a.o_proj"]["activation"]["exp"] == -6
    assert torch.equal(new_t["a.o_proj.weight_scale_exp"], t["a.o_proj.weight_scale_exp"])
