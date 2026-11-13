"""Pack-scale plan (src/exporter/pack_plan.py, docs/algos/tile-aware.md#pack-plan): REPORTING ONLY. Unit tests on synthetic exponents and a synthetic
FX graph text, bit-identical-export checks, and an end-to-end tiny Llama (static recipe, native-FP8 FX capture, verify_bundle)."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
import exporter.fx_export as fx_export
import exporter.pack_plan as pp
import exporter.quant_export as quant_export
import exporter.tile_native as tn
from refutil import need_refs
from test_export import _quantize_export, _reload
from test_ladder import FAST  # noqa: F401

SCRIPTS = Path(__file__).resolve().parents[1] / "src" / "scripts" / "tile_flow"
STATIC = "atlas_v1_awq_tile_static"
INPUTS = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
          (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# -- 1. the fold on synthetic exponents ---------------------------------------------------------------------

def test_admission_constants_come_from_tile_native():
    assert pp.PACK_ADMISSION == tn.PACK_ADMISSION == (122, 128) and pp.E8M0_BIAS == tn.E8M0_BIAS == 127


@pytest.mark.parametrize("exp,code,in_range,k,target,k127", [
    (0, 127, True, 0, 127, 0), (-5, 122, True, 0, 122, 5), (1, 128, True, 0, 128, -1),
    (-6, 121, False, 1, 122, 6), (-8, 119, False, 3, 122, 8), (-10, 117, False, 5, 122, 10),
    (2, 129, False, -1, 128, -2), (6, 133, False, -5, 128, -6)])
def test_fold_to_nearest_in_range_code(exp, code, in_range, k, target, k127):
    f = pp.fold_for_code(pp.e8m0_pack_code(exp))
    assert pp.e8m0_pack_code(exp) == code
    assert (f["in_range"], f["k"], f["target_code"], f["k_to_unit_127"], f["feasible"]) == (in_range, k, target, k127, True)
    assert abs(f["k"]) <= abs(f["k_to_unit_127"]) or in_range        # nearest never needs a larger fold than unit 127


def test_fold_unit_target_and_infeasible():
    f = pp.fold_for_code(119, target=127)
    assert (f["k"], f["target_code"]) == (8, 127)
    assert pp.fold_for_code(133, target=127)["k"] == -6
    assert pp.fold_for_code(100)["feasible"] is False and pp.fold_for_code(100)["k"] == 22
    assert pp.fold_for_code(122 - pp.MAX_ABS_K)["feasible"] is True and pp.fold_for_code(122 - pp.MAX_ABS_K - 1)["feasible"] is False


# -- 2. the record: class, cost, placement --------------------------------------------------------------------

@pytest.mark.parametrize("cls,cost,free", [
    ("norm_affine", pp.COST_FREE, True), ("layer_norm", pp.COST_FREE, True), ("gated_product", pp.COST_FREE, True),
    ("attn_out_pv", pp.COST_FREE, True), ("linear_to_linear", pp.COST_FREE, True),
    ("attn_p", pp.COST_VMUL_ROW_BLOCK, False), ("attn_out_sdpa", pp.COST_VMUL_TILE, False),
    ("modulation_add", pp.COST_VMUL_TILE, False), ("bare_activation", pp.COST_ACT_REWRITE, False),
    ("x_sigmoid_product", pp.COST_ACT_REWRITE, False), ("raw_input", pp.COST_HOST, False),
    ("other:aten.sin.default", pp.COST_VMUL_TILE, False), ("unknown", pp.COST_VMUL_TILE, False)])
def test_cost_per_producer_class(cls, cost, free):
    r = pp.plan_record(-8, {"class": cls, "other_consumers": [], "source": "fx_graph"})
    assert r["e8m0_code"] == 119 and r["in_range"] is False and r["status"] == "fold_planned"
    assert (r["cost"], r["fold_free"]) == (cost, free) and r["fold"]["k"] == 3 and r["fold"]["k_to_unit_127"] == 8


def test_attention_p_needs_vmul_on_recip_sum():
    r = pp.plan_record(-8, {"class": "attn_p", "other_consumers": [], "source": "policy_operand"})
    assert r["placement"] == "softmax_multiplier" and "recip(sum)" in r["note"] and "recip(sum)" in r["how"]
    assert r["fold_free"] is False


def test_second_consumer_loses_the_free_placement():
    r = pp.plan_record(-8, {"class": "linear_to_linear", "other_consumers": ["aten.expand.default"], "source": "fx_graph"})
    assert r["cost"] == pp.COST_VMUL_TILE and r["fold_free"] is False and "other consumers" in r["how"]
    h = pp.plan_record(-8, {"class": "raw_input", "other_consumers": ["aten.add.Tensor"], "source": "fx_graph"})
    assert h["cost"] == pp.COST_HOST


def test_in_range_and_infeasible_records():
    r = pp.plan_record(-5, {"class": "norm_affine", "other_consumers": [], "source": "fx_graph"})
    assert r["in_range"] and r["fold"] is None and r["cost"] == pp.COST_NONE and r["status"] == "in_range"
    bad = pp.plan_record(-30, {"class": "norm_affine", "other_consumers": [], "source": "fx_graph"})
    assert bad["status"] == "infeasible" and bad["cost"] == pp.COST_INFEASIBLE and not bad["fold_free"]
    assert pp.plan_record(-8)["producer"]["class"] == "unknown"


def _policy(exps, p_exp=-8):
    sites = {f"l{i}": {"kind": "linear", "status": "quantized", "activation": {"mode": "static", "exp": e},
                       "weight": {"e8m0_code_range": {"min": 115, "max": 119}}} for i, e in enumerate(exps)}
    sites["attn"] = {"kind": "attention", "status": "quantized", "operands": {
        "q": {"mode": "dynamic", "source": "reference"}, "k": {"mode": "dynamic", "source": "reference"},
        "p": {"mode": "static", "exp": p_exp}, "v": {"mode": "dynamic", "source": "reference"}}}
    sites["dyn"] = {"kind": "linear", "status": "quantized", "activation": {"mode": "dynamic", "source": "reference"}}
    sites["skip"] = {"kind": "Linear", "status": "skipped"}
    return {"sites": sites}


def test_static_and_dynamic_site_keys():
    pol = _policy([-9, 0])
    assert pp.static_sites(pol) == {"l0": {"exp": -9, "kind": "linear", "site": "l0", "operand": None},
                                    "l1": {"exp": 0, "kind": "linear", "site": "l1", "operand": None},
                                    "attn:p": {"exp": -8, "kind": "attention", "site": "attn", "operand": "p"}}
    assert set(pp.dynamic_sites(pol)) == {"dyn", "attn:q", "attn:k", "attn:v"}


def test_summary_counts_per_class_and_cost_and_attach():
    pol = _policy([-9, 0, -7])
    prod = {"l0": {"class": "norm_affine", "other_consumers": [], "source": "fx_graph"},
            "l1": {"class": "norm_affine", "other_consumers": [], "source": "fx_graph"},
            "l2": {"class": "attn_out_sdpa", "other_consumers": [], "source": "fx_graph"}}
    out = pp.with_plan(pol, prod)
    assert "pack_plan" not in pol["sites"]["l0"] and "pack_plan_summary" not in pol          # input untouched
    s = out["pack_plan_summary"]
    assert (s["static_sites"], s["sites_outside"], s["sites_in_range"]) == (4, 3, 1)
    assert s["outside_by_class"] == {"attn_p": 1, "attn_out_sdpa": 1, "norm_affine": 1}
    assert s["outside_by_cost"] == {"free": 1, "vmul_per_row_block": 1, "vmul_per_tile": 1}
    assert s["free_fold_sites"] == 1 and s["vmul_sites"] == 2 and s["code_range"] == [118, 127] and s["k_range"] == [2, 4]
    assert out["sites"]["l0"]["pack_plan"]["fold"]["target_code"] == 122
    assert out["sites"]["attn"]["pack_plans"]["p"]["producer"]["class"] == "attn_p"          # by structure, no graph
    assert s["dynamic_sites_not_expressible_as_scale_code"]["count"] == 4 and s["informational"] is True
    assert s["pop_exponent_bound"] == {"sites": 3, "min": -9 + 115 - 127, "max": 0 + 119 - 127, "ok": True, "required": [-126, 127]}


def test_summary_without_static_sites_is_not_checked_not_zero():
    pol = {"sites": {"d": {"kind": "linear", "status": "quantized", "activation": {"mode": "dynamic"}}}}
    s = pp.summarize(pp.build_plan(pol), pol)
    assert s["status"] == "not_checked_dynamic" and s["static_sites"] is None and s["sites_outside"] is None
    assert s["dynamic_sites_not_expressible_as_scale_code"]["count"] == 1


def test_require_pack_range_flags_missing_and_infeasible_only():
    ok = pp.plan_record(-8, {"class": "norm_affine", "other_consumers": [], "source": "fx_graph"})
    bad = pp.plan_record(-30)
    assert pp.require_pack_range({"a": ok, "b": bad, "c": None}) == ["b", "c"]
    assert pp.require_pack_range({"a": ok}) == []


# -- 3. the fold is exact (spec item 7 helper) ------------------------------------------------------------------

@pytest.mark.parametrize("s,k", [(-8, 3), (-10, 5), (-7, 2), (6, -5)])
def test_fold_is_bit_exact_and_clamping_alone_is_not(s, k):
    torch.manual_seed(3)
    x = (torch.randn(64, 128) * (2.0 ** (s + 5))).to(torch.bfloat16)
    r = pp.fold_exactness(x, s, k)
    assert r["differing_bytes"] == 0 and r["scaling_exact_in_bf16"] is True and r["elements"] == x.numel()
    import reference.ref_numerics as rn
    moved_exp_only = rn.e4m3_vpack(x, torch.tensor(s + k, dtype=torch.int32))       # exponent clamped, value left alone
    assert int((rn.e4m3_vpack(x, torch.tensor(s, dtype=torch.int32)) != moved_exp_only).sum()) > 0


# -- 4. producer classes from an FX graph text ------------------------------------------------------------------

def _node(name, tgt, args, users=1):
    return f"    %{name} : [num_users={users}] = call_function[target={tgt}](args = ({args}), kwargs = {{}})\n"


def _synthetic_graph(second_consumer=False):
    g = _node("mul", "torch.ops.aten.mul.Tensor", "%x, %rsqrt")
    g += _node("mul_1", "torch.ops.aten.mul.Tensor", "%p_model_norm_weight, %mul")
    g += _node("to_1", "torch.ops.aten.to.dtype", "%mul_1, torch.bfloat16")
    g += _node("quantize_act", "torch.ops.atlas.quantize_act.default", "%to_1, static, tensor, ref, -9, uint8", 2)
    g += _node("getitem", "operator.getitem", "%quantize_act, 0") + _node("getitem_1", "operator.getitem", "%quantize_act, 1")
    g += _node("qlinear", "torch.ops.atlas.qlinear.default",
               "%getitem, %getitem_1, %b_model_model_a_up_proj_weight_codes, %b_model_model_a_up_proj_weight_scale_exp, None, bfloat16")
    g += _node("qlinear_1", "torch.ops.atlas.qlinear.default",
               "%getitem, %getitem_1, %b_model_model_a_gate_proj_weight_codes, %b_model_model_a_gate_proj_weight_scale_exp, None, bfloat16")
    g += _node("silu", "torch.ops.aten.silu.default", "%qlinear")
    g += _node("mul_2", "torch.ops.aten.mul.Tensor", "%silu, %qlinear_1", 2 if second_consumer else 1)
    g += _node("to_2", "torch.ops.aten.to.dtype", "%mul_2, torch.bfloat16")
    g += _node("quantize_act_1", "torch.ops.atlas.quantize_act.default", "%to_2, static, tensor, ref, -7, uint8", 2)
    g += _node("getitem_2", "operator.getitem", "%quantize_act_1, 0") + _node("getitem_3", "operator.getitem", "%quantize_act_1, 1")
    g += _node("qlinear_2", "torch.ops.atlas.qlinear.default",
               "%getitem_2, %getitem_3, %b_model_model_a_down_proj_weight_codes, %b_model_model_a_down_proj_weight_scale_exp, None, bfloat16")
    if second_consumer:
        g += _node("add", "torch.ops.aten.add.Tensor", "%mul_2, %x")
    g += _node("softmax", "torch.ops.aten.softmax.int", "%scores, -1")
    g += _node("quantize_act_2", "torch.ops.atlas.quantize_act.default", "%softmax, static, tensor, ref, -8, uint8, model.a.attn:p", 2)
    g += _node("getitem_4", "operator.getitem", "%quantize_act_2, 0") + _node("getitem_5", "operator.getitem", "%quantize_act_2, 1")
    g += _node("pv_matmul", "torch.ops.atlas.pv_matmul.default", "%getitem_4, %getitem_5, %vt, %vs")
    g += _node("quantize_act_3", "torch.ops.atlas.quantize_act.default", "%pv_matmul, static, tensor, ref, -6, uint8", 2)
    g += _node("getitem_6", "operator.getitem", "%quantize_act_3, 0") + _node("getitem_7", "operator.getitem", "%quantize_act_3, 1")
    g += _node("qlinear_3", "torch.ops.atlas.qlinear.default",
               "%getitem_6, %getitem_7, %b_model_model_a_o_proj_weight_codes, %b_model_model_a_o_proj_weight_scale_exp, None, bfloat16")
    return g


SIG = "".join(f"b_model_model_a_{n}_weight_codes: BUFFER target='model.model.a.{n}.weight_codes' persistent=False\n"
              for n in ("up_proj", "down_proj", "o_proj"))


def _graph_policy():
    sites = {f"a.{n}": {"kind": "linear", "status": "quantized", "activation": {"mode": "static", "exp": e}}
             for n, e in (("up_proj", -9), ("down_proj", -7), ("o_proj", -6))}
    sites["model.a.attn"] = {"kind": "attention", "status": "quantized", "operands": {"p": {"mode": "static", "exp": -8}}}
    return {"sites": sites}


def test_producer_classes_from_graph_text():
    pol = _graph_policy()
    pol["sites"]["a.attn"] = pol["sites"].pop("model.a.attn")
    prod, shared = pp.producers_from_graph(_synthetic_graph(), SIG, pol)
    assert {k: v["class"] for k, v in prod.items()} == {"a.up_proj": "norm_affine", "a.down_proj": "gated_product",
                                                         "a.attn:p": "attn_p", "a.o_proj": "attn_out_pv"}
    assert all(v["other_consumers"] == [] and v["source"] == "fx_graph" for v in prod.values())
    plan = pp.plan_from_files(pol, _synthetic_graph(), SIG)
    assert {k: r["cost"] for k, r in plan.items()} == {"a.up_proj": "free", "a.down_proj": "free", "a.attn:p": "vmul_per_row_block",
                                                       "a.o_proj": "free"}
    assert plan["a.o_proj"]["in_range"] is False and plan["a.up_proj"]["fold"]["k"] == 4
    assert shared == {}


def test_other_consumers_are_found_and_cost_a_vmul():
    pol = _graph_policy()
    pol["sites"]["a.attn"] = pol["sites"].pop("model.a.attn")
    prod, _ = pp.producers_from_graph(_synthetic_graph(second_consumer=True), SIG, pol)
    assert prod["a.down_proj"]["other_consumers"] == ["aten.add.Tensor"]
    plan = pp.plan_from_files(pol, _synthetic_graph(second_consumer=True), SIG)
    assert plan["a.down_proj"]["cost"] == pp.COST_VMUL_TILE and plan["a.up_proj"]["cost"] == pp.COST_FREE


def test_no_graph_means_unknown_producer():
    pol = _graph_policy()
    plan = pp.plan_from_files(pol)
    assert {r["producer"]["class"] for k, r in plan.items()} == {"unknown", "attn_p"} and all(
        r["producer"]["source"] in ("none", "policy_operand") for r in plan.values())
    assert pp.summarize(plan, pol)["outside_unknown_producer"] == 3


# -- 5. exports: bit-identical numerics, real tiny Llama end to end -----------------------------------------------

def _tree(d):
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted(Path(d).rglob("*")) if p.is_file()}


def _tensors(path):
    """name -> (dtype, shape, raw bytes): the file bytes of a safetensors file carry a run-dependent header, the tensors do not."""
    from safetensors import safe_open
    with safe_open(str(path), "pt") as f:
        return {k: (str(f.get_tensor(k).dtype), tuple(f.get_tensor(k).shape), f.get_tensor(k).contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
                for k in f.keys()}


def _strip(policy):
    policy = json.loads(json.dumps(policy))
    policy.pop("pack_plan_summary", None)
    for s in policy["sites"].values():
        s.pop("pack_plan", None)
        s.pop("pack_plans", None)
    return policy


def test_export_is_bit_identical_with_and_without_the_plan(tmp_path, monkeypatch):
    need_refs()
    _, d_plan, _, _ = _quantize_export(tmp_path / "a", STATIC, layers=1)
    monkeypatch.setattr(quant_export.pack_plan, "attach", lambda policy, plan, summary=None: policy)
    _, d_none, _, _ = _quantize_export(tmp_path / "b", STATIC, layers=1)
    a, b = _tree(d_plan), _tree(d_none)
    assert _tensors(d_plan / "model.safetensors") == _tensors(d_none / "model.safetensors")
    pa, pb = json.loads(a["policy.json"]), json.loads(b["policy.json"])
    assert "pack_plan_summary" in pa and "pack_plan_summary" not in pb
    assert _strip(pa) == pb and pa["compiler_pack_scale_admission"] == pb["compiler_pack_scale_admission"]
    # every static site carries a record; admission report unchanged
    rec = pp.recorded_plan(pa)
    assert rec and all(r is not None for r in rec.values())
    assert pa["compiler_pack_scale_admission"]["sites_checked"] == pa["pack_plan_summary"]["static_sites"] == len(rec)
    assert pa["compiler_pack_scale_admission"]["sites_outside"] == pa["pack_plan_summary"]["sites_outside"]
    assert not any(r["producer"]["class"] not in ("unknown", "attn_p") for r in rec.values())     # no graph at export time


def test_dynamic_recipe_plan_is_not_checked(tmp_path):
    need_refs()
    _, d, _, _ = _quantize_export(tmp_path, "atlas_v1_awq_tile", layers=1)
    pol = json.loads((d / "policy.json").read_text())
    assert pol["compiler_pack_scale_admission"]["status"] == "not_checked_dynamic"
    assert pol["pack_plan_summary"]["status"] == "not_checked_dynamic" and pol["pack_plan_summary"]["static_sites"] is None


@pytest.fixture(scope="module")
def tiny_bundle(tmp_path_factory):
    """Real tiny Llama, static recipe: the export, then two native-FP8 captures (with the plan, and with it switched off)."""
    need_refs()
    root = tmp_path_factory.mktemp("pack")
    _, d, _, opts = _quantize_export(root / "ex", STATIC, layers=1)
    loaded = _reload(d, "ref_conv", opts, layers=1)
    policy = json.loads((d / "policy.json").read_text())
    wrapped = fx_export.CausalLogits(loaded).eval()
    fx_export.capture_fx(wrapped, INPUTS, root / "fx_plan", policy=policy, fresh_process=True, native_fp8="uint8")
    real = fx_export._pack_plan
    fx_export._pack_plan = lambda program, pol: (pol, {}, None)
    try:
        fx_export.capture_fx(wrapped, INPUTS, root / "fx_none", policy=policy, fresh_process=False, native_fp8="uint8")
    finally:
        fx_export._pack_plan = real
    return d, root / "fx_plan", root / "fx_none", policy


def test_fx_graph_and_numerics_bit_identical_with_and_without_the_plan(tiny_bundle):
    d, fx_a, fx_b, _ = tiny_bundle
    a, b = _tree(fx_a), _tree(fx_b)
    for name in ("graph.txt", "graph_signature.txt", "buffer_inventory.json", "parity.json"):
        assert a[name] == b[name], name
    assert _tensors(fx_a / "fixtures.safetensors") == _tensors(fx_b / "fixtures.safetensors")
    assert json.loads(a["verification.json"])["model_pt2_bytes"] == json.loads(b["verification.json"])["model_pt2_bytes"]
    ga, gb = json.loads(a["graph_inspection.json"]), json.loads(b["graph_inspection.json"])
    assert "pack_plan_summary" in ga and gb["pack_plan_summary"] is None
    ga.pop("pack_plan_summary"), ga.pop("pack_plan"), gb.pop("pack_plan_summary"), gb.pop("pack_plan")
    assert ga == gb
    assert _strip(json.loads(a["policy.json"])) == _strip(json.loads(b["policy.json"]))


def test_tiny_llama_end_to_end_classes_and_verify_bundle(tiny_bundle, tmp_path):
    d, fx, _, export_policy = tiny_bundle
    gi = json.loads((fx / "graph_inspection.json").read_text())
    fx_pol = json.loads((fx / "policy.json").read_text())
    s = gi["pack_plan_summary"]
    assert s == fx_pol["pack_plan_summary"] and s["static_sites"] == 7 + 1                   # 7 Linears + attention p
    assert s["producer_source"] == {"fx_graph": 8} and s["outside_unknown_producer"] == 0
    assert gi["compiler_pack_scale_admission"]["sites_outside"] == s["sites_outside"] and s["sites_outside"] >= 1
    cls = {k: r["producer"]["class"] for k, r in pp.recorded_plan(fx_pol).items()}
    assert {k.rsplit(".", 1)[-1]: c for k, c in cls.items()} == {
        "q_proj": "norm_affine", "k_proj": "norm_affine", "v_proj": "norm_affine", "gate_proj": "norm_affine",
        "up_proj": "norm_affine", "o_proj": "attn_out_pv", "down_proj": "gated_product", "self_attn:p": "attn_p"}
    assert set(gi["pack_plan"]) == set(cls) and gi["pack_plan"] == pp.recorded_plan(fx_pol)
    p = next(r for k, r in gi["pack_plan"].items() if k.endswith(":p"))
    assert p["e8m0_code"] == 119 and p["fold"]["k"] == 3 and p["cost"] == "vmul_per_row_block" and "recip(sum)" in p["note"]
    # verify_bundle: informational report, no failing check, opt-in flag passes on a planned bundle
    vb = _script("verify_bundle")
    res = vb.run(d, fx, tmp_path / "v.json")
    assert "pack_range_plan_present" not in res["checks"] and "pack_plan" in res
    assert res["pack_plan"]["recomputed"]["sites_outside"] == s["sites_outside"]
    assert res["pack_plan"]["recorded_matches_recomputed"]["graph_inspection"] is True
    assert res["pack_plan"]["recorded_matches_recomputed"]["fx_policy"] is True
    on = vb.run(d, fx, tmp_path / "v2.json", require_pack_range=True)
    assert on["checks"]["pack_range_plan_present"]["ok"] is True
    assert set(on["failed"]) == set(res["failed"])


def test_require_pack_range_fails_on_out_of_range_site_without_plan(tiny_bundle, tmp_path):
    d, fx, _, _ = tiny_bundle
    vb = _script("verify_bundle")
    ex2, fx2 = tmp_path / "ex", tmp_path / "fx"
    ex2.mkdir()
    fx2.mkdir()
    for f in d.iterdir():
        (ex2 / f.name).symlink_to(f)
    for f in fx.iterdir():
        (fx2 / f.name).symlink_to(f)
    for root in (ex2, fx2):                                       # a legacy bundle: no plan records anywhere
        (root / "policy.json").unlink()
        pol = _strip(json.loads((fx if root is fx2 else d).joinpath("policy.json").read_text()))
        (root / "policy.json").write_text(json.dumps(pol))
    res = vb.run(ex2, fx2, tmp_path / "v.json", require_pack_range=True)
    chk = res["checks"]["pack_range_plan_present"]
    assert chk["ok"] is False and "pack_range_plan_present" in res["failed"] and chk["detail"]["sites_outside_without_plan"] >= 1
    assert vb.run(ex2, fx2, tmp_path / "v2.json")["checks"].get("pack_range_plan_present") is None     # informational by default
    assert vb.run(ex2, fx2, tmp_path / "v3.json")["pack_plan"]["sites_with_recorded_plan"] == 0
