"""capture_fx counts how often the eager replay calls each quantized module and expects the graph to agree:
0 calls = the module is absent from this graph, 1 call = exactly one node set (the existing strict check), N calls = N node
sets, each with its own quantize_act. Linear, Conv2d and attention sites, strict and non-strict export, plus the negative
cases that show the relaxed checks still fail on a graph that disagrees with the count (docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest)."""
import json

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
import exporter.fx_export as fx_export
import pytest
import exporter.quant_export as quant_export
import torch
from test_ladder import FAST

SPEC = "granularity:w=channel,a=token,rule=nc"


def _policy(module):
    algo = algos.make(SPEC)
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    return quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})


class _Branches(torch.nn.Module):
    """Two Linear layers; `order` names the layers the boundary calls, in order (a name may repeat or be absent)."""

    def __init__(self, order):
        super().__init__()
        self.a, self.b, self.c = (torch.nn.Linear(8, 8) for _ in range(3))
        self.order = tuple(order)

    def forward(self, x):
        for name in self.order:
            x = x + getattr(self, name)(x)
        return x


def _lin(order):
    torch.manual_seed(5)
    module = _Branches(order).to(torch.bfloat16).eval()
    return module, _policy(module)


def _x(scale):
    return (torch.randn(2, 8, generator=torch.Generator().manual_seed(3)) * scale).to(torch.bfloat16)


def _targets(directory):
    program = torch.export.load(directory / "model.pt2")
    return [str(n.target) for n in program.graph.nodes if n.op == "call_function"]


# -- Linear: 0, 1 and N calls, strict and non-strict, decode and native ------------------------------------------------

@pytest.mark.parametrize("strict", [True, False])
@pytest.mark.parametrize("native", [None, "uint8"])
@pytest.mark.parametrize("order,calls", [(("a",), {"a": 1}), (("a", "a"), {"a": 2}), (("a", "b", "a", "a"), {"a": 3, "b": 1})])
def test_linear_call_counts_zero_one_and_n(tmp_path, order, calls, native, strict):
    module, policy = _lin(order)
    result = fx_export.capture_fx(module, [(_x(1.0),), (_x(4.0),)], tmp_path / "fx", policy=policy, strict=strict,
                                  fresh_process=False, native_fp8=native)
    assert result["status"] == "passed"
    unused = sorted(n for n in "abc" if n not in calls)
    assert result["unused_quantized_modules"] == unused
    assert result["quantized_module_calls"] == {k: n for k, n in calls.items() if n != 1}
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert {r["module"]: r["calls"] for r in inspection["weight_site_bindings"]} == calls
    assert sorted(r["module"] for r in inspection["unused_quantized_modules"]) == unused
    inventory = json.loads((tmp_path / "fx" / "buffer_inventory.json").read_text())
    assert sorted({k.split(".")[0] for k in inventory}) == sorted(calls)                 # unused layers are not in the inventory
    targets = _targets(tmp_path / "fx")
    if native:
        assert targets.count("atlas.qlinear.default") == sum(calls.values()) == inspection["native_fp8_linear_calls"]
        assert targets.count("atlas.quantize_act.default") == sum(calls.values())
        assert inspection["native_fp8_linear_sites"] == len(calls)
        for row in inspection["weight_site_bindings"]:
            if row["calls"] > 1:
                assert len(row["qlinear_nodes"]) == len(set(row["quantize_act_nodes"])) == row["calls"]
        assert inspection["still_emulated"]["emulated_modules"] == {}
    else:
        assert "atlas.qlinear.default" not in targets


def test_non_strict_export_lifts_unused_buffers_and_they_stay_unconsumed(tmp_path):
    module, policy = _lin(("a",))
    fx_export.capture_fx(module, [(_x(1.0),), (_x(4.0),)], tmp_path / "fx", policy=policy, strict=False,
                         fresh_process=False, native_fp8="uint8")
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    lifted = {k for k, _ in program.named_buffers() if k.startswith(("b.", "c."))}
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    declared = {name for row in inspection["unused_quantized_modules"] for name in row.get("buffers_lifted_unused", [])}
    assert lifted and declared                                                         # non-strict keeps them as graph inputs
    by_name = {n.name: n for n in program.graph.nodes}
    assert all(not by_name[n].users for n in declared)


# -- negative cases -------------------------------------------------------------------------------------------------------

class _DataDependent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.a(x) + self.a(x) if bool(x.sum() > 0) else self.a(x)


def test_call_count_that_differs_between_cases_is_refused(tmp_path):
    torch.manual_seed(1)
    module = _DataDependent().to(torch.bfloat16).eval()
    x = _x(1.0).abs()
    with pytest.raises(fx_export.FXVerificationError, match="different number of times"):
        fx_export.capture_fx(module, [(x,), (-x,)], tmp_path / "fx", policy=_policy(module), strict=False,
                             fresh_process=False, native_fp8="uint8")


def _native_program(module, policy, strict=True, case=None):
    case = case if case is not None else (_x(1.0),)
    with fx_export._meta_placeholders(module), fx_export._native_sites(module, policy, "uint8") as plan:
        program = torch.export.export(module, case, strict=strict)
    return program, plan


def test_inspect_graph_refuses_a_call_count_that_disagrees_with_the_graph():
    module, policy = _lin(("a", "a"))
    program, plan = _native_program(module, policy)
    ok = fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b", "c"), calls={"a": 2})
    assert ok["native_fp8_linear_calls"] == 2
    with pytest.raises(fx_export.FXVerificationError, match="found 2"):                   # default = 1 call: strict as before
        fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b", "c"))
    with pytest.raises(fx_export.FXVerificationError, match="exactly 3"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b", "c"), calls={"a": 3})
    with pytest.raises(fx_export.FXVerificationError, match="not listed as unused"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b",), calls={"a": 2, "b": 0, "c": 0})


def test_inspect_graph_refuses_to_treat_a_called_module_as_unused():
    module, policy = _lin(("a", "b"))
    program, plan = _native_program(module, policy)
    with pytest.raises(fx_export.FXVerificationError, match="never called by the eager replay but present"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b", "c"), calls={"a": 1})


def test_inspect_graph_refuses_two_calls_that_share_one_quantize_act():
    module, policy = _lin(("a", "a"))
    program, plan = _native_program(module, policy)
    qlinears = [n for n in program.graph.nodes if n.target == torch.ops.atlas.qlinear.default]
    assert len(qlinears) == 2
    first = qlinears[0]
    qlinears[1].replace_input_with(qlinears[1].args[0], first.args[0])             # call 2 reads call 1's codes
    qlinears[1].replace_input_with(qlinears[1].args[1], first.args[1])
    with pytest.raises(fx_export.FXVerificationError, match="not fed by one atlas.quantize_act of its own"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8", unused=("b", "c"), calls={"a": 2})


# -- Conv2d ------------------------------------------------------------------------------------------------------------

class _Conv(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1)
        self.fc = torch.nn.Linear(4, 4)
        self.calls = calls

    def forward(self, x):
        for _ in range(self.calls):
            x = self.conv(x)
        return self.fc(x.mean(dim=(2, 3)))


@pytest.mark.parametrize("calls", [1, 2])
def test_conv2d_called_n_times_has_n_im2col_and_qconv2d(tmp_path, calls):
    torch.manual_seed(2)
    module = _Conv(calls).to(torch.bfloat16).eval()
    x = torch.randn(1, 4, 6, 6, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    fx_export.capture_fx(module, [(x,), (x * 3,)], tmp_path / "fx", policy=_policy(module), strict=True, fresh_process=False,
                         native_fp8="uint8")
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    (row,) = [r for r in inspection["weight_site_bindings"] if r["kind"] == "conv2d"]
    assert row["calls"] == calls and len(row["im2col_nodes"]) == calls
    targets = _targets(tmp_path / "fx")
    assert targets.count("atlas.qconv2d.default") == calls == inspection["native_fp8_conv2d_calls"]


def test_conv2d_im2col_count_must_match_the_call_count():
    torch.manual_seed(2)
    module = _Conv(2).to(torch.bfloat16).eval()
    policy = _policy(module)
    x = torch.randn(1, 4, 6, 6).to(torch.bfloat16)
    program, plan = _native_program(module, policy, case=(x,))
    fx_export.inspect_graph(program, module, policy, plan, "uint8", calls={"conv": 2, "fc": 1})
    with pytest.raises(fx_export.FXVerificationError, match="exactly one"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8", calls={"conv": 1, "fc": 1})


# -- attention sites: called 0, 1 and N times -----------------------------------------------------------------------------

mt = pytest.importorskip("lerobot.policies.multi_task_dit.modeling_multi_task_dit")
HIDDEN, HEADS, T = 128, 2, 24


class _Blocks(torch.nn.Module):
    """Two blocks, each a RoPEAttention site and a Linear; `order` names the blocks the boundary calls."""

    def __init__(self, order):
        super().__init__()
        self.a, self.b = (torch.nn.ModuleDict({"attn": mt.RoPEAttention(HIDDEN, HEADS), "proj": torch.nn.Linear(HIDDEN, HIDDEN)})
                          for _ in range(2))
        self.order = tuple(order)

    def forward(self, x):
        for name in self.order:
            block = getattr(self, name)
            x = x + block["proj"](block["attn"](x))
        return x


def _attn_model(order):
    import reference.ref_attention as ra

    torch.manual_seed(0)
    model = _Blocks(order).to(torch.bfloat16).eval()
    algo = algos.make(SPEC)
    report = algos.quantize(model, "ref_conv", [], dict(FAST), algo=algo)
    policy = quant_export.build_policy(model, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    assert {n for n, s in policy["sites"].items() if s["kind"] == "attention"} == {"a.attn", "b.attn"}
    assert ra  # patch_attention ran inside algos.quantize (the RoPEAttention sites exist in the policy)
    return model, policy


def _ax(seed):
    return torch.randn(1, T, HIDDEN, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


@pytest.mark.parametrize("strict,order,calls", [(True, ("a",), {"a": 1}), (False, ("a",), {"a": 1}),
                                                (True, ("a", "a"), {"a": 2}), (False, ("b", "a", "a"), {"a": 2, "b": 1})])
def test_attention_site_called_zero_one_and_n_times(tmp_path, strict, order, calls):
    model, policy = _attn_model(order)
    result = fx_export.capture_fx(model, [(_ax(1),), (_ax(2),)], tmp_path / "fx", policy=policy, strict=strict,
                                  fresh_process=False, native_fp8="uint8")
    assert result["status"] == "passed" and result["native_fp8_detail"]["outputs_identical_to_decode_graph"]
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    rows = {r["module"]: r for r in inspection["native_fp8_attention_modules"]}
    assert {m: r["pairs"] for m, r in rows.items()} == {f"{k}.attn": n for k, n in calls.items()}
    assert all(len(r["qk_matmul"]) == len(r["pv_matmul"]) == r["pairs"] and len(r["quantize_act"]) == 4 * r["pairs"]
               for r in rows.values())
    absent = sorted(f"{k}.attn" for k in "ab" if k not in calls)
    assert sorted(r["module"] for r in inspection["unused_attention_modules"]) == absent
    linears = ("attn.qkv_proj", "attn.out_proj", "proj")                                # three Linears per block call
    assert sorted(r["module"] for r in inspection["unused_quantized_modules"]) == \
        sorted(f"{k}.{m}" for k in "ab" if k not in calls for m in linears)
    assert {r["module"]: r["calls"] for r in inspection["weight_site_bindings"]} == \
        {f"{k}.{m}": n for k, n in calls.items() for m in linears}
    assert inspection["still_emulated"]["emulated_modules"] == {}                      # absent sites are not "emulated"
    assert result["native_fp8_detail"]["attention_modules"] == sorted(rows)
    targets = _targets(tmp_path / "fx")
    pairs = sum(calls.values())
    assert targets.count("atlas.qk_matmul.default") == targets.count("atlas.pv_matmul.default") == pairs
    assert targets.count("atlas.quantize_act.default") == pairs * 4 + pairs * 3            # q k p v + the three Linears
    assert not any("scaled_dot_product_attention" in t for t in targets)


def _attn_program(order):
    model, policy = _attn_model(order)
    case = (_ax(1),)
    import reference.ref_attention as ref_attention  # noqa: F401  (restored by _native_sites)
    with fx_export._meta_placeholders(model), fx_export._native_sites(model, policy, "uint8") as plan:
        stats = fx_export._observe_acts(model, [case, (_ax(2),)])
        attn_calls, unused = fx_export._attention_calls(stats, plan["attention"])
        plan["attention"] = {**plan["attention"], "paths": {p: s for p, s in plan["attention"]["paths"].items()
                                                            if p not in unused}}
        program = torch.export.export(model, case, strict=True)
    return model, policy, program, plan, attn_calls, unused


def test_attention_counts_are_taken_by_tag_and_checked_against_the_graph():
    model, policy, program, plan, attn_calls, unused = _attn_program(("a", "a"))
    assert attn_calls == {"a.attn": 2} and unused == {"b.attn": "b.attn"}
    lin = lambda block, n: {f"{block}.{m}": n for m in ("attn.qkv_proj", "attn.out_proj", "proj")}   # noqa: E731
    gone = tuple(lin("b", 0))
    calls = lin("a", 2)
    ok = fx_export.inspect_graph(program, model, policy, plan, "uint8", unused=gone, calls=calls,
                                 attn_calls=attn_calls, unused_attention=unused)
    assert [r["pairs"] for r in ok["native_fp8_attention_modules"]] == [2]
    with pytest.raises(fx_export.FXVerificationError, match="called it 3 time"):
        fx_export.inspect_graph(program, model, policy, plan, "uint8", unused=gone, calls=calls,
                                attn_calls={"a.attn": 3}, unused_attention=unused)
    with pytest.raises(fx_export.FXVerificationError, match="never called it"):
        fx_export.inspect_graph(program, model, policy, plan, "uint8", unused=gone, calls=calls,
                                attn_calls={}, unused_attention=unused)
    # an attention module declared absent whose quantize_act nodes are in the graph
    with pytest.raises(fx_export.FXVerificationError, match="declared never called"):
        fx_export.inspect_graph(program, model, policy, plan, "uint8", unused=gone, calls=calls,
                                attn_calls=attn_calls, unused_attention={"a.attn": "a.attn"})


def test_attention_operands_cast_a_different_number_of_times_are_refused():
    class Stats:
        sites = {}
        case_calls = [{"s:q": 2, "s:k": 2, "s:p": 1, "s:v": 2}]
    with pytest.raises(fx_export.FXVerificationError, match="operands q, k, p, v were cast"):
        fx_export._attention_calls(Stats, {"paths": {"m": "s"}, "sites": {}})
    Stats.case_calls = [{"s:q": 1, "s:k": 1, "s:p": 1, "s:v": 1}, {"s:q": 2, "s:k": 2, "s:p": 2, "s:v": 2}]
    with pytest.raises(fx_export.FXVerificationError, match="different number of times in different cases"):
        fx_export._attention_calls(Stats, {"paths": {"m": "s"}, "sites": {}})
