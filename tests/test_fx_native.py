"""--native-fp8 FX export: one atlas.qlinear node per quantized Linear (src/exporter/atlas_fx_ops.py)."""

import hashlib
import json
import subprocess
import sys
import textwrap

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers atlas::quantize_act / atlas::qlinear)
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
import pytest
import reference.ref_numerics as rn
import torch
import torch.nn.functional as F
from test_export import _quantize_export, _reload
from test_fx_export import PINNED_TORCH
from test_ladder import FAST

# Same guard as tests/test_fx_export.py: older torch cannot torch.export these captures.
pytestmark = pytest.mark.skipif(torch.__version__ < PINNED_TORCH,
                                reason=f"torch.export captures need the pinned torch {PINNED_TORCH} "
                                       f"(found {torch.__version__})")

STORAGES = ["uint8", "float8_e4m3fn"]
INPUTS = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
          (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _linear_replay(spec="granularity:w=channel,a=token,rule=nc", features=(8, 4)):
    torch.manual_seed(13)
    module = torch.nn.Sequential(torch.nn.Linear(*features)).to(torch.bfloat16).eval()
    algo = algos.make(spec)
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    first = torch.randn(2, 3, features[0], dtype=torch.bfloat16)
    second = first * torch.tensor([256.0, 2.0 ** -8], dtype=torch.bfloat16).reshape(2, 1, 1)
    return module, policy, [(first,), (second,)]


@pytest.fixture
def linear_replay():
    return _linear_replay()


# ---- the bitcast ----------------------------------------------------------------------------------

def test_float8_bitcast_agrees_with_ref_numerics_except_on_unreachable_codes():
    """torch.float8_e4m3fn vs ref_numerics.e4m3_unpack on all 254 valid codes (0x7F / 0xFF never occur).
    They agree on 240. They differ on exactly the 14 exponent-0 non-zero codes (subnormal in torch, signed
    zero in ref_numerics), which the encoder never emits: the 239 reachable codes all agree."""
    codes = torch.arange(256, dtype=torch.uint8)
    reference = rn.e4m3_unpack(codes).view(torch.int32)
    native = codes.view(torch.float8_e4m3fn).to(torch.float32).view(torch.int32)
    valid = [c for c in range(256) if c not in (0x7F, 0xFF)]
    assert len(valid) == 254
    differ = sorted(c for c in valid if reference[c] != native[c])
    assert differ == sorted(list(range(0x01, 0x08)) + list(range(0x81, 0x88)))
    assert sorted(c for c in range(256) if bool(fx_export._code_hazards(torch.tensor([c], dtype=torch.uint8)))) \
        == sorted(differ + [0x7F, 0xFF])
    assert max(codes.view(torch.float8_e4m3fn)[valid].to(torch.float32).tolist()) == 448.0
    assert [c for c in valid if (c >> 3) & 0xF == 15 and reference[c] != native[c]] == []
    assert rn.e4m3_unpack(torch.tensor([0x7E], dtype=torch.uint8)).item() == 448.0
    bf16 = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    reachable = set()
    for shift in range(-30, 30):
        reachable |= set(rn.e4m3_vpack(bf16, shift).unique().tolist())
    assert len(reachable) == 239 and not reachable & set(differ) and not reachable & {0x7F, 0xFF}
    assert all(reference[c] == native[c] for c in reachable)


# ---- the operators ----------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["granularity:w=channel,a=token,rule=nc", "granularity:w=tensor,a=tensor,rule=nc",
                                  "granularity:w=channel,a=token,rule=ref", "none"])
@pytest.mark.parametrize("storage", STORAGES)
def test_native_ops_equal_ref_forward_eagerly(spec, storage):
    module, policy, cases = _linear_replay(spec)
    layer = module[0]
    mode, reduce, rule, exp = fx_export.native_rule(policy["sites"]["0"], layer.in_features)
    with torch.no_grad():
        for (x,) in cases:
            codes, s = torch.ops.atlas.quantize_act(x, mode, reduce, rule, exp, storage)
            y = torch.ops.atlas.qlinear(codes, s, layer.weight_codes.view(fx_export.STORAGE_DTYPES[storage]),
                                        layer.weight_scale_exp, layer.bias, "bfloat16", atlas_fx_ops.WEIGHT_FORMAT)
            assert codes.dtype == fx_export.STORAGE_DTYPES[storage]
            assert torch.equal(y, layer(x))


def test_native_ops_have_fake_implementations():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2, 5, 8, dtype=torch.bfloat16)
        codes, s = torch.ops.atlas.quantize_act(x, "dynamic", "token", "nc", None, "uint8")
        assert codes.shape == (2, 5, 8) and codes.dtype == torch.uint8 and list(s.shape) == [2, 5, 1]
        codes, s = torch.ops.atlas.quantize_act(x, "static", "tensor", "ref", -3, "float8_e4m3fn")
        assert codes.dtype == torch.float8_e4m3fn and s.shape == ()
        y = torch.ops.atlas.qlinear(codes, s, torch.empty(4, 8, dtype=torch.uint8),
                                    torch.empty(4, 1, dtype=torch.int32), None, "bfloat16", "atlas_e4m3")
        assert y.shape == (2, 5, 4) and y.dtype == torch.bfloat16


def test_rescale_after_contraction_equals_operand_scaling():
    """The policy's output_rescale (normalized operands, then 2^(sx+sw) in float32, then the layer dtype)
    gives the same bytes as the operator's operand scaling, for per-token x per-channel exponents."""
    from algos import base as algo_hooks

    module, policy, cases = _linear_replay("granularity:w=channel,a=token,rule=nc", features=(96, 40))
    layer = module[0]
    with torch.no_grad():
        for (x,) in cases:
            codes, s = torch.ops.atlas.quantize_act(x, "dynamic", "token", "nc", None, "uint8")
            y = torch.ops.atlas.qlinear(codes, s, layer.weight_codes, layer.weight_scale_exp, None, "bfloat16",
                                        "atlas_e4m3")
            unscaled = F.linear(rn.e4m3_unpack(codes).to(torch.bfloat16),
                                rn.e4m3_unpack(layer.weight_codes).to(torch.bfloat16))
            rescaled = (unscaled * algo_hooks.output_scale(s, layer.weight_scale_exp)).to(torch.bfloat16)
            assert torch.equal(y, rescaled)


# ---- the capture --------------------------------------------------------------------------------------

def _hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("storage", STORAGES)
def test_linear_capture_is_one_node_with_exact_buffers(tmp_path, linear_replay, storage):
    module, policy, cases = linear_replay
    layer = module[0]
    codes, exp, forward = layer.weight_codes, layer.weight_scale_exp, layer.forward
    uint8_hash = _hash(codes)
    report = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=True,
                                  native_fp8=storage)
    assert report["status"] == "passed" and report["native_fp8"] == storage
    assert report["fresh_process_reload"]["atlas_qlinear_nodes"] == 1
    assert report["fresh_process_reload"]["model_replay_modules_imported"] == []
    assert report["native_fp8_detail"]["outputs_identical_to_decode_graph"]
    assert report["same_process_reload"]["all_outputs_byte_identical"]
    # (iii) inventory: float8 dtype, hash equal to the hash of the original uint8 bytes
    inventory = json.loads((tmp_path / "fx" / "buffer_inventory.json").read_text())
    assert inventory["0.weight_codes"]["dtype"] == "torch." + storage
    assert inventory["0.weight_codes"]["sha256"] == uint8_hash
    assert inventory["0.weight_scale_exp"]["dtype"] == "torch.int32"
    # (ii) one qlinear consuming the buffers; no decode/contraction ops
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    nodes = [n for n in program.graph.nodes if n.op == "call_function"]
    targets = [str(n.target) for n in nodes]
    assert targets.count("atlas.qlinear.default") == 1 and targets.count("atlas.quantize_act.default") == 1
    # the only aten ops left are the input dtype guard/cast; no exp2/mul/convert/linear/frexp decode
    assert sorted(t for t in targets if t.startswith("aten.")) == ["aten._assert_tensor_metadata.default",
                                                                  "aten.to.dtype"]
    buffers = dict(program.named_buffers())
    assert buffers["0.weight_codes"].dtype == fx_export.STORAGE_DTYPES[storage]
    assert torch.equal(buffers["0.weight_codes"].view(torch.uint8), codes)
    assert torch.equal(buffers["0.weight_scale_exp"], exp)
    qlinear = next(n for n in nodes if n.target == torch.ops.atlas.qlinear.default)
    placeholders = {n.name: n.target for n in program.graph.nodes if n.op == "placeholder"}
    assert placeholders[qlinear.args[2].name] == "b_0_weight_codes"
    assert placeholders[qlinear.args[3].name] == "b_0_weight_scale_exp"
    # (i) outputs: eager replay, decode-based graph and native graph agree byte for byte on both inputs
    decode_graph = torch.export.export(module, cases[0], strict=True)
    with torch.no_grad():
        for args in cases:
            expected = module(*args)
            assert torch.equal(program.module()(*args), expected)
            assert torch.equal(decode_graph.module()(*args), expected)
    # the caller's module is restored
    assert layer.weight_codes is codes and layer.weight_codes.dtype == torch.uint8
    assert "forward" not in layer.__dict__ and layer.forward == forward


def test_default_capture_is_unchanged_by_the_option(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    report = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False)
    assert report["native_fp8"] is None and "native_fp8_detail" not in report
    manifest = json.loads((tmp_path / "fx" / "fx_manifest.json").read_text())
    assert "native_fp8" not in manifest
    targets = [str(n.target) for n in torch.export.load(tmp_path / "fx" / "model.pt2").graph.nodes
               if n.op == "call_function"]
    assert "aten.linear.default" in targets and not any("atlas" in t for t in targets)


@pytest.mark.parametrize("spec", ["none", "atlas_v1_awq", "atlas_v1_static"])
@pytest.mark.parametrize("storage", ["uint8"])
def test_tiny_llama_native_capture(tmp_path, spec, storage):
    eager, directory, _, options = _quantize_export(tmp_path, spec, layers=1)
    loaded = _reload(directory, "ref_conv", options, layers=1)
    policy = json.loads((directory / "policy.json").read_text())
    wrapped = fx_export.CausalLogits(loaded).eval()
    with torch.no_grad():
        for (ids,) in INPUTS:
            assert torch.equal(eager(ids, use_cache=False).logits, wrapped(ids))
    report = fx_export.capture_fx(wrapped, INPUTS, tmp_path / "fx", policy=policy, fresh_process=True,
                                  native_fp8=storage)
    assert report["status"] == "passed" and report["weight_sites_verified"] == 7
    assert report["fresh_process_reload"]["atlas_qlinear_nodes"] == 7
    assert report["fresh_process_reload"]["atlas_qk_matmul_nodes"] == 1
    assert report["fresh_process_reload"]["atlas_pv_matmul_nodes"] == 1
    assert report["fresh_process_reload"]["model_replay_modules_imported"] == []
    detail = report["native_fp8_detail"]
    assert len(detail["sites"]) == 7 and detail["outputs_identical_to_decode_graph"]
    assert detail["native_graph_nodes"] < detail["decode_graph_nodes"]
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    rows = inspection["weight_site_bindings"]
    assert len(rows) == 7 and all(row["native"] for row in rows)
    # attention is native too: nothing is left emulated; Q.K and P.V are atlas ops around BF16 softmax nodes
    assert inspection["still_emulated"]["emulated_modules"] == {}
    (attention,) = inspection["native_fp8_attention_modules"]
    assert attention["module"] == "model.model.layers.0.self_attn" and attention["pairs"] == 1
    assert len(attention["quantize_act"]) == 4
    assert {"aten._softmax.default", "aten.softmax.int"} & set(attention["remaining_nodes_in_module_by_target"])
    assert not any(t.endswith(("frexp.Tensor", "exp2.default", "matmul.default", "bitwise_and.Tensor"))
                   for t in attention["remaining_nodes_in_module_by_target"])
    assert [row["site"] for row in inspection["bf16_by_policy"]] == ["lm_head"]
    assert inspection["bf16_by_policy"][0]["weight_dtype"] == "bfloat16"
    # outputs against eager and the saved decode graph on both inputs
    saved = torch.export.load(tmp_path / "fx" / "model.pt2").module()
    with torch.no_grad():
        for (ids,) in INPUTS:
            assert torch.equal(saved(ids), eager(ids, use_cache=False).logits)


def test_loading_needs_the_ops_registered(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False, native_fp8="uint8")
    script = textwrap.dedent("""
        import sys, torch
        try:
            torch.export.load(sys.argv[1])
        except Exception as error:
            print("FAILED", type(error).__name__)
            sys.exit(0)
        print("LOADED")
    """)
    bare = subprocess.run([sys.executable, "-I", "-c", script, str(tmp_path / "fx" / "model.pt2")],
                          capture_output=True, text=True)
    assert bare.stdout.strip().startswith("FAILED"), bare.stdout + bare.stderr
    fx_export.verify_saved(tmp_path / "fx")


def test_float8_storage_refuses_codes_torch_decodes_differently(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    with torch.no_grad():
        module[0].weight_codes[0, 0] = 0x03             # exponent-0 non-zero: torch subnormal, ref zero
    with pytest.raises(fx_export.FXVerificationError, match="decode"):
        fx_export.capture_fx(module, cases, tmp_path / "f8", policy=policy, fresh_process=False,
                             native_fp8="float8_e4m3fn")
    # uint8 storage keeps the reference decode, so the graph still equals eager replay
    report = fx_export.capture_fx(module, cases, tmp_path / "u8", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert report["status"] == "passed"
    assert module[0].weight_codes.dtype == torch.uint8


@pytest.mark.parametrize("buffer_name", ["weight_codes", "weight_scale_exp"])
def test_native_graph_reads_the_code_and_exponent_buffers(tmp_path, linear_replay, buffer_name):
    module, policy, cases = linear_replay
    fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False, native_fp8="uint8")
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    with torch.no_grad():
        baseline = program.module()(*cases[0]).clone()
        buffer = dict(program.named_buffers())["0." + buffer_name]
        buffer.zero_() if buffer_name == "weight_codes" else buffer.add_(2)
        assert not torch.equal(baseline, program.module()(*cases[0]))
        getattr(module[0], buffer_name).copy_(buffer)
        assert torch.equal(program.module()(*cases[0]), module(*cases[0]))


def test_frozen_activation_input_is_refused_in_native_graph(linear_replay):
    module, policy, cases = linear_replay
    layer = module[0]
    layer.register_buffer("frozen_input", cases[0][0].clone())
    with fx_export._native_sites(module, policy, "uint8") as plan:
        def frozen(x):
            codes, scale = torch.ops.atlas.quantize_act(layer.frozen_input, "dynamic", "token", "nc", None, "uint8")
            return torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp, layer.bias,
                                           "bfloat16", "atlas_e4m3")
        layer.forward = frozen
        program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="does not depend on the graph input"):
        fx_export.inspect_graph(program, module, policy, plan, "uint8")


def test_wrong_activation_rule_is_refused_in_native_graph(linear_replay):
    module, policy, cases = linear_replay
    with fx_export._native_sites(module, policy, "uint8") as plan:
        program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="differ from the policy rule"):
        fx_export.inspect_graph(program, module, policy, {**plan, "linear": {"0": ("dynamic", "tensor", "ref", None)}}, "uint8")
    assert plan["linear"] == {"0": ("dynamic", "token", "nc", None)}


def test_block_scaled_linear_stays_on_the_decode_path(tmp_path):
    module, policy, cases = _linear_replay("granularity:w=block32,a=block32,rule=nc", features=(64, 8))
    assert fx_export.native_candidates(module, policy) == {}
    with pytest.raises(fx_export.FXVerificationError, match="no Linear site can be carried natively"):
        fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                             native_fp8="uint8")
    # the default decode-based capture of the same layer is unaffected
    assert fx_export.capture_fx(module, cases, tmp_path / "ok", policy=policy,
                                fresh_process=False)["status"] == "passed"


# ---- attention -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("reduce", ["head", "token"])
def test_attention_ops_equal_ref_attention_mm_and_cast(reduce):
    import reference.ref_attention as ra

    torch.manual_seed(3)
    q = torch.randn(2, 4, 5, 16, dtype=torch.bfloat16)
    k = torch.randn(2, 4, 7, 16, dtype=torch.bfloat16)
    p = torch.softmax(torch.randn(2, 4, 5, 7), dim=-1).to(torch.bfloat16)
    v = torch.randn(2, 4, 7, 16, dtype=torch.bfloat16)
    dims = {"head": (-2, -1), "token": (-1,)}[reduce]
    with torch.no_grad():
        q_q, s_q = torch.ops.atlas.quantize_act(q, "dynamic", reduce, "nc", None, "uint8")
        assert torch.equal(s_q, __import__("algos.granularity", fromlist=["x"]).scale_exp(q, dims, "nc"))
        assert list(s_q.shape) == ([2, 4, 1, 1] if reduce == "head" else [2, 4, 5, 1])
        k_q, s_k = torch.ops.atlas.quantize_act(k, "dynamic", "head", "nc", None, "uint8")
        assert torch.equal(torch.ops.atlas.qk_matmul(q_q, s_q, k_q, s_k), ra._mm(None, q_q, s_q, k_q, s_k))
        p_q, s_p = torch.ops.atlas.quantize_act(p, "dynamic", reduce, "ref", None, "float8_e4m3fn")
        v_q, s_v = torch.ops.atlas.quantize_act(v, "dynamic", "head", "ref", None, "float8_e4m3fn")
        vt = v_q.transpose(-1, -2)
        assert torch.equal(torch.ops.atlas.pv_matmul(p_q, s_p, vt, s_v), ra._mm(None, p_q, s_p, vt, s_v))
        # the reference cast is the same code path as rn.quantize with dims
        ref_q, ref_s = rn.quantize(q, (-2, -1))
        head_q, head_s = torch.ops.atlas.quantize_act(q, "dynamic", "head", "ref", None, "uint8")
        assert torch.equal(ref_q, head_q) and torch.equal(ref_s, head_s)


def test_attention_native_ops_have_fake_implementations():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        q = torch.empty(2, 4, 5, 16, dtype=torch.bfloat16)
        codes, s = torch.ops.atlas.quantize_act(q, "dynamic", "head", "nc", None, "uint8")
        assert list(s.shape) == [2, 4, 1, 1]
        scores = torch.ops.atlas.qk_matmul(codes, s, torch.empty(2, 4, 7, 16, dtype=torch.uint8), s)
        assert scores.shape == (2, 4, 5, 7) and scores.dtype == torch.float32
        out = torch.ops.atlas.pv_matmul(torch.empty(2, 4, 5, 7, dtype=torch.uint8), s,
                                        torch.empty(2, 4, 16, 7, dtype=torch.uint8), s)
        assert out.shape == (2, 4, 5, 16)


def _tiny_native(tmp_path, spec, storage="uint8", fresh=True):
    eager, directory, _, options = _quantize_export(tmp_path, spec, layers=1)
    loaded = fx_export.CausalLogits(_reload(directory, "ref_conv", options, layers=1)).eval()
    policy = json.loads((directory / "policy.json").read_text())
    report = fx_export.capture_fx(loaded, INPUTS, tmp_path / "fx", policy=policy, fresh_process=fresh,
                                  native_fp8=storage)
    return eager, report, json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())


def test_token_attention_rule_and_float8_storage_are_native(tmp_path):
    eager, report, inspection = _tiny_native(tmp_path, "granularity:w=channel,a=token,rule=nc,attn=token",
                                             "float8_e4m3fn")
    (attention,) = inspection["native_fp8_attention_modules"]
    assert {op: rule["reduce"] for op, rule in attention["operand_rules"].items()} == \
        {"q": "token", "k": "head", "p": "token", "v": "head"}
    assert report["native_fp8_detail"]["outputs_identical_to_decode_graph"]
    saved = torch.export.load(tmp_path / "fx" / "model.pt2").module()
    with torch.no_grad():
        for (ids,) in INPUTS:
            assert torch.equal(saved(ids), eager(ids, use_cache=False).logits)
    targets = [str(n.target) for n in torch.export.load(tmp_path / "fx" / "model.pt2").graph.nodes
               if n.op == "call_function"]
    assert targets.count("atlas.qk_matmul.default") == 1 and targets.count("atlas.pv_matmul.default") == 1
    assert targets.count("atlas.quantize_act.default") == 7 + 4


def test_attention_softmax_stays_a_visible_node_between_the_matmuls(tmp_path):
    _, _, inspection = _tiny_native(tmp_path, "atlas_v1_awq", fresh=False)
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    nodes = list(program.graph.nodes)
    qk = next(n for n in nodes if n.target == torch.ops.atlas.qk_matmul.default)
    pv = next(n for n in nodes if n.target == torch.ops.atlas.pv_matmul.default)
    softmax = [n for n in nodes if "softmax" in str(n.target)]
    assert len(softmax) == 1 and nodes.index(qk) < nodes.index(softmax[0]) < nodes.index(pv)
    assert qk in fx_export._ancestors(softmax[0]) and softmax[0] in fx_export._ancestors(pv)
    remaining = inspection["native_fp8_attention_modules"][0]["remaining_nodes_in_module_by_target"]
    assert remaining["aten.softmax.int"] == 1 and "aten.frexp.Tensor" not in remaining


def test_attention_native_refuses_frozen_operand_and_wrong_rule(tmp_path):
    eager, directory, _, options = _quantize_export(tmp_path, "atlas_v1_awq", layers=1)
    wrapped = fx_export.CausalLogits(_reload(directory, "ref_conv", options, layers=1)).eval()
    policy = json.loads((directory / "policy.json").read_text())
    import reference.ref_attention as ra

    with fx_export._meta_placeholders(wrapped), fx_export._native_sites(wrapped, policy, "uint8") as plan:
        program = torch.export.export(wrapped, INPUTS[0], strict=True)
    fx_export.inspect_graph(program, wrapped, policy, plan, "uint8")          # the genuine capture passes
    wrong = {**plan, "attention": {**plan["attention"], "sites": {
        site: {**rules, "p": ("dynamic", "token", "nc", None)} for site, rules in plan["attention"]["sites"].items()}}}
    with pytest.raises(fx_export.FXVerificationError, match="differ from the policy rule"):
        fx_export.inspect_graph(program, wrapped, policy, wrong, "uint8")
    original = ra._cast
    with fx_export._meta_placeholders(wrapped), fx_export._native_sites(wrapped, policy, "uint8") as plan:
        native_cast = ra._cast
        frozen = torch.randn(1, 4, 8, 16, dtype=torch.bfloat16)

        def cast_frozen_k(algo, site, operand, x, dims):
            return native_cast(algo, site, operand, frozen if operand == "k" else x, dims)
        ra._cast = cast_frozen_k
        program = torch.export.export(wrapped, INPUTS[0], strict=True)
    assert ra._cast is original
    # a constant tensor is not a graph input, so the k exponent no longer depends on the input
    with pytest.raises(fx_export.FXVerificationError, match="does not depend on the graph input"):
        fx_export.inspect_graph(program, wrapped, policy, plan, "uint8")


def test_native_capture_restores_ref_attention_functions(tmp_path):
    import reference.ref_attention as ra

    cast, mm = ra._cast, ra._mm
    _tiny_native(tmp_path, "none", fresh=False)
    assert ra._cast is cast and ra._mm is mm


# ---- Conv2d and BF16-by-policy sites --------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["granularity:w=channel,a=token,rule=nc", "none"])
def test_conv2d_native_capture(tmp_path, spec):
    from test_ladder import ConvNet

    torch.manual_seed(0)
    module = ConvNet().to(torch.bfloat16).eval()
    algo = algos.make(spec)
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    x = torch.randn(2, 3, 8, 8, dtype=torch.bfloat16)
    cases = [(x,), (x * 4,)]
    result = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=True,
                                  native_fp8="uint8")
    assert result["status"] == "passed" and result["fresh_process_reload"]["atlas_qconv2d_nodes"] == 1
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert inspection["native_fp8_conv2d_sites"] == 1 and inspection["native_fp8_linear_sites"] == 1
    conv_row = next(row for row in inspection["weight_site_bindings"] if row["kind"] == "conv2d")
    assert len(conv_row["im2col_nodes"]) == 1 and conv_row["code_dtype"] == "uint8"
    assert inspection["still_emulated"]["emulated_modules"] == {}
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert targets.count("atlas.qconv2d.default") == 1 and targets.count("aten.im2col.default") == 1
    codes = dict(program.named_buffers())["conv.weight_codes"]
    assert codes.shape == (8, 27) and codes.dtype == torch.uint8            # flattened [out, in*kh*kw]
    decode = torch.export.export(module, cases[0], strict=True)
    with torch.no_grad():
        for args in cases:
            assert torch.equal(program.module()(*args), module(*args))
            assert torch.equal(program.module()(*args), decode.module()(*args))


def test_conv2d_geometry_mismatch_keeps_the_decode_path(tmp_path):
    from test_ladder import ConvNet

    torch.manual_seed(0)
    module = ConvNet().to(torch.bfloat16).eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    policy["sites"]["conv"]["conv"]["stride"] = [2, 2]
    assert list(fx_export.native_candidates(module, policy)) == ["fc"]


def test_dit_patch_embedding_exception_stays_a_visible_bf16_conv(tmp_path):
    from test_algo_atlas_v1 import DIT_PATCH, TinyDiT, dit_calib

    torch.manual_seed(21)
    module = TinyDiT()
    algo = algos.make("atlas_v1+keep_bf16:rule=input_conv")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo, calib=dit_calib())
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [],
                                       report["algo"]["bf16_opt_out"], {}, {})
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
    result = fx_export.capture_fx(module, [(x,), (x * 3,)], tmp_path / "fx", policy=policy,
                                  fresh_process=True, native_fp8="uint8")
    assert result["status"] == "passed"
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    (row,) = inspection["bf16_by_policy"]
    assert row["site"] == DIT_PATCH and row["status"] == "host" and row["in_graph"]
    assert row["weight_dtype"] == "bfloat16" and row["contraction_nodes"]
    assert inspection["native_fp8_conv2d_sites"] == 1 and inspection["native_fp8_linear_sites"] == 1
    assert row["module"] not in {r["module"] for r in inspection["weight_site_bindings"]}
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    patch_nodes = [n for n in program.graph.nodes if n.op == "call_function"
                   and list((n.meta.get("nn_module_stack") or {"": ("",)}).values())[-1][0] == DIT_PATCH]
    assert [str(n.target) for n in patch_nodes if "atlas" in str(n.target)] == []
    assert any("conv" in str(n.target) for n in patch_nodes)
    assert not program.state_dict[DIT_PATCH + ".weight"].is_meta
    with torch.no_grad():
        assert torch.equal(program.module()(x), module(x))


def test_default_capture_also_reports_bf16_by_policy(tmp_path):
    eager, directory, _, options = _quantize_export(tmp_path, "atlas_v1_awq", layers=1)
    wrapped = fx_export.CausalLogits(_reload(directory, "ref_conv", options, layers=1)).eval()
    policy = json.loads((directory / "policy.json").read_text())
    fx_export.capture_fx(wrapped, INPUTS, tmp_path / "fx", policy=policy, fresh_process=False)
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert [(r["site"], r["status"]) for r in inspection["bf16_by_policy"]] == [("lm_head", "skipped")]
    assert "still_emulated" not in inspection


class _OneOfTwo(torch.nn.Module):
    """A quantized layer the boundary never calls (like SmolVLA's VLM lm_head)."""

    def __init__(self):
        super().__init__()
        self.used = torch.nn.Linear(8, 4)
        self.idle = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.used(x)


@pytest.mark.parametrize("native", [None, "uint8"])
def test_quantized_layer_the_boundary_never_calls_is_recorded(tmp_path, native):
    torch.manual_seed(5)
    module = _OneOfTwo().to(torch.bfloat16).eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    result = fx_export.capture_fx(module, [(x,), (x * 2,)], tmp_path / "fx", policy=policy, fresh_process=True,
                                  native_fp8=native)
    assert result["status"] == "passed" and result["unused_quantized_modules"] == ["idle"]
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert inspection["unused_quantized_modules"] == [{"site": "idle", "module": "idle"}]
    assert inspection["weight_sites_verified"] == 1
    assert ("still_emulated" not in inspection) == (native is None)
    if native:
        assert inspection["still_emulated"]["emulated_modules"] == {}      # an idle layer is not "emulated"
    inventory = json.loads((tmp_path / "fx" / "buffer_inventory.json").read_text())
    assert sorted(inventory) == ["used.weight_codes", "used.weight_scale_exp"]


class _InPlaceRotate(torch.nn.Module):
    """SmolVLA's apply_rope fills `empty_like(x)` through slice assignments: the data flow is real but there is
    no FX edge from x to the buffer's later uses."""

    def __init__(self, features, frozen=False):
        super().__init__()
        self.proj = torch.nn.Linear(features, 4)
        self.frozen = frozen
        self.register_buffer("constant", torch.ones(2, features, dtype=torch.bfloat16))

    def forward(self, x):
        half = x.shape[-1] // 2
        source = self.constant if self.frozen else x
        res = torch.empty_like(x)
        res[..., :half] = source[..., half:]
        res[..., half:] = source[..., :half]
        return self.proj(res)


def _in_place_capture(tmp_path, frozen):
    torch.manual_seed(7)
    module = _InPlaceRotate(8, frozen).to(torch.bfloat16).eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    return fx_export.capture_fx(module, [(x,), (x * 2,)], tmp_path / "fx", policy=policy, fresh_process=False,
                                native_fp8="uint8")


def test_in_place_filled_buffer_counts_as_input_dependent(tmp_path):
    assert _in_place_capture(tmp_path, frozen=False)["status"] == "passed"


def test_in_place_filled_buffer_from_a_constant_is_still_refused(tmp_path):
    with pytest.raises(fx_export.FXVerificationError, match="does not depend on the graph input"):
        _in_place_capture(tmp_path, frozen=True)
