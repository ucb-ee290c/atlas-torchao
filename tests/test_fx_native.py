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
from test_ladder import FAST

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
    assert report["fresh_process_reload"]["model_replay_modules_imported"] == []
    detail = report["native_fp8_detail"]
    assert len(detail["sites"]) == 7 and detail["outputs_identical_to_decode_graph"]
    assert detail["native_graph_nodes"] < detail["decode_graph_nodes"]
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    rows = inspection["weight_site_bindings"]
    assert len(rows) == 7 and all(row["native"] for row in rows)
    emulated = inspection["still_emulated"]["emulated_modules"]
    # exactly the attention operand quantization remains emulated: Q.K and P.V contractions
    assert list(emulated) == ["model.model.layers.0.self_attn"]
    assert emulated["model.model.layers.0.self_attn"]["kind"] == "attention"
    assert len(emulated["model.model.layers.0.self_attn"]["contractions"]) == 2
    assert emulated["model.model.layers.0.self_attn"]["scale_derivation"]
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
    with fx_export._native_linears(module, policy, "uint8") as rules:
        def frozen(x):
            codes, scale = torch.ops.atlas.quantize_act(layer.frozen_input, "dynamic", "token", "nc", None, "uint8")
            return torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp, layer.bias,
                                           "bfloat16", "atlas_e4m3")
        layer.forward = frozen
        program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="does not depend on the graph input"):
        fx_export.inspect_graph(program, module, policy, rules, "uint8")


def test_wrong_activation_rule_is_refused_in_native_graph(linear_replay):
    module, policy, cases = linear_replay
    with fx_export._native_linears(module, policy, "uint8") as rules:
        program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="differ from the policy rule"):
        fx_export.inspect_graph(program, module, policy, {"0": ("dynamic", "tensor", "ref", None)}, "uint8")
    assert rules == {"0": ("dynamic", "token", "nc", None)}


def test_block_scaled_linear_stays_on_the_decode_path(tmp_path):
    module, policy, cases = _linear_replay("granularity:w=block32,a=block32,rule=nc", features=(64, 8))
    assert fx_export.native_candidates(module, policy) == {}
    with pytest.raises(fx_export.FXVerificationError, match="no Linear site can be carried natively"):
        fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                             native_fp8="uint8")
    # the default decode-based capture of the same layer is unaffected
    assert fx_export.capture_fx(module, cases, tmp_path / "ok", policy=policy,
                                fresh_process=False)["status"] == "passed"
