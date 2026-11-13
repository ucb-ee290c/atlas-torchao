"""FX capture details the backend intake depends on: a BF16-by-policy convolution with string padding
(aten.conv2d.padding, as SigLIP's "valid" patch embedding exports) is recognized as a contraction, and the
native Linear / Conv2d forwards emit no identity aten.to node when the input already has the weight dtype."""

import json

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers the atlas ops)
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
import pytest
import torch
from test_fx_export import PINNED_TORCH
from test_ladder import FAST

pytestmark = pytest.mark.skipif(torch.__version__ < PINNED_TORCH,
                                reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")


class _ValidPatchNet(torch.nn.Module):
    """An RGB patch embedding with padding="valid" (exports as aten.conv2d.padding), then a Linear."""

    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 8, 2, stride=2, padding="valid")
        self.fc = torch.nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.patch(x).flatten(2).transpose(1, 2))


def _quantized(module, spec):
    algo = algos.make(spec)
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [],
                                       report["algo"].get("bf16_opt_out", []), {}, {})
    return policy


def _valid_patch():
    torch.manual_seed(5)
    module = _ValidPatchNet().to(torch.bfloat16).eval()
    policy = _quantized(module, "granularity:w=channel,a=token,rule=nc+keep_bf16:rule=input_conv")
    assert policy["sites"]["patch"]["status"] == "host"
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
    return module, policy, [(x,), (x * 3,)]


def test_string_padding_conv_is_a_bf16_by_policy_contraction(tmp_path):
    module, policy, cases = _valid_patch()
    result = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert result["status"] == "passed"
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    (row,) = inspection["bf16_by_policy"]
    assert row["site"] == "patch" and row["in_graph"] and row["weight_dtype"] == "bfloat16"
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    nodes = {n.name: str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    assert [nodes[name] for name in row["contraction_nodes"]] == ["aten.conv2d.padding"]


def test_string_padding_conv_without_the_overload_is_refused(tmp_path, monkeypatch):
    """The SmolVLA keep_bf16 failure: without aten.conv2d.padding the host site has no contraction."""
    module, policy, cases = _valid_patch()
    monkeypatch.delitem(fx_export.CONTRACTIONS, torch.ops.aten.conv2d.padding)
    with pytest.raises(fx_export.FXVerificationError, match="lacks its BF16 contraction or weight"):
        fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                             native_fp8="uint8")


def _quantize_act_input_targets(program):
    return [str(n.args[0].target) for n in program.graph.nodes
            if n.op == "call_function" and n.target == torch.ops.atlas.quantize_act.default]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_linear_has_no_identity_cast(tmp_path, dtype):
    """Input already in the weight dtype (f32 like SmolVLA's action-time MLP, or BF16): quantize_act reads the
    input directly, no aten.to node; outputs byte-identical to the eager replay."""
    torch.manual_seed(13)
    module = torch.nn.Sequential(torch.nn.Linear(8, 4)).to(dtype).eval()
    policy = _quantized(module, "granularity:w=channel,a=token,rule=nc")
    first = torch.randn(2, 3, 8, dtype=dtype)
    cases = [(first,), (first * 4,)]
    result = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert result["status"] == "passed" and result["native_fp8_detail"]["outputs_identical_to_decode_graph"]
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert "aten.to.dtype" not in targets and "aten._assert_tensor_metadata.default" not in targets
    assert [n.args[0].op for n in program.graph.nodes
            if n.target == torch.ops.atlas.quantize_act.default] == ["placeholder"]
    with torch.no_grad():
        for args in cases:
            assert torch.equal(program.module()(*args), module(*args))


def test_native_linear_keeps_a_real_cast(tmp_path):
    """A BF16 Linear fed f32 still casts (one aten.to.dtype into quantize_act), as RefQuantLinear does."""
    torch.manual_seed(13)
    module = torch.nn.Sequential(torch.nn.Linear(8, 4)).to(torch.bfloat16).eval()
    policy = _quantized(module, "granularity:w=channel,a=token,rule=nc")
    first = torch.randn(2, 3, 8, dtype=torch.float32)
    cases = [(first,), (first * 4,)]
    result = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert result["status"] == "passed"
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    assert _quantize_act_input_targets(program) == ["aten.to.dtype"]
    with torch.no_grad():
        for args in cases:
            assert torch.equal(program.module()(*args), module(*args))
