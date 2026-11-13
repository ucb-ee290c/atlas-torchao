"""Backend intake of saved native-FP8 FX bundles: the CausalLayer boundary, non-persistent buffers held as graph
constants, real-file export directories, and the m2m capture-worker loader's helpers (src/exporter/backend_loader.py)."""

import json
import os

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import exporter.backend_loader as backend_loader
import exporter.fx_export as fx_export
from test_export import _quantize_export, _reload
from test_fx_export import PINNED_TORCH

pytestmark = pytest.mark.skipif(torch.__version__ < PINNED_TORCH,
                                reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")


@pytest.fixture(scope="module")
def two_layer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("two_layer")
    _, directory, _, options = _quantize_export(tmp, "atlas_v1", layers=2)
    loaded = _reload(directory, "ref_conv", options, layers=2)
    return loaded, directory


def _input_kinds(program):
    return {str(spec.target): spec.kind.name for spec in program.graph_signature.input_specs if spec.target}


def test_causal_layer_cases_reproduce_the_models_own_layer(two_layer):
    model, _ = two_layer
    cases = fx_export.causal_layer_cases(model, 0, 8, 0, "cpu")
    assert len(cases) == 2 and not torch.equal(cases[0][0], cases[1][0])
    hidden, cos, sin, mask = cases[0]
    assert all(t.dtype == torch.bfloat16 for t in cases[0])
    assert mask.shape == (1, 1, 8, 8) and torch.all(mask.tril() == 0)
    assert torch.all(mask.triu(1)[0, 0][torch.ones(8, 8, dtype=torch.bool).triu(1)] == torch.finfo(torch.bfloat16).min)
    boundary = fx_export.CausalLayer(model, 0).eval()
    ids = torch.randint(model.config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = model(ids, use_cache=False, output_hidden_states=True).hidden_states[1]
        assert torch.equal(boundary(hidden, cos, sin, mask), expected)


def test_causal_layer_native_capture_keeps_source_names_and_scope(tmp_path, two_layer):
    model, export_dir = two_layer
    policy = json.loads((export_dir / "policy.json").read_text())
    boundary = fx_export.CausalLayer(model, 1).eval()
    cases = fx_export.causal_layer_cases(model, 1, 8, 0, "cpu")
    report = fx_export.capture_fx(boundary, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert report["status"] == "passed" and report["weight_sites_verified"] == 7
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    assert inspection["policy_site_scope"] == ["model.layers.1."]
    assert inspection["sites_outside_boundary"] == 7
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    names = list(program.state_dict) + list(program.constants)
    assert names and all(n.startswith("model.model.layers.1.") for n in names)
    assert "model.model.layers.1.mlp.down_proj.weight_codes" in program.state_dict
    assert "model.model.layers.1.mlp.down_proj.weight_codes" in dict(program.named_buffers())
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert targets.count("atlas.qlinear.default") == 7
    assert "atlas.qk_matmul.default" in targets and "atlas.pv_matmul.default" in targets
    with torch.no_grad():
        for case in cases:
            assert torch.equal(program.module()(*case), boundary(*case))


class _CachedRotary(torch.nn.Module):
    """A constructor-built non-persistent cache read in forward (the shape of the Multi-Task DiT RoPE)."""

    def __init__(self):
        super().__init__()
        self.register_buffer("_cos_cached", torch.linspace(-1, 1, 8, dtype=torch.bfloat16), persistent=False)

    def forward(self, x):
        return x * self._cos_cached


def test_nonpersistent_buffers_are_captured_as_constants_and_restored(tmp_path, linear_replay_with_cache):
    module, policy, cases = linear_replay_with_cache
    cache = module[0]._cos_cached
    report = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    assert report["status"] == "passed"
    assert report["capture_buffer_roles"] == {"made_persistent": 2, "held_as_constants": ["0._cos_cached"]}
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    kinds = _input_kinds(program)
    assert kinds["0._cos_cached"] == "CONSTANT_TENSOR"
    assert kinds["1.weight_codes"] == "BUFFER" and kinds["1.weight_scale_exp"] == "BUFFER"
    assert "0._cos_cached" not in dict(program.named_buffers())
    assert torch.equal(program.constants["0._cos_cached"], cache)
    # export tensors are persistent state of the saved program (read from state_dict by the backend reference)
    assert torch.equal(program.state_dict["1.weight_codes"], module[1].weight_codes)
    assert torch.equal(program.state_dict["1.weight_scale_exp"], module[1].weight_scale_exp)
    # the caller's module is unchanged: same non-persistent buffers, same tensors
    assert module[0]._buffers["_cos_cached"] is cache and "_cos_cached" in module[0]._non_persistent_buffers_set
    assert {"weight_codes", "weight_scale_exp"} <= module[1]._non_persistent_buffers_set
    assert "1.weight_codes" not in module.state_dict()
    with torch.no_grad():
        for case in cases:
            assert torch.equal(program.module()(*case), module(*case))


@pytest.fixture
def linear_replay_with_cache():
    import algos
    import exporter.quant_export as quant_export
    from test_ladder import FAST

    torch.manual_seed(13)
    module = torch.nn.Sequential(_CachedRotary(), torch.nn.Linear(8, 4)).to(torch.bfloat16).eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    x = torch.randn(2, 8, generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)
    return module, policy, [(x,), (x * 4,)]


def test_link_export_places_real_files(tmp_path):
    source = tmp_path / "export"
    source.mkdir()
    for name in fx_export.EXPORT_FILES:
        (source / name).write_bytes(name.encode())
    dest = fx_export.link_export(source, tmp_path / "bundle" / "export")
    for name in fx_export.EXPORT_FILES:
        assert not (dest / name).is_symlink() and (dest / name).read_bytes() == name.encode()
    assert fx_export.link_export(source, dest) == dest          # idempotent on identical files
    (dest / "policy.json").unlink()
    (dest / "policy.json").write_bytes(b"other")
    with pytest.raises(fx_export.FXVerificationError, match="differs"):
        fx_export.link_export(source, dest)
    os.symlink(source, tmp_path / "linked")
    with pytest.raises(fx_export.FXVerificationError, match="symlink"):
        fx_export.link_export(source, tmp_path / "linked")


def _fake_bundle(tmp_path, inputs, outputs):
    fx = tmp_path / "bundle" / "fx"
    fx.mkdir(parents=True)
    fixtures = {f"case_0.input_{i}": t for i, t in enumerate(inputs)}
    fixtures.update({f"case_0.output_{i}": t for i, t in enumerate(outputs)})
    save_file(fixtures, str(fx / "fixtures.safetensors"))
    (fx / "parity.json").write_text(json.dumps({"cases": [{
        "inputs": [f"case_0.input_{i}" for i in range(len(inputs))],
        "outputs": [f"case_0.output_{i}" for i in range(len(outputs))]}]}))
    (fx / "model.pt2").write_bytes(b"program")
    files = {name: fx_export._file_record(fx / name) for name in ("fixtures.safetensors", "parity.json", "model.pt2")}
    (fx / "fx_manifest.json").write_text(json.dumps({"files": files, "native_fp8": {"storage": "uint8"}}))
    return tmp_path / "bundle"


def test_write_boundary_records_the_packager_contract(tmp_path):
    hidden = torch.randn(1, 4, 8).to(torch.bfloat16)
    ids = torch.arange(4).reshape(1, 4)
    out = torch.randn(1, 4, 8).to(torch.bfloat16)
    bundle = _fake_bundle(tmp_path, [hidden, ids], [out])
    source = backend_loader.write_boundary(bundle, "fx", 0, tmp_path / "boundary")
    on_disk = json.loads((tmp_path / "boundary" / "source.json").read_text())
    assert on_disk == source and source["input_keys"] == ["input_0", "input_1"]
    assert source["tensors"]["input_0"]["dtype"] == "torch.bfloat16"          # prepare_live_execution's str(dtype)
    assert source["tensors"]["input_0"]["shape"] == [1, 4, 8]
    assert source["model_sha256"] == fx_export._file_record(bundle / "fx" / "model.pt2")["sha256"]
    from safetensors.torch import load_file
    saved = load_file(str(tmp_path / "boundary" / "inputs.safetensors"))
    assert torch.equal(saved["input_0"], hidden) and torch.equal(saved["input_1"], ids)
    expected = np.load(tmp_path / "boundary" / "expected.npz")
    assert expected["output_0"].dtype == np.uint16
    assert np.array_equal(expected["output_0"], out.view(torch.uint16).numpy())


def test_select_case_refuses_changed_fixtures(tmp_path):
    bundle = _fake_bundle(tmp_path, [torch.zeros(2)], [torch.ones(2)])
    save_file({"case_0.input_0": torch.ones(2), "case_0.output_0": torch.ones(2)},
              str(bundle / "fx" / "fixtures.safetensors"))
    with pytest.raises(ValueError, match="differs"):
        backend_loader.select_case(bundle / "fx", 0)


class _Program:
    """The two ExportedProgram members export_prefix reads."""

    def __init__(self, state, buffers=()):
        self.state_dict = state
        self.constants = {name: torch.zeros(1) for name in buffers}
        self.graph_signature = type("Sig", (), {"input_specs": [
            type("Spec", (), {"kind": "InputKind.BUFFER", "target": name})() for name in buffers]})()


def test_export_prefix_maps_lifted_state_onto_export_names():
    tensors = {"model.layers.0.mlp.weight_codes": 0, "model.norm.weight": 0}
    program = _Program({"model.model.norm.weight": torch.zeros(1),
                        "model.model.layers.0.mlp.weight": torch.empty(1, device="meta")},
                       buffers=["model.model.layers.0.mlp.weight_codes"])
    assert backend_loader.export_prefix(program, tensors) == "model."
    assert backend_loader.export_prefix(_Program({"norm.weight": torch.zeros(1)}), {"norm.weight": 0}) == ""
    with pytest.raises(ValueError, match="no unique wrapper prefix"):
        backend_loader.export_prefix(_Program({"model.model.extra": torch.zeros(1)}), tensors)
