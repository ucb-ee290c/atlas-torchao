"""Reference replay capture: live numerical dependencies and independent .pt2 reload."""

import copy
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import algos
import fx_export
import pytest
import quant_export
import ref_numerics as rn
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from test_export import _calib, _quantize_export, _reload
from test_ladder import FAST, tiny_llama


@pytest.fixture
def linear_replay():
    torch.manual_seed(13)
    module = torch.nn.Sequential(torch.nn.Linear(8, 4)).to(torch.bfloat16).eval()
    algo = algos.make("granularity:w=channel,a=token,rule=nc")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST),
                                       [], [], {}, {})
    first = torch.tensor([[0.1, -0.2, 0.7, 1.3, -0.4, 0.03, -1.1, 0.8],
                          [0.5, -1.4, 0.2, 0.9, 0.02, -0.8, 1.7, -0.1]], dtype=torch.bfloat16)
    second = first * torch.tensor([[256.0], [2.0 ** -8]], dtype=torch.bfloat16)
    return module, policy, [(first,), (second,)]


@pytest.mark.parametrize("strict", [True, False])
def test_canonical_llama_export_replay_capture_and_fresh_process(tmp_path, strict):
    eager, directory, _, options = _quantize_export(tmp_path, "atlas_v1", layers=1)
    loaded = _reload(directory, "ref_conv", options, layers=1)
    policy = json.loads((directory / "policy.json").read_text())
    inputs = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
              (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]
    with torch.no_grad():
        for (ids,) in inputs:
            assert torch.equal(eager(ids, use_cache=False).logits,
                               loaded(ids, use_cache=False).logits)
    capture = tmp_path / "fx"
    original_weights = {name: layer.weight for name, layer in loaded.named_modules()
                        if type(layer).__name__ == "RefQuantLinear"}
    report = fx_export.capture_fx(fx_export.CausalLogits(loaded).eval(), inputs, capture,
                                 policy=policy, strict=strict, fresh_process=True)
    assert report["status"] == "passed"
    assert report["strict"] is strict
    assert report["cases"] == 2 and report["weight_sites_verified"] == 7
    assert report["same_process_reload"]["all_outputs_byte_identical"]
    assert report["fresh_process_reload"]["all_outputs_byte_identical"]
    assert report["fresh_process_reload"]["model_replay_modules_imported"] == []
    assert report["hardware_compatibility_verified"] is False
    assert report["native_fp8_mlir_verified"] is False
    saved = torch.export.load(capture / "model.pt2")
    for name, weight in original_weights.items():
        assert loaded.get_submodule(name).weight is weight and not weight.is_meta
        placeholder = saved.state_dict.get("model." + name + ".weight")
        assert placeholder is None or placeholder.is_meta


def test_causal_cli_captures_local_export_with_local_architecture(tmp_path):
    model = tiny_llama(layers=1)
    # Match production modelutils.load_llama: complete RoPE state is exported, because
    # the weight-free architecture loader restores these buffers as persistent too.
    for layer in model.modules():
        layer._non_persistent_buffers_set.clear()
    config_dir = tmp_path / "architecture"
    model.config.save_pretrained(config_dir)
    algo = algos.make("atlas_v1")
    batches, calibration = _calib()
    quantized = algos.quantize(model, "ref_conv", ["lm_head"], dict(FAST),
                               algo=algo, calib=calibration)
    export_dir = tmp_path / "export"
    manifest = quant_export.export(model, export_dir, mode="ref_conv", report=quantized,
                                   algo=algo, algo_spec="atlas_v1", options=dict(FAST), skip=["lm_head"],
                                   model_record={"kind": "causal_lm", "id": str(config_dir)},
                                   calib_batches=batches, calib_record={"source": "test"})
    directory = tmp_path / "cli_fx"
    done = subprocess.run([sys.executable, str(Path(fx_export.__file__).resolve()),
                           "--export-dir", str(export_dir), "--model", str(config_dir),
                           "--out-dir", str(directory), "--device", "cpu", "--seqlen", "8"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((directory / "verification.json").read_text())
    assert report["status"] == "passed"
    assert report["weight_sites_verified"] == 7
    assert report["fresh_process_reload"]["model_replay_modules_imported"] == []
    assert (report["provenance"]["export_provenance"]["atlas_torchao_commit"]
            == manifest["provenance"]["atlas_torchao_commit"])
    fixtures = load_file(str(directory / "fixtures.safetensors"))
    parity = json.loads((directory / "parity.json").read_text())
    assert all(fixtures[case["inputs"][0]].shape == (1, 8) for case in parity["cases"])


def test_canonical_tiny_dit_capture_retains_exact_bf16_exception(tmp_path):
    from test_algo_atlas_v1 import DIT_PATCH, TinyDiT, dit_calib

    torch.manual_seed(21)
    module = TinyDiT()
    algo = algos.make("atlas_v1")
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo, calib=dit_calib())
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [],
                                       report["algo"]["bf16_opt_out"], {}, {})
    assert sum(site["status"] == "host" for site in policy["sites"].values()) == 1
    native_weight = module.get_submodule(DIT_PATCH).weight
    expected_weight = native_weight.detach().clone()
    x = torch.randn(2, 3, 4, 4, dtype=torch.bfloat16)
    directory = tmp_path / "dit_fx"
    result = fx_export.capture_fx(module, [(x,), (x * 3,)], directory,
                                 policy=policy, fresh_process=True)
    assert result["status"] == "passed" and result["weight_sites_verified"] == 2
    assert module.get_submodule(DIT_PATCH).weight is native_weight
    saved = torch.export.load(directory / "model.pt2")
    preserved = saved.state_dict[DIT_PATCH + ".weight"]
    assert preserved.dtype == torch.bfloat16 and not preserved.is_meta
    assert torch.equal(preserved, expected_weight)
    assert DIT_PATCH + "_extra.weight_codes" in dict(saved.named_buffers())
    assert DIT_PATCH + ".weight_codes" not in dict(saved.named_buffers())
    inspection = json.loads((directory / "graph_inspection.json").read_text())
    assert {row["site"] for row in inspection["weight_site_bindings"]} == {DIT_PATCH + "_extra", "head"}


def test_runtime_token_scales_and_nonpersistent_buffers_survive_save(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    layer = module[0]
    assert {"weight_codes", "weight_scale_exp"} <= layer._non_persistent_buffers_set
    assert "0.weight_codes" not in module.state_dict()
    before_codes = layer.weight_codes.clone()
    before_exp = layer.weight_scale_exp.clone()
    scales = [layer.algo.act_scale("0", x, "linear").exp for (x,) in cases]
    assert not torch.equal(scales[0], scales[1])
    directory = tmp_path / "fx"
    report = fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    assert report["status"] == "passed"
    program = torch.export.load(directory / "model.pt2")
    buffers = dict(program.named_buffers())
    assert buffers["0.weight_codes"].dtype == torch.uint8
    assert buffers["0.weight_scale_exp"].dtype == torch.int32
    assert torch.equal(buffers["0.weight_codes"], before_codes)
    assert torch.equal(buffers["0.weight_scale_exp"], before_exp)
    inspection = fx_export.inspect_graph(program, module, policy)
    assert inspection["weight_site_bindings"][0]["dynamic_scale_witnesses"]
    with torch.no_grad():
        for args in cases:
            assert torch.equal(program.module()(*args), module(*args))


@pytest.mark.parametrize("buffer_name", ["weight_codes", "weight_scale_exp"])
def test_code_or_exponent_mutation_changes_reloaded_graph(tmp_path, linear_replay, buffer_name):
    module, policy, cases = linear_replay
    directory = tmp_path / "fx"
    fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    program = torch.export.load(directory / "model.pt2")
    replay = program.module()
    with torch.no_grad():
        baseline = replay(*cases[0]).clone()
        buffer = dict(program.named_buffers())["0." + buffer_name]
        if buffer_name == "weight_codes":
            buffer.zero_()
        else:
            buffer.add_(2)
        changed = replay(*cases[0]).clone()
        assert not torch.equal(baseline, changed)
        # Change the eager code/exponent source as well: no float weight is recomputed.
        getattr(module[0], buffer_name).copy_(buffer)
        assert torch.equal(changed, module(*cases[0]))
    changed_path = tmp_path / "changed.pt2"
    torch.export.save(program, changed_path)
    with torch.no_grad():
        assert torch.equal(changed, torch.export.load(changed_path).module()(*cases[0]))


def test_float_placeholder_value_is_not_an_operand(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    original = module[0].weight
    pointer = original.untyped_storage().data_ptr()
    with torch.no_grad():
        expected = [module(*args).clone() for args in cases]
        module[0].weight.fill_(float("nan"))
        for args, output in zip(cases, expected):
            assert torch.equal(module(*args), output)
    report = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False)
    assert report["status"] == "passed"
    assert module[0].weight is original
    assert not original.is_meta and original.untyped_storage().data_ptr() == pointer
    assert torch.isnan(original).all()


def test_capture_failure_restores_original_weight(tmp_path, linear_replay, monkeypatch):
    module, policy, cases = linear_replay
    original = module[0].weight
    before = original.detach().clone()
    def fail_capture(boundary, *args, **kwargs):
        assert boundary[0].weight.is_meta
        raise RuntimeError("deliberate capture failure")
    monkeypatch.setattr(torch.export, "export", fail_capture)
    with pytest.raises(RuntimeError, match="deliberate capture failure"):
        fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=False)
    assert module[0].weight is original and not original.is_meta
    assert torch.equal(original, before)


def test_existing_output_directory_is_preserved(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    directory = tmp_path / "existing"
    directory.mkdir()
    marker = directory / "keep.txt"
    marker.write_text("retain existing contents")
    with pytest.raises(FileExistsError):
        fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    assert marker.read_text() == "retain existing contents"
    assert sorted(p.name for p in directory.iterdir()) == ["keep.txt"]


def test_mutable_per_step_policy_is_refused(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    policy = copy.deepcopy(policy)
    policy["sites"]["0"]["activation"] = {"mode": "static_per_step", "exps": [-8, -2]}
    directory = tmp_path / "fx"
    with pytest.raises(fx_export.FXVerificationError, match="per-step"):
        fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    assert not (directory / "model.pt2").exists()
    assert json.loads((directory / "verification.json").read_text())["status"] == "diagnostic_failure"


def test_corrupted_saved_quant_buffer_is_refused(tmp_path, linear_replay):
    module, policy, cases = linear_replay
    directory = tmp_path / "fx"
    fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    program = torch.export.load(directory / "model.pt2")
    with torch.no_grad():
        dict(program.named_buffers())["0.weight_scale_exp"].add_(1)
    torch.export.save(program, directory / "model.pt2")
    with pytest.raises(fx_export.FXVerificationError, match="inventory|sha256|digest"):
        fx_export.verify_saved(directory)


def _replace_json_and_refresh_manifest(directory, name, value):
    """Bypass byte-integrity rejection so semantic verification sees the altered record."""
    path = directory / name
    path.write_text(json.dumps(value, indent=2) + "\n")
    manifest_path = directory / "fx_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "bytes": path.stat().st_size}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


@pytest.mark.parametrize("empty", [False, True])
def test_incomplete_bundle_manifest_is_refused(tmp_path, linear_replay, empty):
    module, policy, cases = linear_replay
    directory = tmp_path / "fx"
    fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    manifest_path = directory / "fx_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if empty:
        manifest["files"] = {}
    else:
        del manifest["files"]["graph_inspection.json"]
    manifest_path.write_text(json.dumps(manifest) + "\n")
    with pytest.raises(fx_export.FXVerificationError, match="inventory.*incomplete"):
        fx_export.verify_saved(directory)


@pytest.mark.parametrize("mutation,message", [
    ("empty_cases", "at least two distinct"),
    ("duplicate_inputs", "at least two distinct"),
    ("empty_inputs", "empty or incomplete case"),
])
def test_rehashed_invalid_parity_cases_are_refused(tmp_path, linear_replay, mutation, message):
    module, policy, cases = linear_replay
    directory = tmp_path / "fx"
    fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    parity = json.loads((directory / "parity.json").read_text())
    if mutation == "empty_cases":
        parity["cases"] = []
    elif mutation == "duplicate_inputs":
        parity["cases"][1]["inputs"] = list(parity["cases"][0]["inputs"])
    else:
        parity["cases"][0]["inputs"] = []
    _replace_json_and_refresh_manifest(directory, "parity.json", parity)
    with pytest.raises(fx_export.FXVerificationError, match=message):
        fx_export.verify_saved(directory)


@pytest.mark.parametrize("record", ["buffer_inventory.json", "graph_inspection.json"])
def test_rehashed_empty_saved_inventory_is_refused(tmp_path, linear_replay, record):
    module, policy, cases = linear_replay
    directory = tmp_path / "fx"
    fx_export.capture_fx(module, cases, directory, policy=policy, fresh_process=False)
    value = json.loads((directory / record).read_text())
    if record == "buffer_inventory.json":
        value = {}
        message = "code/exponent inventory"
    else:
        value["weight_site_bindings"] = []
        value["weight_sites_verified"] = 0
        message = "weight binding inventory"
    _replace_json_and_refresh_manifest(directory, record, value)
    with pytest.raises(fx_export.FXVerificationError, match=message):
        fx_export.verify_saved(directory)


@pytest.mark.parametrize("missing", ["weight_codes", "weight_scale_exp"])
@pytest.mark.parametrize("shape_only", [False, True])
def test_disconnected_code_or_scale_is_refused(linear_replay, missing, shape_only):
    module, policy, cases = linear_replay
    layer = module[0]
    layer.register_buffer("other_source", getattr(layer, missing).clone())
    def disconnected(self):
        substitute = torch.zeros_like(getattr(self, missing)) if shape_only else self.other_source
        codes = substitute if missing == "weight_codes" else self.weight_codes
        exp = substitute if missing == "weight_scale_exp" else self.weight_scale_exp
        return rn.dequantize(codes, exp)
    layer._dequant_weight = types.MethodType(disconnected, layer)
    program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="lifted|consumes both"):
        fx_export.inspect_graph(program, module, policy)


def test_consumed_float_placeholder_is_refused(linear_replay):
    module, policy, cases = linear_replay
    def contaminated(self):
        return rn.dequantize(self.weight_codes, self.weight_scale_exp) + self.weight.float() * 0
    module[0]._dequant_weight = types.MethodType(contaminated, module[0])
    program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="float placeholder"):
        fx_export.inspect_graph(program, module, policy)


@pytest.mark.parametrize("derive_from_constant", [False, True, "shape_only"])
def test_frozen_dynamic_activation_scale_is_refused(linear_replay, derive_from_constant):
    from algos.granularity import scale_exp

    module, policy, cases = linear_replay
    layer = module[0]
    layer.register_buffer("frozen_input", cases[0][0].clone())
    layer.register_buffer("frozen_exp", scale_exp(cases[0][0], (-1,), "nc"))
    def frozen_forward(self, x):
        x = x.to(self.weight.dtype)
        if derive_from_constant == "shape_only":
            exp = scale_exp(torch.ones_like(x), (-1,), "nc")
        else:
            exp = scale_exp(self.frozen_input, (-1,), "nc") if derive_from_constant else self.frozen_exp
        codes = rn.e4m3_vpack(x, exp)
        result = F.linear(rn.dequantize(codes, exp).to(self.weight.dtype),
                          self._dequant_weight().to(self.weight.dtype))
        return result + self.bias
    layer.forward = types.MethodType(frozen_forward, layer)
    program = torch.export.export(module, cases[0], strict=True)
    with pytest.raises(fx_export.FXVerificationError, match="dynamic exponent.*absent|input-independent"):
        fx_export.inspect_graph(program, module, policy)
