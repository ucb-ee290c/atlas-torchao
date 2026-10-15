"""Capture reference export replay as a saved, checked FX ExportedProgram.

This is reference emulation, not native-FP8 MLIR or device code. No calibration
or quantizer search runs here. The initial CLI boundary is causal-LM logits.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import _pytree

SCHEMA = "atlas_torchao.fx_verification.v1"
QUANT_BUFFERS = ("weight_codes", "weight_scale_exp", "weight_lo_codes", "weight_lo_scale_exp")
BUNDLE_FILES = ("model.pt2", "fixtures.safetensors", "parity.json", "buffer_inventory.json",
                "graph_inspection.json", "policy.json", "graph.txt", "graph_signature.txt")
CONTRACTIONS = {
    torch.ops.aten.linear.default: (0, 1),
    torch.ops.aten.mm.default: (0, 1),
    torch.ops.aten.addmm.default: (1, 2),
    torch.ops.aten.matmul.default: (0, 1),
    torch.ops.aten.bmm.default: (0, 1),
    torch.ops.aten.conv2d.default: (0, 1),
    torch.ops.aten.convolution.default: (0, 1),
}
SHAPE_ONLY_FACTORIES = {
    torch.ops.aten.ones_like.default, torch.ops.aten.zeros_like.default,
    torch.ops.aten.full_like.default, torch.ops.aten.empty_like.default,
    torch.ops.aten.new_ones.default, torch.ops.aten.new_zeros.default,
    torch.ops.aten.new_full.default, torch.ops.aten.new_empty.default,
}


class FXVerificationError(RuntimeError):
    pass


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "sha256": hashlib.sha256(raw).hexdigest()}


def _file_record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"sha256": digest.hexdigest(), "bytes": Path(path).stat().st_size}


def _inventory(items):
    result = {}
    for name, value in items:
        suffix = name.rpartition(".")[2]
        if suffix in QUANT_BUFFERS:
            required = torch.uint8 if suffix.endswith("codes") else torch.int32
            if value.dtype != required:
                raise FXVerificationError(f"{name}: expected {required}, found {value.dtype}")
            result[name] = tensor_record(value)
    return result


def _leaves(output):
    leaves, spec = _pytree.tree_flatten(output)
    if not leaves or not all(isinstance(value, torch.Tensor) for value in leaves):
        raise FXVerificationError("the capture boundary must return only tensors")
    if not all(bool(torch.isfinite(value).all()) for value in leaves):
        raise FXVerificationError("non-finite output at the capture boundary")
    return leaves, _pytree.treespec_dumps(spec)


def _compare(output, expected, expected_spec):
    leaves, spec = _leaves(output)
    if spec != expected_spec or len(leaves) != len(expected):
        raise FXVerificationError("output structure changed")
    if any(tensor_record(a) != tensor_record(b) for a, b in zip(leaves, expected)):
        raise FXVerificationError("output tensors are not byte-identical")


def _value_parents(node):
    if node.target in SHAPE_ONLY_FACTORIES:
        return []
    return [parent for parent in node.all_input_nodes if any(isinstance(value, torch.Tensor)
            for value in _pytree.tree_leaves(parent.meta.get("val")))]


def _ancestors(root, *, stop_at_contractions=False, value_only=False):
    found, pending = set(), [root]
    while pending:
        node = pending.pop()
        if not isinstance(node, torch.fx.Node) or node in found:
            continue
        found.add(node)
        if not (stop_at_contractions and node.target in CONTRACTIONS):
            pending.extend(_value_parents(node) if value_only else node.all_input_nodes)
    return found


def _module_scope(node, path):
    return any(value[0] == path for value in node.meta.get("nn_module_stack", {}).values())


@contextlib.contextmanager
def _meta_placeholders(module):
    """Avoid saving unused reconstructed float weights; restore the caller's model."""
    originals = []
    try:
        for layer in module.modules():
            if type(layer).__name__ in ("RefQuantLinear", "RefQuantConv2d"):
                original = layer.weight
                originals.append((layer, original))
                layer.weight = torch.nn.Parameter(torch.empty_strided(
                    original.shape, original.stride(), dtype=original.dtype, device="meta"),
                    requires_grad=original.requires_grad)
        yield
    finally:
        for layer, original in originals:
            layer.weight = original


def inspect_graph(program, module, policy):
    """Verify real contraction dependencies, not buffer labels alone."""
    nodes = list(program.graph.nodes)
    by_name = {node.name: node for node in nodes}
    lifted = {spec.target: by_name[spec.arg.name] for spec in program.graph_signature.input_specs
              if isinstance(spec.target, str) and hasattr(spec.arg, "name")}
    user_inputs = {by_name[spec.arg.name] for spec in program.graph_signature.input_specs
                   if spec.kind.name == "USER_INPUT" and hasattr(spec.arg, "name")}
    depends_on_input = {}
    for node in nodes:
        # A shape edge is not a value edge: ones_like(x) has x's shape but its
        # values cannot make an activation exponent dynamic. Export metadata
        # also distinguishes Tensor values from symbolic sizes/strides.
        depends_on_input[node] = node in user_inputs or (
            any(depends_on_input[parent] for parent in _value_parents(node)))
    contractions = [node for node in nodes if node.op == "call_function" and node.target in CONTRACTIONS]
    weight_dependencies = {node: _ancestors(node.args[CONTRACTIONS[node.target][1]],
                                          stop_at_contractions=True, value_only=True)
                           for node in contractions}
    selected = {name: site for name, site in policy["sites"].items()
                if site["status"] == "quantized" and site["kind"] in ("linear", "conv2d")}
    actual, rows = set(), []
    for path, layer in module.named_modules():
        if type(layer).__name__ not in ("RefQuantLinear", "RefQuantConv2d"):
            continue
        site = layer.layer_name
        if site not in selected or site in actual:
            raise FXVerificationError(f"unexpected or duplicated quantized module: {site}")
        actual.add(site)
        if layer.fm is not None or getattr(layer, "residual", None):
            raise FXVerificationError(f"{site}: initial FX path requires framework accumulation and no residual passes")
        prefix = path + "." if path else ""
        code = lifted.get(prefix + "weight_codes")
        scale = lifted.get(prefix + "weight_scale_exp")
        placeholder = lifted.get(prefix + "weight")
        if code is None or scale is None:
            raise FXVerificationError(f"{site}: codes or exponents were not lifted")
        if placeholder is not None and placeholder.users:
            raise FXVerificationError(f"{site}: graph consumes the float placeholder")
        matches = []
        for node in contractions:
            left_index = CONTRACTIONS[node.target][0]
            if code in weight_dependencies[node] and scale in weight_dependencies[node]:
                matches.append((node, node.args[left_index]))
        if not matches:
            raise FXVerificationError(f"{site}: no contraction consumes both codes and exponents")
        rule = selected[site]["activation"]
        if rule.get("mode") == "static_per_step":
            raise FXVerificationError("mutable per-step activation scales are not supported")
        witnesses = []
        if rule.get("source") == "reference" or rule.get("mode") == "dynamic":
            for contraction, operand in matches:
                local = _ancestors(operand, stop_at_contractions=True, value_only=True)
                exponents = [node for node in local if node.target == torch.ops.aten.frexp.Tensor
                             and _module_scope(node, path) and depends_on_input[node]]
                if not exponents:
                    raise FXVerificationError(f"{site}: dynamic exponent derivation is absent or input-independent")
                for exponent in exponents:
                    reductions = [node for node in _ancestors(exponent, stop_at_contractions=True, value_only=True)
                                  if node.target in (torch.ops.aten.amax.default, torch.ops.aten.max.default)
                                  and _module_scope(node, path) and depends_on_input[node]]
                    if not reductions:
                        raise FXVerificationError(f"{site}: input-dependent amax is absent")
                if rule.get("rule") == "nc" and not any(
                        node.target == torch.ops.aten.gt.Tensor and _module_scope(node, path)
                        and depends_on_input[node] for node in local):
                    raise FXVerificationError(f"{site}: live nc comparison is absent")
                witnesses.append({"contraction": contraction.name,
                                  "frexp": sorted(node.name for node in exponents)})
        rows.append({"site": site, "module": path, "kind": selected[site]["kind"],
                     "code_buffer": prefix + "weight_codes", "scale_buffer": prefix + "weight_scale_exp",
                     "contractions": [node.name for node, _ in matches],
                     "dynamic_scale_witnesses": witnesses})
    if actual != set(selected):
        raise FXVerificationError(f"quantized module census differs: missing {sorted(set(selected) - actual)}")
    attention = [name for name, site in policy["sites"].items()
                 if site["status"] == "quantized" and site["kind"] == "attention"]
    unused = [spec.target for spec in program.graph_signature.input_specs
              if spec.kind.name == "PARAMETER" and isinstance(spec.target, str)
              and not by_name[spec.arg.name].users]
    for path, layer in module.named_modules():
        if type(layer).__name__ in ("RefQuantLinear", "RefQuantConv2d"):
            name = (path + "." if path else "") + "weight"
            if name in program.state_dict and not program.state_dict[name].is_meta:
                raise FXVerificationError(f"{name}: unused float placeholder payload was retained")
    return {"weight_site_bindings": rows, "weight_sites_verified": len(rows),
            "functional_attention_sites": attention,
            "functional_attention_structural_binding": "not certified; covered by boundary output parity",
            "unused_parameters": unused,
            "unused_quantized_weight_storage": "meta placeholders; no serialized float weight payload",
            "scope": "reference-emulation FX; not native-FP8 MLIR or hardware compatibility"}


def verify_saved(directory, device="cpu"):
    """Replay a .pt2 using only PyTorch and recorded fixtures, without model classes."""
    from safetensors.torch import load_file

    directory = Path(directory)
    manifest = json.loads((directory / "fx_manifest.json").read_text())
    if manifest.get("schema") != "atlas_torchao.fx_bundle.v1":
        raise FXVerificationError("unsupported FX bundle manifest")
    if set(manifest.get("files", {})) != set(BUNDLE_FILES):
        raise FXVerificationError("FX bundle integrity inventory is incomplete or unexpected")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or _file_record(directory / name) != expected:
            raise FXVerificationError(f"FX bundle integrity/inventory mismatch: {name}")
    record = json.loads((directory / "parity.json").read_text())
    fixtures = load_file(str(directory / "fixtures.safetensors"), device=device)
    cases = record.get("cases", [])
    if not isinstance(cases, list) or len(cases) < 2:
        raise FXVerificationError("saved parity requires at least two distinct input cases")
    signatures = set()
    for case in cases:
        if not case.get("inputs") or not case.get("outputs") or not isinstance(case.get("output_spec"), str):
            raise FXVerificationError("saved parity has an empty or incomplete case")
        signatures.add(json.dumps([tensor_record(fixtures[name]) for name in case["inputs"]], sort_keys=True))
    if len(signatures) < 2:
        raise FXVerificationError("saved parity requires at least two distinct input cases")
    program = torch.export.load(directory / "model.pt2")
    expected_inventory = json.loads((directory / "buffer_inventory.json").read_text())
    if not expected_inventory or _inventory(program.named_buffers()) != expected_inventory:
        raise FXVerificationError("saved code/exponent inventory differs")
    inspected = json.loads((directory / "graph_inspection.json").read_text())
    if not inspected.get("weight_site_bindings") or inspected.get("weight_sites_verified") != len(
            inspected["weight_site_bindings"]):
        raise FXVerificationError("saved weight binding inventory is empty or inconsistent")
    for row in inspected["weight_site_bindings"]:
        name = row["module"] + ".weight" if row["module"] else "weight"
        if name in program.state_dict and not program.state_dict[name].is_meta:
            raise FXVerificationError("saved float placeholder unexpectedly materialized")
    replay = program.module()
    with torch.no_grad():
        for case in cases:
            args = tuple(fixtures[name] for name in case["inputs"])
            _compare(replay(*args), [fixtures[name] for name in case["outputs"]], case["output_spec"])
    return {"status": "passed", "cases": len(cases), "all_outputs_byte_identical": True,
            "code_scale_bytes_preserved": True, "unused_float_weights_not_materialized": True,
            "model_replay_modules_imported": [name for name in ("load_quantized", "ref_profile", "ref_attention")
                                              if name in sys.modules]}


def capture_fx(module, cases, directory, *, policy, provenance=None, strict=True, fresh_process=True):
    """Capture fixed-shape reference replay, save, reload and fail closed on mismatch."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "running", "torch": torch.__version__,
              "strict": strict, "capture_mode": "ref_conv framework reference emulation",
              "provenance": provenance or {}, "native_fp8_mlir_verified": False,
              "hardware_compatibility_verified": False, "fresh_process_requested": fresh_process}
    try:
        if len(cases) < 2 or any(not isinstance(case, tuple) or not all(
                isinstance(value, torch.Tensor) for value in case) for case in cases):
            raise FXVerificationError("provide at least two positional tensor-input cases")
        if len({json.dumps([tensor_record(value) for value in case], sort_keys=True) for case in cases}) < 2:
            raise FXVerificationError("verification inputs must differ")
        for site in policy["sites"].values():
            rules = [site.get("activation", {})] + list(site.get("operands", {}).values())
            if any(rule.get("mode") == "static_per_step" for rule in rules):
                raise FXVerificationError("mutable per-step activation scales are not supported")
        if module.training:
            raise FXVerificationError("capture requires eval mode")
        fixtures, parity, expected = {}, {"cases": []}, []
        with torch.no_grad():
            for i, args in enumerate(cases):
                leaves, spec = _leaves(module(*args))
                outputs = [value.detach().cpu().clone() for value in leaves]
                expected.append((outputs, spec))
                input_keys = [f"case_{i}.input_{j}" for j in range(len(args))]
                output_keys = [f"case_{i}.output_{j}" for j in range(len(outputs))]
                fixtures.update({name: value.detach().cpu().contiguous() for name, value in zip(input_keys, args)})
                fixtures.update(dict(zip(output_keys, outputs)))
                parity["cases"].append({"inputs": input_keys, "outputs": output_keys, "output_spec": spec})
            before = _inventory(module.named_buffers())
            if not before:
                raise FXVerificationError("no quantized weight buffers at this boundary")
            with _meta_placeholders(module):
                program = torch.export.export(module, cases[0], strict=strict)
            if _inventory(program.named_buffers()) != before:
                raise FXVerificationError("capture changed code/exponent buffers")
            graph_report = inspect_graph(program, module, policy)
            for args, (outputs, spec) in zip(cases, expected):
                _compare(program.module()(*args), outputs, spec)
            save_file(fixtures, str(directory / "fixtures.safetensors"))
            _json(directory / "parity.json", parity)
            _json(directory / "buffer_inventory.json", before)
            _json(directory / "graph_inspection.json", graph_report)
            _json(directory / "policy.json", policy)
            (directory / "graph.txt").write_text(str(program.graph) + "\n")
            (directory / "graph_signature.txt").write_text(str(program.graph_signature) + "\n")
            torch.export.save(program, directory / "model.pt2")
        _json(directory / "fx_manifest.json", {"schema": "atlas_torchao.fx_bundle.v1",
              "torch": torch.__version__, "device": str(cases[0][0].device),
              "provenance": provenance or {},
              "files": {name: _file_record(directory / name) for name in BUNDLE_FILES}})
        report["same_process_reload"] = verify_saved(directory, str(cases[0][0].device))
        if fresh_process:
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--verify-dir", str(directory),
                                     "--device", str(cases[0][0].device)], capture_output=True, text=True)
            (directory / "fresh_process.log").write_text(result.stdout + result.stderr)
            if result.returncode:
                raise FXVerificationError(f"fresh-process reload failed with exit {result.returncode}")
            report["fresh_process_reload"] = json.loads((directory / "fresh_process_verification.json").read_text())
            if report["fresh_process_reload"]["model_replay_modules_imported"]:
                raise FXVerificationError("fresh-process replay imported original model implementation")
        report.update(status="passed", cases=len(cases), weight_sites_verified=graph_report["weight_sites_verified"],
                      functional_attention_structural_binding=graph_report["functional_attention_structural_binding"],
                      model_pt2_sha256=_file_record(directory / "model.pt2")["sha256"],
                      model_pt2_bytes=(directory / "model.pt2").stat().st_size)
    except Exception as error:
        report.update(status="diagnostic_failure", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _json(directory / "verification.json", report)
    return report


class CausalLogits(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids, use_cache=False).logits


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--model", help="architecture config locator; defaults to the export model id")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seqlen", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--non-strict", action="store_true", help="explicitly select non-strict capture")
    parser.add_argument("--verify-dir", help="fresh-process replay of an existing FX bundle")
    args = parser.parse_args(argv)
    if args.verify_dir:
        report = verify_saved(args.verify_dir, args.device)
        _json(Path(args.verify_dir) / "fresh_process_verification.json", report)
    else:
        if not args.export_dir or not args.out_dir or args.seqlen < 2:
            parser.error("--export-dir, --out-dir and --seqlen >= 2 are required")
        import load_quantized
        import quant_export

        _, policy, manifest = load_quantized.read_export(args.export_dir)
        if (manifest.get("model") or {}).get("kind", "causal_lm") != "causal_lm":
            parser.error("initial CLI supports causal-LM logits only; policy action inference is a separate boundary")
        model, _, _ = load_quantized.load_quantized(args.model, args.export_dir, args.device,
                                                   mode="ref_conv", seqlen=args.seqlen)
        vocab = model.config.vocab_size
        cases = [(torch.randint(vocab, (1, args.seqlen), generator=torch.Generator().manual_seed(args.seed + i))
                  .to(args.device),) for i in range(2)]
        report = capture_fx(CausalLogits(model).eval(), cases, args.out_dir, policy=policy,
                            strict=not args.non_strict,
                            provenance={"export_dir": str(Path(args.export_dir).resolve()),
                                        "export_manifest_sha256": quant_export.sha256_file(
                                            Path(args.export_dir) / "manifest.json"),
                                        "export_provenance": manifest.get("provenance", {}),
                                        "export_recipe": manifest.get("recipe", {}),
                                        "architecture_locator": args.model or (manifest.get("model") or {}).get("id"),
                                        "config_sha256": hashlib.sha256(json.dumps(model.config.to_dict(),
                                            sort_keys=True, default=str).encode()).hexdigest(),
                                        "capture_source": _capture_source(),
                                        "seed": args.seed, "seqlen": args.seqlen})
    print(json.dumps(report, indent=2))
    return 0


def _capture_source():
    root = Path(__file__).resolve().parent.parent
    result = {"fx_export_py": _file_record(Path(__file__))}
    for key, command in (("commit", ["rev-parse", "HEAD"]), ("dirty", ["status", "--porcelain"])):
        process = subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True)
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
