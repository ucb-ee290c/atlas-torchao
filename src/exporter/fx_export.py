"""Capture reference export replay as a saved, checked FX ExportedProgram.

This is reference emulation, not native-FP8 MLIR or device code. No calibration
or quantizer search runs here. The initial CLI boundary is causal-LM logits.

Default: each quantized Linear is decoded (uint8 codes -> E4M3 value, times 2^exp) and contracted in
BF16. With --native-fp8 each eligible Linear is instead two opaque nodes, atlas.quantize_act and
atlas.qlinear (src/exporter/atlas_fx_ops.py), consuming the code and exponent buffers directly; attention
operands, Conv2d and other sites stay decode-based and are listed in graph_inspection.json under
still_emulated. A saved native bundle needs `import exporter.atlas_fx_ops` before torch.export.load.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import operator
import os
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils import _pytree

SCHEMA = "atlas_torchao.fx_verification.v1"
STORAGE_DTYPES = {"uint8": torch.uint8, "float8_e4m3fn": torch.float8_e4m3fn}
RECOGNIZED_QUANT_MODULES = ("RefQuantLinear", "RefQuantConv2d")
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


def _inventory(items, code_dtype=torch.uint8):
    result = {}
    for name, value in items:
        suffix = name.rpartition(".")[2]
        if suffix in QUANT_BUFFERS:
            required = code_dtype if suffix.endswith("codes") else torch.int32
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


def _scope(node):
    """Innermost module path of a node (empty for nodes with no module stack)."""
    stack = node.meta.get("nn_module_stack") or {}
    return list(stack.values())[-1][0] if stack else ""


def _code_hazards(codes):
    """Codes where torch.float8_e4m3fn and ref_numerics.e4m3_unpack disagree: 0x7F / 0xFF (NaN vs
    reserved) and the 14 exponent-field-0 non-zero codes (subnormal vs signed zero)."""
    c = codes.to(torch.int32) & 0xFF
    return ((c & 0x7F) == 0x7F) | ((((c >> 3) & 0xF) == 0) & ((c & 7) != 0))


def native_rule(site, in_features):
    """(mode, reduce, rule, static_exp) of a Linear site's activation rule, or None when
    atlas::quantize_act does not express it (block scales, per-step exponents)."""
    rule = site["activation"]
    if rule.get("source") == "reference":
        return ("dynamic", "tensor", "ref", None)
    if rule["mode"] == "static":
        return ("static", "tensor", "ref", int(rule["exp"]))
    if rule["mode"] != "dynamic":
        return None
    reduce = rule["reduce"]
    if reduce == "block32_or_token":
        if in_features % int(rule.get("block", 32)) == 0:
            return None
        reduce = "token"
    return ("dynamic", reduce, rule["rule"], None) if reduce in ("tensor", "token") else None


def native_candidates(module, policy):
    """{path: (layer, rule)} of the Linear sites that one atlas::qlinear node can carry; the rest
    (Conv2d, per-block weight scales, residual passes, MXU models) stay on the decode path."""
    result = {}
    for path, layer in module.named_modules():
        if type(layer).__name__ != "RefQuantLinear":
            continue
        site = policy["sites"].get(layer.layer_name)
        if (site is None or site["status"] != "quantized" or layer.fm is not None
                or getattr(layer, "residual", None) or layer.weight_block):
            continue
        rule = native_rule(site, layer.in_features)
        if rule is not None and layer.weight_scale_exp.numel() in (1, layer.out_features):
            result[path] = (layer, rule)
    return result


@contextlib.contextmanager
def _native_linears(module, policy, storage):
    """For the capture only: route each candidate Linear through atlas::quantize_act + atlas::qlinear
    and, for float8 storage, hold its codes as a float8_e4m3fn bitcast of the same bytes. The
    caller's forwards and buffers are restored afterwards."""
    import exporter.atlas_fx_ops as atlas_fx_ops

    code_dtype = STORAGE_DTYPES[storage]
    candidates = native_candidates(module, policy)
    undo = []
    try:
        for path, (layer, (mode, reduce, rule, exp)) in candidates.items():
            if storage != "uint8":
                bad = _code_hazards(layer.weight_codes)
                if bool(bad.any()):
                    raise FXVerificationError(
                        f"{path}: {int(bad.sum())} weight codes (0x7F/0xFF or exponent-0 non-zero) decode "
                        f"differently as {storage} than in ref_numerics; use --native-storage uint8")
            undo.append((layer, layer._buffers["weight_codes"], "forward" in layer.__dict__))
            layer._buffers["weight_codes"] = layer.weight_codes.view(code_dtype)
            out_dtype = str(layer.weight.dtype).replace("torch.", "")

            def forward(x, layer=layer, mode=mode, reduce=reduce, rule=rule, exp=exp, out_dtype=out_dtype):
                x = x.to(layer.weight.dtype)
                codes, scale = torch.ops.atlas.quantize_act(x, mode, reduce, rule, exp, storage)
                return torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp,
                                               layer.bias, out_dtype, atlas_fx_ops.WEIGHT_FORMAT)
            layer.forward = forward
        yield {path: rule for path, (_, rule) in candidates.items()}
    finally:
        for layer, codes, had_forward in undo:
            layer._buffers["weight_codes"] = codes
            if not had_forward:
                layer.__dict__.pop("forward", None)


ALLOWED_NATIVE_SCOPE_TARGETS = {operator.getitem, torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype,
                                torch.ops.aten.to.dtype_layout, torch.ops.aten._assert_tensor_metadata.default}


def _inspect_native_site(path, layer, site, rule, nodes, lifted, depends_on_input, storage):
    """One Linear = one atlas::qlinear node fed directly by the lifted code/exponent buffers and by
    one atlas::quantize_act node whose input depends on the user input; nothing else in its scope."""
    import exporter.atlas_fx_ops as atlas_fx_ops

    prefix = path + "." if path else ""
    code, exp = lifted.get(prefix + "weight_codes"), lifted.get(prefix + "weight_scale_exp")
    placeholder, bias = lifted.get(prefix + "weight"), lifted.get(prefix + "bias")
    if code is None or exp is None:
        raise FXVerificationError(f"{layer.layer_name}: codes or exponents were not lifted")
    if placeholder is not None and placeholder.users:
        raise FXVerificationError(f"{layer.layer_name}: graph consumes the float placeholder")
    mine = [n for n in nodes if n.op == "call_function" and n.target == torch.ops.atlas.qlinear.default
            and n.args[2] is code and n.args[3] is exp]
    if len(mine) != 1 or set(code.users) != set(mine) or set(exp.users) != set(mine):
        raise FXVerificationError(f"{layer.layer_name}: expected exactly one atlas.qlinear consuming "
                                  f"the code and exponent buffers directly, found {len(mine)}")
    q = mine[0]
    x_codes, x_exp, _, _, q_bias, out_dtype, weight_format = q.args
    if (not isinstance(x_codes, torch.fx.Node) or x_codes.target is not operator.getitem or x_codes.args[1] != 0
            or not isinstance(x_exp, torch.fx.Node) or x_exp.target is not operator.getitem
            or x_exp.args[1] != 1 or x_codes.args[0] is not x_exp.args[0]):
        raise FXVerificationError(f"{layer.layer_name}: qlinear is not fed by one atlas.quantize_act")
    act = x_codes.args[0]
    if act.target != torch.ops.atlas.quantize_act.default:
        raise FXVerificationError(f"{layer.layer_name}: qlinear is not fed by one atlas.quantize_act")
    mode, reduce, scale_rule, static_exp = rule
    if not depends_on_input[act.args[0]]:
        raise FXVerificationError(f"{layer.layer_name}: quantize_act input does not depend on the graph input")
    if tuple(act.args[1:]) != (mode, reduce, scale_rule, static_exp, storage):
        raise FXVerificationError(f"{layer.layer_name}: quantize_act arguments {act.args[1:]} differ from "
                                  f"the policy rule {rule}")
    if out_dtype != str(layer.weight.dtype).replace("torch.", "") or weight_format != atlas_fx_ops.WEIGHT_FORMAT:
        raise FXVerificationError(f"{layer.layer_name}: qlinear dtype/format attributes differ from the layer")
    if (layer.bias is None) != (q_bias is None) or (bias is not None and q_bias is not bias):
        raise FXVerificationError(f"{layer.layer_name}: bias is not consumed by qlinear")
    scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
    stray = sorted({str(n.target) for n in scope if n.target not in ALLOWED_NATIVE_SCOPE_TARGETS
                    and n not in (q, act)})
    if stray:
        raise FXVerificationError(f"{layer.layer_name}: decode/contraction ops remain in the layer: {stray}")
    return {"site": layer.layer_name, "module": path, "kind": "linear", "native": True,
            "code_buffer": prefix + "weight_codes", "scale_buffer": prefix + "weight_scale_exp",
            "code_dtype": str(layer.weight_codes.dtype).replace("torch.", ""),
            "qlinear": q.name, "quantize_act": act.name,
            "activation_rule": {"mode": mode, "reduce": reduce, "rule": scale_rule, "static_exp": static_exp},
            "nodes_in_scope": [n.name for n in scope]}


def _emulated_report(nodes, module, native_paths):
    """Modules whose quantized operations are still emulated (decode and contract), keyed by module path:
    their contraction nodes (attention Q.K and P.V, Conv2d, any Linear kept on the decode path), the
    amax / frexp / gt nodes that derive their scales, and the module's total node count (which also
    holds ordinary BF16 nodes such as RoPE and softmax). `other_nodes_outside_quantized_modules`
    counts the remaining call_function nodes (norms, residual adds, SiLU, lm_head, embeddings)."""
    emulated = {path: {"kind": "attention" if isinstance(getattr(layer, "_atlas_ref_attn", None), dict)
                       else ("linear" if type(layer).__name__ == "RefQuantLinear" else "conv2d"),
                       "node_count": 0, "contractions": [], "scale_derivation": []}
                for path, layer in module.named_modules()
                if path not in native_paths and (isinstance(getattr(layer, "_atlas_ref_attn", None), dict)
                                                 or type(layer).__name__ in RECOGNIZED_QUANT_MODULES)}
    scale_ops = (torch.ops.aten.frexp.Tensor, torch.ops.aten.amax.default, torch.ops.aten.max.default,
                 torch.ops.aten.gt.Tensor)
    other = 0
    for node in nodes:
        if node.op != "call_function":
            continue
        entry = emulated.get(_scope(node))
        if entry is None:
            other += _scope(node) not in native_paths
            continue
        entry["node_count"] += 1
        if node.target in CONTRACTIONS:
            entry["contractions"].append(node.name)
        if node.target in scale_ops:
            entry["scale_derivation"].append(node.name)
    return {"emulated_modules": emulated, "other_nodes_outside_quantized_modules": other}


def inspect_graph(program, module, policy, native_paths=None, storage="uint8"):
    """Verify real contraction dependencies, not buffer labels alone.

    native_paths: {module path: (mode, reduce, rule, static_exp)} of the Linear layers captured as
    atlas.quantize_act + atlas.qlinear (fx_export --native-fp8); those are checked by
    _inspect_native_site, the rest by the decode-graph rules below. `storage` is the code buffer dtype
    name those nodes were captured with."""
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
        if native_paths and path in native_paths:
            rows.append(_inspect_native_site(path, layer, selected[site], native_paths[path], nodes, lifted,
                                             depends_on_input, storage))
            continue
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
    report = {"weight_site_bindings": rows, "weight_sites_verified": len(rows),
              "functional_attention_sites": attention,
              "functional_attention_structural_binding": "not certified; covered by boundary output parity",
              "unused_parameters": unused,
              "unused_quantized_weight_storage": "meta placeholders; no serialized float weight payload",
              "scope": "reference-emulation FX; not native-FP8 MLIR or hardware compatibility"}
    if native_paths:
        report["native_fp8_sites"] = len(native_paths)
        report["still_emulated"] = _emulated_report(nodes, module, native_paths)
        report["scope"] = ("native-FP8-carrying FX (atlas.qlinear custom op) for the listed Linear sites; "
                           "everything in still_emulated stays decode-and-contract reference emulation; "
                           "not native-FP8 MLIR or hardware compatibility")
    return report


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
    native = manifest.get("native_fp8")
    if native:
        # The saved graph names atlas.quantize_act / atlas.qlinear; they must be registered before load.
        import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
    program = torch.export.load(directory / "model.pt2")
    expected_inventory = json.loads((directory / "buffer_inventory.json").read_text())
    code_dtype = STORAGE_DTYPES[native["storage"]] if native else torch.uint8
    if not expected_inventory or _inventory(program.named_buffers(), code_dtype) != expected_inventory:
        raise FXVerificationError("saved code/exponent inventory differs")
    inspected = json.loads((directory / "graph_inspection.json").read_text())
    if native:
        qlinears = [n for n in program.graph.nodes if n.op == "call_function"
                    and n.target == torch.ops.atlas.qlinear.default]
        if len(qlinears) != inspected.get("native_fp8_sites") or not qlinears:
            raise FXVerificationError("saved graph's atlas.qlinear count differs from its inspection record")
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
            **({"native_fp8_ops_registered": True, "atlas_qlinear_nodes": len(qlinears)} if native else {}),
            "code_scale_bytes_preserved": True, "unused_float_weights_not_materialized": True,
            "model_replay_modules_imported": [name for name in ("exporter.load_quantized", "reference.ref_profile", "reference.ref_attention")
                                              if name in sys.modules]}


def capture_fx(module, cases, directory, *, policy, provenance=None, strict=True, fresh_process=True,
               native_fp8=None):
    """Capture fixed-shape reference replay, save, reload and fail closed on mismatch.

    native_fp8: None keeps the decode-based graph. "uint8" or "float8_e4m3fn" captures each candidate
    Linear (native_candidates) as atlas.quantize_act + atlas.qlinear with its code buffer held in that
    dtype (the same bytes); the decode-based graph is also captured in memory and must give
    byte-identical outputs. Conv2d, attention and other sites stay decode-based and are listed."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "running", "torch": torch.__version__,
              "strict": strict, "capture_mode": "ref_conv framework reference emulation",
              "native_fp8": native_fp8,
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
            native_info, baseline = None, None
            with _meta_placeholders(module):
                if native_fp8:
                    if native_fp8 not in STORAGE_DTYPES:
                        raise FXVerificationError(f"unknown native storage {native_fp8!r}")
                    baseline = torch.export.export(module, cases[0], strict=strict)
                    for args, (outputs, spec) in zip(cases, expected):
                        _compare(baseline.module()(*args), outputs, spec)
                    with _native_linears(module, policy, native_fp8) as rules:
                        if not rules:
                            raise FXVerificationError("--native-fp8: no Linear site can be carried natively")
                        held = _inventory(module.named_buffers(), STORAGE_DTYPES[native_fp8])
                        program = torch.export.export(module, cases[0], strict=strict)
                    # the held buffers are a bitcast: same bytes, so same hash and shape as the uint8 codes
                    if any(held[k]["sha256"] != before[k]["sha256"] or held[k]["shape"] != before[k]["shape"]
                           for k in before):
                        raise FXVerificationError("native buffers differ in bytes from the uint8 codes")
                    native_info = {"storage": native_fp8, "ops": ["atlas.quantize_act", "atlas.qlinear"],
                                   "ops_module": "exporter.atlas_fx_ops", "weight_format": "atlas_e4m3",
                                   "sites": sorted(rules),
                                   "code_bytes_equal_uint8_source": True}
                    before = held
                else:
                    program = torch.export.export(module, cases[0], strict=strict)
            if _inventory(program.named_buffers(), STORAGE_DTYPES[native_fp8] if native_fp8 else torch.uint8) != before:
                raise FXVerificationError("capture changed code/exponent buffers")
            graph_report = inspect_graph(program, module, policy, rules if native_fp8 else None,
                                         native_fp8 or "uint8")
            if native_fp8:
                for args, (outputs, spec) in zip(cases, expected):
                    _compare(program.module()(*args), _leaves(baseline.module()(*args))[0], spec)
                native_info["outputs_identical_to_decode_graph"] = True
                native_info["decode_graph_nodes"] = len(baseline.graph.nodes)
                native_info["native_graph_nodes"] = len(program.graph.nodes)
                report["native_fp8_detail"] = native_info
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
              **({"native_fp8": native_info} if native_info else {}),
              "files": {name: _file_record(directory / name) for name in BUNDLE_FILES}})
        report["same_process_reload"] = verify_saved(directory, str(cases[0][0].device))
        if fresh_process:
            result = subprocess.run([sys.executable, "-m", "exporter.fx_export", "--verify-dir", str(directory),
                                     "--device", str(cases[0][0].device)], capture_output=True, text=True,
                                    env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                        [str(Path(__file__).resolve().parent.parent), os.environ.get("PYTHONPATH", "")])})
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
    parser.add_argument("--native-fp8", action="store_true",
                        help="carry each eligible quantized Linear as atlas.quantize_act + atlas.qlinear on its "
                             "code/exponent buffers (src/exporter/atlas_fx_ops.py); the default is the decode-based graph")
    parser.add_argument("--native-storage", choices=sorted(STORAGE_DTYPES), default="uint8",
                        help="dtype of the held weight-code buffers with --native-fp8. uint8 (default) keeps the "
                             "export bytes; float8_e4m3fn is a bitcast of the same bytes, refused when a layer "
                             "holds a code on which torch and ref_numerics decode differently")
    parser.add_argument("--verify-dir", help="fresh-process replay of an existing FX bundle")
    args = parser.parse_args(argv)
    if args.verify_dir:
        report = verify_saved(args.verify_dir, args.device)
        _json(Path(args.verify_dir) / "fresh_process_verification.json", report)
    else:
        if not args.export_dir or not args.out_dir or args.seqlen < 2:
            parser.error("--export-dir, --out-dir and --seqlen >= 2 are required")
        import exporter.load_quantized as load_quantized
        import exporter.quant_export as quant_export

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
                            native_fp8=args.native_storage if args.native_fp8 else None,
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
    root = Path(__file__).resolve().parent.parent.parent
    result = {"fx_export_py": _file_record(Path(__file__))}
    for key, command in (("commit", ["rev-parse", "HEAD"]), ("dirty", ["status", "--porcelain"])):
        process = subprocess.run(["git", "-C", str(root), *command], capture_output=True, text=True)
        result[key] = process.stdout.strip() if process.returncode == 0 else None
    return result


if __name__ == "__main__":
    raise SystemExit(main())
