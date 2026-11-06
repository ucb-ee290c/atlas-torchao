"""Capture reference export replay as a saved, checked FX ExportedProgram.

This is reference emulation, not native-FP8 MLIR or device code. No calibration
or quantizer search runs here. The initial CLI boundary is causal-LM logits.

Default: each quantized Linear is decoded (uint8 codes -> E4M3 value, times 2^exp) and contracted in
BF16. With --native-fp8 each eligible Linear is instead two opaque nodes, atlas.quantize_act and
atlas.qlinear (src/atlas_fx_ops.py), consuming the code and exponent buffers directly; attention
operands, Conv2d and other sites stay decode-based and are listed in graph_inspection.json under
still_emulated. A saved native bundle needs `import atlas_fx_ops` before torch.export.load.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import operator
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


def _schema_args(node):
    schema = getattr(node.target, "_schema", None)
    if schema is None or node.op != "call_function":
        return []
    return list(zip(node.args, schema.arguments))


def _propagate_mutation(node, depends_on_input):
    """An in-place op (`res = empty_like(x); res[..., :d] = a; res[..., d:] = b`, as SmolVLA's apply_rope)
    makes the tensor it writes, and the base the written view aliases, depend on the values written into
    it, although no FX edge says so. Mark them, so a later use of the buffer counts as input-dependent."""
    written = [arg for arg, spec in _schema_args(node)
               if isinstance(arg, torch.fx.Node) and spec.alias_info is not None and spec.alias_info.is_write]
    if not written:
        return
    sources = [arg for arg, spec in _schema_args(node) if isinstance(arg, torch.fx.Node) and arg not in written]
    if not any(depends_on_input.get(source, False) for source in _value_parents_of(sources)):
        return
    for target in written:
        while isinstance(target, torch.fx.Node):
            depends_on_input[target] = True
            returns = getattr(getattr(target.target, "_schema", None), "returns", [])
            is_view = bool(returns) and returns[0].alias_info is not None and not returns[0].alias_info.is_write
            target = target.args[0] if is_view and target.args else None


def _value_parents_of(nodes):
    return [n for n in nodes if any(isinstance(v, torch.Tensor) for v in _pytree.tree_leaves(n.meta.get("val")))]


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


class NativeRule(tuple):
    """(mode, reduce, rule, static_exp) of one quantize_act call, plus `.rows` (NEW, the group size of
    reduce='block' / 'head_block'; 32 otherwise). A 4-tuple, so it unpacks and compares as before."""

    def __new__(cls, mode, reduce, rule, static_exp, rows=32):
        obj = super().__new__(cls, (mode, reduce, rule, static_exp))
        obj.rows = int(rows)
        return obj


BLOCK_REDUCES = ("block", "head_block")


class ActStats:
    """Observed quantize_act results of the eager capture cases, per site (NEW; informational): dynamic
    exponent E8M0 code range, reserved codes, exponent-field-0 (flushed) fraction, and for block
    reductions the constancy of the exponent within each group, the token split and the padding check."""

    def __init__(self):
        self.sites = {}

    def record(self, label, x, rule, codes, exp):
        import tile_native as tn

        e = self.sites.setdefault(label, {"calls": 0, "e8m0_code_min": None, "e8m0_code_max": None,
                                          "reserved_codes": 0, "flushed_fraction_sum": 0.0, "reduce": rule[1],
                                          "mode": rule[0], "groups_constant": True, "token_splits": [],
                                          "group_flush": None})
        c = tn.code_range(exp)
        e["calls"] += 1
        e["e8m0_code_min"] = c["min"] if e["e8m0_code_min"] is None else min(e["e8m0_code_min"], c["min"])
        e["e8m0_code_max"] = c["max"] if e["e8m0_code_max"] is None else max(e["e8m0_code_max"], c["max"])
        raw = codes.view(torch.uint8)
        e["reserved_codes"] += tn.reserved_code_count(raw)
        e["flushed_fraction_sum"] += tn.flushed_fraction(raw)
        if rule[1] in BLOCK_REDUCES:
            g = tn.row_group_flush_summary(raw, rule.rows, per_matrix=rule[1] == "head_block")
            e["group_flush"] = g if e["group_flush"] is None else {
                "rows": g["rows"], "groups": e["group_flush"]["groups"] + g["groups"],
                "min": min(e["group_flush"]["min"], g["min"]), "max": max(e["group_flush"]["max"], g["max"]),
                "mean": (e["group_flush"]["mean"] * e["group_flush"]["groups"] + g["mean"] * g["groups"])
                        / (e["group_flush"]["groups"] + g["groups"])}
            layout = "per_matrix" if rule[1] == "head_block" else "flat"
            chk = tn.check_constant_groups(exp, rule.rows, layout)
            e["groups_constant"] = e["groups_constant"] and chk.ok
            m = x.shape[-2] if layout == "per_matrix" else x.numel() // max(1, x.shape[-1])
            split = tn.token_split_report(m, rule.rows)
            split["padding_encodes_to_plus_zero"] = tn.padding_encodes_to_plus_zero(
                min(int(x.shape[-1]), 64), rule.rows, int(exp.reshape(-1)[-1]))
            e["token_splits"].append(split)

    def summary(self):
        out = {}
        for label, e in sorted(self.sites.items()):
            row = {k: v for k, v in e.items() if k not in ("flushed_fraction_sum", "token_splits", "group_flush")}
            if e["group_flush"] is not None:
                row["flushed_fraction_per_group"] = e["group_flush"]       # per group of `rows` rows, all calls
            row["flushed_fraction_mean"] = e["flushed_fraction_sum"] / max(1, e["calls"])
            if e["token_splits"]:
                uniq = {json.dumps(t, sort_keys=True) for t in e["token_splits"]}
                row["token_splits"] = [json.loads(t) for t in sorted(uniq)]
                row["token_split_all_32_aligned"] = all(t["aligned_32"] for t in e["token_splits"])
            out[label] = row
        return out


_ACT_RECORDER = None          # set by capture_fx for the eager statistics pass only


def _qact(x, rule, storage, tag=None, label=None):
    """torch.ops.atlas.quantize_act(x, *rule, storage[, tag]); `rows` is passed (after tag) only for the
    block reductions, so every other call keeps its old argument list. With an active ActStats recorder
    (eager pass only) the result is observed under `label`."""
    mode, reduce, rule_name, exp = rule
    if reduce in BLOCK_REDUCES:
        out = torch.ops.atlas.quantize_act(x, mode, reduce, rule_name, exp, storage, tag or "", rule.rows)
    elif tag is None:
        out = torch.ops.atlas.quantize_act(x, mode, reduce, rule_name, exp, storage)
    else:
        out = torch.ops.atlas.quantize_act(x, mode, reduce, rule_name, exp, storage, tag)
    if _ACT_RECORDER is not None and label is not None:
        _ACT_RECORDER.record(label, x, rule, *out)
    return out


def native_rule(rule_or_site, in_features):
    """(mode, reduce, rule, static_exp) of a Linear / Conv2d site's activation rule (a site dict or its
    `activation` rule), or None when atlas::quantize_act does not express it (block scales, per-step
    exponents). `in_features` is K, the length of the reduced row."""
    rule = rule_or_site.get("activation", rule_or_site)
    if rule.get("source") == "reference":
        return NativeRule("dynamic", "tensor", "ref", None)
    if rule["mode"] == "static":
        return NativeRule("static", "tensor", "ref", int(rule["exp"]))
    if rule["mode"] != "dynamic":
        return None
    reduce = rule["reduce"]
    if reduce == "block32_or_token":
        if in_features % int(rule.get("block", 32)) == 0:
            return None
        reduce = "token"
    return (NativeRule("dynamic", reduce, rule["rule"], None, rule.get("rows", 32))
            if reduce in ("tensor", "token", "block") else None)


def native_operand_rule(rule):
    """The same for an attention operand (q, k, p, v): the reference rule and `reference_dims` are one
    exponent per (batch, head) matrix ("head"); `token` is one per row."""
    if rule.get("source") == "reference":
        return NativeRule("dynamic", "head", "ref", None)
    if rule["mode"] == "static":
        return NativeRule("static", "tensor", "ref", int(rule["exp"]))
    if rule["mode"] != "dynamic":
        return None
    reduce = {"reference_dims": "head", "token": "token", "tensor": "tensor",
              "head_block": "head_block"}.get(rule["reduce"])
    return NativeRule("dynamic", reduce, rule["rule"], None, rule.get("rows", 32)) if reduce else None


def _without(inventory, unused):
    """An inventory minus the buffers of the quantized modules the boundary never calls."""
    return {name: value for name, value in inventory.items()
            if not any(name.startswith(path + ".") for path in unused)}


def _is_quant_layer(layer):
    return type(layer).__name__ in RECOGNIZED_QUANT_MODULES


def native_candidates(module, policy):
    """{path: (layer, rule)} of the Linear and Conv2d sites that one atlas.qlinear / atlas.qconv2d node can
    carry; the rest (per-block weight scales, residual passes, MXU models, unsupported rules) stay on the
    decode path."""
    result = {}
    for path, layer in module.named_modules():
        if not _is_quant_layer(layer):
            continue
        site = policy["sites"].get(layer.layer_name)
        if (site is None or site["status"] != "quantized" or layer.fm is not None
                or getattr(layer, "residual", None) or layer.weight_block):
            continue
        conv = type(layer).__name__ == "RefQuantConv2d"
        if conv:
            geometry = site.get("conv") or {}
            if any(list(getattr(layer, key)) != list(geometry.get(key, [])) for key in
                   ("kernel_size", "stride", "padding", "dilation")):
                continue
        rule = native_rule(site, layer.weight_codes.shape[-1])
        out = layer.out_channels if conv else layer.out_features
        if rule is not None and layer.weight_scale_exp.numel() in (1, out):
            result[path] = (layer, rule)
    return result


def native_attention(module, policy):
    """{"paths": {module path: site}, "sites": {site: {operand: rule}}} when every attention module in
    `module` can run Q.K and P.V as atlas.qk_matmul / atlas.pv_matmul (framework accumulation, operand
    rules quantize_act expresses), else None (all attention then stays on the decode path)."""
    paths = {path: layer._atlas_ref_attn for path, layer in module.named_modules()
             if isinstance(getattr(layer, "_atlas_ref_attn", None), dict)}
    if not paths:
        return None
    sites = {}
    for name, site in policy["sites"].items():
        if site["status"] == "quantized" and site["kind"] == "attention":
            rules = {op: native_operand_rule(rule) for op, rule in site["operands"].items()}
            if any(rule is None for rule in rules.values()):
                return None
            sites[name] = rules
    for cfg in paths.values():
        if cfg.get("fm_qk") is not None or cfg.get("fm_av") is not None:
            return None
        if cfg.get("algo") is not None and cfg.get("site") not in sites:
            return None
    return {"paths": {path: cfg.get("site") for path, cfg in paths.items()}, "sites": sites}


REFERENCE_OPERAND = NativeRule("dynamic", "head", "ref", None)


@contextlib.contextmanager
def _native_sites(module, policy, storage, *, attention=True, conv=True):
    """For the capture only: route each candidate Linear through atlas.quantize_act + atlas.qlinear, each
    Conv2d through im2col + quantize_act + atlas.qconv2d, and each attention module's Q.K and P.V through
    quantize_act + atlas.qk_matmul / atlas.pv_matmul (softmax and masks stay ordinary nodes); for float8
    storage, hold the weight codes as a float8_e4m3fn bitcast of the same bytes. The caller's forwards,
    buffers and ref_attention functions are restored afterwards. Yields the plan:
    {"linear": {path: rule}, "conv2d": {path: rule}, "attention": native_attention(...) or None}."""
    import atlas_fx_ops
    import ref_attention

    code_dtype = STORAGE_DTYPES[storage]
    candidates = native_candidates(module, policy)
    undo, patched = [], []
    plan = {"linear": {}, "conv2d": {}, "attention": native_attention(module, policy) if attention else None}
    try:
        for path, (layer, nrule) in candidates.items():
            is_conv = type(layer).__name__ == "RefQuantConv2d"
            if is_conv and not conv:
                continue
            if storage != "uint8":
                bad = _code_hazards(layer.weight_codes)
                if bool(bad.any()):
                    raise FXVerificationError(
                        f"{path}: {int(bad.sum())} weight codes (0x7F/0xFF or exponent-0 non-zero) decode "
                        f"differently as {storage} than in ref_numerics; use --native-storage uint8")
            undo.append((layer, layer._buffers["weight_codes"], "forward" in layer.__dict__))
            layer._buffers["weight_codes"] = layer.weight_codes.view(code_dtype)
            out_dtype = str(layer.weight.dtype).replace("torch.", "")
            plan["conv2d" if is_conv else "linear"][path] = nrule

            def linear_forward(x, layer=layer, nrule=nrule, out_dtype=out_dtype):
                x = x.to(layer.weight.dtype)
                codes, scale = _qact(x, nrule, storage, label=layer.layer_name)
                return torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp,
                                               layer.bias, out_dtype, atlas_fx_ops.WEIGHT_FORMAT)

            def conv_forward(x, layer=layer, nrule=nrule, out_dtype=out_dtype):
                # RefQuantConv2d.forward up to the quantizer: im2col stays visible (aten.im2col, permute)
                B, _, H, W = x.shape
                oh = (H + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
                ow = (W + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
                cols = torch.nn.functional.unfold(x, kernel_size=layer.kernel_size, stride=layer.stride,
                                                  padding=layer.padding, dilation=layer.dilation)
                x_flat = cols.permute(0, 2, 1).reshape(B * oh * ow, -1).to(layer.weight.dtype)
                codes, scale = _qact(x_flat, nrule, storage, label=layer.layer_name)
                return torch.ops.atlas.qconv2d(codes, scale, layer.weight_codes, layer.weight_scale_exp,
                                               layer.bias, out_dtype, atlas_fx_ops.WEIGHT_FORMAT, B, oh, ow)
            layer.forward = conv_forward if is_conv else linear_forward
        if plan["attention"]:
            sites = plan["attention"]["sites"]
            original_cast, original_mm = ref_attention._cast, ref_attention._mm

            def cast(algo, site, operand, x, dims):
                nrule = REFERENCE_OPERAND if algo is None else sites[site][operand]
                return _qact(x, nrule, storage, f"{site}:{operand}", label=f"{site}:{operand}")

            def mm(fm, a_q, a_s, b_q, b_s, dtype_out=torch.float32, role=None):
                if fm is not None or role not in ("qk", "pv"):
                    return original_mm(fm, a_q, a_s, b_q, b_s, dtype_out, role)
                op = torch.ops.atlas.qk_matmul if role == "qk" else torch.ops.atlas.pv_matmul
                return op(a_q, a_s, b_q, b_s)
            patched = [("_cast", original_cast), ("_mm", original_mm)]
            ref_attention._cast, ref_attention._mm = cast, mm
        yield plan
    finally:
        for name, original in patched:
            setattr(ref_attention, name, original)
        for layer, codes, had_forward in undo:
            layer._buffers["weight_codes"] = codes
            if not had_forward:
                layer.__dict__.pop("forward", None)


ALLOWED_NATIVE_SCOPE_TARGETS = {operator.getitem, torch.ops.aten._to_copy.default, torch.ops.aten.to.dtype,
                                torch.ops.aten.to.dtype_layout, torch.ops.aten._assert_tensor_metadata.default}
CONV_SCOPE_TARGETS = {torch.ops.aten.im2col.default, torch.ops.aten.permute.default,
                      torch.ops.aten.reshape.default, torch.ops.aten.view.default,
                      torch.ops.aten._unsafe_view.default, torch.ops.aten.clone.default,
                      torch.ops.aten.contiguous.default}
DECODE_TARGETS = {torch.ops.aten.frexp.Tensor, torch.ops.aten.exp2.default, torch.ops.aten.amax.default,
                  torch.ops.aten.bitwise_and.Tensor, torch.ops.aten.bitwise_and.Scalar,
                  torch.ops.aten.bitwise_or.Tensor, torch.ops.aten.bitwise_left_shift.Tensor,
                  torch.ops.aten.bitwise_right_shift.Tensor, torch.ops.aten.bitwise_right_shift.Tensor_Scalar,
                  torch.ops.aten.bitwise_left_shift.Tensor_Scalar, torch.ops.aten.linear.default,
                  torch.ops.aten.matmul.default, torch.ops.aten.mm.default, torch.ops.aten.bmm.default}


def _is_node(value, target=None):
    return isinstance(value, torch.fx.Node) and (target is None or value.target == target)


def _act_source(output, index):
    """The quantize_act node behind getitem(act, index), or None."""
    if (_is_node(output, operator.getitem) and output.args[1] == index
            and _is_node(output.args[0], torch.ops.atlas.quantize_act.default)):
        return output.args[0]
    return None


def _check_act(label, act, rule, storage, depends_on_input):
    if not depends_on_input[act.args[0]]:
        raise FXVerificationError(f"{label}: quantize_act input does not depend on the graph input")
    if tuple(act.args[1:6]) != (*rule, storage):
        raise FXVerificationError(f"{label}: quantize_act arguments {act.args[1:6]} differ from the policy "
                                  f"rule {rule}")
    rows = act.args[7] if len(act.args) > 7 else act.kwargs.get("rows", 32)   # absent in bundles saved before `rows`
    if rule[1] in BLOCK_REDUCES and rows != getattr(rule, "rows", 32):
        raise FXVerificationError(f"{label}: quantize_act rows={rows} differs from the policy rule's "
                                  f"rows={getattr(rule, 'rows', 32)}")


def _inspect_native_site(path, layer, site, rule, nodes, lifted, depends_on_input, storage):
    """One Linear (or Conv2d) = one atlas.qlinear (atlas.qconv2d) node fed directly by the lifted code and
    exponent buffers and by one atlas.quantize_act node whose input depends on the user input; nothing
    else decodes or contracts in its scope (a Conv2d keeps its visible im2col)."""
    import atlas_fx_ops

    conv = type(layer).__name__ == "RefQuantConv2d"
    op = torch.ops.atlas.qconv2d.default if conv else torch.ops.atlas.qlinear.default
    prefix = path + "." if path else ""
    code, exp = lifted.get(prefix + "weight_codes"), lifted.get(prefix + "weight_scale_exp")
    placeholder, bias = lifted.get(prefix + "weight"), lifted.get(prefix + "bias")
    if code is None or exp is None:
        raise FXVerificationError(f"{layer.layer_name}: codes or exponents were not lifted")
    if placeholder is not None and placeholder.users:
        raise FXVerificationError(f"{layer.layer_name}: graph consumes the float placeholder")
    mine = [n for n in nodes if n.op == "call_function" and n.target == op
            and n.args[2] is code and n.args[3] is exp]
    if len(mine) != 1 or set(code.users) != set(mine) or set(exp.users) != set(mine):
        raise FXVerificationError(f"{layer.layer_name}: expected exactly one {op} consuming "
                                  f"the code and exponent buffers directly, found {len(mine)}")
    q = mine[0]
    x_codes, x_exp, _, _, q_bias, out_dtype, weight_format = q.args[:7]
    act = _act_source(x_codes, 0)
    if act is None or act is not _act_source(x_exp, 1):
        raise FXVerificationError(f"{layer.layer_name}: {op} is not fed by one atlas.quantize_act")
    _check_act(layer.layer_name, act, rule, storage, depends_on_input)
    if out_dtype != str(layer.weight.dtype).replace("torch.", "") or weight_format != atlas_fx_ops.WEIGHT_FORMAT:
        raise FXVerificationError(f"{layer.layer_name}: {op} dtype/format attributes differ from the layer")
    if (layer.bias is None) != (q_bias is None) or (bias is not None and q_bias is not bias):
        raise FXVerificationError(f"{layer.layer_name}: bias is not consumed by {op}")
    scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
    allowed = ALLOWED_NATIVE_SCOPE_TARGETS | (CONV_SCOPE_TARGETS if conv else set())
    stray = sorted({str(n.target) for n in scope if n.target not in allowed and n not in (q, act)})
    if stray:
        raise FXVerificationError(f"{layer.layer_name}: decode/contraction ops remain in the layer: {stray}")
    row = {"site": layer.layer_name, "module": path, "kind": "conv2d" if conv else "linear", "native": True,
           "code_buffer": prefix + "weight_codes", "scale_buffer": prefix + "weight_scale_exp",
           "code_dtype": str(layer.weight_codes.dtype).replace("torch.", ""),
           "qlinear" if not conv else "qconv2d": q.name, "quantize_act": act.name,
           "activation_rule": {"mode": rule[0], "reduce": rule[1], "rule": rule[2], "static_exp": rule[3]},
           "nodes_in_scope": [n.name for n in scope]}
    if conv:
        row["im2col_nodes"] = [n.name for n in scope if n.target == torch.ops.aten.im2col.default]
        if len(row["im2col_nodes"]) != 1:
            raise FXVerificationError(f"{layer.layer_name}: expected one im2col node before quantize_act")
    return row


def _inspect_native_attention(plan_attention, nodes, storage, depends_on_input, claimed_acts):
    """Attention: every Q.K / P.V is an atlas.qk_matmul / atlas.pv_matmul whose operands are the codes and
    exponents of quantize_act nodes tagged "<site>:<operand>", carrying the policy's operand rules and
    depending on the user input; the P.V input depends on a Q.K output through the (visible) softmax chain.
    Matched by tag, not by module scope: a module called as `module.forward(...)` (SmolVLA's text/expert
    stack) is not on the nn_module_stack, so its nodes carry the caller's scope. Returns one row per site."""
    qks = [n for n in nodes if n.op == "call_function" and n.target == torch.ops.atlas.qk_matmul.default]
    pvs = [n for n in nodes if n.op == "call_function" and n.target == torch.ops.atlas.pv_matmul.default]
    acts = [n for n in nodes if n.op == "call_function" and n.target == torch.ops.atlas.quantize_act.default
            and n not in claimed_acts]
    sites_rules = plan_attention["sites"]
    covered, per_site = set(), {}
    for matmul, (left, right) in [(n, ("q", "k")) for n in qks] + [(n, ("p", "v")) for n in pvs]:
        tags, names = [], []
        for operand, codes, exp in ((left, matmul.args[0], matmul.args[1]), (right, matmul.args[2], matmul.args[3])):
            if operand == "v" and _is_node(codes, torch.ops.aten.transpose.int):
                codes = codes.args[0]               # P.V takes V^T, a visible transpose of the codes
            act = _act_source(codes, 0)
            if act is None or act is not _act_source(exp, 1) or act in covered:
                raise FXVerificationError(f"{matmul.name}: {operand} is not fed by its own quantize_act")
            covered.add(act)
            site, _, tagged = act.args[6].rpartition(":")
            if tagged != operand:
                raise FXVerificationError(f"{matmul.name}: operand {operand} fed by quantize_act tagged {act.args[6]!r}")
            tags.append(site)
            names.append(act.name)
            rule = REFERENCE_OPERAND if site == "" else sites_rules[site][operand]   # "" = no algorithm: reference rule
            _check_act(f"{matmul.name} {operand}", act, rule, storage, depends_on_input)
        if tags[0] != tags[1]:
            raise FXVerificationError(f"{matmul.name}: operands belong to different sites {tags}")
        entry = per_site.setdefault(tags[0], {"qk_matmul": [], "pv_matmul": [], "quantize_act": []})
        entry["qk_matmul" if matmul.target == torch.ops.atlas.qk_matmul.default else "pv_matmul"].append(matmul.name)
        entry["quantize_act"] += names
    if len(qks) != len(pvs) or covered != set(acts):
        raise FXVerificationError(f"attention: {len(qks)} Q.K, {len(pvs)} P.V and {len(acts)} unclaimed quantize_act nodes "
                                  f"do not pair up ({len(covered)} covered)")
    for pv in pvs:
        p_act = _act_source(pv.args[0], 0)
        if not any(n.target == torch.ops.atlas.qk_matmul.default for n in _ancestors(p_act, value_only=True)):
            raise FXVerificationError(f"P of {pv.name} does not depend on a Q.K output")
    rows = []
    for path, site in plan_attention["paths"].items():
        key = site if site is not None else ""
        entry = per_site.get(key)
        if entry is None:
            raise FXVerificationError(f"attention site {site!r} ({path}) has no atlas.qk_matmul in the graph")
        rules = sites_rules[site] if site is not None else dict.fromkeys("qkpv", REFERENCE_OPERAND)
        scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
        own_scope = any(n.name in entry["qk_matmul"] for n in scope)
        remaining = {}
        if own_scope:
            native = set(entry["qk_matmul"] + entry["pv_matmul"] + entry["quantize_act"])
            for n in scope:
                if n.name not in native and n.target is not operator.getitem:
                    remaining[str(n.target)] = remaining.get(str(n.target), 0) + 1
            stray = sorted({str(n.target) for n in scope if n.target in DECODE_TARGETS})
            if stray:
                raise FXVerificationError(f"{path}: decode/contraction ops remain in the attention module: {stray}")
        rows.append({"site": site, "module": path, "pairs": len(entry["qk_matmul"]),
                     "qk_matmul": entry["qk_matmul"], "pv_matmul": entry["pv_matmul"],
                     "quantize_act": sorted(set(entry["quantize_act"])),
                     "operand_rules": {op: {"mode": r[0], "reduce": r[1], "rule": r[2], "static_exp": r[3]}
                                       for op, r in rules.items()},
                     "remaining_nodes_in_module_by_target": dict(sorted(remaining.items())) if own_scope else None,
                     "remaining_node_count": sum(remaining.values()) if own_scope else None,
                     "scope_note": None if own_scope else "module is called as module.forward(), so its nodes carry the "
                                                          "caller's scope; per-module remaining nodes are not attributed"})
    return rows


def _scope_counts(decode, native, plan):
    """call_function node counts inside the native sites' modules, decode-based graph vs native graph, per
    kind. Attention scopes also hold ordinary BF16 nodes (RoPE, mask, softmax) that both graphs keep."""
    def counts(program):
        result = {}
        for node in program.graph.nodes:
            if node.op == "call_function":
                result[_scope(node)] = result.get(_scope(node), 0) + 1
        return result
    before, after = counts(decode), counts(native)
    groups = {"linear": list(plan["linear"]), "conv2d": list(plan["conv2d"]),
              "attention": list(plan["attention"]["paths"]) if plan["attention"] else []}
    return {kind: {"modules": len(paths), "decode_nodes": sum(before.get(p, 0) for p in paths),
                   "native_nodes": sum(after.get(p, 0) for p in paths)} for kind, paths in groups.items()}


def _prefix_of(module, policy):
    """The path prefix the capture boundary adds to the policy's site names (e.g. 'model.')."""
    for path, layer in module.named_modules():
        if _is_quant_layer(layer) and path.endswith(layer.layer_name):
            return path[:len(path) - len(layer.layer_name)]
    return ""


def _bf16_by_policy_report(program, nodes, module, policy):
    """Sites the policy deliberately keeps in BF16 (status host / skipped): the module, what ran there and
    whether the graph really holds an ordinary BF16 contraction with its own weight."""
    prefix = _prefix_of(module, policy)
    modules = dict(module.named_modules())
    rows = []
    for name, site in policy["sites"].items():
        if site["status"] not in ("host", "skipped"):
            continue
        path = prefix + name
        layer = modules.get(path)
        if layer is None:
            raise FXVerificationError(f"{name}: BF16-by-policy site has no module at {path}")
        scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
        if any(str(n.target).startswith("atlas.") for n in scope):
            raise FXVerificationError(f"{name}: BF16-by-policy site runs an atlas.* operator")
        if not scope:
            rows.append({"site": name, "module": path, "status": site["status"], "reason": site.get("reason"),
                         "kind": site.get("kind"), "in_graph": False})
            continue
        contractions = [n for n in scope if n.target in CONTRACTIONS]
        weight = program.state_dict.get(path + ".weight")
        if not contractions or weight is None or weight.is_meta:
            raise FXVerificationError(f"{name}: BF16-by-policy site lacks its BF16 contraction or weight")
        rows.append({"site": name, "module": path, "status": site["status"], "reason": site.get("reason"),
                     "kind": site.get("kind"), "in_graph": True, "contraction_nodes": [n.name for n in contractions],
                     "weight_dtype": str(weight.dtype).replace("torch.", ""), "weight_shape": list(weight.shape),
                     "node_count": len(scope)})
    return rows


def _emulated_report(nodes, module, native_paths):
    """Modules whose quantized operations are still emulated (decode and contract), keyed by module path:
    their contraction nodes, the amax / frexp / gt nodes that derive their scales, and the module's total
    node count (which also holds ordinary BF16 nodes such as RoPE and softmax).
    `other_nodes_outside_quantized_modules` counts the remaining call_function nodes (norms, residual
    adds, SiLU, embeddings, BF16-by-policy sites)."""
    def kind(layer):
        if isinstance(getattr(layer, "_atlas_ref_attn", None), dict):
            return "attention"
        return {"RefQuantLinear": "linear", "RefQuantConv2d": "conv2d"}.get(type(layer).__name__)
    emulated = {path: {"kind": kind(layer), "node_count": 0, "contractions": [], "scale_derivation": []}
                for path, layer in module.named_modules() if path not in native_paths and kind(layer)}
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


def _tile_native_report(module, policy, native_paths, plan, act_stats):
    """Per-site `tile_native` (NEW): true when the weight exponent table and the activation / attention
    operand exponents are constant along K at 32-granularity (src/tile_native.py), with the reasons for every
    site that is not (per-channel or per-token recipes, block32 along K). `native_fp8` says whether the site
    is carried by atlas.qlinear / qconv2d / qk_matmul / pv_matmul in this graph."""
    import tile_native as tn

    rows, status = {}, {}
    for path, layer in module.named_modules():
        if not _is_quant_layer(layer):
            continue
        site = policy["sites"].get(layer.layer_name)
        if site is None or site.get("status") != "quantized":
            continue
        w = site["weight"]
        st = tn.linear_site_status(w["granularity"], layer.weight_scale_exp.detach().cpu(),
                                   int(w.get("panel_rows") or 0), site["activation"])
        if "block" in w and w["block"]:
            st["tile_native"] = False
        status[layer.layer_name] = st
        rows[layer.layer_name] = {"module": path, "kind": site["kind"], "tile_native": st["tile_native"],
                                  "reasons": st["reasons"], "weight_granularity": w["granularity"],
                                  "activation": site["activation"].get("label") or site["activation"].get("mode"),
                                  "native_fp8": path in native_paths}
    attn_paths = (plan or {}).get("attention", {}) or {}
    for name, site in policy["sites"].items():
        if site.get("status") == "quantized" and site.get("kind") == "attention":
            st = tn.attention_site_status(site["operands"])
            status[name] = st
            rows[name] = {"module": None, "kind": "attention", "tile_native": st["tile_native"],
                          "reasons": st["reasons"],
                          "operands": {op: r.get("label") or r.get("mode") for op, r in site["operands"].items()},
                          "native_fp8": name in (attn_paths.get("sites") or {})}
    report = {"sites": dict(sorted(rows.items())), **tn.tile_summary(status),
              "fixtures": ("native-FP8 graph: eager capture cases (quantize_act outputs observed)"
                           if act_stats else "not available (decode-based graph has no quantize_act)")}
    if act_stats:
        report["fixture_activation_stats"] = act_stats.summary()
    return report


def _admission_report(policy, act_stats):
    """compiler_pack_scale_admission for graph_inspection.json: the policy's static exponents plus the
    dynamic exponents observed in the capture cases (INFORMATIONAL; nothing is clamped)."""
    import tile_native as tn

    rep = dict(policy.get("compiler_pack_scale_admission") or {})
    if act_stats:
        dyn = {k: (v["e8m0_code_min"], v["e8m0_code_max"]) for k, v in act_stats.summary().items()
               if v["mode"] == "dynamic"}
        flags = tn.pack_admission_flags(dyn)
        rep["dynamic_activation_scales"] = {
            "source": "capture-case fixtures only (data dependent)", "sites_checked": flags["sites_checked"],
            "sites_outside": flags["sites_outside"], "offending_sites": flags["offending_sites"],
            "codes": {k: {"min": a, "max": b} for k, (a, b) in sorted(dyn.items())}}
    return rep


def inspect_graph(program, module, policy, plan=None, storage="uint8", unused=(), act_stats=None):
    """Verify real contraction dependencies, not buffer labels alone.

    plan: what `_native_sites` yielded ({"linear": {path: rule}, "conv2d": {path: rule}, "attention":
    {...} or None}) for a --native-fp8 capture. Those sites are checked by _inspect_native_site /
    _inspect_native_attention, the rest by the decode-graph rules below. `storage` is the code buffer
    dtype name the native nodes were captured with. `unused`: module paths of quantized layers the boundary
    never calls (their buffers are not in the program); each must really be absent from the graph."""
    native_paths = {**plan["linear"], **plan["conv2d"]} if plan else {}
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
        _propagate_mutation(node, depends_on_input)
    contractions = [node for node in nodes if node.op == "call_function" and node.target in CONTRACTIONS]
    weight_dependencies = {node: _ancestors(node.args[CONTRACTIONS[node.target][1]],
                                          stop_at_contractions=True, value_only=True)
                           for node in contractions}
    selected = {name: site for name, site in policy["sites"].items()
                if site["status"] == "quantized" and site["kind"] in ("linear", "conv2d")}
    actual, rows, unused_rows = set(), [], []
    for path, layer in module.named_modules():
        if type(layer).__name__ not in ("RefQuantLinear", "RefQuantConv2d"):
            continue
        site = layer.layer_name
        if site not in selected or site in actual:
            raise FXVerificationError(f"unexpected or duplicated quantized module: {site}")
        actual.add(site)
        if path in unused:
            prefix = path + "."
            if lifted.get(prefix + "weight_codes") is not None or any(_scope(n) == path for n in nodes):
                raise FXVerificationError(f"{site}: never called by the eager replay but present in the graph")
            unused_rows.append({"site": site, "module": path})
            continue
        if path in native_paths:
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
    unused_params = [spec.target for spec in program.graph_signature.input_specs
              if spec.kind.name == "PARAMETER" and isinstance(spec.target, str)
              and not by_name[spec.arg.name].users]
    for path, layer in module.named_modules():
        if type(layer).__name__ in ("RefQuantLinear", "RefQuantConv2d"):
            name = (path + "." if path else "") + "weight"
            if name in program.state_dict and not program.state_dict[name].is_meta:
                raise FXVerificationError(f"{name}: unused float placeholder payload was retained")
    attention_rows = []
    if plan and plan["attention"]:
        claimed = {by_name[row["quantize_act"]] for row in rows if row.get("native")}
        attention_rows = _inspect_native_attention(plan["attention"], nodes, storage, depends_on_input, claimed)
    report = {"weight_site_bindings": rows, "weight_sites_verified": len(rows),
              "unused_quantized_modules": unused_rows,
              "bf16_by_policy": _bf16_by_policy_report(program, nodes, module, policy),
              "functional_attention_sites": attention,
              "functional_attention_structural_binding": "not certified; covered by boundary output parity",
              "unused_parameters": unused_params,
              "unused_quantized_weight_storage": "meta placeholders; no serialized float weight payload",
              "scope": "reference-emulation FX; not native-FP8 MLIR or hardware compatibility"}
    report["tile_native"] = _tile_native_report(module, policy, native_paths, plan, act_stats)
    report["compiler_pack_scale_admission"] = _admission_report(policy, act_stats)
    if plan:
        native_rows = [row for row in rows if row.get("native")]
        native_modules = {row["module"] for row in native_rows} | {row["module"] for row in attention_rows}
        report["native_fp8_sites"] = len(native_rows)
        report["native_fp8_linear_sites"] = sum(row["kind"] == "linear" for row in native_rows)
        report["native_fp8_conv2d_sites"] = sum(row["kind"] == "conv2d" for row in native_rows)
        report["native_fp8_attention_modules"] = attention_rows
        report["still_emulated"] = _emulated_report(nodes, module, native_modules | set(unused))
        if not report["still_emulated"]["emulated_modules"]:
            # nothing is left to emulate, so no contraction may be fed by a decode (exp2 / frexp) anywhere
            for node in contractions:
                decoders = [n for n in _ancestors(node, stop_at_contractions=True, value_only=True)
                            if n.target in (torch.ops.aten.exp2.default, torch.ops.aten.frexp.Tensor)]
                if decoders:
                    raise FXVerificationError(f"{node.name} ({_scope(node)}): a contraction is still fed by "
                                              f"{decoders[0].name}, a decode op, with no emulated module listed")
        report["scope"] = ("native-FP8-carrying FX (atlas.qlinear / qconv2d / qk_matmul / pv_matmul custom ops) for "
                           "the listed sites; everything in still_emulated stays decode-and-contract reference "
                           "emulation and bf16_by_policy lists sites the policy keeps in BF16; not native-FP8 "
                           "MLIR or hardware compatibility")
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
        import atlas_fx_ops  # noqa: F401
    program = torch.export.load(directory / "model.pt2")
    expected_inventory = json.loads((directory / "buffer_inventory.json").read_text())
    code_dtype = STORAGE_DTYPES[native["storage"]] if native else torch.uint8
    if not expected_inventory or _inventory(program.named_buffers(), code_dtype) != expected_inventory:
        raise FXVerificationError("saved code/exponent inventory differs")
    inspected = json.loads((directory / "graph_inspection.json").read_text())
    if native:
        counts = {name: sum(1 for n in program.graph.nodes if n.op == "call_function"
                            and n.target == getattr(torch.ops.atlas, name).default)
                  for name in ("qlinear", "qconv2d", "qk_matmul", "pv_matmul")}
        pairs = sum(row["pairs"] for row in inspected.get("native_fp8_attention_modules", []))
        if (counts["qlinear"] != inspected.get("native_fp8_linear_sites")
                or counts["qconv2d"] != inspected.get("native_fp8_conv2d_sites")
                or counts["qk_matmul"] != pairs or counts["pv_matmul"] != pairs
                or not any(counts.values())):
            raise FXVerificationError(f"saved graph's atlas operator counts {counts} differ from its inspection record")
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
            **({"native_fp8_ops_registered": True, "atlas_qlinear_nodes": counts["qlinear"],
                                                          "atlas_qconv2d_nodes": counts["qconv2d"],
                                                          "atlas_qk_matmul_nodes": counts["qk_matmul"],
                                                          "atlas_pv_matmul_nodes": counts["pv_matmul"]} if native else {}),
            "code_scale_bytes_preserved": True, "unused_float_weights_not_materialized": True,
            "model_replay_modules_imported": [name for name in ("load_quantized", "ref_profile", "ref_attention")
                                              if name in sys.modules]}


def _observe_acts(module, cases):
    """Eager pass over the capture cases with the native quantize_act forwards installed, recording each
    activation cast (ActStats). Run inside _native_sites; never under tracing."""
    global _ACT_RECORDER
    _ACT_RECORDER = stats = ActStats()
    try:
        with torch.no_grad():
            for args in cases:
                module(*args)
    finally:
        _ACT_RECORDER = None
    return stats


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
        # a quantized layer the boundary never calls has no node in the graph and its buffers are not lifted;
        # that is recorded (and checked against the eager replay), not silently tolerated
        called = set()
        hooks = [layer.register_forward_hook(lambda mod, inp, out, path=path: called.add(path))
                 for path, layer in module.named_modules() if _is_quant_layer(layer)]
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
            for hook in hooks:
                hook.remove()
            unused = sorted(path for path, layer in module.named_modules()
                            if _is_quant_layer(layer) and path not in called)
            report["unused_quantized_modules"] = unused
            before = _without(_inventory(module.named_buffers()), unused)
            if not before:
                raise FXVerificationError("no quantized weight buffers at this boundary")
            native_info, baseline, act_stats = None, None, None
            with _meta_placeholders(module):
                if native_fp8:
                    if native_fp8 not in STORAGE_DTYPES:
                        raise FXVerificationError(f"unknown native storage {native_fp8!r}")
                    baseline = torch.export.export(module, cases[0], strict=strict)
                    for args, (outputs, spec) in zip(cases, expected):
                        _compare(baseline.module()(*args), outputs, spec)
                    with _native_sites(module, policy, native_fp8) as plan:
                        if not (plan["linear"] or plan["conv2d"] or plan["attention"]):
                            raise FXVerificationError("--native-fp8: no Linear site can be carried natively")
                        held = _without(_inventory(module.named_buffers(), STORAGE_DTYPES[native_fp8]), unused)
                        act_stats = _observe_acts(module, cases)
                        program = torch.export.export(module, cases[0], strict=strict)
                    # the held buffers are a bitcast: same bytes, so same hash and shape as the uint8 codes
                    if any(held[k]["sha256"] != before[k]["sha256"] or held[k]["shape"] != before[k]["shape"]
                           for k in before):
                        raise FXVerificationError("native buffers differ in bytes from the uint8 codes")
                    native_info = {"storage": native_fp8, "ops": ["atlas.quantize_act", "atlas.qlinear", "atlas.qconv2d", "atlas.qk_matmul",
                                           "atlas.pv_matmul"],
                                   "ops_module": "atlas_fx_ops", "weight_format": "atlas_e4m3",
                                   "sites": sorted(set({**plan["linear"], **plan["conv2d"]}) - set(unused)),
                                   "conv2d_sites": sorted(set(plan["conv2d"]) - set(unused)),
                                   "attention_modules": sorted(plan["attention"]["paths"]) if plan["attention"] else [],
                                   "code_bytes_equal_uint8_source": True}
                    before = held
                else:
                    program = torch.export.export(module, cases[0], strict=strict)
            if _without(_inventory(program.named_buffers(), STORAGE_DTYPES[native_fp8] if native_fp8 else torch.uint8),
                        unused) != before:
                raise FXVerificationError("capture changed code/exponent buffers")
            graph_report = inspect_graph(program, module, policy, plan if native_fp8 else None,
                                         native_fp8 or "uint8", unused=unused, act_stats=act_stats)
            if native_fp8:
                for args, (outputs, spec) in zip(cases, expected):
                    _compare(program.module()(*args), _leaves(baseline.module()(*args))[0], spec)
                native_info["outputs_identical_to_decode_graph"] = True
                native_info["node_counts_by_kind"] = _scope_counts(baseline, program, plan)
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


class SmolVLAFlowLoss(torch.nn.Module):
    """Tensor-in / tensor-out boundary of a LeRobot SmolVLA policy: the flow-matching loss forward
    (`VLAFlowMatching.forward`) for given images, language tokens, state, actions, noise and time. It runs the
    vision tower (patch-embedding Conv2d, vision attention), the connector, the text/expert layers and the
    action projections once, with no sampling loop, no KV cache and no random draw; the full action
    sampler (10 denoising steps with a KV cache) is a different boundary."""

    def __init__(self, policy, n_images=1):
        super().__init__()
        self.policy = policy
        self.n_images = n_images

    def forward(self, *tensors):
        images = list(tensors[:self.n_images])
        lang_tokens, lang_masks, state, actions, noise, time = tensors[self.n_images:]
        masks = [torch.ones(image.shape[0], dtype=torch.bool, device=image.device) for image in images]
        return self.policy.model.forward(images, masks, lang_tokens, lang_masks, state, actions, noise, time)


def smolvla_cases(policy, n_images, seed, device, count=2, image_size=None):
    """`count` fixed input tuples for SmolVLAFlowLoss (images in [-1, 1], as prepare_images gives)."""
    config = policy.model.config
    size = image_size or tuple(config.resize_imgs_with_padding)
    dtype = next(policy.parameters()).dtype
    cases = []
    for i in range(count):
        g = torch.Generator().manual_seed(seed + i)
        images = [(torch.rand(1, 3, *size, generator=g) * 2 - 1).to(dtype) for _ in range(n_images)]
        length = config.tokenizer_max_length
        case = (*images, torch.randint(0, 1000, (1, length), generator=g), torch.ones(1, length, dtype=torch.bool),
                torch.randn(1, config.max_state_dim, generator=g).to(dtype),
                torch.randn(1, config.chunk_size, config.max_action_dim, generator=g),
                torch.randn(1, config.chunk_size, config.max_action_dim, generator=g), torch.tensor([0.25 + 0.5 * i]))
        cases.append(tuple(value.to(device) for value in case))
    return cases


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
                             "code/exponent buffers (src/atlas_fx_ops.py); the default is the decode-based graph")
    parser.add_argument("--native-storage", choices=sorted(STORAGE_DTYPES), default="uint8",
                        help="dtype of the held weight-code buffers with --native-fp8. uint8 (default) keeps the "
                             "export bytes; float8_e4m3fn is a bitcast of the same bytes, refused when a layer "
                             "holds a code on which torch and ref_numerics decode differently")
    parser.add_argument("--boundary", choices=("causal_lm", "smolvla_flow_loss"), default="causal_lm",
                        help="capture boundary: causal-LM logits (default) or the SmolVLA flow-matching loss forward")
    parser.add_argument("--n-images", type=int, default=1, help="cameras at the smolvla_flow_loss boundary")
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
        kind = (manifest.get("model") or {}).get("kind", "causal_lm")
        if (kind == "causal_lm") != (args.boundary == "causal_lm"):
            parser.error(f"export kind {kind!r} does not match --boundary {args.boundary}; the CLI captures causal-LM "
                         "logits or the SmolVLA flow-loss forward (policy action sampling is a separate boundary)")
        if args.boundary == "smolvla_flow_loss":
            import eval_policy

            model_id = args.model or manifest["model"]["id"]
            lerobot_policy, _ = eval_policy.load_policy(model_id, args.device, torch.bfloat16)
            load_quantized.load_quantized(model_id, args.export_dir, args.device, build=lambda: lerobot_policy,
                                          mode="ref_conv", options={})
            boundary = SmolVLAFlowLoss(lerobot_policy, args.n_images).eval()
            cases = smolvla_cases(lerobot_policy, args.n_images, args.seed, args.device)
            config_dict = __import__("dataclasses").asdict(lerobot_policy.config)
        else:
            model, _, _ = load_quantized.load_quantized(args.model, args.export_dir, args.device,
                                                       mode="ref_conv", seqlen=args.seqlen)
            vocab = model.config.vocab_size
            cases = [(torch.randint(vocab, (1, args.seqlen), generator=torch.Generator().manual_seed(args.seed + i))
                      .to(args.device),) for i in range(2)]
            boundary, config_dict = CausalLogits(model).eval(), model.config.to_dict()
        report = capture_fx(boundary, cases, args.out_dir, policy=policy,
                            strict=not args.non_strict,
                            native_fp8=args.native_storage if args.native_fp8 else None,
                            provenance={"export_dir": str(Path(args.export_dir).resolve()),
                                        "export_manifest_sha256": quant_export.sha256_file(
                                            Path(args.export_dir) / "manifest.json"),
                                        "export_provenance": manifest.get("provenance", {}),
                                        "export_recipe": manifest.get("recipe", {}),
                                        "architecture_locator": args.model or (manifest.get("model") or {}).get("id"),
                                        "boundary": args.boundary,
                                        "config_sha256": hashlib.sha256(json.dumps(config_dict,
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
