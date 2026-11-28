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
import collections
import contextlib
import hashlib
import json
import operator
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils import _pytree

import exporter.pack_plan as pack_plan

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
    torch.ops.aten.conv2d.padding: (0, 1),             # string padding ("valid"), e.g. a SigLIP patch embedding
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



EXPORT_FILES = ("manifest.json", "model.safetensors", "policy.json")


def link_export(export_dir, dest):
    """Place the export's files in `dest` as real files (hard links, or copies across file systems), never a
    symlink: the backend's atlas_fp8 adapter refuses any export path with a symlinked component. Existing
    files must already hold the same bytes. Returns dest."""
    import shutil

    export_dir, dest = Path(export_dir).resolve(), Path(dest)
    if dest.is_symlink():
        raise FXVerificationError(f"{dest} is a symlink; remove it (the backend refuses symlinked export paths)")
    dest.mkdir(parents=True, exist_ok=True)
    for name in EXPORT_FILES:
        source, target = export_dir / name, dest / name
        if target.exists():
            if target.is_symlink() or _file_record(target) != _file_record(source):
                raise FXVerificationError(f"{target} exists and differs from {source}")
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return dest

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



@contextlib.contextmanager
def _capture_buffer_roles(module):
    """For the capture only, give every non-persistent buffer the role the backend intake expects:

    - an exported quantization tensor (name ending in quant_export.QUANT_SUFFIXES: weight codes / exponents,
      static input exponents; non-persistent on the module because quant_export writes them separately) is made
      persistent, so the saved program holds it in `state_dict` like every other export tensor (the backend's
      independent reference interpreter and capsule tooling read lifted buffers from `state_dict`);
    - any other non-persistent buffer (not model state: absent from state_dict and so from the export's tensor
      inventory, e.g. a RoPE cos/sin cache the constructor builds) is held as a plain tensor attribute, so
      torch.export lifts it as a CONSTANT_TENSOR; the atlas_fp8 adapter authenticates every lifted
      BUFFER/PARAMETER against the export tensors and records constants separately as graph literals.

    Same tensors, same values, same graph arithmetic; the caller's buffers are restored afterwards. Yields
    {"persistent": [paths], "constants": [paths]}."""
    from exporter.quant_export import QUANT_SUFFIXES

    persistent, constants, seen = [], [], set()
    try:
        for path, mod in module.named_modules():
            if id(mod) in seen:
                continue
            seen.add(id(mod))
            for name in sorted(mod._non_persistent_buffers_set):
                tensor = mod._buffers.get(name)
                if tensor is None:
                    continue
                full = f"{path}.{name}" if path else name
                mod._non_persistent_buffers_set.discard(name)
                if ("." + name).endswith(QUANT_SUFFIXES):
                    persistent.append((mod, name, full))
                else:
                    del mod._buffers[name]
                    setattr(mod, name, tensor)
                    constants.append((mod, name, tensor, full))
        yield {"persistent": sorted(m[2] for m in persistent), "constants": sorted(m[3] for m in constants)}
    finally:
        for mod, name, _ in persistent:
            mod._non_persistent_buffers_set.add(name)
        for mod, name, tensor, _ in constants:
            mod.__dict__.pop(name, None)
            mod.register_buffer(name, tensor, persistent=False)


@contextlib.contextmanager
def _alias_parameters_as_attributes(module, policy):
    """For the capture only: hold every tied parameter the export records as an alias (policy["aliases"]: alias ->
    source name; quant_export writes the tensor once, under the source name) as a plain attribute of its module.
    torch.export then lists only the source name in the program's state, which is what the backend's atlas_fp8
    adapter authenticates against the export tensors (it has no alias notion). An alias the boundary never reads
    leaves the graph; one it does read is lifted as a CONSTANT_TENSOR with the same values. Restored afterwards.
    Yields the sorted list of alias paths held this way."""
    prefix = _prefix_of(module, policy)
    held = []
    try:
        for alias in sorted((policy.get("aliases") or {})):
            owner, _, name = (prefix + alias).rpartition(".")
            try:
                mod = module.get_submodule(owner)
            except AttributeError:
                continue
            param = mod._parameters.get(name)
            if param is None:
                continue
            del mod._parameters[name]
            mod.__dict__[name] = param
            held.append((mod, name, param, prefix + alias))
        yield sorted(h[3] for h in held)
    finally:
        for mod, name, param, _ in held:
            mod.__dict__.pop(name, None)
            mod._parameters[name] = param


@contextlib.contextmanager
def _inert_autocast_disable():
    """For the capture only: a `torch.autocast(..., enabled=False)` entered while autocast is not active for that
    device (the Gemma / Llama rotary "force float32" blocks on CPU) changes nothing numerically, but torch.export
    records it as a wrap_with_autocast higher-order submodule, whose nodes the backend's frontend trace cannot
    put in correspondence ("quantized -> prepared correspondence incomplete"). Such a context is entered as a
    no-op here; any autocast that would change dtypes runs unchanged. torch.autocast is restored afterwards."""
    original = torch.autocast

    class _Autocast(original):
        def __init__(self, device_type, dtype=None, enabled=True, cache_enabled=None):
            self._inert = not enabled and not torch.is_autocast_enabled(device_type)
            if not self._inert:
                super().__init__(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)

        def __enter__(self):
            return self if self._inert else super().__enter__()

        def __exit__(self, *exc):
            return None if self._inert else super().__exit__(*exc)

    torch.autocast = _Autocast
    try:
        yield
    finally:
        torch.autocast = original


@contextlib.contextmanager
def _mse_loss_without_identity_broadcast():
    """For the capture only: F.mse_loss on two tensors of the same shape calls aten.mse_loss directly instead of
    first running aten.broadcast_tensors, which is then an identity (SmolVLA's flow loss: u_t and v_t are both
    [B, chunk, action_dim]). The backend's frontend trace does not record that eliminated multi-result identity
    ("quantized -> prepared correspondence incomplete"). Same aten.mse_loss call on the same tensors, so the same
    values; any other call (different shapes, legacy size_average / reduce, weight) is unchanged."""
    import torch.nn.functional as F
    from torch.nn.modules.loss import _Reduction

    original = F.mse_loss

    def mse_loss(input, target, size_average=None, reduce=None, reduction="mean", weight=None):
        if (isinstance(input, torch.Tensor) and isinstance(target, torch.Tensor) and input.shape == target.shape
                and size_average is None and reduce is None and weight is None):
            return torch._C._nn.mse_loss(input, target, _Reduction.get_enum(reduction))
        return original(input, target, size_average, reduce, reduction, weight)

    F.mse_loss = mse_loss
    try:
        yield
    finally:
        F.mse_loss = original


BF16_MIN = torch.finfo(torch.bfloat16).min


def additive_attention_mask(keep):
    """Boolean keep mask (True = attend) -> additive BF16 mask: 0 where kept, the BF16 minimum (finite) elsewhere.
    The expression transformers' eager_mask uses, so it captures as TinyLlama's full-model mask does:
    aten.zeros([], bfloat16) and aten.where.ScalarOther(keep, zeros, -3.3895313892515355e+38)."""
    return torch.where(keep, torch.zeros((), device=keep.device, dtype=torch.bfloat16), BF16_MIN)


@contextlib.contextmanager
def _smolvla_additive_attention_mask(module):
    """For the capture only: every SmolVLA text / expert attention softmax takes an additive mask,
    softmax(scores + where(keep, 0, BF16_MIN)), the form TinyLlama's graph has, instead of the boolean select
    softmax(where(keep, scores, finfo(float32).min)) of the reference text attention (ref_attention.py, the
    SmolVLMWithExpertModel.eager_attention_forward replacement). The backend's qualified masked softmax recognizes
    a score + mask addition; the float32 minimum of the select form rounds to -inf in BF16.

    Each SmolVLMWithExpertModel whose attention is the reference routine (`_atlas_ref_attn`) gets an instance
    `forward` that converts its boolean [B, Sq, Skv] attention_mask once (additive_attention_mask; rank kept, the
    layers slice dims 1 and 2) and calls the original forward; ref_attention's text variant then adds the BF16 mask
    to the float32 scores. Kept scores are unchanged (s + 0 = s) and masked ones become about -3.39e38, so a row with
    a kept entry gives the same softmax words as the select form (both exponentials underflow to exactly 0). A fully
    masked row (a padded query) is uniform in both forms in the float32 softmax. Restored afterwards; eager
    evaluation is unchanged. Yields the module paths converted."""
    import types

    patched = []
    try:
        for path, mod in module.named_modules():
            if type(mod).__name__ != "SmolVLMWithExpertModel" or not isinstance(
                    getattr(mod, "_atlas_ref_attn", None), dict):
                continue
            original = mod.forward

            def forward(self, attention_mask=None, *args, _original=original, **kwargs):
                if isinstance(attention_mask, torch.Tensor) and attention_mask.dtype == torch.bool:
                    attention_mask = additive_attention_mask(attention_mask)
                return _original(attention_mask, *args, **kwargs)

            patched.append((mod, mod.__dict__.get("forward"), path))
            mod.forward = types.MethodType(forward, mod)
        yield [p[2] for p in patched]
    finally:
        for mod, previous, _ in patched:
            if previous is None:
                del mod.forward
            else:
                mod.forward = previous


VISION_EMBEDDINGS = ("SmolVLMVisionEmbeddings", "Idefics3VisionEmbeddings")


def _vision_position_ids(embeddings):
    """The position ids the embeddings module's own forward gives a full-size image with an all-true patch mask
    (num_patches_per_side patches per side): run eagerly on a zero image, read at the position_embedding call."""
    n, p = embeddings.num_patches_per_side, embeddings.patch_size
    weight = embeddings.position_embedding.weight
    seen = []
    hook = embeddings.position_embedding.register_forward_pre_hook(lambda m, args: seen.append(args[0].clone()))
    try:
        with torch.no_grad():
            embeddings.forward(torch.zeros(1, embeddings.patch_embedding.weight.shape[1], n * p, n * p,
                                           dtype=embeddings.patch_embedding.weight.dtype, device=weight.device),
                               torch.ones(1, n, n, dtype=torch.bool, device=weight.device))
    finally:
        hook.remove()
    if len(seen) != 1:
        raise FXVerificationError(f"vision embeddings: position_embedding called {len(seen)} times")
    return seen[0]


@contextlib.contextmanager
def _static_vision_position_ids(module):
    """For the capture only: the SmolVLM (Idefics3-style) vision embeddings at full image size take the static path.
    Their forward builds position ids from the patch attention mask (bucketized fractional coordinates, then a
    boolean-mask index_put), which torch.export records with a data-dependent size (tensor<?xi64>). SmolVLA's
    embed_image passes no patch mask, so the vision transformer uses an all-true one, and at
    num_patches_per_side patches per side the ids are 0 .. n*n - 1 in row-major order.

    Checked, not assumed: before the capture each such module's own forward is run eagerly on a full-size image with
    an all-true mask and its ids must equal arange(n*n). position_embedding(arange(n*n)) then gathers every row of the
    table in order, i.e. it is the table itself, so the capture-time forward computes
    patch_embedding(x).flatten(2).transpose(1, 2) + position_embedding.weight[None] (the same words, added with
    broadcast over the batch; no embedding_lookup node), and on every concrete (eager) call it asserts the patch mask
    it was given is all true. Any other input size, or a mask of another shape, runs the original forward (and the
    capture's static-shape check then fails closed). Restored afterwards. Yields the module paths converted."""
    import types

    patched = []
    try:
        for path, mod in module.named_modules():
            if type(mod).__name__ not in VISION_EMBEDDINGS:
                continue
            n = mod.num_patches_per_side
            ids = _vision_position_ids(mod)
            static = torch.arange(n * n, device=ids.device)
            if not (ids.dtype == torch.int64 and ids.shape == (1, n * n) and torch.equal(ids[0], static)):
                raise FXVerificationError(f"{path}: full-size vision position ids are not arange({n * n})")
            original = mod.forward

            def forward(self, pixel_values, patch_attention_mask, _original=original, _path=path):
                batch, _, height, width = pixel_values.shape
                side = self.num_patches_per_side
                if ((height // self.patch_size, width // self.patch_size) != (side, side)
                        or tuple(patch_attention_mask.shape) != (batch, side, side)):
                    return _original(pixel_values, patch_attention_mask)
                if not (torch.compiler.is_compiling() or torch.compiler.is_exporting()):
                    if not bool(patch_attention_mask.all()):
                        raise FXVerificationError(f"{_path}: the patch attention mask is not all true")
                embeddings = self.patch_embedding(pixel_values).flatten(2).transpose(1, 2)
                # position_embedding(arange(n*n)) gathers every row of the table in order: it is the table itself
                # ([1, n*n, D], broadcast over the batch), so no embedding lookup is captured
                return embeddings + self.position_embedding.weight[None]

            patched.append((mod, mod.__dict__.get("forward"), path))
            mod.forward = types.MethodType(forward, mod)
        yield [p[2] for p in patched]
    finally:
        for mod, previous, _ in patched:
            if previous is None:
                del mod.forward
            else:
                mod.forward = previous


def smolvla_attention_inputs(pad_masks, att_masks, keep_self_for_padded=False):
    """Host-side attention inputs of SmolVLA's text / expert layers, computed as VLAFlowMatching.forward does in its
    graph: the boolean keep mask make_att_2d_masks(pad_masks, att_masks) (lerobot's own function) as the additive BF16
    mask additive_attention_mask(keep), [B, S, S] (the layers slice dims 1 and 2), and the position ids
    cumsum(pad_masks, dim=1) - 1, int64 [B, S]. Passing both as boundary inputs (as TinyLlama's layer boundary
    passes its mask) leaves no cumsum in the graph.

    keep_self_for_padded: also keep the diagonal entry of every padded query row (pad_masks False), so that no row
    is fully masked. Off by default; see tests/test_smolvla_host_inputs.py for when it is exact.
    Returns (mask, position_ids, keep)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    keep = make_att_2d_masks(pad_masks, att_masks)
    if keep_self_for_padded:
        eye = torch.eye(keep.shape[-1], dtype=torch.bool, device=keep.device)
        keep = keep | (eye & ~pad_masks.bool()[:, :, None])
        if not bool(keep.any(-1).all()):
            raise ValueError("a query row keeps no entry")
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    return additive_attention_mask(keep).contiguous(), position_ids.contiguous(), keep


ROPE_TABLES = ("joint", "prefix", "expert")


def _rope_table(positions, head_dim, max_wavelength=10_000):
    """(cos, sin), float32 [B, L, 1, head_dim / 2]: lerobot smolvlm_with_expert.apply_rope's angle arithmetic for
    `positions` [B, L], line for line (timescale = max_wavelength ** ((2 / D) * arange(D / 2)); radians =
    positions / timescale)."""
    d_half = head_dim // 2
    freq_exponents = (2.0 / head_dim) * torch.arange(d_half, dtype=torch.float32, device=positions.device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)
    radians = radians[..., None, :]
    return torch.cos(radians), torch.sin(radians)


def smolvla_rope_tables(position_ids, prefix_len, head_dim):
    """Host RoPE tables of SmolVLA's text / expert layers: {"joint", "prefix", "expert"} -> (cos, sin), the angles
    apply_rope derives from the three position tensors SmolVLMWithExpertModel passes it with use_cache=False:
    position_ids (joint self-attention layers), position_ids[:, :prefix_len] (the prefix branch of a cross-attention
    layer) and the expert's position_ids[:, prefix_len:] - its row minimum. Each table has the shape of the in-graph
    angles it replaces."""
    expert = position_ids[:, prefix_len:]
    expert = expert - torch.min(expert, dim=1, keepdim=True).values
    return {"joint": _rope_table(position_ids, head_dim), "prefix": _rope_table(position_ids[:, :prefix_len], head_dim),
            "expert": _rope_table(expert, head_dim)}


class _HostRope:
    """smolvlm_with_expert.apply_rope with the angles' cos / sin taken from host tables (smolvla_rope_tables) instead
    of computed in the graph: the float32 x1 * cos - x2 * sin / x2 * cos + x1 * sin rotation itself is lerobot's,
    unchanged. A call is matched to its table by the sequence length of `positions` (joint S, prefix P, expert M,
    which must differ), or, while `role` names a table, to that table (the denoising step, whose self-attention and
    cross-attention layers both rotate chunk-length positions; see SmolVLADenoiseStep); the position values are not
    read. install() replaces the module function, restore() puts it back; `tables` is set by the boundary's forward."""

    def __init__(self):
        self.tables, self._saved, self.role = None, None, None

    def install(self):
        import lerobot.policies.smolvla.smolvlm_with_expert as swe

        if self._saved is None:
            self._saved = swe.apply_rope
            swe.apply_rope = self.apply_rope

    def restore(self):
        import lerobot.policies.smolvla.smolvlm_with_expert as swe

        if self._saved is not None:
            swe.apply_rope, self._saved = self._saved, None

    def apply_rope(self, x, positions, max_wavelength=10_000):
        lengths = {name: table[0].shape[1] for name, table in self.tables.items()}
        if self.role is not None:
            name = self.role
            if lengths.get(name) != positions.shape[1]:
                raise ValueError(f"host RoPE table {name!r} does not cover {positions.shape[1]} positions: {lengths}")
        else:
            if len(set(lengths.values())) != len(lengths):
                raise ValueError(f"host RoPE tables need distinct sequence lengths: {lengths}")
            name = next((n for n, length in lengths.items() if length == positions.shape[1]), None)
            if name is None:
                raise ValueError(f"no host RoPE table for {positions.shape[1]} positions: {lengths}")
        cos, sin = self.tables[name]
        d_half = x.shape[-1] // 2
        dtype = x.dtype
        x = x.to(torch.float32)
        x1, x2 = x.split(d_half, dim=-1)
        res = torch.empty_like(x)
        res[..., :d_half] = x1 * cos - x2 * sin
        res[..., d_half:] = x2 * cos + x1 * sin
        return res.to(dtype)


class _HostTokenEmbeddings:
    """SmolVLMWithExpertModel.embed_language_tokens taking its result from a host input: lerobot's own
    embed_language_tokens (the text model's input-embedding lookup) is run on the host (smolvla_flow_inputs) and the
    boundary hands the result in, so the graph holds no token embedding_lookup. embed_prefix itself is unchanged and
    still applies its sqrt(hidden) scaling in the graph. install() sets an instance attribute on the
    SmolVLMWithExpertModel, restore() removes it; `embeddings` is set by the boundary's forward and must match the
    token shape it is called with. Refused when the policy adds image special tokens (their lookups go through the
    same function)."""

    def __init__(self, inner):
        self.inner, self.embeddings, self._installed = inner, None, False

    def install(self):
        if getattr(self.inner, "add_image_special_tokens", False):
            raise ValueError("host text embeddings: image special tokens also use embed_language_tokens")
        if not self._installed:
            self.inner.embed_language_tokens = self.embed_language_tokens
            self._installed = True

    def restore(self):
        if self._installed:
            del self.inner.embed_language_tokens
            self._installed = False

    def embed_language_tokens(self, tokens):
        if tuple(tokens.shape) != tuple(self.embeddings.shape[:-1]):
            raise ValueError(f"host text embeddings {tuple(self.embeddings.shape)} do not match tokens {tuple(tokens.shape)}")
        return self.embeddings


def smolvla_time_embedding(policy, actions, noise, time):
    """Host value of the flow-timestep embedding embed_suffix computes: lerobot's own
    create_sinusoidal_pos_embedding(time, expert_hidden_size, min_period, max_period) (float64 linspace, pow, sin,
    cos) followed by embed_suffix's own .type(dtype=action_emb.dtype), with action_emb = action_in_proj(x_t) run
    eagerly for its dtype (x_t as make_flow_matching_inputs builds it, the noise at t = 1). [B, expert_hidden]."""
    from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

    model = policy.model
    with torch.no_grad():
        x_t, _, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
        action_emb = model.action_in_proj(x_t)
        emb = create_sinusoidal_pos_embedding(time, model.vlm_with_expert.expert_hidden_size, model.config.min_period,
                                              model.config.max_period, device=action_emb.device)
        return emb.type(dtype=action_emb.dtype).contiguous()


class _HostTimeEmbedding:
    """modeling_smolvla.create_sinusoidal_pos_embedding taking its result from a host input (smolvla_time_embedding),
    so the graph holds none of its float64 linspace / pow / sin / cos. The host value already has embed_suffix's
    dtype, so embed_suffix's .type(dtype=...) of it is the identity. install() replaces the module function, restore()
    puts it back; `embedding` is set by the boundary's forward and must have shape [len(time), dimension]."""

    def __init__(self):
        self.embedding, self._saved = None, None

    def install(self):
        import lerobot.policies.smolvla.modeling_smolvla as modeling

        if self._saved is None:
            self._saved = modeling.create_sinusoidal_pos_embedding
            modeling.create_sinusoidal_pos_embedding = self.create_sinusoidal_pos_embedding

    def restore(self):
        import lerobot.policies.smolvla.modeling_smolvla as modeling

        if self._saved is not None:
            modeling.create_sinusoidal_pos_embedding, self._saved = self._saved, None

    def create_sinusoidal_pos_embedding(self, time, dimension, min_period, max_period, device="cpu"):
        if tuple(self.embedding.shape) != (time.shape[0], dimension) or time.ndim != 1:
            raise ValueError(f"host time embedding {tuple(self.embedding.shape)} does not match time "
                             f"{tuple(time.shape)} and dimension {dimension}")
        return self.embedding


_MASK_VIEWS = {torch.ops.aten.slice.Tensor, torch.ops.aten.unsqueeze.default, torch.ops.aten.expand.default,
               torch.ops.aten.view.default, torch.ops.aten.reshape.default, torch.ops.aten.alias.default}
_SOFTMAX = {torch.ops.aten.softmax.int, torch.ops.aten._softmax.default}


def _dtype(node):
    val = node.meta.get("val") if isinstance(node, torch.fx.Node) else None
    return getattr(val, "dtype", None)


def softmax_mask_forms(graph):
    """The input form of every softmax in a captured graph, in graph order:
      "additive_bf16_min"  aten.add.Tensor(scores, mask): float32 scores plus a BF16 mask that is, through views
                           (slice / unsqueeze / expand / view / reshape), aten.where.ScalarOther(keep, zeros BF16,
                           BF16 minimum) built in the graph (TinyLlama's full-model form) or a BF16 graph input
                           (TinyLlama's layer boundary form);
      "additive_other"     any other aten.add.Tensor input;
      "select"             an aten.where input (the boolean select form);
      "unmasked"           anything else.
    Returns [{"node", "form", ...}]."""
    rows = []
    for node in graph.nodes:
        if node.op != "call_function" or node.target not in _SOFTMAX:
            continue
        x = node.args[0]
        row = {"node": node.name, "input": getattr(x, "name", None)}
        target = getattr(x, "target", None)
        if target == torch.ops.aten.add.Tensor and len(x.args) == 2 and all(isinstance(a, torch.fx.Node) for a in x.args):
            scores, mask = x.args
            source = mask
            while source.op == "call_function" and source.target in _MASK_VIEWS:
                source = source.args[0]
            row.update(scores_dtype=str(_dtype(scores)), mask_dtype=str(_dtype(mask)), mask_source=source.name)
            built = (source.op == "call_function" and source.target == torch.ops.aten.where.ScalarOther
                     and _dtype(source.args[0]) == torch.bool and isinstance(source.args[1], torch.fx.Node)
                     and source.args[1].target == torch.ops.aten.zeros.default and _dtype(source.args[1]) == torch.bfloat16
                     and source.args[2] == BF16_MIN)
            boundary = source.op == "placeholder"
            ok = (built or boundary) and _dtype(mask) == torch.bfloat16 and _dtype(scores) == torch.float32
            row["form"] = "additive_bf16_min" if ok else "additive_other"
        elif target in (torch.ops.aten.where.self, torch.ops.aten.where.ScalarOther, torch.ops.aten.where.ScalarSelf,
                        torch.ops.aten.where.Scalar):
            row.update(form="select", fill=[a for a in x.args if not isinstance(a, torch.fx.Node)])
        else:
            row["form"] = "unmasked"
        rows.append(row)
    return rows


def fold_identity_nodes(program):
    """Replace capture-side identities by their input in a captured program; returns the replaced node names.
    Currently aten.cat over a single tensor (lerobot's embed_suffix concatenates a one-element list): it returns a copy
    with the input's values, shape and dtype, so every user can read the input itself. The pinned backend lowers it
    to an `identity` it has no application route for. Only call_function nodes whose input is a tensor of the same
    dtype and shape are replaced. Run before drop_dead_pure_nodes (boundaries with drop_dead_nodes)."""
    replaced = []
    for node in list(program.graph.nodes):
        if node.op != "call_function" or node.target != torch.ops.aten.cat.default:
            continue
        tensors = node.args[0]
        if not (isinstance(tensors, (list, tuple)) and len(tensors) == 1 and isinstance(tensors[0], torch.fx.Node)):
            continue
        source, out = tensors[0], node.meta.get("val")
        val = source.meta.get("val")
        if (not isinstance(val, torch.Tensor) or not isinstance(out, torch.Tensor) or val.dtype != out.dtype
                or tuple(val.shape) != tuple(out.shape)):
            continue
        node.replace_all_uses_with(source)
        program.graph.erase_node(node)
        replaced.append(node.name)
    if replaced:
        program.graph_module.recompile()
    return replaced


def drop_dead_pure_nodes(program):
    """Remove computations nothing reads from a captured program; returns the removed node names. Repeatedly:
      - call_function nodes with no users and no side effect (Node.is_impure() False);
      - an aten.detach_ with no users whose input is an aten.lift_fresh_copy (torch.tensor(...) in the traced code:
        the in-place detach acts on a fresh private copy, so nothing can observe it);
      - then a lifted CONSTANT_TENSOR input with no users (placeholder, graph-signature input spec and constant).
    Parameters and buffers are kept even when unread (the backend authenticates them against the export).
    Used when a boundary's host inputs leave a computation unread: the expert's position - min with host RoPE tables,
    and embed_prefix / embed_suffix's pad and block masks (including their torch.tensor constants) at the flow-loss
    boundary. torch.export keeps them, and the backend's stage-capture refuses a program with a node it eliminates
    without correspondence ("quantized -> prepared correspondence incomplete" / "frontend trace does not bind
    complete captured IR"). Values that reach an output are untouched. Applied to boundaries with drop_dead_nodes
    (the SmolVLA ones)."""
    from torch.export.graph_signature import InputKind

    removed = []
    detach_, fresh = torch.ops.aten.detach_.default, torch.ops.aten.lift_fresh_copy.default
    changed = True
    while changed:
        changed = False
        for node in reversed(list(program.graph.nodes)):
            if node.op != "call_function" or node.users:
                continue
            fresh_detach = (node.target == detach_ and isinstance(node.args[0], torch.fx.Node)
                            and node.args[0].target == fresh and len(node.args[0].users) == 1)
            if not node.is_impure() or fresh_detach:
                removed.append(node.name)
                program.graph.erase_node(node)
                changed = True
    specs = program.graph_signature.input_specs
    by_name = {node.name: node for node in program.graph.nodes if node.op == "placeholder"}
    for spec in list(specs):
        node = by_name.get(getattr(spec.arg, "name", None))
        if spec.kind == InputKind.CONSTANT_TENSOR and node is not None and not node.users:
            specs.remove(spec)
            program.constants.pop(spec.target, None)
            removed.append(node.name)
            program.graph.erase_node(node)
    if removed:
        program.graph_module.recompile()
    return removed


def graph_shape_census(graph):
    """{"dynamic_nodes": [names of nodes whose value has a symbolic (data-dependent or dynamic) dimension],
    "cumsum_nodes": [...]}."""
    dynamic, cumsum = [], []
    for node in graph.nodes:
        values = node.meta.get("val")
        values = values if isinstance(values, (tuple, list)) else [values]
        if any(isinstance(v, torch.Tensor) and not all(isinstance(d, int) for d in v.shape) for v in values) or \
                isinstance(node.meta.get("val"), torch.SymInt):
            dynamic.append(node.name)
        if node.op == "call_function" and node.target == torch.ops.aten.cumsum.default:
            cumsum.append(node.name)
    return {"dynamic_nodes": dynamic, "cumsum_nodes": cumsum}


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
        import exporter.tile_native as tn

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


def _cast_if_needed(x, dtype):
    """x.to(dtype) only when the dtype differs. Same values either way, but a same-dtype `.to` is captured as
    an identity aten.to node, and the backend importer cannot bind an eliminated identity whose input op
    decomposes into several values with one lineage and ABI (e.g. an f32 silu -> sigmoid, mul)."""
    return x if x.dtype == dtype else x.to(dtype)


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
PATCH_ROWS, IM2COL = "patch_rows_qlinear", "im2col_qconv2d"


def conv_patch_rows(layer, height, width):
    """True when RefQuantConv2d `layer` on an [.., height, width] input is a patch embedding: stride equal to the
    kernel, no padding, dilation 1 (groups is always 1) and the input an exact multiple of the kernel, and its bias
    (if any) already in the weight dtype. Its im2col rows are then the non-overlapping patches in the same order
    (F.unfold: column index c * kh * kw + i * kw + j, row index oh_i * ow + ow_j), so
    x.reshape(B, C, oh, kh, ow, kw).permute(0, 2, 4, 1, 3, 5).reshape(B * oh * ow, C * kh * kw) is the same
    tensor, and qlinear's `y + bias` is qconv2d's bias add on the same words. Any other Conv2d keeps atlas.qconv2d."""
    kh, kw = layer.kernel_size
    return (tuple(layer.stride) == (kh, kw) and tuple(layer.padding) == (0, 0) and tuple(layer.dilation) == (1, 1)
            and height % kh == 0 and width % kw == 0
            and (layer.bias is None or layer.bias.dtype == layer.weight.dtype))


@contextlib.contextmanager
def _native_sites(module, policy, storage, *, attention=True, conv=True):
    """For the capture only: route each candidate Linear through atlas.quantize_act + atlas.qlinear, each
    Conv2d through im2col + quantize_act + atlas.qconv2d (a patch-embedding Conv2d, conv_patch_rows: through its
    patch rows + quantize_act + atlas.qlinear), and each attention module's Q.K and P.V through
    quantize_act + atlas.qk_matmul / atlas.pv_matmul (softmax and masks stay ordinary nodes); for float8
    storage, hold the weight codes as a float8_e4m3fn bitcast of the same bytes. The caller's forwards,
    buffers and ref_attention functions are restored afterwards. Yields the plan:
    {"linear": {path: rule}, "conv2d": {path: rule}, "attention": native_attention(...) or None,
     "conv2d_lowering": {path: PATCH_ROWS | IM2COL} (filled as the convolutions are called)}."""
    import exporter.atlas_fx_ops as atlas_fx_ops
    import reference.ref_attention as ref_attention

    code_dtype = STORAGE_DTYPES[storage]
    candidates = native_candidates(module, policy)
    # a boundary with bias_add (the SmolVLA ones) carries each Linear's bias as a separate BF16 addition after a
    # bias-free atlas.qlinear: the pinned backend's already-quantized intake admits only the 4-operand contraction
    bias_add = bool(getattr(module, "bias_add", False))
    undo, patched = [], []
    plan = {"linear": {}, "conv2d": {}, "attention": native_attention(module, policy) if attention else None,
            "conv2d_lowering": {}, "bias_add": bias_add}
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

            def qlinear(codes, scale, layer, out_dtype):
                if bias_add and layer.bias is not None:
                    # atlas.qlinear's own body is linear(...) then `y + bias`: the same addition as a separate node
                    y = torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp, None,
                                                out_dtype, atlas_fx_ops.WEIGHT_FORMAT)
                    return y + layer.bias
                return torch.ops.atlas.qlinear(codes, scale, layer.weight_codes, layer.weight_scale_exp,
                                               layer.bias, out_dtype, atlas_fx_ops.WEIGHT_FORMAT)

            def linear_forward(x, layer=layer, nrule=nrule, out_dtype=out_dtype):
                x = _cast_if_needed(x, layer.weight.dtype)
                codes, scale = _qact(x, nrule, storage, label=layer.layer_name)
                return qlinear(codes, scale, layer, out_dtype)

            def conv_forward(x, layer=layer, nrule=nrule, out_dtype=out_dtype, path=path):
                B, _, H, W = x.shape
                if conv_patch_rows(layer, H, W):
                    # stride == kernel, no padding, dilation 1: the unfolded input is the image cut into
                    # non-overlapping patches, so the same [B*oh*ow, C*kh*kw] rows (same order, same values) are
                    # a reshape / permute, and the conv is one qlinear on them with the same codes, exponents,
                    # bias and output dtype, reshaped back as RefQuantConv2d.forward does.
                    kh, kw = layer.kernel_size
                    oh, ow = H // kh, W // kw
                    x_flat = _cast_if_needed(x.reshape(B, -1, oh, kh, ow, kw).permute(0, 2, 4, 1, 3, 5)
                                             .reshape(B * oh * ow, -1), layer.weight.dtype)
                    codes, scale = _qact(x_flat, nrule, storage, label=layer.layer_name)
                    y = qlinear(codes, scale, layer, out_dtype)
                    plan["conv2d_lowering"][path] = PATCH_ROWS
                    return y.reshape(B, oh * ow, layer.out_channels).permute(0, 2, 1).reshape(
                        B, layer.out_channels, oh, ow)
                plan["conv2d_lowering"][path] = IM2COL
                # RefQuantConv2d.forward up to the quantizer: im2col stays visible (aten.im2col, permute)
                oh = (H + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
                ow = (W + 2 * layer.padding[1] - layer.dilation[1] * (layer.kernel_size[1] - 1) - 1) // layer.stride[1] + 1
                cols = torch.nn.functional.unfold(x, kernel_size=layer.kernel_size, stride=layer.stride,
                                                  padding=layer.padding, dilation=layer.dilation)
                x_flat = _cast_if_needed(cols.permute(0, 2, 1).reshape(B * oh * ow, -1), layer.weight.dtype)
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

            def mm(fm, a_q, a_s, b_q, b_s, dtype_out=torch.float32, role=None, pop_shift=0):
                if fm is not None or role not in ("qk", "pv"):
                    return original_mm(fm, a_q, a_s, b_q, b_s, dtype_out, role, **({"pop_shift": pop_shift} if pop_shift else {}))
                op = torch.ops.atlas.qk_matmul if role == "qk" else torch.ops.atlas.pv_matmul
                return op(a_q, a_s, b_q, b_s, int(pop_shift)) if pop_shift else op(a_q, a_s, b_q, b_s)
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


def _inspect_native_site(path, layer, site, rule, nodes, lifted, depends_on_input, storage, calls=1, lowering=None,
                         bias_add=False, shared_scopes=frozenset()):
    """One Linear (or Conv2d) call = one atlas.qlinear (atlas.qconv2d) node fed directly by the lifted code and
    exponent buffers and by its own atlas.quantize_act node whose input depends on the user input; nothing
    else decodes or contracts in its scope (a Conv2d keeps its visible im2col). A module the boundary calls
    `calls` times (the eager replay's count, 1 by default) has exactly `calls` such nodes, each with its own
    quantize_act, all sharing the one code / exponent buffer pair. A Conv2d captured as patch rows (lowering
    PATCH_ROWS, conv_patch_rows) is an atlas.qlinear on its reshaped / permuted input, with no im2col.
    `shared_scopes`: module paths of tied quantized embeddings (exporter.head_embed), which may also read the
    buffers (a row lookup, no contraction)."""
    import exporter.atlas_fx_ops as atlas_fx_ops

    conv = type(layer).__name__ == "RefQuantConv2d"
    patch_rows = conv and lowering == PATCH_ROWS
    op = torch.ops.atlas.qconv2d.default if conv and not patch_rows else torch.ops.atlas.qlinear.default
    key = "qconv2d" if op == torch.ops.atlas.qconv2d.default else "qlinear"
    prefix = path + "." if path else ""
    code, exp = lifted.get(prefix + "weight_codes"), lifted.get(prefix + "weight_scale_exp")
    placeholder, bias = lifted.get(prefix + "weight"), lifted.get(prefix + "bias")
    if code is None or exp is None:
        raise FXVerificationError(f"{layer.layer_name}: codes or exponents were not lifted")
    if placeholder is not None and placeholder.users:
        raise FXVerificationError(f"{layer.layer_name}: graph consumes the float placeholder")
    mine = [n for n in nodes if n.op == "call_function" and n.target == op
            and n.args[2] is code and n.args[3] is exp]
    shared = {n for n in set(code.users) | set(exp.users) if _scope(n) in shared_scopes}
    if len(mine) != calls or set(code.users) - shared != set(mine) or set(exp.users) - shared != set(mine):
        what = "exactly one" if calls == 1 else f"exactly {calls} (the eager replay's call count),"
        raise FXVerificationError(f"{layer.layer_name}: expected {what} {op} consuming "
                                  f"the code and exponent buffers directly, found {len(mine)}")
    acts, bias_adds = [], []
    for q in mine:
        x_codes, x_exp, _, _, q_bias, out_dtype, weight_format = q.args[:7]
        act = _act_source(x_codes, 0)
        if act is None or act is not _act_source(x_exp, 1) or act in acts:
            raise FXVerificationError(f"{layer.layer_name}: {op} is not fed by one atlas.quantize_act of its own")
        _check_act(layer.layer_name, act, rule, storage, depends_on_input)
        if out_dtype != str(layer.weight.dtype).replace("torch.", "") or weight_format != atlas_fx_ops.WEIGHT_FORMAT:
            raise FXVerificationError(f"{layer.layer_name}: {op} dtype/format attributes differ from the layer")
        if bias_add and key == "qlinear" and layer.bias is not None:
            # bias-free qlinear whose only consumer is add(qlinear, bias)
            users = list(q.users)
            if (q_bias is not None or bias is None or len(users) != 1 or users[0].target != torch.ops.aten.add.Tensor
                    or tuple(users[0].args) != (q, bias)):
                raise FXVerificationError(f"{layer.layer_name}: bias is not one add(qlinear, bias) after a bias-free {op}")
            bias_adds.append(users[0])
        elif (layer.bias is None) != (q_bias is None) or (bias is not None and q_bias is not bias):
            raise FXVerificationError(f"{layer.layer_name}: bias is not consumed by {op}")
        acts.append(act)
    scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
    allowed = ALLOWED_NATIVE_SCOPE_TARGETS | (CONV_SCOPE_TARGETS if conv else set())
    stray = sorted({str(n.target) for n in scope if n.target not in allowed and n not in mine and n not in acts
                    and n not in bias_adds})
    if stray:
        raise FXVerificationError(f"{layer.layer_name}: decode/contraction ops remain in the layer: {stray}")
    row = {"site": layer.layer_name, "module": path, "kind": "conv2d" if conv else "linear", "native": True,
           "calls": calls,
           "code_buffer": prefix + "weight_codes", "scale_buffer": prefix + "weight_scale_exp",
           "code_dtype": str(layer.weight_codes.dtype).replace("torch.", ""),
           "op": key, key: mine[0].name, "quantize_act": acts[0].name,
           **({"bias_add": [n.name for n in bias_adds]} if bias_adds else {}),
           "activation_rule": {"mode": rule[0], "reduce": rule[1], "rule": rule[2], "static_exp": rule[3]},
           "nodes_in_scope": [n.name for n in scope]}
    if calls > 1:                                    # per-call node names (the keys above name the first call)
        row[key + "_nodes"] = [n.name for n in mine]
        row["quantize_act_nodes"] = [a.name for a in acts]
    if conv:
        row["lowering"] = PATCH_ROWS if patch_rows else IM2COL
        row["im2col_nodes"] = [n.name for n in scope if n.target == torch.ops.aten.im2col.default]
        if patch_rows and row["im2col_nodes"]:
            raise FXVerificationError(f"{layer.layer_name}: patch-row Conv2d still holds im2col nodes")
        if not patch_rows and len(row["im2col_nodes"]) != calls:
            raise FXVerificationError(f"{layer.layer_name}: expected {calls} im2col node(s) before quantize_act, "
                                      f"found {len(row['im2col_nodes'])}")
    return row


def _inspect_native_attention(plan_attention, nodes, storage, depends_on_input, claimed_acts, attn_calls=None,
                              absent=None):
    """Attention: every Q.K / P.V is an atlas.qk_matmul / atlas.pv_matmul whose operands are the codes and
    exponents of quantize_act nodes tagged "<site>:<operand>", carrying the policy's operand rules and
    depending on the user input; the P.V input depends on a Q.K output through the (visible) softmax chain.
    Matched by tag, not by module scope: a module called as `module.forward(...)` (SmolVLA's text/expert
    stack) is not on the nn_module_stack, so its nodes carry the caller's scope. Returns one row per site.
    attn_calls: {site: calls per case} counted by tag in the eager replay; when given, each site has exactly that
    many Q.K / P.V pairs (a site called N times has N pairs). absent: {module path: site} of attention modules the
    boundary never calls; none of their tagged quantize_act nodes may be in the graph."""
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
    if attn_calls is not None:
        for key, n_calls in attn_calls.items():
            have = len(per_site.get(key, {"qk_matmul": []})["qk_matmul"])
            if have != n_calls:
                raise FXVerificationError(f"attention site {key!r}: the eager replay called it {n_calls} time(s) per case "
                                          f"but the graph has {have} Q.K / P.V pair(s)")
        for key in per_site:
            if key not in attn_calls:
                raise FXVerificationError(f"attention site {key!r} is in the graph but the eager replay never called it")
    for path, site in (absent or {}).items():
        tag = (site if site is not None else "") + ":"
        if any(str(n.args[6]).startswith(tag) for n in nodes if n.op == "call_function"
               and n.target == torch.ops.atlas.quantize_act.default and len(n.args) > 6):
            raise FXVerificationError(f"attention module {path} is declared never called but the graph holds its "
                                      f"quantize_act nodes ({tag!r})")
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
    site_scope = getattr(module, "policy_site_scope", None)
    for name, site in policy["sites"].items():
        if site["status"] not in ("host", "skipped"):
            continue
        if site_scope is not None and not name.startswith(tuple(site_scope)):
            continue                                      # outside a sub-boundary (inspect_graph's scope rule)
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


def _storage_report(program, nodes, module, policy, lifted):
    """Quantized-storage modules (exporter.head_embed: QuantEmbedding, NVFP4Linear; none by default): each must read
    its codes from a lifted buffer (a tied embedding: the head's), hold no float weight, and is listed with its
    node count. Their decode nodes (e4m3 / e2m1 decode, exp2) are storage decoding, not an emulated matmul."""
    import exporter.head_embed as head_embed

    prefix = _prefix_of(module, policy)
    rows = []
    for path, layer in module.named_modules():
        if not isinstance(layer, (head_embed.QuantEmbedding, head_embed.NVFP4Linear)):
            continue
        owner = path if getattr(layer, "tied_to", None) is None else prefix + layer.tied_to
        code = lifted.get(owner + ".weight_codes")
        scope = [n for n in nodes if n.op == "call_function" and _scope(n) == path]
        if scope and (code is None or not code.users):
            raise FXVerificationError(f"{path}: quantized storage does not read its lifted codes {owner}.weight_codes")
        weight = program.state_dict.get(path + ".weight")
        if weight is not None and not weight.is_meta:
            raise FXVerificationError(f"{path}: quantized storage module holds a float weight")
        site = path[len(prefix):] if prefix and path.startswith(prefix) else path
        rows.append({"site": site, "module": path, "kind": "embedding" if isinstance(layer, head_embed.QuantEmbedding)
                     else "nvfp4_linear", "format": layer.fmt, "code_buffer": owner + ".weight_codes",
                     "tied_to": getattr(layer, "tied_to", None), "in_graph": bool(scope), "node_count": len(scope),
                     "contractions": [n.name for n in scope if n.target in CONTRACTIONS]})
    return rows


def _split_name(name):
    """(base, suffix) of an FX node name as torch.fx's namespace forms it: `to_4` -> ("to", 4), `to` -> ("to", 0)."""
    base, sep, num = name.rpartition("_")
    return (base, int(num)) if sep and base and num.isdigit() else (name, 0)


def _name_storage_nodes(program, module):
    """Own names for the storage-decode nodes (exporter.head_embed; none by default, then the graph is untouched).

    torch.export numbers call_function nodes per op name in creation order (`to`, `to_1`, ...), so the decode
    nodes of a quantized embedding, which run first, shift every later node of the same op: layer 0's rotary
    cos / sin casts become `to_7` / `to_8` instead of the default's `to_4` / `to_5`, and backend tools that
    identify boundaries by node name see a different graph. This renames each non-contraction node in a storage
    module's scope to `<module path>_<op name>[_k]` and lowers the suffix of every other node by the number of
    storage nodes of its op name numbered before it, which gives the names the default capture has. Names only:
    the targets, arguments and order of the nodes are unchanged. Returns {old: new} of the renamed nodes."""
    import exporter.head_embed as head_embed

    paths = {path: path.replace(".", "_") for path, layer in module.named_modules()
             if isinstance(layer, (head_embed.QuantEmbedding, head_embed.NVFP4Linear))}
    if not paths:
        return {}
    graph = program.graph
    nodes = [n for n in graph.nodes if n.op == "call_function"]
    storage = {n for n in nodes if _scope(n) in paths and n.target not in CONTRACTIONS}
    removed = collections.defaultdict(list)                  # op name -> suffixes of its storage nodes
    for node in storage:
        base, num = _split_name(node.name)
        removed[base].append(num)
    renames, own = {}, collections.Counter()
    for node in nodes:
        base, num = _split_name(node.name)
        if node in storage:
            stem = f"{paths[_scope(node)]}_{base}"
            new = stem if not own[stem] else f"{stem}_{own[stem]}"
            own[stem] += 1
        else:
            num -= sum(1 for k in removed.get(base, ()) if k < num)
            new = f"{base}_{num}" if num else base
        if new != node.name:
            renames[node.name] = new
    if not renames:
        return {}
    names = [renames.get(n.name, n.name) for n in graph.nodes]
    if len(set(names)) != len(names):
        raise FXVerificationError("storage-node renaming would give two nodes one name")
    for node in nodes:
        node.name = renames.get(node.name, node.name)
    signature = program.graph_signature
    for spec in list(signature.input_specs) + list(signature.output_specs):
        if getattr(spec.arg, "name", None) in renames:
            spec.arg.name = renames[spec.arg.name]
    for entry in program.module_call_graph:
        if entry.signature is not None:
            for arg in list(entry.signature.inputs) + list(entry.signature.outputs):
                if getattr(arg, "name", None) in renames:
                    arg.name = renames[arg.name]
    program.graph_module.recompile()
    program.validate()
    return renames


def _tile_native_report(module, policy, native_paths, plan, act_stats):
    """Per-site `tile_native` (NEW): true when the weight exponent table and the activation / attention
    operand exponents are constant along K at 32-granularity (src/exporter/tile_native.py), with the reasons for every
    site that is not (per-channel or per-token recipes, block32 along K). `native_fp8` says whether the site
    is carried by atlas.qlinear / qconv2d / qk_matmul / pv_matmul in this graph."""
    import exporter.tile_native as tn

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


def _pack_plan(program, policy):
    """REPORTING ONLY: the pack plan of the policy's static activations with the producer class of each packed tensor read
    from the captured graph (docs/algos/tile-aware.md#pack-plan). Returns (policy copy carrying the plan, plan, summary). A graph without
    atlas.quantize_act nodes (decode-based capture) gives producer class 'unknown'."""
    producers, shared = pack_plan.producers_from_graph(str(program.graph), str(program.graph_signature), policy)
    with_plan = pack_plan.with_plan(policy, producers, shared)
    plan = {k: r for k, r in pack_plan.recorded_plan(with_plan).items()}
    return with_plan, plan, with_plan["pack_plan_summary"]


def _admission_report(policy, act_stats):
    """compiler_pack_scale_admission for graph_inspection.json: the policy's static exponents plus the
    dynamic exponents observed in the capture cases (INFORMATIONAL; nothing is clamped)."""
    import exporter.tile_native as tn

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


def inspect_graph(program, module, policy, plan=None, storage="uint8", unused=(), act_stats=None, calls=None,
                  attn_calls=None, unused_attention=None):
    """Verify real contraction dependencies, not buffer labels alone.

    plan: what `_native_sites` yielded ({"linear": {path: rule}, "conv2d": {path: rule}, "attention":
    {...} or None}) for a --native-fp8 capture. Those sites are checked by _inspect_native_site /
    _inspect_native_attention, the rest by the decode-graph rules below. `storage` is the code buffer
    dtype name the native nodes were captured with. `unused`: module paths of quantized layers the boundary
    never calls; each must really be absent from the graph (a strict export does not lift their buffers, a
    non-strict one lifts them as inputs that nothing may consume).
    calls: {module path: calls per case} of the quantized layers from the eager replay; a layer called N times
    must have N nodes (default: 1 for every layer that is not in `unused`). attn_calls: {attention site: calls per
    case} counted by tag (default: no count is enforced). unused_attention: {module path: site} of attention
    modules the boundary never calls (pruned from the plan)."""
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
    scope = getattr(module, "policy_site_scope", None)
    outside = []
    if scope is not None:      # a sub-boundary (e.g. CausalLayer): sites of other modules are outside the graph
        outside = sorted(name for name in selected if not name.startswith(tuple(scope)))
        selected = {name: site for name, site in selected.items() if name not in outside}
    actual, rows, unused_rows = set(), [], []
    storage_rows = _storage_report(program, nodes, module, policy, lifted)      # none by default
    tied_scopes = frozenset(row["module"] for row in storage_rows if row["tied_to"])
    for path, layer in module.named_modules():
        if type(layer).__name__ not in ("RefQuantLinear", "RefQuantConv2d"):
            continue
        site = layer.layer_name
        if site not in selected or site in actual:
            raise FXVerificationError(f"unexpected or duplicated quantized module: {site}")
        actual.add(site)
        if path in unused:
            prefix = path + "."
            held = [lifted[prefix + b] for b in QUANT_BUFFERS if lifted.get(prefix + b) is not None]
            if any(h.users for h in held) or any(_scope(n) == path for n in nodes):
                raise FXVerificationError(f"{site}: never called by the eager replay but present in the graph")
            unused_rows.append({"site": site, "module": path,
                                **({"buffers_lifted_unused": [h.name for h in held]} if held else {})})
            continue
        expected_calls = 1 if calls is None else calls.get(path, 0)
        if expected_calls < 1:
            raise FXVerificationError(f"{site}: the eager replay called it {expected_calls} times but it is not listed as unused")
        if path in native_paths:
            rows.append(_inspect_native_site(path, layer, selected[site], native_paths[path], nodes, lifted,
                                             depends_on_input, storage, expected_calls,
                                             (plan.get("conv2d_lowering") or {}).get(path), plan.get("bias_add", False),
                                             shared_scopes=tied_scopes))
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
        if calls is not None and len(matches) != expected_calls:
            raise FXVerificationError(f"{site}: the eager replay called it {expected_calls} time(s) but "
                                      f"{len(matches)} contraction(s) consume its codes and exponents")
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
        rows.append({"site": site, "module": path, "kind": selected[site]["kind"], "calls": expected_calls,
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
        claimed = {by_name[name] for row in rows if row.get("native")
                   for name in row.get("quantize_act_nodes", [row["quantize_act"]])}
        attention_rows = _inspect_native_attention(plan["attention"], nodes, storage, depends_on_input, claimed,
                                                   attn_calls, unused_attention)
    report = {"weight_site_bindings": rows, "weight_sites_verified": len(rows),
              **({"policy_site_scope": list(scope), "sites_outside_boundary": len(outside)} if scope is not None else {}),
              "unused_quantized_modules": unused_rows,
              "unused_attention_modules": [{"site": site, "module": path} for path, site in sorted((unused_attention or {}).items())],
              "bf16_by_policy": _bf16_by_policy_report(program, nodes, module, policy),
              "functional_attention_sites": attention,
              "functional_attention_structural_binding": "not certified; covered by boundary output parity",
              "unused_parameters": unused_params,
              "unused_quantized_weight_storage": "meta placeholders; no serialized float weight payload",
              "scope": "reference-emulation FX; not native-FP8 MLIR or hardware compatibility"}
    if storage_rows:
        report["quantized_storage_sites"] = storage_rows
    storage_paths = {row["module"] for row in storage_rows}
    report["tile_native"] = _tile_native_report(module, policy, native_paths, plan, act_stats)
    report["compiler_pack_scale_admission"] = _admission_report(policy, act_stats)
    _, pack_records, pack_summary = _pack_plan(program, policy)
    report["pack_plan_summary"] = pack_summary                    # INFORMATIONAL; compiler_pack_scale_admission is unchanged
    report["pack_plan"] = pack_records
    if plan:
        native_rows = [row for row in rows if row.get("native")]
        native_modules = {row["module"] for row in native_rows} | {row["module"] for row in attention_rows}
        report["native_fp8_sites"] = len(native_rows)
        # by the carrying operator: a Conv2d captured as patch rows (atlas.qlinear) counts as a qlinear site
        report["native_fp8_linear_sites"] = sum(row["op"] == "qlinear" for row in native_rows)
        report["native_fp8_conv2d_sites"] = sum(row["op"] == "qconv2d" for row in native_rows)
        report["native_fp8_conv2d_patch_row_sites"] = [row["site"] for row in native_rows
                                                       if row.get("lowering") == PATCH_ROWS]
        # nodes, not sites: a module called N times is one site and N qlinear nodes (calls == 1 everywhere: same numbers)
        report["native_fp8_linear_calls"] = sum(row["calls"] for row in native_rows if row["op"] == "qlinear")
        report["native_fp8_conv2d_calls"] = sum(row["calls"] for row in native_rows if row["op"] == "qconv2d")
        report["native_fp8_attention_modules"] = attention_rows
        report["still_emulated"] = _emulated_report(nodes, module, native_modules | set(unused) | set(unused_attention or {}))
        if not report["still_emulated"]["emulated_modules"]:
            # nothing is left to emulate, so no contraction may be fed by a decode (exp2 / frexp) anywhere
            for node in contractions:
                decoders = [n for n in _ancestors(node, stop_at_contractions=True, value_only=True)
                            if n.target in (torch.ops.aten.exp2.default, torch.ops.aten.frexp.Tensor)
                            and _scope(n) not in storage_paths]
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
        import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
    program = torch.export.load(directory / "model.pt2")
    expected_inventory = json.loads((directory / "buffer_inventory.json").read_text())
    code_dtype = STORAGE_DTYPES[native["storage"]] if native else torch.uint8
    inspected = json.loads((directory / "graph_inspection.json").read_text())
    # A non-strict export lifts the buffers of quantized layers the boundary never calls as graph inputs. The saved
    # inventory leaves them out, so they are excluded here only for modules the inspection declares unused, and only
    # when no node consumes them.
    declared_unused = [row["module"] for row in inspected.get("unused_quantized_modules", [])]
    by_placeholder = {node.name: node for node in program.graph.nodes}
    consumed = sorted(spec.target for spec in program.graph_signature.input_specs
                      if isinstance(spec.target, str) and hasattr(spec.arg, "name")
                      and spec.target.rpartition(".")[2] in QUANT_BUFFERS
                      and any(spec.target.startswith(path + ".") for path in declared_unused)
                      and by_placeholder[spec.arg.name].users)
    if consumed:
        raise FXVerificationError(f"declared unused, but the graph consumes: {consumed[:3]}")
    if not expected_inventory or _without(_inventory(program.named_buffers(), code_dtype),
                                          declared_unused) != expected_inventory:
        raise FXVerificationError("saved code/exponent inventory differs")
    if native:
        counts = {name: sum(1 for n in program.graph.nodes if n.op == "call_function"
                            and n.target == getattr(torch.ops.atlas, name).default)
                  for name in ("qlinear", "qconv2d", "qk_matmul", "pv_matmul")}
        pairs = sum(row["pairs"] for row in inspected.get("native_fp8_attention_modules", []))
        # per-node counts; bundles saved before call counting have one node per site
        if (counts["qlinear"] != inspected.get("native_fp8_linear_calls", inspected.get("native_fp8_linear_sites"))
                or counts["qconv2d"] != inspected.get("native_fp8_conv2d_calls", inspected.get("native_fp8_conv2d_sites"))
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
            "model_replay_modules_imported": [name for name in ("exporter.load_quantized", "reference.ref_profile", "reference.ref_attention")
                                              if name in sys.modules]}


def _observe_acts(module, cases):
    """Eager pass over the capture cases with the native quantize_act forwards installed, recording each
    activation cast (ActStats) and, per case, how many times each label was cast (`stats.case_calls`).
    Run inside _native_sites; never under tracing."""
    global _ACT_RECORDER
    _ACT_RECORDER = stats = ActStats()
    stats.case_calls = []
    try:
        with torch.no_grad():
            for args in cases:
                before = {label: e["calls"] for label, e in stats.sites.items()}
                module(*args)
                stats.case_calls.append({label: e["calls"] - before.get(label, 0) for label, e in stats.sites.items()})
    finally:
        _ACT_RECORDER = None
    return stats


def _attention_calls(stats, plan_attention):
    """({site: calls per case}, {module path: site} of the never-called attention modules). Counted by tag, not by
    forward hooks: a module called as `module.forward(...)` fires no hook. Every operand of a site must be cast the
    same number of times, in every case. Paths that share one site key (no algorithm) cannot be told apart, so they
    are pruned only when the whole key was never cast."""
    if not plan_attention:
        return {}, {}
    per_case = []
    for counts in stats.case_calls:
        by_key = {}
        for label, n in counts.items():
            key, _, operand = label.rpartition(":")
            if ":" in label and operand in ("q", "k", "p", "v"):
                by_key.setdefault(key, {})[operand] = n
        for key, ops in by_key.items():
            if sorted(ops) != ["k", "p", "q", "v"] or len(set(ops.values())) != 1:
                raise FXVerificationError(f"attention site {key!r}: operands q, k, p, v were cast {ops} times")
        per_case.append({key: next(iter(ops.values())) for key, ops in by_key.items()})
    if any(c != per_case[0] for c in per_case[1:]):
        raise FXVerificationError(f"attention sites are called a different number of times in different cases: {per_case}")
    calls = per_case[0]
    keys = {site if site is not None else "" for site in plan_attention["paths"].values()}
    unused = {path: site for path, site in plan_attention["paths"].items()
              if calls.get(site if site is not None else "", 0) == 0}
    return {k: calls[k] for k in sorted(keys) if calls.get(k, 0)}, unused


def capture_fx(module, cases, directory, *, policy, provenance=None, strict=True, fresh_process=True,
               native_fp8=None):
    """Capture fixed-shape reference replay, save, reload and fail closed on mismatch.

    native_fp8: None keeps the decode-based graph. "uint8" or "float8_e4m3fn" captures each candidate
    Linear (native_candidates) as atlas.quantize_act + atlas.qlinear with its code buffer held in that
    dtype (the same bytes); the decode-based graph is also captured in memory and must give
    byte-identical outputs. Conv2d, attention and other sites stay decode-based and are listed."""
    from exporter.quant_export import save_file

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
        # (the call count per quantized module, from the eager replay: 0 = absent from this graph, N = N nodes)
        counter, case_calls = collections.Counter(), []
        hooks = [layer.register_forward_hook(lambda mod, inp, out, path=path: counter.update([path]))
                 for path, layer in module.named_modules() if _is_quant_layer(layer)]
        with torch.no_grad():
            for i, args in enumerate(cases):
                counter.clear()
                leaves, spec = _leaves(module(*args))
                case_calls.append(collections.Counter(counter))
                # contiguous copies: a phase output may be a transposed view (the K / V of LlamaAttention)
                outputs = [value.detach().cpu().clone().contiguous() for value in leaves]
                expected.append((outputs, spec))
                input_keys = [f"case_{i}.input_{j}" for j in range(len(args))]
                output_keys = [f"case_{i}.output_{j}" for j in range(len(outputs))]
                fixtures.update({name: value.detach().cpu().contiguous() for name, value in zip(input_keys, args)})
                fixtures.update(dict(zip(output_keys, outputs)))
                parity["cases"].append({"inputs": input_keys, "outputs": output_keys, "output_spec": spec})
            for hook in hooks:
                hook.remove()
            if any(c != case_calls[0] for c in case_calls[1:]):
                raise FXVerificationError("quantized modules are called a different number of times in different cases "
                                          f"(data-dependent control flow): {[dict(c) for c in case_calls]}")
            calls = case_calls[0]
            unused = sorted(path for path, layer in module.named_modules()
                            if _is_quant_layer(layer) and calls[path] == 0)
            report["unused_quantized_modules"] = unused
            report["quantized_module_calls"] = {path: n for path, n in sorted(calls.items()) if n != 1}
            before = _without(_inventory(module.named_buffers()), unused)
            if not before:
                raise FXVerificationError("no quantized weight buffers at this boundary")
            native_info, baseline, act_stats, attn_calls, unused_attention = None, None, None, {}, {}
            with _static_vision_position_ids(module) as static_vision, _meta_placeholders(module), \
                    _capture_buffer_roles(module) as roles, \
                    _alias_parameters_as_attributes(module, policy) as aliases, _inert_autocast_disable(), \
                    _mse_loss_without_identity_broadcast(), _smolvla_additive_attention_mask(module) as additive:
                report["capture_buffer_roles"] = {"made_persistent": len(roles["persistent"]),
                                                  "held_as_constants": roles["constants"],
                                                  "tied_aliases_held_as_attributes": aliases}
                if additive:
                    report["additive_attention_mask"] = {"modules": additive, "form": "scores + where(keep, 0, bf16_min)"}
                if static_vision:
                    report["static_vision_position_ids"] = {"modules": static_vision,
                                                            "form": "arange(num_patches) (checked against the module's "
                                                                    "own ids at full size, all-true patch mask)"}
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
                        attn_calls, unused_attention = _attention_calls(act_stats, plan["attention"])
                        if unused_attention:         # absent from this graph; the export below runs without them
                            plan["attention"] = {**plan["attention"], "paths": {
                                p: s for p, s in plan["attention"]["paths"].items() if p not in unused_attention}}
                        program = torch.export.export(module, cases[0], strict=strict)
                    # the held buffers are a bitcast: same bytes, so same hash and shape as the uint8 codes
                    if any(held[k]["sha256"] != before[k]["sha256"] or held[k]["shape"] != before[k]["shape"]
                           for k in before):
                        raise FXVerificationError("native buffers differ in bytes from the uint8 codes")
                    report["attention_site_calls"] = {k or "(no algorithm)": n for k, n in sorted(attn_calls.items()) if n != 1}
                    native_info = {"storage": native_fp8, "ops": ["atlas.quantize_act", "atlas.qlinear", "atlas.qconv2d", "atlas.qk_matmul",
                                           "atlas.pv_matmul"],
                                   "ops_module": "exporter.atlas_fx_ops", "weight_format": "atlas_e4m3",
                                   "sites": sorted(set({**plan["linear"], **plan["conv2d"]}) - set(unused)),
                                   "conv2d_sites": sorted(set(plan["conv2d"]) - set(unused)),
                                   "conv2d_lowering": dict(sorted(plan["conv2d_lowering"].items())),
                                   "attention_modules": sorted(plan["attention"]["paths"]) if plan["attention"] else [],
                                   "code_bytes_equal_uint8_source": True}
                    before = held
                else:
                    program = torch.export.export(module, cases[0], strict=strict)
            renamed = _name_storage_nodes(program, module)          # {} (graph untouched) by default
            if renamed:
                report["renamed_for_storage_decode"] = len(renamed)
            if _without(_inventory(program.named_buffers(), STORAGE_DTYPES[native_fp8] if native_fp8 else torch.uint8),
                        unused) != before:
                raise FXVerificationError("capture changed code/exponent buffers")
            if getattr(module, "drop_dead_nodes", False):
                report["identity_nodes_folded"] = fold_identity_nodes(program)
                report["dead_pure_nodes_removed"] = drop_dead_pure_nodes(program)
            graph_report = inspect_graph(program, module, policy, plan if native_fp8 else None,
                                         native_fp8 or "uint8", unused=unused, act_stats=act_stats, calls=calls,
                                         attn_calls=attn_calls if native_fp8 else None,
                                         unused_attention=unused_attention if native_fp8 else None)
            report["graph_shape_census"] = census = graph_shape_census(program.graph)
            if getattr(module, "static_graph", False) and census["dynamic_nodes"]:
                raise FXVerificationError(f"dynamically shaped values in a static boundary's graph: "
                                          f"{census['dynamic_nodes'][:5]}")
            forbidden = sorted({str(n.target) for n in program.graph.nodes if n.op == "call_function"
                                and str(n.target) in getattr(module, "forbidden_graph_targets", ())})
            if forbidden:
                raise FXVerificationError(f"the boundary's graph holds {forbidden}")
            if report.get("additive_attention_mask"):
                forms = softmax_mask_forms(program.graph)
                counts = dict(collections.Counter(row["form"] for row in forms))
                report["additive_attention_mask"]["softmax_input_forms"] = counts
                if counts.get("select") or counts.get("additive_other"):
                    raise FXVerificationError(f"a masked softmax is not in the additive BF16 form: {counts}")
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
            _json(directory / "policy.json", _pack_plan(program, policy)[0])      # policy + pack_plan records (graph classes)
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


class CausalLayer(torch.nn.Module):
    """One decoder layer of a causal LM at its own boundary: (hidden, cos, sin, additive mask) -> hidden.

    The layer stays registered under its original path (`model.model.layers.<index>`, the same names
    CausalLogits gives it), so its lifted buffers keep the export's tensor names behind the `model.` prefix
    and its policy sites keep their identity. Rotary cos/sin and the additive BF16 mask are inputs, so the
    graph holds no rotary or mask construction (the layer-0 boundary of the unchanged_fx backend route)."""

    def __init__(self, model, index):
        super().__init__()
        self.index = str(index)
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.layers = torch.nn.ModuleDict({self.index: model.model.layers[index]})
        # policy sites of this boundary; inspect_graph treats every other site as outside the graph
        self.policy_site_scope = (f"model.layers.{index}.",)

    def forward(self, hidden, cos, sin, attention_mask):
        # transformers >= 5 hangs output-capturing hooks on the layer's submodules that read a ContextVar the
        # model's own forward sets; strict export cannot trace an unset ContextVar.get, so set "no collector"
        # here exactly as the model forward does (inactive hooks, same arithmetic).
        collector = _output_collector()
        token = collector.set(None) if collector is not None else None
        try:
            out = self.model.model.layers[self.index](hidden, attention_mask=attention_mask,
                                                      position_embeddings=(cos, sin))
        finally:
            if collector is not None:
                collector.reset(token)
        return out[0] if isinstance(out, tuple) else out


def _output_collector():
    try:
        from transformers.utils.output_capturing import _active_collector
    except ImportError:
        return None
    return _active_collector


def causal_layer_cases(model, index, seqlen, seed, device, count=2):
    """`count` input tuples for CausalLayer: the hidden state entering layer `index` for random token ids
    (the model's own embedding and preceding layers), its rotary cos/sin at positions 0..seqlen-1 and a
    causal additive mask (0 on and below the diagonal, the BF16 minimum above), all BF16."""
    vocab = model.config.vocab_size
    cases = []
    with torch.no_grad():
        for i in range(count):
            ids = torch.randint(vocab, (1, seqlen), generator=torch.Generator().manual_seed(seed + i)).to(device)
            hidden = model(ids, use_cache=False, output_hidden_states=True).hidden_states[index]
            positions = torch.arange(seqlen, device=device)[None]
            cos, sin = model.model.rotary_emb(hidden, positions)
            dtype = hidden.dtype
            mask = torch.full((seqlen, seqlen), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
            cases.append(tuple(t.contiguous() for t in (hidden, cos.to(dtype), sin.to(dtype), mask[None, None])))
    return cases


# ---------------------------------------------------------------------------------------------------------------------
# Causal-LM inference phases for a host-owned token loop: prefill (fills the KV cache) and one decode step
# ---------------------------------------------------------------------------------------------------------------------
# Cache (both phases): one K and one V buffer per decoder layer, post-RoPE and pre-GQA, BF16
# [batch, n_kv_heads, C, head_dim] (C, the capacity, is a capture parameter): transformers' StaticCache layout, slot p
# holding position p, every slot zero until written. Both graphs attend over all C slots, as LlamaAttention does on a
# StaticCache (StaticLayer.update returns the whole buffer, create_causal_mask spans max_cache_len): the prefill over
# [its P new K / V; cache slots P..C-1], a decode step at length L over the cache with its new K / V in slot L
# (cache + slot * new, the host's one-hot slot; the cache holds zero there). The host mask hides the slots past the
# current position. The graphs write nothing: their new K / V are outputs, and the host writes them into its buffers.
# Why C keys and not the L + 1 valid ones: the float32 Q.K^T and P.V contractions of the reference attention round
# differently for different key counts (the zero products do not change the value, the summation order does), so a
# static graph equals transformers' static-cache loop word for word, not its DynamicCache loop (L + 1 keys).
# Unused slots must hold +0.0 (as StaticCache's): the K / V operand exponents are dynamic per (batch, head) over every
# key, masked or not, and a zero never raises the amax.

def _split_kv(flat):
    return [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]


class _CacheStandIn:
    """past_key_values for LlamaAttention at the phase boundaries (update() is the only method it calls): the keys and
    values StaticLayer.update returns, computed without writing into the cache inputs.

    past: {layer index: (k, v)}, the host's buffers [B, H_kv, C, D]. Prefill (slot None): P new K / V are slots 0..P-1,
    so the attention reads cat([new, past[:, :, P:]]) (P is static). Decode step (slot: BF16 [B, 1, C, 1], 1.0 at the
    write position L, 0 elsewhere): past + slot * new, which is the buffer with the new K / V in slot L when that slot
    holds zero, as StaticLayer.update's index_copy_ leaves it (x * 1 and x + 0 are exact)."""

    def __init__(self, past, slot=None):
        self.past, self.slot = past, slot
        self.new = {}

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if layer_idx in self.new:
            raise ValueError(f"cache stand-in: layer {layer_idx} updated twice")
        self.new[layer_idx] = (key_states, value_states)
        past_k, past_v = self.past[layer_idx]
        if self.slot is None:
            n = key_states.shape[-2]
            return torch.cat([key_states, past_k[:, :, n:]], dim=-2), torch.cat([value_states, past_v[:, :, n:]], dim=-2)
        return past_k + self.slot * key_states, past_v + self.slot * value_states

    def flat(self):
        return tuple(t for index in sorted(self.new) for t in self.new[index])


def _decoder_stack(layers, hidden, attention_mask, cos, sin, cache):
    """LlamaModel.forward's layer loop with the host mask and RoPE tables and the cache stand-in (the output-capturing
    ContextVar set to "no collector" as the model forward sets it; see CausalLayer)."""
    collector = _output_collector()
    token = collector.set(None) if collector is not None else None
    try:
        for layer in layers:
            out = layer(hidden, attention_mask=attention_mask, position_embeddings=(cos, sin), past_key_values=cache)
            hidden = out[0] if isinstance(out, tuple) else out
    finally:
        if collector is not None:
            collector.reset(token)
    return hidden


def _phase_tree(model, host_embeddings):
    """The modules a phase graph reads, under their original paths (model.model.layers.N, model.model.norm,
    model.lm_head, and model.model.embed_tokens unless the host gathers the embeddings), so the lifted state keeps the
    export names behind `model.`. The rotary embedding is not registered: the host computes the RoPE tables."""
    tree = torch.nn.Module()
    tree.model = torch.nn.Module()
    tree.model.layers = torch.nn.ModuleList(list(model.model.layers[:model.config.num_hidden_layers]))
    tree.model.norm = model.model.norm
    tree.lm_head = model.lm_head
    if not host_embeddings:
        tree.model.embed_tokens = model.model.embed_tokens
    return tree


class CausalPrefill(torch.nn.Module):
    """The prefill of a causal LM's greedy generation on a static cache (GenerationMixin.generate's first forward with
    a StaticCache of max_cache_len C) at its own boundary: the prompt and the empty cache -> the last position's logits
    and the post-RoPE K / V of every decoder layer for slots 0..P-1.

    Inputs: input_ids (int64 [B, P]) or, with host_embeddings, the token embeddings gathered on the host (BF16 [B, P,
    hidden]; causal_token_embedder), the host additive mask (BF16 [B, 1, P, C]: 0 where key <= query, the BF16 minimum
    elsewhere; causal_prefill_mask), the host RoPE rows 0..P-1 (cos, sin BF16 [B, P, head_dim]; causal_rope_tables),
    then k_0, v_0, ..., k_{L-1}, v_{L-1}: the cache buffers, BF16 [B, n_kv_heads, C, head_dim], all zero.
    Outputs: logits (BF16 [B, vocab], the last position; with all_logits [B, P, vocab]), then the new k_0, v_0, ...,
    BF16 [B, n_kv_heads, P, head_dim] each, for slots 0..P-1. The forward is LlamaModel's layer loop, its final norm and
    the LM head on the last position (generate's logits_to_keep=1), with the model's own modules; only the mask / RoPE
    construction and the cache object are replaced by inputs and a stand-in (_CacheStandIn)."""

    static_graph = True          # capture_fx refuses a dynamically shaped value (no dead-node pass: the graph is as traced)

    def __init__(self, model, host_embeddings=False, all_logits=False):
        super().__init__()
        self.model = _phase_tree(model, host_embeddings)
        self.host_embeddings = host_embeddings
        self.all_logits = all_logits

    def forward(self, tokens, attention_mask, cos, sin, *kv):
        inner = self.model.model
        if len(kv) != 2 * len(inner.layers):
            raise ValueError(f"expected {2 * len(inner.layers)} K / V inputs, got {len(kv)}")
        hidden = tokens if self.host_embeddings else inner.embed_tokens(tokens)
        cache = _CacheStandIn(dict(enumerate(_split_kv(kv))))
        hidden = _decoder_stack(inner.layers, hidden, attention_mask, cos, sin, cache)
        hidden = inner.norm(hidden)
        if self.all_logits:
            logits = self.model.lm_head(hidden)
        else:
            logits = self.model.lm_head(hidden[:, -1:]).view(hidden.shape[0], -1)
        return (logits, *cache.flat())


class CausalDecodeStep(torch.nn.Module):
    """One decode step of a causal LM's greedy generation on a static cache at its own boundary: one new token and the
    KV cache -> logits and the new token's K / V. The host owns the loop (argmax, cache writes, mask, slot and RoPE row
    per step; causal_host_loop, loop_manifest.json).

    Inputs: input_ids (int64 [B, 1]) or, with host_embeddings, its embedding (BF16 [B, 1, hidden]), the host additive
    mask (BF16 [B, 1, 1, C]; causal_decode_mask: 0 on slots 0..L, the BF16 minimum on slots L+1..C-1), the host RoPE row
    of position L (cos, sin BF16 [B, 1, head_dim]), the write slot (BF16 [B, 1, C, 1], 1.0 at L; causal_decode_slot),
    then k_0, v_0, ..., k_{L-1}, v_{L-1}: the cache, BF16 [B, n_kv_heads, C, head_dim] each, zero from slot L on.
    Outputs: logits (BF16 [B, vocab]), then the new token's k_0, v_0, ..., BF16 [B, n_kv_heads, 1, head_dim] each; the
    host writes them into slot L. Every shape is static; the length L reaches the graph only through the mask, the slot
    and the RoPE row, so nothing is sliced by a run-time length and no input is mutated."""

    static_graph = True          # capture_fx refuses a dynamically shaped value (no dead-node pass: the graph is as traced)

    def __init__(self, model, host_embeddings=False):
        super().__init__()
        self.model = _phase_tree(model, host_embeddings)
        self.host_embeddings = host_embeddings

    def forward(self, tokens, attention_mask, cos, sin, slot, *kv):
        inner = self.model.model
        if len(kv) != 2 * len(inner.layers):
            raise ValueError(f"expected {2 * len(inner.layers)} K / V inputs, got {len(kv)}")
        hidden = tokens if self.host_embeddings else inner.embed_tokens(tokens)
        cache = _CacheStandIn(dict(enumerate(_split_kv(kv))), slot)
        hidden = _decoder_stack(inner.layers, hidden, attention_mask, cos, sin, cache)
        hidden = inner.norm(hidden)
        logits = self.model.lm_head(hidden).view(hidden.shape[0], -1)
        return (logits, *cache.flat())


class CausalDecodeLayer(torch.nn.Module):
    """One decoder layer of CausalDecodeStep at its own boundary (the per-layer route of the backend):
    (hidden [B, 1, hidden], cos, sin [B, 1, head_dim], mask [B, 1, 1, C], slot [B, 1, C, 1], k, v [B, n_kv_heads, C,
    head_dim]) -> (hidden, new k, new v [B, n_kv_heads, 1, head_dim]); the input order of CausalLayer, then the slot and
    the layer's cache. Registered under its original path as CausalLayer, so the lifted buffers keep the export names
    behind `model.` and the policy sites keep their identity."""

    static_graph = True          # capture_fx refuses a dynamically shaped value (no dead-node pass: the graph is as traced)

    def __init__(self, model, index):
        super().__init__()
        self.index = str(index)
        self.model = torch.nn.Module()
        self.model.model = torch.nn.Module()
        self.model.model.layers = torch.nn.ModuleDict({self.index: model.model.layers[index]})
        self.policy_site_scope = (f"model.layers.{index}.",)

    def forward(self, hidden, cos, sin, attention_mask, slot, past_k, past_v):
        layer = self.model.model.layers[self.index]
        cache = _CacheStandIn({layer.self_attn.layer_idx: (past_k, past_v)}, slot)
        hidden = _decoder_stack([layer], hidden, attention_mask, cos, sin, cache)
        return (hidden, *cache.flat())


# --- host side: what the host computes around the two graphs (loop_manifest.json "host") ------------------------------

def causal_geometry(model, capacity, prefill_length):
    """The cache geometry and loop constants the host needs, read from the model config."""
    config = model.config
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    if not 0 < prefill_length < capacity:
        raise ValueError(f"prefill length {prefill_length} leaves no decode slot in capacity {capacity}")
    return SimpleNamespace(layers=config.num_hidden_layers, n_heads=config.num_attention_heads,
                           n_kv_heads=config.num_key_value_heads, head_dim=head_dim, hidden=config.hidden_size,
                           vocab=config.vocab_size, capacity=capacity, prefill_length=prefill_length,
                           dtype=model.model.norm.weight.dtype)


def causal_rope_tables(model, capacity, batch=1, dtype=torch.bfloat16, device="cpu"):
    """(cos, sin) BF16 [batch, capacity, head_dim]: the model's own rotary embedding (LlamaRotaryEmbedding.forward:
    float32 inv_freq x position outer product, cat, cos / sin, attention_scaling, cast to the activation dtype) at
    positions 0..capacity-1. Row p is what LlamaModel.forward computes for position p; the prefill takes rows 0..P-1,
    the decode step at length L row L."""
    x = torch.zeros(1, dtype=dtype, device=device)
    positions = torch.arange(capacity, device=device)[None].expand(batch, -1)
    cos, sin = model.model.rotary_emb(x, positions)
    return cos.contiguous(), sin.contiguous()


def causal_prefill_mask(length, capacity, batch=1, dtype=torch.bfloat16, device="cpu"):
    """BF16 [batch, 1, length, capacity]: 0 where key <= query, the dtype minimum elsewhere (transformers'
    create_causal_mask with the eager mask function for an unpadded prompt on a StaticCache of max_cache_len capacity)."""
    mask = torch.full((length, capacity), torch.finfo(dtype).min, dtype=dtype, device=device).triu(1)
    return mask[None, None].expand(batch, 1, length, capacity).contiguous()


def causal_decode_mask(length, capacity, batch=1, dtype=torch.bfloat16, device="cpu"):
    """BF16 [batch, 1, 1, capacity] for a decode step at cache length `length` (the new token in slot `length`): 0 on
    slots 0..length, the dtype minimum on slots length+1..capacity-1."""
    if not 0 <= length < capacity:
        raise ValueError(f"cache length {length} outside 0..{capacity - 1}")
    mask = torch.zeros(batch, 1, 1, capacity, dtype=dtype, device=device)
    mask[..., length + 1:] = torch.finfo(dtype).min
    return mask


def causal_decode_slot(length, capacity, batch=1, dtype=torch.bfloat16, device="cpu"):
    """BF16 [batch, 1, capacity, 1]: 1.0 at slot `length` (where the step's new K / V go), 0 elsewhere."""
    if not 0 <= length < capacity:
        raise ValueError(f"cache length {length} outside 0..{capacity - 1}")
    slot = torch.zeros(batch, 1, capacity, 1, dtype=dtype, device=device)
    slot[:, :, length] = 1
    return slot


def causal_empty_cache(geometry, batch=1, device="cpu"):
    """2 * layers zero buffers BF16 [batch, n_kv_heads, capacity, head_dim] (k_0, v_0, k_1, ...)."""
    return [torch.zeros(batch, geometry.n_kv_heads, geometry.capacity, geometry.head_dim, dtype=geometry.dtype,
                        device=device) for _ in range(2 * geometry.layers)]


def causal_cache_write(cache, new, position):
    """Write each graph output K / V [B, H_kv, n, D] into its buffer at slots position..position+n-1 (host side)."""
    if len(cache) != len(new):
        raise ValueError(f"{len(new)} new K / V tensors for {len(cache)} cache buffers")
    for buffer, value in zip(cache, new):
        if position + value.shape[2] > buffer.shape[2]:
            raise ValueError(f"cache capacity {buffer.shape[2]} exceeded at position {position} + {value.shape[2]}")
        buffer[:, :, position:position + value.shape[2]] = value


def causal_token_embedder(tensors, policy, name="model.embed_tokens"):
    """ids -> BF16 token embeddings gathered on the host from the export's own tensors (the loop manifest's recipe):
    a BF16 table: its rows; an FP8 embedding (exporter.head_embed, --embedding fp8): the stored decode table at the
    code rows times 2^(row exponent), cast to BF16 (QuantEmbedding.forward, exact in BF16)."""
    site = policy["sites"].get(name, {})
    if site.get("kind") != "embedding" or site.get("status") != "quantized":
        table = tensors[f"{name}.weight"]
        return lambda ids: torch.nn.functional.embedding(ids, table)
    if site.get("format") != "fp8" or site.get("tied_to"):
        raise ValueError(f"host embeddings cover a BF16 table or an untied FP8 embedding, not {site.get('format')!r}"
                         f"{' tied' if site.get('tied_to') else ''}")
    codes, exp = tensors[site["weight"]["codes"]], tensors[site["weight"]["scale_exp"]]
    lut = tensors[site["decode_tables"]["weight_decode_atlas_lut"]]
    dtype = getattr(torch, site.get("weight_dtype", "bfloat16"))
    return lambda ids: (lut[torch.nn.functional.embedding(ids, codes).long()]
                        * torch.exp2(torch.nn.functional.embedding(ids, exp).float())).to(dtype)


def causal_prefill_inputs(tokens, rope, cache, embed=None):
    """Prefill graph inputs for a prompt (int64 [B, P]): tokens or host embeddings, mask, cos / sin rows 0..P-1, the
    (zero) cache buffers."""
    batch, length = tokens.shape
    cos, sin = rope
    first = tokens if embed is None else embed(tokens)
    return (first, causal_prefill_mask(length, cache[0].shape[2], batch, cos.dtype, cos.device),
            cos[:, :length].contiguous(), sin[:, :length].contiguous(), *cache)


def causal_step_inputs(token, length, capacity, rope, cache, embed=None):
    """Decode-step graph inputs for one token (int64 [B, 1]) at cache length `length`."""
    cos, sin = rope
    first = token if embed is None else embed(token)
    batch = token.shape[0]
    return (first, causal_decode_mask(length, capacity, batch, cos.dtype, cos.device),
            cos[:, length:length + 1].contiguous(), sin[:, length:length + 1].contiguous(),
            causal_decode_slot(length, capacity, batch, cos.dtype, cos.device), *cache)


def causal_host_loop(prefill, step, prompt, n_steps, geometry, rope, embed=None, eos_token_id=None):
    """The greedy token loop the host runs around the two graphs (loop_manifest.json "loop"): zero the cache, prefill
    once, write its K / V into slots 0..P-1, then token = argmax(logits) (torch.argmax: the first index on ties, as
    generate's greedy search); stop after an EOS token or after n_steps decode steps, otherwise run the step at length
    L, write its K / V into slot L, L += 1. Returns (tokens [B, n] int64, the prefill's token first, [logits per
    generated token], cache buffers)."""
    if prompt.shape[1] != geometry.prefill_length:
        raise ValueError(f"prompt of {prompt.shape[1]} tokens; the prefill graph takes {geometry.prefill_length}")
    cache = causal_empty_cache(geometry, prompt.shape[0], prompt.device)
    out = prefill(*causal_prefill_inputs(prompt, rope, cache, embed))
    causal_cache_write(cache, out[1:], 0)
    length, tokens, all_logits = prompt.shape[1], [], [out[0]]
    while True:
        token = torch.argmax(all_logits[-1], dim=-1, keepdim=True)
        tokens.append(token)
        if len(tokens) > n_steps or (eos_token_id is not None and bool((token == eos_token_id).all())):
            break
        out = step(*causal_step_inputs(token, length, geometry.capacity, rope, cache, embed))
        all_logits.append(out[0])
        causal_cache_write(cache, out[1:], length)
        length += 1
    return torch.cat(tokens, dim=1), all_logits, cache


def causal_generate(model, prompt, n_steps, capacity=None):
    """transformers' own greedy loop on the same model: generate(do_sample=False, use_cache=True) with
    max_new_tokens = n_steps + 1 (the prefill's token and one per decode step), on a StaticCache of max_cache_len
    `capacity` (the loop the phase graphs reproduce), or with capacity None on generate's default DynamicCache. Returns
    (new tokens [B, n], [raw logits per generated token], as generate records them: float32)."""
    kwargs = {}
    if capacity is not None:
        from transformers import StaticCache

        kwargs["past_key_values"] = StaticCache(config=model.config, max_cache_len=capacity)
    out = model.generate(prompt, max_new_tokens=n_steps + 1, do_sample=False, use_cache=True, output_logits=True,
                         return_dict_in_generate=True, num_beams=1, **kwargs)
    return out.sequences[:, prompt.shape[1]:], list(out.logits)


def causal_phase_cases(model, boundary, geometry, seed, device, count=2, host_embeddings=False, steps=(0, 3),
                       embed=None, prompts=None):
    """`count` input tuples for a CausalPrefill / CausalDecodeStep / CausalDecodeLayer capture on prompts of the
    prefill length (`prompts`, int64 [count, P], or random ids from seed + i). A decode-step case is the greedy loop's
    input at decode step steps[i] of the eager phases (a cache of real K / V, different lengths per case); a
    decode-layer case is that step's input to layer `boundary.index` (host_embeddings selects the step's token input)."""
    vocab = model.config.vocab_size
    if host_embeddings and embed is None:
        embed = model.model.embed_tokens
    embed = embed if host_embeddings else None
    rope = causal_rope_tables(model, geometry.capacity, dtype=geometry.dtype, device=device)
    prefill = CausalPrefill(model, host_embeddings=host_embeddings).eval()
    step = CausalDecodeStep(model, host_embeddings=host_embeddings).eval()
    cases = []
    with torch.no_grad():
        for i in range(count):
            prompt = (prompts[i:i + 1].to(device) if prompts is not None else torch.randint(
                vocab, (1, geometry.prefill_length), generator=torch.Generator().manual_seed(seed + i)).to(device))
            if isinstance(boundary, CausalPrefill):
                cache = causal_empty_cache(geometry, 1, device)
                cases.append(tuple(t.clone() for t in causal_prefill_inputs(prompt, rope, cache, embed)))  # own storage
                continue
            at = steps[i % len(steps)] + i // len(steps)
            captured = []

            def record(*args):
                captured.append(tuple(t.clone() for t in args))     # the host writes into its buffers after the step
                return step(*args)
            causal_host_loop(prefill, record, prompt, at + 1, geometry, rope, embed)
            args = captured[at]
            if isinstance(boundary, CausalDecodeStep):
                cases.append(args)
                continue
            index = int(boundary.index)
            token, mask, cos, sin, slot, *kv = args
            hidden = token if host_embeddings else model.model.embed_tokens(token)
            hidden = _decoder_stack(model.model.layers[:index], hidden, mask, cos, sin,
                                    _CacheStandIn(dict(enumerate(_split_kv(kv))), slot))
            cases.append(tuple(t.contiguous() for t in (hidden, cos, sin, mask, slot, kv[2 * index], kv[2 * index + 1])))
    return cases


def _rope_head_dim(policy):
    inner = policy.model.vlm_with_expert
    dims = {inner.vlm.config.text_config.head_dim, inner.lm_expert.config.head_dim}
    if len(dims) != 1:
        raise ValueError(f"host RoPE needs one head_dim for the VLM and the expert: {dims}")
    return dims.pop()


def _text_inputs(boundary, batch, length, device, rest):
    """(position_ids for SmolVLMWithExpertModel.forward) from the boundary's trailing inputs. Without host_rope that
    is the position-id input. With host_rope the trailing inputs are the six host RoPE tables (cos, sin per
    ROPE_TABLES entry), installed for apply_rope, and the forward gets an all-zero position tensor of the right shape
    whose values nothing reads (its slices only select the table; the expert's min-subtraction is dead)."""
    if not boundary.host_rope:
        (position_ids,) = rest
        return position_ids
    if len(rest) != 2 * len(ROPE_TABLES):
        raise ValueError(f"host_rope: expected {2 * len(ROPE_TABLES)} table inputs, got {len(rest)}")
    boundary._rope.tables = {name: (rest[2 * i], rest[2 * i + 1]) for i, name in enumerate(ROPE_TABLES)}
    return torch.zeros(batch, length, dtype=torch.int64, device=device)


class SmolVLAFlowLoss(torch.nn.Module):
    """Tensor-in / tensor-out boundary of a LeRobot SmolVLA policy: the flow-matching loss forward
    (`VLAFlowMatching.forward`) for given images, language tokens, state, actions, noise and time, with the text /
    expert attention mask and position ids as inputs (smolvla_attention_inputs; host-computed from the pad and block
    masks with lerobot's own functions, as TinyLlama's layer boundary takes its mask). It runs the vision tower
    (patch-embedding Conv2d, vision attention), the connector, the text/expert layers and the action projections
    once, with no sampling loop, no KV cache and no random draw; the full action sampler (10 denoising steps with a
    KV cache) is a different boundary.

    Inputs: n_images images, lang_tokens, state, actions, noise, time, attention_mask (BF16 additive [B, S, S]),
    position_ids (int64 [B, S]); with host_text_embeddings, lang_tokens is replaced by lerobot's
    embed_language_tokens(lang_tokens) computed on the host (BF16 [B, L, hidden]; _HostTokenEmbeddings), so the graph
    holds no token embedding_lookup (embed_prefix still scales it in the graph); with host_time_embedding, a last input
    holds the flow-timestep embedding computed on the host (smolvla_time_embedding, _HostTimeEmbedding; `time` stays an
    input for x_t); with host_rope, the six host RoPE tables (smolvla_rope_tables: joint, prefix, expert
    cos / sin, float32) instead of the position ids, so no RoPE angle, pow, sin or cos is computed in the graph. The forward is VLAFlowMatching.forward with make_att_2d_masks and the cumsum
    replaced by the two inputs; the pad / block masks embed_prefix / embed_suffix return only feed those, so the
    language pad mask is not an input (embed_prefix gets an all-true one, which the graph never reads)."""

    static_graph = True
    drop_dead_nodes = True
    bias_add = True
    forbidden_graph_targets = ("aten.cumsum.default",)

    def __init__(self, policy, n_images=1, host_rope=False, host_text_embeddings=False, host_time_embedding=False):
        super().__init__()
        self.policy = policy
        self.n_images = n_images
        self.host_rope = host_rope
        self.host_text_embeddings = host_text_embeddings
        self.host_time_embedding = host_time_embedding
        self._rope = _HostRope()
        self._tokens = _HostTokenEmbeddings(policy.model.vlm_with_expert)
        self._time = _HostTimeEmbedding()
        if host_rope:
            self._rope.install()
        if host_text_embeddings:
            self._tokens.install()
        if host_time_embedding:
            self._time.install()

    def restore(self):
        self._rope.restore()
        self._tokens.restore()
        self._time.restore()

    def forward(self, *tensors):
        from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs

        model = self.policy.model
        images = list(tensors[:self.n_images])
        lang_tokens, state, actions, noise, time, attention_mask, *rest = tensors[self.n_images:]
        if self.host_time_embedding:
            self._time.embedding, rest = rest[-1], rest[:-1]
        position_ids = _text_inputs(self, attention_mask.shape[0], attention_mask.shape[-1], attention_mask.device, rest)
        if self.host_text_embeddings:
            # lang_tokens is the host embedding; embed_prefix gets an all-zero token tensor of its shape, whose values
            # nothing reads (the patched embed_language_tokens returns the host embedding)
            self._tokens.embeddings = lang_tokens
            lang_tokens = torch.zeros(lang_tokens.shape[:-1], dtype=torch.int64, device=lang_tokens.device)
        masks = [torch.ones(image.shape[0], dtype=torch.bool, device=image.device) for image in images]
        x_t, u_t, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
        prefix_embs, _, _ = model.embed_prefix(images, masks, lang_tokens, torch.ones_like(lang_tokens, dtype=torch.bool),
                                               state=state)
        suffix_embs, _, _ = model.embed_suffix(x_t, time)
        (_, suffix_out), _ = model.vlm_with_expert.forward(
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
        suffix_out = suffix_out[:, -model.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = model.action_out_proj(suffix_out)
        return torch.nn.functional.mse_loss(u_t, v_t, reduction="none")


class SmolVLATextLayers(torch.nn.Module):
    """The first `layers` layers of a LeRobot SmolVLA policy's VLM text model and action expert (SmolVLMWithExpertModel,
    use_cache=False, as VLAFlowMatching.forward runs them) at their own boundary: (prefix embeddings, suffix embeddings,
    attention mask, position ids) -> suffix output after the expert's final norm. The attention mask (BF16 additive
    [B, S, S]) and the position ids (int64 [B, S]) are inputs, computed on the host by smolvla_attention_inputs from the
    pad and block masks exactly as the flow-loss forward builds them; the vision tower, the connector and the action
    projections are outside (their sites are listed as unused). Layer 0 is joint self-attention and layer 1 the
    prefix + expert cross-attention when self_attn_every_n_layers is 2, so two layers hold both masked attention
    patterns.

    The boundary sets the instance's num_vlm_layers / num_expert_layers to `layers` (so get_model_layers pairs expert
    layer i with VLM layer i and forward stops after `layers` layers); restore() puts the counts back (and, with
    host_rope, apply_rope: the trailing input is then the six host RoPE tables instead of the position ids)."""

    static_graph = True
    drop_dead_nodes = True
    bias_add = True
    forbidden_graph_targets = ("aten.cumsum.default",)

    def __init__(self, policy, layers=2, host_rope=False):
        super().__init__()
        self.policy = policy
        self.host_rope = host_rope
        self._rope = _HostRope()
        inner = policy.model.vlm_with_expert
        if not 1 <= layers <= min(inner.num_vlm_layers, inner.num_expert_layers):
            raise ValueError(f"layers {layers} outside 1..{min(inner.num_vlm_layers, inner.num_expert_layers)}")
        if inner.num_vlm_layers != inner.num_expert_layers:
            raise ValueError("SmolVLATextLayers needs one expert layer per VLM layer")
        self._saved = (inner.num_vlm_layers, inner.num_expert_layers)
        inner.num_vlm_layers = inner.num_expert_layers = layers
        if host_rope:
            self._rope.install()

    def restore(self):
        inner = self.policy.model.vlm_with_expert
        inner.num_vlm_layers, inner.num_expert_layers = self._saved
        self._rope.restore()

    def forward(self, prefix_embs, suffix_embs, attention_mask, *rest):
        position_ids = _text_inputs(self, attention_mask.shape[0], attention_mask.shape[-1], attention_mask.device, rest)
        (_, suffix_out), _ = self.policy.model.vlm_with_expert.forward(
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs], use_cache=False)
        return suffix_out


# -- SmolVLA inference phases: the prefix pass (KV cache) and one denoising step -------------------------------------------

STEP_ROPE_TABLES = ("suffix", "expert")


class _KVLayer:
    def __init__(self, keys, values):
        self.keys, self.values = keys, values


class _PrefixKV:
    """The KV cache of SmolVLA's sample_actions as plain tensors. Inside, as in DynamicCache, a layer holds
    [B, H_kv, P, D]; at the boundaries (kv_tensors(), the step's inputs) a layer is [B, P, H_kv, D], the layout
    k_proj / v_proj produce and SmolVLMWithExpertModel attends in, so the transposes cancel and no copy is made.

    Prefill (built empty): forward_attn_layer hands each layer's post-RoPE K / V to update(), which keeps them and
    returns them unchanged. DynamicCache.update on an empty layer returns cat([empty, k]), the same values; here no
    empty constant or concatenation enters the graph.
    Denoising step (built from the K / V inputs): layers[i].keys / .values are the inputs (the cross-attention layers
    read them), and update() returns cat([prefix, new]) along the sequence axis, DynamicLayer.update's concatenation,
    without storing it. Nothing is written into the inputs, so no crop back to the prefix is needed after the step."""

    def __init__(self, kv=()):
        self.layers = [_KVLayer(k.transpose(1, 2), v.transpose(1, 2)) for k, v in kv]
        self.read_only = len(self.layers) > 0

    def kv_tensors(self):
        return [(layer.keys.transpose(1, 2), layer.values.transpose(1, 2)) for layer in self.layers]

    def __len__(self):
        return len(self.layers)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if not self.read_only:
            if layer_idx != len(self.layers):
                raise ValueError(f"prefill stores layer {layer_idx} after {len(self.layers)} layers")
            self.layers.append(_KVLayer(key_states, value_states))
            return key_states, value_states
        layer = self.layers[layer_idx]
        return torch.cat([layer.keys, key_states], dim=-2), torch.cat([layer.values, value_states], dim=-2)


def _smolvla_layer_tail(layer, att_output, hidden_states):
    """The rest of a VLM text layer after its attention, for the prefix stream alone: SmolVLMWithExpertModel.forward's
    per-layer loop body (o_proj, residual, post-attention norm, MLP, residual), line for line."""
    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
    start, end = 0, hidden_states.shape[1]
    att_out = att_output[:, start:end]
    out_emb = layer.self_attn.o_proj(att_out)
    out_emb += hidden_states
    after_first_residual = out_emb.clone()
    out_emb = layer.post_attention_layernorm(out_emb)
    out_emb = layer.mlp(out_emb)
    out_emb += after_first_residual
    return out_emb


def _smolvla_layer_kv(layer, hidden_states, position_ids):
    """The K / V forward_attn_layer stores for a prefix-only layer, without its query, attention, o_proj and MLP: the
    input norm, k_proj / v_proj, RoPE on the keys (smolvlm_with_expert.py forward_attn_layer, line for line; its
    one-element concatenation over the streams is an identity and is left out). Returned as forward_attn_layer hands
    them to the cache, transposed to [B, H_kv, P, D]."""
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    hidden_states = layer.input_layernorm(hidden_states)
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
    hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)
    key_state = swe.apply_rope(key_state, position_ids)
    return key_state.transpose(1, 2), value_state.transpose(1, 2)


def smolvla_prefix_kv(inner, prefix_embs, attention_mask, position_ids):
    """[(k_i, v_i)] for every VLM text layer: what SmolVLMWithExpertModel.forward(inputs_embeds=[prefix, None],
    use_cache=True, past_key_values=None) stores in its DynamicCache (sample_actions' prefix pass). Every layer runs
    forward_attn_layer (lerobot's own) and the loop body of forward; the last layer stops after its K / V, because its
    query, attention, o_proj, MLP and the final norm reach no cache entry (in sample_actions they are computed and
    dropped)."""
    models = [inner.get_vlm_model().text_model, inner.lm_expert]
    model_layers = inner.get_model_layers(models)
    batch_size = prefix_embs.shape[0]
    head_dim = inner.vlm.config.text_config.head_dim
    cache = _PrefixKV()
    hidden = prefix_embs
    last = inner.num_vlm_layers - 1
    for layer_idx in range(inner.num_vlm_layers):
        layer = model_layers[0][layer_idx]
        if layer_idx == last:
            cache.update(*_smolvla_layer_kv(layer, hidden, position_ids), layer_idx)
            break
        att_outputs, _ = inner.forward_attn_layer(model_layers, [hidden, None], layer_idx, position_ids, attention_mask,
                                                  batch_size, head_dim, use_cache=True, past_key_values=cache)
        hidden = _smolvla_layer_tail(layer, att_outputs[0], hidden)
    return cache.kv_tensors()


class SmolVLAPrefix(torch.nn.Module):
    """The prefix pass of SmolVLA's action sampling (VLAFlowMatching.sample_actions up to the KV cache) at its own
    boundary: images, language tokens and state -> the post-RoPE K / V of every VLM text layer, once per action chunk.

    Inputs: n_images images, lang_tokens, state, attention_mask (BF16 additive [B, P, P]), position_ids (int64 [B, P]),
    with host_rope the host "prefix" RoPE table (cos, sin; float32 [B, P, 1, head_dim / 2]) instead of the position ids,
    with host_text_embeddings lerobot's embed_language_tokens(lang_tokens) computed on the host (BF16 [B, L, hidden];
    _HostTokenEmbeddings, as at the flow-loss boundary) instead of the token ids.
    Both come from smolvla_prefix_inputs (lerobot's make_att_2d_masks and cumsum on embed_prefix's pad / block masks),
    as at the flow-loss boundary; the language pad mask is therefore not an input (embed_prefix gets an all-true one,
    which only feeds those masks).
    Outputs: k_0, v_0, ..., k_{L-1}, v_{L-1}, BF16 [B, P, H_kv, head_dim] each (DynamicCache holds the same values
    transposed, [B, H_kv, P, head_dim]; this is the layout k_proj / v_proj produce, so no copy). The self-attention
    layers of the denoising step append the suffix K / V to theirs; the cross-attention layers project them with the
    expert's k_proj / v_proj (SmolVLADenoiseStep). The host keeps the prefix pad mask (smolvla_prefix_masks) for the
    step's mask and positions."""

    static_graph = True
    drop_dead_nodes = True
    bias_add = True
    forbidden_graph_targets = ("aten.cumsum.default",)

    def __init__(self, policy, n_images=1, host_rope=False, host_text_embeddings=False):
        super().__init__()
        self.policy = policy
        self.n_images = n_images
        self.host_rope = host_rope
        self.host_text_embeddings = host_text_embeddings
        self._rope = _HostRope()
        self._tokens = _HostTokenEmbeddings(policy.model.vlm_with_expert)
        if host_rope:
            self._rope.install()
        if host_text_embeddings:
            self._tokens.install()

    def restore(self):
        self._rope.restore()
        self._tokens.restore()

    def forward(self, *tensors):
        model = self.policy.model
        images = list(tensors[:self.n_images])
        lang_tokens, state, attention_mask, *rest = tensors[self.n_images:]
        if self.host_text_embeddings:
            # lang_tokens is the host embedding; embed_prefix gets an all-zero token tensor of its shape, whose values
            # nothing reads (the patched embed_language_tokens returns the host embedding)
            self._tokens.embeddings = lang_tokens
            lang_tokens = torch.zeros(lang_tokens.shape[:-1], dtype=torch.int64, device=lang_tokens.device)
        batch, length = attention_mask.shape[0], attention_mask.shape[-1]
        if self.host_rope:
            if len(rest) != 2:
                raise ValueError(f"host_rope: expected the prefix (cos, sin) table, got {len(rest)} inputs")
            self._rope.tables = {"prefix": (rest[0], rest[1])}
            position_ids = torch.zeros(batch, length, dtype=torch.int64, device=attention_mask.device)
        else:
            (position_ids,) = rest
        masks = [torch.ones(image.shape[0], dtype=torch.bool, device=image.device) for image in images]
        prefix_embs, _, _ = model.embed_prefix(images, masks, lang_tokens, torch.ones_like(lang_tokens, dtype=torch.bool),
                                               state=state)
        kv = smolvla_prefix_kv(model.vlm_with_expert, prefix_embs, attention_mask, position_ids)
        return tuple(t for pair in kv for t in pair)


def _with_rope_role(method, rope, role):
    def call(*args, **kwargs):
        rope.role = role
        try:
            return method(*args, **kwargs)
        finally:
            rope.role = None
    return call


class SmolVLADenoiseStep(torch.nn.Module):
    """One denoising step of SmolVLA's action sampling (VLAFlowMatching.denoise_step) at its own boundary: the prefix
    K / V, the noisy actions x_t and the time t -> the velocity v_t. The host owns the Euler loop
    (x_t + dt * v_t over euler_integrate's time grid, loop_manifest.json) and calls this num_steps times per chunk.

    Inputs: x_t (float32 [B, chunk, max_action_dim]), time (float32 [B]), with host_time_embedding the flow-timestep
    embedding computed on the host (smolvla_time_inputs: lerobot's create_sinusoidal_pos_embedding cast as embed_suffix
    casts it; _HostTimeEmbedding), attention_mask (BF16 additive
    [B, chunk, P + chunk]), position_ids (int64 [B, chunk]) or with host_rope the "suffix" and "expert" RoPE tables
    (cos, sin each), then k_0, v_0, ..., k_{L-1}, v_{L-1} (SmolVLAPrefix outputs). smolvla_step_host_inputs computes the
    mask and positions on the host exactly as denoise_step does in its graph (they depend only on the prefix pad mask,
    so they are the same for every step of a chunk).
    The forward is denoise_step with that mask and those positions, and the cache replaced by _PrefixKV over the
    inputs: the self-attention layers concatenate [prefix; suffix] K / V, the cross-attention layers read the prefix
    K / V; nothing is written into the inputs, so the step is pure and no crop follows.

    Host RoPE: both kinds of layer rotate chunk-length positions (self-attention: prefix_offset + 0..chunk-1 on q and k;
    cross-attention: 0..chunk-1 on the expert query), so a table cannot be chosen by length. With host_rope the instance's
    forward_attn_layer / forward_cross_attn_layer are wrapped to name the table ("suffix" / "expert") for the duration
    of the call; restore() removes the wrappers."""

    static_graph = True
    drop_dead_nodes = True
    bias_add = True
    forbidden_graph_targets = ("aten.cumsum.default",)

    def __init__(self, policy, host_rope=False, host_time_embedding=False):
        super().__init__()
        self.policy = policy
        self.host_rope = host_rope
        self.host_time_embedding = host_time_embedding
        self._rope = _HostRope()
        self._time = _HostTimeEmbedding()
        self._wrapped = []
        if host_time_embedding:
            self._time.install()
        if not policy.model.config.use_cache:
            raise ValueError("config.use_cache is off: denoise_step recomputes the prefix and reads no KV cache")
        self.layers = policy.model.vlm_with_expert.num_vlm_layers
        if host_rope:
            self._rope.install()
            inner = policy.model.vlm_with_expert
            for name, role in (("forward_attn_layer", "suffix"), ("forward_cross_attn_layer", "expert")):
                self._wrapped.append((name, inner.__dict__.get(name)))
                setattr(inner, name, _with_rope_role(getattr(inner, name), self._rope, role))

    def restore(self):
        inner = self.policy.model.vlm_with_expert
        for name, previous in reversed(self._wrapped):
            if previous is None:
                delattr(inner, name)
            else:
                setattr(inner, name, previous)
        self._wrapped = []
        self._rope.restore()
        self._time.restore()

    def forward(self, x_t, time, *rest):
        model = self.policy.model
        if self.host_time_embedding:
            self._time.embedding, rest = rest[0], rest[1:]
        attention_mask, *rest = rest
        n_kv = 2 * self.layers
        if len(rest) < n_kv:
            raise ValueError(f"expected {n_kv} K / V inputs, got {len(rest)} trailing inputs")
        positions, kv = rest[:-n_kv], rest[-n_kv:]
        if self.host_rope:
            if len(positions) != 2 * len(STEP_ROPE_TABLES):
                raise ValueError(f"host_rope: expected {2 * len(STEP_ROPE_TABLES)} table inputs, got {len(positions)}")
            self._rope.tables = {name: (positions[2 * i], positions[2 * i + 1]) for i, name in enumerate(STEP_ROPE_TABLES)}
            position_ids = torch.zeros(x_t.shape[0], x_t.shape[1], dtype=torch.int64, device=x_t.device)
        else:
            (position_ids,) = positions
        suffix_embs, _, _ = model.embed_suffix(x_t, time)
        outputs_embeds, _ = model.vlm_with_expert.forward(
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=_PrefixKV(_split_kv(kv)),
            inputs_embeds=[None, suffix_embs], use_cache=True)
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -model.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return model.action_out_proj(suffix_out)


def smolvla_geometry(policy):
    """What the host needs to build the SmolVLA phase inputs: tokens_per_image ((image_size / patch_size)^2 /
    scale_factor^2 of the VLM config, the connector's pixel shuffle), chunk_size, the RoPE head_dim and the time
    embedding's dimension, periods and dtype (smolvla_time_inputs). Refuses the
    configurations whose embed_prefix adds more tokens (image special tokens, prefix_length padding). The loop manifest
    records the same three numbers (compose_phases.py builds the host inputs from those)."""
    model = policy.model
    if model.add_image_special_tokens or model.prefix_length > 0:
        raise ValueError("the SmolVLA phase inputs cover add_image_special_tokens=False and prefix_length=0")
    vlm_config = model.vlm_with_expert.vlm.config
    side = vlm_config.vision_config.image_size // vlm_config.vision_config.patch_size
    with torch.no_grad():           # embed_suffix casts the time embedding to action_in_proj's output dtype
        time_dtype = model.action_in_proj(torch.zeros(1, 1, model.config.max_action_dim, dtype=torch.float32,
                                                      device=next(model.parameters()).device)).dtype
    return SimpleNamespace(tokens_per_image=side * side // vlm_config.scale_factor ** 2,
                           chunk_size=model.config.chunk_size, head_dim=_rope_head_dim(policy),
                           time_dim=model.vlm_with_expert.expert_hidden_size, min_period=model.config.min_period,
                           max_period=model.config.max_period, time_dtype=time_dtype)


def smolvla_prefix_masks(geometry, img_masks, lang_masks):
    """(pad_masks, att_masks) of embed_prefix, without embedding anything: per camera tokens_per_image copies of its
    image mask, the language mask, one state token; block mask 0 everywhere but the state token (1)."""
    batch, device = lang_masks.shape[0], lang_masks.device
    pads = [m[:, None].expand(batch, geometry.tokens_per_image) for m in img_masks]
    pads += [lang_masks, torch.ones(batch, 1, dtype=torch.bool, device=device)]
    pad = torch.cat(pads, dim=1)
    att = torch.zeros(pad.shape[1], dtype=torch.bool, device=device)
    att[-1] = True
    return pad, att[None, :].expand(batch, -1)


def smolvla_prefix_inputs(geometry, images, img_masks, lang_tokens, lang_masks, state, host_rope=False, embed=None):
    """(SmolVLAPrefix inputs, prefix_pad_masks) for one observation: sample_actions' make_att_2d_masks and cumsum on
    the prefix masks (smolvla_attention_inputs), with host_rope the "prefix" RoPE table of those positions. With
    `embed` (host_text_embeddings): the language tokens are replaced by embed(lang_tokens), lerobot's
    embed_language_tokens or the same gather from the export's embed_tokens table (smolvla_token_embedder)."""
    pad, att = smolvla_prefix_masks(geometry, img_masks, lang_masks)
    mask, position_ids, _ = smolvla_attention_inputs(pad, att)
    positions = _rope_table(position_ids, geometry.head_dim) if host_rope else (position_ids,)
    if embed is not None:
        with torch.no_grad():
            lang_tokens = embed(lang_tokens)
    return tuple(t.contiguous() for t in (*images, lang_tokens, state, mask, *positions)), pad


SMOLVLA_EMBED_TOKENS = "model.vlm_with_expert.vlm.model.text_model.embed_tokens.weight"


def smolvla_token_embedder(table):
    """The host side of embed_language_tokens from the export's table (SMOLVLA_EMBED_TOKENS, BF16 [vocab, hidden]):
    nn.Embedding's forward is this gather (padding_idx only affects gradients, max_norm is None)."""
    return lambda tokens: torch.nn.functional.embedding(tokens, table)


def smolvla_time_inputs(geometry, time, host_time_embedding=False):
    """() or (time embedding,): with host_time_embedding, lerobot's create_sinusoidal_pos_embedding(time, time_dim,
    min_period, max_period) cast to geometry.time_dtype (embed_suffix's .type(dtype=action_emb.dtype)). Changes with
    the time, i.e. at every denoising step."""
    if not host_time_embedding:
        return ()
    from lerobot.policies.common.vla_utils import create_sinusoidal_pos_embedding

    with torch.no_grad():
        emb = create_sinusoidal_pos_embedding(time, geometry.time_dim, geometry.min_period, geometry.max_period,
                                              device=time.device)
        return (emb.type(dtype=geometry.time_dtype).contiguous(),)


def smolvla_step_inputs(geometry, x_t, time, host, kv, host_time_embedding=False):
    """The SmolVLADenoiseStep input tuple: x_t, time, (the host time embedding), the per-chunk host inputs
    (smolvla_step_host_inputs) and the prefix K / V."""
    return (x_t, time, *smolvla_time_inputs(geometry, time, host_time_embedding), *host, *kv)


def smolvla_step_host_inputs(geometry, prefix_pad_masks, host_rope=False):
    """(attention_mask, *positions) of SmolVLADenoiseStep: the lines of denoise_step that build them, on the host. The
    suffix is chunk_size action tokens, all valid, each its own block (embed_suffix: pad mask ones, block mask 1s in the
    embedding dtype), so keep = [prefix pad mask broadcast over the queries | causal over the chunk] and
    position_ids = sum(prefix pad mask) + cumsum(suffix pad mask) - 1. With host_rope: the "suffix" table of those
    positions (self-attention layers) and the "expert" table of position_ids - row minimum (cross-attention layers)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    chunk = geometry.chunk_size
    batch_size, prefix_len = prefix_pad_masks.shape
    device = prefix_pad_masks.device
    suffix_pad_masks = torch.ones(batch_size, chunk, dtype=torch.bool, device=device)
    suffix_att_masks = torch.ones(batch_size, chunk, dtype=torch.float32, device=device)
    prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, chunk, prefix_len)
    suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
    keep = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
    prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
    position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
    if host_rope:
        expert = position_ids - torch.min(position_ids, dim=1, keepdim=True).values
        tables = {"suffix": _rope_table(position_ids, geometry.head_dim),
                  "expert": _rope_table(expert, geometry.head_dim)}
        positions = tuple(t for name in STEP_ROPE_TABLES for t in tables[name])
    else:
        positions = (position_ids,)
    return tuple(t.contiguous() for t in (additive_attention_mask(keep), *positions))


def smolvla_prefix_outputs(policy, n_images, prefix_cases, host_rope=False, host_text_embeddings=False):
    """The eager SmolVLAPrefix outputs (flat K / V tuples) for `prefix_cases`; the boundary is restored afterwards."""
    boundary = SmolVLAPrefix(policy, n_images, host_rope=host_rope, host_text_embeddings=host_text_embeddings).eval()
    try:
        with torch.no_grad():
            return [boundary(*case) for case in prefix_cases]
    finally:
        boundary.restore()


def smolvla_phase_cases(policy, n_images, observations, host_rope=False, host_text_embeddings=False,
                        host_time_embedding=False):
    """(prefix cases, step cases) for observations [(images, img_masks, lang_tokens, lang_masks, state, x_t, time)]:
    the prefix inputs, and the step inputs on the eager prefix K / V of the same observation with that x_t and time."""
    geometry = smolvla_geometry(policy)
    embed = policy.model.vlm_with_expert.embed_language_tokens if host_text_embeddings else None
    prefix_cases, pads = [], []
    for images, img_masks, lang_tokens, lang_masks, state, _, _ in observations:
        case, pad = smolvla_prefix_inputs(geometry, images, img_masks, lang_tokens, lang_masks, state, host_rope, embed)
        prefix_cases.append(case)
        pads.append(pad)
    kvs = smolvla_prefix_outputs(policy, n_images, prefix_cases, host_rope, host_text_embeddings)
    step_cases = [smolvla_step_inputs(geometry, obs[5].contiguous(), obs[6].contiguous(),
                                      smolvla_step_host_inputs(geometry, pad, host_rope),
                                      tuple(t.contiguous() for t in kv), host_time_embedding)
                  for obs, pad, kv in zip(observations, pads, kvs)]
    return prefix_cases, step_cases


def _smolvla_embedded(policy, n_images, case):
    """(prefix, suffix, pad_masks, att_masks) of one smolvla_raw_cases tuple: the policy's own embed_prefix /
    embed_suffix (x_t = t * noise + (1 - t) * actions, make_flow_matching_inputs with the noise at t = 1) and the
    concatenated pad / block masks, as VLAFlowMatching.forward has them."""
    from lerobot.policies.common.flow_matching import FlowConvention, make_flow_matching_inputs

    model = policy.model
    images = list(case[:n_images])
    lang_tokens, lang_masks, state, actions, noise, time = case[n_images:]
    masks = [torch.ones(image.shape[0], dtype=torch.bool, device=image.device) for image in images]
    with torch.no_grad():
        x_t, _, _ = make_flow_matching_inputs(actions, noise, time, convention=FlowConvention.NOISE_AT_ONE)
        prefix, prefix_pad, prefix_att = model.embed_prefix(images, masks, lang_tokens, lang_masks, state=state)
        suffix, suffix_pad, suffix_att = model.embed_suffix(x_t, time)
    if not bool(suffix_pad.all()):
        raise ValueError("padded suffix (action) tokens: the expert's mask slice would drop their diagonal")
    return (prefix, suffix, torch.cat([prefix_pad, suffix_pad], dim=1), torch.cat([prefix_att, suffix_att], dim=1))


def _position_inputs(policy, position_ids, prefix_len, host_rope):
    if not host_rope:
        return (position_ids,)
    tables = smolvla_rope_tables(position_ids, prefix_len, _rope_head_dim(policy))
    return tuple(t.contiguous() for name in ROPE_TABLES for t in tables[name])


def smolvla_flow_inputs(policy, n_images, case, keep_self_for_padded=False, host_rope=False,
                        host_text_embeddings=False, host_time_embedding=False):
    """SmolVLAFlowLoss inputs for one smolvla_raw_cases tuple (images, lang_tokens, lang_masks, state, actions, noise,
    time): the same tensors without lang_masks, then the host attention mask and position ids (with host_rope: the
    host RoPE tables of those position ids). With host_text_embeddings, lang_tokens is replaced by lerobot's
    embed_language_tokens(lang_tokens) on the host; with host_time_embedding, smolvla_time_embedding is appended."""
    prefix, _, pad, att = _smolvla_embedded(policy, n_images, case)
    mask, position_ids, _ = smolvla_attention_inputs(pad, att, keep_self_for_padded)
    images, (lang_tokens, _, state, actions, noise, time) = case[:n_images], case[n_images:]
    if host_text_embeddings:
        with torch.no_grad():
            lang_tokens = policy.model.vlm_with_expert.embed_language_tokens(lang_tokens).contiguous()
    tail = (smolvla_time_embedding(policy, actions, noise, time),) if host_time_embedding else ()
    return (*images, lang_tokens, state, actions, noise, time, mask,
            *_position_inputs(policy, position_ids, prefix.shape[1], host_rope), *tail)


def smolvla_text_cases(policy, n_images, seed, device, count=2, keep_self_for_padded=False, host_rope=False):
    """Inputs of SmolVLATextLayers for the smolvla_raw_cases inputs: the policy's own embed_prefix / embed_suffix on
    them and the host attention mask and position ids (smolvla_attention_inputs)."""
    cases = []
    for case in smolvla_raw_cases(policy, n_images, seed, device, count):
        prefix, suffix, pad, att = _smolvla_embedded(policy, n_images, case)
        mask, position_ids, _ = smolvla_attention_inputs(pad, att, keep_self_for_padded)
        cases.append(tuple(v.contiguous() for v in (prefix, suffix, mask,
                                                     *_position_inputs(policy, position_ids, prefix.shape[1], host_rope))))
    return cases


def smolvla_raw_cases(policy, n_images, seed, device, count=2, image_size=None):
    """`count` fixed policy-level input tuples (images in [-1, 1], as prepare_images gives; lang_tokens, an all-true
    lang_masks, state, actions, noise, time): the inputs VLAFlowMatching.forward takes."""
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


def smolvla_cases(policy, n_images, seed, device, count=2, image_size=None, keep_self_for_padded=False,
                  host_rope=False, host_text_embeddings=False, host_time_embedding=False):
    """`count` fixed input tuples for SmolVLAFlowLoss: smolvla_raw_cases through smolvla_flow_inputs."""
    return [smolvla_flow_inputs(policy, n_images, case, keep_self_for_padded, host_rope, host_text_embeddings,
                                host_time_embedding)
            for case in smolvla_raw_cases(policy, n_images, seed, device, count, image_size)]


# the capture boundaries each host-input flag applies to; main() refuses a flag with any other boundary
HOST_FLAG_BOUNDARIES = {
    "host_rope": ("smolvla_flow_loss", "smolvla_text_layers", "smolvla_prefix", "smolvla_denoise_step"),
    "host_text_embeddings": ("smolvla_flow_loss", "smolvla_prefix"),
    "host_time_embedding": ("smolvla_flow_loss", "smolvla_denoise_step"),
}


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
    parser.add_argument("--boundary", choices=("causal_lm", "causal_lm_layer", "causal_lm_prefill", "causal_lm_decode_step",
                                               "causal_lm_decode_layer", "smolvla_flow_loss", "smolvla_text_layers",
                                               "smolvla_prefix", "smolvla_denoise_step", "dronet"),
                        default="causal_lm",
                        help="capture boundary: causal-LM logits (default), one decoder layer of a causal LM "
                             "(CausalLayer: hidden, cos, sin, mask -> hidden; see --layer), the causal-LM token-loop phases "
                             "causal_lm_prefill (CausalPrefill: prompt of --seqlen tokens, mask, cos, sin, zero --capacity "
                             "K / V -> logits, K / V), causal_lm_decode_step (CausalDecodeStep: token, mask, cos, sin, slot, "
                             "--capacity K / V -> logits, new K / V) and causal_lm_decode_layer (one layer of the step, "
                             "--layer); or the SmolVLA flow-matching loss forward, or its first --layers text / expert layers "
                             "(SmolVLATextLayers: prefix, suffix embeddings, attention mask, position ids -> suffix output), "
                             "or one of the two inference phases: smolvla_prefix (SmolVLAPrefix: images, tokens, state -> "
                             "per-layer K / V) and smolvla_denoise_step (SmolVLADenoiseStep: x_t, time, prefix K / V -> v_t); "
                             "every SmolVLA boundary takes the host-computed attention mask and position ids; or DroNet "
                             "(frame -> steering, collision logit; needs --frames)")
    parser.add_argument("--host-rope", action="store_true",
                        help="SmolVLA boundaries: take the host RoPE cos / sin tables (smolvla_rope_tables) instead of "
                             "the position ids, so the graph computes no RoPE angle")
    parser.add_argument("--host-text-embeddings", action="store_true",
                        help="--boundary smolvla_flow_loss / smolvla_prefix: take lerobot's embed_language_tokens(lang_tokens), "
                             "computed on the host, instead of the token ids, so the graph holds no token embedding lookup")
    parser.add_argument("--host-time-embedding", action="store_true",
                        help="--boundary smolvla_flow_loss / smolvla_denoise_step: take the flow-timestep embedding (lerobot's "
                             "create_sinusoidal_pos_embedding, cast as embed_suffix casts it) computed on the host as a "
                             "last input, so the graph holds none of its float64 math")
    parser.add_argument("--layers", type=int, default=2, help="text / expert layers with --boundary smolvla_text_layers")
    parser.add_argument("--layer", type=int, default=0,
                        help="decoder layer index with --boundary causal_lm_layer / causal_lm_decode_layer")
    parser.add_argument("--capacity", type=int, default=128,
                        help="KV-cache capacity C (static; the StaticCache max_cache_len) of the causal_lm_prefill / "
                             "causal_lm_decode_step / causal_lm_decode_layer boundaries")
    parser.add_argument("--host-embeddings", action="store_true",
                        help="causal_lm_prefill / causal_lm_decode_step: the first input is the token embeddings, "
                             "gathered on the host from the export's own table (causal_token_embedder), not the ids")
    parser.add_argument("--all-logits", action="store_true",
                        help="causal_lm_prefill: logits of every prompt position, not only the last")
    parser.add_argument("--n-images", type=int, default=1, help="cameras at the smolvla_flow_loss / smolvla_prefix boundary")
    parser.add_argument("--frames", help="npz with a uint8 `frames` array (N, 200, 200): the two capture cases of "
                                         "--boundary dronet are two distinct frames drawn from it with --seed")
    parser.add_argument("--verify-dir", help="fresh-process replay of an existing FX bundle")
    args = parser.parse_args(argv)
    for flag, boundaries in HOST_FLAG_BOUNDARIES.items():
        if getattr(args, flag) and args.boundary not in boundaries:
            parser.error(f"--{flag.replace('_', '-')} applies to --boundary {' / '.join(boundaries)}, "
                         f"not {args.boundary}")
    if args.verify_dir:
        report = verify_saved(args.verify_dir, args.device)
        _json(Path(args.verify_dir) / "fresh_process_verification.json", report)
    else:
        if not args.export_dir or not args.out_dir or args.seqlen < 2:
            parser.error("--export-dir, --out-dir and --seqlen >= 2 are required")
        import exporter.load_quantized as load_quantized
        import exporter.quant_export as quant_export

        _, policy, manifest = load_quantized.read_export(args.export_dir)
        kind = (manifest.get("model") or {}).get("kind", "causal_lm")
        if (kind == "causal_lm") != args.boundary.startswith("causal_lm"):
            parser.error(f"export kind {kind!r} does not match --boundary {args.boundary}; the CLI captures causal-LM "
                         "boundaries for a causal-LM export and SmolVLA boundaries for a policy export")
        causal_flags = [f for f, on in (("--host-embeddings", args.host_embeddings), ("--all-logits", args.all_logits)) if on]
        if args.boundary.startswith("smolvla_") and causal_flags:
            parser.error(f"{', '.join(causal_flags)} apply to the causal-LM phase boundaries, not --boundary {args.boundary}")
        if args.boundary == "dronet":
            if kind != "dronet":
                parser.error(f"--boundary dronet needs a DroNet export (found kind {kind!r})")
            if not args.frames:
                parser.error("--boundary dronet needs --frames (real frames: the capture cases must exercise real "
                             "activation ranges)")
            import numpy as np
            from dronet import model as dronet_model

            boundary, _, _ = load_quantized.load_quantized(
                None, args.export_dir, args.device, mode="ref_conv",
                build=lambda: (dronet_model.FusedHeadDroNet() if (manifest.get("model") or {}).get("fused_heads")
                               else dronet_model.DroNet()).to(torch.bfloat16).to(args.device))
            boundary.eval()
            images = np.load(args.frames)["frames"]
            picks = np.random.default_rng(args.seed).choice(len(images), 2, replace=False).tolist()
            cases = [(torch.from_numpy(images[i]).float().div(255)[None, None].to(torch.bfloat16).to(args.device),)
                     for i in picks]
            config_dict = {"architecture": "dronet_resnet8", "input": [1, 1, 200, 200], "frame_indices": picks}
            frames_record = {"frames": str(Path(args.frames).resolve()), "frames_sha256": _file_record(args.frames)["sha256"],
                             "frame_indices": picks}
        elif args.boundary.startswith("smolvla_"):
            import eval_policy

            model_id = args.model or manifest["model"]["id"]
            lerobot_policy, _ = eval_policy.load_policy(model_id, args.device, torch.bfloat16)
            load_quantized.load_quantized(model_id, args.export_dir, args.device, build=lambda: lerobot_policy,
                                          mode="ref_conv", options={})
            if args.boundary == "smolvla_text_layers":
                cases = smolvla_text_cases(lerobot_policy, args.n_images, args.seed, args.device, host_rope=args.host_rope)
                boundary = SmolVLATextLayers(lerobot_policy, args.layers, host_rope=args.host_rope).eval()
            elif args.boundary in ("smolvla_prefix", "smolvla_denoise_step"):
                # fixed random observations (smolvla_raw_cases: all-true language masks); the step case takes x_t = the
                # case's noise and its time. Real-frame cases: src/scripts/tile_flow/smolvla_phase_capture.py
                observations = [(case[:args.n_images],
                                 [torch.ones(1, dtype=torch.bool, device=args.device)] * args.n_images,
                                 *case[args.n_images:args.n_images + 3], case[args.n_images + 4], case[args.n_images + 5])
                                for case in smolvla_raw_cases(lerobot_policy, args.n_images, args.seed, args.device)]
                prefix_cases, step_cases = smolvla_phase_cases(lerobot_policy, args.n_images, observations,
                                                               host_rope=args.host_rope,
                                                               host_text_embeddings=args.host_text_embeddings,
                                                               host_time_embedding=args.host_time_embedding)
                if args.boundary == "smolvla_prefix":
                    cases, boundary = prefix_cases, SmolVLAPrefix(lerobot_policy, args.n_images, args.host_rope,
                                                                  args.host_text_embeddings).eval()
                else:
                    cases, boundary = step_cases, SmolVLADenoiseStep(lerobot_policy, args.host_rope,
                                                                     args.host_time_embedding).eval()
            else:
                # the host token embeddings are computed before the boundary replaces embed_language_tokens
                cases = smolvla_cases(lerobot_policy, args.n_images, args.seed, args.device, host_rope=args.host_rope,
                                      host_text_embeddings=args.host_text_embeddings,
                                      host_time_embedding=args.host_time_embedding)
                boundary = SmolVLAFlowLoss(lerobot_policy, args.n_images, host_rope=args.host_rope,
                                           host_text_embeddings=args.host_text_embeddings,
                                           host_time_embedding=args.host_time_embedding).eval()
            config_dict = __import__("dataclasses").asdict(lerobot_policy.config)
        else:
            model, _, _ = load_quantized.load_quantized(args.model, args.export_dir, args.device, mode="ref_conv",
                                                       seqlen=max(args.seqlen, args.capacity)
                                                       if args.boundary.startswith(("causal_lm_decode", "causal_lm_prefill"))
                                                       else args.seqlen)
            config_dict = model.config.to_dict()
            if args.boundary in ("causal_lm_layer", "causal_lm_decode_layer"):
                if not 0 <= args.layer < model.config.num_hidden_layers:
                    parser.error(f"--layer {args.layer} outside 0..{model.config.num_hidden_layers - 1}")
            phase = args.boundary in ("causal_lm_prefill", "causal_lm_decode_step", "causal_lm_decode_layer")
            if (args.host_embeddings or args.all_logits) and not phase:
                parser.error("--host-embeddings / --all-logits apply to the causal_lm_prefill / decode boundaries")
            if args.boundary == "causal_lm_layer":
                cases = causal_layer_cases(model, args.layer, args.seqlen, args.seed, args.device)
                boundary = CausalLayer(model, args.layer).eval()
            elif phase:
                geometry = causal_geometry(model, args.capacity, args.seqlen)
                embed = None
                if args.host_embeddings:
                    tensors, _, _ = load_quantized.read_export(args.export_dir, verify=False)
                    embed = causal_token_embedder(tensors, policy)
                if args.boundary == "causal_lm_prefill":
                    boundary = CausalPrefill(model, args.host_embeddings, args.all_logits).eval()
                elif args.boundary == "causal_lm_decode_step":
                    boundary = CausalDecodeStep(model, args.host_embeddings).eval()
                else:
                    boundary = CausalDecodeLayer(model, args.layer).eval()
                cases = causal_phase_cases(model, boundary, geometry, args.seed, args.device,
                                           host_embeddings=args.host_embeddings, embed=embed)
            else:
                vocab = model.config.vocab_size
                cases = [(torch.randint(vocab, (1, args.seqlen), generator=torch.Generator().manual_seed(args.seed + i))
                          .to(args.device),) for i in range(2)]
                boundary = CausalLogits(model).eval()
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
                                        "seed": args.seed, "seqlen": args.seqlen,
                                        **({"layer": args.layer} if args.boundary in ("causal_lm_layer",
                                                                                      "causal_lm_decode_layer") else {}),
                                        **({"capacity": args.capacity, "prefill_length": args.seqlen,
                                            "host_embeddings": args.host_embeddings, "all_logits": args.all_logits}
                                           if args.boundary.startswith(("causal_lm_prefill", "causal_lm_decode"))
                                           else {}),
                                        **({"layers": args.layers} if args.boundary == "smolvla_text_layers" else {}),
                                        **({"n_images": args.n_images} if args.boundary in ("smolvla_flow_loss",
                                                                                            "smolvla_prefix") else {}),
                                        **({"host_rope": True} if args.host_rope else {}),
                                        **({"host_text_embeddings": True} if args.host_text_embeddings else {}),
                                        **({"host_time_embedding": True} if args.host_time_embedding else {}),
                                        **(frames_record if args.boundary == "dronet" else {})})
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
