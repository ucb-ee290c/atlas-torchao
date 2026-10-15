"""Repair checkpoint keys that a newer library version nests one module level deeper (NEW).

A checkpoint saved against one version of a wrapped library (e.g. transformers' CLIP) can name its
tensors `<prefix>.<rest>` while the current model expects `<prefix>.<extra>.<rest>`. A non-strict
load then reports the same keys as missing and unexpected and silently keeps the constructor's
weights for them. `repair_checkpoint_keys` detects that case after the normal load and fixes it:

  - an unexpected key u pairs with a missing key m when m is u with exactly one dotted segment
    inserted; a rule (from_prefix -> to_prefix) must map EVERY unexpected key under from_prefix
    onto a missing key, so the inserted segment is the same for all keys sharing that prefix;
  - the pairing must be a bijection covering all missing and all unexpected keys, with identical
    shapes; then the full state dict (remapped keys included) is loaded with strict=True;
  - anything else raises with the unmatched keys. It never continues with missing keys.

Tied (storage-sharing) parameters that the checkpoint stores under one name only are not missing.
Nothing here knows a model name: the rules are derived from the two key sets alone.
"""

import os
from collections import defaultdict

import torch


class CheckpointKeyError(RuntimeError):
    pass


def _alias_groups(state_dict):
    """Model keys grouped by the tensor storage they share (tied weights)."""
    groups = defaultdict(list)
    for k, v in state_dict.items():
        if v.device.type == "meta" or v.numel() == 0:
            groups[("own", k)].append(k)
            continue
        groups[(v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape), tuple(v.stride()))].append(k)
    return list(groups.values())


def _inserted(u, m):
    """[(position, segment)] such that inserting segment at position of u's segments gives m."""
    us, ms = u.split("."), m.split(".")
    if len(ms) != len(us) + 1:
        return []
    return [(i, ms[i]) for i in range(len(ms)) if ms[:i] + ms[i + 1:] == us]


def derive_rules(missing, unexpected):
    """Insertion rules mapping every unexpected key onto a distinct missing key, or raise.

    Returns (mapping {unexpected: missing}, rules [{from_prefix, to_prefix}]).
    """
    missing, unexpected = set(missing), set(unexpected)
    by_len = defaultdict(set)
    for m in missing:
        by_len[m.count(".")].add(m)
    candidates = set()
    for u in unexpected:
        for m in by_len[u.count(".") + 1]:
            for i, seg in _inserted(u, m):
                candidates.add((".".join(u.split(".")[:i]), seg))

    def apply(rule, u):
        pre, seg = rule
        if pre and not u.startswith(pre + "."):
            return None
        rest = u[len(pre) + 1:] if pre else u
        return ".".join(p for p in (pre, seg, rest) if p)

    # a rule holds only if it maps every unexpected key under its prefix onto a missing key
    rules = []
    for rule in candidates:
        under = [u for u in unexpected if apply(rule, u) is not None]
        if under and all(apply(rule, u) in missing for u in under):
            rules.append((rule, len(under)))
    mapping, used = {}, {}
    unmatched_u = []
    for u in sorted(unexpected):
        targets = {apply(r, u) for r, _ in rules if apply(r, u) is not None}
        if len(targets) != 1:
            unmatched_u.append(u)
            continue
        mapping[u] = targets.pop()
        # report the rule covering the most keys (ties: the shortest prefix) for this key
        best = min((r for r, _ in rules if apply(r, u) == mapping[u]),
                   key=lambda r: (-dict(rules)[r], len(r[0]), r))
        used[best] = used.get(best, 0) + 1
    hit = defaultdict(list)
    for u, m in mapping.items():
        hit[m].append(u)
    dup = {m: us for m, us in hit.items() if len(us) > 1}
    unmatched_m = sorted(missing - set(hit))
    if unmatched_u or unmatched_m or dup:
        raise CheckpointKeyError(
            "checkpoint keys do not match the model and no single-segment insertion repairs them: "
            f"{len(unmatched_m)} missing unmatched {unmatched_m[:20]}, "
            f"{len(unmatched_u)} unexpected unmatched {unmatched_u[:20]}, "
            f"{len(dup)} missing keys claimed twice {dict(list(dup.items())[:5])}")
    out_rules = [{"from_prefix": pre, "to_prefix": ".".join(p for p in (pre, seg) if p), "n_keys": n}
                 for (pre, seg), n in sorted(used.items())]
    return mapping, out_rules


def key_diff(model_sd, ckpt_keys):
    """(missing, unexpected) of a checkpoint against a model state dict, tied aliases excused."""
    ckpt_keys = set(ckpt_keys)
    missing = set()
    for group in _alias_groups(model_sd):
        if not any(k in ckpt_keys for k in group):
            missing.update(group)
    model_keys = set(model_sd)
    unexpected = ckpt_keys - model_keys
    return missing, unexpected


def repair_state_dict(model, ckpt):
    """Load `ckpt` (a {key: tensor} dict) into `model`, repairing nesting mismatches. Returns the record."""
    model_sd = model.state_dict()
    missing, unexpected = key_diff(model_sd, ckpt.keys())
    if not missing and not unexpected:
        return {"n_remapped": 0, "rules": []}
    mapping, rules = derive_rules(missing, unexpected)
    bad = {u: (tuple(ckpt[u].shape), tuple(model_sd[m].shape)) for u, m in mapping.items()
           if tuple(ckpt[u].shape) != tuple(model_sd[m].shape)}
    if bad:
        raise CheckpointKeyError(f"remapped keys differ in shape (checkpoint, model): {dict(list(bad.items())[:20])}")
    renamed = {mapping.get(k, k): v for k, v in ckpt.items()}
    full = {}
    for group in _alias_groups(model_sd):
        src = next(k for k in group if k in renamed)
        for k in group:
            full[k] = renamed[src]
    model.load_state_dict(full, strict=True)
    return {"n_remapped": len(mapping), "rules": rules}


def resolve_checkpoint_file(path, filename="model.safetensors"):
    """The checkpoint's weights file: <dir>/<filename> for a local dir, else from the HF cache/hub."""
    if os.path.isdir(path):
        return os.path.join(path, filename)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=str(path), filename=filename)


def repair_checkpoint_keys(model, path, filename="model.safetensors"):
    """After `from_pretrained(path)`: re-read the checkpoint and repair key nesting (see module doc)."""
    from safetensors import safe_open
    from safetensors.torch import load_file

    file = resolve_checkpoint_file(path, filename)
    with safe_open(file, framework="pt") as f:
        keys = list(f.keys())
    missing, unexpected = key_diff(model.state_dict(), keys)
    if not missing and not unexpected:
        return {"n_remapped": 0, "rules": [], "new": True}
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    rec = repair_state_dict(model, load_file(file, device=str(device)))
    return {**rec, "new": True}
