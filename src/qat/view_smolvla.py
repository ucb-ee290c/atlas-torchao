#!/usr/bin/env python3
"""Quick viewer for a SmolVLA checkpoint.

Usage:
  python view_smolvla.py                          # weights summary (no lerobot needed)
  python view_smolvla.py --keys                   # also list every tensor name + shape
  python view_smolvla.py --grep self_attn         # only keys containing a substring
  python view_smolvla.py --depth 5                # group params by deeper key prefix
  python view_smolvla.py --module                 # load with lerobot and print the nn.Module tree
"""
import argparse
import glob
import json
import os
from collections import defaultdict

from safetensors import safe_open


def human(n):
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def view_weights(path, show_keys, grep, depth):
    cfg = os.path.join(path, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            c = json.load(f)
        print("== config.json (top-level) ==")
        for k, v in c.items():
            if not isinstance(v, (dict, list)):
                print(f"  {k}: {v}")
        print()

    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise SystemExit(f"No .safetensors files in {path}")

    total = 0
    groups = defaultdict(int)
    dtypes = defaultdict(int)
    rows = []
    for fn in files:
        with safe_open(fn, framework="pt") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                shape = sl.get_shape()
                n = 1
                for s in shape:
                    n *= s
                total += n
                groups[".".join(k.split(".")[:depth])] += n
                dtypes[sl.get_dtype()] += n
                if (show_keys or grep) and (grep is None or grep in k):
                    rows.append((k, shape, sl.get_dtype()))

    print(f"== {len(files)} file(s), {human(total)} params ==")
    print("dtypes:", {d: human(n) for d, n in dtypes.items()})
    print(f"\n== params by prefix (depth={depth}) ==")
    for g, n in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {human(n):>8}  {100 * n / total:5.1f}%  {g}")

    if rows:
        print(f"\n== tensors ({len(rows)}) ==")
        for k, shape, dt in rows:
            print(f"  {k:<90} {str(list(shape)):<20} {dt}")


def view_module(path):
    try:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # older lerobot

    policy = SmolVLAPolicy.from_pretrained(path)
    print(policy)
    print("\n== params per top-level child ==")
    for name, child in policy.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"  {human(n):>8}  {name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="/rscratch/agarcia/smolvla_base")
    ap.add_argument("--keys", action="store_true", help="list every tensor")
    ap.add_argument("--grep", default=None, help="only list keys containing this")
    ap.add_argument("--depth", type=int, default=4, help="key-prefix depth for grouping")
    ap.add_argument("--module", action="store_true", help="load via lerobot and print module tree")
    a = ap.parse_args()

    if a.module:
        view_module(a.path)
    else:
        view_weights(a.path, a.keys, a.grep, a.depth)