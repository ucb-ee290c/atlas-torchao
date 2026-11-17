"""Stored bytes of an export (or any safetensors file), split into embedding, LM head and the rest (NEW).

    python src/scripts/export_sizes.py EXPORT_DIR_OR_SAFETENSORS [...] [--budget 1073741824] [--json out.json]
        [--kv-bytes-per-token 22528]

Read from the safetensors header only (no tensor is loaded): every tensor's byte count is its data_offsets span.
Groups (by tensor name, after an optional wrapper prefix such as `model.` in an FX capture's weights file):
  embedding     model.embed_tokens.*   (BF16 weight, or codes and scales of a quantized embedding)
  lm_head       lm_head.*              (BF16 weight, FP8 codes / exponents / static input exponent, or NVFP4)
  body_u8_codes every other uint8 tensor (the decoder layers' E4M3 weight codes)
  body_other    every other tensor (norm weights, exponents, static activation exponents, ...)
A tied BF16 pair is stored once (quant_export writes the alias name in policy.json, not in the file), so its bytes
are in whichever group holds the stored name. `room_bytes` = budget - tensor data bytes; `file_bytes` adds the
safetensors header. With --kv-bytes-per-token, `room_tokens` = room_bytes // that (the KV cache the room would hold
if nothing else used it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 1 << 30
GROUPS = ("embedding", "lm_head", "body_u8_codes", "body_other")


def header(path):
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(n)), n


def group_of(name, dtype):
    for prefix in ("", "model."):            # an FX capture may hold the export names behind `model.`
        if name.startswith(prefix + "model.embed_tokens."):
            return "embedding"
        if name.startswith(prefix + "lm_head."):
            return "lm_head"
    return "body_u8_codes" if dtype == "U8" else "body_other"


def sizes(path, budget=GIB, kv_bytes_per_token=None) -> dict:
    p = Path(path)
    f = p / "model.safetensors" if p.is_dir() else p
    h, hbytes = header(f)
    out = {g: 0 for g in GROUPS}
    tensors = {g: {} for g in GROUPS}
    for name, rec in h.items():
        if name == "__metadata__":
            continue
        b = rec["data_offsets"][1] - rec["data_offsets"][0]
        g = group_of(name, rec["dtype"])
        out[g] += b
        if g in ("embedding", "lm_head"):
            tensors[g][name] = {"dtype": rec["dtype"], "shape": rec["shape"], "bytes": b}
    data = sum(out.values())
    res = {"file": str(f), "file_bytes": f.stat().st_size, "header_bytes": 8 + hbytes, "tensor_bytes": data,
           "tensor_count": len(h) - ("__metadata__" in h), **{f"{g}_bytes": out[g] for g in GROUPS},
           "body_bytes": out["body_u8_codes"] + out["body_other"],
           "budget_bytes": budget, "room_bytes": budget - data, "fits": data <= budget,
           "percent_of_budget": round(100.0 * data / budget, 3), "embedding_tensors": tensors["embedding"],
           "lm_head_tensors": tensors["lm_head"]}
    if kv_bytes_per_token:
        res["kv_bytes_per_token"] = kv_bytes_per_token
        res["room_tokens"] = max(res["room_bytes"], 0) // kv_bytes_per_token
    pol = (p / "policy.json") if p.is_dir() else None
    if pol is not None and pol.is_file():
        policy = json.loads(pol.read_text())
        res["aliases"] = policy.get("aliases", {})
        res["head_embed"] = (policy.get("head_embed") or {}).get("options")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--budget", type=int, default=GIB)
    ap.add_argument("--kv-bytes-per-token", type=int, default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    rows = [sizes(p, a.budget, a.kv_bytes_per_token) for p in a.paths]
    for r in rows:
        print(f"{r['file']}: tensors {r['tensor_bytes']:,} B (embedding {r['embedding_bytes']:,}, lm_head "
              f"{r['lm_head_bytes']:,}, body {r['body_bytes']:,}), file {r['file_bytes']:,} B, room "
              f"{r['room_bytes']:,} B {'fits' if r['fits'] else 'DOES NOT FIT'}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
