"""Write a hardware-verification sidecar for an atlas_v1 export.

    python scripts/hw_provenance.py --export DIR --rtl DIR --npu-model DIR \\
        --results results/hw-verify/l0.json results/hw-verify/l1.json [--out DIR/hw_verification.json]

The sidecar ties "verified against the RTL" to specific things: the RTL and npu-model commits, the
atlas-torchao commit, the sha256 and counts of each verification result file, and the sha256 of the
export it was written next to. It is a separate file, not a manifest.json field, because the manifest
is verified downstream (manifest checks, adapter contracts) and hashed by the export round-trip tests.
`scope` states what the results cover, so the record cannot be read as more than it is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SCOPE = ("Arithmetic datapaths (inner-product tree, systolic-array multiply-add, FP8 converters) and the "
         "listed vector functions, compared bit for bit. Not a full-SoC or cycle-accurate validation; "
         "see the per-file `kind`.")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo) -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def summarize_result(path) -> dict:
    """{kind, sha256, comparisons, mismatches} of one verification result JSON (L0, L1 or L1b shape)."""
    data = json.loads(Path(path).read_text())
    if "total_cases" in data:                                   # scripts/npu_model_crosscheck.py (L0)
        kind, n, bad = "differential vs npu-model RTL-fitted numerics (L0)", data["total_cases"], data["total_mismatches"]
    elif "total_comparisons" in data:                           # scripts/rtl/compare_l1.py (L1) and l1b
        kind, n, bad = "atlas-torchao stimuli through the RTL under Verilator", data["total_comparisons"], data["total_mismatches"]
    else:
        raise ValueError(f"{path}: not a recognized verification result")
    return {"file": Path(path).name, "kind": kind, "sha256": sha256_file(path), "comparisons": n, "mismatches": bad}


def build_record(export_dir, results, rtl=None, npu_model=None, torchao=None, commits=None, now=None) -> dict:
    """The sidecar dict. `commits` ({name: sha}) overrides the git lookups (tests, offline use)."""
    export_dir = Path(export_dir)
    commits = commits or {}
    get = lambda name, repo: commits.get(name) or (git_commit(repo) if repo else None)  # noqa: E731
    return {
        "schema": 1,
        "written": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "scope": SCOPE,
        "export": {"model_safetensors_sha256": sha256_file(export_dir / "model.safetensors"),
                   "manifest_sha256": sha256_file(export_dir / "manifest.json")},
        "commits": {"rtl": get("rtl", rtl), "npu_model": get("npu_model", npu_model),
                    "atlas_torchao": get("atlas_torchao", torchao)},
        "results": [summarize_result(r) for r in results],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--export", required=True)
    ap.add_argument("--rtl", default="/scratch2/lorenhung/atlas-npu-rtl/src")
    ap.add_argument("--npu-model", default="/scratch2/lorenhung/atlas-npu-rtl/src/npu-model")
    ap.add_argument("--torchao", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rec = build_record(args.export, args.results, args.rtl, args.npu_model, args.torchao)
    out = Path(args.out or Path(args.export) / "hw_verification.json")
    out.write_text(json.dumps(rec, indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
