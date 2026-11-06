"""Rank a causal LM's Linear layers by FP8 sensitivity on the calibration split (offline input of
--algo mixed_precision:rank_file=...; docs/algos/mixed-precision.md).

    python src/scripts/rank_layers.py --model /path/TinyLlama-1.1B-Chat-v1.0 --seqlen 512 \
        --calib-n 4 --calib-seed 1234 --device cuda --out out/rank_tinyllama.json

Calibration windows come from algos.calib.wikitext2_train (WikiText-2 TRAIN; the evaluation reads
the test split). Scores (algos.mixed_precision.score_layers) are computed against the profile
--profile (default ref_linear: the reference cast with framework accumulation). The JSON carries
the scores per layer and its provenance: model, profile, calibration record, code commit, versions.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _git(*a):
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *a], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rank(clean, batches, calib_rec, profile="ref_linear", skip=("lm_head",), iso=True, model_id=""):
    """The ranking dict for an unquantized causal LM `clean` and calibration `batches` ([1, T] ids)."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    import quant_modes
    from algos.mixed_precision import score_layers

    quant = copy.deepcopy(clean)
    quant_modes.quantize(quant, profile, list(skip), {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"})
    names = [n for n, m in clean.named_modules()
             if type(m) is nn.Linear and not any(s in n for s in skip)]

    def loss_fn(model, ids, no_grad=False):
        ids = ids.to(next(model.parameters()).device)   # batches stay on the CPU; the model may not
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            logits = model(ids, use_cache=False).logits[0, :-1].float()
            return F.cross_entropy(logits, ids[0, 1:], reduction="sum")

    layers = score_layers(clean, quant, batches, loss_fn, names, iso=iso)
    meta = {"model": model_id, "profile": profile, "skip": list(skip), "calibration": calib_rec,
            "scores": ["fisher_out", "hawq_w", "iso_rel_rmse"] if iso else ["fisher_out", "hawq_w"],
            "macs": "in_features x out_features per token",
            "device": str(next(clean.parameters()).device),
            "commit": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain", "src", "scripts")),
            "torch": torch.__version__, "host": platform.node()}
    if iso:
        import layer_rmse
        meta["reference_provenance"] = layer_rmse.ref_provenance()
    return {"meta": meta, "layers": layers}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--calib-n", type=int, default=4)
    ap.add_argument("--calib-seed", type=int, default=1234)
    ap.add_argument("--profile", default="ref_linear")
    ap.add_argument("--no-iso", action="store_true", help="skip the rel-RMSE pass (needs the reference repos)")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    from algos import calib
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    from loaders.modelutils import load_llama

    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    batches, rec = calib.wikitext2_train(a.model, a.calib_n, a.seqlen, a.calib_seed)
    clean = load_llama(a.model, seqlen=a.seqlen, device=device)
    out = rank(clean, batches, rec, a.profile, iso=not a.no_iso, model_id=a.model)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    top = sorted(out["layers"], key=lambda n: -out["layers"][n]["fisher_out"])[:5]
    print("top fisher_out:", top)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
