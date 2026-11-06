"""Compare the outputs of two captured FX bundles (model.pt2) on the same inputs: the number of differing logits (bit pattern),
the maximum absolute difference, and whether the outputs are byte-identical. Used to show that a pack-fold bundle
(src/pack_fold.py, docs/algos/pack-fold.md) equals its unfolded twin.
usage: compare_fx_logits.py FX_A FX_B OUT_JSON [--random N] [--seed S]     (run with PYTHONPATH=<worktree>/src)
Inputs: the capture-case inputs stored in FX_A/fixtures.safetensors (case_*.input_*) plus N random token-id sequences of the
same shape drawn below the vocabulary size seen in those inputs' model (max token id of the fixtures is not a bound, so
the vocabulary is read from the output's last dimension)."""
import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

import atlas_fx_ops  # noqa: F401  registers atlas.* ops


def _cmp(a: torch.Tensor, b: torch.Tensor) -> dict:
    a32, b32 = a.float(), b.float()
    same = a.contiguous().view(torch.uint8).reshape(-1).eq(b.contiguous().view(torch.uint8).reshape(-1)) if a.shape == b.shape else None
    differing = int((a32 != b32).sum()) if a.shape == b.shape else None
    diff = (a32 - b32).abs()
    return {"shape": list(a.shape), "dtype": str(a.dtype), "elements": a.numel(), "differing_logits": differing,
            "max_abs_difference": float(diff.max()) if a.shape == b.shape else None,
            "byte_identical": bool(same.all()) if same is not None else False,
            "max_abs_logit": float(a32.abs().max()), "nan_a": int(a32.isnan().sum()), "nan_b": int(b32.isnan().sum())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("fx_a")
    ap.add_argument("fx_b")
    ap.add_argument("out")
    ap.add_argument("--random", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    fa, fb = Path(args.fx_a), Path(args.fx_b)
    fixtures = load_file(str(fa / "fixtures.safetensors"))
    cases = sorted({k.split(".")[0] for k in fixtures if k.endswith(".input_0")})
    inputs = {c: fixtures[f"{c}.input_0"] for c in cases}
    prog_a = torch.export.load(str(fa / "model.pt2")).module()
    prog_b = torch.export.load(str(fb / "model.pt2")).module()
    res = {"fx_a": str(fa), "fx_b": str(fb), "cases": {}}
    with torch.no_grad():
        first = prog_a(inputs[cases[0]])
        vocab = int(first.shape[-1])
        g = torch.Generator().manual_seed(args.seed)
        shape = inputs[cases[0]].shape
        for i in range(args.random):
            inputs[f"random_{i}"] = torch.randint(0, vocab, tuple(shape), generator=g, dtype=inputs[cases[0]].dtype)
        for name, ids in inputs.items():
            res["cases"][name] = _cmp(prog_a(ids), prog_b(ids))
            print(name, res["cases"][name]["differing_logits"], res["cases"][name]["max_abs_difference"], flush=True)
    c = res["cases"].values()
    res["total_logits"] = sum(x["elements"] for x in c)
    res["total_differing_logits"] = sum(x["differing_logits"] or 0 for x in c)
    res["max_abs_difference"] = max(x["max_abs_difference"] for x in c)
    res["all_byte_identical"] = all(x["byte_identical"] for x in c)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))
    return 0 if res["all_byte_identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
