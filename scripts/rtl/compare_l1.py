"""Compare the RTL results of AtlasTorchaoStimTest with atlas-torchao's models, bit for bit.

    python scripts/rtl/compare_l1.py --stim stim.json --rtl rtl_out.json [--out result.json]
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import accum_torch as at, ref_numerics as rn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", required=True); ap.add_argument("--rtl", required=True); ap.add_argument("--out")
    args = ap.parse_args()
    cases = json.loads(Path(args.stim).read_text())["cases"]
    rtl = json.loads(Path(args.rtl).read_text())
    assert len(cases) == len(rtl), (len(cases), len(rtl))
    lut = at.e4m3_mul_to_prod_lut(); h, iw = at._ipt_consts(32, "numba")
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0])); examples = defaultdict(list)
    for c, r in zip(cases, rtl):
        a = torch.tensor(c["a"], dtype=torch.int32); w = torch.tensor(c["w"], dtype=torch.int32)
        partial = torch.tensor([c["partial"]], dtype=torch.int64)
        ours = {"ipt": int(at._ipt_tile(a[None, None], w[None, None], lut, partial[None, None], 2, h, iw, "numba", at.OUT_BF16, 0).flatten()[0]),
                "fma": int(at._sa_step(partial.to(torch.int32)[None, None, :], a[:1][None, :], w[:1][None, :], lut).flatten()[0]),
                "quant": int(at._bf16_scale_to_e4m3(partial.to(torch.int32), c["scale"] - 127).flatten()[0]),
                "dequant": int((rn.e4m3_unpack(a[:1].to(torch.uint8), 0).to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int32) & 0xFFFF).flatten()[0])}
        for k, v in ours.items():
            s = stats[c["group"]][k]; s[0] += 1
            if v != r[k]:
                s[1] += 1
                if len(examples[(c["group"], k)]) < 3:
                    examples[(c["group"], k)].append({"case": {x: c[x] for x in ("partial", "scale")}, "ours": v, "rtl": r[k]})
    out = {"groups": {g: {k: {"cases": v[0], "mismatches": v[1]} for k, v in d.items()} for g, d in stats.items()},
           "examples": {f"{g}:{k}": v for (g, k), v in examples.items()}}
    tot = sum(v[0] for d in stats.values() for v in d.values()); bad = sum(v[1] for d in stats.values() for v in d.values())
    out["total_comparisons"], out["total_mismatches"] = tot, bad
    for g, d in out["groups"].items():
        print(g, {k: f'{v["mismatches"]}/{v["cases"]}' for k, v in d.items()})
    print("total", bad, "/", tot)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
