"""Compare the popped tiles of an L1b RTL run (AtlasMxuScriptTest) with atlas-torchao's models and with npu-model's
RTL-fitted rtl_math chain, bit for bit.

    python l1b_compare.py probe   --meta M --trace T [--out layout.json]      weight-layout probe
    python l1b_compare.py compare --meta M --trace T --out result.json [--tiles-out compact.json]
    python l1b_compare.py compare --meta M --tiles compact.json --out result.json   (from a stored RTL result)
    python l1b_compare.py aggregate --out l1b.json RESULT.json...

A pop is a [32 x 32] tile: row m (activation row), lane n (output channel). PopAccBF16 writes lanes 0..15 of a row to
mreg d and lanes 16..31 to mreg d+1; PopAccFP8 writes the 32 bytes to mreg d. Models compared:
  ours      accum_torch.ipt_linear_bits / sa_linear_bits (BF16 chain); FP8 = output_conv_stage(chain, OUT_E4M3, scale-127),
            and the model's direct out_fmt_sel=OUT_E4M3 call (key vs_ours_out_fmt_direct for K=32; for K>32 that call quantizes and feeds
            back after every tile, which the RTL does not do (it pops once): key vs_ours_out_fmt_per_tile_feedback, expected to differ)
  rtl_math  npu_model rtl_math ipt_row / sa_fma chain; FP8 = pack_row(mxu=True)
  conv-only the FP8 conversion of the RTL's own BF16 pop, which separates a pop-conversion fault from a chain fault
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import accum_torch as at
import npu_model_crosscheck as xc


def unhex(h: str, n: int) -> torch.Tensor:
    return torch.tensor(list(bytes.fromhex(h)), dtype=torch.uint8).reshape(n, -1)


def extract_tiles(meta: dict, trace: dict) -> list:
    """Decode every pop of the run from the MREG writes of the trace. Raises on a stray or duplicate write."""
    written, anomalies = {}, []
    expected = set()
    for p in meta["pops"]:
        banks = [p["bank"], p["bank"] + 1] if p["kind"] == "bf16" else [p["bank"]]
        expected |= {(b, r) for b in banks for r in range(32)}
    for ev in trace["events"]:
        for bank, row, h in ev["writes"]:
            if (bank, row) not in expected:
                anomalies.append(f"stray write cycle {ev['cycle']} mreg {bank} row {row}")
            elif (bank, row) in written:
                anomalies.append(f"duplicate write cycle {ev['cycle']} mreg {bank} row {row}")
            written[(bank, row)] = int(h, 16)
    missing = sorted(expected - set(written))
    if missing:
        anomalies.append(f"{len(missing)} expected rows never written, first {missing[:3]}")
    tiles = []
    for p in meta["pops"]:
        words = []
        for r in range(32):
            if p["kind"] == "bf16":
                lo, hi = written.get((p["bank"], r), 0), written.get((p["bank"] + 1, r), 0)
                words += [(lo >> (16 * l)) & 0xFFFF for l in range(16)] + [(hi >> (16 * l)) & 0xFFFF for l in range(16)]
            else:
                v = written.get((p["bank"], r), 0)
                words += [(v >> (8 * l)) & 0xFF for l in range(32)]
        tiles.append({"chain": p["chain"], "kind": p["kind"], "scale": p["scale"], "words": words})
    return tiles, anomalies


def ours_bf16(unit, x, w):
    return (at.ipt_linear_bits(x, w) if unit == "ipt" else at.sa_linear_bits(x, w, compile=False)).to(torch.int64).reshape(32, 32)


def rtl_math_bf16(R, unit, x, w):
    K = x.shape[1]
    if unit == "ipt":
        out = torch.zeros(32, 32, dtype=torch.int64)
        for m in range(32):
            acc = torch.zeros(32, dtype=torch.int32)
            for t in range(0, K, 32):
                acc = xc._bits16(R.ipt_row(x[m, t:t + 32], w[:, t:t + 32], xc.as_bf16(acc)))
            out[m] = acc
        return out
    xs = x[:, None, :].expand(32, 32, K).reshape(1024, K)
    ws = w[None, :, :].expand(32, 32, K).reshape(1024, K)
    acc = xc.as_bf16(torch.zeros(1024, dtype=torch.int32))
    for k in range(K):
        acc = R.sa_fma(xs[:, k].contiguous(), ws[:, k].contiguous(), acc)
    return xc._bits16(acc).to(torch.int64).reshape(32, 32)


def fp8_ours(bf16_bits, scale):
    return at.output_conv_stage(bf16_bits.to(torch.int32).clone(), at.OUT_E4M3, scale - 127).to(torch.int64)


def fp8_rtl_math(R, bf16_bits, scale):
    return R.pack_row(xc.as_bf16(bf16_bits.to(torch.int32).flatten()), scale, mxu=True).to(torch.int64).reshape(32, 32)


def cmp(name, a, b):
    r = xc._result(name, a.to(torch.int64).reshape(-1), b.to(torch.int64).reshape(-1), nan_equal=False)
    return {"cases": r["cases"], "mismatches": r["mismatches"],
            "examples": [{"m": e["index"] // 32, "n": e["index"] % 32, "a": e["ours"], "b": e["npu_model"]} for e in r["examples"][:3]]}


def compare_run(meta: dict, tiles: list, anomalies=(), npu_model=xc.DEFAULT_NPU_MODEL, use_rtl_math=True) -> dict:
    unit = meta["unit"]
    R = xc.load_rtl_math(npu_model) if use_rtl_math else None
    chains = []
    for c in meta["chains"]:
        K = c["K"]
        chains.append((unhex(c["x"], 32).reshape(32, K), unhex(c["w"], 32).reshape(32, K)))
    cache = {}
    results = []
    for ti, t in enumerate(tiles):
        ci = t["chain"]
        if ci not in cache:
            x, w = chains[ci]
            cache[ci] = {"ours": ours_bf16(unit, x, w), "rm": rtl_math_bf16(R, unit, x, w) if R else None}
        got = torch.tensor(t["words"], dtype=torch.int64).reshape(32, 32)
        o, rm = cache[ci]["ours"], cache[ci]["rm"]
        ch = meta["chains"][ci]
        row = {"chain": ch["name"], "K": ch["K"], "tiles": ch["K"] // 32, "kind": t["kind"], "scale": t["scale"]}
        if t["kind"] == "bf16":
            row["vs_ours"] = cmp("bf16 vs ours", got, o)
            if rm is not None:
                row["vs_rtl_math"] = cmp("bf16 vs rtl_math", got, rm)
                row["ours_vs_rtl_math"] = cmp("ours vs rtl_math", o, rm)
            cache[ci]["rtl_bf16"] = got
        else:
            s = t["scale"]
            row["vs_ours"] = cmp("fp8 vs ours", got, fp8_ours(o, s))
            if rm is not None:
                row["vs_rtl_math"] = cmp("fp8 vs rtl_math", got, fp8_rtl_math(R, rm, s))
            if "rtl_bf16" in cache[ci]:
                row["conv_only_ours"] = cmp("fp8 conv-only ours", got, fp8_ours(cache[ci]["rtl_bf16"], s))
                if R:
                    row["conv_only_rtl_math"] = cmp("fp8 conv-only rtl_math", got, fp8_rtl_math(R, cache[ci]["rtl_bf16"], s))
            x, w = chains[ci]                                    # the model's own out_fmt path (E4M3 byte fed back as psum after EVERY tile)
            direct = (at.ipt_linear_bits(x, w, scale_exp=s - 127, out_fmt_sel=at.OUT_E4M3) if unit == "ipt"
                      else at.sa_linear_bits(x, w, scale_exp=s - 127, out_fmt_sel=at.OUT_E4M3, compile=False))
            row["vs_ours_out_fmt_direct" if ch["K"] == 32 else "vs_ours_out_fmt_per_tile_feedback"] = cmp(
                "fp8 vs out_fmt", got, direct.to(torch.int64).reshape(32, 32))
        results.append(row)
    keys = ("vs_ours", "vs_rtl_math", "ours_vs_rtl_math", "conv_only_ours", "conv_only_rtl_math", "vs_ours_out_fmt_direct", "vs_ours_out_fmt_per_tile_feedback")
    tot = {k: [sum(r[k]["cases"] for r in results if k in r), sum(r[k]["mismatches"] for r in results if k in r)] for k in keys}
    return {"unit": unit, "mode": meta["mode"], "weight_layout": meta["layout"], "cycles": meta["cycles"],
            "anomalies": list(anomalies), "pops": results,
            "totals_cases_mismatches": {k: v for k, v in tot.items() if v[0]}}


def probe(meta, trace):
    """Weight layout: which of the two readings of the pushed mreg tile reproduces the RTL's matmul."""
    tiles, anomalies = extract_tiles(meta, trace)
    c = meta["chains"][0]
    x, w = unhex(c["x"], 32).reshape(32, 32), unhex(c["w"], 32).reshape(32, 32)   # w[r][l] is what mreg row r lane l held
    got = torch.tensor(tiles[0]["words"], dtype=torch.int64).reshape(32, 32)
    unit = meta["unit"]
    f = at.ipt_linear_bits if unit == "ipt" else (lambda a, b: at.sa_linear_bits(a, b, compile=False))
    # mreg row r holds the 32-vector of output channel r (K along lanes), as stored by gen layout nk
    row_is_n = f(x, w).to(torch.int64).reshape(32, 32)
    row_is_k = f(x, w.t().contiguous()).to(torch.int64).reshape(32, 32)
    res = {"unit": unit, "mreg_row_is_output_channel_mismatches": int((got != row_is_n).sum()),
           "mreg_row_is_k_index_mismatches": int((got != row_is_k).sum()), "of": 1024, "anomalies": anomalies}
    res["layout"] = ("nk" if res["mreg_row_is_output_channel_mismatches"] == 0 else
                     "kn" if res["mreg_row_is_k_index_mismatches"] == 0 else "undetermined")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "compare", "aggregate"]); ap.add_argument("files", nargs="*")
    ap.add_argument("--meta"); ap.add_argument("--trace"); ap.add_argument("--tiles"); ap.add_argument("--out"); ap.add_argument("--tiles-out")
    ap.add_argument("--no-rtl-math", action="store_true")
    a = ap.parse_args()
    if a.cmd == "aggregate":
        runs = {Path(f).stem: json.loads(Path(f).read_text()) for f in a.files}
        Path(a.out).write_text(json.dumps({"runs": runs}, indent=1)); return 0
    meta = json.loads(Path(a.meta).read_text())
    if a.cmd == "probe":
        res = probe(meta, json.loads(Path(a.trace).read_text()))
        print(json.dumps(res))
        if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
        return 0
    if a.tiles:
        tiles, anomalies = json.loads(Path(a.tiles).read_text()), []
    else:
        tiles, anomalies = extract_tiles(meta, json.loads(Path(a.trace).read_text()))
        if a.tiles_out: Path(a.tiles_out).write_text(json.dumps(tiles))
    res = compare_run(meta, tiles, anomalies, use_rtl_math=not a.no_rtl_math)
    print(res["unit"], res["mode"], "anomalies", res["anomalies"], res["totals_cases_mismatches"])
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
    # per-tile-feedback is the model's unused out_fmt path, expected to differ for K > 32 (see module docstring)
    gate = {k: v for k, v in res["totals_cases_mismatches"].items() if k != "vs_ours_out_fmt_per_tile_feedback"}
    return 0 if all(v[1] == 0 for v in gate.values()) and not anomalies else 1


if __name__ == "__main__":
    raise SystemExit(main())
