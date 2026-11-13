"""Stimulus and comparison for the VectorEngineTop run (AtlasVpuScriptTest.scala): vmul, vadd, vsub, vredsum.row,
vredmax.row, vredmin.row, vredsum.bf16 (column) and friends, random plus special inputs, vs ChipVPU bit for bit.

    python vpu_l0b.py gen     --out-prefix P          writes P.script.json and P.meta.json
    python vpu_l0b.py compare --meta P.meta.json --rtl rtl_out.json [--out result.json]

Operand layout and op semantics are those of replay_vector_traces.expected (verified there against npu-model's
Verilator fixture): vs1=0, vs2=2, vd=4; a BF16 register pair is banks (v, v+1) of 32 rows x 16 lanes.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_vector_traces as RV

OPS = {"add": 1, "sub": 2, "mul": 3, "rsum": 14, "csum": 15, "fp8pack": 17, "fp8unpack": 18, "rmax": 20, "rmin": 21,
       "cmax": 22, "cmin": 23, "pairmax": 24, "pairmin": 25}
SPECIAL = [0x0000, 0x8000, 0x0001, 0x8001, 0x0040, 0x007F, 0x807F, 0x0080, 0x8080, 0x00FF, 0x0100, 0x1000, 0x3F7F, 0x3F80, 0xBF80,
           0x3F81, 0x3FFF, 0x4000, 0x4040, 0x7F00, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC0, 0x7FC1, 0xFFC0, 0x7F81, 0x0001 | 0x0100,
           0x3880, 0x3C00, 0x4300, 0x43EF, 0x2000, 0x5F00, 0x5F80, 0x6000, 0x1F80, 0x2080, 0x7E80, 0x0200, 0x7C00, 0x7EFF, 0xBC00]


def rnd(g, shape, lo, hi):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)


def bf16_words(g, shape, mode):
    """Random BF16 bit patterns. modes: uniform (any encoding), normal (moderate exponents), wide (all finite exponents),
    sub (exponents 0..3, so subnormals and tiny normals), special (from SPECIAL)."""
    sign, mant = rnd(g, shape, 0, 2), rnd(g, shape, 0, 128)
    if mode == "uniform":
        return rnd(g, shape, 0, 65536)
    if mode == "special":
        return torch.tensor(SPECIAL)[rnd(g, shape, 0, len(SPECIAL))]
    lo, hi = {"normal": (118, 138), "wide": (1, 255), "sub": (0, 4)}[mode]
    return (sign << 15) | (rnd(g, shape, lo, hi) << 7) | mant


def pair_words(g, mode):
    """(A, B) [32,16] each, for the pointwise binary ops."""
    shape = (32, 16)
    if mode in ("uniform", "normal", "wide", "sub", "special"):
        return bf16_words(g, shape, mode), bf16_words(g, shape, mode)
    a = bf16_words(g, shape, "wide")
    ea = (a >> 7) & 0xFF
    if mode == "mul_underflow":          # product exponent ea+eb-127 around 0 / -10: subnormal and flush region
        target = rnd(g, shape, -12, 6)
    elif mode == "mul_overflow":         # around 254
        target = rnd(g, shape, 248, 258)
    elif mode == "mul_sub_in":           # one operand subnormal, the other large: product normal/subnormal boundary
        a = (rnd(g, shape, 0, 2) << 15) | rnd(g, shape, 1, 128)
        ea = torch.zeros(shape, dtype=torch.int64)
        target = rnd(g, shape, -4, 8)
    elif mode == "cancel":               # a - b / a + b with nearby magnitudes, opposite signs
        b = a ^ 0x8000
        flip = rnd(g, shape, 0, 4) == 0
        b = torch.where(flip, b ^ rnd(g, shape, 0, 4), b)
        return a, b
    elif mode == "close":                # exponents within 12 of each other
        b = (rnd(g, shape, 0, 2) << 15) | (torch.clamp(ea + rnd(g, shape, -12, 13), 1, 254) << 7) | rnd(g, shape, 0, 128)
        return a, b
    eb = torch.clamp(target - ea + 127, 0, 254)
    b = (rnd(g, shape, 0, 2) << 15) | (eb << 7) | rnd(g, shape, 0, 128)
    return a, b


def hex_rows(t):            # [32,16] words -> 32 hex rows
    return [format(sum(int(v) << (16 * l) for l, v in enumerate(row)), "x") for row in t.tolist()]


def build(seed=5):
    g = torch.Generator().manual_seed(seed)
    runs = []

    def add(name, op, banks, scale=127, **kw):
        runs.append({"name": name, "op": OPS[op], "opname": op, "vs1": 0, "vs2": 2, "vd": 4, "scale": scale, "imm": 0x3F80, "cycles": 140,
                     "banks": {str(b): v for b, v in banks.items()}, **kw})

    binary_modes = {"mul": ["uniform", "normal", "wide", "sub", "mul_underflow", "mul_overflow", "mul_sub_in", "special"],
                    "add": ["uniform", "normal", "wide", "sub", "cancel", "close", "special"],
                    "sub": ["uniform", "normal", "wide", "sub", "cancel", "close", "special"],
                    "pairmax": ["uniform", "special"], "pairmin": ["uniform", "special"]}
    reps = {"mul": 8, "add": 5, "sub": 5, "pairmax": 3, "pairmin": 3}
    for op, modes in binary_modes.items():
        for mode in modes:
            for i in range(reps[op] if mode != "special" else 1):
                a0, b0 = pair_words(g, mode)
                a1, b1 = pair_words(g, mode)
                add(f"{op}:{mode}:{i}", op, {0: a0, 1: a1, 2: b0, 3: b1})
    # all ordered pairs of the special values (44 x 44 = 1936 pairs -> two runs, padded with random special pairs)
    pairs = [(a, b) for a in SPECIAL for b in SPECIAL]
    pairs += [(SPECIAL[int(rnd(g, (1,), 0, len(SPECIAL)))], SPECIAL[int(rnd(g, (1,), 0, len(SPECIAL)))]) for _ in range(2048 - len(pairs))]
    for op in ("mul", "add", "sub"):
        for k in range(2):
            chunk = torch.tensor(pairs[k * 1024:(k + 1) * 1024])
            A, B = chunk[:, 0].reshape(64, 16), chunk[:, 1].reshape(64, 16)
            add(f"{op}:allspecialpairs:{k}", op, {0: A[:32], 1: A[32:], 2: B[:32], 3: B[32:]})
    # reductions
    red_modes = ["normal", "wide", "sub", "special", "uniform", "cancel"]
    for op, reps_ in (("rsum", 6), ("rmax", 3), ("rmin", 3), ("csum", 6), ("cmax", 2), ("cmin", 2)):
        for mode in red_modes:
            for i in range(reps_ if mode != "special" else 2):
                if mode == "cancel":
                    x = bf16_words(g, (2, 32, 16), "normal")
                    x = torch.where(rnd(g, x.shape, 0, 3) == 0, x ^ 0x8000, x)
                    # exactly cancelling partner for half of the elements
                    x[1] = torch.where(rnd(g, (32, 16), 0, 2) == 0, x[0] ^ 0x8000, x[1])
                else:
                    x = bf16_words(g, (2, 32, 16), mode)
                add(f"{op}:{mode}:{i}", op, {0: x[0], 1: x[1]})
    # fp8 conversions: vpack random/special at several scales, vunpack all 256 codes at several scales
    for s in (127, 120, 133, 100, 150, 0, 255):
        for mode in ("normal", "wide", "special"):
            add(f"fp8pack:{mode}:s{s}", "fp8pack", {2: bf16_words(g, (32, 16), mode), 3: bf16_words(g, (32, 16), mode)}, scale=s)
        words = (torch.arange(512) * 2 % 256 | ((torch.arange(512) * 2 + 1) % 256) << 8).reshape(32, 16)   # lane l of row r holds bytes 2l, 2l+1: every code 4 times
        add(f"fp8unpack:allcodes:s{s}", "fp8unpack", {2: words}, scale=s)
    script = {"runs": [{k: r[k] for k in ("op", "vs1", "vs2", "vd", "scale", "imm", "cycles")} |
                       {"mem": {b: hex_rows(v) for b, v in r["banks"].items()}} for r in runs]}
    meta = {"runs": [{"name": r["name"], "op": r["opname"], "scale": r["scale"],
                      "banks": {b: v.tolist() for b, v in r["banks"].items()}} for r in runs]}
    return script, meta


def is_nan(w):
    return (w & 0x7F80) == 0x7F80 and (w & 0x7F) != 0


def compare(meta, rtl):
    stats = defaultdict(lambda: {"runs": 0, "words": 0, "mismatches": 0, "nan_payload_only": 0, "examples": []})
    for run, res in zip(meta["runs"], rtl):
        op = run["op"]; st = stats[op]; st["runs"] += 1
        m = torch.zeros(64, 32, 16, dtype=torch.int32)
        for b, v in run["banks"].items():
            m[int(b)] = torch.tensor(v, dtype=torch.int32)
        exp = RV.expected(op, False, m=m, shift=run["scale"] - 127, vs2=2)
        # The RTL clamps the E8M0 scale exponent to [-128, 127] (VectorEngine.scala e8m0ToScaleExpClamped), so scale byte 255
        # means +127; ref_numerics takes the raw exponent (+128). Words that differ only because of that are counted apart.
        clamped = RV.expected(op, False, m=m, shift=min(127, max(-128, run["scale"] - 127)), vs2=2) if op.startswith("fp8") and run["scale"] == 255 else None
        got = {}
        for cyc, bank, row, h in res["writes"]:
            got[(bank, row)] = h
        byte_mode = op == "fp8pack"
        for key, want in exp.items():
            if key not in got:
                st["words"] += len(want); st["mismatches"] += len(want)
                if len(st["examples"]) < 3: st["examples"].append({"run": run["name"], "missing": list(key)})
                continue
            have = RV.words(got[key], 32 if byte_mode else 16, 8 if byte_mode else 16)
            for l, (w, g_) in enumerate(zip(want, have)):
                st["words"] += 1
                if w != g_:
                    if clamped is not None and clamped[key][l] == g_:
                        st["scale255_unclamped_model_only"] = st.get("scale255_unclamped_model_only", 0) + 1
                    elif not byte_mode and is_nan(w) and is_nan(g_):
                        st["nan_payload_only"] += 1
                    else:
                        st["mismatches"] += 1
                        if len(st["examples"]) < 6:
                            ex = {"run": run["name"], "bank": key[0], "row": key[1], "lane": l, "model": hex(w), "rtl": hex(g_)}
                            if op in ("add", "sub", "mul", "pairmax", "pairmin"):
                                b0 = 0 + (key[0] - 4)
                                ex["a"] = hex(int(m[b0, key[1], l])); ex["b"] = hex(int(m[2 + b0, key[1], l]))
                            st["examples"].append(ex)
        extra = set(got) - set(exp)
        if extra:
            st["unmodelled_writes"] = st.get("unmodelled_writes", 0) + len(extra)
    return dict(stats)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["gen", "compare"])
    ap.add_argument("--out-prefix"); ap.add_argument("--meta"); ap.add_argument("--rtl"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "gen":
        script, meta = build()
        Path(a.out_prefix + ".script.json").write_text(json.dumps(script)); Path(a.out_prefix + ".meta.json").write_text(json.dumps(meta))
        print(len(script["runs"]), "runs")
        return 0
    stats = compare(json.loads(Path(a.meta).read_text()), json.loads(Path(a.rtl).read_text()))
    for op, s in stats.items():
        print(f"{op:<10} runs {s['runs']:>3} words {s['words']:>7} mismatches {s['mismatches']:>6} nan-payload-only {s['nan_payload_only']:>5}", s["examples"][:2])
    tot = sum(s["mismatches"] for s in stats.values())
    print({op: s["scale255_unclamped_model_only"] for op, s in stats.items() if "scale255_unclamped_model_only" in s})
    if a.out: Path(a.out).write_text(json.dumps({"ops": stats, "total_mismatches": tot}, indent=1))
    return 0 if tot == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
