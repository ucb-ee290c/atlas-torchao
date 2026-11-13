"""Replay npu-model's Verilator trace of VectorEngineTop (tests/rtl/vector_traces.json, no build needed) through
ChipVPU / ref_numerics and compare every output word.

    python replay_vector_traces.py [--out result.json] [--npu-model DIR]

Harness encoding (NpuModelVectorTraceTest.scala): opcode = list index + 1 over
add sub mul rcp sqrt sin cos tanh log exp exp2 square cube rsum(vredsum.row) csum(vredsum.bf16) fp8 fp8pack
fp8unpack relu rmax rmin cmax cmin pairmax pairmin mov vliOne vliCol vliRow vliAll; vs1=0, vs2=2, vd=4,
scale 127, imm 0x3f80. MREG bank b row r holds 16 BF16 lanes (lane l at bits 16l). Initial memory:
bank*32+row*4+lane+0x3e80, or for *_special scenarios edge[(bank*3+row+lane)%16] (zeros, subnormals, inf, NaN, ...).
Banks of a BF16 register pair are (vs, vs+1) = 32 elements of a row: lanes 0..15 in vs, 16..31 in vs+1.
Operation semantics replayed here:
  binary   out[4+h][r] = op(mem[vs1+h][r], mem[vs2+h][r])               h in {0,1}
  unary    out[4+h][r] = f(mem[0+h][r])                                  (vs1 pair)
  row red  out[4][r] = out[5][r] = broadcast(red(mem[0][r] ++ mem[1][r])) (vredsum.row, vredmax.row, vredmin.row)
  col red  out[4+h][r][l] = red over the 64 rows of lane l of banks 0 and 1, broadcast over r (vredsum.bf16 etc.)
  vpack    out[4][j] = 32 bytes = elements 32j..32j+31 of mem[2] ++ mem[3] (row-major, 1024 elements), scale 127
  vunpack  out[4+h][r] = BF16 of bytes (h*32+r)*16.. +15 of mem[2] viewed as 32 rows x 32 bytes
which are checked here by the zero-mismatch result itself; any op that does not replay is reported.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import chip.chip_ops as co
import reference.ref_numerics as rn

NPU_MODEL = "/scratch2/lorenhung/atlas-npu-rtl/src/npu-model"
NAMES = ["add", "sub", "mul", "rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2", "square", "cube", "rsum", "csum", "fp8",
         "fp8pack", "fp8unpack", "relu", "rmax", "rmin", "cmax", "cmin", "pairmax", "pairmin", "mov", "vliOne", "vliCol", "vliRow", "vliAll"]
EDGE = [0, 0x8000, 1, 0x8001, 0x007f, 0x807f, 0x7f80, 0xff80, 0x7fc1, 0xffa1, 0x7f7f, 0xff7f, 0x3f80, 0xbf80, 0x3c80, 0x43ef]


def mem_init(special: bool):
    return torch.tensor([[[EDGE[(b * 3 + r + l) % 16] if special else 0x3e80 + b * 32 + r * 4 + l for l in range(16)]
                          for r in range(32)] for b in range(64)], dtype=torch.int32)


def bf(bits):               # int bits -> float32 tensor holding the BF16 value
    return co.bf16_bits_to_f32(bits.to(torch.int32))


def to_bits(x):             # float32 tensor of BF16 values -> raw bits (no re-rounding: exact)
    return co.f32_to_bf16_bits(x).to(torch.int64)


def col_sum(vpu_bits):      # vredsum.bf16 model: FP32 column accumulator over rows, upper-16-bit slice of the result
    acc = torch.zeros(16)
    for r in range(vpu_bits.shape[0]):
        acc = acc + bf(vpu_bits[r])
    return ((acc.contiguous().view(torch.int32) >> 16) & 0xFFFF).to(torch.int64)


def expected(name, special, m=None, shift=0, vs2=None):
    """Dict {(bank,row): list of 16 words (BF16) or 32 bytes} the op should write, or None if not modelled here."""
    base = name.replace("_special", "")
    if name in ("mov_overlap", "mov_handoff"):
        return None                       # two overlapping commands; the mov datapath itself is covered by "mov" and "mov_inplace"
    base = base.split("_")[0]
    m = mem_init(special) if m is None else m
    v = co.ChipVPU()
    out = {}
    if base in ("add", "sub", "mul", "pairmax", "pairmin"):
        fn = {"add": v.add, "sub": v.sub, "mul": v.mul, "pairmax": v.pairwise_max, "pairmin": v.pairwise_min}[base]
        s2 = 0 if name == "add_mirrored" else (2 if vs2 is None else vs2)
        for h in range(2):
            for r in range(32):
                out[(4 + h, r)] = to_bits(fn(bf(m[0 + h, r]), bf(m[s2 + h, r]))).tolist()
    elif base in ("rcp", "sqrt", "sin", "cos", "tanh", "log", "exp", "exp2", "square", "cube", "relu", "mov"):
        fn = {"log": v.log2}.get(base) or getattr(v, base)
        dst = 0 if name == "mov_inplace" else 4
        for h in range(2):
            for r in range(32):
                out[(dst + h, r)] = to_bits(fn(bf(m[0 + h, r]))).tolist()
    elif base in ("rsum", "rmax", "rmin"):
        fn = getattr(v, base)
        for r in range(32):
            row = torch.cat([bf(m[0, r]), bf(m[1, r])])
            res = to_bits(fn(row[None, :]))[0].item()
            out[(4, r)] = out[(5, r)] = [res] * 16
    elif base in ("csum", "cmax", "cmin"):
        allrows = torch.cat([m[0], m[1]])                       # 64 rows x 16 lanes
        if base == "csum":
            res = col_sum(allrows).tolist()
        else:
            k = co.ordered_key(allrows.to(torch.int64) & 0xFFFF)
            res = co.key_to_bits(k.amax(0) if base == "cmax" else k.amin(0)).tolist()
        for h in range(2):
            for r in range(32):
                out[(4 + h, r)] = res
    elif base == "fp8pack":                         # vs2 pair (2,3): 1024 elements row-major, 32 per output row
        flat = torch.cat([m[2].reshape(-1), m[3].reshape(-1)])
        for j in range(32):
            x = bf(flat[32 * j:32 * j + 32]).to(torch.bfloat16)
            out[(4, j)] = rn.e4m3_vpack(x, shift).to(torch.int64).tolist()
    elif base == "fp8unpack":                       # vs2 = 2: 32 rows x 32 bytes, out bank 4 = elements 0..511, bank 5 = 512..1023
        flat = torch.tensor([(int(m[2, r, l // 2]) >> (8 * (l % 2))) & 0xFF for r in range(32) for l in range(32)], dtype=torch.uint8)
        u = rn.e4m3_unpack(flat, shift).to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
        for h in range(2):
            for r in range(32):
                out[(4 + h, r)] = u[(h * 32 + r) * 16:(h * 32 + r) * 16 + 16].tolist()
    else:
        return None
    return out


def words(h: str, n_lane: int, width: int):
    v = int(h, 16)
    return [(v >> (width * l)) & ((1 << width) - 1) for l in range(n_lane)]


def is_nan(w):
    return (w & 0x7F80) == 0x7F80 and (w & 0x7F) != 0


def replay(npu_model=NPU_MODEL):
    """Rows per scenario of vector_traces.json replayed through ChipVPU / ref_numerics."""
    traces = json.loads((Path(npu_model) / "tests" / "rtl" / "vector_traces.json").read_text())
    rows = []
    for name, events in traces.items():
        exp = expected(name, name.endswith("_special"))
        if exp is None:
            rows.append({"scenario": name, "status": "not modelled by ChipVPU (move/vli/overlap variants of other ops are covered by their own scenario)"}); continue
        got = {}
        for ev in events:
            for bank, row, h in ev["writes"]:
                got[(bank, row)] = h
        cases = bad = nan_only = 0; ex = []
        byte_mode = name.startswith("fp8pack")
        for key, want in exp.items():
            if key not in got:
                bad += len(want); cases += len(want); ex.append({"missing": key}); continue
            have = words(got[key], 32 if byte_mode else 16, 8 if byte_mode else 16)
            for l, (w, g) in enumerate(zip(want, have)):
                cases += 1
                if w != g:
                    if not byte_mode and is_nan(w) and is_nan(g):
                        nan_only += 1
                    else:
                        bad += 1
                        if len(ex) < 3: ex.append({"bank": key[0], "row": key[1], "lane": l, "model": w, "rtl": g})
        extra = sorted(set(got) - set(exp))
        rows.append({"scenario": name, "words": cases, "mismatches": bad, "nan_payload_only": nan_only,
                     "unmodelled_writes": len(extra), "examples": ex})
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out"); ap.add_argument("--npu-model", default=NPU_MODEL)
    a = ap.parse_args()
    rows = replay(a.npu_model)
    tot = sum(r.get("words", 0) for r in rows); bad = sum(r.get("mismatches", 0) for r in rows)
    for r in rows:
        print(f"{r['scenario']:<18}", r.get("status") or f"{r['mismatches']}/{r['words']} (nan payload only {r['nan_payload_only']}, unmodelled writes {r['unmodelled_writes']}) {r['examples'][:2]}")
    print("total", bad, "/", tot)
    if a.out:
        Path(a.out).write_text(json.dumps({"scenarios": rows, "words": tot, "mismatches": bad}, indent=1))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
