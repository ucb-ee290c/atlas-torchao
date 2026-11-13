"""Stimulus for the L1b run (AtlasMxuScriptTest.scala): real TinyLlama weight tiles through the actual
InnerProductTreesTop / SystolicArrayTop, K chained in full, then PopAccBF16 and PopAccFP8 (several scale bytes).

    python l1b_gen.py --unit ipt|sa --mode probe|main|tight --layout nk|kn --out-prefix P

writes P.script.json (read by AtlasMxuScriptTest via MXU_SCRIPT) and P.meta.json (operands + pop list,
read by l1b_compare.py). Commands are [cycle, op, mreg bank, accSel, weightSlot, scaleE8M0].

  probe   one K=32 tile of random codes, to find which weight layout the push expects
  main    conservative spacing (100 cycles between commands): K=32 (single tile, FP8 pop via out_fmt),
          K=2048 on two real matrices, K=5632 (down_proj); first tile Matmul, later MatmulAcc, ascending
  tight   the minimum issue gaps of npu_spec/04_functional_units (with one cycle of slack where single
          issue forces it): K=256, weight slots alternated so a push overlaps the previous matmul
Weight layout nk: mreg row n holds output channel n's 32-vector of the K tile (PyTorch [N,K] rows);
kn: mreg row r lane l holds W[n=l][k=r].
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mxu.accum_torch as at
import reference.ref_numerics as rn
import npu_model_crosscheck as xc

PUSH_W, PUSH_ACC_FP8, PUSH_ACC_BF16, POP_FP8, POP_BF16, MATMUL, MATMUL_ACC = range(7)
W_BANK, A_BANK, FIRST_DEST = 0, 1, 8
NAMES = {"q": "model.layers.0.self_attn.q_proj.weight_codes", "gate": "model.layers.21.mlp.gate_proj.weight_codes",
         "down": "model.layers.21.mlp.down_proj.weight_codes"}


def row_hex(row_bytes) -> str:
    return format(int.from_bytes(bytes(int(v) for v in row_bytes), "little"), "x")


def weight_rows(w_tile: torch.Tensor, layout: str):
    """w_tile [32 n, 32 k] -> 32 mreg rows (hex)."""
    m = w_tile if layout == "nk" else w_tile.t()
    return [row_hex(m[r].tolist()) for r in range(32)]


def sane_codes(g, shape):
    c = torch.randint(0, 256, shape, generator=g, dtype=torch.int32)
    return torch.where((c & 0x7F) == 0x7F, c ^ 0x01, c).to(torch.uint8)


def load_chain(f, name, row0, K, g):
    w = f.get_tensor(NAMES[name])[row0:row0 + 32, :K].contiguous()
    xf = torch.randn(32, K, generator=g)
    x = rn.e4m3_vpack(xf.to(torch.bfloat16), rn.po2_scale_exp(xf, dims=-1)).to(torch.uint8)
    return x, w


def pick_scales(x, w):
    """Scale bytes centred on the output magnitude (so FP8 pops are mostly in range), plus identity and clamps."""
    bits = at.ipt_linear_bits(x, w).flatten().to(torch.int64)
    e = ((bits >> 7) & 0xFF) - 127
    e = e[(bits & 0x7FFF) != 0]
    med = int(e.sort().values[len(e) // 2]) if len(e) else 0
    c = 127 - (med - 3)                      # puts the median at 2^3
    return sorted({max(1, min(254, v)) for v in (c, c - 3, c + 3, 127)} | {0, 255})


def build(unit, mode, layout, _unused=None, seed=21):
    from safetensors import safe_open
    g = torch.Generator().manual_seed(seed)
    chains = []
    if mode == "probe":
        chains.append({"name": "probe", "x": sane_codes(g, (32, 32)), "w": sane_codes(g, (32, 32)), "acc": 0, "scales": []})
    else:
        with safe_open(xc.DEFAULT_EXPORT, "pt") as f:
            if mode == "main":
                spec = [("K32_q", "q", 0, 32, 0), ("K2048_q", "q", 0, 2048, 1), ("K2048_gate", "gate", 64, 2048, 0), ("K5632_down", "down", 32, 5632, 1)]
            else:
                spec = [("tight_K256_q", "q", 96, 256, 0), ("tight_K256_gate", "gate", 128, 256, 1)]
            for nm, t, r0, K, acc in spec:
                x, w = load_chain(f, t, r0, K, g)
                chains.append({"name": nm, "x": x, "w": w, "acc": acc, "scales": pick_scales(x, w)})
    cmds, loads, pops = [], [], []
    dest = FIRST_DEST
    cyc = 1
    gap_mpop, gap_pop = ((32 if unit == "ipt" else 64), 32) if mode == "tight" else (100, 100)
    for ci, ch in enumerate(chains):
        K = ch["x"].shape[1]; T = K // 32
        slot_alt = mode == "tight"
        for t in range(T):
            slot = (t % 2) if slot_alt else 0
            wt, xt = ch["w"][:, t * 32:(t + 1) * 32], ch["x"][:, t * 32:(t + 1) * 32]
            if mode == "tight":
                p_cyc, m_cyc, step = (cyc, cyc + 1, 64) if unit == "sa" else (cyc, cyc + 32, 33)
            else:
                p_cyc, m_cyc, step = cyc, cyc + 100, 200
            loads.append([p_cyc, W_BANK, weight_rows(wt, layout)])
            cmds.append([p_cyc, PUSH_W, W_BANK, 0, slot, 127])
            loads.append([m_cyc, A_BANK, [row_hex(xt[r].tolist()) for r in range(32)]])
            cmds.append([m_cyc, MATMUL if t == 0 else MATMUL_ACC, A_BANK, ch["acc"], slot, 127])
            last_m = m_cyc
            cyc += step
        cyc = last_m + gap_mpop
        cmds.append([cyc, POP_BF16, dest, ch["acc"], 0, 127])
        pops.append({"chain": ci, "kind": "bf16", "scale": None, "bank": dest, "cycle": cyc}); dest += 2; cyc += gap_pop
        for s in ch["scales"]:
            cmds.append([cyc, POP_FP8, dest, ch["acc"], 0, s])
            pops.append({"chain": ci, "kind": "fp8", "scale": s, "bank": dest, "cycle": cyc}); dest += 1; cyc += gap_pop
        cyc += gap_pop
    cycles = cyc + 120
    script = {"sa": unit == "sa", "cycles": cycles, "commands": cmds, "loads": loads}
    meta = {"unit": unit, "mode": mode, "layout": layout, "cycles": cycles, "pops": pops,
            "chains": [{"name": c["name"], "K": int(c["x"].shape[1]), "acc": c["acc"], "scales": c["scales"],
                        "x": bytes(c["x"].flatten().tolist()).hex(), "w": bytes(c["w"].flatten().tolist()).hex()} for c in chains]}
    return script, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", required=True, choices=["ipt", "sa"]); ap.add_argument("--mode", required=True, choices=["probe", "main", "tight"])
    ap.add_argument("--layout", default="nk", choices=["nk", "kn"]); ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()
    script, meta = build(args.unit, args.mode, args.layout, None)
    Path(args.out_prefix + ".script.json").write_text(json.dumps(script))
    Path(args.out_prefix + ".meta.json").write_text(json.dumps(meta))
    print(args.unit, args.mode, args.layout, "cycles", script["cycles"], "commands", len(script["commands"]), "pops", len(meta["pops"]))


if __name__ == "__main__":
    main()
