"""Calibration for the L1b harness: regenerate the scripts that make AtlasMxuScriptTest replay
npu-model's NpuModelMatrixTraceTest, and compare its traces with npu-model's ipt_traces.json / sa_traces.json.

    python l1b_calibrate.py gen   <outdir>                 writes calib_{ipt,sa}_script.json
    python l1b_calibrate.py check <outdir> [npu_model_dir] byte-compares calib_{ipt,sa}_trace.json
"""
import json, sys
from pathlib import Path

NPU_MODEL = "/scratch2/lorenhung/atlas-npu-rtl/src/npu-model"
COMMANDS = [(1, 0, 0, 0, 0), (34, 2, 2, 1, 0), (67, 3, 6, 1, 0), (100, 1, 8, 0, 0), (133, 4, 10, 0, 0), (166, 5, 8, 0, 0),
            (262, 4, 12, 0, 0), (295, 6, 8, 0, 0), (391, 4, 14, 0, 0), (424, 3, 16, 0, 0), (457, 0, 18, 0, 1),
            (460, 5, 20, 1, 0), (557, 4, 22, 1, 0)]


def init_mem():
    mem = {}
    for bank in range(64):
        rows = []
        for row in range(32):
            if bank in (2, 3):
                v = sum((0x3d00 + bank * 64 + row * 4 + l) << (16 * l) for l in range(16))
            else:
                v = sum((0x20 + ((bank * 7 + row * 3 + l) % 48) + (128 if (row + l) % 3 == 0 else 0)) << (8 * l) for l in range(32))
            rows.append(format(v, "x"))
        mem[str(bank)] = rows
    return mem


def main():
    mode, outdir = sys.argv[1], Path(sys.argv[2])
    if mode == "gen":
        outdir.mkdir(parents=True, exist_ok=True)
        for sa in (True, False):
            name = "sa" if sa else "ipt"
            (outdir / f"calib_{name}_script.json").write_text(json.dumps(
                {"sa": sa, "cycles": 592, "commands": [list(c) for c in COMMANDS], "mem": init_mem()}))
    else:
        npu = Path(sys.argv[3] if len(sys.argv) > 3 else NPU_MODEL)
        bad = 0
        for name in ("ipt", "sa"):
            a = (outdir / f"calib_{name}_trace.json").read_bytes()
            b = (npu / "tests" / "rtl" / f"{name}_traces.json").read_bytes()
            print(name, "BYTE-IDENTICAL" if a == b else "DIFFERENT", len(a), len(b))
            bad += a != b
        raise SystemExit(bad)


if __name__ == "__main__":
    main()
