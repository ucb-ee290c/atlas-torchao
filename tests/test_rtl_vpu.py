"""L0b: vector-unit gaps (vmul, vadd/vsub, vredsum.row, vredmax/vredmin.row, vredsum.bf16 column, vpack/vunpack).

Verilator runs (src/scripts/rtl/run_vpu.sh, needs mill) are not started here. This file replays
  - npu-model's captured vector_traces.json through ChipVPU (needs the npu-model checkout and the pi0-quant refs);
  - a stored subset of our own VectorEngineTop run (src/scripts/rtl/data/vpu_stored.json) through the comparer.
"""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "scripts" / "rtl"))
sys.path.insert(0, str(ROOT / "tests"))

import replay_vector_traces as rv  # noqa: E402
import vpu_l0b  # noqa: E402

NPU_MODEL = os.environ.get("NPU_MODEL", rv.NPU_MODEL)
STORED = ROOT / "src" / "scripts" / "rtl" / "data" / "vpu_stored.json"


@pytest.mark.skipif(not (Path(NPU_MODEL) / "tests" / "rtl" / "vector_traces.json").exists(), reason="needs an npu-model checkout ($NPU_MODEL)")
def test_npu_model_vector_trace_replays_through_chipvpu():
    from refutil import need_refs
    need_refs()
    rows = rv.replay(NPU_MODEL)
    modelled = [r for r in rows if "words" in r]
    assert len(modelled) >= 40
    bad = {r["scenario"]: (r["mismatches"], r["examples"]) for r in modelled if r["mismatches"]}
    assert not bad, bad
    assert sum(r["words"] for r in modelled) >= 40000


@pytest.mark.skipif(not STORED.exists(), reason="no stored VPU stimulus")
def test_stored_vpu_run_matches_chipvpu():
    d = json.loads(STORED.read_text())
    stats = vpu_l0b.compare(d["meta"], d["rtl"])
    assert {"mul", "add", "rsum", "csum", "rmax", "rmin", "fp8pack", "fp8unpack"} <= set(stats)
    assert {op: s["mismatches"] for op, s in stats.items() if s["mismatches"]} == {}
    assert sum(s["words"] for s in stats.values()) > 10000


@pytest.mark.skipif(not STORED.exists(), reason="no stored VPU stimulus")
def test_vpu_comparer_catches_a_flipped_bit():
    d = json.loads(STORED.read_text())
    rtl = json.loads(json.dumps(d["rtl"]))
    i = next(k for k, r in enumerate(d["meta"]["runs"]) if r["op"] == "mul")
    cyc, bank, row, h = rtl[i]["writes"][0]
    rtl[i]["writes"][0] = [cyc, bank, row, format(int(h, 16) ^ 0x1, "x")]
    stats = vpu_l0b.compare(d["meta"], rtl)
    assert stats["mul"]["mismatches"] >= 1
