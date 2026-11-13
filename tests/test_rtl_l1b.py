"""L1b: the whole matrix unit (InnerProductTreesTop / SystolicArrayTop) against our models, bit for bit.

The Verilator runs themselves (scripts/rtl/run_l1b.sh, which needs mill) are NOT started from here. This file
  - validates checked-in small stimuli (scripts/rtl/data/l1b_tight_{ipt,sa}.*) against the RTL result stored next to
    them, with our models, and with npu-model's rtl_math chain when a checkout is present;
  - tests the comparer logic (a corrupted word must be caught; stray / duplicate MREG writes must be flagged);
  - checks the harness copy still wires the core like npu-model's (skips without the RTL clone) and that npu-model's
    fixtures are the ones the harness was calibrated against.
"""
import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RTL_DIR = ROOT / "scripts" / "rtl"
DATA = RTL_DIR / "data"
sys.path.insert(0, str(RTL_DIR))

import l1b_compare as cmpmod  # noqa: E402
import l1b_gen  # noqa: E402

RTL_CLONE = Path(os.environ.get("ATLAS_NPU_RTL", "/scratch2/lorenhung/atlas-npu-rtl/src"))
NPU_MODEL = Path(os.environ.get("NPU_MODEL", str(RTL_CLONE / "npu-model")))
UNITS = [u for u in ("ipt", "sa") if (DATA / f"l1b_tight_{u}.meta.json").exists() and (DATA / f"l1b_tight_{u}.tiles.json").exists()]
need_stored = pytest.mark.skipif(not UNITS, reason="no stored L1b stimulus/result in scripts/rtl/data")
need_clone = pytest.mark.skipif(not (RTL_CLONE / "mill").exists() or not (NPU_MODEL / "npu_model" / "configs" / "rtl_math.py").exists(),
                                reason="needs the atlas-npu RTL clone with mill and npu-model ($ATLAS_NPU_RTL, $NPU_MODEL)")


def load(unit):
    return (json.loads((DATA / f"l1b_tight_{unit}.meta.json").read_text()),
            json.loads((DATA / f"l1b_tight_{unit}.tiles.json").read_text()))


# The model's own out_fmt_sel=OUT_E4M3 path feeds the E4M3 byte back after every K tile; the RTL pops once. Unused by the
# chip flow (no caller passes out_fmt_sel), so it is expected to differ for K > 32 and excluded from the gate.
PER_TILE = "vs_ours_out_fmt_per_tile_feedback"


def mismatches(res):
    return {k: v for k, (n, v) in res["totals_cases_mismatches"].items() if v and k != PER_TILE}


@need_stored
@pytest.mark.parametrize("unit", UNITS)
def test_stored_rtl_tiles_match_our_models(unit):
    meta, tiles = load(unit)
    assert len(tiles) == len(meta["pops"]) and meta["mode"] == "tight" and meta["layout"] == "nk"
    res = cmpmod.compare_run(meta, tiles, use_rtl_math=False)
    assert not mismatches(res), res["pops"]
    n = res["totals_cases_mismatches"]
    assert n["vs_ours"][0] == 1024 * len(tiles)
    assert n[PER_TILE][1] > 0, "the per-tile-feedback out_fmt path is expected to differ from the RTL for K > 32"


@need_stored
@need_clone
@pytest.mark.parametrize("unit", UNITS)
def test_stored_rtl_tiles_match_rtl_math_chain(unit):
    meta, tiles = load(unit)
    res = cmpmod.compare_run(meta, tiles, npu_model=str(NPU_MODEL))
    assert not mismatches(res), res["pops"]


@need_stored
@pytest.mark.parametrize("unit", UNITS)
def test_comparer_catches_a_corrupted_tile(unit):
    meta, tiles = load(unit)
    bad = copy.deepcopy(tiles)
    bf = next(i for i, t in enumerate(bad) if t["kind"] == "bf16")
    f8 = next(i for i, t in enumerate(bad) if t["kind"] == "fp8")
    bad[bf]["words"][37] ^= 0x0100
    bad[f8]["words"][1000] ^= 0x01
    res = cmpmod.compare_run(meta, bad, use_rtl_math=False)
    assert res["pops"][bf]["vs_ours"]["mismatches"] == 1
    assert res["pops"][f8]["vs_ours"]["mismatches"] == 1
    assert res["totals_cases_mismatches"]["vs_ours"][1] == 2


def test_extract_tiles_flags_stray_duplicate_and_missing_writes():
    meta = {"pops": [{"chain": 0, "kind": "fp8", "scale": 127, "bank": 8, "cycle": 1}]}
    rows = [[8, r, format(0x0102030405060708 + r, "x")] for r in range(32)]
    ok, anomalies = cmpmod.extract_tiles(meta, {"events": [{"cycle": 1, "writes": rows}]})
    assert anomalies == [] and ok[0]["words"][:3] == [0x08, 0x07, 0x06]
    bad = rows[:-1] + [[8, 0, "1"], [9, 3, "2"]]
    _, anomalies = cmpmod.extract_tiles(meta, {"events": [{"cycle": 1, "writes": bad}]})
    assert any("stray" in a for a in anomalies) and any("duplicate" in a for a in anomalies) and any("never written" in a for a in anomalies)


def test_bf16_pop_word_order():
    """PopAccBF16: lanes 0..15 of a row in mreg d, lanes 16..31 in mreg d+1, lane l at bits 16l."""
    meta = {"pops": [{"chain": 0, "kind": "bf16", "scale": None, "bank": 8, "cycle": 1}]}
    lo = sum((0x100 + l) << (16 * l) for l in range(16))
    hi = sum((0x200 + l) << (16 * l) for l in range(16))
    writes = [[8, r, format(lo, "x")] for r in range(32)] + [[9, r, format(hi, "x")] for r in range(32)]
    tiles, anomalies = cmpmod.extract_tiles(meta, {"events": [{"cycle": 1, "writes": writes}]})
    assert not anomalies and tiles[0]["words"][:3] == [0x100, 0x101, 0x102] and tiles[0]["words"][16] == 0x200


@need_stored
@pytest.mark.parametrize("unit", UNITS)
def test_stored_stimulus_is_what_the_generator_makes(unit):
    if not Path(l1b_gen.xc.DEFAULT_EXPORT).exists():
        pytest.skip("TinyLlama export not available")
    meta, _ = load(unit)
    script, regenerated = l1b_gen.build(unit, "tight", "nk", None)
    assert json.loads(json.dumps(regenerated)) == meta


@need_clone
def test_harness_wiring_is_npu_models():
    ours = (RTL_DIR / "AtlasMxuScriptTest.scala").read_text()
    theirs = (NPU_MODEL / "tests" / "rtl" / "scala" / "mxu" / "NpuModelMatrixTraceTest.scala").read_text()
    wiring = [ln.strip() for ln in theirs.splitlines() if "core." in ln or "io.reads(" in ln or "io.writes(" in ln or "RegNext" in ln]
    assert wiring
    for ln in wiring:
        assert ln in [x.strip() for x in ours.splitlines()], ln


@need_clone
def test_calibration_targets_unchanged():
    rec = json.loads((DATA / "l1b_calibration.json").read_text())
    for name in ("ipt", "sa"):
        data = (NPU_MODEL / "tests" / "rtl" / f"{name}_traces.json").read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec[f"{name}_traces_sha256"], f"npu-model {name}_traces.json changed: recalibrate"
