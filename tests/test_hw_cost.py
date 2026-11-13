"""Tests for src/hw_cost.py: padding, DMA/MXU cycle formulas against hw_params, reuse threshold, flush stats,
and the pop-scale consumer classification on the traced TinyLlama decoder layer."""
import math
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import hw_cost as hc  # noqa: E402
import hw_params as hp  # noqa: E402

EXPORT = ("/scratch2/lorenhung/atlas-fp8/handoff/atlas_v1/export_smoke/tinyllama/atlas_v1/model.safetensors")

TL_SHAPES = {"q_proj": (2048, 2048), "k_proj": (256, 2048), "v_proj": (256, 2048), "o_proj": (2048, 2048),
             "gate_proj": (5632, 2048), "up_proj": (5632, 2048), "down_proj": (2048, 5632)}


# ---- padding --------------------------------------------------------------------------------------
@pytest.mark.parametrize("n,p", [(1, 32), (32, 32), (33, 64), (4416, 4416), (588, 608), (0, 0)])
def test_pad_to(n, p):
    assert hc.pad_to(n) == p


def test_pad_waste_values():
    assert hc.pad_waste(33) == (31, 31 / 64)
    assert hc.pad_waste(588) == (20, 20 / 608)          # a 3*14*14 patch-embedding K
    assert hc.pad_waste(4416) == (0, 0.0)               # 138 tiles, the DiT adaLN K


def test_layer_cost_padding_counts_padded_tiles():
    lc = hc.layer_cost("x.proj", 33, 588)
    assert (lc.N_pad, lc.K_pad) == (64, 608)
    assert lc.tiles == 2 * 19
    assert lc.fp8_bytes == 33 * 588 and lc.fp8_bytes_padded == 64 * 608
    assert lc.pad_waste_elems == pytest.approx(1 - 33 * 588 / (64 * 608))
    assert lc.pad_waste_n == pytest.approx(31 / 64) and lc.pad_waste_k == pytest.approx(20 / 608)


# ---- DMA / MXU formulas against hw_params ------------------------------------------------------------
def test_dma_cycles_use_the_measured_4k_rate():
    lc = hc.layer_cost("model.layers.0.self_attn.q_proj", 2048, 2048)
    assert lc.tiles == 64 * 64
    assert lc.dma_cycles == (2048 * 2048 // hp.DMA_MAX_TRANSFER_BYTES) * hp.DMA_LOAD_CYCLES[4 * hp.KIB]
    assert lc.dma_cycles == hp.dma_load_cycles(2048 * 2048)
    # 4 KiB at 3056 cycles: 764 cycles per 1 KiB tile, 1.34 B/cycle
    assert lc.dma_cycles / lc.tiles == hp.DMA_LOAD_CYCLES[4 * hp.KIB] / (4 * hp.KIB // hp.WEIGHT_SLOT_BYTES)
    assert hp.DMA_LOAD_BYTES_PER_CYCLE == pytest.approx(hp.DMA_MAX_TRANSFER_BYTES / hp.DMA_LOAD_CYCLES[4 * hp.KIB])


def test_strided_dma_is_32_byte_transfers_per_tile_row():
    lc = hc.layer_cost("m.k_proj", 256, 2048)
    assert lc.dma_cycles_strided == lc.tiles * hp.TILE * hp.DMA_LOAD_CYCLES[32]
    assert lc.dma_cycles_strided > lc.dma_cycles


@pytest.mark.parametrize("mxu", ["ipt", "sa"])
def test_mxu_cycles_one_block_and_T(mxu):
    tiles = 4096
    lat = hp.MXU_MATMUL_LATENCY[mxu]
    assert hc.mxu_cycles(tiles, 32, mxu) == tiles * hp.MXU_ISSUE_GAP + lat
    assert hc.mxu_cycles(tiles, 64, mxu) == 2 * tiles * hp.MXU_ISSUE_GAP + lat
    assert hc.mxu_cycles(tiles, 33, mxu) == hc.mxu_cycles(tiles, 64, mxu)       # blocks of 32 rows
    # the sequencer streams tileRows = 32 rows whatever the token count: T=1 costs a full block
    assert hc.mxu_cycles(tiles, 1, mxu) == hc.mxu_cycles(tiles, 32, mxu)
    assert hc.mxu_cycles(tiles, 2048, mxu) == tiles * 64 * hp.MXU_ISSUE_GAP + lat


def test_sa_latency_exceeds_ipt_by_the_hw_params_gap():
    assert hc.mxu_cycles(10, 32, "sa") - hc.mxu_cycles(10, 32, "ipt") == \
        hp.MXU_MATMUL_LATENCY["sa"] - hp.MXU_MATMUL_LATENCY["ipt"]


# ---- reuse threshold ---------------------------------------------------------------------------------
@pytest.mark.parametrize("ty", sorted(TL_SHAPES))
def test_reuse_threshold_matches_definition(ty):
    N, K = TL_SHAPES[ty]
    lc = hc.layer_cost(f"model.layers.0.x.{ty}", N, K)
    b = lc.compute_bound_blocks
    assert hc.mxu_cycles(lc.tiles, hp.TILE * b) >= lc.dma_cycles
    assert hc.mxu_cycles(lc.tiles, hp.TILE * (b - 1)) < lc.dma_cycles
    assert lc.compute_bound_tokens_min == (b - 1) * hp.TILE + 1
    # DMA per tile (764) over 32 MXU cycles per block-tile: identical for every layer type
    per_tile = hp.DMA_LOAD_CYCLES[4 * hp.KIB] / (4 * hp.KIB // hp.WEIGHT_SLOT_BYTES)
    assert lc.reuse_blocks_continuous == pytest.approx(per_tile / hp.MXU_ISSUE_GAP)
    assert b == math.ceil(lc.reuse_blocks_continuous)


def test_reuse_threshold_is_layer_type_independent():
    bs = {hc.layer_cost(f"x.{t}", *s).compute_bound_blocks for t, s in TL_SHAPES.items()}
    assert len(bs) == 1 and bs == {math.ceil(hp.DMA_LOAD_CYCLES[4 * hp.KIB] / 4 / hp.MXU_ISSUE_GAP)}


def test_threshold_with_output_stores_depends_on_k():
    store = hp.DMA_STORE_CYCLES[4 * hp.KIB] / 2                    # one 2 KiB BF16 output tile
    load = hp.DMA_LOAD_CYCLES[4 * hp.KIB] / 4
    for K in (2048, 5632):
        kt = K // 32
        b = hc.compute_bound_blocks_with_stores(K)
        assert b == pytest.approx(kt * load / (kt * hp.MXU_ISSUE_GAP - store))
        assert b * kt * hp.MXU_ISSUE_GAP == pytest.approx(kt * load + b * store)     # DMA == MXU at b
        assert b > hc.layer_cost("x", 2048, K).compute_bound_blocks
    assert hc.compute_bound_blocks_with_stores(2048) > hc.compute_bound_blocks_with_stores(5632)
    assert hc.compute_bound_blocks_with_stores(32) == math.inf     # stores alone exceed one tile of MXU time


def test_vmem_activation_cap():
    assert hc.vmem_activation_token_cap(2048) == hp.VMEM_BYTES // 2048 == 768
    assert hc.vmem_activation_token_cap(5632) == 279


def test_mxu_over_dma_below_one_at_decode_and_above_at_large_T():
    lc = hc.layer_cost("x.q_proj", 2048, 2048, T=1)
    assert lc.mxu_over_dma_T < 0.05
    big = hc.layer_cost("x.q_proj", 2048, 2048, T=2048)
    assert big.mxu_over_dma_T > 1


def test_vmem_fit():
    assert hc.layer_cost("x.k_proj", 256, 2048).fits_vmem            # 512 KiB < 1.5 MiB
    assert not hc.layer_cost("x.q_proj", 2048, 2048).fits_vmem       # 4 MiB
    assert hc.layer_cost("x.q_proj", 2048, 2048).vmem_fraction == pytest.approx(4 * hp.MIB / hp.VMEM_BYTES)


# ---- flush stats -------------------------------------------------------------------------------------
def test_flush_stats_counts_zero_subnormal_reserved():
    hist = [0] * 256
    hist[0x00], hist[0x80], hist[0x03], hist[0x7F], hist[0xFF], hist[0x38] = 5, 1, 2, 3, 4, 85
    s = hc.flush_stats(hist)
    assert s["elements"] == 100
    assert s["zero_codes"] == 6 and s["zero_code_frac_upper_bound_of_flushed"] == pytest.approx(0.06)
    assert s["subnormal_codes"] == 2 and s["reserved_codes"] == 7


# ---- aggregate ---------------------------------------------------------------------------------------
def tinyllama_layers():
    n = hc.TINYLLAMA["num_hidden_layers"]
    return [hc.layer_cost(f"model.layers.{i}.x.{t}", *s) for i in range(n) for t, s in TL_SHAPES.items()]


def test_aggregate_tinyllama_totals():
    a = hc.aggregate(tinyllama_layers())
    assert a["layers"] == 154
    assert a["fp8_mib"] == pytest.approx(924.0)
    assert a["tiles"] == 22 * (4096 * 2 + 512 * 2 + 11264 * 3)
    assert a["weight_pass_dma_cycles"] == a["tiles"] * hp.DMA_LOAD_CYCLES[4 * hp.KIB] / 4
    assert a["weight_pass_seconds"] == pytest.approx(a["weight_pass_dma_cycles"] / hp.CLOCK_HZ)
    assert a["per_T"]["1"]["bound"] == "dma" and a["per_T"]["2048"]["bound"] == "compute"
    assert {p["compute_bound_blocks"] for p in a["per_type"].values()} == {24}      # 23.875 rounded up


# ---- pop scale: graph classification -----------------------------------------------------------------
@pytest.fixture(scope="module")
def graph():
    pytest.importorskip("transformers")
    return hc.trace_layer_graph()


def test_graph_has_the_seven_linears_with_real_shapes(graph):
    got = {hc.layer_type(k): v["out_shape"][-1] for k, v in graph["linears"].items()}
    assert got == {t: s[0] for t, s in TL_SHAPES.items()}


def test_consumer_classification_tinyllama(graph):
    cls = hc.classify_linears(graph)
    assert cls["q_proj"]["consumer_class"] == "vpu_op"            # RoPE on the VPU
    assert cls["k_proj"]["consumer_class"] == "vpu_op"
    assert cls["v_proj"]["consumer_class"] == "fp8_matmul_operand"
    assert cls["gate_proj"]["consumer_class"] == "vpu_op"         # SiLU
    assert cls["up_proj"]["consumer_class"] == "vpu_op"           # SiLU(gate) * up in BF16 before down_proj
    assert cls["o_proj"]["consumer_class"] == "residual_add"
    assert cls["down_proj"]["consumer_class"] == "residual_add"
    assert cls["attn.qk"]["consumer_class"] == "vpu_op"           # scale, softmax
    assert cls["attn.pv"]["consumer_class"] == "fp8_matmul_operand"   # -> o_proj
    assert [t[0] for t in cls["up_proj"]["terminals"]] == ["mul"]
    assert {t[0] for t in cls["q_proj"]["terminals"]} == {"mul", "neg", "cat"}
    assert cls["attn.pv"]["terminals"] == [("linear", 0)]


def test_only_v_proj_and_pv_feed_fp8_directly(graph):
    cls = hc.classify_linears(graph)
    direct = sorted(k for k, v in cls.items() if v["consumer_class"] == "fp8_matmul_operand")
    assert direct == ["attn.pv", "v_proj"]


def test_v_contracts_over_tokens_needs_a_transpose(graph):
    cls = hc.classify_linears(graph)
    assert cls["v_proj"]["contracts_over_tokens"] == [True]
    # layout only between v_proj and the matmul
    assert set(cls["v_proj"]["via_layout"][0].split("/")) <= hc.LAYOUT_OPS


def test_consumer_class_precedence():
    mm, vp, ra = dict(op="matmul"), dict(op="mul"), dict(op="residual_add")
    assert hc.consumer_class([mm]) == "fp8_matmul_operand"
    assert hc.consumer_class([mm, vp]) == "vpu_op"
    assert hc.consumer_class([ra]) == "residual_add"
    assert hc.consumer_class([]) == "unconsumed"


# ---- pop scale: rescale plan -------------------------------------------------------------------------
def test_pop_scale_free_only_when_all_conditions_hold():
    free = hc.rescale_plan("fp8_matmul_operand", "static", True, consumer_scale_static=True)
    assert free["pop_scale_free"] and free["extra_cycles_per_tile"] == 0 and free["layer_extra_cycles"] == 0
    for kw in (dict(consumer_cls="vpu_op"), dict(consumer_cls="residual_add"), dict(act_scale="per_token"),
               dict(act_scale="per_head"), dict(w=False), dict(cs=False)):
        a = dict(consumer_cls="fp8_matmul_operand", act_scale="static", w=True, cs=True)
        a.update(kw)
        p = hc.rescale_plan(a["consumer_cls"], a["act_scale"], a["w"], consumer_scale_static=a["cs"])
        assert not p["pop_scale_free"], kw
        assert p["extra_cycles_per_tile"] > 0 and p["why_not_free"]


def test_vmul_fallback_costs_from_hw_params():
    t = hp.VPU_TILE_ELEMENTWISE_CYCLES
    assert hc.rescale_plan("vpu_op", "static", True)["extra_cycles_per_tile"] == t            # one vmul
    assert hc.rescale_plan("vpu_op", "per_token", True)["extra_cycles_per_tile"] == 2 * t     # row x column
    assert hc.rescale_plan("residual_add", "per_head", False)["extra_cycles_per_tile"] == t
    fp8 = hc.rescale_plan("fp8_matmul_operand", "per_token", False, consumer_scale_static=False)
    assert fp8["extra_cycles_per_tile"] == 3 * t and "vpack" in fp8["mechanism"]


def test_layer_extra_cycles_scale_with_output_tiles():
    p32 = hc.rescale_plan("vpu_op", "per_token", True, tokens=32, N=5632)
    assert p32["layer_extra_cycles"] == (5632 // 32) * p32["extra_cycles_per_tile"]
    p100 = hc.rescale_plan("vpu_op", "per_token", True, tokens=100, N=5632)
    assert p100["layer_extra_cycles"] == 4 * p32["layer_extra_cycles"]                    # ceil(100/32) = 4 blocks
    p1 = hc.rescale_plan("vpu_op", "per_token", True, tokens=1, N=5632)
    assert p1["layer_extra_cycles"] == p32["layer_extra_cycles"]


# ---- export (skipped when the export is absent) ------------------------------------------------------
@pytest.mark.skipif(not Path(EXPORT).exists(), reason="TinyLlama export not present")
def test_export_stores_n_by_k_row_major():
    import json
    from safetensors import safe_open
    policy = json.loads((Path(EXPORT).parent / "policy.json").read_text())
    with safe_open(EXPORT, "numpy") as f:
        for site in ("model.layers.0.self_attn.v_proj", "model.layers.5.mlp.down_proj"):
            w = policy["sites"][site]["weight"]
            sl = f.get_slice(w["codes"])
            assert sl.get_dtype() == "U8"
            assert sl.get_shape() == policy["sites"][site]["weight_shape"]            # [N, K]
            assert f.get_slice(w["scale_exp"]).get_shape() == [sl.get_shape()[0], 1]  # one exponent per output row
            codes = f.get_tensor(w["codes"])
            frac = hc.flush_stats(__import__("numpy").bincount(codes.ravel(), minlength=256))
            assert frac["subnormal_codes"] == 0 and frac["reserved_codes"] == 0
