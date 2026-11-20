"""DroNet cost model: the counts are exact (hand count == torch FlopCounterMode == shape hooks); the
cycle model is only checked for internal consistency (it is an estimate, not a measurement)."""
import pytest
import torch
import torch.nn as nn

import dronet.cost as dc
from dronet.model import DroNet

# hand count: M = output pixels, K = Cin*kh*kw, N = Cout, MACs = M*K*N
HAND = {
    "conv1": 100 * 100 * 25 * 32,
    "block1.conv_a": 25 * 25 * 288 * 32, "block1.conv_b": 25 * 25 * 288 * 32, "block1.skip": 25 * 25 * 32 * 32,
    "block2.conv_a": 13 * 13 * 288 * 64, "block2.conv_b": 13 * 13 * 576 * 64, "block2.skip": 13 * 13 * 32 * 64,
    "block3.conv_a": 7 * 7 * 576 * 128, "block3.conv_b": 7 * 7 * 1152 * 128, "block3.skip": 7 * 7 * 64 * 128,
    "steer": 6272, "collision": 6272,
}
TOTAL = 41_103_104


def test_hand_count_per_layer_and_total():
    rows = {r["name"]: r for r in dc.layer_table()}
    assert set(rows) == set(HAND)
    for name, macs in HAND.items():
        assert rows[name]["macs"] == macs, name
    assert sum(HAND.values()) == TOTAL == dc.hand_macs() == dc.totals()["macs"]
    assert dc.totals()["flops"] == 2 * TOTAL


def test_matches_flop_counter():
    assert dc.flopcounter_macs() == TOTAL


def test_shapes_and_params_match_torch_model():
    m = DroNet().eval()
    seen = {}
    named = {m.conv1: "conv1", m.steer: "steer", m.collision: "collision"}
    for i, b in enumerate((m.block1, m.block2, m.block3), 1):
        named.update({b.conv_a: f"block{i}.conv_a", b.conv_b: f"block{i}.conv_b", b.skip: f"block{i}.skip"})

    def hook(mod, inp, out):
        seen[named[mod]] = (tuple(inp[0].shape), tuple(out.shape))

    for mod in named:
        mod.register_forward_hook(hook)
    with torch.no_grad():
        m(torch.zeros(1, 1, 200, 200))
    rows = {r["name"]: r for r in dc.layer_table()}
    for name, (ishape, oshape) in seen.items():
        r = rows[name]
        assert r["out_elems"] == oshape[1] * (oshape[2] * oshape[3] if len(oshape) == 4 else 1), name
        assert r["in_elems"] == (ishape[1] * ishape[2] * ishape[3] if len(ishape) == 4 else ishape[1]), name
        mod = next(k for k, v in named.items() if v == name)
        assert r["w_params"] == mod.weight.numel(), name
        assert r["bias_params"] == mod.bias.numel(), name
    assert dc.totals()["w_params"] == sum(mod.weight.numel() for mod in named)


def test_fp8_share_and_tile_counts():
    t = dc.totals()
    assert t["mac_share_fp8"] == 1.0
    assert t["w_fp8_bytes"] == t["w_params"] and t["w_bf16_bytes"] == 2 * t["w_params"]
    rows = {r["name"]: r for r in dc.layer_table()}
    assert rows["conv1"]["Kt"] == 1 and rows["conv1"]["K"] == 25              # K = 25 padded to 32
    assert rows["steer"]["Kt"] == 196
    assert t["matmuls"] == 1869 and t["matmuls_fused_heads"] == 1673
    # padding can only add work
    assert all(r["macs"] <= r["matmuls"] * 32 ** 3 for r in rows.values())


def test_nonconv_table_has_the_host_and_data_ops():
    ops = {o["op"]: o for o in dc.nonconv_table()}
    assert ops["maxpool3x3/2"]["unit"] == "host" and ops["maxpool3x3/2"]["elems"] == 32 * 49 * 49
    assert ops["pad(1,2,1,2)"]["unit"] == "data" and ops["pad(1,2,1,2)"]["elems"] == 203 * 203
    assert sum(1 for o in dc.nonconv_table() if o["op"] == "residual add") == 3


def test_memory_summary_numbers():
    m = dc.memory_summary()
    assert m["peak_stage"] == "conv1" and m["peak_activation_bytes"] == 41_209 + 320_000 + 640_000
    assert m["conv_weights_fp8_padded"] == 307_200
    assert m["stages_over_window_with_conv_weights"] == ["conv1", "maxpool"]
    assert m["stages_over_full_with_conv_weights"] == []


def test_scenarios_are_ordered():
    est = dc.estimate()
    assert est["high"]["cycles_throughput"] < est["nominal"]["cycles_throughput"] < est["low"]["cycles_throughput"]
    for e in est.values():
        fps = e["fps"]
        assert fps["500 MHz (design / RTL timing model)"]["throughput"] > fps["100 MHz (backend-kit chip bench)"]["throughput"]
        # the MXU alone is never the whole story: even `high` is above the 2-MXU MAC roofline
        assert e["cycles_throughput"] > dc.totals()["macs"] / (2 * 1024)


@pytest.mark.parametrize("key", ["high", "nominal", "low"])
def test_layer_rows_complete(key):
    e = dc.frame_estimate(dc.SCENARIOS[key])
    assert len(e["layers"]) == (11 if dc.SCENARIOS[key].fuse_heads else 12)
    assert all(l["cycles"] > 0 and l["bound"] in l["resources"] for l in e["layers"])
