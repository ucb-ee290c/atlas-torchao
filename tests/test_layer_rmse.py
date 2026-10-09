"""--rmse: the reference StatsTracker/ReferenceStore wiring and the run_eval.py CSV schema.

Needs the reference author's smolVLA-quant (or pi0-quant) checkout: third_party/<repo>, or
SMOLVLA_QUANT_ROOT / PI0_QUANT_ROOT / ATLAS_REF_ROOT. Skipped when none is present.
"""

import copy
import csv
import math
import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import layer_rmse  # noqa: E402

pytestmark = pytest.mark.skipif(
    layer_rmse.ref_root("smolVLA-quant") is None and layer_rmse.ref_root("pi0-quant") is None,
    reason="reference repositories not checked out")


class Noisy(nn.Module):
    """A stand-in quantized linear: the original layer plus a fixed output offset."""

    def __init__(self, lin, delta):
        super().__init__()
        self.lin, self.delta = lin, delta

    def forward(self, x):
        return self.lin(x) + self.delta


def _pair():
    torch.manual_seed(0)
    clean = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))
    quant = copy.deepcopy(clean)
    quant[0] = Noisy(quant[0], 0.25)
    quant[2] = Noisy(quant[2], 0.0)  # exact itself, but sees the corrupted input of layer 0
    return clean, quant


def test_quantized_fqns_finds_replaced_layers():
    clean, quant = _pair()
    assert layer_rmse.quantized_fqns(clean, quant) == ["0", "2"]
    assert layer_rmse.quantized_fqns(clean, copy.deepcopy(clean)) == []


def test_local_and_cumulative_rmse_by_hand():
    clean, quant = _pair()
    x = torch.randn(3, 8)
    rm = layer_rmse.LayerRMSE(clean, quant)
    rm.capture(lambda: clean(x))
    quant(x)
    rm.close()
    calls = {c["name"]: c for c in rm.tracker.calls}
    # layer 0: local == cumulative == the injected offset
    assert calls["0"]["rmse"] == pytest.approx(0.25, rel=1e-6)
    assert calls["0"]["cumulative_rmse"] == pytest.approx(0.25, rel=1e-6)
    # layer 2: no local error, but the propagated error from layer 0 shows up as cumulative
    assert calls["2"]["rmse"] == 0.0
    h_q = torch.relu(clean[0](x) + 0.25)
    y_clean = clean(x)
    want = math.sqrt(float((clean[2](h_q) - y_clean).pow(2).mean()))
    assert calls["2"]["cumulative_rmse"] == pytest.approx(want, rel=1e-5)
    assert calls["2"]["cumulative_ref_rms"] == pytest.approx(math.sqrt(float(y_clean.pow(2).mean())), rel=1e-5)


def test_call_order_pairing_over_repeated_calls():
    """A layer called several times per input is paired call by call (reference_store.py:66-78)."""
    clean = nn.Linear(4, 4)
    quant = Noisy(copy.deepcopy(clean), 0.0)
    holder_c, holder_q = nn.ModuleDict({"l": clean}), nn.ModuleDict({"l": quant})
    xs = [torch.randn(2, 4) for _ in range(3)]
    rm = layer_rmse.LayerRMSE(holder_c, holder_q)
    rm.capture(lambda: [holder_c["l"](x) for x in xs])
    for x in xs:
        holder_q["l"](x)
    rm.close()
    assert [c["cumulative_rmse"] for c in rm.tracker.calls] == [0.0, 0.0, 0.0]


def test_csv_schema_matches_run_eval(tmp_path):
    clean, quant = _pair()
    rm = layer_rmse.LayerRMSE(clean, quant)
    for _ in range(2):
        x = torch.randn(3, 8)
        rm.capture(lambda: clean(x))
        quant(x)
    rm.close()
    paths = rm.write_csvs(tmp_path)
    assert set(paths) == {"chronological", "grouped", "summary", "worst_layers"}

    def read(name):
        with open(paths[name]) as f:
            r = csv.DictReader(f)
            return r.fieldnames, list(r)

    # field lists of smolVLA-quant experiments/run_eval.py:191-203
    fields, rows = read("chronological")
    assert fields == ["seq", "tag", "layer_name", "component", "rmse", "ref_rms", "rel_rmse",
                      "cumulative_rmse", "cumulative_rel_rmse"]
    assert [int(r["seq"]) for r in rows] == [0, 1, 2, 3]
    fields, rows = read("grouped")
    assert [r["layer_name"] for r in rows] == ["0", "0", "2", "2"]
    fields, rows = read("summary")
    assert fields[:3] == ["tag", "component", "n_layers"] and fields[-2:] == [
        "mean_cumulative_rmse", "mean_cumulative_rel_rmse"]
    assert rows[0]["tag"] == "mx" and rows[0]["n_layers"] == "2"
    fields, rows = read("worst_layers")
    assert fields == ["rank", "tag", "component", "layer_name", "rel_rmse", "rmse", "n_calls"]
    assert rows[0]["layer_name"] == "0"  # layer 2 has zero local error


def test_component_rules():
    C = layer_rmse.load_ref_file("stats_tracker").Component
    assert layer_rmse.component_for("model.vlm_with_expert.lm_expert.layers.0.mlp.up_proj") is C.ACTION_EXPERT
    assert layer_rmse.component_for("model.vlm_with_expert.vlm.model.vision_model.x") is C.VISION
    assert layer_rmse.component_for("model.action_out_proj") is C.ACTION_HEAD
    assert layer_rmse.component_for("noise_predictor.transformer_blocks.0.mlp.0") is C.ACTION_EXPERT
    assert layer_rmse.component_for("model.layers.0.mlp.up_proj") is C.UNKNOWN
    assert layer_rmse.component_for("model.layers.0.mlp.up_proj", "LANGUAGE") is C.LANGUAGE


def test_provenance_records_loaded_files():
    layer_rmse.load_ref_file("stats_tracker")
    prov = layer_rmse.ref_provenance()
    files = {f for rec in prov.values() for f in rec["files"]}
    assert any(f.endswith("stats_tracker.py") for f in files)
    assert all(len(h) == 64 for rec in prov.values() for h in rec["files"].values())


def test_with_atlas_quantized_tiny_llama():
    pytest.importorskip("torchao")
    from transformers import LlamaConfig, LlamaForCausalLM

    from quant_modes import quantize

    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=32)
    torch.manual_seed(0)
    clean = LlamaForCausalLM(cfg).to(torch.bfloat16).eval()
    quant = copy.deepcopy(clean)
    quantize(quant, "atlas", skip=["lm_head"])
    rm = layer_rmse.LayerRMSE(clean, quant, default_component="LANGUAGE")
    assert len(rm.fqns) == 2 * 7
    ids = torch.randint(0, 64, (1, 16))
    rm.capture(lambda: clean(ids, use_cache=False))
    with torch.no_grad():
        quant(ids, use_cache=False)
    rm.close()
    rows = rm.tracker.component_rows()
    assert rows[0]["component"] == "language" and rows[0]["n_layers"] == 14
    assert 0 < rows[0]["mean_rel_rmse"] < 0.5
    assert rows[0]["mean_cumulative_rel_rmse"] >= rows[0]["mean_rel_rmse"] * 0.5
