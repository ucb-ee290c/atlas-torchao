"""The coverage ladder: each step patches exactly the op classes it names, records them, and the
subset steps are compared against ref_conv on the same windows / frames."""
import json

import pytest
import torch
import torch.nn as nn

import quant_modes as qm
from refutil import need_refs

FAST = {"crosscheck": "off", "mxu": "ipt", "mxu_impl": "torch"}


def tiny_llama(layers=1):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      attn_implementation="eager", use_cache=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


def _types(model):
    return {type(m).__name__ for m in model.modules()}


def test_ladder_order_and_alias():
    assert qm.LADDER == ("bf16", "collab_v0", "ref_linear", "ref_attn", "ref_conv", "reference_hw",
                         "chip", "chip_rope_vpu")
    assert qm.canonical("reference") == "ref_conv"
    assert set(qm.LADDER) <= set(qm.ALL_MODES) and "reference" in qm.ALL_MODES
    for n in qm.LADDER[1:]:
        assert qm.PROFILES[n].adds


@pytest.mark.parametrize("mode,classes,equivalent", [
    ("ref_linear", {"linear"}, None),
    ("ref_attn", {"linear", "attention"}, None),
    ("ref_conv", {"linear", "attention", "conv2d"}, "ref_attn"),     # Llama has no Conv2d
    ("reference", {"linear", "attention", "conv2d"}, "ref_attn"),
])
def test_reference_steps_patch_their_op_classes(mode, classes, equivalent):
    need_refs()
    model = tiny_llama()
    rep = qm.quantize(model, mode, skip=["lm_head"])
    lad = rep["ladder"]
    json.dumps(lad)
    assert set(lad["quantized_op_classes"]) == classes
    assert lad["quantized_op_classes"]["linear"] == rep["quantized"] > 0
    assert ("RefQuantLinear" in _types(model))
    attn = [m for m in model.modules() if getattr(m, "_atlas_ref_attn", None) is not None]
    assert bool(attn) == ("attention" in classes)
    if "attention" in classes:
        assert lad["quantized_op_classes"]["attention"] == len(rep["attention_sites"]) > 0
    assert lad["accumulation"] == "framework" and lad["vector_ops"] == "framework"
    assert lad["on_ladder"] and lad.get("equivalent_to") == equivalent
    assert lad["profile"] == qm.canonical(mode) and lad["mode"] == mode


def test_ref_conv_adds_conv2d_only():
    need_refs()
    m_attn, m_conv = ConvNet(), ConvNet()
    a = qm.quantize(m_attn, "ref_attn")
    c = qm.quantize(m_conv, "ref_conv")
    assert "RefQuantConv2d" not in _types(m_attn) and "RefQuantConv2d" in _types(m_conv)
    assert c["ladder"]["quantized_op_classes"]["conv2d"] == 1 and "equivalent_to" not in c["ladder"]
    assert a["ladder"]["quantized_op_classes"].get("conv2d") is None
    # ConvNet has no attention: ref_attn adds nothing over ref_linear and says so
    assert a["ladder"]["equivalent_to"] == "ref_linear"


def test_explicit_ops_is_off_ladder():
    need_refs()
    rep = qm.quantize(tiny_llama(), "ref_conv", options={"ops": "linear"})
    assert set(rep["ladder"]["quantized_op_classes"]) == {"linear"}
    assert not rep["ladder"]["on_ladder"] and "--ops" in rep["ladder"]["off_ladder_reasons"][0]


def test_reference_hw_records_mxu():
    need_refs()
    rep = qm.quantize(tiny_llama(), "reference_hw", skip=["lm_head"], options=FAST)
    lad = rep["ladder"]
    assert lad["accumulation"] == "mxu ipt (torch)" and lad["previous"] == "ref_conv"
    assert set(lad["quantized_op_classes"]) == {"linear", "attention", "conv2d"} and lad["on_ladder"]


@pytest.mark.parametrize("mode,options,rope,on_ladder", [
    ("chip", {}, "host", True),                            # host tables are the default (NEW)
    ("chip_rope_vpu", {}, "vpu", True),
    ("chip_rope_vpu", {"chip_rope": "host"}, "vpu", True),  # the step fixes its own option
    ("chip", {"chip_rope": "vpu"}, "vpu", False),           # that is chip_rope_vpu, not chip
])
def test_chip_steps_rope(mode, options, rope, on_ladder):
    need_refs()
    model = tiny_llama()
    rep = qm.quantize(model, mode, skip=["lm_head"], options={**FAST, **options})
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        model(ids)
    lad = rep["ladder"]
    assert rep["profile"]["rope"] == rope == lad["rope_tables"]
    assert lad["vector_ops"].startswith("vpu") and lad["accumulation"] == "mxu ipt (torch)"
    assert lad["on_ladder"] is on_ladder
    host = [h["reason"] for h in rep["coverage"]["host_fallback"]]
    assert any("RoPE table" in r for r in host) == (rope == "host")


def test_chip_framework_smoke_is_off_ladder():
    need_refs()
    rep = qm.quantize(tiny_llama(), "chip", options={"chip_mxu": "framework"})
    assert not rep["ladder"]["on_ladder"]
    assert rep["ladder"]["accumulation"].startswith("framework")


def test_collab_and_bf16_records():
    lad = qm.ladder_record("bf16", {}, None)
    assert lad["step"] == 1 and lad["quantized_op_classes"] == {} and lad["on_ladder"]
    m = nn.Sequential(nn.Linear(16, 16)).to(torch.bfloat16)
    rep = qm.quantize(m, "atlas")
    assert rep["ladder"]["profile"] == "collab_v0" and rep["ladder"]["quantized_op_classes"] == {"linear": 1}
    assert rep["ladder"]["step"] == 2 and rep["ladder"]["previous"] == "bf16"


def test_subset_compare_modes():
    f = qm.subset_compare_modes
    assert f(["bf16", "ref_linear"]) == (["bf16", "ref_linear"], [])            # no subset step
    assert f(["reference_hw", "chip"]) == (["bf16", "ref_conv", "reference_hw", "chip"],
                                           ["bf16", "ref_conv"])
    assert f(["bf16", "reference", "chip_rope_vpu"]) == (["bf16", "reference", "chip_rope_vpu"], [])
    assert f(["ref_linear", "chip"], enabled=False) == (["ref_linear", "chip"], [])
    assert f(["bf16", "ref_attn", "chip"])[0] == ["bf16", "ref_attn", "ref_conv", "chip"]


def test_subset_compare_flag_in_clis():
    import eval_policy

    args = eval_policy.build_parser().parse_args(
        ["--policy", "p", "--dataset", "d", "--out", "o", "--modes", "chip_rope_vpu", "ref_linear"])
    assert args.subset_compare == "on" and args.ops is None and args.chip_rope is None
    args = eval_policy.build_parser().parse_args(
        ["--policy", "p", "--dataset", "d", "--out", "o", "--modes", "chip", "--subset-compare", "off"])
    assert args.subset_compare == "off"


def test_ladder_deltas_same_run_only():
    metrics = {"bf16": {"ppl": 7.0}, "ref_conv": {"ppl": 7.3}, "reference_hw": {"ppl": 7.5},
               "chip": {"ppl": 7.6}, "torchao": {"ppl": 9.0}}
    lad = {"chip": {"on_ladder": False}}
    d = qm.ladder_deltas(metrics, lad)
    assert [(x["from"], x["to"], x["steps"]) for x in d] == [("bf16", "ref_conv", 4),
                                                            ("ref_conv", "reference_hw", 1)]
    assert d[1]["delta"]["ppl"] == pytest.approx(0.2)


def test_eval_subset_compare_and_duplicate(monkeypatch, tmp_path):
    """eval.py on a tiny Llama: --max-windows makes a subset, so reference_hw brings bf16 and
    ref_conv onto the same windows; ref_conv adds nothing over ref_attn and is not re-run."""
    need_refs()
    import sys
    from types import SimpleNamespace

    import datautils
    import eval as ev
    import modelutils

    def load(path, seqlen=None, device="cpu"):
        m = tiny_llama()
        m.seqlen = seqlen
        return m

    ids = torch.randint(0, 128, (1, 64), generator=torch.Generator().manual_seed(2))
    monkeypatch.setattr(modelutils, "load_llama", load)
    monkeypatch.setattr(datautils, "get_wikitext2", lambda **k: (None, SimpleNamespace(input_ids=ids)))
    out = tmp_path / "r.json"
    monkeypatch.setattr(sys, "argv", ["eval.py", "--model", "x", "--seqlen", "16", "--max-windows", "2",
                                      "--modes", "ref_attn", "reference_hw", "--crosscheck", "off",
                                      "--device", "cpu", "--out", str(out)])
    ev.main()
    r = json.loads(out.read_text())
    assert list(r["ppl"]) == ["ref_attn", "bf16", "ref_conv", "reference_hw"]
    assert r["subset_compare"]["added_modes"] == ["bf16", "ref_conv"]
    assert r["ladder"]["ref_conv"]["duplicate_of"] == "ref_attn"
    assert r["ppl"]["ref_conv"] == r["ppl"]["ref_attn"]
    assert [(d["from"], d["to"]) for d in r["ladder_deltas"]] == [
        ("bf16", "ref_attn"), ("ref_attn", "ref_conv"), ("ref_conv", "reference_hw")]
    assert r["ladder"]["reference_hw"]["accumulation"].startswith("mxu ipt")
