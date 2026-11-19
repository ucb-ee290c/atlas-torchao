"""The chip profile end to end on a small random Llama: every op classified, nothing unmapped."""
import json

import pytest
import torch

import chip.chip_coverage as cc
import quant_modes
from refutil import need_refs


def tiny_llama(layers=2):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      attn_implementation="eager", use_cache=False)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()


@pytest.mark.parametrize("mxu", ["framework", "model"])
def test_tiny_llama_chip(mxu):
    need_refs()
    model = tiny_llama()
    ids = torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        ref = model(ids, labels=ids).loss.item()
    rep = quant_modes.quantize(model, "chip", skip=["lm_head"],
                               options={"chip_mxu": mxu, "mxu": "ipt", "mxu_impl": "ref_cpu"})
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    assert torch.isfinite(torch.tensor(loss))
    assert abs(loss - ref) < 0.5
    cov = rep["coverage"]                       # updated in place by the router
    json.dumps(cov)
    u = cov["units"]
    assert u["MXU"]["work"] > 0 and u["VPU"]["work"] > 0
    host = {h["op"]: h["reason"] for h in cov["host_fallback"]}
    assert not any("unmapped" in r for r in host.values()), host
    # lm_head was skipped on request: it is host, with the reason
    assert any("--skip lm_head" in r for r in host.values())
    ops = {(o["unit"], o["op"]) for o in cov["ops"]}
    for need in [("VPU", "_softmax"), ("VPU", "silu"), ("VPU", "rsqrt"), ("VPU", "mean"),
                 ("MXU", "linear" if mxu == "model" else "mm"), ("MXU", "attention"), ("eval", "_log_softmax")]:
        assert need in ops, need
    assert cov["silent_bf16_ops"] == 0
    # attention runs through the reference per-head routine (ref_attention), counted as "attention"
    assert ("MXU", "bmm") not in ops


def test_reference_target_ops_all_handled():
    need_refs()
    assert cc.unmapped_reference_ops() == []


def test_chip_rope_host_marks_tables():
    need_refs()
    model = tiny_llama(1)
    ids = torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1))
    rep = quant_modes.quantize(model, "chip", options={"chip_mxu": "framework", "chip_rope": "host"})
    with torch.no_grad():
        model(ids)
    reasons = [h["reason"] for h in rep["coverage"]["host_fallback"]]
    assert any("RoPE table" in r for r in reasons)


def test_dynamic_scale_chain_rows_are_additive_and_new():
    """NEW (hw_scale): the real dynamic-scale chain is counted next to, not instead of, the
    vredmax/vredmin/vpack count of the cast; a=token and a=block add rows, the default adds none."""
    need_refs()
    import algos

    def run(spec):
        model = tiny_llama()
        rep = algos.quantize(model, "chip", ["lm_head"], {"chip_mxu": "framework"}, algo=algos.make(spec))
        model(torch.randint(0, 128, (1, 32), generator=torch.Generator().manual_seed(0)))
        return rep["coverage"]

    base = run("granularity")
    assert not any(o["op"].startswith("dyn_scale_chain") for o in base["ops"])
    assert "dynamic_scale_chain_NEW" not in base
    cov = run("granularity:a=token,rule=nc")
    ops = {o["op"]: o for o in cov["ops"]}
    cast = ops["fp8_operand_cast"]["instructions"]
    assert cast["vredmax.row.bf16"] == cast["vredmin.row.bf16"] == cast["vpack.bf16.fp8"] > 0   # old counts kept
    chain = ops["dyn_scale_chain"]
    assert chain["unit"] == "VPU" and chain["reason"].startswith("NEW")
    assert chain["instructions"]["vstore"] > 0 and ops["dyn_scale_chain_scalar"]["instructions"]["rv32i"] > 0
    tot = cov["dynamic_scale_chain_NEW"]["token"]
    assert tot["serialized_cycles"] > 0 and tot["seld"] == 0
    assert tot["instructions"] == sum(tot["vpu"].values()) + tot["rv32i"] + tot["seld"]
    json.dumps(cov)
