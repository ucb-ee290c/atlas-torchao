"""--protocol mxq: microscaling-quant experiments/llm_ppl.py behaviour, on tiny inputs."""

import math
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ppl_protocols import mxq_eval, mxq_select, mxq_text, sample_nll  # noqa: E402


def test_empty_lines_dropped():
    # llm_ppl.py:48
    assert mxq_text(["a\n", "", "  \n", " = H = \n", "b"]) == "a\n\n\n = H = \n\n\nb"


def test_select_full_split_and_truncation():
    ids = torch.arange(10)
    w, idx = mxq_select(ids, 3, 0, 0)
    assert w.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]] and idx == [0, 1, 2]
    w, idx = mxq_select(ids, 3, 5, 0)  # nsamples >= nbatch is the whole split (:52-53)
    assert idx == [0, 1, 2]


def test_select_seeded_matches_reference_rule():
    ids = torch.arange(100)
    w, idx = mxq_select(ids, 5, 4, seed=7)
    torch.manual_seed(7)
    want = torch.randperm(20)[:4].tolist()  # llm_ppl.py:56-57
    assert idx == want
    assert w.tolist() == [list(range(5 * i, 5 * i + 5)) for i in want]
    w, idx = mxq_select(ids, 5, 4, seed=None)  # --sequential (:54-55)
    assert idx == [0, 1, 2, 3]


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=50, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=64)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


def test_token_summed_nll_matches_hf_mean_loss():
    model = _tiny_llama()
    ids = torch.randint(0, 50, (12,))
    nll, n = sample_nll(model, ids)
    assert n == 11  # seqlen-1 predicted tokens (llm_ppl.py:64-66)
    with torch.no_grad():
        loss = model(ids[None], labels=ids[None], use_cache=False).loss
    assert nll / n == pytest.approx(float(loss), rel=1e-5)


def test_ppl_is_token_weighted_over_windows():
    model = _tiny_llama()
    windows = torch.randint(0, 50, (3, 9))
    seen = []
    r = mxq_eval(model, windows, before_window=lambda w: seen.append(w.shape))
    assert seen == [torch.Size([1, 9])] * 3
    total = sum(s["nll"] for s in r["per_sample"])
    assert r["tokens"] == 24
    assert r["ppl"] == pytest.approx(math.exp(total / 24))
    with torch.no_grad():
        logits = model(windows, use_cache=False).logits[:, :-1].float()
    ref = F.cross_entropy(logits.reshape(-1, 50), windows[:, 1:].reshape(-1), reduction="sum")
    assert total == pytest.approx(float(ref), rel=1e-5)


def test_kl_meter_on_mxq_windows():
    """--kl under --protocol mxq: the meter sees every window's own forward (all seqlen positions),
    and the perplexity is the same with or without it."""
    from eval import KLMeter

    model = _tiny_llama()
    windows = torch.randint(0, 50, (3, 9), generator=torch.Generator().manual_seed(4))
    plain = mxq_eval(model, windows)
    k = KLMeter(model)
    r = mxq_eval(model, windows, after_window=k)
    assert r["ppl"] == plain["ppl"] and r["per_sample"] == plain["per_sample"]
    s = k.summary()
    assert s["tokens"] == 27 and abs(s["kl_vs_bf16"]) < 1e-6 and s["top1_agree_vs_bf16"] == 1.0
    other = _tiny_llama()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.05 * torch.randn_like(p))
    k2 = KLMeter(model)
    mxq_eval(other, windows, after_window=k2)
    assert k2.summary()["kl_vs_bf16"] > 0 and k2.summary()["tokens"] == 27
