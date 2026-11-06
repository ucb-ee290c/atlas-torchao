"""WikiText-2 perplexity protocols for eval.py.

  default  the collaborator's protocol, unchanged (datautils.get_wikitext2 + eval.llama_eval):
           slow tokenizer over "\\n\\n".join of every line (empty ones included), consecutive
           non-overlapping windows, perplexity = exp(mean of per-window mean losses).
  mxq      microscaling-quant experiments/llm_ppl.py (@47f6ba3), which follows MXQuant's
           eval_complete.py so its numbers are comparable to the published ones:
             - fast tokenizer (AutoTokenizer default; llm_ppl.py:46)
             - empty lines dropped before joining (:48)
             - the token stream cut into nbatch = N // seqlen windows (:49-51)
             - nsamples = 0 means the whole split, the number to quote (:40-43, :52-53);
               otherwise `nsamples` windows chosen by torch.randperm under `seed` (:54-57)
             - per window, NLL from float32 log_softmax summed over its seqlen-1 predicted
               tokens (:60-66); perplexity = exp(total NLL / total tokens) (:182-185)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

PROTOCOLS = ("default", "mxq")


def mxq_text(lines) -> str:
    """llm_ppl.py:48: drop empty (whitespace-only) lines, join the rest with blank lines."""
    return "\n\n".join([x for x in lines if x.strip()])


def mxq_select(ids: torch.Tensor, seqlen: int, nsamples: int, seed) -> tuple[torch.Tensor, list[int]]:
    """llm_ppl.py:49-57 on a 1-D token stream. Returns ([n, seqlen] windows, their indices).

    The indices are NEW (the reference returns only the windows); they are recorded so a run
    states which windows it averaged.
    """
    ids = ids.reshape(-1)
    nbatch = ids.numel() // seqlen
    ids = ids[:nbatch * seqlen].reshape(nbatch, seqlen)
    if nsamples == 0 or nsamples >= nbatch:
        return ids, list(range(nbatch))
    if seed is None:
        return ids[:nsamples], list(range(nsamples))
    torch.manual_seed(seed)  # :56, global RNG as in the reference
    idx = torch.randperm(nbatch)[:nsamples]
    return ids[idx], idx.tolist()


def load_mxq_windows(model_id: str, seqlen: int, nsamples: int, seed):
    """llm_ppl.py:39-57: WikiText-2 test split, fast tokenizer, empty lines dropped."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)  # :46 (use_fast default)
    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")  # :47
    ids = tokenizer(mxq_text(data["text"]), return_tensors="pt")["input_ids"][0]  # :48-49
    return mxq_select(ids, seqlen, nsamples, seed)


@torch.no_grad()
def sample_nll(model, ids: torch.Tensor, after_window=None) -> tuple[float, int]:
    """llm_ppl.py:60-66: summed next-token NLL of one window and the number of tokens scored.

    `after_window(window, output)` (NEW, --kl) sees the [1, seqlen] window and the model output of
    this same forward; it does not change the NLL."""
    device = next(model.parameters()).device
    ids = ids.unsqueeze(0).to(device)
    out = model(ids, use_cache=False)
    logits = out.logits[0, :-1, :]
    nll = -F.log_softmax(logits.float(), dim=-1).gather(1, ids[0, 1:].unsqueeze(1)).squeeze(1)
    if after_window is not None:
        after_window(ids, out)
    return nll.sum().item(), nll.numel()


def mxq_eval(model, windows: torch.Tensor, before_window=None, after_window=None) -> dict:
    """Perplexity over `windows` ([n, seqlen]) as llm_ppl.py:182-185 aggregates it.

    `before_window(window)` (NEW) runs before each window's forward; eval.py uses it for the clean
    pass of --rmse. `after_window(window, output)` (NEW) runs after it; eval.py uses it for --kl
    (eval.KLMeter), so the KL is over every position of the same windows the perplexity scores.
    """
    per_sample, total, tokens = [], 0.0, 0
    for i in range(windows.shape[0]):
        if before_window is not None:
            before_window(windows[i:i + 1])
        nll, n = sample_nll(model, windows[i], after_window=after_window)
        per_sample.append({"nll": nll, "tokens": n})
        total += nll
        tokens += n
    return {"ppl": math.exp(total / tokens), "nll": total, "tokens": tokens, "per_sample": per_sample}
