from torch import nn

import torch 
import math

import torch
from tqdm import tqdm


@torch.no_grad()
def llama_eval(model, testenc, seqlen=None):
    """Perplexity over non-overlapping windows of `seqlen` tokens.
    testenc: tokenizer output (with .input_ids) or a [1, N] tensor of token ids."""
    ids = getattr(testenc, "input_ids", testenc)          # accept BatchEncoding or tensor
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    seqlen = seqlen or getattr(model, "seqlen", 512)     # default: the model's own seqlen
    dev = next(model.parameters()).device

    losses = []
    
    for i in tqdm(range(ids.shape[1] // seqlen)):
        window = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        losses.append(model(window, labels=window, use_cache=False).loss.float())

    return torch.exp(torch.stack(losses).mean()).item()


if __name__ == "__main__":
    from modelutils import load_llama, load_smolvla
    from torchao.quantization import quantize_
    from atlas_torchao import AtlasQConfig, skip_lm_head

    model_path = "/rscratch/agarcia/TinyLlama-1.1B-Chat-v1.0"
    #model_path = "/rscratch/agarcia/Llama-2-7b-hf"

    SEQLEN = 2048
    model = load_llama(model_path, seqlen=SEQLEN)
    dev = next(model.parameters()).device

    from datautils import get_wikitext2
    trainloader, testenc = get_wikitext2(nsamples=16, seed=0, seqlen=SEQLEN, model=model_path)
    ppl = llama_eval(model, testenc, seqlen=SEQLEN)
    print("Baseline ppl:", ppl)
        
    quantize_(model, AtlasQConfig(), filter_fn=skip_lm_head)
    print("Quantized model")
    
    ppl = llama_eval(model, testenc, seqlen=SEQLEN)
    print("Quantized ppl:", ppl)