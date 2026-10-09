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


def main():
    import argparse
    import json
    import platform
    import time
    from pathlib import Path

    import torch

    from datautils import get_wikitext2
    from modelutils import load_llama
    from quant_modes import ALL_MODES, quantize, require_mode_supported

    ap = argparse.ArgumentParser(description="WikiText-2 perplexity of a Llama-family model per quantization mode")
    ap.add_argument("--model", required=True, help="local path or HF id of the model")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None, help="evaluate only the first N windows (smoke test)")
    ap.add_argument("--modes", nargs="+", default=["bf16", "atlas"], choices=list(ALL_MODES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write results as JSON")
    args = ap.parse_args()
    for m in args.modes:
        try:
            require_mode_supported(m, args.device)
        except RuntimeError as e:
            ap.error(str(e))

    _, testenc = get_wikitext2(nsamples=1, seed=0, seqlen=args.seqlen, model=args.model)
    ids = testenc.input_ids
    if args.max_windows:
        ids = ids[:, : args.max_windows * args.seqlen]
    result = {"args": vars(args), "torch": torch.__version__, "host": platform.node(),
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "windows": ids.shape[1] // args.seqlen, "ppl": {}, "layers": {}}
    for mode in args.modes:
        model = load_llama(args.model, seqlen=args.seqlen, device=args.device)  # fresh weights per mode
        if mode != "bf16":
            result["layers"][mode] = quantize(model, mode, skip=["lm_head"])
        t0 = time.perf_counter()
        result["ppl"][mode] = llama_eval(model, ids, seqlen=args.seqlen)
        print(f"{mode} ppl: {result['ppl'][mode]:.4f}  ({time.perf_counter() - t0:.1f}s, "
              f"{result['windows']} windows of {args.seqlen})", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
