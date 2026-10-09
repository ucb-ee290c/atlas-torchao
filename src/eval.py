from torch import nn

import torch 
import math

import torch
from tqdm import tqdm


@torch.no_grad()
def llama_eval(model, testenc, seqlen=None, before_window=None):
    """Perplexity over non-overlapping windows of `seqlen` tokens.
    testenc: tokenizer output (with .input_ids) or a [1, N] tensor of token ids.
    before_window(window), if given, runs before each window's forward (NEW; --rmse clean pass)."""
    ids = getattr(testenc, "input_ids", testenc)          # accept BatchEncoding or tensor
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    seqlen = seqlen or getattr(model, "seqlen", 512)     # default: the model's own seqlen
    dev = next(model.parameters()).device

    losses = []
    
    for i in tqdm(range(ids.shape[1] // seqlen)):
        window = ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        if before_window is not None:
            before_window(window)
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
    from ppl_protocols import PROTOCOLS, load_mxq_windows, mxq_eval
    from quant_modes import (ALL_MODES, add_profile_args, add_subset_compare_arg, finalize_report, ladder_deltas,
                             ladder_record, quantize, require_mode_supported, subset_compare_modes)

    ap = argparse.ArgumentParser(description="WikiText-2 perplexity of a Llama-family model per quantization mode")
    ap.add_argument("--model", required=True, help="local path or HF id of the model")
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="default protocol: evaluate only the first N windows (smoke test)")
    ap.add_argument("--protocol", choices=PROTOCOLS, default="default",
                    help="default: the collaborator's protocol (unchanged); mxq: microscaling-quant "
                         "experiments/llm_ppl.py (fast tokenizer, empty lines dropped, seeded windows, "
                         "token-summed NLL)")
    ap.add_argument("--nsamples", type=int, default=0,
                    help="mxq protocol: windows to evaluate, 0 = the whole test split (llm_ppl.py:40-43)")
    ap.add_argument("--seed", type=int, default=0, help="mxq protocol: which windows (llm_ppl.py:54-57)")
    ap.add_argument("--sequential", action="store_true",
                    help="mxq protocol: the first --nsamples windows in order instead of seeded ones")
    ap.add_argument("--rmse", action="store_true",
                    help="per-layer (isolated) and cumulative relative RMSE of every quantized layer "
                         "against a clean bf16 pass, written as the reference author's CSVs")
    ap.add_argument("--rmse-dir", default=None,
                    help="where the --rmse CSVs go (<dir>/<mode>/); default: next to --out")
    ap.add_argument("--modes", nargs="+", default=["bf16", "atlas"], choices=list(ALL_MODES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write results as JSON")
    add_profile_args(ap)
    add_subset_compare_arg(ap)
    args = ap.parse_args()
    # a subset: seeded / first-N mxq windows, or the first --max-windows of the split
    subset = bool(args.max_windows) if args.protocol != "mxq" else args.nsamples > 0
    args.modes, added = subset_compare_modes(args.modes, args.subset_compare == "on" and subset)
    for m in args.modes:
        try:
            require_mode_supported(m, args.device)
        except RuntimeError as e:
            ap.error(str(e))
    if args.rmse and not (args.rmse_dir or args.out):
        ap.error("--rmse needs --rmse-dir or --out")
    rmse_dir = Path(args.rmse_dir) if args.rmse_dir else (
        Path(args.out).with_name(Path(args.out).stem + "_rmse") if args.rmse else None)

    result = {"args": vars(args), "torch": torch.__version__, "host": platform.node(),
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              "protocol": args.protocol, "ppl": {}, "layers": {}, "ladder": {},
              "subset_compare": {"enabled": bool(added) or None, "added_modes": added}}
    if args.protocol == "mxq":
        windows, picked = load_mxq_windows(args.model, args.seqlen, args.nsamples,
                                           None if args.sequential else args.seed)
        result.update(windows=windows.shape[0], window_indices=picked, per_sample={})
    else:
        _, testenc = get_wikitext2(nsamples=1, seed=0, seqlen=args.seqlen, model=args.model)
        ids = testenc.input_ids
        if args.max_windows:
            ids = ids[:, : args.max_windows * args.seqlen]
        result["windows"] = ids.shape[1] // args.seqlen

    clean = None
    if args.rmse:
        import layer_rmse

        clean = load_llama(args.model, seqlen=args.seqlen, device=args.device)  # the clean bf16 pass
        result["rmse"] = {}
    for mode in args.modes:
        model = load_llama(args.model, seqlen=args.seqlen, device=args.device)  # fresh weights per mode
        if mode != "bf16":
            result["layers"][mode] = quantize(model, mode, skip=["lm_head"], options=args)
        result["ladder"][mode] = (result["layers"][mode]["ladder"] if mode != "bf16"
                                  else ladder_record("bf16", {}, args))
        eq = result["ladder"][mode].get("equivalent_to")
        if eq in result["ppl"]:
            # the op class this step adds is absent (TinyLlama has no Conv2d): same model, same numbers
            result["ladder"][mode]["duplicate_of"] = eq
            result["ppl"][mode] = result["ppl"][eq]
            if args.protocol == "mxq":
                result["per_sample"][mode] = result["per_sample"][eq]
            print(f"{mode}: identical to {eq} (nothing added to quantize); not re-run", flush=True)
            del model
            continue
        rm = None
        if clean is not None and mode != "bf16":
            # NEW: a causal LM is all language model; the reference taxonomy has no rule for its paths
            rm = layer_rmse.LayerRMSE(clean, model, default_component="LANGUAGE")
        before = (lambda w: rm.capture(lambda: clean(w.to(args.device), use_cache=False))) if rm else None
        t0 = time.perf_counter()
        if args.protocol == "mxq":
            r = mxq_eval(model, windows, before_window=before)
            result["ppl"][mode] = r["ppl"]
            result["per_sample"][mode] = r["per_sample"]
        else:
            result["ppl"][mode] = llama_eval(model, ids, seqlen=args.seqlen, before_window=before)
        finalize_report(result["layers"].get(mode))
        print(f"{mode} ppl: {result['ppl'][mode]:.4f}  ({time.perf_counter() - t0:.1f}s, "
              f"{result['windows']} windows of {args.seqlen}, protocol {args.protocol})", flush=True)
        if rm is not None:
            rm.close()
            result["rmse"][mode] = {**rm.summary(), "csv": rm.write_csvs(rmse_dir / mode)}
            for row in result["rmse"][mode]["components"]:
                print(f"  {mode} rmse {row['component']}: layers {row['n_layers']}  "
                      f"mean_rel_rmse {row['mean_rel_rmse']:.4e}  "
                      f"mean_cumulative_rel_rmse {row['mean_cumulative_rel_rmse']:.4e}", flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    if args.rmse:
        result["reference_provenance"] = layer_rmse.ref_provenance()
    result["ladder_deltas"] = ladder_deltas({m: {"ppl": v} for m, v in result["ppl"].items()}, result["ladder"])
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print("wrote", args.out)


if __name__ == "__main__":
    main()
