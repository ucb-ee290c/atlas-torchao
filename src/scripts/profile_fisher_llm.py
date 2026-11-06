"""Fisher-delta profile of a causal LM under the reference FP8 profile (linear layers + attention q/k/p/v).

    python src/scripts/profile_fisher_llm.py --model /rscratch/agarcia/models/TinyLlama-1.1B-Chat-v1.0 \\
        --n 16 --seqlen 512 --out src/out/fisher/tinyllama/tinyllama_n16.json
    python src/scripts/plot_fisher.py src/out/fisher/tinyllama/tinyllama_n16.json

Loss: next-token cross-entropy summed over the window's tokens. Score per layer / operand:
0.5 * sum(grad^2 * (quantized - clean)^2), averaged over windows."""
import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="/rscratch/agarcia/models/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--n", type=int, default=16, help="WikiText-2 train windows")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default=None, help="default: cuda if available")
    ap.add_argument("--include", default=None, help="regex over layer fqns, e.g. 'layers\\.(0|1)\\.'")
    ap.add_argument("--exclude", nargs="*", default=["lm_head"], help="fqn substrings left out")
    ap.add_argument("--no-attn", action="store_true", help="skip the attention q/k/p/v profile")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=None,
                    help="report JSON (default: src/out/fisher/<model>/<model>_n<N>.json)")
    a = ap.parse_args()
    if a.out is None:
        tag = re.split(r"[-_]", Path(a.model).name)[0].lower()
        a.out = str(REPO / "src" / "out" / "fisher" / tag / f"{tag}_n{a.n}.json")

    import torch.nn.functional as F

    import reference.ref_attention as ra
    from algos import calib
    from loaders.modelutils import load_llama
    from reference.ref_profile import profile_fisher_model, quantize_reference, write_report

    batches, rec = calib.wikitext2_train(a.model, a.n, a.seqlen, a.seed, use_fast=True)
    model = load_llama(a.model, seqlen=a.seqlen, device=a.device)
    print(quantize_reference(model, skip=tuple(a.exclude)))
    if not a.no_attn:
        print(len(ra.patch_attention(model)), "attention sites routed through ref_attention")

    def loss_fn(m, ids):
        ids = ids.to(next(m.parameters()).device)
        logits = m(ids, use_cache=False).logits[0, :-1].float()
        return F.cross_entropy(logits, ids[0, 1:], reduction="sum")

    res = profile_fisher_model(model, batches, loss_fn, include=a.include, exclude=tuple(a.exclude),
                               attention=not a.no_attn)
    bad = [n for n, r in res.items() if not 0 < r["fisher"] < float("inf")]
    print(f"{len(res)} entries profiled; zero / non-finite: {bad or 'none'}")
    for n, r in sorted(res.items(), key=lambda kv: -kv[1]["fisher"])[:a.top]:
        print(f"{n:55s} fisher={r['fisher']:.3e}" + (" (attn)" if r.get("kind") == "attn" else ""))
    write_report(res, {"model": a.model, "seqlen": a.seqlen, "n": a.n, "seed": a.seed, "include": a.include,
                       "calibration": rec, "loss": "cross_entropy(reduction=sum)"}, a.out)


if __name__ == "__main__":
    main()
