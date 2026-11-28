"""Capture a causal LM's inference phases (exporter.fx_export CausalPrefill / CausalDecodeStep / CausalDecodeLayer) on WikiText-2
prompts and lay the bundle out as the other phase bundles: export/ (hard links of the export files), fx/ (capture_fx),
host_cases.safetensors (the prompts, int64 [n, P]) and host_cases.json (where each prompt comes from).

Prompts: WikiText-2 (Salesforce/wikitext, wikitext-2-raw-v1, test split) article lines (not headings) whose tokenization
(the model tokenizer, BOS included) holds at least P tokens, n of them evenly spaced over the eligible lines, each cut to its
first P tokens. The capture cases are the first two prompts: the prefill on each, the decode step (and a decode layer) at
decode steps --case-steps of the eager phases' greedy loop on each (the cache then holds real K / V of two different lengths).

usage (PYTHONPATH=<worktree>/src, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=1):
  causal_phase_capture.py --export-dir EXPORT --out BUNDLE --phase prefill|decode_step|decode_layer [--layer N]
      [--prefill-length 64] [--capacity 128] [--host-embeddings] [--prompts 6] [--case-steps 0 31] [--native-fp8]"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def wikitext_prompts(tokenizer, count, length):
    """(int64 [count, length], [record per prompt]): see the module docstring."""
    from datasets import load_dataset

    lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"]
    eligible = []
    for index, line in enumerate(lines):
        text = line.strip()
        if not text or text.startswith("="):
            continue
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        if ids.shape[0] >= length:
            eligible.append((index, text, ids[:length]))
    if len(eligible) < count:
        raise SystemExit(f"only {len(eligible)} WikiText-2 test lines hold {length} tokens")
    picks = [round(i * (len(eligible) - 1) / max(count - 1, 1)) for i in range(count)]
    prompts = torch.stack([eligible[p][2] for p in picks]).to(torch.int64)
    records = [{"line": eligible[p][0], "eligible_rank": p, "text_head": eligible[p][1][:80],
                "ids_sha256": hashlib.sha256(eligible[p][2].numpy().tobytes()).hexdigest()} for p in picks]
    return prompts, {"source": "Salesforce/wikitext wikitext-2-raw-v1 test", "eligible_lines": len(eligible),
                     "length": length, "bos": tokenizer.bos_token_id, "prompts": records}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out", required=True, help="bundle directory (export/, fx/, host_cases.*)")
    ap.add_argument("--phase", required=True, choices=("prefill", "decode_step", "decode_layer"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--model", help="architecture locator; defaults to the export's model id")
    ap.add_argument("--prefill-length", type=int, default=64)
    ap.add_argument("--capacity", type=int, default=128)
    ap.add_argument("--host-embeddings", action="store_true")
    ap.add_argument("--all-logits", action="store_true")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--case-steps", type=int, nargs=2, default=(0, 31))
    ap.add_argument("--native-fp8", action="store_true")
    a = ap.parse_args(argv)
    from transformers import AutoTokenizer

    import exporter.fx_export as fx_export
    import exporter.load_quantized as load_quantized
    import exporter.quant_export as quant_export

    out = Path(a.out)
    tensors, policy, manifest = load_quantized.read_export(a.export_dir)
    model_id = a.model or manifest["model"]["id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    prompts, meta = wikitext_prompts(tokenizer, a.prompts, a.prefill_length)
    model, _, _ = load_quantized.load_quantized(model_id, a.export_dir, "cpu", mode="ref_conv", seqlen=a.capacity)
    geometry = fx_export.causal_geometry(model, a.capacity, a.prefill_length)
    embed = fx_export.causal_token_embedder(tensors, policy) if a.host_embeddings else None
    del tensors
    if a.phase == "prefill":
        boundary = fx_export.CausalPrefill(model, a.host_embeddings, a.all_logits)
    elif a.phase == "decode_step":
        boundary = fx_export.CausalDecodeStep(model, a.host_embeddings)
    else:
        boundary = fx_export.CausalDecodeLayer(model, a.layer)
    boundary.eval()
    cases = fx_export.causal_phase_cases(model, boundary, geometry, 0, "cpu", count=2, host_embeddings=a.host_embeddings,
                                         steps=tuple(a.case_steps), embed=embed, prompts=prompts)
    out.mkdir(parents=True, exist_ok=True)
    fx_export.link_export(a.export_dir, out / "export")
    from exporter.quant_export import save_file

    save_file({"prompts": prompts.contiguous()}, str(out / "host_cases.safetensors"))
    meta.update(capture_cases=[0, 1], case_steps=list(a.case_steps) if a.phase != "prefill" else None)
    (out / "host_cases.json").write_text(json.dumps(meta, indent=1) + "\n")
    boundary_name = {"prefill": "causal_lm_prefill", "decode_step": "causal_lm_decode_step",
                     "decode_layer": "causal_lm_decode_layer"}[a.phase]
    report = fx_export.capture_fx(boundary, cases, out / "fx", policy=policy, strict=True,
                                  native_fp8="uint8" if a.native_fp8 else None,
                                  provenance={"export_dir": str(Path(a.export_dir).resolve()),
                                              "export_manifest_sha256": quant_export.sha256_file(
                                                  Path(a.export_dir) / "manifest.json"),
                                              "export_provenance": manifest.get("provenance", {}),
                                              "export_recipe": manifest.get("recipe", {}),
                                              "architecture_locator": model_id, "boundary": boundary_name,
                                              "capture_source": fx_export._capture_source(),
                                              "driver": "src/scripts/tile_flow/causal_phase_capture.py",
                                              "host_cases_sha256": quant_export.sha256_file(out / "host_cases.safetensors"),
                                              "seed": 0, "seqlen": a.prefill_length, "capacity": a.capacity,
                                              "prefill_length": a.prefill_length, "host_embeddings": a.host_embeddings,
                                              "all_logits": a.all_logits, "case_steps": list(a.case_steps),
                                              **({"layer": a.layer} if a.phase == "decode_layer" else {})})
    print(json.dumps({k: report[k] for k in ("status", "cases", "weight_sites_verified", "model_pt2_bytes")}, indent=1))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
