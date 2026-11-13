"""Weight bytes each per-phase graph reads, from the saved graphs: per call and per action chunk.

For each phase bundle under PHASES_DIR (the graphs named in its loop_manifest.json): the quantized weight payload of every
atlas.qlinear / atlas.qconv2d site (E4M3 codes, int32 exponents, bias) times the site's calls in that graph
(graph_inspection.json weight_site_bindings), and the graph's other state (BF16 parameters and buffers the graph holds:
patch embeddings kept in BF16, norms, position tables). Per chunk = per call x calls_per_action_chunk from the manifest.
A site called N times in one graph is counted N times (each call streams it), the upper bound for a compiler that does not
keep the weights resident between calls.
usage: phase_sizes.py PHASES_DIR [PHASES_DIR ...] --out sizes.json"""
import argparse
import json
import sys
from pathlib import Path

import torch

import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  registers atlas.* ops for torch.export.load

MIB = 1 << 20


def _nbytes(t):
    return t.numel() * t.element_size()


def graph_sizes(fx_dir):
    fx_dir = Path(fx_dir)
    program = torch.export.load(fx_dir / "model.pt2")
    gi = json.loads((fx_dir / "graph_inspection.json").read_text())
    state = {**dict(program.named_parameters()), **dict(program.named_buffers())}
    state.update({k: v for k, v in program.constants.items() if isinstance(v, torch.Tensor)})
    sites, counted = [], set()
    for row in gi["weight_site_bindings"]:
        names = [row["code_buffer"], row["scale_buffer"], row["module"] + ".bias"]
        once = sum(_nbytes(state[n]) for n in names if n in state)
        counted.update(n for n in names if n in state)
        sites.append({"site": row["site"], "calls": row["calls"], "bytes_once": once, "bytes_per_graph_call": once * row["calls"]})
    other = {k: _nbytes(v) for k, v in state.items() if k not in counted}
    q = sum(s["bytes_per_graph_call"] for s in sites)
    return {"quantized_sites": len(sites), "qlinear_or_qconv_calls": sum(s["calls"] for s in sites),
            "quantized_weight_bytes_per_call": q, "quantized_weight_MiB_per_call": q / MIB,
            "quantized_weight_bytes_distinct": sum(s["bytes_once"] for s in sites),
            "other_state_bytes": sum(other.values()), "other_state_MiB": sum(other.values()) / MIB,
            "sites_called_more_than_once": {s["site"]: s["calls"] for s in sites if s["calls"] > 1}}


def phases_sizes(phases_dir):
    phases_dir = Path(phases_dir)
    lm = json.loads((phases_dir / "loop_manifest.json").read_text())
    out = {"phases_dir": str(phases_dir), "model": lm["model"], "layout": lm.get("layout", "two_phase"),
           "denoising_steps": lm["denoising_steps"], "graphs": {}}
    chunk_q = chunk_all = 0
    for name, g in lm["graphs"].items():
        fx = phases_dir / g["path"]
        if not (fx / "model.pt2").exists():
            out["graphs"][name] = {"missing": str(fx)}
            continue
        s = graph_sizes(fx)
        n = g["calls_per_action_chunk"]
        s.update({"calls_per_action_chunk": n,
                  "quantized_weight_bytes_per_chunk": s["quantized_weight_bytes_per_call"] * n,
                  "quantized_weight_MiB_per_chunk": s["quantized_weight_bytes_per_call"] * n / MIB,
                  "all_state_bytes_per_chunk": (s["quantized_weight_bytes_per_call"] + s["other_state_bytes"]) * n})
        chunk_q += s["quantized_weight_bytes_per_chunk"]
        chunk_all += s["all_state_bytes_per_chunk"]
        out["graphs"][name] = s
    out["quantized_weight_bytes_per_chunk"] = chunk_q
    out["quantized_weight_MiB_per_chunk"] = chunk_q / MIB
    out["all_state_bytes_per_chunk"] = chunk_all
    out["all_state_MiB_per_chunk"] = chunk_all / MIB
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases_dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"note": __doc__.split("\n\n")[0], "bundles": [phases_sizes(d) for d in a.phases_dirs]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
