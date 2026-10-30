"""Per-layer shape / DMA / MXU / flush report for an Atlas FP8 export, plus the pop-scale applicability table.

    python scripts/hw_report.py [--export model.safetensors] [--T 1 32 128 2048] [--mxu ipt|sa]
                                [--out-json results/hw-verify/hw_report_tinyllama.json] [--out-md -]

Reads the export's header and uint8 E4M3 code tensors (safetensors), policy.json and manifest.json beside
it. Weight flush mass is an UPPER BOUND (zero codes = flushed values + true zeros). Activation flush mass
needs captured activations and is reported as NOT MEASURED. Cycle counts are DERIVED from hw_params.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import hw_cost as hc  # noqa: E402
import hw_params as hp  # noqa: E402

DEFAULT_EXPORT = ("/scratch2/lorenhung/atlas-fp8/handoff/atlas_v1/export_smoke/tinyllama/atlas_v1/"
                  "model.safetensors")
RESULTS = Path("/scratch2/lorenhung/atlas-fp8/results/hw-verify")


def read_export(path: Path):
    from safetensors import safe_open
    policy = json.loads((path.parent / "policy.json").read_text())
    manifest = json.loads((path.parent / "manifest.json").read_text())
    rows = []
    with safe_open(str(path), "numpy") as f:
        for site, d in policy["sites"].items():
            if d.get("kind") != "linear" or d.get("status") != "quantized":
                continue
            w = d["weight"]
            codes = f.get_tensor(w["codes"])
            sexp = f.get_tensor(w["scale_exp"])
            assert codes.dtype == np.uint8 and codes.ndim == 2, (site, codes.dtype, codes.shape)
            N, K = codes.shape
            assert [N, K] == list(d["weight_shape"]), (site, N, K, d["weight_shape"])
            hist = np.bincount(codes.ravel(), minlength=256)
            e = sexp.reshape(N, -1)[:, 0].astype(np.int64)
            groups = e.reshape(-1, hp.TILE) if N % hp.TILE == 0 else None
            uniform = float((groups.min(1) == groups.max(1)).mean()) if groups is not None else None
            rows.append(dict(site=site, N=N, K=K, hist=hist, scale_granularity=w["granularity"],
                             scale_exp_min=int(e.min()), scale_exp_max=int(e.max()),
                             scale_exp_distinct=int(len(np.unique(e))),
                             group32_uniform_frac=uniform, act=d["activation"]["label"],
                             bias=d.get("bias")))
    return rows, policy, manifest


def build(args) -> dict:
    path = Path(args.export)
    rows, policy, manifest = read_export(path)
    layers, per_layer = [], []
    for r in rows:
        lc = hc.layer_cost(r["site"], r["N"], r["K"], T=args.T[-1] if args.T_main is None else args.T_main,
                           mxu=args.mxu)
        layers.append(lc)
        fl = hc.flush_stats(r["hist"])
        per_layer.append(dict(**hc.cost_dict(lc), **fl, scale_granularity=r["scale_granularity"],
                              scale_exp_min=r["scale_exp_min"], scale_exp_max=r["scale_exp_max"],
                              scale_exp_distinct=r["scale_exp_distinct"],
                              group32_uniform_frac=r["group32_uniform_frac"], activation_scale=r["act"]))
    agg = hc.aggregate(layers, T_list=args.T, mxu=args.mxu)
    # weighted flush upper bound per type and overall
    el_tot = sum(p["elements"] for p in per_layer)
    z_tot = sum(p["zero_codes"] for p in per_layer)
    flush = dict(label="UPPER BOUND on weight flushed mass: zero codes = values flushed below 2^-6 plus exact zeros",
                 elements=el_tot, zero_codes=z_tot, zero_code_frac=z_tot / el_tot,
                 subnormal_codes=sum(p["subnormal_codes"] for p in per_layer),
                 reserved_codes=sum(p["reserved_codes"] for p in per_layer), per_type={},
                 activation_flush_mass="NOT MEASURED (needs captured activations)")
    for ty in sorted({p["type"] for p in per_layer}):
        ps = [p for p in per_layer if p["type"] == ty]
        e, z = sum(p["elements"] for p in ps), sum(p["zero_codes"] for p in ps)
        fr = [p["zero_code_frac_upper_bound_of_flushed"] for p in ps]
        flush["per_type"][ty] = dict(zero_code_frac=z / e, min_layer=min(fr), max_layer=max(fr))
    # pop scale
    g = hc.trace_layer_graph()
    cls = hc.classify_linears(g)
    seq = g["seq"]
    grp = {}
    for ty in hc.QUANTIZED_LINEARS:
        ps = [p for p in per_layer if p["type"] == ty]
        grp[ty] = dict(frac=float(np.mean([p["group32_uniform_frac"] for p in ps])),
                       all_uniform=all(p["group32_uniform_frac"] == 1.0 for p in ps),
                       N=ps[0]["N"], act=ps[0]["activation_scale"])
    pop = dict(classification={k: dict(v, terminals=[list(t) for t in v["terminals"]]) for k, v in cls.items()},
               group32_uniform_frac_by_type={k: v["frac"] for k, v in grp.items()}, plans={})
    for T in args.T:
        plans = {}
        for ty in hc.QUANTIZED_LINEARS:
            c = cls[ty]
            v_site = ty == "v_proj"
            for variant, kw in (("as_exported", dict(act_scale="per_token", w=grp[ty]["all_uniform"], cs=False)),
                                ("best_case", dict(act_scale="static", w=True, cs=True))):
                plans[f"{ty}/{variant}"] = hc.rescale_plan(
                    c["consumer_class"], kw["act_scale"], kw["w"], consumer_scale_static=kw["cs"] if v_site else None,
                    tokens=T, N=grp[ty]["N"])
        # P.V -> o_proj operand (attention matmul output; the matmul is not a linear)
        for variant, kw in (("as_exported", dict(act_scale="per_head", cs=False)),
                            ("best_case", dict(act_scale="static", cs=True))):
            plans[f"attn.pv/{variant}"] = hc.rescale_plan(
                cls["attn.pv"]["consumer_class"], kw["act_scale"], True, consumer_scale_static=kw["cs"],
                tokens=T, N=grp["o_proj"]["N"])
        pop["plans"][str(T)] = plans
    n_layers = hc.TINYLLAMA["num_hidden_layers"]
    pop["extra_cycles_per_forward"] = {}
    for T in args.T:
        for variant in ("as_exported", "best_case"):
            tot = sum(p["layer_extra_cycles"] for k, p in pop["plans"][str(T)].items() if k.endswith(variant)) * n_layers
            pop["extra_cycles_per_forward"][f"T={T}/{variant}"] = dict(cycles=tot, seconds=tot / hp.CLOCK_HZ)
    layout = dict(
        stored="codes [N, K] uint8 row-major, row n = output channel n (shape asserted equal to weight_shape [N, K] "
               "for all %d sites); scale_exp [N, 1] int32" % len(per_layer),
        rtl_push="IPT: InnerProductTreesSequencer.scala:424-428 writes mreg row r to weight lane laneIdx=r; "
                 "WeightBuffers.scala:62-67 stores it as slot(lane); InnerProductTrees.scala:60-64 gives output "
                 "column n the lane-n K-vector; AnchorAccumulationTree.scala:53 pairs weight element j with "
                 "activation element j. SA: SystolicArrayTop.scala:116-117 wires weights(i)(j) = slot(j)(i), "
                 "lane j = output column j, i = K index.",
        conclusion="A 32x32 block W[n0:n0+32, k0:k0+32] of the export is exactly the mreg image (row n = K-vector of "
                   "output channel n). No transpose is needed for the weight push on either MXU. The shipped "
                   "npu-model parameterized_matmul kernel stores B as [K,N] and therefore runs vtrpose.xlu "
                   "(66 cycles) per tile; the [N,K] export avoids it.",
        status="RTL read settles the IPT and SA lane orientation; how vload fills an mreg row from VMEM bytes is "
               "settled by the npu-model simulator run (scripts/npu_model_l2.py, bit comparison) -- see that result.",
        dma_caveat="The export is row-major over the full K, so a tile is 32 pieces of 32 B at stride K, not one "
                   "contiguous 1 KiB. Headline cycles assume an offline tile-major re-layout (a permutation, no "
                   "transpose) so the DMA streams 4 KiB bursts; the as-stored cost (32 B transfers) is reported "
                   "as dma_cycles_strided (DERIVED UPPER BOUND, serialized transfers; pipelined cost not measured).",
    )
    return dict(
        schema="atlas_torchao.hw_report.v1", export=str(path), export_sha256=manifest["files"]["model.safetensors"]["sha256"],
        model=manifest["model"]["id"], mxu=args.mxu, T_list=args.T,
        labels=dict(MEASURED="shapes, code histograms, scale exponents from the export",
                    DERIVED="cycles and seconds from hw_params constants", INFERRED="layout reuse, push hidden by 2 slots",
                    NOT_MEASURED="activation flush mass"),
        hw=dict(clock_hz=hp.CLOCK_HZ, dma_load_bytes_per_cycle=hp.DMA_LOAD_BYTES_PER_CYCLE, vmem_bytes=hp.VMEM_BYTES,
                mxu_issue_gap=hp.MXU_ISSUE_GAP, mxu_latency=hp.MXU_MATMUL_LATENCY[args.mxu]),
        aggregate=agg, weight_flush=flush, pop_scale=pop, layout=layout, layers=per_layer,
    )


def fmt_md(rep: dict) -> str:
    a = rep["aggregate"]
    L = []
    L.append("| type | count | N x K | tiles/layer | MiB/layer | DMA cyc/layer | MXU cyc (32 tok) | compute-bound at | fits VMEM | zero-code frac (upper bound) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for ty, p in a["per_type"].items():
        fz = rep["weight_flush"]["per_type"][ty]["zero_code_frac"]
        first = next(l for l in rep["layers"] if l["type"] == ty)
        L.append(f"| {ty} | {p['count']} | {p['N']} x {p['K']} | {p['tiles_each']} | {p['fp8_mib'] / p['count']:.2f} | "
                 f"{first['dma_cycles']:,.0f} | {first['mxu_cycles_block']:,} | {p['compute_bound_blocks']} blocks = {p['compute_bound_tokens_min']} tokens | "
                 f"{'yes' if p['fits_vmem'] else 'no'} | {fz:.4%} |")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_EXPORT)
    ap.add_argument("--T", type=int, nargs="+", default=[1, 32, 128, 2048])
    ap.add_argument("--T-main", type=int, default=None, help="token count for the per-layer mxu_cycles_T column")
    ap.add_argument("--mxu", choices=("ipt", "sa"), default="ipt")
    ap.add_argument("--out-json", default=str(RESULTS / "hw_report_tinyllama.json"))
    ap.add_argument("--out-md", default="-", help="markdown summary table path, or - for stdout")
    args = ap.parse_args()
    rep = build(args)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(rep, indent=1))
    md = fmt_md(rep)
    if args.out_md == "-":
        print(md)
    else:
        Path(args.out_md).write_text(md + "\n")
    a = rep["aggregate"]
    print(f"\nlayers {a['layers']}  FP8 {a['fp8_mib']:.1f} MiB  one weight pass {a['weight_pass_dma_cycles']:,.0f} cycles = "
          f"{a['weight_pass_seconds']:.3f} s at 500 MHz (DERIVED)  weight zero-code frac (upper bound) "
          f"{rep['weight_flush']['zero_code_frac']:.4%}")


if __name__ == "__main__":
    main()
