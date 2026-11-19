"""SmolVLA inference-time activation ranges against the export's static activation scales, per denoising step.

The static_calib exponents of an export (atlas_v1_awq_tile_static) were calibrated on the flow-matching training forward
(VLAFlowMatching.forward: one random time per sample, x_t = t * noise + (1 - t) * actions). At inference the expert runs ten
times per chunk on x_t from pure noise (t = 1) to near the action (t = 0.1), and the prefix once. This runs the eager quantized
policy's sample_actions (the computation the two phase graphs reproduce byte for byte) on LIBERO frames with a forward
pre-hook on every quantized Linear / Conv2d whose activation rule is static, and records, per site and per call, the input's
BF16 magnitude maximum (operands are cast to BF16 before the vpack encoder) and how many elements exceed the largest code.

Per site with static exponent e (x is encoded as e4m3(x / 2^e), saturating at 448):
  saturation threshold     448 * 2^e; elements above it are clipped to +-448 * 2^e;
  admitted amax range      [256 * 2^e, 512 * 2^e): the amaxes for which the calibration rule (static_calib absmax:
                           floor(log2(amax / 256)), algos/static_calib.ref_exp) gives e;
  outside (hot)            a step whose amax is >= 512 * 2^e, i.e. the reference per-call rule would have chosen a larger
                           exponent (always saturates);
  outside (cold)           a step whose amax is < 256 * 2^e (the rule would choose a smaller exponent: lost resolution, no
                           clipping). Reported, not flagged.
A step of an expert / action site is its call index within one sample_actions (one call per denoising step); prefix sites
(vision tower, connector, VLM text layers, state_proj) run once per chunk (the vision tower once per camera).

usage (PYTHONPATH=<worktree>/src): smolvla_act_ranges.py --export-dir EXPORT --out JSON [--frames 16]"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def ref_exp(amax):
    from algos.static_calib import ref_exp as rule

    return rule(amax)


def site_rules(policy_json):
    """{site: static exponent} of every site whose activation rule is static."""
    return {name: int(site["activation"]["exp"]) for name, site in policy_json["sites"].items()
            if (site.get("activation") or {}).get("mode") == "static"}


def observe(policy, observations, rules, num_steps):
    """{site: [[(amax, above, numel) per call] per frame]} over sample_actions on each observation."""
    from exporter.fx_export import _is_quant_layer

    records = defaultdict(list)
    current = {}
    hooks = []
    for name, module in policy.named_modules():
        if name in rules and _is_quant_layer(module):
            limit = 448.0 * 2.0 ** rules[name]

            def hook(mod, args, name=name, limit=limit):
                x = args[0].detach().to(torch.bfloat16).float().abs()
                current.setdefault(name, []).append((float(x.max()), int((x > limit).sum()), x.numel()))
            hooks.append(module.register_forward_pre_hook(hook))
    missing = sorted(set(rules) - {n for n, m in policy.named_modules() if _is_quant_layer(m)})
    try:
        with torch.no_grad():
            for o in observations:
                current.clear()
                policy.model.sample_actions(o["images"], o["img_masks"], o["lang_tokens"], o["lang_masks"], o["state"],
                                            noise=o["noise"])
                for name, calls in current.items():
                    records[name].append(list(calls))
    finally:
        for h in hooks:
            h.remove()
    return records, missing


def summarize(records, rules, num_steps, times):
    sites, hot, saturating = {}, [], []
    by_step = defaultdict(lambda: {"sites": 0, "hot": 0, "cold": 0, "saturating": 0})
    for name, frames in sorted(records.items()):
        e = rules[name]
        calls = {len(f) for f in frames}
        if len(calls) != 1:
            raise SystemExit(f"{name}: called {sorted(calls)} times in different frames")
        n = calls.pop()
        per_step = n == num_steps
        rows = []
        for c in range(n):
            amax = max(f[c][0] for f in frames)
            above = sum(f[c][1] for f in frames)
            numel = sum(f[c][2] for f in frames)
            r = ref_exp(amax)
            label = f"step_{c}" if per_step else ("prefix" if n == 1 else f"prefix_call_{c}")
            row = {"call": label, **({"t": times[c]} if per_step else {}), "amax": amax,
                   "amax_over_saturation": amax / (448.0 * 2.0 ** e), "rule_exponent": r,
                   "status": "hot" if r > e else ("cold" if r < e else "admitted"),
                   "elements_above_448": above, "elements": numel}
            rows.append(row)
            key = label if per_step else "prefix"
            by_step[key]["sites"] += 1
            by_step[key]["hot"] += r > e
            by_step[key]["cold"] += r < e
            by_step[key]["saturating"] += above > 0
            if r > e:
                hot.append({"site": name, "call": label, "exp": e, "amax": amax, "rule_exponent": r,
                            "amax_over_saturation": row["amax_over_saturation"], "elements_above_448": above})
            if above:
                saturating.append({"site": name, "call": label, "exp": e, "amax": amax, "elements_above_448": above,
                                   "fraction": above / numel})
        sites[name] = {"exp": e, "saturation_threshold": 448.0 * 2.0 ** e,
                       "admitted_amax_range": [256.0 * 2.0 ** e, 512.0 * 2.0 ** e],
                       "calls_per_chunk": n, "phase": "denoise_step" if per_step else "prefix", "calls": rows}
    order = ["prefix"] + [f"step_{c}" for c in range(num_steps)]
    return sites, hot, saturating, {k: dict(by_step[k]) for k in order if k in by_step}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--dataset-root")
    a = ap.parse_args(argv)
    import smolvla_phase_capture as spc
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import make_loop_manifest as mlm

    t0 = time.time()
    policy, cfg, policy_json, manifest = spc.load_policy(a.export_dir)
    if cfg.rtc_config is not None and cfg.rtc_config.enabled:
        raise SystemExit("RTC is enabled in the policy config")
    obs, record = spc.observations(policy, cfg, manifest["model"]["id"], a.frames, a.dataset_root)
    rules = site_rules(policy_json)
    times, _, _ = mlm.euler_integrate_grid(cfg.num_steps)
    records, missing = observe(policy, obs, rules, cfg.num_steps)
    sites, hot, saturating, by_step = summarize(records, rules, cfg.num_steps, times)
    never = sorted(set(rules) - set(records) - set(missing))
    out = {"schema": "atlas_torchao.smolvla_inference_act_ranges.v1", "export_dir": str(Path(a.export_dir).resolve()),
           "export_recipe": manifest.get("recipe", {}).get("expanded"), "frames": record,
           "observations": [o["meta"] for o in obs], "num_steps": cfg.num_steps, "times": times,
           "definitions": {"saturation_threshold": "448 * 2^exp", "admitted_amax_range": "[256 * 2^exp, 512 * 2^exp): "
                           "the calibration rule floor(log2(amax / 256)) gives exp", "hot": "amax >= 512 * 2^exp",
                           "cold": "amax < 256 * 2^exp", "amax": "max over frames of the BF16 |x| maximum of that call"},
           "static_sites": len(rules), "sites_observed": len(records), "sites_never_called": never,
           "sites_not_quantized_modules": missing,
           "summary": {"hot_site_calls": len(hot), "hot_sites": sorted({h["site"] for h in hot}),
                       "saturating_site_calls": len(saturating), "saturating_sites": sorted({s["site"] for s in saturating}),
                       "by_call": by_step},
           "hot": hot, "saturating": saturating, "sites": sites, "seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("static_sites", "sites_observed", "summary", "seconds")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
