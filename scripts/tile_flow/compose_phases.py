"""Compose the per-phase DiT graphs the way a host would and compare with the existing single-step bundle (and, optionally, with the
eager 100-step sampler). Uses only torch.export.load on the two saved graphs plus the loop manifest; no model class is imported
for the one-step comparison.

  (a) one step: step(encode(obs), noisy_actions, timestep) against (1) the single-step bundle's graph (fx/model.pt2) run in the same
      process on the same inputs and (2) that bundle's recorded fixture output, byte for byte, and encode's output against the
      encode bundle's own fixture. Only (1) decides `ok`: a recorded fixture is a different process's run, and the eager original
      boundary can itself differ from it on this host (see docs/algos/phase-graphs.md).
  (b) --eager-loop EXPORT_DIR: the host loop of loop_manifest.json (100 steps, DDPM arithmetic from the per_step table, variance noise
      from the global RNG in the same order diffusers draws it) over the step graph, against DiffusionObjective.conditional_sample
      of the eagerly rebuilt quantized policy with the same seed.
  Pi0 (atlas-pi0 env): (a) prefix, then step with time 1.0, then x = noise + dt * v with dt = -1 (the single-step bundle is
  sample_actions(num_steps=1)) against that bundle's graph in the same process and against its fixture; (b) with --eager-loop the 10-step
  host loop of the manifest (euler.times, float32 dt) against model.sample_actions(num_steps=10) on the same noise.
usage (PYTHONPATH=<worktree>/src):
  compose_phases.py dit PHASES_DIR SINGLE_STEP_BUNDLE OUT_JSON [--eager-loop EXPORT_DIR] [--seed N]            # atlas-lerobot env
  compose_phases.py pi0 PHASES_DIR SINGLE_STEP_BUNDLE OUT_JSON [--eager-loop EXPORT_DIR --openpi-dir D --checkpoint-dir D]  # atlas-pi0 env"""
import argparse
import json
import sys
from pathlib import Path

import torch

import atlas_fx_ops  # noqa: F401  registers atlas.* ops for torch.export.load
from fx_export import tensor_record


def _bytes_equal(a, b):
    return tensor_record(a) == tensor_record(b)


def _diff(a, b):
    return float((a.float() - b.float()).abs().max())


def ddpm_host_step(row, x, eps, z, clip=1.0):
    """DDPMScheduler.step from one per_step row of the manifest, with diffusers' operation order and dtypes (0-dim float32
    coefficients times bf16 tensors stay bf16)."""
    f = lambda v: torch.tensor(v, dtype=torch.float32)       # noqa: E731
    x0 = (x - f(row["sqrt_beta_prod_t"]) * eps) / f(row["sqrt_alpha_prod_t"])
    x0 = x0.clamp(-clip, clip)
    prev = f(row["pred_original_sample_coeff"]) * x0 + f(row["current_sample_coeff"]) * x
    return prev + f(row["variance_std"]) * z if row["variance_noise_draw"] else prev + 0


def host_loop(step, cond, manifest, seed):
    """The manifest's loop: initial noise, then len(per_step) calls of the step graph with host scheduler arithmetic."""
    info = manifest["scheduler"]["initial_noise"]
    dtype = getattr(torch, {"BF16": "bfloat16", "F32": "float32"}[info["dtype"]])
    torch.manual_seed(seed)
    x = torch.randn(info["shape"], dtype=dtype)
    calls = 0
    for row in manifest["scheduler"]["per_step"]:
        eps = step(cond, x, torch.full((x.shape[0],), row["t"], dtype=torch.long))
        calls += 1
        z = torch.randn(eps.shape, dtype=eps.dtype) if row["variance_noise_draw"] else None
        x = ddpm_host_step(row, x, eps, z, manifest["scheduler"]["config"]["clip_sample_range"])
    return x, calls


def pi0_host_loop(step, state, pad, noise, kv, manifest):
    """The manifest's Euler loop: float32 dt and times, x = x + dt * v on the host."""
    dt = torch.tensor(manifest["euler"]["dt"], dtype=torch.float32)
    x, calls = noise, 0
    for t in manifest["euler"]["times"]:
        time = torch.tensor(t, dtype=torch.float32).expand(x.shape[0])
        x = x + dt * step(state, pad, x, time, *kv)
        calls += 1
    return x, calls


def compose_pi0(a):
    from safetensors.torch import load_file

    ph, ss = Path(a.phases_dir), Path(a.single_step_bundle)
    manifest = json.loads((ph / "loop_manifest.json").read_text())
    pre = torch.export.load(ph / "prefix" / "fx" / "model.pt2").module()
    stp = torch.export.load(ph / "step" / "fx" / "model.pt2").module()
    single = torch.export.load(ss / "fx" / "model.pt2").module()
    fx_ss = load_file(str(ss / "fx" / "fixtures.safetensors"))
    parity = json.loads((ss / "fx" / "parity.json").read_text())
    kv_shape = manifest["graphs"]["prefix"]["outputs"]["kv_cache"]["per_tensor_shape"]
    res = {"phases_dir": str(ph), "single_step_bundle": str(ss), "one_step": [], "ok": True}
    one = torch.tensor(1.0, dtype=torch.float32)
    with torch.no_grad():
        for i, case in enumerate(parity["cases"]):
            ins = [fx_ss[k] for k in case["inputs"]]
            pad, *kv = pre(*ins[:8])
            v = stp(ins[8], pad, ins[9], one.expand(1), *kv)
            out = ins[9] + torch.tensor(-1.0, dtype=torch.float32) * v
            same, want = single(*ins), fx_ss[case["outputs"][0]]
            row = {"case": i, "kv_tensors": len(kv), "kv_shapes_match_manifest": all(list(k.shape) == kv_shape for k in kv)
                   and len(kv) == manifest["graphs"]["prefix"]["outputs"]["kv_cache"]["tensors"],
                   "prefix_pad_masks_shape": list(pad.shape),
                   "composition_equals_single_step_graph_same_process": _bytes_equal(out, same),
                   "max_abs_diff_vs_single_step_graph": _diff(out, same),
                   "composition_equals_recorded_fixture": _bytes_equal(out, want),
                   "single_step_graph_equals_recorded_fixture": _bytes_equal(same, want),
                   "max_abs_diff_vs_recorded_fixture": _diff(out, want)}
            res["one_step"].append(row)
            res["ok"] &= row["composition_equals_single_step_graph_same_process"] and row["kv_shapes_match_manifest"]
        if a.eager_loop:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            sys.path.insert(0, str(ss))
            import pi0_phase_capture as ppc

            model, _ = ppc.load_pi0(a.eager_loop, a.openpi_dir, a.checkpoint_dir, a.tokenizer_path)
            loops = []
            for i, case in enumerate(parity["cases"]):
                ins = [fx_ss[k] for k in case["inputs"]]
                obs = ppc.SimpleNamespace(
                    images=dict(zip(ppc.KEYS, ins[:3])), image_masks=dict(zip(ppc.KEYS, ins[3:6])), state=ins[8],
                    tokenized_prompt=ins[6], tokenized_prompt_mask=ins[7], token_ar_mask=torch.zeros_like(ins[7]),
                    token_loss_mask=torch.zeros_like(ins[7]))
                eager = model.sample_actions("cpu", obs, noise=ins[9], num_steps=manifest["denoising_steps"])
                pad, *kv = pre(*ins[:8])
                graph, calls = pi0_host_loop(stp, ins[8], pad, ins[9], kv, manifest)
                loops.append({"case": i, "step_graph_calls": calls, "prefix_graph_calls": 1,
                              "bytes_equal": _bytes_equal(graph, eager), "max_abs_diff": _diff(graph, eager),
                              "shape": list(graph.shape), "dtype": str(graph.dtype)})
                res["ok"] &= loops[-1]["bytes_equal"]
            res["host_loop_vs_eager_sample_actions"] = loops
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit", "pi0"))
    ap.add_argument("phases_dir")
    ap.add_argument("single_step_bundle")
    ap.add_argument("out")
    ap.add_argument("--eager-loop", metavar="EXPORT_DIR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--openpi-dir")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--tokenizer-path", default="/scratch2/lorenhung/models/big_vision/paligemma_tokenizer.model")
    a = ap.parse_args(argv)
    if a.model == "pi0":
        return compose_pi0(a)
    from safetensors.torch import load_file

    ph, ss = Path(a.phases_dir), Path(a.single_step_bundle)
    manifest = json.loads((ph / "loop_manifest.json").read_text())
    enc = torch.export.load(ph / "encode" / "fx" / "model.pt2").module()
    stp = torch.export.load(ph / "step" / "fx" / "model.pt2").module()
    fx_ss = load_file(str(ss / "fx" / "fixtures.safetensors"))
    parity = json.loads((ss / "fx" / "parity.json").read_text())
    single = torch.export.load(ss / "fx" / "model.pt2").module()
    fx_enc = load_file(str(ph / "encode" / "fx" / "fixtures.safetensors"))
    enc_parity = json.loads((ph / "encode" / "fx" / "parity.json").read_text())
    res = {"phases_dir": str(ph), "single_step_bundle": str(ss), "one_step": [], "ok": True}
    with torch.no_grad():
        for i, case in enumerate(parity["cases"]):
            ins = [fx_ss[k] for k in case["inputs"]]
            cond = enc(*ins[:4])
            out = stp(cond, ins[4], ins[5])
            want = fx_ss[case["outputs"][0]]
            enc_want = fx_enc[enc_parity["cases"][i]["outputs"][0]]
            same = single(*ins)
            row = {"case": i, "encode_output_shape": list(cond.shape), "encode_equals_encode_bundle_fixture": _bytes_equal(cond, enc_want),
                   "composition_equals_single_step_graph_same_process": _bytes_equal(out, same),
                   "max_abs_diff_vs_single_step_graph": _diff(out, same),
                   "composition_equals_recorded_fixture": _bytes_equal(out, want),
                   "single_step_graph_equals_recorded_fixture": _bytes_equal(same, want),
                   "max_abs_diff_vs_recorded_fixture": _diff(out, want), "output_shape": list(out.shape), "dtype": str(out.dtype)}
            row["manifest_cond_shape_matches"] = row["encode_output_shape"] == manifest["graphs"]["encode"]["outputs"]["conditioning_vec"]["shape"]
            res["one_step"].append(row)
            res["ok"] &= row["composition_equals_single_step_graph_same_process"] and row["manifest_cond_shape_matches"]
        if a.eager_loop:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from policy_fx_capture_phases import load_dit

            policy, _ = load_dit(a.eager_loop)
            ins = [fx_ss[k] for k in parity["cases"][0]["inputs"]][:4]
            cond_eager = policy.observation_encoder.encode(
                __import__("policy_fx_capture_phases")._batch(*ins))
            torch.manual_seed(a.seed)
            eager = policy.objective.conditional_sample(policy.noise_predictor, 1, cond_eager)
            graph, calls = host_loop(stp, enc(*ins), manifest, a.seed)
            res["host_loop_vs_eager_conditional_sample"] = {
                "seed": a.seed, "step_graph_calls": calls, "encode_graph_calls": 1,
                "bytes_equal": _bytes_equal(graph, eager), "max_abs_diff": _diff(graph, eager),
                "shape": list(graph.shape), "dtype": str(graph.dtype)}
            res["ok"] &= res["host_loop_vs_eager_conditional_sample"]["bytes_equal"]
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
