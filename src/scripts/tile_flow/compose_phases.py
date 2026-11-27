"""Compose the per-phase DiT graphs the way a host would and compare with the existing single-step bundle (and, optionally, with the
eager 100-step sampler). Uses only torch.export.load on the two saved graphs plus the loop manifest; no model class is imported
for the one-step comparison.

  (a) one step: step(encode(obs), noisy_actions, timestep) against (1) the single-step bundle's graph (fx/model.pt2) run in the same
      process on the same inputs and (2) that bundle's recorded fixture output, byte for byte, and encode's output against the
      encode bundle's own fixture. Only (1) decides `ok`: a recorded fixture is a different process's run, and the eager original
      boundary can itself differ from it on this host (see docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).
  (b) --eager-loop EXPORT_DIR: the host loop of loop_manifest.json (100 steps, DDPM arithmetic from the per_step table, variance noise
      from the global RNG in the same order diffusers draws it) over the step graph, against DiffusionObjective.conditional_sample
      of the eagerly rebuilt quantized policy with the same seed.
  Hoisted DiT layout (the manifest has a `modulate` graph): (a) step(x, modulate(encode(obs), timesteps)[:, i]) for the case's
  timestep i, against the single-step graph (same process) or, when that bundle has no graph, its recorded fixture;
  (b) the hoisted host loop (encode once, modulate once, step 100 times reading modulation[:, i]) against conditional_sample.
  Pi0 (atlas-pi0 env): (a) prefix, then step with time 1.0, then x = noise + dt * v with dt = -1 (the single-step bundle is
  sample_actions(num_steps=1)) against that bundle's graph in the same process and against its fixture; (b) with --eager-loop the 10-step
  host loop of the manifest (euler.times, float32 dt) against model.sample_actions(num_steps=10) on the same noise.
  SmolVLA (atlas-lerobot env; no single-step bundle exists, the two positional bundles are the prefix and the denoise_step bundle
  of smolvla_phase_capture.py): (a) per case of host_cases.safetensors, the host inputs rebuilt from the observation and the
  manifest alone against the bundles' recorded inputs, the prefix graph's K / V against the manifest (count, per-tensor shape,
  dtype) and its fixture, step purity (same inputs twice; the K / V inputs unchanged), the step graph on the step bundle's fixture
  inputs, and the 10-step host loop (euler.times, Python float dt as euler_integrate); (b) with --eager-loop the eager quantized
  policy rebuilt from the export, in this process: lerobot's own prefill (boolean masks, DynamicCache) against the prefix graph,
  denoise_step on the eager trajectory against the step graph at every step, and the host loop against sample_actions with the
  same noise, byte for byte (else the max abs difference); --unpadded adds a case with an all-true language mask.
usage (PYTHONPATH=<worktree>/src):
  compose_phases.py dit PHASES_DIR SINGLE_STEP_BUNDLE OUT_JSON [--eager-loop EXPORT_DIR] [--seed N]            # atlas-lerobot env
  compose_phases.py pi0 PHASES_DIR SINGLE_STEP_BUNDLE OUT_JSON [--eager-loop EXPORT_DIR --openpi-dir D --checkpoint-dir D]  # atlas-pi0 env
  compose_phases.py smolvla PREFIX_BUNDLE DENOISE_STEP_BUNDLE OUT_JSON [--eager-loop EXPORT_DIR] [--unpadded]   # atlas-lerobot env
  compose_phases.py causal_lm PREFILL_BUNDLE STEP_BUNDLE OUT_JSON [--decode-steps N] [--reference-cache F]   # causal LM token loop
  (see compose_causal)"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  registers atlas.* ops for torch.export.load
from exporter.fx_export import tensor_record


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
    one_step_out = {}
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
            one_step_out[i] = out
            res["ok"] &= row["composition_equals_single_step_graph_same_process"] and row["kv_shapes_match_manifest"]
        if a.eager_loop:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            sys.path.insert(0, str(ss))
            import pi0_phase_capture as ppc

            model, _ = ppc.load_pi0(a.eager_loop, a.openpi_dir, a.checkpoint_dir, a.tokenizer_path)
            loops = []
            want_cases = range(len(parity["cases"])) if a.eager_cases is None else a.eager_cases
            for i, case in enumerate(parity["cases"]):
                if i not in want_cases:
                    continue
                ins = [fx_ss[k] for k in case["inputs"]]
                obs = ppc.SimpleNamespace(
                    images=dict(zip(ppc.KEYS, ins[:3])), image_masks=dict(zip(ppc.KEYS, ins[3:6])), state=ins[8],
                    tokenized_prompt=ins[6], tokenized_prompt_mask=ins[7], token_ar_mask=torch.zeros_like(ins[7]),
                    token_loss_mask=torch.zeros_like(ins[7]))
                t0 = time.time()
                eager = model.sample_actions("cpu", obs, noise=ins[9], num_steps=manifest["denoising_steps"])
                t_eager = time.time() - t0
                t0 = time.time()
                pad, *kv = pre(*ins[:8])
                graph, calls = pi0_host_loop(stp, ins[8], pad, ins[9], kv, manifest)
                t_graph = time.time() - t0
                neq = int((graph != eager).sum())
                loops.append({"case": i, "step_graph_calls": calls, "prefix_graph_calls": 1,
                              "bytes_equal": _bytes_equal(graph, eager), "max_abs_diff": _diff(graph, eager),
                              "differing_elements": neq, "elements": graph.numel(),
                              "eager_seconds": round(t_eager, 1), "composed_loop_seconds": round(t_graph, 1),
                              "shape": list(graph.shape), "dtype": str(graph.dtype)})
                if a.cache_dir:
                    Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
                    torch.save({"eager": eager, "composed": graph}, Path(a.cache_dir) / f"loop_case{i}.pt")
                if not loops[-1]["bytes_equal"]:
                    # step-1 localisation: eager sample_actions(num_steps=1) vs the composed one-step graphs
                    e1 = model.sample_actions("cpu", obs, noise=ins[9], num_steps=1)
                    loops[-1]["step1_check"] = {
                        "composed_one_step_vs_eager_num_steps_1_max_abs": _diff(one_step_out[i], e1),
                        "differing_elements": int((one_step_out[i] != e1).sum()),
                        "bytes_equal": _bytes_equal(one_step_out[i], e1)}
                res["ok"] &= loops[-1]["bytes_equal"]
                Path(a.out).write_text(json.dumps(res, indent=1))   # partial result survives a kill
                print(json.dumps(loops[-1], indent=1), flush=True)
            res["host_loop_vs_eager_sample_actions"] = loops
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


def smolvla_host_loop(step, noise, host, kv, manifest, geometry=None):
    """The manifest's Euler loop, as euler_integrate runs it: time = full([batch], t, float32) for t in euler.times,
    x = x + dt * v with the Python float dt (with host_time_embedding the host also computes each time's embedding)."""
    from exporter.fx_export import smolvla_step_inputs

    host_time = manifest["policy"].get("host_time_embedding", False)
    dt, x, calls = manifest["euler"]["dt"], noise, 0
    for t in manifest["euler"]["times"]:
        tt = torch.full((x.shape[0],), t, dtype=torch.float32)
        x = x + dt * step(*smolvla_step_inputs(geometry, x, tt, host, kv, host_time))
        calls += 1
    return x, calls


def smolvla_manifest_geometry(manifest):
    """The host-input geometry (fx_export.smolvla_geometry's fields) as the loop manifest records it."""
    from types import SimpleNamespace

    pol = manifest["policy"]
    te = pol.get("time_embedding") or {}
    return SimpleNamespace(tokens_per_image=pol["tokens_per_image"], chunk_size=pol["chunk_size"], head_dim=pol["head_dim"],
                           time_dim=te.get("time_dim"), min_period=te.get("min_period"), max_period=te.get("max_period"),
                           time_dtype={"F32": torch.float32, "BF16": torch.bfloat16}.get(te.get("dtype")))


def _kv_matches_manifest(kv, manifest):
    spec = manifest["graphs"]["prefix"]["outputs"]["kv_cache"]
    dtype = {"BF16": torch.bfloat16}[spec["dtype"]]
    rows = [{"name": name, "role": spec["layer_roles"][i // 2], "shape": list(t.shape), "dtype": str(t.dtype),
             "ok": list(t.shape) == spec["per_tensor_shape"] and t.dtype == dtype}
            for i, (name, t) in enumerate(zip(spec["names"], kv))]
    return len(kv) == spec["tensors"] and all(r["ok"] for r in rows), rows


def _graph_census(program):
    """Graph-level checks of a saved phase graph: dead pure nodes, symbolic shapes, softmax mask forms, RoPE / cumsum /
    float64 nodes, input mutation."""
    from exporter.fx_export import graph_shape_census, softmax_mask_forms

    calls = [n for n in program.graph.nodes if n.op == "call_function"]
    forms = softmax_mask_forms(program.graph)
    return {"dead_pure_nodes": [n.name for n in calls if not n.users and not n.is_impure() and "_unused_" not in n.name],
            "dynamic_nodes": graph_shape_census(program.graph)["dynamic_nodes"],
            "softmax_mask_forms": dict(__import__("collections").Counter(r["form"] for r in forms)),
            "rope_trig_nodes": len([n for n in calls if str(n.target) in ("aten.sin.default", "aten.cos.default")]),
            "cumsum_nodes": len([n for n in calls if "cumsum" in str(n.target)]),
            "float64_nodes": [n.name for n in calls if getattr(n.meta.get("val"), "dtype", None) == torch.float64],
            "user_inputs_mutated": dict(program.graph_signature.user_inputs_to_mutate),
            "nodes": len(program.graph.nodes)}


def compose_smolvla(a):
    from types import SimpleNamespace

    from safetensors.torch import load_file

    import exporter.fx_export as fx_export

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import smolvla_phase_capture as spc

    pb, sb = Path(a.phases_dir), Path(a.single_step_bundle)
    manifest = json.loads((pb / "loop_manifest.json").read_text())
    if (pb / "loop_manifest.json").read_bytes() != (sb / "loop_manifest.json").read_bytes():
        raise SystemExit("the two bundles carry different loop manifests")
    pol = manifest["policy"]
    geometry = smolvla_manifest_geometry(manifest)
    host_rope = pol["host_rope"]
    host_time = pol.get("host_time_embedding", False)
    embed = None
    if pol.get("host_text_embeddings"):          # the host gathers from the export's own table (the manifest's recipe)
        from safetensors import safe_open

        with safe_open(str(pb / "export" / "model.safetensors"), "pt") as f:
            embed = fx_export.smolvla_token_embedder(f.get_tensor(fx_export.SMOLVLA_EMBED_TOKENS))
    pre_program = torch.export.load(pb / "fx" / "model.pt2")
    stp_program = torch.export.load(sb / "fx" / "model.pt2")
    pre, stp = pre_program.module(), stp_program.module()
    fx_pre, fx_stp = load_file(str(pb / "fx" / "fixtures.safetensors")), load_file(str(sb / "fx" / "fixtures.safetensors"))
    par_pre = json.loads((pb / "fx" / "parity.json").read_text())["cases"]
    par_stp = json.loads((sb / "fx" / "parity.json").read_text())["cases"]
    cases, meta = spc.load_host_cases(pb)
    if (sb / "host_cases.safetensors").read_bytes() != (pb / "host_cases.safetensors").read_bytes():
        raise SystemExit("the two bundles were captured on different observations")
    res = {"prefix_bundle": str(pb), "denoise_step_bundle": str(sb), "host_rope": host_rope,
           "host_text_embeddings": embed is not None, "host_time_embedding": host_time,
           "graphs": {"prefix": _graph_census(pre_program), "denoise_step": _graph_census(stp_program)},
           "cases": [], "ok": True}
    composed = {}
    with torch.no_grad():
        for i, (images, img_masks, lang_tokens, lang_masks, state, noise) in enumerate(cases):
            ins, pad = fx_export.smolvla_prefix_inputs(geometry, images, img_masks, lang_tokens, lang_masks, state, host_rope,
                                                       embed)
            want_ins = [fx_pre[k] for k in par_pre[i]["inputs"]]
            kv = pre(*ins)
            kv_ok, kv_rows = _kv_matches_manifest(kv, manifest)
            host = fx_export.smolvla_step_host_inputs(geometry, pad, host_rope)
            before = [tensor_record(t) for t in kv]
            t0 = torch.full((1,), manifest["euler"]["times"][0], dtype=torch.float32)
            step_in = fx_export.smolvla_step_inputs(geometry, noise, t0, host, kv, host_time)
            v1, v2 = stp(*step_in), stp(*step_in)
            pure = _bytes_equal(v1, v2) and [tensor_record(t) for t in kv] == before
            s_ins = [fx_stp[k] for k in par_stp[i]["inputs"]]
            n_host, n_time = len(host), int(host_time)
            s_time_ok = all(_bytes_equal(x, y) for x, y in zip(
                fx_export.smolvla_time_inputs(geometry, s_ins[1], host_time), s_ins[2:2 + n_time]))
            s_host_ok = s_time_ok and all(_bytes_equal(x, y) for x, y in zip(host, s_ins[2 + n_time:2 + n_time + n_host]))
            s_kv_ok = all(_bytes_equal(x, y) for x, y in zip(kv, s_ins[2 + n_time + n_host:]))
            s_out = stp(*s_ins)
            t1 = time.time()
            actions, calls = smolvla_host_loop(stp, noise, host, kv, manifest, geometry)
            row = {"case": i, **meta["observations"][i], "padded_language_tokens":
                   meta["observations"][i]["language_tokens"] - meta["observations"][i]["valid_language_tokens"],
                   "host_prefix_inputs_equal_recorded": len(ins) == len(want_ins) and all(
                       _bytes_equal(x, y) for x, y in zip(ins, want_ins)),
                   "kv_tensors": len(kv), "kv_matches_manifest": kv_ok,
                   "kv_roles": dict(__import__("collections").Counter(r["role"] for r in kv_rows[::2])),
                   "prefix_graph_equals_recorded_fixture": all(
                       _bytes_equal(x, fx_pre[k]) for x, k in zip(kv, par_pre[i]["outputs"])),
                   "step_pure_same_inputs_twice": _bytes_equal(v1, v2),
                   "step_kv_inputs_unchanged": [tensor_record(t) for t in kv] == before,
                   "step_bundle_host_inputs_equal_rebuilt": s_host_ok, "step_bundle_kv_equal_prefix_graph": s_kv_ok,
                   "step_graph_equals_recorded_fixture": _bytes_equal(s_out, fx_stp[par_stp[i]["outputs"][0]]),
                   "host_loop_step_calls": calls, "host_loop_seconds": round(time.time() - t1, 1),
                   "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype)}
            if i == 0:
                res["kv_per_tensor"] = kv_rows
            res["cases"].append(row)
            composed[i] = (actions, kv, host, pad)
            res["ok"] &= (row["host_prefix_inputs_equal_recorded"] and kv_ok and pure and s_host_ok and s_kv_ok
                          and row["step_graph_equals_recorded_fixture"] and row["prefix_graph_equals_recorded_fixture"])
            Path(a.out).write_text(json.dumps(res, indent=1))
            print(json.dumps(row), flush=True)
        if a.eager_loop:
            res["eager"] = _smolvla_eager(a, spc, fx_export, pre, stp, cases, composed, manifest, geometry, host_rope, res,
                                          embed)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "kv_per_tensor"}, indent=1))
    return 0 if res["ok"] else 1


def _smolvla_eager(a, spc, fx_export, pre, stp, cases, composed, manifest, geometry, host_rope, res, embed=None):
    """Checks 1, 4 and 6 against lerobot's own inference path on the eager quantized policy (no boundary is constructed
    here, so apply_rope is lerobot's)."""
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    policy, _, _, _ = spc.load_policy(a.eager_loop)
    model = policy.model
    times, dt = manifest["euler"]["times"], manifest["euler"]["dt"]
    work = [(i, c, composed[i]) for i, c in enumerate(cases)]
    if a.unpadded:
        images, img_masks, lang_tokens, lang_masks, state, noise = cases[0]
        c = (images, img_masks, lang_tokens, torch.ones_like(lang_masks), state, noise)
        ins, pad = fx_export.smolvla_prefix_inputs(geometry, *c[:5], host_rope, embed)
        kv = pre(*ins)
        host = fx_export.smolvla_step_host_inputs(geometry, pad, host_rope)
        work.append(("unpadded_case_0", c, (smolvla_host_loop(stp, noise, host, kv, manifest, geometry)[0], kv, host, pad)))
    rows = []
    for i, (images, img_masks, lang_tokens, lang_masks, state, noise), (actions, kv, host, pad) in work:
        t0 = time.time()
        embs, e_pad, e_att = model.embed_prefix(list(images), img_masks, lang_tokens, lang_masks, state=state)
        _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(e_pad, e_att),
                                                 position_ids=torch.cumsum(e_pad, dim=1) - 1, past_key_values=None,
                                                 inputs_embeds=[embs, None], use_cache=True)
        eager_kv = [t for layer in cache.layers for t in (layer.keys.transpose(1, 2), layer.values.transpose(1, 2))]
        prefix_diff = max(_diff(x, y) for x, y in zip(kv, eager_kv))
        x, steps = noise, []
        for t in times:
            tt = torch.full((x.shape[0],), t, dtype=torch.float32)
            v_e = model.denoise_step(e_pad, cache, x, tt)
            v_g = stp(*fx_export.smolvla_step_inputs(geometry, x, tt, host, kv,
                                                     manifest["policy"].get("host_time_embedding", False)))
            steps.append({"t": t, "bytes_equal": _bytes_equal(v_g, v_e), "max_abs_diff": _diff(v_g, v_e)})
            x = x + dt * v_e
        t1 = time.time()
        eager = model.sample_actions(list(images), img_masks, lang_tokens, lang_masks, state, noise=noise)
        row = {"case": i, "padded_language_tokens": int((~lang_masks.bool()).sum()),
               "prefix_masks_equal_embed_prefix": torch.equal(pad, e_pad),
               "prefix_kv_bytes_equal_lerobot_prefill": all(_bytes_equal(x, y) for x, y in zip(kv, eager_kv)),
               "prefix_kv_max_abs_diff": prefix_diff,
               "step_bytes_equal_denoise_step_all_steps": all(r["bytes_equal"] for r in steps),
               "step_max_abs_diff": max(r["max_abs_diff"] for r in steps), "per_step": steps,
               "eager_loop_equals_sample_actions": _bytes_equal(x, eager),
               "host_loop_bytes_equal_sample_actions": _bytes_equal(actions, eager),
               "host_loop_max_abs_diff": _diff(actions, eager),
               "differing_elements": int((actions != eager).sum()), "elements": actions.numel(),
               "eager_seconds": round(time.time() - t1, 1), "reference_seconds": round(t1 - t0, 1)}
        rows.append(row)
        res["ok"] &= (row["prefix_kv_bytes_equal_lerobot_prefill"] and row["step_bytes_equal_denoise_step_all_steps"]
                      and row["host_loop_bytes_equal_sample_actions"] and row["prefix_masks_equal_embed_prefix"])
        if a.cache_dir:
            Path(a.cache_dir).mkdir(parents=True, exist_ok=True)
            torch.save({"eager": eager, "composed": actions}, Path(a.cache_dir) / f"smolvla_loop_case{i}.pt")
        res["eager"] = rows
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in row.items() if k != "per_step"}), flush=True)
    return rows


def compose_dit_hoisted(a):
    """Hoisted DiT layout (encode, modulate, step): (a) one step, step(x, modulate(encode(obs), timesteps)[:, i]) with i the
    case's timestep, against the single-step bundle's graph in the same process when it has one (else against its recorded
    fixture), plus the modulate output shape against the manifest; (b) --eager-loop: the host loop of the manifest over the
    three graphs against eager conditional_sample."""
    from safetensors.torch import load_file

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import policy_fx_capture_phases as cap

    ph, ss = Path(a.phases_dir), Path(a.single_step_bundle)
    manifest = json.loads((ph / "loop_manifest.json").read_text())
    enc = torch.export.load(ph / "encode" / "fx" / "model.pt2").module()
    mod = torch.export.load(ph / "modulate" / "fx" / "model.pt2").module()
    stp = torch.export.load(ph / "step" / "fx" / "model.pt2").module()
    fx_ss = load_file(str(ss / "fx" / "fixtures.safetensors"))
    parity = json.loads((ss / "fx" / "parity.json").read_text())
    single = torch.export.load(ss / "fx" / "model.pt2").module() if (ss / "fx" / "model.pt2").exists() else None
    ts = cap.manifest_timesteps(manifest)
    want_shape = manifest["graphs"]["modulate"]["outputs"]["modulation"]["shape"]
    res = {"phases_dir": str(ph), "single_step_bundle": str(ss), "layout": "hoisted",
           "reference": "single-step graph, same process" if single is not None else "single-step bundle's recorded fixture",
           "one_step": [], "ok": True}
    with torch.no_grad():
        for i, case in enumerate(parity["cases"]):
            ins = [fx_ss[k] for k in case["inputs"]]
            m = mod(enc(*ins[:4]), ts)
            row_i = int((ts == int(ins[5].reshape(-1)[0])).nonzero()[0])
            out = stp(ins[4], m[:, row_i].contiguous())
            want = fx_ss[case["outputs"][0]]
            row = {"case": i, "timestep": int(ins[5].reshape(-1)[0]), "modulation_shape": list(m.shape),
                   "modulation_shape_matches_manifest": list(m.shape) == want_shape,
                   "composition_equals_recorded_fixture": _bytes_equal(out, want),
                   "max_abs_diff_vs_recorded_fixture": _diff(out, want)}
            if single is not None:
                same = single(*ins)
                row["composition_equals_single_step_graph_same_process"] = _bytes_equal(out, same)
                row["max_abs_diff_vs_single_step_graph"] = _diff(out, same)
            res["one_step"].append(row)
            res["ok"] &= row["modulation_shape_matches_manifest"] and (
                row["composition_equals_single_step_graph_same_process"] if single is not None
                else row["composition_equals_recorded_fixture"])
        if a.eager_loop:
            # every case (or --eager-cases), each under its recorded sampling seed when it has one (dit_hoist_check.py
            # --write-cases records the evaluation's seed + frame), else --seed
            policy, _ = cap.load_dit(a.eager_loop)
            info = manifest["scheduler"]["initial_noise"]
            clip = manifest["scheduler"]["config"]["clip_sample_range"]
            loops = []
            for i, case in enumerate(parity["cases"]):
                if a.eager_cases is not None and i not in a.eager_cases:
                    continue
                seed = case.get("seed", a.seed)
                ins = [fx_ss[k] for k in case["inputs"]][:4]
                cond_eager = policy.observation_encoder.encode(cap._batch(*ins))
                torch.manual_seed(seed)
                eager = policy.objective.conditional_sample(policy.noise_predictor, 1, cond_eager)
                torch.manual_seed(seed)
                x = torch.randn(info["shape"], dtype=getattr(torch, {"BF16": "bfloat16", "F32": "float32"}[info["dtype"]]))
                graph, calls = cap.hoisted_host_loop(
                    stp, mod(enc(*ins), ts), x, manifest["scheduler"]["per_step"],
                    lambda row, eps: torch.randn(eps.shape, dtype=eps.dtype) if row["variance_noise_draw"] else None,
                    lambda row, x, eps, z: ddpm_host_step(row, x, eps, z, clip))
                loops.append({"case": i, "frame": case.get("frame"), "seed": seed, "step_graph_calls": calls,
                              "encode_graph_calls": 1, "modulate_graph_calls": 1,
                              "bytes_equal": _bytes_equal(graph, eager), "max_abs_diff": _diff(graph, eager),
                              "differing_elements": int((graph != eager).sum()), "elements": graph.numel(),
                              "shape": list(graph.shape), "dtype": str(graph.dtype)})
                res["ok"] &= loops[-1]["bytes_equal"]
                res["host_loop_vs_eager_conditional_sample"] = loops
                Path(a.out).write_text(json.dumps(res, indent=1))       # partial result survives a kill
                print(json.dumps(loops[-1]), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0 if res["ok"] else 1


def _causal_graph_census(program):
    """Graph-level checks of a saved phase graph: symbolic shapes, RoPE trig, embedding lookups, float64 values, input
    mutation, node counts."""
    import collections

    calls = [n for n in program.graph.nodes if n.op == "call_function"]
    symbolic = [n.name for n in program.graph.nodes if any(
        not isinstance(d, int) for d in getattr(n.meta.get("val"), "shape", ()))]
    return {"symbolic_shape_nodes": symbolic,
            "rope_trig_nodes": [n.name for n in calls if str(n.target) in ("aten.sin.default", "aten.cos.default")],
            "embedding_nodes": [n.name for n in calls if str(n.target) == "aten.embedding.default"],
            "float64_nodes": [n.name for n in calls if getattr(n.meta.get("val"), "dtype", None) == torch.float64],
            "user_inputs_mutated": dict(program.graph_signature.user_inputs_to_mutate),
            "atlas_ops": dict(collections.Counter(str(n.target) for n in calls if str(n.target).startswith("atlas."))),
            "nodes": len(program.graph.nodes)}


def _words_differing(a, b):
    """Logit words that differ (a: the graph's BF16 logits, b: generate's float32 record of the same BF16 values)."""
    a = a.float()
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def compose_causal(a):
    """Causal-LM token loop (fx_export.CausalPrefill / CausalDecodeStep bundles, loop_manifest.json "causal_lm"):
      (1) each saved graph replays its recorded fixture cases word for word, from host inputs rebuilt from the prompts;
      (2) per prompt, the host loop (causal_host_loop: prefill once, decode_steps greedy steps, fixed-capacity cache) over
          the saved graphs against transformers' generate(do_sample=False) on the quantized model rebuilt from the export,
          with a StaticCache of the manifest's capacity (decides `ok`): tokens and every logit word; with
          --dynamic-reference-cache also against generate's default DynamicCache loop (reported, not decided on: its
          attention contracts over L + 1 keys, the graphs' over C, and the float32 sums can round differently);
      (3) the step is pure (same inputs twice, inputs unchanged; no mutated user input in the signature);
      (4) the K / V shapes and dtypes against the manifest; graph census."""
    from safetensors.torch import load_file

    from exporter.quant_export import save_file

    import exporter.fx_export as fx_export
    import exporter.load_quantized as load_quantized

    pb, sb = Path(a.phases_dir), Path(a.single_step_bundle)
    if (pb / "loop_manifest.json").read_bytes() != (sb / "loop_manifest.json").read_bytes():
        raise SystemExit("the two bundles carry different loop manifests")
    if (pb / "host_cases.safetensors").read_bytes() != (sb / "host_cases.safetensors").read_bytes():
        raise SystemExit("the two bundles were captured on different prompts")
    manifest = json.loads((pb / "loop_manifest.json").read_text())
    lm, kv_spec, loop = manifest["causal_lm"], manifest["kv_cache"], manifest["loop"]
    n_steps = a.decode_steps if a.decode_steps is not None else loop["decode_steps"]
    prompts = load_file(str(pb / "host_cases.safetensors"))["prompts"]
    indices = list(range(prompts.shape[0]))[:a.prompts] if a.prompt_indices is None else a.prompt_indices
    # seqlen: the context the loop needs (build_causal_lm adds linear RoPE scaling only beyond the model's own context)
    model, _, _ = load_quantized.load_quantized(lm["id"], pb / "export", "cpu", mode="ref_conv", seqlen=kv_spec["capacity"])
    geometry = fx_export.causal_geometry(model, kv_spec["capacity"], manifest["graphs"]["prefill"]["prefill_length"])
    geometry_ok = (geometry.layers, geometry.n_kv_heads, geometry.head_dim) == (lm["layers"], lm["n_kv_heads"], lm["head_dim"])
    rope = fx_export.causal_rope_tables(model, geometry.capacity)
    embed = None
    if lm["host_embeddings"]:
        tensors, policy, _ = load_quantized.read_export(pb / "export")
        embed = fx_export.causal_token_embedder(tensors, policy)
        del tensors
    programs = {name: torch.export.load(b / "fx" / "model.pt2") for name, b in (("prefill", pb), ("decode_step", sb))}
    prefill, step = programs["prefill"].module(), programs["decode_step"].module()
    res = {"prefill_bundle": str(pb), "decode_step_bundle": str(sb), "host_embeddings": lm["host_embeddings"],
           "head_embed": lm["head_embed"].get("options"), "capacity": geometry.capacity,
           "prefill_length": geometry.prefill_length, "decode_steps": n_steps, "geometry_matches_manifest": geometry_ok,
           "torch_threads": torch.get_num_threads(), "graphs": {k: _causal_graph_census(v) for k, v in programs.items()},
           "fixtures": [], "prompts": [], "ok": geometry_ok}
    res["ok"] &= all(not c["symbolic_shape_nodes"] and not c["rope_trig_nodes"] and not c["user_inputs_mutated"]
                     and bool(c["embedding_nodes"]) == (not lm["host_embeddings"]) for c in res["graphs"].values())
    ref_cache = Path(a.reference_cache) if a.reference_cache else None
    refs = load_file(str(ref_cache)) if ref_cache and ref_cache.is_file() else {}
    dyn_cache = Path(a.dynamic_reference_cache) if a.dynamic_reference_cache else None
    dyn_refs = load_file(str(dyn_cache)) if dyn_cache and dyn_cache.is_file() else {}
    res["library_loop"] = f"generate(do_sample=False) on StaticCache(max_cache_len={geometry.capacity})"
    with torch.no_grad():
        if embed is not None:
            res["host_embeddings_equal_embed_tokens"] = _bytes_equal(embed(prompts), model.model.embed_tokens(prompts))
            res["ok"] &= res["host_embeddings_equal_embed_tokens"]
        res["prompt_indices"] = indices
        # (1) recorded fixtures, replayed in this process, from rebuilt host inputs
        for name, b, program in (("prefill", pb, prefill), ("decode_step", sb, step)):
            fixtures = load_file(str(b / "fx" / "fixtures.safetensors"))
            for i, case in enumerate(json.loads((b / "fx" / "parity.json").read_text())["cases"]):
                ins, outs = [fixtures[k] for k in case["inputs"]], [fixtures[k] for k in case["outputs"]]
                got = program(*ins)
                row = {"graph": name, "case": i, "outputs_equal_recorded": len(got) == len(outs) and all(
                    _bytes_equal(x.contiguous(), y) for x, y in zip(got, outs))}
                if name == "prefill":
                    rebuilt = fx_export.causal_prefill_inputs(prompts[i:i + 1], rope,
                                                              fx_export.causal_empty_cache(geometry), embed)
                    row["inputs_equal_host_rebuilt"] = len(rebuilt) == len(ins) and all(
                        _bytes_equal(x, y) for x, y in zip(rebuilt, ins))
                else:
                    length = int((ins[1] == 0).sum()) - 1
                    rebuilt = fx_export.causal_step_inputs(torch.zeros(1, 1, dtype=torch.int64), length, geometry.capacity,
                                                           rope, ins[5:], None)
                    row["cache_length"] = length
                    row["mask_rope_slot_equal_host_rebuilt"] = all(_bytes_equal(x, y) for x, y in zip(rebuilt[1:5], ins[1:5]))
                    row["unused_slots_zero"] = all(not t[:, :, length:].any() for t in ins[5:])
                    row["kv_inputs_match_manifest"] = all(list(t.shape) == kv_spec["per_tensor_shape"] and
                                                          t.dtype == torch.bfloat16 for t in ins[5:])
                res["fixtures"].append(row)
                res["ok"] &= all(v for k, v in row.items() if isinstance(v, bool))
        # (2) host loop over the saved graphs vs generate, per prompt
        for i in indices:
            prompt = prompts[i:i + 1]
            key = hashlib.sha256(prompt.numpy().tobytes()).hexdigest()[:16] + f"_{n_steps}"
            dyn_key, key = key, key + f"_static{geometry.capacity}"
            t0 = time.time()
            if f"{key}.tokens" not in refs:
                want_tokens, want_logits = fx_export.causal_generate(model, prompt, n_steps, geometry.capacity)
                refs[f"{key}.tokens"] = want_tokens.contiguous()
                refs[f"{key}.logits"] = torch.stack([t.contiguous() for t in want_logits])
                if ref_cache:
                    save_file(refs, str(ref_cache))
            want_tokens, want_logits = refs[f"{key}.tokens"], list(refs[f"{key}.logits"])
            t1 = time.time()
            tokens, logits, cache = fx_export.causal_host_loop(prefill, step, prompt, n_steps, geometry, rope, embed,
                                                               loop["eos_token_id"])
            t2 = time.time()
            diffs = [_words_differing(x, y) for x, y in zip(logits, want_logits)]
            row = {"prompt": i, "generate_seconds": round(t1 - t0, 1), "host_loop_seconds": round(t2 - t1, 1),
                   "tokens_generated": int(tokens.shape[1]), "tokens_equal_generate": torch.equal(tokens, want_tokens),
                   "logits_compared": len(diffs), "logit_words_differing_per_token": diffs,
                   "logits_byte_identical": len(logits) == len(want_logits) and not any(diffs),
                   "max_abs_diff": max(float((x.float() - y).abs().max()) for x, y in zip(logits, want_logits)),
                   "tokens": tokens[0].tolist(),
                   "kv_matches_manifest": len(cache) == kv_spec["tensors"] and all(
                       list(t.shape) == kv_spec["per_tensor_shape"] and t.dtype == torch.bfloat16 for t in cache)}
            if dyn_cache is not None:    # generate's default DynamicCache loop (reported)
                if f"{dyn_key}.tokens" not in dyn_refs:
                    dyn_tokens, dyn_logits = fx_export.causal_generate(model, prompt, n_steps)
                    dyn_refs[f"{dyn_key}.tokens"] = dyn_tokens.contiguous()
                    dyn_refs[f"{dyn_key}.logits"] = torch.stack([t.contiguous() for t in dyn_logits])
                    save_file(dyn_refs, str(dyn_cache))
                dyn_tokens, dyn_logits = dyn_refs[f"{dyn_key}.tokens"], list(dyn_refs[f"{dyn_key}.logits"])
                dd = [_words_differing(x, y) for x, y in zip(logits, dyn_logits)]
                st = [_words_differing(x.to(torch.bfloat16), y) for x, y in zip(want_logits, dyn_logits)]
                row["dynamic_cache_generate"] = {
                    "tokens_equal": torch.equal(tokens, dyn_tokens), "logit_words_differing_per_token": dd,
                    "first_differing_token": next((j for j, x in enumerate(dd) if x), None),
                    "max_abs_diff": max(float((x.float() - y).abs().max()) for x, y in zip(logits, dyn_logits)),
                    "static_generate_vs_dynamic_generate_first_differing_token": next(
                        (j for j, x in enumerate(st) if x), None)}
            if i == indices[0]:          # (3) purity of the step on the last cache state of this loop
                args = fx_export.causal_step_inputs(tokens[:, -1:], geometry.prefill_length + tokens.shape[1] - 1,
                                                    geometry.capacity, rope, cache, embed)
                before = [tensor_record(t) for t in args]
                first, second = step(*args), step(*args)
                res["step_pure_same_inputs_twice"] = all(_bytes_equal(x, y) for x, y in zip(first, second))
                res["step_inputs_unchanged"] = [tensor_record(t) for t in args] == before
                res["ok"] &= res["step_pure_same_inputs_twice"] and res["step_inputs_unchanged"]
            res["prompts"].append(row)
            res["ok"] &= row["tokens_equal_generate"] and row["logits_byte_identical"] and row["kv_matches_manifest"]
            Path(a.out).write_text(json.dumps(res, indent=1))
            print(json.dumps({k: v for k, v in row.items() if k != "tokens"}), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("prompts", "fixtures")}, indent=1))
    return 0 if res["ok"] else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", choices=("dit", "pi0", "smolvla", "causal_lm"))
    ap.add_argument("phases_dir", help="dit / pi0: the phases directory; smolvla: the prefix bundle; causal_lm: the prefill bundle")
    ap.add_argument("single_step_bundle", help="dit / pi0: the single-step bundle; smolvla: the denoise_step bundle; "
                                               "causal_lm: the decode-step bundle")
    ap.add_argument("out")
    ap.add_argument("--eager-loop", metavar="EXPORT_DIR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eager-cases", type=int, nargs="*", default=None, help="only these case indices for the eager loop")
    ap.add_argument("--cache-dir", default=None, help="pi0: save eager and composed loop outputs here")
    ap.add_argument("--openpi-dir")
    ap.add_argument("--checkpoint-dir")
    ap.add_argument("--tokenizer-path", default=os.environ.get("PALIGEMMA_TOKENIZER"),
                    help="pi0 --eager-loop: paligemma_tokenizer.model (default $PALIGEMMA_TOKENIZER)")
    ap.add_argument("--unpadded", action="store_true", help="smolvla: also an all-true language mask case (eager loop)")
    ap.add_argument("--decode-steps", type=int, default=None, help="causal_lm: decode steps (default: the manifest's)")
    ap.add_argument("--prompts", type=int, default=None, help="causal_lm: only the first N prompts of host_cases")
    ap.add_argument("--prompt-indices", type=int, nargs="+", default=None, help="causal_lm: only these prompts of host_cases")
    ap.add_argument("--reference-cache", help="causal_lm: safetensors file caching generate's tokens / logits per prompt")
    ap.add_argument("--dynamic-reference-cache",
                    help="causal_lm: also compare with generate's default DynamicCache loop (cached / computed here)")
    a = ap.parse_args(argv)
    if a.model == "pi0" and a.eager_loop and not a.tokenizer_path:
        ap.error("pi0 --eager-loop needs --tokenizer-path (or $PALIGEMMA_TOKENIZER)")
    # the loop manifest names its model family: refuse a bundle of another family (e.g. a causal-LM bundle under
    # `smolvla` or `dit`, a SmolVLA or DiT bundle under `causal_lm`); Pi0's manifest records a dict there
    family = {"dit": "multi_task_dit", "smolvla": "smolvla", "causal_lm": "causal_lm"}
    loop_path = Path(a.phases_dir) / "loop_manifest.json"
    recorded = json.loads(loop_path.read_text()).get("model") if loop_path.is_file() else None
    if isinstance(recorded, str) and recorded != family.get(a.model, "pi0"):
        raise SystemExit(f"{loop_path} is a {recorded!r} loop manifest, not {a.model}")
    if a.model == "pi0":
        return compose_pi0(a)
    if a.model == "smolvla":
        return compose_smolvla(a)
    if a.model == "causal_lm":
        return compose_causal(a)
    if "modulate" in json.loads((Path(a.phases_dir) / "loop_manifest.json").read_text())["graphs"]:
        return compose_dit_hoisted(a)
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
