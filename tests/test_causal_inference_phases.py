"""Causal-LM inference phases (fx_export.CausalPrefill / CausalDecodeStep / CausalDecodeLayer) on a tiny Llama exported with
today's static recipe (test_head_embed._export), with the default BF16 head and embedding and with the FP8 head and FP8
embedding (--lm-head fp8 --embedding fp8):
  - the prefill's logits and K / V are transformers' own prefill (DynamicCache), the decode step's are transformers' own step
    on a DynamicCache of the same contents, and the host loop over the two (greedy argmax, fixed-capacity cache, host mask,
    RoPE rows and embeddings) is generate(do_sample=False) token for token and logit word for logit word;
  - the host inputs are transformers' (create_causal_mask, rotary_emb, embed_tokens), the step is pure, the capacity only
    changes masked slots (results identical across capacities), and the per-layer decode boundary chains to the step;
  - the captured native-FP8 graphs have no symbolic shape, no RoPE trig, no input mutation (with host embeddings no
    embedding lookup), replay the same words after save / load, and compose to generate."""
import json

import pytest
import torch
from test_fx_export import PINNED_TORCH
from test_head_embed import _export, _reload

import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers the atlas ops)
import exporter.fx_export as fx_export
import exporter.load_quantized as load_quantized

pinned = pytest.mark.skipif(torch.__version__ < PINNED_TORCH, reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")
DEFAULT = {"tie": "none", "lm_head": "bf16", "embedding": "bf16"}
FP8 = {"tie": "none", "lm_head": "fp8", "embedding": "fp8"}
P, C, N = 8, 24, 10


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _prompts(count=3, length=P, seed=11):
    return torch.randint(0, 128, (count, length), generator=torch.Generator().manual_seed(seed))


@pytest.fixture(scope="module", params=[DEFAULT, FP8], ids=["default", "head_fp8_emb_fp8"])
def exported(request, tmp_path_factory):
    _, d = _export(tmp_path_factory.mktemp("causal_phases"), request.param)
    model = _reload(d)
    tensors, policy, _ = load_quantized.read_export(d)
    return request.param, model, d, tensors, policy


def _geometry(model, capacity=C):
    return fx_export.causal_geometry(model, capacity, P)


def _host(model, tensors, policy, host_embeddings, capacity=C):
    geometry = _geometry(model, capacity)
    rope = fx_export.causal_rope_tables(model, capacity)
    embed = fx_export.causal_token_embedder(tensors, policy) if host_embeddings else None
    return geometry, rope, embed


@pytest.mark.parametrize("host_embeddings", [False, True], ids=["ids", "host_embeddings"])
def test_host_loop_is_generate(exported, host_embeddings):
    opts, model, _, tensors, policy = exported
    geometry, rope, embed = _host(model, tensors, policy, host_embeddings)
    prefill = fx_export.CausalPrefill(model, host_embeddings).eval()
    step = fx_export.CausalDecodeStep(model, host_embeddings).eval()
    with torch.no_grad():
        for prompt in _prompts():
            prompt = prompt[None]
            want_tokens, want_logits = fx_export.causal_generate(model, prompt, N)
            tokens, logits, cache = fx_export.causal_host_loop(prefill, step, prompt, N, geometry, rope, embed)
            assert torch.equal(tokens, want_tokens)
            assert len(logits) == len(want_logits) == N + 1
            assert all(got.dtype == torch.bfloat16 and _equal(got.float(), want) for got, want in zip(logits, want_logits))
            # the slots past the last written one are still zero
            assert all(not buffer[:, :, P + N:].any() for buffer in cache)


def test_prefill_is_transformers_prefill(exported):
    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prompt = _prompts(1)
    with torch.no_grad():
        out = model(prompt, use_cache=True, logits_to_keep=0)
        last = model(prompt, use_cache=True, logits_to_keep=1)          # what generate's prefill computes
        got = fx_export.CausalPrefill(model).eval()(*fx_export.causal_prefill_inputs(prompt, rope))
        every = fx_export.CausalPrefill(model, all_logits=True).eval()(*fx_export.causal_prefill_inputs(prompt, rope))
    assert _equal(every[0], out.logits)
    assert _equal(got[0], last.logits[:, -1])
    want_kv = [t for layer in out.past_key_values.layers for t in (layer.keys, layer.values)]
    assert len(got) - 1 == len(want_kv) == 2 * geometry.layers
    assert all(_equal(a, b) for a, b in zip(got[1:], want_kv))
    assert all(list(t.shape) == [1, geometry.n_kv_heads, P, geometry.head_dim] for t in got[1:])


def test_decode_step_is_transformers_step(exported):
    """One step at length L on the fixed-capacity buffers equals the model's forward of that token on a DynamicCache
    holding the same L positions: logits, and the new K / V equal the DynamicCache's appended slot."""
    from transformers import DynamicCache

    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prefill, step = fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval()
    prompt = _prompts(1)
    with torch.no_grad():
        _, _, cache = fx_export.causal_host_loop(prefill, step, prompt, 3, geometry, rope)
        length = P + 3
        token = torch.tensor([[5]])
        got = step(*fx_export.causal_step_inputs(token, length, C, rope, cache))
        dyn = DynamicCache(config=model.config)
        for i in range(geometry.layers):
            dyn.update(cache[2 * i][:, :, :length].clone(), cache[2 * i + 1][:, :, :length].clone(), i)
        out = model(token, past_key_values=dyn, use_cache=True)
    assert _equal(got[0], out.logits[:, -1])
    for i in range(geometry.layers):
        assert _equal(got[1 + 2 * i], dyn.layers[i].keys[:, :, length:])
        assert _equal(got[2 + 2 * i], dyn.layers[i].values[:, :, length:])


def test_host_inputs_are_transformers(exported):
    from transformers.masking_utils import create_causal_mask

    _, model, _, tensors, policy = exported
    rope = fx_export.causal_rope_tables(model, C)
    prompt = _prompts(1)
    embeds = model.model.embed_tokens(prompt)
    with torch.no_grad():
        mask = create_causal_mask(config=model.config, inputs_embeds=embeds, attention_mask=None, past_key_values=None,
                                  position_ids=torch.arange(P)[None])
        assert _equal(fx_export.causal_prefill_mask(P), mask)
        for position in (0, 7, C - 1):          # the row of the table is the model's own per-position computation
            cos, sin = model.model.rotary_emb(embeds[:, :1], torch.tensor([[position]]))
            assert _equal(rope[0][:, position:position + 1], cos) and _equal(rope[1][:, position:position + 1], sin)
        assert _equal(fx_export.causal_token_embedder(tensors, policy)(prompt), embeds)
    mask = fx_export.causal_decode_mask(10, C)
    assert mask.shape == (1, 1, 1, C + 1) and mask.dtype == torch.bfloat16
    assert not mask[..., :10].any() and not mask[..., C].any()
    assert (mask[..., 10:C] == torch.finfo(torch.bfloat16).min).all()
    with pytest.raises(ValueError):
        fx_export.causal_decode_mask(C, C)


def test_decode_step_is_pure_and_capacity_independent(exported):
    _, model, _, tensors, policy = exported
    prompt = _prompts(1)
    results = []
    with torch.no_grad():
        for capacity in (16, C):
            geometry, rope, _ = _host(model, tensors, policy, False, capacity)
            prefill, step = fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval()
            tokens, logits, cache = fx_export.causal_host_loop(prefill, step, prompt, 4, geometry, rope)
            args = fx_export.causal_step_inputs(tokens[:, -1:], P + 4, capacity, rope, cache)
            before = [t.clone() for t in args]
            first, second = step(*args), step(*args)
            assert all(_equal(a, b) for a, b in zip(first, second))
            assert all(_equal(a, b) for a, b in zip(args, before))
            results.append((tokens, logits, first))
    (t16, l16, s16), (t24, l24, s24) = results
    assert torch.equal(t16, t24)
    assert all(_equal(a, b) for a, b in zip(l16, l24)) and all(_equal(a, b) for a, b in zip(s16, s24))


def test_decode_layers_chain_to_the_step(exported):
    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prefill, step = fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval()
    with torch.no_grad():
        tokens, _, cache = fx_export.causal_host_loop(prefill, step, _prompts(1), 2, geometry, rope)
        token, mask, cos, sin, *kv = fx_export.causal_step_inputs(tokens[:, -1:], P + 2, C, rope, cache)
        want = step(token, mask, cos, sin, *kv)
        hidden = model.model.embed_tokens(token)
        for i in range(geometry.layers):
            hidden, k, v = fx_export.CausalDecodeLayer(model, i).eval()(hidden, cos, sin, mask, kv[2 * i], kv[2 * i + 1])
            assert _equal(k, want[1 + 2 * i]) and _equal(v, want[2 + 2 * i])
        logits = model.lm_head(model.model.norm(hidden)).view(1, -1)
    assert _equal(logits, want[0])


def test_phase_cases(exported):
    _, model, _, tensors, policy = exported
    geometry = _geometry(model)
    for boundary, count in ((fx_export.CausalPrefill(model), 4), (fx_export.CausalDecodeStep(model, True), 2),
                            (fx_export.CausalDecodeLayer(model, 1), 2)):
        cases = fx_export.causal_phase_cases(model, boundary, geometry, 0, "cpu", count=count,
                                             host_embeddings=getattr(boundary, "host_embeddings", False))
        assert len(cases) == count
        assert len({tuple(t.sum().item() for t in case) for case in cases}) == count
        if isinstance(boundary, fx_export.CausalDecodeStep):
            lengths = [int((case[1] == 0).sum()) - 1 for case in cases]
            assert lengths == [P, P + 3]
            assert cases[0][0].shape == (1, 1, geometry.hidden)
            assert all(not t[:, :, length:].any() for case, length in zip(cases, lengths) for t in case[4:])


def _census(program):
    calls = [n for n in program.graph.nodes if n.op == "call_function"]
    shapes = [n.name for n in program.graph.nodes
              if any(not isinstance(s, int) for s in getattr(getattr(n.meta.get("val"), "shape", None), "__iter__",
                                                               lambda: iter(()))())]
    return {"symbolic": shapes, "trig": [n.name for n in calls if str(n.target) in ("aten.sin.default", "aten.cos.default")],
            "embedding": [n.name for n in calls if str(n.target) == "aten.embedding.default"],
            "mutated": dict(program.graph_signature.user_inputs_to_mutate)}


@pinned
@pytest.mark.parametrize("host_embeddings", [False, True], ids=["ids", "host_embeddings"])
def test_phase_graphs_capture_and_compose_to_generate(tmp_path, exported, host_embeddings):
    opts, model, d, tensors, policy = exported
    geometry, rope, embed = _host(model, tensors, policy, host_embeddings)
    programs = {}
    for name, boundary in (("prefill", fx_export.CausalPrefill(model, host_embeddings)),
                           ("step", fx_export.CausalDecodeStep(model, host_embeddings))):
        boundary.eval()
        cases = fx_export.causal_phase_cases(model, boundary, geometry, 0, "cpu", host_embeddings=host_embeddings,
                                             embed=embed)
        report = fx_export.capture_fx(boundary, cases, tmp_path / name, policy=policy, strict=True, fresh_process=False,
                                      native_fp8="uint8")
        assert report["status"] == "passed"
        programs[name] = torch.export.load(tmp_path / name / "model.pt2")
        census = _census(programs[name])
        assert census["symbolic"] == [] and census["trig"] == [] and census["mutated"] == {}
        assert bool(census["embedding"]) == (not host_embeddings)
        sites = report["native_fp8_detail"]["sites"]
        assert ("model.lm_head" in sites) == (opts["lm_head"] == "fp8")
    with torch.no_grad():
        for prompt in _prompts(2, seed=23):
            prompt = prompt[None]
            want_tokens, want_logits = fx_export.causal_generate(model, prompt, N)
            tokens, logits, _ = fx_export.causal_host_loop(programs["prefill"].module(), programs["step"].module(),
                                                           prompt, N, geometry, rope, embed)
            assert torch.equal(tokens, want_tokens)
            assert all(_equal(got.float(), want) for got, want in zip(logits, want_logits))


def _persistent_export(tmp_path, opts):
    """An export the CLI / compose path can rebuild with load_quantized.build_causal_lm (every buffer persistent, as
    modelutils.load_llama; test_fx_export's CLI test), and its architecture config directory."""
    import algos
    import exporter.head_embed as he
    import exporter.quant_export as quant_export
    from test_export import _calib
    from test_head_embed import SPEC
    from test_ladder import FAST, tiny_llama

    model = tiny_llama(2)
    for layer in model.modules():
        layer._non_persistent_buffers_set.clear()
    config_dir = tmp_path / "architecture"
    model.config.save_pretrained(config_dir)
    algo = algos.make(SPEC)
    batches, calib = _calib()
    rec = he.prepare(model, opts)
    rep = algos.quantize(model, "ref_conv", he.skip_for(opts), dict(FAST), algo=algo, calib=calib)
    rec = he.finish(model, opts, rec)
    d = tmp_path / "export"
    quant_export.export(model, d, mode="ref_conv", report=rep, algo=algo, algo_spec=SPEC, options=dict(FAST),
                        skip=he.skip_for(opts), model_record={"kind": "causal_lm", "id": str(config_dir)},
                        calib_batches=batches, calib_record={"source": "test"},
                        head_embed=None if he.is_default(opts) else he.export_record(opts, rec))
    return d, config_dir


@pinned
def test_cli_captures_decode_step_and_layer(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    d, config_dir = _persistent_export(tmp_path, FP8)
    for boundary, extra in (("causal_lm_decode_step", ["--host-embeddings"]), ("causal_lm_decode_layer", ["--layer", "1"])):
        out = tmp_path / boundary
        done = subprocess.run([sys.executable, str(Path(fx_export.__file__).resolve()), "--export-dir", str(d), "--model",
                               str(config_dir), "--out-dir", str(out), "--seqlen", str(P), "--capacity", str(C),
                               "--native-fp8", "--boundary", boundary, *extra], capture_output=True, text=True,
                              env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                  [str(Path(fx_export.__file__).resolve().parent.parent), os.environ.get("PYTHONPATH", "")])})
        assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
        report = json.loads((out / "verification.json").read_text())
        assert report["status"] == "passed" and report["provenance"]["capacity"] == C
        assert report["fresh_process_reload"]["model_replay_modules_imported"] == []


@pinned
@pytest.mark.parametrize("opts", [DEFAULT, FP8], ids=["default", "head_fp8_emb_fp8"])
def test_loop_manifest_and_composition_scripts(tmp_path, opts):
    """make_loop_manifest.py causal_lm and compose_phases.py causal_lm on two tiny bundles laid out as the capture driver
    lays them out (export/, fx/, host_cases.safetensors, loop_manifest.json)."""
    from safetensors.torch import save_file
    from test_tile_flow_defects import _script

    d, config_dir = _persistent_export(tmp_path, opts)
    model, _, _ = load_quantized.load_quantized(str(config_dir), d, "cpu", mode="ref_conv", seqlen=C)
    tensors, policy, manifest = load_quantized.read_export(d)
    geometry = _geometry(model)
    embed = fx_export.causal_token_embedder(tensors, policy)
    prompts = _prompts(3, seed=31)
    bundles = {}
    for name, boundary in (("prefill", fx_export.CausalPrefill(model, True)), ("decode_step", fx_export.CausalDecodeStep(model, True))):
        bundle = tmp_path / name
        fx_export.link_export(d, bundle / "export")
        save_file({"prompts": prompts.contiguous()}, str(bundle / "host_cases.safetensors"))
        cases = fx_export.causal_phase_cases(model, boundary.eval(), geometry, 0, "cpu", host_embeddings=True, embed=embed,
                                             prompts=prompts, steps=(0, 5))
        report = fx_export.capture_fx(boundary, cases, bundle / "fx", policy=policy, native_fp8="uint8",
                                      provenance={"boundary": f"causal_lm_{name}", "architecture_locator": str(config_dir),
                                                  "export_manifest_sha256": "x", "capacity": C, "prefill_length": P,
                                                  "host_embeddings": True, "all_logits": False})
        assert report["status"] == "passed"
        bundles[name] = bundle
    mlm, compose, verify = _script("make_loop_manifest"), _script("compose_phases"), _script("verify_bundle")
    for bundle in bundles.values():           # the independent bundle checks accept the phase bundles as they are
        checked = verify.run(bundle / "export", bundle / "fx", tmp_path / f"verify_{bundle.name}.json")
        # panel_not_collapsed_to_per_tensor is about the tiny export's weights (64-row Linears: one or two 32-row
        # panels, often with one exponent), not about the bundle
        assert set(checked["failed"]) <= {"panel_not_collapsed_to_per_tensor"}, checked["failed"]
    out = tmp_path / "loop_manifest.json"
    assert mlm.main(["causal_lm", "--prefill-bundle", str(bundles["prefill"]), "--step-bundle", str(bundles["decode_step"]),
                     "--decode-steps", "6", "--out", str(out)]) == 0
    loop = json.loads(out.read_text())
    assert loop["kv_cache"]["per_tensor_shape"] == [1, 2, C, 16] and loop["kv_cache"]["tensors"] == 4
    assert loop["kv_cache"]["bytes_per_token"] == 2 * 2 * 2 * 16 * 2
    assert [s["shape"] for s in loop["graphs"]["decode_step"]["inputs"][:4]] == [[1, 1, 64], [1, 1, 1, C + 1], [1, 1, 16],
                                                                                 [1, 1, 16]]
    assert loop["causal_lm"]["lm_head"] == ("quantized" if opts == FP8 else "skipped")
    for bundle in bundles.values():
        (bundle / "loop_manifest.json").write_bytes(out.read_bytes())
    result = tmp_path / "compose.json"
    assert compose.main(["causal_lm", str(bundles["prefill"]), str(bundles["decode_step"]), str(result),
                         "--reference-cache", str(tmp_path / "refs.safetensors")]) == 0
    res = json.loads(result.read_text())
    assert res["ok"] and len(res["prompts"]) == 3 and len(res["fixtures"]) == 4
    assert all(r["tokens_equal_generate"] and r["logits_byte_identical"] and r["logits_compared"] == 7 for r in res["prompts"])
    assert res["step_pure_same_inputs_twice"] and res["step_inputs_unchanged"]
