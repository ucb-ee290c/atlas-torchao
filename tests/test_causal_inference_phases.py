"""Causal-LM inference phases (fx_export.CausalPrefill / CausalDecodeStep / CausalDecodeLayer) on a tiny Llama exported with
today's static recipe (test_head_embed._export), with the default BF16 head and embedding and with the FP8 head and FP8
embedding (--lm-head fp8 --embedding fp8):
  - structurally: every attention call of the phases receives the query, keys, values and mask, word for word and shape for
    shape, that the same call receives in transformers' own static-cache loop (generate on a StaticCache of max_cache_len C),
    at the prefill and at every decode step;
  - the host loop over the two phases (greedy argmax, fixed-capacity cache, host mask, slot, RoPE rows and embeddings) is
    generate(do_sample=False) on that StaticCache token for token and logit word for logit word, and the prefill's logits /
    K / V and a decode step's are transformers' own forward on a StaticCache;
  - the host inputs are transformers' (create_causal_mask on a StaticCache, rotary_emb, embed_tokens), the step is pure,
    and the per-layer decode boundary chains to the step;
  - the captured native-FP8 graphs have no symbolic shape, no RoPE trig, no input mutation (with host embeddings no
    embedding lookup), replay the same words after save / load, and compose to the static-cache generate."""
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


def _static(model, capacity=C):
    from transformers import StaticCache

    return StaticCache(config=model.config, max_cache_len=capacity)


@pytest.mark.parametrize("host_embeddings", [False, True], ids=["ids", "host_embeddings"])
def test_host_loop_is_static_cache_generate(exported, host_embeddings):
    opts, model, _, tensors, policy = exported
    geometry, rope, embed = _host(model, tensors, policy, host_embeddings)
    prefill = fx_export.CausalPrefill(model, host_embeddings).eval()
    step = fx_export.CausalDecodeStep(model, host_embeddings).eval()
    with torch.no_grad():
        for prompt in _prompts():
            prompt = prompt[None]
            want_tokens, want_logits = fx_export.causal_generate(model, prompt, N, C)
            tokens, logits, cache = fx_export.causal_host_loop(prefill, step, prompt, N, geometry, rope, embed)
            assert torch.equal(tokens, want_tokens)
            assert len(logits) == len(want_logits) == N + 1
            assert all(got.dtype == torch.bfloat16 and _equal(got.float(), want) for got, want in zip(logits, want_logits))
            # the slots past the last written one are still zero
            assert all(not buffer[:, :, P + N:].any() for buffer in cache)


def test_attention_receives_what_static_cache_loop_gives(exported, monkeypatch):
    """Structural: the reference attention (ref_attention.ref_attention, which _hf_attention calls after the GQA
    expansion) gets byte-identical, same-shape q, k, v and mask from the phases and from generate on a StaticCache,
    call for call (layers x (prefill + steps)). End-to-end logit equality alone could pass by luck."""
    import reference.ref_attention as ref_attention

    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    calls, original = [], ref_attention.ref_attention

    def record(q, k, v, scaling, mask=None, *args, **kwargs):
        calls.append(tuple(t.clone() for t in (q, k, v, mask)))
        return original(q, k, v, scaling, mask, *args, **kwargs)
    monkeypatch.setattr(ref_attention, "ref_attention", record)
    prompt = _prompts(1)
    with torch.no_grad():
        fx_export.causal_generate(model, prompt, 4, C)
        library, calls[:] = list(calls), []
        fx_export.causal_host_loop(fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval(), prompt,
                                   4, geometry, rope)
    assert len(calls) == len(library) == geometry.layers * 5
    for ours, theirs in zip(calls, library):
        assert all(_equal(a, b) for a, b in zip(ours, theirs))
    assert calls[0][1].shape[-2] == calls[-1][1].shape[-2] == C          # C keys at the prefill and at a step


def test_prefill_and_step_are_transformers_static_forward(exported):
    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prompt = _prompts(1)
    with torch.no_grad():
        cache = _static(model)
        last = model(prompt, past_key_values=cache, use_cache=True, logits_to_keep=1)   # what generate's prefill computes
        every = model(prompt, past_key_values=_static(model), use_cache=True, logits_to_keep=0)
        empty = fx_export.causal_empty_cache(geometry)
        got = fx_export.CausalPrefill(model).eval()(*fx_export.causal_prefill_inputs(prompt, rope, empty))
        all_pos = fx_export.CausalPrefill(model, all_logits=True).eval()(*fx_export.causal_prefill_inputs(prompt, rope, empty))
        assert _equal(got[0], last.logits[:, -1]) and _equal(all_pos[0], every.logits)
        assert len(got) - 1 == 2 * geometry.layers
        for i, layer in enumerate(cache.layers):
            assert _equal(got[1 + 2 * i], layer.keys[:, :, :P]) and _equal(got[2 + 2 * i], layer.values[:, :, :P])
            assert not layer.keys[:, :, P:].any()
        assert all(list(t.shape) == [1, geometry.n_kv_heads, P, geometry.head_dim] for t in got[1:])
        # one step at length P on the same contents
        kv = fx_export.causal_empty_cache(geometry)
        fx_export.causal_cache_write(kv, got[1:], 0)
        token = torch.tensor([[5]])
        step = fx_export.CausalDecodeStep(model).eval()(*fx_export.causal_step_inputs(token, P, C, rope, kv))
        out = model(token, past_key_values=cache, use_cache=True)
    assert _equal(step[0], out.logits[:, -1])
    for i, layer in enumerate(cache.layers):
        assert _equal(step[1 + 2 * i], layer.keys[:, :, P:P + 1]) and _equal(step[2 + 2 * i], layer.values[:, :, P:P + 1])


def test_host_inputs_are_transformers(exported):
    from transformers.masking_utils import create_causal_mask

    _, model, _, tensors, policy = exported
    rope = fx_export.causal_rope_tables(model, C)
    prompt = _prompts(1)
    embeds = model.model.embed_tokens(prompt)
    with torch.no_grad():
        cache = _static(model)
        mask = create_causal_mask(config=model.config, inputs_embeds=embeds, attention_mask=None, past_key_values=cache,
                                  position_ids=torch.arange(P)[None])
        assert _equal(fx_export.causal_prefill_mask(P, C), mask)
        model(prompt, past_key_values=cache, use_cache=True)
        for length in (P, P + 1):
            mask = create_causal_mask(config=model.config, inputs_embeds=embeds[:, :1], attention_mask=None,
                                      past_key_values=cache, position_ids=torch.tensor([[length]]))
            assert _equal(fx_export.causal_decode_mask(length, C), mask)
            model(torch.tensor([[3]]), past_key_values=cache, use_cache=True)
        for position in (0, 7, C - 1):          # the row of the table is the model's own per-position computation
            cos, sin = model.model.rotary_emb(embeds[:, :1], torch.tensor([[position]]))
            assert _equal(rope[0][:, position:position + 1], cos) and _equal(rope[1][:, position:position + 1], sin)
        assert _equal(fx_export.causal_token_embedder(tensors, policy)(prompt), embeds)
    slot = fx_export.causal_decode_slot(10, C)
    assert slot.shape == (1, 1, C, 1) and slot.dtype == torch.bfloat16 and slot.sum() == 1 and slot[0, 0, 10, 0] == 1
    for bad in (C, -1):
        with pytest.raises(ValueError):
            fx_export.causal_decode_mask(bad, C)


def test_decode_step_is_pure(exported):
    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prefill, step = fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval()
    with torch.no_grad():
        tokens, _, cache = fx_export.causal_host_loop(prefill, step, _prompts(1), 4, geometry, rope)
        args = fx_export.causal_step_inputs(tokens[:, -1:], P + 4, C, rope, cache)
        before = [t.clone() for t in args]
        first, second = step(*args), step(*args)
    assert all(_equal(a, b) for a, b in zip(first, second))
    assert all(_equal(a, b) for a, b in zip(args, before))


def test_decode_layers_chain_to_the_step(exported):
    _, model, _, tensors, policy = exported
    geometry, rope, _ = _host(model, tensors, policy, False)
    prefill, step = fx_export.CausalPrefill(model).eval(), fx_export.CausalDecodeStep(model).eval()
    with torch.no_grad():
        tokens, _, cache = fx_export.causal_host_loop(prefill, step, _prompts(1), 2, geometry, rope)
        token, mask, cos, sin, slot, *kv = fx_export.causal_step_inputs(tokens[:, -1:], P + 2, C, rope, cache)
        want = step(token, mask, cos, sin, slot, *kv)
        hidden = model.model.embed_tokens(token)
        for i in range(geometry.layers):
            hidden, k, v = fx_export.CausalDecodeLayer(model, i).eval()(hidden, cos, sin, mask, slot, kv[2 * i],
                                                                         kv[2 * i + 1])
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
        assert len({tuple(t.float().sum().item() for t in case) for case in cases}) == count
        if isinstance(boundary, fx_export.CausalPrefill):
            assert all(not t.any() for case in cases for t in case[4:])
        if isinstance(boundary, fx_export.CausalDecodeStep):
            lengths = [int((case[1] == 0).sum()) - 1 for case in cases]
            assert lengths == [P, P + 3]
            assert [int(case[4][0, 0, :, 0].argmax()) for case in cases] == lengths
            assert cases[0][0].shape == (1, 1, geometry.hidden)
            assert all(not t[:, :, length:].any() for case, length in zip(cases, lengths) for t in case[5:])


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
def test_phase_graphs_capture_and_compose_to_static_cache_generate(tmp_path, exported, host_embeddings):
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
            want_tokens, want_logits = fx_export.causal_generate(model, prompt, N, C)
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
def test_cli_captures_prefill_decode_step_and_layer(tmp_path):
    import os
    import subprocess
    import sys
    from pathlib import Path

    d, config_dir = _persistent_export(tmp_path, FP8)
    for boundary, extra in (("causal_lm_prefill", ["--all-logits"]), ("causal_lm_decode_step", ["--host-embeddings"]),
                            ("causal_lm_decode_layer", ["--layer", "1"])):
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
        if boundary == "causal_lm_prefill":       # all-position logits, then 2 x 2 K / V of the prompt's slots
            parity = json.loads((out / "parity.json").read_text())
            from safetensors.torch import load_file
            fixtures = load_file(str(out / "fixtures.safetensors"))
            outs = [fixtures[k] for k in parity["cases"][0]["outputs"]]
            assert list(outs[0].shape) == [1, P, 128] and [list(t.shape) for t in outs[1:]] == [[1, 2, P, 16]] * 4


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
    assert [s["shape"] for s in loop["graphs"]["decode_step"]["inputs"][:5]] == [[1, 1, 64], [1, 1, 1, C], [1, 1, 16],
                                                                                 [1, 1, 16], [1, 1, C, 1]]
    assert [s["shape"] for s in loop["graphs"]["prefill"]["inputs"][:5]] == [[1, P, 64], [1, 1, P, C], [1, P, 16],
                                                                             [1, P, 16], [1, 2, C, 16]]
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
