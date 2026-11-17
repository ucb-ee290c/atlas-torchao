"""SmolVLA inference phases (fx_export.SmolVLAPrefix / SmolVLADenoiseStep) on a tiny lerobot VLAFlowMatching (smolvla_tiny):
  - the prefix boundary's K / V are byte-identical to lerobot's own prefill (boolean masks, DynamicCache), and its host masks to
    embed_prefix's, with and without padded language tokens and host RoPE;
  - the step boundary is byte-identical to denoise_step on that cache at every Euler time, pure (same inputs twice, inputs
    unchanged), and the host Euler loop over it equals sample_actions;
  - the captured graphs (native FP8) have no dead node, no symbolic shape, no cumsum, only additive BF16 softmax masks, no
    input mutation, with host RoPE no RoPE angle math (the step's one sin / cos pair is the float64 time embedding), and the
    composed saved graphs reproduce sample_actions;
  - the Euler grid of the loop manifest is euler_integrate's (probed), and the step's host RoPE tables are chosen by role, since
    both of its RoPE position sets are chunk-length."""
from types import SimpleNamespace

import pytest
import torch
from smolvla_tiny import CHUNK, LANG, observation, quantize, tiny_policy
from test_fx_export import PINNED_TORCH
from test_tile_flow_defects import _script

import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers the atlas ops)
import exporter.fx_export as fx_export

pinned = pytest.mark.skipif(torch.__version__ < PINNED_TORCH, reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")
N_IMAGES, STEPS = 2, 10


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _equal(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


@pytest.fixture(scope="module")
def quantized():
    policy = tiny_policy(num_steps=STEPS)
    return policy, quantize(policy)


def _grid(steps=STEPS):
    return [1.0 + k * (-1.0 / steps) for k in range(steps)], -1.0 / steps


def _lerobot_prefill(model, images, img_masks, lang_tokens, lang_masks, state):
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    embs, pad, att = model.embed_prefix(list(images), img_masks, lang_tokens, lang_masks, state=state)
    _, cache = model.vlm_with_expert.forward(attention_mask=make_att_2d_masks(pad, att),
                                             position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                                             inputs_embeds=[embs, None], use_cache=True)
    return pad, att, cache


def _host_loop(step, noise, host, kv):
    times, dt = _grid()
    x = noise
    for t in times:
        x = x + dt * step(x, torch.full((1,), t, dtype=torch.float32), *host, *kv)
    return x


# -- eager equivalence with lerobot's own inference path --------------------------------------------------------------------

@pytest.mark.parametrize("host_rope", [False, True])
@pytest.mark.parametrize("valid", [LANG, 2])
def test_phases_are_lerobots_prefill_denoise_step_and_sample_actions(quantized, host_rope, valid):
    policy, _ = quantized
    model = policy.model
    geometry = fx_export.smolvla_geometry(policy)
    images, img_masks, lang_tokens, lang_masks, state, noise = observation(3, N_IMAGES, valid_tokens=valid)
    with torch.no_grad():
        pad, att, cache = _lerobot_prefill(model, images, img_masks, lang_tokens, lang_masks, state)
        mpad, matt = fx_export.smolvla_prefix_masks(geometry, img_masks, lang_masks)
        assert torch.equal(pad, mpad) and torch.equal(att, matt)
        assert int((~pad).sum()) == LANG - valid
        case, ppad = fx_export.smolvla_prefix_inputs(geometry, images, img_masks, lang_tokens, lang_masks, state, host_rope)
        (kv,) = fx_export.smolvla_prefix_outputs(policy, N_IMAGES, [case], host_rope)
        assert len(kv) == 2 * len(cache.layers)
        for i, layer in enumerate(cache.layers):
            assert kv[2 * i].is_contiguous() and kv[2 * i].shape == (1, pad.shape[1], 2, 16)
            assert _equal(kv[2 * i], layer.keys.transpose(1, 2)) and _equal(kv[2 * i + 1], layer.values.transpose(1, 2))
        host = fx_export.smolvla_step_host_inputs(geometry, ppad, host_rope)
        step = fx_export.SmolVLADenoiseStep(policy, host_rope=host_rope).eval()
        try:
            times, dt = _grid()
            x, before = noise, [_bits(t).clone() for t in kv]
            for t in times:
                tt = torch.full((1,), t, dtype=torch.float32)
                v = step(x, tt, *host, *kv)
                assert _equal(v, step(x, tt, *host, *kv))                       # pure: same inputs, same words
                assert all(torch.equal(b, _bits(t)) for b, t in zip(before, kv))  # the K / V inputs are not written
                step.restore()
                assert _equal(v, model.denoise_step(pad, cache, x, tt))          # lerobot's, on its DynamicCache
                step = fx_export.SmolVLADenoiseStep(policy, host_rope=host_rope).eval()
                x = x + dt * v
            loop = _host_loop(step, noise, host, kv)
        finally:
            step.restore()
        assert _equal(loop, x)
        assert _equal(loop, model.sample_actions(list(images), img_masks, lang_tokens, lang_masks, state, noise=noise))


def test_boundaries_restore_rope_and_layer_methods(quantized):
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    policy, _ = quantized
    inner = policy.model.vlm_with_expert
    original = swe.apply_rope
    step = fx_export.SmolVLADenoiseStep(policy, host_rope=True)
    assert swe.apply_rope is not original and "forward_attn_layer" in inner.__dict__
    step.restore()
    assert swe.apply_rope is original
    assert "forward_attn_layer" not in inner.__dict__ and "forward_cross_attn_layer" not in inner.__dict__
    prefix = fx_export.SmolVLAPrefix(policy, N_IMAGES, host_rope=True)
    prefix.restore()
    assert swe.apply_rope is original


def test_step_rope_tables_are_chosen_by_role_not_length():
    """In the step, self-attention rotates prefix_offset + 0..chunk-1 and cross-attention 0..chunk-1: two chunk-length
    tables, which the length match of the flow-loss boundaries cannot tell apart."""
    rope = fx_export._HostRope()
    positions = torch.arange(4)[None]
    tables = {"suffix": fx_export._rope_table(positions + 9, 8), "expert": fx_export._rope_table(positions, 8)}
    rope.tables = tables
    x = torch.randn(1, 4, 2, 8)
    with pytest.raises(ValueError, match="distinct sequence lengths"):
        rope.apply_rope(x, positions)
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    for role, offset in (("suffix", 9), ("expert", 0)):
        rope.role = role
        assert torch.equal(rope.apply_rope(x, positions), swe.apply_rope(x, positions + offset))
    rope.role = "suffix"
    with pytest.raises(ValueError, match="does not cover"):
        rope.apply_rope(x, torch.arange(5)[None])


def test_step_host_inputs_are_denoise_steps():
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    pad = torch.tensor([[True] * 6 + [False] * 3 + [True]])
    geometry = SimpleNamespace(tokens_per_image=4, chunk_size=CHUNK, head_dim=8)
    mask, positions = fx_export.smolvla_step_host_inputs(geometry, pad)
    suffix_pad = torch.ones(1, CHUNK, dtype=torch.bool)
    keep = torch.cat([pad[:, None, :].expand(1, CHUNK, 10),
                      make_att_2d_masks(suffix_pad, torch.ones(1, CHUNK))], dim=2)
    assert mask.dtype == torch.bfloat16 and torch.equal(mask, fx_export.additive_attention_mask(keep))
    assert positions.tolist() == [[7, 8, 9]]                                # 7 valid prefix tokens, then the chunk
    mask2, *tables = fx_export.smolvla_step_host_inputs(geometry, pad, host_rope=True)
    assert torch.equal(mask2, mask) and len(tables) == 4
    assert all(torch.equal(a, b) for a, b in zip(tables, (*fx_export._rope_table(positions, 8),
                                                          *fx_export._rope_table(positions - 7, 8))))


# -- loop manifest ----------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("steps", [10, 3, 7])
def test_euler_grid_is_euler_integrates(steps):
    pytest.importorskip("lerobot.policies.common.flow_matching")
    mlm = _script("make_loop_manifest")
    times, dt, record = mlm.euler_integrate_grid(steps)
    assert dt == -1.0 / steps and len(times) == steps
    assert times == [float(torch.tensor(1.0 + k * dt, dtype=torch.float32)) for k in range(steps)]
    assert record["update_reproduced_with_python_float_dt"]
    # not Pi0's float32 recurrence (time += dt in float32), which drifts from the Python-float products (0.79999995 against
    # 0.8 at the third of ten steps)
    assert mlm.euler_times(steps)[1] != times


# -- captured graphs --------------------------------------------------------------------------------------------------------

@pinned
@pytest.mark.parametrize("host_rope", [False, True])
def test_phase_graphs_capture_and_compose_to_sample_actions(tmp_path, quantized, host_rope):
    policy, qpolicy = quantized
    model = policy.model
    observations, raw = [], []
    for i, valid in enumerate((LANG, 2)):
        images, img_masks, lang_tokens, lang_masks, state, noise = observation(10 + i, N_IMAGES, valid_tokens=valid)
        raw.append((images, img_masks, lang_tokens, lang_masks, state, noise))
        observations.append((images, img_masks, lang_tokens, lang_masks, state, noise,
                             torch.full((1,), [1.0, 0.4][i], dtype=torch.float32)))
    prefix_cases, step_cases = fx_export.smolvla_phase_cases(policy, N_IMAGES, observations, host_rope)
    programs = {}
    for name, cases in (("prefix", prefix_cases), ("step", step_cases)):
        boundary = (fx_export.SmolVLAPrefix(policy, N_IMAGES, host_rope) if name == "prefix" else
                    fx_export.SmolVLADenoiseStep(policy, host_rope)).eval()
        try:
            report = fx_export.capture_fx(boundary, cases, tmp_path / name, policy=qpolicy, fresh_process=name == "step",
                                          native_fp8="uint8")
        finally:
            boundary.restore()
        assert report["status"] == "passed"
        assert report["graph_shape_census"] == {"dynamic_nodes": [], "cumsum_nodes": []}
        forms = report["additive_attention_mask"]["softmax_input_forms"]
        assert set(forms) <= {"additive_bf16_min", "unmasked"} and forms["additive_bf16_min"] > 0
        program = torch.export.load(tmp_path / name / "model.pt2")
        calls = [n for n in program.graph.nodes if n.op == "call_function"]
        assert not [n for n in calls if not n.users and not n.is_impure() and "_unused_" not in n.name]
        assert not program.graph_signature.user_inputs_to_mutate
        trig = [n for n in calls if str(n.target) in ("aten.sin.default", "aten.cos.default")]
        if host_rope:                       # only the step's time embedding (create_sinusoidal_pos_embedding, float64)
            assert len(trig) == (2 if name == "step" else 0)
            assert all(n.meta["val"].dtype == torch.float64 for n in trig)
        programs[name] = program.module()
    # the prefix graph stops after the last layer's K / V: that layer's query, o_proj and MLP are never called
    last = model.vlm_with_expert.num_vlm_layers - 1
    assert f"policy.model.vlm_with_expert.vlm.model.text_model.layers.{last}.mlp.up_proj" in \
        __import__("json").loads((tmp_path / "prefix" / "verification.json").read_text())["unused_quantized_modules"]
    geometry = fx_export.smolvla_geometry(policy)
    with torch.no_grad():
        for images, img_masks, lang_tokens, lang_masks, state, noise in raw:
            case, pad = fx_export.smolvla_prefix_inputs(geometry, images, img_masks, lang_tokens, lang_masks, state, host_rope)
            kv = programs["prefix"](*case)
            host = fx_export.smolvla_step_host_inputs(geometry, pad, host_rope)
            loop = _host_loop(programs["step"], noise, host, kv)
            want = model.sample_actions(list(images), img_masks, lang_tokens, lang_masks, state, noise=noise)
            assert _equal(loop, want)
