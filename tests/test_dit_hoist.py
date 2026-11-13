"""Multi-Task DiT adaLN modulation hoisted out of the denoising loop (docs/algos/tile-aware.md#per-phase-graphs-and-loop-manifest).

On a tiny lerobot DiffusionTransformer, quantized with a static-activation recipe: the hoisted step equals the stock
noise_predictor byte for byte, the hoisted 100-step loop equals the step-by-step loop byte for byte, the modulate and step
graphs (capture_fx, native FP8) hold disjoint quantized sites whose union is the noise predictor, and the composed graphs
reproduce the eager loop. A dynamic per-32-row recipe is refused unless each modulate call holds one step, and is exact then."""
import json
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
from test_ladder import FAST
from test_tile_flow_defects import SCRIPTS, _script

mt = pytest.importorskip("lerobot.policies.multi_task_dit.modeling_multi_task_dit")
pytest.importorskip("diffusers")

STATIC = "granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib"
DYNAMIC = "granularity:w=panel,a=block,rows=32,rule=nc,attn=head"
COND, ACTION, HORIZON = 32, 4, 16


@pytest.fixture(scope="module")
def cap():
    # imported by name (not _script): torch.export records the boundary class's module, and a reload must import it
    sys.path.insert(0, str(SCRIPTS))
    import policy_fx_capture_phases
    return policy_fx_capture_phases


@pytest.fixture(scope="module")
def composer():
    return _script("compose_phases")


class TinyPolicy(nn.Module):
    """The two attributes the phase boundaries read (noise_predictor, objective); forward = one noise-predictor call."""

    def __init__(self, steps=100):
        super().__init__()
        cfg = SimpleNamespace(action_feature=SimpleNamespace(shape=(ACTION,)), horizon=HORIZON, hidden_dim=64, num_layers=2,
                              num_heads=2, dropout=0.0, use_rope=True, timestep_embed_dim=32, use_positional_encoding=False,
                              rope_base=10000.0, compile_model=False)
        self.noise_predictor = mt.DiffusionTransformer(cfg, conditioning_dim=COND)
        for block in self.noise_predictor.transformer_blocks:          # lerobot zero-initializes adaLN; give it weights
            nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.05)
            nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.05)
        sched = SimpleNamespace(num_train_timesteps=steps, beta_start=1e-4, beta_end=0.02, beta_schedule="squaredcos_cap_v2",
                                clip_sample=True, clip_sample_range=1.0, prediction_type="epsilon", noise_scheduler_type="DDPM",
                                num_inference_steps=None)
        self.objective = mt.DiffusionObjective(sched, ACTION, HORIZON)
        # as eval_policy.load_policy: the float32 sinusoidal embedding is rounded to the policy dtype
        self.noise_predictor.time_mlp[0].register_forward_hook(lambda mod, inp, out: out.to(torch.bfloat16))

    def forward(self, noisy, timestep, cond):
        return self.noise_predictor(noisy, timestep, conditioning_vec=cond)


def _cond(seed):
    return torch.randn(1, COND, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _noisy(seed):
    return torch.randn(1, HORIZON, ACTION, generator=torch.Generator().manual_seed(seed)).to(torch.bfloat16)


def _quantized(spec, steps=100):
    torch.manual_seed(0)
    policy = TinyPolicy(steps).to(torch.bfloat16).eval()
    batches = [(_noisy(10 + i), torch.tensor([t]), _cond(20 + i)) for i, t in enumerate((99, 50, 3, 0))]
    algo = algos.make(spec)
    report = algos.quantize(policy, "ref_conv", options=dict(FAST), algo=algo,
                            calib=lambda: (batches, lambda m, b: m(*b), {"source": "synthetic", "n_samples": 4, "seed": 0}))
    return policy, quant_export.build_policy(policy, "ref_conv", report, algo, dict(FAST), [], [], {}, {})


@pytest.fixture(scope="module")
def static_policy():
    return _quantized(STATIC)


def _timesteps(policy):
    sched = policy.objective.noise_scheduler
    sched.set_timesteps(policy.objective.num_inference_steps)
    return sched.timesteps.to(torch.long)


def _hoisted_loop(cap, policy, cond, seed, steps_per_call):
    """Eager hoisted loop: modulate once, then the scheduler's own step on each hoisted step output (same RNG order as
    conditional_sample: initial noise, then the scheduler's variance draws)."""
    sched = policy.objective.noise_scheduler
    timesteps = _timesteps(policy)
    with torch.no_grad():
        mod = cap.DiTModulate(policy, steps_per_call)(cond, timesteps)
        torch.manual_seed(seed)
        x = torch.randn(1, HORIZON, ACTION, dtype=torch.bfloat16)
        step = cap.DiTStepHoisted(policy)
        for i, t in enumerate(sched.timesteps):
            x = sched.step(step(x, mod[:, i].contiguous()), t, x).prev_sample
    return x, mod


def _stock_loop(policy, cond, seed):
    torch.manual_seed(seed)
    with torch.no_grad():
        return policy.objective.conditional_sample(policy.noise_predictor, 1, cond)


def test_hoisted_paths_are_time_mlp_and_adaln(cap, static_policy):
    policy, _ = static_policy
    assert cap.hoisted_paths(policy.noise_predictor) == [
        "noise_predictor.time_mlp.1", "noise_predictor.time_mlp.3",
        "noise_predictor.transformer_blocks.0.adaLN_modulation.1", "noise_predictor.transformer_blocks.1.adaLN_modulation.1"]


def test_hoisted_step_equals_stock_noise_predictor(cap, static_policy):
    """Guards the copy of TransformerBlock.forward in block_with_modulation against lerobot drift."""
    policy, _ = static_policy
    with torch.no_grad():
        for t, seed in ((99, 1), (40, 2), (0, 3)):
            cond, x = _cond(seed), _noisy(seed)
            mod = cap.DiTModulate(policy, 1)(cond, torch.tensor([t]))
            assert mod.shape == (2, 1, 1, 6 * 64)
            want = policy.noise_predictor(x, torch.tensor([t]), conditioning_vec=cond)
            assert torch.equal(cap.DiTStepHoisted(policy)(x, mod[:, 0]), want)


@pytest.mark.parametrize("steps_per_call", [1, 25, 100])
def test_hoisted_loop_equals_conditional_sample_static(cap, static_policy, steps_per_call):
    policy, _ = static_policy
    for seed in (0, 7):
        cond = _cond(30 + seed)
        got, _ = _hoisted_loop(cap, policy, cond, seed, steps_per_call)
        want = _stock_loop(policy, cond, seed)
        assert got.dtype == want.dtype and torch.equal(got, want), float((got.float() - want.float()).abs().max())


def test_modulation_rows_equal_per_step_rows(cap, static_policy):
    policy, _ = static_policy
    cond, ts = _cond(5), _timesteps(policy)
    with torch.no_grad():
        one = cap.DiTModulate(policy, 1)(cond, ts)
        for k in (7, 25, 100):                       # 7 does not divide 100: the last call is short
            assert torch.equal(cap.DiTModulate(policy, k)(cond, ts), one)


def test_dynamic_block_recipe_is_refused_when_batched_and_exact_per_step(cap):
    policy, policy_json = _quantized(DYNAMIC)
    paths = cap.hoisted_paths(policy.noise_predictor)
    with pytest.raises(ValueError, match="spanning rows"):
        cap.check_hoist_rules(policy_json, paths, 25)
    assert cap.check_hoist_rules(policy_json, paths, 1) == paths
    cond = _cond(3)
    got, _ = _hoisted_loop(cap, policy, cond, 0, 1)
    assert torch.equal(got, _stock_loop(policy, cond, 0))


def test_static_recipe_passes_the_rule_check(cap, static_policy):
    policy, policy_json = static_policy
    assert cap.check_hoist_rules(policy_json, cap.hoisted_paths(policy.noise_predictor), 100) == []


def _ddpm_rows(policy):
    mlm = _script("make_loop_manifest")
    sched = policy.objective.noise_scheduler
    sched.set_timesteps(policy.objective.num_inference_steps)
    return mlm.ddpm_schedule(sched)


def test_captured_modulate_and_step_graphs(cap, composer, static_policy, tmp_path):
    """Both graphs capture with native FP8; their quantized sites are disjoint and together are every noise-predictor
    site; the composed host loop over the saved graphs equals the eager stock loop."""
    policy, policy_json = static_policy
    ts = _timesteps(policy)
    k = 25
    mod_cases = [(_cond(1), ts.clone()), (_cond(2), ts.clone())]
    mod_report = fx_export.capture_fx(cap.DiTModulate(policy, k).eval(), mod_cases, tmp_path / "modulate", policy=policy_json,
                                      strict=True, fresh_process=False, native_fp8="uint8")
    with torch.no_grad():
        step_cases = [(_noisy(i), cap.DiTModulate(policy, 1)(_cond(i), ts[i * 40:i * 40 + 1])[:, 0]) for i in (1, 2)]
    step_report = fx_export.capture_fx(cap.DiTStepHoisted(policy).eval(), step_cases, tmp_path / "step", policy=policy_json,
                                       strict=True, fresh_process=False, native_fp8="uint8")
    assert mod_report["status"] == step_report["status"] == "passed"
    hoisted = set(cap.hoisted_paths(policy.noise_predictor, prefix="model.noise_predictor."))
    mod_gi = json.loads((tmp_path / "modulate" / "graph_inspection.json").read_text())
    step_gi = json.loads((tmp_path / "step" / "graph_inspection.json").read_text())
    mod_sites = {r["module"]: r["calls"] for r in mod_gi["weight_site_bindings"]}
    step_sites = {r["module"]: r["calls"] for r in step_gi["weight_site_bindings"]}
    assert set(mod_sites) == hoisted and all(n == 100 // k for n in mod_sites.values())
    assert not set(step_sites) & set(mod_sites) and all(n == 1 for n in step_sites.values())
    all_sites = {"model." + n for n, m in policy.named_modules() if type(m).__name__ == "RefQuantLinear"}
    assert set(step_sites) | set(mod_sites) == all_sites
    mod_g = torch.export.load(tmp_path / "modulate" / "model.pt2").module()
    step_g = torch.export.load(tmp_path / "step" / "model.pt2").module()
    rows = _ddpm_rows(policy)
    cond = _cond(9)
    with torch.no_grad():
        torch.manual_seed(4)
        x = torch.randn(1, HORIZON, ACTION, dtype=torch.bfloat16)
        got, calls = cap.hoisted_host_loop(step_g, mod_g(cond, ts), x, rows,
                                           lambda row, eps: torch.randn(eps.shape, dtype=eps.dtype) if row["variance_noise_draw"] else None,
                                           composer.ddpm_host_step)
    assert calls == 100
    assert torch.equal(got, _stock_loop(policy, cond, 4))
