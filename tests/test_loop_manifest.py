"""src/scripts/tile_flow/make_loop_manifest.py: the numbers in loop_manifest.json come from the scheduler / source they describe.
The DDPM table is checked against diffusers' own DDPMScheduler.step for every timestep (and a perturbed table must fail the
same check), the Pi0 float32 time recurrence against its step count for many step counts, the step-count reader against
synthetic sources. The end-to-end DiT / Pi0 runs need the real bundles and skip when they are absent."""
import json
import textwrap
from pathlib import Path

import pytest
import torch

from test_tile_flow_defects import _script

DIT_BUNDLE = Path("/scratch2/lorenhung/atlas-fp8/exports-tile/multitask-dit/atlas_v1_awq_tile_static_attn")
PI0_BUNDLE = Path("/scratch2/lorenhung/atlas-fp8/exports-tile/pi0/gptq_panel_static_noawq")


@pytest.fixture(scope="module")
def mlm():
    return _script("make_loop_manifest")


def _apply(row, x, eps, z):
    """One DDPM step from a manifest row only (epsilon prediction, clip_sample range 1.0)."""
    f = lambda v: torch.tensor(v, dtype=torch.float32)       # noqa: E731
    x0 = (x - f(row["sqrt_beta_prod_t"]) * eps) / f(row["sqrt_alpha_prod_t"])
    x0 = x0.clamp(-1.0, 1.0)
    out = f(row["pred_original_sample_coeff"]) * x0 + f(row["current_sample_coeff"]) * x
    return out + f(row["variance_std"]) * z if row["variance_noise_draw"] else out


def _reference_step(sched, eps, t, x, seed):
    g = torch.Generator().manual_seed(seed)
    return sched.step(eps, t, x, generator=g).prev_sample


@pytest.mark.parametrize("steps", [100, 10])
def test_ddpm_table_reproduces_scheduler_step_at_every_timestep(mlm, steps):
    diffusers = pytest.importorskip("diffusers")
    sched = diffusers.DDPMScheduler(num_train_timesteps=100, beta_start=1e-4, beta_end=0.02,
                                    beta_schedule="squaredcos_cap_v2", clip_sample=True, clip_sample_range=1.0,
                                    prediction_type="epsilon")
    sched.set_timesteps(steps)
    table = mlm.ddpm_schedule(sched)
    assert [r["t"] for r in table] == [int(t) for t in sched.timesteps] and len(table) == steps
    assert sum(r["variance_noise_draw"] for r in table) == steps - 1             # every step but t = 0
    torch.manual_seed(0)
    for row, t in zip(table, sched.timesteps):
        x, eps = torch.randn(1, 24, 7) * 2, torch.randn(1, 24, 7)
        z = torch.randn(1, 24, 7, generator=torch.Generator().manual_seed(7))
        # the scheduler draws its noise with randn_tensor(generator=g): same stream as z for the same seed
        want = _reference_step(sched, eps, int(t), x, seed=7)
        got = _apply(row, x, eps, z)
        assert torch.allclose(got, want, atol=1e-5, rtol=1e-5), (row["t"], float((got - want).abs().max()))


def test_perturbed_ddpm_table_fails_the_same_check(mlm):
    diffusers = pytest.importorskip("diffusers")
    sched = diffusers.DDPMScheduler(num_train_timesteps=100, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                    prediction_type="epsilon")
    sched.set_timesteps(100)
    row = dict(mlm.ddpm_schedule(sched)[40])
    row["current_sample_coeff"] *= 1.01
    x, eps = torch.randn(1, 24, 7), torch.randn(1, 24, 7)
    z = torch.randn(1, 24, 7, generator=torch.Generator().manual_seed(7))
    want = _reference_step(sched, eps, row["t"], x, seed=7)
    assert not torch.allclose(_apply(row, x, eps, z), want, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("steps", list(range(1, 41)) + [50, 100])
def test_pi0_euler_times_count_equals_num_steps(mlm, steps):
    dt, times = mlm.euler_times(steps)
    assert len(times) == steps, (steps, len(times))
    assert times[0] == 1.0 and all(a > b for a, b in zip(times, times[1:]))
    assert dt == pytest.approx(-1.0 / steps, rel=1e-6)
    assert times[-1] > 0                                                           # the last evaluated time is dt-ish above 0


def test_pi0_euler_times_are_float32_values(mlm):
    _, times = mlm.euler_times(10)
    assert all(float(torch.tensor(t, dtype=torch.float32)) == t for t in times)
    assert times[1] != 0.9                                                         # 0.9 in float64 would be a different value


def test_default_step_reader_parses_the_signature_and_refuses_a_missing_default(mlm, tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text(textwrap.dedent("""
        class M:
            def other(self, num_steps=3): pass
            def sample_actions(self, device, observation, noise=None, num_steps=7): pass
    """))
    assert mlm.sample_actions_default_steps(ok)[0] == 7
    bad = tmp_path / "bad.py"
    bad.write_text("class M:\n    def sample_actions(self, device, observation, noise=None): pass\n")
    with pytest.raises(KeyError):
        mlm.sample_actions_default_steps(bad)
    none = tmp_path / "none.py"
    none.write_text("def f(): pass\n")
    with pytest.raises(LookupError):
        mlm.sample_actions_default_steps(none)


def test_line_range_cites_the_line_and_fails_when_absent(mlm, tmp_path):
    p = tmp_path / "x.py"
    p.write_text("a = 1\nnum_steps = 10\n")
    assert mlm._line_range(p, "num_steps") == f"{p}:2"
    with pytest.raises(LookupError):
        mlm._line_range(p, "absent")


@pytest.mark.skipif(not DIT_BUNDLE.exists(), reason="needs the DiT bundle")
def test_dit_manifest_end_to_end(mlm, tmp_path):
    pytest.importorskip("lerobot")
    pytest.importorskip("diffusers")
    out = tmp_path / "loop_manifest.json"
    assert mlm.main(["dit", "--single-step-bundle", str(DIT_BUNDLE), "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    cfg = json.loads(Path(m["sources"]["policy_config"]).read_text())
    assert cfg["num_inference_steps"] is None and m["denoising_steps"] == cfg["num_train_timesteps"] == 100
    assert m["graphs"]["step"]["calls_per_action_chunk"] == 100 and m["graphs"]["encode"]["calls_per_action_chunk"] == 1
    assert len(m["scheduler"]["per_step"]) == 100 and m["scheduler"]["variance_noise_draws"] == 99
    assert m["scheduler"]["config"]["clip_sample"] is True and m["scheduler"]["config"]["prediction_type"] == "epsilon"
    assert m["graphs"]["step"]["inputs"]["conditioning_vec"]["shape"] == m["graphs"]["encode"]["outputs"]["conditioning_vec"]["shape"]


@pytest.mark.skipif(not (PI0_BUNDLE.exists() and Path("/scratch2/lorenhung/atlas-fp8/openpi").exists()), reason="needs the Pi0 bundle and openpi")
def test_pi0_manifest_end_to_end(mlm, tmp_path):
    pytest.importorskip("transformers")
    try:
        import sys
        sys.path.insert(0, "/scratch2/lorenhung/atlas-fp8/openpi/src")
        import openpi.models.gemma  # noqa: F401
    except Exception as error:                                                     # noqa: BLE001
        pytest.skip(f"openpi gemma config not importable here: {error}")
    out = tmp_path / "loop_manifest.json"
    assert mlm.main(["pi0", "--single-step-bundle", str(PI0_BUNDLE), "--checkpoint-dir",
                     "/scratch2/lorenhung/models/openpi-pytorch/pi0_droid_jointpos_polaris", "--openpi-dir",
                     "/scratch2/lorenhung/atlas-fp8/openpi", "--out", str(out)]) == 0
    m = json.loads(out.read_text())
    assert m["denoising_steps"] == m["euler"]["count"] == len(m["euler"]["times"]) == 10
    kv = m["graphs"]["prefix"]["outputs"]["kv_cache"]
    assert kv["tensors"] == 36 and kv["per_tensor_shape"] == [1, 1, 3 * 256 + 100, 256]
    assert m["graphs"]["step"]["calls_per_action_chunk"] == 10
