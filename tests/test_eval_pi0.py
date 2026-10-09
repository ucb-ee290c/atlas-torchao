"""eval_pi0.py on a tiny random-weight Pi0 (CPU): the reference's noise and observations are
reproduced exactly, every profile runs end to end with its report fields, and the metrics.

Needs the atlas-pi0 environment and $OPENPI_DIR (see tests/test_pi0.py); skipped otherwise. The
policy-level test also needs the PaliGemma tokenizer ($PALIGEMMA_TOKENIZER).
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_pi0 import rt, tiny_pi0  # noqa: F401  (rt is a fixture)

import eval_pi0

H, A = 4, 32      # tiny_pi0: action_horizon, action_dim


def _args(tmp_path, *extra):
    return eval_pi0.build_parser().parse_args(
        ["--checkpoint-dir", str(tmp_path), "--out", str(tmp_path / "r.json"), "--device", "cpu",
         "--steps", "2", *extra])


def _build(rt, policy_tok=None):
    import pi0_model
    from quant_modes import quantize

    def build(mode, args):
        m = tiny_pi0(rt)
        inv = pi0_model.inventory(m)
        layers = quantize(m, mode, args.skip, options=args) if mode != "bf16" else None
        pol = None
        if policy_tok:
            pol = rt.Pi0PyTorchPolicy(model=pi0_model.FixedNoise(m), device=torch.device("cpu"),
                                      norm_stats=None, is_joint_position=True, max_token_len=8,
                                      tokenizer_path=policy_tok)
        return m, inv, layers, pol
    return build


def test_reference_noise_is_the_reference_draws(rt):
    m = tiny_pi0(rt)
    # run_output_eval.py:176: one manual_seed per pass, sample_noise per frame
    torch.manual_seed(3)
    stream = [m.sample_noise((1, H, A), "cpu") for _ in range(3)]
    # run_eval_mxu_io.py:652: manual_seed(i) per frame
    per = []
    for i in range(3):
        torch.manual_seed(i)
        per.append(m.sample_noise((1, H, A), "cpu"))
    state = torch.random.get_rng_state()
    for i in range(3):
        assert torch.equal(eval_pi0.frame_noise("reference", "policy", 3, i, (1, H, A), "cpu"), stream[i])
        assert torch.equal(eval_pi0.frame_noise("reference", "model", 0, i, (1, H, A), "cpu"), per[i])
    assert torch.equal(torch.random.get_rng_state(), state)          # global RNG untouched
    a = eval_pi0.frame_noise("per_frame", "policy", 0, 1, (1, H, A), "cpu")
    assert torch.equal(a, eval_pi0.frame_noise("per_frame", "model", 1, 0, (1, H, A), "cpu"))


def test_mxu_dummy_observations_match_reference(rt):
    import ref_loader

    rio = ref_loader.load_file("pi0-quant", "experiments/run_eval_mxu_io.py")
    cfg = SimpleNamespace(max_token_len=8)
    torch.manual_seed(0)                                              # run_eval_mxu_io.py:793
    want = [rio._make_dummy_obs(cfg, torch.device("cpu")) for _ in range(2)]
    torch.manual_seed(123)
    state = torch.random.get_rng_state()
    got = eval_pi0.mxu_dummy_observations(2, cfg, "cpu")
    assert torch.equal(torch.random.get_rng_state(), state)
    for g, w in zip(got, want):
        assert torch.equal(g["obs"].state, w.state)
        assert all(torch.equal(g["obs"].images[k], w.images[k]) for k in w.images)


def _run(rt, tmp_path, n_obs, *extra):
    args = _args(tmp_path, "--obs", "mxu_dummy", "--n-obs", str(n_obs), "--rmse", *extra)
    frames = eval_pi0.mxu_dummy_observations(n_obs, SimpleNamespace(max_token_len=8), "cpu")
    b = _build(rt)
    res = eval_pi0.evaluate(args, frames, lambda mode: b(mode, args), "model", H, A, tmp_path / "r.json")
    assert res["valid"] and all(e["deterministic"] for e in res["modes"].values())
    assert len(res["noise"]["sha256"]) == 64
    for name, e in res["modes"].items():
        assert np.asarray(e["actions"]).shape == (n_obs, H, eval_pi0.ACTION_DIMS)
        if name == "bf16":
            continue
        v = e["vs_bf16"]
        assert 0 < v["rel_l2"] < 1 and v["cosine"] > 0.5, (name, v)
        assert len(v["rmse_per_dim"]) == len(v["cosine_per_dim"]) == eval_pi0.ACTION_DIMS
        assert (tmp_path / "r_actions" / name / "action_rmse.csv").is_file()
        assert v["reference_action_rmse"]["overall_rmse"] > 0
        # model-level outputs are deltas already: the delta metric is the same numbers
        assert e["vs_bf16_delta"]["rel_l2"] == pytest.approx(v["rel_l2"])
        assert e["headline"]["delta_rel_l2"] == e["vs_bf16_delta"]["rel_l2"]
        comps = {c["component"] for c in e["rmse"]["components"]}
        assert {"vision", "language", "action_expert", "action_head"} & comps, (name, comps)
    return res["modes"]


def test_evaluate_software_profiles(rt, tmp_path):
    """bf16 + the software profiles: determinism, agreement, the reference action metric, per-layer
    RMSE by component and patch coverage."""
    modes = _run(rt, tmp_path, 2, "--modes", "bf16", "collab_v0", "collab_v0_hw", "reference")
    assert list(modes) == ["bf16", "collab_v0", "collab_v0_hw", "reference"]
    comps = {c["component"] for c in modes["reference"]["rmse"]["components"]}
    assert comps == {"vision", "language", "action_expert", "action_head"}
    # collab_v0 quantizes bf16 Linears only: the float32 action head stays, with that reason
    reasons = {r["site"]: r["reason"] for r in modes["collab_v0"]["patch_coverage"]["not_patched"]}
    assert any("action_out_proj" in s and "float32" in r for s, r in reasons.items())


def test_evaluate_hardware_profiles(rt, tmp_path):
    """reference_hw (MXU model, cross-checked against the reference CPU model) and chip: the
    cross-check verdict and the chip unit coverage reach the report (one frame, one flow step;
    --mxu sa is bit-exact-tested in test_pi0.py and only slower here)."""
    modes = _run(rt, tmp_path, 1, "--steps", "1", "--modes", "reference_hw", "chip", "--mxu", "ipt",
                 "--crosscheck-frac", "0", "--crosscheck-budget", "10")
    for name in ("reference_hw", "chip"):
        cc = modes[name]["layers"]["crosscheck"]
        assert cc["ok"] and cc["calls_checked"] > 0 and modes[name]["valid"], (name, cc)
    cov = modes["chip"]["layers"]["coverage"]
    assert cov["units"]["MXU"]["calls"] > 0 and cov["units"]["VPU"]["calls"] > 0


def test_policy_level_bf16_reproduces_run_output_eval(rt, tmp_path):
    """--obs dummy, bf16: our explicit-noise pass equals the reference's run (run_output_eval.py:
    176-177: torch.manual_seed(seed), then policy.infer on each dummy observation)."""
    tok = os.environ.get("PALIGEMMA_TOKENIZER")
    if not tok or not os.path.isfile(tok):
        pytest.skip("PALIGEMMA_TOKENIZER not set")
    import pi0_model

    frames = pi0_model.dummy_observations(2, seed=0)
    m = tiny_pi0(rt)
    pol = rt.Pi0PyTorchPolicy(model=m, device=torch.device("cpu"), norm_stats=None,
                              is_joint_position=True, max_token_len=8, tokenizer_path=tok)
    torch.manual_seed(0)
    want = [pol.infer(f["obs"])["actions"] for f in frames]          # num_steps=10 inside
    args = _args(tmp_path, "--obs", "dummy", "--steps", "10", "--modes", "bf16",
                 "--no-determinism-check")
    b = _build(rt, policy_tok=tok)
    res = eval_pi0.evaluate(args, frames, lambda mode: b(mode, args), "policy", H, A, tmp_path / "r.json")
    got = res["modes"]["bf16"]["actions"]
    for g, w in zip(got, want):
        assert np.array_equal(np.asarray(g, np.float32), w)


def test_ground_truth_metrics():
    p = [np.zeros((4, 8), np.float32), np.ones((4, 8), np.float32)]
    g = [np.ones((4, 8), np.float32), np.ones((4, 8), np.float32)]
    r = eval_pi0.vs_ground_truth(p, g)
    assert r["mse"] == pytest.approx(0.5) and r["mae"] == pytest.approx(0.5) and r["elements"] == 64
    assert r["mse_per_dim"] == pytest.approx([0.5] * 8)
    d = eval_pi0.per_dim(p, g)
    assert d["rmse_per_dim"] == pytest.approx([0.5 ** 0.5] * 8)


def test_cli_refuses_torchao_on_cpu(tmp_path, capsys):
    ap = eval_pi0.build_parser()
    args = ap.parse_args(["--checkpoint-dir", str(tmp_path), "--out", "x.json", "--device", "cpu",
                          "--modes", "torchao", "--obs", "mxu_dummy"])
    with pytest.raises(SystemExit):
        eval_pi0.check_args(ap, args)
    assert "torchao" in capsys.readouterr().err
    args = ap.parse_args(["--checkpoint-dir", str(tmp_path), "--out", "x.json", "--obs", "dummy",
                          "--modes", "reference", "bf16"])
    args.tokenizer_path = "t.model"
    eval_pi0.check_args(ap, args)
    assert args.modes == ["bf16", "reference"]



def test_delta_space_metrics():
    """Policy-level chunks are absolute joint positions (state added back); the delta metric
    subtracts the frame's state on the 7 joint dims and leaves the gripper absolute."""
    state_j, grip = np.arange(7, dtype=np.float32) + 10, np.array([0.5], np.float32)
    fr = {"obs": {"observation/joint_position": state_j, "observation/gripper_position": grip}}
    delta = np.full((4, 8), 0.01, np.float32)
    absolute = delta.copy()
    absolute[:, :7] += state_j
    got = eval_pi0.to_delta([absolute], [fr], "policy")[0]
    assert np.allclose(got, delta, atol=1e-5)
    assert np.array_equal(eval_pi0.to_delta([absolute], [fr], "model")[0], absolute)   # model level: unchanged
    # a 10% error on the deltas is ~0.1% of the absolute norm: the reason delta is the headline
    noisy = absolute.copy()
    noisy[:, :7] += 0.001
    a = eval_pi0.agreement(eval_pi0._t([noisy]), eval_pi0._t([absolute]))
    d = eval_pi0.agreement(eval_pi0._t(eval_pi0.to_delta([noisy], [fr], "policy")),
                           eval_pi0._t([got]))
    assert d["rel_l2"] > 50 * a["rel_l2"]
    gt = eval_pi0.vs_ground_truth(eval_pi0.to_delta([noisy], [fr], "policy"), [got])
    assert gt["mae"] == pytest.approx(0.001 * 7 / 8, rel=1e-2)


def test_droid_frames_resized_on_client(rt):
    """DROID's 180x320 frames are padded to 224x224 before the policy shim, as openpi's DROID
    client does; the model's stubbed resize path drops the batch dimension."""
    import pi0_model

    obs = {k: np.full((180, 320, 3), 200, np.uint8) for k in pi0_model.DROID_CAMERAS}
    fr = eval_pi0.client_resize([{"obs": obs, "gt": None, "meta": {}}])[0]
    for k in pi0_model.DROID_CAMERAS:
        im = fr["obs"][k]
        assert im.shape == (224, 224, 3) and im.dtype == np.uint8
        assert im[0].max() == 0 and im[112].min() > 0               # letterbox padding, top and bottom
