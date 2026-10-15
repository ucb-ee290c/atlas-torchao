"""The LeRobot normalization protocol (src/policy_norm.py, eval_policy --norm-stats): the guard
against silent pass-through, the probe, the recorded status and the export check."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("lerobot")

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature  # noqa: E402
from lerobot.processor import (NormalizerProcessorStep, PolicyProcessorPipeline,  # noqa: E402
                               UnnormalizerProcessorStep)
from lerobot.processor.converters import (batch_to_transition, policy_action_to_transition,  # noqa: E402
                                          transition_to_batch, transition_to_policy_action)

import eval_policy as ep  # noqa: E402
import policy_norm  # noqa: E402

STATE, ACTION, IMAGE = "observation.state", "action", "observation.image"
FEATURES = {STATE: PolicyFeature(FeatureType.STATE, (3,)),
            IMAGE: PolicyFeature(FeatureType.VISUAL, (3, 4, 4)),
            ACTION: PolicyFeature(FeatureType.ACTION, (2,))}


def _stats(mode, n):
    g = torch.Generator().manual_seed(n)
    lo = torch.randn(n, generator=g)
    hi = lo + 1 + torch.rand(n, generator=g)
    return {"MEAN_STD": {"mean": lo, "std": hi - lo},
            "MIN_MAX": {"min": lo, "max": hi},
            "QUANTILES": {"q01": lo, "q99": hi},
            "QUANTILE10": {"q10": lo, "q90": hi}}[mode]


def _cfg(mode="MEAN_STD", visual=NormalizationMode.IDENTITY):
    """The policy config the processors are checked against."""
    m = NormalizationMode(mode)
    return SimpleNamespace(input_features={STATE: FEATURES[STATE], IMAGE: FEATURES[IMAGE]},
                           output_features={ACTION: FEATURES[ACTION]},
                           normalization_mapping={"STATE": m, "ACTION": m, "VISUAL": visual})


def _pipes(mode="MEAN_STD", stats=None, visual=NormalizationMode.IDENTITY, features=FEATURES):
    m = NormalizationMode(mode)
    norm_map = {FeatureType.STATE: m, FeatureType.ACTION: m, FeatureType.VISUAL: visual}
    if stats is None:
        stats = {STATE: _stats(mode, 3), ACTION: _stats(mode, 2)}
    pre = PolicyProcessorPipeline(steps=[NormalizerProcessorStep(features=features, norm_map=norm_map,
                                                                 stats=stats)],
                                  to_transition=batch_to_transition, to_output=transition_to_batch)
    post = PolicyProcessorPipeline(steps=[UnnormalizerProcessorStep(features={ACTION: FEATURES[ACTION]},
                                                                    norm_map=norm_map, stats=stats)],
                                   to_transition=policy_action_to_transition,
                                   to_output=transition_to_policy_action)
    return pre, post


def _sample():
    return {STATE: torch.tensor([10.0, -20.0, 30.0]), IMAGE: torch.rand(3, 4, 4),
            ACTION: torch.tensor([[5.0, 6.0]])}


def _to_batch(sample, pre):
    return pre({k: v.unsqueeze(0) for k, v in sample.items()})


# ---------------------------------------------------------------------------------------------
# The guard on synthetic processors
# ---------------------------------------------------------------------------------------------

def test_missing_key_is_refused_and_identity_ignored():
    # the smolvla_base layout: stats only under a key no feature looks up
    pre, post = _pipes(stats={"so100.buffer.action": {"mean": torch.zeros(2), "std": torch.ones(2)}})
    g = policy_norm.check(pre, post, _cfg())
    assert not g["ok"]
    assert set(g["pre"]) == {STATE, ACTION} and set(g["post"]) == {ACTION}   # the image is IDENTITY
    text = " ".join(g["problems"])
    assert repr(STATE) in text and repr(ACTION) in text and IMAGE not in text
    with pytest.raises(policy_norm.NormalizationError, match="--norm-stats dataset") as e:
        policy_norm.require(g, "checkpoint", allow_unnormalized=False)
    assert STATE in str(e.value)


def test_identity_features_without_stats_pass():
    pre, post = _pipes()          # no image stats at all, image mode IDENTITY
    g = policy_norm.check(pre, post, _cfg())
    assert g["ok"], g["problems"]
    assert policy_norm.require(g, "checkpoint", False) == policy_norm.VERIFIED


def test_normalized_visual_feature_needs_stats():
    pre, post = _pipes(visual=NormalizationMode.MEAN_STD)
    g = policy_norm.check(pre, post, _cfg(visual=NormalizationMode.MEAN_STD))
    assert not g["ok"] and any(IMAGE in p for p in g["problems"])


@pytest.mark.parametrize("mode", ["MIN_MAX", "QUANTILES", "QUANTILE10", "MEAN_STD"])
def test_modes_pass_with_their_stats(mode):
    pre, post = _pipes(mode)
    g = policy_norm.check(pre, post, _cfg(mode))
    assert g["ok"], g["problems"]
    assert all(rec["sha256"] and rec["transform_sha256"] for rec in g["pre"].values())
    assert policy_norm.combined_sha256(g) and policy_norm.combined_sha256(g, "transform_sha256")
    probe = policy_norm.probe(pre, post, _sample(), _to_batch, g["required"], action_shape=(2,))
    assert probe["ok"], probe
    assert probe["observations"][STATE]["matches"] and probe["observations"][STATE]["finite"]
    rt = probe["action_round_trip"]
    assert rt["identity"] and rt["matches"] and rt["modes"] == [mode]


@pytest.mark.parametrize("mode,bad", [("MEAN_STD", {"std": torch.tensor([1.0, 0.0])}),
                                      ("MIN_MAX", {"max": torch.tensor([-9.0, 9.0])}),
                                      ("QUANTILES", {"q99": torch.tensor([float("nan"), 1.0])}),
                                      ("MIN_MAX", {"min": None})])
def test_degenerate_stats_are_refused(mode, bad):
    stats = {STATE: _stats(mode, 3), ACTION: dict(_stats(mode, 2))}
    for k, v in bad.items():
        if v is None:
            del stats[ACTION][k]
        else:
            stats[ACTION][k] = v
    g = policy_norm.check(*_pipes(mode, stats), _cfg(mode))
    assert not g["ok"] and all(repr(ACTION) in p for p in g["problems"])


def test_wrong_stats_name_for_the_mode_is_refused():
    # MIN_MAX declared, only mean/std stored: lerobot would raise only when data arrives
    g = policy_norm.check(*_pipes("MIN_MAX", {STATE: _stats("MEAN_STD", 3), ACTION: _stats("MEAN_STD", 2)}),
                          _cfg("MIN_MAX"))
    assert not g["ok"] and any("lacks ['min', 'max']" in p for p in g["problems"])


def test_config_feature_the_processor_omits_is_refused():
    # the saved normalizer's own features leave the state out; the config declares it MEAN_STD
    pre, post = _pipes(features={IMAGE: FEATURES[IMAGE], ACTION: FEATURES[ACTION]})
    g = policy_norm.check(pre, post, _cfg())
    assert not g["ok"] and g["required"]["pre"] == {STATE: "MEAN_STD", ACTION: "MEAN_STD"}
    assert any(repr(STATE) in p and "no step covers" in p for p in g["problems"])


def test_processor_mode_differing_from_the_config_is_refused():
    stats = {k: {**_stats("MEAN_STD", n), **_stats("MIN_MAX", n)} for k, n in ((STATE, 3), (ACTION, 2))}
    g = policy_norm.check(*_pipes("MIN_MAX", stats), _cfg("MEAN_STD"))
    assert not g["ok"] and any("MIN_MAX in the processor, MEAN_STD in the config" in p for p in g["problems"])


def test_required_observation_missing_from_the_frame_fails_the_probe():
    pre, post = _pipes()
    g = policy_norm.check(pre, post, _cfg())
    sample = {k: v for k, v in _sample().items() if k != STATE}
    probe = policy_norm.probe(pre, post, sample, _to_batch, g["required"], action_shape=(2,))
    assert not probe["ok"] and probe["observations"][STATE] == {"mode": "MEAN_STD", "present": False,
                                                                "ok": False}


def test_degenerate_range_is_refused_by_the_guard():
    stats = {STATE: {"min": torch.zeros(3), "max": torch.full((3,), 1e-40)}, ACTION: _stats("MIN_MAX", 2)}
    g = policy_norm.check(*_pipes("MIN_MAX", stats), _cfg("MIN_MAX"))
    assert not g["ok"] and any("degenerate range" in p for p in g["problems"])


def test_non_finite_normalized_observation_fails_the_probe():
    # a range the guard accepts (its reciprocal is finite) that still overflows on a real frame
    stats = {STATE: {"min": torch.zeros(3), "max": torch.full((3,), 1e-37)}, ACTION: _stats("MIN_MAX", 2)}
    pre, post = _pipes("MIN_MAX", stats)
    g = policy_norm.check(pre, post, _cfg("MIN_MAX"))
    assert g["ok"], g["problems"]
    probe = policy_norm.probe(pre, post, _sample(), _to_batch, g["required"], action_shape=(2,))
    assert not probe["ok"] and not probe["observations"][STATE]["finite"]


def test_identity_transform_passes_the_probe():
    unit = {"mean": torch.zeros(3), "std": torch.ones(3)}
    pre, post = _pipes(stats={STATE: unit, ACTION: {"mean": torch.zeros(2), "std": torch.ones(2)}})
    g = policy_norm.check(pre, post, _cfg())
    probe = policy_norm.probe(pre, post, _sample(), _to_batch, g["required"], action_shape=(2,))
    assert g["ok"] and probe["ok"], probe


def test_output_that_does_not_follow_the_formula_fails_the_probe():
    def raw_batch(sample, _pre):   # skips the processors: the state stays raw
        return {k: v.unsqueeze(0) for k, v in sample.items()}

    pre, post = _pipes()
    g = policy_norm.check(pre, post, _cfg())
    probe = policy_norm.probe(pre, post, _sample(), raw_batch, g["required"], action_shape=(2,))
    assert not probe["ok"] and not probe["observations"][STATE]["matches"]


def test_stats_sha256_is_canonical():
    a = {"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([3.0, 4.0])}
    b = {"std": torch.tensor([3.0, 4.0], dtype=torch.float64), "mean": torch.tensor([1.0, 2.0])}
    assert policy_norm.stats_sha256(a) == policy_norm.stats_sha256(b)
    assert policy_norm.stats_sha256(a) != policy_norm.stats_sha256({**a, "std": torch.tensor([3.0, 4.5])})


def test_dataset_overrides_mirror_lerobot_train():
    cfg = SimpleNamespace(input_features={STATE: FEATURES[STATE]}, output_features={ACTION: FEATURES[ACTION]},
                          normalization_mapping={"STATE": NormalizationMode.MEAN_STD})
    stats = {STATE: {}}
    pre, post = policy_norm.overrides(cfg, stats)
    assert pre["normalizer_processor"] == {"features": {STATE: FEATURES[STATE], ACTION: FEATURES[ACTION]},
                                           "norm_map": cfg.normalization_mapping, "stats": stats}
    assert post["unnormalizer_processor"] == {"features": {ACTION: FEATURES[ACTION]},
                                              "norm_map": cfg.normalization_mapping, "stats": stats}
    merged = policy_norm.merge_overrides({"device_processor": {"device": "cpu"}}, pre)
    assert set(merged) == {"device_processor", "normalizer_processor"}


# ---------------------------------------------------------------------------------------------
# --allow-unnormalized and the recorded status
# ---------------------------------------------------------------------------------------------

def _args(*extra):
    return ep.build_parser().parse_args(["--policy", "p", "--dataset", "d", "--out", "o", *extra])


def test_allow_unnormalized_is_recorded(monkeypatch, capsys):
    broken = _pipes(stats={"so100.buffer.action": {"mean": torch.zeros(2), "std": torch.ones(2)}})
    monkeypatch.setattr(ep, "build_processors", lambda *a, **k: broken)
    monkeypatch.setattr(ep, "to_batch", _to_batch)
    cfg = _cfg()
    with pytest.raises(policy_norm.NormalizationError):
        ep.setup_normalization(_args(), cfg, {}, [_sample()], [0])
    norm, stats = ep.setup_normalization(_args("--allow-unnormalized"), cfg, {}, [_sample()], [0])
    assert norm["status"] == policy_norm.LEGACY == "legacy-unverified"
    assert norm["protocol"] == "checkpoint" and stats is None
    assert norm["guard"]["problems"] and not norm["probe"]["ok"]
    assert "legacy-unverified" in capsys.readouterr().out


def test_allow_unnormalized_needs_checkpoint_protocol():
    ap = ep.build_parser()
    args = ap.parse_args(["--policy", "p", "--dataset", "d", "--out", "o",
                          "--norm-stats", "dataset", "--allow-unnormalized"])
    with pytest.raises(SystemExit):
        policy_norm.check_args(ap, args)
    assert _args().norm_stats == "checkpoint" and not _args().allow_unnormalized


# ---------------------------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------------------------

def _norm(protocol="dataset", status=policy_norm.VERIFIED, sha="a" * 64, transform="d" * 64):
    return {"protocol": protocol, "status": status, "stats_sha256": sha, "transform_sha256": transform,
            "stats_source": {"kind": "dataset", "repo_id": "d"},
            "guard": {"pre": {STATE: {"mode": "MEAN_STD", "eps": 1e-8, "sha256": "b" * 64,
                                      "transform_sha256": "e" * 64}}, "post": {}}}


def test_export_carries_the_protocol(tmp_path):
    import algos
    import quant_export
    from test_ladder import FAST, tiny_llama

    m = tiny_llama(1)
    algo = algos.make("none")
    rep = algos.quantize(m, "ref_conv", ["lm_head"], dict(FAST), algo=algo)
    rec = policy_norm.export_record(_norm())
    man = quant_export.export(m, tmp_path, mode="ref_conv", report=rep, algo=algo, algo_spec="none",
                              options=dict(FAST), skip=["lm_head"],
                              model_record={"kind": "lerobot_policy", "id": "tiny"}, normalization=rec)
    for f in ("manifest.json", "policy.json"):
        got = json.loads((tmp_path / f).read_text())["normalization"]
        assert got["protocol"] == "dataset" and got["status"] == "verified"
        assert got["stats_sha256"] == "a" * 64
        assert got["stats_sha256_per_key"]["pre"][STATE] == "b" * 64
        assert got["record_version"] == 2 and got["transform_sha256"] == "d" * 64
        assert got["transform"]["pre"][STATE] == {"mode": "MEAN_STD", "eps": 1e-8, "sha256": "e" * 64}
    assert man["normalization"]["protocol"] == "dataset"
    policy_norm.check_export(man, _norm())                     # same protocol and stats: accepted
    with pytest.raises(policy_norm.NormalizationError, match="re-export"):
        policy_norm.check_export(man, _norm("checkpoint"))
    with pytest.raises(policy_norm.NormalizationError, match="stats"):
        policy_norm.check_export(man, _norm(sha="c" * 64))


def test_pre_protocol_export_counts_as_legacy():
    with pytest.raises(policy_norm.NormalizationError, match="legacy-unverified"):
        policy_norm.check_export({}, _norm())
    policy_norm.check_export({}, _norm("checkpoint", policy_norm.LEGACY, None))


def _guarded_norm(mode, stats):
    g = policy_norm.check(*_pipes(mode, stats), _cfg(mode))
    assert g["ok"], g["problems"]
    return {"protocol": "dataset", "status": policy_norm.VERIFIED, "stats_source": None, "guard": g,
            "stats_sha256": policy_norm.combined_sha256(g),
            "transform_sha256": policy_norm.combined_sha256(g, "transform_sha256")}


def test_export_refuses_a_changed_transform_with_the_same_stats():
    stats = {k: {**_stats("MEAN_STD", n), **_stats("MIN_MAX", n)} for k, n in ((STATE, 3), (ACTION, 2))}
    mean_std, min_max = _guarded_norm("MEAN_STD", stats), _guarded_norm("MIN_MAX", stats)
    assert mean_std["stats_sha256"] == min_max["stats_sha256"]
    manifest = {"normalization": policy_norm.export_record(mean_std)}
    policy_norm.check_export(manifest, mean_std)
    with pytest.raises(policy_norm.NormalizationError, match="transform"):
        policy_norm.check_export(manifest, min_max)


def test_verified_export_without_the_transform_is_refused():
    v1 = {k: v for k, v in policy_norm.export_record(_norm()).items()
          if k not in ("record_version", "transform_sha256", "transform")}
    with pytest.raises(policy_norm.NormalizationError, match="re-export"):
        policy_norm.check_export({"normalization": v1}, _norm())


def test_from_export_mismatch_is_refused_by_the_cli(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"normalization": policy_norm.export_record(
        _norm("checkpoint"))}))
    monkeypatch.setattr(ep, "load_config", lambda *a: SimpleNamespace())
    monkeypatch.setattr(ep, "camera_rename_map", lambda *a: {})
    monkeypatch.setattr(ep, "make_data", lambda *a: ([], SimpleNamespace(fps=1), [], []))
    monkeypatch.setattr(ep, "setup_normalization", lambda *a: (_norm(), {}))
    monkeypatch.setattr(sys, "argv", ["eval_policy.py", "--policy", "p", "--dataset", "d",
                                      "--out", str(tmp_path / "o.json"), "--modes", "bf16", "ref_conv",
                                      "--device", "cpu", "--norm-stats", "dataset",
                                      "--from-export", str(tmp_path)])
    with pytest.raises(SystemExit, match="re-export"):
        ep.main()


# ---------------------------------------------------------------------------------------------
# Real checkpoints (slow; skipped unless cached)
# ---------------------------------------------------------------------------------------------

SMOLVLA, SVLA_DATA = "lerobot/smolvla_base", "lerobot/svla_so100_pickplace"
DIT, DIT_DATA = "TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000", "TRI-ML/PretrainFinetune"


def _cached(repo, data):
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    model = any((home / "hub" / ("models--" + repo.replace("/", "--"))).glob("snapshots/*/config.json"))
    ds = any((home / "lerobot" / "hub" / ("datasets--" + data.replace("/", "--"))).glob("snapshots/*"))
    return model and ds


def _real_args(policy, data, *extra):
    return ep.build_parser().parse_args(["--policy", policy, "--dataset", data, "--out", "o",
                                         "--device", "cpu", "--n-frames", "16", "--n-test-episodes", "8",
                                         *extra])


def _real_setup(policy, data, *extra):
    args = _real_args(policy, data, *extra)
    cfg = ep.load_config(policy, "cpu")
    rename = ep.camera_rename_map(cfg, data)
    ds, _, _, idx = ep.make_data(cfg, data, args.n_test_episodes, args.n_frames, args.seed)
    return args, cfg, rename, ds, idx


needs_smolvla = pytest.mark.skipif(not _cached(SMOLVLA, SVLA_DATA), reason="SmolVLA / svla_so100_pickplace not cached")
needs_dit = pytest.mark.skipif(not _cached(DIT, DIT_DATA), reason="Multi-Task DiT / PretrainFinetune not cached")


@pytest.mark.slow
@needs_smolvla
def test_smolvla_checkpoint_stats_are_refused():
    args, cfg, rename, ds, idx = _real_setup(SMOLVLA, SVLA_DATA)
    with pytest.raises(policy_norm.NormalizationError) as e:
        ep.setup_normalization(args, cfg, rename, ds, idx)
    assert "'observation.state'" in str(e.value) and "'action'" in str(e.value)


@pytest.fixture(scope="module")
def smolvla_dataset():
    if not _cached(SMOLVLA, SVLA_DATA):
        pytest.skip("SmolVLA / svla_so100_pickplace not cached")
    args, cfg, rename, ds, idx = _real_setup(SMOLVLA, SVLA_DATA, "--norm-stats", "dataset")
    norm, stats = ep.setup_normalization(args, cfg, rename, ds, idx)
    return args, cfg, rename, ds, idx, norm, stats


@pytest.mark.slow
def test_smolvla_dataset_stats_normalize_state(smolvla_dataset):
    args, cfg, rename, ds, idx, norm, stats = smolvla_dataset
    assert norm["status"] == "verified" and norm["guard"]["ok"]
    assert norm["stats_source"]["repo_id"] == SVLA_DATA
    assert norm["probe"]["observations"]["observation.state"]["matches"]
    pre, _ = ep.build_processors(cfg, args, rename, stats)
    states = torch.cat([ep.to_batch(ds[i], pre)["observation.state"].float() for i in idx])
    raw = torch.stack([ds[i]["observation.state"].float() for i in idx])
    assert raw.abs().mean() > 10                            # joint degrees
    assert states.mean().abs() < 0.75                       # dataset-normalized
    assert 0.2 < states.std() < 3.0


@pytest.mark.slow
def test_smolvla_action_round_trip_is_identity(smolvla_dataset):
    rt = smolvla_dataset[5]["probe"]["action_round_trip"]
    assert rt["identity"] and rt["matches"] and rt["modes"] == ["MEAN_STD"]
    assert rt["unnormalized_std"] > 5                      # back in degrees


@pytest.mark.slow
@needs_dit
def test_dit_checkpoint_passes_the_guard():
    args, cfg, rename, ds, idx = _real_setup(DIT, DIT_DATA)
    norm, stats = ep.setup_normalization(args, cfg, rename, ds, idx)
    assert stats is None and norm["status"] == "verified", norm["guard"]["problems"]
    assert norm["guard"]["pre"]["observation.state"]["mode"] == "MIN_MAX"
    assert norm["guard"]["pre"]["observation.images.wrist"]["mode"] == "MEAN_STD"
    assert norm["probe"]["ok"] and norm["probe"]["action_round_trip"]["identity"]
