"""DroNet port: parity with the independent NumPy Keras reference, negative controls, native-FP8 capture.

The tests use random weights in the Keras layout (written to a temporary npz), so they need no download.
"""
import json

import algos
import exporter.atlas_fx_ops  # noqa: F401  (registers atlas::qconv2d before any native capture)
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from dronet import keras_ref
from dronet import model as dm
from test_fx_export import PINNED_TORCH
from test_ladder import FAST

CONVS = {1: (5, 1, 32), 2: (3, 32, 32), 3: (3, 32, 32), 4: (1, 32, 32), 5: (3, 32, 64), 6: (3, 64, 64),
         7: (1, 32, 64), 8: (3, 64, 128), 9: (3, 128, 128), 10: (1, 64, 128)}


def random_keras_weights(seed=0):
    g = np.random.default_rng(seed)
    z = {}
    for i, (k, cin, cout) in CONVS.items():
        z[f"conv2d_{i}.kernel"] = (g.normal(size=(k, k, cin, cout)) / np.sqrt(k * k * cin)).astype(np.float32)
        z[f"conv2d_{i}.bias"] = (0.1 * g.normal(size=cout)).astype(np.float32)
    for i, c in enumerate([32, 32, 32, 64, 64, 128], start=1):
        n = f"batch_normalization_{i}"
        z[f"{n}.gamma"] = (1 + 0.2 * g.normal(size=c)).astype(np.float32)
        z[f"{n}.beta"] = (0.1 * g.normal(size=c)).astype(np.float32)
        z[f"{n}.moving_mean"] = (0.1 * g.normal(size=c)).astype(np.float32)
        z[f"{n}.moving_variance"] = (0.5 + g.random(size=c)).astype(np.float32)
    for role in ("steer", "collision"):                       # distinct heads: a swap must change the output
        z[f"{role}.kernel"] = (g.normal(size=(dm.FEATURES, 1)) / np.sqrt(dm.FEATURES)).astype(np.float32)
        z[f"{role}.bias"] = np.array([0.3 if role == "steer" else -0.2], dtype=np.float32)
    return z


def frames(n=3, seed=1):
    g = np.random.default_rng(seed)
    out = [np.clip(np.cumsum(np.cumsum(g.normal(size=(200, 200)), 0), 1) / 200 + 0.5
                   + 0.2 * g.normal(size=(200, 200)), 0, 1) for _ in range(n)]
    return np.stack(out).astype(np.float32)


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    z = random_keras_weights()
    path = tmp_path_factory.mktemp("dronet") / "w.npz"
    np.savez(path, **z)
    return z, path


def torch_forward(model, images):
    with torch.no_grad():
        return model(torch.from_numpy(images)[:, None]).numpy()


def test_port_matches_independent_keras_reference(weights):
    z, path = weights
    images = frames()
    steer, _, logit = keras_ref.forward(z, images)
    out = torch_forward(dm.load_keras_npz(path), images)
    assert np.abs(out[:, 0] - steer).max() < 1e-4
    assert np.abs(out[:, 1] - logit).max() < 1e-4
    assert not np.allclose(steer, logit)                      # the two heads are distinguishable


def test_same_padding_at_stride_two_is_asymmetric():
    # TF SAME, k=5, s=2, n=200 pads (1, 2); the other strided convs of the net are symmetric.
    assert keras_ref._same_pad(200, 5, 2) == (1, 2)
    assert [keras_ref._same_pad(n, 3, 2) for n in (49, 25, 13)] == [(1, 1)] * 3
    assert [keras_ref._same_pad(n, 1, 2) for n in (49, 25, 13)] == [(0, 0)] * 3


def test_negative_controls_are_detected(weights, monkeypatch):
    z, path = weights
    images = frames(2)
    ref_steer, _, ref_logit = keras_ref.forward(z, images)
    good = torch_forward(dm.load_keras_npz(path), images)

    def error(out):
        return max(np.abs(out[:, 0] - ref_steer).max(), np.abs(out[:, 1] - ref_logit).max())

    # symmetric padding instead of TF's (1, 2)
    real_pad = F.pad
    monkeypatch.setattr(dm.F, "pad", lambda x, p: real_pad(x, (2, 2, 2, 2)))
    assert error(torch_forward(dm.load_keras_npz(path), images)) > 100 * error(good)
    monkeypatch.undo()
    # heads swapped
    swapped = dm.load_keras_npz(path)
    swapped.steer, swapped.collision = swapped.collision, swapped.steer
    assert error(torch_forward(swapped, images)) > 100 * error(good)
    # head weights in NHWC order against an NCHW flatten
    wrong = dm.load_keras_npz(path)
    with torch.no_grad():
        wrong.steer.weight.copy_(torch.from_numpy(z["steer.kernel"][:, 0])[None])
    assert error(torch_forward(wrong, images)) > 100 * error(good)


def test_fused_heads_match_the_two_heads(weights):
    _, path = weights
    images = frames(2)
    two = torch_forward(dm.load_keras_npz(path), images)
    one = torch_forward(dm.load_keras_npz(path, fused_heads=True), images)
    assert one.shape == (2, 2)
    assert np.abs(one - two).max() < 1e-5
    fused = dm.load_keras_npz(path, fused_heads=True)
    assert not hasattr(fused, "steer") and fused.head.weight.shape == (2, dm.FEATURES)
    # row 0 is steering, row 1 the collision logit: swapping the rows must change the output
    with torch.no_grad():
        fused.head.weight.copy_(fused.head.weight.flip(0))
        fused.head.bias.copy_(fused.head.bias.flip(0))
    assert np.abs(torch_forward(fused, images) - two).max() > 1e-2


def test_fused_head_gets_a_per_panel_label(weights):
    # a single-row weight is labelled per_tensor by the exporter and rejected by the bundle verifier;
    # the fused two-row head is per_panel(32,partial_tail)
    _, path = weights
    model = dm.load_keras_npz(path, torch.bfloat16, fused_heads=True)
    algo = algos.make("granularity:w=panel,a=token,rows=32,rule=nc")
    report = algos.quantize(model, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(model, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    assert policy["sites"]["head"]["weight"]["granularity"] == "per_panel(32,partial_tail)"
    assert policy["sites"]["conv1"]["weight"]["granularity"] == "per_panel(32)"


def test_boundary_returns_logit_not_probability(weights):
    _, path = weights
    out = torch_forward(dm.load_keras_npz(path), frames(1))
    assert out.shape == (1, 2)
    p = dm.collision_probability(torch.from_numpy(out[:, 1]))
    assert 0 <= float(p) <= 1


@pytest.mark.skipif(torch.__version__ < PINNED_TORCH, reason=f"torch.export captures need torch {PINNED_TORCH}")
@pytest.mark.parametrize("spec", ["none", "granularity:w=channel,a=token,rule=nc"])
def test_native_fp8_capture_carries_every_conv_and_linear(weights, tmp_path, spec):
    _, path = weights
    model = dm.load_keras_npz(path, torch.bfloat16)
    algo = algos.make(spec)
    report = algos.quantize(model, "ref_conv", options=dict(FAST), algo=algo)
    assert report["conv2d"]["quantized"] == 10 and not report["conv2d"]["untouched"]
    policy = quant_export.build_policy(model, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    cases = [(torch.from_numpy(frames(1, 7))[:, None].bfloat16(),),
             (torch.from_numpy(frames(1, 8))[:, None].bfloat16() * 0.3,)]
    out = tmp_path / "fx"
    result = fx_export.capture_fx(model, cases, out, policy=policy, native_fp8="uint8", fresh_process=False)
    assert result["status"] == "passed" and result["weight_sites_verified"] == 12
    inspection = json.loads((out / "graph_inspection.json").read_text())
    assert inspection["native_fp8_conv2d_sites"] == 10 and inspection["native_fp8_linear_sites"] == 2
    assert inspection["still_emulated"]["emulated_modules"] == {}
