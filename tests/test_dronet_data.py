"""DroNet data helpers: preprocessing, label discovery and metrics, on synthetic data (no dataset needed)."""
import os

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from dronet import data  # noqa: E402


def _write(path, bgr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert cv2.imwrite(str(path), bgr)


def test_preprocess_shape_range_and_crop(tmp_path):
    # 480x640 image whose gray level is a function of (row, col) so the crop window is identifiable
    rows = np.arange(480)[:, None].repeat(640, 1)
    cols = np.arange(640)[None, :].repeat(480, 0)
    gray = ((rows // 4 + cols // 8) % 256).astype(np.uint8)
    p = tmp_path / "a.png"
    _write(p, np.stack([gray] * 3, -1))  # gray image saved as BGR: cvtColor keeps the level

    x = data.preprocess(p)
    assert x.shape == (200, 200) and x.dtype == np.float32
    assert 0.0 <= x.min() and x.max() <= 1.0

    ref = cv2.resize(gray, (320, 240))[240 - 200:240, 160 - 100:160 + 100]  # bottom 200 rows, centre 200 cols
    u8 = data.preprocess_u8(p)
    assert u8.dtype == np.uint8 and np.array_equal(u8, ref)
    assert np.array_equal(x, ref.astype(np.float32) * np.float32(1 / 255))


def test_preprocess_color_uses_bgr2gray(tmp_path):
    bgr = np.zeros((240, 320, 3), np.uint8)
    bgr[..., 2] = 255  # pure red in BGR order
    p = tmp_path / "red.png"
    _write(p, bgr)
    expect = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)[-200:, 60:260]
    assert np.array_equal(data.preprocess_u8(p), expect)


def test_central_crop_matches_original_odd_width():
    img = np.arange(10 * 11).reshape(10, 11)
    out = data.central_image_crop(img, 4, 3)
    assert np.array_equal(out, img[7:10, 5 - 2:5 + 2])


def test_discover_ordering_and_labels(tmp_path):
    split = tmp_path / "testing"
    # experiment names sort: exp_a < exp_b; frames sort by first integer in the name, not lexicographically
    for exp, ids, labels in (("exp_b", [2, 10, 1], [0, 1, 1]), ("exp_a", [3, 20], [1, 0])):
        for i in ids:
            _write(split / exp / "images" / f"frame_{i:05d}.jpg", np.full((240, 320, 3), 7, np.uint8))
        os.makedirs(split / exp, exist_ok=True)
        # labels.txt order = sorted-frame order: exp_b frames 1,2,10 -> labels [0,1,1] etc.
        (split / exp / "labels.txt").write_text("\n".join(str(v) for v in labels) + "\n")
    # a non-image file counts toward the frame index but is skipped, as in the original
    (split / "exp_a" / "images" / "frame_00004.txt").write_text("x")  # sorted between 3 and 20 -> index 1
    (split / "exp_a" / "labels.txt").write_text("1\n1\n0\n")

    names, labels, experiments = data.discover(str(split))
    assert experiments == ["exp_a", "exp_b"]
    assert [os.path.basename(n) for n in names] == [
        "frame_00003.jpg", "frame_00020.jpg", "frame_00001.jpg", "frame_00002.jpg", "frame_00010.jpg"]
    # frame_00020 is index 2 of exp_a's listing (3, 4.txt, 20) -> third label
    assert labels.tolist() == [1, 0, 0, 1, 1]


def test_build_and_subset_seed(tmp_path):
    split = tmp_path / "root" / "validation" / "e0"
    for i in range(6):
        _write(split / "images" / f"{i}.jpg", np.full((240, 320, 3), 10 * i, np.uint8))
    (split / "labels.txt").write_text("\n".join("01"[i % 2] for i in range(6)) + "\n")
    full = data.build(str(tmp_path / "root"), "validation")
    assert full["frames"].shape == (6, 200, 200) and full["labels"].tolist() == [0, 1, 0, 1, 0, 1]
    a = data.build(str(tmp_path / "root"), "validation", subset=3, seed=0)
    b = data.build(str(tmp_path / "root"), "validation", subset=3, seed=0)
    assert np.array_equal(a["indices"], b["indices"]) and len(set(a["indices"])) == 3
    assert np.array_equal(a["frames"], full["frames"][a["indices"]])


def test_metrics_toy():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    prob = np.array([0.9, 0.5, 0.1, 0.5, 0.2, 0.0, 0.49, 0.6])  # preds 1,1,0,1,0,0,0,1 (>= 0.5)
    m = data.metrics(prob, labels)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 1, 3)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)          # not equal to precision: the original's bug
    assert m["f1"] == pytest.approx(2 * (0.5 * 2 / 3) / (0.5 + 2 / 3))


def test_metrics_degenerate():
    m = data.metrics(np.zeros(4), np.zeros(4))
    assert m["accuracy"] == 1.0 and m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0
    with pytest.raises(ValueError):
        data.metrics(np.zeros(3), np.zeros(4))
