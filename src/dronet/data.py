"""DroNet collision data: Keras-exact preprocessing, label discovery, metrics and a cache builder.

Everything here replicates the original DroNet code (uzh-rpg/rpg_public_dronet: `img_utils.load_img`,
`utils.DroneDirectoryIterator._decode_experiment_dir`, `evaluation.py`), so that numbers computed on
our PyTorch port are comparable with the original Keras model.

Preprocessing (img_mode=grayscale, target_size=(320, 240) as (width, height), crop 200x200, rescale 1/255):
  cv2.imread (BGR) -> cv2.cvtColor(BGR2GRAY) -> cv2.resize(img, (320, 240)) -> bottom 200 rows,
  centre 200 columns -> float32 * (1/255).
The original guards the resize with `(img.shape[0], img.shape[1]) != target_size`, i.e. (H, W) against
(W, H); that is only false for a 320 (H) x 240 (W) image, which we replicate rather than "fix".

Label discovery (testing/validation/training directory = a folder of experiments):
  * experiments are the sub-directories, in `sorted(os.listdir)` order;
  * labels are `labels.txt` column 0 (`np.loadtxt`), one per frame;
  * frames come from `os.walk(<exp>/images)` sorted by directory path, and within a directory sorted by the
    integer of the FIRST digit run in the file name; label i goes to the i-th file of that sorted listing
    (the index counts every file, valid image or not, and restarts for each walked directory).

Known bug in the original `evaluation.evaluate_classification`: `recall` is computed with
`metrics.precision_score`, so the "Recall" it prints is a duplicate of precision. `metrics()` here computes
recall properly (TP / (TP + FN)).

CLI:
  python -m dronet.data build --split testing --out /path/collision_test_u8.npz
  python -m dronet.data build --split validation --subset 256 --seed 0 --out /path/collision_val256_u8.npz
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

DEFAULT_ROOT = "/scratch/lorenhung/dronet-data/collision/extracted"
TARGET_SIZE = (320, 240)   # (width, height), as cv2.resize wants it and as common_flags defines it
CROP_SIZE = (200, 200)     # (width, height)
RESCALE = 1.0 / 255
IMAGE_FORMATS = ("png", "jpg")  # the original's accepted extensions (case-insensitive, tested with endswith)


# --------------------------------------------------------------------------- preprocessing

def central_image_crop(img, crop_width=150, crop_heigth=150):
    """Bottom `crop_heigth` rows, `crop_width` columns centred (identical to img_utils.central_image_crop)."""
    half_the_width = int(img.shape[1] / 2)
    return img[img.shape[0] - crop_heigth: img.shape[0],
               half_the_width - int(crop_width / 2): half_the_width + int(crop_width / 2)]


def preprocess_u8(path) -> np.ndarray:
    """File -> (200, 200) uint8 grayscale frame, exactly as the Keras pipeline before the 1/255 rescale."""
    import cv2  # lazy: only needed when reading real images

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cv2.imread could not read {path}")
    if len(img.shape) != 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if (img.shape[0], img.shape[1]) != TARGET_SIZE:
        img = cv2.resize(img, TARGET_SIZE)
    return np.ascontiguousarray(central_image_crop(img, CROP_SIZE[0], CROP_SIZE[1]))


def rescale(u8: np.ndarray) -> np.ndarray:
    """uint8 frames -> float32 in [0, 1] the way the Keras generator does it (float32 array * 1/255)."""
    return u8.astype(np.float32) * np.float32(RESCALE)


def preprocess(path) -> np.ndarray:
    """File -> (200, 200) float32 in [0, 1], the model input (add batch/channel dims yourself)."""
    return rescale(preprocess_u8(path))


# --------------------------------------------------------------------------- discovery

def _natural_key(fname: str) -> int:
    return int(re.search(r"\d+", fname).group())


def decode_experiment_dir(directory: str, subdir: str):
    """(relative file names, labels) of one collision experiment, as `_decode_experiment_dir` does."""
    subpath = os.path.join(directory, subdir)
    labels_filename = os.path.join(subpath, "labels.txt")
    if os.path.exists(os.path.join(subpath, "sync_steering.txt")):
        raise ValueError(f"{subpath} is a steering experiment (sync_steering.txt); only collision data is supported")
    try:
        ground_truth = np.atleast_1d(np.loadtxt(labels_filename, usecols=0))
    except OSError as e:
        raise IOError(f"No labels.txt in {subpath}") from e

    names, labels = [], []
    for root, _, files in sorted(os.walk(os.path.join(subpath, "images")), key=lambda tpl: tpl[0]):
        sorted_files = sorted(files, key=_natural_key)
        for frame_number, fname in enumerate(sorted_files):
            if not any(fname.lower().endswith("." + ext) for ext in IMAGE_FORMATS):
                continue
            if frame_number >= len(ground_truth):
                raise IndexError(f"{subpath}: image #{frame_number} ({fname}) but only {len(ground_truth)} labels")
            names.append(os.path.relpath(os.path.join(root, fname), directory))
            labels.append(ground_truth[frame_number])
    return names, labels


def discover(directory: str):
    """All (relative file name, label) of a split directory, in the original generator's (unshuffled) order."""
    experiments = [d for d in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory, d))]
    names, labels = [], []
    for sub in experiments:
        n, l = decode_experiment_dir(directory, sub)
        names += n
        labels += l
    if not names:
        raise AssertionError(f"Did not find any data in {directory}")
    return names, np.asarray(labels, dtype=np.float32), experiments


def split_dir(root: str, split: str) -> str:
    """`<root>/<split>`, tolerating one wrapper folder (zip top-level) between root and the split."""
    direct = os.path.join(root, split)
    if os.path.isdir(direct):
        return direct
    for entry in sorted(os.listdir(root)):
        cand = os.path.join(root, entry, split)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(f"no '{split}' directory under {root}")


# --------------------------------------------------------------------------- metrics

def metrics(prob, labels, threshold: float = 0.5) -> dict:
    """Binary collision metrics at `prob >= threshold` (the original's rule), pure numpy.

    precision/recall/F1 follow sklearn's defaults (0.0 when a denominator is 0). Recall is the real
    TP/(TP+FN); the original evaluation.py computed 'recall' with precision_score (bug).
    """
    prob = np.asarray(prob, dtype=np.float64).reshape(-1)
    y = np.asarray(labels).reshape(-1)
    if prob.shape != y.shape:
        raise ValueError(f"prob {prob.shape} and labels {y.shape} differ")
    yb = y == 1
    pb = prob >= threshold
    tp, fp = int(np.sum(pb & yb)), int(np.sum(pb & ~yb))
    fn, tn = int(np.sum(~pb & yb)), int(np.sum(~pb & ~yb))
    div = lambda a, b: a / b if b else 0.0  # noqa: E731
    return {"accuracy": (tp + tn) / max(len(y), 1),
            "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn),
            "f1": div(2 * tp, 2 * tp + fp + fn),
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y)), "positives": int(yb.sum())}


# --------------------------------------------------------------------------- cache build / load

def build(root: str, split: str, subset: int | None = None, seed: int = 0, workers: int = 8) -> dict:
    """Preprocess a split into arrays. `subset` draws that many frames uniformly without replacement
    (numpy default_rng(seed)); the chosen indices (into the full discovery order) are kept, sorted."""
    directory = split_dir(root, split)
    names, labels, experiments = discover(directory)
    idx = np.arange(len(names))
    if subset is not None:
        if subset > len(names):
            raise ValueError(f"subset {subset} > {len(names)} frames")
        idx = np.sort(np.random.default_rng(seed).choice(len(names), size=subset, replace=False))
    paths = [os.path.join(directory, names[i]) for i in idx]
    with ThreadPoolExecutor(workers) as ex:
        frames = np.stack(list(ex.map(preprocess_u8, paths)))
    lab = labels[idx]
    if not np.isin(lab, (0.0, 1.0)).all():
        raise ValueError(f"labels outside {{0, 1}}: {np.unique(lab)}")
    return {"frames": frames, "labels": lab.astype(np.uint8), "names": np.asarray([names[i] for i in idx]),
            "indices": idx.astype(np.int64), "num_experiments": np.int64(len(experiments))}


def load_npz(path) -> dict:
    """Read a cache written by `build` (allow_pickle is not needed: names are a unicode array)."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m dronet.data")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="preprocess a split to a uint8 npz")
    b.add_argument("--root", default=DEFAULT_ROOT, help="folder holding <split>/<experiment>/{images,labels.txt}")
    b.add_argument("--split", required=True, help="testing | validation | training")
    b.add_argument("--out", required=True)
    b.add_argument("--subset", type=int, default=None, help="seeded random subset of this many frames")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    d = build(a.root, a.split, a.subset, a.seed, a.workers)
    np.savez(a.out, **d)
    lab = d["labels"]
    print(f"wrote {a.out}: {len(lab)} frames {d['frames'].shape} uint8, positives {int(lab.sum())} "
          f"({100 * lab.mean():.1f}%), experiments in split {int(d['num_experiments'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
