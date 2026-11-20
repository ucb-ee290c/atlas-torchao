"""Convert the published DroNet Keras weights (best_weights.h5) to a plain .npz in Keras layout.

h5py is needed only here (a one-off conversion), not by the model or the export. The npz keeps
Keras layouts (HWIO conv kernels, (in, out) dense kernels, NHWC flatten order) so that
`dronet.model.load_keras_npz` is the only place that knows about the PyTorch layout.

    python -m dronet.convert_keras --h5 .../dronet/best_weights.h5 --json .../dronet/model_struct.json \
        --out /scratch/.../dronet_keras.npz

Source: http://rpg.ifi.uzh.ch/data/dronet_model.zip (linked from github.com/uzh-rpg/rpg_public_dronet).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

CONV_LAYERS = [f"conv2d_{i}" for i in range(1, 11)]
BN_LAYERS = [f"batch_normalization_{i}" for i in range(1, 7)]


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def head_layers(struct: dict) -> dict:
    """Read which Dense is steering and which is collision from the wiring, not from the names.

    model_struct.json lists dense_2 before dense_1; the steering head is the model's first output
    (dense_1) and the collision head feeds the sigmoid (dense_2 -> activation_8).
    """
    cfg = struct["config"]
    by_name = {layer["name"]: layer for layer in cfg["layers"]}
    outputs = [o[0] for o in cfg["output_layers"]]
    steer_out, coll_out = outputs
    if by_name[steer_out]["class_name"] != "Dense":
        raise ValueError(f"first model output {steer_out} is not a Dense layer")
    if by_name[coll_out]["class_name"] != "Activation" or by_name[coll_out]["config"]["activation"] != "sigmoid":
        raise ValueError(f"second model output {coll_out} is not a sigmoid")
    coll_dense = by_name[coll_out]["inbound_nodes"][0][0][0]
    if by_name[coll_dense]["class_name"] != "Dense" or coll_dense == steer_out:
        raise ValueError("steering and collision heads are not two distinct Dense layers")
    return {"steer": steer_out, "collision": coll_dense}


def convert(h5_path, json_path, out_path) -> dict:
    import h5py  # one-off dependency, kept out of the package import path

    struct = json.loads(Path(json_path).read_text())
    heads = head_layers(struct)
    bn_eps = {layer["config"]["epsilon"] for layer in struct["config"]["layers"]
              if layer["class_name"] == "BatchNormalization"}
    if bn_eps != {1e-3}:
        raise ValueError(f"unexpected BatchNormalization epsilons {bn_eps}")
    arrays = {}
    with h5py.File(h5_path, "r") as f:
        for name in CONV_LAYERS:
            arrays[f"{name}.kernel"] = f[name][name]["kernel:0"][()]
            arrays[f"{name}.bias"] = f[name][name]["bias:0"][()]
        for name in BN_LAYERS:
            for key in ("gamma", "beta", "moving_mean", "moving_variance"):
                arrays[f"{name}.{key}"] = f[name][name][f"{key}:0"][()]
        for role, name in heads.items():
            arrays[f"{role}.kernel"] = f[name][name]["kernel:0"][()]
            arrays[f"{role}.bias"] = f[name][name]["bias:0"][()]
    for key, value in arrays.items():
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"{key}: expected finite float32, found {value.dtype}")
    np.savez(out_path, **arrays)
    record = {"source_h5_sha256": _sha256(h5_path), "source_json_sha256": _sha256(json_path),
              "heads": heads, "bn_eps": 1e-3, "tensors": len(arrays), "npz_sha256": _sha256(out_path)}
    Path(str(out_path) + ".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h5", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(convert(a.h5, a.json, a.out), indent=2))
