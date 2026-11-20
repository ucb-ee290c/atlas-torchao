"""Independent float64 NumPy forward of the published Keras DroNet (no torch, no shared code).

It re-implements Keras semantics from the layer config directly: TF `same` padding from its own
formula, NHWC tensors, NHWC flatten, batch norm from the moving statistics. It exists so that the
PyTorch port's padding, flatten order, BN folding and head order are checked against something
that does not share them. Slow (direct sliding windows); use it on a handful of frames.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _same_pad(n: int, k: int, s: int):
    out = -(-n // s)
    total = max((out - 1) * s + k - n, 0)
    return total // 2, total - total // 2


def conv_same(x, kernel, bias, stride):  # x (B,H,W,C), kernel (kh,kw,cin,cout)
    kh, kw = kernel.shape[:2]
    (t, b), (l, r) = _same_pad(x.shape[1], kh, stride), _same_pad(x.shape[2], kw, stride)
    x = np.pad(x, ((0, 0), (t, b), (l, r), (0, 0)))
    win = sliding_window_view(x, (kh, kw), axis=(1, 2))[:, ::stride, ::stride]  # B,oh,ow,C,kh,kw
    return np.einsum("bhwcij,ijco->bhwo", win, kernel) + bias


def maxpool_valid(x, k=3, s=2):
    win = sliding_window_view(x, (k, k), axis=(1, 2))[:, ::s, ::s]
    return win.max(axis=(-1, -2))


def bn(x, z, name, eps=1e-3):
    g, b, m, v = (z[f"{name}.{k}"].astype(np.float64) for k in ("gamma", "beta", "moving_mean", "moving_variance"))
    return (x - m) / np.sqrt(v + eps) * g + b


def forward(z, image):
    """image: (B, 200, 200) in [0, 1]. Returns (steer (B,), collision_probability (B,))."""
    z = {k: v.astype(np.float64) if k.endswith(("kernel", "bias")) else v for k, v in z.items()}
    relu = lambda a: np.maximum(a, 0)  # noqa: E731
    c = lambda x, i, s: conv_same(x, z[f"conv2d_{i}.kernel"], z[f"conv2d_{i}.bias"], s)  # noqa: E731
    x1 = maxpool_valid(c(image[..., None].astype(np.float64), 1, 2))
    x3 = c(x1, 4, 2) + c(relu(bn(c(relu(bn(x1, z, "batch_normalization_1")), 2, 2), z, "batch_normalization_2")), 3, 1)
    x5 = c(x3, 7, 2) + c(relu(bn(c(relu(bn(x3, z, "batch_normalization_3")), 5, 2), z, "batch_normalization_4")), 6, 1)
    x7 = c(x5, 10, 2) + c(relu(bn(c(relu(bn(x5, z, "batch_normalization_5")), 8, 2), z, "batch_normalization_6")), 9, 1)
    f = relu(x7.reshape(x7.shape[0], -1))                                   # NHWC flatten, as Keras
    steer = f @ z["steer.kernel"].astype(np.float64) + z["steer.bias"]
    logit = f @ z["collision.kernel"].astype(np.float64) + z["collision.bias"]
    return steer[:, 0], 1 / (1 + np.exp(-logit[:, 0])), logit[:, 0]
