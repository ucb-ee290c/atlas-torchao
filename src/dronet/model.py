"""DroNet (Loquercio et al., RA-L 2018) ResNet-8 in PyTorch, matching the published Keras model.

Boundary: a preprocessed grayscale frame (B, 1, 200, 200), float, already scaled by 1/255, to
(B, 2) = (steering, collision LOGIT). The Keras model applies a sigmoid to the collision head; that
stays on the host (`collision_probability`), so the exported graph has no transcendental.

Differences from a plain PyTorch ResNet that matter for matching Keras:
  * `padding='same'` at stride 2 is asymmetric. Only the first conv (5x5, 200 -> 100) needs it:
    TF pads (1, 2) per axis. The 3x3 and 1x1 convs see 49 / 25 / 13 pixels, where SAME is
    symmetric (1 for 3x3, 0 for 1x1), so they use plain integer padding.
  * Batch norm is pre-activation (after the max-pool or after an add, before the ReLU), and the
    un-normalised tensor also feeds the 1x1 skip conv. It cannot fold into a neighbouring conv, so
    it is a frozen per-channel affine (`Affine`), scale = gamma / sqrt(var + 1e-3).
  * Keras flattens NHWC. The head weights are permuted once at load time so that flattening the
    NCHW tensor gives the same dot products: no transpose in the graph.
  * The two heads stay two Linear layers (as in Keras) so that per-tensor weight scales are
    chosen per head, then concatenated. `FusedHeadDroNet` (`load_keras_npz(..., fused_heads=True)`)
    stacks them into one Linear(6272, 2) for the tile-aware export: that flow labels a weight with a
    single exponent `per_tensor`, which the bundle verifier rejects, while a two-row weight gets a
    `per_panel(32,partial_tail)` label. The two heads then share one weight exponent.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BN_EPS = 1e-3
FEATURES = 128 * 7 * 7  # final map 7 x 7 x 128 for a 200 x 200 input


class Affine(nn.Module):
    """y = x * scale + shift per channel: an eval-mode batch norm with the statistics folded."""

    def __init__(self, channels: int):
        super().__init__()
        self.register_buffer("scale", torch.ones(1, channels, 1, 1))
        self.register_buffer("shift", torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        return x * self.scale + self.shift


class ResBlock(nn.Module):
    """BN-ReLU-Conv3x3/2, BN-ReLU-Conv3x3, plus a 1x1/2 conv on the raw (pre-BN) input."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.bn_a, self.bn_b = Affine(cin), Affine(cout)
        self.conv_a = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.conv_b = nn.Conv2d(cout, cout, 3, stride=1, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1, stride=2, padding=0)

    def forward(self, x):
        y = self.conv_a(F.relu(self.bn_a(x)))
        y = self.conv_b(F.relu(self.bn_b(y)))
        return self.skip(x) + y


class DroNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, stride=2, padding=0)
        self.pool = nn.MaxPool2d(3, stride=2)
        self.block1, self.block2, self.block3 = ResBlock(32, 32), ResBlock(32, 64), ResBlock(64, 128)
        self.steer = nn.Linear(FEATURES, 1)
        self.collision = nn.Linear(FEATURES, 1)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2))          # TF SAME for k=5, s=2, n=200: (before, after) = (1, 2)
        x = self.pool(self.conv1(x))
        x = self.block3(self.block2(self.block1(x)))
        x = F.relu(x.flatten(1))
        return torch.cat([self.steer(x), self.collision(x)], dim=1)


class FusedHeadDroNet(DroNet):
    """DroNet with the two Linear(6272, 1) heads stacked into `head` = Linear(6272, 2): row 0 steering, row 1 collision."""

    def __init__(self):
        super().__init__()
        del self.steer, self.collision
        self.head = nn.Linear(FEATURES, 2)

    def forward(self, x):
        x = F.pad(x, (1, 2, 1, 2))
        x = self.pool(self.conv1(x))
        x = self.block3(self.block2(self.block1(x)))
        return self.head(F.relu(x.flatten(1)))


def fuse_heads(model: DroNet) -> FusedHeadDroNet:
    """A FusedHeadDroNet with the weights of `model` (same dtype and device)."""
    ref = model.conv1.weight
    fused = FusedHeadDroNet().to(device=ref.device, dtype=ref.dtype)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith(("steer.", "collision."))}
    state["head.weight"] = torch.cat([model.steer.weight, model.collision.weight])
    state["head.bias"] = torch.cat([model.steer.bias, model.collision.bias])
    fused.load_state_dict(state)
    return fused.eval()


def collision_probability(logit: torch.Tensor) -> torch.Tensor:
    """Host-side sigmoid of the collision logit (the Keras model's second output)."""
    return torch.sigmoid(logit.float())


def _kernel(a):  # Keras HWIO -> torch OIHW
    return torch.from_numpy(np.ascontiguousarray(a.transpose(3, 2, 0, 1)))


def _head(kernel):  # Keras (H*W*C, 1) in NHWC order -> (1, C*H*W) in NCHW order
    k = kernel[:, 0].reshape(7, 7, 128).transpose(2, 0, 1).reshape(1, FEATURES)
    return torch.from_numpy(np.ascontiguousarray(k))


def _affine(z, name, dst: Affine):
    scale = z[f"{name}.gamma"] / np.sqrt(z[f"{name}.moving_variance"].astype(np.float64) + BN_EPS)
    shift = z[f"{name}.beta"] - z[f"{name}.moving_mean"] * scale
    dst.scale.copy_(torch.from_numpy(scale.astype(np.float32)).view(1, -1, 1, 1))
    dst.shift.copy_(torch.from_numpy(shift.astype(np.float32)).view(1, -1, 1, 1))


def load_keras_npz(path, dtype=torch.float32, fused_heads: bool = False) -> DroNet:
    """Build DroNet from the npz written by `dronet.convert_keras` (float32 first, then cast).

    `fused_heads=True` returns the `FusedHeadDroNet` form (one Linear(6272, 2)) with identical weights."""
    z = dict(np.load(path))
    m = DroNet()
    conv = {m.conv1: "conv2d_1",
            m.block1.conv_a: "conv2d_2", m.block1.conv_b: "conv2d_3", m.block1.skip: "conv2d_4",
            m.block2.conv_a: "conv2d_5", m.block2.conv_b: "conv2d_6", m.block2.skip: "conv2d_7",
            m.block3.conv_a: "conv2d_8", m.block3.conv_b: "conv2d_9", m.block3.skip: "conv2d_10"}
    bn = {m.block1.bn_a: "batch_normalization_1", m.block1.bn_b: "batch_normalization_2",
          m.block2.bn_a: "batch_normalization_3", m.block2.bn_b: "batch_normalization_4",
          m.block3.bn_a: "batch_normalization_5", m.block3.bn_b: "batch_normalization_6"}
    with torch.no_grad():
        for mod, name in conv.items():
            mod.weight.copy_(_kernel(z[f"{name}.kernel"]))
            mod.bias.copy_(torch.from_numpy(z[f"{name}.bias"]))
        for mod, name in bn.items():
            _affine(z, name, mod)
        for mod, role in ((m.steer, "steer"), (m.collision, "collision")):
            mod.weight.copy_(_head(z[f"{role}.kernel"]))
            mod.bias.copy_(torch.from_numpy(z[f"{role}.bias"]))
    m = m.to(dtype).eval()
    return fuse_heads(m) if fused_heads else m
