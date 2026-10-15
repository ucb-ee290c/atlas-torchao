"""keep_bf16: keep the layers an explicit structural rule selects in BF16 (NEW; docs/algos/keep-bf16.md).

Opt-in only: no recipe includes it. Stack it onto a recipe or algorithm, e.g.
`--algo atlas_v1_awq+keep_bf16:rule=input_conv`. The selection is derived from the module tree
(types and shapes), never from a model name or a module path, and every selected site is listed
with the rule's own reason in the algo record (`bf16_opt_out_exact`, `bf16_opt_out_reasons`) and
in an export's policy.json.

Parameters (--algo keep_bf16:k=v,...):
  rule=input_conv  (required) every nn.Conv2d with in_channels == `channels` and groups == 1: a
                   convolution that reads image pixels directly, such as the patch embedding of a
                   ViT / CLIP / SigLIP vision tower or the stem of a CNN encoder. NEW structural
                   rule: the channel count of an image input is the proxy for "consumes the model's
                   pixel input". A deeper convolution that happens to have `channels` input
                   channels would also be selected; the record lists every selected fqn so that
                   this is visible. A model without such a convolution (TinyLlama) selects nothing.
  channels=N       image channels (default 3, RGB)

The selected sites are matched by exact fqn (algos.quantize: validated as nn.Conv2d before
calibration), so similarly named siblings stay quantized. BF16 has no MXU operand path: on chip
these convolutions are host / VPU work (pending ruling R4).
"""

from __future__ import annotations

import torch.nn as nn

from algos.base import Algo

RULES = ("input_conv",)
DEFAULT_CHANNELS = 3   # RGB


class KeepBF16(Algo):
    name = "keep_bf16"

    def __init__(self, **params):
        unknown = set(params) - {"rule", "channels"}
        if unknown:
            raise ValueError(f"keep_bf16: unknown parameters {sorted(unknown)}")
        if params.get("rule") not in RULES:
            raise ValueError(f"keep_bf16: rule= is required, one of {RULES}")
        channels = params.get("channels", DEFAULT_CHANNELS)
        if not isinstance(channels, int) or channels < 1:
            raise ValueError(f"keep_bf16: channels must be a positive integer, got {channels!r}")
        super().__init__(**params)
        self.kept = {}

    @property
    def rule(self):
        return self.params["rule"]

    @property
    def channels(self):
        return self.params.get("channels", DEFAULT_CHANNELS)

    def select(self, model) -> dict:
        """{fqn: reason} of the sites the rule selects, in module order."""
        selected = {}
        for n, m in model.named_modules():
            if n and type(m) is nn.Conv2d and m.in_channels == self.channels and m.groups == 1:
                selected[n] = (f"keep_bf16 rule={self.rule}: Conv2d in_channels={m.in_channels} == "
                               f"channels={self.channels}, reads image pixels directly (NEW structural "
                               "rule, docs/algos/keep-bf16.md); no MXU path for BF16 operands (pending R4)")
        return selected

    def bf16_exact(self, model, mode):
        self.kept = self.select(model)
        return dict(self.kept)

    def record(self) -> dict:
        return {**super().record(), "rule": self.rule, "channels": self.channels,
                "selected": list(self.kept)}


ALGO = KeepBF16
