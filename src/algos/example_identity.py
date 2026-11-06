"""example_identity: a no-op algorithm that exercises every hook (NEW; a template for tracks).

Every hook returns what the reference profile would have used anyway, but through the explicit
algorithm path: weight and activation scales are the reference rule floor(log2(amax / 256))
(ref_numerics.po2_scale_exp, a port of pi0-quant hw_float_ops.py:38-57) handed back as a Scale,
the online transform is the identity, and nothing is folded. Its outputs must therefore equal
`--algo none` bit for bit (tests/test_algo_base.py), which is what makes it a plumbing check.

Parameters:
  calib=true     run (f) calibrate: observe every Linear / Conv2d input amax (recorded, unused)
  bf16=SUBSTR    (e) keep the layers whose fqn contains SUBSTR in BF16 (this changes outputs)
"""

from __future__ import annotations

import reference.ref_numerics as rn
from algos.base import Algo, Scale, observe_inputs


class ExampleIdentity(Algo):
    name = "example_identity"

    @property
    def needs_calibration(self):
        return bool(self.params.get("calib", False))

    def bf16_layers(self, model, mode):
        sub = self.params.get("bf16")
        return [n for n, _ in model.named_modules() if sub and n and str(sub) in n
                and type(_).__name__ in ("Linear", "Conv2d")]

    def calibrate(self, model, batches, run):
        amax = observe_inputs(model, batches, run)
        return {"layers_observed": len(amax)}

    def prepare(self, model, mode):
        return {"folded": 0}

    def transform_weight(self, name, w, kind):
        return w

    def weight_scale(self, name, w, kind):
        return Scale(rn.po2_scale_exp(w), "per_tensor")

    def act_scale(self, name, x, kind, dims=None):
        return Scale(rn.po2_scale_exp(x, dims), "per_tensor" if dims is None else "per_head")

    def online_transform(self, name, kind):
        return _identity


def _identity(x):
    return x


ALGO = ExampleIdentity
