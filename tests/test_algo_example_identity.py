"""example_identity: every hook through the explicit-algorithm path, outputs equal to --algo none.
(The cross-tree check on every ladder step is in test_algo_base.py; this is the per-track template.)"""
import copy

import pytest
import torch

import algos
from refutil import need_refs
from test_ladder import FAST, ConvNet, tiny_llama


@pytest.mark.parametrize("mode", ["collab_v0", "ref_linear", "ref_attn", "ref_conv", "reference_hw", "chip"])
def test_identity_equals_none(mode):
    need_refs()
    for build, inp, skip in ((tiny_llama, torch.randint(0, 128, (1, 16), generator=torch.Generator().manual_seed(1)),
                              ["lm_head"]),
                             (lambda: ConvNet().to(torch.bfloat16),
                              torch.randn(2, 3, 8, 8, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16), [])):
        torch.manual_seed(0)
        m0 = build()
        m1 = copy.deepcopy(m0)
        algos.quantize(m0, mode, skip, dict(FAST), algo=None)
        rep = algos.quantize(m1, mode, skip, dict(FAST), algo=algos.make("example_identity"))
        with torch.no_grad():
            y0, y1 = m0(inp), m1(inp)
        y0, y1 = getattr(y0, "logits", y0), getattr(y1, "logits", y1)
        assert torch.equal(y0.view(torch.int16), y1.view(torch.int16)), mode
        rec = rep["algo"]
        assert rec["name"] == "example_identity" and rec["applied"]
        if mode != "collab_v0":
            assert rec["module_hooks_applied"] and rec["granularity"]["weight"]
            assert all(k.endswith(("per_tensor", "per_head")) for k in rec["granularity"]["activation"])
