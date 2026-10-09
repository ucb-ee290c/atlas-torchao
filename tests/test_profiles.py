import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import quant_modes

REPO = Path(__file__).resolve().parent.parent
GOLDEN_COMMIT = "daf8e788c4a1bd6d407b82d455822a139501d433"   # atlas-eval before the profile registry

# Builds a small model, quantizes it under `mode`, runs one input and saves outputs and the report.
RUNNER = textwrap.dedent("""
    import sys, torch, torch.nn as nn
    from quant_modes import quantize
    mode, out = sys.argv[1], sys.argv[2]
    torch.manual_seed(0)
    m = nn.Sequential()
    m.add_module("up", nn.Linear(64, 96))
    m.add_module("act", nn.GELU())
    m.add_module("down", nn.Linear(96, 48, bias=False))
    m.add_module("lm_head", nn.Linear(48, 40))
    m = m.to(torch.bfloat16)
    x = (torch.randn(7, 64) * torch.exp2(torch.randn(7, 64) * 3)).to(torch.bfloat16)
    rep = quantize(m, mode, skip=["lm_head"])
    with torch.no_grad():
        y = m(x)
    torch.save({"y": y, "rep": rep, "sd": {k: v.clone() for k, v in m.state_dict().items()}}, out)
""")


def _run(src_dir, mode, out):
    env = {**os.environ, "PYTHONPATH": str(src_dir)}
    subprocess.run([sys.executable, "-c", RUNNER, mode, str(out)], check=True, env=env,
                   capture_output=True)
    return torch.load(out, weights_only=False)


@pytest.fixture(scope="module")
def golden_src(tmp_path_factory):
    d = tmp_path_factory.mktemp("golden_src")
    try:
        for f in ("atlas_quant.py", "atlas_torchao.py", "quant_modes.py", "modelutils.py"):
            blob = subprocess.check_output(["git", "-C", str(REPO), "show", f"{GOLDEN_COMMIT}:src/{f}"])
            (d / f).write_bytes(blob)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("golden commit not available (not a full git checkout)")
    return d


@pytest.mark.parametrize("old,new", [("atlas", "collab_v0"), ("atlas_hw", "collab_v0_hw"),
                                     ("atlas", "atlas"), ("atlas_hw", "atlas_hw")])
def test_collab_v0_golden(golden_src, tmp_path, old, new):
    """collab_v0 (and its aliases) == the atlas-eval code at daf8e78, bit for bit."""
    a = _run(golden_src, old, tmp_path / "old.pt")
    b = _run(REPO / "src", new, tmp_path / "new.pt")
    assert a["rep"] == b["rep"]
    assert torch.equal(a["y"].view(torch.int16), b["y"].view(torch.int16))
    assert a["sd"].keys() == b["sd"].keys()
    for k in a["sd"]:
        assert torch.equal(a["sd"][k].reshape(-1).view(torch.uint8), b["sd"][k].reshape(-1).view(torch.uint8)), k


def test_aliases_resolve():
    assert quant_modes.canonical("atlas") == "collab_v0"
    assert quant_modes.canonical("atlas_hw") == "collab_v0_hw"
    assert {"atlas", "atlas_hw", "collab_v0", "collab_v0_hw", "torchao"} <= set(quant_modes.QUANT_MODES)
    with pytest.raises(ValueError):
        quant_modes.canonical("nope")
