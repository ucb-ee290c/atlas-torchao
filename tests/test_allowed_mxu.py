"""--allowed-mxu / --site-allowed-mxu (docs/algos/export.md#allowed-mxu-units): the opt-in policy.json
fields that let the compiler run the FP8 contractions on MXU0 (systolic array) as well as MXU1
(inner-product tree). Without the flag, policy.json must stay exactly as before."""
import argparse
import json

import pytest
import torch

import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export
from test_export import IDS, SPECS, _calib, _logits, _reload
from test_ladder import FAST, tiny_llama

import algos

# the accumulate block every export wrote before --allowed-mxu existed (ref_conv: framework accumulation)
DEFAULT_ACCUMULATE = {
    "unit": "MXU",
    "exported_mode_accumulation": "framework",
    "assumed_on_chip": "MXU inner-product tree (--mxu ipt, the reference author's choice; accum_backends) "
                       "with BF16 readout",
    "readout": "bf16",
    "mxu": "ipt",
}
NEW_ACCUMULATE_KEYS = {"allowed_mxu", "allowed_mxu_semantics", "k_order", "units"}


def _export(tmp_path, name, spec=SPECS[1], **kw):
    m = tiny_llama(2)
    algo = algos.make(spec)
    batches, calib = _calib()
    rep = algos.quantize(m, "ref_conv", ["lm_head"], dict(FAST), algo=algo, calib=calib)
    d = tmp_path / name
    quant_export.export(m, d, mode="ref_conv", report=rep, algo=algo, algo_spec=spec, options=dict(FAST),
                        skip=["lm_head"], model_record={"kind": "causal_lm", "id": "tiny"},
                        calib_batches=batches, calib_record={"source": "test"}, **kw)
    return m, d, json.loads((d / "policy.json").read_text())


def test_default_policy_unchanged(tmp_path):
    """No flag: the accumulate block is the literal old one and no site carries allowed_mxu; passing
    allowed_mxu=None explicitly writes the same bytes."""
    _, d0, pol = _export(tmp_path, "default")
    assert pol["accumulate"] == DEFAULT_ACCUMULATE
    assert list(pol["accumulate"]) == list(DEFAULT_ACCUMULATE)          # key order too
    assert not any("allowed_mxu" in s for s in pol["sites"].values())
    _, d1, _ = _export(tmp_path, "explicit_none", allowed_mxu=None, site_allowed_mxu={})
    assert (d0 / "policy.json").read_bytes() == (d1 / "policy.json").read_bytes()


def test_allowed_mxu_fields_and_nothing_else(tmp_path):
    _, d0, pol0 = _export(tmp_path, "default")
    m, d1, pol1 = _export(tmp_path, "both", allowed_mxu=["mxu1", "mxu0"])
    acc = pol1["accumulate"]
    assert acc["allowed_mxu"] == ["mxu1", "mxu0"]
    assert set(acc) == set(DEFAULT_ACCUMULATE) | NEW_ACCUMULATE_KEYS
    assert acc["k_order"]["order"] == "ascending" and acc["k_order"]["tile"] == 32
    assert acc["units"]["mxu0"]["mxu_model"] == "sa" and acc["units"]["mxu1"]["mxu_model"] == "ipt"
    assert "every MAC" in acc["units"]["mxu0"]["rounding"] and "per tile" in acc["units"]["mxu1"]["rounding"]
    # removing the new keys gives back the default policy exactly
    for k in NEW_ACCUMULATE_KEYS:
        del acc[k]
    assert pol1 == pol0
    # the loader accepts the new fields and rebuilds the same model
    assert torch.equal(_logits(m, IDS), _logits(_reload(d1, "ref_conv", dict(FAST)), IDS))


def test_site_override(tmp_path):
    site = "model.layers.0.mlp.down_proj"
    m, d, pol = _export(tmp_path, "site", allowed_mxu="mxu1,mxu0", site_allowed_mxu={site: ["mxu1"]})
    assert pol["accumulate"]["allowed_mxu"] == ["mxu1", "mxu0"]
    assert pol["sites"][site]["allowed_mxu"] == ["mxu1"]
    assert [n for n, s in pol["sites"].items() if "allowed_mxu" in s] == [site]
    load_quantized.read_export(d)                                        # digests still match
    # an attention site can be narrowed too
    att = next(n for n, s in pol["sites"].items() if s["kind"] == "attention")
    _, _, pol2 = _export(tmp_path, "site_att", allowed_mxu=["mxu1", "mxu0"], site_allowed_mxu={att: ["mxu1"]})
    assert pol2["sites"][att]["allowed_mxu"] == ["mxu1"]


@pytest.mark.parametrize("kw,match", [
    ({"allowed_mxu": ["mxu1"], "site_allowed_mxu": {"model.layers.0.mlp.down_proj": ["mxu0"]}}, "not in"),
    ({"allowed_mxu": ["mxu1", "mxu0"], "site_allowed_mxu": {"model.layers.9.mlp.down_proj": ["mxu1"]}},
     "not a quantized site"),
    ({"allowed_mxu": ["mxu1", "mxu0"], "site_allowed_mxu": {"lm_head": ["mxu1"]}}, "not a quantized site"),
    ({"allowed_mxu": None, "site_allowed_mxu": {"model.layers.0.mlp.down_proj": ["mxu1"]}}, "needs --allowed-mxu"),
    ({"allowed_mxu": ["mxu2"]}, "unknown"),
    ({"allowed_mxu": ["mxu1", "mxu1"]}, "twice"),
    ({"allowed_mxu": []}, "empty"),
])
def test_export_refuses_bad_lists(tmp_path, kw, match):
    with pytest.raises(ValueError, match=match):
        _export(tmp_path, "bad", **kw)


def test_parsers():
    assert quant_export.parse_allowed_mxu(None) is None
    assert quant_export.parse_allowed_mxu("mxu1, mxu0") == ["mxu1", "mxu0"]
    assert quant_export.parse_allowed_mxu("mxu0") == ["mxu0"]
    assert quant_export.parse_site_allowed_mxu(["a.b=mxu1", "c=mxu1,mxu0"]) == {"a.b": ["mxu1"], "c": ["mxu1", "mxu0"]}
    for bad in (["a.b"], ["=mxu1"], ["a=mxu1", "a=mxu0"]):
        with pytest.raises(ValueError):
            quant_export.parse_site_allowed_mxu(bad)


def _cli(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["bf16", "ref_conv"])
    quant_export.add_args(ap)
    args = ap.parse_args(argv)
    return ap, args


def test_cli_flags():
    ap, args = _cli([])
    assert args.allowed_mxu is None and args.site_allowed_mxu is None
    assert quant_export.check_args(ap, args) is None
    assert quant_export.allowed_mxu_kwargs(args) == {"allowed_mxu": None, "site_allowed_mxu": {}}
    ap, args = _cli(["--export-dir", "x", "--allowed-mxu", "mxu1,mxu0",
                     "--site-allowed-mxu", "model.layers.0.mlp.down_proj=mxu1"])
    assert quant_export.check_args(ap, args) == "ref_conv"
    assert quant_export.allowed_mxu_kwargs(args) == {
        "allowed_mxu": ["mxu1", "mxu0"], "site_allowed_mxu": {"model.layers.0.mlp.down_proj": ["mxu1"]}}
    for argv in (["--allowed-mxu", "mxu1,mxu0"],                                   # no --export-dir
                 ["--export-dir", "x", "--allowed-mxu", "mxu3"],
                 ["--export-dir", "x", "--site-allowed-mxu", "a=mxu1"],            # no --allowed-mxu
                 ["--export-dir", "x", "--allowed-mxu", "mxu1", "--site-allowed-mxu", "a=mxu0"]):
        ap, args = _cli(argv)
        with pytest.raises(SystemExit):
            quant_export.check_args(ap, args)


def test_fields_survive_the_fx_policy_copy(tmp_path):
    """fx_export writes fx/policy.json from pack_plan.with_plan(policy): a deep copy plus the plan."""
    import exporter.pack_plan as pack_plan

    site = "model.layers.1.self_attn.o_proj"
    _, _, pol = _export(tmp_path, "fx", allowed_mxu=["mxu1", "mxu0"], site_allowed_mxu={site: ["mxu1"]})
    fx = pack_plan.with_plan(pol)
    assert fx["accumulate"] == pol["accumulate"]
    assert fx["sites"][site]["allowed_mxu"] == ["mxu1"]


def test_kind_override(tmp_path):
    """kind:linear narrows every quantized Linear; an exact site entry wins over its kind."""
    site = "model.layers.0.mlp.down_proj"
    _, _, pol = _export(tmp_path, "kind", allowed_mxu=["mxu1", "mxu0"],
                        site_allowed_mxu={site: ["mxu1", "mxu0"], "kind:linear": ["mxu1"]})
    for n, s in pol["sites"].items():
        if s["status"] != "quantized":
            assert "allowed_mxu" not in s
        elif n == site:
            assert s["allowed_mxu"] == ["mxu1", "mxu0"]
        elif s["kind"] == "linear":
            assert s["allowed_mxu"] == ["mxu1"]
        else:                                   # attention: the global list applies
            assert "allowed_mxu" not in s
    with pytest.raises(ValueError, match="unknown site kind"):
        _export(tmp_path, "badkind", allowed_mxu=["mxu1", "mxu0"], site_allowed_mxu={"kind:matmul": ["mxu1"]})
    with pytest.raises(ValueError, match="no quantized conv2d"):
        _export(tmp_path, "noconv", allowed_mxu=["mxu1", "mxu0"], site_allowed_mxu={"kind:conv2d": ["mxu1"]})
