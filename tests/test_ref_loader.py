import pytest

import reference.ref_loader as ref_loader


def _need_refs():
    bad = ref_loader.check_pins()
    if bad:
        pytest.skip(f"reference submodules not initialised at their pins: {bad}")


def test_submodules_at_pins():
    _need_refs()
    assert ref_loader.check_pins() == {}


def test_load_file_and_provenance():
    _need_refs()
    fp8 = ref_loader.load_file("pi0-quant", "funct_models_vector/fp8_e4m3.py")
    assert fp8.bf16_to_e4m3_byte(0x3F80, 0) == 0x38          # 1.0 -> E4M3 1.0
    hw = ref_loader.import_ref("pi0_inout.hw_float_ops")
    assert hasattr(hw, "compute_po2_scale_exp")
    prov = ref_loader.provenance()
    files = prov["pi0-quant"]["files"]
    assert prov["pi0-quant"]["commit"] == ref_loader.PINS["pi0-quant"]
    assert "funct_models_vector/fp8_e4m3.py" in files and "pi0_inout/hw_float_ops.py" in files
    assert all(len(h) == 64 for h in files.values())
