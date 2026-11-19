"""atlas-torchao's Atlas numerics against npu-model's RTL-fitted numerics, bit for bit.

The comparison code is src/scripts/npu_model_crosscheck.py (run it directly for a report). Skipped when
no npu-model checkout is found ($NPU_MODEL, default /scratch2/lorenhung/atlas-npu-rtl/src/npu-model)
or the pi0-quant submodule is missing.
"""
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "scripts"))

import npu_model_crosscheck as xc  # noqa: E402
from refutil import need_refs  # noqa: E402

NPU_MODEL = os.environ.get("NPU_MODEL", xc.DEFAULT_NPU_MODEL)

pytestmark = pytest.mark.skipif(
    not (Path(NPU_MODEL) / "npu_model" / "configs" / "rtl_math.py").exists(),
    reason="needs an npu-model checkout ($NPU_MODEL)")

# sin/cos are the only ops that differ: pi0-quant's lane box and the RTL table disagree where the
# Q9.12 input conversion saturates, i.e. |x| >= 2^16 (BF16 exponent field >= 143). Every input
# below that, negative ones included, agrees bit for bit.
SINCOS_DIVERGES_FROM_EXP_FIELD = 143


@pytest.fixture(scope="module")
def R():
    need_refs()
    return xc.load_rtl_math(NPU_MODEL)


def clean(rows):
    bad = {r["name"]: r["mismatches"] for r in rows if r["mismatches"]}
    assert not bad, f"differs from npu-model: {bad}; examples {[r['examples'] for r in rows if r['mismatches']]}"


def test_ipt_tile_and_k_chain_match(R):
    clean(xc.check_ipt_tile(R, n_cases=60) + xc.check_ipt_chain(R))


def test_sa_chain_matches(R):
    clean(xc.check_sa_chain(R))


def test_fp8_pack_pop_unpack_match_for_every_input(R):
    clean(xc.check_fp8(R))


def test_vadd_truncation_and_minmax_match(R):
    clean(xc.check_binary(R, n=50000))


@pytest.mark.parametrize("op", [o for o in xc.UNARY_NAMES if o not in ("sin", "cos")])
def test_unary_tables_match_the_rtl_tables(R, op):
    import chip.chip_ops as co
    allv = xc.as_bf16(torch.arange(65536, dtype=torch.int32))
    clean([xc._result(op, co.unary_table(op), xc._bits16(R.unary(xc.UNARY_NAMES[op], allv)))])


@pytest.mark.parametrize("op", ["sin", "cos"])
def test_sincos_differ_only_where_the_input_saturates(R, op):
    import chip.chip_ops as co
    allv = xc.as_bf16(torch.arange(65536, dtype=torch.int32))
    ours = co.unary_table(op).to(torch.int64)
    theirs = xc._bits16(R.unary(op, allv)).to(torch.int64)
    bad = (ours != theirs) & ~(xc._isnan_bits(ours.to(torch.int32)) & xc._isnan_bits(theirs.to(torch.int32)))
    exp_field = (torch.arange(65536) >> 7) & 0xFF
    assert bad.any(), "expected the documented saturation divergence; the models now agree, update this test"
    assert (exp_field[bad] >= SINCOS_DIVERGES_FROM_EXP_FIELD).all()


def test_verilator_arithmetic_fixture_replays_through_our_models(R):
    """tests/rtl/arithmetic.json holds Verilator output of E4M3FMA, AnchorAccumulationTree and the
    converters: this compares our models against the RTL itself, 512 cases."""
    rows = xc.check_rtl_fixture(NPU_MODEL)
    if not rows:
        pytest.skip("no tests/rtl/arithmetic.json in the npu-model checkout")
    clean(rows)


@pytest.mark.skipif(not Path(xc.DEFAULT_EXPORT).exists(), reason="needs the TinyLlama export_smoke export")
def test_real_export_weight_codes_match_both_matrix_units(R):
    clean(xc.check_real_weights(R))
