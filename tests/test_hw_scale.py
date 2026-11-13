"""hw_scale: the RV32I-only dynamic-scale exponent (closes H2) equals the model's scale rules on ALL
65,536 BF16 amax patterns, and the chain cost model."""
import pytest
import torch

import hw_scale as H
import ref_numerics as rn
from algos.granularity import scale_exp

BITS = torch.arange(1 << 16, dtype=torch.int32)
X = BITS.to(torch.int16).view(torch.bfloat16).reshape(-1, 1)         # every BF16 pattern, incl. NaN/Inf/subnormal


def _model(rule):
    s = rn.po2_scale_exp(X, (-1,)) if rule == "ref" else scale_exp(X, (-1,), "nc")
    return s.reshape(-1).tolist()


@pytest.mark.parametrize("rule", ["ref", "nc"])
def test_rv32_equals_model_on_all_65536_patterns(rule):
    want = _model(rule)
    for b in range(1 << 16):
        got = H.scale_exp_rv32(b, rule)
        # the chip clamps to the smallest E8M0 scale (byte 0 = 2^-127); the model has no clamp
        assert got == max(want[b], -127), (hex(b), got, want[b])


def test_clamp_only_below_2_pow_minus_119_and_never_above():
    want = _model("nc")
    clamped = [b for b in range(1 << 16) if want[b] < -127]
    assert clamped and all(((b >> 7) & 0xFF) < 8 for b in clamped)      # subnormal / tiny exponent fields
    assert max(H.scale_exp_rv32(b) for b in range(1 << 16)) <= 120      # E8M0 upper end never reached


def test_specials():
    assert H.scale_exp_rv32(0x0000) == H.scale_exp_rv32(0x8000) == 0                # +-0
    assert H.scale_exp_rv32(0x7F80) == H.scale_exp_rv32(0xFF80) == 0                # +-Inf
    assert H.scale_exp_rv32(0x7FC0) == H.scale_exp_rv32(0xFFFF) == 0                # NaN
    assert H.scale_exp_rv32(0x0001) == -127                                          # smallest subnormal
    # 1.75 * 2^E fits (448 after scaling), the next BF16 does not
    assert H.scale_exp_rv32(0x3FE0, "nc") == -8 and H.scale_exp_rv32(0x3FE1, "nc") == -7


def test_only_rv32i_and_counts_are_honest():
    for rule, prog in H.PROGRAMS.items():
        ops = {i[0] for i in prog if isinstance(i, tuple)}
        assert ops <= H.RV32I_ALLOWED and not ops & {"mul", "mulh", "div", "rem", "fadd.s"}
    c = H.program_counts("nc")
    assert c["static"] == 15 and c["normal_path"] == 10 and c["max"] == 14 and c["min"] == 5
    assert H.program_counts("ref")["normal_path"] == 6
    assert H.scale_exp_rv32_counted(0x3F80)[1] == 10


def test_bf16_rounding_is_monotone_so_amax_commutes():
    """BF16 RNE is monotone: amax of the BF16-rounded values == BF16 of the fp32 amax. The model's
    po2_scale_exp (rounds first) therefore equals the chip's vredmax-on-BF16 route exactly."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(512, 257, generator=g) * torch.exp(torch.randn(512, 1, generator=g))
    a1 = x.abs().amax(-1).to(torch.bfloat16)
    a2 = x.to(torch.bfloat16).abs().amax(-1)
    assert torch.equal(a1.view(torch.int16), a2.view(torch.int16))


def test_chain_cost_structure():
    for mode in ("token", "block", "block_vec"):
        c = H.dynamic_scale_cost(128, 2048, mode)
        assert c["instructions"] == sum(c["vpu"].values()) + c["scalar"]["rv32i"] + c["seld"]
        assert c["cycles"] == sum(c["cycle_parts"].values()) > 0
    tok, blk = H.dynamic_scale_cost(128, 2048, "token"), H.dynamic_scale_cost(128, 2048, "block")
    assert tok["seld"] == 0 and blk["seld"] == 4                       # one seld per 32-token block
    assert tok["scalar"]["rv32i"] == 128 * (1 + H.SEQ_NORMAL["nc"] + 3 + 1)
    # the cross-K max grows with K/32
    assert H.dynamic_scale_cost(32, 4096, "block")["scalar"]["rv32i"] > H.dynamic_scale_cost(32, 2048, "block")["scalar"]["rv32i"]
    assert blk["instructions"] < tok["instructions"]
    assert H.dynamic_scale_cost(128, 2048, "block_vec")["cycles"] < tok["cycles"]
