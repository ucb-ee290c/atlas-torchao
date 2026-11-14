"""SmolVLA capture without in-graph cumsum, dynamic shapes or convolution (fx_export):
  - the text / expert attention mask and position ids are host inputs (smolvla_attention_inputs: lerobot's own
    make_att_2d_masks and cumsum), at the SmolVLATextLayers and SmolVLAFlowLoss boundaries; the captured graph holds no
    cumsum, its masked softmaxes add a BF16 mask placeholder, and its output is byte-identical to the old boundary
    that built both in the graph;
  - the SmolVLM vision embeddings take the static position-id path (an arange, checked against the module's own ids)
    during the capture, so no data-dependent (tensor<?xi64>) value remains;
  - a patch-embedding Conv2d (stride == kernel, no padding) is captured as its patch rows + atlas.qlinear, bit-identical
    to atlas.qconv2d on the unfolded rows; every other Conv2d keeps atlas.qconv2d;
  - the padded-query diagonal (smolvla_attention_inputs keep_self_for_padded) leaves every row non-empty and is exact
    without quantization; it stays off by default (see test_padded_diagonal_*)."""

import json
from types import SimpleNamespace

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers the atlas ops)
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
import pytest
import torch
import torch.nn.functional as F
from test_fx_export import PINNED_TORCH
from test_ladder import FAST

pinned = pytest.mark.skipif(torch.__version__ < PINNED_TORCH, reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")


# -- host attention inputs ------------------------------------------------------------------------------------------------

def _pad_ar(n=6, m=3, padded=(3, 5)):
    pad = torch.ones(1, n + m, dtype=torch.bool)
    pad[0, padded[0]:padded[1]] = False
    ar = torch.cat([torch.zeros(1, n - 1), torch.ones(1, 2), torch.zeros(1, m - 1)], dim=1)     # prefix block, then actions
    return pad, ar


def test_host_attention_inputs_are_lerobots_mask_and_positions():
    pytest.importorskip("lerobot.policies.common.vla_utils")
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    pad, ar = _pad_ar()
    mask, positions, keep = fx_export.smolvla_attention_inputs(pad, ar)
    assert torch.equal(keep, make_att_2d_masks(pad, ar))
    assert mask.dtype == torch.bfloat16 and mask.shape == (1, 9, 9)
    assert torch.equal(mask, fx_export.additive_attention_mask(keep))
    assert positions.dtype == torch.int64 and torch.equal(positions, torch.cumsum(pad, dim=1) - 1)
    assert not bool(keep.any(-1).all())                                   # the padded queries keep nothing
    mask2, positions2, keep2 = fx_export.smolvla_attention_inputs(pad, ar, keep_self_for_padded=True)
    assert bool(keep2.any(-1).all()) and torch.equal(positions2, positions)
    changed = keep2 & ~keep                                               # only the padded rows' diagonal
    assert changed.nonzero().tolist() == [[0, 3, 3], [0, 4, 4]]


# -- tiny SmolVLMWithExpertModel boundaries ---------------------------------------------------------------------------------

class _OldTextLayers(torch.nn.Module):
    """The previous SmolVLATextLayers: mask (make_att_2d_masks) and position ids (cumsum) built in the graph."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, prefix, suffix, pad, ar):
        from lerobot.policies.common.vla_utils import make_att_2d_masks

        (_, out), _ = self.policy.model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(pad, ar), position_ids=torch.cumsum(pad, dim=1) - 1,
            past_key_values=None, inputs_embeds=[prefix, suffix], use_cache=False)
        return out


def _tiny_policy(quantize=True):
    from test_algo_smoothquant import _smolvla

    model = _smolvla().to(torch.bfloat16)
    policy = torch.nn.Module()
    policy.model = torch.nn.Module()
    policy.model.vlm_with_expert = model
    qpolicy = None
    if quantize:
        algo = algos.make("granularity:w=channel,a=token")
        report = algos.quantize(policy, "ref_conv", [], options=dict(FAST), algo=algo)
        qpolicy = quant_export.build_policy(policy, "ref_conv", report, algo, dict(FAST), [],
                                            report["algo"].get("bf16_opt_out", []), {}, {})
    return policy, qpolicy


def _raw_cases(padded):
    g = torch.Generator().manual_seed(7)
    cases = []
    for i in range(2):
        pad, ar = _pad_ar(padded=(3, 5) if (padded and i) else (0, 0))
        cases.append((torch.randn(1, 6, 64, generator=g).to(torch.bfloat16),
                      torch.randn(1, 3, 48, generator=g).to(torch.bfloat16), pad, ar))
    return cases


@pinned
@pytest.mark.parametrize("native,host_rope", [(None, False), ("uint8", False), ("uint8", True)])
def test_text_layers_boundary_takes_the_host_mask_and_positions(tmp_path, native, host_rope):
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    import lerobot.policies.smolvla.smolvlm_with_expert as swe

    original_rope = swe.apply_rope
    policy, qpolicy = _tiny_policy()
    raw = _raw_cases(padded=True)
    boundary = fx_export.SmolVLATextLayers(policy, layers=2, host_rope=host_rope).eval()
    cases = []
    for prefix, suffix, pad, ar in raw:
        mask, positions, _ = fx_export.smolvla_attention_inputs(pad, ar)
        if host_rope:
            tables = fx_export.smolvla_rope_tables(positions, prefix.shape[1], 16)
            cases.append((prefix, suffix, mask, *(t for name in fx_export.ROPE_TABLES for t in tables[name])))
        else:
            cases.append((prefix, suffix, mask, positions))
    with torch.no_grad():
        saved, swe.apply_rope = swe.apply_rope, original_rope
        before = [_OldTextLayers(policy).eval()(*case) for case in raw]   # the old boundary, also at 2 layers
        swe.apply_rope = saved
    result = fx_export.capture_fx(boundary, cases, tmp_path / "fx", policy=qpolicy, fresh_process=False,
                                  native_fp8=native)
    boundary.restore()
    assert swe.apply_rope is original_rope
    assert result["status"] == "passed"
    assert result["graph_shape_census"] == {"dynamic_nodes": [], "cumsum_nodes": []}
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    user = [s for s in program.graph_signature.input_specs if s.kind.name == "USER_INPUT"]
    vals = [next(n for n in program.graph.nodes if n.name == s.arg.name).meta["val"] for s in user]
    tail = ([(torch.float32, (1, n, 1, 8))] * 2 for n in (9, 6, 3)) if host_rope else ([(torch.int64, (1, 9))],)
    assert [(v.dtype, tuple(v.shape)) for v in vals] == [
        (torch.bfloat16, (1, 6, 64)), (torch.bfloat16, (1, 3, 48)), (torch.bfloat16, (1, 9, 9))] + sum(tail, [])
    assert all(next(n for n in program.graph.nodes if n.name == s.arg.name).users for s in user)
    forms = fx_export.softmax_mask_forms(program.graph)
    # layer 0 joint self-attention; layer 1 prefix self-attention + expert cross-attention
    assert [row["form"] for row in forms] == ["additive_bf16_min"] * 3
    assert {row["mask_source"] for row in forms} == {user[2].arg.name}         # the mask placeholder, no where
    calls = [n for n in program.graph.nodes if n.op == "call_function"]
    assert not [n for n in calls if "cumsum" in str(n.target)]
    # drop_dead_pure_nodes (a reloaded program names the unused results of a multi-output op '<op>_unused_<i>')
    assert not [n for n in calls if not n.users and not n.is_impure() and "_unused_" not in n.name]
    if host_rope:                                                         # no RoPE angle math left in the graph
        assert not [n for n in calls if str(n.target) in ("aten.sin.default", "aten.cos.default", "aten.pow.Scalar")]
    mins = (fx_export.BF16_MIN, torch.finfo(torch.float32).min)                # no mask built in the graph
    assert not [n for n in calls if "where" in str(n.target) and any(a in mins for a in n.args if isinstance(a, float))]
    # byte-identical to the old in-graph mask / position boundary on the same raw inputs
    with torch.no_grad():
        for case, old_out in zip(cases, before):
            assert torch.equal(program.module()(*case).view(torch.int16), old_out.view(torch.int16))


def test_text_layers_restores_layer_counts():
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    policy, _ = _tiny_policy(quantize=False)
    inner = policy.model.vlm_with_expert
    boundary = fx_export.SmolVLATextLayers(policy, layers=2)
    assert (inner.num_vlm_layers, inner.num_expert_layers) == (2, 2)
    boundary.restore()
    assert (inner.num_vlm_layers, inner.num_expert_layers) == (4, 4)


# -- padded-query diagonal ------------------------------------------------------------------------------------------------

def _run(policy, layers, case, keep_self, additive=True):
    prefix, suffix, pad, ar = case
    mask, positions, keep = fx_export.smolvla_attention_inputs(pad, ar, keep_self_for_padded=keep_self)
    boundary = fx_export.SmolVLATextLayers(policy, layers).eval()
    try:
        with torch.no_grad():
            return boundary(prefix, suffix, mask if additive else keep, positions), mask
    finally:
        boundary.restore()


def test_padded_diagonal_is_exact_without_quantization():
    """Padded columns are masked for every valid query (exp is exactly 0 for any finite score) and the suffix output
    reads no padded position, so the padded rows' own outputs never reach a valid one: bit-identical at every depth,
    and with the diagonal no row is empty (the backend's masked-softmax policy)."""
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    policy, _ = _tiny_policy(quantize=False)
    case = _raw_cases(padded=True)[1]
    for layers in (1, 2, 4):
        without, mask0 = _run(policy, layers, case, False, additive=False)
        with_diag, mask1 = _run(policy, layers, case, True, additive=False)
        assert not bool((mask0 == 0).any(-1).all()) and bool((mask1 == 0).any(-1).all())
        assert torch.equal(without.view(torch.int16), with_diag.view(torch.int16)), layers


def test_padded_diagonal_is_not_exact_under_per_head_dynamic_operand_exponents():
    """Under the reference attention rule the q / k / v exponents are one per (batch, head) matrix over all tokens,
    padded ones included. The diagonal changes the padded tokens' hidden states from the next layer on, so it can move
    an exponent the valid rows share: measured on the real policy (atlas_v1_awq_tile_static, 5 padded language tokens),
    the suffix output is byte-identical through 4 layers and differs at 16. This test pins the mechanism on the tiny
    model: the padded rows' values enter the shared exponent."""
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    from algos.granularity import scale_exp

    g = torch.Generator().manual_seed(0)
    k = torch.randn(1, 2, 9, 16, generator=g).to(torch.bfloat16)
    k2 = k.clone()
    k2[:, :, 3:5] *= 64                                                    # padded tokens 3, 4 grow
    e1, e2 = scale_exp(k, (-2, -1), "ref"), scale_exp(k2, (-2, -1), "ref")
    assert e1.shape == (1, 2, 1, 1) and bool((e2 > e1).all())             # the valid rows' shared exponent moved


# -- static vision position ids -------------------------------------------------------------------------------------------

def _vision_embeddings():
    smolvlm = pytest.importorskip("transformers.models.smolvlm.modeling_smolvlm")
    from transformers import SmolVLMVisionConfig

    torch.manual_seed(3)
    config = SmolVLMVisionConfig(hidden_size=32, image_size=32, patch_size=8, num_channels=3)
    return smolvlm.SmolVLMVisionEmbeddings(config).to(torch.bfloat16).eval()


class _Vision(torch.nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.embeddings = embeddings

    def forward(self, pixels):
        b, _, h, w = pixels.shape
        p = self.embeddings.patch_size
        return self.embeddings(pixels, torch.ones(b, h // p, w // p, dtype=torch.bool, device=pixels.device))


@pinned
def test_vision_position_ids_are_static_and_identical():
    module = _Vision(_vision_embeddings()).eval()
    x = torch.randn(2, 3, 32, 32).to(torch.bfloat16)
    with torch.no_grad():
        eager = module(x)
    original = torch.export.export(module, (x,), strict=True)
    assert fx_export.graph_shape_census(original.graph)["dynamic_nodes"]   # the boolean-mask index_put: tensor<?xi64>
    with fx_export._static_vision_position_ids(module) as paths:
        assert paths == ["embeddings"]
        program = torch.export.export(module, (x,), strict=True)
        with torch.no_grad():
            assert torch.equal(module(x).view(torch.int16), eager.view(torch.int16))
            with pytest.raises(fx_export.FXVerificationError, match="not all true"):
                mask = torch.ones(2, 4, 4, dtype=torch.bool)
                mask[1, 3, 3] = False
                module.embeddings(x, mask)
    assert "forward" not in module.embeddings.__dict__
    assert fx_export.graph_shape_census(program.graph)["dynamic_nodes"] == []
    targets = {str(n.target) for n in program.graph.nodes if n.op == "call_function"}
    assert not targets & {"aten.index_put_.default", "aten.index_put.default", "aten.bucketize.Tensor", "aten.nonzero.default"}
    with torch.no_grad():
        assert torch.equal(program.module()(x).view(torch.int16), eager.view(torch.int16))
    y = torch.randn(1, 3, 16, 32).to(torch.bfloat16)                       # not full size: the original path
    with fx_export._static_vision_position_ids(module), torch.no_grad():
        assert torch.equal(module(y).view(torch.int16), _Vision(module.embeddings)(y).view(torch.int16))


def test_vision_position_ids_check_refuses_ids_that_are_not_an_arange():
    smolvlm = pytest.importorskip("transformers.models.smolvlm.modeling_smolvlm")

    class SmolVLMVisionEmbeddings(smolvlm.SmolVLMVisionEmbeddings):         # same class name, other ids
        def forward(self, pixel_values, patch_attention_mask):
            ids = torch.arange(self.num_patches, device=pixel_values.device).flip(0)[None]
            return self.patch_embedding(pixel_values).flatten(2).transpose(1, 2) + self.position_embedding(ids)

    module = _Vision(_vision_embeddings())
    module.embeddings.__class__ = SmolVLMVisionEmbeddings
    with pytest.raises(fx_export.FXVerificationError, match="not arange"):
        with fx_export._static_vision_position_ids(module):
            pass
    assert "forward" not in module.embeddings.__dict__


# -- patch-embedding Conv2d as patch rows + qlinear -----------------------------------------------------------------------

def _layer(kernel, stride, padding=(0, 0), dilation=(1, 1), bias=torch.bfloat16):
    return SimpleNamespace(kernel_size=kernel, stride=stride, padding=padding, dilation=dilation,
                           bias=None if bias is None else torch.zeros(4, dtype=bias), weight=torch.zeros(1, dtype=torch.bfloat16))


def test_conv_patch_rows_predicate():
    assert fx_export.conv_patch_rows(_layer((4, 4), (4, 4)), 16, 24)
    assert fx_export.conv_patch_rows(_layer((2, 4), (2, 4), bias=None), 6, 8)
    assert not fx_export.conv_patch_rows(_layer((4, 4), (2, 2)), 16, 16)          # overlapping
    assert not fx_export.conv_patch_rows(_layer((4, 4), (4, 4), padding=(1, 1)), 16, 16)
    assert not fx_export.conv_patch_rows(_layer((4, 4), (4, 4), dilation=(2, 2)), 16, 16)
    assert not fx_export.conv_patch_rows(_layer((4, 4), (4, 4)), 18, 16)          # not a multiple of the kernel
    assert not fx_export.conv_patch_rows(_layer((4, 4), (4, 4), bias=torch.float32), 16, 16)


@pytest.mark.parametrize("rule", [("static", "tensor", "ref", -6), ("dynamic", "token", "ref", None),
                                  ("dynamic", "tensor", "nc", None), ("dynamic", "block", "ref", None)])
@pytest.mark.parametrize("bias", [True, False])
def test_patch_rows_qlinear_is_bit_identical_to_qconv2d(rule, bias):
    """Same rows (unfold vs reshape / permute, bitwise), so the same quantize_act codes and exponents (static, per
    token or per block of rows map one to one), and the same contraction and bias add."""
    g = torch.Generator().manual_seed(5)
    B, C, H, W, k, out = 2, 3, 16, 24, 4, 8
    x = (torch.randn(B, C, H, W, generator=g) * 3).to(torch.bfloat16)
    oh, ow = H // k, W // k
    unfold = F.unfold(x, kernel_size=(k, k), stride=(k, k)).permute(0, 2, 1).reshape(B * oh * ow, -1)
    rows = x.reshape(B, C, oh, k, ow, k).permute(0, 2, 4, 1, 3, 5).reshape(B * oh * ow, -1)
    assert torch.equal(unfold.view(torch.int16), rows.view(torch.int16))
    nrule = fx_export.NativeRule(*rule, rows=32)
    w_codes, w_exp = atlas_fx_ops.quantize_act(torch.randn(out, C * k * k, generator=g).to(torch.bfloat16),
                                               "dynamic", "token", "ref", None, "uint8")
    b = torch.randn(out, generator=g).to(torch.bfloat16) if bias else None
    c1, e1 = fx_export._qact(unfold, nrule, "uint8")
    c2, e2 = fx_export._qact(rows, nrule, "uint8")
    assert torch.equal(c1, c2) and torch.equal(e1, e2)
    ref = torch.ops.atlas.qconv2d(c1, e1, w_codes, w_exp, b, "bfloat16", "atlas_e4m3", B, oh, ow)
    lin = torch.ops.atlas.qlinear(c2, e2, w_codes, w_exp, b, "bfloat16", "atlas_e4m3")
    lin = lin.reshape(B, oh * ow, out).permute(0, 2, 1).reshape(B, out, oh, ow)
    assert torch.equal(ref.view(torch.int16), lin.view(torch.int16))


class _Patch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = torch.nn.Conv2d(3, 8, 4, stride=4)
        self.fc = torch.nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(self.patch(x).flatten(2).transpose(1, 2))


@pinned
@pytest.mark.parametrize("spec", ["granularity:w=channel,a=token,rule=nc", "none"])
def test_patch_embedding_capture_has_no_convolution(tmp_path, spec):
    torch.manual_seed(0)
    module = _Patch().to(torch.bfloat16).eval()
    algo = algos.make(spec)
    report = algos.quantize(module, "ref_conv", options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(module, "ref_conv", report, algo, dict(FAST), [], [], {}, {})
    x = torch.randn(2, 3, 16, 12, dtype=torch.bfloat16)
    cases = [(x,), (x * 4,)]
    with torch.no_grad():
        eager = [module(*c) for c in cases]
    result = fx_export.capture_fx(module, cases, tmp_path / "fx", policy=policy, fresh_process=True, native_fp8="uint8")
    assert result["status"] == "passed"
    assert result["native_fp8_detail"]["conv2d_lowering"] == {"patch": fx_export.PATCH_ROWS}
    assert result["fresh_process_reload"]["atlas_qconv2d_nodes"] == 0
    assert result["fresh_process_reload"]["atlas_qlinear_nodes"] == 2
    inspection = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    row = next(r for r in inspection["weight_site_bindings"] if r["kind"] == "conv2d")
    assert row["op"] == "qlinear" and row["lowering"] == fx_export.PATCH_ROWS and row["im2col_nodes"] == []
    assert inspection["native_fp8_conv2d_patch_row_sites"] == ["patch"]
    assert (inspection["native_fp8_linear_sites"], inspection["native_fp8_conv2d_sites"]) == (2, 0)
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    targets = [str(n.target) for n in program.graph.nodes if n.op == "call_function"]
    assert not [t for t in targets if "conv" in t or "im2col" in t]
    assert targets.count("atlas.qlinear.default") == 2
    with torch.no_grad():
        for case, out in zip(cases, eager):
            assert torch.equal(program.module()(*case).view(torch.int16), out.view(torch.int16))


# -- dead pure nodes ------------------------------------------------------------------------------------------------------

class _Dead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(4))

    def embed(self, x, y):
        """Returns a value and masks the caller does not read (as embed_prefix returns its pad / block masks)."""
        att = torch.tensor([True, False, True], device=x.device)[None, :].expand(x.shape[0], -1)
        pad = torch.cat([y > 0, torch.ones_like(y, dtype=torch.bool)], dim=1)
        return x * self.w * torch.tensor(3.0), pad, att

    def forward(self, x, y):
        out, _, _ = self.embed(x, y)
        positions = y - torch.min(y, dim=1, keepdim=True).values                 # computed, never read
        return out + 0 * positions.shape[1]


@pinned
def test_drop_dead_pure_nodes_removes_unread_work_and_constants(tmp_path):
    module = _Dead().eval()
    args = (torch.randn(2, 4), torch.arange(6).reshape(2, 3))
    program = torch.export.export(module, args, strict=True)
    targets = lambda p: [str(n.target) for n in p.graph.nodes if n.op == "call_function"]   # noqa: E731
    assert targets(program).count("aten.lift_fresh_copy.default") == 2
    expected = program.module()(*args)
    removed = fx_export.drop_dead_pure_nodes(program)
    assert removed
    assert targets(program).count("aten.lift_fresh_copy.default") == 1                # the read constant stays
    assert not [n for n in program.graph.nodes if n.op == "call_function" and not n.users and not n.is_impure()]
    kinds = [s.kind.name for s in program.graph_signature.input_specs]
    assert kinds.count("CONSTANT_TENSOR") == 1 and kinds.count("PARAMETER") == 1
    assert len(program.constants) == 1
    torch.export.save(program, tmp_path / "p.pt2")
    loaded = torch.export.load(tmp_path / "p.pt2")
    assert torch.equal(loaded.module()(*args), expected) and torch.equal(program.module()(*args), expected)
