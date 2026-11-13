"""SmolVLA's masked attention softmax in the additive form during FX capture (fx_export._smolvla_additive_attention_mask).

The reference text attention masks with a boolean select, softmax(where(keep, s, finfo(float32).min)); the capture gives
softmax(s + where(keep, 0, finfo(bfloat16).min)) instead, the form of TinyLlama's graph. Shown here:
  - the two forms give the same softmax words (float32 and after the cast to BF16) for any keep mask, including the
    fully masked rows SmolVLA gives a padded query (make_att_2d_masks ANDs the pad outer product): uniform in both;
  - the same through ref_attention (text variant, bool vs additive mask);
  - a capture of SmolVLMWithExpertModel (lerobot's class, tiny Llama stacks) behind the flow-loss mask construction has
    every masked softmax as add(float32 scores, BF16 where(keep, zeros, BF16 min)), one shared where, no select, and
    byte-identical outputs to the eager select form; eager evaluation is unchanged afterwards."""

import json

import algos
import exporter.atlas_fx_ops as atlas_fx_ops  # noqa: F401  (registers the atlas ops)
import exporter.fx_export as fx_export
import exporter.quant_export as quant_export
import pytest
import torch
from test_fx_export import PINNED_TORCH
from test_ladder import FAST

F32_MIN = torch.finfo(torch.float32).min
BF16_MIN = torch.finfo(torch.bfloat16).min


def _select(scores, keep):
    return torch.softmax(torch.where(keep, scores, F32_MIN), dim=-1)


def _additive(scores, keep):
    return torch.softmax(scores + fx_export.additive_attention_mask(keep), dim=-1)


def _make_att_2d_masks(pad, ar):
    """lerobot.policies.common.vla_utils.make_att_2d_masks."""
    cumsum = torch.cumsum(ar, dim=1)
    return (cumsum[:, None, :] <= cumsum[:, :, None]) & (pad[:, None, :] * pad[:, :, None])


def _keep_masks(g, rows, cols):
    random = torch.rand(2, rows, cols, generator=g) > 0.4
    random[0, 3] = False                                  # fully masked rows
    random[1, -1] = False
    random[0, 5] = True                                   # fully kept row
    random[1, 7] = False
    random[1, 7, 11] = True                               # one kept entry
    pad = torch.ones(2, cols, dtype=torch.bool)
    pad[1, 15:20] = False                                 # padded tokens: their query rows are fully masked
    ar = torch.zeros(2, cols)
    ar[:, cols // 2] = 1
    ar[:, cols // 2 + 3:] = 1
    flow = _make_att_2d_masks(pad, ar)[:, :rows]
    return torch.cat([random, flow])


def test_additive_mask_is_the_bf16_eager_mask():
    keep = torch.tensor([[True, False], [False, True]])
    mask = fx_export.additive_attention_mask(keep)
    assert mask.dtype == torch.bfloat16 and fx_export.BF16_MIN == BF16_MIN == -3.3895313892515355e+38
    assert mask.view(torch.int16).tolist() == [[0, -129], [-129, 0]]        # 0x0000 and 0xFF7F (finite)


@pytest.mark.parametrize("scale", [1.0, 30.0, 1e4])
def test_softmax_words_identical_including_fully_masked_rows(scale):
    g = torch.Generator().manual_seed(int(scale))
    rows, cols = 24, 40
    keep = _keep_masks(g, rows, cols)[:, None].expand(-1, 3, -1, -1)       # [B, H, Sq, Skv]
    scores = torch.randn(keep.shape, generator=g) * scale
    scores[0, 0, 0, :5] = -0.0                                             # -0.0 + 0 is +0.0; exp is the same
    old, new = _select(scores, keep), _additive(scores, keep)
    assert torch.equal(old.view(torch.int32), new.view(torch.int32))
    assert torch.equal(old.to(torch.bfloat16).view(torch.int16), new.to(torch.bfloat16).view(torch.int16))
    empty = ~keep.any(-1)
    assert empty.any() and empty[3].any()                                  # the flow mask's padded queries
    uniform = torch.full((cols,), 1.0 / cols)
    for p in (old, new):                                                   # fully masked row: uniform in both forms
        assert torch.isfinite(p).all()
        assert torch.allclose(p[empty], uniform.expand(int(empty.sum()), -1), rtol=0, atol=1e-7)
        assert (p[~empty][~keep[~empty]] == 0).all()                       # masked entries of a nonempty row: exactly 0


def test_fully_masked_row_with_the_constants_in_bf16():
    """The float32 minimum is not a BF16 value: cast to BF16 it rounds to -inf, and a fully masked row of -inf is NaN.
    The BF16 minimum is finite: a fully masked row stays uniform."""
    assert torch.tensor(F32_MIN).to(torch.bfloat16).item() == float("-inf")
    assert torch.isnan(torch.softmax(torch.full((8,), F32_MIN).to(torch.bfloat16).float(), -1)).all()
    p = torch.softmax(torch.full((8,), BF16_MIN, dtype=torch.bfloat16).float(), -1)
    assert torch.equal(p, torch.full((8,), 0.125))


def test_ref_attention_text_variant_bool_and_additive_give_the_same_words():
    import reference.ref_attention as ref_attention

    g = torch.Generator().manual_seed(3)
    keep = _keep_masks(g, 24, 40)
    b, h, d = keep.shape[0], 4, 16
    q = torch.randn(b, h, 24, d, generator=g).to(torch.bfloat16)
    k = torch.randn(b, h, 40, d, generator=g).to(torch.bfloat16)
    v = torch.randn(b, h, 40, d, generator=g).to(torch.bfloat16)
    old = ref_attention.ref_attention(q, k, v, d ** -0.5, keep)
    new = ref_attention.ref_attention(q, k, v, d ** -0.5, fx_export.additive_attention_mask(keep))
    assert torch.equal(old.view(torch.int16), new.view(torch.int16))


# -- capture ----------------------------------------------------------------------------------------------------------------

class _FlowMasks(torch.nn.Module):
    """VLAFlowMatching.forward's attention path: prefix / suffix embeddings and a prefix pad mask in, the
    make_att_2d_masks mask (prefix block, then a causal action block) and cumsum position ids built in the graph,
    SmolVLMWithExpertModel.forward(use_cache=False) on both streams, suffix output out."""

    def __init__(self, model):
        super().__init__()
        self.vlm_with_expert = model

    def forward(self, prefix, suffix, prefix_pad):
        n, m = prefix.shape[1], suffix.shape[1]
        pad = torch.cat([prefix_pad, torch.ones(1, m, dtype=torch.bool)], dim=1)
        ar = torch.cat([torch.zeros(1, n - 1), torch.ones(1, 2), torch.zeros(1, m - 1)], dim=1)
        (_, out), _ = self.vlm_with_expert.forward(attention_mask=_make_att_2d_masks(pad, ar),
                                                   position_ids=torch.cumsum(pad, dim=1) - 1, past_key_values=None,
                                                   inputs_embeds=[prefix, suffix], use_cache=False)
        return out


def _captured(tmp_path):
    from test_algo_smoothquant import _smolvla

    model = _smolvla().to(torch.bfloat16)
    algo = algos.make("granularity:w=channel,a=token")
    report = algos.quantize(model, "ref_conv", [], options=dict(FAST), algo=algo)
    policy = quant_export.build_policy(model, "ref_conv", report, algo, dict(FAST), [],
                                       report["algo"].get("bf16_opt_out", []), {}, {})
    assert isinstance(getattr(model, "_atlas_ref_attn", None), dict)
    boundary = _FlowMasks(model).eval()
    g = torch.Generator().manual_seed(7)
    cases = []
    for i in range(2):
        pad = torch.ones(1, 6, dtype=torch.bool)
        if i:
            pad[0, 3:5] = False                       # padded prefix tokens: fully masked query rows
        cases.append((torch.randn(1, 6, 64, generator=g).to(torch.bfloat16),
                      torch.randn(1, 3, 48, generator=g).to(torch.bfloat16), pad))
    with torch.no_grad():
        eager = [boundary(*case) for case in cases]
    result = fx_export.capture_fx(boundary, cases, tmp_path / "fx", policy=policy, fresh_process=False,
                                  native_fp8="uint8")
    return model, boundary, cases, eager, result


@pytest.mark.skipif(torch.__version__ < PINNED_TORCH, reason=f"torch.export captures need the pinned torch {PINNED_TORCH}")
def test_smolvla_capture_has_only_additive_masked_softmax(tmp_path):
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    model, boundary, cases, eager, result = _captured(tmp_path)
    # outputs byte-identical to the eager select form (capture_fx compares fixtures made before the capture contexts)
    assert result["status"] == "passed"
    assert result["additive_attention_mask"]["modules"] == ["vlm_with_expert"]
    program = torch.export.load(tmp_path / "fx" / "model.pt2")
    forms = fx_export.softmax_mask_forms(program.graph)
    # 4 VLM layers and 4 expert layers (2 self-attention calls on the joint stream, 2 cross: prefix + expert)
    assert [row["form"] for row in forms] == ["additive_bf16_min"] * 6
    assert {(row["scores_dtype"], row["mask_dtype"]) for row in forms} == {("torch.float32", "torch.bfloat16")}
    assert result["additive_attention_mask"]["softmax_input_forms"] == {"additive_bf16_min": 6}
    wheres = [n for n in program.graph.nodes if n.op == "call_function" and "where" in str(n.target)]
    assert len(wheres) == 1 and wheres[0].args[2] == BF16_MIN                     # one shared mask, as TinyLlama's
    assert {row["mask_source"] for row in forms} == {wheres[0].name}
    assert not any(isinstance(a, float) and a in (F32_MIN, float("-inf"))
                   for n in program.graph.nodes if n.op == "call_function" for a in n.args)
    inspection = json.loads((tmp_path / "fx" / "verification.json").read_text())
    assert inspection["status"] == "passed"
    # eager evaluation is restored: the instance forward is gone and the eager graph is the select form again
    assert "forward" not in model.__dict__
    with torch.no_grad():
        for case, out in zip(cases, eager):
            assert torch.equal(boundary(*case).view(torch.int16), out.view(torch.int16))
    eager_program = torch.export.export(boundary, cases[0])
    assert {row["form"] for row in fx_export.softmax_mask_forms(eager_program.graph)} == {"select"}


def test_softmax_mask_forms_on_tinyllama_layer_boundary_form():
    """The layer boundary form (a BF16 mask input added to float32 scores) is also recognized."""

    class Layer(torch.nn.Module):
        def forward(self, scores, mask):
            return torch.softmax(scores + mask, dim=-1)

    keep = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    program = torch.export.export(Layer(), (torch.randn(1, 2, 4, 4), fx_export.additive_attention_mask(keep)))
    assert [row["form"] for row in fx_export.softmax_mask_forms(program.graph)] == ["additive_bf16_min"]
