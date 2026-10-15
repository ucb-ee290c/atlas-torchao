"""smoothquant: the folds are identities, po2 folds are exact, tied groups are tied, and anything not
implemented is refused (src/algos/smoothquant.py, docs/algos/smoothquant.md)."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import algos
from algos import smoothquant as sq
from refutil import need_refs
from test_ladder import FAST


def _llama(kv=2, layers=2, dtype=torch.float32, seed=0):
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=kv, max_position_embeddings=64,
                      attn_implementation="eager", use_cache=False)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).eval()
    with torch.no_grad():                         # non-trivial norms and a few outlier channels
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(torch.rand_like(p) * 2 + 0.1)
                p[::7] *= 16
    return m.to(dtype)


def _ids(seed=1, n=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, 128, (1, 16), generator=g) for _ in range(n)]


def _run(m, b):
    return m(b)


def _fold(model, batches, run=_run, **params):
    algo = algos.make("smoothquant:" + ",".join(f"{k}={v}" for k, v in params.items()) if params else "smoothquant")
    algo.calibrate(model, batches, run)
    rec = algo.prepare(model, "ref_linear")
    return algo, rec


def _logits(m, ids):
    with torch.no_grad():
        return m(ids).logits.float()


@pytest.mark.parametrize("po2", [True, False])
@pytest.mark.parametrize("alpha", [0.5, 0.85, "off"])
def test_fold_is_identity_fp32(po2, alpha):
    m0 = _llama()
    m1 = copy.deepcopy(m0)
    _, rec = _fold(m1, _ids(), alpha=alpha, po2=po2)
    assert rec["groups"] == {"qkv": 2, "fc1": 2, "down": 2, "o": 2}
    x = _ids(seed=5, n=1)[0]
    torch.testing.assert_close(_logits(m1, x), _logits(m0, x), rtol=1e-4, atol=1e-4)
    # the fold really changed the parameters
    assert not torch.equal(m0.model.layers[0].input_layernorm.weight, m1.model.layers[0].input_layernorm.weight)


def test_po2_fold_exact_in_bf16():
    """Power-of-two s and gain only shift exponents: every folded BF16 parameter is exact, and the
    BF16 forward is bit-identical (every product is scaled by an exact power of two)."""
    m0 = _llama(dtype=torch.bfloat16)
    m1 = copy.deepcopy(m0)
    _, rec = _fold(m1, _ids(), alpha=0.5, po2=True)
    assert sum(rec["inexact_params"].values()) == 0
    x = _ids(seed=5, n=1)[0]
    assert torch.equal(_logits(m1, x), _logits(m0, x))


def test_non_po2_fold_is_inexact_in_bf16():
    m = _llama(dtype=torch.bfloat16)
    _, rec = _fold(m, _ids(), alpha=0.5, po2=False)
    assert sum(rec["inexact_params"].values()) > 0          # recorded, not hidden


def _col_factor(old, new):
    return (new.double() / old.double())[0]                  # one factor per input column


@pytest.mark.parametrize("kv", [2, 1])
def test_o_fold_tied_over_kv_groups(kv):
    """GQA (2 kv heads for 4 query heads) and single-kv-head: the o_proj columns of the query heads
    sharing a kv head get the same factor, and v_proj rows the reciprocal."""
    m0 = _llama(kv=kv)
    m1 = copy.deepcopy(m0)
    _fold(m1, _ids(), alpha=0.5, sites="o", gain="false")
    a0, a1 = m0.model.layers[0].self_attn, m1.model.layers[0].self_attn
    hd = a0.head_dim
    f = _col_factor(a0.o_proj.weight, a1.o_proj.weight).view(4, hd)
    groups = 4 // kv
    for k in range(kv):
        for h in range(k * groups, (k + 1) * groups):
            assert torch.equal(f[h], f[k * groups])
    r = (a1.v_proj.weight.double() / a0.v_proj.weight.double())[:, 0].view(kv, hd)
    torch.testing.assert_close(r * f[::groups], torch.ones_like(r))
    x = _ids(seed=5, n=1)[0]
    torch.testing.assert_close(_logits(m1, x), _logits(m0, x), rtol=1e-4, atol=1e-4)


def _gemma(dtype):
    from transformers import GemmaConfig, GemmaForCausalLM

    cfg = GemmaConfig(vocab_size=128, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=1, head_dim=16, max_position_embeddings=64,
                      attn_implementation="eager", use_cache=False)
    torch.manual_seed(0)
    m = GemmaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.normal_(0.0, 0.5)
    return m.to(dtype)


def test_gemma_one_plus_norm_single_kv_head():
    m0 = _gemma(torch.float32)
    m1 = copy.deepcopy(m0)
    _, rec = _fold(m1, _ids(), alpha=0.5)
    assert rec["norm_fold_max_rel_err"] < 1e-6 and not rec["norm_weights_promoted_fp32"]
    x = _ids(seed=5, n=1)[0]
    torch.testing.assert_close(_logits(m1, x), _logits(m0, x), rtol=1e-4, atol=1e-4)
    a0, a1 = m0.model.layers[0].self_attn, m1.model.layers[0].self_attn
    f = _col_factor(a0.o_proj.weight, a1.o_proj.weight).view(4, 16)
    assert all(torch.equal(f[h], f[0]) for h in range(4))            # one kv head: all tied


def test_gemma_bf16_norm_promoted_or_refused():
    m = _gemma(torch.bfloat16)
    with pytest.raises(NotImplementedError, match=r"\(1\+w\) RMSNorm"):
        _fold(copy.deepcopy(m), _ids(), gemma_norm="refuse")
    _, rec = _fold(m, _ids())
    assert len(rec["norm_weights_promoted_fp32"]) == 4
    assert m.model.layers[0].input_layernorm.weight.dtype == torch.float32


# ---------------------------------------------------------------------------------------------
# SmolVLA: the real SmolVLMWithExpertModel forward on tiny Llama stacks
# ---------------------------------------------------------------------------------------------

def _smolvla(n_layers=4, every=2):
    pytest.importorskip("lerobot.policies.smolvla.smolvlm_with_expert")
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
    from transformers import LlamaConfig, LlamaModel

    vcfg = LlamaConfig(vocab_size=8, hidden_size=64, intermediate_size=96, num_hidden_layers=n_layers,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=16, attn_implementation="eager")
    ecfg = copy.deepcopy(vcfg)
    ecfg.hidden_size, ecfg.intermediate_size = 48, 64
    torch.manual_seed(0)
    m = SmolVLMWithExpertModel.__new__(SmolVLMWithExpertModel)
    nn.Module.__init__(m)
    vlm = nn.Module()
    vlm.model = nn.Module()
    vlm.model.text_model = LlamaModel(vcfg)
    vlm.config = SimpleNamespace(text_config=vcfg)
    m.vlm, m.lm_expert, m.config = vlm, LlamaModel(ecfg), SimpleNamespace(text_config=vcfg)
    kvd = 2 * 16
    for j, layer in enumerate(m.lm_expert.layers):      # smolvlm_with_expert.py:122-135
        if j % every:
            layer.self_attn.k_proj = nn.Linear(kvd, kvd, bias=False)
            layer.self_attn.v_proj = nn.Linear(kvd, kvd, bias=False)
    m.num_vlm_layers = m.num_expert_layers = n_layers
    m.self_attn_every_n_layers, m.attention_mode = every, "cross_attn"
    m.num_attention_heads, m.num_key_value_heads = 4, 2
    m.freeze_vision_encoder, m.train_expert_only = False, False
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith("norm.weight"):
                p.copy_(torch.rand_like(p) * 2 + 0.1)
    m.eval()                                       # its train() override returns None
    return m


def _smol_batch(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(1, 6, 64, generator=g), torch.randn(1, 3, 48, generator=g)


def _smol_run(m, b):
    pre, suf = b
    Lp, Ls = pre.shape[1], suf.shape[1]
    pos = torch.arange(Lp + Ls)[None]
    mask = torch.ones(1, Lp + Ls, Lp + Ls, dtype=torch.bool)
    (out_p, _), cache = m(attention_mask=mask[:, :Lp, :Lp], position_ids=pos[:, :Lp], past_key_values=None,
                          inputs_embeds=[pre, None], use_cache=True)
    (_, out_s), _ = m(attention_mask=mask[:, Lp:, :], position_ids=pos[:, Lp:], past_key_values=cache,
                      inputs_embeds=[None, suf], use_cache=True)
    return out_p, out_s


def test_smolvla_cross_attention_undo():
    m0 = _smolvla()
    m1 = copy.deepcopy(m0)
    _, rec = _fold(m1, [_smol_batch(1), _smol_batch(2)], run=_smol_run, alpha=0.5)
    assert rec["structures"] == ["smolvla:<root>"]
    # 4 VLM + 4 expert layers: qkv/fc1/down each 8; o: 2 joint (tied) + 2 cross x (VLM + expert)
    assert rec["groups"] == {"qkv": 8, "fc1": 8, "down": 8, "o": 6}
    e0, e1 = m0.lm_expert.layers[1].self_attn, m1.lm_expert.layers[1].self_attn
    v0, v1 = m0.vlm.model.text_model.layers[1].self_attn, m1.vlm.model.text_model.layers[1].self_attn
    # the expert's cross v_proj columns undo the VLM's v_proj row fold
    rows = (v1.v_proj.weight.double() / v0.v_proj.weight.double())[:, 0]          # 1 / f_vlm
    ratio = e1.v_proj.weight.double() / e0.v_proj.weight.double()                  # r_e[i] * c[j]
    t = ratio * rows[None, :]                       # c[j] = f_vlm[j] leaves only the expert's row factor
    torch.testing.assert_close(t, t[:, :1].expand_as(t))
    assert not torch.allclose(rows, torch.ones_like(rows))
    with torch.no_grad():
        b = _smol_batch(9)
        for y1, y0 in zip(_smol_run(m1, b), _smol_run(m0, b)):
            torch.testing.assert_close(y1, y0, rtol=1e-4, atol=1e-4)


def test_smolvla_joint_layer_tied():
    m0 = _smolvla()
    m1 = copy.deepcopy(m0)
    _fold(m1, [_smol_batch(1)], run=_smol_run, alpha=0.5, sites="o")
    fv = _col_factor(m0.vlm.model.text_model.layers[0].self_attn.o_proj.weight,
                     m1.vlm.model.text_model.layers[0].self_attn.o_proj.weight)
    fe = _col_factor(m0.lm_expert.layers[0].self_attn.o_proj.weight, m1.lm_expert.layers[0].self_attn.o_proj.weight)
    assert torch.equal(fv, fe)


# ---------------------------------------------------------------------------------------------
# Refusals and the profile path
# ---------------------------------------------------------------------------------------------

class FooEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)

    def forward(self, x):
        return self.fc(x)


def test_unsupported_block_refused_unless_partial():
    m = _llama()
    m.vision = FooEncoderLayer()
    with pytest.raises(NotImplementedError, match="FooEncoderLayer"):
        _fold(copy.deepcopy(m), _ids())
    _, rec = _fold(m, _ids(), partial="true")
    assert rec["not_folded_blocks"] == ["vision (FooEncoderLayer)"]


def test_unknown_norm_refused():
    m = _llama()
    for layer in m.model.layers:
        w = layer.input_layernorm.weight
        layer.input_layernorm = nn.LayerNorm(w.numel())
    with pytest.raises(NotImplementedError, match="LayerNorm"):
        _fold(m, _ids())


@pytest.mark.parametrize("mode", ["collab_v0_hw", "ref_linear", "chip"])
def test_profile_path_records_gain_and_static(mode):
    need_refs()
    m = _llama(dtype=torch.bfloat16)
    algo = algos.make("smoothquant:alpha=0.5,static=true")
    calib = lambda: (_ids(), _run, {"source": "test", "n_samples": 2, "seed": 1})   # noqa: E731
    rep = algos.quantize(m, mode, ["lm_head"], dict(FAST), algo=algo, calib=calib)
    rec = rep["algo"]
    assert rec["prepare"]["gain"]["exp_histogram"] and rec["prepare"]["static_act_scales"] > 0
    with torch.no_grad():
        y = m(_ids(seed=5, n=1)[0]).logits
    assert torch.isfinite(y).all()
    if mode == "collab_v0_hw":
        assert not rec["module_hooks_applied"]
    else:
        assert any("static_per_tensor(smoothquant)" in k for k in rec["granularity"]["activation"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("alpha", [0.5, "off"])
def test_static_on_cuda_matches_cpu(alpha):
    """static=true with the model on CUDA: the calibration maxima live on the GPU while the fold
    divisors are moved to the CPU. prepare() must not mix devices, and the static exponents must
    equal a CPU run's."""
    need_refs()
    exps = {}
    for dev in ("cpu", "cuda"):
        m = _llama(dtype=torch.bfloat16).to(dev)
        batches = [b.to(dev) for b in _ids()]
        algo, rec = _fold(m, batches, alpha=alpha, gain="true", static="true")
        assert rec["static_act_scales"] > 0
        exps[dev] = {k: v.cpu() for k, v in algo._static_exp.items()}
    assert exps["cuda"].keys() == exps["cpu"].keys()
    for k in exps["cpu"]:
        assert torch.equal(exps["cuda"][k], exps["cpu"][k]), k


def test_gain_brings_fixed_scale_inputs_into_range():
    """With the gain, every folded linear input's calibrated maximum sits in (224, 448]: the range
    the collaborator's fixed 2^0 activation scale covers without flushing small values early."""
    m = _llama()
    batches = _ids(n=3)
    algo, _ = _fold(m, batches, alpha="off")
    after = algo.calibrate(m, batches, _run) and algo._amax
    for layer in range(2):
        for k in ("self_attn.q_proj", "mlp.gate_proj", "mlp.down_proj", "self_attn.o_proj"):
            a = after[f"model.layers.{layer}.{k}"].max().item()
            assert 224 * 0.999 < a <= 448 * 1.001, (k, a)


def test_pi0_tiny_fold():
    need_refs()
    import pi0_tiny
    why = pi0_tiny.available()
    if why:
        pytest.skip(why)
    m0, cfg = pi0_tiny.build("float32")
    m1 = copy.deepcopy(m0)
    obs, nz = pi0_tiny.observation(cfg), pi0_tiny.noise(cfg)
    run = lambda m, b: m.sample_actions("cpu", obs, noise=nz, num_steps=2)     # noqa: E731
    with pytest.raises(NotImplementedError, match="Siglip"):
        _fold(copy.deepcopy(m0), [0], run=run)
    _, rec = _fold(m1, [0], run=run, partial="true")
    assert rec["structures"] == ["pi0:paligemma_with_expert"] and rec["not_folded_blocks"]
    with torch.no_grad():
        torch.testing.assert_close(run(m1, 0), run(m0, 0), rtol=1e-3, atol=1e-4)


# ---------------------------------------------------------------------------------------------
# partial=true on a model with no supported decoder layers (a ViT/DiT-like stack)
# ---------------------------------------------------------------------------------------------

class FooBlock(nn.Module):
    """A pre-LayerNorm transformer-ish block smoothquant does not fold (LayerNorm with bias)."""

    def __init__(self, d=64):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, 2 * d)
        self.fc2 = nn.Linear(2 * d, d)

    def forward(self, x):
        return x + self.fc2(torch.nn.functional.gelu(self.fc1(self.norm(x))))


class FooStack(nn.Module):
    def __init__(self, n=2, d=64):
        super().__init__()
        torch.manual_seed(0)
        self.blocks = nn.ModuleList([FooBlock(d) for _ in range(n)])
        self.head = nn.Linear(d, 8)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.head(x)


def _feats(seed=1, n=2):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(1, 16, 64, generator=g) * 3 for _ in range(n)]


def test_no_supported_layers_refused_without_partial():
    with pytest.raises(NotImplementedError, match="FooBlock"):
        _fold(FooStack(), _feats())
    # nothing unsupported either: the refusal names the missing decoder layers
    with pytest.raises(NotImplementedError, match="no supported decoder layers"):
        sq.build_groups(nn.Sequential(nn.Linear(2, 2)), sq.SITES)


def test_no_supported_layers_partial_is_a_recorded_no_op():
    m = FooStack()
    before = copy.deepcopy(m.state_dict())
    with pytest.warns(UserWarning, match="nothing folded"):
        algo, rec = _fold(m, _feats(), partial="true")
    assert rec["folded_groups"] == 0 and rec["groups"] == {}
    assert rec["not_folded_blocks"] == ["blocks.0 (FooBlock)", "blocks.1 (FooBlock)"]
    assert rec["warnings"] and "nothing folded" in rec["warnings"][0]
    assert rec["gain"]["exp_histogram"] == {} and rec["static_act_scales"] == 0
    after = m.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_no_supported_layers_partial_static_scales_every_linear():
    m = FooStack()
    with pytest.warns(UserWarning, match="nothing folded"):
        algo, rec = _fold(m, _feats(), partial="true", static="true")
    linears = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    assert rec["static_act_scales"] == len(linears) == 5
    assert set(algo._static_exp) == linears
    import ref_numerics as rn
    for n in linears:   # no fold: the exponent comes straight from the calibrated maximum
        assert torch.equal(algo._static_exp[n], rn.po2_scale_exp(algo._amax[n].max().reshape(())))
    s = algo.act_scale("blocks.0.fc1", torch.zeros(1), "linear")
    assert s is not None and s.dynamic is False


@pytest.mark.parametrize("mode", ["ref_linear", "chip"])
def test_no_supported_layers_partial_profile_path_runs(mode):
    """Through algos.quantize: prepare is a no-op, the quantization after it still runs, and the
    static scales reach the reference modules."""
    need_refs()
    m = FooStack().to(torch.bfloat16)
    feats = [f.to(torch.bfloat16) for f in _feats()]
    algo = algos.make("smoothquant:partial=true,static=true")
    calib = lambda: (feats, _run, {"source": "test", "n_samples": 2, "seed": 1})   # noqa: E731
    with pytest.warns(UserWarning, match="nothing folded"):
        rep = algos.quantize(m, mode, [], dict(FAST), algo=algo, calib=calib)
    prep = rep["algo"]["prepare"]
    assert prep["folded_groups"] == 0 and prep["static_act_scales"] == 5
    with torch.no_grad():
        y = m(feats[0])
    assert torch.isfinite(y).all()
    # the activation record fills in during the forward
    assert rep["algo"]["granularity"]["activation"] == {"linear:static_per_tensor(smoothquant)": 5}
