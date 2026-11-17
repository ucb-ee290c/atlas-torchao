"""Embedding / LM-head options (src/exporter/head_embed.py, docs/algos/head-embed.md): the codecs, the default
being byte-identical, export -> reload bit-identity per option, one stored tensor for a tied pair, and the FX
capture of the quantized storage."""
import json

import pytest
import torch
from safetensors.torch import load_file

import algos
import exporter.fx_export as fx_export
import exporter.head_embed as he
import exporter.load_quantized as load_quantized
import exporter.quant_export as quant_export
from test_export import _calib
from test_ladder import FAST, tiny_llama

SPEC = "gptq_awq:method=awq_gptq,actorder=true+granularity:w=panel,a=defer,attn=defer,rows=32,rule=nc+static_calib"
IDS = torch.randint(0, 128, (2, 16), generator=torch.Generator().manual_seed(1))
OPTION_SETS = [
    {"tie": "lm_head_from_embedding", "lm_head": "bf16", "embedding": "bf16"},
    {"tie": "embedding_from_lm_head", "lm_head": "bf16", "embedding": "bf16"},
    {"tie": "none", "lm_head": "fp8", "embedding": "bf16"},
    {"tie": "none", "lm_head": "nvfp4", "embedding": "nvfp4"},
    {"tie": "none", "lm_head": "fp8", "embedding": "nvfp4"},
    {"tie": "none", "lm_head": "bf16", "embedding": "fp8"},
    {"tie": "none", "lm_head": "fp8", "embedding": "fp8"},
    {"tie": "lm_head_from_embedding", "lm_head": "fp8", "embedding": "fp8"},
    {"tie": "embedding_from_lm_head", "lm_head": "nvfp4", "embedding": "nvfp4"},
]


def _export(tmp_path, opts, spec=SPEC, name="export"):
    m = tiny_llama(2)
    algo = algos.make(spec)
    batches, calib = _calib()
    rec = he.prepare(m, opts)
    skip = he.skip_for(opts)
    rep = algos.quantize(m, "ref_conv", skip, dict(FAST), algo=algo, calib=calib)
    rec = he.finish(m, opts, rec)
    d = tmp_path / name
    quant_export.export(m, d, mode="ref_conv", report=rep, algo=algo, algo_spec=spec, options=dict(FAST),
                        skip=skip, model_record={"kind": "causal_lm", "id": "tiny"}, calib_batches=batches,
                        calib_record={"source": "test"},
                        head_embed=None if he.is_default(opts) else he.export_record(opts, rec))
    return m, d


def _reload(d):
    m, _, _ = load_quantized.load_quantized(None, d, "cpu", build=lambda: tiny_llama(2), mode="ref_conv",
                                            options=dict(FAST))
    return m


def _content(path):
    """(header as a dict, data bytes) of a safetensors file."""
    raw = path.read_bytes()
    n = int.from_bytes(raw[:8], "little")
    return json.loads(raw[8:8 + n]), raw[8 + n:]


def _logits(m, ids=IDS):
    with torch.no_grad():
        return m(ids, use_cache=False).logits


# --- codecs ------------------------------------------------------------------------------------

def test_e2m1_rounding_matches_torchao():
    kernels = pytest.importorskip("torchao.prototype.mx_formats.kernels")
    x = torch.cat([torch.linspace(-6, 6, 20001),
                   torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, -0.25, -0.75, -5.0, -0.0, 0.0])])
    assert torch.equal(he.e2m1_rne(x), kernels.f32_to_f4_unpacked(x))


def test_nvfp4_decode_exact_in_bf16_and_po2_tensor_scale():
    g = torch.Generator().manual_seed(0)
    w = (torch.randn(48, 64, generator=g) * torch.logspace(-3, 0, 48).unsqueeze(1)).to(torch.bfloat16)
    packed, scales, ts = he.nvfp4_quantize(w)
    assert packed.shape == (48, 32) and scales.shape == (48, 4) and ts.dtype == torch.float32
    m, _ = torch.frexp(ts)
    assert float(m) == 0.5                                      # a power of two
    assert int(scales.view(torch.float8_e4m3fn).float().max().item()) <= 448
    v = he.nvfp4_decode(packed, scales, ts)
    assert torch.equal(v.to(torch.bfloat16).float(), v)
    assert torch.equal(he.unpack_codes(he.pack_codes(he.unpack_codes(packed))), he.unpack_codes(packed))
    rel = (v - w.float()).norm() / w.float().norm()
    assert 0 < rel < 0.15


def test_fp8_rows_exact_decode():
    w = (torch.randn(32, 64, generator=torch.Generator().manual_seed(2)) * 0.02).to(torch.bfloat16)
    codes, exp = he.fp8_rows(w)
    assert codes.dtype == torch.uint8 and exp.shape == (32, 1) and exp.dtype == torch.int32
    v = he.fp8_decode(codes, exp)
    assert torch.equal(v.to(torch.bfloat16).float(), v)
    assert (v - w.float()).norm() / w.float().norm() < 0.05
    assert float(v.abs().amax()) <= float(w.float().abs().amax()) * (1 + 2 ** -3)   # never clips (nc)


def test_options_check():
    assert he.skip_for(dict(he.DEFAULTS)) == ["lm_head"]
    assert he.skip_for({**he.DEFAULTS, "lm_head": "fp8"}) == []
    with pytest.raises(ValueError, match="same precision"):
        he.check({"tie": "lm_head_from_embedding", "lm_head": "fp8", "embedding": "bf16"})
    assert he.label({"tie": "none", "lm_head": "nvfp4", "embedding": "nvfp4"}) == "head_nvfp4_emb_nvfp4"


# --- export ------------------------------------------------------------------------------------

def test_default_export_byte_identical(tmp_path):
    m0 = tiny_llama(2)
    algo = algos.make(SPEC)
    batches, calib = _calib()
    rep = algos.quantize(m0, "ref_conv", ["lm_head"], dict(FAST), algo=algo, calib=calib)
    quant_export.export(m0, tmp_path / "plain", mode="ref_conv", report=rep, algo=algo, algo_spec=SPEC,
                        options=dict(FAST), skip=["lm_head"], model_record={"kind": "causal_lm", "id": "tiny"},
                        calib_batches=batches, calib_record={"source": "test"})
    _, d = _export(tmp_path, dict(he.DEFAULTS), name="default")
    assert (tmp_path / "plain" / "policy.json").read_bytes() == (d / "policy.json").read_bytes()
    # safetensors writes the header metadata from a hash map, so its key order (not its content) varies per save
    assert _content(tmp_path / "plain" / "model.safetensors") == _content(d / "model.safetensors")
    a = json.loads((tmp_path / "plain" / "manifest.json").read_text())
    b = json.loads((d / "manifest.json").read_text())
    for m in (a, b):             # run-specific: time, host, argv, timings, the header-order-dependent file digest
        for k in ("created_utc", "provenance"):
            m.pop(k)
        m["files"]["model.safetensors"].pop("sha256")
        for member in m["recipe"]["algorithm_record"].get("members", []):
            member.get("hw_aware", {}).pop("timing_s", None)
    assert a == b and "head_embed" not in b


@pytest.mark.parametrize("opts", OPTION_SETS, ids=he.label)
def test_option_round_trip(tmp_path, opts):
    m0, d = _export(tmp_path, opts)
    m1 = _reload(d)
    assert torch.equal(_logits(m0), _logits(m1))
    tensors = load_file(str(d / "model.safetensors"))
    policy = json.loads((d / "policy.json").read_text())
    assert policy["head_embed"]["options"] == opts
    emb, head = "model.embed_tokens", "lm_head"
    if opts["embedding"] == "bf16":
        assert f"{emb}.weight" in tensors or policy["aliases"].get(f"{emb}.weight")
    else:
        assert f"{emb}.weight" not in tensors
        assert policy["sites"][emb]["kind"] == "embedding" and policy["sites"][emb]["format"] == opts["embedding"]
    if opts["lm_head"] != "bf16":
        assert f"{head}.weight" not in tensors and policy["sites"][head]["status"] == "quantized"
    if opts["tie"] != "none":
        if opts["embedding"] == "bf16":           # one BF16 tensor, the other name an alias
            assert len([k for k in (f"{emb}.weight", f"{head}.weight") if k in tensors]) == 1
        else:                                     # one set of codes: the head's
            assert policy["sites"][emb]["tied_to"] == head
            assert not [k for k in tensors if k.startswith(emb + ".") and ".weight_decode_" not in k]
            assert f"{head}.weight_codes" in tensors
    # decoder layers: the same codes as the default export unless the embedding feeds calibration
    if opts["tie"] != "embedding_from_lm_head":
        _, d0 = _export(tmp_path, dict(he.DEFAULTS), name="default")
        t0 = load_file(str(d0 / "model.safetensors"))
        body = [k for k in t0 if k.startswith("model.layers.")]
        assert body and all(torch.equal(t0[k], tensors[k]) for k in body)


def test_tied_embedding_reads_head_codes(tmp_path):
    opts = {"tie": "lm_head_from_embedding", "lm_head": "fp8", "embedding": "fp8"}
    m0, _ = _export(tmp_path, opts)
    emb, head = m0.get_input_embeddings(), m0.get_output_embeddings()
    assert isinstance(emb, he.QuantEmbedding) and emb.source() is head
    ids = torch.arange(128)
    want = he.fp8_decode(head.weight_codes, head.weight_scale_exp).to(torch.bfloat16)
    assert torch.equal(emb(ids), want)


@pytest.mark.parametrize("opts", [OPTION_SETS[7], OPTION_SETS[8], OPTION_SETS[3]], ids=he.label)
def test_native_capture(tmp_path, opts):
    _, d = _export(tmp_path, opts)
    m = _reload(d)
    policy = json.loads((d / "policy.json").read_text())
    cases = [(torch.arange(8, dtype=torch.long).reshape(1, 8),), (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]
    report = fx_export.capture_fx(fx_export.CausalLogits(m).eval(), cases, tmp_path / "fx", policy=policy,
                                  strict=True, fresh_process=False, native_fp8="uint8")
    assert report["status"] == "passed"
    inspected = json.loads((tmp_path / "fx" / "graph_inspection.json").read_text())
    storage = {r["site"]: r for r in inspected["quantized_storage_sites"]}
    assert storage["model.embed_tokens"]["in_graph"]
    if opts["tie"] != "none":
        assert storage["model.embed_tokens"]["code_buffer"] == "model.lm_head.weight_codes"
    if opts["lm_head"] == "fp8":
        assert "model.lm_head" in report["native_fp8_detail"]["sites"]


CAPTURE_CASES = [(torch.arange(8, dtype=torch.long).reshape(1, 8),),
                 (torch.arange(8, 16, dtype=torch.long).reshape(1, 8),)]


def _capture(tmp_path, opts):
    """(native-FP8 capture report, saved program) of the option's export, reloaded."""
    _, d = _export(tmp_path, opts)
    m = _reload(d)
    policy = json.loads((d / "policy.json").read_text())
    report = fx_export.capture_fx(fx_export.CausalLogits(m).eval(), CAPTURE_CASES, tmp_path / "fx", policy=policy,
                                  strict=True, fresh_process=False, native_fp8="uint8")
    return report, torch.export.load(tmp_path / "fx" / "model.pt2")


def _rotary_inputs(program):
    """Names of the casts that deliver rotary cos / sin to the layers: the first aten.to after aten.cos / aten.sin."""
    found = []
    for op in (torch.ops.aten.cos.default, torch.ops.aten.sin.default):
        (src,) = [n for n in program.graph.nodes if n.target == op]
        frontier = list(src.users)
        while frontier:
            node = frontier.pop(0)
            if getattr(node.target, "overloadpacket", None) is torch.ops.aten.to:
                found.append(node.name)
                break
            frontier.extend(node.users)
    return found


def _names_outside(program, scopes):
    return [(n.name, str(n.target)) for n in program.graph.nodes
            if n.op == "call_function" and not fx_export._scope(n).startswith(scopes)]


@pytest.fixture(scope="module")
def default_capture(tmp_path_factory):
    return _capture(tmp_path_factory.mktemp("default"), dict(he.DEFAULTS))


# a tied BF16 pair is not captured: strict export cannot name the held alias parameter (KeyError in torch.export);
# ties are not an option for TinyLlama (docs/algos/head-embed.md), so this is recorded, not fixed
_TIED_BF16 = pytest.mark.xfail(raises=KeyError, strict=True, reason="tied BF16 alias is not captured by strict export")


@pytest.mark.parametrize("opts", [pytest.param(o, marks=_TIED_BF16) if o["tie"] != "none" and o["lm_head"] == "bf16"
                                  else o for o in OPTION_SETS], ids=he.label)
def test_capture_keeps_default_node_names(tmp_path, opts, default_capture):
    """The storage-decode nodes get their own names, so the rotary cos / sin inputs (and every node outside the
    embedding and the head) keep the default capture's names, which backend tools key boundaries on."""
    default_report, default_program = default_capture
    assert "renamed_for_storage_decode" not in default_report          # default: the graph is not touched
    report, program = _capture(tmp_path, opts)
    assert report["status"] == "passed"
    rotary = _rotary_inputs(program)
    assert len(rotary) == 2 and rotary == _rotary_inputs(default_program)
    outside = ("model.model.embed_tokens", "model.lm_head")
    assert _names_outside(program, outside) == _names_outside(default_program, outside)
    if opts["embedding"] != "bf16":
        decode = [n.name for n in program.graph.nodes
                  if n.op == "call_function" and fx_export._scope(n) == "model.model.embed_tokens"]
        assert decode and all(name.startswith("model_model_embed_tokens_") for name in decode)


def test_tied_fp8_embedding_with_per_tensor_head_exponent():
    """A tied FP8 head quantized without a recipe (reference rule: one per-tensor exponent) still serves lookups."""
    m = tiny_llama(1)
    opts = {"tie": "lm_head_from_embedding", "lm_head": "fp8", "embedding": "fp8"}
    rec = he.prepare(m, opts)
    he.quantize_head_reference(m)
    he.finish(m, opts, rec)
    head = m.get_output_embeddings()
    assert head.weight_scale_exp.numel() == 1
    want = he.fp8_decode(head.weight_codes, head.weight_scale_exp).to(torch.bfloat16)
    assert torch.equal(m.get_input_embeddings()(torch.arange(128)), want)


def test_table_decodes_equal_bit_decodes():
    """The modules decode by table gathers (no uint8 bit ops in a captured graph); same values as the bit-op forms."""
    import reference.ref_numerics as rn

    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(16, 16)
    exp = torch.arange(-8, 8, dtype=torch.int32).reshape(16, 1)
    emb = he.QuantEmbedding(16, 16, "fp8", torch.bfloat16, (codes, exp))
    assert set(emb.tables()) == {"weight_decode_atlas_lut"}
    assert torch.equal(emb._fp8_rows(codes, exp), rn.dequantize(codes, exp))
    pair = he.e2m1_pair_table()
    assert torch.equal(pair[codes.long()].reshape(16, 32), he.e2m1_table()[he.unpack_codes(codes).long()])
