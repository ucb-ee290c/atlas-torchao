"""Bit-exactness of src/accum_torch.py against the reference author's functional models.

Oracles are built from the reference sources at test time:
  - SA C model: the C shim embedded in func_models_sa/systolic_c/systolic_array_rtl_linear.py
    (_SHIM_C, :34-72), compiled with that file's gcc command (:119-132) against the headers next
    to it, plus a few element-wise entry points (PE, product LUT, output stage) appended here.
  - IPT C model: the shim embedded in funct_models_ipt/ipt_c/ipt_rtl_linear_c.py (_SHIM_C,
    :194-325), which also carries the float -> E4M3 encoder shim_float_to_e4m3.
  - IPT numba model: funct_models_ipt/ipt_numba (_numba_kernels.compute_lanes_batch and
    IPTLinearRTLFunction), imported from the reference tree.

The reference tree is PI0_QUANT_DIR if set, else third_party/pi0-quant. Tests skip with a
message when the tree, a C compiler or numba is missing.
"""

import ast
import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

import accum_torch as at  # noqa: E402


def _ref_root():
    cands = [os.environ.get("PI0_QUANT_DIR"), str(REPO / "third_party" / "pi0-quant")]
    for c in cands:
        if c and (Path(c) / "func_models_sa" / "systolic_c" / "systolic_array_model.h").exists():
            return Path(c)
    return None


REF = _ref_root()
needs_ref = pytest.mark.skipif(REF is None, reason="reference tree not found: set PI0_QUANT_DIR "
                               "or initialise third_party/pi0-quant (pi0-quant 8bc0f6a)")
needs_cc = pytest.mark.skipif(shutil.which("gcc") is None,
                              reason="gcc not found: the reference C models cannot be built")

u8p = ctypes.POINTER(ctypes.c_uint8)
u16p = ctypes.POINTER(ctypes.c_uint16)

# Element-wise entry points appended to the SA shim (test-only).
_SA_EXTRA = textwrap.dedent("""
    void t_pe_batch(const uint16_t *p, const uint16_t *a, int n, uint16_t *out)
    { for (int i = 0; i < n; i++) out[i] = sa_e4m3_prod_add_bf16(p[i], a[i]); }
    void t_prod_batch(const uint8_t *a, const uint8_t *b, int n, uint16_t *out)
    { for (int i = 0; i < n; i++) out[i] = e4m3_mul_to_prod(a[i], b[i]); }
    void t_outconv_batch(const uint16_t *b, int n, int sel, int s, uint16_t *out)
    { for (int i = 0; i < n; i++) out[i] = (uint16_t)output_conv_stage(b[i], (OutputFmtSel)sel, s); }
""")


def _extract_shim(py_file: Path) -> str:
    for node in ast.parse(py_file.read_text()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "_SHIM_C":
            return textwrap.dedent(ast.literal_eval(node.value.args[0]))
    raise RuntimeError(f"_SHIM_C not found in {py_file}")


def _build(src: str, header_dir: Path, out_dir: Path, name: str) -> ctypes.CDLL:
    c = out_dir / f"{name}.c"
    so = out_dir / f"lib{name}.so"
    c.write_text(src)
    cmd = ["gcc", "-O3", "-march=native", "-Wall", "-Wno-unused-function", "-shared", "-fPIC",
           f"-I{header_dir}", "-o", str(so), str(c), "-lm"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        pytest.fail(f"building the reference oracle failed:\n{r.stderr}")
    return ctypes.CDLL(str(so))


@pytest.fixture(scope="module")
def sa_lib(tmp_path_factory):
    if REF is None or shutil.which("gcc") is None:
        pytest.skip("reference tree or gcc missing")
    d = REF / "func_models_sa" / "systolic_c"
    lib = _build(_extract_shim(d / "systolic_array_rtl_linear.py") + _SA_EXTRA, d,
                 tmp_path_factory.mktemp("sa"), "sa_ref")
    lib.sa_shim_init()
    return lib


@pytest.fixture(scope="module")
def ipt_lib(tmp_path_factory):
    if REF is None or shutil.which("gcc") is None:
        pytest.skip("reference tree or gcc missing")
    d = REF / "funct_models_ipt" / "ipt_c"
    lib = _build(_extract_shim(d / "ipt_rtl_linear_c.py"), d, tmp_path_factory.mktemp("ipt"), "ipt_ref")
    lib.shim_init()
    return lib


@pytest.fixture(scope="module")
def numba_ref():
    if REF is None:
        pytest.skip("reference tree missing")
    pytest.importorskip("numba", reason="numba not installed: the ipt_numba oracle cannot run")
    if str(REF) not in sys.path:
        sys.path.append(str(REF))
    import importlib
    kern = importlib.import_module("funct_models_ipt.ipt_numba._numba_kernels")
    lin = importlib.import_module("funct_models_ipt.ipt_numba.ipt_rtl_linear")
    return kern, lin


def _np(t):
    return np.ascontiguousarray(t.cpu().numpy())


def c_sa_linear(lib, x, w, b=None, rows=32, cols=32, scale_exp=0, sel=0):
    """sa_shim_linear_call on bytes (systolic_array_rtl_linear.py:50-70)."""
    M, K = x.shape
    N = w.shape[0]
    xa, wa = _np(x), _np(w)
    ba = _np(b) if b is not None else None
    out = np.empty(M * N, np.uint16)
    lib.sa_shim_linear_call(rows, cols, xa.ctypes.data_as(u8p), wa.ctypes.data_as(u8p),
                            ba.ctypes.data_as(u8p) if ba is not None else None,
                            M, K, N, scale_exp, sel, out.ctypes.data_as(u16p))
    return torch.from_numpy(out.astype(np.int32)).reshape(M, N)


def c_ipt_linear(lib, x, w, b=None, lanes=32, vec_len=32, scale_exp=0, sel=0):
    """shim_ipt_linear_call on bytes (ipt_rtl_linear_c.py:299-323)."""
    M, K = x.shape
    N = w.shape[0]
    xa, wa = _np(x), _np(w)
    ba = _np(b) if b is not None else None
    out = np.empty(M * N, np.uint16)
    lib.shim_ipt_linear_call(lanes, vec_len, 1, xa.ctypes.data_as(u8p), wa.ctypes.data_as(u8p),
                             ba.ctypes.data_as(u8p) if ba is not None else None,
                             M, K, N, scale_exp, sel, out.ctypes.data_as(u16p))
    return torch.from_numpy(out.astype(np.int32)).reshape(M, N)


def numba_ipt_linear(kern, x, w, b=None, lanes=32, vec_len=32, scale_exp=0, sel=0):
    """The IPTLinearRTLFunction tiling loop (ipt_rtl_linear.py:232-297) on raw bytes, so that
    operand bytes the float entry point cannot carry (-0, subnormals, 0x7F) are reachable."""
    x, w = _np(x).astype(np.uint8), _np(w).astype(np.uint8)
    M, K = x.shape
    N = w.shape[0]
    nk = (K + vec_len - 1) // vec_len
    xp = np.zeros((M, nk * vec_len), np.uint8)
    xp[:, :K] = x
    wp = np.zeros((N, nk * vec_len), np.uint8)
    wp[:, :K] = w
    y = np.zeros((M, N), np.int32)
    sexp = np.full(lanes, scale_exp, np.int32)
    zb = np.zeros(lanes, np.uint8)
    for ob in range(0, N, lanes):
        lc = min(lanes, N - ob)
        bias = zb.copy()
        if b is not None:
            bias[:lc] = _np(b).astype(np.uint8)[ob:ob + lc]
        psum = np.zeros((M, lanes), np.int32)
        for t in range(nk):
            wb = np.zeros((lanes, vec_len), np.uint8)
            wb[:lc] = wp[ob:ob + lc, t * vec_len:(t + 1) * vec_len]
            if t == 0 and b is not None:
                sel_a, cb = 1, bias
            elif t == 0:
                sel_a, cb = 0, zb
            else:
                sel_a, cb = 2, zb
            psum = kern.compute_lanes_batch(np.ascontiguousarray(xp[:, t * vec_len:(t + 1) * vec_len]),
                                            wb, wb, cb, psum, sexp, False, np.int32(sel_a),
                                            np.int32(sel), kern.MUL_LUT)
        y[:, ob:ob + lc] = psum[:, :lc]
    return torch.from_numpy(y)


# ------------------------------------------------------------------------------------------------
# Operand generators
# ------------------------------------------------------------------------------------------------

def rand_bytes(shape, g, mode="full"):
    """E4M3 operand bytes.
    full    every byte value, including -0 (0x80), subnormals, exponent 15 (0x78-0x7F, 0xF8-0xFF)
    normal  finite normal values and zeros only, as the vpack encoder produces them
    ties    +-2^e over a few binades, with repeats: sums land on BF16 halfway points often
    big     exponent fields 13-15 only (the saturation band of the encoder, 0x7E/0x7F)
    """
    if mode == "full":
        return torch.randint(0, 256, shape, generator=g, dtype=torch.int32).to(torch.uint8)
    if mode == "normal":
        e = torch.randint(1, 16, shape, generator=g)
        m = torch.randint(0, 8, shape, generator=g)
        s = torch.randint(0, 2, shape, generator=g)
        v = (s << 7) | (e << 3) | m
        v = torch.where((v & 0x7F) == 0x7F, v - 1, v)          # no 0x7F/0xFF (480)
        v = torch.where(torch.rand(shape, generator=g) < 0.05, torch.zeros_like(v), v)
        return v.to(torch.uint8)
    if mode == "ties":
        e = torch.randint(5, 11, shape, generator=g)
        s = torch.randint(0, 2, shape, generator=g)
        m = torch.where(torch.rand(shape, generator=g) < 0.2, torch.randint(0, 8, shape, generator=g), 0)
        return ((s << 7) | (e << 3) | m).to(torch.uint8)
    if mode == "big":
        e = torch.randint(13, 16, shape, generator=g)
        m = torch.randint(0, 8, shape, generator=g)
        s = torch.randint(0, 2, shape, generator=g)
        return ((s << 7) | (e << 3) | m).to(torch.uint8)
    raise ValueError(mode)


def bytes_to_float(b):
    """Decode E4M3 bytes as the reference input values, exponent 15 finite (fp_formats.h:229-241)."""
    b = b.to(torch.int32)
    s, e, m = (b >> 7) & 1, (b >> 3) & 0xF, b & 7
    v = torch.where(e == 0, m.double() / 8 * 2.0**-6, (1 + m.double() / 8) * torch.exp2((e - 7).double()))
    return torch.where(s == 1, -v, v).float()


K_LIST = [1, 31, 32, 33, 64, 96, 2048, 5632]


def _shape_for(K):
    return (3, 21) if K >= 2048 else (7, 37)


# ------------------------------------------------------------------------------------------------
# Encoder, product LUT, output stage
# ------------------------------------------------------------------------------------------------

def _all_bf16_as_f32():
    return (torch.arange(65536, dtype=torch.int32) << 16).view(torch.float32)


def _encoder_inputs():
    g = torch.Generator().manual_seed(0)
    rnd = torch.randint(-2**31, 2**31 - 1, (1 << 20,), generator=g, dtype=torch.int64).to(torch.int32)
    edge = torch.tensor([0.0, -0.0, 2.0**-6, -2.0**-6, 2.0**-6 * (1 - 2**-24), 448, 464, 464.0001,
                         479.99, 480, 495.99, 496, -496, 512, float("inf"), -float("inf"),
                         float("nan"), 1.0625, 1.1875, 3e38, -1e-45])
    return torch.cat([_all_bf16_as_f32(), rnd.view(torch.float32), edge])


@needs_ref
@needs_cc
def test_encoder_matches_c_shim(ipt_lib):
    x = _encoder_inputs()
    ref = np.empty(x.numel(), np.uint8)
    xa = _np(x)
    ipt_lib.shim_float_to_e4m3(xa.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                               ref.ctypes.data_as(u8p), x.numel())
    assert torch.equal(at.float_to_e4m3_bytes(x), torch.from_numpy(ref))


@needs_ref
def test_encoder_matches_numba_python_scalar(numba_ref):
    _, lin = numba_ref
    x = torch.cat([_all_bf16_as_f32(), _encoder_inputs()[-21:],
                   torch.randn(20000, generator=torch.Generator().manual_seed(1)) * 100])
    ref = torch.tensor([lin._float_to_e4m3_byte_scalar(v) for v in x.tolist()], dtype=torch.uint8)
    assert torch.equal(at.float_to_e4m3_bytes(x), ref)


@needs_ref
@needs_cc
def test_encoder_idempotent_on_every_byte(ipt_lib):
    """Every byte except -0 and subnormals (flushed, ipt_rtl_linear_c.py:255-258) re-encodes to
    itself, exponent 15 included; the float entry points rely on this (quant_linear.py:158-160)."""
    b = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    keep = ((b.to(torch.int32) >> 3) & 0xF) != 0
    assert torch.equal(at.float_to_e4m3_bytes(bytes_to_float(b))[keep], b[keep])


@needs_ref
@needs_cc
def test_product_lut_exhaustive(sa_lib, numba_ref):
    kern, _ = numba_ref
    i = torch.arange(65536, dtype=torch.int32)
    a, b = _np((i >> 8).to(torch.uint8)), _np((i & 0xFF).to(torch.uint8))
    out = np.empty(65536, np.uint16)
    sa_lib.t_prod_batch(a.ctypes.data_as(u8p), b.ctypes.data_as(u8p), 65536, out.ctypes.data_as(u16p))
    lut = at.e4m3_mul_to_prod_lut()
    assert torch.equal(lut, torch.from_numpy(out.astype(np.int32)))
    assert torch.equal(lut, torch.from_numpy(kern.MUL_LUT.astype(np.int32)))


@needs_ref
@needs_cc
@pytest.mark.parametrize("sel,scale", [(0, 0), (1, 0), (1, -9), (1, 7), (1, -130), (1, 130)])
def test_output_stage_exhaustive(sa_lib, sel, scale):
    bits = torch.arange(65536, dtype=torch.int32)
    out = np.empty(65536, np.uint16)
    b16 = np.ascontiguousarray(bits.numpy().astype(np.uint16))
    sa_lib.t_outconv_batch(b16.ctypes.data_as(u16p), 65536, sel, scale, out.ctypes.data_as(u16p))
    assert torch.equal(at.output_conv_stage(bits, sel, scale).to(torch.int32),
                       torch.from_numpy(out.astype(np.int32)))


# ------------------------------------------------------------------------------------------------
# SA
# ------------------------------------------------------------------------------------------------

def _pe_ref(lib, p, a):
    pa, aa = p.numpy().astype(np.uint16), a.numpy().astype(np.uint16)
    out = np.empty(p.numel(), np.uint16)
    lib.t_pe_batch(pa.ctypes.data_as(u16p), aa.ctypes.data_as(u16p), p.numel(), out.ctypes.data_as(u16p))
    return torch.from_numpy(out.astype(np.int32))


@needs_ref
@needs_cc
def test_sa_pe_every_addend_and_every_product(sa_lib):
    """E4M3ProdAddBF16 on all 65536 addend words (subnormals, +-0, Inf, NaN, near-max values
    that overflow-clamp) x 48 random products, and all 8192 product words x 48 random addends."""
    g = torch.Generator().manual_seed(5)
    a_all = torch.arange(65536, dtype=torch.int32)
    p_all = torch.arange(8192, dtype=torch.int32)
    p1 = torch.randint(0, 8192, (48,), generator=g, dtype=torch.int32)
    a1 = torch.randint(0, 65536, (48,), generator=g, dtype=torch.int32)
    P = torch.cat([p1.repeat_interleave(65536), p_all.repeat(48)])
    A = torch.cat([a_all.repeat(48), a1.repeat_interleave(8192)])
    assert torch.equal(at.sa_prod_add_bf16(P, A), _pe_ref(sa_lib, P, A))


@needs_ref
@needs_cc
@pytest.mark.parametrize("K", K_LIST)
@pytest.mark.parametrize("mode", ["full", "normal", "ties", "big"])
def test_sa_bitexact_vs_c(sa_lib, K, mode):
    M, N = _shape_for(K)
    g = torch.Generator().manual_seed(K * 7 + len(mode))
    x, w = rand_bytes((M, K), g, mode), rand_bytes((N, K), g, mode)
    if K >= 64:
        x[:, 32:64] = 0                                     # an all-zero K tile
        w[: N // 3, :] = 0                                  # all-zero output columns
    assert torch.equal(at.sa_linear_bits(x, w), c_sa_linear(sa_lib, x, w))


@needs_ref
@needs_cc
@pytest.mark.parametrize("K", [1, 33, 96])
@pytest.mark.parametrize("sel,scale", [(0, 0), (1, 0), (1, -6), (1, 5)])
def test_sa_bias_and_e4m3_output_vs_c(sa_lib, K, sel, scale):
    g = torch.Generator().manual_seed(K + sel * 100 + scale)
    M, N = 6, 45
    x, w, b = rand_bytes((M, K), g, "full"), rand_bytes((N, K), g, "full"), rand_bytes((N,), g, "full")
    b[:4] = torch.tensor([0x00, 0x80, 0x03, 0x83], dtype=torch.uint8)   # +-0, +-subnormal bias
    x[:2] = 0
    ref = c_sa_linear(sa_lib, x, w, b, scale_exp=scale, sel=sel)
    assert torch.equal(at.sa_linear_bits(x, w, b, scale_exp=scale, out_fmt_sel=sel), ref)
    assert ref[0, 1].item() == 0x8000 or sel == 1        # -0 bias with no products survives


@needs_ref
@needs_cc
@pytest.mark.parametrize("rows", [8, 16, 64])
def test_sa_other_tile_heights(sa_lib, rows):
    g = torch.Generator().manual_seed(rows)
    x, w = rand_bytes((4, 150), g, "normal"), rand_bytes((19, 150), g, "normal")
    assert torch.equal(at.sa_linear_bits(x, w, rows=rows), c_sa_linear(sa_lib, x, w, rows=rows, cols=8))


@needs_ref
@needs_cc
def test_sa_float_entry_matches_reference_class():
    """The reference SARTLLinearFunction on float input (systolic_array_rtl_linear.py:257-291)."""
    pytest.importorskip("numba")
    if str(REF) not in sys.path:
        sys.path.append(str(REF))
    import importlib
    m = importlib.import_module("func_models_sa.systolic_c.systolic_array_rtl_linear")
    g = torch.Generator().manual_seed(11)
    x = bytes_to_float(rand_bytes((2, 3, 100), g, "normal"))
    w = bytes_to_float(rand_bytes((24, 100), g, "normal")) * 2.0**-4   # re-encoded by the encoder
    ref = m.SARTLLinearFunction(rows=32, cols=32)(x, w, None)
    assert torch.equal(at.sa_linear(x, w), ref)


# ------------------------------------------------------------------------------------------------
# IPT
# ------------------------------------------------------------------------------------------------

@needs_ref
@pytest.mark.parametrize("K", K_LIST)
@pytest.mark.parametrize("mode", ["full", "normal", "ties", "big"])
def test_ipt_numba_bitexact(numba_ref, K, mode):
    kern, _ = numba_ref
    M, N = _shape_for(K)
    g = torch.Generator().manual_seed(K * 13 + len(mode))
    x, w = rand_bytes((M, K), g, mode), rand_bytes((N, K), g, mode)
    if K >= 64:
        x[:, 32:64] = 0
        w[: N // 3, :] = 0
    assert torch.equal(at.ipt_linear_bits(x, w), numba_ipt_linear(kern, x, w))


@needs_ref
@needs_cc
@pytest.mark.parametrize("K", K_LIST)
@pytest.mark.parametrize("mode", ["full", "normal", "ties", "big"])
def test_ipt_c_bitexact(ipt_lib, K, mode):
    M, N = _shape_for(K)
    g = torch.Generator().manual_seed(K * 17 + len(mode))
    x, w = rand_bytes((M, K), g, mode), rand_bytes((N, K), g, mode)
    if K >= 64:
        x[:, 32:64] = 0
    assert torch.equal(at.ipt_linear_bits(x, w, variant="c"), c_ipt_linear(ipt_lib, x, w))


@needs_ref
@needs_cc
@pytest.mark.parametrize("K", [1, 33, 96])
@pytest.mark.parametrize("sel,scale", [(0, 0), (1, 0), (1, -6), (1, 5)])
def test_ipt_bias_and_e4m3_output(ipt_lib, numba_ref, K, sel, scale):
    kern, _ = numba_ref
    g = torch.Generator().manual_seed(K + sel * 100 + scale + 1)
    M, N = 6, 45
    x, w, b = rand_bytes((M, K), g, "full"), rand_bytes((N, K), g, "full"), rand_bytes((N,), g, "full")
    b[:4] = torch.tensor([0x00, 0x80, 0x03, 0x83], dtype=torch.uint8)
    x[:2] = 0
    kw = dict(scale_exp=scale, out_fmt_sel=sel)
    assert torch.equal(at.ipt_linear_bits(x, w, b, **kw),
                       numba_ipt_linear(kern, x, w, b, scale_exp=scale, sel=sel))
    assert torch.equal(at.ipt_linear_bits(x, w, b, variant="c", **kw),
                       c_ipt_linear(ipt_lib, x, w, b, scale_exp=scale, sel=sel))


@needs_ref
@needs_cc
def test_ipt_lanes_16_32_and_numba(ipt_lib, numba_ref):
    """Plan section 3 / test_ipt_lanes: lane count cannot change results (lanes are independent,
    inner_product_trees_model.h:336-349). 16 == 32 lanes in the C model and in the numba model,
    and the torch ports agree with each. ipt_c and ipt_numba themselves are compared in
    test_ipt_c_vs_numba_disagreement."""
    kern, _ = numba_ref
    g = torch.Generator().manual_seed(21)
    x, w = rand_bytes((5, 300), g, "normal"), rand_bytes((70, 300), g, "normal")
    c16, c32 = c_ipt_linear(ipt_lib, x, w, lanes=16), c_ipt_linear(ipt_lib, x, w, lanes=32)
    n16, n32 = numba_ipt_linear(kern, x, w, lanes=16), numba_ipt_linear(kern, x, w, lanes=32)
    assert torch.equal(c16, c32) and torch.equal(n16, n32)
    assert torch.equal(at.ipt_linear_bits(x, w, variant="c"), c32)
    assert torch.equal(at.ipt_linear_bits(x, w), n32)


@needs_ref
@needs_cc
def test_ipt_c_vs_numba_disagreement(ipt_lib, numba_ref):
    """ipt_c rounds int -> float32 -> BF16 with a 30-bit accumulator (converters.h:244-254,
    params_and_requests.h:88-93); ipt_numba rounds once with 32 bits (_numba_kernels.py:156-167).
    The torch variants reproduce each model on the case where they differ."""
    kern, _ = numba_ref
    x = torch.zeros(1, 32, dtype=torch.uint8)
    w = torch.zeros(1, 32, dtype=torch.uint8)
    x[0, :3] = at.float_to_e4m3_bytes(torch.tensor([256.0, 16.0, 2.0**-4]))
    w[0, :3] = x[0, :3]
    n = numba_ipt_linear(kern, x, w)
    c = c_ipt_linear(ipt_lib, x, w)
    assert n.item() != c.item()
    assert torch.equal(at.ipt_linear_bits(x, w), n)
    assert torch.equal(at.ipt_linear_bits(x, w, variant="c"), c)


@needs_ref
def test_ipt_float_entry_matches_reference_class(numba_ref):
    """IPTLinearRTLFunction(vec_len=32, num_lanes=32) on float input, the oracle that
    src/accum_backends.py names for ('ipt', 'ref_cpu')."""
    _, lin = numba_ref
    g = torch.Generator().manual_seed(12)
    x = bytes_to_float(rand_bytes((2, 3, 100), g, "normal"))
    w = bytes_to_float(rand_bytes((24, 100), g, "normal")) * 2.0**-4
    ref = lin.IPTLinearRTLFunction(vec_len=32, num_lanes=32)(x, w, None)
    assert torch.equal(at.ipt_linear(x, w), ref)


# ------------------------------------------------------------------------------------------------
# Contract properties (no oracle needed)
# ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [at.sa_linear, at.ipt_linear])
def test_batched_form_equals_per_index(fn):
    g = torch.Generator().manual_seed(3)
    x = bytes_to_float(rand_bytes((2, 3, 6, 40), g, "normal"))
    w = bytes_to_float(rand_bytes((2, 3, 5, 40), g, "normal"))
    y = fn(x, w)
    assert y.shape == (2, 3, 6, 5) and y.dtype == torch.float32
    for i in range(2):
        for j in range(3):
            assert torch.equal(y[i, j], fn(x[i, j], w[i, j]))


@pytest.mark.parametrize("bits_fn", [at.sa_linear_bits, at.ipt_linear_bits])
def test_chunking_does_not_change_results(bits_fn):
    g = torch.Generator().manual_seed(4)
    x, w = rand_bytes((4, 70), g, "full"), rand_bytes((33, 70), g, "full")
    full = bits_fn(x, w)
    assert torch.equal(bits_fn(x, w, chunk_elems=1), full)
    assert torch.equal(bits_fn(x, w, chunk_elems=1000), full)


def test_float_and_byte_inputs_agree_and_shapes():
    g = torch.Generator().manual_seed(6)
    xb, wb = rand_bytes((2, 5, 40), g, "normal"), rand_bytes((9, 40), g, "normal")
    for fn in (at.sa_linear, at.ipt_linear):
        y = fn(bytes_to_float(xb), bytes_to_float(wb))
        assert y.shape == (2, 5, 9)
        assert torch.equal(y, fn(xb, wb))


def test_zero_k_gives_zeros():
    x, w = torch.zeros(3, 0), torch.zeros(4, 0)
    assert torch.equal(at.sa_linear(x, w), torch.zeros(3, 4))
    assert torch.equal(at.ipt_linear(x, w), torch.zeros(3, 4))


def test_repeat_runs_identical():
    g = torch.Generator().manual_seed(8)
    x, w = rand_bytes((4, 200), g, "full"), rand_bytes((17, 200), g, "full")
    assert torch.equal(at.sa_linear_bits(x, w), at.sa_linear_bits(x, w))
    assert torch.equal(at.ipt_linear_bits(x, w), at.ipt_linear_bits(x, w))


@pytest.mark.skipif(os.environ.get("ATLAS_TEST_COMPILE", "0") == "0",
                    reason="torch.compile path is slow to build; set ATLAS_TEST_COMPILE=1")
def test_sa_compiled_equals_eager():
    g = torch.Generator().manual_seed(9)
    x, w = rand_bytes((4, 70), g, "full"), rand_bytes((13, 70), g, "full")
    assert torch.equal(at.sa_linear_bits(x, w, compile=True), at.sa_linear_bits(x, w, compile=False))


# (M, N, K, batch, bias, out_fmt_sel, scale_exp): layer-like shape mix, incl. M == 1, N == 1,
# M == N, K not a multiple of rows, batched w and the E4M3 output feedback.
_SA_COMPILE_SHAPES = [
    (16, 64, 96, 0, False, 0, 0), (7, 33, 70, 0, True, 0, 0), (1, 48, 64, 0, False, 0, 0),
    (5, 1, 40, 0, True, 0, 0), (12, 12, 12, 0, False, 0, 0), (3, 80, 33, 0, False, 1, 0),
    (9, 20, 128, 0, True, 1, -6), (1, 1, 5, 0, False, 0, 0), (32, 8, 65, 0, False, 0, 0),
    (4, 24, 31, 2, False, 0, 0), (6, 6, 48, 3, False, 0, 0), (2, 100, 17, 0, True, 0, 0),
    (20, 3, 1, 0, False, 0, 0), (8, 40, 200, 0, False, 1, 5),
]


@pytest.mark.skipif(os.environ.get("ATLAS_TEST_COMPILE", "0") == "0",
                    reason="torch.compile path is slow to build; set ATLAS_TEST_COMPILE=1")
def test_sa_compiled_many_shapes_no_recompile_fallback(request):
    """The compiled SA step must serve every layer shape from a bounded set of graphs: with
    recompile_limit set to 2 and fail_on_recompile_limit_hit, any per-shape recompile beyond
    that raises instead of silently falling back to eager. Bits equal eager and the C model."""
    try:
        lib = request.getfixturevalue("sa_lib")
    except pytest.skip.Exception:
        lib = None
    from torch._dynamo.utils import counters
    torch._dynamo.reset()
    at._sa_step_compiled = None
    counters.clear()
    g = torch.Generator().manual_seed(12)
    with torch._dynamo.config.patch(recompile_limit=2, fail_on_recompile_limit_hit=True):
        for M, N, K, nb, bias, sel, scale in _SA_COMPILE_SHAPES:
            lead = (nb,) if nb else ()
            x, w = rand_bytes((*lead, M, K), g, "full"), rand_bytes((*lead, N, K), g, "full")
            b = rand_bytes((N,), g, "full") if bias else None
            kw = dict(scale_exp=scale, out_fmt_sel=sel)
            got = at.sa_linear_bits(x, w, b, compile=True, **kw)
            assert torch.equal(got, at.sa_linear_bits(x, w, b, compile=False, **kw)), (M, N, K)
            if lib is not None and not nb:
                assert torch.equal(got, c_sa_linear(lib, x, w, b, scale_exp=scale, sel=sel)), (M, N, K)
    assert counters["stats"]["unique_graphs"] <= 2, dict(counters["stats"])
    assert not counters["unimplemented"] and not counters["graph_break"]
    torch._dynamo.reset()
    at._sa_step_compiled = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
@pytest.mark.parametrize("fn", [at.sa_linear_bits, at.ipt_linear_bits])
def test_cuda_equals_cpu(fn):
    g = torch.Generator().manual_seed(10)
    x, w = rand_bytes((8, 2048), g, "full"), rand_bytes((40, 2048), g, "full")
    cpu = fn(x, w)
    dev = fn(x.cuda(), w.cuda())
    assert dev.device.type == "cuda"
    assert torch.equal(dev.cpu(), cpu)
    assert torch.equal(fn(x.cuda(), w.cuda()).cpu(), cpu)    # run-to-run determinism



# ------------------------------------------------------------------------------------------------
# Determinism settings
# ------------------------------------------------------------------------------------------------

def test_runs_under_deterministic_algorithms():
    g = torch.Generator().manual_seed(14)
    x, w = rand_bytes((3, 70), g, "full"), rand_bytes((9, 70), g, "full")
    a, b = at.sa_linear_bits(x, w), at.ipt_linear_bits(x, w)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        assert torch.equal(at.sa_linear_bits(x, w), a)
        assert torch.equal(at.ipt_linear_bits(x, w), b)
        assert at.numerics_report()["deterministic_algorithms"]
    finally:
        torch.use_deterministic_algorithms(prev)


def test_enforce_exact_settings():
    rep = at.enforce_exact_settings()
    assert rep["cuda_matmul_allow_tf32"] is False and rep["cudnn_allow_tf32"] is False
