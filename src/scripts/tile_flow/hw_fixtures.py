"""Regenerate an FX bundle's fixtures with Atlas hardware accumulation in every FP8 contraction.

A saved bundle (`<bundle>/fx/model.pt2`, `fixtures.safetensors`, `parity.json`) records its outputs from the
torch bodies of src/exporter/atlas_fx_ops.py: each atlas.qlinear / qconv2d / qk_matmul / pv_matmul decodes the
E4M3 operands, contracts in float32 and rounds to BF16 once. Atlas rounds inside the contraction. This tool
replays the saved graph with torch.fx.Interpreter on the bundle's own fixture inputs, computes every
contraction node with the hardware accumulation of one MXU unit, and keeps every other node (quantize_act,
norms, softmax, adds, RoPE, casts, ...) exactly as captured. Outputs:

    <out>/fixtures_hw_<unit>.safetensors   same keys, dtypes and shapes as fixtures.safetensors; inputs copied
                                           unchanged, outputs recomputed
    <out>/fixtures_hw.json                 unit semantics, replaced and remaining ops, differing words, sha256

Contraction semantics: those of the Atlas backend's independent reference (rotary-host/reference/atlas_ref,
contractions.py `contraction_nt` / `qlinear`, arithmetic.py `matmul_codes` / `readout_bf16`), bit for bit:

    acc  = matmul_codes(A codes, B codes)        raw BF16 accumulator over the E4M3 code values, K ascending in
                                                 32-wide instructions, each continuing from the previous BF16
                                                 partial, a short last instruction zero-padded:
                                                   mxu1  one exact anchor-tree sum per 32 lanes, rounded to BF16
                                                         (accum_torch.ipt_linear_bits, variant "model47")
                                                   mxu0  one E4M3ProdAddBF16 per MAC, rounded every step
                                                         (accum_torch.sa_linear_bits)
    out  = BF16_RNE(acc * BF16(2^(ex_row + ew_col)))   readout: the power-of-two exponents, after accumulation
    out  = trunc_BF16(float32(out) + float32(bias))    bias (BF16), only when the node has one (VADD truncates)

then widened to the node's logical dtype (BF16 qlinear; float32 for qk/pv and float32-out qlinear). The torch
bodies apply the exponents before the contraction and add the bias with RNE; neither difference survives here.

Every replay first runs the graph unchanged (the torch bodies) and requires byte-identical outputs to
fixtures.safetensors, so the harness itself is checked on every case. Graph nodes run with one torch thread
(the capture setting); the integer accumulation uses --threads (integer sums are order independent).

--crosscheck ATLAS_REF_ROOT (the directory holding atlas_ref/) re-computes contraction outputs with atlas_ref's
own NumPy engine from the operands captured at each call and requires equal words: a random row x column x
batch subset of every call, plus all columns of a row block for a stratified sample of calls (first and last
call of each op kind, the first float32-out qlinear and the first call with K % 32 != 0).

Usage (CPU only):
    CUDA_VISIBLE_DEVICES= python src/scripts/tile_flow/hw_fixtures.py --bundle /path/to/bundle \\
        --out /path/to/out [--units mxu1 mxu0] [--threads 16] [--crosscheck /path/to/reference]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parent
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.utils._pytree as pytree  # noqa: E402

import exporter.atlas_fx_ops  # noqa: E402,F401  (registers atlas.* before torch.export.load)
import mxu.accum_torch as accum  # noqa: E402

SCHEMA = "atlas_torchao.fixtures_hw.v1"
UNITS = ("mxu1", "mxu0")
K_TILE = 32
IPT_VARIANT = "model47"
QLINEAR = torch.ops.atlas.qlinear.default
QCONV2D = torch.ops.atlas.qconv2d.default
QK = torch.ops.atlas.qk_matmul.default
PV = torch.ops.atlas.pv_matmul.default
CONTRACTIONS = {QLINEAR: "atlas.qlinear.default", QCONV2D: "atlas.qconv2d.default",
                QK: "atlas.qk_matmul.default", PV: "atlas.pv_matmul.default"}

UNIT_SEMANTICS = {
    "mxu1": {
        "name": "MXU1, inner-product trees",
        "rounding": "per 32-wide K instruction: the 32 E4M3 code products and the incoming BF16 partial are "
                    "aligned to one anchor (max exponent + 7 headroom) as 32-bit integers, summed exactly, and "
                    "rounded once to BF16 (RNE); the BF16 result is the partial carried into the next "
                    "instruction",
        "implementation": "accum_torch.ipt_linear_bits(variant='model47', vec_len=32, b=None, scale_exp=0)",
        "model47_note": "a tile total with fewer than 8 significant bits keeps its low 7 bits as the mantissa "
                        "field without left normalization, as npu-model 47b8599 rtl_math.py ipt_row and "
                        "atlas_ref arithmetic.py _inner_chunk do; accum_torch's 'numba' variant (the reference "
                        "author's IPT model) normalizes it",
    },
    "mxu0": {
        "name": "MXU0, systolic array",
        "rounding": "after every MAC: each E4M3 code product is added to the BF16 partial by E4M3ProdAddBF16 "
                    "and rounded to BF16 (RNE), K in ascending order; the 32-row instruction boundaries carry "
                    "the BF16 partial unchanged",
        "implementation": "accum_torch.sa_linear_bits(rows=32, b=None, scale_exp=0)",
    },
}
K_ORDER = {"order": "ascending (k = 0, 1, ..., K-1)",
           "instruction_width": K_TILE,
           "short_final_instruction": "zero-padded to 32 lanes (zero codes give zero products, which change "
                                      "nothing)",
           "initial_partial": "+0 (no accumulator bias)"}
EPILOGUE = {
    "accumulate_on": "raw E4M3 code values (no exponent applied); codes with exponent field 0 and nonzero "
                     "mantissa, and 0x7F/0xFF, are refused",
    "readout": "BF16_RNE(acc_bf16 * BF16(2^(x_exp[row] + w_exp[col]))) (atlas_ref readout_bf16)",
    "bias": "trunc_to_BF16(float32(readout) + float32(bias_bf16)) (atlas_ref _add_bf16); no node in these "
            "bundles has a bias",
    "output": "BF16 word widened to the node's logical dtype: BF16 for qlinear, float32 for qk_matmul / "
              "pv_matmul and for qlinear with out_dtype float32",
    "pv_pop_shift": "non-zero pop_shift is refused (atlas_ref forbids it; none in these bundles)",
}


# ------------------------------------------------------------------------------------------------
# Hardware contraction (torch port of atlas_ref contraction_nt / qlinear)
# ------------------------------------------------------------------------------------------------

def _codes(t: torch.Tensor) -> torch.Tensor:
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    if t.dtype != torch.uint8:
        raise ValueError(f"code operand must be uint8 or float8_e4m3fn bytes, got {t.dtype}")
    e, m = (t >> 3) & 15, t & 7
    if bool((((e == 0) & (m != 0)) | ((e == 15) & (m == 7))).any()):
        raise ValueError("operand holds a forbidden E4M3 subnormal or reserved code")
    return t


def _row_exp(exp: torch.Tensor, shape) -> torch.Tensor:
    """atlas_ref _row_exponents: scalar or constant over each contraction row, broadcast to shape[:-1]+(1,)."""
    if exp.dtype.is_floating_point or exp.dtype.is_complex or exp.dtype == torch.bool:
        raise ValueError(f"exponents must be integers, got {exp.dtype}")
    return torch.broadcast_to(exp.to(torch.int64), tuple(shape[:-1]) + (1,))


def bf16_rne_bits(f: torch.Tensor) -> torch.Tensor:
    """float32 -> BF16 word (int32), RNE, NaN -> 0x7FC0 (atlas_ref bf16_bits)."""
    f = f.to(torch.float32)
    raw = f.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    rounded = ((raw + 0x7FFF + ((raw >> 16) & 1)) >> 16) & 0xFFFF
    return torch.where(torch.isnan(f), 0x7FC0, rounded).to(torch.int32)


def bf16_value(bits: torch.Tensor) -> torch.Tensor:
    return ((bits.to(torch.int32) & 0xFFFF) << 16).view(torch.float32)


def readout(acc_bits: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
    """atlas_ref readout_bf16: VMUL of the accumulator by BF16(2^exponent), RNE."""
    # 2^e built exactly in float64 (|e| <= 1000 is normal there), then IEEE RNE to float32: inf above the
    # float32 range, subnormal or 0 below it, as NumPy's float32 ldexp gives.
    e = exponent.to(torch.int64).clamp(-1000, 1000)
    scale = ((e + 1023) << 52).view(torch.float64).to(torch.float32)
    return bf16_rne_bits(bf16_value(acc_bits) * bf16_value(bf16_rne_bits(scale)))


def vadd_trunc(a_bits: torch.Tensor, b_bits: torch.Tensor) -> torch.Tensor:
    """atlas_ref _add_bf16: float32 sum, truncated to BF16; NaN -> 0x7FC0."""
    v = bf16_value(a_bits) + bf16_value(b_bits)
    bits = (v.view(torch.int32) >> 16) & 0xFFFF
    return torch.where(torch.isnan(v), 0x7FC0, bits).to(torch.int32)


def accumulate(a: torch.Tensor, b: torch.Tensor, unit: str, compile: bool = True,
               ipt_variant: str = IPT_VARIANT) -> torch.Tensor:
    """a [..., M, K], b [..., N, K] uint8 codes -> raw BF16 accumulators [..., M, N] (int32 words)."""
    if a.shape[-1] != b.shape[-1]:
        raise ValueError(f"contraction K mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    M, N, K = a.shape[-2], b.shape[-2], a.shape[-1]
    a3 = a.expand(*batch, M, K).reshape(-1, M, K)
    b3 = b.expand(*batch, N, K).reshape(-1, N, K)
    if a3.shape[0] == 1:
        a3, b3 = a3[0], b3[0]
    if unit == "mxu1":
        bits = accum.ipt_linear_bits(a3, b3, vec_len=K_TILE, variant=ipt_variant, compile=compile)
    elif unit == "mxu0":
        bits = accum.sa_linear_bits(a3, b3, rows=K_TILE, compile=compile)
    else:
        raise ValueError(f"unit {unit!r}")
    return bits.reshape(*batch, M, N).to(torch.int32)


def contraction_nt(a_codes, a_exp, b_codes, b_exp, unit, bias=None, compile=True, ipt_variant=IPT_VARIANT):
    """atlas_ref contractions.contraction_nt: [..., M, K] x [..., N, K] -> BF16 words [..., M, N]."""
    a, b = _codes(a_codes), _codes(b_codes)
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("contraction needs [..., M, K] and [..., N, K]")
    exponent = _row_exp(a_exp, a.shape) + _row_exp(b_exp, b.shape).transpose(-1, -2)
    out = readout(accumulate(a, b, unit, compile, ipt_variant), exponent)
    if bias is not None:
        if bias.dtype != torch.bfloat16:
            raise ValueError(f"bias must be BF16, got {bias.dtype}")
        out = vadd_trunc(out, bias.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF)
    return out


def words_to(bits: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    signed = ((bits.to(torch.int32) & 0xFFFF) ^ 0x8000) - 0x8000
    return signed.to(torch.int16).view(torch.bfloat16).to(dtype)


def hw_node(target, args, kwargs, unit, compile=True, ipt_variant=IPT_VARIANT):
    """(value, operands, bits) for one contraction node; operands are what atlas_ref receives."""
    schema = target._schema
    full = [args[i] if i < len(args) else kwargs.get(a.name, a.default_value)
            for i, a in enumerate(schema.arguments)]
    if target in (QLINEAR, QCONV2D):
        x, xe, w, we, bias, out_dtype, fmt = full[:7]
        if fmt != "atlas_e4m3" or out_dtype not in ("bfloat16", "float32"):
            raise ValueError(f"unsupported {CONTRACTIONS[target]} format {fmt!r} / out_dtype {out_dtype!r}")
        if w.dim() != 2:
            raise ValueError("weight codes must be [N, K]")
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        xe2 = _row_exp(xe, x.shape).reshape(-1, 1)
        bits = contraction_nt(x2, xe2, w, we, unit, bias, compile, ipt_variant)
        value = words_to(bits, getattr(torch, out_dtype))
        operands = dict(kind="qlinear", a=x2, ax=xe2, b=w, bx=we, bias=bias)
        if target == QLINEAR:
            value = value.reshape(*lead, w.shape[0])
        else:
            batch, oh, ow = full[7:10]
            value = value.reshape(batch, oh * ow, w.shape[0]).permute(0, 2, 1).reshape(batch, w.shape[0], oh, ow)
        return value, operands, bits
    a, ax, b, bx = full[:4]
    if target == PV and len(full) > 4 and int(full[4]) != 0:
        raise ValueError("pv_matmul with non-zero pop_shift is outside the atlas_ref contract")
    bits = contraction_nt(a, ax, b, bx, unit, None, compile, ipt_variant)
    return words_to(bits, torch.float32), dict(kind="nt", a=a, ax=ax, b=b, bx=bx, bias=None), bits


# ------------------------------------------------------------------------------------------------
# Cross-check against atlas_ref (NumPy engine)
# ------------------------------------------------------------------------------------------------

def _np(t):
    t = t.detach().contiguous()
    if t.dtype == torch.float8_e4m3fn:
        t = t.view(torch.uint8)
    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


class CrossCheck:
    """Re-computes contraction outputs with atlas_ref and compares words."""

    def __init__(self, root, unit, budget=1 << 27, seed=0):
        sys.path.insert(0, str(Path(root)))
        from atlas_ref import contractions as ref
        self.ref, self.unit, self.budget = ref, unit, budget
        self.rng = np.random.default_rng(seed)
        self.root = str(root)
        self.rows = []

    def check(self, node, call_index, operands, bits, full_sample):
        a, ax, b, bx, bias = (operands[k] for k in ("a", "ax", "b", "bx", "bias"))
        batch = tuple(torch.broadcast_shapes(a.shape[:-2], b.shape[:-2]))
        M, N, K = a.shape[-2], b.shape[-2], a.shape[-1]
        a_full = a.expand(*batch, M, K)
        b_full = b.expand(*batch, N, K)
        ax_full = _row_exp(ax, a.shape).expand(*batch, M, 1)
        bx_full = _row_exp(bx, b.shape).expand(*batch, N, 1)
        nb = int(np.prod(batch)) if batch else 1
        flat = lambda t: t.reshape(nb, *t.shape[-2:])  # noqa: E731
        A, B, AX, BX, OUT = flat(a_full), flat(b_full), flat(ax_full), flat(bx_full), flat(bits)
        if full_sample:
            bi = np.arange(nb)
            per_row = max(1, nb * N * K)
            nrows = max(1, min(M, self.budget // per_row))
            ri = np.arange(nrows)
            ci = np.arange(N)
        else:
            bi = np.sort(self.rng.choice(nb, size=min(nb, 2), replace=False))
            ri = np.sort(self.rng.choice(M, size=min(M, 4), replace=False))
            ci = np.sort(self.rng.choice(N, size=min(N, 32), replace=False))
        t = time.time()
        bi_t, ri_t, ci_t = (torch.from_numpy(v) for v in (bi, ri, ci))
        a_s = _np(A[bi_t][:, ri_t]); b_s = _np(B[bi_t][:, ci_t])
        ax_s = _np(AX[bi_t][:, ri_t]); bx_s = _np(BX[bi_t][:, ci_t])
        bias_s = None if bias is None else _np(bias)[ci]
        res = self.ref.contraction_nt(a_s, ax_s, b_s, bx_s, unit=self.unit, bias_bf16=bias_s)
        want = res.output_bf16.astype(np.int64)
        got = OUT[bi_t][:, ri_t][:, :, ci_t].numpy().astype(np.int64) & 0xFFFF
        mism = int((want != got).sum())
        row = {"node": node.name, "op": CONTRACTIONS[node.target], "call": call_index,
               "mode": "row_block_all_columns" if full_sample else "random_subset",
               "batch_shape": list(batch), "M": M, "N": N, "K": K,
               "batches_checked": len(bi), "rows_checked": len(ri), "cols_checked": len(ci),
               "words": int(want.size), "mismatched_words": mism,
               "seconds": round(time.time() - t, 3)}
        self.rows.append(row)
        if mism:
            print(json.dumps({"crosscheck_mismatch": row}), flush=True)
        return row


# ------------------------------------------------------------------------------------------------
# Replay
# ------------------------------------------------------------------------------------------------

class Replay(torch.fx.Interpreter):
    """Runs the saved graph; contraction nodes use the hardware accumulation when unit is set."""

    def __init__(self, gm, unit=None, threads=1, compile=True, crosscheck=None, sample=(),
                 numba_compare=False):
        super().__init__(gm)
        self.unit, self.threads, self.compile = unit, threads, compile
        self.crosscheck, self.sample = crosscheck, set(sample)
        self.numba_compare = numba_compare
        self.calls = Counter()
        self.other = Counter()
        self.call_index = 0
        self.contraction_seconds = 0.0
        self.numba_diff_words = 0
        self.numba_diff_calls = []

    def run_node(self, n):
        if n.op == "call_function" and n.target in CONTRACTIONS:
            self.calls[CONTRACTIONS[n.target]] += 1
            if self.unit is not None:
                args, kwargs = self.fetch_args_kwargs_from_env(n)
                t = time.time()
                torch.set_num_threads(self.threads)
                try:
                    value, operands, bits = hw_node(n.target, args, kwargs, self.unit, self.compile)
                    if self.numba_compare and self.unit == "mxu1":
                        _, _, nb = hw_node(n.target, args, kwargs, self.unit, self.compile, "numba")
                        d = int((nb != bits).sum())
                        if d:
                            self.numba_diff_words += d
                            self.numba_diff_calls.append({"node": n.name, "words": d})
                finally:
                    torch.set_num_threads(1)
                self.contraction_seconds += time.time() - t
                if self.crosscheck is not None:
                    self.crosscheck.check(n, self.call_index, operands, bits, self.call_index in self.sample)
                self.call_index += 1
                return value
            self.call_index += 1
        elif n.op == "call_function":
            self.other[str(n.target)] += 1
        return super().run_node(n)


def _sample_calls(gm):
    """Stratified call sample: first and last call of each op kind, the first float32-out qlinear and the first
    call whose K is not a multiple of 32 (K read from the node's fake-tensor metadata when present)."""
    order, by_kind, picks = [], {}, set()
    for n in gm.graph.nodes:
        if n.op == "call_function" and n.target in CONTRACTIONS:
            by_kind.setdefault(CONTRACTIONS[n.target], []).append(len(order))
            order.append(n)
    for idx in by_kind.values():
        picks.update({idx[0], idx[-1]})
    f32 = [i for i, n in enumerate(order) if n.target == QLINEAR and n.args[5] == "float32"]
    if f32:
        picks.add(f32[0])
    for i, n in enumerate(order):
        val = getattr(n.args[0], "meta", {}).get("val")
        if val is not None and int(val.shape[-1]) % K_TILE:
            picks.add(i)
            break
    return sorted(picks)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _bitwise(t):
    t = t.detach().contiguous()
    width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()]
    return t.view(width)


def _git(*args):
    try:
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bundle", required=True, help="bundle root (holding fx/) or its fx/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--units", nargs="+", default=list(UNITS), choices=UNITS)
    ap.add_argument("--threads", type=int, default=16, help="torch threads inside the integer accumulation")
    ap.add_argument("--no-compile", action="store_true", help="eager accum_torch (slow; same words)")
    ap.add_argument("--crosscheck", default=None, help="directory holding atlas_ref/ (NumPy engine)")
    ap.add_argument("--crosscheck-budget", type=int, default=1 << 27, help="MACs per row-block check")
    ap.add_argument("--numba-compare", action="store_true",
                    help="also run accum_torch's 'numba' IPT variant on each MXU1 call and count words that differ")
    ap.add_argument("--skip-baseline", action="store_true", help="skip the unchanged-graph replay check")
    args = ap.parse_args(argv)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("run on CPU only: set CUDA_VISIBLE_DEVICES=''")
    torch.set_num_threads(1)
    for name in ("recompile_limit", "cache_size_limit", "accumulated_recompile_limit",
                 "accumulated_cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, max(getattr(torch._dynamo.config, name), 256))

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file

    fx = Path(args.bundle)
    fx = fx / "fx" if (fx / "fx" / "model.pt2").is_file() else fx
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((fx / "fx_manifest.json").read_text())
    if not manifest.get("native_fp8"):
        raise SystemExit("bundle is not a native-FP8 FX bundle")
    for name, rec in manifest["files"].items():  # the bundle is read only; check it is the recorded one
        if name in ("fixtures.safetensors", "parity.json") and _sha256(fx / name) != rec["sha256"]:
            raise SystemExit(f"{name} differs from fx_manifest.json")
    model_sha = manifest["files"]["model.pt2"]["sha256"]
    parity = json.loads((fx / "parity.json").read_text())
    fixtures = load_file(str(fx / "fixtures.safetensors"))
    with safe_open(str(fx / "fixtures.safetensors"), "pt") as f:
        src_meta = f.metadata() or {}

    t0 = time.time()
    program = torch.export.load(fx / "model.pt2")
    gm = program.module()
    load_seconds = time.time() - t0
    sample = _sample_calls(gm)

    record = {
        "schema": SCHEMA,
        "bundle": str(fx.parent),
        "fx_dir": str(fx),
        "source_files": {"model.pt2": {"path": str(fx / "model.pt2"), "sha256": model_sha,
                                       "bytes": manifest["files"]["model.pt2"]["bytes"],
                                       "sha256_source": "fx_manifest.json (checked by the bundle's own "
                                                        "verify_saved)"},
                         "fixtures.safetensors": {"path": str(fx / "fixtures.safetensors"),
                                                  "sha256": _sha256(fx / "fixtures.safetensors")},
                         "parity.json": {"path": str(fx / "parity.json"), "sha256": _sha256(fx / "parity.json")},
                         "fx_manifest.json": {"path": str(fx / "fx_manifest.json"),
                                              "sha256": _sha256(fx / "fx_manifest.json")}},
        "k_order": K_ORDER,
        "units": {u: UNIT_SEMANTICS[u] for u in args.units},
        "contraction_epilogue": EPILOGUE,
        "reference_semantics": "Atlas backend independent reference atlas_ref (contractions.py contraction_nt / "
                               "qlinear, arithmetic.py matmul_codes / readout_bf16 / _add_bf16), bit for bit",
        "scope": "exact for the FP8 contractions only. Every other node (quantize_act, RMSNorm, softmax, residual "
                 "and other adds, RoPE, SiLU/GELU, masks, casts, host-side ops) is the unchanged torch graph "
                 "arithmetic, so these fixtures are not the Atlas whole-program output; the compiler's exact "
                 "target for those ops remains the independent reference (rotary-host/reference "
                 "whole_reference.py).",
        "generator": {
            "tool": "src/scripts/tile_flow/hw_fixtures.py",
            "command": " ".join([sys.executable, *sys.argv]),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": _git("rev-parse", "HEAD"),
            "worktree_clean": _git("status", "--porcelain", "--untracked-files=no") == "",
            "accum_torch": {"path": "src/mxu/accum_torch.py",
                            "sha256": _sha256(SRC / "mxu" / "accum_torch.py"),
                            "last_commit": _git("log", "-1", "--format=%H", "--", "src/mxu/accum_torch.py"),
                            "ipt_variant": IPT_VARIANT, "compiled": not args.no_compile},
            "atlas_fx_ops": {"path": "src/exporter/atlas_fx_ops.py",
                             "sha256": _sha256(SRC / "exporter" / "atlas_fx_ops.py")},
            "torch": torch.__version__,
            "device": "cpu",
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "graph_threads": 1,
            "accumulation_threads": args.threads,
            "model_load_seconds": round(load_seconds, 1),
        },
        "cases": len(parity["cases"]),
        "baseline_replay": None,
        "replaced_ops": None,
        "remaining_torch_ops": None,
        "results": {},
        "files": {},
    }

    def run_case(case, unit, crosscheck=None, numba_compare=False):
        interp = Replay(gm, unit, args.threads, not args.no_compile, crosscheck, sample if case == 0 else (),
                        numba_compare)
        inputs = [fixtures[k] for k in case_rec(case)["inputs"]]
        t = time.time()
        with torch.no_grad():
            value = interp.run(*inputs)
        leaves = [v for v in pytree.tree_leaves(value)]
        names = case_rec(case)["outputs"]
        if len(leaves) != len(names) or not all(isinstance(v, torch.Tensor) for v in leaves):
            raise SystemExit(f"case {case}: {len(leaves)} outputs, parity.json lists {len(names)}")
        return interp, dict(zip(names, leaves)), time.time() - t

    def case_rec(case):
        return parity["cases"][case]

    # 1. Harness check: the unchanged graph through the same interpreter reproduces fixtures.safetensors.
    if not args.skip_baseline:
        base = {"byte_identical": True, "cases": []}
        for case in range(len(parity["cases"])):
            interp, outs, secs = run_case(case, None)
            same = {k: bool(torch.equal(_bitwise(v), _bitwise(fixtures[k]))) and v.dtype == fixtures[k].dtype
                    and v.shape == fixtures[k].shape for k, v in outs.items()}
            base["cases"].append({"case": case, "outputs": len(outs), "byte_identical": all(same.values()),
                                  "seconds": round(secs, 1)})
            base["byte_identical"] &= all(same.values())
            if case == 0:
                record["replaced_ops"] = {"targets": sorted(CONTRACTIONS.values()),
                                          "calls_per_case": dict(sorted(interp.calls.items())),
                                          "calls_total": sum(interp.calls.values()) * len(parity["cases"])}
                record["remaining_torch_ops"] = {"note": record["scope"],
                                                 "calls_per_case": dict(sorted(interp.other.items()))}
        record["baseline_replay"] = base
        print(json.dumps({"baseline": base}), flush=True)
        if not base["byte_identical"]:
            raise SystemExit("unchanged-graph replay does not reproduce fixtures.safetensors; refusing")

    # 2. Hardware replays.
    for unit in args.units:
        new = {k: v for k, v in fixtures.items()}
        res = {"file": f"fixtures_hw_{unit}.safetensors", "cases": [], "differing_words": {}}
        cc = CrossCheck(args.crosscheck, unit, args.crosscheck_budget) if args.crosscheck else None
        numba_words, numba_calls = 0, []
        for case in range(len(parity["cases"])):
            interp, outs, secs = run_case(case, unit, cc, args.numba_compare)
            if record["replaced_ops"] is None:
                record["replaced_ops"] = {"targets": sorted(CONTRACTIONS.values()),
                                          "calls_per_case": dict(sorted(interp.calls.items())),
                                          "calls_total": sum(interp.calls.values()) * len(parity["cases"])}
                record["remaining_torch_ops"] = {"note": record["scope"],
                                                 "calls_per_case": dict(sorted(interp.other.items()))}
            for k, v in outs.items():
                ref = fixtures[k]
                if v.dtype != ref.dtype or v.shape != ref.shape:
                    raise SystemExit(f"{k}: {v.dtype} {tuple(v.shape)} vs fixture {ref.dtype} {tuple(ref.shape)}")
                new[k] = v.detach().clone().contiguous()
                diff = int((_bitwise(new[k]) != _bitwise(ref)).sum())
                res["differing_words"][k] = {"dtype": str(ref.dtype).replace("torch.", ""),
                                             "shape": list(ref.shape), "words": ref.numel(),
                                             "differing": diff,
                                             "fraction": round(diff / max(1, ref.numel()), 6)}
            res["cases"].append({"case": case, "contraction_calls": sum(interp.calls.values()),
                                 "seconds": round(secs, 1),
                                 "contraction_seconds": round(interp.contraction_seconds, 1)})
            numba_words += interp.numba_diff_words
            numba_calls += [dict(c, case=case) for c in interp.numba_diff_calls]
            print(json.dumps({"unit": unit, "case": case, "seconds": round(secs, 1),
                              "differing": {k: res["differing_words"][k]["differing"] for k in outs}}), flush=True)
        tot = sum(r["differing"] for r in res["differing_words"].values())
        words = sum(r["words"] for r in res["differing_words"].values())
        res["differing_words_total"] = {"differing": tot, "words": words, "fraction": round(tot / max(1, words), 6)}
        if args.numba_compare and unit == "mxu1":
            res["mxu1_model47_vs_numba"] = {
                "note": "words of MXU1 contraction outputs (same operands) that change if the short-total "
                        "conversion is normalized as in accum_torch's 'numba' variant",
                "differing_words": numba_words, "calls": numba_calls}
        if cc is not None:
            res["crosscheck"] = {"reference": "atlas_ref.contractions.contraction_nt, NumPy engine",
                                 "atlas_ref_root": cc.root,
                                 "atlas_ref_sources_sha256": {
                                     n: _sha256(Path(cc.root) / "atlas_ref" / n)
                                     for n in ("arithmetic.py", "contractions.py")},
                                 "calls_checked": len(cc.rows),
                                 "calls_row_block_all_columns": sum(r["mode"] != "random_subset" for r in cc.rows),
                                 "words": sum(r["words"] for r in cc.rows),
                                 "mismatched_words": sum(r["mismatched_words"] for r in cc.rows),
                                 "row_block_calls": [r for r in cc.rows if r["mode"] != "random_subset"]}
            print(json.dumps({"unit": unit, "crosscheck": {k: v for k, v in res["crosscheck"].items()
                                                           if k != "row_block_calls"}}), flush=True)
        # One header key holding sorted JSON, so the file bytes do not depend on a map's iteration order.
        meta = dict(src_meta)
        meta["atlas_fixtures_hw"] = json.dumps(
            {"contractions": "Atlas hardware accumulation", "mxu_unit": unit, "k_order": "ascending",
             "k_instruction_width": K_TILE, "short_k_instruction": "zero-padded",
             "other_ops": "unchanged torch graph arithmetic", "details": "fixtures_hw.json"},
            sort_keys=True, separators=(",", ":"))
        if len(meta) > 1:
            raise SystemExit("source fixtures carry safetensors metadata; refusing to reorder it")
        path = out / res["file"]
        tmp = path.with_suffix(".tmp")
        save_file({k: new[k] for k in fixtures}, str(tmp), metadata=meta)
        os.replace(tmp, path)
        record["results"][unit] = res
    record["files"] = {p.name: {"sha256": _sha256(p), "bytes": p.stat().st_size}
                       for p in sorted(out.glob("fixtures_hw_*.safetensors"))}
    (out / "fixtures_hw.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"done": str(out), "files": record["files"]}), flush=True)


if __name__ == "__main__":
    main()
