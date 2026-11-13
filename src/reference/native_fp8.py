"""native_fp8 (NEW): the ref_conv operand casts, with every quantized matmul on FP8 tensor cores.

The codes and power-of-two exponents are exactly those of `ref_conv` (the same RefQuantLinear,
RefQuantConv2d and ref_attention casts). Only the contraction changes:

  ref_conv    F.linear / torch.matmul on the dequantized operands (e4m3(code) * 2^s), in the layer
              dtype for Linear / Conv2d and in float32 for attention
  native_fp8  torch._scaled_mm on the codes viewed as torch.float8_e4m3fn with unit scales and a
              float32 output (cuBLAS), then x 2^(sx+sw) (algos.base.output_scale, exact: a power of two),
              then one cast to the layer dtype and the bias after (RefQuantLinear's fm path, quant_linear.py:169-172)

Every scale in the recipes this mode serves is constant along K (per tensor, per token, per 32-token block,
per output channel, per 32-channel panel, per (batch, head), per row block), so one GEMM per site and
an output rescale is exact. Per-K-block scales (granularity block32) are refused, as under reference_hw.

Code compatibility: the Atlas encoder (ref_numerics.e4m3_vpack) never emits the 16 codes on which Atlas
and torch decode E4M3 differently (exponent-field-0 non-zero codes 0x01-0x07 / 0x81-0x87, which torch reads
as subnormals and Atlas as zero, and 0x7F / 0xFF, NaN in torch). Every operand handed to the kernel is checked
for them on the device; any occurrence makes the run invalid (`summary()["forbidden_codes"]`).

Shapes: cuBLAS FP8 needs K % 16 == 0 and N % 16 == 0 (M is free). Other shapes are zero-padded (code 0x00 is
+0 in both decodings), which adds exact zeros to every sum; the padded output columns are sliced off.

Kernel gate: `summary()` reports the torch._scaled_mm calls this backend issued, the calls a process-wide
wrapper of torch._scaled_mm saw with float8_e4m3fn operands, and, per Linear / Conv2d site, forward calls
against kernel calls. A site that ran without reaching the kernel, a mismatch between the two counts, or no
kernel call at all marks the result invalid (quant_modes.finalize_report).

Per-site comparison: the first `compare_calls` calls of each site also compute the ref_conv product of the
same codes and record the difference (max abs, max rel, fraction of differing output words).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import reference.ref_numerics as rn

FP8 = torch.float8_e4m3fn
ALIGN = 16
KERNEL = "torch._scaled_mm (cuBLAS FP8 E4M3 x E4M3, unit tensor scales, float32 out)"


class _ScaledMMCounter:
    """Process-wide wrapper of torch._scaled_mm that counts calls by operand dtype (the kernel gate)."""

    def __init__(self):
        self.calls = {}
        self._orig = None

    def install(self):
        if self._orig is None:
            self._orig = torch._scaled_mm

            def wrapped(a, b, *args, **kw):
                key = f"{a.dtype}x{b.dtype}".replace("torch.", "")
                self.calls[key] = self.calls.get(key, 0) + 1
                return self._orig(a, b, *args, **kw)

            torch._scaled_mm = wrapped
        return self

    def fp8_calls(self) -> int:
        return self.calls.get("float8_e4m3fnxfloat8_e4m3fn", 0)


COUNTER = _ScaledMMCounter()


def is_native(fm) -> bool:
    return bool(getattr(fm, "native_fp8", False))


def check_device(device) -> None:
    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"native_fp8 needs a CUDA GPU with FP8 tensor cores (compute capability >= 8.9); "
                           f"device is {device!r}")
    cap = torch.cuda.get_device_capability(dev.index if dev.index is not None else torch.cuda.current_device())
    if cap < (8, 9):
        raise RuntimeError(f"native_fp8 needs compute capability >= 8.9, found {cap[0]}.{cap[1]}")


def _round_up(n: int, m: int = ALIGN) -> int:
    return -(-n // m) * m


def forbidden_mask(codes: torch.Tensor) -> torch.Tensor:
    """True where a code byte is one of the 16 codes Atlas and torch decode differently."""
    c = codes.view(torch.uint8) if codes.dtype == FP8 else codes
    c = c.to(torch.int32)
    return (((c & 0x78) == 0) & ((c & 0x07) != 0)) | ((c & 0x7F) == 0x7F)


def _as_fp8_2d(codes: torch.Tensor, rows_pad: int, cols_pad: int) -> torch.Tensor:
    """[R, C] uint8 / float8 codes -> contiguous float8_e4m3fn [rows_pad, cols_pad], zero-padded."""
    c = codes.view(torch.uint8) if codes.dtype == FP8 else codes
    if c.dtype != torch.uint8:
        raise TypeError(f"native_fp8: operand codes must be uint8 or float8_e4m3fn, got {c.dtype}")
    r, k = c.shape
    if (r, k) != (rows_pad, cols_pad):
        c = F.pad(c, (0, cols_pad - k, 0, rows_pad - r))
    return c.contiguous().view(FP8)


class NativeFP8:
    """The `fm` of the native_fp8 profile: takes E4M3 codes (never unpacked values) and runs cuBLAS FP8."""

    native_fp8 = True

    def __init__(self, compare_calls: int = 1):
        COUNTER.install()
        self.compare_calls = int(compare_calls)
        self.kernel_calls = 0
        self.kernel_calls_by_role = {}
        self.site_calls = {}           # site -> forward calls that went through this backend
        self.site_kernel_calls = {}    # site -> torch._scaled_mm calls issued for it
        self.padded_calls = 0
        self.compare = {}              # site -> list of per-call comparison records
        self._bad = None               # device-side count of forbidden codes seen
        self._checked_weights = set()
        self._counter_start = COUNTER.fp8_calls()

    # -- bookkeeping -------------------------------------------------------------------------------
    def _check_codes(self, codes: torch.Tensor, weight: bool = False) -> None:
        if weight:
            key = (codes.data_ptr(), tuple(codes.shape))
            if key in self._checked_weights:
                return
            self._checked_weights.add(key)
        n = forbidden_mask(codes).sum()
        self._bad = n if self._bad is None else self._bad + n

    # -- kernels -----------------------------------------------------------------------------------
    def _mm2d(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a [M, K] codes times b [N, K]^T codes -> float32 [M, N] on the FP8 tensor cores."""
        m, k = a.shape
        n = b.shape[0]
        kp, np_ = _round_up(k), _round_up(n)
        if kp != k or np_ != n:
            self.padded_calls += 1
        a8 = _as_fp8_2d(a, m, kp)
        b8 = _as_fp8_2d(b, np_, kp)
        one = torch.ones((), dtype=torch.float32, device=a.device)
        y = torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one, out_dtype=torch.float32)
        self.kernel_calls += 1
        return y if np_ == n else y[:, :n]

    def mm_nt(self, a: torch.Tensor, b: torch.Tensor, site: str, role: str, weight_b: bool = False) -> torch.Tensor:
        """a [..., M, K] times b [..., N, K]^T on codes, float32 out (no exponents applied).

        b 2-D (a Linear / Conv2d weight): a's leading dims are flattened into M and one GEMM runs.
        b with leading dims (attention): one GEMM per broadcast (batch, head) matrix."""
        if a.device.type != "cuda":
            raise RuntimeError(f"native_fp8: {site}: operands on {a.device}; the FP8 kernel needs CUDA")
        if a.shape[-1] != b.shape[-1]:
            raise ValueError(f"native_fp8: {site}: K mismatch {tuple(a.shape)} x {tuple(b.shape)}^T")
        self._check_codes(a)
        self._check_codes(b, weight=weight_b)
        before = self.kernel_calls
        if b.dim() == 2:
            y = self._mm2d(a.reshape(-1, a.shape[-1]), b).reshape(*a.shape[:-1], b.shape[0])
        else:
            batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            aa = a.expand(*batch, *a.shape[-2:]).reshape(-1, *a.shape[-2:])
            bb = b.expand(*batch, *b.shape[-2:]).reshape(-1, *b.shape[-2:])
            y = torch.stack([self._mm2d(aa[i], bb[i]) for i in range(aa.shape[0])])
            y = y.reshape(*batch, a.shape[-2], b.shape[-2])
        self.site_calls[site] = self.site_calls.get(site, 0) + 1
        self.site_kernel_calls[site] = self.site_kernel_calls.get(site, 0) + (self.kernel_calls - before)
        self.kernel_calls_by_role[role] = self.kernel_calls_by_role.get(role, 0) + (self.kernel_calls - before)
        return y

    def linear(self, x_q, s_x, w_q, s_w, dtype, site: str, role: str = "linear") -> torch.Tensor:
        """RefQuantLinear's fm path on the kernel: (x_codes . w_codes^T) * 2^(sx+sw) in float32, then `dtype`.
        The bias is added by the caller, after the cast."""
        from algos import base as algo_hooks

        y = self.mm_nt(x_q, w_q, site, role, weight_b=True)
        out = (y * algo_hooks.output_scale(s_x, s_w)).to(dtype)
        if self._want_compare(site):
            ref = F.linear(rn.dequantize(x_q, s_x).to(dtype), rn.dequantize(w_q, s_w).to(dtype))
            self._record(site, out, ref)
        return out

    def attention(self, a_q, a_s, b_q, b_s, pop_shift: int, site: str, role: str) -> torch.Tensor:
        """ref_attention._mm on the kernel: float32 (a . b^T) * 2^(sa+sb+pop_shift)."""
        y = self.mm_nt(a_q, b_q, site, role)
        out = y * torch.exp2((a_s + b_s + int(pop_shift)).float())
        if self._want_compare(site):
            ref = rn.dq_matmul_nt(a_q, a_s, b_q, b_s)
            if pop_shift:
                ref = ref * (2.0 ** int(pop_shift))
            self._record(site, out, ref)
        return out

    # -- per-site comparison against the ref_conv product of the same codes -------------------------
    def _want_compare(self, site: str) -> bool:
        return self.compare_calls > 0 and len(self.compare.get(site, ())) < self.compare_calls

    def _record(self, site: str, out: torch.Tensor, ref: torch.Tensor) -> None:
        self.compare.setdefault(site, []).append(compare_outputs(out, ref))

    # -- report ------------------------------------------------------------------------------------
    def summary(self, linear_sites=None) -> dict:
        """The kernel gate. `linear_sites`: {site: forward calls} of the Linear / Conv2d modules (counted
        by their own forward), so a site that ran without reaching the kernel is caught."""
        bad = int(self._bad.item()) if self._bad is not None else 0
        seen = COUNTER.fp8_calls() - self._counter_start
        problems = []
        if self.kernel_calls == 0:
            problems.append("no torch._scaled_mm call")
        if seen != self.kernel_calls:
            problems.append(f"torch._scaled_mm wrapper saw {seen} FP8 calls, the backend issued {self.kernel_calls}")
        if bad:
            problems.append(f"{bad} operand codes are among the 16 codes Atlas and torch decode differently")
        missed = {}
        for site, n in (linear_sites or {}).items():
            k = self.site_kernel_calls.get(site, 0)
            if n > 0 and k < 1:
                missed[site] = n
        if missed:
            problems.append(f"{len(missed)} sites ran without a kernel call: {sorted(missed)[:8]}")
        recs = [r for v in self.compare.values() for r in v]
        cmp = None
        if recs:
            n_el = sum(r["elements"] for r in recs)
            cmp = {"sites": len(self.compare), "calls": len(recs), "elements": n_el,
                   "max_abs": max(r["max_abs"] for r in recs),
                   "max_rel_to_site_amax": max(r["max_rel_to_amax"] for r in recs),
                   "max_rel_elementwise": max(r["max_rel"] for r in recs),
                   "frac_words_differing": sum(r["words_differing"] for r in recs) / max(n_el, 1),
                   "frac_words_differing_bf16": sum(r["bf16_words_differing"] for r in recs) / max(n_el, 1),
                   "worst_site_by_rel": max(((s, max(r["max_rel_to_amax"] for r in v)) for s, v in self.compare.items()),
                                            key=lambda t: t[1])[0],
                   "per_site": {s: v for s, v in self.compare.items()}}
        return {"kernel": KERNEL, "ok": not problems, "problems": problems,
                "kernel_calls": self.kernel_calls, "scaled_mm_fp8_calls_seen": seen,
                "kernel_calls_by_role": dict(self.kernel_calls_by_role), "padded_calls": self.padded_calls,
                "forbidden_codes": bad,
                "linear_conv_sites_called": sum(1 for n in (linear_sites or {}).values() if n > 0),
                "linear_conv_sites_not_called": sorted(s for s, n in (linear_sites or {}).items() if n == 0),
                "sites_with_kernel_calls": len(self.site_kernel_calls),
                "min_kernel_calls_per_called_site": min((self.site_kernel_calls.get(s, 0)
                                                         for s, n in (linear_sites or {}).items() if n > 0), default=None),
                "compare_vs_ref_conv": cmp}


def compare_outputs(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """native vs ref_conv on one call: max |diff|, max |diff| / max |ref|, max elementwise relative
    difference (where ref != 0), and the number of output words (in the output dtype, and after rounding
    both to BF16) that differ."""
    o, r = out.float(), ref.float()
    d = (o - r).abs()
    amax = r.abs().max().clamp_min(1e-30)
    nz = r != 0
    rel = (d[nz] / r[nz].abs()).max().item() if bool(nz.any()) else 0.0
    if out.dtype in (torch.bfloat16, torch.float16):
        words = int((out.view(torch.int16) != ref.to(out.dtype).view(torch.int16)).sum())
    else:
        words = int((out.contiguous().view(torch.int32) != ref.to(out.dtype).contiguous().view(torch.int32)).sum())
    bwords = int((out.to(torch.bfloat16).view(torch.int16) != ref.to(torch.bfloat16).view(torch.int16)).sum())
    return {"elements": out.numel(), "dtype": str(out.dtype).replace("torch.", ""),
            "max_abs": d.max().item(), "max_rel_to_amax": (d.max() / amax).item(), "max_rel": rel,
            "words_differing": words, "bf16_words_differing": bwords}


def to_native(model, backend: NativeFP8) -> dict:
    """Switch a model quantized under ref_conv (fm None) to the kernel in place: the same modules, codes
    and exponents. Returns counts of what was switched."""
    from reference.ref_attention import RefQuantConv2d
    from reference.ref_profile import RefQuantLinear

    n_lin = n_conv = n_attn = 0
    for m in model.modules():
        if isinstance(m, (RefQuantLinear, RefQuantConv2d)):
            if m.fm is not None:
                raise ValueError(f"native_fp8: {m.layer_name} already has an accumulation model ({type(m.fm).__name__})")
            if isinstance(m, RefQuantLinear) and m.weight_block:
                raise NotImplementedError(f"{m.layer_name}: per-K-block weight scales; native_fp8 runs one GEMM per site")
            m.fm = backend
            n_lin += isinstance(m, RefQuantLinear)
            n_conv += isinstance(m, RefQuantConv2d)
        cfg = getattr(m, "_atlas_ref_attn", None)
        if isinstance(cfg, dict) and "fm_qk" in cfg:
            if cfg["fm_qk"] is not None or cfg["fm_av"] is not None:
                raise ValueError("native_fp8: attention site already has an accumulation model")
            cfg["fm_qk"] = cfg["fm_av"] = backend
            n_attn += 1
    return {"linear": n_lin, "conv2d": n_conv, "attention_sites": n_attn}


class SiteCallCounter:
    """Forward calls of every RefQuantLinear / RefQuantConv2d of a model (the gate's denominator)."""

    def __init__(self, model):
        from reference.ref_attention import RefQuantConv2d
        from reference.ref_profile import RefQuantLinear

        self.calls = {}
        self._handles = []
        for m in model.modules():
            if isinstance(m, (RefQuantLinear, RefQuantConv2d)) and is_native(m.fm):
                self.calls[m.layer_name] = 0
                self._handles.append(m.register_forward_pre_hook(self._hook(m.layer_name)))

    def _hook(self, name):
        def pre(mod, args):
            self.calls[name] += 1
        return pre

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []
