"""Export of a quantized model as files (NEW; docs/algos/export.md).

`export(model, export_dir, ...)` writes, for one model x recipe, right after algos.quantize (so
with calibration, folds and algorithm transforms applied) and before any evaluation:

  model.safetensors  the post-quantization state: per quantized Linear / Conv2d its E4M3 codes
                     (`<fqn>.weight_codes`, uint8, as ref_numerics.e4m3_vpack encodes them) and
                     power-of-two exponents (`<fqn>.weight_scale_exp`, int32), and every other
                     tensor of the state dict as it stands after the folds (norm weights, producer
                     rows, biases, embeddings, BF16 opt-out layers). The float weight of a
                     quantized layer is not stored.
  policy.json        the per-site execution policy: operand format, weight / activation scale
                     rules, output rescale, accumulation, residual passes, BF16 opt-outs, runtime
                     requirements. load_quantized.py executes it (its round trip is the test).
  manifest.json      provenance, sha256 of the two files above, NEW decisions, pending rulings.

Only the reference-numerics modes (ref_linear .. chip) can be exported: they are the ones whose
weights are E4M3 codes with E8M0-style exponents (src/ref_profile.py, src/ref_attention.py).
Everything not taken from a source is marked NEW here and listed in NEW_DECISIONS.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

import pack_plan
import tile_native

POLICY_SCHEMA = "atlas_torchao.execution_policy.v1"        # NEW
MANIFEST_SCHEMA = "atlas_torchao.export_manifest.v1"       # NEW
FORMAT_VERSION = 1
EXPORTABLE_BASES = ("reference", "reference_hw", "chip")
PREFERRED_MODE = "ref_conv"
QUANT_SUFFIXES = (".weight_codes", ".weight_scale_exp", ".weight_lo_codes", ".weight_lo_scale_exp",
                  ".input_scale_exp")

# The operand format as the reference author's encoder implements it (ref_numerics.e4m3_vpack, a
# port of pi0-quant fp8_e4m3.py:38-86 = FP8Pack.scala:57-95). Field names follow Merlin's
# merlin.quantization_adapter_contract.v1 `operand` block where one applies.
OPERAND = {
    "dtype": "f8E4M3",
    "exponent_bits": 4, "mantissa_bits": 3, "bias": 7,
    "exponent_range": [1, 15],
    "exponent_15_finite": True,
    "finite_max": 448.0,
    "min_normal": 2.0 ** -6,
    "reserved_codes": [0x7F, 0xFF],
    "subnormal_flush": True,
    "signed_zero": False,
    "rounding": "rne",
    "saturate": True,
    "encoder": "ref_numerics.e4m3_vpack (fp8_e4m3.py:38-86, FP8Pack.scala:57-95)",
    "notes": ["0x7F / 0xFF (480) are clamped to 0x7E / 0xFE (448) (fp8_e4m3.py:84-85)",
              "results below 2^-6 flush to +0 (code 0x00); the encoder never emits 0x80",
              "BF16 zero / subnormal / NaN encode as 0x00; +-Inf saturate to +-448",
              "operands are cast to BF16 (RNE) before encoding (hw_float_ops.py:89)"],
}

SCALE = {
    "encoding": "e8m0_exponent_int32",
    "value": "real = e4m3(code) * 2^exp",
    "reference_rule": "floor(log2(amax_bf16 / 256)), 0 when amax is 0 (hw_float_ops.py:38-57; "
                      "ref_numerics.po2_scale_exp)",
    "nc_rule": "NEW (algos/granularity.py): reference rule + 1 where amax > 448 * 2^s, i.e. "
               "ceil(log2(amax / 448))",
    "amax": "max |x| in BF16 after nan_to_num(nan=0, posinf=0) over the reduction dims",
}

NEW_DECISIONS = [
    "The export format itself (three files, tensor names, policy and manifest schemas) is NEW; "
    "no source defines a file format for these numerics.",
    "Export happens right after algos.quantize of the first reference-numerics mode in --modes "
    "(ref_conv preferred), before evaluation; the weights are the same for every reference mode.",
    "Weight exponents keep the in-memory shape (scalar per tensor, [out, 1] per channel, "
    "[out, K/32, 1] per block) as int32; activation exponents of static sites are stored as "
    "<fqn>.input_scale_exp.",
    "The loader replaces each quantized layer's float weight with dequant(codes) in the layer "
    "dtype so that `layer.weight.dtype` / shape introspection keeps working; the quantized "
    "forward reads only the codes and exponents.",
    "Every non-quantized tensor is stored in its own dtype (BF16 for the BF16 models; FP32 where "
    "the model keeps FP32, e.g. smoothquant's promoted (1+w) norms and SmolVLA's action projections).",
    "BF16 opt-outs and --skip layers are recorded with m2m manifest status 'host' / 'skipped'.",
    "Calibration data are identified by a sha256 over the calibration batches as the CLI held them "
    "(tensor bytes, numpy bytes, nested dict / list structure).",
]

PENDING_RULINGS = [
    {"id": "Q2", "ruling": "run-time vs static activation scales (is a dynamic amax on chip intended?)",
     "affects": "every site whose activation mode is dynamic (atlas_v1_awq / atlas_v1_sq per-token; "
                "attention operands of every recipe)", "status": "open"},
    {"id": "H2", "ruling": "how a per-row exponent is derived from vredmax.row (no floor / exponent-extract "
                           "instruction)", "affects": "per-token activation and dynamic per-tensor scales",
     "status": "open"},
    {"id": "vadd", "ruling": "vadd rounding: trunc (as built) vs rn (24-instruction sequence) vs an RNE adder",
     "affects": "chip residual adds (runtime.chip_options)", "status": "open"},
    {"id": "448_vs_240", "ruling": "operand exponent 15: 448 (RTL: exponent 15 finite) vs 240 (spec "
                                   "exponent_range [1,14])", "affects": "every FP8 cast (operand.finite_max)",
     "status": "open; this export uses 448 (the reference author's numerics)"},
    {"id": "R4", "ruling": "can residual (fp8x3) passes share the MXU accumulator; BF16 fallback",
     "affects": "residual_passes sites and BF16 opt-outs", "status": "open"},
]

M2M_V2_DIFFERENCES = [
    "schema is atlas_torchao.export_manifest.v1, not m2m.quantization_manifest.v2",
    "adapter_id is absent: no m2m.quantization_adapters entry point exists yet (pyproject.toml)",
    "contract_sha256 is absent: no merlin.quantization_adapter_contract.v1 was read; the operand "
    "facts are the reference author's numerics, not a target contract",
    "site status 'preserved' (with execution_route) is not used; BF16 opt-outs are 'host'",
    "extra top-level fields (files, provenance, recipe, calibration, new_decisions, "
    "pending_hardware_rulings) are not part of v2",
]
M2M_V2_MATCHING = ["policy_sha256", "sites[].site_id", "sites[].status (quantized | host | skipped)",
                   "site_id uniqueness", "JSON without NaN"]


# ---------------------------------------------------------------------------------------------
# Mode selection
# ---------------------------------------------------------------------------------------------

def pick_mode(modes) -> str | None:
    """The mode whose quantized model is exported: ref_conv if present, else the first mode of
    `modes` whose profile is a reference-numerics one (ref_linear .. chip); None if there is none."""
    import quant_modes

    ok = [m for m in modes if m != "bf16"
          and quant_modes.PROFILES[quant_modes.canonical(m)].base in EXPORTABLE_BASES]
    for m in ok:
        if quant_modes.canonical(m) == PREFERRED_MODE:
            return m
    return ok[0] if ok else None


def add_args(ap) -> None:
    g = ap.add_argument_group("export (docs/algos/export.md)")
    g.add_argument("--export-dir", default=None,
                   help="write model.safetensors, policy.json and manifest.json of the first "
                        "reference-numerics mode in --modes (ref_conv preferred) right after quantization")
    g.add_argument("--export-only", action="store_true",
                   help="with --export-dir: quantize the exported mode, write the export and stop "
                        "(no evaluation)")
    g.add_argument("--allow-uncalibrated-fallback", action="store_true",
                   help="declare in policy.json that Linear / Conv2d sites of a static recipe without "
                        "calibration data (e.g. an unused lm_head) may keep the dynamic per-tensor rule; "
                        "undeclared sites make verify_bundle.py fail 'static recipe is fully static'")
    g.add_argument("--pack-fold", action="store_true",
                   help="after the export, fold the static activation pack exponents into the compiler's admitted E8M0 range "
                        "122..128 where the producer already multiplies by a constant (RMSNorm gamma, up_proj weight exponent, "
                        "PV pop rescale, a Linear feeding a Linear); exact, no re-calibration (docs/algos/pack-fold.md). "
                        "Default off: the export is unchanged. The unfolded export is kept in <export-dir>.unfolded")
    g.add_argument("--pack-fold-graph", default=None,
                   help="with --pack-fold: fx/ directory (graph.txt, graph_signature.txt) of a native-FP8 capture of this model's "
                        "export; default: capture the graph in process (causal LMs only)")
    g.add_argument("--from-export", default=None,
                   help="build every quantized mode from this export (load_quantized.py) instead of "
                        "quantizing: no calibration, no float weights of the quantized layers")


def check_args(ap, args) -> str | None:
    """Validate the export flags; returns the mode to export (None without --export-dir)."""
    if getattr(args, "export_only", False) and not getattr(args, "export_dir", None):
        ap.error("--export-only needs --export-dir")
    if getattr(args, "from_export", None) and getattr(args, "export_dir", None):
        ap.error("--from-export and --export-dir exclude each other")
    if not getattr(args, "export_dir", None):
        return None
    mode = pick_mode(args.modes)
    if mode is None:
        ap.error("--export-dir needs a reference-numerics mode in --modes (ref_linear, ref_attn, ref_conv, "
                 "reference, reference_hw, chip, chip_rope_vpu)")
    return mode


# ---------------------------------------------------------------------------------------------
# Activation rules: what each algorithm's act_scale does, as data
# ---------------------------------------------------------------------------------------------

REFERENCE_ACT = {"mode": "dynamic", "source": "reference", "rule": "ref"}
# Linear / Conv2d of a static recipe with no calibration data: same numerics as REFERENCE_ACT (source
# "reference" is what replay and the FX capture dispatch on), but declared in policy.json and never tile-native.
UNCALIBRATED_FALLBACK_ACT = {**REFERENCE_ACT, "uncalibrated_fallback": True, "label": "uncalibrated_fallback"}


def _granularity_rule(g, name, kind):
    if kind.startswith("attention."):
        if g.attn == "defer":
            return None
        if g.attn == "token" and kind in ("attention.q", "attention.p"):
            return {"mode": "dynamic", "reduce": "token", "rule": g.rule, "label": "per_token",
                    "out_cols": 0, "source": "granularity"}
        if g.attn == "block" and kind in ("attention.q", "attention.p"):
            return {"mode": "dynamic", "reduce": "head_block", "rule": "nc", "rows": g.rows,
                    "label": "per_row_block", "tile_label": tile_native.block_rows_label(g.rows),
                    "source": "granularity"}
        if g.rule == "nc" or g.attn == "block":
            return {"mode": "dynamic", "reduce": "reference_dims", "rule": "nc", "label": "per_head",
                    "source": "granularity"}
        return None
    if g.a == "defer":
        return None
    out_cols = int(g._out.get(name, 0))
    if g.a == "block" and kind == "linear":
        # algos.granularity._token_block: always the never-clip rule, Linear inputs only
        return {"mode": "dynamic", "reduce": "block", "rule": "nc", "rows": g.rows,
                "label": "per_token_block", "tile_label": tile_native.block_rows_label(g.rows),
                "out_cols": out_cols, "source": "granularity"}
    if g.a in ("tensor", "block"):                      # a=block: a Conv2d input keeps the per-tensor rule
        return None if g.rule == "ref" else {"mode": "dynamic", "reduce": "tensor", "rule": "nc",
                                             "label": "per_tensor", "source": "granularity"}
    if g.a == "block32":
        return {"mode": "dynamic", "reduce": "block32_or_token", "rule": g.rule, "block": 32,
                "label": "per_block", "tail_label": "per_token(block_tail)", "out_cols": out_cols,
                "source": "granularity"}
    return {"mode": "dynamic", "reduce": "token", "rule": g.rule, "label": "per_token",
            "out_cols": out_cols, "source": "granularity"}


def _smoothquant_rule(s, name, kind):
    if not s.static or kind != "linear" or name not in s._static_exp:
        return None
    return {"mode": "static", "exp": int(s._static_exp[name]), "label": "static_per_tensor(smoothquant)",
            "source": "smoothquant"}


def _static_calib_rule(sc, name, kind):
    if kind == "attention.p":
        if sc.p_exp is None:
            return None
        return {"mode": "static", "exp": int(sc.p_exp), "label": "static", "source": "static_calib"}
    if kind.startswith("attention.") or name not in sc.exps:
        return None
    steps = sc.step_exps.get(name)
    if steps:
        return {"mode": "static_per_step", "exps": [int(e) for e in steps], "label": "static_per_step",
                "source": "static_calib"}
    return {"mode": "static", "exp": int(sc.exps[name]), "label": "static", "source": "static_calib"}


def _example_identity_rule(_a, name, kind):
    return {"mode": "dynamic", "reduce": "reference_dims", "rule": "ref",
            "label": "per_tensor" if not kind.startswith("attention.") else "per_head",
            "source": "example_identity"}


ACT_RULES = {"granularity": _granularity_rule, "smoothquant": _smoothquant_rule,
             "static_calib": _static_calib_rule, "example_identity": _example_identity_rule}


def _members(algo):
    if algo is None:
        return []
    return list(getattr(algo, "members", None) or [algo])


def _overrides(a, hook):
    from algos.base import Algo

    return getattr(type(a), hook) is not getattr(Algo, hook)


def check_exportable(algo) -> None:
    """Refuse algorithms whose run-time behaviour this format cannot express."""
    for m in _members(algo):
        if _overrides(m, "online_transform") and m.name != "example_identity":
            raise NotImplementedError(f"export: {m.name} applies an online transform at run time, which "
                                      "the policy format does not express")
        if _overrides(m, "act_scale") and m.name not in ACT_RULES:
            raise NotImplementedError(f"export: no activation-rule descriptor for {m.name!r} "
                                      "(add one to quant_export.ACT_RULES)")


def is_static_recipe(algo) -> bool:
    """True when a member gives Linear / Conv2d inputs calibrated static exponents (static_calib, or
    smoothquant with static=true)."""
    return any(m.name == "static_calib" or (m.name == "smoothquant" and getattr(m, "static", False))
               for m in _members(algo))


def _static_covers(algo, kind) -> bool:
    """Does a static member give `kind` ("linear" / "conv2d") a calibrated exponent? static_calib: both;
    smoothquant static=true: Linear only (its Conv2d inputs keep the reference rule by design, so a Conv2d is
    never reported as uncalibrated under it)."""
    return any(m.name == "static_calib" or (m.name == "smoothquant" and getattr(m, "static", False)
                                            and kind == "linear") for m in _members(algo))


def act_rule(algo, name, kind) -> dict:
    """The activation scale rule of (site, kind) as data; REFERENCE_ACT when no member sets one, except
    that a Linear / Conv2d the calibration never saw, of a kind a static member covers, gets
    UNCALIBRATED_FALLBACK_ACT (identical numerics, declared)."""
    got = []
    for m in _members(algo):
        if not _overrides(m, "act_scale"):
            continue
        r = ACT_RULES[m.name](m, name, kind)
        if r is not None:
            got.append(r)
    if len(got) > 1:
        raise ValueError(f"export: {name} {kind}: several activation rules {got}")
    if got:
        return got[0]
    if not kind.startswith("attention.") and _static_covers(algo, kind):
        return dict(UNCALIBRATED_FALLBACK_ACT)
    return dict(REFERENCE_ACT)


# ---------------------------------------------------------------------------------------------
# Sites
# ---------------------------------------------------------------------------------------------

def _weight_granularity(exp: torch.Tensor, block: int, panel_rows: int = 0) -> str:
    if block:
        return f"per_block({block})"
    if exp.numel() == 1:
        return "per_tensor"
    if exp.dim() == 2 and exp.shape[-1] == 1:
        if panel_rows:                       # granularity w=panel: same string as algos.granularity.panel_label
            return f"per_panel({panel_rows}" + (",partial_tail)" if exp.shape[0] % panel_rows else ")")
        return "per_channel"
    return f"shape{tuple(exp.shape)}"


def weight_panel_rows(algo) -> int:
    """Rows per weight panel when a stacked granularity member chose w=panel, else 0."""
    for m in _members(algo):
        if m.name == "granularity" and getattr(m, "w", None) == "panel":
            return int(m.rows)
    return 0


def _act_description(rule, kind):
    if rule.get("uncalibrated_fallback"):
        return ("dynamic per tensor: reference rule; UNCALIBRATED FALLBACK (a static recipe, but this layer "
                "saw no calibration data, so there is no static exponent)")
    if rule.get("source") == "reference":
        return ("dynamic per (batch, head): reference rule over the last two dims"
                if kind.startswith("attention.") else "dynamic per tensor: reference rule")
    if rule["mode"] == "static":
        return f"static exponent {rule['exp']}"
    if rule["mode"] == "static_per_step":
        return f"static exponent per call index {rule['exps']}"
    red = {"tensor": "per tensor", "token": "per token (last dim reduced)",
           "reference_dims": "per (batch, head)" if kind.startswith("attention.") else "per tensor",
           "block32_or_token": "per 32-element block along K (per token if K % 32 != 0; NOT hardware-realizable, H3)",
           "block": f"per {rule.get('rows', 32)}-token block over the full K (flattened B*T rows, last block "
                    "may be partial; NEW)",
           "head_block": f"per {rule.get('rows', 32)}-row block of each (batch, head) matrix (NEW)"}[rule["reduce"]]
    return f"dynamic {red}: {'reference' if rule['rule'] == 'ref' else 'NEW nc'} rule"


def _is_dynamic(site) -> bool:
    rules = [site.get("activation")] + list((site.get("operands") or {}).values())
    return any(r and r.get("mode") == "dynamic" for r in rules)


def quantized_modules(model):
    """{fqn: module} of the reference modules (RefQuantLinear / RefQuantConv2d) in `model`."""
    return {n: m for n, m in model.named_modules()
            if type(m).__name__ in ("RefQuantLinear", "RefQuantConv2d")}


def attention_sites(model):
    return {n: m._atlas_ref_attn for n, m in model.named_modules()
            if isinstance(getattr(m, "_atlas_ref_attn", None), dict)}


def _accumulation(ladder):
    return {"unit": "MXU",
            "exported_mode_accumulation": ladder.get("accumulation"),
            "assumed_on_chip": "MXU inner-product tree (--mxu ipt, the reference author's choice; "
                               "accum_backends) with BF16 readout",
            "readout": "bf16"}


def build_policy(model, mode, report, algo, options, skip, bf16_opt_out, opt_out_reasons,
                 tensors_out: dict, allow_uncalibrated_fallback: bool = False) -> dict:
    """The policy dict; fills `tensors_out` with the codes / exponent tensors it names."""
    import quant_modes

    check_exportable(algo)
    ladder = report.get("ladder", {})
    sites = {}
    panel_rows = weight_panel_rows(algo)
    tile, static_act, weight_ranges = {}, {}, {}
    fallback_sites = []
    for n, m in quantized_modules(model).items():
        kind = "linear" if type(m).__name__ == "RefQuantLinear" else "conv2d"
        codes, exp = m.weight_codes.detach().cpu(), m.weight_scale_exp.detach().cpu().to(torch.int32)
        tensors_out[f"{n}.weight_codes"] = codes.contiguous()
        tensors_out[f"{n}.weight_scale_exp"] = exp.contiguous()
        rule = act_rule(algo, n, kind)
        if rule["mode"] == "static":
            tensors_out[f"{n}.input_scale_exp"] = torch.tensor(rule["exp"], dtype=torch.int32)
        block = int(getattr(m, "weight_block", 0) or 0)
        wgran = _weight_granularity(exp, block, panel_rows)
        site = {"kind": kind, "status": "quantized",
                "weight_shape": list(m.weight.shape), "weight_dtype": str(m.weight.dtype).replace("torch.", ""),
                "bias": None if m.bias is None else {"tensor": f"{n}.bias",
                                                     "dtype": str(m.bias.dtype).replace("torch.", ""),
                                                     "added": "after the output rescale, in the layer dtype"},
                "weight": {"codes": f"{n}.weight_codes", "scale_exp": f"{n}.weight_scale_exp",
                           "codes_shape": list(codes.shape),
                           "granularity": wgran, "block": block,
                           "scale_exp_shape": list(exp.shape), "mode": "static (offline)",
                           **({"panel_rows": panel_rows} if wgran.startswith("per_panel") else {}),
                           "e8m0_code_range": tile_native.code_range(exp),
                           "code_report": tile_native.weight_code_report(
                               codes, panel_rows or None, name=n)},
                "activation": {**rule, "description": _act_description(rule, kind)},
                "output_rescale": "y_bf16 * 2^(s_x + s_w) in float32, then the layer dtype "
                                  "(quant_linear.py:169-170; per-row / per-column exponents broadcast "
                                  "over [..., out], algos.base.output_scale)",
                "residual_passes": None}
        if rule["mode"] == "static":
            site["activation"]["tensor"] = f"{n}.input_scale_exp"
            site["activation"]["e8m0_code"] = tile_native.E8M0_BIAS + int(rule["exp"])
            static_act[n] = [int(rule["exp"])]
        elif rule["mode"] == "static_per_step":
            static_act[n] = [int(e) for e in rule["exps"]]
        weight_ranges[n] = tuple(site["weight"]["e8m0_code_range"].values())
        site["tile_native"] = tile[n] = tile_native.linear_site_status(wgran, exp, panel_rows, rule)
        if rule.get("uncalibrated_fallback"):
            fallback_sites.append(n)
        if kind == "conv2d":
            site["conv"] = {"kernel_size": list(m.kernel_size), "stride": list(m.stride),
                            "padding": list(m.padding), "dilation": list(m.dilation),
                            "lowering": "im2col; per-tensor cast of the unfolded input (model_patcher.py:345-375)"}
        res = getattr(m, "residual", None)
        if res:
            tensors_out[f"{n}.weight_lo_codes"] = m.weight_lo_codes.detach().cpu().contiguous()
            tensors_out[f"{n}.weight_lo_scale_exp"] = m.weight_lo_scale_exp.detach().cpu().to(torch.int32).contiguous()
            site["residual_passes"] = {
                "kind": res, "lo_codes": f"{n}.weight_lo_codes", "lo_scale_exp": f"{n}.weight_lo_scale_exp",
                "passes": ["x_hi.w_hi", "x_hi.w_lo"] + (["x_lo.w_hi"] if res == "wx" else []),
                "sum": "each pass rescaled, summed in the layer dtype (VPU vmul / vadd); pending ruling R4",
                "lo_rule": "reference per-tensor rule on w - dequant(w_hi) (src/ref_profile.py _init_residual)"}
        sites[n] = site
    for n, cfg in attention_sites(model).items():
        ops = {}
        for op in ("q", "k", "p", "v"):
            r = act_rule(algo, n, "attention." + op)
            ops[op] = {**r, "description": _act_description(r, "attention." + op)}
        tile[n] = tile_native.attention_site_status(ops)
        for op, r in ops.items():
            if r.get("mode") == "static":
                r["e8m0_code"] = tile_native.E8M0_BIAS + int(r["exp"])
                static_act[f"{n}:{op}"] = [int(r["exp"])]
        sites[n] = {"kind": "attention", "status": "quantized", "variant": cfg.get("variant"),
                    "tile_native": tile[n],
                    "matmuls": ["q.k^T (FP8 x FP8)", "softmax in BF16 / VPU", "p.v (FP8 x FP8)"],
                    "operands": ops,
                    "output_rescale": "2^(s_q + s_k) on the scores, 2^(s_p + s_v) on the output "
                                      "(src/ref_attention.py, model_patcher.py:624-723)"}
    for n in bf16_opt_out:
        mods = dict(model.named_modules())
        mod = mods.get(n)
        sites[n] = {"kind": type(mod).__name__ if mod is not None else "unknown", "status": "host",
                    "reason": opt_out_reasons.get(n, "algorithm BF16 opt-out"),
                    "weight": {"tensor": f"{n}.weight", "dtype": str(mod.weight.dtype).replace("torch.", "")
                               if mod is not None and getattr(mod, "weight", None) is not None else None}}
    for n, mod in model.named_modules():
        if n and n not in sites and type(mod) in (nn.Linear, nn.Conv2d) and any(s and s in n for s in skip):
            sites[n] = {"kind": type(mod).__name__, "status": "skipped",
                        "reason": "--skip " + next(s for s in skip if s and s in n),
                        "weight": {"tensor": f"{n}.weight", "dtype": str(mod.weight.dtype).replace("torch.", "")}}
    opts = quant_modes.resolved_options(mode, options)
    prof = quant_modes.profile_options(opts)
    base = quant_modes.PROFILES[quant_modes.canonical(mode)].base
    chip_opts = {}
    recipe = getattr(algo, "recipe", None)
    if recipe:
        chip_opts.update(recipe.get("chip_defaults") or {})
    for k in ("chip_vadd", "chip_vadd_sites", "chip_rope", "chip_mxu"):
        v = opts.get(k)
        if v is not None and (base == "chip" or k not in chip_opts):
            chip_opts[k] = v
    policy = {
        "schema": POLICY_SCHEMA, "format_version": FORMAT_VERSION,
        "exported_mode": mode, "profile": quant_modes.canonical(mode), "ops": list(prof["ops"]),
        "operand": dict(OPERAND), "scale": dict(SCALE),
        "accumulate": {**_accumulation(ladder), "mxu": prof["mxu"]},
        "skip": list(skip), "bf16_opt_out": [{"site": n, "reason": opt_out_reasons.get(n, "algorithm BF16 opt-out")}
                                             for n in bf16_opt_out],
        "sites": dict(sorted(sites.items())),
        "runtime": {
            "chip_options": chip_opts,
            "requirements": [
                "RoPE / timestep sin-cos tables are host constants (--chip-rope host, the default chip step)",
                "every op outside the listed sites runs in BF16 on the VPU / scalar core or the host "
                "(src/op_unit_map.py, chip_coverage)",
            ] + (["dynamic activation scales need a run-time amax (pending rulings Q2 / H2)"]
                 if any(_is_dynamic(s) for s in sites.values()) else []) + ([f"chip vadd {chip_opts['chip_vadd']} on {chip_opts.get('chip_vadd_sites', 'residual')} sites "
                  "(recipe chip default; NEW, off-ladder)"] if chip_opts.get("chip_vadd") else []),
        },
        "uncalibrated_fallback": {
            "static_recipe": is_static_recipe(algo),
            "count": len(fallback_sites), "sites": sorted(fallback_sites),
            "declared": bool(allow_uncalibrated_fallback),
            "note": "Linear / Conv2d sites of a static recipe without calibration data: they keep the dynamic "
                    "per-tensor reference rule (activation.uncalibrated_fallback, tile_native false). "
                    "declared=true only when the export was run with --allow-uncalibrated-fallback"},
        "tile_native": {**tile_native.tile_summary(tile),
                        "token_grouping": "activation block groups = consecutive rows of the flattened "
                                          "row-major [B*T, K] operand; the last group may be partial; padding "
                                          "rows are zeros and encode to +0 (docs/algos/tile-aware.md)",
                        "padding_encodes_to_plus_zero": tile_native.padding_encodes_to_plus_zero()},
        "compiler_pack_scale_admission": tile_native.pack_admission_report(static_act, weight_ranges),
        "pending_hardware_rulings": PENDING_RULINGS,
    }
    # REPORTING ONLY (docs/algos/pack-plan.md): per static activation site the pack code, in-range flag, fold k and cost.
    # The producer class needs the FX graph (capture_fx fills it into fx/policy.json); here it is 'unknown' except attention p.
    pack_plan.attach(policy, pack_plan.build_plan(policy))
    return policy


# ---------------------------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------------------------

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def digest_batches(obj) -> str | None:
    """sha256 over calibration batches: tensors and numpy arrays by dtype, shape and bytes, nested
    dicts (sorted keys) / lists / tuples by structure, scalars and strings by repr. None if some
    leaf has none of these forms."""
    h = hashlib.sha256()

    def feed(o):
        if torch.is_tensor(o):
            t = o.detach().cpu().contiguous()
            h.update(f"T{t.dtype}{tuple(t.shape)}".encode())
            h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
            return True
        np = sys.modules.get("numpy")
        if np is not None and isinstance(o, np.ndarray):
            h.update(f"N{o.dtype}{o.shape}".encode())
            h.update(np.ascontiguousarray(o).tobytes())
            return True
        if isinstance(o, dict):
            h.update(b"{")
            for k in sorted(o, key=str):
                h.update(repr(k).encode())
                if not feed(o[k]):
                    return False
            h.update(b"}")
            return True
        if isinstance(o, (list, tuple)):
            h.update(b"[")
            ok = all(feed(x) for x in o)
            h.update(b"]")
            return ok
        if o is None or isinstance(o, (bool, int, float, str, bytes)):
            h.update(repr(o).encode())
            return True
        return False

    return h.hexdigest() if feed(obj) else None


def git_state() -> dict:
    here = Path(__file__).resolve().parent

    def sh(*cmd):
        try:
            return subprocess.check_output(cmd, text=True, cwd=here, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    st = sh("git", "status", "--porcelain")
    return {"atlas_torchao_commit": sh("git", "rev-parse", "HEAD"),
            "atlas_torchao_branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "atlas_torchao_dirty": None if st is None else bool(st)}


def versions() -> dict:
    out = {"torch": torch.__version__, "python": platform.python_version()}
    for mod in ("torchao", "transformers", "safetensors", "lerobot", "numpy"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", None)
        except Exception:
            out[mod] = None
    return out


def checkpoint_record(path_or_id, hash_weights=True) -> dict:
    """Where the BF16 checkpoint came from, with sha256 of its *.safetensors when it is a local
    directory or a cached HF snapshot (never downloads)."""
    rec = {"id": str(path_or_id)}
    d = Path(str(path_or_id))
    if not d.is_dir():
        try:
            from huggingface_hub import snapshot_download

            d = Path(snapshot_download(str(path_or_id), local_files_only=True))
            rec["snapshot"] = str(d)
        except Exception as e:
            rec["files"] = None
            rec["note"] = f"no local snapshot to hash ({type(e).__name__})"
            return rec
    files = sorted(d.glob("*.safetensors"))
    rec["files"] = {f.name: {"bytes": f.stat().st_size,
                             "sha256": sha256_file(f) if hash_weights else None} for f in files}
    for name in ("config.json",):
        if (d / name).is_file():
            rec[name.replace(".", "_") + "_sha256"] = sha256_file(d / name)
    return rec


def jsonable(o, depth=0):
    """A JSON-safe copy (tensors: values if small, else a summary; Counters and sets as dicts / lists)."""
    if depth > 30:
        return repr(o)
    if torch.is_tensor(o):
        t = o.detach().cpu()
        if t.numel() <= 64:
            return t.tolist()
        return {"tensor": str(t.dtype), "shape": list(t.shape)}
    if isinstance(o, dict):
        return {str(k): jsonable(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v, depth + 1) for v in o]
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else repr(o)
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if hasattr(o, "summary"):
        try:
            return jsonable(o.summary(), depth + 1)
        except Exception:
            pass
    return repr(o)


def _opt_out_reasons(algo, bf16) -> dict:
    out = {}
    for m in _members(algo):
        sel = getattr(m, "selected", None) or []
        for n in bf16:
            if n in sel and n not in out:
                s = getattr(m, "selection", {}) or {}
                out[n] = (f"{m.name} keep_as={s.get('keep_as', 'bf16')} (score {s.get('score')}, k={s.get('k')}); "
                          "no MXU path for BF16 operands (pending ruling R4)")
    for n in bf16:
        out.setdefault(n, "algorithm BF16 opt-out (bf16_layers)")
    return out


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

def export(model: nn.Module, export_dir, *, mode: str, report: dict, algo, algo_spec: str, options=None,
           skip=(), model_record: dict | None = None, calib_batches=None, calib_record=None,
           device=None, extra_provenance: dict | None = None, normalization: dict | None = None,
           allow_uncalibrated_fallback: bool = False) -> dict:
    """Write model.safetensors, policy.json and manifest.json for `model`, quantized under `mode`
    by algos.quantize (whose return value is `report`). Returns the manifest.

    normalization (NEW; LeRobot policies, src/policy_norm.export_record): the input / output
    normalization protocol, its stats sha256 and its transform (per-key mode and eps) sha256,
    written to both files; --from-export refuses an export whose protocol or transform differs
    from the requested one.

    allow_uncalibrated_fallback: declare (policy.json uncalibrated_fallback.declared) that Linear sites of a
    static recipe without calibration data may stay on the dynamic reference rule; without it the export is
    still written but warns, and tools/verify_bundle.py fails the static check."""
    from safetensors.torch import save_file

    import quant_modes

    if quant_modes.PROFILES[quant_modes.canonical(mode)].base not in EXPORTABLE_BASES:
        raise ValueError(f"export: mode {mode!r} is not a reference-numerics mode")
    out = Path(export_dir)
    out.mkdir(parents=True, exist_ok=True)
    algo_rec = report.get("algo") or {}
    bf16 = list(algo_rec.get("bf16_opt_out") or [])
    reasons = _opt_out_reasons(algo, bf16)
    exact_bf16 = list(algo_rec.get("bf16_opt_out_exact") or [])
    # the reason of an exact site is the one its selecting rule gave (algos.base.Algo.bf16_exact)
    exact_reasons = algo_rec.get("bf16_opt_out_reasons") or {}
    for n in exact_bf16:
        reasons[n] = exact_reasons.get(n, "algorithm exact BF16 opt-out (bf16_exact)")
    quant = {}
    policy = build_policy(model, mode, report, algo, options, list(skip or ()), bf16, reasons, quant,
                          allow_uncalibrated_fallback=allow_uncalibrated_fallback)
    fb = policy["uncalibrated_fallback"]
    if fb["count"]:
        status = "declared" if fb["declared"] else "NOT declared; pass --allow-uncalibrated-fallback to declare"
        print(f"export: WARNING {fb['count']} Linear / Conv2d site(s) of a static recipe have no calibration "
              f"data and stay on the dynamic per-tensor rule ({status}): {fb['sites'][:5]}", flush=True)
    # NEW: exact site identity avoids excluding similarly named sibling layers on replay.
    for row in policy["bf16_opt_out"]:
        if row["site"] in exact_bf16:
            row["match"] = "exact"

    qmods = quantized_modules(model)
    skip_keys = {f"{n}.weight" for n in qmods}
    tensors, aliases, seen = {}, {}, {}
    for k, v in model.state_dict(keep_vars=False).items():
        if k in skip_keys:
            continue
        t = v.detach()
        key = (t.data_ptr(), t.dtype, tuple(t.shape), tuple(t.stride())) if t.numel() else None
        if key is not None and key in seen:
            aliases[k] = seen[key]
            continue
        if key is not None:
            seen[key] = k
        tensors[k] = t.cpu().contiguous().clone()
    clash = set(tensors) & set(quant)
    if clash:
        raise RuntimeError(f"export: state-dict keys clash with quantized tensor names: {sorted(clash)[:5]}")
    tensors.update(quant)
    meta = {"schema": POLICY_SCHEMA, "format_version": str(FORMAT_VERSION), "exported_mode": mode}
    sf = out / "model.safetensors"
    save_file(tensors, str(sf), metadata=meta)
    policy["aliases"] = aliases
    if normalization is not None:
        policy["normalization"] = jsonable(normalization)
    policy["tensors"] = {"count": len(tensors), "quantized_layers": len(qmods),
                         "float_weights_of_quantized_layers": 0}
    pf = out / "policy.json"
    pf.write_text(json.dumps(policy, indent=2, sort_keys=False, allow_nan=False) + "\n")

    stack = None
    try:
        import algos as _algos

        stack = [{"name": n, "params": p} for n, p in _algos.parse_stack(algo_spec)]
        expand = getattr(_algos, "expand", None)
        expanded = expand(algo_spec) if expand else algo_spec
    except Exception as e:                                   # never blocks an export
        expanded = f"unparsed ({e})"
    calib = None
    if calib_record is not None or calib_batches is not None:
        calib = {**(jsonable(calib_record) or {}),
                 "data_sha256": digest_batches(calib_batches) if calib_batches is not None else None}
    site_list = [{"site_id": n, "status": s["status"], "kind": s["kind"]} for n, s in policy["sites"].items()]
    recipe_rec = getattr(algo, "recipe", None) or {}
    manifest = {
        "schema": MANIFEST_SCHEMA, "format_version": FORMAT_VERSION,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": {"model.safetensors": {"sha256": sha256_file(sf), "bytes": sf.stat().st_size},
                  "policy.json": {"sha256": sha256_file(pf), "bytes": pf.stat().st_size}},
        "policy_sha256": sha256_file(pf),
        "model": model_record or {},
        "exported_mode": mode, "ladder": jsonable(report.get("ladder")),
        "recipe": {"algo": algo_spec, "alias": recipe_rec.get("alias"), "resolved": recipe_rec.get("name"),
                   "expanded": expanded, "stack": jsonable(stack),
                   "recipe": jsonable(getattr(algo, "recipe", None)),
                   "algorithm_record": jsonable(algo_rec)},
        "calibration": calib,
        **({"normalization": jsonable(normalization)} if normalization is not None else {}),
        "provenance": {**git_state(), "versions": versions(), "host": platform.node(),
                       "device": str(device) if device is not None else None,
                       "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                       "argv": list(sys.argv), **(jsonable(extra_provenance) or {})},
        "sites": site_list,
        "counts": {s: sum(1 for x in site_list if x["status"] == s) for s in ("quantized", "host", "skipped")},
        "new_decisions": NEW_DECISIONS + list((getattr(algo, "recipe", None) or {}).get("new_decisions", [])),
        "pending_hardware_rulings": PENDING_RULINGS,
        "m2m_manifest_v2_aligned": "partial",
        "m2m_manifest_v2_matching_fields": M2M_V2_MATCHING,
        "m2m_manifest_v2_differences": M2M_V2_DIFFERENCES,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(f"export: wrote {out} ({manifest['files']['model.safetensors']['bytes'] / 2**20:.1f} MiB, "
          f"{manifest['counts']['quantized']} quantized sites)", flush=True)
    if getattr(options, "pack_fold", False):
        manifest = _pack_fold_in_place(out, model, policy, getattr(options, "pack_fold_graph", None))
    return manifest


def _pack_fold_in_place(out: Path, model, policy: dict, graph_dir) -> dict:
    """--pack-fold: replace the export in `out` by its folded twin (src/pack_fold.py); the unfolded export stays in
    <out>.unfolded. The producer classes come from an FX capture of the unfolded export."""
    import os

    import pack_fold

    if graph_dir:
        graph, sig = (Path(graph_dir) / "graph.txt").read_text(), (Path(graph_dir) / "graph_signature.txt").read_text()
    else:
        graph, sig = pack_fold.capture_graph_text(model, policy)
    unfolded = out.with_name(out.name + ".unfolded")
    if unfolded.exists():
        raise FileExistsError(f"--pack-fold: {unfolded} exists")
    os.replace(out, unfolded)
    try:
        summary = pack_fold.fold_export_dir(unfolded, out, graph, sig)
    except BaseException:
        if out.exists():
            import shutil
            shutil.rmtree(out)
        os.replace(unfolded, out)
        raise
    print(f"export: pack fold {summary['outside_before']} -> {summary['outside_after']} static sites outside "
          f"{summary['admitted_range']} ({summary['sites_folded']} folded); unfolded export kept in {unfolded}", flush=True)
    return json.loads((out / "manifest.json").read_text())
