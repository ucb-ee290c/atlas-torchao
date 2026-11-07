"""Rebuild a quantized model from an export (NEW; docs/algos/export.md).

    model, policy, manifest = load_quantized(model_id_or_policy, export_dir, device)

No calibration runs and no algorithm is re-applied: every quantized layer gets its E4M3 codes and
exponents from model.safetensors, every other tensor (the folded norms, producer rows, biases,
embeddings, BF16 opt-outs) is assigned from the same file in its stored dtype, and the activation
scale of every site is executed from policy.json by `PolicyAlgo`, an algos.base.Algo whose hooks
read the policy instead of calibration statistics. The reference modules (src/ref_profile.py,
src/ref_attention.py) then run exactly the code they run after algos.quantize, so the loaded model
is bit-identical to the in-memory quantized one (tests/test_export.py).

Architecture: a causal LM (eval.py exports) is built from its config alone, with no checkpoint
weights read. A LeRobot policy or Pi0 has no weight-free constructor here, so their loaders build
the module tree from the checkpoint (`build=`); every tensor the export covers is then overwritten
and any state-dict entry the export does not cover raises, so no original value survives.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

import algos
import quant_modes
import tile_native
from algos.base import Algo, Scale


class PolicyAlgo(Algo):
    """The module-level hooks of an exported policy: weights from the export, activation scales
    from the policy's per-site rules."""

    name = "export_policy"

    def __init__(self, policy: dict, tensors: dict, **params):
        super().__init__(**params)
        self.policy = policy
        self.tensors = tensors
        self.sites = policy["sites"]
        self.calls = {}

    def _site(self, name):
        s = self.sites.get(name)
        if s is None or s.get("status") != "quantized":
            raise KeyError(f"export policy has no quantized site {name!r}")
        return s

    # weights: the placeholder weight already is dequant(codes); the exponent is the exported one
    def transform_weight(self, name, w, kind):
        return w

    def weight_scale(self, name, w, kind):
        s = self._site(name)["weight"]
        exp = self.tensors[s["scale_exp"]].to(w.device)
        return Scale(exp, s["granularity"], block=int(s.get("block", 0)), dynamic=False)

    def pv_pop_shift(self, name):
        """Folded pack exponent of the attention output (docs/algos/pack-fold.md); 0 when the export has no fold."""
        return int((self.sites.get(name) or {}).get("pv_pop_shift", 0))

    def residual_passes(self, name, kind):
        r = self._site(name).get("residual_passes")
        return r["kind"] if r else None

    def act_scale(self, name, x, kind, dims=None):
        site = self._site(name)
        rule = site["operands"][kind.split(".", 1)[1]] if kind.startswith("attention.") else site["activation"]
        if (not kind.startswith("attention.") and str(site["weight"].get("granularity", "")).startswith("per_panel")
                and rule.get("reduce") not in ("token", "block32_or_token")):
            from algos import granularity as gr      # same chip costing as the in-memory w=panel run
            gr._count_tile_output(x, site["weight"]["scale_exp_shape"][0])
        return rule_scale(rule, x, kind, dims, self, name)


def rule_scale(rule, x, kind, dims, holder=None, name=""):
    """The Scale a policy rule gives for operand `x` (None: the reference dynamic rule)."""
    if rule.get("source") == "reference":
        return None
    mode = rule["mode"]
    if mode == "static":
        return Scale(torch.tensor(int(rule["exp"]), dtype=torch.int32, device=x.device), rule["label"],
                     dynamic=False)
    if mode == "static_per_step":
        steps = rule["exps"]
        k = holder.calls.get(name, 0)
        holder.calls[name] = k + 1
        return Scale(torch.tensor(int(steps[k % len(steps)]), dtype=torch.int32, device=x.device),
                     rule["label"], dynamic=False)
    if mode != "dynamic":
        raise ValueError(f"unknown activation mode {mode!r}")
    from algos import granularity as gr

    red = rule["reduce"]
    if red == "tensor":
        return Scale(gr.scale_exp(x, None, rule["rule"]), rule["label"])
    if red == "reference_dims":
        return Scale(gr.scale_exp(x, dims, rule["rule"]), rule["label"])
    if red == "block":                       # granularity a=block: tokens = flattened leading dims
        k = x.shape[-1]
        m = x.numel() // max(1, k)
        gr._count_block_cast(m, k)
        s = gr.row_group_scale_exp(x.reshape(m, k), int(rule.get("rows", 32)), rule["rule"])
        return Scale(s.reshape(*x.shape[:-1], 1), rule["label"])
    if red == "head_block":                  # granularity attn=block: row groups per (batch, head) matrix
        gr._count_block_cast(x.numel() // max(1, x.shape[-1]), x.shape[-1])
        return Scale(gr.row_group_scale_exp(x, int(rule.get("rows", 32)), rule["rule"]), rule["label"])
    if red == "block32_or_token" and x.shape[-1] % int(rule.get("block", 32)) == 0:
        return Scale(gr.block_scale_exp(x, rule["rule"], int(rule.get("block", 32))), rule["label"],
                     block=int(rule.get("block", 32)))
    if red in ("token", "block32_or_token"):
        gr._count_token_cast(x, int(rule.get("out_cols", 0)))
        label = rule["label"] if red == "token" else rule["tail_label"]
        return Scale(gr.scale_exp(x, (-1,), rule["rule"]), label)
    raise ValueError(f"unknown activation reduction {red!r}")


# ---------------------------------------------------------------------------------------------
# Reading an export
# ---------------------------------------------------------------------------------------------

def read_export(export_dir, verify=True):
    """(tensors, policy, manifest); with verify, the files' sha256 must match the manifest."""
    from safetensors.torch import load_file

    import quant_export

    d = Path(export_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    if verify:
        for fname, rec in manifest["files"].items():
            got = quant_export.sha256_file(d / fname)
            if got != rec["sha256"]:
                raise ValueError(f"{d / fname}: sha256 {got} differs from the manifest's {rec['sha256']}")
    policy = json.loads((d / "policy.json").read_text())
    if policy.get("schema") != quant_export.POLICY_SCHEMA:
        raise ValueError(f"{d}: policy schema {policy.get('schema')!r}, expected {quant_export.POLICY_SCHEMA}")
    return load_file(str(d / "model.safetensors")), policy, manifest


def build_causal_lm(model_id, seqlen=2048, device="cpu"):
    """The architecture of modelutils.load_llama from its config alone (no checkpoint weights):
    same rope scaling, eager attention, BF16, and non-persistent buffers made persistent."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = False
    ctx = getattr(config, "max_position_embeddings", None)
    if ctx and seqlen > ctx:
        config.rope_scaling = {"type": "linear", "factor": float(math.ceil(seqlen / ctx))}
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16, attn_implementation="eager")
    for mod in model.modules():
        mod._non_persistent_buffers_set.clear()
    return model.to(device).eval()


def _get(model, path):
    obj = model
    for p in path.split("."):
        obj = getattr(obj, p)
    return obj


def _assign(model, key, t):
    mod_path, _, attr = key.rpartition(".")
    mod = _get(model, mod_path) if mod_path else model
    cur = getattr(mod, attr)
    if isinstance(cur, nn.Parameter):
        cur.data = t
    elif torch.is_tensor(cur):
        if attr in mod._buffers:
            mod._buffers[attr] = t
        else:
            setattr(mod, attr, t)
    else:
        raise KeyError(f"{key}: not a tensor of the model")


def apply_export(model, tensors, policy, device, mode=None, options=None):
    """Quantize `model` (fresh architecture) from an export in place; returns the quantize report."""
    from algos import base as algo_hooks

    mode = mode or policy["exported_mode"]
    base = quant_modes.PROFILES[quant_modes.canonical(mode)].base
    if base not in ("reference", "reference_hw", "chip"):
        raise ValueError(f"load_quantized: mode {mode!r} is not a reference-numerics mode")
    need = set(quant_modes.profile_options(quant_modes.resolved_options(mode, options or {}))["ops"])
    have = set(policy["ops"])
    if not need <= have:
        raise ValueError(f"load_quantized: mode {mode!r} quantizes {sorted(need)}; the export "
                         f"({policy['exported_mode']}) covers {sorted(have)}")
    sites = policy["sites"]
    tile_native.verify_export(policy, tensors)      # fail loudly: panel-labeled exponents, reserved codes
    quant = {n for n, s in sites.items() if s["status"] == "quantized" and s["kind"] in ("linear", "conv2d")}
    qkeys = {f"{n}.weight" for n in quant}
    aliases = policy.get("aliases", {})
    state_keys = set(model.state_dict().keys())
    named = {k for k in tensors if not k.endswith(tuple(__import__("quant_export").QUANT_SUFFIXES))}
    missing = state_keys - qkeys - named - set(aliases)
    if missing:
        raise KeyError(f"load_quantized: state-dict entries not in the export: {sorted(missing)[:8]}")
    extra = named - state_keys
    if extra:
        raise KeyError(f"load_quantized: export tensors the model does not have: {sorted(extra)[:8]}")
    with torch.no_grad():
        for k in sorted(named):
            _assign(model, k, tensors[k].to(device))
        for k, src in aliases.items():
            if _get(model, k) is not _get(model, src):
                _assign(model, k, tensors[src].to(device))
        mods = dict(model.named_modules())
        for n in quant:
            s = sites[n]
            codes = tensors[s["weight"]["codes"]]
            exp = tensors[s["weight"]["scale_exp"]]
            blk = int(s["weight"].get("block", 0))
            w = algo_hooks.dequant(codes, exp, blk) if blk else __import__("ref_numerics").dequantize(codes, exp)
            dtype = getattr(torch, s["weight_dtype"])
            mods[n].weight.data = w.reshape(s["weight_shape"]).to(device=device, dtype=dtype)
    skip = list(policy.get("skip", []))
    bf16 = [r["site"] for r in policy.get("bf16_opt_out", [])]
    exact_bf16 = [r["site"] for r in policy.get("bf16_opt_out", []) if r.get("match") == "exact"]
    opts = dict(quant_modes._options_dict(options))
    if base == "chip":
        for k, v in (policy.get("runtime", {}).get("chip_options") or {}).items():
            if opts.get(k) is None:
                opts[k] = v
    if bf16:
        opts["_algo_bf16"] = tuple(bf16)
        opts["_algo_bf16_exact"] = tuple(exact_bf16)
    replay = PolicyAlgo(policy, {k: v.to(device) for k, v in tensors.items()
                                 if k.endswith(tuple(__import__("quant_export").QUANT_SUFFIXES))})
    with algos.active(replay):
        rep = quant_modes.quantize(model, mode, skip + [n for n in bf16 if n not in exact_bf16], opts)
    # the codes the modules derived must be the exported ones, bit for bit
    with torch.no_grad():
        for n, m in model.named_modules():
            if type(m).__name__ not in ("RefQuantLinear", "RefQuantConv2d"):
                continue
            if n not in quant:
                raise RuntimeError(f"load_quantized: {n} was quantized but the export has no codes for it")
            s = sites[n]
            if not torch.equal(m.weight_codes.cpu(), tensors[s["weight"]["codes"]]):
                raise RuntimeError(f"load_quantized: {n}: re-derived codes differ from the export")
            r = s.get("residual_passes")
            if r:
                lo = tensors[r["lo_codes"]].to(device)
                m.weight_lo_codes = lo
                m.weight_lo_scale_exp = tensors[r["lo_scale_exp"]].to(device)
                if getattr(m, "fm", None) is not None:
                    m.weight_lo_norm = __import__("ref_numerics").e4m3_unpack(lo)
    rep["loaded_from_export"] = True
    rep["algo"] = {"name": "export_policy", "applied": True, "bf16_opt_out": bf16,
                   "bf16_opt_out_exact": exact_bf16,
                   "granularity": replay.stats}
    return rep


def load_quantized(model_id_or_policy, export_dir, device="cpu", *, build=None, mode=None, options=None,
                   seqlen=2048, verify=True):
    """(model, report, manifest): the reference-module model rebuilt from `export_dir`.

    build(): returns the module tree to fill (default: a causal LM from the config of
    `model_id_or_policy`, or of the manifest's model id when that is None). mode: any reference
    mode the export's ops cover (default: the exported one). options: profile options as for
    quant_modes.quantize (e.g. {"chip_mxu": "framework"})."""
    tensors, policy, manifest = read_export(export_dir, verify=verify)
    if build is None:
        mid = model_id_or_policy or (manifest.get("model") or {}).get("id")
        kind = (manifest.get("model") or {}).get("kind", "causal_lm")
        if kind != "causal_lm":
            raise ValueError(f"load_quantized: a {kind} export needs build= (its CLI's loader)")
        model = build_causal_lm(mid, seqlen=(manifest.get("model") or {}).get("seqlen") or seqlen, device=device)
    else:
        model = build()
    rep = apply_export(model, tensors, policy, device, mode=mode, options=options)
    return model, rep, manifest
