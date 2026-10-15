"""Per-layer (isolated) and cumulative relative RMSE of a quantized model against a clean pass.

This is the reference author's measurement, reused rather than reimplemented:

  - The statistics are the reference `StatsTracker` (smolVLA-quant smolvla_inout/stats_tracker.py:181-237,
    `record` at :220-278), loaded by file path. For every quantized layer call it is given
      y_fp        the clean layer applied to the input that actually reached the quantized layer
                  (local / isolated error; quant_linear.py:189-197 computes F.linear(x, w, b) with
                  the original weights on the same, possibly already-corrupted, x),
      y_quant     the quantized layer's output, and
      y_clean_ref the same layer's output in an unquantized forward on the same input
                  (cumulative / end-to-end error; quant_linear.py:178-180).
  - The clean outputs are captured with the reference `ReferenceStore` forward hooks
    (reference_store.py:45-66), the way run_eval.py:530-546 does: hooks on the clean model and
    one forward per input under the same seed; the patched pass then reads the outputs back in
    call order (reference_store.py:66-78).
  - The CSVs are run_eval.py's (:769-778): chronological.csv, grouped.csv, summary.csv and
    worst_layers.csv, with the field lists of run_eval.py:191-203. Only the matrix path exists
    here, so every row is tagged "mx".

Differences from the reference, all NEW:
  - The reference patches its own QuantLinear and records inside its forward. Here quantization
    is whatever the selected profile installs, so the measurement is a forward hook on each module
    the profile changed, and the local reference is the corresponding module of a second, clean
    copy of the model (applied to the quantized layer's input cast to the clean weight dtype, as
    quant_linear.py:136-137 casts x to w.dtype).
  - The reference captures the outputs for all batches into one store before the patched run
    (run_eval.py:538-544). Here the clean pass runs per input (window / frame) just before the
    quantized one into a fresh store, which yields the same call-order pairing with memory
    bounded by one input.

The reference repositories cannot be pip-installed next to this one (their numpy<2 and
transformers 4.53 pins conflict with ours), so single files are loaded by path from
  $SMOLVLA_QUANT_ROOT / $PI0_QUANT_ROOT, or $ATLAS_REF_ROOT/<repo>, or third_party/<repo>.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
REF_REPOS = {  # repo -> (env var, package dir)
    "smolVLA-quant": ("SMOLVLA_QUANT_ROOT", "smolvla_inout"),
    "pi0-quant": ("PI0_QUANT_ROOT", "pi0_inout"),
}
_loaded: dict[str, object] = {}
_provenance: dict[str, dict] = {}


# --------------------------------------------------------------------------------------------
# Reference file loader (NEW)
# --------------------------------------------------------------------------------------------

def ref_root(repo: str) -> Path | None:
    """Where the reference repository `repo` is checked out, or None."""
    env, pkg = REF_REPOS[repo]
    candidates = [os.environ.get(env)]
    if os.environ.get("ATLAS_REF_ROOT"):
        candidates.append(str(Path(os.environ["ATLAS_REF_ROOT"]) / repo))
    candidates.append(str(REPO_ROOT / "third_party" / repo))
    for c in candidates:
        if c and (Path(c) / pkg).is_dir():
            return Path(c)
    return None


def _git_head(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_ref_file(name: str, repos=("smolVLA-quant", "pi0-quant")):
    """Load `<pkg>/<name>.py` from the first available reference repo as a standalone module.

    Only files whose imports are stdlib/torch can be loaded this way (stats_tracker.py,
    reference_store.py); the package __init__ imports the whole model patcher.
    """
    if name in _loaded:
        return _loaded[name]
    for repo in repos:
        root = ref_root(repo)
        if root is None:
            continue
        path = root / REF_REPOS[repo][1] / f"{name}.py"
        if not path.is_file():
            continue
        mod_name = f"_atlas_ref_{REF_REPOS[repo][1]}_{name}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod  # dataclasses resolve annotations through sys.modules
        spec.loader.exec_module(mod)
        _loaded[name] = mod
        rec = _provenance.setdefault(repo, {"root": str(root), "commit": _git_head(root), "files": {}})
        rec["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return mod
    raise RuntimeError(
        f"reference file {name}.py not found: set SMOLVLA_QUANT_ROOT / PI0_QUANT_ROOT (or "
        f"ATLAS_REF_ROOT), or check out third_party/smolVLA-quant (git submodule update --init)")


def ref_provenance() -> dict:
    """{repo: {root, commit, files: {path: sha256}}} for every reference file loaded so far."""
    return {k: {**v, "files": dict(v["files"])} for k, v in _provenance.items()}


# --------------------------------------------------------------------------------------------
# Component taxonomy
# --------------------------------------------------------------------------------------------

# smolVLA-quant smolvla_inout/model_patcher.py:80-95 (first match wins), as names of the reference
# Component enum (stats_tracker.py:38-43). model_patcher.py cannot be loaded by path (relative
# imports), so the table is copied.
_REF_COMPONENT_RULES = [
    ("action_in_proj", "ACTION_HEAD"), ("action_out_proj", "ACTION_HEAD"),
    ("action_time_mlp_in", "ACTION_HEAD"), ("action_time_mlp_out", "ACTION_HEAD"),
    ("state_proj", "ACTION_HEAD"), ("lm_expert", "ACTION_EXPERT"), ("vision_model", "VISION"),
    ("text_model", "LANGUAGE"), ("vlm_with_expert", "LANGUAGE"),
]
# NEW: Multi-Task DiT has no reference taxonomy. Its CLIP towers already match vision_model /
# text_model above; the text projection is language and the denoiser plays the action expert.
_DIT_COMPONENT_RULES = [
    ("observation_encoder.text_encoder", "LANGUAGE"), ("observation_encoder.vision_encoder", "VISION"),
    ("noise_predictor", "ACTION_EXPERT"),
]


# Pi0 (openpi PI0Pytorch): pi0-quant pi0_inout/model_patcher.py:92-109, the reference's own table
# for this model. Its paths never match the rules above except `vision_model` inside
# `vision_tower`, which both tables call VISION.
_PI0_COMPONENT_RULES = [
    ("action_in_proj", "ACTION_HEAD"), ("action_out_proj", "ACTION_HEAD"),
    ("action_time_mlp_in", "ACTION_HEAD"), ("action_time_mlp_out", "ACTION_HEAD"),
    ("state_proj", "ACTION_HEAD"), ("time_mlp_in", "ACTION_HEAD"), ("time_mlp_out", "ACTION_HEAD"),
    ("gemma_expert", "ACTION_EXPERT"), ("vision_tower", "VISION"), ("language_model", "LANGUAGE"),
    ("paligemma", "LANGUAGE"),
]


def component_for(path: str, default: str = "UNKNOWN"):
    """The reference Component of a module path (model_patcher.py:98-113, plus the DiT and Pi0
    rules)."""
    Component = load_ref_file("stats_tracker").Component
    rules = _PI0_COMPONENT_RULES if "paligemma_with_expert" in path else (
        *_REF_COMPONENT_RULES, *_DIT_COMPONENT_RULES)
    for substr, comp in rules:
        if substr in path:
            return Component[comp]
    return Component[default]


# --------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------

MEASURED_TYPES = (nn.Linear, nn.Conv2d)  # run_eval.py:532-535 hooks exactly these types


def quantized_fqns(clean: nn.Module, quant: nn.Module) -> list[str]:
    """Module paths that are a plain nn.Linear / nn.Conv2d in `clean` and were changed in `quant`
    (replaced, or given a tensor-subclass weight as torchao does). NEW: the profile decides what is
    quantized; this recovers it without the profile's cooperation."""
    qmods = dict(quant.named_modules())
    out = []
    for fqn, m in clean.named_modules():
        if type(m) not in MEASURED_TYPES or fqn not in qmods:
            continue
        q = qmods[fqn]
        if type(q) is not type(m) or type(getattr(q, "weight", None)) not in (torch.Tensor, nn.Parameter):
            out.append(fqn)
    return out


class LayerRMSE:
    """Hooks a quantized model so each changed layer's call is recorded against a clean model.

        rm = LayerRMSE(clean_model, quant_model, default_component="LANGUAGE")
        for inp in inputs:
            rm.capture(lambda: clean_model(inp))   # clean outputs for this input
            quant_model(inp)                        # recorded by the hooks
        rm.close()
        rm.write_csvs(out_dir)
    """

    def __init__(self, clean: nn.Module, quant: nn.Module, fqns=None, default_component="UNKNOWN"):
        st = load_ref_file("stats_tracker")
        self._RefStore = load_ref_file("reference_store").ReferenceStore
        self.tracker = st.StatsTracker()
        self.fqns = list(fqns) if fqns is not None else quantized_fqns(clean, quant)
        clean_mods = dict(clean.named_modules())
        self._clean = clean
        self.store = None
        self._handles = []
        for fqn, mod in quant.named_modules():
            if fqn in self.fqns:
                comp = component_for(fqn, default_component)
                self._handles.append(mod.register_forward_hook(self._hook(fqn, comp, clean_mods[fqn])))

    def _hook(self, fqn, comp, clean_mod):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            x = inp[0]
            w = clean_mod.weight
            with torch.no_grad():
                # local reference: the clean layer on the input the quantized layer saw, cast to
                # the weight dtype (quant_linear.py:136-137, 189-197)
                y_fp = clean_mod(x.to(dtype=w.dtype, device=w.device))
                y_clean = self.store.get(fqn) if self.store is not None else None  # :176-180
                self.tracker.record(name=fqn, component=comp, y_fp=y_fp, y_quant=out,
                                    y_clean_ref=y_clean)
        return hook

    def capture(self, run_clean) -> None:
        """Run the clean forward(s) for the next input and keep its layer outputs (call order)."""
        store = self._RefStore()
        handles = store.register_hooks(self._clean, set(self.fqns))  # reference_store.py:45-64
        try:
            with torch.no_grad():
                run_clean()
        finally:
            for h in handles:
                h.remove()
        store.reset_counters()  # reference_store.py:90-92 (fresh store: lookups start at call 0)
        self.store = store

    def close(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        self.store = None

    def summary(self) -> dict:
        rows = self.tracker.component_rows()
        return {"layers_measured": len(self.fqns), "calls": len(self.tracker.calls),
                "components": rows, "worst_layers": worst_layer_rows(self.tracker, 5)}

    def write_csvs(self, out_dir) -> dict:
        return write_csvs(out_dir, self.tracker)


# --------------------------------------------------------------------------------------------
# CSV output: smolVLA-quant experiments/run_eval.py:191-283, matrix path only
# --------------------------------------------------------------------------------------------

CHRON_FIELDS = ["seq", "tag", "layer_name", "component", "rmse", "ref_rms", "rel_rmse",
                "cumulative_rmse", "cumulative_rel_rmse"]                         # :191-195
SUMMARY_FIELDS = ["tag", "component", "n_layers", "mean_rmse", "std_rmse", "max_rmse", "min_rmse",
                  "mean_rel_rmse", "std_rel_rmse", "max_rel_rmse", "max_rel_rmse_layer",
                  "total_calls", "mean_cumulative_rmse", "mean_cumulative_rel_rmse"]  # :196-202
WORST_LAYERS_FIELDS = ["rank", "tag", "component", "layer_name", "rel_rmse", "rmse", "n_calls"]  # :203


def _rel_rmse(rmse: float, ref_rms: float) -> float:  # :206-209
    if ref_rms > 0 and math.isfinite(rmse):
        return rmse / ref_rms
    return float("nan")


def calls_to_rows(calls, tag="mx") -> list[dict]:  # :212-228
    rows = []
    for rec in calls:
        cum_rmse = rec.get("cumulative_rmse", float("nan"))
        cum_ref = rec.get("cumulative_ref_rms", float("nan"))
        rows.append({"seq": rec["seq"], "tag": tag, "layer_name": rec["name"],
                     "component": rec["component"], "rmse": rec["rmse"], "ref_rms": rec["ref_rms"],
                     "rel_rmse": _rel_rmse(rec["rmse"], rec["ref_rms"]),
                     "cumulative_rmse": cum_rmse,
                     "cumulative_rel_rmse": _rel_rmse(cum_rmse, cum_ref)})
    return rows


def worst_layer_rows(tracker, top_n=20, tag="mx") -> list[dict]:  # :249-269
    rows = [{"tag": tag, **r} for r in tracker.layer_rows() if math.isfinite(r.get("rel_rmse", float("nan")))]
    rows.sort(key=lambda r: r["rel_rmse"], reverse=True)
    return [{"rank": k, "tag": r["tag"], "component": r["component"], "layer_name": r["layer"],
             "rel_rmse": r["rel_rmse"], "rmse": r["rmse"], "n_calls": r["n_calls"]}
            for k, r in enumerate(rows[:top_n], 1)]


def _write(path: Path, fields, rows) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_csvs(out_dir, tracker, tag="mx") -> dict:
    """chronological / grouped / summary / worst_layers CSVs (run_eval.py:769-778)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = calls_to_rows(tracker.calls, tag)
    paths = {k: out / f"{k}.csv" for k in ("chronological", "grouped", "summary", "worst_layers")}
    _write(paths["chronological"], CHRON_FIELDS, sorted(rows, key=lambda r: r["seq"]))       # :231-237
    _write(paths["grouped"], CHRON_FIELDS,
           sorted(rows, key=lambda r: (r["component"], r["layer_name"], r["tag"])))         # :240-246
    _write(paths["summary"], SUMMARY_FIELDS,
           [{"tag": tag, **r} for r in tracker.component_rows()])                            # :343-350
    _write(paths["worst_layers"], WORST_LAYERS_FIELDS, worst_layer_rows(tracker, 20, tag))   # :249-269
    return {k: str(v) for k, v in paths.items()}
