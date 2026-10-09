"""Pi0 (openpi's PyTorch PI0Pytorch) for the atlas-torchao profiles.

The reference author's pi0-quant is the Pi0 reference profile, so the model is loaded the way it
loads it (pi0-quant pi0_inout/serve_quant.py):

  - openpi's source tree is found through $OPENPI_DIR (serve_quant.py:55-62), which `runtime()`
    sets from --openpi-dir; openpi is not pip-installed. The tree is pinned (OPENPI_PIN) and its
    commit is recorded.
  - PI0Pytorch is built from the reference's known-config table and the converted safetensors
    checkpoint is loaded (serve_quant.py:93-149, 167-257); openpi's JAX-importing modules are
    replaced by the reference's stubs (_jax_stubs.py), injected by serve_quant at import.
  - The torch.compile wrapper that PI0Pytorch.__init__ puts on sample_actions is removed, as
    run_eval.py:943-947 does (pi0_pytorch.py:112-113), for every mode, the baseline included.
  - openpi's PyTorch Pi0 needs transformers 4.53.2 with openpi's transformers_replace files
    (openpi README "PyTorch Support"); `openpi_info()` records whether they are installed.

Observations (eval_pi0.py --obs):
  dummy  the reference's action-output methodology (experiments/run_output_eval.py): seeded random
         DROID-format observations (_make_dummy_obs, :73-81) through the reference policy shim
         Pi0PyTorchPolicy.infer (serve_quant.py:395-495: state normalized, actions unnormalized,
         joint deltas made absolute, first 8 dims kept). Loaded from the reference file itself.
  droid  NEW: real DROID frames with recorded actions from the public LeRobot dataset
         lerobot/droid_1.0.1 (joint-position state and actions, the action space of
         pi0_droid_jointpos_polaris; openpi polaris_config.py JOINT_POSITION). The reference's
         real-data loader (run_eval.py:183-298) reads lerobot/droid_100, whose state and actions are
         cartesian (meta/stats.json: observation.state is xyz + euler + gripper), so it does not
         match a joint-position policy; this loader follows its structure (seeded episode shuffle,
         one frame per episode, prompt tokenization through the same policy shim) on droid_1.0.1.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import ref_loader

OPENPI_PIN = "215abfb217dbac7d5f1273282331b9b1866c0479"   # github.com/Physical-Intelligence/openpi
OPENPI_URL = "https://github.com/Physical-Intelligence/openpi"
DEFAULT_CONFIG = "pi0_droid_jointpos_polaris"               # pi0-quant run_eval.py:855
# gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris (openpi polaris_config.py),
# converted with openpi examples/convert_jax_model_to_pytorch.py (default precision bfloat16).
CHECKPOINT_SOURCE = "gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris"
TOKENIZER_SOURCE = "gs://big_vision/paligemma_tokenizer.model"
DROID_REPO = "lerobot/droid_1.0.1"
DROID_REVISION = "0eabc778f959c54b8c5aa3626cc1128d2d2e54d4"
DROID_CAMERAS = {"observation/exterior_image_1_left": "observation.images.exterior_1_left",
                 "observation/wrist_image_left": "observation.images.wrist_left"}

# Linear layers the reference patches but Pi0 inference never calls (run_eval.py:20-22):
# patching them changes nothing and costs ~3 GB of float32 normalized weights under reference_hw.
DEFAULT_SKIP = ("lm_head",)


def _git(path, *args):
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def openpi_dir(path=None) -> Path:
    p = Path(path or os.environ.get("OPENPI_DIR") or "")
    if not (p / "src" / "openpi" / "models_pytorch" / "pi0_pytorch.py").is_file():
        raise SystemExit(f"openpi source tree not found at {str(p)!r}: pass --openpi-dir or set "
                         f"OPENPI_DIR (git clone {OPENPI_URL}; git checkout {OPENPI_PIN})")
    return p.resolve()


def openpi_info(path) -> dict:
    import transformers

    try:
        from transformers.models.siglip import check
        replaced = bool(check.check_whether_transformers_replace_is_installed_correctly())
    except ImportError:
        replaced = False
    commit = _git(path, "rev-parse", "HEAD")
    return {"dir": str(path), "commit": commit, "pinned": OPENPI_PIN, "at_pin": commit == OPENPI_PIN,
            "dirty": bool(_git(path, "status", "--porcelain")), "transformers": transformers.__version__,
            "transformers_replace_installed": replaced}


_RT = None


def runtime(path=None):
    """The reference loader module (pi0_inout.serve_quant) with openpi on sys.path.

    serve_quant reads $OPENPI_DIR at import (serve_quant.py:55) and inserts openpi's src, the
    openpi-client src and the pi0-quant root at the front of sys.path (60-62); they are moved to
    the end so our own modules keep precedence."""
    global _RT
    if _RT is not None:
        return _RT
    path = openpi_dir(path)
    os.environ["OPENPI_DIR"] = str(path)
    before = list(sys.path)
    _RT = ref_loader.import_ref("pi0_inout.serve_quant")
    added = [p for p in sys.path if p not in before]
    sys.path[:] = [p for p in sys.path if p not in added] + added
    return _RT


def sha256_file(path, cache=True) -> str:
    """sha256 of a (large) file; cached next to it as <file>.sha256 keyed by size and mtime."""
    path = Path(path)
    st = path.stat()
    side = path.with_name(path.name + ".sha256.json")
    if cache and side.is_file():
        try:
            rec = json.loads(side.read_text())
            if rec.get("size") == st.st_size and rec.get("mtime") == st.st_mtime:
                return rec["sha256"]
        except (ValueError, KeyError):
            pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache:
        try:
            side.write_text(json.dumps({"size": st.st_size, "mtime": st.st_mtime, "sha256": digest}))
        except OSError:
            pass
    return digest


def checkpoint_info(ckpt_dir, hash_weights=True) -> dict:
    d = Path(ckpt_dir)
    sf = d / "model.safetensors"
    info = {"dir": str(d.resolve()), "source": CHECKPOINT_SOURCE,
            "converter": "openpi examples/convert_jax_model_to_pytorch.py"}
    if (d / "config.json").is_file():
        info["config_json"] = json.loads((d / "config.json").read_text())
    if sf.is_file():
        info["model_safetensors_bytes"] = sf.stat().st_size
        info["model_safetensors_sha256"] = sha256_file(sf) if hash_weights else None
    return info


def load_model(rt, config: str, ckpt_dir: str, device) -> torch.nn.Module:
    """PI0Pytorch with the checkpoint, eager (compile wrapper removed), eval mode.

    serve_quant._load_checkpoint only warns when loading fails and keeps random weights
    (serve_quant.py:233-257); here a missing file or a mismatch with the file raises."""
    sf = Path(ckpt_dir) / "model.safetensors"
    if not sf.is_file():
        raise FileNotFoundError(f"{sf} not found (convert the JAX checkpoint first; see README)")
    model = rt.load_pi0_pytorch(config, str(ckpt_dir), torch.device(device))
    if "sample_actions" in model.__dict__:          # run_eval.py:943-947
        del model.sample_actions
    model.eval()
    from safetensors import safe_open

    with safe_open(str(sf), framework="pt") as f:
        key = "action_out_proj.weight"
        want = f.get_tensor(key)
    got = dict(model.named_parameters())[key].detach().cpu()
    if not torch.equal(got.to(want.dtype), want):
        raise RuntimeError(f"checkpoint {sf} did not load ({key} differs): random weights?")
    return model


def model_config(rt, config: str):
    return rt._get_model_config(config)                  # serve_quant.py:152-160


def make_policy(rt, model, device, config: str, norm_stats_dir, tokenizer_path):
    """The reference policy shim (serve_quant.py:309-495) around `model`, with a noise proxy."""
    norm_stats = rt._load_norm_stats(str(norm_stats_dir)) if norm_stats_dir else None
    cfg = model_config(rt, config)
    policy = rt.Pi0PyTorchPolicy(model=FixedNoise(model), device=torch.device(device),
                                 norm_stats=norm_stats, use_quantile_norm=False,
                                 is_joint_position="jointpos" in config,      # run_output_eval.py:442
                                 max_token_len=cfg.max_token_len, tokenizer_path=str(tokenizer_path))
    return policy


class FixedNoise:
    """Stands in for the model inside Pi0PyTorchPolicy so that infer() (which calls
    sample_actions(device, obs, num_steps=10), serve_quant.py:479) uses this frame's noise and the
    requested number of flow steps. NEW: the reference seeds the global RNG once per pass and lets
    sample_noise draw (run_output_eval.py:135-160, pi0_pytorch.py:173-180); here each frame's noise
    comes from its own seeded generator, so a frame's noise does not depend on the frames before it.
    """

    def __init__(self, model):
        self.model = model
        self.noise = None
        self.num_steps = 10

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        return self.model.sample_actions(device, observation, noise=self.noise, num_steps=self.num_steps)


def frame_noise(seed: int, horizon: int, action_dim: int, device) -> torch.Tensor:
    """Standard normal noise of PI0Pytorch.sample_noise's shape and dtype (pi0_pytorch.py:173-180),
    drawn on the CPU from a generator seeded per frame (device-independent)."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn((1, horizon, action_dim), generator=g, dtype=torch.float32).to(device)


def predict(policy, obs, noise, num_steps) -> np.ndarray:
    """One action chunk through the reference policy shim: (horizon, 8) in robot units."""
    proxy = policy.model
    proxy.noise, proxy.num_steps = noise, num_steps
    try:
        return np.asarray(policy.infer(obs)["actions"], dtype=np.float32)
    finally:
        proxy.noise = None


# ---------------------------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------------------------

def dummy_observations(n: int, seed: int) -> list[dict]:
    """The reference's seeded dummy observations (run_output_eval.py:73-81, 450-451), generated by
    the reference function itself."""
    roe = ref_loader.load_file("pi0-quant", "experiments/run_output_eval.py")
    rng = np.random.default_rng(seed)
    return [{"obs": roe._make_dummy_obs(rng), "gt": None, "meta": {"source": "dummy", "index": i}}
            for i in range(n)]


def _decode_frame(path, t: float, fps: float) -> np.ndarray:
    """The video frame shown at time t (s) of a concatenated LeRobot v3 video, HWC uint8 RGB.
    Seeks to the preceding keyframe and decodes forward, as run_eval.py:166-180 does by pts."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(int(max(t - 1.0, 0.0) / stream.time_base), any_frame=False, backward=True,
                       stream=stream)
        best = None
        for frame in container.decode(stream):
            ft = float(frame.pts * stream.time_base)
            if ft > t + 0.5 / fps:
                break
            best = frame
        if best is None:
            raise RuntimeError(f"no frame at t={t:.3f}s in {path}")
        return best.to_ndarray(format="rgb24")


def droid_observations(n: int, horizon: int, frame_idx: int = 0, seed: int = 0,
                       repo: str = DROID_REPO, revision: str = DROID_REVISION) -> list[dict]:
    """NEW: n DROID frames with their recorded action chunks (see the module docstring).

    Candidate episodes are those stored in the first data file and the first video file of both
    cameras (bounded download: ~0.1 GB of parquet and ~1 GB of video); they are shuffled with
    RandomState(seed) and the first n that are longer than frame_idx + horizon are used, one frame
    (frame_idx) per episode, as run_eval.py:208-213 does. State: joint_position (7) + gripper (1);
    the ground truth is the next `horizon` rows of `action` (absolute joint-position targets and
    gripper, openpi JOINT_POSITION), the space the reference shim's output is in after
    AbsoluteActions (serve_quant.py:488-495). DROID has no held-out split; these episodes may be in
    the policy's training data. Episodes with no language instruction are skipped (NEW).
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    def get(f):
        return hf_hub_download(repo, f, repo_type="dataset", revision=revision)

    info = json.loads(Path(get("meta/info.json")).read_text())
    fps = float(info["fps"])
    eps = pd.read_parquet(get("meta/episodes/chunk-000/file-000.parquet"))
    vcols = {k: f"videos/{v}" for k, v in DROID_CAMERAS.items()}
    sel = (eps["data/chunk_index"] == 0) & (eps["data/file_index"] == 0)
    for col in vcols.values():
        sel &= (eps[f"{col}/chunk_index"] == 0) & (eps[f"{col}/file_index"] == 0)
    # NEW: episodes without a language instruction are skipped (Pi0 is prompt-conditioned)
    has_task = eps["tasks"].map(lambda t: len(t) > 0 and bool(str(t[0]).strip()))
    cand = eps[sel & has_task & (eps["length"] > frame_idx + horizon)]
    order = sorted(cand["episode_index"].tolist())
    np.random.RandomState(seed).shuffle(order)
    chosen = order[:n]
    if len(chosen) < n:
        raise ValueError(f"only {len(chosen)} DROID episodes available in the first file")
    data = pd.read_parquet(get("data/chunk-000/file-000.parquet"))
    tasks = pd.read_parquet(get("meta/tasks.parquet"))
    task_text = dict(zip(tasks["task_index"].tolist(), tasks.index.tolist()))
    videos = {k: get(info["video_path"].format(video_key=v.removeprefix("videos/"), chunk_index=0,
                                               file_index=0)) for k, v in vcols.items()}
    out = []
    for ep in chosen:
        erow = eps[eps["episode_index"] == ep].iloc[0]
        rows = data[data["episode_index"] == ep].sort_values("frame_index")
        row = rows.iloc[frame_idx]
        gt = np.stack(rows["action"].iloc[frame_idx:frame_idx + horizon].to_numpy()).astype(np.float32)
        obs = {key: _decode_frame(videos[key], float(erow[f"{col}/from_timestamp"]) + frame_idx / fps, fps)
               for key, col in vcols.items()}
        obs["observation/joint_position"] = np.asarray(row["observation.state.joint_position"], np.float32)
        obs["observation/gripper_position"] = np.asarray(row["observation.state.gripper_position"], np.float32)
        obs["prompt"] = task_text.get(int(row["task_index"]), "")
        out.append({"obs": obs, "gt": gt,
                    "meta": {"source": repo, "revision": revision, "episode": int(ep),
                             "frame_index": int(frame_idx), "global_index": int(row["index"]),
                             "prompt": obs["prompt"]}})
    return out


def load_pi0(checkpoint_dir, *, openpi_dir=None, device="cpu", dtype=None, config=DEFAULT_CONFIG,
             norm_stats_dir=None, tokenizer_path=None):
    """One-call entry point: (model, policy, example_obs_fn).

    model           PI0Pytorch, eager, weights verified (load_model); quantize it in place with
                    quant_modes.quantize(model, mode, DEFAULT_SKIP, options) before use.
    policy          the reference policy shim around it (make_policy): predict(policy, obs, noise,
                    steps) returns a (horizon, 8) action chunk in robot units.
    example_obs_fn  example_obs_fn(n, seed) -> the reference's dummy observations
                    (dummy_observations); droid_observations gives real frames with ground truth.
    dtype is accepted for interface symmetry and must be None or torch.bfloat16: the model's
    precision is openpi's own mixed bfloat16 config (gemma_pytorch.py:62-82)."""
    if dtype not in (None, torch.bfloat16):
        raise ValueError("Pi0 runs in openpi's mixed bfloat16 precision only")
    rt = runtime(openpi_dir)
    model = load_model(rt, config, checkpoint_dir, device)
    norm_stats_dir = norm_stats_dir or str(Path(checkpoint_dir) / "assets" / "droid")
    tokenizer_path = tokenizer_path or os.environ.get("PALIGEMMA_TOKENIZER")
    policy = make_policy(rt, model, device, config, norm_stats_dir, tokenizer_path) if tokenizer_path else None
    return model, policy, dummy_observations


# ---------------------------------------------------------------------------------------------
# Patch coverage
# ---------------------------------------------------------------------------------------------

def _canonical(mode):
    from quant_modes import canonical

    return canonical(mode)


def inventory(model: nn.Module) -> dict:
    """The matmul sites of an unpatched Pi0: {fqn: {kind, dtype, component}}."""
    sites = {}
    for n, m in model.named_modules():
        kind = ("linear" if type(m) is nn.Linear else "conv2d" if type(m) is nn.Conv2d else
                "embedding" if type(m) is nn.Embedding else "attention" if n.split(".")[-1] == "self_attn"
                else None)
        if kind:
            w = getattr(m, "weight", None)
            sites[n] = {"kind": kind, "dtype": str(w.dtype).removeprefix("torch.") if w is not None else None,
                        "component": __import__('ref_attention').pi0_component(n)}
    return sites


class CallCounter:
    """Counts forward calls per module path (patch coverage: patched AND exercised)."""

    def __init__(self, model, fqns):
        self.counts = dict.fromkeys(fqns, 0)
        mods = dict(model.named_modules())
        self.handles = [mods[f].register_forward_hook(self._hook(f)) for f in fqns if f in mods]

    def _hook(self, f):
        def h(*_):
            self.counts[f] += 1
        return h

    def close(self):
        for h in self.handles:
            h.remove()


def patch_coverage(inv: dict, model: nn.Module, mode: str, layers, calls: dict, skip) -> dict:
    """Per site: patched or not (with the reason), and how often it ran in the first pass."""
    name = _canonical(mode)
    mods = dict(model.named_modules())
    sites = {s["site"]: s for s in (layers or {}).get("attention_sites", [])}
    rows, not_patched = [], []
    for fqn, info in inv.items():
        m = mods.get(fqn)
        if info["kind"] in ("linear", "conv2d"):
            patched = m is not None and (type(m).__name__ not in ("Linear", "Conv2d") or type(
                getattr(m, "weight", None)) not in (torch.Tensor, nn.Parameter))
        elif info["kind"] == "attention":
            patched = fqn in sites
        else:
            patched = False
        reason = None
        if not patched:
            if name == "bf16":
                reason = "baseline"
            elif info["kind"] == "embedding":
                reason = "embedding lookup (gather), no matmul; the reference leaves it as is"
            elif any(s and s in fqn for s in skip):
                reason = ("never called in Pi0 inference (run_eval.py:20-22); --skip" if "lm_head" in fqn
                          else "--skip")
            elif name in ("collab_v0", "collab_v0_hw", "torchao") and info["kind"] != "linear":
                reason = f"{name} covers nn.Linear only"
            elif name in ("collab_v0", "collab_v0_hw", "torchao") and info["dtype"] != "bfloat16":
                reason = f"{name} quantizes bfloat16 layers only (quant_modes.py:210-213); this one is float32"
            elif info["kind"] == "attention":
                reason = "attention not in --ops" if name not in ("collab_v0", "collab_v0_hw", "torchao") \
                    else f"{name} has no attention path"
            else:
                reason = "not replaced (see layers.untouched)"
        n_calls = (sum(sites[fqn]["calls"].values()) if info["kind"] == "attention" and fqn in sites
                   else calls.get(fqn))
        row = {"site": fqn, **info, "patched": patched, "calls": n_calls}
        if info["kind"] == "attention" and fqn in sites:
            row["routes"] = dict(sites[fqn]["calls"])
        if reason:
            row["reason"] = reason
            not_patched.append({"site": fqn, "kind": info["kind"], "reason": reason})
        rows.append(row)
    summary = {}
    for r in rows:
        s = summary.setdefault(r["kind"], {"sites": 0, "patched": 0, "patched_and_called": 0,
                                           "patched_not_called": []})
        s["sites"] += 1
        if r["patched"]:
            s["patched"] += 1
            if r["calls"]:
                s["patched_and_called"] += 1
            else:
                s["patched_not_called"].append(r["site"])
    return {"summary": summary, "not_patched": not_patched, "sites": rows}


