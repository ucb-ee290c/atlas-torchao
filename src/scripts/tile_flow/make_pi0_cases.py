"""Write Pi0 capture cases in the single-step bundle layout, for make_loop_manifest.py pi0 and pi0_phase_capture.py.

pi0_phase_capture.py reads its capture cases from an existing single-step bundle's fx/fixtures.safetensors + parity.json, but the
single-step capture driver (pi0_fx.py) is outside the repository. This writer produces only those cases (no single-step graph):

  inputs  (img_base, img_left_wrist, img_right_wrist, mask x3, tokenized_prompt, tokenized_prompt_mask, state, noise): the
          Observation the reference policy shim (Pi0PyTorchPolicy.infer) hands to sample_actions for DROID frames of the eval
          protocol (eval_pi0 --obs droid: pi0_model.droid_observations + client_resize, --n-obs, --seed), captured by a recorder that
          stands in for the model inside the shim (tokenization, state normalization and image scaling are the shim's own), and
          the frame's noise from eval_pi0.frame_noise (--noise reference, policy level);
  output  the quantized model's sample_actions(num_steps=1) on those inputs (x_1 = noise - v at time 1.0), eager, in this process.

The quantized model is rebuilt from the export with pi0_phase_capture.load_pi0 (read only). Modeled on the DiT cases writer
(fx-bundles/dit/tools/make_dit_cases.py).

usage (PYTHONPATH=<worktree>/src:<worktree>/src/scripts/tile_flow, openpi env, HF offline, fixed OMP_NUM_THREADS):
  make_pi0_cases.py --export-dir EXPORT --out CASES_DIR --openpi-dir OPENPI --checkpoint-dir CKPT --tokenizer-path TOK [--n-obs 2] [--seed 0]
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch


class _Stop(Exception):
    pass


class _Recorder:
    """Takes the model's place inside the shim's FixedNoise proxy: records the Observation sample_actions receives, then stops."""

    def __init__(self):
        self.observation = None

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        self.observation = observation
        raise _Stop


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--openpi-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--n-obs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import eval_pi0
    import loaders.pi0_model as pi0_model
    import pi0_phase_capture as ppc
    from safetensors.torch import save_file

    out = Path(a.out)
    model, cfg = ppc.load_pi0(a.export_dir, a.openpi_dir, a.checkpoint_dir, a.tokenizer_path, "cpu")
    rt = pi0_model.runtime(pi0_model.openpi_dir(a.openpi_dir))
    frames = eval_pi0.client_resize(pi0_model.droid_observations(a.n_obs, cfg.action_horizon, 0, a.seed))
    policy = pi0_model.make_policy(rt, model, "cpu", pi0_model.DEFAULT_CONFIG, Path(a.checkpoint_dir) / "assets" / "droid",
                                   a.tokenizer_path)
    recorder = _Recorder()
    policy.model.model = recorder                     # FixedNoise proxy -> recorder (the shim's sample_actions call lands here)
    shape = (1, cfg.action_horizon, cfg.action_dim)
    (out / "fx").mkdir(parents=True, exist_ok=True)
    tensors, parity = {}, {"cases": []}
    with torch.no_grad():
        for j, fr in enumerate(frames):
            try:
                policy.infer(fr["obs"])
            except _Stop:
                pass
            o = recorder.observation
            noise = eval_pi0.frame_noise("reference", "policy", a.seed, j, shape, "cpu")
            ins = [*(o.images[k] for k in ppc.KEYS), *(o.image_masks[k] for k in ppc.KEYS), o.tokenized_prompt,
                   o.tokenized_prompt_mask, o.state, noise]
            x1 = model.sample_actions("cpu", o, noise=noise, num_steps=1)
            keys = [f"case_{j}.input_{n}" for n in range(len(ins))]
            tensors.update({k: v.detach().cpu().contiguous().clone() for k, v in zip(keys, ins)})
            tensors[f"case_{j}.output_0"] = x1.detach().cpu().contiguous()
            parity["cases"].append({"inputs": keys, "outputs": [f"case_{j}.output_0"], "frame": fr["meta"],
                                    "output": "sample_actions(num_steps=1) of the quantized model (eager, this process)"})
    save_file(tensors, str(out / "fx" / "fixtures.safetensors"))
    (out / "fx" / "parity.json").write_text(json.dumps(parity, indent=1, default=str))
    ckpt = Path(a.checkpoint_dir)
    bm = {"note": "capture cases only (make_pi0_cases.py): no single-step graph; the single-step driver pi0_fx.py is absent",
          "writer": str(Path(__file__).resolve()), "writer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          "inputs": ["image " + k for k in ppc.KEYS] + ["image_mask " + k for k in ppc.KEYS] +
                    ["tokenized_prompt", "tokenized_prompt_mask", "state (normalized, padded to action_dim)", "noise"],
          "data": {"obs": "droid", "n_obs": a.n_obs, "seed": a.seed, "frames": [fr["meta"] for fr in frames],
                   "noise": "eval_pi0.frame_noise('reference', 'policy', seed, frame)"},
          "torch_num_threads": torch.get_num_threads(),
          "export_manifest_sha256": hashlib.sha256((Path(a.export_dir) / "manifest.json").read_bytes()).hexdigest(),
          "model": {"id": pi0_model.DEFAULT_CONFIG, "checkpoint": {
              "dir": str(ckpt.resolve()), "config_json_sha256": hashlib.sha256((ckpt / "config.json").read_bytes()).hexdigest()}}}
    (out / "bundle_manifest.json").write_text(json.dumps(bm, indent=1, default=str))
    print(json.dumps({"dir": str(out), "cases": len(frames), "episodes": [fr["meta"]["episode"] for fr in frames]}))


if __name__ == "__main__":
    main()
