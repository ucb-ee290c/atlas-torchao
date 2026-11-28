# Pi0 (openpi PyTorch Pi0)

The reference author's pi0-quant is the Pi0 reference profile. This repo follows it for the
model, checkpoint, config, observations, attention routes and action metric.
- `src/loaders/pi0_model.py`: loading, observations, patch coverage.
- `ref_attention.patch_pi0_attention`: the three attention routes.
- `src/eval_pi0.py`: the eval.

## Setup

openpi's PyTorch Pi0 needs transformers 4.53.2 with openpi's `transformers_replace` files. That
conflicts with the main environment (transformers 5.5.4, numpy 2), so Pi0 runs in a separate
environment built from openpi's own lock file. This package runs there unchanged: the full test
suite passes in it.

    # openpi, pinned (outside this repo; the reference reads it through OPENPI_DIR)
    git clone https://github.com/Physical-Intelligence/openpi.git $OPENPI_DIR
    git -C $OPENPI_DIR checkout 215abfb217dbac7d5f1273282331b9b1866c0479

    # environment from openpi's uv.lock: torch 2.7.1, transformers 4.53.2, numpy 1.26, jax 0.5.3
    cd $OPENPI_DIR && UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=$PI0_ENV GIT_LFS_SKIP_SMUDGE=1 \
        uv sync --python 3.11 --frozen && cd -
    cp -r $OPENPI_DIR/src/openpi/models_pytorch/transformers_replace/* \
        $PI0_ENV/lib/python3.11/site-packages/transformers/
    VIRTUAL_ENV=$PI0_ENV UV_LINK_MODE=copy uv pip install torchao==0.12.0

Notes on the environment:
- `UV_LINK_MODE=copy` matters. With uv's default hard links, the `transformers_replace` copy also
  rewrites transformers in the uv cache (openpi README).
- The torch 2.7.1 wheel that uv resolves on Linux x86-64 is the CUDA 12.6 build (`2.7.1+cu126`).
  If the GPU driver needs a different build, reinstall it from the PyTorch index:

      VIRTUAL_ENV=$PI0_ENV uv pip install --reinstall torch==2.7.1 \
          --index-url https://download.pytorch.org/whl/cu126

- torchao 0.12.0 is the release for torch 2.7. The `torchao` mode (native FP8) has only been run
  under torchao 0.18 (main environment). **Its behaviour under 0.12 must be verified on a GPU**:
  the result JSON records the version.
- numba comes with the lock. gcc must be on PATH for the C oracles.

Checkpoint: `pi0_droid_jointpos_polaris`, the reference's default (pi0-quant
experiments/run_eval.py:854-855; openpi polaris_config.py).
- Source: `gs://openpi-assets/checkpoints/polaris/pi0_droid_jointpos_polaris`, a public bucket,
  12 GB of JAX/Orbax parameters.
- Convert it with openpi's converter at its default precision (bfloat16), as the reference's
  `pi0_droid_jointpos_safetensors` was.

Commands:

    # download (gsutil -m cp -r ..., or the GCS JSON API), then:
    JAX_PLATFORMS=cpu PYTHONPATH=$OPENPI_DIR/src $PI0_ENV/bin/python \
        $OPENPI_DIR/examples/convert_jax_model_to_pytorch.py \
        --checkpoint_dir $CKPT_JAX --config_name pi0_droid_jointpos_polaris --output_path $PI0_CHECKPOINT
    cp -r $CKPT_JAX/assets $PI0_CHECKPOINT/                 # norm_stats.json (assets/droid)
    curl -o $PALIGEMMA_TOKENIZER https://storage.googleapis.com/big_vision/paligemma_tokenizer.model

Converted on this host:
- `model.safetensors` sha256 `fea1abb2ef028526818a285bcdbcb8a5907d4a811ec364bf4711da4ff768c823`,
  7002873776 bytes;
- all 54 downloaded objects match their GCS md5.

The result JSON records the checkpoint sha256 (cached next to the file), the openpi commit and
dirty state, whether `transformers_replace` is installed, and the norm-stats and tokenizer hashes.

Environment variables: `OPENPI_DIR`, `PI0_CHECKPOINT`, `PALIGEMMA_TOKENIZER` (or `--openpi-dir`,
`--checkpoint-dir`, `--tokenizer-path`), and `HF_HOME` for the DROID data.

Tests (tiny random-weight Pi0 with the real structure; they skip without OPENPI_DIR and
transformers_replace):

    OPENPI_DIR=... PYTHONPATH=src $PI0_ENV/bin/python -m pytest -q tests

## What is patched (every profile, through the existing registry)

| Profile | Linear | Conv2d (SigLIP patch) | Attention Q.K^T / P.V |
|---|---|---|---|
| `collab_v0`, `collab_v0_hw`, `torchao` | bfloat16 nn.Linear (415 of 422; the 5 float32 action-head layers and the 2 lm_heads stay) | no | no |
| `reference`, `reference_hw` | every nn.Linear, the float32 action head included, except the 2 lm_heads | yes | all 63 `self_attn` |
| `chip` | as `reference_hw`, plus every other op on the VPU / scalar core or reported as host fallback | yes | yes |

- Components follow the reference table (model_patcher.py:92-109): vision, language,
  action_expert, action_head. These components are the rows of the `--rmse` CSVs.
- Attention follows the reference's three routes (model_patcher.py:658-1185):
  - Gemma `eager_attention_forward` covers the language model, the expert and the joint layers.
    Pi0 adds a float32 additive mask, so the softmax runs in float32.
  - SigLIP runs through `F.scaled_dot_product_attention`, because its config uses sdpa in
    transformers 4.53.2. The softmax is BF16 with no mask.
  - The SigLIP eager route is also patched, and fires only if the config is eager.
- Each attention site records the route that fired.
- Embeddings (gather) are never patched.
- The per-site list, with the reason for every site left alone, is `patch_coverage` in the JSON.

## Eval (`src/eval_pi0.py`)

- `--obs dummy` (default) is the reference's action-output methodology (run_output_eval.py).
  - The observations are the reference's seeded random DROID-format observations, made by the
    reference function.
  - They go through the reference policy shim (state normalized, actions unnormalized, joint
    deltas made absolute, first 8 dims).
  - Reported against the bf16 baseline: rel-L2, cosine, max abs, per dimension, and the
    reference's own per-observation and overall action RMSE / rel RMSE (`action_rmse.csv`).
- Headline (NEW, `modes.<mode>.headline`): metrics in joint-delta space, pred - state on the 7
  joint dims (gripper absolute): rel-L2, cosine and per dimension vs bf16 (`vs_bf16_delta`) and,
  for `--obs droid`, MSE / MAE vs the recorded deltas (`vs_ground_truth_delta`). The reference's
  action metric is on absolute joint positions (the shim adds the state back,
  serve_quant.py:488-492), where the robot's pose dominates the norm and hides the quantization
  error; it is kept unchanged as `vs_bf16.reference_action_rmse` (run_output_eval.py:251-295).
- `--obs mxu_dummy` is the observation source of the reference's per-layer runner
  (run_eval_mxu_io.py:114-133, 793-801: `torch.manual_seed(0)`, model-level random tensors and an
  empty prompt, straight into `sample_actions`); use it with `--rmse` to compare per-component
  RMSE with the reference's CSVs. `--obs sim --obs-dir DIR` loads sim-evals `obs_*.npz` files
  through the reference's `_load_real_obs` (the files are not public). Model-level sources report
  the first 8 dims of the model's normalized output.
- `--obs droid` uses real frames with recorded actions from `lerobot/droid_1.0.1` (revision pinned
  in code), joint-position space, one frame per episode. It adds MSE and MAE against the
  recorded action chunk.
  - DROID has no held-out split, and droid 1.0.1 is 0.9 of this checkpoint's fine-tuning mixture
    (openpi polaris_config.py:116-142), so these episodes may be in the policy's training data;
    read the MSE / MAE as fidelity to recorded behaviour, with BF16 agreement as the primary metric.
  - The reference's own real-data loader reads `lerobot/droid_100`, whose state and actions are
    cartesian. That does not match a joint-position policy.
- `--rmse` gives per-layer (isolated) and cumulative relative RMSE per component (reference
  StatsTracker and CSVs).
- Noise: always passed explicitly to `sample_actions`, never drawn from the global RNG.
  `--noise reference` (default) reproduces the reference's draws bit for bit with a per-frame
  `torch.Generator` on the model device: policy-level sources take draw i of a stream seeded once
  per pass (run_output_eval.py:176, 216), model-level sources a generator seeded `seed + i`
  (run_eval_mxu_io.py:579, 652). `--noise per_frame` draws on the CPU from `seed + frame`, so it is
  identical across devices. The JSON records the noise sha256.
- Determinism: each mode runs twice with the same noise and must match bit for bit.
- Agreement with bf16 also has per-dimension RMSE, max abs and cosine.
- The reference's policy shim always runs 10 flow steps (serve_quant.py:479), so its
  `--steps` is a label only; here `--steps` reaches `sample_actions`.
- `chip` adds the coverage table and the crosscheck, as for the other models.
- `--steps` defaults to the reference's 10. Fewer steps set `reduced_steps: true`.

## NEW decisions for Pi0 (no source in pi0-quant; also in the JSON `new_decisions`)

| Decision | Reference | Reason |
|---|---|---|
| headline metrics on joint deltas (pred - state, gt - state) | action RMSE on absolute joint positions (run_output_eval.py:251-295) | the state dominates the absolute norm and hides the quantization error |
| flow noise from per-frame generators that reproduce the reference's draws, passed to `sample_actions` | global RNG: `torch.manual_seed(seed)` once per pass (run_output_eval.py:176, 216) or per frame (run_eval_mxu_io.py:579, 652) | a frame's noise must not depend on earlier frames, modes or profiles |
| every mode on a freshly loaded model | baseline and patched pass on one model object (run_output_eval.py:172-236) | the profiles are not undone; equivalent under fixed noise |
| DROID ground truth from `lerobot/droid_1.0.1`, joint positions, prompts required, client-side `resize_with_pad` as openpi's DROID client (examples/droid/main.py:120-123) | run_eval.py:183-298: `droid_100` (cartesian), 224x224 stretch | the matching action space; openpi's own DROID preprocessing |
| `lm_head` left unpatched (`--skip lm_head` default) | every nn.Linear (model_patcher.py:246-288) | never called (run_eval.py:20-22); about 3 GB less under reference_hw |
| one score-matmul accumulator pair per attention module | one pair per patch function | the MXU models keep no state for activations |
| chip keeps SigLIP on the SDPA route | SigLIP runs through SDPA in the reference | the eager route would change the softmax precision |
| checkpoint converted at the converter's default bfloat16 | precision of the reference's file not recorded | converter default |

The checkpoint load fails loudly. The reference loader keeps random weights when loading fails
(serve_quant.py:233-257); here a missing file or a mismatch raises.

PI0Pytorch's `torch.compile` wrapper is removed for every mode, as the reference does
(run_eval.py:943-947).

## Run matrix for Pi0 (benchmark first)

One Pi0 frame (10 flow steps) is 2.14e12 MAC, measured with torch's FlopCounterMode on the real
checkpoint.
- The prefix pass is 2.09e12: three 224x224 images through SigLIP (the masked third camera is
  still encoded) and the Gemma 2B prefix of 768 image + 100 text tokens.
- Each flow step adds about 4.2e9: the expert's 11 tokens attending to the cached prefix.

So the step count barely changes the cost, and `--skip paligemma_with_expert.paligemma` (expert
and action head only) cuts it by ~500x.

Size the `reference_hw` / `chip` frame count F from the measured MXU throughput in
`out/bench_accum.txt`, so that each job takes about 1–2 h:

    F = floor(seconds * MAC_per_s / 2.14e12)

Start with F = 4, the reference's `--n-obs` default.

    export OPENPI_DIR=... PI0_CHECKPOINT=... PALIGEMMA_TOKENIZER=... HF_HOME=... PYTHONPATH=src
    P=$PI0_ENV/bin/python
    $P -m pytest -q tests                                   # includes tests/test_pi0.py
    $P src/scripts/bench_accum.py --device cuda --compile --check --ref | tee out/bench_accum_pi0env.txt
    # steps 1-5 (fast): reference methodology (dummy obs) and real DROID frames with ground truth
    $P src/eval_pi0.py --obs dummy --n-obs 32 --modes bf16 collab_v0 ref_linear ref_attn ref_conv --rmse --out out/pi0_ladder_fast_dummy.json
    $P src/eval_pi0.py --obs droid --n-obs 32 --modes bf16 collab_v0 ref_linear ref_attn ref_conv --rmse --out out/pi0_ladder_fast_droid.json
    # per-layer RMSE on the reference's per-layer observations (run_eval_mxu_io.py)
    $P src/eval_pi0.py --obs mxu_dummy --modes bf16 ref_conv --rmse --out out/pi0_mxu_reference.json
    # steps 6-8 on F frames; --subset-compare (default on) adds bf16 and ref_conv on the same frames
    $P src/eval_pi0.py --obs droid --n-obs $F --modes reference_hw chip chip_rope_vpu --mxu ipt --out out/pi0_ladder_subset_droid.json
    # side profiles
    $P src/eval_pi0.py --obs droid --n-obs 32 --modes bf16 collab_v0_hw torchao --out out/pi0_side_droid.json
    $P src/eval_pi0.py --obs dummy --n-obs $F --modes reference_hw --mxu sa --accum-compile --out out/pi0_refhw_sa.json

`chip` puts the RoPE and timestep tables on the host by default: Pi0's positions (~880) and
timestep angles (~1571 rad) exceed the vsin/vcos argument range (~400 rad), see [chip.md, Pi0](chip.md#pi0). `chip_rope_vpu` is the reference treatment, kept as the ablation that shows
that effect (its delta from `chip` is the cost of on-VPU tables).

Check before quoting, as for the other models:
- `valid` is true;
- `crosscheck.ok` is true with 0 bit mismatches;
- `deterministic` is true;
- `patch_coverage.summary` has no `patched_not_called`;
- `data.reduced_steps` is false.

## Pi0 findings

CPU smokes on the real checkpoint. Host: a shared 160-core machine under load, 8 threads, no GPU.
The flow steps were reduced (`reduced_steps: true`), so these are smoke numbers, not results.

1. `--obs droid --n-obs 1 --steps 2 --modes bf16 reference collab_v0 collab_v0_hw --rmse`

   | mode | rel-L2 vs bf16 | cosine | MSE vs DROID action | deterministic |
   |---|---|---|---|---|
   | bf16 | - | - | 9.91e-4 | yes |
   | reference | 4.9e-3 | 0.99999 | 1.18e-3 | yes |
   | collab_v0 | 1.8e-3 | 0.999998 | 1.05e-3 | yes |
   | collab_v0_hw | 2.6e-3 | 0.999997 | 1.12e-3 | yes |

   Patch coverage:
   - `reference`: 420/422 Linear (the two lm_heads skipped), 1/1 conv and 63/63 attention sites,
     all patched and all called.
   - `collab_v0*`: 415 bf16 Linear only.

   Mean per-layer / cumulative relative RMSE under `reference` (vision 0.020 / 0.048, language
   0.020 / 0.083, action_expert 0.024 / 0.085, action_head 0.023 / 0.038).

2. `--obs dummy --n-obs 1 --steps 1 --modes bf16 reference_hw --mxu ipt --skip lm_head
   paligemma_with_expert.paligemma` (expert and action head only; the full model at CPU MXU-model
   speed would take about a day per frame).

   Coverage: 131 Linear and 18 attention sites, all patched and called.

   | metric | value |
   |---|---|
   | crosscheck | ok, 0 bit mismatches over 12 calls |
   | valid | true |
   | deterministic | yes |
   | rel-L2 vs bf16 | 1.9e-2 |
   | reference rel action RMSE | 1.9e-2 |

   Not run here: `torchao` (needs CUDA), full-model `reference_hw` and `chip` (CPU MXU model too
   slow; chip coverage is tested on the tiny Pi0 in tests/test_chip_pi0.py).

Findings:
- **The reference's action metric understates error.** Its actions (run_output_eval) are absolute
  joint positions: the shim adds the current state (serve_quant.py:488-492). The baseline's RMS
  is dominated by the robot's pose (about 0.56 rad here), so rel-L2 and rel RMSE look small even
  when the predicted motion differs. Compare the per-dimension numbers, and the DROID ground-truth
  error, as well.
- **The reference's real-data loader does not fit this checkpoint.** It reads
  `lerobot/droid_100`, whose `observation.state` is cartesian pose plus gripper. It feeds that to
  a joint-position policy as joint angles (run_eval.py:259-266).
- **The reference's checkpoint loader can fail silently.** On any load error it keeps random
  weights with only a warning (serve_quant.py:233-257). Its stubbed `resize_with_pad_torch` breaks
  on non-224 images.
- **SigLIP attention runs through SDPA under transformers 4.53.2**, so the reference's
  `patch_attn_siglip_eager` never fires there. Its SDPA route computes the vision softmax in
  BF16, while SmolVLA's vision route uses float32.
