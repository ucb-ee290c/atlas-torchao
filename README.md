# atlas-torchao

## Setup

    export HF_HOME=/rscratch/$USER/hf_cache
    conda env create -f environment.yml
    conda activate atlas
    git submodule update --init
    pip install -e model2MLIR --no-deps     # deps (xdsl, structlog, PyYAML) are in requirements.txt

### Notes
- Python 3.12 is required: model2MLIR uses Python 3.11+ features (enum.StrEnum).
- Install model2MLIR with --no-deps; its optional extras can upgrade torch/torchao.
- Create conda envs on /scratch or /rscratch, not your home directory.

## Reference numerics (third_party)

The reference author's quantization code is pinned as two submodules and loaded from there
(`src/ref_loader.py` checks the pins and records the sha256 of every file it loads):

    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    uv pip install numba            # or: pip install -e .[reference]; gcc is needed for the C models

Do not use `--recursive`: smolVLA-quant's nested `npu_model` submodule is not needed.

## Profiles

`--modes` accepts `bf16`, `collab_v0` (alias `atlas`), `collab_v0_hw` (alias `atlas_hw`),
`reference`, `reference_hw`, `chip` and `torchao`. `chip` is the headline: the whole model as it
would run on Atlas (MXU model + VPU + scalar core, host fallbacks reported). `reference_hw` is the
MXU-only comparison. Every numerics decision and its source, the findings and the open questions are
in [docs/reference_profile.md](docs/reference_profile.md).

Reference-profile options: `--mxu ipt|sa` (default `ipt`, the ipt_numba model, which is the reference
author's choice), `--mxu-impl torch|ref_cpu` (default `torch`), `--accum-compile` (compile the SA
step), `--ops linear,conv2d,attention`, and `--crosscheck on|off` (default on). With the cross-check on,
a seeded sample of real calls is recomputed by the reference CPU model, and any bit mismatch sets
`valid: false` in the result JSON.

## L40S run matrix

Setup, once per machine:

    git clone -b reference-profile git@github.com:ucb-ee290c/atlas-torchao.git && cd atlas-torchao
    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    pip install numba                                        # gcc must be on PATH for the C oracles
    export PYTHONPATH=src HF_HOME=/path/to/hf_cache TL=/path/to/TinyLlama-1.1B-Chat-v1.0
    mkdir -p out

Gates (stop if any fail):

    python -m pytest -q tests
    ATLAS_TEST_COMPILE=1 python -m pytest -q tests/test_accum_torch.py
    python scripts/bench_accum.py --device cuda --compile --check --ref | tee out/bench_accum.txt

Size the slow jobs from the measured throughput, so that each job fits in about 1–2 h:

    N = floor(seconds * MAC_per_s / (2048 * 0.97e9))

Use the ipt or SA compiled MAC/s from `out/bench_accum.txt` for 2048-token TinyLlama windows.
Start with N = 4 if in doubt.

TinyLlama (WikiText-2):

    # fast profiles, full split; the collaborator's protocol for comparison
    python src/eval.py --model $TL --modes bf16 collab_v0 collab_v0_hw reference torchao --out out/tl_fast_default.json
    python src/eval.py --model $TL --protocol mxq --nsamples 0 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/tl_fast_mxq.json
    # MXU models, seeded windows
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes reference_hw --mxu ipt --rmse --out out/tl_refhw_ipt.json
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes reference_hw --mxu sa --accum-compile --rmse --out out/tl_refhw_sa.json
    # whole chip (headline)
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes chip --mxu ipt --out out/tl_chip_ipt.json

SmolVLA (lerobot/svla_so100_pickplace; set F, the frame count for the slow jobs, from the
benchmark: one frame is roughly one forward of a few hundred tokens times the flow steps):

    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames 200 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/smolvla_fast.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames $F --modes reference_hw --mxu ipt --rmse --out out/smolvla_refhw_ipt.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --n-frames $F --modes chip --mxu ipt --out out/smolvla_chip_ipt.json

Policy-eval options (eval_policy.py):

- Determinism check: each mode runs twice by default and the result JSON records `determinism` and
  `deterministic` per mode. `--no-determinism-check` runs each mode once (halves the time, determinism
  unchecked). `--deterministic` turns on deterministic kernels (`torch.use_deterministic_algorithms`,
  cuDNN deterministic, cuBLAS workspace config). Compare runs only at equal thread counts: CPU
  reductions are split by thread, so the thread count changes bf16 results.
- `--dtype {bfloat16,float32}` (default bfloat16) sets the policy dtype for every mode, the baseline
  included; the reference author loads SmolVLA in float32. The baseline mode label stays `bf16` even
  with `--dtype float32` (it is the mode name used throughout the result JSON and the `vs_bf16`
  keys, so it is documented rather than renamed); read it as "the unquantized baseline in --dtype".

Multi-Task DiT. The dedicated DiT profile lives in multitask-dit-quant (see its README:
git@github.com:lorenshung/multitask-dit-quant.git). Here there is no DiT-specific profile: the
generic rules apply, and its custom attention is left unpatched (see `attention_sites` in the JSON).

    python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --n-frames 200 --modes bf16 collab_v0 collab_v0_hw reference --out out/dit_fast.json

Pi0 (openpi PI0Pytorch, `pi0_droid_jointpos_polaris`; set `OPENPI_DIR`, `PI0_CHECKPOINT` and
`PALIGEMMA_TOKENIZER`, see `src/pi0_model.py`). `--obs dummy` is the reference author's action-RMSE
methodology (pi0-quant experiments/run_output_eval.py: 4 seeded random observations, seed 0),
`--obs mxu_dummy` the observations of its per-layer runner (run_eval_mxu_io.py), `--obs droid`
real lerobot/droid_1.0.1 frames with recorded joint-position actions (MSE / MAE vs ground truth;
DROID is in this checkpoint's training mixture, so the episodes are not held out). The noise is
the reference's own draws, passed explicitly (`--noise reference`); every mode runs twice.

    python src/eval_pi0.py --obs dummy --modes bf16 collab_v0 collab_v0_hw reference torchao --out out/pi0_dummy_fast.json
    python src/eval_pi0.py --obs droid --n-obs 32 --modes bf16 collab_v0 collab_v0_hw reference torchao --out out/pi0_droid_fast.json
    python src/eval_pi0.py --obs mxu_dummy --modes reference --rmse --out out/pi0_mxu_reference.json
    python src/eval_pi0.py --obs dummy --modes reference_hw --mxu ipt --rmse --out out/pi0_refhw_ipt.json
    python src/eval_pi0.py --obs dummy --modes reference_hw --mxu sa --accum-compile --rmse --out out/pi0_refhw_sa.json
    python src/eval_pi0.py --obs dummy --modes chip --mxu ipt --out out/pi0_chip_ipt.json
    python src/eval_pi0.py --obs droid --n-obs 32 --modes reference_hw chip --mxu ipt --out out/pi0_droid_hw.json

Check every reference_hw and chip result before quoting it:

- `layers.<mode>.valid` is true;
- `crosscheck.ok` is true and `bit_mismatches` is 0;
- `reference_sources` records the pinned commits.

## Pi0 (openpi PyTorch Pi0)

The reference author's pi0-quant is the Pi0 reference profile. This repo follows it for the
model, checkpoint, config, observations, attention routes and action metric.
- `src/pi0_model.py`: loading, observations, patch coverage.
- `ref_attention.patch_pi0_attention`: the three attention routes.
- `src/eval_pi0.py`: the eval.

### Setup

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
  If the L40S driver needs a different build, reinstall it from the PyTorch index:

      VIRTUAL_ENV=$PI0_ENV uv pip install --reinstall torch==2.7.1 \
          --index-url https://download.pytorch.org/whl/cu126

- torchao 0.12.0 is the release for torch 2.7. The `torchao` mode (native FP8) has only been run
  under torchao 0.18 (main environment). **Its behaviour under 0.12 must be verified on the L40S**:
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

### What is patched (every profile, through the existing registry)

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

### Eval (`src/eval_pi0.py`)

- `--obs dummy` (default) is the reference's action-output methodology (run_output_eval.py).
  - The observations are the reference's seeded random DROID-format observations, made by the
    reference function.
  - They go through the reference policy shim (state normalized, actions unnormalized, joint
    deltas made absolute, first 8 dims).
  - Reported against the bf16 baseline: rel-L2, cosine, max abs, per dimension, and the
    reference's own per-observation and overall action RMSE / rel RMSE (`action_rmse.csv`).
- `--obs droid` uses real frames with recorded actions from `lerobot/droid_1.0.1` (revision pinned
  in code), joint-position space, one frame per episode. It adds MSE and MAE against the
  recorded action chunk.
  - DROID has no held-out split, so these episodes may be in the policy's training data.
  - The reference's own real-data loader reads `lerobot/droid_100`, whose state and actions are
    cartesian. That does not match a joint-position policy.
- `--rmse` gives per-layer (isolated) and cumulative relative RMSE per component (reference
  StatsTracker and CSVs).
- Also `--obs mxu_dummy` and `--obs sim`: the observations of the reference's per-layer runner
  (run_eval_mxu_io.py).
- Noise is always passed explicitly to `sample_actions`. The default, `--noise reference`,
  reproduces the reference's own draws bit for bit. `--noise per_frame` draws from a CPU generator
  seeded with `seed + frame`, which is device-independent.
- Determinism: each mode runs twice with the same noise.
- `chip` adds the coverage table and the crosscheck, as for the other models.
- `--steps` defaults to the reference's 10. Fewer steps set `reduced_steps: true`.

### NEW decisions for Pi0 (no source in pi0-quant; also in the JSON `new_decisions`)

| Decision | Reference | Reason |
|---|---|---|
| flow noise always passed explicitly to `sample_actions`. The default reproduces the reference's draws; the option `--noise per_frame` uses a CPU generator seeded per frame | `torch.manual_seed(seed)` once per pass (run_output_eval.py) | noise isolated from whatever else uses the global RNG (clean `--rmse` pass, other modes) |
| DROID ground truth from `lerobot/droid_1.0.1`, joint positions, prompts required, client-side `resize_with_pad` as openpi's DROID client (examples/droid/main.py:120-123) | run_eval.py:183-298: `droid_100` (cartesian), 224x224 stretch | the matching action space; openpi's own DROID preprocessing |
| `lm_head` left unpatched (`--skip lm_head` default) | every nn.Linear (model_patcher.py:246-288) | never called (run_eval.py:20-22); about 3 GB less under reference_hw |
| one score-matmul accumulator pair per attention module | one pair per patch function | the MXU models keep no state for activations |
| chip keeps SigLIP on the SDPA route | SigLIP runs through SDPA in the reference | the eager route would change the softmax precision |
| checkpoint converted at the converter's default bfloat16 | precision of the reference's file not recorded | converter default |

The checkpoint load fails loudly. The reference loader keeps random weights when loading fails
(serve_quant.py:233-257); here a missing file or a mismatch raises.

PI0Pytorch's `torch.compile` wrapper is removed for every mode, as the reference does
(run_eval.py:943-947).

### L40S run matrix for Pi0 (benchmark first)

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
    $P scripts/bench_accum.py --device cuda --compile --check --ref | tee out/bench_accum_pi0env.txt
    # fast profiles, reference methodology (dummy obs) and real DROID frames with ground truth
    $P src/eval_pi0.py --obs dummy --n-obs 32 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/pi0_fast_dummy.json
    $P src/eval_pi0.py --obs droid --n-obs 32 --modes bf16 collab_v0 collab_v0_hw reference torchao --rmse --out out/pi0_fast_droid.json
    # MXU models
    $P src/eval_pi0.py --obs dummy --n-obs $F --modes reference_hw --mxu ipt --rmse --out out/pi0_refhw_ipt.json
    $P src/eval_pi0.py --obs dummy --n-obs $F --modes reference_hw --mxu sa --accum-compile --rmse --out out/pi0_refhw_sa.json
    # whole chip (headline). RoPE and timestep tables precomputed on the host (--chip-rope host):
    # Pi0's positions (~880) and timestep angles (~1571 rad) exceed the vsin/vcos argument range
    # (~400 rad), see "### Pi0" under the chip profile
    $P src/eval_pi0.py --obs droid --n-obs $F --modes chip --mxu ipt --chip-rope host --out out/pi0_chip_ipt_ropehost.json
    # the reference treatment (sin/cos on the VPU), labelled as such, on a small subset
    $P src/eval_pi0.py --obs droid --n-obs 4 --modes chip --mxu ipt --chip-rope vpu --out out/pi0_chip_ipt_ropevpu.json

Check before quoting, as for the other models:
- `valid` is true;
- `crosscheck.ok` is true with 0 bit mismatches;
- `deterministic` is true;
- `patch_coverage.summary` has no `patched_not_called`;
- `data.reduced_steps` is false.

### Pi0 findings

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

## The `chip` profile: the whole model as Atlas would run it

`--modes chip` = `reference_hw` (MXU accumulation model for the matmuls) + every other op of the
forward pass routed to the Atlas unit that would execute it (`src/chip_profile.py`,
`src/chip_coverage.py`, `src/chip_ops.py`, `src/op_unit_map.py`):

| unit | what runs there | how it is emulated |
|---|---|---|
| MXU | nn.Linear (reference_hw), attention Q.K^T / P.V and stray mm/bmm, convolutions (im2col) | FP8 E4M3 operands with the reference po2 cast, `--mxu ipt\|sa`, `--mxu-impl`; `--chip-mxu framework` swaps in an FP8-operand float32 matmul rounded to BF16 (NEW, for CPU smokes) |
| VPU | elementwise arithmetic, exp/sin/cos/tanh/log2/rcp/sqrt, RMSNorm/LayerNorm, softmax, SiLU/GELU/sigmoid, reductions, rescale and bias adds after the MXU, additive masks | `ChipVPU`: vectorized torch, bit-exact to the reference `VectorRTLFunctions` (tests/test_chip_ops.py: unary ops on all 65536 BF16 inputs); composites are the reference sequences in `vpu_decomp.py` |
| scalar | integer / bool index arithmetic, comparisons, cumsum of masks | exact (RV32I ALU, ScalarALU.scala:26-36); integer mul/div is a software routine (NEW) |
| data | views, casts, concat, gather/embedding, copies, constants | exact, run natively |
| host | anything with no Atlas instruction (fused SDPA, image resize, bucketize, RNG, float compares, pow with other exponents, select between two computed tensors) and modules kept off-chip with `--skip` | torch on the host, **listed with a reason** |
| eval | perplexity loss | listed separately |

Every result JSON written with `chip` carries `layers.chip.coverage` (eval.py) or
`modes.chip.layers.coverage` (eval_policy.py): calls and work per unit (MXU FLOPs = 2MNK, VPU
lane-element ops, output elements otherwise), the host-fallback list with reasons, the
`new_decisions` list (op, choice, analogue, reason) and `silent_bf16_ops: 0`. The same table is
printed after the first pass and at exit. Options: `--chip-rope vpu|host` (RoPE tables on the VPU,
the reference treatment, or precomputed on the host together with the sinusoidal timestep table), `--chip-strict` (raise on an unmapped op).

    python src/eval.py --model /path/TinyLlama-1.1B-Chat-v1.0 --seqlen 128 --max-windows 2 \
        --modes bf16 chip --chip-mxu framework --device cpu --out out/tinyllama_chip.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
        --modes bf16 chip --chip-mxu framework --n-test-episodes 1 --n-frames 2 --device cpu \
        --out out/smolvla_chip.json

### Pi0

Every aten op of openpi's PI0Pytorch (`sample_actions`, pi0=True) has a unit: `MODEL_OPS` rows
`"pi0"` in `src/op_unit_map.py`, cited to pi0-quant's `patch_vector_ops` (quant_vector.py:136-171,
360-433) and vpu_decomp.py, which this repo follows op for op (GELU-tanh, RMSNorm with `1 + w` as
vsquare / staged rsum / vsqrt+vrecip / vadd / vmul, softmax, SiLU, embedding scale, RoPE apply,
Euler step). On a tiny random-weight Pi0 (tests/pi0_tiny.py, tests/test_chip_pi0.py)
`silent_bf16_ops` is 0 and the only host fallbacks are the float `le` of the block-causal mask
build, the loop test `ge` and the loop time update `add_` (plus the RoPE and timestep tables with
`--chip-rope host`). Use `--chip-rope host` for Pi0: its prefix positions reach ~880 and its
timestep angles 1571 rad, where the VPU's sin/cos (and BF16 angles) are wrong by up to 2.0
(the reference puts them on the VPU, so `vpu` stays the default and the reference treatment).
The masked logits (-2.38e38) are zeroed by the vexp floor (-128).

### NEW decisions (no reference or ISA source; the analogue followed is cited)

| op | choice | analogue |
|---|---|---|
| `add`/`sub` with alpha, in-place forms | vmul by alpha, write back in place | vpu_decomp.py:232-246 (the reference drops alpha and returns a new tensor) |
| `rsub` | vsub with the operands swapped | vpu_decomp.py:241-246 |
| `sigmoid` (QuickGELU) | vsub, vexp, vadd, vrecip | the sigmoid part of the reference SiLU, vpu_decomp.py:188-191 |
| `log` | vlog2 then vmul by ln 2 | vlog2.bf16 (green card 57/4C) |
| `abs` | vmaximum(x, 0 - x) | neg as vpu_decomp.py:263-264, vmaximum 57/06 |
| `pow` exponents 0.5, -1, -0.5, 1; `c ** x` | vsqrt, vrecip, vsqrt+vrecip, none; vexp2(x * log2 c) | _pow_dispatch, vpu_decomp.py:199-208 (other exponents stay host, as there) |
| `cumsum` (float) | sequential vadd scan | in TARGET_OPS (quant_vector.py:182) with no dispatch entry |
| multi-dim `sum`/`mean`/`amax`, `amin`, `max`/`min` | staged row reductions per dim; rmin pads with +inf | _reduce_with_dim, quant_vector.py:220-235; _vpu_rmax_all, vpu_decomp.py:53-72 |
| `where`/`masked_fill` with a constant branch | vadd of a host-precomputed additive mask (fill rounded to BF16); two computed branches are host | Llama's additive eager mask; no select instruction |
| `_fused_rms_norm`, `_safe_softmax` | the Llama RMSNorm primitives; the softmax sequence | quant_vector.py:150-188, vpu_decomp.py:132-158 |
| `mm`/`addmm` outside nn.Linear | per-tensor cast, MXU, vmul rescale, vadd bias | quant_linear.py:145-172 |
| `bmm`/`baddbmm` (attention when not patched) | per-matrix cast (per batch, head), MXU, vmul rescale; K = 1 is a vmul | model_patcher.py:667-699 |
| `exp2` of the integer scale exponent in RefQuantLinear | scalar-core exponent construction (exact) | 2^(sx+sw) in quant_linear.py:168 is a Python scalar |
| `--chip-mxu framework` | FP8 operands, float32 matmul, one BF16 rounding | ref_profile.py `reference` F.linear stand-in |
| exp inside `_softmax`, `silu`, `sigmoid` | `vmaximum(x, -128)` before `vexp` | the vexp model (bit-exact to ExpLane.scala) wraps below x = -336: exp(-3.3e38) = 1, so an additive min/-inf mask did not mask (TinyLlama chip PPL 942 -> 9.5 on the first window); analogue: ExpLane.scala:47 overflow pre-check |
| float64 0-d scalar arithmetic | host | schedule constants; Atlas has no FP64 |
| 0-d float `add_`/`sub_` (Pi0 flow `time += dt`) | host, float32 | pi0-quant TARGET_OPS has no `add_` (quant_vector.py:136-171), so the reference keeps it float32; the loop compare is host anyway. A BF16 vadd would truncate the timestep each step (1.0, 0.898, 0.797, ...) |
| `where(mask, c1, c2)` with two constant branches (Pi0 `torch.where(att, 0, -2.38e38)`) | data: the precomputed additive mask | the where/masked_fill decision above; openpi pi0_pytorch.py:160 |
| sinusoidal timestep table (`create_sinusoidal_pos_embedding`, Pi0 / SmolVLA) under `--chip-rope host` | host table, like RoPE | angles reach 2π/4e-3 = 1571 rad: outside vsin/vcos's range and 8 rad per BF16 ulp; the table depends only on the step schedule |

The model-level op -> unit map (with the aten ops each model operation becomes) is `MODEL_OPS` in
`src/op_unit_map.py`; the aten-level map with instruction sequences and citations is `ATEN`.

### Multi-Task DiT

multitask-dit-quant's op router accepts an external emulator. With this repo's `src` on
`sys.path`:

    from mtdit_inout.op_emulation import load_op_emulator
    em = load_op_emulator("chip_coverage")          # chip_coverage.make_mtdit_op_emulator()

`em.classify(func, args, kwargs)` returns "VPU" / "scalar" / None (None for data movement, which
that router counts as movement, and for matmuls, which its FP8 patches own: a matmul reaching the
router is host); `em.emulate(func, args, kwargs)` runs the op on ChipVPU with the chip profile's
handlers. Ops the emulator cannot run are classified None, i.e. host fallback in that report.

### Findings for the Atlas hardware team

- `vexp.bf16` (reference FPEX model, bit-exact to ExpLane.scala) is correct only for x >= -336; for
  more negative inputs the Q(m,n) conversion wraps (exp(-1000) = 2.6e10, exp(-3.3e38) = 1.0). Masked
  softmax with the framework's min/-inf mask is wrong unless software clamps first. Is a saturating
  underflow (return +0) intended in the RTL?
- `vsin`/`vcos` take a Q(9,12) argument (x 2/pi), i.e. |x| < ~400 rad, and BF16 angles have an ulp of
  0.5 rad at 64-128: RoPE angles for positions beyond a few hundred are not representable on the VPU.
  Should RoPE tables be precomputed on the host (`--chip-rope host`) in deployment?
  Pi0 is worse: prefix positions to ~880 and flow-timestep angles to 1571 rad (max |sin - sin_ref|
  up to 2.0 on the VPU; 406 of 1024 timestep-embedding values off by more than 0.1 at t = 1).
- `vexp.bf16` wraps already at x = -336 (sigmoid via the reference SiLU sequence comes out 0 for
  x = 336), not only below it.
- `vadd`/`vsub` truncate the FP32 sum to BF16 (raw upper half, AddSubSumVec.scala:133) while
  `vmul` and `vredsum.row` round to nearest even: intended?
- The scalar core has no multiply/divide and no FP (ScalarALU.scala:26-36): float schedule
  constants (flow-matching dt, DDPM coefficients) and integer index multiplies need host
  precomputation or software routines.
- No select/compare instruction on the VPU: masks must be additive and precomputed on the host.
