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

Multi-Task DiT. There is no DiT-specific profile: the generic rules apply, and its custom attention
is left unpatched (see `attention_sites` in the JSON).

    python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --n-frames 200 --modes bf16 collab_v0 collab_v0_hw reference --out out/dit_fast.json

Check every reference_hw and chip result before quoting it:

- `layers.<mode>.valid` is true;
- `crosscheck.ok` is true and `bit_mismatches` is 0;
- `reference_sources` records the pinned commits.
