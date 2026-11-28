# atlas-torchao

New here? [docs/user-guide.md](docs/user-guide.md) walks from a fresh clone to a quantized export.

One command does the main jobs (`pip install -e .` installs it; `python -m atlas` without installing):

    MODEL=$(hf download --quiet TinyLlama/TinyLlama-1.1B-Chat-v1.0)              # local snapshot path
    atlas export $MODEL out/tinyllama/export                                     # quantize + export (atlas_v1_awq_tile_static)
    atlas eval $MODEL --modes bf16 ref_conv --from-export out/tinyllama/export
    atlas fx out/tinyllama/export out/tinyllama/fx --seqlen 64                    # native-FP8 FX bundle
    atlas verify out/tinyllama/export out/tinyllama/fx

Each runs an existing script unchanged (`src/eval.py`, `src/eval_policy.py`, `src/eval_dronet.py`,
`python -m exporter.fx_export`, `verify_bundle.py`); those keep working, and `--dry-run` prints the equivalent
command ([user guide](docs/user-guide.md#equivalent-commands)).

## Setup

Step by step, from a fresh clone to a verified export: [docs/user-guide.md](docs/user-guide.md). In short:

    conda env create -p /scratch/$USER/atlas-env -f environment.yml    # Python 3.12, .[reference,policy], pytest
    conda activate /scratch/$USER/atlas-env
    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    export PYTHONPATH=src HF_HOME=/scratch/$USER/hf_home

- `third_party/atlas-kernels` and `third_party/npu_model` (hardware cross-checks only) use SSH URLs, so cloning
  them needs an SSH key registered with GitHub; `pi0-quant` and `smolVLA-quant` use HTTPS.
- Create conda envs on /scratch or /rscratch, not your home directory.
- Pi0 runs in openpi's own environment: [docs/pi0.md](docs/pi0.md).

## Layout of `src/`

Entry points sit at the top of `src/` (run them as `python src/eval.py ...`, or through `atlas`); everything else is
a package.
`src/` must be on `PYTHONPATH` (the entry points and `tests/conftest.py` do this).

| Path | What lives there |
|---|---|
| `atlas.py` | the `atlas` command: `export`, `eval`, `fx`, `verify`, a thin translation into the CLIs below |
| `eval.py`, `eval_policy.py`, `eval_pi0.py`, `eval_dronet.py` | CLIs: LLM perplexity, LeRobot policies (SmolVLA, DiT), Pi0, DroNet |
| `quant_modes.py` | the coverage ladder: mode name to profile, and `quantize()` dispatch |
| `layer_rmse.py` | per-layer isolated and cumulative RMSE against a clean BF16 pass |
| `reference/` | the reference author's cast and module replacements: `ref_loader` (pinned submodules), `ref_numerics` (E4M3 cast, po2 scales), `ref_profile` (Linear), `ref_attention` (Q.K, P.V, Conv2d) |
| `mxu/` | MXU accumulation: `accum_torch` (ports of the `ipt` and `sa` models), `accum_backends` (backend choice), `accum_crosscheck` (bit-level check against the reference CPU model) |
| `chip/` | the `chip` profile: `chip_ops` (lane-exact VPU), `chip_numerics` (vadd rounding), `chip_coverage` (aten op router), `chip_profile` (installs it), `op_unit_map` (op to unit table) |
| `algos/` | quantization algorithms and recipes (SmoothQuant, GPTQ/AWQ, granularity, static calibration, mixed precision) |
| `exporter/` | `quant_export` / `load_quantized` (SafeTensors + policy + manifest), `fx_export` (FX replay bundle; run as `python -m exporter.fx_export`) |
| `loaders/` | model and data loading: `modelutils`, `datautils`, `ppl_protocols`, `pi0_model`, `ckpt_keys`, `policy_norm` |
| `collab/` | the earlier collaborator path: `atlas_quant`, `atlas_torchao` (`collab_v0` mode) |
| `chip_coverage.py` | compatibility shim for `load_op_emulator("chip_coverage")` in multitask-dit-quant; the module is `chip/chip_coverage.py` |

## Reference numerics (third_party)

The reference author's quantization code is pinned as two submodules and loaded from there
(`src/reference/ref_loader.py` checks the pins and records the sha256 of every file it loads):

    git submodule update --init third_party/pi0-quant third_party/smolVLA-quant
    uv pip install numba            # or: pip install -e .[reference]; gcc is needed for the C models

Do not use `--recursive`: smolVLA-quant's nested `npu_model` submodule is not needed.

## Profiles and the coverage ladder

`--modes` accepts the ladder steps below plus `collab_v0_hw` (alias `atlas_hw`) and `torchao`.
The ladder adds one source of error per step, so the difference between two consecutive steps,
measured on the same windows / frames, isolates that source:

| step | mode | adds over the previous step | quantized / emulated |
|---|---|---|---|
| 1 | `bf16` | baseline | nothing |
| 2 | `collab_v0` (alias `atlas`) | Linear FP8, the collaborator's way: ceil/448 weight scale, fixed 2^0 activation scale, torch cast | Linear; framework accumulation |
| 3 | `ref_linear` | reference numerics: dynamic po2 floor(log2(amax/256)) scales, vpack cast, BF16 bias after the rescale | Linear; framework accumulation |
| 4 | `ref_attn` | attention Q·Kᵀ and P·V, per-head FP8 | + attention |
| 5 | `ref_conv` (alias `reference`) | Conv2d | + Conv2d (SmolVLA, Multi-Task DiT, Pi0). TinyLlama has none: `ref_conv` ≡ `ref_attn` (the JSON says `equivalent_to`; do not run both) |
| 6 | `reference_hw` | Atlas MXU accumulation (`--mxu ipt`, default, or `sa`), cross-checked | all matmuls through the MXU model |
| 7 | `chip` | VPU / scalar-core vector ops, lane-exact; RoPE and timestep sin/cos tables as compile-time host constants (`--chip-rope host`, the default, NEW) | everything; host fallbacks listed |
| 8 | `chip_rope_vpu` | the tables computed on the VPU instead (the reference author's treatment) | everything |

Steps 3-5 are the reference profile with an op subset (`--ops` defaults to the step's own classes;
an explicit `--ops` overrides every profile and is recorded as off the ladder). Every result JSON
records, per mode, `ladder`: the step, what it adds, `quantized_op_classes` with counts
(`linear`, `attention` sites, `conv2d`), `operand_cast`, `accumulation`, `vector_ops`,
`rope_tables`, and `on_ladder` with `off_ladder_reasons` (an explicit `--ops`, `chip` with
`--chip-rope vpu`, `--chip-mxu framework`). `ladder_deltas` lists the consecutive-step differences
of the modes of that run. How to read them: [docs/reference_profile.md](docs/reference_profile.md#reading-the-ladder).

Subset steps: `reference_hw`, `chip` and `chip_rope_vpu` run an MXU model and are evaluated on a
subset (seeded windows, a few frames). `--subset-compare on` (the default) adds `bf16` and
`ref_conv` to such a run, so steps 5 → 6 → 7 → 8 are measured on identical windows / frames and
noise (`subset_compare.added_modes`, `added_by_subset_compare` in the JSON). Never subtract a
subset result from a full-split result.

Reference-profile options: `--mxu ipt|sa` (default `ipt`, the ipt_numba model, which is the reference
author's choice), `--mxu-impl torch|ref_cpu` (default `torch`), `--accum-compile` (compile the SA
step), `--ops linear,conv2d,attention`, and `--crosscheck on|off` (default on). With the cross-check on,
a seeded sample of real calls is recomputed by the reference CPU model, and any bit mismatch sets
`valid: false` in the result JSON. Every numerics decision and its source, the findings and the open
questions are in [docs/reference_profile.md](docs/reference_profile.md).

## Run matrix: the ladder per model

Setup as above (gcc must be on PATH for the C oracles), then:

    export TL=/path/to/TinyLlama-1.1B-Chat-v1.0
    mkdir -p out

Gates (stop if any fail), then the benchmark that sizes every subset job:

    python -m pytest -q tests
    ATLAS_TEST_COMPILE=1 python -m pytest -q tests/test_accum_torch.py
    python src/scripts/bench_accum.py --device cuda --compile --check --ref | tee out/bench_accum.txt

Benchmark first: size each subset job from the measured MAC/s of the chosen MXU model in
`out/bench_accum.txt` (ipt, or SA compiled), so that it fits in about 1–2 h:

    N = floor(seconds * MAC_per_s / (MAC per window or frame))

TinyLlama is 0.97 GMAC per token (2048-token windows: 2.0e12); a Pi0 frame is 2.14e12; SmolVLA and
Multi-Task DiT: one frame is a forward of a few hundred tokens times the flow / denoising steps (time
one `reference_hw` frame and scale). Start with N = 4 if in doubt. `chip` costs about the same as
`reference_hw` on the MXU plus the VPU emulation; `chip_rope_vpu` the same as `chip`.

TinyLlama (WikiText-2; no Conv2d, so step 5 is not run):

    # steps 1-4 (fast), the whole split
    python src/eval.py --model $TL --protocol mxq --nsamples 0 --modes bf16 collab_v0 ref_linear ref_attn --rmse --out out/tl_ladder_fast.json
    # steps 6-8 on N seeded windows; --subset-compare adds bf16 and ref_conv (== ref_attn) on the same windows
    python src/eval.py --model $TL --protocol mxq --nsamples $N --modes reference_hw chip chip_rope_vpu --mxu ipt --out out/tl_ladder_subset.json
    # side profiles: the collaborator's protocol, the hardware cast, stock torchao
    python src/eval.py --model $TL --modes bf16 collab_v0 collab_v0_hw ref_conv torchao --out out/tl_fast_default.json

SmolVLA (lerobot/svla_so100_pickplace; F from the benchmark). `--norm-stats dataset` is required:
smolvla_base's saved processors hold no stats for its state / action keys, which the normalization
guard refuses (docs/algos/atlas_v1.md, "LeRobot normalization protocol"):

    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --norm-stats dataset --n-frames 200 --modes bf16 collab_v0 ref_linear ref_attn ref_conv --rmse --out out/smolvla_ladder_fast.json
    python src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace --norm-stats dataset --n-frames $F --modes reference_hw chip chip_rope_vpu --mxu ipt --out out/smolvla_ladder_subset.json

Multi-Task DiT. Steps 1-5 here with the generic rules: `ref_attn` patches only the CLIP vision
attention (DiT's own RoPEAttention is left unpatched, see `attention_sites`); the full ladder, with
DiT's attention and its
chip profile, is in multitask-dit-quant (git@github.com:lorenshung/multitask-dit-quant.git,
`experiments/run_eval.py --modes bf16 ref_linear ref_attn ref_conv reference_hw chip chip_rope_vpu`,
same step names; step 2 `collab_v0` only exists here):

    python src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 --dataset TRI-ML/PretrainFinetune --n-frames 200 --modes bf16 collab_v0 ref_linear ref_attn ref_conv --out out/dit_ladder_fast.json

Pi0: see [docs/pi0.md](docs/pi0.md#run-matrix-for-pi0-benchmark-first) (same ladder, `src/eval_pi0.py`).

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

Check every reference_hw and chip result before quoting it:

- `layers.<mode>.valid` is true;
- `crosscheck.ok` is true and `bit_mismatches` is 0;
- `reference_sources` records the pinned commits.

## More

- [docs/pi0.md](docs/pi0.md): Pi0 (openpi PyTorch Pi0): environment, checkpoint, eval, run matrix, findings.
- [docs/chip.md](docs/chip.md): the `chip` profile, its NEW decisions and the findings for the Atlas hardware team.
- [docs/reference_profile.md](docs/reference_profile.md), [docs/algos/](docs/algos/README.md): numerics and algorithms.
