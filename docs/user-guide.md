# User guide: from a checkpoint to a quantized export

This guide takes you by hand from a fresh clone to an Atlas FP8 export: `model.safetensors` (E4M3 codes and
power-of-two exponents), `policy.json` (how each site executes) and `manifest.json` (hashes, provenance, recipe).
Format details: [algos/export.md](algos/export.md). Recipes: [algos/atlas_v1.md](algos/atlas_v1.md),
[algos/tile-aware.md](algos/tile-aware.md).

All commands run from the repository root. Sections 1-6 (all but the Multi-Task DiT export) were run in a fresh clone
and environment on a CPU-only Linux box (2026-11-27); times are ranges seen on that shared box. The smoke test (section
2) and the FX bundle and check (section 6) were run again with the `atlas` command, in a fresh environment, on
2026-11-27.

One command, `atlas`, does the main jobs; each subcommand runs an existing command unchanged, and every option of that
command works after the positional arguments ([equivalent commands](#equivalent-commands)):

| job | command |
|---|---|
| quantize and export | `atlas export MODEL EXPORT_DIR [--algo RECIPE] [options]` |
| evaluate, or re-evaluate an export | `atlas eval MODEL --modes ... [--from-export EXPORT_DIR] [options]` |
| FX graph bundle (native FP8) | `atlas fx EXPORT_DIR FX_DIR [options]` |
| check an export + FX bundle | `atlas verify EXPORT_DIR FX_DIR` |

Help: `atlas --help`, `atlas export --help`, and `atlas export MODEL EXPORT_DIR --help` for every option of the model's
script. `python -m atlas ...` is the same without the installed command.

## 1. Setup (once per machine)

```bash
git clone git@github.com:ucb-ee290c/atlas-torchao.git && cd atlas-torchao
git submodule update --init third_party/pi0-quant third_party/smolVLA-quant   # not --recursive
conda create -y -p /scratch/$USER/atlas-env python=3.12 pip                    # on /scratch, not $HOME
conda activate /scratch/$USER/atlas-env          # if this fails: eval "$(conda shell.bash hook)" first
pip install -e .[reference,policy] pytest       # torch 2.11, transformers 5.22.2, torchao 0.18, numba, lerobot 0.7, `atlas`
```

`conda env create -p /scratch/$USER/atlas-env -f environment.yml` does the same in one step, and
`python3.12 -m venv` works too if python3.12 is on your PATH.

Every shell that runs the commands below needs:

```bash
export PYTHONPATH=src
export HF_HOME=/scratch/$USER/hf_home           # the Hub cache: models and datasets land here
```

Check the install (40-70 minutes on CPU; GPU tests skip without a GPU):

```bash
python -m pytest -q tests > pytest.log 2>&1; grep -E "[0-9]+ (passed|failed)" pytest.log   # 0 failed
```

The summary line is not the last line: the chip tests print their coverage tables at exit.

## 2. Smoke test (about 1 minute, CPU)

A 2-layer random Llama runs the whole path before you spend time on a real model.

```bash
export MODEL=$(hf download --quiet TinyLlama/TinyLlama-1.1B-Chat-v1.0)   # 2.2 GB; prints the local snapshot path
export TINY=/scratch/$USER/tiny-llama SMOKE=/scratch/$USER/exports/smoke

python src/scripts/tile_flow/make_tiny_llama.py $TINY $MODEL
atlas export $TINY $SMOKE/export --seqlen 128 --calib-n 4 --device cpu
# prints: export: wrote .../export (7.9 MiB, 16 quantized sites)
ls $SMOKE/export          # manifest.json  model.safetensors  policy.json
```

`atlas export` first prints the command it runs:
`python src/eval.py --model $TINY --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --out
$SMOKE/export_cli.json --export-dir $SMOKE/export --seqlen 128 --calib-n 4 --device cpu`. The model family (here an
LLM, so `src/eval.py`) comes from the model's `config.json`; see [equivalent commands](#equivalent-commands).

## 3. Export TinyLlama

```bash
git status --short        # should be empty: the manifest records the commit and a dirty flag
export OUT=/scratch/$USER/exports/tinyllama/atlas_v1_awq_tile_static      # must not exist yet
atlas export $MODEL $OUT/export --device cpu
# prints: export: wrote .../export (1175.7 MiB, 176 quantized sites)       30-45 min on CPU, ~6 min on a GPU
```

- `--algo` picks the recipe; `atlas export` defaults to `atlas_v1_awq_tile_static`, the tile-aware, static-activation
  recipe the current flow uses. Name any other recipe explicitly, never the `atlas_v1` alias (its meaning changed
  between tags); they are listed in [atlas_v1.md](algos/atlas_v1.md#the-recipes).
- The run's record (arguments, layers, recipe) goes to `$OUT/export_cli.json` (`--out` changes it).
- Calibration defaults to 16 WikiText-2 train windows of 2048 tokens, seed 1234 (`--calib-n`, `--calib-seed`).
- `atlas export` quantizes, writes the export and stops (no evaluation): it adds `--modes ref_conv --export-only`.
- Use `--device cuda` on a GPU machine.

## 4. Check the export

Re-evaluate it, rebuilt from the files alone (no calibration, no float weights of quantized layers):

```bash
atlas eval $MODEL --protocol mxq --nsamples 4 --modes bf16 ref_conv \
    --from-export $OUT/export --device cpu --out $OUT/reeval.json      # about 12 min on CPU
```

Expected on 4 windows: bf16 perplexity 8.06, ref_conv 8.18. The loader refuses any file whose sha256 does not
match the manifest. In `$OUT/export/manifest.json`, check:

- `provenance.atlas_torchao_dirty` is false and `provenance.atlas_torchao_commit` is the commit you ran;
- `recipe.resolved` is the recipe you meant;
- `calibration` is the data you expected;
- `counts`: `quantized` 176 for TinyLlama (154 Linear + 22 attention sites; the loader's `report` counts the 154
  Linear sites only).

From Python (same shell, so `MODEL` and `OUT` are set):

```python
import os
from exporter.load_quantized import load_quantized
model, report, manifest = load_quantized(os.environ["MODEL"], os.environ["OUT"] + "/export", "cpu")
```

## 5. Other models

`atlas export` takes a LeRobot policy the same way; `--dataset` (required for a policy) selects
`src/eval_policy.py`. LeRobot policies come with the `policy` extra installed above. The SmolVLA export warns that `model.vlm_with_expert.vlm.lm_head` has no calibration data: expected, the policy
never calls the VLM's `lm_head` (lerobot freezes it as unused).

```bash
export EXP=/scratch/$USER/exports/atlas_v1_awq_tile_static
# SmolVLA (--norm-stats dataset is required); prints 476.9 MiB, 317 quantized sites
atlas export lerobot/smolvla_base $EXP/smolvla/export --dataset lerobot/svla_so100_pickplace \
    --norm-stats dataset --seed 0 --device cuda                                          # ~6 min on CPU
# Multi-Task DiT (keeps the image-input Conv2d in BF16)
atlas export TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 $EXP/dit/export \
    --dataset TRI-ML/PretrainFinetune --seed 0 --algo "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv" --device cuda
```

A DroNet weights file (`.npz`, from `python -m dronet.convert_keras`) selects `src/eval_dronet.py`:
`atlas export W.npz EXPORT_DIR --calib-cache VAL.npz` (its docstring has the details).

Pi0 needs openpi's separate environment (torch 2.7.1, transformers 4.53.2) and is not an `atlas` subcommand; its
setup and commands (`src/eval_pi0.py`) are in [pi0.md](pi0.md).

## 6. Next steps

FX graph bundle for the compiler (`model.pt2`, native-FP8 nodes), then the bundle check. On the smoke export:

```bash
atlas fx $SMOKE/export $SMOKE/fx --device cpu --seqlen 64       # native FP8 (--no-native-fp8: the decode graph)
atlas verify $SMOKE/export $SMOKE/fx                            # "ok": true; writes $SMOKE/fx_verify_bundle.json
```

`atlas fx` reads the architecture config from the export's model id (`--model` overrides it); `atlas fx --verify-dir
$SMOKE/fx --device cpu` replays a saved bundle in a fresh process.

Details: [algos/fx-export.md](algos/fx-export.md), [algos/tile-aware.md](algos/tile-aware.md). Accuracy through the
hardware models (`reference_hw`, `chip`): the README's coverage ladder.

## Equivalent commands

Each `atlas` command is a translation into the command below, which then runs unchanged, in the same process, as if it
had been typed (same parser, same code, same files: `model.safetensors` and `policy.json` are byte-identical). Those
commands keep working as before. `--dry-run` prints the translation without running it.

| `atlas` | runs (from the repository root, `PYTHONPATH=src`) |
|---|---|
| `atlas export MODEL DIR` (LLM) | `python src/eval.py --model MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --out DIR_cli.json --export-dir DIR` |
| `atlas export MODEL DIR --dataset D` | `python src/eval_policy.py --policy MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --out DIR_cli.json --export-dir DIR --dataset D` |
| `atlas export W.npz DIR` | `python src/eval_dronet.py --weights W.npz --modes ref_conv --export-only --algo atlas_v1_awq_tile_static --out DIR_cli.json --export-dir DIR` |
| `atlas eval MODEL ...` | the same script with `--model` / `--policy` / `--weights` MODEL and nothing added |
| `atlas fx EXPORT FX` | `python -m exporter.fx_export --export-dir EXPORT --out-dir FX --native-fp8` |
| `atlas fx --verify-dir FX` | `python -m exporter.fx_export --verify-dir FX` |
| `atlas verify EXPORT FX [OUT]` | `python src/scripts/tile_flow/verify_bundle.py EXPORT FX OUT` (default `FX_verify_bundle.json`) |

- Options after the positional arguments are passed on verbatim, after the added defaults, so an explicit `--modes`,
  `--algo`, `--out` or `--export-only` replaces the default.
- The family is `--family llm|policy|dronet`, or else: a `.npz` MODEL is DroNet, `--dataset` means a LeRobot policy,
  otherwise the model's `config.json` (a local directory or the Hugging Face cache; never the network) decides. If
  none of these applies, `atlas` asks for `--family`.
- The manifest's `provenance.argv` records the equivalent command above.
- Not covered, use the scripts: Pi0 (`src/eval_pi0.py`, openpi's environment), `make_tiny_llama.py` and the per-phase
  capture and loop-manifest scripts in `src/scripts/tile_flow/` ([tile-aware.md](algos/tile-aware.md)).

## Troubleshooting

- `Failed to load ... torchao/_C_*.so` at startup: harmless, those are optional torchao kernels.
- "unauthenticated requests to the HF Hub": harmless; `export HF_HUB_OFFLINE=1` once everything is cached.
- The export directory and the FX out directory must not exist; nothing is ever overwritten.
- "TorchCodec could not load the FFmpeg libraries ... Falling back to 'pyav'": harmless, the video frames decode with
  pyav instead.
- SmolVLA refuses to run: add `--norm-stats dataset`.
- `atlas: error: cannot tell the model family`: the model is neither a local directory nor in the Hugging Face cache
  yet; add `--family llm` (or `policy`, `dronet`), or `hf download` it first.
- `atlas` not found: `pip install -e .` installs it; `python -m atlas` works with `PYTHONPATH=src` alone.
