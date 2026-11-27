# User guide: from a checkpoint to a quantized export

This guide takes you by hand from a fresh clone to an Atlas FP8 export: `model.safetensors` (E4M3 codes and
power-of-two exponents), `policy.json` (how each site executes) and `manifest.json` (hashes, provenance, recipe).
Format details: [algos/export.md](algos/export.md). Recipes: [algos/atlas_v1.md](algos/atlas_v1.md),
[algos/tile-aware.md](algos/tile-aware.md).

All commands run from the repository root. Sections 1-4 and 6 were run as written on a CPU-only Linux box
(2026-11-27, main `50d6c30`); times are from that box under load.

## 1. Setup (once per machine)

```bash
git clone git@github.com:ucb-ee290c/atlas-torchao.git && cd atlas-torchao
git submodule update --init third_party/pi0-quant third_party/smolVLA-quant   # not --recursive
conda create -y -p /scratch/$USER/atlas-env python=3.12 pip                    # on /scratch, not $HOME
conda activate /scratch/$USER/atlas-env
pip install -e .[reference,policy] pytest       # torch 2.11, transformers 5.22.2, torchao 0.18, numba, lerobot 0.7
```

`conda env create -p /scratch/$USER/atlas-env -f environment.yml` does the same in one step, and
`python3.12 -m venv` works too if python3.12 is on your PATH.

Every shell that runs the commands below needs:

```bash
export PYTHONPATH=src
export HF_HOME=/scratch/$USER/hf_home           # the Hub cache: models and datasets land here
```

Check the install (about 40 minutes on CPU; GPU tests skip without a GPU):

```bash
python -m pytest -q tests      # 0 failed
```

## 2. Smoke test (about 1 minute, CPU)

A 2-layer random Llama runs the whole path before you spend time on a real model.

```bash
export MODEL=$(hf download --quiet TinyLlama/TinyLlama-1.1B-Chat-v1.0)   # 2.2 GB; prints the local snapshot path
export TINY=/scratch/$USER/tiny-llama SMOKE=/scratch/$USER/exports/smoke

python src/scripts/tile_flow/make_tiny_llama.py $TINY $MODEL
python src/eval.py --model $TINY --seqlen 128 --calib-n 4 --modes ref_conv --export-only \
    --algo atlas_v1_awq_tile_static --device cpu --export-dir $SMOKE/export
# prints: export: wrote .../export (7.9 MiB, 16 quantized sites)
ls $SMOKE/export          # manifest.json  model.safetensors  policy.json
```

## 3. Export TinyLlama

```bash
git status --short        # should be empty: the manifest records the commit and a dirty flag
export OUT=/scratch/$USER/exports/tinyllama/atlas_v1_awq_tile_static      # must not exist yet
python -u src/eval.py --model $MODEL --modes ref_conv --export-only --algo atlas_v1_awq_tile_static \
    --device cpu --export-dir $OUT/export --out $OUT/export_cli.json
# prints: export: wrote .../export (1175.7 MiB, 176 quantized sites)       about 28 min on CPU
```

- `--algo` picks the recipe. Name it explicitly, never the `atlas_v1` alias (its meaning changed between tags).
  `atlas_v1_awq_tile_static` is the tile-aware, static-activation recipe the current flow uses; the others are
  listed in [atlas_v1.md](algos/atlas_v1.md#the-recipes).
- Calibration defaults to 16 WikiText-2 train windows of 2048 tokens, seed 1234 (`--calib-n`, `--calib-seed`).
- `--export-only` quantizes, writes the export and stops (no evaluation).
- Use `--device cuda` on a GPU machine.

## 4. Check the export

Re-evaluate it, rebuilt from the files alone (no calibration, no float weights of quantized layers):

```bash
python src/eval.py --model $MODEL --protocol mxq --nsamples 4 --modes bf16 ref_conv \
    --from-export $OUT/export --device cpu --out $OUT/reeval.json      # about 12 min on CPU
```

Expected on 4 windows: bf16 perplexity 8.06, ref_conv 8.18. The loader refuses any file whose sha256 does not
match the manifest. In `$OUT/export/manifest.json`, check:

- `provenance.atlas_torchao_dirty` is false and `provenance.atlas_torchao_commit` is the commit you ran;
- `recipe.resolved` is the recipe you meant;
- `calibration` is the data you expected;
- `counts`: `quantized` 176 for TinyLlama.

From Python (same shell, so `MODEL` and `OUT` are set):

```python
import os
from exporter.load_quantized import load_quantized
model, report, manifest = load_quantized(os.environ["MODEL"], os.environ["OUT"] + "/export", "cpu")
```

## 5. Other models

The same flags (`--algo`, `--modes ref_conv --export-only`, `--export-dir`) work on the other CLIs. LeRobot
policies come with the `policy` extra installed above.

```bash
# SmolVLA (--norm-stats dataset is required)
python -u src/eval_policy.py --policy lerobot/smolvla_base --dataset lerobot/svla_so100_pickplace \
    --norm-stats dataset --seed 0 --modes ref_conv --export-only --algo atlas_v1_awq_tile_static \
    --device cuda --export-dir /scratch/$USER/exports/smolvla/atlas_v1_awq_tile_static/export
# Multi-Task DiT (keeps the image-input Conv2d in BF16)
python -u src/eval_policy.py --policy TRI-ML/multitask-dit-robosuite-wrist-diffusion-064000 \
    --dataset TRI-ML/PretrainFinetune --seed 0 --modes ref_conv --export-only \
    --algo "atlas_v1_awq_tile_static+keep_bf16:rule=input_conv" --device cuda \
    --export-dir /scratch/$USER/exports/dit/atlas_v1_awq_tile_static/export
```

Pi0 needs openpi's separate environment (torch 2.7.1, transformers 4.53.2); its setup and commands are in
[pi0.md](pi0.md).

## 6. Next steps

FX graph bundle for the compiler (`model.pt2`, native-FP8 nodes), then the bundle check. On the smoke export:

```bash
python -m exporter.fx_export --export-dir $SMOKE/export --model $TINY --out-dir $SMOKE/fx --device cpu --seqlen 64 --native-fp8
python src/scripts/tile_flow/verify_bundle.py $SMOKE/export $SMOKE/fx $SMOKE/verify_bundle.json   # "ok": true
```

Details: [algos/fx-export.md](algos/fx-export.md), [algos/tile-aware.md](algos/tile-aware.md). Accuracy through the
hardware models (`reference_hw`, `chip`): the README's coverage ladder.

## Troubleshooting

- `Failed to load ... torchao/_C_*.so` at startup: harmless, those are optional torchao kernels.
- "unauthenticated requests to the HF Hub": harmless; `export HF_HUB_OFFLINE=1` once everything is cached.
- The export directory and the FX out directory must not exist; nothing is ever overwritten.
- `model.safetensors` is written with mode 0600; `chmod a+r` it before sharing an export.
- SmolVLA refuses to run: add `--norm-stats dataset`.
