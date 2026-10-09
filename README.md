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
