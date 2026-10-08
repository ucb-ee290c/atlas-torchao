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