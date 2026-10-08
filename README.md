# atlas-torchao

## Setup

    export HF_HOME=/rscratch/$USER/hf_cache     # keep model downloads out of your home quota
    conda env create -f environment.yml
    conda activate atlas
    git submodule update --init                 # model2MLIR
    pip install -e model2MLIR

### Why these versions are pinned
- lerobot 0.4.4 (Python 3.10) requires torch < 2.11, so torch is 2.10.
- torchao must be 0.15.0: newer versions remove classes that diffusers 0.35
  (pulled in by lerobot) imports, which crashes `import lerobot`.
- Do not `pip install -U torchao` in this environment.
- The "Skipping import of cpp extensions" warning from torchao is expected and harmless.
- Create conda envs on /scratch or /rscratch, not your home directory
  (home quota is small; a full disk corrupts installs).