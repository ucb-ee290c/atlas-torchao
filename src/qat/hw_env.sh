# Shared caches on /rscratch (the NFS home is full). Source before any src/qat/*hw* run.
export HF_HOME=/rscratch/agarcia/hf_cache
export HF_LEROBOT_HOME=/rscratch/agarcia/hf_cache/lerobot
export HF_DATASETS_CACHE=/rscratch/agarcia/hf_cache/datasets
export HF_TOKEN_PATH=/home/eecs/agarcia05/.cache/huggingface/token
export TORCHINDUCTOR_CACHE_DIR=/rscratch/agarcia/hf_cache/inductor
