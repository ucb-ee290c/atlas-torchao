#!/usr/bin/env bash
# Top-k Fisher eval for an LLM report: keep the top-k Fisher linear layers in bf16, the rest FP8, and
# measure how much of the FP8 loss increase that recovers (next-token NLL on held-out WikiText-2 windows).
#
#   bash src/scripts/run_topk_fisher_llm.sh                       # defaults below
#   GPU=3 N=32 bash src/scripts/run_topk_fisher_llm.sh --attn-fp8 # env overrides + extra eval_topk_fisher.py args
#
# Env: PYTHON (default: python from the active env, use atlas-torchao), GPU (CUDA_VISIBLE_DEVICES, default: unset),
#      FISHER (report JSON), N (held-out windows), KS (top-k list), RANDOM_DRAWS (random-layer chance baselines, default 0 = none; e.g. RANDOM_DRAWS=3 adds 3 sets for the largest k), OUT.
# Output: <fisher dir>/validation/topk_<fisher stem>.json, and the same text in a .log next to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
FISHER="${FISHER:-$REPO/src/out/fisher/tinyllama/tinyllama_n128.json}"
N="${N:-64}"
KS="${KS:-10 5 3}"
RANDOM_DRAWS="${RANDOM_DRAWS:-0}"
OUT="${OUT:-$(dirname "$FISHER")/validation/topk_$(basename "${FISHER%.json}").json}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$(dirname "$OUT")"
# shellcheck disable=SC2086   # KS is a space-separated list
"$PYTHON" "$REPO/src/scripts/eval_topk_fisher.py" \
    --fisher "$FISHER" --n "$N" --ks $KS --random "$RANDOM_DRAWS" --out "$OUT" "$@" \
    2>&1 | tee "${OUT%.json}.log"
