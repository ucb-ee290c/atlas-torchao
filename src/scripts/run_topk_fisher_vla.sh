#!/usr/bin/env bash
# Top-k Fisher eval for a VLA (SmolVLA) report: keep the top-k Fisher linear layers in bf16, the rest FP8,
# and measure how much of the FP8 action-chunk change that recovers (relative L2 of the generated action
# chunk from the bf16 chunk, same initial noise per frame; --vla-metric flow_loss selects the training loss).
#
#   bash src/scripts/run_topk_fisher_vla.sh                       # defaults below
#   GPU=1 N=16 DRAWS=2 bash src/scripts/run_topk_fisher_vla.sh    # env overrides + extra eval_topk_fisher.py args
#
# Env: PYTHON (default: python from the active env, use atlas-torchao: needs lerobot), GPU (CUDA_VISIBLE_DEVICES,
#      default: unset), FISHER (report JSON), N (held-out frames), DRAWS (noise draws per frame, default 1),
#      KS (top-k list), RANDOM_DRAWS (random-layer chance baselines, default 0 = none; e.g. RANDOM_DRAWS=3 adds 3 sets for the largest k), NORM (--norm-stats, default dataset), OUT.
# The VLA settings (--norm-stats, --n-test-episodes, --eval-seed) must match the profile run.
# Each config takes ~5 s per frame-draw, so the defaults (32 frames, 1 draw, no random sets: 5 configs) are ~10-15 minutes.
# Output: <fisher dir>/validation/topk_<fisher stem>.json, and the same text in a .log next to it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-python}"
FISHER="${FISHER:-$REPO/src/out/fisher/smolvla/smolvla_n8.json}"
N="${N:-32}"
KS="${KS:-10 5 3}"
RANDOM_DRAWS="${RANDOM_DRAWS:-0}"
NORM="${NORM:-dataset}"
OUT="${OUT:-$(dirname "$FISHER")/validation/topk_$(basename "${FISHER%.json}").json}"
[[ -n "${GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$GPU"

mkdir -p "$(dirname "$OUT")"
DRAWS="${DRAWS:-1}"
# shellcheck disable=SC2086   # KS is a space-separated list
"$PYTHON" "$REPO/src/scripts/eval_topk_fisher.py" \
    --fisher "$FISHER" --n "$N" --ks $KS --random "$RANDOM_DRAWS" --norm-stats "$NORM" \
    --draws "$DRAWS" --out "$OUT" "$@" \
    2>&1 | tee "${OUT%.json}.log"
