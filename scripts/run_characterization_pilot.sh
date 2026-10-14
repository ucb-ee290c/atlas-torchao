#!/usr/bin/env bash
# NEW: one-task native SmolVLA export replay; no fitting or benchmark expansion.
set -euo pipefail
: "${MODEL_CONFIG:?Set the pinned local SmolVLA config.json path}"
: "${DATASET_ROOT:?Set the pinned local SO100 dataset snapshot directory}"
: "${EXPORT_DIR:?Set an existing verified SmolVLA export directory}"
: "${PILOT_DIR:?Set a new output directory, never a historical run directory}"
PILOT_PYTHON="${PILOT_PYTHON:-python}"
PILOT_SOURCE="$(dirname "$(dirname "$(readlink -f "$0")")")/src/characterize_actions.py"
test ! -e "$PILOT_DIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
"$PILOT_PYTHON" "$PILOT_SOURCE" preflight \
  --model-config "$MODEL_CONFIG" --dataset-root "$DATASET_ROOT" \
  --policy-id lerobot/smolvla_base --dataset-id lerobot/svla_so100_pickplace \
  --task-id 0 --episodes 10 --frames-per-episode 10 --seed 0 \
  --export-dir "$EXPORT_DIR" --out "$PILOT_DIR/input_manifest.json"
if [[ "${1:-}" == "--preflight-only" ]]; then exit 0; fi
"$PILOT_PYTHON" "$PILOT_SOURCE" record \
  --manifest "$PILOT_DIR/input_manifest.json" --export-dir "$EXPORT_DIR" \
  --dataset-root "$DATASET_ROOT" --model-root "$(dirname "$MODEL_CONFIG")" \
  --out-dir "$PILOT_DIR/paired" "$@"
"$PILOT_PYTHON" "$PILOT_SOURCE" analyze \
  --record-dir "$PILOT_DIR/paired" --out "$PILOT_DIR/summary.json"
