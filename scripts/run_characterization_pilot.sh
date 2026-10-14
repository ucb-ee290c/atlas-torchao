#!/usr/bin/env bash
# Pinned one-task native SmolVLA replay. No fitting, hardware timing or upstream writes.
set -euo pipefail
PILOT_CHECKOUT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
: "${MODEL_CONFIG:?Set the pinned local SmolVLA config.json path}"
: "${DATASET_ROOT:?Set the pinned local SO100 dataset snapshot directory}"
: "${EXPORT_DIR:?Set an existing verified SmolVLA export directory}"
: "${PILOT_DIR:?Set a new output directory, never a historical run directory}"
: "${EXPORT_MANIFEST_SHA256:?Copy the full SHA256 from trusted run-5.3 exports_summary.json}"
: "${PILOT_EXPECTED_COMMIT:?Set the full source commit from the characterization handoff}"
: "${PILOT_PYTHON:?Set the absolute existing environment Python path}"
PILOT_MODE=all
case "${1:-}" in
  --preflight-only) PILOT_MODE=preflight; shift ;;
  --record-only) PILOT_MODE=record; shift ;;
  --report-only) PILOT_MODE=report; shift ;;
esac
for PILOT_ARG in "$@"; do
  case "$PILOT_ARG" in
    --device|--trace-sites|--trace-samples|--trace-frames-per-episode|--trace-max-records|\
    --device=*|--trace-sites=*|--trace-samples=*|--trace-frames-per-episode=*|--trace-max-records=*) ;;
    -*) printf 'Only device/trace options may be forwarded; refused: %s\n' "$PILOT_ARG" >&2; exit 2 ;;
    *) ;;
  esac
done
for PILOT_PATH in "$MODEL_CONFIG" "$DATASET_ROOT" "$EXPORT_DIR" "$PILOT_DIR" "$PILOT_PYTHON"; do
  [[ "$PILOT_PATH" = /* ]] || { printf 'Use absolute paths: %s\n' "$PILOT_PATH" >&2; exit 2; }
done
[[ "$PILOT_EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || { printf 'Use a full source commit.\n' >&2; exit 2; }
test -x "$PILOT_PYTHON"
test -f "$MODEL_CONFIG"
test -d "$DATASET_ROOT"
test "$(git -C "$PILOT_CHECKOUT" rev-parse HEAD)" = "$PILOT_EXPECTED_COMMIT"
test -z "$(git -C "$PILOT_CHECKOUT" status --porcelain)"
if git -C "$PILOT_CHECKOUT" symbolic-ref -q HEAD >/dev/null; then
  printf 'Use a clean detached source checkout, not an active branch.\n' >&2; exit 2
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONPATH="$PILOT_CHECKOUT/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
PILOT_SOURCE="$PILOT_CHECKOUT/src/characterize_actions.py"
PILOT_REPORT="$PILOT_CHECKOUT/src/characterize_report.py"
if [[ "$PILOT_MODE" == all || "$PILOT_MODE" == preflight ]]; then
  test ! -e "$PILOT_DIR"
  mkdir -p "$PILOT_DIR"
else
  test -f "$PILOT_DIR/input_manifest.json"
fi
PILOT_LOGDIR="$(mktemp -d "$PILOT_DIR/starter-logs-XXXXXX")"
trap 'PILOT_STATUS=$?; if [[ "$PILOT_STATUS" != 0 ]]; then printf "PILOT_STAGE_FAILED status=%s logs=%s\n" "$PILOT_STATUS" "$PILOT_LOGDIR" >&2; fi' EXIT
"$PILOT_PYTHON" "$PILOT_SOURCE" verify-export-pin --export-dir "$EXPORT_DIR" \
  --expected-sha256 "$EXPORT_MANIFEST_SHA256" --out "$PILOT_LOGDIR/export_pin.json" \
  > "$PILOT_LOGDIR/export-pin.log" 2>&1
if [[ "$PILOT_MODE" == all || "$PILOT_MODE" == preflight ]]; then
  "$PILOT_PYTHON" "$PILOT_SOURCE" preflight \
  --model-config "$MODEL_CONFIG" --dataset-root "$DATASET_ROOT" \
  --policy-id lerobot/smolvla_base --dataset-id lerobot/svla_so100_pickplace \
  --task-id 0 --episodes 10 --frames-per-episode 10 --seed 0 \
  --export-dir "$EXPORT_DIR" --out "$PILOT_DIR/input_manifest.json" > "$PILOT_LOGDIR/preflight.log" 2>&1
fi
export PILOT_DIR EXPORT_MANIFEST_SHA256 PILOT_EXPECTED_COMMIT PILOT_MODE PILOT_LOGDIR PILOT_CHECKOUT
"$PILOT_PYTHON" - <<'PY' > "$PILOT_LOGDIR/frozen-contract.log" 2>&1
import json, os
from pathlib import Path
from characterize_actions import canonical_digest, write_new
m = json.loads((Path(os.environ['PILOT_DIR']) / 'input_manifest.json').read_text())
assert canonical_digest({k: v for k, v in m.items() if k != 'content_sha256'}) == m['content_sha256']
assert m['calibration']['status'] == 'existing_export'
assert m['export']['manifest_sha256'] == os.environ['EXPORT_MANIFEST_SHA256']
assert m['sampling']['chunks'] == len(m['observations']) == 100
assert len(m['sampling']['episodes']) == 10 and m['sampling']['frames_per_episode'] == 10
assert (m['action_contract']['returned_horizon'], m['action_contract']['dimension']) == (50, 6)
assert m['task']['id'] == 0
write_new(Path(os.environ['PILOT_LOGDIR']) / 'job_manifest.json', {
 'source_commit': os.environ['PILOT_EXPECTED_COMMIT'], 'source': os.environ['PILOT_CHECKOUT'],
 'mode': os.environ['PILOT_MODE'], 'frozen_content_sha256': m['content_sha256'],
 'export_manifest_sha256': os.environ['EXPORT_MANIFEST_SHA256'], 'native_shape': [50, 6], 'chunks': 100,
 'scope': 'same-input offline native fidelity, not XYZ/closed-loop/performance qualification'})
print(json.dumps({'frozen_content_sha256': m['content_sha256'], 'chunks': 100, 'native_shape': [50, 6],
                  'calibration_scope': 'disjointness remains provisional until original data digest passes'}))
PY
printf 'CHARACTERIZATION_STARTED\n  commit=%s\n  mode=%s\n  source=%s\n  results=%s\n  logs=%s\n' \
  "$PILOT_EXPECTED_COMMIT" "$PILOT_MODE" "$PILOT_CHECKOUT" "$PILOT_DIR" "$PILOT_LOGDIR"
if [[ "$PILOT_MODE" == preflight ]]; then
  printf 'PILOT_PREFLIGHT_DONE manifest=%s\n' "$PILOT_DIR/input_manifest.json"; exit 0
fi
if [[ "$PILOT_MODE" == all || "$PILOT_MODE" == record ]]; then
  test ! -e "$PILOT_DIR/paired"
  test ! -e "$PILOT_DIR/summary.json"
  test ! -e "$PILOT_DIR/report"
  "$PILOT_PYTHON" "$PILOT_SOURCE" record \
  --manifest "$PILOT_DIR/input_manifest.json" --export-dir "$EXPORT_DIR" \
  --dataset-root "$DATASET_ROOT" --model-root "$(dirname "$MODEL_CONFIG")" \
  --out-dir "$PILOT_DIR/paired" "$@" > "$PILOT_LOGDIR/record.log" 2>&1
fi
test ! -e "$PILOT_DIR/summary.json"
test ! -e "$PILOT_DIR/report"
"$PILOT_PYTHON" - <<'PY' > "$PILOT_LOGDIR/record-binding.log" 2>&1
import os
from pathlib import Path
from characterize_actions import digest, load_verified_recording
root = Path(os.environ['PILOT_DIR'])
_, header, _, _ = load_verified_recording(root / 'paired')
assert header['input_manifest_sha256'] == digest(root / 'input_manifest.json'), 'record/frozen manifest mismatch'
assert header['export_manifest_sha256'] == os.environ['EXPORT_MANIFEST_SHA256'], 'record/export pin mismatch'
assert header['source_commit'] == os.environ['PILOT_EXPECTED_COMMIT'], 'record/source commit mismatch'
print('recording bound to this exact frozen job')
PY
"$PILOT_PYTHON" "$PILOT_SOURCE" analyze \
  --record-dir "$PILOT_DIR/paired" --out "$PILOT_DIR/summary.json" > "$PILOT_LOGDIR/analyze.log" 2>&1
"$PILOT_PYTHON" "$PILOT_REPORT" --record-dir "$PILOT_DIR/paired" --out-dir "$PILOT_DIR/report" \
  > "$PILOT_LOGDIR/report.log" 2>&1
printf 'CHARACTERIZATION_PILOT_DONE\n  commit=%s\n  results=%s\n  report=%s\n  logs=%s\n' \
  "$PILOT_EXPECTED_COMMIT" "$PILOT_DIR" "$PILOT_DIR/report/report_manifest.json" "$PILOT_LOGDIR"
