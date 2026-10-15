#!/usr/bin/env bash
set -euo pipefail

# Run only from the selected detached FX checkout. No source edits or upstream writes.
FX_SRC="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FX_COMMIT="${FX_COMMIT:?set the full commit from the GPU handoff}"
FX_EXPECTED_MANIFEST_SHA256="${FX_EXPECTED_MANIFEST_SHA256:?copy the full TinyLlama AWQ digest from run-5.3 exports_summary.json}"
FX_ROOT="${FX_ROOT:-/scratch/lsh/atlas-fp8}"
FX_BASE="${FX_BASE:-$FX_ROOT/release-check-pgu1ehd7/source}"
FX_EXPORT="${FX_EXPORT:-$FX_ROOT/exports/atlas_v1/tinyllama/atlas_v1_awq}"
FX_MODEL="${FX_MODEL:?set the absolute original TinyLlama configuration/checkpoint directory}"
FX_PY="${FX_PY:-$(command -v python)}"
FX_RUN="${FX_RUN:-$(dirname -- "$FX_SRC")}"
FX_OUTPUT="$FX_RUN/fx"
for FX_PATH in "$FX_SRC" "$FX_ROOT" "$FX_BASE" "$FX_EXPORT" "$FX_MODEL" "$FX_PY" "$FX_RUN"; do
  [[ "$FX_PATH" = /* ]] || { printf 'Use absolute paths: %s\n' "$FX_PATH" >&2; exit 2; }
done
test -x "$FX_PY"
test -f "$FX_EXPORT/manifest.json"
test -f "$FX_MODEL/config.json"
test ! -e "$FX_OUTPUT"
test "$(git -C "$FX_SRC" rev-parse HEAD)" = "$FX_COMMIT"
if git -C "$FX_SRC" symbolic-ref -q HEAD >/dev/null; then
  printf 'Use a new detached checkout, not an existing branch.\n' >&2
  exit 2
fi
test -z "$(git -C "$FX_SRC" status --porcelain)"
mkdir -p "$FX_RUN"
export FX_COMMIT FX_EXPECTED_MANIFEST_SHA256 FX_ROOT FX_BASE FX_EXPORT FX_MODEL FX_RUN FX_SRC FX_OUTPUT
export HF_HUB_OFFLINE=1

# Reuse the already pinned local reference repositories; never recurse into device models.
test "$(git -C "$FX_BASE/third_party/pi0-quant" rev-parse HEAD)" = 8bc0f6a6cb3faa237eb20b2354a0c8bc7afa524e
test "$(git -C "$FX_BASE/third_party/smolVLA-quant" rev-parse HEAD)" = 337363277c347927c45121a40c8ed049d72cff82
git -C "$FX_SRC" -c protocol.file.allow=always \
  -c "submodule.third_party/pi0-quant.url=$FX_BASE/third_party/pi0-quant" \
  -c "submodule.third_party/smolVLA-quant.url=$FX_BASE/third_party/smolVLA-quant" \
  submodule update --init --no-fetch third_party/pi0-quant third_party/smolVLA-quant
export PYTHONPATH="$FX_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

"$FX_PY" - <<'PY'
import json
import os
from pathlib import Path
import torch
import algos
import load_quantized
import quant_export

assert torch.cuda.is_available(), 'use the existing GPU atlas-lerobot environment'
assert torch.__version__.split('+')[0].startswith('2.11.'), 'CPU checks used torch 2.11; do not upgrade the existing environment'
directory = Path(os.environ['FX_EXPORT'])
expected_hash = os.environ['FX_EXPECTED_MANIFEST_SHA256']
assert len(expected_hash) == 64 and all(c in '0123456789abcdef' for c in expected_hash), 'use a full lowercase SHA256'
assert quant_export.sha256_file(directory / 'manifest.json') == expected_hash, 'source export is not the pinned historical AWQ export'
tensors, policy, manifest = load_quantized.read_export(directory)
assert (manifest.get('model') or {}).get('kind', 'causal_lm') == 'causal_lm'
assert manifest['recipe']['expanded'] == algos.expand('atlas_v1'), 'not the selected AWQ+GPTQ recipe; do not use historical atlas_v1 SmoothQuant'
assert manifest['counts'] == {'quantized': 176, 'host': 0, 'skipped': 1}, manifest['counts']
assert not policy.get('bf16_opt_out'), 'TinyLlama should have no BF16 exception'
linear = [site for site in policy['sites'].values() if site['status'] == 'quantized' and site['kind'] == 'linear']
assert len(linear) == 154
for site in linear:
    assert site['weight']['granularity'] == 'per_channel'
    assert site['activation']['mode'] == 'dynamic' and site['activation']['reduce'] == 'token'
    assert site['activation']['rule'] == 'nc'
runtime = policy.get('runtime', {}).get('chip_options', {})
assert runtime.get('chip_vadd') == 'rn', 'original chip policy must remain recorded even though FX uses ref_conv'
record = {'commit': os.environ['FX_COMMIT'], 'source_export': str(directory),
          'source_manifest_sha256': quant_export.sha256_file(directory / 'manifest.json'),
          'source_policy_sha256': quant_export.sha256_file(directory / 'policy.json'),
          'original_export_recipe': manifest['recipe']['algo'],
          'source_export_provenance': manifest.get('provenance'),
          'model_config_directory': os.environ['FX_MODEL'],
          'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
          'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
          'free_total_gpu_bytes': list(torch.cuda.mem_get_info()),
          'scope': 'full TinyLlama logits; ref_conv framework emulation, not native FP8 or chip execution'}
(Path(os.environ['FX_RUN']) / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
PY

printf 'FX_GPU_CHECK_STARTED\n  commit=%s\n  source=%s\n  source_export=%s\n  results=%s\n  fx=%s\n  pid=%s\n' \
  "$FX_COMMIT" "$FX_SRC" "$FX_EXPORT" "$FX_RUN" "$FX_OUTPUT" "$$"

cd "$FX_SRC"
"$FX_PY" -m pytest -q tests/test_fx_export.py > "$FX_RUN/pytest-fx-export.log" 2>&1
FX_ARGS=(--export-dir "$FX_EXPORT" --model "$FX_MODEL" --out-dir "$FX_OUTPUT" --device cuda --seqlen 8 --seed 0)
"$FX_PY" src/fx_export.py "${FX_ARGS[@]}" > "$FX_RUN/capture.log" 2>&1
# A further standalone process verifies the same saved artifact; no model-class imports.
"$FX_PY" src/fx_export.py --verify-dir "$FX_OUTPUT" --device cuda > "$FX_RUN/standalone-reload.log" 2>&1

"$FX_PY" - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import subprocess

root, output = Path(os.environ['FX_RUN']), Path(os.environ['FX_OUTPUT'])
record = json.loads((output / 'verification.json').read_text())
inspection = json.loads((output / 'graph_inspection.json').read_text())
inventory = json.loads((output / 'buffer_inventory.json').read_text())
independent = json.loads((output / 'fresh_process_verification.json').read_text())
preflight = json.loads((root / 'preflight.json').read_text())
assert record['status'] == 'passed' and record['strict'] is True and record['cases'] == 2
assert record['weight_sites_verified'] == inspection['weight_sites_verified'] == 154
assert len(inspection['functional_attention_sites']) == 22
assert 'not certified' in inspection['functional_attention_structural_binding']
assert len(inventory) == 308, len(inventory)
assert all(row['dynamic_scale_witnesses'] for row in inspection['weight_site_bindings'])
for check in (record['same_process_reload'], record['fresh_process_reload'], independent):
    assert check['status'] == 'passed' and check['cases'] == 2
    assert check['all_outputs_byte_identical'] and check['code_scale_bytes_preserved']
    assert check['unused_float_weights_not_materialized']
assert record['fresh_process_reload']['model_replay_modules_imported'] == []
assert independent['model_replay_modules_imported'] == []
assert record['native_fp8_mlir_verified'] is False and record['hardware_compatibility_verified'] is False
assert record['provenance']['capture_source']['commit'] == os.environ['FX_COMMIT']
assert record['provenance']['capture_source']['dirty'] == ''
assert record['provenance']['export_manifest_sha256'] == preflight['source_manifest_sha256']
assert subprocess.check_output(['git', '-C', os.environ['FX_SRC'], 'status', '--porcelain'], text=True) == ''
summary = {'status': 'passed', 'commit': os.environ['FX_COMMIT'], 'source': os.environ['FX_SRC'],
           'source_export': os.environ['FX_EXPORT'], 'results': str(root), 'fx': str(output),
           'model_pt2_sha256': record['model_pt2_sha256'], 'model_pt2_bytes': record['model_pt2_bytes'],
           'fx_manifest_sha256': hashlib.sha256((output / 'fx_manifest.json').read_bytes()).hexdigest(),
           'weight_sites_verified': 154, 'code_scale_buffers': 308,
           'functional_attention_sites': 22, 'functional_attention_structural_binding': 'not certified',
           'input_cases': 2, 'batch': 1, 'sequence_length': 8, 'strict_capture': True,
           'eager_captured_saved_outputs_byte_identical': True,
           'fresh_process_without_original_model_code': True,
           'native_fp8_mlir_verified': False, 'hardware_compatibility_verified': False,
           'source_export_manifest_sha256': preflight['source_manifest_sha256'],
           'pytest_log': str(root / 'pytest-fx-export.log'),
           'scope': 'full-checkpoint TinyLlama logits reference-emulation FX; no new accuracy benchmark'}
(root / 'gpu_fx_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print('FX_GPU_CHECK_DONE', flush=True)
print(json.dumps(summary, indent=2), flush=True)
PY
