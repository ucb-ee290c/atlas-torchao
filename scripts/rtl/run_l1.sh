#!/usr/bin/env bash
# Run AtlasTorchaoStimTest under Verilator (Chisel EphemeralSimulator) in an atlas-npu checkout.
#   run_l1.sh <atlas-npu dir> <stimuli.json> <rtl_out.json>
# Installs the harness into a temporary directory of the RTL test tree (as npu-model's
# regenerate_rtl_fixtures.py does for its harnesses) and removes it afterwards. No RTL source changes.
set -euo pipefail
RTL=$(readlink -f "$1"); STIM=$(readlink -f "$2"); OUT=$(readlink -f -m "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d "$RTL/src/test/scala/atlas/atlas_torchao_l1_XXXX")
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/AtlasTorchaoStimTest.scala" "$WORK/"
cd "$RTL"
L1_STIM="$STIM" L1_OUT="$OUT" ./mill --no-server atlas.test.testOnly atlas.mxu.AtlasTorchaoStimTest
echo "wrote $OUT"
