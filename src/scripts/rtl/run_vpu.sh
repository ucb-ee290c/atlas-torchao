#!/usr/bin/env bash
# Run AtlasVpuScriptTest (actual VectorEngineTop under Verilator) in an atlas-npu checkout.
#   run_vpu.sh <atlas-npu dir> <script.json> <out.json>
# Installs the harness into a temporary directory of the RTL test tree and removes it afterwards. No RTL changes.
set -euo pipefail
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-17.0.20.1.1-1.2.el9.x86_64   # the login shell exports a stale JDK path
export PATH=$JAVA_HOME/bin:$PATH
RTL=$(readlink -f "$1"); SCRIPT=$(readlink -f "$2"); OUT=$(readlink -f -m "$3")
HERE=$(cd "$(dirname "$0")" && pwd)
WORK=$(mktemp -d "$RTL/src/test/scala/atlas/atlas_torchao_vpu_XXXX")
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/AtlasVpuScriptTest.scala" "$WORK/"
cd "$RTL"
VPU_SCRIPT="$SCRIPT" VPU_OUT="$OUT" ./mill --no-server atlas.test.testOnly atlas.vector.AtlasVpuScriptTest
echo "wrote $OUT"
