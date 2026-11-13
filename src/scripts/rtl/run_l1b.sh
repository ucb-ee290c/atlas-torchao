#!/usr/bin/env bash
# Run AtlasMxuScriptTest (actual InnerProductTreesTop / SystolicArrayTop under Verilator) in an atlas-npu checkout.
#   run_l1b.sh <atlas-npu dir> <script.json[,script2.json...]> <trace_out.json[,trace2.json...]>
# Installs the harness into a temporary directory of the RTL test tree and removes it afterwards.
# No RTL source changes. Run one mill at a time. All scripts of one call must target the same unit (sa flag).
set -euo pipefail
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-17.0.20.1.1-1.2.el9.x86_64   # the login shell exports a stale JDK path
export PATH=$JAVA_HOME/bin:$PATH
RTL=$(readlink -f "$1"); SCRIPT=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
abspath() { echo "$1" | tr ',' '\n' | while read -r p; do readlink -f -m "$p"; done | paste -sd,; }
SCRIPT=$(abspath "$SCRIPT"); OUT=$(abspath "$OUT")
WORK=$(mktemp -d "$RTL/src/test/scala/atlas/atlas_torchao_l1b_XXXX")
trap 'rm -rf "$WORK"' EXIT
cp "$HERE/AtlasMxuScriptTest.scala" "$WORK/"
cd "$RTL"
MXU_SCRIPT="$SCRIPT" MXU_OUT="$OUT" ./mill --no-server atlas.test.testOnly atlas.mxu.AtlasMxuScriptTest
echo "wrote $OUT"
