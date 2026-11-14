#!/bin/sh
# Build the clean v12 with a lab's printed picks (V4_MODE, V10_PDL, V12_BOUNDARY: each only if the log printed it) and
# run its local checks (no mutation phase, user's call Nov 4 ~13:55 UTC). Usage: sh decode_layer/local/clean_v12.sh <log>
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
LOG="$1"
cd "$ROOT/decode_layer"
FLAGS=""
grep -q "V4_MODE = " "profiles/$LOG" && FLAGS="$FLAGS --modes $LOG"
grep -q "V10_PDL = " "profiles/$LOG" && FLAGS="$FLAGS --pdl $LOG"
grep -q "V12_BOUNDARY = " "profiles/$LOG" && FLAGS="$FLAGS --boundary $LOG"
echo "make_v4.py --v12$FLAGS"
# shellcheck disable=SC2086
"$PY" local/make_v4.py --v12 $FLAGS
head -1 submission_v12.py | grep -qx "# okc: problem=muse_decode_layer"
"$PY" local/lint_submission.py submission_v12.py
"$PY" -m py_compile submission_v12.py
"$PY" local/check_cuda_syntax.py > /dev/null
MODES_LOG=$(grep -q "V4_MODE = " "profiles/$LOG" && echo "$LOG" || echo lab_de12.8ca776512a3b4b7a.log)
DRYRUN_PICKS_LOG="$MODES_LOG" "$PY" local/dryrun_v4.py submission_v12.py | tail -1
if grep -q "V10_PDL = " "profiles/$LOG"; then DRYRUN_PDL_LOG="$LOG" "$PY" local/dryrun_v10.py submission_v12.py | tail -1
else "$PY" local/dryrun_v10.py submission_v12.py | tail -1; fi
if grep -q "V12_BOUNDARY = " "profiles/$LOG"; then DRYRUN_V12_LOG="$LOG" "$PY" local/dryrun_v12.py submission_v12.py | tail -1
else "$PY" local/dryrun_v12.py submission_v12.py | tail -1; fi
grep -n -E "^V4_MODE = |^V10_PDL = |^V12_BOUNDARY = " submission_v12.py | cut -c1-400
shasum -a 256 submission_v12.py | cut -c1-16
echo "clean v12: local checks passed ($LOG)"
