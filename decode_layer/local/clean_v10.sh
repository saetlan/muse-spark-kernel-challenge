#!/bin/sh
# Build the clean v10 with a lab's printed picks (V4_MODE, V10_PDL) and run its local checks (no mutation phase,
# user's call Nov 4 ~13:55 UTC). Usage: sh decode_layer/local/clean_v10.sh <profiles log name>
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
LOG="$1"
cd "$ROOT/decode_layer"
grep -q "V4_MODE = " "profiles/$LOG" && grep -q "V10_PDL = " "profiles/$LOG"
"$PY" local/make_v4.py --v10 --modes "$LOG" --pdl "$LOG"
head -1 submission_v10.py | grep -qx "# okc: problem=muse_decode_layer"
"$PY" local/lint_submission.py submission_v10.py
"$PY" -m py_compile submission_v10.py
"$PY" local/check_cuda_syntax.py > /dev/null
DRYRUN_PICKS_LOG="$LOG" "$PY" local/dryrun_v4.py submission_v10.py | tail -1
DRYRUN_PDL_LOG="$LOG" "$PY" local/dryrun_v10.py submission_v10.py | tail -1
grep -n -E "^V4_MODE = |^V10_PDL = " submission_v10.py | cut -c1-400
shasum -a 256 submission_v10.py | cut -c1-16
echo "clean v10: local checks passed ($LOG)"
