#!/bin/sh
# Every local check, before any okc check or submit. Usage: sh moe/local/presubmit.sh moe/submission.py [more files]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/moe/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/moe/local/check_v0_lab.py"
"$PY" "$ROOT/moe/local/check_v1_vs_baseline.py" > /dev/null
"$PY" "$ROOT/moe/local/test_dequant_trick.py" > /dev/null
"$PY" "$ROOT/moe/local/check_m2.py" > /dev/null
"$PY" "$ROOT/moe/local/dryrun_lab_e2.py" > /dev/null
"$PY" "$ROOT/moe/local/check_cuda_syntax.py" > /dev/null
"$PY" "$ROOT/moe/local/check_m11.py" > /dev/null
"$PY" "$ROOT/moe/local/dryrun_lab_e4.py" > /dev/null
"$PY" "$ROOT/moe/local/check_cuda_syntax.py" "$ROOT/moe/submission_m13.py" > /dev/null
"$PY" "$ROOT/moe/local/check_m13.py" > /dev/null
"$PY" "$ROOT/moe/local/check_m16.py" > /dev/null
"$PY" "$ROOT/moe/local/check_m18.py" > /dev/null
"$PY" "$ROOT/moe/local/dryrun_lab_e5.py" > /dev/null
"$PY" "$ROOT/moe/local/dryrun_lab_e6.py" > /dev/null
"$PY" "$ROOT/moe/local/dryrun_lab_e7.py" > /dev/null
"$PY" "$ROOT/moe/local/check_cuda_syntax.py" "$ROOT/moe/submission_m13t.py" > /dev/null
"$PY" "$ROOT/moe/local/check_cuda_syntax.py" "$ROOT/moe/submission_m13t.py" --nowg > /dev/null
"$PY" "$ROOT/moe/local/check_smem_attrs.py" "$ROOT/moe/submission_m13t.py"
"$PY" "$ROOT/moe/local/dryrun_tune.py" > /dev/null
echo "presubmit: all local checks passed for $*"
