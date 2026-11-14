#!/bin/sh
# Every local check, before any okc check or submit (v5). Usage: sh prefill_layer/local/presubmit_v5.sh prefill_layer/submission.py [more files]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/prefill_layer/local/check_v5_vs_baseline.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/make_lab.py" "$ROOT/prefill_layer/lab/re5.py" "$ROOT/prefill_layer/lab_re5.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/prefill_layer/local/check_v2_cpu.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v5_cpu.py" 0 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v5_cpu.py" 1 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_cuda_syntax.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/dryrun_lab_re5.py" > /dev/null
echo "presubmit (v5): all local checks passed for $*"
