#!/bin/sh
# Every local check, before any okc check or submit (v4). Usage: sh prefill_layer/local/presubmit.sh prefill_layer/submission.py [more files]
# Mutation tests of the checks (slower): prefill_layer/local/mutate_v{2,3,4}_checks.py.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/prefill_layer/local/check_v4_vs_baseline.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/make_lab.py" "$ROOT/prefill_layer/lab/re4.py" "$ROOT/prefill_layer/lab_re4.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/prefill_layer/local/check_v2_cpu.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v4_cpu.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_cuda_syntax.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/dryrun_lab_re4.py" > /dev/null
echo "presubmit: all local checks passed for $*"
