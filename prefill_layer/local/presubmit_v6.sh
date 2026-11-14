#!/bin/sh
# Every local check, before any okc check or submit (v6). Usage: sh prefill_layer/local/presubmit_v6.sh prefill_layer/submission.py [more files]
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/prefill_layer/local/check_v6_vs_baseline.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/make_lab.py" "$ROOT/prefill_layer/lab/re20.py" "$ROOT/prefill_layer/lab_re20.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/prefill_layer/local/check_v2_cpu.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v6_cpu.py" 0 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v6_cpu.py" 1 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v6_cpu.py" 2 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_v6_cpu.py" 3 > /dev/null
"$PY" "$ROOT/prefill_layer/local/check_cuda_syntax.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/sim_cluster_ring.py" > /dev/null
"$PY" "$ROOT/prefill_layer/local/dryrun_lab_re20.py" > /dev/null
echo "presubmit (v6): all local checks passed for $*"
