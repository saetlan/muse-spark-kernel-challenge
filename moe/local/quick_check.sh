#!/bin/sh
# Fast pre-launch check (user, Nov 4 ~16:10 UTC: the full presubmit takes too long near the deadline).
# Only what has caught bugs that would fail or waste an okc run: the judge's static checks and CUDA-graph
# breakers (lint), Python syntax, CUDA syntax of both builds (main + fallback without wgmma), and the
# dynamic shared-memory limits (E17's failure). ~30 s. Usage: sh moe/local/quick_check.sh moe/submission_m13t.py
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/moe/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
for f in "$@"; do
  "$PY" "$ROOT/moe/local/check_cuda_syntax.py" "$f"
  "$PY" "$ROOT/moe/local/check_cuda_syntax.py" "$f" --nowg
  "$PY" "$ROOT/moe/local/check_smem_attrs.py" "$f"
done
echo "quick check passed for $*"
