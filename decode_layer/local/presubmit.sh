#!/bin/sh
# Every local check for decode_layer, before any okc check or submit.
# Usage: sh decode_layer/local/presubmit.sh decode_layer/lab_d0.py [more files]
# Lint copied from moe/local/lint_submission.py (Nov 3), so MoE-side edits do not change our checks.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
for f in "$@"; do head -1 "$f" | grep -qx "# okc: problem=muse_decode_layer" || { echo "$f: bad first line"; exit 1; }; done
"$PY" "$ROOT/decode_layer/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/decode_layer/local/dryrun_d0.py" "$ROOT/decode_layer/lab_d0.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v2.py" "$ROOT/decode_layer/lab_de1.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/check_x5.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/check_cuda_syntax.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_de2.py" "$ROOT/decode_layer/lab_de4.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v3.py" "$ROOT/decode_layer/lab_de5.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v3.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/lab_de6.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/submission_v4.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v4.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/check_m11pf.py" > /dev/null
for sc in normal dirty err tiny slowlayer; do
  "$PY" "$ROOT/decode_layer/local/dryrun_v5.py" "$ROOT/decode_layer/lab_de7.py" "$sc" > /dev/null
done
"$PY" "$ROOT/decode_layer/local/mutate_v5.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/lab_de8.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/submission_v6.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v4.py" "$ROOT/decode_layer/lab_de8.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/check_l9.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v7.py" "$ROOT/decode_layer/lab_de9.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v7.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v7.py" "$ROOT/decode_layer/lab_de10.py" > /dev/null
# the clean v7 candidate (DE9's picks appended by make_v4.py --v7 --modes/--stage)
DRYRUN_PICKS_LOG=lab_de9.6d043f8d93604684.log "$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/submission_v7.py" > /dev/null
DRYRUN_STAGE_LOG=lab_de9.6d043f8d93604684.log "$PY" "$ROOT/decode_layer/local/dryrun_v7.py" "$ROOT/decode_layer/submission_v7.py" > /dev/null
for sc in normal tiny; do
  "$PY" "$ROOT/decode_layer/local/dryrun_v11.py" "$ROOT/decode_layer/lab_de11.py" "$sc" > /dev/null
done
"$PY" "$ROOT/decode_layer/local/mutate_v11.py" > /dev/null
# v9 (attention v6 + bf16_proj v6 with 12-bit lossless dense weights) and its DE12 lab
"$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/lab_de12.py" > /dev/null
DRYRUN_PICKS_LOG=lab_de9.6d043f8d93604684.log "$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/submission_v9.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v9.py" "$ROOT/decode_layer/lab_de12.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v9.py" > /dev/null
# v10 (v9 + MoE E15 / E16's PDL path, lab-chosen per M11 class) and DE12 built on it
DRYRUN_PICKS_LOG=lab_de9.6d043f8d93604684.log "$PY" "$ROOT/decode_layer/local/dryrun_v4.py" "$ROOT/decode_layer/submission_v10.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v10.py" "$ROOT/decode_layer/lab_de12.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v10.py" "$ROOT/decode_layer/lab_de12.py" equal > /dev/null
"$PY" "$ROOT/decode_layer/local/dryrun_v10.py" "$ROOT/decode_layer/submission_v10.py" > /dev/null
"$PY" "$ROOT/decode_layer/local/mutate_v10.py" > /dev/null
echo "presubmit: all local checks passed for $*"
