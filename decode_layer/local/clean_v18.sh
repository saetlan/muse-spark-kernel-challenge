#!/bin/sh
# Build the clean v18 from the labs' printed picks and run its local checks (no mutation phase, user's call Nov 4
# ~13:55 UTC). Usage: sh decode_layer/local/clean_v18.sh <DE21 log>
#   V4_MODE from <DE21 log>, V12_BOUNDARY / V13_CAT from DE16, V15_SIDE / V10_PDL from DE17, V14_FUSED from DE18,
#   V17_CHOICE from DE20.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
L16=lab_de16.f51947d8d6cc49ab.log
L17=lab_de17.de78219134a04839.log
L18=lab_de18.a8d99e0c891e4204.log
L20=lab_de20.bb831464d0174f04.log
L21="$1"
cd "$ROOT/decode_layer"
grep -q "V4_MODE = " "profiles/$L21"
"$PY" local/make_v4.py --v18 --modes "$L21" --pdl $L17 --boundary $L16 --cat $L16 --side $L17 --fused $L18 --choice $L20
head -1 submission_v18.py | grep -qx "# okc: problem=muse_decode_layer"
"$PY" local/lint_submission.py submission_v18.py
"$PY" -m py_compile submission_v18.py
"$PY" local/check_cuda_syntax.py > /dev/null
DRYRUN_PICKS_LOG="$L21" "$PY" local/dryrun_v4.py submission_v18.py | tail -1
DRYRUN_V17_LOG=$L20 "$PY" local/dryrun_v17.py submission_v18.py | tail -1
DRYRUN_PDL_LOG=$L17 "$PY" local/dryrun_v10.py submission_v18.py | tail -1
DRYRUN_V12_LOG=$L16 "$PY" local/dryrun_v12.py submission_v18.py | tail -1
DRYRUN_V13_LOG=$L16 "$PY" local/dryrun_v13.py submission_v18.py | tail -1
DRYRUN_V15_LOG=$L17 "$PY" local/dryrun_v15.py submission_v18.py | tail -1
grep -n -E "^V4_MODE = |^V17_CHOICE = " submission_v18.py | tail -2 | cut -c1-300
shasum -a 256 submission_v18.py | cut -c1-16
echo "clean v18: local checks passed ($L21)"
