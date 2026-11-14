#!/bin/sh
# Build the clean v21 from the labs' printed picks and run its local checks (no mutation phase, user's call Nov 4
# ~13:55 UTC). Usage: sh decode_layer/local/clean_v21.sh <DE24 log>
#   V21_DGRS from <DE24 log>, V4_MODE and V17_CHOICE from DE23, V12_BOUNDARY / V13_CAT from DE16, V15_SIDE / V10_PDL from DE17.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
L16=lab_de16.f51947d8d6cc49ab.log
L17=lab_de17.de78219134a04839.log
L23=lab_de23.0ac9a55dfd394bee.log
L24="$1"
cd "$ROOT/decode_layer"
grep -q "V4_MODE = " "profiles/$L23"
grep -q "V17_CHOICE = " "profiles/$L23"
grep -q "V21_DGRS = " "profiles/$L24"
"$PY" local/make_v4.py --v21 --modes "$L23" --pdl $L17 --boundary $L16 --cat $L16 --side $L17 --choice "$L23" --dgrs "$L24"
head -1 submission_v21.py | grep -qx "# okc: problem=muse_decode_layer"
"$PY" local/lint_submission.py submission_v21.py
"$PY" -m py_compile submission_v21.py
"$PY" local/check_cuda_syntax.py > /dev/null
DRYRUN_PICKS_LOG="$L23" "$PY" local/dryrun_v4.py submission_v21.py | tail -1
DRYRUN_V17_LOG="$L23" "$PY" local/dryrun_v17.py submission_v21.py | tail -1
DRYRUN_PDL_LOG=$L17 "$PY" local/dryrun_v10.py submission_v21.py | tail -1
DRYRUN_V12_LOG=$L16 "$PY" local/dryrun_v12.py submission_v21.py | tail -1
DRYRUN_V13_LOG=$L16 "$PY" local/dryrun_v13.py submission_v21.py | tail -1
DRYRUN_V15_LOG=$L17 "$PY" local/dryrun_v15.py submission_v21.py | tail -1
DRYRUN_V21_LOG="$L24" "$PY" local/dryrun_v21.py clean | tail -1
grep -n -E "^V4_MODE = |^V17_CHOICE = |^V21_DGRS = " submission_v21.py | tail -3 | cut -c1-300
wc -c < submission_v21.py
shasum -a 256 submission_v21.py | cut -c1-16
echo "clean v21: local checks passed ($L24)"
