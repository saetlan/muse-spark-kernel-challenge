#!/bin/sh
# Build the clean v16 from the labs' printed picks and run its local checks (no mutation phase, user's call Nov 4
# ~13:55 UTC). Usage: sh decode_layer/local/clean_v16.sh <DE18 log>
#   V4_MODE from DE15, V12_BOUNDARY / V13_CAT from DE16, V15_SIDE / V10_PDL from DE17, V16_M19 / V14_FUSED from <DE18 log>.
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
L15=lab_de15.8dd233aacbc14db6.log
L16=lab_de16.f51947d8d6cc49ab.log
L17=lab_de17.de78219134a04839.log
L18="$1"
cd "$ROOT/decode_layer"
grep -q "V16_M19 = " "profiles/$L18" && grep -q "V14_FUSED = " "profiles/$L18"
"$PY" local/make_v4.py --v16 --modes $L15 --pdl $L17 --boundary $L16 --cat $L16 --side $L17 --m19 "$L18" --fused "$L18"
head -1 submission_v16.py | grep -qx "# okc: problem=muse_decode_layer"
"$PY" local/lint_submission.py submission_v16.py
"$PY" -m py_compile submission_v16.py
"$PY" local/check_cuda_syntax.py > /dev/null
DRYRUN_V16_LOG="$L18" "$PY" local/dryrun_v16.py submission_v16.py | tail -1
DRYRUN_PICKS_LOG=$L15 "$PY" local/dryrun_v4.py submission_v16.py | tail -1
DRYRUN_PDL_LOG=$L17 "$PY" local/dryrun_v10.py submission_v16.py | tail -1
DRYRUN_V12_LOG=$L16 "$PY" local/dryrun_v12.py submission_v16.py | tail -1
DRYRUN_V13_LOG=$L16 "$PY" local/dryrun_v13.py submission_v16.py | tail -1
DRYRUN_V15_LOG=$L17 "$PY" local/dryrun_v15.py submission_v16.py | tail -1
grep -n -E "^V4_MODE = |^V10_PDL = |^V12_BOUNDARY = |^V13_CAT = |^V15_SIDE = |^V16_M19 = |^V14_FUSED = " submission_v16.py | tail -7 | cut -c1-200
shasum -a 256 submission_v16.py | cut -c1-16
echo "clean v16: local checks passed ($L18)"
