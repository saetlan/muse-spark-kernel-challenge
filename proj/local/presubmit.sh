#!/bin/sh
# Every local check, before any okc check or submit. Usage: sh proj/local/presubmit.sh proj/submission.py [more files]
# The lint is a copy of the MoE track's moe/local/lint_submission.py (problem-agnostic; rules from Triton 3.8 / PyTorch 2.14.1 source).
set -e
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
PY="$ROOT/.venv/bin/python"
"$PY" "$ROOT/proj/local/lint_submission.py" "$@"
"$PY" -m py_compile "$@"
"$PY" "$ROOT/proj/local/make_lab.py" "$ROOT/proj/lab/e1.py" "$ROOT/proj/lab_e1.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e1.py"
"$PY" "$ROOT/proj/local/make_v1.py" > /dev/null
"$PY" "$ROOT/proj/local/check_v1.py"
"$PY" "$ROOT/proj/local/make_v2.py" > /dev/null
"$PY" "$ROOT/proj/local/check_v1.py" submission_v2.py
"$PY" "$ROOT/proj/local/make_v3.py" > /dev/null
"$PY" "$ROOT/proj/local/check_v1.py" submission_v3.py
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_v1.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e2.py" "$ROOT/proj/lab_e2.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e2.py"
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_v2.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e3.py" "$ROOT/proj/lab_e3.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e3.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/lab/e3.py"
"$PY" "$ROOT/proj/local/check_p9.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/p9/p9.py"
"$PY" "$ROOT/proj/local/make_p9.py" --base submission_v3.py --lab "$ROOT/proj/submission_p9lab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_p9lab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e4.py" "$ROOT/proj/lab_e4.py" > /dev/null
cp "$ROOT/proj/lab_e4.py" "$ROOT/proj/lab_e5.py"
"$PY" "$ROOT/proj/local/dryrun_lab_e4.py"
"$PY" "$ROOT/proj/local/make_p9.py" --base submission_v3.py --instances ROUTER --lab "$ROOT/proj/submission_p9router.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_p9router.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/mb.py" "$ROOT/proj/lab/e6.py" "$ROOT/proj/lab_e6.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e6.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/lab/mb.py"
"$PY" "$ROOT/proj/local/check_p22.py"
"$PY" "$ROOT/proj/local/make_v4.py" > /dev/null
"$PY" "$ROOT/proj/local/make_p9.py" --base submission_v4_base.py --instances F12_V4 --lab "$ROOT/proj/submission_p22lab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_p22lab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e4.py" "$ROOT/proj/lab_e7.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e4.py" lab_e7.py
"$PY" "$ROOT/proj/local/check_f12v.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/f12v/f12v.py"
"$PY" "$ROOT/proj/local/make_f12v.py" --base submission_v4.py --lab "$ROOT/proj/submission_f12vlab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_f12vlab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e8.py" "$ROOT/proj/lab_e8.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e8.py"
"$PY" "$ROOT/proj/local/check_f12m.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/f12m/f12m.py"
"$PY" "$ROOT/proj/local/make_f12m.py" --base submission_f12vlab.py --lab "$ROOT/proj/submission_f12mlab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_f12mlab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e8.py" "$ROOT/proj/lab/e9.py" "$ROOT/proj/lab_e9.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e9.py"
"$PY" "$ROOT/proj/local/make_v5.py" > /dev/null
"$PY" "$ROOT/proj/local/check_v1.py" submission_v5.py
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_v5.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e10.py" "$ROOT/proj/lab_e10.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e10.py"
"$PY" "$ROOT/proj/local/check_wg.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/wg/wg.py"
"$PY" "$ROOT/proj/local/make_wg.py" --base submission_v5.py --lab "$ROOT/proj/submission_wglab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_wglab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e10.py" "$ROOT/proj/lab/e11.py" "$ROOT/proj/lab_e11.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e11.py"
"$PY" "$ROOT/proj/local/make_v7a.py" > /dev/null
"$PY" "$ROOT/proj/local/check_v1.py" submission_v7a.py
"$PY" "$ROOT/proj/local/make_wg.py" --base submission_v7a.py --instances LAB --wgf WGF_LAB --lab "$ROOT/proj/submission_wgflab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_wgflab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e10.py" "$ROOT/proj/lab/e11.py" "$ROOT/proj/lab/e12.py" "$ROOT/proj/lab_e12.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e12.py"
"$PY" "$ROOT/proj/local/make_wg.py" --base submission_v7a.py --instances LAB_DEEP --lab "$ROOT/proj/submission_wgdeep.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_wgdeep.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e10.py" "$ROOT/proj/lab/e11.py" "$ROOT/proj/lab/e13.py" "$ROOT/proj/lab_e13.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e13.py"
"$PY" "$ROOT/proj/local/check_rg.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/rg/rg.py"
"$PY" "$ROOT/proj/local/make_rg.py" --base submission_v7a.py --lab "$ROOT/proj/submission_rglab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_rglab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e14.py" "$ROOT/proj/lab_e14.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e14.py"
"$PY" "$ROOT/proj/local/check_sx.py"
"$PY" "$ROOT/proj/local/check_cuda_syntax.py" "$ROOT/proj/sx/sx.py"
"$PY" "$ROOT/proj/local/make_sx.py" --base submission_v7a.py --lab "$ROOT/proj/submission_sxlab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_sxlab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e15.py" "$ROOT/proj/lab_e15.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e15.py"
"$PY" "$ROOT/proj/local/make_sx.py" --base submission_v7a.py --instances NONE --w-instances W_LAB --lab "$ROOT/proj/submission_sxwlab.py" > /dev/null
"$PY" "$ROOT/proj/local/make_lab.py" --base submission_sxwlab.py "$ROOT/proj/lab/sweep.py" "$ROOT/proj/lab/e16.py" "$ROOT/proj/lab_e16.py" > /dev/null
"$PY" "$ROOT/proj/local/dryrun_lab_e16.py"
"$PY" "$ROOT/proj/local/mutation_test.py"
# every scored candidate given as an argument: all its chosen shapes correct, off-grid shapes = baseline
for f in "$@"; do case "$(basename "$f")" in submission_v[0-9]*.py) "$PY" "$ROOT/proj/local/check_v1.py" "$(basename "$f")";; esac; done
echo "presubmit: all local checks passed for $*"
