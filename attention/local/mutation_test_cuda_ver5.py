"""Mutation test for emulate_cuda_ver5.py: each planted single bug (window-edge or causal term dropped from
the stage mask condition, raw max not scaled, P chunks swapped, V rows of chunk 1 read from chunk 0,
row A's rescale on row B) must make at least one case fail the judge's check_implementation; the
unmutated emulator must pass them all.
Run: .venv/bin/python attention/local/mutation_test_cuda_ver5.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_ver5 as emu  # noqa: E402
import reference as ref  # noqa: E402
from emulate import load_submission, window_rows_case  # noqa: E402

sub = load_submission()
CASES = [emu.CASES[0], emu.CASES[7]]


def caught():
    try:
        return caught_cases()
    except AssertionError:      # e.g. P beyond E4M3's range (the kernel would saturate): a detected bug
        return True


def caught_cases():
    for i, (case, splits) in enumerate(CASES):
        data = ref.generate_input(seed=760 + i, device="cpu", **case)
        out = emu.run_ver5(data, sub, splits[-1], 3)
        ok, _ = ref.check_implementation(data, ref.collect(data, out))
        if not ok:
            return True
    data = window_rows_case(ref, 1)
    out = emu.run_ver5(data, sub, 3, 3)
    ok, _ = ref.check_implementation(data, ref.collect(data, out))
    return not ok


bad = 0
emu.MUT.clear()
if caught():
    print("FAIL unmutated emulator fails a case")
    bad += 1
for m in ("mask_window", "mask_causal", "scale_max", "pack_chunk", "chunk_v", "al_row"):
    emu.MUT.clear()
    emu.MUT.add(m)
    hit = caught()
    print(("caught   " if hit else "SURVIVED ") + m)
    bad += not hit
emu.MUT.clear()
print("mutation_test_cuda_ver5:", "all caught" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
