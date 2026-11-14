"""Mutation test for emulate_cuda_w1.py: each planted single bug (K tile written with key groups and rows
swapped, V re-laid in natural key order instead of the MMA order, the two 16-dim halves of a query k-step swapped, Q.K
k-step offset of one chunk instead of two, P registers in the wrong order, Vt descriptor SBO 512 vs
the 528 the transposers write, partial dims permuted) must make the judge's check_implementation fail
on some case (or trip the emulator's own NaN / bounds assertions); the unmutated emulator must pass.
Run: .venv/bin/python attention/local/mutation_test_cuda_w1.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_w1 as emu  # noqa: E402
import reference as ref  # noqa: E402
from emulate import load_submission  # noqa: E402

sub = load_submission()


def caught():
    try:
        for i, (case, splits) in enumerate([emu.CASES[0], emu.CASES[1]]):
            data = ref.generate_input(seed=860 + i, device="cpu", **case)
            out = emu.run_w1(data, sub, splits[-1], 4)
            ok, _ = ref.check_implementation(data, ref.collect(data, out))
            if not ok:
                return True
        return False
    except AssertionError:
        return True


bad = 0
emu.MUT.clear()
if caught():
    print("FAIL unmutated emulator fails a case")
    bad += 1
for m in ("k_layout", "sigma", "q_perm", "kstep", "pack", "sbo_v", "out_dims"):
    emu.MUT.clear()
    emu.MUT.add(m)
    hit = caught()
    print(("caught   " if hit else "SURVIVED ") + m)
    bad += not hit
emu.MUT.clear()
print("mutation_test_cuda_w1:", "all caught" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
