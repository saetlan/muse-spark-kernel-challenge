"""Mutation test for emulate_cuda_ver5t.py: each planted layout bug in the re-laid V tile (consumer reads
the other head's block, chunk/head block order swapped, uint4 slots k swapped pairwise, lanes shifted)
must make the judge's check_implementation fail on some case; the unmutated emulator must pass.
Run: .venv/bin/python attention/local/mutation_test_cuda_ver5t.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_ver5 as e5  # noqa: E402
import emulate_cuda_ver5t as e5t  # noqa: E402
import reference as ref  # noqa: E402
from emulate import load_submission  # noqa: E402

sub = load_submission()


def caught():
    e5.VFRAG = e5t.vt_frags
    try:
        for i, (case, splits) in enumerate([e5.CASES[0], e5.CASES[1]]):
            data = ref.generate_input(seed=790 + i, device="cpu", **case)
            out = e5.run_ver5(data, sub, splits[-1], 3)
            ok, _ = ref.check_implementation(data, ref.collect(data, out))
            if not ok:
                return True
        return False
    except AssertionError:
        return True
    finally:
        e5.VFRAG = None


bad = 0
e5t.MUT.clear()
if caught():
    print("FAIL unmutated emulator fails a case")
    bad += 1
for m in ("head_swap", "blk_order", "k_order", "lane_shift"):
    e5t.MUT.clear()
    e5t.MUT.add(m)
    hit = caught()
    print(("caught   " if hit else "SURVIVED ") + m)
    bad += not hit
e5t.MUT.clear()
print("mutation_test_cuda_ver5t:", "all caught" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
