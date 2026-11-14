"""Mutation test for emulate_cuda_ver4.py: each planted single bug (query register order, K word
order, P packing, V key order, transpose selectors, row rescale, output dim map) must make at
least one case fail the judge's check_implementation; the unmutated emulator must pass them all.
Run: .venv/bin/python attention/local/mutation_test_cuda_ver4.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_ver4 as emu  # noqa: E402
import reference as ref  # noqa: E402
from emulate import load_submission  # noqa: E402

sub = load_submission()
CASES = [emu.CASES[0], emu.CASES[1], emu.CASES[3]]


def caught():
    for i, (case, splits) in enumerate(CASES):
        data = ref.generate_input(seed=760 + i, device="cpu", **case)
        out = emu.run_ver4(data, sub, splits[-1], 3, 2 if i % 2 == 0 else 1)
        ok, _ = ref.check_implementation(data, ref.collect(data, out))
        if not ok:
            return True
    return False


bad = 0
emu.MUT.clear()
if caught():
    print("FAIL unmutated emulator fails a case")
    bad += 1
for m in ("qa_order", "k_words", "p_pack", "v_keys", "tr_sel", "rescale_row", "out_dim"):
    emu.MUT.clear()
    emu.MUT.add(m)
    hit = caught()
    print(("caught   " if hit else "SURVIVED ") + m)
    bad += not hit
emu.MUT.clear()
print("mutation_test_cuda_ver4:", "all caught" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
