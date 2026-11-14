"""Mutation test for emulate_cuda_uv4.py: each planted single bug (query fold RMS group or RoPE
frequency, qk_scale_factor left out of the score scale, new-row RMS group or RoPE frequency,
new rows read from the ring, select without the end-1 clamp, gate piece map, m and l slots
swapped, push owner map, second level reading the neighbour row or skipping the last cluster; l not reduced over the row's lanes is a near-uniform scale that the
output RMSNorm cancels, so it is not a usable mutant) must make at least one case fail the judge's check_implementation
(or the emulator's own receive-buffer coverage assertion); the unmutated emulator must pass them all.
Run: .venv/bin/python attention/local/mutation_test_cuda_uv4.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import emulate_cuda_uv4 as emu  # noqa: E402
import reference as ref  # noqa: E402

# sliding cs 8 (rope, one holder), the split-holder case (cs 4), the full-layer case (no rope, cs 4)
CASES = [(emu.CASES[0][0], emu.CASES[0][1][0]), (emu.CASES[7][0], emu.CASES[7][1][0]),
         (emu.CASES[5][0], emu.CASES[5][1][0]), (emu.CASES[9][0], emu.CASES[9][1][0])]


def caught():
    for i, (case, (cs, stages, mt, *rest)) in enumerate(CASES):
        data = ref.generate_input(seed=780 + i, device="cpu", **case)
        try:
            out = emu.uv4(data, cs, stages, mt, seed=i, ncl=rest[0] if rest else 1)
            ok, _ = ref.check_implementation(data, ref.collect(data, out))
        except AssertionError:
            ok = False
        if not ok:
            return True
    return False


bad = 0
emu.MUT.clear()
if caught():
    print("FAIL unmutated emulator fails a case")
    bad += 1
for m in ("fold_norm", "fold_freq", "qk_factor", "newrow_norm", "newrow_rope", "no_sel", "no_clamp", "gate_piece",
          "ml_swap", "push_owner", "l2_row", "l2_skip_last"):
    emu.MUT.clear()
    emu.MUT.add(m)
    hit = caught()
    print(("caught   " if hit else "SURVIVED ") + m)
    bad += not hit
emu.MUT.clear()
print("mutation_test_cuda_uv4:", "all caught" if not bad else f"{bad} problems")
sys.exit(1 if bad else 0)
