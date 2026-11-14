"""Mutation test for sim_pt_ring.py: plants protocol bugs in D3's source text (cuda_dec4.py) and
requires the simulation to flag every one.
Run: .venv/bin/python attention/local/mutation_test_pt_ring.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sim_pt_ring as sim  # noqa: E402

SRC = sim.source("cuda_dec4.py:CUDA_DEC4_SRC")
MUTATIONS = [
    ("page entries one chunk too late", "issue_pt(i + STAGES);", "issue_pt(i + STAGES - 1);"),
    # (PT_SLOTS = STAGES is an equivalent mutation: issue(j) reads chunk j's entries, then issues
    # the copy that refills that slot; the shuffle orders them. STAGES - 1 slots is a real bug.)
    ("too few PT slots (overwrite)", "constexpr int PT_SLOTS = 2 * STAGES;", "constexpr int PT_SLOTS = STAGES - 1;"),
    ("loop waits one group too few", '::"n"(STAGES - 1));\n    __syncwarp();', '::"n"(STAGES));\n    __syncwarp();'),
    ("prologue does not wait", 'asm volatile("cp.async.wait_group 0;");', 'asm volatile("cp.async.wait_group 1;");'),
    ("prologue loads too few entries", "for (int j = 0; j < STAGES; ++j) issue_pt(j);",
     "for (int j = 0; j < STAGES - 1; ++j) issue_pt(j);"),
    ("entries on K bytes (no pad offset)", "* PITCH + 128 + 4 * (e & 3)", "* PITCH + 4 * (e & 3)"),
    ("slots collide (stage only)", "(4 * (p / STAGES) + (e >> 2)) * PITCH", "(e >> 2) * PITCH"),
    ("rows from the wrong chunk's entries", "*pt_slot(i, lane & 15)", "*pt_slot(i + 1, lane & 15)"),
    ("loop issues one stage too far ahead", "issue(i + STAGES - 1);\n    asm", "issue(i + STAGES);\n    asm"),
]

missed = 0
for name, old, new in MUTATIONS:
    assert SRC.count(old) == 1, f"anchor for {name!r} found {SRC.count(old)} times"
    try:
        caught = not sim.check(SRC.replace(old, new), f"mutant: {name}")
    except AssertionError as exc:   # the parser refusing the mutant counts as caught
        print(f"caught (parse): {name}: {exc}")
        caught = True
    missed += not caught
    if not caught:
        print(f"MISSED: {name}")
print("mutation_test_pt_ring:", "every mutation caught" if not missed else f"{missed} missed")
sys.exit(1 if missed else 0)
