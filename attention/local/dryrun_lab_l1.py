"""CPU dry run of lab L1: every variant the lab will time goes through plan() and the CPU
emulator on small cases shaped like each class, and must pass the judge's checker."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from emulate import load_submission, run  # noqa: E402

import reference as ref  # noqa: E402

lab = load_submission(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lab_l1.py"))
cases = {(True, False): dict(batch=2, ctx=7000, heads="tp8", layer="full", page=1, spread="agentic", dist="coherent"),
         (True, True): dict(batch=2, ctx=5000, heads="tp8", layer="full", q_len=8, page=1, dist="coherent"),
         (False, False): dict(batch=3, ctx=5000, heads="tp8", layer="sliding", page=1, dist="soft"),
         (False, True): dict(batch=2, ctx=5000, heads="tp8", layer="sliding", q_len=8, page=1, dist="edges")}
bad = n = 0
for (full, verify), case in cases.items():
    for tune in lab._variants(full, verify):
        data = ref.generate_input(seed=7 + n, device="cpu", **case)
        for b in (1, 16, 32):
            cfg = dict(data["config"], batch=b)
            lab.plan(cfg, lab.scored_max_pages(b, cfg["q_len"]), tune)
        ok, msg = ref.check_implementation(data, ref.collect(data, run(data, lab, tune)))
        n += 1
        if not ok:
            bad += 1
            print("FAIL", full, verify, tune, msg)
print(f"dryrun_lab_l1: {n} variants, {bad} failed")
sys.exit(1 if bad else 0)
