"""PE41 (P41, DG12-RS v10, pinned copy proj/dg/pinned/dgrs_v10.py): ring depth. Every instance so far ran 8 stages; fully
specialized m32 twins with 10 / 12 stages (prefill's layout: unpooler, pooler S4 r1, qkv S5 r2 / S4 r2 reach 10, pooler S3 r0 12)
next to the 8-stage ones PE40 committed, plus o m32 with MATH 2 (smaller CTAs, a long shot).
Winners >= 3% enter CHOICE (fallback: v10's entry), printed as "V34 CHOICE" lines: the check's own score measures
v10 + the winners."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0



def prepare(weights):
    t0 = time.monotonic()
    dgrs_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgrs_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgrs":
            log(f"V34 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
