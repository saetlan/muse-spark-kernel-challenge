"""PE39 (P41, DG12-RS v9 specialized, pinned copy proj/dg/pinned/dgrs_v9.py): PE38 showed that an instance specialized for
its reduction mode and split count (RED / SPL template parameters) runs unpooler m32 in 20.96 us vs 23.81 generic and SGLang's
22.86. Sweep of specialized d0 instances (no ring corrections, which lost) on every m32 / m128 / m256 shape against the base
(v10). Winners >= 3% enter CHOICE (fallback: v10's entry), printed as "V32 CHOICE" lines: that check's own score then measures
v10 + the winners."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DGRS_DIAGS = []


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
            log(f"V32 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
