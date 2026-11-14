"""PE40 (P41, DG12-RS v10, pinned copy proj/dg/pinned/dgrs_v10.py): A/B on the m32 shapes of PE39's RED / SPL instances (S) against
fully specialized twins (F: ring depth and M / N / K also compile-time constants), against the base (v10). PE39 put
unpooler m32 at 16.64 us on the judge (SGLang 18.75); pooler m32 tied v10, qkv m32 was 8% behind.
Winners >= 3% enter CHOICE (fallback: v10's entry), printed as "V33 CHOICE" lines: the check's own score measures
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
            log(f"V33 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
