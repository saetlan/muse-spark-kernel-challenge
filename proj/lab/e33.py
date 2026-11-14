"""PE33 (P41, DG12-RS v5): the DIAG split at unpooler m32 and o m32 (BM32 S2 red0; math 2 / 4 x d0 full / d16 no finalize
with the math kept / d32 no escape precompute), registers, ring stages and cluster occupancy per config, and PE32's sweep
with MATH 4 back in (S <= 2 for red 0, plus pooler m32 S3 red 0 and qkv S4 red 0, whose clusters fit one wave). Winners
>= 3% enter CHOICE (fallback: v10's entry), printed as "V26 CHOICE" lines."""

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
    for d in DGRS_DIAGS:
        lab.section(f"dgrs diag {d['layer']}", sec_dgrs_diag, d, min_left=10.0)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgrs":
            log(f"V26 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
