"""PE6: v3 timed by the judge, plus in prepare(): P9 on the router at m 1/8/16/32 (prints the chosen
configs, for v4), then the CUDA microbenchmarks PE3 lost. v3's own prepare runs FlashInfer's
autotuner first, so its output comes before the lab's lines and can't push them out of the log."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_router(lab, sweep):
    for m in (1, 8, 16, 32):
        cands = [dict(kind="p9", **c) for c in P9_LAB_CANDIDATES.get(("router", m), [])]
        if cands:
            sweep.shape(lab, "router", m, cands)


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"v3 prepare + p9_setup ({len(P9_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    lab = Lab(budget)
    sweep = Sweep(state)
    lab.section("router p9", sec_router, sweep, min_left=20.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] == "p9":
            log(f"V4 CHOICE {layer} m{m}: {c!r}")
    mb = _MB()
    for part in ("core", "bulk", "cluster"):
        lab.section(f"mb build {part}", sec_mb_build, mb, part, min_left=40.0)
    if "core" in mb.mods:
        lab.section("mb basics", sec_mb_basics, mb)
        lab.section("mb stream", sec_mb_stream, mb, min_left=30.0)
    if "cluster" in mb.mods:
        lab.section("mb cluster", sec_mb_cluster, mb)
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
