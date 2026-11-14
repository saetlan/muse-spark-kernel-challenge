"""PE13: WG (wgmma, TMA + mbarrier, in-cluster split-K, PF31 epilogue) with deeper pipelines (5-6 stages)
and clusters of at most 4, at m = 128/256 on the four BF16 layers, on top of v7a. Needs lab/e11.py
appended before this file (sec_wg / sec_wg_info). Winners >= 3% faster enter CHOICE with the shape's
previous kernel as fallback, printed as "V9 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare + WG setup ({len(WG_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if WG_MOD is not None:
        lab.section("wg info", sec_wg_info)
        for m in (128, 256):
            for layer in ("unpooler", "pooler", "qkv_gate", "o"):
                lab.section(f"wg {layer} m{m}", sec_wg, sweep, layer, m, min_left=12.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] == "wg":
            log(f"V9 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
