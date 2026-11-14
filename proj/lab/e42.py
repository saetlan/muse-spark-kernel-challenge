"""PE42 (P43, F12K = stream-K F12M): v11b (the board entry, scored/submission_v11b.py) byte for byte as the timed path, plus
F12K candidates on the few-tile shapes (qkv m16 / m8 / m1: 144 tiles; unpooler slice m8 / m1: 64 tiles) with grids of
2-4 CTAs per SM. Winners >= 3% over the faster of SGLang and v11b's entry enter CHOICE (fallback: v11b's entry) and are
printed as "SK CHOICE" lines, so the check's own score measures v11b + the winners."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 170.0


def prepare(weights):
    t0 = time.monotonic()
    f12k_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v11b) took {time.monotonic() - t0:.1f}s")
    targets = f12k_targets(torch.cuda.get_device_properties(0).multi_processor_count)
    f12k_setup_lab(state, t0, targets)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in targets:
        lab.section(f"f12k {layer} m{m}", sec_f12k, sweep, targets, layer, m, min_left=8.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    lab.section("f12k counters", sec_f12k_cnt, min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12k":
            log(f"SK CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
