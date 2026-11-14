"""PE30 (recon for P41 v3 and F12M split-K): does a cluster launch cost F12M what it cost DeepGEMM (PF41)? v10's F12M
kernels at qkv m16, pooler m16 and qkv m8 launched plainly, with a 1-CTA and with a 2-CTA cluster attribute, plus PE17's
real 2-CTA split-K at qkv m16 (section lab/f12mlv.py). Nothing enters CHOICE: the timed kernel_fn is v10's."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    f12mlv_setup()
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in F12MLV_TARGETS:
        lab.section(f"f12mlv {layer} m{m}", sec_f12mlv, sweep, layer, m, min_left=6.0)
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
