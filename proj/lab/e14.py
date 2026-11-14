"""PE14: RG (router hi + lo mma.sync kernel: x staged once per CTA in shared memory and read by ldmatrix,
weights streamed in fragment order, split-K across a cluster with a fixed-order DSMEM reduction) at router
m = 32/128/256 against SGLang and the base's (v7a: PE10's Triton) configs. Winners >= 3% faster enter
CHOICE with the base kernel as fallback, printed as "V10 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_rg_info(lab):
    rows = []
    for inst in RG_INSTANCES:
        v = int(getattr(RG_MOD, rg_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("rg regs (RTC/NTW/CS/XS/D): " + " ".join(rows))


def sec_rg(lab, sweep, m):
    fallback = CHOICE.get(("router", m))
    cands = [dict(kind="rg", fallback=fallback, **c) for c in RG_LAB_CANDIDATES.get(("router", m), [])]
    if not cands or RG_MOD is None:
        return
    sweep.shape(lab, "router", m, cands)
    rec = sweep.results[("router", m)]
    times = " ".join(f"{c['RTC']}/{c['CS']}/{c['D']}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"rg router m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
        f"{rec['wrong']} wrong) RTC/CS/D:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare + RG setup ({len(RG_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if RG_MOD is not None:
        lab.section("rg info", sec_rg_info)
        for m in (128, 256, 32):
            lab.section(f"rg router m{m}", sec_rg, sweep, m, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if layer == "router":
            log(f"V10 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
