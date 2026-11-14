"""PE11: WG (wgmma GEMM, TMA + mbarrier pipeline, in-cluster split-K) at m = 128/256 on the four BF16
layers, plus PE10's router wide-tile sweep, timed by the judge on top of the base (v5). Needs lab/e10.py
appended before this file. Winners >= 3% faster go into CHOICE with the shape's previous kernel as
fallback and are printed as "V7 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_wg_info(lab):
    rows = []
    for inst in WG_INSTANCES:
        v = int(getattr(WG_MOD, wg_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("wg regs (KS/ST/TT): " + " ".join(rows))


def sec_wg(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="wg", fallback=fallback, **c) for c in WG_LAB_CANDIDATES.get((layer, m), [])]
    if not cands or WG_MOD is None:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    times = " ".join(f"{c['KS']}/{c['ST']}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"wg {layer} m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
        f"{rec['wrong']} wrong) KS/ST:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare + WG setup ({len(WG_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if WG_MOD is not None:
        lab.section("wg info", sec_wg_info)
        for m in (128, 256):
            for layer in ("qkv_gate", "pooler", "unpooler", "o"):
                lab.section(f"wg {layer} m{m}", sec_wg, sweep, layer, m, min_left=15.0)
    for m in (128, 256, 32):
        lab.section(f"router wide m{m}", sec_router_wide, sweep, m, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] == "wg" or (layer == "router" and m >= 32):
            log(f"V7 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
