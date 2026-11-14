"""PE9: F12M (12-bit lossless weights decoded in 7 ops per 8 weights into mma.sync A fragments, the
GEMV's many-small-CTAs structure) at m = 8/16/32 on the four BF16 layers, plus (when the base carries
it) PE8's F12V sweep at m = 1; timed by the judge on top of the base's choices. Each F12M shape also
times the same kernel on BF16 fragments (F12 = False), which separates the structure's gain from the
compression's. Needs lab/e8.py appended before this file (its F12V sections). Winners >= 3% faster
go into CHOICE with the shape's previous kernel as fallback and are printed as "V6 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_f12m_info(lab, state):
    rows = []
    for inst in F12M_INSTANCES:
        v = int(getattr(F12M_MOD, f12m_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("f12m regs (NT/RTW/RG/KSW/NS/D/F12): " + " ".join(rows))
    log("f12m escapes: " + ", ".join(f"{name} {pl['rate'] * 100:.4f}% max {pl['max_row']}/row"
                                     for name, pl in state["f12m"].items()))


def sec_f12m(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="f12m", fallback=fallback, **c) for c in F12M_LAB_CANDIDATES.get((layer, m), [])]
    if not cands or F12M_MOD is None:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    n, k = lab_n(layer, m), FULL[layer][1]
    times = " ".join(f"{c['RTW']}/{c['RG']}/{c['KSW']}/{c['D']}{'' if c['F12'] else 'b'}:{us:.2f}"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"f12m {layer} m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; 12-bit {n * k * 1.5 / 1e6:.1f} MB) "
        f"RTW/RG/KSW/D:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare + setups (F12M {len(F12M_INSTANCES)} instances"
        + (f", F12V {len(F12V_INSTANCES)}" if "F12V_LAB_CANDIDATES" in globals() else "") + f") took {time.monotonic() - t0:.1f}s")
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    lab = Lab(budget)
    sweep = Sweep(state)
    if F12M_MOD is not None:
        lab.section("f12m info", sec_f12m_info, state)
    for m in (8, 16, 32):
        for layer in ("pooler", "unpooler", "qkv_gate", "o"):
            lab.section(f"f12m {layer} m{m}", sec_f12m, sweep, layer, m, min_left=15.0)
    if "F12V_LAB_CANDIDATES" in globals() and F12V_MOD is not None:
        lab.section("f12v info", sec_f12v_info, state)
        for layer in ("pooler", "qkv_gate", "o", "unpooler"):
            lab.section(f"f12v {layer}", sec_f12v, sweep, layer, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] in ("f12m", "f12v"):
            log(f"V6 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
