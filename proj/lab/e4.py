"""PE4: P9 (swap-AB mma.sync, fragment-ordered weights, cluster split-K) on the 20 shapes with
m <= 32, timed by the judge on top of v2. In prepare(): v2's prepare plus p9_setup (pack + nvcc),
then registers/spills per instance, then a sweep of P9_LAB_CANDIDATES per shape through the gate
(FP64, probe rows, all-probe input, outliers) against SGLang and v2's choice; winners >= 3% faster
go into CHOICE."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_p9_info(lab):
    rows = []
    for inst in P9_INSTANCES:
        v = int(getattr(P9_MOD, p9_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append(f"{'/'.join(str(int(e)) for e in inst[:6])}:{regs}r" + (f"+{local}B" if local else ""))
    log("p9 regs (NT/RTW/RG/KSW/D/CS): " + " ".join(rows))


def sec_p9(lab, sweep, m, layers):
    for layer in layers:
        cands = [dict(kind="p9", **c) for c in P9_LAB_CANDIDATES.get((layer, m), [])]
        if cands:
            sweep.shape(lab, layer, m, cands)


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"v2 prepare + p9_setup ({len(P9_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    lab = Lab(budget)
    sweep = Sweep(state)
    lab.section("p9 info", sec_p9_info)
    if state.get("p22"):
        log("p22 escapes: " + ", ".join(f"{name} {pl['rate'] * 100:.4f}%" for name, pl in state["p22"].items()))
    for m in (1, 8, 16, 32):
        lab.section(f"p9 m{m}", sec_p9, sweep, m, ("router", "unpooler", "o", "qkv_gate", "pooler"), min_left=15.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    log(f"CHOICE now has {len(CHOICE)} shapes, {sum(1 for c in CHOICE.values() if c['kind'] == 'p9')} on P9; "
        f"prepare took {time.monotonic() - t0:.0f}s")
    return state
