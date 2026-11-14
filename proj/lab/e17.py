"""PE17: F12M split-K across a cluster of CS = 2/4/8 CTAs (P36) at m = 1/8/16 on the shapes with too few
16-row tiles per SM (qkv: 144 tiles; the unpooler's 1024-row slice at m <= 8: 64 tiles) or short K (o: 16
chunks), plus pooler and the unpooler at m = 16 for comparison; timed against SGLang and the base's
configs. Winners >= 3% faster enter CHOICE with the base kernel as fallback, printed as "V13 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
E17_STATE = {}
E17_ORDER = [("qkv_gate", 16), ("qkv_gate", 8), ("unpooler", 8), ("unpooler", 1), ("o", 8), ("o", 16), ("o", 1),
             ("qkv_gate", 1), ("pooler", 8), ("pooler", 16), ("unpooler", 16), ("pooler", 1)]


def sec_cs_info(lab):
    rows = []
    for inst in F12M_INSTANCES:
        v = int(getattr(F12M_MOD, f12m_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("f12m regs (NT/RTW/RG/KSW/NS/D/F12[/CS]): " + " ".join(rows))


def sec_cs(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="f12m", fallback=fallback, **c) for c in F12M_CS_CANDIDATES.get((layer, m), [])]
    if not cands or F12M_MOD is None or layer not in E17_STATE.get("f12m", {}):
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = lab_n(layer, m) * FULL[layer][1] * 1.5
    times = " ".join(f"{c['NT']}/{c['RTW']}/{c['KSW']}/{c['D']}/{c['CS']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"cs {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) NT/RTW/KSW/D/CS:us(12-bit TB/s) {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    E17_STATE.update(state)
    log(f"base prepare + F12M setup ({len(F12M_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if F12M_MOD is not None:
        lab.section("cs info", sec_cs_info)
        for layer, m in E17_ORDER:
            lab.section(f"cs {layer} m{m}", sec_cs, sweep, layer, m, min_left=10.0)
    else:
        log("f12m: no module (build failed)")
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12m" and c.get("CS", 1) > 1:
            log(f"V13 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
