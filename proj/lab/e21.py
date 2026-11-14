"""PE21: F12M token groups (TG warps per (row group, k part) read the same weight chunks through L1/L2, each with 8 NT
tokens): m = 32 at m16's registers (PE9: F12M at m16 beat cuBLAS on pooler/qkv by 12-24%, its NT 4 version lost at
m32 on registers) on pooler/qkv/unpooler/o m32, and m = 16 at m8's registers on the same layers. Timed against
SGLang and the base (v10's kernels); winners >= 3% enter CHOICE with the base kernel as fallback, printed as
"V17 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
E21_STATE = {}
TG_ORDER = [("pooler", 32), ("qkv_gate", 32), ("unpooler", 32), ("o", 32),
            ("qkv_gate", 16), ("pooler", 16), ("unpooler", 16), ("o", 16)]


def sec_tg_info(lab):
    rows = []
    for inst in F12M_INSTANCES:
        v = int(getattr(F12M_MOD, f12m_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("f12m regs (NT/RTW/RG/KSW/NS/D/F12[/CS/XD/TG]): " + " ".join(rows))


def sec_tg(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="f12m", fallback=fallback, **c) for c in F12M_TG_CANDIDATES.get((layer, m), [])]
    if not cands or F12M_MOD is None or layer not in E21_STATE.get("f12m", {}):
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = lab_n(layer, m) * FULL[layer][1] * 1.5
    times = " ".join(f"{c['NT']}/{c['RTW']}/{c['KSW']}/{c['D']}/{c.get('XD', c['D'])}/{c['TG']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"tg {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) NT/RTW/KSW/D/XD/TG:us(12-bit TB/s) {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    E21_STATE.update(state)
    log(f"base prepare (F12M {len(F12M_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if F12M_MOD is not None:
        lab.section("tg info", sec_tg_info)
        for layer, m in TG_ORDER:
            lab.section(f"tg {layer} m{m}", sec_tg, sweep, layer, m, min_left=10.0)
    else:
        log("f12m: no module (build failed)")
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12m" and c.get("TG", 1) > 1:
            log(f"V17 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
