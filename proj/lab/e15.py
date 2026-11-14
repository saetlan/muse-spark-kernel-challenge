"""PE15: SX (12-bit planes in natural k order streamed into mma.sync A fragments, x staged per CTA in shared
memory by cp.async and read by ldmatrix, split-K inside the CTA and across a cluster, one DSMEM reduction) on
the four BF16 layers at m = 32/128 (and qkv/o at m = 256) against SGLang and the base's (v7a) configs.
Winners >= 3% faster enter CHOICE with the base kernel as fallback, printed as "V11 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_sx_info(lab):
    rows = []
    for inst in SX_INSTANCES:
        v = int(getattr(SX_MOD, sx_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst[:8]) + f"/{inst[9]}:{regs}r" + (f"+{local}B" if local else ""))
    log("sx regs (NT8/TG/RTW/RGN/KSW/NS/D/XS/CS): " + " ".join(rows))
    rates = {k: round(float(v["flags"].float().mean()), 4) for k, v in SX_STATE["sx"].items()}
    log(f"sx flagged (tile, chunk) fraction: {rates}")


def sec_sx(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(c, fallback=fallback) for c in SX_LAB_CANDIDATES.get((layer, m), [])]
    if not cands or SX_MOD is None or layer not in SX_STATE["sx"]:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = sum(v.numel() * v.element_size() for v in (SX_STATE["sx"][layer]["p"],)) * lab_n(layer, m) // FULL[layer][0]
    times = " ".join(f"{c['RTW']}/{c['RGN']}/{c['KSW']}/{c['NS']}/{c['XS']}/{c['CS']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"sx {layer} m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
        f"{rec['wrong']} wrong) RTW/RGN/KSW/NS/XS/CS:us(TB/s) {times}")


SX_STATE = {}
SX_ORDER = [(layer, m) for m in (32, 128, 256) for layer in ("pooler", "unpooler", "qkv_gate", "o")
            if (layer, m) in SX_LAB_CANDIDATES]


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    SX_STATE.update(state)
    log(f"base prepare + SX setup ({len(SX_INSTANCES)} instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if SX_MOD is not None:
        lab.section("sx info", sec_sx_info)
        for layer, m in SX_ORDER:
            lab.section(f"sx {layer} m{m}", sec_sx, sweep, layer, m, min_left=10.0)
    else:
        log("sx: no module (build failed)")
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "sx":
            log(f"V11 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
