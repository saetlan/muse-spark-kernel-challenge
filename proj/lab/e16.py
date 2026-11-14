"""PE16: SXW (wgmma RS: 12-bit planes streamed by cp.async into per-warp shared-memory rings, decoded in 7 ops
into double-buffered RS A fragments, x in the swizzled cp.async stages read by a K-major 128B-swizzle
descriptor, wait_group 1 per chunk, escapes after the loop from L2-resident x, SX's cluster reduction) on the
four BF16 layers at m = 128/256 against SGLang and the base's configs. Winners >= 3% faster enter CHOICE with
the base kernel as fallback, printed as "V12 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_sxw_info(lab):
    rows = []
    for inst in SXW_INSTANCES:
        v = int(getattr(SX_MOD, sxw_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("sxw regs (NB/WGS/KSW/NS/P/CS): " + " ".join(rows))


def sec_sxw(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(c, fallback=fallback) for c in SXW_LAB_CANDIDATES.get((layer, m), [])]
    if not cands or SX_MOD is None or layer not in SXW_STATE["sx"]:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = SXW_STATE["sx"][layer]["p"].numel() * 4 * lab_n(layer, m) // FULL[layer][0]
    times = " ".join(f"{c['NB']}/{c['WGS']}/{c['KSW']}/{c['NS']}/{c['P']}/{c['CS']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"sxw {layer} m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
        f"{rec['wrong']} wrong) NB/WGS/KSW/NS/P/CS:us(TB/s) {times}")


SXW_STATE = {}
SXW_ORDER = [(layer, m) for m in (128, 256) for layer in ("pooler", "unpooler", "qkv_gate", "o")
             if (layer, m) in SXW_LAB_CANDIDATES]


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    SXW_STATE.update(state)
    log(f"base prepare + SX setup ({len(SX_INSTANCES)} SX, {len(SXW_INSTANCES)} SXW instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if SX_MOD is not None:
        lab.section("sxw info", sec_sxw_info)
        for layer, m in SXW_ORDER:
            lab.section(f"sxw {layer} m{m}", sec_sxw, sweep, layer, m, min_left=10.0)
    else:
        log("sx: no module (build failed)")
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "sx":
            log(f"V12 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
