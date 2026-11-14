"""PE12: WGF (12-bit weights from registers into RS wgmma, x by TMA, in-cluster split-K) at m = 16/32/128/256
on the four BF16 layers, timed by the judge on top of the base. Needs lab/e11.py appended before this file
(sec_wg_info / sec_wg for any WG candidates the base carries). Winners >= 3% faster enter CHOICE with the
shape's previous kernel as fallback, printed as "V8 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def sec_wgf_info(lab):
    rows = []
    for inst in WGF_INSTANCES:
        v = int(getattr(WG_MOD, wgf_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("wgf regs (KS/ST/TT/D): " + " ".join(rows))


def sec_wgf(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="wgf", fallback=fallback, **c) for c in WGF_LAB_CANDIDATES.get((layer, m), [])]
    if not cands or WG_MOD is None or layer not in sweep.state.get("wgf", {}):
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    times = " ".join(f"{c['KS']}/{c['ST']}/{c['D']}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"wgf {layer} m{m} (sglang {rec['sglang']:.2f}, base {rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, "
        f"{rec['wrong']} wrong) KS/ST/D:us {times}")


CANDS_PER_SHAPE = 3   # keeps the sweep inside prepare()'s budget with 35 instances to compile


def prepare(weights):
    for table in (WGF_LAB_CANDIDATES, WG_LAB_CANDIDATES):
        for key in table:
            table[key] = table[key][:CANDS_PER_SHAPE]
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare + WG setup ({len(WG_INSTANCES)} WG, {len(WGF_INSTANCES)} WGF instances) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if WG_MOD is not None:
        lab.section("wgf info", sec_wgf_info)
        for m in (128, 32, 256, 16):
            for layer in ("pooler", "unpooler", "qkv_gate", "o"):
                lab.section(f"wgf {layer} m{m}", sec_wgf, sweep, layer, m, min_left=12.0)
        for (layer, m) in sorted(WG_LAB_CANDIDATES):
            lab.section(f"wg {layer} m{m}", sec_wg, sweep, layer, m, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c["kind"] in ("wgf", "wg"):
            log(f"V8 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
