"""PE19: (1) F12M with a shallower x ring (XD < D) so the weight ring goes deeper at the same registers (PF37: the
total weight bytes in flight, not the work split, sets the m <= 16 rate) on qkv/pooler m8/16, unpooler m16, o m8/16;
(2) the router's hi/lo Triton kernels (PE18 winners in the base) refined around each shape's best config at
m = 256/128/32/16/8. Timed against SGLang and the base; winners >= 3% enter CHOICE with the base kernel as
fallback, printed as "V15 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
E19_STATE = {}
XD_ORDER = [("qkv_gate", 16), ("qkv_gate", 8), ("unpooler", 16), ("pooler", 16), ("pooler", 8), ("o", 8), ("o", 16)]
HL_SEEDS = {   # PE18's best per router shape (lab): refine around these
    256: [dict(kind="hl_gemm", BM=256, BN=32, BK=64, split=16, warps=8, stages=3)],
    128: [dict(kind="hl_gemm", BM=128, BN=32, BK=128, split=16, warps=8, stages=3)],
    32: [dict(kind="hl_swap", BM=32, BN=64, BK=128, split=32, warps=4, stages=4),
         dict(kind="hl_gemm", BM=32, BN=64, BK=128, split=32, warps=4, stages=3)],
    16: [dict(kind="hl_swap", BM=16, BN=64, BK=128, split=32, warps=4, stages=3),
         dict(kind="hl_gemm", BM=16, BN=32, BK=128, split=32, warps=4, stages=3)],
    8: [dict(kind="hl_swap", BM=16, BN=64, BK=128, split=32, warps=4, stages=4),
        dict(kind="hl_gemm", BM=16, BN=64, BK=128, split=32, warps=4, stages=4)],
}
HL_AXES = {"BN": (16, 32, 64, 128), "BK": (32, 64, 128, 256), "split": (8, 16, 32, 64), "warps": (2, 4, 8),
           "stages": (2, 3, 4, 5, 6)}


def hl_ok(c, m):
    bm, bn, bk, split = c["BM"], c["BN"], c["BK"], c["split"]
    if 8192 % (split * bk) or 8192 // split < bk or (c["kind"] == "hl_swap" and bn < 32):
        return False
    per_thread = bm * bn // (c["warps"] * 32)
    return 2 <= per_thread <= 128 and c["stages"] * (bm * bk * 2 + 2 * bk * bn * 2) <= 200 * 1024


def hl_refine(m, max_n=40):
    """Configs one or two steps from a seed along HL_AXES (one-step first), plus BM halved for m >= 128."""
    out = []
    for seed in HL_SEEDS[m]:
        for axis, vals in HL_AXES.items():
            for v in vals:
                c = dict(seed, **{axis: v})
                if c != seed and hl_ok(c, m) and c not in out:
                    out.append(c)
    for seed in HL_SEEDS[m]:
        if m >= 128:
            c = dict(seed, BM=seed["BM"] // 2)
            if hl_ok(c, m) and c not in out:
                out.append(c)
        for a1, a2 in (("BN", "split"), ("BK", "stages"), ("split", "BK"), ("warps", "stages")):
            for v1 in HL_AXES[a1]:
                for v2 in HL_AXES[a2]:
                    c = dict(seed, **{a1: v1, a2: v2})
                    if c != seed and hl_ok(c, m) and c not in out:
                        out.append(c)
    return HL_SEEDS[m] + out[:max_n - len(HL_SEEDS[m])]


def sec_xd_info(lab):
    rows = []
    for inst in F12M_INSTANCES:
        v = int(getattr(F12M_MOD, f12m_name(inst) + "_info")())
        regs, local = v // 1000000, v % 1000000
        rows.append("/".join(str(int(e)) for e in inst) + f":{regs}r" + (f"+{local}B" if local else ""))
    log("f12m regs (NT/RTW/RG/KSW/NS/D/F12[/CS/XD]): " + " ".join(rows))


def sec_xd(lab, sweep, layer, m):
    fallback = CHOICE.get((layer, m))
    cands = [dict(kind="f12m", fallback=fallback, **c) for c in F12M_XD_CANDIDATES.get((layer, m), [])]
    if not cands or F12M_MOD is None or layer not in E19_STATE.get("f12m", {}):
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = lab_n(layer, m) * FULL[layer][1] * 1.5
    times = " ".join(f"{c['NT']}/{c['RTW']}/{c['D']}/{c['XD']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"xd {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) NT/RTW/D/XD:us(12-bit TB/s) {times}")


def hl_name(c):
    return f"{'g' if c['kind'] == 'hl_gemm' else 's'}{c['BM']}/{c['BN']}/{c['BK']}/{c['split']}/{c['warps']}/{c['stages']}"


def sec_hl2(lab, sweep, m):
    sweep.shape(lab, "router", m, hl_refine(m))
    rec = sweep.results[("router", m)]
    times = " ".join(f"{hl_name(c)}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0])[:10])
    log(f"hl2 router m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) BM/BN/BK/split/warps/stages:us {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    E19_STATE.update(state)
    log(f"base prepare (F12M {len(F12M_INSTANCES)} instances, router hl) took {time.monotonic() - t0:.1f}s")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if F12M_MOD is not None:
        lab.section("xd info", sec_xd_info)
        for layer, m in XD_ORDER:
            lab.section(f"xd {layer} m{m}", sec_xd, sweep, layer, m, min_left=10.0)
    else:
        log("f12m: no module (build failed)")
    for m in (256, 128, 32, 16, 8):
        lab.section(f"hl2 router m{m}", sec_hl2, sweep, m, min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") in ("hl_gemm", "hl_swap") or (c.get("kind") == "f12m" and "XD" in c):
            log(f"V15 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
