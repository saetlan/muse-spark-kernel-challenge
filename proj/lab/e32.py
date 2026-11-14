"""PE32: two independent experiments in one check.
(1) P42, the router at m = 16/32/128/256 with programmatic dependent launch (lab/hlp.py): v10's HL tiles and
atomics, the zero fill launched as our own kernel that releases the GEMM at once (hlp) or v10's torch.zeros (hlpt), the
GEMM launched with launch_pdl=True and gdc_wait() before its atomic adds; v10's config and half the splits.
(2) P41, DG12-RS v4 (lab/dgrs4.py): 2 vs 4 math warpgroups, the DIAG split at unpooler m32 (d0 / d16 no finalize with
the math kept / d1 skeleton), and the best PE31 shapes with math 4 where it fits (BM <= 64, CTAs <= 132).
Winners >= 3% enter CHOICE (fallback: v10's entry), printed as "V25 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0


def hlp_targets():
    t = {}
    for m in (16, 32, 128, 256):
        b = HL_CHOICE[("router", m)]
        bases = [dict(b)]
        if b["split"] >= 4:
            bases.append(dict(b, split=b["split"] // 2))
        t[("router", m)] = [dict(kind=kind, base=bb) for bb in bases for kind in ("hlp", "hlpt")]
    return t


HLP_TARGETS = hlp_targets()


def hlp_tag(c):
    b = c["base"]
    return f"{c['kind']}:{b['kind'][3:]}{b['BM']}x{b['BN']}k{b['BK']}s{b['split']}"


def sec_hlp(lab, sweep, layer, m):
    cands = [dict(c, fallback=CHOICE.get((layer, m))) for c in HLP_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    times = " ".join(f"{hlp_tag(c)}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"hlp {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) {times}")


def prepare(weights):
    t0 = time.monotonic()
    dgrs_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgrs_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in HLP_TARGETS:
        lab.section(f"hlp {layer} m{m}", sec_hlp, sweep, layer, m, min_left=6.0)
    lab.section("dgrs diag", sec_dgrs_diag, min_left=10.0)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") in ("hlp", "hlpt", "dgrs"):
            log(f"V25 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
