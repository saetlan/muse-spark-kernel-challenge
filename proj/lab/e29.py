"""PE29 (P41): DG12-RS, built by the muse_prefill_layer session (shared/gemm/dg12rs/dgrs.py): 12-bit F12 weights decoded
by the math warps straight into register-A wgmma (weights = A, m64 per warpgroup; tokens = B through a SW128 tile),
128 weight rows x BM tokens per CTA, chunk-major planes bulk-copied per 64-k stage, split-K over K/64 chunks with a
fixed-order last-arriver reduction and F12M's exact epilogue (scale, escapes from rec), plain launch. All four BF16
layers at m = 32/128/256 with a few (BM, splits) per shape against the base (v10). Winners >= 3% enter CHOICE with the
base kernel as fallback, printed as "V23 CHOICE" lines; registers, spills and the kernel's error flag are logged."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DGRS_EXT = None
DGRS_STATE = {}


def dgrs_key(c, layer, m):
    return (layer, m, c["BM"], c["S"])


def dgrs_setup_lab(state):
    global DGRS_EXT
    t0 = time.monotonic()
    planes = {}
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        pl = f12m_device_planes(state[layer], natural=True)
        if pl is not None:
            planes[layer] = {"wp": dgrs_pack(pl["p"]), "rec": pl["rec"]}
    DGRS_STATE["planes"] = planes
    log(f"dgrs planes: {sorted(planes)} in {time.monotonic() - t0:.1f}s")
    t0 = time.monotonic()
    try:
        DGRS_EXT = dgrs_build(bms=(32, 64, 128))
        attrs = " ".join(f"bm{b}:{int(v[0])}r+{int(v[1])}B" for b in (32, 64, 128) for v in [DGRS_EXT.dgrs_attrs(b).tolist()])
        log(f"dgrs build ok in {time.monotonic() - t0:.1f}s; regs/spill {attrs}")
    except Exception as exc:
        errs = [ln.strip() for ln in str(exc).splitlines() if "error" in ln.lower()][:5]
        log(f"dgrs build FAILED after {time.monotonic() - t0:.1f}s ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs)
            + " || tail: " + " ".join(str(exc).split())[-500:])
        DGRS_EXT = None
        return
    bufs = {}
    for (layer, m), cs in DGRS_TARGETS.items():
        if layer not in planes:
            continue
        for c in cs:
            bufs[dgrs_key(c, layer, m)] = dgrs_buffers(m, FULL[layer][0], c["BM"], c["S"], state[layer].device)
    DGRS_STATE["bufs"] = bufs


def run_dgrs(c, x, layer, n, state):
    m, k = x.shape
    pl = DGRS_STATE.get("planes", {}).get(layer)
    bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
    if DGRS_EXT is None or pl is None or bufs is None or n != FULL[layer][0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    return dgrs_run(DGRS_EXT, x.contiguous(), pl["wp"], pl["rec"], out, bufs, c["BM"], c["S"], c.get("SMAX", 8))


_run_choice_pre_dgrs = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dgrs":
        return run_dgrs(c, x, layer, n, state)
    return _run_choice_pre_dgrs(c, x, layer, n, state)


def sec_dgrs(lab, sweep, layer, m):
    if DGRS_EXT is None or layer not in DGRS_STATE.get("planes", {}):
        return
    n, k = FULL[layer]
    cands = [dict(kind="dgrs", fallback=CHOICE.get((layer, m)), **c) for c in DGRS_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = n * k * 1.5
    times = " ".join(f"bm{c['BM']}s{c['S']}:{us:.2f}({nb / us / 1e6:.2f})" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"dgrs {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) bm/splits:us(12-bit TB/s) {times}")


def sec_dgrs_err(lab):
    flags = {key: int(b["err"].item()) for key, b in DGRS_STATE.get("bufs", {}).items()}
    bad = {f"{k[0]} m{k[1]} bm{k[2]} s{k[3]}": v for k, v in flags.items() if v}
    log(f"dgrs error flags: {len(bad)} of {len(flags)} set" + (f": {bad}" if bad else ""))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgrs_setup_lab(state)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in DGRS_TARGETS:
        lab.section(f"dgrs {layer} m{m}", sec_dgrs, sweep, layer, m, min_left=6.0)
    lab.section("dgrs err", sec_dgrs_err, min_left=-1e9)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgrs":
            log(f"V23 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
