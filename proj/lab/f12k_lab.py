"""PE42 (P43) lab runtime for F12K (stream-K F12M, proj/f12m/f12k.py, appended before this section): the module builds in
a worker thread during the base prepare; scratch (FP32 piece slots + self-resetting counters) per candidate config, so a
faulty config can't disturb another; run_f12k behind run_choice; per-shape sweep lines; a counters check.
Candidates fall back to the base's entry for their shape (kind f12k without a module, planes or scratch runs it)."""

import concurrent.futures

F12K_MOD = None
F12K_STATE = {"bufs": {}}
F12K_BUILD = {}
F12K_LAB_INSTANCES = ((1, 8, 4), (1, 8, 2), (1, 4, 4), (1, 4, 2), (2, 8, 2), (2, 8, 4), (2, 4, 2))


def f12k_targets(sms):
    """(layer, m) -> candidate configs, the few-tile shapes first (grids from the SM count)."""
    def c(nt, ksw, d, g):
        return dict(kind="f12k", NT=nt, KSW=ksw, D=d, G=g)
    s2, s3, s4 = 2 * sms, 3 * sms, 4 * sms
    return {
        ("qkv_gate", 16): [c(2, 8, 2, s2), c(2, 8, 2, 256), c(2, 8, 4, s2), c(2, 4, 2, s4), c(2, 4, 2, s3), c(2, 8, 2, s3)],
        ("qkv_gate", 8): [c(1, 8, 4, s2), c(1, 8, 4, 256), c(1, 8, 2, s2), c(1, 4, 4, s4), c(1, 4, 4, s3), c(1, 8, 4, s3)],
        ("unpooler", 8): [c(1, 8, 2, 256), c(1, 8, 2, s2), c(1, 4, 2, s4), c(1, 4, 2, 512), c(1, 8, 2, sms), c(1, 8, 4, 128)],
        ("unpooler", 1): [c(1, 8, 2, 256), c(1, 8, 2, s2), c(1, 4, 2, s4), c(1, 4, 2, 512), c(1, 8, 2, sms)],
        ("qkv_gate", 1): [c(1, 8, 4, s2), c(1, 4, 4, s4), c(1, 8, 2, s2)],
    }


def f12k_key(c, layer, n, k):
    return (layer, n, k, c["NT"], c["KSW"], c["D"], c["G"], bool(c.get("HL")))


def f12k_inst(c):
    return (c["NT"], c["KSW"], c["D"], True) if c.get("HL") else (c["NT"], c["KSW"], c["D"])


def f12k_tag(c):
    return f"nt{c['NT']}k{c['KSW']}d{c['D']}g{c['G']}" + ("hl" if c.get("HL") else "")


def f12k_buffers(n, k, c, device):
    n_tiles, chs = n // 16, k // 64
    maxp = max(f12k_pieces(n_tiles, chs, c["G"]))
    return {"slots": torch.zeros(n_tiles * maxp * 8 * c["NT"] * 16, dtype=torch.float32, device=device),
            "cnt": torch.zeros(n_tiles, dtype=torch.int32, device=device), "maxp": maxp}


def f12k_planes(c, layer, state):
    """(weights, records) of a config: the router's hi + lo P9 fragments (HL) or the layer's F12M planes."""
    if c.get("HL"):
        wp = state.get("p9", {}).get(layer) if layer == "router" else None
        return (wp, state.get("p9_dummy")) if wp is not None else (None, None)
    pl = state.get("f12m", {}).get(layer)
    return (pl["p"], pl["rec"]) if pl is not None else (None, None)


def run_f12k(c, x, layer, n, state):
    m, k = x.shape
    wp, rec = f12k_planes(c, layer, state)
    bufs = F12K_STATE["bufs"].get(f12k_key(c, layer, n, k))
    if F12K_MOD is None or wp is None or bufs is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.float32 if c.get("HL") else torch.bfloat16, device=x.device)
    getattr(F12K_MOD, f12k_name(f12k_inst(c)))(wp, rec, x.contiguous(), out, bufs["slots"], bufs["cnt"], m, n, c["G"], bufs["maxp"])
    return out


_run_choice_pre_f12k = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12k":
        return run_f12k(c, x, layer, n, state)
    return _run_choice_pre_f12k(c, x, layer, n, state)


def f12k_build_start():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def timed():
        t0 = time.monotonic()
        mod = f12k_build(F12K_LAB_INSTANCES)
        return mod, time.monotonic() - t0
    F12K_BUILD["fut"] = pool.submit(timed)
    pool.shutdown(wait=False)


def f12k_setup_lab(state, t_start, targets):
    global F12K_MOD
    fut = F12K_BUILD.get("fut")
    if fut is None:
        return
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 170.0 - (time.monotonic() - t_start)))
    if not done:
        log(f"f12k build not finished after {time.monotonic() - t_start:.0f}s: skipped")
        return
    try:
        mod, secs = fut.result()
    except Exception as exc:
        errs = [ln.strip() for ln in str(exc).splitlines() if "error" in ln.lower()][:5]
        log(f"f12k build FAILED ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs)
            + " || tail: " + " ".join(str(exc).split())[-500:])
        return
    attrs = []
    for inst in F12K_LAB_INSTANCES:
        try:
            v = int(getattr(mod, f12k_name(inst) + "_info")())
            attrs.append(f"{'/'.join(map(str, inst))}:{v // 1000000}r+{(v % 1000000) // 100}B occ{v % 100}")
        except Exception as exc:
            attrs.append(f"{'/'.join(map(str, inst))}:info FAILED {type(exc).__name__}")
    log(f"f12k build ok in {secs:.1f}s (parallel to the base prepare); NT/KSW/D:regs+stack CTAs/SM " + " ".join(attrs))
    if not state.get("f12m") and "router" not in state.get("p9", {}):
        log("f12k: no F12M planes and no router fragments in the base state: skipped")
        return
    for (layer, m), cs in targets.items():
        n, k = lab_n(layer, m), FULL[layer][1]
        for c in cs:
            key = f12k_key(c, layer, n, k)
            if key not in F12K_STATE["bufs"]:
                F12K_STATE["bufs"][key] = f12k_buffers(n, k, c, state[layer].device)
    F12K_MOD = mod
    log(f"f12k scratch: {len(F12K_STATE['bufs'])} sets, maxp " + " ".join(
        f"{key[0][:4]}{key[1]}g{key[6]}:{b['maxp']}" for key, b in sorted(F12K_STATE["bufs"].items())[:12]))


def sec_f12k(lab, sweep, targets, layer, m):
    if F12K_MOD is None:
        return
    n, k = lab_n(layer, m), FULL[layer][1]
    cands = [dict(c, fallback=CHOICE.get((layer, m))) for c in targets[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = n * k * 1.5
    times = " ".join(f"{f12k_tag(c)}:{us:.2f}({nb / max(us - 4.13, 0.1) / 1e6:.2f})" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"f12k {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us raw; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) cfg:us raw(12-bit TB/s net of 4.13 us) {times}")


def f12k_dirty():
    return {key: int(b["cnt"].abs().sum().item()) for key, b in F12K_STATE["bufs"].items() if int(b["cnt"].abs().sum().item())}


def sec_f12k_cnt(lab):
    dirty = f12k_dirty()
    log(f"f12k counters: {len(F12K_STATE['bufs']) - len(dirty)} of {len(F12K_STATE['bufs'])} clean"
        + (f"; DIRTY {dirty}" if dirty else ""))
    for (layer, m), c in list(CHOICE.items()):
        if c.get("kind") == "f12k" and f12k_key(c, layer, lab_n(layer, m), FULL[layer][1]) in dirty:
            CHOICE[(layer, m)] = c.get("fallback")
            log(f"f12k {layer} m{m}: chosen config left its counters dirty -> back to the base entry")


_cfg_name_pre_f12k = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12k":
        return "f12k " + f12k_tag(c)
    return _cfg_name_pre_f12k(c)
