"""DG12-RS v10 lab section (P41; pinned copy proj/dg/pinned/dgrs_v10.py of muse_prefill_layer's v10: instances up to
(BM, CPS, DIAG, MATH, RED, SPL, SMX, M, N, K), every set field fixed at compile time, the most specialized matching instance
wins the dispatch). A/B of v9-style RED / SPL instances (module "dgrs_ext_spec") against fully specialized twins with the ring
depth and M / N / K also fixed (module "dgrs_ext_full"), each config routed to its module by MOD; both modules build in worker
threads started before the base prepare."""

import concurrent.futures

DGRS_EXT = {}       # "spec" / "full" -> module
DGRS_STATE = {}
DGRS_BUILD = {}
DGRS_FUNCS = ["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_attrs_x", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"]
DGRS_SMX = 8   # the smax every call passes (dgrs_run's default)


def dgrs_inst(c):
    t = (c["BM"], c.get("CPS", 1), c.get("DIAG", 0), c.get("MATH", 2), c.get("RED", 0), c["S"])
    return t + (DGRS_SMX, c["M"], c["N"], c["K"]) if c["MOD"] == "full" else t


def dgrs_cfg_insts():
    spec, full = set(), set()
    for cs in DGRS_TARGETS.values():
        for c in cs:
            (full if c["MOD"] == "full" else spec).add(dgrs_inst(c))
    return tuple(sorted(spec)), tuple(sorted(full))


def dgrs_build_mod(name, insts):
    return load_inline(
        name=name, cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(insts), functions=DGRS_FUNCS,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=False)


def dgrs_build_start():
    spec, full = dgrs_cfg_insts()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)   # as dgx.py's builds (judge-accepted in PE22-PE28)

    def timed(name, insts):
        t0 = time.monotonic()
        mod = dgrs_build_mod(name, insts)
        return mod, time.monotonic() - t0
    if spec:
        DGRS_BUILD["spec"] = pool.submit(timed, "dgrs_ext_spec", spec)
    if full:
        DGRS_BUILD["full"] = pool.submit(timed, "dgrs_ext_full", full)
    pool.shutdown(wait=False)


def dgrs_key(c, layer, m):
    return (layer, m, c["BM"], c["S"], c.get("CPS", 1), c.get("RED", 0), c.get("MATH", 2), c.get("DIAG", 0), c["MOD"])


def dgrs_tag(c):
    return (f"bm{c['BM']}s{c['S']}r{c.get('RED', 0)}m{c.get('MATH', 2)}" + (f"d{c['DIAG']}" if c.get("DIAG") else "")
            + ("F" if c["MOD"] == "full" else "S"))


def dgrs_setup_lab(state, t_start):
    spec, full = dgrs_cfg_insts()
    for key, insts in (("spec", spec), ("full", full)):
        fut = DGRS_BUILD.get(key)
        if fut is None:
            continue
        done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t_start)))
        if not done:
            log(f"dgrs {key} build not finished after {time.monotonic() - t_start:.0f}s: skipped")
            continue
        try:
            mod, secs = fut.result()
        except Exception as exc:
            errs = [ln.strip() for ln in str(exc).splitlines() if "error" in ln.lower()][:5]
            log(f"dgrs {key} build FAILED ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs)
                + " || tail: " + " ".join(str(exc).split())[-500:])
            continue
        attrs = []
        for t in insts:
            try:
                tt = tuple(t) + (0, 0, 0, 0)[: 10 - len(t)]
                v = mod.dgrs_attrs_x(*tt).tolist()
                attrs.append(f"{'/'.join(map(str, t))}:{int(v[0])}r+{int(v[1])}B")
            except Exception as exc:
                attrs.append(f"{'/'.join(map(str, t))}:attr FAILED {type(exc).__name__}")
        log(f"dgrs {key} build ok in {secs:.1f}s (in parallel with the base prepare); BM/CPS/DIAG/MATH/RED/SPL[/SMX/M/N/K]:regs+stack "
            + " ".join(attrs))
        DGRS_EXT[key] = mod
    if not DGRS_EXT:
        return
    ext = DGRS_EXT.get("spec") or DGRS_EXT.get("full")
    stg = []
    for (layer, m), cs in DGRS_TARGETS.items():
        for c in cs:
            try:
                v = int(ext.dgrs_stages_d(c["BM"], c.get("CPS", 1), c.get("MATH", 2), c["S"], c.get("RED", 0), 8, c.get("DIAG", 0)))
            except Exception as exc:
                v = f"FAILED {type(exc).__name__}"
            stg.append(f"{layer[:4]}{m}/{dgrs_tag(c)}:{v}")
    log("dgrs ring stages (smax 8): " + " ".join(stg))
    occ = {}
    for cs in DGRS_TARGETS.values():
        for c in cs:
            if c["S"] > 1 and c.get("RED", 0) in (0, 2):
                key = (c["BM"], c.get("CPS", 1), c.get("MATH", 2), c["S"])
                if key not in occ:
                    try:
                        occ[key] = int(ext.dgrs_max_clusters(c["BM"], c.get("CPS", 1), 0, c.get("MATH", 2), c["S"]))
                    except Exception as exc:
                        occ[key] = f"FAILED {type(exc).__name__}"
    log("dgrs max active clusters BM/CPS/MATH/S: " + " ".join(f"{'/'.join(map(str, k))}:{v}" for k, v in occ.items()))
    t0 = time.monotonic()
    planes = {}
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        pl = f12m_device_planes(state[layer], natural=True)
        if pl is not None:
            planes[layer] = {"wp": dgrs_pack(pl["p"]), "rec": pl["rec"]}
    bufs = {}
    for (layer, m), cs in DGRS_TARGETS.items():
        if layer in planes:
            for c in cs:
                bufs[dgrs_key(c, layer, m)] = dgrs_buffers(m, FULL[layer][0], c["BM"], c["S"], state[layer].device)
    DGRS_STATE["planes"] = planes
    DGRS_STATE["bufs"] = bufs
    log(f"dgrs planes {sorted(planes)} and {len(bufs)} buffer sets in {time.monotonic() - t0:.1f}s")


def dgrs_call(c, x, pl, out, bufs):
    ext = DGRS_EXT[c["MOD"]]
    return dgrs_run(ext, x, pl["wp"], pl["rec"], out, bufs, c["BM"], c["S"], smax=DGRS_SMX, cps=c.get("CPS", 1),
                    red=c.get("RED", 0), math=c.get("MATH", 2), diag=c.get("DIAG", 0))


def run_dgrs(c, x, layer, n, state):
    m, k = x.shape
    pl = DGRS_STATE.get("planes", {}).get(layer)
    bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
    if c["MOD"] not in DGRS_EXT or pl is None or bufs is None or n != FULL[layer][0] or (c["MOD"] == "full" and m != c["M"]):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    return dgrs_call(c, x.contiguous(), pl, out, bufs)


_run_choice_pre_dgrs = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dgrs":
        return run_dgrs(c, x, layer, n, state)
    return _run_choice_pre_dgrs(c, x, layer, n, state)


def sec_dgrs(lab, sweep, layer, m):
    if not DGRS_EXT or layer not in DGRS_STATE.get("planes", {}):
        return
    n, k = FULL[layer]
    cands = [dict(kind="dgrs", fallback=CHOICE.get((layer, m)), **c) for c in DGRS_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = n * k * 1.5
    times = " ".join(f"{dgrs_tag(c)}:{us:.2f}({nb / us / 1e6:.2f})" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"dgrs {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) bm/splits/red/math + S (RED/SPL) or F (+SMX/M/N/K):us(12-bit TB/s) {times}")


def sec_dgrs_err(lab):
    flags = {key: int(b["err"].item()) for key, b in DGRS_STATE.get("bufs", {}).items()}
    bad = {"/".join(str(v) for v in k): v for k, v in flags.items() if v}
    log(f"dgrs error flags: {len(bad)} of {len(flags)} set" + (f": {bad}" if bad else ""))
