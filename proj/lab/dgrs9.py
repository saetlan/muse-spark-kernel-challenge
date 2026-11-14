"""DG12-RS v9 lab section (P41; pinned copy proj/dg/pinned/dgrs_v9.py of muse_prefill_layer's shared/gemm/dg12rs/dgrs.py v9:
DIAG 512 = escape corrections from the ring's x tiles (pre_ring, no global gathers), folded into the partials; 6-tuple
instances (BM, CPS, DIAG, MATH, RED, SPL) specialized for one reduction mode / split count, which dgrs_gemm prefers when one
matches). Because a specialized instance always wins the dispatch, the specialized instances are built in a second module
("dgrs_ext_spec", same source generator) so generic and specialized builds of the same config can be timed side by side.
Both modules build in worker threads started before the base prepare. Configs carry DIAG and SPEC."""

import concurrent.futures

DGRS_EXT = {}       # "gen" / "spec" -> module
DGRS_STATE = {}
DGRS_BUILD = {}
DGRS_FUNCS = ["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"]
DGRS_DIAGS = [dict(layer=layer, m=32, BM=32, S=2, CPS=1,
                   runs=tuple((b, 4, 2, sp) for sp in (False, True) for b in (0, 512, 576)))
              for layer in ("unpooler", "o")]


def dgrs_inst(c):
    t = (c["BM"], c.get("CPS", 1), c.get("DIAG", 0), c.get("MATH", 2))
    return t + (c.get("RED", 0), c["S"]) if c.get("SPEC") else t


def dgrs_diag_cfg(d, b, mm, r, sp):
    return dict(BM=d["BM"], S=d["S"], CPS=d["CPS"], RED=r, MATH=mm, DIAG=b, SPEC=sp)


def dgrs_cfg_insts():
    gen, spec = set(), set()
    for cs in DGRS_TARGETS.values():
        for c in cs:
            (spec if c.get("SPEC") else gen).add(dgrs_inst(c))
    for d in DGRS_DIAGS:
        for b, mm, r, sp in d["runs"]:
            (spec if sp else gen).add(dgrs_inst(dgrs_diag_cfg(d, b, mm, r, sp)))
    return tuple(sorted(gen)), tuple(sorted(spec))


def dgrs_build_spec(insts):
    return load_inline(
        name="dgrs_ext_spec", cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(insts), functions=DGRS_FUNCS,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="dgrs_spec_"), verbose=False)


def dgrs_build_start():
    gen, spec = dgrs_cfg_insts()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)   # as dgx.py's builds (judge-accepted in PE22-PE28)

    def timed(fn, insts):
        t0 = time.monotonic()
        mod = fn(insts)
        return mod, time.monotonic() - t0
    if gen:
        DGRS_BUILD["gen"] = pool.submit(timed, dgrs_build, gen)
    if spec:
        DGRS_BUILD["spec"] = pool.submit(timed, dgrs_build_spec, spec)
    pool.shutdown(wait=False)


def dgrs_key(c, layer, m):
    return (layer, m, c["BM"], c["S"], c.get("CPS", 1), c.get("RED", 0), c.get("MATH", 2), c.get("DIAG", 0),
            bool(c.get("SPEC")))


def dgrs_tag(c):
    return (f"bm{c['BM']}s{c['S']}r{c.get('RED', 0)}m{c.get('MATH', 2)}" + (f"d{c['DIAG']}" if c.get("DIAG") else "")
            + ("S" if c.get("SPEC") else ""))


def dgrs_setup_lab(state, t_start):
    gen, spec = dgrs_cfg_insts()
    for key, insts in (("gen", gen), ("spec", spec)):
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
                v = mod.dgrs_attrs_s(t[0], t[1], t[2], t[3], t[4] if len(t) > 4 else -1, t[5] if len(t) > 4 else 0).tolist()
                attrs.append(f"{'/'.join(map(str, t))}:{int(v[0])}r+{int(v[1])}B")
            except Exception as exc:
                attrs.append(f"{'/'.join(map(str, t))}:attr FAILED {type(exc).__name__}")
        log(f"dgrs {key} build ok in {secs:.1f}s (in parallel with the base prepare); BM/CPS/DIAG/MATH[/RED/SPL]:regs+stack "
            + " ".join(attrs))
        DGRS_EXT[key] = mod
    if not DGRS_EXT:
        return
    ext = DGRS_EXT.get("gen") or DGRS_EXT.get("spec")
    stg = []
    for (layer, m), cs in DGRS_TARGETS.items():
        for c in cs:
            try:
                v = int(ext.dgrs_stages_d(c["BM"], c.get("CPS", 1), c.get("MATH", 2), c["S"], c.get("RED", 0), 8, c.get("DIAG", 0)))
            except Exception as exc:
                v = f"FAILED {type(exc).__name__}"
            stg.append(f"{layer[:4]}{m}/{dgrs_tag(c)}:{v}")
    log("dgrs ring stages (smax 8): " + " ".join(stg))
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
    for d in DGRS_DIAGS:
        for b, mm, r, sp in d["runs"]:
            c = dgrs_diag_cfg(d, b, mm, r, sp)
            bufs.setdefault(dgrs_key(c, d["layer"], d["m"]),
                            dgrs_buffers(d["m"], FULL[d["layer"]][0], d["BM"], d["S"], state[d["layer"]].device))
    DGRS_STATE["planes"] = planes
    DGRS_STATE["bufs"] = bufs
    log(f"dgrs planes {sorted(planes)} and {len(bufs)} buffer sets in {time.monotonic() - t0:.1f}s")


def dgrs_call(c, x, pl, out, bufs):
    ext = DGRS_EXT.get("spec" if c.get("SPEC") else "gen")
    return dgrs_run(ext, x, pl["wp"], pl["rec"], out, bufs, c["BM"], c["S"], cps=c.get("CPS", 1), red=c.get("RED", 0),
                    math=c.get("MATH", 2), diag=c.get("DIAG", 0))


def run_dgrs(c, x, layer, n, state):
    m, k = x.shape
    pl = DGRS_STATE.get("planes", {}).get(layer)
    bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
    if ("spec" if c.get("SPEC") else "gen") not in DGRS_EXT or pl is None or bufs is None or n != FULL[layer][0]:
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
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) bm/splits/red/math[d][S]:us(12-bit TB/s) {times}")


def sec_dgrs_diag(lab, d):
    layer, m = d["layer"], d["m"]
    pl = DGRS_STATE.get("planes", {}).get(layer)
    if not DGRS_EXT or pl is None:
        return
    n, k = FULL[layer]
    x = lab_input(m, k, 4300 + m)
    parts = []
    for b, mm, r, sp in d["runs"]:
        c = dgrs_diag_cfg(d, b, mm, r, sp)
        tag = f"d{b}{'S' if sp else 'G'}"
        bufs = DGRS_STATE["bufs"].get(dgrs_key(c, layer, m))
        if bufs is None or ("spec" if sp else "gen") not in DGRS_EXT:
            parts.append(f"{tag}:not built")
            continue
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

        def run(c=c, bufs=bufs, out=out):
            return dgrs_call(c, x, pl, out, bufs)
        try:
            run()
            parts.append(f"{tag}:{lab.time_graph(run):.2f}")
        except Exception as exc:
            parts.append(f"{tag}:ERR {type(exc).__name__} {_error_text(exc, 120)}")
    log(f"dgrs diag {layer} m{m} bm{d['BM']}s{d['S']}r2m4: " + " ".join(parts)
        + "  [G generic / S specialized (RED 2, SPL 2); d0 full, d512 ring corrections, d576 = 512 + stamps]")


def sec_dgrs_err(lab):
    flags = {key: int(b["err"].item()) for key, b in DGRS_STATE.get("bufs", {}).items()}
    bad = {"/".join(str(v) for v in k): v for k, v in flags.items() if v}
    log(f"dgrs error flags: {len(bad)} of {len(flags)} set" + (f": {bad}" if bad else ""))
