"""PE22 (P40 step 1): our own build of the image's DeepGEMM sm90 BF16 GEMM (proj/dg/dgx.py: the header unchanged,
DeepGEMM's heuristic configs, our TMA descriptors and launch) on pooler/qkv/unpooler/o at m = 128, pooler at m = 256
and m = 32, timed next to the base (v10: DeepGEMM's own call at pooler m128, cuBLAS elsewhere). Shows whether our
build path reproduces DeepGEMM and where its skeleton stands per shape, before the 12-bit fork (PE23). Winners >= 3%
enter CHOICE, printed as "V18 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DGX_BUILD_MAX_START_S = 150.0
DGX_BUILD_DEADLINE_S = 250.0   # stop waiting for builds this long into prepare (the 300 s limit)
DGX_MOD = None
DGX_OK = set()


def dgx_inst(c, n, k):
    return (c["BM"], c["BN"], c["ST"], c["SWD"], c["NMATH"], c["MC"], c["MN"], n, k)


def dgx_setup_lab(t_start):
    global DGX_MOD
    log("dgx recon: " + dgx_recon())
    if time.monotonic() - t_start > DGX_BUILD_MAX_START_S:
        log(f"dgx build skipped: {time.monotonic() - t_start:.0f}s already used")
        return
    insts = tuple(dict.fromkeys(dgx_inst(c, FULL[layer][0], FULL[layer][1]) for (layer, m), c in DGX_TARGETS.items()))
    t0 = time.monotonic()
    try:
        mod, errors = dgx_build(insts, timeout=max(10.0, DGX_BUILD_DEADLINE_S - (time.monotonic() - t_start)))
    except Exception as exc:
        log(f"dgx build FAILED to start ({type(exc).__name__}): {' '.join(str(exc).split())[-500:]}")
        return
    shown = 0
    for name, exc in errors.items():
        text = str(exc)
        if shown < 2:
            errs = [ln.strip() for ln in text.splitlines() if "error" in ln.lower()][:4]
            log(f"dgx {name} FAILED ({type(exc).__name__}): " + " | ".join(e[-260:] for e in errs)
                + " || tail: " + " ".join(text.split())[-500:])
        shown += 1
    ok = [inst for inst in insts if dgx_name(inst) not in errors]
    regs = []
    for inst in ok:
        try:
            getattr(mod, dgx_name(inst) + "_attr")()
            v = int(getattr(mod, dgx_name(inst) + "_info")())
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}:{v // 1000000}r" + (f"+{v % 1000000}B" if v % 1000000 else ""))
        except Exception as exc:
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}: attr FAILED {' '.join(str(exc).split())[-120:]}")
    log(f"dgx build: {len(ok)}/{len(insts)} instances ok in {time.monotonic() - t0:.1f}s; regs {' '.join(regs)}")
    if ok:
        DGX_MOD = mod
        DGX_OK.update(dgx_name(inst) for inst in ok)


def run_dgx(c, x, layer, n, state):
    m, k = x.shape
    if DGX_MOD is None or dgx_name(dgx_inst(c, n, k)) not in DGX_OK:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(DGX_MOD, dgx_name(dgx_inst(c, n, k)))(x.contiguous(), state[layer], out)
    return out


_run_choice_pre_dgx = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dgx":
        return run_dgx(c, x, layer, n, state)
    return _run_choice_pre_dgx(c, x, layer, n, state)


def sec_dgx(lab, sweep, layer, m):
    if DGX_MOD is None or dgx_name(dgx_inst(DGX_TARGETS[(layer, m)], FULL[layer][0], FULL[layer][1])) not in DGX_OK:
        return
    c = dict(kind="dgx", fallback=CHOICE.get((layer, m)), **DGX_TARGETS[(layer, m)])
    sweep.shape(lab, layer, m, [c])
    rec = sweep.results[(layer, m)]
    nb = lab_n(layer, m) * FULL[layer][1] * 2
    times = " ".join(f"{us:.2f}({nb / us / 1e6:.2f} TB/s)" for us, _ in rec.get("all", []))
    log(f"dgx {layer} m{m} BM{c['BM']} BN{c['BN']} st{c['ST']} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} "
        f"{rec['cur_us']:.2f} us; {rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) dgx {times}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dgx_setup_lab(t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in DGX_TARGETS:
        lab.section(f"dgx {layer} m{m}", sec_dgx, sweep, layer, m, min_left=6.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dgx":
            log(f"V18 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
