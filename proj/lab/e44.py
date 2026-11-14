"""PE44 (diagnostic): where the 12-bit kernels' ~2 us above a pure read goes. v11b byte for byte as the timed path; the lab
rebuilds v11b's own F12M and F12V instances three ways from the base's source text (namespaces renamed, separate modules):
  e0  unchanged (must time like v11b's kernel: the control);
  e1  escape loop removed (no x reads for out-of-window weights in the epilogue; output wrong, timing only);
  e2  e1 + no KSW sum (one part read per output instead of KSW).
and times them back to back on every shape v11b runs on F12M / F12V (two rounds, min per variant). If e1 is clearly faster
than e0, precomputing the escape corrections off the critical path pays; if e2 is, the reduction does."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
DIAG = {}
DIAG_BUILD = {}
KSW_SUM_M = "for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];"
KSW_SUM_V = "for (int p = 0; p < KSW; ++p) sum += red[p][tid];"
ESC = "if (j < count) {"
VARIANTS = {"e0": [], "e1": [(ESC, "if (j < count && false) {")],
            "e2": [(ESC, "if (j < count && false) {"), (KSW_SUM_M, "sum = part[0][row][tok];"), (KSW_SUM_V, "sum = red[0][tid];")]}


def diag_source(fam, tag):
    if fam == "f12m":
        src, decls = f12m_source(F12M_INSTANCES), f12m_decls(F12M_INSTANCES)
        own = (ESC, KSW_SUM_M)
    else:
        src, decls = f12v_source(F12V_INSTANCES), f12v_decls(F12V_INSTANCES)
        own = (ESC, KSW_SUM_V)
    for a, b in VARIANTS[tag]:
        if a in own:
            assert src.count(a) == 1, (fam, tag, a)
            src = src.replace(a, b)
    ns = fam + tag
    assert src.count(f"namespace {fam} {{") == 1
    src = (src.replace(f"namespace {fam} {{", f"namespace {ns} {{").replace(f"{fam}::", f"{ns}::")
           .replace(f"void {fam}_run_", f"void {ns}_run_").replace(f"int64_t {fam}_run_", f"int64_t {ns}_run_"))
    decls = [d.replace(f"{fam}_run_", f"{ns}_run_") for d in decls]
    return src, decls


def diag_build(fam, tag):
    from torch.utils.cpp_extension import load_inline
    src, decls = diag_source(fam, tag)
    t0 = time.monotonic()
    mod = load_inline(name=f"okc_diag_{fam}{tag}", cpp_sources="\n".join(decls), cuda_sources=src,
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix=f"okc_diag_{fam}{tag}_"), verbose=False)
    return mod, time.monotonic() - t0


def diag_build_start():
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=3)
    for fam in ("f12m", "f12v"):
        for tag in VARIANTS:
            DIAG_BUILD[(fam, tag)] = pool.submit(diag_build, fam, tag)
    pool.shutdown(wait=False)


def diag_collect(t_start):
    secs = []
    for key, fut in DIAG_BUILD.items():
        done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t_start)))
        if not done:
            log(f"diag {key} build not finished: skipped")
            continue
        try:
            mod, s = fut.result()
        except Exception as exc:
            log(f"diag {key} build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
            continue
        DIAG[key] = mod
        secs.append(f"{key[0]}{key[1]}:{s:.0f}s")
    log("diag builds " + " ".join(secs))


def diag_calls(c, layer, m, state):
    """{variant: fn} for one chosen F12M / F12V shape, each writing into its own output."""
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 6100 + m)
    calls = {}
    if c["kind"] == "f12m":
        pl = state.get("f12m", {}).get(layer)
        if pl is None:
            return {}, x
        cs, xd = c.get("CS", 1), c.get("XD", c["D"])
        inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
        inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
        name, args = f12m_name(inst), (pl["p"], pl["rec"])
        tail = (m, n)
    else:
        pl = state.get("f12v", {}).get(layer)
        if pl is None:
            return {}, x
        inst = (c["R"], c["RG"], c["KSW"], k // 1024 // c["KSW"], c["D"])
        name, args = f12v_name(inst), (pl["p"], pl["rec"])
        tail = (n,)
    fam = c["kind"]
    for tag in VARIANTS:
        mod = DIAG.get((fam, tag))
        if mod is None:
            continue
        fn = getattr(mod, name.replace(f"{fam}_run_", f"{fam}{tag}_run_"))
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)

        def call(fn=fn, out=out):
            fn(*args, x, out, *tail)
            return out
        calls[tag] = call
    return calls, x


def sec_diag(lab, state, layer, m):
    c = CHOICE[(layer, m)]
    n = lab_n(layer, m)
    calls, x = diag_calls(c, layer, m, state)
    if not calls:
        return
    ref = run_choice(c, x, layer, n, state)
    e0_same = "e0" in calls and torch.equal(calls["e0"](), ref)
    best = {}
    t_base = lab.time_graph(lambda: run_choice(c, x, layer, n, state))
    for _ in range(2):
        for tag, fn in calls.items():
            us = lab.time_graph(fn)
            best[tag] = min(best.get(tag, 1e9), us)
    e0 = best.get("e0")
    parts = " ".join(f"{t} {us:.2f}" + (f" ({us - e0:+.2f})" if e0 and t != "e0" else "") for t, us in sorted(best.items()))
    log(f"diag {layer} m{m} {cfg_name(c)}: base {t_base:.2f} | {parts} us raw | e0 output == base: {e0_same}")


def prepare(weights):
    t0 = time.monotonic()
    diag_build_start()
    state = _submission_prepare(weights)
    log(f"base prepare (v11b) took {time.monotonic() - t0:.1f}s")
    diag_collect(t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    for (layer, m), c in sorted(CHOICE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if c.get("kind") in ("f12m", "f12v"):
            lab.section(f"diag {layer} m{m}", sec_diag, state, layer, m, min_left=5.0)
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
