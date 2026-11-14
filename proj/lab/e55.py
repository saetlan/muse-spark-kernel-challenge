"""PE55 (capped helper on the unpooler slice): v14 runs the unpooler's 1024-row slice at m8 on padded F12M NT1 KSW8 D4
(PE49); the helper warp (capped at 2 CTAs per SM, PE53) gained on o m1 / o m8 / qkv m16 where the escape step's latency sits
at the tail. Races the capped helper version of that config on unpooler m8 (vs v14's F12M) and m1 (vs SGLang's tiny GEMM).
Base = proj/submission_v14.py byte for byte; e53's code (module okc_proj_f12mh2, kind f12mh2); gate 2%. Winners -> "MH2 CHOICE"."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
F12MH2_MOD = None
F12MH2_BUILD = {}


def f12mh2_src(instances):
    src = f12mh_source(instances, f12m_source)
    cap_old = "__launch_bounds__(RG * KSW * 32 + 32)"
    assert src.count(cap_old) == 1
    src = src.replace(cap_old, "__launch_bounds__(RG * KSW * 32 + 32, 2)")   # PE53: <= 113 registers, 2 CTAs per SM
    return (src.replace("namespace f12mh {", "namespace f12mh2 {")
            .replace("f12mh::", "f12mh2::").replace("void f12mh_run_", "void f12mh2_run_").replace("int64_t f12mh_run_", "int64_t f12mh2_run_"))


def f12mh2_decls(instances):
    return [d.replace("f12m_run_", "f12mh2_run_") for d in f12m_decls(instances)]


def f12mh2_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12mh2_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12mh2", cpp_sources="\n".join(decls), cuda_sources=f12mh2_src(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mh2_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12mh2(c, x, layer, n, state):
    """F12M's run_f12m with the helper-warp module (same planes, same instance tuple)."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MH2_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MH2_MOD, f12m_name(inst).replace("f12m_run_", "f12mh2_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_mh2 = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mh2":
        return run_f12mh2(c, x, layer, n, state)
    return _run_choice_pre_mh2(c, x, layer, n, state)


_cfg_name_pre_mh2 = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12mh2":
        return "f12mh2" + _cfg_name_pre_mh2(dict(c, kind="f12m"))[4:]
    return _cfg_name_pre_mh2(c)


E48_TARGETS = {
    ("unpooler", 8): [dict(kind="f12mh2", NT=1, RTW=1, RG=1, KSW=8, D=4, F12=True)],
    ("unpooler", 1): [dict(kind="f12mh2", NT=1, RTW=1, RG=1, KSW=8, D=4, F12=True)],
}
E48_K = {"o": 1024, "qkv_gate": 8192, "pooler": 8192, "unpooler": 4096}
E48_INSTANCES = tuple(dict.fromkeys((c["NT"], c["RTW"], c["RG"], c["KSW"], E48_K[l] // 64 // c["KSW"], c["D"], c["F12"])
                                    + ((1, c["XD"]) if c.get("XD", c["D"]) != c["D"] else ())
                                    for (l, m), cs in E48_TARGETS.items() for c in cs))


def sec_mh2(lab, sweep, state, layer, m):
    cands = [dict(c, fallback=CHOICE.get((layer, m))) for c in E48_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    times = " ".join(f"{cfg_name(c)[7:]}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"mh2 {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us raw; {rec['n']} ok, "
        f"{rec['fails']} failed, {rec['wrong']} wrong) {times}")


def prepare(weights):
    global F12MH2_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12mh2_build, E48_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v14) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12MH2_MOD, secs = fut.result()
            regs = " ".join(f"{f12m_name(i)[9:]}:{int(getattr(F12MH2_MOD, f12m_name(i).replace('f12m_run_', 'f12mh2_run_') + '_info')()) // 1000000}r"
                            for i in E48_INSTANCES)
            log(f"f12mh2 build ok in {secs:.1f}s; regs {regs}")
        except Exception as exc:
            log(f"f12mh2 build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12mh2 build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.02
    sweep = Sweep(state)
    if F12MH2_MOD is not None:
        for (layer, m) in E48_TARGETS:
            lab.section(f"mh2 {layer} m{m}", sec_mh2, sweep, state, layer, m, min_left=6.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12mh2":
            log(f"MH2 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
