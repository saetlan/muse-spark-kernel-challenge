"""PE46 (F12M helper warp for the escape corrections, lab/f12mh.py): base = submission_v12.py (v11b + F12M part[] padded,
PE45) byte for byte; the lab builds the base's F12M instances with f12mh_source (one more warp per CTA computes each row's
escape corrections into shared memory while the others stream; the epilogue adds them without global reads) as module
okc_proj_f12mh and races kind "f12mh" (same configs) against the base's F12M on its six shapes. Outputs differ from F12M only
by FP32 rounding of the corrections, so each candidate goes through the sweep's FP64 checks (normal, probe, outliers); gate 1%.
Winners are printed as "MH CHOICE" lines and the check's own score measures v12 + them."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
F12MH_MOD = None
F12MH_BUILD = {}


def f12mh_src(instances):
    return f12mh_source(instances, f12m_source)


def f12mh_decls(instances):
    return [d.replace("f12m_run_", "f12mh_run_") for d in f12m_decls(instances)]


def f12mh_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12mh_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12mh", cpp_sources="\n".join(decls), cuda_sources=f12mh_src(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mh_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12mh(c, x, layer, n, state):
    """F12M's run_f12m with the helper-warp module (same planes, same instance tuple)."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MH_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MH_MOD, f12m_name(inst).replace("f12m_run_", "f12mh_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_mh = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mh":
        return run_f12mh(c, x, layer, n, state)
    return _run_choice_pre_mh(c, x, layer, n, state)


_cfg_name_pre_mh = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12mh":
        return "f12mh" + _cfg_name_pre_mh(dict(c, kind="f12m"))[4:]
    return _cfg_name_pre_mh(c)


def sec_mh(lab, sweep, state, layer, m):
    base = CHOICE[(layer, m)]
    cand = dict(base, kind="f12mh", fallback=base)
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 8100 + m)
    same = torch.equal(run_choice(cand, x, layer, n, state), run_choice(base, x, layer, n, state))
    sweep.shape(lab, layer, m, [cand])
    rec = sweep.results[(layer, m)]
    # second opinion: alternate base / padded, min of 3 each
    tb = min(lab.time_graph(lambda: run_choice(base, x, layer, n, state)) for _ in range(3))
    tp = min(lab.time_graph(lambda: run_choice(cand, x, layer, n, state)) for _ in range(3))
    log(f"mh {layer} m{m} {cfg_name(base)}: base {rec['cur_us']:.2f} / {tb:.2f}, helper "
        f"{rec['best'][0] if rec['best'] else float('nan'):.2f} / {tp:.2f} us raw ({tb / tp:.3f}x), outputs bit-identical: {same} (rounding may differ; FP64 checks decide)")


def prepare(weights):
    global F12MH_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12mh_build, F12M_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v12) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12MH_MOD, secs = fut.result()
            regs = " ".join(f"{f12m_name(i)[9:]}:{int(getattr(F12MH_MOD, f12m_name(i).replace('f12m_run_', 'f12mh_run_') + '_info')()) // 1000000}r"
                            for i in F12M_INSTANCES)
            log(f"f12mh build ok in {secs:.1f}s; regs {regs}")
        except Exception as exc:
            log(f"f12mh build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12mh build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.01
    sweep = Sweep(state)
    if F12MH_MOD is not None:
        for (layer, m), c in sorted(CHOICE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if c.get("kind") == "f12m":
                lab.section(f"mh {layer} m{m}", sec_mh, sweep, state, layer, m, min_left=5.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12mh":
            log(f"MH CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
