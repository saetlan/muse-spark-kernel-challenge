"""PE56 (unpooler m32 on padded F12M): PE54's pooler m32 winner (NT4 RTW1 KSW8 D1, +5.1% vs DG12-RS) was never tried on
unpooler m32 (PE47's configs there lost 3.5% to DG12-RS). Base = proj/submission_v15.py byte for byte; e47's code with NT4 RTW1 KSW8
D1 / D2 and NT4 RTW1 RG2 KSW4 D2 on unpooler m32 (module okc_proj_f12mq, kind f12mq); gate 2%. Winner -> "MQ CHOICE"."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
F12MQ_MOD = None
F12MQ_BUILD = {}


def f12mq_source(instances):
    src = f12m_source(instances)
    assert "__shared__ float part[KSW][RB][MT + 3];" in src and src.count("namespace f12m {") == 1, "base F12M is not v12's padded one"
    return (src.replace("namespace f12m {", "namespace f12mq {").replace("f12m::", "f12mq::")
            .replace("void f12m_run_", "void f12mq_run_").replace("int64_t f12m_run_", "int64_t f12mq_run_"))


def f12mq_decls(instances):
    return [d.replace("f12m_run_", "f12mq_run_") for d in f12m_decls(instances)]


def f12mq_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12mq_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12mq", cpp_sources="\n".join(decls), cuda_sources=f12mq_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mq_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12mq(c, x, layer, n, state):
    """F12M's run_f12m on the lab module (same planes, same instance tuple)."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MQ_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MQ_MOD, f12m_name(inst).replace("f12m_run_", "f12mq_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_mq = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mq":
        return run_f12mq(c, x, layer, n, state)
    return _run_choice_pre_mq(c, x, layer, n, state)


_cfg_name_pre_mq = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12mq":
        return "f12mq" + _cfg_name_pre_mq(dict(c, kind="f12m"))[4:]
    return _cfg_name_pre_mq(c)


def sec_mq(lab, sweep, state, layer, m):
    cands = [dict(c, fallback=CHOICE.get((layer, m))) for c in E47_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    n, k = lab_n(layer, m), FULL[layer][1]
    times = " ".join(f"{cfg_name(c)[6:]}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"mq {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us raw; {rec['n']} ok, "
        f"{rec['fails']} failed, {rec['wrong']} wrong) {times}")


def prepare(weights):
    global F12MQ_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12mq_build, F12MQ_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v15) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12MQ_MOD, secs = fut.result()
            regs = " ".join(f"{f12m_name(i)[9:]}:{int(getattr(F12MQ_MOD, f12m_name(i).replace('f12m_run_', 'f12mq_run_') + '_info')()) // 1000000}r"
                            for i in F12MQ_INSTANCES)
            log(f"f12mq build ok in {secs:.1f}s; regs {regs}")
        except Exception as exc:
            log(f"f12mq build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12mq build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.02
    sweep = Sweep(state)
    if F12MQ_MOD is not None:
        for (layer, m) in E47_TARGETS:
            lab.section(f"mq {layer} m{m}", sec_mq, sweep, state, layer, m, min_left=6.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12mq":
            log(f"MQ CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
