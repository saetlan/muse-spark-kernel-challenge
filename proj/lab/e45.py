"""PE45 (F12M bank conflicts): PE44 found F12M's epilogue costs 0.4-1.1 us at m16 / o m8, mostly the KSW sum. Its
part[KSW][RB][MTC] layout makes the sum's reads 8-way (m16) / 16-way (m32) bank-conflicted and the fragment writes 4-8-way;
a row stride of MTC + 3 makes both <= 2-way for every config v11b runs (checked offline). v11b byte for byte as the base;
the lab builds v11b's F12M instances from its own source with only that declaration changed (module okc_proj_f12mp,
namespace f12mp) and races kind "f12mp" (same configs) against v11b's F12M on its six shapes. Same math, same order, same
outputs bit for bit (checked here per shape), so the gate is 1% instead of 3%; winners are printed as "MP CHOICE" lines and
the check's own score measures v11b + them."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
MP_DECL = "__shared__ float part[KSW][RB][MT];"
MP_PAD = 3
F12MP_MOD = None
F12MP_BUILD = {}


def f12mp_source(instances):
    src = f12m_source(instances)
    assert src.count(MP_DECL) == 1 and src.count("namespace f12m {") == 1
    src = src.replace(MP_DECL, f"__shared__ float part[KSW][RB][MT + {MP_PAD}];   // row stride MT + {MP_PAD}: <= 2-way banks")
    return (src.replace("namespace f12m {", "namespace f12mp {").replace("f12m::", "f12mp::")
            .replace("void f12m_run_", "void f12mp_run_").replace("int64_t f12m_run_", "int64_t f12mp_run_"))


def f12mp_decls(instances):
    return [d.replace("f12m_run_", "f12mp_run_") for d in f12m_decls(instances)]


def f12mp_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12mp_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12mp", cpp_sources="\n".join(decls), cuda_sources=f12mp_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12mp_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12mp(c, x, layer, n, state):
    """F12M's run_f12m with the padded module (same planes, same instance tuple)."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MP_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MP_MOD, f12m_name(inst).replace("f12m_run_", "f12mp_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_mp = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12mp":
        return run_f12mp(c, x, layer, n, state)
    return _run_choice_pre_mp(c, x, layer, n, state)


_cfg_name_pre_mp = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12mp":
        return "f12mp" + _cfg_name_pre_mp(dict(c, kind="f12m"))[4:]
    return _cfg_name_pre_mp(c)


def sec_mp(lab, sweep, state, layer, m):
    base = CHOICE[(layer, m)]
    cand = dict(base, kind="f12mp", fallback=base)
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 7100 + m)
    same = torch.equal(run_choice(cand, x, layer, n, state), run_choice(base, x, layer, n, state))
    sweep.shape(lab, layer, m, [cand])
    rec = sweep.results[(layer, m)]
    # second opinion: alternate base / padded, min of 3 each
    tb = min(lab.time_graph(lambda: run_choice(base, x, layer, n, state)) for _ in range(3))
    tp = min(lab.time_graph(lambda: run_choice(cand, x, layer, n, state)) for _ in range(3))
    log(f"mp {layer} m{m} {cfg_name(base)}: base {rec['cur_us']:.2f} / {tb:.2f}, padded "
        f"{rec['best'][0] if rec['best'] else float('nan'):.2f} / {tp:.2f} us raw ({tb / tp:.3f}x), outputs bit-identical: {same}")
    if not same:
        rec["best"] = None


def prepare(weights):
    global F12MP_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12mp_build, F12M_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v11b) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12MP_MOD, secs = fut.result()
            regs = " ".join(f"{f12m_name(i)[9:]}:{int(getattr(F12MP_MOD, f12m_name(i).replace('f12m_run_', 'f12mp_run_') + '_info')()) // 1000000}r"
                            for i in F12M_INSTANCES)
            log(f"f12mp build ok in {secs:.1f}s; regs {regs}")
        except Exception as exc:
            log(f"f12mp build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12mp build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.01
    sweep = Sweep(state)
    if F12MP_MOD is not None:
        for (layer, m), c in sorted(CHOICE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if c.get("kind") == "f12m":
                lab.section(f"mp {layer} m{m}", sec_mp, sweep, state, layer, m, min_left=5.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12mp":
            log(f"MP CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
