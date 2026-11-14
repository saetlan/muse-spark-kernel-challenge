"""PE57 (P9 two-phase cluster reduction on the router, on v15): PE52 measured router m16 two-phase P9 9.73 vs the HL Triton
kernel 9.97-10.05 us raw (+2.5%, under its 3% gate) and router m8 +0.9% against the like-for-like rebuild. Same lab code as PE52
(module okc_proj_p9l, kind p9l) with gates 2% on router m16 and 1.5% on router m8. Base = proj/submission_v15.py byte for
byte. Winners -> "PL CHOICE"."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
PL_SUM_OLD = ("        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];\n      }\n  cluster_sync<CS>();\n")
PL_SUM_NEW = ("        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];\n      }\n"
              "  if constexpr (CS > 1) {\n"
              "    // PE52: each CTA sums its own K parts in place first (local, fixed order); the cross-CTA pass then reads CS values\n"
              "    __syncthreads();\n"
              "    for (int idx = tid; idx < ROWS * MT; idx += blockDim.x) {\n"
              "      float own = 0.0f;\n"
              "#pragma unroll\n"
              "      for (int kp = 0; kp < KSW; ++kp) own += part[static_cast<size_t>(kp) * ROWS * MT + idx];\n"
              "      part[idx] = own;\n"
              "    }\n"
              "  }\n"
              "  cluster_sync<CS>();\n")
PL_LOOP_OLD = ("      for (int kp = 0; kp < KSW; ++kp) {\n        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;\n")
PL_LOOP_NEW = ("      for (int kp = 0; kp < (CS > 1 ? 1 : KSW); ++kp) {   // PE52: with CS > 1, part[0] holds each peer's KSW sum\n"
               "        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;\n")
P9L_MOD = None


def p9l_source(instances):
    src = p9_source(instances)
    for a in (PL_SUM_OLD, PL_LOOP_OLD, "namespace p9 {"):
        assert src.count(a) == 1, ("P9 text changed", a[:60])
    src = src.replace(PL_SUM_OLD, PL_SUM_NEW).replace(PL_LOOP_OLD, PL_LOOP_NEW)
    return (src.replace("namespace p9 {", "namespace p9l {").replace("p9::", "p9l::")
            .replace("void p9_run_", "void p9l_run_").replace("int64_t p9_run_", "int64_t p9l_run_"))


def p9l_decls(instances):
    return [d.replace("p9_run_", "p9l_run_") for d in p9_decls(instances)]


def p9l_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = p9l_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_p9l", cpp_sources="\n".join(decls), cuda_sources=p9l_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_p9l_"), verbose=False)
    return mod, time.monotonic() - t0


def run_p9l(c, x, layer, n, state):
    """run_p9 for the router (hi + lo, FP32 out) on the two-phase module."""
    if P9L_MOD is None or layer != "router" or "router" not in state.get("p9", {}):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], True, True)
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    smem = p9_smem_bytes(c["NT"], c["RTW"], c["RG"], c["KSW"], c["CS"], k)
    dmy = state["p9_dummy"]
    getattr(P9L_MOD, p9_name(inst).replace("p9_run_", "p9l_run_"))(state["p9"][layer], dmy, dmy, dmy, dmy, dmy, dmy,
                                                                    x.contiguous(), out, m, n, k, smem)
    return out


_run_choice_pre_pl = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "p9l":
        return run_p9l(c, x, layer, n, state)
    return _run_choice_pre_pl(c, x, layer, n, state)


_cfg_name_pre_pl = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "p9l":
        return "p9l" + _cfg_name_pre_pl(dict(c, kind="p9"))[2:]
    return _cfg_name_pre_pl(c)


PL_TARGETS = {("router", 8): dict(kind="p9l", NT=1, RTW=1, RG=2, KSW=8, D=2, CS=8),
              ("router", 16): dict(kind="p9l", NT=2, RTW=1, RG=1, KSW=8, D=2, CS=8)}


def sec_pl(lab, sweep, state, layer, m):
    base = CHOICE.get((layer, m))
    cand = dict(PL_TARGETS[(layer, m)], fallback=base)
    n, k = lab_n(layer, m), FULL[layer][1]
    sweep.shape(lab, layer, m, [cand])
    rec = sweep.results[(layer, m)]
    x = lab_input(m, k, 7900 + m)
    tb = min(lab.time_graph(lambda: run_choice(base, x, layer, n, state)) for _ in range(3))
    tp = min(lab.time_graph(lambda: run_choice(cand, x, layer, n, state)) for _ in range(3))
    old = dict(cand, kind="p9", fallback=None)
    to = min(lab.time_graph(lambda: run_choice(old, x, layer, n, state)) for _ in range(3)) if layer == "router" else float("nan")
    log(f"pl {layer} m{m}: base {cfg_name(base)} {rec['cur_us']:.2f} / {tb:.2f}, P9 one-phase {to:.2f}, two-phase "
        f"{rec['best'][0] if rec['best'] else float('nan'):.2f} / {tp:.2f} us raw ({tb / tp:.3f}x vs base); "
        f"{rec['n']} ok, {rec['wrong']} wrong")


def prepare(weights):
    global P9L_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(p9l_build, P9_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v15) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 220.0 - (time.monotonic() - t0)))
    if done:
        try:
            mod, secs = fut.result()
            for inst in P9_INSTANCES:
                getattr(mod, p9_name(inst).replace("p9_run_", "p9l_run_") + "_attr")(P9_MAX_SMEM)
            P9L_MOD = mod
            log(f"p9l build ok in {secs:.1f}s")
        except Exception as exc:
            log(f"p9l build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("p9l build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if P9L_MOD is not None:
        MIN_GAIN = 1.015
        lab.section("pl router m8", sec_pl, sweep, state, "router", 8, min_left=5.0)
        lab.section("commit m8", lambda lab_: sweep.commit(lab_), min_left=-1e9)
        sweep.results.clear()
        MIN_GAIN = 1.02
        lab.section("pl router m16", sec_pl, sweep, state, "router", 16, min_left=5.0)
        lab.section("commit m16", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "p9l":
            log(f"PL CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
