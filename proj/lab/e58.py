"""PE58 (two-phase P9 at router m32): v16 runs router m8 / m16 on P9 with the two-phase cluster reduction (PE57: m16 beat the
HL Triton kernel by 2.4%); router m32 (HL swap kernel, 6.1 us judged) was never tried with it. Base = proj/submission_v16.py byte
for byte; the lab builds two-phase P9 instances at 32 tokens (NT 4: RG 1 / 2, KSW 8, D 2, CS 8) as okc_proj_p9l2 (namespace
p9l2) and races kind "p9l2" against v16's router m32 kernel; FP64 checks; gate 2%. Winner -> "PL2 CHOICE"."""

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
P9L2_MOD = None


def p9l2_source(instances):
    src = p9_source(instances)
    for a in (PL_SUM_OLD, PL_LOOP_OLD, "namespace p9 {"):
        assert src.count(a) == 1, ("P9 text changed", a[:60])
    src = src.replace(PL_SUM_OLD, PL_SUM_NEW).replace(PL_LOOP_OLD, PL_LOOP_NEW)
    return (src.replace("namespace p9 {", "namespace p9l2 {").replace("p9::", "p9l2::")
            .replace("void p9_run_", "void p9l2_run_").replace("int64_t p9_run_", "int64_t p9l2_run_"))


def p9l2_decls(instances):
    return [d.replace("p9_run_", "p9l2_run_") for d in p9_decls(instances)]


def p9l2_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = p9l2_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_p9l2", cpp_sources="\n".join(decls), cuda_sources=p9l2_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_p9l2_"), verbose=False)
    return mod, time.monotonic() - t0


def run_p9l2(c, x, layer, n, state):
    """run_p9 for the router (hi + lo, FP32 out) on the two-phase module."""
    if P9L2_MOD is None or layer != "router" or "router" not in state.get("p9", {}):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], True, True)
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32, device=x.device)
    smem = p9_smem_bytes(c["NT"], c["RTW"], c["RG"], c["KSW"], c["CS"], k)
    dmy = state["p9_dummy"]
    getattr(P9L2_MOD, p9_name(inst).replace("p9_run_", "p9l2_run_"))(state["p9"][layer], dmy, dmy, dmy, dmy, dmy, dmy,
                                                                    x.contiguous(), out, m, n, k, smem)
    return out


_run_choice_pre_pl2 = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "p9l2":
        return run_p9l2(c, x, layer, n, state)
    return _run_choice_pre_pl2(c, x, layer, n, state)


_cfg_name_pre_pl2 = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "p9l2":
        return "p9l2" + _cfg_name_pre_pl2(dict(c, kind="p9"))[2:]
    return _cfg_name_pre_pl2(c)


PL2_TARGETS = {("router", 32): [dict(kind="p9l2", NT=4, RTW=1, RG=1, KSW=8, D=2, CS=8),
                                 dict(kind="p9l2", NT=4, RTW=1, RG=2, KSW=8, D=2, CS=8),
                                 dict(kind="p9l2", NT=4, RTW=1, RG=1, KSW=8, D=1, CS=8)]}
PL2_INSTANCES = tuple(dict.fromkeys((c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], True, True)
                                    for cs in PL2_TARGETS.values() for c in cs))


def sec_pl2(lab, sweep, state, layer, m):
    cands = [dict(c, fallback=CHOICE.get((layer, m))) for c in PL2_TARGETS[(layer, m)]]
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    times = " ".join(f"{cfg_name(c)[5:]}:{us:.2f}" for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"pl2 {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us raw; {rec['n']} ok, "
        f"{rec['wrong']} wrong) {times}")


def prepare(weights):
    global P9L2_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(p9l2_build, PL2_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v16) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 240.0 - (time.monotonic() - t0)))
    if done:
        try:
            mod, secs = fut.result()
            for inst in PL2_INSTANCES:
                getattr(mod, p9_name(inst).replace("p9_run_", "p9l2_run_") + "_attr")(P9_MAX_SMEM)
            P9L2_MOD = mod
            log(f"p9l2 build ok in {secs:.1f}s; regs " + " ".join(
                f"{p9_name(i)[7:]}:{int(getattr(mod, p9_name(i).replace('p9_run_', 'p9l2_run_') + '_info')()) // 1000000}r" for i in PL2_INSTANCES))
        except Exception as exc:
            log(f"p9l2 build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("p9l2 build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    if P9L2_MOD is not None:
        MIN_GAIN = 1.02
        lab.section("pl2 router m32", sec_pl2, sweep, state, "router", 32, min_left=5.0)
        lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "p9l2":
            log(f"PL2 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
