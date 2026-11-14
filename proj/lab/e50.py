"""PE50 (F12M epilogue: escape reads issued before the KSW sum): PE44 timed F12M's escape step at 0.2-0.4 us on every shape;
the helper warp (PE46) removes it but only fits where 288 threads x registers leave 2 CTAs per SM, so pooler m8 / m16, qkv m8
(D4) and unpooler m16 (D8) still pay it. This variant loads each item's record and its escape x values first, then sums the
KSW parts (so the x reads' latency overlaps the shared-memory sum), then applies scale and escapes in the same order as F12M:
bit-identical outputs. Base = proj/submission_v13.py byte for byte; the lab builds the base's F12M instances with that one
edit (module okc_proj_f12meo, namespace f12meo) and races kind "f12meo" (same configs) against v13's F12M on its F12M shapes;
bit-identity checked per shape; gate 1%. Winners -> "EO CHOICE"."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
EO_OLD = (
    "    const int tok = idx / RB, row = idx % RB;\n    float sum = 0.0f;\n    if constexpr (CS > 1) {\n"
    "      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&part[0][row][tok]));\n#pragma unroll\n"
    "      for (int p = 0; p < CS; ++p) sum += ld_peer_f32(a, static_cast<uint32_t>(p));\n    } else {\n#pragma unroll\n"
    "      for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];\n    }\n    if constexpr (F12) {\n"
    "      const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];\n"
    "      const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;\n      sum *= f12m_scale(base);\n"
    "      const float wdec = f12m_wdec(base);\n      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};\n#pragma unroll\n"
    "      for (int j = 0; j < 7; ++j) {\n        if (j < count) {\n"
    "          const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);\n"
    "          sum = fmaf(-xk, wdec, sum);\n          sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);\n        }\n      }\n    }\n")
EO_NEW = (
    "    const int tok = idx / RB, row = idx % RB;\n"
    "    float xe[7] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};\n"
    "    uint32_t base = 0u, count = 0u, esc[7] = {0u, 0u, 0u, 0u, 0u, 0u, 0u};\n"
    "    if constexpr (F12) {   // the escape x reads go out before the KSW sum (PE50)\n"
    "      const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];\n"
    "      base = q0.x & 0xFFu;\n      count = (q0.x >> 8) & 0xFFu;\n"
    "      esc[0] = q0.y; esc[1] = q0.z; esc[2] = q0.w; esc[3] = q1.x; esc[4] = q1.y; esc[5] = q1.z; esc[6] = q1.w;\n#pragma unroll\n"
    "      for (int j = 0; j < 7; ++j)\n"
    "        if (j < count) xe[j] = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);\n"
    "    }\n    float sum = 0.0f;\n    if constexpr (CS > 1) {\n"
    "      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&part[0][row][tok]));\n#pragma unroll\n"
    "      for (int p = 0; p < CS; ++p) sum += ld_peer_f32(a, static_cast<uint32_t>(p));\n    } else {\n#pragma unroll\n"
    "      for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];\n    }\n    if constexpr (F12) {\n"
    "      sum *= f12m_scale(base);\n      const float wdec = f12m_wdec(base);\n#pragma unroll\n"
    "      for (int j = 0; j < 7; ++j) {\n        if (j < count) {\n"
    "          sum = fmaf(-xe[j], wdec, sum);\n          sum = fmaf(xe[j], __uint_as_float(esc[j] & 0xFFFF0000u), sum);\n        }\n      }\n    }\n")
F12MEO_MOD = None
F12MEO_BUILD = {}


def f12meo_source(instances):
    src = f12m_source(instances)
    assert src.count(EO_OLD) == 1 and src.count("namespace f12m {") == 1, "F12M epilogue text changed"
    src = src.replace(EO_OLD, EO_NEW)
    return (src.replace("namespace f12m {", "namespace f12meo {").replace("f12m::", "f12meo::")
            .replace("void f12m_run_", "void f12meo_run_").replace("int64_t f12m_run_", "int64_t f12meo_run_"))


def f12meo_decls(instances):
    return [d.replace("f12m_run_", "f12meo_run_") for d in f12m_decls(instances)]


def f12meo_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12meo_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12meo", cpp_sources="\n".join(decls), cuda_sources=f12meo_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12meo_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12meo(c, x, layer, n, state):
    """F12M's run_f12m with the reordered-epilogue module (same planes, same instance tuple)."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    if F12MEO_MOD is None or pl is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12MEO_MOD, f12m_name(inst).replace("f12m_run_", "f12meo_run_"))(pl["p"], pl["rec"], x, out, m, n)
    return out


_run_choice_pre_eo = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12meo":
        return run_f12meo(c, x, layer, n, state)
    return _run_choice_pre_eo(c, x, layer, n, state)


_cfg_name_pre_eo = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12meo":
        return "f12meo" + _cfg_name_pre_eo(dict(c, kind="f12m"))[4:]
    return _cfg_name_pre_eo(c)


def sec_eo(lab, sweep, state, layer, m):
    base = CHOICE[(layer, m)]
    cand = dict(base, kind="f12meo", fallback=base)
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 7400 + m)
    same = torch.equal(run_choice(cand, x, layer, n, state), run_choice(base, x, layer, n, state))
    sweep.shape(lab, layer, m, [cand])
    rec = sweep.results[(layer, m)]
    # second opinion: alternate base / padded, min of 3 each
    tb = min(lab.time_graph(lambda: run_choice(base, x, layer, n, state)) for _ in range(3))
    tp = min(lab.time_graph(lambda: run_choice(cand, x, layer, n, state)) for _ in range(3))
    log(f"eo {layer} m{m} {cfg_name(base)}: base {rec['cur_us']:.2f} / {tb:.2f}, reordered "
        f"{rec['best'][0] if rec['best'] else float('nan'):.2f} / {tp:.2f} us raw ({tb / tp:.3f}x), outputs bit-identical: {same}")
    if not same:
        rec["best"] = None


def prepare(weights):
    global F12MEO_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12meo_build, F12M_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v13) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 200.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12MEO_MOD, secs = fut.result()
            regs = " ".join(f"{f12m_name(i)[9:]}:{int(getattr(F12MEO_MOD, f12m_name(i).replace('f12m_run_', 'f12meo_run_') + '_info')()) // 1000000}r"
                            for i in F12M_INSTANCES)
            log(f"f12meo build ok in {secs:.1f}s; regs {regs}")
        except Exception as exc:
            log(f"f12meo build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12meo build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.01
    sweep = Sweep(state)
    if F12MEO_MOD is not None:
        for (layer, m), c in sorted(CHOICE.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            if c.get("kind") == "f12m":
                lab.section(f"eo {layer} m{m}", sec_eo, sweep, state, layer, m, min_left=5.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12meo":
            log(f"EO CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
