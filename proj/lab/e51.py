"""PE51 (F12V x staging without bank conflicts): F12V (m = 1: qkv, pooler) stages x in shared memory for its escapes with
xs[kb * 128 + lane * 4 + g]: each 16-B store has a 64-B lane stride, 4-way bank-conflicted, on every main-loop step of the
row-group-0 warps. This variant stores xs[kb * 128 + g * 32 + lane] (lanes contiguous: conflict-free) and looks the escapes up
through f12v_xpos(k) (exact remap, checked for all 8192 k); outputs bit-identical. Base = proj/submission_v14.py byte for byte;
the lab builds the base's F12V instances with those edits (module okc_proj_f12vx, namespace f12vx) and races kind "f12vx"
(same configs) against v14's F12V on qkv m1 and pooler m1; bit-identity checked; gate 1%. Winners -> "VX CHOICE"."""

import concurrent.futures
import tempfile
import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 150.0
VX_STORE_OLD = "      for (int g = 0; g < 4; ++g) xs[f12v_xidx(f12v_kb(ks, s, NS), lane, g)] = xv[slot][g];"
VX_STORE_NEW = "      for (int g = 0; g < 4; ++g) xs[f12v_kb(ks, s, NS) * 128 + g * 32 + lane] = xv[slot][g];   // lanes contiguous (PE51)"
VX_READ_OLD = "        const float xk = __uint_as_float(static_cast<uint32_t>(xh[esc[j] & 0xFFFFu]) << 16);"
VX_READ_NEW = "        const float xk = __uint_as_float(static_cast<uint32_t>(xh[f12v_xpos(static_cast<int>(esc[j] & 0xFFFFu))]) << 16);"
VX_HELPER_ANCHOR = "__device__ __forceinline__ long long f12v_ridx(long long row, int h) { return row * 2 + h; }"
VX_HELPER = ("\n// PE51: element k of x in the conflict-free staging (uint4 i = k / 8 of block kb sits at kb * 128 + (i & 3) * 32 + (i & 127) / 4).\n"
             "__device__ __forceinline__ int f12v_xpos(int k) { return ((((k >> 3) & ~127) | (((k >> 3) & 3) << 5) | (((k >> 3) & 127) >> 2)) << 3) | (k & 7); }")
F12VX_MOD = None


def f12vx_source(instances):
    src = f12v_source(instances)
    for a in (VX_STORE_OLD, VX_READ_OLD, VX_HELPER_ANCHOR, "namespace f12v {"):
        assert src.count(a) == 1, ("F12V text changed", a[:60])
    src = src.replace(VX_STORE_OLD, VX_STORE_NEW).replace(VX_READ_OLD, VX_READ_NEW).replace(VX_HELPER_ANCHOR, VX_HELPER_ANCHOR + VX_HELPER)
    return (src.replace("namespace f12v {", "namespace f12vx {").replace("f12v::", "f12vx::")
            .replace("void f12v_run_", "void f12vx_run_").replace("int64_t f12v_run_", "int64_t f12vx_run_"))


def f12vx_decls(instances):
    return [d.replace("f12v_run_", "f12vx_run_") for d in f12v_decls(instances)]


def f12vx_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12vx_decls(instances)
    t0 = time.monotonic()
    mod = load_inline(name="okc_proj_f12vx", cpp_sources="\n".join(decls), cuda_sources=f12vx_source(instances),
                      functions=[d.split("(")[0].split()[-1] for d in decls],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                      build_directory=tempfile.mkdtemp(prefix="okc_proj_f12vx_"), verbose=False)
    return mod, time.monotonic() - t0


def run_f12vx(c, x, layer, n, state):
    """run_f12v on the conflict-free-staging module (same planes, same instance tuple)."""
    pl = state.get("f12v", {}).get(layer)
    if F12VX_MOD is None or pl is None or x.shape[0] != 1:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    k = x.shape[1]
    inst = (c["R"], c["RG"], c["KSW"], k // 1024 // c["KSW"], c["D"])
    x = x.contiguous()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12VX_MOD, f12v_name(inst).replace("f12v_run_", "f12vx_run_"))(pl["p"], pl["rec"], x, out, n)
    return out


_run_choice_pre_vx = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "f12vx":
        return run_f12vx(c, x, layer, n, state)
    return _run_choice_pre_vx(c, x, layer, n, state)


_cfg_name_pre_vx = cfg_name


def cfg_name(c):
    if c is not None and c.get("kind") == "f12vx":
        return "f12vx" + _cfg_name_pre_vx(dict(c, kind="f12v"))[4:]
    return _cfg_name_pre_vx(c)


def sec_vx(lab, sweep, state, layer, m):
    base = CHOICE[(layer, m)]
    cand = dict(base, kind="f12vx", fallback=base)
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 7500 + m)
    xp = lab_input(m, k, 7550 + m, probe_all=True)
    same = all(torch.equal(run_choice(cand, xx, layer, n, state), run_choice(base, xx, layer, n, state)) for xx in (x, xp))
    sweep.shape(lab, layer, m, [cand])
    rec = sweep.results[(layer, m)]
    tb = min(lab.time_graph(lambda: run_choice(base, x, layer, n, state)) for _ in range(3))
    tp = min(lab.time_graph(lambda: run_choice(cand, x, layer, n, state)) for _ in range(3))
    log(f"vx {layer} m{m} {cfg_name(base)}: base {rec['cur_us']:.2f} / {tb:.2f}, conflict-free "
        f"{rec['best'][0] if rec['best'] else float('nan'):.2f} / {tp:.2f} us raw ({tb / tp:.3f}x), outputs bit-identical: {same}")
    if not same:
        rec["best"] = None


def prepare(weights):
    global F12VX_MOD, MIN_GAIN
    t0 = time.monotonic()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(f12vx_build, F12V_INSTANCES)
    pool.shutdown(wait=False)
    state = _submission_prepare(weights)
    log(f"base prepare (v14) took {time.monotonic() - t0:.1f}s")
    done, _ = concurrent.futures.wait([fut], timeout=max(1.0, 220.0 - (time.monotonic() - t0)))
    if done:
        try:
            F12VX_MOD, secs = fut.result()
            log(f"f12vx build ok in {secs:.1f}s")
        except Exception as exc:
            log(f"f12vx build FAILED {type(exc).__name__}: {' '.join(str(exc).split())[-400:]}")
    else:
        log("f12vx build not finished: skipped")
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    MIN_GAIN = 1.01
    sweep = Sweep(state)
    if F12VX_MOD is not None:
        for (layer, m), c in sorted(CHOICE.items()):
            if c.get("kind") == "f12v":
                lab.section(f"vx {layer} m{m}", sec_vx, sweep, state, layer, m, min_left=5.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "f12vx":
            log(f"VX CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
