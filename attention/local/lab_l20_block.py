

# ---------------------------------------------------------------------------------------------
# Lab L20 (check runs only), on v8 (submission_v8.py: L16-L18 routes):
#   1. W1 (cuda_w1.py: verify on wgmma FP8 / QGMMA, AF41) built here as its own module (ptxas -v in the
#      log; a build failure leaves v8's module and routes untouched), timed against the current path
#      on all six verify classes with its error vs the FP32 reference; a class moves to W1 only when
#      correct and >= 3% faster (the lab's _cuda_ok / _cuda_run accept ("w1", ctas per SM, stages));
#   2. probe: the best W1 route on Fv16 with an L2-resident cache;
#   3. the Triton PDL chain's launch settings re-swept on Sv32 / Sv16 if they stay on v5's path
#      (lab_l20_tune_block.py).
# Prints ROUTES = {...} and TUNE_UPDATE = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W1 = {}
_L20_W1 = {(True, True, 16): [(1, 4), (1, 6), (1, 3), (2, 4)], (True, True, 32): [(1, 4), (1, 6), (2, 4)],
           (False, True, 16): [(1, 4), (1, 3), (2, 4)], (False, True, 32): [(1, 4), (1, 3), (2, 4)],
           (True, True, 1): [(1, 4), (1, 6), (2, 4)], (False, True, 1): [(1, 3), (1, 4)]}   # (ctas per SM, stages)
_L20_ERR = 0.12


def _l20_build():
    from torch.utils.cpp_extension import load_inline
    import tempfile
    _W1["mod"] = load_inline(
        name="okc_l20_w1", cpp_sources=CUDA_W1_DECL, cuda_sources=CUDA_W1_SRC,
        functions=["attend_w1", "attend_w1_setup", "attend_w1_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas", "-v"],
        build_directory=tempfile.mkdtemp(prefix="okc_l20_"), verbose=True)
    err = _W1["mod"].attend_w1_setup()
    if err:
        raise RuntimeError(f"attend_w1_setup returned {err}")
    for s in (3, 4, 6):
        a = _W1["mod"].attend_w1_attrs(s)
        _log(f"w1 s{s} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")


def _w1_run(data, cps, stages):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W1["mod"].attend_w1(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l20 = _cuda_ok
_cuda_run_l20 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W1 routes run on the lab's module)
    if route[0] == "w1":
        return ("mod" in _W1 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l20(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w1":
        return _w1_run(data, route[1], route[2])
    return _cuda_run_l20(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2020)
    _log(f"L20: timed path until a class wins: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    try:
        _l20_build()
    except Exception as exc:
        _log(f"L20: W1 build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen, best_fv16 = {}, None
    for cls, cfgs in _L20_W1.items() if "mod" in _W1 else ():
        full, _, bclass = cls
        name = f"{'F' if full else 'S'}v{bclass}"
        if lab.left() < 120:
            _log("w1: leaving budget for the tune sweep")
            break
        try:
            data = _lab_inputs(bclass, 8, 0 if full else 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq = data["cache_seqlens"] if full else data["cache_seqlens"].clamp(max=2048 + 8)
            kv = float(seq.sum()) * 256
            cur = CUDA_ROUTE.get(cls) or (("fa3t",) + FA3T_ROUTE[cls] if cls in FA3T_ROUTE else "v5")
            _log(f"{name} current {cur}: {base:.1f}us {kv / base / 1e6:.2f}TB/s")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for cps, s in cfgs:
            route = ("w1", cps, s)
            try:
                err = float((_w1_run(data, cps, s).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w1_run(data, cps, s))
                ns = _nsplit_w(data["config"], data["page_table"].shape[1], cps, 64)
                _log(f"{name} w1 c{cps} s{s} n{ns}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                if err < _L20_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} w1 c{cps} s{s}: FAILED {_why(exc)}")
        if cls == (True, True, 16) and best is not None:
            best_fv16 = (best[1], data)
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            FA3T_ROUTE.pop(cls, None)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    if best_fv16 is not None and lab.left() > 100:
        route, data = best_fv16
        try:
            hot = dict(data)
            hot["page_table"] = data["page_table"] % 8192
            hot["cache_loc"] = torch.gather(hot["page_table"].long(), 1, data["positions"].view(16, 8)).reshape(-1)
            us_cold = lab.time_graph(lambda: _w1_run(data, route[1], route[2]))
            us_hot = lab.time_graph(lambda: _w1_run(hot, route[1], route[2]))
            _log(f"probe Fv16 L2-hot: {route} hot {us_hot:.1f}us cold {us_cold:.1f}us")
        except Exception as exc:
            _log(f"probe FAILED {_why(exc)}")
    _log(f"ROUTES = {chosen!r}")
    try:
        _l20_tune(lab, k_cache, v_cache, gen)
    except Exception as exc:
        _log(f"tune FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
