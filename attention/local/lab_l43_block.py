


# ---------------------------------------------------------------------------------------------
# Lab L43 (check runs only), on v14 (the timed path = v14 byte for byte): D3t (cuda_dec4t.py, make_dec4t.py, B7 / AF66: D3
# with one 128 B TMA bulk copy per K / V row and per-(warp, stage) data barriers; page entries as D3; watchdog) on Fd1,
# Fd16, Fd32 with D3's split rules (('dec4t', cps, warps, stages, 'w') whole waves, or 4 fields: D3's _nsplit) vs v14's
# D3 routes: error vs FP32, HANG check, time; routed when correct (< 0.02) and >= 3% faster.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_D3T = {}
_L43_ERR = 0.02
_L43_CASES = [("Fd1", (True, False, 1), [("dec4t", 2, 4, 3, "w"), ("dec4t", 2, 4, 4, "w"), ("dec4t", 3, 4, 3, "w")]),
              ("Fd16", (True, False, 16), [("dec4t", 3, 4, 3, "w"), ("dec4t", 3, 4, 4, "w"), ("dec4t", 3, 8, 3, "w")]),
              ("Fd32", (True, False, 32), [("dec4t", 6, 4, 3), ("dec4t", 6, 4, 4), ("dec4t", 3, 4, 4, "w")])]


def _d3t_run(data, route):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    cps, warps, stages = route[1], route[2], route[3]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 32 * warps) if len(route) == 5 else _nsplit(cfg, max_pages, cps, 32 * warps)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _D3T["mod"].attend_dec4t(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                             data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                             cfg["page_size"], cfg["window"], cfg["sm_scale"], warps, stages)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l43 = _cuda_ok
_cuda_run_l43 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: D3t routes run on the lab's module)
    if route[0] == "dec4t":
        return ("mod" in _D3T and cfg["q_len"] == 1 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l43(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "dec4t":
        return _d3t_run(data, route)
    return _cuda_run_l43(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4343)
    _log(f"L43: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _D3T["mod"] = load_inline(name="okc_l43_d3t", cpp_sources=CUDA_DEC4T_DECL, cuda_sources=CUDA_DEC4T_SRC,
                                  functions=["attend_dec4t", "attend_dec4t_setup", "attend_dec4t_attrs", "attend_dec4t_hang"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l43_"), verbose=False)
        if _D3T["mod"].attend_dec4t_setup():
            raise RuntimeError("attend_dec4t_setup refused")
        for w, s in ((4, 3), (4, 4), (8, 3)):
            a = _D3T["mod"].attend_dec4t_attrs(w, s)
            _log(f"dec4t w{w} s{s} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _D3T.pop("mod", None)
        _log(f"d3t: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, routes in _L43_CASES:
        if "mod" not in _D3T or lab.left() < 40:
            break
        try:
            data = _lab_inputs(cls[2], 1, 0, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for route in routes:
            try:
                err = float((_d3t_run(data, route).float() - ref).norm() / ref.norm())   # float(): the kernel finished
                if _D3T["mod"].attend_dec4t_hang():
                    res.append(f"{route} HANG (watchdog)")
                    _D3T.pop("mod", None)
                    break
                us = lab.time_graph(lambda: _d3t_run(data, route))
                if _D3T["mod"].attend_dec4t_hang():
                    res.append(f"{route} HANG in timing (watchdog)")
                    _D3T.pop("mod", None)
                    break
                res.append(f"{route} {us:.1f}us err {err:.1e}")
                if err < _L43_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:160]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls)} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
