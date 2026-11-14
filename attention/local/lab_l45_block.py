


# ---------------------------------------------------------------------------------------------
# Lab L45 (check runs only), on v14 (the timed path = v14 byte for byte): W4 (cuda_w4.py, make_w4.py, AF68): verify with
# consumer pairs that copy and transpose their own stages (no producer warps; NPAIR pairs per CTA, each its own split and
# ring), W3's consumer math per pair. On Fv16, Fv32, Sv32, Fv1 vs v14's path: attrs, error vs FP32, time; routed when
# correct (< 0.12) and >= 3% faster (("w4", stages, npair, minb)). Splits: whole waves at 2 pairs per SM.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W4 = {}
_L45_ERR = 0.12
_L45_CASES = [("Fv16", (True, True, 16), 0), ("Fv32", (True, True, 32), 0), ("Sv32", (False, True, 32), 2048),
              ("Fv1", (True, True, 1), 0)]


def _w4_run(data, stages, npair, minb):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 2, 64)
    nsplit = max(npair, nsplit // npair * npair)
    rows = batch * length * hq * nsplit
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((rows,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W4["mod"].attend_w4(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                         o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                         cfg["sm_scale"], stages, npair, minb)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l45 = _cuda_ok
_cuda_run_l45 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W4 routes run on the lab's module)
    if route[0] == "w4":
        return ("mod" in _W4 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l45(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w4":
        return _w4_run(data, *route[1:])
    return _cuda_run_l45(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4545)
    _log(f"L45: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _W4["mod"] = load_inline(name="okc_l45_w4", cpp_sources=CUDA_W4_DECL, cuda_sources=CUDA_W4_SRC,
                                 functions=["attend_w4", "attend_w4_setup", "attend_w4_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l45_"), verbose=False)
        if _W4["mod"].attend_w4_setup():
            raise RuntimeError("attend_w4_setup refused")
        for s, p, m in W4_VARIANTS:
            a = _W4["mod"].attend_w4_attrs(s, p, m)
            _log(f"w4 s{s} npair{p} minb{m} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _W4.pop("mod", None)
        _log(f"w4: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window in _L45_CASES:
        if lab.left() < 40:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for v in W4_VARIANTS:
            route = ("w4",) + tuple(v)
            try:
                err = float((_w4_run(data, *v).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w4_run(data, *v))
                res.append(f"{route} {us:.1f}us err {err:.1e}")
                if err < _L45_ERR and (best is None or us < best[0]):
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
