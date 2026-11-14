

# ---------------------------------------------------------------------------------------------
# Lab L41 (check runs only), on the board's file (the timed path byte for byte): W3t (cuda_w3t.py, make_w3t.py, B7): W3b
# with a TMA bulk-copy copier (one 128 B cp.async.bulk per K / V row, completion by expect-tx on the land barrier; the
# transposers re-lay K from row-major staging into the K-major tile), barrier watchdog. On Fv16, Fv32 vs the current
# path: attributes, error vs FP32, HANG check, time; routed when correct (< 0.12) and >= 3% faster
# (("w3t", stages, pp, nwg, rc)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3T = {}
_L41_CASES = [("Fv16", (True, True, 16), 0), ("Fv32", (True, True, 32), 0)]
_L41_ERR = 0.12


def _w3t_run(data, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    nparts = nsplit * (nwg // 2)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nparts, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nparts,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W3T["mod"].attend_w3t(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


_cuda_ok_l41 = _cuda_ok
_cuda_run_l41 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W3 routes run on the lab's module)
    if route[0] == "w3t":
        return ("mod" in _W3T and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l41(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3t":
        return _w3t_run(data, *route[1:])
    return _cuda_run_l41(data, route)


def _l41_w3t(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        _W3T["mod"] = load_inline(name="okc_l41_w3t", cpp_sources=CUDA_W3T_DECL, cuda_sources=CUDA_W3T_SRC,
                                 functions=["attend_w3t", "attend_w3t_setup", "attend_w3t_attrs", "attend_w3t_hang"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l41_w3t_"), verbose=False)
        refused = _W3T["mod"].attend_w3t_setup()
        ok_vars = []
        for i, (s, p, nw, rc, rp) in enumerate(W3T_VARIANTS):
            a = _W3T["mod"].attend_w3t_attrs(s, p, nw, rc)
            ok = not (refused >> i) & 1
            _log(f"w3t s{s} pp{p} nwg{nw} rc{rc}/{rp} attrs (regs, local B, static smem B, pool ok): {a}"
                 f"{'  <-- LOCAL MEMORY' if a[1] else ''}{'' if ok else '  REFUSED'}")
            if ok:
                ok_vars.append((s, p, nw, rc))
    except Exception as exc:
        _W3T.pop("mod", None)
        _log(f"w3t: build FAILED {_why(exc)}")
        return {}
    chosen = {}
    for name, cls, window in _L41_CASES:
        if lab.left() < 120:
            _log("w3: out of budget")
            break
        bclass = cls[2]
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for v in list(ok_vars):
            try:
                err = float((_w3t_run(data, *v).float() - ref).norm() / ref.norm())   # float(): the kernel has finished
                if _W3T["mod"].attend_w3t_hang():
                    res.append(f"{v} HANG (watchdog)")
                    ok_vars.remove(v)
                    continue
                us = lab.time_graph(lambda: _w3t_run(data, *v))
                if _W3T["mod"].attend_w3t_hang():
                    res.append(f"{v} HANG in timing (watchdog)")
                    ok_vars.remove(v)
                    continue
                res.append(f"{v} {us:.1f}us err {err:.1e}")
                if err < _L41_ERR and (best is None or us < best[0]):
                    best = (us, ("w3t",) + v)
            except Exception as exc:
                res.append(f"{v} FAILED {_why(exc)[:120]}")
        _log(f"{name} current {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    return chosen


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4141)
    _log(f"L41: timed path: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = _l41_w3t(lab, k_cache, v_cache, gen)
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
