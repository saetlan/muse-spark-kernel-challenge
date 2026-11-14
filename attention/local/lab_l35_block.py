

# ---------------------------------------------------------------------------------------------
# Lab L35 (check runs only), on v14 (the timed path = v14 byte for byte): W3d (cuda_w3d.py, make_w3d.py, AF60): W3b with
# all four producer warps copying (16 keys each, whole rows, own page-entry slice) and re-laying their own V chunk, so
# copies issue from all four SM sub-partitions (W3b: two copier warps). On Fv16, Fv32, Sv32, Fv1 vs v14's path:
# attributes, error vs FP32, time; routed when correct (< 0.12) and >= 3% faster (("w3d", stages, pp, nwg, rc)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3D = {}
_L35_CASES = [("Fv16", (True, True, 16), 0), ("Fv32", (True, True, 32), 0), ("Sv32", (False, True, 32), 2048),
              ("Fv1", (True, True, 1), 0)]
_L35_ERR = 0.12


def _w3d_run(data, stages, pp, nwg, rc):
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
    _W3D["mod"].attend_w3d(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


_cuda_ok_l35 = _cuda_ok
_cuda_run_l35 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W3 routes run on the lab's module)
    if route[0] == "w3d":
        return ("mod" in _W3D and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l35(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3d":
        return _w3d_run(data, *route[1:])
    return _cuda_run_l35(data, route)


def _l35_w3d(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        _W3D["mod"] = load_inline(name="okc_l35_w3d", cpp_sources=CUDA_W3D_DECL, cuda_sources=CUDA_W3D_SRC,
                                 functions=["attend_w3d", "attend_w3d_setup", "attend_w3d_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l35_w3d_"), verbose=False)
        refused = _W3D["mod"].attend_w3d_setup()
        ok_vars = []
        for i, (s, p, nw, rc, rp) in enumerate(W3D_VARIANTS):
            a = _W3D["mod"].attend_w3d_attrs(s, p, nw, rc)
            ok = not (refused >> i) & 1
            _log(f"w3d s{s} pp{p} nwg{nw} rc{rc}/{rp} attrs (regs, local B, static smem B, pool ok): {a}"
                 f"{'  <-- LOCAL MEMORY' if a[1] else ''}{'' if ok else '  REFUSED'}")
            if ok:
                ok_vars.append((s, p, nw, rc))
    except Exception as exc:
        _W3D.pop("mod", None)
        _log(f"w3d: build FAILED {_why(exc)}")
        return {}
    chosen = {}
    for name, cls, window in _L35_CASES:
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
        for v in ok_vars:
            try:
                err = float((_w3d_run(data, *v).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w3d_run(data, *v))
                res.append(f"{v} {us:.1f}us err {err:.1e}")
                if err < _L35_ERR and (best is None or us < best[0]):
                    best = (us, ("w3d",) + v)
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
    gen = torch.Generator(device=dev).manual_seed(3535)
    _log(f"L35: timed path v14: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = _l35_w3d(lab, k_cache, v_cache, gen)
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
