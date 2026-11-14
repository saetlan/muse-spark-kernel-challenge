


# ---------------------------------------------------------------------------------------------
# Lab L51 (check runs only), on v14 (the timed path = v14 byte for byte): UW2g (make_uw2g.py: UW2 with UV4's second-level
# merge: ncl clusters per request, one launch per full verify layer; v14's Fv1 runs prologue + W2 + combine) on Fv1, Fv16
# vs v14's path; error checked before and after the timing replays, tickets read back at zero. Routed when correct
# (< 0.12) and >= 3% faster (("uw2g", stages, pp, cs, ncl)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UG = {}
_L51_ERR = 0.12
_L51_CASES = [("Fv1", (True, True, 1), 0, [(6, 0, 16, 7), (4, 1, 16, 7), (6, 0, 8, 16), (6, 1, 8, 16), (4, 1, 8, 16), (4, 1, 4, 32), (6, 1, 16, 8)]),
              ("Fv16", (True, True, 16), 0, [(6, 0, 8, 1), (4, 1, 8, 1)]),
              ("Sv1", (False, True, 1), 2048, [(6, 0, 16, 1)])]


def _ug_bufs(batch, hkv, cs, ncl, dev):
    key = (batch, hkv, cs, ncl)
    if key not in _UG:
        bp = batch * (hkv // 2)
        _UG[key] = (torch.zeros(max(1, bp * ncl * 128 * 68), dtype=torch.float32, device=dev),
                    torch.zeros(max(1, bp * cs), dtype=torch.int32, device=dev))
    return _UG[key]


def _ug_run(data, stages, pp, cs, ncl):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    dev = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    gpart, tickets = _ug_bufs(batch, hkv, cs, ncl, dev)
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=dev)
    _UG["mod"].uw2g(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, dev),
                    data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                    data["cache_seqlens"], out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"],
                    int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs, ncl, gpart, tickets)
    return out


_cuda_ok_l51 = _cuda_ok
_cuda_run_l51 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UW2g routes run on the lab's module)
    if route[0] == "uw2g":
        return ("mod" in _UG and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l51(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "uw2g":
        return _ug_run(data, *route[1:])
    return _cuda_run_l51(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5151)
    _log(f"L51: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UG["mod"] = load_inline(name="okc_l51_uw2g", cpp_sources=CUDA_UW2G_DECL, cuda_sources=CUDA_UW2G_SRC,
                                 functions=["uw2g", "uw2g_setup", "uw2g_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l51_"), verbose=False)
        r = _UG["mod"].uw2g_setup()
        _log(f"uw2g setup {r}; attrs s6 p0 cs16 {_UG['mod'].uw2g_attrs(6, 0, 16)}, s6 p1 cs8 {_UG['mod'].uw2g_attrs(6, 1, 8)}")
        if r:
            raise RuntimeError(f"uw2g_setup refused {r}")
    except Exception as exc:
        _UG.pop("mod", None)
        _log(f"uw2g: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window, variants in _L51_CASES:
        if lab.left() < 40:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for s, p, cs, ncl in variants:
            route = ("uw2g", s, p, cs, ncl)
            try:
                err = float((_ug_run(data, s, p, cs, ncl).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ug_run(data, s, p, cs, ncl))
                err2 = float((_ug_run(data, s, p, cs, ncl).float() - ref).norm() / ref.norm())
                tk = int(_ug_bufs(cls[2], 2, cs, ncl, dev)[1].abs().sum())
                res.append(f"{route[1:]} {us:.1f}us err {err:.1e}/{err2:.1e} tickets {tk}")
                if max(err, err2) < _L51_ERR and tk == 0 and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us err {berr:.1e}; " + "; ".join(res))
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
