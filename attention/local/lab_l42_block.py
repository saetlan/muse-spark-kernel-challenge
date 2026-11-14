

# ---------------------------------------------------------------------------------------------
# Lab L42 (check runs only), on v14 (the timed path = v14 byte for byte): UW2t (cuda_uw2t.py, make_uw2t.py: UW2 with W3t's
# TMA bulk-copy copier, B7 / AF66; the fresh-row patch into K staging, K re-laid by the transposers; watchdog) on Sv1,
# Sv16, Sv32 in (stages, pp, cs) variants vs v14's path: attrs, error vs FP32, HANG check, time; routed when correct and
# >= 3% faster (("uw2t", stages, pp, cs)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UW2T = {}
_L42_CASES = [((False, True, 1), [(4, 0, 16), (4, 1, 16)]),
              ((False, True, 16), [(4, 1, 4), (4, 1, 8)]),
              ((False, True, 32), [(4, 1, 4), (4, 1, 8)])]
_L42_ERR = 0.12


def _uw2t_run(data, stages, pp, cs):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
    _UW2T["mod"].uw2t(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                    data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                    data["cache_seqlens"], out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"],
                    int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
    return out


_cuda_ok_l42 = _cuda_ok
_cuda_run_l42 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UW2t routes run on the lab's module)
    if route[0] == "uw2t":
        return ("mod" in _UW2T and cfg["q_len"] == 8 and cfg["window"] > 0 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l42(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "uw2t":
        return _uw2t_run(data, *route[1:])
    return _cuda_run_l42(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4242)
    _log(f"L42: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UW2T["mod"] = load_inline(name="okc_l42_uw2t", cpp_sources=CUDA_UW2T_DECL, cuda_sources=CUDA_UW2T_SRC,
                                  functions=["uw2t", "uw2t_setup", "uw2t_attrs", "uw2t_hang"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l42_"), verbose=False)
        for s, p, c in UW2T_VARIANTS:
            a = _UW2T["mod"].uw2t_attrs(s, p, c)
            _log(f"uw2t s{s} pp{p} cs{c} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] > 32 else ''}")
        err = _UW2T["mod"].uw2t_setup()
        if err:
            raise RuntimeError(f"uw2t_setup returned {err}")
    except Exception as exc:
        _UW2T.pop("mod", None)
        _log(f"uw2t: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for cls, variants in _L42_CASES:
        if "mod" not in _UW2T:
            break
        _, _, bclass = cls
        name = f"Sv{bclass}"
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            _log(f"{name} v14 {CUDA_ROUTE.get(cls) or 'pdl'}: {base:.1f}us")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for s, p, c in variants:
            route = ("uw2t", s, p, c)
            try:
                err = float((_uw2t_run(data, s, p, c).float() - ref).norm() / ref.norm())   # float(): the kernel finished
                if _UW2T["mod"].uw2t_hang():
                    _log(f"{name} {route}: HANG (watchdog): UW2t dropped")
                    _UW2T.pop("mod", None)
                    break
                us = lab.time_graph(lambda: _uw2t_run(data, s, p, c))
                if _UW2T["mod"].uw2t_hang():
                    _log(f"{name} {route}: HANG in timing (watchdog): UW2t dropped")
                    _UW2T.pop("mod", None)
                    break
                _log(f"{name} {route}: {us:.1f}us err {err:.1e}")
                if err < _L42_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} {route}: FAILED {_why(exc)}")
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
