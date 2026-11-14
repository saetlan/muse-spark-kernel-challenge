


# ---------------------------------------------------------------------------------------------
# Lab L59 (check runs only), on v15 (the timed path = v15 byte for byte): UD9 (make_ud9.py: UD8 with a compact receive
# buffer, so cs8 / cs16 with 4-warp CTAs fit 3 CTAs per SM: Sd32 cs8 = 256 CTAs and Sd16 cs16 = 256 CTAs in one wave, twice
# the warps per SM of v15's routes) vs v15's UD5 routes; per variant max active clusters for the grid; base and variants
# timed twice (min); routed when correct (< 0.02) and >= 3% faster (('ud9', cs, warps, stages)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UD9 = {}
_L59_ERR = 0.02
_L59_CASES = [("Sd32", (False, False, 32), [(8, 4, 3), (8, 4, 2), (4, 4, 3), (16, 4, 2)]),
              ("Sd16", (False, False, 16), [(16, 4, 3), (16, 4, 2), (8, 4, 3)]),
              ("Sd1", (False, False, 1), [(16, 4, 3)])]


def _ud9_run(data, cs, warps, stages):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _UD9["mod"].ud9(data["qkv"], data["positions"], data["cache_loc"],
                      _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                      data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                      hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                      cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
    return out


_cuda_ok_l59 = _cuda_ok
_cuda_run_l59 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UD4c routes run on the lab's module)
    if route[0] == "ud9":
        return ("mod" in _UD9 and cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1 and cfg["window"] > 0)
    return _cuda_ok_l59(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud9":
        return _ud9_run(data, *route[1:])
    return _cuda_run_l59(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5959)
    _log(f"L59: timed path v15: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UD9["mod"] = load_inline(name="okc_l59_ud9", cpp_sources=CUDA_UD9_DECL, cuda_sources=CUDA_UD9_SRC,
                                   functions=["ud9", "ud9_setup", "ud9_attrs"],
                                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                   build_directory=tempfile.mkdtemp(prefix="okc_l59_"), verbose=False)
        r = _UD9["mod"].ud9_setup()
        _log(f"ud9 setup err {r}")
        if r:
            raise RuntimeError(f"ud9_setup refused {r}")
    except Exception as exc:
        _UD9.pop("mod", None)
        _log(f"ud9: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, variants in _L59_CASES:
        if lab.left() < 40:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 1, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = min(lab.time_graph(lambda: _kernel_fn_submitted(data)), lab.time_graph(lambda: _kernel_fn_submitted(data)))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for cs, w, s in variants:
            route = ("ud9", cs, w, s)
            try:
                info = list(_UD9["mod"].ud9_attrs(w, s, cs, cls[2]))   # regs, local B, smem B, max clusters, CTAs/SM
                err = float((_ud9_run(data, cs, w, s).float() - ref).norm() / ref.norm())
                us = min(lab.time_graph(lambda: _ud9_run(data, cs, w, s)), lab.time_graph(lambda: _ud9_run(data, cs, w, s)))
                res.append(f"cs{cs} w{w} s{s} {us:.1f}us err {err:.1e} info {info}")
                if err < _L59_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"cs{cs} w{w} s{s} FAILED {_why(exc)[:120]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls)} {base:.1f}us err {berr:.1e}; " + "; ".join(res))
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
