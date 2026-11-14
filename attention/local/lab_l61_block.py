


# ---------------------------------------------------------------------------------------------
# Lab L61 (check runs only), on v18 (the timed path = v18 byte for byte): small candidates, each timed twice (min) vs v18:
# Fd1 D3p at 1 CTA per SM (v18's Fd1 is D3 (1, 4, 3, 'w'); L38 timed D3p only at cps 2 / 3); Fv16 / Fv32 W3e (cuda_w3e.py,
# L44: Fv16 -2.5% once); Sd16 UD9 cs16 s2 / s3 (L59 lab: no gain, but the judge gave UD9 more than the lab at Sd32).
# Routed when correct and >= 3% faster.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3E = {}
_U9 = {}
_L61_CASES = [("Fd1", (True, False, 1), 1, 0, [("dec4p", 1, 4, 3, 1), ("dec4p", 1, 4, 3, 0), ("dec4p", 1, 4, 4, 1)]),
              ("Fv16", (True, True, 16), 8, 0, [("w3e", 6, 0, 4, 104), ("w3e", 6, 1, 4, 104)]),
              ("Fv32", (True, True, 32), 8, 0, [("w3e", 6, 0, 4, 104)]),
              ("Sd16", (False, False, 16), 1, 2048, [("ud9", 16, 4, 2), ("ud9", 16, 4, 3)])]
_L61_ERR = {"dec4p": 0.02, "ud9": 0.02, "w3e": 0.12}


def _w3e_run(data, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    nparts = nsplit * (nwg // 2)
    rows = batch * length * hq * nparts
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((rows,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W3E["mod"].attend_w3e(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                           o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                           cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


def _l61_run(data, route):
    if route[0] == "w3e":
        return _w3e_run(data, *route[1:])
    if route[0] == "ud9":   # the lab's own UD9 module (v18 instantiates only its routed variant)
        _, cs, warps, stages = route
        cfg = data["config"]
        batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
        device = data["qkv"].device
        rope = bool(cfg["rope_theta"])
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        _U9["mod"].ud9(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                       data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"],
                       out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                       cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
        return out
    return _cuda_run(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(6161)
    _log(f"L61: timed path v18: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _W3E["mod"] = load_inline(name="okc_l61_w3e", cpp_sources=CUDA_W3E_DECL, cuda_sources=CUDA_W3E_SRC,
                                  functions=["attend_w3e", "attend_w3e_setup", "attend_w3e_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l61_"), verbose=False)
        _log(f"w3e setup {_W3E['mod'].attend_w3e_setup()}")
        _U9["mod"] = load_inline(name="okc_l61_ud9", cpp_sources=CUDA_U9_DECL, cuda_sources=CUDA_U9_SRC,
                                 functions=["ud9", "ud9_setup", "ud9_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l61b_"), verbose=False)
        _log(f"ud9 setup {_U9['mod'].ud9_setup()}")
    except Exception as exc:
        _W3E.pop("mod", None) if "mod" in _W3E and "mod" not in _U9 else None
        _log(f"w3e build FAILED {_why(exc)}")
    mod = _CUDA.get("mod")
    _log(f"v18 module has dec4p {hasattr(mod, 'attend_dec4p')}, ud9 {hasattr(mod, 'ud9')}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    for name, cls, length, window, routes in _L61_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], length, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = min(lab.time_graph(lambda: _kernel_fn_submitted(data)), lab.time_graph(lambda: _kernel_fn_submitted(data)))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        res = []
        for route in routes:
            try:
                if (route[0] == "w3e" and "mod" not in _W3E) or (route[0] == "ud9" and "mod" not in _U9):
                    raise RuntimeError("no w3e module")
                err = float((_l61_run(data, route).float() - ref).norm() / ref.norm())
                us = min(lab.time_graph(lambda: _l61_run(data, route)), lab.time_graph(lambda: _l61_run(data, route)))
                ok = err < _L61_ERR[route[0]]
                res.append(f"{route} {us:.1f}us ({(us / base - 1) * 100:+.1f}%) err {err:.1e}{'' if ok else ' BAD'}")
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v18 {CUDA_ROUTE.get(cls)} {base:.1f}us (min of 2) err {berr:.1e}; " + "; ".join(res))
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
