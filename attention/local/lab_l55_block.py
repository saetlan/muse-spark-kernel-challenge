


# ---------------------------------------------------------------------------------------------
# Lab L55 (check runs only), on v15 (the timed path = v15 byte for byte): more CTAs per sliding request. UW2 at Sv1 runs 16
# CTAs x 3 stages (AF62: loop 2.7 us of a 10.2 us span); UW2g (L51: correct, tickets 0) splits each request over ncl
# clusters; UD7 (L52 / L54) does the same for sliding decode with `groups`. Sv1 / Sv16 with UW2g ncl 1..4, Sd1 / Sd16 with
# UD7 groups 1..4, vs v15's routes. Routed when correct and >= 3% faster.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UG = {}
_UD7 = {}
_L55_CASES = [("Sv1", (False, True, 1), 8, [("uw2g", 6, 0, 16, 2), ("uw2g", 6, 0, 16, 3), ("uw2g", 6, 0, 16, 4), ("uw2g", 4, 1, 16, 2), ("uw2g", 6, 0, 8, 4)]),
              ("Sv16", (False, True, 16), 8, [("uw2g", 4, 1, 4, 2), ("uw2g", 4, 1, 8, 1), ("uw2g", 6, 0, 8, 1)]),
              ("Sd1", (False, False, 1), 1, [("ud7", 16, 4, 3, 2), ("ud7", 16, 4, 3, 3), ("ud7", 8, 4, 3, 4), ("ud7", 8, 4, 3, 2), ("ud7", 4, 4, 3, 8)]),
              ("Sd16", (False, False, 16), 1, [("ud7", 8, 4, 3, 2), ("ud7", 4, 4, 3, 2), ("ud7", 2, 4, 3, 4)])]
_L55_ERR = {"uw2g": 0.12, "ud7": 0.02}


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


def _ud7_bufs(batch, hkv, cs, groups, dev):
    key = (batch, hkv, cs, groups)
    if key not in _UD7:
        bp = batch * (hkv // 2)
        _UD7[key] = (torch.zeros(max(1, bp * groups * 16 * 68), dtype=torch.float32, device=dev),
                     torch.zeros(max(1, bp * cs), dtype=torch.int32, device=dev))
    return _UD7[key]


def _ud7_run(data, cs, warps, stages, groups):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    gpart, tickets = _ud7_bufs(batch, hkv, cs, groups, device)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _UD7["mod"].ud7(data["qkv"], data["positions"], data["cache_loc"],
                    _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                    data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                    hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                    cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs, groups, gpart, tickets)
    return out



def _l55_run(data, route):
    if route[0] == "uw2g":
        return _ug_run(data, *route[1:])
    return _ud7_run(data, *route[1:])


def _l55_bufs(route, batch):
    if route[0] == "uw2g":
        return _ug_bufs(batch, 2, route[3], route[4], torch.device("cuda"))[1]
    return _ud7_bufs(batch, 2, route[1], route[4], torch.device("cuda"))[1]


_cuda_ok_l55 = _cuda_ok
_cuda_run_l55 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: routes run on the lab's modules)
    if route[0] == "uw2g":
        return "mod" in _UG and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
    if route[0] == "ud7":
        return "mod" in _UD7 and cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 1
    return _cuda_ok_l55(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] in ("uw2g", "ud7"):
        return _l55_run(data, route)
    return _cuda_run_l55(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5555)
    _log(f"L55: timed path v15: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UG["mod"] = load_inline(name="okc_l55_uw2g", cpp_sources=CUDA_UW2G_DECL, cuda_sources=CUDA_UW2G_SRC,
                                 functions=["uw2g", "uw2g_setup", "uw2g_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l55a_"), verbose=False)
        _UD7["mod"] = load_inline(name="okc_l55_ud7", cpp_sources=CUDA_UD7_DECL, cuda_sources=CUDA_UD7_SRC,
                                  functions=["ud7", "ud7_setup", "ud7_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l55b_"), verbose=False)
        r = (_UG["mod"].uw2g_setup(), _UD7["mod"].ud7_setup())
        _log(f"setup {r}")
        if any(r):
            raise RuntimeError(f"setup refused {r}")
    except Exception as exc:
        _UG.pop("mod", None)
        _UD7.pop("mod", None)
        _log(f"build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, length, routes in _L55_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], length, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            base2 = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        base = min(base, base2)
        best, res = None, []
        for route in routes:
            try:
                err = float((_l55_run(data, route).float() - ref).norm() / ref.norm())
                us = min(lab.time_graph(lambda: _l55_run(data, route)), lab.time_graph(lambda: _l55_run(data, route)))
                err2 = float((_l55_run(data, route).float() - ref).norm() / ref.norm())
                tk = int(_l55_bufs(route, cls[2]).abs().sum())
                res.append(f"{route} {us:.1f}us err {err:.1e}/{err2:.1e} tk {tk}")
                if max(err, err2) < _L55_ERR[route[0]] and tk == 0 and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v15 {CUDA_ROUTE.get(cls)} {base:.1f}us (min of 2) err {berr:.1e}; " + "; ".join(res))
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
