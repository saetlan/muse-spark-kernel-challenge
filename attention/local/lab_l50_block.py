


# ---------------------------------------------------------------------------------------------
# Lab L50 (check runs only), on v14 (the timed path = v14 byte for byte): UD6 (make_ud6.py: UD5 with `groups` clusters
# per request and a ticketed second-level merge: one launch per full decode layer, no prologue / combine kernels) on
# full decode Fd1 / Fd16 / Fd32 (and Sd32 at groups 1 = UD5) vs v14's path; per variant: registers, shared memory, max
# active clusters for the grid; tickets checked back at zero after the runs. Routed when correct (< 0.02) and >= 3%
# faster (("ud6", cs, warps, stages, groups)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UD6 = {}
_L50_ERR = 0.02
_L50_CASES = [("Fd1", (True, False, 1), 0, [(16, 4, 3, 7), (8, 4, 3, 16), (4, 4, 3, 33), (4, 4, 3, 66), (8, 8, 3, 16)]),
              ("Fd16", (True, False, 16), 0, [(8, 4, 3, 1), (4, 4, 3, 2), (4, 4, 3, 4), (4, 4, 3, 5), (4, 8, 3, 2)]),
              ("Fd32", (True, False, 32), 0, [(4, 4, 3, 1), (4, 4, 3, 2), (4, 8, 3, 1), (4, 4, 4, 2)]),
              ("Sd32", (False, False, 32), 2048, [(4, 4, 3, 1)])]


def _ud6_bufs(batch, hkv, cs, groups, dev):
    key = (batch, hkv, cs, groups)
    if key not in _UD6:
        bp = batch * (hkv // 2)
        _UD6[key] = (torch.zeros(max(1, bp * groups * 16 * 68), dtype=torch.float32, device=dev),
                     torch.zeros(max(1, bp * cs), dtype=torch.int32, device=dev))
    return _UD6[key]


def _ud6_run(data, cs, warps, stages, groups):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    gpart, tickets = _ud6_bufs(batch, hkv, cs, groups, device)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _UD6["mod"].ud6(data["qkv"], data["positions"], data["cache_loc"],
                    _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                    data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                    hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                    cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs, groups, gpart, tickets)
    return out


_cuda_ok_l50 = _cuda_ok
_cuda_run_l50 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UD6 routes run on the lab's module)
    if route[0] == "ud6":
        return ("mod" in _UD6 and cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1)
    return _cuda_ok_l50(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud6":
        return _ud6_run(data, *route[1:])
    return _cuda_run_l50(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5050)
    _log(f"L50: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UD6["mod"] = load_inline(name="okc_l50_ud6", cpp_sources=CUDA_UD6_DECL, cuda_sources=CUDA_UD6_SRC,
                                  functions=["ud6", "ud6_setup", "ud6_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l50_"), verbose=False)
        r = _UD6["mod"].ud6_setup()
        _log(f"ud6 setup err {r}")
        if r:
            raise RuntimeError(f"ud6_setup refused {r}")
    except Exception as exc:
        _UD6.pop("mod", None)
        _log(f"ud6: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window, variants in _L50_CASES:
        if lab.left() < 40:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 1, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for cs, w, s, g in variants:
            route = ("ud6", cs, w, s, g)
            try:
                info = list(_UD6["mod"].ud6_attrs(w, s, cs, cls[2] * g))   # regs, local B, smem B, max clusters, CTAs/SM
                err = float((_ud6_run(data, cs, w, s, g).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ud6_run(data, cs, w, s, g))
                err2 = float((_ud6_run(data, cs, w, s, g).float() - ref).norm() / ref.norm())   # after many replays
                tk = int(_ud6_bufs(cls[2], 2, cs, g, dev)[1].abs().sum())
                res.append(f"cs{cs} w{w} s{s} g{g} {us:.1f}us err {err:.1e}/{err2:.1e} tickets {tk} info {info}")
                if max(err, err2) < _L50_ERR and tk == 0 and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"cs{cs} w{w} s{s} g{g} FAILED {_why(exc)[:120]}")
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
