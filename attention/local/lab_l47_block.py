


# ---------------------------------------------------------------------------------------------
# Lab L47 (check runs only), on v14 (the timed path = v14 byte for byte): UD4c (make_ud4c.py: UD4's kernel byte for
# byte, clusters of 2 / 4 / 8 / 16, more warps / stages) on sliding decode Sd1 / Sd16 / Sd32 vs v14's UD4 routes, with
# each variant's registers, shared memory, max active clusters for the call's grid and CTAs per SM. L34 saw cs16 cost
# 40 vs 25 us (cs8) at Sd32: if cluster placement makes waves, smaller clusters fix it. Routed when correct (< 0.02)
# and >= 3% faster (("ud4c", cs, warps, stages)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UD4C = {}
_L47_ERR = 0.02
_L47_CASES = [("Sd32", (False, False, 32), [(8, 4, 3), (4, 4, 3), (4, 4, 4), (4, 8, 3), (4, 8, 2), (2, 8, 3), (2, 8, 4), (8, 8, 3)]),
              ("Sd16", (False, False, 16), [(8, 4, 3), (4, 4, 3), (4, 4, 4), (4, 8, 3), (2, 8, 3), (8, 8, 3)]),
              ("Sd1", (False, False, 1), [(16, 4, 3), (8, 8, 3), (16, 8, 2)])]


def _ud4c_run(data, cs, warps, stages):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _UD4C["mod"].ud4c(data["qkv"], data["positions"], data["cache_loc"],
                      _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                      data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], out, batch, hq,
                      hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"], int(rope),
                      cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
    return out


_cuda_ok_l47 = _cuda_ok
_cuda_run_l47 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UD4c routes run on the lab's module)
    if route[0] == "ud4c":
        return ("mod" in _UD4C and cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1 and cfg["window"] > 0)
    return _cuda_ok_l47(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud4c":
        return _ud4c_run(data, *route[1:])
    return _cuda_run_l47(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4747)
    _log(f"L47: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UD4C["mod"] = load_inline(name="okc_l47_ud4c", cpp_sources=CUDA_UD4C_DECL, cuda_sources=CUDA_UD4C_SRC,
                                   functions=["ud4c", "ud4c_setup", "ud4c_attrs"],
                                   extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                   build_directory=tempfile.mkdtemp(prefix="okc_l47_"), verbose=False)
        r = _UD4C["mod"].ud4c_setup()
        _log(f"ud4c setup err {r}")
        if r:
            raise RuntimeError(f"ud4c_setup refused {r}")
    except Exception as exc:
        _UD4C.pop("mod", None)
        _log(f"ud4c: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, variants in _L47_CASES:
        if lab.left() < 40:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 1, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for cs, w, s in variants:
            route = ("ud4c", cs, w, s)
            try:
                info = list(_UD4C["mod"].ud4c_attrs(w, s, cs, cls[2]))   # regs, local B, smem B, max clusters, CTAs/SM
                err = float((_ud4c_run(data, cs, w, s).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ud4c_run(data, cs, w, s))
                res.append(f"cs{cs} w{w} s{s} {us:.1f}us err {err:.1e} info {info}")
                if err < _L47_ERR and (best is None or us < best[0]):
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
