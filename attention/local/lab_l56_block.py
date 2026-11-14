


# ---------------------------------------------------------------------------------------------
# Lab L56 (check runs only), on v15 (the timed path = v15 byte for byte): UW2h (make_uw2h.py: UW2 pushing o / l as FP16
# pairs + FP32 m, l: half the DSMEM bytes; L36: the push scales with cluster traffic, 1.95 us at cs16) on sliding verify
# Sv1 / Sv16 / Sv32 vs v15's path; routed when correct (< 0.12) and >= 3% faster (("uw2h", stages, pp, cs)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UV = {}
_L56_ERR = 0.12
_L56_CASES = [("Sv1", (False, True, 1), [(6, 0, 16), (4, 1, 16)]),
              ("Sv16", (False, True, 16), [(4, 1, 4), (4, 1, 8)]),
              ("Sv32", (False, True, 32), [(4, 1, 4), (4, 1, 8)])]


def _uw2h_run(data, stages, pp, cs):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    _UV["mod"].uw2h(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, data["qkv"].device),
                    data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                    data["cache_seqlens"], out, batch, hq, hkv, max_pages, cfg["page_size"], cfg["window"], int(rope),
                    cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
    return out


_cuda_ok_l56 = _cuda_ok
_cuda_run_l56 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UW2h routes run on the lab's module)
    if route[0] == "uw2h":
        return ("mod" in _UV and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0 and cfg["window"] > 0)
    return _cuda_ok_l56(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "uw2h":
        return _uw2h_run(data, *route[1:])
    return _cuda_run_l56(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5656)
    _log(f"L56: timed path v15: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UV["mod"] = load_inline(name="okc_l56_uw2h", cpp_sources=CUDA_UW2H_DECL, cuda_sources=CUDA_UW2H_SRC,
                                 functions=["uw2h", "uw2h_setup", "uw2h_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l56_"), verbose=False)
        r = _UV["mod"].uw2h_setup()
        _log(f"uw2h setup {r}; attrs s6 p0 cs16 {_UV['mod'].uw2h_attrs(6, 0, 16)}, s4 p1 cs4 {_UV['mod'].uw2h_attrs(4, 1, 4)}")
        if r:
            raise RuntimeError(f"uw2h_setup refused {r}")
    except Exception as exc:
        _UV.pop("mod", None)
        _log(f"uw2h: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, variants in _L56_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for s, p, cs in variants:
            route = ("uw2h", s, p, cs)
            try:
                err = float((_uw2h_run(data, s, p, cs).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _uw2h_run(data, s, p, cs))
                res.append(f"uw2h{(s, p, cs)} {us:.1f}us err {err:.1e}")
                if err < _L56_ERR and (best is None or us < best[0]):
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
