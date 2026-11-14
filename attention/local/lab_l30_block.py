

# ---------------------------------------------------------------------------------------------
# Lab L30 (check runs only), on v10 (the timed path = v10 byte for byte): UW2 (cuda_uw2.py: one launch per sliding
# verify layer, UV4's prologue + cluster merge around W2c) on Sv1, Sv16, Sv32 in (stages, pp, cs) variants: error vs the
# FP32 reference and time vs v10's path; a class moves to the fastest correct variant >= 3% faster (the lab's _cuda_ok /
# _cuda_run accept ("uw2", stages, pp, cs)). ptxas -v and attrs for UW2; prints ROUTES = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UW2 = {}
_L30_CASES = [((False, True, 1), [(4, 1, 16), (6, 0, 16), (4, 1, 8)]),
              ((False, True, 16), [(4, 1, 8), (4, 1, 4), (4, 1, 16)]),
              ((False, True, 32), [(4, 1, 4), (4, 1, 8)])]
_L30_ERR = 0.12


def _uw2_run(data, stages, pp, cs):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
    _UW2["mod"].uw2(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                    data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                    data["cache_seqlens"], out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"], cfg["window"],
                    int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
    return out


_cuda_ok_l30 = _cuda_ok
_cuda_run_l30 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UW2 routes run on the lab's module)
    if route[0] == "uw2":
        return ("mod" in _UW2 and cfg["q_len"] == 8 and cfg["window"] > 0 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l30(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "uw2":
        return _uw2_run(data, *route[1:])
    return _cuda_run_l30(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3030)
    _log(f"L30: timed path v10: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UW2["mod"] = load_inline(name="okc_l30_uw2", cpp_sources=CUDA_UW2_DECL, cuda_sources=CUDA_UW2_SRC,
                                  functions=["uw2", "uw2_setup", "uw2_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas", "-v"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l30_"), verbose=True)
        err = _UW2["mod"].uw2_setup()
        if err:
            raise RuntimeError(f"uw2_setup returned {err}")
        for s, p, c in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)):
            a = _UW2["mod"].uw2_attrs(s, p, c)
            _log(f"uw2 s{s} pp{p} cs{c} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _UW2.pop("mod", None)
        _log(f"uw2: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for cls, variants in _L30_CASES:
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
            _log(f"{name} current {CUDA_ROUTE.get(cls) or 'pdl'}: {base:.1f}us")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for s, p, c in variants:
            route = ("uw2", s, p, c)
            try:
                err = float((_uw2_run(data, s, p, c).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _uw2_run(data, s, p, c))
                _log(f"{name} {route}: {us:.1f}us err {err:.1e}")
                if err < _L30_ERR and (best is None or us < best[0]):
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
