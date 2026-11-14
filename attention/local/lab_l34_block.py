


# ---------------------------------------------------------------------------------------------
# Lab L34 (check runs only), on v13 (the timed path = v13 byte for byte): UD4 (cuda_ud4.py, make_ud4.py, AF59: UD3 with
# row-owner finalize and the new token's K / V prefetched at entry) on Sd1, Sd16, Sd32 vs v13's path (UD3 on Sd1 /
# Sd16, the PDL Triton chain on Sd32): error vs the FP32 reference and time; a class moves to the fastest correct
# variant >= 3% faster (the lab's _cuda_ok / _cuda_run accept ("ud4", cs, warps, stages)). Prints ROUTES = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UD4 = {}
_L34_ERR = 0.05
_L34_CASES = [("Sd1", (False, False, 1), [(16, 4, 3), (16, 8, 2), (8, 4, 3)]),
              ("Sd16", (False, False, 16), [(8, 4, 3), (8, 8, 2), (16, 4, 3)]),
              ("Sd32", (False, False, 32), [(8, 4, 3), (8, 8, 2), (16, 4, 3)])]


def _ud4_run(data, cs, warps, stages):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _UD4["mod"].ud4(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                    data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                    data["cache_seqlens"], out, batch, hq, hkv, data["page_table"].shape[1], cfg["page_size"],
                    cfg["window"], int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs)
    return out


_cuda_ok_l34 = _cuda_ok
_cuda_run_l34 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UD4 routes run on the lab's module)
    if route[0] == "ud4":
        return ("mod" in _UD4 and cfg["q_len"] == 1 and cfg["window"] > 0 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l34(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud4":
        return _ud4_run(data, *route[1:])
    return _cuda_run_l34(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3434)
    _log(f"L34: timed path v13: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UD4["mod"] = load_inline(name="okc_l34_ud4", cpp_sources=CUDA_UD4_DECL, cuda_sources=CUDA_UD4_SRC,
                                  functions=["ud4", "ud4_setup", "ud4_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l34_"), verbose=False)
        err = _UD4["mod"].ud4_setup()
        if err:
            raise RuntimeError(f"ud4_setup returned {err}")
        for w, c in ((4, 16), (8, 16), (4, 8), (8, 8)):
            a = _UD4["mod"].ud4_attrs(w, c)
            _log(f"ud4 w{w} cs{c} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] > 32 else ''}")
    except Exception as exc:
        _UD4.pop("mod", None)
        _log(f"ud4: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, variants in _L34_CASES:
        if lab.left() < 25:
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
        for v in variants:
            try:
                err = float((_ud4_run(data, *v).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ud4_run(data, *v))
                res.append(f"ud4{v} {us:.2f}us err {err:.1e}")
                if err < _L34_ERR and (best is None or us < best[0]):
                    best = (us, ("ud4",) + v)
            except Exception as exc:
                res.append(f"ud4{v} FAILED {_why(exc)[:120]}")
        _log(f"{name} v13 {CUDA_ROUTE.get(cls) or 'pdl/triton'} {base:.2f}us err {berr:.1e}; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.2f}us vs {base:.2f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.2f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
