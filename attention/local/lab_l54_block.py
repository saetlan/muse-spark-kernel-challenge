


# ---------------------------------------------------------------------------------------------
# Lab L54 (check runs only), on v15 (the timed path = v15 byte for byte): UD7 at L52's winning shape (clusters of 2, ~384
# CTAs) on Fd32, Fd16 re-check and neighbours, Fd1 at cs16 / 8 / 4. UD7 (make_ud7.py: UD6 with multi-row owners, so
# clusters of 1 / 2 with 4-warp CTAs: D3's occupancy and loop + in-kernel prologue + ticketed merge = one launch) on full
# decode Fd16 / Fd32 / Fd1 vs v15's path (D3 + prologue + combine); error before / after the replays, tickets back at 0.
# Routed when correct (< 0.02) and >= 3% faster (('ud7', cs, warps, stages, groups)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_UD7 = {}
_L54_ERR = 0.02
_L54_CASES = [("Fd32", (True, False, 32), 0, [(2, 4, 3, 6), (2, 4, 3, 5), (2, 4, 3, 4), (2, 4, 4, 6), (1, 4, 3, 8)]),
              ("Fd16", (True, False, 16), 0, [(2, 4, 3, 12), (2, 4, 3, 10), (2, 4, 3, 11), (2, 4, 4, 12)]),
              ("Fd1", (True, False, 1), 0, [(16, 4, 3, 7), (8, 4, 3, 16), (4, 4, 3, 33)])]


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


_cuda_ok_l54 = _cuda_ok
_cuda_run_l54 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: UD7 routes run on the lab's module)
    if route[0] == "ud7":
        return ("mod" in _UD7 and cfg["num_q_heads"] // cfg["num_kv_heads"] == 8 and cfg["num_kv_heads"] % 2 == 0
                and cfg["q_len"] == 1)
    return _cuda_ok_l54(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "ud7":
        return _ud7_run(data, *route[1:])
    return _cuda_run_l54(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5454)
    _log(f"L54: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _UD7["mod"] = load_inline(name="okc_l54_ud7", cpp_sources=CUDA_UD7_DECL, cuda_sources=CUDA_UD7_SRC,
                                  functions=["ud7", "ud7_setup", "ud7_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l54_"), verbose=False)
        r = _UD7["mod"].ud7_setup()
        _log(f"ud7 setup err {r}")
        if r:
            raise RuntimeError(f"ud7_setup refused {r}")
    except Exception as exc:
        _UD7.pop("mod", None)
        _log(f"ud7: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window, variants in _L54_CASES:
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
            route = ("ud7", cs, w, s, g)
            try:
                info = list(_UD7["mod"].ud7_attrs(w, s, cs, cls[2] * g))   # regs, local B, smem B, max clusters, CTAs/SM
                err = float((_ud7_run(data, cs, w, s, g).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _ud7_run(data, cs, w, s, g))
                err2 = float((_ud7_run(data, cs, w, s, g).float() - ref).norm() / ref.norm())   # after many replays
                tk = int(_ud7_bufs(cls[2], 2, cs, g, dev)[1].abs().sum())
                res.append(f"cs{cs} w{w} s{s} g{g} {us:.1f}us err {err:.1e}/{err2:.1e} tickets {tk} info {info}")
                if max(err, err2) < _L54_ERR and tk == 0 and (best is None or us < best[0]):
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
