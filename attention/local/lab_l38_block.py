


# ---------------------------------------------------------------------------------------------
# Lab L38 (check runs only), on v14 (the timed path = v14 byte for byte):
# (1) D3p whole-wave PDL chain (cuda_dec4p.py, make_pdl.py: D3 as a PDL dependent of the Triton prologue, page entries
#     issued before griddepcontrol.wait; _prologue_p / _combine_p with launch_pdl) on Fd1, Fd16, Fd32 vs v14's D3 'w'
#     chain (same split rule: _nsplit_w(cps, 32 * warps)); late and early triggers; error vs FP32 on two input sets in a
#     row (a read before the wait shows on the second); routed when correct and >= 3% faster (("dec4p", cps, warps,
#     stages, early)). L13 measured D3p on the old split: Fd1 40.1 -> 37.7 us.
# (2) the A22 tail test (lab_tail_block.py, B6).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_D3P = {}
_L38_ERR = 0.02
_L38_CASES = [("Fd1", (True, False, 1), [(2, 4, 3, 0), (2, 4, 3, 1), (3, 4, 3, 0)]),
              ("Fd16", (True, False, 16), [(3, 4, 3, 0), (3, 4, 3, 1)]),
              ("Fd32", (True, False, 32), [(6, 4, 3, 0), (6, 4, 3, 1)])]


def _d3p_run(data, cps, warps, stages, early):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue_p[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                          q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                          HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS,
                          TRIGGER_EARLY=bool(early), num_warps=4)
    nsplit = _nsplit_w(cfg, max_pages, cps, 32 * warps)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _D3P["mod"].attend_dec4p(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                             data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                             cfg["page_size"], cfg["window"], cfg["sm_scale"], warps, stages, int(early))
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    cb = min(MERGE_CB, triton.next_power_of_2(nsplit))
    _combine_p[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                            HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=8 if cb > 64 else 4, launch_pdl=True)
    return out


_cuda_ok_l38 = _cuda_ok
_cuda_run_l38 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: D3p routes run on the lab's module)
    if route[0] == "dec4p":
        return ("mod" in _D3P and cfg["q_len"] == 1 and cfg["window"] == 0
                and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l38(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "dec4p":
        return _d3p_run(data, *route[1:])
    return _cuda_run_l38(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3838)
    _log(f"L38: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _D3P["mod"] = load_inline(name="okc_l38_d3p", cpp_sources=CUDA_DEC4P_DECL, cuda_sources=CUDA_DEC4P_SRC,
                                  functions=["attend_dec4p", "attend_dec4p_setup"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l38_"), verbose=False)
        if _D3P["mod"].attend_dec4p_setup():
            raise RuntimeError("attend_dec4p_setup refused")
    except Exception as exc:
        _D3P.pop("mod", None)
        _log(f"d3p: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, variants in _L38_CASES:
        if "mod" not in _D3P or lab.left() < 120:
            break
        try:
            data = _lab_inputs(cls[2], 1, 0, k_cache, v_cache, gen)
            data_b = _lab_inputs(cls[2], 1, 0, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            _kernel_fn_submitted(data_b)
            ref_b = _lab_reference(data_b)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for v in variants:
            try:
                _d3p_run(data, *v)
                err = float((_d3p_run(data_b, *v).float() - ref_b).norm() / ref_b.norm())
                us = lab.time_graph(lambda: _d3p_run(data, *v))
                res.append(f"dec4p{v} {us:.2f}us err {err:.1e}")
                if err < _L38_ERR and (best is None or us < best[0]):
                    best = (us, ("dec4p",) + v)
            except Exception as exc:
                res.append(f"dec4p{v} FAILED {_why(exc)[:120]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls)} {base:.2f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.2f}us vs {base:.2f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.2f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _lab_tail(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
