


# ---------------------------------------------------------------------------------------------
# Lab L32 (check runs only), on v12 (the timed path = v12 byte for byte): AF56's whole-row copier. The wgmma verify
# kernels (W3, UW2, W1-N) copy each random 128 B K / V row as two 64 B halves from different warp instructions, the
# pattern AF13 measured at ~1.7-2.1 TB/s; L31 put W3's Fv32 at 2.2 TB/s. make_wholerow.py's b variants change only
# the copier's lane -> (key, chunk) map to whole rows per instruction (cuda_w3b.py, cuda_uw2b.py, cuda_w1nb.py, one
# lab module). Per class: error vs the FP32 reference and time vs v12's path; a class moves to the fastest correct
# variant >= 3% faster (the lab's _cuda_ok / _cuda_run accept ("w3b", s, pp, nwg, rc), ("uw2b", s, pp, cs),
# ("w1nb", cps, stages)). Prints ROUTES = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_WB = {}
_L32_ERR = 0.12
_L32_CASES = [
    ("Fv16", (True, True, 16), 0, [("w3b", 6, 0, 4, 104), ("w3b", 6, 1, 4, 104), ("w3b", 7, 1, 4, 104), ("w1nb", 1, 4)]),
    ("Fv32", (True, True, 32), 0, [("w3b", 6, 1, 4, 104), ("w3b", 6, 0, 4, 104), ("w3b", 7, 1, 4, 104), ("w1nb", 1, 4)]),
    ("Sv1", (False, True, 1), 2048, [("uw2b", 6, 0, 16), ("uw2b", 4, 1, 16)]),
    ("Sv16", (False, True, 16), 2048, [("uw2b", 4, 1, 4), ("uw2b", 4, 1, 8)]),
    ("Sv32", (False, True, 32), 2048, [("w1nb", 1, 4), ("w1nb", 1, 6), ("w3b", 4, 1, 2, 0)]),
    ("Fv1", (True, True, 1), 0, [("w3b", 4, 1, 2, 0), ("w3b", 6, 1, 2, 0), ("w1nb", 1, 4)]),
]


def _wb_parts(data, nsplit):
    cfg = data["config"]
    rows = cfg["batch"] * cfg["q_len"] * cfg["num_q_heads"] * nsplit
    dev = data["qkv"].device
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=dev)
    m_part = torch.empty((rows,), dtype=torch.float32, device=dev)
    return o_part, m_part, torch.empty_like(m_part)


def _wb_run(data, route):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    mod = _WB["mod"]
    if route[0] == "uw2b":
        _, stages, pp, cs = route
        rope = bool(cfg["rope_theta"])
        out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
        mod.uw2b(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, data["qkv"].device),
                 data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                 data["cache_seqlens"], out, batch, hq, hkv, max_pages, cfg["page_size"], cfg["window"], int(rope),
                 cfg["qk_scale_factor"], cfg["sm_scale"], stages, pp, cs)
        return out
    q = _prologue_q(data)
    if route[0] == "w3b":
        _, stages, pp, nwg, rc = route
        nsplit = _nsplit_w(cfg, max_pages, 1, 64)
        nparts = nsplit * (nwg // 2)
        o_part, m_part, l_part = _wb_parts(data, nparts)
        mod.attend_w3b(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                       o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                       cfg["sm_scale"], stages, pp, nwg, rc)
        return _merge(data, o_part, m_part, l_part, nparts)
    _, cps, stages = route
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    o_part, m_part, l_part = _wb_parts(data, nsplit)
    mod.attend_w1nb(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                    o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                    cfg["sm_scale"], stages)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l32 = _cuda_ok
_cuda_run_l32 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: whole-row routes run on the lab's module)
    if route[0] in ("w3b", "uw2b", "w1nb"):
        ok = "mod" in _WB and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0
        return ok and (route[0] != "uw2b" or cfg["window"] > 0)
    return _cuda_ok_l32(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] in ("w3b", "uw2b", "w1nb"):
        return _wb_run(data, route)
    return _cuda_run_l32(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3232)
    _log(f"L32: timed path v12: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _WB["mod"] = load_inline(
            name="okc_l32_wholerow", cpp_sources="\n".join([CUDA_W3B_DECL, CUDA_UW2B_DECL, CUDA_W1NB_DECL]),
            cuda_sources="\n".join([CUDA_W3B_SRC, CUDA_UW2B_SRC, CUDA_W1NB_SRC]),
            functions=["attend_w3b", "attend_w3b_setup", "attend_w3b_attrs", "uw2b", "uw2b_setup", "uw2b_attrs",
                       "attend_w1nb", "attend_w1nb_setup"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=tempfile.mkdtemp(prefix="okc_l32_"), verbose=False)
        r = (_WB["mod"].attend_w3b_setup(), _WB["mod"].uw2b_setup(), _WB["mod"].attend_w1nb_setup())
        _log(f"setup (w3b refused bits, uw2b, w1nb): {r}; w3b s6 nwg4 attrs {_WB['mod'].attend_w3b_attrs(6, 1, 4, 104)}, "
             f"uw2b s6 cs16 attrs {_WB['mod'].uw2b_attrs(6, 0, 16)}")
        if r[1] or r[2]:
            raise RuntimeError(f"setup refused {r}")
    except Exception as exc:
        _WB.pop("mod", None)
        _log(f"build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window, routes in _L32_CASES:
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for route in routes:
            try:
                err = float((_wb_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _wb_run(data, route))
                res.append(f"{route[0]}{route[1:]} {us:.1f}us err {err:.1e}")
                if err < _L32_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v12 {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us err {berr:.1e}; " + "; ".join(res))
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
