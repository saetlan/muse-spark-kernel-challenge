


# ---------------------------------------------------------------------------------------------
# Lab L39 (check runs only), on v14 (the timed path = v14 byte for byte): AF64's cheap A22. W3b's equal splits per
# request in one wave leave full verify waiting on the longest request (~5%); D3 at Fd32 runs several waves and shows no
# tail. Here W3b (v14's module, kind w3b) with more, shorter CTAs: splits for cps = 2, 3, 4 CTAs per SM's worth
# (_nsplit_w(cfg, max_pages, cps, 64); nsplit x 2 partials) on Fv16, Fv32, Fv1; and D3 at Fd16 with Fd32's multi-wave
# split ('dec4', 6, 4, 3). Routed when correct and >= 3% faster (("w3bm", cps, stages, pp, nwg, rc); dec4 routes as
# v14's own). Also re-times v14's path on agentic vs mean-length inputs for the chosen W3b split (tail after the change).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L39_ERR = 0.12
_L39_CASES = [("Fv16", (True, True, 16), 0, [("w3bm", 2, 6, 1, 4, 104), ("w3bm", 3, 6, 1, 4, 104), ("w3bm", 4, 6, 1, 4, 104)]),
              ("Fv32", (True, True, 32), 0, [("w3bm", 2, 6, 1, 4, 104), ("w3bm", 3, 6, 1, 4, 104), ("w3bm", 4, 6, 1, 4, 104)]),
              ("Fv1", (True, True, 1), 0, [("w3bm", 2, 6, 1, 4, 104), ("w3bm", 3, 4, 1, 4, 104)]),
              ("Fd16", (True, False, 16), 0, [("dec4", 6, 4, 3), ("dec4", 6, 4, 3, "w")])]


def _w3bm_run(data, cps, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    nparts = nsplit * (nwg // 2)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nparts, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nparts,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _CUDA["mod"].attend_w3b(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                            data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                            cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


_cuda_ok_l39 = _cuda_ok
_cuda_run_l39 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: multi-wave W3b on v14's module)
    if route[0] == "w3bm":
        return cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0 and cfg["q_len"] == 8
    return _cuda_ok_l39(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3bm":
        return _w3bm_run(data, *route[1:])
    return _cuda_run_l39(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3939)
    _log(f"L39: timed path v14: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, window, routes in _L39_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            length = 8 if cls[1] else 1
            data = _lab_inputs(cls[2], length, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for route in routes:
            try:
                err = float((_cuda_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _cuda_run(data, route))
                res.append(f"{route} {us:.1f}us err {err:.1e}")
                if err < _L39_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls)} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _lab_tail(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
