

# ---------------------------------------------------------------------------------------------
# Lab L46 (check runs only), on v14 (the timed path = v14 byte for byte):
# (1) D3 config sweep on full decode (no new kernel; v14's module, kind dec4): deeper rings (4 stages at 3 CTAs per SM
#     fit 221 KB), 8-warp CTAs and other whole-wave splits on Fd1 / Fd16 / Fd32 vs v14's D3 routes; routed when correct
#     (< 0.02) and >= 3% faster.
# (2) UD4 anatomy (lab_anat_ud4_block.py, make_anatud4.py) on Sd1 / Sd16 / Sd32: where the sliding decode call still goes.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L46_ERR = 0.02
_L46_CASES = [("Fd1", (True, False, 1), [("dec4", 2, 4, 4, "w"), ("dec4", 1, 4, 3, "w"), ("dec4", 1, 8, 3, "w"), ("dec4", 2, 8, 2, "w")]),
              ("Fd16", (True, False, 16), [("dec4", 3, 4, 4, "w"), ("dec4", 2, 8, 3, "w"), ("dec4", 2, 8, 2, "w"), ("dec4", 6, 4, 4)]),
              ("Fd32", (True, False, 32), [("dec4", 6, 4, 4), ("dec4", 3, 4, 4, "w"), ("dec4", 3, 4, 3, "w"), ("dec4", 4, 8, 3)])]


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4646)
    _log(f"L46: timed path v14: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, routes in _L46_CASES:
        if lab.left() < 80:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 1, 0, k_cache, v_cache, gen)
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
                if err < _L46_ERR and (best is None or us < best[0]):
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
    _anat_ud4(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
