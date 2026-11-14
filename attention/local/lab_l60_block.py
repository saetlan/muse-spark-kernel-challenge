


# ---------------------------------------------------------------------------------------------
# Lab L60 (check runs only), on v17 (the timed path = v17 byte for byte): W3b (v17's full-verify kernel: two consumer pairs,
# shared copier ring) on sliding verify Sv16 / Sv32 / Sv1 vs v17's routes (UW2h, w1n + multi-row combine). Only the
# one-pair W3e was timed there (L44: Sv32 30.4 vs w1n 30.2). No new kernel: v17's module, kind w3b. Base and variants
# timed twice (min); routed when correct (< 0.12) and >= 3% faster.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L60_CASES = [("Sv16", (False, True, 16), [("w3b", 6, 1, 4, 104), ("w3b", 7, 1, 4, 104), ("w3b", 6, 0, 4, 104), ("w3b", 6, 1, 2, 0)]),
              ("Sv32", (False, True, 32), [("w3b", 6, 1, 4, 104), ("w3b", 6, 0, 4, 104), ("w3b", 4, 1, 2, 0)]),
              ("Sv1", (False, True, 1), [("w3b", 6, 1, 4, 104), ("w3b", 4, 1, 2, 0)])]


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(6060)
    _log(f"L60: timed path v17: {sorted(CUDA_ROUTE.items())}")
    if "mod" not in _CUDA:
        _log("no CUDA module")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls, routes in _L60_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = min(lab.time_graph(lambda: _kernel_fn_submitted(data)), lab.time_graph(lambda: _kernel_fn_submitted(data)))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for route in routes:
            try:
                err = float((_cuda_run(data, route).float() - ref).norm() / ref.norm())
                us = min(lab.time_graph(lambda: _cuda_run(data, route)), lab.time_graph(lambda: _cuda_run(data, route)))
                res.append(f"{route[1:]} {us:.1f}us err {err:.1e}")
                if err < 0.12 and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:120]}")
        _log(f"{name} v17 {CUDA_ROUTE.get(cls)} {base:.1f}us (min of 2) err {berr:.1e}; w3b " + "; ".join(res))
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
