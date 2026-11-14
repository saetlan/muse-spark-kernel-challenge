

# ---------------------------------------------------------------------------------------------
# Lab L17 (check runs only), on v7-lab = v6 (L16's routes) + UV4 guesses on Sv1/16/32 + UD3 guesses
# on Sd16/32 (self-checked at warm-up on disjoint per-request rows; L16 dropped UD3 there only
# because dummy_inputs' wrapped page table made requests share rows). Every candidate is timed
# against the class's current path, with its error vs the FP32 reference (and new cache rows vs
# the current path's for kernels that write them); a class moves only when correct and >= 3% faster:
#   1. UV4 (cuda_uv4.py: one launch per verify layer) on Sv1/16/32 and full verify bs16/32 (one
#      cluster per request), stages 3..8 (bytes in flight: V4 ran at ~1 TB/s whatever its split);
#   2. UV4 with a second level (ncl clusters per request, ticket merge) on full verify bs1;
#   3. UD3 on Sd16 / Sd32 (L16: Sd1 2.27x over SGLang on the judge);
#   4. probe: the best UV4 route and the current path on Fv16 with an L2-resident cache (page
#      table % 8192): memory-bound if the hot time drops a lot, consumer-bound if not.
# The routes it leaves are printed as ROUTES = {...} (ud3 and uv4 entries) for the scored file.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L17_UD3 = [(8, 4, 3), (8, 8, 2), (16, 4, 3), (16, 8, 2)]                       # (cs, warps, stages)
_L17_SV = [(16, 3, 1), (8, 3, 1), (4, 3, 1), (8, 4, 1), (4, 4, 1), (8, 6, 1), (4, 6, 1), (16, 3, 2), (8, 3, 2),
           (4, 3, 2)]                                                             # (cs, stages, mt)
_L17_FV = {(True, True, 16): [(8, 3, 1), (8, 4, 1), (8, 6, 1), (8, 8, 1), (16, 3, 1), (8, 3, 2)],
           (True, True, 32): [(4, 3, 1), (4, 4, 1), (4, 6, 1), (4, 8, 1), (8, 3, 1), (8, 6, 1)],
           (True, True, 1): [(8, 3, 1, 16), (16, 3, 1, 8), (8, 4, 1, 16), (8, 6, 1, 16), (8, 3, 1, 32), (16, 3, 2, 8)]}
_L17_ERR = {"uv4": 0.12, "ud3": 0.02}   # FP8 queries ~0.06-0.08 from FP32 here, FP16 ~5e-3; a bug is O(1)


def _l17_restore(cls):
    """Back to the class's measured v6 route (L16), or v5's path when it has none (UD3 on Sd16 / Sd32
    is a lab guess, not a measured route) or its kind was not built."""
    CUDA_ROUTE.pop(cls, None)
    prev = V6_ROUTES.get(cls)
    if prev is None or (prev[0] == "ud3" and cls[2] > 1):
        return
    if prev[0] in _KINDS and all(hasattr(_CUDA["mod"], f) for f in _KINDS[prev[0]][2]):
        CUDA_ROUTE[cls] = prev


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1717)
    if "mod" not in _CUDA:
        _log("L17: no CUDA module; timed path = v5")
        return
    kept = dict(CUDA_ROUTE)
    guesses = {c: r for c, r in kept.items() if r[0] == "uv4" or (r[0] == "ud3" and c[2] > 1)}
    for c in guesses:
        _l17_restore(c)
    planted = set(V7_ROUTES) | {c for c, r in V6_ROUTES.items() if r[0] == "ud3" and c[2] > 1}
    _log(f"L17: guesses kept at warm-up {sorted(guesses)}; dropped at warm-up {sorted(planted - set(guesses))}")
    _log(f"timed path until a class wins: {sorted(CUDA_ROUTE.items())}")
    uv4_ok = any(r[0] == "uv4" for r in guesses.values())
    ud3_ok = hasattr(_CUDA["mod"], "ud3")
    if uv4_ok:
        for s, mt, cs in sorted({(r[1], r[2], r[0]) for r in _L17_SV} | {(r[1], r[2], r[0]) for v in _L17_FV.values() for r in v}):
            a = _CUDA["mod"].uv4_attrs(s, mt, cs)
            _log(f"uv4 s{s} mt{mt} cs{cs} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    _log(f"L17 start: budget {_LAB_BUDGET_S:.0f}s; line = class route: us TB/s err (cache rows vs current path)")
    plan = []
    if uv4_ok:
        plan += [(c, [("uv4",) + r for r in _L17_FV[c]]) for c in ((True, True, 16), (True, True, 32))]
        plan += [(c, [("uv4",) + r for r in _L17_SV]) for c in ((False, True, 1), (False, True, 16), (False, True, 32))]
    if ud3_ok:
        plan += [(c, [("ud3",) + r for r in _L17_UD3]) for c in ((False, False, 16), (False, False, 32))]
    if uv4_ok:
        plan += [((True, True, 1), [("uv4",) + r for r in _L17_FV[(True, True, 1)]])]
    chosen, best_fv16 = {}, None
    for cls, cands in plan:
        full, verify, bclass = cls
        name = f"{'F' if full else 'S'}{'v' if verify else 'd'}{bclass}"
        length = 8 if verify else 1
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, length, 0 if full else 2048, k_cache, v_cache, gen)
            try:
                _kernel_fn_submitted(data)
                ref = _lab_reference(data)
                err0 = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
                if not err0 < 0.12:
                    raise RuntimeError(f"error {err0:.2e}")
            except Exception as exc:
                _log(f"{name}: current route {CUDA_ROUTE.get(cls)} failed ({_why(exc)}); base = v5 path")
                CUDA_ROUTE.pop(cls, None)
                _kernel_fn_submitted(data)
                ref = _lab_reference(data)
                err0 = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
            krow = data["k_cache"].view(-1, 2, 64)[data["cache_loc"]].float().clone()
            vrow = data["v_cache"].view(-1, 2, 64)[data["cache_loc"]].float().clone()
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq = data["cache_seqlens"] if full else data["cache_seqlens"].clamp(max=2048 + length)
            kv = float(seq.sum()) * 256
            _log(f"{name} current {CUDA_ROUTE.get(cls, 'v5')}: {base:.1f}us {kv / base / 1e6:.2f}TB/s err {err0:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for route in cands:
            tag = " ".join(str(x) for x in route)
            if lab.left() < 15:
                break
            try:
                if not _cuda_ok(data["config"], route):
                    raise RuntimeError("config not supported")
                err = float((_cuda_run(data, route).float() - ref).norm() / ref.norm())
                dk = float((data["k_cache"].view(-1, 2, 64)[data["cache_loc"]].float() - krow).abs().max())
                dv = float((data["v_cache"].view(-1, 2, 64)[data["cache_loc"]].float() - vrow).abs().max())
                us = lab.time_graph(lambda: _cuda_run(data, route))
                _log(f"{name} {tag}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (k {dk:.1e} v {dv:.1e})")
                if err < _L17_ERR[route[0]] and dk <= 0.5 and dv <= 0.5 and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} {tag}: FAILED {_why(exc)}")
        if cls == (True, True, 16) and best is not None:
            best_fv16 = (best[1], data, base)
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    if best_fv16 is not None and lab.left() > 10:
        route, data, base = best_fv16
        try:                     # L2-resident probe: same work, every row among 8192 (2 MB of K and V)
            hot = dict(data)
            hot["page_table"] = data["page_table"] % 8192
            hot["cache_loc"] = torch.gather(hot["page_table"].long(), 1, data["positions"].view(16, 8)).reshape(-1)
            us_hot = lab.time_graph(lambda: _cuda_run(hot, route))
            CUDA_ROUTE.pop((True, True, 16), None)
            base_hot = lab.time_graph(lambda: _kernel_fn_submitted(hot))
            if (True, True, 16) in chosen:
                CUDA_ROUTE[(True, True, 16)] = chosen[(True, True, 16)]
            _log(f"probe Fv16 L2-hot: {route} {us_hot:.1f}us, v5 path {base_hot:.1f}us (cold: v5 {base:.1f}us)")
        except Exception as exc:
            _log(f"probe FAILED {_why(exc)}")
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
