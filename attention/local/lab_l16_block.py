

# ---------------------------------------------------------------------------------------------
# Lab L16 (check runs only): V4 (cuda_ver4.py: S2c's producer, FP8 Q.K and P.V consumers) on the
# six verify classes, UD3 (cuda_ud3.py: one launch per sliding decode layer, AF32) on the three
# sliding decode classes, and D3 with a whole-wave split (AF33) on the three full decode classes,
# against v5's path for each class (PDL chain on sliding verify and Fd1, Triton on Fv1, SGLang
# on Fv16/32, D3 c6 on Fd16/32, Triton/PDL on sliding decode). The base file is v6 with a V4
# guess on every verify class and a UD3 guess on every sliding decode class
# (built and self-checked at warm-up); the lab first removes those routes (the timed path is
# v5's if anything below fails), times the candidates with the error vs the FP32 reference on
# synthetic inputs, and routes a class to its best candidate only when it is correct and >= 3%
# faster than v5's path: the judge then times v6 with the measured routes.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L16_VARIANTS = [(3, 2, 1), (3, 2, 3), (4, 2, 1), (3, 1, 1), (4, 1, 1)]
_L16_CPS = {(True, True, 1): (1, 2, 3), (True, True, 16): (2, 3, 4, 6), (True, True, 32): (2, 4, 6, 8),
            (False, True, 1): (1,), (False, True, 16): (1, 2, 4), (False, True, 32): (1, 2, 4)}
_L16_DEC = {(True, False, 1): [(2, 4, 3), (3, 4, 3), (2, 4, 4), (3, 4, 4)],
            (True, False, 16): [(3, 4, 3), (6, 4, 3), (9, 4, 3), (6, 4, 4)],
            (True, False, 32): [(3, 4, 3), (6, 4, 3), (9, 4, 3), (6, 4, 4)]}
_L16_ERR_MAX = {"ver4": 0.12, "dec4": 0.02, "ud3": 0.02}   # FP8 queries ~0.06 from FP32 here; FP16 ~2e-3; a bug is O(1)


def ns_of(route, data, verify):
    if route[0] == "ud3":
        return route[1]
    split_of = _nsplit_w if route[0] == "dec4" else _nsplit
    return split_of(data["config"], data["page_table"].shape[1], route[1], 64 if verify else 32 * route[2])


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1616)
    init = {c: r for c, r in CUDA_ROUTE.items() if r[0] in ("ver4", "ud3")}
    for c in init:
        CUDA_ROUTE.pop(c)
    _log(f"L16: ver4 routes after warm-up: {sorted(init)} (dropped at warm-up: "
         f"{sorted(set(V6_ROUTES) - set(init))}); timed path = v5 until a class wins")
    if "mod" not in _CUDA:
        return
    for s, mt, mb in _L16_VARIANTS if init else ():
        a = _CUDA["mod"].attend_ver4_attrs(s, mt, mb)
        _log(f"ver4 s{s} mt{mt} minb{mb} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    _log(f"L16 start: budget {_LAB_BUDGET_S:.0f}s; line = class variant: us err (nsplit)")
    chosen = {}
    plan = [(cls, [("ver4", cps, s, mt, mb) for cps in _L16_CPS[cls] for s, mt, mb in _L16_VARIANTS])
            for cls in sorted(init, key=lambda c: (c[0], c[2])) if init[cls][0] == "ver4"]
    plan += [(cls, [("ud3", cs, w, s) for cs in (8, 16) for w, s in ((4, 3), (8, 2))])
             for cls in sorted(init, key=lambda c: c[2]) if init[cls][0] == "ud3"]
    plan += [(cls, [("dec4", cps, w, s, "w") for cps, w, s in cands]) for cls, cands in sorted(_L16_DEC.items())]
    for cls, cands in plan:
        full, verify, bclass = cls
        name = f"{'F' if full else 'S'}{'v' if verify else 'd'}{bclass}"
        length = 8 if verify else 1
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, length, 0 if full else 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            kv = float((data["cache_seqlens"].clamp(max=2048 + length) if not full else data["cache_seqlens"]).sum()) * 256
            _log(f"{name} v5 path {CUDA_ROUTE.get(cls, 'triton/pdl/sglang')}: {base:.1f}us {kv / base / 1e6:.2f}TB/s")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for route in cands:
            tag = " ".join(str(x) for x in route[1:])
            try:
                ns = ns_of(route, data, verify)
                err = float((_cuda_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _cuda_run(data, route))
                _log(f"{name} {route[0]} {tag}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
                if err < _L16_ERR_MAX[route[0]] and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} {route[0]} {tag}: FAILED {_why(exc)}")
        if best is not None and best[1][0] in ("ver4", "dec4") and ns_of(best[1], data, verify) > 64:
            global MERGE_CB
            try:                                     # the wide merge (AF35) against v5's 64-wide one
                MERGE_CB = 64
                us64 = lab.time_graph(lambda: _cuda_run(data, best[1]))
            finally:
                MERGE_CB = 256
            _log(f"{name} merge: cb256/8 warps {best[0]:.1f}us vs cb64/4 warps {us64:.1f}us")
            if us64 < best[0]:
                best = (us64, best[1])
                _log(f"{name}: the 64-wide merge was faster (MERGE_CB stays 256 for the other classes)")
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs v5 path {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP v5 path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    _log(f"V6_ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
