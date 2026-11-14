

# ---------------------------------------------------------------------------------------------
# Lab L18 (check runs only), on v8-lab = v7 (L16 + L17 routes) + V5 guesses on Fv16/32 and Sv16/32
# (built and self-checked at warm-up; the lab removes them first, so the timed path is v7's if
# anything below fails). Candidates, each timed against the class's current path with its error vs
# the FP32 reference; a class moves only when correct and >= 3% faster:
#   1. V5 (cuda_ver5.py: 64-key softmax rounds, edge-only masks, fused scale, optional 112-register
#      cap for 2 CTAs/SM) on every verify class (L17: verify is consumer-bound, AF37);
#   2. fa3t (SGLang's FA3 kernel behind one Triton prologue, FA3's num_splits / pack_gqa) on Fv16/32;
#   3. probe: the best V5 route on Fv16 with an L2-resident cache (consumer- vs memory-bound).
# Prints ROUTES = {...} (ver5 and fa3t entries) for the scored file.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L18_SM = [(3, 2), (4, 2), (3, 1), (4, 1)]                    # (stages, min blocks per SM)
_L18_CPS = {(True, True, 16): (2, 4, 6), (True, True, 32): (2, 4, 8), (True, True, 1): (1, 2),
            (False, True, 16): (1, 2), (False, True, 32): (1, 2), (False, True, 1): (1,)}
_L18_FA3 = [(0, None), (2, True), (4, True), (8, True), (16, True)]
_L18_ERR = 0.12


def _l18_run(data, route):
    return run_fa3t(data, *route[1:]) if route[0] == "fa3t" else _cuda_run(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1818)
    if "mod" not in _CUDA:
        _log("L18: no CUDA module; timed path = v5")
        return
    guesses = {c: r for c, r in CUDA_ROUTE.items() if r[0] == "ver5"}
    for c in guesses:
        CUDA_ROUTE.pop(c)
    _log(f"L18: ver5 guesses kept at warm-up {sorted(guesses)}; dropped {sorted(set(V8_ROUTES) - set(guesses))}")
    _log(f"timed path until a class wins: {sorted(CUDA_ROUTE.items())}")
    ver5_ok = bool(guesses)
    if ver5_ok:
        for s, mb in _L18_SM:
            a = _CUDA["mod"].attend_ver5_attrs(s, mb)
            _log(f"ver5 s{s} minb{mb} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    _log(f"L18 start: budget {_LAB_BUDGET_S:.0f}s; line = class route: us TB/s err")
    order = [(True, True, 16), (True, True, 32), (False, True, 16), (False, True, 32), (True, True, 1), (False, True, 1)]
    chosen, best_fv16 = {}, None
    for cls in order:
        full, _, bclass = cls
        name = f"{'F' if full else 'S'}v{bclass}"
        cands = [("ver5", cps, s, mb) for cps in _L18_CPS[cls] for s, mb in _L18_SM] if ver5_ok else []
        if full and bclass > 1:
            cands += [("fa3t",) + r for r in _L18_FA3]
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, 0 if full else 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            err0 = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq = data["cache_seqlens"] if full else data["cache_seqlens"].clamp(max=2048 + 8)
            kv = float(seq.sum()) * 256
            cur = CUDA_ROUTE.get(cls) or (("fa3t",) + FA3T_ROUTE[cls] if cls in FA3T_ROUTE else "v5")
            _log(f"{name} current {cur}: {base:.1f}us {kv / base / 1e6:.2f}TB/s err {err0:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for route in cands:
            tag = " ".join(str(x) for x in route)
            if lab.left() < 15:
                break
            try:
                err = float((_l18_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _l18_run(data, route))
                ns = _nsplit(data["config"], data["page_table"].shape[1], route[1], 64) if route[0] == "ver5" else "-"
                _log(f"{name} {tag}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
                if err < _L18_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} {tag}: FAILED {_why(exc)}")
        if cls == (True, True, 16):
            v5 = [c for c in cands if c[0] == "ver5"]
            best_fv16 = (best[1] if best and best[1][0] == "ver5" else (v5[0] if v5 else None), data, base)
        if best is not None and best[0] < 0.97 * base:
            if best[1][0] == "fa3t":
                CUDA_ROUTE.pop(cls, None)
                FA3T_ROUTE[cls] = best[1][1:]
            else:
                CUDA_ROUTE[cls] = best[1]
                FA3T_ROUTE.pop(cls, None)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    if best_fv16 is not None and best_fv16[0] is not None and lab.left() > 10:
        route, data, base = best_fv16
        try:                     # L2-resident probe: same work, every row among 8192 (2 MB of K and V)
            hot = dict(data)
            hot["page_table"] = data["page_table"] % 8192
            hot["cache_loc"] = torch.gather(hot["page_table"].long(), 1, data["positions"].view(16, 8)).reshape(-1)
            us_cold = lab.time_graph(lambda: _cuda_run(data, route))
            us_hot = lab.time_graph(lambda: _cuda_run(hot, route))
            _log(f"probe Fv16 L2-hot: {route} hot {us_hot:.1f}us cold {us_cold:.1f}us")
        except Exception as exc:
            _log(f"probe FAILED {_why(exc)}")
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
