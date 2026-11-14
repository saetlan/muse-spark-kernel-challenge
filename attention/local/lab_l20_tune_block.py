

# ---------------------------------------------------------------------------------------------
# Lab section "tune" (check runs only): re-sweep the Triton PDL chain's launch settings on the
# sliding verify shapes still on v5's path (Sv16 1.32x, Sv32 0.98x on the judge in L18). TUNE dates
# from L1/L2, before the PDL-late chain (AF35: the merge is 2-9 us of every chain; Sv32's
# _combine_p takes 7 us). Candidates (target programs per SM, min keys per slice, keys per step,
# warps, stages) run through run_pdl(data, tune=...); a class takes the best one when correct and
# >= 3% faster than its TUNE entry, and the lab prints TUNE_UPDATE = {...} for the scored file.
# SGLang's own path is timed too (the judge has Sv32 at 35.07 us ours vs 34.54 us SGLang): if it beats
# both the current and the best tuned setting by >= 2%, the class leaves PDL_ROUTE (falls back to
# SGLang, as v2.2 did) and the lab prints SGLANG_FALLBACK = {...}.
# ---------------------------------------------------------------------------------------------
_L20_TUNE_CLASSES = [(False, True, 32), (False, True, 16)]
TUNE_BASE = dict(TUNE)
_L20_TUNE = [(16, mk, blk, w, st) for mk in (64, 128, 256, 512) for blk in (32, 64) for w in (4, 8) for st in (1, 2, 3)]


def _l20_tune(lab, k_cache, v_cache, gen):
    updates, fallback = {}, set()
    for cls in _L20_TUNE_CLASSES:
        if CUDA_ROUTE.get(cls) is not None or FA3T_ROUTE.get(cls) is not None:
            _log(f"tune: {cls} is routed elsewhere; skipped")
            continue
        full, _, bclass = cls
        name = f"Sv{bclass}"
        if lab.left() < 40:
            _log("tune: out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: run_pdl(data))
            _log(f"tune {name} TUNE {TUNE[cls]} nsplit {plan(data['config'], data['page_table'].shape[1])[0]}: {base:.1f}us")
        except Exception as exc:
            _log(f"tune {name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for t in _L20_TUNE:
            if t == TUNE[cls]:
                continue
            if lab.left() < 20:
                _log("tune: out of budget")
                break
            try:
                err = float((run_pdl(data, tune=t).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: run_pdl(data, tune=t))
                if err < 2e-2 and (best is None or us < best[0]):
                    best = (us, t)
                _log(f"tune {name} {t}: {us:.1f}us err {err:.1e}")
            except Exception as exc:
                _log(f"tune {name} {t}: FAILED {_why(exc)}")
        if best is not None and best[0] < 0.97 * base:
            TUNE[cls] = best[1]
            updates[cls] = best[1]
            _log(f"TUNE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"tune: KEEP {cls} {TUNE[cls]}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
        try:
            mine = min(base, best[0]) if best is not None else base
            err = float((sglang_path(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: sglang_path(data))
            _log(f"tune {name} SGLang path: {us:.1f}us err {err:.1e} (ours best {mine:.1f}us)")
            if err < 0.12 and us < 0.98 * mine and cls in PDL_ROUTE and cls not in OURS:
                PDL_ROUTE.discard(cls)
                TUNE[cls] = TUNE_BASE[cls] if cls in TUNE_BASE else TUNE[cls]
                updates.pop(cls, None)
                fallback.add(cls)
                _log(f"SGLANG {cls}: {us:.1f}us vs ours {mine:.1f}us ({mine / us:.2f}x)")
        except Exception as exc:
            _log(f"tune {name} SGLang path: FAILED {_why(exc)}")
    _log(f"TUNE_UPDATE = {updates!r}")
    _log(f"SGLANG_FALLBACK = {sorted(fallback)!r}")
