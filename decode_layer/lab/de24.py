"""DE24 lab (check runs only), on top of submission_v21 (= v20 + parts/dgrs.py + parts/v21.py: bf16_proj v11b's DG12-RS
for the pooler / unpooler at 32 tokens; serves nothing until a lab picks), after lab/de6.py, lab/de12.py, lab/de14.py and
lab/de16.py (de16_ab) are appended before it. V21_FORCE builds and checks both shapes in prepare.

dgrs     op: per built shape, one projection as a cold-L2 graph on the same x: v20's kernel (bf16_proj v6's swap for the
         pooler, cuBLAS for the unpooler) vs DG12-RS; a shape passes if >= 3% and >= 0.3 us faster (the hybrid-dispatch
         bar). layer: each non-empty set of passing shapes at decode 32 (sliding and full, weights 46 / 16, routing
         locked, cold-L2 graphs) vs v20; the fastest set is kept if >= 0.5% faster than v20 (sign-safe) and the layer
         outputs agree (< 2e-2), else none.
Modes are not re-picked (DE21-DE23: all att1 pj1). Prints V21_DGRS (make_v4.py --v21 --dgrs copies the last such line).
"""

import itertools

V21_FORCE = True
LAB_PREPARE_LIMIT_S = 265.0  # prepare's hard limit is 300 s; the section below takes ~1-2 s
DE24_OP_GAIN, DE24_OP_US, DE24_LAYER_BAR = 0.03, 0.3, 0.005


def sec_dgrs(lab, state, deadline):
    ready = dict(state.get("v21_ready") or {})
    if not ready:
        LAB_SUMMARY.append("dgrs: nothing built")
        log("dgrs: nothing built (see the [v21] line)")
        log("V21_DGRS = {}")
        return
    fixed = lab.empty_graph_us()
    gen = torch.Generator(device=state["dense"]["pooler"].device).manual_seed(2401)
    passing, cells = [], []
    for (layer, m), entry in sorted(ready.items()):
        w = state["dense"][layer]
        n, k = w.shape
        cfg = {"tokens": m, "window": 0}
        x = torch.randn((m, k), generator=gen, device=w.device).to(torch.bfloat16)
        state["v21_on"] = set()
        t_old = lab.time_graph(lambda: v4_project(x, layer, n, cfg, state)) - fixed
        state["v21_on"] = {(layer, m)}
        t_new = lab.time_graph(lambda: v4_project(x, layer, n, cfg, state)) - fixed
        err = _v3_rel(v4_project(x, layer, n, cfg, state), F.linear(x, w))
        state["v21_on"] = set()
        ok = err < V21_TOL and t_new < t_old * (1.0 - DE24_OP_GAIN) and t_old - t_new >= DE24_OP_US
        if ok:
            passing.append((layer, m))
        pct = 100 * (1 - t_new / t_old) if t_old > 0 else 0.0
        cells.append(f"{layer} m{m} {t_old:.2f} -> {t_new:.2f} ({pct:+.1f}%) err {err:.0e}"
                     f" -> {'pass' if ok else 'no'}")
    log("dgrs op (cold-L2 us, v20's kernel -> DG12-RS): " + "; ".join(cells))
    shapes = [(i, s) for i, s in enumerate(SCORED) if s[1] * s[2] == 32]
    best, lcells = None, []
    for size in range(len(passing), 0, -1):
        for subset in itertools.combinations(passing, size):
            if lab.elapsed() > deadline:
                LAB_SUMMARY.append("dgrs layer stopped early")
                break
            try:
                totals, err, wsum = de16_ab(lab, state, shapes, lambda on: state.__setitem__(
                    "v21_on", set(subset) if on else set()), 24000 + 7 * size)
            except Exception as exc:
                state["v21_on"] = set()
                lab_fail(f"dgrs layer {subset}", exc)
                continue
            gain = 1.0 - totals[True] / totals[False] if totals[False] > 0 else 0.0
            keep = err < 2e-2 and totals[True] < totals[False] - DE24_LAYER_BAR * abs(totals[False])
            lcells.append(f"{'+'.join(layer for layer, _m in subset)}: {totals[False] / wsum:.1f} -> "
                          f"{totals[True] / wsum:.1f} ({100 * gain:+.1f}%) err {err:.0e}{' ok' if keep else ''}")
            if keep and (best is None or gain > best[1]):
                best = (subset, gain)
    chosen = set(best[0]) if best else set()
    state["v21_on"] = chosen
    log("dgrs layer (weighted us at decode 32, routing locked, v20 -> with DG12-RS): " + ("; ".join(lcells) or "none"))
    LAB_SUMMARY.append(f"dgrs {sorted(chosen) or 'none'}")
    log(f"V21_DGRS = {dict(sorted((key, True) for key in chosen))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v21 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; att {'on' if state.get('v4_att') else 'OFF'}, DG12-RS built "
        f"{sorted(state.get('v21_ready') or {})}")
    lab.section("dgrs", sec_dgrs, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
