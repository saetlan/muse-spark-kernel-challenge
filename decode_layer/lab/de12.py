"""DE12 lab (check runs only), on top of submission_v10 (= v9: v7 without L9, with the attention session's v6 and
the bf16_proj session's v6 (12-bit lossless dense weights); + parts/m11_pdl.py / parts/v10.py, the MoE session's
PDL path, off until a lab installs it), after lab/de6.py (modes) is appended before it. With nothing installed the
timed kernel_fn is v9's.

modes   de6's section with v9's drop-ins: per scored shape, the whole layer as a cold-L2 graph with the attention
        and projection drop-ins each off/on; a case keeps the fastest variant if >= 2% faster than all off, its
        layer output within the blunder gate (0.6, DF30). A case the section didn't reach in its time keeps the
        file's V4_MODE (DE9's picks, judged in v7) instead of all off, so a short budget can't cost a case.
        Prints V4_MODE (the last such line is what make_v4.py --modes copies) and v9's prepare stage times.
pdl     (v10, run first: cheap) per M11 class with an M11P instance (1-4 and 5-8 tokens): the whole layer at its
        scored shapes (sliding and full, weighted 46 / 16, busiest-GPU local_start, cold-L2 graphs) as v9, with
        the CUDA post-norm PDL-launched, and with GEMM2 PDL-launched too; the best option is installed if the
        weighted time is >= 1.5% lower and the layer's outputs stay within 2e-2 of v9's path (v10_setup already
        checked every option against the plain path); prints V10_PDL (make_v4.py --pdl copies it).
"""

DE12_PDL_OPTS = ({"s2": 0, "post": 1}, {"s2": 1, "post": 1})
PDL_BAR = 0.015  # DE12: the 1.5% rule (DE13 lowers it: lab/de13.py)

DE6_GATE = 0.6  # DF30: the drop-ins pass their own segment checks in prepare; this only catches gross breakage


def sec_modes_keep(lab, state, deadline):
    keep = {key: dict(val) for key, val in state["v4_mode"].items()}
    n_summary = len(LAB_SUMMARY)
    sec_modes(lab, state, deadline)
    names = {name: (batch * q_len, layer == "sliding") for name, batch, q_len, layer, _w in SCORED}
    kept = []
    for entry in LAB_SUMMARY[n_summary:]:
        if entry.startswith("modes stopped before "):
            name = entry[len("modes stopped before "):]
            key = names[name]
            state["v4_mode"][key] = dict(keep.get(key, {"att": False, "pj": False}))
            kept.append(name)
    if kept:
        log(f"modes not reached, file's picks kept: {', '.join(kept)}")
    log(f"V4_MODE = {dict(sorted(state['v4_mode'].items()))!r}")


def sec_pdl(lab, state, deadline):
    if not state.get("v10_ok"):
        LAB_SUMMARY.append("pdl: v10 off")
        return
    fixed = lab.empty_graph_us()
    picks, cells = {}, []
    for below in V10_PDL_CLASSES:
        low = max([b for b in V10_PDL_CLASSES if b < below], default=1)  # the class's own token range
        shapes = [(i, s) for i, s in enumerate(SCORED) if low <= s[1] * s[2] < below]
        if not shapes or dict(state["m11_configs"]).get(below) is None:
            continue
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"pdl stopped before class <{below}")
            break
        try:
            totals, outs = {}, {}
            for j in (-1,) + tuple(range(len(DE12_PDL_OPTS))):  # -1: v9's path (no PDL)
                v10_install(state, {} if j < 0 else {below: DE12_PDL_OPTS[j]})
                tot = 0.0
                for i, (name, batch, q_len, layer, weight) in shapes:
                    data = lab_input(batch, q_len, layer, seed=17000 + i)
                    _, _, ids = kernel_fn(data, state)
                    data["local_start"].fill_(_busiest_start(ids))
                    kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                    outs[(j, i)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)[0]
                    del kc, vc
                    tot += weight * (lab.time_graph(lambda: kernel_fn(data, state)) - fixed)
                    del data
                totals[j] = tot
            errs = {j: max(_v3_rel(outs[(j, i)], outs[(-1, i)]) for i, _s in shapes) for j in range(len(DE12_PDL_OPTS))}
            ok = [j for j in errs if errs[j] < 2e-2]
            best = min(ok, key=lambda j: totals[j]) if ok else None
            if best is not None and totals[best] < totals[-1] - PDL_BAR * abs(totals[-1]):  # sign-safe bar
                picks[below] = dict(DE12_PDL_OPTS[best])
            cells.append(f"<{below} " + "/".join(f"{totals[j] / sum(s[4] for _i, s in shapes):.1f}" for j in (-1, 0, 1))
                         + f" err {'/'.join(f'{errs[j]:.0e}' for j in (0, 1))} -> {picks.get(below, '-')}")
            LAB_SUMMARY.append(f"pdl <{below} {'on' if below in picks else 'off'}")
        except Exception as exc:
            lab_fail(f"pdl <{below}", exc)
        torch.cuda.empty_cache()
    v10_install(state, picks)
    for below in picks:
        m11_warm(state, dict(state["m11_configs"])[below])
    state["v10_pdl"] = picks
    log("pdl (weighted us v9 / post PDL / GEMM2 + post PDL; err vs v9): " + "; ".join(cells))
    log(f"V10_PDL = {dict(sorted(picks.items()))!r}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v10 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, att {'on' if state.get('v4_att') else 'OFF'}")
    lab.section("pdl", sec_pdl, state, until=0.3 * lab.budget_s)
    lab.section("modes", sec_modes_keep, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
