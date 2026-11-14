"""DE9 lab (check runs only), on top of submission_v7 (= v6 + parts/l9.py, L2 staging in HBM-idle windows),
after lab/de6.py's sections (appended before this file): it CHOOSES v7's attention / projection modes per case
(de6's sec_modes), then L9's staging per case, and installs both before the judge times kernel_fn.

stage   per scored shape (judge-like inputs, busiest-GPU local_start), the whole layer as a cold-L2 graph,
        greedy per prefetch form pm (0 bulk, 1 lines): o, then e13 in (4, 8, 16), then e2 in (8, 16), then
        the unpooler slice (<= 8 tokens); each option's best level is kept if the layer gets >= 1% faster; the
        case keeps its best only if >= 2% faster than no staging, and only if the layer's outputs equal
        no staging's bit for bit (staging is hints only). Prints L9_MODE.
"""

DE9_STEPS = (("o", (1,)), ("e13", (4, 8, 16)), ("e2", (8, 16)), ("slice", (1,)))


def sec_stage(lab, state, deadline):
    if "l9" not in state:
        LAB_SUMMARY.append("stage: L9 off")
        return
    fixed = lab.empty_graph_us()
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        key = (batch * q_len, layer == "sliding")
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"stage stopped before {name}")
            state["l9_mode"][key] = {}
            continue
        try:
            data = lab_input(batch, q_len, layer, seed=21000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))

            def timed(mode):
                state["l9_mode"][key] = mode
                return lab.time_graph(lambda: kernel_fn(data, state)) - fixed

            base = timed({})
            best_mode, best = {}, base
            for pm in (0, 1):
                cur, cur_t = {"pm": pm}, base
                for opt, levels in DE9_STEPS:
                    if opt == "slice" and batch * q_len > SLICED_UNPOOL_MAX_TOKENS:
                        continue
                    tries = [(timed(dict(cur, **{opt: v})), v) for v in levels]
                    t, v = min(tries)
                    if t < cur_t - 0.01 * abs(cur_t):
                        cur, cur_t = dict(cur, **{opt: v}), t
                if cur_t < best:
                    best_mode, best = cur, cur_t
            pick = best_mode if best < base - 0.02 * abs(base) else {}
            if pick:  # staging changes no data: the layer's outputs must match no staging's exactly
                outs = []
                for mode in ({}, pick):
                    state["l9_mode"][key] = mode
                    kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                    outs.append(kernel_fn(dict(data, k_cache=kc, v_cache=vc), state))
                    del kc, vc
                same = all(torch.equal(a, b) for a, b in zip(*outs))
                if not same:
                    LAB_SUMMARY.append(f"{name}: staged outputs differ, dropped")
                    pick = {}
            state["l9_mode"][key] = pick
            cells.append(f"{name} {base:.1f}->{best:.1f} {pick if pick else '-'}")
            LAB_SUMMARY.append(f"{name} {base:.1f}->{(best if pick else base):.1f}")
            del data
        except Exception as exc:
            lab_fail(f"stage {name}", exc)
            state["l9_mode"][key] = {}
        torch.cuda.empty_cache()
    log("stage (us none -> best; pick): " + "; ".join(cells))
    log(f"L9_MODE = {dict(sorted((k, v) for k, v in state['l9_mode'].items() if v))!r}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v7 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, "
        f"att {'on' if state.get('v4_att') else 'OFF'}, l9 {'on' if 'l9' in state else 'OFF'}")
    lab.section("modes", sec_modes, state, until=0.4 * lab.budget_s)
    lab.section("stage", sec_stage, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
