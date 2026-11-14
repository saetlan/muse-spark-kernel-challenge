"""DE6 lab (check runs only), on top of submission_v4. It CHOOSES v4's mode per case and installs it before the
judge times kernel_fn, so this check's timing is v4 as chosen.

modes   per scored shape (judge-like inputs, busiest-GPU local_start): the whole layer as a cold-L2 graph with
        the attention drop-in and the projections drop-in each off/on (4 variants); errors of the attention
        drop-in vs SGLang's block and of the full layer (all on) vs all off. A case keeps the fastest variant if it
        is >= 2% faster than all off and its full-layer difference stays < 0.3 (blunder gate); prints V4_MODE.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []
DE6_VARIANTS = ((False, False), (True, False), (False, True), (True, True))
# Blunder gate on the whole layer's output vs all off (wiring errors are O(1)). Routing flips at near-ties,
# which the judge accepts, also reach O(0.3) there (DF30); later labs raise it (the self-checks in prepare
# already hold each drop-in to its own segment's reference).
DE6_GATE = 0.3


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def sec_modes(lab, state, deadline):
    fixed = lab.empty_graph_us()
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        key = (batch * q_len, layer == "sliding")
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"modes stopped before {name}")
            state["v4_mode"][key] = {"att": False, "pj": False}
            continue
        try:
            data = lab_input(batch, q_len, layer, seed=13000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            outs, times = {}, {}
            for att, pj in DE6_VARIANTS:
                state["v4_mode"][key] = {"att": att, "pj": pj}
                kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                outs[(att, pj)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)
                del kc, vc
                times[(att, pj)] = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            base = outs[(False, False)]
            errs = {v: _v3_rel(outs[v][0], base[0]) for v in DE6_VARIANTS}  # next layer's input, vs all off
            routed = {v: float((outs[v][2] == base[2]).float().mean()) for v in DE6_VARIANTS}
            # Blunder gate only (wiring errors are O(1)); the judge's tests vs FP32 are the real check. FA3's FP8
            # queries alone sit ~8% off FP32 in attention, and a flipped near-tie expert changes a token's FFN.
            ok = [v for v in DE6_VARIANTS if errs[v] < DE6_GATE]
            best = min(ok, key=lambda v: times[v])
            pick = best if times[best] < 0.98 * times[(False, False)] else (False, False)
            state["v4_mode"][key] = {"att": pick[0], "pj": pick[1]}
            cells.append(f"{name} " + "/".join(f"{times[v]:.1f}" for v in DE6_VARIANTS)
                         + f" err {'/'.join(f'{errs[v]:.0e}' for v in DE6_VARIANTS[1:])}"
                         + f" ids {min(routed.values()):.2f} -> att{int(pick[0])} pj{int(pick[1])}")
            LAB_SUMMARY.append(f"{name} {times[(False, False)]:.1f}->{times[pick]:.1f}")
            del data, outs
        except Exception as exc:
            lab_fail(f"modes {name}", exc)
            state["v4_mode"][key] = {"att": False, "pj": False}
        torch.cuda.empty_cache()
    log("modes (us off/att/pj/both; err att/pj/both vs off; routing agreement): " + "; ".join(cells))
    log(f"V4_MODE = {dict(sorted(state['v4_mode'].items()))!r}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v4 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, "
        f"att {'on' if state.get('v4_att') else 'OFF'}")
    lab.section("modes", sec_modes, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
