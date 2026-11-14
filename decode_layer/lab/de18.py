"""DE18 lab (check runs only), on top of submission_v16 (= v15 with DE17's picks + parts/moe19.py + parts/v16.py: MoE's
scored E19 M19 path at 65-320 tokens, off until a lab installs it), after lab/de6.py, lab/de12.py, lab/de14.py and
lab/de16.py are appended before it. V16_FORCE makes prepare build and check the M19 path with no picks; with nothing
installed the timed kernel_fn is the base file's (Marlin at verify 16 / 32).

m19      per E19 class (65-191 tokens: verify 16; 192-320: verify 32), the whole layer at the class's scored shapes
         (sliding and full, weights 46 / 16, busiest-GPU local_start, cold-L2 graphs, routing LOCKED so both variants
         run the same experts): Marlin vs M19; installed if >= 3% faster and the outputs agree (< 2e-2: different
         kernels; E19 sits 3.4e-3 from FP32 on its own problem, Marlin about the same).
g2p      the prefill session's sec_gemm2_post (shared/glue/lab_g2p.py, appended before this file): attrs, a correctness
         gate (fused vs GEMM2 + post-norm at 1-32 tokens, counters clean), then per M11 class (5 / 9 / 17 / 33) a
         routing-locked whole-layer A/B; fused where >= 0.3% faster and < 1e-2.
Prints V16_M19 and V14_FUSED (make_v4.py --v16 --m19 / --fused copy the last such lines).
"""

V16_FORCE = True
V14_FORCE = True
DE18_GAIN = 0.03
DE18_CLASSES = V16_CLASSES  # (class bound, lowest token count): 17-32 (E20), 65-191, 192-320


def sec_m19(lab, state, deadline):
    if "v16_moe" not in state:
        LAB_SUMMARY.append("m19: path OFF")
        log("m19: the M19 path is off (see the [v16] line); V16_M19 stays {}")
        log("V16_M19 = {}")
        return
    picks, cells = {}, []
    for below, low in DE18_CLASSES:
        shapes = [(i, s) for i, s in enumerate(SCORED) if low <= s[1] * s[2] < below]
        if not shapes:
            continue
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"m19 stopped before class <{below}")
            break
        try:
            totals, err, w = de16_ab(lab, state, shapes,
                                     lambda on: state.__setitem__("v16_m19", {**picks, **({below: True} if on else {})}),
                                     22000 + below)
        except Exception as exc:
            state["v16_m19"] = dict(picks)
            lab_fail(f"m19 <{below}", exc)
            continue
        keep = err < 2e-2 and totals[True] < totals[False] * (1.0 - DE18_GAIN)
        if keep:
            picks[below] = True
        state["v16_m19"] = dict(picks)
        cells.append(f"<{below}: Marlin {totals[False] / w:.1f} -> M19 {totals[True] / w:.1f} err {err:.0e} -> "
                     f"{'M19' if keep else 'Marlin'}")
        LAB_SUMMARY.append(f"m19 <{below} {'on' if keep else 'off'}")
        torch.cuda.empty_cache()
    state["v16_m19"] = dict(picks)
    log("m19 (weighted us, routing locked): " + "; ".join(cells))
    log(f"V16_M19 = {dict(sorted(picks.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v16 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; M19 path {'built' if 'v16_moe' in state else 'OFF'}, G2P {'on' if state.get('v14_ok') else 'OFF'}")
    lab.section("m19", sec_m19, state, until=0.5 * lab.budget_s, min_left=3.0)
    if state.get("v14_ok"):
        lab.section("g2p", sec_gemm2_post, state, min_left=3.0)
    else:
        log("g2p: the fused path is off (see the [v14] line)")
        log("V14_FUSED = {}")
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
