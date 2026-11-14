"""DE10 lab (check runs only), on top of submission_v8 (= v7 + parts/l10.py: pooler / unpooler staging), after
lab/de6.py (modes) and lab/de9.py (staging section) are appended before it: the same sections, with the
staging search extended by L10's options: pool (the pooler's first 4/16 or 8/16, staged during attention)
and unp (the full unpooler's first 4/16 or 8/16 above 8 tokens, staged during the experts); the modes' blunder
gate is 0.6 (DF30).
"""

DE6_GATE = 0.6  # DF30: the drop-ins pass their own segment checks in prepare; this only catches gross breakage
DE9_STEPS = (("o", (1,)), ("e13", (4, 8, 16)), ("e2", (8, 16)), ("slice", (1,)), ("pool", (4, 8)), ("unp", (4, 8)))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v8 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, "
        f"att {'on' if state.get('v4_att') else 'OFF'}, l9 {'on' if 'l9' in state else 'OFF'}")
    lab.section("modes", sec_modes, state, until=0.35 * lab.budget_s)
    lab.section("stage", sec_stage, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
