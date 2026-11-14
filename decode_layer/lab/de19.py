"""DE19 lab (check runs only), on top of the clean submission_v16 (DE18's picks: M19 at 65-191 tokens; G2P off), after
lab/de6.py, lab/de12.py, lab/de14.py, lab/de16.py and lab/de18.py (its sec_m19) are appended before it.

m19      DE18's section over V16_CLASSES: 17-32 tokens (new: MoE E20 puts its bs32 class on the same two M19 instances
         we build, in-tuner 167.7 vs M11's 183.7 us; here the hook on m11_run serves it, prepare checks M19 vs M11 at 32
         tokens), 65-191 (re-confirms DE18), 192-320 (re-tests); routing-locked whole-layer A/B, >= 3%, < 2e-2.
G2P stays off and is not built (V14_FORCE False). Prints V16_M19.
"""

V14_FORCE = False


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v16 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; M19 path {'built' if 'v16_moe' in state else 'OFF'}, file's picks {state.get('v16_m19')}")
    lab.section("m19", sec_m19, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
