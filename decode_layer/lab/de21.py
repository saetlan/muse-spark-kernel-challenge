"""DE21 lab (check runs only), on top of submission_v18 (= v17 with DE20's picks + the attention session's scored v9:
W1-N wgmma FP8 verify on sliding verify 16 / 32 and full verify 1 / 16 / 32), after lab/de6.py, lab/de12.py,
lab/de14.py (routing lock, sec_modes_locked) and lab/de16.py are appended before it.

modes    de14's routing-locked per-case choice (attention / projection drop-ins on or off) over all 12 scored shapes:
         with attention v9 the verify cases change kernels (W1-N instead of FA3T / V5 / the PDL chain), and verify 32
         (both layer kinds) ran SGLang's attention so far; a case keeps the fastest variant if >= 2% faster than all
         off; a case not reached keeps the file's picks.
Prints V4_MODE (make_v4.py --v18 --modes copies the last such line).
"""


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v18 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; att {'on' if state.get('v4_att') else 'OFF'}, E25 choice {state.get('v17_choice')}")
    lab.section("modes", sec_modes_locked, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
