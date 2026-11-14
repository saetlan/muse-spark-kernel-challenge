"""DE23 lab (check runs only), on top of submission_v20 (= v18 with the attention session's scored v14 in place of v9:
W3b on full verify 16 / 32, UD4 on sliding decode 1 / 16 / 32; DE20's E25 choice, DE21's modes), after lab/de6.py,
lab/de12.py, lab/de14.py (routing lock, sec_modes_locked) and lab/de16.py (de16_ab) are appended before it.

modes    de14's routing-locked per-case choice (attention / projection drop-ins on or off) over all 12 scored shapes:
         v14 changes the attention kernels at sliding decode 1 / 16 / 32 (UD4) and full verify 16 / 32 (W3b); a case
         keeps the fastest variant if >= 2% faster than all off; a case not reached keeps the file's picks.
m19_256  192-320 tokens (verify 32, sliding and full; weights 46 / 16): the file's Marlin vs E25's two M19 instances
         (the 65-191 class's path; prepare already checks M19 vs Marlin at 256 tokens), routing-locked whole-layer A/B,
         a third measurement after DE18 (-2.3%) and DE19 (-2.0%); installed only if >= 3% faster (the hybrid-dispatch
         bar) and the outputs agree (< 2e-2).
Prints V4_MODE and V17_CHOICE (make_v4.py --v20 --modes / --choice copy the last such lines).
"""

DE23_GAIN = 0.03
DE23_BOUND = 321  # V17_CLASSES: 192-320 tokens


def sec_m19_256(lab, state, deadline):
    choice = dict(state.get("v17_choice") or {})
    if "v17_moe" not in state:
        LAB_SUMMARY.append("m19_256: E25 path OFF")
        log("m19_256: the E25 path is off (see the [v17] line); the file's V17_CHOICE is printed unchanged")
        log(f"V17_CHOICE = {dict(sorted(V17_CHOICE.items()))!r}")
        return
    low = dict(V17_CLASSES)[DE23_BOUND]
    shapes = [(i, s) for i, s in enumerate(SCORED) if low <= s[1] * s[2] < DE23_BOUND]
    try:
        totals, err, w = de16_ab(lab, state, shapes, lambda on: state.__setitem__(
            "v17_choice", {**choice, DE23_BOUND: "m19"} if on else dict(choice)), 23000 + DE23_BOUND)
    except Exception as exc:
        state["v17_choice"] = dict(choice)
        lab_fail("m19_256", exc)
        log(f"V17_CHOICE = {dict(sorted(choice.items()))!r}")
        return
    keep = err < 2e-2 and totals[True] < totals[False] * (1.0 - DE23_GAIN)
    if keep:
        choice[DE23_BOUND] = "m19"
    state["v17_choice"] = dict(choice)
    log(f"m19_256 (weighted us over {len(shapes)} shapes, routing locked): Marlin {totals[False] / w:.1f} -> M19 "
        f"{totals[True] / w:.1f} ({100 * (1 - totals[True] / totals[False]):+.1f}%) err {err:.0e} -> "
        f"{'M19' if keep else 'Marlin'}")
    LAB_SUMMARY.append(f"m19_256 {'M19' if keep else 'Marlin'} {100 * (1 - totals[True] / totals[False]):+.1f}%")
    log(f"V17_CHOICE = {dict(sorted(choice.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v20 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; att {'on' if state.get('v4_att') else 'OFF'}, E25 choice {state.get('v17_choice')}")
    lab.section("modes", sec_modes_locked, state, min_left=3.0)
    lab.section("m19_256", sec_m19_256, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
