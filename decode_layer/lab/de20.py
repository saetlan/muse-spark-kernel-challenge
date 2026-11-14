"""DE20 lab (check runs only), on top of submission_v17 (= v15 + parts/moe25.py + parts/v17.py: MoE's scored E25 path,
with V17_CHOICE {33: "m19", 192: "m19"} = DE19's picks on E25's M19), after lab/de6.py, lab/de12.py, lab/de14.py and
lab/de16.py are appended before it. V17_FORCE makes prepare build and check the E25 path whatever the choice.

e25      per small class (1-4 / 5-8 / 9-16 / 17-32 tokens), the whole layer at the class's scored shapes (sliding and
         full, weights 46 / 16, busiest-GPU local_start, cold-L2 graphs, routing LOCKED): the file's path (our M11 with
         its PDL GEMM2 + CUDA post-norm; M19 at 17-32) vs E25's M11b pick for the class; M11b installed if >= 2% faster
         and the outputs agree (< 2e-2).
Prints V17_CHOICE (make_v4.py --v17 --choice copies the last such line).
"""

V17_FORCE = True
DE20_GAIN = 0.02
DE20_TESTS = ((5, "m11b"), (9, "m11b"), (17, "m11b"), (33, "m11b"))


def sec_e25(lab, state, deadline):
    choice = dict(state.get("v17_choice") or {})
    if "v17_moe" not in state:
        LAB_SUMMARY.append("e25: path OFF")
        log("e25: the E25 path is off (see the [v17] line)")
        log(f"V17_CHOICE = {dict(sorted(V17_CHOICE.items()))!r}")
        return
    lows, cells = dict(V17_CLASSES), []
    for below, kind in DE20_TESTS:
        shapes = [(i, s) for i, s in enumerate(SCORED) if lows[below] <= s[1] * s[2] < below]
        if not shapes:
            continue
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"e25 stopped before class <{below}")
            break
        before = choice.get(below)
        try:
            totals, err, w = de16_ab(lab, state, shapes,
                                     lambda on: state.__setitem__("v17_choice", {**choice, below: kind} if on else dict(choice)),
                                     23000 + below)
        except Exception as exc:
            state["v17_choice"] = dict(choice)
            lab_fail(f"e25 <{below}", exc)
            continue
        keep = err < 2e-2 and totals[True] < totals[False] * (1.0 - DE20_GAIN)
        if keep:
            choice[below] = kind
        state["v17_choice"] = dict(choice)
        cells.append(f"<{below}: {before or 'M11'} {totals[False] / w:.1f} -> {kind} {totals[True] / w:.1f} "
                     f"({100 * (1 - totals[True] / totals[False]):+.1f}%) err {err:.0e} -> {choice.get(below) or 'M11'}")
        LAB_SUMMARY.append(f"e25 <{below} {choice.get(below) or 'M11'}")
        torch.cuda.empty_cache()
    state["v17_choice"] = dict(choice)
    log("e25 (weighted us, routing locked): " + "; ".join(cells))
    log(f"V17_CHOICE = {dict(sorted(choice.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v17 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; E25 path {'built' if 'v17_moe' in state else 'OFF'}, file's choice {state.get('v17_choice')}")
    lab.section("e25", sec_e25, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
