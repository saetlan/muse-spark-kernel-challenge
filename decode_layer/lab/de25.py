"""DE25 lab (check runs only), on top of submission_v22 (= v20b with MoE's scored E36 in place of E25: same small-class
kernels, plus E36's hot / cold split configs s128 / s256; V17_CHOICE = v20b's, M19 at 65-191 and 192-320 tokens), after
lab/de6.py, lab/de12.py, lab/de14.py and lab/de16.py (de16_ab) are appended before it.

split    per large class, the whole layer at the class's scored shapes (verify 16 = 128 tokens for 65-191, verify 32 =
         256 tokens for 192-320; sliding and full, weights 46 / 16; busiest-GPU local_start, cold-L2 graphs, routing
         LOCKED): the file's plain M19 vs E36's split (s128: cold M11b + hot M19; s256: cold M19 + hot M19 with the late
         GEMM1 and GEMM2 waits); the split is installed if >= 2% faster (DE20's bar for MoE kernel swaps) and the outputs agree
         (< 2e-2). prepare already checked both splits against Marlin.
Prints V17_CHOICE (make_v4.py --v22 --choice copies the last such line).
"""

DE25_GAIN = 0.02
DE25_TESTS = ((192, "s128"), (321, "s256"))


def sec_split(lab, state, deadline):
    choice = dict(state.get("v17_choice") or {})
    if "v17_moe" not in state:
        LAB_SUMMARY.append("split: E36 path OFF")
        log("split: the E36 path is off (see the [v22] line); the file's V17_CHOICE is printed unchanged")
        log(f"V17_CHOICE = {dict(sorted(V17_CHOICE.items()))!r}")
        return
    lows, cells = dict(V17_CLASSES), []
    for below, kind in DE25_TESTS:
        shapes = [(i, s) for i, s in enumerate(SCORED) if lows[below] <= s[1] * s[2] < below]
        if not shapes:
            continue
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"split stopped before class <{below}")
            break
        before = choice.get(below, "Marlin")
        try:
            totals, err, w = de16_ab(lab, state, shapes, lambda on: state.__setitem__(
                "v17_choice", {**choice, below: kind} if on else dict(choice)), 25000 + below)
        except Exception as exc:
            state["v17_choice"] = dict(choice)
            lab_fail(f"split <{below}", exc)
            continue
        keep = err < 2e-2 and totals[True] < totals[False] * (1.0 - DE25_GAIN)
        if keep:
            choice[below] = kind
        state["v17_choice"] = dict(choice)
        gain = 100 * (1 - totals[True] / totals[False]) if totals[False] > 0 else 0.0
        cells.append(f"<{below}: {before} {totals[False] / w:.1f} -> {kind} {totals[True] / w:.1f} ({gain:+.1f}%) "
                     f"err {err:.0e} -> {choice.get(below, 'Marlin')}")
        LAB_SUMMARY.append(f"split <{below} {choice.get(below, 'Marlin')} {gain:+.1f}%")
        torch.cuda.empty_cache()
    state["v17_choice"] = dict(choice)
    log("split (weighted us, routing locked): " + "; ".join(cells))
    log(f"V17_CHOICE = {dict(sorted(choice.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v22 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; att {'on' if state.get('v4_att') else 'OFF'}, E36 choice {state.get('v17_choice')}")
    lab.section("split", sec_split, state, min_left=3.0)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
