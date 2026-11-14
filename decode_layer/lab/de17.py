"""DE17 lab (check runs only), on top of submission_v15 (= v13 with DE15 / DE16's picks, + parts/v15.py: local ids on
the side stream and the pre-expert norm before the join, off until a lab installs it; V10_PDL_CLASSES + 17), after
lab/de6.py, lab/de12.py, lab/de14.py and lab/de16.py are appended before it. With nothing installed the timed
kernel_fn is DE16's.

side     the whole layer at all 12 scored shapes (weights 46 / 16, busiest-GPU local_start, cold-L2 graphs, routing
         locked), v15 off vs on: installed if the weighted time is >= 0.3% lower and the outputs agree (< 1e-3; the same
         kernels run, only their streams and order change).
pdl      DE12's section, routing locked, bar 0.5%, now over classes 1-4 / 5-8 / 9-16 (decode 16 gets the option).
Prints V15_SIDE and V10_PDL (make_v4.py --v15 --side / --pdl copy the last such lines).
"""


def sec_side(lab, state, deadline):
    shapes = list(enumerate(SCORED))
    keep_file = bool(state.get("v15_side"))
    try:
        totals, err, w = de16_ab(lab, state, shapes, lambda on: state.__setitem__("v15_side", on), 21000)
    except Exception:
        state["v15_side"] = keep_file
        raise
    keep = err < 1e-3 and totals[True] < totals[False] - DE16_BAR * abs(totals[False])
    state["v15_side"] = keep
    log(f"side (weighted us over 12 shapes, v15 off / on): {totals[False] / w:.2f} -> {totals[True] / w:.2f} "
        f"err {err:.0e} -> {'on' if keep else 'off'}")
    log(f"V15_SIDE = {keep!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v15 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; PDL classes {V10_PDL_CLASSES}, file's PDL {state.get('v10_pdl')}")
    lab.section("side", sec_side, state, until=0.5 * lab.budget_s)
    lab.section("pdl", sec_pdl_locked, state)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
