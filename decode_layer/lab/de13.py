"""DE13 lab (check runs only), on top of submission_v11 (= v10 + the attention session's v8 + bf16_proj v7a's router
configs at m = 32 / 128 / 256), after lab/de6.py (modes) and lab/de12.py (pdl, modes-keep) are appended before it.
With nothing installed the timed kernel_fn is v11's (V10_PDL {}, DE12's modes).

Sections as DE12 (pdl first, then modes), two changes:
pdl     the bar drops from 1.5% to 0.5% (PDL_BAR): DE12 measured GEMM2 + post-norm PDL at -1.48% on both classes
        (101.1 -> 99.6 us and 168.5 -> 166.0 us weighted, monotonic over the three options, outputs <= 1e-4 from v9's
        path), a consistent gain the 1.5% rule (a noise guard for unproven code) left out.
modes   attention v8 changes full verify 16 (FA3 with num_splits 4 + pack_gqa: judge 198.67 vs 205.57 us on its own
        problem), full verify 1 (V5) and sliding decode 16 (UD3 8,4,3): every case is re-picked (att / pj on or off).
Prints V10_PDL and V4_MODE (make_v4.py --v11 --pdl / --modes copy the last such lines).
"""

PDL_BAR = 0.005


def prepare(weights):  # noqa: F811 (DE12's, with v11's labels)
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v11 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, att {'on' if state.get('v4_att') else 'OFF'}; "
        f"pdl bar {PDL_BAR:.1%}")
    lab.section("pdl", sec_pdl, state, until=0.3 * lab.budget_s)
    lab.section("modes", sec_modes_keep, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
