"""DE11 lab (check runs only), on top of the clean v7 (submission_v7.py with DE9's picks), after lab/de7.py
(appended before it; its prepare renamed): DE7's sections (sec_pf, sec_layer) with a different search. MF9
(the MoE session): CTAs per SM decide M11's speed, and DE5 / DE7 only tried 4 and 8. Here every M11 instance
of the stage x 3..8 CTAs per SM, prefetch off (DF28), per token class on decode's own routing (two routings
each); a stage switches only if >= 3% faster than its base; 128 / 256 tokens take M11 only if < 2e-2 vs Marlin
and >= 3% faster; the whole-layer check reverts any class that got > 1% slower. Prints V5_M11_CONFIGS.
"""

DE7_PF = (0,)
DE7_ROUTINGS = {1: 2, 8: 2, 16: 2, 32: 2, 128: 2, 256: 2}
DE11_C = (3, 4, 5, 6, 7, 8)


def _stage_instances(gated):
    if gated:
        return [dict(nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in M11_INSTANCES if g == 1]
    return [dict(tt=tt, nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in M11_INSTANCES if g == 0]


def _search_small(st, base, pfs):
    """Every instance of this stage (GEMM1 if base has no tt) x DE11_C CTAs per SM, and the base itself."""
    cands = [dict(inst, c=c, pf=0) for inst in _stage_instances("tt" not in base) for c in DE11_C]
    return st.best(cands + [dict(base, pf=0)])


def _search_large(st, instances, pfs):
    return st.best([dict(inst, c=c, pf=0) for inst in instances for c in DE11_C])


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v7 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; M11 {'on' if 'm11' in state else 'OFF'}")
    lab.section("csearch", sec_pf, state, until=0.8 * lab.budget_s)
    lab.section("layer", sec_layer, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
