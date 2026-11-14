"""G2P lab section for decode_layer (append after lab/common.py, lab/de6.py, lab/de14.py and lab/de16.py, on a
submission that embeds shared/glue/g2p.py and routes m11_run through g2p_m11_run; g2p_setup + g2p_warm in prepare).

sec_gemm2_post   per M11 class (bounds 5 / 9 / 17 / 33): the whole layer on the scored shapes of that class (busiest-
                 GPU local_start, cold-L2 graphs, routing locked via de16_ab / DE14Lock), GEMM2 + post-norm vs the
                 fused kernel; installs the fused path for the class if >= 0.3% faster and the layer outputs differ
                 by < 1e-2. First a correctness gate (g2p_check: fused vs two kernels at 1/4/8/16/32 tokens, < 1e-2,
                 counters back to zero) — if it fails nothing is installed. Prints V14_FUSED = {below: bool}.
"""

G2P_BAR = 0.003
G2P_BOUNDS = (5, 9, 17, 33)
G2P_MAX_ERR = 1e-2


def sec_gemm2_post(lab, state, deadline):
    picks = {b: False for b in G2P_BOUNDS}
    g2p_install(state, picks)
    for tt, nt, d, ks, pdl in G2P_INSTANCES:  # fused vs M11's GEMM2: registers / spill bytes / static smem
        try:
            fa = state["g2p_ext"].g2p_attrs(tt, nt, d, ks, pdl).tolist()
            ma = state["m11_ext"].m11_attrs(0, tt, nt, d, ks).tolist()
            log(f"g2p attrs t{tt}n{nt}d{d}k{ks} pdl {pdl}: fused regs/spill/smem {fa} vs M11 GEMM2 {ma}")
        except Exception as exc:
            log(f"g2p attrs t{tt}n{nt}d{d}k{ks}: {type(exc).__name__}: {_error_text(exc, 120)}")
    err, clean = g2p_check(state)
    log(f"g2p check: fused vs GEMM2 + post-norm worst rel err {err:.1e}, counters zero after {clean}")
    if not (err < G2P_MAX_ERR and clean):
        log(f"V14_FUSED = {picks!r}")
        return
    lo = 0
    for below in G2P_BOUNDS:
        shapes = [(i, s) for i, s in enumerate(SCORED) if lo <= s[1] * s[2] < below]
        lo = below
        cfg = next((c for b, c in state["g2p_base_configs"] if b == below), None)
        if not shapes or cfg is None or not g2p_ok(cfg["s2"]):
            log(f"g2p < {below}: no scored shape or no fused instance for {cfg and cfg['s2']}")
            continue
        if lab.elapsed() > deadline:
            log(f"g2p < {below}: past the section deadline, stays off")
            break
        try:
            totals, lerr, w = de16_ab(lab, state, shapes, lambda on, b=below: g2p_install(state, {**picks, b: on}),
                                      21000 + below)
        except Exception as exc:
            g2p_install(state, picks)
            lab_fail(f"g2p < {below}", exc)
            log(f"g2p < {below}: FAILED {type(exc).__name__}: {_error_text(exc, 200)}")
            continue
        keep = lerr < G2P_MAX_ERR and totals[True] < totals[False] - G2P_BAR * abs(totals[False])
        picks[below] = keep
        g2p_install(state, picks)
        log(f"g2p < {below} (weighted us over {len(shapes)} shapes, two kernels / fused): {totals[False] / w:.2f} -> "
            f"{totals[True] / w:.2f} ({100 * (1 - totals[True] / totals[False]):+.2f}%), layer err {lerr:.0e} -> "
            f"{'fused' if keep else 'two kernels'}")
    log(f"V14_FUSED = {picks!r}")
    LAB_SUMMARY.append(f"V14_FUSED {picks}")
