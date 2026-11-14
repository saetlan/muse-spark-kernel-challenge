"""DE16 lab (check runs only), on top of submission_v13 (= v12 with DE15's picks: V4_MODE, V10_PDL, V12_BOUNDARY
{2: 16, 9: 8, 17: 16}; + parts/v13.py, the concat instead of F.pad, off until a lab installs it), after lab/de6.py,
lab/de12.py and lab/de14.py (routing lock, cluster helpers) are appended before it. With nothing installed the timed
kernel_fn is DE15's.

pad      the whole layer at the scored shapes with <= 8 tokens (decode 1 and verify 1, sliding and full; weights 46 / 16;
         busiest-GPU local_start; cold-L2 graphs; routing locked), F.pad vs one concat: installed if >= 0.3% faster and
         the outputs agree (< 1e-3: the concat is bit-identical; M11's route lists come from atomics).
b32      the same A/B at decode 32 (sliding and full) for SGLang's cluster boundary kernel with C 16 (DE15: both
         boundaries 6.72 -> 5.09 us, layer -0.48%, just under its 0.5% bar): kept if >= 0.3% faster, outputs < 2e-2.
Prints V13_CAT and V12_BOUNDARY (make_v4.py --v13 --cat / --boundary copy the last such lines).
"""

DE16_BAR = 0.003


def de16_ab(lab, state, shapes, set_variant, seed):
    """Weighted cold-L2 µs of the layer over `shapes` for set_variant(False) / (True), routing locked; and the worst
    relative difference of the layer outputs."""
    fixed = lab.empty_graph_us()
    totals, outs = {}, {}
    de14_lock(state)
    try:
        for on in (False, True):
            set_variant(on)
            tot = 0.0
            for i, (name, batch, q_len, layer, weight) in shapes:
                data = lab_input(batch, q_len, layer, seed=seed + i)
                _, _, ids = kernel_fn(data, state)
                data["local_start"].fill_(_busiest_start(ids))
                kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                outs[(on, i)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)[0]
                del kc, vc
                tot += weight * (lab.time_graph(lambda: kernel_fn(data, state)) - fixed)
                del data
            totals[on] = tot
    finally:
        de14_unlock(state)
    err = max(_v3_rel(outs[(True, i)], outs[(False, i)]) for i, _s in shapes)
    return totals, err, sum(s[4] for _i, s in shapes)


def sec_pad(lab, state, deadline):
    shapes = [(i, s) for i, s in enumerate(SCORED) if s[1] * s[2] <= SLICED_UNPOOL_MAX_TOKENS]
    try:
        totals, err, w = de16_ab(lab, state, shapes, lambda on: v13_install(state, on), 19000)
    except Exception:
        v13_install(state, V13_CAT)  # the file's choice, as before the section
        raise
    keep = err < 1e-3 and totals[True] < totals[False] - DE16_BAR * abs(totals[False])
    v13_install(state, keep)
    state["v13_cat"] = keep
    log(f"pad (weighted us F.pad / concat over {len(shapes)} shapes <= {SLICED_UNPOOL_MAX_TOKENS} tokens): "
        f"{totals[False] / w:.2f} -> {totals[True] / w:.2f} err {err:.0e} -> {'concat' if keep else 'F.pad'}")
    log(f"V13_CAT = {keep!r}")


def sec_b32(lab, state, deadline):
    picks = dict(state.get("v12_boundary", {}))
    shapes = [(i, s) for i, s in enumerate(SCORED) if 17 <= s[1] * s[2] < 33]
    try:
        totals, err, w = de16_ab(lab, state, shapes,
                                 lambda on: v12_install(state, {**picks, **({33: 16} if on else {})}), 20000)
    except Exception:
        v12_install(state, picks)
        raise
    keep = err < 2e-2 and totals[True] < totals[False] - DE16_BAR * abs(totals[False])
    final = {**picks, **({33: 16} if keep else {})}
    v12_install(state, final)
    log(f"b32 (weighted us C 1 / C 16 at decode 32): {totals[False] / w:.2f} -> {totals[True] / w:.2f} err {err:.0e} -> "
        f"{'C16' if keep else 'C1'}")
    log(f"V12_BOUNDARY = {dict(sorted(final.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v13 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; boundaries {state.get('v12_boundary')}; PDL {state.get('v10_pdl')}")
    state.setdefault("v12_base", {key: state[key] for key in V12_KEYS})
    lab.section("pad", sec_pad, state, until=0.5 * lab.budget_s)
    lab.section("b32", sec_b32, state)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
