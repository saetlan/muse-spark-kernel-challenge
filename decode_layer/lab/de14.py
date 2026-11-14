"""DE14 lab (check runs only), on top of submission_v12 (= v11 with DE13's picks + parts/v12.py: SGLang's cluster
norm-residual kernel for the two layer boundaries, off until a lab installs it). With nothing installed the timed
kernel_fn is v11's.

boundary compiles SGLang's cluster modules one at a time, C = 8 / 4 / 16 (portable 8 first; GB300's own policy uses
         16 / 8 / 4 for <= 8 / 16 / 32 tokens; C 2 is its 33-64-token choice, not a scored size), each only if it and the
         timings still fit the SECTION's deadline (DE14: each build took 9.1 s and the guard watched only the lab's
         total, so the four builds ran past the section's deadline and nothing was timed). Then per scored token count (1, 8, 16, 32, 128): both boundaries (the attention one with the FP32 router copy where
         the layer asks for it) as one cold-L2 graph, at C = 1 (SGLang's dispatch, v11) and each built C; every C's
         three outputs within 1e-2 of C = 1's on fresh inputs; a class takes the fastest C if >= 3% and >= 0.2 us
         faster than C = 1.
layer    per class with a pick, the whole layer at the class's scored shapes (weights 46 / 16, busiest-GPU local_start,
         cold-L2 graphs) with and without it; kept if >= 0.5% faster (sign-safe) and the layer output stays within
         2e-2 of v11's, else reverted.
modes    de6's per-case choice (attention / projection drop-ins on or off) with the routing LOCKED: every variant of a
         case runs the expert stage on the routing its first run produced (DE13: whole-layer variants that flip
         near-tie experts moved the expert stage by +-10-25 us, more than the drop-ins' own effect, so the lab
         picked attention off at sliding verify 1 on noise; v6 and v8 run the same attention there). A case keeps
         the fastest variant if >= 2% faster than all off; a case not reached keeps the file's picks.
pdl      DE12's section (bar 0.5%), routing locked, last (the smallest expected gain).
Every whole-layer comparison here (layer, pdl, modes) runs with the routing locked; the timed kernel_fn after
prepare runs the real routing. Prints V12_BOUNDARY, V10_PDL and V4_MODE (make_v4.py --v12 --boundary / --pdl /
--modes copy the last such lines).
"""

DE14_TOKENS = ((1, 2), (8, 9), (16, 17), (32, 33), (128, 129))  # (token count, class bound: fewer than)
DE14_GAIN, DE14_GAIN_US, DE14_LAYER_BAR = 0.03, 0.2, 0.005
PDL_BAR = 0.005
DE14_BUILD_ORDER = (8, 4, 16)
DE14_TIMING_S = 8.0  # the timings after the builds (5 token counts x 4 graphs): well under this


def de14_both(base, ins, router, size):
    def go():
        for key in V12_KEYS:
            branch, stream = ins[key]
            r = router if key == "attn_boundary" else None
            if size == 1:
                base[key](stream, branch, branch_dtype=torch.bfloat16, router_dtype=r)
            else:
                v12_cluster_call(base[key], stream, branch, r, size)
    return go


class DE14Lock:
    """state["routing"] stand-in for whole-layer comparisons: runs SGLang's routing (its cost stays in every timing),
    then overwrites its outputs with the routing recorded on the first eager call after reset()."""

    def __init__(self, real):
        self.real, self.fixed = real, None

    def reset(self):
        self.fixed = None

    def __call__(self, *args, **kwargs):
        weights_, ids = self.real(*args, **kwargs)
        if self.fixed is None or self.fixed[1].shape != ids.shape:
            self.fixed = (weights_.clone(), ids.clone())
        else:
            weights_.copy_(self.fixed[0])
            ids.copy_(self.fixed[1])
        return weights_, ids


def de14_lock(state):
    if not isinstance(state["routing"], DE14Lock):
        state["routing"] = DE14Lock(state["routing"])
    state["routing"].reset()
    return state["routing"]


def de14_unlock(state):
    if isinstance(state["routing"], DE14Lock):
        state["routing"] = state["routing"].real


def sec_modes_locked(lab, state, deadline):
    fixed = lab.empty_graph_us()
    keep = {key: dict(val) for key, val in state["v4_mode"].items()}
    cells, kept = [], []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        key = (batch * q_len, layer == "sliding")
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"modes stopped before {name}")
            state["v4_mode"][key] = dict(keep.get(key, {"att": False, "pj": False}))
            kept.append(name)
            continue
        try:
            lock = de14_lock(state)
            data = lab_input(batch, q_len, layer, seed=13000 + i)
            state["v4_mode"][key] = {"att": False, "pj": False}
            _, _, ids = kernel_fn(data, state)  # records the case's routing (all off)
            data["local_start"].fill_(_busiest_start(ids))
            outs, times = {}, {}
            for att, pj in DE6_VARIANTS:
                state["v4_mode"][key] = {"att": att, "pj": pj}
                kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                outs[(att, pj)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)
                del kc, vc
                times[(att, pj)] = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            base = outs[(False, False)]
            errs = {v: _v3_rel(outs[v][0], base[0]) for v in DE6_VARIANTS}
            same = all(bool((outs[v][2] == lock.fixed[1]).all()) for v in DE6_VARIANTS)
            ok = [v for v in DE6_VARIANTS if errs[v] < DE6_GATE]
            best = min(ok, key=lambda v: times[v])
            pick = best if times[best] < 0.98 * times[(False, False)] else (False, False)
            state["v4_mode"][key] = {"att": pick[0], "pj": pick[1]}
            cells.append(f"{name} " + "/".join(f"{times[v]:.1f}" for v in DE6_VARIANTS)
                         + f" err {'/'.join(f'{errs[v]:.0e}' for v in DE6_VARIANTS[1:])}"
                         + f"{'' if same else ' UNLOCKED'} -> att{int(pick[0])} pj{int(pick[1])}")
            LAB_SUMMARY.append(f"{name} {times[(False, False)]:.1f}->{times[pick]:.1f}")
            del data, outs
        except Exception as exc:
            lab_fail(f"modes {name}", exc)
            state["v4_mode"][key] = dict(keep.get(key, {"att": False, "pj": False}))
        torch.cuda.empty_cache()
    de14_unlock(state)
    if kept:
        log(f"modes not reached, file's picks kept: {', '.join(kept)}")
    log("modes, routing locked (us off/att/pj/both; err att/pj/both vs off): " + "; ".join(cells))
    log(f"V4_MODE = {dict(sorted(state['v4_mode'].items()))!r}")


def sec_pdl_locked(lab, state, deadline):
    de14_lock(state)
    try:
        sec_pdl(lab, state, deadline)
    finally:
        de14_unlock(state)


def sec_boundary(lab, state, deadline):
    built = v12_build(DE14_BUILD_ORDER, may_start=lambda last: lab.elapsed() + 1.5 * max(last, 10.0) + DE14_TIMING_S
                      < deadline)
    log("boundary builds (s, in order): " + ", ".join(
        f"C{s} {v:.1f}" if not isinstance(v, str) else f"C{s} {'FAILED ' if 'skipped' not in v else ''}{v}"
        for s, v in built.items()))
    sizes = [s for s in V12_SIZES if isinstance(built.get(s), float)]
    if not sizes:
        LAB_SUMMARY.append("boundary: nothing built")
        state["de14_picks"] = {}
        return
    base = state["v12_base"]
    fixed = lab.empty_graph_us()
    gen = torch.Generator(device="cuda").manual_seed(59)
    picks, cells = {}, []
    for tokens, below in DE14_TOKENS:
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"boundary stopped before T{tokens}")
            break
        try:
            router = torch.float32 if tokens <= ROUTER_FP32_MAX_TOKENS else None
            ins = {key: v12_inputs(tokens, "cuda", gen) for key in V12_KEYS}
            errs = {s: max(v12_compare(base[key], tokens, s, "cuda", gen) for key in V12_KEYS) for s in sizes}
            times = {s: lab.time_graph(de14_both(base, ins, router, s)) - fixed for s in [1] + sizes}
            ok = [s for s in sizes if errs[s] < 1e-2]
            best = min(ok, key=lambda s: times[s]) if ok else None
            if best is not None and times[best] < times[1] * (1 - DE14_GAIN) and times[best] < times[1] - DE14_GAIN_US:
                picks[below] = best
            cells.append(f"T{tokens} " + "/".join(f"{times[s]:.2f}" for s in [1] + sizes)
                         + " err " + "/".join(f"{errs[s]:.0e}" for s in sizes) + f" -> C{picks.get(below, 1)}")
            LAB_SUMMARY.append(f"boundary T{tokens} C{picks.get(below, 1)}")
            del ins
        except Exception as exc:
            lab_fail(f"boundary T{tokens}", exc)
        torch.cuda.empty_cache()
    state["de14_picks"] = picks
    log(f"boundary (us, both boundaries, C 1/{'/'.join(map(str, sizes))}; err vs C 1): " + "; ".join(cells))


def sec_layer(lab, state, deadline):
    picks = dict(state.get("de14_picks", {}))
    fixed = lab.empty_graph_us()
    bounds = [b for _t, b in DE14_TOKENS]
    kept, cells = {}, []
    for below in sorted(picks):
        low = max([b for b in bounds if b < below], default=1)
        shapes = [(i, s) for i, s in enumerate(SCORED) if low <= s[1] * s[2] < below]
        if not shapes:
            continue
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"layer stopped before class <{below}")
            break
        try:
            totals, outs = {}, {}
            de14_lock(state)
            for on in (False, True):
                v12_install(state, {**kept, **({below: picks[below]} if on else {})})
                tot = 0.0
                for i, (name, batch, q_len, layer, weight) in shapes:
                    data = lab_input(batch, q_len, layer, seed=18000 + i)
                    _, _, ids = kernel_fn(data, state)
                    data["local_start"].fill_(_busiest_start(ids))
                    kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                    outs[(on, i)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)[0]
                    del kc, vc
                    tot += weight * (lab.time_graph(lambda: kernel_fn(data, state)) - fixed)
                    del data
                totals[on] = tot
            de14_unlock(state)
            err = max(_v3_rel(outs[(True, i)], outs[(False, i)]) for i, _s in shapes)
            wsum = sum(s[4] for _i, s in shapes)
            if err < 2e-2 and totals[True] < totals[False] - DE14_LAYER_BAR * abs(totals[False]):
                kept[below] = picks[below]
            cells.append(f"<{below} C{picks[below]}: {totals[False] / wsum:.1f} -> {totals[True] / wsum:.1f} "
                         f"err {err:.0e} -> {'kept' if below in kept else 'reverted'}")
            LAB_SUMMARY.append(f"layer <{below} {'kept' if below in kept else 'reverted'}")
        except Exception as exc:
            de14_unlock(state)
            lab_fail(f"layer <{below}", exc)
        torch.cuda.empty_cache()
    v12_install(state, kept)
    log("layer (weighted us without / with the pick): " + "; ".join(cells))
    log(f"V12_BOUNDARY = {dict(sorted(kept.items()))!r}")


def prepare(weights):  # noqa: F811
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v12 prepare {base_s:.1f}s ({', '.join(f'{k} {v:.1f}' for k, v in V9_TIMES.items())}); lab budget "
        f"{lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, att {'on' if state.get('v4_att') else 'OFF'}; "
        f"PDL bar {PDL_BAR:.1%}")
    state.setdefault("v12_base", {key: state[key] for key in V12_KEYS})
    lab.section("boundary", sec_boundary, state, until=0.7 * lab.budget_s)
    lab.section("layer", sec_layer, state, until=0.8 * lab.budget_s)
    lab.section("modes", sec_modes_locked, state, until=0.95 * lab.budget_s)
    lab.section("pdl", sec_pdl_locked, state)
    de14_unlock(state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
