"""DE5 lab (check runs only), on top of submission_v3. It CHOOSES v3's settings and installs them in the
state before the judge times kernel_fn, so this check's timing is v3 with the lab's picks:

experts  per token class (1, 8, 16, 32 tokens), on decode's own routing (3 judge-like inputs each): every
         M11 GEMM1 and GEMM2 instance x 4/8 CTAs per SM timed as cold-L2 graphs, best stage pair by mean;
         then M11 (picks) vs D1's Marlin path end to end. M11 serves a class only if correct (< 2e-2 vs
         Marlin on every routing) and >= 3% faster; else Marlin. Prints the config table.
prefetch per scored shape, the whole v3 layer with 0 / 16.8 (o) / 33 / 50 MB prefetched into L2 during
         attention; a size is kept only if >= 2% faster than none. Prints the table.
summary  per shape v3 (final) vs D1, router timing.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []
DE5_CLASSES = ((1, 1, 1, 5), (8, 1, 8, 9), (16, 16, 1, 17), (32, 32, 1, 33))  # (tokens, batch, q_len, below)


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _decode_routings(state, batch, q_len, n, seed0):
    """n judge-like routings for this batch shape: (local ids int32 [T, 8], topk weights) from the real
    layer front on lab inputs (sliding layer: the routing doesn't depend on the layer type's cost)."""
    out = []
    for i in range(n):
        data = lab_input(batch, q_len, "sliding", seed=seed0 + i)
        _, _, ids = kernel_fn(data, state)
        ls = _busiest_start(ids)
        loc = ((ids.long() - ls) % N_EXPERTS).to(torch.int32).contiguous()
        g = torch.Generator(device="cuda").manual_seed(seed0 + 100 + i)
        tw = torch.rand(loc.shape, generator=g, device="cuda")
        out.append((loc, (tw / tw.sum(-1, keepdim=True)).float().contiguous()))
        del data
    torch.cuda.empty_cache()
    return out


def sec_experts(lab, state, deadline):
    if "m11" not in state:
        LAB_SUMMARY.append("experts: M11 off, Marlin everywhere")
        return
    fixed = lab.empty_graph_us()
    g1s = [i for i in M11_INSTANCES if i[0] == 1]
    g2s = [i for i in M11_INSTANCES if i[0] == 0]
    table = []
    for tokens, batch, q_len, below in DE5_CLASSES:
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"experts stopped before T{tokens}")
            table.append((below, None))
            continue
        routings = _decode_routings(state, batch, q_len, 3, 7000 + tokens)
        gen = torch.Generator(device="cuda").manual_seed(tokens)
        x = torch.randn((tokens, D_EXPERT), generator=gen, device="cuda").to(torch.bfloat16)
        local = [int(((loc < LOCAL_EXPERTS)).sum()) for loc, _ in routings]
        distinct = [int(loc[loc < LOCAL_EXPERTS].unique().numel()) for loc, _ in routings]
        s1_best, s2_best = None, None
        for _, tt, nt, d, ks in g1s:
            for c in (4, 8):
                s1 = dict(nt=nt, d=d, ks=ks, c=c)
                try:
                    us = sum(lab.time_graph(lambda: m11_gemm1(x, loc.view(-1), state, s1)) for loc, _ in routings) / 3 - fixed
                except Exception as exc:
                    lab_fail(f"s1 T{tokens} {s1}", exc)
                    continue
                if s1_best is None or us < s1_best[0]:
                    s1_best = (us, s1)
        acts = [m11_gemm1(x, loc.view(-1), state, s1_best[1]) for loc, _ in routings]
        for _, tt, nt, d, ks in g2s:
            for c in (4, 8):
                s2 = dict(tt=tt, nt=nt, d=d, ks=ks, c=c)
                try:
                    us = sum(lab.time_graph(lambda a=a, loc=loc: m11_gemm2(a, loc.view(-1), state, s2))
                             for a, (loc, _) in zip(acts, routings)) / 3 - fixed
                except Exception as exc:
                    lab_fail(f"s2 T{tokens} {s2}", exc)
                    continue
                if s2_best is None or us < s2_best[0]:
                    s2_best = (us, s2)
        cfg = dict(s1=s1_best[1], s2=s2_best[1])
        m11_us = sum(lab.time_graph(lambda loc=loc, tw=tw: m11_run(x, loc, tw, state, cfg)) for loc, tw in routings) / 3 - fixed
        marlin_us = sum(lab.time_graph(lambda loc=loc, tw=tw: _experts(x, tw, loc, state)) for loc, tw in routings) / 3 - fixed
        err = max(_v3_rel(m11_run(x, loc, tw, state, cfg), _experts(x, tw, loc, state)) for loc, tw in routings)
        use = err < 2e-2 and m11_us < 0.97 * marlin_us
        table.append((below, cfg if use else None))
        log(f"experts T{tokens} local {local} distinct {distinct}: s1 {s1_best[0]:.1f} {s1_best[1]} | s2 {s2_best[0]:.1f} "
            f"{s2_best[1]} | M11 {m11_us:.1f} vs Marlin {marlin_us:.1f} err {err:.1e} -> {'M11' if use else 'Marlin'}")
        LAB_SUMMARY.append(f"exp T{tokens} M11 {m11_us:.1f} Marlin {marlin_us:.1f}")
    table.append((321, None))
    state["m11_configs"] = tuple(table)
    for _, cfg in table:
        if cfg is not None:
            m11_warm(state, cfg)
    log(f"V3_M11_CONFIGS = {tuple(table)!r}")


def sec_prefetch(lab, state, deadline):
    if "v3_ext" not in state:
        LAB_SUMMARY.append("prefetch: off (ext failed)")
        return
    fixed = lab.empty_graph_us()
    sizes = (0, 16 << 20, 33 << 20, 50 << 20)
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"prefetch stopped before {name}")
            break
        try:
            data = lab_input(batch, q_len, layer, seed=9000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            key = (batch * q_len, layer == "sliding")
            times = []
            for nb in sizes:
                state["v3_prefetch"][key] = nb
                times.append(lab.time_graph(lambda: kernel_fn(data, state)) - fixed)
            best = min(range(len(sizes)), key=lambda j: times[j])
            keep = sizes[best] if times[best] < 0.98 * times[0] else 0
            state["v3_prefetch"][key] = keep
            cells.append(f"{name} " + "/".join(f"{t:.1f}" for t in times) + f" -> {keep >> 20}MB")
            del data
        except Exception as exc:
            lab_fail(f"prefetch {name}", exc)
        torch.cuda.empty_cache()
    log("prefetch (0/16/33/50 MB, us): " + "; ".join(cells))
    log(f"V3_PREFETCH = {dict(state['v3_prefetch'])!r}")


def sec_summary(lab, state, deadline):
    fixed = lab.empty_graph_us()
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            break
        try:
            data = lab_input(batch, q_len, layer, seed=11000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            v3 = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            d1 = lab.time_graph(lambda: _d1_kernel_fn(data, state)) - fixed
            cells.append(f"{name} {v3:.1f}/{d1:.1f}")
            del data
        except Exception as exc:
            lab_fail(f"summary {name}", exc)
        torch.cuda.empty_cache()
    rt = []
    for t in (1, 2, 4, 8):
        xr = torch.randn((t, HIDDEN), device="cuda")
        mine = lab.time_graph(lambda: v3_router_logits(xr, state)) - fixed
        ref = lab.time_graph(lambda: F.linear(xr, state["router"])) - fixed
        rt.append(f"T{t} {mine:.1f}/{ref:.1f}")
    log("layer v3/D1 (us): " + "; ".join(cells))
    log("router v3/F.linear (us): " + " ".join(rt))
    LAB_SUMMARY.append("layer v3/D1: " + "; ".join(cells))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v3 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s")
    lab.section("experts", sec_experts, state, until=0.45 * lab.budget_s)
    lab.section("prefetch", sec_prefetch, state, until=0.8 * lab.budget_s)
    lab.section("summary", sec_summary, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
