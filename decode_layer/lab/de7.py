"""DE7 lab (check runs only), on top of submission_v5 (= v4 with parts/m11_pf.py: M11 + L8, a next-task L2
prefetch with a runtime pf per stage config). It CHOOSES the M11 table and installs it before the judge
times kernel_fn, so this check's timing is v5 with the lab's picks.

pf      per token class, on decode's own routing (judge-like inputs, busiest-GPU local_start):
        1 / 8 / 16 / 32 tokens: DE5's configs as the base, each stage with pf in DE7_PF (0 off; ahead 1/2 tasks;
        +16 per-lane 128-B line prefetch, +32 the same with evict_last, else one bulk prefetch by lane 0), at
        the base CTAs per SM, then the best pf at the other CTAs per SM;
        128 / 256 tokens (Marlin today): every M11 instance with nt >= 2 x 4/8 CTAs per SM at pf 0, then the
        two best with every pf.
        Outputs with pf must equal pf 0 bit for bit (a prefetch changes no data). A stage keeps a new config
        only if >= 3% faster than its base; M11 serves 128 / 256 only if < 2e-2 vs Marlin and >= 3% faster.
layer   per scored shape, the whole layer with the new table vs v4's; a class whose shapes got slower in sum
        (> 1%) goes back to v4's config. Prints V5_M11_CONFIGS.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []
DE7_PF = (0, 1, 2, 0x11, 0x12, 0x21, 0x22)
DE7_SMALL = ((1, 1, 1, 5), (8, 1, 8, 9), (16, 16, 1, 17), (32, 32, 1, 33))  # (tokens, batch, q_len, below)
DE7_LARGE = ((128, 16, 8, 192), (256, 32, 8, 321))
DE7_ROUTINGS = {1: 3, 8: 3, 16: 3, 32: 3, 128: 2, 256: 2}


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _decode_routings(state, batch, q_len, n, seed0):
    """n judge-like routings for this batch shape: (local ids int32 [T, 8], topk weights) from the real
    layer front on lab inputs (as DE5)."""
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


def _cfg_text(s):
    keys = ("tt", "nt", "d", "ks", "c", "pf")
    return " ".join(f"{k}{s[k]:#x}" if k == "pf" else f"{k}{s[k]}" for k in keys if k in s and (k != "pf" or s[k]))


class _Stage:
    """Times one stage's candidate configs (mean over the class's routings, graph replay cost removed)."""

    def __init__(self, lab, fixed, run, routings):
        self.lab, self.fixed, self.run, self.routings, self.times = lab, fixed, run, routings, {}

    def time(self, s):
        key = tuple(sorted(s.items()))
        if key not in self.times:
            try:
                us = sum(self.lab.time_graph(lambda i=i: self.run(s, i)) for i in range(len(self.routings)))
                self.times[key] = us / len(self.routings) - self.fixed
            except Exception as exc:
                lab_fail(f"stage {_cfg_text(s)}", exc)
                self.times[key] = float("inf")
        return self.times[key]

    def best(self, cands):
        return min(cands, key=self.time)


def _exact(run, s, pfs):
    """Every pf variant of config s gives pf 0's output bit for bit (routing 0)."""
    ref = run(dict(s, pf=0), 0)
    return all(torch.equal(run(dict(s, pf=p), 0), ref) for p in pfs if p)


def _search_small(st, base, pfs):
    """Base config with every pf at its CTAs per SM, then the best pf at the other CTAs per SM."""
    at_c = [dict(base, pf=p) for p in pfs]
    best_pf = st.best(at_c)["pf"]
    other = dict(base, c=12 - base["c"], pf=best_pf)  # 4 <-> 8
    return st.best(at_c + [other])


def _search_large(st, instances, pfs):
    cands = [dict(inst, c=c, pf=0) for inst in instances for c in (4, 8)]
    top = sorted(cands, key=st.time)[:2]
    return st.best(top + [dict(s, pf=p) for s in top for p in pfs if p])


def sec_pf(lab, state, deadline):
    if "m11" not in state:
        LAB_SUMMARY.append("pf: M11 off")
        return
    l2 = getattr(torch.cuda.get_device_properties(0), "L2_cache_size", 0)
    log(f"L2 {l2 / 2 ** 20:.0f} MB; old table {state['m11_configs']!r}")
    fixed = lab.empty_graph_us()
    old = dict(state["m11_configs"])
    g1s = [dict(nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in M11_INSTANCES if g == 1 and nt >= 2]
    g2s = [dict(tt=tt, nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in M11_INSTANCES if g == 0 and nt >= 2]
    table = {}
    for tokens, batch, q_len, below in DE7_SMALL + DE7_LARGE:
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"pf stopped before T{tokens}")
            break
        try:
            routings = _decode_routings(state, batch, q_len, DE7_ROUTINGS[tokens], 17000 + tokens)
            gen = torch.Generator(device="cuda").manual_seed(tokens)
            x = torch.randn((tokens, D_EXPERT), generator=gen, device="cuda").to(torch.bfloat16)
            base = old.get(below)
            large = base is None
            run1 = lambda s, i: m11_gemm1(x, routings[i][0].view(-1), state, s)
            st1 = _Stage(lab, fixed, run1, routings)
            s1 = _search_large(st1, g1s, DE7_PF) if large else _search_small(st1, base["s1"], DE7_PF)
            ex1 = _exact(run1, s1, DE7_PF)
            ref1 = None if large else dict(base["s1"], pf=0)
            acts = [m11_gemm1(x, loc.view(-1), state, dict(s1, pf=0)) for loc, _ in routings]
            run2 = lambda s, i: m11_gemm2(acts[i], routings[i][0].view(-1), state, s)
            st2 = _Stage(lab, fixed, run2, routings)
            s2 = _search_large(st2, g2s, DE7_PF) if large else _search_small(st2, base["s2"], DE7_PF)
            ex2 = _exact(run2, s2, DE7_PF)
            ok = ex1 and ex2
            if not large:  # keep the base stage unless the new one is >= 3% faster (and exact)
                if not (ok and st1.time(s1) < 0.97 * st1.time(ref1)):
                    s1 = ref1
                if not (ok and st2.time(s2) < 0.97 * st2.time(dict(base["s2"], pf=0))):
                    s2 = dict(base["s2"], pf=0)
            cfg = dict(s1=s1, s2=s2)
            m11_warm(state, cfg)
            new_us = sum(lab.time_graph(lambda loc=loc, tw=tw: m11_run(x, loc, tw, state, cfg)) for loc, tw in routings) / len(routings) - fixed
            marlin = [_experts(x, tw, loc, state) for loc, tw in routings]
            err = max(_v3_rel(m11_run(x, loc, tw, state, cfg), ref) for (loc, tw), ref in zip(routings, marlin))
            if large:
                ref_us = sum(lab.time_graph(lambda loc=loc, tw=tw: _experts(x, tw, loc, state)) for loc, tw in routings) / len(routings) - fixed
                use = ok and err < 2e-2 and new_us < 0.97 * ref_us
                table[below] = cfg if use else None
                verdict = f"M11 {new_us:.1f} vs Marlin {ref_us:.1f} -> {'M11' if use else 'Marlin'}"
            else:
                ref_us = sum(lab.time_graph(lambda loc=loc, tw=tw: m11_run(x, loc, tw, state, base)) for loc, tw in routings) / len(routings) - fixed
                use = ok and err < 2e-2
                table[below] = cfg if use else base
                verdict = f"new {new_us:.1f} vs old {ref_us:.1f}"
            pf_cells = "/".join(f"{st1.time(dict(s1, pf=p)):.0f}" for p in DE7_PF)
            log(f"T{tokens}: s1 {_cfg_text(s1)} {st1.time(s1):.1f} (pf {pf_cells}) | s2 {_cfg_text(s2)} "
                f"{st2.time(s2):.1f} (pf {'/'.join(f'{st2.time(dict(s2, pf=p)):.0f}' for p in DE7_PF)}) | {verdict} "
                f"err {err:.1e} exact {int(ex1)}{int(ex2)}")
            LAB_SUMMARY.append(f"T{tokens} {verdict}")
            del acts, marlin, x, routings
        except Exception as exc:
            lab_fail(f"pf T{tokens}", exc)
        torch.cuda.empty_cache()
    new = []
    for below, cfg in ((5, None), (9, None), (17, None), (33, None), (65, None), (192, None), (321, None)):
        new.append((below, table.get(below, old.get(below))))
    state["m11_configs_old"] = state["m11_configs"]
    state["m11_configs"] = tuple(new)
    for _, cfg in new:
        if cfg is not None:
            m11_warm(state, cfg)


def _classes(tokens):
    for below, _ in ((5, 0), (9, 0), (17, 0), (33, 0), (65, 0), (192, 0), (321, 0)):
        if tokens < below:
            return below
    return None


def sec_layer(lab, state, deadline):
    if "m11_configs_old" not in state:
        return
    fixed = lab.empty_graph_us()
    new, old = state["m11_configs"], state["m11_configs_old"]
    sums, cells = {}, []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"layer stopped before {name}")
            break
        try:
            data = lab_input(batch, q_len, layer, seed=19000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            state["m11_configs"] = old
            t_old = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            state["m11_configs"] = new
            t_new = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            below = _classes(batch * q_len)
            a, b = sums.get(below, (0.0, 0.0))
            sums[below] = (a + weight * t_old, b + weight * t_new)
            cells.append(f"{name} {t_old:.1f}->{t_new:.1f}")
            del data
        except Exception as exc:
            lab_fail(f"layer {name}", exc)
            state["m11_configs"] = new
        torch.cuda.empty_cache()
    final = []
    olds = dict(old)
    for below, cfg in new:
        t_old, t_new = sums.get(below, (0.0, 0.0))
        keep_new = below not in sums or t_new <= 1.01 * t_old
        final.append((below, cfg if keep_new else olds.get(below)))
    state["m11_configs"] = tuple(final)
    for _, cfg in final:
        if cfg is not None:
            m11_warm(state, cfg)
    log("layer v4 table -> new (us): " + "; ".join(cells))
    LAB_SUMMARY.append("layer: " + "; ".join(cells))
    log(f"V5_M11_CONFIGS = {state['m11_configs']!r}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v5 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; M11 {'on' if 'm11' in state else 'OFF'}")
    lab.section("pf", sec_pf, state, until=0.75 * lab.budget_s)
    lab.section("layer", sec_layer, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
