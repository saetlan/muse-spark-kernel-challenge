"""E7: where does the bandwidth go, and does PDL work now? Check runs only; kernel_fn is submission_m13's.
  bw     recon on the judge's H200, cold L2: (a) torch int64 sum over the first 3/9/16 experts' GEMM1 codes
         (50/151/268 MB, contiguous); (b) a plain streaming kernel over the same regions with 4/8/16 16-B loads
         per lane in flight x 2/4/8/16 CTAs per SM; (c) M13's GEMM1 and GEMM2 with and without the math
         (MATH=false keeps the loads, cursor and flushes) at bs 1/8/32 (3 routings), as TB/s of weight bytes
  pdl    M13 (E5's best stages) with each (GEMM2 PDL, L2-prefetch groups, post-norm PDL) variant at bs 1/8/32:
         error vs FP32 eager and from a CUDA graph replay, scratch clean, time
  e2e    per class [1,5) [5,9) [9,17) [17,33) [33,65): the fastest variant whose eager and graph errors pass at
         every size of the class (judge-like, skewed, outliers), vs M11 and v1; M13 if >= 3% faster than both"""

from nvfp4 import dequantize, outlier_channels

LAB_BUDGET_S = 240.0
E5_PREPARE_LIMIT_S = 275.0
E5_ROUTINGS = 3
E5_REPS = 10
E5_MAX_ERR, E5_MAX_ROW_ERR = 1e-2, 5e-2
E7_BS = (1, 8, 16, 32, 64)
E7_CLASS = {1: 5, 8: 9, 16: 17, 32: 33, 64: 65}
E7_PDL = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 2, 1), (1, 6, 1))
# E5's best M13 stages (MF8): GEMM1 n1 c4; GEMM2 t2n1c4 up to 16 tokens, t1n1c5 above
E7_S1 = dict(kind="m13", nt=1, c=4)
E7_S2 = {1: dict(kind="m13", tt=2, nt=1, c=4), 8: dict(kind="m13", tt=2, nt=1, c=4), 16: dict(kind="m13", tt=2, nt=1, c=4),
         32: dict(kind="m13", tt=1, nt=1, c=5), 64: dict(kind="m13", tt=2, nt=1, c=4)}
GEMM1_MB, GEMM2_MB = 18.87, 9.44  # weight + scale bytes per expert


def _case(bs, seed, skew=False, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if outliers:  # like the judge's skewed test: 1% of channels scaled 30x
        x = outlier_channels(x, generator=gen)
    if skew:  # every token on experts 0-7: bs pairs per expert
        ids = torch.arange(TOP_K, dtype=torch.int32, device="cuda").repeat(bs, 1)
        w = torch.rand((bs, TOP_K), generator=gen, device="cuda")
        w = w / w.sum(-1, keepdim=True)
    else:
        ids, w = synthetic_routing(bs, seed)
    return x, ids.contiguous(), w.contiguous()


class _Ref:
    """FP32 references from the raw NVFP4 weights, experts dequantized on demand."""

    def __init__(self, weights):
        self.w = weights
        self.cache = {}

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                             dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.cache[e]

    def local_pairs(self, ids):
        flat = ids.reshape(-1).long()
        return flat, [e for e in torch.unique(flat).tolist() if e < 32]

    def act(self, x, ids):
        flat, experts = self.local_pairs(ids)
        out = torch.zeros((flat.numel(), 4096), device="cuda")
        for e in experts:
            p = (flat == e).nonzero().flatten()
            h = x[p // TOP_K].float() @ self.expert(e)[0].T
            out[p] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        return out

    def stage2(self, act, ids, tw):
        flat, experts = self.local_pairs(ids)
        nw = 1.0 + self.w["post_norm_w"].float()
        out = torch.zeros((ids.shape[0], 4096), device="cuda")
        for e in experts:
            p = (flat == e).nonzero().flatten()
            y = (act[p].float() @ self.expert(e)[1].T) * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, p // TOP_K, y * tw.reshape(-1)[p].unsqueeze(1))
        return out

    def full(self, x, ids, tw):
        return self.stage2(self.act(x, ids), ids, tw)


def _errors(out, ref):
    """(relative L2, worst row) as the judge's checks.compare measures them."""
    out, ref = out.float(), ref.float()
    diff = out - ref
    err = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    row = float((diff.norm(dim=-1) / torch.maximum(rows_ref, floor)).max())
    return (err if bool(torch.isfinite(out).all()) else float("inf")), row


def _build_error_lines(exc, keep=3500):
    lines = [l.strip() for l in str(exc).splitlines()]
    hits = [l for l in lines if "error" in l.lower() or "Error" in l]
    text = " | ".join(hits) if hits else " ".join(lines)
    return text[:keep]


def _clean(state):
    """True when M13's scratch and counters are all zero (they must be after every complete call); re-zeroes."""
    bank = state["m13"]
    dirty = [k for k in ("scr13", "scr2", "ctr13", "ctr2") if bool(bank[k].ne(0).any())]
    for k in dirty:
        bank[k].zero_()
    return dirty


def _post(pdl=1):
    return dict(kind="m13", pdl=pdl)




def sec_bw(lab, state):
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    len16 = bank["w13"][0].numel() // 4  # uint4 per expert region (codes only)
    for n in (3, 9, 16):
        mb = n * len16 * 16 / 1e6
        t_torch = lab.time_graph(lambda: bank["w13"][:n].view(torch.int64).sum())
        experts = torch.arange(n, dtype=torch.int32, device="cuda")
        parts = []
        for u in (4, 8, 16):
            row = []
            for c in (2, 4, 8, 16):
                t = lab.time_graph(lambda: ext.m13_stream(bank["w13"], experts, len16, sink, u, c * sms))
                row.append(f"{mb / t:.2f}")
            parts.append(f"U{u} " + "/".join(row))
        log(f"bw n{n} ({mb:.0f} MB codes): torch sum {mb / t_torch:.2f} TB/s ({t_torch:.1f} us); stream TB/s at "
            f"c2/4/8/16: " + "; ".join(parts))
    for bs in (1, 8, 32):
        res = {"g1": [0.0, 0.0], "g2": [0.0, 0.0]}
        mb1 = mb2 = 0.0
        for r in range(E5_ROUTINGS):
            x, ids, w = _case(bs, 900 + bs + 2000 * r)
            flat = ids.reshape(-1)
            ne = int((torch.bincount(flat.long(), minlength=256)[:32] > 0).sum())
            mb1 += ne * GEMM1_MB / E5_ROUTINGS
            mb2 += ne * GEMM2_MB / E5_ROUTINGS
            act = torch.randn((flat.numel(), 4096), device="cuda").to(torch.bfloat16)
            n_pairs = flat.numel()
            out1 = torch.empty((n_pairs, 4096), dtype=torch.bfloat16, device="cuda")
            out2 = torch.empty((n_pairs, 4096), dtype=torch.bfloat16, device="cuda")
            s2 = E7_S2[bs]
            res["g1"][0] += lab.time_graph(lambda: m13_gemm1(x, flat, state, E7_S1)) / E5_ROUTINGS
            res["g1"][1] += lab.time_graph(lambda: ext.m13_recon(
                bank["w13"], bank["w13_s"], bank["c13"], flat, x, out1, bank["scr13"], bank["ctr13"], n_pairs, 0,
                bank["rt_w13"], 0, bank["inter"], 1, 2, 1, 4 * sms)) / E5_ROUTINGS
            res["g2"][0] += lab.time_graph(lambda: m13_gemm2(act, flat, state, dict(kind="m13", tt=2, nt=1, c=4))) / E5_ROUTINGS
            res["g2"][1] += lab.time_graph(lambda: ext.m13_recon(
                bank["w2"], bank["w2_s"], bank["c2"], flat, act, out2, bank["scr2"], bank["ctr2"], n_pairs, 0,
                bank["rt_w2"], 1, bank["hidden"], 0, 2, 1, 4 * sms)) / E5_ROUTINGS
        _clean(state)
        log(f"m13 bs{bs}: GEMM1 {mb1:.0f} MB math {res['g1'][0]:.1f} us ({mb1 / res['g1'][0]:.2f} TB/s), loads only "
            f"{res['g1'][1]:.1f} us ({mb1 / res['g1'][1]:.2f}); GEMM2 {mb2:.0f} MB math {res['g2'][0]:.1f} "
            f"({mb2 / res['g2'][0]:.2f}), loads only {res['g2'][1]:.1f} ({mb2 / res['g2'][1]:.2f})")


def _graph_out(fn):
    """fn's output from a CUDA graph replay (capture on a side stream, as the judge does)."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    graph.replay()
    torch.cuda.synchronize()
    res = out.clone()
    del graph
    return res


def _cfg(bs, variant):
    pdl, pf, ppdl = variant
    return dict(s1=dict(E7_S1), s2=dict(E7_S2[bs], pdl=pdl, pf=pf), post=_post(ppdl))


def sec_pdl(lab, state, ref, store):
    for bs in (1, 8, 32):
        x, ids, w = _case(bs, 1500 + bs)
        want = ref.full(x, ids, w)
        parts = []
        for variant in E7_PDL:
            cfg = _cfg(bs, variant)
            try:
                e_eager = _errors(m13_run(x, ids, w, state, cfg), want)
                torch.cuda.synchronize()
                d1 = _clean(state)
                e_graph = _errors(_graph_out(lambda: m13_run(x, ids, w, state, cfg)), want)
                d2 = _clean(state)
                t = lab.time_graph(lambda: m13_run(x, ids, w, state, cfg))
                d3 = _clean(state)
                ok = max(e_eager[0], e_graph[0]) < E5_MAX_ERR and max(e_eager[1], e_graph[1]) < E5_MAX_ROW_ERR
                ok = ok and not (d1 or d2 or d3)
                store.setdefault(bs, {})[variant] = ok
                parts.append(f"{''.join(map(str, variant))}: e{e_eager[0]:.0e}/g{e_graph[0]:.0e}"
                             f"{' DIRTY' if (d1 or d2 or d3) else ''} {t:.1f}us {'ok' if ok else 'BAD'}")
            except Exception as exc:
                store.setdefault(bs, {})[variant] = False
                parts.append(f"{''.join(map(str, variant))}: FAILED {type(exc).__name__}: {_error_text(exc, 120)}")
                _clean(state)
        log(f"pdl bs{bs}: " + " | ".join(parts))


def sec_e2e(lab, state, ref, store):
    cases = [(bs, False, False) for bs in (1, 2, 4, 5, 8, 9, 12, 16, 17, 24, 32, 33, 48, 64)]
    cases += [(bs, True, False) for bs in (4, 8, 16, 32, 64)] + [(16, True, True), (33, False, True), (64, True, True)]
    chosen = {}
    good = [v for v in E7_PDL if all(store.get(b, {}).get(v, False) for b in (1, 8, 32))]
    log(f"e2e: PDL variants correct at bs 1/8/32 eager and in graphs: {[''.join(map(str, v)) for v in good]}")
    for bs in E7_BS:
        if lab.left() < 12:
            log("e2e: out of budget")
            break
        lo = max([E7_CLASS[b] for b in E7_BS if b < bs], default=1)
        hi = E7_CLASS[bs]
        runs = [_case(bs, 1700 + bs + 2000 * r) for r in range(E5_ROUTINGS)]
        t_v1 = t_m11 = 0.0
        m11cfg = m11_config_for(state, bs)
        for x, ids, w in runs:
            data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                    "config": {"bs": bs, "local_start": 0, "local_count": 32}}
            t_v1 += lab.time_graph(lambda: _v1_kernel_fn(data, state)) / E5_ROUTINGS
            if m11cfg is not None:
                t_m11 += lab.time_graph(lambda: m11_run(x, ids, w, state, m11cfg)) / E5_ROUTINGS
        best_other = min(t_v1, t_m11) if m11cfg is not None else t_v1
        variants = []
        for variant in good:
            cfg = _cfg(bs, variant)
            t = sum(lab.time_graph(lambda: m13_run(x, ids, w, state, cfg)) for x, ids, w in runs) / E5_ROUTINGS
            variants.append((t, variant, cfg))
        _clean(state)
        verdict, pick = "M11/v1", None
        for t, variant, cfg in sorted(variants, key=lambda v: v[0]):
            worst = (0.0, 0.0)
            for cbs, skew, outl in cases:
                if lo <= cbs < hi:
                    x, ids, w = _case(cbs, 1300 + cbs + 50 * skew + 25 * outl, skew, outl)
                    err, row = _errors(m13_run(x, ids, w, state, cfg), ref.full(x, ids, w))
                    worst = (max(worst[0], err), max(worst[1], row))
            torch.cuda.synchronize()
            dirty = _clean(state)
            if not dirty and worst[0] < E5_MAX_ERR and worst[1] < E5_MAX_ROW_ERR:
                pick = (t, variant, cfg, worst)
                break
            log(f"e2e bs{bs}: variant {''.join(map(str, variant))} rejected: err {worst[0]:.1e} row {worst[1]:.1e}"
                f"{' DIRTY' if dirty else ''}")
        if pick and pick[0] < 0.97 * best_other:
            chosen[bs] = pick[2]
            verdict = "M13"
        log(f"e2e bs{bs}: v1 {t_v1:.1f}, M11 {t_m11:.1f}, M13 " + " ".join(f"{''.join(map(str, v[1]))}:{v[0]:.1f}" for v in variants)
            + (f" | pick {''.join(map(str, pick[1]))} err {pick[3][0]:.1e} row {pick[3][1]:.1e}" if pick else " | none valid")
            + f" -> {verdict}")
    configs = list(state["m13_configs"])
    configs = [(below, chosen.get(next((b for b, c in E7_CLASS.items() if c == below), None), cfg))
               for below, cfg in configs]
    state["m13_configs"] = tuple(configs)
    for _, cfg in state["m13_configs"]:
        if cfg is not None:
            m13_warm(state, cfg)
    log(f"M13_CONFIGS = {state['m13_configs']!r}")


def prepare(weights):
    t = time.monotonic()
    try:
        state = _submission_prepare(weights)
    except Exception as exc:
        log(f"prepare FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    took = time.monotonic() - t
    log(f"E7: submission prepare (nvcc, banks, warm-up) took {took:.1f}s")
    budget = min(LAB_BUDGET_S, E5_PREPARE_LIMIT_S - took)
    lab = Lab(budget)
    log(f"E7 start, budget {budget:.0f}s")
    if "m13" not in state:
        log(f"E7: M13 off (scales ok {state.get('m13_ok')})")
        return state
    ref = _Ref(weights)
    store = {}
    lab.section("bw", sec_bw, state, min_left=60)
    lab.section("pdl", sec_pdl, state, ref, store, min_left=40)
    lab.section("e2e", sec_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m13_configs')!r}"[:900])
    return state
