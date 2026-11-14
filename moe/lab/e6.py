"""E6: M14 (CTA-shared activations, stream-K over CTAs) against v1 and the file's current configs at
32-320 tokens, configs chosen on the mean of 3 routings, on the judge's H200, check runs only.
kernel_fn is submission_m13's hybrid (M13 configs from E5 for up to 32 tokens); this prepare()
decides from its own measurements which classes from 17 tokens up go to M14.
  attrs    registers, spills, static + dynamic shared memory, CTAs per SM of every M14 instance
  stage1   GEMM1 per (nt, CTAs per SM): error vs FP32 at bs 33, skewed bs 64 with outliers, skewed bs 128
           (128 pairs on 8 experts: up to 16 pair passes), bs 257; scratch and counters must be clean
           after every run; cold-L2 graph time at bs 32/64/128/256, 3 routings each; the file's M13
           stage for 17-32 tokens alongside
  stage2   GEMM2 + CUDA post-norm per (tt, nt, CTAs per SM), the same
  e2e      per class: best M14 stages x (GEMM2 PDL, L2 prefetch) x post-norm PDL vs FP32 at the class's sizes
           (judge-like, skewed, outliers, 33, 257) and vs v1 and the file's current config; M14 only where
           correct and >= 3% faster than both"""

from nvfp4 import dequantize, outlier_channels

LAB_BUDGET_S = 240.0
E5_PREPARE_LIMIT_S = 275.0  # TASK.md: prepare() gets 300 s; keep a margin
E5_BS = (32, 64, 128, 256)
E5_CLASS = {32: 33, 64: 65, 128: 192, 256: 321}
E5_ROUTINGS = 3
E5_REPS = 10
E5_MAX_STAGE_ERR = 1e-2
E5_MAX_ERR, E5_MAX_ROW_ERR = 1e-2, 5e-2  # stricter than the judge's 1.5e-2 and 7.5e-2
E5_PDL = ((0, 0), (1, 0), (1, 2), (1, 6))  # (GEMM2 pdl, L2-prefetched groups before the wait)


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


def sec_attrs(lab, state):
    parts = []
    occ = {}
    for g, tt, nt, regs, local, smem, dyn, blocks in state["m14_resources"]:
        occ[(g, tt, nt)] = blocks
        parts.append(f"{'g' if g else 'p'}{tt}.{nt} {regs}r{f' {local}B-spill' if local else ''} "
                     f"{(smem + dyn) // 1024}K {blocks}cta")
    state["m13_occ"] = occ
    log("attrs M14 (gated/plain tt.nt): " + ", ".join(parts) + f"; scales ok {state.get('m13_ok')}")


def _name(cfg, stage):
    tt = f"t{cfg['tt']}" if stage == 2 else ""
    if cfg.get("kind") in ("m13", "m14"):
        return f"{cfg['kind'].upper()} {tt}n{cfg['nt']}c{cfg['c']}"
    return f"M11 {tt}n{cfg['nt']}d{cfg['d']}k{cfg['ks']}c{cfg['c']}"


def _stage_cfgs(state, stage):
    cfgs = []
    for gated, tt, nt in M14_INSTANCES:
        if gated != (stage == 1):
            continue
        for c in range(1, state["m13_occ"][(gated, tt, nt)] + 1):
            base = dict(kind="m14", nt=nt, c=c)
            cfgs.append(base if stage == 1 else dict(base, tt=tt))
    cur = m13_config_for(state, 32)  # the file's 17-32 token stage (M13 from E5), for the comparison at bs32
    if cur is not None:
        cfgs.append({k: v for k, v in cur["s1" if stage == 1 else "s2"].items() if k not in ("pdl", "pf")})
    return cfgs


def _post(pdl=1):
    return dict(kind="m13", pdl=pdl)


def sec_stage(lab, stage, state, ref, store):
    check_cases = [_case(33, 933), _case(64, 964, skew=True, outliers=True), _case(128, 928, skew=True),
                   _case(257, 957)]
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(E5_ROUTINGS)] for bs in E5_BS}
    if stage == 1:
        checks = [(x, ids, w, ref.act(x, ids)) for x, ids, w in check_cases]
    else:
        checks = []
        for x, ids, w in check_cases:
            act = ref.act(x, ids).to(torch.bfloat16)
            checks.append((act, ids, w, ref.stage2(act, ids, w)))
        acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in E5_BS}
    results = []
    for cfg in _stage_cfgs(state, stage):
        if lab.left() < (110 if stage == 1 else 60):
            log(f"s{stage}: out of budget")
            break
        try:
            errs = []
            for a, ids, w, want in checks:
                flat = ids.reshape(-1)
                if cfg.get("kind") == "m13" and ids.shape[0] > M13_MAX_TOKENS:
                    continue
                if stage == 1:
                    rows = (flat.long() < 32).nonzero().flatten()
                    errs.append(_errors(m13_gemm1(a, flat, state, cfg)[rows], want[rows])[0])
                else:
                    y = m13_gemm2(a, flat, state, cfg)
                    errs.append(_errors(m13_post_norm(y, flat, w, state, ids.shape[0], _post(0)), want)[0])
            torch.cuda.synchronize()
            dirty = _clean(state)
            err = max(errs) if not dirty else float("inf")
            times = {}
            for bs, routings in cases.items():
                if cfg.get("kind") == "m13" and bs > M13_MAX_TOKENS:
                    times[bs] = float("inf")
                    continue
                ts = []
                for x, ids, w in routings:
                    if stage == 1:
                        fn = lambda x=x, i=ids: m13_gemm1(x, i.reshape(-1), state, cfg)
                    else:
                        fn = lambda a=acts[bs], i=ids, w=w: m13_post_norm(
                            m13_gemm2(a, i.reshape(-1), state, cfg), i.reshape(-1), w, state, i.shape[0], _post(1))
                    ts.append(lab.time_graph(fn, reps=E5_REPS))
                times[bs] = sum(ts) / len(ts)
            torch.cuda.synchronize()
            late = _clean(state)  # always re-zero, even when the check run was already dirty
            dirty = dirty or late
            if dirty:
                err = float("inf")
        except Exception as exc:  # report and go on; a CUDA fault would also show in every later line
            log(f"s{stage} {_name(cfg, stage)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
            continue
        results.append((cfg, err, times))
        log(f"s{stage} {_name(cfg, stage)} e{err:.0e}{' DIRTY ' + ','.join(dirty) if dirty else ''} | "
            + " ".join(f"{t:.1f}" for t in times.values()))
    store[stage] = [r for r in results if r[1] < E5_MAX_STAGE_ERR]
    for bs in E5_BS:
        m14 = [r for r in store[stage] if r[0].get("kind") == "m14"]
        other = [r for r in store[stage] if r[0].get("kind") != "m14"]
        if m14:
            b = min(m14, key=lambda r: r[2][bs])
            bo = min(other, key=lambda r: r[2][bs]) if other else None
            log(f"s{stage} best at bs{bs}: {_name(b[0], stage)} {b[2][bs]:.1f} us"
                + (f" (current {_name(bo[0], stage)} {bo[2][bs]:.1f})" if bo else ""))


def sec_e2e(lab, state, ref, store):
    s1 = [r for r in store.get(1, []) if r[0].get("kind") == "m14"]
    s2 = [r for r in store.get(2, []) if r[0].get("kind") == "m14"]
    if not s1 or not s2:
        log("e2e: no working M14 stage configs; kernel_fn keeps its configs")
        return
    cases = [(bs, False, False) for bs in (17, 24, 32, 33, 48, 64, 65, 100, 128, 160, 191, 192, 256, 257, 320)]
    cases += [(bs, True, False) for bs in (32, 64, 128, 256)] + [(128, True, True), (257, False, True)]
    chosen = dict((below, cfg) for below, cfg in state["m13_configs"])
    for bs in E5_BS:
        if lab.left() < 12:
            log("e2e: out of budget")
            break
        lo = max([E5_CLASS[b] for b in E5_BS if b < bs], default=1)
        hi = E5_CLASS[bs]
        r1 = min(s1, key=lambda r: r[2][bs])
        r2 = min(s2, key=lambda r: r[2][bs])
        runs = [(x, ids, w) for x, ids, w in (_case(bs, 1700 + bs + 2000 * r) for r in range(E5_ROUTINGS))]
        t_v1 = t_m11 = 0.0
        m11cfg = m13_config_for(state, bs)  # the file's current config for this class (M13 or None)
        for x, ids, w in runs:
            data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                    "config": {"bs": bs, "local_start": 0, "local_count": 32}}
            t_v1 += lab.time_graph(lambda: _v1_kernel_fn(data, state)) / E5_ROUTINGS
            if m11cfg is not None:
                t_m11 += lab.time_graph(lambda: m13_run(x, ids, w, state, m11cfg)) / E5_ROUTINGS
        best_other = min(t_v1, t_m11) if m11cfg is not None else t_v1
        variants = []
        for pdl, pf in E5_PDL:
            for post_pdl in (0, 1):
                cfg = dict(s1=r1[0], s2=dict(r2[0], pdl=pdl, pf=pf), post=_post(post_pdl))
                t = sum(lab.time_graph(lambda: m13_run(x, ids, w, state, cfg)) for x, ids, w in runs) / E5_ROUTINGS
                variants.append((t, cfg))
        _clean(state)
        t_best, cfg = min(variants, key=lambda v: v[0])
        worst = (0.0, 0.0)
        for cbs, skew, outl in cases:
            if not lo <= cbs < hi:
                continue
            x, ids, w = _case(cbs, 1300 + cbs + 50 * skew + 25 * outl, skew, outl)
            err, row = _errors(m13_run(x, ids, w, state, cfg), ref.full(x, ids, w))
            worst = (max(worst[0], err), max(worst[1], row))
        torch.cuda.synchronize()
        dirty = _clean(state)
        use = (not dirty and worst[0] < E5_MAX_ERR and worst[1] < E5_MAX_ROW_ERR and t_best < 0.97 * best_other)
        if use:
            chosen[E5_CLASS[bs]] = cfg
        log(f"e2e bs{bs}: v1 {t_v1:.1f}, current {t_m11:.1f}, M14 {t_best:.1f} us ({best_other / t_best:.2f}x) "
            f"| pdl variants " + " ".join(f"{v[1]['s2']['pdl']}{v[1]['s2']['pf']}{v[1]['post']['pdl']}:{v[0]:.1f}" for v in variants)
            + f" | err {worst[0]:.1e} row {worst[1]:.1e}{' DIRTY' if dirty else ''} -> {'M14' if use else 'current'} | {cfg}")
    state["m13_configs"] = tuple(sorted(chosen.items()))
    for cfg in chosen.values():
        if cfg is not None:
            m13_warm(state, cfg)
    log(f"M13_CONFIGS = {state['m13_configs']!r}")


def prepare(weights):
    t = time.monotonic()
    try:
        state = _submission_prepare(weights)
    except Exception as exc:  # nvcc runs inside the submission's prepare
        log(f"prepare FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    took = time.monotonic() - t
    log(f"E6: submission prepare (nvcc, banks, warm-up) took {took:.1f}s")
    budget = min(LAB_BUDGET_S, E5_PREPARE_LIMIT_S - took)  # prepare() as a whole must stay under 300 s
    lab = Lab(budget)
    log(f"E6 start, budget {budget:.0f}s; stage lines: time at bs " + "/".join(map(str, E5_BS)) + " in us")
    if "m13" not in state:
        log(f"E6: M13/M14 off (scales ok {state.get('m13_ok')}); nothing to measure")
        return state
    ref = _Ref(weights)
    store = {}
    lab.section("attrs", sec_attrs, state)
    if "m13_occ" in state:
        lab.section("stage1", sec_stage, 1, state, ref, store, min_left=60)
        lab.section("stage2", sec_stage, 2, state, ref, store, min_left=40)
        lab.section("e2e", sec_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m13_configs')!r}"[:900])
    return state
