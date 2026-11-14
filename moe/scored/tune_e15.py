"""Self-tuning prepare (appended after the M13 section, whose prepare becomes _m13_prepare).

The queue makes lab -> config -> scored-run cycles slow, so prepare() measures its own options on the
judge's GPU, on synthetic inputs only (nothing derived from any call's inputs is ever kept), and picks per
token class the fastest config that passes an FP32 check stricter than the judge's at every size of the
class (judge-like, skewed and outlier routings): M11's E3c configs and v1 are the incumbents; M13 (1-64
tokens) and M14 (33-320 tokens) with or without PDL and with an L2 prefetch distance are the candidates.
Any failure leaves the incumbents in place. A compact log (bandwidth recon, every timing, every
decision) is printed for the run's log tail. Budgeted so prepare() as a whole stays under ~250 s."""

import statistics
import time

TUNE_BUDGET_S = 170.0
TUNE_PREPARE_LIMIT_S = 255.0  # prepare() gets 300 s
TUNE_REPS = 20  # E8/E9 tuned in ~5 s of the 170 s budget: sample more
TUNE_ROUTINGS = 3
TUNE_MAX_ERR, TUNE_MAX_ROW_ERR = 1e-2, 5e-2  # the judge allows 1.5e-2 and 7.5e-2
TUNE_MARGIN = 0.95  # a candidate must be >= 5% faster than the incumbent (E12: +4-5% in-tuner was -2% judged)
TUNE_MARGIN_PDL = 0.97  # the incumbent's own kernels with a PDL tail (same kernels, so no cross-kernel bias)
TUNE_FLUSH_BYTES = 512 << 20
# class upper bound -> representative batch size
TUNE_CLASSES = ((5, 1), (9, 8), (17, 16), (33, 32), (65, 64), (192, 128), (321, 256))
# M13 candidates (E5's best stages, MF8) x L2 prefetch distance x PDL (GEMM2, post-norm)
TUNE_M13_S1 = (dict(kind="m13", nt=1, c=4),)
TUNE_M13_S2 = (dict(kind="m13", tt=2, nt=1, c=4),)  # E8: t1n1c5 lost to t2n1c4 in every class
# Start-up L2 prefetches (E10): npre 255 = the warp's own next groups (lane-parallel, at kernel start);
# nxq = GEMM1 prefetches the first nxq/4 of each GEMM2 warp's weight range (GEMM2 then starts from L2).
TUNE_M13_S1_PF = (dict(npre=0, nxq=0),)  # E12: every start-up prefetch was slower (MF12/MF13)
TUNE_M13_S2_PF = (dict(npre=0),)
TUNE_PFD = (0,)  # E8: L2 prefetch-ahead 2/4 was slower everywhere (MF10)
TUNE_PDL = ((0, 0), (1, 1))


def tlog(msg):
    print(f"[tune] {msg}", flush=True)


class _Tuner:
    def __init__(self, budget):
        self.t0 = time.monotonic()
        self.budget = budget
        self.flush = None

    def left(self):
        return self.budget - (time.monotonic() - self.t0)

    def time_graph(self, fn, reps=TUNE_REPS):
        """Median us of a CUDA-graph replay of fn, each after an L2 flush (as the judge times)."""
        if self.flush is None:
            self.flush = torch.empty(TUNE_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            self.flush.sum()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)


def _routing(bs, seed):
    """The reference's default routing (top-8 of uniform + heavy-tailed popularity, busiest GPU = 0..31)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((bs, 256), generator=gen, device="cuda")
    z = torch.randn((256,), generator=gen, device="cuda")
    u = torch.rand((256,), generator=gen, device="cuda").clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    vals, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return ids.to(torch.int32).contiguous(), (vals / vals.sum(-1, keepdim=True)).float().contiguous()


def _case(bs, seed, skew=False, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if outliers:  # 1% of channels scaled 30x, like the judge's skewed test
        cols = torch.randperm(4096, generator=gen, device="cuda")[:41]
        x[:, cols] = (x[:, cols].float() * 30.0).to(torch.bfloat16)
    if skew:  # every token on experts 0-7
        ids = torch.arange(TOP_K, dtype=torch.int32, device="cuda").repeat(bs, 1).contiguous()
        w = torch.rand((bs, TOP_K), generator=gen, device="cuda")
        w = (w / w.sum(-1, keepdim=True)).contiguous()
    else:
        ids, w = _routing(bs, seed)
    return x, ids, w


class _Ref:
    """FP32 reference from the raw NVFP4 weights (as reference.py), experts dequantized on demand."""

    def __init__(self, weights):
        self.w = weights
        self.cache = {}

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (_tune_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                             _tune_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.cache[e]

    def full(self, x, ids, tw):
        flat = ids.reshape(-1).long()
        nw = 1.0 + self.w["post_norm_w"].float()
        out = torch.zeros((ids.shape[0], 4096), device="cuda")
        for e in [e for e in torch.unique(flat).tolist() if e < 32]:
            p = (flat == e).nonzero().flatten()
            h = x[p // TOP_K].float() @ self.expert(e)[0].T
            act = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]  # FP32, as reference.ref_kernel
            y = (act @ self.expert(e)[1].T) * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, p // TOP_K, y * tw.reshape(-1)[p].unsqueeze(1))
        return out


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _tune_dequant(packed, sf, gs):
    """NVFP4 -> FP32 [rows, K] (e2m1 * e4m3 block scale * global scale), as nvfp4.dequantize."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 0x7).long()]
    vals = torch.where((codes & 0x8) != 0, -mag, mag)
    rows, k = vals.shape
    return (vals.view(rows, k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)).view(rows, k)


def _errors(out, ref):
    out, ref = out.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    diff = out - ref
    err = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    return err, float((diff.norm(dim=-1) / torch.maximum(rows_ref, floor)).max())


def _clean(state):
    bank = state["m13"]
    dirty = [k for k in ("scr13", "scr2", "ctr13", "ctr2") if bool(bank[k].ne(0).any())]
    for k in dirty:
        bank[k].zero_()
    return dirty


def _run(state, cfg, x, ids, w):
    if cfg is None:
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                "config": {"bs": x.shape[0], "local_start": 0, "local_count": 32}}
        return _m11_kernel_fn(data, state)
    return m13_run(x, ids, w, state, cfg)


def _sname(st):
    if st.get("kind") in (None, "m11p"):  # M11's own stage dicts (its E3c configs)
        return (f"{st.get('kind', 'm11')}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}d{st['d']}k{st['ks']}"
                f"c{st['c']}")
    extra = f"b{st['beta']}" if st["kind"] in ("m15", "m17") else f"pf{st.get('pfd', 0)}"
    if st.get("p"):
        extra += f"P{st['p']}"
    if st.get("db"):
        extra += "D"
    if st.get("npre"):
        extra += f"s{st['npre']}"
    if st.get("nxq"):
        extra += f"x{st['nxq']}"
    return f"{st['kind']}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}c{st['c']}{extra}"


def _name(cfg):
    if cfg is None:
        return "incumbent"
    s1, s2 = cfg["s1"], cfg["s2"]
    return f"{_sname(s1)}+{_sname(s2)} pdl{s2.get('pdl', 0)}{cfg['post']['pdl']}"


def _valid(state, ref, cfg, lo, hi):
    """Worst (error, row error) over the class's sizes and routings; None if scratch was left dirty."""
    sizes = sorted({b for b in (1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 191, 192, 256, 257, 320) if lo <= b < hi})
    cases = [(b, False, False) for b in sizes] + [(b, True, False) for b in sizes] + [(sizes[-1], True, True)]
    worst = (0.0, 0.0)
    for b, skew, outl in cases:
        x, ids, w = _case(b, 3100 + b + 50 * skew + 25 * outl, skew, outl)
        err, row = _errors(_run(state, cfg, x, ids, w), ref.full(x, ids, w))
        worst = (max(worst[0], err), max(worst[1], row))
    torch.cuda.synchronize()
    return None if _clean(state) else worst


def _bw_recon(tuner, state):
    """Bandwidth recon for the log: torch sum and a plain streaming kernel over contiguous expert codes; M13
    GEMM1 at about bs1 and bs8 and M14 GEMM1 (n2 c3 P4) at bs128 and bs256, each full and in the recon modes
    (loads only / decode without MMA / MMA without decode), so the log says what binds at each size."""
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    len16 = bank["w13"][0].numel() // 4
    parts = []
    for n in (3, 9):
        mb = n * len16 * 16 / 1e6
        t_torch = tuner.time_graph(lambda: bank["w13"][:n].view(torch.int64).sum())
        experts = torch.arange(n, dtype=torch.int32, device="cuda")
        best = max((mb / tuner.time_graph(lambda: ext.m13_stream(bank["w13"], experts, len16, sink, u, c * sms)), u, c)
                   for u in (8, 16) for c in (4, 8, 16))
        parts.append(f"{mb:.0f}MB torch {mb / t_torch:.2f} stream {best[0]:.2f} (U{best[1]} c{best[2]})")
    tlog("bw TB/s (incl. graph launch): " + "; ".join(parts))
    for bs in (128, 256):  # E12 recorded M13's (bs1/8) and M14's (bs128/256) splits; now M16 g2.4 P4 DB's
        x, ids, w = _case(bs, 900 + bs)
        flat = ids.reshape(-1)
        ne = int((torch.bincount(flat.long(), minlength=256)[:32] > 0).sum())
        mb = ne * 18.87
        out = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")
        st = dict(kind="m16", nt=4, c=2, p=4, db=1)
        t = [tuner.time_graph(lambda: m13_gemm1(x, flat, state, st))]
        t += [tuner.time_graph(lambda m=m: ext.m16_recon(bank["w13n"], bank["w13n_s"], bank["c13"], flat, x, out, bank["scr13"],
                                                        bank["ctr13"], flat.numel(), 0, bank["rt_w13"], 2 * sms, m))
              for m in (1, 2)]
        _clean(state)
        tlog(f"recon bs{bs} GEMM1 m16 g2.4P4D {mb:.0f}MB ({ne} experts), us (TB/s): full {t[0]:.1f} ({mb / t[0]:.2f}) | "
             f"no-wgmma {t[1]:.1f} ({mb / t[1]:.2f}) | no-decode {t[2]:.1f} ({mb / t[2]:.2f})")
    _clean(state)


def _m14_stage_search(tuner, state, bss):
    """Best large-batch GEMM1 and GEMM2 stage per batch size (cold-L2 graph time, 2 routings), over the
    compiled M14 and M15 instances x CTAs per SM up to the measured occupancy (M14 with L2 prefetch 2 groups
    ahead, E6/MF9; M15 with cost weight 0 or 4)."""
    variants = []
    for kind in ("m14", "m17", "m16", "m15"):  # M14 first: the proven one is never cut by the budget
        res = state.get(kind + "_resources", [])
        tlog(f"{kind} attrs: " + ", ".join(f"{'g' if g else 'p'}{tt}.{nt}P{pr} "
                                          f"{regs}r{f' {local}B' if local else ''} {(smem + dyn) // 1024}K {blocks}cta"
                                          for g, tt, nt, pr, regs, local, smem, dyn, blocks in res))
        for g, tt, nt, pr, regs, local, smem, dyn, blocks in res:
            pa, db = pr & 15, pr >> 4  # M16/M17 rows carry P | DB << 4
            for c in range(1, blocks + 1):
                for extra in (tuple(dict(beta=b, p=pa, db=db) for b in (0, 4)) if kind in ("m15", "m17")
                              else (dict(pfd=0, p=pa, db=db),)):
                    st = dict(kind=kind, nt=nt, c=c, **extra)
                    if not g:
                        st["tt"] = tt
                    variants.append((g, st))
    best = {1: {}, 2: {}}
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)] for bs in bss}
    acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in bss}
    rows = []
    for g, st in variants:
        if tuner.left() < 60:
            tlog(f"stage search out of budget after {len(rows)} of {len(variants)} variants")
            break
        stage = 1 if g else 2
        line = {}
        for bs in bss:
            if st["kind"] == "m14" and bs < 64:  # E6: M14 lost to M13 below 64 tokens
                continue
            if st["kind"] in ("m15", "m17") and st["nt"] <= 2 and bs > 32:  # their small-batch instances
                continue
            ts = []
            for x, ids, w in cases[bs]:
                flat = ids.reshape(-1)
                fn = (lambda x=x, f=flat: m13_gemm1(x, f, state, st)) if g else \
                     (lambda a=acts[bs], f=flat: m13_gemm2(a, f, state, st))
                ts.append(tuner.time_graph(fn))
            t = sum(ts) / len(ts)
            line[bs] = t
            best[stage].setdefault(bs, []).append((t, st))
        _clean(state)
        rows.append((stage, st, line))
    # the log keeps variants within 1.25x of their stage's best at some size (the output tail is 16,000 chars)
    top = {(stage, bs): min(t for t, _ in v) for stage in (1, 2) for bs, v in best[stage].items()}
    shown = 0
    for stage, st, line in rows:  # M16/M17 within 1.5x of the best somewhere (E13 logged them all), others 1.25x
        lim = 1.5 if st["kind"] in ("m16", "m17") else 1.25
        if any(t <= lim * top[(stage, bs)] for bs, t in line.items()):
            shown += 1
            tlog(f"s{stage} {_sname(st)} | " + " ".join(f"{bs}:{t:.0f}" for bs, t in line.items()))
    tlog(f"stage search: {len(rows)} variants timed, {shown} shown")
    for stage in (1, 2):  # keep the 2 fastest per batch size, so one wrong stage can't block the others
        for bs in best[stage]:
            best[stage][bs] = sorted(best[stage][bs], key=lambda v: v[0])[:2]
    return best


def tune(state, weights, deadline):
    tuner = _Tuner(min(TUNE_BUDGET_S, deadline - time.monotonic()))
    ref = _Ref(weights)
    tlog(f"start, budget {tuner.budget:.0f}s; incumbents: M11 E3c up to 32 tokens, v1 above")
    try:
        _bw_recon(tuner, state)
    except Exception as exc:
        tlog(f"bw recon FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    m14 = {1: {}, 2: {}}
    try:
        m14 = _m14_stage_search(tuner, state, (1, 8, 16, 32, 64, 128, 256))
    except Exception as exc:
        tlog(f"m14 search FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    configs = dict(state["m13_configs"])
    lo = 1
    for hi, bs in TUNE_CLASSES:
        if tuner.left() < 15:
            tlog("out of budget; remaining classes keep their incumbents")
            break
        cands = []
        if bs <= M13_MAX_TOKENS:
            for s1 in TUNE_M13_S1:
                for s2 in TUNE_M13_S2:
                    for pfd in TUNE_PFD:
                        for p1 in TUNE_M13_S1_PF:
                            for p2 in TUNE_M13_S2_PF:
                                for pdl, ppdl in TUNE_PDL:
                                    cands.append(dict(s1=dict(s1, pfd=pfd, **p1),
                                                      s2=dict(s2, pfd=pfd, pdl=pdl, pf=0, **p2),
                                                      post=dict(kind="m13", pdl=ppdl)))
        if bs in m14[1] and bs in m14[2]:  # M15 at every size, M14 from 64 tokens
            for _, s1b in m14[1][bs]:
                for _, s2b in m14[2][bs]:
                    for pfd in (TUNE_PFD if "m14" in (s1b["kind"], s2b["kind"]) else (0,)):
                        for pdl, ppdl in TUNE_PDL:
                            s1c = dict(s1b, pfd=pfd) if s1b["kind"] == "m14" else dict(s1b)
                            s2c = dict(s2b, pfd=pfd) if s2b["kind"] == "m14" else dict(s2b)
                            cands.append(dict(s1=s1c, s2=dict(s2c, pdl=pdl, pf=0), post=dict(kind="m13", pdl=ppdl)))
        inc11 = m11_config_for(state, bs)
        if inc11 is not None:  # the M11 incumbent with a PDL tail: GEMM2 + post-norm, or the post-norm only (E15)
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"], kind="m11p"), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"]), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
        runs = [_case(bs, 1700 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)]
        try:
            t_inc = sum(tuner.time_graph(lambda: _run(state, None, x, ids, w)) for x, ids, w in runs) / TUNE_ROUTINGS
            timed = []
            for cfg in cands:
                if tuner.left() < 15:
                    break
                timed.append((sum(tuner.time_graph(lambda: _run(state, cfg, x, ids, w)) for x, ids, w in runs)
                              / TUNE_ROUTINGS, cfg))
                _clean(state)
            pick, why = None, "no candidate faster"
            for t, cfg in sorted(timed, key=lambda v: v[0]):
                if t >= cfg.get("margin", TUNE_MARGIN) * t_inc:
                    continue
                worst = _valid(state, ref, cfg, lo, hi)
                if worst is not None and worst[0] < TUNE_MAX_ERR and worst[1] < TUNE_MAX_ROW_ERR:
                    pick, why = (t, cfg, worst), "ok"
                    break
                tlog(f"class [{lo},{hi}) rejected {_name(cfg)} {t:.1f}us: "
                     + ("dirty scratch" if worst is None else f"err {worst[0]:.1e} row {worst[1]:.1e}"))
            grid = [(c["s1"].get("npre", 0), c["s1"].get("nxq", 0), c["s2"].get("npre", 0), c["s2"].get("pdl", 0), t)
                    for t, c in timed if c["s1"].get("kind") == "m13"]
            if grid:  # every start-up prefetch combination: s1 npre/nxq, then s2 npre 0/255 x pdl 0/1
                cells = {}
                for n1, q, n2, pdl, t in grid:
                    cells.setdefault((n1, q), {})[(n2, pdl)] = t
                tlog(f"class [{lo},{hi}) m13 s1(npre,nxq): s2 npre0 pdl0/1, npre255 pdl0/1 | " + " ".join(
                    f"{n1},{q}:" + "/".join(f"{v[k]:.1f}" if k in v else "-" for k in ((0, 0), (0, 1), (255, 0), (255, 1)))
                    for (n1, q), v in sorted(cells.items())))
            top = " ".join(f"{_name(c)}={t:.1f}" for t, c in sorted(timed, key=lambda v: v[0])[:4])
            if pick:
                configs[hi] = pick[1]
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us -> {_name(pick[1])} {pick[0]:.1f}us "
                     f"({t_inc / pick[0]:.2f}x, err {pick[2][0]:.1e}) | best: {top}")
            else:
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us kept ({why}) | best: {top}")
        except Exception as exc:
            tlog(f"class [{lo},{hi}) FAILED {type(exc).__name__}: {str(exc)[-200:]}; incumbent kept")
            _clean(state)
        lo = hi
    state["m13_configs"] = tuple(sorted(configs.items()))
    ref.cache.clear()
    tuner.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    tlog(f"done in {time.monotonic() - tuner.t0:.0f}s; M13_CONFIGS = {state['m13_configs']!r}"[:1500])


def prepare(weights):
    t0 = time.monotonic()
    state = _m13_prepare(weights)
    tlog(f"base prepare (v1, nvcc, banks, warm-up) took {time.monotonic() - t0:.1f}s")
    if "m13" in state:
        try:
            tune(state, weights, t0 + TUNE_PREPARE_LIMIT_S)
        except Exception as exc:  # never fail prepare: the incumbents stay
            tlog(f"tune FAILED {type(exc).__name__}: {str(exc)[-300:]}")
            state["m13_configs"] = M13_CONFIGS
        for _, cfg in state["m13_configs"]:
            if cfg is not None:
                m13_warm(state, cfg)
        # (E14's verbose ptxas probe build is gone: its output filled the 16,000-char log tail; result in MF15)
        tlog(f"prepare done in {time.monotonic() - t0:.0f}s")
    return state
