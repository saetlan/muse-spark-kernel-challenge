"""E3b: M11 with K split across warps, 2-tile GEMM2 tasks and 8 * nt pair chunks, on the judge's
H200, check runs only. kernel_fn is submission_m11's hybrid; this prepare() decides from its own
measurements which batch sizes go to M11 (v1 otherwise).
  build    nvcc via load_inline (time, and the error lines if it fails); weights shuffled
  attrs    registers, spill bytes, static shared memory per compiled instance
  stage1   GEMM1 (gate+up, SwiGLU) per (nt, depth, ks, CTAs per SM): error vs FP32 at bs 8 and at a
           skewed bs 16 (chunks), cold-L2 graph time at bs 1/8/16/32
  stage2   GEMM2 + post-norm per (tt, nt, depth, ks, CTAs per SM), the same
  e2e      best per batch size vs FP32 (judge-like and skewed routings) and vs v1; M11 only where
           correct and >= 3% faster"""

from nvfp4 import dequantize

LAB_BUDGET_S = 240.0
E3_PREPARE_LIMIT_S = 275.0  # TASK.md: prepare() gets 300 s; keep a margin
E3_BS = (1, 8, 16, 32)
# Each measured batch size stands for token counts below this bound; every bound - 1 fits 8 * nt
# of the nt that batch size can use (1, 1, 2, 4).
E3_CLASS = {1: 5, 8: 9, 16: 17, 32: M11_MAX_TOKENS + 1}
E3_CTAS = (1, 2, 4, 8)
E3_MAX_STAGE_ERR = 1e-2
E3_MAX_ERR, E3_MAX_ROW_ERR = 1e-2, 5e-2  # stricter than the judge's 1.5e-2 and 7.5e-2


def _case(bs, seed, skew=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
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
    """nvcc/ninja error lines from load_inline's exception, which carries the whole build log."""
    lines = [l.strip() for l in str(exc).splitlines()]
    hits = [l for l in lines if "error" in l.lower() or "Error" in l]
    text = " | ".join(hits) if hits else " ".join(lines)
    return text[:keep]


def sec_build(lab, weights, state):
    """prepare() of submission_m11 already compiled and shuffled when its M11_CONFIGS is set."""
    if "m11_ext" not in state:
        t = time.monotonic()
        try:
            state["m11_ext"] = m11_build()
        except Exception as exc:
            log(f"build: nvcc FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
            raise
        log(f"build: nvcc load_inline compiled in {time.monotonic() - t:.1f}s")
    if "m11" not in state:
        state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
        state["m11"] = m11_bank(weights, 0, 32)
        torch.cuda.synchronize()
    log(f"build: {len(M11_INSTANCES)} instances, {state['sms']} SMs")


def _inst_name(gated, tt, nt, d, ks):
    return f"{'g' if gated else 'p'}{tt}.{nt}.{d}.{ks}"


def sec_attrs(lab, state):
    parts = []
    for inst in M11_INSTANCES:
        regs, local, smem = state["m11_ext"].m11_attrs(*inst).tolist()
        parts.append(f"{_inst_name(*inst)} {regs}r{f' {local}B-spill' if local else ''} {smem // 1024}K")
    log("attrs (gated/plain tt.nt.depth.ks): " + ", ".join(parts))


def _name(cfg, stage):
    tt = f"t{cfg['tt']} " if stage == 2 else ""
    return f"{tt}nt{cfg['nt']} d{cfg['d']} k{cfg['ks']} c{cfg['c']}"


def sec_stage(lab, stage, state, ref, store):
    x8, ids8, w8 = _case(8, 908)
    xs, ids_s, w_s = _case(16, 916, skew=True)  # 16 pairs per expert: two chunks at nt 1
    cases = {bs: _case(bs, 900 + bs) for bs in E3_BS}
    if stage == 1:
        checks = [(x8, ids8, w8, ref.act(x8, ids8)), (xs, ids_s, w_s, ref.act(xs, ids_s))]
    else:
        checks = []
        for x, ids, w in ((x8, ids8, w8), (xs, ids_s, w_s)):
            act = ref.act(x, ids).to(torch.bfloat16)
            checks.append((act, ids, w, ref.stage2(act, ids, w)))
        acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in E3_BS}
    results = []
    insts = [i for i in M11_INSTANCES if i[0] == (1 if stage == 1 else 0)]
    for gated, tt, nt, d, ks in insts:
        for ctas in E3_CTAS:
            if lab.left() < (125 if stage == 1 else 45):
                log(f"s{stage}: out of budget")
                break
            cfg = dict(nt=nt, d=d, ks=ks, c=ctas) if stage == 1 else dict(tt=tt, nt=nt, d=d, ks=ks, c=ctas)
            try:
                errs = []
                for a, ids, w, want in checks:
                    flat = ids.reshape(-1)
                    if stage == 1:
                        rows = (flat.long() < 32).nonzero().flatten()
                        errs.append(_errors(m11_gemm1(a, flat, state, cfg)[rows], want[rows])[0])
                    else:
                        errs.append(_errors(m11_post_norm(m11_gemm2(a, flat, state, cfg), flat, w, state, ids.shape[0]),
                                            want)[0])
                err = max(errs)
                if stage == 1:
                    times = {bs: lab.time_graph(lambda c=c: m11_gemm1(c[0], c[1].reshape(-1), state, cfg))
                             for bs, c in cases.items()}
                else:
                    times = {bs: lab.time_graph(lambda a=acts[bs], i=cases[bs][1], w=cases[bs][2]: m11_post_norm(
                        m11_gemm2(a, i.reshape(-1), state, cfg), i.reshape(-1), w, state, i.shape[0])) for bs in E3_BS}
                torch.cuda.synchronize()
            except Exception as exc:  # report and go on; a CUDA fault would also show in every later line
                log(f"s{stage} {_name(cfg, stage)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
                continue
            results.append((cfg, err, times))
            log(f"s{stage} {_name(cfg, stage)}: err {err:.1e} | " + " ".join(f"{t:.1f}" for t in times.values()))
    store[stage] = [r for r in results if r[1] < E3_MAX_STAGE_ERR]
    for bs in E3_BS:
        if store[stage]:
            best = min(store[stage], key=lambda r: r[2][bs])
            log(f"s{stage} best at bs{bs}: {_name(best[0], stage)} {best[2][bs]:.1f} us")


def sec_e2e(lab, state, ref, store):
    s1, s2 = store.get(1, []), store.get(2, [])
    if not s1 or not s2:
        log("e2e: no working stage configs; kernel_fn stays on v1")
        return
    cases = [(bs, False) for bs in (1, 2, 4, 5, 8, 12, 16, 24, 32)] + [(8, True), (16, True), (32, True)]
    chosen = {}
    for bs in E3_BS:
        lo = max([E3_CLASS[b] for b in E3_BS if b < bs], default=1)
        hi = E3_CLASS[bs]
        r1 = min(s1, key=lambda r: r[2][bs])
        r2 = min(s2, key=lambda r: r[2][bs])
        cfg = dict(s1=r1[0], s2=r2[0])
        worst = (0.0, 0.0)
        for cbs, skew in cases:
            if not lo <= cbs < hi:
                continue
            x, ids, w = _case(cbs, 1300 + cbs + 50 * skew, skew)
            err, row = _errors(m11_run(x, ids, w, state, cfg), ref.full(x, ids, w))
            worst = (max(worst[0], err), max(worst[1], row))
        x, ids, w = _case(bs, 1700 + bs)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                "config": {"bs": bs, "local_start": 0, "local_count": 32}}
        t_v1 = lab.time_graph(lambda: _v1_kernel_fn(data, state))
        t_m11 = lab.time_graph(lambda: m11_run(x, ids, w, state, cfg))
        use = worst[0] < E3_MAX_ERR and worst[1] < E3_MAX_ROW_ERR and t_m11 < 0.97 * t_v1
        if use:
            chosen[bs] = cfg
        log(f"e2e bs{bs}: v1 {t_v1:.1f} us, M11 {t_m11:.1f} us ({t_v1 / t_m11:.2f}x) err {worst[0]:.1e} "
            f"row {worst[1]:.1e} -> {'M11' if use else 'v1'} | {cfg}")
    state["m11_configs"] = tuple((E3_CLASS[bs], chosen.get(bs)) for bs in E3_BS)
    for cfg in chosen.values():
        m11_warm(state, cfg)
    log(f"M11_CONFIGS = {state['m11_configs']!r}")


def prepare(weights):
    t = time.monotonic()
    try:
        state = _submission_prepare(weights)
    except Exception as exc:  # nvcc runs inside the submission's prepare when M11_CONFIGS is set
        log(f"prepare FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    took = time.monotonic() - t
    log(f"E3b: submission prepare (nvcc, shuffle, warm-up) took {took:.1f}s")
    budget = min(LAB_BUDGET_S, E3_PREPARE_LIMIT_S - took)  # prepare() as a whole must stay under 300 s
    lab = Lab(budget)
    log(f"E3b start, budget {budget:.0f}s; stage lines: time at bs " + "/".join(map(str, E3_BS)) + " in us")
    ref = _Ref(weights)
    store = {}
    lab.section("build", sec_build, weights, state, min_left=120)
    if "m11_ext" in state:
        lab.section("attrs", sec_attrs, state)
        lab.section("stage1", sec_stage, 1, state, ref, store, min_left=60)
        lab.section("stage2", sec_stage, 2, state, ref, store, min_left=40)
        lab.section("e2e", sec_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m11_configs')!r}"[:700])
    return state
