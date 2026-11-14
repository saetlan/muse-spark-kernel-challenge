"""E3: M11 (moe/m11/m11.py, CUDA) on the judge's H200, check runs only. kernel_fn is submission_m11's
hybrid; this prepare() decides from its own measurements which batch sizes go to M11 (v1 otherwise).
  build    nvcc via load_inline (time, and the error lines if it fails); weights shuffled
  attrs    registers, spill bytes, static shared memory per compiled instance
  stage1   GEMM1 (gate+up, SwiGLU) per (nt, depth, CTAs per SM): error vs FP32, cold-L2 graph time
  stage2   GEMM2 + post-norm, the same
  e2e      best per batch size vs FP32 (judge-like and skewed routings) and vs v1; M11 only where
           correct and >= 3% faster"""

from nvfp4 import dequantize

LAB_BUDGET_S = 240.0
E3_BS = (1, 8, 16, 32)
# Each measured batch size stands for token counts below this bound; every bound - 1 fits 8 * nt
# of the nt that batch size can use (1, 1, 2, 4).
E3_CLASS = {1: 5, 8: 9, 16: 17, 32: M11_MAX_TOKENS + 1}
E3_CTAS = (1, 2, 4)
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
    t = time.monotonic()
    try:
        state["m11_ext"] = m11_build()
    except Exception as exc:
        log(f"build: nvcc FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    log(f"build: nvcc load_inline compiled in {time.monotonic() - t:.1f}s")
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    t = time.monotonic()
    state["m11"] = m11_bank(weights, 0, 32)
    torch.cuda.synchronize()
    log(f"build: weights shuffled in {time.monotonic() - t:.1f}s")


def sec_attrs(lab, state):
    parts = []
    for gated in (1, 0):
        for nt, d in M11_INSTANCES:
            regs, local, smem = state["m11_ext"].m11_attrs(gated, nt, d).tolist()
            parts.append(f"{'g' if gated else 'p'}{nt}/{d}: {regs}r {local}B-spill {smem}B-smem")
    log("attrs: " + ", ".join(parts))


def _cfg(nt, d, ctas):
    return dict(nt=nt, d1=d, d2=d, c1=ctas, c2=ctas)


def _name(cfg, stage):
    return f"nt{cfg['nt']} d{cfg['d1'] if stage == 1 else cfg['d2']} c{cfg['c1'] if stage == 1 else cfg['c2']}"


def sec_stage(lab, stage, state, ref, store):
    x8, ids8, w8 = _case(8, 908)
    cases = {bs: _case(bs, 900 + bs) for bs in E3_BS}
    if stage == 1:
        want8 = ref.act(x8, ids8)
        rows8 = (ids8.reshape(-1).long() < 32).nonzero().flatten()
    else:
        act8 = ref.act(x8, ids8).to(torch.bfloat16)
        want8 = ref.stage2(act8, ids8, w8)
        acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in E3_BS}
    results = []
    for nt, d in M11_INSTANCES:
        for ctas in E3_CTAS:
            if lab.left() < (130 if stage == 1 else 45):
                log(f"s{stage}: out of budget")
                break
            cfg = _cfg(nt, d, ctas)
            try:
                if stage == 1:
                    got = m11_gemm1(x8, ids8.reshape(-1), state, cfg)
                    err, _ = _errors(got[rows8], want8[rows8])
                    run = lambda x, ids, w: m11_gemm1(x, ids.reshape(-1), state, cfg)
                    times = {bs: lab.time_graph(lambda c=c: run(*c)) for bs, c in cases.items() if bs <= 8 * nt}
                else:
                    ids_f = ids8.reshape(-1)
                    got = m11_post_norm(m11_gemm2(act8, ids_f, state, cfg), ids_f, w8, state, 8)
                    err, _ = _errors(got, want8)
                    run = lambda a, ids, w: m11_post_norm(m11_gemm2(a, ids.reshape(-1), state, cfg), ids.reshape(-1), w,
                                                          state, ids.shape[0])
                    times = {bs: lab.time_graph(lambda a=acts[bs], i=cases[bs][1], w=cases[bs][2]: run(a, i, w))
                             for bs in E3_BS if bs <= 8 * nt}
                torch.cuda.synchronize()
            except Exception as exc:  # report and go on; a CUDA fault would also show in every later line
                log(f"s{stage} {_name(cfg, stage)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
                continue
            results.append((cfg, err, times))
            log(f"s{stage} {_name(cfg, stage)}: err {err:.1e} | " + " ".join(f"bs{b} {t:.1f}" for b, t in times.items()) + " us")
    store[stage] = [r for r in results if r[1] < E3_MAX_STAGE_ERR]


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
        best = None  # one nt serves both GEMMs of a call; depth and CTAs per SM are per GEMM
        for nt in sorted({r[0]["nt"] for r in s1} & {r[0]["nt"] for r in s2}):
            if 8 * nt < hi - 1:
                continue
            f1 = [r for r in s1 if r[0]["nt"] == nt and bs in r[2]]
            f2 = [r for r in s2 if r[0]["nt"] == nt and bs in r[2]]
            if not f1 or not f2:
                continue
            r1, r2 = min(f1, key=lambda r: r[2][bs]), min(f2, key=lambda r: r[2][bs])
            total = r1[2][bs] + r2[2][bs]
            if best is None or total < best[0]:
                best = (total, dict(nt=nt, d1=r1[0]["d1"], c1=r1[0]["c1"], d2=r2[0]["d2"], c2=r2[0]["c2"]))
        if best is None:
            log(f"e2e bs{bs}: no config covers up to {hi - 1} tokens -> v1")
            continue
        cfg = best[1]
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
    state = _submission_prepare(weights)
    lab = Lab(LAB_BUDGET_S)
    log(f"E3 start, budget {LAB_BUDGET_S:.0f}s")
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
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m11_configs')!r}"[:500])
    return state
