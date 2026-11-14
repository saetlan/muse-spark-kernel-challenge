"""E2: M2 (moe/m2/m2.py) on the judge's H200, check runs only. kernel_fn is submission_m2's hybrid;
this prepare() decides from its own measurements which batch sizes go to M2 (v1 otherwise).
  build    M2 codes repacked for BK 128 too (prepare built 256)
  stage1   GEMM1 (+ split-K reduce) variants: compile, codegen stats, error at bs 8 vs FP32,
           cold-L2 CUDA-graph time at bs 1/8/16/32
  stage2   GEMM2 + post-norm variants, the same
  correct  the chosen configs vs FP32 at bs 1-32, judge-like and skewed routings
  e2e      chosen M2 vs v1 per batch size; M2 only where correct and >= 3% faster"""

import math

from nvfp4 import dequantize

LAB_BUDGET_S = 235.0
E2_BS = (1, 8, 16, 32)
# Each measured batch size stands for token counts below this bound.
E2_CLASS = {1: 5, 8: 13, 16: 25, 32: M2_MAX_TOKENS + 1}
E2_BASE = dict(persist=True, swap=True, bn=64, bk=256, split=1, warps=4, stages=3, progs_per_sm=2)
E2_MAX_STAGE_ERR = 1e-2
# Seconds of lab budget a stage's variant loop leaves for what comes after it (arbitrary split).
E2_RESERVE_S = {1: 120.0, 2: 35.0}
E2_MAX_ERR, E2_MAX_ROW_ERR = 1e-2, 5e-2  # stricter than the judge's 1.5e-2 and 7.5e-2


def _name(cfg):
    return (f"{'P' if cfg['persist'] else 'G'}{'S' if cfg['swap'] else 'N'}{'2' if cfg.get('scale2d') else ''} "
            f"n{cfg['bn']} k{cfg['bk']} s{cfg['split']} w{cfg['warps']} st{cfg['stages']} p{cfg['progs_per_sm']}")


def _codegen(handle):
    try:
        g, p = handle.asm.get("ttgir", ""), handle.asm.get("ptx", "")
        return (f"regs {handle.n_regs} spill {handle.n_spills} smem {handle.metadata.shared} | "
                f"wgdot {g.count('ttng.warp_group_dot')} dot {g.count('tt.dot ')} cvt {g.count('ttg.convert_layout')} "
                f"lload {g.count('ttg.local_load')} acp {g.count('async_copy_global_to_local')} | "
                f"wgmma {p.count('wgmma.mma_async')} mma {p.count('mma.sync')} cpa {p.count('cp.async.')} "
                f"ldm {p.count('ldmatrix')} sts {p.count('st.shared')} lds {p.count('ld.shared')}")
    except Exception as exc:  # stats are a bonus; never fail the variant over them
        return f"codegen n/a ({type(exc).__name__})"


def _case(bs, seed, skew=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if skew:  # every token on experts 0-7: bs pairs per expert, so two 16-pair chunks at bs 32
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
    finite = bool(torch.isfinite(out).all())
    return (err if finite else float("inf")), row


def sec_build(lab, weights, state):
    bank = state["m2"]
    for bk in (128, 256):
        if ("w13", bk) not in bank:
            bank[("w13", bk)] = torch.stack([m2_repack(weights["w13"][e], bk) for e in range(32)])
            bank[("w2", bk)] = torch.stack([m2_repack(weights["w2"][e], bk) for e in range(32)])
    log(f"build: layouts {sorted(k[1] for k in bank if isinstance(k, tuple) and k[0] == 'w13')}")


def _launch_stats(stage, cfg, state, x, ids):
    """One direct GEMM launch, for Triton's compiled-kernel handle."""
    bank, sms, n_pairs = state["m2"], state["sms"], ids.numel()
    flat = ids.reshape(-1)
    if stage == 1:
        width = 4096 if cfg["split"] == 1 else 2 * 4096
        out = torch.empty((cfg["split"], n_pairs, width) if cfg["split"] > 1 else (n_pairs, width),
                          dtype=torch.bfloat16 if cfg["split"] == 1 else torch.float32, device="cuda")
        return m2_gemm(x, bank, "w13", out, flat, n_pairs, True, False, cfg, sms)
    out = torch.empty((cfg["split"], n_pairs, 4096) if cfg["split"] > 1 else (n_pairs, 4096),
                      dtype=torch.bfloat16 if cfg["split"] == 1 else torch.float32, device="cuda")
    return m2_gemm(x, bank, "w2", out, flat, n_pairs, False, True, cfg, sms)


def _try_variant(lab, stage, cfg, state, ref, cases, check_case, results):
    bank, sms = state["m2"], state["sms"]
    x8, ids8, w8, want8 = check_case
    try:
        if stage == 1:
            run = lambda x, ids, w: m2_stage1(x, ids.reshape(-1), bank, cfg, sms)
            got = run(x8, ids8, w8)
            flat = ids8.reshape(-1).long()
            rows = (flat < 32).nonzero().flatten()
            err, _ = _errors(got[rows], want8[rows])
        else:
            run = lambda a, ids, w: m2_stage2(a, ids.reshape(-1), w, state["norm_w"], bank, cfg, sms, ids.shape[0])
            got = run(x8, ids8, w8)
            err, _ = _errors(got, want8)
        torch.cuda.synchronize()
        stats = _codegen(_launch_stats(stage, cfg, state, x8, ids8))
        times = {bs: lab.time_graph(lambda a=a, i=i, w=w: run(a, i, w)) for bs, (a, i, w) in cases.items()}
    except Exception as exc:  # a variant that fails to compile or run only costs its own line
        log(f"s{stage} {_name(cfg)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
        return
    results.append((cfg, err, times))
    log(f"s{stage} {_name(cfg)}: err {err:.1e} | " + " ".join(f"bs{b} {t:.1f}" for b, t in times.items())
        + f" us | {stats}")


def _refine(best):
    return [dict(best, persist=False), dict(best, split=2), dict(best, split=4), dict(best, stages=2),
            dict(best, stages=4), dict(best, warps=8), dict(best, progs_per_sm=1), dict(best, progs_per_sm=4)]


def _geo(times):
    return math.exp(sum(math.log(t) for t in times.values()) / len(times))


def sec_stage(lab, stage, state, ref, store):
    x8, ids8, w8 = _case(8, 908)
    if stage == 1:
        cases = {bs: _case(bs, 900 + bs) for bs in E2_BS}
        check = (x8, ids8, w8, ref.act(x8, ids8))
    else:
        cases = {}
        for bs in E2_BS:
            x, ids, w = _case(bs, 900 + bs)
            cases[bs] = (torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16), ids, w)
        act8 = ref.act(x8, ids8).to(torch.bfloat16)
        check = (act8, ids8, w8, ref.stage2(act8, ids8, w8))
    results = []
    grid = [dict(E2_BASE, swap=s, bn=n, bk=k) for s in (True, False) for n in (64, 128) for k in (128, 256)]
    # Hedges in case the 3-D scale broadcast or the persistent loop doesn't compile.
    grid += [dict(E2_BASE, scale2d=True), dict(E2_BASE, swap=False, scale2d=True), dict(E2_BASE, persist=False)]
    reserve = E2_RESERVE_S[stage]  # leave the later sections their share of the budget
    for cfg in grid:
        if lab.left() < reserve:
            log(f"s{stage}: out of budget before {_name(cfg)}")
            break
        _try_variant(lab, stage, cfg, state, ref, cases, check, results)
    good = [r for r in results if r[1] < E2_MAX_STAGE_ERR]
    if good:
        best = min(good, key=lambda r: _geo(r[2]))[0]
        for cfg in _refine(best):
            if lab.left() < reserve + 10:
                break
            _try_variant(lab, stage, cfg, state, ref, cases, check, results)
    store[stage] = [r for r in results if r[1] < E2_MAX_STAGE_ERR]


def _best_per_bs(results):
    return {bs: min(results, key=lambda r: r[2][bs])[0] for bs in E2_BS} if results else {}


def sec_correct_and_e2e(lab, state, ref, store):
    best1, best2 = _best_per_bs(store.get(1, [])), _best_per_bs(store.get(2, []))
    if not best1 or not best2:
        log("e2e: no working stage configs; kernel_fn stays on v1")
        return
    cases = [(bs, False) for bs in (1, 2, 4, 8, 12, 16, 24, 32)] + [(32, True), (5, True)]
    chosen = {}
    for bs in E2_BS:
        cfg1, cfg2 = best1[bs], best2[bs]
        worst = (0.0, 0.0)
        lo = max([E2_CLASS[b] for b in E2_BS if b < bs], default=1)
        for cbs, skew in cases:
            if not lo <= cbs < E2_CLASS[bs]:  # the token counts this class's configs will serve
                continue
            x, ids, w = _case(cbs, 1300 + cbs + 50 * skew, skew)
            got = m2_run(x, ids, w, state, cfg1, cfg2)
            err, row = _errors(got, ref.full(x, ids, w))
            worst = (max(worst[0], err), max(worst[1], row))
        x, ids, w = _case(bs, 1700 + bs)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                "config": {"bs": bs, "local_start": 0, "local_count": 32}}
        t_v1 = lab.time_graph(lambda: _v1_kernel_fn(data, state))
        t_m2 = lab.time_graph(lambda: m2_run(x, ids, w, state, cfg1, cfg2))
        ok = worst[0] < E2_MAX_ERR and worst[1] < E2_MAX_ROW_ERR
        use = ok and t_m2 < 0.97 * t_v1
        if use:
            chosen[bs] = (cfg1, cfg2)
        log(f"e2e bs{bs}: v1 {t_v1:.1f} us, M2 {t_m2:.1f} us ({t_v1 / t_m2:.2f}x) err {worst[0]:.1e} row {worst[1]:.1e} "
            f"-> {'M2' if use else 'v1'} | s1 {_name(cfg1)} | s2 {_name(cfg2)}")
    state["m2_configs"] = tuple((E2_CLASS[bs], *(chosen.get(bs) or (None, None))) for bs in E2_BS)
    for cfgs in chosen.values():
        m2_warm(state, *cfgs)
    log(f"M2_CONFIGS = {state['m2_configs']!r}")


def prepare(weights):
    state = _submission_prepare(weights)
    lab = Lab(LAB_BUDGET_S)
    log(f"E2 start, budget {LAB_BUDGET_S:.0f}s, {state['sms']} SMs")
    ref = _Ref(weights)
    store = {}
    lab.section("build", sec_build, weights, state)
    lab.section("stage1", sec_stage, 1, state, ref, store, min_left=60)
    lab.section("stage2", sec_stage, 2, state, ref, store, min_left=50)
    lab.section("e2e", sec_correct_and_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state['m2_configs']!r}"[:400])
    return state
