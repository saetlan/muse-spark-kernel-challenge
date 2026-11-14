"""RE3 lab (check runs only): the fused NVFP4 grouped GEMMs (R3) on the judge's H200.
1. Stage sweeps at 3012 tokens (real-router routing): GEMM1 (gated, SwiGLU epilogue) and GEMM2 configs,
   each checked against an FP32 reference built from the exact BF16 decode, with registers/spills.
2. The fastest configs re-timed at 753 and 16384 tokens (no recompiles: max_tiles isn't specialized).
3. End to end per class (753 / 3012 / 16384, real-router and synthetic routing): Marlin, v2's grouped
   path and R3, errors vs Marlin; plus R3 correctness at the judge's test sizes in each class.
4. Hybrid dispatch: a class uses R3 only where it was correct everywhere and >= 3% faster than what v3
   runs there otherwise (Marlin below 8192 tokens, the grouped path from 8192). Prints R3_CONFIGS.
5. If time is left: dequant kernel block sizes (R10).
Appended after submission.py (v3) by local/make_lab.py; kernel_fn stays the submission's, and reads
state["r3"]["cfg"], which this lab fills."""

RE3_PREPARE_LIMIT_S = 255.0
RE3_SWEEP_TOKENS = 3012
RE3_CLASS_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
RE3_TEST_TOKENS = {"t1": (300, 600), "t4": (1506, 4096), "ft": ()}
# (swap, bt, bn, bk, warps, stages, scale2d), most promising first (the sweep stops when time runs low).
RE3_G1 = [
    (0, 64, 128, 256, 8, 3, 0), (0, 128, 64, 256, 8, 3, 0), (1, 64, 128, 256, 8, 3, 0), (0, 64, 64, 256, 4, 3, 0),
    (0, 128, 64, 128, 8, 3, 0), (1, 128, 64, 128, 8, 3, 0), (0, 64, 128, 128, 4, 4, 0), (0, 64, 128, 256, 8, 2, 1),
    (1, 64, 64, 256, 4, 3, 0), (0, 32, 128, 256, 4, 3, 0), (0, 128, 128, 128, 8, 2, 0),
]
RE3_G2 = [
    (0, 64, 128, 256, 8, 3, 0), (0, 128, 128, 128, 8, 3, 0), (1, 64, 128, 256, 8, 3, 0), (0, 64, 64, 256, 4, 3, 0),
    (0, 128, 64, 256, 8, 3, 0), (1, 128, 128, 128, 8, 3, 0), (0, 64, 256, 128, 8, 3, 0), (0, 64, 128, 256, 8, 2, 1),
    (0, 128, 128, 256, 8, 2, 0), (0, 32, 128, 256, 4, 3, 0),
]
RE3_TOP = 3
RE3_BKS = (128, 256)


def re3_cfg_name(c):
    return f"s{c[0]} bt{c[1]} bn{c[2]} bk{c[3]} w{c[4]} st{c[5]}{' 2d' if c[6] else ''}"


def re3_real_routing(weights, tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mean = torch.randn((1, HIDDEN), generator=gen, device="cuda")
    h = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") + mean
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    scores = torch.sigmoid(0.1767766952966369 * (h @ weights["router"].float().T))
    ids = (scores + weights["router_bias"].float()).topk(TOP_K, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True)
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), w.float()


def re3_inputs(weights, tokens, kind):
    gen = torch.Generator(device="cuda").manual_seed(13)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    ids, w = re3_real_routing(weights, tokens, tokens) if kind == "real" else synthetic_routing(tokens, seed=tokens)
    return x, ids, w


def re3_rel(out, ref):
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-30))


class Re3Stage:
    """Sorted rows, tables and FP32 stage references for one routing."""

    def __init__(self, state, x, ids):
        gmm = state["gmm"]
        self.state = state
        self.slots = x.shape[0] * TOP_K
        self.local_ids, self.counts, self.offs, self.inv, self.a = _r3_route(x, ids, state)
        self.n = int(self.offs[-1])
        bounds = [0] + self.offs.tolist()
        _dequant(gmm["w13_q"], gmm["w13_sf"], self.counts, gmm["w13"])
        _dequant(gmm["w2_q"], gmm["w2_sf"], self.counts, gmm["w2"])
        self.segs = [(e, bounds[e], bounds[e + 1]) for e in range(LOCAL_EXPERTS) if bounds[e + 1] > bounds[e]]
        act = torch.zeros((self.n, D_EXPERT), device="cuda")
        for e, s, t in self.segs:
            h = self.a[s:t].float() @ gmm["w13"][e].float().T
            g, u = h[:, :D_EXPERT] * gmm["w13_gs"][e], h[:, D_EXPERT:] * gmm["w13_gs"][e]
            act[s:t] = g * torch.sigmoid(g) * u
        self.act_ref = act
        self.act_in = torch.zeros((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        self.act_in[: self.n] = act.to(torch.bfloat16)
        y = torch.zeros((self.n, D_EXPERT), device="cuda")
        for e, s, t in self.segs:
            y[s:t] = self.act_in[s:t].float() @ gmm["w2"][e].float().T
        self.y_ref = y
        self.tables = {}
        self.out1 = torch.empty((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        self.out2 = torch.empty((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)

    def table(self, bt):
        if bt not in self.tables:
            mt = _r3_max_tiles(self.slots, bt)
            self.tables[bt] = (_r3_tiles(self.counts, self.offs, bt, mt), mt)
        return self.tables[bt]

    def launch(self, cfg, stage):
        r3, gmm = self.state["r3"], self.state["gmm"]
        table, mt = self.table(cfg[1])
        if stage == 1:
            return _r3_gemm(self.a, r3["w13"][cfg[3]], gmm["w13_sf"], r3["c1"], table, mt, self.out1, cfg, True)
        return _r3_gemm(self.act_in, r3["w2"][cfg[3]], gmm["w2_sf"], r3["c2"], table, mt, self.out2, cfg, False)

    def run(self, lab, cfg, stage, reps=5):
        k = self.launch(cfg, stage)
        torch.cuda.synchronize()
        err = re3_rel(self.out1[: self.n] if stage == 1 else self.out2[: self.n], self.act_ref if stage == 1 else self.y_ref)
        us = lab.time_graph(lambda: self.launch(cfg, stage), reps=reps)
        return us, err, getattr(k, "n_regs", -1), getattr(k, "n_spills", -1)


def sec_sweep(lab, weights, state, out):
    x, ids, w = re3_inputs(weights, RE3_SWEEP_TOKENS, "real")
    st = Re3Stage(state, x, ids)
    log(f"sweep T={RE3_SWEEP_TOKENS}: {st.n} local rows over {len(st.segs)} experts")
    good = {1: [], 2: []}
    order = [(i, stage) for i in range(max(len(RE3_G1), len(RE3_G2))) for stage in (1, 2)]
    for i, stage in order:
        cfgs = RE3_G1 if stage == 1 else RE3_G2
        if i >= len(cfgs):
            continue
        cfg = cfgs[i]
        if lab.left() < 110:
            log("sweep: budget low, stopped")
            break
        if True:
            t = time.monotonic()
            try:
                us, err, regs, spills = st.run(lab, cfg, stage)
                ok = err < R3_MAX_ERR
                log(f"g{stage} {re3_cfg_name(cfg)}: {us:.1f}us err {err:.1e} regs {regs} spill {spills} "
                    f"({time.monotonic() - t:.1f}s){'' if ok else ' WRONG'}")
                if ok:
                    good[stage].append((us, cfg))
            except Exception as exc:
                log(f"g{stage} {re3_cfg_name(cfg)}: FAILED {type(exc).__name__}: {_error_text(exc, 160)}")
    for stage in (1, 2):
        good[stage].sort()
        out[stage] = [c for _, c in good[stage][:RE3_TOP]]


def sec_retime(lab, weights, state, top, best):
    """Top configs of each stage at every class size; best[(tokens, stage)] = fastest correct config."""
    for tokens in (RE3_CLASS_TOKENS["t1"], RE3_CLASS_TOKENS["t4"], RE3_CLASS_TOKENS["ft"]):
        if lab.left() < 70:
            log("retime: budget low, stopping")
            return
        x, ids, w = re3_inputs(weights, tokens, "real")
        st = Re3Stage(state, x, ids)
        for stage in (1, 2):
            row = []
            for cfg in top.get(stage, []):
                try:
                    us, err, _, _ = st.run(lab, cfg, stage, reps=5)
                    row.append(f"{re3_cfg_name(cfg)} {us:.1f}us/{err:.0e}")
                    if err < R3_MAX_ERR and (us < best.get((tokens, stage), (1e30, None))[0]):
                        best[(tokens, stage)] = (us, cfg)
                except Exception as exc:
                    row.append(f"{re3_cfg_name(cfg)} FAILED {type(exc).__name__}")
            log(f"retime T={tokens} ({st.n} rows) g{stage}: " + "; ".join(row))
        del st


def sec_e2e(lab, weights, state, best):
    """Per class: Marlin vs grouped vs R3 end to end; R3 correctness at the test sizes; dispatch."""
    lines = {}
    for cls, tokens in RE3_CLASS_TOKENS.items():
        if (tokens, 1) not in best or (tokens, 2) not in best:
            log(f"e2e {cls}: no correct config timed, stays off")
            continue
        if lab.left() < 25:
            log(f"e2e {cls}: budget low, stays off")
            continue
        cfg1, cfg2 = best[(tokens, 1)][1], best[(tokens, 2)][1]
        cur_name = "grouped" if tokens >= GMM_FROM_TOKENS else "marlin"
        wins, ok_all = True, True
        for kind in ("real", "synth"):
            x, ids, w = re3_inputs(weights, tokens, kind)
            ref = _experts(x, w, ids, state).float()
            out = _experts_r3(x, w, ids, state, cfg1, cfg2).float()
            err = re3_rel(out, ref)
            row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            ok = bool(torch.isfinite(out).all()) and err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
            t_m = lab.time_graph(lambda: _experts(x, w, ids, state), reps=5)
            t_g = lab.time_graph(lambda: _experts_gmm(x, w, ids, state), reps=5) if state["gmm"]["ok"] else float("inf")
            t_r = lab.time_graph(lambda: _experts_r3(x, w, ids, state, cfg1, cfg2), reps=5)
            cur = t_g if cur_name == "grouped" else t_m
            log(f"e2e {cls} T={tokens} {kind}: marlin {t_m:.1f} grouped {t_g:.1f} r3 {t_r:.1f}us "
                f"({cur / t_r:.2f}x vs {cur_name}) err {err:.1e} row {row:.1e}")
            wins &= t_r <= 0.97 * cur
            ok_all &= ok
        for t in RE3_TEST_TOKENS[cls]:
            err, row = r3_check(state, cfg1, cfg2, t)
            log(f"e2e {cls} test size {t}: err {err:.1e} row {row:.1e}")
            ok_all &= err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
        if wins and ok_all:
            state["r3"]["cfg"][cls] = (cfg1, cfg2)
            lines[cls] = (cfg1, cfg2)
        log(f"e2e {cls}: {'R3 ON' if wins and ok_all else 'stays ' + cur_name} (wins {wins}, correct {ok_all})")
    log(f"R3_CONFIGS = {lines}")


def sec_dequant_blocks(lab, state):
    gmm = state["gmm"]
    on = torch.ones(LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda")
    row = []
    for bn, bkb, warps in ((32, 128, 4), (16, 256, 4), (64, 128, 8), (8, 512, 4), (32, 256, 8), (64, 64, 4)):
        if lab.left() < 8:
            break

        def run():
            for name in ("w13", "w2"):
                q = gmm[name + "_q"]
                _nvfp4_dequant_kernel[(q.shape[0], q.shape[1] // bn, q.shape[2] // bkb)](
                    q, gmm[name + "_sf"], on, gmm[name], N=q.shape[1], KB=q.shape[2], BN=bn, BKB=bkb, num_warps=warps)
        try:
            row.append(f"{bn}x{bkb}w{warps} {lab.time_graph(run, reps=5):.0f}us")
        except Exception as exc:
            row.append(f"{bn}x{bkb}w{warps} FAILED {type(exc).__name__}")
    log("dequant all 32 (4.13 GB): " + ", ".join(row))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"submission prepare {time.monotonic() - t0:.1f}s, grouped ok={state['gmm']['ok']}, r3 cfg={state['r3']['cfg']}")
    lab = Lab(RE3_PREPARE_LIMIT_S - (time.monotonic() - t0))
    if not state["gmm"]["ok"]:
        log("grouped path off: the lab's references need its decode, skipping")
        return state
    t = time.monotonic()
    for bk in RE3_BKS:
        r3_add_bk(state, bk)
    torch.cuda.synchronize()
    log(f"repacks for BK {RE3_BKS}: {time.monotonic() - t:.1f}s")
    top, best = {}, {}
    lab.section("sweep", sec_sweep, weights, state, top)
    lab.section("retime", sec_retime, weights, state, top, best, min_left=60.0)
    lab.section("e2e", sec_e2e, weights, state, best, min_left=20.0)
    lab.section("dequant_blocks", sec_dequant_blocks, state, min_left=8.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"lab total {time.monotonic() - t0:.1f}s, kernel_fn R3 classes: {sorted(state['r3']['cfg'])}")
    return state
