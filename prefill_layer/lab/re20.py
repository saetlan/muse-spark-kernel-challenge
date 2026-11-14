"""RE20 lab (check runs only): R11 = the first-turn grouped path's NVFP4 -> BF16 expert decode started on a side
stream at the top of kernel_fn (all 32 local experts), overlapping attention and the dense projections; joined before
the grouped GEMMs. Whole-layer A/B in prepare: kernel_fn on lab-built first-turn inputs (batch 1, 16,384 new tokens,
ctx 32,768 sliding = the weight-138 case, and ctx 0 full), CUDA-graph replays with a cold L2, R11 off/on interleaved,
plus a bit-equality check of both outputs. R14 t1/t4 dispatch as RE19 (R14 / R47 only). kernel_fn keeps R11 only if
it is equal and >= 3% faster on the sliding case and not slower on the full one.
Appended after submission.py (v6) by local/make_lab.py; kernel_fn reads state["r14"]["cfg"] and R11_ON."""

RE4_PREPARE_LIMIT_S = 262.0
RE4_CLASS_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
RE4_TEST_TOKENS = {"t1": (300, 600), "t4": (1506, 4096), "ft": ()}
RE4_UNIT_TOKENS = (64, 753)
RE4_SMALL = 64
RE7_SUMMARY = []
R14_VARIANTS = ((2, 8, 1, 1), (2, 8, 1, 1, 14))


def note(msg):
    log(msg)
    RE7_SUMMARY.append(msg)


def vname(v):
    return v if isinstance(v, str) else "-".join(map(str, v))


def re4_routing(weights, tokens, kind, seed=None):
    gen = torch.Generator(device="cuda").manual_seed(seed if seed is not None else tokens)
    if kind == "synth":
        return synthetic_routing(tokens, seed=tokens)
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


def re4_x(tokens):
    gen = torch.Generator(device="cuda").manual_seed(17)
    return (torch.randn((tokens, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)


def re4_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-30))


def re4_unswizzle(sw, rows):
    """[KC, rows, 64] swizzled -> [rows, KC*64]: element col of row p at unit (col%64)//8 ^ (p & 7)."""
    kc_count = sw.shape[0]
    p = torch.arange(rows, device=sw.device).view(-1, 1)
    col = torch.arange(kc_count * 64, device=sw.device).view(1, -1)
    unit = ((col % 64) // 8) ^ (p & 7)
    return sw[col // 64, p, unit * 8 + col % 8]


class Re4Case:
    """Route + gather through the real kernels, and FP32 references for each GEMM stage."""

    def __init__(self, state, x, ids, mode=0):
        r14, gmm, bank = state["r14"], state["gmm"], state["bank"]
        self.state, self.x = state, x
        tokens = x.shape[0]
        self.slots, self.max_pieces, self.rows = _r14_sizes(tokens, mode)
        self.local = bank["expert_map"][ids].contiguous()
        self.pos = torch.empty(self.slots, device="cuda", dtype=torch.int32)
        self.pieces = torch.empty((self.max_pieces, 4), device="cuda", dtype=torch.int32)
        self.n_pieces = torch.empty(1, device="cuda", dtype=torch.int32)
        self.mode = mode
        r14["ext"].r14_route(self.local.view(-1), self.pos, self.pieces, self.n_pieces, self.max_pieces, mode)
        self.a_sw = torch.zeros((D_EXPERT // 64, self.rows, 64), device="cuda", dtype=torch.bfloat16)
        _r14_gather_kernel[(self.slots,)](x.contiguous(), self.pos, self.a_sw, self.rows, H=D_EXPERT, TOPK=TOP_K,
                                          num_warps=4)
        torch.cuda.synchronize()
        flat = self.local.view(-1)
        counts = torch.bincount(flat[flat >= 0].long(), minlength=LOCAL_EXPERTS)
        _dequant(gmm["w13_q"], gmm["w13_sf"], torch.cat((counts, counts[:1])).to(torch.int32), gmm["w13"])
        _dequant(gmm["w2_q"], gmm["w2_sf"], torch.cat((counts, counts[:1])).to(torch.int32), gmm["w2"])
        self.live = (flat >= 0).nonzero().view(-1)
        self.lpos = self.pos[self.live].long()
        self.lexp = flat[self.live].long()
        self.ltok = self.live // TOP_K

    def check_route(self):
        np_ = int(self.n_pieces.item())
        pcs = self.pieces[:np_].cpu().tolist()
        lpos, lexp = self.lpos.cpu(), self.lexp.cpu()
        ok = bool((self.pos.cpu()[(self.local.view(-1).cpu() < 0)] == -1).all()) and len(set(lpos.tolist())) == len(lpos)
        covered = 0
        for e, r, v, n in pcs:
            ok &= r % 16 == 0 and n % 16 == 0 and 16 <= n <= 128 and 0 < v <= n
            mine = lpos[(lexp == e) & (lpos >= r) & (lpos < r + v)]
            covered += len(mine)
        ok &= covered == len(lpos)
        a = re4_unswizzle(self.a_sw, self.rows)[self.lpos]
        gath = re4_rel(a, self.x[self.ltok])
        return ok and gath == 0.0, f"pieces {np_}, rows {len(lpos)}, gather err {gath:.1e}"

    def act_ref(self):
        gmm = self.state["gmm"]
        ref = torch.zeros((len(self.lpos), D_EXPERT), device="cuda")
        for e in self.lexp.unique().tolist():
            m = self.lexp == e
            h = self.x[self.ltok[m]].float() @ gmm["w13"][e].float().T
            g, u = h[:, :D_EXPERT] * gmm["w13_gs"][e], h[:, D_EXPERT:] * gmm["w13_gs"][e]
            ref[m] = g * torch.sigmoid(g) * u
        return ref

    def gemm1(self, cfg, diag=0, err=None):
        r14 = self.state["r14"]
        act_sw = torch.zeros((D_EXPERT // 64, self.rows, 64), device="cuda", dtype=torch.bfloat16)
        blob = r14["w13b"] if (diag in R14_LAYOUT2_DIAGS or (len(cfg) > 4 and cfg[4] in R14_LAYOUT2_DIAGS)) else r14["w13"]
        diag = diag or (cfg[4] if len(cfg) > 4 else 0)
        r14["ext"].r14_gemm(blob, self.a_sw, self.pieces, self.n_pieces, r14["c1"], act_sw,
                            r14["err"] if err is None else err,
                            self.rows, D_EXPERT // 64, self.max_pieces, cfg[0], cfg[1], True, cfg[3], diag)
        return act_sw

    def gemm2(self, act_sw, cfg, diag=0, err=None):
        r14 = self.state["r14"]
        y2 = torch.zeros((self.rows, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        blob = r14["w2b"] if (diag in R14_LAYOUT2_DIAGS or (len(cfg) > 4 and cfg[4] in R14_LAYOUT2_DIAGS)) else r14["w2"]
        diag = diag or (cfg[4] if len(cfg) > 4 else 0)
        r14["ext"].r14_gemm(blob, act_sw, self.pieces, self.n_pieces, r14["c2"], y2,
                            r14["err"] if err is None else err, self.rows,
                            D_EXPERT // 128, self.max_pieces, cfg[0], cfg[1], False, cfg[3], diag)
        return y2


def re4_locate(got, ref, lpos):
    """Error split by output column group and token row mod 8 (localizes layout bugs)."""
    d = (got - ref).float()
    cols = [float(d[:, g::8].norm() / ref[:, g::8].norm().clamp(min=1e-30)) for g in range(8)]
    rows = [float(d[lpos % 8 == r].norm() / ref[lpos % 8 == r].norm().clamp(min=1e-30)) for r in range(8)]
    grp = [float(d[:, 8 * j:8 * j + 8].norm() / ref[:, 8 * j:8 * j + 8].norm().clamp(min=1e-30)) for j in range(8)]
    return f"col%8 {['%.0e' % v for v in cols]} row%8 {['%.0e' % v for v in rows]} colgrp(first 64) {['%.0e' % v for v in grp]}"


def sec_build(lab, state):
    r14 = state["r14"]
    if not r14["ok"]:
        log("R14 not built (see the [v4] line above); nothing to test")
        return
    a = r14["ext"].r14_attrs().tolist()
    note(f"regs/spill: base g {a[0]}/{a[1]} p {a[2]}/{a[3]}, r47 g {a[4]}/{a[5]} p {a[6]}/{a[7]}, r55 g {a[8]}/{a[9]} p {a[10]}/{a[11]}, r55 timed g {a[12]}/{a[13]} p {a[14]}/{a[15]}")


def re7_unit(cs, ref1, stages, tokens, state):
    """GEMM1 and GEMM2 of one variant against the FP32 stage references; True if both pass."""
    r14 = state["r14"]
    r14["err"].zero_()
    act_sw = cs.gemm1(stages)
    torch.cuda.synchronize()
    flags = int(r14["err"].item())
    act = re4_unswizzle(act_sw, cs.rows)[cs.lpos]
    e1 = re4_rel(act, ref1)
    log(f"gemm1 T={tokens} {vname(stages)}: err {e1:.2e} flags {flags} finite {bool(torch.isfinite(act.float()).all())}")
    if not (e1 < R14_MAX_ERR and flags == 0):
        log("gemm1 locate: " + re4_locate(act, ref1, cs.lpos))
        return False
    # GEMM2 on the reference activations (bf16), written swizzled.
    act_in = torch.zeros_like(act_sw)
    pr = cs.lpos.view(-1, 1)
    col = torch.arange(D_EXPERT, device="cuda").view(1, -1)
    act_in[col // 64, pr, (((col % 64) // 8) ^ (pr & 7)) * 8 + col % 8] = ref1.to(torch.bfloat16)
    r14["err"].zero_()
    y2 = cs.gemm2(act_in, stages)
    torch.cuda.synchronize()
    flags = int(r14["err"].item())
    ref2 = torch.zeros_like(ref1)
    gmm = state["gmm"]
    for e in cs.lexp.unique().tolist():
        m = cs.lexp == e
        ref2[m] = ref1[m].to(torch.bfloat16).float() @ gmm["w2"][e].float().T
    e2 = re4_rel(y2[cs.lpos], ref2)
    log(f"gemm2 T={tokens} {vname(stages)}: err {e2:.2e} flags {flags}")
    if not (e2 < R14_MAX_ERR and flags == 0):
        log("gemm2 locate: " + re4_locate(y2[cs.lpos], ref2, cs.lpos))
        return False
    return True


def sec_units(lab, weights, state, out):
    """Per variant: a failure (wrong result, flags, launch error) drops that variant only."""
    out["bad"] = set()
    for tokens in RE4_UNIT_TOKENS:
        x = re4_x(tokens)
        ids, w = re4_routing(weights, tokens, "real")
        for mode in sorted(set(v[2] for v in R14_VARIANTS)):
            cs = Re4Case(state, x, ids, mode)
            ok, msg = cs.check_route()
            log(f"route/gather T={tokens} mode {mode}: {'OK' if ok else 'WRONG'} {msg}")
            if not ok:
                out["units"] = False
                return
            ref1 = cs.act_ref()
            for stages in [v for v in R14_VARIANTS if v[2] == mode and v not in out["bad"]]:
                try:
                    good = re7_unit(cs, ref1, stages, tokens, state)
                except Exception as exc:  # e.g. a rejected cluster launch (TORCH_CHECK in r14_gemm)
                    log(f"unit T={tokens} {vname(stages)}: FAILED {_error_text(exc, 300)}")
                    good = False
                if not good:
                    out["bad"].add(stages)
            del cs
    passed = [v for v in R14_VARIANTS if v not in out["bad"]]
    note(f"units: passed {[vname(v) for v in passed]}, dropped {[vname(v) for v in sorted(out['bad'])]}")
    out["units"] = bool(passed)


def sec_e2e(lab, weights, state, out):
    """Correctness end to end and timings; fills out["t"][(tokens, kind, path)] and out["ok"][(tokens, stages)]."""
    out["t"], out["okc"] = {}, {}
    out_bad = out.get("bad", set())
    for tokens in (RE4_SMALL, RE4_CLASS_TOKENS["t1"], RE4_CLASS_TOKENS["t4"]):  # 16K stays grouped (RF26)
        for kind in ("real", "synth"):
            if lab.left() < 30:
                log(f"e2e T={tokens} {kind}: budget low, stopping")
                return
            x = re4_x(tokens)
            ids, w = re4_routing(weights, tokens, kind)
            ref = _experts(x, w, ids, state).float()
            row_txt = []
            for stages in [v for v in R14_VARIANTS if v not in out_bad]:
                state["r14"]["err"].zero_()
                o = _experts_r14(x, w, ids, state, stages).float()
                torch.cuda.synchronize()
                flags = int(state["r14"]["err"].item())
                err = re4_rel(o, ref)
                row = float(((o - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
                ok = flags == 0 and bool(torch.isfinite(o).all()) and err < R14_MAX_ERR and row < 5 * R14_MAX_ERR
                out["okc"][(tokens, stages)] = out["okc"].get((tokens, stages), True) and ok
                row_txt.append(f"{vname(stages)}: {err:.0e}/{flags}")
                if ok and tokens != RE4_SMALL:
                    out["t"][(tokens, kind, stages)] = lab.time_graph(
                        lambda: _experts_r14(x, w, ids, state, stages), reps=6)
            if tokens != RE4_SMALL:
                out["t"][(tokens, kind, "marlin")] = lab.time_graph(lambda: _experts(x, w, ids, state), reps=6)
            times = " ".join(f"{vname(k[2])} {v:.0f}" for k, v in out["t"].items() if k[0] == tokens and k[1] == kind)
            note(f"e2e T={tokens} {kind}: {'; '.join(row_txt)} | us: {times}")


def re8_prof_line(v, label, t_us, t_prof_us):
    ch, wg, pr = max(v[8], 1), max(v[7], 1), max(v[11], 1)
    return (f"{label}: {t_us:.0f} us ({t_prof_us:.0f} timed); per chunk: issue {v[0] / ch:.0f} wgmma-wait {v[1] / ch:.0f} "
            f"full-wait {v[2] / ch:.0f} decode {v[3] / ch:.0f} loop {v[5] / ch:.0f} cyc; per WG: first {v[4] / wg:.0f} "
            f"epi {v[6] / wg:.0f} N {v[12] / wg:.0f}; producer empty-wait {v[9] / pr:.0f} loop {v[10] / pr:.0f}; "
            f"WGs {v[7]} chunks/WG {v[8] / wg:.0f}")


R11_LAB_CASES = ((32768, "sliding"), (0, "full"))  # first turn: weight 138 and 16


def r11_input(weights, ctx, layer, q_len=16384, seed=11):
    """A first-turn kernel_fn input shaped like the judge's (reference.generate_input, batch 1, page 1, fixed length):
    scattered one-token pages, FP8 cache, residual stream with the reference's mean / low-rank structure (seed 5354),
    local_start = the 32-expert range the routing of this stream favours."""
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(seed)
    total = ctx + q_len
    page_table = torch.randperm(total, generator=gen, device=dev).to(torch.int32).view(1, total)
    positions = ctx + torch.arange(q_len, device=dev)
    cache_loc = page_table[0].long()[positions]
    k_cache = torch.randn((total, 1, 2, 64), generator=gen, device=dev).to(FP8)
    v_cache = torch.randn((total, 1, 2, 64), generator=gen, device=dev).to(FP8)
    basis = torch.Generator().manual_seed(5354)
    mean = torch.randn((HIDDEN,), generator=basis).to(dev)
    shared = torch.randn((4, HIDDEN), generator=basis).to(dev)
    coef = torch.randn((q_len, 4), generator=gen, device=dev)
    stream = 0.7 * mean + (0.9 / 4) ** 0.5 * (coef @ shared) + 0.1 ** 0.5 * torch.randn((q_len, HIDDEN), generator=gen, device=dev)
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(torch.bfloat16)
    scores = torch.sigmoid(0.1767766952966369 * (attn_input.float() @ weights["router"].float().T))
    ids = (scores + weights["router_bias"].float()).topk(TOP_K, dim=-1).indices
    local_start = int(torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS).argmax()) * LOCAL_EXPERTS
    return {
        "stream": stream, "attn_input": attn_input, "positions": positions, "cache_loc": cache_loc,
        "cache_seqlens": torch.tensor([total], dtype=torch.int32, device=dev),
        "cu_seqlens_q": torch.tensor([0, q_len], dtype=torch.int32, device=dev),
        "scheduler_metadata": None, "k_cache": k_cache, "v_cache": v_cache,
        "k_scale": torch.tensor(1.0, device=dev), "v_scale": torch.tensor(1.0, device=dev),
        "page_table": page_table, "seq_lens": torch.tensor([ctx], dtype=torch.int32, device=dev),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=dev),
        "config": {"batch": 1, "q_len": q_len, "tokens": q_len, "hidden": HIDDEN, "num_q_heads": 16,
                   "num_kv_heads": 2, "head_dim": 64, "page_size": 1, "window": 2048 if layer == "sliding" else 0,
                   "rope_theta": ROPE_THETA if layer == "sliding" else 0.0, "qk_scale_factor": 5.4730064863875,
                   "sm_scale": 64 ** -0.5, "kv_scale": 1.0, "n_experts": N_EXPERTS, "top_k": TOP_K,
                   "local_experts": LOCAL_EXPERTS},
    }


def r11_run(data, state, on):
    """kernel_fn with R11 on/off on a fresh copy of the inputs (kernel_fn writes the cache)."""
    global R11_ON
    R11_ON = on
    d = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in data.items()}
    return kernel_fn(d, state)


def r11_timed(data, state, on):
    """kernel_fn as timed: no input copies inside the graph (the cache write is idempotent)."""
    global R11_ON
    R11_ON = on
    return kernel_fn(data, state)


def sec_r11(lab, weights, state, out):
    """Whole-layer A/B of R11 at the first-turn shapes; sets R11_ON for kernel_fn (off on any failure)."""
    global R11_ON
    keep = False
    try:
        keep = _sec_r11(lab, weights, state, out)
    finally:
        R11_ON = keep
        note(f"R11_ON = {keep}")


def _sec_r11(lab, weights, state, out):
    out["r11"] = {}
    if not state["gmm"]["ok"]:
        note("r11: grouped path off, R11 stays off")
        return False
    for ctx, layer in R11_LAB_CASES:
        if lab.left() < 25:
            break
        data = r11_input(weights, ctx, layer)
        a = r11_run(data, state, False)
        b = r11_run(data, state, True)
        torch.cuda.synchronize()
        same = all(torch.equal(x, y) for x, y in zip(a, b))
        diff = max(float((x.float() - y.float()).abs().max()) for x, y in zip(a, b))
        times = {False: [], True: []}
        for on in (False, True, False, True):
            times[on].append(lab.time_graph(lambda: r11_timed(data, state, on), reps=5))
        off_us, on_us = min(times[False]), min(times[True])
        out["r11"][(ctx, layer)] = (same, off_us, on_us)
        note(f"r11 ctx {ctx} {layer}: off {['%.0f' % t for t in times[False]]} on {['%.0f' % t for t in times[True]]} us, "
             f"gain {100 * (1 - on_us / off_us):.1f}%, outputs equal {same} (max diff {diff:.1e})")
        del data, a, b
        torch.cuda.empty_cache()
    res = out["r11"]
    sl, fu = res.get((32768, "sliding")), res.get((0, "full"))
    return bool(sl and fu and sl[0] and fu[0] and sl[2] <= 0.97 * sl[1] and fu[2] <= 1.01 * fu[1])


def sec_dispatch(lab, state, res):
    if not res.get("units"):
        log("dispatch: unit tests failed, R14 stays off")
        return
    cfg = {}
    for cls, tokens in RE4_CLASS_TOKENS.items():
        if cls == "ft":
            note("dispatch ft: stays grouped (RF26: the lab's 16K routing doesn't predict the judge's)")
            continue
        cur = "marlin"
        best = None
        for stages in R14_VARIANTS:
            if not (res["okc"].get((tokens, stages)) and res["okc"].get((RE4_SMALL, stages))):
                continue
            tr = [res["t"].get((tokens, k, stages)) for k in ("real", "synth")]
            tc = [res["t"].get((tokens, k, cur)) for k in ("real", "synth")]
            if None in tr or None in tc:
                continue
            if all(a <= 0.97 * b for a, b in zip(tr, tc)) and (best is None or sum(tr) < best[0]):
                best = (sum(tr), stages)
        if best is not None:
            okt = True
            for t in RE4_TEST_TOKENS[cls]:
                err, row, flags = r14_check(state, best[1], t)
                log(f"dispatch {cls} test size {t}: err {err:.1e} row {row:.1e} flags {flags}")
                okt &= flags == 0 and err < R14_MAX_ERR and row < 5 * R14_MAX_ERR
            if okt:
                cfg[cls] = best[1]
        note(f"dispatch {cls}: {('R14 ' + str(best[1])) if cls in cfg else 'stays ' + cur}")
    state["r14"]["cfg"].update(cfg)
    note(f"R14_CONFIGS = {cfg}")


def sec_probe(lab):
    t = time.monotonic()
    log("probe build (ptxas -v; N = 128, cl 1 and 4), verbose:")
    r14_build(probe=True)
    note(f"probe build {time.monotonic() - t:.0f}s")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    note(f"submission prepare {time.monotonic() - t0:.1f}s, grouped ok={state['gmm']['ok']}, r14 ok={state['r14']['ok']}")
    lab = Lab(RE4_PREPARE_LIMIT_S - (time.monotonic() - t0))
    res = {}
    lab.section("build", sec_build, state)
    if state["r14"]["ok"]:
        lab.section("units", sec_units, weights, state, res, min_left=40.0)
        if res.get("units"):
            lab.section("e2e", sec_e2e, weights, state, res, min_left=30.0)
            lab.section("dispatch", sec_dispatch, state, res, min_left=10.0)
    global R11_ON
    R11_ON = False  # stays off unless sec_r11 measures it equal and faster
    lab.section("r11", sec_r11, weights, state, res, min_left=30.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log("===== summary =====")
    for line in RE7_SUMMARY:
        log(line)
    log(f"lab total {time.monotonic() - t0:.1f}s, kernel_fn R14 classes: {state['r14']['cfg']}")
    return state
