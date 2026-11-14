"""RE29 lab (check runs only): the first-turn grouped path's glue at 16,384 tokens, kernel by kernel, on synthetic
routing (top 8 of 128 experts, this GPU's 32): torch.where/expert_map, torch.sort (stable), the histogram, inv, the
route-order gather (num_warps 2/4/8), act (4/8/16), combine (4/8/16), and the two dequants for scale; CUDA-graph
replays with a cold L2. kernel_fn = v7g's, unchanged. Prints one summary line per kernel and the best num_warps.
Appended after v7g_glue2.py by local/make_lab.py --base v7g_glue2.py."""

RE29_LIMIT_S = 250.0
RE29_TOKENS = 16384


def re29_sec(lab, weights, state):
    dev = "cuda"
    gen = torch.Generator(device=dev).manual_seed(29)
    tokens = RE29_TOKENS
    slots = tokens * TOP_K
    gmm, bank = state["gmm"], state["bank"]
    scores = torch.rand((tokens, N_EXPERTS), generator=gen, device=dev)
    scores[:, :LOCAL_EXPERTS] += 0.15  # this GPU's range gets more routes, like the judge's busiest GPU
    topk_ids = scores.topk(TOP_K, dim=-1).indices.to(torch.int32)
    topk_w = torch.softmax(scores.gather(1, topk_ids.long()), -1).float().contiguous()
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    local_ids = bank["expert_map"][topk_ids].contiguous()
    keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1)).contiguous()
    sorted_keys, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device=dev)
    _key_hist_kernel[(triton.cdiv(slots, HIST_BLOCK),)](keys, counts, slots, NE=LOCAL_EXPERTS, BLOCK=HIST_BLOCK,
                                                        BINS=HIST_BINS, num_warps=4)
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    _inv_perm_kernel[(triton.cdiv(slots, INV_BLOCK),)](perm, inv, slots, BLOCK=INV_BLOCK, num_warps=4)
    n_local = int(offs[-1])
    a = torch.empty((slots, D_EXPERT), device=dev, dtype=torch.bfloat16)
    h1 = (torch.randn((slots, 2 * D_EXPERT), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
    act = torch.empty((slots, D_EXPERT), device=dev, dtype=torch.bfloat16)
    y2 = (torch.randn((slots, D_EXPERT), generator=gen, device=dev) * 0.1).to(torch.bfloat16)
    out = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    note(f"routing: {n_local} local rows of {slots} routes")
    res = {}

    def t(name, fn):
        us = lab.time_graph(fn, reps=10)
        res[name] = us
        note(f"{name}: {us:.1f} us")

    t("where+map", lambda: torch.where(bank["expert_map"][topk_ids].view(-1) < 0, LOCAL_EXPERTS,
                                       bank["expert_map"][topk_ids].view(-1)))
    t("sort", lambda: torch.sort(keys, stable=True))
    t("hist", lambda: _key_hist_kernel[(triton.cdiv(slots, HIST_BLOCK),)](
        keys, counts.zero_(), slots, NE=LOCAL_EXPERTS, BLOCK=HIST_BLOCK, BINS=HIST_BINS, num_warps=4))
    t("cumsum", lambda: counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32))
    t("inv", lambda: _inv_perm_kernel[(triton.cdiv(slots, INV_BLOCK),)](perm, inv, slots, BLOCK=INV_BLOCK,
                                                                         num_warps=4))
    for nw in (2, 4, 8):
        t(f"gather nw{nw}", lambda nw=nw: _gather_routes_kernel[(slots,)](
            x, keys, inv, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS, num_warps=nw))
    for nw in (4, 8, 16):
        t(f"act nw{nw}", lambda nw=nw: _gmm_act_kernel[(slots,)](
            h1, sorted_keys, gmm["w13_gs"], offs, act, I=D_EXPERT, NE=LOCAL_EXPERTS, num_warps=nw))
    for nw in (4, 8, 16):
        t(f"combine nw{nw}", lambda nw=nw: _gmm_combine_kernel[(tokens,)](
            y2, local_ids, inv, gmm["w2_gs"], topk_w, state["post_expert_w"], out, POST_NORM_EPS, H=D_EXPERT,
            TOPK=TOP_K, num_warps=nw))
    t("dequant w13+w2", lambda: (_dequant(gmm["w13_q"], gmm["w13_sf"], counts, gmm["w13"]),
                                 _dequant(gmm["w2_q"], gmm["w2_sf"], counts, gmm["w2"])))
    best = {}
    for k in ("gather", "act", "combine"):
        cand = {n: v for n, v in res.items() if n.startswith(k + " ")}
        if cand:
            best[k] = min(cand, key=cand.get)
    note("RE29 best: " + ", ".join(f"{k} {v} ({res[v]:.1f} us)" for k, v in best.items()))


RE7_SUMMARY = []


def note(msg):
    log(msg)
    RE7_SUMMARY.append(msg)


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    note(f"submission prepare {time.monotonic() - t0:.1f}s, grouped ok={state['gmm']['ok']}")
    lab = Lab(RE29_LIMIT_S - (time.monotonic() - t0))
    if state["gmm"]["ok"]:
        lab.section("re29 glue", re29_sec, weights, state, min_left=30.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log("===== summary =====")
    for line in RE7_SUMMARY:
        log(line)
    return state
