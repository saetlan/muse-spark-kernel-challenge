"""RE2 lab (check runs only): v2's grouped expert path against SGLang's Marlin path at prefill sizes
(cold-L2 graph time and error), and a per-component breakdown of the grouped path.
Appended after submission.py (v2) by local/make_lab.py; kernel_fn stays the submission's."""

RE2_BUDGET_S = 150.0
RE2_TOKENS = (753, 3012, 4096, 16384)
RE2_BREAKDOWN_TOKENS = (3012, 16384)


def re2_real_routing(weights, tokens, seed):
    """As RE1: random router inputs through the real router + bias, busiest GPU's range first."""
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


def re2_inputs(weights, tokens, kind):
    gen = torch.Generator(device="cuda").manual_seed(11)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    if kind == "real":
        ids, w = re2_real_routing(weights, tokens, seed=tokens)
    else:
        ids, w = synthetic_routing(tokens, seed=tokens)
    return x, ids, w


def sec_ab(lab, weights, state):
    for tokens in RE2_TOKENS:
        for kind in ("real", "synth"):
            if lab.left() < 25:
                log(f"ab T={tokens} {kind}: budget low, stopping")
                return
            x, ids, w = re2_inputs(weights, tokens, kind)
            local = int((ids < LOCAL_EXPERTS).sum())
            ref = _experts(x, w, ids, state).float()
            out = _experts_gmm(x, w, ids, state).float()
            err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
            row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            t_marlin = lab.time_graph(lambda: _experts(x, w, ids, state), reps=10)
            t_gmm = lab.time_graph(lambda: _experts_gmm(x, w, ids, state), reps=10)
            log(f"ab T={tokens} {kind} local {local}: marlin {t_marlin:.1f}us grouped {t_gmm:.1f}us "
                f"({t_marlin / t_gmm:.2f}x) err {err:.2e} worst row {row:.2e} finite {bool(torch.isfinite(out).all())}")


def sec_breakdown(lab, weights, state):
    gmm, bank = state["gmm"], state["bank"]
    for tokens in RE2_BREAKDOWN_TOKENS:
        x, ids, w = re2_inputs(weights, tokens, "real")
        slots = tokens * TOP_K
        local_ids = bank["expert_map"][ids].contiguous()
        keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
        sorted_keys, perm = torch.sort(keys, stable=True)
        counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda")
        counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
        offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
        inv = torch.empty_like(perm)
        inv.scatter_(0, perm, torch.arange(slots, device="cuda", dtype=perm.dtype))
        a = torch.empty((slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        act = torch.empty((slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((tokens, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        all_on = torch.ones(LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda")
        _gather_rows_kernel[(slots,)](x, perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS, num_warps=4)
        h1 = torch._grouped_mm(a, gmm["w13"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
        y2 = torch._grouped_mm(act, gmm["w2"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
        n_local, touched = int(offs[-1]), int((counts[:LOCAL_EXPERTS] > 0).sum())

        def route():
            k = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
            _, p = torch.sort(k, stable=True)
            c = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda")
            c.scatter_add_(0, k.long(), torch.ones_like(k))
            o = c[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
            iv = torch.empty_like(p)
            iv.scatter_(0, p, torch.arange(slots, device="cuda", dtype=p.dtype))

        parts = {
            "route(sort,counts,inv)": route,
            "gather": lambda: _gather_rows_kernel[(slots,)](x, perm, offs, a, H=D_EXPERT, TOPK=TOP_K,
                                                            NE=LOCAL_EXPERTS, num_warps=4),
            "dequant(touched)": lambda: (_dequant(gmm["w13_q"], gmm["w13_sf"], counts, gmm["w13"]),
                                         _dequant(gmm["w2_q"], gmm["w2_sf"], counts, gmm["w2"])),
            "dequant(all 32)": lambda: (_dequant(gmm["w13_q"], gmm["w13_sf"], all_on, gmm["w13"]),
                                        _dequant(gmm["w2_q"], gmm["w2_sf"], all_on, gmm["w2"])),
            "gemm1": lambda: torch._grouped_mm(a, gmm["w13"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16),
            "act": lambda: _gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w13_gs"], offs, act, I=D_EXPERT,
                                                     NE=LOCAL_EXPERTS, num_warps=4),
            "gemm2": lambda: torch._grouped_mm(act, gmm["w2"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16),
            "combine": lambda: _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], w, state["post_expert_w"],
                                                              out, POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8),
        }
        row = []
        for name, fn in parts.items():
            if lab.left() < 10:
                break
            row.append(f"{name} {lab.time_graph(fn, reps=10):.1f}us")
        all_bytes = sum(t.numel() * t.element_size() for t in (gmm["w13_q"], gmm["w13_sf"], gmm["w2_q"], gmm["w2_sf"],
                                                              gmm["w13"], gmm["w2"]))
        log(f"breakdown T={tokens} local {n_local} touched {touched}: " + ", ".join(row)
            + f" | dequant(all 32) moves {all_bytes / 1e9:.2f} GB")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"submission prepare: {time.monotonic() - t0:.1f}s, grouped path ok={state['gmm']['ok']}")
    lab = Lab(RE2_BUDGET_S)
    if state["gmm"]["ok"]:
        lab.section("ab", sec_ab, weights, state)
        lab.section("breakdown", sec_breakdown, weights, state, min_left=20.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"lab total {time.monotonic() - t0:.1f}s")
    return state
