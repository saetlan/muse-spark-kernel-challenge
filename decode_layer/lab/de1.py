"""DE1 lab (check runs only), on top of submission_v2. Output kept far below the judge's 16,000-character
log cap: one short line per result, failures counted, a summary printed last.

ops    each scored shape on judge-like inputs: local routes/experts, v2 and D1 whole layer as cold-L2
       graphs, and v2's ops one by one (short labels, us after subtracting the empty-graph cost).
gemm   qkv, o, pooler, unpooler slice, router hi at 1/8/16/32 tokens: cuBLAS vs tiny_gemm (<= 16) and
       SGLang's Hopper GEMV (1 token); us and TB/s.
router v2's FP32 router kernel vs F.linear at 1-8 tokens.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 160)}")


def _v2_op_steps(data, state):
    """v2's kernel_fn as named steps (short labels). Each reads env and returns its outputs."""
    cfg = data["config"]
    tokens, length = cfg["tokens"], cfg["q_len"]
    hq, hkv, d = 16, 2, 64
    dense, bank = state["dense"], state["bank"]
    window = cfg["window"]
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    m11cfg = m11_config_for(state, tokens) if "m11" in state else None
    moe_block = block_size_m(tokens)
    size_n, size_k = bank["size_n"], D_EXPERT
    gemm1_n = 2 * size_n
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)

    def qkn(e):
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](e["qkv"], data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](e["qkv"])
        return {"q": q, "k": k}

    def kvst(e):
        v = e["qkv"][:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        store_cache(e["k"].reshape(tokens, hkv * d).to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["v_cache"].view(-1, hkv * d).view(torch.uint8), data["cache_loc"], reserved_skip_index=-1)
        return {}

    def fa3(e):
        out = flash_attn_with_kvcache(
            q=e["q8"], k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"])
        return {"attn": out[0] if isinstance(out, tuple) else out}

    def bnd1(e):
        f, s, r = state["attn_boundary"](data["stream"], e["a"], branch_dtype=torch.bfloat16,
                                         router_dtype=None if split_router else torch.float32)
        return {"h": f, "s1": s, "rin": r}

    def rtr(e):
        if split_router:
            lg = torch.mm(e["h"], state["router_hi_t"], out_dtype=torch.float32)
            lg += torch.mm(e["h"], state["router_lo_t"], out_dtype=torch.float32)
            return {"logits": lg}
        return {"logits": v2_router_logits(e["rin"], state)}

    def route(e):
        w, ids = state["routing"](None, e["logits"], TOP_K, True)
        return {"tw": w.to(torch.float32), "ids": ids}

    def lid(e):
        loc = ((e["ids"] - data["local_start"]) % N_EXPERTS).to(torch.int32)
        return {"loc": loc, "emap": bank["expert_map"][loc].contiguous()}

    steps = [
        ("qkv", lambda e: {"qkv": _gemm(data["attn_input"], dense["qkv_gate"])}),
        ("qkn", qkn), ("kvst", kvst), ("q8", lambda e: {"q8": e["q"].view(tokens, hq, d).to(FP8)}), ("fa3", fa3),
        ("ngate", lambda e: {"og": state["o_norm_gate"](e["attn"].reshape(tokens, hq * d).to(torch.bfloat16),
                                                         e["qkv"][:, hq * d + 2 * hkv * d:])}),
        ("o", lambda e: {"a": _gemm(e["og"], dense["o"])}), ("bnd1", bnd1),
        ("pool", lambda e: {"pool": _gemm(e["h"], dense["pooler"])}), ("rtr", rtr), ("route", route), ("lid", lid),
        ("pxn", lambda e: {"px": state["pre_expert_norm"](e["pool"])}),
    ]
    if m11cfg is not None:
        steps += [
            ("m11g1", lambda e: {"act": m11_gemm1(e["px"].contiguous(), e["loc"].reshape(-1).contiguous(), state, m11cfg)}),
            ("m11g2", lambda e: {"y": m11_gemm2(e["act"], e["loc"].reshape(-1).contiguous(), state, m11cfg)}),
            ("m11pn", lambda e: {"eo": m11_post_norm(e["y"], e["loc"].reshape(-1).contiguous(), e["tw"], state, tokens)}),
        ]
    else:
        def align(e):
            s, b, n = moe_align_block_size(e["emap"], moe_block, N_EXPERTS)
            return {"sorted": s, "blocks": b, "npad": n}

        def mar1(e):
            c1 = torch.empty((tokens * TOP_K, gemm1_n), device="cuda", dtype=torch.bfloat16)
            return {"c1": moe_wna16_marlin_gemm(
                e["px"], c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, bank["workspace"],
                e["sorted"], e["blocks"], e["npad"], e["tw"], top_k=TOP_K, size_m=tokens, size_n=gemm1_n,
                size_k=size_k, **common)}

        def silu(e):
            c2 = torch.empty((tokens * TOP_K, size_n), device="cuda", dtype=torch.bfloat16)
            silu_and_mul(e["c1"].view(-1, gemm1_n), c2, expert_ids=e["emap"].view(-1), expert_step=1)
            return {"c2": c2}

        def mar2(e):
            c3 = torch.empty((tokens * TOP_K, size_k), device="cuda", dtype=torch.bfloat16)
            y = moe_wna16_marlin_gemm(
                e["c2"], c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, bank["workspace"],
                e["sorted"], e["blocks"], e["npad"], e["tw"], top_k=1, size_m=tokens * TOP_K, size_n=size_k,
                size_k=size_n, **common)
            return {"y": y.view(tokens, TOP_K, size_k)}

        steps += [("align", align), ("mar1", mar1), ("silu", silu), ("mar2", mar2),
                  ("pn", lambda e: {"eo": post_expert_norm_weighted_sum(e["y"], e["tw"], state["post_expert_w"],
                                                                        e["emap"])})]

    def unp(e):
        if tokens <= SLICED_UNPOOL_MAX_TOKENS:
            return {"ffn": F.pad(v2_unpool_slice(e["eo"], state), (0, HIDDEN - UNPOOL_SLICE))}
        return {"ffn": _gemm(e["eo"], dense["unpooler"])}

    steps += [("unp", unp),
              ("bnd2", lambda e: dict(zip(("nx", "s2", "_"), state["ffn_boundary"](
                  e["s1"], e["ffn"], branch_dtype=torch.bfloat16))))]
    return steps


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _load(ids, start):
    loc = (ids.long() - start) % N_EXPERTS
    loc = loc[loc < LOCAL_EXPERTS]
    return int(loc.numel()), int(loc.unique().numel())


def sec_ops(lab, state, deadline):
    fixed = lab.empty_graph_us()
    log(f"ops: fixed {fixed:.1f}us subtracted; us per op")
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"ops stopped before {name} (deadline)")
            break
        try:
            data = lab_input(batch, q_len, layer, seed=1000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            routes, distinct = _load(ids, int(data["local_start"]))
            v2 = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            d1 = lab.time_graph(lambda: _d1_kernel_fn(data, state)) - fixed
            steps = _v2_op_steps(data, state)
            env = {}
            for _, fn in steps:
                env.update(fn(env))
            parts = [(s, lab.time_graph(lambda fn=fn: fn(env)) - fixed) for s, fn in steps]
            total = sum(t for _, t in parts)
            log(f"ops {name} T{batch * q_len} r{routes}/e{distinct}: v2 {v2:.1f} d1 {d1:.1f} sum {total:.1f} | "
                + " ".join(f"{s} {t:.1f}" for s, t in parts))
            LAB_SUMMARY.append(f"{name}: v2 {v2:.1f} d1 {d1:.1f} ({d1 / max(v2, 0.01):.3f}x)")
            del data, env, steps
        except Exception as exc:
            lab_fail(f"ops {name}", exc)
        torch.cuda.empty_cache()


def sec_gemm(lab, state, deadline):
    fixed = lab.empty_graph_us()
    dense = state["dense"]
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
    except Exception as exc:
        tiny_gemm_bf16 = None
        lab_fail("gemm import tiny", exc)
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv
    except Exception as exc:
        hopper_bf16_gemv = None
        lab_fail("gemm import gemv", exc)
    shapes = (("qkv", dense["qkv_gate"]), ("o", dense["o"]), ("pool", dense["pooler"]),
              ("unps", dense["unpooler"][:UNPOOL_SLICE].contiguous()), ("rhi", state["router_hi_t"].t().contiguous()))
    for name, w in shapes:
        n, k = w.shape
        tb = n * k * 2 / 1e6  # MB
        out = []
        for t in (1, 8, 16, 32):
            if lab.elapsed() > deadline:
                LAB_SUMMARY.append(f"gemm stopped at {name} T{t} (deadline)")
                break
            x = torch.randn((t, k), device="cuda").to(torch.bfloat16)
            ref = F.linear(x, w).float()
            cell = []
            cands = [("cub", lambda: F.linear(x, w))]
            if tiny_gemm_bf16 is not None and t <= 16 and can_use_tiny_gemm(n, k, 16):
                cands.append(("tiny", lambda: tiny_gemm_bf16(x, w)))
            if hopper_bf16_gemv is not None and t == 1 and use_hopper_bf16_gemv(1, n, k):
                cands.append(("gemv", lambda: hopper_bf16_gemv(x, w)))
            for label, fn in cands:
                try:
                    err = float((fn().float() - ref).norm() / ref.norm())
                    us = lab.time_graph(fn) - fixed
                    cell.append(f"{label} {us:.1f}" + ("" if err < 1e-2 else f"(e{err:.0e})"))
                except Exception as exc:
                    lab_fail(f"gemm {name} T{t} {label}", exc)
            out.append(f"T{t} " + " ".join(cell))
        log(f"gemm {name} {n}x{k} {tb:.1f}MB: " + " | ".join(out))


def sec_router(lab, state, deadline):
    fixed = lab.empty_graph_us()
    cells = []
    for t in (1, 2, 4, 8):
        x = torch.randn((t, HIDDEN), device="cuda")
        ref = F.linear(x, state["router"])
        try:
            err = _v2_rel(v2_router_logits(x, state), ref)
            mine = lab.time_graph(lambda: v2_router_logits(x, state)) - fixed
            theirs = lab.time_graph(lambda: F.linear(x, state["router"])) - fixed
            cells.append(f"T{t} v2 {mine:.1f} F.linear {theirs:.1f} e{err:.0e}")
        except Exception as exc:
            lab_fail(f"router T{t}", exc)
    log("router fp32 (v2 on: %s): " % state.get("v2_router") + " | ".join(cells))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v2 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; m11 {'on' if 'm11' in state else 'OFF'}, "
        f"router {'on' if state.get('v2_router') else 'OFF'}, tiny {'on' if state.get('v2_tiny') else 'OFF'}")
    lab.section("router", sec_router, state, until=0.1 * lab.budget_s)
    lab.section("gemm", sec_gemm, state, until=0.35 * lab.budget_s)
    lab.section("ops", sec_ops, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
