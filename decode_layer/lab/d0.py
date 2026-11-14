"""D0 recon lab (check runs only). prepare() runs the submission's prepare, then within a budget:

ops    every scored shape on judge-like inputs: local routes / distinct experts, the whole layer
       as a cold-L2 graph (D1 and SGLang's expert path), and each op alone as a cold-L2 graph.
gemm   the dense projections at every scored token count: cuBLAS (the baseline), SGLang's Hopper
       GEMV (1 token) and tiny GEMM (<= 16), and a Triton split-K skinny GEMM sweep. Error vs cuBLAS.
fa3    FA3 per shape with num_splits / pack_gqa variants, as achieved KV bandwidth.
"""

import triton.language as tl

LAB_PREPARE_LIMIT_S = 255.0  # prepare() may take 300 s; keep 45 s of margin
LAB_T = (1, 8, 16, 32, 128, 256)


# ---------------------------------------------------------------- ops

def _sglang_experts(x, topk_weights, topk_ids, state):
    """baseline.py's _experts: SGLang's tile-height rule and its post-norm (unfused below 32)."""
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for bsm in (8, 16, 32, 48, 64):
        if num_tokens * TOP_K / LOCAL_EXPERTS / bsm < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, bsm, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def _op_steps(data, state):
    """kernel_fn split into named steps. Each step reads env and returns its outputs, so a step can
    be replayed alone; running them in order once fills env with real intermediates."""
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = 16, 2, 64
    dense = state["dense"]
    bank = state["bank"]
    window = cfg["window"]
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    moe_block = block_size_m(tokens)
    size_n, size_k = bank["size_n"], D_EXPERT
    gemm1_n = 2 * size_n
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)

    def qk_norm(e):
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](e["qkv"], data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](e["qkv"])
        return {"q": q, "k": k}

    def kv_store(e):
        v = e["qkv"][:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        store_cache(e["k"].reshape(tokens, hkv * d).to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["v_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        return {}

    def fa3(e):
        out = flash_attn_with_kvcache(
            q=e["q8"], k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"])
        return {"attn": out[0] if isinstance(out, tuple) else out}

    def boundary1(e):
        f, s, r = state["attn_boundary"](data["stream"], e["a"], branch_dtype=torch.bfloat16,
                                         router_dtype=None if split_router else torch.float32)
        return {"h": f, "s1": s, "rin": r}

    def router(e):
        if split_router:
            logits = torch.mm(e["h"], state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(e["h"], state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(e["rin"], state["router"])
        return {"logits": logits}

    def routing(e):
        w, ids = state["routing"](None, e["logits"], TOP_K, True)
        return {"tw": w, "ids": ids}

    def local(e):
        lid = bank["expert_map"][(e["ids"] - data["local_start"]) % N_EXPERTS].contiguous()
        return {"lid": lid, "tw32": e["tw"].to(torch.float32).contiguous()}

    def align(e):
        s, b, n = moe_align_block_size(e["lid"], moe_block, N_EXPERTS)
        return {"sorted": s, "blocks": b, "npad": n}

    def marlin1(e):
        c1 = torch.empty((tokens * TOP_K, gemm1_n), device="cuda", dtype=torch.bfloat16)
        c1 = moe_wna16_marlin_gemm(
            e["px"], c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, bank["workspace"],
            e["sorted"], e["blocks"], e["npad"], e["tw32"], top_k=TOP_K,
            size_m=tokens, size_n=gemm1_n, size_k=size_k, **common)
        return {"c1": c1}

    def silu(e):
        c2 = torch.empty((tokens * TOP_K, size_n), device="cuda", dtype=torch.bfloat16)
        silu_and_mul(e["c1"].view(-1, gemm1_n), c2, expert_ids=e["lid"].view(-1), expert_step=1)
        return {"c2": c2}

    def marlin2(e):
        c3 = torch.empty((tokens * TOP_K, size_k), device="cuda", dtype=torch.bfloat16)
        y = moe_wna16_marlin_gemm(
            e["c2"], c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, bank["workspace"],
            e["sorted"], e["blocks"], e["npad"], e["tw32"], top_k=1,
            size_m=tokens * TOP_K, size_n=size_k, size_k=size_n, **common)
        return {"y": y.view(tokens, TOP_K, size_k)}

    def unpool(e):
        if tokens <= SLICED_UNPOOL_MAX_TOKENS:
            return {"ffn": F.pad(_gemm(e["eo"], dense["unpooler"][:UNPOOL_SLICE]), (0, HIDDEN - UNPOOL_SLICE))}
        return {"ffn": _gemm(e["eo"], dense["unpooler"])}

    return [
        ("qkv gemm", lambda e: {"qkv": _gemm(data["attn_input"], dense["qkv_gate"])}),
        ("qk norm", qk_norm),
        ("kv cast+store", kv_store),
        ("q fp8", lambda e: {"q8": e["q"].view(tokens, hq, d).to(FP8)}),
        ("fa3", fa3),
        ("norm gate", lambda e: {"og": state["o_norm_gate"](e["attn"].reshape(tokens, hq * d).to(torch.bfloat16),
                                                            e["qkv"][:, hq * d + 2 * hkv * d:])}),
        ("o gemm", lambda e: {"a": _gemm(e["og"], dense["o"])}),
        ("boundary1", boundary1),
        ("pooler gemm", lambda e: {"pool": _gemm(e["h"], dense["pooler"])}),
        ("router", router),
        ("routing", routing),
        ("local ids", local),
        ("pre-expert norm", lambda e: {"px": state["pre_expert_norm"](e["pool"])}),
        ("align", align),
        ("marlin1", marlin1),
        ("silu", silu),
        ("marlin2", marlin2),
        ("post-norm fused", lambda e: {"eo": post_expert_norm_weighted_sum(e["y"], e["tw32"], state["post_expert_w"],
                                                                           e["lid"])}),
        ("unpooler gemm", unpool),
        ("boundary2", lambda e: dict(zip(("nx", "s2", "_"), state["ffn_boundary"](
            e["s1"], e["ffn"], branch_dtype=torch.bfloat16)))),
    ]


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _load(ids, start):
    loc = (ids.long() - start) % N_EXPERTS
    loc = loc[loc < LOCAL_EXPERTS]
    return int(loc.numel()), int(loc.unique().numel())


def sec_ops(lab, state, deadline):
    fixed = lab.empty_graph_us()
    log(f"ops: empty-graph replay {fixed:.1f} us (subtract from every time below; the judge does)")
    summary = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            log(f"ops: stopped before {name}, deadline")
            break
        data = lab_input(batch, q_len, layer, seed=1000 + i)
        _, _, ids = kernel_fn(data, state)
        data["local_start"].fill_(_busiest_start(ids))
        routes, distinct = _load(ids, int(data["local_start"]))
        d1 = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
        steps = _op_steps(data, state)
        env = {}
        for _, fn in steps:
            env.update(fn(env))
        # The same layer as one chain of steps (no side stream), with D1's expert steps or with
        # SGLang's whole expert path in their place.
        names = [n for n, _ in steps]
        first, last = names.index("align"), names.index("post-norm fused")
        sgl_experts = ("experts SGLang", lambda e: {"eo": _sglang_experts(
            e["px"], e["tw"], (e["ids"] - data["local_start"]) % N_EXPERTS, state)})
        chains = {"D1": steps, "SGLang": steps[:first] + [sgl_experts] + steps[last + 1:]}

        def run_chain(chain):
            e = dict(env)
            for _, fn in chain:
                e.update(fn(e))

        chain_us = {k: lab.time_graph(lambda c=c: run_chain(c)) - fixed for k, c in chains.items()}
        parts = []
        for step, fn in steps:
            parts.append((step, lab.time_graph(lambda fn=fn: fn(env)) - fixed))
        total = sum(t for _, t in parts)
        old_post = lab.time_graph(lambda: _sglang_post(env, state)) - fixed
        kv = kv_bytes(data)
        log(f"ops {name}: T={batch * q_len} local routes {routes} distinct {distinct} kv {kv / 1e6:.1f} MB | "
            f"layer D1 {d1:.1f} us | chain D1 {chain_us['D1']:.1f}, chain SGLang experts {chain_us['SGLang']:.1f} us "
            f"| sum of ops {total:.1f} us")
        log(f"ops {name} parts: " + ", ".join(f"{s} {t:.1f}" for s, t in parts)
            + f" | SGLang post-norm {old_post:.1f}")
        summary.append((name, weight, d1, d1 + chain_us["SGLang"] - chain_us["D1"]))
        del data, env, steps
        torch.cuda.empty_cache()
    if summary:
        log("ops summary (name, weight, D1 layer us, est. SGLang layer us = D1 + chain difference): "
            + "; ".join(f"{n} w{w} {a:.1f} {b:.1f}" for n, w, a, b in summary))


def _sglang_post(env, state):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    return fused_post_expert_norm_weighted_sum(env["y"], env["tw32"], norm_weight=state["post_expert_w"],
                                               eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=env["lid"])


# ---------------------------------------------------------------- gemm

@triton.jit
def _lab_skinny_gemm(x_ptr, w_ptr, out_ptr, T, N, K, stride_x, stride_w,
                     BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT: tl.constexpr, OUT_BF16: tl.constexpr):
    """out[s, t, n] = sum over K-split s of w[n, k] x[t, k], weights as the MMA's M side.
    Requires N % BLOCK_N == 0 and K % (SPLIT * BLOCK_K) == 0. OUT_BF16 (SPLIT == 1): bf16 [T, N]."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    k_per = K // SPLIT
    k0 = pid_s * k_per
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    t_ok = offs_t[None, :] < T
    for k in range(k0, k0 + k_per, BLOCK_K):
        w = tl.load(w_ptr + offs_n[:, None] * stride_w + (k + offs_k)[None, :], eviction_policy="evict_first")
        x = tl.load(x_ptr + offs_t[None, :] * stride_x + (k + offs_k)[:, None], mask=t_ok, other=0.0)
        acc = tl.dot(w, x, acc)
    out = tl.trans(acc)
    if OUT_BF16:
        ptrs = out_ptr + offs_t[:, None] * N + offs_n[None, :]
        tl.store(ptrs, out.to(tl.bfloat16), mask=offs_t[:, None] < T)
    else:
        ptrs = out_ptr + pid_s * T * N + offs_t[:, None] * N + offs_n[None, :]
        tl.store(ptrs, out, mask=offs_t[:, None] < T)


@triton.jit
def _lab_split_reduce(part_ptr, out_ptr, TN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < TN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * TN + offs, mask=ok, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=ok)


def lab_skinny_gemm(x, w, cfg, part=None, out=None):
    bn, bk, split, warps, stages = cfg
    t, k = x.shape
    n = w.shape[0]
    bt = 16 if t <= 16 else 32 if t <= 32 else 64
    if out is None:
        out = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
    grid = (n // bn, split, triton.cdiv(t, bt))
    if split == 1:
        _lab_skinny_gemm[grid](x, w, out, t, n, k, x.stride(0), w.stride(0), BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk,
                               SPLIT=1, OUT_BF16=True, num_warps=warps, num_stages=stages)
        return out
    if part is None:
        part = torch.empty((split, t, n), dtype=torch.float32, device=x.device)
    _lab_skinny_gemm[grid](x, w, part, t, n, k, x.stride(0), w.stride(0), BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk,
                           SPLIT=split, OUT_BF16=False, num_warps=warps, num_stages=stages)
    _lab_split_reduce[(triton.cdiv(t * n, 1024),)](part, out, t * n, SPLIT=split, BLOCK=1024, num_warps=4)
    return out


def _gemm_shapes(state):
    dense = state["dense"]
    return (
        ("qkv", dense["qkv_gate"], LAB_T),
        ("o", dense["o"], LAB_T),
        ("pooler", dense["pooler"], LAB_T),
        ("unpool slice", dense["unpooler"][:UNPOOL_SLICE], (1, 8)),
        ("unpooler", dense["unpooler"], (16, 32, 128, 256)),
        ("router hi", state["router_hi_t"].t().contiguous(), LAB_T),
    )


def _skinny_configs(n, k):
    out = []
    for bn in (64, 128):
        for bk in (128, 256):
            for split in (1, 2, 4, 8, 16):
                if n % bn or k % (split * bk):
                    continue
                for warps in (4, 8):
                    out.append((bn, bk, split, warps, 4))
    return out


def _bw(nbytes, us):
    """TB/s; a time at or below the replay's fixed cost reads as the bandwidth of 0.01 us."""
    return nbytes / (max(us, 0.01) * 1e-6) / 1e12


def sec_gemm(lab, state, deadline):
    fixed = lab.empty_graph_us()
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv
    except Exception as exc:
        hopper_bf16_gemv = None
        log(f"gemm: hopper_bf16_gemv unavailable: {_error_text(exc, 150)}")
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
    except Exception as exc:
        tiny_gemm_bf16 = None
        log(f"gemm: tiny_gemm_bf16 unavailable: {_error_text(exc, 150)}")
    best = {}
    for name, w, ts in _gemm_shapes(state):
        n, k = w.shape
        nbytes = n * k * 2
        for t in ts:
            if lab.elapsed() > deadline:
                log(f"gemm: stopped before {name} T={t}, deadline")
                return
            x = torch.randn((t, k), device="cuda").to(torch.bfloat16)
            ref = F.linear(x, w).float()
            res = {}

            def run(label, fn):
                try:
                    err = float((fn().float() - ref).norm() / ref.norm())
                    us = lab.time_graph(fn) - fixed
                    res[label] = (us, err)
                except Exception as exc:
                    log(f"gemm {name} T={t} {label}: FAILED {type(exc).__name__}: {_error_text(exc, 200)}")

            run("cublas", lambda: F.linear(x, w))
            if hopper_bf16_gemv is not None and t == 1 and use_hopper_bf16_gemv(1, n, k):
                run("hopper_gemv", lambda: hopper_bf16_gemv(x, w))
            if tiny_gemm_bf16 is not None and t <= 16 and can_use_tiny_gemm(n, k, 16):
                run("tiny_gemm", lambda: tiny_gemm_bf16(x, w))
            cfgs = _skinny_configs(n, k)
            if t != ts[0] and name in best:
                cfgs = best[name]  # the T sweep's top configs from the first token count
            for cfg in cfgs:
                if lab.elapsed() > deadline:
                    break
                part = torch.empty((cfg[2], t, n), dtype=torch.float32, device="cuda") if cfg[2] > 1 else None
                out = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")
                run("triton " + "/".join(map(str, cfg)), lambda cfg=cfg, part=part, out=out:
                    lab_skinny_gemm(x, w, cfg, part, out))
            tri = sorted((v[0], lbl) for lbl, v in res.items() if lbl.startswith("triton") and v[1] < 1e-2)
            if t == ts[0] and tri:
                best[name] = [tuple(int(p) for p in lbl.split()[1].split("/")) for _, lbl in tri[:4]]
            cub = res.get("cublas", (float("nan"), 0))[0]
            shown = sorted(res.items(), key=lambda kv: kv[1][0])
            top = [f"{lbl} {us:.1f}us {_bw(nbytes, us):.2f}TB/s e{err:.0e}" for lbl, (us, err) in shown
                   if not lbl.startswith("triton")] + [f"{lbl} {us:.1f}us {_bw(nbytes, us):.2f}TB/s e{err:.0e}"
                                                       for lbl, (us, err) in shown if lbl.startswith("triton")][:3]
            log(f"gemm {name} [{n}x{k}] T={t}: cublas {cub:.1f}us ({_bw(nbytes, cub):.2f}TB/s) | " + "; ".join(top))
            del x, ref
    # The FP32 router as SGLang runs it at <= 8 tokens.
    for t in (1, 8):
        xr = torch.randn((t, HIDDEN), device="cuda")
        us = lab.time_graph(lambda: F.linear(xr, state["router"])) - fixed
        log(f"gemm router fp32 T={t}: F.linear {us:.1f}us ({_bw(256 * HIDDEN * 4, us):.2f}TB/s)")


# ---------------------------------------------------------------- fa3

def sec_fa3(lab, deadline):
    fixed = lab.empty_graph_us()
    for i, (name, batch, q_len, layer, _) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            log(f"fa3: stopped before {name}, deadline")
            return
        data = lab_input(batch, q_len, layer, seed=2000 + i)
        cfg = data["config"]
        tokens, window = cfg["tokens"], cfg["window"]
        q = torch.randn((tokens, 16, 64), device="cuda").to(FP8)
        kv = kv_bytes(data)

        def call(sched, **kw):
            return flash_attn_with_kvcache(
                q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
                cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=q_len,
                softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=sched, **kw)

        variants = [("baseline", dict(sched=data["scheduler_metadata"]))]
        variants += [(f"splits {s}", dict(sched=None, num_splits=s)) for s in (1, 2, 4, 8, 16, 32, 64)]
        variants += [("pack_gqa T", dict(sched=None, pack_gqa=True)), ("pack_gqa F", dict(sched=None, pack_gqa=False))]
        ref = call(**variants[0][1])
        ref = (ref[0] if isinstance(ref, tuple) else ref).float()
        res = []
        for label, kw in variants:
            try:
                out = call(**kw)
                out = (out[0] if isinstance(out, tuple) else out).float()
                err = float((out - ref).norm() / ref.norm())
                us = lab.time_graph(lambda kw=kw: call(**kw)) - fixed
                res.append(f"{label} {us:.1f}us {_bw(kv, us):.2f}TB/s" + (f" e{err:.0e}" if err > 1e-3 else ""))
            except Exception as exc:
                res.append(f"{label} FAILED {type(exc).__name__}: {_error_text(exc, 120)}")
        log(f"fa3 {name} kv {kv / 1e6:.1f} MB: " + "; ".join(res))
        del data, q
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- entry

def _lab_env():
    import importlib.metadata as md
    vers = {}
    for pkg in ("torch", "triton", "sgl-kernel", "sglang", "flashinfer-python"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = "?"
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    log(f"env: {vers}; {props.name} {props.multi_processor_count} SMs; free {free / 2**30:.0f} of {total / 2**30:.0f} GiB")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"submission prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s")
    try:
        _lab_env()
    except Exception as exc:
        log(f"env: FAILED {_error_text(exc)}")
    lab.section("ops", sec_ops, state, until=0.45 * lab.budget_s)
    lab.section("gemm", sec_gemm, state, until=0.8 * lab.budget_s)
    lab.section("fa3", sec_fa3)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"lab total {lab.elapsed():.1f}s")
    return state
