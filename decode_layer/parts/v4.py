# v4 = v3 (E3c M11 experts, local ids, pre-expert norm, tiny unpooler slice) + two sibling drop-ins,
# each switched on per case (tokens, sliding) only where the lab measured a gain:
#   att: the attention session's scored v3 (parts/attn.py) for our attention segment (qkv -> QK-norm,
#        RoPE, FP8 cache store, attention, rmsnorm * sigmoid(gate));
#   pj:  the bf16_proj session's scored v2 (parts/proj.py) for qkv / o / pooler / unpooler and the
#        router, per (layer, tokens) as its CHOICE table says (cuBLAS / SGLang where it has no entry).
# v3's own FP32 router (DF22) and L2 prefetch (DF21) are not used.

V4_ATT_KEYS = ("positions", "cache_loc", "cache_seqlens", "cu_seqlens_q", "scheduler_metadata", "k_cache",
               "v_cache", "k_scale", "v_scale", "page_table", "seq_lens", "config")
# (tokens, sliding) -> {"att": bool, "pj": bool}; the DE6 lab's picks (check d48630d6950a4589, 1.336x).
# Missing cases (test-only token counts): both off.
V4_MODE = {(1, False): {'att': True, 'pj': True}, (1, True): {'att': True, 'pj': True},
           (8, False): {'att': True, 'pj': True}, (8, True): {'att': True, 'pj': True},
           (16, False): {'att': True, 'pj': True}, (16, True): {'att': True, 'pj': True},
           (32, False): {'att': True, 'pj': True}, (32, True): {'att': True, 'pj': True},
           (128, False): {'att': False, 'pj': False}, (128, True): {'att': True, 'pj': True},
           (256, False): {'att': False, 'pj': False}, (256, True): {'att': False, 'pj': False}}
V3_BUILD_PREFETCH = False  # DF21: the L2 prefetch extension isn't used; skip its nvcc time


def v4_mode(cfg, state, key):
    return state["v4_mode"].get((cfg["tokens"], bool(cfg["window"])), {}).get(key, False)


def v4_project(x, layer, n, cfg, state):
    """A dense projection: the bf16_proj session's chosen kernel for (layer, tokens) when on, else cuBLAS."""
    if v4_mode(cfg, state, "pj") and "pj_state" in state:
        c = pj_CHOICE.get((layer, x.shape[0]))
        if c is not None:
            return pj_run_choice(c, x, layer, n, state["pj_state"])
    w = state["dense"]["qkv_gate" if layer == "qkv_gate" else layer]
    return _gemm(x, w if n == w.shape[0] else w[:n])


def v4_router(ffn_input, router_input, cfg, state):
    """FP32 router logits: bf16_proj's kernel for (router, tokens) when on (fed the FP32 router input
    up to 8 tokens, as SGLang computes it there; the BF16 input with a hi + lo split above), else
    SGLang's own path."""
    tokens = ffn_input.shape[0]
    split = tokens > ROUTER_FP32_MAX_TOKENS
    if v4_mode(cfg, state, "pj") and "pj_state" in state:
        c = pj_CHOICE.get(("router", tokens))
        if c is not None:
            return pj_run_choice(c, ffn_input if split else router_input, "router", N_EXPERTS, state["pj_state"])
    if split:
        logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
        logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        return logits
    return F.linear(router_input, state["router"])


def v4_attention(data, qkv, cfg, state):
    """The attention segment: the attention session's kernels when on, else SGLang's (D1's) block."""
    if v4_mode(cfg, state, "att") and state.get("v4_att"):
        adata = {key: data[key] for key in V4_ATT_KEYS}
        adata["qkv"] = qkv.view(cfg["batch"], cfg["q_len"], -1)
        return att_kernel_fn(adata).reshape(cfg["tokens"], -1)
    return _v4_attention_d1(data, qkv, state)


def _v4_check(name, fn):
    try:
        ok, msg = fn()
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {str(exc)[-300:]}"
    print(f"[v4] {name}: {'on' if ok else 'OFF'} ({msg})")
    return ok


def _v4_small_case(batch, q_len, sliding, device, gen, ctx=3000):
    """A small decode-layer input (page 1, kv scale 1) for the self-checks."""
    tokens = batch * q_len
    seq = torch.full((batch,), ctx, dtype=torch.int64, device=device)
    max_pages = ctx + q_len
    total = batch * max_pages
    table = torch.randperm(total, generator=gen, device=device).to(torch.int32).view(batch, max_pages)
    pos = (seq[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    loc = torch.gather(table.long(), 1, pos.view(batch, q_len)).reshape(-1)
    kc = torch.randn((total, 1, 2, 64), generator=gen, device=device).to(FP8)
    vc = torch.randn((total, 1, 2, 64), generator=gen, device=device).to(FP8)
    cfg = dict(batch=batch, q_len=q_len, tokens=tokens, hidden=HIDDEN, num_q_heads=16, num_kv_heads=2, head_dim=64,
               page_size=1, window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0, n_experts=N_EXPERTS, top_k=TOP_K,
               local_experts=LOCAL_EXPERTS)
    cs = (seq + q_len).to(torch.int32)
    cu = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    return {"positions": pos, "cache_loc": loc, "cache_seqlens": cs, "cu_seqlens_q": cu, "scheduler_metadata": None,
            "k_cache": kc, "v_cache": vc, "k_scale": torch.tensor(1.0, device=device),
            "v_scale": torch.tensor(1.0, device=device), "page_table": table, "seq_lens": seq.to(torch.int32),
            "config": cfg}


def prepare(weights):
    state = _v3_prepare(weights)
    state["v3_router"] = False  # DF22: v4 routes the router through v4_router
    state["v4_mode"] = {key: dict(val) for key, val in V4_MODE.items()}
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(41)

    def pj():
        st = {"qkv_gate": state["dense"]["qkv_gate"], "o": state["dense"]["o"], "pooler": state["dense"]["pooler"],
              "unpooler": state["dense"]["unpooler"], "router": state["router"],
              "router_hi_t": state["router_hi_t"], "router_lo_t": state["router_lo_t"]}
        pj__warm(st)  # compiles each CHOICE with BF16 x, as bf16_proj's problem passes it
        # The loop below also compiles the router's <= 8-token kernels with the FP32 x v4 passes them (a
        # different Triton specialization), so no first compile can land inside a graph capture.
        worst = 0.0
        for (layer, m), c in pj_CHOICE.items():
            k = st[layer].shape[1]
            n = 1024 if layer == "unpooler" and m <= 8 else st[layer].shape[0]
            if layer == "router" and m <= ROUTER_FP32_MAX_TOKENS:
                x = torch.randn((m, k), generator=gen, device=device)  # our FP32 router input
                ref = F.linear(x, st["router"])
            elif layer == "router":
                x = torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16)
                ref = pj_sglang_path(x, layer, n, st)
            else:
                x = torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16)
                ref = F.linear(x, st[layer][:n])
            worst = max(worst, _v3_rel(pj_run_choice(c, x, layer, n, st), ref))
        state["pj_state"] = st
        return worst < 1e-2, f"{len(pj_CHOICE)} (layer, m) kernels warmed, worst rel err vs SGLang {worst:.1e}"

    if not _v4_check("projections (bf16_proj v2)", pj):
        state.pop("pj_state", None)

    def att():
        att__warm_up()
        worst, cache_bad, cache_n = 0.0, 0, 0
        for batch, q_len, sliding in ((1, 1, True), (2, 8, False), (4, 1, False), (3, 8, True)):
            data = _v4_small_case(batch, q_len, sliding, device, gen)
            tokens = batch * q_len
            qkv = (torch.randn((tokens, 2304), generator=gen, device=device) * 0.5).to(torch.bfloat16)
            k2, v2 = data["k_cache"].clone(), data["v_cache"].clone()
            mine = v4_attention_forced(data, qkv, state)
            ref_data = dict(data, k_cache=k2, v_cache=v2)
            ref = _v4_attention_d1(ref_data, qkv, state)
            worst = max(worst, _v3_rel(mine, ref))
            loc = data["cache_loc"]
            for a, b in ((data["k_cache"], k2), (data["v_cache"], v2)):
                diff = (a.view(-1, 128)[loc].view(torch.uint8).int() - b.view(-1, 128)[loc].view(torch.uint8).int()).abs()
                cache_bad += int((diff > 1).sum())
                cache_n += diff.numel()
        torch.cuda.synchronize()
        state["v4_att"] = worst < 0.2 and cache_bad <= cache_n // 1000
        return state["v4_att"], (f"rel err vs FA3 path {worst:.1e} (FA3 itself is ~0.08 off the FP32 reference); "
                                 f"new cache bytes differing by > 1 ulp: {cache_bad} of {cache_n}")

    _v4_check("attention (attention v3)", att)
    torch.cuda.synchronize()
    return state


def v4_attention_forced(data, qkv, state):
    """The attention session's path regardless of v4_mode (self-checks and labs)."""
    cfg = data["config"]
    adata = {key: data[key] for key in V4_ATT_KEYS}
    adata["qkv"] = qkv.view(cfg["batch"], cfg["q_len"], -1)
    return att_kernel_fn(adata).reshape(cfg["tokens"], -1)
