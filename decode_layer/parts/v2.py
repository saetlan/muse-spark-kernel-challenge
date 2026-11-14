# v2: D1 plus three changes, each checked against the path it replaces in prepare() and dropped
# (with a printed line) if that check fails:
#
# * Experts at <= 16 tokens run the MoE session's M11 CUDA kernels (fragment-ordered NVFP4 weights,
#   swap-AB mma.sync, SwiGLU in GEMM1's epilogue, no sort/align kernel), with E3's configs. 17+ tokens
#   keep D1's Marlin path. M11 gets this GPU's local ids (0-31 local, 32+ not), so its start is 0.
# * The FP32 router at <= 8 tokens is one CUDA-core kernel reading the 8.4 MB weight once, in FP32
#   (no TF32, no BF16 split): SGLang's F.linear takes 20.8 us at 8 tokens (DE0, DF6).
# * The unpooler slice (1024 x 4096) at <= 8 tokens runs SGLang's tiny_gemm_bf16 (2.7 vs 5.9 us at 1
#   token, DE0).

@triton.jit
def _v2_router_fp32_kernel(x_ptr, w_ptr, out_ptr, T, N, K: tl.constexpr):
    """out[t, n] = sum_k x[t, k] * w[n, k] in FP32; one program per router row n."""
    n = tl.program_id(0)
    offs = tl.arange(0, K)
    w = tl.load(w_ptr + n * K + offs)
    for t in range(T):
        x = tl.load(x_ptr + t * K + offs)
        tl.store(out_ptr + t * N + n, tl.sum(w * x, axis=0))


def v2_router_logits(router_input, state):
    """FP32 router logits [T, 256] for T <= 8 tokens (router_input: FP32 [T, 8192])."""
    if not state.get("v2_router"):
        return F.linear(router_input, state["router"])
    x = router_input.contiguous()
    w = state["router"]
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)
    _v2_router_fp32_kernel[(w.shape[0],)](x, w, out, x.shape[0], w.shape[0], K=w.shape[1], num_warps=8)
    return out


def v2_unpool_slice(expert_output, state):
    """This GPU's 1024 unpooler columns for T <= 8 tokens."""
    if state.get("v2_tiny") is None:
        return _gemm(expert_output, state["dense"]["unpooler"][:UNPOOL_SLICE])
    return state["v2_tiny"](expert_output.contiguous(), state["v2_unpool_slice"])


def v2_experts(x, topk_weights, local_ids, state):
    """The routed experts: M11 where configured for this token count, else D1's Marlin path."""
    cfg = m11_config_for(state, x.shape[0]) if "m11" in state else None
    if cfg is None:
        return _experts(x, topk_weights, local_ids, state)
    return m11_run(x.contiguous(), local_ids.to(torch.int32), topk_weights.to(torch.float32), state, cfg)


def _v2_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _v2_setup_router(state, device):
    try:
        state["v2_router"] = True
        x = torch.randn((8, HIDDEN), device=device)
        errs = [_v2_rel(v2_router_logits(x[:t], state), F.linear(x[:t], state["router"])) for t in (1, 3, 8)]
        state["v2_router"] = max(errs) < 1e-5
        print(f"[v2] fp32 router kernel: rel err vs F.linear {max(errs):.1e} -> {'on' if state['v2_router'] else 'OFF'}")
    except Exception as exc:
        state["v2_router"] = False
        print(f"[v2] fp32 router kernel OFF: {type(exc).__name__}: {str(exc)[-300:]}")


def _v2_setup_tiny(state, device):
    state["v2_tiny"] = None
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
        w = state["dense"]["unpooler"][:UNPOOL_SLICE].contiguous()
        if not can_use_tiny_gemm(w.shape[0], w.shape[1], 16):
            print("[v2] tiny_gemm OFF: can_use_tiny_gemm is False for the unpooler slice")
            return
        state["v2_unpool_slice"] = w
        x = torch.randn((8, D_EXPERT), device=device).to(torch.bfloat16)
        errs = [_v2_rel(tiny_gemm_bf16(x[:t].contiguous(), w), F.linear(x[:t], w)) for t in (1, 3, 8)]
        if max(errs) < 1e-2:
            state["v2_tiny"] = tiny_gemm_bf16
        print(f"[v2] tiny_gemm unpooler slice: rel err vs cuBLAS {max(errs):.1e} -> {'on' if state['v2_tiny'] else 'OFF'}")
    except Exception as exc:
        print(f"[v2] tiny_gemm OFF: {type(exc).__name__}: {str(exc)[-300:]}")


def _v2_setup_m11(state, weights, device):
    try:
        state["norm_w"] = state["post_expert_w"]
        state["m11_configs"] = M11_CONFIGS
        m11_setup(state, weights)
        for _, cfg in M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
        gen = torch.Generator(device=device).manual_seed(7)
        worst = 0.0
        for tokens in (1, 8, 16):
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            ids = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            ids[:, :LOCAL_EXPERTS] += 0.5  # most routes local, like the busiest GPU
            ids = ids.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            ref = _experts(x, tw, ids, state)
            got = v2_experts(x, tw, ids, state)
            worst = max(worst, _v2_rel(got, ref))
        torch.cuda.synchronize()
        if not worst < 2e-2:
            raise RuntimeError(f"M11 vs Marlin rel err {worst:.2e}")
        print(f"[v2] M11 on at < 17 tokens: rel err vs D1's Marlin {worst:.1e}")
    except Exception as exc:
        for key in ("m11", "m11_ext"):
            state.pop(key, None)
        print(f"[v2] M11 OFF: {type(exc).__name__}: {str(exc)[-400:]}")


def prepare(weights):
    state = _d1_prepare(weights)
    device = weights["qkv_gate"].device
    _v2_setup_router(state, device)
    _v2_setup_tiny(state, device)
    _v2_setup_m11(state, weights, device)
    torch.cuda.synchronize()
    return state
