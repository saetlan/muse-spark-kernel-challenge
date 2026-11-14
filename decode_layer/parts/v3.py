# v3: D1 + E3c's M11 experts (configs re-picked on decode's routing by the lab) + X7 L2 prefetch during
# attention + an exact FP32 router that reduces once + fused local ids + a Triton pre-expert RMSNorm.
# Every new path is checked in prepare() against the one it replaces and turned off (printed) if it
# fails; v3_kernel_fn is D1's kernel_fn with a handful of call sites swapped (local/make_v3.py).

V3_ROUTER_NR = tl.constexpr(4)  # router rows per program
V3_ROUTER_BK = tl.constexpr(256)


@triton.jit
def _v3_router_kernel(x_ptr, w_ptr, out_ptr, T, N, K: tl.constexpr, TB: tl.constexpr):
    """out[t, n] = sum_k x[t, k] * w[n, k] in FP32 (no tl.dot, no TF32): each program takes 4 router rows,
    accumulates [4, TB, 256] products across K and reduces once at the end."""
    n0 = tl.program_id(0) * V3_ROUTER_NR
    rows = n0 + tl.arange(0, V3_ROUTER_NR)
    offs_t = tl.arange(0, TB)
    offs_k = tl.arange(0, V3_ROUTER_BK)
    acc = tl.zeros((V3_ROUTER_NR, TB, V3_ROUTER_BK), dtype=tl.float32)
    t_ok = offs_t[:, None] < T
    for k0 in range(0, K, V3_ROUTER_BK):
        w = tl.load(w_ptr + rows[:, None] * K + k0 + offs_k[None, :])
        x = tl.load(x_ptr + offs_t[:, None] * K + k0 + offs_k[None, :], mask=t_ok, other=0.0)
        acc += w[:, None, :] * x[None, :, :]
    res = tl.sum(acc, axis=2)  # [NR, TB]
    tl.store(out_ptr + offs_t[None, :] * N + rows[:, None], res, mask=offs_t[None, :] < T)


def v3_router_logits(router_input, state):
    if not state.get("v3_router"):
        return F.linear(router_input, state["router"])
    x = router_input.contiguous()
    t = x.shape[0]
    w = state["router"]
    out = torch.empty((t, w.shape[0]), dtype=torch.float32, device=x.device)
    tb = 1 if t <= 1 else 2 if t <= 2 else 4 if t <= 4 else 8
    _v3_router_kernel[(w.shape[0] // 4,)](x, w, out, t, w.shape[0], K=w.shape[1], TB=tb, num_warps=8)
    return out


@triton.jit
def _v3_local_ids_kernel(ids_ptr, start_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = (ids - local_start) mod 256 as int32, in one launch (ids int32, global expert ids 0-255)."""
    offs = tl.arange(0, BLOCK)
    ok = offs < n
    start = tl.load(start_ptr)
    v = tl.load(ids_ptr + offs, mask=ok, other=0) - start
    v = tl.where(v < 0, v + 256, v)
    tl.store(out_ptr + offs, v, mask=ok)


def v3_local_ids(topk_ids, local_start, state):
    if not state.get("v3_lid"):
        return (topk_ids - local_start) % N_EXPERTS
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids, dtype=torch.int32)
    n = ids.numel()
    _v3_local_ids_kernel[(1,)](ids, local_start, out, n, BLOCK=triton.next_power_of_2(max(n, 16)), num_warps=4)
    return out


@triton.jit
def _v3_rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, H: tl.constexpr):
    """out[t] = bf16(x[t] * rsqrt(mean(x[t]^2) + eps) * w) with FP32 math, as F.rms_norm on BF16 input."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    x = tl.load(x_ptr + t.to(tl.int64) * H + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    tl.store(out_ptr + t.to(tl.int64) * H + offs, (x * tl.rsqrt(var + eps) * w).to(tl.bfloat16))


def v3_pre_expert_norm(x, state):
    if not state.get("v3_norm"):
        return state["pre_expert_norm"](x)
    x = x.contiguous()
    out = torch.empty_like(x)
    _v3_rmsnorm_kernel[(x.shape[0],)](x, state["v3_pre_expert_w"], out, RMS_EPS, H=x.shape[1], num_warps=8)
    return out


def v3_experts(expert_input, topk_weights, local_ids, state):
    """Pre-expert norm, then M11 where configured for this token count, else D1's Marlin path."""
    x = v3_pre_expert_norm(expert_input, state)
    cfg = m11_config_for(state, x.shape[0]) if "m11" in state else None
    if cfg is None:
        return _experts(x, topk_weights, local_ids, state)
    return m11_run(x, local_ids.to(torch.int32), topk_weights.to(torch.float32), state, cfg)


def v3_unpool_slice(expert_output, state):
    """This GPU's 1024 unpooler columns for T <= 8 tokens (SGLang's tiny_gemm, as v2)."""
    if state.get("v3_tiny") is None:
        return _gemm(expert_output, state["dense"]["unpooler"][:UNPOOL_SLICE])
    return state["v3_tiny"](expert_output.contiguous(), state["v3_unpool_slice"])


# ---------------------------------------------------------------- X7: L2 prefetch during attention

_V3_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

// One thread per 64 KB chunk of two byte ranges queues a bulk L2 prefetch with an evict-last policy.
__global__ void v3_l2_prefetch_kernel(const uint8_t* p0, long long n0, const uint8_t* p1, long long n1) {
  constexpr long long CHUNK = 65536;
  long long off = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * CHUNK;
  const uint8_t* p = nullptr;
  long long left = 0;
  if (off < n0) {
    p = p0 + off;
    left = n0 - off;
  } else {
    off -= (n0 + CHUNK - 1) / CHUNK * CHUNK;
    if (off < 0 || off >= n1) return;
    p = p1 + off;
    left = n1 - off;
  }
  const uint32_t size = static_cast<uint32_t>((left < CHUNK ? left : CHUNK) & ~15LL);
  if (size == 0) return;
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(p), "r"(size), "l"(pol) : "memory");
}

void v3_l2_prefetch(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t chunks = (na + 65535) / 65536 + (nb + 65535) / 65536;
  if (chunks == 0) return;
  const int threads = 128;
  const int blocks = static_cast<int>((chunks + threads - 1) / threads);
  v3_l2_prefetch_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint8_t*>(a.data_ptr()), na,
                                                       reinterpret_cast<const uint8_t*>(b.data_ptr()), nb);
}
"""

_V3_CPP = "void v3_l2_prefetch(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb);"


def v3_source():
    return _V3_CUDA


def v3_build():
    return load_inline(
        name="v3_ext", cpp_sources=_V3_CPP, cuda_sources=_V3_CUDA, functions=["v3_l2_prefetch"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="v3_"), verbose=False)


def v3_prefetch_bytes(state, cfg):
    """Bytes of o + pooler to pull into L2 for this case (0 = off), keyed by (tokens, sliding)."""
    if "v3_ext" not in state:
        return 0
    return state["v3_prefetch"].get((cfg["tokens"], bool(cfg["window"])), 0)


def v3_prefetch_fork(state, cfg):
    """After the qkv GEMM: on a side stream, queue L2 prefetches of o and the head of the pooler, so
    they stream while attention (latency-bound on these shapes) leaves HBM idle."""
    nbytes = v3_prefetch_bytes(state, cfg)
    if nbytes <= 0:
        return False
    main, side = torch.cuda.current_stream(), state["v3_stream"]
    side.wait_stream(main)
    o, pool = state["dense"]["o"], state["dense"]["pooler"]
    no = min(nbytes, o.numel() * 2)
    npool = min(max(nbytes - no, 0), pool.numel() * 2)
    with torch.cuda.stream(side):
        state["v3_ext"].v3_l2_prefetch(o, no, pool, npool)
    return True


def v3_prefetch_join(state, forked):
    if forked:
        torch.cuda.current_stream().wait_stream(state["v3_stream"])


# ---------------------------------------------------------------- setup

def _v3_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def v3_m11_setup(state, weights):
    """m11_setup without the M12 (TMA) instances, which v3 never runs: same M11 kernels, less nvcc."""
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(M11_INSTANCES, ()),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m11_"), verbose=False)
    state["m11"] = m11_bank(weights, 0, 32)


def _v3_check(name, fn):
    try:
        ok, msg = fn()
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {str(exc)[-300:]}"
    print(f"[v3] {name}: {'on' if ok else 'OFF'} ({msg})")
    return ok


def prepare(weights):
    state = _d1_prepare(weights)
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(11)

    def router():
        state["v3_router"] = True
        x = torch.randn((8, HIDDEN), generator=gen, device=device)
        err = max(_v3_rel(v3_router_logits(x[:t], state), F.linear(x[:t], state["router"])) for t in (1, 2, 3, 5, 8))
        return err < 1e-5, f"rel err vs F.linear {err:.1e}"

    state["v3_router"] = _v3_check("fp32 router (one reduction)", router)

    def lid():
        state["v3_lid"] = True
        ids = torch.randint(0, 256, (16, TOP_K), generator=gen, device=device, dtype=torch.int32)
        ls = torch.tensor([96], dtype=torch.int32, device=device)
        same = all(bool((v3_local_ids(ids[:t], ls, state) == ((ids[:t] - ls) % N_EXPERTS)).all()) for t in (1, 8, 16))
        # compile every block size a case can need (BLOCK = next power of 2 of 8 x tokens, up to 320 tokens)
        for blk_tokens in (2, 4, 8, 16, 32, 64, 128, 256, 320):
            big = torch.randint(0, 256, (blk_tokens, TOP_K), generator=gen, device=device, dtype=torch.int32)
            same &= bool((v3_local_ids(big, ls, state) == ((big - ls) % N_EXPERTS)).all())
        return same, "bit-exact vs torch, block sizes 16-4096 compiled"

    state["v3_lid"] = _v3_check("local ids kernel", lid)

    def norm():
        state["v3_norm"] = True
        state["v3_pre_expert_w"] = state["pre_expert_norm"]._get_effective_weight().detach().to(torch.bfloat16).contiguous()
        x = (3 * torch.randn((16, D_EXPERT), generator=gen, device=device)).to(torch.bfloat16)
        err = max(_v3_rel(v3_pre_expert_norm(x[:t], state), state["pre_expert_norm"](x[:t])) for t in (1, 8, 16))
        return err < 1e-2, f"rel err vs MuseSpark1xRMSNorm {err:.1e}"

    state["v3_norm"] = _v3_check("pre-expert rmsnorm", norm)

    def tiny():
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
        w = state["dense"]["unpooler"][:UNPOOL_SLICE].contiguous()
        state["v3_tiny"] = None
        if not can_use_tiny_gemm(w.shape[0], w.shape[1], 16):
            return False, "can_use_tiny_gemm is False"
        state["v3_unpool_slice"] = w
        x = torch.randn((8, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        err = max(_v3_rel(tiny_gemm_bf16(x[:t].contiguous(), w), F.linear(x[:t], w)) for t in (1, 3, 8))
        if err < 1e-2:
            state["v3_tiny"] = tiny_gemm_bf16
        return err < 1e-2, f"rel err vs cuBLAS {err:.1e}"

    _v3_check("tiny_gemm unpooler slice", tiny)

    def m11():
        state["norm_w"] = state["post_expert_w"]
        state["m11_configs"] = V3_M11_CONFIGS
        v3_m11_setup(state, weights)
        worst = 0.0
        for _, cfg in V3_M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
        for tokens in (1, 8, 16, 32):
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            sc[:, :LOCAL_EXPERTS] += 0.5
            ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            cfg = m11_config_for(state, tokens)
            if cfg is not None:
                worst = max(worst, _v3_rel(m11_run(x, ids, tw, state, cfg), _experts(x, tw, ids, state)))
        torch.cuda.synchronize()
        return worst < 2e-2, f"DE5 configs, rel err vs D1's Marlin {worst:.1e}"

    if not _v3_check("M11 (E3c)", m11):
        for key in ("m11", "m11_ext"):
            state.pop(key, None)

    def pref():
        state["v3_ext"] = v3_build()
        state["v3_stream"] = torch.cuda.Stream(device)
        state["v3_prefetch"] = dict(V3_PREFETCH)
        o = state["dense"]["o"]
        state["v3_ext"].v3_l2_prefetch(o, o.numel() * 2, state["dense"]["pooler"], 1 << 20)
        torch.cuda.synchronize()
        return True, f"{len(V3_PREFETCH)} cases enabled"

    if V3_BUILD_PREFETCH and not _v3_check("L2 prefetch", pref):
        state.pop("v3_ext", None)
    torch.cuda.synchronize()
    return state


# Per-case L2 prefetch bytes, keyed by (tokens, sliding). DE5 measured no gain on any shape (DF21): off.
V3_PREFETCH = {}
V3_BUILD_PREFETCH = True
# M11 configs chosen by the DE5 lab on decode's own routing (job 2887af7404324f6d, DF20); 33+ tokens: Marlin.
V3_M11_CONFIGS = ((5, {'s1': {'nt': 1, 'd': 4, 'ks': 4, 'c': 8}, 's2': {'tt': 1, 'nt': 1, 'd': 4, 'ks': 4, 'c': 4}}),
                  (9, {'s1': {'nt': 1, 'd': 4, 'ks': 4, 'c': 8}, 's2': {'tt': 2, 'nt': 1, 'd': 4, 'ks': 4, 'c': 8}}),
                  (17, {'s1': {'nt': 1, 'd': 4, 'ks': 2, 'c': 8}, 's2': {'tt': 2, 'nt': 1, 'd': 4, 'ks': 4, 'c': 8}}),
                  (33, {'s1': {'nt': 2, 'd': 2, 'ks': 4, 'c': 4}, 's2': {'tt': 2, 'nt': 2, 'd': 2, 'ks': 4, 'c': 8}}),
                  (321, None))
