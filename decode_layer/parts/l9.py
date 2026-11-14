# L9: L2 staging of upcoming weights in the windows where HBM is idle, on the side stream (prefetch hints only:
# no data changes, nothing is cached across calls). Spread over the whole grid (DF25: v3's X7 queued every
# request from 2 SMs). Per case (tokens, sliding) the lab picks:
#   o      the o projection's 16.8 MB during attention (fork right after the qkv projection);
#   e13/e2 right after routing, while the pooler streams: the first e/16 of every M11 warp slice (codes and
#          scales) of every local expert the routing hit, GEMM1's (w13) and/or GEMM2's (w2), with the slice
#          geometry of the M11 config the call will use (the same fraction for every task, so no task waits
#          on DRAM more than another);
#   slice  the unpooler's 1024-row slice (8.4 MB) during the expert stage (<= 8 tokens);
#   pm     the prefetch form: 0 cp.async.bulk.prefetch.L2 (one request per 4 KB / slice), 1 prefetch.global.L2
#          per 128-B line.

_L9_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace l9 {
__device__ __forceinline__ void pf(const void* p, uint32_t bytes, int mode) {
  if (mode == 0) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes));
  } else {
    const char* c = static_cast<const char*>(p);
    for (uint32_t o = 0; o < bytes; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
  }
}

// [p0, p0 + n0) then [p1, p1 + n1) in 4 KB requests (the tail rounded down to 16 B), grid-strided.
__global__ void ranges_kernel(const uint8_t* p0, long long n0, const uint8_t* p1, long long n1, int mode) {
  constexpr long long CH = 4096;
  const long long c0 = (n0 + CH - 1) / CH, c1 = (n1 + CH - 1) / CH;
  const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; c < c0 + c1; c += step) {
    const bool first = c < c0;
    const uint8_t* base = first ? p0 : p1;
    const long long off = (first ? c : c - c0) * CH, n = first ? n0 : n1;
    const uint32_t bytes = static_cast<uint32_t>((n - off < CH ? n - off : CH) & ~15LL);
    if (bytes) pf(base + off, bytes, mode);
  }
}

// M11's layout: w [32][rt_total][64 chunks][32 lanes] uint4, sf the same index in uint16. A task's warp q of ks
// reads chunks [q * kgw, (q + 1) * kgw), kgw = 64 / ks: 512 code bytes + 64 scale bytes per chunk, contiguous.
// For every local expert the routing hit ((id - local_start) mod 256 < 32), every row tile, every warp slice:
// the slice's first f16 / 16.
__global__ void experts_kernel(const int* ids, int n_ids, const int* local_start, const uint4* w,
                               const uint16_t* sf, int rt_total, int ks, int f16, int mode) {
  __shared__ unsigned s_mask;
  __shared__ int s_list[32];
  __shared__ int s_n;
  if (threadIdx.x == 0) s_mask = 0u;
  __syncthreads();
  const int ls = local_start[0];
  for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
    const int e = ((ids[i] - ls) % 256 + 256) % 256;
    if (e < 32) atomicOr(&s_mask, 1u << e);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n = 0;
    for (int e = 0; e < 32; ++e)
      if ((s_mask >> e) & 1u) s_list[n++] = e;
    s_n = n;
  }
  __syncthreads();
  const int kgw = 64 / ks;
  const long long items = static_cast<long long>(s_n) * rt_total * ks;
  const uint32_t wbytes = static_cast<uint32_t>(kgw * 32 * f16), sbytes = static_cast<uint32_t>(kgw * 4 * f16);
  const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long it = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; it < items; it += step) {
    const int q = static_cast<int>(it % ks);
    const long long tr = it / ks;
    const int r = static_cast<int>(tr % rt_total);
    const int e = s_list[tr / rt_total];
    const size_t base = (static_cast<size_t>(e) * rt_total + r) * 64 * 32 + static_cast<size_t>(q) * kgw * 32;
    pf(w + base, wbytes, mode);
    pf(sf + base, sbytes, mode);
  }
}
}  // namespace l9

void l9_ranges(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb, int64_t mode, int64_t blocks) {
  TORCH_CHECK(na >= 0 && na <= a.numel() * a.element_size() && nb >= 0 && nb <= b.numel() * b.element_size());
  l9::ranges_kernel<<<static_cast<int>(blocks), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(a.data_ptr()), na, reinterpret_cast<const uint8_t*>(b.data_ptr()), nb,
      static_cast<int>(mode));
}

void l9_experts(torch::Tensor ids, torch::Tensor local_start, torch::Tensor w, torch::Tensor sf, int64_t rt_total,
                int64_t ks, int64_t f16, int64_t mode, int64_t blocks) {
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && local_start.scalar_type() == at::kInt);
  TORCH_CHECK((ks == 1 || ks == 2 || ks == 4) && f16 >= 1 && f16 <= 16);
  TORCH_CHECK(w.numel() * w.element_size() == 32 * rt_total * 64 * 32 * 16 && sf.numel() * sf.element_size() == 32 * rt_total * 64 * 32 * 2);
  l9::experts_kernel<<<static_cast<int>(blocks), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), static_cast<int>(ids.numel()), local_start.data_ptr<int>(),
      reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()),
      static_cast<int>(rt_total), static_cast<int>(ks), static_cast<int>(f16), static_cast<int>(mode));
}
"""

_L9_CPP = """
void l9_ranges(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb, int64_t mode, int64_t blocks);
void l9_experts(torch::Tensor ids, torch::Tensor local_start, torch::Tensor w, torch::Tensor sf, int64_t rt_total,
                int64_t ks, int64_t f16, int64_t mode, int64_t blocks);
"""

# (tokens, sliding) -> {"o": 0/1, "e13": 0..16, "e2": 0..16, "slice": 0/1, "pm": 0/1}; filled by a lab. Missing: off.
L9_MODE = {}


def l9_source():
    return _L9_CUDA


def l9_mode(cfg, state):
    return state.get("l9_mode", {}).get((cfg["tokens"], bool(cfg["window"])), {}) if "l9" in state else {}


def l9_fork_attention(cfg, state):
    """After the qkv projection: stage o on the side stream (joined with the router's side work)."""
    m = l9_mode(cfg, state)
    if not m.get("o"):
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["o"]
        state["l9"].l9_ranges(w, w.numel() * w.element_size(), w, 0, m.get("pm", 0), state["l9_blocks"])


def l9_stage_experts(topk_ids, local_start, cfg, state):
    """On the router's side stream right after routing (the pooler streams meanwhile): stage a fraction of
    the routed local experts' M11 weights, with the slice geometry of this call's M11 config."""
    m = l9_mode(cfg, state)
    if not (m.get("e13") or m.get("e2")) or "m11" not in state:
        return
    mcfg = m11_config_for(state, cfg["tokens"])
    if mcfg is None:  # Marlin serves this call (another layout): nothing to stage
        return
    bank, ids = state["m11"], topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    for key, w, sf, rt, ks in (("e13", bank["w13"], bank["w13_s"], bank["rt_w13"], mcfg["s1"]["ks"]),
                               ("e2", bank["w2"], bank["w2_s"], bank["rt_w2"], mcfg["s2"]["ks"])):
        if m.get(key):
            state["l9"].l9_experts(ids.contiguous(), local_start, w, sf, rt, ks, m[key], m.get("pm", 0),
                                   state["l9_blocks"])


def l9_fork_slice(cfg, state):
    """Before the expert stage at <= 8 tokens: stage the unpooler slice on the side stream."""
    m = l9_mode(cfg, state)
    if not m.get("slice") or cfg["tokens"] > SLICED_UNPOOL_MAX_TOKENS:
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["unpooler"]
        state["l9"].l9_ranges(w, UNPOOL_SLICE * w.shape[1] * w.element_size(), w, 0, m.get("pm", 0), state["l9_blocks"])


def l9_join_slice(cfg, state):
    m = l9_mode(cfg, state)
    if m.get("slice") and cfg["tokens"] <= SLICED_UNPOOL_MAX_TOKENS:
        torch.cuda.current_stream().wait_stream(state["side_stream"])


def l9_setup(state, weights):
    """Build the staging kernels and launch each once (no first launch inside a capture); any failure
    leaves L9 off."""
    try:
        state["l9"] = load_inline(
            name="l9_ext", cpp_sources=_L9_CPP, cuda_sources=_L9_CUDA, functions=["l9_ranges", "l9_experts"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=tempfile.mkdtemp(prefix="l9_"), verbose=False)
        state["l9_blocks"] = 2 * torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
        state["l9_mode"] = {key: dict(val) for key, val in L9_MODE.items()}
        w = state["dense"]["o"]
        for pm in (0, 1):
            state["l9"].l9_ranges(w, 1 << 20, w, 0, pm, state["l9_blocks"])
            if "m11" in state:
                ids = torch.zeros((8, TOP_K), dtype=torch.int32, device=w.device)
                bank = state["m11"]
                state["l9"].l9_experts(ids, torch.zeros(1, dtype=torch.int32, device=w.device), bank["w13"],
                                       bank["w13_s"], bank["rt_w13"], 4, 1, pm, state["l9_blocks"])
        torch.cuda.synchronize()
        print(f"[l9] staging kernels: on ({len(state['l9_mode'])} cases)")
    except Exception as exc:
        state.pop("l9", None)
        print(f"[l9] staging kernels: OFF ({type(exc).__name__}: {str(exc)[-200:]})")
