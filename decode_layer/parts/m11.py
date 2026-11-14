# Snapshot of the MoE session's M11 (moe/m11/m11.py, sha256 a496ff5e9fe5a59e..., kernels identical to what its
# E3 check d0ce41e8697245ce ran on the judge: 1.309x on muse_moe_w4a16), without its prepare and
# kernel_fn. Copied, not imported, so later edits there cannot change this file.

"""M11: CUDA expert GEMMs for small batches; v1's Marlin path keeps the larger ones.

Swap-AB mma.sync.m16n8k16: 16 weight rows are the A operand, up to 8 * NT of an expert's pairs
are N. prepare() reorders each expert's NVFP4 codes (lossless) so that one 32-bit word is exactly
one thread's 8 A elements for one MMA, decoded straight into A registers with Marlin's bit trick,
and permutes K inside every 64-wide chunk so that a thread's 4 MMAs read 16 consecutive
activations, which are one NVFP4 scale group: 2 scales per thread per chunk. Every CTA scans the
pairs itself (no sort/align kernel) and its warps work through (active expert, row tile) tasks.
GEMM1 computes gate and up tiles together and applies SwiGLU in its epilogue."""

import tempfile

import triton.language as tl
from torch.utils.cpp_extension import load_inline

M11_MAX_TOKENS = 32
# Block scales are raised by 2^64 before the bf16 multiply (as in M2, exact); epilogues undo 2^(126 - 64).
M11_OUT_SHIFT = 2.0 ** 62
# The (gated, nt, depth) instances compiled; nt = MMA column tiles of 8 pairs, depth = K chunks in flight.
M11_INSTANCES = ((1, 8), (1, 4), (2, 4), (4, 2))
# Per-call configs by token count: (fewer than this many tokens, cfg) with cfg keys nt (both GEMMs),
# d1/d2 (depth) and c1/c2 (CTAs per SM) per GEMM; None means v1. A token count must be at most 8 * nt.
# Chosen by E3 (job d0ce41e8697245ce) from cold-L2 graph timings on the judge's H200; 17-32 tokens
# stay on v1, where nt 4 lost to Marlin.
M11_CONFIGS = (
    (5, dict(nt=1, d1=8, c1=2, d2=8, c2=2)),
    (9, dict(nt=1, d1=4, c1=4, d2=8, c2=4)),
    (17, dict(nt=2, d1=4, c1=4, d2=4, c2=4)),
    (33, None),
)

_M11_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace m11 {

constexpr int K = 4096;
constexpr int KG = K / 64;  // 64-wide K chunks
constexpr int LOCAL = 32;   // local experts; the active-expert ballot below relies on 32
constexpr int TOP_K = 8;
constexpr int MAXP = 32;    // pairs per expert at most: M11 runs only up to 32 tokens

// E2M1 nibbles (3 - R) and (7 - R) of w as bf16x2 lanes holding value * 2^-126 (Marlin's trick).
template <int R>
__device__ __forceinline__ uint32_t dec(uint32_t w) {
  const uint32_t x = w << (4 * R);
  return (x & 0x80008000u) | ((x >> 6) & 0x01C001C0u);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// E4M3 scales of rows g (low byte) and g + 8 (next byte) -> bf16x2 (s * 2^64, s * 2^64) per row; exact.
__device__ __forceinline__ void scales(uint32_t v, uint32_t& sg, uint32_t& sg8) {
  uint32_t h2;
  const unsigned short v16 = static_cast<unsigned short>(v & 0xFFFFu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v16));
  const unsigned short h_lo = static_cast<unsigned short>(h2 & 0xFFFFu);
  const unsigned short h_hi = static_cast<unsigned short>(h2 >> 16);
  float lo, hi;
  asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"(h_lo));
  asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"(h_hi));
  uint32_t b2;  // cvt.rn.bf16x2.f32 d, a, b puts a in the upper half
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b2) : "f"(hi * 18446744073709551616.0f), "f"(lo * 18446744073709551616.0f));
  sg = __byte_perm(b2, 0u, 0x1010u);
  sg8 = __byte_perm(b2, 0u, 0x3232u);
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
template <int TT, int NT, int D, bool GATED>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  const int tid = threadIdx.x;
  if (tid < LOCAL) s_cnt[tid] = 0;
  __syncthreads();
  for (int p = tid; p < n_pairs; p += blockDim.x) {
    const int e = ids[p] - local_start;
    if (e >= 0 && e < LOCAL) {
      const int r = atomicAdd(&s_cnt[e], 1);
      if (r < MAXP) s_list[e][r] = static_cast<short>(p);
    }
  }
  __syncthreads();
  if (tid < 32) {
    const bool active = s_cnt[tid] > 0;
    const unsigned m = __ballot_sync(0xffffffffu, active);
    if (active) s_active[__popc(m & ((1u << tid) - 1u))] = tid;
    if (tid == 0) s_nactive = __popc(m);
  }
  __syncthreads();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int warps = gridDim.x * (blockDim.x >> 5);
  for (int task = blockIdx.x * (blockDim.x >> 5) + (tid >> 5); task < n_tasks; task += warps) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP), 8 * NT);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const uint4* wp[TT];
    const uint16_t* sp[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t base = (static_cast<size_t>(e) * rt_total + rts[i]) * KG * 32 + lane;
      wp[i] = w + base;
      sp[i] = sf + base;
    }
    // B operand: this thread loads pair column g of each 8-pair tile, 16 activations per K chunk.
    const uint4* ap[NT];
    bool aok[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      aok[n] = idx < cnt;
      const int p = aok[n] ? s_list[e][idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
    }
    float acc[TT][NT][4];
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

    uint4 rw[D][TT];
    uint32_t rs[D][TT];
    uint4 ra[D][NT][2];
    auto load = [&](int s, int kg) {
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        rw[s][i] = __ldcs(wp[i] + kg * 32);
        rs[s][i] = __ldcs(sp[i] + kg * 32);
      }
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        if (aok[n]) {
          ra[s][n][0] = __ldg(ap[n] + kg * 8);
          ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
        } else {
          ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
          ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    };
    auto compute = [&](int s) {
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        uint32_t sg, sg8;
        scales(rs[s][i], sg, sg8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = word(rw[s][i], j);
          const uint32_t a0 = hmul2(dec<0>(q), sg), a1 = hmul2(dec<1>(q), sg8);
          const uint32_t a2 = hmul2(dec<2>(q), sg), a3 = hmul2(dec<3>(q), sg8);
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            if (n * 8 < cnt) {
              const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
              const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
              mma(acc[i][n], a0, a1, a2, a3, b0, b1);
            }
          }
        }
      }
    };
#pragma unroll
    for (int s = 0; s < D; ++s) load(s, s);
    for (int kg0 = 0; kg0 < KG; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        compute(s);
        if (kg0 + s + D < KG) load(s, kg0 + s + D);
      }
    }

    // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1.
    const float c = cs[e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < cnt) {
          const size_t prow = static_cast<size_t>(s_list[e][idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * c;
              const float uv = acc[1][n][2 * rr + h] * c;
              out[prow + 16 * tile + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * rts[i] + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * c);
            }
          }
        }
      }
    }
  }
}

}  // namespace m11

#define M11_CASE(G, TT, NT, D)                                                                           \
  if (gated == (G) && nt == (NT) && depth == (D)) {                                                      \
    m11::gemm_kernel<TT, NT, D, G><<<static_cast<int>(grid), 128, 0, stream>>>(                          \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

#define M11_ATTR(G, TT, NT, D)                                                                            \
  if (gated == (G) && nt == (NT) && depth == (D)) {                                                       \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::gemm_kernel<TT, NT, D, G>);                                         \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
  }

torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t nt, int64_t depth,
                       int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  M11_CASE(true, 2, 1, 8) M11_CASE(true, 2, 1, 4) M11_CASE(true, 2, 2, 4) M11_CASE(true, 2, 4, 2)
  M11_CASE(false, 1, 1, 8) M11_CASE(false, 1, 1, 4) M11_CASE(false, 1, 2, 4) M11_CASE(false, 1, 4, 2)
  TORCH_CHECK(false, "m11_gemm: no instance for gated=", gated, " nt=", nt, " depth=", depth);
  return out;
}

torch::Tensor m11_attrs(int64_t gated, int64_t nt, int64_t depth) {
  M11_ATTR(true, 2, 1, 8) M11_ATTR(true, 2, 1, 4) M11_ATTR(true, 2, 2, 4) M11_ATTR(true, 2, 4, 2)
  M11_ATTR(false, 1, 1, 8) M11_ATTR(false, 1, 1, 4) M11_ATTR(false, 1, 2, 4) M11_ATTR(false, 1, 4, 2)
  TORCH_CHECK(false, "m11_attrs: no instance for gated=", gated, " nt=", nt, " depth=", depth);
  return torch::Tensor();
}
"""

_M11_CPP = """
torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t nt, int64_t depth,
                       int64_t grid);
torch::Tensor m11_attrs(int64_t gated, int64_t nt, int64_t depth);
"""


def m11_build():
    """Compile the CUDA kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=_M11_CUDA, functions=["m11_gemm", "m11_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m11_"), verbose=False)


def m11_shuffle(packed, sf):
    """One expert's NVFP4 matrix -> M11's layout (a lossless reordering).
    packed uint8 [rows, K/2] (element 2i in the low nibble), sf float8_e4m3fn [rows, K/16] ->
    words int32 [rows/16, K/64, 32 lanes, 4] and scales int16 [rows/16, K/64, 32 lanes].
    Lane (g, t), word j holds rows g and g + 8 of the tile at chunk positions 16 t + 4 j + {0..3};
    nibble p sits where the decode's round (3 - p % 4) puts it: A registers a0..a3 of the MMA."""
    rows, half = packed.shape
    rt, kg = rows // 16, half * 2 // 64
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    c = codes.view(rt, 2, 8, kg, 4, 4, 4)  # (rt, h, g, kg, t, j, q): row 16 rt + 8 h + g, k = 64 kg + 16 t + 4 j + q
    # Nibble p of each word comes from (h, q): row g + 8 h, chunk position 16 t + 4 j + q.
    src = ((1, 2), (0, 2), (1, 0), (0, 0), (1, 3), (0, 3), (1, 1), (0, 1))
    nib = [c[:, h, :, :, :, :, q] for h, q in src]  # each [rt, g, kg, t, j]
    words = torch.stack([nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(4)], dim=-1)
    words = words.permute(0, 2, 1, 3, 4, 5).contiguous().view(torch.int32).view(rt, kg, 32, 4)
    s = sf.view(torch.uint8).view(rt, 2, 8, kg, 4)  # (rt, h, g, kg, t): row 16 rt + 8 h + g, group 4 kg + t
    scales = torch.stack((s[:, 0], s[:, 1]), dim=-1).permute(0, 2, 1, 3, 4).contiguous()
    return words, scales.view(torch.int16).view(rt, kg, 32)


def m11_bank(weights, start, count):
    """M11's weights for experts [start, start + count), shuffled one expert at a time."""
    bank = {"start": start, "count": count, "inter": weights["w13"].shape[1] // 2, "hidden": weights["w2"].shape[1],
            "c13": (weights["w13_gs"][start:start + count].float() * M11_OUT_SHIFT).contiguous(),
            "c2": (weights["w2_gs"][start:start + count].float() * M11_OUT_SHIFT).contiguous()}
    for name in ("w13", "w2"):
        parts = [m11_shuffle(weights[name][e], weights[name + "_sf"][e]) for e in range(start, start + count)]
        bank[name] = torch.stack([p[0] for p in parts])
        bank[name + "_s"] = torch.stack([p[1] for p in parts])
        bank["rt_" + name] = bank[name].shape[1]
    return bank


@triton.jit(do_not_specialize=["local_start"])
def _m11_post_norm_kernel(y_ptr, nw_ptr, tw_ptr, ids_ptr, out_ptr, local_start, eps,
                          H: tl.constexpr, LOCAL: tl.constexpr, TOP_K: tl.constexpr):
    """out[t] = sum over local slots k of tw[t, k] * rmsnorm(y[t, k] * nw), as SGLang's fused kernel
    (weight first); y is bf16 [pairs, H]."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(nw_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = t * TOP_K + k
        e = tl.load(ids_ptr + p) - local_start
        if (e >= 0) & (e < LOCAL):
            y = tl.load(y_ptr + p.to(tl.int64) * H + offs).to(tl.float32) * w
            var = tl.sum(y * y, axis=0) / H
            acc += y * tl.rsqrt(var + eps) * tl.load(tw_ptr + p)
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


def m11_gemm1(x, ids, state, cfg):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair."""
    bank, n_pairs = state["m11"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    state["m11_ext"].m11_gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, n_pairs, bank["start"],
                              bank["rt_w13"], 0, bank["inter"], 1, cfg["nt"], cfg["d1"], cfg["c1"] * state["sms"])
    return act


def m11_gemm2(act, ids, state, cfg):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    state["m11_ext"].m11_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                              bank["rt_w2"], 1, bank["hidden"], 0, cfg["nt"], cfg["d2"], cfg["c2"] * state["sms"])
    return y


def m11_post_norm(y, ids, topk_weights, state, tokens):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
    _m11_post_norm_kernel[(tokens,)](y, state["norm_w"], topk_weights, ids, out, bank["start"], POST_NORM_EPS,
                                     H=bank["hidden"], LOCAL=bank["count"], TOP_K=TOP_K, num_warps=8)
    return out


def m11_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg)
    y = m11_gemm2(act, ids, state, cfg)
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0])


def m11_config_for(state, tokens):
    """The M11 config for this token count, or None for v1."""
    for below, cfg in state["m11_configs"]:
        if tokens < below:
            return cfg if cfg is not None and tokens <= 8 * cfg["nt"] else None
    return None


def m11_warm(state, cfg):
    """Run every kernel of cfg once before any CUDA graph capture (Triton compiles on first launch)."""
    device = state["norm_w"].device
    tokens = min(8 * cfg["nt"], M11_MAX_TOKENS)
    x = torch.zeros((tokens, state["m11"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m11"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m11_run(x, ids + state["m11"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m11_setup(state, weights):
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = m11_build()
    state["m11"] = m11_bank(weights, 0, 32)
