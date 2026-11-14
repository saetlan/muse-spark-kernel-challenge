"""M11: CUDA expert GEMMs for small batches; v1's Marlin path keeps the larger ones.

Swap-AB mma.sync.m16n8k16: 16 weight rows are the A operand, 8 * NT of an expert's pairs at a
time are N. prepare() reorders each expert's NVFP4 codes (lossless) so that one 32-bit word is exactly
one thread's 8 A elements for one MMA, decoded straight into A registers with Marlin's bit trick,
and permutes K inside every 64-wide chunk so that a thread's 4 MMAs read 16 consecutive
activations, which are one NVFP4 scale group: 2 scales per thread per chunk. Every CTA scans the
pairs itself (no sort/align kernel); groups of KS warps work through (active expert, TT row tiles)
tasks, splitting K and adding partial sums in shared memory. GEMM1 computes gate and up tiles
together and applies SwiGLU in its epilogue."""

import tempfile

import triton.language as tl
from torch.utils.cpp_extension import load_inline

# Tokens M11 accepts at most; it is also the kernel's MAXP (pairs per expert it can list), since a
# token routes to an expert at most once. E22b: 64 (was 320): M11 runs only below 33 tokens (m11_config_for maps
# 33+ to v1; every M11 tuner candidate is for a class <= 32), and the static s_list shrinks 20 KB -> 4 KB.
M11_MAX_TOKENS = 64
# Block scales are raised by 2^64 before the bf16 multiply (as in M2, exact); epilogues undo 2^(126 - 64).
M11_OUT_SHIFT = 2.0 ** 62
# Kernel instances compiled, (gated, tt, nt, depth, ks): tt = 16-row tiles per task (gated: gate + up),
# nt = 8-pair MMA column tiles per pass, depth = 64-wide K chunks in flight per warp, ks = warps
# splitting a task's K. Each warp's K share (64 / ks chunks) must be a multiple of depth.
M11_INSTANCES = (
    (1, 2, 1, 8, 1), (1, 2, 1, 4, 1), (1, 2, 1, 4, 2), (1, 2, 1, 4, 4), (1, 2, 1, 2, 4),
    (1, 2, 2, 4, 1), (1, 2, 2, 4, 2), (1, 2, 2, 2, 4), (1, 2, 4, 2, 1), (1, 2, 4, 2, 2),
    (0, 1, 1, 8, 1), (0, 1, 1, 4, 2), (0, 1, 1, 4, 4), (0, 1, 2, 4, 1), (0, 1, 2, 4, 4), (0, 1, 4, 2, 1),
    (0, 2, 1, 4, 1), (0, 2, 1, 4, 2), (0, 2, 1, 4, 4), (0, 2, 1, 2, 4), (0, 2, 2, 4, 2), (0, 2, 2, 2, 4),
    (1, 2, 4, 2, 4), (0, 2, 4, 2, 1), (0, 2, 4, 2, 2),
)
# M12 instances (TMA weight ring), (gated, tt, nt, ks, ch, r): ch = 64-wide K chunks per ring slot,
# r = slots per warp; each warp's K share (64 / ks chunks) must fill an even number of slots.
# Dynamic shared memory per CTA = 4 * r * tt * ch * 576 bytes.
M12_INSTANCES = (
    (1, 2, 1, 1, 4, 4), (1, 2, 1, 1, 4, 2), (1, 2, 1, 1, 8, 2), (1, 2, 1, 2, 4, 4), (1, 2, 1, 4, 4, 4),
    (1, 2, 1, 4, 2, 4), (1, 2, 2, 1, 4, 2), (1, 2, 2, 4, 4, 2), (1, 2, 4, 1, 2, 2),
    (0, 1, 1, 1, 4, 4), (0, 2, 1, 1, 4, 4), (0, 2, 1, 1, 4, 2), (0, 2, 1, 2, 4, 4), (0, 2, 1, 4, 4, 4),
    (0, 1, 1, 4, 4, 4), (0, 2, 2, 1, 4, 2), (0, 2, 2, 4, 4, 2), (0, 2, 4, 1, 2, 2),
)
# Per-call configs by token count: (fewer than this many tokens, cfg) with cfg["s1"] = GEMM1's
# dict(nt, d, ks, c) and cfg["s2"] = GEMM2's dict(tt, nt, d, ks, c); c = CTAs per SM; None means v1.
# A stage dict with kind="m12" runs the TMA kernel instead: s1 = dict(kind, nt, ks, ch, r, c),
# s2 = dict(kind, tt, nt, ks, ch, r, c).
# These are E3c's choices (job 8ce4aba7fd53499c, means over 3 routings); 33+ tokens stay on v1, where
# M11 only tied Marlin.
M11_CONFIGS = (
    (5, dict(s1=dict(nt=1, d=4, ks=4, c=8), s2=dict(tt=1, nt=1, d=4, ks=4, c=4))),
    (9, dict(s1=dict(nt=1, d=4, ks=2, c=8), s2=dict(tt=2, nt=1, d=4, ks=4, c=8))),
    (17, dict(s1=dict(nt=1, d=4, ks=2, c=8), s2=dict(tt=2, nt=1, d=4, ks=4, c=8))),
    (33, dict(s1=dict(nt=2, d=2, ks=4, c=8), s2=dict(tt=2, nt=2, d=2, ks=4, c=4))),
    (192, None),
    (321, None),
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
constexpr int MAXP = @@MAXP@@;  // pairs per expert at most (= M11_MAX_TOKENS in Python)

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

// A group of KS warps of one CTA's barrier id (1-4); bar.sync orders their shared-memory accesses.
__device__ __forceinline__ void group_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// PDL (E15, as M13's): the next grid may launch once every CTA of this one has signalled; a PDL-launched grid
// waits before reading the previous grid's output, through loads that depend on the wait's (zero) token.
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t pdl_wait_token() {
  uint32_t z;
  asm volatile("griddepcontrol.wait;\n\tmov.u32 %0, 0;" : "=r"(z) : : "memory");
  return z;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
// A task (expert, TT row tiles) goes to a group of KS warps that split K and add their partial sums
// in shared memory; an expert's pairs are taken 8 * NT at a time (a second chunk re-reads its weights).
// PDL (GEMM2 only): launched early with programmatic stream serialization; scans the routing, then every CTA waits
// and reads the activations with coherent token-ordered loads (__ldg's non-coherent path isn't safe behind the wait).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once
  constexpr int KGW = KG / KS;    // K chunks per warp
  static_assert(KGW % D == 0, "depth must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched
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
  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads
  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int total = min(s_cnt[e], MAXP);
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
    const float c = cs[e];
    for (int c0 = 0; c0 < total; c0 += 8 * NT) {
      const int cnt = min(total - c0, 8 * NT);
      // B operand: this thread loads pair column g of each 8-pair tile, 16 activations per K chunk.
      const uint4* ap[NT];
      bool aok[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int idx = n * 8 + g;
        aok[n] = idx < cnt;
        const int p = aok[n] ? s_list[e][c0 + idx] : 0;
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
            if constexpr (PDL) {
              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);
              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);
            } else {
              ra[s][n][0] = __ldg(ap[n] + kg * 8);
              ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
            }
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
      const int kbeg = kpart * KGW;
#pragma unroll
      for (int s = 0; s < D; ++s) load(s, kbeg + s);
      for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
        for (int s = 0; s < D; ++s) {
          compute(s);
          if (kg0 + s + D < KGW) load(s, kbeg + kg0 + s + D);
        }
      }

      if constexpr (KS > 1) {  // warp kpart 0 of the group adds the others' partial sums
        if (kpart > 0) {
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
        }
        group_sync(grp + 1, KS * 32);
        if (kpart == 0) {
#pragma unroll
          for (int w2 = 1; w2 < KS; ++w2)
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[i][n][r] += s_red[warp + w2][(i * NT + n) * 4 + r][lane];
        }
        group_sync(grp + 1, KS * 32);  // the next chunk may overwrite s_red
      }

      // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1.
      if (kpart == 0) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < cnt) {
              const size_t prow = static_cast<size_t>(s_list[e][c0 + idx]) * out_cols;
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
  }
}

// E21: gemm_kernel with the register ring kept streaming across (task, pair pass) units. During a unit's last D
// chunks each warp loads the next unit's first D chunks into the same ring slots (KGW % D == 0, so slot = chunk % D
// lines up), so only a group's first unit pays the memory latency; the split-K reduction and the epilogue overlap
// those loads. Same scan, decode, MMA, reduction and epilogue as gemm_kernel.
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) cr_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  static_assert(KGW % D == 0 && KGW >= D, "depth must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();
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
  uint32_t tok = 0u;
  if constexpr (PDL) tok = pdl_wait_token();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;

  // A unit is (task, first pair of the pass); units of a group: its tasks in stride order, each task's passes.
  struct Ops {
    const uint4* wp[TT];
    const uint16_t* sp[TT];
    const uint4* ap[NT];
    bool aok[NT];
  };
  auto ops_for = [&](int task, int c0, Ops& o) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
      const size_t base = (static_cast<size_t>(e) * rt_total + rt) * KG * 32 + lane;
      o.wp[i] = w + base;
      o.sp[i] = sf + base;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      o.aok[n] = idx < cnt;
      const int p = o.aok[n] ? s_list[e][c0 + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      o.ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
    }
  };
  auto next_unit = [&](int& task, int& c0) {
    const int e = s_active[task / per_expert];
    c0 += 8 * NT;
    if (c0 >= min(s_cnt[e], MAXP)) {
      c0 = 0;
      task += stride;
    }
  };

  uint4 rw[D][TT];
  uint32_t rs[D][TT];
  uint4 ra[D][NT][2];
  auto load = [&](const Ops& o, int s, int kg) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      rw[s][i] = __ldcs(o.wp[i] + kg * 32);
      rs[s][i] = __ldcs(o.sp[i] + kg * 32);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (o.aok[n]) {
        if constexpr (PDL) {
          ra[s][n][0] = ld_cg4(o.ap[n] + kg * 8 + tok);
          ra[s][n][1] = ld_cg4(o.ap[n] + kg * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(o.ap[n] + kg * 8);
          ra[s][n][1] = __ldg(o.ap[n] + kg * 8 + 1);
        }
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  int task = blockIdx.x * GROUPS + grp, c0 = 0;
  if (task >= n_tasks) return;  // the whole group (all KS warps) has no unit: no group_sync is reached
  Ops oc, on;
  ops_for(task, c0, oc);
#pragma unroll
  for (int s = 0; s < D; ++s) load(oc, s, kbeg + s);
  while (task < n_tasks) {
    int ntask = task, nc0 = c0;
    next_unit(ntask, nc0);
    const bool more = ntask < n_tasks;
    if (more) ops_for(ntask, nc0, on);
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    float acc[TT][NT][4];
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
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
    for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        compute(s);
        const int j = kg0 + s + D;
        if (j < KGW) {
          load(oc, s, kbeg + j);
        } else if (more) {
          load(on, s, kbeg + j - KGW);  // the next unit's chunk j - KGW (< D) -> slot s = (j - KGW) % D
        }
      }
    }

    if constexpr (KS > 1) {
      if (kpart > 0) {
#pragma unroll
        for (int i = 0; i < TT; ++i)
#pragma unroll
          for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
      }
      group_sync(grp + 1, KS * 32);
      if (kpart == 0) {
#pragma unroll
        for (int w2 = 1; w2 < KS; ++w2)
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) acc[i][n][r] += s_red[warp + w2][(i * NT + n) * 4 + r][lane];
      }
      group_sync(grp + 1, KS * 32);
    }

    if (kpart == 0) {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < cnt) {
            const size_t prow = static_cast<size_t>(s_list[e][c0 + idx]) * out_cols;
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
    task = ntask;
    c0 = nc0;
    if (more) oc = on;
  }
}

// E22 helpers (M11's own smem_addr comes later, with M12)
__device__ __forceinline__ uint32_t sr_smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void sr_cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void sr_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void sr_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// E22: cr_kernel with the weight codes and scales in a per-warp shared-memory ring DW chunks deep (cp.async), so
// the bytes in flight no longer shrink while a chunk is decoded (MF24: loads alone reach ~4.3 TB/s at 8 tokens, the
// full kernel ~2.5) and ~40 registers leave the ring. Codes: each lane copies its own 16 B; scales: lanes 0-3 copy
// the warp's 64-B block (cp.async needs >= 4 B) and a __syncwarp publishes it. One commit group per chunk, so
// wait_group<DW - 1> covers the chunk about to be computed. Activations keep cr_kernel's register ring (L1/L2 hits).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false, int DW = 8>
__global__ void __launch_bounds__(128) sr_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  static_assert(KGW % D == 0 && KGW >= D, "depth must divide each warp's K chunks");
  static_assert(KGW % DW == 0 && KGW >= DW && DW >= 2 && DW <= 8, "the weight ring must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  extern __shared__ __align__(16) unsigned char s_ring[];  // [4 warps][DW][TT] codes x 512 B, then scales x 64 B
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();
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
  uint32_t tok = 0u;
  if constexpr (PDL) tok = pdl_wait_token();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const uint32_t ring0 = sr_smem_addr(s_ring);
  auto coff = [&](int slot, int i) { return ((warp * DW + slot) * TT + i) * 512 + lane * 16; };
  auto soff = [&](int slot, int i) { return 4 * DW * TT * 512 + ((warp * DW + slot) * TT + i) * 64; };
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;

  // A unit is (task, first pair of the pass); units of a group: its tasks in stride order, each task's passes.
  struct Ops {
    const uint4* wp[TT];
    const uint16_t* sp[TT];
    const uint4* ap[NT];
    bool aok[NT];
  };
  auto ops_for = [&](int task, int c0, Ops& o) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
      const size_t base = (static_cast<size_t>(e) * rt_total + rt) * KG * 32 + lane;
      o.wp[i] = w + base;
      o.sp[i] = sf + base;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      o.aok[n] = idx < cnt;
      const int p = o.aok[n] ? s_list[e][c0 + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      o.ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
    }
  };
  auto next_unit = [&](int& task, int& c0) {
    const int e = s_active[task / per_expert];
    c0 += 8 * NT;
    if (c0 >= min(s_cnt[e], MAXP)) {
      c0 = 0;
      task += stride;
    }
  };

  uint4 ra[D][NT][2];
  auto load_w = [&](const Ops& o, int slot, int kg) {  // codes (own 16 B) and the warp's 64-B scale block
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      sr_cp16(ring0 + coff(slot, i), o.wp[i] + kg * 32);
      if (lane < 4) sr_cp16(ring0 + soff(slot, i) + lane * 16, reinterpret_cast<const unsigned char*>(o.sp[i] - lane + kg * 32) + lane * 16);
    }
  };
  auto load = [&](const Ops& o, int s, int kg) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (o.aok[n]) {
        if constexpr (PDL) {
          ra[s][n][0] = ld_cg4(o.ap[n] + kg * 8 + tok);
          ra[s][n][1] = ld_cg4(o.ap[n] + kg * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(o.ap[n] + kg * 8);
          ra[s][n][1] = __ldg(o.ap[n] + kg * 8 + 1);
        }
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  int task = blockIdx.x * GROUPS + grp, c0 = 0;
  if (task >= n_tasks) return;  // the whole group (all KS warps) has no unit: no group_sync is reached
  Ops oc, on;
  ops_for(task, c0, oc);
#pragma unroll
  for (int s = 0; s < D; ++s) load(oc, s, kbeg + s);
#pragma unroll
  for (int s = 0; s < DW; ++s) {  // the first unit's first DW chunks, one commit group each
    load_w(oc, s, kbeg + s);
    sr_commit();
  }
  int wc = 0;  // chunks consumed: chunk wc's codes and scales are in ring slot wc % DW
  while (task < n_tasks) {
    int ntask = task, nc0 = c0;
    next_unit(ntask, nc0);
    const bool more = ntask < n_tasks;
    if (more) ops_for(ntask, nc0, on);
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    float acc[TT][NT][4];
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
    auto compute = [&](int s, int wsl) {
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const uint4 rwv = *reinterpret_cast<const uint4*>(s_ring + coff(wsl, i));
        const uint32_t rsv = *reinterpret_cast<const uint16_t*>(s_ring + soff(wsl, i) + lane * 2);
        uint32_t sg, sg8;
        scales(rsv, sg, sg8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = word(rwv, j);
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
    for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        sr_wait<DW - 1>();  // chunk wc's group has landed (one group per chunk)
        __syncwarp();       // ... and lanes 0-3's scale copies are visible to the whole warp
        compute(s, wc % DW);
        __syncwarp();       // every lane has read slot wc % DW before it is refilled
        const int jw = kg0 + s + DW;  // weights DW chunks ahead (into the next unit past KGW)
        if (jw < KGW) {
          load_w(oc, wc % DW, kbeg + jw);
        } else if (more) {
          load_w(on, wc % DW, kbeg + jw - KGW);
        }
        sr_commit();
        ++wc;
        const int j = kg0 + s + D;
        if (j < KGW) {
          load(oc, s, kbeg + j);
        } else if (more) {
          load(on, s, kbeg + j - KGW);  // the next unit's chunk j - KGW (< D) -> slot s = (j - KGW) % D
        }
      }
    }

    if constexpr (KS > 1) {
      if (kpart > 0) {
#pragma unroll
        for (int i = 0; i < TT; ++i)
#pragma unroll
          for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
      }
      group_sync(grp + 1, KS * 32);
      if (kpart == 0) {
#pragma unroll
        for (int w2 = 1; w2 < KS; ++w2)
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) acc[i][n][r] += s_red[warp + w2][(i * NT + n) * 4 + r][lane];
      }
      group_sync(grp + 1, KS * 32);
    }

    if (kpart == 0) {
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < cnt) {
            const size_t prow = static_cast<size_t>(s_list[e][c0 + idx]) * out_cols;
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
    task = ntask;
    c0 = nc0;
    if (more) oc = on;
  }
}


// ------------------------------------------------------------------------------------------------
// M12: M11's work split and arithmetic, but each warp's weight and scale stream is copied by TMA
// (cp.async.bulk) into its own ring of R shared-memory slots of CH K chunks, which lane 0 refills
// as soon as the warp has read a slot; activations keep a two-step register prefetch. PTX per the
// ISA 9.4: mbarrier.init/arrive.expect_tx/try_wait.parity, fence.mbarrier_init, fence.proxy.async.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

// Makes the inits visible to the async proxy (the TMA unit) before any copy signals them.
__device__ __forceinline__ void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
               ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  return done != 0;
}

// A wait that can never finish (bytes expected vs copied, or a slot never refilled) traps within
// seconds instead of hanging the run until its 15-minute limit.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  for (uint32_t spins = 0; !mbar_try_wait(bar, parity); ++spins)
    if (spins > (1u << 24)) __trap();
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Orders this warp's prior generic-proxy reads of a slot before the TMA write that refills it.
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

template <int TT, int NT, int KS, int CH, int R, bool GATED>
__global__ void __launch_bounds__(128) tma_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  constexpr int STEPS = KGW / CH;
  constexpr int TILE_BYTES = CH * 576;  // CH chunks of 512 B of codes, then CH x 64 B of scales
  constexpr int SLOT_BYTES = TT * TILE_BYTES;
  static_assert(KGW % CH == 0 && STEPS % 2 == 0, "each warp's K chunks must fill an even number of slots");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  extern __shared__ __align__(128) unsigned char ring[];  // [4 warps][R slots][SLOT_BYTES]
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid < LOCAL) s_cnt[tid] = 0;
  if ((tid & 31) == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[tid >> 5][r]), 1);
    mbar_init_fence();
  }
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
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int first = blockIdx.x * GROUPS + grp;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;
  const unsigned char* myring = ring + warp * (R * SLOT_BYTES);
  const uint32_t ring_s = smem_addr(myring);
  const uint32_t bar_s = smem_addr(&bars[warp][0]);  // slot r's barrier at bar_s + 8 r

  // The refill cursor walks the same (task, pair chunk, step) sequence as the loop below, R slots ahead.
  int p_task = first, p_c0 = 0, p_step = 0;
  auto issue = [&](int slot) {
    if (p_task >= n_tasks) return;
    const int e = s_active[p_task / per_expert];
    const int tile = p_task % per_expert;
    if (lane == 0) {
      const uint32_t bar = bar_s + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT_BYTES);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
        const size_t chunk = (static_cast<size_t>(e) * rt_total + rt) * KG + kbeg + p_step * CH;
        const uint32_t dst = ring_s + slot * SLOT_BYTES + i * TILE_BYTES;
        bulk_g2s(dst, w + chunk * 32, CH * 512, bar);
        bulk_g2s(dst + CH * 512, sf + chunk * 32, CH * 64, bar);
      }
    }
    if (++p_step == STEPS) {
      p_step = 0;
      p_c0 += 8 * NT;
      if (p_c0 >= min(s_cnt[e], MAXP)) {
        p_c0 = 0;
        p_task += stride;
      }
    }
  };
#pragma unroll
  for (int r = 0; r < R; ++r) issue(r);

  int seq = 0;  // slots consumed by this warp so far
  for (int task = first; task < n_tasks; task += stride) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int total = min(s_cnt[e], MAXP);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    for (int c0 = 0; c0 < total; c0 += 8 * NT) {
      const int cnt = min(total - c0, 8 * NT);
      const uint4* ap[NT];
      bool aok[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int idx = n * 8 + g;
        aok[n] = idx < cnt;
        const int p = aok[n] ? s_list[e][c0 + idx] : 0;
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

      uint4 ra[2][CH][NT][2];
      auto load_act = [&](int b, int st) {
#pragma unroll
        for (int jj = 0; jj < CH; ++jj) {
          const int kg = kbeg + st * CH + jj;
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            if (aok[n]) {
              ra[b][jj][n][0] = __ldg(ap[n] + kg * 8);
              ra[b][jj][n][1] = __ldg(ap[n] + kg * 8 + 1);
            } else {
              ra[b][jj][n][0] = make_uint4(0u, 0u, 0u, 0u);
              ra[b][jj][n][1] = make_uint4(0u, 0u, 0u, 0u);
            }
          }
        }
      };
      auto step = [&](int b) {
        const int slot = seq % R;
        mbar_wait(bar_s + 8 * slot, (seq / R) & 1);
        const unsigned char* sp = myring + slot * SLOT_BYTES;
#pragma unroll
        for (int jj = 0; jj < CH; ++jj) {
#pragma unroll
          for (int i = 0; i < TT; ++i) {
            const unsigned char* tp = sp + i * TILE_BYTES;
            const uint4 q4 = *reinterpret_cast<const uint4*>(tp + jj * 512 + lane * 16);
            const uint32_t sv = *reinterpret_cast<const uint16_t*>(tp + CH * 512 + jj * 64 + lane * 2);
            uint32_t sg, sg8;
            scales(sv, sg, sg8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t q = word(q4, j);
              const uint32_t a0 = hmul2(dec<0>(q), sg), a1 = hmul2(dec<1>(q), sg8);
              const uint32_t a2 = hmul2(dec<2>(q), sg), a3 = hmul2(dec<3>(q), sg8);
#pragma unroll
              for (int n = 0; n < NT; ++n) {
                if (n * 8 < cnt) {
                  const uint32_t b0 = word(ra[b][jj][n][j >> 1], (2 * j) & 3);
                  const uint32_t b1 = word(ra[b][jj][n][j >> 1], (2 * j + 1) & 3);
                  mma(acc[i][n], a0, a1, a2, a3, b0, b1);
                }
              }
            }
          }
        }
        __syncwarp();
        if (lane == 0) fence_proxy_async();
        issue(slot);
        ++seq;
      };
      load_act(0, 0);
      for (int st = 0; st < STEPS; st += 2) {
        load_act(1, st + 1);
        step(0);
        if (st + 2 < STEPS) load_act(0, st + 2);
        step(1);
      }

      if constexpr (KS > 1) {  // warp kpart 0 of the group adds the others' partial sums
        if (kpart > 0) {
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
        }
        group_sync(grp + 1, KS * 32);
        if (kpart == 0) {
#pragma unroll
          for (int w2 = 1; w2 < KS; ++w2)
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[i][n][r] += s_red[warp + w2][(i * NT + n) * 4 + r][lane];
        }
        group_sync(grp + 1, KS * 32);
      }

      if (kpart == 0) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < cnt) {
              const size_t prow = static_cast<size_t>(s_list[e][c0 + idx]) * out_cols;
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
  }
}

}  // namespace m11

#define M11_CASE(G, TT, NT, D, KS)                                                                       \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                          \
    m11::gemm_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                      \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

// E21: the continuous-ring kernel, selected by depth | 16.
#define M11_CR_CASE(G, TT, NT, D, KS)                                                                    \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | 16) && ks == (KS)) {                   \
    m11::cr_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                        \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

// E22: the shared-memory-ring kernel (DW = 8), selected by depth | 32; its ring is dynamic shared memory
// (<= 36 KB, under the 48 KB default, so no attribute call, none during a capture).
#define M11_SR_CASE(G, TT, NT, D, KS)                                                                    \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | 32) && ks == (KS)) {                   \
    m11::sr_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 4 * 8 * (TT) * 576, stream>>>(       \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    const cudaError_t err = cudaGetLastError();                                                          \
    TORCH_CHECK(err == cudaSuccess, "m11 sr launch failed: ", cudaGetErrorString(err));                   \
    return out;                                                                                          \
  }

#define M11_ATTR(G, TT, NT, D, KS)                                                                        \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                           \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::gemm_kernel<TT, NT, D, KS, G>);                                     \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
  }

torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                       int64_t ks, int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  @@CASES@@
  TORCH_CHECK(false, "m11_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return out;
}

#define M11_KATTR(KERN, G, TT, NT, D, KS, BIT)                                                           \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | (BIT)) && ks == (KS)) {                 \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::KERN<TT, NT, D, KS, G>);                                            \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
  }

torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks) {
  @@ATTRS@@
  TORCH_CHECK(false, "m11_attrs: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return torch::Tensor();
}

#define M12_CASE(G, TT, NT, KS, CH, R)                                                                   \
  if (gated == (G) && tt == (TT) && nt == (NT) && ks == (KS) && ch == (CH) && r == (R)) {                \
    m11::tma_kernel<TT, NT, KS, CH, R, G><<<static_cast<int>(grid), 128, 4 * R * TT * CH * 576, stream>>>( \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

// Raises each instance's dynamic shared-memory limit (needed above 48 KB) and reports its resources:
// rows of [gated, tt, nt, ks, ch, r, registers, local (spill) bytes, static smem, dynamic smem].
#define M12_SETUP(G, TT, NT, KS, CH, R)                                                                  \
  {                                                                                                      \
    const int dyn = 4 * R * TT * CH * 576;                                                               \
    cudaFuncSetAttribute(m11::tma_kernel<TT, NT, KS, CH, R, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                             \
    cudaFuncGetAttributes(&attr, m11::tma_kernel<TT, NT, KS, CH, R, G>);                                 \
    const int64_t row[10] = {(G) ? 1 : 0, TT, NT, KS, CH, R, static_cast<int64_t>(attr.numRegs),        \
                             static_cast<int64_t>(attr.localSizeBytes),                                  \
                             static_cast<int64_t>(attr.sharedSizeBytes), dyn};                          \
    for (int64_t v : row) rows.push_back(v);                                                             \
  }

torch::Tensor m12_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t ks,
                       int64_t ch, int64_t r, int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  @@M12_CASES@@
  TORCH_CHECK(false, "m12_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " ks=", ks, " ch=", ch,
              " r=", r);
  return out;
}

torch::Tensor m12_setup() {
  std::vector<int64_t> rows;
  @@M12_SETUPS@@
  return torch::tensor(rows).view({-1, 10});
}
"""

_M11_CPP = """
torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                       int64_t ks, int64_t grid);
torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);
torch::Tensor m12_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t ks,
                       int64_t ch, int64_t r, int64_t grid);
torch::Tensor m12_setup();
"""


def m11_source(instances=M11_INSTANCES, m12_instances=M12_INSTANCES, cr_instances=(), sr_instances=()):
    """The CUDA source with one dispatch case per instance (cr_instances: E21's continuous-ring kernel)."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in instances]
    args_cr = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in cr_instances]
    args_sr = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in sr_instances]
    args12 = [f"{'true' if g else 'false'}, {tt}, {nt}, {ks}, {ch}, {r}" for g, tt, nt, ks, ch, r in m12_instances]
    return (_M11_CUDA.replace("@@CASES@@", "\n  ".join([f"M11_CASE({a})" for a in args] + [f"M11_CR_CASE({a})" for a in args_cr]
                                                + [f"M11_SR_CASE({a})" for a in args_sr]))
            .replace("@@ATTRS@@", "\n  ".join([f"M11_ATTR({a})" for a in args] + [f"M11_KATTR(cr_kernel, {a}, 16)" for a in args_cr]
                                              + [f"M11_KATTR(sr_kernel, {a}, 32)" for a in args_sr]))
            .replace("@@M12_CASES@@", "\n  ".join(f"M12_CASE({a})" for a in args12))
            .replace("@@M12_SETUPS@@", "\n  ".join(f"M12_SETUP({a})" for a in args12))
            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))


def m11_build():
    """Compile the CUDA kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(), functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],
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


def m11_gemm1(x, ids, state, s1):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair; s1 = cfg["s1"]."""
    bank, n_pairs = state["m11"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    args = (bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, n_pairs, bank["start"], bank["rt_w13"], 0,
            bank["inter"], 1, 2, s1["nt"])
    if s1.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s1["ks"], s1["ch"], s1["r"], s1["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s1["d"] | (s1.get("cr", 0) << 4) | (s1.get("sr", 0) << 5), s1["ks"], s1["c"] * state["sms"])
    return act


def m11_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]; s2 = cfg["s2"]."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    args = (bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"], bank["rt_w2"], 1,
            bank["hidden"], 0, s2["tt"], s2["nt"])
    if s2.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s2["ks"], s2["ch"], s2["r"], s2["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s2["d"] | (s2.get("cr", 0) << 4) | (s2.get("sr", 0) << 5), s2["ks"], s2["c"] * state["sms"])
    return y


def m11_post_norm(y, ids, topk_weights, state, tokens):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
    _m11_post_norm_kernel[(tokens,)](y, state["norm_w"], topk_weights, ids, out, bank["start"], POST_NORM_EPS,
                                     H=bank["hidden"], LOCAL=bank["count"], TOP_K=TOP_K, num_warps=8)
    return out


def m11_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg["s1"])
    y = m11_gemm2(act, ids, state, cfg["s2"])
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0])


def m11_config_for(state, tokens):
    """The M11 config for this token count, or None for v1 (M11 takes at most M11_MAX_TOKENS)."""
    if tokens > M11_MAX_TOKENS:
        return None
    for below, cfg in state["m11_configs"]:
        if tokens < below:
            return cfg
    return None


def m11_warm(state, cfg):
    """Run every kernel of cfg once before any CUDA graph capture (Triton compiles on first launch)."""
    device = state["norm_w"].device
    tokens = 8
    x = torch.zeros((tokens, state["m11"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m11"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m11_run(x, ids + state["m11"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m11_setup(state, weights):
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = m11_build()
    # Dynamic shared-memory limits for the M12 instances, set once here and never during a capture.
    state["m12_resources"] = state["m11_ext"].m12_setup().tolist()
    state["m11"] = m11_bank(weights, 0, 32)


def prepare(weights):
    state = _v1_prepare(weights)
    state["m11_configs"] = M11_CONFIGS
    if any(cfg is not None for _, cfg in M11_CONFIGS):
        m11_setup(state, weights)
        for _, cfg in M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    m11cfg = m11_config_for(state, x.shape[0])
    if (m11cfg is None or "m11" not in state
            or (cfg["local_start"], cfg["local_count"]) != (state["m11"]["start"], state["m11"]["count"])):
        return _v1_kernel_fn(data, state)
    return m11_run(x, data["topk_ids"], data["topk_weights"], state, m11cfg)
