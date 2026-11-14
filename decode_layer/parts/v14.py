# v14 = G2P: M11 GEMM2 + the post-expert norm in one kernel (TRACKER shared item 5, built by the muse_prefill_layer
# session in shared/glue/; this is shared/glue/g2p.py byte for byte, sha256 fe7ce03988b6dd87..., then v14's setup below).
# A per-token last arriver computes M13's post-norm from the bf16 y rows (counters zero on entry and exit, no
# memset node, no grid barrier; per-thread release fence before each token-counter increment, the PTX-model
# pattern MoE asked for); PDL path unchanged. Verified by its author (protocol emulation, math vs M13's post-norm
# 4.7e-5, clang, 18/18 mutations); a lab installs it per M11 class (V14_FUSED).

"""G2P: M11 GEMM2 + post-expert norm weighted sum in one kernel (TRACKER shared item 5; built by
shared/glue/make_g2p.py from decode_layer/parts/m11_pdl.py, whose GEMM it copies byte for byte). Do not
edit by hand. Needs the submission namespace: torch, load_inline, tempfile, M11_MAX_TOKENS, POST_NORM_EPS."""

_G2P_CUDA = r'''
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

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
// A task (expert, TT row tiles) goes to a group of KS warps that split K and add their partial sums
// in shared memory; an expert's pairs are taken 8 * NT at a time (a second chunk re-reads its weights).
// L8: L2 prefetch of one warp's K slice of a tile, kgw 64-wide chunks: 512 code bytes and 64 scale bytes
// per chunk, both contiguous and 16-B aligned. mode 0: two bulk prefetches by lane 0
// (cp.async.bulk.prefetch.L2.global, PTX ISA 9.4 9.7.10.28.4.3); 1: each lane prefetches 128-B lines
// (prefetch.global.L2, 9.7.10.16); 2: the same with .L2::evict_last. Hints only: no data changes.
__device__ __forceinline__ void m11_prefetch(const uint4* wp, const uint16_t* sp, int kgw, int mode, int lane) {
  const int wbytes = kgw * 512, sbytes = kgw * 64;
  if (mode == 0) {
    if (lane == 0) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(wp), "r"(wbytes));
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(sp), "r"(sbytes));
    }
    return;
  }
  const char* wc = reinterpret_cast<const char*>(wp);
  const char* sc = reinterpret_cast<const char*>(sp);
  for (int off = lane * 128; off < wbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(wc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(wc + off));
  }
  for (int off = lane * 128; off < sbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(sc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(sc + off));
  }
}

// PDL (MoE E15, as M13's): the next grid may launch once every CTA of this one has signalled; a PDL-launched grid
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


// ------------------------------------------------------------------------------------------------ G2P
// Group barrier of KS warps (M11's group_sync ids 1-4), or a warp sync for a one-warp group.
__device__ __forceinline__ void g2p_sync(int grp, int ks) {
  if (ks == 1) __syncwarp();
  else group_sync(grp + 1, ks * 32);
}

__device__ __forceinline__ const __nv_bfloat16* a_y(const __nv_bfloat16* p) { return p; }

// M13's post-norm for one token by the KS * 32 threads of a group: v = y * nw (y bf16, nw bf16, fp32 products),
// rs[k] = rsqrt(sum v^2 / K + eps), out = bf16_rn(sum over local k of (tw * rs[k]) * v). Two passes over the rows
// (sum of squares, then the weighted sum) keep registers low; thread tig owns uint4 columns tig + i * KS * 32.
template <int KS>
__device__ __forceinline__ void g2p_token(int tk, int tig, int grp, const int* __restrict__ ids, int local_start,
                                          const __nv_bfloat16* y, const __nv_bfloat16* __restrict__ nw,
                                          const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout, float eps,
                                          float (*s_pss)[KS][TOP_K], float (*s_rs)[TOP_K]) {
  constexpr int NT_ = KS * 32;
  constexpr int PER = (K / 8) / NT_;  // uint4 columns per thread
  const int lane = tig & 31, wq = tig >> 5;
  bool loc[TOP_K];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tk * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    ss[k] = 0.0f;
  }
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        const float v0 = __uint_as_float(u << 16) * __uint_as_float(wu << 16);
        const float v1 = __uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u);
        ss[k] += v0 * v0 + v1 * v1;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0)
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_pss[grp][wq][k] = ss[k];
  g2p_sync(grp, KS);
  if (tig < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < KS; ++i) s += s_pss[grp][i][tig];
    s_rs[grp][tig] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  g2p_sync(grp, KS);
  float f[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) f[k] = loc[k] ? tw[tk * TOP_K + k] * s_rs[grp][k] : 0.0f;
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        o[2 * j] += f[k] * (__uint_as_float(u << 16) * __uint_as_float(wu << 16));
        o[2 * j + 1] += f[k] * (__uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u));
      }
    }
    uint32_t pk[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
      const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
      pk[j] = lo | (hi << 16);
    }
    reinterpret_cast<uint4*>(pout + static_cast<size_t>(tk) * K)[col] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  g2p_sync(grp, KS);  // s_rs is rewritten by the next token
}

// PDL (GEMM2 only): launched early with programmatic stream serialization; scans the routing, then every CTA waits
// and reads the activations with coherent token-ordered loads (__ldg's non-coherent path isn't safe behind the wait).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_post_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf,
    const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout,
    int* __restrict__ cnt, float eps) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  static_assert(!GATED, "G2P: GEMM2 only");
  __shared__ int s_fin[4 / KS];
  __shared__ int s_ndone[4 / KS];
  __shared__ int s_done[4 / KS][MAXP];
  __shared__ float s_pss[4 / KS][KS][TOP_K];
  __shared__ float s_rs[4 / KS][TOP_K];
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
  for (int zt = blockIdx.x; zt < n_pairs / TOP_K; zt += gridDim.x) {  // G2P: tokens without a local route: zero rows
    int any = 0;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      const int ze = ids[zt * TOP_K + k] - local_start;
      any |= (ze >= 0 && ze < LOCAL) ? 1 : 0;
    }
    if (!any)
      for (int i = tid; i < K / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(pout + static_cast<size_t>(zt) * K)[i] = make_uint4(0u, 0u, 0u, 0u);
  }

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {
    if (pf & 15) {  // L8: this warp's K slice of the task `ahead` strides on (on the first task: 1..ahead)
      const int ahead = pf & 15, stride = gridDim.x * GROUPS;
      for (int k = task < stride ? 1 : ahead; k <= ahead; ++k) {
        const int nxt = task + k * stride;
        if (nxt >= n_tasks) break;
        const int e2 = s_active[nxt / per_expert], tile2 = nxt % per_expert;
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          const int rt2 = GATED ? tile2 + i * (rt_total / 2) : tile2 * TT + i;
          const size_t b2 = (static_cast<size_t>(e2) * rt_total + rt2) * KG * 32 + static_cast<size_t>(kpart) * KGW * 32;
          m11_prefetch(w + b2, sf + b2, KGW, pf >> 4, lane);
        }
      }
    }
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
    {  // G2P: this task's y tiles are written (warp kpart 0 of the group stored them)
      const int tig = kpart * 32 + lane;  // thread in group
      if (kpart == 0) {
        __threadfence();  // release: this lane's y stores before the counter
        __syncwarp();
        int fin = 0;
        if (lane == 0) {
          fin = atomicAdd(cnt + e, 1) == per_expert - 1;
          if (fin) {
            __threadfence();  // acquire: every other task's y stores of expert e
            atomicExch(cnt + e, 0);  // self-reset: no other task of this call touches cnt[e] again
          }
        }
        fin = __shfl_sync(0xffffffffu, fin, 0);
        if (lane == 0) {
          s_fin[grp] = fin;
          s_ndone[grp] = 0;
        }
      }
      g2p_sync(grp, KS);
      if (s_fin[grp]) {  // this group finished expert e: count its pairs' tokens
        for (int i = tig; i < total; i += KS * 32) {
          const int tk = s_list[e][i] / TOP_K;
          int nl = 0;
#pragma unroll
          for (int k = 0; k < TOP_K; ++k) {
            const int e2 = ids[tk * TOP_K + k] - local_start;
            nl += (e2 >= 0 && e2 < LOCAL) ? 1 : 0;
          }
          __threadfence();  // release by this thread: expert e's y rows (acquired by the group above) before the token counter
          if (atomicAdd(cnt + LOCAL + tk, 1) == nl - 1) {
            __threadfence();  // acquire: the other experts' y rows of token tk
            atomicExch(cnt + LOCAL + tk, 0);
            s_done[grp][atomicAdd(&s_ndone[grp], 1)] = tk;
          }
        }
        g2p_sync(grp, KS);
        const int nd = s_ndone[grp];
        for (int d = 0; d < nd; ++d)
          g2p_token<KS>(s_done[grp][d], tig, grp, ids, local_start, a_y(out), nw, tw, pout, eps, s_pss, s_rs);
      }
      g2p_sync(grp, KS);  // s_fin / s_done / s_ndone are rewritten by the next task
    }
  }
}

}  // namespace m11

template <typename... Exp, typename... Act>
static void g2p_launch(void (*kernel)(Exp...), int grid, int pdl, Act... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "g2p launch failed: ", cudaGetErrorString(err));
}

#define G2P_CASE(TT, NT, D, KS, P)                                                                          \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                               \
    g2p_launch(m11::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>, static_cast<int>(grid), (P),           \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total), 1,             \
               static_cast<int>(out_cols), 0, reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()),           \
               static_cast<const float*>(tw.data_ptr<float>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
               cnt.data_ptr<int>(), static_cast<float>(eps));                                                 \
    return out;                                                                                              \
  }

#define G2P_ATTR(TT, NT, D, KS, P)                                                                           \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                                \
    cudaFuncAttributes attr;                                                                                 \
    cudaFuncGetAttributes(&attr, m11::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>);                     \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes),    \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                      \
  }

// M11's GEMM2 (y = act @ w2^T for every local pair, y bf16 [n_pairs, 4096]) + the post-norm weighted sum into out
// [n_pairs / 8, 4096]; cnt int32 [32 + tokens] must be zero on entry and is zero again on exit.
torch::Tensor g2p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor y, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t out_cols,
                       int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid, int64_t pdl, torch::Tensor nw,
                       torch::Tensor tw, torch::Tensor out, torch::Tensor cnt, double eps) {
  TORCH_CHECK(out_cols == m11::K && nw.numel() == m11::K && out.size(1) == m11::K && out.size(0) * 8 == n_pairs,
              "g2p_gemm: shapes");
  TORCH_CHECK(cnt.numel() >= m11::LOCAL + out.size(0) && out.size(0) <= m11::MAXP, "g2p_gemm: counters");
  @@CASES@@
  TORCH_CHECK(false, "g2p_gemm: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks, " pdl=", pdl);
  return out;
}

torch::Tensor g2p_attrs(int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t pdl) {
  @@ATTRS@@
  TORCH_CHECK(false, "g2p_attrs: no instance");
  return torch::Tensor();
}
'''

_G2P_CPP = '''
torch::Tensor g2p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor y, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t out_cols,
                       int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid, int64_t pdl, torch::Tensor nw,
                       torch::Tensor tw, torch::Tensor out, torch::Tensor cnt, double eps);
torch::Tensor g2p_attrs(int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t pdl);
'''


# G2P_INSTANCES: (tt, nt, depth, ks, pdl) compiled.
G2P_INSTANCES = ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 1, 4, 4, 0), (2, 2, 2, 4, 0))


def g2p_source(instances=None):
    """The CUDA text with one case per instance; instances default to G2P_INSTANCES read at call time (so a submission
    may reassign the module-level tuple to build fewer instances)."""
    instances = G2P_INSTANCES if instances is None else instances
    args = [f"{tt}, {nt}, {d}, {ks}, {p}" for tt, nt, d, ks, p in instances]
    return (_G2P_CUDA.replace("@@CASES@@", "\n  ".join(f"G2P_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\n  ".join(f"G2P_ATTR({a})" for a in args))
            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))


def g2p_build():
    """Compile the fused kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="g2p_ext", cpp_sources=_G2P_CPP, cuda_sources=g2p_source(), functions=["g2p_gemm", "g2p_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="g2p_"), verbose=False)


def g2p_setup(state):
    """After m11_setup: the extension and the self-resetting counters (32 experts + up to M11_MAX_TOKENS tokens)."""
    state["g2p_ext"] = g2p_build()
    state["g2p_cnt"] = torch.zeros(32 + M11_MAX_TOKENS, dtype=torch.int32, device=state["norm_w"].device)


def m11_gemm2_post(act, ids, tw, state, tokens, s2):
    """m11_post_norm(m11_gemm2(act, ids, state, s2), ids, tw, state, tokens) in one kernel: act bf16 [pairs, I],
    ids int32 [pairs] (global expert ids), tw float32 [tokens, 8] contiguous -> bf16 [tokens, 4096]. s2 = cfg["s2"]
    (tt, nt, d, ks, c, optional pdl) of an instance in G2P_INSTANCES."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    return state["g2p_ext"].g2p_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                                     bank["rt_w2"], bank["hidden"], s2["tt"], s2["nt"], s2["d"], s2["ks"],
                                     s2["c"] * state["sms"], int(s2.get("pdl", 0)), state["norm_w"], tw, out,
                                     state["g2p_cnt"], POST_NORM_EPS)


def g2p_ok(s2):
    """True when GEMM2 stage s2 has a fused instance."""
    return (s2.get("kind") != "m12" and (s2["tt"], s2["nt"], s2["d"], s2["ks"], int(s2.get("pdl", 0)))
            in G2P_INSTANCES)


def g2p_m11_run(x, topk_ids, topk_weights, state, cfg):
    """m11_run, with cfg["g2p"] taking the fused GEMM2 + post-norm (cfg["post"] is then unused)."""
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg["s1"])
    if cfg.get("g2p"):
        return m11_gemm2_post(act, ids, topk_weights.contiguous(), state, x.shape[0], cfg["s2"])
    y = m11_gemm2(act, ids, state, cfg["s2"])
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0], cfg.get("post"))


def g2p_install(state, picks):
    """state["m11_configs"] with cfg["g2p"] = picks[below] for the classes in picks (new dicts; classes whose s2 has
    no fused instance stay off); the configs before the first install are kept in state["g2p_base_configs"]."""
    base = state.setdefault("g2p_base_configs", state["m11_configs"])
    state["m11_configs"] = tuple(
        (below, dict(cfg, g2p=bool(picks.get(below)) and g2p_ok(cfg["s2"])) if cfg is not None else cfg)
        for below, cfg in base)
    state["g2p_picks"] = {b: bool(v) for b, v in picks.items()}


def g2p_warm(state):
    """Run every fused instance the current configs can use once (8 tokens), before any graph capture."""
    for _below, cfg in state["m11_configs"]:
        if cfg is not None and g2p_ok(cfg["s2"]):
            dev = state["norm_w"].device
            x = torch.zeros((8, state["m11"]["hidden"]), dtype=torch.bfloat16, device=dev)
            ids = (torch.arange(64, dtype=torch.int32, device=dev) % state["m11"]["count"]).view(8, 8)
            tw = torch.full((8, 8), 0.125, dtype=torch.float32, device=dev)
            g2p_m11_run(x, ids + state["m11"]["start"], tw, state, dict(cfg, g2p=True))
    torch.cuda.synchronize()


def g2p_check(state, token_counts=(1, 4, 8, 16, 32), seed=5):
    """Worst relative error of the fused path vs GEMM2 + post-norm (the same cfg) over random routings with
    ~1/4 local routes, tokens without local routes included; and whether the counters are zero afterwards."""
    dev = state["norm_w"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    worst = 0.0
    for tokens in token_counts:
        cfg = None
        for below, c in state["m11_configs"]:
            if tokens < below:
                cfg = c
                break
        if cfg is None or not g2p_ok(cfg["s2"]):
            continue
        x = (torch.randn((tokens, state["m11"]["hidden"]), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
        scores = torch.rand((tokens, 256), generator=gen, device=dev)
        scores[:, :32] += 0.3
        ids = scores.topk(8, dim=-1).indices.to(torch.int32) + state["m11"]["start"]
        ids = torch.where(ids >= 256, ids - 256, ids)
        tw = torch.softmax(torch.randn((tokens, 8), generator=gen, device=dev), -1)
        ref = g2p_m11_run(x, ids, tw, state, dict(cfg, g2p=False)).float()
        out = g2p_m11_run(x, ids, tw, state, dict(cfg, g2p=True)).float()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-30)) if bool(torch.isfinite(out).all()) else float("inf")
        worst = max(worst, err)
    torch.cuda.synchronize()
    return worst, int(state["g2p_cnt"].abs().sum().item()) == 0

# ---------------------------------------------------------------- v14 setup (decode_layer)

V14_FUSED = {}  # M11 class bound -> True, from a lab's printed V14_FUSED
# Only the instances the current M11 configs use (class 5: (1,1,4,4) PDL; 9 and 17: (2,1,4,4) PDL since DE17;
# 33: (2,2,2,4) plain): (2,1,4,4) without PDL has no class left, so it isn't compiled (prepare time, 300 s cap).
G2P_INSTANCES = ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 2, 2, 4, 0))
g2p_build = _v9_timed("g2p_nvcc", g2p_build)  # its nvcc time joins the prepare-stage line (V9_TIMES)
V14_FORCE = False  # a lab sets it: build and check even with no picks (a clean file with no picks skips the build)
m11_run = g2p_m11_run  # v3_experts / v15_experts call m11_run: cfg["g2p"] selects the fused kernel, else as before


def v14_setup(state, weights):
    """After v10_setup (G2P installs on top of the PDL configs): build, check the fused path against GEMM2 +
    post-norm at 1 / 4 / 8 / 16 / 32 tokens (< 1e-2, counters back to zero), install V14_FUSED, warm."""
    picks = {below: bool(on) for below, on in V14_FUSED.items() if on}
    state["v14_ok"] = False
    if (not picks and not V14_FORCE) or "m11" not in state:
        print(f"[v14] G2P fused GEMM2 + post-norm: not used ({'no picks' if 'm11' in state else 'no M11'})")
        return
    try:
        g2p_setup(state)
        g2p_install(state, {})
        err, clean = g2p_check(state)
        if not (err < 1e-2 and clean):
            raise RuntimeError(f"fused path differs from GEMM2 + post-norm: rel err {err:.1e}, counters zero {clean}")
        g2p_install(state, picks)
        g2p_warm(state)
        state["v14_ok"] = True
        print(f"[v14] G2P fused GEMM2 + post-norm: on (check rel err {err:.1e}, counters clean); installed {picks}")
    except Exception as exc:
        if "g2p_base_configs" in state:
            state["m11_configs"] = state["g2p_base_configs"]
        print(f"[v14] G2P fused GEMM2 + post-norm: OFF ({type(exc).__name__}: {str(exc)[-200:]})")
