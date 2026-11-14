"""DG12-RS v3: BF16 x (m tokens) times 12-bit F12 weights (bf16_proj's lossless format), out = x @ W^T in BF16, for
m = 32 / 64 / 128 per token tile (m = 256 -> 2 tiles), on sm_90a with register-A wgmma.

Built by muse_prefill_layer for muse_bf16_proj (TRACKER shared item 3); skeleton = prefill's R14 (judged correct):
- CTA = 128 weight rows (2 math warpgroups x 64) x BM tokens x one K slice, 384 threads: warpgroup 0 = producer
  (thread 0 only), 1-2 = math. CPS CTAs per SM (1 or 2: __launch_bounds__(384, CPS), ring = 200 KB / CPS).
- Producer, per 64-wide K chunk: one cp.async.bulk of the stage's 12,288 weight bytes (planes chunk-major
  [K/64][N/16][3][32][4] int32) + one TMA 2-D load of the x tile (tensor map over x [M][K] bf16, box 64 x BM,
  SWIZZLE_128B: token t's 16-byte unit u at t*128 + ((u ^ (t & 7)) << 4); rows >= M zero-filled), both on the stage's
  full barrier (count 1, expect_tx = both). Ring of S >= 4 stages. (v2's 128-thread cp.async + noinc arrivals and the
  math warps' per-chunk proxy fence are gone: TMA writes are async-proxy.)
- Math warp w of warpgroup c owns row tile 4c + w: per chunk 3 x ld.shared.v4, F12M's 7-op decode per k16 step
  into the 4 A registers, 4 x wgmma m64nBMk16 RS against the x tile (SW128 descriptor), wait_group<1>, release.
- Split-K (grid z = S <= 8) as a thread-block cluster (1, 1, S) when S > 1 (plain launch when S = 1, PF41): each
  CTA stages its fp32 tile [BM][128] in its own (free) ring memory; cluster barrier; CTA rank r finalizes rows
  [r * ceil(128 / S), ...): thread -> (row, token lane), its record, scale and escapes loaded once; it sums the S
  tiles over DSMEM in rank order (deterministic), applies F12M's epilogue (x 2^(base - 112), escapes with x from
  global) and stores BF16 (rows contiguous across lanes); final cluster barrier. No global partials, no counters.
- DIAG bits (lab): 1 no finalize, 2 no decode (raw words as A), 4 no x TMA, 8 no weight copy.
"""

import tempfile

import torch
from torch.utils.cpp_extension import load_inline

DGRS_INSTANCES = ((32, 1, 0), (64, 1, 0), (128, 1, 0))  # (BM, CPS, DIAG) compiled by default
DGRS_W_CHUNK = 8 * 3 * 32 * 16  # weight bytes per 128-row x 64-k stage


def _wgmma_rs(n):
    regs = n // 2
    d = ", ".join(f"%{i}" for i in range(regs))
    a = ", ".join(f"%{regs + i}" for i in range(4))
    asm = (r"{\n.reg .pred p;\nsetp.ne.b32 p, %" + str(regs + 5) + r", 0;\n"
           f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, {{{a}}}, %{regs + 4}, p, 1, 1, 0;"
           + r"\n}\n")
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    return (f"__device__ __forceinline__ void wgmma{n}(float (&d)[{regs}], uint32_t a0, uint32_t a1, uint32_t a2, "
            f"uint32_t a3, uint64_t desc, int sc) {{\n  asm volatile(\"{asm}\"\n      : {outs}\n"
            f"      : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc), \"r\"(sc));\n}}\n")


_DGRS_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <stdint.h>

namespace dgrs {

constexpr int W_CHUNK = 12288;          // 8 row tiles x 3 slots x 32 lanes x 16 B
constexpr int SMEM_BYTES = 200 * 1024;  // ring bytes requested: stages = what fits
constexpr int SMAX = 12;                // most ring slots
constexpr long long SPIN_LIMIT = 1ll << 22;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
// Bounded parity wait: a protocol fault flags err and bails instead of hanging the run.
__device__ __forceinline__ bool mbar_wait(uint32_t bar, uint32_t parity, int* err) {
  for (long long i = 0; i < SPIN_LIMIT; ++i) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return true;
    if ((i & 1023) == 1023 && *reinterpret_cast<volatile int*>(err) != 0) return false;
  }
  atomicAdd(err, 1);
  return false;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// TMA 2-D tile load (inner coordinate c0 in elements, outer c1 in rows) into this CTA's shared memory.
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// fp32 at this CTA's shared offset `addr` in cluster rank `rank`'s shared memory (DSMEM)
__device__ __forceinline__ float ld_peer_f32(uint32_t addr, uint32_t rank) {
  uint32_t ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void keep(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void keepf(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// K-major 128B-swizzled B tile: start address, LBO 1 (unused when swizzled), SBO 1024 B (8 rows x 128 B).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

// F12M's decode (bf16_proj, proj/f12m/f12m.py): one lane's A fragment of one k16 step, value * 2^(112 - base).
__device__ __forceinline__ uint4 f12m_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}
__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }

@@WGMMA@@

template <int N>
__device__ __forceinline__ void mma_k64(float (&acc)[N / 2], const uint32_t (&a)[16], uint64_t desc, int first);
#define DGRS_MMA_K64(NN)                                                                            \
  template <>                                                                                       \
  __device__ __forceinline__ void mma_k64<NN>(float (&acc)[NN / 2], const uint32_t (&a)[16], uint64_t desc, \
                                              int first) {                                          \
    wgmma##NN(acc, a[0], a[1], a[2], a[3], desc, first ? 0 : 1);                                    \
    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \
    wgmma##NN(acc, a[8], a[9], a[10], a[11], desc + 4, 1);                                          \
    wgmma##NN(acc, a[12], a[13], a[14], a[15], desc + 6, 1);                                        \
  }
DGRS_MMA_K64(32)
DGRS_MMA_K64(64)
DGRS_MMA_K64(128)

// This lane's 16 A registers for one chunk: step j = 4 registers, Q word j of slot 0, L words of slot 1 + (j >> 1).
// NODEC (DIAG 2, lab): the raw words as A (wrong values, same loads).
template <bool NODEC>
__device__ __forceinline__ void decode_frag(const uint8_t* sw, int woff, uint32_t (&a)[16]) {
  const uint4 q = *reinterpret_cast<const uint4*>(sw + woff);
  const uint4 l01 = *reinterpret_cast<const uint4*>(sw + woff + 512);
  const uint4 l23 = *reinterpret_cast<const uint4*>(sw + woff + 1024);
  if constexpr (NODEC) {
    const uint32_t raw[16] = {q.x, q.y, q.z, q.w, l01.x, l01.y, l01.z, l01.w, l23.x, l23.y, l23.z, l23.w, q.x, l01.y, l23.z, q.w};
#pragma unroll
    for (int r = 0; r < 16; ++r) a[r] = raw[r];
    return;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4& l = j < 2 ? l01 : l23;
    const uint4 r = f12m_decode(word(q, j), word(l, 2 * (j & 1)), word(l, 2 * (j & 1) + 1));
    a[4 * j] = r.x;
    a[4 * j + 1] = r.y;
    a[4 * j + 2] = r.z;
    a[4 * j + 3] = r.w;
  }
}

__device__ __forceinline__ int ring_next(int s, int S) { return s + 1 == S ? 0 : s + 1; }

// grid (N / 128, token tiles, splits), 384 threads, cluster (1, 1, splits) when splits > 1. wp: chunk-major planes
// [K/64][N/16][3][32][4] int32; rec [N][8] int32 (base | count << 8, then k | true bits << 16); xmap: x [M][K] bf16
// (box 64 x BM, SWIZZLE_128B); x itself for the escapes; out [M][N] bf16.
template <int BM, int CPS, int DIAG>
__global__ void __launch_bounds__(384, CPS) rs_kernel(const __grid_constant__ CUtensorMap xmap,
                                                      const uint8_t* __restrict__ wp, const uint4* __restrict__ rec,
                                                      const __nv_bfloat16* __restrict__ x,
                                                      __nv_bfloat16* __restrict__ out, float* __restrict__ part,
                                                      int* __restrict__ cnt, int* __restrict__ err, int M, int N, int K,
                                                      int splits, int smax, int red) {
  static_assert(CPS == 1 || BM <= 64, "2 CTAs per SM only for BM <= 64 (ring of 4 stages in 100 KB)");
  const int rb = blockIdx.x, tt = blockIdx.y, sl = blockIdx.z;
  const int KC = K / 64;
  const int per = (KC + splits - 1) / splits;
  const int kc0 = sl * per;
  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)
  constexpr int RING = SMEM_BYTES / CPS;
  constexpr int X_BYTES = BM * 128;
  constexpr int STAGE = X_BYTES + W_CHUNK;
  constexpr uint32_t TX = ((DIAG & 8) ? 0u : W_CHUNK) + ((DIAG & 4) ? 0u : X_BYTES);
  int S = RING / STAGE;
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  S = S > 3 ? S : 4;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sX = smem;                  // S x BM*128 (1024-aligned: BM % 8 == 0)
  uint8_t* sW = smem + S * X_BYTES;    // S x 12288
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + S * W_CHUNK);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);  // thread 0's expect_tx arrive (bytes: the weight copy + the x TMA)
      mbar_init(empty0 + 8 * s, 2);  // both math warpgroups
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  bool ok = true;
  float acc[BM / 2];
#pragma unroll
  for (int i = 0; i < BM / 2; ++i) acc[i] = 0.f;
  const int c = (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
  if (tid < 128) {
    // ---- producer: thread 0
    if (tid == 0) {
      const uint8_t* wsrc = wp + (static_cast<size_t>(kc0) * (N / 16) + static_cast<size_t>(rb) * 8) * 1536;
      const size_t wstride = static_cast<size_t>(N / 16) * 1536;  // one chunk of all rows
      int s = 0;
      uint32_t round = 0;
      for (int i = 0; i < nk; ++i) {
        if (round > 0) ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
        mbar_expect_tx(full0 + 8 * s, TX);
        if (!(DIAG & 8))
          bulk_g2s(smem_u32(sW + s * W_CHUNK), wsrc + static_cast<size_t>(i) * wstride, W_CHUNK, full0 + 8 * s);
        if (!(DIAG & 4)) tma_2d(smem_u32(sX + s * X_BYTES), &xmap, (kc0 + i) * 64, tt * BM, full0 + 8 * s);
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
    }
  } else {
    // ---- math warpgroups: c = 0, 1; warp w owns row tile 4c + w of the CTA's 8
    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024
    uint32_t fa[16], fb[16], fc[16];
    const uint32_t sX0 = smem_u32(sX);
    // chunk i sits in slot i % S, parity (i / S) & 1; release slot of chunk i - 1 after wait_group<1> at step i
    auto load = [&](int i, uint32_t (&f)[16]) {
      const int s = i % S;
      ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);
      decode_frag<(DIAG & 2) != 0>(sW + s * W_CHUNK, woff, f);
    };
    auto step = [&](int i, const uint32_t (&cur)[16], uint32_t (&nxt)[16]) {
      wg_fence();
      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == 0);
      wg_commit();
      wg_wait<1>();  // chunk i - 1 is done: its slot and its fragment buffer are free
#pragma unroll
      for (int r = 0; r < 16; ++r) keep(nxt[r]);
      if (i >= 1 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % S));
      if (i + 1 < nk) load(i + 1, nxt);
    };
    load(0, fa);
    int i = 0;
    for (; i + 3 <= nk; i += 3) {
      step(i, fa, fb);
      step(i + 1, fb, fc);
      step(i + 2, fc, fa);
    }
    if (i < nk) step(i, fa, fb);
    if (i + 1 < nk) step(i + 1, fb, fc);
    wg_wait<0>();
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) keepf(acc[r]);
  }
  if constexpr ((DIAG & 1) == 0) {
    // ---- epilogue: fp32 tile [BM][128] in this CTA's ring memory (free: every chunk landed and was consumed)
    __syncthreads();
    float* tile = reinterpret_cast<float*>(smem);
    if (tid >= 128) {
      const int r0 = 64 * c + 16 * w + g;  // rows r0, r0 + 8; tokens 8j + 2q (+1)
#pragma unroll
      for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int t = 8 * j + 2 * q + h;
          tile[t * 128 + r0] = acc[4 * j + h];
          tile[t * 128 + r0 + 8] = acc[4 * j + 2 + h];
        }
      }
    }
    // red 0 (cluster, S > 1): DSMEM; red 1 (S > 1, plain launch): global fp32 partials + last-arriver counter
    const bool glob = red != 0 && splits > 1;
    const int Mp = gridDim.y * BM;
    __shared__ int s_last;
    if (glob) {
      __syncthreads();
      for (int idx = tid; idx < BM * 128; idx += 384)
        part[(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N + rb * 128 + (idx & 127)] = tile[idx];
      __threadfence();  // release: this thread's partials before the counter
      __syncthreads();
      if (tid == 0) {
        __threadfence();
        const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;
        if (last) {
          __threadfence();  // acquire: every slice's partials
          atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again
        }
        s_last = last;
      }
      __syncthreads();
    } else if (splits > 1) {
      cluster_sync();  // every slice's tile is written (release / acquire across the cluster)
    } else {
      __syncthreads();
    }
    // rows [rlo, rlo + rn) of this row block are this CTA's (rank sl of nr); thread -> (row ri, token lane tl)
    const int nr = glob ? 1 : splits, rank = glob ? 0 : sl;
    const int RS = (128 + nr - 1) / nr;
    const int rlo = rank * RS;
    const int rhi = rlo + RS < 128 ? rlo + RS : 128;
    const int rn = rhi > rlo ? rhi - rlo : 0;
    const int TL = rn > 0 ? 384 / rn : 0;
    if (rn > 0 && tid < TL * rn && (!glob || s_last)) {
      const int ri = tid % rn, tl = tid / rn;
      const int row = rb * 128 + rlo + ri;
      const uint4 q0 = rec[2 * row], q1 = rec[2 * row + 1];
      const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;
      const float scale = f12m_scale(base), wdec = f12m_wdec(base);
      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
      const uint16_t* xh = reinterpret_cast<const uint16_t*>(x);
      const uint32_t tile0 = smem_u32(tile);
      constexpr int FU = 4;
      for (int t0 = tl; t0 < BM; t0 += TL * FU) {
        int tok[FU];
        bool live[FU];
        float sum[FU];
#pragma unroll
        for (int u = 0; u < FU; ++u) {
          const int t = t0 + TL * u;
          tok[u] = tt * BM + t;
          live[u] = t < BM && tok[u] < M;
          sum[u] = 0.f;
        }
        for (int s2 = 0; s2 < splits; ++s2) {  // rank order: deterministic
          float v[FU];
#pragma unroll
          for (int u = 0; u < FU; ++u) {
            const uint32_t a = tile0 + static_cast<uint32_t>(((t0 + TL * u) * 128 + rlo + ri) * 4);
            v[u] = !live[u] ? 0.f
                   : glob ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row)
                   : splits > 1 ? ld_peer_f32(a, static_cast<uint32_t>(s2))
                                : tile[(t0 + TL * u) * 128 + rlo + ri];
          }
#pragma unroll
          for (int u = 0; u < FU; ++u) sum[u] += v[u];
        }
        float xe[FU][7];
#pragma unroll
        for (int e = 0; e < 7; ++e)
#pragma unroll
          for (int u = 0; u < FU; ++u)
            xe[u][e] = (live[u] && e < static_cast<int>(count))
                           ? __uint_as_float(static_cast<uint32_t>(xh[static_cast<size_t>(tok[u]) * K + (esc[e] & 0xFFFFu)]) << 16)
                           : 0.f;
#pragma unroll
        for (int u = 0; u < FU; ++u) {
          if (!live[u]) continue;
          float acc_s = sum[u] * scale;
#pragma unroll
          for (int e = 0; e < 7; ++e) {
            if (e < static_cast<int>(count)) {
              acc_s = fmaf(-xe[u][e], wdec, acc_s);
              acc_s = fmaf(xe[u][e], __uint_as_float(esc[e] & 0xFFFF0000u), acc_s);
            }
          }
          out[static_cast<size_t>(tok[u]) * N + row] = __float2bfloat16_rn(acc_s);
        }
      }
    }
    if (splits > 1 && !glob) cluster_sync();  // peers may still read this CTA's tile
  }
  if (!ok) atomicAdd(err, 1);
}

}  // namespace dgrs

typedef CUresult (*DgrsEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static DgrsEncodeFn dgrs_encode_fn() {  // the driver's cuTensorMapEncodeTiled, without linking libcuda (as dg12.py)
  static DgrsEncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) fn = reinterpret_cast<DgrsEncodeFn>(dlsym(h, "cuTensorMapEncodeTiled"));
  }
  return fn;
}

// x [M][K] bf16 -> tensor map with box 64 (inner, 128 B) x BM rows, 128B swizzle; rows >= M read as zeros.
static void dgrs_xmap(CUtensorMap* map, const void* x, int M, int K, int bm) {
  DgrsEncodeFn fn = dgrs_encode_fn();
  TORCH_CHECK(fn != nullptr, "dgrs: cuTensorMapEncodeTiled not found");
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(K) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(bm)};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = fn(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), gdims, gstrides, box, estr,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dgrs: cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
}

template <int BM, int CPS, int DIAG>
static void dgrs_launch(dim3 grid, int splits, int red, const CUtensorMap& xmap, const uint8_t* w, const uint4* rc,
                        const __nv_bfloat16* xb, __nv_bfloat16* o, float* part, int* cnt, int* err, int M, int N, int K,
                        int smax) {
  auto kern = dgrs::rs_kernel<BM, CPS, DIAG>;
  const size_t smem = static_cast<size_t>(dgrs::SMEM_BYTES / CPS) + 16 * dgrs::SMAX + 1024;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = static_cast<unsigned>(splits);
  cfg.attrs = attr;
  cfg.numAttrs = (splits > 1 && red == 0) ? 1 : 0;  // a real cluster only for DSMEM split-K (PF41: size 1 costs ~30%)
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kern, xmap, w, rc, xb, o, part, cnt, err, M, N, K, splits, smax, red);
  if (e != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dgrs launch failed: ", cudaGetErrorString(e));
}

void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red) {
  const int KC = static_cast<int>(K / 64);
  const int per = static_cast<int>((KC + splits - 1) / splits);
  const int tiles = static_cast<int>((M + bm - 1) / bm);
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0 && splits >= 1 && splits <= 8 && (splits - 1) * per < KC,
              "dgrs_gemm: shapes / splits (1..8, every slice non-empty)");
  TORCH_CHECK(red == 0 || splits == 1 || (part.numel() >= splits * static_cast<int64_t>(tiles) * bm * N &&
                                           cnt.numel() >= (N / 128) * tiles), "dgrs_gemm: red 1 needs part / cnt");
  const dim3 grid(static_cast<unsigned>(N / 128), static_cast<unsigned>(tiles), static_cast<unsigned>(splits));
  CUtensorMap xmap;
  dgrs_xmap(&xmap, x.data_ptr(), static_cast<int>(M), static_cast<int>(K), static_cast<int>(bm));
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const uint4* rc = reinterpret_cast<const uint4*>(rec.data_ptr());
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define DGRS_CASE(BM, CPS, DIAG)                                                                              \
  if (bm == BM && cps == CPS && diag == DIAG) {                                                               \
    dgrs_launch<BM, CPS, DIAG>(grid, static_cast<int>(splits), static_cast<int>(red), xmap, w, rc, xb, o,      \
                               part.data_ptr<float>(), cnt.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(M),  \
                               static_cast<int>(N), static_cast<int>(K), static_cast<int>(smax));                       \
    return;                                                                                                   \
  }
  @@CASES@@
#undef DGRS_CASE
  TORCH_CHECK(false, "dgrs_gemm: no instance for bm=", bm, " cps=", cps, " diag=", diag);
}

torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag) {
  cudaFuncAttributes a;
#define DGRS_ATTR(BM, CPS, DIAG) if (bm == BM && cps == CPS && diag == DIAG) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM, CPS, DIAG>); } else
  @@ATTRS@@
  { TORCH_CHECK(false, "dgrs_attrs: no instance"); }
#undef DGRS_ATTR
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}
"""

_DGRS_CPP = """
void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red);
torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag);
"""


def _norm_instances(instances):
    out = []
    for inst in (DGRS_INSTANCES if instances is None else instances):
        out.append((inst, 1, 0) if isinstance(inst, int) else tuple(inst))
    return tuple(out)


def dgrs_source(instances=None):
    insts = _norm_instances(instances)
    args = [f"{b}, {c}, {d}" for b, c, d in insts]
    return (_DGRS_CUDA.replace("@@WGMMA@@", "".join(_wgmma_rs(n) for n in (32, 64, 128)))
            .replace("@@CASES@@", "\n  ".join(f"DGRS_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\n  ".join(f"DGRS_ATTR({a})" for a in args)))


def dgrs_build(instances=None):
    """One load_inline module; instances = (BM, CPS, DIAG) tuples (or plain BM ints = (BM, 1, 0)); sm_90a only."""
    return load_inline(
        name="dgrs_ext", cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(instances),
        functions=["dgrs_gemm", "dgrs_attrs"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="dgrs_"), verbose=False)


def dgrs_pack(p):
    """bf16_proj's f12m_device_planes(w, natural=True)["p"] int32 [N/16][K/64][3][32][4] -> chunk-major
    [K/64][N/16][3][32][4] (a stage's 128 rows = one contiguous 12,288-byte block). Lossless reordering."""
    return p.permute(1, 0, 2, 3, 4).contiguous()


def dgrs_buffers(m, n, bm, splits, device):
    """Per-layer scratch: int32 error flag; for red 1 (global split-K reduction) fp32 partials [splits][tiles * bm][n]
    and int32 counters [(n / 128) * tiles] zeroed once (zero again after every call). red 0 (cluster DSMEM) ignores
    part / cnt."""
    tiles = -(-m // bm)
    return {"part": torch.empty(splits * tiles * bm * n, dtype=torch.float32, device=device),
            "cnt": torch.zeros((n // 128) * tiles, dtype=torch.int32, device=device),
            "err": torch.zeros(1, dtype=torch.int32, device=device)}


def dgrs_run(ext, x, wp_cm, rec, out, bufs, bm, splits, smax=8, cps=1, diag=0, red=0):
    """out [m, n] bf16 = x [m, k] bf16 (contiguous) @ W^T, W from the chunk-major 12-bit planes and rec. (bm, cps, diag)
    must be a built instance; splits: K-chunk slices (grid z), 1..8, every slice non-empty; red (splits > 1): 0 = a
    cluster (1, 1, splits) reducing through DSMEM, 1 = plain launch, global fp32 partials + last-arriver counter."""
    m, k = x.shape
    n = out.shape[1]
    ext.dgrs_gemm(wp_cm, rec, x, out, bufs["part"], bufs["cnt"], bufs["err"], m, n, k, bm, splits, smax, cps, diag, red)
    return out
