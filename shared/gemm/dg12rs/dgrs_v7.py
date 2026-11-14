"""DG12-RS v6: BF16 x (m tokens) times 12-bit F12 weights (bf16_proj's lossless format), out = x @ W^T in BF16, for
m = 32 / 64 / 128 per token tile (m = 256 -> 2 tiles), on sm_90a with register-A wgmma.

Built by muse_prefill_layer for muse_bf16_proj (TRACKER shared item 3); skeleton = prefill's R14 (judged correct):
- CTA = 128 weight rows (2 math warpgroups x 64) x BM tokens x one K slice, 384 threads: warpgroup 0 = producer
  (thread 0 copies; v5: warps 1-3 precompute), 1-2 = math (1-4 with MATH 4). CPS CTAs per SM (1 or 2).
- Producer, per 64-wide K chunk: one cp.async.bulk of the stage's 12,288 weight bytes (planes chunk-major
  [K/64][N/16][3][32][4] int32) + one TMA 2-D load of the x tile (tensor map over x [M][K] bf16, box 64 x BM,
  SWIZZLE_128B: token t's 16-byte unit u at t*128 + ((u ^ (t & 7)) << 4); rows >= M zero-filled), both on the stage's
  full barrier (count 1, expect_tx = both). Ring of S >= 4 stages. (v2's 128-thread cp.async + noinc arrivals and the
  math warps' per-chunk proxy fence are gone: TMA writes are async-proxy.)
- Math warp w of warpgroup c owns row tile 4c + w: per chunk 3 x ld.shared.v4, F12M's 7-op decode per k16 step
  into the 4 A registers, 4 x wgmma m64nBMk16 RS against the x tile (SW128 descriptor), wait_group<1>, release.
- Split-K (grid z = S <= 8) as a thread-block cluster (1, 1, S) when S > 1 (plain launch when S = 1, PF41): each
  CTA stages its fp32 tile [BM][128] in its own (free) ring memory; cluster barrier; CTA rank r finalizes rows
  [r * ceil(128 / S), ...): thread -> (row, token lane); it sums the S tiles over DSMEM in rank order
  (deterministic), applies F12M's epilogue (x 2^(base - 112) + the precomputed escape corrections) and stores BF16
  (rows contiguous across lanes); final cluster barrier. red 1: global partials + last-arriver counter instead.
- DIAG bits (lab): 1 no finalize, 2 no decode (raw words as A), 4 no x TMA, 8 no weight copy, 16 keep the math but
  no finalize, 32 no escape precompute (wrong values: times pre_corr's share), 64 timestamps into `part` (v7, values
  unchanged; red 0 / 2), 128 no tensor-map prefetch.
- v4: MATH 4 = two warpgroup pairs on alternate chunks (setmaxnreg 24 / 112). v5 (bf16_proj's PE32: the finalize cost
  7.3 us of 29.2 at unpooler m32, and MATH 4's full build spilled 336 B): the producer warpgroup never runs the
  epilogue; its warps 1-3 precompute the finalize rows' scales and escape corrections (x gathers from global) into
  shared memory during the main loop; the math warpgroups alone finalize = slices (rank order) * scale + corr.
  Shared memory: 226 KB per CTA (112 KB at CPS 2); the ring gets what the scale / escape / correction areas leave.
- v6 (bf16_proj's PE33: unpooler m32 22.54 us = SGLang, 3.9 us after the main loop): red 2 = push: each slice
  stores its accumulators (both MATH 4 pairs) straight into the owner rank's receive area (st.shared::cluster),
  one cluster barrier, finalize from local shared memory (no DSMEM pulls, no pair-sum pass, no exit barrier);
  pre_corr gathers x with 4-byte cp.async, double-buffered (no registers held: PE33's 40 B MATH 4 stack was
  pre_corr's, d32 had 0 B).
"""

import tempfile

import torch
from torch.utils.cpp_extension import load_inline

DGRS_INSTANCES = ((32, 1, 0, 2), (64, 1, 0, 2), (128, 1, 0, 2))  # (BM, CPS, DIAG, MATH) compiled by default
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
constexpr int SMEM_DYN1 = 226 * 1024;  // dynamic shared memory per CTA at 1 CTA/SM (227 KB opt-in max, F47)
constexpr int SMEM_DYN2 = 112 * 1024;  // per CTA at 2 CTAs/SM (228 KB per SM, 1 KB reserved per CTA)
constexpr int PRE_WARPS = 3;           // producer warps 1-3 precompute the finalize rows (pre_corr)
constexpr int ESC_CAP = 304;           // escapes listed per warp: ceil(128 / 3) rows x 7 = 301
constexpr int GATHER = 512;            // pre_corr gather buffer (u32 words): GATHER / BM list entries x BM tokens
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
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
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

// Non-aligned CTA barriers (PTX barrier.sync / barrier.arrive without .aligned): the producer warpgroup and the math
// warpgroups reach them at different instructions (PTX ISA 9.4, barrier{.cta}: arrive releases this thread's prior
// shared-memory writes to the threads that sync on the same barrier).
__device__ __forceinline__ void bar_sync_na(int id, int threads) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void bar_arrive_na(int id, int threads) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
// Split cluster barrier, non-aligned (red 2): arrive (release, or relaxed at kernel start) and wait (acquire).
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }
// fp32 store into cluster shared memory (an address from mapa)
__device__ __forceinline__ uint32_t mapa_u32(uint32_t addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_cluster_f32(uint32_t addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}
// Lab timers (DIAG 64): %globaltimer (ns, comparable across SMs), %clock (this SM's cycles), %smid
__device__ __forceinline__ uint32_t gtimer_lo() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return static_cast<uint32_t>(t);
}
__device__ __forceinline__ uint32_t clk32() {
  uint32_t c;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
  return c;
}
__device__ __forceinline__ uint32_t smid32() {
  uint32_t v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}
// 4-byte cp.async (zero-filled when src_bytes = 0), per-thread groups
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// Shared-memory map (bytes from the 1024-aligned base), the same in every CTA of a launch (DSMEM uses the same offsets
// in the peers): sX [S][BM * 128] | sW [S][12288] | mbarriers 16 * SMAX | sScale [128] f32 | sEsc [PRE_WARPS][ESC_CAP]
// uint2 | sStage [PRE_WARPS][2][GATHER] u32 | sCorr [BM][CS] f32, CS = RS + 1 (the rank's rows, odd stride) |
// red 2 (push) only: sRecv [splits * NP][BM][RSP] f32, RSP = RS rounded up to 4 mod 8 (conflict-free pushes and
// reads). S = the ring stages that fit.
struct DgrsLayout {
  int S, rs, cs, rsp, push, bars, scale, esc, stage, corr, recv, end;
};
__host__ __device__ inline DgrsLayout dgrs_layout(int bm, int cps, int math, int splits, int red, int smax) {
  DgrsLayout L;
  const int nr = (red == 1 && splits > 1) ? 1 : splits;
  L.push = red == 2 && splits > 1;
  L.rs = (128 + nr - 1) / nr;
  L.cs = L.rs + 1;
  L.rsp = L.rs + ((4 - L.rs) & 7);
  const int stage = bm * 128 + W_CHUNK;
  const int fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8 + PRE_WARPS * 2 * GATHER * 4;
  const int corr = (bm * L.cs * 4 + 15) / 16 * 16;
  const int recv = L.push ? splits * (math / 2) * bm * L.rsp * 4 : 0;
  const int avail = (cps == 1 ? SMEM_DYN1 : SMEM_DYN2) - 1024 - fixed - corr - recv;
  int S = avail / stage;
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  if (math == 4) S &= ~1;  // even: slot s only ever holds chunks of parity s % 2, i.e. of pair s % 2
  L.S = S;
  L.bars = S * stage;
  L.scale = L.bars + 16 * SMAX;
  L.esc = L.scale + 128 * 4;
  L.stage = L.esc + PRE_WARPS * ESC_CAP * 8;
  L.corr = L.stage + PRE_WARPS * 2 * GATHER * 4;
  L.recv = L.corr + corr;
  L.end = L.recv + recv;
  return L;
}
// launchable: >= 4 ring stages, and (no push) the epilogue's math / 2 fp32 tiles [BM][128] fit in the ring
__host__ __device__ inline bool dgrs_layout_ok(const DgrsLayout& L, int bm, int math) {
  return L.S >= 4 && (L.push || (math / 2) * bm * 128 * 4 <= L.bars);
}

// Producer warps 1-3 (pw = 0..2), during the main loop: scales and escape corrections of this CTA's finalize rows
// [row0, row0 + rn) of out. Warp pw takes rows [pw * RW, ...), RW = ceil(rn / 3): zeroes their corrections, writes
// their scales, lists their escapes in row then escape order (warp scan of the counts). Then (v6) lane = token gathers
// x with cp.async, G = GATHER / BM list entries per batch, double-buffered (no registers held while loads are in
// flight): the 4-byte word holding x[t][k] (k & ~1) of entry e, token t lands at stage[(buf G + e - b) BM + t]; then
// corr[t][ri] = fma(x, true, fma(-x, wdec, corr[t][ri])) per escape of row ri, in list order.
template <int BM>
__device__ __forceinline__ void pre_corr(const uint4* __restrict__ rec, const uint16_t* __restrict__ xh,
                                         float* sScale, uint2* sEsc, uint32_t* sStage, float* sCorr, int row0, int rn,
                                         int cs, int tok0, int M, int K, int pw, int lane) {
  constexpr int G = GATHER / BM;
  const int RW = (rn + PRE_WARPS - 1) / PRE_WARPS;
  const int wlo = pw * RW;
  const int whi = wlo + RW < rn ? wlo + RW : rn;
  uint2* list = sEsc + pw * ESC_CAP;
#pragma unroll 1
  for (int t = 0; t < BM; ++t)
    for (int ri = wlo + lane; ri < whi; ri += 32) sCorr[t * cs + ri] = 0.f;
  int ne = 0;
  for (int r0 = wlo; r0 < whi; r0 += 32) {
    const int ri = r0 + lane;
    const bool v = ri < whi;
    const uint4 q0 = v ? rec[2 * (row0 + ri)] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t base = q0.x & 0xFFu;
    uint32_t cnt = (q0.x >> 8) & 0xFFu;
    cnt = cnt < 7u ? cnt : 7u;
    const uint4 q1 = cnt > 3u ? rec[2 * (row0 + ri) + 1] : make_uint4(0u, 0u, 0u, 0u);
    if (v) sScale[ri] = f12m_scale(base);
    int inc = static_cast<int>(cnt);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int nb = __shfl_up_sync(0xFFFFFFFFu, inc, o);
      if (lane >= o) inc += nb;
    }
    const int off = ne + inc - static_cast<int>(cnt);
    const uint32_t tag = static_cast<uint32_t>(ri) | (base << 8);
#pragma unroll
    for (int e = 0; e < 7; ++e)
      if (e < static_cast<int>(cnt)) list[off + e] = make_uint2(e < 3 ? word(q0, e + 1) : word(q1, e - 3), tag);
    ne += __shfl_sync(0xFFFFFFFFu, inc, 31);
  }
  __syncwarp();
  uint32_t* stg = sStage + pw * (2 * GATHER);
  const uint32_t stg0 = smem_u32(stg);
  auto issue = [&](int b, int buf) {
    const int be = b + G < ne ? b + G : ne;
#pragma unroll 1
    for (int e = b; e < be; ++e) {
      const uint32_t k2 = list[e].x & 0xFFFEu;  // the 4-byte word holding element k
#pragma unroll
      for (int v2 = 0; v2 < BM / 32; ++v2) {
        const int t = lane + 32 * v2;
        const bool live = tok0 + t < M;
        cp_async4(stg0 + static_cast<uint32_t>(((buf * G + e - b) * BM + t) * 4),
                  xh + static_cast<size_t>(live ? tok0 + t : 0) * K + k2, live ? 4u : 0u);
      }
    }
    cp_async_commit();
  };
  if (ne > 0) issue(0, 0);
  int buf = 0;
#pragma unroll 1
  for (int b = 0; b < ne; b += G) {
    if (b + G < ne) {
      issue(b + G, buf ^ 1);
      cp_async_wait<1>();  // batch b landed (this lane's own words)
    } else {
      cp_async_wait<0>();
    }
    const int be = b + G < ne ? b + G : ne;
#pragma unroll 1
    for (int e = b; e < be; ++e) {
      const uint2 d = list[e];
      const float wdec = f12m_wdec(d.y >> 8), wtrue = __uint_as_float(d.x & 0xFFFF0000u);
#pragma unroll
      for (int v2 = 0; v2 < BM / 32; ++v2) {
        const int t = lane + 32 * v2;
        const uint32_t wv = stg[(buf * G + e - b) * BM + t];
        const float xv = __uint_as_float((d.x & 1u) ? (wv & 0xFFFF0000u) : (wv << 16));
        float* cp = sCorr + t * cs + (d.y & 0xFFu);
        float c = *cp;
        c = fmaf(-xv, wdec, c);
        c = fmaf(xv, wtrue, c);
        *cp = c;
      }
    }
    buf ^= 1;
  }
}

// grid (N / 128, token tiles, splits), 128 + 128 * MATH threads, cluster (1, 1, splits) when splits > 1 and red 0 / 2.
// wp: chunk-major planes [K/64][N/16][3][32][4] int32; rec [N][8] int32 (base | count << 8, then k | true bits << 16);
// xmap: x [M][K] bf16 (box 64 x BM, SWIZZLE_128B); x itself for the escapes; out [M][N] bf16.
// MATH = 2 or 4 math warpgroups; with 4, two pairs (warpgroups 2p + 1, 2p + 2 = pair p) take alternate chunks of the
// same 128 rows (pair p: chunks p, p + 2, ...; even ring, so a slot always serves the same pair), each into its own
// fp32 tile, summed (pair 0 + pair 1) before the finalize. setmaxnreg: producer 24, math 112 (F57: 24 * 128 + 112 * 512
// <= 96 * 640 = the CTA's launch allocation).
// v5: the producer warpgroup never runs the epilogue (it may hold only 24 registers): thread 0 issues the copies,
// warps 1-3 precompute the finalize rows' scales and escape corrections (pre_corr) while the math runs; the math
// warpgroups alone write the tiles, reduce split-K and finalize = sum of slices * scale + corr from shared memory.
template <int BM, int CPS, int DIAG, int MATH>
__global__ void __launch_bounds__(128 + 128 * MATH, CPS) rs_kernel(const __grid_constant__ CUtensorMap xmap,
                                                      const uint8_t* __restrict__ wp, const uint4* __restrict__ rec,
                                                      const __nv_bfloat16* __restrict__ x,
                                                      __nv_bfloat16* __restrict__ out, float* __restrict__ part,
                                                      int* __restrict__ cnt, int* __restrict__ err, int M, int N, int K,
                                                      int splits, int smax, int red) {
  static_assert(CPS == 1 || BM <= 64, "2 CTAs per SM only for BM <= 64");
  static_assert(MATH == 2 || (MATH == 4 && CPS == 1 && BM <= 64), "4 math warpgroups: 1 CTA per SM, BM <= 64");
  static_assert(MATH == 2 || 24 * 128 + 112 * 512 <= 96 * 640, "setmaxnreg split must fit the launch allocation (F57)");
  constexpr int THREADS = 128 + 128 * MATH;
  constexpr int TM = 128 * MATH;   // math threads: the epilogue is theirs alone
  constexpr int NP = MATH / 2;     // warpgroup pairs taking alternate chunks
  constexpr int FU = 16 / MATH;    // tokens per finalize step
  const int rb = blockIdx.x, tt = blockIdx.y, sl = blockIdx.z;
  const int KC = K / 64;
  const int per = (KC + splits - 1) / splits;
  const int kc0 = sl * per;
  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)
  constexpr int X_BYTES = BM * 128;
  constexpr uint32_t TX = ((DIAG & 8) ? 0u : W_CHUNK) + ((DIAG & 4) ? 0u : X_BYTES);
  // splits > 1: red 0 = cluster, peers' tiles pulled over DSMEM; red 1 = plain launch, global fp32 partials +
  // last-arriver counter; red 2 = cluster, each slice pushes its accumulators into the owner rank's receive area
  const bool glob = red == 1 && splits > 1;
  const DgrsLayout L = dgrs_layout(BM, CPS, MATH, splits, red, smax);
  const bool push = L.push;
  const int S = L.S;
  // this CTA's finalize rows [rlo, rlo + rn) of the row block: rank sl of nr (red 1: all 128, by the last arriver)
  const int rank = glob ? 0 : sl;
  const int rlo = rank * L.rs;
  const int rhi = rlo + L.rs < 128 ? rlo + L.rs : 128;
  const int rn = rhi > rlo ? rhi - rlo : 0;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sX = smem;                  // S x BM*128 (1024-aligned: BM % 8 == 0)
  uint8_t* sW = smem + S * X_BYTES;    // S x 12288
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L.bars);
  float* sScale = reinterpret_cast<float*>(smem + L.scale);
  uint2* sEsc = reinterpret_cast<uint2*>(smem + L.esc);
  uint32_t* sStage = reinterpret_cast<uint32_t*>(smem + L.stage);
  float* sCorr = reinterpret_cast<float*>(smem + L.corr);
  float* sRecv = reinterpret_cast<float*>(smem + L.recv);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  // DIAG 64 (lab): 16 u32 per CTA in `part` (red 0 / 2 only): 0 globaltimer at entry, 1 clock at entry, 2 producer's
  // last copy issued, 3 pre_corr done, 4 math loop end (pair 0), 5 reduced (after push / pull barrier), 6 bar 2 passed,
  // 7 finalize done, 8 exit, 9 globaltimer at exit, 10 SM id, 11 math loop end (pair 1, MATH 4); clocks are this SM's
  constexpr bool STAMP = (DIAG & 64) != 0;
  uint32_t* stamps = reinterpret_cast<uint32_t*>(part) +
                     16 * ((static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
  if (STAMP && tid == 0) {
    stamps[0] = gtimer_lo();
    stamps[1] = clk32();
    stamps[10] = smid32();
  }
  if (!(DIAG & 128) && tid == 0)  // the x tensor map's descriptor, before the first TMA (CUTLASS does the same)
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&xmap)) : "memory");
  if (!dgrs_layout_ok(L, BM, MATH)) {  // dgrs_gemm rejects these launches; never taken
    if (tid == 0) atomicAdd(err, 1);
    return;
  }
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);  // thread 0's expect_tx arrive (bytes: the weight copy + the x TMA)
      mbar_init(empty0 + 8 * s, 2);  // both math warpgroups
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if ((DIAG & 17) == 0 && push) cl_arrive_relaxed();  // red 2 phase A: waited before the first push (CTAs started)
  bool ok = true;
  float* tile = reinterpret_cast<float*>(smem);  // epilogue: NP fp32 tiles [BM][128] in the ring (free by then)
  if (tid < 128) {
    if (MATH == 4) setmaxnreg_dec<24>();
    // ---- producer: thread 0 issues the copies; warps 1-3 precompute the finalize rows' scales and corrections
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
      if (STAMP) stamps[2] = clk32();
    } else if (tid >= 32 && (DIAG & 49) == 0) {
      pre_corr<BM>(rec, reinterpret_cast<const uint16_t*>(x), sScale, sEsc, sStage, sCorr, rb * 128 + rlo, rn, L.cs,
                   tt * BM, M, K, (tid >> 5) - 1, tid & 31);
      if (STAMP && tid == 32) stamps[3] = clk32();
    }
    if constexpr ((DIAG & 17) == 0) {
      __syncwarp();
      bar_arrive_na(2, THREADS);  // this CTA's scales and corrections are in shared memory
      if (push) {  // red 2: phase A, then the math warpgroups' push barrier B
        cl_wait();
        cl_arrive();
        cl_wait();
      } else if (splits > 1 && !glob) {  // red 0: the math warpgroups' two cluster barriers
        cluster_sync();
        cluster_sync();
      }
    }
  } else {
    if (MATH == 4) setmaxnreg_inc<112>();
    // ---- math warpgroup mw = pair p (chunks p, p + NP, ...), row half c; warp w owns row tile 4c + w of the CTA's 8
    const int mw = (tid >> 7) - 1, c = mw & 1, p = mw >> 1;
    const int w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024
    float acc[BM / 2];
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) acc[r] = 0.f;
    uint32_t fa[16], fb[16], fc[16];
    const uint32_t sX0 = smem_u32(sX);
    // chunk i sits in slot i % S, parity (i / S) & 1; at this pair's step for chunk i, after wait_group<1>, its
    // previous chunk i - NP is done: release that slot
    auto load = [&](int i, uint32_t (&f)[16]) {
      const int s = i % S;
      ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);
      decode_frag<(DIAG & 2) != 0>(sW + s * W_CHUNK, woff, f);
    };
    auto step = [&](int i, const uint32_t (&cur)[16], uint32_t (&nxt)[16]) {
      wg_fence();
      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == p);
      wg_commit();
      wg_wait<1>();  // this pair's chunk i - NP is done: its slot and its fragment buffer are free
#pragma unroll
      for (int r = 0; r < 16; ++r) keep(nxt[r]);
      if (i >= NP && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - NP) % S));
      if (i + NP < nk) load(i + NP, nxt);
    };
    if (p < nk) {
      load(p, fa);
      int i = p;
      for (; i + 2 * NP < nk; i += 3 * NP) {
        step(i, fa, fb);
        step(i + NP, fb, fc);
        step(i + 2 * NP, fc, fa);
      }
      if (i < nk) step(i, fa, fb);
      if (i + NP < nk) step(i + NP, fb, fc);
      wg_wait<0>();
    }
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) keepf(acc[r]);
    if (STAMP && (tid == 128 || tid == 384)) stamps[tid == 128 ? 4 : 11] = clk32();
    if constexpr ((DIAG & 16) != 0) {
      // lab: keep the math (a store ptxas can't drop), skip the finalize
      float sacc = 0.f;
#pragma unroll
      for (int r = 0; r < BM / 2; ++r) sacc += acc[r];
      if (__float_as_uint(sacc) == 0x7F8FEDCBu) out[tid] = __float2bfloat16_rn(sacc);
    } else if constexpr ((DIAG & 1) == 0) {
      // ---- epilogue (math threads only): split-K reduction, then finalize
      const int mt = tid - 128;
      const int r0 = 64 * c + 16 * w + g;  // rows r0, r0 + 8; tokens 8j + 2q (+1)
      const int Mp = gridDim.y * BM;
      __shared__ int s_last;
      if (push) {
        // red 2: this slice's accumulators go straight into the owner rank's (row / RS) receive slot sl * NP + p;
        // one cluster barrier; every rank then finalizes from its own shared memory (no pulls, no exit barrier)
        cl_wait();  // phase A: every CTA of the cluster is running (its receive area exists)
        const int ra = r0 / L.rs, rb8 = (r0 + 8) / L.rs;
        const int slot = (sl * NP + p) * BM * L.rsp;
        const uint32_t recv0 = smem_u32(sRecv);
        const uint32_t ga = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 - ra * L.rs) * 4), static_cast<uint32_t>(ra));
        const uint32_t gb = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 + 8 - rb8 * L.rs) * 4), static_cast<uint32_t>(rb8));
#pragma unroll
        for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t t = static_cast<uint32_t>(8 * j + 2 * q + h);
            st_cluster_f32(ga + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + h]);
            st_cluster_f32(gb + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + 2 + h]);
          }
        }
        cl_arrive();  // phase B: release this CTA's pushes ...
        cl_wait();    // ... and acquire every slice's
        if (STAMP && tid == 128) stamps[5] = clk32();
      } else {
        // red 0 / 1 / S = 1: the CTA's fp32 tile [BM][128] in the (free) ring, pairs summed in order
        named_sync(1, TM);  // every math warpgroup is done with the ring (all chunks landed and were read)
        float* mtile = tile + p * (BM * 128);
#pragma unroll
        for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int t = 8 * j + 2 * q + h;
            mtile[t * 128 + r0] = acc[4 * j + h];
            mtile[t * 128 + r0 + 8] = acc[4 * j + 2 + h];
          }
        }
        named_sync(1, TM);
        if (MATH == 4) {
          for (int idx = mt; idx < BM * 128; idx += TM) tile[idx] += tile[BM * 128 + idx];
          named_sync(1, TM);
        }
        if (glob) {
          for (int idx = mt; idx < BM * 128; idx += TM)
            part[(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N + rb * 128 + (idx & 127)] = tile[idx];
          __threadfence();  // release: this thread's partials before the counter
          named_sync(1, TM);
          if (mt == 0) {
            __threadfence();
            const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;
            if (last) {
              __threadfence();  // acquire: every slice's partials
              atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again
            }
            s_last = last;
          }
          named_sync(1, TM);
        } else if (splits > 1) {
          cluster_sync();  // every slice's tile is written (release / acquire across the cluster)
        }
        if (STAMP && tid == 128) stamps[5] = clk32();
      }
      bar_sync_na(2, THREADS);  // the producer warps' scales and corrections
      if (STAMP && tid == 128) stamps[6] = clk32();
      // thread -> (row ri, token lane tl): sum the slices in order, * scale + corr, BF16 (rows contiguous)
      const int TL = rn > 0 ? TM / rn : 0;
      if (rn > 0 && mt < TL * rn && (!glob || s_last)) {
        const int ri = mt % rn, tl = mt / rn;
        const int row = rb * 128 + rlo + ri;
        const float scale = sScale[ri];
        const uint32_t tile0 = smem_u32(tile);
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
          const int ns = push ? splits * NP : splits;
          for (int s2 = 0; s2 < ns; ++s2) {  // slice (then pair) order: deterministic
            float v[FU];
#pragma unroll
            for (int u = 0; u < FU; ++u) {
              const uint32_t a = tile0 + static_cast<uint32_t>(((t0 + TL * u) * 128 + rlo + ri) * 4);
              v[u] = !live[u] ? 0.f
                     : push ? sRecv[(s2 * BM + t0 + TL * u) * L.rsp + ri]
                     : glob ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row)
                     : splits > 1 ? ld_peer_f32(a, static_cast<uint32_t>(s2))
                                  : tile[(t0 + TL * u) * 128 + rlo + ri];
            }
#pragma unroll
            for (int u = 0; u < FU; ++u) sum[u] += v[u];
          }
#pragma unroll
          for (int u = 0; u < FU; ++u)
            if (live[u])
              out[static_cast<size_t>(tok[u]) * N + row] =
                  __float2bfloat16_rn(fmaf(sum[u], scale, sCorr[(t0 + TL * u) * L.cs + ri]));
        }
      }
      if (STAMP && tid == 128) stamps[7] = clk32();
      if (splits > 1 && red == 0) cluster_sync();  // peers may still read this CTA's tile
    }
  }
  if (!ok) atomicAdd(err, 1);
  if (STAMP && tid == 128) {
    stamps[8] = clk32();
    stamps[9] = gtimer_lo();
  }
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

template <int BM, int CPS, int DIAG, int MATH>
static void dgrs_launch(dim3 grid, int splits, int red, const CUtensorMap& xmap, const uint8_t* w, const uint4* rc,
                        const __nv_bfloat16* xb, __nv_bfloat16* o, float* part, int* cnt, int* err, int M, int N, int K,
                        int smax) {
  auto kern = dgrs::rs_kernel<BM, CPS, DIAG, MATH>;
  const size_t smem = static_cast<size_t>(CPS == 1 ? dgrs::SMEM_DYN1 : dgrs::SMEM_DYN2);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(128 + 128 * MATH, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = static_cast<unsigned>(splits);
  cfg.attrs = attr;
  cfg.numAttrs = (splits > 1 && red != 1) ? 1 : 0;  // a real cluster only for DSMEM split-K (PF41: size 1 costs ~30%)
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kern, xmap, w, rc, xb, o, part, cnt, err, M, N, K, splits, smax, red);
  if (e != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dgrs launch failed: ", cudaGetErrorString(e));
}

void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red, int64_t math) {
  const int KC = static_cast<int>(K / 64);
  const int per = static_cast<int>((KC + splits - 1) / splits);
  const int tiles = static_cast<int>((M + bm - 1) / bm);
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0 && splits >= 1 && splits <= 8 && (splits - 1) * per < KC,
              "dgrs_gemm: shapes / splits (1..8, every slice non-empty)");
  TORCH_CHECK(red >= 0 && red <= 2, "dgrs_gemm: red 0 (pull), 1 (global), 2 (push)");
  TORCH_CHECK(!(diag & 64) || (!(red == 1 && splits > 1) && part.numel() >= 16 * (N / 128) * tiles * splits),
              "dgrs_gemm: DIAG 64 stamps go to part (red 0 / 2, 16 words per CTA)");
  TORCH_CHECK(red != 1 || splits == 1 || (part.numel() >= splits * static_cast<int64_t>(tiles) * bm * N &&
                                           cnt.numel() >= (N / 128) * tiles), "dgrs_gemm: red 1 needs part / cnt");
  const dgrs::DgrsLayout L = dgrs::dgrs_layout(static_cast<int>(bm), static_cast<int>(cps), static_cast<int>(math),
                                               static_cast<int>(splits), static_cast<int>(red), static_cast<int>(smax));
  TORCH_CHECK(dgrs::dgrs_layout_ok(L, static_cast<int>(bm), static_cast<int>(math)), "dgrs_gemm: shared memory gives ",
              L.S, " ring stages (need >= 4 and room for the epilogue tiles)");
  const dim3 grid(static_cast<unsigned>(N / 128), static_cast<unsigned>(tiles), static_cast<unsigned>(splits));
  CUtensorMap xmap;
  dgrs_xmap(&xmap, x.data_ptr(), static_cast<int>(M), static_cast<int>(K), static_cast<int>(bm));
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const uint4* rc = reinterpret_cast<const uint4*>(rec.data_ptr());
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define DGRS_CASE(BM, CPS, DIAG, MATH)                                                                        \
  if (bm == BM && cps == CPS && diag == DIAG && math == MATH) {                                               \
    dgrs_launch<BM, CPS, DIAG, MATH>(grid, static_cast<int>(splits), static_cast<int>(red), xmap, w, rc, xb, o,      \
                               part.data_ptr<float>(), cnt.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(M),  \
                               static_cast<int>(N), static_cast<int>(K), static_cast<int>(smax));                       \
    return;                                                                                                   \
  }
  @@CASES@@
#undef DGRS_CASE
  TORCH_CHECK(false, "dgrs_gemm: no instance for bm=", bm, " cps=", cps, " diag=", diag, " math=", math);
}

torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag, int64_t math) {
  cudaFuncAttributes a;
#define DGRS_ATTR(BM, CPS, DIAG, MATH) if (bm == BM && cps == CPS && diag == DIAG && math == MATH) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM, CPS, DIAG, MATH>); } else
  @@ATTRS@@
  { TORCH_CHECK(false, "dgrs_attrs: no instance"); }
#undef DGRS_ATTR
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}

// Ring stages of a launch (0 if dgrs_gemm would reject it), for lab logs.
int64_t dgrs_stages(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax) {
  const dgrs::DgrsLayout L = dgrs::dgrs_layout(static_cast<int>(bm), static_cast<int>(cps), static_cast<int>(math),
                                               static_cast<int>(splits), static_cast<int>(red), static_cast<int>(smax));
  return dgrs::dgrs_layout_ok(L, static_cast<int>(bm), static_cast<int>(math)) ? L.S : 0;
}

// Clusters of `splits` CTAs that can be co-resident (cudaOccupancyMaxActiveClusters) for an instance; 0 if splits < 2.
int64_t dgrs_max_clusters(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t splits) {
  if (splits < 2) return 0;
  int n = -1;
#define DGRS_OCC(BM, CPS, DIAG, MATH)                                                                         \
  if (bm == BM && cps == CPS && diag == DIAG && math == MATH) {                                               \
    auto kern = dgrs::rs_kernel<BM, CPS, DIAG, MATH>;                                                         \
    const size_t smem = static_cast<size_t>(CPS == 1 ? dgrs::SMEM_DYN1 : dgrs::SMEM_DYN2);                 \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));           \
    cudaLaunchConfig_t cfg = {};                                                                              \
    cfg.gridDim = dim3(1, 1, static_cast<unsigned>(splits));                                                 \
    cfg.blockDim = dim3(128 + 128 * MATH, 1, 1);                                                              \
    cfg.dynamicSmemBytes = smem;                                                                              \
    cudaLaunchAttribute attr[1];                                                                              \
    attr[0].id = cudaLaunchAttributeClusterDimension;                                                         \
    attr[0].val.clusterDim.x = 1;                                                                             \
    attr[0].val.clusterDim.y = 1;                                                                             \
    attr[0].val.clusterDim.z = static_cast<unsigned>(splits);                                                 \
    cfg.attrs = attr;                                                                                         \
    cfg.numAttrs = 1;                                                                                         \
    if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) n = -1;                               \
    (void)cudaGetLastError();                                                                                 \
  } else
  @@OCCS@@
  { TORCH_CHECK(false, "dgrs_max_clusters: no instance"); }
#undef DGRS_OCC
  return n;
}
"""

_DGRS_CPP = """
void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red, int64_t math);
torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag, int64_t math);
int64_t dgrs_max_clusters(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t splits);
int64_t dgrs_stages(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax);
"""


def _norm_instances(instances):
    out = []
    for inst in (DGRS_INSTANCES if instances is None else instances):
        t = (inst, 1, 0) if isinstance(inst, int) else tuple(inst)
        out.append(t if len(t) == 4 else t + (2,))
    return tuple(out)


def dgrs_source(instances=None):
    insts = _norm_instances(instances)
    args = [f"{b}, {c}, {d}, {m}" for b, c, d, m in insts]
    return (_DGRS_CUDA.replace("@@WGMMA@@", "".join(_wgmma_rs(n) for n in (32, 64, 128)))
            .replace("@@CASES@@", "\n  ".join(f"DGRS_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\n  ".join(f"DGRS_ATTR({a})" for a in args))
            .replace("@@OCCS@@", "\n  ".join(f"DGRS_OCC({a})" for a in args)))


def dgrs_build(instances=None):
    """One load_inline module; instances = (BM, CPS, DIAG[, MATH]) tuples (or plain BM ints = (BM, 1, 0, 2)); sm_90a."""
    return load_inline(
        name="dgrs_ext", cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(instances),
        functions=["dgrs_gemm", "dgrs_attrs", "dgrs_max_clusters", "dgrs_stages"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
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


def dgrs_run(ext, x, wp_cm, rec, out, bufs, bm, splits, smax=8, cps=1, diag=0, red=0, math=2):
    """out [m, n] bf16 = x [m, k] bf16 (contiguous) @ W^T, W from the chunk-major 12-bit planes and rec. (bm, cps, diag)
    must be a built instance; splits: K-chunk slices (grid z), 1..8, every slice non-empty; red (splits > 1): 0 = a
    cluster (1, 1, splits) pulling peers' tiles over DSMEM, 1 = plain launch, global fp32 partials + last-arriver
    counter, 2 = a cluster where each slice pushes into the owner rank's receive area (v6; ext.dgrs_stages(...) = 0
    for configs that don't fit, e.g. BM 128 at S 2)."""
    m, k = x.shape
    n = out.shape[1]
    ext.dgrs_gemm(wp_cm, rec, x, out, bufs["part"], bufs["cnt"], bufs["err"], m, n, k, bm, splits, smax, cps, diag, red,
                  math)
    return out
