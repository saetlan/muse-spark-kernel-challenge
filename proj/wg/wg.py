"""WG: wgmma GEMM (load_inline, sm_90a) for out = x @ W.T at m = 128/256 on BF16 weights, with split-K
inside a cluster (P31).

Tile: 128 weight rows (two consumer warpgroups, 64 rows each, as wgmma's M) x 128 tokens (wgmma N) per
CTA; m = 256 is two token tiles, adjacent in launch order. K in 64-wide chunks; a CTA of a KS-CTA cluster
takes chunks [rank * C / KS, (rank + 1) * C / KS).
Pipeline: ST stages of (A 128 x 64, B 128 x 64) BF16 tiles, each a TMA 2D load with 128B swizzle into a
1024-B aligned 16 KB buffer; full[s] (1 arrival + tx bytes) and empty[s] (256 consumer arrivals)
mbarriers; one producer thread (warpgroup 0) issues the loads; each consumer warpgroup issues 4
`wgmma.m64n128k16` per chunk from K-major descriptors (SBO 1024 B, 128B swizzle, start + 32 B per k16),
commits, waits for the previous group and frees its stage. Bounded mbarrier waits: a protocol error
gives wrong numbers (rejected by the lab gate), not a hang.
Epilogue: KS = 1 stores BF16 straight from the accumulators; KS > 1 writes the FP32 tile into the (then
idle) stage memory, and after a cluster barrier rank r sums rows [r * 128/KS, ...) over the KS peers in
a fixed order (mapa + ld.shared::cluster) and stores BF16. One kernel, deterministic, no workspace.
check_wg.py parses this file's index helpers and checks the swizzle, descriptor and fragment math.
"""

import tempfile


def wgmma_asm(n=128):
    """The inline asm of one wgmma.m64n{n}k16 (FP32 accumulate, A and B from descriptors, scale-d by a register)."""
    nd = n // 2
    regs = ", ".join(f"%{i}" for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 2}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nd}, %{nd + 1}, p, 1, 1, 0, 0;\\n}}"\n'
            f'      : ' + ", ".join(f'"+f"(d[{i}])' for i in range(nd)) +
            f'\n      : "l"(da), "l"(db), "r"(scale_d));')


def wgmma_rs_asm(n=128):
    """One wgmma.m64n{n}k16 with A from four registers (RS form), B from a descriptor, scale-d by a register."""
    nd = n // 2
    regs = ", ".join(f"%{i}" for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 5}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nd}, %{nd + 1}, %{nd + 2}, %{nd + 3}}}, '
            f'%{nd + 4}, p, 1, 1, 0;\\n}}"\n'
            f'      : ' + ", ".join(f'"+f"(d[{i}])' for i in range(nd)) +
            f'\n      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(db), "r"(scale_d));')


WG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace wg {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

// Bounded wait for the phase of the given parity; after a failed wait every later one returns at once
// (flag in shared memory), so a protocol error costs at most ~one bounded spin, never a hang.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity, volatile uint32_t* failed) {
  uint32_t done = 0;
  for (int it = 0; it < (1 << 20) && !done && !*failed; ++it) {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  }
  if (!done) *failed = 1u;
}

__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
               : "memory");
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

// K-major, 128B-swizzle shared memory descriptor (PTX ISA 9.4 sec. 9.7.17.5.1.2.2): start address,
// LBO 1 (unused for swizzled K-major), SBO 1024 B (next 8 rows), base offset 0 (1024-B aligned tiles),
// swizzle mode 1 (128B).
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1u) << 16) |
         (static_cast<uint64_t>(1024u >> 4) << 32) | (static_cast<uint64_t>(1u) << 62);
}

__device__ __forceinline__ void wgmma_128(float (&d)[64], uint64_t da, uint64_t db, uint32_t scale_d) {
  @@WGMMA128@@
}

template <int KS>
__device__ __forceinline__ void cluster_sync() {
  if constexpr (KS > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
    __syncthreads();
  }
}

template <int KS>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (KS > 1) {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
  } else {
    return 0u;
  }
}

__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}

// Index helpers (check_wg.py evaluates their return expressions): first and end chunk of rank r of ks;
// accumulator element e of consumer warpgroup w's thread (warp q, lane g/t) -> tile row and token;
// rows rank r reduces; byte offset of the k16 step j inside a stage's A rows of warpgroup w / B.
__device__ __forceinline__ int wg_c0(int r, int chunks, int ks) { return r * chunks / ks; }
__device__ __forceinline__ int wg_row(int w, int q, int g, int e) { return 64 * w + 16 * q + g + 8 * ((e >> 1) & 1); }
__device__ __forceinline__ int wg_tok(int t, int e) { return 8 * (e >> 2) + 2 * t + (e & 1); }
__device__ __forceinline__ int wg_aoff(int w, int j) { return 8192 * w + 32 * j; }
__device__ __forceinline__ int wg_boff(int j) { return 32 * j; }

__device__ __forceinline__ float4 ld_peer_v4(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float4 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t bf16x2_bits(float lo, float hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// Epilogue items (check_wg.py): item idx of rank r covers token wg_etok and rows wg_erow .. + 7.
__device__ __forceinline__ int wg_etok(int idx, int groups) { return idx / groups; }
__device__ __forceinline__ int wg_erow(int idx, int groups, int r, int share) { return r * share + (idx % groups) * 8; }

// Shared epilogue (all 384 threads): the consumers' accumulators go to shared memory as [token][row] FP32
// over the idle stages; after a cluster barrier rank r sums rows [r * 128/KS, ...) over its KS peers, 8
// consecutive rows of one token per item (two 16-B loads per peer, all issued before the sum), scales each
// row by 2^(base - 112) for WGF, and stores the 8 BF16 as one 16-B vector (coalesced along the token row).
template <int KS, bool SCALED>
__device__ __forceinline__ void wg_epilogue(const float (&d)[64], uint8_t* smem, const uint32_t* bases, int wgi, int tid,
                                            uint32_t rank, __nv_bfloat16* out, long long n_rows, long long row0,
                                            long long tok0, int m) {
  const int lane = tid & 31, q = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
  float* red = reinterpret_cast<float*>(smem);
  if (wgi > 0) {
    asm volatile("bar.sync 1, 256;" ::: "memory");          // both consumer warpgroups done with the stages
#pragma unroll
    for (int e = 0; e < 64; ++e) red[wg_tok(t, e) * 128 + wg_row(wgi - 1, q, g, e)] = d[e];
  }
  cluster_sync<KS>();
  constexpr int SHARE = 128 / KS, GROUPS = SHARE / 8;
  const uint32_t red_addr = smem_u32(red);
  for (int idx = tid; idx < GROUPS * 128; idx += 384) {
    const int tk = wg_etok(idx, GROUPS), row = wg_erow(idx, GROUPS, static_cast<int>(rank), SHARE);
    float4 lo[KS], hi[KS];
#pragma unroll
    for (int p = 0; p < KS; ++p) {
      if constexpr (KS > 1) {
        const uint32_t a = red_addr + 4u * static_cast<uint32_t>(tk * 128 + row);
        lo[p] = ld_peer_v4(a, static_cast<uint32_t>(p));
        hi[p] = ld_peer_v4(a + 16u, static_cast<uint32_t>(p));
      } else {
        lo[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row);
        hi[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row + 4);
      }
    }
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < KS; ++p) {
      v[0] += lo[p].x; v[1] += lo[p].y; v[2] += lo[p].z; v[3] += lo[p].w;
      v[4] += hi[p].x; v[5] += hi[p].y; v[6] += hi[p].z; v[7] += hi[p].w;
    }
    if constexpr (SCALED) {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] *= __uint_as_float((bases[row + i] + 15u) << 23);   // 2^(base - 112)
    }
    const long long tok = tok0 + tk;
    if (tok < m) {
      *reinterpret_cast<uint4*>(out + tok * n_rows + row0 + row) =
          make_uint4(bf16x2_bits(v[0], v[1]), bf16x2_bits(v[2], v[3]), bf16x2_bits(v[4], v[5]), bf16x2_bits(v[6], v[7]));
    }
  }
  if constexpr (KS > 1) cluster_sync<KS>();                // peers may still read this CTA's tile
}

constexpr int STAGE_BYTES = 32768;     // A 16 KB + B 16 KB
constexpr int A_BYTES = 16384;

template <int KS, int ST, int TT>
__global__ void __launch_bounds__(384, 1)
wg_kernel(const __grid_constant__ CUtensorMap tw, const __grid_constant__ CUtensorMap tx,
          __nv_bfloat16* __restrict__ out, int n_rows, int chunks, int m) {
  static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "row shares need KS | 128");
  static_assert(ST >= 2 && ST * STAGE_BYTES >= 128 * 128 * 4, "the split-K tile reuses the stage memory");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES);
  uint64_t* empty = full + ST;
  volatile uint32_t* failed = reinterpret_cast<volatile uint32_t*>(empty + ST);
  const int tid = threadIdx.x, wgi = tid >> 7;
  const uint32_t rank = cluster_rank<KS>();
  const int tile = blockIdx.x / KS;
  const int rowtile = tile / TT, tt = tile % TT;
  const int c0 = wg_c0(static_cast<int>(rank), chunks, KS), c1 = wg_c0(static_cast<int>(rank) + 1, chunks, KS);
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1u);
      mbar_init(&empty[s], 256u);
    }
    *failed = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float d[64];
  if (wgi == 0) {
    if (tid == 0) {
      for (int c = c0, i = 0; c < c1; ++c, ++i) {
        const int s = i % ST;
        mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);
        mbar_expect_tx(&full[s], static_cast<uint32_t>(STAGE_BYTES));
        tma_load_2d(smem + s * STAGE_BYTES, &tw, &full[s], 64 * c, 128 * rowtile);
        tma_load_2d(smem + s * STAGE_BYTES + A_BYTES, &tx, &full[s], 64 * c, 128 * tt);
      }
    }
  } else {
    const int w = wgi - 1;
#pragma unroll
    for (int e = 0; e < 64; ++e) d[e] = 0.0f;
    for (int c = c0, i = 0; c < c1; ++c, ++i) {
      const int s = i % ST;
      mbar_wait(&full[s], static_cast<uint32_t>((i / ST) & 1), failed);
      const uint32_t a0 = smem_u32(smem + s * STAGE_BYTES), b0 = smem_u32(smem + s * STAGE_BYTES + A_BYTES);
      wgmma_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_128(d, wg_desc(a0 + wg_aoff(w, j)), wg_desc(b0 + wg_boff(j)), 1u);
      wgmma_commit();
      wgmma_wait<1>();
      if (i > 0) mbar_arrive(&empty[(i - 1) % ST]);
    }
    wgmma_wait<0>();
  }

  wg_epilogue<KS, false>(d, smem, nullptr, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);
}

// ---------------- WGF: the same pipeline with 12-bit weights in registers (RS wgmma, P31) ----------------
// A = the warp's 16-row tile of F12M planes in natural k order (f12m_layout natural=True), decoded per k16
// step in 7 integer ops (scaled by 2^(112 - base) per row); B = x from the TMA stage. Escapes are applied
// in the loop, after each chunk's MMAs completed, from the x values already in shared memory; each row's
// 2^(base - 112) is applied once after the split-K sum. Exact, no tail loads.
__device__ __forceinline__ uint32_t wword(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

__device__ __forceinline__ uint4 wgf_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

__device__ __forceinline__ void wgmma_rs_128(float (&d)[64], const uint4& a, uint64_t db, uint32_t scale_d) {
  @@WGMMARS128@@
}

__device__ __forceinline__ void ld_wf(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

// Index helpers (check_wg.py): uint4 index of plane slot j of 16-row tile `tile`, chunk c (chs chunks per
// row); the tile of warp q of consumer warpgroup w in row tile rt; byte offset of (token, kk) in a 128B-
// swizzled 128 x 64 BF16 stage.
__device__ __forceinline__ long long wgf_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long wgf_tile(long long rt, int w, int q) { return 8 * rt + 4 * w + q; }
__device__ __forceinline__ int wgf_xoff(int tok, int kk) { return tok * 128 + ((((2 * kk) >> 4) ^ (tok & 7)) << 4) + ((2 * kk) & 15); }

constexpr int XB = 16384;   // one x stage: 128 tokens x 64 k BF16

template <int KS, int ST, int TT, int D>
__global__ void __launch_bounds__(384, 1)
wgf_kernel(const __grid_constant__ CUtensorMap tx, const uint4* __restrict__ wp, const uint4* __restrict__ rec,
           __nv_bfloat16* __restrict__ out, int n_rows, int chunks, int m) {
  static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "row shares need KS | 128");
  static_assert(ST >= 4 && D >= 1, "the 64 KB split-K tile reuses the x stages");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * XB);
  uint64_t* empty = full + ST;
  volatile uint32_t* failed = reinterpret_cast<volatile uint32_t*>(empty + ST);
  uint32_t* bases = reinterpret_cast<uint32_t*>(smem + ST * XB + 16 * ST + 16);   // [128] base per tile row
  const int tid = threadIdx.x, wgi = tid >> 7;
  const uint32_t rank = cluster_rank<KS>();
  const int tile = blockIdx.x / KS;
  const int rowtile = tile / TT, tt = tile % TT;
  const int c0 = wg_c0(static_cast<int>(rank), chunks, KS), c1 = wg_c0(static_cast<int>(rank) + 1, chunks, KS);
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1u);
      mbar_init(&empty[s], 256u);
    }
    *failed = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (wgi == 0) bases[tid] = rec[2LL * (128LL * rowtile + tid)].x & 0xFFu;   // row bases for the epilogue
  __syncthreads();

  float d[64];
  if (wgi == 0) {
    if (tid == 0) {
      for (int c = c0, i = 0; c < c1; ++c, ++i) {
        const int s = i % ST;
        mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);
        mbar_expect_tx(&full[s], static_cast<uint32_t>(XB));
        tma_load_2d(smem + s * XB, &tx, &full[s], 64 * c, 128 * tt);
      }
    }
  } else {
    const int w = wgi - 1, lane = tid & 31, q = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
    const long long mytile = wgf_tile(rowtile, w, q);
    const int chs = chunks;
    // the thread's two rows (g, g + 8 of its tile): records (base | count << 8, escapes k | bits << 16)
    uint4 rc[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long long row = 16 * mytile + g + 8 * h;
      rc[h][0] = rec[2 * row];
      rc[h][1] = rec[2 * row + 1];
    }
    uint4 wv[D][3];
    const int nch = c1 - c0;
#pragma unroll
    for (int sl = 0; sl < D; ++sl)
      if (sl < nch)
#pragma unroll
        for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c0 + sl, j, lane), wv[sl][j]);
#pragma unroll
    for (int e = 0; e < 64; ++e) d[e] = 0.0f;
    for (int i0 = 0; i0 < nch; i0 += D) {
#pragma unroll
      for (int sl = 0; sl < D; ++sl) {
        const int i = i0 + sl;
        if (i < nch) {
          const int c = c0 + i, s = i % ST;
          uint4 a[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint4& lv = wv[sl][1 + (j >> 1)];
            a[j] = wgf_decode(wword(wv[sl][0], j), wword(lv, 2 * (j & 1)), wword(lv, 2 * (j & 1) + 1));
          }
          if (i + D < nch) {
#pragma unroll
            for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c + D, j, lane), wv[sl][j]);
          }
          mbar_wait(&full[s], static_cast<uint32_t>((i / ST) & 1), failed);
          const uint32_t b0 = smem_u32(smem + s * XB);
          wgmma_fence();
#pragma unroll
          for (int j = 0; j < 4; ++j) wgmma_rs_128(d, a[j], wg_desc(b0 + wg_boff(j)), 1u);
          wgmma_commit();
          wgmma_wait<0>();
          // escapes with k in this chunk: d += x * (w_true - w_dec), scaled by 2^(112 - base); x from the stage
          const unsigned short* xs = reinterpret_cast<const unsigned short*>(smem + s * XB);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t base = rc[h][0].x & 0xFFu, count = (rc[h][0].x >> 8) & 0xFFu;
            const uint32_t esc[7] = {rc[h][0].y, rc[h][0].z, rc[h][0].w, rc[h][1].x, rc[h][1].y, rc[h][1].z, rc[h][1].w};
#pragma unroll
            for (int j = 0; j < 7; ++j) {
              const int kk = static_cast<int>(esc[j] & 0xFFFFu) - 64 * c;
              if (j < count && kk >= 0 && kk < 64) {
                const float wt = ldexpf(__uint_as_float(esc[j] & 0xFFFF0000u), 112 - static_cast<int>(base));
#pragma unroll
                for (int e = 0; e < 64; ++e) {
                  if (((e >> 1) & 1) == h) {
                    const float xk = __uint_as_float(static_cast<uint32_t>(xs[wgf_xoff(wg_tok(t, e), kk) >> 1]) << 16);
                    d[e] = fmaf(-xk, 3.0517578125e-05f, d[e]);   // 2^-15: what the stored escape decodes to
                    d[e] = fmaf(xk, wt, d[e]);
                  }
                }
              }
            }
          }
          mbar_arrive(&empty[s]);
        }
      }
    }
  }

  wg_epilogue<KS, true>(d, smem, bases, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);
}

}  // namespace wg

// cuTensorMapEncodeTiled's signature (cuda.h, CUDA 12.0+), fetched through the runtime so no libcuda link is needed.
typedef CUresult (*wg_encode_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static wg_encode_t wg_encode_fn() {
  static wg_encode_t fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess
                && q == cudaDriverEntryPointSuccess && p != nullptr, "wg: cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<wg_encode_t>(p);
  }
  return fn;
}

// 2D BF16 tensor [rows][k] (row-major), box 64 k x 128 rows, 128B swizzle, zero fill out of bounds.
static CUtensorMap wg_map(const void* ptr, int64_t rows, int64_t k) {
  CUtensorMap map;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(k * 2)};
  cuuint32_t box[2] = {64u, 128u};
  cuuint32_t estr[2] = {1u, 1u};
  const CUresult r = wg_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "wg: cuTensorMapEncodeTiled failed (", static_cast<int>(r), ")");
  return map;
}

@@INSTANCES@@
"""

# (KS, ST, TT) WG instances and (KS, ST, TT, D) WGF instances compiled into the module.
WG_INSTANCES = ()
WGF_INSTANCES = ()
WG_CHECK_INSTANCES = ((1, 4, 1), (4, 4, 2), (8, 3, 1))
WGF_CHECK_INSTANCES = ((1, 4, 1, 2), (4, 4, 2, 1), (8, 5, 1, 4))


def wg_name(inst):
    return "wg_run_" + "_".join(str(int(v)) for v in inst)


def wgf_name(inst):
    return "wgf_run_" + "_".join(str(int(v)) for v in inst)


def wg_smem(inst):
    ks, st, tt = inst
    return st * 32768 + 1024 + 2 * st * 8 + 16


def wgf_smem(inst):
    ks, st, tt, d = inst
    return st * 16384 + 1024 + 2 * st * 8 + 16 + 512


def _launch(kernel, smem, ks, tt, args):
    return f"""  const unsigned ctas = static_cast<unsigned>((n_rows / 128) * {tt} * {ks});
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {ks};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(ctas, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = {smem};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, {kernel}, {args});
  TORCH_CHECK(err == cudaSuccess, "wg launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();"""


def wg_source(instances, wgf_instances=()):
    wrappers = []
    for inst in instances:
        ks, st, tt = inst
        kern = f"wg::wg_kernel<{ks}, {st}, {tt}>"
        wrappers.append(f"""
void {wg_name(inst)}(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(w.scalar_type() == at::ScalarType::BFloat16 && x.scalar_type() == at::ScalarType::BFloat16
              && out.scalar_type() == at::ScalarType::BFloat16, "wg: dtypes");
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.is_contiguous() && w.size(1) == k && w.size(0) >= n_rows, "wg: layouts");
  TORCH_CHECK(n_rows % 128 == 0 && k % 64 == 0 && k / 64 >= {ks} && m >= 1 && m <= 128 * {tt} && out.numel() == m * n_rows, "wg: shape");
  const CUtensorMap tw = wg_map(w.data_ptr(), n_rows, k);
  const CUtensorMap tx = wg_map(x.data_ptr(), m, k);
{_launch(kern, wg_smem(inst), ks, tt, "tw, tx, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_rows), static_cast<int>(k / 64), static_cast<int>(m)")}
}}
void {wg_name(inst)}_attr() {{
  cudaFuncSetAttribute({kern}, cudaFuncAttributeMaxDynamicSharedMemorySize, {wg_smem(inst)});
}}
int64_t {wg_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, {kern});
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    for inst in wgf_instances:
        ks, st, tt, d = inst
        kern = f"wg::wgf_kernel<{ks}, {st}, {tt}, {d}>"
        wrappers.append(f"""
void {wgf_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::BFloat16, "wgf: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && rec.is_contiguous(), "wgf: layouts");
  TORCH_CHECK(n_rows % 128 == 0 && k % 64 == 0 && k / 64 >= {ks} && m >= 1 && m <= 128 * {tt} && out.numel() == m * n_rows, "wgf: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * (k / 64) * 1536 && rec.numel() >= n_rows * 8, "wgf: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "wgf: unaligned");
  const CUtensorMap tx = wg_map(x.data_ptr(), m, k);
{_launch(kern, wgf_smem(inst), ks, tt, "tx, reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_rows), static_cast<int>(k / 64), static_cast<int>(m)")}
}}
void {wgf_name(inst)}_attr() {{
  cudaFuncSetAttribute({kern}, cudaFuncAttributeMaxDynamicSharedMemorySize, {wgf_smem(inst)});
}}
int64_t {wgf_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, {kern});
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return (WG_SRC.replace("@@WGMMA128@@", wgmma_asm(128)).replace("@@WGMMARS128@@", wgmma_rs_asm(128))
            .replace("@@INSTANCES@@", "\n".join(wrappers)))


def wg_decls(instances, wgf_instances=()):
    out = []
    for inst in instances:
        name = wg_name(inst)
        out.append(f"void {name}(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    for inst in wgf_instances:
        name = wgf_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    return out


def wg_build(instances, wgf_instances=()):
    from torch.utils.cpp_extension import load_inline
    decls = wg_decls(instances, wgf_instances)
    return load_inline(
        name="okc_proj_wg", cpp_sources="\n".join(decls), cuda_sources=wg_source(instances, wgf_instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_wg_"), verbose=False)
