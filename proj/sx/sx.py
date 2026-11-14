"""SX: tensor-core GEMM family for m = 32/128/256 (load_inline, sm_90a): out = x @ W.T in BF16 on the four BF16
layers, weights as 12-bit planes (F12 = True, lossless, natural k order) or BF16 fragments (F12 = False).

Work split (compile-time; K = 64 * NS * KSW * CS):
  cluster  CS CTAs split K; fixed-order DSMEM reduction (CS = 1: none).
  CTA      ROWS = 16 * RTW * RGN rows x all MT = 8 * NT8 * TG tokens; warps = RGN row groups x TG token
           groups x KSW K slices; x staged once per step in shared memory by cp.async ([KSW][MT][64] BF16
           per stage, 16-B pieces swizzled j ^ (tok & 7), zero fill past m, XS stages) and read by ldmatrix
           (x4: two n8 tiles), so every warp of a K slice shares it.
  warp     RTW 16-row tiles x NT8 n8 tiles of its token group over NS 64-k chunks of its slice; A fragments
           stream from global into a D-chunk register ring (12-bit: decoded in 7 integer ops per k16 step);
           mma.sync.m16n8k16, FP32 accumulators.
  escapes  (F12) after each chunk's MMAs, from the row records kept in shared memory and the x already in
           the stage: acc += x * (ldexp(w_true, 112 - base) - 2^-15); exact, no tail loads.
  reduce   accumulators to shared memory as [ks][token][row] FP32; after a cluster barrier rank r sums its
           share of tokens over KSW parts and CS peers (16-B loads), scales each row by 2^(base - 112)
           (F12) and stores 8 BF16 per item as one 16-B vector. One kernel, deterministic, no memset.
check_sx.py parses this file's index helpers and decode and emulates the kernel with the PTX layouts.
"""

import tempfile


def sx_wgmma_rs_asm(n=128):
    """One wgmma.m64n{n}k16 with A from four registers (RS form), B from a descriptor, scale-d by a register
    (operand order per PTX ISA 9.4: d, a, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-b)."""
    nd = n // 2
    regs = ", ".join(f"%{i}" for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 5}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nd}, %{nd + 1}, %{nd + 2}, %{nd + 3}}}, '
            f'%{nd + 4}, p, 1, 1, 0;\\n}}"\n'
            f'      : ' + ", ".join(f'"+f"(d[{i}])' for i in range(nd)) +
            f'\n      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(db), "r"(scale_d));')


SX_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace sx {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

// One escape flag byte (zero-extended; PTX ISA: a destination register wider than the type).
__device__ __forceinline__ uint32_t ld_flag(const uint8_t* p) {
  uint32_t v;
  asm("ld.global.nc.u8 %0, [%1];" : "=r"(v) : "l"(p));
  return v;
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int CS>
__device__ __forceinline__ void cluster_sync() {
  if constexpr (CS > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
    __syncthreads();
  }
}

template <int CS>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (CS > 1) {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
  } else {
    return 0u;
  }
}

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

// 12-bit decode of one lane's A fragment for one k16 step (f12m_layout format): four BF16x2 registers
// scaled by 2^(112 - base) per row.
__device__ __forceinline__ uint4 sx_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

// Index helpers (check_sx.py evaluates their return expressions).
__device__ __forceinline__ int sx_wrg(int warp, int tg, int ksw) { return warp / (tg * ksw); }
__device__ __forceinline__ int sx_wtg(int warp, int tg, int ksw) { return (warp / ksw) % tg; }
__device__ __forceinline__ int sx_wks(int warp, int ksw) { return warp % ksw; }
__device__ __forceinline__ int sx_chunk(int r, int ns, int ksw, int ks, int i) { return r * (ns * ksw) + ks * ns + i; }
__device__ __forceinline__ long long sx_tile(long long cl, int rg, int rtw, int rgn, int r) { return (cl * rgn + rg) * rtw + r; }
__device__ __forceinline__ int sx_xoff(int tok, int j) { return tok * 128 + ((j ^ (tok & 7)) << 4); }
__device__ __forceinline__ long long sx_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long sx_bidx(long long tile, int kss, int c, int j, int lane) { return (tile * kss + 4 * c + j) * 32 + lane; }
__device__ __forceinline__ long long sx_fidx(long long tile, int chs, int c) { return tile * chs + c; }
__device__ __forceinline__ int sx_lslot(int j) { return 1 + (j >> 1); }
__device__ __forceinline__ int sx_lword(int j, int h) { return 2 * (j & 1) + h; }
__device__ __forceinline__ int sx_ltok(int tb, int p, int lane) { return tb + 16 * p + (lane & 7) + 8 * ((lane >> 3) >> 1); }
__device__ __forceinline__ int sx_lkch(int j, int lane) { return 2 * j + ((lane >> 3) & 1); }
__device__ __forceinline__ int sx_crow(int rg, int rtw, int r, int g, int q) { return 16 * (rg * rtw + r) + g + 8 * (q >> 1); }
__device__ __forceinline__ int sx_ctok(int tb, int n, int t, int q) { return tb + 8 * n + 2 * t + (q & 1); }
__device__ __forceinline__ int sx_etok(int r, int share, int idx, int groups) { return r * share + idx / groups; }
__device__ __forceinline__ int sx_erow(int idx, int groups) { return 8 * (idx % groups); }

template <int NT8, int TG, int RTW, int RGN, int KSW, int NS, int D, int XS, bool F12, int CS>
__global__ void __launch_bounds__(32 * RGN * TG * KSW)
sx_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint8_t* __restrict__ flags,
          const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out, int m, int n_rows) {
  constexpr int MT = TG * 8 * NT8;                 // tokens covered
  constexpr int ROWS = 16 * RTW * RGN;             // rows per CTA
  constexpr int NTH = 32 * RGN * TG * KSW;
  constexpr int CHS = NS * KSW * CS;               // 64-k chunks per row
  constexpr int K = 64 * CHS, KSS = 4 * CHS;
  constexpr int SL = F12 ? 3 : 4;
  constexpr int STAGE = KSW * MT * 128;            // bytes per x stage
  constexpr bool RES = XS >= NS;                   // resident: every chunk of x staged once, no barrier per chunk
  constexpr int PROW = ROWS + 4;                   // partial-tile row pitch (floats): conflict-free stores
  constexpr int XAREA = (RES ? NS : XS) * STAGE, PAREA = KSW * MT * PROW * 4;
  constexpr int AREA = XAREA > PAREA ? XAREA : PAREA;
  static_assert(NT8 == 1 || NT8 % 2 == 0, "n8 tiles come in pairs (ldmatrix.x4)");
  static_assert(D >= 1 && XS >= 2 && MT % CS == 0 && ROWS % 8 == 0 && AREA % 16 == 0, "sx config");
  extern __shared__ __align__(128) uint8_t smem[];
  uint4* recs = reinterpret_cast<uint4*>(smem + AREA);         // F12: this CTA's row records [ROWS][2]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int rg = sx_wrg(warp, TG, KSW), tgi = sx_wtg(warp, TG, KSW), ks = sx_wks(warp, KSW);
  const uint32_t rank = cluster_rank<CS>();
  const long long cl = blockIdx.x / CS;
  const int tb = tgi * 8 * NT8;
  const uint32_t sbase = smem_u32(smem);

  if constexpr (F12) {
    for (int r = tid; r < 2 * ROWS; r += NTH) recs[r] = rec[2 * cl * ROWS + r];
  }
  auto copy_x = [&](int i, int s) {
    for (int p = tid; p < KSW * MT * 8; p += NTH) {
      const int ksp = p / (MT * 8), rem = p % (MT * 8), tok = rem >> 3, j = rem & 7;
      const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ksp, i);
      const bool ok = tok < m;
      const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;
      cp16(sbase + static_cast<uint32_t>(s * STAGE + ksp * MT * 128 + sx_xoff(tok, j)), src, ok ? 16u : 0u);
    }
  };
  uint4 wv[D][RTW][SL];
  uint32_t fl[D][RTW];
  auto load_w = [&](int sl, int i) {
    const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ks, i);
#pragma unroll
    for (int r = 0; r < RTW; ++r) {
      const long long tile = sx_tile(cl, rg, RTW, RGN, r);
      if constexpr (F12) fl[sl][r] = ld_flag(flags + sx_fidx(tile, CHS, c));
#pragma unroll
      for (int j = 0; j < SL; ++j) {
        if constexpr (F12) {
          ld_w(wp + sx_widx(tile, CHS, c, j, lane), wv[sl][r][j]);
        } else {
          ld_w(wp + sx_bidx(tile, KSS, c, j, lane), wv[sl][r][j]);
        }
      }
    }
  };

#pragma unroll
  for (int sl = 0; sl < D; ++sl)
    if (sl < NS) load_w(sl, sl);
  if constexpr (RES) {                                     // chunk 0 in its own group, then the rest
    copy_x(0, 0);
    cp_commit();
    for (int s = 1; s < NS; ++s) copy_x(s, s);
    cp_commit();
  } else {
#pragma unroll
    for (int s = 0; s < XS - 1; ++s) {
      if (s < NS) copy_x(s, s);
      cp_commit();
    }
  }

  float acc[RTW][NT8][4];
#pragma unroll
  for (int r = 0; r < RTW; ++r)
#pragma unroll
    for (int n = 0; n < NT8; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q) acc[r][n][q] = 0.0f;

  for (int i0 = 0; i0 < NS; i0 += D) {
#pragma unroll
    for (int sl = 0; sl < D; ++sl) {
      const int i = i0 + sl;
      if (i < NS) {
        if constexpr (RES) {
          if (i == 0) {
            cp_wait<1>();
            __syncthreads();                               // chunk 0 landed (and the records)
          } else if (i == 1) {
            cp_wait<0>();
            __syncthreads();                               // every other chunk landed
          }
        } else {
          cp_wait<XS - 2>();
          __syncthreads();                                 // step i landed; every warp done with step i - 1
          if (i + XS - 1 < NS) copy_x(i + XS - 1, (i + XS - 1) % XS);
          cp_commit();
        }
        const uint32_t st = sbase + static_cast<uint32_t>((i % XS) * STAGE + ks * MT * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint4 a[RTW];
#pragma unroll
          for (int r = 0; r < RTW; ++r) {
            if constexpr (F12) {
              const uint4& lv = wv[sl][r][sx_lslot(j)];
              a[r] = sx_decode(word(wv[sl][r][0], j), word(lv, sx_lword(j, 0)), word(lv, sx_lword(j, 1)));
            } else {
              a[r] = wv[sl][r][j];
            }
          }
          if constexpr (NT8 == 1) {
            uint32_t b0, b1;
            ldm_x2(st + static_cast<uint32_t>(sx_xoff(sx_ltok(tb, 0, lane & 15), sx_lkch(j, lane & 15))), b0, b1);
#pragma unroll
            for (int r = 0; r < RTW; ++r) mma(acc[r][0], a[r], b0, b1);
          } else {
#pragma unroll
            for (int p = 0; p < NT8 / 2; ++p) {
              uint32_t b[4];
              ldm_x4(st + static_cast<uint32_t>(sx_xoff(sx_ltok(tb, p, lane), sx_lkch(j, lane))), b[0], b[1], b[2], b[3]);
#pragma unroll
              for (int r = 0; r < RTW; ++r) {
                mma(acc[r][2 * p], a[r], b[0], b[1]);
                mma(acc[r][2 * p + 1], a[r], b[2], b[3]);
              }
            }
          }
        }
        if constexpr (F12) {   // escapes with k in this chunk, x from the stage, scaled domain
          const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ks, i);
          const unsigned short* xs = reinterpret_cast<const unsigned short*>(smem + (i % XS) * STAGE + ks * MT * 128);
          const uint32_t* rw = reinterpret_cast<const uint32_t*>(recs);   // [ROWS][8] record words
#pragma unroll
          for (int r = 0; r < RTW; ++r) {
            if (fl[sl][r] == 0u) continue;                 // warp-uniform: ~6% of (tile, chunk) have escapes
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int lrow = sx_crow(rg, RTW, r, g, 2 * h);
              const uint32_t w0 = rw[8 * lrow];
              const int base = static_cast<int>(w0 & 0xFFu), count = static_cast<int>((w0 >> 8) & 0xFFu);
#pragma unroll 1
              for (int jj = 0; jj < count; ++jj) {         // records hold at most 7 escapes per row
                const uint32_t ew = rw[8 * lrow + 1 + jj];
                const int kk = static_cast<int>(ew & 0xFFFFu) - 64 * c;
                if (kk >= 0 && kk < 64) {
                  const float wt = ldexpf(__uint_as_float(ew & 0xFFFF0000u), 112 - base);
#pragma unroll
                  for (int n = 0; n < NT8; ++n) {
#pragma unroll
                    for (int e = 0; e < 2; ++e) {
                      const int tok = sx_ctok(tb, n, t, 2 * h + e);
                      const float xk = __uint_as_float(static_cast<uint32_t>(xs[(sx_xoff(tok, kk >> 3) >> 1) + (kk & 7)]) << 16);
                      acc[r][n][2 * h + e] = fmaf(-xk, 3.0517578125e-05f, acc[r][n][2 * h + e]);   // 2^-15
                      acc[r][n][2 * h + e] = fmaf(xk, wt, acc[r][n][2 * h + e]);
                    }
                  }
                }
              }
            }
          }
        }
        if (i + D < NS) load_w(sl, i + D);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();                                         // stages idle: the partial tile goes over them

  float* part = reinterpret_cast<float*>(smem);             // [KSW][MT][PROW]
#pragma unroll
  for (int r = 0; r < RTW; ++r)
#pragma unroll
    for (int n = 0; n < NT8; ++n)
#pragma unroll
      for (int q = 0; q < 4; ++q)
        part[(ks * MT + sx_ctok(tb, n, t, q)) * PROW + sx_crow(rg, RTW, r, g, q)] = acc[r][n][q];
  cluster_sync<CS>();
  constexpr int SHARE = MT / CS, GROUPS = ROWS / 8;
  for (int idx = tid; idx < SHARE * GROUPS; idx += NTH) {
    const int tok = sx_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = sx_erow(idx, GROUPS);
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int pr = 0; pr < CS; ++pr) {
#pragma unroll
      for (int kp = 0; kp < KSW; ++kp) {
        const int off = (kp * MT + tok) * PROW + row;
        float4 lo, hi;
        if constexpr (CS > 1) {
          lo = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(pr));
          hi = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off + 4), static_cast<uint32_t>(pr));
        } else {
          lo = *reinterpret_cast<const float4*>(part + off);
          hi = *reinterpret_cast<const float4*>(part + off + 4);
        }
        v[0] += lo.x; v[1] += lo.y; v[2] += lo.z; v[3] += lo.w;
        v[4] += hi.x; v[5] += hi.y; v[6] += hi.z; v[7] += hi.w;
      }
    }
    if constexpr (F12) {
#pragma unroll
      for (int ii = 0; ii < 8; ++ii) v[ii] *= __uint_as_float(((recs[2 * (row + ii)].x & 0xFFu) + 15u) << 23);   // 2^(base - 112)
    }
    if (tok < m) {
      *reinterpret_cast<uint4*>(out + static_cast<long long>(tok) * n_rows + cl * ROWS + row) =
          make_uint4(bf16x2_bits(v[0], v[1]), bf16x2_bits(v[2], v[3]), bf16x2_bits(v[4], v[5]), bf16x2_bits(v[6], v[7]));
    }
  }
  if constexpr (CS > 1) cluster_sync<CS>();                // peers may still read this CTA's tile
}

// ---------------------------------------------------------------------------------------------------------
// SXW: the SX structure with wgmma (m = 128/256). Warpgroups = WGS row groups x KSW K slices; each warp of a
// warpgroup owns one 16-row tile of the warpgroup's 64 rows; 12-bit planes stream into a per-warp shared
// memory ring by cp.async (P chunks ahead, no registers held), are decoded in 7 ops into RS A fragments
// (double-buffered), and wgmma.m64n128k16 reads x from the same swizzled cp.async stages through a K-major
// 128B-swizzle descriptor (the stage layout is that canonical layout). wait_group 1 after each chunk keeps
// one chunk of wgmmas in flight. Escapes are added after the loop from L2-resident x; the reduction,
// scale and stores are SX's.
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

// K-major, 128B-swizzle shared memory descriptor (PTX ISA 9.4 sec. 9.7.17.5.1.2.2): start address >> 4,
// LBO 1 (unused for swizzled K-major), SBO 1024 B >> 4 (next 8 rows), base offset 0 (1024-B aligned), mode 1.
__device__ __forceinline__ uint64_t sxw_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1u) << 16) |
         (static_cast<uint64_t>(1024u >> 4) << 32) | (static_cast<uint64_t>(1u) << 62);
}

__device__ __forceinline__ void wgmma_rs_128(float (&d)[64], const uint4& a, uint64_t db, uint32_t scale_d) {
  @@SXWGMMARS128@@
}

// Index helpers (check_sx.py evaluates their return expressions).
__device__ __forceinline__ int sxw_wrg(int wgi, int ksw) { return wgi / ksw; }
__device__ __forceinline__ int sxw_wks(int wgi, int ksw) { return wgi % ksw; }
__device__ __forceinline__ long long sxw_tile(long long cl, int wrg, int q, int wgs) { return (cl * wgs + wrg) * 4 + q; }
__device__ __forceinline__ int sxw_row(int wrg, int q, int g, int e) { return 64 * wrg + 16 * q + g + 8 * ((e >> 1) & 1); }
__device__ __forceinline__ int sxw_tok(int nb, int t, int e) { return 128 * nb + 8 * (e >> 2) + 2 * t + (e & 1); }
__device__ __forceinline__ int sxw_wring(int warp, int slot, int wsl) { return (warp * wsl + slot) * 1536; }
__device__ __forceinline__ int sxw_boff(int ks, int nb, int j, int n) { return ks * n * 128 + nb * 16384 + 32 * j; }

template <int NB, int WGS, int KSW, int NS, int P, int CS>
__global__ void __launch_bounds__(128 * WGS * KSW)
sxw_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const __nv_bfloat16* __restrict__ x,
           __nv_bfloat16* __restrict__ out, int m, int n_rows) {
  constexpr int N = 128 * NB;                      // tokens covered (one wgmma n128 per 128)
  constexpr int ROWS = 64 * WGS;                   // rows per CTA
  constexpr int NTH = 128 * WGS * KSW;
  constexpr int CHS = NS * KSW * CS;               // 64-k chunks per row
  constexpr int K = 64 * CHS;
  constexpr int XSL = P + 2, WSL = P + 1;          // x slots, weight-ring slots per warp
  constexpr int STAGE = KSW * N * 128;             // bytes per x slot
  constexpr int XAREA = XSL * STAGE, WAREA = (NTH / 32) * WSL * 1536;
  constexpr int PROW = ROWS + 4;
  constexpr int PAREA = KSW * N * PROW * 4;
  constexpr int AREA = (XAREA + WAREA) > PAREA ? (XAREA + WAREA) : PAREA;
  static_assert(NB >= 1 && NB <= 2 && P >= 1 && N % CS == 0 && STAGE % 1024 == 0 && AREA % 16 == 0, "sxw config");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint4* recs = reinterpret_cast<uint4*>(smem + AREA);         // this CTA's row records [ROWS][2]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int wgi = warp >> 2, q = warp & 3;
  const int wrg = sxw_wrg(wgi, KSW), wks = sxw_wks(wgi, KSW);
  const uint32_t rank = cluster_rank<CS>();
  const long long cl = blockIdx.x / CS;
  const long long tile = sxw_tile(cl, wrg, q, WGS);
  const uint32_t sbase = smem_u32(smem);

  for (int r = tid; r < 2 * ROWS; r += NTH) recs[r] = rec[2 * cl * ROWS + r];
  auto copy = [&](int i) {                                 // chunk i: every slice's x, this warp's weights
    const int s = i % XSL;
    for (int p = tid; p < KSW * N * 8; p += NTH) {
      const int ksp = p / (N * 8), rem = p % (N * 8), tok = rem >> 3, j = rem & 7;
      const int c = sx_chunk(static_cast<int>(rank), NS, KSW, ksp, i);
      const bool ok = tok < m;
      const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;
      cp16(sbase + static_cast<uint32_t>(s * STAGE + ksp * N * 128 + sx_xoff(tok, j)), src, ok ? 16u : 0u);
    }
    const int c = sx_chunk(static_cast<int>(rank), NS, KSW, wks, i);
#pragma unroll
    for (int j = 0; j < 3; ++j)
      cp16(sbase + static_cast<uint32_t>(XAREA + sxw_wring(warp, i % WSL, WSL) + 512 * j + 16 * lane),
           wp + sx_widx(tile, CHS, c, j, lane), 16u);
  };

#pragma unroll
  for (int s = 0; s < P; ++s) {
    if (s < NS) copy(s);
    cp_commit();
  }
  float acc[NB][64];
#pragma unroll
  for (int nb = 0; nb < NB; ++nb)
#pragma unroll
    for (int e = 0; e < 64; ++e) acc[nb][e] = 0.0f;
  uint4 a[2][4];

  for (int i0 = 0; i0 < NS; i0 += 2) {
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      const int i = i0 + b;
      if (i < NS) {
        cp_wait<P - 1>();
        __syncthreads();                                   // chunk i landed everywhere; chunk i - 2's wgmmas done
        if (i + P < NS) copy(i + P);
        cp_commit();
        const uint4* wr = reinterpret_cast<const uint4*>(smem + XAREA + sxw_wring(warp, i % WSL, WSL));
        const uint4 w0 = wr[lane], w1 = wr[32 + lane], w2 = wr[64 + lane];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4& lv = j < 2 ? w1 : w2;
          a[b][j] = sx_decode(word(w0, j), word(lv, sx_lword(j, 0)), word(lv, sx_lword(j, 1)));
        }
        const uint32_t xs = sbase + static_cast<uint32_t>((i % XSL) * STAGE);
        wgmma_fence();
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int nb = 0; nb < NB; ++nb) wgmma_rs_128(acc[nb], a[b][j], sxw_desc(xs + sxw_boff(wks, nb, j, N)), 1u);
        wgmma_commit();
        wgmma_wait<1>();                                   // chunk i - 1 done: its A registers are free
      }
    }
  }
  wgmma_wait<0>();
  cp_wait<0>();

  // escapes with k in this warpgroup's chunks: acc += x * (ldexp(w_true, 112 - base) - 2^-15), x from L2
  {
    const uint32_t* rw = reinterpret_cast<const uint32_t*>(recs);
    const int k0 = 64 * sx_chunk(static_cast<int>(rank), NS, KSW, wks, 0);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int lrow = sxw_row(wrg, q, g, 2 * h);
      const uint32_t w0 = rw[8 * lrow];
      const int base = static_cast<int>(w0 & 0xFFu), count = static_cast<int>((w0 >> 8) & 0xFFu);
#pragma unroll 1
      for (int jj = 0; jj < count; ++jj) {
        const uint32_t ew = rw[8 * lrow + 1 + jj];
        const int kk = static_cast<int>(ew & 0xFFFFu);
        if (kk >= k0 && kk < k0 + 64 * NS) {
          const float wt = ldexpf(__uint_as_float(ew & 0xFFFF0000u), 112 - base);
#pragma unroll
          for (int nb = 0; nb < NB; ++nb) {
#pragma unroll
            for (int e = 0; e < 64; ++e) {
              if (((e >> 1) & 1) == h) {
                const int tok = sxw_tok(nb, t, e);
                const float xk = tok < m ? __uint_as_float(static_cast<uint32_t>(__bfloat16_as_ushort(x[static_cast<long long>(tok) * K + kk])) << 16)
                                          : 0.0f;
                acc[nb][e] = fmaf(-xk, 3.0517578125e-05f, acc[nb][e]);   // 2^-15
                acc[nb][e] = fmaf(xk, wt, acc[nb][e]);
              }
            }
          }
        }
      }
    }
  }
  __syncthreads();                                         // stages and rings idle: the partial tile goes over them

  float* part = reinterpret_cast<float*>(smem);             // [KSW][N][PROW]
#pragma unroll
  for (int nb = 0; nb < NB; ++nb)
#pragma unroll
    for (int e = 0; e < 64; ++e)
      part[(wks * N + sxw_tok(nb, t, e)) * PROW + sxw_row(wrg, q, g, e)] = acc[nb][e];
  cluster_sync<CS>();
  constexpr int SHARE = N / CS, GROUPS = ROWS / 8;
  for (int idx = tid; idx < SHARE * GROUPS; idx += NTH) {
    const int tok = sx_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = sx_erow(idx, GROUPS);
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int pr = 0; pr < CS; ++pr) {
#pragma unroll
      for (int kp = 0; kp < KSW; ++kp) {
        const int off = (kp * N + tok) * PROW + row;
        float4 lo, hi;
        if constexpr (CS > 1) {
          lo = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(pr));
          hi = ld_peer_v4(sbase + 4u * static_cast<uint32_t>(off + 4), static_cast<uint32_t>(pr));
        } else {
          lo = *reinterpret_cast<const float4*>(part + off);
          hi = *reinterpret_cast<const float4*>(part + off + 4);
        }
        v[0] += lo.x; v[1] += lo.y; v[2] += lo.z; v[3] += lo.w;
        v[4] += hi.x; v[5] += hi.y; v[6] += hi.z; v[7] += hi.w;
      }
    }
#pragma unroll
    for (int ii = 0; ii < 8; ++ii) v[ii] *= __uint_as_float(((recs[2 * (row + ii)].x & 0xFFu) + 15u) << 23);   // 2^(base - 112)
    if (tok < m) {
      *reinterpret_cast<uint4*>(out + static_cast<long long>(tok) * n_rows + cl * ROWS + row) =
          make_uint4(bf16x2_bits(v[0], v[1]), bf16x2_bits(v[2], v[3]), bf16x2_bits(v[4], v[5]), bf16x2_bits(v[6], v[7]));
    }
  }
  if constexpr (CS > 1) cluster_sync<CS>();                // peers may still read this CTA's tile
}

}  // namespace sx

@@INSTANCES@@
"""

# (NT8, TG, RTW, RGN, KSW, NS, D, XS, F12, CS) instances compiled into the module.
SX_INSTANCES = ()
# (NB, WGS, KSW, NS, P, CS) SXW instances (wgmma, N = 128 NB tokens).
SXW_INSTANCES = ()
SXW_CHECK_INSTANCES = ((1, 2, 1, 32, 4, 4), (2, 1, 2, 8, 3, 2), (1, 1, 1, 16, 1, 8))
SX_CHECK_INSTANCES = ((4, 1, 4, 2, 2, 64, 2, 3, True, 1), (8, 2, 2, 1, 2, 32, 2, 3, True, 2),
                      (4, 1, 2, 4, 4, 8, 2, 8, True, 4), (1, 4, 1, 4, 1, 64, 1, 2, False, 2),
                      (8, 4, 1, 1, 1, 128, 2, 3, False, 1), (16, 1, 1, 2, 4, 4, 2, 4, False, 8))


def sx_name(inst):
    nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = inst
    return f"sx_run_{nt8}_{tg}_{rtw}_{rgn}_{ksw}_{ns}_{d}_{xs}_{int(f12)}_{cs}"


def sx_smem(inst):
    """Dynamic shared memory of an instance, as the kernel lays it out (AREA + records)."""
    nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = inst
    mt, rows = tg * 8 * nt8, 16 * rtw * rgn
    xarea = (ns if xs >= ns else xs) * ksw * mt * 128
    parea = ksw * mt * (rows + 4) * 4
    return max(xarea, parea) + (rows * 32 if f12 else 0)


def sxw_name(inst):
    nb, wgs, ksw, ns, pp, cs = inst
    return f"sxw_run_{nb}_{wgs}_{ksw}_{ns}_{pp}_{cs}"


def sxw_smem(inst):
    """Dynamic shared memory of an SXW instance: 1024-B alignment slack + AREA + records."""
    nb, wgs, ksw, ns, pp, cs = inst
    n, rows, nth = 128 * nb, 64 * wgs, 128 * wgs * ksw
    main = (pp + 2) * ksw * n * 128 + (nth // 32) * (pp + 1) * 1536
    parea = ksw * n * (rows + 4) * 4
    return 1024 + max(main, parea) + rows * 32


def sxw_wrappers(instances):
    out = []
    for inst in instances:
        nb, wgs, ksw, ns, pp, cs = inst
        tmpl = f"{nb}, {wgs}, {ksw}, {ns}, {pp}, {cs}"
        n, rows, k, threads = 128 * nb, 64 * wgs, 64 * ns * ksw * cs, 128 * wgs * ksw
        out.append(f"""
void {sxw_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::BFloat16, "sxw: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && rec.is_contiguous() && x.size(1) == {k}, "sxw: layouts");
  TORCH_CHECK(m >= 1 && m <= {n} && n_rows > 0 && n_rows % {rows} == 0 && out.numel() == m * n_rows, "sxw: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * {(k // 64) * 1536}LL && rec.numel() >= n_rows * 8
              && rec.element_size() == 4, "sxw: weights too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "sxw: unaligned");
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((n_rows / {rows}) * {cs}), 1, 1);
  cfg.blockDim = dim3({threads}, 1, 1);
  cfg.dynamicSmemBytes = {sxw_smem(inst)};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, sx::sxw_kernel<{tmpl}>, reinterpret_cast<const uint4*>(wp.data_ptr()),
      reinterpret_cast<const uint4*>(rec.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(m), static_cast<int>(n_rows));
  TORCH_CHECK(err == cudaSuccess, "sxw launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
void {sxw_name(inst)}_attr() {{
  cudaFuncSetAttribute(sx::sxw_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, {sxw_smem(inst)});
  if ({cs} > 8) cudaFuncSetAttribute(sx::sxw_kernel<{tmpl}>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}}
int64_t {sxw_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, sx::sxw_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return out


def sx_source(instances, w_instances=()):
    wrappers = sxw_wrappers(w_instances)
    for inst in instances:
        nt8, tg, rtw, rgn, ksw, ns, d, xs, f12, cs = inst
        tmpl = f"{nt8}, {tg}, {rtw}, {rgn}, {ksw}, {ns}, {d}, {xs}, {'true' if f12 else 'false'}, {cs}"
        mt, rows, k = tg * 8 * nt8, 16 * rtw * rgn, 64 * ns * ksw * cs
        threads = 32 * rgn * tg * ksw
        tile_bytes = (k // 64) * 1536 if f12 else (k // 16) * 512
        wrappers.append(f"""
void {sx_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor flags, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::BFloat16, "sx: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && x.size(1) == {k}, "sx: layouts");
  TORCH_CHECK(m >= 1 && m <= {mt} && n_rows > 0 && n_rows % {rows} == 0 && out.numel() == m * n_rows, "sx: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * {tile_bytes}LL, "sx: weights too small");
  TORCH_CHECK(!{'true' if f12 else 'false'} || (rec.is_contiguous() && rec.numel() >= n_rows * 8 && rec.element_size() == 4
              && flags.is_contiguous() && flags.element_size() == 1 && flags.numel() >= (n_rows / 16) * {k // 64}), "sx: records");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "sx: unaligned");
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((n_rows / {rows}) * {cs}), 1, 1);
  cfg.blockDim = dim3({threads}, 1, 1);
  cfg.dynamicSmemBytes = {sx_smem(inst)};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, sx::sx_kernel<{tmpl}>, reinterpret_cast<const uint4*>(wp.data_ptr()),
      reinterpret_cast<const uint4*>(rec.data_ptr()), reinterpret_cast<const uint8_t*>(flags.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(m), static_cast<int>(n_rows));
  TORCH_CHECK(err == cudaSuccess, "sx launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
void {sx_name(inst)}_attr() {{
  cudaFuncSetAttribute(sx::sx_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, {sx_smem(inst)});
  if ({cs} > 8) cudaFuncSetAttribute(sx::sx_kernel<{tmpl}>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}}
int64_t {sx_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, sx::sx_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return SX_SRC.replace("@@SXWGMMARS128@@", sx_wgmma_rs_asm(128)).replace("@@INSTANCES@@", "\n".join(wrappers))


def sx_decls(instances, w_instances=()):
    out = []
    for inst in w_instances:
        name = sxw_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    for inst in instances:
        name = sx_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor flags, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    return out


def sx_build(instances, w_instances=()):
    from torch.utils.cpp_extension import load_inline
    decls = sx_decls(instances, w_instances)
    return load_inline(
        name="okc_proj_sx", cpp_sources="\n".join(decls), cuda_sources=sx_source(instances, w_instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_sx_"), verbose=False)
