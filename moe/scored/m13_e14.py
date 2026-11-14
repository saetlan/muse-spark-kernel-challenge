"""M13: stream-K expert GEMMs for small batches (appended after the M11 section, whose prepare and
kernel_fn become _m11_prepare and _m11_kernel_fn).

What changes against M11 (MF7: M11 streamed only ~2 TB/s at bs1):
- Even split of the work over every resident warp (stream-K): the (active expert, task, pair pass,
  K group) list is cut into equal contiguous ranges, one per warp, whatever the routing; no waves.
- One register pipeline per warp that runs straight across task boundaries (the next group's loads are
  in flight while the current one is computed), instead of restarting at every task.
- A task whose K is split between warps adds its partial sums into an FP32 scratch with atomics; the
  warp that completes its K (a per-task counter of K groups) applies the epilogue from the scratch and
  clears scratch and counter, so the buffers are clean again at the end of every call.
- Pair lists built in pair order (deterministic), so every CTA agrees on which pairs a pass holds.
- A lossless bit re-layout of each code word: the 4 bf16x2 A registers are rotr(w, 2|6|12) & M and
  (rotr(w, 8) & M') | (w & M''), 9 instructions instead of ~15 for Marlin's shifts.
- E4M3 scales (all normal and positive, checked in prepare) become bf16x2 (s * 2^64, s * 2^64) with one
  PRMT and one IMAD each.
- GEMM2 and a CUDA post-norm are launched with programmatic dependent launch (PDL): they start as
  GEMM1/GEMM2 CTAs retire, scan the routing and load their first weights before griddepcontrol.wait.
"""

# Tokens M13 takes at most (pairs per expert <= tokens, so MAXP below needs no cap at this size).
M13_MAX_TOKENS = 64
# (gated, tt, nt) instances: tt = 16-row tiles per task (gated: gate + up), nt = 8-pair column tiles.
M13_INSTANCES = ((1, 2, 1), (1, 2, 2), (0, 1, 1), (0, 2, 1), (0, 1, 2), (0, 2, 2))
# M14 (CTA-shared activations, stream-K over CTAs) takes up to M14_MAX_TOKENS (= its MAXP); (gated, tt, nt).
M14_MAX_TOKENS = 320
# (gated, tt, nt, P): P = activation chunks staged ahead (P + 1 shared-memory stages); E9 ran P = 2 only.
# n8 dropped (E6/E8/E9: always the slowest, 237-254 registers).
M14_INSTANCES = ((1, 2, 2, 2), (1, 2, 2, 4), (1, 2, 2, 6), (1, 2, 4, 2), (1, 2, 4, 4),
                 (0, 2, 2, 2), (0, 2, 2, 4), (0, 2, 2, 6), (0, 2, 4, 4), (0, 1, 4, 4))
# M15 (weights through a per-warp shared-memory TMA ring, one pass up to 8 nt pairs): (gated, tt, nt, ring slots)
# (gated, tt, nt, P): RS-wgmma, N = 8 nt pairs per pass at most (128, 64 or 32), P = activation lead; p1.16 takes
# bs256's hottest expert (~128 pairs) in one pass.
# v2 (E13): (gated, tt, nt, P, DB); DB = double-buffered A + wgmma.wait_group 1 (E12's v1 waited for every chunk's
# MMAs and was > 1.25x M14's time). DB costs 16-32 registers, so g2.8 / p2.8 stay single-buffered.
M16_INSTANCES = ((1, 2, 8, 4, 0), (1, 2, 4, 4, 0), (1, 2, 4, 4, 1), (0, 2, 8, 4, 0), (0, 2, 4, 4, 1), (0, 1, 8, 4, 1),
                 (0, 1, 16, 4, 0), (0, 1, 16, 4, 1))
# (gated, tt, nt, R, P): E9's ran P = 2 (R 3/4/4/4/3) and lost to M14 and M13; these stage activations 3-4 chunks
# ahead (MF11), with R cut where needed to keep 2 CTAs per SM.
# (gated, tt, nt, R, P): M15's ring and cost split with M16's wgmma (natural-K bank); sized like M15's for 2 CTAs/SM.
# The NT 1/2 ones (R 2: ~63 KB of shared memory, 3 CTAs/SM) are for 1-32 tokens: deep async weight and activation rings
# with small accumulators, to test whether small batches are latency-bound (MF10/MF11).
# v2 (E13): (gated, tt, nt, R, P, DB), shared memory sized for 2 CTAs/SM (3 for the NT 1 ones).
M17_INSTANCES = ((1, 2, 8, 2, 4, 0), (1, 2, 4, 3, 4, 1), (0, 1, 16, 2, 2, 1), (0, 1, 8, 4, 4, 1), (0, 2, 8, 2, 4, 1),
                 (1, 2, 1, 2, 4, 1), (0, 2, 1, 2, 4, 1))
M15_INSTANCES = ((1, 2, 8, 2, 4), (1, 2, 4, 3, 4), (0, 1, 16, 2, 3), (0, 1, 8, 4, 4), (0, 2, 8, 2, 4),
                 (1, 2, 1, 2, 4), (0, 2, 1, 2, 4), (1, 2, 2, 2, 4), (0, 2, 2, 2, 4))
# The M11 instances M11_CONFIGS uses (E3c/E4 picks); compiled next to M13 as the fallback.
M13_M11_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (1, 2, 2, 2, 4), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4), (0, 2, 2, 2, 4))
# Per-call configs by token count, like M11_CONFIGS; a stage dict with kind="m13" runs the stream-K
# kernel (s1 = dict(kind, nt, c), s2 = dict(kind, tt, nt, c, pdl, pf)), kind="m14" the CTA-shared one
# (same keys); post = dict(kind="m13", pdl) or None for the Triton post-norm. None for a class means the
# M11/v1 path (_m11_kernel_fn). kind="m13" stages only in classes up to M13_MAX_TOKENS tokens.
M13_CONFIGS = (
    (5, None),
    (9, None),
    (17, None),
    (33, None),
    (65, None),
    (192, None),
    (321, None),
)

# New bit n of an M13 code word = bit M13_BIT_SRC[n] of the M11 word (built below).
_M13_TARGETS = (6, 7, 8, 15, 22, 23, 24, 31)  # bf16x2 sign + 3 magnitude bits per lane


def _m13_bit_sources():
    # M11's dec<R>: target bit p of A register R comes from word bit p - 4R (p = 15, 31) or
    # p + 6 - 4R (p = 6..8, 22..24).
    src = [None] * 32
    for r in range(4):
        for p in _M13_TARGETS:
            old = (p if p in (15, 31) else p + 6) - 4 * r
            if r < 3:
                new = (p + (2, 6, 12)[r]) % 32
            else:
                new = p if p in (6, 22) else (p + 8) % 32
            assert src[new] is None
            src[new] = old
    assert sorted(src) == list(range(32))
    return tuple(src)


M13_BIT_SRC = _m13_bit_sources()

_M13_CUDA = r"""
namespace m13 {
using m11::hmul2;
using m11::mma;
using m11::word;

constexpr int K = 4096;
constexpr int KG = K / 64;   // 64-wide K chunks
constexpr int D = 4;         // chunks per group = register ring depth; ranges are cut at groups
constexpr int GPT = KG / D;  // groups per (task, pass)
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M13_MAXP@@;          // pairs per expert at most
constexpr int MAXPASS = (MAXP + 7) / 8;     // pair passes per task at most (nt >= 1)
constexpr int MAXPAIRS = @@M13_MAXPAIRS@@;  // n_pairs at most
constexpr uint32_t PM = 0x81C081C0u;    // sign + magnitude bits of both bf16 lanes
constexpr uint32_t PM3 = 0x81808180u;   // a3: these bits from rotr(w, 8) ...
constexpr uint32_t PM3W = 0x00400040u;  // ... and these straight from w

__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }

// Byte B of v (an E4M3 scale, normal and positive) -> bf16x2 (s * 2^64, s * 2^64): bf16 bits are
// (b << 4) + (184 << 7) for b = E << 3 | M; exact.
template <int B>
__device__ __forceinline__ uint32_t scale2(uint32_t v) {
  return __byte_perm(v, 0u, 0x4440u | B) * 0x00100010u + 0x5C005C00u;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void l2_prefetch(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// Loads of data the previous grid wrote. E5: with __ldg (ld.global.nc, treated as invariant) they could
// be hoisted above griddepcontrol.wait and every PDL config produced garbage; E8: volatile asm loads
// with a memory clobber fixed that but pinned every load in place (~10% slower). Now the wait returns a
// zero token that the callers add to those pointers: the loads depend on the wait through data, so they
// can't move above it, and are otherwise free to be scheduled (non-volatile ld.global.cg, coherent at L2).
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

// A position in a warp's range: active expert ai (expert e), task, pair pass, K group; passes and
// the valid pairs of this pass (cntp) follow from e.
struct Cur {
  int ai, e, task, pass, grp, passes, cntp;
};

// w: [LOCAL][rt_total][KG][32 lanes] uint4 (M13 bit layout); sf: [LOCAL][rt_total][GPT][32] uint2 =
// 4 chunks of (row g, row g + 8) E4M3 scales; scr: FP32 [pairs][16 rt_total], zero between calls;
// ctr: [LOCAL][per_expert][MAXPASS] K groups done, zero between calls.
// MODE (recon only when not 0): 1 = the same loads, cursor and flushes, loaded words only XORed together;
// 2 = loads + decode, no MMA (decoded words XORed); 3 = loads + MMA on raw words, no decode. Recon modes
// write garbage to out but keep scratch and counters clean (the flushes are unchanged).
// CG (GEMM2 launched with PDL only): activations are loads ordered after griddepcontrol.wait (ld.global.cg +
// the wait's token); otherwise __ldg, which caches them in L1 (every warp of an expert re-reads the same
// rows; E8/E9 lost ~10 us at bs1 when GEMM1 read them through L2 only).
template <int TT, int NT, bool GATED, int MODE = 0, bool CG = false>
__global__ void __launch_bounds__(128) sk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf, const char* __restrict__ nxw, const char* __restrict__ nxs,
    long long nxwb, long long nxsb, int nxwarps) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(!(GATED && CG), "GEMM1 is never launched with PDL");
  constexpr int NB = 8 * NT;  // pairs per pass
  __shared__ int s_e[MAXPAIRS];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_base[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid == 0) pdl_launch();
  if (tid < LOCAL) s_cnt[tid] = 0;
  for (int p = tid; p < n_pairs; p += blockDim.x) s_e[p] = ids[p] - local_start;
  __syncthreads();
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  if (warp == 0) {
    // Pair lists in increasing pair order: 32 pairs at a time, ranks from __match_any_sync.
    for (int base = 0; base < n_pairs; base += 32) {
      const int p = base + lane;
      const int e = p < n_pairs ? s_e[p] : -1;
      const bool valid = e >= 0 && e < LOCAL;
      const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
      const unsigned below = m & ((1u << lane) - 1u);
      const int cur = valid ? s_cnt[e] : 0;
      if (valid) {
        const int pos = cur + __popc(below);
        if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
      }
      __syncwarp();
      if (valid && below == 0u) s_cnt[e] = cur + __popc(m);
      __syncwarp();
    }
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int v = act ? per_expert * ((cnt + NB - 1) / NB) * GPT : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (act) {
      const int ai = __popc(am & ((1u << lane) - 1u));
      s_active[ai] = lane;
      s_base[ai] = incl - v;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_base[__popc(am)] = incl;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total = s_base[nact];
  const int gw = blockIdx.x * 4 + warp, nw = gridDim.x * 4;
  const int g0 = static_cast<int>(static_cast<long long>(total) * gw / nw);
  const int g1 = static_cast<int>(static_cast<long long>(total) * (gw + 1) / nw);
  if (g0 >= g1) {  // no work; still wait, so this grid never completes before the previous one
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.pass == c.passes) {
        c.pass = 0;
        if (++c.task == per_expert) {
          c.task = 0;
          if (++c.ai < nact) set_expert(c);
        }
      }
      if (c.ai < nact) set_cntp(c);
    }
  };
  // The cursor of flat group index gi (s_base[0] <= gi < total).
  auto locate = [&](int gi) {
    Cur c;
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= gi) ++ai;
    c.ai = ai;
    set_expert(c);
    const int local = gi - s_base[ai], per_task = c.passes * GPT;
    c.task = local / per_task;
    const int rem = local - c.task * per_task;
    c.pass = rem / GPT;
    c.grp = rem - c.pass * GPT;
    set_cntp(c);
    return c;
  };
  Cur L = locate(g0);  // the group whose loads are issued next

  auto tile_row = [&](const Cur& c, int i) -> size_t {
    const int rt = GATED ? c.task + i * (rt_total / 2) : c.task * TT + i;
    return static_cast<size_t>(c.e) * rt_total + rt;
  };
  const uint4* lw[TT];
  const uint2* ls[TT];
  const uint4* la[NT];
  bool lok[NT];
  auto point = [&](const Cur& c) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(c, i);
      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;
      ls[i] = sf + (row * GPT + c.grp) * 32 + lane;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      lok[n] = idx < c.cntp;
      const int p = lok[n] ? s_list[c.e][c.pass * NB + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      la[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t) + c.grp * D * 8;
    }
  };

  uint32_t tok = 0u;  // zero; after the PDL wait it carries the dependency on the wait (see pdl_wait_token)
  uint32_t sink = 0u;  // recon modes only
  uint4 rw[D][TT];
  uint4 ra[D][NT][2];
  uint2 rsn[TT];
  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) rw[s][i] = __ldcs(lw[i] + s * 32);
  };
  auto load_s = [&]() {
#pragma unroll
    for (int i = 0; i < TT; ++i) rsn[i] = __ldcs(ls[i]);
  };
  auto load_a = [&](int s) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (lok[n]) {
        if constexpr (CG) {
          ra[s][n][0] = ld_cg4(la[n] + s * 8 + tok);
          ra[s][n][1] = ld_cg4(la[n] + s * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(la[n] + s * 8);
          ra[s][n][1] = __ldg(la[n] + s * 8 + 1);
        }
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        uint32_t a0, a1, a2, a3;
        if constexpr (MODE == 3) {  // recon: MMA on raw words
          a0 = q;
          a1 = q ^ sg;
          a2 = q;
          a3 = q ^ sg8;
        } else {
          a0 = hmul2(rotr(q, 2) & PM, sg);
          a1 = hmul2(rotr(q, 6) & PM, sg8);
          a2 = hmul2(rotr(q, 12) & PM, sg);
          a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (n * 8 < cntp) {
            const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
            if constexpr (MODE == 2) {  // recon: decode, no MMA
              sink ^= a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1;
            } else {
              mma(acc[i][n], a0, a1, a2, a3, b0, b1);
            }
          }
        }
      }
    }
  };
  // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1 of each 8-pair tile.
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * c.task + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * (c.task * TT + i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  // End of this warp's part of (task, pass): the whole K -> epilogue; else add into the scratch and
  // count K groups; the warp that brings the count to GPT takes the sums, clears them, and finishes.
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i) {
              const int col = 16 * static_cast<int>(tile_row(c, i) - static_cast<size_t>(c.e) * rt_total);
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + col + g + 8 * rr, acc[i][n][2 * rr + h]);
            }
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + c.task) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i) {
                const int col = 16 * static_cast<int>(tile_row(c, i) - static_cast<size_t>(c.e) * rt_total);
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* q = prow + col + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
                }
              }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  load_s();
  // Start-up L2 prefetches (weights only, so they may run before the PDL wait). pf bits 0-7: the next
  // npre groups of this warp's own range (codes and scales), one group per lane.
  {
    const int npre = pf & 0xFF;
    for (int k = 1 + lane; k <= npre && g0 + k < g1; k += 32) {
      const Cur q = locate(g0 + k);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(q, i);
        l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
        l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
      }
    }
  }
  // pf bits 24-26 = q: the next kernel's weights. Its warps j < nxwarps split the active experts' code bytes
  // (nxw, nxwb per expert, experts in s_active order) and scale bytes (nxs, nxsb) into equal ranges; this
  // grid prefetches the first q/4 of every range (one range per lane), so the next kernel's first loads hit
  // L2. Each piece stays inside one expert's region; sizes and offsets are multiples of 16.
  {
    const int nxq = (pf >> 24) & 7;
    if (nxq > 0 && nxwarps > 0) {
      auto comb = [&](const char* base, long long per, int j) {
        const long long total = static_cast<long long>(nact) * per;
        long long b = (total * j / nxwarps) & ~15LL;
        const long long b1 = total * (j + 1) / nxwarps;
        long long e = b + ((b1 - b) * nxq) / 4;
        e = min((e + 15) & ~15LL, total);
        while (b < e) {
          const int ai = static_cast<int>(b / per);
          const long long off = b - static_cast<long long>(ai) * per;
          const long long len = min(min(e - b, per - off), 32768LL);
          l2_prefetch(base + static_cast<long long>(s_active[ai]) * per + off, static_cast<uint32_t>(len));
          b += len;
        }
      };
      for (int j = gw + nw * lane; j < nxwarps; j += nw * 32) {
        comb(nxw, nxwb, j);
        comb(nxs, nxsb, j);
      }
    }
  }
  if (pdl) tok = pdl_wait_token();
#pragma unroll
  for (int s = 0; s < D; ++s) load_a(s);

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur P = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(P);
      ++pg;
      prefetch_group(P);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(P);
      ++pg;
      prefetch_group(P);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (MODE != 1) {
        compute(s, rsc, cntp);
      } else {
#pragma unroll
        for (int i = 0; i < TT; ++i) sink ^= rw[s][i].x ^ rw[s][i].y ^ rw[s][i].z ^ rw[s][i].w ^ rsc[i].x ^ rsc[i].y;
#pragma unroll
        for (int n = 0; n < NT; ++n) sink ^= ra[s][n][0].x ^ ra[s][n][1].w;
      }
      if (more) {
        load_w(s);
        load_a(s);
        if (s == 0) load_s();
      }
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the loads
  }
}

// out[t] = sum over local slots k of tw[t, k] * rmsnorm(y[t, k] * nw), as SGLang's fused kernel; y and nw
// bf16, H = 4096 = 512 threads x 8.
__global__ void __launch_bounds__(512) post_norm_kernel(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw,
    const int* __restrict__ ids, __nv_bfloat16* __restrict__ out, int local_start, float eps, int pdl) {
  __shared__ float s_part[16][TOP_K];
  __shared__ float s_rs[TOP_K];
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + tid);
  float wv[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t u = word(wraw, j);
    wv[2 * j] = __uint_as_float(u << 16);
    wv[2 * j + 1] = __uint_as_float(u & 0xFFFF0000u);
  }
  bool loc[TOP_K];
  float gk[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tok * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    gk[k] = tw[tok * TOP_K + k];
  }
  const uint32_t ptok = pdl ? pdl_wait_token() : 0u;
  float v[TOP_K][8];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    ss[k] = 0.0f;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (loc[k]) r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tok * TOP_K + k) * K) + tid + ptok);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t u = word(r, j);
      v[k][2 * j] = __uint_as_float(u << 16) * wv[2 * j];
      v[k][2 * j + 1] = __uint_as_float(u & 0xFFFF0000u) * wv[2 * j + 1];
      ss[k] += v[k][2 * j] * v[k][2 * j] + v[k][2 * j + 1] * v[k][2 * j + 1];
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_part[warp][k] = ss[k];
  }
  __syncthreads();
  if (tid < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += s_part[i][tid];
    s_rs[tid] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  __syncthreads();
  float o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (loc[k]) {
      const float f = gk[k] * s_rs[k];
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] += f * v[k][j];
    }
  }
  uint4 res;
  uint32_t pk[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
    const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
    pk[j] = lo | (hi << 16);
  }
  res = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  reinterpret_cast<uint4*>(out + static_cast<size_t>(tok) * K)[tid] = res;
}

// Recon: stream the active experts' contiguous code regions (len16 uint4 each, region e at base +
// e * len16) with U independent 16-B loads per lane in flight; nothing but an XOR per load.
template <int U>
__global__ void __launch_bounds__(128) stream_kernel(const uint4* __restrict__ base, const int* __restrict__ experts,
                                                     int nact, long long len16, unsigned* __restrict__ sink_out) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5), nwarp = gridDim.x * 4;
  const int idx = gw % nact, parts = (nwarp + nact - 1) / nact, part = gw / nact;
  const long long b0 = len16 * part / parts, b1 = len16 * (part + 1) / parts;
  const uint4* src = base + static_cast<long long>(experts[idx]) * len16;
  uint32_t x = 0u;
  for (long long i = b0 + lane; i < b1; i += 32 * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = i + 32 * u < b1 ? __ldcs(src + i + 32 * u) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; ++u) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (x == 0x9E3779B9u) sink_out[0] = x;
}

}  // namespace m13

// ------------------------------------------------------------------------------------------------
// M14: M13 for large batches. A CTA's 4 warps work on the same (expert, block of 4 tasks, pair pass,
// K group) at once, so each 64-wide K chunk of the pass's activations (up to 8 NT pairs) is staged
// once in shared memory with cp.async (3 stages, 16-B segments XOR-swizzled by pair so the B-fragment
// loads are conflict-free) and shared by the 4 warps; weights per warp as in M13 (register ring, M13
// bit layout). Stream-K over CTAs; partial K through M13's scratch and counters.
namespace m14 {
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::l2_prefetch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;       // pairs per expert at most
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes
constexpr int SB = 3;                    // activation stages of M15 (M14: P + 1)

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

// P: activation chunks staged ahead of the one being computed (P + 1 stages; E9's kernel was P = 2).
// MODE (recon only when not 0), as M13's: 1 loads + staging only, 2 no MMA, 3 no decode.
template <int TT, int NT, bool GATED, int P, int MODE = 0>
__global__ void __launch_bounds__(128) ck_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= 8, "activation lead");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 1;  // activation stages
  extern __shared__ __align__(128) unsigned char s_b[];  // [SBP][NB pairs][128 B]
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_base[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  __syncthreads();
  // Pair lists in pair order with 4 warps: per-warp-block counts, offsets per expert, then placement.
  const int q = ((n_pairs + 127) / 128) * 32;
  const int pb0 = warp * q, pb1 = min(n_pairs, pb0 + q);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] += __popc(m);
    __syncwarp();
  }
  __syncthreads();
  if (tid < LOCAL) {
    int run = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = s_wcnt[v][tid];
      s_wcnt[v][tid] = run;
      run += c;
    }
    s_cnt[tid] = run;
  }
  __syncthreads();
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    const int cur = valid ? s_wcnt[warp][e] : 0;
    if (valid) {
      const int pos = cur + __popc(m & lt);
      if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
    }
    __syncwarp();
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] = cur + __popc(m);
    __syncwarp();
  }
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;  // tasks per expert
  const int n_rb = per_expert / 4;                                // 4-task blocks per expert
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int v = act ? n_rb * ((cnt + NB - 1) / NB) * GPT : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_base[ai] = incl - v;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_base[__popc(am)] = incl;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total = s_base[nact];
  const int g0 = static_cast<int>(static_cast<long long>(total) * blockIdx.x / gridDim.x);
  const int g1 = static_cast<int>(static_cast<long long>(total) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {  // the whole CTA has no work; still wait, so this grid never completes before the previous one
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // c.task is the 4-task block here
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.pass == c.passes) {
        c.pass = 0;
        if (++c.task == n_rb) {
          c.task = 0;
          if (++c.ai < nact) set_expert(c);
        }
      }
      if (c.ai < nact) set_cntp(c);
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_base[ai], per_rb = L.passes * GPT;
    L.task = local / per_rb;
    const int rem = local - L.task * per_rb;
    L.pass = rem / GPT;
    L.grp = rem - L.pass * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint4* lw[TT];
  const uint2* ls[TT];
  auto point = [&](const Cur& c) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(c, i);
      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;
      ls[i] = sf + (row * GPT + c.grp) * 32 + lane;
    }
  };
  const uint32_t sb0 = smem_addr(s_b);
  // Chunk s of group c's activations -> stage slot: NB pairs x 8 segments of 16 B, invalid pairs zero-filled.
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {  // only the 8-pair tiles compute() reads
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  uint4 rw[D][TT];
  uint2 rsn[TT];
  float acc[TT][NT][4];
  uint32_t sink = 0u;  // recon modes only
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) rw[s][i] = __ldcs(lw[i] + s * 32);
  };
  auto load_s = [&]() {
#pragma unroll
    for (int i = 0; i < TT; ++i) rsn[i] = __ldcs(ls[i]);
  };
  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp, int slot) {
    uint32_t A[TT][4][4];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        if constexpr (MODE == 1 || MODE == 3) {  // recon: raw words
          A[i][j][0] = q;
          A[i][j][1] = q ^ sg;
          A[i][j][2] = q;
          A[i][j][3] = q ^ sg8;
        } else {
          A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
          A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
          A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
          A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
      }
    }
    const unsigned char* sb = s_b + slot * STAGE;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (n * 8 < cntp) {
        const unsigned char* st = sb + (n * 8 + g) * 128;
        const uint4 blo = *reinterpret_cast<const uint4*>(st + (((2 * t) ^ g) << 4));
        const uint4 bhi = *reinterpret_cast<const uint4*>(st + (((2 * t + 1) ^ g) << 4));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4& bv = j < 2 ? blo : bhi;
          const uint32_t b0 = word(bv, (2 * j) & 3), b1 = word(bv, (2 * j + 1) & 3);
#pragma unroll
          for (int i = 0; i < TT; ++i) {
            if constexpr (MODE == 1 || MODE == 2) {  // recon: no MMA
              sink ^= A[i][j][0] ^ A[i][j][1] ^ A[i][j][2] ^ A[i][j][3] ^ b0 ^ b1;
            } else {
              mma(acc[i][n], A[i][j][0], A[i][j][1], A[i][j][2], A[i][j][3], b0, b1);
            }
          }
        }
      }
    }
  };
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * wtask(c) + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * tile_rt(c, i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + 16 * tile_rt(c, i) + g + 8 * rr, acc[i][n][2 * rr + h]);
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* q = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
                }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  load_s();
  if (pdl) {
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait<P - 1>();
      __syncthreads();  // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      issue_b();        // chunk + P -> the stage the previous chunk used
      compute(s, rsc, cntp, cslot);
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m14

// ------------------------------------------------------------------------------------------------
// M16: M14's CTA (one warpgroup), units, scan, activation ring and stream-K flush, with the MMAs done by
// RS-wgmma (PTX ISA 9.4 wgmma.mma_async m64nNk16 bf16, A = the 4 warps' decoded 16-row tiles in registers,
// B = the activation stage through a K-major 128B-swizzle matrix descriptor, N = this pass's pairs rounded
// up to 8). Weights come from M16's natural-K bank (m16_shuffle; B is read in natural K order); scales are
// one uint2 per (tile, chunk, row g) with byte j = 16-K block j, so every k16 step has its own scale.
namespace m16 {
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::l2_prefetch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;       // pairs per expert at most
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
// This thread's cp.async (generic proxy) writes of a stage -> ordered before wgmma's async-proxy reads.
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
#ifndef M13_NO_WGMMA  // the fallback build (ptxas rejected the wgmma code) compiles none of it
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
#else
__device__ __forceinline__ void wg_wait1() {}
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
__device__ __forceinline__ void wg_wait0() {}
#endif
// Byte b of v (an E4M3 scale, normal and positive) -> bf16x2 (s * 2^64, s * 2^64), as m13::scale2.
__device__ __forceinline__ uint32_t scale2r(uint32_t v, int b) {
  return __byte_perm(v, 0u, 0x4440u | b) * 0x00100010u + 0x5C005C00u;
}
// K-major, 128B-swizzle matrix descriptor of the k16 slice j of a 1024-B-aligned stage (NB rows of 128 B):
// start address >> 4, LBO 1 (unused for swizzled K-major), SBO 1024 B >> 4 (8-row groups), swizzle 1 = 128B.
__device__ __forceinline__ uint64_t b_desc(uint32_t addr, int j) {
  return (static_cast<uint64_t>(((addr + 32u * j) & 0x3FFFFu) >> 4)) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_n8(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n16(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n24(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n32(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n40(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %25, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n48(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n56(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %33, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n64(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n72(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n80(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, {%40, %41, %42, %43}, %44, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n88(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %49, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, {%44, %45, %46, %47}, %48, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n96(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n104(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %57, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, {%52, %53, %54, %55}, %56, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n112(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%56, %57, %58, %59}, %60, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n120(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %65, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, {%60, %61, %62, %63}, %64, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n128(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
// D (+)= A B for the first 8 nv pair columns; acc is [NT][4] (column block, mma.sync C order = wgmma's).
template <int NT>
__device__ __forceinline__ void wg_mma(int nv, float (&acc)[NT][4], const uint32_t (&a)[4], uint64_t desc) {
  switch (nv) {
    case 1: if constexpr (NT >= 1) wg_n8(&acc[0][0], a, desc); break;
    case 2: if constexpr (NT >= 2) wg_n16(&acc[0][0], a, desc); break;
    case 3: if constexpr (NT >= 3) wg_n24(&acc[0][0], a, desc); break;
    case 4: if constexpr (NT >= 4) wg_n32(&acc[0][0], a, desc); break;
    case 5: if constexpr (NT >= 5) wg_n40(&acc[0][0], a, desc); break;
    case 6: if constexpr (NT >= 6) wg_n48(&acc[0][0], a, desc); break;
    case 7: if constexpr (NT >= 7) wg_n56(&acc[0][0], a, desc); break;
    case 8: if constexpr (NT >= 8) wg_n64(&acc[0][0], a, desc); break;
    case 9: if constexpr (NT >= 9) wg_n72(&acc[0][0], a, desc); break;
    case 10: if constexpr (NT >= 10) wg_n80(&acc[0][0], a, desc); break;
    case 11: if constexpr (NT >= 11) wg_n88(&acc[0][0], a, desc); break;
    case 12: if constexpr (NT >= 12) wg_n96(&acc[0][0], a, desc); break;
    case 13: if constexpr (NT >= 13) wg_n104(&acc[0][0], a, desc); break;
    case 14: if constexpr (NT >= 14) wg_n112(&acc[0][0], a, desc); break;
    case 15: if constexpr (NT >= 15) wg_n120(&acc[0][0], a, desc); break;
    case 16: if constexpr (NT >= 16) wg_n128(&acc[0][0], a, desc); break;
    default: break;
  }
}

// N as a template argument -> the wrapper above.
template <int N>
__device__ __forceinline__ void wg_n(float* d, const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 8) wg_n8(d, a, desc);
  else if constexpr (N == 16) wg_n16(d, a, desc);
  else if constexpr (N == 24) wg_n24(d, a, desc);
  else if constexpr (N == 32) wg_n32(d, a, desc);
  else if constexpr (N == 40) wg_n40(d, a, desc);
  else if constexpr (N == 48) wg_n48(d, a, desc);
  else if constexpr (N == 56) wg_n56(d, a, desc);
  else if constexpr (N == 64) wg_n64(d, a, desc);
  else if constexpr (N == 72) wg_n72(d, a, desc);
  else if constexpr (N == 80) wg_n80(d, a, desc);
  else if constexpr (N == 88) wg_n88(d, a, desc);
  else if constexpr (N == 96) wg_n96(d, a, desc);
  else if constexpr (N == 104) wg_n104(d, a, desc);
  else if constexpr (N == 112) wg_n112(d, a, desc);
  else if constexpr (N == 120) wg_n120(d, a, desc);
  else if constexpr (N == 128) wg_n128(d, a, desc);
}
// One chunk's wgmmas at N (4 k16 steps x TT tiles), with no branch between them.
template <int N, int TT, int NT>
__device__ __forceinline__ void wg_steps(float (&acc)[TT][NT][4], const uint32_t (&A)[TT][4][4], uint32_t sbase) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint64_t desc = b_desc(sbase, j);
#pragma unroll
    for (int i = 0; i < TT; ++i) wg_n<N>(&acc[i][0][0], A[i][j], desc);
  }
}
// A chunk at N = 8 nv: one CTA-uniform switch per chunk (v1 switched before every wgmma).
template <int TT, int NT>
__device__ __forceinline__ void wg_chunk(int nv, float (&acc)[TT][NT][4], const uint32_t (&A)[TT][4][4], uint32_t sbase) {
  switch (nv) {
    case 1: if constexpr (NT >= 1) wg_steps<8>(acc, A, sbase); break;
    case 2: if constexpr (NT >= 2) wg_steps<16>(acc, A, sbase); break;
    case 3: if constexpr (NT >= 3) wg_steps<24>(acc, A, sbase); break;
    case 4: if constexpr (NT >= 4) wg_steps<32>(acc, A, sbase); break;
    case 5: if constexpr (NT >= 5) wg_steps<40>(acc, A, sbase); break;
    case 6: if constexpr (NT >= 6) wg_steps<48>(acc, A, sbase); break;
    case 7: if constexpr (NT >= 7) wg_steps<56>(acc, A, sbase); break;
    case 8: if constexpr (NT >= 8) wg_steps<64>(acc, A, sbase); break;
    case 9: if constexpr (NT >= 9) wg_steps<72>(acc, A, sbase); break;
    case 10: if constexpr (NT >= 10) wg_steps<80>(acc, A, sbase); break;
    case 11: if constexpr (NT >= 11) wg_steps<88>(acc, A, sbase); break;
    case 12: if constexpr (NT >= 12) wg_steps<96>(acc, A, sbase); break;
    case 13: if constexpr (NT >= 13) wg_steps<104>(acc, A, sbase); break;
    case 14: if constexpr (NT >= 14) wg_steps<112>(acc, A, sbase); break;
    case 15: if constexpr (NT >= 15) wg_steps<120>(acc, A, sbase); break;
    case 16: if constexpr (NT >= 16) wg_steps<128>(acc, A, sbase); break;
    default: break;
  }
}
// The accumulators as a compiler fence (CUTLASS's warpgroup_fence_operand on accum): no instruction that reads or
// writes them may be moved across this point, so nothing touches them between wgmma.fence and the wgmmas, and the
// flush can't read them before the wait. No code is emitted.
template <int TT, int NT>
__device__ __forceinline__ void fence_acc(float (&acc)[TT][NT][4]) {
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(acc[i][n][r])::"memory");
}
// Keeps a buffer's registers live (and untouched by the compiler) up to here, where its wgmmas are known done:
// CUTLASS's warpgroup_fence_operand, so the register allocator can't reuse them while a wgmma may read them.
template <int TT>
__device__ __forceinline__ void fence_regs(uint32_t (&A)[TT][4][4]) {
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(A[i][j][k])::"memory");
}


template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

// P: activation chunks staged ahead (P + 1 stages). NT: 8-pair column blocks per pass (NB = 8 NT <= 128).
// sf: [LOCAL][rt_total][KG][8] uint2 (M16 scales); w: [LOCAL][rt_total][KG][32] uint4 (natural-K words).
// DB: two A buffers, wgmma.wait_group 1 (a warp decodes chunk c while its chunk c - 1 MMAs run) and one more
// activation stage. MODE (recon only when not 0): 1 = decode but no wgmma, 2 = wgmma on raw words (no decode).
template <int TT, int NT, bool GATED, int P, bool DB = false, int MODE = 0>
__global__ void __launch_bounds__(128) wg_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= 8, "activation lead");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 1 + (DB ? 1 : 0);  // activation stages (DB: a chunk's MMAs may still read it one chunk later)
  static_assert(NT >= 1 && NT <= 16, "N = 8 NT <= 128");
  extern __shared__ __align__(128) unsigned char s_b[];  // 1024 B of slack, then [SBP][NB pairs][128 B] at 1024 B
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_base[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  __syncthreads();
  // Pair lists in pair order with 4 warps: per-warp-block counts, offsets per expert, then placement.
  const int q = ((n_pairs + 127) / 128) * 32;
  const int pb0 = warp * q, pb1 = min(n_pairs, pb0 + q);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] += __popc(m);
    __syncwarp();
  }
  __syncthreads();
  if (tid < LOCAL) {
    int run = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = s_wcnt[v][tid];
      s_wcnt[v][tid] = run;
      run += c;
    }
    s_cnt[tid] = run;
  }
  __syncthreads();
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    const int cur = valid ? s_wcnt[warp][e] : 0;
    if (valid) {
      const int pos = cur + __popc(m & lt);
      if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
    }
    __syncwarp();
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] = cur + __popc(m);
    __syncwarp();
  }
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;  // tasks per expert
  const int n_rb = per_expert / 4;                                // 4-task blocks per expert
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int v = act ? n_rb * ((cnt + NB - 1) / NB) * GPT : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_base[ai] = incl - v;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_base[__popc(am)] = incl;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total = s_base[nact];
  const int g0 = static_cast<int>(static_cast<long long>(total) * blockIdx.x / gridDim.x);
  const int g1 = static_cast<int>(static_cast<long long>(total) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {  // the whole CTA has no work; still wait, so this grid never completes before the previous one
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // c.task is the 4-task block here
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.pass == c.passes) {
        c.pass = 0;
        if (++c.task == n_rb) {
          c.task = 0;
          if (++c.ai < nact) set_expert(c);
        }
      }
      if (c.ai < nact) set_cntp(c);
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_base[ai], per_rb = L.passes * GPT;
    L.task = local / per_rb;
    const int rem = local - L.task * per_rb;
    L.pass = rem / GPT;
    L.grp = rem - L.pass * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint4* lw[TT];
  const uint2* ls[TT];
  auto point = [&](const Cur& c) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(c, i);
      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;
      ls[i] = sf + (row * KG + c.grp * D) * 8 + g;
    }
  };
  const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;  // the 128B swizzle pattern starts at 1024 B
  // Chunk s of group c's activations -> stage slot: NB pairs x 8 segments of 16 B, invalid pairs zero-filled.
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {  // only the 8-pair tiles compute() reads
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  uint4 rw[D][TT];
  uint2 rs[D][TT];  // scales of the chunk in ring slot s: (row g, row g + 8), byte j = 16-K block j
  float acc[TT][NT][4];
  uint32_t Ab[DB ? 2 : 1][TT][4][4];  // A fragments (DB: chunk parity)
  uint32_t sink = 0u;                 // recon modes only
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      rw[s][i] = __ldcs(lw[i] + s * 32);
      rs[s][i] = __ldcs(ls[i] + s * 8);
    }
  };
  // Chunk s: decode the 4 k16 steps of every tile into A fragments (natural K, the step's own scale), then
  // one wgmma per (step, tile) on the stage; the caller waits for them before the A registers or the stage
  // are reused (wg_wait0) and before reading acc (flush).
  auto compute = [&](int s, int cntp, int slot) {
    auto& A = Ab[DB ? (s & 1) : 0];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        const uint32_t sg = scale2r(rs[s][i].x, j), sg8 = scale2r(rs[s][i].y, j);
        if constexpr (MODE == 2) {  // recon: raw words
          A[i][j][0] = q;
          A[i][j][1] = q ^ sg;
          A[i][j][2] = q;
          A[i][j][3] = q ^ sg8;
        } else {
          A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
          A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
          A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
          A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
      }
    }
    const uint32_t sbase = sb0 + slot * STAGE;
    const int nv = (cntp + 7) >> 3;  // 8-pair column blocks this pass uses (1..NT), CTA-uniform
    if constexpr (MODE == 1) {  // recon: no wgmma
#pragma unroll
      for (int i = 0; i < TT; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sink ^= A[i][j][0] ^ A[i][j][1] ^ A[i][j][2] ^ A[i][j][3];
    } else {
      fence_regs<TT>(A);       // the decode is complete before wgmma.fence (it can't sink between fence and wgmma)
      fence_acc<TT, NT>(acc);  // so is any write of acc (the flush's zeroing)
      wg_fence();
      wg_chunk<TT, NT>(nv, acc, A, sbase);
      wg_commit();
    }
  };
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * wtask(c) + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * tile_rt(c, i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + 16 * tile_rt(c, i) + g + 8 * rr, acc[i][n][2 * rr + h]);
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* q = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
                }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  if (pdl) {
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * KG + q.grp * D) * 8, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (DB) {
        wg_wait1();       // this warp's wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage are free
        fence_regs<TT>(Ab[s & 1]);
      } else {
        wg_wait0();       // this warp's wgmmas of the previous chunk are done: its A registers and stage are free
      }
      fence_acc<TT, NT>(acc);
      cp_wait<P - 1>();
      fence_async_smem();  // this thread's copies of this chunk -> visible to wgmma (async proxy)
      __syncthreads();    // this chunk is in shared memory for all; no warp still reads the previous chunk's stage
      issue_b();          // chunk + P -> the stage chunk - 1 (DB: chunk - 2) used
      compute(s, cntp, cslot);
      if (more) load_w(s);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      wg_wait0();  // acc complete
      fence_acc<TT, NT>(acc);
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m16

// ------------------------------------------------------------------------------------------------
// M15: large batches, one decode per weight per call. As M14 (a CTA's 4 warps share each unit; the
// pass's activations staged per K chunk in a swizzled cp.async ring), but each warp's weights come in
// through its own ring of R shared-memory slots of one K group (TT tiles x (2 KB codes + 256 B scales)),
// filled by cp.async.bulk with mbarrier expect-tx (M12's machinery), so the bytes in flight cost no
// registers and the accumulators can cover 8 NT pairs (64 for GEMM1, 128 for GEMM2) in one pass.
// Units are ordered (expert, pass, 4-task block, K group) and split over CTAs by cost: a unit costs
// 8 + beta * (its pass's 8-pair tiles), so CTAs on the hot expert get fewer units (beta = pf >> 16).
namespace m15 {
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m11::mbar_init;
using m11::mbar_init_fence;
using m11::mbar_arrive_expect_tx;
using m11::mbar_wait;
using m11::bulk_g2s;
using m11::fence_proxy_async;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;
using m14::cp_async16;
using m14::cp_commit;
using m14::cp_wait1;
using m14::cp_wait;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes (shared with M13/M14)
// activation stages: P + 1 (P = chunks staged ahead; E9's kernel had P = 2)
constexpr int TILE_BYTES = D * 512 + 256;  // one tile's K group: 4 chunks of codes, then its scales

template <int TT, int NT, int R, int P>
constexpr int dyn_bytes() { return 4 * R * TT * TILE_BYTES + (P + 1) * 8 * NT * 128; }

template <int TT, int NT, bool GATED, int R, int P>
__global__ void __launch_bounds__(128) wk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = NB * 128;
  constexpr int SLOT = TT * TILE_BYTES;
  constexpr int SBP = P + 1;  // activation stages
  static_assert(P >= 2 && P <= 8, "activation lead");
  extern __shared__ __align__(128) unsigned char smem[];  // [4 warps][R][SLOT] weights, then [SB][NB][128] activations
  unsigned char* s_b = smem + 4 * R * SLOT;
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_ubase[LOCAL + 1];
  __shared__ int s_cbase[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int beta = (pf >> 16) & 0xFF;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[warp][r]), 1);
    mbar_init_fence();
  }
  __syncthreads();
  // Pair lists in pair order with 4 warps (as M14).
  const int q = ((n_pairs + 127) / 128) * 32;
  const int pb0 = warp * q, pb1 = min(n_pairs, pb0 + q);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] += __popc(m);
    __syncwarp();
  }
  __syncthreads();
  if (tid < LOCAL) {
    int run = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = s_wcnt[v][tid];
      s_wcnt[v][tid] = run;
      run += c;
    }
    s_cnt[tid] = run;
  }
  __syncthreads();
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    const int cur = valid ? s_wcnt[warp][e] : 0;
    if (valid) {
      const int pos = cur + __popc(m & lt);
      if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
    }
    __syncwarp();
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] = cur + __popc(m);
    __syncwarp();
  }
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_rb = per_expert / 4;
  const int blk = n_rb * GPT;  // units per (expert, pass)
  auto pass_cost = [&](int cnt, int p) { return 8 + beta * ((min(cnt - p * NB, NB) + 7) / 8); };
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int passes = act ? (cnt + NB - 1) / NB : 0;
    int cost = 0;
    for (int p = 0; p < passes; ++p) cost += pass_cost(cnt, p);
    int u = passes * blk, cc = cost * blk;
    int ui = u, ci = cc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a1 = __shfl_up_sync(0xffffffffu, ui, o);
      const int a2 = __shfl_up_sync(0xffffffffu, ci, o);
      if (lane >= o) {
        ui += a1;
        ci += a2;
      }
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_ubase[ai] = ui - u;
      s_cbase[ai] = ci - cc;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_ubase[__popc(am)] = ui;
      s_cbase[__popc(am)] = ci;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total_cost = s_cbase[nact];
  // A cost position -> the first unit that starts at or after it (CTA ranges in cost space, cut at units).
  auto cost_to_unit = [&](long long pos) -> int {
    if (pos >= total_cost) return s_ubase[nact];
    int ai = 0;
    while (ai + 1 < nact && s_cbase[ai + 1] <= pos) ++ai;
    const int cnt = min(s_cnt[s_active[ai]], MAXP);
    long long rem = pos - s_cbase[ai];
    int p = 0;
    while (true) {
      const int c = pass_cost(cnt, p);
      if (rem < static_cast<long long>(c) * blk) return s_ubase[ai] + p * blk + static_cast<int>((rem + c - 1) / c);
      rem -= static_cast<long long>(c) * blk;
      ++p;
    }
  };
  const int g0 = cost_to_unit(static_cast<long long>(total_cost) * blockIdx.x / gridDim.x);
  const int g1 = cost_to_unit(static_cast<long long>(total_cost) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // order: group, 4-task block (c.task), pass, expert
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.task == n_rb) {
        c.task = 0;
        if (++c.pass == c.passes) {
          c.pass = 0;
          if (++c.ai < nact) set_expert(c);
        }
        if (c.ai < nact) set_cntp(c);
      }
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_ubase[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_ubase[ai];
    L.pass = local / blk;
    const int rem = local - L.pass * blk;
    L.task = rem / GPT;
    L.grp = rem - L.task * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint32_t ring = smem_addr(smem + warp * (R * SLOT));
  const uint32_t bar0 = smem_addr(&bars[warp][0]);
  const unsigned char* myring = smem + warp * (R * SLOT);
  // Weight producer: lane 0 fills slot (seq % R) with group c of this warp's tiles.
  auto issue_w = [&](const Cur& c, int slot) {
    if (lane == 0) {
      const uint32_t bar = bar0 + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(c, i);
        const uint32_t dst = ring + slot * SLOT + i * TILE_BYTES;
        bulk_g2s(dst, w + (row * KG + c.grp * D) * 32, D * 512, bar);
        bulk_g2s(dst + D * 512, sf + (row * GPT + c.grp) * 32, 256, bar);
      }
    }
  };
  const uint32_t sb0 = smem_addr(s_b);
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  // Chunk s of the warp's weight slot -> decoded A registers, then MMAs over the pass's valid n-tiles.
  auto compute = [&](int s, const unsigned char* wslot, int cntp, int bslot) {
    uint32_t A[TT][4][4];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const unsigned char* tp = wslot + i * TILE_BYTES;
      const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);
      const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + lane * 8);
      const uint32_t sv = s < 2 ? sc.x : sc.y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t qw = word(q4, j);
        A[i][j][0] = hmul2(rotr(qw, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(qw, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(qw, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(qw, 8) & PM3) | (qw & PM3W), sg8);
      }
    }
    const unsigned char* sb = s_b + bslot * STAGE;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (n * 8 < cntp) {
        const unsigned char* st = sb + (n * 8 + g) * 128;
        const uint4 blo = *reinterpret_cast<const uint4*>(st + (((2 * t) ^ g) << 4));
        const uint4 bhi = *reinterpret_cast<const uint4*>(st + (((2 * t + 1) ^ g) << 4));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4& bv = j < 2 ? blo : bhi;
          const uint32_t b0 = word(bv, (2 * j) & 3), b1 = word(bv, (2 * j + 1) & 3);
#pragma unroll
          for (int i = 0; i < TT; ++i) mma(acc[i][n], A[i][j][0], A[i][j][1], A[i][j][2], A[i][j][3], b0, b1);
        }
      }
    }
  };
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * wtask(c) + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * tile_rt(c, i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + 16 * tile_rt(c, i) + g + 8 * rr, acc[i][n][2 * rr + h]);
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* qp = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(qp, 0.0f);  // the sum, read and cleared in one atomic
                }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  // Prologue: the first R groups' weights (they don't depend on the previous grid), then the wait, then
  // the first two activation chunks.
  Cur Wc = L;  // the next group whose weights are issued
  int wg = g0;
  for (int r = 0; r < R && wg < g1; ++r) {
    issue_w(Wc, r);
    ++wg;
    if (wg < g1) advance(Wc);
  }
  if (pdl) pdl_wait();
  // Activation producer, P chunks ahead of compute (as M14's): Bc = the group of the next chunk to stage, bsub
  // its chunk, bg its flat index, pslot its stage; past the range it only commits empty groups.
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    const int slot = (gi - g0) % R;
    if (more) advance(L);
    mbar_wait(bar0 + 8 * slot, ((gi - g0) / R) & 1);
    const unsigned char* wslot = myring + slot * SLOT;
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait<P - 1>();
      __syncthreads();    // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      issue_b();          // chunk + P -> the stage the previous chunk used
      compute(s, wslot, cntp, cslot);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    // The warp is done with this slot: refill it with the group R ahead.
    __syncwarp();
    if (wg < g1) {
      if (lane == 0) fence_proxy_async();
      issue_w(Wc, slot);
      ++wg;
      if (wg < g1) advance(Wc);
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m15

// ------------------------------------------------------------------------------------------------
// M17: M15 (per-warp TMA weight ring, activation ring P ahead, cost-weighted stream-K split) with M16's
// RS-wgmma: each chunk's A fragments are decoded from the warp's weight slot into registers (so the slot can be
// refilled as in M15), then one wgmma per (k16 step, tile) reads the activation stage through M16's descriptor.
// Natural-K bank (M16's): a tile's K group is still 4 x 512 B of codes then 256 B of scales, now 4 chunks x
// 8 rows g x uint2 (row g, row g + 8; byte j = 16-K block j).
namespace m17 {
using m16::wg_mma;
using m16::wg_fence;
using m16::wg_commit;
using m16::wg_wait0;
using m16::wg_wait1;
using m16::wg_chunk;
using m16::fence_regs;
using m16::fence_acc;
using m16::b_desc;
using m16::scale2r;
using m16::fence_async_smem;
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m11::mbar_init;
using m11::mbar_init_fence;
using m11::mbar_arrive_expect_tx;
using m11::mbar_wait;
using m11::bulk_g2s;
using m11::fence_proxy_async;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;
using m14::cp_async16;
using m14::cp_commit;
using m14::cp_wait1;
using m14::cp_wait;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes (shared with M13/M14)
// activation stages: P + 1 (P = chunks staged ahead; E9's kernel had P = 2)
constexpr int TILE_BYTES = D * 512 + 256;  // one tile's K group: 4 chunks of codes, then its scales

template <int TT, int NT, int R, int P, bool DB>
constexpr int dyn_bytes() { return 4 * R * TT * TILE_BYTES + 1024 + (P + 1 + (DB ? 1 : 0)) * 8 * NT * 128; }  // +1024: alignment

// DB as M16's: two A buffers, wgmma.wait_group 1, one more activation stage.
template <int TT, int NT, bool GATED, int R, int P, bool DB = false>
__global__ void __launch_bounds__(128) wq_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = NB * 128;
  constexpr int SLOT = TT * TILE_BYTES;
  constexpr int SBP = P + 1 + (DB ? 1 : 0);  // activation stages (DB: a chunk's MMAs may still read it one chunk later)
  static_assert(NT >= 1 && NT <= 16, "N = 8 NT <= 128");
  static_assert(P >= 2 && P <= 8, "activation lead");
  extern __shared__ __align__(128) unsigned char smem[];  // [4 warps][R][SLOT] weights, then [SB][NB][128] activations
  unsigned char* s_b = smem + 4 * R * SLOT;
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_ubase[LOCAL + 1];
  __shared__ int s_cbase[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int beta = (pf >> 16) & 0xFF;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[warp][r]), 1);
    mbar_init_fence();
  }
  __syncthreads();
  // Pair lists in pair order with 4 warps (as M14).
  const int q = ((n_pairs + 127) / 128) * 32;
  const int pb0 = warp * q, pb1 = min(n_pairs, pb0 + q);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] += __popc(m);
    __syncwarp();
  }
  __syncthreads();
  if (tid < LOCAL) {
    int run = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = s_wcnt[v][tid];
      s_wcnt[v][tid] = run;
      run += c;
    }
    s_cnt[tid] = run;
  }
  __syncthreads();
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    const int cur = valid ? s_wcnt[warp][e] : 0;
    if (valid) {
      const int pos = cur + __popc(m & lt);
      if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
    }
    __syncwarp();
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] = cur + __popc(m);
    __syncwarp();
  }
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_rb = per_expert / 4;
  const int blk = n_rb * GPT;  // units per (expert, pass)
  auto pass_cost = [&](int cnt, int p) { return 8 + beta * ((min(cnt - p * NB, NB) + 7) / 8); };
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int passes = act ? (cnt + NB - 1) / NB : 0;
    int cost = 0;
    for (int p = 0; p < passes; ++p) cost += pass_cost(cnt, p);
    int u = passes * blk, cc = cost * blk;
    int ui = u, ci = cc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a1 = __shfl_up_sync(0xffffffffu, ui, o);
      const int a2 = __shfl_up_sync(0xffffffffu, ci, o);
      if (lane >= o) {
        ui += a1;
        ci += a2;
      }
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_ubase[ai] = ui - u;
      s_cbase[ai] = ci - cc;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_ubase[__popc(am)] = ui;
      s_cbase[__popc(am)] = ci;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total_cost = s_cbase[nact];
  // A cost position -> the first unit that starts at or after it (CTA ranges in cost space, cut at units).
  auto cost_to_unit = [&](long long pos) -> int {
    if (pos >= total_cost) return s_ubase[nact];
    int ai = 0;
    while (ai + 1 < nact && s_cbase[ai + 1] <= pos) ++ai;
    const int cnt = min(s_cnt[s_active[ai]], MAXP);
    long long rem = pos - s_cbase[ai];
    int p = 0;
    while (true) {
      const int c = pass_cost(cnt, p);
      if (rem < static_cast<long long>(c) * blk) return s_ubase[ai] + p * blk + static_cast<int>((rem + c - 1) / c);
      rem -= static_cast<long long>(c) * blk;
      ++p;
    }
  };
  const int g0 = cost_to_unit(static_cast<long long>(total_cost) * blockIdx.x / gridDim.x);
  const int g1 = cost_to_unit(static_cast<long long>(total_cost) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // order: group, 4-task block (c.task), pass, expert
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.task == n_rb) {
        c.task = 0;
        if (++c.pass == c.passes) {
          c.pass = 0;
          if (++c.ai < nact) set_expert(c);
        }
        if (c.ai < nact) set_cntp(c);
      }
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_ubase[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_ubase[ai];
    L.pass = local / blk;
    const int rem = local - L.pass * blk;
    L.task = rem / GPT;
    L.grp = rem - L.task * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint32_t ring = smem_addr(smem + warp * (R * SLOT));
  const uint32_t bar0 = smem_addr(&bars[warp][0]);
  const unsigned char* myring = smem + warp * (R * SLOT);
  // Weight producer: lane 0 fills slot (seq % R) with group c of this warp's tiles.
  auto issue_w = [&](const Cur& c, int slot) {
    if (lane == 0) {
      const uint32_t bar = bar0 + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(c, i);
        const uint32_t dst = ring + slot * SLOT + i * TILE_BYTES;
        bulk_g2s(dst, w + (row * KG + c.grp * D) * 32, D * 512, bar);
        bulk_g2s(dst + D * 512, sf + (row * KG + c.grp * D) * 8, 256, bar);  // M16 scales: 4 chunks x 8 x uint2
      }
    }
  };
  const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;  // the 128B swizzle pattern starts at 1024 B
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  // Chunk s of the warp's weight slot -> A fragments (natural K, each k16 step its own scale), then one wgmma per
  // (step, tile) on activation stage bslot; waited for before the stage is reused (wg_wait0) and before flush.
  uint32_t Ab[DB ? 2 : 1][TT][4][4];  // A fragments (DB: chunk parity)
  auto compute = [&](int s, const unsigned char* wslot, int cntp, int bslot) {
    auto& A = Ab[DB ? (s & 1) : 0];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const unsigned char* tp = wslot + i * TILE_BYTES;
      const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);
      const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + s * 64 + g * 8);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t qw = word(q4, j);
        const uint32_t sg = scale2r(sc.x, j), sg8 = scale2r(sc.y, j);
        A[i][j][0] = hmul2(rotr(qw, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(qw, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(qw, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(qw, 8) & PM3) | (qw & PM3W), sg8);
      }
    }
    const uint32_t sbase = sb0 + bslot * STAGE;
    const int nv = (cntp + 7) >> 3;  // 8-pair column blocks this pass uses (1..NT), CTA-uniform
    fence_regs<TT>(A);       // as M16: decode and acc writes complete before wgmma.fence
    fence_acc<TT, NT>(acc);
    wg_fence();
    wg_chunk<TT, NT>(nv, acc, A, sbase);
    wg_commit();
  };
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * wtask(c) + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * tile_rt(c, i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + 16 * tile_rt(c, i) + g + 8 * rr, acc[i][n][2 * rr + h]);
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* qp = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(qp, 0.0f);  // the sum, read and cleared in one atomic
                }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  // Prologue: the first R groups' weights (they don't depend on the previous grid), then the wait, then
  // the first two activation chunks.
  Cur Wc = L;  // the next group whose weights are issued
  int wg = g0;
  for (int r = 0; r < R && wg < g1; ++r) {
    issue_w(Wc, r);
    ++wg;
    if (wg < g1) advance(Wc);
  }
  if (pdl) pdl_wait();
  // Activation producer, P chunks ahead of compute (as M14's): Bc = the group of the next chunk to stage, bsub
  // its chunk, bg its flat index, pslot its stage; past the range it only commits empty groups.
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    const int slot = (gi - g0) % R;
    if (more) advance(L);
    mbar_wait(bar0 + 8 * slot, ((gi - g0) / R) & 1);
    const unsigned char* wslot = myring + slot * SLOT;
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (DB) {
        wg_wait1();       // this warp's wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage are free
        fence_regs<TT>(Ab[s & 1]);
      } else {
        wg_wait0();       // this warp's wgmmas of the previous chunk are done: that stage is free
      }
      fence_acc<TT, NT>(acc);
      cp_wait<P - 1>();
      fence_async_smem();  // this thread's copies of this chunk -> visible to wgmma (async proxy)
      __syncthreads();    // this chunk is in shared memory for all; no warp still reads the previous chunk's stage
      issue_b();          // chunk + P -> the stage the previous chunk used
      compute(s, wslot, cntp, cslot);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    // The warp is done with this slot: refill it with the group R ahead.
    __syncwarp();
    if (wg < g1) {
      if (lane == 0) fence_proxy_async();
      issue_w(Wc, slot);
      ++wg;
      if (wg < g1) advance(Wc);
    }
    if (C.grp == GPT - 1 || !more) {
      wg_wait0();  // acc complete
      fence_acc<TT, NT>(acc);
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m17



// Launch with programmatic stream serialization when pdl (the kernel then may start before the
// previous kernel in the stream completes; it calls griddepcontrol.wait before reading its outputs).
template <typename... Exp, typename... Act>
static void m13_launch(void (*kernel)(Exp...), int grid, int threads, size_t smem, int pdl, Act... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "m13 launch failed: ", cudaGetErrorString(err));
}

#define M13_CASE(G, TT, NT) M13_CASE_CG(G, TT, NT, false, (G) || !pdl)
#define M13_CASE_P(G, TT, NT) M13_CASE_CG(G, TT, NT, true, pdl) M13_CASE(G, TT, NT)
// One kernel instance (CG fixed) when cond holds; plain GEMM2 instances come in both CG flavours.
#define M13_CASE_CG(G, TT, NT, CGV, COND)                                                                   \
  if (gated == (G) && tt == (TT) && nt == (NT) && (COND)) {                                                \
    m13_launch(m13::sk_kernel<TT, NT, G, 0, CGV>, static_cast<int>(grid), 128, 0, static_cast<int>(pdl),    \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf), reinterpret_cast<const char*>(nxw), reinterpret_cast<const char*>(nxs),  \
               static_cast<long long>(nxwb), static_cast<long long>(nxsb), static_cast<int>(nxwarps));      \
    return out;                                                                                            \
  }

// [registers, spill bytes, static shared memory, CTAs per SM at 128 threads]
#define M13_ATTR(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m13::sk_kernel<TT, NT, G>);                                               \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m13::sk_kernel<TT, NT, G>, 128, 0);            \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes),  \
                          static_cast<int64_t>(attr.sharedSizeBytes), static_cast<int64_t>(blocks)});     \
  }

torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf, int64_t nxw, int64_t nxs, int64_t nxwb, int64_t nxsb,
                       int64_t nxwarps) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_gemm: ", n_pairs, " pairs, at most ", m13::MAXPAIRS);
  TORCH_CHECK(((pf >> 24) & 7) == 0 || (nxw != 0 && nxs != 0 && nxwb > 0 && nxsb > 0 && nxwb % 16 == 0 &&
                                       nxsb % 16 == 0 && nxwarps > 0), "m13_gemm: bad next-stage prefetch");
  @@M13_CASES@@
  TORCH_CHECK(false, "m13_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

#define M13_RECON_CASE(G, TT, NT, MODE)                                                                     \
  if (gated == (G) && tt == (TT) && nt == (NT) && mode == (MODE)) {                                        \
    m13_launch(m13::sk_kernel<TT, NT, G, MODE>, static_cast<int>(grid), 128, 0, 0,                         \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), 0, 0, nullptr, nullptr, 0LL, 0LL, 0); \
    return out;                                                                                            \
  }

// The M13 GEMM in a recon mode (1 loads only, 2 no MMA, 3 no decode), for recon labs.
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t mode) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_recon: too many pairs");
  M13_RECON_CASE(true, 2, 1, 1)
  M13_RECON_CASE(true, 2, 1, 2)
  M13_RECON_CASE(true, 2, 1, 3)
  M13_RECON_CASE(false, 2, 1, 1)
  TORCH_CHECK(false, "m13_recon: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

#define M13_STREAM_CASE(U)                                                                                  \
  if (unroll == (U)) {                                                                                     \
    m13_launch(m13::stream_kernel<U>, static_cast<int>(grid), 128, 0, 0,                                   \
               reinterpret_cast<const uint4*>(base.data_ptr()), static_cast<const int*>(experts.data_ptr<int>()), \
               static_cast<int>(experts.numel()), static_cast<long long>(len16),                           \
               reinterpret_cast<unsigned*>(sink.data_ptr<int>()));                                         \
    return sink;                                                                                           \
  }

// Contiguous streaming of the listed experts' regions of base (len16 uint4 each), for recon labs.
torch::Tensor m13_stream(torch::Tensor base, torch::Tensor experts, int64_t len16, torch::Tensor sink, int64_t unroll,
                         int64_t grid) {
  M13_STREAM_CASE(4)
  M13_STREAM_CASE(8)
  M13_STREAM_CASE(16)
  TORCH_CHECK(false, "m13_stream: unroll ", unroll);
  return sink;
}

torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt) {
  @@M13_ATTRS@@
  TORCH_CHECK(false, "m13_attrs: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return torch::Tensor();
}

torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl) {
  m13_launch(m13::post_norm_kernel, static_cast<int>(tokens), 512, 0, static_cast<int>(pdl),
             reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()),
             static_cast<const float*>(tw.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(local_start), static_cast<float>(eps),
             static_cast<int>(pdl));
  return out;
}

#define M14_CASE(G, TT, NT, P)                                                                              \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P)) {                                             \
    m13_launch(m14::ck_kernel<TT, NT, G, P>, static_cast<int>(grid), 128,                                  \
               static_cast<size_t>(((P) + 1) * m14::stage_bytes<NT>()), static_cast<int>(pdl),             \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// Raises each instance's dynamic shared-memory limit (once, in prepare) and reports
// [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM].
#define M14_SETUP(G, TT, NT, P)                                                                             \
  {                                                                                                        \
    const int dyn = ((P) + 1) * m14::stage_bytes<NT>();                                                    \
    cudaFuncSetAttribute(m14::ck_kernel<TT, NT, G, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m14::ck_kernel<TT, NT, G, P>);                                            \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m14::ck_kernel<TT, NT, G, P>, 128, dyn);       \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, P, static_cast<int64_t>(attr.numRegs),                    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_gemm: ", n_pairs, " pairs, at most ", 8 * m14::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P (0 = E9's 2)
  @@M14_CASES@@
  TORCH_CHECK(false, "m14_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa);
  return out;
}

#define M14_RECON_CASE(G, TT, NT, P, MODE)                                                                  \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && mode == (MODE)) {                           \
    m13_launch(m14::ck_kernel<TT, NT, G, P, MODE>, static_cast<int>(grid), 128,                            \
               static_cast<size_t>(((P) + 1) * m14::stage_bytes<NT>()), 0,                                 \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), 0, 0);                            \
    return out;                                                                                            \
  }

// M14 GEMM1 n2 (P = 4) in a recon mode (1 loads + staging only, 2 no MMA, 3 no decode), for the tuner's recon.
torch::Tensor m14_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t pa, int64_t mode) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_recon: too many pairs");
  M14_RECON_CASE(true, 2, 2, 4, 1)
  M14_RECON_CASE(true, 2, 2, 4, 2)
  M14_RECON_CASE(true, 2, 2, 4, 3)
  TORCH_CHECK(false, "m14_recon: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa, " mode=", mode);
  return out;
}

torch::Tensor m14_setup() {
  std::vector<int64_t> rows;
  @@M14_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M15_CASE(G, TT, NT, R, P)                                                                           \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P)) {                                             \
    m13_launch(m15::wk_kernel<TT, NT, G, R, P>, static_cast<int>(grid), 128,                               \
               static_cast<size_t>(m15::dyn_bytes<TT, NT, R, P>()), static_cast<int>(pdl),                 \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP: [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM] (R is
// fixed per (gated, tt, nt, P)).
#define M15_SETUP(G, TT, NT, R, P)                                                                          \
  {                                                                                                        \
    const int dyn = m15::dyn_bytes<TT, NT, R, P>();                                                        \
    cudaFuncSetAttribute(m15::wk_kernel<TT, NT, G, R, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m15::wk_kernel<TT, NT, G, R, P>);                                         \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m15::wk_kernel<TT, NT, G, R, P>, 128, dyn);    \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, P, static_cast<int64_t>(attr.numRegs),                    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m15_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m15::MAXP, "m15_gemm: ", n_pairs, " pairs, at most ", 8 * m15::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  @@M15_CASES@@
  TORCH_CHECK(false, "m15_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m15_setup() {
  std::vector<int64_t> rows;
  @@M15_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M17_CASE(G, TT, NT, R, P, DB)                                                                       \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && db == (DB)) {                               \
    m13_launch(m17::wq_kernel<TT, NT, G, R, P, DB>, static_cast<int>(grid), 128,                           \
               static_cast<size_t>(m17::dyn_bytes<TT, NT, R, P, DB>()), static_cast<int>(pdl),             \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP: [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM] (R is
// fixed per (gated, tt, nt, P)).
#define M17_SETUP(G, TT, NT, R, P, DB)                                                                      \
  {                                                                                                        \
    const int dyn = m17::dyn_bytes<TT, NT, R, P, DB>();                                                    \
    cudaFuncSetAttribute(m17::wq_kernel<TT, NT, G, R, P, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m17::wq_kernel<TT, NT, G, R, P, DB>);                                     \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m17::wq_kernel<TT, NT, G, R, P, DB>, 128, dyn); \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((DB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m17_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m17::MAXP, "m17_gemm: ", n_pairs, " pairs, at most ", 8 * m17::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t db = (pf >> 28) & 1;                              // double-buffered A
  @@M17_CASES@@
  TORCH_CHECK(false, "m17_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m17_setup() {
  std::vector<int64_t> rows;
  @@M17_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M16_CASE(G, TT, NT, P, DB)                                                                          \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && db == (DB)) {                               \
    m13_launch(m16::wg_kernel<TT, NT, G, P, DB>, static_cast<int>(grid), 128,                              \
               static_cast<size_t>(1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>()), static_cast<int>(pdl), \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP, with P | DB << 4 in the P column: [gated, tt, nt, P | DB << 4, registers, spill bytes, static
// smem, dynamic smem, CTAs per SM].
#define M16_SETUP(G, TT, NT, P, DB)                                                                         \
  {                                                                                                        \
    const int dyn = 1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>();                                      \
    cudaFuncSetAttribute(m16::wg_kernel<TT, NT, G, P, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m16::wg_kernel<TT, NT, G, P, DB>);                                        \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m16::wg_kernel<TT, NT, G, P, DB>, 128, dyn);   \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((DB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m16_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m16::MAXP, "m16_gemm: ", n_pairs, " pairs, at most ", 8 * m16::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t db = (pf >> 28) & 1;                              // double-buffered A
  @@M16_CASES@@
  TORCH_CHECK(false, "m16_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa);
  return out;
}

torch::Tensor m16_setup() {
  std::vector<int64_t> rows;
  @@M16_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

// M16 GEMM1 g2.4 (P 4, DB) in a recon mode (1 = decode without wgmma, 2 = wgmma without decode).
torch::Tensor m16_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode) {
  TORCH_CHECK(n_pairs <= 8 * m16::MAXP, "m16_recon: too many pairs");
  TORCH_CHECK(mode == 1 || mode == 2, "m16_recon: mode ", mode);
  @@M16_RECON@@
  return out;
}
"""

_M13_CPP = """
torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf, int64_t nxw, int64_t nxs, int64_t nxwb, int64_t nxsb,
                       int64_t nxwarps);
torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt);
torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl);
torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m14_setup();
torch::Tensor m14_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t pa, int64_t mode);
torch::Tensor m15_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m15_setup();
torch::Tensor m16_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m16_setup();
torch::Tensor m16_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode);
torch::Tensor m17_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m17_setup();
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t mode);
torch::Tensor m13_stream(torch::Tensor base, torch::Tensor experts, int64_t len16, torch::Tensor sink, int64_t unroll,
                         int64_t grid);
"""


def m13_source(instances=M13_INSTANCES, m11_instances=M13_M11_INSTANCES, m14_instances=M14_INSTANCES,
               m15_instances=M15_INSTANCES, m16_instances=M16_INSTANCES, m17_instances=None):
    """M11's CUDA (only the instances M11_CONFIGS uses, no M12) followed by M13's and M14's."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in instances]
    args14 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}" for g, tt, nt, pa in m14_instances]
    args15 = [f"{'true' if g else 'false'}, {tt}, {nt}, {r}, {pa}" for g, tt, nt, r, pa in m15_instances]
    args16 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}, {'true' if db else 'false'}" for g, tt, nt, pa, db in m16_instances]
    recon16 = ("\n  ".join(f"if (mode == {md}) {{ m13_launch(m16::wg_kernel<2, 4, true, 4, true, {md}>, static_cast<int>(grid), "
                           "128, static_cast<size_t>(1024 + 6 * m16::stage_bytes<4>()), 0, "
                           "reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), "
                           "static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), "
                           "reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), "
                           "scr.data_ptr<float>(), ctr.data_ptr<int>(), static_cast<int>(n_pairs), static_cast<int>(local_start), "
                           "static_cast<int>(rt_total), 0, static_cast<int>(rt_total / 2 * 16), 0, (4 << 24) | (1 << 28)); }"
                           for md in (1, 2)) if (1, 2, 4, 4, 1) in m16_instances else 'TORCH_CHECK(false, "m16_recon: not built");')
    if m17_instances is None:  # M17 uses M16's wgmma code: built (or dropped) with it
        m17_instances = M17_INSTANCES if m16_instances else ()
    args17 = [f"{'true' if g else 'false'}, {tt}, {nt}, {r}, {pa}, {'true' if db else 'false'}" for g, tt, nt, r, pa, db in m17_instances]
    cases = [f"M13_CASE{'' if g else '_P'}({a})" for (g, _, _), a in zip(instances, args)]
    m13 = (_M13_CUDA.replace("@@M13_CASES@@", "\n  ".join(cases))
           .replace("@@M13_ATTRS@@", "\n  ".join(f"M13_ATTR({a})" for a in args))
           .replace("@@M13_MAXP@@", str(M13_MAX_TOKENS))
           .replace("@@M13_MAXPAIRS@@", str(M13_MAX_TOKENS * TOP_K))
           .replace("@@M14_MAXP@@", str(M14_MAX_TOKENS))
           .replace("@@M14_CASES@@", "\n  ".join(f"M14_CASE({a})" for a in args14))
           .replace("@@M14_SETUPS@@", "\n  ".join(f"M14_SETUP({a})" for a in args14))
           .replace("@@M15_CASES@@", "\n  ".join(f"M15_CASE({a})" for a in args15))
           .replace("@@M15_SETUPS@@", "\n  ".join(f"M15_SETUP({a})" for a in args15))
           .replace("@@M16_CASES@@", "\n  ".join(f"M16_CASE({a})" for a in args16))
           .replace("@@M16_SETUPS@@", "\n  ".join(f"M16_SETUP({a})" for a in args16))
           .replace("@@M16_RECON@@", recon16)
           .replace("@@M17_CASES@@", "\n  ".join(f"M17_CASE({a})" for a in args17))
           .replace("@@M17_SETUPS@@", "\n  ".join(f"M17_SETUP({a})" for a in args17)))
    return m11_source(m11_instances, ()) + m13


def m13_build(m16_instances=M16_INSTANCES):
    """One extension with M11's and M13's kernels (sm_90a only, as m11_build)."""
    return load_inline(
        name="m13_ext" if m16_instances else "m13_ext_nowg", cpp_sources=_M11_CPP + _M13_CPP,
        cuda_sources=m13_source(m16_instances=m16_instances),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                   "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + ([] if m16_instances else ["-DM13_NO_WGMMA"]),
        build_directory=tempfile.mkdtemp(prefix="m13_"), verbose=False)


def m13_probe_build():
    """One M16 and one M17 instance compiled with the build log shown (verbose), so ptxas's wgmma warnings (e.g.
    "wgmma.mma_async instructions are serialized due to ...") reach the run's output. Diagnostic only."""
    src = m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(),
                     m16_instances=((1, 2, 4, 4, 1),), m17_instances=((0, 2, 8, 2, 4, 1),))
    return load_inline(
        name="m13_probe", cpp_sources=_M11_CPP + _M13_CPP, cuda_sources=src,
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                   "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m13p_"), verbose=True)


def m13_relayout(words):
    """M11 code words (int32, any shape) -> M13's bit layout, a lossless permutation of each word's bits."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for new, old in enumerate(M13_BIT_SRC):
        out |= ((w >> old) & 1) << new
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


def m16_shuffle(packed, sf):
    """One expert's NVFP4 matrix -> M16's natural-K layout (a lossless reordering) for RS-wgmma, whose B operand
    (activations in shared memory) is read in natural K order. packed uint8 [rows, K/2], sf float8_e4m3fn
    [rows, K/16] -> words int32 [rows/16, K/64, 32 lanes, 4] in M11's nibble placement (so M11's / M13's decode
    applies), but lane (g, t), word j holds the positions wgmma's A fragment of k16 step j needs: 16 j + 2 t +
    {0, 1} (registers a0 / a1: rows g / g + 8) and 16 j + 2 t + 8 + {0, 1} (a2 / a3); scales int32
    [rows/16, K/64, 8, 2]: (x, y) = rows g and g + 8, byte j = the chunk's 16-K block j (read by all 4 t lanes)."""
    rows, half = packed.shape
    rt, kg = rows // 16, half * 2 // 64
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    c = codes.view(rt, 2, 8, kg, 64)  # (rt, h, g, kg, pos): row 16 rt + 8 h + g, k = 64 kg + pos
    t = torch.arange(4, device=packed.device).view(4, 1, 1)
    j = torch.arange(4, device=packed.device).view(1, 4, 1)
    q = torch.arange(4, device=packed.device).view(1, 1, 4)
    pos = 16 * j + 2 * t + 8 * (q >> 1) + (q & 1)  # [t, j, q]: q = 2 * (register >= 2) + element
    c = c[..., pos]  # (rt, h, g, kg, t, j, q)
    src = ((1, 2), (0, 2), (1, 0), (0, 0), (1, 3), (0, 3), (1, 1), (0, 1))  # as m11_shuffle
    nib = [c[:, h, :, :, :, :, qq] for h, qq in src]  # each [rt, g, kg, t, j]
    words = torch.stack([nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(4)], dim=-1)
    words = words.permute(0, 2, 1, 3, 4, 5).contiguous().view(torch.int32).view(rt, kg, 32, 4)
    sc = sf.view(torch.uint8).view(rt, 2, 8, kg, 4)  # (rt, h, g, kg, block)
    scales = sc.permute(0, 3, 2, 1, 4).contiguous().view(torch.int32).view(rt, kg, 8, 2)
    return words, scales


def m13_scales_ok(weights, start, count):
    """True when every local E4M3 block scale is positive and normal (no zero, subnormal or NaN),
    which scale2's bit trick needs; M13 stays off otherwise."""
    for name in ("w13_sf", "w2_sf"):
        b = weights[name][start:start + count].view(torch.uint8)
        bad = ((b & 0x80) != 0) | ((b & 0x78) == 0) | ((b & 0x7F) == 0x7F)
        if bool(bad.any()):
            return False
    return True


def m16_bank(weights, bank, start, count):
    """M16's natural-K codes (M13 bit layout) and per-block scales, one expert at a time; they share M13's
    scratch, counters and global scales."""
    for name in ("w13", "w2"):
        ws, ss = [], []
        for e in range(start, start + count):
            words, scales = m16_shuffle(weights[name][e], weights[name + "_sf"][e])
            ws.append(m13_relayout(words))
            ss.append(scales)
        bank[name + "n"] = torch.stack(ws)
        bank[name + "n_s"] = torch.stack(ss)


def m13_bank(m11, device):
    """M13's weights from M11's bank (same fragment order): re-laid-out words, scales grouped 4 chunks per
    lane, and the zeroed FP32 scratch and K-group counters the kernels keep clean between calls."""
    bank = {k: m11[k] for k in ("start", "count", "inter", "hidden", "c13", "c2", "rt_w13", "rt_w2")}
    for name in ("w13", "w2"):
        bank[name] = m13_relayout(m11[name])
        s = m11[name + "_s"]  # int16 [count, rt, KG, 32]
        n, rt, kg, lanes = s.shape
        bank[name + "_s"] = s.view(n, rt, kg // 4, 4, lanes).permute(0, 1, 2, 4, 3).contiguous()
    pairs = max(M13_MAX_TOKENS, M14_MAX_TOKENS) * TOP_K  # scratch rows: one per pair
    maxpass = (max(M13_MAX_TOKENS, M14_MAX_TOKENS) + 7) // 8  # counters laid out for 8-pair passes
    bank["scr13"] = torch.zeros((pairs, 16 * bank["rt_w13"]), dtype=torch.float32, device=device)
    bank["scr2"] = torch.zeros((pairs, 16 * bank["rt_w2"]), dtype=torch.float32, device=device)
    bank["ctr13"] = torch.zeros((32 * (bank["rt_w13"] // 2) * maxpass,), dtype=torch.int32, device=device)
    bank["ctr2"] = torch.zeros((32 * bank["rt_w2"] * maxpass,), dtype=torch.int32, device=device)
    return bank


def _m13_next(state, s1, s2):
    """M13 GEMM1's next-stage prefetch arguments: GEMM2's code and scale arrays in M13's bank (M13, M14 and M15
    GEMM2 all read it; addresses and bytes per expert taken once in m13_setup), and GEMM2's warp count; zeros
    when s1 has nxq 0."""
    if not s1.get("nxq", 0) or s2 is None or s2.get("kind") not in ("m13", "m14", "m15"):
        return (0, 0, 0, 0, 0)
    return state["m13_nx"] + (4 * s2["c"] * state["sms"],)


def m13_gemm1(x, ids, state, s1, s2=None):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair. s2 (GEMM2's stage) only
    sizes M13's next-stage L2 prefetch."""
    kind = s1.get("kind")
    if kind not in ("m13", "m14", "m15", "m16", "m17"):
        return m11_gemm1(x, ids, state, s1)
    bank, n_pairs = state["m13"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    gemm = getattr(state["m11_ext"], kind + "_gemm")
    pf = (s1.get("pfd", 0) << 8) | (s1.get("beta", 0) << 16)
    wk = "w13n" if kind in ("m16", "m17") else "w13"
    args = [bank[wk], bank[wk + "_s"], bank["c13"], ids, x, act, bank["scr13"], bank["ctr13"],
            n_pairs, bank["start"], bank["rt_w13"], 0, bank["inter"], 1, 2, s1["nt"], s1["c"] * state["sms"], 0]
    if kind == "m13":
        nx = _m13_next(state, s1, s2)
        pf |= s1.get("npre", 0) | ((s1.get("nxq", 0) if nx[4] else 0) << 24)
        gemm(*args, pf, *nx)
    else:
        gemm(*args, pf | (s1.get("p", 0) << 24) | (s1.get("db", 0) << 28))
    return act


def m13_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    kind = s2.get("kind")
    if kind not in ("m13", "m14", "m15", "m16", "m17"):
        return m11_gemm2(act, ids, state, s2)
    bank, n_pairs = state["m13"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    gemm = getattr(state["m11_ext"], kind + "_gemm")
    wk = "w2n" if kind in ("m16", "m17") else "w2"
    args = [bank[wk], bank[wk + "_s"], bank["c2"], ids, act, y, bank["scr2"], bank["ctr2"],
            n_pairs, bank["start"], bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"],
            s2["c"] * state["sms"], s2.get("pdl", 0),
            s2.get("pf", 0) | (s2.get("pfd", 0) << 8) | (s2.get("beta", 0) << 16)]
    if kind == "m13":
        args[-1] |= s2.get("npre", 0)
        gemm(*args, 0, 0, 0, 0, 0)
    else:
        args[-1] |= (s2.get("p", 0) << 24) | (s2.get("db", 0) << 28)
        gemm(*args)
    return y


def m13_post_norm(y, ids, topk_weights, state, tokens, post):
    if not post:
        return m11_post_norm(y, ids, topk_weights, state, tokens)
    out = torch.empty((tokens, state["m13"]["hidden"]), dtype=torch.bfloat16, device=y.device)
    state["m11_ext"].m13_post_norm(y, state["norm_w"], topk_weights, ids, out, tokens, state["m13"]["start"],
                                   POST_NORM_EPS, post.get("pdl", 0))
    return out


def m13_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m13_gemm1(x, ids, state, cfg["s1"], cfg["s2"])
    y = m13_gemm2(act, ids, state, cfg["s2"])
    return m13_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0], cfg.get("post"))


def m13_config_for(state, tokens):
    if tokens > max(M13_MAX_TOKENS, M14_MAX_TOKENS) or "m13" not in state:
        return None
    for below, cfg in state["m13_configs"]:
        if tokens < below:
            return cfg
    return None


def m13_warm(state, cfg, tokens=8):
    """Run cfg once before any capture (and check it doesn't fault)."""
    device = state["norm_w"].device
    x = torch.zeros((tokens, state["m13"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m13"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m13_run(x, ids + state["m13"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m13_setup(state, weights):
    """Build M11 + M13 in one extension, M11's bank (the M11 fallback configs) and M13's bank."""
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    try:
        state["m11_ext"] = m13_build()
        state["m16_ok"] = True
    except Exception as exc:  # M16 is the first wgmma code: if ptxas rejects it, build everything else
        print(f"[m13] build with M16 FAILED ({type(exc).__name__}: {str(exc)[-300:]}); rebuilding without M16",
              flush=True)
        state["m11_ext"] = m13_build(())
        state["m16_ok"] = False
    state["m12_resources"] = []
    # M14's dynamic shared-memory limits, set once here and never during a capture
    state["m14_resources"] = state["m11_ext"].m14_setup().tolist()
    state["m15_resources"] = state["m11_ext"].m15_setup().tolist()
    state["m16_resources"] = state["m11_ext"].m16_setup().tolist()
    state["m17_resources"] = state["m11_ext"].m17_setup().tolist()
    state["m11"] = m11_bank(weights, 0, 32)
    state["m13_ok"] = m13_scales_ok(weights, 0, 32)
    if state["m13_ok"]:
        state["m13"] = m13_bank(state["m11"], weights["w13"].device)
        m16_bank(weights, state["m13"], 0, 32)
        w, s = state["m13"]["w2"], state["m13"]["w2_s"]  # the persistent bank: its addresses never change
        state["m13_nx"] = (w.data_ptr(), s.data_ptr(), w[0].numel() * w.element_size(), s[0].numel() * s.element_size())
    torch.cuda.synchronize()


def prepare(weights):
    state = _v1_prepare(weights)
    state["m11_configs"] = M11_CONFIGS
    state["m13_configs"] = M13_CONFIGS
    m13_setup(state, weights)
    for _, cfg in M11_CONFIGS:
        if cfg is not None:
            m11_warm(state, cfg)
    if "m13" in state:
        for _, cfg in M13_CONFIGS:
            if cfg is not None:
                m13_warm(state, cfg)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m13cfg = m13_config_for(state, x.shape[0])
    if m13cfg is None or (cfg["local_start"], cfg["local_count"]) != (state["m13"]["start"], state["m13"]["count"]):
        return _m11_kernel_fn(data, state)
    return m13_run(x.contiguous(), data["topk_ids"], data["topk_weights"], state, m13cfg)
