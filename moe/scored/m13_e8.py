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
M14_INSTANCES = ((1, 2, 2), (1, 2, 4), (1, 2, 8), (0, 1, 4), (0, 1, 8), (0, 2, 2), (0, 2, 4), (0, 2, 8))
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

// A load of data the previous grid wrote: volatile asm with a memory clobber so the compiler can't move
// it above griddepcontrol.wait (E5: __ldg's ld.global.nc is treated as invariant and CUDA's __ldcg is
// an asm without a clobber, so both may be hoisted, and every PDL config produced garbage).
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p) : "memory");
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
// MATH = false (recon only): the same loads, cursor and flushes, but the loaded words are only XORed
// together, to measure the memory pipeline without the decode and MMAs.
template <int TT, int NT, bool GATED, bool MATH = true>
__global__ void __launch_bounds__(128) sk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
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
  Cur L;  // the group whose loads are issued next
  {
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_base[ai], per_task = L.passes * GPT;
    L.task = local / per_task;
    const int rem = local - L.task * per_task;
    L.pass = rem / GPT;
    L.grp = rem - L.pass * GPT;
    set_cntp(L);
  }

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
        ra[s][n][0] = ld_cg4(la[n] + s * 8);
        ra[s][n][1] = ld_cg4(la[n] + s * 8 + 1);
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
        const uint32_t a0 = hmul2(rotr(q, 2) & PM, sg);
        const uint32_t a1 = hmul2(rotr(q, 6) & PM, sg8);
        const uint32_t a2 = hmul2(rotr(q, 12) & PM, sg);
        const uint32_t a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (n * 8 < cntp) {
            const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
            mma(acc[i][n], a0, a1, a2, a3, b0, b1);
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
  if (pdl) {
    // Weights don't depend on the previous grid: ask L2 for the next pf groups before waiting.
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
#pragma unroll
  for (int s = 0; s < D; ++s) load_a(s);

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = pf >> 8;
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
  uint32_t sink = 0u;
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
      if constexpr (MATH) {
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
  if constexpr (!MATH) {
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
  if (pdl) pdl_wait();
  float v[TOP_K][8];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    ss[k] = 0.0f;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (loc[k]) r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tok * TOP_K + k) * K) + tid);
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
constexpr int SB = 3;                    // activation stages (K chunks) in shared memory

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

template <int TT, int NT, bool GATED>
__global__ void __launch_bounds__(128) ck_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  extern __shared__ __align__(128) unsigned char s_b[];  // [SB][NB pairs][128 B]
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
        A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
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
  load_b(L, 0, 0);
  cp_commit();
  load_b(L, 1, 1);
  cp_commit();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = pf >> 8;
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
  int chunk = 0;  // chunks consumed; activation stage = chunk % SB
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
      cp_wait1();
      __syncthreads();  // chunk `chunk` is in shared memory for all; everyone is done with chunk - 1's stage
      const int ahead = (chunk + 2) % SB;
      if (s < 2) {
        load_b(C, s + 2, ahead);
      } else if (more) {
        load_b(L, s - 2, ahead);
      }
      cp_commit();
      compute(s, rsc, cntp, chunk % SB);
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      ++chunk;
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m14


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

#define M13_CASE(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    m13_launch(m13::sk_kernel<TT, NT, G>, static_cast<int>(grid), 128, 0, static_cast<int>(pdl),          \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
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
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_gemm: ", n_pairs, " pairs, at most ", m13::MAXPAIRS);
  @@M13_CASES@@
  TORCH_CHECK(false, "m13_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

#define M13_RECON_CASE(G, TT, NT)                                                                           \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    m13_launch(m13::sk_kernel<TT, NT, G, false>, static_cast<int>(grid), 128, 0, 0,                        \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), 0, 0);                            \
    return out;                                                                                            \
  }

// The M13 GEMM with MATH = false (loads, cursor and flushes only), for recon labs.
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_recon: too many pairs");
  M13_RECON_CASE(true, 2, 1)
  M13_RECON_CASE(false, 2, 1)
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

#define M14_CASE(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    m13_launch(m14::ck_kernel<TT, NT, G>, static_cast<int>(grid), 128,                                     \
               static_cast<size_t>(m14::SB * m14::stage_bytes<NT>()), static_cast<int>(pdl),               \
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
// [gated, tt, nt, registers, spill bytes, static smem, dynamic smem, CTAs per SM].
#define M14_SETUP(G, TT, NT)                                                                                \
  {                                                                                                        \
    const int dyn = m14::SB * m14::stage_bytes<NT>();                                                      \
    cudaFuncSetAttribute(m14::ck_kernel<TT, NT, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);    \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m14::ck_kernel<TT, NT, G>);                                               \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m14::ck_kernel<TT, NT, G>, 128, dyn);          \
    const int64_t row[8] = {(G) ? 1 : 0, TT, NT, static_cast<int64_t>(attr.numRegs),                       \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_gemm: ", n_pairs, " pairs, at most ", 8 * m14::MAXP);
  @@M14_CASES@@
  TORCH_CHECK(false, "m14_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m14_setup() {
  std::vector<int64_t> rows;
  @@M14_SETUPS@@
  return torch::tensor(rows).view({-1, 8});
}
"""

_M13_CPP = """
torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt);
torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl);
torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m14_setup();
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid);
torch::Tensor m13_stream(torch::Tensor base, torch::Tensor experts, int64_t len16, torch::Tensor sink, int64_t unroll,
                         int64_t grid);
"""


def m13_source(instances=M13_INSTANCES, m11_instances=M13_M11_INSTANCES, m14_instances=M14_INSTANCES):
    """M11's CUDA (only the instances M11_CONFIGS uses, no M12) followed by M13's and M14's."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in instances]
    args14 = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in m14_instances]
    m13 = (_M13_CUDA.replace("@@M13_CASES@@", "\n  ".join(f"M13_CASE({a})" for a in args))
           .replace("@@M13_ATTRS@@", "\n  ".join(f"M13_ATTR({a})" for a in args))
           .replace("@@M13_MAXP@@", str(M13_MAX_TOKENS))
           .replace("@@M13_MAXPAIRS@@", str(M13_MAX_TOKENS * TOP_K))
           .replace("@@M14_MAXP@@", str(M14_MAX_TOKENS))
           .replace("@@M14_CASES@@", "\n  ".join(f"M14_CASE({a})" for a in args14))
           .replace("@@M14_SETUPS@@", "\n  ".join(f"M14_SETUP({a})" for a in args14)))
    return m11_source(m11_instances, ()) + m13


def m13_build():
    """One extension with M11's and M13's kernels (sm_90a only, as m11_build)."""
    return load_inline(
        name="m13_ext", cpp_sources=_M11_CPP + _M13_CPP, cuda_sources=m13_source(),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup", "m13_recon", "m13_stream"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m13_"), verbose=False)


def m13_relayout(words):
    """M11 code words (int32, any shape) -> M13's bit layout, a lossless permutation of each word's bits."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for new, old in enumerate(M13_BIT_SRC):
        out |= ((w >> old) & 1) << new
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


def m13_scales_ok(weights, start, count):
    """True when every local E4M3 block scale is positive and normal (no zero, subnormal or NaN),
    which scale2's bit trick needs; M13 stays off otherwise."""
    for name in ("w13_sf", "w2_sf"):
        b = weights[name][start:start + count].view(torch.uint8)
        bad = ((b & 0x80) != 0) | ((b & 0x78) == 0) | ((b & 0x7F) == 0x7F)
        if bool(bad.any()):
            return False
    return True


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


def m13_gemm1(x, ids, state, s1):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair."""
    kind = s1.get("kind")
    if kind not in ("m13", "m14"):
        return m11_gemm1(x, ids, state, s1)
    bank, n_pairs = state["m13"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    gemm = state["m11_ext"].m13_gemm if kind == "m13" else state["m11_ext"].m14_gemm
    gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, bank["scr13"], bank["ctr13"],
                              n_pairs, bank["start"], bank["rt_w13"], 0, bank["inter"], 1, 2, s1["nt"],
                              s1["c"] * state["sms"], 0, s1.get("pfd", 0) << 8)
    return act


def m13_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    kind = s2.get("kind")
    if kind not in ("m13", "m14"):
        return m11_gemm2(act, ids, state, s2)
    bank, n_pairs = state["m13"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    gemm = state["m11_ext"].m13_gemm if kind == "m13" else state["m11_ext"].m14_gemm
    gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, bank["scr2"], bank["ctr2"],
                              n_pairs, bank["start"], bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"],
                              s2["c"] * state["sms"], s2.get("pdl", 0), s2.get("pf", 0) | (s2.get("pfd", 0) << 8))
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
    act = m13_gemm1(x, ids, state, cfg["s1"])
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
    state["m11_ext"] = m13_build()
    state["m12_resources"] = []
    # M14's dynamic shared-memory limits, set once here and never during a capture
    state["m14_resources"] = state["m11_ext"].m14_setup().tolist()
    state["m11"] = m11_bank(weights, 0, 32)
    state["m13_ok"] = m13_scales_ok(weights, 0, 32)
    if state["m13_ok"]:
        state["m13"] = m13_bank(state["m11"], weights["w13"].device)
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
