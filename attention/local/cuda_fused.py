"""S1b: one CUDA kernel per decode layer (q_len 1): prologue + attention + merge + norm-gate.
Builds on A5b (cuda_dec2.py: whole-row loads, stacked heads, rows two chunks ahead) and adds:
- A23 prologue fold: each CTA computes its 16 query heads (RMSNorm x qk factor, interleaved RoPE
  on sliding layers) from qkv in registers. A lane's query values are bytes 32*t4 .. 32*t4+31 of
  the pair = head hl = t4 >> 1, d = 32*(t4 & 1) + 0..31: RoPE pairs (2i, 2i+1) stay inside the
  lane, the RMS sum needs one shuffle with lane t4 ^ 1. Queries are rounded to FP16 (finer than
  the BF16/FP8 the reference and SGLang use).
- The new token: the CTA whose slice holds position n-1 computes the pair's new key (RMSNorm,
  RoPE on sliding layers, / k_scale) and value (/ v_scale), rounds them to E4M3 (satfinite, as the
  Triton prologue does), stores them at cache row cache_loc[b], and patches them into its own
  fragments for token n-1 (from shared memory). No other CTA reads row n-1, so nothing waits.
- A25 merge: every CTA writes its partial (log2 max, sum, output); the last CTA of each (request,
  pair), by an atomic ticket, merges all slices, applies RMSNorm x sigmoid(gate), writes the BF16
  output and resets the ticket (graph-safe: tickets are back to 0 after every call).
RoPE frequencies come in as a tensor computed by torch exactly as reference.py does; angles use
full-range sincosf.
"""

CUDA_FUSED_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_fused {

constexpr int CHUNK = 16;
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
// Two FP32 -> E4M3 (satfinite, round to nearest even): lo -> low byte, hi -> high byte of 16 bits.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf16(float x) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}
__device__ __forceinline__ void set_word(uint4& v, int i, uint32_t x) {
  if (i == 0) v.x = x; else if (i == 1) v.y = x; else if (i == 2) v.z = x; else v.w = x;
}

struct Chunk {
  uint4 k[4];
  uint4 v[4];
};

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) fused_dec_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, float* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part, int* __restrict__ tickets, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ int last_flag;
  __shared__ float rope_c[32], rope_s[32];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3, hl = t4 >> 1;
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int pair = blockIdx.y % pairs;
  const int kv0 = pair * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  const float pos = static_cast<float>(positions[b]);

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // ---- RoPE table for this request's position: one full-range sincosf per frequency (the
  // arguments reach ~1e5 rad, CUDA's slow reduction path), shared by every lane.
  if (rope) {
    if (warp == 0) sincosf(pos * inv_freq[lane], &rope_s[lane], &rope_c[lane]);
    __syncthreads();
  }

  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.
  if (has_new && warp == 0) {
    const int h = lane >> 4, d0 = 4 * (lane & 15);          // head kv0 + h, dims d0 .. d0+3
    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;
    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;
    float k4[4], v4[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      k4[j] = bf16f(kp[j]);
      v4[j] = bf16f(vp[j]);
      ss += k4[j] * k4[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 8);                       // sum over the head's 16 lanes
    const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
    for (int j = 0; j < 4; ++j) k4[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const float sn = rope_s[d0 / 2 + p], cs = rope_c[d0 / 2 + p];
        const float e = k4[2 * p], o = k4[2 * p + 1];
        k4[2 * p] = e * cs - o * sn;
        k4[2 * p + 1] = e * sn + o * cs;
      }
    }
    const float kinv = 1.f / (*k_scale_p), vinv = 1.f / (*v_scale_p);
    const uint32_t kw = e4m3x2(k4[0] * kinv, k4[1] * kinv) | (e4m3x2(k4[2] * kinv, k4[3] * kinv) << 16);
    const uint32_t vw = e4m3x2(v4[0] * vinv, v4[1] * vinv) | (e4m3x2(v4[2] * vinv, v4[3] * vinv) << 16);
    reinterpret_cast<uint32_t*>(new_k)[lane] = kw;            // bytes 4*lane .. = head h, dims d0 ..
    reinterpret_cast<uint32_t*>(new_v)[lane] = vw;
    const long row = cache_loc[b];
    reinterpret_cast<uint32_t*>(kc + row * row_bytes + poff)[lane] = kw;
    reinterpret_cast<uint32_t*>(vc + row * row_bytes + poff)[lane] = vw;
  }

  // ---- queries: lane = head hl, dims 32*(t4&1) .. +31 of query g; RMSNorm x factor, RoPE, FP16.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
    float x[32];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      x[j] = bf16f(qp[j]);
      ss += x[j] * x[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)
    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];
        const float e = x[2 * p], o = x[2 * p + 1];
        x[2 * p] = e * cs - o * sn;
        x[2 * p + 1] = e * sn + o * cs;
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);
      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);
      qa[ks][0] = hl == 0 ? lo2 : 0u;
      qa[ks][1] = hl == 1 ? lo2 : 0u;
      qa[ks][2] = hl == 0 ? hi2 : 0u;
      qa[ks][3] = hl == 1 ? hi2 : 0u;
    }
  }
  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  auto row_of = [&](int c) -> long {
    const int s = min(start + c * CHUNK + (lane & 15), end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto load = [&](long my_row, Chunk& ch) {
    const long r0 = __shfl_sync(FULL, my_row, g);
    const long r1 = __shfl_sync(FULL, my_row, 8 + g);
    const long v0 = __shfl_sync(FULL, my_row, 2 * t4);
    const long v1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);
    const long v2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);
    const long v3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);
    const uint8_t* kb0 = kc + r0 * row_bytes + poff + 32 * t4;
    const uint8_t* kb1 = kc + r1 * row_bytes + poff + 32 * t4;
    ch.k[0] = __ldg(reinterpret_cast<const uint4*>(kb0));
    ch.k[2] = __ldg(reinterpret_cast<const uint4*>(kb1));
    ch.k[1] = __ldg(reinterpret_cast<const uint4*>(kb0 + 16));
    ch.k[3] = __ldg(reinterpret_cast<const uint4*>(kb1 + 16));
    ch.v[0] = __ldg(reinterpret_cast<const uint4*>(vc + v0 * row_bytes + poff + 16 * g));
    ch.v[1] = __ldg(reinterpret_cast<const uint4*>(vc + v1 * row_bytes + poff + 16 * g));
    ch.v[2] = __ldg(reinterpret_cast<const uint4*>(vc + v2 * row_bytes + poff + 16 * g));
    ch.v[3] = __ldg(reinterpret_cast<const uint4*>(vc + v3 * row_bytes + poff + 16 * g));
  };
  // Token n-1 of this chunk takes the new key/value from shared memory (the cache row may not be
  // written yet as far as this CTA's loads are concerned).
  auto patch = [&](int s0, Chunk& ch) {
    if (!has_new || n - 1 < s0 || n - 1 >= s0 + CHUNK) return;
    const int tn = n - 1 - s0;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      if (nt * 8 + g == tn) {
        ch.k[nt * 2] = *reinterpret_cast<const uint4*>(new_k + 32 * t4);
        ch.k[nt * 2 + 1] = *reinterpret_cast<const uint4*>(new_k + 32 * t4 + 16);
      }
    const int slot[4] = {2 * t4, 2 * t4 + 1, 2 * t4 + 8, 2 * t4 + 9};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (slot[i] == tn) ch.v[i] = *reinterpret_cast<const uint4*>(new_v + 16 * g);
  };

  Chunk cur, nxt;
  long row_next = 0, row_next2 = 0;
  int c = warp;
  if (c < nchunks) {
    load(row_of(c), cur);
    if (c + WARPS < nchunks) row_next = row_of(c + WARPS);
  }
  for (; c < nchunks; c += WARPS) {
    if (c + 2 * WARPS < nchunks) row_next2 = row_of(c + 2 * WARPS);
    if (c + WARPS < nchunks) load(row_next, nxt);
    const int s0 = start + c * CHUNK;
    patch(s0, cur);

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    cur = nxt;
    row_next = row_next2;
  }

  // ---- CTA merge of the warps (shared memory), then this slice's partial.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj) sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  const long part0 = (static_cast<long>(b) * hq + kv0 * 8) * nsplit;   // head kv0*8 + 8h + row, slice split
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const long part = part0 + static_cast<long>(h * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }

  // ---- A25: the last slice of this (request, pair) merges every slice and writes the output.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int ticket = atomicAdd(tickets + blockIdx.y, 1);
    last_flag = ticket == nsplit - 1;
  }
  __syncthreads();
  if (!last_flag) return;
  __threadfence();
  // CTA-wide merge of nsplit (<= 128, host-enforced) slices for the pair's 16 (head, row)s,
  // reusing sm_o: wts[16][128] slice weights, then outs[16][64]. Partials are read through L2
  // (ld.cg): other CTAs wrote them.
  float* wts = &sm_o[0][0][0][0];
  float* outs = wts + 16 * 128;
  __shared__ float row_l[16];
  for (int hr = warp; hr < 16; hr += WARPS) {
    const long pbase = part0 + static_cast<long>(hr) * nsplit;
    float mm = -1.0e30f;
    for (int sp = lane; sp < nsplit; sp += 32) mm = fmaxf(mm, __ldcg(m_part + pbase + sp));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));
    float l = 0.f;
    for (int sp = lane; sp < nsplit; sp += 32) {
      const float wt = ex2(__ldcg(m_part + pbase + sp) - mm);
      wts[hr * 128 + sp] = wt;
      l += wt * __ldcg(l_part + pbase + sp);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);
    if (lane == 0) row_l[hr] = l;
  }
  __syncthreads();
  constexpr int PER = 1024 / (WARPS * 32);                    // output elements per thread
  float acc_o[PER];
#pragma unroll
  for (int k = 0; k < PER; ++k) acc_o[k] = 0.f;
#pragma unroll 2
  for (int sp = 0; sp < nsplit; ++sp) {
#pragma unroll
    for (int k = 0; k < PER; ++k) {
      const int e = threadIdx.x + k * WARPS * 32, hr = e >> 6, d = e & 63;
      acc_o[k] += wts[hr * 128 + sp] * __ldcg(o_part + (part0 + static_cast<long>(hr) * nsplit + sp) * 64 + d);
    }
  }
#pragma unroll
  for (int k = 0; k < PER; ++k) outs[threadIdx.x + k * WARPS * 32] = acc_o[k];
  __syncthreads();
  const float vsc = *v_scale_p;
  for (int hr = warp; hr < 16; hr += WARPS) {                 // RMSNorm x sigmoid(gate), one warp per row
    const float scale = vsc / row_l[hr];
    const float o0 = outs[hr * 64 + lane] * scale, o1 = outs[hr * 64 + 32 + lane] * scale;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float r = rsqrtf(ss / 64.f + 1e-5f);
    const int head = kv0 * 8 + hr;
    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + head * 64;
    const float g0 = bf16f(gp[lane]), g1 = bf16f(gp[32 + lane]);
    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;
    op[lane] = f2bf16(o0 * r / (1.f + __expf(-g0)));
    op[32 + lane] = f2bf16(o1 * r / (1.f + __expf(-g1)));
  }
  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;              // ready for the next call (and replay)
}

}  // namespace okc_fused

void fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq,
               torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale,
               torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
               torch::Tensor l_part, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq,
               int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope,
               double qk_factor, double sm_scale, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_FUSED_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(),                     \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      tickets.data_ptr<int>(), reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq),           \
      static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), \
      static_cast<int>(window), static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)
  if (warps == 8) okc_fused::fused_dec_kernel<8><<<grid, 256, 0, stream>>>(OKC_FUSED_ARGS);
  else okc_fused::fused_dec_kernel<4><<<grid, 128, 0, stream>>>(OKC_FUSED_ARGS);
#undef OKC_FUSED_ARGS
}

std::vector<int64_t> fused_dec_attrs(int64_t warps) {
  cudaFuncAttributes a;
  if (warps == 8) cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<8>);
  else cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<4>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_FUSED_DECL = (
    "void fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
    "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
    "torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, torch::Tensor tickets, "
    "torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, "
    "int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps);\n"
    "std::vector<int64_t> fused_dec_attrs(int64_t warps);")
