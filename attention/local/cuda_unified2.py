"""UD2: UD (cuda_unified.py) with the page-table ring of make_pt_ring.py (generated; see its
docstring): page entries carried by the cp.async ring in the K tiles' row pads, the warp
merge in the ring's shared memory, and the ring prologue issued before the query fold, RoPE
table and new row so those loads overlap the K/V round trip. Math, chunk assignment, new-row
patch and cluster merge are UD's (emulate_cuda_ud.py applies).
"""

CUDA_UD2_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_ud2 {

constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}

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

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

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

__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint16_t f2bf16(float x) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_peer(const float* local, uint32_t rank) {   // DSMEM load
  uint32_t a, r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a));
  (void)r;
  return v;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(WARPS * 32) ud2_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, float* __restrict__ cpart_o, float* __restrict__ cpart_m,
    float* __restrict__ cpart_l, int* __restrict__ tickets, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {
  __shared__ float cta_m[16], cta_l[16];          // this CTA's partial, read by the cluster leader
  __shared__ float cta_o[16][64];
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ float rope_c[32], rope_s[32];
  __shared__ int last_flag;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  const float pos = static_cast<float>(positions[b]);
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token

  // ---- ring prologue first (page entries, then the first K/V stages): the fold below
  // overlaps their round trips.
  const int* pt = page_table + static_cast<long>(b) * max_pages;
  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
  // Page entries ride the ring: chunk j's 16 entries sit in PT slot p = j % PT_SLOTS, in the unused
  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at
  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.
  constexpr int PT_SLOTS = 2 * STAGES;
  auto pt_slot = [&](int j, int e) -> int* {
    const int p = j % PT_SLOTS;
    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));
  };
  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j's page entries (join the open commit group)
    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);
  };
  // One commit group per call: chunk i's K/V rows (its page entries landed in an earlier group,
  // waited and warp-synced) and chunk i + STAGES's page entries.
  auto issue = [&](int i) {
    if (i < mine) {
      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    issue_pt(i + STAGES);
    asm volatile("cp.async.commit_group;");
  };
  // Prologue: the first STAGES chunks' page entries (the one exposed round trip), then the first
  // STAGES - 1 stages of K/V rows.
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);

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

  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const int c = warp + i * WARPS;
    uint8_t* stw = ring + (i % STAGES) * STAGE_BYTES;
    if (has_new && n - 1 >= start + c * CHUNK && n - 1 < start + (c + 1) * CHUNK) {
      // the new token's row: take the key/value computed in-kernel (the copy read the cache row
      // before or while it was being written)
      const int tn = n - 1 - (start + c * CHUNK);
      reinterpret_cast<uint32_t*>(stw + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_k)[lane];
      reinterpret_cast<uint32_t*>(stw + CHUNK * PITCH + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_v)[lane];
      __syncwarp();
    }
    const uint8_t* st = stw;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

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
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
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
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
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
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
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
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
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
    const int hr = h * 8 + row;                    // pair-local (head, row) index 0..15
    cta_o[hr][d] = o;
    if (d == 0) {
      cta_m[hr] = mm;
      cta_l[hr] = l;
    }
  }

  // ---- A42 cluster merge: the leader (rank 0) reads every rank's partial over DSMEM.
  const uint32_t rank = cluster_rank();
  const int nc = nsplit / CS;                       // clusters per (request, pair)
  const int ci = split / CS;
  cluster_sync();                                   // every CTA's partial written and visible
  constexpr int RPW = 16 / WARPS;                   // (head, row)s per leader warp
  float fo0[RPW], fo1[RPW], fl[RPW], fm[RPW];
  if (rank == 0) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) {
      const int hr = warp + k2 * WARPS;
      fm[k2] = -1.0e30f; fl[k2] = 0.f; fo0[k2] = 0.f; fo1[k2] = 0.f;
      if (hr < 16) {
        float mm = -1.0e30f;
#pragma unroll
        for (int r = 0; r < CS; ++r) mm = fmaxf(mm, ld_peer(&cta_m[hr], r));
        float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int r = 0; r < CS; ++r) {
          const float wt = ex2(ld_peer(&cta_m[hr], r) - mm);
          l += wt * ld_peer(&cta_l[hr], r);
          o0 += wt * ld_peer(&cta_o[hr][lane], r);
          o1 += wt * ld_peer(&cta_o[hr][32 + lane], r);
        }
        fm[k2] = mm; fl[k2] = l; fo0[k2] = o0; fo1[k2] = o1;
      }
    }
  }
  cluster_sync();                                   // the leader is done reading: peers may exit
  if (rank != 0) return;

  const float vsc = *v_scale_p;
  auto finalize = [&](int hr, float o0, float o1, float l) {   // RMSNorm x sigmoid(gate), BF16 store
    const float scale = vsc / l;
    o0 *= scale;
    o1 *= scale;
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
  };
  if (nc == 1) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);
    return;
  }

  // ---- A43 second level: cluster partials in global memory; the last leader (ticket) merges.
  const long cbase = (static_cast<long>(b) * hq + kv0 * 8) * nc;      // (head kv0*8 + hr, cluster)
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    const int hr = warp + k2 * WARPS;
    {
      const long p = cbase + static_cast<long>(hr) * nc + ci;
      cpart_o[p * 64 + lane] = fo0[k2];
      cpart_o[p * 64 + 32 + lane] = fo1[k2];
      if (lane == 0) {
        cpart_m[p] = fm[k2];
        cpart_l[p] = fl[k2];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_flag = atomicAdd(tickets + blockIdx.y, 1) == nc - 1;
  __syncthreads();
  if (!last_flag) return;
  __threadfence();
  for (int hr = warp; hr < 16; hr += WARPS) {       // nc <= 64: one warp per (head, row), lanes over clusters
    const long pb = cbase + static_cast<long>(hr) * nc;
    float mm = -1.0e30f;
    for (int c2 = lane; c2 < nc; c2 += 32) mm = fmaxf(mm, __ldcg(cpart_m + pb + c2));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll 4
    for (int c2 = 0; c2 < nc; ++c2) {
      const float wt = ex2(__ldcg(cpart_m + pb + c2) - mm);
      l += wt * __ldcg(cpart_l + pb + c2);
      o0 += wt * __ldcg(cpart_o + (pb + c2) * 64 + lane);
      o1 += wt * __ldcg(cpart_o + (pb + c2) * 64 + 32 + lane);
    }
    finalize(hr, o0, o1, l);
  }
  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)
}

}  // namespace okc_ud2

void ud2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,
        torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,
        torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out,
        int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window,
        int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_ud2::STAGE_BYTES;
#define OKC_UD2_ARGS                                                                                           \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \
      cpart_o.data_ptr<float>(), cpart_m.data_ptr<float>(), cpart_l.data_ptr<float>(), tickets.data_ptr<int>(), \
      reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),             \
      static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), \
      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)
  if (cs == 16) {
    if (warps == 8) okc_ud2::ud2_kernel<8, 2, 16><<<grid, 256, smem, stream>>>(OKC_UD2_ARGS);
    else okc_ud2::ud2_kernel<4, 3, 16><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
  } else {
    if (warps == 8) okc_ud2::ud2_kernel<8, 2, 8><<<grid, 256, smem, stream>>>(OKC_UD2_ARGS);
    else if (stages == 4) okc_ud2::ud2_kernel<4, 4, 8><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
    else okc_ud2::ud2_kernel<4, 3, 8><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
  }
#undef OKC_UD2_ARGS
  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later
  TORCH_CHECK(e == cudaSuccess, "ud2 launch failed: ", cudaGetErrorString(e));
}

int64_t ud2_setup() {   // once, outside any capture: each variant's ring, and 16-CTA clusters
  int err = 0;
  using okc_ud2::STAGE_BYTES;
#define OKC_UD2_SMEM(W, S, C) \
  err |= cudaFuncSetAttribute(okc_ud2::ud2_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES)
  OKC_UD2_SMEM(4, 3, 8);
  OKC_UD2_SMEM(4, 4, 8);
  OKC_UD2_SMEM(8, 2, 8);
  OKC_UD2_SMEM(4, 3, 16);
  OKC_UD2_SMEM(8, 2, 16);
#undef OKC_UD2_SMEM
  err |= cudaFuncSetAttribute(okc_ud2::ud2_kernel<4, 3, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  err |= cudaFuncSetAttribute(okc_ud2::ud2_kernel<8, 2, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> ud2_attrs(int64_t warps, int64_t stages, int64_t cs) {
  cudaFuncAttributes a;
  if (cs == 16) {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud2::ud2_kernel<8, 2, 16>);
    else cudaFuncGetAttributes(&a, okc_ud2::ud2_kernel<4, 3, 16>);
  } else {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud2::ud2_kernel<8, 2, 8>);
    else if (stages == 4) cudaFuncGetAttributes(&a, okc_ud2::ud2_kernel<4, 4, 8>);
    else cudaFuncGetAttributes(&a, okc_ud2::ud2_kernel<4, 3, 8>);
  }
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_UD2_DECL = (
    "void ud2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
    "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
    "torch::Tensor seqlens, torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, "
    "torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, "
    "int64_t nsplit, int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, "
    "int64_t warps, int64_t stages, int64_t cs);\n"
    "int64_t ud2_setup();\n"
    "std::vector<int64_t> ud2_attrs(int64_t warps, int64_t stages, int64_t cs);")
