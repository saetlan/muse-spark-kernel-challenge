"""A5: CUDA split-context decode attention (q_len 1), source + host launcher. Kept in one place
so the lab blocks and, later, the submission include the same text (make_lab.py appends it).

Layout trick: the dot products are invariant to permuting the head dimension d consistently, so
each lane's mma.sync fragment values are made contiguous in memory:
- QK (m16n8k16 f16, A = queries [16 rows = 8 heads of the group + 8 zero rows, 16 d], B = keys):
  logical k column 2*t4 + {0,1} (+8) of k-step ks  <->  physical byte 16*t4 + 4*ks + {0,1} ({2,3})
  so lane (g, t4) loads 16 contiguous bytes of key row g (one 16 B load per head per 8 tokens).
- PV (A = P [rows, 16 tokens] from the QK accumulators, B = values [16 tokens, 8 d]):
  logical n column g of d-tile j  <->  physical byte 8*g + j
  so lane (g, t4) loads 8 contiguous bytes of value rows 2*t4, 2*t4+1, 2*t4+8, 2*t4+9.
FP8 E4M3 -> FP16 is exact (cvt.rn.f16x2.e4m3x2, sm_89+); queries are rounded to FP16 (finer than
the FP8 the rules allow); P is FP16; accumulation FP32. Each CTA: one request, a pair of KV heads
(whole 128 B cache rows), one slice of keys; its 4 warps take 16-token chunks round-robin, keep
their own online softmax, prefetch their next chunk into registers, and merge through shared
memory at the end. Partials (log2-domain max, sum, output) match _combine's layout.
"""

CUDA_DEC_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_dec {

constexpr int CHUNK = 16;
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2_lo(uint32_t w) {
  // Low 16 bits of w hold two E4M3 bytes; the low byte becomes the low half.
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));   // a -> upper half, b -> lower
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attend_dec_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;

  const int n = seqlens[b];                          // keys 0 .. n-1; the query sits at n-1
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Query A fragments (rows g; rows g+8 are zero), d permuted as in the module docstring.
  uint32_t qa[2][4][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[h][ks][0] = pack_f16x2(bf16_bits_to_f32(qp[4 * ks + 0]), bf16_bits_to_f32(qp[4 * ks + 1]));
      qa[h][ks][1] = pack_f16x2(bf16_bits_to_f32(qp[4 * ks + 2]), bf16_bits_to_f32(qp[4 * ks + 3]));
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[2][8][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[h][j][i] = 0.f;

  uint4 kr[2][2], kn[2][2];
  uint2 vr[2][4], vn[2][4];

  // Loads chunk c of this CTA's slice into (k, v). Tokens past `end` read the last valid row
  // (always in bounds) and are masked out of the softmax.
  auto load = [&](int c, uint4 (&k)[2][2], uint2 (&v)[2][4]) {
    const int s = min(start + c * CHUNK + (lane & 15), end - 1);
    const long my_row = static_cast<long>(pt[s / page]) * page + s % page;
    const long rk0 = __shfl_sync(FULL, my_row, g);
    const long rk1 = __shfl_sync(FULL, my_row, 8 + g);
    const long rv0 = __shfl_sync(FULL, my_row, 2 * t4);
    const long rv1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);
    const long rv2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);
    const long rv3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long hoff = (kv0 + h) * 64;
      k[h][0] = __ldg(reinterpret_cast<const uint4*>(kc + rk0 * row_bytes + hoff + 16 * t4));
      k[h][1] = __ldg(reinterpret_cast<const uint4*>(kc + rk1 * row_bytes + hoff + 16 * t4));
      v[h][0] = __ldg(reinterpret_cast<const uint2*>(vc + rv0 * row_bytes + hoff + 8 * g));
      v[h][1] = __ldg(reinterpret_cast<const uint2*>(vc + rv1 * row_bytes + hoff + 8 * g));
      v[h][2] = __ldg(reinterpret_cast<const uint2*>(vc + rv2 * row_bytes + hoff + 8 * g));
      v[h][3] = __ldg(reinterpret_cast<const uint2*>(vc + rv3 * row_bytes + hoff + 8 * g));
    }
  };

  if (warp < nchunks) load(warp, kr, vr);
  for (int c = warp; c < nchunks; c += WARPS) {
    if (c + WARPS < nchunks) load(c + WARPS, kn, vn);
    const int s0 = start + c * CHUNK;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float sc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        const uint32_t w[4] = {kr[h][nt].x, kr[h][nt].y, kr[h][nt].z, kr[h][nt].w};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          mma16816(sc[nt], qa[h][ks][0], 0u, qa[h][ks][1], 0u, f16x2_from_e4m3x2_lo(w[ks]),
                   f16x2_from_e4m3x2_hi(w[ks]));
      }
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int tok = s0 + nt * 8 + 2 * t4 + jj;
          const float x = tok < end ? sc[nt][jj] * qk_scale : -INFINITY;
          sc[nt][jj] = x;
          mx = fmaxf(mx, x);
        }
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
      const float m_new = fmaxf(m_run[h], mx);
      const float alpha = ex2(m_run[h] - m_new);
      const float p00 = ex2(sc[0][0] - m_new), p01 = ex2(sc[0][1] - m_new);
      const float p10 = ex2(sc[1][0] - m_new), p11 = ex2(sc[1][1] - m_new);
      l_run[h] = l_run[h] * alpha + (p00 + p01 + p10 + p11);
      m_run[h] = m_new;
      const uint32_t pa0 = pack_f16x2(p00, p01);     // row g, tokens 2*t4, 2*t4+1
      const uint32_t pa2 = pack_f16x2(p10, p11);     // row g, tokens 8+2*t4, 9+2*t4
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[h][j][0] *= alpha;
        acc[h][j][1] *= alpha;
        const uint32_t w0 = j < 4 ? vr[h][0].x : vr[h][0].y;
        const uint32_t w1 = j < 4 ? vr[h][1].x : vr[h][1].y;
        const uint32_t w2 = j < 4 ? vr[h][2].x : vr[h][2].y;
        const uint32_t w3 = j < 4 ? vr[h][3].x : vr[h][3].y;
        const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
        const uint32_t b0 = f16x2_from_e4m3x2_lo(__byte_perm(w0, w1, sel));   // tokens 2*t4, 2*t4+1
        const uint32_t b1 = f16x2_from_e4m3x2_lo(__byte_perm(w2, w3, sel));   // tokens 2*t4+8, 2*t4+9
        mma16816(acc[h][j], pa0, 0u, pa2, 0u, b0, b1);
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) kr[h][nt] = kn[h][nt];
#pragma unroll
      for (int i = 0; i < 4; ++i) vr[h][i] = vn[h][i];
    }
  }

  // Merge the warps: each row's state to shared memory (output d un-permuted), then all threads.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = l_run[h];
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
    if (t4 == 0) {
      sm_m[warp][h][g] = m_run[h];
      sm_l[warp][h][g] = l;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sm_o[warp][h][g][8 * (2 * t4) + j] = acc[h][j][0];
      sm_o[warp][h][g][8 * (2 * t4 + 1) + j] = acc[h][j][1];
    }
  }
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
    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }
}

}  // namespace okc_dec

void attend_dec(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                int64_t page, int64_t window, double sm_scale, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_DEC_ARGS                                                                                         \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (warps == 8) okc_dec::attend_dec_kernel<8><<<grid, 256, 0, stream>>>(OKC_DEC_ARGS);
  else okc_dec::attend_dec_kernel<4><<<grid, 128, 0, stream>>>(OKC_DEC_ARGS);
#undef OKC_DEC_ARGS
}

std::vector<int64_t> attend_dec_attrs(int64_t warps) {
  cudaFuncAttributes a;
  if (warps == 8) cudaFuncGetAttributes(&a, okc_dec::attend_dec_kernel<8>);
  else cudaFuncGetAttributes(&a, okc_dec::attend_dec_kernel<4>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_DEC_DECL = ("void attend_dec(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                 "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                 "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                 "int64_t page, int64_t window, double sm_scale, int64_t warps);\n"
                 "std::vector<int64_t> attend_dec_attrs(int64_t warps);")
