"""S2a (A28): CUDA split-context attention for 8-token verify blocks (q_len 8, causal inside the
block), between the Triton prologue (q scratch, new cache rows) and the Triton _combine.

One CTA = (request b, KV-head pair, slice of keys), 4 warps. Each 16-key chunk's whole 128-byte
K and V rows are copied once into a STAGES-deep shared-memory ring with 16-byte cp.async (one K
and one V copy per thread per chunk; rows fetched one chunk ahead of their copies). Warp w owns
head kv0 + (w >> 1) and m-tiles 2*(w&1), 2*(w&1)+1: an m-tile's rows g / g+8 are query head g of
block tokens 2*mt / 2*mt+1, so all 64 rows per head are real (no stacking: verify can't afford
the 2x MMA work, AF17) and warps own disjoint rows (no intra-CTA merge).
Fragments come from shared memory with A5a's per-head byte permutation:
- K (B of QK, m16n8k16 f16): lane (g, t4) reads 16 bytes at 16*t4 of token g (tile 0) / 8+g
  (tile 1) in its head; logical k 2*t4+{0,1} (+8) of step ks <-> byte 16*t4 + 4*ks + {0,1} ({2,3}).
- V (B of PV): lane (g, t4) reads 8 bytes at 8*g of tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9;
  logical column g of d-tile j <-> byte 8*g + j. Output column 2*t4+jj of tile j -> d = 8*(2*t4+jj) + j.
Row pitches make both the 16 B K reads (8 lanes per phase: 2 tokens x 4 chunks) and the 8 B V
reads (16 lanes per phase) bank-conflict free: K pitch 192 B (consecutive tokens shift by 4
16-byte units), V pitch 144 B (by 2 8-byte units); the 16 B cp.async writes of a row stay
conflict-free under both. Causal mask: key <= n - 8 + token; sliding window: key > that - window.
"""

CUDA_VER_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_ver {

constexpr int CHUNK = 16;
constexpr int L = 8;
constexpr int KP = 192;   // K row pitch in shared memory (bytes)
constexpr int VP = 144;   // V row pitch
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
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ uint32_t w4(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

template <int STAGES>
__global__ void __launch_bounds__(128) attend_ver_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  __shared__ __align__(16) uint8_t sk[STAGES][CHUNK * KP];
  __shared__ __align__(16) uint8_t sv[STAGES][CHUNK * VP];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int h = warp >> 1, mt0 = 2 * (warp & 1);
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;

  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l
  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Query fragments: m-tile mt = mt0 + mi, rows g -> token 2*mt, g+8 -> token 2*mt+1, head (kv0+h)*8+g.
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int ta = 2 * (mt0 + mi);
    const uint16_t* qA = q + ((static_cast<long>(b) * L + ta) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;
    const uint16_t* qB = qA + static_cast<long>(hq) * 64;               // token ta + 1
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[mi][ks][0] = pack_f16x2(bf16f(qA[4 * ks + 0]), bf16f(qA[4 * ks + 1]));
      qa[mi][ks][1] = pack_f16x2(bf16f(qB[4 * ks + 0]), bf16f(qB[4 * ks + 1]));
      qa[mi][ks][2] = pack_f16x2(bf16f(qA[4 * ks + 2]), bf16f(qA[4 * ks + 3]));
      qa[mi][ks][3] = pack_f16x2(bf16f(qB[4 * ks + 2]), bf16f(qB[4 * ks + 3]));
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  const int ctok = tid >> 3, cch = tid & 7;          // this thread's copy: token ctok, 16 B chunk cch
  auto row_of = [&](int c) -> long {
    const int s = min(start + c * CHUNK + ctok, end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto issue = [&](int c, long row) {
    const int st = c % STAGES;
    cp16(&sk[st][ctok * KP + 16 * cch], kc + row * row_bytes + poff + 16 * cch);
    cp16(&sv[st][ctok * VP + 16 * cch], vc + row * row_bytes + poff + 16 * cch);
  };

  float m_run[2][2], l_run[2][2], acc[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    m_run[mi][0] = m_run[mi][1] = -1.0e30f;
    l_run[mi][0] = l_run[mi][1] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  }

#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) {
    if (i < nchunks) issue(i, row_of(i));
    asm volatile("cp.async.commit_group;");
  }
  long rnext = STAGES - 1 < nchunks ? row_of(STAGES - 1) : 0;

  for (int c = 0; c < nchunks; ++c) {
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));
    __syncthreads();
    if (c + STAGES - 1 < nchunks) issue(c + STAGES - 1, rnext);
    asm volatile("cp.async.commit_group;");
    if (c + STAGES < nchunks) rnext = row_of(c + STAGES);

    const int st = c % STAGES;
    const int s0 = start + c * CHUNK;
    const uint4 kb0 = *reinterpret_cast<const uint4*>(&sk[st][g * KP + 64 * h + 16 * t4]);
    const uint4 kb1 = *reinterpret_cast<const uint4*>(&sk[st][(8 + g) * KP + 64 * h + 16 * t4]);
    const uint2 vb0 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4) * VP + 64 * h + 8 * g]);
    const uint2 vb1 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 1) * VP + 64 * h + 8 * g]);
    const uint2 vb2 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 8) * VP + 64 * h + 8 * g]);
    const uint2 vb3 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 9) * VP + 64 * h + 8 * g]);
    // a chunk needs masks only at the slice tail or where it reaches the block's own keys
    const bool masked = s0 + CHUNK > end || s0 + CHUNK > n - L || window > 0;

#pragma unroll
    for (int mi = 0; mi < 2; ++mi) {
      float sc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        const uint4& kb = nt == 0 ? kb0 : kb1;
        sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 4; ++ks) {
          const uint32_t w = w4(kb, ks);
          mma16816(sc[nt], qa[mi][ks][0], qa[mi][ks][1], qa[mi][ks][2], qa[mi][ks][3], f16x2_lo(w), f16x2_hi(w));
        }
      }
      const int posA = n - L + 2 * (mt0 + mi), posB = posA + 1;     // rows g / g+8
      float mxA = -INFINITY, mxB = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int key = s0 + nt * 8 + 2 * t4 + jj;
          float xa = sc[nt][jj] * qk_scale, xb = sc[nt][2 + jj] * qk_scale;
          if (masked) {
            const bool ok = key < end;
            const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);
            const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);
            xa = okA ? xa : -INFINITY;
            xb = okB ? xb : -INFINITY;
          }
          sc[nt][jj] = xa;
          sc[nt][2 + jj] = xb;
          mxA = fmaxf(mxA, xa);
          mxB = fmaxf(mxB, xb);
        }
      mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));
      mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));
      mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));
      mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));
      const float mnA = fmaxf(m_run[mi][0], mxA), mnB = fmaxf(m_run[mi][1], mxB);
      const float alA = ex2(m_run[mi][0] - mnA), alB = ex2(m_run[mi][1] - mnB);
      float p[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        p[nt][0] = ex2(sc[nt][0] - mnA);
        p[nt][1] = ex2(sc[nt][1] - mnA);
        p[nt][2] = ex2(sc[nt][2] - mnB);
        p[nt][3] = ex2(sc[nt][3] - mnB);
      }
      l_run[mi][0] = l_run[mi][0] * alA + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
      l_run[mi][1] = l_run[mi][1] * alB + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
      m_run[mi][0] = mnA;
      m_run[mi][1] = mnB;
      const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);
      const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);
      const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);
      const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[mi][j][0] *= alA;
        acc[mi][j][1] *= alA;
        acc[mi][j][2] *= alB;
        acc[mi][j][3] *= alB;
        const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
        const uint32_t b0 = f16x2_lo(__byte_perm(j < 4 ? vb0.x : vb0.y, j < 4 ? vb1.x : vb1.y, sel));
        const uint32_t b1 = f16x2_lo(__byte_perm(j < 4 ? vb2.x : vb2.y, j < 4 ? vb3.x : vb3.y, sel));
        mma16816(acc[mi][j], pa0, pa1, pa2, pa3, b0, b1);
      }
    }
  }
  asm volatile("cp.async.wait_all;");

  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4+jj) + j.
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float l = l_run[mi][r];
      l += __shfl_xor_sync(FULL, l, 1);
      l += __shfl_xor_sync(FULL, l, 2);
      const int tok = 2 * (mt0 + mi) + r;
      const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) o_part[part * 64 + 8 * (2 * t4 + jj) + j] = acc[mi][j][2 * r + jj];
      if (t4 == 0) {
        m_part[part] = m_run[mi][r];
        l_part[part] = l;
      }
    }
}

}  // namespace okc_ver

void attend_ver(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                int64_t page, int64_t window, double sm_scale, int64_t stages) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_VER_ARGS                                                                                         \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (stages == 4) okc_ver::attend_ver_kernel<4><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);
  else if (stages == 6) okc_ver::attend_ver_kernel<6><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);
  else okc_ver::attend_ver_kernel<3><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);
#undef OKC_VER_ARGS
}

std::vector<int64_t> attend_ver_attrs(int64_t stages) {
  cudaFuncAttributes a;
  if (stages == 4) cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<4>);
  else if (stages == 6) cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<6>);
  else cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_VER_DECL = ("void attend_ver(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                 "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                 "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                 "int64_t page, int64_t window, double sm_scale, int64_t stages);\n"
                 "std::vector<int64_t> attend_ver_attrs(int64_t stages);")
