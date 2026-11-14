"""Instruction-throughput microbenchmark for W1's softmax (lab only; hand-written, no generator).

W1's consumer spends each 64-key stage on 8192 exp2 and 4096 FP8 pack conversions per SM (AF43 / A-W2):
on Hopper both may run on the 16 / clk / SM special-function path, which would make the softmax, not the
tensor cores, the stage floor. Each OP below runs 8 independent chains per thread (256 threads, 4 CTAs per
SM); the lab times one launch per OP with CUDA events and reports ops / clk / SM relative to FFMA (taken as
128 / clk / SM), so no clock reading is needed.
  0 FFMA f32 (calibration)          1 ex2.approx.ftz.f32            2 ex2.approx.f16x2 (2 values)
  3 ex2.approx.ftz.bf16x2 (2 values) 4 cvt.rn.satfinite.e4m3x2.f32  5 cvt.rn.satfinite.e4m3x2.f16x2
  6 cvt.rn.f16x2.f32                 7 fma.rn.f16x2                  8 W1 mix: 2 ex2.f32 + 1 cvt e4m3x2.f32
  9 ex2.f16x2 + cvt e4m3x2.f16x2    10 FMNMX f32                    11 PRMT
Inputs only change through cheap ALU ops (FADD / IADD at >= 64 / clk) so the measured op dominates."""

CUDA_XU_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace okc_xu {

template <int OP>
__global__ void __launch_bounds__(256) xu_kernel(float* __restrict__ out, int iters, float seed) {
  float f[8];
  uint32_t h[8];
  uint32_t acc = 0;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    f[i] = seed * static_cast<float>(threadIdx.x + 3 * i + 1) * 1.0e-3f;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(h[i]) : "f"(-0.5f * f[i]), "f"(-0.25f * f[i]));
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if constexpr (OP == 0) {
        f[i] = fmaf(f[i], 0.9999f, 1.0e-4f);
      } else if constexpr (OP == 1) {
        asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[i]));
      } else if constexpr (OP == 2) {
        asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(h[i]));
      } else if constexpr (OP == 3) {
        asm volatile("ex2.approx.ftz.bf16x2 %0, %0;" : "+r"(h[i]));
      } else if constexpr (OP == 4) {
        uint16_t t;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(t) : "f"(f[i]), "f"(f[(i + 1) & 7]));
        acc += t;
        f[i] += 1.0f;
      } else if constexpr (OP == 5) {
        uint16_t t;
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(t) : "r"(h[i]));
        acc += t;
        h[i] += 0x00010001u;
      } else if constexpr (OP == 6) {
        uint32_t t;
        asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(t) : "f"(f[i]), "f"(f[(i + 1) & 7]));
        acc += t;
        f[i] += 1.0f;
      } else if constexpr (OP == 7) {
        asm volatile("fma.rn.f16x2 %0, %0, %1, %2;" : "+r"(h[i]) : "r"(0x3BFF3BFFu), "r"(0x14001400u));
      } else if constexpr (OP == 8) {
        uint16_t t;
        asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[i]));
        asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(f[(i + 4) & 7]));
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(t) : "f"(f[i]), "f"(f[(i + 4) & 7]));
        acc += t;
      } else if constexpr (OP == 9) {
        uint16_t t;
        asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(h[i]));
        asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(t) : "r"(h[i]));
        acc += t;
      } else if constexpr (OP == 10) {
        f[i] = fmaxf(f[i], f[(i + 1) & 7] - 1.0e-7f) ;
      } else {
        h[i] = __byte_perm(h[i], h[(i + 3) & 7], 0x5140);
      }
    }
  }
  float r = static_cast<float>(acc);
#pragma unroll
  for (int i = 0; i < 8; ++i) r += f[i] + __uint_as_float(h[i] & 0x3FFFFFFFu);
  out[blockIdx.x * blockDim.x + threadIdx.x] = r;
}

}  // namespace okc_xu

void xu_run(torch::Tensor out, int64_t op, int64_t iters, int64_t blocks) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(out.numel() >= blocks * 256, "xu_run: out too small");
  float* o = out.data_ptr<float>();
  const int it = static_cast<int>(iters);
  const unsigned nb = static_cast<unsigned>(blocks);
  switch (op) {
#define OKC_XU_CASE(K) case K: okc_xu::xu_kernel<K><<<nb, 256, 0, stream>>>(o, it, 1.0f); break;
    OKC_XU_CASE(0) OKC_XU_CASE(1) OKC_XU_CASE(2) OKC_XU_CASE(3) OKC_XU_CASE(4) OKC_XU_CASE(5)
    OKC_XU_CASE(6) OKC_XU_CASE(7) OKC_XU_CASE(8) OKC_XU_CASE(9) OKC_XU_CASE(10) OKC_XU_CASE(11)
#undef OKC_XU_CASE
    default: TORCH_CHECK(false, "xu_run: op ", op);
  }
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "xu_run launch failed: ", cudaGetErrorString(e));
}
"""

CUDA_XU_DECL = "void xu_run(torch::Tensor out, int64_t op, int64_t iters, int64_t blocks);"

# per OP: (name, thread-ops counted per inner step (the measured op), values per op)
XU_OPS = [("ffma", 1, 1), ("ex2.f32", 1, 1), ("ex2.f16x2", 1, 2), ("ex2.bf16x2", 1, 2), ("cvt.e4m3x2.f32", 1, 2),
          ("cvt.e4m3x2.f16x2", 1, 2), ("cvt.f16x2.f32", 1, 2), ("hfma2", 1, 2), ("mix 2ex2+cvt", 3, 1),
          ("ex2.f16x2+cvt", 2, 2), ("fmnmx", 1, 1), ("prmt", 1, 1)]
