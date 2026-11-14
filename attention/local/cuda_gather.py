"""Gather-engine microbenchmarks for lab L5 (A19): read the K and V rows listed in `idx` (128 bytes
each, the pair's piece under tp8) as fast as possible, with different load mechanisms, so the
attention kernel can be built on the fastest one. Each kernel only XORs what it read (so the
loads stay live) and writes one word if a magic value appears.

ld modes (registers, __ldg 16 B):
  0 whole rows: 8 lanes x 16 B per row, 4 rows per warp instruction (L2 G2's pattern)
  1 64-byte halves: 4 lanes x 16 B at +0, then at +64, 8 rows per instruction (A5a's pattern)
  2 split-32: lane t4 reads [32*t4, +16) then [32*t4+16, +16) of 8 rows (A5b's K pattern)
cp: cp.async.cg 16 B into a per-warp shared-memory ring of S stages (16 rows x 2 arrays each).
bulk: cp.async.bulk of each 128-byte row into a per-warp ring of S stages, completion counted by
  one mbarrier per stage (expect_tx 4096 bytes); waits are bounded so a mistake can't hang the run
  (a timeout sets err[0]).
"""

CUDA_GATHER_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace okc_g {

constexpr unsigned MAGIC = 0x9E3779B9u;

template <int MODE, int U>
__global__ void gld_kernel(const uint4* __restrict__ k, const uint4* __restrict__ v, const int* __restrict__ idx,
                           long n, unsigned* __restrict__ out) {
  constexpr int RPI = MODE == 0 ? 4 : 8;          // rows per warp instruction
  const int lane = threadIdx.x & 31;
  const long warp = (blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x) >> 5;
  const long nwarps = (gridDim.x * static_cast<long>(blockDim.x)) >> 5;
  const int sub = MODE == 0 ? lane >> 3 : lane >> 2;
  unsigned acc = 0;
  for (long base = warp * RPI * U; base < n; base += nwarps * RPI * U) {
    int row[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long r = base + u * RPI + sub;
      row[u] = r < n ? __ldg(idx + r) : 0;
    }
    uint4 a[U], b[U], a2[U], b2[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      const long rb = static_cast<long>(row[u]) * 8;
      if (MODE == 0) {
        a[u] = __ldg(k + rb + (lane & 7));
        b[u] = __ldg(v + rb + (lane & 7));
      } else if (MODE == 1) {
        a[u] = __ldg(k + rb + (lane & 3));
        b[u] = __ldg(v + rb + (lane & 3));
      } else {
        a[u] = __ldg(k + rb + 2 * (lane & 3));
        b[u] = __ldg(v + rb + 2 * (lane & 3));
      }
    }
    if (MODE != 0) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        const long rb = static_cast<long>(row[u]) * 8;
        if (MODE == 1) {
          a2[u] = __ldg(k + rb + 4 + (lane & 3));
          b2[u] = __ldg(v + rb + 4 + (lane & 3));
        } else {
          a2[u] = __ldg(k + rb + 2 * (lane & 3) + 1);
          b2[u] = __ldg(v + rb + 2 * (lane & 3) + 1);
        }
      }
    }
#pragma unroll
    for (int u = 0; u < U; ++u) {
      acc ^= a[u].x ^ a[u].y ^ a[u].z ^ a[u].w ^ b[u].x ^ b[u].y ^ b[u].z ^ b[u].w;
      if (MODE != 0) acc ^= a2[u].x ^ a2[u].y ^ a2[u].z ^ a2[u].w ^ b2[u].x ^ b2[u].y ^ b2[u].z ^ b2[u].w;
    }
  }
  if (acc == MAGIC) out[0] = acc;
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async ring: per warp S stages x 4096 B (16 rows x 128 B of K, then of V).
template <int S>
__global__ void gcp_kernel(const uint8_t* __restrict__ k, const uint8_t* __restrict__ v, const int* __restrict__ idx,
                           long n, unsigned* __restrict__ out) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const long warp = (blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x) >> 5;
  const long nwarps = (gridDim.x * static_cast<long>(blockDim.x)) >> 5;
  uint8_t* ring = smem + static_cast<long>(wib) * S * 4096;
  const long nchunks = (n + 15) / 16;
  const long mine = warp < nchunks ? (nchunks - warp + nwarps - 1) / nwarps : 0;
  auto issue = [&](long i) {
    if (i < mine) {
      const long c = warp + i * nwarps;
      uint8_t* st = ring + (i % S) * 4096;
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        const int rr = q * 4 + (lane >> 3);
        const long t = c * 16 + rr;
        const long row = t < n ? __ldg(idx + t) : 0;
        const int off = 16 * (lane & 7);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(st + rr * 128 + off)),
                     "l"(k + row * 128 + off));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(st + 2048 + rr * 128 + off)),
                     "l"(v + row * 128 + off));
      }
    }
    asm volatile("cp.async.commit_group;");
  };
  unsigned acc = 0;
#pragma unroll
  for (int i = 0; i < S - 1; ++i) issue(i);
  for (long i = 0; i < mine; ++i) {
    issue(i + S - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(S - 1));
    const uint8_t* st = ring + (i % S) * 4096;
#pragma unroll
    for (int q = 0; q < 4; ++q) {        // read back exactly what this lane copied
      const int rr = q * 4 + (lane >> 3);
      const int off = 16 * (lane & 7);
      const uint4 a = *reinterpret_cast<const uint4*>(st + rr * 128 + off);
      const uint4 b = *reinterpret_cast<const uint4*>(st + 2048 + rr * 128 + off);
      acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w;
    }
  }
  asm volatile("cp.async.wait_all;");
  if (acc == MAGIC) out[0] = acc;
}

}  // namespace okc_g

void gather_ld(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, int64_t mode, int64_t unroll,
               int64_t blocks, int64_t threads) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint4* kp = reinterpret_cast<const uint4*>(k.data_ptr());
  const uint4* vp = reinterpret_cast<const uint4*>(v.data_ptr());
  const int* ip = idx.data_ptr<int>();
  unsigned* op = reinterpret_cast<unsigned*>(out.data_ptr());
  const long n = idx.numel();
  const dim3 g(static_cast<unsigned>(blocks)), t(static_cast<unsigned>(threads));
  if (mode == 0 && unroll == 4) okc_g::gld_kernel<0, 4><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
  else if (mode == 0) okc_g::gld_kernel<0, 8><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
  else if (mode == 1 && unroll == 4) okc_g::gld_kernel<1, 4><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
  else if (mode == 1) okc_g::gld_kernel<1, 8><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
  else if (unroll == 4) okc_g::gld_kernel<2, 4><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
  else okc_g::gld_kernel<2, 8><<<g, t, 0, stream>>>(kp, vp, ip, n, op);
}

int64_t gather_setup(int64_t max_smem) {
  int ok = 0;
  ok |= cudaFuncSetAttribute(okc_g::gcp_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem));
  ok |= cudaFuncSetAttribute(okc_g::gcp_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem));
  ok |= cudaFuncSetAttribute(okc_g::gcp_kernel<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem));
  return ok;
}

void gather_cp(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, int64_t stages, int64_t blocks,
               int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* kp = reinterpret_cast<const uint8_t*>(k.data_ptr());
  const uint8_t* vp = reinterpret_cast<const uint8_t*>(v.data_ptr());
  const int* ip = idx.data_ptr<int>();
  unsigned* op = reinterpret_cast<unsigned*>(out.data_ptr());
  const long n = idx.numel();
  const size_t smem = static_cast<size_t>(warps) * stages * 4096;
  const dim3 g(static_cast<unsigned>(blocks)), t(static_cast<unsigned>(warps * 32));
  if (stages == 2) okc_g::gcp_kernel<2><<<g, t, smem, stream>>>(kp, vp, ip, n, op);
  else if (stages == 4) okc_g::gcp_kernel<4><<<g, t, smem, stream>>>(kp, vp, ip, n, op);
  else okc_g::gcp_kernel<6><<<g, t, smem, stream>>>(kp, vp, ip, n, op);
}

"""

CUDA_GATHER_DECL = (
    "void gather_ld(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, int64_t mode, "
    "int64_t unroll, int64_t blocks, int64_t threads);\n"
    "int64_t gather_setup(int64_t max_smem);\n"
    "void gather_cp(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, int64_t stages, "
    "int64_t blocks, int64_t warps);\n"
    "")

CUDA_BULK_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

namespace okc_gb {

constexpr unsigned MAGIC = 0x9E3779B9u;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Bulk copies: per warp S stages x 4096 B, one mbarrier per stage after all warps' data.
template <int S>
__global__ void gbulk_kernel(const uint8_t* __restrict__ k, const uint8_t* __restrict__ v,
                             const int* __restrict__ idx, long n, unsigned* __restrict__ out, int* __restrict__ err) {
  extern __shared__ __align__(128) uint8_t smem[];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5, wpb = blockDim.x >> 5;
  const long warp = (blockIdx.x * static_cast<long>(blockDim.x) + threadIdx.x) >> 5;
  const long nwarps = (gridDim.x * static_cast<long>(blockDim.x)) >> 5;
  uint8_t* ring = smem + static_cast<long>(wib) * S * 4096;
  uint64_t* mbar = reinterpret_cast<uint64_t*>(smem + static_cast<long>(wpb) * S * 4096) + wib * S;
  const long nchunks = (n + 15) / 16;
  const long mine = warp < nchunks ? (nchunks - warp + nwarps - 1) / nwarps : 0;
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < S; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(mbar + s)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncwarp();
  auto issue = [&](long i) {
    if (i >= mine) return;
    const long c = warp + i * nwarps;
    const int s = static_cast<int>(i % S);
    uint8_t* st = ring + s * 4096;
    if (lane == 0)
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], 4096;" ::"r"(smem_u32(mbar + s)));
    __syncwarp();
    const int rr = lane & 15;
    const long t = c * 16 + rr;
    const long row = t < n ? __ldg(idx + t) : 0;
    const uint8_t* src = (lane < 16 ? k : v) + row * 128;
    uint8_t* dst = st + (lane < 16 ? 0 : 2048) + rr * 128;
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], 128, [%2];"
                 ::"r"(smem_u32(dst)), "l"(src), "r"(smem_u32(mbar + s)) : "memory");
  };
  unsigned acc = 0;
  int timeout = 0;
#pragma unroll
  for (int i = 0; i < S - 1; ++i) issue(i);
  for (long i = 0; i < mine; ++i) {
    issue(i + S - 1);
    const int s = static_cast<int>(i % S);
    const uint32_t parity = static_cast<uint32_t>((i / S) & 1);
    uint32_t done = 0;
    for (int spin = 0; spin < (1 << 22) && !done; ++spin) {
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                   : "=r"(done) : "r"(smem_u32(mbar + s)), "r"(parity) : "memory");
    }
    if (!done) timeout = 1;
    const uint8_t* st = ring + s * 4096;
    const uint4 a = *reinterpret_cast<const uint4*>(st + lane * 128);
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");   // generic reads before the next async writes
  }
  if (timeout) atomicExch(err, 1);
  if (acc == MAGIC) out[0] = acc;
}

}  // namespace okc_gb

int64_t bulk_setup(int64_t max_smem) {
  int ok = 0;
  ok |= cudaFuncSetAttribute(okc_gb::gbulk_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem));
  ok |= cudaFuncSetAttribute(okc_gb::gbulk_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(max_smem));
  return ok;
}

void gather_bulk(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, torch::Tensor err,
                 int64_t stages, int64_t blocks, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* kp = reinterpret_cast<const uint8_t*>(k.data_ptr());
  const uint8_t* vp = reinterpret_cast<const uint8_t*>(v.data_ptr());
  const int* ip = idx.data_ptr<int>();
  unsigned* op = reinterpret_cast<unsigned*>(out.data_ptr());
  const long n = idx.numel();
  const size_t smem = static_cast<size_t>(warps) * stages * (4096 + 8);
  const dim3 g(static_cast<unsigned>(blocks)), t(static_cast<unsigned>(warps * 32));
  if (stages == 2) okc_gb::gbulk_kernel<2><<<g, t, smem, stream>>>(kp, vp, ip, n, op, err.data_ptr<int>());
  else okc_gb::gbulk_kernel<4><<<g, t, smem, stream>>>(kp, vp, ip, n, op, err.data_ptr<int>());
}
"""

CUDA_BULK_DECL = ("int64_t bulk_setup(int64_t max_smem);\n"
                  "void gather_bulk(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, "
                  "torch::Tensor err, int64_t stages, int64_t blocks, int64_t warps);")
