"""Sliding-layer floor probes for L31 (B5 / A65): how fast can one launch per sliding layer be on the judge's H200, with
no attention math? One cluster of CS CTAs per request (NT threads each) runs the dependent chain a one-shot sliding
kernel needs: seqlens -> page entries of the request's window (2,048 + 8 keys, split over the cluster) -> every K and V
row of the CTA's keys issued at once (16 B cp.async into shared memory, one wait) -> a read of every byte (sum) ->
push of one float per CTA to the leader (DSMEM st.shared::cluster) -> cluster barrier -> the leader writes a 16 x 64
BF16 output row. Runtime mode bits drop pieces of the chain: 1 no K/V rows, 2 no page entries (rows = key index:
contiguous), 4 no cluster merge (each CTA writes its share of the output); 8 copies whole 128 B rows per instruction
(8 lanes per row; without it each lane copies its own key in 16 B pieces). Plus an empty kernel and an empty
cluster-16 kernel (arrive / wait only). Graph times of these vs UD3 / UW2 / the v11 path on the same inputs tell
how much of a sliding call is the chain's floor and how much is our kernels' own work.
Host: probe(seqlens, page_table, kc, vc, out, batch, max_pages, window, cs, nt, mode), probe_empty(out, cluster),
probe_setup()."""

CUDA_PROBE_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

namespace okc_probe {
constexpr unsigned FULL = 0xffffffffu;
constexpr int MAXK_ROWS = 2056;      // window + q_len keys

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ uint32_t map_rank(uint32_t local_addr, uint32_t rank) {
  uint32_t a;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(local_addr), "r"(rank));
  return a;
}
__device__ __forceinline__ void st_cluster(uint32_t addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}
__device__ __forceinline__ uint16_t f2bf(float x) {
  uint32_t u;
  asm("mov.b32 %0, %1;" : "=r"(u) : "f"(x));
  return static_cast<uint16_t>(u >> 16);
}

template <int CS, int NT>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(NT) probe_kernel(
    const int* __restrict__ seqlens, const int* __restrict__ page_table, const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc, uint16_t* __restrict__ out, int max_pages, int window, int mode) {
  constexpr int KPC = (MAXK_ROWS + CS - 1) / CS;     // keys per CTA at most
  constexpr int R = (KPC + NT - 1) / NT;             // keys per thread at most
  extern __shared__ __align__(16) uint8_t sm[];      // [KPC][256]: K row | V row
  __shared__ float red[NT / 32];
  __shared__ float buf[CS];
  __shared__ long srow[KPC];                         // mode 8: rows for the whole-row copy pattern
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int b = blockIdx.x / CS;
  const uint32_t rank = cluster_rank();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");   // every CTA running before remote stores
  const int n = seqlens[b];
  const int lo = max(n - window - 8, 0);
  const int per = (n - lo + CS - 1) / CS;
  const int start = lo + static_cast<int>(rank) * per;
  const int cnt = max(0, min(per, n - start));
  long rows[R];
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int k = tid + r * NT;
    rows[r] = 0;
    if (k < cnt) rows[r] = (mode & 2) ? static_cast<long>(start + k) : static_cast<long>(page_table[static_cast<long>(b) * max_pages + start + k]);
  }
  float acc = 0.f;
  if (!(mode & 1) && (mode & 8)) {                     // whole rows per instruction: lanes 0-7 one key's K row, 8-15 its V row
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int k = tid + r * NT;
      if (k < cnt) srow[k] = rows[r];
    }
    __syncthreads();
    for (int idx = tid; idx < cnt * 16; idx += NT) {
      const int k = idx >> 4, c = idx & 7, kv = (idx >> 3) & 1;
      cp16(sm + k * 256 + kv * 128 + 16 * c, (kv ? vc : kc) + srow[k] * 128 + 16 * c);
    }
    asm volatile("cp.async.commit_group;\n\tcp.async.wait_all;" ::: "memory");
    __syncthreads();
    const uint4* s4 = reinterpret_cast<const uint4*>(sm);
    for (int i = tid; i < cnt * 16; i += NT) {
      const uint4 v = s4[i];
      acc += __uint_as_float((v.x ^ v.y ^ v.z ^ v.w) & 0x3f7fffffu);
    }
  } else if (!(mode & 1)) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int k = tid + r * NT;
      if (k < cnt) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
          cp16(sm + k * 256 + 16 * c, kc + rows[r] * 128 + 16 * c);
          cp16(sm + k * 256 + 128 + 16 * c, vc + rows[r] * 128 + 16 * c);
        }
      }
    }
    asm volatile("cp.async.commit_group;\n\tcp.async.wait_all;" ::: "memory");
    __syncthreads();
    const uint4* s4 = reinterpret_cast<const uint4*>(sm);
    for (int i = tid; i < cnt * 16; i += NT) {         // every byte read once (16 B per step, NT threads apart)
      const uint4 v = s4[i];
      acc += __uint_as_float((v.x ^ v.y ^ v.z ^ v.w) & 0x3f7fffffu);
    }
  } else {
#pragma unroll
    for (int r = 0; r < R; ++r) acc += static_cast<float>(rows[r] & 7);
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(FULL, acc, o);
  if (lane == 0) red[warp] = acc;
  __syncthreads();
  if (tid == 0) {
    float s = 0.f;
#pragma unroll
    for (int w = 0; w < NT / 32; ++w) s += red[w];
    red[0] = s;
  }
  if (mode & 4) {                                      // no merge: each CTA writes its 1024 / CS outputs
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
    __syncthreads();
    const float s = red[0];
    for (int i = tid; i < 1024 / CS; i += NT) out[static_cast<long>(b) * 1024 + rank * (1024 / CS) + i] = f2bf(s + i);
    return;
  }
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (tid == 0) st_cluster(map_rank(smem_u32(&buf[rank]), 0), red[0]);
  cluster_sync();
  if (rank == 0) {
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < CS; ++c) s += buf[c];
    for (int i = tid; i < 1024; i += NT) out[static_cast<long>(b) * 1024 + i] = f2bf(s + i);
  }
}

__global__ void probe_empty_kernel(uint16_t* out) {
  if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = 0;
}
__global__ void __cluster_dims__(16, 1, 1) probe_empty_cluster_kernel(uint16_t* out) {
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  if (threadIdx.x == 0) out[blockIdx.x] = 0;
}
}  // namespace okc_probe

#define OKC_PROBE_VARIANTS(X) X(16, 128) X(16, 256) X(8, 256) X(4, 256)

inline size_t okc_probe_smem(int cs) { return static_cast<size_t>((okc_probe::MAXK_ROWS + cs - 1) / cs) * 256; }

void probe(torch::Tensor seqlens, torch::Tensor page_table, torch::Tensor kc, torch::Tensor vc, torch::Tensor out,
           int64_t batch, int64_t max_pages, int64_t window, int64_t cs, int64_t nt, int64_t mode) {
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(out.numel() >= batch * 1024, "probe: out needs batch x 1024 elements");
  bool launched = false;
#define OKC_PROBE_LAUNCH(C, T)                                                                                   \
  if (!launched && cs == C && nt == T) {                                                                         \
    okc_probe::probe_kernel<C, T><<<static_cast<unsigned>(batch * C), T, okc_probe_smem(C), stream>>>(           \
        seqlens.data_ptr<int>(), page_table.data_ptr<int>(), reinterpret_cast<const uint8_t*>(kc.data_ptr()),    \
        reinterpret_cast<const uint8_t*>(vc.data_ptr()), reinterpret_cast<uint16_t*>(out.data_ptr()),            \
        static_cast<int>(max_pages), static_cast<int>(window), static_cast<int>(mode));                          \
    launched = true;                                                                                             \
  }
  OKC_PROBE_VARIANTS(OKC_PROBE_LAUNCH)
#undef OKC_PROBE_LAUNCH
  TORCH_CHECK(launched, "probe: no variant cs=", cs, " nt=", nt);
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "probe launch failed: ", cudaGetErrorString(e));
}

void probe_empty(torch::Tensor out, int64_t cluster) {
  auto stream = at::cuda::getCurrentCUDAStream();
  if (cluster) okc_probe::probe_empty_cluster_kernel<<<16, 128, 0, stream>>>(reinterpret_cast<uint16_t*>(out.data_ptr()));
  else okc_probe::probe_empty_kernel<<<1, 32, 0, stream>>>(reinterpret_cast<uint16_t*>(out.data_ptr()));
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "probe_empty launch failed: ", cudaGetErrorString(e));
}

int64_t probe_setup() {
  int err = 0;
#define OKC_PROBE_SETUP(C, T)                                                                                     \
  err |= cudaFuncSetAttribute(okc_probe::probe_kernel<C, T>, cudaFuncAttributeMaxDynamicSharedMemorySize,        \
                              static_cast<int>(okc_probe_smem(C)));                                               \
  if (C > 8) err |= cudaFuncSetAttribute(okc_probe::probe_kernel<C, T>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  OKC_PROBE_VARIANTS(OKC_PROBE_SETUP)
#undef OKC_PROBE_SETUP
  err |= cudaFuncSetAttribute(okc_probe::probe_empty_cluster_kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  (void)cudaGetLastError();
  return err;
}
"""

CUDA_PROBE_DECL = ("void probe(torch::Tensor seqlens, torch::Tensor page_table, torch::Tensor kc, torch::Tensor vc, "
                   "torch::Tensor out, int64_t batch, int64_t max_pages, int64_t window, int64_t cs, int64_t nt, int64_t mode);\n"
                   "void probe_empty(torch::Tensor out, int64_t cluster);\nint64_t probe_setup();")
