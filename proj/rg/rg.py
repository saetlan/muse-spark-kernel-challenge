"""RG: router kernel (load_inline, sm_90a) for out = x @ W.T in FP32 at m = 32/128/256, W the FP32 router as
SGLang's BF16 hi + lo pair (proj/rg/rg_layout.py), two mma.sync.m16n8k16 per step into one FP32 accumulator.

Work split (compile-time; K = 8192 fixed, MT = 32 * NTW * RTC... see below):
  cluster  CS CTAs split K (chunks [r * 128/CS, (r+1) * 128/CS) of 64 k); fixed-order DSMEM reduction.
  CTA      RTC 16-row tiles x all MT tokens, 4 warps = RTC row tiles x (4 / RTC) token groups; x is staged
           once per CTA per chunk in shared memory (cp.async, 16-B pieces, swizzled: piece j of token row r
           at r * 128 + ((j ^ (r & 7)) << 4)), XS stages; B fragments by ldmatrix (x4: two n8 tiles).
  warp     its row tile's hi/lo A fragments from global into a D-chunk register ring (L1-cached: the other
           token-group warps of the same tile hit L1), NTW n8 tiles of tokens.
  reduce   accumulators to shared memory as [token][row] FP32; after a cluster barrier rank r sums its
           tokens' rows over the CS peers (16-B DSMEM loads, all issued before the sum) and stores FP32
           with 16-B stores. One kernel, deterministic, no memset, no atomics.
check_rg.py parses this file's index helpers and emulates the kernel with the PTX fragment layouts.
"""

import tempfile

RG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace rg {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ld_a(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
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

// Index helpers (check_rg.py evaluates their return expressions): first chunk of rank r; byte offset of
// 16-B piece j of token row r in an x stage; uint4 index of tile `tile`'s step s hi (h = 0) / lo (h = 1)
// fragment of lane `lane` (KSS k16 steps per row); a warp's row tile and token group; the token a lane
// addresses for ldmatrix matrix mi of n8-tile pair p (group base tb); C fragment row (in the CTA) and
// token; first token of rank r's share, item -> token offset / 4-row group.
__device__ __forceinline__ int rg_c0(int r, int chunks, int cs) { return r * chunks / cs; }
__device__ __forceinline__ int rg_xoff(int tok, int j) { return tok * 128 + ((j ^ (tok & 7)) << 4); }
__device__ __forceinline__ long long rg_aidx(long long tile, int kss, int s, int h, int lane) { return ((tile * kss + s) * 2 + h) * 32 + lane; }
__device__ __forceinline__ int rg_wrt(int warp, int tg) { return warp / tg; }
__device__ __forceinline__ int rg_wtg(int warp, int tg) { return warp % tg; }
__device__ __forceinline__ int rg_ltok(int tb, int p, int lane) { return tb + 16 * p + (lane & 7) + 8 * ((lane >> 3) >> 1); }
__device__ __forceinline__ int rg_lkch(int j, int lane) { return 2 * j + ((lane >> 3) & 1); }
__device__ __forceinline__ int rg_crow(int rt, int g, int r) { return 16 * rt + g + 8 * (r >> 1); }
__device__ __forceinline__ int rg_ctok(int tb, int n, int t, int r) { return tb + 8 * n + 2 * t + (r & 1); }
__device__ __forceinline__ int rg_etok(int r, int share, int idx, int groups) { return r * share + idx / groups; }
__device__ __forceinline__ int rg_erow(int idx, int groups) { return 4 * (idx % groups); }

template <int RTC, int NTW, int CS, int XS, int D>
__global__ void __launch_bounds__(128)
rg_kernel(const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int m) {
  constexpr int TG = 4 / RTC;                 // token groups (warps per row tile)
  constexpr int MT = TG * 8 * NTW;            // tokens covered (padded)
  constexpr int K = 8192, CHUNKS = K / 64, KSS = K / 16, ROWS = 16 * RTC;
  static_assert(RTC == 1 || RTC == 2, "row tiles per CTA");
  static_assert(NTW == 1 || NTW % 2 == 0, "n8 tiles per warp come in pairs (ldmatrix.x4)");
  static_assert(XS >= 2 && D >= 1 && MT % CS == 0, "rg config");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int rt = rg_wrt(warp, TG), tgi = rg_wtg(warp, TG);
  const uint32_t rank = cluster_rank<CS>();
  const long long cl = blockIdx.x / CS;
  const long long tile = cl * RTC + rt;
  const int c0 = rg_c0(static_cast<int>(rank), CHUNKS, CS), c1 = rg_c0(static_cast<int>(rank) + 1, CHUNKS, CS);
  const int nch = c1 - c0;
  const uint32_t sbase = smem_u32(smem);

  auto copy_x = [&](int c, int s) {
    for (int p = tid; p < MT * 8; p += 128) {
      const int tok = p >> 3, j = p & 7;
      const bool ok = tok < m;
      const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;
      cp16(sbase + static_cast<uint32_t>(s * MT * 128 + rg_xoff(tok, j)), src, ok ? 16u : 0u);
    }
  };
  uint4 wa[D][4][2];
  auto load_a = [&](int sl, int c) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) ld_a(wp + rg_aidx(tile, KSS, 4 * c + j, h, lane), wa[sl][j][h]);
  };

#pragma unroll
  for (int sl = 0; sl < D; ++sl)
    if (sl < nch) load_a(sl, c0 + sl);
#pragma unroll
  for (int s = 0; s < XS - 1; ++s) {
    if (s < nch) copy_x(c0 + s, s);
    cp_commit();
  }

  float acc[NTW][4];
#pragma unroll
  for (int n = 0; n < NTW; ++n)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
  const int tb = tgi * 8 * NTW;

  for (int i0 = 0; i0 < nch; i0 += D) {
#pragma unroll
    for (int sl = 0; sl < D; ++sl) {
      const int i = i0 + sl;
      if (i < nch) {
        cp_wait<XS - 2>();
        __syncthreads();                                   // chunk i landed; every warp done with chunk i - 1
        if (i + XS - 1 < nch) copy_x(c0 + i + XS - 1, (i + XS - 1) % XS);
        cp_commit();
        const uint32_t st = sbase + static_cast<uint32_t>((i % XS) * MT * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (NTW == 1) {
            uint32_t b0, b1;
            ldm_x2(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, 0, lane & 15), rg_lkch(j, lane & 15))), b0, b1);
            mma(acc[0], wa[sl][j][0], b0, b1);
            mma(acc[0], wa[sl][j][1], b0, b1);
          } else {
#pragma unroll
            for (int p = 0; p < NTW / 2; ++p) {
              uint32_t b[4];
              ldm_x4(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, p, lane), rg_lkch(j, lane))), b[0], b[1], b[2], b[3]);
              mma(acc[2 * p], wa[sl][j][0], b[0], b[1]);
              mma(acc[2 * p], wa[sl][j][1], b[0], b[1]);
              mma(acc[2 * p + 1], wa[sl][j][0], b[2], b[3]);
              mma(acc[2 * p + 1], wa[sl][j][1], b[2], b[3]);
            }
          }
        }
        if (i + D < nch) load_a(sl, c0 + i + D);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();                                         // stages idle: reuse them for the partial tile

  float* part = reinterpret_cast<float*>(smem);             // [MT tokens][ROWS rows]
#pragma unroll
  for (int n = 0; n < NTW; ++n)
#pragma unroll
    for (int r = 0; r < 4; ++r) part[rg_ctok(tb, n, t, r) * ROWS + rg_crow(rt, g, r)] = acc[n][r];
  cluster_sync<CS>();
  constexpr int SHARE = MT / CS, GROUPS = ROWS / 4;
  const uint32_t paddr = sbase;
  for (int idx = tid; idx < SHARE * GROUPS; idx += 128) {
    const int tok = rg_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = rg_erow(idx, GROUPS);
    float4 v[CS];
#pragma unroll
    for (int pr = 0; pr < CS; ++pr) {
      if constexpr (CS > 1) {
        v[pr] = ld_peer_v4(paddr + 4u * static_cast<uint32_t>(tok * ROWS + row), static_cast<uint32_t>(pr));
      } else {
        v[pr] = *reinterpret_cast<const float4*>(part + tok * ROWS + row);
      }
    }
    float4 s = v[0];
#pragma unroll
    for (int pr = 1; pr < CS; ++pr) {
      s.x += v[pr].x; s.y += v[pr].y; s.z += v[pr].z; s.w += v[pr].w;
    }
    if (tok < m) *reinterpret_cast<float4*>(out + static_cast<long long>(tok) * 256 + cl * ROWS + row) = s;
  }
  if constexpr (CS > 1) cluster_sync<CS>();                // peers may still read this CTA's tile
}

}  // namespace rg

@@INSTANCES@@
"""

# (RTC, NTW, CS, XS, D) instances compiled into the module.
RG_INSTANCES = ()
RG_CHECK_INSTANCES = ((1, 1, 8, 3, 1), (2, 2, 16, 4, 2), (2, 16, 8, 3, 1), (1, 8, 16, 3, 2))


def rg_name(inst):
    return "rg_run_" + "_".join(str(int(v)) for v in inst)


def rg_mt(inst):
    rtc, ntw = inst[0], inst[1]
    return (4 // rtc) * 8 * ntw


def rg_smem(inst):
    rtc, ntw, cs, xs, d = inst
    return max(xs * rg_mt(inst) * 128, rg_mt(inst) * 16 * rtc * 4)


def rg_source(instances):
    wrappers = []
    for inst in instances:
        rtc, ntw, cs, xs, d = inst
        tmpl = f"{rtc}, {ntw}, {cs}, {xs}, {d}"
        mt = rg_mt(inst)
        wrappers.append(f"""
void {rg_name(inst)}(torch::Tensor wp, torch::Tensor x, torch::Tensor out) {{
  const int64_t m = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::Float, "rg: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && x.size(1) == 8192, "rg: layouts");
  TORCH_CHECK(m >= 1 && m <= {mt} && out.size(0) == m && out.size(1) == 256, "rg: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= 256LL * 8192 * 4, "rg: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0, "rg: unaligned");
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((16 / {rtc}) * {cs}), 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = {rg_smem(inst)};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, rg::rg_kernel<{tmpl}>, reinterpret_cast<const uint4*>(wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<float*>(out.data_ptr()), static_cast<int>(m));
  TORCH_CHECK(err == cudaSuccess, "rg launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
void {rg_name(inst)}_attr() {{
  cudaFuncSetAttribute(rg::rg_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, {rg_smem(inst)});
  if ({cs} > 8) cudaFuncSetAttribute(rg::rg_kernel<{tmpl}>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}}
int64_t {rg_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, rg::rg_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return RG_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def rg_decls(instances):
    out = []
    for inst in instances:
        name = rg_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor x, torch::Tensor out);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    return out


def rg_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = rg_decls(instances)
    return load_inline(
        name="okc_proj_rg", cpp_sources="\n".join(decls), cuda_sources=rg_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_rg_"), verbose=False)
