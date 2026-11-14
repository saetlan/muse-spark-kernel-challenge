"""P9: one CUDA kernel (load_inline, sm_90a) for out = x @ W.T at m <= 32: swap-AB mma.sync.m16n8k16,
weights as the 16-row A operand (pre-packed by p9_layout.p9_pack into fragment order: one 16-B load
per lane per 16x16 tile), tokens as N (8 per MMA, NT tiles). The FP32 router runs as the BF16 hi + lo
pair (SGLang's split, allowed at any batch size): two MMAs per k16 step into one FP32 accumulator.

Work split (all compile-time except shapes):
  CTA       RG row groups x RTW 16-row tiles = RT rows tiles, over K slice [rank * K/CS, (rank+1) * K/CS)
            where rank is the CTA's rank in a cluster of CS CTAs (CS = 1: no cluster).
  warp      (row group, K part) = (warp / KSW, warp % KSW); its K part is K/CS/KSW wide.
  ring      each warp keeps D 64-wide K chunks in flight in registers: per chunk 4 A fragments per tile
            (16 B per lane each) and, per 8-token tile, the lane's 16 x values (two 16-B loads from
            L2; K is permuted per chunk by p9_layout.k_perm so they are contiguous). No staging, no
            prologue: the first loads go out in the kernel's first instructions.
  reduce    every warp writes its C fragments to part[kpart][row][token] in shared memory; after a
            cluster barrier, CTA rank r sums rows [r * R/CS, (r+1) * R/CS) of the tile over all CS
            peers and KSW parts in a fixed order (mapa + ld.shared::cluster) and writes the output.
            Deterministic, one kernel, no workspace, no memset.
"""

import tempfile

import torch

P9_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace p9 {

__device__ __forceinline__ void ld_frag(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
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

__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// P22 (F12) decode of one k16 step into the four A registers (proj/p9/p22_layout.py documents the
// format; check_p22.py proves it exact on all 65,536 BF16 patterns with these same operations).
__device__ __forceinline__ uint4 f12_decode(uint32_t q, uint32_t ra, uint32_t rb, uint32_t top4) {
  const uint32_t ca = q & 0x0F0F0F0Fu, cb = (q >> 4) & 0x0F0F0F0Fu;
  const uint32_t ea = top4 - ca, eb = top4 - cb;
  const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x7F7F7F7Fu);
  const uint32_t loa = ((ea << 7) & 0x80808080u) | (ra & 0x7F7F7F7Fu);
  const uint32_t hib = (rb & 0x80808080u) | ((eb >> 1) & 0x7F7F7F7Fu);
  const uint32_t lob = ((eb << 7) & 0x80808080u) | (rb & 0x7F7F7F7Fu);
  return make_uint4(__byte_perm(loa, hia, 0x5140u), __byte_perm(loa, hia, 0x7362u),
                    __byte_perm(lob, hib, 0x5140u), __byte_perm(lob, hib, 0x7362u));
}

// Shared memory: part[KSW][RT*16][8*NT] f32 only. K is permuted inside each 64-wide chunk (k_perm),
// so lane (g, t)'s B words for the chunk's 4 MMAs are x[row][64 c + 16 t .. + 16): two 16-B loads.
template <int NT, int RTW, int RG, int KSW, int D, int CS, bool HILO, bool OUT_F32, bool F12>
__global__ void __launch_bounds__(RG * KSW * 32)
p9_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ qp, const uint4* __restrict__ rp,
          const uint8_t* __restrict__ tops, const int* __restrict__ erow, const int* __restrict__ ecol,
          const unsigned short* __restrict__ ebits, const uint4* __restrict__ x, void* __restrict__ out,
          int m, int n_rows, int k) {
  constexpr int RT = RG * RTW;          // 16-row tiles per CTA
  constexpr int MT = 8 * NT;            // token rows covered
  constexpr int AS = HILO ? 2 : 1;      // A fragments per tile per step
  constexpr int SLOTS = F12 ? 3 : 4 * AS;   // uint4 per tile per chunk: Q + 2 R, or 4 steps x AS
  static_assert(!(F12 && HILO), "the router stays hi + lo");
  extern __shared__ __align__(16) unsigned char smem[];
  float* part = reinterpret_cast<float*>(smem);
  const int kc = k / CS;                // this CTA's K slice
  const int kw = kc / KSW;              // one warp's K part
  const int chunks = kw / 64;           // 64-wide chunks (4 MMA steps each)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rg = warp / KSW, kpart = warp % KSW;
  const uint32_t rank = cluster_rank<CS>();
  const int rowblock = blockIdx.x / CS;
  const int c0 = (static_cast<int>(rank) * kc + kpart * kw) / 64;   // first chunk of this warp
  const int ksteps_total = k / 16;

  const int chunks_total = k / 64;
  const uint4* wbase[RTW];
  const uint4* qbase[RTW];
  const uint4* rbase[RTW];
  uint32_t top4[RTW];
#pragma unroll
  for (int i = 0; i < RTW; ++i) {
    const long long tile = static_cast<long long>(rowblock) * RT + rg * RTW + i;
    if constexpr (F12) {
      qbase[i] = qp + (tile * chunks_total + c0) * 32 + lane;            // Q plane [tile][chunk][lane]
      rbase[i] = rp + (tile * chunks_total + c0) * 64 + lane;            // R plane [tile][chunk][2][lane]
      top4[i] = static_cast<uint32_t>(tops[tile]) * 0x01010101u;
    } else {
      wbase[i] = wp + (tile * ksteps_total + 4LL * c0) * (32 * AS) + lane;
    }
  }
  const uint4* xb[NT];
  bool bok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int row = 8 * n + g;
    bok[n] = row < m;
    xb[n] = x + static_cast<long long>(bok[n] ? row : 0) * (k / 8) + (64LL * c0 + 16 * t) / 8;
  }

  uint4 ra[D][RTW][SLOTS];
  uint4 rb[D][NT][2];
  auto load = [&](int slot, int cc) {
#pragma unroll
    for (int i = 0; i < RTW; ++i) {
      if constexpr (F12) {
        ld_frag(qbase[i] + 32LL * cc, ra[slot][i][0]);
        ld_frag(rbase[i] + 64LL * cc, ra[slot][i][1]);
        ld_frag(rbase[i] + 64LL * cc + 32, ra[slot][i][2]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int h = 0; h < AS; ++h) ld_frag(wbase[i] + (4LL * cc + j) * (32 * AS) + h * 32, ra[slot][i][j * AS + h]);
      }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (bok[n]) {
        ld_x(xb[n] + 8LL * cc, rb[slot][n][0]);
        ld_x(xb[n] + 8LL * cc + 1, rb[slot][n][1]);
      } else {
        rb[slot][n][0] = make_uint4(0u, 0u, 0u, 0u);
        rb[slot][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s)
    if (s < chunks) load(s, s);

  float acc[RTW][NT][4];
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

  for (int cc0 = 0; cc0 < chunks; cc0 += D) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      const int cc = cc0 + s;
      if (cc < chunks) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            const uint32_t b0 = word(rb[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(rb[s][n][j >> 1], (2 * j + 1) & 3);
#pragma unroll
            for (int i = 0; i < RTW; ++i) {
              if constexpr (F12) {
                const uint4& rv = ra[s][i][1 + (j >> 1)];
                const uint4 a = f12_decode(word(ra[s][i][0], j), word(rv, 2 * (j & 1)), word(rv, 2 * (j & 1) + 1), top4[i]);
                mma(acc[i][n], a, b0, b1);
              } else {
#pragma unroll
                for (int h = 0; h < AS; ++h) mma(acc[i][n], ra[s][i][j * AS + h], b0, b1);
              }
            }
          }
        }
        if (cc + D < chunks) load(s, cc + D);
      }
    }
  }

  // C fragment: row g + 8 (r >> 1) of the 16-row tile, token 8 n + 2 t + (r & 1).
  constexpr int ROWS = RT * 16;
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = (rg * RTW + i) * 16 + g + 8 * (r >> 1);
        const int tok = 8 * n + 2 * t + (r & 1);
        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];
      }
  cluster_sync<CS>();

  // CTA rank r reduces rows [r * ROWS/CS, (r+1) * ROWS/CS) over peers and K parts in fixed order.
  constexpr int SHARE = ROWS / CS;
  const uint32_t part_addr = smem_u32(part);
  const int m_out = m < MT ? m : MT;
  for (int idx = tid; idx < SHARE * m_out; idx += blockDim.x) {
    const int tok = idx / SHARE;
    const int row = static_cast<int>(rank) * SHARE + idx % SHARE;
    float sum = 0.0f;
#pragma unroll
    for (int peer = 0; peer < CS; ++peer)
#pragma unroll
      for (int kp = 0; kp < KSW; ++kp) {
        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;
        if constexpr (CS > 1) {
          sum += ld_peer_f32(part_addr + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(peer));
        } else {
          sum += part[off];
        }
      }
    const long long orow = static_cast<long long>(rowblock) * ROWS + row;
    if constexpr (F12) {  // escapes: + x * w_true - x * w_dec (w_dec: exponent top - 15, sign/mantissa 0)
      if (orow < n_rows) {
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(x) + static_cast<long long>(tok) * k;
        const float wdec = __uint_as_float((static_cast<uint32_t>(tops[orow >> 4]) - 15u) << 23);
        for (int e = erow[orow]; e < erow[orow + 1]; ++e) {
          const float xv = __uint_as_float(static_cast<uint32_t>(xs[ecol[e]]) << 16);
          sum = fmaf(xv, __uint_as_float(static_cast<uint32_t>(ebits[e]) << 16), sum);
          sum = fmaf(-xv, wdec, sum);
        }
      }
    }
    if (orow < n_rows) {
      if constexpr (OUT_F32) {
        reinterpret_cast<float*>(out)[static_cast<long long>(tok) * n_rows + orow] = sum;
      } else {
        reinterpret_cast<__nv_bfloat16*>(out)[static_cast<long long>(tok) * n_rows + orow] = __float2bfloat16_rn(sum);
      }
    }
  }
  if constexpr (CS > 1) cluster_sync<CS>();  // peers may still read this CTA's part[]
}

}  // namespace p9

@@INSTANCES@@
"""

# (NT, RTW, RG, KSW, D, CS, HILO, OUT_F32) instances compiled into the module.
P9_INSTANCES = ()
# Instances rendered by the local clang syntax check (one per template path: CS 1 and > 1, HILO, F32).
P9_CHECK_INSTANCES = ((1, 1, 8, 1, 4, 4, False, False), (4, 2, 2, 1, 1, 1, False, False),
                      (1, 1, 1, 8, 2, 8, True, True), (2, 1, 4, 2, 2, 2, False, False),
                      (1, 2, 4, 2, 4, 4, False, False, True), (4, 1, 8, 1, 2, 1, False, False, True))


def p9_smem_bytes(nt, rtw, rg, ksw, cs, k):
    return ksw * rg * rtw * 16 * 8 * nt * 4  # part[KSW][RT*16][8*NT] f32


def p9_name(inst):
    nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
    f12 = inst[8] if len(inst) > 8 else False
    return f"p9_run_{nt}_{rtw}_{rg}_{ksw}_{d}_{cs}_{int(hilo)}_{int(f32)}" + ("_f12" if f12 else "")


def p9_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
        f12 = inst[8] if len(inst) > 8 else False
        tmpl = (f"{nt}, {rtw}, {rg}, {ksw}, {d}, {cs}, {'true' if hilo else 'false'}, {'true' if f32 else 'false'}, "
                f"{'true' if f12 else 'false'}")
        wrappers.append(f"""
void {p9_name(inst)}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow,
                     torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out,
                     int64_t m, int64_t n_rows, int64_t k, int64_t smem) {{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned ctas = static_cast<unsigned>((n_rows / ({rg} * {rtw} * 16)) * {cs});
  const uint4* wptr = reinterpret_cast<const uint4*>(wp.data_ptr());
  const uint4* qptr = reinterpret_cast<const uint4*>(qp.data_ptr());
  const uint4* rptr = reinterpret_cast<const uint4*>(rp.data_ptr());
  const uint8_t* tptr = reinterpret_cast<const uint8_t*>(tops.data_ptr());
  const int* erp = reinterpret_cast<const int*>(erow.data_ptr());
  const int* ecp = reinterpret_cast<const int*>(ecol.data_ptr());
  const unsigned short* ebp = reinterpret_cast<const unsigned short*>(ebits.data_ptr());
  const uint4* xptr = reinterpret_cast<const uint4*>(x.data_ptr());
  void* optr = out.data_ptr();
  const int mi = static_cast<int>(m), ni = static_cast<int>(n_rows), ki = static_cast<int>(k);
  if ({cs} == 1) {{
    p9::p9_kernel<{tmpl}><<<ctas, {rg} * {ksw} * 32, static_cast<size_t>(smem), stream>>>(wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }} else {{
    cudaLaunchConfig_t cfg = {{}};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = {cs};
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(ctas, 1, 1);
    cfg.blockDim = dim3({rg} * {ksw} * 32, 1, 1);
    cfg.dynamicSmemBytes = static_cast<size_t>(smem);
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, p9::p9_kernel<{tmpl}>, wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);
    TORCH_CHECK(err == cudaSuccess, "p9 cluster launch failed: ", cudaGetErrorString(err));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }}
}}
void {p9_name(inst)}_attr(int64_t smem) {{
  cudaFuncSetAttribute(p9::p9_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
}}
int64_t {p9_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, p9::p9_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return P9_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def p9_decls(instances):
    out = []
    for inst in instances:
        name = p9_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow, "
                   f"torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows, "
                   f"int64_t k, int64_t smem);")
        out.append(f"void {name}_attr(int64_t smem);")
        out.append(f"int64_t {name}_info();")
    return out


def p9_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = p9_decls(instances)
    return load_inline(
        name="okc_proj_p9", cpp_sources="\n".join(decls), cuda_sources=p9_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_p9_"), verbose=False)
