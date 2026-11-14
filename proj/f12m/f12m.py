"""F12M: swap-AB mma.sync.m16n8k16 kernel (load_inline, sm_90a) for out = x @ W.T at m = 2..32 on the
12-bit lossless planes of proj/f12m/f12m_layout.py (F12 = true) or on P9's BF16 fragments (F12 = false,
same structure without compression, for comparison).

Work split (compile-time; the instance fixes K = 64 * NS * KSW and the token tiles NT, m <= 8 NT):
  CTA    RT = RG * RTW 16-row tiles (RB = 16 RT rows), full K, RG * KSW warps; many small CTAs per SM
         (the GEMV structure), no clusters.
  warp   (rg, ks) = (warp / KSW, warp % KSW): its RTW tiles over chunks ks * NS .. + NS (64 k each);
         per chunk and tile 3 (F12) or 4 (BF16) 16-B weight loads per lane, per token tile two 16-B x
         loads (K is permuted per chunk, so they are 16 consecutive x of token 8 n + g); a register
         ring keeps D chunks in flight, the first loads go out in the kernel's first instructions.
  math   per k16 step: 7-op decode of each tile's A fragment (F12), then RTW x NT MMAs, FP32 accumulate.
  reduce C fragments to part[ks][row][token] in shared memory; after one barrier, thread loops over
         (row, token) sum the KSW parts in a fixed order, scale by 2^(base - 112), apply the row's
         escapes (x from global, L2-resident) and store BF16. Deterministic, one kernel, no memset.
CS > 1 (split-K across a cluster of CS CTAs, launched with cudaLaunchKernelEx): CTA rank r of the
cluster streams chunks [r NS KSW, (r + 1) NS KSW) of the same tiles; each CTA sums its KSW parts in place
(fixed order), a cluster barrier, then rank r finalizes items idx = r (mod CS), reading every rank's sum
through DSMEM in rank order (deterministic), and a last cluster barrier keeps shared memory alive for
the peers. CS = 1 is the single-CTA kernel above.
XD < D (x ring depth): x is L2-resident, so its ring can be shallower than the weight ring; chunk s's x
goes to x slot s % XD (D % XD == 0) and is reloaded XD chunks ahead, which frees 8 NT registers per slot
for weight bytes in flight. XD = D is the original interleaved ring.
TG > 1 (token groups, m up to 8 NT TG): the CTA's warps also split the tokens; the TG warps of a (row group, k
part) read the same weight chunks with L1-allocating loads (one DRAM read, the siblings hit L1/L2), each keeping
one token group's x and accumulators, so registers per warp stay those of 8 NT tokens. TG = 1 is unchanged.
check_f12m.py parses and runs this file's decode, scale and index helpers.
"""

import tempfile

F12M_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace f12m {

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_w_l1(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
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

__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Index helpers (check_f12m.py evaluates their return expressions): tile i of warp row group rg in CTA
// bx; chunk of cluster rank rank, warp part ks, step s; uint4 index of an F12 slot, a BF16 fragment (P9 layout), an x
// half-chunk of token tok (k8 = K / 8); slot and word of step j's L words; C fragment row and token;
// record half.
__device__ __forceinline__ long long f12m_tile(long long bx, int rg, int i, int rt, int rtw) { return bx * rt + rg * rtw + i; }
__device__ __forceinline__ int f12m_chunk(int rank, int ks, int s, int ns, int ksw) { return (rank * ksw + ks) * ns + s; }
__device__ __forceinline__ long long f12m_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long f12m_bidx(long long tile, int kss, int c, int j, int lane) { return (tile * kss + 4 * c + j) * 32 + lane; }
__device__ __forceinline__ long long f12m_xidx(long long tok, int k8, int c, int t, int h) { return tok * k8 + 8 * c + 2 * t + h; }
__device__ __forceinline__ int f12m_lslot(int j) { return 1 + (j >> 1); }
__device__ __forceinline__ int f12m_lword(int j, int h) { return 2 * (j & 1) + h; }
__device__ __forceinline__ int f12m_crow(int rg, int rtw, int i, int g, int r) { return 16 * (rg * rtw + i) + g + 8 * (r >> 1); }
__device__ __forceinline__ int f12m_ctok(int n, int t, int r) { return 8 * n + 2 * t + (r & 1); }
__device__ __forceinline__ long long f12m_ridx(long long row, int h) { return row * 2 + h; }
__device__ __forceinline__ int f12m_wrg(int warp, int tg, int ksw) { return warp / (tg * ksw); }
__device__ __forceinline__ int f12m_wtg(int warp, int tg, int ksw) { return (warp / ksw) % tg; }

// Decode of one lane's A fragment for one k16 step (format in f12m_layout.py): four BF16x2 registers
// scaled by 2^(112 - base) per row.
__device__ __forceinline__ uint4 f12m_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }

template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12, int CS, int XD, int TG>
__global__ void __launch_bounds__(RG * TG * KSW * 32)
f12m_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint4* __restrict__ x,
            __nv_bfloat16* __restrict__ out, int m) {
  static_assert(D >= 1 && D <= NS && NS % D == 0 && CS >= 1 && CS <= 8, "f12m ring depth or cluster");
  static_assert(XD >= 1 && XD <= D && D % XD == 0, "f12m x ring depth");
  static_assert(TG >= 1 && TG <= 4, "f12m token groups");
  constexpr int RT = RG * RTW;          // 16-row tiles per CTA
  constexpr int RB = 16 * RT;           // rows per CTA
  constexpr int MT = 8 * NT;            // token slots per warp
  constexpr int MTC = MT * TG;          // token slots per CTA
  constexpr int NTH = RG * TG * KSW * 32;
  constexpr int CHS = NS * KSW * CS;    // 64-wide chunks per row
  constexpr int KSS = 4 * CHS;          // k16 steps per row
  constexpr int K8 = 8 * CHS;           // K / 8: uint4 per x row
  constexpr int SL = F12 ? 3 : 4;       // uint4 per tile per chunk
  __shared__ float part[KSW][RB][MTC];
  __shared__ uint4 recs[F12 ? 2 * RB : 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rg = f12m_wrg(warp, TG, KSW), tg = f12m_wtg(warp, TG, KSW), ks = warp % KSW;
  const int tb = MT * tg;               // this warp's first token
  const uint32_t rank = cluster_rank<CS>();
  const long long bx = blockIdx.x / CS;

  bool tok_ok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) tok_ok[n] = tb + 8 * n + g < m;
  uint4 wv[D][RTW][SL];
  uint4 xv[XD][NT][2];
  auto loadw = [&](int slot, int s) {
    const int c = f12m_chunk(static_cast<int>(rank), ks, s, NS, KSW);
#pragma unroll
    for (int i = 0; i < RTW; ++i) {
      const long long tile = f12m_tile(bx, rg, i, RT, RTW);
#pragma unroll
      for (int j = 0; j < SL; ++j) {
        if constexpr (F12) {
          if constexpr (TG > 1) {
            ld_w_l1(wp + f12m_widx(tile, CHS, c, j, lane), wv[slot][i][j]);
          } else {
            ld_w(wp + f12m_widx(tile, CHS, c, j, lane), wv[slot][i][j]);
          }
        } else {
          ld_w(wp + f12m_bidx(tile, KSS, c, j, lane), wv[slot][i][j]);
        }
      }
    }
  };
  auto loadx = [&](int xslot, int s) {
    const int c = f12m_chunk(static_cast<int>(rank), ks, s, NS, KSW);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (tok_ok[n]) {
        ld_x(x + f12m_xidx(tb + 8 * n + g, K8, c, t, 0), xv[xslot][n][0]);
        ld_x(x + f12m_xidx(tb + 8 * n + g, K8, c, t, 1), xv[xslot][n][1]);
      } else {
        xv[xslot][n][0] = make_uint4(0u, 0u, 0u, 0u);
        xv[xslot][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s) {
    loadw(s, s);
    if (s < XD) loadx(s, s);
  }

  // Row records (F12): issued behind the first weight loads, kept in registers, written to shared
  // memory after the main loop.
  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);
  if (F12 && tid < RB) {
    ld_x(rec + f12m_ridx(bx * RB + tid, 0), rc0);
    ld_x(rec + f12m_ridx(bx * RB + tid, 1), rc1);
  }

  float acc[RTW][NT][4];
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

#pragma unroll 1
  for (int s0 = 0; s0 < NS; s0 += D) {
#pragma unroll
  for (int slot = 0; slot < D; ++slot) {
    const int s = s0 + slot;
    const int xs = slot % XD;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint4 a[RTW];
#pragma unroll
      for (int i = 0; i < RTW; ++i) {
        if constexpr (F12) {
          const uint4& lv = wv[slot][i][f12m_lslot(j)];
          a[i] = f12m_decode(word(wv[slot][i][0], j), word(lv, f12m_lword(j, 0)), word(lv, f12m_lword(j, 1)));
        } else {
          a[i] = wv[slot][i][j];
        }
      }
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const uint32_t b0 = word(xv[xs][n][j >> 1], (2 * j) & 3);
        const uint32_t b1 = word(xv[xs][n][j >> 1], (2 * j + 1) & 3);
#pragma unroll
        for (int i = 0; i < RTW; ++i) mma(acc[i][n], a[i], b0, b1);
      }
    }
    if (s + D < NS) loadw(slot, s + D);
    if (s + XD < NS) loadx(xs, s + XD);
  }
  }

#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][tb + f12m_ctok(n, t, r)] = acc[i][n][r];
  if (F12 && tid < RB) {
    recs[2 * tid] = rc0;
    recs[2 * tid + 1] = rc1;
  }
  __syncthreads();

  const int mo = m < MTC ? m : MTC;
  const long long n_rows = static_cast<long long>(gridDim.x / CS) * RB;
  const unsigned short* xh = reinterpret_cast<const unsigned short*>(x);
  if constexpr (CS > 1) {
    // this CTA's KSW parts summed in place (one thread per item), then every rank's sum is visible
    for (int idx = tid; idx < RB * mo; idx += NTH) {
      const int tok = idx / RB, row = idx % RB;
      float own = 0.0f;
#pragma unroll
      for (int p = 0; p < KSW; ++p) own += part[p][row][tok];
      part[0][row][tok] = own;
    }
    cluster_sync_all();
  }
  for (int idx = tid * CS + static_cast<int>(rank); idx < RB * mo; idx += NTH * CS) {
    const int tok = idx / RB, row = idx % RB;
    float sum = 0.0f;
    if constexpr (CS > 1) {
      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&part[0][row][tok]));
#pragma unroll
      for (int p = 0; p < CS; ++p) sum += ld_peer_f32(a, static_cast<uint32_t>(p));
    } else {
#pragma unroll
      for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];
    }
    if constexpr (F12) {
      const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];
      const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;
      sum *= f12m_scale(base);
      const float wdec = f12m_wdec(base);
      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        if (j < count) {
          const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);
          sum = fmaf(-xk, wdec, sum);
          sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);
        }
      }
    }
    out[static_cast<long long>(tok) * n_rows + bx * RB + row] = __float2bfloat16_rn(sum);
  }
  if constexpr (CS > 1) cluster_sync_all();   // peers may still read this CTA's sums
}

}  // namespace f12m

@@INSTANCES@@
"""

# (NT, RTW, RG, KSW, NS, D, F12[, CS]) instances compiled into the module; CS (cluster split-K) defaults to 1.
F12M_INSTANCES = ()
# Rendered by the local clang syntax check: F12 and BF16, ring depths 1/2/4, 1-4 token tiles, clusters of 2/4/8.
F12M_CHECK_INSTANCES = ((1, 2, 2, 4, 32, 2, True), (2, 1, 4, 2, 64, 4, True), (4, 2, 2, 2, 8, 1, True),
                        (1, 1, 4, 4, 32, 2, False), (4, 4, 1, 4, 4, 4, False), (2, 1, 1, 8, 8, 2, True, 2),
                        (1, 1, 1, 8, 2, 2, True, 4), (1, 2, 1, 4, 2, 2, True, 8), (2, 1, 1, 8, 16, 4, True, 1, 1),
                        (1, 1, 1, 8, 16, 8, True, 1, 2), (2, 2, 1, 8, 8, 4, True, 1, 2), (2, 1, 1, 4, 32, 2, True, 1, 2, 2),
                        (1, 2, 1, 2, 8, 4, True, 1, 4, 4))


def f12m_unpack(inst):
    """(NT, RTW, RG, KSW, NS, D, F12, CS, XD, TG) from a 7- to 10-tuple (CS defaults to 1, XD to D, TG to 1)."""
    inst = tuple(inst)
    if len(inst) == 7:
        inst += (1,)
    if len(inst) == 8:
        inst += (inst[5],)
    if len(inst) == 9:
        inst += (1,)
    return inst


def f12m_name(inst):
    nt, rtw, rg, ksw, ns, d, f12, cs, xd, tg = f12m_unpack(inst)
    return (f"f12m_run_{nt}_{rtw}_{rg}_{ksw}_{ns}_{d}_{int(f12)}" + (f"_c{cs}" if cs > 1 else "")
            + (f"_x{xd}" if xd != d else "") + (f"_t{tg}" if tg > 1 else ""))


def f12m_smem(inst):
    nt, rtw, rg, ksw, ns, d, f12, cs, xd, tg = f12m_unpack(inst)
    rb = 16 * rg * rtw
    return ksw * rb * 8 * nt * tg * 4 + (2 * rb if f12 else 1) * 16


def f12m_launch(tmpl, rb, threads, cs):
    """Launch statement: <<<>>> for CS = 1 (unchanged), cudaLaunchKernelEx with a cluster of CS otherwise."""
    args = ("reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()),\n"
            "      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(m)")
    if cs == 1:
        return (f"f12m::f12m_kernel<{tmpl}><<<static_cast<unsigned>(n_rows / {rb}), {threads}, 0, stream>>>(\n      {args});")
    return f"""cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((n_rows / {rb}) * {cs}), 1, 1);
  cfg.blockDim = dim3({threads}, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, f12m::f12m_kernel<{tmpl}>, {args});
  TORCH_CHECK(err == cudaSuccess, "f12m cluster launch failed: ", cudaGetErrorString(err));"""


def f12m_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, ns, d, f12, cs, xd, tg = f12m_unpack(inst)
        rb, chs, threads = 16 * rg * rtw, ns * ksw * cs, 32 * rg * ksw * tg
        k = 64 * chs
        tmpl = f"{nt}, {rtw}, {rg}, {ksw}, {ns}, {d}, {'true' if f12 else 'false'}, {cs}, {xd}, {tg}"
        tile_bytes = chs * 1536 if f12 else chs * 2048   # bytes per 16-row tile: 3 or 4 uint4 x 32 lanes per chunk
        wrappers.append(f"""
void {f12m_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows) {{
  TORCH_CHECK(n_rows > 0 && n_rows % {rb} == 0 && m >= 1 && m <= {8 * nt * tg}, "f12m: rows or tokens out of range");
  TORCH_CHECK(x.numel() == m * {k} && out.numel() == m * n_rows, "f12m: x or out has the wrong size");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * {tile_bytes}LL, "f12m: weight planes too small");
  TORCH_CHECK(!{'true' if f12 else 'false'} || rec.numel() >= n_rows * 8, "f12m: records too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "f12m: unaligned");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  {f12m_launch(tmpl, rb, threads, cs)}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
int64_t {f12m_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f12m::f12m_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return F12M_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def f12m_decls(instances):
    out = []
    for inst in instances:
        name = f12m_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows);")
        out.append(f"int64_t {name}_info();")
    return out


def f12m_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12m_decls(instances)
    return load_inline(
        name="okc_proj_f12m", cpp_sources="\n".join(decls), cuda_sources=f12m_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_f12m_"), verbose=False)
