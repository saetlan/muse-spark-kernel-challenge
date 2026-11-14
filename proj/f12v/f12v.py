"""F12V: CUDA-core GEMV (load_inline, sm_90a) for out = x @ W.T at m = 1 on 12-bit lossless weights
(proj/f12v/f12v_layout.py: per-row 16-exponent window, 13 integer ops per 8 weights to FP32).

Work split (compile-time; the instance fixes K = 1024 * NS * KSW):
  CTA    RB = R * RG rows, full K, RG * KSW warps.
  warp   (rg, ks) = (warp / KSW, warp % KSW): rows rg * R .. + R of the CTA over the k blocks
         ks * NS .. + NS (1024 k each: lane l takes k 32 l .. 32 l + 31, three 16-B weight loads per
         row and four 16-B x loads per block); a register ring keeps D blocks in flight, the first
         loads go out in the kernel's first instructions.
  math   FP32 FMAs in the scaled domain (w * 2^(112 - base)), four accumulators per row.
  reduce warp shuffles, then red[ks][row] in shared memory; thread t < RB sums its row over ks in a
         fixed order, scales by 2^(base - 112), applies its row's escapes (x from the copy of x the
         rg == 0 warps leave in shared memory) and stores BF16. Deterministic, one kernel, no
         workspace, no memset.
check_f12v.py parses and runs this file's decode, x conversion, scale and index helpers.
"""

import tempfile

F12V_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace f12v {

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Index helpers (check_f12v.py evaluates their return expressions): row of warp row-group rg, row r
// in CTA bx; k block of warp part ks, step s; uint4 index of a weight slot, an x group, a record half;
// the slot and word holding group g's L words (h = 0: L0, h = 1: L1).
__device__ __forceinline__ long long f12v_row(long long bx, int rg, int r, int rb, int rr) { return bx * rb + rg * rr + r; }
__device__ __forceinline__ int f12v_kb(int ks, int s, int ns) { return ks * ns + s; }
__device__ __forceinline__ long long f12v_widx(long long row, int kbs, int kb, int slot, int lane) { return ((row * kbs + kb) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ int f12v_xidx(int kb, int lane, int g) { return kb * 128 + lane * 4 + g; }
__device__ __forceinline__ long long f12v_ridx(long long row, int h) { return row * 2 + h; }
__device__ __forceinline__ int f12v_lslot(int g) { return 1 + (g >> 1); }
__device__ __forceinline__ int f12v_lword(int g, int h) { return 2 * (g & 1) + h; }

// Decode of one group of 8 weights to FP32 scaled by 2^(112 - base) (format in f12v_layout.py).
__device__ __forceinline__ void f12v_decode8(uint32_t q, uint32_t l0, uint32_t l1, float (&f)[8]) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t h0 = (q & 0x00870087u) | 0x00380038u;
  const uint32_t h1 = (q & 0x87008700u) | 0x38003800u;
  const uint32_t h2 = (r4 & 0x00870087u) | 0x00380038u;
  const uint32_t h3 = (r4 & 0x87008700u) | 0x38003800u;
  f[0] = __uint_as_float(__byte_perm(l0, h0, 0x4055u));
  f[1] = __uint_as_float(__byte_perm(l0, h1, 0x5144u));
  f[2] = __uint_as_float(__byte_perm(l0, h0, 0x6277u));
  f[3] = __uint_as_float(__byte_perm(l0, h1, 0x7366u));
  f[4] = __uint_as_float(__byte_perm(l1, h2, 0x4055u));
  f[5] = __uint_as_float(__byte_perm(l1, h3, 0x5144u));
  f[6] = __uint_as_float(__byte_perm(l1, h2, 0x6277u));
  f[7] = __uint_as_float(__byte_perm(l1, h3, 0x7366u));
}

// Two BF16 x values (low half first) to FP32.
__device__ __forceinline__ void f12v_xpair(uint32_t w, float& lo, float& hi) {
  lo = __uint_as_float(w << 16);
  hi = __uint_as_float(w & 0xFFFF0000u);
}

// 2^(base - 112): undoes the decode's scale. 2^(base - 127): what an escape's stored pattern decodes to.
__device__ __forceinline__ float f12v_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12v_wdec(uint32_t base) { return __uint_as_float(base << 23); }

template <int R, int RG, int KSW, int NS, int D>
__global__ void __launch_bounds__(RG * KSW * 32)
f12v_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint4* __restrict__ x,
            __nv_bfloat16* __restrict__ out) {
  static_assert(D >= 1 && D <= NS && R >= 1 && R <= 32, "f12v config");
  constexpr int RB = R * RG;            // rows per CTA
  constexpr int KBS = NS * KSW;         // 1024-wide k blocks per row
  __shared__ float red[KSW][RB];
  __shared__ uint4 xs[KBS * 128];       // x (BF16), for the escapes
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rg = warp / KSW, ks = warp % KSW;
  const long long bx = blockIdx.x;

  uint4 wv[D][R][3];
  uint4 xv[D][4];
  auto load = [&](int slot, int s) {
    const int kb = f12v_kb(ks, s, NS);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const long long row = f12v_row(bx, rg, r, RB, R);
#pragma unroll
      for (int j = 0; j < 3; ++j) ld_w(wp + f12v_widx(row, KBS, kb, j, lane), wv[slot][r][j]);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) ld_x(x + f12v_xidx(kb, lane, g), xv[slot][g]);
  };
#pragma unroll
  for (int s = 0; s < D; ++s) load(s, s);

  // Row records for the finishing threads: issued behind the first weight loads, used after the reduction.
  const bool fin = tid < RB;
  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);
  if (fin) {
    const long long row = f12v_row(bx, 0, tid, RB, R);
    ld_x(rec + f12v_ridx(row, 0), rc0);
    ld_x(rec + f12v_ridx(row, 1), rc1);
  }

  float acc[R][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int g = 0; g < 4; ++g) acc[r][g] = 0.0f;

#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const int slot = s % D;
    if (rg == 0) {
#pragma unroll
      for (int g = 0; g < 4; ++g) xs[f12v_xidx(f12v_kb(ks, s, NS), lane, g)] = xv[slot][g];
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float xf[8];
#pragma unroll
      for (int t = 0; t < 4; ++t) f12v_xpair(word(xv[slot][g], t), xf[2 * t], xf[2 * t + 1]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4& lv = wv[slot][r][f12v_lslot(g)];
        float f[8];
        f12v_decode8(word(wv[slot][r][0], g), word(lv, f12v_lword(g, 0)), word(lv, f12v_lword(g, 1)), f);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][g] = fmaf(xf[i], f[i], acc[r][g]);
      }
    }
    if (s + D < NS) load(slot, s + D);
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) red[ks][rg * R + r] = v;
  }
  __syncthreads();
  if (fin) {
    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < KSW; ++p) sum += red[p][tid];
    const uint32_t base = rc0.x & 0xFFu, count = (rc0.x >> 8) & 0xFFu;
    sum *= f12v_scale(base);
    const float wdec = f12v_wdec(base);
    const uint32_t esc[7] = {rc0.y, rc0.z, rc0.w, rc1.x, rc1.y, rc1.z, rc1.w};
    const unsigned short* xh = reinterpret_cast<const unsigned short*>(xs);
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      if (j < count) {
        const float xk = __uint_as_float(static_cast<uint32_t>(xh[esc[j] & 0xFFFFu]) << 16);
        sum = fmaf(-xk, wdec, sum);
        sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);
      }
    }
    out[f12v_row(bx, 0, tid, RB, R)] = __float2bfloat16_rn(sum);
  }
}

}  // namespace f12v

@@INSTANCES@@
"""

# (R, RG, KSW, NS, D) instances compiled into the module.
F12V_INSTANCES = ()
# Rendered by the local clang syntax check: ring depth 1, 2 and 4, one k block, several warps along K.
F12V_CHECK_INSTANCES = ((1, 4, 8, 1, 1), (2, 2, 2, 4, 2), (4, 4, 1, 8, 4), (2, 8, 1, 1, 1))


def f12v_name(inst):
    return "f12v_run_" + "_".join(str(int(v)) for v in inst)


def f12v_source(instances):
    wrappers = []
    for inst in instances:
        r, rg, ksw, ns, d = inst
        rb, kbs, threads = r * rg, ns * ksw, 32 * rg * ksw
        wrappers.append(f"""
void {f12v_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  TORCH_CHECK(n_rows > 0 && n_rows % {rb} == 0, "f12v: rows not a multiple of {rb}");
  TORCH_CHECK(x.numel() == {1024 * kbs} && out.numel() == n_rows, "f12v: x or out has the wrong size");
  TORCH_CHECK(wp.numel() >= n_rows * {kbs * 384} && rec.numel() >= n_rows * 8, "f12v: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "f12v: unaligned");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  f12v::f12v_kernel<{r}, {rg}, {ksw}, {ns}, {d}><<<static_cast<unsigned>(n_rows / {rb}), {threads}, 0, stream>>>(
      reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()),
      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
int64_t {f12v_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f12v::f12v_kernel<{r}, {rg}, {ksw}, {ns}, {d}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return F12V_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def f12v_decls(instances):
    out = []
    for inst in instances:
        name = f12v_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"int64_t {name}_info();")
    return out


def f12v_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12v_decls(instances)
    return load_inline(
        name="okc_proj_f12v", cpp_sources="\n".join(decls), cuda_sources=f12v_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_f12v_"), verbose=False)
