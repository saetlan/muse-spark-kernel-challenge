"""Generates attention/local/cuda_w1unit.py: a one-warpgroup wgmma.m64n64k32 e4m3 unit test for the L21 lab
(AF42: W1 runs at speed on the judge but its output is uncorrelated with the reference). W1's own helpers
(gdesc, fences, the CUTLASS-form wgmma asm, mbarrier wait) are copied by anchor from cuda_w1.py, so the
test exercises exactly W1's code. One CTA of 128 threads computes D (64 x 64, f32) = A (64 x 32) . B^T
with A (row-major 64 x 32 e4m3) loaded into each thread's registers by the assumed A layout and B
(n-major 64 x 32 e4m3) written into shared memory in the assumed K-major canonical layout
((n % 8)*16 + (n / 8)*SBO + (k % 16) + (k / 16)*LBO), then D written out by the assumed accumulator
layout. Template knobs (one kernel per variant):
  SBO      B's 8-row group stride (512 = W1's K tile, 528 = W1's Vt)
  SWAP     1: descriptor built with LBO and SBO exchanged (tests the field order)
  CPMODE   0: B by plain 16 B stores + fence.proxy.async + __syncthreads;
           1: cp.async + cp.async.mbarrier.arrive.noinc, wait, consumer-side fence.proxy.async (W1's K path);
           2: cp.async + wait_group 0 + writer-side fence.proxy.async + __syncthreads
  AMODE    0: A register r = rows g + 8*(r & 1), k = 4*t4 + byte + 16*(r >> 1) (CUTLASS ALayout_64x32);
           1: rows g + 8*(r >> 1), k + 16*(r & 1) (the other reading of the register order)
  CHAIN    0: one k32 wgmma on a zeroed accumulator (L21);
           1: K = 64, two chained wgmmas (scale-d 0 then 1, second at start + 2*LBO): W1's Q.K;
           2: as 1 on an accumulator preloaded from D, scale-d 1 then 1: D + A.B^T (accumulate semantics);
           3: W1's consumer step in one warpgroup: S = A.B^T (as 1), P = e4m3(S * pscale) packed by W1's
              formula, O = P.V with V^T laid out as W1's transposers do (SBO_V 528, permuted key order);
           4: as 1 with A from shared memory (SS): A written into a K-major tile by W2's tile_off, W2's
              wgmma2_ss_issue (async pair + commit), a plain wait 0 and an operand fence (W2's SS form on the GPU)
The lab compares D with torch (A.B^T; D0 + A.B^T; e4m3(A.B^T * pscale).V) and prints, for the base
variant, which reference row / column each output row / column matches best.
Run: .venv/bin/python attention/local/make_w1unit.py [--check]
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cuda_w1unit.py")


def load(fname, var):
    ns = {}
    exec(compile(open(os.path.join(HERE, fname)).read(), fname, "exec"), ns)
    return ns[var]


W1 = load("cuda_w1.py", "CUDA_W1_SRC")
a, b = "#include <torch/extension.h>", "constexpr int KC_HEAD"
HEAD = W1[W1.index(a):W1.index(b)].replace("namespace okc_w1 {", "namespace okc_w1u {")
a, b = "__device__ __forceinline__ float ex2f(float x) {", "template <int STAGES"
assert W1.count(a) == 1
HELP = W1[W1.index(a):W1.index(b)]
assert "wgmma_e4m3" in HELP and "gdesc" in HELP and "bar_wait" in HELP and "fence_a" in HELP
W2 = load("cuda_w2.py", "CUDA_W2_SRC")   # CHAIN 4 uses W2's own SS issue / wait helpers and tile layout
a, b = "// ---- W2 helpers", "template <int STAGES, int PP"
assert W2.count(a) == 1
HELP += W2[W2.index(a):W2.index(b)]
assert "wgmma2_ss_issue" in HELP and "wg_wait1()" in HELP and "tile_off" in HELP

KERNEL = r"""template <int SBO, int SWAP, int CPMODE, int AMODE, int CHAIN>
__global__ void __launch_bounds__(128, 1) w1u_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B,
                                                     const uint8_t* __restrict__ Vg, float* __restrict__ D, float pscale) {
  constexpr int LBO = 128;
  constexpr int KD = CHAIN ? 64 : 32;                // k extent of A and B (CHAIN > 0: two chained k32 steps)
  constexpr int VT_OFF = 8 * SBO;                    // CHAIN 3: Vt tile (SBO_V 528) after the B tile; CHAIN 4: A tile
  extern __shared__ __align__(128) uint8_t sm[];
  __shared__ __align__(8) uint64_t bar;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 128;" ::"r"(smem_u32(&bar)));
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  const int n = tid >> 1, ch = tid & 1;
  if (CHAIN == 0) {
    uint8_t* dst = sm + (n & 7) * 16 + (n >> 3) * SBO + ch * LBO;
    if (CPMODE == 0) {
      *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(B + n * 32 + ch * 16);
      proxy_fence();
      __syncthreads();
    } else if (CPMODE == 1) {
      cp16(dst, B + n * 32 + ch * 16);
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&bar)) : "memory");
      bar_wait(&bar, 0);
      proxy_fence();
    } else {
      cp16(dst, B + n * 32 + ch * 16);
      asm volatile("cp.async.commit_group;");
      asm volatile("cp.async.wait_group 0;");
      proxy_fence();
      __syncthreads();
    }
  } else {
#pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int c = 2 * ch + cc;
      *reinterpret_cast<uint4*>(sm + (n & 7) * 16 + (n >> 3) * SBO + c * LBO) =
          *reinterpret_cast<const uint4*>(B + n * 64 + c * 16);
    }
    if (CHAIN == 3) {   // W1's Vt: element (dim dn, k-position 16p + kp) <- V[key 16p + 8*((kp&3)>>1) + 2*(kp>>2) + (kp&1)][dn]
      for (int e = tid; e < 64 * 64; e += 128) {
        const int dn = e >> 6, kap = e & 63, p = kap >> 4, kp = kap & 15;
        const int key = 16 * p + 8 * ((kp & 3) >> 1) + 2 * (kp >> 2) + (kp & 1);
        sm[VT_OFF + (dn & 7) * 16 + (dn >> 3) * 528 + p * LBO + kp] = Vg[key * 64 + dn];
      }
    }
    if (CHAIN == 4) {   // A (64 x 64) into a K-major tile at VT_OFF (W2's tile_off: LBO 128, SBO 512)
      for (int e = tid; e < 64 * 16; e += 128) {
        const int m = e >> 4, k4 = 4 * (e & 15);
        *reinterpret_cast<uint32_t*>(sm + VT_OFF + tile_off(m, k4)) = *reinterpret_cast<const uint32_t*>(A + m * 64 + k4);
      }
    }
    proxy_fence();
    __syncthreads();
  }
  uint32_t a[2][4];
#pragma unroll
  for (int s = 0; s < 2; ++s)
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      const int m = 16 * warp + g + 8 * (AMODE == 0 ? (r & 1) : (r >> 1));
      const int kb = 32 * s + 4 * t4 + 16 * (AMODE == 0 ? (r >> 1) : (r & 1));
      a[s][r] = (CHAIN || s == 0) ? *reinterpret_cast<const uint32_t*>(A + m * KD + kb) : 0u;
    }
  const uint32_t base = smem_u32(sm);
  float d[32];
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int j = i >> 2, half = (i >> 1) & 1, c = i & 1;
    d[i] = CHAIN == 2 ? D[(16 * warp + g + 8 * half) * 64 + 8 * j + 2 * t4 + c] : 0.f;
  }
  fence_regs(d);
  fence_a(a[0]);
  fence_a(a[1]);
  wg_fence();
  if (CHAIN == 0) {
    const uint64_t desc = SWAP ? gdesc(base, SBO, LBO) : gdesc(base, LBO, SBO);
    wgmma_e4m3(d, a[0][0], a[0][1], a[0][2], a[0][3], desc, 0);
  } else if (CHAIN == 4) {
    wgmma2_ss_issue(d, gdesc(base + VT_OFF, 128, 512), gdesc(base + VT_OFF + 256, 128, 512), gdesc(base, LBO, SBO),
                    gdesc(base + 2 * LBO, LBO, SBO));
    wg_wait0();
    fence_regs(d);
  } else {
    wgmma_e4m3(d, a[0][0], a[0][1], a[0][2], a[0][3], gdesc(base, LBO, SBO), CHAIN == 2 ? 1 : 0);
    wgmma_e4m3(d, a[1][0], a[1][1], a[1][2], a[1][3], gdesc(base + 2 * LBO, LBO, SBO), 1);
  }
  wg_commit();
  wg_wait0();
  fence_regs(d);
  fence_a(a[0]);
  fence_a(a[1]);
  if (CHAIN == 3) {   // W1's P pack (k-position 4*t4 + byte <- n-blocks 4c, 4c+1; +16 <- 4c+2, 4c+3), then P.V
    uint32_t pa[2][4];
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      pa[c][0] = e4m3x4(d[16 * c + 0] * pscale, d[16 * c + 1] * pscale, d[16 * c + 4] * pscale, d[16 * c + 5] * pscale);
      pa[c][1] = e4m3x4(d[16 * c + 2] * pscale, d[16 * c + 3] * pscale, d[16 * c + 6] * pscale, d[16 * c + 7] * pscale);
      pa[c][2] = e4m3x4(d[16 * c + 8] * pscale, d[16 * c + 9] * pscale, d[16 * c + 12] * pscale, d[16 * c + 13] * pscale);
      pa[c][3] = e4m3x4(d[16 * c + 10] * pscale, d[16 * c + 11] * pscale, d[16 * c + 14] * pscale, d[16 * c + 15] * pscale);
    }
    float o[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) o[i] = 0.f;
    fence_regs(o);
    fence_a(pa[0]);
    fence_a(pa[1]);
    wg_fence();
    wgmma_e4m3(o, pa[0][0], pa[0][1], pa[0][2], pa[0][3], gdesc(base + VT_OFF, LBO, 528), 0);
    wgmma_e4m3(o, pa[1][0], pa[1][1], pa[1][2], pa[1][3], gdesc(base + VT_OFF + 2 * LBO, LBO, 528), 1);
    wg_commit();
    wg_wait0();
    fence_regs(o);
    fence_a(pa[0]);
    fence_a(pa[1]);
#pragma unroll
    for (int i = 0; i < 32; ++i) d[i] = o[i];
  }
#pragma unroll
  for (int i = 0; i < 32; ++i) {
    const int j = i >> 2, half = (i >> 1) & 1, c = i & 1;
    D[(16 * warp + g + 8 * half) * 64 + 8 * j + 2 * t4 + c] = d[i];
  }
}

}  // namespace okc_w1u

#define OKC_W1U_VARIANTS(X) X(512, 0, 0, 0, 0) X(528, 0, 0, 0, 0) X(512, 1, 0, 0, 0) X(512, 0, 1, 0, 0) \
  X(512, 0, 2, 0, 0) X(528, 0, 1, 0, 0) X(512, 0, 0, 1, 0) X(512, 0, 0, 0, 1) X(512, 0, 0, 0, 2) X(512, 0, 0, 0, 3) \
  X(512, 0, 0, 0, 4)

void w1u(torch::Tensor A, torch::Tensor B, torch::Tensor V, torch::Tensor D, int64_t sbo, int64_t swap,
         int64_t cpmode, int64_t amode, int64_t chain, double pscale) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const int kd = chain ? 64 : 32;
  TORCH_CHECK(A.numel() == 64 * kd && B.numel() == 64 * kd && D.numel() == 64 * 64 && V.numel() == 64 * 64, "w1u: shapes");
  bool launched = false;
#define OKC_W1U_LAUNCH(S, W, C, M, H)                                                                              \
  if (!launched && sbo == S && swap == W && cpmode == C && amode == M && chain == H) {                             \
    okc_w1u::w1u_kernel<S, W, C, M, H><<<1, 128, 8 * S + (H == 3 ? 8 * 528 : H == 4 ? 4096 : 0), stream>>>(                        \
        reinterpret_cast<const uint8_t*>(A.data_ptr()), reinterpret_cast<const uint8_t*>(B.data_ptr()),           \
        reinterpret_cast<const uint8_t*>(V.data_ptr()), D.data_ptr<float>(), static_cast<float>(pscale));         \
    launched = true;                                                                                               \
  }
  OKC_W1U_VARIANTS(OKC_W1U_LAUNCH)
#undef OKC_W1U_LAUNCH
  TORCH_CHECK(launched, "w1u: no variant");
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "w1u launch failed: ", cudaGetErrorString(e));
}
"""

DECL = ("void w1u(torch::Tensor A, torch::Tensor B, torch::Tensor V, torch::Tensor D, int64_t sbo, int64_t swap, "
        "int64_t cpmode, int64_t amode, int64_t chain, double pscale);")


def render():
    src = HEAD + HELP + KERNEL
    return ('"""Generated by make_w1unit.py (do not edit): one-warpgroup wgmma e4m3 unit test (see make_w1unit.py)."""\n\n'
            f"CUDA_W1U_SRC = r\"\"\"\n{src}\"\"\"\n\nCUDA_W1U_DECL = {DECL!r}\n")


if __name__ == "__main__":
    text = render()
    if "--check" in sys.argv:
        same = os.path.exists(OUT) and open(OUT).read() == text
        print("cuda_w1unit.py up to date" if same else "cuda_w1unit.py STALE")
        sys.exit(0 if same else 1)
    open(OUT, "w").write(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
