# X5 + X10: our own skinny GEMM for the dense BF16 weights, y[t, n] = sum_k x[t, k] * W[n, k], with
# up to 32 tokens. prepare() reorders W into mma.sync A-fragment order (lossless permutation) and
# optionally re-encodes it losslessly in 12 bits per weight (X10): an 8-bit plane holding sign and
# mantissa, and a 4-bit plane holding each exponent's offset below its 16 x 16 tile's largest
# exponent. A tile whose exponents span more than 15, or that holds a zero or subnormal, is
# flagged (base 0) and read from the plain array instead (~1.5% of tiles on these weights).
#
# Kernel: a persistent grid over (16-row tile, 512-wide K chunk) items, each CTA a contiguous item
# range. One producer thread streams each item's weight bytes, its 32 tile-base bytes and its token
# rows (padded to 1,040 B) with cp.async.bulk into a 4-stage shared-memory ring (mbarrier transaction
# counts), so no consumer load goes to global memory except raw tiles; NC consumer warps split each chunk's 32 k-steps, run
# mma.sync.m16n8k16 (weights = A, 16 rows; tokens = B, 8 per n-tile, NT n-tiles) with FP32
# accumulators, each warp taking whole k-step pairs. When the row tile changes, the warps' partials are
# reduced in shared memory and added into a persistent zeroed FP32 workspace; the CTA that completes
# a row tile writes BF16 and restores the workspace to zero (graph-safe, no memset or convert kernels).

import tempfile

X5_KC = 512
X5_STEPS = X5_KC // 16
# (enc, nc, nt) instances compiled; stages are fixed per format (plain 6 x 16 KB, enc 8 x 12 KB).
# (enc, nt, mode, nc) instances compiled (mode 1 = loader only); stages hold weights, tile bases and activations.
X5_INSTANCES = ((0, 1, 0, 8), (0, 1, 0, 16), (0, 2, 0, 8), (0, 2, 0, 16), (1, 1, 0, 8), (1, 1, 0, 12), (1, 1, 0, 16),
                (1, 2, 0, 8), (1, 2, 0, 16), (0, 1, 1, 8), (1, 1, 1, 8))
X5_MAX_TOKENS = 16
X5_XROW = X5_KC * 2 + 16

_X5_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace x5 {

constexpr int KC = 512;
constexpr int STEPS = KC / 16;
constexpr int PLAIN_BYTES = 16 * KC * 2;  // one item of the plain fragment-ordered array
constexpr int ENC_B = 16 * KC;            // sign + mantissa plane of one item
constexpr int ENC_I = 16 * KC / 2;        // 4-bit exponent-offset plane of one item

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}
// Waits for the phase with this parity to complete; traps after ~2 s so a protocol bug fails fast
// instead of hanging the run.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  uint32_t done = 0;
  const long long t0 = clock64();
  while (true) {
    asm volatile(
        "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done)
        : "r"(bar), "r"(parity)
        : "memory");
    if (done) return;
    if (clock64() - t0 > 4000000000LL) __trap();
  }
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// One bf16x2 A register from the 12-bit planes: bytes (sign|mantissa) of the pair's two weights
// at byte slots s0/s1 of bw, their exponent offsets iL/iH, and the tile's largest exponent pair bb.
__device__ __forceinline__ uint32_t dec_pair(uint32_t bw, uint32_t sel, uint32_t iL, uint32_t iH, uint32_t bb) {
  const uint32_t X = __byte_perm(bw, 0u, sel);  // 0x00 b1 00 b0
  const uint32_t ms = (X & 0x007F007Fu) | ((X & 0x00800080u) << 8);
  const uint32_t e = (bb - (iL | (iH << 16))) << 7;
  return ms | e;
}

constexpr int XROW = KC * 2 + 16;  // one token's activation slice in smem, padded: 260 words -> no bank conflicts
constexpr int PAIRS = STEPS / 2;    // k-step pairs per chunk; one warp handles whole pairs

template <bool ENC, int NT, int NC>
struct Layout {
  static constexpr int STAGES = NT == 1 ? 4 : 3;
  static constexpr int W = ENC ? ENC_B + ENC_I : PLAIN_BYTES;
  static constexpr int BASES = ENC ? 32 : 0;
  static constexpr int XOFF = W + BASES;
  static constexpr int RAW = XOFF + NT * 8 * XROW;
  static constexpr int STAGE = (RAW + 127) / 128 * 128;
  static constexpr int SCRATCH = NC * 32 * NT * 4 * 4;  // per-warp partial sums for the cross-warp reduction
  static constexpr int SMEM = STAGES * STAGE + SCRATCH + 2 * STAGES * 8;
};

__device__ __forceinline__ void consumer_sync(int nthreads) {
  asm volatile("bar.sync 1, %0;" ::"r"(nthreads) : "memory");
}

// MODE 0: full GEMV. MODE 1: loader only (consumers wait and release stages, no math).
// Output: BF16 out[t, n]. Partial sums go to a persistent FP32 workspace ws[t, n] (all zero between
// calls) with atomics; the CTA that completes a row tile (per-tile item counter cnt[rt] reaches
// K / KC) reads the tile back from L2, writes BF16, and restores ws and cnt[rt] to zero.
template <bool ENC, int NT, int MODE, int NC>
__global__ void __launch_bounds__((NC + 1) * 32) gemv_kernel(
    const uint8_t* __restrict__ wa, const uint8_t* __restrict__ wi, const uint8_t* __restrict__ bases,
    const uint4* __restrict__ wraw, const uint8_t* __restrict__ x, float* __restrict__ ws, int* __restrict__ cnt,
    __nv_bfloat16* __restrict__ out, int N, int K, int T, int items) {
  using L = Layout<ENC, NT, NC>;
  constexpr int STAGES = L::STAGES;
  extern __shared__ __align__(1024) uint8_t smem[];
  float* scratch = reinterpret_cast<float*>(smem + STAGES * L::STAGE);
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + STAGES * L::STAGE + L::SCRATCH);
  uint64_t* empty = full + STAGES;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int chunks = K / KC;
  const int begin = static_cast<int>(static_cast<long long>(items) * blockIdx.x / gridDim.x);
  const int end = static_cast<int>(static_cast<long long>(items) * (blockIdx.x + 1) / gridDim.x);
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_addr(full + s), 1);
      mbar_init(smem_addr(empty + s), NC);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();

  if (warp == NC) {  // producer warp: one thread streams weights, tile bases and activation rows
    if (lane == 0) {
      const uint32_t xbytes = static_cast<uint32_t>(T) * (KC * 2);
      for (int it = begin, n = 0; it < end; ++it, ++n) {
        const int s = n % STAGES;
        mbar_wait(smem_addr(empty + s), ((n / STAGES) & 1) ^ 1);
        const int rt = it / chunks, c = it - rt * chunks;
        const uint32_t dst = smem_addr(smem + s * L::STAGE);
        const uint32_t bar = smem_addr(full + s);
        mbar_expect_tx(bar, L::W + L::BASES + xbytes);
        if constexpr (ENC) {
          bulk_g2s(dst, wa + static_cast<size_t>(it) * ENC_B, ENC_B, bar);
          bulk_g2s(dst + ENC_B, wi + static_cast<size_t>(it) * ENC_I, ENC_I, bar);
          bulk_g2s(dst + L::W, bases + static_cast<size_t>(rt) * (K / 16) + c * STEPS, 32, bar);
        } else {
          bulk_g2s(dst, wa + static_cast<size_t>(it) * PLAIN_BYTES, PLAIN_BYTES, bar);
        }
        for (int tok = 0; tok < T; ++tok)
          bulk_g2s(dst + L::XOFF + tok * XROW, x + (static_cast<size_t>(tok) * K + c * KC) * 2, KC * 2, bar);
      }
    }
    return;
  }

  const int g = lane >> 2, t = lane & 3;
  bool xok[NT];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) xok[nt] = nt * 8 + g < T;
  float acc[NT][4];
#pragma unroll
  for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
  int cur = -1, nitems = 0;

  // Cross-warp reduction, atomics into ws by warp 0, then the last-arriver BF16 write for tile rt.
  auto flush = [&](int rt) {
    float* mine = scratch + (warp * 32 + lane) * (NT * 4);
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r) mine[nt * 4 + r] = acc[nt][r];
    consumer_sync(NC * 32);
    if (warp == 0) {
      float tot[NT][4];
#pragma unroll
      for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int r = 0; r < 4; ++r) tot[nt][r] = 0.0f;
      for (int w = 0; w < NC; ++w) {
        const float* p = scratch + (w * 32 + lane) * (NT * 4);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
          for (int r = 0; r < 4; ++r) tot[nt][r] += p[nt * 4 + r];
      }
      const int r0 = rt * 16 + g;
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const int t0 = nt * 8 + 2 * t;
        if (t0 < T) {
          atomicAdd(ws + static_cast<size_t>(t0) * N + r0, tot[nt][0]);
          atomicAdd(ws + static_cast<size_t>(t0) * N + r0 + 8, tot[nt][2]);
        }
        if (t0 + 1 < T) {
          atomicAdd(ws + static_cast<size_t>(t0 + 1) * N + r0, tot[nt][1]);
          atomicAdd(ws + static_cast<size_t>(t0 + 1) * N + r0 + 8, tot[nt][3]);
        }
      }
      __threadfence();
      __syncwarp();  // every lane's atomics and fence come before lane 0's counter update
      int last = 0;
      if (lane == 0) last = (atomicAdd(cnt + rt, nitems) + nitems == chunks);
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        __threadfence();
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          const int t0 = nt * 8 + 2 * t;
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            if (t0 + h < T) {
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) {
                float* src = ws + static_cast<size_t>(t0 + h) * N + r0 + 8 * rr;
                const float v = __ldcg(src);
                out[static_cast<size_t>(t0 + h) * N + r0 + 8 * rr] = __float2bfloat16_rn(v);
                *src = 0.0f;
              }
            }
          }
        }
        if (lane == 0) cnt[rt] = 0;
      }
    }
    consumer_sync(NC * 32);  // warp 0 is done with scratch before it is reused
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;
  };

  for (int it = begin, n = 0; it < end; ++it, ++n) {
    const int rt = it / chunks, c = it - rt * chunks;
    if (rt != cur) {
      if (cur >= 0 && MODE == 0) flush(cur);
      cur = rt;
      nitems = 0;
    }
    ++nitems;
    const int s = n % STAGES;
    mbar_wait(smem_addr(full + s), (n / STAGES) & 1);
    if constexpr (MODE == 0) {
      const uint8_t* st = smem + s * L::STAGE;
      const uint32_t* xs = reinterpret_cast<const uint32_t*>(st + L::XOFF);
#pragma unroll 2
      for (int sp = warp; sp < PAIRS; sp += NC) {
        uint32_t a[2][4];
        if constexpr (ENC) {
          const uint4 bv = reinterpret_cast<const uint4*>(st)[sp * 32 + lane];
          const uint2 iv = reinterpret_cast<const uint2*>(st + ENC_B)[sp * 32 + lane];
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const int j = 2 * sp + q;
            const uint32_t base = st[L::W + j];
            if (base == 0) {  // raw tile: plain fragment words from global memory (~1.4% of tiles)
              const uint4 v = wraw[(static_cast<size_t>(rt) * chunks + c) * (STEPS * 32) + j * 32 + lane];
              a[q][0] = v.x; a[q][1] = v.y; a[q][2] = v.z; a[q][3] = v.w;
            } else {
              const uint32_t w0 = q ? bv.z : bv.x, w1 = q ? bv.w : bv.y;
              const uint32_t bb = base | (base << 16);
              const int sh = 16 * q;
              a[q][0] = dec_pair(w0, 0x4140u, (iv.x >> (sh + 0)) & 0xFu, (iv.y >> (sh + 0)) & 0xFu, bb);
              a[q][1] = dec_pair(w0, 0x4342u, (iv.x >> (sh + 4)) & 0xFu, (iv.y >> (sh + 4)) & 0xFu, bb);
              a[q][2] = dec_pair(w1, 0x4140u, (iv.x >> (sh + 8)) & 0xFu, (iv.y >> (sh + 8)) & 0xFu, bb);
              a[q][3] = dec_pair(w1, 0x4342u, (iv.x >> (sh + 12)) & 0xFu, (iv.y >> (sh + 12)) & 0xFu, bb);
            }
          }
        } else {
#pragma unroll
          for (int q = 0; q < 2; ++q) {
            const uint4 v = reinterpret_cast<const uint4*>(st)[(2 * sp + q) * 32 + lane];
            a[q][0] = v.x; a[q][1] = v.y; a[q][2] = v.z; a[q][3] = v.w;
          }
        }
#pragma unroll
        for (int q = 0; q < 2; ++q) {
          const int j = 2 * sp + q;
#pragma unroll
          for (int nt = 0; nt < NT; ++nt) {
            uint32_t b0 = 0u, b1 = 0u;
            if (xok[nt]) {
              const uint32_t* xr = xs + (nt * 8 + g) * (XROW / 4) + j * 8 + t;
              b0 = xr[0];
              b1 = xr[4];
            }
            mma(acc[nt], a[q][0], a[q][1], a[q][2], a[q][3], b0, b1);
          }
        }
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(smem_addr(empty + s));
  }
  if (cur >= 0 && MODE == 0) flush(cur);
}

}  // namespace x5

#define X5_CASE(E, NT, MODE, NC)                                                                               \
  if (enc == (E) && nt == (NT) && mode == (MODE) && nc == (NC)) {                                                \
    constexpr int SM = x5::Layout<(E) != 0, NT, NC>::SMEM;                                                       \
    static bool ready = false;                                                                                   \
    if (!ready) {                                                                                                \
      cudaFuncSetAttribute(x5::gemv_kernel<(E) != 0, NT, MODE, NC>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                           SM);                                                                                  \
      ready = true;                                                                                              \
    }                                                                                                            \
    x5::gemv_kernel<(E) != 0, NT, MODE, NC><<<static_cast<int>(grid), (NC + 1) * 32, SM, stream>>>(              \
        reinterpret_cast<const uint8_t*>(wa.data_ptr()), reinterpret_cast<const uint8_t*>(wi.data_ptr()),       \
        reinterpret_cast<const uint8_t*>(bases.data_ptr()), reinterpret_cast<const uint4*>(wraw.data_ptr()),    \
        reinterpret_cast<const uint8_t*>(x.data_ptr()), ws.data_ptr<float>(), cnt.data_ptr<int>(),              \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(N), static_cast<int>(K),             \
        static_cast<int>(T), static_cast<int>(items));                                                           \
    return out;                                                                                                  \
  }

torch::Tensor x5_gemv(torch::Tensor wa, torch::Tensor wi, torch::Tensor bases, torch::Tensor wraw, torch::Tensor x,
                      torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t N, int64_t K, int64_t T,
                      int64_t enc, int64_t nc, int64_t nt, int64_t grid, int64_t mode) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t items = (N / 16) * (K / x5::KC);
  X5_CASE(0, 1, 0, 8) X5_CASE(0, 1, 0, 16) X5_CASE(0, 2, 0, 8) X5_CASE(0, 2, 0, 16)
  X5_CASE(1, 1, 0, 8) X5_CASE(1, 1, 0, 12) X5_CASE(1, 1, 0, 16) X5_CASE(1, 2, 0, 8) X5_CASE(1, 2, 0, 16)
  X5_CASE(0, 1, 1, 8) X5_CASE(1, 1, 1, 8)
  TORCH_CHECK(false, "x5_gemv: no instance for enc=", enc, " nc=", nc, " nt=", nt, " mode=", mode);
  return out;
}
"""

_X5_CPP = """
torch::Tensor x5_gemv(torch::Tensor wa, torch::Tensor wi, torch::Tensor bases, torch::Tensor wraw, torch::Tensor x,
                      torch::Tensor ws, torch::Tensor cnt, torch::Tensor out, int64_t N, int64_t K, int64_t T,
                      int64_t enc, int64_t nc, int64_t nt, int64_t grid, int64_t mode);
"""


def x5_source():
    return _X5_CUDA


def x5_build():
    return load_inline(
        name="x5_ext", cpp_sources=_X5_CPP, cuda_sources=_X5_CUDA, functions=["x5_gemv"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="x5_"), verbose=False)


def x5_frag_maps(device):
    """Row and column inside a 16 x 16 tile of each lane's 8 A elements, in register order
    (a0..a7 of mma.m16n8k16: rows g, g+8; columns 2t, 2t+1, 2t+8, 2t+9)."""
    lane = torch.arange(32, device=device)
    g, t = lane // 4, lane % 4
    rows = torch.stack([g, g, g + 8, g + 8, g, g, g + 8, g + 8], 1)
    cols = torch.stack([2 * t, 2 * t + 1, 2 * t, 2 * t + 1, 2 * t + 8, 2 * t + 9, 2 * t + 8, 2 * t + 9], 1)
    return rows, cols


def x5_pack(w):
    """W bf16 [N, K] -> dict of the plain fragment-ordered array and the 12-bit planes.
    plain: int16 [N/16, K/16, 32, 8] (item it = rt * (K/512) + c is bytes [it * 16384, +16384));
    enc_b: uint8 [N/16, K/512, 16, 32, 16]; enc_i: int32 [N/16, K/512, 16, 32, 2]; bases uint8
    [N/16, K/16] (0 = raw tile)."""
    n, k = w.shape
    assert n % 16 == 0 and k % X5_KC == 0 and w.dtype == torch.bfloat16
    rows, cols = x5_frag_maps(w.device)
    wt = w.contiguous().view(n // 16, 16, k // 16, 16).permute(0, 2, 1, 3)  # [rt, ks, row, col]
    frag = wt[:, :, rows, cols].contiguous()  # adjacent advanced indices: [rt, ks, lane, e]
    plain = frag.view(torch.int16)
    bits = plain.to(torch.int32) & 0xFFFF
    e8 = (bits >> 7) & 0xFF
    maxe = e8.amax(dim=(2, 3))  # [rt, ks]
    idx = maxe[:, :, None, None] - e8
    raw = (idx > 15).any(dim=3).any(dim=2) | (e8 == 0).any(dim=3).any(dim=2)
    bases = torch.where(raw, torch.zeros_like(maxe), maxe).to(torch.uint8)
    idx = torch.where(raw[:, :, None, None], torch.zeros_like(idx), idx)
    byte = (((bits >> 8) & 0x80) | (bits & 0x7F)).to(torch.uint8)
    rt, ks = n // 16, k // 16
    chunks = k // X5_KC
    byte = byte.view(rt, chunks, X5_STEPS // 2, 2, 32, 8).permute(0, 1, 2, 4, 3, 5).contiguous()
    i6 = idx.view(rt, chunks, X5_STEPS // 2, 2, 32, 4, 2)  # [rt, c, sp, q, lane, m, eo]
    i6 = i6.permute(0, 1, 2, 4, 6, 3, 5)  # [rt, c, sp, lane, eo, q, m]
    shifts = (4 * (4 * torch.arange(2, device=w.device)[:, None] + torch.arange(4, device=w.device)[None, :])).view(1, 1, 1, 1, 1, 2, 4)
    words = (i6.to(torch.int64) << shifts).sum(dim=(5, 6))  # [rt, c, sp, lane, eo]
    enc_i = words.to(torch.int64).bitwise_and(0xFFFFFFFF)
    enc_i = torch.where(enc_i >= 2 ** 31, enc_i - 2 ** 32, enc_i).to(torch.int32).contiguous()
    return {"n": n, "k": k, "plain": plain, "enc_b": byte, "enc_i": enc_i, "bases": bases.contiguous(),
            "raw_tiles": int(raw.sum()), "tiles": rt * ks,
            # zero between calls: the last CTA of each row tile restores both after writing BF16
            "ws": torch.zeros((X5_MAX_TOKENS, n), dtype=torch.float32, device=w.device),
            "cnt": torch.zeros((n // 16,), dtype=torch.int32, device=w.device)}


def x5_enc_bytes(pack):
    """Bytes a 12-bit pass reads: both planes, the tile bases, and the raw tiles' plain words."""
    return pack["enc_b"].numel() + pack["enc_i"].numel() * 4 + pack["bases"].numel() + pack["raw_tiles"] * 512


def x5_gemv(x, pack, ext, enc, nc, nt, grid, mode=0):
    """x bf16 [T, K] (T <= 8 * nt <= 16) -> bf16 [T, N], written by the kernel itself."""
    t = x.shape[0]
    assert t <= 8 * nt and t <= X5_MAX_TOKENS
    out = torch.empty((t, pack["n"]), dtype=torch.bfloat16, device=x.device)
    wa = pack["enc_b"] if enc else pack["plain"]
    ext.x5_gemv(wa, pack["enc_i"], pack["bases"], pack["plain"], x.contiguous(), pack["ws"], pack["cnt"], out,
                pack["n"], pack["k"], t, enc, nc, nt, grid, mode)
    return out
