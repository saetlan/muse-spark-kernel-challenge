
# ---------------------------------------------------------------------------------------------
# v4 (R14): Hopper W4A16 NVFP4 grouped expert GEMMs in CUDA (local/make_v4.py inserts this after
# v2/gmm.py).
#
# Swapped operands: the expert weights are wgmma's A operand, decoded from NVFP4 straight into the
# register fragment (A from registers, "RS"); the tokens are the B operand (N = 16..128 per tile), read
# from shared memory through a 128B-swizzle descriptor. prepare() repacks the weights losslessly into
# fragment order (one 32-bit word = one thread's 8 values of one k16 step; nibble positions picked so
# Marlin's shift trick yields the 4 A registers in order) with the E4M3 scales as exact bf16 (s * 2^64),
# and for GEMM1 interleaves gate and up rows per 8 so each thread holds gate (row g) and up (row g + 8)
# of the same output column: SwiGLU runs on the accumulators. A routing kernel sorts the local routes by
# expert into 16-row-aligned segments and cuts them into tiles of 128/64/32/16 rows; a gather kernel
# writes the activations K-chunked and pre-swizzled, so every pipeline stage is two 1-D bulk copies
# (cp.async.bulk + mbarrier; no tensor maps). One producer warp feeds a STAGES-deep ring; two consumer
# warpgroups decode (double-buffered across wgmma.wait_group) and issue wgmma. GEMM1 writes the SwiGLU
# output K-chunked and swizzled for GEMM2; GEMM2 writes per-route rows for v2's post-norm combine.
# Every mbarrier wait is bounded: a protocol fault sets an error flag instead of hanging, and prepare()
# keeps the fallback whenever the flag is set or the results disagree with the reference.
# ---------------------------------------------------------------------------------------------

import tempfile

from torch.utils.cpp_extension import load_inline

R14_STAGES = (4, 6)
# class -> pipeline depth, for the token classes of R3_CLASSES-like bounds below; empty = off.
R14_CONFIGS = {}
R14_CLASSES = ((1200, "t1"), (8192, "t4"))
R14_REP_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
R14_OUT_SHIFT = 2.0 ** 62
R14_SEG = 16       # expert segments start on 16-row boundaries (tiles may be 16 rows)
R14_BT = 128       # largest tile (wgmma N)
R14_MAX_ERR = 1e-2
R14_BLOB = 5120    # bytes per (expert, 128-row block, 64-wide K chunk): 4096 B of codes + 1024 B of scales


def _r14_wgmma_fn(n):
    regs = n // 2
    d = ", ".join(f"%{i}" for i in range(regs))
    a = ", ".join(f"%{regs + i}" for i in range(4))
    asm = (r"{\n.reg .pred p;\nsetp.ne.b32 p, %" + str(regs + 5) + r", 0;\n"
           f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, {{{a}}}, %{regs + 4}, p, 1, 1, 0;"
           + r"\n}\n")
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    return (f"__device__ __forceinline__ void wgmma{n}(float (&d)[{regs}], uint32_t a0, uint32_t a1, uint32_t a2, "
            f"uint32_t a3, uint64_t desc, int sc) {{\n  asm volatile(\"{asm}\"\n      : {outs}\n"
            f"      : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc), \"r\"(sc));\n}}\n")


_R14_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace r14 {

constexpr int K = 4096;
constexpr int KC = K / 64;            // 64-wide K chunks
constexpr int W_BYTES = 4096;         // codes per stage: 2 WG x 4 warps x 32 lanes x 4 words
constexpr int S_BYTES = 1024;         // bf16 scales per stage: 2 WG x 4 warps x 8 rows-pairs x 4 k16 x 2
constexpr int WS_BYTES = W_BYTES + S_BYTES;
constexpr int BT = 128;               // largest token tile
constexpr int B_BYTES = BT * 128;     // one swizzled B tile: BT rows x 64 bf16
constexpr long long SPIN_LIMIT = 1ll << 22;
constexpr int LOCAL = 32;
constexpr int SEG = 16;

template <int R>
__device__ __forceinline__ uint32_t dec(uint32_t w) {
  // E2M1 nibbles at bits [12 - 4R, 16 - 4R) of each 16-bit lane -> bf16 lanes holding value * 2^-126.
  const uint32_t x = w << (4 * R);
  return (x & 0x80008000u) | ((x >> 6) & 0x01C001C0u);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// One k16 step of this thread's A fragment: word wd holds (a6 a4 a2 a0 | a7 a5 a3 a1) nibbles, sw holds
// bf16 scales (row g low, row g + 8 high) * 2^64; out = a0..a7 as 4 bf16x2 registers, value * scale * 2^-62.
__device__ __forceinline__ void dec_k16(uint32_t wd, uint32_t sw, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                        uint32_t& r3) {
  const uint32_t sg = __byte_perm(sw, 0u, 0x1010u);
  const uint32_t sg8 = __byte_perm(sw, 0u, 0x3232u);
  r0 = hmul2(dec<0>(wd), sg);
  r1 = hmul2(dec<1>(wd), sg8);
  r2 = hmul2(dec<2>(wd), sg);
  r3 = hmul2(dec<3>(wd), sg8);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
               ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}

// Bounded parity wait: false after SPIN_LIMIT polls, or at once when another CTA already flagged an error
// (so a protocol fault ends every CTA quickly instead of hanging the run); the caller flags and carries on.
__device__ __forceinline__ bool mbar_wait(uint32_t bar, uint32_t parity, const int* err) {
  for (long long i = 0; i < SPIN_LIMIT; ++i) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return true;
    if ((i & 1023) == 1023 && *reinterpret_cast<const volatile int*>(err) != 0) return false;
  }
  return false;
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void keep(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void keepf(float& r) { asm volatile("" : "+f"(r)::"memory"); }

__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// K-major B tile, 128B swizzle: start address, LBO 1 (unused when swizzled), SBO = 1024 B (8 rows x 128 B).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

@@WGMMA@@

template <int N>
__device__ __forceinline__ void mma_k64(float (&acc)[N / 2], const uint32_t (&a)[16], uint64_t desc, int first);

#define R14_MMA_K64(NN)                                                                             \
  template <>                                                                                       \
  __device__ __forceinline__ void mma_k64<NN>(float (&acc)[NN / 2], const uint32_t (&a)[16], uint64_t desc, \
                                              int first) {                                          \
    wgmma##NN(acc, a[0], a[1], a[2], a[3], desc, first ? 0 : 1);                                    \
    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \
    wgmma##NN(acc, a[8], a[9], a[10], a[11], desc + 4, 1);                                          \
    wgmma##NN(acc, a[12], a[13], a[14], a[15], desc + 6, 1);                                        \
  }
R14_MMA_K64(16)
R14_MMA_K64(32)
R14_MMA_K64(64)
R14_MMA_K64(128)

template <int STAGES>
__device__ __forceinline__ bool load_frag(const uint8_t* sW, uint32_t full0, int kc, int woff, int soff,
                                          uint32_t (&a)[16], const int* err) {
  const int s = kc % STAGES;
  const bool ok = mbar_wait(full0 + 8 * s, (kc / STAGES) & 1, err);
  const uint4 wv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + woff);
  const uint4 sv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + soff);
  dec_k16(wv.x, sv.x, a[0], a[1], a[2], a[3]);
  dec_k16(wv.y, sv.y, a[4], a[5], a[6], a[7]);
  dec_k16(wv.z, sv.z, a[8], a[9], a[10], a[11]);
  dec_k16(wv.w, sv.w, a[12], a[13], a[14], a[15]);
  return ok;
}

template <int STAGES, int N>
__device__ __forceinline__ void mainloop_step(float (&acc)[N / 2], const uint32_t (&cur)[16], uint32_t (&nxt)[16],
                                              uint32_t sB0, const uint8_t* sW, uint32_t full0, uint32_t empty0,
                                              int kc, int woff, int soff, int tid, bool& ok, const int* err) {
  const int s = kc % STAGES;
  wg_fence();
  mma_k64<N>(acc, cur, make_desc(sB0 + s * B_BYTES), kc == 0);
  wg_commit();
  wg_wait<1>();  // the group of step kc - 1 is done: its A registers (nxt) and its stage are free
#pragma unroll
  for (int i = 0; i < 16; ++i) keep(nxt[i]);
  if (kc >= 1 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((kc - 1) % STAGES));
  if (kc + 1 < KC) ok &= load_frag<STAGES>(sW, full0, kc + 1, woff, soff, nxt, err);
}

template <int STAGES, bool GATED, int N>
__device__ __forceinline__ void consume(uint8_t* sB, const uint8_t* sW, uint32_t full0, uint32_t empty0, int tid,
                                        int e, int row0, int valid, int nb, const float* __restrict__ cscale,
                                        __nv_bfloat16* __restrict__ out, int* __restrict__ err, int P) {
  const int c = (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int woff = ((c * 4 + w) * 32 + lane) * 16;
  const int soff = W_BYTES + ((c * 4 + w) * 8 + g) * 16;
  const uint32_t sB0 = smem_u32(sB);
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t fa[16], fb[16];
  bool ok = load_frag<STAGES>(sW, full0, 0, woff, soff, fa, err);
  for (int kc = 0; kc < KC; kc += 2) {
    mainloop_step<STAGES, N>(acc, fa, fb, sB0, sW, full0, empty0, kc, woff, soff, tid, ok, err);
    mainloop_step<STAGES, N>(acc, fb, fa, sB0, sW, full0, empty0, kc + 1, woff, soff, tid, ok, err);
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) keepf(acc[i]);
  if (!ok) atomicAdd(err, 1);
  named_sync(1, 256);  // both consumer warpgroups are done with the B tiles: reuse them for staging
  const float cs = cscale[e];
  uint8_t* stage = sB;
  if (GATED) {
    // Rows g (gate) and g + 8 (up) of output column col; tokens 8j + 2q (+1). Staged swizzled as GEMM2's B.
    const int col = c * 32 + w * 8 + g;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = 8 * j + 2 * q + h;
        const float gt = acc[4 * j + h] * cs;
        const float up = acc[4 * j + 2 + h] * cs;
        const float v = gt / (1.f + expf(-gt)) * up;
        *reinterpret_cast<__nv_bfloat16*>(stage + t * 128 + ((((col >> 3) ^ (t & 7))) << 4) + (col & 7) * 2) =
            __float2bfloat16_rn(v);
      }
    }
    named_sync(1, 256);
    const int rows = (valid + 7) & ~7;
    uint4* dst = reinterpret_cast<uint4*>(out + (static_cast<size_t>(nb) * P + row0) * 64);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = tid - 128; i < rows * 8; i += 256) dst[i] = src[i];
  } else {
    // Rows (output columns) c*64 + w*16 + g (+8); tokens 8j + 2q (+1). Staged as [token][128 cols].
    const int r = c * 64 + w * 16 + g;
#pragma unroll
    for (int j = 0; j < N / 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = 8 * j + 2 * q + h;
        *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + r * 2) = __float2bfloat16_rn(acc[4 * j + h] * cs);
        *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + (r + 8) * 2) = __float2bfloat16_rn(acc[4 * j + 2 + h] * cs);
      }
    }
    named_sync(1, 256);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = tid - 128; i < valid * 16; i += 256) {
      const int t = i >> 4, u = i & 15;
      reinterpret_cast<uint4*>(out + static_cast<size_t>(row0 + t) * K + nb * 128)[u] = src[i];
    }
  }
}

// grid (NB, max pieces), 384 threads: warpgroup 0 = producer (one thread), 1-2 = consumers.
// piece = (expert, first row, valid rows, tile rows N). B source: [KC][P][64] swizzled bf16.
template <int STAGES, bool GATED>
__global__ void __launch_bounds__(384, 1) gemm_kernel(const uint8_t* __restrict__ wblob,
                                                      const __nv_bfloat16* __restrict__ bsrc,
                                                      const int4* __restrict__ pieces, const int* __restrict__ n_pieces,
                                                      const float* __restrict__ cscale, __nv_bfloat16* __restrict__ out,
                                                      int* __restrict__ err, int P, int NB) {
  const int nb = blockIdx.x, pc = blockIdx.y;
  if (pc >= *n_pieces) return;
  const int4 piece = pieces[pc];
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sB = smem;
  uint8_t* sW = smem + STAGES * B_BYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + STAGES * WS_BYTES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + STAGES);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int e = piece.x, row0 = piece.y, valid = piece.z, nsel = piece.w;
  if (tid < 128) {
    if (tid == 0) {
      const uint8_t* wsrc = wblob + static_cast<size_t>(e * NB + nb) * KC * WS_BYTES;
      const uint32_t bbytes = static_cast<uint32_t>(nsel) * 128u;
      bool ok = true;
      for (int kc = 0; kc < KC; ++kc) {
        const int s = kc % STAGES;
        if (kc >= STAGES) ok &= mbar_wait(empty0 + 8 * s, ((kc / STAGES) - 1) & 1, err);
        mbar_expect_tx(full0 + 8 * s, WS_BYTES + bbytes);
        bulk_g2s(smem_u32(sW + s * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
        bulk_g2s(smem_u32(sB + s * B_BYTES), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * s);
      }
      if (!ok) atomicAdd(err, 1);
    }
    return;
  }
  switch (nsel) {
    case 16: consume<STAGES, GATED, 16>(sB, sW, full0, empty0, tid, e, row0, valid, nb, cscale, out, err, P); break;
    case 32: consume<STAGES, GATED, 32>(sB, sW, full0, empty0, tid, e, row0, valid, nb, cscale, out, err, P); break;
    case 64: consume<STAGES, GATED, 64>(sB, sW, full0, empty0, tid, e, row0, valid, nb, cscale, out, err, P); break;
    default: consume<STAGES, GATED, 128>(sB, sW, full0, empty0, tid, e, row0, valid, nb, cscale, out, err, P); break;
  }
}

// One block: per-expert counts of the local routes, 16-aligned segments, tiles of 128/64/32/16 rows,
// and each route's row (pos, -1 if not local). Rows inside a segment are in arbitrary order (results
// don't depend on it).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ ids, int slots, int* __restrict__ pos,
                                                     int4* __restrict__ pieces, int* __restrict__ n_pieces,
                                                     int max_pieces) {
  __shared__ int cnt[LOCAL], cur[LOCAL], start[LOCAL + 1];
  const int tid = threadIdx.x;
  if (tid < LOCAL) cnt[tid] = 0;
  __syncthreads();
  for (int i = tid; i < slots; i += blockDim.x) {
    const int e = ids[i];
    if (e >= 0) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (tid == 0) {
    start[0] = 0;
    int np = 0;
    for (int e = 0; e < LOCAL; ++e) {
      start[e + 1] = start[e] + ((cnt[e] + SEG - 1) / SEG) * SEG;
      int rem = cnt[e], r = start[e];
      while (rem > 0) {
        const int n = rem >= 128 ? 128 : rem >= 64 ? 64 : rem >= 32 ? 32 : 16;
        const int v = rem < n ? rem : n;
        if (np < max_pieces) pieces[np] = make_int4(e, r, v, n);
        ++np;
        r += n;
        rem -= v;
      }
    }
    *n_pieces = np < max_pieces ? np : max_pieces;
  }
  __syncthreads();
  if (tid < LOCAL) cur[tid] = start[tid];
  __syncthreads();
  for (int i = tid; i < slots; i += blockDim.x) {
    const int e = ids[i];
    pos[i] = e >= 0 ? atomicAdd(&cur[e], 1) : -1;
  }
}

}  // namespace r14

void r14_gemm(torch::Tensor wblob, torch::Tensor bsrc, torch::Tensor pieces, torch::Tensor n_pieces,
              torch::Tensor cscale, torch::Tensor out, torch::Tensor err, int64_t P, int64_t NB,
              int64_t max_pieces, int64_t stages, bool gated) {
  const size_t smem = static_cast<size_t>(stages) * (r14::B_BYTES + r14::WS_BYTES) + 16 * stages + 1024;
  const dim3 grid(static_cast<unsigned>(NB), static_cast<unsigned>(max_pieces));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define R14_LAUNCH(S, G)                                                                                \
  do {                                                                                                  \
    auto kern = r14::gemm_kernel<S, G>;                                                                 \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));    \
    kern<<<grid, 384, smem, stream>>>(wblob.data_ptr<uint8_t>(),                                        \
                                      reinterpret_cast<const __nv_bfloat16*>(bsrc.data_ptr()),          \
                                      reinterpret_cast<const int4*>(pieces.data_ptr()),                 \
                                      n_pieces.data_ptr<int>(), cscale.data_ptr<float>(),               \
                                      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),                 \
                                      err.data_ptr<int>(), static_cast<int>(P), static_cast<int>(NB));  \
  } while (0)
  if (stages == 4) {
    if (gated) R14_LAUNCH(4, true); else R14_LAUNCH(4, false);
  } else {
    if (gated) R14_LAUNCH(6, true); else R14_LAUNCH(6, false);
  }
#undef R14_LAUNCH
}

// Registers and local (spill) bytes of each GEMM instance: (stages 4, gated), (4, plain), (6, gated), (6, plain).
torch::Tensor r14_attrs() {
  int64_t v[8];
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, r14::gemm_kernel<4, true>);
  v[0] = a.numRegs; v[1] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<4, false>);
  v[2] = a.numRegs; v[3] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<6, true>);
  v[4] = a.numRegs; v[5] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<6, false>);
  v[6] = a.numRegs; v[7] = static_cast<int64_t>(a.localSizeBytes);
  return torch::tensor({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
}

void r14_route(torch::Tensor ids, torch::Tensor pos, torch::Tensor pieces, torch::Tensor n_pieces,
               int64_t max_pieces) {
  r14::route_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), static_cast<int>(ids.numel()), pos.data_ptr<int>(),
      reinterpret_cast<int4*>(pieces.data_ptr()), n_pieces.data_ptr<int>(), static_cast<int>(max_pieces));
}
"""

_R14_CPP = """
void r14_gemm(torch::Tensor wblob, torch::Tensor bsrc, torch::Tensor pieces, torch::Tensor n_pieces,
              torch::Tensor cscale, torch::Tensor out, torch::Tensor err, int64_t P, int64_t NB,
              int64_t max_pieces, int64_t stages, bool gated);
void r14_route(torch::Tensor ids, torch::Tensor pos, torch::Tensor pieces, torch::Tensor n_pieces, int64_t max_pieces);
torch::Tensor r14_attrs();
"""


def r14_source():
    return _R14_CUDA.replace("@@WGMMA@@", "".join(_r14_wgmma_fn(n) for n in (16, 32, 64, 128)))


def r14_build():
    return load_inline(
        name="r14_moe", cpp_sources=_R14_CPP, cuda_sources=r14_source(), functions=["r14_gemm", "r14_route", "r14_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"],
        build_directory=tempfile.mkdtemp(prefix="r14_"), verbose=False)


@triton.jit(do_not_specialize=["P"])
def _r14_gather_kernel(x_ptr, pos_ptr, out_ptr, P, H: tl.constexpr, TOPK: tl.constexpr):
    """out[kc, p, :] = x[token] chunk kc, its 16-byte units u stored at u ^ (p & 7) (128B swizzle)."""
    i = tl.program_id(0)
    p = tl.load(pos_ptr + i)
    if p < 0:
        return
    tok = (i // TOPK).to(tl.int64)
    offs = tl.arange(0, H)
    kc = offs // 64
    u = (offs // 8) % 8
    p64 = p.to(tl.int64)
    dst = kc.to(tl.int64) * P * 64 + p64 * 64 + ((u ^ (p & 7)) * 8 + offs % 8)
    tl.store(out_ptr + dst, tl.load(x_ptr + tok * H + offs))


def _r14_rows_gemm1():
    """GEMM1 weight-row order: block nb (64 output columns), WG c, warp w, row r: r < 8 gate column
    nb*64 + c*32 + w*8 + r, r >= 8 up column of the same index (row I + ...)."""
    nb = torch.arange(D_EXPERT // 64).view(-1, 1, 1, 1)
    c = torch.arange(2).view(1, -1, 1, 1)
    w = torch.arange(4).view(1, 1, -1, 1)
    r = torch.arange(16).view(1, 1, 1, -1)
    col = nb * 64 + c * 32 + w * 8 + (r % 8)
    return (col + (r >= 8).long() * D_EXPERT).reshape(-1)


def r14_pack(q, sf, gated):
    """Lossless repack of one tensor's experts: uint8 [E, R, K/2] codes + E4M3 [E, R, K/16] scales ->
    uint8 [E, NB, KC, 5120]. Per (expert, block, chunk): words [c][w][lane][k16] (a fragment's 8 nibbles,
    positions a0 12, a1 28, a2 8, a3 24, a4 4, a5 20, a6 0, a7 16), then bf16 scales * 2^64 [c][w][g][k16][2]."""
    experts, rows, kh = q.shape
    k = 2 * kh
    nb_count, kc_count = rows // 128, k // 64
    order = _r14_rows_gemm1().to(q.device) if gated else torch.arange(rows, device=q.device)
    shift = torch.tensor([12, 28, 8, 24, 4, 20, 0, 16], dtype=torch.int64, device=q.device)
    blobs = []
    for e in range(experts):
        codes = torch.stack((q[e] & 0x0F, q[e] >> 4), dim=-1).reshape(rows, k)[order]
        # [nb, c, w, half, g, kc, ks, khalf, qq, kpair]: row = half*8 + g, k = kc*64 + ks*16 + khalf*8 + qq*2 + kpair
        cv = codes.view(nb_count, 2, 4, 2, 8, kc_count, 4, 2, 4, 2)
        # -> [nb, kc, c, w, g, qq, ks, khalf, half, kpair]; element index i = khalf*4 + half*2 + kpair
        cv = cv.permute(0, 5, 1, 2, 4, 8, 6, 7, 3, 9).reshape(nb_count, kc_count, 2, 4, 32, 4, 8)
        words = (cv.to(torch.int64) << shift).sum(-1).to(torch.int32)  # [nb, kc, c, w, lane, ks]
        wbytes = words.contiguous().view(torch.uint8).view(nb_count, kc_count, 4096)
        s = sf[e][order].float().view(nb_count, 2, 4, 2, 8, kc_count, 4)  # [nb, c, w, half, g, kc, ks]
        s = (s * 2.0**64).to(torch.bfloat16).permute(0, 5, 1, 2, 4, 6, 3).contiguous()  # [nb, kc, c, w, g, ks, half]
        sbytes = s.view(torch.uint8).view(nb_count, kc_count, 1024)
        blobs.append(torch.cat((wbytes, sbytes), dim=-1))
    return torch.stack(blobs).contiguous()


def r14_epilogue_scales(gmm):
    """GEMM1 epilogue: x 2^62 x global scale (decode gives value * scale * 2^-62); GEMM2: x 2^62 (the combine
    applies GEMM2's global scale, as v2)."""
    return (gmm["w13_gs"] * R14_OUT_SHIFT).contiguous(), torch.full_like(gmm["w2_gs"], R14_OUT_SHIFT)


def _r14_class(tokens):
    for bound, name in R14_CLASSES:
        if tokens < bound:
            return name
    return "ft"


def _r14_sizes(tokens):
    slots = tokens * TOP_K
    max_pieces = -(-slots // R14_BT) + 4 * LOCAL_EXPERTS
    rows = slots + LOCAL_EXPERTS * R14_SEG + R14_BT  # padded segments + one tile of slack for bulk reads
    return slots, max_pieces, rows


def _experts_r14(x, topk_weights, topk_ids, state, stages):
    """Same contract as _experts."""
    r14, gmm, bank = state["r14"], state["gmm"], state["bank"]
    tokens = x.shape[0]
    slots, max_pieces, rows = _r14_sizes(tokens)
    dev = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    pos = torch.empty(slots, device=dev, dtype=torch.int32)
    pieces = torch.empty((max_pieces, 4), device=dev, dtype=torch.int32)
    n_pieces = torch.empty(1, device=dev, dtype=torch.int32)
    ext = r14["ext"]
    ext.r14_route(local_ids.view(-1), pos, pieces, n_pieces, max_pieces)
    a_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    _r14_gather_kernel[(slots,)](x.contiguous(), pos, a_sw, rows, H=D_EXPERT, TOPK=TOP_K, num_warps=4)
    act_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(r14["w13"], a_sw, pieces, n_pieces, r14["c1"], act_sw, r14["err"], rows, D_EXPERT // 64,
                 max_pieces, stages, True)
    y2 = torch.empty((rows, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(r14["w2"], act_sw, pieces, n_pieces, r14["c2"], y2, r14["err"], rows, D_EXPERT // 128,
                 max_pieces, stages, False)
    out = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, pos, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe_v4(x, topk_weights, topk_ids, state, tokens):
    stages = state["r14"]["cfg"].get(_r14_class(tokens))
    if stages is not None:
        return _experts_r14(x, topk_weights, topk_ids, state, stages)
    if tokens >= 8192 and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def r14_check(state, stages, tokens, seed=2):
    """(rel err, worst row err, error flags) of _experts_r14 against SGLang's Marlin path."""
    dev = state["bank"]["expert_map"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    ids, w = _gmm_selftest_routing(tokens, dev, seed=seed)
    state["r14"]["err"].zero_()
    ref = _experts(x, w, ids, state).float()
    out = _experts_r14(x, w, ids, state, stages).float()
    torch.cuda.synchronize()
    flags = int(state["r14"]["err"].item())
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf"), flags
    err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
    row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
    return err, row, flags


def _r14_prepare(state, weights):
    """Repacked weights (lossless), epilogue scales, the CUDA build, and a GPU check of each configured
    class against Marlin; any failure (build, flags, error) leaves that class on the fallback."""
    gmm = state["gmm"]
    c1, c2 = r14_epilogue_scales(gmm)
    r14 = {"cfg": {}, "ok": False, "c1": c1, "c2": c2,
           "err": torch.zeros(1, dtype=torch.int32, device=weights["w13"].device)}
    state["r14"] = r14
    try:
        r14["w13"] = r14_pack(gmm["w13_q"], gmm["w13_sf"], True)
        r14["w2"] = r14_pack(gmm["w2_q"], gmm["w2_sf"], False)
        r14["ext"] = r14_build()
        r14["ok"] = True
    except Exception as exc:
        print(f"[v4] R14 build/pack failed, Marlin/v2 stay: {type(exc).__name__}: {str(exc)[-600:]}", flush=True)
        return
    for cls, stages in R14_CONFIGS.items():
        try:
            err, row, flags = r14_check(state, stages, R14_REP_TOKENS[cls])
            ok = flags == 0 and err < R14_MAX_ERR and row < 5 * R14_MAX_ERR
            print(f"[v4] {cls}: R14 vs Marlin err {err:.2e} row {row:.2e} flags {flags} -> {'R14' if ok else 'fallback'}",
                  flush=True)
            if ok:
                r14["cfg"][cls] = stages
        except Exception as exc:
            print(f"[v4] {cls}: R14 failed, fallback: {type(exc).__name__}: {str(exc)[-300:]}", flush=True)
    torch.cuda.synchronize()
