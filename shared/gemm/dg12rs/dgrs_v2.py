"""DG12-RS: BF16 x (m tokens) times 12-bit F12 weights (bf16_proj's lossless format), out = x @ W^T in BF16, for
m = 32 / 64 / 128 per token tile (m = 256 -> 2 tiles), on sm_90a with register-A wgmma.

Built by muse_prefill_layer for muse_bf16_proj (TRACKER shared item 3); skeleton = prefill's R14 (judged correct):
- CTA = 128 weight rows (2 math warpgroups x 64) x BM tokens, 384 threads: warpgroup 0 = producer, 1-2 = math.
- Producer, per 64-wide K chunk: thread 0 bulk-copies the stage's 12,288 weight bytes (8 row tiles x 3 slots x 32
  lanes x 16 B; planes chunk-major [K/64][N/16][3][32][4] int32, so one contiguous block) with expect_tx on the
  stage's full barrier; all 128 threads cp.async the x tile [BM tokens][64 bf16] into the 128B-swizzled layout
  (16-byte unit u of token t at t*128 + ((u ^ (t & 7)) << 4)) and arrive with cp.async.mbarrier.arrive.noinc
  (full barrier count 1 + 128). Ring of S >= 4 stages.
- Math warp w of warpgroup c owns row tile 4c + w of the CTA's 8: per chunk it reads its lane's 3 uint4
  (Q words of k16 steps 0-3; L0/L1 of steps 0-1; of steps 2-3), decodes each step with F12M's 7-op decode into
  the 4 A registers (bf16_proj's lane/register order = wgmma RS's per-warp A layout, natural k), waits
  (fence.proxy.async: the x tile came through the generic proxy), and issues 4 wgmma m64nBMk16 against the x tile
  (SW128 K-major descriptor, +32 B per k16). Fragments triple-buffered across wait_group<1> (R14's DEPTH 2).
- Split-K over K/64 chunks (grid z = S): every CTA writes fp32 partials part[s][tok][n]; a self-resetting int32
  counter per (row block, token tile) picks the last CTA, which sums s = 0..S-1 in order (deterministic), applies
  F12M's epilogue exactly (x 2^(base - 112), escapes from rec with x from global) and stores BF16. Counters are zero
  on entry and on exit (no memset node). Plain launch (no cluster attribute, PF41).
"""

import tempfile

import torch
from torch.utils.cpp_extension import load_inline

DGRS_BMS = (32, 64, 128)       # token-tile widths compiled (wgmma N)
DGRS_W_CHUNK = 8 * 3 * 32 * 16  # weight bytes per 128-row x 64-k stage


def _wgmma_rs(n):
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


_DGRS_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace dgrs {

constexpr int W_CHUNK = 12288;          // 8 row tiles x 3 slots x 32 lanes x 16 B
constexpr int SMEM_BYTES = 200 * 1024;  // ring bytes requested: stages = what fits
constexpr int SMAX = 12;                // most ring slots
constexpr long long SPIN_LIMIT = 1ll << 22;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
// Bounded parity wait: a protocol fault flags err and bails instead of hanging the run.
__device__ __forceinline__ bool mbar_wait(uint32_t bar, uint32_t parity, int* err) {
  for (long long i = 0; i < SPIN_LIMIT; ++i) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return true;
    if ((i & 1023) == 1023 && *reinterpret_cast<volatile int*>(err) != 0) return false;
  }
  atomicAdd(err, 1);
  return false;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void proxy_fence() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void keep(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void keepf(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p));
  return v;
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// K-major 128B-swizzled B tile: start address, LBO 1 (unused when swizzled), SBO 1024 B (8 rows x 128 B).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

// F12M's decode (bf16_proj, proj/f12m/f12m.py): one lane's A fragment of one k16 step, value * 2^(112 - base).
__device__ __forceinline__ uint4 f12m_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}
__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }

@@WGMMA@@

template <int N>
__device__ __forceinline__ void mma_k64(float (&acc)[N / 2], const uint32_t (&a)[16], uint64_t desc, int first);
#define DGRS_MMA_K64(NN)                                                                            \
  template <>                                                                                       \
  __device__ __forceinline__ void mma_k64<NN>(float (&acc)[NN / 2], const uint32_t (&a)[16], uint64_t desc, \
                                              int first) {                                          \
    wgmma##NN(acc, a[0], a[1], a[2], a[3], desc, first ? 0 : 1);                                    \
    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \
    wgmma##NN(acc, a[8], a[9], a[10], a[11], desc + 4, 1);                                          \
    wgmma##NN(acc, a[12], a[13], a[14], a[15], desc + 6, 1);                                        \
  }
DGRS_MMA_K64(32)
DGRS_MMA_K64(64)
DGRS_MMA_K64(128)

// This lane's 16 A registers for one chunk: step j = 4 registers, Q word j of slot 0, L words of slot 1 + (j >> 1).
__device__ __forceinline__ void decode_frag(const uint8_t* sw, int woff, uint32_t (&a)[16]) {
  const uint4 q = *reinterpret_cast<const uint4*>(sw + woff);
  const uint4 l01 = *reinterpret_cast<const uint4*>(sw + woff + 512);
  const uint4 l23 = *reinterpret_cast<const uint4*>(sw + woff + 1024);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4& l = j < 2 ? l01 : l23;
    const uint4 r = f12m_decode(word(q, j), word(l, 2 * (j & 1)), word(l, 2 * (j & 1) + 1));
    a[4 * j] = r.x;
    a[4 * j + 1] = r.y;
    a[4 * j + 2] = r.z;
    a[4 * j + 3] = r.w;
  }
}

__device__ __forceinline__ int ring_next(int s, int S) { return s + 1 == S ? 0 : s + 1; }

// grid (N / 128, token tiles, splits), 384 threads. wp: chunk-major planes [K/64][N/16][3][32][4] int32;
// rec [N][8] int32 (base | count << 8, then k | true bits << 16); x [M][K] bf16; out [M][N] bf16;
// part [splits][tiles * BM][N] fp32; cnt [(N / 128) * tiles] int32, zero on entry, zero again on exit.
template <int BM>
__global__ void __launch_bounds__(384, 1) rs_kernel(const uint8_t* __restrict__ wp, const uint4* __restrict__ rec,
                                                    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ out,
                                                    float* __restrict__ part, int* __restrict__ cnt,
                                                    int* __restrict__ err, int M, int N, int K, int splits, int smax) {
  const int rb = blockIdx.x, tt = blockIdx.y, sl = blockIdx.z;
  const int KC = K / 64;
  const int per = (KC + splits - 1) / splits;
  const int kc0 = sl * per;
  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)
  constexpr int X_BYTES = BM * 128;
  constexpr int STAGE = X_BYTES + W_CHUNK;
  int S = SMEM_BYTES / STAGE;
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  S = S > 3 ? S : 4;
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sX = smem;                  // S x BM*128 (1024-aligned: BM % 8 == 0)
  uint8_t* sW = smem + S * X_BYTES;    // S x 12288
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + S * W_CHUNK);
  __shared__ int s_last;
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1 + 128);  // thread 0's expect_tx arrive + 128 cp.async noinc arrivals
      mbar_init(empty0 + 8 * s, 2);       // both math warpgroups
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  bool ok = true;
  if (tid < 128) {
    // ---- producer warpgroup
    const uint8_t* wsrc = wp + (static_cast<size_t>(kc0) * (N / 16) + static_cast<size_t>(rb) * 8) * 1536;
    const size_t wstride = static_cast<size_t>(N / 16) * 1536;  // one chunk of all rows
    int s = 0;
    uint32_t round = 0;
    for (int i = 0; i < nk; ++i) {
      if (round > 0) ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
      const int kc = kc0 + i;
      if (tid == 0) {
        mbar_expect_tx(full0 + 8 * s, W_CHUNK);
        bulk_g2s(smem_u32(sW + s * W_CHUNK), wsrc + static_cast<size_t>(i) * wstride, W_CHUNK, full0 + 8 * s);
      }
      const uint32_t xs = smem_u32(sX + s * X_BYTES);
      for (int u = tid; u < BM * 8; u += 128) {
        const int t = u >> 3, unit = u & 7;
        const int tok = tt * BM + t;
        if (tok < M) cp16(xs + t * 128 + ((unit ^ (t & 7)) << 4), x + static_cast<size_t>(tok) * K + kc * 64 + unit * 8);
      }
      cp_arrive_noinc(full0 + 8 * s);
      if (++s == S) {
        s = 0;
        ++round;
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
  } else {
    // ---- math warpgroups: c = 0, 1; warp w owns row tile 4c + w of the CTA's 8
    const int c = (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024
    float acc[BM / 2];
#pragma unroll
    for (int i = 0; i < BM / 2; ++i) acc[i] = 0.f;
    uint32_t fa[16], fb[16], fc[16];
    const uint32_t sX0 = smem_u32(sX);
    // chunk i sits in slot i % S, parity (i / S) & 1; release slot of chunk i - 2 after wait_group<1> at step i
    auto load = [&](int i, uint32_t (&f)[16]) {
      const int s = i % S;
      ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);
      decode_frag(sW + s * W_CHUNK, woff, f);
    };
    auto step = [&](int i, const uint32_t (&cur)[16], uint32_t (&nxt)[16]) {
      proxy_fence();  // the x tile of chunk i arrived through cp.async (generic proxy); wgmma reads it (async proxy)
      wg_fence();
      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == 0);
      wg_commit();
      wg_wait<1>();  // chunk i - 1 is done: its slot and its fragment buffer are free
#pragma unroll
      for (int r = 0; r < 16; ++r) keep(nxt[r]);
      if (i >= 1 && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - 1) % S));
      if (i + 1 < nk) load(i + 1, nxt);
    };
    load(0, fa);
    int i = 0;
    for (; i + 3 <= nk; i += 3) {
      step(i, fa, fb);
      step(i + 1, fb, fc);
      step(i + 2, fc, fa);
    }
    if (i < nk) step(i, fa, fb);
    if (i + 1 < nk) step(i + 1, fb, fc);
    wg_wait<0>();
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) keepf(acc[r]);
    // partials: rows r = 64c + 16w + g (+8), tokens 8j + 2q (+1)
    const int Mp = gridDim.y * BM;
    float* pp = part + (static_cast<size_t>(sl) * Mp + static_cast<size_t>(tt) * BM) * N + static_cast<size_t>(rb) * 128;
    const int r0 = 64 * c + 16 * w + g;
#pragma unroll
    for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int t = 8 * j + 2 * q + h;
        pp[static_cast<size_t>(t) * N + r0] = acc[4 * j + h];
        pp[static_cast<size_t>(t) * N + r0 + 8] = acc[4 * j + 2 + h];
      }
    }
    __threadfence();  // release: this thread's partials before the counter
  }
  // ---- split-K finalize by every thread of the CTA (the producer warpgroup is idle by now)
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;
    if (last) {
      __threadfence();  // acquire: every slice's partials
      atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again
    }
    s_last = last;
  }
  __syncthreads();
  if (s_last) {
    // Thread -> fixed row r = tid & 127 (its record, scale and escapes loaded once), tokens tid / 128 + 3 i in groups
    // of FU with every load of a group issued before its arithmetic; F12M's epilogue: slices summed in order s = 0..,
    // x 2^(base - 112), escapes (x from global), bf16.
    constexpr int FU = 4;
    const int r = tid & 127, row = rb * 128 + r;
    const uint4 q0 = rec[2 * row], q1 = rec[2 * row + 1];
    const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;
    const float scale = f12m_scale(base), wdec = f12m_wdec(base);
    const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
    const int Mp = gridDim.y * BM;
    const uint16_t* xh = reinterpret_cast<const uint16_t*>(x);
    for (int t0 = tid >> 7; t0 < BM; t0 += 3 * FU) {
      int tok[FU];
      bool live[FU];
      float sum[FU];
#pragma unroll
      for (int u = 0; u < FU; ++u) {
        const int t = t0 + 3 * u;
        tok[u] = tt * BM + t;
        live[u] = t < BM && tok[u] < M;
        sum[u] = 0.f;
      }
      for (int s2 = 0; s2 < splits; ++s2) {
        float v[FU];
#pragma unroll
        for (int u = 0; u < FU; ++u)
          v[u] = live[u] ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row) : 0.f;
#pragma unroll
        for (int u = 0; u < FU; ++u) sum[u] += v[u];
      }
      float xe[FU][7];
#pragma unroll
      for (int e = 0; e < 7; ++e)
#pragma unroll
        for (int u = 0; u < FU; ++u)
          xe[u][e] = (live[u] && e < static_cast<int>(count))
                         ? __uint_as_float(static_cast<uint32_t>(xh[static_cast<size_t>(tok[u]) * K + (esc[e] & 0xFFFFu)]) << 16)
                         : 0.f;
#pragma unroll
      for (int u = 0; u < FU; ++u) {
        if (!live[u]) continue;
        float acc_s = sum[u] * scale;
#pragma unroll
        for (int e = 0; e < 7; ++e) {
          if (e < static_cast<int>(count)) {
            acc_s = fmaf(-xe[u][e], wdec, acc_s);
            acc_s = fmaf(xe[u][e], __uint_as_float(esc[e] & 0xFFFF0000u), acc_s);
          }
        }
        out[static_cast<size_t>(tok[u]) * N + row] = __float2bfloat16_rn(acc_s);
      }
    }
  }
  if (!ok) atomicAdd(err, 1);
}

}  // namespace dgrs

constexpr size_t DGRS_LAUNCH_SMEM = static_cast<size_t>(dgrs::SMEM_BYTES) + 16 * dgrs::SMAX + 1024;

void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax) {
  const int KC = static_cast<int>(K / 64);
  const int per = static_cast<int>((KC + splits - 1) / splits);
  const int tiles = static_cast<int>((M + bm - 1) / bm);
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0 && splits >= 1 && (splits - 1) * per < KC, "dgrs_gemm: shapes / splits");
  TORCH_CHECK(part.numel() >= splits * static_cast<int64_t>(tiles) * bm * N && cnt.numel() >= (N / 128) * tiles,
              "dgrs_gemm: buffers");
  const dim3 grid(static_cast<unsigned>(N / 128), static_cast<unsigned>(tiles), static_cast<unsigned>(splits));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const uint4* rc = reinterpret_cast<const uint4*>(rec.data_ptr());
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define DGRS_CASE(BM)                                                                                         \
  if (bm == BM) {                                                                                             \
    cudaFuncSetAttribute(dgrs::rs_kernel<BM>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DGRS_LAUNCH_SMEM)); \
    dgrs::rs_kernel<BM><<<grid, 384, DGRS_LAUNCH_SMEM, stream>>>(w, rc, xb, o, part.data_ptr<float>(),         \
        cnt.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), \
        static_cast<int>(splits), static_cast<int>(smax));                                                    \
  } else
  @@CASES@@
  { TORCH_CHECK(false, "dgrs_gemm: no instance for bm=", bm); }
#undef DGRS_CASE
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dgrs_gemm launch failed: ", cudaGetErrorString(e));
}

torch::Tensor dgrs_attrs(int64_t bm) {
  cudaFuncAttributes a;
#define DGRS_ATTR(BM) if (bm == BM) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM>); } else
  @@ATTRS@@
  { TORCH_CHECK(false, "dgrs_attrs: no instance"); }
#undef DGRS_ATTR
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}
"""

_DGRS_CPP = """
void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax);
torch::Tensor dgrs_attrs(int64_t bm);
"""


def dgrs_source(bms=None):
    bms = DGRS_BMS if bms is None else bms
    return (_DGRS_CUDA.replace("@@WGMMA@@", "".join(_wgmma_rs(n) for n in (32, 64, 128)))
            .replace("@@CASES@@", "\n  ".join(f"DGRS_CASE({b})" for b in bms))
            .replace("@@ATTRS@@", "\n  ".join(f"DGRS_ATTR({b})" for b in bms)))


def dgrs_build(bms=None):
    """One load_inline module with the token-tile widths bms (sm_90a only)."""
    return load_inline(
        name="dgrs_ext", cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(bms), functions=["dgrs_gemm", "dgrs_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="dgrs_"), verbose=False)


def dgrs_pack(p):
    """bf16_proj's f12m_device_planes(w, natural=True)["p"] int32 [N/16][K/64][3][32][4] -> chunk-major
    [K/64][N/16][3][32][4] (a stage's 128 rows = one contiguous 12,288-byte block). Lossless reordering."""
    return p.permute(1, 0, 2, 3, 4).contiguous()


def dgrs_buffers(m, n, bm, splits, device):
    """Per-layer scratch: fp32 partials [splits][tiles * bm][n] and the zeroed counters [(n / 128) * tiles] (zero on
    exit of every call), plus an int32 error flag (mbarrier timeouts)."""
    tiles = -(-m // bm)
    return {"part": torch.empty(splits * tiles * bm * n, dtype=torch.float32, device=device),
            "cnt": torch.zeros((n // 128) * tiles, dtype=torch.int32, device=device),
            "err": torch.zeros(1, dtype=torch.int32, device=device)}


def dgrs_run(ext, x, wp_cm, rec, out, bufs, bm, splits, smax=8):
    """out [m, n] bf16 = x [m, k] bf16 @ W^T, W from the chunk-major 12-bit planes and rec. bm in DGRS_BMS,
    splits: K-chunk slices (grid z), (splits - 1) * ceil(K/64 / splits) < K/64."""
    m, k = x.shape
    n = out.shape[1]
    ext.dgrs_gemm(wp_cm, rec, x, out, bufs["part"], bufs["cnt"], bufs["err"], m, n, k, bm, splits, smax)
    return out
