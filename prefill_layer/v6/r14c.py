
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
# (cp.async.bulk + mbarrier; no tensor maps). One producer warp feeds a ring sized to the tile; two consumer
# warpgroups decode (double-buffered across wgmma.wait_group) and issue wgmma. GEMM1 writes the SwiGLU
# output K-chunked and swizzled for GEMM2; GEMM2 writes per-route rows for v2's post-norm combine.
# Every mbarrier wait is bounded: a protocol fault sets an error flag instead of hanging, and prepare()
# keeps the fallback whenever the flag is set or the results disagree with the reference.
#
# v6 (R38): thread-block clusters of CL = 1, 2 or 4 CTAs on adjacent weight blocks of the same tile share
# the token (B) tile: each CTA bulk-copies 1/CL of it with .multicast::cluster into every CTA's ring slot,
# so the L2->SM traffic for activations drops by CL (RE6: the GEMMs ran at ~4 TB/s of L2->SM traffic, two
# thirds of it the same activations re-read by every weight block). A slot is refilled only after the
# consumers of all CL CTAs released it (remote mbarrier arrivals, 2*CL per phase); a cluster barrier after
# the barrier init and before exit keeps every CTA's shared memory alive while others may still signal it.
# DIAG (lab only): 1 = the producer skips the copies (compute alone), 2 = no wgmma (loads + decode alone),
# 3 = the normal kernel with clock64() per phase (R40), summed into counters after the error flag; 4 / 5 = the
# no-load / no-MMA kernels with the same timers; 6 = lane 0 of each warp waits on the full barrier (a production
# candidate: a config's optional 5th element), 7 = 6 with timers; 8 = early shared-memory reads (read chunk kc+1's
# raw words before issuing chunk kc's wgmma, decode in registers while it runs; production candidate), 9 = 8 + timers;
# 10 = ping-pong wgmma issue between the two consumer warpgroups (named barriers 2/3), 11 = 10 + timers;
# 12 = ping-pong + wait_group 0 before each decode (CUTLASS pingpong style), 13 = 12 + timers.
# ---------------------------------------------------------------------------------------------

import tempfile

from torch.utils.cpp_extension import load_inline

# A config is (depth, smax, mode, cl): DEPTH wgmma groups in flight (v6 builds 2 only: triple-buffered
# decoded A registers), at most smax ring stages (the kernel fits as many as 200 KB allows for its tile), the
# tiling mode (0: 128/64/32/16-row tiles; 1: 128-row tiles with a ceil16 last tile, one pass per 128 routes),
# and the cluster size CL (1, 2, 4) sharing each token tile by multicast; an optional 5th element selects a
# kernel variant (6: lane-0 full-barrier waits).
R14_VARIANTS = ((2, 8, 1, 1), (2, 8, 1, 1, 14), (2, 8, 1, 1, 24))  # 24 = R55 (dq_kernel, swapped operands)
R14_CLUSTERS = (1, 2, 4)
R14_LAYOUT2_DIAGS = (14, 15, 22, 23)  # kernel variants that decode the R47 bit layout (blobs w13b / w2b)
R14_NEED_LAYOUT2 = True       # pack the R47 blobs too (lab; a clean file sets this from its configs)
# class -> config, for the token classes below; empty = off.
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


def _r14_wgmma_ss_fn(n):
    """R53: wgmma with A and B from shared memory (descriptors); same accumulator layout as the RS form."""
    regs = n // 2
    d = ", ".join(f"%{i}" for i in range(regs))
    asm = (r"{\n.reg .pred p;\nsetp.ne.b32 p, %" + str(regs + 2) + r", 0;\n"
           f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, %{regs}, %{regs + 1}, p, 1, 1, 0, 0;"
           + r"\n}\n")
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    return (f"__device__ __forceinline__ void wgmma_ss{n}(float (&d)[{regs}], uint64_t da, uint64_t db, int sc) {{\n"
            f"  asm volatile(\"{asm}\"\n      : {outs}\n      : \"l\"(da), \"l\"(db), \"r\"(sc));\n}}\n")


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

__device__ __forceinline__ uint32_t rotr(uint32_t w, int n) { return __funnelshift_r(w, w, n); }

// R47 layout (r14_pack layout 1): 11 instead of 15 integer ops per k16; same output bits as dec_k16.
__device__ __forceinline__ void dec_k16_v2(uint32_t wd, uint32_t sw, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                           uint32_t& r3) {
  const uint32_t sg = __byte_perm(sw, 0u, 0x1010u);
  const uint32_t sg8 = __byte_perm(sw, 0u, 0x3232u);
  r0 = hmul2(wd & 0x81C081C0u, sg);
  r1 = hmul2(rotr(wd, 3) & 0x81C081C0u, sg8);
  r2 = hmul2((rotr(wd, 1) & 0x80008000u) | (rotr(wd, 6) & 0x01C001C0u), sg);
  r3 = hmul2((rotr(wd, 2) & 0x80008000u) | (rotr(wd, 13) & 0x01C001C0u), sg8);
}

template <bool D2>
__device__ __forceinline__ void dec_k16_any(uint32_t wd, uint32_t sw, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  if constexpr (D2) dec_k16_v2(wd, sw, r0, r1, r2, r3);
  else dec_k16(wd, sw, r0, r1, r2, r3);
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
  atomicAdd(const_cast<int*>(err), 1);  // flag at once: every other wait then bails within 1024 polls (no 15-min hang)
  return false;
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Cluster helpers (CL > 1). The multicast copy lands at the same CTA-relative offset in every CTA of ctaMask
// and signals each CTA's mbarrier at the same offset (PTX 9.4 cp.async.bulk); remote arrivals go through the
// shared::cluster address of the target CTA's barrier (mapa); the empty-slot wait acquires at cluster scope.
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release;\nbarrier.cluster.wait.acquire;" ::: "memory");
}

__device__ __forceinline__ void bulk_g2s_mc(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar,
                                            uint16_t mask) {
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], %2, [%3], %4;"
      ::"r"(dst), "l"(src), "r"(bytes), "r"(bar), "h"(mask) : "memory");
}

// Arrive on barrier `bar` (a shared::cta address) of every CTA of the cluster, CL arrivals.
template <int CL>
__device__ __forceinline__ void mbar_arrive_cluster(uint32_t bar) {
#pragma unroll
  for (int j = 0; j < CL; ++j) {
    uint32_t ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(bar), "r"(j));
    asm volatile("mbarrier.arrive.release.cluster.shared::cluster.b64 _, [%0];" ::"r"(ra) : "memory");
  }
}

__device__ __forceinline__ bool mbar_wait_cluster(uint32_t bar, uint32_t parity, const int* err) {
  for (long long i = 0; i < SPIN_LIMIT; ++i) {
    uint32_t done;
    asm volatile(
        "{\n.reg .pred p;\nmbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 p, [%1], %2;\n"
        "selp.u32 %0, 1, 0, p;\n}\n"
        : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return true;
    if ((i & 1023) == 1023 && *reinterpret_cast<const volatile int*>(err) != 0) return false;
  }
  atomicAdd(const_cast<int*>(err), 1);  // flag at once: every other wait then bails within 1024 polls (no 15-min hang)
  return false;
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

__device__ __forceinline__ void named_arrive(int id, int threads) {
  asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
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
R14_MMA_K64(48)
R14_MMA_K64(64)
R14_MMA_K64(80)
R14_MMA_K64(96)
R14_MMA_K64(112)
R14_MMA_K64(128)

constexpr int SMEM_BYTES = 200 * 1024;  // ring bytes each CTA requests: stages = what fits for its tile
constexpr int SMAX = 24;                 // most ring slots (barriers allocated for this many)

// DIAG modes (lab): 1 no loads, 2 no wgmma, 3 timers, 4 = 1 + timers, 5 = 2 + timers, 6 lane-0 full waits (a
// production candidate), 7 = 6 + timers, 8 early shared-memory reads (production candidate), 9 = 8 + timers,
// 10 ping-pong wgmma issue between the two consumer warpgroups (production candidate), 11 = 10 + timers,
// 12 ping-pong + each warpgroup drains its own MMAs (wait_group 0) before decoding (production candidate),
// 13 = 12 + timers.
__host__ __device__ constexpr bool prof_on(int d) {
  return d == 3 || d == 4 || d == 5 || d == 7 || d == 9 || d == 11 || d == 13 || d == 15 || d == 17 || d == 19 ||
         d == 23;
}
__host__ __device__ constexpr bool pingpong(int d) { return d == 10 || d == 11 || d == 12 || d == 13; }
__host__ __device__ constexpr bool full_drain(int d) { return d == 12 || d == 13; }
__host__ __device__ constexpr bool new_dec(int d) { return d == 14 || d == 15 || d == 22 || d == 23; }
__host__ __device__ constexpr bool wide(int d) { return d == 22 || d == 23; }  // R54: 4 consumer WGs, 256 rows
__host__ __device__ constexpr bool pair_mode(int d) { return d == 16 || d == 17; }
__host__ __device__ constexpr bool no_dec(int d) { return d == 18 || d == 19; }  // lab: MMA + pipeline, no lds/decode
__host__ __device__ constexpr bool early_lds(int d) { return d == 8 || d == 9; }
__host__ __device__ constexpr bool no_load(int d) { return d == 1 || d == 4; }
__host__ __device__ constexpr bool no_mma(int d) { return d == 2 || d == 5; }
__host__ __device__ constexpr bool lane0_wait(int d) { return d == 6 || d == 7; }

// Full-slot wait. L0: only lane 0 of each warp polls the barrier (RE9: 256 threads polling one barrier cost
// ~250 cycles per chunk even when its phase had completed); __syncwarp then orders the warp's later shared-memory
// reads after lane 0's acquire, and the result is broadcast.
template <bool L0>
__device__ __forceinline__ bool wait_full(uint32_t bar, uint32_t par, const int* err) {
  if (!L0) return mbar_wait(bar, par, err);
  int ok = 1;
  if ((threadIdx.x & 31) == 0) ok = mbar_wait(bar, par, err) ? 1 : 0;
  ok = __shfl_sync(0xffffffffu, ok, 0);
  __syncwarp();
  return ok != 0;
}

template <bool L0 = false, bool D2 = false>
__device__ __forceinline__ bool load_frag(const uint8_t* sW, uint32_t full0, int s, uint32_t par, int woff, int soff,
                                          uint32_t (&a)[16], const int* err) {
  const bool ok = wait_full<L0>(full0 + 8 * s, par, err);
  const uint4 wv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + woff);
  const uint4 sv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + soff);
  dec_k16_any<D2>(wv.x, sv.x, a[0], a[1], a[2], a[3]);
  dec_k16_any<D2>(wv.y, sv.y, a[4], a[5], a[6], a[7]);
  dec_k16_any<D2>(wv.z, sv.z, a[8], a[9], a[10], a[11]);
  dec_k16_any<D2>(wv.w, sv.w, a[12], a[13], a[14], a[15]);
  return ok;
}

// DIAG 3 (lab only, R40): per-phase cycle sums of one thread per consumer warpgroup, added at the end of the CTA
// to 64-bit counters at err + 2 (the lab passes a larger err tensor): see prof_add for the slots.
struct Prof {
  long long issue, wmma, wfull, dec;
};

template <bool D2 = false>
__device__ __forceinline__ void decode_frag(const uint8_t* sW, int s, int woff, int soff, uint32_t (&a)[16]) {
  const uint4 wv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + woff);
  const uint4 sv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + soff);
  dec_k16_any<D2>(wv.x, sv.x, a[0], a[1], a[2], a[3]);
  dec_k16_any<D2>(wv.y, sv.y, a[4], a[5], a[6], a[7]);
  dec_k16_any<D2>(wv.z, sv.z, a[8], a[9], a[10], a[11]);
  dec_k16_any<D2>(wv.w, sv.w, a[12], a[13], a[14], a[15]);
}

__device__ __forceinline__ void prof_add(int* err, int slot, long long v) {
  atomicAdd(reinterpret_cast<unsigned long long*>(err + 2) + slot, static_cast<unsigned long long>(v));
}

// Ring positions advanced by one per chunk (no division in the loop): si = stage of chunk kc (issue),
// sl/pl = stage and parity of chunk kc + 1 (load), sr = stage of chunk kc - DEPTH (release).
struct Ring {
  int si, sl, sr;
  uint32_t pl;
};

__device__ __forceinline__ int ring_next(int s, int S) { return s + 1 == S ? 0 : s + 1; }

// One K chunk: issue its 4 wgmma (A = cur), keep DEPTH groups in flight, then the chunk DEPTH back is done:
// release its stage and decode chunk kc + 1 into its (now free) registers nxt.
template <int DEPTH, int N, int CL, int DIAG>
__device__ __forceinline__ void mainloop_step(float (&acc)[N / 2], const uint32_t (&cur)[16], uint32_t (&nxt)[16],
                                              uint32_t sB0, const uint8_t* sW, uint32_t full0, uint32_t empty0,
                                              int kc, int S, Ring& rg, int woff, int soff, int tid, bool& ok,
                                              const int* err, Prof& pf) {
  if constexpr (early_lds(DIAG)) {
    // RE10: shared-memory instructions issued behind in-flight wgmma waited ~250-300 cycles each chunk. So: wait
    // for chunk kc+1 and read its raw words first, then issue chunk kc's wgmma, decode in registers (ALU only)
    // while it runs, then wait_group<DEPTH - 1> and release chunk kc - (DEPTH - 1). Fragment buffers rotate by 3:
    // nxt is the buffer of chunk kc - 2, whose group completed at the previous step's wait.
    static_assert(DEPTH == 2, "early order needs triple-buffered fragments");
    const long long t0 = prof_on(DIAG) ? clock64() : 0;
    const bool more = kc + 1 < KC;
    uint4 wv = make_uint4(0u, 0u, 0u, 0u), sv = make_uint4(0u, 0u, 0u, 0u);
    if (more) {
      ok &= mbar_wait(full0 + 8 * rg.sl, rg.pl, err);
      wv = *reinterpret_cast<const uint4*>(sW + rg.sl * WS_BYTES + woff);
      sv = *reinterpret_cast<const uint4*>(sW + rg.sl * WS_BYTES + soff);
    }
    const long long t1 = prof_on(DIAG) ? clock64() : 0;
    wg_fence();
    mma_k64<N>(acc, cur, make_desc(sB0 + rg.si * (N * 128)), kc == 0);
    wg_commit();
    const long long t2 = prof_on(DIAG) ? clock64() : 0;
    if (more) {
      dec_k16(wv.x, sv.x, nxt[0], nxt[1], nxt[2], nxt[3]);
      dec_k16(wv.y, sv.y, nxt[4], nxt[5], nxt[6], nxt[7]);
      dec_k16(wv.z, sv.z, nxt[8], nxt[9], nxt[10], nxt[11]);
      dec_k16(wv.w, sv.w, nxt[12], nxt[13], nxt[14], nxt[15]);
      rg.sl = ring_next(rg.sl, S);
      if (rg.sl == 0) rg.pl ^= 1u;
    }
#pragma unroll
    for (int i = 0; i < 16; ++i) keep(nxt[i]);
    const long long t3 = prof_on(DIAG) ? clock64() : 0;
    wg_wait<DEPTH - 1>();
    const long long t4 = prof_on(DIAG) ? clock64() : 0;
    if (kc >= DEPTH - 1) {
      if ((tid & 127) == 0) {
        if (CL == 1) mbar_arrive(empty0 + 8 * rg.sr);
        else mbar_arrive_cluster<CL>(empty0 + 8 * rg.sr);
      }
      rg.sr = ring_next(rg.sr, S);
    }
    rg.si = ring_next(rg.si, S);
    if (prof_on(DIAG)) {  // slots: issue = wgmma issue, wmma = wait_group, wfull = wait + raw reads, dec = decode
      pf.wfull += t1 - t0;
      pf.issue += t2 - t1;
      pf.dec += t3 - t2;
      pf.wmma += t4 - t3;
    }
    return;
  }
  const long long t0 = prof_on(DIAG) ? clock64() : 0;
  // Ping-pong (RE11: decode time grew with the in-flight MMA size, so the two warpgroups' lockstep MMA issue and
  // decode serialize): warpgroup 0 issues chunk kc after warpgroup 1 issued chunk kc - 1 (barrier 3), warpgroup 1
  // issues chunk kc after warpgroup 0 issued it (barrier 2); each decodes while the other's MMA runs.
  const int wgi = (tid >> 7) - 1;
  if (pingpong(DIAG) && (wgi == 1 || kc > 0)) named_sync(wgi == 0 ? 3 : 2, 256);
  wg_fence();
  if (!no_mma(DIAG)) mma_k64<N>(acc, cur, make_desc(sB0 + rg.si * (N * 128)), kc == 0);
  wg_commit();
  if (pingpong(DIAG) && !(wgi == 1 && kc == KC - 1)) named_arrive(wgi == 0 ? 2 : 3, 256);
  const long long t1 = prof_on(DIAG) ? clock64() : 0;
  // full_drain (RE12): a warp's other work stretched while its own wgmma groups were in flight; drain them and
  // decode while the other warpgroup's MMA (ping-pong) keeps the tensor cores busy. Releases stay DEPTH behind.
  if constexpr (full_drain(DIAG)) wg_wait<0>();
  else wg_wait<DEPTH>();
  const long long t2 = prof_on(DIAG) ? clock64() : 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) keep(nxt[i]);
  if (kc >= DEPTH) {
    if ((tid & 127) == 0) {
      if (CL == 1) mbar_arrive(empty0 + 8 * rg.sr);
      else mbar_arrive_cluster<CL>(empty0 + 8 * rg.sr);
    }
    rg.sr = ring_next(rg.sr, S);
  }
  if (prof_on(DIAG)) {
    pf.issue += t1 - t0;
    pf.wmma += t2 - t1;
  }
  if (kc + 1 < KC && prof_on(DIAG)) {
    ok &= wait_full<lane0_wait(DIAG)>(full0 + 8 * rg.sl, rg.pl, err);
    const long long t3 = clock64();
    if (!no_dec(DIAG)) decode_frag<new_dec(DIAG)>(sW, rg.sl, woff, soff, nxt);
#pragma unroll
    for (int i = 0; i < 16; ++i) keep(nxt[i]);  // the decode is done before the next clock read
    pf.wfull += t3 - t2;
    pf.dec += clock64() - t3;
    rg.sl = ring_next(rg.sl, S);
    if (rg.sl == 0) rg.pl ^= 1u;
  } else if (kc + 1 < KC) {
    if (no_dec(DIAG)) ok &= mbar_wait(full0 + 8 * rg.sl, rg.pl, err);  // lab: fragments keep their old values
    else ok &= load_frag<lane0_wait(DIAG), new_dec(DIAG)>(sW, full0, rg.sl, rg.pl, woff, soff, nxt, err);
    if (no_mma(DIAG)) {
#pragma unroll
      for (int i = 0; i < 16; ++i) keep(nxt[i]);  // keep the decode alive without its wgmma
    }
    rg.sl = ring_next(rg.sl, S);
    if (rg.sl == 0) rg.pl ^= 1u;
  }
  rg.si = ring_next(rg.si, S);
}

// Pair stages (DIAG 16/17, RF22): each ring slot holds two 64-wide chunks (W: 2 x 5120 B, B: 2 x N*128 B), so the
// full-barrier wait and the release (shared-memory round trips of ~300 cycles) happen once per 2 chunks; the raw
// weight words of chunk kc+2 are read at the end of step kc and decoded at step kc+1, so their latency hides behind
// the MMA. Step kc: issue chunk kc; wait_group<2> (chunk kc-2 done: its fragment buffer is nxt); at odd kc >= 3
// release pair (kc-3)/2 (its second chunk kc-2 is done); decode chunk kc+1 from registers; if chunk kc+2 starts a
// pair, wait for that pair; read chunk kc+2's raw words. Needs S >= 3 pair slots (sim_cluster_ring.py "pairs").
struct PairRing {
  int s_issue, s_load, s_next, sr;
  uint32_t p_next;
};

template <int N, int CL, int DIAG>
__device__ __forceinline__ void pair_step(float (&acc)[N / 2], const uint32_t (&cur)[16], uint32_t (&nxt)[16],
                                          uint4& wv, uint4& sv, uint32_t sB0, const uint8_t* sW, uint32_t full0,
                                          uint32_t empty0, int kc, int S, PairRing& pr, int woff, int soff, int tid,
                                          bool& ok, const int* err, Prof& pf) {
  constexpr int NB_BYTES = N * 128;
  const long long t0 = prof_on(DIAG) ? clock64() : 0;
  wg_fence();
  mma_k64<N>(acc, cur, make_desc(sB0 + pr.s_issue * (2 * NB_BYTES) + (kc & 1) * NB_BYTES), kc == 0);
  wg_commit();
  const long long t1 = prof_on(DIAG) ? clock64() : 0;
  wg_wait<2>();
  const long long t2 = prof_on(DIAG) ? clock64() : 0;
  if ((kc & 1) && kc >= 3) {
    if ((tid & 127) == 0) {
      if (CL == 1) mbar_arrive(empty0 + 8 * pr.sr);
      else mbar_arrive_cluster<CL>(empty0 + 8 * pr.sr);
    }
    pr.sr = ring_next(pr.sr, S);
  }
  if (kc + 1 < KC) {
    dec_k16(wv.x, sv.x, nxt[0], nxt[1], nxt[2], nxt[3]);
    dec_k16(wv.y, sv.y, nxt[4], nxt[5], nxt[6], nxt[7]);
    dec_k16(wv.z, sv.z, nxt[8], nxt[9], nxt[10], nxt[11]);
    dec_k16(wv.w, sv.w, nxt[12], nxt[13], nxt[14], nxt[15]);
  }
#pragma unroll
  for (int i = 0; i < 16; ++i) keep(nxt[i]);
  const long long t3 = prof_on(DIAG) ? clock64() : 0;
  if (kc + 2 < KC) {
    if (((kc + 2) & 1) == 0) {
      ok &= mbar_wait(full0 + 8 * pr.s_next, pr.p_next, err);
      pr.s_load = pr.s_next;
      pr.s_next = ring_next(pr.s_next, S);
      if (pr.s_next == 0) pr.p_next ^= 1u;
    }
    const uint8_t* base = sW + pr.s_load * (2 * WS_BYTES) + ((kc + 2) & 1) * WS_BYTES;
    wv = *reinterpret_cast<const uint4*>(base + woff);
    sv = *reinterpret_cast<const uint4*>(base + soff);
  }
  if (kc & 1) pr.s_issue = ring_next(pr.s_issue, S);
  if (prof_on(DIAG)) {  // slots: issue, wmma = wait_group, dec = register decode, wfull = pair wait + raw reads
    pf.issue += t1 - t0;
    pf.wmma += t2 - t1;
    pf.dec += t3 - t2;
    pf.wfull += clock64() - t3;
  }
}

// Epilogue shared by the R14 and R53 kernels: the 256 MMA threads (local index lt) stage the accumulators in shared
// memory and copy them out. GEMM1 (GATED): SwiGLU, written K-chunked and swizzled as GEMM2's B; GEMM2: per-route rows.
template <bool GATED, int N>
__device__ __forceinline__ void store_out(const float (&acc)[N / 2], uint8_t* stage, int c, int w, int g, int q, int lt,
                                          int valid, int row0, int nb, int P, float cs,
                                          __nv_bfloat16* __restrict__ out, int bar = 1) {
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
    named_sync(bar, 256);
    const int rows = (valid + 7) & ~7;
    uint4* dst = reinterpret_cast<uint4*>(out + (static_cast<size_t>(nb) * P + row0) * 64);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = lt; i < rows * 8; i += 256) dst[i] = src[i];
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
    named_sync(bar, 256);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = lt; i < valid * 16; i += 256) {
      const int t = i >> 4, u = i & 15;
      reinterpret_cast<uint4*>(out + static_cast<size_t>(row0 + t) * K + nb * 128)[u] = src[i];
    }
  }
}

template <int DEPTH, bool GATED, int N, int CL, int DIAG>
__device__ __forceinline__ void consume(uint8_t* sB, const uint8_t* sW, uint32_t full0, uint32_t empty0, int tid,
                                        int S, int e, int row0, int valid, int nb, const float* __restrict__ cscale,
                                        __nv_bfloat16* __restrict__ out, int* __restrict__ err, int P) {
  // R54 (wide): consumer warpgroups 0-3 are warps 0-15; WG C works on block nb = 2 * blockIdx.x + (C >> 1), half C & 1.
  const int c = wide(DIAG) ? ((tid >> 7) & 1) : (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2,
            q = lane & 3;
  const int woff = ((c * 4 + w) * 32 + lane) * 16;
  const int soff = W_BYTES + ((c * 4 + w) * 8 + g) * 16;
  const uint32_t sB0 = smem_u32(sB);
  const long long tc0 = prof_on(DIAG) ? clock64() : 0;
  Prof pf{0, 0, 0, 0};
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t fa[16], fb[16], fc[16];
  bool ok = true;
  long long tc1 = 0;
  if constexpr (pair_mode(DIAG)) {
    static_assert(DEPTH == 2, "pair stages need triple-buffered fragments");
    ok &= mbar_wait(full0, 0u, err);  // pair 0 = chunks 0 and 1 in slot 0
    decode_frag(sW, 0, woff, soff, fa);  // chunk 0 (decode_frag's slot stride is 5120 B: slot 0 = chunk 0)
    uint4 wv = *reinterpret_cast<const uint4*>(sW + WS_BYTES + woff);  // chunk 1's raw words
    uint4 sv = *reinterpret_cast<const uint4*>(sW + WS_BYTES + soff);
    if (prof_on(DIAG)) {
#pragma unroll
      for (int i = 0; i < 16; ++i) keep(fa[i]);
      tc1 = clock64();
    }
    PairRing pr{0, 0, 1, 0, 0u};  // S >= 3: pair 1 is slot 1, parity 0
    int kc = 0;
    for (; kc + 3 <= KC; kc += 3) {
      pair_step<N, CL, DIAG>(acc, fa, fb, wv, sv, sB0, sW, full0, empty0, kc, S, pr, woff, soff, tid, ok, err, pf);
      pair_step<N, CL, DIAG>(acc, fb, fc, wv, sv, sB0, sW, full0, empty0, kc + 1, S, pr, woff, soff, tid, ok, err, pf);
      pair_step<N, CL, DIAG>(acc, fc, fa, wv, sv, sB0, sW, full0, empty0, kc + 2, S, pr, woff, soff, tid, ok, err, pf);
    }
    if (kc < KC) pair_step<N, CL, DIAG>(acc, fa, fb, wv, sv, sB0, sW, full0, empty0, kc, S, pr, woff, soff, tid, ok, err, pf);
  } else {
  ok = load_frag<lane0_wait(DIAG), new_dec(DIAG)>(sW, full0, 0, 0, woff, soff, fa, err);
  if (prof_on(DIAG)) {
#pragma unroll
    for (int i = 0; i < 16; ++i) keep(fa[i]);
    tc1 = clock64();
  }
  if constexpr (no_dec(DIAG)) {
#pragma unroll
    for (int i = 0; i < 16; ++i) fb[i] = fc[i] = fa[i];
  }
  Ring rg{0, 1, 0, 0u};  // S >= 3, so chunk 1 is stage 1, parity 0
  if constexpr (DEPTH == 1) {
    for (int kc = 0; kc < KC; kc += 2) {
      mainloop_step<1, N, CL, DIAG>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err, pf);
      mainloop_step<1, N, CL, DIAG>(acc, fb, fa, sB0, sW, full0, empty0, kc + 1, S, rg, woff, soff, tid, ok, err, pf);
    }
  } else {
    int kc = 0;
    for (; kc + 3 <= KC; kc += 3) {
      mainloop_step<2, N, CL, DIAG>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err, pf);
      mainloop_step<2, N, CL, DIAG>(acc, fb, fc, sB0, sW, full0, empty0, kc + 1, S, rg, woff, soff, tid, ok, err, pf);
      mainloop_step<2, N, CL, DIAG>(acc, fc, fa, sB0, sW, full0, empty0, kc + 2, S, rg, woff, soff, tid, ok, err, pf);
    }
    if (kc < KC) mainloop_step<2, N, CL, DIAG>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err, pf);
  }
  }  // not pair_mode
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) keepf(acc[i]);
  const long long tc2 = prof_on(DIAG) ? clock64() : 0;
  if (!ok) atomicAdd(err, 1);
  // every consumer warpgroup is done with the B tiles: reuse them for staging (R54: one N*256-byte area per block,
  // inside the S >= 4 B slots, each pair of warpgroups then syncs on its own barrier 2 + pair)
  named_sync(1, wide(DIAG) ? 512 : 256);
  const float cs = cscale[e];
  uint8_t* stage = wide(DIAG) ? sB + (tid >> 8) * (N * 256) : sB;
  store_out<GATED, N>(acc, stage, c, w, g, q, wide(DIAG) ? (tid & 255) : tid - 128, valid, row0, nb, P, cs, out,
                      wide(DIAG) ? 2 + (tid >> 8) : 1);
  if (prof_on(DIAG) && (tid & 127) == 0) {
    // slots: 0 wgmma issue, 1 wgmma wait, 2 full-barrier wait, 3 lds + decode, 4 first chunk (entry -> first
    // fragment), 5 main loop, 6 epilogue, 7 warpgroups, 8 chunks, 12 sum of N
    const long long tc3 = clock64();
    prof_add(err, 0, pf.issue);
    prof_add(err, 1, pf.wmma);
    prof_add(err, 2, pf.wfull);
    prof_add(err, 3, pf.dec);
    prof_add(err, 4, tc1 - tc0);
    prof_add(err, 5, tc2 - tc1);
    prof_add(err, 6, tc3 - tc2);
    prof_add(err, 7, 1);
    prof_add(err, 8, KC);
    prof_add(err, 12, N);
  }
}

// grid (NB, max pieces), 384 threads: warpgroup 0 = producer (one thread), 1-2 = consumers.
// piece = (expert, first row, valid rows, tile rows N in 16..128). B source: [KC][P][64] swizzled bf16.
// The ring holds S = min(smax, SMAX, SMEM_BYTES / (5120 + N*128)) stages (at least DEPTH + 2), sized per CTA.
// CL > 1: launched in clusters of CL along x (weight blocks nb of the same piece, so every CTA of a cluster has
// the same piece, S and shared-memory layout); CTA rank r multicasts rows [r*N/CL, (r+1)*N/CL) of each B tile.
template <int DEPTH, bool GATED, int CL, int DIAG>
__global__ void __launch_bounds__(384, 1) gemm_kernel(const uint8_t* __restrict__ wblob,
                                                      const __nv_bfloat16* __restrict__ bsrc,
                                                      const int4* __restrict__ pieces, const int* __restrict__ n_pieces,
                                                      const float* __restrict__ cscale, __nv_bfloat16* __restrict__ out,
                                                      int* __restrict__ err, int P, int NB, int smax) {
  const int nb = blockIdx.x, pc = blockIdx.y;
  if (pc >= *n_pieces) return;  // the same for every CTA of a cluster (same blockIdx.y): no cluster op yet
  const int4 piece = pieces[pc];
  const int e = piece.x, row0 = piece.y, valid = piece.z, nsel = piece.w;
  const int bbytes = nsel * 128;
  constexpr int PAIR = pair_mode(DIAG) ? 2 : 1;  // chunks per ring slot
  int S = SMEM_BYTES / (PAIR * (WS_BYTES + bbytes));
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  if (PAIR == 1) {
    S = S > DEPTH + 1 ? S : DEPTH + 2;  // the producer refills chunk kc - S after the consumers release it at
                                       // step kc - S + DEPTH, before they need chunk kc - S + DEPTH + 1: S >= DEPTH + 2
  } else {
    S = S > 3 ? S : 3;  // pair stages: >= 3 slots (sim_cluster_ring.py "pairs"); 3 x 2 x (5120 + 16384) B fits
  }
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sB = smem;
  uint8_t* sW = smem + PAIR * S * bbytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + PAIR * S * WS_BYTES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 2 * CL);  // both consumer warpgroups of every CTA in the cluster
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (CL > 1) cluster_sync();  // every CTA's barriers exist before any multicast or remote arrival reaches them
  if (tid < 128) {
    if (tid == 0) {
      const uint8_t* wsrc = wblob + static_cast<size_t>(e * NB + nb) * KC * WS_BYTES;
      const uint32_t slice = static_cast<uint32_t>(bbytes / CL);  // N/CL rows: a multiple of 512 B
      const uint32_t boff = CL > 1 ? cluster_rank() * slice : 0u;
      bool ok = true;
      int s = 0;
      uint32_t round = 0;  // kc / S, advanced without division
      long long tw = 0;
      const long long tp0 = prof_on(DIAG) ? clock64() : 0;
      for (int kc = 0; kc < KC; kc += PAIR) {  // PAIR 2: one slot per chunk pair
        if (round > 0) {
          const long long ta = prof_on(DIAG) ? clock64() : 0;
          if (CL == 1) ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
          else ok &= mbar_wait_cluster(empty0 + 8 * s, (round - 1) & 1u, err);
          if (prof_on(DIAG)) tw += clock64() - ta;
        }
        if (no_load(DIAG)) {
          mbar_arrive(full0 + 8 * s);  // no data: the consumers run on whatever the slot holds
        } else if (PAIR == 2) {
          // chunks kc, kc+1: their weight blobs are contiguous (10240 B); their B tiles are P rows apart
          mbar_expect_tx(full0 + 8 * s, 2 * (WS_BYTES + bbytes));
          bulk_g2s(smem_u32(sW + s * (2 * WS_BYTES)), wsrc + static_cast<size_t>(kc) * WS_BYTES, 2 * WS_BYTES,
                   full0 + 8 * s);
          bulk_g2s(smem_u32(sB + s * (2 * bbytes)), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes,
                   full0 + 8 * s);
          bulk_g2s(smem_u32(sB + s * (2 * bbytes) + bbytes), bsrc + (static_cast<size_t>(kc + 1) * P + row0) * 64,
                   bbytes, full0 + 8 * s);
        } else {
          mbar_expect_tx(full0 + 8 * s, WS_BYTES + bbytes);
          bulk_g2s(smem_u32(sW + s * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
          const __nv_bfloat16* bs = bsrc + (static_cast<size_t>(kc) * P + row0) * 64;
          if (CL == 1) {
            bulk_g2s(smem_u32(sB + s * bbytes), bs, bbytes, full0 + 8 * s);
          } else {
            bulk_g2s_mc(smem_u32(sB + s * bbytes) + boff, reinterpret_cast<const uint8_t*>(bs) + boff, slice,
                        full0 + 8 * s, static_cast<uint16_t>((1u << CL) - 1u));
          }
        }
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
      if (prof_on(DIAG)) {  // slots 9 producer empty-slot wait, 10 producer loop, 11 producers
        prof_add(err, 9, tw);
        prof_add(err, 10, clock64() - tp0);
        prof_add(err, 11, 1);
      }
      if (!ok) atomicAdd(err, 1);
    }
  } else {
#define R14_CASE(NN)                                                                                       \
  case NN:                                                                                                 \
    consume<DEPTH, GATED, NN, CL, DIAG>(sB, sW, full0, empty0, tid, S, e, row0, valid, nb, cscale, out, err, P); \
    break;
    switch (nsel) {
#ifndef R14_PROBE
      R14_CASE(16) R14_CASE(32) R14_CASE(48) R14_CASE(64) R14_CASE(80) R14_CASE(96) R14_CASE(112)
#endif
      default: consume<DEPTH, GATED, 128, CL, DIAG>(sB, sW, full0, empty0, tid, S, e, row0, valid, nb, cscale, out,
                                                    err, P); break;
    }
#undef R14_CASE
  }
  // CL > 1: no CTA leaves while another may still arrive on its barriers (multicast data all landed before
  // its consumers finished; remote releases do not).
  if (CL > 1) cluster_sync();
}

// ------------------------------------------------------------------------------------------------- R54
// Four consumer warpgroups (RF21 lever c): RE8-RE16 put R14's ~790 cycles per chunk in per-warp latency (barrier
// round trip ~280, lds + decode ~250-300, wgmma issue ~120) with the tensor cores busy a third of the time. Here a CTA
// takes two adjacent 128-row weight blocks (2 * blockIdx.x and + 1) of one token tile: twice the consumer warps per
// scheduler to hide that latency, and each B tile is copied once for 256 weight rows instead of 128. Warps 0-15 are
// the consumers (warpgroup C: block 2 * blockIdx.x + (C >> 1), rows (C & 1) * 64..+63, R14's consume() as is), warp
// 16 the producer (one thread: per chunk both blocks' 5120-byte blobs into their own rings and the B tile, one full
// barrier with expect_tx 2 * 5120 + N * 128). Empty barriers count the 4 consumer warpgroups. 544 threads: no
// setmaxnreg (120 registers each), so only N <= 64 is built (tile mode 2: 64-row tiles); a larger tile flags an error
// and the whole CTA leaves before any barrier exists.
constexpr int WIDE_THREADS = 544;

template <bool GATED, int DIAG>
__global__ void __launch_bounds__(WIDE_THREADS, 1) wide_kernel(const uint8_t* __restrict__ wblob,
                                                             const __nv_bfloat16* __restrict__ bsrc,
                                                             const int4* __restrict__ pieces,
                                                             const int* __restrict__ n_pieces,
                                                             const float* __restrict__ cscale,
                                                             __nv_bfloat16* __restrict__ out, int* __restrict__ err,
                                                             int P, int NB, int smax) {
  const int nb2 = blockIdx.x, pc = blockIdx.y;
  if (pc >= *n_pieces) return;
  const int4 piece = pieces[pc];
  const int e = piece.x, row0 = piece.y, valid = piece.z, nsel = piece.w;
  if (nsel > 64) {
    if (threadIdx.x == 0) atomicAdd(err, 1);
    return;
  }
  const int bbytes = nsel * 128;
  int S = SMEM_BYTES / (2 * WS_BYTES + bbytes);
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  S = S > 3 ? S : 4;  // >= DEPTH + 2 (and 4 B slots hold both blocks' epilogue staging, 2 x N * 256 bytes)
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sB = smem;
  uint8_t* sW0 = sB + S * bbytes;
  uint8_t* sW1 = sW0 + S * WS_BYTES;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW1 + S * WS_BYTES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);  // the 4 consumer warpgroups
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid >= 512) {
    if (tid == 512) {
      const uint8_t* wsrc0 = wblob + static_cast<size_t>(e * NB + 2 * nb2) * KC * WS_BYTES;
      const uint8_t* wsrc1 = wsrc0 + static_cast<size_t>(KC) * WS_BYTES;  // block 2 * nb2 + 1 follows in the blob
      bool ok = true;
      int s = 0;
      uint32_t round = 0;
      long long tw = 0;
      const long long tp0 = prof_on(DIAG) ? clock64() : 0;
      for (int kc = 0; kc < KC; ++kc) {
        if (round > 0) {
          const long long ta = prof_on(DIAG) ? clock64() : 0;
          ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
          if (prof_on(DIAG)) tw += clock64() - ta;
        }
        mbar_expect_tx(full0 + 8 * s, 2 * WS_BYTES + bbytes);
        bulk_g2s(smem_u32(sW0 + s * WS_BYTES), wsrc0 + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
        bulk_g2s(smem_u32(sW1 + s * WS_BYTES), wsrc1 + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
        bulk_g2s(smem_u32(sB + s * bbytes), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * s);
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
      if (prof_on(DIAG)) {
        prof_add(err, 9, tw);
        prof_add(err, 10, clock64() - tp0);
        prof_add(err, 11, 1);
      }
      if (!ok) atomicAdd(err, 1);
    }
  } else {
    const int h = tid >> 8;  // which of the two blocks
    const uint8_t* sWh = h ? sW1 : sW0;
#define R14_WCASE(NN)                                                                                          \
  case NN:                                                                                                     \
    consume<2, GATED, NN, 1, DIAG>(sB, sWh, full0, empty0, tid, S, e, row0, valid, 2 * nb2 + h, cscale, out, err, P); \
    break;
    switch (nsel) {
      R14_WCASE(16) R14_WCASE(32) R14_WCASE(48)
      default: consume<2, GATED, 64, 1, DIAG>(sB, sWh, full0, empty0, tid, S, e, row0, valid, 2 * nb2 + h, cscale,
                                              out, err, P); break;
    }
#undef R14_WCASE
  }
}

// ------------------------------------------------------------------------------------------------- R53
// Dequant warp specialization (RF24): the MMA warps neither decode nor wait for TMA data. 640 threads:
//   warpgroup 0: producer (thread 0: TMA of each chunk's weight blob + B tile into a ring of S slots);
//   warpgroups 1-2: decoders; warpgroup 1 + (kc & 1) decodes chunk kc: waits for its TMA slot and for a free A slot,
//     decodes the 128 x 64 NVFP4 weights into a bf16 A tile (K-major, 128B swizzle, the same layout as a B tile),
//     fences the async proxy and signals A-full (one arrival per warp) and its share of the TMA slot's release;
//   warpgroups 3-4: MMA; warpgroup 3 + c owns weight rows c*64..c*64+63: waits A-full of chunk kc, issues 4 SS wgmma
//     (A rows from the A slot, B from the TMA slot), wait_group<1>, then releases chunk kc-1's A slot and TMA slot.
// Barriers: full (1 + tx), empty (4 decoder warps + 2 MMA warpgroups), A-full (4), A-empty (2). Both rings have an even
// slot count, so a decoder only ever waits on slots of its own parity, in order (sim_cluster_ring.py simulate_dq).
// setmaxnreg: producer 40, decoders 72, MMA 144 registers per thread. The pool setmaxnreg draws from is the CTA's own
// launch allocation (96 x 640 = 61440 registers under __launch_bounds__(640, 1)), not the SM's free registers: RE17
// hung in setmaxnreg.inc with 40/80/152 (needs 63488). The static_assert below keeps the split inside it.
constexpr int DQ_THREADS = 640;
constexpr int DQ_SA = 4;                          // A slots (even)
constexpr int DQ_A_BYTES = 16384;                 // one chunk's A tile: 2 x 64 rows x 128 B
constexpr int DQ_RING = SMEM_BYTES - DQ_SA * DQ_A_BYTES;
constexpr size_t DQ_LAUNCH_SMEM = static_cast<size_t>(SMEM_BYTES) + 2048;  // + barriers (448 B) + alignment (< 1 KB)
constexpr int DQ_REG_PROD = 40, DQ_REG_DEC = 72, DQ_REG_MMA = 144;
constexpr int DQ_REG_LAUNCH = (65536 / DQ_THREADS) & ~7;  // 96: what ptxas allocates per thread at launch
static_assert(DQ_REG_PROD * 128 + DQ_REG_DEC * 256 + DQ_REG_MMA * 256 <= DQ_REG_LAUNCH * DQ_THREADS,
              "setmaxnreg split must fit the CTA's launch allocation, or setmaxnreg.inc blocks forever");
static_assert(DQ_REG_PROD <= DQ_REG_LAUNCH && DQ_REG_DEC <= DQ_REG_LAUNCH && DQ_REG_MMA >= DQ_REG_LAUNCH,
              "dec must not raise, inc must not lower the per-thread count");

__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }

template <int N>
__device__ __forceinline__ void mma_k64_ss(float (&acc)[N / 2], uint64_t da, uint64_t db, int first);
#define R14_MMA_SS(NN)                                                                                            \
  template <>                                                                                                     \
  __device__ __forceinline__ void mma_k64_ss<NN>(float (&acc)[NN / 2], uint64_t da, uint64_t db, int first) {   \
    wgmma_ss##NN(acc, da, db, first ? 0 : 1);                                                                     \
    wgmma_ss##NN(acc, da + 2, db + 2, 1);                                                                         \
    wgmma_ss##NN(acc, da + 4, db + 4, 1);                                                                         \
    wgmma_ss##NN(acc, da + 6, db + 6, 1);                                                                         \
  }
R14_MMA_SS(16)
R14_MMA_SS(32)
R14_MMA_SS(48)
R14_MMA_SS(64)
R14_MMA_SS(80)
R14_MMA_SS(96)
R14_MMA_SS(112)
R14_MMA_SS(128)

// Decoder thread (w, lane) of warpgroup d: for roles c = 0, 1 it decodes the fragment words R14's consumer thread
// (c, w, lane) used to decode, and stores each bf16x2 register at its A-tile position: register j of word ks holds
// rows 16w + g + 8(j & 1) and k = 16 ks + 2q + 8(j >> 1), +1 (the A fragment layout), in WG c's 64-row tile.
// PROF (lab, diag 21 / 25): cycle sums of thread 0 of each decoder warpgroup into slots 13 (full wait), 14 (A-empty
// wait), 3 (lds + decode + stores), 15 (fence + arrive), 16 (decoder warpgroups), 17 (chunks decoded).
template <bool PROF>
__device__ __forceinline__ void dq_decode(const uint8_t* sW, uint8_t* sA, uint32_t full0, uint32_t empty0,
                                          uint32_t afull0, uint32_t aempty0, int S, int d, int tid, bool& ok,
                                          int* err) {
  const int w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
  int kc = d, s = d, a = d, ra = 0;  // S, DQ_SA even and >= 4: chunk d sits in slot d (round 0), A slot d
  uint32_t ps = 0u;
  long long t_full = 0, t_ae = 0, t_dec = 0, t_arr = 0;
  int n_dec = 0;
  for (; kc < KC; kc += 2) {
    const long long p0 = PROF ? clock64() : 0;
    ok &= mbar_wait(full0 + 8 * s, ps, err);
    const long long p1 = PROF ? clock64() : 0;
    if (kc >= DQ_SA) ok &= mbar_wait(aempty0 + 8 * a, static_cast<uint32_t>(ra - 1) & 1u, err);
    const long long p2 = PROF ? clock64() : 0;
    uint8_t* tile = sA + a * DQ_A_BYTES;
#pragma unroll
    for (int c = 0; c < 2; ++c) {
      const int woff = ((c * 4 + w) * 32 + lane) * 16;
      const int soff = W_BYTES + ((c * 4 + w) * 8 + g) * 16;
      const uint4 wv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + woff);
      const uint4 sv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + soff);
      const uint32_t wd[4] = {wv.x, wv.y, wv.z, wv.w}, sd[4] = {sv.x, sv.y, sv.z, sv.w};
      uint8_t* wgt = tile + c * 8192;
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        uint32_t r[4];
        dec_k16(wd[ks], sd[ks], r[0], r[1], r[2], r[3]);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const int row = 16 * w + g + 8 * (j & 1);
          const int unit = 2 * ks + (j >> 1);
          *reinterpret_cast<uint32_t*>(wgt + row * 128 + ((unit ^ g) << 4) + 4 * q) = r[j];
        }
      }
    }
    const long long p3 = PROF ? clock64() : 0;
    fence_proxy_async();  // this thread's A stores, before the MMA warps' wgmma (async proxy) read them
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(afull0 + 8 * a);
      mbar_arrive(empty0 + 8 * s);  // this warp is done with the slot's weight words
    }
    if (PROF) {
      t_full += p1 - p0;
      t_ae += p2 - p1;
      t_dec += p3 - p2;
      t_arr += clock64() - p3;
      ++n_dec;
    }
    s += 2;
    if (s >= S) {
      s -= S;
      ps ^= 1u;
    }
    a += 2;
    if (a >= DQ_SA) {
      a -= DQ_SA;
      ++ra;
    }
  }
  if (PROF && (tid & 127) == 0) {
    prof_add(err, 13, t_full);
    prof_add(err, 14, t_ae);
    prof_add(err, 3, t_dec);
    prof_add(err, 15, t_arr);
    prof_add(err, 16, 1);
    prof_add(err, 17, n_dec);
  }
}

// PROF (lab): MMA warpgroup thread 0 sums into slots 2 (A-full wait), 0 (issue), 1 (wait_group), 5 (loop), 6 (epilogue),
// 7 (MMA warpgroups), 8 (chunks); the producer into 9 (empty wait), 10 (loop), 11 (producers).
template <bool PROF>
__device__ __forceinline__ void dq_prof_mma(int* err, int lt, long long tw, long long ti, long long tm, long long tl,
                                            long long te) {
  if (PROF && (lt & 127) == 0) {
    prof_add(err, 2, tw);
    prof_add(err, 0, ti);
    prof_add(err, 1, tm);
    prof_add(err, 5, tl);
    prof_add(err, 6, te);
    prof_add(err, 7, 1);
    prof_add(err, 8, KC);
  }
}

template <bool GATED, int N, bool PROF>
__device__ __forceinline__ void dq_mma(uint8_t* sB, uint8_t* sA, uint32_t empty0, uint32_t afull0, uint32_t aempty0,
                                       int S, int tid, int e, int row0, int valid, int nb,
                                       const float* __restrict__ cscale, __nv_bfloat16* __restrict__ out,
                                       int* __restrict__ err, int P, bool& ok) {
  const int lt = tid - 384, c = lt >> 7, w = (lt >> 5) & 3, lane = lt & 31, g = lane >> 2, q = lane & 3;
  const uint32_t sB0 = smem_u32(sB), sA0 = smem_u32(sA) + c * 8192;
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  int a = 0, s = 0, a_prev = 0, s_prev = 0;
  uint32_t pa = 0u;
  long long tw = 0, ti = 0, tm = 0;
  const long long tl0 = PROF ? clock64() : 0;
  for (int kc = 0; kc < KC; ++kc) {
    const long long p0 = PROF ? clock64() : 0;
    ok &= mbar_wait(afull0 + 8 * a, pa, err);
    const long long p1 = PROF ? clock64() : 0;
    wg_fence();
    mma_k64_ss<N>(acc, make_desc(sA0 + a * DQ_A_BYTES), make_desc(sB0 + s * (N * 128)), kc == 0);
    wg_commit();
    const long long p2 = PROF ? clock64() : 0;
    wg_wait<1>();
    if (PROF) {
      tw += p1 - p0;
      ti += p2 - p1;
      tm += clock64() - p2;
    }
    if (kc >= 1 && (lt & 127) == 0) {
      mbar_arrive(aempty0 + 8 * a_prev);
      mbar_arrive(empty0 + 8 * s_prev);
    }
    a_prev = a;
    s_prev = s;
    if (++a == DQ_SA) {
      a = 0;
      pa ^= 1u;
    }
    s = ring_next(s, S);
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < N / 2; ++i) keepf(acc[i]);
  const long long tl1 = PROF ? clock64() : 0;
  named_sync(1, 256);  // both MMA warpgroups are done with the B tiles: reuse them for staging
  store_out<GATED, N>(acc, sB, c, w, g, q, lt, valid, row0, nb, P, cscale[e], out);
  dq_prof_mma<PROF>(err, lt, tw, ti, tm, tl1 - tl0, PROF ? clock64() - tl1 : 0);
}

// ------------------------------------------------------------------------------------------------- R55
// Swapped operands on R53's pipeline (RF28: R14's chunk costs ~8 wgmma x (60 + N/2) cycles; an isolated m64nNk16 is
// 32 cycles at N = 64, 64 at N = 128, near peak for N >= 64, arXiv 2501.12084): the tokens are wgmma's A (M = 64 per
// MMA warpgroup, K-major tile straight from the TMA slot) and the decoded 128-row weight tile is B (N = 128, the same
// K-major 128B-swizzle tile R53 writes). One chunk = 4 m64n128k16 per active warpgroup instead of 8 small ones.
// Warpgroup m takes tokens 64m..64m+63 (m = 1 only when the tile has > 64 tokens; otherwise it waits and releases like
// m = 0 without issuing). Token slots are ceil64(N) * 128 bytes so a warpgroup's 64 A rows stay inside its slot;
// rows past the tile's tokens hold stale data whose accumulator rows are never stored. D = [token][weight row].
template <bool GATED>
__device__ __forceinline__ void store_out_t(const float (&acc)[64], uint8_t* stage, bool active, int m, int w, int g,
                                            int q, int lt, int valid, int row0, int nb, int P, float cs,
                                            __nv_bfloat16* __restrict__ out) {
  if (GATED) {
    // Weight row n = 16i + 2q + h is gate (blob order: rows n & 15 < 8), n + 8 its up row; output column 8i + 2q + h.
    if (active) {
#pragma unroll
      for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int t = 64 * m + 16 * w + g + 8 * r;
            const int col = 8 * i + 2 * q + h;
            const float gt = acc[8 * i + 2 * r + h] * cs;
            const float up = acc[8 * i + 4 + 2 * r + h] * cs;
            const float v = gt / (1.f + expf(-gt)) * up;
            *reinterpret_cast<__nv_bfloat16*>(stage + t * 128 + ((((col >> 3) ^ (t & 7))) << 4) + (col & 7) * 2) =
                __float2bfloat16_rn(v);
          }
        }
      }
    }
    named_sync(1, 256);
    const int rows = (valid + 7) & ~7;
    uint4* dst = reinterpret_cast<uint4*>(out + (static_cast<size_t>(nb) * P + row0) * 64);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = lt; i < rows * 8; i += 256) dst[i] = src[i];
  } else {
    // Token t = 64m + 16w + g + 8r, output column n = 8j + 2q + h of block nb. Staged as [token][128 cols].
    if (active) {
#pragma unroll
      for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
          const int t = 64 * m + 16 * w + g + 8 * r;
          const int n = 8 * j + 2 * q;
          *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + n * 2) = __float2bfloat16_rn(acc[4 * j + 2 * r] * cs);
          *reinterpret_cast<__nv_bfloat16*>(stage + t * 256 + (n + 1) * 2) = __float2bfloat16_rn(acc[4 * j + 2 * r + 1] * cs);
        }
      }
    }
    named_sync(1, 256);
    const uint4* src = reinterpret_cast<const uint4*>(stage);
    for (int i = lt; i < valid * 16; i += 256) {
      const int t = i >> 4, u = i & 15;
      reinterpret_cast<uint4*>(out + static_cast<size_t>(row0 + t) * K + nb * 128)[u] = src[i];
    }
  }
}

template <bool GATED, bool PROF>
__device__ __forceinline__ void dq_mma_swap(uint8_t* sB, uint8_t* sA, uint32_t empty0, uint32_t afull0,
                                            uint32_t aempty0, int S, int tstride, int nsel, int tid, int e, int row0,
                                            int valid, int nb, const float* __restrict__ cscale,
                                            __nv_bfloat16* __restrict__ out, int* __restrict__ err, int P, bool& ok) {
  const int lt = tid - 384, m = lt >> 7, w = (lt >> 5) & 3, lane = lt & 31, g = lane >> 2, q = lane & 3;
  const bool active = m == 0 || nsel > 64;
  const uint32_t sB0 = smem_u32(sB) + m * 8192, sA0 = smem_u32(sA);
  float acc[64];
#pragma unroll
  for (int i = 0; i < 64; ++i) acc[i] = 0.f;
  int a = 0, s = 0, a_prev = 0, s_prev = 0;
  uint32_t pa = 0u;
  long long tw = 0, ti = 0, tm = 0;
  const long long tl0 = PROF ? clock64() : 0;
  for (int kc = 0; kc < KC; ++kc) {
    const long long p0 = PROF ? clock64() : 0;
    ok &= mbar_wait(afull0 + 8 * a, pa, err);
    const long long p1 = PROF ? clock64() : 0;
    wg_fence();
    if (active) mma_k64_ss<128>(acc, make_desc(sB0 + s * tstride), make_desc(sA0 + a * DQ_A_BYTES), kc == 0);
    wg_commit();
    const long long p2 = PROF ? clock64() : 0;
    wg_wait<1>();
    if (PROF) {
      tw += p1 - p0;
      ti += p2 - p1;
      tm += clock64() - p2;
    }
    if (kc >= 1 && (lt & 127) == 0) {
      mbar_arrive(aempty0 + 8 * a_prev);
      mbar_arrive(empty0 + 8 * s_prev);
    }
    a_prev = a;
    s_prev = s;
    if (++a == DQ_SA) {
      a = 0;
      pa ^= 1u;
    }
    s = ring_next(s, S);
  }
  wg_wait<0>();
#pragma unroll
  for (int i = 0; i < 64; ++i) keepf(acc[i]);
  const long long tl1 = PROF ? clock64() : 0;
  named_sync(1, 256);  // both MMA warpgroups are done with the token tiles: reuse them for staging
  store_out_t<GATED>(acc, sB, active, m, w, g, q, lt, valid, row0, nb, P, cscale[e], out);
  dq_prof_mma<PROF>(err, lt, tw, ti, tm, tl1 - tl0, PROF ? clock64() - tl1 : 0);
}

// grid (NB, max pieces), 640 threads; same arguments and outputs as gemm_kernel.
// VAR: 20 = R53, 21 = R53 + timers, 24 = R55 (swapped operands), 25 = R55 + timers.
__host__ __device__ constexpr bool dq_swap(int v) { return v == 24 || v == 25; }
__host__ __device__ constexpr bool dq_prof(int v) { return v == 21 || v == 25; }

template <bool GATED, int VAR>
__global__ void __launch_bounds__(DQ_THREADS, 1) dq_kernel(const uint8_t* __restrict__ wblob,
                                                         const __nv_bfloat16* __restrict__ bsrc,
                                                         const int4* __restrict__ pieces,
                                                         const int* __restrict__ n_pieces,
                                                         const float* __restrict__ cscale,
                                                         __nv_bfloat16* __restrict__ out, int* __restrict__ err, int P,
                                                         int NB, int smax) {
  const int nb = blockIdx.x, pc = blockIdx.y;
  if (pc >= *n_pieces) return;
  const int4 piece = pieces[pc];
  const int e = piece.x, row0 = piece.y, valid = piece.z, nsel = piece.w;
  const int bbytes = nsel * 128;
  const int tstride = dq_swap(VAR) ? ((nsel + 63) & ~63) * 128 : bbytes;  // token slot (R55: whole 64-row A tiles)
  int S = DQ_RING / (WS_BYTES + tstride);
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  S &= ~1;            // even (see above)
  S = S > 4 ? S : 4;  // 4 x (5120 + 16384) + 4 x 16384 B fits SMEM_BYTES
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sA = smem;                          // DQ_SA x 16 KB, 1024-aligned
  uint8_t* sB = sA + DQ_SA * DQ_A_BYTES;       // S x tstride (multiples of 2048)
  uint8_t* sW = sB + S * tstride;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + S * WS_BYTES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const uint32_t afull0 = smem_u32(bars + 2 * SMAX), aempty0 = smem_u32(bars + 2 * SMAX + DQ_SA);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int i = 0; i < S; ++i) {
      mbar_init(full0 + 8 * i, 1);
      mbar_init(empty0 + 8 * i, 6);  // 4 decoder warps (weights read) + 2 MMA warpgroups (B read)
    }
    for (int i = 0; i < DQ_SA; ++i) {
      mbar_init(afull0 + 8 * i, 4);   // the decoding warpgroup's 4 warps
      mbar_init(aempty0 + 8 * i, 2);  // both MMA warpgroups
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  const int wgi = tid >> 7;
  bool ok = true;
  if (wgi == 0) {
    setmaxnreg_dec<DQ_REG_PROD>();
    if (tid == 0) {
      const uint8_t* wsrc = wblob + static_cast<size_t>(e * NB + nb) * KC * WS_BYTES;
      int s = 0;
      uint32_t round = 0;
      long long tw = 0;
      const long long tp0 = dq_prof(VAR) ? clock64() : 0;
      for (int kc = 0; kc < KC; ++kc) {
        if (round > 0) {
          const long long ta = dq_prof(VAR) ? clock64() : 0;
          ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
          if (dq_prof(VAR)) tw += clock64() - ta;
        }
        mbar_expect_tx(full0 + 8 * s, WS_BYTES + bbytes);
        bulk_g2s(smem_u32(sW + s * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
        bulk_g2s(smem_u32(sB + s * tstride), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * s);
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
      if (dq_prof(VAR)) {
        prof_add(err, 9, tw);
        prof_add(err, 10, clock64() - tp0);
        prof_add(err, 11, 1);
      }
    }
  } else if (wgi <= 2) {
    setmaxnreg_dec<DQ_REG_DEC>();
    dq_decode<dq_prof(VAR)>(sW, sA, full0, empty0, afull0, aempty0, S, wgi - 1, tid, ok, err);
  } else {
    setmaxnreg_inc<DQ_REG_MMA>();
    if constexpr (dq_swap(VAR)) {
      dq_mma_swap<GATED, dq_prof(VAR)>(sB, sA, empty0, afull0, aempty0, S, tstride, nsel, tid, e, row0, valid, nb,
                                       cscale, out, err, P, ok);
    } else {
#define R14_DQ_CASE(NN) \
  case NN: dq_mma<GATED, NN, dq_prof(VAR)>(sB, sA, empty0, afull0, aempty0, S, tid, e, row0, valid, nb, cscale, out, err, P, ok); break;
    switch (nsel) {
      R14_DQ_CASE(16) R14_DQ_CASE(32) R14_DQ_CASE(48) R14_DQ_CASE(64) R14_DQ_CASE(80) R14_DQ_CASE(96)
      R14_DQ_CASE(112)
      default: dq_mma<GATED, 128, dq_prof(VAR)>(sB, sA, empty0, afull0, aempty0, S, tid, e, row0, valid, nb, cscale,
                                                out, err, P, ok);
               break;
    }
#undef R14_DQ_CASE
    }
  }
  if (!ok) atomicAdd(err, 1);
}

// One block: per-expert counts of the local routes, 16-aligned segments, tiles (see mode),
// and each route's row (pos, -1 if not local). Rows inside a segment are in arbitrary order (results
// don't depend on it).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ ids, int slots, int* __restrict__ pos,
                                                     int4* __restrict__ pieces, int* __restrict__ n_pieces,
                                                     int max_pieces, int mode) {
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
        // mode 0: tiles 128/64/32/16; mode 1: 128-row tiles, the last one ceil16(rem) rows (one pass per 128 routes);
        // mode 2 (R54): the same with 64-row tiles
        const int cap = mode == 2 ? 64 : 128;
        const int n = rem >= cap ? cap : mode ? ((rem + SEG - 1) / SEG) * SEG : rem >= 64 ? 64 : rem >= 32 ? 32 : 16;
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

namespace r14 {

constexpr size_t LAUNCH_SMEM = static_cast<size_t>(SMEM_BYTES) + 16 * SMAX + 1024;

template <int CL, int DIAG, bool G>
cudaError_t launch(dim3 grid, cudaStream_t stream, const uint8_t* wblob, const __nv_bfloat16* bsrc, const int4* pieces,
            const int* n_pieces, const float* cscale, __nv_bfloat16* out, int* err, int P, int NB, int smax) {
  auto kern = gemm_kernel<2, G, CL, DIAG>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(LAUNCH_SMEM));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = LAUNCH_SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = CL;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = CL > 1 ? 1 : 0;
  return cudaLaunchKernelEx(&cfg, kern, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
}

template <int VAR, bool G>
cudaError_t launch_dq(dim3 grid, cudaStream_t stream, const uint8_t* wblob, const __nv_bfloat16* bsrc, const int4* pieces,
                      const int* n_pieces, const float* cscale, __nv_bfloat16* out, int* err, int P, int NB, int smax) {
  auto kern = dq_kernel<G, VAR>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(DQ_LAUNCH_SMEM));
  kern<<<grid, DQ_THREADS, DQ_LAUNCH_SMEM, stream>>>(wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  return cudaGetLastError();
}

template <int DIAG, bool G>
cudaError_t launch_wide(dim3 grid, cudaStream_t stream, const uint8_t* wblob, const __nv_bfloat16* bsrc,
                        const int4* pieces, const int* n_pieces, const float* cscale, __nv_bfloat16* out, int* err,
                        int P, int NB, int smax) {
  auto kern = wide_kernel<G, DIAG>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(LAUNCH_SMEM));
  const dim3 g2(grid.x / 2, grid.y);  // two 128-row blocks per CTA
  kern<<<g2, WIDE_THREADS, LAUNCH_SMEM, stream>>>(wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  return cudaGetLastError();
}

template <bool G>
cudaError_t launch_any(int64_t cl, int64_t diag, dim3 grid, cudaStream_t stream, const uint8_t* wblob,
                const __nv_bfloat16* bsrc, const int4* pieces, const int* n_pieces, const float* cscale,
                __nv_bfloat16* out, int* err, int P, int NB, int smax) {
#ifdef R14_PROBE
  if (cl == 4) return launch<4, 0, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  return launch<1, 0, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
#else
  // RE7-RE18 ruled out clusters, diag 1-2, 4-13, 15-19; only these instances are built (compile time).
  (void)cl;
  if (diag == 20) return launch_dq<20, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  if (diag == 21) return launch_dq<21, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  if (diag == 24) return launch_dq<24, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  if (diag == 25) return launch_dq<25, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  if (diag == 3) return launch<1, 3, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  if (diag == 14) return launch<1, 14, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
  return launch<1, 0, G>(grid, stream, wblob, bsrc, pieces, n_pieces, cscale, out, err, P, NB, smax);
#endif
}

}  // namespace r14

// depth must be 2 (the only instance v6 builds); cl in {1, 2, 4} must divide NB; diag 1/2/3 only with cl 1 (lab;
// diag 3 needs err to hold 2 + 2 * 16 ints).
void r14_gemm(torch::Tensor wblob, torch::Tensor bsrc, torch::Tensor pieces, torch::Tensor n_pieces,
              torch::Tensor cscale, torch::Tensor out, torch::Tensor err, int64_t P, int64_t NB,
              int64_t max_pieces, int64_t depth, int64_t smax, bool gated, int64_t cl, int64_t diag) {
  TORCH_CHECK(depth == 2 && cl == 1 && (diag == 0 || diag == 3 || diag == 14 || diag == 20 || diag == 21 || diag == 24 || diag == 25),
              "r14_gemm: unsupported config (v6 builds cl 1 and diag 0, 3, 14, 20, 21, 24, 25; R54 22/23 not built)");
  const dim3 grid(static_cast<unsigned>(NB), static_cast<unsigned>(max_pieces));
  const auto stream = at::cuda::getCurrentCUDAStream();
  const auto* w = wblob.data_ptr<uint8_t>();
  const auto* b = reinterpret_cast<const __nv_bfloat16*>(bsrc.data_ptr());
  const auto* pc = reinterpret_cast<const int4*>(pieces.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
  cudaError_t e;
  if (gated)
    e = r14::launch_any<true>(cl, diag, grid, stream, w, b, pc, n_pieces.data_ptr<int>(), cscale.data_ptr<float>(), o,
                          err.data_ptr<int>(), static_cast<int>(P), static_cast<int>(NB), static_cast<int>(smax));
  else
    e = r14::launch_any<false>(cl, diag, grid, stream, w, b, pc, n_pieces.data_ptr<int>(), cscale.data_ptr<float>(), o,
                               err.data_ptr<int>(), static_cast<int>(P), static_cast<int>(NB), static_cast<int>(smax));
  if (e != cudaSuccess) {
    cudaGetLastError();  // a rejected launch (e.g. cluster resources) must not surface in a later, unrelated call
    TORCH_CHECK(false, "r14_gemm launch failed: ", cudaGetErrorString(e));
  }
}

// Registers and local (spill) bytes per instance: (cl 1, gated), (cl 1, plain), (cl 2, gated), (cl 2, plain),
// (cl 4, gated), (cl 4, plain); then the most co-resident clusters of 2 and 4 (gated, plain).
torch::Tensor r14_attrs() {
  int64_t v[16] = {0};
#ifndef R14_PROBE
  cudaFuncAttributes a;
#define R14_ATTR(I, D, G)                                            \
  cudaFuncGetAttributes(&a, r14::gemm_kernel<2, G, 1, D>);           \
  v[I] = a.numRegs;                                                  \
  v[I + 1] = static_cast<int64_t>(a.localSizeBytes);
  // slots 0-3: diag 0 (gated, plain); 4-7: diag 14 (R47 decode)
  R14_ATTR(0, 0, true) R14_ATTR(2, 0, false) R14_ATTR(4, 14, true) R14_ATTR(6, 14, false)
#undef R14_ATTR
#define R14_DATTR(I, D, G)                                           \
  cudaFuncGetAttributes(&a, r14::dq_kernel<G, D>);                   \
  v[I] = a.numRegs;                                                  \
  v[I + 1] = static_cast<int64_t>(a.localSizeBytes);
  // slots 8-11: R55 (diag 24), 12-15: R55 timed (diag 25); R53 (20/21) built but not reported
  R14_DATTR(8, 24, true) R14_DATTR(10, 24, false) R14_DATTR(12, 25, true) R14_DATTR(14, 25, false)
#undef R14_DATTR
  cudaGetLastError();
#endif
  return torch::tensor(std::vector<int64_t>(v, v + 16));
}

void r14_route(torch::Tensor ids, torch::Tensor pos, torch::Tensor pieces, torch::Tensor n_pieces,
               int64_t max_pieces, int64_t mode) {
  r14::route_kernel<<<1, 1024, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), static_cast<int>(ids.numel()), pos.data_ptr<int>(),
      reinterpret_cast<int4*>(pieces.data_ptr()), n_pieces.data_ptr<int>(), static_cast<int>(max_pieces),
      static_cast<int>(mode));
}
"""

_R14_CPP = """
void r14_gemm(torch::Tensor wblob, torch::Tensor bsrc, torch::Tensor pieces, torch::Tensor n_pieces,
              torch::Tensor cscale, torch::Tensor out, torch::Tensor err, int64_t P, int64_t NB,
              int64_t max_pieces, int64_t depth, int64_t smax, bool gated, int64_t cl, int64_t diag);
void r14_route(torch::Tensor ids, torch::Tensor pos, torch::Tensor pieces, torch::Tensor n_pieces, int64_t max_pieces,
               int64_t mode);
torch::Tensor r14_attrs();
"""


def r14_source(probe=False):
    src = _R14_CUDA.replace("@@WGMMA@@", "".join(_r14_wgmma_fn(n) + _r14_wgmma_ss_fn(n)
                                                  for n in (16, 32, 48, 64, 80, 96, 112, 128)))
    return ("#define R14_PROBE 1\n" + src) if probe else src


def r14_build(probe=False):
    """probe (lab only): N = 128 and CL 1 / 4 only, built verbosely with ptxas -v so its register report and any
    wgmma serialization advisory reach the log."""
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-std=c++20"]
    return load_inline(
        name="r14_probe" if probe else "r14_moe", cpp_sources=_R14_CPP, cuda_sources=r14_source(probe),
        functions=["r14_gemm", "r14_route", "r14_attrs"],
        extra_cuda_cflags=flags + (["-Xptxas", "-v"] if probe else []),
        build_directory=tempfile.mkdtemp(prefix="r14p_" if probe else "r14_"), verbose=probe)


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


def _r14_bitpos_v2():
    """R47 bit layout: [value i][bit b] -> word bit, b = 3 sign, 2 e1, 1 e0, 0 m. Value 2r is the low bf16 lane of
    output register r, 2r + 1 the high lane. r0 = w & 0x81C081C0; r1 = rotr(w, 3) & 0x81C081C0;
    r2 = (rotr(w, 1) & 0x80008000) | (rotr(w, 6) & 0x01C001C0); r3 = (rotr(w, 2) & 0x80008000) | (rotr(w, 13) & 0x01C001C0)."""
    out = {0: (15, 8, 7, 6), 1: (31, 24, 23, 22)}  # lane -> output bits of (sign, e1, e0, m)
    pos = []
    for r, (ts, tm) in enumerate(((0, 0), (3, 3), (1, 6), (2, 13))):
        for lane in (0, 1):
            o = out[lane]
            pos.append(((o[0] + ts) % 32, (o[1] + tm) % 32, (o[2] + tm) % 32, (o[3] + tm) % 32))
    flat = [b for p_ in pos for b in p_]
    assert sorted(flat) == list(range(32)), "R47 layout must use every word bit exactly once"
    return [[p_[3 - b] for b in range(4)] for p_ in pos]  # [i][b]: b = 0 m, 1 e0, 2 e1, 3 sign


def r14_pack(q, sf, gated, layout=0):
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
        if layout == 0:
            words = (cv.to(torch.int64) << shift).sum(-1).to(torch.int32)  # [nb, kc, c, w, lane, ks]
        else:  # R47: each value's 4 bits at their own positions (lossless: the same bits, rearranged)
            bp = torch.tensor(_r14_bitpos_v2(), dtype=torch.int64, device=q.device)  # [8, 4]
            bits = (cv.to(torch.int64).unsqueeze(-1) >> torch.arange(4, device=q.device)) & 1  # [..., 8, 4]
            words = (bits << bp).sum((-1, -2)).to(torch.int64)
            words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
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


def _r14_sizes(tokens, mode=1):
    slots = tokens * TOP_K
    tile = 64 if mode == 2 else R14_BT  # mode 2 (R54): at most ceil(routes / 64) + 1 tiles per expert
    max_pieces = -(-slots // tile) + 4 * LOCAL_EXPERTS
    rows = slots + LOCAL_EXPERTS * R14_SEG + R14_BT  # padded segments + one tile of slack for bulk reads
    return slots, max_pieces, rows


def _experts_r14(x, topk_weights, topk_ids, state, cfg):
    """Same contract as _experts."""
    r14, gmm, bank = state["r14"], state["gmm"], state["bank"]
    tokens = x.shape[0]
    depth, smax, mode, cl = cfg[:4]
    slots, max_pieces, rows = _r14_sizes(tokens, mode)
    dev = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    pos = torch.empty(slots, device=dev, dtype=torch.int32)
    pieces = torch.empty((max_pieces, 4), device=dev, dtype=torch.int32)
    n_pieces = torch.empty(1, device=dev, dtype=torch.int32)
    ext = r14["ext"]
    diag = cfg[4] if len(cfg) > 4 else 0
    b13, b2 = (r14["w13b"], r14["w2b"]) if diag in R14_LAYOUT2_DIAGS else (r14["w13"], r14["w2"])
    ext.r14_route(local_ids.view(-1), pos, pieces, n_pieces, max_pieces, mode)
    a_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    _r14_gather_kernel[(slots,)](x.contiguous(), pos, a_sw, rows, H=D_EXPERT, TOPK=TOP_K, num_warps=4)
    act_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(b13, a_sw, pieces, n_pieces, r14["c1"], act_sw, r14["err"], rows, D_EXPERT // 64,
                 max_pieces, depth, smax, True, cl, diag)
    y2 = torch.empty((rows, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(b2, act_sw, pieces, n_pieces, r14["c2"], y2, r14["err"], rows, D_EXPERT // 128,
                 max_pieces, depth, smax, False, cl, diag)
    out = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, pos, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


# R11: on the grouped path (first-turn class), the NVFP4 -> BF16 decode of the experts depends on the weights only,
# so it starts on a side stream at the top of kernel_fn and runs under attention and the dense projections
# (compute-bound) instead of between the routing and the expert GEMMs. All 32 local experts are decoded (the touched
# set isn't known yet); the main stream joins before the grouped GEMMs. Same bytes in the same BF16 scratch, same
# GEMMs: the output is bit-identical. A fork/join inside the captured graph, like the baseline's router side stream.
R11_ON = True
R11_MIN_TOKENS = 8192


def _r11_fork(state, tokens):
    """Called first thing in kernel_fn. Starts the decode of every local expert on gmm["deq_stream"] when the call
    will take the grouped path; _moe_v4 joins it."""
    gmm, r14 = state["gmm"], state.get("r14") or {}
    state["r11_live"] = False
    if not (R11_ON and gmm.get("ok") and "deq_stream" in gmm and tokens >= R11_MIN_TOKENS
            and r14.get("cfg", {}).get(_r14_class(tokens)) is None):
        return
    main, side = torch.cuda.current_stream(), gmm["deq_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _dequant(gmm["w13_q"], gmm["w13_sf"], gmm["all_counts"], gmm["w13"])
        _dequant(gmm["w2_q"], gmm["w2_sf"], gmm["all_counts"], gmm["w2"])
    state["r11_live"] = True


def _experts_gmm_r11(x, topk_weights, topk_ids, state):
    """_experts_gmm without its two _dequant calls (R11 already decoded every local expert); otherwise the same
    statements (check_v6_vs_baseline.py compares the ASTs)."""
    gmm, bank = state["gmm"], state["bank"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    device = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
    sorted_keys, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,
                                  num_warps=4)
    h1 = torch._grouped_mm(a, gmm["w13"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    act = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w13_gs"], offs, act, I=D_EXPERT, NE=LOCAL_EXPERTS,
                              num_warps=4)
    y2 = torch._grouped_mm(act, gmm["w2"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    out = torch.empty((tokens, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe_v4(x, topk_weights, topk_ids, state, tokens):
    if state.pop("r11_live", False):  # R11's decode ran on the side stream: join it, then the grouped GEMMs
        torch.cuda.current_stream().wait_stream(state["gmm"]["deq_stream"])
        return _experts_gmm_r11(x, topk_weights, topk_ids, state)
    cfg = state["r14"]["cfg"].get(_r14_class(tokens))
    if cfg is not None:
        return _experts_r14(x, topk_weights, topk_ids, state, cfg)
    if tokens >= 8192 and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def r14_check(state, cfg, tokens, seed=2):
    """(rel err, worst row err, error flags) of _experts_r14 against SGLang's Marlin path."""
    dev = state["bank"]["expert_map"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=dev) * 0.5).to(torch.bfloat16)
    ids, w = _gmm_selftest_routing(tokens, dev, seed=seed)
    state["r14"]["err"].zero_()
    ref = _experts(x, w, ids, state).float()
    out = _experts_r14(x, w, ids, state, cfg).float()
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
    gmm["deq_stream"] = torch.cuda.Stream(gmm["w13_q"].device)  # R11
    gmm["all_counts"] = torch.ones(LOCAL_EXPERTS + 1, dtype=torch.int32, device=gmm["w13_q"].device)
    c1, c2 = r14_epilogue_scales(gmm)
    r14 = {"cfg": {}, "ok": False, "c1": c1, "c2": c2,
           "err": torch.zeros(1, dtype=torch.int32, device=weights["w13"].device)}
    state["r14"] = r14
    try:
        r14["w13"] = r14_pack(gmm["w13_q"], gmm["w13_sf"], True)
        r14["w2"] = r14_pack(gmm["w2_q"], gmm["w2_sf"], False)
        if R14_NEED_LAYOUT2:
            r14["w13b"] = r14_pack(gmm["w13_q"], gmm["w13_sf"], True, layout=1)
            r14["w2b"] = r14_pack(gmm["w2_q"], gmm["w2_sf"], False, layout=1)
        r14["ext"] = r14_build()
        r14["ok"] = True
    except Exception as exc:
        print(f"[v6] R14 build/pack failed, Marlin/v2 stay: {type(exc).__name__}: {str(exc)[-600:]}", flush=True)
        return
    for cls, cfg in R14_CONFIGS.items():
        try:
            err, row, flags = r14_check(state, cfg, R14_REP_TOKENS[cls])
            ok = flags == 0 and err < R14_MAX_ERR and row < 5 * R14_MAX_ERR
            print(f"[v6] {cls}: R14 vs Marlin err {err:.2e} row {row:.2e} flags {flags} -> {'R14' if ok else 'fallback'}",
                  flush=True)
            if ok:
                r14["cfg"][cls] = cfg
        except Exception as exc:
            print(f"[v6] {cls}: R14 failed, fallback: {type(exc).__name__}: {str(exc)[-300:]}", flush=True)
    torch.cuda.synchronize()
