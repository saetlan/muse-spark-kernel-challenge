# Adapts DeepGEMM's sm90 BF16 GEMM kernel (MIT License, Copyright (c) 2025 DeepSeek); see the NOTICE file at the repo root.
"""DG12 (P40 step 2): DeepGEMM's sm90 BF16 GEMM (image header deep_gemm/impls/sm90_bf16_gemm.cuh, sgl-project/DeepGEMM
cbe1727) with only the weight load changed to the 12-bit lossless format of proj/dg/dg12_layout.py.

out[m, n] = x[m, :] . W[n, :] with DeepGEMM's A = x (TMA, SW128 [64 k, BLOCK_M]), B = W. Per 64-k stage the producer
thread TMAs A and the raw planes (chunk-major in global memory, PF42: a stage's Q rows = one contiguous BLOCK_N x 32 B
block read as [128 B x BLOCK_N / 4] rows, L likewise [128 B x BLOCK_N / 2]; no swizzle) into a raw stage; the producer
warpgroup's three other warps decode each (row, 8-k unit) in 7 ops (dg12_decode = the F12 decode) into one 16-byte
unit of the K-major SW128 B tile at row * 128 + ((u ^ (row & 7)) << 4) (exactly where TMA SW128 would put it), fence the
generic->async proxy and arrive on the stage's full barrier (count 1 producer + 3 decoders, plus A's tx bytes); the math
warpgroups run DeepGEMM's SS wgmma loop unchanged. Epilogue, before DeepGEMM's STSM + TMA store: escapes (decoded as the
placeholder 2^-15) are corrected per 8-column group from a sorted list (acc -= x_k 2^-15; acc += x_k w_true_scaled),
then column n is scaled by 2^(base_n - 112). Normal GEMM only, K-major, 1-CTA clusters, plain launch (as DeepJIT).

Instance: (BM, BN, ST, SWD, NMATH, N, K[, DIAG]). DIAG (lab diagnostics, wrong results): 1 = decoders skip the decode
(wait, fence, arrive only), 2 = the producer skips the raw-plane TMAs (plain arrive), 4 = decoders skip the proxy
fence, 8 = no escape corrections; bits combine; 0 = the real kernel. Escapes (PF45): per (n-block, lane class t) tables,
each thread's entries and their x values loaded at block start, applied from registers in the epilogue.
"""

import os
import tempfile

DG12_CU = r"""
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

namespace deep_gemm {

// The F12 7-op decode: 3 words -> 8 BF16 (element e = k0 + e), scaled by 2^(112 - base).
__device__ __forceinline__ uint4 dg12_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

// Index helpers (check_dg12.py evaluates their return expressions): byte offsets of (row r, 8-k unit u) in a raw Q
// stage, a raw L stage and the SW128 B tile; the decoder's (row, unit) for pass row r0 and lane; the accumulator index
// of n8 group j, row half h, column parity c in wave l; the row in the block of wave l, warpgroup g, warp w, lane row
// gr, half h.
__device__ __forceinline__ uint32_t dg12_smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t dg12_lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void dg12_lds64(uint32_t a, uint32_t& x, uint32_t& y) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(a));
}
__device__ __forceinline__ void dg12_sts128(uint32_t a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint32_t dg12_qoff(uint32_t r, uint32_t u) { return r * 32 + u * 4; }
__device__ __forceinline__ uint32_t dg12_loff(uint32_t r, uint32_t u) { return r * 64 + u * 8; }
__device__ __forceinline__ uint32_t dg12_boff(uint32_t r, uint32_t u) { return r * 128 + ((u ^ (r & 7)) << 4); }
__device__ __forceinline__ uint32_t dg12_drow(uint32_t r0, uint32_t lane) { return r0 + (lane >> 3); }
__device__ __forceinline__ uint32_t dg12_dunit(uint32_t lane) { return lane & 7; }
__device__ __forceinline__ uint32_t dg12_acc(uint32_t na, uint32_t l, uint32_t j, uint32_t h, uint32_t c) { return na * l + 4 * j + 2 * h + c; }
__device__ __forceinline__ uint32_t dg12_row(uint32_t wave, uint32_t l, uint32_t g, uint32_t wm, uint32_t w, uint32_t gr, uint32_t h) { return l * wave + g * wm + 16 * w + gr + 8 * h; }

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumMathThreads, uint32_t kNumSMs, uint32_t kDiag>
CUTLASS_GLOBAL __launch_bounds__(128 + kNumMathThreads, 1) void
dg12_gemm_impl(uint32_t shape_m,
               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
               const __grid_constant__ cute::TmaDescriptor tensor_map_l,
               const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
               const float* __restrict__ col_scale, const int* __restrict__ esc_offs,
               const int4* __restrict__ esc_ent, const __nv_bfloat16* __restrict__ x) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumDecodeWarps = 3;
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    DG_STATIC_ASSERT(SHAPE_N % BLOCK_N == 0 and SHAPE_K % BLOCK_K == 0 and BLOCK_N % 8 == 0 and BLOCK_N / 2 <= 256, "dg12 shape");
    const uint32_t shape_n = SHAPE_N, shape_k = SHAPE_K;

    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_N * 32;
    static constexpr uint32_t SMEM_L_SIZE_PER_STAGE = BLOCK_N * 64;
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0 and SMEM_Q_SIZE_PER_STAGE % 128 == 0,
                     "dg12 smem alignment");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_l);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    uint8_t* const smem_raw = smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) { return smem_raw + i * SMEM_Q_SIZE_PER_STAGE; });
    auto smem_l = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_raw + kNumStages * SMEM_Q_SIZE_PER_STAGE + i * SMEM_L_SIZE_PER_STAGE;
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_raw + kNumStages * (SMEM_Q_SIZE_PER_STAGE + SMEM_L_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto raw_barriers   = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (2 * kNumStages + i); });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1 + kNumDecodeWarps);
            empty_barriers[i]->init(kNumMathThreads / 32);
            raw_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = kNumMathThreads == 128 ? 64 : 56;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, 1, false, kNumSMs>(shape_m, shape_n, shape_k, nullptr);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;

    if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        const uint32_t tma_warp = warp_idx - kNumMathThreads / 32;
        if (tma_warp == 2) {
            // Producer: one elected thread issues A and the raw planes for every stage
            if (cute::elect_one_sync()) {
                while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                    for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        const uint32_t m_idx = m_block_idx * BLOCK_M, n_idx = n_block_idx * BLOCK_N;
                        auto& full_barrier = *full_barriers[stage_idx];
                        tma::copy<BLOCK_K, BLOCK_M, 128, cutlass::bfloat16_t>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_block_idx * BLOCK_K, m_idx);
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                        auto& raw_barrier = *raw_barriers[stage_idx];
                        if constexpr ((kDiag & 2) == 0) {
                            tma::copy<128, BLOCK_N / 4, 0, uint8_t>(&tensor_map_q, &raw_barrier, smem_q[stage_idx], 0,
                                                                    (k_block_idx * SHAPE_N + n_idx) / 4);
                            tma::copy<128, BLOCK_N / 2, 0, uint8_t>(&tensor_map_l, &raw_barrier, smem_l[stage_idx], 0,
                                                                    (k_block_idx * SHAPE_N + n_idx) / 2);
                            raw_barrier.arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_L_SIZE_PER_STAGE);
                        } else {
                            raw_barrier.arrive();             // DIAG 2: no raw planes (decoders read stale bytes)
                        }
                    }
                }
            }
        } else {
            // Decoders (PF43): raw stage -> SW128 B tile, then arrive on the full barrier. Per batch of kBatch passes the
            // warp issues all its shared loads (ld.shared, shared-window addresses) before any decode and store, so the
            // load latency overlaps instead of serializing behind the previous pass's store.
            constexpr uint32_t kRowsPerPass = kNumDecodeWarps * 4;
            constexpr uint32_t kPasses = (BLOCK_N + kRowsPerPass - 1) / kRowsPerPass;
            constexpr uint32_t kBatch = kPasses < 6 ? kPasses : 6;
            const uint32_t dw = tma_warp == 3 ? 2 : tma_warp;
            const uint32_t u = dg12_dunit(lane_idx);
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                    raw_barriers[stage_idx]->wait(phase);
                    const uint32_t q_base = dg12_smem_u32(smem_q[stage_idx]);
                    const uint32_t l_base = dg12_smem_u32(smem_l[stage_idx]);
                    const uint32_t b_base = dg12_smem_u32(smem_b[stage_idx]);
                    #pragma unroll
                    for (uint32_t p0 = 0; p0 < kPasses and (kDiag & 1) == 0; p0 += kBatch) {   // DIAG 1: no decode
                        uint32_t qv[kBatch], l0v[kBatch], l1v[kBatch];
                        #pragma unroll
                        for (uint32_t i = 0; i < kBatch; ++ i) {
                            const uint32_t r = dg12_drow(dw * 4 + (p0 + i) * kRowsPerPass, lane_idx);
                            if (p0 + i < kPasses and r < BLOCK_N) {
                                qv[i] = dg12_lds32(q_base + dg12_qoff(r, u));
                                dg12_lds64(l_base + dg12_loff(r, u), l0v[i], l1v[i]);
                            }
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < kBatch; ++ i) {
                            const uint32_t r = dg12_drow(dw * 4 + (p0 + i) * kRowsPerPass, lane_idx);
                            if (p0 + i < kPasses and r < BLOCK_N)
                                dg12_sts128(b_base + dg12_boff(r, u), dg12_decode(qv[i], l0v[i], l1v[i]));
                        }
                    }
                    if constexpr ((kDiag & 4) == 0)
                        cutlass::arch::fence_view_async_shared();   // DIAG 4: no proxy fence (measures its cost)
                    __syncwarp();
                    if (lane_idx == 0)
                        full_barriers[stage_idx]->arrive();
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            constexpr uint32_t kNumWaves = BLOCK_M / WAVE_BLOCK_M;
            float accum[WGMMA::kNumAccum * kNumWaves] = {0};
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");

            // 12-bit epilogue data (PF45), loaded now so the latency hides behind the k-loop: this thread's escapes
            // (n-block, lane class tq) with x at its rows, and its column scales
            constexpr uint32_t kRows = 2 * kNumWaves;
            constexpr uint32_t kMaxEsc = (BLOCK_N / 8) * kNumWaves >= 16 ? 6 : 12;
            const uint32_t gr = lane_idx >> 2, tq = lane_idx & 3, wq = warp_idx % 4;
            int e_lo = 0, e_hi = 0;
            int4 ent[kMaxEsc];
            float xe[kMaxEsc][kRows];
            float2 sc[BLOCK_N / 8];
            #pragma unroll
            for (uint32_t j = 0; j < BLOCK_N / 8; ++ j)
                sc[j] = __ldg(reinterpret_cast<const float2*>(col_scale + n_block_idx * BLOCK_N + 8 * j + 2 * tq));
            if constexpr ((kDiag & 8) == 0) {
                e_lo = __ldg(esc_offs + n_block_idx * 4 + tq);
                e_hi = __ldg(esc_offs + n_block_idx * 4 + tq + 1);
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i)
                    ent[i] = static_cast<int>(i) < e_hi - e_lo ? __ldg(esc_ent + e_lo + i) : make_int4(0, 0, 0, 0);
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i) {
                    #pragma unroll
                    for (uint32_t rr = 0; rr < kRows; ++ rr) {
                        const uint32_t row = dg12_row(WAVE_BLOCK_M, rr >> 1, math_wg_idx, WGMMA::M, wq, gr, rr & 1);
                        const uint32_t mg = m_block_idx * BLOCK_M + row;
                        xe[i][rr] = (static_cast<int>(i) < e_hi - e_lo and row < BLOCK_M and mg < shape_m)
                            ? __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(ent[i].y)]) : 0.0f;
                    }
                }
            }
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
                full_barriers[stage_idx]->wait(phase);
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumWaves; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < kNumWaves; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, k * WGMMA::K, 0);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, nv_bfloat16>(
                            b_desc_base_lo, 0, k * WGMMA::K, 0);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumWaves; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();
                if (lane_idx == 0)
                    empty_barriers[stage_idx]->arrive();
            }

            // 12-bit epilogue: escape corrections in the scaled domain (acc -= x_k 2^-15; acc += x_k w_true_scaled), from
            // registers for the first kMaxEsc entries and the old serial path for any overflow, then the column scale
            if constexpr ((kDiag & 8) == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i) {
                    if (static_cast<int>(i) < e_hi - e_lo) {
                        const uint32_t nl = static_cast<uint32_t>(ent[i].x), c = nl & 1u;
                        const float val = __int_as_float(ent[i].z);
                        #pragma unroll
                        for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                            if ((nl >> 3) == j) {
                                #pragma unroll
                                for (uint32_t rr = 0; rr < kRows; ++ rr) {
                                    float& a0 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)];
                                    float& a1 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)];
                                    if (c == 0) {
                                        a0 = fmaf(-xe[i][rr], 3.0517578125e-05f, a0);
                                        a0 = fmaf(xe[i][rr], val, a0);
                                    } else {
                                        a1 = fmaf(-xe[i][rr], 3.0517578125e-05f, a1);
                                        a1 = fmaf(xe[i][rr], val, a1);
                                    }
                                }
                            }
                        }
                    }
                }
                for (int e = e_lo + static_cast<int>(kMaxEsc); e < e_hi; ++ e) {
                    const int4 en = __ldg(esc_ent + e);
                    const uint32_t nl = static_cast<uint32_t>(en.x), c = nl & 1u;
                    const float val = __int_as_float(en.z);
                    #pragma unroll
                    for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                        if ((nl >> 3) == j) {
                            #pragma unroll
                            for (uint32_t rr = 0; rr < kRows; ++ rr) {
                                const uint32_t row = dg12_row(WAVE_BLOCK_M, rr >> 1, math_wg_idx, WGMMA::M, wq, gr, rr & 1);
                                const uint32_t mg = m_block_idx * BLOCK_M + row;
                                if (row < BLOCK_M and mg < shape_m) {
                                    const float xk = __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(en.y)]);
                                    float& a0 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)];
                                    float& a1 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)];
                                    if (c == 0) {
                                        a0 = fmaf(-xk, 3.0517578125e-05f, a0);
                                        a0 = fmaf(xk, val, a0);
                                    } else {
                                        a1 = fmaf(-xk, 3.0517578125e-05f, a1);
                                        a1 = fmaf(xk, val, a1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                #pragma unroll
                for (uint32_t rr = 0; rr < kRows; ++ rr) {
                    accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)] *= sc[j].x;
                    accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)] *= sc[j].y;
                }
            }

            // DeepGEMM's epilogue (BF16 D): STSM into the swizzled D tile, TMA store
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
            if (not do_wgmma_store)
                continue;
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
            DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < kNumWaves; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    constexpr uint32_t kNumBankGroupBytes = 16;
                    auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                    auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                    col ^= row % (kSwizzleDMode / 16);
                    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                        warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
                        m_offset * kSwizzleDMode +
                        atom_offset * BLOCK_M * kSwizzleDMode +
                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr);
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
            const auto m_idx = m_block_idx * BLOCK_M;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

}  // namespace deep_gemm

namespace dg12_@@NAME@@ {

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (fn == nullptr) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) fn = reinterpret_cast<EncodeTiledFn>(dlsym(h, "cuTensorMapEncodeTiled"));
  }
  return fn;
}

static CUtensorMapSwizzle swizzle_of(int mode) {
  switch (mode) {
    case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default: return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

// gmem [outer][inner] elements of elem_bytes (row stride outer_stride_bytes); with a swizzle the smem inner box is
// swizzle / elem_bytes elements (DeepGEMM's make_tma_2d_desc).
static int make_2d(CUtensorMap* map, CUtensorMapDataType dtype, int elem_bytes, void* ptr, int inner, int outer,
                   int smem_inner, int smem_outer, long long outer_stride_bytes, int swizzle) {
  EncodeTiledFn fn = encode_fn();
  if (fn == nullptr) return 9001;
  if (swizzle != 0) smem_inner = swizzle / elem_bytes;
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint32_t sdims[2] = {static_cast<cuuint32_t>(smem_inner), static_cast<cuuint32_t>(smem_outer)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(outer_stride_bytes)};
  const cuuint32_t estrides[2] = {1, 1};
  const CUresult r = fn(map, dtype, 2, ptr, gdims, gstrides, sdims, estrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        swizzle_of(swizzle), CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 10000 + static_cast<int>(r);
}

constexpr uint32_t BM = @@BM@@, BN = @@BN@@, ST = @@ST@@, SWD = @@SWD@@, NMATH = @@NMATH@@;
constexpr uint32_t N = @@N@@, K = @@K@@, SMEM = @@SMEM@@, NUM_SMS = 132, DIAG = @@DIAG@@;

static auto kernel = &deep_gemm::dg12_gemm_impl<N, K, BM, BN, SWD, ST, NMATH, NUM_SMS, DIAG>;

}  // namespace dg12_@@NAME@@

int dg12_launch_@@NAME@@(const void* x, const void* q, const void* l, void* out, const void* scale, const void* offs8,
                         const void* ent, int m, void* stream) {
  using namespace dg12_@@NAME@@;
  CUtensorMap ta, tq, tl, td;
  int e = make_2d(&ta, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), K, m, 64, BM, 2LL * K, 128);
  // chunk-major planes: Q = (K/64)*N rows of 32 B viewed as rows of 128 B; L = rows of 64 B viewed as rows of 128 B
  if (e == 0) e = make_2d(&tq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(q), 128, (K / 64) * N / 4, 128, BN / 4, 128, 0);
  if (e == 0) e = make_2d(&tl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(l), 128, (K / 64) * N / 2, 128, BN / 2, 128, 0);
  if (e == 0) e = make_2d(&td, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, out, N, m, BN, BM, 2LL * N, SWD);
  if (e != 0) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NUM_SMS, 1, 1);
  cfg.blockDim = dim3(128 + NMATH, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, static_cast<uint32_t>(m), ta, tq, tl, td,
                                             static_cast<const float*>(scale), static_cast<const int*>(offs8),
                                             static_cast<const int4*>(ent), static_cast<const __nv_bfloat16*>(x));
  return static_cast<int>(err);
}

int dg12_attr_@@NAME@@() {
  return static_cast<int>(cudaFuncSetAttribute(dg12_@@NAME@@::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(dg12_@@NAME@@::SMEM)));
}

long long dg12_info_@@NAME@@() {
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, dg12_@@NAME@@::kernel) != cudaSuccess) return -1;
  return static_cast<long long>(a.numRegs) * 1000000 + static_cast<long long>(a.localSizeBytes);
}
"""

# Instances compiled into the module, (BM, BN, ST, SWD, NMATH, N, K).
DG12_INSTANCES = ()
SMEM_CAPACITY = 232448


def dg12_unpack(inst):
    return tuple(inst) + ((0,) if len(inst) == 7 else ())


def dg12_name(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    return f"dg12_{bm}_{bn}_{st}_{swd}_{nmath}_{n}_{k}" + (f"_d{diag}" if diag else "")


def dg12_smem(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    smem_d = -(-(bm * bn * 2) // 1024) * 1024
    return smem_d + st * (bm * 128 + bn * 128 + bn * 96) + 3 * st * 8


def dg12_stages(bm, bn, max_stages=16):
    smem_d = -(-(bm * bn * 2) // 1024) * 1024
    return min(max_stages, (SMEM_CAPACITY - smem_d) // (bm * 128 + bn * 128 + bn * 96 + 24))


def dg12_cu(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    src = DG12_CU
    for key, val in (("NAME", dg12_name(inst)), ("BM", bm), ("BN", bn), ("ST", st), ("SWD", swd), ("NMATH", nmath),
                     ("N", n), ("K", k), ("SMEM", dg12_smem(inst)), ("DIAG", diag)):
        src = src.replace(f"@@{key}@@", str(val))
    return src


def dg12_cpp(inst):
    """C++ side of one instance's module: torch wrapper (shape checks, current stream), attr and info."""
    name, n, k = dg12_name(inst), inst[5], inst[6]
    return f"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int dg12_launch_{name}(const void* x, const void* q, const void* l, void* out, const void* scale, const void* offs8,
                       const void* ent, int m, void* stream);
int dg12_attr_{name}();
long long dg12_info_{name}();
void {name}(torch::Tensor x, torch::Tensor q, torch::Tensor l, torch::Tensor scale, torch::Tensor offs8, torch::Tensor ent,
            torch::Tensor out) {{
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "dg12: x/out dtype");
  TORCH_CHECK(q.scalar_type() == at::kInt && l.scalar_type() == at::kInt && offs8.scalar_type() == at::kInt
              && ent.scalar_type() == at::kInt && scale.scalar_type() == at::kFloat, "dg12: plane dtypes");
  TORCH_CHECK(x.is_contiguous() && q.is_contiguous() && l.is_contiguous() && out.is_contiguous() && scale.is_contiguous()
              && offs8.is_contiguous() && ent.is_contiguous(), "dg12: layouts");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == {k} && x.size(0) >= 1 && out.size(0) == x.size(0) && out.size(1) == {n}, "dg12: x/out shape");
  TORCH_CHECK(q.numel() >= {n}LL * {k // 8} && l.numel() >= {n}LL * {k // 4} && scale.numel() >= {n}
              && offs8.numel() >= {(n // inst[1]) * 4 + 1}, "dg12: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(l.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(scale.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(ent.data_ptr()) % 16 == 0,
              "dg12: unaligned");
  const int e = dg12_launch_{name}(x.data_ptr(), q.data_ptr(), l.data_ptr(), out.data_ptr(), scale.data_ptr(),
                                    offs8.data_ptr(), ent.data_ptr(), static_cast<int>(x.size(0)),
                                    static_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
  TORCH_CHECK(e == 0, "dg12 launch failed: ", e);
}}
void {name}_attr() {{
  const int e = dg12_attr_{name}();
  TORCH_CHECK(e == 0, "dg12 attr failed: ", e);
}}
int64_t {name}_info() {{ return static_cast<int64_t>(dg12_info_{name}()); }}
"""


def dg12_include_dirs():
    import deep_gemm
    return [os.path.join(os.path.dirname(deep_gemm.__file__), "include")]


def dg12_build_one(inst, include_dirs):
    from torch.utils.cpp_extension import load_inline
    name = dg12_name(inst)
    return load_inline(
        name="okc_proj_" + name, cpp_sources=dg12_cpp(inst), cuda_sources=dg12_cu(inst),
        functions=[name, name + "_attr", name + "_info"], extra_include_paths=list(include_dirs),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "--ptxas-options=--register-usage-level=10"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_" + name + "_"), verbose=False, no_implicit_headers=True)


def dg12_build(instances, include_dirs=None, timeout=None):
    """Every instance's module built from a thread pool; waits at most `timeout` seconds. Returns (namespace of the
    built functions, {name: error})."""
    import concurrent.futures
    import types
    inc = include_dirs or dg12_include_dirs()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(instances)))
    futures = {pool.submit(dg12_build_one, inst, inc): inst for inst in instances}
    done, _ = concurrent.futures.wait(list(futures), timeout=timeout)
    pool.shutdown(wait=False)
    fns, errors = {}, {}
    for fut, inst in futures.items():
        name = dg12_name(inst)
        if fut not in done:
            errors[name] = TimeoutError(f"build not finished within {timeout:.0f}s")
            continue
        try:
            mod = fut.result()
            for fn in (name, name + "_attr", name + "_info"):
                fns[fn] = getattr(mod, fn)
        except Exception as exc:
            errors[name] = exc
    return types.SimpleNamespace(**fns), errors
