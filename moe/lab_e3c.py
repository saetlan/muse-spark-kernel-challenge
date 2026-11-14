# okc: problem=muse_moe_w4a16
"""v1: SGLang's own Marlin W4A16 expert path, with two changes.

* Marlin's tile height (moe_block_size) is the one measured fastest at each token count. SGLang
  sizes it from num_tokens * top_k / local_experts, as if every route landed on this GPU; under
  8-way expert parallelism only about 1/8 do, so from 128 tokens up its tiles are mostly padding.
* The post-expert norm and weighted sum always run as SGLang's fused Triton kernel. Below 32
  tokens SGLang falls back to about ten separate PyTorch ops on a [tokens, 8, 4096] FP32 tensor.

Everything else is SGLang's code path."""

from typing import TypedDict

import types

import torch
import triton
from sgl_kernel.scalar_type import scalar_types

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    _post_expert_norm_weighted_sum_kernel,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096, the width the MoE block runs its experts at
    d_expert: int     # I = 4096
    n_experts: int    # 256 experts in the model
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32: one GPU under 8-way expert parallelism
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
POST_NORM_EPS = 1e-8
# (fewer than this many tokens, tile height). Measured on judge-like routings (E1): 8 is best
# through 16 tokens and within 4% of 16 at 32; 16 is best at 128; 32 is best at 256.
BLOCK_SIZE_M_STEPS = ((32, 8), (192, 16))
BLOCK_SIZE_M_LARGE = 32


def block_size_m(num_tokens: int) -> int:
    for below, size in BLOCK_SIZE_M_STEPS:
        if num_tokens < below:
            return size
    return BLOCK_SIZE_M_LARGE


def post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    """out[t] = sum over live slots k of topk_weights[t, k] * rmsnorm(y[t, k] * norm_w), using
    SGLang's fused kernel at every token count. y: [T, top_k, H]; slot_ids < 0 marks a slot whose
    expert isn't on this GPU."""
    y = y.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        y, norm_w, topk_weights, out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True,
        WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4, num_stages=2,
    )
    return out


def _marlin_layer(weights, start, count):
    """The parameters prepare_moe_nvfp4_layer_for_marlin reads, for experts [start, start+count)."""
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        # One global scale per expert for gate and up, as in SGLang's Marlin path.
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def _v1_prepare(weights):
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    # MuseSpark1xFp4Experts binds the effective (1 + w) weight of the post-expert norm.
    norm_w = (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16)
    # Compile and load the post-norm kernel now, so its first launch never lands inside a CUDA
    # graph capture; the dummy rows have no live slot and are discarded.
    hidden = norm_w.shape[0]
    post_expert_norm_weighted_sum(
        torch.zeros((1, TOP_K, hidden), dtype=torch.bfloat16, device=device),
        torch.zeros((1, TOP_K), dtype=torch.float32, device=device), norm_w,
        torch.full((1, TOP_K), -1, dtype=torch.int32, device=device))
    torch.cuda.synchronize()
    return {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": norm_w,
    }


def run_experts(x, topk_ids, topk_weights, state, bank, moe_block):
    """MuseSpark1xFp4Experts._forward_marlin from the pre-expert norm on, at a given tile height."""
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = topk_ids.to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, moe_block)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, moe_block, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return post_expert_norm_weighted_sum(y, topk_weights, state["norm_w"], local_ids)


def _v1_kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    return run_experts(x, data["topk_ids"], data["topk_weights"], state, bank, block_size_m(x.shape[0]))


# ====================================================================== M11 (moe/m11/m11.py)

"""M11: CUDA expert GEMMs for small batches; v1's Marlin path keeps the larger ones.

Swap-AB mma.sync.m16n8k16: 16 weight rows are the A operand, 8 * NT of an expert's pairs at a
time are N. prepare() reorders each expert's NVFP4 codes (lossless) so that one 32-bit word is exactly
one thread's 8 A elements for one MMA, decoded straight into A registers with Marlin's bit trick,
and permutes K inside every 64-wide chunk so that a thread's 4 MMAs read 16 consecutive
activations, which are one NVFP4 scale group: 2 scales per thread per chunk. Every CTA scans the
pairs itself (no sort/align kernel); groups of KS warps work through (active expert, TT row tiles)
tasks, splitting K and adding partial sums in shared memory. GEMM1 computes gate and up tiles
together and applies SwiGLU in its epilogue."""

import tempfile

import triton.language as tl
from torch.utils.cpp_extension import load_inline

# Tokens M11 accepts at most; it is also the kernel's MAXP (pairs per expert it can list), since a
# token routes to an expert at most once.
M11_MAX_TOKENS = 320
# Block scales are raised by 2^64 before the bf16 multiply (as in M2, exact); epilogues undo 2^(126 - 64).
M11_OUT_SHIFT = 2.0 ** 62
# Kernel instances compiled, (gated, tt, nt, depth, ks): tt = 16-row tiles per task (gated: gate + up),
# nt = 8-pair MMA column tiles per pass, depth = 64-wide K chunks in flight per warp, ks = warps
# splitting a task's K. Each warp's K share (64 / ks chunks) must be a multiple of depth.
M11_INSTANCES = (
    (1, 2, 1, 8, 1), (1, 2, 1, 4, 1), (1, 2, 1, 4, 2), (1, 2, 1, 4, 4), (1, 2, 1, 2, 4),
    (1, 2, 2, 4, 1), (1, 2, 2, 4, 2), (1, 2, 2, 2, 4), (1, 2, 4, 2, 1), (1, 2, 4, 2, 2),
    (0, 1, 1, 8, 1), (0, 1, 1, 4, 2), (0, 1, 1, 4, 4), (0, 1, 2, 4, 1), (0, 1, 2, 4, 4), (0, 1, 4, 2, 1),
    (0, 2, 1, 4, 1), (0, 2, 1, 4, 2), (0, 2, 1, 4, 4), (0, 2, 1, 2, 4), (0, 2, 2, 4, 2), (0, 2, 2, 2, 4),
    (1, 2, 4, 2, 4), (0, 2, 4, 2, 1), (0, 2, 4, 2, 2),
)
# Per-call configs by token count: (fewer than this many tokens, cfg) with cfg["s1"] = GEMM1's
# dict(nt, d, ks, c) and cfg["s2"] = GEMM2's dict(tt, nt, d, ks, c); c = CTAs per SM; None means v1.
# These are E3's choices (job d0ce41e8697245ce, ks 1 and tt 1 only); 17-32 tokens stay on v1.
M11_CONFIGS = (
    (5, dict(s1=dict(nt=1, d=8, ks=1, c=2), s2=dict(tt=1, nt=1, d=8, ks=1, c=2))),
    (9, dict(s1=dict(nt=1, d=4, ks=1, c=4), s2=dict(tt=1, nt=1, d=8, ks=1, c=4))),
    (17, dict(s1=dict(nt=2, d=4, ks=1, c=4), s2=dict(tt=1, nt=2, d=4, ks=1, c=4))),
    (33, None),
)

_M11_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace m11 {

constexpr int K = 4096;
constexpr int KG = K / 64;  // 64-wide K chunks
constexpr int LOCAL = 32;   // local experts; the active-expert ballot below relies on 32
constexpr int TOP_K = 8;
constexpr int MAXP = @@MAXP@@;  // pairs per expert at most (= M11_MAX_TOKENS in Python)

// E2M1 nibbles (3 - R) and (7 - R) of w as bf16x2 lanes holding value * 2^-126 (Marlin's trick).
template <int R>
__device__ __forceinline__ uint32_t dec(uint32_t w) {
  const uint32_t x = w << (4 * R);
  return (x & 0x80008000u) | ((x >> 6) & 0x01C001C0u);
}

__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t d;
  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));
  return d;
}

// E4M3 scales of rows g (low byte) and g + 8 (next byte) -> bf16x2 (s * 2^64, s * 2^64) per row; exact.
__device__ __forceinline__ void scales(uint32_t v, uint32_t& sg, uint32_t& sg8) {
  uint32_t h2;
  const unsigned short v16 = static_cast<unsigned short>(v & 0xFFFFu);
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(v16));
  const unsigned short h_lo = static_cast<unsigned short>(h2 & 0xFFFFu);
  const unsigned short h_hi = static_cast<unsigned short>(h2 >> 16);
  float lo, hi;
  asm("cvt.f32.f16 %0, %1;" : "=f"(lo) : "h"(h_lo));
  asm("cvt.f32.f16 %0, %1;" : "=f"(hi) : "h"(h_hi));
  uint32_t b2;  // cvt.rn.bf16x2.f32 d, a, b puts a in the upper half
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(b2) : "f"(hi * 18446744073709551616.0f), "f"(lo * 18446744073709551616.0f));
  sg = __byte_perm(b2, 0u, 0x1010u);
  sg8 = __byte_perm(b2, 0u, 0x3232u);
}

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                    uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// A group of KS warps of one CTA's barrier id (1-4); bar.sync orders their shared-memory accesses.
__device__ __forceinline__ void group_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
// A task (expert, TT row tiles) goes to a group of KS warps that split K and add their partial sums
// in shared memory; an expert's pairs are taken 8 * NT at a time (a second chunk re-reads its weights).
template <int TT, int NT, int D, int KS, bool GATED>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once
  constexpr int KGW = KG / KS;    // K chunks per warp
  static_assert(KGW % D == 0, "depth must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid < LOCAL) s_cnt[tid] = 0;
  __syncthreads();
  for (int p = tid; p < n_pairs; p += blockDim.x) {
    const int e = ids[p] - local_start;
    if (e >= 0 && e < LOCAL) {
      const int r = atomicAdd(&s_cnt[e], 1);
      if (r < MAXP) s_list[e][r] = static_cast<short>(p);
    }
  }
  __syncthreads();
  if (tid < 32) {
    const bool active = s_cnt[tid] > 0;
    const unsigned m = __ballot_sync(0xffffffffu, active);
    if (active) s_active[__popc(m & ((1u << tid) - 1u))] = tid;
    if (tid == 0) s_nactive = __popc(m);
  }
  __syncthreads();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  for (int task = blockIdx.x * GROUPS + grp; task < n_tasks; task += gridDim.x * GROUPS) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int total = min(s_cnt[e], MAXP);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const uint4* wp[TT];
    const uint16_t* sp[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t base = (static_cast<size_t>(e) * rt_total + rts[i]) * KG * 32 + lane;
      wp[i] = w + base;
      sp[i] = sf + base;
    }
    const float c = cs[e];
    for (int c0 = 0; c0 < total; c0 += 8 * NT) {
      const int cnt = min(total - c0, 8 * NT);
      // B operand: this thread loads pair column g of each 8-pair tile, 16 activations per K chunk.
      const uint4* ap[NT];
      bool aok[NT];
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const int idx = n * 8 + g;
        aok[n] = idx < cnt;
        const int p = aok[n] ? s_list[e][c0 + idx] : 0;
        const int row = a_per_pair ? p : p / TOP_K;
        ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
      }
      float acc[TT][NT][4];
#pragma unroll
      for (int i = 0; i < TT; ++i)
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
          for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

      uint4 rw[D][TT];
      uint32_t rs[D][TT];
      uint4 ra[D][NT][2];
      auto load = [&](int s, int kg) {
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          rw[s][i] = __ldcs(wp[i] + kg * 32);
          rs[s][i] = __ldcs(sp[i] + kg * 32);
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (aok[n]) {
            ra[s][n][0] = __ldg(ap[n] + kg * 8);
            ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
          } else {
            ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
            ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      };
      auto compute = [&](int s) {
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          uint32_t sg, sg8;
          scales(rs[s][i], sg, sg8);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t q = word(rw[s][i], j);
            const uint32_t a0 = hmul2(dec<0>(q), sg), a1 = hmul2(dec<1>(q), sg8);
            const uint32_t a2 = hmul2(dec<2>(q), sg), a3 = hmul2(dec<3>(q), sg8);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
              if (n * 8 < cnt) {
                const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
                const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
                mma(acc[i][n], a0, a1, a2, a3, b0, b1);
              }
            }
          }
        }
      };
      const int kbeg = kpart * KGW;
#pragma unroll
      for (int s = 0; s < D; ++s) load(s, kbeg + s);
      for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
        for (int s = 0; s < D; ++s) {
          compute(s);
          if (kg0 + s + D < KGW) load(s, kbeg + kg0 + s + D);
        }
      }

      if constexpr (KS > 1) {  // warp kpart 0 of the group adds the others' partial sums
        if (kpart > 0) {
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
        }
        group_sync(grp + 1, KS * 32);
        if (kpart == 0) {
#pragma unroll
          for (int w2 = 1; w2 < KS; ++w2)
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int r = 0; r < 4; ++r) acc[i][n][r] += s_red[warp + w2][(i * NT + n) * 4 + r][lane];
        }
        group_sync(grp + 1, KS * 32);  // the next chunk may overwrite s_red
      }

      // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1.
      if (kpart == 0) {
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < cnt) {
              const size_t prow = static_cast<size_t>(s_list[e][c0 + idx]) * out_cols;
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) {
                const int r = g + 8 * rr;
                if constexpr (GATED) {
                  const float gv = acc[0][n][2 * rr + h] * c;
                  const float uv = acc[1][n][2 * rr + h] * c;
                  out[prow + 16 * tile + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
                } else {
#pragma unroll
                  for (int i = 0; i < TT; ++i)
                    out[prow + 16 * rts[i] + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * c);
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace m11

#define M11_CASE(G, TT, NT, D, KS)                                                                       \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                          \
    m11::gemm_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                      \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

#define M11_ATTR(G, TT, NT, D, KS)                                                                        \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                           \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::gemm_kernel<TT, NT, D, KS, G>);                                     \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
  }

torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                       int64_t ks, int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  @@CASES@@
  TORCH_CHECK(false, "m11_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return out;
}

torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks) {
  @@ATTRS@@
  TORCH_CHECK(false, "m11_attrs: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return torch::Tensor();
}
"""

_M11_CPP = """
torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                       int64_t ks, int64_t grid);
torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);
"""


def m11_source(instances=M11_INSTANCES):
    """The CUDA source with one dispatch case per instance."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in instances]
    return (_M11_CUDA.replace("@@CASES@@", "\n  ".join(f"M11_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\n  ".join(f"M11_ATTR({a})" for a in args))
            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))


def m11_build():
    """Compile the CUDA kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(), functions=["m11_gemm", "m11_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m11_"), verbose=False)


def m11_shuffle(packed, sf):
    """One expert's NVFP4 matrix -> M11's layout (a lossless reordering).
    packed uint8 [rows, K/2] (element 2i in the low nibble), sf float8_e4m3fn [rows, K/16] ->
    words int32 [rows/16, K/64, 32 lanes, 4] and scales int16 [rows/16, K/64, 32 lanes].
    Lane (g, t), word j holds rows g and g + 8 of the tile at chunk positions 16 t + 4 j + {0..3};
    nibble p sits where the decode's round (3 - p % 4) puts it: A registers a0..a3 of the MMA."""
    rows, half = packed.shape
    rt, kg = rows // 16, half * 2 // 64
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    c = codes.view(rt, 2, 8, kg, 4, 4, 4)  # (rt, h, g, kg, t, j, q): row 16 rt + 8 h + g, k = 64 kg + 16 t + 4 j + q
    # Nibble p of each word comes from (h, q): row g + 8 h, chunk position 16 t + 4 j + q.
    src = ((1, 2), (0, 2), (1, 0), (0, 0), (1, 3), (0, 3), (1, 1), (0, 1))
    nib = [c[:, h, :, :, :, :, q] for h, q in src]  # each [rt, g, kg, t, j]
    words = torch.stack([nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(4)], dim=-1)
    words = words.permute(0, 2, 1, 3, 4, 5).contiguous().view(torch.int32).view(rt, kg, 32, 4)
    s = sf.view(torch.uint8).view(rt, 2, 8, kg, 4)  # (rt, h, g, kg, t): row 16 rt + 8 h + g, group 4 kg + t
    scales = torch.stack((s[:, 0], s[:, 1]), dim=-1).permute(0, 2, 1, 3, 4).contiguous()
    return words, scales.view(torch.int16).view(rt, kg, 32)


def m11_bank(weights, start, count):
    """M11's weights for experts [start, start + count), shuffled one expert at a time."""
    bank = {"start": start, "count": count, "inter": weights["w13"].shape[1] // 2, "hidden": weights["w2"].shape[1],
            "c13": (weights["w13_gs"][start:start + count].float() * M11_OUT_SHIFT).contiguous(),
            "c2": (weights["w2_gs"][start:start + count].float() * M11_OUT_SHIFT).contiguous()}
    for name in ("w13", "w2"):
        parts = [m11_shuffle(weights[name][e], weights[name + "_sf"][e]) for e in range(start, start + count)]
        bank[name] = torch.stack([p[0] for p in parts])
        bank[name + "_s"] = torch.stack([p[1] for p in parts])
        bank["rt_" + name] = bank[name].shape[1]
    return bank


@triton.jit(do_not_specialize=["local_start"])
def _m11_post_norm_kernel(y_ptr, nw_ptr, tw_ptr, ids_ptr, out_ptr, local_start, eps,
                          H: tl.constexpr, LOCAL: tl.constexpr, TOP_K: tl.constexpr):
    """out[t] = sum over local slots k of tw[t, k] * rmsnorm(y[t, k] * nw), as SGLang's fused kernel
    (weight first); y is bf16 [pairs, H]."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(nw_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = t * TOP_K + k
        e = tl.load(ids_ptr + p) - local_start
        if (e >= 0) & (e < LOCAL):
            y = tl.load(y_ptr + p.to(tl.int64) * H + offs).to(tl.float32) * w
            var = tl.sum(y * y, axis=0) / H
            acc += y * tl.rsqrt(var + eps) * tl.load(tw_ptr + p)
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


def m11_gemm1(x, ids, state, s1):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair; s1 = cfg["s1"]."""
    bank, n_pairs = state["m11"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    state["m11_ext"].m11_gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, n_pairs, bank["start"],
                              bank["rt_w13"], 0, bank["inter"], 1, 2, s1["nt"], s1["d"], s1["ks"],
                              s1["c"] * state["sms"])
    return act


def m11_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]; s2 = cfg["s2"]."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    state["m11_ext"].m11_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                              bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"], s2["d"], s2["ks"],
                              s2["c"] * state["sms"])
    return y


def m11_post_norm(y, ids, topk_weights, state, tokens):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
    _m11_post_norm_kernel[(tokens,)](y, state["norm_w"], topk_weights, ids, out, bank["start"], POST_NORM_EPS,
                                     H=bank["hidden"], LOCAL=bank["count"], TOP_K=TOP_K, num_warps=8)
    return out


def m11_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg["s1"])
    y = m11_gemm2(act, ids, state, cfg["s2"])
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0])


def m11_config_for(state, tokens):
    """The M11 config for this token count, or None for v1 (M11 takes at most M11_MAX_TOKENS)."""
    if tokens > M11_MAX_TOKENS:
        return None
    for below, cfg in state["m11_configs"]:
        if tokens < below:
            return cfg
    return None


def m11_warm(state, cfg):
    """Run every kernel of cfg once before any CUDA graph capture (Triton compiles on first launch)."""
    device = state["norm_w"].device
    tokens = 8
    x = torch.zeros((tokens, state["m11"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m11"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m11_run(x, ids + state["m11"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m11_setup(state, weights):
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = m11_build()
    state["m11"] = m11_bank(weights, 0, 32)


def _submission_prepare(weights):
    state = _v1_prepare(weights)
    state["m11_configs"] = M11_CONFIGS
    if any(cfg is not None for _, cfg in M11_CONFIGS):
        m11_setup(state, weights)
        for _, cfg in M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    m11cfg = m11_config_for(state, x.shape[0])
    if (m11cfg is None or "m11" not in state
            or (cfg["local_start"], cfg["local_count"]) != (state["m11"]["start"], state["m11"]["count"])):
        return _v1_kernel_fn(data, state)
    return m11_run(x, data["topk_ids"], data["topk_weights"], state, m11cfg)


# ====================================================================== lab (common): check runs only

"""Lab helpers shared by every check-run lab file (appended after submission.py by make_lab.py).
Everything here runs on synthetic inputs inside prepare(); none of it touches the timed calls."""

import statistics
import time

LAB_FLUSH_BYTES = 512 << 20  # larger than the H200's 60 MB L2, like the judge's cold-L2 read


def log(msg):
    print(f"[lab] {msg}", flush=True)


def _error_text(exc, keep=500):
    """The end of an exception's message, where Triton puts the actual reason."""
    text = " ".join(str(exc).split())
    return text if len(text) <= keep else "... " + text[-keep:]


class Lab:
    def __init__(self, budget_s):
        self.t0 = time.monotonic()
        self.budget_s = budget_s
        self.flush = None

    def left(self):
        return self.budget_s - (time.monotonic() - self.t0)

    def section(self, name, fn, *args, min_left=5.0):
        if self.left() < min_left:
            log(f"{name}: skipped, {self.left():.0f}s of budget left")
            return
        t = time.monotonic()
        try:
            fn(self, *args)
        except Exception as exc:  # a lab failure must never fail the run
            log(f"{name}: FAILED {type(exc).__name__}: {_error_text(exc)}")
        torch.cuda.synchronize()
        log(f"{name}: {time.monotonic() - t:.1f}s")

    def cold_l2(self):
        if self.flush is None:
            self.flush = torch.empty(LAB_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        return self.flush.sum()

    def time_graph(self, fn, reps=20, cold=True):
        """Median microseconds of a CUDA graph replay of fn, each after an L2 flush if cold."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            fn()
            fn()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = []
        for _ in range(reps):
            if cold:
                self.cold_l2()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)


def synthetic_routing(bs, seed, device="cuda", local_count=32, popularity=0.03):
    """Top-8 of uniform scores plus a heavy-tailed per-expert popularity, rotated so experts
    0..local_count-1 are the busiest GPU's, like the judge's default routing."""
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    z = torch.randn((N_EXPERTS,), generator=gen, device=device)
    u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
    scores = scores + popularity * z / (-u.log()).sqrt()
    vals, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // local_count).flatten(), minlength=N_EXPERTS // local_count)
    ids = (ids - local_count * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), (vals / vals.sum(-1, keepdim=True)).float()


# ====================================================================== lab (e3c): check runs only

"""E3c: M11 at every scored batch size (1-256), configs chosen on the mean of 3 routings, on the
judge's H200, check runs only. kernel_fn is submission_m11's hybrid; this prepare() decides from its own
measurements which batch sizes go to M11 (v1 otherwise).
  build    nvcc via load_inline (time, and the error lines if it fails); weights shuffled
  attrs    registers, spill bytes, static shared memory per compiled instance
  stage1   GEMM1 (gate+up, SwiGLU) per (nt, depth, ks, CTAs per SM): error vs FP32 at bs 8, a skewed
           bs 16 and a skewed bs 128 with outlier channels (many chunks); cold-L2 graph time at bs
           1/8/16/32/128/256, each the mean of 3 routings
  stage2   GEMM2 + post-norm per (tt, nt, depth, ks, CTAs per SM), the same
  e2e      best per batch size vs FP32 (judge-like and skewed routings) and vs v1; M11 only where
           correct and >= 3% faster"""

from nvfp4 import dequantize, outlier_channels

LAB_BUDGET_S = 240.0
E3_PREPARE_LIMIT_S = 275.0  # TASK.md: prepare() gets 300 s; keep a margin
E3_BS = (1, 8, 16, 32, 128, 256)
# Each measured batch size stands for token counts below this bound; every bound - 1 fits 8 * nt
# of the nt that batch size can use (1, 1, 2, 4).
E3_CLASS = {1: 5, 8: 9, 16: 17, 32: 33, 128: 192, 256: M11_MAX_TOKENS + 1}
E3_CTAS = (2, 4, 8)
E3_ROUTINGS = 3  # timings are means over this many routings per batch size
E3_REPS = 10
E3_MAX_STAGE_ERR = 1e-2
E3_MAX_ERR, E3_MAX_ROW_ERR = 1e-2, 5e-2  # stricter than the judge's 1.5e-2 and 7.5e-2


def _case(bs, seed, skew=False, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if outliers:  # like the judge's skewed test: 1% of channels scaled 30x
        x = outlier_channels(x, generator=gen)
    if skew:  # every token on experts 0-7: bs pairs per expert
        ids = torch.arange(TOP_K, dtype=torch.int32, device="cuda").repeat(bs, 1)
        w = torch.rand((bs, TOP_K), generator=gen, device="cuda")
        w = w / w.sum(-1, keepdim=True)
    else:
        ids, w = synthetic_routing(bs, seed)
    return x, ids.contiguous(), w.contiguous()


class _Ref:
    """FP32 references from the raw NVFP4 weights, experts dequantized on demand."""

    def __init__(self, weights):
        self.w = weights
        self.cache = {}

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (dequantize(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                             dequantize(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.cache[e]

    def local_pairs(self, ids):
        flat = ids.reshape(-1).long()
        return flat, [e for e in torch.unique(flat).tolist() if e < 32]

    def act(self, x, ids):
        flat, experts = self.local_pairs(ids)
        out = torch.zeros((flat.numel(), 4096), device="cuda")
        for e in experts:
            p = (flat == e).nonzero().flatten()
            h = x[p // TOP_K].float() @ self.expert(e)[0].T
            out[p] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        return out

    def stage2(self, act, ids, tw):
        flat, experts = self.local_pairs(ids)
        nw = 1.0 + self.w["post_norm_w"].float()
        out = torch.zeros((ids.shape[0], 4096), device="cuda")
        for e in experts:
            p = (flat == e).nonzero().flatten()
            y = (act[p].float() @ self.expert(e)[1].T) * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, p // TOP_K, y * tw.reshape(-1)[p].unsqueeze(1))
        return out

    def full(self, x, ids, tw):
        return self.stage2(self.act(x, ids), ids, tw)


def _errors(out, ref):
    """(relative L2, worst row) as the judge's checks.compare measures them."""
    out, ref = out.float(), ref.float()
    diff = out - ref
    err = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    row = float((diff.norm(dim=-1) / torch.maximum(rows_ref, floor)).max())
    return (err if bool(torch.isfinite(out).all()) else float("inf")), row


def _build_error_lines(exc, keep=3500):
    """nvcc/ninja error lines from load_inline's exception, which carries the whole build log."""
    lines = [l.strip() for l in str(exc).splitlines()]
    hits = [l for l in lines if "error" in l.lower() or "Error" in l]
    text = " | ".join(hits) if hits else " ".join(lines)
    return text[:keep]


def sec_build(lab, weights, state):
    """prepare() of submission_m11 already compiled and shuffled when its M11_CONFIGS is set."""
    if "m11_ext" not in state:
        t = time.monotonic()
        try:
            state["m11_ext"] = m11_build()
        except Exception as exc:
            log(f"build: nvcc FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
            raise
        log(f"build: nvcc load_inline compiled in {time.monotonic() - t:.1f}s")
    if "m11" not in state:
        state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
        state["m11"] = m11_bank(weights, 0, 32)
        torch.cuda.synchronize()
    log(f"build: {len(M11_INSTANCES)} instances, {state['sms']} SMs")


def _inst_name(gated, tt, nt, d, ks):
    return f"{'g' if gated else 'p'}{tt}.{nt}.{d}.{ks}"


def sec_attrs(lab, state):
    parts = []
    for inst in M11_INSTANCES:
        regs, local, smem = state["m11_ext"].m11_attrs(*inst).tolist()
        parts.append(f"{_inst_name(*inst)} {regs}r{f' {local}B-spill' if local else ''} {smem // 1024}K")
    log("attrs (gated/plain tt.nt.depth.ks): " + ", ".join(parts))


def _name(cfg, stage):
    tt = f"t{cfg['tt']} " if stage == 2 else ""
    return f"{tt}nt{cfg['nt']} d{cfg['d']} k{cfg['ks']} c{cfg['c']}"


def sec_stage(lab, stage, state, ref, store):
    check_cases = [_case(8, 908), _case(16, 916, skew=True), _case(128, 928, skew=True, outliers=True)]
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(E3_ROUTINGS)] for bs in E3_BS}
    if stage == 1:
        checks = [(x, ids, w, ref.act(x, ids)) for x, ids, w in check_cases]
    else:
        checks = []
        for x, ids, w in check_cases:
            act = ref.act(x, ids).to(torch.bfloat16)
            checks.append((act, ids, w, ref.stage2(act, ids, w)))
        acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in E3_BS}
    results = []
    insts = [i for i in M11_INSTANCES if i[0] == (1 if stage == 1 else 0)]
    for gated, tt, nt, d, ks in insts:
        for ctas in E3_CTAS:
            if lab.left() < (130 if stage == 1 else 60):
                log(f"s{stage}: out of budget")
                break
            cfg = dict(nt=nt, d=d, ks=ks, c=ctas) if stage == 1 else dict(tt=tt, nt=nt, d=d, ks=ks, c=ctas)
            try:
                errs = []
                for a, ids, w, want in checks:
                    flat = ids.reshape(-1)
                    if stage == 1:
                        rows = (flat.long() < 32).nonzero().flatten()
                        errs.append(_errors(m11_gemm1(a, flat, state, cfg)[rows], want[rows])[0])
                    else:
                        errs.append(_errors(m11_post_norm(m11_gemm2(a, flat, state, cfg), flat, w, state, ids.shape[0]),
                                            want)[0])
                err = max(errs)
                times = {}
                for bs, routings in cases.items():
                    ts = []
                    for x, ids, w in routings:
                        if stage == 1:
                            fn = lambda x=x, i=ids: m11_gemm1(x, i.reshape(-1), state, cfg)
                        else:
                            fn = lambda a=acts[bs], i=ids, w=w: m11_post_norm(
                                m11_gemm2(a, i.reshape(-1), state, cfg), i.reshape(-1), w, state, i.shape[0])
                        ts.append(lab.time_graph(fn, reps=E3_REPS))
                    times[bs] = sum(ts) / len(ts)
                torch.cuda.synchronize()
            except Exception as exc:  # report and go on; a CUDA fault would also show in every later line
                log(f"s{stage} {_name(cfg, stage)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
                continue
            results.append((cfg, err, times))
            log(f"s{stage} {_name(cfg, stage)}: err {err:.1e} | " + " ".join(f"{t:.1f}" for t in times.values()))
    store[stage] = [r for r in results if r[1] < E3_MAX_STAGE_ERR]
    for bs in E3_BS:
        if store[stage]:
            best = min(store[stage], key=lambda r: r[2][bs])
            log(f"s{stage} best at bs{bs}: {_name(best[0], stage)} {best[2][bs]:.1f} us")


def sec_e2e(lab, state, ref, store):
    s1, s2 = store.get(1, []), store.get(2, [])
    if not s1 or not s2:
        log("e2e: no working stage configs; kernel_fn stays on v1")
        return
    cases = [(bs, False, False) for bs in (1, 2, 4, 5, 8, 12, 16, 24, 32, 33, 64, 128, 160, 192, 256, 257, 320)]
    cases += [(bs, True, False) for bs in (8, 16, 32, 128, 256)] + [(128, True, True), (257, False, True)]
    chosen = {}
    for bs in E3_BS:
        lo = max([E3_CLASS[b] for b in E3_BS if b < bs], default=1)
        hi = E3_CLASS[bs]
        r1 = min(s1, key=lambda r: r[2][bs])
        r2 = min(s2, key=lambda r: r[2][bs])
        cfg = dict(s1=r1[0], s2=r2[0])
        worst = (0.0, 0.0)
        for cbs, skew, outl in cases:
            if not lo <= cbs < hi:
                continue
            x, ids, w = _case(cbs, 1300 + cbs + 50 * skew + 25 * outl, skew, outl)
            err, row = _errors(m11_run(x, ids, w, state, cfg), ref.full(x, ids, w))
            worst = (max(worst[0], err), max(worst[1], row))
        t_v1 = t_m11 = 0.0
        for r in range(E3_ROUTINGS):
            x, ids, w = _case(bs, 1700 + bs + 2000 * r)
            data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                    "config": {"bs": bs, "local_start": 0, "local_count": 32}}
            t_v1 += lab.time_graph(lambda: _v1_kernel_fn(data, state)) / E3_ROUTINGS
            t_m11 += lab.time_graph(lambda: m11_run(x, ids, w, state, cfg)) / E3_ROUTINGS
        use = worst[0] < E3_MAX_ERR and worst[1] < E3_MAX_ROW_ERR and t_m11 < 0.97 * t_v1
        if use:
            chosen[bs] = cfg
        log(f"e2e bs{bs}: v1 {t_v1:.1f} us, M11 {t_m11:.1f} us ({t_v1 / t_m11:.2f}x) err {worst[0]:.1e} "
            f"row {worst[1]:.1e} -> {'M11' if use else 'v1'} | {cfg}")
    state["m11_configs"] = tuple((E3_CLASS[bs], chosen.get(bs)) for bs in E3_BS)
    for cfg in chosen.values():
        m11_warm(state, cfg)
    log(f"M11_CONFIGS = {state['m11_configs']!r}")


def prepare(weights):
    t = time.monotonic()
    try:
        state = _submission_prepare(weights)
    except Exception as exc:  # nvcc runs inside the submission's prepare when M11_CONFIGS is set
        log(f"prepare FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    took = time.monotonic() - t
    log(f"E3c: submission prepare (nvcc, shuffle, warm-up) took {took:.1f}s")
    budget = min(LAB_BUDGET_S, E3_PREPARE_LIMIT_S - took)  # prepare() as a whole must stay under 300 s
    lab = Lab(budget)
    log(f"E3c start, budget {budget:.0f}s; stage lines: time at bs " + "/".join(map(str, E3_BS)) + " in us")
    ref = _Ref(weights)
    store = {}
    lab.section("build", sec_build, weights, state, min_left=120)
    if "m11_ext" in state:
        lab.section("attrs", sec_attrs, state)
        lab.section("stage1", sec_stage, 1, state, ref, store, min_left=60)
        lab.section("stage2", sec_stage, 2, state, ref, store, min_left=40)
        lab.section("e2e", sec_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m11_configs')!r}"[:700])
    return state
