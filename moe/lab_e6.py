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
# M12 instances (TMA weight ring), (gated, tt, nt, ks, ch, r): ch = 64-wide K chunks per ring slot,
# r = slots per warp; each warp's K share (64 / ks chunks) must fill an even number of slots.
# Dynamic shared memory per CTA = 4 * r * tt * ch * 576 bytes.
M12_INSTANCES = (
    (1, 2, 1, 1, 4, 4), (1, 2, 1, 1, 4, 2), (1, 2, 1, 1, 8, 2), (1, 2, 1, 2, 4, 4), (1, 2, 1, 4, 4, 4),
    (1, 2, 1, 4, 2, 4), (1, 2, 2, 1, 4, 2), (1, 2, 2, 4, 4, 2), (1, 2, 4, 1, 2, 2),
    (0, 1, 1, 1, 4, 4), (0, 2, 1, 1, 4, 4), (0, 2, 1, 1, 4, 2), (0, 2, 1, 2, 4, 4), (0, 2, 1, 4, 4, 4),
    (0, 1, 1, 4, 4, 4), (0, 2, 2, 1, 4, 2), (0, 2, 2, 4, 4, 2), (0, 2, 4, 1, 2, 2),
)
# Per-call configs by token count: (fewer than this many tokens, cfg) with cfg["s1"] = GEMM1's
# dict(nt, d, ks, c) and cfg["s2"] = GEMM2's dict(tt, nt, d, ks, c); c = CTAs per SM; None means v1.
# A stage dict with kind="m12" runs the TMA kernel instead: s1 = dict(kind, nt, ks, ch, r, c),
# s2 = dict(kind, tt, nt, ks, ch, r, c).
# These are E3c's choices (job 8ce4aba7fd53499c, means over 3 routings); 33+ tokens stay on v1, where
# M11 only tied Marlin.
M11_CONFIGS = (
    (5, dict(s1=dict(nt=1, d=4, ks=4, c=8), s2=dict(tt=1, nt=1, d=4, ks=4, c=4))),
    (9, dict(s1=dict(nt=1, d=4, ks=2, c=8), s2=dict(tt=2, nt=1, d=4, ks=4, c=8))),
    (17, dict(s1=dict(nt=1, d=4, ks=2, c=8), s2=dict(tt=2, nt=1, d=4, ks=4, c=8))),
    (33, dict(s1=dict(nt=2, d=2, ks=4, c=8), s2=dict(tt=2, nt=2, d=2, ks=4, c=4))),
    (192, None),
    (321, None),
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


// ------------------------------------------------------------------------------------------------
// M12: M11's work split and arithmetic, but each warp's weight and scale stream is copied by TMA
// (cp.async.bulk) into its own ring of R shared-memory slots of CH K chunks, which lane 0 refills
// as soon as the warp has read a slot; activations keep a two-step register prefetch. PTX per the
// ISA 9.4: mbarrier.init/arrive.expect_tx/try_wait.parity, fence.mbarrier_init, fence.proxy.async.
__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

// Makes the inits visible to the async proxy (the TMA unit) before any copy signals them.
__device__ __forceinline__ void mbar_init_fence() { asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory"); }

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}"
               ::"r"(bar), "r"(bytes) : "memory");
}

__device__ __forceinline__ bool mbar_try_wait(uint32_t bar, uint32_t parity) {
  uint32_t done;
  asm volatile("{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n selp.u32 %0, 1, 0, p;\n}"
               : "=r"(done) : "r"(bar), "r"(parity) : "memory");
  return done != 0;
}

// A wait that can never finish (bytes expected vs copied, or a slot never refilled) traps within
// seconds instead of hanging the run until its 15-minute limit.
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  for (uint32_t spins = 0; !mbar_try_wait(bar, parity); ++spins)
    if (spins > (1u << 24)) __trap();
}

__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Orders this warp's prior generic-proxy reads of a slot before the TMA write that refills it.
__device__ __forceinline__ void fence_proxy_async() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }

template <int TT, int NT, int KS, int CH, int R, bool GATED>
__global__ void __launch_bounds__(128) tma_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  constexpr int STEPS = KGW / CH;
  constexpr int TILE_BYTES = CH * 576;  // CH chunks of 512 B of codes, then CH x 64 B of scales
  constexpr int SLOT_BYTES = TT * TILE_BYTES;
  static_assert(KGW % CH == 0 && STEPS % 2 == 0, "each warp's K chunks must fill an even number of slots");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  extern __shared__ __align__(128) unsigned char ring[];  // [4 warps][R slots][SLOT_BYTES]
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid < LOCAL) s_cnt[tid] = 0;
  if ((tid & 31) == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[tid >> 5][r]), 1);
    mbar_init_fence();
  }
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
  const int first = blockIdx.x * GROUPS + grp;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;
  const unsigned char* myring = ring + warp * (R * SLOT_BYTES);
  const uint32_t ring_s = smem_addr(myring);
  const uint32_t bar_s = smem_addr(&bars[warp][0]);  // slot r's barrier at bar_s + 8 r

  // The refill cursor walks the same (task, pair chunk, step) sequence as the loop below, R slots ahead.
  int p_task = first, p_c0 = 0, p_step = 0;
  auto issue = [&](int slot) {
    if (p_task >= n_tasks) return;
    const int e = s_active[p_task / per_expert];
    const int tile = p_task % per_expert;
    if (lane == 0) {
      const uint32_t bar = bar_s + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT_BYTES);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
        const size_t chunk = (static_cast<size_t>(e) * rt_total + rt) * KG + kbeg + p_step * CH;
        const uint32_t dst = ring_s + slot * SLOT_BYTES + i * TILE_BYTES;
        bulk_g2s(dst, w + chunk * 32, CH * 512, bar);
        bulk_g2s(dst + CH * 512, sf + chunk * 32, CH * 64, bar);
      }
    }
    if (++p_step == STEPS) {
      p_step = 0;
      p_c0 += 8 * NT;
      if (p_c0 >= min(s_cnt[e], MAXP)) {
        p_c0 = 0;
        p_task += stride;
      }
    }
  };
#pragma unroll
  for (int r = 0; r < R; ++r) issue(r);

  int seq = 0;  // slots consumed by this warp so far
  for (int task = first; task < n_tasks; task += stride) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int total = min(s_cnt[e], MAXP);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    for (int c0 = 0; c0 < total; c0 += 8 * NT) {
      const int cnt = min(total - c0, 8 * NT);
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

      uint4 ra[2][CH][NT][2];
      auto load_act = [&](int b, int st) {
#pragma unroll
        for (int jj = 0; jj < CH; ++jj) {
          const int kg = kbeg + st * CH + jj;
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            if (aok[n]) {
              ra[b][jj][n][0] = __ldg(ap[n] + kg * 8);
              ra[b][jj][n][1] = __ldg(ap[n] + kg * 8 + 1);
            } else {
              ra[b][jj][n][0] = make_uint4(0u, 0u, 0u, 0u);
              ra[b][jj][n][1] = make_uint4(0u, 0u, 0u, 0u);
            }
          }
        }
      };
      auto step = [&](int b) {
        const int slot = seq % R;
        mbar_wait(bar_s + 8 * slot, (seq / R) & 1);
        const unsigned char* sp = myring + slot * SLOT_BYTES;
#pragma unroll
        for (int jj = 0; jj < CH; ++jj) {
#pragma unroll
          for (int i = 0; i < TT; ++i) {
            const unsigned char* tp = sp + i * TILE_BYTES;
            const uint4 q4 = *reinterpret_cast<const uint4*>(tp + jj * 512 + lane * 16);
            const uint32_t sv = *reinterpret_cast<const uint16_t*>(tp + CH * 512 + jj * 64 + lane * 2);
            uint32_t sg, sg8;
            scales(sv, sg, sg8);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
              const uint32_t q = word(q4, j);
              const uint32_t a0 = hmul2(dec<0>(q), sg), a1 = hmul2(dec<1>(q), sg8);
              const uint32_t a2 = hmul2(dec<2>(q), sg), a3 = hmul2(dec<3>(q), sg8);
#pragma unroll
              for (int n = 0; n < NT; ++n) {
                if (n * 8 < cnt) {
                  const uint32_t b0 = word(ra[b][jj][n][j >> 1], (2 * j) & 3);
                  const uint32_t b1 = word(ra[b][jj][n][j >> 1], (2 * j + 1) & 3);
                  mma(acc[i][n], a0, a1, a2, a3, b0, b1);
                }
              }
            }
          }
        }
        __syncwarp();
        if (lane == 0) fence_proxy_async();
        issue(slot);
        ++seq;
      };
      load_act(0, 0);
      for (int st = 0; st < STEPS; st += 2) {
        load_act(1, st + 1);
        step(0);
        if (st + 2 < STEPS) load_act(0, st + 2);
        step(1);
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
        group_sync(grp + 1, KS * 32);
      }

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

#define M12_CASE(G, TT, NT, KS, CH, R)                                                                   \
  if (gated == (G) && tt == (TT) && nt == (NT) && ks == (KS) && ch == (CH) && r == (R)) {                \
    m11::tma_kernel<TT, NT, KS, CH, R, G><<<static_cast<int>(grid), 128, 4 * R * TT * CH * 576, stream>>>( \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

// Raises each instance's dynamic shared-memory limit (needed above 48 KB) and reports its resources:
// rows of [gated, tt, nt, ks, ch, r, registers, local (spill) bytes, static smem, dynamic smem].
#define M12_SETUP(G, TT, NT, KS, CH, R)                                                                  \
  {                                                                                                      \
    const int dyn = 4 * R * TT * CH * 576;                                                               \
    cudaFuncSetAttribute(m11::tma_kernel<TT, NT, KS, CH, R, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                             \
    cudaFuncGetAttributes(&attr, m11::tma_kernel<TT, NT, KS, CH, R, G>);                                 \
    const int64_t row[10] = {(G) ? 1 : 0, TT, NT, KS, CH, R, static_cast<int64_t>(attr.numRegs),        \
                             static_cast<int64_t>(attr.localSizeBytes),                                  \
                             static_cast<int64_t>(attr.sharedSizeBytes), dyn};                          \
    for (int64_t v : row) rows.push_back(v);                                                             \
  }

torch::Tensor m12_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t ks,
                       int64_t ch, int64_t r, int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  @@M12_CASES@@
  TORCH_CHECK(false, "m12_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " ks=", ks, " ch=", ch,
              " r=", r);
  return out;
}

torch::Tensor m12_setup() {
  std::vector<int64_t> rows;
  @@M12_SETUPS@@
  return torch::tensor(rows).view({-1, 10});
}
"""

_M11_CPP = """
torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                       int64_t ks, int64_t grid);
torch::Tensor m11_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);
torch::Tensor m12_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t ks,
                       int64_t ch, int64_t r, int64_t grid);
torch::Tensor m12_setup();
"""


def m11_source(instances=M11_INSTANCES, m12_instances=M12_INSTANCES):
    """The CUDA source with one dispatch case per instance."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in instances]
    args12 = [f"{'true' if g else 'false'}, {tt}, {nt}, {ks}, {ch}, {r}" for g, tt, nt, ks, ch, r in m12_instances]
    return (_M11_CUDA.replace("@@CASES@@", "\n  ".join(f"M11_CASE({a})" for a in args))
            .replace("@@ATTRS@@", "\n  ".join(f"M11_ATTR({a})" for a in args))
            .replace("@@M12_CASES@@", "\n  ".join(f"M12_CASE({a})" for a in args12))
            .replace("@@M12_SETUPS@@", "\n  ".join(f"M12_SETUP({a})" for a in args12))
            .replace("@@MAXP@@", str(M11_MAX_TOKENS)))


def m11_build():
    """Compile the CUDA kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(), functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],
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
    args = (bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, n_pairs, bank["start"], bank["rt_w13"], 0,
            bank["inter"], 1, 2, s1["nt"])
    if s1.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s1["ks"], s1["ch"], s1["r"], s1["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s1["d"], s1["ks"], s1["c"] * state["sms"])
    return act


def m11_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]; s2 = cfg["s2"]."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    args = (bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"], bank["rt_w2"], 1,
            bank["hidden"], 0, s2["tt"], s2["nt"])
    if s2.get("kind") == "m12":
        state["m11_ext"].m12_gemm(*args, s2["ks"], s2["ch"], s2["r"], s2["c"] * state["sms"])
    else:
        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"])
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
    # Dynamic shared-memory limits for the M12 instances, set once here and never during a capture.
    state["m12_resources"] = state["m11_ext"].m12_setup().tolist()
    state["m11"] = m11_bank(weights, 0, 32)


def _m11_prepare(weights):
    state = _v1_prepare(weights)
    state["m11_configs"] = M11_CONFIGS
    if any(cfg is not None for _, cfg in M11_CONFIGS):
        m11_setup(state, weights)
        for _, cfg in M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
    return state


def _m11_kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    m11cfg = m11_config_for(state, x.shape[0])
    if (m11cfg is None or "m11" not in state
            or (cfg["local_start"], cfg["local_count"]) != (state["m11"]["start"], state["m11"]["count"])):
        return _v1_kernel_fn(data, state)
    return m11_run(x, data["topk_ids"], data["topk_weights"], state, m11cfg)


# ====================================================================== M13 (moe/m13/m13.py)

"""M13: stream-K expert GEMMs for small batches (appended after the M11 section, whose prepare and
kernel_fn become _m11_prepare and _m11_kernel_fn).

What changes against M11 (MF7: M11 streamed only ~2 TB/s at bs1):
- Even split of the work over every resident warp (stream-K): the (active expert, task, pair pass,
  K group) list is cut into equal contiguous ranges, one per warp, whatever the routing; no waves.
- One register pipeline per warp that runs straight across task boundaries (the next group's loads are
  in flight while the current one is computed), instead of restarting at every task.
- A task whose K is split between warps adds its partial sums into an FP32 scratch with atomics; the
  warp that completes its K (a per-task counter of K groups) applies the epilogue from the scratch and
  clears scratch and counter, so the buffers are clean again at the end of every call.
- Pair lists built in pair order (deterministic), so every CTA agrees on which pairs a pass holds.
- A lossless bit re-layout of each code word: the 4 bf16x2 A registers are rotr(w, 2|6|12) & M and
  (rotr(w, 8) & M') | (w & M''), 9 instructions instead of ~15 for Marlin's shifts.
- E4M3 scales (all normal and positive, checked in prepare) become bf16x2 (s * 2^64, s * 2^64) with one
  PRMT and one IMAD each.
- GEMM2 and a CUDA post-norm are launched with programmatic dependent launch (PDL): they start as
  GEMM1/GEMM2 CTAs retire, scan the routing and load their first weights before griddepcontrol.wait.
"""

# Tokens M13 takes at most (pairs per expert <= tokens, so MAXP below needs no cap at this size).
M13_MAX_TOKENS = 64
# (gated, tt, nt) instances: tt = 16-row tiles per task (gated: gate + up), nt = 8-pair column tiles.
M13_INSTANCES = ((1, 2, 1), (1, 2, 2), (0, 1, 1), (0, 2, 1), (0, 1, 2), (0, 2, 2))
# M14 (CTA-shared activations, stream-K over CTAs) takes up to M14_MAX_TOKENS (= its MAXP); (gated, tt, nt).
M14_MAX_TOKENS = 320
M14_INSTANCES = ((1, 2, 2), (1, 2, 4), (1, 2, 8), (0, 1, 4), (0, 1, 8), (0, 2, 2), (0, 2, 4), (0, 2, 8))
# The M11 instances M11_CONFIGS uses (E3c/E4 picks); compiled next to M13 as the fallback.
M13_M11_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (1, 2, 2, 2, 4), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4), (0, 2, 2, 2, 4))
# Per-call configs by token count, like M11_CONFIGS; a stage dict with kind="m13" runs the stream-K
# kernel (s1 = dict(kind, nt, c), s2 = dict(kind, tt, nt, c, pdl, pf)), kind="m14" the CTA-shared one
# (same keys); post = dict(kind="m13", pdl) or None for the Triton post-norm. None for a class means the
# M11/v1 path (_m11_kernel_fn). kind="m13" stages only in classes up to M13_MAX_TOKENS tokens.
M13_CONFIGS = (
    (5, None),
    (9, None),
    (17, None),
    (33, None),
    (65, None),
    (192, None),
    (321, None),
)

# New bit n of an M13 code word = bit M13_BIT_SRC[n] of the M11 word (built below).
_M13_TARGETS = (6, 7, 8, 15, 22, 23, 24, 31)  # bf16x2 sign + 3 magnitude bits per lane


def _m13_bit_sources():
    # M11's dec<R>: target bit p of A register R comes from word bit p - 4R (p = 15, 31) or
    # p + 6 - 4R (p = 6..8, 22..24).
    src = [None] * 32
    for r in range(4):
        for p in _M13_TARGETS:
            old = (p if p in (15, 31) else p + 6) - 4 * r
            if r < 3:
                new = (p + (2, 6, 12)[r]) % 32
            else:
                new = p if p in (6, 22) else (p + 8) % 32
            assert src[new] is None
            src[new] = old
    assert sorted(src) == list(range(32))
    return tuple(src)


M13_BIT_SRC = _m13_bit_sources()

_M13_CUDA = r"""
namespace m13 {
using m11::hmul2;
using m11::mma;
using m11::word;

constexpr int K = 4096;
constexpr int KG = K / 64;   // 64-wide K chunks
constexpr int D = 4;         // chunks per group = register ring depth; ranges are cut at groups
constexpr int GPT = KG / D;  // groups per (task, pass)
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M13_MAXP@@;          // pairs per expert at most
constexpr int MAXPASS = (MAXP + 7) / 8;     // pair passes per task at most (nt >= 1)
constexpr int MAXPAIRS = @@M13_MAXPAIRS@@;  // n_pairs at most
constexpr uint32_t PM = 0x81C081C0u;    // sign + magnitude bits of both bf16 lanes
constexpr uint32_t PM3 = 0x81808180u;   // a3: these bits from rotr(w, 8) ...
constexpr uint32_t PM3W = 0x00400040u;  // ... and these straight from w

__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }

// Byte B of v (an E4M3 scale, normal and positive) -> bf16x2 (s * 2^64, s * 2^64): bf16 bits are
// (b << 4) + (184 << 7) for b = E << 3 | M; exact.
template <int B>
__device__ __forceinline__ uint32_t scale2(uint32_t v) {
  return __byte_perm(v, 0u, 0x4440u | B) * 0x00100010u + 0x5C005C00u;
}

__device__ __forceinline__ void pdl_wait() { asm volatile("griddepcontrol.wait;" ::: "memory"); }
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ void l2_prefetch(const void* p, uint32_t bytes) {
  asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes) : "memory");
}

// A position in a warp's range: active expert ai (expert e), task, pair pass, K group; passes and
// the valid pairs of this pass (cntp) follow from e.
struct Cur {
  int ai, e, task, pass, grp, passes, cntp;
};

// w: [LOCAL][rt_total][KG][32 lanes] uint4 (M13 bit layout); sf: [LOCAL][rt_total][GPT][32] uint2 =
// 4 chunks of (row g, row g + 8) E4M3 scales; scr: FP32 [pairs][16 rt_total], zero between calls;
// ctr: [LOCAL][per_expert][MAXPASS] K groups done, zero between calls.
template <int TT, int NT, bool GATED>
__global__ void __launch_bounds__(128) sk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;  // pairs per pass
  __shared__ int s_e[MAXPAIRS];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_base[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid == 0) pdl_launch();
  if (tid < LOCAL) s_cnt[tid] = 0;
  for (int p = tid; p < n_pairs; p += blockDim.x) s_e[p] = ids[p] - local_start;
  __syncthreads();
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  if (warp == 0) {
    // Pair lists in increasing pair order: 32 pairs at a time, ranks from __match_any_sync.
    for (int base = 0; base < n_pairs; base += 32) {
      const int p = base + lane;
      const int e = p < n_pairs ? s_e[p] : -1;
      const bool valid = e >= 0 && e < LOCAL;
      const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
      const unsigned below = m & ((1u << lane) - 1u);
      const int cur = valid ? s_cnt[e] : 0;
      if (valid) {
        const int pos = cur + __popc(below);
        if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
      }
      __syncwarp();
      if (valid && below == 0u) s_cnt[e] = cur + __popc(m);
      __syncwarp();
    }
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int v = act ? per_expert * ((cnt + NB - 1) / NB) * GPT : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (act) {
      const int ai = __popc(am & ((1u << lane) - 1u));
      s_active[ai] = lane;
      s_base[ai] = incl - v;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_base[__popc(am)] = incl;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total = s_base[nact];
  const int gw = blockIdx.x * 4 + warp, nw = gridDim.x * 4;
  const int g0 = static_cast<int>(static_cast<long long>(total) * gw / nw);
  const int g1 = static_cast<int>(static_cast<long long>(total) * (gw + 1) / nw);
  if (g0 >= g1) {  // no work; still wait, so this grid never completes before the previous one
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.pass == c.passes) {
        c.pass = 0;
        if (++c.task == per_expert) {
          c.task = 0;
          if (++c.ai < nact) set_expert(c);
        }
      }
      if (c.ai < nact) set_cntp(c);
    }
  };
  Cur L;  // the group whose loads are issued next
  {
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_base[ai], per_task = L.passes * GPT;
    L.task = local / per_task;
    const int rem = local - L.task * per_task;
    L.pass = rem / GPT;
    L.grp = rem - L.pass * GPT;
    set_cntp(L);
  }

  auto tile_row = [&](const Cur& c, int i) -> size_t {
    const int rt = GATED ? c.task + i * (rt_total / 2) : c.task * TT + i;
    return static_cast<size_t>(c.e) * rt_total + rt;
  };
  const uint4* lw[TT];
  const uint2* ls[TT];
  const uint4* la[NT];
  bool lok[NT];
  auto point = [&](const Cur& c) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(c, i);
      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;
      ls[i] = sf + (row * GPT + c.grp) * 32 + lane;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      lok[n] = idx < c.cntp;
      const int p = lok[n] ? s_list[c.e][c.pass * NB + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      la[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t) + c.grp * D * 8;
    }
  };

  uint4 rw[D][TT];
  uint4 ra[D][NT][2];
  uint2 rsn[TT];
  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) rw[s][i] = __ldcs(lw[i] + s * 32);
  };
  auto load_s = [&]() {
#pragma unroll
    for (int i = 0; i < TT; ++i) rsn[i] = __ldcs(ls[i]);
  };
  auto load_a = [&](int s) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (lok[n]) {
        ra[s][n][0] = __ldg(la[n] + s * 8);
        ra[s][n][1] = __ldg(la[n] + s * 8 + 1);
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        const uint32_t a0 = hmul2(rotr(q, 2) & PM, sg);
        const uint32_t a1 = hmul2(rotr(q, 6) & PM, sg8);
        const uint32_t a2 = hmul2(rotr(q, 12) & PM, sg);
        const uint32_t a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (n * 8 < cntp) {
            const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
            mma(acc[i][n], a0, a1, a2, a3, b0, b1);
          }
        }
      }
    }
  };
  // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1 of each 8-pair tile.
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * c.task + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * (c.task * TT + i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  // End of this warp's part of (task, pass): the whole K -> epilogue; else add into the scratch and
  // count K groups; the warp that brings the count to GPT takes the sums, clears them, and finishes.
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i) {
              const int col = 16 * static_cast<int>(tile_row(c, i) - static_cast<size_t>(c.e) * rt_total);
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + col + g + 8 * rr, acc[i][n][2 * rr + h]);
            }
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + c.task) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i) {
                const int col = 16 * static_cast<int>(tile_row(c, i) - static_cast<size_t>(c.e) * rt_total);
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* q = prow + col + g + 8 * rr;
                  acc[i][n][2 * rr + h] = __ldcg(q);
                  __stcg(q, 0.0f);
                }
              }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  load_s();
  if (pdl) {
    // Weights don't depend on the previous grid: ask L2 for the next pf groups before waiting.
    if (lane == 0 && pf > 0) {
      Cur q = L;
      for (int k = 1; k <= pf && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
#pragma unroll
  for (int s = 0; s < D; ++s) load_a(s);

  Cur C = L;
  int seg0 = C.grp;
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      compute(s, rsc, cntp);
      if (more) {
        load_w(s);
        load_a(s);
        if (s == 0) load_s();
      }
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

// out[t] = sum over local slots k of tw[t, k] * rmsnorm(y[t, k] * nw), as SGLang's fused kernel; y and nw
// bf16, H = 4096 = 512 threads x 8.
__global__ void __launch_bounds__(512) post_norm_kernel(
    const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw,
    const int* __restrict__ ids, __nv_bfloat16* __restrict__ out, int local_start, float eps, int pdl) {
  __shared__ float s_part[16][TOP_K];
  __shared__ float s_rs[TOP_K];
  const int tok = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + tid);
  float wv[8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t u = word(wraw, j);
    wv[2 * j] = __uint_as_float(u << 16);
    wv[2 * j + 1] = __uint_as_float(u & 0xFFFF0000u);
  }
  bool loc[TOP_K];
  float gk[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tok * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    gk[k] = tw[tok * TOP_K + k];
  }
  if (pdl) pdl_wait();
  float v[TOP_K][8];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    ss[k] = 0.0f;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (loc[k]) r = __ldg(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tok * TOP_K + k) * K) + tid);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t u = word(r, j);
      v[k][2 * j] = __uint_as_float(u << 16) * wv[2 * j];
      v[k][2 * j + 1] = __uint_as_float(u & 0xFFFF0000u) * wv[2 * j + 1];
      ss[k] += v[k][2 * j] * v[k][2 * j] + v[k][2 * j + 1] * v[k][2 * j + 1];
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0) {
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_part[warp][k] = ss[k];
  }
  __syncthreads();
  if (tid < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) s += s_part[i][tid];
    s_rs[tid] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  __syncthreads();
  float o[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    if (loc[k]) {
      const float f = gk[k] * s_rs[k];
#pragma unroll
      for (int j = 0; j < 8; ++j) o[j] += f * v[k][j];
    }
  }
  uint4 res;
  uint32_t pk[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
    const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
    pk[j] = lo | (hi << 16);
  }
  res = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  reinterpret_cast<uint4*>(out + static_cast<size_t>(tok) * K)[tid] = res;
}

}  // namespace m13

// ------------------------------------------------------------------------------------------------
// M14: M13 for large batches. A CTA's 4 warps work on the same (expert, block of 4 tasks, pair pass,
// K group) at once, so each 64-wide K chunk of the pass's activations (up to 8 NT pairs) is staged
// once in shared memory with cp.async (3 stages, 16-B segments XOR-swizzled by pair so the B-fragment
// loads are conflict-free) and shared by the 4 warps; weights per warp as in M13 (register ring, M13
// bit layout). Stream-K over CTAs; partial K through M13's scratch and counters.
namespace m14 {
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::l2_prefetch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;       // pairs per expert at most
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes
constexpr int SB = 3;                    // activation stages (K chunks) in shared memory

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

template <int TT, int NT, bool GATED>
__global__ void __launch_bounds__(128) ck_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  extern __shared__ __align__(128) unsigned char s_b[];  // [SB][NB pairs][128 B]
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_base[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  __syncthreads();
  // Pair lists in pair order with 4 warps: per-warp-block counts, offsets per expert, then placement.
  const int q = ((n_pairs + 127) / 128) * 32;
  const int pb0 = warp * q, pb1 = min(n_pairs, pb0 + q);
  const unsigned lt = (1u << lane) - 1u;
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] += __popc(m);
    __syncwarp();
  }
  __syncthreads();
  if (tid < LOCAL) {
    int run = 0;
#pragma unroll
    for (int v = 0; v < 4; ++v) {
      const int c = s_wcnt[v][tid];
      s_wcnt[v][tid] = run;
      run += c;
    }
    s_cnt[tid] = run;
  }
  __syncthreads();
  for (int base = pb0; base < pb1; base += 32) {
    const int p = base + lane;
    const int e = p < pb1 ? ids[p] - local_start : -1;
    const bool valid = e >= 0 && e < LOCAL;
    const unsigned m = __match_any_sync(0xffffffffu, valid ? e : -1);
    const int cur = valid ? s_wcnt[warp][e] : 0;
    if (valid) {
      const int pos = cur + __popc(m & lt);
      if (pos < MAXP) s_list[e][pos] = static_cast<short>(p);
    }
    __syncwarp();
    if (valid && (m & lt) == 0u) s_wcnt[warp][e] = cur + __popc(m);
    __syncwarp();
  }
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;  // tasks per expert
  const int n_rb = per_expert / 4;                                // 4-task blocks per expert
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int v = act ? n_rb * ((cnt + NB - 1) / NB) * GPT : 0;
    int incl = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int u = __shfl_up_sync(0xffffffffu, incl, o);
      if (lane >= o) incl += u;
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_base[ai] = incl - v;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_base[__popc(am)] = incl;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total = s_base[nact];
  const int g0 = static_cast<int>(static_cast<long long>(total) * blockIdx.x / gridDim.x);
  const int g1 = static_cast<int>(static_cast<long long>(total) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {  // the whole CTA has no work; still wait, so this grid never completes before the previous one
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // c.task is the 4-task block here
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.pass == c.passes) {
        c.pass = 0;
        if (++c.task == n_rb) {
          c.task = 0;
          if (++c.ai < nact) set_expert(c);
        }
      }
      if (c.ai < nact) set_cntp(c);
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_base[ai], per_rb = L.passes * GPT;
    L.task = local / per_rb;
    const int rem = local - L.task * per_rb;
    L.pass = rem / GPT;
    L.grp = rem - L.pass * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint4* lw[TT];
  const uint2* ls[TT];
  auto point = [&](const Cur& c) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(c, i);
      lw[i] = w + (row * KG + c.grp * D) * 32 + lane;
      ls[i] = sf + (row * GPT + c.grp) * 32 + lane;
    }
  };
  const uint32_t sb0 = smem_addr(s_b);
  // Chunk s of group c's activations -> stage slot: NB pairs x 8 segments of 16 B, invalid pairs zero-filled.
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {  // only the 8-pair tiles compute() reads
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  uint4 rw[D][TT];
  uint2 rsn[TT];
  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) rw[s][i] = __ldcs(lw[i] + s * 32);
  };
  auto load_s = [&]() {
#pragma unroll
    for (int i = 0; i < TT; ++i) rsn[i] = __ldcs(ls[i]);
  };
  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp, int slot) {
    uint32_t A[TT][4][4];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
      }
    }
    const unsigned char* sb = s_b + slot * STAGE;
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (n * 8 < cntp) {
        const unsigned char* st = sb + (n * 8 + g) * 128;
        const uint4 blo = *reinterpret_cast<const uint4*>(st + (((2 * t) ^ g) << 4));
        const uint4 bhi = *reinterpret_cast<const uint4*>(st + (((2 * t + 1) ^ g) << 4));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint4& bv = j < 2 ? blo : bhi;
          const uint32_t b0 = word(bv, (2 * j) & 3), b1 = word(bv, (2 * j + 1) & 3);
#pragma unroll
          for (int i = 0; i < TT; ++i) mma(acc[i][n], A[i][j][0], A[i][j][1], A[i][j][2], A[i][j][3], b0, b1);
        }
      }
    }
  };
  auto epilogue = [&](const Cur& c) {
    const float cc = cs[c.e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < c.cntp) {
          const size_t prow = static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * out_cols;
#pragma unroll
          for (int rr = 0; rr < 2; ++rr) {
            const int r = g + 8 * rr;
            if constexpr (GATED) {
              const float gv = acc[0][n][2 * rr + h] * cc;
              const float uv = acc[1][n][2 * rr + h] * cc;
              out[prow + 16 * wtask(c) + r] = __float2bfloat16_rn(gv / (1.0f + __expf(-gv)) * uv);
            } else {
#pragma unroll
              for (int i = 0; i < TT; ++i)
                out[prow + 16 * tile_rt(c, i) + r] = __float2bfloat16_rn(acc[i][n][2 * rr + h] * cc);
            }
          }
        }
      }
    }
  };
  auto flush = [&](const Cur& c, int seg0) {
    const int nseg = c.grp - seg0 + 1;
    bool epi = nseg == GPT;
    if (!epi) {
      const int sc_cols = 16 * rt_total;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int idx = n * 8 + 2 * t + h;
          if (idx < c.cntp) {
            float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
            for (int i = 0; i < TT; ++i)
#pragma unroll
              for (int rr = 0; rr < 2; ++rr) atomicAdd(prow + 16 * tile_rt(c, i) + g + 8 * rr, acc[i][n][2 * rr + h]);
          }
        }
      }
      __threadfence();
      __syncwarp();
      const int ci = (c.e * per_expert + wtask(c)) * MAXPASS + c.pass;
      int old = 0;
      if (lane == 0) old = atomicAdd(ctr + ci, nseg);
      old = __shfl_sync(0xffffffffu, old, 0);
      if (old + nseg == GPT) {
        __threadfence();
#pragma unroll
        for (int n = 0; n < NT; ++n) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int idx = n * 8 + 2 * t + h;
            if (idx < c.cntp) {
              float* prow = scr + static_cast<size_t>(s_list[c.e][c.pass * NB + idx]) * sc_cols;
#pragma unroll
              for (int i = 0; i < TT; ++i)
#pragma unroll
                for (int rr = 0; rr < 2; ++rr) {
                  float* q = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = __ldcg(q);
                  __stcg(q, 0.0f);
                }
            }
          }
        }
        if (lane == 0) ctr[ci] = 0;
        epi = true;
      }
    }
    if (epi) epilogue(c);
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  };

  point(L);
#pragma unroll
  for (int s = 0; s < D; ++s) load_w(s);
  load_s();
  if (pdl) {
    if (lane == 0 && pf > 0) {
      Cur q = L;
      for (int k = 1; k <= pf && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  load_b(L, 0, 0);
  cp_commit();
  load_b(L, 1, 1);
  cp_commit();

  Cur C = L;
  int seg0 = C.grp;
  int chunk = 0;  // chunks consumed; activation stage = chunk % SB
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait1();
      __syncthreads();  // chunk `chunk` is in shared memory for all; everyone is done with chunk - 1's stage
      const int ahead = (chunk + 2) % SB;
      if (s < 2) {
        load_b(C, s + 2, ahead);
      } else if (more) {
        load_b(L, s - 2, ahead);
      }
      cp_commit();
      compute(s, rsc, cntp, chunk % SB);
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      ++chunk;
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m14


// Launch with programmatic stream serialization when pdl (the kernel then may start before the
// previous kernel in the stream completes; it calls griddepcontrol.wait before reading its outputs).
template <typename... Exp, typename... Act>
static void m13_launch(void (*kernel)(Exp...), int grid, int threads, size_t smem, int pdl, Act... args) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(grid), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, args...);
  TORCH_CHECK(err == cudaSuccess, "m13 launch failed: ", cudaGetErrorString(err));
}

#define M13_CASE(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    m13_launch(m13::sk_kernel<TT, NT, G>, static_cast<int>(grid), 128, 0, static_cast<int>(pdl),          \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// [registers, spill bytes, static shared memory, CTAs per SM at 128 threads]
#define M13_ATTR(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m13::sk_kernel<TT, NT, G>);                                               \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m13::sk_kernel<TT, NT, G>, 128, 0);            \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes),  \
                          static_cast<int64_t>(attr.sharedSizeBytes), static_cast<int64_t>(blocks)});     \
  }

torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_gemm: ", n_pairs, " pairs, at most ", m13::MAXPAIRS);
  @@M13_CASES@@
  TORCH_CHECK(false, "m13_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt) {
  @@M13_ATTRS@@
  TORCH_CHECK(false, "m13_attrs: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return torch::Tensor();
}

torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl) {
  m13_launch(m13::post_norm_kernel, static_cast<int>(tokens), 512, 0, static_cast<int>(pdl),
             reinterpret_cast<const __nv_bfloat16*>(y.data_ptr()), reinterpret_cast<const __nv_bfloat16*>(nw.data_ptr()),
             static_cast<const float*>(tw.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(local_start), static_cast<float>(eps),
             static_cast<int>(pdl));
  return out;
}

#define M14_CASE(G, TT, NT)                                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT)) {                                                          \
    m13_launch(m14::ck_kernel<TT, NT, G>, static_cast<int>(grid), 128,                                     \
               static_cast<size_t>(m14::SB * m14::stage_bytes<NT>()), static_cast<int>(pdl),               \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// Raises each instance's dynamic shared-memory limit (once, in prepare) and reports
// [gated, tt, nt, registers, spill bytes, static smem, dynamic smem, CTAs per SM].
#define M14_SETUP(G, TT, NT)                                                                                \
  {                                                                                                        \
    const int dyn = m14::SB * m14::stage_bytes<NT>();                                                      \
    cudaFuncSetAttribute(m14::ck_kernel<TT, NT, G>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);    \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m14::ck_kernel<TT, NT, G>);                                               \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m14::ck_kernel<TT, NT, G>, 128, dyn);          \
    const int64_t row[8] = {(G) ? 1 : 0, TT, NT, static_cast<int64_t>(attr.numRegs),                       \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_gemm: ", n_pairs, " pairs, at most ", 8 * m14::MAXP);
  @@M14_CASES@@
  TORCH_CHECK(false, "m14_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m14_setup() {
  std::vector<int64_t> rows;
  @@M14_SETUPS@@
  return torch::tensor(rows).view({-1, 8});
}
"""

_M13_CPP = """
torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt);
torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl);
torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m14_setup();
"""


def m13_source(instances=M13_INSTANCES, m11_instances=M13_M11_INSTANCES, m14_instances=M14_INSTANCES):
    """M11's CUDA (only the instances M11_CONFIGS uses, no M12) followed by M13's and M14's."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in instances]
    args14 = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in m14_instances]
    m13 = (_M13_CUDA.replace("@@M13_CASES@@", "\n  ".join(f"M13_CASE({a})" for a in args))
           .replace("@@M13_ATTRS@@", "\n  ".join(f"M13_ATTR({a})" for a in args))
           .replace("@@M13_MAXP@@", str(M13_MAX_TOKENS))
           .replace("@@M13_MAXPAIRS@@", str(M13_MAX_TOKENS * TOP_K))
           .replace("@@M14_MAXP@@", str(M14_MAX_TOKENS))
           .replace("@@M14_CASES@@", "\n  ".join(f"M14_CASE({a})" for a in args14))
           .replace("@@M14_SETUPS@@", "\n  ".join(f"M14_SETUP({a})" for a in args14)))
    return m11_source(m11_instances, ()) + m13


def m13_build():
    """One extension with M11's and M13's kernels (sm_90a only, as m11_build)."""
    return load_inline(
        name="m13_ext", cpp_sources=_M11_CPP + _M13_CPP, cuda_sources=m13_source(),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m13_"), verbose=False)


def m13_relayout(words):
    """M11 code words (int32, any shape) -> M13's bit layout, a lossless permutation of each word's bits."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for new, old in enumerate(M13_BIT_SRC):
        out |= ((w >> old) & 1) << new
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


def m13_scales_ok(weights, start, count):
    """True when every local E4M3 block scale is positive and normal (no zero, subnormal or NaN),
    which scale2's bit trick needs; M13 stays off otherwise."""
    for name in ("w13_sf", "w2_sf"):
        b = weights[name][start:start + count].view(torch.uint8)
        bad = ((b & 0x80) != 0) | ((b & 0x78) == 0) | ((b & 0x7F) == 0x7F)
        if bool(bad.any()):
            return False
    return True


def m13_bank(m11, device):
    """M13's weights from M11's bank (same fragment order): re-laid-out words, scales grouped 4 chunks per
    lane, and the zeroed FP32 scratch and K-group counters the kernels keep clean between calls."""
    bank = {k: m11[k] for k in ("start", "count", "inter", "hidden", "c13", "c2", "rt_w13", "rt_w2")}
    for name in ("w13", "w2"):
        bank[name] = m13_relayout(m11[name])
        s = m11[name + "_s"]  # int16 [count, rt, KG, 32]
        n, rt, kg, lanes = s.shape
        bank[name + "_s"] = s.view(n, rt, kg // 4, 4, lanes).permute(0, 1, 2, 4, 3).contiguous()
    pairs = max(M13_MAX_TOKENS, M14_MAX_TOKENS) * TOP_K  # scratch rows: one per pair
    maxpass = (max(M13_MAX_TOKENS, M14_MAX_TOKENS) + 7) // 8  # counters laid out for 8-pair passes
    bank["scr13"] = torch.zeros((pairs, 16 * bank["rt_w13"]), dtype=torch.float32, device=device)
    bank["scr2"] = torch.zeros((pairs, 16 * bank["rt_w2"]), dtype=torch.float32, device=device)
    bank["ctr13"] = torch.zeros((32 * (bank["rt_w13"] // 2) * maxpass,), dtype=torch.int32, device=device)
    bank["ctr2"] = torch.zeros((32 * bank["rt_w2"] * maxpass,), dtype=torch.int32, device=device)
    return bank


def m13_gemm1(x, ids, state, s1):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair."""
    kind = s1.get("kind")
    if kind not in ("m13", "m14"):
        return m11_gemm1(x, ids, state, s1)
    bank, n_pairs = state["m13"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    gemm = state["m11_ext"].m13_gemm if kind == "m13" else state["m11_ext"].m14_gemm
    gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, bank["scr13"], bank["ctr13"],
                              n_pairs, bank["start"], bank["rt_w13"], 0, bank["inter"], 1, 2, s1["nt"],
                              s1["c"] * state["sms"], 0, 0)
    return act


def m13_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    kind = s2.get("kind")
    if kind not in ("m13", "m14"):
        return m11_gemm2(act, ids, state, s2)
    bank, n_pairs = state["m13"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    gemm = state["m11_ext"].m13_gemm if kind == "m13" else state["m11_ext"].m14_gemm
    gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, bank["scr2"], bank["ctr2"],
                              n_pairs, bank["start"], bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"],
                              s2["c"] * state["sms"], s2.get("pdl", 0), s2.get("pf", 0))
    return y


def m13_post_norm(y, ids, topk_weights, state, tokens, post):
    if not post:
        return m11_post_norm(y, ids, topk_weights, state, tokens)
    out = torch.empty((tokens, state["m13"]["hidden"]), dtype=torch.bfloat16, device=y.device)
    state["m11_ext"].m13_post_norm(y, state["norm_w"], topk_weights, ids, out, tokens, state["m13"]["start"],
                                   POST_NORM_EPS, post.get("pdl", 0))
    return out


def m13_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m13_gemm1(x, ids, state, cfg["s1"])
    y = m13_gemm2(act, ids, state, cfg["s2"])
    return m13_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0], cfg.get("post"))


def m13_config_for(state, tokens):
    if tokens > max(M13_MAX_TOKENS, M14_MAX_TOKENS) or "m13" not in state:
        return None
    for below, cfg in state["m13_configs"]:
        if tokens < below:
            return cfg
    return None


def m13_warm(state, cfg, tokens=8):
    """Run cfg once before any capture (and check it doesn't fault)."""
    device = state["norm_w"].device
    x = torch.zeros((tokens, state["m13"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m13"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m13_run(x, ids + state["m13"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m13_setup(state, weights):
    """Build M11 + M13 in one extension, M11's bank (the M11 fallback configs) and M13's bank."""
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = m13_build()
    state["m12_resources"] = []
    # M14's dynamic shared-memory limits, set once here and never during a capture
    state["m14_resources"] = state["m11_ext"].m14_setup().tolist()
    state["m11"] = m11_bank(weights, 0, 32)
    state["m13_ok"] = m13_scales_ok(weights, 0, 32)
    if state["m13_ok"]:
        state["m13"] = m13_bank(state["m11"], weights["w13"].device)
    torch.cuda.synchronize()


def _submission_prepare(weights):
    state = _v1_prepare(weights)
    state["m11_configs"] = M11_CONFIGS
    state["m13_configs"] = M13_CONFIGS
    m13_setup(state, weights)
    for _, cfg in M11_CONFIGS:
        if cfg is not None:
            m11_warm(state, cfg)
    if "m13" in state:
        for _, cfg in M13_CONFIGS:
            if cfg is not None:
                m13_warm(state, cfg)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"]
    m13cfg = m13_config_for(state, x.shape[0])
    if m13cfg is None or (cfg["local_start"], cfg["local_count"]) != (state["m13"]["start"], state["m13"]["count"]):
        return _m11_kernel_fn(data, state)
    return m13_run(x.contiguous(), data["topk_ids"], data["topk_weights"], state, m13cfg)


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


# ====================================================================== lab (e6): check runs only

"""E6: M14 (CTA-shared activations, stream-K over CTAs) against v1 and the file's current configs at
32-320 tokens, configs chosen on the mean of 3 routings, on the judge's H200, check runs only.
kernel_fn is submission_m13's hybrid (M13 configs from E5 for up to 32 tokens); this prepare()
decides from its own measurements which classes from 17 tokens up go to M14.
  attrs    registers, spills, static + dynamic shared memory, CTAs per SM of every M14 instance
  stage1   GEMM1 per (nt, CTAs per SM): error vs FP32 at bs 33, skewed bs 64 with outliers, skewed bs 128
           (128 pairs on 8 experts: up to 16 pair passes), bs 257; scratch and counters must be clean
           after every run; cold-L2 graph time at bs 32/64/128/256, 3 routings each; the file's M13
           stage for 17-32 tokens alongside
  stage2   GEMM2 + CUDA post-norm per (tt, nt, CTAs per SM), the same
  e2e      per class: best M14 stages x (GEMM2 PDL, L2 prefetch) x post-norm PDL vs FP32 at the class's sizes
           (judge-like, skewed, outliers, 33, 257) and vs v1 and the file's current config; M14 only where
           correct and >= 3% faster than both"""

from nvfp4 import dequantize, outlier_channels

LAB_BUDGET_S = 240.0
E5_PREPARE_LIMIT_S = 275.0  # TASK.md: prepare() gets 300 s; keep a margin
E5_BS = (32, 64, 128, 256)
E5_CLASS = {32: 33, 64: 65, 128: 192, 256: 321}
E5_ROUTINGS = 3
E5_REPS = 10
E5_MAX_STAGE_ERR = 1e-2
E5_MAX_ERR, E5_MAX_ROW_ERR = 1e-2, 5e-2  # stricter than the judge's 1.5e-2 and 7.5e-2
E5_PDL = ((0, 0), (1, 0), (1, 2), (1, 6))  # (GEMM2 pdl, L2-prefetched groups before the wait)


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
    lines = [l.strip() for l in str(exc).splitlines()]
    hits = [l for l in lines if "error" in l.lower() or "Error" in l]
    text = " | ".join(hits) if hits else " ".join(lines)
    return text[:keep]


def _clean(state):
    """True when M13's scratch and counters are all zero (they must be after every complete call); re-zeroes."""
    bank = state["m13"]
    dirty = [k for k in ("scr13", "scr2", "ctr13", "ctr2") if bool(bank[k].ne(0).any())]
    for k in dirty:
        bank[k].zero_()
    return dirty


def sec_attrs(lab, state):
    parts = []
    occ = {}
    for g, tt, nt, regs, local, smem, dyn, blocks in state["m14_resources"]:
        occ[(g, tt, nt)] = blocks
        parts.append(f"{'g' if g else 'p'}{tt}.{nt} {regs}r{f' {local}B-spill' if local else ''} "
                     f"{(smem + dyn) // 1024}K {blocks}cta")
    state["m13_occ"] = occ
    log("attrs M14 (gated/plain tt.nt): " + ", ".join(parts) + f"; scales ok {state.get('m13_ok')}")


def _name(cfg, stage):
    tt = f"t{cfg['tt']}" if stage == 2 else ""
    if cfg.get("kind") in ("m13", "m14"):
        return f"{cfg['kind'].upper()} {tt}n{cfg['nt']}c{cfg['c']}"
    return f"M11 {tt}n{cfg['nt']}d{cfg['d']}k{cfg['ks']}c{cfg['c']}"


def _stage_cfgs(state, stage):
    cfgs = []
    for gated, tt, nt in M14_INSTANCES:
        if gated != (stage == 1):
            continue
        for c in range(1, state["m13_occ"][(gated, tt, nt)] + 1):
            base = dict(kind="m14", nt=nt, c=c)
            cfgs.append(base if stage == 1 else dict(base, tt=tt))
    cur = m13_config_for(state, 32)  # the file's 17-32 token stage (M13 from E5), for the comparison at bs32
    if cur is not None:
        cfgs.append({k: v for k, v in cur["s1" if stage == 1 else "s2"].items() if k not in ("pdl", "pf")})
    return cfgs


def _post(pdl=1):
    return dict(kind="m13", pdl=pdl)


def sec_stage(lab, stage, state, ref, store):
    check_cases = [_case(33, 933), _case(64, 964, skew=True, outliers=True), _case(128, 928, skew=True),
                   _case(257, 957)]
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(E5_ROUTINGS)] for bs in E5_BS}
    if stage == 1:
        checks = [(x, ids, w, ref.act(x, ids)) for x, ids, w in check_cases]
    else:
        checks = []
        for x, ids, w in check_cases:
            act = ref.act(x, ids).to(torch.bfloat16)
            checks.append((act, ids, w, ref.stage2(act, ids, w)))
        acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in E5_BS}
    results = []
    for cfg in _stage_cfgs(state, stage):
        if lab.left() < (110 if stage == 1 else 60):
            log(f"s{stage}: out of budget")
            break
        try:
            errs = []
            for a, ids, w, want in checks:
                flat = ids.reshape(-1)
                if cfg.get("kind") == "m13" and ids.shape[0] > M13_MAX_TOKENS:
                    continue
                if stage == 1:
                    rows = (flat.long() < 32).nonzero().flatten()
                    errs.append(_errors(m13_gemm1(a, flat, state, cfg)[rows], want[rows])[0])
                else:
                    y = m13_gemm2(a, flat, state, cfg)
                    errs.append(_errors(m13_post_norm(y, flat, w, state, ids.shape[0], _post(0)), want)[0])
            torch.cuda.synchronize()
            dirty = _clean(state)
            err = max(errs) if not dirty else float("inf")
            times = {}
            for bs, routings in cases.items():
                if cfg.get("kind") == "m13" and bs > M13_MAX_TOKENS:
                    times[bs] = float("inf")
                    continue
                ts = []
                for x, ids, w in routings:
                    if stage == 1:
                        fn = lambda x=x, i=ids: m13_gemm1(x, i.reshape(-1), state, cfg)
                    else:
                        fn = lambda a=acts[bs], i=ids, w=w: m13_post_norm(
                            m13_gemm2(a, i.reshape(-1), state, cfg), i.reshape(-1), w, state, i.shape[0], _post(1))
                    ts.append(lab.time_graph(fn, reps=E5_REPS))
                times[bs] = sum(ts) / len(ts)
            torch.cuda.synchronize()
            late = _clean(state)  # always re-zero, even when the check run was already dirty
            dirty = dirty or late
            if dirty:
                err = float("inf")
        except Exception as exc:  # report and go on; a CUDA fault would also show in every later line
            log(f"s{stage} {_name(cfg, stage)}: FAILED {type(exc).__name__}: {_error_text(exc, 300)}")
            continue
        results.append((cfg, err, times))
        log(f"s{stage} {_name(cfg, stage)} e{err:.0e}{' DIRTY ' + ','.join(dirty) if dirty else ''} | "
            + " ".join(f"{t:.1f}" for t in times.values()))
    store[stage] = [r for r in results if r[1] < E5_MAX_STAGE_ERR]
    for bs in E5_BS:
        m14 = [r for r in store[stage] if r[0].get("kind") == "m14"]
        other = [r for r in store[stage] if r[0].get("kind") != "m14"]
        if m14:
            b = min(m14, key=lambda r: r[2][bs])
            bo = min(other, key=lambda r: r[2][bs]) if other else None
            log(f"s{stage} best at bs{bs}: {_name(b[0], stage)} {b[2][bs]:.1f} us"
                + (f" (current {_name(bo[0], stage)} {bo[2][bs]:.1f})" if bo else ""))


def sec_e2e(lab, state, ref, store):
    s1 = [r for r in store.get(1, []) if r[0].get("kind") == "m14"]
    s2 = [r for r in store.get(2, []) if r[0].get("kind") == "m14"]
    if not s1 or not s2:
        log("e2e: no working M14 stage configs; kernel_fn keeps its configs")
        return
    cases = [(bs, False, False) for bs in (17, 24, 32, 33, 48, 64, 65, 100, 128, 160, 191, 192, 256, 257, 320)]
    cases += [(bs, True, False) for bs in (32, 64, 128, 256)] + [(128, True, True), (257, False, True)]
    chosen = dict((below, cfg) for below, cfg in state["m13_configs"])
    for bs in E5_BS:
        if lab.left() < 12:
            log("e2e: out of budget")
            break
        lo = max([E5_CLASS[b] for b in E5_BS if b < bs], default=1)
        hi = E5_CLASS[bs]
        r1 = min(s1, key=lambda r: r[2][bs])
        r2 = min(s2, key=lambda r: r[2][bs])
        runs = [(x, ids, w) for x, ids, w in (_case(bs, 1700 + bs + 2000 * r) for r in range(E5_ROUTINGS))]
        t_v1 = t_m11 = 0.0
        m11cfg = m13_config_for(state, bs)  # the file's current config for this class (M13 or None)
        for x, ids, w in runs:
            data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                    "config": {"bs": bs, "local_start": 0, "local_count": 32}}
            t_v1 += lab.time_graph(lambda: _v1_kernel_fn(data, state)) / E5_ROUTINGS
            if m11cfg is not None:
                t_m11 += lab.time_graph(lambda: m13_run(x, ids, w, state, m11cfg)) / E5_ROUTINGS
        best_other = min(t_v1, t_m11) if m11cfg is not None else t_v1
        variants = []
        for pdl, pf in E5_PDL:
            for post_pdl in (0, 1):
                cfg = dict(s1=r1[0], s2=dict(r2[0], pdl=pdl, pf=pf), post=_post(post_pdl))
                t = sum(lab.time_graph(lambda: m13_run(x, ids, w, state, cfg)) for x, ids, w in runs) / E5_ROUTINGS
                variants.append((t, cfg))
        _clean(state)
        t_best, cfg = min(variants, key=lambda v: v[0])
        worst = (0.0, 0.0)
        for cbs, skew, outl in cases:
            if not lo <= cbs < hi:
                continue
            x, ids, w = _case(cbs, 1300 + cbs + 50 * skew + 25 * outl, skew, outl)
            err, row = _errors(m13_run(x, ids, w, state, cfg), ref.full(x, ids, w))
            worst = (max(worst[0], err), max(worst[1], row))
        torch.cuda.synchronize()
        dirty = _clean(state)
        use = (not dirty and worst[0] < E5_MAX_ERR and worst[1] < E5_MAX_ROW_ERR and t_best < 0.97 * best_other)
        if use:
            chosen[E5_CLASS[bs]] = cfg
        log(f"e2e bs{bs}: v1 {t_v1:.1f}, current {t_m11:.1f}, M14 {t_best:.1f} us ({best_other / t_best:.2f}x) "
            f"| pdl variants " + " ".join(f"{v[1]['s2']['pdl']}{v[1]['s2']['pf']}{v[1]['post']['pdl']}:{v[0]:.1f}" for v in variants)
            + f" | err {worst[0]:.1e} row {worst[1]:.1e}{' DIRTY' if dirty else ''} -> {'M14' if use else 'current'} | {cfg}")
    state["m13_configs"] = tuple(sorted(chosen.items()))
    for cfg in chosen.values():
        if cfg is not None:
            m13_warm(state, cfg)
    log(f"M13_CONFIGS = {state['m13_configs']!r}")


def prepare(weights):
    t = time.monotonic()
    try:
        state = _submission_prepare(weights)
    except Exception as exc:  # nvcc runs inside the submission's prepare
        log(f"prepare FAILED after {time.monotonic() - t:.1f}s: {_build_error_lines(exc)}")
        raise
    took = time.monotonic() - t
    log(f"E6: submission prepare (nvcc, banks, warm-up) took {took:.1f}s")
    budget = min(LAB_BUDGET_S, E5_PREPARE_LIMIT_S - took)  # prepare() as a whole must stay under 300 s
    lab = Lab(budget)
    log(f"E6 start, budget {budget:.0f}s; stage lines: time at bs " + "/".join(map(str, E5_BS)) + " in us")
    if "m13" not in state:
        log(f"E6: M13/M14 off (scales ok {state.get('m13_ok')}); nothing to measure")
        return state
    ref = _Ref(weights)
    store = {}
    lab.section("attrs", sec_attrs, state)
    if "m13_occ" in state:
        lab.section("stage1", sec_stage, 1, state, ref, store, min_left=60)
        lab.section("stage2", sec_stage, 2, state, ref, store, min_left=40)
        lab.section("e2e", sec_e2e, state, ref, store, min_left=15)
    ref.cache.clear()
    lab.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s; kernel_fn: {state.get('m13_configs')!r}"[:900])
    return state
