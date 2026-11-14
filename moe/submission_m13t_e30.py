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
# token routes to an expert at most once. E22b: 64 (was 320): M11 runs only below 33 tokens (m11_config_for maps
# 33+ to v1; every M11 tuner candidate is for a class <= 32), and the static s_list shrinks 20 KB -> 4 KB.
M11_MAX_TOKENS = 64
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

// PDL (E15, as M13's): the next grid may launch once every CTA of this one has signalled; a PDL-launched grid
// waits before reading the previous grid's output, through loads that depend on the wait's (zero) token.
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t pdl_wait_token() {
  uint32_t z;
  asm volatile("griddepcontrol.wait;\n\tmov.u32 %0, 0;" : "=r"(z) : : "memory");
  return z;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
// A task (expert, TT row tiles) goes to a group of KS warps that split K and add their partial sums
// in shared memory; an expert's pairs are taken 8 * NT at a time (a second chunk re-reads its weights).
// PDL (GEMM2 only): launched early with programmatic stream serialization; scans the routing, then every CTA waits
// and reads the activations with coherent token-ordered loads (__ldg's non-coherent path isn't safe behind the wait).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
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
  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched
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
  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads
  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)

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
            if constexpr (PDL) {
              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);
              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);
            } else {
              ra[s][n][0] = __ldg(ap[n] + kg * 8);
              ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
            }
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

// E21: gemm_kernel with the register ring kept streaming across (task, pair pass) units. During a unit's last D
// chunks each warp loads the next unit's first D chunks into the same ring slots (KGW % D == 0, so slot = chunk % D
// lines up), so only a group's first unit pays the memory latency; the split-K reduction and the epilogue overlap
// those loads. Same scan, decode, MMA, reduction and epilogue as gemm_kernel.
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) cr_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  static_assert(KGW % D == 0 && KGW >= D, "depth must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();
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
  uint32_t tok = 0u;
  if constexpr (PDL) tok = pdl_wait_token();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;

  // A unit is (task, first pair of the pass); units of a group: its tasks in stride order, each task's passes.
  struct Ops {
    const uint4* wp[TT];
    const uint16_t* sp[TT];
    const uint4* ap[NT];
    bool aok[NT];
  };
  auto ops_for = [&](int task, int c0, Ops& o) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
      const size_t base = (static_cast<size_t>(e) * rt_total + rt) * KG * 32 + lane;
      o.wp[i] = w + base;
      o.sp[i] = sf + base;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      o.aok[n] = idx < cnt;
      const int p = o.aok[n] ? s_list[e][c0 + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      o.ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
    }
  };
  auto next_unit = [&](int& task, int& c0) {
    const int e = s_active[task / per_expert];
    c0 += 8 * NT;
    if (c0 >= min(s_cnt[e], MAXP)) {
      c0 = 0;
      task += stride;
    }
  };

  uint4 rw[D][TT];
  uint32_t rs[D][TT];
  uint4 ra[D][NT][2];
  auto load = [&](const Ops& o, int s, int kg) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      rw[s][i] = __ldcs(o.wp[i] + kg * 32);
      rs[s][i] = __ldcs(o.sp[i] + kg * 32);
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (o.aok[n]) {
        if constexpr (PDL) {
          ra[s][n][0] = ld_cg4(o.ap[n] + kg * 8 + tok);
          ra[s][n][1] = ld_cg4(o.ap[n] + kg * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(o.ap[n] + kg * 8);
          ra[s][n][1] = __ldg(o.ap[n] + kg * 8 + 1);
        }
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  int task = blockIdx.x * GROUPS + grp, c0 = 0;
  if (task >= n_tasks) return;  // the whole group (all KS warps) has no unit: no group_sync is reached
  Ops oc, on;
  ops_for(task, c0, oc);
#pragma unroll
  for (int s = 0; s < D; ++s) load(oc, s, kbeg + s);
  while (task < n_tasks) {
    int ntask = task, nc0 = c0;
    next_unit(ntask, nc0);
    const bool more = ntask < n_tasks;
    if (more) ops_for(ntask, nc0, on);
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    float acc[TT][NT][4];
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
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
    for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        compute(s);
        const int j = kg0 + s + D;
        if (j < KGW) {
          load(oc, s, kbeg + j);
        } else if (more) {
          load(on, s, kbeg + j - KGW);  // the next unit's chunk j - KGW (< D) -> slot s = (j - KGW) % D
        }
      }
    }

    if constexpr (KS > 1) {
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
    task = ntask;
    c0 = nc0;
    if (more) oc = on;
  }
}

// E22 helpers (M11's own smem_addr comes later, with M12)
__device__ __forceinline__ uint32_t sr_smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void sr_cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void sr_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void sr_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// E22: cr_kernel with the weight codes and scales in a per-warp shared-memory ring DW chunks deep (cp.async), so
// the bytes in flight no longer shrink while a chunk is decoded (MF24: loads alone reach ~4.3 TB/s at 8 tokens, the
// full kernel ~2.5) and ~40 registers leave the ring. Codes: each lane copies its own 16 B; scales: lanes 0-3 copy
// the warp's 64-B block (cp.async needs >= 4 B) and a __syncwarp publishes it. One commit group per chunk, so
// wait_group<DW - 1> covers the chunk about to be computed. Activations keep cr_kernel's register ring (L1/L2 hits).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false, int DW = 8>
__global__ void __launch_bounds__(128) sr_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  constexpr int GROUPS = 4 / KS;
  constexpr int KGW = KG / KS;
  static_assert(KGW % D == 0 && KGW >= D, "depth must divide each warp's K chunks");
  static_assert(KGW % DW == 0 && KGW >= DW && DW >= 2 && DW <= 8, "the weight ring must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  extern __shared__ __align__(16) unsigned char s_ring[];  // [4 warps][DW][TT] codes x 512 B, then scales x 64 B
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();
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
  uint32_t tok = 0u;
  if constexpr (PDL) tok = pdl_wait_token();

  const int lane = tid & 31, g = lane >> 2, t = lane & 3;
  const int warp = tid >> 5, grp = warp / KS, kpart = warp % KS;
  const uint32_t ring0 = sr_smem_addr(s_ring);
  auto coff = [&](int slot, int i) { return ((warp * DW + slot) * TT + i) * 512 + lane * 16; };
  auto soff = [&](int slot, int i) { return 4 * DW * TT * 512 + ((warp * DW + slot) * TT + i) * 64; };
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int stride = gridDim.x * GROUPS;
  const int kbeg = kpart * KGW;

  // A unit is (task, first pair of the pass); units of a group: its tasks in stride order, each task's passes.
  struct Ops {
    const uint4* wp[TT];
    const uint16_t* sp[TT];
    const uint4* ap[NT];
    bool aok[NT];
  };
  auto ops_for = [&](int task, int c0, Ops& o) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int rt = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
      const size_t base = (static_cast<size_t>(e) * rt_total + rt) * KG * 32 + lane;
      o.wp[i] = w + base;
      o.sp[i] = sf + base;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      o.aok[n] = idx < cnt;
      const int p = o.aok[n] ? s_list[e][c0 + idx] : 0;
      const int row = a_per_pair ? p : p / TOP_K;
      o.ap[n] = reinterpret_cast<const uint4*>(a + static_cast<size_t>(row) * K + 16 * t);
    }
  };
  auto next_unit = [&](int& task, int& c0) {
    const int e = s_active[task / per_expert];
    c0 += 8 * NT;
    if (c0 >= min(s_cnt[e], MAXP)) {
      c0 = 0;
      task += stride;
    }
  };

  uint4 ra[D][NT][2];
  auto load_w = [&](const Ops& o, int slot, int kg) {  // codes (own 16 B) and the warp's 64-B scale block
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      sr_cp16(ring0 + coff(slot, i), o.wp[i] + kg * 32);
      if (lane < 4) sr_cp16(ring0 + soff(slot, i) + lane * 16, reinterpret_cast<const unsigned char*>(o.sp[i] - lane + kg * 32) + lane * 16);
    }
  };
  auto load = [&](const Ops& o, int s, int kg) {
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (o.aok[n]) {
        if constexpr (PDL) {
          ra[s][n][0] = ld_cg4(o.ap[n] + kg * 8 + tok);
          ra[s][n][1] = ld_cg4(o.ap[n] + kg * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(o.ap[n] + kg * 8);
          ra[s][n][1] = __ldg(o.ap[n] + kg * 8 + 1);
        }
      } else {
        ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
        ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };

  int task = blockIdx.x * GROUPS + grp, c0 = 0;
  if (task >= n_tasks) return;  // the whole group (all KS warps) has no unit: no group_sync is reached
  Ops oc, on;
  ops_for(task, c0, oc);
#pragma unroll
  for (int s = 0; s < D; ++s) load(oc, s, kbeg + s);
#pragma unroll
  for (int s = 0; s < DW; ++s) {  // the first unit's first DW chunks, one commit group each
    load_w(oc, s, kbeg + s);
    sr_commit();
  }
  int wc = 0;  // chunks consumed: chunk wc's codes and scales are in ring slot wc % DW
  while (task < n_tasks) {
    int ntask = task, nc0 = c0;
    next_unit(ntask, nc0);
    const bool more = ntask < n_tasks;
    if (more) ops_for(ntask, nc0, on);
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP) - c0, 8 * NT);
    int rts[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rts[i] = GATED ? tile + i * (rt_total / 2) : tile * TT + i;
    const float c = cs[e];
    float acc[TT][NT][4];
#pragma unroll
    for (int i = 0; i < TT; ++i)
#pragma unroll
      for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
    auto compute = [&](int s, int wsl) {
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const uint4 rwv = *reinterpret_cast<const uint4*>(s_ring + coff(wsl, i));
        const uint32_t rsv = *reinterpret_cast<const uint16_t*>(s_ring + soff(wsl, i) + lane * 2);
        uint32_t sg, sg8;
        scales(rsv, sg, sg8);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const uint32_t q = word(rwv, j);
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
    for (int kg0 = 0; kg0 < KGW; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        sr_wait<DW - 1>();  // chunk wc's group has landed (one group per chunk)
        __syncwarp();       // ... and lanes 0-3's scale copies are visible to the whole warp
        compute(s, wc % DW);
        __syncwarp();       // every lane has read slot wc % DW before it is refilled
        const int jw = kg0 + s + DW;  // weights DW chunks ahead (into the next unit past KGW)
        if (jw < KGW) {
          load_w(oc, wc % DW, kbeg + jw);
        } else if (more) {
          load_w(on, wc % DW, kbeg + jw - KGW);
        }
        sr_commit();
        ++wc;
        const int j = kg0 + s + D;
        if (j < KGW) {
          load(oc, s, kbeg + j);
        } else if (more) {
          load(on, s, kbeg + j - KGW);  // the next unit's chunk j - KGW (< D) -> slot s = (j - KGW) % D
        }
      }
    }

    if constexpr (KS > 1) {
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
    task = ntask;
    c0 = nc0;
    if (more) oc = on;
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

// E21: the continuous-ring kernel, selected by depth | 16.
#define M11_CR_CASE(G, TT, NT, D, KS)                                                                    \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | 16) && ks == (KS)) {                   \
    m11::cr_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 0, stream>>>(                        \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

// E22: the shared-memory-ring kernel (DW = 8), selected by depth | 32; its ring is dynamic shared memory
// (<= 36 KB, under the 48 KB default, so no attribute call, none during a capture).
#define M11_SR_CASE(G, TT, NT, D, KS)                                                                    \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | 32) && ks == (KS)) {                   \
    m11::sr_kernel<TT, NT, D, KS, G><<<static_cast<int>(grid), 128, 4 * 8 * (TT) * 576, stream>>>(       \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    const cudaError_t err = cudaGetLastError();                                                          \
    TORCH_CHECK(err == cudaSuccess, "m11 sr launch failed: ", cudaGetErrorString(err));                   \
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

#define M11_KATTR(KERN, G, TT, NT, D, KS, BIT)                                                           \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | (BIT)) && ks == (KS)) {                 \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::KERN<TT, NT, D, KS, G>);                                            \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
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


def m11_source(instances=M11_INSTANCES, m12_instances=M12_INSTANCES, cr_instances=(), sr_instances=()):
    """The CUDA source with one dispatch case per instance (cr_instances: E21's continuous-ring kernel)."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in instances]
    args_cr = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in cr_instances]
    args_sr = [f"{'true' if g else 'false'}, {tt}, {nt}, {d}, {ks}" for g, tt, nt, d, ks in sr_instances]
    args12 = [f"{'true' if g else 'false'}, {tt}, {nt}, {ks}, {ch}, {r}" for g, tt, nt, ks, ch, r in m12_instances]
    return (_M11_CUDA.replace("@@CASES@@", "\n  ".join([f"M11_CASE({a})" for a in args] + [f"M11_CR_CASE({a})" for a in args_cr]
                                                + [f"M11_SR_CASE({a})" for a in args_sr]))
            .replace("@@ATTRS@@", "\n  ".join([f"M11_ATTR({a})" for a in args] + [f"M11_KATTR(cr_kernel, {a}, 16)" for a in args_cr]
                                              + [f"M11_KATTR(sr_kernel, {a}, 32)" for a in args_sr]))
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
        state["m11_ext"].m11_gemm(*args, s1["d"] | (s1.get("cr", 0) << 4) | (s1.get("sr", 0) << 5), s1["ks"], s1["c"] * state["sms"])
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
        state["m11_ext"].m11_gemm(*args, s2["d"] | (s2.get("cr", 0) << 4) | (s2.get("sr", 0) << 5), s2["ks"], s2["c"] * state["sms"])
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


# ====================================================================== G2P (moe/m13/g2p_moe.py)

"""G2P for MoE (namespace m11f): M11 GEMM2 + post-expert norm weighted sum in one kernel; built by
shared/glue/make_g2p.py from moe/submission_m13t_e22.py's _M11_CUDA (its gemm_kernel copied byte for byte).
Do not edit by hand. Needs MoE's namespace: torch, M11_MAX_TOKENS, POST_NORM_EPS, and m13_launch in CUDA."""

_M11F_CUDA = r'''namespace m11f {

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

// PDL (E15, as M13's): the next grid may launch once every CTA of this one has signalled; a PDL-launched grid
// waits before reading the previous grid's output, through loads that depend on the wait's (zero) token.
__device__ __forceinline__ void pdl_launch() { asm volatile("griddepcontrol.launch_dependents;" ::: "memory"); }
__device__ __forceinline__ uint32_t pdl_wait_token() {
  uint32_t z;
  asm volatile("griddepcontrol.wait;\n\tmov.u32 %0, 0;" : "=r"(z) : : "memory");
  return z;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
// A task (expert, TT row tiles) goes to a group of KS warps that split K and add their partial sums
// in shared memory; an expert's pairs are taken 8 * NT at a time (a second chunk re-reads its weights).

// ------------------------------------------------------------------------------------------------ G2P
// Group barrier of KS warps (M11's group_sync ids 1-4), or a warp sync for a one-warp group.
__device__ __forceinline__ void g2p_sync(int grp, int ks) {
  if (ks == 1) __syncwarp();
  else group_sync(grp + 1, ks * 32);
}

__device__ __forceinline__ const __nv_bfloat16* a_y(const __nv_bfloat16* p) { return p; }

// M13's post-norm for one token by the KS * 32 threads of a group: v = y * nw (y bf16, nw bf16, fp32 products),
// rs[k] = rsqrt(sum v^2 / K + eps), out = bf16_rn(sum over local k of (tw * rs[k]) * v). Two passes over the rows
// (sum of squares, then the weighted sum) keep registers low; thread tig owns uint4 columns tig + i * KS * 32.
template <int KS>
__device__ __forceinline__ void g2p_token(int tk, int tig, int grp, const int* __restrict__ ids, int local_start,
                                          const __nv_bfloat16* y, const __nv_bfloat16* __restrict__ nw,
                                          const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout, float eps,
                                          float (*s_pss)[KS][TOP_K], float (*s_rs)[TOP_K]) {
  constexpr int NT_ = KS * 32;
  constexpr int PER = (K / 8) / NT_;  // uint4 columns per thread
  const int lane = tig & 31, wq = tig >> 5;
  bool loc[TOP_K];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    const int e = ids[tk * TOP_K + k] - local_start;
    loc[k] = e >= 0 && e < LOCAL;
    ss[k] = 0.0f;
  }
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        const float v0 = __uint_as_float(u << 16) * __uint_as_float(wu << 16);
        const float v1 = __uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u);
        ss[k] += v0 * v0 + v1 * v1;
      }
    }
  }
#pragma unroll
  for (int k = 0; k < TOP_K; ++k)
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss[k] += __shfl_xor_sync(0xffffffffu, ss[k], o);
  if (lane == 0)
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) s_pss[grp][wq][k] = ss[k];
  g2p_sync(grp, KS);
  if (tig < TOP_K) {
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < KS; ++i) s += s_pss[grp][i][tig];
    s_rs[grp][tig] = rsqrtf(s / static_cast<float>(K) + eps);
  }
  g2p_sync(grp, KS);
  float f[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) f[k] = loc[k] ? tw[tk * TOP_K + k] * s_rs[grp][k] : 0.0f;
#pragma unroll
  for (int c = 0; c < PER; ++c) {
    const int col = tig + c * NT_;
    const uint4 wraw = __ldg(reinterpret_cast<const uint4*>(nw) + col);
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = 0.0f;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      if (!loc[k]) continue;
      const uint4 r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tk * TOP_K + k) * K) + col);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t u = word(r, j), wu = word(wraw, j);
        o[2 * j] += f[k] * (__uint_as_float(u << 16) * __uint_as_float(wu << 16));
        o[2 * j + 1] += f[k] * (__uint_as_float(u & 0xFFFF0000u) * __uint_as_float(wu & 0xFFFF0000u));
      }
    }
    uint32_t pk[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j]));
      const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(o[2 * j + 1]));
      pk[j] = lo | (hi << 16);
    }
    reinterpret_cast<uint4*>(pout + static_cast<size_t>(tk) * K)[col] = make_uint4(pk[0], pk[1], pk[2], pk[3]);
  }
  g2p_sync(grp, KS);  // s_rs is rewritten by the next token
}

// PDL (GEMM2 only): launched early with programmatic stream serialization; scans the routing, then every CTA waits
// and reads the activations with coherent token-ordered loads (__ldg's non-coherent path isn't safe behind the wait).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_post_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols,
    const __nv_bfloat16* __restrict__ nw, const float* __restrict__ tw, __nv_bfloat16* __restrict__ pout,
    int* __restrict__ cnt, float eps) {
  static_assert(!(GATED && PDL), "GEMM1 is the first kernel of a call");
  static_assert(!GATED, "G2P: GEMM2 only");
  __shared__ int s_fin[4 / KS];
  __shared__ int s_ndone[4 / KS];
  __shared__ int s_done[4 / KS][MAXP];
  __shared__ float s_pss[4 / KS][KS][TOP_K];
  __shared__ float s_rs[4 / KS][TOP_K];
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
  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched
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
  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads
  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)
  for (int zt = blockIdx.x; zt < n_pairs / TOP_K; zt += gridDim.x) {  // G2P: tokens without a local route: zero rows
    int any = 0;
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
      const int ze = ids[zt * TOP_K + k] - local_start;
      any |= (ze >= 0 && ze < LOCAL) ? 1 : 0;
    }
    if (!any)
      for (int i = tid; i < K / 8; i += blockDim.x)
        reinterpret_cast<uint4*>(pout + static_cast<size_t>(zt) * K)[i] = make_uint4(0u, 0u, 0u, 0u);
  }

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
            if constexpr (PDL) {
              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);
              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);
            } else {
              ra[s][n][0] = __ldg(ap[n] + kg * 8);
              ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
            }
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
    {  // G2P: this task's y tiles are written (warp kpart 0 of the group stored them)
      const int tig = kpart * 32 + lane;  // thread in group
      if (kpart == 0) {
        __threadfence();  // release: this lane's y stores before the counter
        __syncwarp();
        int fin = 0;
        if (lane == 0) {
          fin = atomicAdd(cnt + e, 1) == per_expert - 1;
          if (fin) {
            __threadfence();  // acquire: every other task's y stores of expert e
            atomicExch(cnt + e, 0);  // self-reset: no other task of this call touches cnt[e] again
          }
        }
        fin = __shfl_sync(0xffffffffu, fin, 0);
        if (lane == 0) {
          s_fin[grp] = fin;
          s_ndone[grp] = 0;
        }
      }
      g2p_sync(grp, KS);
      if (s_fin[grp]) {  // this group finished expert e: count its pairs' tokens
        for (int i = tig; i < total; i += KS * 32) {
          const int tk = s_list[e][i] / TOP_K;
          int nl = 0;
#pragma unroll
          for (int k = 0; k < TOP_K; ++k) {
            const int e2 = ids[tk * TOP_K + k] - local_start;
            nl += (e2 >= 0 && e2 < LOCAL) ? 1 : 0;
          }
          __threadfence();  // release by this thread: expert e's y rows (acquired by the group above) before the token counter
          if (atomicAdd(cnt + LOCAL + tk, 1) == nl - 1) {
            __threadfence();  // acquire: the other experts' y rows of token tk
            atomicExch(cnt + LOCAL + tk, 0);
            s_done[grp][atomicAdd(&s_ndone[grp], 1)] = tk;
          }
        }
        g2p_sync(grp, KS);
        const int nd = s_ndone[grp];
        for (int d = 0; d < nd; ++d)
          g2p_token<KS>(s_done[grp][d], tig, grp, ids, local_start, a_y(out), nw, tw, pout, eps, s_pss, s_rs);
      }
      g2p_sync(grp, KS);  // s_fin / s_done / s_ndone are rewritten by the next task
    }
  }
}
#define M11F_CASE(TT, NT, D, KS, P)                                                                         \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS) && pdl == (P)) {                                \
    m13_launch(m11f::gemm_post_kernel<TT, NT, D, KS, false, (P) != 0>, static_cast<int>(grid), 128, 0, (P),   \
               reinterpret_cast<const uint4*>(w2.data_ptr()), reinterpret_cast<const uint16_t*>(w2_s.data_ptr()), \
               static_cast<const float*>(c2.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),  \
               reinterpret_cast<const __nv_bfloat16*>(act.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(y_scratch.data_ptr()), static_cast<int>(n_pairs),           \
               static_cast<int>(local_start), static_cast<int>(rt_w2), 1, static_cast<int>(hidden),         \
               reinterpret_cast<const __nv_bfloat16*>(norm_w.data_ptr()),                                    \
               static_cast<const float*>(topk_w.data_ptr<float>()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
               ctr.data_ptr<int>(), static_cast<float>(eps));                                                  \
    return out;                                                                                               \
  }

}  // namespace m11f

// G2P for MoE: M11's GEMM2 (y_scratch = act @ w2^T for every local pair) + the post-norm weighted sum into out
// [tokens, hidden], one kernel, launched with MoE's m13_launch (128 threads, no dynamic shared memory; PDL when pdl).
// ctr: int32 [32 + tokens] (allocate 32 + M11_MAX_TOKENS once), zero on entry and zero again on exit.
torch::Tensor m11f_gemm2_post(torch::Tensor w2, torch::Tensor w2_s, torch::Tensor c2, torch::Tensor ids,
                              torch::Tensor act, torch::Tensor y_scratch, torch::Tensor topk_w, torch::Tensor norm_w,
                              torch::Tensor out, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                              int64_t rt_w2, int64_t hidden, int64_t tokens, double eps, int64_t tt, int64_t nt,
                              int64_t depth, int64_t ks, int64_t grid, int64_t pdl) {
  TORCH_CHECK(hidden == m11f::K && norm_w.numel() == m11f::K && out.size(1) == m11f::K && tokens * 8 == n_pairs &&
              out.size(0) == tokens && tokens <= m11f::MAXP && ctr.numel() >= m11f::LOCAL + tokens,
              "m11f_gemm2_post: shapes / counters");
  @@CASES@@
  TORCH_CHECK(false, "m11f_gemm2_post: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks, " pdl=", pdl);
  return out;
}
'''

_M11F_CPP = '''
torch::Tensor m11f_gemm2_post(torch::Tensor w2, torch::Tensor w2_s, torch::Tensor c2, torch::Tensor ids,
                              torch::Tensor act, torch::Tensor y_scratch, torch::Tensor topk_w, torch::Tensor norm_w,
                              torch::Tensor out, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                              int64_t rt_w2, int64_t hidden, int64_t tokens, double eps, int64_t tt, int64_t nt,
                              int64_t depth, int64_t ks, int64_t grid, int64_t pdl);
'''

G2P_MOE_TEST_PRELUDE = r'''
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
template <typename... Exp, typename... Act>
static void m13_launch(void (*kernel)(Exp...), int grid, int threads, size_t smem, int pdl, Act... args) {}
'''


# MoE variant of G2P (TRACKER shared item 5). Splice _M11F_CUDA (device code in namespace m11f + the host function
# m11f_gemm2_post, which calls MoE's m13_launch) into MoE's CUDA module after m13_launch's definition, add _M11F_CPP's
# prototype to its cpp sources and "m11f_gemm2_post" to its functions.
M11F_INSTANCES = ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 1, 4, 4, 0))


def g2p_moe_source(instances=None):
    """The device + host text with one launch case per (tt, nt, depth, ks, pdl) instance; instances default to
    M11F_INSTANCES read at call time."""
    instances = M11F_INSTANCES if instances is None else instances
    cases = "\n  ".join(f"M11F_CASE({tt}, {nt}, {d}, {ks}, {p})" for tt, nt, d, ks, p in instances)
    return _M11F_CUDA.replace("@@CASES@@", cases).replace("@@MAXP@@", str(M11_MAX_TOKENS))


def m11f_counters(device):
    """The counters m11f_gemm2_post needs: int32 [32 + M11_MAX_TOKENS], zero; allocate once in prepare."""
    return torch.zeros(32 + M11_MAX_TOKENS, dtype=torch.int32, device=device)


def m11f_gemm2_post_py(act, ids, topk_w, state, tokens, s2, ext, ctr, sms):
    """act bf16 [pairs, I], ids int32 [pairs] (global ids), topk_w float32 [tokens, 8] contiguous -> bf16 [tokens, H]
    = M13's post-norm of M11's GEMM2; s2 = dict(tt, nt, d, ks, c, optional pdl) of an instance in M11F_INSTANCES."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    return ext.m11f_gemm2_post(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, topk_w, state["norm_w"], out, ctr,
                               n_pairs, bank["start"], bank["rt_w2"], bank["hidden"], tokens, POST_NORM_EPS,
                               s2["tt"], s2["nt"], s2["d"], s2["ks"], s2["c"] * sms, int(s2.get("pdl", 0)))


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
# (gated, tt, nt, P): P = activation chunks staged ahead (P + 1 shared-memory stages); E9 ran P = 2 only.
# n8 dropped (E6/E8/E9: always the slowest, 237-254 registers).
M14_INSTANCES = ((1, 2, 2, 2), (1, 2, 2, 4), (1, 2, 2, 6), (1, 2, 4, 2), (1, 2, 4, 4),
                 (0, 2, 2, 2), (0, 2, 2, 4), (0, 2, 2, 6), (0, 2, 4, 4), (0, 1, 4, 4))
# M15 (weights through a per-warp shared-memory TMA ring, one pass up to 8 nt pairs): (gated, tt, nt, ring slots)
# (gated, tt, nt, P): RS-wgmma, N = 8 nt pairs per pass at most (128, 64 or 32), P = activation lead; p1.16 takes
# bs256's hottest expert (~128 pairs) in one pass.
# v2 (E13): (gated, tt, nt, P, DB); DB = double-buffered A + wgmma.wait_group 1 (E12's v1 waited for every chunk's
# MMAs and was > 1.25x M14's time). DB costs 16-32 registers, so g2.8 / p2.8 stay single-buffered.
M16_INSTANCES = ((1, 2, 8, 4, 0), (1, 2, 4, 4, 0), (1, 2, 4, 4, 1), (0, 2, 8, 4, 0), (0, 2, 4, 4, 1), (0, 1, 8, 4, 1),
                 (0, 1, 16, 4, 0), (0, 1, 16, 4, 1))
# (gated, tt, nt, R, P): E9's ran P = 2 (R 3/4/4/4/3) and lost to M14 and M13; these stage activations 3-4 chunks
# ahead (MF11), with R cut where needed to keep 2 CTAs per SM.
# (gated, tt, nt, R, P): M15's ring and cost split with M16's wgmma (natural-K bank); sized like M15's for 2 CTAs/SM.
# The NT 1/2 ones (R 2: ~63 KB of shared memory, 3 CTAs/SM) are for 1-32 tokens: deep async weight and activation rings
# with small accumulators, to test whether small batches are latency-bound (MF10/MF11).
# (gated, tt, nt, P): decode to shared memory + SS-wgmma (E16); N = 8 nt pairs per pass at most (64, or 128 for tt 1).
# M19 (E19): (gated, tt, nt, P, MB): M14's best stages with the code ring in shared memory, 3 or 4 CTAs per SM
M19_INSTANCES = ((1, 2, 2, 2, 3), (1, 2, 2, 2, 4), (0, 2, 2, 4, 3), (0, 2, 2, 4, 4),
                 (1, 2, 4, 2, 3), (1, 2, 4, 2, 4), (0, 2, 4, 2, 3), (0, 2, 4, 2, 4))  # E20: 32-pair passes for the hot launch
M18_INSTANCES = ((1, 2, 8, 2), (1, 2, 4, 2), (1, 2, 4, 4), (0, 2, 8, 2), (0, 2, 4, 2), (0, 1, 16, 2), (0, 1, 8, 2))
# v2 (E13): (gated, tt, nt, R, P, DB), shared memory sized for 2 CTAs/SM (3 for the NT 1 ones).
M17_INSTANCES = ((1, 2, 8, 2, 4, 0), (1, 2, 4, 3, 4, 1), (0, 1, 16, 2, 2, 1), (0, 1, 8, 4, 4, 1), (0, 2, 8, 2, 4, 1),
                 (1, 2, 1, 2, 4, 1), (0, 2, 1, 2, 4, 1))
M15_INSTANCES = ((1, 2, 8, 2, 4), (1, 2, 4, 3, 4), (0, 1, 16, 2, 3), (0, 1, 8, 4, 4), (0, 2, 8, 2, 4),
                 (1, 2, 1, 2, 4), (0, 2, 1, 2, 4), (1, 2, 2, 2, 4), (0, 2, 2, 2, 4))
# The M11 instances M11_CONFIGS uses (E3c/E4 picks); compiled next to M13 as the fallback.
# E17: + nt2 d4 ks2 for GEMM1 and GEMM2 (decode 32 runs E3c's d2: 2 K chunks in flight per warp; ~35% more time per
# expert than decode 16's d4 config)
# E21: M11's continuous-ring kernel for the small classes' incumbent shapes (gated, tt, nt, d, ks)
M13_M11_CR_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (1, 2, 2, 2, 4), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4), (0, 2, 2, 2, 4))
# E24: M11b = M11's kernel on the M13 bank and decode, for the small classes' incumbent shapes (gated, tt, nt, d, ks)
M11B_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4),
                  (1, 2, 2, 2, 4), (0, 2, 2, 2, 4), (1, 2, 2, 4, 2), (0, 2, 2, 4, 2))  # E25: 16-pair passes for 17-32 tokens
# E30 (MF32): early-refill builds of the classes' shapes, (gated, tt, nt, d, ks, minb): minb 1 = uncapped, 4 = 128 registers
M11B_EARLY_INSTANCES = tuple(sh + (mb,) for sh in ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4))
                             for mb in (1, 4))
# E29: M11b recon shapes (the classes' GEMM1 / GEMM2 picks), each in modes 1-3 (loads only / no MMA / no decode)
M11B_RECON_SHAPES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4))
# E27 (MF29): M11b at 5-6 CTAs per SM, (gated, tt, nt, d, ks, minb): __launch_bounds__(128, minb) caps registers (~96 at
# 5, ~80 at 6) so 20-24 warps per SM hide the decode chain's latency (16 today at 128 registers); d 2 halves the ring's
# registers. Dispatched with depth | minb << 4.
M11B_OCC_INSTANCES = ((1, 2, 1, 4, 4, 5), (1, 2, 1, 2, 4, 5), (1, 2, 1, 2, 4, 6),
                      (1, 2, 1, 4, 2, 5), (1, 2, 1, 2, 2, 5), (1, 2, 1, 2, 2, 6),
                      (0, 1, 1, 4, 4, 5), (0, 1, 1, 2, 4, 6),
                      (0, 2, 1, 4, 4, 5), (0, 2, 1, 2, 4, 5), (0, 2, 1, 2, 4, 6),
                      (1, 2, 1, 4, 2, 4), (1, 2, 1, 4, 4, 4), (0, 2, 1, 4, 4, 4))  # E28: the 136-138-register shapes at 4
# E22: M11's shared-memory-ring kernel (DW = 8) for the same shapes
M13_M11_SR_INSTANCES = M13_M11_CR_INSTANCES
M13_M11_INSTANCES = ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (1, 2, 2, 2, 4), (1, 2, 2, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4),
                     (0, 2, 2, 2, 4), (0, 2, 2, 4, 2))
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

// Loads of data the previous grid wrote. E5: with __ldg (ld.global.nc, treated as invariant) they could
// be hoisted above griddepcontrol.wait and every PDL config produced garbage; E8: volatile asm loads
// with a memory clobber fixed that but pinned every load in place (~10% slower). Now the wait returns a
// zero token that the callers add to those pointers: the loads depend on the wait through data, so they
// can't move above it, and are otherwise free to be scheduled (non-volatile ld.global.cg, coherent at L2).
__device__ __forceinline__ uint32_t pdl_wait_token() {
  uint32_t z;
  asm volatile("griddepcontrol.wait;\n\tmov.u32 %0, 0;" : "=r"(z) : : "memory");
  return z;
}
__device__ __forceinline__ uint4 ld_cg4(const void* p) {
  uint4 v;
  asm("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}

// A position in a warp's range: active expert ai (expert e), task, pair pass, K group; passes and
// the valid pairs of this pass (cntp) follow from e.
struct Cur {
  int ai, e, task, pass, grp, passes, cntp;
};

// w: [LOCAL][rt_total][KG][32 lanes] uint4 (M13 bit layout); sf: [LOCAL][rt_total][GPT][32] uint2 =
// 4 chunks of (row g, row g + 8) E4M3 scales; scr: FP32 [pairs][16 rt_total], zero between calls;
// ctr: [LOCAL][per_expert][MAXPASS] K groups done, zero between calls.
// MODE (recon only when not 0): 1 = the same loads, cursor and flushes, loaded words only XORed together;
// 2 = loads + decode, no MMA (decoded words XORed); 3 = loads + MMA on raw words, no decode. Recon modes
// write garbage to out but keep scratch and counters clean (the flushes are unchanged).
// CG (GEMM2 launched with PDL only): activations are loads ordered after griddepcontrol.wait (ld.global.cg +
// the wait's token); otherwise __ldg, which caches them in L1 (every warp of an expert re-reads the same
// rows; E8/E9 lost ~10 us at bs1 when GEMM1 read them through L2 only).
template <int TT, int NT, bool GATED, int MODE = 0, bool CG = false>
__global__ void __launch_bounds__(128) sk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf, const char* __restrict__ nxw, const char* __restrict__ nxs,
    long long nxwb, long long nxsb, int nxwarps) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(!(GATED && CG), "GEMM1 is never launched with PDL");
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
  // The cursor of flat group index gi (s_base[0] <= gi < total).
  auto locate = [&](int gi) {
    Cur c;
    int ai = 0;
    while (ai + 1 < nact && s_base[ai + 1] <= gi) ++ai;
    c.ai = ai;
    set_expert(c);
    const int local = gi - s_base[ai], per_task = c.passes * GPT;
    c.task = local / per_task;
    const int rem = local - c.task * per_task;
    c.pass = rem / GPT;
    c.grp = rem - c.pass * GPT;
    set_cntp(c);
    return c;
  };
  Cur L = locate(g0);  // the group whose loads are issued next

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

  uint32_t tok = 0u;  // zero; after the PDL wait it carries the dependency on the wait (see pdl_wait_token)
  uint32_t sink = 0u;  // recon modes only
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
        if constexpr (CG) {
          ra[s][n][0] = ld_cg4(la[n] + s * 8 + tok);
          ra[s][n][1] = ld_cg4(la[n] + s * 8 + 1 + tok);
        } else {
          ra[s][n][0] = __ldg(la[n] + s * 8);
          ra[s][n][1] = __ldg(la[n] + s * 8 + 1);
        }
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
        uint32_t a0, a1, a2, a3;
        if constexpr (MODE == 3) {  // recon: MMA on raw words
          a0 = q;
          a1 = q ^ sg;
          a2 = q;
          a3 = q ^ sg8;
        } else {
          a0 = hmul2(rotr(q, 2) & PM, sg);
          a1 = hmul2(rotr(q, 6) & PM, sg8);
          a2 = hmul2(rotr(q, 12) & PM, sg);
          a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (n * 8 < cntp) {
            const uint32_t b0 = word(ra[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(ra[s][n][j >> 1], (2 * j + 1) & 3);
            if constexpr (MODE == 2) {  // recon: decode, no MMA
              sink ^= a0 ^ a1 ^ a2 ^ a3 ^ b0 ^ b1;
            } else {
              mma(acc[i][n], a0, a1, a2, a3, b0, b1);
            }
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
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
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
  // Start-up L2 prefetches (weights only, so they may run before the PDL wait). pf bits 0-7: the next
  // npre groups of this warp's own range (codes and scales), one group per lane.
  {
    const int npre = pf & 0xFF;
    for (int k = 1 + lane; k <= npre && g0 + k < g1; k += 32) {
      const Cur q = locate(g0 + k);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(q, i);
        l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
        l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
      }
    }
  }
  // pf bits 24-26 = q: the next kernel's weights. Its warps j < nxwarps split the active experts' code bytes
  // (nxw, nxwb per expert, experts in s_active order) and scale bytes (nxs, nxsb) into equal ranges; this
  // grid prefetches the first q/4 of every range (one range per lane), so the next kernel's first loads hit
  // L2. Each piece stays inside one expert's region; sizes and offsets are multiples of 16.
  {
    const int nxq = (pf >> 24) & 7;
    if (nxq > 0 && nxwarps > 0) {
      auto comb = [&](const char* base, long long per, int j) {
        const long long total = static_cast<long long>(nact) * per;
        long long b = (total * j / nxwarps) & ~15LL;
        const long long b1 = total * (j + 1) / nxwarps;
        long long e = b + ((b1 - b) * nxq) / 4;
        e = min((e + 15) & ~15LL, total);
        while (b < e) {
          const int ai = static_cast<int>(b / per);
          const long long off = b - static_cast<long long>(ai) * per;
          const long long len = min(min(e - b, per - off), 32768LL);
          l2_prefetch(base + static_cast<long long>(s_active[ai]) * per + off, static_cast<uint32_t>(len));
          b += len;
        }
      };
      for (int j = gw + nw * lane; j < nxwarps; j += nw * 32) {
        comb(nxw, nxwb, j);
        comb(nxs, nxsb, j);
      }
    }
  }
  if (pdl) tok = pdl_wait_token();
#pragma unroll
  for (int s = 0; s < D; ++s) load_a(s);

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur P = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(P);
      ++pg;
      prefetch_group(P);
    }
  }
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
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(P);
      ++pg;
      prefetch_group(P);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (MODE != 1) {
        compute(s, rsc, cntp);
      } else {
#pragma unroll
        for (int i = 0; i < TT; ++i) sink ^= rw[s][i].x ^ rw[s][i].y ^ rw[s][i].z ^ rw[s][i].w ^ rsc[i].x ^ rsc[i].y;
#pragma unroll
        for (int n = 0; n < NT; ++n) sink ^= ra[s][n][0].x ^ ra[s][n][1].w;
      }
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
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the loads
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
  const uint32_t ptok = pdl ? pdl_wait_token() : 0u;
  float v[TOP_K][8];
  float ss[TOP_K];
#pragma unroll
  for (int k = 0; k < TOP_K; ++k) {
    ss[k] = 0.0f;
    uint4 r = make_uint4(0u, 0u, 0u, 0u);
    if (loc[k]) r = ld_cg4(reinterpret_cast<const uint4*>(y + static_cast<size_t>(tok * TOP_K + k) * K) + tid + ptok);
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

// Recon: stream the active experts' contiguous code regions (len16 uint4 each, region e at base +
// e * len16) with U independent 16-B loads per lane in flight; nothing but an XOR per load.
template <int U>
__global__ void __launch_bounds__(128) stream_kernel(const uint4* __restrict__ base, const int* __restrict__ experts,
                                                     int nact, long long len16, unsigned* __restrict__ sink_out) {
  const int lane = threadIdx.x & 31;
  const int gw = blockIdx.x * 4 + (threadIdx.x >> 5), nwarp = gridDim.x * 4;
  const int idx = gw % nact, parts = (nwarp + nact - 1) / nact, part = gw / nact;
  const long long b0 = len16 * part / parts, b1 = len16 * (part + 1) / parts;
  const uint4* src = base + static_cast<long long>(experts[idx]) * len16;
  uint32_t x = 0u;
  for (long long i = b0 + lane; i < b1; i += 32 * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; ++u) v[u] = i + 32 * u < b1 ? __ldcs(src + i + 32 * u) : make_uint4(0u, 0u, 0u, 0u);
#pragma unroll
    for (int u = 0; u < U; ++u) x ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (x == 0x9E3779B9u) sink_out[0] = x;
}

}  // namespace m13

// ------------------------------------------------------------------------------------------------
// M11b (E24): M11's gemm_kernel (its work split, ring, split-K reduction and epilogue, byte for byte otherwise)
// reading the M13 bank and decoding with M13's rotate-and-mask layout (~13 instead of ~19 instructions per
// 8 weights): MF26 points at instruction issue as M11's small-batch limit. Generated from moe/m11/m11.py.
namespace m11b {
using m11::K;
using m11::KG;
using m11::LOCAL;
using m11::TOP_K;
using m11::MAXP;
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::group_sync;
using m11::pdl_launch;
using m11::pdl_wait_token;
using m11::ld_cg4;
using m13::rotr;
using m13::scale2;
using m13::PM;
using m13::PM3;
using m13::PM3W;

// E28 (MF30): split-K group barrier. IMMB = immediate barrier ids (1, or 1/2 for two groups) so ptxas reserves only
// barriers 0-2; a register id (group_sync) makes it reserve all 16, which may cap residency at 4 CTAs per SM.
template <int ID>
__device__ __forceinline__ void bar_imm(int threads) {
  asm volatile("bar.sync %0, %1;" ::"n"(ID), "r"(threads) : "memory");
}
template <int KS, bool IMMB>
__device__ __forceinline__ void group_bar(int grp) {
  if constexpr (!IMMB) {
    group_sync(grp + 1, KS * 32);
  } else if constexpr (KS == 4) {
    bar_imm<1>(128);
  } else {
    static_assert(KS == 2, "immediate barriers for 1 or 2 groups");
    if (grp == 0) bar_imm<1>(64); else bar_imm<2>(64);
  }
}

// E29: MODE 0 = full; recon modes (timing only, output not valid): 1 = loads only (weights, scales, activations
// folded into a sink), 2 = loads + decode without MMA, 3 = loads + MMA on raw code words (no decode).
// E30 (MF32): EARLY = each ring step copies its slot out, refills the slot, then decodes the copy, so a warp's next
// loads never wait behind its own decode (+TT*5 + NT*8 registers for the copy).
template <int TT, int NT, int D, int KS, bool GATED, bool PDL, bool IMMB, int MODE = 0, bool EARLY = false>
__device__ __forceinline__ void gemm_body(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int sel) {
  // sel (E26, T34): hot/cold split as M14's pf bits 16-23/30: thr = sel & 0xFFFF (0 = every expert), bit 16 = hot
  // side (experts with > thr pairs; else <= thr). A gated PDL instance is only the hot launch after a cold GEMM1;
  // every CTA waits, so this grid never completes before the previous one (the next kernels rely on that).
  constexpr int GROUPS = 4 / KS;  // tasks a CTA works on at once
  constexpr int KGW = KG / KS;    // K chunks per warp
  static_assert(KGW % D == 0, "depth must divide each warp's K chunks");
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(D % 2 == 0, "ring slot s must have its chunk's parity (scale byte pair)");
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ float s_red[KS > 1 ? 4 : 1][TT * NT * 4][32];
  const int tid = threadIdx.x;
  if (tid == 0) pdl_launch();  // a no-op unless the next kernel is PDL-launched
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
    const int thr = sel & 0xFFFF, cnt = s_cnt[tid];
    const bool active = cnt > 0 && (thr == 0 || (((sel >> 16) & 1) ? cnt > thr : cnt <= thr));
    const unsigned m = __ballot_sync(0xffffffffu, active);
    if (active) s_active[__popc(m & ((1u << tid) - 1u))] = tid;
    if (tid == 0) s_nactive = __popc(m);
  }
  __syncthreads();
  uint32_t tok = 0u;  // zero; with PDL it carries the dependency on the wait into the activation loads
  if constexpr (PDL) tok = pdl_wait_token();  // every CTA waits (this grid never completes before the previous one)

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
    const uint2* sp[TT];  // M13 scale layout: one uint2 per lane per 4-chunk group
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t base = (static_cast<size_t>(e) * rt_total + rts[i]) * KG * 32 + lane;
      wp[i] = w + base;
      sp[i] = sf + (static_cast<size_t>(e) * rt_total + rts[i]) * (KG / 4) * 32 + lane;
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
          const uint2 sv = __ldg(sp[i] + (kg >> 2) * 32);  // the group's scales (L1 after its first chunk)
          rs[s][i] = (kg & 2) ? sv.y : sv.x;               // chunks 0-1 of a group: x, 2-3: y (as M14)
        }
#pragma unroll
        for (int n = 0; n < NT; ++n) {
          if (aok[n]) {
            if constexpr (PDL) {
              ra[s][n][0] = ld_cg4(ap[n] + kg * 8 + tok);
              ra[s][n][1] = ld_cg4(ap[n] + kg * 8 + 1 + tok);
            } else {
              ra[s][n][0] = __ldg(ap[n] + kg * 8);
              ra[s][n][1] = __ldg(ap[n] + kg * 8 + 1);
            }
          } else {
            ra[s][n][0] = make_uint4(0u, 0u, 0u, 0u);
            ra[s][n][1] = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      };
      uint32_t sink = 0u;  // E29 recon modes: keeps the skipped work's inputs alive
      auto compute = [&](int s) {
        if constexpr (MODE == 1) {
#pragma unroll
          for (int i = 0; i < TT; ++i)
            sink ^= rw[s][i].x ^ rw[s][i].y ^ rw[s][i].z ^ rw[s][i].w ^ rs[s][i];
#pragma unroll
          for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int h = 0; h < 2; ++h) sink ^= ra[s][n][h].x ^ ra[s][n][h].y ^ ra[s][n][h].z ^ ra[s][n][h].w;
          return;
        }
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          const uint32_t sg = (s & 1) ? scale2<2>(rs[s][i]) : scale2<0>(rs[s][i]);  // kg & 1 == s & 1 (D even)
          const uint32_t sg8 = (s & 1) ? scale2<3>(rs[s][i]) : scale2<1>(rs[s][i]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t q = word(rw[s][i], j);
            uint32_t a0, a1, a2, a3;
            if constexpr (MODE == 3) {  // raw bits as A (one op per register, the scale kept as an input)
              a0 = q; a1 = q ^ sg; a2 = q ^ sg8; a3 = q + sg;
            } else {
              a0 = hmul2(rotr(q, 2) & PM, sg), a1 = hmul2(rotr(q, 6) & PM, sg8);  // M13 bit layout
              a2 = hmul2(rotr(q, 12) & PM, sg), a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
            }
            if constexpr (MODE == 2) {
              sink ^= a0 ^ a1 ^ a2 ^ a3;
#pragma unroll
              for (int n = 0; n < NT; ++n) sink ^= ra[s][n][j >> 1].x ^ ra[s][n][j >> 1].w;
            } else {
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
        }
      };
      // E30: decode + MMA of one chunk held outside the ring (MODE 0 only)
      auto compute_on = [&](const uint4 (&W)[TT], const uint32_t (&S)[TT], const uint4 (&A)[NT][2], int s) {
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          const uint32_t sg = (s & 1) ? scale2<2>(S[i]) : scale2<0>(S[i]);  // kg & 1 == s & 1 (D even)
          const uint32_t sg8 = (s & 1) ? scale2<3>(S[i]) : scale2<1>(S[i]);
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint32_t q = word(W[i], j);
            const uint32_t a0 = hmul2(rotr(q, 2) & PM, sg), a1 = hmul2(rotr(q, 6) & PM, sg8);
            const uint32_t a2 = hmul2(rotr(q, 12) & PM, sg), a3 = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
#pragma unroll
            for (int n = 0; n < NT; ++n) {
              if (n * 8 < cnt) {
                const uint32_t b0 = word(A[n][j >> 1], (2 * j) & 3);
                const uint32_t b1 = word(A[n][j >> 1], (2 * j + 1) & 3);
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
          if constexpr (EARLY && MODE == 0) {
            uint4 cw[TT];
            uint32_t csr[TT];
            uint4 ca[NT][2];
#pragma unroll
            for (int i = 0; i < TT; ++i) { cw[i] = rw[s][i]; csr[i] = rs[s][i]; }
#pragma unroll
            for (int n = 0; n < NT; ++n) { ca[n][0] = ra[s][n][0]; ca[n][1] = ra[s][n][1]; }
            if (kg0 + s + D < KGW) load(s, kbeg + kg0 + s + D);
            compute_on(cw, csr, ca, s);
          } else {
            compute(s);
            if (kg0 + s + D < KGW) load(s, kbeg + kg0 + s + D);
          }
        }
      }

      if constexpr (MODE == 1 || MODE == 2) acc[0][0][0] += __uint_as_float(sink & 0x007FFFFFu);  // E29: keep the sink
      if constexpr (KS > 1) {  // warp kpart 0 of the group adds the others' partial sums
        if (kpart > 0) {
#pragma unroll
          for (int i = 0; i < TT; ++i)
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
              for (int r = 0; r < 4; ++r) s_red[warp][(i * NT + n) * 4 + r][lane] = acc[i][n][r];
        }
        group_bar<KS, IMMB>(grp);
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
        group_bar<KS, IMMB>(grp);  // the next chunk may overwrite s_red
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

// E28: the incumbents keep plain __launch_bounds__(128) (E24-E26 code; E27's explicit (128, 1) likely changed it, MF30);
// occupancy builds cap registers for MINB CTAs per SM and use immediate barrier ids.
template <int TT, int NT, int D, int KS, bool GATED, bool PDL = false>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int sel) {
  gemm_body<TT, NT, D, KS, GATED, PDL, false>(w, sf, cs, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, sel);
}

template <int TT, int NT, int D, int KS, bool GATED, bool PDL, int MINB>
__global__ void __launch_bounds__(128, MINB) occ_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int sel) {
  gemm_body<TT, NT, D, KS, GATED, PDL, true>(w, sf, cs, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, sel);
}

// E29: recon kernel (plain launch bounds, as gemm_kernel), MODE 1-3; timing only.
template <int TT, int NT, int D, int KS, bool GATED, int MODE>
__global__ void __launch_bounds__(128) recon_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int sel) {
  gemm_body<TT, NT, D, KS, GATED, false, false, MODE>(w, sf, cs, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, sel);
}

// E30: early-refill builds (immediate barrier ids, __launch_bounds__(128, MINB))
template <int TT, int NT, int D, int KS, bool GATED, bool PDL, int MINB>
__global__ void __launch_bounds__(128, MINB) early_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int sel) {
  gemm_body<TT, NT, D, KS, GATED, PDL, true, 0, true>(w, sf, cs, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, sel);
}

template <int TT, int NT, int D, int KS, bool GATED, bool PDL, int MB, bool E = false>
constexpr auto kern() {
  if constexpr (E) return &early_kernel<TT, NT, D, KS, GATED, PDL, MB>;
  else if constexpr (MB > 1) return &occ_kernel<TT, NT, D, KS, GATED, PDL, MB>;
  else return &gemm_kernel<TT, NT, D, KS, GATED, PDL>;
}

}  // namespace m11b

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
constexpr int SB = 3;                    // activation stages of M15 (M14: P + 1)

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

// P: activation chunks staged ahead of the one being computed (P + 1 stages; E9's kernel was P = 2).
// MODE (recon only when not 0), as M13's: 1 loads + staging only, 2 no MMA, 3 no decode.
template <int TT, int NT, bool GATED, int P, int MODE = 0>
__global__ void __launch_bounds__(128) ck_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= 8, "activation lead");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 1;  // activation stages
  extern __shared__ __align__(128) unsigned char s_b[];  // [SBP][NB pairs][128 B]
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
    // E18 hot/cold split: pf bits 16-23 = T (0 = every expert), bit 30 = side (0: experts with <= T pairs,
    // 1: experts with > T pairs), so one GEMM can run as two launches over disjoint experts (MF19).
    const int thr = (pf >> 16) & 0xFF;
    const bool act = cnt > 0 && (thr == 0 || (((pf >> 30) & 1) ? cnt > thr : cnt <= thr));
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
  uint32_t sink = 0u;  // recon modes only
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
        if constexpr (MODE == 1 || MODE == 3) {  // recon: raw words
          A[i][j][0] = q;
          A[i][j][1] = q ^ sg;
          A[i][j][2] = q;
          A[i][j][3] = q ^ sg8;
        } else {
          A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
          A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
          A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
          A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
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
          for (int i = 0; i < TT; ++i) {
            if constexpr (MODE == 1 || MODE == 2) {  // recon: no MMA
              sink ^= A[i][j][0] ^ A[i][j][1] ^ A[i][j][2] ^ A[i][j][3] ^ b0 ^ b1;
            } else {
              mma(acc[i][n], A[i][j][0], A[i][j][1], A[i][j][2], A[i][j][3], b0, b1);
            }
          }
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
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
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
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait<P - 1>();
      __syncthreads();  // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      issue_b();        // chunk + P -> the stage the previous chunk used
      compute(s, rsc, cntp, cslot);
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m14

// ------------------------------------------------------------------------------------------------
// M16: M14's CTA (one warpgroup), units, scan, activation ring and stream-K flush, with the MMAs done by
// RS-wgmma (PTX ISA 9.4 wgmma.mma_async m64nNk16 bf16, A = the 4 warps' decoded 16-row tiles in registers,
// B = the activation stage through a K-major 128B-swizzle matrix descriptor, N = this pass's pairs rounded
// up to 8). Weights come from M16's natural-K bank (m16_shuffle; B is read in natural K order); scales are
// one uint2 per (tile, chunk, row g) with byte j = 16-K block j, so every k16 step has its own scale.
namespace m16 {
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

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
// This thread's cp.async (generic proxy) writes of a stage -> ordered before wgmma's async-proxy reads.
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;" ::: "memory"); }
#ifndef M13_NO_WGMMA  // the fallback build (ptxas rejected the wgmma code) compiles none of it
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_wait0() { asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory"); }
__device__ __forceinline__ void wg_wait1() { asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory"); }
#else
__device__ __forceinline__ void wg_wait1() {}
__device__ __forceinline__ void wg_fence() {}
__device__ __forceinline__ void wg_commit() {}
__device__ __forceinline__ void wg_wait0() {}
#endif
// Byte b of v (an E4M3 scale, normal and positive) -> bf16x2 (s * 2^64, s * 2^64), as m13::scale2.
__device__ __forceinline__ uint32_t scale2r(uint32_t v, int b) {
  return __byte_perm(v, 0u, 0x4440u | b) * 0x00100010u + 0x5C005C00u;
}
// K-major, 128B-swizzle matrix descriptor of the k16 slice j of a 1024-B-aligned stage (NB rows of 128 B):
// start address >> 4, LBO 1 (unused for swizzled K-major), SBO 1024 B >> 4 (8-row groups), swizzle 1 = 128B.
__device__ __forceinline__ uint64_t b_desc(uint32_t addr, int j) {
  return (static_cast<uint64_t>(((addr + 32u * j) & 0x3FFFFu) >> 4)) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
__device__ __forceinline__ void wg_n8(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %9, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n16(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %13, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n24(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n32(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %21, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n40(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %25, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23}, %24, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n48(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n56(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %33, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, {%28, %29, %30, %31}, %32, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n64(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n72(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %41, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, {%36, %37, %38, %39}, %40, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n80(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %45, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, {%40, %41, %42, %43}, %44, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n88(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %49, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, {%44, %45, %46, %47}, %48, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n96(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %53, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, {%48, %49, %50, %51}, %52, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n104(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %57, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, {%52, %53, %54, %55}, %56, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n112(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %61, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, {%56, %57, %58, %59}, %60, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n120(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %65, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, {%60, %61, %62, %63}, %64, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
__device__ __forceinline__ void wg_n128(float* d, const uint32_t (&a)[4], uint64_t desc) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
#endif
}
// D (+)= A B for the first 8 nv pair columns; acc is [NT][4] (column block, mma.sync C order = wgmma's).
template <int NT>
__device__ __forceinline__ void wg_mma(int nv, float (&acc)[NT][4], const uint32_t (&a)[4], uint64_t desc) {
  switch (nv) {
    case 1: if constexpr (NT >= 1) wg_n8(&acc[0][0], a, desc); break;
    case 2: if constexpr (NT >= 2) wg_n16(&acc[0][0], a, desc); break;
    case 3: if constexpr (NT >= 3) wg_n24(&acc[0][0], a, desc); break;
    case 4: if constexpr (NT >= 4) wg_n32(&acc[0][0], a, desc); break;
    case 5: if constexpr (NT >= 5) wg_n40(&acc[0][0], a, desc); break;
    case 6: if constexpr (NT >= 6) wg_n48(&acc[0][0], a, desc); break;
    case 7: if constexpr (NT >= 7) wg_n56(&acc[0][0], a, desc); break;
    case 8: if constexpr (NT >= 8) wg_n64(&acc[0][0], a, desc); break;
    case 9: if constexpr (NT >= 9) wg_n72(&acc[0][0], a, desc); break;
    case 10: if constexpr (NT >= 10) wg_n80(&acc[0][0], a, desc); break;
    case 11: if constexpr (NT >= 11) wg_n88(&acc[0][0], a, desc); break;
    case 12: if constexpr (NT >= 12) wg_n96(&acc[0][0], a, desc); break;
    case 13: if constexpr (NT >= 13) wg_n104(&acc[0][0], a, desc); break;
    case 14: if constexpr (NT >= 14) wg_n112(&acc[0][0], a, desc); break;
    case 15: if constexpr (NT >= 15) wg_n120(&acc[0][0], a, desc); break;
    case 16: if constexpr (NT >= 16) wg_n128(&acc[0][0], a, desc); break;
    default: break;
  }
}

// N as a template argument -> the wrapper above.
template <int N>
__device__ __forceinline__ void wg_n(float* d, const uint32_t (&a)[4], uint64_t desc) {
  if constexpr (N == 8) wg_n8(d, a, desc);
  else if constexpr (N == 16) wg_n16(d, a, desc);
  else if constexpr (N == 24) wg_n24(d, a, desc);
  else if constexpr (N == 32) wg_n32(d, a, desc);
  else if constexpr (N == 40) wg_n40(d, a, desc);
  else if constexpr (N == 48) wg_n48(d, a, desc);
  else if constexpr (N == 56) wg_n56(d, a, desc);
  else if constexpr (N == 64) wg_n64(d, a, desc);
  else if constexpr (N == 72) wg_n72(d, a, desc);
  else if constexpr (N == 80) wg_n80(d, a, desc);
  else if constexpr (N == 88) wg_n88(d, a, desc);
  else if constexpr (N == 96) wg_n96(d, a, desc);
  else if constexpr (N == 104) wg_n104(d, a, desc);
  else if constexpr (N == 112) wg_n112(d, a, desc);
  else if constexpr (N == 120) wg_n120(d, a, desc);
  else if constexpr (N == 128) wg_n128(d, a, desc);
}
// One chunk's wgmmas at N (4 k16 steps x TT tiles), with no branch between them.
template <int N, int TT, int NT>
__device__ __forceinline__ void wg_steps(float (&acc)[TT][NT][4], const uint32_t (&A)[TT][4][4], uint32_t sbase) {
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint64_t desc = b_desc(sbase, j);
#pragma unroll
    for (int i = 0; i < TT; ++i) wg_n<N>(&acc[i][0][0], A[i][j], desc);
  }
}
// A chunk at N = 8 nv: one CTA-uniform switch per chunk (v1 switched before every wgmma).
template <int TT, int NT>
__device__ __forceinline__ void wg_chunk(int nv, float (&acc)[TT][NT][4], const uint32_t (&A)[TT][4][4], uint32_t sbase) {
  switch (nv) {
    case 1: if constexpr (NT >= 1) wg_steps<8>(acc, A, sbase); break;
    case 2: if constexpr (NT >= 2) wg_steps<16>(acc, A, sbase); break;
    case 3: if constexpr (NT >= 3) wg_steps<24>(acc, A, sbase); break;
    case 4: if constexpr (NT >= 4) wg_steps<32>(acc, A, sbase); break;
    case 5: if constexpr (NT >= 5) wg_steps<40>(acc, A, sbase); break;
    case 6: if constexpr (NT >= 6) wg_steps<48>(acc, A, sbase); break;
    case 7: if constexpr (NT >= 7) wg_steps<56>(acc, A, sbase); break;
    case 8: if constexpr (NT >= 8) wg_steps<64>(acc, A, sbase); break;
    case 9: if constexpr (NT >= 9) wg_steps<72>(acc, A, sbase); break;
    case 10: if constexpr (NT >= 10) wg_steps<80>(acc, A, sbase); break;
    case 11: if constexpr (NT >= 11) wg_steps<88>(acc, A, sbase); break;
    case 12: if constexpr (NT >= 12) wg_steps<96>(acc, A, sbase); break;
    case 13: if constexpr (NT >= 13) wg_steps<104>(acc, A, sbase); break;
    case 14: if constexpr (NT >= 14) wg_steps<112>(acc, A, sbase); break;
    case 15: if constexpr (NT >= 15) wg_steps<120>(acc, A, sbase); break;
    case 16: if constexpr (NT >= 16) wg_steps<128>(acc, A, sbase); break;
    default: break;
  }
}
// The accumulators as a compiler fence (CUTLASS's warpgroup_fence_operand on accum): no instruction that reads or
// writes them may be moved across this point, so nothing touches them between wgmma.fence and the wgmmas, and the
// flush can't read them before the wait. No code is emitted.
template <int TT, int NT>
__device__ __forceinline__ void fence_acc(float (&acc)[TT][NT][4]) {
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) asm volatile("" : "+f"(acc[i][n][r])::"memory");
}
// Keeps a buffer's registers live (and untouched by the compiler) up to here, where its wgmmas are known done:
// CUTLASS's warpgroup_fence_operand, so the register allocator can't reuse them while a wgmma may read them.
template <int TT>
__device__ __forceinline__ void fence_regs(uint32_t (&A)[TT][4][4]) {
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) asm volatile("" : "+r"(A[i][j][k])::"memory");
}


template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }

// P: activation chunks staged ahead (P + 1 stages). NT: 8-pair column blocks per pass (NB = 8 NT <= 128).
// sf: [LOCAL][rt_total][KG][8] uint2 (M16 scales); w: [LOCAL][rt_total][KG][32] uint4 (natural-K words).
// DB: two A buffers, wgmma.wait_group 1 (a warp decodes chunk c while its chunk c - 1 MMAs run) and one more
// activation stage. MODE (recon only when not 0): 1 = decode but no wgmma, 2 = wgmma on raw words (no decode).
template <int TT, int NT, bool GATED, int P, bool DB = false, int MODE = 0>
__global__ void __launch_bounds__(128) wg_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= 8, "activation lead");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 1 + (DB ? 1 : 0);  // activation stages (DB: a chunk's MMAs may still read it one chunk later)
  static_assert(NT >= 1 && NT <= 16, "N = 8 NT <= 128");
  extern __shared__ __align__(128) unsigned char s_b[];  // 1024 B of slack, then [SBP][NB pairs][128 B] at 1024 B
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
      ls[i] = sf + (row * KG + c.grp * D) * 8 + g;
    }
  };
  const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;  // the 128B swizzle pattern starts at 1024 B
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
  uint2 rs[D][TT];  // scales of the chunk in ring slot s: (row g, row g + 8), byte j = 16-K block j
  float acc[TT][NT][4];
  uint32_t Ab[DB ? 2 : 1][TT][4][4];  // A fragments (DB: chunk parity)
  uint32_t sink = 0u;                 // recon modes only
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  auto load_w = [&](int s) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      rw[s][i] = __ldcs(lw[i] + s * 32);
      rs[s][i] = __ldcs(ls[i] + s * 8);
    }
  };
  // Chunk s: decode the 4 k16 steps of every tile into A fragments (natural K, the step's own scale), then
  // one wgmma per (step, tile) on the stage; the caller waits for them before the A registers or the stage
  // are reused (wg_wait0) and before reading acc (flush).
  auto compute = [&](int s, int cntp, int slot) {
    auto& A = Ab[DB ? (s & 1) : 0];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rw[s][i], j);
        const uint32_t sg = scale2r(rs[s][i].x, j), sg8 = scale2r(rs[s][i].y, j);
        if constexpr (MODE == 2) {  // recon: raw words
          A[i][j][0] = q;
          A[i][j][1] = q ^ sg;
          A[i][j][2] = q;
          A[i][j][3] = q ^ sg8;
        } else {
          A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
          A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
          A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
          A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
      }
    }
    const uint32_t sbase = sb0 + slot * STAGE;
    const int nv = (cntp + 7) >> 3;  // 8-pair column blocks this pass uses (1..NT), CTA-uniform
    if constexpr (MODE == 1) {  // recon: no wgmma
#pragma unroll
      for (int i = 0; i < TT; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j) sink ^= A[i][j][0] ^ A[i][j][1] ^ A[i][j][2] ^ A[i][j][3];
    } else {
      fence_regs<TT>(A);       // the decode is complete before wgmma.fence (it can't sink between fence and wgmma)
      fence_acc<TT, NT>(acc);  // so is any write of acc (the flush's zeroing)
      wg_fence();
      wg_chunk<TT, NT>(nv, acc, A, sbase);
      wg_commit();
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
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
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
  if (pdl) {
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * KG + q.grp * D) * 8, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (DB) {
        wg_wait1();       // this warp's wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage are free
        fence_regs<TT>(Ab[s & 1]);
      } else {
        wg_wait0();       // this warp's wgmmas of the previous chunk are done: its A registers and stage are free
      }
      fence_acc<TT, NT>(acc);
      cp_wait<P - 1>();
      fence_async_smem();  // this thread's copies of this chunk -> visible to wgmma (async proxy)
      __syncthreads();    // this chunk is in shared memory for all; no warp still reads the previous chunk's stage
      issue_b();          // chunk + P -> the stage chunk - 1 (DB: chunk - 2) used
      compute(s, cntp, cslot);
      if (more) load_w(s);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      wg_wait0();  // acc complete
      fence_acc<TT, NT>(acc);
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m16

// ------------------------------------------------------------------------------------------------
// M15: large batches, one decode per weight per call. As M14 (a CTA's 4 warps share each unit; the
// pass's activations staged per K chunk in a swizzled cp.async ring), but each warp's weights come in
// through its own ring of R shared-memory slots of one K group (TT tiles x (2 KB codes + 256 B scales)),
// filled by cp.async.bulk with mbarrier expect-tx (M12's machinery), so the bytes in flight cost no
// registers and the accumulators can cover 8 NT pairs (64 for GEMM1, 128 for GEMM2) in one pass.
// Units are ordered (expert, pass, 4-task block, K group) and split over CTAs by cost: a unit costs
// 8 + beta * (its pass's 8-pair tiles), so CTAs on the hot expert get fewer units (beta = pf >> 16).
namespace m15 {
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m11::mbar_init;
using m11::mbar_init_fence;
using m11::mbar_arrive_expect_tx;
using m11::mbar_wait;
using m11::bulk_g2s;
using m11::fence_proxy_async;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;
using m14::cp_async16;
using m14::cp_commit;
using m14::cp_wait1;
using m14::cp_wait;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes (shared with M13/M14)
// activation stages: P + 1 (P = chunks staged ahead; E9's kernel had P = 2)
constexpr int TILE_BYTES = D * 512 + 256;  // one tile's K group: 4 chunks of codes, then its scales

template <int TT, int NT, int R, int P>
constexpr int dyn_bytes() { return 4 * R * TT * TILE_BYTES + (P + 1) * 8 * NT * 128; }

template <int TT, int NT, bool GATED, int R, int P>
__global__ void __launch_bounds__(128) wk_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = NB * 128;
  constexpr int SLOT = TT * TILE_BYTES;
  constexpr int SBP = P + 1;  // activation stages
  static_assert(P >= 2 && P <= 8, "activation lead");
  extern __shared__ __align__(128) unsigned char smem[];  // [4 warps][R][SLOT] weights, then [SB][NB][128] activations
  unsigned char* s_b = smem + 4 * R * SLOT;
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_ubase[LOCAL + 1];
  __shared__ int s_cbase[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int beta = (pf >> 16) & 0xFF;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[warp][r]), 1);
    mbar_init_fence();
  }
  __syncthreads();
  // Pair lists in pair order with 4 warps (as M14).
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
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_rb = per_expert / 4;
  const int blk = n_rb * GPT;  // units per (expert, pass)
  auto pass_cost = [&](int cnt, int p) { return 8 + beta * ((min(cnt - p * NB, NB) + 7) / 8); };
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int passes = act ? (cnt + NB - 1) / NB : 0;
    int cost = 0;
    for (int p = 0; p < passes; ++p) cost += pass_cost(cnt, p);
    int u = passes * blk, cc = cost * blk;
    int ui = u, ci = cc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a1 = __shfl_up_sync(0xffffffffu, ui, o);
      const int a2 = __shfl_up_sync(0xffffffffu, ci, o);
      if (lane >= o) {
        ui += a1;
        ci += a2;
      }
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_ubase[ai] = ui - u;
      s_cbase[ai] = ci - cc;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_ubase[__popc(am)] = ui;
      s_cbase[__popc(am)] = ci;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total_cost = s_cbase[nact];
  // A cost position -> the first unit that starts at or after it (CTA ranges in cost space, cut at units).
  auto cost_to_unit = [&](long long pos) -> int {
    if (pos >= total_cost) return s_ubase[nact];
    int ai = 0;
    while (ai + 1 < nact && s_cbase[ai + 1] <= pos) ++ai;
    const int cnt = min(s_cnt[s_active[ai]], MAXP);
    long long rem = pos - s_cbase[ai];
    int p = 0;
    while (true) {
      const int c = pass_cost(cnt, p);
      if (rem < static_cast<long long>(c) * blk) return s_ubase[ai] + p * blk + static_cast<int>((rem + c - 1) / c);
      rem -= static_cast<long long>(c) * blk;
      ++p;
    }
  };
  const int g0 = cost_to_unit(static_cast<long long>(total_cost) * blockIdx.x / gridDim.x);
  const int g1 = cost_to_unit(static_cast<long long>(total_cost) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // order: group, 4-task block (c.task), pass, expert
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.task == n_rb) {
        c.task = 0;
        if (++c.pass == c.passes) {
          c.pass = 0;
          if (++c.ai < nact) set_expert(c);
        }
        if (c.ai < nact) set_cntp(c);
      }
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_ubase[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_ubase[ai];
    L.pass = local / blk;
    const int rem = local - L.pass * blk;
    L.task = rem / GPT;
    L.grp = rem - L.task * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint32_t ring = smem_addr(smem + warp * (R * SLOT));
  const uint32_t bar0 = smem_addr(&bars[warp][0]);
  const unsigned char* myring = smem + warp * (R * SLOT);
  // Weight producer: lane 0 fills slot (seq % R) with group c of this warp's tiles.
  auto issue_w = [&](const Cur& c, int slot) {
    if (lane == 0) {
      const uint32_t bar = bar0 + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(c, i);
        const uint32_t dst = ring + slot * SLOT + i * TILE_BYTES;
        bulk_g2s(dst, w + (row * KG + c.grp * D) * 32, D * 512, bar);
        bulk_g2s(dst + D * 512, sf + (row * GPT + c.grp) * 32, 256, bar);
      }
    }
  };
  const uint32_t sb0 = smem_addr(s_b);
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  // Chunk s of the warp's weight slot -> decoded A registers, then MMAs over the pass's valid n-tiles.
  auto compute = [&](int s, const unsigned char* wslot, int cntp, int bslot) {
    uint32_t A[TT][4][4];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const unsigned char* tp = wslot + i * TILE_BYTES;
      const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);
      const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + lane * 8);
      const uint32_t sv = s < 2 ? sc.x : sc.y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t qw = word(q4, j);
        A[i][j][0] = hmul2(rotr(qw, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(qw, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(qw, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(qw, 8) & PM3) | (qw & PM3W), sg8);
      }
    }
    const unsigned char* sb = s_b + bslot * STAGE;
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
                  float* qp = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(qp, 0.0f);  // the sum, read and cleared in one atomic
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

  // Prologue: the first R groups' weights (they don't depend on the previous grid), then the wait, then
  // the first two activation chunks.
  Cur Wc = L;  // the next group whose weights are issued
  int wg = g0;
  for (int r = 0; r < R && wg < g1; ++r) {
    issue_w(Wc, r);
    ++wg;
    if (wg < g1) advance(Wc);
  }
  if (pdl) pdl_wait();
  // Activation producer, P chunks ahead of compute (as M14's): Bc = the group of the next chunk to stage, bsub
  // its chunk, bg its flat index, pslot its stage; past the range it only commits empty groups.
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    const int slot = (gi - g0) % R;
    if (more) advance(L);
    mbar_wait(bar0 + 8 * slot, ((gi - g0) / R) & 1);
    const unsigned char* wslot = myring + slot * SLOT;
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait<P - 1>();
      __syncthreads();    // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      issue_b();          // chunk + P -> the stage the previous chunk used
      compute(s, wslot, cntp, cslot);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    // The warp is done with this slot: refill it with the group R ahead.
    __syncwarp();
    if (wg < g1) {
      if (lane == 0) fence_proxy_async();
      issue_w(Wc, slot);
      ++wg;
      if (wg < g1) advance(Wc);
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m15

// ------------------------------------------------------------------------------------------------
// M17: M15 (per-warp TMA weight ring, activation ring P ahead, cost-weighted stream-K split) with M16's
// RS-wgmma: each chunk's A fragments are decoded from the warp's weight slot into registers (so the slot can be
// refilled as in M15), then one wgmma per (k16 step, tile) reads the activation stage through M16's descriptor.
// Natural-K bank (M16's): a tile's K group is still 4 x 512 B of codes then 256 B of scales, now 4 chunks x
// 8 rows g x uint2 (row g, row g + 8; byte j = 16-K block j).
namespace m17 {
using m16::wg_mma;
using m16::wg_fence;
using m16::wg_commit;
using m16::wg_wait0;
using m16::wg_wait1;
using m16::wg_chunk;
using m16::fence_regs;
using m16::fence_acc;
using m16::b_desc;
using m16::scale2r;
using m16::fence_async_smem;
using m11::hmul2;
using m11::mma;
using m11::word;
using m11::smem_addr;
using m11::mbar_init;
using m11::mbar_init_fence;
using m11::mbar_arrive_expect_tx;
using m11::mbar_wait;
using m11::bulk_g2s;
using m11::fence_proxy_async;
using m13::rotr;
using m13::scale2;
using m13::pdl_wait;
using m13::pdl_launch;
using m13::Cur;
using m13::PM;
using m13::PM3;
using m13::PM3W;
using m14::cp_async16;
using m14::cp_commit;
using m14::cp_wait1;
using m14::cp_wait;

constexpr int K = 4096;
constexpr int KG = K / 64;
constexpr int D = 4;
constexpr int GPT = KG / D;
constexpr int LOCAL = 32;
constexpr int TOP_K = 8;
constexpr int MAXP = @@M14_MAXP@@;
constexpr int MAXPASS = (MAXP + 7) / 8;  // counters are laid out for 8-pair passes (shared with M13/M14)
// activation stages: P + 1 (P = chunks staged ahead; E9's kernel had P = 2)
constexpr int TILE_BYTES = D * 512 + 256;  // one tile's K group: 4 chunks of codes, then its scales

template <int TT, int NT, int R, int P, bool DB>
constexpr int dyn_bytes() { return 4 * R * TT * TILE_BYTES + 1024 + (P + 1 + (DB ? 1 : 0)) * 8 * NT * 128; }  // +1024: alignment

// DB as M16's: two A buffers, wgmma.wait_group 1, one more activation stage.
template <int TT, int NT, bool GATED, int R, int P, bool DB = false>
__global__ void __launch_bounds__(128) wq_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = NB * 128;
  constexpr int SLOT = TT * TILE_BYTES;
  constexpr int SBP = P + 1 + (DB ? 1 : 0);  // activation stages (DB: a chunk's MMAs may still read it one chunk later)
  static_assert(NT >= 1 && NT <= 16, "N = 8 NT <= 128");
  static_assert(P >= 2 && P <= 8, "activation lead");
  extern __shared__ __align__(128) unsigned char smem[];  // [4 warps][R][SLOT] weights, then [SB][NB][128] activations
  unsigned char* s_b = smem + 4 * R * SLOT;
  __shared__ __align__(8) uint64_t bars[4][R];
  __shared__ int s_wcnt[4][LOCAL];
  __shared__ int s_cnt[LOCAL];
  __shared__ short s_list[LOCAL][MAXP];
  __shared__ int s_active[LOCAL];
  __shared__ int s_ubase[LOCAL + 1];
  __shared__ int s_cbase[LOCAL + 1];
  __shared__ int s_nactive;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int beta = (pf >> 16) & 0xFF;
  if (tid == 0) pdl_launch();
  s_wcnt[warp][lane] = 0;
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) mbar_init(smem_addr(&bars[warp][r]), 1);
    mbar_init_fence();
  }
  __syncthreads();
  // Pair lists in pair order with 4 warps (as M14).
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
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_rb = per_expert / 4;
  const int blk = n_rb * GPT;  // units per (expert, pass)
  auto pass_cost = [&](int cnt, int p) { return 8 + beta * ((min(cnt - p * NB, NB) + 7) / 8); };
  if (warp == 0) {
    const int cnt = min(s_cnt[lane], MAXP);
    const bool act = cnt > 0;
    const unsigned am = __ballot_sync(0xffffffffu, act);
    const int passes = act ? (cnt + NB - 1) / NB : 0;
    int cost = 0;
    for (int p = 0; p < passes; ++p) cost += pass_cost(cnt, p);
    int u = passes * blk, cc = cost * blk;
    int ui = u, ci = cc;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int a1 = __shfl_up_sync(0xffffffffu, ui, o);
      const int a2 = __shfl_up_sync(0xffffffffu, ci, o);
      if (lane >= o) {
        ui += a1;
        ci += a2;
      }
    }
    if (act) {
      const int ai = __popc(am & lt);
      s_active[ai] = lane;
      s_ubase[ai] = ui - u;
      s_cbase[ai] = ci - cc;
    }
    if (lane == 31) {
      s_nactive = __popc(am);
      s_ubase[__popc(am)] = ui;
      s_cbase[__popc(am)] = ci;
    }
  }
  __syncthreads();

  const int nact = s_nactive;
  const int total_cost = s_cbase[nact];
  // A cost position -> the first unit that starts at or after it (CTA ranges in cost space, cut at units).
  auto cost_to_unit = [&](long long pos) -> int {
    if (pos >= total_cost) return s_ubase[nact];
    int ai = 0;
    while (ai + 1 < nact && s_cbase[ai + 1] <= pos) ++ai;
    const int cnt = min(s_cnt[s_active[ai]], MAXP);
    long long rem = pos - s_cbase[ai];
    int p = 0;
    while (true) {
      const int c = pass_cost(cnt, p);
      if (rem < static_cast<long long>(c) * blk) return s_ubase[ai] + p * blk + static_cast<int>((rem + c - 1) / c);
      rem -= static_cast<long long>(c) * blk;
      ++p;
    }
  };
  const int g0 = cost_to_unit(static_cast<long long>(total_cost) * blockIdx.x / gridDim.x);
  const int g1 = cost_to_unit(static_cast<long long>(total_cost) * (blockIdx.x + 1) / gridDim.x);
  if (g0 >= g1) {
    if (pdl) pdl_wait();
    return;
  }

  auto set_expert = [&](Cur& c) {
    c.e = s_active[c.ai];
    c.passes = (min(s_cnt[c.e], MAXP) + NB - 1) / NB;
  };
  auto set_cntp = [&](Cur& c) { c.cntp = min(min(s_cnt[c.e], MAXP) - c.pass * NB, NB); };
  auto advance = [&](Cur& c) {  // order: group, 4-task block (c.task), pass, expert
    if (++c.grp == GPT) {
      c.grp = 0;
      if (++c.task == n_rb) {
        c.task = 0;
        if (++c.pass == c.passes) {
          c.pass = 0;
          if (++c.ai < nact) set_expert(c);
        }
        if (c.ai < nact) set_cntp(c);
      }
    }
  };
  Cur L;
  {
    int ai = 0;
    while (ai + 1 < nact && s_ubase[ai + 1] <= g0) ++ai;
    L.ai = ai;
    set_expert(L);
    const int local = g0 - s_ubase[ai];
    L.pass = local / blk;
    const int rem = local - L.pass * blk;
    L.task = rem / GPT;
    L.grp = rem - L.task * GPT;
    set_cntp(L);
  }

  auto wtask = [&](const Cur& c) { return c.task * 4 + warp; };
  auto tile_rt = [&](const Cur& c, int i) { return GATED ? wtask(c) + i * (rt_total / 2) : wtask(c) * TT + i; };
  auto tile_row = [&](const Cur& c, int i) -> size_t { return static_cast<size_t>(c.e) * rt_total + tile_rt(c, i); };
  const uint32_t ring = smem_addr(smem + warp * (R * SLOT));
  const uint32_t bar0 = smem_addr(&bars[warp][0]);
  const unsigned char* myring = smem + warp * (R * SLOT);
  // Weight producer: lane 0 fills slot (seq % R) with group c of this warp's tiles.
  auto issue_w = [&](const Cur& c, int slot) {
    if (lane == 0) {
      const uint32_t bar = bar0 + 8 * slot;
      mbar_arrive_expect_tx(bar, SLOT);
#pragma unroll
      for (int i = 0; i < TT; ++i) {
        const size_t row = tile_row(c, i);
        const uint32_t dst = ring + slot * SLOT + i * TILE_BYTES;
        bulk_g2s(dst, w + (row * KG + c.grp * D) * 32, D * 512, bar);
        bulk_g2s(dst + D * 512, sf + (row * KG + c.grp * D) * 8, 256, bar);  // M16 scales: 4 chunks x 8 x uint2
      }
    }
  };
  const uint32_t sb0 = (smem_addr(s_b) + 1023u) & ~1023u;  // the 128B swizzle pattern starts at 1024 B
  auto load_b = [&](const Cur& c, int s, int slot) {
    const int kg = c.grp * D + s;
#pragma unroll
    for (int k = 0; k < (NB * 8 + 127) / 128; ++k) {
      const int seg = tid + 128 * k;
      const int pr = seg >> 3, part = seg & 7;
      if (seg < NB * 8 && pr < ((c.cntp + 7) & ~7)) {
        const bool ok = pr < c.cntp;
        const int p = ok ? s_list[c.e][c.pass * NB + pr] : 0;
        const int row = ok ? (a_per_pair ? p : p / TOP_K) : 0;
        cp_async16(sb0 + slot * STAGE + pr * 128 + ((part ^ (pr & 7)) << 4),
                   a + static_cast<size_t>(row) * K + kg * 64 + part * 8, ok ? 16u : 0u);
      }
    }
  };

  float acc[TT][NT][4];
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  // Chunk s of the warp's weight slot -> A fragments (natural K, each k16 step its own scale), then one wgmma per
  // (step, tile) on activation stage bslot; waited for before the stage is reused (wg_wait0) and before flush.
  uint32_t Ab[DB ? 2 : 1][TT][4][4];  // A fragments (DB: chunk parity)
  auto compute = [&](int s, const unsigned char* wslot, int cntp, int bslot) {
    auto& A = Ab[DB ? (s & 1) : 0];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const unsigned char* tp = wslot + i * TILE_BYTES;
      const uint4 q4 = *reinterpret_cast<const uint4*>(tp + s * 512 + lane * 16);
      const uint2 sc = *reinterpret_cast<const uint2*>(tp + D * 512 + s * 64 + g * 8);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t qw = word(q4, j);
        const uint32_t sg = scale2r(sc.x, j), sg8 = scale2r(sc.y, j);
        A[i][j][0] = hmul2(rotr(qw, 2) & PM, sg);
        A[i][j][1] = hmul2(rotr(qw, 6) & PM, sg8);
        A[i][j][2] = hmul2(rotr(qw, 12) & PM, sg);
        A[i][j][3] = hmul2((rotr(qw, 8) & PM3) | (qw & PM3W), sg8);
      }
    }
    const uint32_t sbase = sb0 + bslot * STAGE;
    const int nv = (cntp + 7) >> 3;  // 8-pair column blocks this pass uses (1..NT), CTA-uniform
    fence_regs<TT>(A);       // as M16: decode and acc writes complete before wgmma.fence
    fence_acc<TT, NT>(acc);
    wg_fence();
    wg_chunk<TT, NT>(nv, acc, A, sbase);
    wg_commit();
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
                  float* qp = prow + 16 * tile_rt(c, i) + g + 8 * rr;
                  acc[i][n][2 * rr + h] = atomicExch(qp, 0.0f);  // the sum, read and cleared in one atomic
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

  // Prologue: the first R groups' weights (they don't depend on the previous grid), then the wait, then
  // the first two activation chunks.
  Cur Wc = L;  // the next group whose weights are issued
  int wg = g0;
  for (int r = 0; r < R && wg < g1; ++r) {
    issue_w(Wc, r);
    ++wg;
    if (wg < g1) advance(Wc);
  }
  if (pdl) pdl_wait();
  // Activation producer, P chunks ahead of compute (as M14's): Bc = the group of the next chunk to stage, bsub
  // its chunk, bg its flat index, pslot its stage; past the range it only commits empty groups.
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    const int slot = (gi - g0) % R;
    if (more) advance(L);
    mbar_wait(bar0 + 8 * slot, ((gi - g0) / R) & 1);
    const unsigned char* wslot = myring + slot * SLOT;
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      if constexpr (DB) {
        wg_wait1();       // this warp's wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage are free
        fence_regs<TT>(Ab[s & 1]);
      } else {
        wg_wait0();       // this warp's wgmmas of the previous chunk are done: that stage is free
      }
      fence_acc<TT, NT>(acc);
      cp_wait<P - 1>();
      fence_async_smem();  // this thread's copies of this chunk -> visible to wgmma (async proxy)
      __syncthreads();    // this chunk is in shared memory for all; no warp still reads the previous chunk's stage
      issue_b();          // chunk + P -> the stage the previous chunk used
      compute(s, wslot, cntp, cslot);
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    // The warp is done with this slot: refill it with the group R ahead.
    __syncwarp();
    if (wg < g1) {
      if (lane == 0) fence_proxy_async();
      issue_w(Wc, slot);
      ++wg;
      if (wg < g1) advance(Wc);
    }
    if (C.grp == GPT - 1 || !more) {
      wg_wait0();  // acc complete
      fence_acc<TT, NT>(acc);
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
}

}  // namespace m17

// ------------------------------------------------------------------------------------------------
// M18 (E16): M14's CTA units, scan, register weight ring, activation stages and stream-K flush, but each chunk's
// weights are decoded once into a 64-row bf16 A tile in shared memory (row-chunk bank: each word = 8 consecutive K
// of one row, stored with one conflict-free STS.128), and SS-wgmma reads A and the activation stage through K-major
// 128B-swizzle descriptors. Double-buffered A + P + 2 activation stages + wgmma.wait_group 1: the warps decode chunk
// c while the tensor cores run chunk c - 1. N = this pass's pairs rounded up to 8 (up to 64, or 128 for TT 1).
namespace m18 {
using m16::wg_wait0;
using m16::wg_wait1;
using m16::fence_acc;
using m16::scale2r;
using m16::b_desc;
using m16::fence_async_smem;
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

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }
constexpr int ATILE = 64 * 128;  // one 64-row x 64-K bf16 A tile
template <int TT, int NT, int P>
constexpr int dyn_bytes() { return 1024 + 2 * TT * ATILE + (P + 2) * stage_bytes<NT>(); }  // +1024: alignment
__device__ __forceinline__ void ss2_n8(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %11, 0;\n"
      "mov.b64 rb0, %10;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %8;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %9;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%4, %5, %6, %7}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%4, %5, %6, %7}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%4, %5, %6, %7}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%4, %5, %6, %7}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n16(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %19, 0;\n"
      "mov.b64 rb0, %18;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %16;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %17;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%8, %9, %10, %11, %12, %13, %14, %15}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%8, %9, %10, %11, %12, %13, %14, %15}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%8, %9, %10, %11, %12, %13, %14, %15}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%8, %9, %10, %11, %12, %13, %14, %15}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n24(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %27, 0;\n"
      "mov.b64 rb0, %26;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %24;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %25;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n32(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %35, 0;\n"
      "mov.b64 rb0, %34;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %32;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %33;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11]), "+f"(d1[12]), "+f"(d1[13]), "+f"(d1[14]), "+f"(d1[15])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n40(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %43, 0;\n"
      "mov.b64 rb0, %42;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %40;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %41;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11]), "+f"(d1[12]), "+f"(d1[13]), "+f"(d1[14]), "+f"(d1[15]), "+f"(d1[16]), "+f"(d1[17]), "+f"(d1[18]), "+f"(d1[19])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n48(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %51, 0;\n"
      "mov.b64 rb0, %50;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %48;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %49;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11]), "+f"(d1[12]), "+f"(d1[13]), "+f"(d1[14]), "+f"(d1[15]), "+f"(d1[16]), "+f"(d1[17]), "+f"(d1[18]), "+f"(d1[19]), "+f"(d1[20]), "+f"(d1[21]), "+f"(d1[22]), "+f"(d1[23])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n56(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %59, 0;\n"
      "mov.b64 rb0, %58;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %56;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %57;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11]), "+f"(d1[12]), "+f"(d1[13]), "+f"(d1[14]), "+f"(d1[15]), "+f"(d1[16]), "+f"(d1[17]), "+f"(d1[18]), "+f"(d1[19]), "+f"(d1[20]), "+f"(d1[21]), "+f"(d1[22]), "+f"(d1[23]), "+f"(d1[24]), "+f"(d1[25]), "+f"(d1[26]), "+f"(d1[27])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss2_n64(float* d0, float* d1, uint64_t a0, uint64_t a1, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %67, 0;\n"
      "mov.b64 rb0, %66;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %64;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "mov.b64 ra4, %65;\n"
      "add.s64 ra5, ra4, 2;\n"
      "add.s64 ra6, ra4, 4;\n"
      "add.s64 ra7, ra4, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra4, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra5, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra6, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra7, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d1[0]), "+f"(d1[1]), "+f"(d1[2]), "+f"(d1[3]), "+f"(d1[4]), "+f"(d1[5]), "+f"(d1[6]), "+f"(d1[7]), "+f"(d1[8]), "+f"(d1[9]), "+f"(d1[10]), "+f"(d1[11]), "+f"(d1[12]), "+f"(d1[13]), "+f"(d1[14]), "+f"(d1[15]), "+f"(d1[16]), "+f"(d1[17]), "+f"(d1[18]), "+f"(d1[19]), "+f"(d1[20]), "+f"(d1[21]), "+f"(d1[22]), "+f"(d1[23]), "+f"(d1[24]), "+f"(d1[25]), "+f"(d1[26]), "+f"(d1[27]), "+f"(d1[28]), "+f"(d1[29]), "+f"(d1[30]), "+f"(d1[31])
      : "l"(a0), "l"(a1), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n8(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "mov.b64 rb0, %5;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %4;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n16(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "mov.b64 rb0, %9;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %8;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n24(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %14, 0;\n"
      "mov.b64 rb0, %13;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %12;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n32(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "mov.b64 rb0, %17;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %16;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n40(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %22, 0;\n"
      "mov.b64 rb0, %21;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %20;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n40k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n48(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %26, 0;\n"
      "mov.b64 rb0, %25;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %24;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n56(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %30, 0;\n"
      "mov.b64 rb0, %29;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %28;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n56k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n64(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %34, 0;\n"
      "mov.b64 rb0, %33;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %32;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n72(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %38, 0;\n"
      "mov.b64 rb0, %37;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %36;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n72k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n80(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %42, 0;\n"
      "mov.b64 rb0, %41;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %40;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n80k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n88(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %46, 0;\n"
      "mov.b64 rb0, %45;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %44;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n88k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n96(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %50, 0;\n"
      "mov.b64 rb0, %49;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %48;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n96k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43]), "+f"(d0[44]), "+f"(d0[45]), "+f"(d0[46]), "+f"(d0[47])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n104(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %54, 0;\n"
      "mov.b64 rb0, %53;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %52;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n104k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43]), "+f"(d0[44]), "+f"(d0[45]), "+f"(d0[46]), "+f"(d0[47]), "+f"(d0[48]), "+f"(d0[49]), "+f"(d0[50]), "+f"(d0[51])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n112(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %58, 0;\n"
      "mov.b64 rb0, %57;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %56;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n112k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43]), "+f"(d0[44]), "+f"(d0[45]), "+f"(d0[46]), "+f"(d0[47]), "+f"(d0[48]), "+f"(d0[49]), "+f"(d0[50]), "+f"(d0[51]), "+f"(d0[52]), "+f"(d0[53]), "+f"(d0[54]), "+f"(d0[55])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n120(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %62, 0;\n"
      "mov.b64 rb0, %61;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %60;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n120k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43]), "+f"(d0[44]), "+f"(d0[45]), "+f"(d0[46]), "+f"(d0[47]), "+f"(d0[48]), "+f"(d0[49]), "+f"(d0[50]), "+f"(d0[51]), "+f"(d0[52]), "+f"(d0[53]), "+f"(d0[54]), "+f"(d0[55]), "+f"(d0[56]), "+f"(d0[57]), "+f"(d0[58]), "+f"(d0[59])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
__device__ __forceinline__ void ss1_n128(float* d0, float*, uint64_t a0, uint64_t, uint64_t b) {
#ifndef M13_NO_WGMMA
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      ".reg .b64 ra<8>;\n"
      ".reg .b64 rb<4>;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "mov.b64 rb0, %65;\n"
      "add.s64 rb1, rb0, 2;\n"
      "add.s64 rb2, rb0, 4;\n"
      "add.s64 rb3, rb0, 6;\n"
      "mov.b64 ra0, %64;\n"
      "add.s64 ra1, ra0, 2;\n"
      "add.s64 ra2, ra0, 4;\n"
      "add.s64 ra3, ra0, 6;\n"
      "wgmma.fence.sync.aligned;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra0, rb0, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra1, rb1, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra2, rb2, p, 1, 1, 0, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, ra3, rb3, p, 1, 1, 0, 0;\n"
      "wgmma.commit_group.sync.aligned;\n"
      "}\n"
      : "+f"(d0[0]), "+f"(d0[1]), "+f"(d0[2]), "+f"(d0[3]), "+f"(d0[4]), "+f"(d0[5]), "+f"(d0[6]), "+f"(d0[7]), "+f"(d0[8]), "+f"(d0[9]), "+f"(d0[10]), "+f"(d0[11]), "+f"(d0[12]), "+f"(d0[13]), "+f"(d0[14]), "+f"(d0[15]), "+f"(d0[16]), "+f"(d0[17]), "+f"(d0[18]), "+f"(d0[19]), "+f"(d0[20]), "+f"(d0[21]), "+f"(d0[22]), "+f"(d0[23]), "+f"(d0[24]), "+f"(d0[25]), "+f"(d0[26]), "+f"(d0[27]), "+f"(d0[28]), "+f"(d0[29]), "+f"(d0[30]), "+f"(d0[31]), "+f"(d0[32]), "+f"(d0[33]), "+f"(d0[34]), "+f"(d0[35]), "+f"(d0[36]), "+f"(d0[37]), "+f"(d0[38]), "+f"(d0[39]), "+f"(d0[40]), "+f"(d0[41]), "+f"(d0[42]), "+f"(d0[43]), "+f"(d0[44]), "+f"(d0[45]), "+f"(d0[46]), "+f"(d0[47]), "+f"(d0[48]), "+f"(d0[49]), "+f"(d0[50]), "+f"(d0[51]), "+f"(d0[52]), "+f"(d0[53]), "+f"(d0[54]), "+f"(d0[55]), "+f"(d0[56]), "+f"(d0[57]), "+f"(d0[58]), "+f"(d0[59]), "+f"(d0[60]), "+f"(d0[61]), "+f"(d0[62]), "+f"(d0[63])
      : "l"(a0), "l"(b), "r"(1));
#endif
}
// One chunk (4 k16 steps x TT tiles) of SS-wgmma at N = 8 nv, inside one asm statement: descriptors, wgmma.fence,
// the wgmmas and the commit, so the compiler can't put anything between them (E14: ptxas injected an arrive
// before most RS wgmmas, MF15).
template <int TT, int NT>
__device__ __forceinline__ void ss_chunk(int nv, float (&acc)[TT][NT][4], uint64_t a0, uint64_t a1, uint64_t b) {
  float* d1 = TT == 2 ? &acc[TT - 1][0][0] : nullptr;
  switch (nv) {
    case 1: if constexpr (NT >= 1) { if constexpr (TT == 2) ss2_n8(&acc[0][0][0], d1, a0, a1, b); else ss1_n8(&acc[0][0][0], d1, a0, a1, b); } break;
    case 2: if constexpr (NT >= 2) { if constexpr (TT == 2) ss2_n16(&acc[0][0][0], d1, a0, a1, b); else ss1_n16(&acc[0][0][0], d1, a0, a1, b); } break;
    case 3: if constexpr (NT >= 3) { if constexpr (TT == 2) ss2_n24(&acc[0][0][0], d1, a0, a1, b); else ss1_n24(&acc[0][0][0], d1, a0, a1, b); } break;
    case 4: if constexpr (NT >= 4) { if constexpr (TT == 2) ss2_n32(&acc[0][0][0], d1, a0, a1, b); else ss1_n32(&acc[0][0][0], d1, a0, a1, b); } break;
    case 5: if constexpr (NT >= 5) { if constexpr (TT == 2) ss2_n40(&acc[0][0][0], d1, a0, a1, b); else ss1_n40(&acc[0][0][0], d1, a0, a1, b); } break;
    case 6: if constexpr (NT >= 6) { if constexpr (TT == 2) ss2_n48(&acc[0][0][0], d1, a0, a1, b); else ss1_n48(&acc[0][0][0], d1, a0, a1, b); } break;
    case 7: if constexpr (NT >= 7) { if constexpr (TT == 2) ss2_n56(&acc[0][0][0], d1, a0, a1, b); else ss1_n56(&acc[0][0][0], d1, a0, a1, b); } break;
    case 8: if constexpr (NT >= 8) { if constexpr (TT == 2) ss2_n64(&acc[0][0][0], d1, a0, a1, b); else ss1_n64(&acc[0][0][0], d1, a0, a1, b); } break;
    case 9: if constexpr (TT == 1 && NT >= 9) ss1_n72(&acc[0][0][0], d1, a0, a1, b); break;
    case 10: if constexpr (TT == 1 && NT >= 10) ss1_n80(&acc[0][0][0], d1, a0, a1, b); break;
    case 11: if constexpr (TT == 1 && NT >= 11) ss1_n88(&acc[0][0][0], d1, a0, a1, b); break;
    case 12: if constexpr (TT == 1 && NT >= 12) ss1_n96(&acc[0][0][0], d1, a0, a1, b); break;
    case 13: if constexpr (TT == 1 && NT >= 13) ss1_n104(&acc[0][0][0], d1, a0, a1, b); break;
    case 14: if constexpr (TT == 1 && NT >= 14) ss1_n112(&acc[0][0][0], d1, a0, a1, b); break;
    case 15: if constexpr (TT == 1 && NT >= 15) ss1_n120(&acc[0][0][0], d1, a0, a1, b); break;
    case 16: if constexpr (TT == 1 && NT >= 16) ss1_n128(&acc[0][0][0], d1, a0, a1, b); break;
    default: break;
  }
}


// w: [LOCAL][rt_total][KG][32] uint4 (row-chunk words), sf: [LOCAL][rt_total][GPT][32] uint2 (M18 scales).
// MODE (recon only when not 0, E17): 1 = weight ring + activation stages + barrier only (no decode, no wgmma);
// 2 = decode + store, no wgmma; 3 = wgmma on a stale A tile, no decode; 4 = full, but __syncwarp for the barrier.
// Recon modes write garbage but keep scratch and counters clean (the flushes are unchanged).
template <int TT, int NT, bool GATED, int P, int MODE = 0>
__global__ void __launch_bounds__(128) ss_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= 8, "activation lead");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 2;  // activation stages (a chunk's wgmmas may still read its stage one chunk later)
  static_assert(NT >= 1 && (TT == 2 ? NT <= 8 : NT <= 16), "N = 8 NT <= 64 (TT 2) or 128 (TT 1)");
  extern __shared__ __align__(128) unsigned char s_raw[];  // slack, then at 1024 B: [2][TT] A tiles, [SBP] stages
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
    // E18 hot/cold split: pf bits 16-23 = T (0 = every expert), bit 30 = side (0: experts with <= T pairs,
    // 1: experts with > T pairs), so one GEMM can run as two launches over disjoint experts (MF19).
    const int thr = (pf >> 16) & 0xFF;
    const bool act = cnt > 0 && (thr == 0 || (((pf >> 30) & 1) ? cnt > thr : cnt <= thr));
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
  unsigned char* s_a = s_raw + ((1024u - (smem_addr(s_raw) & 1023u)) & 1023u);  // the 128B swizzle starts at 1024 B
  const uint32_t sa0 = smem_addr(s_a);
  const uint32_t sb0 = sa0 + 2 * TT * ATILE;
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
  uint32_t sink = 0u;  // recon modes only
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
  // Chunk s of this warp's tiles -> A buffer par: lane (r = lane / 2, hf = lane % 2) decodes its 4 words (row r,
  // K 8 (4 hf + m) .. + 7; one scale per word) and stores each as 16 B at row 16 warp + r, segment 4 hf + m.
  const int arow = 16 * warp + (lane >> 1);
  const uint32_t aoff = (arow & 7) * 128 + (arow >> 3) * 1024;
  auto decode_store = [&](int s, const uint2 (&sc)[TT], int par) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      unsigned char* tile = s_a + (par * TT + i) * ATILE + aoff;
#pragma unroll
      for (int mm = 0; mm < 4; ++mm) {
        const uint32_t q = word(rw[s][i], mm);
        const uint32_t sg = scale2r(sv, 2 * (s & 1) + (mm >> 1));
        const int seg = 4 * (lane & 1) + mm;
        uint4 v;
        v.x = hmul2(rotr(q, 2) & PM, sg);
        v.y = hmul2(rotr(q, 6) & PM, sg);
        v.z = hmul2(rotr(q, 12) & PM, sg);
        v.w = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg);
        *reinterpret_cast<uint4*>(tile + ((seg ^ (arow & 7)) << 4)) = v;
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
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
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
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      wg_wait1();          // this warp's (the warpgroup's) wgmmas of chunk - 2 are done: A buffer (s & 1) and that stage
      fence_acc<TT, NT>(acc);
      if constexpr (MODE == 1 || MODE == 3) {  // recon: no decode (keep the loads)
#pragma unroll
        for (int i = 0; i < TT; ++i) sink ^= rw[s][i].x ^ rw[s][i].y ^ rw[s][i].z ^ rw[s][i].w ^ rsc[i].x;
      } else {
        decode_store(s, rsc, s & 1);
      }
      if (more) {
        load_w(s);
        if (s == 0) load_s();
      }
      cp_wait<P - 1>();
      fence_async_smem();  // this thread's A stores and stage copies -> visible to wgmma (async proxy)
      if constexpr (MODE == 4) {
        __syncwarp();      // recon: no CTA barrier (the wgmma still syncs the warpgroup)
      } else {
        __syncthreads();   // all of chunk s's A tile and stage are written
      }
      issue_b();           // chunk + P -> the stage chunk - 2 used
      if constexpr (MODE != 1 && MODE != 2) {
        ss_chunk<TT, NT>((cntp + 7) >> 3, acc, b_desc(sa0 + ((s & 1) * TT) * ATILE, 0),
                         b_desc(sa0 + ((s & 1) * TT + TT - 1) * ATILE, 0), b_desc(sb0 + cslot * STAGE, 0));
      }
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      wg_wait0();  // acc complete
      fence_acc<TT, NT>(acc);
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m18


// M19 (E19): M14 with the weight-code ring in shared memory (cp.async, D + 1 chunk slots per warp) instead of
// 32 registers, and __launch_bounds__(128, MB), so 4 CTAs per SM may fit. Generated by moe/local/make_m19.py.
namespace m19 {
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
constexpr int SB = 3;                    // activation stages of M15 (M14: P + 1)

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void cp_wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

template <int NT>
constexpr int stage_bytes() { return 8 * NT * 128; }
// activation stages, then each warp's code ring: D + 1 chunk slots x TT tiles x 32 lanes x 16 B
template <int TT, int NT, int P>
constexpr int dyn_bytes() { return (P + 1) * stage_bytes<NT>() + 4 * (D + 1) * TT * 512; }

// P: activation chunks staged ahead of the one being computed (P + 1 stages; E9's kernel was P = 2).
// MODE (recon only when not 0), as M13's: 1 loads + staging only, 2 no MMA, 3 no decode.
template <int TT, int NT, bool GATED, int P, int MB, int MODE = 0>
__global__ void __launch_bounds__(128, MB) wr_kernel(
    const uint4* __restrict__ w, const uint2* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    float* __restrict__ scr, int* __restrict__ ctr, int n_pairs, int local_start, int rt_total,
    int a_per_pair, int out_cols, int pdl, int pf) {
  static_assert(!GATED || TT == 2, "a gated task is one gate tile and its up tile");
  static_assert(P >= 2 && P <= D, "the code ring (D chunks ahead) must cover the activation lead");
  static_assert(MODE == 0, "M19 has no recon modes");
  constexpr int NB = 8 * NT;
  constexpr int STAGE = stage_bytes<NT>();
  constexpr int SBP = P + 1;  // activation stages
  extern __shared__ __align__(128) unsigned char s_b[];  // [SBP][NB pairs][128 B]
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
    // E18 hot/cold split: pf bits 16-23 = T (0 = every expert), bit 30 = side (0: experts with <= T pairs,
    // 1: experts with > T pairs), so one GEMM can run as two launches over disjoint experts (MF19).
    const int thr = (pf >> 16) & 0xFF;
    const bool act = cnt > 0 && (thr == 0 || (((pf >> 30) & 1) ? cnt > thr : cnt <= thr));
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
  constexpr int WSL = D + 1;  // code ring slots per warp: the next group's D chunks + the chunk in use
  auto woff = [&](int slot, int i) { return SBP * STAGE + ((warp * WSL + slot) * TT + i) * 512 + lane * 16; };
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

  uint2 rsn[TT];
  float acc[TT][NT][4];
  uint32_t sink = 0u;  // recon modes only
#pragma unroll
  for (int i = 0; i < TT; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;
  auto issue_w = [&](int s, int slot) {  // chunk s of the group lw points at -> code ring slot
#pragma unroll
    for (int i = 0; i < TT; ++i) cp_async16(sb0 + woff(slot, i), lw[i] + s * 32, 16u);
  };
  auto load_s = [&]() {
#pragma unroll
    for (int i = 0; i < TT; ++i) rsn[i] = __ldcs(ls[i]);
  };
  auto compute = [&](int s, const uint2 (&sc)[TT], int cntp, int slot, int wslot) {
    uint32_t A[TT][4][4];
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const uint4 rwv = *reinterpret_cast<const uint4*>(s_b + woff(wslot, i));  // own words, landed (wait_group)
      const uint32_t sv = s < 2 ? sc[i].x : sc[i].y;
      const uint32_t sg = (s & 1) ? scale2<2>(sv) : scale2<0>(sv);
      const uint32_t sg8 = (s & 1) ? scale2<3>(sv) : scale2<1>(sv);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t q = word(rwv, j);
        if constexpr (MODE == 1 || MODE == 3) {  // recon: raw words
          A[i][j][0] = q;
          A[i][j][1] = q ^ sg;
          A[i][j][2] = q;
          A[i][j][3] = q ^ sg8;
        } else {
          A[i][j][0] = hmul2(rotr(q, 2) & PM, sg);
          A[i][j][1] = hmul2(rotr(q, 6) & PM, sg8);
          A[i][j][2] = hmul2(rotr(q, 12) & PM, sg);
          A[i][j][3] = hmul2((rotr(q, 8) & PM3) | (q & PM3W), sg8);
        }
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
          for (int i = 0; i < TT; ++i) {
            if constexpr (MODE == 1 || MODE == 2) {  // recon: no MMA
              sink ^= A[i][j][0] ^ A[i][j][1] ^ A[i][j][2] ^ A[i][j][3] ^ b0 ^ b1;
            } else {
              mma(acc[i][n], A[i][j][0], A[i][j][1], A[i][j][2], A[i][j][3], b0, b1);
            }
          }
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
                  acc[i][n][2 * rr + h] = atomicExch(q, 0.0f);  // the sum, read and cleared in one atomic
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
  for (int s = 0; s < D; ++s) issue_w(s, s);  // the first group's chunks -> slots 0..D-1 (first commit group)
  load_s();
  if (pdl) {
    if (lane == 0 && (pf & 0xFF) > 0) {
      Cur q = L;
      for (int k = 1; k <= (pf & 0xFF) && g0 + k < g1; ++k) {
        advance(q);
#pragma unroll
        for (int i = 0; i < TT; ++i) l2_prefetch(w + (tile_row(q, i) * KG + q.grp * D) * 32, D * 512);
      }
    }
    pdl_wait();
  }
  // Activation producer, P chunks ahead of compute: Bc = the group of the next chunk to stage, bsub its chunk
  // in that group, bg its flat index, pslot its stage. Past the range it only commits empty groups, so
  // cp.async.wait_group P - 1 always means "the chunk about to be computed has landed".
  Cur Bc = L;
  int bsub = 0, bg = g0, pslot = 0;
  auto issue_b = [&]() {
    if (bg < g1) load_b(Bc, bsub, pslot);
    cp_commit();
    pslot = pslot + 1 == SBP ? 0 : pslot + 1;
    if (++bsub == D) {
      bsub = 0;
      if (++bg < g1) advance(Bc);
    }
  };
#pragma unroll
  for (int k = 0; k < P; ++k) issue_b();

  // Steady-state L2 prefetch (pf >> 8 groups ahead; 0 = off): lane 0 asks for each upcoming group's codes
  // and scales with bulk prefetches, so DRAM sees 2 KB / 256 B requests instead of 512 B / 64 B loads.
  const int ahead = (pf >> 8) & 0xFF;
  Cur Pq = L;
  int pg = g0;
  auto prefetch_group = [&](const Cur& q) {
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const size_t row = tile_row(q, i);
      l2_prefetch(w + (row * KG + q.grp * D) * 32, D * 512);
      l2_prefetch(sf + (row * GPT + q.grp) * 32, 256);
    }
  };
  if (lane == 0) {
    for (int k = 0; k < ahead && pg + 1 < g1; ++k) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
  }
  Cur C = L;
  int seg0 = C.grp;
  int cslot = 0;  // stage of the chunk being computed (chunks consumed mod SBP)
  int wc = 0;     // chunks consumed: chunk wc's codes are in ring slot wc % WSL
  for (int gi = g0; gi < g1; ++gi) {
    const bool more = gi + 1 < g1;
    uint2 rsc[TT];
#pragma unroll
    for (int i = 0; i < TT; ++i) rsc[i] = rsn[i];
    if (more) {
      advance(L);
      point(L);
    }
    if (lane == 0 && ahead > 0 && pg + 1 < g1) {
      advance(Pq);
      ++pg;
      prefetch_group(Pq);
    }
    const int cntp = C.cntp;
#pragma unroll
    for (int s = 0; s < D; ++s) {
      cp_wait<P - 1>();
      __syncthreads();  // this chunk is in shared memory for all; everyone is done with the previous chunk's stage
      // the next group's chunk s, D chunks ahead, into the slot chunk wc - 1 released; issued before issue_b so
      // it joins the commit group of activation chunk wc + P <= wc + D, which wait_group covers by then
      if (more) issue_w(s, (wc + D) % WSL);
      issue_b();        // chunk + P -> the stage the previous chunk used
      compute(s, rsc, cntp, cslot, wc % WSL);
      if (more && s == 0) load_s();
      ++wc;
      cslot = cslot + 1 == SBP ? 0 : cslot + 1;
    }
    if (C.grp == GPT - 1 || !more) {
      flush(C, seg0);
      seg0 = 0;
    }
    C = L;
  }
  if constexpr (MODE != 0) {
    if (sink == 0x9E3779B9u) out[0] = __float2bfloat16_rn(1.0f);  // keeps the work
  }
}

}  // namespace m19



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
  // A refused launch (bad config) also sets the thread's last error; clear it so the failure stays with this call
  // instead of surfacing at PyTorch's next launch check (E17: one recon launch took the whole tuner down).
  if (err != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "m13 launch failed: ", cudaGetErrorString(err), " (grid ", grid, ", threads ",
              threads, ", dynamic smem ", smem, " B)");
}

#define M13_CASE(G, TT, NT) M13_CASE_CG(G, TT, NT, false, (G) || !pdl)
#define M13_CASE_P(G, TT, NT) M13_CASE_CG(G, TT, NT, true, pdl) M13_CASE(G, TT, NT)
// One kernel instance (CG fixed) when cond holds; plain GEMM2 instances come in both CG flavours.
#define M13_CASE_CG(G, TT, NT, CGV, COND)                                                                   \
  if (gated == (G) && tt == (TT) && nt == (NT) && (COND)) {                                                \
    m13_launch(m13::sk_kernel<TT, NT, G, 0, CGV>, static_cast<int>(grid), 128, 0, static_cast<int>(pdl),    \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf), reinterpret_cast<const char*>(nxw), reinterpret_cast<const char*>(nxs),  \
               static_cast<long long>(nxwb), static_cast<long long>(nxsb), static_cast<int>(nxwarps));      \
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
                       int64_t grid, int64_t pdl, int64_t pf, int64_t nxw, int64_t nxs, int64_t nxwb, int64_t nxsb,
                       int64_t nxwarps) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_gemm: ", n_pairs, " pairs, at most ", m13::MAXPAIRS);
  TORCH_CHECK(((pf >> 24) & 7) == 0 || (nxw != 0 && nxs != 0 && nxwb > 0 && nxsb > 0 && nxwb % 16 == 0 &&
                                       nxsb % 16 == 0 && nxwarps > 0), "m13_gemm: bad next-stage prefetch");
  @@M13_CASES@@
  TORCH_CHECK(false, "m13_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

#define M13_RECON_CASE(G, TT, NT, MODE)                                                                     \
  if (gated == (G) && tt == (TT) && nt == (NT) && mode == (MODE)) {                                        \
    m13_launch(m13::sk_kernel<TT, NT, G, MODE>, static_cast<int>(grid), 128, 0, 0,                         \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), 0, 0, nullptr, nullptr, 0LL, 0LL, 0); \
    return out;                                                                                            \
  }

// The M13 GEMM in a recon mode (1 loads only, 2 no MMA, 3 no decode), for recon labs.
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t mode) {
  TORCH_CHECK(n_pairs <= m13::MAXPAIRS, "m13_recon: too many pairs");
  M13_RECON_CASE(true, 2, 1, 1)
  M13_RECON_CASE(true, 2, 1, 2)
  M13_RECON_CASE(true, 2, 1, 3)
  M13_RECON_CASE(false, 2, 1, 1)
  TORCH_CHECK(false, "m13_recon: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

#define M13_STREAM_CASE(U)                                                                                  \
  if (unroll == (U)) {                                                                                     \
    m13_launch(m13::stream_kernel<U>, static_cast<int>(grid), 128, 0, 0,                                   \
               reinterpret_cast<const uint4*>(base.data_ptr()), static_cast<const int*>(experts.data_ptr<int>()), \
               static_cast<int>(experts.numel()), static_cast<long long>(len16),                           \
               reinterpret_cast<unsigned*>(sink.data_ptr<int>()));                                         \
    return sink;                                                                                           \
  }

// Contiguous streaming of the listed experts' regions of base (len16 uint4 each), for recon labs.
torch::Tensor m13_stream(torch::Tensor base, torch::Tensor experts, int64_t len16, torch::Tensor sink, int64_t unroll,
                         int64_t grid) {
  M13_STREAM_CASE(4)
  M13_STREAM_CASE(8)
  M13_STREAM_CASE(16)
  TORCH_CHECK(false, "m13_stream: unroll ", unroll);
  return sink;
}

#define M11P_CASE(TT, NT, D, KS)                                                                            \
  if (tt == (TT) && nt == (NT) && depth == (D) && ks == (KS)) {                                            \
    m13_launch(m11::gemm_kernel<TT, NT, D, KS, false, true>, static_cast<int>(grid), 128, 0, 1,             \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                \
               static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),     \
               static_cast<int>(out_cols));                                                                 \
    return out;                                                                                            \
  }

#define M11P_CR_CASE(TT, NT, D, KS)                                                                         \
  if (tt == (TT) && nt == (NT) && depth == ((D) | 16) && ks == (KS)) {                                     \
    m13_launch(m11::cr_kernel<TT, NT, D, KS, false, true>, static_cast<int>(grid), 128, 0, 1,               \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                \
               static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),     \
               static_cast<int>(out_cols));                                                                 \
    return out;                                                                                            \
  }

#define M11P_SR_CASE(TT, NT, D, KS)                                                                         \
  if (tt == (TT) && nt == (NT) && depth == ((D) | 32) && ks == (KS)) {                                     \
    m13_launch(m11::sr_kernel<TT, NT, D, KS, false, true>, static_cast<int>(grid), 128,                    \
               static_cast<size_t>(4 * 8 * (TT) * 576), 1,                                                 \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                \
               static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),     \
               static_cast<int>(out_cols));                                                                 \
    return out;                                                                                            \
  }

// E22b: M11's shared-memory-ring kernels: raise their dynamic limit once in prepare (static + dynamic > 48 KB in E22).
// E24: M11b (M11's kernel on the M13 bank and decode), launched with m13_launch; GEMM2 instances also with PDL.
#define M11B_LAUNCH(G, TT, NT, D, KS, P, MB, E)                                                             \
  m13_launch(m11b::kern<TT, NT, D, KS, G, P, MB, E>(), static_cast<int>(grid), 128, 0, (P) ? 1 : 0,             \
             reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()),     \
             static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()),    \
             reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                           \
             reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
             static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
             static_cast<int>(out_cols), static_cast<int>(sel))

torch::Tensor m11b_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t a_per_pair,
                        int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid,
                        int64_t pdl, int64_t sel) {
  @@M11B_CASES@@
  TORCH_CHECK(false, "m11b_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return out;
}

// E27: registers, spill bytes, static smem and CTAs per SM of every M11b instance (depth | minb << 4)
#define M11B_ATTR(G, TT, NT, D, KS, MB, E)                                                                 \
  if (gated == (G) && tt == (TT) && nt == (NT) && depth == ((D) | ((MB) > 1 ? (MB) << 4 : 0) | ((E) ? 256 : 0)) \
      && ks == (KS)) {                                                                                    \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11b::kern<TT, NT, D, KS, G, false, MB, E>());                           \
    int blocks = 0;                                                                                       \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m11b::kern<TT, NT, D, KS, G, false, MB, E>(), 128, 0); \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes), static_cast<int64_t>(blocks)});     \
  }

// E29: M11b recon (MODE 1-3) for the shapes the classes use (M11B_RECON_SHAPES); timing only, out is scratch.
torch::Tensor m11b_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                         torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t a_per_pair,
                         int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid,
                         int64_t sel, int64_t mode) {
  @@M11B_RECON_CASES@@
  TORCH_CHECK(false, "m11b_recon: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks,
              " mode=", mode);
  return out;
}

torch::Tensor m11b_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks) {
  @@M11B_ATTRS@@
  TORCH_CHECK(false, "m11b_attrs: no instance for gated=", gated, " tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
  return torch::Tensor();
}

torch::Tensor m11_sr_setup() {
  @@M11_SR_ATTRS@@
  std::vector<int64_t> done{1};
  return torch::tensor(done);
}

// M11's GEMM2 launched with PDL (E15): m11_gemm's arguments; it scans the routing while GEMM1 finishes.
torch::Tensor m11p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                        int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                        int64_t ks, int64_t grid) {
  TORCH_CHECK(gated == 0, "m11p_gemm: GEMM2 only");
  @@M11P_CASES@@
  TORCH_CHECK(false, "m11p_gemm: no instance for tt=", tt, " nt=", nt, " depth=", depth, " ks=", ks);
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

#define M14_CASE(G, TT, NT, P)                                                                              \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P)) {                                             \
    m13_launch(m14::ck_kernel<TT, NT, G, P>, static_cast<int>(grid), 128,                                  \
               static_cast<size_t>(((P) + 1) * m14::stage_bytes<NT>()), static_cast<int>(pdl),             \
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
// [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM].
#define M14_SETUP(G, TT, NT, P)                                                                             \
  {                                                                                                        \
    const int dyn = ((P) + 1) * m14::stage_bytes<NT>();                                                    \
    cudaFuncSetAttribute(m14::ck_kernel<TT, NT, G, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m14::ck_kernel<TT, NT, G, P>);                                            \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m14::ck_kernel<TT, NT, G, P>, 128, dyn);       \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, P, static_cast<int64_t>(attr.numRegs),                    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_gemm: ", n_pairs, " pairs, at most ", 8 * m14::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P (0 = E9's 2)
  @@M14_CASES@@
  TORCH_CHECK(false, "m14_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa);
  return out;
}

#define M14_RECON_CASE(G, TT, NT, P, MODE)                                                                  \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && mode == (MODE)) {                           \
    m13_launch(m14::ck_kernel<TT, NT, G, P, MODE>, static_cast<int>(grid), 128,                            \
               static_cast<size_t>(((P) + 1) * m14::stage_bytes<NT>()), 0,                                 \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), 0, 0);                            \
    return out;                                                                                            \
  }

// M14 GEMM1 n2 (P = 4) in a recon mode (1 loads + staging only, 2 no MMA, 3 no decode), for the tuner's recon.
torch::Tensor m14_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t pa, int64_t mode) {
  TORCH_CHECK(n_pairs <= 8 * m14::MAXP, "m14_recon: too many pairs");
  M14_RECON_CASE(true, 2, 2, 4, 1)
  M14_RECON_CASE(true, 2, 2, 4, 2)
  M14_RECON_CASE(true, 2, 2, 4, 3)
  TORCH_CHECK(false, "m14_recon: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa, " mode=", mode);
  return out;
}

torch::Tensor m14_setup() {
  std::vector<int64_t> rows;
  @@M14_SETUPS@@
  {  // E22b: the recon instances too (static + dynamic shared memory counts toward 48 KB)
    cudaFuncSetAttribute(m14::ck_kernel<2, 2, true, 4, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, 5 * m14::stage_bytes<2>());
    cudaFuncSetAttribute(m14::ck_kernel<2, 2, true, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 5 * m14::stage_bytes<2>());
    cudaFuncSetAttribute(m14::ck_kernel<2, 2, true, 4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 5 * m14::stage_bytes<2>());
  }
  return torch::tensor(rows).view({-1, 9});
}

#define M15_CASE(G, TT, NT, R, P)                                                                           \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P)) {                                             \
    m13_launch(m15::wk_kernel<TT, NT, G, R, P>, static_cast<int>(grid), 128,                               \
               static_cast<size_t>(m15::dyn_bytes<TT, NT, R, P>()), static_cast<int>(pdl),                 \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP: [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM] (R is
// fixed per (gated, tt, nt, P)).
#define M15_SETUP(G, TT, NT, R, P)                                                                          \
  {                                                                                                        \
    const int dyn = m15::dyn_bytes<TT, NT, R, P>();                                                        \
    cudaFuncSetAttribute(m15::wk_kernel<TT, NT, G, R, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m15::wk_kernel<TT, NT, G, R, P>);                                         \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m15::wk_kernel<TT, NT, G, R, P>, 128, dyn);    \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, P, static_cast<int64_t>(attr.numRegs),                    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m15_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m15::MAXP, "m15_gemm: ", n_pairs, " pairs, at most ", 8 * m15::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  @@M15_CASES@@
  TORCH_CHECK(false, "m15_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m15_setup() {
  std::vector<int64_t> rows;
  @@M15_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M17_CASE(G, TT, NT, R, P, DB)                                                                       \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && db == (DB)) {                               \
    m13_launch(m17::wq_kernel<TT, NT, G, R, P, DB>, static_cast<int>(grid), 128,                           \
               static_cast<size_t>(m17::dyn_bytes<TT, NT, R, P, DB>()), static_cast<int>(pdl),             \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP: [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM] (R is
// fixed per (gated, tt, nt, P)).
#define M17_SETUP(G, TT, NT, R, P, DB)                                                                      \
  {                                                                                                        \
    const int dyn = m17::dyn_bytes<TT, NT, R, P, DB>();                                                    \
    cudaFuncSetAttribute(m17::wq_kernel<TT, NT, G, R, P, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m17::wq_kernel<TT, NT, G, R, P, DB>);                                     \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m17::wq_kernel<TT, NT, G, R, P, DB>, 128, dyn); \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((DB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m17_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m17::MAXP, "m17_gemm: ", n_pairs, " pairs, at most ", 8 * m17::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t db = (pf >> 28) & 1;                              // double-buffered A
  @@M17_CASES@@
  TORCH_CHECK(false, "m17_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt);
  return out;
}

torch::Tensor m17_setup() {
  std::vector<int64_t> rows;
  @@M17_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M18_CASE(G, TT, NT, P)                                                                              \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P)) {                                             \
    m13_launch(m18::ss_kernel<TT, NT, G, P>, static_cast<int>(grid), 128,                                  \
               static_cast<size_t>(m18::dyn_bytes<TT, NT, P>()), static_cast<int>(pdl),                    \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP: [gated, tt, nt, P, registers, spill bytes, static smem, dynamic smem, CTAs per SM].
#define M18_SETUP(G, TT, NT, P)                                                                             \
  {                                                                                                        \
    const int dyn = m18::dyn_bytes<TT, NT, P>();                                                           \
    cudaFuncSetAttribute(m18::ss_kernel<TT, NT, G, P>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m18::ss_kernel<TT, NT, G, P>);                                            \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m18::ss_kernel<TT, NT, G, P>, 128, dyn);       \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, P, static_cast<int64_t>(attr.numRegs),                    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m18_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m18::MAXP, "m18_gemm: ", n_pairs, " pairs, at most ", 8 * m18::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  @@M18_CASES@@
  TORCH_CHECK(false, "m18_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa);
  return out;
}

// The recon instances need ~49 KB of dynamic shared memory (> the 48 KB default), so m18_setup raises their limit
// once in prepare, as M18_SETUP does for the MODE 0 instances (E17 failed with "invalid argument" without it).
#define M18_RECON_ATTR(MODE)                                                                                \
  cudaFuncSetAttribute(m18::ss_kernel<2, 4, true, 2, MODE>, cudaFuncAttributeMaxDynamicSharedMemorySize,   \
                       m18::dyn_bytes<2, 4, 2>());

torch::Tensor m18_setup() {
  std::vector<int64_t> rows;
  @@M18_SETUPS@@
  @@M18_RECON_ATTRS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M18_RECON_CASE(MODE)                                                                                \
  if (mode == (MODE)) {                                                                                    \
    m13_launch(m18::ss_kernel<2, 4, true, 2, MODE>, static_cast<int>(grid), 128,                           \
               static_cast<size_t>(m18::dyn_bytes<2, 4, 2>()), 0,                                          \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total), 0,    \
               static_cast<int>(rt_total / 2 * 16), 0, 2 << 24);                                           \
    return out;                                                                                            \
  }

// M18 GEMM1 g2.4 (P 2) in a recon mode 1-4 (E17), for the tuner's skeleton recon.
torch::Tensor m18_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode) {
  TORCH_CHECK(n_pairs <= 8 * m18::MAXP, "m18_recon: too many pairs");
  @@M18_RECON@@
  TORCH_CHECK(false, "m18_recon: mode ", mode);
  return out;
}

#define M19_CASE(G, TT, NT, P, MB)                                                                          \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && mb == (MB)) {                               \
    m13_launch(m19::wr_kernel<TT, NT, G, P, MB>, static_cast<int>(grid), 128,                              \
               static_cast<size_t>(m19::dyn_bytes<TT, NT, P>()), static_cast<int>(pdl),                    \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP; column 3 = P | MB << 4 (MB = CTAs per SM the launch bounds ask for).
#define M19_SETUP(G, TT, NT, P, MB)                                                                         \
  {                                                                                                        \
    const int dyn = m19::dyn_bytes<TT, NT, P>();                                                           \
    cudaFuncSetAttribute(m19::wr_kernel<TT, NT, G, P, MB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m19::wr_kernel<TT, NT, G, P, MB>);                                        \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m19::wr_kernel<TT, NT, G, P, MB>, 128, dyn);   \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((MB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m19_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m19::MAXP, "m19_gemm: ", n_pairs, " pairs, at most ", 8 * m19::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t mb = ((pf >> 28) & 1) ? 4 : 3;                    // launch-bounds CTAs per SM
  @@M19_CASES@@
  TORCH_CHECK(false, "m19_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa, " MB=", mb);
  return out;
}

torch::Tensor m19_setup() {
  std::vector<int64_t> rows;
  @@M19_SETUPS@@
  return torch::tensor(rows).view({-1, 9});
}

#define M16_CASE(G, TT, NT, P, DB)                                                                          \
  if (gated == (G) && tt == (TT) && nt == (NT) && pa == (P) && db == (DB)) {                               \
    m13_launch(m16::wg_kernel<TT, NT, G, P, DB>, static_cast<int>(grid), 128,                              \
               static_cast<size_t>(1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>()), static_cast<int>(pdl), \
               reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), \
               static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), \
               reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),                                       \
               reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), scr.data_ptr<float>(), ctr.data_ptr<int>(), \
               static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total),       \
               static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(pdl),            \
               static_cast<int>(pf));                                                                       \
    return out;                                                                                            \
  }

// As M14_SETUP, with P | DB << 4 in the P column: [gated, tt, nt, P | DB << 4, registers, spill bytes, static
// smem, dynamic smem, CTAs per SM].
#define M16_SETUP(G, TT, NT, P, DB)                                                                         \
  {                                                                                                        \
    const int dyn = 1024 + ((P) + 1 + (DB)) * m16::stage_bytes<NT>();                                      \
    cudaFuncSetAttribute(m16::wg_kernel<TT, NT, G, P, DB>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn); \
    cudaFuncAttributes attr;                                                                               \
    cudaFuncGetAttributes(&attr, m16::wg_kernel<TT, NT, G, P, DB>);                                        \
    int blocks = 0;                                                                                        \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, m16::wg_kernel<TT, NT, G, P, DB>, 128, dyn);   \
    const int64_t row[9] = {(G) ? 1 : 0, TT, NT, (P) | ((DB) << 4), static_cast<int64_t>(attr.numRegs),    \
                            static_cast<int64_t>(attr.localSizeBytes), static_cast<int64_t>(attr.sharedSizeBytes), \
                            dyn, blocks};                                                                  \
    for (int64_t v : row) rows.push_back(v);                                                               \
  }

torch::Tensor m16_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf) {
  TORCH_CHECK(n_pairs <= 8 * m16::MAXP, "m16_gemm: ", n_pairs, " pairs, at most ", 8 * m16::MAXP);
  const int64_t pa = ((pf >> 24) & 15) ? ((pf >> 24) & 15) : 2;  // activation lead P
  const int64_t db = (pf >> 28) & 1;                              // double-buffered A
  @@M16_CASES@@
  TORCH_CHECK(false, "m16_gemm: no instance for gated=", gated, " tt=", tt, " nt=", nt, " P=", pa);
  return out;
}

torch::Tensor m16_setup() {
  std::vector<int64_t> rows;
  @@M16_SETUPS@@
  @@M16_RECON_ATTRS@@
  return torch::tensor(rows).view({-1, 9});
}

// M16 GEMM1 g2.4 (P 4, DB) in a recon mode (1 = decode without wgmma, 2 = wgmma without decode).
torch::Tensor m16_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode) {
  TORCH_CHECK(n_pairs <= 8 * m16::MAXP, "m16_recon: too many pairs");
  TORCH_CHECK(mode == 1 || mode == 2, "m16_recon: mode ", mode);
  @@M16_RECON@@
  return out;
}
"""

_M13_CPP = """
torch::Tensor m13_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf, int64_t nxw, int64_t nxs, int64_t nxwb, int64_t nxsb,
                       int64_t nxwarps);
torch::Tensor m13_attrs(int64_t gated, int64_t tt, int64_t nt);
torch::Tensor m11p_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                        int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth,
                        int64_t ks, int64_t grid);
torch::Tensor m13_post_norm(torch::Tensor y, torch::Tensor nw, torch::Tensor tw, torch::Tensor ids, torch::Tensor out,
                            int64_t tokens, int64_t local_start, double eps, int64_t pdl);
torch::Tensor m14_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m14_setup();
torch::Tensor m14_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t pa, int64_t mode);
torch::Tensor m15_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m15_setup();
torch::Tensor m16_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m16_setup();
torch::Tensor m16_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode);
torch::Tensor m17_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m17_setup();
torch::Tensor m18_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m18_setup();
torch::Tensor m11_sr_setup();
torch::Tensor m11b_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t a_per_pair,
                        int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid,
                        int64_t pdl, int64_t sel);
torch::Tensor m11b_attrs(int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks);
torch::Tensor m11b_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                         torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total, int64_t a_per_pair,
                         int64_t out_cols, int64_t gated, int64_t tt, int64_t nt, int64_t depth, int64_t ks, int64_t grid,
                         int64_t sel, int64_t mode);
torch::Tensor m19_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                       int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                       int64_t grid, int64_t pdl, int64_t pf);
torch::Tensor m19_setup();
torch::Tensor m18_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t grid, int64_t mode);
torch::Tensor m13_recon(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                        torch::Tensor out, torch::Tensor scr, torch::Tensor ctr, int64_t n_pairs, int64_t local_start,
                        int64_t rt_total, int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t tt, int64_t nt,
                        int64_t grid, int64_t mode);
torch::Tensor m13_stream(torch::Tensor base, torch::Tensor experts, int64_t len16, torch::Tensor sink, int64_t unroll,
                         int64_t grid);
"""


def _m11b_case(g, tt, nt, d, ks, mb, e=0):
    """m11b_gemm's dispatch for one instance (depth | minb << 4 when minb > 1 | 256 for E30's early refill).
    Non-gated: PDL or not; gated: PDL only for the 16-pair hot launch (E26)."""
    G, E = ("true" if g else "false"), ("true" if e else "false")
    dep = d | (mb << 4 if mb > 1 else 0) | (256 if e else 0)
    head = f"if (gated == {g} && tt == {tt} && nt == {nt} && depth == {dep} && ks == {ks}) {{ "
    if g and nt != 2:
        return head + (f'TORCH_CHECK(!pdl, "m11b_gemm: no PDL GEMM1 instance"); '
                       f"M11B_LAUNCH({G}, {tt}, {nt}, {d}, {ks}, false, {mb}, {E}); return out; }}")
    return head + (f"if (pdl) M11B_LAUNCH({G}, {tt}, {nt}, {d}, {ks}, true, {mb}, {E}); "
                   f"else M11B_LAUNCH({G}, {tt}, {nt}, {d}, {ks}, false, {mb}, {E}); return out; }}")


def _m11b_all():
    """Every M11b instance as (gated, tt, nt, d, ks, minb, early)."""
    return ([i + (1, 0) for i in M11B_INSTANCES] + [i + (0,) for i in M11B_OCC_INSTANCES]
            + [i + (1,) for i in M11B_EARLY_INSTANCES])


def m13_source(instances=M13_INSTANCES, m11_instances=M13_M11_INSTANCES, m14_instances=M14_INSTANCES,
               m15_instances=M15_INSTANCES, m16_instances=M16_INSTANCES, m17_instances=None, m18_instances=None,
               m19_instances=M19_INSTANCES):
    """M11's CUDA (only the instances M11_CONFIGS uses, no M12) followed by M13's and M14's."""
    args = [f"{'true' if g else 'false'}, {tt}, {nt}" for g, tt, nt in instances]
    args14 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}" for g, tt, nt, pa in m14_instances]
    args15 = [f"{'true' if g else 'false'}, {tt}, {nt}, {r}, {pa}" for g, tt, nt, r, pa in m15_instances]
    args16 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}, {'true' if db else 'false'}" for g, tt, nt, pa, db in m16_instances]
    recon16 = ("\n  ".join(f"if (mode == {md}) {{ m13_launch(m16::wg_kernel<2, 4, true, 4, true, {md}>, static_cast<int>(grid), "
                           "128, static_cast<size_t>(1024 + 6 * m16::stage_bytes<4>()), 0, "
                           "reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), "
                           "static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), "
                           "reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), "
                           "scr.data_ptr<float>(), ctr.data_ptr<int>(), static_cast<int>(n_pairs), static_cast<int>(local_start), "
                           "static_cast<int>(rt_total), 0, static_cast<int>(rt_total / 2 * 16), 0, (4 << 24) | (1 << 28)); }"
                           for md in (1, 2)) if (1, 2, 4, 4, 1) in m16_instances else 'TORCH_CHECK(false, "m16_recon: not built");')
    if m17_instances is None:  # M17 uses M16's wgmma code: built (or dropped) with it
        m17_instances = M17_INSTANCES if m16_instances else ()
    if m18_instances is None:  # so does M18 (wgmma; compiled out in the fallback build)
        m18_instances = M18_INSTANCES if m16_instances else ()
    args18 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}" for g, tt, nt, pa in m18_instances]
    args19 = [f"{'true' if g else 'false'}, {tt}, {nt}, {pa}, {mb}" for g, tt, nt, pa, mb in m19_instances]
    args17 = [f"{'true' if g else 'false'}, {tt}, {nt}, {r}, {pa}, {'true' if db else 'false'}" for g, tt, nt, r, pa, db in m17_instances]
    cases = [f"M13_CASE{'' if g else '_P'}({a})" for (g, _, _), a in zip(instances, args)]
    m11p = "\n  ".join([f"M11P_CASE({tt}, {nt}, {d}, {ks})" for g, tt, nt, d, ks in m11_instances if not g]
                        + [f"M11P_CR_CASE({tt}, {nt}, {d}, {ks})" for g, tt, nt, d, ks in M13_M11_CR_INSTANCES if not g]
                        + [f"M11P_SR_CASE({tt}, {nt}, {d}, {ks})" for g, tt, nt, d, ks in M13_M11_SR_INSTANCES if not g])
    m13 = (_M13_CUDA.replace("@@M13_CASES@@", "\n  ".join(cases))
           .replace("@@M13_ATTRS@@", "\n  ".join(f"M13_ATTR({a})" for a in args))
           .replace("@@M13_MAXP@@", str(M13_MAX_TOKENS))
           .replace("@@M13_MAXPAIRS@@", str(M13_MAX_TOKENS * TOP_K))
           .replace("@@M14_MAXP@@", str(M14_MAX_TOKENS))
           .replace("@@M14_CASES@@", "\n  ".join(f"M14_CASE({a})" for a in args14))
           .replace("@@M14_SETUPS@@", "\n  ".join(f"M14_SETUP({a})" for a in args14))
           .replace("@@M15_CASES@@", "\n  ".join(f"M15_CASE({a})" for a in args15))
           .replace("@@M15_SETUPS@@", "\n  ".join(f"M15_SETUP({a})" for a in args15))
           .replace("@@M16_CASES@@", "\n  ".join(f"M16_CASE({a})" for a in args16))
           .replace("@@M16_SETUPS@@", "\n  ".join(f"M16_SETUP({a})" for a in args16))
           .replace("@@M16_RECON@@", recon16)
           .replace("@@M16_RECON_ATTRS@@", "\n  ".join(
               f"cudaFuncSetAttribute(m16::wg_kernel<2, 4, true, 4, true, {md}>, cudaFuncAttributeMaxDynamicSharedMemorySize, "
               "1024 + 6 * m16::stage_bytes<4>());" for md in (1, 2)) if (1, 2, 4, 4, 1) in m16_instances else "")
           .replace("@@M11B_CASES@@", "\n  ".join(_m11b_case(*i) for i in _m11b_all()))
           .replace("@@M11B_RECON_CASES@@", "\n  ".join(
               f"if (gated == {g} && tt == {tt} && nt == {nt} && depth == {d} && ks == {ks} && mode == {md}) {{ "
               f"m13_launch(m11b::recon_kernel<{tt}, {nt}, {d}, {ks}, {'true' if g else 'false'}, {md}>, static_cast<int>(grid), 128, 0, 0, "
               "reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint2*>(sf.data_ptr()), "
               "static_cast<const float*>(c.data_ptr<float>()), static_cast<const int*>(ids.data_ptr<int>()), "
               "reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), "
               "static_cast<int>(n_pairs), static_cast<int>(local_start), static_cast<int>(rt_total), "
               "static_cast<int>(a_per_pair), static_cast<int>(out_cols), static_cast<int>(sel)); return out; }"
               for g, tt, nt, d, ks in M11B_RECON_SHAPES for md in (1, 2, 3)))
           .replace("@@M11B_ATTRS@@", "\n  ".join(
               f"M11B_ATTR({'true' if i[0] else 'false'}, {i[1]}, {i[2]}, {i[3]}, {i[4]}, {i[5]}, {'true' if i[6] else 'false'})"
               for i in _m11b_all()))
           .replace("@@M11_SR_ATTRS@@", "\n  ".join(
               [f"cudaFuncSetAttribute(m11::sr_kernel<{tt}, {nt}, {d}, {ks}, {'true' if g else 'false'}>, "
                f"cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 8 * {tt} * 576);" for g, tt, nt, d, ks in M13_M11_SR_INSTANCES]
               + [f"cudaFuncSetAttribute(m11::sr_kernel<{tt}, {nt}, {d}, {ks}, false, true>, "
                  f"cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 8 * {tt} * 576);" for g, tt, nt, d, ks in M13_M11_SR_INSTANCES if not g]))
           .replace("@@M11P_CASES@@", m11p)
           .replace("@@M17_CASES@@", "\n  ".join(f"M17_CASE({a})" for a in args17))
           .replace("@@M17_SETUPS@@", "\n  ".join(f"M17_SETUP({a})" for a in args17))
           .replace("@@M19_CASES@@", "\n  ".join(f"M19_CASE({a})" for a in args19))
           .replace("@@M19_SETUPS@@", "\n  ".join(f"M19_SETUP({a})" for a in args19))
           .replace("@@M18_CASES@@", "\n  ".join(f"M18_CASE({a})" for a in args18))
           .replace("@@M18_SETUPS@@", "\n  ".join(f"M18_SETUP({a})" for a in args18))
           .replace("@@M18_RECON@@", "\n  ".join(f"M18_RECON_CASE({md})" for md in (1, 2, 3, 4)) if m18_instances else "")
           .replace("@@M18_RECON_ATTRS@@", "\n  ".join(f"M18_RECON_ATTR({md})" for md in (1, 2, 3, 4)) if m18_instances else ""))
    return m11_source(m11_instances, (), M13_M11_CR_INSTANCES, M13_M11_SR_INSTANCES) + m13 + "\n" + g2p_moe_source()


def m13_build(m16_instances=M16_INSTANCES):
    """One extension with M11's and M13's kernels (sm_90a only, as m11_build)."""
    return load_inline(
        name="m13_ext" if m16_instances else "m13_ext_nowg", cpp_sources=_M11_CPP + _M13_CPP + _M11F_CPP,
        cuda_sources=m13_source(m16_instances=m16_instances),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                   "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon",
                   "m11p_gemm", "m18_gemm", "m18_setup", "m18_recon", "m19_gemm", "m19_setup", "m11_sr_setup", "m11f_gemm2_post", "m11b_gemm", "m11b_attrs", "m11b_recon"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + ([] if m16_instances else ["-DM13_NO_WGMMA"]),
        build_directory=tempfile.mkdtemp(prefix="m13_"), verbose=False)


def m13_probe_build():
    """One M16 and one M17 instance compiled with the build log shown (verbose), so ptxas's wgmma warnings (e.g.
    "wgmma.mma_async instructions are serialized due to ...") reach the run's output. Diagnostic only."""
    src = m13_source(instances=(), m11_instances=(), m14_instances=(), m15_instances=(),
                     m16_instances=((1, 2, 4, 4, 1),), m17_instances=((0, 2, 8, 2, 4, 1),))
    return load_inline(
        name="m13_probe", cpp_sources=_M11_CPP + _M13_CPP, cuda_sources=src,
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup", "m13_gemm", "m13_attrs", "m13_post_norm",
                   "m14_gemm", "m14_setup", "m13_recon", "m13_stream", "m15_gemm", "m15_setup", "m14_recon",
                   "m16_gemm", "m16_setup", "m17_gemm", "m17_setup", "m16_recon",
                   "m11p_gemm", "m18_gemm", "m18_setup", "m18_recon"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m13p_"), verbose=True)


def m13_relayout(words):
    """M11 code words (int32, any shape) -> M13's bit layout, a lossless permutation of each word's bits."""
    w = words.to(torch.int64) & 0xFFFFFFFF
    out = torch.zeros_like(w)
    for new, old in enumerate(M13_BIT_SRC):
        out |= ((w >> old) & 1) << new
    return torch.where(out >= 2**31, out - 2**32, out).to(torch.int32)


def m16_shuffle(packed, sf):
    """One expert's NVFP4 matrix -> M16's natural-K layout (a lossless reordering) for RS-wgmma, whose B operand
    (activations in shared memory) is read in natural K order. packed uint8 [rows, K/2], sf float8_e4m3fn
    [rows, K/16] -> words int32 [rows/16, K/64, 32 lanes, 4] in M11's nibble placement (so M11's / M13's decode
    applies), but lane (g, t), word j holds the positions wgmma's A fragment of k16 step j needs: 16 j + 2 t +
    {0, 1} (registers a0 / a1: rows g / g + 8) and 16 j + 2 t + 8 + {0, 1} (a2 / a3); scales int32
    [rows/16, K/64, 8, 2]: (x, y) = rows g and g + 8, byte j = the chunk's 16-K block j (read by all 4 t lanes)."""
    rows, half = packed.shape
    rt, kg = rows // 16, half * 2 // 64
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    c = codes.view(rt, 2, 8, kg, 64)  # (rt, h, g, kg, pos): row 16 rt + 8 h + g, k = 64 kg + pos
    t = torch.arange(4, device=packed.device).view(4, 1, 1)
    j = torch.arange(4, device=packed.device).view(1, 4, 1)
    q = torch.arange(4, device=packed.device).view(1, 1, 4)
    pos = 16 * j + 2 * t + 8 * (q >> 1) + (q & 1)  # [t, j, q]: q = 2 * (register >= 2) + element
    c = c[..., pos]  # (rt, h, g, kg, t, j, q)
    src = ((1, 2), (0, 2), (1, 0), (0, 0), (1, 3), (0, 3), (1, 1), (0, 1))  # as m11_shuffle
    nib = [c[:, h, :, :, :, :, qq] for h, qq in src]  # each [rt, g, kg, t, j]
    words = torch.stack([nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(4)], dim=-1)
    words = words.permute(0, 2, 1, 3, 4, 5).contiguous().view(torch.int32).view(rt, kg, 32, 4)
    sc = sf.view(torch.uint8).view(rt, 2, 8, kg, 4)  # (rt, h, g, kg, block)
    scales = sc.permute(0, 3, 2, 1, 4).contiguous().view(torch.int32).view(rt, kg, 8, 2)
    return words, scales


def m13_scales_ok(weights, start, count):
    """True when every local E4M3 block scale is positive and normal (no zero, subnormal or NaN),
    which scale2's bit trick needs; M13 stays off otherwise."""
    for name in ("w13_sf", "w2_sf"):
        b = weights[name][start:start + count].view(torch.uint8)
        bad = ((b & 0x80) != 0) | ((b & 0x78) == 0) | ((b & 0x7F) == 0x7F)
        if bool(bad.any()):
            return False
    return True


def m18_shuffle(packed, sf):
    """One expert's NVFP4 matrix -> M18's row-chunk layout (lossless) for decode-to-shared-memory: words int32
    [rows/16, K/64, 32 lanes, 4] in M11's nibble placement (so M13's relayout + decode apply), lane l = 2 r + hf of
    tile row r, word m = 8 consecutive K at 8 (4 hf + m), placed so that decoded register R holds K (2 R, 2 R + 1):
    the 4 registers are the segment's 16 bytes in natural order (one STS.128). Scales int16 [rows/16, K/256, 32, 4]
    (M13's shapes): lane (r, hf), group g, byte 2 s + b = row r's block 2 hf + b of chunk 4 g + s."""
    rows, half = packed.shape
    rt, kg = rows // 16, half * 2 // 64
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    c = codes.view(rt, 16, kg, 2, 4, 8)  # (rt, r, kg, hf, m, k): row 16 rt + r, K = 64 kg + 8 (4 hf + m) + k
    src = ((1, 2), (0, 2), (1, 0), (0, 0), (1, 3), (0, 3), (1, 1), (0, 1))  # M11's (h, q) per nibble
    nib = [c[..., 2 * (h + 2 * (q >> 1)) + (q & 1)] for h, q in src]  # register h + 2 (q >> 1), element q & 1
    words = torch.stack([nib[2 * b] | (nib[2 * b + 1] << 4) for b in range(4)], dim=-1)  # (rt, r, kg, hf, m, 4)
    words = words.permute(0, 2, 1, 3, 4, 5).contiguous().view(torch.int32).view(rt, kg, 32, 4)
    sc = sf.view(torch.uint8).view(rt, 16, kg // 4, 4, 2, 2)  # (rt, r, grp, s, hf, b): block 4 (4 grp + s) + 2 hf + b
    scales = sc.permute(0, 2, 1, 4, 3, 5).contiguous().view(torch.int16).view(rt, kg // 4, 32, 4)
    return words, scales


def m18_bank(weights, bank, start, count):
    """M18's row-chunk codes (M13 bit layout) and scales, one expert at a time (shares M13's scratch/counters)."""
    for name in ("w13", "w2"):
        ws, ss = [], []
        for e in range(start, start + count):
            words, scales = m18_shuffle(weights[name][e], weights[name + "_sf"][e])
            ws.append(m13_relayout(words))
            ss.append(scales)
        bank[name + "r"] = torch.stack(ws)
        bank[name + "r_s"] = torch.stack(ss)


def m16_bank(weights, bank, start, count):
    """M16's natural-K codes (M13 bit layout) and per-block scales, one expert at a time; they share M13's
    scratch, counters and global scales."""
    for name in ("w13", "w2"):
        ws, ss = [], []
        for e in range(start, start + count):
            words, scales = m16_shuffle(weights[name][e], weights[name + "_sf"][e])
            ws.append(m13_relayout(words))
            ss.append(scales)
        bank[name + "n"] = torch.stack(ws)
        bank[name + "n_s"] = torch.stack(ss)


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
    bank["ctrf"] = m11f_counters(device)  # E23: G2P's per-expert + per-token counters (zero on entry and exit)
    return bank


def _m13_next(state, s1, s2):
    """M13 GEMM1's next-stage prefetch arguments: GEMM2's code and scale arrays in M13's bank (M13, M14 and M15
    GEMM2 all read it; addresses and bytes per expert taken once in m13_setup), and GEMM2's warp count; zeros
    when s1 has nxq 0."""
    if not s1.get("nxq", 0) or s2 is None or s2.get("kind") not in ("m13", "m14", "m15"):
        return (0, 0, 0, 0, 0)
    return state["m13_nx"] + (4 * s2["c"] * state["sms"],)


def _m11b_launch(state, bank, gemm_no, ids, a, out, n_pairs, st, pdl, sel):
    """M11b (E24) on the M13 bank; sel = the hot/cold split (E26: thr | hot side << 16, 0 = every expert)."""
    if gemm_no == 1:
        state["m11_ext"].m11b_gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, a, out, n_pairs, bank["start"],
                                   bank["rt_w13"], 0, bank["inter"], 1, 2, st["nt"], st["d"] | (st.get("occ", 0) << 4) | (st.get("early", 0) << 8),
                                   st["ks"], st["c"] * state["sms"], pdl, sel)
    else:
        state["m11_ext"].m11b_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, a, out, n_pairs, bank["start"],
                                   bank["rt_w2"], 1, bank["hidden"], 0, st["tt"], st["nt"], st["d"] | (st.get("occ", 0) << 4) | (st.get("early", 0) << 8),
                                   st["ks"], st["c"] * state["sms"], pdl, sel)


def _split_thr(st, n_pairs):
    """E18/E26 hot/cold split: experts with > thr pairs run in a second launch (st["hot"]); 0 = no split."""
    thr = st.get("thr", 0)
    if thr:
        kinds = ("m14", "m18", "m19", "m11b")
        assert st["kind"] in kinds and st["hot"]["kind"] in kinds and 0 < thr < 256, st
        # M11b lists at most M11_MAX_TOKENS (its MAXP) pairs per expert: as the cold side thr must fit, as the hot
        # side every expert must (pairs per expert <= tokens)
        assert st["kind"] != "m11b" or thr <= M11_MAX_TOKENS, st
        assert st["hot"]["kind"] != "m11b" or n_pairs // TOP_K <= M11_MAX_TOKENS, st
    return thr


def _hot_pf(h, thr):
    """pf of an M14/M18/M19 hot launch: prefetch distance, P, MB (E26: MB 4 was never passed), thr, hot side."""
    return ((h.get("pfd", 0) << 8) | (h.get("p", 0) << 24) | ((h.get("mb", 3) == 4) << 28) | (thr << 16)
            | (1 << 30))


def m13_gemm1(x, ids, state, s1, s2=None):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair. s2 (GEMM2's stage) only
    sizes M13's next-stage L2 prefetch."""
    kind = s1.get("kind")
    if kind == "m11b":  # E24: M11's kernel on the M13 bank; E26: also as the cold side of the split
        bank, n_pairs = state["m13"], ids.numel()
        act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
        thr = _split_thr(s1, n_pairs)
        _m11b_launch(state, bank, 1, ids, x, act, n_pairs, s1, 0, thr)
        if thr:
            _gemm1_hot(state, bank, ids, x, act, n_pairs, s1["hot"], thr)
        return act
    if kind not in ("m13", "m14", "m15", "m16", "m17", "m18", "m19"):
        return m11_gemm1(x, ids, state, s1)
    bank, n_pairs = state["m13"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    gemm = getattr(state["m11_ext"], kind + "_gemm")
    pf = (s1.get("pfd", 0) << 8) | (s1.get("beta", 0) << 16)
    wk = {"m16": "w13n", "m17": "w13n", "m18": "w13r"}.get(kind, "w13")
    args = [bank[wk], bank[wk + "_s"], bank["c13"], ids, x, act, bank["scr13"], bank["ctr13"],
            n_pairs, bank["start"], bank["rt_w13"], 0, bank["inter"], 1, 2, s1["nt"], s1["c"] * state["sms"], 0]
    if kind == "m13":
        nx = _m13_next(state, s1, s2)
        pf |= s1.get("npre", 0) | ((s1.get("nxq", 0) if nx[4] else 0) << 24)
        gemm(*args, pf, *nx)
    else:
        thr = _split_thr(s1, n_pairs)  # E18: experts with > thr pairs go to a second launch (s1["hot"]), MF19
        gemm(*args, pf | (s1.get("p", 0) << 24) | (s1.get("db", 0) << 28) | (thr << 16) | ((s1.get("mb", 3) == 4) << 28))
        if thr:
            _gemm1_hot(state, bank, ids, x, act, n_pairs, s1["hot"], thr)
    return act


def _gemm1_hot(state, bank, ids, x, act, n_pairs, h, thr):
    """GEMM1's hot launch (experts with > thr pairs), PDL after the cold launch."""
    if h["kind"] == "m11b":
        _m11b_launch(state, bank, 1, ids, x, act, n_pairs, h, 1, thr | (1 << 16))
        return
    hk = {"m18": "w13r"}.get(h["kind"], "w13")
    getattr(state["m11_ext"], h["kind"] + "_gemm")(
        bank[hk], bank[hk + "_s"], bank["c13"], ids, x, act, bank["scr13"], bank["ctr13"], n_pairs, bank["start"],
        bank["rt_w13"], 0, bank["inter"], 1, 2, h["nt"], h["c"] * state["sms"], 1, _hot_pf(h, thr))


def m13_gemm2(act, ids, state, s2):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    kind = s2.get("kind")
    if kind == "m11p":  # M11's GEMM2 (its bank and config keys), launched with PDL
        bank, n_pairs = state["m11"], ids.numel()
        y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
        state["m11_ext"].m11p_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                                   bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"], s2["d"] | (s2.get("cr", 0) << 4) | (s2.get("sr", 0) << 5), s2["ks"],
                                   s2["c"] * state["sms"])
        return y
    if kind == "m11b":  # E24; E26: also as the cold side of the split
        bank, n_pairs = state["m13"], ids.numel()
        y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
        thr = _split_thr(s2, n_pairs)
        _m11b_launch(state, bank, 2, ids, act, y, n_pairs, s2, s2.get("pdl", 0), thr)
        if thr:
            _gemm2_hot(state, bank, ids, act, y, n_pairs, s2["hot"], thr, s2.get("pdl", 0))
        return y
    if kind not in ("m13", "m14", "m15", "m16", "m17", "m18", "m19"):
        return m11_gemm2(act, ids, state, s2)
    bank, n_pairs = state["m13"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    gemm = getattr(state["m11_ext"], kind + "_gemm")
    wk = {"m16": "w2n", "m17": "w2n", "m18": "w2r"}.get(kind, "w2")
    args = [bank[wk], bank[wk + "_s"], bank["c2"], ids, act, y, bank["scr2"], bank["ctr2"],
            n_pairs, bank["start"], bank["rt_w2"], 1, bank["hidden"], 0, s2["tt"], s2["nt"],
            s2["c"] * state["sms"], s2.get("pdl", 0),
            s2.get("pf", 0) | (s2.get("pfd", 0) << 8) | (s2.get("beta", 0) << 16)]
    if kind == "m13":
        args[-1] |= s2.get("npre", 0)
        gemm(*args, 0, 0, 0, 0, 0)
    else:
        thr = _split_thr(s2, n_pairs)  # E18 hot/cold split, as m13_gemm1
        args[-1] |= (s2.get("p", 0) << 24) | (s2.get("db", 0) << 28) | (thr << 16) | ((s2.get("mb", 3) == 4) << 28)
        gemm(*args)
        if thr:
            _gemm2_hot(state, bank, ids, act, y, n_pairs, s2["hot"], thr, s2.get("pdl", 0))
    return y


def _gemm2_hot(state, bank, ids, act, y, n_pairs, h, thr, pdl):
    """GEMM2's hot launch (experts with > thr pairs), after the cold launch (PDL as the cold one)."""
    if h["kind"] == "m11b":
        _m11b_launch(state, bank, 2, ids, act, y, n_pairs, h, pdl, thr | (1 << 16))
        return
    hk = {"m18": "w2r"}.get(h["kind"], "w2")
    getattr(state["m11_ext"], h["kind"] + "_gemm")(
        bank[hk], bank[hk + "_s"], bank["c2"], ids, act, y, bank["scr2"], bank["ctr2"], n_pairs, bank["start"],
        bank["rt_w2"], 1, bank["hidden"], 0, h["tt"], h["nt"], h["c"] * state["sms"], pdl, _hot_pf(h, thr))


def m13_post_norm(y, ids, topk_weights, state, tokens, post):
    if not post:
        return m11_post_norm(y, ids, topk_weights, state, tokens)
    out = torch.empty((tokens, state["m13"]["hidden"]), dtype=torch.bfloat16, device=y.device)
    state["m11_ext"].m13_post_norm(y, state["norm_w"], topk_weights, ids, out, tokens, state["m13"]["start"],
                                   POST_NORM_EPS, post.get("pdl", 0))
    return out


def m13_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m13_gemm1(x, ids, state, cfg["s1"], cfg["s2"])
    if cfg["s2"].get("kind") == "m11f":  # E23: G2P (prefill's fused M11 GEMM2 + post-norm, moe/m13/g2p_moe.py)
        return m11f_gemm2_post_py(act, ids, topk_weights.contiguous(), state, x.shape[0], cfg["s2"], state["m11_ext"],
                                  state["m13"]["ctrf"], state["sms"])
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
    try:
        state["m11_ext"] = m13_build()
        state["m16_ok"] = True
    except Exception as exc:  # M16 is the first wgmma code: if ptxas rejects it, build everything else
        print(f"[m13] build with M16 FAILED ({type(exc).__name__}: {str(exc)[-300:]}); rebuilding without M16",
              flush=True)
        state["m11_ext"] = m13_build(())
        state["m16_ok"] = False
    state["m12_resources"] = []
    # M14's dynamic shared-memory limits, set once here and never during a capture
    state["m14_resources"] = state["m11_ext"].m14_setup().tolist()
    state["m15_resources"] = state["m11_ext"].m15_setup().tolist()
    state["m16_resources"] = state["m11_ext"].m16_setup().tolist()
    state["m17_resources"] = state["m11_ext"].m17_setup().tolist()
    state["m18_resources"] = state["m11_ext"].m18_setup().tolist()
    state["m19_resources"] = state["m11_ext"].m19_setup().tolist()
    state["m11_ext"].m11_sr_setup()  # E22b: before any SR launch or capture
    state["m11"] = m11_bank(weights, 0, 32)
    state["m13_ok"] = m13_scales_ok(weights, 0, 32)
    if state["m13_ok"]:
        state["m13"] = m13_bank(state["m11"], weights["w13"].device)
        m16_bank(weights, state["m13"], 0, 32)
        m18_bank(weights, state["m13"], 0, 32)
        w, s = state["m13"]["w2"], state["m13"]["w2_s"]  # the persistent bank: its addresses never change
        state["m13_nx"] = (w.data_ptr(), s.data_ptr(), w[0].numel() * w.element_size(), s[0].numel() * s.element_size())
    torch.cuda.synchronize()


def _m13_prepare(weights):
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


# ====================================================================== self-tuning prepare (moe/m13/tune.py)

"""Self-tuning prepare (appended after the M13 section, whose prepare becomes _m13_prepare).

The queue makes lab -> config -> scored-run cycles slow, so prepare() measures its own options on the
judge's GPU, on synthetic inputs only (nothing derived from any call's inputs is ever kept), and picks per
token class the fastest config that passes an FP32 check stricter than the judge's at every size of the
class (judge-like, skewed and outlier routings): M11's E3c configs and v1 are the incumbents; M13 (1-64
tokens) and M14 (33-320 tokens) with or without PDL and with an L2 prefetch distance are the candidates.
Any failure leaves the incumbents in place. A compact log (bandwidth recon, every timing, every
decision) is printed for the run's log tail. Budgeted so prepare() as a whole stays under ~250 s."""

import statistics
import time

TUNE_BUDGET_S = 170.0
TUNE_PREPARE_LIMIT_S = 255.0  # prepare() gets 300 s
TUNE_REPS = 20  # E8/E9 tuned in ~5 s of the 170 s budget: sample more
TUNE_ROUTINGS = 3
TUNE_MAX_ERR, TUNE_MAX_ROW_ERR = 1e-2, 5e-2  # the judge allows 1.5e-2 and 7.5e-2
TUNE_MARGIN = 0.95  # a candidate must be >= 5% faster than the incumbent (E12: +4-5% in-tuner was -2% judged)
TUNE_MARGIN_PDL = 0.98  # the incumbent's own kernels with a PDL tail (same kernels; E15: bs8 98.0 vs 101.0 just missed 3%)
TUNE_FLUSH_BYTES = 512 << 20
# class upper bound -> representative batch size
TUNE_CLASSES = ((5, 1), (9, 8), (17, 16), (33, 32), (65, 64), (192, 128), (321, 256))
# M13 candidates (E5's best stages, MF8) x L2 prefetch distance x PDL (GEMM2, post-norm)
TUNE_M13_S1 = (dict(kind="m13", nt=1, c=4),)
TUNE_M13_S2 = (dict(kind="m13", tt=2, nt=1, c=4),)  # E8: t1n1c5 lost to t2n1c4 in every class
# Start-up L2 prefetches (E10): npre 255 = the warp's own next groups (lane-parallel, at kernel start);
# nxq = GEMM1 prefetches the first nxq/4 of each GEMM2 warp's weight range (GEMM2 then starts from L2).
TUNE_M13_S1_PF = (dict(npre=0, nxq=0),)  # E12: every start-up prefetch was slower (MF12/MF13)
TUNE_M13_S2_PF = (dict(npre=0),)
TUNE_PFD = (0,)  # E8: L2 prefetch-ahead 2/4 was slower everywhere (MF10)
TUNE_PDL = ((0, 0), (1, 1))
# E18 hot/cold split (MF19): experts with > T pairs run as a second launch on a wide-pass kernel (decoded 2-4x
# instead of ~8x at 16 pairs per pass); the cold part keeps the stage search's best M14/M18 stage.
TUNE_SPLIT_T = (16, 32, 48)  # E26: + 48 (at bs256 T 16 sends about half the experts to the hot side, MF21)
# E20: M19 with 32-pair passes (4 or 3 CTAs per SM) as the hot kernel; E18/E19 never chose M18 g2.8 or p1.16
TUNE_HOT1 = (dict(kind="m19", nt=4, c=4, p=2, mb=4), dict(kind="m19", nt=4, c=3, p=2, mb=3), dict(kind="m14", nt=4, c=3, p=4))
TUNE_HOT2 = (dict(kind="m19", tt=2, nt=4, c=4, p=2, mb=4), dict(kind="m18", tt=2, nt=8, c=2, p=2),
             dict(kind="m14", tt=1, nt=4, c=4, p=4))
# E26 (T34): M11b as the cold side at 128/256 tokens (it beat M19 by 11% at bs32, where experts also carry 1-15
# tokens); cold GEMM1 / GEMM2 stages (8- or 16-pair passes), thresholds, and E25's best hot pairs (+ M19 at MB 4,
# which the hot launch never got before E26: its MB bit wasn't passed)
TUNE_COLD11B_S1 = (dict(kind="m11b", nt=1, d=4, ks=2, c=8), dict(kind="m11b", nt=2, d=4, ks=2, c=8))
TUNE_COLD11B_S2 = (dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8), dict(kind="m11b", tt=2, nt=2, d=4, ks=2, c=8))
TUNE_COLD11B_T = (8, 16, 32)
TUNE_HOT_PAIRS = ((TUNE_HOT1[1], TUNE_HOT2[2]), (TUNE_HOT1[1], TUNE_HOT2[1]), (TUNE_HOT1[0], TUNE_HOT2[0]))
# E27 (MF29): M11b at 5-6 CTAs per SM. Base = E26's picks per class (s1, s2); bs128's cold side with E26's hot pair,
# bs256 with M11b cold under E26's hot M19 pair. Variants replace s1, s2 or both with an occupancy instance of the same
# shape (M11B_OCC_INSTANCES), grid c = minb or 2 minb.
_B1 = (dict(kind="m11b", nt=1, d=4, ks=4, c=8), dict(kind="m11b", tt=1, nt=1, d=4, ks=4, c=4, pdl=1))
_B8 = (dict(kind="m11b", nt=1, d=4, ks=2, c=8), dict(kind="m11b", tt=2, nt=1, d=4, ks=4, c=8, pdl=1))
TUNE_OCC_BASE = {1: _B1, 8: _B8, 16: _B8, 32: _B8,
                 128: (dict(_B8[0], thr=16, hot=dict(TUNE_HOT1[1])), dict(_B8[1], thr=16, hot=dict(TUNE_HOT2[1]))),
                 256: (dict(_B8[0], thr=16, hot=dict(TUNE_HOT1[1])), dict(_B8[1], thr=16, hot=dict(TUNE_HOT2[0])))}


def _early_variants(st, gated):
    """E30: st's M11b shape with the early ring refill (M11B_EARLY_INSTANCES), uncapped and at 128 registers, same c."""
    tt = 2 if gated else st["tt"]
    return [dict(st, early=1, occ=(mb if mb > 1 else 0)) for g, t, nt, d, ks, mb in M11B_EARLY_INSTANCES
            if (g, t, nt, d, ks) == (int(gated), tt, st["nt"], st["d"], st["ks"])]


def _occ_variants(st, gated):
    """st's M11b shape at 5-6 CTAs per SM (M11B_OCC_INSTANCES), each at grid c = minb and 2 minb."""
    tt = 2 if gated else st["tt"]
    out = []
    for g, t, nt, d, ks, mb in M11B_OCC_INSTANCES:
        if (g, t, nt, ks) == (int(gated), tt, st["nt"], st["ks"]):
            for c in (mb, 2 * mb):
                out.append(dict(st, d=d, occ=mb, c=c))
    return out


def tlog(msg):
    print(f"[tune] {msg}", flush=True)


class _Tuner:
    def __init__(self, budget):
        self.t0 = time.monotonic()
        self.budget = budget
        self.flush = None

    def left(self):
        return self.budget - (time.monotonic() - self.t0)

    def time_graph(self, fn, reps=TUNE_REPS):
        """Median us of a CUDA-graph replay of fn, each after an L2 flush (as the judge times)."""
        if self.flush is None:
            self.flush = torch.empty(TUNE_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
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
            self.flush.sum()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)


def _routing(bs, seed):
    """The reference's default routing (top-8 of uniform + heavy-tailed popularity, busiest GPU = 0..31)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand((bs, 256), generator=gen, device="cuda")
    z = torch.randn((256,), generator=gen, device="cuda")
    u = torch.rand((256,), generator=gen, device="cuda").clamp(min=1e-12)
    scores = scores + 0.03 * z / (-u.log()).sqrt()
    vals, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // 32).flatten(), minlength=8)
    ids = (ids - 32 * pairs.argmax()) % 256
    return ids.to(torch.int32).contiguous(), (vals / vals.sum(-1, keepdim=True)).float().contiguous()


def _case(bs, seed, skew=False, outliers=False):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((bs, 4096), generator=gen, device="cuda").to(torch.bfloat16)
    if outliers:  # 1% of channels scaled 30x, like the judge's skewed test
        cols = torch.randperm(4096, generator=gen, device="cuda")[:41]
        x[:, cols] = (x[:, cols].float() * 30.0).to(torch.bfloat16)
    if skew:  # every token on experts 0-7
        ids = torch.arange(TOP_K, dtype=torch.int32, device="cuda").repeat(bs, 1).contiguous()
        w = torch.rand((bs, TOP_K), generator=gen, device="cuda")
        w = (w / w.sum(-1, keepdim=True)).contiguous()
    else:
        ids, w = _routing(bs, seed)
    return x, ids, w


class _Ref:
    """FP32 reference from the raw NVFP4 weights (as reference.py), experts dequantized on demand."""

    def __init__(self, weights):
        self.w = weights
        self.cache = {}

    def expert(self, e):
        if e not in self.cache:
            w = self.w
            self.cache[e] = (_tune_dequant(w["w13"][e], w["w13_sf"][e], w["w13_gs"][e]),
                             _tune_dequant(w["w2"][e], w["w2_sf"][e], w["w2_gs"][e]))
        return self.cache[e]

    def full(self, x, ids, tw):
        flat = ids.reshape(-1).long()
        nw = 1.0 + self.w["post_norm_w"].float()
        out = torch.zeros((ids.shape[0], 4096), device="cuda")
        for e in [e for e in torch.unique(flat).tolist() if e < 32]:
            p = (flat == e).nonzero().flatten()
            h = x[p // TOP_K].float() @ self.expert(e)[0].T
            act = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]  # FP32, as reference.ref_kernel
            y = (act @ self.expert(e)[1].T) * nw
            y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + POST_NORM_EPS)
            out.index_add_(0, p // TOP_K, y * tw.reshape(-1)[p].unsqueeze(1))
        return out


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _tune_dequant(packed, sf, gs):
    """NVFP4 -> FP32 [rows, K] (e2m1 * e4m3 block scale * global scale), as nvfp4.dequantize."""
    grid = torch.tensor(_E2M1, dtype=torch.float32, device=packed.device)
    p = packed.view(torch.uint8)
    codes = torch.stack((p & 0x0F, p >> 4), dim=-1).flatten(-2)
    mag = grid[(codes & 0x7).long()]
    vals = torch.where((codes & 0x8) != 0, -mag, mag)
    rows, k = vals.shape
    return (vals.view(rows, k // 16, 16) * (sf.float() * gs.float()).unsqueeze(-1)).view(rows, k)


def _errors(out, ref):
    out, ref = out.float(), ref.float()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    diff = out - ref
    err = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    floor = (0.1 * rows_ref.mean()).clamp(min=1e-12)
    return err, float((diff.norm(dim=-1) / torch.maximum(rows_ref, floor)).max())


def _clean(state):
    bank = state["m13"]
    dirty = [k for k in ("scr13", "scr2", "ctr13", "ctr2", "ctrf") if bool(bank[k].ne(0).any())]
    for k in dirty:
        bank[k].zero_()
    return dirty


def _run(state, cfg, x, ids, w):
    if cfg is None:
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w,
                "config": {"bs": x.shape[0], "local_start": 0, "local_count": 32}}
        return _m11_kernel_fn(data, state)
    return m13_run(x, ids, w, state, cfg)


def _sname(st):
    if st.get("kind") in (None, "m11p", "m11f", "m11b"):  # M11's stage dicts (E3c; m11f = G2P, E23; m11b, E24)
        return (f"{st.get('kind', 'm11')}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}d{st['d']}k{st['ks']}"
                f"c{st['c']}{'R' if st.get('cr') else ''}{'S' if st.get('sr') else ''}{'o' + str(st['occ']) if st.get('occ') else ''}{'E' if st.get('early') else ''}"
                + (f"|>{st['thr']}:{_sname(st['hot'])}" if st.get("thr") else ""))  # E26: M11b as a split's cold side
    extra = f"b{st['beta']}" if st["kind"] in ("m15", "m17") else f"pf{st.get('pfd', 0)}"
    if st.get("p"):
        extra += f"P{st['p']}"
    if st.get("db"):
        extra += "D"
    if st.get("mb"):
        extra += f"B{st['mb']}"
    if st.get("npre"):
        extra += f"s{st['npre']}"
    if st.get("nxq"):
        extra += f"x{st['nxq']}"
    if st.get("thr"):
        extra += f"|>{st['thr']}:{_sname(st['hot'])}"
    return f"{st['kind']}:{'t' + str(st['tt']) if 'tt' in st else ''}n{st['nt']}c{st['c']}{extra}"


def _name(cfg):
    if cfg is None:
        return "incumbent"
    s1, s2 = cfg["s1"], cfg["s2"]
    return f"{_sname(s1)}+{_sname(s2)} pdl{s2.get('pdl', 0)}{cfg['post']['pdl']}"


def _valid(state, ref, cfg, lo, hi):
    """Worst (error, row error) over the class's sizes and routings; None if scratch was left dirty."""
    sizes = sorted({b for b in (1, 2, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65, 128, 191, 192, 256, 257, 320) if lo <= b < hi})
    cases = [(b, False, False) for b in sizes] + [(b, True, False) for b in sizes] + [(sizes[-1], True, True)]
    worst = (0.0, 0.0)
    for b, skew, outl in cases:
        x, ids, w = _case(b, 3100 + b + 50 * skew + 25 * outl, skew, outl)
        err, row = _errors(_run(state, cfg, x, ids, w), ref.full(x, ids, w))
        worst = (max(worst[0], err), max(worst[1], row))
    torch.cuda.synchronize()
    return None if _clean(state) else worst


def _bw_recon(tuner, state):
    """Bandwidth recon for the log: torch sum and a plain streaming kernel over contiguous expert codes; M13
    GEMM1 at about bs1 and bs8 and M14 GEMM1 (n2 c3 P4) at bs128 and bs256, each full and in the recon modes
    (loads only / decode without MMA / MMA without decode), so the log says what binds at each size."""
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    sink = torch.zeros(4, dtype=torch.int32, device="cuda")
    len16 = bank["w13"][0].numel() // 4
    parts = []
    for n in (3, 9):
        mb = n * len16 * 16 / 1e6
        t_torch = tuner.time_graph(lambda: bank["w13"][:n].view(torch.int64).sum())
        experts = torch.arange(n, dtype=torch.int32, device="cuda")
        best = max((mb / tuner.time_graph(lambda: ext.m13_stream(bank["w13"], experts, len16, sink, u, c * sms)), u, c)
                   for u in (8, 16) for c in (4, 8, 16))
        parts.append(f"{mb:.0f}MB torch {mb / t_torch:.2f} stream {best[0]:.2f} (U{best[1]} c{best[2]})")
    tlog("bw TB/s (incl. graph launch): " + "; ".join(parts))
    # E17: where M18's ~2,000 cycles per chunk go (MF17); M14 for reference, same routing. M18 modes: 1 = ring + stages
    # + barrier only, 2 = decode + store without wgmma, 3 = wgmma without decode, 4 = full with __syncwarp for the barrier.
    for bs in (128, 256):
        x, ids, w = _case(bs, 900 + bs)
        flat = ids.reshape(-1)
        ne = int((torch.bincount(flat.long(), minlength=256)[:32] > 0).sum())
        mb = ne * 18.87
        out = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")

        def guarded(name, fn):  # E17: one refused launch lost every number; now each one fails alone
            try:
                return fn()
            except Exception as exc:
                tlog(f"skeleton bs{bs} {name} FAILED {type(exc).__name__}: {str(exc)[-160:]}")
                _clean(state)
                return float("nan")

        def g1(st):
            return guarded(f"{st['kind']} P{st['p']} pfd{st.get('pfd', 0)}",
                           lambda: tuner.time_graph(lambda: m13_gemm1(x, flat, state, st)))
        t14 = g1(dict(kind="m14", nt=2, c=3, p=2))
        t14l = guarded("m14 loads", lambda: tuner.time_graph(lambda: ext.m14_recon(bank["w13"], bank["w13_s"], bank["c13"], flat, x, out, bank["scr13"],
                                                      bank["ctr13"], flat.numel(), 0, bank["rt_w13"], 0, bank["inter"], 1, 2, 2,
                                                      3 * sms, 4, 1)))
        t18 = g1(dict(kind="m18", nt=4, c=3, p=2))
        t18p4 = g1(dict(kind="m18", nt=4, c=3, p=4))
        t18f = [g1(dict(kind="m18", nt=4, c=3, p=2, pfd=k)) for k in (2, 4)]
        t18m = [guarded(f"m18 mode {m}", lambda m=m: tuner.time_graph(lambda: ext.m18_recon(
                    bank["w13r"], bank["w13r_s"], bank["c13"], flat, x, out, bank["scr13"], bank["ctr13"], flat.numel(), 0,
                    bank["rt_w13"], 3 * sms, m))) for m in (1, 2, 3, 4)]
        _clean(state)
        tlog(f"skeleton bs{bs} GEMM1 {mb:.0f}MB ({ne} exp), us: M14 n2c3 {t14:.0f} (loads {t14l:.0f}) | M18 g2.4c3 P2 {t18:.0f}, "
             f"P4 {t18p4:.0f}, L2-ahead 2/4 {t18f[0]:.0f}/{t18f[1]:.0f} | M18 modes: loads+stages+barrier {t18m[0]:.0f}, "
             f"decode+store {t18m[1]:.0f}, wgmma only {t18m[2]:.0f}, no CTA barrier {t18m[3]:.0f}")
    _clean(state)


def _m11b_recon(tuner, state):
    """E29 (MF31): where M11b's time goes at 1-128 tokens. Per batch size, the class's GEMM1 and GEMM2 shapes (bs128:
    the cold side, T 16) full and in recon modes 1-3 (loads only / decode without MMA / MMA without decode), in us
    and in TB/s of weight bytes read (codes + scales of the experts the launch covers), one routing each."""
    bank, ext, sms = state["m13"], state["m11_ext"], state["sms"]
    for bs, (nt1, d1, k1, c1), (tt2, d2, k2, c2), thr in ((1, (1, 4, 4, 8), (1, 4, 4, 4), 0), (8, (1, 4, 2, 8), (2, 4, 4, 8), 0),
                                                         (32, (1, 4, 2, 8), (2, 4, 4, 8), 0), (128, (1, 4, 2, 8), (2, 4, 4, 8), 16)):
        try:
            x, ids, _ = _case(bs, 950 + bs)
            flat = ids.reshape(-1).contiguous()
            cnt = torch.bincount(flat.long(), minlength=256)[:32]
            ne = int(((cnt > 0) & (cnt <= thr if thr else cnt > 0)).sum())
            act = torch.randn((flat.numel(), 4096), device="cuda").to(torch.bfloat16)
            out1 = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")
            out2 = torch.empty((flat.numel(), 4096), dtype=torch.bfloat16, device="cuda")
            parts = []
            for name, mb, args in (
                    ("GEMM1", ne * 18.87, (bank["w13"], bank["w13_s"], bank["c13"], flat, x, out1, flat.numel(), 0,
                                           bank["rt_w13"], 0, bank["inter"], 1, 2, nt1, d1, k1, c1 * sms)),
                    ("GEMM2", ne * 9.44, (bank["w2"], bank["w2_s"], bank["c2"], flat, act, out2, flat.numel(), 0,
                                          bank["rt_w2"], 1, bank["hidden"], 0, tt2, 1, d2, k2, c2 * sms))):
                ts = [tuner.time_graph(lambda: ext.m11b_gemm(*args, 0, thr))]
                ts += [tuner.time_graph(lambda m=m: ext.m11b_recon(*args, thr, m)) for m in (1, 2, 3)]
                parts.append(f"{name} {mb:.0f}MB: full {ts[0]:.1f} loads {ts[1]:.1f} no-MMA {ts[2]:.1f} no-decode {ts[3]:.1f} us"
                             f" ({mb / ts[0]:.2f}/{mb / ts[1]:.2f} TB/s)")
            tlog(f"m11b recon bs{bs}{' cold<=16' if thr else ''} ({ne} exp): " + " | ".join(parts))
        except Exception as exc:
            tlog(f"m11b recon bs{bs} FAILED {type(exc).__name__}: {str(exc)[-160:]}")
            _clean(state)


def _m14_stage_search(tuner, state, bss):
    """Best large-batch GEMM1 and GEMM2 stage per batch size (cold-L2 graph time, 2 routings), over the
    compiled M14 and M15 instances x CTAs per SM up to the measured occupancy (M14 with L2 prefetch 2 groups
    ahead, E6/MF9; M15 with cost weight 0 or 4)."""
    variants = []
    for kind in ("m14", "m19", "m18", "m15"):  # M14 first (proven); M16/M17 left the search in E16 (2x slower, MF14/MF15)
        res = state.get(kind + "_resources", [])
        tlog(f"{kind} attrs: " + ", ".join(f"{'g' if g else 'p'}{tt}.{nt}P{pr} "
                                          f"{regs}r{f' {local}B' if local else ''} {(smem + dyn) // 1024}K {blocks}cta"
                                          for g, tt, nt, pr, regs, local, smem, dyn, blocks in res))
        for g, tt, nt, pr, regs, local, smem, dyn, blocks in res:
            pa, db = pr & 15, pr >> 4  # M16/M17 rows carry P | DB << 4
            for c in range(1, blocks + 1):
                for extra in (tuple(dict(beta=b, p=pa, db=db) for b in (0, 4)) if kind in ("m15", "m17")
                              else (dict(pfd=0, p=pa, mb=pr >> 4),) if kind == "m19"  # E19: column 3 = P | MB << 4
                              else (dict(pfd=0, p=pa, db=db),)):
                    st = dict(kind=kind, nt=nt, c=c, **extra)
                    if not g:
                        st["tt"] = tt
                    variants.append((g, st))
    best = {1: {}, 2: {}}
    cases = {bs: [_case(bs, 900 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)] for bs in bss}
    acts = {bs: torch.randn((bs * TOP_K, 4096), device="cuda").to(torch.bfloat16) for bs in bss}
    rows = []
    for g, st in variants:
        if tuner.left() < 60:
            tlog(f"stage search out of budget after {len(rows)} of {len(variants)} variants")
            break
        stage = 1 if g else 2
        line = {}
        for bs in bss:
            if (st["kind"] == "m14" and bs < 64) or (st["kind"] == "m19" and bs < 16):  # E6: M14 lost below 64 tokens; E20 tries M19 (4 CTAs) from 16
                continue
            if st["kind"] in ("m15", "m17") and st["nt"] <= 2 and bs > 32:  # their small-batch instances
                continue
            ts = []
            for x, ids, w in cases[bs]:
                flat = ids.reshape(-1)
                fn = (lambda x=x, f=flat: m13_gemm1(x, f, state, st)) if g else \
                     (lambda a=acts[bs], f=flat: m13_gemm2(a, f, state, st))
                ts.append(tuner.time_graph(fn))
            t = sum(ts) / len(ts)
            line[bs] = t
            best[stage].setdefault(bs, []).append((t, st))
        _clean(state)
        rows.append((stage, st, line))
    # the log keeps variants within 1.25x of their stage's best at some size (the output tail is 16,000 chars)
    top = {(stage, bs): min(t for t, _ in v) for stage in (1, 2) for bs, v in best[stage].items()}
    shown = 0
    for stage, st, line in rows:  # M16/M17 within 1.5x of the best somewhere (E13 logged them all), others 1.25x
        lim = 1.5 if st["kind"] in ("m16", "m17") else 1.25
        if any(t <= lim * top[(stage, bs)] for bs, t in line.items()):
            shown += 1
            tlog(f"s{stage} {_sname(st)} | " + " ".join(f"{bs}:{t:.0f}" for bs, t in line.items()))
    tlog(f"stage search: {len(rows)} variants timed, {shown} shown")
    for stage in (1, 2):  # keep the 2 fastest per batch size, so one wrong stage can't block the others
        for bs in best[stage]:
            best[stage][bs] = sorted(best[stage][bs], key=lambda v: v[0])[:2]
    return best


def tune(state, weights, deadline):
    tuner = _Tuner(min(TUNE_BUDGET_S, deadline - time.monotonic()))
    ref = _Ref(weights)
    tlog(f"start, budget {tuner.budget:.0f}s; incumbents: M11 E3c up to 32 tokens, v1 above")
    try:
        _bw_recon(tuner, state)
    except Exception as exc:
        tlog(f"bw recon FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    try:
        rows = []  # E22: registers / spill / static smem of the small-batch M11 kernels (incumbent, CR, SR)
        for bit, insts in ((0, M13_M11_INSTANCES), (16, M13_M11_CR_INSTANCES), (32, M13_M11_SR_INSTANCES)):
            for g, tt, nt, d, ks in insts:
                r, loc, shm = state["m11_ext"].m11_attrs(g, tt, nt, d | bit, ks).tolist()
                rows.append(f"{'gpc'[0 if g else 1]}{'' if not bit else 'R' if bit == 16 else 'S'}{tt}.{nt}d{d}k{ks} {r}r"
                            f"{f' {loc}B' if loc else ''} {shm // 1024}K")
        tlog("m11 attrs: " + ", ".join(rows))
    except Exception as exc:
        tlog(f"m11 attrs FAILED {type(exc).__name__}: {str(exc)[-160:]}")
    try:  # E27: registers / spill / static smem / CTAs per SM of every M11b instance (occupancy ones: minb)
        rows = []
        for g, tt, nt, d, ks, mb, e in _m11b_all():  # E30: + early-refill builds (label E)
            r, loc, shm, blk = state["m11_ext"].m11b_attrs(g, tt, nt, d | (mb << 4 if mb > 1 else 0) | (256 if e else 0), ks).tolist()
            rows.append(f"{'gp'[0 if g else 1]}{tt}.{nt}d{d}k{ks}{f'o{mb}' if mb > 1 else ''}{'E' if e else ''} {r}r{f' {loc}B' if loc else ''} "
                        f"{shm // 1024}K {blk}c")
        tlog("m11b attrs: " + ", ".join(rows))
    except Exception as exc:
        tlog(f"m11b attrs FAILED {type(exc).__name__}: {str(exc)[-160:]}")
    _m11b_recon(tuner, state)  # E29
    m14 = {1: {}, 2: {}}
    try:
        m14 = _m14_stage_search(tuner, state, (1, 8, 16, 32, 64, 128, 256))
    except Exception as exc:
        tlog(f"m14 search FAILED {type(exc).__name__}: {str(exc)[-200:]}")
        _clean(state)
    configs = dict(state["m13_configs"])
    lo = 1
    for hi, bs in TUNE_CLASSES:
        if tuner.left() < 15:
            tlog("out of budget; remaining classes keep their incumbents")
            break
        cands = []
        if bs <= M13_MAX_TOKENS:
            for s1 in TUNE_M13_S1:
                for s2 in TUNE_M13_S2:
                    for pfd in TUNE_PFD:
                        for p1 in TUNE_M13_S1_PF:
                            for p2 in TUNE_M13_S2_PF:
                                for pdl, ppdl in TUNE_PDL:
                                    cands.append(dict(s1=dict(s1, pfd=pfd, **p1),
                                                      s2=dict(s2, pfd=pfd, pdl=pdl, pf=0, **p2),
                                                      post=dict(kind="m13", pdl=ppdl)))
        if bs in m14[1] and bs in m14[2]:  # M15 at every size, M14 from 64 tokens
            for _, s1b in m14[1][bs]:
                for _, s2b in m14[2][bs]:
                    for pfd in (TUNE_PFD if "m14" in (s1b["kind"], s2b["kind"]) else (0,)):
                        for pdl, ppdl in TUNE_PDL:
                            s1c = dict(s1b, pfd=pfd) if s1b["kind"] == "m14" else dict(s1b)
                            s2c = dict(s2b, pfd=pfd) if s2b["kind"] == "m14" else dict(s2b)
                            cands.append(dict(s1=s1c, s2=dict(s2c, pdl=pdl, pf=0), post=dict(kind="m13", pdl=ppdl)))
        if bs >= 128 and bs in m14[1] and bs in m14[2]:  # E18: hot experts in a second launch (MF19)
            c1 = [st for _, st in m14[1][bs] if st["kind"] in ("m14", "m18", "m19")]
            c2 = [st for _, st in m14[2][bs] if st["kind"] in ("m14", "m18", "m19")]
            if c1 and c2:
                for thr in TUNE_SPLIT_T:
                    for h1 in TUNE_HOT1:
                        for h2 in TUNE_HOT2:
                            cands.append(dict(s1=dict(c1[0], thr=thr, hot=dict(h1)),
                                              s2=dict(c2[0], pdl=1, pf=0, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
        if bs >= 128:  # E26 (T34): M11b for the cold experts (<= T pairs), M19/M14/M18 for the hot ones
            c2 = [st for _, st in m14[2].get(bs, ()) if st["kind"] in ("m14", "m18", "m19")][:1]
            for thr in TUNE_COLD11B_T:
                for h1, h2 in TUNE_HOT_PAIRS:
                    for i1, s1x in enumerate(TUNE_COLD11B_S1):
                        s1c = dict(s1x, thr=thr, hot=dict(h1))
                        for i2, s2x in enumerate(TUNE_COLD11B_S2):
                            cands.append(dict(s1=dict(s1c), s2=dict(s2x, pdl=1, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
                        if c2:  # cold GEMM2 on the stage search's best M19/M14/M18 stage (E25's choice)
                            cands.append(dict(s1=dict(s1c), s2=dict(c2[0], pdl=1, pf=0, thr=thr, hot=dict(h2)),
                                              post=dict(kind="m13", pdl=1)))
        inc11 = m11_config_for(state, bs)
        if inc11 is not None:  # the M11 incumbent with a PDL tail: GEMM2 + post-norm, or the post-norm only (E15)
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"], kind="m11p"), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
            cands.append(dict(s1=dict(inc11["s1"]), s2=dict(inc11["s2"]), post=dict(kind="m13", pdl=1),
                              margin=TUNE_MARGIN_PDL))
        if inc11 is not None:  # E21: M11 with its ring streaming across units (MF23), at the incumbent's c and at 4
            for c1 in sorted({inc11["s1"]["c"], 4}):
                for c2 in sorted({inc11["s2"]["c"], 4}):
                    s1x, s2x = dict(inc11["s1"], cr=1, c=c1), dict(inc11["s2"], cr=1, c=c2)
                    cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(s1x), s2=dict(s2x), post=dict(kind="m13", pdl=1)))
        if inc11 is not None:  # E24: M11b (M11 on the M13 bank and decode) for GEMM1, GEMM2 or both
            s1i, s2i = inc11["s1"], inc11["s2"]
            b1 = (1, 2, s1i["nt"], s1i["d"], s1i["ks"]) in M11B_INSTANCES
            b2 = (0, s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"]) in M11B_INSTANCES
            if b1:
                s1b = dict(s1i, kind="m11b")
                for s2x in ([dict(s2i, kind="m11b", pdl=1), dict(s2i, kind="m11b", pdl=0)] if b2 else []) + \
                           [dict(s2i, kind="m11p"), dict(s2i)]:
                    cands.append(dict(s1=dict(s1b), s2=s2x, post=dict(kind="m13", pdl=1)))
                for p in (1, 0):  # with G2P for GEMM2 (E23)
                    if (s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"], p) in M11F_INSTANCES:
                        cands.append(dict(s1=dict(s1b), s2=dict(s2i, kind="m11f", pdl=p), post=dict(kind="m11f", pdl=0)))
            if b2:
                cands.append(dict(s1=dict(s1i), s2=dict(s2i, kind="m11b", pdl=1), post=dict(kind="m13", pdl=1)))
        if bs in TUNE_OCC_BASE:  # E27: M11b at 5-6 CTAs per SM (MF29), GEMM1, GEMM2 or both
            b1, b2 = TUNE_OCC_BASE[bs]
            v1, v2 = _occ_variants(b1, True), _occ_variants(b2, False)
            post = dict(kind="m13", pdl=1)
            cands.append(dict(s1=dict(b1), s2=dict(b2), post=dict(post)))  # the base itself (E26's pick)
            cands += [dict(s1=x, s2=dict(b2), post=dict(post)) for x in v1]
            cands += [dict(s1=dict(b1), s2=y, post=dict(post)) for y in v2]
            cands += [dict(s1=x, s2=y, post=dict(post)) for x in v1 for y in v2 if x["occ"] == y["occ"]]
            e1, e2 = _early_variants(b1, True), _early_variants(b2, False)  # E30 (MF32): refill before decode
            cands += [dict(s1=x, s2=dict(b2), post=dict(post)) for x in e1]
            cands += [dict(s1=dict(b1), s2=y, post=dict(post)) for y in e2]
            cands += [dict(s1=x, s2=y, post=dict(post)) for x in e1 for y in e2]
        if bs in (16, 32):  # E26 (T34b): experts above 8 pairs take one 16-pair pass instead of two 8-pair passes
            for s1x, s2x in ((dict(nt=1, d=4, ks=2, c=8), dict(tt=2, nt=1, d=4, ks=4, c=8)),
                             (dict(nt=1, d=4, ks=4, c=8), dict(tt=1, nt=1, d=4, ks=4, c=4))):
                for h1, h2 in ((dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8)),
                               (dict(nt=2, d=2, ks=4, c=8), dict(tt=2, nt=2, d=2, ks=4, c=8)),
                               (dict(nt=2, d=4, ks=2, c=4), dict(tt=2, nt=2, d=4, ks=2, c=4))):
                    cands.append(dict(s1=dict(s1x, kind="m11b", thr=8, hot=dict(h1, kind="m11b")),
                                      s2=dict(s2x, kind="m11b", pdl=1, thr=8, hot=dict(h2, kind="m11b")),
                                      post=dict(kind="m13", pdl=1)))
        if inc11 is not None and bs == 32:  # E25: M11b at 17-32 tokens (M19 holds the class since E20)
            for s1x, s2x in ((dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8)),
                             (dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=4)),
                             (dict(nt=2, d=2, ks=4, c=8), dict(tt=2, nt=2, d=2, ks=4, c=8)),
                             (dict(nt=1, d=4, ks=2, c=8), dict(tt=2, nt=1, d=4, ks=4, c=8))):
                for p in (1, 0):
                    cands.append(dict(s1=dict(s1x, kind="m11b"), s2=dict(s2x, kind="m11b", pdl=p), post=dict(kind="m13", pdl=1)))
        if inc11 is not None:  # E23: G2P = M11 GEMM2 + post-norm in one kernel (prefill's shared/glue, M11F_INSTANCES)
            s2i = inc11["s2"]
            for p in (1, 0):
                if (s2i["tt"], s2i["nt"], s2i["d"], s2i["ks"], p) in M11F_INSTANCES:
                    cands.append(dict(s1=dict(inc11["s1"]), s2=dict(s2i, kind="m11f", pdl=p), post=dict(kind="m11f", pdl=0)))
        if inc11 is not None:  # E22: M11 with an 8-deep shared-memory weight ring (MF24), c 4/5/8
            for c1 in sorted({inc11["s1"]["c"], 4, 5}):
                for c2 in sorted({inc11["s2"]["c"], 4}):
                    s1x, s2x = dict(inc11["s1"], sr=1, c=c1), dict(inc11["s2"], sr=1, c=c2)
                    cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1)))
                    cands.append(dict(s1=dict(s1x), s2=dict(s2x), post=dict(kind="m13", pdl=1)))
        if inc11 is not None and bs == 32:  # decode 32 (E17): M11 configs with 4 K chunks in flight per warp (E3c: d2)
            for s1x, s2x in ((dict(nt=1, d=4, ks=2, c=8), dict(inc11["s2"])),
                             (dict(nt=2, d=4, ks=2, c=8), dict(inc11["s2"])),
                             (dict(nt=2, d=4, ks=2, c=8), dict(tt=2, nt=2, d=4, ks=2, c=8))):
                cands.append(dict(s1=s1x, s2=dict(s2x, kind="m11p"), post=dict(kind="m13", pdl=1), margin=0.97))
        runs = [_case(bs, 1700 + bs + 2000 * r) for r in range(TUNE_ROUTINGS)]
        try:
            t_inc = sum(tuner.time_graph(lambda: _run(state, None, x, ids, w)) for x, ids, w in runs) / TUNE_ROUTINGS
            timed = []
            nfail = 0
            for cfg in cands:
                if tuner.left() < 15:
                    break
                try:  # E22b: a refused or faulty candidate is skipped, not the whole class (E22 lost every small class)
                    timed.append((sum(tuner.time_graph(lambda: _run(state, cfg, x, ids, w)) for x, ids, w in runs)
                                  / TUNE_ROUTINGS, cfg))
                except Exception as exc:
                    if nfail == 0:
                        tlog(f"class [{lo},{hi}) candidate {_name(cfg)} FAILED {type(exc).__name__}: {str(exc)[-140:]}")
                    nfail += 1
                _clean(state)
            if nfail:
                tlog(f"class [{lo},{hi}) {nfail} candidate(s) failed and were skipped")
            pick, why = None, "no candidate faster"
            for t, cfg in sorted(timed, key=lambda v: v[0]):
                if t >= cfg.get("margin", TUNE_MARGIN) * t_inc:
                    continue
                worst = _valid(state, ref, cfg, lo, hi)
                if worst is not None and worst[0] < TUNE_MAX_ERR and worst[1] < TUNE_MAX_ROW_ERR:
                    pick, why = (t, cfg, worst), "ok"
                    break
                tlog(f"class [{lo},{hi}) rejected {_name(cfg)} {t:.1f}us: "
                     + ("dirty scratch" if worst is None else f"err {worst[0]:.1e} row {worst[1]:.1e}"))
            grid = [(c["s1"].get("npre", 0), c["s1"].get("nxq", 0), c["s2"].get("npre", 0), c["s2"].get("pdl", 0), t)
                    for t, c in timed if c["s1"].get("kind") == "m13"]
            if grid:  # every start-up prefetch combination: s1 npre/nxq, then s2 npre 0/255 x pdl 0/1
                cells = {}
                for n1, q, n2, pdl, t in grid:
                    cells.setdefault((n1, q), {})[(n2, pdl)] = t
                tlog(f"class [{lo},{hi}) m13 s1(npre,nxq): s2 npre0 pdl0/1, npre255 pdl0/1 | " + " ".join(
                    f"{n1},{q}:" + "/".join(f"{v[k]:.1f}" if k in v else "-" for k in ((0, 0), (0, 1), (255, 0), (255, 1)))
                    for (n1, q), v in sorted(cells.items())))
            top = " ".join(f"{_name(c)}={t:.1f}" for t, c in sorted(timed, key=lambda v: v[0])[:4])
            if pick:
                configs[hi] = pick[1]
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us -> {_name(pick[1])} {pick[0]:.1f}us "
                     f"({t_inc / pick[0]:.2f}x, err {pick[2][0]:.1e}) | best: {top}")
            else:
                tlog(f"class [{lo},{hi}) bs{bs}: incumbent {t_inc:.1f}us kept ({why}) | best: {top}")
        except Exception as exc:
            tlog(f"class [{lo},{hi}) FAILED {type(exc).__name__}: {str(exc)[-200:]}; incumbent kept")
            _clean(state)
        lo = hi
    state["m13_configs"] = tuple(sorted(configs.items()))
    ref.cache.clear()
    tuner.flush = None
    _clean(state)
    torch.cuda.empty_cache()
    tlog(f"done in {time.monotonic() - tuner.t0:.0f}s; M13_CONFIGS = {state['m13_configs']!r}"[:1500])


def prepare(weights):
    t0 = time.monotonic()
    state = _m13_prepare(weights)
    tlog(f"base prepare (v1, nvcc, banks, warm-up) took {time.monotonic() - t0:.1f}s")
    if "m13" in state:
        try:
            tune(state, weights, t0 + TUNE_PREPARE_LIMIT_S)
        except Exception as exc:  # never fail prepare: the incumbents stay
            tlog(f"tune FAILED {type(exc).__name__}: {str(exc)[-300:]}")
            state["m13_configs"] = M13_CONFIGS
        for _, cfg in state["m13_configs"]:
            if cfg is not None:
                m13_warm(state, cfg)
        # (E14's verbose ptxas probe build is gone: its output filled the 16,000-char log tail; result in MF15)
        tlog(f"prepare done in {time.monotonic() - t0:.0f}s")
    return state
