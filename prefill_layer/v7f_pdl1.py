# okc: problem=muse_prefill_layer
"""v5: SGLang's decoder layer (baseline.py, commit a55ab7d5) with our own routed-expert paths.

* R14 (per token class, where a lab measured it correct and faster): Hopper W4A16 NVFP4 grouped GEMMs in
  CUDA. Weights are wgmma's register A operand, decoded from NVFP4 in registers (weights repacked
  losslessly in prepare into fragment order, scales as exact bf16); tokens are the shared-memory B operand
  (128B swizzle, written pre-swizzled by the gather); a producer warp streams each stage with two bulk
  async copies into an mbarrier ring; two consumer warpgroups decode and run wgmma; SwiGLU on the
  accumulators; SGLang's post-expert norm weighted sum over the per-route rows.
* v2 from 8192 tokens otherwise: touched experts decoded exactly into a BF16 scratch each call,
  torch._grouped_mm for both expert GEMMs.
* Otherwise, and whenever prepare()'s GPU check against Marlin fails, SGLang's Marlin path.

Everything else is SGLang's code path, call for call.

SGLang's decoder layer for Muse Spark Open, as one H200 runs it under TP8/EP8, at SGLang commit
a55ab7d5ae1bba59ac351e55026fe7f7644ecc5f (MuseSpark1xDecoderLayer.forward), with the same modules:

  qkv_gate projection              bf16_gemm_dispatch (cuBLAS on H200)
  FusedQKNorm (Triton)             per-head RMSNorm of q and k, read from the fused projection,
                                   with the interleaved RoPE fused in on sliding layers
  cache store, q to FP8            as FlashAttentionBackend and set_kv_buffer do for an FP8 cache
                                   (cast, then the store_cache JIT kernel at the step's rows)
  flash_attn_with_kvcache (FA3)    paged, causal, window (2047, 0) on sliding layers
  FusedRMSNormGate (JIT)           per-head output norm x sigmoid(gate)
  o projection                     bf16_gemm_dispatch; then the all-reduce (identity here)
  FusedPostNormResidualPreNorm     post-attention norm + residual gate + pre-FFN norm (JIT);
                                   it also writes the FP32 router input up to 8 tokens
  router                           FP32 F.linear up to 8 tokens, above that two BF16 GEMMs
                                   on a hi/lo split of the weight (MuseSpark1xRouter); on a
                                   side stream, overlapping the pooler
  FusedMuseSparkRouting (JIT)      sigmoid, biased top-8, renormalize (on the side stream)
  pooler                           bf16_gemm_dispatch, replicated on every GPU
  pre-expert norm                  MuseSpark1xRMSNorm (F.rms_norm)
  experts                          moe_align_block_size, two moe_wna16_marlin_gemm (W4A16),
                                   silu_and_mul, Triton post-expert norm weighted sum
                                   (MuseSpark1xFp4Experts._forward_marlin); then the
                                   all-reduce (identity here)
  unpooler                         up to 8 tokens, this GPU's 1/8 slice of the weight, then
                                   the all-gather (identity here: zeros in the other columns);
                                   above that, the whole weight (bf16_gemm_dispatch)
  FusedPostNormResidualPreNorm     post-FFN norm + residual gate + the next layer's pre-attention
                                   norm (JIT)

Differs from SGLang:
  * The step's positions, cache rows, FA3 lengths and FA3 scheduler metadata come in with the
    inputs; SGLang builds them once per step for all layers.
  * All-reduces are the identity; the all-gather is a zero pad of the unpooler's output."""

from typing import TypedDict

import types

import torch
import torch.nn.functional as F
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sgl_kernel.scalar_type import scalar_types
from torch import nn

from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_muse_spark_routing import FusedMuseSparkRouting
from sglang.kernels.ops.muse_spark_v12.fused_norm_gate import FusedRMSNormGate
from sglang.kernels.ops.muse_spark_v12.fused_norm_residual import FusedPostNormResidualPreNorm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.muse_spark_layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl


class Weights(TypedDict):
    qkv_gate: torch.Tensor            # bfloat16 [2304, 8192]: q 16x64, k 2x64, v 2x64, output gate 16x64
    o: torch.Tensor                   # bfloat16 [8192, 1024]
    attn_residual_gate: torch.Tensor  # bfloat16 [8192], the post-attention residual gate's weight
    pre_ffn_norm: torch.Tensor        # bfloat16 [8192] (applied as 1 + w)
    router: torch.Tensor              # float32 [256, 8192]
    router_bias: torch.Tensor         # float32 [256], added to the scores only to choose experts
    pooler: torch.Tensor              # bfloat16 [4096, 8192]
    pre_expert_norm: torch.Tensor     # bfloat16 [4096] (applied as 1 + w)
    w13: torch.Tensor                 # uint8 [32, 8192, 2048]: this GPU's experts, gate then up rows, packed E2M1
    w13_sf: torch.Tensor              # float8_e4m3fn [32, 8192, 256]
    w13_gs: torch.Tensor              # float32 [32]
    w2: torch.Tensor                  # uint8 [32, 4096, 2048]
    w2_sf: torch.Tensor               # float8_e4m3fn [32, 4096, 256]
    w2_gs: torch.Tensor               # float32 [32]
    post_expert_norm: torch.Tensor    # bfloat16 [4096] (applied as 1 + w, before normalizing)
    unpooler: torch.Tensor            # bfloat16 [8192, 4096]; up to 8 tokens only rows 0-1023 (this GPU's slice)
    post_ffn_norm: torch.Tensor       # bfloat16 [8192] (applied as 1 + w, before normalizing)
    ffn_residual_gate: torch.Tensor   # bfloat16 [8192]
    next_pre_attn_norm: torch.Tensor  # bfloat16 [8192], the next layer's input norm (applied as 1 + w)


class Config(TypedDict):
    batch: int            # B sequences
    q_len: int            # L: 1 for decode, 8 for a verify block (causal inside it); 200-16384 on muse_prefill_layer
    tokens: int           # T = B * L, request-major
    hidden: int           # 8192
    num_q_heads: int      # 16 per GPU under TP8
    num_kv_heads: int     # 2
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers (interleaved RoPE); 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)
    n_experts: int        # 256
    top_k: int            # 8
    local_experts: int    # 32: this GPU computes experts local_start .. local_start + 31


class Inputs(TypedDict):
    stream: torch.Tensor      # float32 [T, 8192], the residual stream
    attn_input: torch.Tensor  # bfloat16 [T, 8192], the normalized stream (the previous layer's boundary wrote it)
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, 2, 64], written in place
    positions: torch.Tensor   # int64 [T], each new token's position
    cache_loc: torch.Tensor   # int64 [T], each new token's flat cache row
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1], 0, L, 2L, ...
    scheduler_metadata: torch.Tensor | None  # int32, FA3's tile schedule on full-attention decode steps
    k_scale: torch.Tensor     # float32 [], key = e4m3 * k_scale
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages], physical page of each logical page (-1 unused)
    seq_lens: torch.Tensor    # int32 [B], cached tokens per request; the block starts at this position
    local_start: torch.Tensor # int32 [1], this GPU's first expert (a multiple of 32; its weights are w13[0], ...)
    config: Config


HIDDEN = 8192
RMS_EPS = 1e-5
POST_NORM_EPS = 1e-8
FP8 = torch.float8_e4m3fn
N_EXPERTS = 256
TOP_K = 8
LOCAL_EXPERTS = 32
D_EXPERT = 4096
# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8
# Up to this many tokens the unpooler runs on this GPU's slice and is all-gathered.
SLICED_UNPOOL_MAX_TOKENS = 8
UNPOOL_SLICE = HIDDEN // 8
# MuseSpark1xFp4Experts' candidates, smallest first.
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def _gemm(x, w):
    return _bf16_gemm_dispatch_impl(x, w, None)


def _param(w):
    return nn.Parameter(w.contiguous(), requires_grad=False)


def _norm(cls, w, size, eps, **kw):
    """A Muse norm module holding w (applied as 1 + w), as the checkpoint loads it."""
    norm = cls(size, eps=eps, has_weight=w is not None, zero_centered_gamma=True, **kw)
    if w is not None:
        norm.weight = _param(w)
        norm.precompute_effective_weight()
    return norm


def _residual(w):
    gate = MuseSpark1xNormPreservingResidualConnection(HIDDEN, temperature=0.3)
    gate.gate = _param(w)
    gate.precompute_coefficients()
    return gate


def _marlin_bank(weights):
    """The local experts in Marlin's layout, via SGLang's own loader step."""
    device = weights["w13"].device
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"].contiguous(),
        w2_weight=weights["w2"].contiguous(),
        w13_weight_scale=weights["w13_sf"].contiguous(),
        w2_weight_scale=weights["w2_sf"].contiguous(),
        w13_weight_scale_2=weights["w13_gs"].contiguous(),
        w2_weight_scale_2=weights["w2_gs"].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
    expert_map[:LOCAL_EXPERTS] = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return {
        "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
        "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
        "expert_map": expert_map, "size_n": layer.intermediate_size_per_partition,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
    }


def prepare(weights):
    device = weights["qkv_gate"].device
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    # [2^21, 64] float32 cos | sin, as SGLang's RotaryEmbedding builds it.
    inv_freq = 1.0 / (ROPE_THETA ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    freqs = torch.einsum("i,j -> ij", torch.arange(ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
    pre_ffn = _norm(MuseSpark1xRMSNorm, weights["pre_ffn_norm"], HIDDEN, RMS_EPS)
    next_pre_attn = _norm(MuseSpark1xRMSNorm, weights["next_pre_attn_norm"], HIDDEN, RMS_EPS)
    post_attn = _norm(MuseSpark1xRMSNorm, None, HIDDEN, POST_NORM_EPS, is_post_norm=True,
                      post_norm_gain_center_type="one", residual_id=1)
    post_ffn = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_ffn_norm"], HIDDEN, POST_NORM_EPS,
                     is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    post_expert = _norm(MuseSpark1xRMSNormWithInputScale, weights["post_expert_norm"], D_EXPERT, POST_NORM_EPS,
                        is_post_norm=True, post_norm_gain_center_type="one", residual_id=2)
    state = {
        "dense": {name: weights[name].contiguous() for name in ("qkv_gate", "o", "pooler", "unpooler")},
        "router": router.contiguous(),
        "router_hi_t": hi.t().contiguous(),
        "router_lo_t": lo.t().contiguous(),
        "routing": FusedMuseSparkRouting(expert_bias=weights["router_bias"].float().contiguous(), route_scale=1.0,
                                         router_output_multiplier=0.1767766952966369,
                                         router_score_division_eps=1e-15),
        "cos_sin": torch.cat((freqs.cos(), freqs.sin()), dim=-1),
        "qk_norm": FusedQKNorm(head_dim=64, num_q_heads=16, num_kv_heads=2, eps=RMS_EPS),
        "o_norm_gate": FusedRMSNormGate(MuseSpark1xRMSNorm(64, eps=RMS_EPS, has_weight=False), True, 64),
        "attn_boundary": FusedPostNormResidualPreNorm(post_attn, _residual(weights["attn_residual_gate"]), pre_ffn),
        "ffn_boundary": FusedPostNormResidualPreNorm(post_ffn, _residual(weights["ffn_residual_gate"]), next_pre_attn),
        "pre_expert_norm": _norm(MuseSpark1xRMSNorm, weights["pre_expert_norm"], D_EXPERT, RMS_EPS),
        # MuseSpark1xFp4Experts binds the post-expert norm's effective (1 + w) weight.
        "post_expert_w": post_expert._get_effective_weight().detach().to(torch.bfloat16),
        "bank": _marlin_bank(weights),
        "side_stream": torch.cuda.Stream(device),
        "_modules": (pre_ffn, next_pre_attn, post_attn, post_ffn, post_expert),
    }
    _gmm_prepare(state, weights)
    _r14_prepare(state, weights)
    del freqs
    torch.cuda.synchronize()
    return state


def _experts(x, topk_weights, topk_ids, state):
    """MuseSpark1xFp4Experts._marlin_expert_outputs and its post-expert-norm finalize."""
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for block_size_m in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / LOCAL_EXPERTS / block_size_m < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, block_size_m, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=block_size_m, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = bank["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["post_expert_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


# ---------------------------------------------------------------------------------------------
# v2: the routed experts as BF16 grouped GEMMs from 4096 tokens up (local/make_v2.py inserts this).
#
# Each call, the NVFP4 weights of every local expert that got a route are decoded into a BF16
# scratch: E2M1 code x E4M3 block scale, exact in BF16 (at most 6 significant bits); the FP32
# global scale is applied to the GEMM outputs instead, so no weight is rounded. The local routes
# are sorted by expert on the GPU (no host sync), gathered, and run through torch._grouped_mm
# (CUTLASS wgmma) twice, with SiLU-and-mul and the post-expert norm weighted sum as Triton
# kernels that read the sorted rows. Below 4096 tokens SGLang's Marlin path runs unchanged.
# prepare() checks the path against Marlin on the GPU and keeps Marlin if it disagrees.
# ---------------------------------------------------------------------------------------------

import triton
import triton.language as tl

GMM_MIN_TOKENS = 4096
GMM_SELFTEST_TOKENS = 4096
GMM_SELFTEST_MAX_ERR = 2e-2
DEQ_BLOCK_N = 32      # rows per dequant program
DEQ_BLOCK_KB = 128    # packed bytes per dequant program (256 values)


@triton.jit
def _e2m1_value(code):
    """E2M1 code (low 4 bits of an int32) -> float32 value."""
    sign = (code >> 3) & 1
    exp = (code >> 1) & 3
    man = (code & 1).to(tl.float32)
    normal = (1.0 + 0.5 * man) * tl.where(exp == 3, 4.0, tl.where(exp == 2, 2.0, 1.0))
    mag = tl.where(exp == 0, 0.5 * man, normal)
    return tl.where(sign == 1, -mag, mag)


@triton.jit
def _nvfp4_dequant_kernel(q_ptr, sf_ptr, counts_ptr, out_ptr, N: tl.constexpr, KB: tl.constexpr,
                          BN: tl.constexpr, BKB: tl.constexpr):
    """out[e, n, :] = e2m1(q[e, n, :]) * sf[e, n, :/16] in BF16 (element 2i = low nibble), for experts
    with at least one local route."""
    e = tl.program_id(0)
    if tl.load(counts_ptr + e) == 0:
        return
    rows = tl.program_id(1) * BN + tl.arange(0, BN)
    cols = tl.program_id(2) * BKB + tl.arange(0, BKB)
    e64 = e.to(tl.int64)
    q = tl.load(q_ptr + e64 * (N * KB) + rows[:, None] * KB + cols[None, :]).to(tl.int32)
    sf = tl.load(sf_ptr + e64 * (N * (KB // 8)) + rows[:, None] * (KB // 8) + (cols // 8)[None, :]).to(tl.float32)
    lo = _e2m1_value(q & 15) * sf
    hi = _e2m1_value(q >> 4) * sf
    vals = tl.reshape(tl.join(lo, hi), (BN, 2 * BKB))
    out_cols = tl.program_id(2) * (2 * BKB) + tl.arange(0, 2 * BKB)
    tl.store(out_ptr + e64 * (N * 2 * KB) + rows[:, None] * (2 * KB) + out_cols[None, :], vals.to(tl.bfloat16))


@triton.jit
def _gather_rows_kernel(x_ptr, perm_ptr, offs_ptr, out_ptr, H: tl.constexpr, TOPK: tl.constexpr,
                        NE: tl.constexpr):
    """out[r] = x[perm[r] // TOPK] for the local routes (r < offs[NE - 1])."""
    r = tl.program_id(0)
    if r >= tl.load(offs_ptr + NE - 1):
        return
    tok = (tl.load(perm_ptr + r) // TOPK).to(tl.int64)
    cols = tl.arange(0, H)
    tl.store(out_ptr + r.to(tl.int64) * H + cols, tl.load(x_ptr + tok * H + cols))


@triton.jit
def _gmm_act_kernel(h_ptr, keys_ptr, gs_ptr, offs_ptr, out_ptr, I: tl.constexpr, NE: tl.constexpr):
    """out[r] = silu(gs * gate) * (gs * up) in BF16, gs = the row's expert's global scale."""
    r = tl.program_id(0)
    if r >= tl.load(offs_ptr + NE - 1):
        return
    gs = tl.load(gs_ptr + tl.load(keys_ptr + r))
    cols = tl.arange(0, I)
    base = r.to(tl.int64) * (2 * I)
    gate = tl.load(h_ptr + base + cols).to(tl.float32) * gs
    up = tl.load(h_ptr + base + I + cols).to(tl.float32) * gs
    tl.store(out_ptr + r.to(tl.int64) * I + cols, (gate * tl.sigmoid(gate) * up).to(tl.bfloat16))


@triton.jit
def _gmm_combine_kernel(y_ptr, ids_ptr, inv_ptr, gs_ptr, s_ptr, w_ptr, out_ptr, eps, H: tl.constexpr,
                        TOPK: tl.constexpr):
    """SGLang's post-expert norm weighted sum (weight first), reading each local slot's row of the
    sorted GEMM output and applying its expert's global scale."""
    tok = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(w_ptr + offs).to(tl.float32)
    acc = tl.zeros([H], dtype=tl.float32)
    for k in tl.static_range(TOPK):
        expert = tl.load(ids_ptr + tok * TOPK + k)
        if expert >= 0:
            row = tl.load(inv_ptr + tok * TOPK + k)
            x = tl.load(y_ptr + row * H + offs).to(tl.float32) * tl.load(gs_ptr + expert)
            y = x * w
            var = tl.sum(y * y, axis=0) / H
            y = y * tl.rsqrt(var + eps)
            acc += y * tl.load(s_ptr + tok * TOPK + k).to(tl.float32)
    tl.store(out_ptr + tok * H + offs, acc.to(tl.bfloat16))


def _dequant(q, sf, counts, out):
    experts, n, kb = q.shape
    _nvfp4_dequant_kernel[(experts, n // DEQ_BLOCK_N, kb // DEQ_BLOCK_KB)](
        q, sf, counts, out, N=n, KB=kb, BN=DEQ_BLOCK_N, BKB=DEQ_BLOCK_KB, num_warps=4)


def _experts_gmm(x, topk_weights, topk_ids, state):
    """Same contract as _experts: x [T, 4096] BF16, topk_ids in this GPU's numbering."""
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
    _dequant(gmm["w13_q"], gmm["w13_sf"], counts, gmm["w13"])
    _dequant(gmm["w2_q"], gmm["w2_sf"], counts, gmm["w2"])
    h1 = torch._grouped_mm(a, gmm["w13"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    act = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_act_kernel[(slots,)](h1, sorted_keys, gmm["w13_gs"], offs, act, I=D_EXPERT, NE=LOCAL_EXPERTS,
                              num_warps=4)
    y2 = torch._grouped_mm(act, gmm["w2"].transpose(-2, -1), offs=offs, out_dtype=torch.bfloat16)
    out = torch.empty((tokens, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe(x, topk_weights, topk_ids, state, tokens):
    if tokens >= GMM_MIN_TOKENS and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def _gmm_selftest_routing(tokens, device, seed=0):
    """Random top-8 over 256 experts, tilted so ~1/4 of routes hit experts 0-31 (like the busiest GPU)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
    scores[:, :LOCAL_EXPERTS] += 0.35
    vals, ids = scores.topk(TOP_K, dim=-1)
    return ids.to(torch.int32), (vals / vals.sum(-1, keepdim=True)).float()


def _gmm_prepare(state, weights):
    """Raw NVFP4 weights (same tensors, same precision), BF16 scratch for the per-call decode, and
    a GPU check of the grouped path against SGLang's Marlin path (which also compiles every kernel)."""
    device = weights["w13"].device
    gmm = {
        "w13_q": weights["w13"].contiguous(), "w13_sf": weights["w13_sf"].contiguous(),
        "w2_q": weights["w2"].contiguous(), "w2_sf": weights["w2_sf"].contiguous(),
        "w13_gs": weights["w13_gs"].float().contiguous(), "w2_gs": weights["w2_gs"].float().contiguous(),
        "ok": False,
    }
    experts, n13, kb13 = gmm["w13_q"].shape
    _, n2, kb2 = gmm["w2_q"].shape
    gmm["w13"] = torch.empty((experts, n13, 2 * kb13), device=device, dtype=torch.bfloat16)
    gmm["w2"] = torch.empty((experts, n2, 2 * kb2), device=device, dtype=torch.bfloat16)
    state["gmm"] = gmm
    try:
        gen = torch.Generator(device=device).manual_seed(1)
        x = (torch.randn((GMM_SELFTEST_TOKENS, D_EXPERT), generator=gen, device=device) * 0.5).to(torch.bfloat16)
        ids, w = _gmm_selftest_routing(GMM_SELFTEST_TOKENS, device)
        ref = _experts(x, w, ids, state).float()
        out = _experts_gmm(x, w, ids, state).float()
        torch.cuda.synchronize()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
        row_err = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
        finite = bool(torch.isfinite(out).all())
        gmm["ok"] = finite and err < GMM_SELFTEST_MAX_ERR and row_err < 5 * GMM_SELFTEST_MAX_ERR
        print(f"[v2] grouped path vs Marlin at {GMM_SELFTEST_TOKENS} tokens: rel err {err:.2e}, worst row "
              f"{row_err:.2e}, finite {finite} -> {'grouped' if gmm['ok'] else 'Marlin'} from {GMM_MIN_TOKENS} tokens",
              flush=True)
    except Exception as exc:
        print(f"[v2] grouped path failed its check, keeping Marlin: {type(exc).__name__}: {str(exc)[-400:]}",
              flush=True)
        gmm["ok"] = False
    torch.cuda.synchronize()

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
# ---------------------------------------------------------------------------------------------

import tempfile

from torch.utils.cpp_extension import load_inline

# A config is (depth, smax, mode): DEPTH wgmma groups in flight (1 or 2; 2 triple-buffers the decoded A
# registers), at most smax ring stages (the kernel fits as many as 200 KB allows for its tile), and the
# tiling mode (0: 128/64/32/16-row tiles; 1: 128-row tiles with a ceil16 last tile, one pass per 128 routes).
R14_VARIANTS = ((1, 4, 0), (1, 4, 1), (1, 8, 1), (1, 24, 1), (2, 8, 1))
# class -> config, for the token classes below; empty = off.
R14_CONFIGS = {'t1': (2, 8, 1), 't4': (2, 8, 1)}  # v7c: v5's classes (RE21: R14 at ft loses)
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
R14_MMA_K64(48)
R14_MMA_K64(64)
R14_MMA_K64(80)
R14_MMA_K64(96)
R14_MMA_K64(112)
R14_MMA_K64(128)

constexpr int SMEM_BYTES = 200 * 1024;  // ring bytes each CTA requests: stages = what fits for its tile
constexpr int SMAX = 24;                 // most ring slots (barriers allocated for this many)

__device__ __forceinline__ bool load_frag(const uint8_t* sW, uint32_t full0, int s, uint32_t par, int woff, int soff,
                                          uint32_t (&a)[16], const int* err) {
  const bool ok = mbar_wait(full0 + 8 * s, par, err);
  const uint4 wv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + woff);
  const uint4 sv = *reinterpret_cast<const uint4*>(sW + s * WS_BYTES + soff);
  dec_k16(wv.x, sv.x, a[0], a[1], a[2], a[3]);
  dec_k16(wv.y, sv.y, a[4], a[5], a[6], a[7]);
  dec_k16(wv.z, sv.z, a[8], a[9], a[10], a[11]);
  dec_k16(wv.w, sv.w, a[12], a[13], a[14], a[15]);
  return ok;
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
template <int DEPTH, int N>
__device__ __forceinline__ void mainloop_step(float (&acc)[N / 2], const uint32_t (&cur)[16], uint32_t (&nxt)[16],
                                              uint32_t sB0, const uint8_t* sW, uint32_t full0, uint32_t empty0,
                                              int kc, int S, Ring& rg, int woff, int soff, int tid, bool& ok,
                                              const int* err) {
  wg_fence();
  mma_k64<N>(acc, cur, make_desc(sB0 + rg.si * (N * 128)), kc == 0);
  wg_commit();
  wg_wait<DEPTH>();
#pragma unroll
  for (int i = 0; i < 16; ++i) keep(nxt[i]);
  if (kc >= DEPTH) {
    if ((tid & 127) == 0) mbar_arrive(empty0 + 8 * rg.sr);
    rg.sr = ring_next(rg.sr, S);
  }
  if (kc + 1 < KC) {
    ok &= load_frag(sW, full0, rg.sl, rg.pl, woff, soff, nxt, err);
    rg.sl = ring_next(rg.sl, S);
    if (rg.sl == 0) rg.pl ^= 1u;
  }
  rg.si = ring_next(rg.si, S);
}

template <int DEPTH, bool GATED, int N>
__device__ __forceinline__ void consume(uint8_t* sB, const uint8_t* sW, uint32_t full0, uint32_t empty0, int tid,
                                        int S, int e, int row0, int valid, int nb, const float* __restrict__ cscale,
                                        __nv_bfloat16* __restrict__ out, int* __restrict__ err, int P) {
  const int c = (tid >> 7) - 1, w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
  const int woff = ((c * 4 + w) * 32 + lane) * 16;
  const int soff = W_BYTES + ((c * 4 + w) * 8 + g) * 16;
  const uint32_t sB0 = smem_u32(sB);
  float acc[N / 2];
#pragma unroll
  for (int i = 0; i < N / 2; ++i) acc[i] = 0.f;
  uint32_t fa[16], fb[16], fc[16];
  bool ok = load_frag(sW, full0, 0, 0, woff, soff, fa, err);
  Ring rg{0, 1, 0, 0u};  // S >= 3, so chunk 1 is stage 1, parity 0
  if (DEPTH == 1) {
    for (int kc = 0; kc < KC; kc += 2) {
      mainloop_step<1, N>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err);
      mainloop_step<1, N>(acc, fb, fa, sB0, sW, full0, empty0, kc + 1, S, rg, woff, soff, tid, ok, err);
    }
  } else {
    int kc = 0;
    for (; kc + 3 <= KC; kc += 3) {
      mainloop_step<2, N>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err);
      mainloop_step<2, N>(acc, fb, fc, sB0, sW, full0, empty0, kc + 1, S, rg, woff, soff, tid, ok, err);
      mainloop_step<2, N>(acc, fc, fa, sB0, sW, full0, empty0, kc + 2, S, rg, woff, soff, tid, ok, err);
    }
    if (kc < KC) mainloop_step<2, N>(acc, fa, fb, sB0, sW, full0, empty0, kc, S, rg, woff, soff, tid, ok, err);
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
// piece = (expert, first row, valid rows, tile rows N in 16..128). B source: [KC][P][64] swizzled bf16.
// The ring holds S = min(smax, SMAX, SMEM_BYTES / (5120 + N*128)) stages (at least DEPTH + 2), sized per CTA.
template <int DEPTH, bool GATED>
__global__ void __launch_bounds__(384, 1) gemm_kernel(const uint8_t* __restrict__ wblob,
                                                      const __nv_bfloat16* __restrict__ bsrc,
                                                      const int4* __restrict__ pieces, const int* __restrict__ n_pieces,
                                                      const float* __restrict__ cscale, __nv_bfloat16* __restrict__ out,
                                                      int* __restrict__ err, int P, int NB, int smax, int pre) {
  const int nb = blockIdx.x, pc = blockIdx.y;
  // v7b (PDL): let the next kernel (GEMM2 after GEMM1) start on SMs this grid frees; it waits on
  // griddepcontrol.wait before reading our output, so this only overlaps its prologue and weight prefetch.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  if (pc >= *n_pieces) return;
  const int4 piece = pieces[pc];
  const int e = piece.x, row0 = piece.y, valid = piece.z, nsel = piece.w;
  const int bbytes = nsel * 128;
  int S = SMEM_BYTES / (WS_BYTES + bbytes);
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  S = S > DEPTH + 1 ? S : DEPTH + 2;  // the producer refills chunk kc - S after the consumers release it at
                                     // step kc - S + DEPTH, before they need chunk kc - S + DEPTH + 1: S >= DEPTH + 2
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sB = smem;
  uint8_t* sW = smem + S * bbytes;
  uint64_t* bars = reinterpret_cast<uint64_t*>(sW + S * WS_BYTES);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (tid < 128) {
    if (tid == 0) {
      const uint8_t* wsrc = wblob + static_cast<size_t>(e * NB + nb) * KC * WS_BYTES;
      bool ok = true;
      // v7f: only the first wave of a PDL-launched grid (pre = 1, linear CTA id < 132) can overlap the previous
      // kernel's tail: it issues the first S stages' weight copies (constant data) before griddepcontrol.wait and
      // their B tiles (the previous kernel's output) after it. Every other CTA keeps v5's interleaved order (RE22: a
      // pre-issue in every CTA delayed each CTA's first chunk). Each stage's full barrier expects both, as before.
      const bool early = pre != 0 && static_cast<int>(blockIdx.y * gridDim.x + blockIdx.x) < 132;
      const int npre = early ? (KC < S ? KC : S) : 0;
      for (int kc = 0; kc < npre; ++kc) {
        mbar_expect_tx(full0 + 8 * kc, WS_BYTES + bbytes);
        bulk_g2s(smem_u32(sW + kc * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * kc);
      }
      asm volatile("griddepcontrol.wait;" ::: "memory");  // no-op unless launched with PDL
      for (int kc = 0; kc < npre; ++kc)
        bulk_g2s(smem_u32(sB + kc * bbytes), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * kc);
      int s = npre == S ? 0 : npre;
      uint32_t round = npre == S ? 1u : 0u;  // kc / S, advanced without division
      for (int kc = npre; kc < KC; ++kc) {
        if (round > 0) ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
        mbar_expect_tx(full0 + 8 * s, WS_BYTES + bbytes);
        bulk_g2s(smem_u32(sW + s * WS_BYTES), wsrc + static_cast<size_t>(kc) * WS_BYTES, WS_BYTES, full0 + 8 * s);
        bulk_g2s(smem_u32(sB + s * bbytes), bsrc + (static_cast<size_t>(kc) * P + row0) * 64, bbytes, full0 + 8 * s);
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
      if (!ok) atomicAdd(err, 1);
    }
    return;
  }
#define R14_CASE(NN) \
  case NN: consume<DEPTH, GATED, NN>(sB, sW, full0, empty0, tid, S, e, row0, valid, nb, cscale, out, err, P); break;
  switch (nsel) {
    R14_CASE(16) R14_CASE(32) R14_CASE(48) R14_CASE(64) R14_CASE(80) R14_CASE(96) R14_CASE(112)
    default: consume<DEPTH, GATED, 128>(sB, sW, full0, empty0, tid, S, e, row0, valid, nb, cscale, out, err, P); break;
  }
#undef R14_CASE
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
        // mode 0: tiles 128/64/32/16; mode 1: 128-row tiles, the last one ceil16(rem) rows (one pass per 128 routes)
        const int n = rem >= 128 ? 128 : mode ? ((rem + SEG - 1) / SEG) * SEG : rem >= 64 ? 64 : rem >= 32 ? 32 : 16;
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
              int64_t max_pieces, int64_t depth, int64_t smax, bool gated, bool pdl) {
  const size_t smem = static_cast<size_t>(r14::SMEM_BYTES) + 16 * r14::SMAX + 1024;
  const dim3 grid(static_cast<unsigned>(NB), static_cast<unsigned>(max_pieces));
  const auto stream = at::cuda::getCurrentCUDAStream();
#define R14_LAUNCH(D, G)                                                                                \
  do {                                                                                                  \
    auto kern = r14::gemm_kernel<D, G>;                                                                 \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));    \
    cudaLaunchConfig_t cfg = {};                                                                        \
    cfg.gridDim = grid;                                                                                 \
    cfg.blockDim = dim3(384, 1, 1);                                                                     \
    cfg.dynamicSmemBytes = smem;                                                                        \
    cfg.stream = stream;                                                                                \
    cudaLaunchAttribute attr[1];                                                                        \
    attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                    \
    attr[0].val.programmaticStreamSerializationAllowed = 1;                                             \
    cfg.attrs = attr;                                                                                   \
    cfg.numAttrs = pdl ? 1 : 0;                                                                         \
    const cudaError_t le = cudaLaunchKernelEx(&cfg, kern, wblob.data_ptr<uint8_t>(),                    \
        reinterpret_cast<const __nv_bfloat16*>(bsrc.data_ptr()), reinterpret_cast<const int4*>(pieces.data_ptr()), \
        n_pieces.data_ptr<int>(), cscale.data_ptr<float>(), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), \
        err.data_ptr<int>(), static_cast<int>(P), static_cast<int>(NB), static_cast<int>(smax), pdl ? 1 : 0); \
    if (le != cudaSuccess) (void)cudaGetLastError();                                                    \
    TORCH_CHECK(le == cudaSuccess, "r14 launch failed: ", cudaGetErrorString(le));                      \
  } while (0)
  if (depth == 1) {
    if (gated) R14_LAUNCH(1, true); else R14_LAUNCH(1, false);
  } else {
    if (gated) R14_LAUNCH(2, true); else R14_LAUNCH(2, false);
  }
#undef R14_LAUNCH
}

// Registers and local (spill) bytes of each GEMM instance: (depth 1, gated), (1, plain), (2, gated), (2, plain).
torch::Tensor r14_attrs() {
  int64_t v[8];
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, r14::gemm_kernel<1, true>);
  v[0] = a.numRegs; v[1] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<1, false>);
  v[2] = a.numRegs; v[3] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<2, true>);
  v[4] = a.numRegs; v[5] = static_cast<int64_t>(a.localSizeBytes);
  cudaFuncGetAttributes(&a, r14::gemm_kernel<2, false>);
  v[6] = a.numRegs; v[7] = static_cast<int64_t>(a.localSizeBytes);
  return torch::tensor({v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7]});
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
              int64_t max_pieces, int64_t depth, int64_t smax, bool gated, bool pdl);
void r14_route(torch::Tensor ids, torch::Tensor pos, torch::Tensor pieces, torch::Tensor n_pieces, int64_t max_pieces,
               int64_t mode);
torch::Tensor r14_attrs();
"""


def r14_source():
    return _R14_CUDA.replace("@@WGMMA@@", "".join(_r14_wgmma_fn(n) for n in (16, 32, 48, 64, 80, 96, 112, 128)))


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


def _experts_r14(x, topk_weights, topk_ids, state, cfg):
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
    depth, smax, mode = cfg
    ext.r14_route(local_ids.view(-1), pos, pieces, n_pieces, max_pieces, mode)
    a_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    _r14_gather_kernel[(slots,)](x.contiguous(), pos, a_sw, rows, H=D_EXPERT, TOPK=TOP_K, num_warps=4)
    act_sw = torch.empty((D_EXPERT // 64, rows, 64), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(r14["w13"], a_sw, pieces, n_pieces, r14["c1"], act_sw, r14["err"], rows, D_EXPERT // 64,
                 max_pieces, depth, smax, True, False)
    y2 = torch.empty((rows, D_EXPERT), device=dev, dtype=torch.bfloat16)
    ext.r14_gemm(r14["w2"], act_sw, pieces, n_pieces, r14["c2"], y2, r14["err"], rows, D_EXPERT // 128,
                 max_pieces, depth, smax, False, True)  # v7b: GEMM2 with PDL after GEMM1
    out = torch.empty((tokens, D_EXPERT), device=dev, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, pos, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe_v4(x, topk_weights, topk_ids, state, tokens):
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
    for cls, cfg in R14_CONFIGS.items():
        try:
            err, row, flags = r14_check(state, cfg, R14_REP_TOKENS[cls])
            ok = flags == 0 and err < R14_MAX_ERR and row < 5 * R14_MAX_ERR
            print(f"[v4] {cls}: R14 vs Marlin err {err:.2e} row {row:.2e} flags {flags} -> {'R14' if ok else 'fallback'}",
                  flush=True)
            if ok:
                r14["cfg"][cls] = cfg
        except Exception as exc:
            print(f"[v4] {cls}: R14 failed, fallback: {type(exc).__name__}: {str(exc)[-300:]}", flush=True)
    torch.cuda.synchronize()


def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = _gemm(data["attn_input"], dense["qkv_gate"])
    if cfg["rope_theta"]:
        q, k = state["qk_norm"](qkv, data["positions"], state["cos_sin"])
    else:
        q, k = state["qk_norm"](qkv)
    v = qkv[:, hq * d + hkv * d: hq * d + 2 * hkv * d]
    gate = qkv[:, hq * d + 2 * hkv * d:]
    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
    # The pool keeps FP8 rows as bytes; slot 0 is a live row here (SGLang reserves it for padding).
    store_cache(k.to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8),
                data["cache_loc"], reserved_skip_index=-1)
    window = cfg["window"]
    descale = {}
    if scaled:
        descale = dict(k_descale=data["k_scale"].reshape(1, 1).expand(batch, hkv),
                       v_descale=data["v_scale"].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(
        q=q.view(tokens, hq, d).to(FP8),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=data["page_table"],
        cache_seqlens=data["cache_seqlens"],
        cu_seqlens_q=data["cu_seqlens_q"],
        max_seqlen_q=length,
        softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"],
        causal=True,
        window_size=(window - 1, 0) if window else (-1, -1),
        scheduler_metadata=data["scheduler_metadata"],
        **descale,
    )
    if isinstance(out, tuple):
        out = out[0]
    o = state["o_norm_gate"](out.reshape(tokens, hq * d).to(torch.bfloat16), gate)
    attn_output = _gemm(o, dense["o"])  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = _gemm(ffn_input, dense["pooler"])
    with torch.cuda.stream(side_stream):
        if split_router:
            logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(router_input, state["router"])
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = _moe_v4(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state, tokens)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids
