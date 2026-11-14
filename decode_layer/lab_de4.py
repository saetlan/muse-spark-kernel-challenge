# okc: problem=muse_decode_layer
"""v2 = D1 + M11 experts, an FP32 router kernel and tiny_gemm (see the v2 section at the end).

D1: SGLang's decoder layer (baseline.py, unchanged otherwise) with the two expert-path changes
the MoE session measured (moe/submission.py v1, 1.27x on muse_moe_w4a16):

* Marlin's tile height (moe_block_size) is the one measured fastest per token count: 8 below 32
  tokens, 16 below 192, else 32. SGLang sizes it from tokens * top_k / local_experts as if every
  route were local; under EP8 about 1/8 are, so from 128 tokens up its tiles are mostly padding.
* The post-expert norm and weighted sum always run as SGLang's fused Triton kernel. Below 32
  tokens SGLang falls back to about ten PyTorch ops on a [tokens, 8, 4096] FP32 tensor.

SGLang's own docstring follows.

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
import triton
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
    _post_expert_norm_weighted_sum_kernel,
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
# D1: Marlin tile height by token count, measured on muse_moe_w4a16 (E1 sweep, F18).
BLOCK_SIZE_M_STEPS = ((32, 8), (192, 16))
BLOCK_SIZE_M_LARGE = 32
ROPE_POSITIONS = 1 << 21
ROPE_THETA = 500_000.0


def block_size_m(num_tokens):
    for below, size in BLOCK_SIZE_M_STEPS:
        if num_tokens < below:
            return size
    return BLOCK_SIZE_M_LARGE


def post_expert_norm_weighted_sum(y, topk_weights, norm_w, slot_ids):
    """out[t] = sum over live slots k of topk_weights[t, k] * rmsnorm(y[t, k] * norm_w), with
    SGLang's fused kernel at every token count. slot_ids < 0 marks a slot whose expert isn't on
    this GPU."""
    y = y.contiguous()
    tokens, topk, hidden = y.shape
    out = torch.empty((tokens, hidden), dtype=y.dtype, device=y.device)
    block_h = triton.next_power_of_2(hidden)
    _post_expert_norm_weighted_sum_kernel[(tokens,)](
        y, norm_w, topk_weights.contiguous(), out, slot_ids,
        TOPK=topk, H=hidden, eps=POST_NORM_EPS, APPLY_NORM=True, HAS_WEIGHT=True,
        WEIGHT_FIRST=True, HAS_IDS=True, BLOCK_H=block_h,
        num_warps=8 if block_h >= 4096 else 4, num_stages=2,
    )
    return out


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


def _d1_prepare(weights):
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
    del freqs
    # Compile the post-norm kernel now, so its first launch never lands inside a graph capture.
    post_expert_norm_weighted_sum(
        torch.zeros((1, TOP_K, D_EXPERT), dtype=torch.bfloat16, device=device),
        torch.zeros((1, TOP_K), dtype=torch.float32, device=device), state["post_expert_w"],
        torch.full((1, TOP_K), -1, dtype=torch.int32, device=device))
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
    moe_block = block_size_m(num_tokens)
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, moe_block, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=True,
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
    return post_expert_norm_weighted_sum(y, topk_weights, state["post_expert_w"], local_ids)


def _d1_kernel_fn(data, state):
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
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# ====================================================================== parts/m11.py

# Snapshot of the MoE session's M11 (moe/m11/m11.py, sha256 a496ff5e9fe5a59e..., kernels identical to what its
# E3 check d0ce41e8697245ce ran on the judge: 1.309x on muse_moe_w4a16), without its prepare and
# kernel_fn. Copied, not imported, so later edits there cannot change this file.

"""M11: CUDA expert GEMMs for small batches; v1's Marlin path keeps the larger ones.

Swap-AB mma.sync.m16n8k16: 16 weight rows are the A operand, up to 8 * NT of an expert's pairs
are N. prepare() reorders each expert's NVFP4 codes (lossless) so that one 32-bit word is exactly
one thread's 8 A elements for one MMA, decoded straight into A registers with Marlin's bit trick,
and permutes K inside every 64-wide chunk so that a thread's 4 MMAs read 16 consecutive
activations, which are one NVFP4 scale group: 2 scales per thread per chunk. Every CTA scans the
pairs itself (no sort/align kernel) and its warps work through (active expert, row tile) tasks.
GEMM1 computes gate and up tiles together and applies SwiGLU in its epilogue."""

import tempfile

import triton.language as tl
from torch.utils.cpp_extension import load_inline

M11_MAX_TOKENS = 32
# Block scales are raised by 2^64 before the bf16 multiply (as in M2, exact); epilogues undo 2^(126 - 64).
M11_OUT_SHIFT = 2.0 ** 62
# The (gated, nt, depth) instances compiled; nt = MMA column tiles of 8 pairs, depth = K chunks in flight.
M11_INSTANCES = ((1, 8), (1, 4), (2, 4), (4, 2))
# Per-call configs by token count: (fewer than this many tokens, cfg) with cfg keys nt (both GEMMs),
# d1/d2 (depth) and c1/c2 (CTAs per SM) per GEMM; None means v1. A token count must be at most 8 * nt.
# Chosen by E3 (job d0ce41e8697245ce) from cold-L2 graph timings on the judge's H200; 17-32 tokens
# stay on v1, where nt 4 lost to Marlin.
M11_CONFIGS = (
    (5, dict(nt=1, d1=8, c1=2, d2=8, c2=2)),
    (9, dict(nt=1, d1=4, c1=4, d2=8, c2=4)),
    (17, dict(nt=2, d1=4, c1=4, d2=4, c2=4)),
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
constexpr int MAXP = 32;    // pairs per expert at most: M11 runs only up to 32 tokens

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

// out[pair, :] for every local pair: GATED: silu(x @ gate^T) * (x @ up^T), rows of w are gate then up;
// otherwise act @ w^T. w: [LOCAL][rt_total][KG][32 lanes] uint4, sf: [LOCAL][rt_total][KG][32] uint16.
template <int TT, int NT, int D, bool GATED>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols) {
  __shared__ int s_cnt[LOCAL];
  __shared__ int s_active[LOCAL];
  __shared__ int s_nactive;
  __shared__ short s_list[LOCAL][MAXP];
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
  const int per_expert = GATED ? rt_total / 2 : rt_total / TT;
  const int n_tasks = s_nactive * per_expert;
  const int warps = gridDim.x * (blockDim.x >> 5);
  for (int task = blockIdx.x * (blockDim.x >> 5) + (tid >> 5); task < n_tasks; task += warps) {
    const int e = s_active[task / per_expert];
    const int tile = task % per_expert;
    const int cnt = min(min(s_cnt[e], MAXP), 8 * NT);
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
    // B operand: this thread loads pair column g of each 8-pair tile, 16 activations per K chunk.
    const uint4* ap[NT];
    bool aok[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int idx = n * 8 + g;
      aok[n] = idx < cnt;
      const int p = aok[n] ? s_list[e][idx] : 0;
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
#pragma unroll
    for (int s = 0; s < D; ++s) load(s, s);
    for (int kg0 = 0; kg0 < KG; kg0 += D) {
#pragma unroll
      for (int s = 0; s < D; ++s) {
        compute(s);
        if (kg0 + s + D < KG) load(s, kg0 + s + D);
      }
    }

    // C fragment: rows g and g + 8 of each tile, pair columns 2t and 2t + 1.
    const float c = cs[e];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int idx = n * 8 + 2 * t + h;
        if (idx < cnt) {
          const size_t prow = static_cast<size_t>(s_list[e][idx]) * out_cols;
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

}  // namespace m11

#define M11_CASE(G, TT, NT, D)                                                                           \
  if (gated == (G) && nt == (NT) && depth == (D)) {                                                      \
    m11::gemm_kernel<TT, NT, D, G><<<static_cast<int>(grid), 128, 0, stream>>>(                          \
        reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()), \
        c.data_ptr<float>(), ids.data_ptr<int>(), reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_pairs),                    \
        static_cast<int>(local_start), static_cast<int>(rt_total), static_cast<int>(a_per_pair),         \
        static_cast<int>(out_cols));                                                                     \
    return out;                                                                                          \
  }

#define M11_ATTR(G, TT, NT, D)                                                                            \
  if (gated == (G) && nt == (NT) && depth == (D)) {                                                       \
    cudaFuncAttributes attr;                                                                              \
    cudaFuncGetAttributes(&attr, m11::gemm_kernel<TT, NT, D, G>);                                         \
    return torch::tensor({static_cast<int64_t>(attr.numRegs), static_cast<int64_t>(attr.localSizeBytes), \
                          static_cast<int64_t>(attr.sharedSizeBytes)});                                   \
  }

torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t nt, int64_t depth,
                       int64_t grid) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  M11_CASE(true, 2, 1, 8) M11_CASE(true, 2, 1, 4) M11_CASE(true, 2, 2, 4) M11_CASE(true, 2, 4, 2)
  M11_CASE(false, 1, 1, 8) M11_CASE(false, 1, 1, 4) M11_CASE(false, 1, 2, 4) M11_CASE(false, 1, 4, 2)
  TORCH_CHECK(false, "m11_gemm: no instance for gated=", gated, " nt=", nt, " depth=", depth);
  return out;
}

torch::Tensor m11_attrs(int64_t gated, int64_t nt, int64_t depth) {
  M11_ATTR(true, 2, 1, 8) M11_ATTR(true, 2, 1, 4) M11_ATTR(true, 2, 2, 4) M11_ATTR(true, 2, 4, 2)
  M11_ATTR(false, 1, 1, 8) M11_ATTR(false, 1, 1, 4) M11_ATTR(false, 1, 2, 4) M11_ATTR(false, 1, 4, 2)
  TORCH_CHECK(false, "m11_attrs: no instance for gated=", gated, " nt=", nt, " depth=", depth);
  return torch::Tensor();
}
"""

_M11_CPP = """
torch::Tensor m11_gemm(torch::Tensor w, torch::Tensor sf, torch::Tensor c, torch::Tensor ids, torch::Tensor a,
                       torch::Tensor out, int64_t n_pairs, int64_t local_start, int64_t rt_total,
                       int64_t a_per_pair, int64_t out_cols, int64_t gated, int64_t nt, int64_t depth,
                       int64_t grid);
torch::Tensor m11_attrs(int64_t gated, int64_t nt, int64_t depth);
"""


def m11_build():
    """Compile the CUDA kernels (sm_90a only; an explicit -gencode stops PyTorch adding plain sm_90)."""
    return load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=_M11_CUDA, functions=["m11_gemm", "m11_attrs"],
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


def m11_gemm1(x, ids, state, cfg):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair."""
    bank, n_pairs = state["m11"], ids.numel()
    act = torch.empty((n_pairs, bank["inter"]), dtype=torch.bfloat16, device=x.device)
    state["m11_ext"].m11_gemm(bank["w13"], bank["w13_s"], bank["c13"], ids, x, act, n_pairs, bank["start"],
                              bank["rt_w13"], 0, bank["inter"], 1, cfg["nt"], cfg["d1"], cfg["c1"] * state["sms"])
    return act


def m11_gemm2(act, ids, state, cfg):
    """act [pairs, I] -> y bf16 [pairs, H]."""
    bank, n_pairs = state["m11"], ids.numel()
    y = torch.empty((n_pairs, bank["hidden"]), dtype=torch.bfloat16, device=act.device)
    state["m11_ext"].m11_gemm(bank["w2"], bank["w2_s"], bank["c2"], ids, act, y, n_pairs, bank["start"],
                              bank["rt_w2"], 1, bank["hidden"], 0, cfg["nt"], cfg["d2"], cfg["c2"] * state["sms"])
    return y


def m11_post_norm(y, ids, topk_weights, state, tokens):
    bank = state["m11"]
    out = torch.empty((tokens, bank["hidden"]), dtype=torch.bfloat16, device=y.device)
    _m11_post_norm_kernel[(tokens,)](y, state["norm_w"], topk_weights, ids, out, bank["start"], POST_NORM_EPS,
                                     H=bank["hidden"], LOCAL=bank["count"], TOP_K=TOP_K, num_warps=8)
    return out


def m11_run(x, topk_ids, topk_weights, state, cfg):
    ids = topk_ids.reshape(-1).contiguous()
    act = m11_gemm1(x, ids, state, cfg)
    y = m11_gemm2(act, ids, state, cfg)
    return m11_post_norm(y, ids, topk_weights.contiguous(), state, x.shape[0])


def m11_config_for(state, tokens):
    """The M11 config for this token count, or None for v1."""
    for below, cfg in state["m11_configs"]:
        if tokens < below:
            return cfg if cfg is not None and tokens <= 8 * cfg["nt"] else None
    return None


def m11_warm(state, cfg):
    """Run every kernel of cfg once before any CUDA graph capture (Triton compiles on first launch)."""
    device = state["norm_w"].device
    tokens = min(8 * cfg["nt"], M11_MAX_TOKENS)
    x = torch.zeros((tokens, state["m11"]["hidden"]), dtype=torch.bfloat16, device=device)
    ids = (torch.arange(tokens * TOP_K, dtype=torch.int32, device=device) % state["m11"]["count"]).view(tokens, TOP_K)
    tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
    m11_run(x, ids + state["m11"]["start"], tw, state, cfg)
    torch.cuda.synchronize()


def m11_setup(state, weights):
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = m11_build()
    state["m11"] = m11_bank(weights, 0, 32)


# ====================================================================== parts/v2.py

# v2: D1 plus three changes, each checked against the path it replaces in prepare() and dropped
# (with a printed line) if that check fails:
#
# * Experts at <= 16 tokens run the MoE session's M11 CUDA kernels (fragment-ordered NVFP4 weights,
#   swap-AB mma.sync, SwiGLU in GEMM1's epilogue, no sort/align kernel), with E3's configs. 17+ tokens
#   keep D1's Marlin path. M11 gets this GPU's local ids (0-31 local, 32+ not), so its start is 0.
# * The FP32 router at <= 8 tokens is one CUDA-core kernel reading the 8.4 MB weight once, in FP32
#   (no TF32, no BF16 split): SGLang's F.linear takes 20.8 us at 8 tokens (DE0, DF6).
# * The unpooler slice (1024 x 4096) at <= 8 tokens runs SGLang's tiny_gemm_bf16 (2.7 vs 5.9 us at 1
#   token, DE0).

@triton.jit
def _v2_router_fp32_kernel(x_ptr, w_ptr, out_ptr, T, N, K: tl.constexpr):
    """out[t, n] = sum_k x[t, k] * w[n, k] in FP32; one program per router row n."""
    n = tl.program_id(0)
    offs = tl.arange(0, K)
    w = tl.load(w_ptr + n * K + offs)
    for t in range(T):
        x = tl.load(x_ptr + t * K + offs)
        tl.store(out_ptr + t * N + n, tl.sum(w * x, axis=0))


def v2_router_logits(router_input, state):
    """FP32 router logits [T, 256] for T <= 8 tokens (router_input: FP32 [T, 8192])."""
    if not state.get("v2_router"):
        return F.linear(router_input, state["router"])
    x = router_input.contiguous()
    w = state["router"]
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device=x.device)
    _v2_router_fp32_kernel[(w.shape[0],)](x, w, out, x.shape[0], w.shape[0], K=w.shape[1], num_warps=8)
    return out


def v2_unpool_slice(expert_output, state):
    """This GPU's 1024 unpooler columns for T <= 8 tokens."""
    if state.get("v2_tiny") is None:
        return _gemm(expert_output, state["dense"]["unpooler"][:UNPOOL_SLICE])
    return state["v2_tiny"](expert_output.contiguous(), state["v2_unpool_slice"])


def v2_experts(x, topk_weights, local_ids, state):
    """The routed experts: M11 where configured for this token count, else D1's Marlin path."""
    cfg = m11_config_for(state, x.shape[0]) if "m11" in state else None
    if cfg is None:
        return _experts(x, topk_weights, local_ids, state)
    return m11_run(x.contiguous(), local_ids.to(torch.int32), topk_weights.to(torch.float32), state, cfg)


def _v2_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def _v2_setup_router(state, device):
    try:
        state["v2_router"] = True
        x = torch.randn((8, HIDDEN), device=device)
        errs = [_v2_rel(v2_router_logits(x[:t], state), F.linear(x[:t], state["router"])) for t in (1, 3, 8)]
        state["v2_router"] = max(errs) < 1e-5
        print(f"[v2] fp32 router kernel: rel err vs F.linear {max(errs):.1e} -> {'on' if state['v2_router'] else 'OFF'}")
    except Exception as exc:
        state["v2_router"] = False
        print(f"[v2] fp32 router kernel OFF: {type(exc).__name__}: {str(exc)[-300:]}")


def _v2_setup_tiny(state, device):
    state["v2_tiny"] = None
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
        w = state["dense"]["unpooler"][:UNPOOL_SLICE].contiguous()
        if not can_use_tiny_gemm(w.shape[0], w.shape[1], 16):
            print("[v2] tiny_gemm OFF: can_use_tiny_gemm is False for the unpooler slice")
            return
        state["v2_unpool_slice"] = w
        x = torch.randn((8, D_EXPERT), device=device).to(torch.bfloat16)
        errs = [_v2_rel(tiny_gemm_bf16(x[:t].contiguous(), w), F.linear(x[:t], w)) for t in (1, 3, 8)]
        if max(errs) < 1e-2:
            state["v2_tiny"] = tiny_gemm_bf16
        print(f"[v2] tiny_gemm unpooler slice: rel err vs cuBLAS {max(errs):.1e} -> {'on' if state['v2_tiny'] else 'OFF'}")
    except Exception as exc:
        print(f"[v2] tiny_gemm OFF: {type(exc).__name__}: {str(exc)[-300:]}")


def _v2_setup_m11(state, weights, device):
    try:
        state["norm_w"] = state["post_expert_w"]
        state["m11_configs"] = M11_CONFIGS
        m11_setup(state, weights)
        for _, cfg in M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
        gen = torch.Generator(device=device).manual_seed(7)
        worst = 0.0
        for tokens in (1, 8, 16):
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            ids = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            ids[:, :LOCAL_EXPERTS] += 0.5  # most routes local, like the busiest GPU
            ids = ids.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            ref = _experts(x, tw, ids, state)
            got = v2_experts(x, tw, ids, state)
            worst = max(worst, _v2_rel(got, ref))
        torch.cuda.synchronize()
        if not worst < 2e-2:
            raise RuntimeError(f"M11 vs Marlin rel err {worst:.2e}")
        print(f"[v2] M11 on at < 17 tokens: rel err vs D1's Marlin {worst:.1e}")
    except Exception as exc:
        for key in ("m11", "m11_ext"):
            state.pop(key, None)
        print(f"[v2] M11 OFF: {type(exc).__name__}: {str(exc)[-400:]}")


def _submission_prepare(weights):
    state = _d1_prepare(weights)
    device = weights["qkv_gate"].device
    _v2_setup_router(state, device)
    _v2_setup_tiny(state, device)
    _v2_setup_m11(state, weights, device)
    torch.cuda.synchronize()
    return state


# ====================================================================== v2 kernel_fn: D1's with the router, experts and unpooler-slice calls swapped

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
            logits = v2_router_logits(router_input, state)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = (topk_ids - data["local_start"]) % N_EXPERTS
    # all-reduce: identity
    expert_output = v2_experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = v2_unpool_slice(expert_output, state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


# ====================================================================== lab (common): check runs only

"""Lab helpers shared by decode_layer check-run labs (appended after the submission by
decode_layer/local/make_lab.py). Everything here runs inside prepare() on synthetic inputs; none
of it touches the timed calls."""

import math
import statistics
import time

LAB_FLUSH_BYTES = 512 << 20  # larger than the H200's 60 MB L2, like the judge's cold-L2 read


def log(msg):
    print(f"[lab] {msg}", flush=True)


def _error_text(exc, keep=400):
    """The end of an exception's message, where Triton puts the actual reason."""
    text = " ".join(str(exc).split())
    return text if len(text) <= keep else "... " + text[-keep:]


class Lab:
    def __init__(self, budget_s):
        self.t0 = time.monotonic()
        self.budget_s = budget_s
        self.flush = None

    def elapsed(self):
        return time.monotonic() - self.t0

    def left(self):
        return self.budget_s - self.elapsed()

    def section(self, name, fn, *args, until=None, min_left=5.0):
        """Run fn(lab, *args, deadline) where deadline is lab-elapsed seconds; failures only print."""
        if self.left() < min_left:
            log(f"{name}: skipped, {self.left():.0f}s of budget left")
            return
        deadline = self.budget_s if until is None else min(until, self.budget_s)
        t = time.monotonic()
        try:
            fn(self, *args, deadline)
        except Exception as exc:  # a lab failure must never fail the run
            log(f"{name}: FAILED {type(exc).__name__}: {_error_text(exc)}")
        torch.cuda.synchronize()
        log(f"{name}: {time.monotonic() - t:.1f}s")

    def cold_l2(self):
        if self.flush is None:
            self.flush = torch.empty(LAB_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        return self.flush.sum()

    def time_graph(self, fn, reps=15, cold=True):
        """Median microseconds of a CUDA graph replay of fn, each after an L2 flush if cold.
        Includes the replay's fixed cost (~4 us, which the judge subtracts)."""
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

    def empty_graph_us(self):
        """A replay of a graph holding one trivial kernel: the fixed cost to subtract."""
        x = torch.zeros(1, device="cuda")
        return self.time_graph(lambda: x.add_(1), cold=True)


# The 12 scored shapes: (name, batch, q_len, layer, weight). All ctx 80000, page 1, dist soft,
# spread agentic above batch 1.
SCORED = (
    ("dec1 sw", 1, 1, "sliding", 46), ("dec1 full", 1, 1, "full", 16),
    ("dec16 sw", 16, 1, "sliding", 46), ("dec16 full", 16, 1, "full", 16),
    ("dec32 sw", 32, 1, "sliding", 46), ("dec32 full", 32, 1, "full", 16),
    ("ver1 sw", 1, 8, "sliding", 46), ("ver1 full", 1, 8, "full", 16),
    ("ver16 sw", 16, 8, "sliding", 46), ("ver16 full", 16, 8, "full", 16),
    ("ver32 sw", 32, 8, "sliding", 46), ("ver32 full", 32, 8, "full", 16),
)
LAB_CTX = 80000
LAB_SPREAD = 0.075  # agentic: lengths uniform in ctx * (1 +/- 0.075)
LAB_STREAM_SEED = 5354  # the judge's stream basis (a fixed mean direction the router aligns with)
LAB_STREAM_MEAN, LAB_STREAM_SHARED, LAB_STREAM_RANK, LAB_DRAFT_SHARE = 0.7, 0.9, 4, 0.6
LAB_CACHE_CHUNK = 1 << 20


def lab_stream_basis(device):
    gen = torch.Generator().manual_seed(LAB_STREAM_SEED)
    mean = torch.randn((HIDDEN,), generator=gen)
    shared = torch.randn((LAB_STREAM_RANK, HIDDEN), generator=gen)
    return mean.to(device), shared.to(device)


def lab_input(batch, q_len, layer, seed, device="cuda", ctx=LAB_CTX, local_start=0):
    """Inputs shaped like the judge's for one scored case (page 1, kv scale 1). The cache holds
    N(0, 1) keys and values in FP8: their values don't change timing."""
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = batch * q_len
    if batch == 1:
        seq_lens = torch.full((1,), ctx, dtype=torch.int64, device=device)
        longest = ctx
    else:
        u = torch.rand((batch,), generator=gen, device=device)
        longest = math.ceil(ctx * (1 + LAB_SPREAD))
        seq_lens = (ctx * (1 + LAB_SPREAD * (2 * u - 1))).long().clamp(1, longest)
    max_pages = longest + q_len
    total = batch * max_pages
    perm = torch.randperm(total, generator=gen, device=device).to(torch.int32)
    pages_per = seq_lens + q_len
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    offsets = torch.cumsum(pages_per, 0) - pages_per
    for b, (n, o) in enumerate(zip(pages_per.tolist(), offsets.tolist())):
        page_table[b, :n] = perm[o:o + n]
    positions = (seq_lens[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    cache_loc = torch.gather(page_table.long(), 1, positions.view(batch, q_len)).reshape(-1)
    k_cache = torch.empty((total, 1, 2, 64), dtype=FP8, device=device)
    v_cache = torch.empty((total, 1, 2, 64), dtype=FP8, device=device)
    for dst in (k_cache, v_cache):
        flat = dst.view(-1, 128)
        for at in range(0, total, LAB_CACHE_CHUNK):
            n = min(LAB_CACHE_CHUNK, total - at)
            flat[at:at + n] = torch.randn((n, 128), generator=gen, device=device).to(FP8)
    mean, shared = lab_stream_basis(device)
    coef = torch.randn((tokens, LAB_STREAM_RANK), generator=gen, device=device)
    scale = torch.ones((batch, q_len, 1), device=device)
    if 1 < q_len <= 8:
        scale[:, 1:] = LAB_DRAFT_SHARE
    scale = scale.reshape(tokens, 1)
    stream = (scale * (LAB_STREAM_MEAN * mean + math.sqrt(LAB_STREAM_SHARED / LAB_STREAM_RANK) * coef @ shared)
              + (1 - LAB_STREAM_SHARED * scale * scale).sqrt()
              * torch.randn((tokens, HIDDEN), generator=gen, device=device))
    attn_input = (stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(torch.bfloat16)
    cache_seqlens = (seq_lens + q_len).to(torch.int32)
    cu_seqlens_q = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    sched = None
    if layer == "full" and q_len == 1:
        from sgl_kernel.flash_attn import get_scheduler_metadata
        sched = get_scheduler_metadata(
            batch_size=batch, max_seqlen_q=1, max_seqlen_k=int(cache_seqlens.max()), num_heads=16,
            num_heads_k=2, headdim=64, cache_seqlens=cache_seqlens, qkv_dtype=FP8,
            cu_seqlens_q=cu_seqlens_q, page_size=1, causal=True, num_splits=0)
    sliding = layer == "sliding"
    return {
        "stream": stream, "attn_input": attn_input, "k_cache": k_cache, "v_cache": v_cache,
        "positions": positions, "cache_loc": cache_loc, "cache_seqlens": cache_seqlens,
        "cu_seqlens_q": cu_seqlens_q, "scheduler_metadata": sched,
        "k_scale": torch.tensor(1.0, device=device), "v_scale": torch.tensor(1.0, device=device),
        "page_table": page_table, "seq_lens": seq_lens.to(torch.int32),
        "local_start": torch.tensor([local_start], dtype=torch.int32, device=device),
        "config": {
            "batch": batch, "q_len": q_len, "tokens": tokens, "hidden": HIDDEN, "num_q_heads": 16,
            "num_kv_heads": 2, "head_dim": 64, "page_size": 1, "window": 2048 if sliding else 0,
            "rope_theta": ROPE_THETA if sliding else 0.0, "qk_scale_factor": 5.4730064863875,
            "sm_scale": 0.125, "kv_scale": 1.0, "n_experts": N_EXPERTS, "top_k": TOP_K,
            "local_experts": LOCAL_EXPERTS,
        },
    }


def kv_bytes(data):
    """KV bytes attention must read: each request's attended tokens x 256 B (K and V, 2 heads)."""
    cfg = data["config"]
    lens = data["cache_seqlens"].long()
    if cfg["window"]:
        lens = lens.clamp(max=cfg["window"] + cfg["q_len"] - 1)
    return int(lens.sum()) * 256


# ====================================================================== lab (x5): check runs only

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


# ====================================================================== lab (de4): check runs only

"""DE4 lab (check runs only), on top of submission_v2 + parts/x5.py (X5 v3: k-step pairs per warp, 8-16 consumer
warps, cross-warp smem reduction, last-arriver BF16 output with an in-kernel workspace reset). X5 (our bulk-copy mma.sync skinny
GEMM) and X10 (12-bit lossless BF16 weights) on the five dense weights, against cuBLAS. Log kept short;
summary last.

build   nvcc for the 12 X5 instances; FlashInfer frost probe (X13).
pack    plain + 12-bit packs of qkv, o, pooler, unpooler slice, unpooler; raw-tile share and bytes.
correct each pack x plain/enc at 1/8/16/32 tokens vs FP32 x @ W^T (rel err), before any timing.
time    per weight, 1/8/16 tokens: cuBLAS (bf16 out) vs X5 plain and 12-bit (1-8 tokens: nc 4/8 x 1/2 CTAs per
        SM; 16 tokens: nc 8, 1 CTA per SM), kernel only, plus loader-only (no math) at 1 token: the
        bulk-copy ceiling.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def sec_build(lab, store, deadline):
    t = time.monotonic()
    store["ext"] = x5_build()
    store["sms"] = torch.cuda.get_device_properties(0).multi_processor_count
    log(f"x5 build {time.monotonic() - t:.1f}s; {store['sms']} SMs")
    try:
        import importlib
        m = importlib.import_module("flashinfer.fused_moe.backends.frost_low_latency")
        log(f"frost: present ({', '.join(sorted(n for n in dir(m) if not n.startswith('_'))[:12])})")
    except Exception as exc:
        log(f"frost: absent ({type(exc).__name__}: {_error_text(exc, 100)})")


def sec_pack(lab, state, store, deadline):
    dense = state["dense"]
    weights = {"qkv": dense["qkv_gate"], "o": dense["o"], "pool": dense["pooler"],
               "unps": dense["unpooler"][:UNPOOL_SLICE].contiguous(), "unp": dense["unpooler"]}
    store["packs"] = {}
    cells = []
    for name, w in weights.items():
        t = time.monotonic()
        pk = x5_pack(w)
        pk["w"] = w
        store["packs"][name] = pk
        ratio = x5_enc_bytes(pk) / (w.numel() * 2)
        cells.append(f"{name} raw {100 * pk['raw_tiles'] / pk['tiles']:.2f}% bytes {ratio:.3f} ({time.monotonic() - t:.1f}s)")
    torch.cuda.synchronize()
    log("x5 packs: " + "; ".join(cells))


def _run(store, pk, x, enc, nc, cps, mode=0):
    t = x.shape[0]
    return x5_gemv(x, pk, store["ext"], enc, nc, 1 if t <= 8 else 2, store["sms"] * cps, mode)


NC_CHOICES = {(0, 1): (8, 16), (0, 2): (8, 16), (1, 1): (8, 12, 16), (1, 2): (8, 16)}


def sec_correct(lab, store, deadline):
    bad = []
    worst = 0.0
    dirty = 0
    for name, pk in store["packs"].items():
        for t in (1, 8, 16):
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            ref = x.float() @ pk["w"].float().T
            nt = 1 if t <= 8 else 2
            for enc in (0, 1):
                for nc in NC_CHOICES[(enc, nt)]:
                    for rep in range(2):
                        got = _run(store, pk, x, enc, nc, 1 if rep == 0 else 2).float()
                        err = float((got - ref).norm() / ref.norm())
                        worst = max(worst, err)
                        if not err < 1e-2:
                            bad.append(f"{name} T{t} enc{enc} nc{nc} rep{rep} {err:.1e}")
                    torch.cuda.synchronize()
                    dirty += int(pk["ws"].abs().sum().item() != 0) + int(pk["cnt"].abs().sum().item() != 0)
    store["ok"] = not bad and dirty == 0
    log(f"x5 correct: worst rel err {worst:.1e} (BF16 output vs FP32); workspace/counters left dirty {dirty} times"
        + ("" if not bad else " | BAD: " + ", ".join(bad[:8])))
    LAB_SUMMARY.append(f"x5 correct {'yes' if store['ok'] else 'NO'} (worst {worst:.1e}, dirty {dirty})")


def sec_time(lab, store, deadline):
    if not store.get("ok"):
        LAB_SUMMARY.append("x5 timing skipped (correctness failed)")
        return
    fixed = lab.empty_graph_us()
    for name, pk in store["packs"].items():
        nbytes = pk["w"].numel() * 2
        ebytes = x5_enc_bytes(pk)
        w = pk["w"]
        cells = []
        for t in (1, 8, 16):
            if lab.elapsed() > deadline:
                LAB_SUMMARY.append(f"x5 timing stopped at {name} T{t}")
                break
            x = torch.randn((t, pk["k"]), device="cuda").to(torch.bfloat16)
            nt = 1 if t <= 8 else 2
            try:
                cub = lab.time_graph(lambda: F.linear(x, w)) - fixed
                res = {}
                for enc in (0, 1):
                    best = None
                    for nc in NC_CHOICES[(enc, nt)]:
                        for cps in (1, 2):
                            us = lab.time_graph(lambda: _run(store, pk, x, enc, nc, cps)) - fixed
                            if best is None or us < best[0]:
                                best = (us, nc, cps)
                    res[enc] = best
                p, e = res[0], res[1]
                cell = (f"T{t} cub {cub:.1f}({nbytes / max(cub, .01) / 1e6:.2f}) plain {p[0]:.1f}({nbytes / max(p[0], .01) / 1e6:.2f})"
                        f"n{p[1]}c{p[2]} enc {e[0]:.1f}({nbytes / max(e[0], .01) / 1e6:.2f}eff)n{e[1]}c{e[2]}")
                if t == 1:
                    lp = lab.time_graph(lambda: _run(store, pk, x, 0, 8, 2, 1)) - fixed
                    le = lab.time_graph(lambda: _run(store, pk, x, 1, 8, 2, 1)) - fixed
                    cell += f" load-only {lp:.1f}/{le:.1f}"
                cells.append(cell)
                LAB_SUMMARY.append(f"{name}T{t} cub {cub:.1f} plain {p[0]:.1f} enc {e[0]:.1f}")
            except Exception as exc:
                lab_fail(f"time {name} T{t}", exc)
        log(f"x5 {name} [{pk['n']}x{pk['k']}] " + " | ".join(cells))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v2 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s")
    store = {}
    lab.section("build", sec_build, store)
    if "ext" in store:
        lab.section("pack", sec_pack, state, store)
        if "packs" in store:
            lab.section("correct", sec_correct, store)
            lab.section("time", sec_time, store)
    store.clear()
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
