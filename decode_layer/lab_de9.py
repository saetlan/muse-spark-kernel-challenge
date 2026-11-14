# okc: problem=muse_decode_layer
"""v6 candidate = v5 with the attention session's v5 and the bf16_proj session's v4 (parts/v6.py).
v5 = v4 with parts/m11_pf.py (M11 + L8: next-task L2 prefetch, pf per config).

v4 = v3 + the attention session's kernels and the bf16_proj session's projections, per case
(see the v4 section at the end).

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


# ====================================================================== parts/m11_pf.py

# parts/m11_pf.py: generated by local/make_m11pf.py from parts/m11_e3c.py; M11 + L8 (next-task L2
# prefetch, runtime pf = ahead | mode << 4; pf = 0 is M11 unchanged). Do not edit by hand.
# Snapshot of the MoE session's scored M11 with E3c's configs (moe/scored/m11_e3cconfigs.py, sha256 d7ff3d7b414b4106...;
# E3c check 8ce4aba7fd53499c: 1.446x on muse_moe_w4a16), without its prepare and kernel_fn. Copied, not
# imported. v3 builds it without the M12 instances (v3_m11_build), so only the M11 kernels it uses compile.

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
// L8: L2 prefetch of one warp's K slice of a tile, kgw 64-wide chunks: 512 code bytes and 64 scale bytes
// per chunk, both contiguous and 16-B aligned. mode 0: two bulk prefetches by lane 0
// (cp.async.bulk.prefetch.L2.global, PTX ISA 9.4 9.7.10.28.4.3); 1: each lane prefetches 128-B lines
// (prefetch.global.L2, 9.7.10.16); 2: the same with .L2::evict_last. Hints only: no data changes.
__device__ __forceinline__ void m11_prefetch(const uint4* wp, const uint16_t* sp, int kgw, int mode, int lane) {
  const int wbytes = kgw * 512, sbytes = kgw * 64;
  if (mode == 0) {
    if (lane == 0) {
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(wp), "r"(wbytes));
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(sp), "r"(sbytes));
    }
    return;
  }
  const char* wc = reinterpret_cast<const char*>(wp);
  const char* sc = reinterpret_cast<const char*>(sp);
  for (int off = lane * 128; off < wbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(wc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(wc + off));
  }
  for (int off = lane * 128; off < sbytes; off += 32 * 128) {
    if (mode == 1) asm volatile("prefetch.global.L2 [%0];" ::"l"(sc + off));
    else asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(sc + off));
  }
}

template <int TT, int NT, int D, int KS, bool GATED>
__global__ void __launch_bounds__(128) gemm_kernel(
    const uint4* __restrict__ w, const uint16_t* __restrict__ sf, const float* __restrict__ cs,
    const int* __restrict__ ids, const __nv_bfloat16* __restrict__ a, __nv_bfloat16* __restrict__ out,
    int n_pairs, int local_start, int rt_total, int a_per_pair, int out_cols, int pf) {
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
    if (pf & 15) {  // L8: this warp's K slice of the task `ahead` strides on (on the first task: 1..ahead)
      const int ahead = pf & 15, stride = gridDim.x * GROUPS;
      for (int k = task < stride ? 1 : ahead; k <= ahead; ++k) {
        const int nxt = task + k * stride;
        if (nxt >= n_tasks) break;
        const int e2 = s_active[nxt / per_expert], tile2 = nxt % per_expert;
#pragma unroll
        for (int i = 0; i < TT; ++i) {
          const int rt2 = GATED ? tile2 + i * (rt_total / 2) : tile2 * TT + i;
          const size_t b2 = (static_cast<size_t>(e2) * rt_total + rt2) * KG * 32 + static_cast<size_t>(kpart) * KGW * 32;
          m11_prefetch(w + b2, sf + b2, KGW, pf >> 4, lane);
        }
      }
    }
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
        static_cast<int>(out_cols), static_cast<int>(pf));                                               \
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
                       int64_t ks, int64_t grid, int64_t pf) {
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
                       int64_t ks, int64_t grid, int64_t pf);
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
        name="m11pf_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(), functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],
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
        state["m11_ext"].m11_gemm(*args, s1["d"], s1["ks"], s1["c"] * state["sms"], s1.get("pf", 0))
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
        state["m11_ext"].m11_gemm(*args, s2["d"], s2["ks"], s2["c"] * state["sms"], s2.get("pf", 0))
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


# ====================================================================== parts/v3.py

# v3: D1 + E3c's M11 experts (configs re-picked on decode's routing by the lab) + X7 L2 prefetch during
# attention + an exact FP32 router that reduces once + fused local ids + a Triton pre-expert RMSNorm.
# Every new path is checked in prepare() against the one it replaces and turned off (printed) if it
# fails; v3_kernel_fn is D1's kernel_fn with a handful of call sites swapped (local/make_v3.py).

V3_ROUTER_NR = tl.constexpr(4)  # router rows per program
V3_ROUTER_BK = tl.constexpr(256)


@triton.jit
def _v3_router_kernel(x_ptr, w_ptr, out_ptr, T, N, K: tl.constexpr, TB: tl.constexpr):
    """out[t, n] = sum_k x[t, k] * w[n, k] in FP32 (no tl.dot, no TF32): each program takes 4 router rows,
    accumulates [4, TB, 256] products across K and reduces once at the end."""
    n0 = tl.program_id(0) * V3_ROUTER_NR
    rows = n0 + tl.arange(0, V3_ROUTER_NR)
    offs_t = tl.arange(0, TB)
    offs_k = tl.arange(0, V3_ROUTER_BK)
    acc = tl.zeros((V3_ROUTER_NR, TB, V3_ROUTER_BK), dtype=tl.float32)
    t_ok = offs_t[:, None] < T
    for k0 in range(0, K, V3_ROUTER_BK):
        w = tl.load(w_ptr + rows[:, None] * K + k0 + offs_k[None, :])
        x = tl.load(x_ptr + offs_t[:, None] * K + k0 + offs_k[None, :], mask=t_ok, other=0.0)
        acc += w[:, None, :] * x[None, :, :]
    res = tl.sum(acc, axis=2)  # [NR, TB]
    tl.store(out_ptr + offs_t[None, :] * N + rows[:, None], res, mask=offs_t[None, :] < T)


def v3_router_logits(router_input, state):
    if not state.get("v3_router"):
        return F.linear(router_input, state["router"])
    x = router_input.contiguous()
    t = x.shape[0]
    w = state["router"]
    out = torch.empty((t, w.shape[0]), dtype=torch.float32, device=x.device)
    tb = 1 if t <= 1 else 2 if t <= 2 else 4 if t <= 4 else 8
    _v3_router_kernel[(w.shape[0] // 4,)](x, w, out, t, w.shape[0], K=w.shape[1], TB=tb, num_warps=8)
    return out


@triton.jit
def _v3_local_ids_kernel(ids_ptr, start_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """out = (ids - local_start) mod 256 as int32, in one launch (ids int32, global expert ids 0-255)."""
    offs = tl.arange(0, BLOCK)
    ok = offs < n
    start = tl.load(start_ptr)
    v = tl.load(ids_ptr + offs, mask=ok, other=0) - start
    v = tl.where(v < 0, v + 256, v)
    tl.store(out_ptr + offs, v, mask=ok)


def v3_local_ids(topk_ids, local_start, state):
    if not state.get("v3_lid"):
        return (topk_ids - local_start) % N_EXPERTS
    ids = topk_ids.contiguous()
    out = torch.empty_like(ids, dtype=torch.int32)
    n = ids.numel()
    _v3_local_ids_kernel[(1,)](ids, local_start, out, n, BLOCK=triton.next_power_of_2(max(n, 16)), num_warps=4)
    return out


@triton.jit
def _v3_rmsnorm_kernel(x_ptr, w_ptr, out_ptr, eps, H: tl.constexpr):
    """out[t] = bf16(x[t] * rsqrt(mean(x[t]^2) + eps) * w) with FP32 math, as F.rms_norm on BF16 input."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    x = tl.load(x_ptr + t.to(tl.int64) * H + offs).to(tl.float32)
    w = tl.load(w_ptr + offs).to(tl.float32)
    var = tl.sum(x * x, axis=0) / H
    tl.store(out_ptr + t.to(tl.int64) * H + offs, (x * tl.rsqrt(var + eps) * w).to(tl.bfloat16))


def v3_pre_expert_norm(x, state):
    if not state.get("v3_norm"):
        return state["pre_expert_norm"](x)
    x = x.contiguous()
    out = torch.empty_like(x)
    _v3_rmsnorm_kernel[(x.shape[0],)](x, state["v3_pre_expert_w"], out, RMS_EPS, H=x.shape[1], num_warps=8)
    return out


def v3_experts(expert_input, topk_weights, local_ids, state):
    """Pre-expert norm, then M11 where configured for this token count, else D1's Marlin path."""
    x = v3_pre_expert_norm(expert_input, state)
    cfg = m11_config_for(state, x.shape[0]) if "m11" in state else None
    if cfg is None:
        return _experts(x, topk_weights, local_ids, state)
    return m11_run(x, local_ids.to(torch.int32), topk_weights.to(torch.float32), state, cfg)


def v3_unpool_slice(expert_output, state):
    """This GPU's 1024 unpooler columns for T <= 8 tokens (SGLang's tiny_gemm, as v2)."""
    if state.get("v3_tiny") is None:
        return _gemm(expert_output, state["dense"]["unpooler"][:UNPOOL_SLICE])
    return state["v3_tiny"](expert_output.contiguous(), state["v3_unpool_slice"])


# ---------------------------------------------------------------- X7: L2 prefetch during attention

_V3_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

// One thread per 64 KB chunk of two byte ranges queues a bulk L2 prefetch with an evict-last policy.
__global__ void v3_l2_prefetch_kernel(const uint8_t* p0, long long n0, const uint8_t* p1, long long n1) {
  constexpr long long CHUNK = 65536;
  long long off = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * CHUNK;
  const uint8_t* p = nullptr;
  long long left = 0;
  if (off < n0) {
    p = p0 + off;
    left = n0 - off;
  } else {
    off -= (n0 + CHUNK - 1) / CHUNK * CHUNK;
    if (off < 0 || off >= n1) return;
    p = p1 + off;
    left = n1 - off;
  }
  const uint32_t size = static_cast<uint32_t>((left < CHUNK ? left : CHUNK) & ~15LL);
  if (size == 0) return;
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(pol));
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;" ::"l"(p), "r"(size), "l"(pol) : "memory");
}

void v3_l2_prefetch(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb) {
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t chunks = (na + 65535) / 65536 + (nb + 65535) / 65536;
  if (chunks == 0) return;
  const int threads = 128;
  const int blocks = static_cast<int>((chunks + threads - 1) / threads);
  v3_l2_prefetch_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint8_t*>(a.data_ptr()), na,
                                                       reinterpret_cast<const uint8_t*>(b.data_ptr()), nb);
}
"""

_V3_CPP = "void v3_l2_prefetch(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb);"


def v3_source():
    return _V3_CUDA


def v3_build():
    return load_inline(
        name="v3_ext", cpp_sources=_V3_CPP, cuda_sources=_V3_CUDA, functions=["v3_l2_prefetch"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="v3_"), verbose=False)


def v3_prefetch_bytes(state, cfg):
    """Bytes of o + pooler to pull into L2 for this case (0 = off), keyed by (tokens, sliding)."""
    if "v3_ext" not in state:
        return 0
    return state["v3_prefetch"].get((cfg["tokens"], bool(cfg["window"])), 0)


def v3_prefetch_fork(state, cfg):
    """After the qkv GEMM: on a side stream, queue L2 prefetches of o and the head of the pooler, so
    they stream while attention (latency-bound on these shapes) leaves HBM idle."""
    nbytes = v3_prefetch_bytes(state, cfg)
    if nbytes <= 0:
        return False
    main, side = torch.cuda.current_stream(), state["v3_stream"]
    side.wait_stream(main)
    o, pool = state["dense"]["o"], state["dense"]["pooler"]
    no = min(nbytes, o.numel() * 2)
    npool = min(max(nbytes - no, 0), pool.numel() * 2)
    with torch.cuda.stream(side):
        state["v3_ext"].v3_l2_prefetch(o, no, pool, npool)
    return True


def v3_prefetch_join(state, forked):
    if forked:
        torch.cuda.current_stream().wait_stream(state["v3_stream"])


# ---------------------------------------------------------------- setup

def _v3_rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))


def v3_m11_setup(state, weights):
    """m11_setup without the M12 (TMA) instances, which v3 never runs: same M11 kernels, less nvcc."""
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    state["m11_ext"] = load_inline(
        name="m11_ext", cpp_sources=_M11_CPP, cuda_sources=m11_source(M11_INSTANCES, ()),
        functions=["m11_gemm", "m11_attrs", "m12_gemm", "m12_setup"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="m11_"), verbose=False)
    state["m11"] = m11_bank(weights, 0, 32)


def _v3_check(name, fn):
    try:
        ok, msg = fn()
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {str(exc)[-300:]}"
    print(f"[v3] {name}: {'on' if ok else 'OFF'} ({msg})")
    return ok


def _v3_prepare(weights):
    state = _d1_prepare(weights)
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(11)

    def router():
        state["v3_router"] = True
        x = torch.randn((8, HIDDEN), generator=gen, device=device)
        err = max(_v3_rel(v3_router_logits(x[:t], state), F.linear(x[:t], state["router"])) for t in (1, 2, 3, 5, 8))
        return err < 1e-5, f"rel err vs F.linear {err:.1e}"

    state["v3_router"] = _v3_check("fp32 router (one reduction)", router)

    def lid():
        state["v3_lid"] = True
        ids = torch.randint(0, 256, (16, TOP_K), generator=gen, device=device, dtype=torch.int32)
        ls = torch.tensor([96], dtype=torch.int32, device=device)
        same = all(bool((v3_local_ids(ids[:t], ls, state) == ((ids[:t] - ls) % N_EXPERTS)).all()) for t in (1, 8, 16))
        # compile every block size a case can need (BLOCK = next power of 2 of 8 x tokens, up to 320 tokens)
        for blk_tokens in (2, 4, 8, 16, 32, 64, 128, 256, 320):
            big = torch.randint(0, 256, (blk_tokens, TOP_K), generator=gen, device=device, dtype=torch.int32)
            same &= bool((v3_local_ids(big, ls, state) == ((big - ls) % N_EXPERTS)).all())
        return same, "bit-exact vs torch, block sizes 16-4096 compiled"

    state["v3_lid"] = _v3_check("local ids kernel", lid)

    def norm():
        state["v3_norm"] = True
        state["v3_pre_expert_w"] = state["pre_expert_norm"]._get_effective_weight().detach().to(torch.bfloat16).contiguous()
        x = (3 * torch.randn((16, D_EXPERT), generator=gen, device=device)).to(torch.bfloat16)
        err = max(_v3_rel(v3_pre_expert_norm(x[:t], state), state["pre_expert_norm"](x[:t])) for t in (1, 8, 16))
        return err < 1e-2, f"rel err vs MuseSpark1xRMSNorm {err:.1e}"

    state["v3_norm"] = _v3_check("pre-expert rmsnorm", norm)

    def tiny():
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
        w = state["dense"]["unpooler"][:UNPOOL_SLICE].contiguous()
        state["v3_tiny"] = None
        if not can_use_tiny_gemm(w.shape[0], w.shape[1], 16):
            return False, "can_use_tiny_gemm is False"
        state["v3_unpool_slice"] = w
        x = torch.randn((8, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
        err = max(_v3_rel(tiny_gemm_bf16(x[:t].contiguous(), w), F.linear(x[:t], w)) for t in (1, 3, 8))
        if err < 1e-2:
            state["v3_tiny"] = tiny_gemm_bf16
        return err < 1e-2, f"rel err vs cuBLAS {err:.1e}"

    _v3_check("tiny_gemm unpooler slice", tiny)

    def m11():
        state["norm_w"] = state["post_expert_w"]
        state["m11_configs"] = V3_M11_CONFIGS
        v3_m11_setup(state, weights)
        worst = 0.0
        for _, cfg in V3_M11_CONFIGS:
            if cfg is not None:
                m11_warm(state, cfg)
        for tokens in (1, 8, 16, 32):
            x = torch.randn((tokens, D_EXPERT), generator=gen, device=device).to(torch.bfloat16)
            sc = torch.rand((tokens, N_EXPERTS), generator=gen, device=device)
            sc[:, :LOCAL_EXPERTS] += 0.5
            ids = sc.topk(TOP_K, dim=-1).indices.to(torch.int32)
            tw = torch.rand((tokens, TOP_K), generator=gen, device=device)
            tw = tw / tw.sum(-1, keepdim=True)
            cfg = m11_config_for(state, tokens)
            if cfg is not None:
                worst = max(worst, _v3_rel(m11_run(x, ids, tw, state, cfg), _experts(x, tw, ids, state)))
        torch.cuda.synchronize()
        return worst < 2e-2, f"DE5 configs, rel err vs D1's Marlin {worst:.1e}"

    if not _v3_check("M11 (E3c)", m11):
        for key in ("m11", "m11_ext"):
            state.pop(key, None)

    def pref():
        state["v3_ext"] = v3_build()
        state["v3_stream"] = torch.cuda.Stream(device)
        state["v3_prefetch"] = dict(V3_PREFETCH)
        o = state["dense"]["o"]
        state["v3_ext"].v3_l2_prefetch(o, o.numel() * 2, state["dense"]["pooler"], 1 << 20)
        torch.cuda.synchronize()
        return True, f"{len(V3_PREFETCH)} cases enabled"

    if V3_BUILD_PREFETCH and not _v3_check("L2 prefetch", pref):
        state.pop("v3_ext", None)
    torch.cuda.synchronize()
    return state


# Per-case L2 prefetch bytes, keyed by (tokens, sliding). DE5 measured no gain on any shape (DF21): off.
V3_PREFETCH = {}
V3_BUILD_PREFETCH = True
# M11 configs chosen by the DE5 lab on decode's own routing (job 2887af7404324f6d, DF20); 33+ tokens: Marlin.
V3_M11_CONFIGS = ((5, {'s1': {'nt': 1, 'd': 4, 'ks': 4, 'c': 8}, 's2': {'tt': 1, 'nt': 1, 'd': 4, 'ks': 4, 'c': 4}}),
                  (9, {'s1': {'nt': 1, 'd': 4, 'ks': 4, 'c': 8}, 's2': {'tt': 2, 'nt': 1, 'd': 4, 'ks': 4, 'c': 8}}),
                  (17, {'s1': {'nt': 1, 'd': 4, 'ks': 2, 'c': 8}, 's2': {'tt': 2, 'nt': 1, 'd': 4, 'ks': 4, 'c': 8}}),
                  (33, {'s1': {'nt': 2, 'd': 2, 'ks': 4, 'c': 4}, 's2': {'tt': 2, 'nt': 2, 'd': 2, 'ks': 4, 'c': 8}}),
                  (321, None))


# ====================================================================== parts/attn5.py

# Embedded: the attention session's scored v5 (attention/scored/v5_submission.py, sha256 addc85831f9b2fc1...;
# scored 1.406x on muse_attention_decode, job 47db0b38ebaf49ec), every top-level name prefixed with 'att_' by
# decode_layer/local/make_attn.py. att_kernel_fn(data) computes our layer's attention segment (D3 on full decode 16/32, PDL chain).

"""v2.2 (hybrid): gated attention decode in three Triton kernels where they were measured faster,
SGLang's own path (FusedQKNorm, store_cache, FA3, rms_norm_gate) elsewhere (OURS below).

1. _prologue: per token, scaleless RMSNorm of q (times qk_scale_factor) and the new keys,
   interleaved RoPE on sliding layers, q kept in BF16 in a scratch buffer, the new keys and
   values cast to FP8 E4M3 (divided by the cache scales) and stored at their cache rows.
2. _attend: split-context flash decode. One program covers a pair of key/value heads of one
   request and one slice of its context. With page size 1 each token's cache row holds both
   heads (128 bytes), so the program reads whole rows: k0/k1 and v0/v1 come from the same
   lines. Each head gets its own dot (queries [rows, 64], rows = L x heads per KV head, padded
   to 16). FP8 keys and values are widened to BF16; scores and softmax in FP32, log2 domain.
3. _combine: one pass over the slices (running max, log-sum-exp), then RMSNorm x
   sigmoid(gate), BF16 out.

Queries stay BF16 (finer than FA3's FP8). Every key in the window (sliding) or the context
(full) is attended."""
import math
from typing import TypedDict
import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.kernels.ops.kvcache.kvcache import store_cache
from sglang.kernels.ops.muse_spark_v12.fused_qk_norm import FusedQKNorm
from sglang.kernels.ops.muse_spark_v12.norm_gate import rms_norm_gate

class att_Config(TypedDict):
    batch: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    window: int
    rope_theta: float
    qk_scale_factor: float
    sm_scale: float
    kv_scale: float

class att_Inputs(TypedDict):
    qkv: torch.Tensor
    positions: torch.Tensor
    cache_loc: torch.Tensor
    cache_seqlens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_scale: torch.Tensor
    v_scale: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    config: att_Config
att_RMS_EPS = 1e-05
att_LOG2E = 1.4426950408889634
att_NUM_SMS = 132
att_MAX_SPLITS = 512
att__INV_FREQ = {}
att__WARM = []
att_TUNE = {(True, False, 1): (32, 256, 64, 4, 3), (True, False, 16): (16, 256, 64, 4, 5), (True, False, 32): (16, 256, 64, 4, 3), (True, True, 1): (2, 256, 64, 4, 2), (True, True, 16): (4, 256, 64, 4, 2), (True, True, 32): (8, 256, 64, 4, 2), (False, False, 1): (16, 64, 64, 4, 2), (False, False, 16): (16, 128, 64, 4, 2), (False, False, 32): (16, 256, 64, 4, 2), (False, True, 1): (16, 64, 32, 4, 1), (False, True, 16): (16, 64, 32, 4, 1), (False, True, 32): (16, 256, 64, 4, 2)}

def att_batch_class(batch):
    return 1 if batch <= 1 else 16 if batch <= 16 else 32

@triton.jit
def att__prologue(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, QK_FACTOR: tl.constexpr, EPS: tl.constexpr):
    t = tl.program_id(0)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    base = qkv_ptr + t.to(tl.int64) * ROW
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    i = tl.arange(0, 32)
    q_e = tl.load(base + hq[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    q_o = tl.load(base + hq[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    k_e = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    k_o = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    v_e = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    v_o = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    q_r = tl.rsqrt((tl.sum(q_e * q_e, 1) + tl.sum(q_o * q_o, 1)) / 64.0 + EPS)
    k_r = tl.rsqrt((tl.sum(k_e * k_e, 1) + tl.sum(k_o * k_o, 1)) / 64.0 + EPS)
    q_e = q_e * q_r[:, None] * QK_FACTOR
    q_o = q_o * q_r[:, None] * QK_FACTOR
    k_e = k_e * k_r[:, None]
    k_o = k_o * k_r[:, None]
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(inv_freq_ptr + i)
        c = libdevice.cos(ang)[None, :]
        s = libdevice.sin(ang)[None, :]
        q_e, q_o = (q_e * c - q_o * s, q_e * s + q_o * c)
        k_e, k_o = (k_e * c - k_o * s, k_e * s + k_o * c)
    q_dst = q_out_ptr + t.to(tl.int64) * (HQ * 64) + hq[:, None] * 64 + 2 * i[None, :]
    tl.store(q_dst, q_e.to(tl.bfloat16))
    tl.store(q_dst + 1, q_o.to(tl.bfloat16))
    k_inv = 1.0 / tl.load(k_scale_ptr)
    v_inv = 1.0 / tl.load(v_scale_ptr)
    row = tl.load(loc_ptr + t)
    off = row * (HKV * 64) + hk[:, None] * 64 + 2 * i[None, :]
    tl.store(k_cache_ptr + off, (k_e * k_inv).to(tl.float8e4nv))
    tl.store(k_cache_ptr + off + 1, (k_o * k_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off, (v_e * v_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off + 1, (v_o * v_inv).to(tl.float8e4nv))

@triton.jit
def att__head_step(q, k, v, m_i, l_i, acc, vis, qk_scale):
    """One online-softmax step for one KV head: q [MP, 64], k and v [BLOCK, 64] BF16."""
    sc = tl.dot(q, tl.trans(k)) * qk_scale
    sc = tl.where(vis, sc, float('-inf'))
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(sc - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return (m_new, l_i, acc)

@triton.jit(do_not_specialize=['max_pages', 'nsplit'])
def att__attend(q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, page_table_ptr, seqlens_ptr, o_part_ptr, m_part_ptr, l_part_ptr, max_pages, nsplit, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, SM_SCALE: tl.constexpr, MP: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    kv0 = tl.program_id(1) * 2
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    ML: tl.constexpr = L * G
    r = tl.arange(0, MP)
    r_ok = r < ML
    r_l = r // G
    r_g = r % G
    d = tl.arange(0, 64)
    t0 = b * L
    tok = (t0 + r_l).to(tl.int64)
    head0 = kv0 * G + r_g
    head1 = head0 + G
    q0 = tl.load(q_ptr + (tok * HQ + head0)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0)
    q1 = tl.load(q_ptr + (tok * HQ + head1)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0)
    n = tl.load(seqlens_ptr + b)
    row_pos = n - L + r_l
    if WINDOW > 0:
        lo = tl.maximum(n - L - WINDOW + 1, 0)
    else:
        lo = 0
    per = tl.cdiv(tl.cdiv(n - lo, nsplit), BLOCK) * BLOCK
    start = lo + split * per
    end = tl.minimum(start + per, n)
    qk_scale = SM_SCALE * tl.load(k_scale_ptr) * 1.4426950408889634
    m0 = tl.full([MP], -1e+30, tl.float32)
    l0 = tl.zeros([MP], tl.float32)
    a0 = tl.zeros([MP, 64], tl.float32)
    m1 = tl.full([MP], -1e+30, tl.float32)
    l1 = tl.zeros([MP], tl.float32)
    a1 = tl.zeros([MP, 64], tl.float32)
    nb = tl.arange(0, BLOCK)
    pt_row = page_table_ptr + b.to(tl.int64) * max_pages
    for s0 in range(start, end, BLOCK):
        s = s0 + nb
        live = s < end
        page = tl.load(pt_row + s // PAGE, mask=live, other=0)
        kv_row = page.to(tl.int64) * PAGE + s % PAGE
        off = kv_row[:, None] * (HKV * 64) + kv0 * 64 + d[None, :]
        k0 = tl.load(k_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        k1 = tl.load(k_cache_ptr + off + 64, mask=live[:, None], other=0.0).to(tl.bfloat16)
        v0 = tl.load(v_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        v1 = tl.load(v_cache_ptr + off + 64, mask=live[:, None], other=0.0).to(tl.bfloat16)
        vis = live[None, :] & (s[None, :] <= row_pos[:, None])
        if WINDOW > 0:
            vis = vis & (s[None, :] > row_pos[:, None] - WINDOW)
        m0, l0, a0 = att__head_step(q0, k0, v0, m0, l0, a0, vis, qk_scale)
        m1, l1, a1 = att__head_step(q1, k1, v1, m1, l1, a1, vis, qk_scale)
    p0 = (tok * HQ + head0) * nsplit + split
    p1 = (tok * HQ + head1) * nsplit + split
    tl.store(o_part_ptr + p0[:, None] * 64 + d[None, :], a0, mask=r_ok[:, None])
    tl.store(o_part_ptr + p1[:, None] * 64 + d[None, :], a1, mask=r_ok[:, None])
    tl.store(m_part_ptr + p0, m0, mask=r_ok)
    tl.store(m_part_ptr + p1, m1, mask=r_ok)
    tl.store(l_part_ptr + p0, l0, mask=r_ok)
    tl.store(l_part_ptr + p1, l1, mask=r_ok)

@triton.jit(do_not_specialize=['nsplit'])
def att__combine(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit, HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    big = tl.max(tl.full([CB], -1e+30, tl.float32), 0)
    den = tl.sum(tl.zeros([CB], tl.float32), 0)
    num = tl.zeros([64], tl.float32)
    for s0 in range(0, nsplit, CB):
        ok = s0 + j < nsplit
        idx = th * nsplit + s0 + j
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1e+30)
        lsum = tl.load(l_part_ptr + idx, mask=ok, other=0.0)
        o = tl.load(o_part_ptr + idx[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.0)
        new = tl.maximum(big, tl.max(m, 0))
        scale = tl.exp2(big - new)
        w = tl.exp2(m - new)
        den = den * scale + tl.sum(w * lsum, 0)
        num = num * scale + tl.sum(w[:, None] * o, 0)
        big = new
    a = num / den * tl.load(v_scale_ptr)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / 64.0 + EPS)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    g = tl.load(qkv_ptr + t.to(tl.int64) * ROW + (HQ + 2 * HKV) * 64 + h * 64 + d).to(tl.float32)
    a = a / (1.0 + tl.exp(-g))
    tl.store(out_ptr + t.to(tl.int64) * (HQ * 64) + h * 64 + d, a.to(tl.bfloat16))

def att_plan(cfg, max_pages, tune=None):
    """Launch settings from the config and shapes only (never from lengths or values):
    (slices per request, keys per step, warps, stages, combine block, query rows)."""
    full, verify = (not cfg['window'], cfg['q_len'] > 1)
    per_sm, min_keys, block, warps, stages = tune or att_TUNE[full, verify, att_batch_class(cfg['batch'])]
    if cfg['num_kv_heads'] % 2:
        raise ValueError('this kernel pairs KV heads: num_kv_heads must be even (tp8: 2, dp: 16)')
    tokens = max_pages * cfg['page_size']
    if cfg['window']:
        tokens = min(tokens, cfg['window'] + cfg['q_len'])
    programs = cfg['batch'] * (cfg['num_kv_heads'] // 2)
    nsplit = max(1, min(math.ceil(per_sm * att_NUM_SMS / programs), tokens // min_keys, att_MAX_SPLITS))
    cb = min(64, triton.next_power_of_2(nsplit))
    rows = cfg['q_len'] * cfg['num_q_heads'] // cfg['num_kv_heads']
    mp = max(16, triton.next_power_of_2(rows))
    return (nsplit, block, warps, stages, cb, mp)

def att__inv_freq(theta, device):
    key = (theta, str(device))
    if key not in att__INV_FREQ:
        att__INV_FREQ[key] = 1.0 / theta ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64)
    return att__INV_FREQ[key]

def att_run(data, tune=None):
    cfg = data['config']
    batch, length = (cfg['batch'], cfg['q_len'])
    hq, hkv = (cfg['num_q_heads'], cfg['num_kv_heads'])
    tokens = batch * length
    qkv = data['qkv']
    device = qkv.device
    page_table = data['page_table']
    max_pages = page_table.shape[1]
    nsplit, block, warps, stages, cb, mp = att_plan(cfg, max_pages, tune)
    rope = bool(cfg['rope_theta'])
    inv_freq = att__inv_freq(cfg['rope_theta'] if rope else 1.0, device)
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    att__prologue[tokens,](qkv, data['positions'], data['cache_loc'], inv_freq, q, data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg['qk_scale_factor'], EPS=att_RMS_EPS, num_warps=4)
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    att__attend[batch, hkv // 2, nsplit](q, data['k_cache'], data['v_cache'], data['k_scale'], page_table, data['cache_seqlens'], o_part, m_part, l_part, max_pages, nsplit, HQ=hq, HKV=hkv, L=length, PAGE=cfg['page_size'], WINDOW=cfg['window'], SM_SCALE=cfg['sm_scale'], MP=mp, BLOCK=block, num_warps=warps, num_stages=stages)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    att__combine[tokens, hq](o_part, m_part, l_part, qkv, data['v_scale'], out, nsplit, HQ=hq, HKV=hkv, EPS=att_RMS_EPS, CB=cb, num_warps=4)
    return out
att_FP8 = torch.float8_e4m3fn
att__SG_NORMS = {}
att__SG_ROPE = {}
att_SG_ROPE_POSITIONS = 1 << 21

def att__sg_qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in att__SG_NORMS:
        att__SG_NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=att_RMS_EPS)
    return att__SG_NORMS[key]

def att__sg_cos_sin(theta, d, device):
    cache = att__SG_ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d)
        freqs = torch.einsum('i,j -> ij', torch.arange(att_SG_ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = att__SG_ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache

def att_sglang_path(data):
    cfg = data['config']
    batch, length = (cfg['batch'], cfg['q_len'])
    hq, hkv, d = (cfg['num_q_heads'], cfg['num_kv_heads'], cfg['head_dim'])
    tokens = batch * length
    qkv = data['qkv'].view(tokens, -1)
    device = qkv.device
    if cfg['rope_theta']:
        q, k = att__sg_qk_norm(hq, hkv, d)(qkv, data['positions'], att__sg_cos_sin(cfg['rope_theta'], d, device))
    else:
        q, k = att__sg_qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]
    k_cache, v_cache = (data['k_cache'], data['v_cache'])
    scaled = cfg['kv_scale'] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = (k / data['k_scale'], v / data['v_scale'])
    store_cache(k.to(att_FP8).view(torch.uint8), v.to(att_FP8).view(torch.uint8), k_cache.view(-1, hkv * d).view(torch.uint8), v_cache.view(-1, hkv * d).view(torch.uint8), data['cache_loc'], reserved_skip_index=-1)
    window = cfg['window']
    descale = {}
    if scaled:
        descale = dict(k_descale=data['k_scale'].reshape(1, 1).expand(batch, hkv), v_descale=data['v_scale'].reshape(1, 1).expand(batch, hkv))
    out = flash_attn_with_kvcache(q=q.view(tokens, hq, d).to(att_FP8), k_cache=k_cache, v_cache=v_cache, page_table=data['page_table'], cache_seqlens=data['cache_seqlens'], cu_seqlens_q=data['cu_seqlens_q'], max_seqlen_q=length, softmax_scale=cfg['sm_scale'] * cfg['qk_scale_factor'], causal=True, window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data['scheduler_metadata'], **descale)
    if isinstance(out, tuple):
        out = out[0]
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=att_RMS_EPS, has_norm=True)
    return o.view(batch, length, hq * d)
att_OURS = {(False, False, 1), (False, False, 16), (False, False, 32), (False, True, 1), (False, True, 16), (True, False, 1), (True, False, 16), (True, False, 32), (True, True, 1)}

def att_shape_class(cfg):
    return (not cfg['window'], cfg['q_len'] > 1, att_batch_class(cfg['batch']))

def att_kernel_fn(data):
    if not att__WARM and (not torch.cuda.is_current_stream_capturing()):
        att__warm_up()
    if att_shape_class(data['config']) in att_OURS:
        return att_run(data)
    return att_sglang_path(data)

def att_scored_max_pages(batch, length):
    """The scored cases' page-table width: 80,000-token context, page 1, fixed length at
    batch 1, agentic spread (+7.5%) above (reference.py _longest)."""
    longest = 80000 if batch == 1 else math.ceil(80000 * 1.075)
    return longest + length

def att_dummy_inputs(batch, length, window, max_pages, device, ctx=300, hq=16, hkv=2):
    """Small inputs with the scored shapes' config and page-table width (so plan() picks the
    same launch settings), used to compile every variant before any capture."""
    pages = batch * (ctx + length)
    cfg = dict(batch=batch, q_len=length, num_q_heads=hq, num_kv_heads=hkv, head_dim=64, page_size=1, window=window, rope_theta=500000.0 if window else 0.0, qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0)
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    table = (torch.arange(batch * max_pages, dtype=torch.int32, device=device) % pages).view(batch, max_pages)
    pos = (seq.long()[:, None] + torch.arange(length, device=device)).reshape(-1)
    return dict(qkv=torch.randn((batch, length, (2 * hq + 2 * hkv) * 64), device=device).to(torch.bfloat16), positions=pos, cache_loc=torch.gather(table.long(), 1, pos.view(batch, length)).reshape(-1), cache_seqlens=seq + length, cu_seqlens_q=torch.arange(0, batch * length + 1, length, dtype=torch.int32, device=device), scheduler_metadata=None, k_cache=torch.zeros((pages, 1, hkv, 64), device=device).to(torch.float8_e4m3fn), v_cache=torch.zeros((pages, 1, hkv, 64), device=device).to(torch.float8_e4m3fn), k_scale=torch.tensor(1.0, device=device), v_scale=torch.tensor(1.0, device=device), page_table=table, seq_lens=seq, config=cfg)

def att__warm_up():
    """Compile and load every scored variant on the first eager call (a correctness test),
    so no first launch (Triton compile + module load) can land inside a CUDA graph capture.
    Not at import: CUDA stays uninitialised until the harness first calls us."""
    att__WARM.append(True)
    dev = torch.device('cuda')
    for length in (1, 8):
        for window in (2048, 0):
            for batch in (1, 16, 32):
                data = att_dummy_inputs(batch, length, window, att_scored_max_pages(batch, length), dev)
                if att_shape_class(data['config']) in att_OURS:
                    att_run(data)
                else:
                    att_sglang_path(data)
"A5b: CUDA split-context decode attention (q_len 1) reading whole 128-byte cache rows (A17)\nwith both KV heads of a pair stacked in one MMA (A18), and page rows fetched two chunks ahead\n(A21). Source + declaration; make_lab.py appends this file to lab files.\n\nOne CTA = (request b, KV-head pair kv0/kv0+1, slice of keys). The pair's two 64-byte head rows\nare one contiguous 128-byte piece of every cache row (whole row under tp8). Warps take 16-token\nchunks round-robin, each with its own online softmax, and merge through shared memory at the end.\n\nStacked MMA (mma.m16n8k16, f32 += f16 x f16), lane (g, t4) = (lane >> 2, lane & 3):\n- QK: A rows g = head-0 query g, rows g+8 = head-1 query g; K-dim = the pair's 128 bytes in 8\n  k-steps. Logical column 2*t4+{0,1} (+8) of k-step ks <-> physical byte 32*t4 + 4*ks + {0,1}\n  ({2,3}); bytes 0-63 are head 0, 64-127 head 1, so lanes t4 = 0,1 carry head-0 columns and\n  t4 = 2,3 head-1 columns: the queries are block-diagonal (row g zero on head-1 columns, row g+8\n  zero on head-0 columns). B = keys: lane (g, t4) holds bytes [32*t4, 32*t4+32) of token g's\n  piece (tile 0) and token 8+g's (tile 1) as two 16 B loads each; every load instruction of a\n  warp touches all four 32 B sectors of each row, so each row is fetched whole.\n  C: c0,c1 = head-0 scores of query g, c2,c3 = head-1 scores, tokens 2*t4+{0,1} of the tile.\n- PV: A = P (rows g head 0, g+8 head 1; tokens 0-15) straight from the QK accumulators.\n  B = values over 16 d-tiles of the 128-byte piece: logical column g of d-tile j <-> physical\n  byte 16*g + j, so lane (g, t4) loads 16 contiguous bytes [16*g, 16*g+16) of tokens 2*t4,\n  2*t4+1, 2*t4+8, 2*t4+9 (8 lanes cover each row: whole-row loads), packed with __byte_perm.\n  C: column n = 2*t4+jj of tile j <-> byte 16*(2*t4+jj) + j: lanes t4 = 0,1 keep c0,c1 (head 0,\n  d = 32*t4 + 16*jj + j), lanes t4 = 2,3 keep c2,c3 (head 1, d = 32*(t4-2) + 16*jj + j).\nFP8 E4M3 -> FP16 is exact; queries FP16 (finer than the FP8 the rules allow); P FP16; FP32\naccumulation. Partials (log2-domain max, sum, output) in _combine's layout.\n"
att_CUDA_DEC2_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_dec2 {\n\nconstexpr int CHUNK = 16;\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\nstruct Chunk {\n  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16\n  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16\n};\n\ntemplate <int WARPS>\n__global__ void __launch_bounds__(WARPS * 32) attend_dec2_kernel(\n    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,\n    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n  __shared__ float sm_m[WARPS][2][8];\n  __shared__ float sm_l[WARPS][2][8];\n  __shared__ float sm_o[WARPS][2][8][64];\n\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int hl = t4 >> 1;                              // the head whose columns this lane carries\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;       // the pair\'s 128-byte piece within a row\n\n  const int n = seqlens[b];\n  const int lo = window > 0 ? max(n - window, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.\n  uint32_t qa[8][4];\n  {\n    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);\n#pragma unroll\n    for (int ks = 0; ks < 8; ++ks) {\n      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j\n      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));\n      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));\n      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}\n      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8\n      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}\n      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8\n    }\n  }\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  float m_run[2] = {-1.0e30f, -1.0e30f};\n  float l_run[2] = {0.f, 0.f};\n  float acc[16][4];\n#pragma unroll\n  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n\n  // Row of token (start + c*CHUNK + (lane & 15)) of chunk c, clamped into the slice.\n  auto row_of = [&](int c) -> long {\n    const int s = min(start + c * CHUNK + (lane & 15), end - 1);\n    return static_cast<long>(pt[s / page]) * page + s % page;\n  };\n  auto load = [&](long my_row, Chunk& ch) {\n    const long r0 = __shfl_sync(FULL, my_row, g);\n    const long r1 = __shfl_sync(FULL, my_row, 8 + g);\n    const long v0 = __shfl_sync(FULL, my_row, 2 * t4);\n    const long v1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);\n    const long v2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);\n    const long v3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);\n    const uint8_t* kb0 = kc + r0 * row_bytes + poff + 32 * t4;\n    const uint8_t* kb1 = kc + r1 * row_bytes + poff + 32 * t4;\n    ch.k[0] = __ldg(reinterpret_cast<const uint4*>(kb0));\n    ch.k[2] = __ldg(reinterpret_cast<const uint4*>(kb1));\n    ch.k[1] = __ldg(reinterpret_cast<const uint4*>(kb0 + 16));\n    ch.k[3] = __ldg(reinterpret_cast<const uint4*>(kb1 + 16));\n    ch.v[0] = __ldg(reinterpret_cast<const uint4*>(vc + v0 * row_bytes + poff + 16 * g));\n    ch.v[1] = __ldg(reinterpret_cast<const uint4*>(vc + v1 * row_bytes + poff + 16 * g));\n    ch.v[2] = __ldg(reinterpret_cast<const uint4*>(vc + v2 * row_bytes + poff + 16 * g));\n    ch.v[3] = __ldg(reinterpret_cast<const uint4*>(vc + v3 * row_bytes + poff + 16 * g));\n  };\n\n  // Pipeline per warp: rows two chunks ahead, data one chunk ahead.\n  Chunk cur, nxt;\n  long row_next = 0, row_next2 = 0;\n  int c = warp;\n  if (c < nchunks) {\n    load(row_of(c), cur);\n    if (c + WARPS < nchunks) row_next = row_of(c + WARPS);\n  }\n  for (; c < nchunks; c += WARPS) {\n    if (c + 2 * WARPS < nchunks) row_next2 = row_of(c + 2 * WARPS);\n    if (c + WARPS < nchunks) load(row_next, nxt);\n    const int s0 = start + c * CHUNK;\n\n    float sc[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n      for (int ks = 0; ks < 8; ++ks) {\n        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);\n        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));\n      }\n    }\n    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n      for (int jj = 0; jj < 2; ++jj) {\n        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;\n        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;\n        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;\n        mx0 = fmaxf(mx0, sc[nt][jj]);\n        mx1 = fmaxf(mx1, sc[nt][2 + jj]);\n      }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);\n    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);\n    float p[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      p[nt][0] = ex2(sc[nt][0] - mn0);\n      p[nt][1] = ex2(sc[nt][1] - mn0);\n      p[nt][2] = ex2(sc[nt][2] - mn1);\n      p[nt][3] = ex2(sc[nt][3] - mn1);\n    }\n    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n    m_run[0] = mn0;\n    m_run[1] = mn1;\n    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1\n    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)\n    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4\n    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8\n#pragma unroll\n    for (int j = 0; j < 16; ++j) {\n      acc[j][0] *= al0;\n      acc[j][1] *= al0;\n      acc[j][2] *= al1;\n      acc[j][3] *= al1;\n      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));\n      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));\n      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);\n    }\n    cur = nxt;\n    row_next = row_next2;\n  }\n\n  // Merge the warps through shared memory (output d un-permuted), then write the partials.\n  float l0 = l_run[0], l1 = l_run[1];\n  l0 += __shfl_xor_sync(FULL, l0, 1);\n  l0 += __shfl_xor_sync(FULL, l0, 2);\n  l1 += __shfl_xor_sync(FULL, l1, 1);\n  l1 += __shfl_xor_sync(FULL, l1, 2);\n  if (t4 == 0) {\n    sm_m[warp][0][g] = m_run[0];\n    sm_m[warp][1][g] = m_run[1];\n    sm_l[warp][0][g] = l0;\n    sm_l[warp][1][g] = l1;\n  }\n#pragma unroll\n  for (int j = 0; j < 16; ++j)\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj)\n      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;\n  __syncthreads();\n  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {\n    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;\n    float mm = -1.0e30f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);\n    float o = 0.f, l = 0.f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) {\n      const float wt = ex2(sm_m[w][h][row] - mm);\n      o += wt * sm_o[w][h][row][d];\n      l += wt * sm_l[w][h][row];\n    }\n    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;\n    o_part[part * 64 + d] = o;\n    if (d == 0) {\n      m_part[part] = mm;\n      l_part[part] = l;\n    }\n  }\n}\n\n}  // namespace okc_dec2\n\nvoid attend_dec2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,\n                 int64_t page, int64_t window, double sm_scale, int64_t warps) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n#define OKC_DEC2_ARGS                                                                                        \\\n  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \\\n      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \\\n      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)\n  if (warps == 8) okc_dec2::attend_dec2_kernel<8><<<grid, 256, 0, stream>>>(OKC_DEC2_ARGS);\n  else okc_dec2::attend_dec2_kernel<4><<<grid, 128, 0, stream>>>(OKC_DEC2_ARGS);\n#undef OKC_DEC2_ARGS\n}\n\nstd::vector<int64_t> attend_dec2_attrs(int64_t warps) {\n  cudaFuncAttributes a;\n  if (warps == 8) cudaFuncGetAttributes(&a, okc_dec2::attend_dec2_kernel<8>);\n  else cudaFuncGetAttributes(&a, okc_dec2::attend_dec2_kernel<4>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_DEC2_DECL = 'void attend_dec2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t warps);\nstd::vector<int64_t> attend_dec2_attrs(int64_t warps);'
"D2: A5b's stacked-head decode attention (cuda_dec2.py: whole 128-byte rows, both KV heads of a\npair in one MMA, the same fragment permutation) fed from a warp-private cp.async ring instead of\nregister prefetch (AF21/AF24: register-direct kernels plateau at ~2 TB/s; L5's warp-private\nrings reached 3.79 TB/s).\n\nEach warp streams its own 16-token chunks (c = warp, warp+WARPS, ...) through STAGES stages of\n16 tokens x (K row 128 B, V row 128 B) at a 144-byte pitch: lane copies 16 B for token\n4*q + (lane >> 3), byte chunk lane & 7 (q = 0..3, K and V), so every copy instruction covers\nwhole rows. Page rows come one iteration ahead (shuffled from lanes 0..15), so address\ngeneration never waits. Only cp.async.wait_group + __syncwarp synchronize; no CTA barrier in\nthe loop. Fragments are read from shared memory exactly where A5b loaded them from global:\nK bytes 32*t4 (+16) of tokens g / 8+g, V bytes 16*g of tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9.\nPitch 144 B = 9 units of 16 B makes the 16 B reads and the row copies bank-conflict free\n(unit = 9*token + chunk mod 8 is distinct in every 8-lane phase). The ring is dynamic shared\nmemory (attend_dec3_setup raises the limit once, outside any capture).\n"
att_CUDA_DEC3_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_dec3 {\n\nconstexpr int CHUNK = 16;\nconstexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B\nconstexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void cp16(void* dst, const void* src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));\n}\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\nstruct Chunk {\n  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16\n  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16\n};\n\ntemplate <int WARPS, int STAGES>\n__global__ void __launch_bounds__(WARPS * 32) attend_dec3_kernel(\n    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,\n    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n  __shared__ float sm_m[WARPS][2][8];\n  __shared__ float sm_l[WARPS][2][8];\n  __shared__ float sm_o[WARPS][2][8][64];\n\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int hl = t4 >> 1;                              // the head whose columns this lane carries\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;       // the pair\'s 128-byte piece within a row\n\n  const int n = seqlens[b];\n  const int lo = window > 0 ? max(n - window, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.\n  uint32_t qa[8][4];\n  {\n    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);\n#pragma unroll\n    for (int ks = 0; ks < 8; ++ks) {\n      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j\n      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));\n      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));\n      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}\n      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8\n      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}\n      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8\n    }\n  }\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  float m_run[2] = {-1.0e30f, -1.0e30f};\n  float l_run[2] = {0.f, 0.f};\n  float acc[16][4];\n#pragma unroll\n  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n\n  extern __shared__ __align__(16) uint8_t dsm[];\n  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;\n  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp\'s chunks\n  // Row of token (start + c*CHUNK + (lane & 15)) of the warp\'s i-th chunk, clamped into the slice.\n  auto row_of = [&](int i) -> long {\n    const int s = min(start + (warp + i * WARPS) * CHUNK + (lane & 15), end - 1);\n    return static_cast<long>(pt[s / page]) * page + s % page;\n  };\n  auto issue = [&](int i, long my_row) {\n    if (i < mine) {\n      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;\n#pragma unroll\n      for (int qq = 0; qq < 4; ++qq) {\n        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;\n        const long row = __shfl_sync(FULL, my_row, tok);\n        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);\n        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);\n      }\n    }\n    asm volatile("cp.async.commit_group;");\n  };\n\n  long row_next = 0;\n#pragma unroll\n  for (int i = 0; i < STAGES - 1; ++i) issue(i, i < mine ? row_of(i) : 0);\n  if (STAGES - 1 < mine) row_next = row_of(STAGES - 1);\n  for (int i = 0; i < mine; ++i) {\n    issue(i + STAGES - 1, row_next);\n    if (i + STAGES < mine) row_next = row_of(i + STAGES);\n    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));\n    __syncwarp();\n    const int c = warp + i * WARPS;\n    const uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;\n    const uint8_t* sk = st;\n    const uint8_t* sv = st + CHUNK * PITCH;\n    Chunk cur;\n    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);\n    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);\n    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);\n    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);\n    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);\n    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);\n    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);\n    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);\n    const int s0 = start + c * CHUNK;\n\n    float sc[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n      for (int ks = 0; ks < 8; ++ks) {\n        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);\n        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));\n      }\n    }\n    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n      for (int jj = 0; jj < 2; ++jj) {\n        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;\n        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;\n        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;\n        mx0 = fmaxf(mx0, sc[nt][jj]);\n        mx1 = fmaxf(mx1, sc[nt][2 + jj]);\n      }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);\n    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);\n    float p[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      p[nt][0] = ex2(sc[nt][0] - mn0);\n      p[nt][1] = ex2(sc[nt][1] - mn0);\n      p[nt][2] = ex2(sc[nt][2] - mn1);\n      p[nt][3] = ex2(sc[nt][3] - mn1);\n    }\n    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n    m_run[0] = mn0;\n    m_run[1] = mn1;\n    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1\n    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)\n    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4\n    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8\n#pragma unroll\n    for (int j = 0; j < 16; ++j) {\n      acc[j][0] *= al0;\n      acc[j][1] *= al0;\n      acc[j][2] *= al1;\n      acc[j][3] *= al1;\n      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));\n      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));\n      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);\n    }\n    __syncwarp();          // every lane done reading this stage before it is refilled\n  }\n  asm volatile("cp.async.wait_all;");\n\n  // Merge the warps through shared memory (output d un-permuted), then write the partials.\n  float l0 = l_run[0], l1 = l_run[1];\n  l0 += __shfl_xor_sync(FULL, l0, 1);\n  l0 += __shfl_xor_sync(FULL, l0, 2);\n  l1 += __shfl_xor_sync(FULL, l1, 1);\n  l1 += __shfl_xor_sync(FULL, l1, 2);\n  if (t4 == 0) {\n    sm_m[warp][0][g] = m_run[0];\n    sm_m[warp][1][g] = m_run[1];\n    sm_l[warp][0][g] = l0;\n    sm_l[warp][1][g] = l1;\n  }\n#pragma unroll\n  for (int j = 0; j < 16; ++j)\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj)\n      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;\n  __syncthreads();\n  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {\n    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;\n    float mm = -1.0e30f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);\n    float o = 0.f, l = 0.f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) {\n      const float wt = ex2(sm_m[w][h][row] - mm);\n      o += wt * sm_o[w][h][row][d];\n      l += wt * sm_l[w][h][row];\n    }\n    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;\n    o_part[part * 64 + d] = o;\n    if (d == 0) {\n      m_part[part] = mm;\n      l_part[part] = l;\n    }\n  }\n}\n\n}  // namespace okc_dec3\n\nvoid attend_dec3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,\n                 int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n  const size_t smem = static_cast<size_t>(warps) * stages * okc_dec3::STAGE_BYTES;\n#define OKC_DEC3_ARGS                                                                                        \\\n  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \\\n      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \\\n      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)\n  if (warps == 8 && stages == 2) okc_dec3::attend_dec3_kernel<8, 2><<<grid, 256, smem, stream>>>(OKC_DEC3_ARGS);\n  else if (warps == 8) okc_dec3::attend_dec3_kernel<8, 3><<<grid, 256, smem, stream>>>(OKC_DEC3_ARGS);\n  else if (stages == 4) okc_dec3::attend_dec3_kernel<4, 4><<<grid, 128, smem, stream>>>(OKC_DEC3_ARGS);\n  else okc_dec3::attend_dec3_kernel<4, 3><<<grid, 128, smem, stream>>>(OKC_DEC3_ARGS);\n#undef OKC_DEC3_ARGS\n}\n\nint64_t attend_dec3_setup() {   // once, outside any capture: allow each variant\'s ring\n  // Each limit is exactly the ring a variant needs (static + dynamic must stay <= 227 KB per\n  // block); the last-error state is cleared so a refusal can\'t surface in a later PyTorch call.\n  int err = 0;\n  using okc_dec3::STAGE_BYTES;\n  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 3 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 3 * STAGE_BYTES);\n  (void)cudaGetLastError();\n  return err;\n}\n\nstd::vector<int64_t> attend_dec3_attrs(int64_t warps, int64_t stages) {\n  cudaFuncAttributes a;\n  if (warps == 8 && stages == 2) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<8, 2>);\n  else if (warps == 8) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<8, 3>);\n  else if (stages == 4) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<4, 4>);\n  else cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<4, 3>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_DEC3_DECL = 'void attend_dec3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages);\nint64_t attend_dec3_setup();\nstd::vector<int64_t> attend_dec3_attrs(int64_t warps, int64_t stages);'
"D3: D2 (cuda_dec3.py) with its page entries carried by the cp.async ring (generated by\nmake_pt_ring.py; see its docstring): commit group i = chunk i's K/V rows + chunk i+STAGES's\npage entries (4-byte copies into the K tiles' unused row pads), so no iteration waits on a\ndependent page-table load; the warp merge reuses the ring instead of static shared memory.\nMath, chunk assignment and fragment layout are D2's (emulate_cuda_dec3.py applies).\n"
att_CUDA_DEC4_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_dec4 {\n\nconstexpr int CHUNK = 16;\nconstexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B\nconstexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void cp16(void* dst, const void* src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));\n}\n__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)\n  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));\n}\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\nstruct Chunk {\n  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16\n  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16\n};\n\ntemplate <int WARPS, int STAGES>\n__global__ void __launch_bounds__(WARPS * 32) attend_dec4_kernel(\n    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,\n    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int hl = t4 >> 1;                              // the head whose columns this lane carries\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;       // the pair\'s 128-byte piece within a row\n\n  const int n = seqlens[b];\n  const int lo = window > 0 ? max(n - window, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.\n  uint32_t qa[8][4];\n  {\n    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);\n#pragma unroll\n    for (int ks = 0; ks < 8; ++ks) {\n      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j\n      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));\n      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));\n      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}\n      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8\n      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}\n      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8\n    }\n  }\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  float m_run[2] = {-1.0e30f, -1.0e30f};\n  float l_run[2] = {0.f, 0.f};\n  float acc[16][4];\n#pragma unroll\n  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n\n  extern __shared__ __align__(16) uint8_t dsm[];\n  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;\n  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp\'s chunks\n  // Token of lane (lane & 15) in the warp\'s chunk j (c = warp + j*WARPS), clamped into the slice.\n  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };\n  // Page entries ride the ring: chunk j\'s 16 entries sit in PT slot p = j % PT_SLOTS, in the unused\n  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at\n  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.\n  constexpr int PT_SLOTS = 2 * STAGES;\n  auto pt_slot = [&](int j, int e) -> int* {\n    const int p = j % PT_SLOTS;\n    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));\n  };\n  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j\'s page entries (join the open commit group)\n    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);\n  };\n  // One commit group per call: chunk i\'s K/V rows (its page entries landed in an earlier group,\n  // waited and warp-synced) and chunk i + STAGES\'s page entries.\n  auto issue = [&](int i) {\n    if (i < mine) {\n      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;\n      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;\n#pragma unroll\n      for (int qq = 0; qq < 4; ++qq) {\n        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;\n        const long row = __shfl_sync(FULL, my_row, tok);\n        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);\n        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);\n      }\n    }\n    issue_pt(i + STAGES);\n    asm volatile("cp.async.commit_group;");\n  };\n  // Prologue: the first STAGES chunks\' page entries (the one exposed round trip), then the first\n  // STAGES - 1 stages of K/V rows.\n  for (int j = 0; j < STAGES; ++j) issue_pt(j);\n  asm volatile("cp.async.commit_group;");\n  asm volatile("cp.async.wait_group 0;");\n  __syncwarp();\n#pragma unroll\n  for (int i = 0; i < STAGES - 1; ++i) issue(i);\n  for (int i = 0; i < mine; ++i) {\n    issue(i + STAGES - 1);\n    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));\n    __syncwarp();\n    const int c = warp + i * WARPS;\n    const uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;\n    const uint8_t* sk = st;\n    const uint8_t* sv = st + CHUNK * PITCH;\n    Chunk cur;\n    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);\n    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);\n    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);\n    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);\n    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);\n    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);\n    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);\n    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);\n    const int s0 = start + c * CHUNK;\n\n    float sc[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n      for (int ks = 0; ks < 8; ++ks) {\n        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);\n        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));\n      }\n    }\n    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n      for (int jj = 0; jj < 2; ++jj) {\n        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;\n        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;\n        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;\n        mx0 = fmaxf(mx0, sc[nt][jj]);\n        mx1 = fmaxf(mx1, sc[nt][2 + jj]);\n      }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);\n    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);\n    float p[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      p[nt][0] = ex2(sc[nt][0] - mn0);\n      p[nt][1] = ex2(sc[nt][1] - mn0);\n      p[nt][2] = ex2(sc[nt][2] - mn1);\n      p[nt][3] = ex2(sc[nt][3] - mn1);\n    }\n    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n    m_run[0] = mn0;\n    m_run[1] = mn1;\n    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1\n    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)\n    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4\n    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8\n#pragma unroll\n    for (int j = 0; j < 16; ++j) {\n      acc[j][0] *= al0;\n      acc[j][1] *= al0;\n      acc[j][2] *= al1;\n      acc[j][3] *= al1;\n      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));\n      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));\n      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);\n    }\n    __syncwarp();          // every lane done reading this stage before it is refilled\n  }\n  asm volatile("cp.async.wait_all;");\n  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it\n  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");\n  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);\n  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);\n  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);\n\n  // Merge the warps through shared memory (output d un-permuted), then write the partials.\n  float l0 = l_run[0], l1 = l_run[1];\n  l0 += __shfl_xor_sync(FULL, l0, 1);\n  l0 += __shfl_xor_sync(FULL, l0, 2);\n  l1 += __shfl_xor_sync(FULL, l1, 1);\n  l1 += __shfl_xor_sync(FULL, l1, 2);\n  if (t4 == 0) {\n    sm_m[warp][0][g] = m_run[0];\n    sm_m[warp][1][g] = m_run[1];\n    sm_l[warp][0][g] = l0;\n    sm_l[warp][1][g] = l1;\n  }\n#pragma unroll\n  for (int j = 0; j < 16; ++j)\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj)\n      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;\n  __syncthreads();\n  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {\n    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;\n    float mm = -1.0e30f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);\n    float o = 0.f, l = 0.f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) {\n      const float wt = ex2(sm_m[w][h][row] - mm);\n      o += wt * sm_o[w][h][row][d];\n      l += wt * sm_l[w][h][row];\n    }\n    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;\n    o_part[part * 64 + d] = o;\n    if (d == 0) {\n      m_part[part] = mm;\n      l_part[part] = l;\n    }\n  }\n}\n\n}  // namespace okc_dec4\n\nvoid attend_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,\n                 int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n  const size_t smem = static_cast<size_t>(warps) * stages * okc_dec4::STAGE_BYTES;\n#define OKC_DEC4_ARGS                                                                                        \\\n  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \\\n      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \\\n      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)\n  if (warps == 8 && stages == 2) okc_dec4::attend_dec4_kernel<8, 2><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);\n  else if (warps == 8) okc_dec4::attend_dec4_kernel<8, 3><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);\n  else if (stages == 4) okc_dec4::attend_dec4_kernel<4, 4><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);\n  else okc_dec4::attend_dec4_kernel<4, 3><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);\n#undef OKC_DEC4_ARGS\n}\n\nint64_t attend_dec4_setup() {   // once, outside any capture: allow each variant\'s ring\n  // Each limit is exactly the ring a variant needs (static + dynamic must stay <= 227 KB per\n  // block); the last-error state is cleared so a refusal can\'t surface in a later PyTorch call.\n  int err = 0;\n  using okc_dec4::STAGE_BYTES;\n  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 3 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 3 * STAGE_BYTES);\n  (void)cudaGetLastError();\n  return err;\n}\n\nstd::vector<int64_t> attend_dec4_attrs(int64_t warps, int64_t stages) {\n  cudaFuncAttributes a;\n  if (warps == 8 && stages == 2) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<8, 2>);\n  else if (warps == 8) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<8, 3>);\n  else if (stages == 4) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<4, 4>);\n  else cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<4, 3>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_DEC4_DECL = 'void attend_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages);\nint64_t attend_dec4_setup();\nstd::vector<int64_t> attend_dec4_attrs(int64_t warps, int64_t stages);'
"UD: unified decode kernel (q_len 1), one launch per layer (B2: A40-A43). D2's warp-private\ncp.async ring streaming (cuda_dec3.py) with A5b's stacked-head MMA, the fused kernel's prologue\nfold, RoPE table and in-kernel new key/value (cuda_fused.py; the new row is patched into the ring\nslot after its copy lands, and stored to the cache), then:\n- CTA partial (16 (head, row)s of the pair: log2 max, sum, 64 outputs) in shared memory;\n- A42 cluster merge: CS (8 or 16) co-scheduled CTAs per cluster; barrier.cluster; the leader\n  (rank 0) reads every rank's partial with mapa + ld.shared::cluster, merges; a second cluster\n  barrier keeps the peers' shared memory alive until the leader is done;\n- one cluster per (request, pair): the leader applies RMSNorm x sigmoid(gate) and stores BF16;\n- A43 several clusters: each leader writes its cluster partial, the last one (atomic ticket,\n  reset after use) merges them and stores.\nGrid x = nsplit = CS * clusters per (request, pair); cluster rank = blockIdx.x % CS.\n"
att_CUDA_UD_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_ud {\n\nconstexpr int CHUNK = 16;\nconstexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B\nconstexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void cp16(void* dst, const void* src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));\n}\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\n__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {\n  uint16_t d;\n  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n__device__ __forceinline__ uint16_t f2bf16(float x) {\n  uint16_t d;\n  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));\n  return d;\n}\n__device__ __forceinline__ uint32_t cluster_rank() {\n  uint32_t r;\n  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n  return r;\n}\n__device__ __forceinline__ void cluster_sync() {\n  asm volatile("barrier.cluster.arrive.release.aligned;\\n\\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n}\n__device__ __forceinline__ float ld_peer(const float* local, uint32_t rank) {   // DSMEM load\n  uint32_t a, r;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));\n  float v;\n  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a));\n  (void)r;\n  return v;\n}\n\nstruct Chunk {\n  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16\n  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16\n};\n\ntemplate <int WARPS, int STAGES, int CS>\n__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(WARPS * 32) ud_kernel(\n    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,\n    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,\n    const int* __restrict__ seqlens, float* __restrict__ cpart_o, float* __restrict__ cpart_m,\n    float* __restrict__ cpart_l, int* __restrict__ tickets, uint16_t* __restrict__ out,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {\n  __shared__ float sm_m[WARPS][2][8];\n  __shared__ float sm_l[WARPS][2][8];\n  __shared__ float sm_o[WARPS][2][8][64];\n  __shared__ float cta_m[16], cta_l[16];          // this CTA\'s partial, read by the cluster leader\n  __shared__ float cta_o[16][64];\n  __shared__ __align__(16) uint8_t new_k[128];\n  __shared__ __align__(16) uint8_t new_v[128];\n  __shared__ float rope_c[32], rope_s[32];\n  __shared__ int last_flag;\n\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int hl = t4 >> 1;                              // the head whose columns this lane carries\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;       // the pair\'s 128-byte piece within a row\n\n  const int n = seqlens[b];\n  const int lo = window > 0 ? max(n - window, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;\n  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;\n  const float pos = static_cast<float>(positions[b]);\n  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA\'s slice holds the new token\n\n  // ---- RoPE table for this request\'s position: one full-range sincosf per frequency (the\n  // arguments reach ~1e5 rad, CUDA\'s slow reduction path), shared by every lane.\n  if (rope) {\n    if (warp == 0) sincosf(pos * inv_freq[lane], &rope_s[lane], &rope_c[lane]);\n    __syncthreads();\n  }\n\n  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.\n  if (has_new && warp == 0) {\n    const int h = lane >> 4, d0 = 4 * (lane & 15);          // head kv0 + h, dims d0 .. d0+3\n    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;\n    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;\n    float k4[4], v4[4];\n    float ss = 0.f;\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n      k4[j] = bf16f(kp[j]);\n      v4[j] = bf16f(vp[j]);\n      ss += k4[j] * k4[j];\n    }\n    ss += __shfl_xor_sync(FULL, ss, 1);\n    ss += __shfl_xor_sync(FULL, ss, 2);\n    ss += __shfl_xor_sync(FULL, ss, 4);\n    ss += __shfl_xor_sync(FULL, ss, 8);                       // sum over the head\'s 16 lanes\n    const float r = rsqrtf(ss / 64.f + 1e-5f);\n#pragma unroll\n    for (int j = 0; j < 4; ++j) k4[j] *= r;\n    if (rope) {\n#pragma unroll\n      for (int p = 0; p < 2; ++p) {\n        const float sn = rope_s[d0 / 2 + p], cs = rope_c[d0 / 2 + p];\n        const float e = k4[2 * p], o = k4[2 * p + 1];\n        k4[2 * p] = e * cs - o * sn;\n        k4[2 * p + 1] = e * sn + o * cs;\n      }\n    }\n    const float kinv = 1.f / (*k_scale_p), vinv = 1.f / (*v_scale_p);\n    const uint32_t kw = e4m3x2(k4[0] * kinv, k4[1] * kinv) | (e4m3x2(k4[2] * kinv, k4[3] * kinv) << 16);\n    const uint32_t vw = e4m3x2(v4[0] * vinv, v4[1] * vinv) | (e4m3x2(v4[2] * vinv, v4[3] * vinv) << 16);\n    reinterpret_cast<uint32_t*>(new_k)[lane] = kw;            // bytes 4*lane .. = head h, dims d0 ..\n    reinterpret_cast<uint32_t*>(new_v)[lane] = vw;\n    const long row = cache_loc[b];\n    reinterpret_cast<uint32_t*>(kc + row * row_bytes + poff)[lane] = kw;\n    reinterpret_cast<uint32_t*>(vc + row * row_bytes + poff)[lane] = vw;\n  }\n\n  // ---- queries: lane = head hl, dims 32*(t4&1) .. +31 of query g; RMSNorm x factor, RoPE, FP16.\n  uint32_t qa[8][4];\n  {\n    const uint16_t* qp = qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);\n    float x[32];\n    float ss = 0.f;\n#pragma unroll\n    for (int j = 0; j < 32; ++j) {\n      x[j] = bf16f(qp[j]);\n      ss += x[j] * x[j];\n    }\n    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)\n    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;\n#pragma unroll\n    for (int j = 0; j < 32; ++j) x[j] *= r;\n    if (rope) {\n#pragma unroll\n      for (int p = 0; p < 16; ++p) {\n        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];\n        const float e = x[2 * p], o = x[2 * p + 1];\n        x[2 * p] = e * cs - o * sn;\n        x[2 * p + 1] = e * sn + o * cs;\n      }\n    }\n#pragma unroll\n    for (int ks = 0; ks < 8; ++ks) {\n      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);\n      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);\n      qa[ks][0] = hl == 0 ? lo2 : 0u;\n      qa[ks][1] = hl == 1 ? lo2 : 0u;\n      qa[ks][2] = hl == 0 ? hi2 : 0u;\n      qa[ks][3] = hl == 1 ? hi2 : 0u;\n    }\n  }\n  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  float m_run[2] = {-1.0e30f, -1.0e30f};\n  float l_run[2] = {0.f, 0.f};\n  float acc[16][4];\n#pragma unroll\n  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n\n  extern __shared__ __align__(16) uint8_t dsm[];\n  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;\n  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp\'s chunks\n  // Row of token (start + c*CHUNK + (lane & 15)) of the warp\'s i-th chunk, clamped into the slice.\n  auto row_of = [&](int i) -> long {\n    const int s = min(start + (warp + i * WARPS) * CHUNK + (lane & 15), end - 1);\n    return static_cast<long>(pt[s / page]) * page + s % page;\n  };\n  auto issue = [&](int i, long my_row) {\n    if (i < mine) {\n      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;\n#pragma unroll\n      for (int qq = 0; qq < 4; ++qq) {\n        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;\n        const long row = __shfl_sync(FULL, my_row, tok);\n        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);\n        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);\n      }\n    }\n    asm volatile("cp.async.commit_group;");\n  };\n\n  long row_next = 0;\n#pragma unroll\n  for (int i = 0; i < STAGES - 1; ++i) issue(i, i < mine ? row_of(i) : 0);\n  if (STAGES - 1 < mine) row_next = row_of(STAGES - 1);\n  for (int i = 0; i < mine; ++i) {\n    issue(i + STAGES - 1, row_next);\n    if (i + STAGES < mine) row_next = row_of(i + STAGES);\n    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));\n    __syncwarp();\n    const int c = warp + i * WARPS;\n    uint8_t* stw = ring + (i % STAGES) * STAGE_BYTES;\n    if (has_new && n - 1 >= start + c * CHUNK && n - 1 < start + (c + 1) * CHUNK) {\n      // the new token\'s row: take the key/value computed in-kernel (the copy read the cache row\n      // before or while it was being written)\n      const int tn = n - 1 - (start + c * CHUNK);\n      reinterpret_cast<uint32_t*>(stw + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_k)[lane];\n      reinterpret_cast<uint32_t*>(stw + CHUNK * PITCH + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_v)[lane];\n      __syncwarp();\n    }\n    const uint8_t* st = stw;\n    const uint8_t* sk = st;\n    const uint8_t* sv = st + CHUNK * PITCH;\n    Chunk cur;\n    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);\n    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);\n    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);\n    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);\n    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);\n    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);\n    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);\n    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);\n    const int s0 = start + c * CHUNK;\n\n    float sc[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n      for (int ks = 0; ks < 8; ++ks) {\n        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);\n        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));\n      }\n    }\n    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n      for (int jj = 0; jj < 2; ++jj) {\n        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;\n        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;\n        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;\n        mx0 = fmaxf(mx0, sc[nt][jj]);\n        mx1 = fmaxf(mx1, sc[nt][2 + jj]);\n      }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);\n    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);\n    float p[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      p[nt][0] = ex2(sc[nt][0] - mn0);\n      p[nt][1] = ex2(sc[nt][1] - mn0);\n      p[nt][2] = ex2(sc[nt][2] - mn1);\n      p[nt][3] = ex2(sc[nt][3] - mn1);\n    }\n    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n    m_run[0] = mn0;\n    m_run[1] = mn1;\n    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1\n    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)\n    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4\n    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8\n#pragma unroll\n    for (int j = 0; j < 16; ++j) {\n      acc[j][0] *= al0;\n      acc[j][1] *= al0;\n      acc[j][2] *= al1;\n      acc[j][3] *= al1;\n      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));\n      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));\n      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);\n    }\n    __syncwarp();          // every lane done reading this stage before it is refilled\n  }\n  asm volatile("cp.async.wait_all;");\n\n  // Merge the warps through shared memory (output d un-permuted), then write the partials.\n  float l0 = l_run[0], l1 = l_run[1];\n  l0 += __shfl_xor_sync(FULL, l0, 1);\n  l0 += __shfl_xor_sync(FULL, l0, 2);\n  l1 += __shfl_xor_sync(FULL, l1, 1);\n  l1 += __shfl_xor_sync(FULL, l1, 2);\n  if (t4 == 0) {\n    sm_m[warp][0][g] = m_run[0];\n    sm_m[warp][1][g] = m_run[1];\n    sm_l[warp][0][g] = l0;\n    sm_l[warp][1][g] = l1;\n  }\n#pragma unroll\n  for (int j = 0; j < 16; ++j)\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj)\n      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;\n  __syncthreads();\n  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {\n    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;\n    float mm = -1.0e30f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);\n    float o = 0.f, l = 0.f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) {\n      const float wt = ex2(sm_m[w][h][row] - mm);\n      o += wt * sm_o[w][h][row][d];\n      l += wt * sm_l[w][h][row];\n    }\n    const int hr = h * 8 + row;                    // pair-local (head, row) index 0..15\n    cta_o[hr][d] = o;\n    if (d == 0) {\n      cta_m[hr] = mm;\n      cta_l[hr] = l;\n    }\n  }\n\n  // ---- A42 cluster merge: the leader (rank 0) reads every rank\'s partial over DSMEM.\n  const uint32_t rank = cluster_rank();\n  const int nc = nsplit / CS;                       // clusters per (request, pair)\n  const int ci = split / CS;\n  cluster_sync();                                   // every CTA\'s partial written and visible\n  constexpr int RPW = 16 / WARPS;                   // (head, row)s per leader warp\n  float fo0[RPW], fo1[RPW], fl[RPW], fm[RPW];\n  if (rank == 0) {\n#pragma unroll\n    for (int k2 = 0; k2 < RPW; ++k2) {\n      const int hr = warp + k2 * WARPS;\n      fm[k2] = -1.0e30f; fl[k2] = 0.f; fo0[k2] = 0.f; fo1[k2] = 0.f;\n      if (hr < 16) {\n        float mm = -1.0e30f;\n#pragma unroll\n        for (int r = 0; r < CS; ++r) mm = fmaxf(mm, ld_peer(&cta_m[hr], r));\n        float l = 0.f, o0 = 0.f, o1 = 0.f;\n#pragma unroll\n        for (int r = 0; r < CS; ++r) {\n          const float wt = ex2(ld_peer(&cta_m[hr], r) - mm);\n          l += wt * ld_peer(&cta_l[hr], r);\n          o0 += wt * ld_peer(&cta_o[hr][lane], r);\n          o1 += wt * ld_peer(&cta_o[hr][32 + lane], r);\n        }\n        fm[k2] = mm; fl[k2] = l; fo0[k2] = o0; fo1[k2] = o1;\n      }\n    }\n  }\n  cluster_sync();                                   // the leader is done reading: peers may exit\n  if (rank != 0) return;\n\n  const float vsc = *v_scale_p;\n  auto finalize = [&](int hr, float o0, float o1, float l) {   // RMSNorm x sigmoid(gate), BF16 store\n    const float scale = vsc / l;\n    o0 *= scale;\n    o1 *= scale;\n    float ss = o0 * o0 + o1 * o1;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);\n    const float r = rsqrtf(ss / 64.f + 1e-5f);\n    const int head = kv0 * 8 + hr;\n    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + head * 64;\n    const float g0 = bf16f(gp[lane]), g1 = bf16f(gp[32 + lane]);\n    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;\n    op[lane] = f2bf16(o0 * r / (1.f + __expf(-g0)));\n    op[32 + lane] = f2bf16(o1 * r / (1.f + __expf(-g1)));\n  };\n  if (nc == 1) {\n#pragma unroll\n    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);\n    return;\n  }\n\n  // ---- A43 second level: cluster partials in global memory; the last leader (ticket) merges.\n  const long cbase = (static_cast<long>(b) * hq + kv0 * 8) * nc;      // (head kv0*8 + hr, cluster)\n#pragma unroll\n  for (int k2 = 0; k2 < RPW; ++k2) {\n    const int hr = warp + k2 * WARPS;\n    {\n      const long p = cbase + static_cast<long>(hr) * nc + ci;\n      cpart_o[p * 64 + lane] = fo0[k2];\n      cpart_o[p * 64 + 32 + lane] = fo1[k2];\n      if (lane == 0) {\n        cpart_m[p] = fm[k2];\n        cpart_l[p] = fl[k2];\n      }\n    }\n  }\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x == 0) last_flag = atomicAdd(tickets + blockIdx.y, 1) == nc - 1;\n  __syncthreads();\n  if (!last_flag) return;\n  __threadfence();\n  for (int hr = warp; hr < 16; hr += WARPS) {       // nc <= 64: one warp per (head, row), lanes over clusters\n    const long pb = cbase + static_cast<long>(hr) * nc;\n    float mm = -1.0e30f;\n    for (int c2 = lane; c2 < nc; c2 += 32) mm = fmaxf(mm, __ldcg(cpart_m + pb + c2));\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));\n    float l = 0.f, o0 = 0.f, o1 = 0.f;\n#pragma unroll 4\n    for (int c2 = 0; c2 < nc; ++c2) {\n      const float wt = ex2(__ldcg(cpart_m + pb + c2) - mm);\n      l += wt * __ldcg(cpart_l + pb + c2);\n      o0 += wt * __ldcg(cpart_o + (pb + c2) * 64 + lane);\n      o1 += wt * __ldcg(cpart_o + (pb + c2) * 64 + 32 + lane);\n    }\n    finalize(hr, o0, o1, l);\n  }\n  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)\n}\n\n}  // namespace okc_ud\n\nvoid ud(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,\n        torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,\n        torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out,\n        int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window,\n        int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n  const size_t smem = static_cast<size_t>(warps) * stages * okc_ud::STAGE_BYTES;\n#define OKC_UD_ARGS                                                                                           \\\n  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \\\n      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \\\n      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \\\n      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \\\n      cpart_o.data_ptr<float>(), cpart_m.data_ptr<float>(), cpart_l.data_ptr<float>(), tickets.data_ptr<int>(), \\\n      reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),             \\\n      static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), \\\n      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)\n  if (cs == 16) {\n    if (warps == 8) okc_ud::ud_kernel<8, 2, 16><<<grid, 256, smem, stream>>>(OKC_UD_ARGS);\n    else okc_ud::ud_kernel<4, 3, 16><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);\n  } else {\n    if (warps == 8) okc_ud::ud_kernel<8, 2, 8><<<grid, 256, smem, stream>>>(OKC_UD_ARGS);\n    else if (stages == 4) okc_ud::ud_kernel<4, 4, 8><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);\n    else okc_ud::ud_kernel<4, 3, 8><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);\n  }\n#undef OKC_UD_ARGS\n  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later\n  TORCH_CHECK(e == cudaSuccess, "ud launch failed: ", cudaGetErrorString(e));\n}\n\nint64_t ud_setup() {   // once, outside any capture: each variant\'s ring, and 16-CTA clusters\n  int err = 0;\n  using okc_ud::STAGE_BYTES;\n#define OKC_UD_SMEM(W, S, C) \\\n  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES)\n  OKC_UD_SMEM(4, 3, 8);\n  OKC_UD_SMEM(4, 4, 8);\n  OKC_UD_SMEM(8, 2, 8);\n  OKC_UD_SMEM(4, 3, 16);\n  OKC_UD_SMEM(8, 2, 16);\n#undef OKC_UD_SMEM\n  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<4, 3, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<8, 2, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);\n  (void)cudaGetLastError();\n  return err;\n}\n\nstd::vector<int64_t> ud_attrs(int64_t warps, int64_t stages, int64_t cs) {\n  cudaFuncAttributes a;\n  if (cs == 16) {\n    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<8, 2, 16>);\n    else cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 3, 16>);\n  } else {\n    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<8, 2, 8>);\n    else if (stages == 4) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 4, 8>);\n    else cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 3, 8>);\n  }\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_UD_DECL = 'void ud(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs);\nint64_t ud_setup();\nstd::vector<int64_t> ud_attrs(int64_t warps, int64_t stages, int64_t cs);'
"S2b: verify attention (q_len 8) with S2a's per-warp compute (cuda_ver.py: warp w owns head\nw>>1 and m-tiles 2(w&1), 2(w&1)+1; A5a's per-head fragment permutation; masks only where needed)\nand a Hopper producer/consumer ring instead of S2a's per-16-key __syncthreads (AF22):\n- one producer warp (warp 4) copies 64-key stages (whole 128-byte rows: 8 lanes x 16 B per row,\n  4 rows per instruction; page rows fetched a stage ahead) into a STAGES-deep ring in dynamic\n  shared memory (K rows at pitch 192 B, V rows at 144 B, as S2a), then each lane's\n  cp.async.mbarrier.arrive.noinc signals the stage's full barrier (count 32) when its copies land;\n- four consumer warps wait on the full barrier (try_wait.parity), compute the stage's four\n  16-key sub-chunks, and one lane per warp arrives on the stage's empty barrier (count 4);\n- the producer waits on the empty barrier before refilling a stage; no CTA-wide sync in the loop.\nBetween the Triton prologue (q scratch) and the Triton _combine (partials as S2a).\n"
att_CUDA_VER2_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_ver2 {\n\nconstexpr int CHUNK = 16;\nconstexpr int L = 8;\nconstexpr int KP = 192;   // K row pitch in shared memory (bytes)\nconstexpr int VP = 144;   // V row pitch\nconstexpr int SKEYS = 64; // keys per ring stage\nconstexpr int STAGE_BYTES = SKEYS * (KP + VP);\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void cp16(void* dst, const void* src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));\n}\n__device__ __forceinline__ uint32_t w4(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }\n\ntemplate <int STAGES>\n__global__ void __launch_bounds__(160) attend_ver2_kernel(\n    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,\n    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n  extern __shared__ __align__(16) uint8_t ring[];   // STAGES x (64 K rows at KP, 64 V rows at VP)\n  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int h = warp >> 1, mt0 = 2 * (warp & 1);\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;\n\n  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l\n  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // Query fragments: m-tile mt = mt0 + mi, rows g -> token 2*mt, g+8 -> token 2*mt+1, head (kv0+h)*8+g.\n  uint32_t qa[2][4][4];\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi) {\n    const int ta = 2 * (mt0 + mi);\n    const uint16_t* qA = q + ((static_cast<long>(b) * L + ta) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;\n    const uint16_t* qB = qA + static_cast<long>(hq) * 64;               // token ta + 1\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      qa[mi][ks][0] = pack_f16x2(bf16f(qA[4 * ks + 0]), bf16f(qA[4 * ks + 1]));\n      qa[mi][ks][1] = pack_f16x2(bf16f(qB[4 * ks + 0]), bf16f(qB[4 * ks + 1]));\n      qa[mi][ks][2] = pack_f16x2(bf16f(qA[4 * ks + 2]), bf16f(qA[4 * ks + 3]));\n      qa[mi][ks][3] = pack_f16x2(bf16f(qB[4 * ks + 2]), bf16f(qB[4 * ks + 3]));\n    }\n  }\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  const int nst = (nchunks + 3) / 4;                  // 64-key stages of this slice\n  float m_run[2][2], l_run[2][2], acc[2][8][4];\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi) {\n    m_run[mi][0] = m_run[mi][1] = -1.0e30f;\n    l_run[mi][0] = l_run[mi][1] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;\n  }\n\n  if (tid == 0) {\n#pragma unroll\n    for (int s = 0; s < STAGES; ++s) {\n      asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&full_bar[s])));   // producer lanes\n      asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(smem_u32(&empty_bar[s])));   // consumer warps\n    }\n    asm volatile("fence.mbarrier_init.release.cluster;");\n  }\n  __syncthreads();\n\n  if (warp == 4) {\n    // ---- producer: whole-row 16 B copies of each 64-key stage, rows one stage ahead.\n    auto rows_of = [&](int i, long& r0, long& r1) {\n      const int s0 = min(start + i * SKEYS + lane, end - 1), s1 = min(start + i * SKEYS + 32 + lane, end - 1);\n      r0 = static_cast<long>(pt[s0 / page]) * page + s0 % page;\n      r1 = static_cast<long>(pt[s1 / page]) * page + s1 % page;\n    };\n    long r0 = 0, r1 = 0, n0 = 0, n1 = 0;\n    if (nst > 0) rows_of(0, r0, r1);\n    for (int i = 0; i < nst; ++i) {\n      const int st = i % STAGES;\n      if (i + 1 < nst) rows_of(i + 1, n0, n1);\n      if (i >= STAGES) {                              // wait until the consumers released this stage\n        const uint32_t parity = static_cast<uint32_t>((i / STAGES - 1) & 1);\n        uint32_t done = 0;\n        while (!done)\n          asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n                       : "=r"(done) : "r"(smem_u32(&empty_bar[st])), "r"(parity) : "memory");\n      }\n      uint8_t* skw = ring + st * STAGE_BYTES;\n      uint8_t* svw = skw + SKEYS * KP;\n#pragma unroll\n      for (int qq = 0; qq < 16; ++qq) {               // 8 lanes per whole row, 4 rows per instruction\n        const int tq = qq * 4 + (lane >> 3), ch = lane & 7;\n        const long row = __shfl_sync(FULL, qq < 8 ? r0 : r1, tq & 31);\n        cp16(skw + tq * KP + 16 * ch, kc + row * row_bytes + poff + 16 * ch);\n        cp16(svw + tq * VP + 16 * ch, vc + row * row_bytes + poff + 16 * ch);\n      }\n      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&full_bar[st])) : "memory");\n      r0 = n0;\n      r1 = n1;\n    }\n    asm volatile("cp.async.wait_all;");\n    return;                                           // the producer writes no partials\n  }\n  // ---- consumers: wait for a stage, compute its four 16-key sub-chunks, release it.\n  for (int i = 0; i < nst; ++i) {\n    const int st = i % STAGES;\n    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);\n    uint32_t done = 0;\n    while (!done)\n      asm volatile("{\\n\\t.reg .pred p;\\n\\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n\\tselp.u32 %0, 1, 0, p;\\n\\t}"\n                   : "=r"(done) : "r"(smem_u32(&full_bar[st])), "r"(parity) : "memory");\n    const uint8_t* sk = ring + st * STAGE_BYTES;\n    const uint8_t* sv = sk + SKEYS * KP;\n#pragma unroll 1\n    for (int sc = 0; sc < 4; ++sc) {\n      const int c = i * 4 + sc;\n      if (c >= nchunks) break;\n        const int s0 = start + c * CHUNK;\n        const uint4 kb0 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + g) * KP + 64 * h + 16 * t4]);\n        const uint4 kb1 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + 8 + g) * KP + 64 * h + 16 * t4]);\n        const uint2 vb0 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4) * VP + 64 * h + 8 * g]);\n        const uint2 vb1 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 1) * VP + 64 * h + 8 * g]);\n        const uint2 vb2 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 8) * VP + 64 * h + 8 * g]);\n        const uint2 vb3 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 9) * VP + 64 * h + 8 * g]);\n        // a chunk needs masks only at the slice tail or where it reaches the block\'s own keys\n        const bool masked = s0 + CHUNK > end || s0 + CHUNK > n - L || window > 0;\n\n    #pragma unroll\n        for (int mi = 0; mi < 2; ++mi) {\n          float sc[2][4];\n    #pragma unroll\n          for (int nt = 0; nt < 2; ++nt) {\n            const uint4& kb = nt == 0 ? kb0 : kb1;\n            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n    #pragma unroll\n            for (int ks = 0; ks < 4; ++ks) {\n              const uint32_t w = w4(kb, ks);\n              mma16816(sc[nt], qa[mi][ks][0], qa[mi][ks][1], qa[mi][ks][2], qa[mi][ks][3], f16x2_lo(w), f16x2_hi(w));\n            }\n          }\n          const int posA = n - L + 2 * (mt0 + mi), posB = posA + 1;     // rows g / g+8\n          float mxA = -INFINITY, mxB = -INFINITY;\n    #pragma unroll\n          for (int nt = 0; nt < 2; ++nt)\n    #pragma unroll\n            for (int jj = 0; jj < 2; ++jj) {\n              const int key = s0 + nt * 8 + 2 * t4 + jj;\n              float xa = sc[nt][jj] * qk_scale, xb = sc[nt][2 + jj] * qk_scale;\n              if (masked) {\n                const bool ok = key < end;\n                const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);\n                const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);\n                xa = okA ? xa : -INFINITY;\n                xb = okB ? xb : -INFINITY;\n              }\n              sc[nt][jj] = xa;\n              sc[nt][2 + jj] = xb;\n              mxA = fmaxf(mxA, xa);\n              mxB = fmaxf(mxB, xb);\n            }\n          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));\n          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));\n          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));\n          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));\n          const float mnA = fmaxf(m_run[mi][0], mxA), mnB = fmaxf(m_run[mi][1], mxB);\n          const float alA = ex2(m_run[mi][0] - mnA), alB = ex2(m_run[mi][1] - mnB);\n          float p[2][4];\n    #pragma unroll\n          for (int nt = 0; nt < 2; ++nt) {\n            p[nt][0] = ex2(sc[nt][0] - mnA);\n            p[nt][1] = ex2(sc[nt][1] - mnA);\n            p[nt][2] = ex2(sc[nt][2] - mnB);\n            p[nt][3] = ex2(sc[nt][3] - mnB);\n          }\n          l_run[mi][0] = l_run[mi][0] * alA + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n          l_run[mi][1] = l_run[mi][1] * alB + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n          m_run[mi][0] = mnA;\n          m_run[mi][1] = mnB;\n          const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);\n          const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);\n          const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);\n          const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);\n    #pragma unroll\n          for (int j = 0; j < 8; ++j) {\n            acc[mi][j][0] *= alA;\n            acc[mi][j][1] *= alA;\n            acc[mi][j][2] *= alB;\n            acc[mi][j][3] *= alB;\n            const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n            const uint32_t b0 = f16x2_lo(__byte_perm(j < 4 ? vb0.x : vb0.y, j < 4 ? vb1.x : vb1.y, sel));\n            const uint32_t b1 = f16x2_lo(__byte_perm(j < 4 ? vb2.x : vb2.y, j < 4 ? vb3.x : vb3.y, sel));\n            mma16816(acc[mi][j], pa0, pa1, pa2, pa3, b0, b1);\n          }\n        }\n\n    }\n    __syncwarp();\n    if (lane == 0) {\n      uint64_t state;\n      asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem_u32(&empty_bar[st])) : "memory");\n      (void)state;\n    }\n  }\n\n  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4+jj) + j.\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi)\n#pragma unroll\n    for (int r = 0; r < 2; ++r) {\n      float l = l_run[mi][r];\n      l += __shfl_xor_sync(FULL, l, 1);\n      l += __shfl_xor_sync(FULL, l, 2);\n      const int tok = 2 * (mt0 + mi) + r;\n      const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;\n#pragma unroll\n      for (int j = 0; j < 8; ++j)\n#pragma unroll\n        for (int jj = 0; jj < 2; ++jj) o_part[part * 64 + 8 * (2 * t4 + jj) + j] = acc[mi][j][2 * r + jj];\n      if (t4 == 0) {\n        m_part[part] = m_run[mi][r];\n        l_part[part] = l;\n      }\n    }\n}\n\n}  // namespace okc_ver2\n\nvoid attend_ver2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,\n                 int64_t page, int64_t window, double sm_scale, int64_t stages) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n  const size_t smem = static_cast<size_t>(stages) * okc_ver2::STAGE_BYTES;\n#define OKC_VER2_ARGS                                                                                        \\\n  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \\\n      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \\\n      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)\n  if (stages == 4) okc_ver2::attend_ver2_kernel<4><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);\n  else if (stages == 2) okc_ver2::attend_ver2_kernel<2><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);\n  else okc_ver2::attend_ver2_kernel<3><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);\n#undef OKC_VER2_ARGS\n  const cudaError_t e = cudaGetLastError();\n  TORCH_CHECK(e == cudaSuccess, "attend_ver2 launch failed: ", cudaGetErrorString(e));\n}\n\nint64_t attend_ver2_setup() {\n  int err = 0;\n  using okc_ver2::STAGE_BYTES;\n  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * STAGE_BYTES);\n  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * STAGE_BYTES);\n  (void)cudaGetLastError();\n  return err;\n}\n\nstd::vector<int64_t> attend_ver2_attrs(int64_t stages) {\n  cudaFuncAttributes a;\n  if (stages == 4) cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<4>);\n  else if (stages == 2) cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<2>);\n  else cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<3>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_VER2_DECL = 'void attend_ver2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t stages);\nint64_t attend_ver2_setup();\nstd::vector<int64_t> attend_ver2_attrs(int64_t stages);'
import tempfile as _tempfile
att__CUDA = {}
att__TICKETS = []
att_CUDA_ROUTE = {(True, False, 16): ('dec4', 6, 4, 3), (True, False, 32): ('dec4', 6, 4, 3)}
att__KINDS = {'dec2': (att_CUDA_DEC2_SRC, att_CUDA_DEC2_DECL, ['attend_dec2'], None), 'dec3': (att_CUDA_DEC3_SRC, att_CUDA_DEC3_DECL, ['attend_dec3', 'attend_dec3_setup'], 'attend_dec3_setup'), 'dec4': (att_CUDA_DEC4_SRC, att_CUDA_DEC4_DECL, ['attend_dec4', 'attend_dec4_setup'], 'attend_dec4_setup'), 'ud': (att_CUDA_UD_SRC, att_CUDA_UD_DECL, ['ud', 'ud_setup'], 'ud_setup'), 'ver2': (att_CUDA_VER2_SRC, att_CUDA_VER2_DECL, ['attend_ver2', 'attend_ver2_setup'], 'attend_ver2_setup')}
att_SELF_CHECK_TOL = 0.02
att__WARM_DEVICE = 'cuda'

def att__cuda_build():
    from torch.utils.cpp_extension import load_inline
    kinds = sorted({route[0] for route in att_CUDA_ROUTE.values()})
    if not kinds:
        return
    try:
        mod = load_inline(name='okc_attention_v4_' + '_'.join(kinds), cpp_sources='\n'.join((att__KINDS[k][1] for k in kinds)), cuda_sources='\n'.join((att__KINDS[k][0] for k in kinds)), functions=[f for k in kinds for f in att__KINDS[k][2]], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'], build_directory=_tempfile.mkdtemp(prefix='okc_attention_v4_'), verbose=False)
    except Exception as exc:
        print(f'v4: CUDA build failed, staying on v2.2 paths: {type(exc).__name__}: {str(exc)[-300:]}')
        return
    att__CUDA['mod'] = mod
    att__CUDA['refused'] = set()
    for k in kinds:
        setup = att__KINDS[k][3]
        if setup is not None and getattr(mod, setup)() != 0:
            print(f'v4: {setup} refused (shared memory or cluster attribute); {k} routes dropped')
            att__CUDA['refused'].add(k)

def att__nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg['page_size']
    if cfg['window']:
        tokens = min(tokens, cfg['window'] + cfg['q_len'])
    programs = cfg['batch'] * (cfg['num_kv_heads'] // 2)
    return max(1, min(math.ceil(ctas_per_sm * att_NUM_SMS / programs), tokens // min_keys, att_MAX_SPLITS))

def att__ud_clusters(cfg, max_pages, cs, ctas_per_sm, warps):
    """Clusters per (request, head pair): one for sliding layers (the window fits one cluster),
    else enough for ctas_per_sm CTAs per SM, at least one warp chunk (32 keys) per warp."""
    if cfg['window']:
        return 1
    tokens = max_pages * cfg['page_size']
    programs = cfg['batch'] * (cfg['num_kv_heads'] // 2)
    nc = max(1, math.ceil(ctas_per_sm * att_NUM_SMS / (programs * cs)))
    return max(1, min(nc, tokens // (cs * 32 * warps)))

def att__prologue_q(data):
    cfg = data['config']
    hq, hkv = (cfg['num_q_heads'], cfg['num_kv_heads'])
    tokens = cfg['batch'] * cfg['q_len']
    device = data['qkv'].device
    rope = bool(cfg['rope_theta'])
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    att__prologue[tokens,](data['qkv'], data['positions'], data['cache_loc'], att__inv_freq(cfg['rope_theta'] if rope else 1.0, device), q, data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg['qk_scale_factor'], EPS=att_RMS_EPS, num_warps=4)
    return q

def att__merge(data, o_part, m_part, l_part, nsplit):
    cfg = data['config']
    batch, length, hq, hkv = (cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads'])
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data['qkv'].device)
    att__combine[batch * length, hq](o_part, m_part, l_part, data['qkv'], data['v_scale'], out, nsplit, HQ=hq, HKV=hkv, EPS=att_RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out

def att__cuda_run(data, route):
    kind = route[0]
    cfg = data['config']
    batch, length, hq, hkv = (cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads'])
    device = data['qkv'].device
    max_pages = data['page_table'].shape[1]
    mod = att__CUDA['mod']
    if kind == 'ud':
        _, cs, cps, warps, stages = route
        nc = att__ud_clusters(cfg, max_pages, cs, cps, warps)
        rope = bool(cfg['rope_theta'])
        cpart_o = torch.empty((batch * hq * nc, 64), dtype=torch.float32, device=device)
        cpart_m = torch.empty((batch * hq * nc,), dtype=torch.float32, device=device)
        cpart_l = torch.empty_like(cpart_m)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        mod.ud(data['qkv'], data['positions'], data['cache_loc'], att__inv_freq(cfg['rope_theta'] if rope else 1.0, device), data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], data['page_table'], data['cache_seqlens'], cpart_o, cpart_m, cpart_l, att__TICKETS[0], out, batch, hq, hkv, max_pages, cs * nc, cfg['page_size'], cfg['window'], int(rope), cfg['qk_scale_factor'], cfg['sm_scale'], warps, stages, cs)
        return out
    q = att__prologue_q(data)
    if kind == 'ver2':
        _, cps, stages = route
        nsplit = att__nsplit(cfg, max_pages, cps, 64)
    else:
        cps, warps = (route[1], route[2])
        nsplit = att__nsplit(cfg, max_pages, cps, 32 * warps)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    args = (q, data['k_cache'], data['v_cache'], data['k_scale'], data['page_table'], data['cache_seqlens'], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg['page_size'], cfg['window'], cfg['sm_scale'])
    if kind == 'dec2':
        mod.attend_dec2(*args, route[2])
    elif kind == 'dec3':
        mod.attend_dec3(*args, route[2], route[3])
    elif kind == 'dec4':
        mod.attend_dec4(*args, route[2], route[3])
    else:
        mod.attend_ver2(*args, route[2])
    return att__merge(data, o_part, m_part, l_part, nsplit)

def att__cuda_ok(cfg, route):
    """Every CUDA kernel needs groups of 8 query heads and paired KV heads; the decode kinds take
    q_len 1, ver2 q_len 8."""
    kind = route[0]
    if cfg['num_q_heads'] // cfg['num_kv_heads'] != 8 or cfg['num_kv_heads'] % 2:
        return False
    return cfg['q_len'] == (8 if kind == 'ver2' else 1)

def att__self_check_inputs(bclass, length, full, device):
    """Synthetic inputs of a class (our own random data, never the judge's): a random FP8 cache
    so every key matters, a context that spans the window on sliding layers."""
    data = att_dummy_inputs(bclass, length, 0 if full else 2048, att_scored_max_pages(bclass, length), device, ctx=3000 if full else 2300)
    gen = torch.Generator(device=device).manual_seed(4040 + bclass + length)
    for name in ('k_cache', 'v_cache'):
        t = data[name]
        data[name] = (torch.randn(t.shape, generator=gen, device=device) * 0.5).to(t.dtype)
    return data
att__kernel_fn_v22 = att_kernel_fn
att__warm_up_v22 = att__warm_up

def att__warm_up():
    att__warm_up_v22()
    att__cuda_build()
    if 'mod' not in att__CUDA:
        att_CUDA_ROUTE.clear()
        return
    dev = torch.device(att__WARM_DEVICE)
    att__TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    for cls, route in sorted(att_CUDA_ROUTE.items()):
        full, verify, bclass = cls
        try:
            if route[0] in att__CUDA['refused']:
                raise RuntimeError('setup refused')
            data = att__self_check_inputs(bclass, 8 if verify else 1, full, dev)
            if not att__cuda_ok(data['config'], route):
                raise RuntimeError('config not supported')
            ours = att__cuda_run(data, route).float()
            want = att_run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-06))
            if not err < att_SELF_CHECK_TOL:
                raise RuntimeError(f'differs from the Triton path: rel. L2 {err:.2e}')
        except Exception as exc:
            print(f'v4: route {cls} -> {route} dropped: {type(exc).__name__}: {str(exc)[-200:]}')
            att_CUDA_ROUTE.pop(cls)

def att_kernel_fn(data):
    if not att__WARM and (not torch.cuda.is_current_stream_capturing()):
        att__warm_up()
    cfg = data['config']
    route = att_CUDA_ROUTE.get(att_shape_class(cfg))
    if route is not None and 'mod' in att__CUDA and att__cuda_ok(cfg, route):
        return att__cuda_run(data, route)
    return att__kernel_fn_v22(data)
import triton.language.extra.cuda as tlcuda

@triton.jit
def att__prologue_p(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr, QK_FACTOR: tl.constexpr, EPS: tl.constexpr, TRIGGER_EARLY: tl.constexpr):
    if TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()
    t = tl.program_id(0)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    base = qkv_ptr + t.to(tl.int64) * ROW
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    i = tl.arange(0, 32)
    q_e = tl.load(base + hq[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    q_o = tl.load(base + hq[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    k_e = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    k_o = tl.load(base + HQ * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    v_e = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :]).to(tl.float32)
    v_o = tl.load(base + (HQ + HKV) * 64 + hk[:, None] * 64 + 2 * i[None, :] + 1).to(tl.float32)
    q_r = tl.rsqrt((tl.sum(q_e * q_e, 1) + tl.sum(q_o * q_o, 1)) / 64.0 + EPS)
    k_r = tl.rsqrt((tl.sum(k_e * k_e, 1) + tl.sum(k_o * k_o, 1)) / 64.0 + EPS)
    q_e = q_e * q_r[:, None] * QK_FACTOR
    q_o = q_o * q_r[:, None] * QK_FACTOR
    k_e = k_e * k_r[:, None]
    k_o = k_o * k_r[:, None]
    if ROPE:
        pos = tl.load(pos_ptr + t).to(tl.float32)
        ang = pos * tl.load(inv_freq_ptr + i)
        c = libdevice.cos(ang)[None, :]
        s = libdevice.sin(ang)[None, :]
        q_e, q_o = (q_e * c - q_o * s, q_e * s + q_o * c)
        k_e, k_o = (k_e * c - k_o * s, k_e * s + k_o * c)
    q_dst = q_out_ptr + t.to(tl.int64) * (HQ * 64) + hq[:, None] * 64 + 2 * i[None, :]
    tl.store(q_dst, q_e.to(tl.bfloat16))
    tl.store(q_dst + 1, q_o.to(tl.bfloat16))
    k_inv = 1.0 / tl.load(k_scale_ptr)
    v_inv = 1.0 / tl.load(v_scale_ptr)
    row = tl.load(loc_ptr + t)
    off = row * (HKV * 64) + hk[:, None] * 64 + 2 * i[None, :]
    tl.store(k_cache_ptr + off, (k_e * k_inv).to(tl.float8e4nv))
    tl.store(k_cache_ptr + off + 1, (k_o * k_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off, (v_e * v_inv).to(tl.float8e4nv))
    tl.store(v_cache_ptr + off + 1, (v_o * v_inv).to(tl.float8e4nv))
    if not TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()

@triton.jit(do_not_specialize=['max_pages', 'nsplit'])
def att__attend_p(q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, page_table_ptr, seqlens_ptr, o_part_ptr, m_part_ptr, l_part_ptr, max_pages, nsplit, HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, PAGE: tl.constexpr, WINDOW: tl.constexpr, SM_SCALE: tl.constexpr, MP: tl.constexpr, BLOCK: tl.constexpr, TRIGGER_EARLY: tl.constexpr):
    tlcuda.gdc_wait()
    if TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()
    b = tl.program_id(0)
    kv0 = tl.program_id(1) * 2
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    ML: tl.constexpr = L * G
    r = tl.arange(0, MP)
    r_ok = r < ML
    r_l = r // G
    r_g = r % G
    d = tl.arange(0, 64)
    t0 = b * L
    tok = (t0 + r_l).to(tl.int64)
    head0 = kv0 * G + r_g
    head1 = head0 + G
    q0 = tl.load(q_ptr + (tok * HQ + head0)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0)
    q1 = tl.load(q_ptr + (tok * HQ + head1)[:, None] * 64 + d[None, :], mask=r_ok[:, None], other=0.0)
    n = tl.load(seqlens_ptr + b)
    row_pos = n - L + r_l
    if WINDOW > 0:
        lo = tl.maximum(n - L - WINDOW + 1, 0)
    else:
        lo = 0
    per = tl.cdiv(tl.cdiv(n - lo, nsplit), BLOCK) * BLOCK
    start = lo + split * per
    end = tl.minimum(start + per, n)
    qk_scale = SM_SCALE * tl.load(k_scale_ptr) * 1.4426950408889634
    m0 = tl.full([MP], -1e+30, tl.float32)
    l0 = tl.zeros([MP], tl.float32)
    a0 = tl.zeros([MP, 64], tl.float32)
    m1 = tl.full([MP], -1e+30, tl.float32)
    l1 = tl.zeros([MP], tl.float32)
    a1 = tl.zeros([MP, 64], tl.float32)
    nb = tl.arange(0, BLOCK)
    pt_row = page_table_ptr + b.to(tl.int64) * max_pages
    for s0 in range(start, end, BLOCK):
        s = s0 + nb
        live = s < end
        page = tl.load(pt_row + s // PAGE, mask=live, other=0)
        kv_row = page.to(tl.int64) * PAGE + s % PAGE
        off = kv_row[:, None] * (HKV * 64) + kv0 * 64 + d[None, :]
        k0 = tl.load(k_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        k1 = tl.load(k_cache_ptr + off + 64, mask=live[:, None], other=0.0).to(tl.bfloat16)
        v0 = tl.load(v_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        v1 = tl.load(v_cache_ptr + off + 64, mask=live[:, None], other=0.0).to(tl.bfloat16)
        vis = live[None, :] & (s[None, :] <= row_pos[:, None])
        if WINDOW > 0:
            vis = vis & (s[None, :] > row_pos[:, None] - WINDOW)
        m0, l0, a0 = att__head_step(q0, k0, v0, m0, l0, a0, vis, qk_scale)
        m1, l1, a1 = att__head_step(q1, k1, v1, m1, l1, a1, vis, qk_scale)
    p0 = (tok * HQ + head0) * nsplit + split
    p1 = (tok * HQ + head1) * nsplit + split
    tl.store(o_part_ptr + p0[:, None] * 64 + d[None, :], a0, mask=r_ok[:, None])
    tl.store(o_part_ptr + p1[:, None] * 64 + d[None, :], a1, mask=r_ok[:, None])
    tl.store(m_part_ptr + p0, m0, mask=r_ok)
    tl.store(m_part_ptr + p1, m1, mask=r_ok)
    tl.store(l_part_ptr + p0, l0, mask=r_ok)
    tl.store(l_part_ptr + p1, l1, mask=r_ok)
    if not TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()

@triton.jit(do_not_specialize=['nsplit'])
def att__combine_p(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit, HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr):
    tlcuda.gdc_wait()
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    big = tl.max(tl.full([CB], -1e+30, tl.float32), 0)
    den = tl.sum(tl.zeros([CB], tl.float32), 0)
    num = tl.zeros([64], tl.float32)
    for s0 in range(0, nsplit, CB):
        ok = s0 + j < nsplit
        idx = th * nsplit + s0 + j
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1e+30)
        lsum = tl.load(l_part_ptr + idx, mask=ok, other=0.0)
        o = tl.load(o_part_ptr + idx[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.0)
        new = tl.maximum(big, tl.max(m, 0))
        scale = tl.exp2(big - new)
        w = tl.exp2(m - new)
        den = den * scale + tl.sum(w * lsum, 0)
        num = num * scale + tl.sum(w[:, None] * o, 0)
        big = new
    a = num / den * tl.load(v_scale_ptr)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / 64.0 + EPS)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    g = tl.load(qkv_ptr + t.to(tl.int64) * ROW + (HQ + 2 * HKV) * 64 + h * 64 + d).to(tl.float32)
    a = a / (1.0 + tl.exp(-g))
    tl.store(out_ptr + t.to(tl.int64) * (HQ * 64) + h * 64 + d, a.to(tl.bfloat16))

def att_run_pdl(data, tune=None, early=False):
    cfg = data['config']
    batch, length = (cfg['batch'], cfg['q_len'])
    hq, hkv = (cfg['num_q_heads'], cfg['num_kv_heads'])
    tokens = batch * length
    qkv = data['qkv']
    device = qkv.device
    page_table = data['page_table']
    max_pages = page_table.shape[1]
    nsplit, block, warps, stages, cb, mp = att_plan(cfg, max_pages, tune)
    rope = bool(cfg['rope_theta'])
    inv_freq = att__inv_freq(cfg['rope_theta'] if rope else 1.0, device)
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    att__prologue_p[tokens,](qkv, data['positions'], data['cache_loc'], inv_freq, q, data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg['qk_scale_factor'], EPS=att_RMS_EPS, TRIGGER_EARLY=early, num_warps=4)
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    att__attend_p[batch, hkv // 2, nsplit](q, data['k_cache'], data['v_cache'], data['k_scale'], page_table, data['cache_seqlens'], o_part, m_part, l_part, max_pages, nsplit, HQ=hq, HKV=hkv, L=length, PAGE=cfg['page_size'], WINDOW=cfg['window'], SM_SCALE=cfg['sm_scale'], MP=mp, BLOCK=block, TRIGGER_EARLY=early, num_warps=warps, num_stages=stages, launch_pdl=True)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    att__combine_p[tokens, hq](o_part, m_part, l_part, qkv, data['v_scale'], out, nsplit, HQ=hq, HKV=hkv, EPS=att_RMS_EPS, CB=cb, num_warps=4, launch_pdl=True)
    return out
att_PDL_ROUTE = {(False, False, 16), (False, False, 32), (True, False, 1), (False, True, 1), (False, True, 16), (False, True, 32)}
att__kernel_fn_v4 = att_kernel_fn
att__warm_up_v4 = att__warm_up

def att__warm_up():
    att__warm_up_v4()
    dev = torch.device(att__WARM_DEVICE)
    try:
        for full, verify, bclass in sorted(att_PDL_ROUTE):
            data = att__self_check_inputs(bclass, 8 if verify else 1, full, dev)
            ours = att_run_pdl(data, early=False).float()
            want = att_run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-06))
            if not err < att_SELF_CHECK_TOL:
                raise RuntimeError(f'class {(full, verify, bclass)} differs from the Triton path: {err:.2e}')
    except Exception as exc:
        print(f'v5: PDL chain dropped: {type(exc).__name__}: {str(exc)[-200:]}')
        att_PDL_ROUTE.clear()

def att_kernel_fn(data):
    if not att__WARM and (not torch.cuda.is_current_stream_capturing()):
        att__warm_up()
    cfg = data['config']
    cls = att_shape_class(cfg)
    if cls not in att_CUDA_ROUTE and cls in att_PDL_ROUTE:
        return att_run_pdl(data, early=False)
    return att__kernel_fn_v4(data)


# ====================================================================== parts/proj4.py

# Embedded: the bf16_proj session's scored v4 (proj/scored/submission_v4.py, sha256 4048d4cec26d1688...;
# scored 1.247x on muse_bf16_proj, job 0d1dc429eebf455f), every top-level name prefixed with 'pj_' by
# decode_layer/local/make_attn.py. pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m); P9 needs pj_p9_setup(state).

"""Dense projections: per-shape dispatch between SGLang's path and our own kernels.

CHOICE maps (layer, m) to a kernel; shapes not in it run SGLang's path exactly as baseline.py
does (cuBLAS; the router FP32 up to 8 tokens, BF16 hi + lo above). Our kernels:

  _swap_kernel  m <= 32: weights as the MMA's wide side, tokens padded to BM (16 or 32) as the
                narrow side; split-K over grid axis 1 with FP32 atomics when SPLIT > 1.
  _gemm_kernel  m >= 128: tokens as M, weights as N, split-K over grid axis 2 likewise.

  _fma_kernel   small m: CUDA-core GEMV. Each program loads its [BN, K / split] weight slice
                into registers at once, then for each token multiplies and row-sums in FP32
                (exact FP32 for the router's FP32 weights); split-K adds with FP32 atomics.

MODE 0: BF16 weights. MODE 1: router, FP32 weights split in registers into a BF16 hi + lo pair,
both accumulated into one FP32 accumulator (only above 8 tokens, as SGLang does). MODE 2:
router, exact FP32 (ieee) dot.
"""
from typing import TypedDict
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

class pj_Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor

class pj_Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str

class pj_Inputs(TypedDict):
    x: torch.Tensor
    config: pj_Config
pj_ROUTER_FP32_MAX_TOKENS = 8
pj_CHOICE = {('o', 1): {'kind': 'fma', 'BN': 4, 'split': 1, 'warps': 8}, ('pooler', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('unpooler', 1): {'kind': 'tiny', 'max_m': 16}, ('pooler', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 8): {'kind': 'p9', 'NT': 1, 'RTW': 1, 'RG': 2, 'KSW': 8, 'D': 2, 'CS': 8}, ('unpooler', 8): {'kind': 'fi_mm', 'backend': 'tinygemm'}, ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('pooler', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 16): {'kind': 'p9', 'NT': 2, 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'CS': 8}, ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 128): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 256): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('qkv_gate', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('router', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('qkv_gate', 8): {'kind': 'fi_mm', 'backend': 'auto'}, ('pooler', 128): {'kind': 'dg', 'native': True}, ('qkv_gate', 128): {'kind': 'fi_mm', 'backend': 'cublaslt'}}
pj_P9_ENABLED = False

@triton.jit(do_not_specialize=['M'])
def pj__swap_kernel(w_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    m_mask = offs_m < M
    k0 = pid_k * k_per_split
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        w = tl.load(w_ptrs + kk, mask=n_mask[:, None], other=0.0, eviction_policy='evict_first')
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        if MODE == 0:
            acc = tl.dot(w, xt, acc)
        elif MODE == 1:
            hi = w.to(tl.bfloat16)
            lo = (w - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(hi, xt, acc)
            acc = tl.dot(lo, xt, acc)
        else:
            acc = tl.dot(w, xt.to(tl.float32), acc, input_precision='ieee')
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem='relaxed')
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)

@triton.jit(do_not_specialize=['M'])
def pj__gemm_kernel(x_ptr, w_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * k_per_split
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        x = tl.load(x_ptrs + kk, mask=m_mask[:, None], other=0.0)
        wt = tl.load(w_ptrs + kk, mask=n_mask[None, :], other=0.0)
        if MODE == 0:
            acc = tl.dot(x, wt, acc)
        elif MODE == 1:
            hi = wt.to(tl.bfloat16)
            lo = (wt - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(x, hi, acc)
            acc = tl.dot(x, lo, acc)
        else:
            acc = tl.dot(x.to(tl.float32), wt, acc, input_precision='ieee')
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem='relaxed')
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)

@triton.jit
def pj__fma_kernel(w_ptr, x_ptr, out_ptr, N, stride_wn, stride_xm, stride_om, M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)
    n_mask = offs_n < N
    w = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0.0, eviction_policy='evict_first').to(tl.float32)
    for j in tl.static_range(M):
        xj = tl.load(x_ptr + j * stride_xm + offs_k).to(tl.float32)
        r = tl.sum(w * xj[None, :], axis=1)
        o_ptrs = out_ptr + j * stride_om + offs_n
        if ATOMIC:
            tl.atomic_add(o_ptrs, r, mask=n_mask, sem='relaxed')
        elif OUT_BF16:
            tl.store(o_ptrs, r.to(tl.bfloat16), mask=n_mask)
        else:
            tl.store(o_ptrs, r, mask=n_mask)

def pj_run_fma(c, x, w, n):
    """_fma_kernel per config c: BN rows per program, split-K, warps."""
    m, k = x.shape
    split = c['split']
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    pj__fma_kernel[triton.cdiv(n, c['BN']), split](w, x, out, n, w.stride(0), x.stride(0), out.stride(0), M=m, BN=c['BN'], BK=k // split, ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'])
    if atomic and (not fp32_out):
        out = out.to(torch.bfloat16)
    return out

def pj_run_triton(c, x, w, n):
    """One of our kernels per config c: kind, BM, BN, BK, split, mode, warps, stages."""
    m, k = x.shape
    split = c['split']
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    kps = k // split
    bm, bn, bk = (c['BM'], c['BN'], c['BK'])
    if c['kind'] == 'swap':
        pj__swap_kernel[triton.cdiv(n, bn), split](w, x, out, m, n, kps, w.stride(0), x.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, MODE=c['mode'], ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'], num_stages=c['stages'])
    else:
        pj__gemm_kernel[triton.cdiv(m, bm), triton.cdiv(n, bn), split](x, w, out, m, n, kps, x.stride(0), w.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, MODE=c['mode'], ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'], num_stages=c['stages'])
    if atomic and (not fp32_out):
        out = out.to(torch.bfloat16)
    return out

def pj_run_library(c, x, w, n):
    """BF16 GEMMs from libraries in the judge image (FlashInfer 0.7.0.post1, DeepGEMM), each with
    its own handle or kernels; none changes torch's or cuBLAS's global settings."""
    kind = c['kind']
    if kind == 'fi_mm':
        import flashinfer
        return flashinfer.mm_bf16(x, w.T, out_dtype=torch.bfloat16, backend=c['backend'])
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    if kind == 'fi_wsk':
        from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import WarpSplitKTactic, run_warp_splitk_dense
        return run_warp_splitk_dense(x, w.T, out, False, WarpSplitKTactic(*c['tactic']))
    if kind == 'fi_direct':
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense
        return run_direct_dense(x, w.T, out, False, DirectTactic(*c['tactic']))
    if kind == 'fi_tiny':
        from flashinfer.gemm.routergemm import tinygemm_bf16
        tinygemm_bf16(x, w, out)
        return out
    import deep_gemm
    deep_gemm.use_deterministic_algorithms(c['native'])
    deep_gemm.bf16_gemm_nt(x, w, out)
    return out

def pj_run_choice(c, x, layer, n, state):
    if c['kind'] in ('swap', 'gemm'):
        return pj_run_triton(c, x, state[layer][:n], n)
    if c['kind'] == 'fma':
        return pj_run_fma(c, x, state[layer][:n], n)
    if c['kind'] == 'p9':
        return pj_run_p9(c, x, layer, n, state)
    if c['kind'] in ('fi_mm', 'fi_wsk', 'fi_direct', 'fi_tiny', 'dg'):
        return pj_run_library(c, x, state[layer][:n], n)
    if c['kind'] == 'sgl_router':
        from sglang.kernels.ops.muse_spark_v12.router_gemm import router_gemm
        return router_gemm(x.float(), state['router_t'])
    if c['kind'] == 'gemv':
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        return hopper_bf16_gemv(x, state[layer][:n])
    if c['kind'] == 'tiny':
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
        return tiny_gemm_bf16(x, state[layer][:n], max_m=c['max_m'])
    raise ValueError(f"unknown kernel kind {c['kind']}")

def pj_sglang_path(x, layer, n, state):
    """Exactly baseline.py's kernel_fn."""
    if layer != 'router':
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= pj_ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state['router'])
    acc = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
    acc += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
    return acc

def pj__warm(state):
    """Run every chosen kernel once on zeros, so compiling happens here and not in a capture."""
    for (layer, m), c in pj_CHOICE.items():
        n = 1024 if layer == 'unpooler' and m <= 8 else state[layer].shape[0]
        x = torch.zeros((m, state[layer].shape[1]), dtype=torch.bfloat16, device=state[layer].device)
        if c['kind'] == 'fi_mm':
            import flashinfer
            with flashinfer.autotune(True):
                pj_run_choice(c, x, layer, n, state)
        pj_run_choice(c, x, layer, n, state)
    torch.cuda.synchronize()

def pj_prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights['router']
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = lo.t().contiguous()
    state['router_t'] = router.t().contiguous()
    if pj_P9_ENABLED:
        pj_p9_setup(state)
    pj__warm(state)
    return state

def pj_kernel_fn(data, state):
    x, cfg = (data['x'], data['config'])
    layer, n = (cfg['layer'], cfg['n'])
    c = pj_CHOICE.get((layer, x.shape[0]))
    if c is None:
        return pj_sglang_path(x, layer, n, state)
    return pj_run_choice(c, x, layer, n, state)
import tempfile

def pj_k_perm(k):
    """Column order of the packed weights: inside every 64-wide K chunk c, packed column 16 j + col
    (MMA step j = 0..3, A column col = 0..15) holds W column 64 c + 16 t + 4 j + 2 h + p, where
    col = 8 h + 2 t + p. Then lane (g, t)'s B values for the chunk's 4 MMAs are x[64 c + 16 t ..
    64 c + 16 t + 16): two 16-byte loads, b0 of step j = 32-bit word 2 j, b1 = word 2 j + 1."""
    idx = []
    for c in range(k // 64):
        for j in range(4):
            for col in range(16):
                h, t, p = (col // 8, col % 8 // 2, col % 2)
                idx.append(64 * c + 16 * t + 4 * j + 2 * h + p)
    return torch.tensor(idx, dtype=torch.long)

def pj_p9_pack(w):
    """W [N, K] bf16 -> [N/16, K/16, 32, 8] bf16 in A-fragment order (one lane's uint4 = 8 bf16),
    columns permuted by k_perm inside each 64-wide chunk."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and (k % 64 == 0), (w.shape, w.dtype)
    w = w[:, pj_k_perm(k).to(w.device)]
    v = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)
    v = v.permute(0, 3, 2, 5, 4, 1, 6)
    return v.reshape(n // 16, k // 16, 32, 8).contiguous()

def pj_p9_split_router(w):
    """FP32 router -> (hi, lo) bf16, exactly SGLang's MuseSpark1xRouter._build_split2."""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return (hi, lo)

def pj_p9_pack_hilo(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] bf16: per k16 step the hi fragment, then the lo one."""
    hi, lo = pj_p9_split_router(w)
    return torch.stack((pj_p9_pack(hi), pj_p9_pack(lo)), dim=2).contiguous()

def pj_a_tile_from_fragments(frag):
    """Inverse view used by the checks: [32, 8] bf16 (one 16x16 tile, all lanes) -> the 16x16 A tile,
    read with the PTX formulas (row = g + 8 (j & 1), col = 2 t + 8 (j >> 1) + p)."""
    a = torch.empty(16, 16, dtype=frag.dtype)
    for lane in range(32):
        g, t = (lane >> 2, lane & 3)
        for j in range(4):
            for p in range(2):
                a[g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + p] = frag[lane, 2 * j + p]
    return a
pj_ESC = 15
pj_M32 = 4294967295

def pj_choose_top(e_tile):
    """e_tile: int64 exponents of one 16-row tile -> top of the best 15-exponent window (>= 15)."""
    hist = torch.bincount(e_tile.flatten(), minlength=256).double()
    win = torch.nn.functional.conv1d(hist.view(1, 1, -1), torch.ones(1, 1, 15, dtype=torch.float64, device=hist.device)).flatten()
    start = int(win.argmax())
    return max(start + 14, 15)

def pj_p22_encode(frag, k_perm_idx):
    """frag: p9_pack output [RT, KS, 32, 8] bf16 (K % 64 == 0). Returns dict of int32 planes, tile tops,
    and the escape list in original (unpermuted) K, plus counts."""
    rt, ks = (frag.shape[0], frag.shape[1])
    assert ks % 4 == 0
    b = frag.contiguous().view(torch.int16).to(torch.int64) & 65535
    e = b >> 7 & 255
    r = b >> 15 << 7 | b & 127
    dev = frag.device
    tops = torch.tensor([pj_choose_top(e[i]) for i in range(rt)], dtype=torch.int64, device=dev)
    c = tops.view(rt, 1, 1, 1) - e
    esc = (c < 0) | (c > 14)
    c = torch.where(esc, torch.full_like(c, pj_ESC), c)
    r = torch.where(esc, torch.zeros_like(r), r)
    q = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        q |= c[..., i] << 8 * i
        q |= c[..., 4 + i] << 8 * i + 4
    ra = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    rb = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        ra |= r[..., i] << 8 * i
        rb |= r[..., 4 + i] << 8 * i
    chunks = ks // 4
    qp = q.view(rt, chunks, 4, 32).permute(0, 1, 3, 2)
    rr = torch.stack((ra, rb), dim=-1).view(rt, chunks, 2, 2, 32, 2)
    rp = rr.permute(0, 1, 2, 4, 3, 5).reshape(rt, chunks, 2, 32, 4)
    ti, si, li, ei = esc.nonzero(as_tuple=True)
    g, t = (li >> 2, li & 3)
    reg = ei >> 1
    row = 16 * ti + g + 8 * (reg & 1)
    col = 2 * t + 8 * (reg >> 1) + (ei & 1)
    korig = k_perm_idx.to(dev)[16 * si + col]
    true_bits = b[ti, si, li, ei]
    to32 = lambda v: (v & pj_M32).to(torch.int64)
    return {'q': to32(qp), 'r': to32(rp), 'tops': tops, 'esc_row': row, 'esc_k': korig, 'esc_bits': true_bits, 'n_esc': int(esc.sum()), 'n': int(esc.numel())}

def pj_prmt(a, b, sel):
    """PTX prmt.b32 default mode for selector nibbles 0..7: byte n = byte sel_n of {b:a}."""
    src = b << 32 | a
    out = torch.zeros_like(a)
    for n in range(4):
        idx = sel >> 4 * n & 7
        out |= (src >> 8 * idx & 255) << 8 * n
    return out

def pj_p22_decode_step(q, ra, rb, top):
    """The kernel's decode of one step for tensors of words (int64 holding uint32). Returns 4 registers."""
    top4 = top * 16843009 & pj_M32
    ca, cb = (q & 252645135, q >> 4 & 252645135)
    ea, eb = (top4 - ca & pj_M32, top4 - cb & pj_M32)
    hia = ra & 2155905152 | ea >> 1 & 2139062143
    loa = ea << 7 & 2155905152 | ra & 2139062143
    hib = rb & 2155905152 | eb >> 1 & 2139062143
    lob = eb << 7 & 2155905152 | rb & 2139062143
    return (pj_prmt(loa, hia, 20800), pj_prmt(loa, hia, 29538), pj_prmt(lob, hib, 20800), pj_prmt(lob, hib, 29538))

def pj_p22_decode_all(enc):
    """Decode every fragment from the planes exactly as the kernel indexes them -> [RT, KS, 32, 8] bits."""
    q, r, tops = (enc['q'], enc['r'], enc['tops'])
    rt, chunks = (q.shape[0], q.shape[1])
    out = torch.zeros(rt, chunks * 4, 32, 8, dtype=torch.int64)
    for j in range(4):
        qw = q[:, :, :, j]
        h, w0 = (j >> 1, 2 * (j & 1))
        ra, rb = (r[:, :, h, :, w0], r[:, :, h, :, w0 + 1])
        regs = pj_p22_decode_step(qw, ra, rb, tops.view(rt, 1, 1))
        for reg, v in enumerate(regs):
            out[:, j::4, :, 2 * reg] = v & 65535
            out[:, j::4, :, 2 * reg + 1] = v >> 16 & 65535
    return out

def pj_p22_dec_bits(top):
    """Bits an escape decodes to: exponent top - 15, sign 0, mantissa 0."""
    return top - 15 << 7
pj_P22_MAX_ESCAPE_RATE = 0.005

def pj_p22_device_planes(w, n_rows):
    """BF16 W [N, K] -> kernel tensors on W's device: q [RT, K/64, 32, 4] int32, r [RT, K/64, 2, 32, 4]
    int32, tops [RT] uint8, CSR erow [N+1] int32 / ecol int32 / ebits int16 (sorted by row), or None when
    escapes exceed P22_MAX_ESCAPE_RATE (that layer then stays on uncompressed kernels)."""
    n, k = w.shape
    enc = pj_p22_encode(pj_p9_pack(w), pj_k_perm(k))
    if enc['n_esc'] > pj_P22_MAX_ESCAPE_RATE * enc['n']:
        return None
    order = torch.argsort(enc['esc_row'] * k + enc['esc_k'])
    rows, cols, bits = (enc['esc_row'][order], enc['esc_k'][order], enc['esc_bits'][order])
    erow = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    erow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    as32 = lambda t: t.to(torch.int64).where(t < 2 ** 31, t - 2 ** 32).to(torch.int32).contiguous()
    return {'q': as32(enc['q']), 'r': as32(enc['r']), 'tops': enc['tops'].to(torch.uint8).contiguous(), 'erow': erow.to(torch.int32).contiguous(), 'ecol': cols.to(torch.int32).contiguous(), 'ebits': bits.where(bits < 2 ** 15, bits - 2 ** 16).to(torch.int16).contiguous(), 'rate': enc['n_esc'] / enc['n']}
pj_P9_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace p9 {\n\n__device__ __forceinline__ void ld_frag(const uint4* p, uint4& v) {\n  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n}\n\n__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\ntemplate <int CS>\n__device__ __forceinline__ void cluster_sync() {\n  if constexpr (CS > 1) {\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n  } else {\n    __syncthreads();\n  }\n}\n\ntemplate <int CS>\n__device__ __forceinline__ uint32_t cluster_rank() {\n  if constexpr (CS > 1) {\n    uint32_t r;\n    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n    return r;\n  } else {\n    return 0u;\n  }\n}\n\n__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {\n  uint32_t remote;\n  float v;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));\n  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");\n  return v;\n}\n\n__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\n// P22 (F12) decode of one k16 step into the four A registers (proj/p9/p22_layout.py documents the\n// format; check_p22.py proves it exact on all 65,536 BF16 patterns with these same operations).\n__device__ __forceinline__ uint4 f12_decode(uint32_t q, uint32_t ra, uint32_t rb, uint32_t top4) {\n  const uint32_t ca = q & 0x0F0F0F0Fu, cb = (q >> 4) & 0x0F0F0F0Fu;\n  const uint32_t ea = top4 - ca, eb = top4 - cb;\n  const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x7F7F7F7Fu);\n  const uint32_t loa = ((ea << 7) & 0x80808080u) | (ra & 0x7F7F7F7Fu);\n  const uint32_t hib = (rb & 0x80808080u) | ((eb >> 1) & 0x7F7F7F7Fu);\n  const uint32_t lob = ((eb << 7) & 0x80808080u) | (rb & 0x7F7F7F7Fu);\n  return make_uint4(__byte_perm(loa, hia, 0x5140u), __byte_perm(loa, hia, 0x7362u),\n                    __byte_perm(lob, hib, 0x5140u), __byte_perm(lob, hib, 0x7362u));\n}\n\n// Shared memory: part[KSW][RT*16][8*NT] f32 only. K is permuted inside each 64-wide chunk (k_perm),\n// so lane (g, t)\'s B words for the chunk\'s 4 MMAs are x[row][64 c + 16 t .. + 16): two 16-B loads.\ntemplate <int NT, int RTW, int RG, int KSW, int D, int CS, bool HILO, bool OUT_F32, bool F12>\n__global__ void __launch_bounds__(RG * KSW * 32)\np9_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ qp, const uint4* __restrict__ rp,\n          const uint8_t* __restrict__ tops, const int* __restrict__ erow, const int* __restrict__ ecol,\n          const unsigned short* __restrict__ ebits, const uint4* __restrict__ x, void* __restrict__ out,\n          int m, int n_rows, int k) {\n  constexpr int RT = RG * RTW;          // 16-row tiles per CTA\n  constexpr int MT = 8 * NT;            // token rows covered\n  constexpr int AS = HILO ? 2 : 1;      // A fragments per tile per step\n  constexpr int SLOTS = F12 ? 3 : 4 * AS;   // uint4 per tile per chunk: Q + 2 R, or 4 steps x AS\n  static_assert(!(F12 && HILO), "the router stays hi + lo");\n  extern __shared__ __align__(16) unsigned char smem[];\n  float* part = reinterpret_cast<float*>(smem);\n  const int kc = k / CS;                // this CTA\'s K slice\n  const int kw = kc / KSW;              // one warp\'s K part\n  const int chunks = kw / 64;           // 64-wide chunks (4 MMA steps each)\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  const int rg = warp / KSW, kpart = warp % KSW;\n  const uint32_t rank = cluster_rank<CS>();\n  const int rowblock = blockIdx.x / CS;\n  const int c0 = (static_cast<int>(rank) * kc + kpart * kw) / 64;   // first chunk of this warp\n  const int ksteps_total = k / 16;\n\n  const int chunks_total = k / 64;\n  const uint4* wbase[RTW];\n  const uint4* qbase[RTW];\n  const uint4* rbase[RTW];\n  uint32_t top4[RTW];\n#pragma unroll\n  for (int i = 0; i < RTW; ++i) {\n    const long long tile = static_cast<long long>(rowblock) * RT + rg * RTW + i;\n    if constexpr (F12) {\n      qbase[i] = qp + (tile * chunks_total + c0) * 32 + lane;            // Q plane [tile][chunk][lane]\n      rbase[i] = rp + (tile * chunks_total + c0) * 64 + lane;            // R plane [tile][chunk][2][lane]\n      top4[i] = static_cast<uint32_t>(tops[tile]) * 0x01010101u;\n    } else {\n      wbase[i] = wp + (tile * ksteps_total + 4LL * c0) * (32 * AS) + lane;\n    }\n  }\n  const uint4* xb[NT];\n  bool bok[NT];\n#pragma unroll\n  for (int n = 0; n < NT; ++n) {\n    const int row = 8 * n + g;\n    bok[n] = row < m;\n    xb[n] = x + static_cast<long long>(bok[n] ? row : 0) * (k / 8) + (64LL * c0 + 16 * t) / 8;\n  }\n\n  uint4 ra[D][RTW][SLOTS];\n  uint4 rb[D][NT][2];\n  auto load = [&](int slot, int cc) {\n#pragma unroll\n    for (int i = 0; i < RTW; ++i) {\n      if constexpr (F12) {\n        ld_frag(qbase[i] + 32LL * cc, ra[slot][i][0]);\n        ld_frag(rbase[i] + 64LL * cc, ra[slot][i][1]);\n        ld_frag(rbase[i] + 64LL * cc + 32, ra[slot][i][2]);\n      } else {\n#pragma unroll\n        for (int j = 0; j < 4; ++j)\n#pragma unroll\n          for (int h = 0; h < AS; ++h) ld_frag(wbase[i] + (4LL * cc + j) * (32 * AS) + h * 32, ra[slot][i][j * AS + h]);\n      }\n    }\n#pragma unroll\n    for (int n = 0; n < NT; ++n) {\n      if (bok[n]) {\n        ld_x(xb[n] + 8LL * cc, rb[slot][n][0]);\n        ld_x(xb[n] + 8LL * cc + 1, rb[slot][n][1]);\n      } else {\n        rb[slot][n][0] = make_uint4(0u, 0u, 0u, 0u);\n        rb[slot][n][1] = make_uint4(0u, 0u, 0u, 0u);\n      }\n    }\n  };\n#pragma unroll\n  for (int s = 0; s < D; ++s)\n    if (s < chunks) load(s, s);\n\n  float acc[RTW][NT][4];\n#pragma unroll\n  for (int i = 0; i < RTW; ++i)\n#pragma unroll\n    for (int n = 0; n < NT; ++n)\n#pragma unroll\n      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;\n\n  for (int cc0 = 0; cc0 < chunks; cc0 += D) {\n#pragma unroll\n    for (int s = 0; s < D; ++s) {\n      const int cc = cc0 + s;\n      if (cc < chunks) {\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int n = 0; n < NT; ++n) {\n            const uint32_t b0 = word(rb[s][n][j >> 1], (2 * j) & 3);\n            const uint32_t b1 = word(rb[s][n][j >> 1], (2 * j + 1) & 3);\n#pragma unroll\n            for (int i = 0; i < RTW; ++i) {\n              if constexpr (F12) {\n                const uint4& rv = ra[s][i][1 + (j >> 1)];\n                const uint4 a = f12_decode(word(ra[s][i][0], j), word(rv, 2 * (j & 1)), word(rv, 2 * (j & 1) + 1), top4[i]);\n                mma(acc[i][n], a, b0, b1);\n              } else {\n#pragma unroll\n                for (int h = 0; h < AS; ++h) mma(acc[i][n], ra[s][i][j * AS + h], b0, b1);\n              }\n            }\n          }\n        }\n        if (cc + D < chunks) load(s, cc + D);\n      }\n    }\n  }\n\n  // C fragment: row g + 8 (r >> 1) of the 16-row tile, token 8 n + 2 t + (r & 1).\n  constexpr int ROWS = RT * 16;\n#pragma unroll\n  for (int i = 0; i < RTW; ++i)\n#pragma unroll\n    for (int n = 0; n < NT; ++n)\n#pragma unroll\n      for (int r = 0; r < 4; ++r) {\n        const int row = (rg * RTW + i) * 16 + g + 8 * (r >> 1);\n        const int tok = 8 * n + 2 * t + (r & 1);\n        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];\n      }\n  cluster_sync<CS>();\n\n  // CTA rank r reduces rows [r * ROWS/CS, (r+1) * ROWS/CS) over peers and K parts in fixed order.\n  constexpr int SHARE = ROWS / CS;\n  const uint32_t part_addr = smem_u32(part);\n  const int m_out = m < MT ? m : MT;\n  for (int idx = tid; idx < SHARE * m_out; idx += blockDim.x) {\n    const int tok = idx / SHARE;\n    const int row = static_cast<int>(rank) * SHARE + idx % SHARE;\n    float sum = 0.0f;\n#pragma unroll\n    for (int peer = 0; peer < CS; ++peer)\n#pragma unroll\n      for (int kp = 0; kp < KSW; ++kp) {\n        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;\n        if constexpr (CS > 1) {\n          sum += ld_peer_f32(part_addr + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(peer));\n        } else {\n          sum += part[off];\n        }\n      }\n    const long long orow = static_cast<long long>(rowblock) * ROWS + row;\n    if constexpr (F12) {  // escapes: + x * w_true - x * w_dec (w_dec: exponent top - 15, sign/mantissa 0)\n      if (orow < n_rows) {\n        const unsigned short* xs = reinterpret_cast<const unsigned short*>(x) + static_cast<long long>(tok) * k;\n        const float wdec = __uint_as_float((static_cast<uint32_t>(tops[orow >> 4]) - 15u) << 23);\n        for (int e = erow[orow]; e < erow[orow + 1]; ++e) {\n          const float xv = __uint_as_float(static_cast<uint32_t>(xs[ecol[e]]) << 16);\n          sum = fmaf(xv, __uint_as_float(static_cast<uint32_t>(ebits[e]) << 16), sum);\n          sum = fmaf(-xv, wdec, sum);\n        }\n      }\n    }\n    if (orow < n_rows) {\n      if constexpr (OUT_F32) {\n        reinterpret_cast<float*>(out)[static_cast<long long>(tok) * n_rows + orow] = sum;\n      } else {\n        reinterpret_cast<__nv_bfloat16*>(out)[static_cast<long long>(tok) * n_rows + orow] = __float2bfloat16_rn(sum);\n      }\n    }\n  }\n  if constexpr (CS > 1) cluster_sync<CS>();  // peers may still read this CTA\'s part[]\n}\n\n}  // namespace p9\n\n@@INSTANCES@@\n'
pj_P9_INSTANCES = ((1, 1, 2, 8, 2, 8, True, True), (2, 1, 1, 8, 2, 8, True, True))
pj_P9_CHECK_INSTANCES = ((1, 1, 8, 1, 4, 4, False, False), (4, 2, 2, 1, 1, 1, False, False), (1, 1, 1, 8, 2, 8, True, True), (2, 1, 4, 2, 2, 2, False, False), (1, 2, 4, 2, 4, 4, False, False, True), (4, 1, 8, 1, 2, 1, False, False, True))

def pj_p9_smem_bytes(nt, rtw, rg, ksw, cs, k):
    return ksw * rg * rtw * 16 * 8 * nt * 4

def pj_p9_name(inst):
    nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
    f12 = inst[8] if len(inst) > 8 else False
    return f'p9_run_{nt}_{rtw}_{rg}_{ksw}_{d}_{cs}_{int(hilo)}_{int(f32)}' + ('_f12' if f12 else '')

def pj_p9_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
        f12 = inst[8] if len(inst) > 8 else False
        tmpl = f"{nt}, {rtw}, {rg}, {ksw}, {d}, {cs}, {('true' if hilo else 'false')}, {('true' if f32 else 'false')}, {('true' if f12 else 'false')}"
        wrappers.append(f'\nvoid {pj_p9_name(inst)}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow,\n                     torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out,\n                     int64_t m, int64_t n_rows, int64_t k, int64_t smem) {{\n  cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n  const unsigned ctas = static_cast<unsigned>((n_rows / ({rg} * {rtw} * 16)) * {cs});\n  const uint4* wptr = reinterpret_cast<const uint4*>(wp.data_ptr());\n  const uint4* qptr = reinterpret_cast<const uint4*>(qp.data_ptr());\n  const uint4* rptr = reinterpret_cast<const uint4*>(rp.data_ptr());\n  const uint8_t* tptr = reinterpret_cast<const uint8_t*>(tops.data_ptr());\n  const int* erp = reinterpret_cast<const int*>(erow.data_ptr());\n  const int* ecp = reinterpret_cast<const int*>(ecol.data_ptr());\n  const unsigned short* ebp = reinterpret_cast<const unsigned short*>(ebits.data_ptr());\n  const uint4* xptr = reinterpret_cast<const uint4*>(x.data_ptr());\n  void* optr = out.data_ptr();\n  const int mi = static_cast<int>(m), ni = static_cast<int>(n_rows), ki = static_cast<int>(k);\n  if ({cs} == 1) {{\n    p9::p9_kernel<{tmpl}><<<ctas, {rg} * {ksw} * 32, static_cast<size_t>(smem), stream>>>(wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n  }} else {{\n    cudaLaunchConfig_t cfg = {{}};\n    cudaLaunchAttribute attr[1];\n    attr[0].id = cudaLaunchAttributeClusterDimension;\n    attr[0].val.clusterDim.x = {cs};\n    attr[0].val.clusterDim.y = 1;\n    attr[0].val.clusterDim.z = 1;\n    cfg.gridDim = dim3(ctas, 1, 1);\n    cfg.blockDim = dim3({rg} * {ksw} * 32, 1, 1);\n    cfg.dynamicSmemBytes = static_cast<size_t>(smem);\n    cfg.stream = stream;\n    cfg.attrs = attr;\n    cfg.numAttrs = 1;\n    const cudaError_t err = cudaLaunchKernelEx(&cfg, p9::p9_kernel<{tmpl}>, wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);\n    TORCH_CHECK(err == cudaSuccess, "p9 cluster launch failed: ", cudaGetErrorString(err));\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n  }}\n}}\nvoid {pj_p9_name(inst)}_attr(int64_t smem) {{\n  cudaFuncSetAttribute(p9::p9_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));\n}}\nint64_t {pj_p9_name(inst)}_info() {{\n  cudaFuncAttributes a;\n  cudaFuncGetAttributes(&a, p9::p9_kernel<{tmpl}>);\n  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);\n}}')
    return pj_P9_SRC.replace('@@INSTANCES@@', '\n'.join(wrappers))

def pj_p9_decls(instances):
    out = []
    for inst in instances:
        name = pj_p9_name(inst)
        out.append(f'void {name}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow, torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows, int64_t k, int64_t smem);')
        out.append(f'void {name}_attr(int64_t smem);')
        out.append(f'int64_t {name}_info();')
    return out

def pj_p9_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = pj_p9_decls(instances)
    return load_inline(name='okc_proj_p9', cpp_sources='\n'.join(decls), cuda_sources=pj_p9_source(instances), functions=[d.split('(')[0].split()[-1] for d in decls], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'], build_directory=tempfile.mkdtemp(prefix='okc_proj_p9_'), verbose=False)
pj_P9_ENABLED = True
pj_P9_MOD = None
pj_P9_BF16_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
pj_P9_MAX_SMEM = 227 * 1024

def pj_p9_setup(state):
    """Pack the weights into fragment order (lossless, same bytes; router as SGLang's hi + lo), build
    every P9 instance, allow the full shared memory, and launch each instance once on zeros so no
    compile or module load lands inside a capture."""
    global pj_P9_MOD
    state['p9'] = {name: pj_p9_pack(state[name]) for name in pj_P9_BF16_LAYERS}
    state['p9']['router'] = pj_p9_pack_hilo(state['router'])
    dev = state['router'].device
    state['p9_dummy'] = torch.zeros(16, dtype=torch.int32, device=dev)
    state['p22'] = {}
    if any((len(inst) > 8 and inst[8] for inst in pj_P9_INSTANCES)):
        for name in pj_P9_BF16_LAYERS:
            planes = pj_p22_device_planes(state[name], state[name].shape[0])
            if planes is not None:
                state['p22'][name] = planes
    pj_P9_MOD = pj_p9_build(pj_P9_INSTANCES)
    for inst in pj_P9_INSTANCES:
        getattr(pj_P9_MOD, pj_p9_name(inst) + '_attr')(pj_P9_MAX_SMEM)

def pj_run_p9(c, x, layer, n, state):
    """c: dict(kind="p9", NT, RTW, RG, KSW, D, CS[, F12])."""
    hilo = layer == 'router'
    f12 = bool(c.get('F12', False))
    inst = (c['NT'], c['RTW'], c['RG'], c['KSW'], c['D'], c['CS'], hilo, hilo) + ((True,) if f12 else ())
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32 if hilo else torch.bfloat16, device=x.device)
    smem = pj_p9_smem_bytes(c['NT'], c['RTW'], c['RG'], c['KSW'], c['CS'], k)
    dmy = state['p9_dummy']
    if f12:
        pl = state['p22'][layer]
        args = (dmy, pl['q'], pl['r'], pl['tops'], pl['erow'], pl['ecol'], pl['ebits'])
    else:
        args = (state['p9'][layer], dmy, dmy, dmy, dmy, dmy, dmy)
    getattr(pj_P9_MOD, pj_p9_name(inst))(*args, x.contiguous(), out, m, n, k, smem)
    return out


# ====================================================================== parts/v6.py

# v6 = v5 (v4 + M11 with L8's next-task L2 prefetch, parts/m11_pf.py) with the siblings' newer judged files,
# each still switched on per case (tokens, sliding) only where a lab measured a gain:
#   att: the attention session's scored v5 (parts/attn5.py, 1.406x on muse_attention_decode): D3 on full
#        decode 16 / 32 (119 / 207 us vs v3's 156 / 300) and a PDL chain on six classes. In our graph its
#        first kernel (reads our qkv) is a plain launch; its PDL kernels wait first thing (griddepcontrol.wait),
#        and our next kernel is a plain launch, so the chain is safe after and before our kernels.
#   pj:  the bf16_proj session's scored v4 (parts/proj4.py, 1.247x on muse_bf16_proj): library winners at
#        8 / 128 tokens and the CUDA P9 router at 16; (router, 8) stays on v2's FMA kernel (below), and the
#        unpooler slice at <= 8 tokens goes through its CHOICE too when pj is on.
# The rest of v4 (parts/v4.py) is unchanged: same call sites, same self-checks plus the new routes.

V4_ATT_KEYS = ("positions", "cache_loc", "cache_seqlens", "cu_seqlens_q", "scheduler_metadata", "k_cache",
               "v_cache", "k_scale", "v_scale", "page_table", "seq_lens", "config")
# (tokens, sliding) -> {"att": bool, "pj": bool}. Until a v6 lab picks: DE6's picks for v4. Missing: both off.
V4_MODE = {(1, False): {'att': True, 'pj': True}, (1, True): {'att': True, 'pj': True},
           (8, False): {'att': True, 'pj': True}, (8, True): {'att': True, 'pj': True},
           (16, False): {'att': True, 'pj': True}, (16, True): {'att': True, 'pj': True},
           (32, False): {'att': True, 'pj': True}, (32, True): {'att': True, 'pj': True},
           (128, False): {'att': False, 'pj': False}, (128, True): {'att': True, 'pj': True},
           (256, False): {'att': False, 'pj': False}, (256, True): {'att': False, 'pj': False}}
V3_BUILD_PREFETCH = False  # DF21: the L2 prefetch extension isn't used; skip its nvcc time

# At <= 8 tokens SGLang computes the router from the FP32 router input; bf16_proj v4 moved (router, 8) to P9,
# which takes BF16 x only (its problem's input). Keep v2's FMA kernel there (the same _fma_kernel source in
# v2 and v4, judged in v4 / DE6), so the router keeps SGLang's FP32 input (rule 6).
pj_CHOICE[("router", 8)] = {'kind': 'fma', 'BN': 16, 'split': 64, 'warps': 8}


def v4_mode(cfg, state, key):
    return state["v4_mode"].get((cfg["tokens"], bool(cfg["window"])), {}).get(key, False)


def v4_project(x, layer, n, cfg, state):
    """A dense projection: the bf16_proj session's chosen kernel for (layer, tokens) when on, else cuBLAS."""
    if v4_mode(cfg, state, "pj") and "pj_state" in state:
        c = pj_CHOICE.get((layer, x.shape[0]))
        if c is not None:
            return pj_run_choice(c, x, layer, n, state["pj_state"])
    w = state["dense"]["qkv_gate" if layer == "qkv_gate" else layer]
    return _gemm(x, w if n == w.shape[0] else w[:n])


def v6_unpool_slice(expert_output, cfg, state):
    """This GPU's 1024 unpooler columns at <= 8 tokens: bf16_proj's (unpooler, m) choice when pj is on
    (m 8: FlashInfer tinygemm), else SGLang's tiny_gemm as v3."""
    if v4_mode(cfg, state, "pj") and "pj_state" in state:
        c = pj_CHOICE.get(("unpooler", expert_output.shape[0]))
        if c is not None:
            return pj_run_choice(c, expert_output, "unpooler", UNPOOL_SLICE, state["pj_state"])
    return v3_unpool_slice(expert_output, state)


def v4_router(ffn_input, router_input, cfg, state):
    """FP32 router logits: bf16_proj's kernel for (router, tokens) when on (fed the FP32 router input
    up to 8 tokens, as SGLang computes it there; the BF16 input with a hi + lo split above), else
    SGLang's own path."""
    tokens = ffn_input.shape[0]
    split = tokens > ROUTER_FP32_MAX_TOKENS
    if v4_mode(cfg, state, "pj") and "pj_state" in state:
        c = pj_CHOICE.get(("router", tokens))
        if c is not None:
            return pj_run_choice(c, ffn_input if split else router_input, "router", N_EXPERTS, state["pj_state"])
    if split:
        logits = torch.mm(ffn_input, state["router_hi_t"], out_dtype=torch.float32)
        logits += torch.mm(ffn_input, state["router_lo_t"], out_dtype=torch.float32)
        return logits
    return F.linear(router_input, state["router"])


def v4_attention(data, qkv, cfg, state):
    """The attention segment: the attention session's kernels when on, else SGLang's (D1's) block."""
    if v4_mode(cfg, state, "att") and state.get("v4_att"):
        adata = {key: data[key] for key in V4_ATT_KEYS}
        adata["qkv"] = qkv.view(cfg["batch"], cfg["q_len"], -1)
        return att_kernel_fn(adata).reshape(cfg["tokens"], -1)
    return _v4_attention_d1(data, qkv, state)


def _v4_check(name, fn):
    try:
        ok, msg = fn()
    except Exception as exc:
        ok, msg = False, f"{type(exc).__name__}: {str(exc)[-300:]}"
    print(f"[v6] {name}: {'on' if ok else 'OFF'} ({msg})")
    return ok


def _v4_small_case(batch, q_len, sliding, device, gen, ctx=3000):
    """A small decode-layer input (page 1, kv scale 1) for the self-checks."""
    tokens = batch * q_len
    seq = torch.full((batch,), ctx, dtype=torch.int64, device=device)
    max_pages = ctx + q_len
    total = batch * max_pages
    table = torch.randperm(total, generator=gen, device=device).to(torch.int32).view(batch, max_pages)
    pos = (seq[:, None] + torch.arange(q_len, device=device)).reshape(-1)
    loc = torch.gather(table.long(), 1, pos.view(batch, q_len)).reshape(-1)
    kc = torch.randn((total, 1, 2, 64), generator=gen, device=device).to(FP8)
    vc = torch.randn((total, 1, 2, 64), generator=gen, device=device).to(FP8)
    cfg = dict(batch=batch, q_len=q_len, tokens=tokens, hidden=HIDDEN, num_q_heads=16, num_kv_heads=2, head_dim=64,
               page_size=1, window=2048 if sliding else 0, rope_theta=ROPE_THETA if sliding else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0, n_experts=N_EXPERTS, top_k=TOP_K,
               local_experts=LOCAL_EXPERTS)
    cs = (seq + q_len).to(torch.int32)
    cu = torch.arange(0, tokens + 1, q_len, dtype=torch.int32, device=device)
    return {"positions": pos, "cache_loc": loc, "cache_seqlens": cs, "cu_seqlens_q": cu, "scheduler_metadata": None,
            "k_cache": kc, "v_cache": vc, "k_scale": torch.tensor(1.0, device=device),
            "v_scale": torch.tensor(1.0, device=device), "page_table": table, "seq_lens": seq.to(torch.int32),
            "config": cfg}


# The attention self-check's cases: v4's four plus every v5 route class (D3 full decode 16 / 32; the PDL
# chain on sliding decode 16 / 32 and sliding verify 1 / 16 / 32; full decode 1).
V6_ATT_CASES = ((1, 1, True), (2, 8, False), (4, 1, False), (3, 8, True), (16, 1, False), (32, 1, False),
                (16, 1, True), (32, 1, True), (1, 8, True), (16, 8, True), (32, 8, True), (1, 1, False))


def _glue_prepare(weights):
    state = _v3_prepare(weights)
    state["v3_router"] = False  # DF22: v4 routes the router through v4_router
    state["v4_mode"] = {key: dict(val) for key, val in V4_MODE.items()}
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(41)

    def pj():
        st = {"qkv_gate": state["dense"]["qkv_gate"], "o": state["dense"]["o"], "pooler": state["dense"]["pooler"],
              "unpooler": state["dense"]["unpooler"], "router": state["router"],
              "router_hi_t": state["router_hi_t"], "router_lo_t": state["router_lo_t"]}
        if any(c["kind"] == "sgl_router" for c in pj_CHOICE.values()):
            st["router_t"] = state["router"].t().contiguous()
        if pj_P9_ENABLED:
            pj_p9_setup(st)  # packs the weights in fragment order (lossless) and builds the P9 kernels
        pj__warm(st)  # compiles each CHOICE with BF16 x, as bf16_proj's problem passes it
        # The loop below also compiles the router's <= 8-token kernels with the FP32 x v6 passes them (a
        # different Triton specialization), so no first compile can land inside a graph capture.
        worst = 0.0
        for (layer, m), c in pj_CHOICE.items():
            k = st[layer].shape[1]
            n = UNPOOL_SLICE if layer == "unpooler" and m <= SLICED_UNPOOL_MAX_TOKENS else st[layer].shape[0]
            if layer == "router" and m <= ROUTER_FP32_MAX_TOKENS:
                assert c["kind"] == "fma", "the router at <= 8 tokens must take the FP32 input"
                x = torch.randn((m, k), generator=gen, device=device)  # our FP32 router input
                ref = F.linear(x, st["router"])
            elif layer == "router":
                x = torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16)
                ref = pj_sglang_path(x, layer, n, st)
            else:
                x = torch.randn((m, k), generator=gen, device=device).to(torch.bfloat16)
                ref = F.linear(x, st[layer][:n])
            worst = max(worst, _v3_rel(pj_run_choice(c, x, layer, n, st), ref))
        state["pj_state"] = st
        return worst < 1e-2, f"{len(pj_CHOICE)} (layer, m) kernels warmed, worst rel err vs SGLang {worst:.1e}"

    if not _v4_check("projections (bf16_proj v4)", pj):
        state.pop("pj_state", None)

    def att():
        att__warm_up()
        worst, cache_bad, cache_n = 0.0, 0, 0
        for batch, q_len, sliding in V6_ATT_CASES:
            data = _v4_small_case(batch, q_len, sliding, device, gen)
            tokens = batch * q_len
            qkv = (torch.randn((tokens, 2304), generator=gen, device=device) * 0.5).to(torch.bfloat16)
            k2, v2 = data["k_cache"].clone(), data["v_cache"].clone()
            mine = v4_attention_forced(data, qkv, state)
            ref_data = dict(data, k_cache=k2, v_cache=v2)
            ref = _v4_attention_d1(ref_data, qkv, state)
            worst = max(worst, _v3_rel(mine, ref))
            loc = data["cache_loc"]
            for a, b in ((data["k_cache"], k2), (data["v_cache"], v2)):
                diff = (a.view(-1, 128)[loc].view(torch.uint8).int() - b.view(-1, 128)[loc].view(torch.uint8).int()).abs()
                cache_bad += int((diff > 1).sum())
                cache_n += diff.numel()
            del data, ref_data, k2, v2
        torch.cuda.synchronize()
        state["v4_att"] = worst < 0.2 and cache_bad <= cache_n // 1000
        return state["v4_att"], (f"{len(V6_ATT_CASES)} cases, rel err vs FA3 path {worst:.1e} (FA3 itself is ~0.08 "
                                 f"off the FP32 reference); new cache bytes differing by > 1 ulp: {cache_bad} of {cache_n}")

    _v4_check("attention (attention v5)", att)
    torch.cuda.synchronize()
    return state


def v4_attention_forced(data, qkv, state):
    """The attention session's path regardless of v4_mode (self-checks and labs)."""
    cfg = data["config"]
    adata = {key: data[key] for key in V4_ATT_KEYS}
    adata["qkv"] = qkv.view(cfg["batch"], cfg["q_len"], -1)
    return att_kernel_fn(adata).reshape(cfg["tokens"], -1)


# ====================================================================== parts/l9.py

# L9: L2 staging of upcoming weights in the windows where HBM is idle, on the side stream (prefetch hints only:
# no data changes, nothing is cached across calls). Spread over the whole grid (DF25: v3's X7 queued every
# request from 2 SMs). Per case (tokens, sliding) the lab picks:
#   o      the o projection's 16.8 MB during attention (fork right after the qkv projection);
#   e13/e2 right after routing, while the pooler streams: the first e/16 of every M11 warp slice (codes and
#          scales) of every local expert the routing hit, GEMM1's (w13) and/or GEMM2's (w2), with the slice
#          geometry of the M11 config the call will use (the same fraction for every task, so no task waits
#          on DRAM more than another);
#   slice  the unpooler's 1024-row slice (8.4 MB) during the expert stage (<= 8 tokens);
#   pm     the prefetch form: 0 cp.async.bulk.prefetch.L2 (one request per 4 KB / slice), 1 prefetch.global.L2
#          per 128-B line.

_L9_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

namespace l9 {
__device__ __forceinline__ void pf(const void* p, uint32_t bytes, int mode) {
  if (mode == 0) {
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(p), "r"(bytes));
  } else {
    const char* c = static_cast<const char*>(p);
    for (uint32_t o = 0; o < bytes; o += 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(c + o));
  }
}

// [p0, p0 + n0) then [p1, p1 + n1) in 4 KB requests (the tail rounded down to 16 B), grid-strided.
__global__ void ranges_kernel(const uint8_t* p0, long long n0, const uint8_t* p1, long long n1, int mode) {
  constexpr long long CH = 4096;
  const long long c0 = (n0 + CH - 1) / CH, c1 = (n1 + CH - 1) / CH;
  const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long c = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; c < c0 + c1; c += step) {
    const bool first = c < c0;
    const uint8_t* base = first ? p0 : p1;
    const long long off = (first ? c : c - c0) * CH, n = first ? n0 : n1;
    const uint32_t bytes = static_cast<uint32_t>((n - off < CH ? n - off : CH) & ~15LL);
    if (bytes) pf(base + off, bytes, mode);
  }
}

// M11's layout: w [32][rt_total][64 chunks][32 lanes] uint4, sf the same index in uint16. A task's warp q of ks
// reads chunks [q * kgw, (q + 1) * kgw), kgw = 64 / ks: 512 code bytes + 64 scale bytes per chunk, contiguous.
// For every local expert the routing hit ((id - local_start) mod 256 < 32), every row tile, every warp slice:
// the slice's first f16 / 16.
__global__ void experts_kernel(const int* ids, int n_ids, const int* local_start, const uint4* w,
                               const uint16_t* sf, int rt_total, int ks, int f16, int mode) {
  __shared__ unsigned s_mask;
  __shared__ int s_list[32];
  __shared__ int s_n;
  if (threadIdx.x == 0) s_mask = 0u;
  __syncthreads();
  const int ls = local_start[0];
  for (int i = threadIdx.x; i < n_ids; i += blockDim.x) {
    const int e = ((ids[i] - ls) % 256 + 256) % 256;
    if (e < 32) atomicOr(&s_mask, 1u << e);
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int n = 0;
    for (int e = 0; e < 32; ++e)
      if ((s_mask >> e) & 1u) s_list[n++] = e;
    s_n = n;
  }
  __syncthreads();
  const int kgw = 64 / ks;
  const long long items = static_cast<long long>(s_n) * rt_total * ks;
  const uint32_t wbytes = static_cast<uint32_t>(kgw * 32 * f16), sbytes = static_cast<uint32_t>(kgw * 4 * f16);
  const long long step = static_cast<long long>(gridDim.x) * blockDim.x;
  for (long long it = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; it < items; it += step) {
    const int q = static_cast<int>(it % ks);
    const long long tr = it / ks;
    const int r = static_cast<int>(tr % rt_total);
    const int e = s_list[tr / rt_total];
    const size_t base = (static_cast<size_t>(e) * rt_total + r) * 64 * 32 + static_cast<size_t>(q) * kgw * 32;
    pf(w + base, wbytes, mode);
    pf(sf + base, sbytes, mode);
  }
}
}  // namespace l9

void l9_ranges(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb, int64_t mode, int64_t blocks) {
  TORCH_CHECK(na >= 0 && na <= a.numel() * a.element_size() && nb >= 0 && nb <= b.numel() * b.element_size());
  l9::ranges_kernel<<<static_cast<int>(blocks), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const uint8_t*>(a.data_ptr()), na, reinterpret_cast<const uint8_t*>(b.data_ptr()), nb,
      static_cast<int>(mode));
}

void l9_experts(torch::Tensor ids, torch::Tensor local_start, torch::Tensor w, torch::Tensor sf, int64_t rt_total,
                int64_t ks, int64_t f16, int64_t mode, int64_t blocks) {
  TORCH_CHECK(ids.scalar_type() == at::kInt && ids.is_contiguous() && local_start.scalar_type() == at::kInt);
  TORCH_CHECK((ks == 1 || ks == 2 || ks == 4) && f16 >= 1 && f16 <= 16);
  TORCH_CHECK(w.numel() * w.element_size() == 32 * rt_total * 64 * 32 * 16 && sf.numel() * sf.element_size() == 32 * rt_total * 64 * 32 * 2);
  l9::experts_kernel<<<static_cast<int>(blocks), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
      ids.data_ptr<int>(), static_cast<int>(ids.numel()), local_start.data_ptr<int>(),
      reinterpret_cast<const uint4*>(w.data_ptr()), reinterpret_cast<const uint16_t*>(sf.data_ptr()),
      static_cast<int>(rt_total), static_cast<int>(ks), static_cast<int>(f16), static_cast<int>(mode));
}
"""

_L9_CPP = """
void l9_ranges(torch::Tensor a, int64_t na, torch::Tensor b, int64_t nb, int64_t mode, int64_t blocks);
void l9_experts(torch::Tensor ids, torch::Tensor local_start, torch::Tensor w, torch::Tensor sf, int64_t rt_total,
                int64_t ks, int64_t f16, int64_t mode, int64_t blocks);
"""

# (tokens, sliding) -> {"o": 0/1, "e13": 0..16, "e2": 0..16, "slice": 0/1, "pm": 0/1}; filled by a lab. Missing: off.
L9_MODE = {}


def l9_source():
    return _L9_CUDA


def l9_mode(cfg, state):
    return state.get("l9_mode", {}).get((cfg["tokens"], bool(cfg["window"])), {}) if "l9" in state else {}


def l9_fork_attention(cfg, state):
    """After the qkv projection: stage o on the side stream (joined with the router's side work)."""
    m = l9_mode(cfg, state)
    if not m.get("o"):
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["o"]
        state["l9"].l9_ranges(w, w.numel() * w.element_size(), w, 0, m.get("pm", 0), state["l9_blocks"])


def l9_stage_experts(topk_ids, local_start, cfg, state):
    """On the router's side stream right after routing (the pooler streams meanwhile): stage a fraction of
    the routed local experts' M11 weights, with the slice geometry of this call's M11 config."""
    m = l9_mode(cfg, state)
    if not (m.get("e13") or m.get("e2")) or "m11" not in state:
        return
    mcfg = m11_config_for(state, cfg["tokens"])
    if mcfg is None:  # Marlin serves this call (another layout): nothing to stage
        return
    bank, ids = state["m11"], topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
    for key, w, sf, rt, ks in (("e13", bank["w13"], bank["w13_s"], bank["rt_w13"], mcfg["s1"]["ks"]),
                               ("e2", bank["w2"], bank["w2_s"], bank["rt_w2"], mcfg["s2"]["ks"])):
        if m.get(key):
            state["l9"].l9_experts(ids.contiguous(), local_start, w, sf, rt, ks, m[key], m.get("pm", 0),
                                   state["l9_blocks"])


def l9_fork_slice(cfg, state):
    """Before the expert stage at <= 8 tokens: stage the unpooler slice on the side stream."""
    m = l9_mode(cfg, state)
    if not m.get("slice") or cfg["tokens"] > SLICED_UNPOOL_MAX_TOKENS:
        return
    main, side = torch.cuda.current_stream(), state["side_stream"]
    side.wait_stream(main)
    with torch.cuda.stream(side):
        w = state["dense"]["unpooler"]
        state["l9"].l9_ranges(w, UNPOOL_SLICE * w.shape[1] * w.element_size(), w, 0, m.get("pm", 0), state["l9_blocks"])


def l9_join_slice(cfg, state):
    m = l9_mode(cfg, state)
    if m.get("slice") and cfg["tokens"] <= SLICED_UNPOOL_MAX_TOKENS:
        torch.cuda.current_stream().wait_stream(state["side_stream"])


def l9_setup(state, weights):
    """Build the staging kernels and launch each once (no first launch inside a capture); any failure
    leaves L9 off."""
    try:
        state["l9"] = load_inline(
            name="l9_ext", cpp_sources=_L9_CPP, cuda_sources=_L9_CUDA, functions=["l9_ranges", "l9_experts"],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=tempfile.mkdtemp(prefix="l9_"), verbose=False)
        state["l9_blocks"] = 2 * torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
        state["l9_mode"] = {key: dict(val) for key, val in L9_MODE.items()}
        w = state["dense"]["o"]
        for pm in (0, 1):
            state["l9"].l9_ranges(w, 1 << 20, w, 0, pm, state["l9_blocks"])
            if "m11" in state:
                ids = torch.zeros((8, TOP_K), dtype=torch.int32, device=w.device)
                bank = state["m11"]
                state["l9"].l9_experts(ids, torch.zeros(1, dtype=torch.int32, device=w.device), bank["w13"],
                                       bank["w13_s"], bank["rt_w13"], 4, 1, pm, state["l9_blocks"])
        torch.cuda.synchronize()
        print(f"[l9] staging kernels: on ({len(state['l9_mode'])} cases)")
    except Exception as exc:
        state.pop("l9", None)
        print(f"[l9] staging kernels: OFF ({type(exc).__name__}: {str(exc)[-200:]})")


# ====================================================================== prepare: the glue's, then each extra section's setup (local/make_v4.py)

def _submission_prepare(weights):
    state = _glue_prepare(weights)
    l9_setup(state, weights)
    return state


# ====================================================================== D1's attention block (local/make_v4.py)

def _v4_attention_d1(data, qkv, state):
    """D1's (SGLang's) attention block, extracted verbatim by local/make_v4.py."""
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
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
    return o


# ====================================================================== v4 kernel_fn: D1's with call sites swapped (local/make_v4.py)

def kernel_fn(data, state):
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    dense = state["dense"]
    # Attention: MuseSpark1xAttention.forward.
    qkv = v4_project(data["attn_input"], "qkv_gate", dense["qkv_gate"].shape[0], cfg, state)
    l9_fork_attention(cfg, state)
    o = v4_attention(data, qkv, cfg, state)
    attn_output = v4_project(o, "o", dense["o"].shape[0], cfg, state)  # all-reduce: identity

    # Post-attention boundary, router and MoE block: MuseSpark1xMoE._forward_experts.
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    ffn_input, stream, router_input = state["attn_boundary"](
        data["stream"], attn_output, branch_dtype=torch.bfloat16, router_dtype=None if split_router else torch.float32)
    # The router and routing run on a side stream, overlapping the pooler on the main one.
    main_stream, side_stream = torch.cuda.current_stream(), state["side_stream"]
    side_stream.wait_stream(main_stream)
    expert_input = v4_project(ffn_input, "pooler", dense["pooler"].shape[0], cfg, state)
    with torch.cuda.stream(side_stream):
        logits = v4_router(ffn_input, router_input, cfg, state)
        topk_weights, topk_ids = state["routing"](None, logits, TOP_K, True)
        l9_stage_experts(topk_ids, data["local_start"], cfg, state)
    main_stream.wait_stream(side_stream)
    if not torch.cuda.is_current_stream_capturing():
        for tensor in (topk_weights, topk_ids):
            tensor.record_stream(main_stream)
    # This GPU's experts start at local_start: as SGLang's expert_map on that EP rank.
    local_ids = v3_local_ids(topk_ids, data["local_start"], state)
    # all-reduce: identity
    l9_fork_slice(cfg, state)
    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)
    l9_join_slice(cfg, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = v6_unpool_slice(expert_output, cfg, state)
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = v4_project(expert_output, "unpooler", dense["unpooler"].shape[0], cfg, state)

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


# ====================================================================== lab (de6): check runs only

"""DE6 lab (check runs only), on top of submission_v4. It CHOOSES v4's mode per case and installs it before the
judge times kernel_fn, so this check's timing is v4 as chosen.

modes   per scored shape (judge-like inputs, busiest-GPU local_start): the whole layer as a cold-L2 graph with
        the attention drop-in and the projections drop-in each off/on (4 variants); errors of the attention
        drop-in vs SGLang's block and of the full layer (all on) vs all off. A case keeps the fastest variant if it
        is >= 2% faster than all off and its full-layer difference stays < 0.3 (blunder gate); prints V4_MODE.
"""

LAB_PREPARE_LIMIT_S = 250.0
LAB_FAILS = []
LAB_SUMMARY = []
DE6_VARIANTS = ((False, False), (True, False), (False, True), (True, True))


def lab_fail(where, exc):
    LAB_FAILS.append(f"{where}: {type(exc).__name__}: {_error_text(exc, 200)}")


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def sec_modes(lab, state, deadline):
    fixed = lab.empty_graph_us()
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        key = (batch * q_len, layer == "sliding")
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"modes stopped before {name}")
            state["v4_mode"][key] = {"att": False, "pj": False}
            continue
        try:
            data = lab_input(batch, q_len, layer, seed=13000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))
            outs, times = {}, {}
            for att, pj in DE6_VARIANTS:
                state["v4_mode"][key] = {"att": att, "pj": pj}
                kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                outs[(att, pj)] = kernel_fn(dict(data, k_cache=kc, v_cache=vc), state)
                del kc, vc
                times[(att, pj)] = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
            base = outs[(False, False)]
            errs = {v: _v3_rel(outs[v][0], base[0]) for v in DE6_VARIANTS}  # next layer's input, vs all off
            routed = {v: float((outs[v][2] == base[2]).float().mean()) for v in DE6_VARIANTS}
            # Blunder gate only (wiring errors are O(1)); the judge's tests vs FP32 are the real check. FA3's FP8
            # queries alone sit ~8% off FP32 in attention, and a flipped near-tie expert changes a token's FFN.
            ok = [v for v in DE6_VARIANTS if errs[v] < 0.3]
            best = min(ok, key=lambda v: times[v])
            pick = best if times[best] < 0.98 * times[(False, False)] else (False, False)
            state["v4_mode"][key] = {"att": pick[0], "pj": pick[1]}
            cells.append(f"{name} " + "/".join(f"{times[v]:.1f}" for v in DE6_VARIANTS)
                         + f" err {'/'.join(f'{errs[v]:.0e}' for v in DE6_VARIANTS[1:])}"
                         + f" ids {min(routed.values()):.2f} -> att{int(pick[0])} pj{int(pick[1])}")
            LAB_SUMMARY.append(f"{name} {times[(False, False)]:.1f}->{times[pick]:.1f}")
            del data, outs
        except Exception as exc:
            lab_fail(f"modes {name}", exc)
            state["v4_mode"][key] = {"att": False, "pj": False}
        torch.cuda.empty_cache()
    log("modes (us off/att/pj/both; err att/pj/both vs off; routing agreement): " + "; ".join(cells))
    log(f"V4_MODE = {dict(sorted(state['v4_mode'].items()))!r}")


def _de6_prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v4 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, "
        f"att {'on' if state.get('v4_att') else 'OFF'}")
    lab.section("modes", sec_modes, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state


# ====================================================================== lab (de9): check runs only

"""DE9 lab (check runs only), on top of submission_v7 (= v6 + parts/l9.py, L2 staging in HBM-idle windows),
after lab/de6.py's sections (appended before this file): it CHOOSES v7's attention / projection modes per case
(de6's sec_modes), then L9's staging per case, and installs both before the judge times kernel_fn.

stage   per scored shape (judge-like inputs, busiest-GPU local_start), the whole layer as a cold-L2 graph,
        greedy per prefetch form pm (0 bulk, 1 lines): o, then e13 in (4, 8, 16), then e2 in (8, 16), then
        the unpooler slice (<= 8 tokens); each option's best level is kept if the layer gets >= 1% faster; the
        case keeps its best only if >= 2% faster than no staging, and only if the layer's outputs equal
        no staging's bit for bit (staging is hints only). Prints L9_MODE.
"""

DE9_STEPS = (("o", (1,)), ("e13", (4, 8, 16)), ("e2", (8, 16)), ("slice", (1,)))


def sec_stage(lab, state, deadline):
    if "l9" not in state:
        LAB_SUMMARY.append("stage: L9 off")
        return
    fixed = lab.empty_graph_us()
    cells = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        key = (batch * q_len, layer == "sliding")
        if lab.elapsed() > deadline:
            LAB_SUMMARY.append(f"stage stopped before {name}")
            state["l9_mode"][key] = {}
            continue
        try:
            data = lab_input(batch, q_len, layer, seed=21000 + i)
            _, _, ids = kernel_fn(data, state)
            data["local_start"].fill_(_busiest_start(ids))

            def timed(mode):
                state["l9_mode"][key] = mode
                return lab.time_graph(lambda: kernel_fn(data, state)) - fixed

            base = timed({})
            best_mode, best = {}, base
            for pm in (0, 1):
                cur, cur_t = {"pm": pm}, base
                for opt, levels in DE9_STEPS:
                    if opt == "slice" and batch * q_len > SLICED_UNPOOL_MAX_TOKENS:
                        continue
                    tries = [(timed(dict(cur, **{opt: v})), v) for v in levels]
                    t, v = min(tries)
                    if t < cur_t - 0.01 * abs(cur_t):
                        cur, cur_t = dict(cur, **{opt: v}), t
                if cur_t < best:
                    best_mode, best = cur, cur_t
            pick = best_mode if best < base - 0.02 * abs(base) else {}
            if pick:  # staging changes no data: the layer's outputs must match no staging's exactly
                outs = []
                for mode in ({}, pick):
                    state["l9_mode"][key] = mode
                    kc, vc = data["k_cache"].clone(), data["v_cache"].clone()
                    outs.append(kernel_fn(dict(data, k_cache=kc, v_cache=vc), state))
                    del kc, vc
                same = all(torch.equal(a, b) for a, b in zip(*outs))
                if not same:
                    LAB_SUMMARY.append(f"{name}: staged outputs differ, dropped")
                    pick = {}
            state["l9_mode"][key] = pick
            cells.append(f"{name} {base:.1f}->{best:.1f} {pick if pick else '-'}")
            LAB_SUMMARY.append(f"{name} {base:.1f}->{(best if pick else base):.1f}")
            del data
        except Exception as exc:
            lab_fail(f"stage {name}", exc)
            state["l9_mode"][key] = {}
        torch.cuda.empty_cache()
    log("stage (us none -> best; pick): " + "; ".join(cells))
    log(f"L9_MODE = {dict(sorted((k, v) for k, v in state['l9_mode'].items() if v))!r}")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"v7 prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s; pj {'on' if 'pj_state' in state else 'OFF'}, "
        f"att {'on' if state.get('v4_att') else 'OFF'}, l9 {'on' if 'l9' in state else 'OFF'}")
    lab.section("modes", sec_modes, state, until=0.4 * lab.budget_s)
    lab.section("stage", sec_stage, state)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"failures {len(LAB_FAILS)}" + ("" if not LAB_FAILS else ": " + " || ".join(LAB_FAILS[:6])))
    log("SUMMARY " + "; ".join(LAB_SUMMARY))
    log(f"lab total {lab.elapsed():.1f}s")
    return state
