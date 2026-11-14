# okc: problem=muse_decode_layer
"""D1: SGLang's decoder layer (baseline.py, unchanged otherwise) with the two expert-path changes
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


def _submission_prepare(weights):
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
    expert_output = _experts(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
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


# ====================================================================== lab (d0): check runs only

"""D0 recon lab (check runs only). prepare() runs the submission's prepare, then within a budget:

ops    every scored shape on judge-like inputs: local routes / distinct experts, the whole layer
       as a cold-L2 graph (D1 and SGLang's expert path), and each op alone as a cold-L2 graph.
gemm   the dense projections at every scored token count: cuBLAS (the baseline), SGLang's Hopper
       GEMV (1 token) and tiny GEMM (<= 16), and a Triton split-K skinny GEMM sweep. Error vs cuBLAS.
fa3    FA3 per shape with num_splits / pack_gqa variants, as achieved KV bandwidth.
"""

import triton.language as tl

LAB_PREPARE_LIMIT_S = 255.0  # prepare() may take 300 s; keep 45 s of margin
LAB_T = (1, 8, 16, 32, 128, 256)


# ---------------------------------------------------------------- ops

def _sglang_experts(x, topk_weights, topk_ids, state):
    """baseline.py's _experts: SGLang's tile-height rule and its post-norm (unfused below 32)."""
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    bank = state["bank"]
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    local_ids = bank["expert_map"][topk_ids].contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    for bsm in (8, 16, 32, 48, 64):
        if num_tokens * TOP_K / LOCAL_EXPERTS / bsm < 0.9:
            break
    sorted_ids, block_experts, n_padded = moe_align_block_size(local_ids, bsm, N_EXPERTS)
    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=bsm, mul_topk_weights=False, is_ep=True,
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


def _op_steps(data, state):
    """kernel_fn split into named steps. Each step reads env and returns its outputs, so a step can
    be replayed alone; running them in order once fills env with real intermediates."""
    cfg = data["config"]
    batch, length, tokens = cfg["batch"], cfg["q_len"], cfg["tokens"]
    hq, hkv, d = 16, 2, 64
    dense = state["dense"]
    bank = state["bank"]
    window = cfg["window"]
    split_router = tokens > ROUTER_FP32_MAX_TOKENS
    moe_block = block_size_m(tokens)
    size_n, size_k = bank["size_n"], D_EXPERT
    gemm1_n = 2 * size_n
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=True,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)

    def qk_norm(e):
        if cfg["rope_theta"]:
            q, k = state["qk_norm"](e["qkv"], data["positions"], state["cos_sin"])
        else:
            q, k = state["qk_norm"](e["qkv"])
        return {"q": q, "k": k}

    def kv_store(e):
        v = e["qkv"][:, hq * d + hkv * d: hq * d + 2 * hkv * d]
        store_cache(e["k"].reshape(tokens, hkv * d).to(FP8).view(torch.uint8), v.to(FP8).view(torch.uint8),
                    data["k_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["v_cache"].view(-1, hkv * d).view(torch.uint8),
                    data["cache_loc"], reserved_skip_index=-1)
        return {}

    def fa3(e):
        out = flash_attn_with_kvcache(
            q=e["q8"], k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
            cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=length,
            softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
            window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=data["scheduler_metadata"])
        return {"attn": out[0] if isinstance(out, tuple) else out}

    def boundary1(e):
        f, s, r = state["attn_boundary"](data["stream"], e["a"], branch_dtype=torch.bfloat16,
                                         router_dtype=None if split_router else torch.float32)
        return {"h": f, "s1": s, "rin": r}

    def router(e):
        if split_router:
            logits = torch.mm(e["h"], state["router_hi_t"], out_dtype=torch.float32)
            logits += torch.mm(e["h"], state["router_lo_t"], out_dtype=torch.float32)
        else:
            logits = F.linear(e["rin"], state["router"])
        return {"logits": logits}

    def routing(e):
        w, ids = state["routing"](None, e["logits"], TOP_K, True)
        return {"tw": w, "ids": ids}

    def local(e):
        lid = bank["expert_map"][(e["ids"] - data["local_start"]) % N_EXPERTS].contiguous()
        return {"lid": lid, "tw32": e["tw"].to(torch.float32).contiguous()}

    def align(e):
        s, b, n = moe_align_block_size(e["lid"], moe_block, N_EXPERTS)
        return {"sorted": s, "blocks": b, "npad": n}

    def marlin1(e):
        c1 = torch.empty((tokens * TOP_K, gemm1_n), device="cuda", dtype=torch.bfloat16)
        c1 = moe_wna16_marlin_gemm(
            e["px"], c1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, bank["workspace"],
            e["sorted"], e["blocks"], e["npad"], e["tw32"], top_k=TOP_K,
            size_m=tokens, size_n=gemm1_n, size_k=size_k, **common)
        return {"c1": c1}

    def silu(e):
        c2 = torch.empty((tokens * TOP_K, size_n), device="cuda", dtype=torch.bfloat16)
        silu_and_mul(e["c1"].view(-1, gemm1_n), c2, expert_ids=e["lid"].view(-1), expert_step=1)
        return {"c2": c2}

    def marlin2(e):
        c3 = torch.empty((tokens * TOP_K, size_k), device="cuda", dtype=torch.bfloat16)
        y = moe_wna16_marlin_gemm(
            e["c2"], c3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, bank["workspace"],
            e["sorted"], e["blocks"], e["npad"], e["tw32"], top_k=1,
            size_m=tokens * TOP_K, size_n=size_k, size_k=size_n, **common)
        return {"y": y.view(tokens, TOP_K, size_k)}

    def unpool(e):
        if tokens <= SLICED_UNPOOL_MAX_TOKENS:
            return {"ffn": F.pad(_gemm(e["eo"], dense["unpooler"][:UNPOOL_SLICE]), (0, HIDDEN - UNPOOL_SLICE))}
        return {"ffn": _gemm(e["eo"], dense["unpooler"])}

    return [
        ("qkv gemm", lambda e: {"qkv": _gemm(data["attn_input"], dense["qkv_gate"])}),
        ("qk norm", qk_norm),
        ("kv cast+store", kv_store),
        ("q fp8", lambda e: {"q8": e["q"].view(tokens, hq, d).to(FP8)}),
        ("fa3", fa3),
        ("norm gate", lambda e: {"og": state["o_norm_gate"](e["attn"].reshape(tokens, hq * d).to(torch.bfloat16),
                                                            e["qkv"][:, hq * d + 2 * hkv * d:])}),
        ("o gemm", lambda e: {"a": _gemm(e["og"], dense["o"])}),
        ("boundary1", boundary1),
        ("pooler gemm", lambda e: {"pool": _gemm(e["h"], dense["pooler"])}),
        ("router", router),
        ("routing", routing),
        ("local ids", local),
        ("pre-expert norm", lambda e: {"px": state["pre_expert_norm"](e["pool"])}),
        ("align", align),
        ("marlin1", marlin1),
        ("silu", silu),
        ("marlin2", marlin2),
        ("post-norm fused", lambda e: {"eo": post_expert_norm_weighted_sum(e["y"], e["tw32"], state["post_expert_w"],
                                                                           e["lid"])}),
        ("unpooler gemm", unpool),
        ("boundary2", lambda e: dict(zip(("nx", "s2", "_"), state["ffn_boundary"](
            e["s1"], e["ffn"], branch_dtype=torch.bfloat16)))),
    ]


def _busiest_start(ids):
    pairs = torch.bincount((ids.long() // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    return LOCAL_EXPERTS * int(pairs.argmax())


def _load(ids, start):
    loc = (ids.long() - start) % N_EXPERTS
    loc = loc[loc < LOCAL_EXPERTS]
    return int(loc.numel()), int(loc.unique().numel())


def sec_ops(lab, state, deadline):
    fixed = lab.empty_graph_us()
    log(f"ops: empty-graph replay {fixed:.1f} us (subtract from every time below; the judge does)")
    summary = []
    for i, (name, batch, q_len, layer, weight) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            log(f"ops: stopped before {name}, deadline")
            break
        data = lab_input(batch, q_len, layer, seed=1000 + i)
        _, _, ids = kernel_fn(data, state)
        data["local_start"].fill_(_busiest_start(ids))
        routes, distinct = _load(ids, int(data["local_start"]))
        d1 = lab.time_graph(lambda: kernel_fn(data, state)) - fixed
        steps = _op_steps(data, state)
        env = {}
        for _, fn in steps:
            env.update(fn(env))
        # The same layer as one chain of steps (no side stream), with D1's expert steps or with
        # SGLang's whole expert path in their place.
        names = [n for n, _ in steps]
        first, last = names.index("align"), names.index("post-norm fused")
        sgl_experts = ("experts SGLang", lambda e: {"eo": _sglang_experts(
            e["px"], e["tw"], (e["ids"] - data["local_start"]) % N_EXPERTS, state)})
        chains = {"D1": steps, "SGLang": steps[:first] + [sgl_experts] + steps[last + 1:]}

        def run_chain(chain):
            e = dict(env)
            for _, fn in chain:
                e.update(fn(e))

        chain_us = {k: lab.time_graph(lambda c=c: run_chain(c)) - fixed for k, c in chains.items()}
        parts = []
        for step, fn in steps:
            parts.append((step, lab.time_graph(lambda fn=fn: fn(env)) - fixed))
        total = sum(t for _, t in parts)
        old_post = lab.time_graph(lambda: _sglang_post(env, state)) - fixed
        kv = kv_bytes(data)
        log(f"ops {name}: T={batch * q_len} local routes {routes} distinct {distinct} kv {kv / 1e6:.1f} MB | "
            f"layer D1 {d1:.1f} us | chain D1 {chain_us['D1']:.1f}, chain SGLang experts {chain_us['SGLang']:.1f} us "
            f"| sum of ops {total:.1f} us")
        log(f"ops {name} parts: " + ", ".join(f"{s} {t:.1f}" for s, t in parts)
            + f" | SGLang post-norm {old_post:.1f}")
        summary.append((name, weight, d1, d1 + chain_us["SGLang"] - chain_us["D1"]))
        del data, env, steps
        torch.cuda.empty_cache()
    if summary:
        log("ops summary (name, weight, D1 layer us, est. SGLang layer us = D1 + chain difference): "
            + "; ".join(f"{n} w{w} {a:.1f} {b:.1f}" for n, w, a, b in summary))


def _sglang_post(env, state):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    return fused_post_expert_norm_weighted_sum(env["y"], env["tw32"], norm_weight=state["post_expert_w"],
                                               eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=env["lid"])


# ---------------------------------------------------------------- gemm

@triton.jit
def _lab_skinny_gemm(x_ptr, w_ptr, out_ptr, T, N, K, stride_x, stride_w,
                     BLOCK_T: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     SPLIT: tl.constexpr, OUT_BF16: tl.constexpr):
    """out[s, t, n] = sum over K-split s of w[n, k] x[t, k], weights as the MMA's M side.
    Requires N % BLOCK_N == 0 and K % (SPLIT * BLOCK_K) == 0. OUT_BF16 (SPLIT == 1): bf16 [T, N]."""
    pid_n = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_t = tl.program_id(2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_k = tl.arange(0, BLOCK_K)
    k_per = K // SPLIT
    k0 = pid_s * k_per
    acc = tl.zeros((BLOCK_N, BLOCK_T), dtype=tl.float32)
    t_ok = offs_t[None, :] < T
    for k in range(k0, k0 + k_per, BLOCK_K):
        w = tl.load(w_ptr + offs_n[:, None] * stride_w + (k + offs_k)[None, :], eviction_policy="evict_first")
        x = tl.load(x_ptr + offs_t[None, :] * stride_x + (k + offs_k)[:, None], mask=t_ok, other=0.0)
        acc = tl.dot(w, x, acc)
    out = tl.trans(acc)
    if OUT_BF16:
        ptrs = out_ptr + offs_t[:, None] * N + offs_n[None, :]
        tl.store(ptrs, out.to(tl.bfloat16), mask=offs_t[:, None] < T)
    else:
        ptrs = out_ptr + pid_s * T * N + offs_t[:, None] * N + offs_n[None, :]
        tl.store(ptrs, out, mask=offs_t[:, None] < T)


@triton.jit
def _lab_split_reduce(part_ptr, out_ptr, TN, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = offs < TN
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for s in tl.static_range(SPLIT):
        acc += tl.load(part_ptr + s * TN + offs, mask=ok, other=0.0)
    tl.store(out_ptr + offs, acc.to(tl.bfloat16), mask=ok)


def lab_skinny_gemm(x, w, cfg, part=None, out=None):
    bn, bk, split, warps, stages = cfg
    t, k = x.shape
    n = w.shape[0]
    bt = 16 if t <= 16 else 32 if t <= 32 else 64
    if out is None:
        out = torch.empty((t, n), dtype=torch.bfloat16, device=x.device)
    grid = (n // bn, split, triton.cdiv(t, bt))
    if split == 1:
        _lab_skinny_gemm[grid](x, w, out, t, n, k, x.stride(0), w.stride(0), BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk,
                               SPLIT=1, OUT_BF16=True, num_warps=warps, num_stages=stages)
        return out
    if part is None:
        part = torch.empty((split, t, n), dtype=torch.float32, device=x.device)
    _lab_skinny_gemm[grid](x, w, part, t, n, k, x.stride(0), w.stride(0), BLOCK_T=bt, BLOCK_N=bn, BLOCK_K=bk,
                           SPLIT=split, OUT_BF16=False, num_warps=warps, num_stages=stages)
    _lab_split_reduce[(triton.cdiv(t * n, 1024),)](part, out, t * n, SPLIT=split, BLOCK=1024, num_warps=4)
    return out


def _gemm_shapes(state):
    dense = state["dense"]
    return (
        ("qkv", dense["qkv_gate"], LAB_T),
        ("o", dense["o"], LAB_T),
        ("pooler", dense["pooler"], LAB_T),
        ("unpool slice", dense["unpooler"][:UNPOOL_SLICE], (1, 8)),
        ("unpooler", dense["unpooler"], (16, 32, 128, 256)),
        ("router hi", state["router_hi_t"].t().contiguous(), LAB_T),
    )


def _skinny_configs(n, k):
    out = []
    for bn in (64, 128):
        for bk in (128, 256):
            for split in (1, 2, 4, 8, 16):
                if n % bn or k % (split * bk):
                    continue
                for warps in (4, 8):
                    out.append((bn, bk, split, warps, 4))
    return out


def _bw(nbytes, us):
    """TB/s; a time at or below the replay's fixed cost reads as the bandwidth of 0.01 us."""
    return nbytes / (max(us, 0.01) * 1e-6) / 1e12


def sec_gemm(lab, state, deadline):
    fixed = lab.empty_graph_us()
    try:
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv, use_hopper_bf16_gemv
    except Exception as exc:
        hopper_bf16_gemv = None
        log(f"gemm: hopper_bf16_gemv unavailable: {_error_text(exc, 150)}")
    try:
        from sglang.kernels.ops.gemm.tiny_gemm import can_use_tiny_gemm, tiny_gemm_bf16
    except Exception as exc:
        tiny_gemm_bf16 = None
        log(f"gemm: tiny_gemm_bf16 unavailable: {_error_text(exc, 150)}")
    best = {}
    for name, w, ts in _gemm_shapes(state):
        n, k = w.shape
        nbytes = n * k * 2
        for t in ts:
            if lab.elapsed() > deadline:
                log(f"gemm: stopped before {name} T={t}, deadline")
                return
            x = torch.randn((t, k), device="cuda").to(torch.bfloat16)
            ref = F.linear(x, w).float()
            res = {}

            def run(label, fn):
                try:
                    err = float((fn().float() - ref).norm() / ref.norm())
                    us = lab.time_graph(fn) - fixed
                    res[label] = (us, err)
                except Exception as exc:
                    log(f"gemm {name} T={t} {label}: FAILED {type(exc).__name__}: {_error_text(exc, 200)}")

            run("cublas", lambda: F.linear(x, w))
            if hopper_bf16_gemv is not None and t == 1 and use_hopper_bf16_gemv(1, n, k):
                run("hopper_gemv", lambda: hopper_bf16_gemv(x, w))
            if tiny_gemm_bf16 is not None and t <= 16 and can_use_tiny_gemm(n, k, 16):
                run("tiny_gemm", lambda: tiny_gemm_bf16(x, w))
            cfgs = _skinny_configs(n, k)
            if t != ts[0] and name in best:
                cfgs = best[name]  # the T sweep's top configs from the first token count
            for cfg in cfgs:
                if lab.elapsed() > deadline:
                    break
                part = torch.empty((cfg[2], t, n), dtype=torch.float32, device="cuda") if cfg[2] > 1 else None
                out = torch.empty((t, n), dtype=torch.bfloat16, device="cuda")
                run("triton " + "/".join(map(str, cfg)), lambda cfg=cfg, part=part, out=out:
                    lab_skinny_gemm(x, w, cfg, part, out))
            tri = sorted((v[0], lbl) for lbl, v in res.items() if lbl.startswith("triton") and v[1] < 1e-2)
            if t == ts[0] and tri:
                best[name] = [tuple(int(p) for p in lbl.split()[1].split("/")) for _, lbl in tri[:4]]
            cub = res.get("cublas", (float("nan"), 0))[0]
            shown = sorted(res.items(), key=lambda kv: kv[1][0])
            top = [f"{lbl} {us:.1f}us {_bw(nbytes, us):.2f}TB/s e{err:.0e}" for lbl, (us, err) in shown
                   if not lbl.startswith("triton")] + [f"{lbl} {us:.1f}us {_bw(nbytes, us):.2f}TB/s e{err:.0e}"
                                                       for lbl, (us, err) in shown if lbl.startswith("triton")][:3]
            log(f"gemm {name} [{n}x{k}] T={t}: cublas {cub:.1f}us ({_bw(nbytes, cub):.2f}TB/s) | " + "; ".join(top))
            del x, ref
    # The FP32 router as SGLang runs it at <= 8 tokens.
    for t in (1, 8):
        xr = torch.randn((t, HIDDEN), device="cuda")
        us = lab.time_graph(lambda: F.linear(xr, state["router"])) - fixed
        log(f"gemm router fp32 T={t}: F.linear {us:.1f}us ({_bw(256 * HIDDEN * 4, us):.2f}TB/s)")


# ---------------------------------------------------------------- fa3

def sec_fa3(lab, deadline):
    fixed = lab.empty_graph_us()
    for i, (name, batch, q_len, layer, _) in enumerate(SCORED):
        if lab.elapsed() > deadline:
            log(f"fa3: stopped before {name}, deadline")
            return
        data = lab_input(batch, q_len, layer, seed=2000 + i)
        cfg = data["config"]
        tokens, window = cfg["tokens"], cfg["window"]
        q = torch.randn((tokens, 16, 64), device="cuda").to(FP8)
        kv = kv_bytes(data)

        def call(sched, **kw):
            return flash_attn_with_kvcache(
                q=q, k_cache=data["k_cache"], v_cache=data["v_cache"], page_table=data["page_table"],
                cache_seqlens=data["cache_seqlens"], cu_seqlens_q=data["cu_seqlens_q"], max_seqlen_q=q_len,
                softmax_scale=cfg["sm_scale"] * cfg["qk_scale_factor"], causal=True,
                window_size=(window - 1, 0) if window else (-1, -1), scheduler_metadata=sched, **kw)

        variants = [("baseline", dict(sched=data["scheduler_metadata"]))]
        variants += [(f"splits {s}", dict(sched=None, num_splits=s)) for s in (1, 2, 4, 8, 16, 32, 64)]
        variants += [("pack_gqa T", dict(sched=None, pack_gqa=True)), ("pack_gqa F", dict(sched=None, pack_gqa=False))]
        ref = call(**variants[0][1])
        ref = (ref[0] if isinstance(ref, tuple) else ref).float()
        res = []
        for label, kw in variants:
            try:
                out = call(**kw)
                out = (out[0] if isinstance(out, tuple) else out).float()
                err = float((out - ref).norm() / ref.norm())
                us = lab.time_graph(lambda kw=kw: call(**kw)) - fixed
                res.append(f"{label} {us:.1f}us {_bw(kv, us):.2f}TB/s" + (f" e{err:.0e}" if err > 1e-3 else ""))
            except Exception as exc:
                res.append(f"{label} FAILED {type(exc).__name__}: {_error_text(exc, 120)}")
        log(f"fa3 {name} kv {kv / 1e6:.1f} MB: " + "; ".join(res))
        del data, q
        torch.cuda.empty_cache()


# ---------------------------------------------------------------- entry

def _lab_env():
    import importlib.metadata as md
    vers = {}
    for pkg in ("torch", "triton", "sgl-kernel", "sglang", "flashinfer-python"):
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = "?"
    props = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    log(f"env: {vers}; {props.name} {props.multi_processor_count} SMs; free {free / 2**30:.0f} of {total / 2**30:.0f} GiB")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    base_s = time.monotonic() - t0
    lab = Lab(LAB_PREPARE_LIMIT_S - base_s)
    log(f"submission prepare {base_s:.1f}s; lab budget {lab.budget_s:.0f}s")
    try:
        _lab_env()
    except Exception as exc:
        log(f"env: FAILED {_error_text(exc)}")
    lab.section("ops", sec_ops, state, until=0.45 * lab.budget_s)
    lab.section("gemm", sec_gemm, state, until=0.8 * lab.budget_s)
    lab.section("fa3", sec_fa3)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"lab total {lab.elapsed():.1f}s")
    return state
