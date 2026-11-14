# okc: problem=muse_prefill_layer
"""v3: SGLang's decoder layer (baseline.py, commit a55ab7d5) with our own routed-expert paths.

* R3 (per token class, where a lab measured it correct and faster): fused NVFP4 grouped GEMMs in Triton.
  Local routes sorted by expert on the GPU, tiles of BT rows per expert, NVFP4 decoded in registers
  per BK step (Marlin's bit trick, weights nibble-permuted losslessly in prepare), SwiGLU in GEMM1's
  epilogue, SGLang's post-expert norm weighted sum over the sorted rows.
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
    _gmm_prepare(state, weights)
    _r3_prepare(state, weights)
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
# v3 (R3): fused NVFP4 grouped expert GEMMs in Triton, for prefill token counts (local/make_v3.py
# inserts this after v2/gmm.py).
#
# The local routes are sorted by expert (as v2) and cut into tiles of BT rows, one expert per tile;
# a device-side tile table (no host sync) maps each program to (expert, rows). Each program decodes
# BN weight rows of its expert straight from NVFP4 per BK step and multiplies them with its BT token
# rows: the decode is paid once per BT tokens, which is what makes it cheap at prefill sizes (at
# decode sizes it isn't: MoE F45). GEMM1 does gate and up together with SwiGLU in the epilogue;
# GEMM2 writes per-route rows for v2's post-norm combine kernel.
#
# The decode (_r3_dequant / _r3_scales / _r3_tile) and the nibble repack (r3_repack) are copied from
# our MoE session's M2 (moe/m2/m2.py: _m2_dequant, _m2_scales, _m2_tile, m2_repack), whose asm was
# interpreted bit for bit locally and ran correctly on the H200 (MoE E2). Weights stay NVFP4:
# r3_repack only permutes nibbles (lossless).
# ---------------------------------------------------------------------------------------------

# Decoded weights are code * scale * 2^-62 (Marlin's bit trick with scales raised by 2^64);
# epilogues multiply by 2^62.
R3_OUT_SHIFT = 2.0 ** 62
# Token classes: (fewer than this many tokens, class); the rest is "ft" (first-turn 16K chunks).
R3_CLASSES = ((1200, "t1"), (8192, "t4"))
# class -> (GEMM1 config, GEMM2 config); a config is (swap, bt, bn, bk, warps, stages, scale2d).
# Empty until a lab measures them; prepare() checks each against Marlin and drops any that disagree.
R3_CONFIGS = {}
R3_REP_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
R3_MAX_ERR = 1e-2
# v2's BF16 grouped path serves from this many tokens when no R3 config does (it lost at 4096: RE2).
GMM_FROM_TOKENS = 8192


def _r3_class(tokens):
    for bound, name in R3_CLASSES:
        if tokens < bound:
            return name
    return "ft"


def r3_repack(packed, bk):
    """uint8 [..., K/2] NVFP4 codes (element 2i in the low nibble) -> the same bytes permuted so
    that within each bk-wide K tile, byte j holds element j (low nibble) and j + bk/2 (high)."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    t = codes.unflatten(-1, (codes.shape[-1] // bk, 2, bk // 2))
    return (t[..., 0, :] | (t[..., 1, :] << 4)).flatten(-2).contiguous()


@triton.jit
def _r3_dequant(q, s_lo, s_hi):
    """Per byte of q: its low and high E2M1 nibbles as bf16 (value * 2^-126, Marlin's bit trick),
    times s_lo / s_hi with mul.rn.bf16x2. prmt first pairs bytes (0,1) and (2,3) into the 16-bit
    lanes, so output element i comes from input byte i whatever elements Triton packs together."""
    return tl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 w, t, s, m;
        prmt.b32 w, $4, $4, 0x3120;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $3, t, $8;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $1, t, $6;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $2, t, $7;
        shl.b32 w, w, 4;
        and.b32 s, w, 0x80008000;
        and.b32 m, w, 0x70007000;
        shr.b32 m, m, 6;
        or.b32 t, s, m;
        mul.rn.bf16x2 $0, t, $5;
        }
        """,
        constraints="=r,=r,=r,=r,r,r,r,r,r", args=[q, s_lo, s_hi],
        dtype=(tl.bfloat16, tl.bfloat16), is_pure=True, pack=4)


@triton.jit
def _r3_scales(ptrs):
    """float8_e4m3fn block scales -> bf16 scale * 2^64 (exact)."""
    return (tl.load(ptrs).to(tl.float32) * 18446744073709551616.0).to(tl.bfloat16)


@triton.jit
def _r3_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN: tl.constexpr, HB: tl.constexpr,
             G: tl.constexpr, SCALE2D: tl.constexpr):
    """Dequantized [BN, HB] bf16 halves of the BK = 2 * HB tile at k0: (elements k0 + j, k0 + HB + j)."""
    if SCALE2D:
        q = tl.load(w_rows[:, None] + (k0 // 2 + offs_j)[None, :])
        s_lo = _r3_scales(s_rows[:, None] + ((k0 + offs_j) // 16)[None, :])
        s_hi = _r3_scales(s_rows[:, None] + ((k0 + HB + offs_j) // 16)[None, :])
        lo, hi = _r3_dequant(q, s_lo, s_hi)
    else:
        s_lo = tl.broadcast_to(_r3_scales(s_tile + k0 // 16)[:, :, None], (BN, G, 16))
        s_hi = tl.broadcast_to(_r3_scales(s_tile + (k0 // 16 + G))[:, :, None], (BN, G, 16))
        lo, hi = _r3_dequant(tl.load(w_tile + k0 // 2), s_lo, s_hi)
        lo = tl.reshape(lo, (BN, HB))
        hi = tl.reshape(hi, (BN, HB))
    return lo, hi


@triton.jit(do_not_specialize=["max_tiles"])
def _r3_gemm_kernel(a_ptr, w_ptr, sf_ptr, c_ptr, table_ptr, out_ptr, max_tiles,
                    N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, GATED: tl.constexpr,
                    SWAP: tl.constexpr, BT: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SCALE2D: tl.constexpr):
    """Output columns [nb * BN, +BN) for the rows of tile t (one expert, rows row0 .. row_end - 1 of
    the expert-sorted activations a). Weight rows: n (gate or plain) and N + n (up, GATED)."""
    pid = tl.program_id(0)
    nb = pid // max_tiles
    t = pid - nb * max_tiles
    e = tl.load(table_ptr + t)
    if e < 0:
        return
    row0 = tl.load(table_ptr + max_tiles + t)
    row_end = tl.load(table_ptr + 2 * max_tiles + t)
    HB: tl.constexpr = BK // 2
    G: tl.constexpr = BK // 32
    e64 = e.to(tl.int64)
    c_e = tl.load(c_ptr + e)
    offs_n = nb * BN + tl.arange(0, BN)
    w_rows = w_ptr + e64 * (ROWS * (K // 2)) + offs_n.to(tl.int64) * (K // 2)
    s_rows = sf_ptr + e64 * (ROWS * (K // 16)) + offs_n.to(tl.int64) * (K // 16)
    offs_g = tl.arange(0, G)
    offs_i = tl.arange(0, 16)
    offs_j = tl.arange(0, HB)
    w_tile = w_rows[:, None, None] + (offs_g[:, None] * 16 + offs_i[None, :])[None, :, :]
    s_tile = s_rows[:, None] + offs_g[None, :]
    a_row = (row0 + tl.arange(0, BT)).to(tl.int64)
    valid = a_row < row_end
    if SWAP:
        a_tile = a_ptr + a_row[None, :] * K + offs_j[:, None]
        a_mask = valid[None, :]
        acc = tl.zeros((BN, BT), dtype=tl.float32)
        acc_u = tl.zeros((BN, BT), dtype=tl.float32)
    else:
        a_tile = a_ptr + a_row[:, None] * K + offs_j[None, :]
        a_mask = valid[:, None]
        acc = tl.zeros((BT, BN), dtype=tl.float32)
        acc_u = tl.zeros((BT, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        a_lo = tl.load(a_tile + k0, mask=a_mask, other=0.0)
        a_hi = tl.load(a_tile + (k0 + HB), mask=a_mask, other=0.0)
        lo, hi = _r3_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN, HB, G, SCALE2D)
        if SWAP:
            acc = tl.dot(lo, a_lo, acc)
            acc = tl.dot(hi, a_hi, acc)
        else:
            acc = tl.dot(a_lo, tl.trans(lo), acc)
            acc = tl.dot(a_hi, tl.trans(hi), acc)
        if GATED:
            lo, hi = _r3_tile(w_rows + N * (K // 2), s_rows + N * (K // 16), w_tile + N * (K // 2),
                              s_tile + N * (K // 16), offs_j, k0, BN, HB, G, SCALE2D)
            if SWAP:
                acc_u = tl.dot(lo, a_lo, acc_u)
                acc_u = tl.dot(hi, a_hi, acc_u)
            else:
                acc_u = tl.dot(a_lo, tl.trans(lo), acc_u)
                acc_u = tl.dot(a_hi, tl.trans(hi), acc_u)
    if SWAP:
        rows = a_row[None, :]
        cols = offs_n[:, None]
    else:
        rows = a_row[:, None]
        cols = offs_n[None, :]
    if GATED:
        g = acc * c_e
        res = g * tl.sigmoid(g) * (acc_u * c_e)
    else:
        res = acc * c_e
    tl.store(out_ptr + rows * N + cols, res.to(tl.bfloat16), mask=a_mask)


def _r3_route(x, topk_ids, state):
    """v2's GPU-side sort of the local routes by expert, plus the gathered activation rows."""
    bank = state["bank"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    device = x.device
    local_ids = bank["expert_map"][topk_ids].contiguous()
    keys = torch.where(local_ids.view(-1) < 0, LOCAL_EXPERTS, local_ids.view(-1))
    _, perm = torch.sort(keys, stable=True)
    counts = torch.zeros(LOCAL_EXPERTS + 1, dtype=torch.int32, device=device)
    counts.scatter_add_(0, keys.long(), torch.ones_like(keys))
    offs = counts[:LOCAL_EXPERTS].cumsum(0, dtype=torch.int32)
    inv = torch.empty_like(perm)
    inv.scatter_(0, perm, torch.arange(slots, device=device, dtype=perm.dtype))
    a = torch.empty((slots, D_EXPERT), device=device, dtype=torch.bfloat16)
    _gather_rows_kernel[(slots,)](x.contiguous(), perm, offs, a, H=D_EXPERT, TOPK=TOP_K, NE=LOCAL_EXPERTS,
                                  num_warps=4)
    return local_ids, counts, offs, inv, a


def _r3_max_tiles(slots, bt):
    return -(-slots // bt) + LOCAL_EXPERTS


def _r3_tiles(counts, offs, bt, max_tiles):
    """int32 [3, max_tiles]: (expert or -1, first row, end row) of each BT-row tile, experts in order."""
    c = counts[:LOCAL_EXPERTS]
    tiles = (c + (bt - 1)) // bt
    t_end = tiles.cumsum(0, dtype=torch.int32)
    t_start = t_end - tiles
    p = torch.arange(max_tiles, device=c.device, dtype=torch.int32)
    e = torch.searchsorted(t_end, p, right=True)
    e_c = e.clamp(max=LOCAL_EXPERTS - 1)
    r_start = (offs - c)[e_c] + (p - t_start[e_c]) * bt
    expert = torch.where(e < LOCAL_EXPERTS, e, torch.full_like(e, -1))
    return torch.stack((expert.to(torch.int32), r_start.to(torch.int32), offs[e_c].to(torch.int32))).contiguous()


def _r3_gemm(a, q, sf, c, table, max_tiles, out, cfg, gated):
    swap, bt, bn, bk, warps, stages, scale2d = cfg
    rows_w = q.shape[1]
    n = rows_w // 2 if gated else rows_w
    k = a.shape[1]
    return _r3_gemm_kernel[((n // bn) * max_tiles,)](
        a, q, sf, c, table, out, max_tiles, N=n, K=k, ROWS=rows_w, GATED=gated, SWAP=swap, BT=bt, BN=bn,
        BK=bk, SCALE2D=scale2d, num_warps=warps, num_stages=stages)


def _experts_r3(x, topk_weights, topk_ids, state, cfg1, cfg2):
    """Same contract as _experts."""
    r3, gmm = state["r3"], state["gmm"]
    tokens = x.shape[0]
    slots = tokens * TOP_K
    local_ids, counts, offs, inv, a = _r3_route(x, topk_ids, state)
    topk_weights = topk_weights.to(torch.float32).contiguous()
    mt1, mt2 = _r3_max_tiles(slots, cfg1[1]), _r3_max_tiles(slots, cfg2[1])
    table1 = _r3_tiles(counts, offs, cfg1[1], mt1)
    table2 = table1 if cfg2[1] == cfg1[1] else _r3_tiles(counts, offs, cfg2[1], mt2)
    act = torch.empty((slots, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _r3_gemm(a, r3["w13"][cfg1[3]], gmm["w13_sf"], r3["c1"], table1, mt1, act, cfg1, True)
    y2 = torch.empty((slots, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _r3_gemm(act, r3["w2"][cfg2[3]], gmm["w2_sf"], r3["c2"], table2, mt2, y2, cfg2, False)
    out = torch.empty((tokens, D_EXPERT), device=x.device, dtype=torch.bfloat16)
    _gmm_combine_kernel[(tokens,)](y2, local_ids, inv, gmm["w2_gs"], topk_weights, state["post_expert_w"], out,
                                   POST_NORM_EPS, H=D_EXPERT, TOPK=TOP_K, num_warps=8)
    return out


def _moe_v3(x, topk_weights, topk_ids, state, tokens):
    cfg = state["r3"]["cfg"].get(_r3_class(tokens))
    if cfg is not None:
        return _experts_r3(x, topk_weights, topk_ids, state, *cfg)
    if tokens >= GMM_FROM_TOKENS and state["gmm"]["ok"]:
        return _experts_gmm(x, topk_weights, topk_ids, state)
    return _experts(x, topk_weights, topk_ids, state)


def r3_add_bk(state, bk):
    """Repacked copies of the expert weights for BK = bk (lossless nibble permutation), per expert."""
    r3, gmm = state["r3"], state["gmm"]
    for name in ("w13", "w2"):
        if bk not in r3[name]:
            src = gmm[name + "_q"]
            r3[name][bk] = torch.stack([r3_repack(src[e], bk) for e in range(src.shape[0])])


def r3_check(state, cfg1, cfg2, tokens, seed=2):
    """(rel err, worst row err) of _experts_r3 against SGLang's Marlin path on a GPU routing."""
    device = state["bank"]["expert_map"].device
    gen = torch.Generator(device=device).manual_seed(seed)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device=device) * 0.5).to(torch.bfloat16)
    ids, w = _gmm_selftest_routing(tokens, device, seed=seed)
    ref = _experts(x, w, ids, state).float()
    out = _experts_r3(x, w, ids, state, cfg1, cfg2).float()
    torch.cuda.synchronize()
    if not bool(torch.isfinite(out).all()):
        return float("inf"), float("inf")
    err = float((out - ref).norm() / ref.norm().clamp(min=1e-30))
    row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
    return err, row


def _r3_prepare(state, weights):
    """Global scales for the epilogues, repacked weights for the configured BKs, and a GPU check of
    each configured class against Marlin (which also compiles its kernels); failures fall back."""
    gmm = state["gmm"]
    state["r3"] = {
        "w13": {}, "w2": {}, "cfg": {},
        "c1": (gmm["w13_gs"] * R3_OUT_SHIFT).contiguous(),
        "c2": torch.full_like(gmm["w2_gs"], R3_OUT_SHIFT),
    }
    for cls, (cfg1, cfg2) in R3_CONFIGS.items():
        try:
            r3_add_bk(state, cfg1[3])
            r3_add_bk(state, cfg2[3])
            err, row = r3_check(state, cfg1, cfg2, R3_REP_TOKENS[cls])
            ok = err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
            print(f"[v3] {cls}: fused vs Marlin rel err {err:.2e}, worst row {row:.2e} -> {'fused' if ok else 'fallback'}",
                  flush=True)
            if ok:
                state["r3"]["cfg"][cls] = (cfg1, cfg2)
        except Exception as exc:
            print(f"[v3] {cls}: fused path failed, falling back: {type(exc).__name__}: {str(exc)[-300:]}", flush=True)
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
    expert_output = _moe_v3(state["pre_expert_norm"](expert_input), topk_weights, local_ids, state, tokens)
    if tokens <= SLICED_UNPOOL_MAX_TOKENS:
        unpooled = _gemm(expert_output, dense["unpooler"][:UNPOOL_SLICE])
        ffn_output = F.pad(unpooled, (0, HIDDEN - UNPOOL_SLICE))  # all-gather: identity
    else:
        ffn_output = _gemm(expert_output, dense["unpooler"])

    # Post-FFN boundary, with the next layer's pre-attention norm.
    next_attn_input, stream, _ = state["ffn_boundary"](stream, ffn_output, branch_dtype=torch.bfloat16)
    return next_attn_input, stream, topk_ids


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


# ====================================================================== lab (re3): check runs only

"""RE3 lab (check runs only): the fused NVFP4 grouped GEMMs (R3) on the judge's H200.
1. Stage sweeps at 3012 tokens (real-router routing): GEMM1 (gated, SwiGLU epilogue) and GEMM2 configs,
   each checked against an FP32 reference built from the exact BF16 decode, with registers/spills.
2. The fastest configs re-timed at 753 and 16384 tokens (no recompiles: max_tiles isn't specialized).
3. End to end per class (753 / 3012 / 16384, real-router and synthetic routing): Marlin, v2's grouped
   path and R3, errors vs Marlin; plus R3 correctness at the judge's test sizes in each class.
4. Hybrid dispatch: a class uses R3 only where it was correct everywhere and >= 3% faster than what v3
   runs there otherwise (Marlin below 8192 tokens, the grouped path from 8192). Prints R3_CONFIGS.
5. If time is left: dequant kernel block sizes (R10).
Appended after submission.py (v3) by local/make_lab.py; kernel_fn stays the submission's, and reads
state["r3"]["cfg"], which this lab fills."""

RE3_PREPARE_LIMIT_S = 255.0
RE3_SWEEP_TOKENS = 3012
RE3_CLASS_TOKENS = {"t1": 753, "t4": 3012, "ft": 16384}
RE3_TEST_TOKENS = {"t1": (300, 600), "t4": (1506, 4096), "ft": ()}
# (swap, bt, bn, bk, warps, stages, scale2d), most promising first (the sweep stops when time runs low).
RE3_G1 = [
    (0, 64, 128, 256, 8, 3, 0), (0, 128, 64, 256, 8, 3, 0), (1, 64, 128, 256, 8, 3, 0), (0, 64, 64, 256, 4, 3, 0),
    (0, 128, 64, 128, 8, 3, 0), (1, 128, 64, 128, 8, 3, 0), (0, 64, 128, 128, 4, 4, 0), (0, 64, 128, 256, 8, 2, 1),
    (1, 64, 64, 256, 4, 3, 0), (0, 32, 128, 256, 4, 3, 0), (0, 128, 128, 128, 8, 2, 0),
]
RE3_G2 = [
    (0, 64, 128, 256, 8, 3, 0), (0, 128, 128, 128, 8, 3, 0), (1, 64, 128, 256, 8, 3, 0), (0, 64, 64, 256, 4, 3, 0),
    (0, 128, 64, 256, 8, 3, 0), (1, 128, 128, 128, 8, 3, 0), (0, 64, 256, 128, 8, 3, 0), (0, 64, 128, 256, 8, 2, 1),
    (0, 128, 128, 256, 8, 2, 0), (0, 32, 128, 256, 4, 3, 0),
]
RE3_TOP = 3
RE3_BKS = (128, 256)


def re3_cfg_name(c):
    return f"s{c[0]} bt{c[1]} bn{c[2]} bk{c[3]} w{c[4]} st{c[5]}{' 2d' if c[6] else ''}"


def re3_real_routing(weights, tokens, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    mean = torch.randn((1, HIDDEN), generator=gen, device="cuda")
    h = torch.randn((tokens, HIDDEN), generator=gen, device="cuda") + mean
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    scores = torch.sigmoid(0.1767766952966369 * (h @ weights["router"].float().T))
    ids = (scores + weights["router_bias"].float()).topk(TOP_K, dim=-1).indices
    w = scores.gather(1, ids)
    w = w / w.sum(-1, keepdim=True)
    pairs = torch.bincount((ids // LOCAL_EXPERTS).flatten(), minlength=N_EXPERTS // LOCAL_EXPERTS)
    ids = (ids - LOCAL_EXPERTS * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), w.float()


def re3_inputs(weights, tokens, kind):
    gen = torch.Generator(device="cuda").manual_seed(13)
    x = (torch.randn((tokens, D_EXPERT), generator=gen, device="cuda") * 0.5).to(torch.bfloat16)
    ids, w = re3_real_routing(weights, tokens, tokens) if kind == "real" else synthetic_routing(tokens, seed=tokens)
    return x, ids, w


def re3_rel(out, ref):
    return float((out.float() - ref).norm() / ref.norm().clamp(min=1e-30))


class Re3Stage:
    """Sorted rows, tables and FP32 stage references for one routing."""

    def __init__(self, state, x, ids):
        gmm = state["gmm"]
        self.state = state
        self.slots = x.shape[0] * TOP_K
        self.local_ids, self.counts, self.offs, self.inv, self.a = _r3_route(x, ids, state)
        self.n = int(self.offs[-1])
        bounds = [0] + self.offs.tolist()
        _dequant(gmm["w13_q"], gmm["w13_sf"], self.counts, gmm["w13"])
        _dequant(gmm["w2_q"], gmm["w2_sf"], self.counts, gmm["w2"])
        self.segs = [(e, bounds[e], bounds[e + 1]) for e in range(LOCAL_EXPERTS) if bounds[e + 1] > bounds[e]]
        act = torch.zeros((self.n, D_EXPERT), device="cuda")
        for e, s, t in self.segs:
            h = self.a[s:t].float() @ gmm["w13"][e].float().T
            g, u = h[:, :D_EXPERT] * gmm["w13_gs"][e], h[:, D_EXPERT:] * gmm["w13_gs"][e]
            act[s:t] = g * torch.sigmoid(g) * u
        self.act_ref = act
        self.act_in = torch.zeros((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        self.act_in[: self.n] = act.to(torch.bfloat16)
        y = torch.zeros((self.n, D_EXPERT), device="cuda")
        for e, s, t in self.segs:
            y[s:t] = self.act_in[s:t].float() @ gmm["w2"][e].float().T
        self.y_ref = y
        self.tables = {}
        self.out1 = torch.empty((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)
        self.out2 = torch.empty((self.slots, D_EXPERT), device="cuda", dtype=torch.bfloat16)

    def table(self, bt):
        if bt not in self.tables:
            mt = _r3_max_tiles(self.slots, bt)
            self.tables[bt] = (_r3_tiles(self.counts, self.offs, bt, mt), mt)
        return self.tables[bt]

    def launch(self, cfg, stage):
        r3, gmm = self.state["r3"], self.state["gmm"]
        table, mt = self.table(cfg[1])
        if stage == 1:
            return _r3_gemm(self.a, r3["w13"][cfg[3]], gmm["w13_sf"], r3["c1"], table, mt, self.out1, cfg, True)
        return _r3_gemm(self.act_in, r3["w2"][cfg[3]], gmm["w2_sf"], r3["c2"], table, mt, self.out2, cfg, False)

    def run(self, lab, cfg, stage, reps=5):
        k = self.launch(cfg, stage)
        torch.cuda.synchronize()
        err = re3_rel(self.out1[: self.n] if stage == 1 else self.out2[: self.n], self.act_ref if stage == 1 else self.y_ref)
        us = lab.time_graph(lambda: self.launch(cfg, stage), reps=reps)
        return us, err, getattr(k, "n_regs", -1), getattr(k, "n_spills", -1)


def sec_sweep(lab, weights, state, out):
    x, ids, w = re3_inputs(weights, RE3_SWEEP_TOKENS, "real")
    st = Re3Stage(state, x, ids)
    log(f"sweep T={RE3_SWEEP_TOKENS}: {st.n} local rows over {len(st.segs)} experts")
    good = {1: [], 2: []}
    order = [(i, stage) for i in range(max(len(RE3_G1), len(RE3_G2))) for stage in (1, 2)]
    for i, stage in order:
        cfgs = RE3_G1 if stage == 1 else RE3_G2
        if i >= len(cfgs):
            continue
        cfg = cfgs[i]
        if lab.left() < 110:
            log("sweep: budget low, stopped")
            break
        if True:
            t = time.monotonic()
            try:
                us, err, regs, spills = st.run(lab, cfg, stage)
                ok = err < R3_MAX_ERR
                log(f"g{stage} {re3_cfg_name(cfg)}: {us:.1f}us err {err:.1e} regs {regs} spill {spills} "
                    f"({time.monotonic() - t:.1f}s){'' if ok else ' WRONG'}")
                if ok:
                    good[stage].append((us, cfg))
            except Exception as exc:
                log(f"g{stage} {re3_cfg_name(cfg)}: FAILED {type(exc).__name__}: {_error_text(exc, 160)}")
    for stage in (1, 2):
        good[stage].sort()
        out[stage] = [c for _, c in good[stage][:RE3_TOP]]


def sec_retime(lab, weights, state, top, best):
    """Top configs of each stage at every class size; best[(tokens, stage)] = fastest correct config."""
    for tokens in (RE3_CLASS_TOKENS["t1"], RE3_CLASS_TOKENS["t4"], RE3_CLASS_TOKENS["ft"]):
        if lab.left() < 70:
            log("retime: budget low, stopping")
            return
        x, ids, w = re3_inputs(weights, tokens, "real")
        st = Re3Stage(state, x, ids)
        for stage in (1, 2):
            row = []
            for cfg in top.get(stage, []):
                try:
                    us, err, _, _ = st.run(lab, cfg, stage, reps=5)
                    row.append(f"{re3_cfg_name(cfg)} {us:.1f}us/{err:.0e}")
                    if err < R3_MAX_ERR and (us < best.get((tokens, stage), (1e30, None))[0]):
                        best[(tokens, stage)] = (us, cfg)
                except Exception as exc:
                    row.append(f"{re3_cfg_name(cfg)} FAILED {type(exc).__name__}")
            log(f"retime T={tokens} ({st.n} rows) g{stage}: " + "; ".join(row))
        del st


def sec_e2e(lab, weights, state, best):
    """Per class: Marlin vs grouped vs R3 end to end; R3 correctness at the test sizes; dispatch."""
    lines = {}
    for cls, tokens in RE3_CLASS_TOKENS.items():
        if (tokens, 1) not in best or (tokens, 2) not in best:
            log(f"e2e {cls}: no correct config timed, stays off")
            continue
        if lab.left() < 25:
            log(f"e2e {cls}: budget low, stays off")
            continue
        cfg1, cfg2 = best[(tokens, 1)][1], best[(tokens, 2)][1]
        cur_name = "grouped" if tokens >= GMM_FROM_TOKENS else "marlin"
        wins, ok_all = True, True
        for kind in ("real", "synth"):
            x, ids, w = re3_inputs(weights, tokens, kind)
            ref = _experts(x, w, ids, state).float()
            out = _experts_r3(x, w, ids, state, cfg1, cfg2).float()
            err = re3_rel(out, ref)
            row = float(((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-30)).max())
            ok = bool(torch.isfinite(out).all()) and err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
            t_m = lab.time_graph(lambda: _experts(x, w, ids, state), reps=5)
            t_g = lab.time_graph(lambda: _experts_gmm(x, w, ids, state), reps=5) if state["gmm"]["ok"] else float("inf")
            t_r = lab.time_graph(lambda: _experts_r3(x, w, ids, state, cfg1, cfg2), reps=5)
            cur = t_g if cur_name == "grouped" else t_m
            log(f"e2e {cls} T={tokens} {kind}: marlin {t_m:.1f} grouped {t_g:.1f} r3 {t_r:.1f}us "
                f"({cur / t_r:.2f}x vs {cur_name}) err {err:.1e} row {row:.1e}")
            wins &= t_r <= 0.97 * cur
            ok_all &= ok
        for t in RE3_TEST_TOKENS[cls]:
            err, row = r3_check(state, cfg1, cfg2, t)
            log(f"e2e {cls} test size {t}: err {err:.1e} row {row:.1e}")
            ok_all &= err < R3_MAX_ERR and row < 5 * R3_MAX_ERR
        if wins and ok_all:
            state["r3"]["cfg"][cls] = (cfg1, cfg2)
            lines[cls] = (cfg1, cfg2)
        log(f"e2e {cls}: {'R3 ON' if wins and ok_all else 'stays ' + cur_name} (wins {wins}, correct {ok_all})")
    log(f"R3_CONFIGS = {lines}")


def sec_dequant_blocks(lab, state):
    gmm = state["gmm"]
    on = torch.ones(LOCAL_EXPERTS + 1, dtype=torch.int32, device="cuda")
    row = []
    for bn, bkb, warps in ((32, 128, 4), (16, 256, 4), (64, 128, 8), (8, 512, 4), (32, 256, 8), (64, 64, 4)):
        if lab.left() < 8:
            break

        def run():
            for name in ("w13", "w2"):
                q = gmm[name + "_q"]
                _nvfp4_dequant_kernel[(q.shape[0], q.shape[1] // bn, q.shape[2] // bkb)](
                    q, gmm[name + "_sf"], on, gmm[name], N=q.shape[1], KB=q.shape[2], BN=bn, BKB=bkb, num_warps=warps)
        try:
            row.append(f"{bn}x{bkb}w{warps} {lab.time_graph(run, reps=5):.0f}us")
        except Exception as exc:
            row.append(f"{bn}x{bkb}w{warps} FAILED {type(exc).__name__}")
    log("dequant all 32 (4.13 GB): " + ", ".join(row))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"submission prepare {time.monotonic() - t0:.1f}s, grouped ok={state['gmm']['ok']}, r3 cfg={state['r3']['cfg']}")
    lab = Lab(RE3_PREPARE_LIMIT_S - (time.monotonic() - t0))
    if not state["gmm"]["ok"]:
        log("grouped path off: the lab's references need its decode, skipping")
        return state
    t = time.monotonic()
    for bk in RE3_BKS:
        r3_add_bk(state, bk)
    torch.cuda.synchronize()
    log(f"repacks for BK {RE3_BKS}: {time.monotonic() - t:.1f}s")
    top, best = {}, {}
    lab.section("sweep", sec_sweep, weights, state, top)
    lab.section("retime", sec_retime, weights, state, top, best, min_left=60.0)
    lab.section("e2e", sec_e2e, weights, state, best, min_left=20.0)
    lab.section("dequant_blocks", sec_dequant_blocks, state, min_left=8.0)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    log(f"lab total {time.monotonic() - t0:.1f}s, kernel_fn R3 classes: {sorted(state['r3']['cfg'])}")
    return state
