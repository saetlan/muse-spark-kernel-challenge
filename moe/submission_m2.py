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


# ====================================================================== M2 (moe/m2/m2.py)

"""M2: our own Triton expert kernels for small batches; v1's Marlin path keeps the larger ones.

Per call: one GEMM1 launch (gate and up together, SwiGLU in the epilogue), one GEMM2 launch, one
post-norm launch. No sort/align kernel: every program scans the (token, slot) pairs itself, finds
the local experts that got any, and works through (expert, column tile, K split) items. The
weights stay NVFP4; prepare() only permutes nibbles (m2_repack)."""

import triton.language as tl

# Tokens up to which kernel_fn may use M2; it never scans more than M2_NPAIR pairs.
M2_MAX_TOKENS = 32
M2_NPAIR = 256
# Pairs per MMA tile; tl.dot needs every dimension >= 16.
M2_TOK = 16
# Block scales are raised by 2^64 before the bf16 multiply, so code * scale stays a normal bf16
# (E1b: exact for every code and all 127 E4M3 scales); GEMM epilogues undo 2^(126 - 64).
M2_OUT_SHIFT = 2.0 ** 62
# Per-call configs by token count (fewer than this many tokens, config). Placeholder until E2 times them.
M2_DEFAULT = dict(persist=True, swap=True, bn=64, bk=256, split=1, warps=4, stages=3, progs_per_sm=2)
M2_CONFIGS = ()


def m2_repack(packed, bk):
    """uint8 [..., K/2] NVFP4 codes (element 2i in the low nibble) -> the same bytes permuted so
    that within each bk-wide K tile, byte j holds element j (low nibble) and j + bk/2 (high)."""
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).flatten(-2)
    t = codes.unflatten(-1, (codes.shape[-1] // bk, 2, bk // 2))
    return (t[..., 0, :] | (t[..., 1, :] << 4)).flatten(-2).contiguous()


@triton.jit
def _m2_dequant(q, s_lo, s_hi):
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
def _m2_scales(ptrs):
    """float8_e4m3fn block scales -> bf16 scale * 2^64 (exact)."""
    return (tl.load(ptrs).to(tl.float32) * 18446744073709551616.0).to(tl.bfloat16)


@triton.jit
def _m2_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN: tl.constexpr, HB: tl.constexpr,
             G: tl.constexpr, SCALE2D: tl.constexpr):
    """Dequantized [BN, HB] bf16 halves of the BK = 2 * HB tile at k0: (elements k0 + j, k0 + HB + j).
    Default: scales loaded [BN, G] and broadcast over 3-D [BN, G, 16] codes. SCALE2D: one scale load
    per element on 2-D tiles (simpler for the compiler, more load instructions)."""
    if SCALE2D:
        q = tl.load(w_rows[:, None] + (k0 // 2 + offs_j)[None, :])
        s_lo = _m2_scales(s_rows[:, None] + ((k0 + offs_j) // 16)[None, :])
        s_hi = _m2_scales(s_rows[:, None] + ((k0 + HB + offs_j) // 16)[None, :])
        lo, hi = _m2_dequant(q, s_lo, s_hi)
    else:
        s_lo = tl.broadcast_to(_m2_scales(s_tile + k0 // 16)[:, :, None], (BN, G, 16))
        s_hi = tl.broadcast_to(_m2_scales(s_tile + (k0 // 16 + G))[:, :, None], (BN, G, 16))
        lo, hi = _m2_dequant(tl.load(w_tile + k0 // 2), s_lo, s_hi)
        lo = tl.reshape(lo, (BN, HB))
        hi = tl.reshape(hi, (BN, HB))
    return lo, hi


@triton.jit
def _m2_item(a_ptr, w_ptr, sf_ptr, c_ptr, out_ptr, ids, e, cnt, pid_n, pid_s, n_pairs,
             N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, TOP_K: tl.constexpr,
             A_PER_PAIR: tl.constexpr, GATED: tl.constexpr, SWAP: tl.constexpr, SPLIT: tl.constexpr,
             NPAIR: tl.constexpr, TOK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
             SCALE2D: tl.constexpr):
    """Columns [pid_n * BN, +BN) of local expert e, K split pid_s, for all cnt pairs routed to e."""
    HB: tl.constexpr = BK // 2
    G: tl.constexpr = BK // 32
    KS: tl.constexpr = K // SPLIT
    offs_p = tl.arange(0, NPAIR)
    hit = ids == e
    rank = tl.cumsum(hit.to(tl.int32), axis=0) - 1
    e64 = e.to(tl.int64)
    c_e = tl.load(c_ptr + e)
    offs_n = pid_n * BN + tl.arange(0, BN)
    w_rows = w_ptr + e64 * (ROWS * (K // 2)) + offs_n.to(tl.int64) * (K // 2)
    s_rows = sf_ptr + e64 * (ROWS * (K // 16)) + offs_n.to(tl.int64) * (K // 16)
    offs_g = tl.arange(0, G)
    offs_i = tl.arange(0, 16)
    offs_j = tl.arange(0, HB)
    w_tile = w_rows[:, None, None] + (offs_g[:, None] * 16 + offs_i[None, :])[None, :, :]
    s_tile = s_rows[:, None] + offs_g[None, :]
    k_begin = pid_s * KS
    for c0 in range(0, cnt, TOK):
        slot = c0 + tl.arange(0, TOK)
        pair = tl.sum(tl.where(hit[:, None] & (rank[:, None] == slot[None, :]), offs_p[:, None], 0), axis=0)
        valid = slot < cnt
        if A_PER_PAIR:
            a_row = pair
        else:
            a_row = pair // TOP_K
        if SWAP:
            a_tile = a_ptr + a_row[None, :].to(tl.int64) * K + offs_j[:, None]
            a_mask = valid[None, :]
            acc = tl.zeros((BN, TOK), dtype=tl.float32)
            acc_u = tl.zeros((BN, TOK), dtype=tl.float32)
        else:
            a_tile = a_ptr + a_row[:, None].to(tl.int64) * K + offs_j[None, :]
            a_mask = valid[:, None]
            acc = tl.zeros((TOK, BN), dtype=tl.float32)
            acc_u = tl.zeros((TOK, BN), dtype=tl.float32)
        for k0 in range(k_begin, k_begin + KS, BK):
            a_lo = tl.load(a_tile + k0, mask=a_mask, other=0.0)
            a_hi = tl.load(a_tile + (k0 + HB), mask=a_mask, other=0.0)
            lo, hi = _m2_tile(w_rows, s_rows, w_tile, s_tile, offs_j, k0, BN, HB, G, SCALE2D)
            if SWAP:
                acc = tl.dot(lo, a_lo, acc)
                acc = tl.dot(hi, a_hi, acc)
            else:
                acc = tl.dot(a_lo, tl.trans(lo), acc)
                acc = tl.dot(a_hi, tl.trans(hi), acc)
            if GATED:
                lo, hi = _m2_tile(w_rows + N * (K // 2), s_rows + N * (K // 16), w_tile + N * (K // 2),
                                  s_tile + N * (K // 16), offs_j, k0, BN, HB, G, SCALE2D)
                if SWAP:
                    acc_u = tl.dot(lo, a_lo, acc_u)
                    acc_u = tl.dot(hi, a_hi, acc_u)
                else:
                    acc_u = tl.dot(a_lo, tl.trans(lo), acc_u)
                    acc_u = tl.dot(a_hi, tl.trans(hi), acc_u)
        if SWAP:
            rows = pair[None, :].to(tl.int64)
            cols = offs_n[:, None]
        else:
            rows = pair[:, None].to(tl.int64)
            cols = offs_n[None, :]
        if SPLIT == 1:
            if GATED:
                g = acc * c_e
                res = g * tl.sigmoid(g) * (acc_u * c_e)
            else:
                res = acc * c_e
            tl.store(out_ptr + rows * N + cols, res.to(tl.bfloat16), mask=a_mask)
        else:
            # fp32 partials [SPLIT, pairs, ROWS]: gate then up columns when GATED.
            part = out_ptr + (pid_s.to(tl.int64) * n_pairs + rows) * ROWS + cols
            tl.store(part, acc * c_e, mask=a_mask)
            if GATED:
                tl.store(part + N, acc_u * c_e, mask=a_mask)


@triton.jit(do_not_specialize=["n_pairs", "local_start"])
def _m2_gemm_kernel(a_ptr, w_ptr, sf_ptr, c_ptr, ids_ptr, out_ptr, n_pairs, local_start,
                    N: tl.constexpr, K: tl.constexpr, ROWS: tl.constexpr, LOCAL: tl.constexpr,
                    TOP_K: tl.constexpr, A_PER_PAIR: tl.constexpr, GATED: tl.constexpr,
                    SWAP: tl.constexpr, SPLIT: tl.constexpr, PERSIST: tl.constexpr,
                    NPAIR: tl.constexpr, TOK: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                    SCALE2D: tl.constexpr):
    """One expert GEMM over every local pair (see _m2_item). PERSIST: a fixed grid that loops over
    the items of the experts that got pairs; otherwise one program per (expert, tile, split)."""
    ITEMS: tl.constexpr = (N // BN) * SPLIT
    offs_p = tl.arange(0, NPAIR)
    ids = tl.load(ids_ptr + offs_p, mask=offs_p < n_pairs, other=-1) - local_start
    if PERSIST:
        offs_e = tl.arange(0, LOCAL)
        counts = tl.sum((ids[:, None] == offs_e[None, :]).to(tl.int32), axis=0)
        active = counts > 0
        act_rank = tl.cumsum(active.to(tl.int32), axis=0) - 1
        total = tl.sum(active.to(tl.int32), axis=0) * ITEMS
        for item in range(tl.program_id(0), total, tl.num_programs(0)):
            a_idx = item // ITEMS
            rem = item % ITEMS
            sel = active & (act_rank == a_idx)
            e = tl.sum(tl.where(sel, offs_e, 0), axis=0)
            cnt = tl.sum(tl.where(sel, counts, 0), axis=0)
            _m2_item(a_ptr, w_ptr, sf_ptr, c_ptr, out_ptr, ids, e, cnt, rem // SPLIT, rem % SPLIT, n_pairs,
                     N, K, ROWS, TOP_K, A_PER_PAIR, GATED, SWAP, SPLIT, NPAIR, TOK, BN, BK, SCALE2D)
    else:
        pid = tl.program_id(0)
        e = pid // ITEMS
        rem = pid % ITEMS
        cnt = tl.sum((ids == e).to(tl.int32), axis=0)
        if cnt > 0:
            _m2_item(a_ptr, w_ptr, sf_ptr, c_ptr, out_ptr, ids, e, cnt, rem // SPLIT, rem % SPLIT, n_pairs,
                     N, K, ROWS, TOP_K, A_PER_PAIR, GATED, SWAP, SPLIT, NPAIR, TOK, BN, BK, SCALE2D)


@triton.jit(do_not_specialize=["n_pairs", "local_start"])
def _m2_silu_reduce_kernel(part_ptr, act_ptr, ids_ptr, n_pairs, local_start, N: tl.constexpr,
                           LOCAL: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    """act[p] = silu(sum of gate partials) * sum of up partials, for local pairs p only."""
    p = tl.program_id(0)
    e = tl.load(ids_ptr + p) - local_start
    if (e >= 0) & (e < LOCAL):
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        g = tl.zeros((BLOCK,), dtype=tl.float32)
        u = tl.zeros((BLOCK,), dtype=tl.float32)
        for s in tl.static_range(SPLIT):
            row = part_ptr + (s * n_pairs + p).to(tl.int64) * (2 * N)
            g += tl.load(row + offs)
            u += tl.load(row + N + offs)
        tl.store(act_ptr + p.to(tl.int64) * N + offs, (g * tl.sigmoid(g) * u).to(tl.bfloat16))


@triton.jit(do_not_specialize=["n_pairs", "local_start"])
def _m2_post_norm_kernel(y_ptr, nw_ptr, tw_ptr, ids_ptr, out_ptr, n_pairs, local_start, eps,
                         H: tl.constexpr, LOCAL: tl.constexpr, TOP_K: tl.constexpr, SPLIT: tl.constexpr):
    """out[t] = sum over local slots k of tw[t, k] * rmsnorm(bf16(y[t, k]) * nw), as SGLang's fused
    kernel (weight first); y is bf16 [pairs, H], or SPLIT fp32 partials summed and rounded to bf16."""
    t = tl.program_id(0)
    offs = tl.arange(0, H)
    w = tl.load(nw_ptr + offs).to(tl.float32)
    acc = tl.zeros((H,), dtype=tl.float32)
    for k in tl.static_range(TOP_K):
        p = t * TOP_K + k
        e = tl.load(ids_ptr + p) - local_start
        if (e >= 0) & (e < LOCAL):
            if SPLIT == 1:
                x = tl.load(y_ptr + p.to(tl.int64) * H + offs).to(tl.float32)
            else:
                x = tl.zeros((H,), dtype=tl.float32)
                for s in tl.static_range(SPLIT):
                    x += tl.load(y_ptr + (s * n_pairs + p).to(tl.int64) * H + offs)
                x = x.to(tl.bfloat16).to(tl.float32)
            y = x * w
            var = tl.sum(y * y, axis=0) / H
            acc += y * tl.rsqrt(var + eps) * tl.load(tw_ptr + p)
    tl.store(out_ptr + t.to(tl.int64) * H + offs, acc.to(tl.bfloat16))


def m2_bank(weights, start, count, bks):
    """M2's weights for experts [start, start + count): codes repacked once per K tile in bks."""
    sl = slice(start, start + count)
    bank = {"start": start, "count": count,
            "w13_sf": weights["w13_sf"][sl].contiguous(), "w2_sf": weights["w2_sf"][sl].contiguous(),
            "c13": (weights["w13_gs"][sl].float() * M2_OUT_SHIFT).contiguous(),
            "c2": (weights["w2_gs"][sl].float() * M2_OUT_SHIFT).contiguous()}
    for bk in bks:
        bank[("w13", bk)] = torch.stack([m2_repack(weights["w13"][e], bk) for e in range(start, start + count)])
        bank[("w2", bk)] = torch.stack([m2_repack(weights["w2"][e], bk) for e in range(start, start + count)])
    return bank


def m2_gemm(a, bank, name, out, ids, n_pairs, gated, a_per_pair, cfg, sms):
    """Launch one M2 GEMM: name "w13" (gated, x -> act) or "w2" (act -> y). Returns Triton's
    compiled-kernel handle."""
    w = bank[(name, cfg["bk"])]
    sf = bank[name + "_sf"]
    rows, k = w.shape[1], w.shape[2] * 2
    n = rows // 2 if gated else rows
    count = bank["count"]
    items = count * (n // cfg["bn"]) * cfg["split"]
    grid = (min(items, cfg["progs_per_sm"] * sms),) if cfg["persist"] else (items,)
    return _m2_gemm_kernel[grid](
        a, w, sf, bank["c13" if gated else "c2"], ids, out, n_pairs, bank["start"],
        N=n, K=k, ROWS=rows, LOCAL=count, TOP_K=TOP_K, A_PER_PAIR=a_per_pair, GATED=gated,
        SWAP=cfg["swap"], SPLIT=cfg["split"], PERSIST=cfg["persist"], NPAIR=M2_NPAIR, TOK=M2_TOK,
        BN=cfg["bn"], BK=cfg["bk"], SCALE2D=cfg.get("scale2d", False), num_warps=cfg["warps"],
        num_stages=cfg["stages"])


def m2_stage1(x, ids, bank, cfg, sms):
    """x [tokens, H] -> act bf16 [pairs, I] = silu(gate) * up for every local pair."""
    n_pairs = ids.numel()
    inter = bank[("w13", cfg["bk"])].shape[1] // 2
    act = torch.empty((n_pairs, inter), dtype=torch.bfloat16, device=x.device)
    if cfg["split"] == 1:
        m2_gemm(x, bank, "w13", act, ids, n_pairs, True, False, cfg, sms)
        return act
    part = torch.empty((cfg["split"], n_pairs, 2 * inter), dtype=torch.float32, device=x.device)
    m2_gemm(x, bank, "w13", part, ids, n_pairs, True, False, cfg, sms)
    _m2_silu_reduce_kernel[(n_pairs, inter // 1024)](
        part, act, ids, n_pairs, bank["start"], N=inter, LOCAL=bank["count"], SPLIT=cfg["split"],
        BLOCK=1024, num_warps=4)
    return act


def m2_stage2(act, ids, topk_weights, norm_w, bank, cfg, sms, tokens):
    """act [pairs, I] -> out bf16 [tokens, H]: GEMM2, then the post-expert norm and weighted sum."""
    n_pairs = ids.numel()
    hidden = bank[("w2", cfg["bk"])].shape[1]
    shape = (n_pairs, hidden) if cfg["split"] == 1 else (cfg["split"], n_pairs, hidden)
    y = torch.empty(shape, dtype=torch.bfloat16 if cfg["split"] == 1 else torch.float32, device=act.device)
    m2_gemm(act, bank, "w2", y, ids, n_pairs, False, True, cfg, sms)
    out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=act.device)
    _m2_post_norm_kernel[(tokens,)](
        y, norm_w, topk_weights, ids, out, n_pairs, bank["start"], POST_NORM_EPS, H=hidden,
        LOCAL=bank["count"], TOP_K=TOP_K, SPLIT=cfg["split"], num_warps=8)
    return out


def m2_run(x, topk_ids, topk_weights, state, cfg1, cfg2):
    bank = state["m2"]
    ids = topk_ids.reshape(-1).contiguous()
    act = m2_stage1(x, ids, bank, cfg1, state["sms"])
    return m2_stage2(act, ids, topk_weights.contiguous(), state["norm_w"], bank, cfg2, state["sms"], x.shape[0])


def m2_configs_for(state, tokens):
    """(cfg1, cfg2) for this token count, or None for v1. A class whose cfg1 is None stays on v1."""
    for below, cfg1, cfg2 in state["m2_configs"]:
        if tokens < below:
            return None if cfg1 is None else (cfg1, cfg2)
    return None


def m2_warm(state, cfg1, cfg2):
    """Compile and load every kernel of (cfg1, cfg2) before any CUDA graph capture."""
    device = state["norm_w"].device
    for tokens in (1, 2):
        x = torch.zeros((tokens, state["norm_w"].shape[0]), dtype=torch.bfloat16, device=device)
        ids = torch.arange(tokens * TOP_K, dtype=torch.int32, device=device).view(tokens, TOP_K) % state["m2"]["count"]
        tw = torch.full((tokens, TOP_K), 1.0 / TOP_K, dtype=torch.float32, device=device)
        m2_run(x, ids + state["m2"]["start"], tw, state, cfg1, cfg2)
    torch.cuda.synchronize()


def prepare(weights):
    state = _v1_prepare(weights)
    state["sms"] = torch.cuda.get_device_properties(weights["w13"].device).multi_processor_count
    bks = {M2_DEFAULT["bk"]} | {c["bk"] for _, c1, c2 in M2_CONFIGS if c1 is not None for c in (c1, c2)}
    state["m2"] = m2_bank(weights, 0, 32, sorted(bks))
    state["m2_configs"] = M2_CONFIGS
    for _, cfg1, cfg2 in M2_CONFIGS:
        if cfg1 is not None:
            m2_warm(state, cfg1, cfg2)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    x = data["hidden_states"].contiguous()
    bank = state["m2"]
    cfgs = m2_configs_for(state, x.shape[0])
    if (cfgs is None or x.shape[0] > M2_MAX_TOKENS
            or (cfg["local_start"], cfg["local_count"]) != (bank["start"], bank["count"])):
        return _v1_kernel_fn(data, state)
    return m2_run(x, data["topk_ids"], data["topk_weights"], state, *cfgs)
