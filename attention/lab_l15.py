# okc: problem=muse_attention_decode
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


class Config(TypedDict):
    batch: int            # B
    q_len: int            # L: 1 for decode, 8 for a speculative verify block (causal inside it)
    num_q_heads: int      # Hq: 16 per GPU under TP8
    num_kv_heads: int     # Hkv: 2; each serves Hq / Hkv query heads
    head_dim: int         # 64
    page_size: int        # 1 in the scored cases (as SGLang serves Muse on H200), 64 in some tests
    window: int           # 2048 on sliding-window layers (a token and the 2,047 before it); 0 = all
    rope_theta: float     # 500000 on sliding-window layers; 0 = no RoPE
    qk_scale_factor: float
    sm_scale: float       # 1/8
    kv_scale: float       # the cache scales' value: 1 in the scored cases (SGLang leaves them unset)


class Inputs(TypedDict):
    qkv: torch.Tensor          # bfloat16 [B, L, (2*Hq + 2*Hkv)*64]
    positions: torch.Tensor    # int64 [B*L]
    cache_loc: torch.Tensor    # int64 [B*L]
    cache_seqlens: torch.Tensor  # int32 [B], seq_lens + L
    cu_seqlens_q: torch.Tensor   # int32 [B+1]
    scheduler_metadata: torch.Tensor | None
    k_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    v_cache: torch.Tensor     # float8_e4m3fn [pages, page_size, Hkv, 64], written in place
    k_scale: torch.Tensor     # float32 []
    v_scale: torch.Tensor     # float32 []
    page_table: torch.Tensor  # int32 [B, max_pages]
    seq_lens: torch.Tensor    # int32 [B]
    config: Config


RMS_EPS = 1e-5
LOG2E = 1.4426950408889634
NUM_SMS = 132
MAX_SPLITS = 512
_INV_FREQ = {}
_WARM = []            # non-empty once every scored variant is compiled and loaded

# Launch settings per shape class, keyed by (full layer, verify block, batch class 1/16/32):
# (target programs per SM, min keys per slice, keys per step, warps, pipeline stages).
# From labs L1 (check b8c59d13124e40b6) and L2 (d0ee39d19b534734): best per scored shape under
# graphs + cold L2.
TUNE = {
    (True, False, 1): (32, 256, 64, 4, 3),
    (True, False, 16): (16, 256, 64, 4, 5),
    (True, False, 32): (16, 256, 64, 4, 3),
    (True, True, 1): (2, 256, 64, 4, 2),
    (True, True, 16): (4, 256, 64, 4, 2),
    (True, True, 32): (8, 256, 64, 4, 2),
    (False, False, 1): (16, 64, 64, 4, 2),
    (False, False, 16): (16, 128, 64, 4, 2),
    (False, False, 32): (16, 256, 64, 4, 2),
    (False, True, 1): (16, 64, 32, 4, 1),
    (False, True, 16): (16, 64, 32, 4, 1),
    (False, True, 32): (16, 256, 64, 4, 2),
}


def batch_class(batch):
    return 1 if batch <= 1 else 16 if batch <= 16 else 32


@triton.jit
def _prologue(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
              k_scale_ptr, v_scale_ptr,
              HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr,
              QK_FACTOR: tl.constexpr, EPS: tl.constexpr):
    t = tl.program_id(0)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    base = qkv_ptr + t.to(tl.int64) * ROW
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    i = tl.arange(0, 32)
    # Even and odd lanes of every head: RoPE rotates the pairs (2i, 2i+1).
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
        q_e, q_o = q_e * c - q_o * s, q_e * s + q_o * c
        k_e, k_o = k_e * c - k_o * s, k_e * s + k_o * c

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
def _head_step(q, k, v, m_i, l_i, acc, vis, qk_scale):
    """One online-softmax step for one KV head: q [MP, 64], k and v [BLOCK, 64] BF16."""
    sc = tl.dot(q, tl.trans(k)) * qk_scale
    sc = tl.where(vis, sc, float("-inf"))
    m_new = tl.maximum(m_i, tl.max(sc, 1))
    alpha = tl.exp2(m_i - m_new)
    p = tl.exp2(sc - m_new[:, None])
    l_i = l_i * alpha + tl.sum(p, 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
    return m_new, l_i, acc


@triton.jit(do_not_specialize=["max_pages", "nsplit"])
def _attend(q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, page_table_ptr, seqlens_ptr,
            o_part_ptr, m_part_ptr, l_part_ptr, max_pages, nsplit,
            HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, PAGE: tl.constexpr,
            WINDOW: tl.constexpr, SM_SCALE: tl.constexpr, MP: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    kv0 = tl.program_id(1) * 2
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    ML: tl.constexpr = L * G        # query rows per KV head: block token x group head

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

    n = tl.load(seqlens_ptr + b)    # keys 0 .. n-1; row l sits at position n - L + l
    row_pos = n - L + r_l
    if WINDOW > 0:
        lo = tl.maximum(n - L - WINDOW + 1, 0)
    else:
        lo = 0
    per = tl.cdiv(tl.cdiv(n - lo, nsplit), BLOCK) * BLOCK
    start = lo + split * per
    end = tl.minimum(start + per, n)

    qk_scale = SM_SCALE * tl.load(k_scale_ptr) * 1.4426950408889634
    m0 = tl.full([MP], -1.0e30, tl.float32)
    l0 = tl.zeros([MP], tl.float32)
    a0 = tl.zeros([MP, 64], tl.float32)
    m1 = tl.full([MP], -1.0e30, tl.float32)
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
        m0, l0, a0 = _head_step(q0, k0, v0, m0, l0, a0, vis, qk_scale)
        m1, l1, a1 = _head_step(q1, k1, v1, m1, l1, a1, vis, qk_scale)

    # Partials per (token, head, slice).
    p0 = (tok * HQ + head0) * nsplit + split
    p1 = (tok * HQ + head1) * nsplit + split
    tl.store(o_part_ptr + p0[:, None] * 64 + d[None, :], a0, mask=r_ok[:, None])
    tl.store(o_part_ptr + p1[:, None] * 64 + d[None, :], a1, mask=r_ok[:, None])
    tl.store(m_part_ptr + p0, m0, mask=r_ok)
    tl.store(m_part_ptr + p1, m1, mask=r_ok)
    tl.store(l_part_ptr + p0, l0, mask=r_ok)
    tl.store(l_part_ptr + p1, l1, mask=r_ok)


@triton.jit(do_not_specialize=["nsplit"])
def _combine(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit,
             HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    big = tl.max(tl.full([CB], -1.0e30, tl.float32), 0)   # fp32 scalars: loop-carried types fixed
    den = tl.sum(tl.zeros([CB], tl.float32), 0)
    num = tl.zeros([64], tl.float32)
    for s0 in range(0, nsplit, CB):       # one pass: running max, rescaled sums
        ok = s0 + j < nsplit
        idx = th * nsplit + s0 + j
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1.0e30)
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


def plan(cfg, max_pages, tune=None):
    """Launch settings from the config and shapes only (never from lengths or values):
    (slices per request, keys per step, warps, stages, combine block, query rows)."""
    full, verify = not cfg["window"], cfg["q_len"] > 1
    per_sm, min_keys, block, warps, stages = tune or TUNE[(full, verify, batch_class(cfg["batch"]))]
    if cfg["num_kv_heads"] % 2:
        raise ValueError("this kernel pairs KV heads: num_kv_heads must be even (tp8: 2, dp: 16)")
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    nsplit = max(1, min(math.ceil(per_sm * NUM_SMS / programs), tokens // min_keys, MAX_SPLITS))
    cb = min(64, triton.next_power_of_2(nsplit))
    rows = cfg["q_len"] * cfg["num_q_heads"] // cfg["num_kv_heads"]
    mp = max(16, triton.next_power_of_2(rows))
    return nsplit, block, warps, stages, cb, mp


def _inv_freq(theta, device):
    key = (theta, str(device))
    if key not in _INV_FREQ:
        # As SGLang's RotaryEmbedding and the reference build it, in FP32 on the GPU.
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    return _INV_FREQ[key]


def run(data, tune=None):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = batch * length
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    nsplit, block, warps, stages, cb, mp = plan(cfg, max_pages, tune)
    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"] if rope else 1.0, device)

    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(tokens,)](qkv, data["positions"], data["cache_loc"], inv_freq, q,
                         data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                         HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS,
                         num_warps=4)

    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _attend[(batch, hkv // 2, nsplit)](
        q, data["k_cache"], data["v_cache"], data["k_scale"], page_table, data["cache_seqlens"],
        o_part, m_part, l_part, max_pages, nsplit,
        HQ=hq, HKV=hkv, L=length, PAGE=cfg["page_size"], WINDOW=cfg["window"],
        SM_SCALE=cfg["sm_scale"], MP=mp, BLOCK=block, num_warps=warps, num_stages=stages)

    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=4)
    return out


# ---- SGLang's path (baseline.py, call for call): the fallback for shape classes where our
# kernels haven't been measured faster on the judge. ----
FP8 = torch.float8_e4m3fn
_SG_NORMS = {}
_SG_ROPE = {}
SG_ROPE_POSITIONS = 1 << 21


def _sg_qk_norm(hq, hkv, d):
    key = (hq, hkv, d)
    if key not in _SG_NORMS:
        _SG_NORMS[key] = FusedQKNorm(head_dim=d, num_q_heads=hq, num_kv_heads=hkv, eps=RMS_EPS)
    return _SG_NORMS[key]


def _sg_cos_sin(theta, d, device):
    cache = _SG_ROPE.get(theta)
    if cache is None:
        inv_freq = 1.0 / (theta ** (torch.arange(0, d, 2, dtype=torch.float, device=device) / d))
        freqs = torch.einsum("i,j -> ij", torch.arange(SG_ROPE_POSITIONS, dtype=torch.float, device=device), inv_freq)
        cache = _SG_ROPE[theta] = torch.cat((freqs.cos(), freqs.sin()), dim=-1)
    return cache


def sglang_path(data):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv, d = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["head_dim"]
    tokens = batch * length
    qkv = data["qkv"].view(tokens, -1)
    device = qkv.device

    if cfg["rope_theta"]:
        q, k = _sg_qk_norm(hq, hkv, d)(qkv, data["positions"], _sg_cos_sin(cfg["rope_theta"], d, device))
    else:
        q, k = _sg_qk_norm(hq, hkv, d)(qkv)
    v = qkv[:, (hq + hkv) * d:(hq + 2 * hkv) * d]

    k_cache, v_cache = data["k_cache"], data["v_cache"]
    scaled = cfg["kv_scale"] != 1.0
    k = k.reshape(tokens, hkv * d)
    if scaled:
        k, v = k / data["k_scale"], v / data["v_scale"]
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
    gate = qkv[:, (hq + 2 * hkv) * d:]
    o = rms_norm_gate(out.reshape(tokens, hq * d).to(torch.bfloat16), gate, None, head_dim=d, eps=RMS_EPS,
                      has_norm=True)
    return o.view(batch, length, hq * d)


# Hybrid dispatch (CLAUDE.md method 3): our kernels only for the shape classes, keyed by (full
# layer, verify block, batch class), where a judge run measured them faster; SGLang otherwise.
# Judge times, L2 check d0ee39d19b534734 (ours vs SGLang, us): Sd 9.6/21.0, 15.4/27.1, 21.5/31.6;
# Sv 14.9/22.8, 22.8/29.1, 36.0/34.5 (SGLang); Fd 24.9/31.3, 195/201 (5 stages: lab -10%),
# 364/383; Fv 33.0/36.3, 335/206 (SGLang), 649/390 (SGLang).
OURS = {
    (False, False, 1), (False, False, 16), (False, False, 32),
    (False, True, 1), (False, True, 16),
    (True, False, 1), (True, False, 16), (True, False, 32),
    (True, True, 1),
}


def shape_class(cfg):
    return (not cfg["window"], cfg["q_len"] > 1, batch_class(cfg["batch"]))


def kernel_fn(data):
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    if shape_class(data["config"]) in OURS:
        return run(data)
    return sglang_path(data)


def scored_max_pages(batch, length):
    """The scored cases' page-table width: 80,000-token context, page 1, fixed length at
    batch 1, agentic spread (+7.5%) above (reference.py _longest)."""
    longest = 80000 if batch == 1 else math.ceil(80000 * 1.075)
    return longest + length


def dummy_inputs(batch, length, window, max_pages, device, ctx=300, hq=16, hkv=2):
    """Small inputs with the scored shapes' config and page-table width (so plan() picks the
    same launch settings), used to compile every variant before any capture."""
    pages = batch * (ctx + length)
    cfg = dict(batch=batch, q_len=length, num_q_heads=hq, num_kv_heads=hkv, head_dim=64, page_size=1,
               window=window, rope_theta=500000.0 if window else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0)
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=device)
    table = (torch.arange(batch * max_pages, dtype=torch.int32, device=device) % pages).view(batch, max_pages)
    pos = (seq.long()[:, None] + torch.arange(length, device=device)).reshape(-1)
    return dict(
        qkv=torch.randn((batch, length, (2 * hq + 2 * hkv) * 64), device=device).to(torch.bfloat16),
        positions=pos,
        cache_loc=torch.gather(table.long(), 1, pos.view(batch, length)).reshape(-1),
        cache_seqlens=seq + length,
        cu_seqlens_q=torch.arange(0, batch * length + 1, length, dtype=torch.int32, device=device),
        scheduler_metadata=None,
        k_cache=torch.zeros((pages, 1, hkv, 64), device=device).to(torch.float8_e4m3fn),
        v_cache=torch.zeros((pages, 1, hkv, 64), device=device).to(torch.float8_e4m3fn),
        k_scale=torch.tensor(1.0, device=device), v_scale=torch.tensor(1.0, device=device),
        page_table=table, seq_lens=seq, config=cfg)


def _warm_up():
    """Compile and load every scored variant on the first eager call (a correctness test),
    so no first launch (Triton compile + module load) can land inside a CUDA graph capture.
    Not at import: CUDA stays uninitialised until the harness first calls us."""
    _WARM.append(True)
    dev = torch.device("cuda")
    for length in (1, 8):
        for window in (2048, 0):
            for batch in (1, 16, 32):
                data = dummy_inputs(batch, length, window, scored_max_pages(batch, length), dev)
                if shape_class(data["config"]) in OURS:
                    run(data)
                else:
                    sglang_path(data)


# ===== appended: attention/local/cuda_dec2.py =====
"""A5b: CUDA split-context decode attention (q_len 1) reading whole 128-byte cache rows (A17)
with both KV heads of a pair stacked in one MMA (A18), and page rows fetched two chunks ahead
(A21). Source + declaration; make_lab.py appends this file to lab files.

One CTA = (request b, KV-head pair kv0/kv0+1, slice of keys). The pair's two 64-byte head rows
are one contiguous 128-byte piece of every cache row (whole row under tp8). Warps take 16-token
chunks round-robin, each with its own online softmax, and merge through shared memory at the end.

Stacked MMA (mma.m16n8k16, f32 += f16 x f16), lane (g, t4) = (lane >> 2, lane & 3):
- QK: A rows g = head-0 query g, rows g+8 = head-1 query g; K-dim = the pair's 128 bytes in 8
  k-steps. Logical column 2*t4+{0,1} (+8) of k-step ks <-> physical byte 32*t4 + 4*ks + {0,1}
  ({2,3}); bytes 0-63 are head 0, 64-127 head 1, so lanes t4 = 0,1 carry head-0 columns and
  t4 = 2,3 head-1 columns: the queries are block-diagonal (row g zero on head-1 columns, row g+8
  zero on head-0 columns). B = keys: lane (g, t4) holds bytes [32*t4, 32*t4+32) of token g's
  piece (tile 0) and token 8+g's (tile 1) as two 16 B loads each; every load instruction of a
  warp touches all four 32 B sectors of each row, so each row is fetched whole.
  C: c0,c1 = head-0 scores of query g, c2,c3 = head-1 scores, tokens 2*t4+{0,1} of the tile.
- PV: A = P (rows g head 0, g+8 head 1; tokens 0-15) straight from the QK accumulators.
  B = values over 16 d-tiles of the 128-byte piece: logical column g of d-tile j <-> physical
  byte 16*g + j, so lane (g, t4) loads 16 contiguous bytes [16*g, 16*g+16) of tokens 2*t4,
  2*t4+1, 2*t4+8, 2*t4+9 (8 lanes cover each row: whole-row loads), packed with __byte_perm.
  C: column n = 2*t4+jj of tile j <-> byte 16*(2*t4+jj) + j: lanes t4 = 0,1 keep c0,c1 (head 0,
  d = 32*t4 + 16*jj + j), lanes t4 = 2,3 keep c2,c3 (head 1, d = 32*(t4-2) + 16*jj + j).
FP8 E4M3 -> FP16 is exact; queries FP16 (finer than the FP8 the rules allow); P FP16; FP32
accumulation. Partials (log2-domain max, sum, output) in _combine's layout.
"""

CUDA_DEC2_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_dec2 {

constexpr int CHUNK = 16;
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attend_dec2_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j
      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));
      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));
      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}
      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8
      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}
      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  // Row of token (start + c*CHUNK + (lane & 15)) of chunk c, clamped into the slice.
  auto row_of = [&](int c) -> long {
    const int s = min(start + c * CHUNK + (lane & 15), end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto load = [&](long my_row, Chunk& ch) {
    const long r0 = __shfl_sync(FULL, my_row, g);
    const long r1 = __shfl_sync(FULL, my_row, 8 + g);
    const long v0 = __shfl_sync(FULL, my_row, 2 * t4);
    const long v1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);
    const long v2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);
    const long v3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);
    const uint8_t* kb0 = kc + r0 * row_bytes + poff + 32 * t4;
    const uint8_t* kb1 = kc + r1 * row_bytes + poff + 32 * t4;
    ch.k[0] = __ldg(reinterpret_cast<const uint4*>(kb0));
    ch.k[2] = __ldg(reinterpret_cast<const uint4*>(kb1));
    ch.k[1] = __ldg(reinterpret_cast<const uint4*>(kb0 + 16));
    ch.k[3] = __ldg(reinterpret_cast<const uint4*>(kb1 + 16));
    ch.v[0] = __ldg(reinterpret_cast<const uint4*>(vc + v0 * row_bytes + poff + 16 * g));
    ch.v[1] = __ldg(reinterpret_cast<const uint4*>(vc + v1 * row_bytes + poff + 16 * g));
    ch.v[2] = __ldg(reinterpret_cast<const uint4*>(vc + v2 * row_bytes + poff + 16 * g));
    ch.v[3] = __ldg(reinterpret_cast<const uint4*>(vc + v3 * row_bytes + poff + 16 * g));
  };

  // Pipeline per warp: rows two chunks ahead, data one chunk ahead.
  Chunk cur, nxt;
  long row_next = 0, row_next2 = 0;
  int c = warp;
  if (c < nchunks) {
    load(row_of(c), cur);
    if (c + WARPS < nchunks) row_next = row_of(c + WARPS);
  }
  for (; c < nchunks; c += WARPS) {
    if (c + 2 * WARPS < nchunks) row_next2 = row_of(c + 2 * WARPS);
    if (c + WARPS < nchunks) load(row_next, nxt);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    cur = nxt;
    row_next = row_next2;
  }

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }
}

}  // namespace okc_dec2

void attend_dec2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_DEC2_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (warps == 8) okc_dec2::attend_dec2_kernel<8><<<grid, 256, 0, stream>>>(OKC_DEC2_ARGS);
  else okc_dec2::attend_dec2_kernel<4><<<grid, 128, 0, stream>>>(OKC_DEC2_ARGS);
#undef OKC_DEC2_ARGS
}

std::vector<int64_t> attend_dec2_attrs(int64_t warps) {
  cudaFuncAttributes a;
  if (warps == 8) cudaFuncGetAttributes(&a, okc_dec2::attend_dec2_kernel<8>);
  else cudaFuncGetAttributes(&a, okc_dec2::attend_dec2_kernel<4>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_DEC2_DECL = ("void attend_dec2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                  "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                  "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                  "int64_t page, int64_t window, double sm_scale, int64_t warps);\n"
                  "std::vector<int64_t> attend_dec2_attrs(int64_t warps);")


# ===== appended: attention/local/cuda_dec3.py =====
"""D2: A5b's stacked-head decode attention (cuda_dec2.py: whole 128-byte rows, both KV heads of a
pair in one MMA, the same fragment permutation) fed from a warp-private cp.async ring instead of
register prefetch (AF21/AF24: register-direct kernels plateau at ~2 TB/s; L5's warp-private
rings reached 3.79 TB/s).

Each warp streams its own 16-token chunks (c = warp, warp+WARPS, ...) through STAGES stages of
16 tokens x (K row 128 B, V row 128 B) at a 144-byte pitch: lane copies 16 B for token
4*q + (lane >> 3), byte chunk lane & 7 (q = 0..3, K and V), so every copy instruction covers
whole rows. Page rows come one iteration ahead (shuffled from lanes 0..15), so address
generation never waits. Only cp.async.wait_group + __syncwarp synchronize; no CTA barrier in
the loop. Fragments are read from shared memory exactly where A5b loaded them from global:
K bytes 32*t4 (+16) of tokens g / 8+g, V bytes 16*g of tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9.
Pitch 144 B = 9 units of 16 B makes the 16 B reads and the row copies bank-conflict free
(unit = 9*token + chunk mod 8 is distinct in every 8-lane phase). The ring is dynamic shared
memory (attend_dec3_setup raises the limit once, outside any capture).
"""

CUDA_DEC3_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_dec3 {

constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32) attend_dec3_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j
      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));
      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));
      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}
      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8
      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}
      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Row of token (start + c*CHUNK + (lane & 15)) of the warp's i-th chunk, clamped into the slice.
  auto row_of = [&](int i) -> long {
    const int s = min(start + (warp + i * WARPS) * CHUNK + (lane & 15), end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto issue = [&](int i, long my_row) {
    if (i < mine) {
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  long row_next = 0;
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i, i < mine ? row_of(i) : 0);
  if (STAGES - 1 < mine) row_next = row_of(STAGES - 1);
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1, row_next);
    if (i + STAGES < mine) row_next = row_of(i + STAGES);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const int c = warp + i * WARPS;
    const uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  asm volatile("cp.async.wait_all;");

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }
}

}  // namespace okc_dec3

void attend_dec3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_dec3::STAGE_BYTES;
#define OKC_DEC3_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (warps == 8 && stages == 2) okc_dec3::attend_dec3_kernel<8, 2><<<grid, 256, smem, stream>>>(OKC_DEC3_ARGS);
  else if (warps == 8) okc_dec3::attend_dec3_kernel<8, 3><<<grid, 256, smem, stream>>>(OKC_DEC3_ARGS);
  else if (stages == 4) okc_dec3::attend_dec3_kernel<4, 4><<<grid, 128, smem, stream>>>(OKC_DEC3_ARGS);
  else okc_dec3::attend_dec3_kernel<4, 3><<<grid, 128, smem, stream>>>(OKC_DEC3_ARGS);
#undef OKC_DEC3_ARGS
}

int64_t attend_dec3_setup() {   // once, outside any capture: allow each variant's ring
  // Each limit is exactly the ring a variant needs (static + dynamic must stay <= 227 KB per
  // block); the last-error state is cleared so a refusal can't surface in a later PyTorch call.
  int err = 0;
  using okc_dec3::STAGE_BYTES;
  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 3 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec3::attend_dec3_kernel<8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 3 * STAGE_BYTES);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_dec3_attrs(int64_t warps, int64_t stages) {
  cudaFuncAttributes a;
  if (warps == 8 && stages == 2) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<8, 2>);
  else if (warps == 8) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<8, 3>);
  else if (stages == 4) cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<4, 4>);
  else cudaFuncGetAttributes(&a, okc_dec3::attend_dec3_kernel<4, 3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_DEC3_DECL = ("void attend_dec3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                  "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                  "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                  "int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages);\n"
                  "int64_t attend_dec3_setup();\n"
                  "std::vector<int64_t> attend_dec3_attrs(int64_t warps, int64_t stages);")


# ===== appended: attention/local/cuda_dec4.py =====
"""D3: D2 (cuda_dec3.py) with its page entries carried by the cp.async ring (generated by
make_pt_ring.py; see its docstring): commit group i = chunk i's K/V rows + chunk i+STAGES's
page entries (4-byte copies into the K tiles' unused row pads), so no iteration waits on a
dependent page-table load; the warp merge reuses the ring instead of static shared memory.
Math, chunk assignment and fragment layout are D2's (emulate_cuda_dec3.py applies).
"""

CUDA_DEC4_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_dec4 {

constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32) attend_dec4_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j
      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));
      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));
      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}
      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8
      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}
      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
  // Page entries ride the ring: chunk j's 16 entries sit in PT slot p = j % PT_SLOTS, in the unused
  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at
  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.
  constexpr int PT_SLOTS = 2 * STAGES;
  auto pt_slot = [&](int j, int e) -> int* {
    const int p = j % PT_SLOTS;
    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));
  };
  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j's page entries (join the open commit group)
    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);
  };
  // One commit group per call: chunk i's K/V rows (its page entries landed in an earlier group,
  // waited and warp-synced) and chunk i + STAGES's page entries.
  auto issue = [&](int i) {
    if (i < mine) {
      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    issue_pt(i + STAGES);
    asm volatile("cp.async.commit_group;");
  };
  // Prologue: the first STAGES chunks' page entries (the one exposed round trip), then the first
  // STAGES - 1 stages of K/V rows.
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const int c = warp + i * WARPS;
    const uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }
}

}  // namespace okc_dec4

void attend_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_dec4::STAGE_BYTES;
#define OKC_DEC4_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (warps == 8 && stages == 2) okc_dec4::attend_dec4_kernel<8, 2><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);
  else if (warps == 8) okc_dec4::attend_dec4_kernel<8, 3><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);
  else if (stages == 4) okc_dec4::attend_dec4_kernel<4, 4><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);
  else okc_dec4::attend_dec4_kernel<4, 3><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);
#undef OKC_DEC4_ARGS
}

int64_t attend_dec4_setup() {   // once, outside any capture: allow each variant's ring
  // Each limit is exactly the ring a variant needs (static + dynamic must stay <= 227 KB per
  // block); the last-error state is cleared so a refusal can't surface in a later PyTorch call.
  int err = 0;
  using okc_dec4::STAGE_BYTES;
  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 3 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_dec4::attend_dec4_kernel<8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 3 * STAGE_BYTES);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_dec4_attrs(int64_t warps, int64_t stages) {
  cudaFuncAttributes a;
  if (warps == 8 && stages == 2) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<8, 2>);
  else if (warps == 8) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<8, 3>);
  else if (stages == 4) cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<4, 4>);
  else cudaFuncGetAttributes(&a, okc_dec4::attend_dec4_kernel<4, 3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_DEC4_DECL = ("void attend_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                  "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                  "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                  "int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages);\n"
                  "int64_t attend_dec4_setup();\n"
                  "std::vector<int64_t> attend_dec4_attrs(int64_t warps, int64_t stages);")


# ===== appended: attention/local/cuda_unified.py =====
"""UD: unified decode kernel (q_len 1), one launch per layer (B2: A40-A43). D2's warp-private
cp.async ring streaming (cuda_dec3.py) with A5b's stacked-head MMA, the fused kernel's prologue
fold, RoPE table and in-kernel new key/value (cuda_fused.py; the new row is patched into the ring
slot after its copy lands, and stored to the cache), then:
- CTA partial (16 (head, row)s of the pair: log2 max, sum, 64 outputs) in shared memory;
- A42 cluster merge: CS (8 or 16) co-scheduled CTAs per cluster; barrier.cluster; the leader
  (rank 0) reads every rank's partial with mapa + ld.shared::cluster, merges; a second cluster
  barrier keeps the peers' shared memory alive until the leader is done;
- one cluster per (request, pair): the leader applies RMSNorm x sigmoid(gate) and stores BF16;
- A43 several clusters: each leader writes its cluster partial, the last one (atomic ticket,
  reset after use) merges them and stores.
Grid x = nsplit = CS * clusters per (request, pair); cluster rank = blockIdx.x % CS.
"""

CUDA_UD_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_ud {

constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint16_t f2bf16(float x) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_peer(const float* local, uint32_t rank) {   // DSMEM load
  uint32_t a, r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a));
  (void)r;
  return v;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(WARPS * 32) ud_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, float* __restrict__ cpart_o, float* __restrict__ cpart_m,
    float* __restrict__ cpart_l, int* __restrict__ tickets, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];
  __shared__ float cta_m[16], cta_l[16];          // this CTA's partial, read by the cluster leader
  __shared__ float cta_o[16][64];
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ float rope_c[32], rope_s[32];
  __shared__ int last_flag;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  const float pos = static_cast<float>(positions[b]);
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token

  // ---- RoPE table for this request's position: one full-range sincosf per frequency (the
  // arguments reach ~1e5 rad, CUDA's slow reduction path), shared by every lane.
  if (rope) {
    if (warp == 0) sincosf(pos * inv_freq[lane], &rope_s[lane], &rope_c[lane]);
    __syncthreads();
  }

  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.
  if (has_new && warp == 0) {
    const int h = lane >> 4, d0 = 4 * (lane & 15);          // head kv0 + h, dims d0 .. d0+3
    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;
    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;
    float k4[4], v4[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      k4[j] = bf16f(kp[j]);
      v4[j] = bf16f(vp[j]);
      ss += k4[j] * k4[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 8);                       // sum over the head's 16 lanes
    const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
    for (int j = 0; j < 4; ++j) k4[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const float sn = rope_s[d0 / 2 + p], cs = rope_c[d0 / 2 + p];
        const float e = k4[2 * p], o = k4[2 * p + 1];
        k4[2 * p] = e * cs - o * sn;
        k4[2 * p + 1] = e * sn + o * cs;
      }
    }
    const float kinv = 1.f / (*k_scale_p), vinv = 1.f / (*v_scale_p);
    const uint32_t kw = e4m3x2(k4[0] * kinv, k4[1] * kinv) | (e4m3x2(k4[2] * kinv, k4[3] * kinv) << 16);
    const uint32_t vw = e4m3x2(v4[0] * vinv, v4[1] * vinv) | (e4m3x2(v4[2] * vinv, v4[3] * vinv) << 16);
    reinterpret_cast<uint32_t*>(new_k)[lane] = kw;            // bytes 4*lane .. = head h, dims d0 ..
    reinterpret_cast<uint32_t*>(new_v)[lane] = vw;
    const long row = cache_loc[b];
    reinterpret_cast<uint32_t*>(kc + row * row_bytes + poff)[lane] = kw;
    reinterpret_cast<uint32_t*>(vc + row * row_bytes + poff)[lane] = vw;
  }

  // ---- queries: lane = head hl, dims 32*(t4&1) .. +31 of query g; RMSNorm x factor, RoPE, FP16.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
    float x[32];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      x[j] = bf16f(qp[j]);
      ss += x[j] * x[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)
    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];
        const float e = x[2 * p], o = x[2 * p + 1];
        x[2 * p] = e * cs - o * sn;
        x[2 * p + 1] = e * sn + o * cs;
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);
      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);
      qa[ks][0] = hl == 0 ? lo2 : 0u;
      qa[ks][1] = hl == 1 ? lo2 : 0u;
      qa[ks][2] = hl == 0 ? hi2 : 0u;
      qa[ks][3] = hl == 1 ? hi2 : 0u;
    }
  }
  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Row of token (start + c*CHUNK + (lane & 15)) of the warp's i-th chunk, clamped into the slice.
  auto row_of = [&](int i) -> long {
    const int s = min(start + (warp + i * WARPS) * CHUNK + (lane & 15), end - 1);
    return static_cast<long>(pt[s / page]) * page + s % page;
  };
  auto issue = [&](int i, long my_row) {
    if (i < mine) {
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    asm volatile("cp.async.commit_group;");
  };

  long row_next = 0;
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i, i < mine ? row_of(i) : 0);
  if (STAGES - 1 < mine) row_next = row_of(STAGES - 1);
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1, row_next);
    if (i + STAGES < mine) row_next = row_of(i + STAGES);
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const int c = warp + i * WARPS;
    uint8_t* stw = ring + (i % STAGES) * STAGE_BYTES;
    if (has_new && n - 1 >= start + c * CHUNK && n - 1 < start + (c + 1) * CHUNK) {
      // the new token's row: take the key/value computed in-kernel (the copy read the cache row
      // before or while it was being written)
      const int tn = n - 1 - (start + c * CHUNK);
      reinterpret_cast<uint32_t*>(stw + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_k)[lane];
      reinterpret_cast<uint32_t*>(stw + CHUNK * PITCH + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_v)[lane];
      __syncwarp();
    }
    const uint8_t* st = stw;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  asm volatile("cp.async.wait_all;");

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const int hr = h * 8 + row;                    // pair-local (head, row) index 0..15
    cta_o[hr][d] = o;
    if (d == 0) {
      cta_m[hr] = mm;
      cta_l[hr] = l;
    }
  }

  // ---- A42 cluster merge: the leader (rank 0) reads every rank's partial over DSMEM.
  const uint32_t rank = cluster_rank();
  const int nc = nsplit / CS;                       // clusters per (request, pair)
  const int ci = split / CS;
  cluster_sync();                                   // every CTA's partial written and visible
  constexpr int RPW = 16 / WARPS;                   // (head, row)s per leader warp
  float fo0[RPW], fo1[RPW], fl[RPW], fm[RPW];
  if (rank == 0) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) {
      const int hr = warp + k2 * WARPS;
      fm[k2] = -1.0e30f; fl[k2] = 0.f; fo0[k2] = 0.f; fo1[k2] = 0.f;
      if (hr < 16) {
        float mm = -1.0e30f;
#pragma unroll
        for (int r = 0; r < CS; ++r) mm = fmaxf(mm, ld_peer(&cta_m[hr], r));
        float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int r = 0; r < CS; ++r) {
          const float wt = ex2(ld_peer(&cta_m[hr], r) - mm);
          l += wt * ld_peer(&cta_l[hr], r);
          o0 += wt * ld_peer(&cta_o[hr][lane], r);
          o1 += wt * ld_peer(&cta_o[hr][32 + lane], r);
        }
        fm[k2] = mm; fl[k2] = l; fo0[k2] = o0; fo1[k2] = o1;
      }
    }
  }
  cluster_sync();                                   // the leader is done reading: peers may exit
  if (rank != 0) return;

  const float vsc = *v_scale_p;
  auto finalize = [&](int hr, float o0, float o1, float l) {   // RMSNorm x sigmoid(gate), BF16 store
    const float scale = vsc / l;
    o0 *= scale;
    o1 *= scale;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float r = rsqrtf(ss / 64.f + 1e-5f);
    const int head = kv0 * 8 + hr;
    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + head * 64;
    const float g0 = bf16f(gp[lane]), g1 = bf16f(gp[32 + lane]);
    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;
    op[lane] = f2bf16(o0 * r / (1.f + __expf(-g0)));
    op[32 + lane] = f2bf16(o1 * r / (1.f + __expf(-g1)));
  };
  if (nc == 1) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);
    return;
  }

  // ---- A43 second level: cluster partials in global memory; the last leader (ticket) merges.
  const long cbase = (static_cast<long>(b) * hq + kv0 * 8) * nc;      // (head kv0*8 + hr, cluster)
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    const int hr = warp + k2 * WARPS;
    {
      const long p = cbase + static_cast<long>(hr) * nc + ci;
      cpart_o[p * 64 + lane] = fo0[k2];
      cpart_o[p * 64 + 32 + lane] = fo1[k2];
      if (lane == 0) {
        cpart_m[p] = fm[k2];
        cpart_l[p] = fl[k2];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_flag = atomicAdd(tickets + blockIdx.y, 1) == nc - 1;
  __syncthreads();
  if (!last_flag) return;
  __threadfence();
  for (int hr = warp; hr < 16; hr += WARPS) {       // nc <= 64: one warp per (head, row), lanes over clusters
    const long pb = cbase + static_cast<long>(hr) * nc;
    float mm = -1.0e30f;
    for (int c2 = lane; c2 < nc; c2 += 32) mm = fmaxf(mm, __ldcg(cpart_m + pb + c2));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll 4
    for (int c2 = 0; c2 < nc; ++c2) {
      const float wt = ex2(__ldcg(cpart_m + pb + c2) - mm);
      l += wt * __ldcg(cpart_l + pb + c2);
      o0 += wt * __ldcg(cpart_o + (pb + c2) * 64 + lane);
      o1 += wt * __ldcg(cpart_o + (pb + c2) * 64 + 32 + lane);
    }
    finalize(hr, o0, o1, l);
  }
  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)
}

}  // namespace okc_ud

void ud(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,
        torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,
        torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out,
        int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window,
        int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_ud::STAGE_BYTES;
#define OKC_UD_ARGS                                                                                           \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \
      cpart_o.data_ptr<float>(), cpart_m.data_ptr<float>(), cpart_l.data_ptr<float>(), tickets.data_ptr<int>(), \
      reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),             \
      static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), \
      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)
  if (cs == 16) {
    if (warps == 8) okc_ud::ud_kernel<8, 2, 16><<<grid, 256, smem, stream>>>(OKC_UD_ARGS);
    else okc_ud::ud_kernel<4, 3, 16><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);
  } else {
    if (warps == 8) okc_ud::ud_kernel<8, 2, 8><<<grid, 256, smem, stream>>>(OKC_UD_ARGS);
    else if (stages == 4) okc_ud::ud_kernel<4, 4, 8><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);
    else okc_ud::ud_kernel<4, 3, 8><<<grid, 128, smem, stream>>>(OKC_UD_ARGS);
  }
#undef OKC_UD_ARGS
  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later
  TORCH_CHECK(e == cudaSuccess, "ud launch failed: ", cudaGetErrorString(e));
}

int64_t ud_setup() {   // once, outside any capture: each variant's ring, and 16-CTA clusters
  int err = 0;
  using okc_ud::STAGE_BYTES;
#define OKC_UD_SMEM(W, S, C) \
  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES)
  OKC_UD_SMEM(4, 3, 8);
  OKC_UD_SMEM(4, 4, 8);
  OKC_UD_SMEM(8, 2, 8);
  OKC_UD_SMEM(4, 3, 16);
  OKC_UD_SMEM(8, 2, 16);
#undef OKC_UD_SMEM
  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<4, 3, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  err |= cudaFuncSetAttribute(okc_ud::ud_kernel<8, 2, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> ud_attrs(int64_t warps, int64_t stages, int64_t cs) {
  cudaFuncAttributes a;
  if (cs == 16) {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<8, 2, 16>);
    else cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 3, 16>);
  } else {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<8, 2, 8>);
    else if (stages == 4) cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 4, 8>);
    else cudaFuncGetAttributes(&a, okc_ud::ud_kernel<4, 3, 8>);
  }
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_UD_DECL = (
    "void ud(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
    "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
    "torch::Tensor seqlens, torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, "
    "torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, "
    "int64_t nsplit, int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, "
    "int64_t warps, int64_t stages, int64_t cs);\n"
    "int64_t ud_setup();\n"
    "std::vector<int64_t> ud_attrs(int64_t warps, int64_t stages, int64_t cs);")


# ===== appended: attention/local/cuda_ver2.py =====
"""S2b: verify attention (q_len 8) with S2a's per-warp compute (cuda_ver.py: warp w owns head
w>>1 and m-tiles 2(w&1), 2(w&1)+1; A5a's per-head fragment permutation; masks only where needed)
and a Hopper producer/consumer ring instead of S2a's per-16-key __syncthreads (AF22):
- one producer warp (warp 4) copies 64-key stages (whole 128-byte rows: 8 lanes x 16 B per row,
  4 rows per instruction; page rows fetched a stage ahead) into a STAGES-deep ring in dynamic
  shared memory (K rows at pitch 192 B, V rows at 144 B, as S2a), then each lane's
  cp.async.mbarrier.arrive.noinc signals the stage's full barrier (count 32) when its copies land;
- four consumer warps wait on the full barrier (try_wait.parity), compute the stage's four
  16-key sub-chunks, and one lane per warp arrives on the stage's empty barrier (count 4);
- the producer waits on the empty barrier before refilling a stage; no CTA-wide sync in the loop.
Between the Triton prologue (q scratch) and the Triton _combine (partials as S2a).
"""

CUDA_VER2_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_ver2 {

constexpr int CHUNK = 16;
constexpr int L = 8;
constexpr int KP = 192;   // K row pitch in shared memory (bytes)
constexpr int VP = 144;   // V row pitch
constexpr int SKEYS = 64; // keys per ring stage
constexpr int STAGE_BYTES = SKEYS * (KP + VP);
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ uint32_t w4(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

template <int STAGES>
__global__ void __launch_bounds__(160) attend_ver2_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  extern __shared__ __align__(16) uint8_t ring[];   // STAGES x (64 K rows at KP, 64 V rows at VP)
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int h = warp >> 1, mt0 = 2 * (warp & 1);
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;

  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l
  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Query fragments: m-tile mt = mt0 + mi, rows g -> token 2*mt, g+8 -> token 2*mt+1, head (kv0+h)*8+g.
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int ta = 2 * (mt0 + mi);
    const uint16_t* qA = q + ((static_cast<long>(b) * L + ta) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;
    const uint16_t* qB = qA + static_cast<long>(hq) * 64;               // token ta + 1
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[mi][ks][0] = pack_f16x2(bf16f(qA[4 * ks + 0]), bf16f(qA[4 * ks + 1]));
      qa[mi][ks][1] = pack_f16x2(bf16f(qB[4 * ks + 0]), bf16f(qB[4 * ks + 1]));
      qa[mi][ks][2] = pack_f16x2(bf16f(qA[4 * ks + 2]), bf16f(qA[4 * ks + 3]));
      qa[mi][ks][3] = pack_f16x2(bf16f(qB[4 * ks + 2]), bf16f(qB[4 * ks + 3]));
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  const int nst = (nchunks + 3) / 4;                  // 64-key stages of this slice
  float m_run[2][2], l_run[2][2], acc[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    m_run[mi][0] = m_run[mi][1] = -1.0e30f;
    l_run[mi][0] = l_run[mi][1] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&full_bar[s])));   // producer lanes
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(smem_u32(&empty_bar[s])));   // consumer warps
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp == 4) {
    // ---- producer: whole-row 16 B copies of each 64-key stage, rows one stage ahead.
    auto rows_of = [&](int i, long& r0, long& r1) {
      const int s0 = min(start + i * SKEYS + lane, end - 1), s1 = min(start + i * SKEYS + 32 + lane, end - 1);
      r0 = static_cast<long>(pt[s0 / page]) * page + s0 % page;
      r1 = static_cast<long>(pt[s1 / page]) * page + s1 % page;
    };
    long r0 = 0, r1 = 0, n0 = 0, n1 = 0;
    if (nst > 0) rows_of(0, r0, r1);
    for (int i = 0; i < nst; ++i) {
      const int st = i % STAGES;
      if (i + 1 < nst) rows_of(i + 1, n0, n1);
      if (i >= STAGES) {                              // wait until the consumers released this stage
        const uint32_t parity = static_cast<uint32_t>((i / STAGES - 1) & 1);
        uint32_t done = 0;
        while (!done)
          asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                       : "=r"(done) : "r"(smem_u32(&empty_bar[st])), "r"(parity) : "memory");
      }
      uint8_t* skw = ring + st * STAGE_BYTES;
      uint8_t* svw = skw + SKEYS * KP;
#pragma unroll
      for (int qq = 0; qq < 16; ++qq) {               // 8 lanes per whole row, 4 rows per instruction
        const int tq = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, qq < 8 ? r0 : r1, tq & 31);
        cp16(skw + tq * KP + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(svw + tq * VP + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&full_bar[st])) : "memory");
      r0 = n0;
      r1 = n1;
    }
    asm volatile("cp.async.wait_all;");
    return;                                           // the producer writes no partials
  }
  // ---- consumers: wait for a stage, compute its four 16-key sub-chunks, release it.
  for (int i = 0; i < nst; ++i) {
    const int st = i % STAGES;
    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                   : "=r"(done) : "r"(smem_u32(&full_bar[st])), "r"(parity) : "memory");
    const uint8_t* sk = ring + st * STAGE_BYTES;
    const uint8_t* sv = sk + SKEYS * KP;
#pragma unroll 1
    for (int sc = 0; sc < 4; ++sc) {
      const int c = i * 4 + sc;
      if (c >= nchunks) break;
        const int s0 = start + c * CHUNK;
        const uint4 kb0 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + g) * KP + 64 * h + 16 * t4]);
        const uint4 kb1 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + 8 + g) * KP + 64 * h + 16 * t4]);
        const uint2 vb0 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4) * VP + 64 * h + 8 * g]);
        const uint2 vb1 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 1) * VP + 64 * h + 8 * g]);
        const uint2 vb2 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 8) * VP + 64 * h + 8 * g]);
        const uint2 vb3 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 9) * VP + 64 * h + 8 * g]);
        // a chunk needs masks only at the slice tail or where it reaches the block's own keys
        const bool masked = s0 + CHUNK > end || s0 + CHUNK > n - L || window > 0;

    #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          float sc[2][4];
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const uint4& kb = nt == 0 ? kb0 : kb1;
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
    #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              const uint32_t w = w4(kb, ks);
              mma16816(sc[nt], qa[mi][ks][0], qa[mi][ks][1], qa[mi][ks][2], qa[mi][ks][3], f16x2_lo(w), f16x2_hi(w));
            }
          }
          const int posA = n - L + 2 * (mt0 + mi), posB = posA + 1;     // rows g / g+8
          float mxA = -INFINITY, mxB = -INFINITY;
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt)
    #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              const int key = s0 + nt * 8 + 2 * t4 + jj;
              float xa = sc[nt][jj] * qk_scale, xb = sc[nt][2 + jj] * qk_scale;
              if (masked) {
                const bool ok = key < end;
                const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);
                const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);
                xa = okA ? xa : -INFINITY;
                xb = okB ? xb : -INFINITY;
              }
              sc[nt][jj] = xa;
              sc[nt][2 + jj] = xb;
              mxA = fmaxf(mxA, xa);
              mxB = fmaxf(mxB, xb);
            }
          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));
          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));
          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));
          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));
          const float mnA = fmaxf(m_run[mi][0], mxA), mnB = fmaxf(m_run[mi][1], mxB);
          const float alA = ex2(m_run[mi][0] - mnA), alB = ex2(m_run[mi][1] - mnB);
          float p[2][4];
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            p[nt][0] = ex2(sc[nt][0] - mnA);
            p[nt][1] = ex2(sc[nt][1] - mnA);
            p[nt][2] = ex2(sc[nt][2] - mnB);
            p[nt][3] = ex2(sc[nt][3] - mnB);
          }
          l_run[mi][0] = l_run[mi][0] * alA + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
          l_run[mi][1] = l_run[mi][1] * alB + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
          m_run[mi][0] = mnA;
          m_run[mi][1] = mnB;
          const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);
          const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);
          const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);
          const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);
    #pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[mi][j][0] *= alA;
            acc[mi][j][1] *= alA;
            acc[mi][j][2] *= alB;
            acc[mi][j][3] *= alB;
            const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
            const uint32_t b0 = f16x2_lo(__byte_perm(j < 4 ? vb0.x : vb0.y, j < 4 ? vb1.x : vb1.y, sel));
            const uint32_t b1 = f16x2_lo(__byte_perm(j < 4 ? vb2.x : vb2.y, j < 4 ? vb3.x : vb3.y, sel));
            mma16816(acc[mi][j], pa0, pa1, pa2, pa3, b0, b1);
          }
        }

    }
    __syncwarp();
    if (lane == 0) {
      uint64_t state;
      asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem_u32(&empty_bar[st])) : "memory");
      (void)state;
    }
  }

  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4+jj) + j.
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float l = l_run[mi][r];
      l += __shfl_xor_sync(FULL, l, 1);
      l += __shfl_xor_sync(FULL, l, 2);
      const int tok = 2 * (mt0 + mi) + r;
      const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) o_part[part * 64 + 8 * (2 * t4 + jj) + j] = acc[mi][j][2 * r + jj];
      if (t4 == 0) {
        m_part[part] = m_run[mi][r];
        l_part[part] = l;
      }
    }
}

}  // namespace okc_ver2

void attend_ver2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t stages) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(stages) * okc_ver2::STAGE_BYTES;
#define OKC_VER2_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (stages == 4) okc_ver2::attend_ver2_kernel<4><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);
  else if (stages == 2) okc_ver2::attend_ver2_kernel<2><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);
  else okc_ver2::attend_ver2_kernel<3><<<grid, 160, smem, stream>>>(OKC_VER2_ARGS);
#undef OKC_VER2_ARGS
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "attend_ver2 launch failed: ", cudaGetErrorString(e));
}

int64_t attend_ver2_setup() {
  int err = 0;
  using okc_ver2::STAGE_BYTES;
  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_ver2::attend_ver2_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * STAGE_BYTES);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> attend_ver2_attrs(int64_t stages) {
  cudaFuncAttributes a;
  if (stages == 4) cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<4>);
  else if (stages == 2) cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<2>);
  else cudaFuncGetAttributes(&a, okc_ver2::attend_ver2_kernel<3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_VER2_DECL = ("void attend_ver2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                  "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                  "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                  "int64_t page, int64_t window, double sm_scale, int64_t stages);\n"
                  "int64_t attend_ver2_setup();\n"
                  "std::vector<int64_t> attend_ver2_attrs(int64_t stages);")


# ===== appended: attention/local/v4_block.py =====


# =============================================================================================
# v4: CUDA kernels with hybrid dispatch over v2.2 (Triton or SGLang paths, above). Kinds:
#   "dec2": A5b, register-direct decode (cuda_dec2.py), between the Triton prologue and combine;
#   "dec3": D2, per-warp cp.async ring decode (cuda_dec3.py), same glue;
#   "dec4": D3, D2 with page entries carried by the ring (cuda_dec4.py, make_pt_ring.py), same glue;
#   "ud":   UD, one launch per decode layer (cuda_unified.py): query fold, ring streaming,
#           new-row patch, cluster DSMEM merge, second level, RMSNorm x gate, BF16 out;
#   "ver2": S2b, verify with a producer warp and mbarrier stages (cuda_ver2.py), Triton glue.
# One load_inline module (sm_90a) holds only the routed kinds, built on the first eager call
# next to the Triton warm-up. Each route is then launched once on synthetic inputs of its class
# (random FP8 cache, our own data) and compared with the Triton path; a route whose build,
# setup, launch or result fails is dropped, so its class keeps its v2.2 path.
# CUDA_ROUTE maps a scored class (full layer, verify block, batch class) to its kernel and
# launch settings, from the labs' judge-H200 timings; classes not listed stay on v2.2.
# =============================================================================================
import tempfile as _tempfile

_CUDA = {}
_TICKETS = []

# class -> (kind, settings...):
#   ("dec2", ctas per SM, warps) | ("dec3" or "dec4", ctas per SM, warps, stages)
#   ("ud", cluster size, ctas per SM, warps, stages) | ("ver2", ctas per SM, stages)
CUDA_ROUTE = {
    (True, False, 16): ("dec4", 6, 4, 3),   # L12 (AE17): D3 125.8 us vs D2 155.6 (L9b), v2.2 182.3
    (True, False, 32): ("dec4", 6, 4, 3),   # L12 (AE17): D3 214.0 us vs D2 263.3 (L9b), v2.2 372.2
}

_KINDS = {   # kind -> (CUDA source, declarations, functions, setup function or None)
    "dec2": (CUDA_DEC2_SRC, CUDA_DEC2_DECL, ["attend_dec2"], None),
    "dec3": (CUDA_DEC3_SRC, CUDA_DEC3_DECL, ["attend_dec3", "attend_dec3_setup"], "attend_dec3_setup"),
    "dec4": (CUDA_DEC4_SRC, CUDA_DEC4_DECL, ["attend_dec4", "attend_dec4_setup"], "attend_dec4_setup"),
    "ud": (CUDA_UD_SRC, CUDA_UD_DECL, ["ud", "ud_setup"], "ud_setup"),
    "ver2": (CUDA_VER2_SRC, CUDA_VER2_DECL, ["attend_ver2", "attend_ver2_setup"], "attend_ver2_setup"),
}
SELF_CHECK_TOL = 2e-2   # relative L2 vs the Triton path (both ~2e-3 from FP32; a bug is O(1))
_WARM_DEVICE = "cuda"   # the CPU glue test (test_v4_glue.py) runs the warm-up on "cpu"


def _cuda_build():
    from torch.utils.cpp_extension import load_inline
    kinds = sorted({route[0] for route in CUDA_ROUTE.values()})
    if not kinds:
        return
    try:
        mod = load_inline(
            name="okc_attention_v4_" + "_".join(kinds), cpp_sources="\n".join(_KINDS[k][1] for k in kinds),
            cuda_sources="\n".join(_KINDS[k][0] for k in kinds),
            functions=[f for k in kinds for f in _KINDS[k][2]],
            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
            build_directory=_tempfile.mkdtemp(prefix="okc_attention_v4_"), verbose=False)
    except Exception as exc:  # every class keeps its v2.2 path
        print(f"v4: CUDA build failed, staying on v2.2 paths: {type(exc).__name__}: {str(exc)[-300:]}")
        return
    _CUDA["mod"] = mod
    _CUDA["refused"] = set()
    for k in kinds:
        setup = _KINDS[k][3]
        if setup is not None and getattr(mod, setup)() != 0:
            print(f"v4: {setup} refused (shared memory or cluster attribute); {k} routes dropped")
            _CUDA["refused"].add(k)


def _nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // min_keys, MAX_SPLITS))


def _ud_clusters(cfg, max_pages, cs, ctas_per_sm, warps):
    """Clusters per (request, head pair): one for sliding layers (the window fits one cluster),
    else enough for ctas_per_sm CTAs per SM, at least one warp chunk (32 keys) per warp."""
    if cfg["window"]:
        return 1
    tokens = max_pages * cfg["page_size"]
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    nc = max(1, math.ceil(ctas_per_sm * NUM_SMS / (programs * cs)))
    return max(1, min(nc, tokens // (cs * 32 * warps)))


def _prologue_q(data):
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = cfg["batch"] * cfg["q_len"]
    device = data["qkv"].device
    rope = bool(cfg["rope_theta"])
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(tokens,)](data["qkv"], data["positions"], data["cache_loc"],
                         _inv_freq(cfg["rope_theta"] if rope else 1.0, device), q, data["k_cache"], data["v_cache"],
                         data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, ROPE=rope,
                         QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    return q


def _merge(data, o_part, m_part, l_part, nsplit):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    _combine[(batch * length, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit,
                                   HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)),
                                   num_warps=4)
    return out


def _cuda_run(data, route):
    kind = route[0]
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    max_pages = data["page_table"].shape[1]
    mod = _CUDA["mod"]
    if kind == "ud":
        _, cs, cps, warps, stages = route
        nc = _ud_clusters(cfg, max_pages, cs, cps, warps)
        rope = bool(cfg["rope_theta"])
        cpart_o = torch.empty((batch * hq * nc, 64), dtype=torch.float32, device=device)
        cpart_m = torch.empty((batch * hq * nc,), dtype=torch.float32, device=device)
        cpart_l = torch.empty_like(cpart_m)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        mod.ud(data["qkv"], data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
               data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
               data["cache_seqlens"], cpart_o, cpart_m, cpart_l, _TICKETS[0], out, batch, hq, hkv, max_pages,
               cs * nc, cfg["page_size"], cfg["window"], int(rope), cfg["qk_scale_factor"], cfg["sm_scale"],
               warps, stages, cs)
        return out
    q = _prologue_q(data)
    if kind == "ver2":
        _, cps, stages = route
        nsplit = _nsplit(cfg, max_pages, cps, 64)
    else:   # dec2 / dec3 / dec4: one warp chunk (32 keys) per warp at least
        cps, warps = route[1], route[2]
        nsplit = _nsplit(cfg, max_pages, cps, 32 * warps)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
            cfg["sm_scale"])
    if kind == "dec2":
        mod.attend_dec2(*args, route[2])
    elif kind == "dec3":
        mod.attend_dec3(*args, route[2], route[3])
    elif kind == "dec4":
        mod.attend_dec4(*args, route[2], route[3])
    else:
        mod.attend_ver2(*args, route[2])
    return _merge(data, o_part, m_part, l_part, nsplit)


def _cuda_ok(cfg, route):
    """Every CUDA kernel needs groups of 8 query heads and paired KV heads; the decode kinds take
    q_len 1, ver2 q_len 8."""
    kind = route[0]
    if cfg["num_q_heads"] // cfg["num_kv_heads"] != 8 or cfg["num_kv_heads"] % 2:
        return False
    return cfg["q_len"] == (8 if kind == "ver2" else 1)


def _self_check_inputs(bclass, length, full, device):
    """Synthetic inputs of a class (our own random data, never the judge's): a random FP8 cache
    so every key matters, a context that spans the window on sliding layers."""
    data = dummy_inputs(bclass, length, 0 if full else 2048, scored_max_pages(bclass, length), device,
                        ctx=3000 if full else 2300)
    gen = torch.Generator(device=device).manual_seed(4040 + bclass + length)
    for name in ("k_cache", "v_cache"):
        t = data[name]
        data[name] = (torch.randn(t.shape, generator=gen, device=device) * 0.5).to(t.dtype)
    return data


_kernel_fn_v22 = kernel_fn
_warm_up_v22 = _warm_up


def _warm_up():  # noqa: F811 (v4: Triton warm-up, then the CUDA build and a checked launch per route)
    _warm_up_v22()
    _cuda_build()
    if "mod" not in _CUDA:
        CUDA_ROUTE.clear()
        return
    dev = torch.device(_WARM_DEVICE)
    _TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    for cls, route in sorted(CUDA_ROUTE.items()):
        full, verify, bclass = cls
        try:
            if route[0] in _CUDA["refused"]:
                raise RuntimeError("setup refused")
            data = _self_check_inputs(bclass, 8 if verify else 1, full, dev)
            if not _cuda_ok(data["config"], route):
                raise RuntimeError("config not supported")
            ours = _cuda_run(data, route).float()
            want = run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-6))
            if not err < SELF_CHECK_TOL:
                raise RuntimeError(f"differs from the Triton path: rel. L2 {err:.2e}")
        except Exception as exc:
            print(f"v4: route {cls} -> {route} dropped: {type(exc).__name__}: {str(exc)[-200:]}")
            CUDA_ROUTE.pop(cls)


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    route = CUDA_ROUTE.get(shape_class(cfg))
    if route is not None and "mod" in _CUDA and _cuda_ok(cfg, route):
        return _cuda_run(data, route)
    return _kernel_fn_v22(data)


# ===== appended: attention/local/pdl_block.py =====


# =============================================================================================
# PDL variant of v2.2's Triton chain (generated by attention/local/make_pdl.py; see its docstring).
# _prologue_p / _attend_p trigger their dependents first thing (TRIGGER_EARLY) or after their last
# store; _attend_p and _combine_p (launched with launch_pdl=True) wait first thing.
# =============================================================================================
import triton.language.extra.cuda as tlcuda


@triton.jit
def _prologue_p(qkv_ptr, pos_ptr, loc_ptr, inv_freq_ptr, q_out_ptr, k_cache_ptr, v_cache_ptr,
              k_scale_ptr, v_scale_ptr,
              HQ: tl.constexpr, HKV: tl.constexpr, ROPE: tl.constexpr,
              QK_FACTOR: tl.constexpr, EPS: tl.constexpr, TRIGGER_EARLY: tl.constexpr):
    if TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()
    t = tl.program_id(0)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    base = qkv_ptr + t.to(tl.int64) * ROW
    hq = tl.arange(0, HQ)
    hk = tl.arange(0, HKV)
    i = tl.arange(0, 32)
    # Even and odd lanes of every head: RoPE rotates the pairs (2i, 2i+1).
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
        q_e, q_o = q_e * c - q_o * s, q_e * s + q_o * c
        k_e, k_o = k_e * c - k_o * s, k_e * s + k_o * c

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


@triton.jit(do_not_specialize=["max_pages", "nsplit"])
def _attend_p(q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, page_table_ptr, seqlens_ptr,
            o_part_ptr, m_part_ptr, l_part_ptr, max_pages, nsplit,
            HQ: tl.constexpr, HKV: tl.constexpr, L: tl.constexpr, PAGE: tl.constexpr,
            WINDOW: tl.constexpr, SM_SCALE: tl.constexpr, MP: tl.constexpr, BLOCK: tl.constexpr, TRIGGER_EARLY: tl.constexpr):
    tlcuda.gdc_wait()   # the prologue's q and new cache row are visible after this
    if TRIGGER_EARLY:
        tlcuda.gdc_launch_dependents()
    b = tl.program_id(0)
    kv0 = tl.program_id(1) * 2
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    ML: tl.constexpr = L * G        # query rows per KV head: block token x group head

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

    n = tl.load(seqlens_ptr + b)    # keys 0 .. n-1; row l sits at position n - L + l
    row_pos = n - L + r_l
    if WINDOW > 0:
        lo = tl.maximum(n - L - WINDOW + 1, 0)
    else:
        lo = 0
    per = tl.cdiv(tl.cdiv(n - lo, nsplit), BLOCK) * BLOCK
    start = lo + split * per
    end = tl.minimum(start + per, n)

    qk_scale = SM_SCALE * tl.load(k_scale_ptr) * 1.4426950408889634
    m0 = tl.full([MP], -1.0e30, tl.float32)
    l0 = tl.zeros([MP], tl.float32)
    a0 = tl.zeros([MP, 64], tl.float32)
    m1 = tl.full([MP], -1.0e30, tl.float32)
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
        m0, l0, a0 = _head_step(q0, k0, v0, m0, l0, a0, vis, qk_scale)
        m1, l1, a1 = _head_step(q1, k1, v1, m1, l1, a1, vis, qk_scale)

    # Partials per (token, head, slice).
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


@triton.jit(do_not_specialize=["nsplit"])
def _combine_p(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit,
             HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr):
    tlcuda.gdc_wait()   # the attention partials are visible after this
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    big = tl.max(tl.full([CB], -1.0e30, tl.float32), 0)   # fp32 scalars: loop-carried types fixed
    den = tl.sum(tl.zeros([CB], tl.float32), 0)
    num = tl.zeros([64], tl.float32)
    for s0 in range(0, nsplit, CB):       # one pass: running max, rescaled sums
        ok = s0 + j < nsplit
        idx = th * nsplit + s0 + j
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1.0e30)
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


def run_pdl(data, tune=None, early=False):
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = batch * length
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    nsplit, block, warps, stages, cb, mp = plan(cfg, max_pages, tune)
    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"] if rope else 1.0, device)

    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue_p[(tokens,)](qkv, data["positions"], data["cache_loc"], inv_freq, q,
                         data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                         HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS,
                         TRIGGER_EARLY=early, num_warps=4)

    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _attend_p[(batch, hkv // 2, nsplit)](
        q, data["k_cache"], data["v_cache"], data["k_scale"], page_table, data["cache_seqlens"],
        o_part, m_part, l_part, max_pages, nsplit,
        HQ=hq, HKV=hkv, L=length, PAGE=cfg["page_size"], WINDOW=cfg["window"],
        SM_SCALE=cfg["sm_scale"], MP=mp, BLOCK=block, TRIGGER_EARLY=early, num_warps=warps, num_stages=stages,
        launch_pdl=True)

    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    _combine_p[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=4, launch_pdl=True)
    return out


# ===== appended: attention/local/v5_block.py =====


# =============================================================================================
# v5: v4 + programmatic dependent launch on the Triton chain (pdl_block.py, make_pdl.py) for the
# classes where L13 (AE18) measured it >= 3% faster than their v4 path, with the late trigger
# (producers trigger after their last store: TRACKER.md checklist 8). Classes: Sd16 20.7 -> 19.7
# us, Sd32 28.3 -> 27.0, Fd1 32.1 -> 31.1, Sv1 19.6 -> 18.5, Sv16 29.1 -> 28.2, Sv32 (SGLang 42.8
# in L11) -> 40.7. The PDL kernels compile at warm-up and are checked against the Triton path on
# synthetic inputs; any failure drops PDL for every class (v4 paths stay).
# =============================================================================================
PDL_ROUTE = {(False, False, 16), (False, False, 32), (True, False, 1),
             (False, True, 1), (False, True, 16), (False, True, 32)}
_kernel_fn_v4 = kernel_fn
_warm_up_v4 = _warm_up


def _warm_up():  # noqa: F811 (v5: v4's warm-up, then compile and check the PDL chain per class)
    _warm_up_v4()
    dev = torch.device(_WARM_DEVICE)
    try:
        for full, verify, bclass in sorted(PDL_ROUTE):
            data = _self_check_inputs(bclass, 8 if verify else 1, full, dev)
            ours = run_pdl(data, early=False).float()
            want = run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-6))
            if not err < SELF_CHECK_TOL:
                raise RuntimeError(f"class {(full, verify, bclass)} differs from the Triton path: {err:.2e}")
    except Exception as exc:
        print(f"v5: PDL chain dropped: {type(exc).__name__}: {str(exc)[-200:]}")
        PDL_ROUTE.clear()


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    cls = shape_class(cfg)
    if cls not in CUDA_ROUTE and cls in PDL_ROUTE:
        return run_pdl(data, early=False)
    return _kernel_fn_v4(data)


# ===== appended: attention/local/lab_common_block.py =====

# ---------------------------------------------------------------------------------------------
# Lab helpers shared by labs L4+ (check runs only): timing under CUDA graphs with a cold L2,
# scored-shape inputs with the judge's "soft" keys, and an FP32 torch reference.
# ---------------------------------------------------------------------------------------------
import statistics as _st
import time as _time

_LAB = []
_kernel_fn_submitted = kernel_fn


def _log(msg):
    print(f"[lab] {msg}", flush=True)


def _why(exc):
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[-300:]}"


class _Lab:
    def __init__(self):
        self.t0 = _time.monotonic()
        self.flush = torch.empty((512 << 20) // 8, dtype=torch.int64, device="cuda")

    def left(self):
        return _LAB_BUDGET_S - (_time.monotonic() - self.t0)

    def time_graph(self, fn, reps=15):
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
            self.flush.fill_(1)
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return _st.median(times)


def _lab_inputs(batch, length, window, k_cache, v_cache, gen):
    dev = k_cache.device
    hq, hkv = 16, 2
    max_pages = scored_max_pages(batch, length)
    if batch == 1:
        seq = torch.full((1,), 80000, dtype=torch.int64, device=dev)
    else:
        u = torch.rand((batch,), generator=gen, device=dev)
        seq = (80000 * (1 + 0.075 * (2 * u - 1))).long()
    rows = torch.randperm(k_cache.shape[0], generator=gen, device=dev)[:batch * max_pages].to(torch.int32)
    table = rows.view(batch, max_pages)
    pos = (seq[:, None] + torch.arange(length, device=dev)).reshape(-1)
    cfg = dict(batch=batch, q_len=length, num_q_heads=hq, num_kv_heads=hkv, head_dim=64, page_size=1,
               window=window, rope_theta=500000.0 if window else 0.0,
               qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0)
    return dict(
        qkv=torch.randn((batch, length, (2 * hq + 2 * hkv) * 64), generator=gen, device=dev).to(torch.bfloat16),
        positions=pos,
        cache_loc=torch.gather(table.long(), 1, pos.view(batch, length)).reshape(-1),
        cache_seqlens=(seq + length).to(torch.int32),
        cu_seqlens_q=torch.arange(0, batch * length + 1, length, dtype=torch.int32, device=dev),
        scheduler_metadata=None, k_cache=k_cache, v_cache=v_cache,
        k_scale=torch.tensor(1.0, device=dev), v_scale=torch.tensor(1.0, device=dev),
        page_table=table, seq_lens=seq.to(torch.int32), config=cfg)


def _lab_reference(data):
    """FP32 output from the cache as it stands (run after the prologue wrote the new rows) and
    BF16 queries recomputed in torch: RMSNorm x qk factor (x interleaved RoPE on sliding layers),
    softmax over each row's keys (its window on sliding layers), RMSNorm x sigmoid(gate)."""
    cfg = data["config"]
    hq, hkv, L, B = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["q_len"], cfg["batch"]
    G = hq // hkv
    qkv = data["qkv"].float().view(B, L, -1)
    q = qkv[..., :hq * 64].view(B, L, hq, 64)
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + RMS_EPS) * cfg["qk_scale_factor"]
    if cfg["rope_theta"]:
        ang = data["positions"].view(B, L).float()[..., None] * _inv_freq(cfg["rope_theta"], qkv.device)
        cos, sin = ang.cos()[:, :, None, :], ang.sin()[:, :, None, :]
        even, odd = q[..., 0::2], q[..., 1::2]
        q = torch.stack([even * cos - odd * sin, even * sin + odd * cos], -1).flatten(-2)
    q = q.bfloat16().float()
    gate = qkv[..., (hq + 2 * hkv) * 64:].view(B, L, hq, 64)
    kf = data["k_cache"].view(-1, hkv, 64)
    vf = data["v_cache"].view(-1, hkv, 64)
    out = torch.empty((B, L, hq, 64), device=qkv.device)
    for b in range(B):
        n = int(data["cache_seqlens"][b])
        rows = data["page_table"][b, :n].long()
        k = kf[rows].float()            # [n, hkv, 64]
        v = vf[rows].float()
        for j in range(L):
            p = n - L + j
            lo = max(0, p - cfg["window"] + 1) if cfg["window"] else 0
            att = torch.einsum("hgd,thd->hgt", q[b, j].view(hkv, G, 64), k[lo:p + 1]) * cfg["sm_scale"]
            out[b, j] = torch.einsum("hgt,thd->hgd", att.softmax(-1), v[lo:p + 1]).reshape(hq, 64)
    out = out * torch.rsqrt(out.pow(2).mean(-1, keepdim=True) + RMS_EPS) * torch.sigmoid(gate)
    return out.reshape(B, L, hq * 64)




def _lab_soft_cache(gen, rows):
    """K with the judge's "soft" structure (shared unit-RMS direction per KV head, c2 = 0.55),
    V standard normal, both FP8 E4M3, [rows, 1, 2, 64]."""
    dev = gen.device
    u = torch.randn((1, 1, 2, 64), generator=gen, device=dev)
    z = torch.randn((rows, 1, 2, 64), generator=gen, device=dev)
    k_cache = (0.55 ** 0.5 * u + 0.45 ** 0.5 * z).to(torch.float8_e4m3fn)
    del z
    v_cache = torch.randn((rows, 1, 2, 64), generator=gen, device=dev).to(torch.float8_e4m3fn)
    return k_cache, v_cache


def kernel_fn(data):  # noqa: F811 (lab wrapper; the timed path is the submission's kernel_fn)
    if not _LAB and not torch.cuda.is_current_stream_capturing():
        _LAB.append(True)
        _kernel_fn_submitted(data)   # warm-up first, exactly as the submission does
        try:
            _lab_main()
        except Exception as exc:
            _log(f"FAILED {_why(exc)}")
    return _kernel_fn_submitted(data)


# ===== appended: attention/local/cuda_anat.py =====
"""Generated by make_anatomy.py (do not edit): instrumented D3 / UD2 / S2c and the micro module
for the L15 anatomy lab; see make_anatomy.py for the slots."""

CUDA_AN_DEC4_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_an_dec4 {

// ---- L15 anatomy stamps (lab only) ----
__device__ __forceinline__ unsigned long long an_clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long an_clk_m() {   // also ordered against memory operations
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
#define AN_PUT(k, v) do { if (threadIdx.x == 0) an_buf[an_base + (k)] = (v); } while (0)
#define AN_PP(k, v) do { if (threadIdx.x == 128) an_buf[an_base + (k)] = (v); } while (0)
#define AN_DEP(k, an_v) do { if (threadIdx.x == 0) { an_buf[an_base + 31] = static_cast<unsigned long long>(an_v); \
    an_buf[an_base + (k)] = an_clk_m(); } } while (0)
#define AN_END() do { if (threadIdx.x == 0) { an_buf[an_base + 28] = an_clk_m(); an_buf[an_base + 29] = an_gt(); } } while (0)


constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32) an_dec4_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale, unsigned long long* __restrict__ an_buf) {
  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 32;
  if (threadIdx.x == 0) { an_buf[an_base + 0] = an_gt(); an_buf[an_base + 1] = an_clk(); an_buf[an_base + 30] = an_smid(); }


  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Stacked block-diagonal query fragments: per k-step, {a0a1, a2a3, a4a5, a6a7}.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      // physical byte 32*t4 + 4*ks + j of the pair = head hl, d = 32*(t4&1) + 4*ks + j
      const uint32_t lo2 = pack_f16x2(bf16f(qp[4 * ks + 0]), bf16f(qp[4 * ks + 1]));
      const uint32_t hi2 = pack_f16x2(bf16f(qp[4 * ks + 2]), bf16f(qp[4 * ks + 3]));
      qa[ks][0] = hl == 0 ? lo2 : 0u;   // row g, logical cols 2*t4+{0,1}
      qa[ks][1] = hl == 1 ? lo2 : 0u;   // row g+8
      qa[ks][2] = hl == 0 ? hi2 : 0u;   // row g, cols 2*t4+8+{0,1}
      qa[ks][3] = hl == 1 ? hi2 : 0u;   // row g+8
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
  // Page entries ride the ring: chunk j's 16 entries sit in PT slot p = j % PT_SLOTS, in the unused
  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at
  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.
  constexpr int PT_SLOTS = 2 * STAGES;
  auto pt_slot = [&](int j, int e) -> int* {
    const int p = j % PT_SLOTS;
    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));
  };
  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j's page entries (join the open commit group)
    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);
  };
  // One commit group per call: chunk i's K/V rows (its page entries landed in an earlier group,
  // waited and warp-synced) and chunk i + STAGES's page entries.
  auto issue = [&](int i) {
    if (i < mine) {
      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    issue_pt(i + STAGES);
    asm volatile("cp.async.commit_group;");
  };
  // Prologue: the first STAGES chunks' page entries (the one exposed round trip), then the first
  // STAGES - 1 stages of K/V rows.
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");
  AN_PUT(2, an_clk());
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
  AN_PUT(3, an_clk());
  AN_PUT(8, mine);
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);
  unsigned long long an_wacc = 0;
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    const unsigned long long an_w0 = an_clk();
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const unsigned long long an_w1 = an_clk();
    an_wacc += an_w1 - an_w0;
    if (i == 0) AN_PUT(4, an_w1);
    const int c = warp + i * WARPS;
    const uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  AN_PUT(5, an_clk_m());
  AN_PUT(6, an_wacc);
  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  AN_PUT(7, an_clk_m());
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const long part = (static_cast<long>(b) * hq + (kv0 + h) * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }
  AN_END();
}

}  // namespace okc_an_dec4

void an_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages, torch::Tensor an_buf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_an_dec4::STAGE_BYTES;
#define OKC_DEC4_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale), reinterpret_cast<unsigned long long*>(an_buf.data_ptr())
  if (warps == 8 && stages == 2) okc_an_dec4::an_dec4_kernel<8, 2><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);
  else if (warps == 8) okc_an_dec4::an_dec4_kernel<8, 3><<<grid, 256, smem, stream>>>(OKC_DEC4_ARGS);
  else if (stages == 4) okc_an_dec4::an_dec4_kernel<4, 4><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);
  else okc_an_dec4::an_dec4_kernel<4, 3><<<grid, 128, smem, stream>>>(OKC_DEC4_ARGS);
#undef OKC_DEC4_ARGS
}

int64_t an_dec4_setup() {   // once, outside any capture: allow each variant's ring
  // Each limit is exactly the ring a variant needs (static + dynamic must stay <= 227 KB per
  // block); the last-error state is cleared so a refusal can't surface in a later PyTorch call.
  int err = 0;
  using okc_an_dec4::STAGE_BYTES;
  err |= cudaFuncSetAttribute(okc_an_dec4::an_dec4_kernel<4, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 3 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_an_dec4::an_dec4_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * 4 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_an_dec4::an_dec4_kernel<8, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 2 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_an_dec4::an_dec4_kernel<8, 3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 8 * 3 * STAGE_BYTES);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> an_dec4_attrs(int64_t warps, int64_t stages) {
  cudaFuncAttributes a;
  if (warps == 8 && stages == 2) cudaFuncGetAttributes(&a, okc_an_dec4::an_dec4_kernel<8, 2>);
  else if (warps == 8) cudaFuncGetAttributes(&a, okc_an_dec4::an_dec4_kernel<8, 3>);
  else if (stages == 4) cudaFuncGetAttributes(&a, okc_an_dec4::an_dec4_kernel<4, 4>);
  else cudaFuncGetAttributes(&a, okc_an_dec4::an_dec4_kernel<4, 3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_AN_DEC4_DECL = 'void an_dec4(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t warps, int64_t stages, torch::Tensor an_buf);\nint64_t an_dec4_setup();\nstd::vector<int64_t> an_dec4_attrs(int64_t warps, int64_t stages);'

CUDA_AN_UD2_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_an_ud2 {

// ---- L15 anatomy stamps (lab only) ----
__device__ __forceinline__ unsigned long long an_clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long an_clk_m() {   // also ordered against memory operations
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
#define AN_PUT(k, v) do { if (threadIdx.x == 0) an_buf[an_base + (k)] = (v); } while (0)
#define AN_PP(k, v) do { if (threadIdx.x == 128) an_buf[an_base + (k)] = (v); } while (0)
#define AN_DEP(k, an_v) do { if (threadIdx.x == 0) { an_buf[an_base + 31] = static_cast<unsigned long long>(an_v); \
    an_buf[an_base + (k)] = an_clk_m(); } } while (0)
#define AN_END() do { if (threadIdx.x == 0) { an_buf[an_base + 28] = an_clk_m(); an_buf[an_base + 29] = an_gt(); } } while (0)


constexpr int CHUNK = 16;
constexpr int PITCH = 144;                      // smem row pitch: 9 units of 16 B
constexpr int STAGE_BYTES = 2 * CHUNK * PITCH;  // K tile then V tile
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ uint16_t f2bf16(float x) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_peer(const float* local, uint32_t rank) {   // DSMEM load
  uint32_t a, r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(a) : "r"(smem_u32(local)), "r"(rank));
  float v;
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(a));
  (void)r;
  return v;
}

struct Chunk {
  uint4 k[4];   // [tile nt * 2 + half]: token g (nt 0) / 8+g (nt 1), bytes 32*t4 + 16*half .. +16
  uint4 v[4];   // tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9: bytes 16*g .. 16*g+16
};

template <int WARPS, int STAGES, int CS>
__global__ void __cluster_dims__(CS, 1, 1) __launch_bounds__(WARPS * 32) an_ud2_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, float* __restrict__ cpart_o, float* __restrict__ cpart_m,
    float* __restrict__ cpart_l, int* __restrict__ tickets, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale, unsigned long long* __restrict__ an_buf) {
  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 32;
  if (threadIdx.x == 0) { an_buf[an_base + 0] = an_gt(); an_buf[an_base + 1] = an_clk(); an_buf[an_base + 30] = an_smid(); }

  __shared__ float cta_m[16], cta_l[16];          // this CTA's partial, read by the cluster leader
  __shared__ float cta_o[16][64];
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ float rope_c[32], rope_s[32];
  __shared__ int last_flag;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int hl = t4 >> 1;                              // the head whose columns this lane carries
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;       // the pair's 128-byte piece within a row

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  const float pos = static_cast<float>(positions[b]);
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token

  // ---- ring prologue first (page entries, then the first K/V stages): the fold below
  // overlaps their round trips.
  const int* pt = page_table + static_cast<long>(b) * max_pages;
  extern __shared__ __align__(16) uint8_t dsm[];
  uint8_t* ring = dsm + static_cast<long>(warp) * STAGES * STAGE_BYTES;
  const int mine = warp < nchunks ? (nchunks - warp + WARPS - 1) / WARPS : 0;   // this warp's chunks
  // Token of lane (lane & 15) in the warp's chunk j (c = warp + j*WARPS), clamped into the slice.
  auto tok_of = [&](int j) -> int { return min(start + (warp + j * WARPS) * CHUNK + (lane & 15), end - 1); };
  // Page entries ride the ring: chunk j's 16 entries sit in PT slot p = j % PT_SLOTS, in the unused
  // pads (bytes 128..143) of K-tile rows 4*(p / STAGES) .. +3 of stage p % STAGES: entry e at
  // row 4*(p / STAGES) + e / 4, byte 128 + 4*(e % 4). Row copies and fragment reads use bytes 0..127.
  constexpr int PT_SLOTS = 2 * STAGES;
  auto pt_slot = [&](int j, int e) -> int* {
    const int p = j % PT_SLOTS;
    return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 2)) * PITCH + 128 + 4 * (e & 3));
  };
  auto issue_pt = [&](int j) {        // lanes 0..15: chunk j's page entries (join the open commit group)
    if (j < mine && lane < CHUNK) cp4(pt_slot(j, lane), pt + tok_of(j) / page);
  };
  // One commit group per call: chunk i's K/V rows (its page entries landed in an earlier group,
  // waited and warp-synced) and chunk i + STAGES's page entries.
  auto issue = [&](int i) {
    if (i < mine) {
      const long my_row = static_cast<long>(*pt_slot(i, lane & 15)) * page + tok_of(i) % page;
      uint8_t* st = ring + (i % STAGES) * STAGE_BYTES;
#pragma unroll
      for (int qq = 0; qq < 4; ++qq) {
        const int tok = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, my_row, tok);
        cp16(st + tok * PITCH + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(st + CHUNK * PITCH + tok * PITCH + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
    }
    issue_pt(i + STAGES);
    asm volatile("cp.async.commit_group;");
  };
  // Prologue: the first STAGES chunks' page entries (the one exposed round trip), then the first
  // STAGES - 1 stages of K/V rows.
  for (int j = 0; j < STAGES; ++j) issue_pt(j);
  asm volatile("cp.async.commit_group;");
  AN_PUT(2, an_clk());
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();
  AN_PUT(3, an_clk());
  AN_PUT(13, mine);
#pragma unroll
  for (int i = 0; i < STAGES - 1; ++i) issue(i);

  // ---- RoPE table for this request's position: one full-range sincosf per frequency (the
  // arguments reach ~1e5 rad, CUDA's slow reduction path), shared by every lane.
  if (rope) {
    if (warp == 0) sincosf(pos * inv_freq[lane], &rope_s[lane], &rope_c[lane]);
    __syncthreads();
  }
  AN_PUT(4, an_clk_m());

  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.
  if (has_new && warp == 0) {
    const int h = lane >> 4, d0 = 4 * (lane & 15);          // head kv0 + h, dims d0 .. d0+3
    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;
    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;
    float k4[4], v4[4];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      k4[j] = bf16f(kp[j]);
      v4[j] = bf16f(vp[j]);
      ss += k4[j] * k4[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);
    ss += __shfl_xor_sync(FULL, ss, 2);
    ss += __shfl_xor_sync(FULL, ss, 4);
    ss += __shfl_xor_sync(FULL, ss, 8);                       // sum over the head's 16 lanes
    const float r = rsqrtf(ss / 64.f + 1e-5f);
#pragma unroll
    for (int j = 0; j < 4; ++j) k4[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 2; ++p) {
        const float sn = rope_s[d0 / 2 + p], cs = rope_c[d0 / 2 + p];
        const float e = k4[2 * p], o = k4[2 * p + 1];
        k4[2 * p] = e * cs - o * sn;
        k4[2 * p + 1] = e * sn + o * cs;
      }
    }
    const float kinv = 1.f / (*k_scale_p), vinv = 1.f / (*v_scale_p);
    const uint32_t kw = e4m3x2(k4[0] * kinv, k4[1] * kinv) | (e4m3x2(k4[2] * kinv, k4[3] * kinv) << 16);
    const uint32_t vw = e4m3x2(v4[0] * vinv, v4[1] * vinv) | (e4m3x2(v4[2] * vinv, v4[3] * vinv) << 16);
    reinterpret_cast<uint32_t*>(new_k)[lane] = kw;            // bytes 4*lane .. = head h, dims d0 ..
    reinterpret_cast<uint32_t*>(new_v)[lane] = vw;
    const long row = cache_loc[b];
    reinterpret_cast<uint32_t*>(kc + row * row_bytes + poff)[lane] = kw;
    reinterpret_cast<uint32_t*>(vc + row * row_bytes + poff)[lane] = vw;
  }

  // ---- queries: lane = head hl, dims 32*(t4&1) .. +31 of query g; RMSNorm x factor, RoPE, FP16.
  uint32_t qa[8][4];
  {
    const uint16_t* qp = qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);
    float x[32];
    float ss = 0.f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
      x[j] = bf16f(qp[j]);
      ss += x[j] * x[j];
    }
    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)
    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;
#pragma unroll
    for (int j = 0; j < 32; ++j) x[j] *= r;
    if (rope) {
#pragma unroll
      for (int p = 0; p < 16; ++p) {
        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];
        const float e = x[2 * p], o = x[2 * p + 1];
        x[2 * p] = e * cs - o * sn;
        x[2 * p + 1] = e * sn + o * cs;
      }
    }
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);
      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);
      qa[ks][0] = hl == 0 ? lo2 : 0u;
      qa[ks][1] = hl == 1 ? lo2 : 0u;
      qa[ks][2] = hl == 0 ? hi2 : 0u;
      qa[ks][3] = hl == 1 ? hi2 : 0u;
    }
  }
  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp
  AN_DEP(5, qa[7][0]);

  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[16][4];
#pragma unroll
  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;

  unsigned long long an_wacc = 0;
  for (int i = 0; i < mine; ++i) {
    issue(i + STAGES - 1);
    const unsigned long long an_w0 = an_clk();
    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));
    __syncwarp();
    const unsigned long long an_w1 = an_clk();
    an_wacc += an_w1 - an_w0;
    if (i == 0) AN_PUT(6, an_w1);
    const int c = warp + i * WARPS;
    uint8_t* stw = ring + (i % STAGES) * STAGE_BYTES;
    if (has_new && n - 1 >= start + c * CHUNK && n - 1 < start + (c + 1) * CHUNK) {
      // the new token's row: take the key/value computed in-kernel (the copy read the cache row
      // before or while it was being written)
      const int tn = n - 1 - (start + c * CHUNK);
      reinterpret_cast<uint32_t*>(stw + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_k)[lane];
      reinterpret_cast<uint32_t*>(stw + CHUNK * PITCH + tn * PITCH)[lane] = reinterpret_cast<const uint32_t*>(new_v)[lane];
      __syncwarp();
    }
    const uint8_t* st = stw;
    const uint8_t* sk = st;
    const uint8_t* sv = st + CHUNK * PITCH;
    Chunk cur;
    cur.k[0] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4);
    cur.k[1] = *reinterpret_cast<const uint4*>(sk + g * PITCH + 32 * t4 + 16);
    cur.k[2] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4);
    cur.k[3] = *reinterpret_cast<const uint4*>(sk + (8 + g) * PITCH + 32 * t4 + 16);
    cur.v[0] = *reinterpret_cast<const uint4*>(sv + (2 * t4) * PITCH + 16 * g);
    cur.v[1] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 1) * PITCH + 16 * g);
    cur.v[2] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 8) * PITCH + 16 * g);
    cur.v[3] = *reinterpret_cast<const uint4*>(sv + (2 * t4 + 9) * PITCH + 16 * g);
    const int s0 = start + c * CHUNK;

    float sc[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);
        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));
      }
    }
    // c0,c1 head 0; c2,c3 head 1 (query g), tokens nt*8 + 2*t4 + jj.
    float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;
        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;
        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;
        mx0 = fmaxf(mx0, sc[nt][jj]);
        mx1 = fmaxf(mx1, sc[nt][2 + jj]);
      }
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));
    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));
    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));
    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);
    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);
    float p[2][4];
#pragma unroll
    for (int nt = 0; nt < 2; ++nt) {
      p[nt][0] = ex2(sc[nt][0] - mn0);
      p[nt][1] = ex2(sc[nt][1] - mn0);
      p[nt][2] = ex2(sc[nt][2] - mn1);
      p[nt][3] = ex2(sc[nt][3] - mn1);
    }
    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
    m_run[0] = mn0;
    m_run[1] = mn1;
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);   // row g (head 0), tokens 2*t4, 2*t4+1
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);   // row g+8 (head 1)
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);   // row g, tokens 8+2*t4, 9+2*t4
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);   // row g+8
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      acc[j][0] *= al0;
      acc[j][1] *= al0;
      acc[j][2] *= al1;
      acc[j][3] *= al1;
      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));
      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));
      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);
    }
    __syncwarp();          // every lane done reading this stage before it is refilled
  }
  AN_PUT(7, an_clk_m());
  AN_PUT(8, an_wacc);
  asm volatile("cp.async.wait_all;");
  __syncthreads();                     // every warp is done with its ring: the warp merge reuses it
  static_assert(WARPS * 2 * 8 * 64 * 4 + 2 * WARPS * 2 * 8 * 4 <= WARPS * STAGES * STAGE_BYTES, "merge fits the ring");
  auto sm_o = reinterpret_cast<float (*)[2][8][64]>(dsm);
  auto sm_m = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4);
  auto sm_l = reinterpret_cast<float (*)[2][8]>(dsm + WARPS * 2 * 8 * 64 * 4 + WARPS * 2 * 8 * 4);

  // Merge the warps through shared memory (output d un-permuted), then write the partials.
  float l0 = l_run[0], l1 = l_run[1];
  l0 += __shfl_xor_sync(FULL, l0, 1);
  l0 += __shfl_xor_sync(FULL, l0, 2);
  l1 += __shfl_xor_sync(FULL, l1, 1);
  l1 += __shfl_xor_sync(FULL, l1, 2);
  if (t4 == 0) {
    sm_m[warp][0][g] = m_run[0];
    sm_m[warp][1][g] = m_run[1];
    sm_l[warp][0][g] = l0;
    sm_l[warp][1][g] = l1;
  }
#pragma unroll
  for (int j = 0; j < 16; ++j)
#pragma unroll
    for (int jj = 0; jj < 2; ++jj)
      sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  AN_PUT(9, an_clk_m());
  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {
    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;
    float mm = -1.0e30f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);
    float o = 0.f, l = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) {
      const float wt = ex2(sm_m[w][h][row] - mm);
      o += wt * sm_o[w][h][row][d];
      l += wt * sm_l[w][h][row];
    }
    const int hr = h * 8 + row;                    // pair-local (head, row) index 0..15
    cta_o[hr][d] = o;
    if (d == 0) {
      cta_m[hr] = mm;
      cta_l[hr] = l;
    }
  }

  AN_PUT(10, an_clk_m());
  // ---- A42 cluster merge: the leader (rank 0) reads every rank's partial over DSMEM.
  const uint32_t rank = cluster_rank();
  const int nc = nsplit / CS;                       // clusters per (request, pair)
  const int ci = split / CS;
  cluster_sync();                                   // every CTA's partial written and visible
  AN_PUT(11, an_clk_m());
  constexpr int RPW = 16 / WARPS;                   // (head, row)s per leader warp
  float fo0[RPW], fo1[RPW], fl[RPW], fm[RPW];
  if (rank == 0) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) {
      const int hr = warp + k2 * WARPS;
      fm[k2] = -1.0e30f; fl[k2] = 0.f; fo0[k2] = 0.f; fo1[k2] = 0.f;
      if (hr < 16) {
        float mm = -1.0e30f;
#pragma unroll
        for (int r = 0; r < CS; ++r) mm = fmaxf(mm, ld_peer(&cta_m[hr], r));
        float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll
        for (int r = 0; r < CS; ++r) {
          const float wt = ex2(ld_peer(&cta_m[hr], r) - mm);
          l += wt * ld_peer(&cta_l[hr], r);
          o0 += wt * ld_peer(&cta_o[hr][lane], r);
          o1 += wt * ld_peer(&cta_o[hr][32 + lane], r);
        }
        fm[k2] = mm; fl[k2] = l; fo0[k2] = o0; fo1[k2] = o1;
      }
    }
  }
  cluster_sync();                                   // the leader is done reading: peers may exit
  AN_PUT(12, an_clk_m());
  if (rank != 0) { AN_END(); return; }

  const float vsc = *v_scale_p;
  auto finalize = [&](int hr, float o0, float o1, float l) {   // RMSNorm x sigmoid(gate), BF16 store
    const float scale = vsc / l;
    o0 *= scale;
    o1 *= scale;
    float ss = o0 * o0 + o1 * o1;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);
    const float r = rsqrtf(ss / 64.f + 1e-5f);
    const int head = kv0 * 8 + hr;
    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + head * 64;
    const float g0 = bf16f(gp[lane]), g1 = bf16f(gp[32 + lane]);
    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;
    op[lane] = f2bf16(o0 * r / (1.f + __expf(-g0)));
    op[32 + lane] = f2bf16(o1 * r / (1.f + __expf(-g1)));
  };
  if (nc == 1) {
#pragma unroll
    for (int k2 = 0; k2 < RPW; ++k2) finalize(warp + k2 * WARPS, fo0[k2], fo1[k2], fl[k2]);
    AN_END();
    return;
  }

  // ---- A43 second level: cluster partials in global memory; the last leader (ticket) merges.
  const long cbase = (static_cast<long>(b) * hq + kv0 * 8) * nc;      // (head kv0*8 + hr, cluster)
#pragma unroll
  for (int k2 = 0; k2 < RPW; ++k2) {
    const int hr = warp + k2 * WARPS;
    {
      const long p = cbase + static_cast<long>(hr) * nc + ci;
      cpart_o[p * 64 + lane] = fo0[k2];
      cpart_o[p * 64 + 32 + lane] = fo1[k2];
      if (lane == 0) {
        cpart_m[p] = fm[k2];
        cpart_l[p] = fl[k2];
      }
    }
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last_flag = atomicAdd(tickets + blockIdx.y, 1) == nc - 1;
  __syncthreads();
  if (!last_flag) { AN_END(); return; }
  __threadfence();
  for (int hr = warp; hr < 16; hr += WARPS) {       // nc <= 64: one warp per (head, row), lanes over clusters
    const long pb = cbase + static_cast<long>(hr) * nc;
    float mm = -1.0e30f;
    for (int c2 = lane; c2 < nc; c2 += 32) mm = fmaxf(mm, __ldcg(cpart_m + pb + c2));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll 4
    for (int c2 = 0; c2 < nc; ++c2) {
      const float wt = ex2(__ldcg(cpart_m + pb + c2) - mm);
      l += wt * __ldcg(cpart_l + pb + c2);
      o0 += wt * __ldcg(cpart_o + (pb + c2) * 64 + lane);
      o1 += wt * __ldcg(cpart_o + (pb + c2) * 64 + 32 + lane);
    }
    finalize(hr, o0, o1, l);
  }
  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;    // ready for the next call (and replay)
  AN_END();
}

}  // namespace okc_an_ud2

void an_ud2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc,
        torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens,
        torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out,
        int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window,
        int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs, torch::Tensor an_buf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(warps) * stages * okc_an_ud2::STAGE_BYTES;
#define OKC_UD2_ARGS                                                                                           \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(), seqlens.data_ptr<int>(), \
      cpart_o.data_ptr<float>(), cpart_m.data_ptr<float>(), cpart_l.data_ptr<float>(), tickets.data_ptr<int>(), \
      reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq), static_cast<int>(hkv),             \
      static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), static_cast<int>(window), \
      static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale), reinterpret_cast<unsigned long long*>(an_buf.data_ptr())
  if (cs == 16) {
    if (warps == 8) okc_an_ud2::an_ud2_kernel<8, 2, 16><<<grid, 256, smem, stream>>>(OKC_UD2_ARGS);
    else okc_an_ud2::an_ud2_kernel<4, 3, 16><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
  } else {
    if (warps == 8) okc_an_ud2::an_ud2_kernel<8, 2, 8><<<grid, 256, smem, stream>>>(OKC_UD2_ARGS);
    else if (stages == 4) okc_an_ud2::an_ud2_kernel<4, 4, 8><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
    else okc_an_ud2::an_ud2_kernel<4, 3, 8><<<grid, 128, smem, stream>>>(OKC_UD2_ARGS);
  }
#undef OKC_UD2_ARGS
  const cudaError_t e = cudaGetLastError();        // surface a refused (cluster) launch here, not later
  TORCH_CHECK(e == cudaSuccess, "an_ud2 launch failed: ", cudaGetErrorString(e));
}

int64_t an_ud2_setup() {   // once, outside any capture: each variant's ring, and 16-CTA clusters
  int err = 0;
  using okc_an_ud2::STAGE_BYTES;
#define OKC_UD2_SMEM(W, S, C) \
  err |= cudaFuncSetAttribute(okc_an_ud2::an_ud2_kernel<W, S, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, W * S * STAGE_BYTES)
  OKC_UD2_SMEM(4, 3, 8);
  OKC_UD2_SMEM(4, 4, 8);
  OKC_UD2_SMEM(8, 2, 8);
  OKC_UD2_SMEM(4, 3, 16);
  OKC_UD2_SMEM(8, 2, 16);
#undef OKC_UD2_SMEM
  err |= cudaFuncSetAttribute(okc_an_ud2::an_ud2_kernel<4, 3, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  err |= cudaFuncSetAttribute(okc_an_ud2::an_ud2_kernel<8, 2, 16>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> an_ud2_attrs(int64_t warps, int64_t stages, int64_t cs) {
  cudaFuncAttributes a;
  if (cs == 16) {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_an_ud2::an_ud2_kernel<8, 2, 16>);
    else cudaFuncGetAttributes(&a, okc_an_ud2::an_ud2_kernel<4, 3, 16>);
  } else {
    if (warps == 8) cudaFuncGetAttributes(&a, okc_an_ud2::an_ud2_kernel<8, 2, 8>);
    else if (stages == 4) cudaFuncGetAttributes(&a, okc_an_ud2::an_ud2_kernel<4, 4, 8>);
    else cudaFuncGetAttributes(&a, okc_an_ud2::an_ud2_kernel<4, 3, 8>);
  }
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_AN_UD2_DECL = 'void an_ud2(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor cpart_o, torch::Tensor cpart_m, torch::Tensor cpart_l, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps, int64_t stages, int64_t cs, torch::Tensor an_buf);\nint64_t an_ud2_setup();\nstd::vector<int64_t> an_ud2_attrs(int64_t warps, int64_t stages, int64_t cs);'

CUDA_AN_VER3_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_an_ver3 {

// ---- L15 anatomy stamps (lab only) ----
__device__ __forceinline__ unsigned long long an_clk() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned long long an_clk_m() {   // also ordered against memory operations
  unsigned long long t;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_gt() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t) :: "memory");
  return t;
}
__device__ __forceinline__ unsigned long long an_smid() {
  unsigned r;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(r));
  return r;
}
#define AN_PUT(k, v) do { if (threadIdx.x == 0) an_buf[an_base + (k)] = (v); } while (0)
#define AN_PP(k, v) do { if (threadIdx.x == 128) an_buf[an_base + (k)] = (v); } while (0)
#define AN_DEP(k, an_v) do { if (threadIdx.x == 0) { an_buf[an_base + 31] = static_cast<unsigned long long>(an_v); \
    an_buf[an_base + (k)] = an_clk_m(); } } while (0)
#define AN_END() do { if (threadIdx.x == 0) { an_buf[an_base + 28] = an_clk_m(); an_buf[an_base + 29] = an_gt(); } } while (0)


constexpr int CHUNK = 16;
constexpr int L = 8;
constexpr int KP = 192;   // K row pitch in shared memory (bytes)
constexpr int VP = 144;   // V row pitch
constexpr int SKEYS = 64; // keys per ring stage
constexpr int STAGE_BYTES = SKEYS * (KP + VP);
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}
__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp4(void* dst, const void* src) {   // 4-byte copy (page entries)
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ uint32_t w4(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

template <int STAGES>
__global__ void __launch_bounds__(160) an_ver3_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale, unsigned long long* __restrict__ an_buf) {
  const long an_base = (static_cast<long>(blockIdx.y) * gridDim.x + blockIdx.x) * 32;
  if (threadIdx.x == 0) { an_buf[an_base + 0] = an_gt(); an_buf[an_base + 1] = an_clk(); an_buf[an_base + 30] = an_smid(); }

  extern __shared__ __align__(16) uint8_t ring[];   // STAGES x (64 K rows at KP, 64 V rows at VP)
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int h = warp >> 1, mt0 = 2 * (warp & 1);
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;

  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l
  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Query fragments: m-tile mt = mt0 + mi, rows g -> token 2*mt, g+8 -> token 2*mt+1, head (kv0+h)*8+g.
  uint32_t qa[2][4][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    const int ta = 2 * (mt0 + mi);
    const uint16_t* qA = q + ((static_cast<long>(b) * L + ta) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;
    const uint16_t* qB = qA + static_cast<long>(hq) * 64;               // token ta + 1
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[mi][ks][0] = pack_f16x2(bf16f(qA[4 * ks + 0]), bf16f(qA[4 * ks + 1]));
      qa[mi][ks][1] = pack_f16x2(bf16f(qB[4 * ks + 0]), bf16f(qB[4 * ks + 1]));
      qa[mi][ks][2] = pack_f16x2(bf16f(qA[4 * ks + 2]), bf16f(qA[4 * ks + 3]));
      qa[mi][ks][3] = pack_f16x2(bf16f(qB[4 * ks + 2]), bf16f(qB[4 * ks + 3]));
    }
  }

  AN_DEP(2, qa[1][3][3]);
  const int* pt = page_table + static_cast<long>(b) * max_pages;
  const int nst = (nchunks + 3) / 4;                  // 64-key stages of this slice
  float m_run[2][2], l_run[2][2], acc[2][8][4];
#pragma unroll
  for (int mi = 0; mi < 2; ++mi) {
    m_run[mi][0] = m_run[mi][1] = -1.0e30f;
    l_run[mi][0] = l_run[mi][1] = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;
  }

  if (tid == 0) {
#pragma unroll
    for (int s = 0; s < STAGES; ++s) {
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" ::"r"(smem_u32(&full_bar[s])));   // producer lanes
      asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;" ::"r"(smem_u32(&empty_bar[s])));   // consumer warps
    }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();

  if (warp == 4) {
    AN_PP(16, an_clk_m());
    // ---- producer: whole-row 16 B copies of each 64-key stage. Page entries ride the ring: stage
    // j's 64 entries sit in PT slot p = j % PT_SLOTS, in the K rows' unused pads (bytes 128..191) of
    // ring slot p % STAGES, rows 4*(p / STAGES) .. +3: entry e at row 4*(p / STAGES) + e / 16, byte
    // 128 + 4*(e % 16). Consumers read K bytes 0..127 only; row copies write bytes 0..127.
    constexpr int PT_SLOTS = 2 * STAGES;
    auto pt_slot = [&](int j, int e) -> int* {
      const int p = j % PT_SLOTS;
      return reinterpret_cast<int*>(ring + (p % STAGES) * STAGE_BYTES + (4 * (p / STAGES) + (e >> 4)) * KP + 128 + 4 * (e & 15));
    };
    auto tok = [&](int j, int e) -> int { return min(start + j * SKEYS + e, end - 1); };
    auto issue_pt = [&](int j) {        // stage j's 64 page entries, 2 per lane (join the open group)
      if (j < nst) {
        cp4(pt_slot(j, lane), pt + tok(j, lane) / page);
        cp4(pt_slot(j, 32 + lane), pt + tok(j, 32 + lane) / page);
      }
    };
    for (int j = 0; j < STAGES; ++j) issue_pt(j);   // the one exposed page-table round trip
    asm volatile("cp.async.commit_group;");
    AN_PP(17, an_clk());
    asm volatile("cp.async.wait_group 0;");
    __syncwarp();
    AN_PP(18, an_clk());
    unsigned long long an_eacc = 0, an_pacc = 0;
    for (int i = 0; i < nst; ++i) {
      const int st = i % STAGES;
      const unsigned long long an_e0 = an_clk();
      if (i >= STAGES) {                              // wait until the consumers released this stage
        const uint32_t parity = static_cast<uint32_t>((i / STAGES - 1) & 1);
        uint32_t done = 0;
        while (!done)
          asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                       : "=r"(done) : "r"(smem_u32(&empty_bar[st])), "r"(parity) : "memory");
      }
      an_eacc += an_clk() - an_e0;
      // stage i's rows from its page entries (landed: waited and warp-synced after stage i-1)
      const long r0 = static_cast<long>(*pt_slot(i, lane)) * page + tok(i, lane) % page;
      const long r1 = static_cast<long>(*pt_slot(i, 32 + lane)) * page + tok(i, 32 + lane) % page;
      uint8_t* skw = ring + st * STAGE_BYTES;
      uint8_t* svw = skw + SKEYS * KP;
#pragma unroll
      for (int qq = 0; qq < 16; ++qq) {               // 8 lanes per whole row, 4 rows per instruction
        const int tq = qq * 4 + (lane >> 3), ch = lane & 7;
        const long row = __shfl_sync(FULL, qq < 8 ? r0 : r1, tq & 31);
        cp16(skw + tq * KP + 16 * ch, kc + row * row_bytes + poff + 16 * ch);
        cp16(svw + tq * VP + 16 * ch, vc + row * row_bytes + poff + 16 * ch);
      }
      asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" ::"r"(smem_u32(&full_bar[st])) : "memory");
      issue_pt(i + STAGES);
      asm volatile("cp.async.commit_group;");
      const unsigned long long an_p0 = an_clk();
      asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 1));   // stage i+1's entries landed
      __syncwarp();
      an_pacc += an_clk() - an_p0;
    }
    AN_PP(19, an_clk_m());
    asm volatile("cp.async.wait_all;");
    AN_PP(20, an_clk_m());
    AN_PP(21, an_eacc);
    AN_PP(22, an_pacc);
    AN_PP(23, an_gt());
    return;                                           // the producer writes no partials
  }
  // ---- consumers: wait for a stage, compute its four 16-key sub-chunks, release it.
  unsigned long long an_facc = 0;
  for (int i = 0; i < nst; ++i) {
    const int st = i % STAGES;
    const uint32_t parity = static_cast<uint32_t>((i / STAGES) & 1);
    const unsigned long long an_f0 = an_clk();
    uint32_t done = 0;
    while (!done)
      asm volatile("{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n\tselp.u32 %0, 1, 0, p;\n\t}"
                   : "=r"(done) : "r"(smem_u32(&full_bar[st])), "r"(parity) : "memory");
    {
      const unsigned long long an_f1 = an_clk();
      an_facc += an_f1 - an_f0;
      if (i == 0) AN_PUT(3, an_f1);
    }
    const uint8_t* sk = ring + st * STAGE_BYTES;
    const uint8_t* sv = sk + SKEYS * KP;
#pragma unroll 1
    for (int sc = 0; sc < 4; ++sc) {
      const int c = i * 4 + sc;
      if (c >= nchunks) break;
        const int s0 = start + c * CHUNK;
        const uint4 kb0 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + g) * KP + 64 * h + 16 * t4]);
        const uint4 kb1 = *reinterpret_cast<const uint4*>(&sk[(16 * sc + 8 + g) * KP + 64 * h + 16 * t4]);
        const uint2 vb0 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4) * VP + 64 * h + 8 * g]);
        const uint2 vb1 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 1) * VP + 64 * h + 8 * g]);
        const uint2 vb2 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 8) * VP + 64 * h + 8 * g]);
        const uint2 vb3 = *reinterpret_cast<const uint2*>(&sv[(16 * sc + 2 * t4 + 9) * VP + 64 * h + 8 * g]);
        // a chunk needs masks only at the slice tail or where it reaches the block's own keys
        const bool masked = s0 + CHUNK > end || s0 + CHUNK > n - L || window > 0;

    #pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          float sc[2][4];
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            const uint4& kb = nt == 0 ? kb0 : kb1;
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
    #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
              const uint32_t w = w4(kb, ks);
              mma16816(sc[nt], qa[mi][ks][0], qa[mi][ks][1], qa[mi][ks][2], qa[mi][ks][3], f16x2_lo(w), f16x2_hi(w));
            }
          }
          const int posA = n - L + 2 * (mt0 + mi), posB = posA + 1;     // rows g / g+8
          float mxA = -INFINITY, mxB = -INFINITY;
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt)
    #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
              const int key = s0 + nt * 8 + 2 * t4 + jj;
              float xa = sc[nt][jj] * qk_scale, xb = sc[nt][2 + jj] * qk_scale;
              if (masked) {
                const bool ok = key < end;
                const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);
                const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);
                xa = okA ? xa : -INFINITY;
                xb = okB ? xb : -INFINITY;
              }
              sc[nt][jj] = xa;
              sc[nt][2 + jj] = xb;
              mxA = fmaxf(mxA, xa);
              mxB = fmaxf(mxB, xb);
            }
          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));
          mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));
          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));
          mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));
          const float mnA = fmaxf(m_run[mi][0], mxA), mnB = fmaxf(m_run[mi][1], mxB);
          const float alA = ex2(m_run[mi][0] - mnA), alB = ex2(m_run[mi][1] - mnB);
          float p[2][4];
    #pragma unroll
          for (int nt = 0; nt < 2; ++nt) {
            p[nt][0] = ex2(sc[nt][0] - mnA);
            p[nt][1] = ex2(sc[nt][1] - mnA);
            p[nt][2] = ex2(sc[nt][2] - mnB);
            p[nt][3] = ex2(sc[nt][3] - mnB);
          }
          l_run[mi][0] = l_run[mi][0] * alA + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);
          l_run[mi][1] = l_run[mi][1] * alB + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);
          m_run[mi][0] = mnA;
          m_run[mi][1] = mnB;
          const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);
          const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);
          const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);
          const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);
    #pragma unroll
          for (int j = 0; j < 8; ++j) {
            acc[mi][j][0] *= alA;
            acc[mi][j][1] *= alA;
            acc[mi][j][2] *= alB;
            acc[mi][j][3] *= alB;
            const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
            const uint32_t b0 = f16x2_lo(__byte_perm(j < 4 ? vb0.x : vb0.y, j < 4 ? vb1.x : vb1.y, sel));
            const uint32_t b1 = f16x2_lo(__byte_perm(j < 4 ? vb2.x : vb2.y, j < 4 ? vb3.x : vb3.y, sel));
            mma16816(acc[mi][j], pa0, pa1, pa2, pa3, b0, b1);
          }
        }

    }
    __syncwarp();
    if (lane == 0) {
      uint64_t state;
      asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];" : "=l"(state) : "r"(smem_u32(&empty_bar[st])) : "memory");
      (void)state;
    }
  }

  AN_PUT(4, an_clk_m());
  AN_PUT(5, an_facc);
  AN_PUT(6, nst);
  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4+jj) + j.
#pragma unroll
  for (int mi = 0; mi < 2; ++mi)
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      float l = l_run[mi][r];
      l += __shfl_xor_sync(FULL, l, 1);
      l += __shfl_xor_sync(FULL, l, 2);
      const int tok = 2 * (mt0 + mi) + r;
      const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;
#pragma unroll
      for (int j = 0; j < 8; ++j)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) o_part[part * 64 + 8 * (2 * t4 + jj) + j] = acc[mi][j][2 * r + jj];
      if (t4 == 0) {
        m_part[part] = m_run[mi][r];
        l_part[part] = l;
      }
    }
  AN_END();
}

}  // namespace okc_an_ver3

void an_ver3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                 torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                 torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                 int64_t page, int64_t window, double sm_scale, int64_t stages, torch::Tensor an_buf) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
  const size_t smem = static_cast<size_t>(stages) * okc_an_ver3::STAGE_BYTES;
#define OKC_VER3_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale), reinterpret_cast<unsigned long long*>(an_buf.data_ptr())
  if (stages == 4) okc_an_ver3::an_ver3_kernel<4><<<grid, 160, smem, stream>>>(OKC_VER3_ARGS);
  else if (stages == 2) okc_an_ver3::an_ver3_kernel<2><<<grid, 160, smem, stream>>>(OKC_VER3_ARGS);
  else okc_an_ver3::an_ver3_kernel<3><<<grid, 160, smem, stream>>>(OKC_VER3_ARGS);
#undef OKC_VER3_ARGS
  const cudaError_t e = cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "an_ver3 launch failed: ", cudaGetErrorString(e));
}

int64_t an_ver3_setup() {
  int err = 0;
  using okc_an_ver3::STAGE_BYTES;
  err |= cudaFuncSetAttribute(okc_an_ver3::an_ver3_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_an_ver3::an_ver3_kernel<3>, cudaFuncAttributeMaxDynamicSharedMemorySize, 3 * STAGE_BYTES);
  err |= cudaFuncSetAttribute(okc_an_ver3::an_ver3_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, 4 * STAGE_BYTES);
  (void)cudaGetLastError();
  return err;
}

std::vector<int64_t> an_ver3_attrs(int64_t stages) {
  cudaFuncAttributes a;
  if (stages == 4) cudaFuncGetAttributes(&a, okc_an_ver3::an_ver3_kernel<4>);
  else if (stages == 2) cudaFuncGetAttributes(&a, okc_an_ver3::an_ver3_kernel<2>);
  else cudaFuncGetAttributes(&a, okc_an_ver3::an_ver3_kernel<3>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_AN_VER3_DECL = 'void an_ver3(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t stages, torch::Tensor an_buf);\nint64_t an_ver3_setup();\nstd::vector<int64_t> an_ver3_attrs(int64_t stages);'

CUDA_AN_MICRO_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <vector>

namespace okc_an_micro {

// One thread follows a chain of dependent loads (ld.global.cg: L2, no L1); clock64 around it.
__global__ void chase_kernel(const int* __restrict__ next, int start, int steps, unsigned long long* __restrict__ out) {
  if (threadIdx.x != 0) return;
  int idx = start;
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
  for (int s = 0; s < steps; ++s) {
    int v;
    asm volatile("ld.global.cg.s32 %0, [%1];" : "=r"(v) : "l"(next + idx) : "memory");
    idx = v;
  }
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
  out[0] = t1 - t0;
  out[1] = static_cast<unsigned long long>(idx);
}

__global__ void null_kernel(int* __restrict__ sink) {
  if (sink == nullptr) return;
}

// One cold load and one store per CTA (thread 0): a kernel with one exposed DRAM round trip.
__global__ void one_trip_kernel(const int* __restrict__ src, int* __restrict__ dst) {
  if (threadIdx.x == 0) dst[blockIdx.x] = src[static_cast<long>(blockIdx.x) * 4096];
}

__global__ void __cluster_dims__(8, 1, 1) null_cluster_kernel(int* __restrict__ sink) {
  asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  if (sink == nullptr) return;
}

// Throughput: each thread runs `iters` x 8 independent operations of one kind; clock64 per CTA.
// KIND 0: mma.sync m16n8k16 f16 -> f32; 1: mma.sync m16n8k32 e4m3 -> f32; 2: ex2.approx.f32;
// 3: ex2.approx.f16x2; 4: cvt.rn.f16x2.e4m3x2. Values stay bounded; a never-true test keeps them live.
template <int KIND>
__global__ void bench_kernel(int iters, unsigned long long* __restrict__ out, float* __restrict__ sink) {
  const unsigned lane = threadIdx.x & 31;
  const uint32_t a0 = 0x3c003c00u ^ lane, a1 = a0 ^ 1u, a2 = a0 ^ 2u, a3 = a0 ^ 3u, b0 = 0x3c003c00u, b1 = b0 ^ 7u;
  float c[8][4];
  float e[8];
  uint32_t h[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    c[j][0] = c[j][1] = c[j][2] = c[j][3] = 0.f;
    e[j] = -static_cast<float>(lane + j) / 64.f;
    h[j] = 0xb800b800u ^ (lane << 3) ^ static_cast<uint32_t>(j);
  }
  __syncthreads();
  unsigned long long t0;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (KIND == 0) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 1) {
        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                     : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      } else if (KIND == 2) {
        float y;
        asm volatile("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(e[j]));
        e[j] = y - 1.0f;
      } else if (KIND == 3) {
        uint32_t y;
        asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(h[j]));
        h[j] = y ^ 0x80008000u;
      } else {
        uint32_t y;
        asm volatile("{\n\t.reg .b16 l, u;\n\tmov.b32 {l, u}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(y) : "r"(h[j]));
        h[j] = y ^ (h[j] >> 16);
      }
    }
  }
  unsigned long long t1;
  asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");
  float s = 0.f;
#pragma unroll
  for (int j = 0; j < 8; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3] + e[j] + static_cast<float>(h[j] & 0xffu);
  if (s == 1.2345e30f) sink[0] = s;
  if (threadIdx.x == 0) out[blockIdx.x] = t1 - t0;
}

}  // namespace okc_an_micro

void an_bench(int64_t kind, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor out, torch::Tensor sink) {
  auto stream = at::cuda::getCurrentCUDAStream();
  auto o = reinterpret_cast<unsigned long long*>(out.data_ptr());
  const dim3 grid(static_cast<unsigned>(ctas)), block(static_cast<unsigned>(threads));
  const int n = static_cast<int>(iters);
  float* sk = sink.data_ptr<float>();
  if (kind == 0) okc_an_micro::bench_kernel<0><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 1) okc_an_micro::bench_kernel<1><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 2) okc_an_micro::bench_kernel<2><<<grid, block, 0, stream>>>(n, o, sk);
  else if (kind == 3) okc_an_micro::bench_kernel<3><<<grid, block, 0, stream>>>(n, o, sk);
  else okc_an_micro::bench_kernel<4><<<grid, block, 0, stream>>>(n, o, sk);
}

void an_chase(torch::Tensor next, int64_t start, int64_t steps, torch::Tensor out) {
  okc_an_micro::chase_kernel<<<1, 32, 0, at::cuda::getCurrentCUDAStream()>>>(
      next.data_ptr<int>(), static_cast<int>(start), static_cast<int>(steps),
      reinterpret_cast<unsigned long long*>(out.data_ptr()));
}

// kind 0: empty kernel; 1: one cold load per CTA; 2: empty 8-CTA clusters with one cluster barrier.
// pdl: launch with programmatic stream serialization (the previous kernel in the stream may overlap).
void an_null(int64_t kind, int64_t ctas, int64_t threads, int64_t smem, int64_t pdl, torch::Tensor src, torch::Tensor dst) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(ctas), 1, 1);
  cfg.blockDim = dim3(static_cast<unsigned>(threads), 1, 1);
  cfg.dynamicSmemBytes = static_cast<size_t>(smem);
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  cudaError_t e;
  if (kind == 1) e = cudaLaunchKernelEx(&cfg, okc_an_micro::one_trip_kernel, static_cast<const int*>(src.data_ptr<int>()), dst.data_ptr<int>());
  else if (kind == 2) e = cudaLaunchKernelEx(&cfg, okc_an_micro::null_cluster_kernel, dst.data_ptr<int>());
  else e = cudaLaunchKernelEx(&cfg, okc_an_micro::null_kernel, dst.data_ptr<int>());
  TORCH_CHECK(e == cudaSuccess, "an_null launch failed: ", cudaGetErrorString(e));
}

int64_t an_null_setup(int64_t smem) {   // once: allow `smem` bytes of dynamic shared memory
  int err = 0;
  err |= cudaFuncSetAttribute(okc_an_micro::null_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  err |= cudaFuncSetAttribute(okc_an_micro::null_cluster_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  (void)cudaGetLastError();
  return err;
}
"""

CUDA_AN_MICRO_DECL = 'void an_chase(torch::Tensor next, int64_t start, int64_t steps, torch::Tensor out);\nvoid an_null(int64_t kind, int64_t ctas, int64_t threads, int64_t smem, int64_t pdl, torch::Tensor src, torch::Tensor dst);\nint64_t an_null_setup(int64_t smem);\nvoid an_bench(int64_t kind, int64_t ctas, int64_t threads, int64_t iters, torch::Tensor out, torch::Tensor sink);'


# ===== appended: attention/local/lab_l15_block.py =====


# ---------------------------------------------------------------------------------------------
# Lab L15 (check runs only): anatomy. Where do our kernels spend their time on latency-bound
# shapes? Instrumented copies of D3, UD2 and S2c (cuda_anat.py, generated by make_anatomy.py:
# the verified kernels plus %clock64 / %globaltimer stamps at phase boundaries) run in the same
# chains as their lab paths, timed cold (L2 flushed, as the judge) and hot; per-CTA stamps are
# summarized as start skew, durations and phase medians. Micro: dependent-load latency cold vs
# hot, across 2 MB pages vs inside one; graph cost of 1-3 empty kernels, PDL, 54 KB smem, one
# cold load, 8-CTA clusters. Triton chain: prologue / attend / combine alone and chained.
# The timed kernel_fn is the base submission's (v5) byte for byte.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 430.0
_AN = {}
_AN_TICKETS = []


def _an_build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


class _AnHost:
    """Host reads for the lab (eager, before any capture): stamp buffers and hot timings."""

    @staticmethod
    def rows(buf, n):
        return buf[: n * 32].view(n, 32).cpu().tolist()

    @staticmethod
    def vals(t):
        return t.cpu().tolist()

    @staticmethod
    def time_hot(fn, reps=15):
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
            graph.replay()
            start.record()
            graph.replay()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        return _st.median(times)


def _an_q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def _an_report(tag, rows, phases, accs=(), counts=(), leader=None, extra=""):
    """rows: per-CTA slot lists. phases: (label, slot a, slot b) clock deltas; accs: (label, slot)
    accumulated clocks; counts: (label, slot) raw values; leader(i) -> bool picks the rows of the
    `fin` phase set (UD2 cluster leaders)."""
    idx = [i for i, r in enumerate(rows) if r[1] and r[28]]
    if not idx:
        _log(f"AN {tag}: no stamps")
        return
    rs = [rows[i] for i in idx]
    e = [r[0] for r in rs]
    x = [r[29] for r in rs]
    t0, first_end = min(e), min(x)
    durs = [b - a for a, b in zip(e, x)]
    f = [(r[28] - r[1]) / (r[29] - r[0]) for r in rs if r[29] - r[0] > 2000]
    ghz = _an_q(f, 0.5) if f else 1.755
    ghz = ghz if 0.8 < ghz < 2.2 else 1.755     # implausible (coarse timer): nominal clock
    sms = {}
    for r in rs:
        sms[r[30]] = sms.get(r[30], 0) + 1
    ent = [v - t0 for v in e]
    distinct = sorted(set(ent))
    res = min((b - a for a, b in zip(distinct, distinct[1:])), default=0)
    parts = []
    for label, a, b, *only in phases:
        sel = [rows[i] for i in idx if not only or leader(i)]
        vals = [(r[b] - r[a]) / ghz / 1000 for r in sel if r[a] and r[b]]
        parts.append(f"{label} {_an_q(vals, .5):.1f}/{_an_q(vals, .9):.1f}")
    for label, s in accs:
        parts.append(f"{label} {_an_q([r[s] / ghz / 1000 for r in rs], .5):.1f}")
    for label, s in counts:
        parts.append(f"{label} {_an_q([r[s] for r in rs], .5)}")
    _log(f"AN {tag}: n{len(rs)} sm{len(sms)} x{max(sms.values())} span {(max(x) - t0) / 1000:.1f} "
         f"start {_an_q(ent, .5) / 1000:.1f}/{_an_q(ent, .9) / 1000:.1f}/{max(ent) / 1000:.1f} "
         f"late{sum(1 for v in e if v > first_end)} dur {_an_q(durs, .5) / 1000:.1f}/{max(durs) / 1000:.1f} | "
         + " ".join(parts) + f" | {ghz:.2f}GHz res{res}ns {extra}")


D3_PHASES = [("seq", 1, 2), ("pt", 2, 3), ("kv1", 3, 4), ("loop", 4, 5), ("mrg", 5, 7), ("st", 7, 28)]
UD2_PHASES = [("seq", 1, 2), ("pt", 2, 3), ("rope", 3, 4), ("fold", 4, 5), ("kv1", 5, 6), ("loop", 6, 7),
              ("wm", 7, 9), ("part", 9, 10), ("cb1", 10, 11), ("lead", 11, 12), ("fin", 12, 28, True)]
VER3_PHASES = [("q", 1, 2), ("first", 2, 3), ("loop", 3, 4), ("st", 4, 28), ("p0", 1, 16), ("ppt", 17, 18),
               ("ploop", 18, 19), ("pdrain", 19, 20)]


def _an_nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // min_keys, MAX_SPLITS))


def _an_q_prologue(data):
    cfg = data["config"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = cfg["batch"] * cfg["q_len"]
    rope = bool(cfg["rope_theta"])
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=data["qkv"].device)
    _prologue[(tokens,)](data["qkv"], data["positions"], data["cache_loc"],
                         _inv_freq(cfg["rope_theta"] if rope else 1.0, q.device), q, data["k_cache"], data["v_cache"],
                         data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, ROPE=rope,
                         QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    return q


def _an_combine(data, o_part, m_part, l_part, nsplit):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    _combine[(batch * length, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit,
                                   HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)),
                                   num_warps=4)
    return out


def _an_d3(data, cps, warps, stages, buf):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    nsplit = _an_nsplit(cfg, max_pages, cps, 32 * warps)
    q = _an_q_prologue(data)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _AN["d3"].an_dec4(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                      o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                      cfg["sm_scale"], warps, stages, buf)
    return _an_combine(data, o_part, m_part, l_part, nsplit)


def _an_ud2_split(cfg, max_pages, cs, cps, warps):
    if cfg["window"]:
        return 1
    tokens = max_pages * cfg["page_size"]
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    nc = max(1, math.ceil(cps * NUM_SMS / (programs * cs)))
    return max(1, min(nc, tokens // (cs * 32 * warps)))


def _an_ud2(data, cs, cps, warps, stages, buf):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    device = data["qkv"].device
    max_pages = data["page_table"].shape[1]
    nc = _an_ud2_split(cfg, max_pages, cs, cps, warps)
    rope = bool(cfg["rope_theta"])
    cpart_o = torch.empty((batch * hq * nc, 64), dtype=torch.float32, device=device)
    cpart_m = torch.empty((batch * hq * nc,), dtype=torch.float32, device=device)
    cpart_l = torch.empty_like(cpart_m)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _AN["ud2"].an_ud2(data["qkv"], data["positions"], data["cache_loc"],
                      _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"], data["v_cache"],
                      data["k_scale"], data["v_scale"], data["page_table"], data["cache_seqlens"], cpart_o, cpart_m,
                      cpart_l, _AN_TICKETS[0], out, batch, hq, hkv, max_pages, cs * nc, cfg["page_size"],
                      cfg["window"], int(rope), cfg["qk_scale_factor"], cfg["sm_scale"], warps, stages, cs, buf)
    return out


def _an_ver3(data, cps, stages, buf):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    nsplit = _an_nsplit(cfg, max_pages, cps, 64)
    q = _an_q_prologue(data)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _AN["ver3"].an_ver3(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                        data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                        cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, buf)
    return _an_combine(data, o_part, m_part, l_part, nsplit)


def _an_triton_parts(data):
    """v2.2's run() split into its three launches over preallocated tensors."""
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = batch * length
    qkv = data["qkv"]
    max_pages = data["page_table"].shape[1]
    nsplit, block, warps, stages, cb, mp = plan(cfg, max_pages, None)
    rope = bool(cfg["rope_theta"])
    inv_freq = _inv_freq(cfg["rope_theta"] if rope else 1.0, qkv.device)
    q = torch.empty((tokens, hq, 64), dtype=torch.bfloat16, device=qkv.device)
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=qkv.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=qkv.device)
    l_part = torch.empty_like(m_part)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=qkv.device)

    def p():
        _prologue[(tokens,)](qkv, data["positions"], data["cache_loc"], inv_freq, q, data["k_cache"], data["v_cache"],
                             data["k_scale"], data["v_scale"], HQ=hq, HKV=hkv, ROPE=rope,
                             QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)

    def a():
        _attend[(batch, hkv // 2, nsplit)](
            q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part, max_pages, nsplit, HQ=hq, HKV=hkv, L=length, PAGE=cfg["page_size"],
            WINDOW=cfg["window"], SM_SCALE=cfg["sm_scale"], MP=mp, BLOCK=block, num_warps=warps, num_stages=stages)

    def c():
        _combine[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                               HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, num_warps=4)

    return p, a, c


def _an_micro(lab):
    mod = _AN["micro"]
    dev = torch.device("cuda")
    out = torch.zeros(2, dtype=torch.int64, device=dev)
    words, page_words, hops = (512 << 20) // 4, (2 << 20) // 4, 64
    nxt = torch.zeros(words, dtype=torch.int32, device=dev)
    gc = torch.Generator().manual_seed(77)
    pages = torch.randperm(words // page_words, generator=gc)[:hops]
    lines = torch.randint(0, page_words // 32, (hops,), generator=gc)
    chains = (("64 pages", (pages * page_words + lines * 32).long()),
              ("1 page", (int(pages[0]) * page_words + torch.randperm(page_words // 32, generator=gc)[:hops] * 32).long()))
    for name, pos in chains:
        p = pos.to(dev)
        nxt[p] = torch.roll(p, -1).to(torch.int32)
        res = []
        for cold in (True, False, True):
            if cold:
                lab.flush.fill_(1)
            mod.an_chase(nxt, int(pos[0]), hops, out)
            res.append(_AnHost.vals(out)[0] / hops)
        _log(f"AN chase {name}: cold {res[0]:.0f} cyc/hop, hot {res[1]:.0f}, cold again {res[2]:.0f}")
    del nxt
    bout = torch.zeros(528, dtype=torch.int64, device=dev)
    sink = torch.zeros(4, dtype=torch.float32, device=dev)
    iters = 2048
    for kind, label, per in ((0, "mma f16 k16", "warp"), (1, "mma e4m3 k32", "warp"), (2, "ex2 f32", "thread"),
                             (3, "ex2 f16x2", "thread"), (4, "cvt e4m3x2->f16x2", "thread")):
        line = []
        for ctas in (528, 132):
            try:
                mod.an_bench(kind, ctas, 128, iters, bout, sink)
                clk = _an_q(_AnHost.vals(bout[:ctas]), 0.5)
                ops = (ctas // 132) * (4 if per == "warp" else 128) * iters * 8
                line.append(f"{ctas // 132} CTA/SM: {ops / clk:.2f}/clk/SM")
            except Exception as exc:
                line.append(f"{ctas}: FAILED {_why(exc)}")
        _log(f"AN bench {label} (per {per} instr): " + ", ".join(line))
    src = torch.zeros(512 * 4096 + 1, dtype=torch.int32, device=dev)
    dst = torch.zeros(1024, dtype=torch.int32, device=dev)
    _log(f"AN null setup {mod.an_null_setup(55296)}")
    cases = [("1 empty", [(0, 528, 128, 0, 0)]), ("2 empty", [(0, 528, 128, 0, 0)] * 2),
             ("3 empty", [(0, 528, 128, 0, 0)] * 3),
             ("3 empty pdl", [(0, 528, 128, 0, 0), (0, 528, 128, 0, 1), (0, 528, 128, 0, 1)]),
             ("1 empty 54KB smem", [(0, 512, 128, 55296, 0)]), ("1 one-load", [(1, 512, 128, 0, 0)]),
             ("2 one-load", [(1, 512, 128, 0, 0)] * 2), ("1 cluster8", [(2, 256, 128, 55296, 0)])]
    for name, launches in cases:
        try:
            def fn(launches=launches):
                for kind, ctas, threads, smem, pdl in launches:
                    mod.an_null(kind, ctas, threads, smem, pdl, src, dst)
            _log(f"AN graph {name}: cold {lab.time_graph(fn):.2f}us hot {_AnHost.time_hot(fn):.2f}us")
        except Exception as exc:
            _log(f"AN graph {name}: FAILED {_why(exc)}")


def _an_case(lab, tag, fn, buf, nrows, phases, accs, counts, ref=None, leader=None):
    buf.zero_()      # no stamp left over from an earlier case
    err = ""
    if ref is not None:
        err = f"err {float((fn().float() - ref).norm() / ref.norm()):.1e}"
    cold = lab.time_graph(fn)
    rows_cold = _AnHost.rows(buf, nrows)
    hot = _AnHost.time_hot(fn)
    rows_hot = _AnHost.rows(buf, nrows)
    _an_report(tag + " cold", rows_cold, phases, accs, counts, leader, f"| {cold:.1f}us {err}")
    _an_report(tag + " hot", rows_hot, phases, accs, counts, leader, f"| {hot:.1f}us")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1515)
    builds = (("micro", "okc_l15_micro", CUDA_AN_MICRO_DECL, CUDA_AN_MICRO_SRC,
               ["an_chase", "an_null", "an_null_setup", "an_bench"]),
              ("d3", "okc_l15_d3", CUDA_AN_DEC4_DECL, CUDA_AN_DEC4_SRC, ["an_dec4", "an_dec4_setup", "an_dec4_attrs"]),
              ("ud2", "okc_l15_ud2", CUDA_AN_UD2_DECL, CUDA_AN_UD2_SRC, ["an_ud2", "an_ud2_setup", "an_ud2_attrs"]),
              ("ver3", "okc_l15_ver3", CUDA_AN_VER3_DECL, CUDA_AN_VER3_SRC, ["an_ver3", "an_ver3_setup", "an_ver3_attrs"]))
    for key, name, decl, src, funcs in builds:
        try:
            _AN[key] = _an_build(name, decl, src, funcs)
            if key == "d3":
                _log(f"d3 setup {_AN[key].an_dec4_setup()} attrs w4s3 {_AN[key].an_dec4_attrs(4, 3)}")
            elif key == "ud2":
                _log(f"ud2 setup {_AN[key].an_ud2_setup()} attrs w4s3cs8 {_AN[key].an_ud2_attrs(4, 3, 8)} "
                     f"cs16 {_AN[key].an_ud2_attrs(4, 3, 16)}")
            elif key == "ver3":
                _log(f"ver3 setup {_AN[key].an_ver3_setup()} attrs s3 {_AN[key].an_ver3_attrs(3)} s4 {_AN[key].an_ver3_attrs(4)}")
        except Exception as exc:
            _AN.pop(key, None)
            _log(f"build {name} FAILED {_why(exc)}")
    if "micro" in _AN:
        try:
            _an_micro(lab)
        except Exception as exc:
            _log(f"AN micro FAILED {_why(exc)}")
    _AN_TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    buf = torch.zeros((2048 * 32 * 32,), dtype=torch.int64, device=dev)
    _log(f"L15 start: budget {_LAB_BUDGET_S:.0f}s; AN line = n CTAs, SMs, max CTAs/SM, span us, start p50/p90/max us, "
         "late starts, CTA dur p50/max us | phase p50/p90 us (thread 0) | clock, globaltimer step | graph us")
    shapes = [("Sd1", 1, 1, 2048), ("Sd16", 16, 1, 2048), ("Sd32", 32, 1, 2048), ("Fd1", 1, 1, 0), ("Fd16", 16, 1, 0),
              ("Sv16", 16, 8, 2048), ("Fv16", 16, 8, 0), ("Fv1", 1, 8, 0)]
    d3_cfg = {"Sd1": [(4, 4, 3)], "Sd16": [(4, 4, 3)], "Sd32": [(4, 4, 3), (2, 4, 3)], "Fd1": [(4, 4, 3), (2, 4, 3)],
              "Fd16": [(6, 4, 3)]}
    ud2_cfg = {"Sd1": [(8, 0, 4, 3)], "Sd16": [(8, 0, 4, 3)], "Sd32": [(8, 0, 4, 3)], "Fd1": [(16, 2, 4, 3)]}
    ver3_cfg = {"Sv16": [(4, 4)], "Fv16": [(4, 4)], "Fv1": [(1, 3)]}
    for name, batch, length, window in shapes:
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, length, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            cfg, max_pages = data["config"], data["page_table"].shape[1]
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        if name in ("Sd1", "Sd32", "Fd1", "Sv16"):
            try:
                p, a, c = _an_triton_parts(data)
                p(), a(), c()
                line = []
                for label, parts in (("P", (p,)), ("A", (a,)), ("C", (c,)), ("PA", (p, a)), ("PAC", (p, a, c))):
                    def fn(parts=parts):
                        for f in parts:
                            f()
                    line.append(f"{label} {lab.time_graph(fn):.1f}/{_AnHost.time_hot(fn):.1f}")
                _log(f"AN {name} triton cold/hot us: " + " ".join(line))
            except Exception as exc:
                _log(f"AN {name} triton FAILED {_why(exc)}")
        for cps, warps, stages in d3_cfg.get(name, []) if "d3" in _AN else ():
            try:
                ns = _an_nsplit(cfg, max_pages, cps, 32 * warps)
                _an_case(lab, f"D3 {name} c{cps}w{warps}s{stages} n{ns}", lambda: _an_d3(data, cps, warps, stages, buf),
                         buf, batch * ns, D3_PHASES, [("wait", 6)], [("chunks", 8)], ref)
            except Exception as exc:
                _log(f"AN D3 {name}: FAILED {_why(exc)}")
        for cs, cps, warps, stages in ud2_cfg.get(name, []) if "ud2" in _AN else ():
            try:
                ns = cs * _an_ud2_split(cfg, max_pages, cs, cps, warps)
                _an_case(lab, f"UD2 {name} cs{cs}c{cps}w{warps}s{stages} n{ns}",
                         lambda: _an_ud2(data, cs, cps, warps, stages, buf), buf, batch * ns, UD2_PHASES,
                         [("wait", 8)], [("chunks", 13)], ref, leader=lambda i, ns=ns, cs=cs: (i % ns) % cs == 0)
                _log(f"AN UD2 {name} tickets after: {int(_AN_TICKETS[0][:batch].abs().sum())}")
            except Exception as exc:
                _log(f"AN UD2 {name}: FAILED {_why(exc)}")
        for cps, stages in ver3_cfg.get(name, []) if "ver3" in _AN else ():
            try:
                ns = _an_nsplit(cfg, max_pages, cps, 64)
                _an_case(lab, f"S2c {name} c{cps}s{stages} n{ns}", lambda: _an_ver3(data, cps, stages, buf), buf,
                         batch * ns, VER3_PHASES, [("wfull", 5), ("pwempty", 21), ("pwpt", 22)], [("stages", 6)], ref)
            except Exception as exc:
                _log(f"AN S2c {name}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
