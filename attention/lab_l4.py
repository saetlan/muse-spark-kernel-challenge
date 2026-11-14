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


# ===== appended: attention/local/cuda_dec.py =====
"""A5: CUDA split-context decode attention (q_len 1), source + host launcher. Kept in one place
so the lab blocks and, later, the submission include the same text (make_lab.py appends it).

Layout trick: the dot products are invariant to permuting the head dimension d consistently, so
each lane's mma.sync fragment values are made contiguous in memory:
- QK (m16n8k16 f16, A = queries [16 rows = 8 heads of the group + 8 zero rows, 16 d], B = keys):
  logical k column 2*t4 + {0,1} (+8) of k-step ks  <->  physical byte 16*t4 + 4*ks + {0,1} ({2,3})
  so lane (g, t4) loads 16 contiguous bytes of key row g (one 16 B load per head per 8 tokens).
- PV (A = P [rows, 16 tokens] from the QK accumulators, B = values [16 tokens, 8 d]):
  logical n column g of d-tile j  <->  physical byte 8*g + j
  so lane (g, t4) loads 8 contiguous bytes of value rows 2*t4, 2*t4+1, 2*t4+8, 2*t4+9.
FP8 E4M3 -> FP16 is exact (cvt.rn.f16x2.e4m3x2, sm_89+); queries are rounded to FP16 (finer than
the FP8 the rules allow); P is FP16; accumulation FP32. Each CTA: one request, a pair of KV heads
(whole 128 B cache rows), one slice of keys; its 4 warps take 16-token chunks round-robin, keep
their own online softmax, prefetch their next chunk into registers, and merge through shared
memory at the end. Partials (log2-domain max, sum, output) match _combine's layout.
"""

CUDA_DEC_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_dec {

constexpr int CHUNK = 16;
constexpr unsigned FULL = 0xffffffffu;

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2_lo(uint32_t w) {
  // Low 16 bits of w hold two E4M3 bytes; the low byte becomes the low half.
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, l;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t f16x2_from_e4m3x2_hi(uint32_t w) {
  uint32_t d;
  asm("{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, %1;\n\tcvt.rn.f16x2.e4m3x2 %0, h;\n\t}" : "=r"(d) : "r"(w));
  return d;
}

__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {
  uint32_t d;
  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));   // a -> upper half, b -> lower
  return d;
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16_bits_to_f32(uint16_t v) {
  return __uint_as_float(static_cast<uint32_t>(v) << 16);
}

__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                         uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) attend_dec_kernel(
    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,
    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3;
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int kv0 = (blockIdx.y % pairs) * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;

  const int n = seqlens[b];                          // keys 0 .. n-1; the query sits at n-1
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

  // Query A fragments (rows g; rows g+8 are zero), d permuted as in the module docstring.
  uint32_t qa[2][4][2];
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint16_t* qp = q + (static_cast<long>(b) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      qa[h][ks][0] = pack_f16x2(bf16_bits_to_f32(qp[4 * ks + 0]), bf16_bits_to_f32(qp[4 * ks + 1]));
      qa[h][ks][1] = pack_f16x2(bf16_bits_to_f32(qp[4 * ks + 2]), bf16_bits_to_f32(qp[4 * ks + 3]));
    }
  }

  const int* pt = page_table + static_cast<long>(b) * max_pages;
  float m_run[2] = {-1.0e30f, -1.0e30f};
  float l_run[2] = {0.f, 0.f};
  float acc[2][8][4];
#pragma unroll
  for (int h = 0; h < 2; ++h)
#pragma unroll
    for (int j = 0; j < 8; ++j)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc[h][j][i] = 0.f;

  uint4 kr[2][2], kn[2][2];
  uint2 vr[2][4], vn[2][4];

  // Loads chunk c of this CTA's slice into (k, v). Tokens past `end` read the last valid row
  // (always in bounds) and are masked out of the softmax.
  auto load = [&](int c, uint4 (&k)[2][2], uint2 (&v)[2][4]) {
    const int s = min(start + c * CHUNK + (lane & 15), end - 1);
    const long my_row = static_cast<long>(pt[s / page]) * page + s % page;
    const long rk0 = __shfl_sync(FULL, my_row, g);
    const long rk1 = __shfl_sync(FULL, my_row, 8 + g);
    const long rv0 = __shfl_sync(FULL, my_row, 2 * t4);
    const long rv1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);
    const long rv2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);
    const long rv3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long hoff = (kv0 + h) * 64;
      k[h][0] = __ldg(reinterpret_cast<const uint4*>(kc + rk0 * row_bytes + hoff + 16 * t4));
      k[h][1] = __ldg(reinterpret_cast<const uint4*>(kc + rk1 * row_bytes + hoff + 16 * t4));
      v[h][0] = __ldg(reinterpret_cast<const uint2*>(vc + rv0 * row_bytes + hoff + 8 * g));
      v[h][1] = __ldg(reinterpret_cast<const uint2*>(vc + rv1 * row_bytes + hoff + 8 * g));
      v[h][2] = __ldg(reinterpret_cast<const uint2*>(vc + rv2 * row_bytes + hoff + 8 * g));
      v[h][3] = __ldg(reinterpret_cast<const uint2*>(vc + rv3 * row_bytes + hoff + 8 * g));
    }
  };

  if (warp < nchunks) load(warp, kr, vr);
  for (int c = warp; c < nchunks; c += WARPS) {
    if (c + WARPS < nchunks) load(c + WARPS, kn, vn);
    const int s0 = start + c * CHUNK;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      float sc[2][4];
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) {
        sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        const uint32_t w[4] = {kr[h][nt].x, kr[h][nt].y, kr[h][nt].z, kr[h][nt].w};
#pragma unroll
        for (int ks = 0; ks < 4; ++ks)
          mma16816(sc[nt], qa[h][ks][0], 0u, qa[h][ks][1], 0u, f16x2_from_e4m3x2_lo(w[ks]),
                   f16x2_from_e4m3x2_hi(w[ks]));
      }
      float mx = -INFINITY;
#pragma unroll
      for (int nt = 0; nt < 2; ++nt)
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
          const int tok = s0 + nt * 8 + 2 * t4 + jj;
          const float x = tok < end ? sc[nt][jj] * qk_scale : -INFINITY;
          sc[nt][jj] = x;
          mx = fmaxf(mx, x);
        }
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 1));
      mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, 2));
      const float m_new = fmaxf(m_run[h], mx);
      const float alpha = ex2(m_run[h] - m_new);
      const float p00 = ex2(sc[0][0] - m_new), p01 = ex2(sc[0][1] - m_new);
      const float p10 = ex2(sc[1][0] - m_new), p11 = ex2(sc[1][1] - m_new);
      l_run[h] = l_run[h] * alpha + (p00 + p01 + p10 + p11);
      m_run[h] = m_new;
      const uint32_t pa0 = pack_f16x2(p00, p01);     // row g, tokens 2*t4, 2*t4+1
      const uint32_t pa2 = pack_f16x2(p10, p11);     // row g, tokens 8+2*t4, 9+2*t4
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        acc[h][j][0] *= alpha;
        acc[h][j][1] *= alpha;
        const uint32_t w0 = j < 4 ? vr[h][0].x : vr[h][0].y;
        const uint32_t w1 = j < 4 ? vr[h][1].x : vr[h][1].y;
        const uint32_t w2 = j < 4 ? vr[h][2].x : vr[h][2].y;
        const uint32_t w3 = j < 4 ? vr[h][3].x : vr[h][3].y;
        const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));
        const uint32_t b0 = f16x2_from_e4m3x2_lo(__byte_perm(w0, w1, sel));   // tokens 2*t4, 2*t4+1
        const uint32_t b1 = f16x2_from_e4m3x2_lo(__byte_perm(w2, w3, sel));   // tokens 2*t4+8, 2*t4+9
        mma16816(acc[h][j], pa0, 0u, pa2, 0u, b0, b1);
      }
    }
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
      for (int nt = 0; nt < 2; ++nt) kr[h][nt] = kn[h][nt];
#pragma unroll
      for (int i = 0; i < 4; ++i) vr[h][i] = vn[h][i];
    }
  }

  // Merge the warps: each row's state to shared memory (output d un-permuted), then all threads.
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    float l = l_run[h];
    l += __shfl_xor_sync(FULL, l, 1);
    l += __shfl_xor_sync(FULL, l, 2);
    if (t4 == 0) {
      sm_m[warp][h][g] = m_run[h];
      sm_l[warp][h][g] = l;
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      sm_o[warp][h][g][8 * (2 * t4) + j] = acc[h][j][0];
      sm_o[warp][h][g][8 * (2 * t4 + 1) + j] = acc[h][j][1];
    }
  }
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

}  // namespace okc_dec

void attend_dec(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,
                torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
                torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,
                int64_t page, int64_t window, double sm_scale, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_DEC_ARGS                                                                                         \
  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \
      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \
      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)
  if (warps == 8) okc_dec::attend_dec_kernel<8><<<grid, 256, 0, stream>>>(OKC_DEC_ARGS);
  else okc_dec::attend_dec_kernel<4><<<grid, 128, 0, stream>>>(OKC_DEC_ARGS);
#undef OKC_DEC_ARGS
}

std::vector<int64_t> attend_dec_attrs(int64_t warps) {
  cudaFuncAttributes a;
  if (warps == 8) cudaFuncGetAttributes(&a, okc_dec::attend_dec_kernel<8>);
  else cudaFuncGetAttributes(&a, okc_dec::attend_dec_kernel<4>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_DEC_DECL = ("void attend_dec(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, "
                 "torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, "
                 "torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, "
                 "int64_t page, int64_t window, double sm_scale, int64_t warps);\n"
                 "std::vector<int64_t> attend_dec_attrs(int64_t warps);")


# ===== appended: attention/local/lab_l4_block.py =====


# ---------------------------------------------------------------------------------------------
# Lab L4 (check runs only): A5, the CUDA decode kernel (attention/local/cuda_dec.py, appended
# above), against the submission's Triton path on the six decode shapes at the scored sizes.
# Per variant: error vs the FP32 reference, time of prologue + attend + combine under a CUDA
# graph with a cold L2, KV bandwidth. Build once with load_inline (sm_90a).
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
_DEC = []


def _dec_build():
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name="okc_attn_dec_l4", cpp_sources=CUDA_DEC_DECL, cuda_sources=CUDA_DEC_SRC,
                      functions=["attend_dec", "attend_dec_attrs"],
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix="okc_l4_"), verbose=False)
    _log(f"build {_time.monotonic() - t:.1f}s; attrs (regs, local B, smem B): "
         f"w4 {mod.attend_dec_attrs(4)} w8 {mod.attend_dec_attrs(8)}")
    _DEC.append(mod)


def _dec_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _dec_run(data, ctas_per_sm, warps):
    """run() with the CUDA decode kernel in place of _attend (q_len 1, groups of 8)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _DEC[0].attend_dec(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                       data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                       cfg["page_size"], cfg["window"], cfg["sm_scale"], warps)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4004)
    _dec_build()
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L4 start: budget {_LAB_BUDGET_S:.0f}s; line = shape variant: us TB/s err-vs-FP32 (nsplit)")
    shapes = [(16, 0), (32, 0), (1, 0), (1, 2048), (16, 2048), (32, 2048)]
    variants = [("cuda c2 w4", 2, 4), ("cuda c4 w4", 4, 4), ("cuda c8 w4", 8, 4), ("cuda c16 w4", 16, 4),
                ("cuda c2 w8", 2, 8), ("cuda c4 w8", 4, 8), ("cuda c8 w8", 8, 8)]
    for batch, window in shapes:
        name = f"{'S' if window else 'F'}d{batch}"
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv_bytes = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} triton TUNE: {us:.1f}us {kv_bytes / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for label, cps, warps in variants:
            if lab.left() < 10:
                _log("out of budget")
                break
            try:
                ns = _dec_nsplit(data["config"], data["page_table"].shape[1], cps, warps)
                out = _dec_run(data, cps, warps).float()
                err = float((out - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _dec_run(data, cps, warps))
                _log(f"{name} {label}: {us:.1f}us {kv_bytes / us / 1e6:.2f}TB/s err {err:.1e} (n{ns})")
            except Exception as exc:
                _log(f"{name} {label}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
