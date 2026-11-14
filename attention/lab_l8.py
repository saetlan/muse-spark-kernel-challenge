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


# ===== appended: attention/local/cuda_fused.py =====
"""S1b: one CUDA kernel per decode layer (q_len 1): prologue + attention + merge + norm-gate.
Builds on A5b (cuda_dec2.py: whole-row loads, stacked heads, rows two chunks ahead) and adds:
- A23 prologue fold: each CTA computes its 16 query heads (RMSNorm x qk factor, interleaved RoPE
  on sliding layers) from qkv in registers. A lane's query values are bytes 32*t4 .. 32*t4+31 of
  the pair = head hl = t4 >> 1, d = 32*(t4 & 1) + 0..31: RoPE pairs (2i, 2i+1) stay inside the
  lane, the RMS sum needs one shuffle with lane t4 ^ 1. Queries are rounded to FP16 (finer than
  the BF16/FP8 the reference and SGLang use).
- The new token: the CTA whose slice holds position n-1 computes the pair's new key (RMSNorm,
  RoPE on sliding layers, / k_scale) and value (/ v_scale), rounds them to E4M3 (satfinite, as the
  Triton prologue does), stores them at cache row cache_loc[b], and patches them into its own
  fragments for token n-1 (from shared memory). No other CTA reads row n-1, so nothing waits.
- A25 merge: every CTA writes its partial (log2 max, sum, output); the last CTA of each (request,
  pair), by an atomic ticket, merges all slices, applies RMSNorm x sigmoid(gate), writes the BF16
  output and resets the ticket (graph-safe: tickets are back to 0 after every call).
RoPE frequencies come in as a tensor computed by torch exactly as reference.py does; angles use
full-range sincosf.
"""

CUDA_FUSED_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cmath>
#include <vector>

namespace okc_fused {

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
// Two FP32 -> E4M3 (satfinite, round to nearest even): lo -> low byte, hi -> high byte of 16 bits.
__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {
  uint16_t d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}
__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}
__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }
__device__ __forceinline__ uint16_t f2bf16(float x) {
  uint16_t d;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));
  return d;
}
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
__device__ __forceinline__ void set_word(uint4& v, int i, uint32_t x) {
  if (i == 0) v.x = x; else if (i == 1) v.y = x; else if (i == 2) v.z = x; else v.w = x;
}

struct Chunk {
  uint4 k[4];
  uint4 v[4];
};

template <int WARPS>
__global__ void __launch_bounds__(WARPS * 32) fused_dec_kernel(
    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,
    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,
    const int* __restrict__ seqlens, float* __restrict__ o_part, float* __restrict__ m_part,
    float* __restrict__ l_part, int* __restrict__ tickets, uint16_t* __restrict__ out,
    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {
  __shared__ float sm_m[WARPS][2][8];
  __shared__ float sm_l[WARPS][2][8];
  __shared__ float sm_o[WARPS][2][8][64];
  __shared__ __align__(16) uint8_t new_k[128];
  __shared__ __align__(16) uint8_t new_v[128];
  __shared__ int last_flag;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = lane >> 2, t4 = lane & 3, hl = t4 >> 1;
  const int pairs = hkv >> 1;
  const int b = blockIdx.y / pairs;
  const int pair = blockIdx.y % pairs;
  const int kv0 = pair * 2;
  const int split = blockIdx.x;
  const long row_bytes = static_cast<long>(hkv) * 64;
  const long poff = static_cast<long>(kv0) * 64;
  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;
  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;
  const float pos = static_cast<float>(positions[b]);

  const int n = seqlens[b];
  const int lo = window > 0 ? max(n - window, 0) : 0;
  int per = (n - lo + nsplit - 1) / nsplit;
  per = (per + CHUNK - 1) / CHUNK * CHUNK;
  const int start = lo + split * per;
  const int end = min(start + per, n);
  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;
  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA's slice holds the new token
  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;

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
        float sn, cs;
        sincosf(pos * inv_freq[d0 / 2 + p], &sn, &cs);
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
        float sn, cs;
        sincosf(pos * inv_freq[16 * (t4 & 1) + p], &sn, &cs);
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
  // Token n-1 of this chunk takes the new key/value from shared memory (the cache row may not be
  // written yet as far as this CTA's loads are concerned).
  auto patch = [&](int s0, Chunk& ch) {
    if (!has_new || n - 1 < s0 || n - 1 >= s0 + CHUNK) return;
    const int tn = n - 1 - s0;
#pragma unroll
    for (int nt = 0; nt < 2; ++nt)
      if (nt * 8 + g == tn) {
        ch.k[nt * 2] = *reinterpret_cast<const uint4*>(new_k + 32 * t4);
        ch.k[nt * 2 + 1] = *reinterpret_cast<const uint4*>(new_k + 32 * t4 + 16);
      }
    const int slot[4] = {2 * t4, 2 * t4 + 1, 2 * t4 + 8, 2 * t4 + 9};
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (slot[i] == tn) ch.v[i] = *reinterpret_cast<const uint4*>(new_v + 16 * g);
  };

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
    patch(s0, cur);

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
    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);
    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);
    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);
    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);
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

  // ---- CTA merge of the warps (shared memory), then this slice's partial.
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
    for (int jj = 0; jj < 2; ++jj) sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;
  __syncthreads();
  const long part0 = (static_cast<long>(b) * hq + kv0 * 8) * nsplit;   // head kv0*8 + 8h + row, slice split
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
    const long part = part0 + static_cast<long>(h * 8 + row) * nsplit + split;
    o_part[part * 64 + d] = o;
    if (d == 0) {
      m_part[part] = mm;
      l_part[part] = l;
    }
  }

  // ---- A25: the last slice of this (request, pair) merges every slice and writes the output.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int ticket = atomicAdd(tickets + blockIdx.y, 1);
    last_flag = ticket == nsplit - 1;
  }
  __syncthreads();
  if (!last_flag) return;
  __threadfence();
  const float vsc = *v_scale_p;
  for (int hr = warp; hr < 16; hr += WARPS) {                 // one warp per (head, row), 2 dims per lane
    const long pbase = part0 + static_cast<long>(hr) * nsplit;
    float mm = -1.0e30f;
    for (int s = lane; s < nsplit; s += 32) mm = fmaxf(mm, __ldcg(m_part + pbase + s));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));
    float l = 0.f, o0 = 0.f, o1 = 0.f;
#pragma unroll 4
    for (int s = 0; s < nsplit; ++s) {
      const float wt = ex2(__ldcg(m_part + pbase + s) - mm);
      l += wt * __ldcg(l_part + pbase + s);
      o0 += wt * __ldcg(o_part + (pbase + s) * 64 + lane);
      o1 += wt * __ldcg(o_part + (pbase + s) * 64 + 32 + lane);
    }
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
  }
  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;              // ready for the next call (and replay)
}

}  // namespace okc_fused

void fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq,
               torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale,
               torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,
               torch::Tensor l_part, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq,
               int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope,
               double qk_factor, double sm_scale, int64_t warps) {
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));
#define OKC_FUSED_ARGS                                                                                        \
  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \
      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \
      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \
      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(),                     \
      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \
      tickets.data_ptr<int>(), reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq),           \
      static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), \
      static_cast<int>(window), static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)
  if (warps == 8) okc_fused::fused_dec_kernel<8><<<grid, 256, 0, stream>>>(OKC_FUSED_ARGS);
  else okc_fused::fused_dec_kernel<4><<<grid, 128, 0, stream>>>(OKC_FUSED_ARGS);
#undef OKC_FUSED_ARGS
}

std::vector<int64_t> fused_dec_attrs(int64_t warps) {
  cudaFuncAttributes a;
  if (warps == 8) cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<8>);
  else cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<4>);
  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};
}
"""

CUDA_FUSED_DECL = (
    "void fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, "
    "torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, "
    "torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, torch::Tensor tickets, "
    "torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, "
    "int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps);\n"
    "std::vector<int64_t> fused_dec_attrs(int64_t warps);")


# ===== appended: attention/local/lab_l8_block.py =====


# ---------------------------------------------------------------------------------------------
# Lab L8 (check runs only): A5b and the S1b fused kernel after the local-memory fix (the
# accumulator was indexed with a lane-dependent value at the merge, so it lived in local memory:
# L5 reported 256 B of local memory for A5b). Both on the six decode shapes against the
# submission's path, error vs the FP32 reference, graph timing with a cold L2. Any kernel that
# still reports local memory is flagged.
# ---------------------------------------------------------------------------------------------
import tempfile as _tempfile

_LAB_BUDGET_S = 330.0
_MODS = {}
_TICKETS = []


def _build(name, decl, src, funcs):
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    mod = load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                      extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                      build_directory=_tempfile.mkdtemp(prefix=name + "_"), verbose=False)
    _log(f"build {name} {_time.monotonic() - t:.1f}s")
    return mod


def _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _dec2_run(data, ctas_per_sm, warps):
    """Triton prologue, A5b attention, Triton combine (q_len 1, groups of 8)."""
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    qkv = data["qkv"]
    device = qkv.device
    max_pages = data["page_table"].shape[1]
    nsplit = _dec2_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    q = torch.empty((batch, hq, 64), dtype=torch.bfloat16, device=device)
    _prologue[(batch,)](qkv, data["positions"], data["cache_loc"], _inv_freq(cfg["rope_theta"] if rope else 1.0, device),
                        q, data["k_cache"], data["v_cache"], data["k_scale"], data["v_scale"],
                        HQ=hq, HKV=hkv, ROPE=rope, QK_FACTOR=cfg["qk_scale_factor"], EPS=RMS_EPS, num_warps=4)
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    _MODS["main"].attend_dec2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                              data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                              cfg["page_size"], cfg["window"], cfg["sm_scale"], warps)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(batch, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                          HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=min(64, triton.next_power_of_2(nsplit)), num_warps=4)
    return out


def _fused_nsplit(cfg, max_pages, ctas_per_sm, warps):
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + 1)
    programs = cfg["batch"] * (cfg["num_kv_heads"] // 2)
    return max(1, min(math.ceil(ctas_per_sm * NUM_SMS / programs), tokens // (32 * warps), MAX_SPLITS))


def _fused_run(data, ctas_per_sm, warps):
    cfg = data["config"]
    batch, hq, hkv = cfg["batch"], cfg["num_q_heads"], cfg["num_kv_heads"]
    assert cfg["q_len"] == 1 and hq // hkv == 8 and hkv % 2 == 0
    device = data["qkv"].device
    if not _TICKETS:
        _TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=device))
    max_pages = data["page_table"].shape[1]
    nsplit = _fused_nsplit(cfg, max_pages, ctas_per_sm, warps)
    rope = bool(cfg["rope_theta"])
    o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
    m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
    l_part = torch.empty_like(m_part)
    out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
    _MODS["fused"].fused_dec(data["qkv"], data["positions"], data["cache_loc"],
                             _inv_freq(cfg["rope_theta"] if rope else 1.0, device), data["k_cache"],
                             data["v_cache"], data["k_scale"], data["v_scale"], data["page_table"],
                             data["cache_seqlens"], o_part, m_part, l_part, _TICKETS[0], out, batch, hq, hkv,
                             max_pages, nsplit, cfg["page_size"], cfg["window"], int(rope),
                             cfg["qk_scale_factor"], cfg["sm_scale"], warps)
    return out



def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(8008)
    try:
        _MODS["main"] = _build("okc_l8_dec2", CUDA_DEC2_DECL, CUDA_DEC2_SRC, ["attend_dec2", "attend_dec2_attrs"])
        _MODS["fused"] = _build("okc_l8_fused", CUDA_FUSED_DECL, CUDA_FUSED_SRC, ["fused_dec", "fused_dec_attrs"])
        for name, attrs in (("A5b", _MODS["main"].attend_dec2_attrs), ("fused", _MODS["fused"].fused_dec_attrs)):
            for w in (4, 8):
                a = attrs(w)
                _log(f"{name} w{w} attrs (regs, local B, smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _log(f"build FAILED {_why(exc)}")
        return
    rows = 32 * scored_max_pages(32, 1)
    k_cache, v_cache = _lab_soft_cache(gen, rows)
    _log(f"L8 start: budget {_LAB_BUDGET_S:.0f}s")
    variants = [(2, 4), (4, 4), (8, 4), (4, 8), (8, 8)]
    for batch, window in [(16, 0), (32, 0), (1, 2048), (16, 2048), (32, 2048), (1, 0)]:
        name = f"{'S' if window else 'F'}d{batch}"
        if lab.left() < 20:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, 1, window, k_cache, v_cache, gen)
            run(data)
            ref = _lab_reference(data)
            kv = float((data["cache_seqlens"].clamp(max=window + 1) if window else data["cache_seqlens"]).sum()) * 256
            err = float((run(data).float() - ref).norm() / ref.norm())
            us = lab.time_graph(lambda: run(data))
            _log(f"{name} v2.2 path: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        for label, fn in (("A5b", _dec2_run), ("fused", _fused_run)):
            for cps, warps in variants:
                try:
                    out = fn(data, cps, warps).float()
                    err = float((out - ref).norm() / ref.norm())
                    us = lab.time_graph(lambda: fn(data, cps, warps))
                    _log(f"{name} {label} c{cps} w{warps}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                except Exception as exc:
                    _log(f"{name} {label} c{cps} w{warps}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
