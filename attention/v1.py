# okc: problem=muse_attention_decode
"""v1 (A10): gated attention decode in three Triton kernels.

1. _prologue: per token, scaleless RMSNorm of q (times qk_scale_factor) and the new keys,
   interleaved RoPE on sliding layers, q kept in BF16 in a scratch buffer, the new keys and
   values cast to FP8 E4M3 (divided by the cache scales) and stored at their cache rows.
2. _attend: split-context flash decode. One program covers KVP key/value heads of one
   request and one slice of its context, so with page size 1 it reads each token's whole
   KVP*64-byte cache row. The queries go in block-diagonal: row (l, kvp, g) holds query head
   (kv0+kvp)*G+g of block token l in columns kvp*64 .. kvp*64+63 and zeros elsewhere, so one
   dot against the gathered [N, KVP*64] keys gives every head its own scores, and P @ V puts
   each head's output in its own column block. FP8 keys and values are widened to BF16.
3. _combine: merges the slices (log-sum-exp), then RMSNorm x sigmoid(gate), BF16 out.

Scores and the softmax run in FP32 in the log2 domain; queries stay BF16 (finer than FA3's
FP8). Every key in the window (sliding) or the context (full) is attended."""

import math
from typing import TypedDict

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


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
BLOCK_N = 64          # keys per inner step
COMBINE_BLOCK = 16    # slices merged per step in _combine
_INV_FREQ = {}
_WARM = []            # non-empty once every scored variant is compiled and loaded


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


@triton.jit(do_not_specialize=["max_pages", "nsplit"])
def _attend(q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, page_table_ptr, seqlens_ptr,
            o_part_ptr, m_part_ptr, l_part_ptr, max_pages, nsplit,
            HQ: tl.constexpr, HKV: tl.constexpr, KVP: tl.constexpr, L: tl.constexpr,
            PAGE: tl.constexpr, WINDOW: tl.constexpr, SM_SCALE: tl.constexpr,
            BLOCK: tl.constexpr):
    b = tl.program_id(0)
    kv0 = tl.program_id(1) * KVP
    split = tl.program_id(2)
    G: tl.constexpr = HQ // HKV
    M: tl.constexpr = L * KVP * G
    W: tl.constexpr = KVP * 64

    r = tl.arange(0, M)
    r_l = r // (KVP * G)            # block token of each query row
    r_kvp = (r // G) % KVP          # which of the program's KV heads
    r_head = (kv0 + r_kvp) * G + r % G
    c = tl.arange(0, W)
    c_kvp = c // 64
    c_d = c % 64
    diag = r_kvp[:, None] == c_kvp[None, :]

    t0 = b * L
    q_off = ((t0 + r_l).to(tl.int64) * HQ + r_head)[:, None] * 64 + c_d[None, :]
    q = tl.load(q_ptr + q_off, mask=diag, other=0.0)  # [M, W] bf16, block-diagonal

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
    m_i = tl.full([M], -1.0e30, tl.float32)
    l_i = tl.zeros([M], tl.float32)
    acc = tl.zeros([M, W], tl.float32)
    nb = tl.arange(0, BLOCK)
    pt_row = page_table_ptr + b.to(tl.int64) * max_pages
    for s0 in range(start, end, BLOCK):
        s = s0 + nb
        live = s < end
        page = tl.load(pt_row + s // PAGE, mask=live, other=0)
        kv_row = page.to(tl.int64) * PAGE + s % PAGE
        off = kv_row[:, None] * (HKV * 64) + kv0 * 64 + c[None, :]
        k = tl.load(k_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        v = tl.load(v_cache_ptr + off, mask=live[:, None], other=0.0).to(tl.bfloat16)
        sc = tl.dot(q, tl.trans(k)) * qk_scale       # [M, BLOCK], log2 domain
        vis = live[None, :] & (s[None, :] <= row_pos[:, None])
        if WINDOW > 0:
            vis = vis & (s[None, :] > row_pos[:, None] - WINDOW)
        sc = tl.where(vis, sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(sc - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    # Partials per (token, head, split): only each row's own column block is meaningful.
    th = (t0 + r_l).to(tl.int64) * HQ + r_head
    part = th * nsplit + split
    tl.store(o_part_ptr + part[:, None] * 64 + c_d[None, :], acc, mask=diag)
    tl.store(m_part_ptr + part, m_i)
    tl.store(l_part_ptr + part, l_i)


@triton.jit(do_not_specialize=["nsplit"])
def _combine(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit,
             HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    m_max = tl.full([CB], -1.0e30, tl.float32)
    for s0 in range(0, nsplit, CB):
        m_max = tl.maximum(m_max, tl.load(m_part_ptr + th * nsplit + s0 + j, mask=s0 + j < nsplit,
                                          other=-1.0e30))
    big = tl.max(m_max, 0)
    den = tl.zeros([CB], tl.float32)
    num = tl.zeros([CB, 64], tl.float32)
    for s0 in range(0, nsplit, CB):
        ok = s0 + j < nsplit
        idx = th * nsplit + s0 + j
        w = tl.exp2(tl.load(m_part_ptr + idx, mask=ok, other=-1.0e30) - big)
        den += w * tl.load(l_part_ptr + idx, mask=ok, other=0.0)
        num += w[:, None] * tl.load(o_part_ptr + idx[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.0)
    a = tl.sum(num, 0) / tl.sum(den, 0) * tl.load(v_scale_ptr)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / 64.0 + EPS)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    g = tl.load(qkv_ptr + t.to(tl.int64) * ROW + (HQ + 2 * HKV) * 64 + h * 64 + d).to(tl.float32)
    a = a / (1.0 + tl.exp(-g))
    tl.store(out_ptr + t.to(tl.int64) * (HQ * 64) + h * 64 + d, a.to(tl.bfloat16))


def plan(cfg, max_pages):
    """(KV heads per program, slices per request) from the config and shapes only."""
    hkv = cfg["num_kv_heads"]
    kvp = 2 if hkv % 2 == 0 else 1
    tokens = max_pages * cfg["page_size"]
    if cfg["window"]:
        tokens = min(tokens, cfg["window"] + cfg["q_len"])
    programs = cfg["batch"] * (hkv // kvp)
    nsplit = max(1, min(math.ceil(2 * NUM_SMS / programs), tokens // (2 * BLOCK_N)))
    return kvp, nsplit


def _inv_freq(theta, device):
    key = (theta, str(device))
    if key not in _INV_FREQ:
        # As SGLang's RotaryEmbedding and the reference build it, in FP32 on the GPU.
        _INV_FREQ[key] = 1.0 / (theta ** (torch.arange(0, 64, 2, dtype=torch.float, device=device) / 64))
    return _INV_FREQ[key]


def kernel_fn(data):
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    batch, length = cfg["batch"], cfg["q_len"]
    hq, hkv = cfg["num_q_heads"], cfg["num_kv_heads"]
    tokens = batch * length
    qkv = data["qkv"]
    device = qkv.device
    page_table = data["page_table"]
    max_pages = page_table.shape[1]
    kvp, nsplit = plan(cfg, max_pages)
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
    m_rows = length * kvp * (hq // hkv)
    _attend[(batch, hkv // kvp, nsplit)](
        q, data["k_cache"], data["v_cache"], data["k_scale"], page_table, data["cache_seqlens"],
        o_part, m_part, l_part, max_pages, nsplit,
        HQ=hq, HKV=hkv, KVP=kvp, L=length, PAGE=cfg["page_size"], WINDOW=cfg["window"],
        SM_SCALE=cfg["sm_scale"], BLOCK=BLOCK_N,
        num_warps=8 if m_rows >= 128 else 4, num_stages=3)

    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=device)
    _combine[(tokens, hq)](o_part, m_part, l_part, qkv, data["v_scale"], out, nsplit,
                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=COMBINE_BLOCK, num_warps=2)
    return out


def _warm_up():
    """Compile and load every scored variant on the first eager call (a correctness test),
    so no first launch (Triton compile + module load) can land inside a CUDA graph capture.
    Not at import: CUDA stays uninitialised until the harness first calls us. Tiny dummy
    inputs; the variants depend only on the config, never on the lengths."""
    _WARM.append(True)
    dev = torch.device("cuda")
    for length in (1, 8):
        for window, theta in ((2048, 500000.0), (0, 0.0)):
            for batch in (1, 16, 32):
                hq, hkv, page, ctx = 16, 2, 1, 300
                pages = batch * (ctx + length)
                cfg = dict(batch=batch, q_len=length, num_q_heads=hq, num_kv_heads=hkv, head_dim=64,
                           page_size=page, window=window, rope_theta=theta,
                           qk_scale_factor=5.4730064863875, sm_scale=0.125, kv_scale=1.0)
                seq = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
                table = torch.arange(pages, dtype=torch.int32, device=dev).view(batch, ctx + length)
                pos = (seq.long()[:, None] + torch.arange(length, device=dev)).reshape(-1)
                data = dict(
                    qkv=torch.randn((batch, length, (2 * hq + 2 * hkv) * 64), device=dev).to(torch.bfloat16),
                    positions=pos,
                    cache_loc=torch.gather(table.long(), 1, pos.view(batch, length)).reshape(-1),
                    cache_seqlens=seq + length,
                    cu_seqlens_q=torch.arange(0, batch * length + 1, length, dtype=torch.int32, device=dev),
                    scheduler_metadata=None,
                    k_cache=torch.zeros((pages, page, hkv, 64), device=dev).to(torch.float8_e4m3fn),
                    v_cache=torch.zeros((pages, page, hkv, 64), device=dev).to(torch.float8_e4m3fn),
                    k_scale=torch.tensor(1.0, device=dev), v_scale=torch.tensor(1.0, device=dev),
                    page_table=table, seq_lens=seq, config=cfg)
                kernel_fn(data)  # compiles and loads on the host at launch; no sync needed
