# okc: problem=muse_attention_decode
"""v2: gated attention decode in three Triton kernels.

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
# From lab L1 (check b8c59d13124e40b6): best per scored shape under graphs + cold L2.
TUNE = {
    (True, False, 1): (32, 256, 64, 4, 3),
    (True, False, 16): (16, 256, 64, 4, 3),
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


def kernel_fn(data):
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    return run(data)


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
                run(dummy_inputs(batch, length, window, scored_max_pages(batch, length), dev))


# ---------------------------------------------------------------------------------------------
# Lab L2 (check runs only; generated by attention/local/make_lab.py, so the code above is the
# submission byte for byte). Runs once, on the first eager kernel_fn call, after the warm-up.
# Question: what limits reading scattered 128-byte KV rows (page size 1)? v2 and FA3 both
# stall at ~1.6 TB/s on the full-context shapes. Sections:
#   G1 Triton gather-sum over the pool's rows (K and V), by row order and launch settings
#   G2 CUDA gather-sum (load_inline, sm_90a): each thread keeps U 16-byte loads per array in
#      flight, 8 threads per 128-byte row, to find the H200's ceiling for this access pattern
#   V  v2 attention with deeper pipelines on the full-context shapes
# Every section is time-boxed and wrapped; nothing here runs inside a capture.
# ---------------------------------------------------------------------------------------------
import os as _os
import statistics as _st
import tempfile as _tempfile
import time as _time

_LAB = []
_LAB_BUDGET_S = 300.0
_kernel_fn_submitted = kernel_fn


def _log(msg):
    print(f"[lab] {msg}", flush=True)


def _why(exc):
    return f"{type(exc).__name__}: {' '.join(str(exc).split())[-400:]}"


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

    def section(self, name, fn, *args):
        if self.left() < 10:
            _log(f"{name}: skipped, budget spent")
            return
        t = _time.monotonic()
        try:
            fn(self, *args)
        except Exception as exc:
            _log(f"{name}: FAILED {_why(exc)}")
        torch.cuda.synchronize()
        _log(f"{name}: {_time.monotonic() - t:.1f}s")


@triton.jit
def _lab_gsum(k_ptr, v_ptr, idx_ptr, n_per, out_ptr, BLOCK: tl.constexpr):
    """Sum of the K and V rows listed in idx[pid*n_per : (pid+1)*n_per] (128 bytes each)."""
    pid = tl.program_id(0)
    start = pid * n_per
    c = tl.arange(0, 128)
    acc = tl.zeros([128], tl.float32)
    for s0 in range(start, start + n_per, BLOCK):
        rows = tl.load(idx_ptr + s0 + tl.arange(0, BLOCK))
        off = rows.to(tl.int64)[:, None] * 128 + c[None, :]
        k = tl.load(k_ptr + off).to(tl.float32)
        v = tl.load(v_ptr + off).to(tl.float32)
        acc += tl.sum(k + v, 0)
    tl.store(out_ptr + pid * 128 + c, acc)


_CUDA_GATHER = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

template <int U>
__global__ void gsum_kernel(const uint4* __restrict__ k, const uint4* __restrict__ v,
                            const int* __restrict__ idx, long n, unsigned* __restrict__ out) {
  const int lane = threadIdx.x & 31;
  const int chunk = lane & 7;       // 16-byte chunk of the 128-byte row
  const int sub = lane >> 3;        // which of the warp's 4 rows
  const long warp = (blockIdx.x * (long)blockDim.x + threadIdx.x) >> 5;
  const long nwarps = (gridDim.x * (long)blockDim.x) >> 5;
  unsigned acc = 0;
  for (long base = warp * 4 * U; base < n; base += nwarps * 4 * U) {
    int row[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      long r = base + u * 4 + sub;
      row[u] = r < n ? __ldg(idx + r) : 0;
    }
    uint4 a[U], b[U];
#pragma unroll
    for (int u = 0; u < U; ++u) {
      a[u] = __ldg(k + (long)row[u] * 8 + chunk);
      b[u] = __ldg(v + (long)row[u] * 8 + chunk);
    }
#pragma unroll
    for (int u = 0; u < U; ++u) acc ^= a[u].x ^ a[u].y ^ a[u].z ^ a[u].w ^ b[u].x ^ b[u].y ^ b[u].z ^ b[u].w;
  }
  if (acc == 0x9E3779B9u) out[0] = acc;   // keeps the loads alive; practically never true
}

void gsum(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, int64_t blocks,
          int64_t threads, int64_t unroll) {
  auto stream = at::cuda::getCurrentCUDAStream();
  const uint4* kp = reinterpret_cast<const uint4*>(k.data_ptr());
  const uint4* vp = reinterpret_cast<const uint4*>(v.data_ptr());
  const int* ip = idx.data_ptr<int>();
  unsigned* op = reinterpret_cast<unsigned*>(out.data_ptr());
  long n = idx.numel();
  if (unroll == 4) gsum_kernel<4><<<blocks, threads, 0, stream>>>(kp, vp, ip, n, op);
  else if (unroll == 8) gsum_kernel<8><<<blocks, threads, 0, stream>>>(kp, vp, ip, n, op);
  else gsum_kernel<16><<<blocks, threads, 0, stream>>>(kp, vp, ip, n, op);
}
"""


_CUDA_GATHER_DECL = ("void gsum(torch::Tensor k, torch::Tensor v, torch::Tensor idx, torch::Tensor out, "
                     "int64_t blocks, int64_t threads, int64_t unroll);")


def _orders(rows, gen, dev):
    """Row lists over the pool: sequential, random (as page 1 scatters them), and the same
    random rows sorted (what a physical-order traversal would read)."""
    perm = torch.randperm(rows, generator=gen, device=dev).to(torch.int32)
    half = perm[: rows // 2]
    return {
        "seq": torch.arange(rows, dtype=torch.int32, device=dev),
        "rand": perm,
        "rand50": half,
        "sorted50": torch.sort(half).values,
    }


def _g1(lab, k, v, orders):
    """Triton gather-sum: BLOCK rows per step, stages, warps, programs."""
    out = torch.empty((4096 * 128,), dtype=torch.float32, device=k.device)
    for name, idx in orders.items():
        for block, stages, warps, progs in ((64, 2, 4, 1056), (64, 3, 4, 1056), (64, 4, 4, 2112),
                                            (128, 3, 4, 1056), (128, 3, 8, 1056), (256, 2, 8, 528),
                                            (32, 4, 4, 4096)):
            if lab.left() < 10:
                return
            n_per = (idx.numel() // progs) // block * block
            used = n_per * progs
            try:
                us = lab.time_graph(lambda: _lab_gsum[(progs,)](k, v, idx, n_per, out, BLOCK=block,
                                                                num_warps=warps, num_stages=stages))
                _log(f"G1 {name} b{block} s{stages} w{warps} p{progs}: {us:.1f}us {used * 256 / us / 1e6:.2f}TB/s")
            except Exception as exc:
                _log(f"G1 {name} b{block} s{stages} w{warps}: FAILED {_why(exc)}")
                return


def _g2(lab, k, v, orders):
    """CUDA gather-sum with U loads per array in flight per thread."""
    from torch.utils.cpp_extension import load_inline
    t = _time.monotonic()
    build = _tempfile.mkdtemp(prefix="okc_lab_l2_")
    mod = load_inline(
        name="okc_lab_l2_gather", cpp_sources=_CUDA_GATHER_DECL, cuda_sources=_CUDA_GATHER, functions=["gsum"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], build_directory=build,
        verbose=False)
    _log(f"G2 build {_time.monotonic() - t:.1f}s")
    out = torch.zeros((4,), dtype=torch.int32, device=k.device)
    for name, idx in orders.items():
        for unroll, threads, blocks in ((4, 256, 132 * 8), (8, 256, 132 * 8), (16, 256, 132 * 8),
                                        (8, 512, 132 * 4), (16, 128, 132 * 16), (8, 1024, 132 * 2)):
            if lab.left() < 10:
                return
            us = lab.time_graph(lambda: mod.gsum(k, v, idx, out, blocks, threads, unroll))
            _log(f"G2 {name} u{unroll} t{threads} b{blocks}: {us:.1f}us {idx.numel() * 256 / us / 1e6:.2f}TB/s")


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


def _v(lab, k_cache, v_cache, gen):
    """v2 attention with deeper pipelines and bigger steps on the full-context shapes."""
    variants = {
        (1, 1): [None, (32, 256, 64, 4, 4), (32, 256, 128, 4, 3), (32, 128, 64, 4, 4)],
        (16, 1): [None, (16, 256, 64, 4, 4), (16, 256, 64, 4, 5), (16, 256, 128, 4, 3), (16, 256, 128, 4, 4),
                  (32, 256, 64, 4, 4), (8, 256, 128, 8, 3), (16, 256, 32, 4, 6)],
        (32, 1): [None, (16, 256, 64, 4, 4), (16, 256, 128, 4, 3), (8, 256, 128, 4, 4)],
        (16, 8): [None, (4, 256, 64, 4, 3), (8, 256, 64, 4, 3), (4, 256, 32, 4, 4)],
        (32, 8): [None, (8, 256, 64, 4, 3), (8, 256, 32, 4, 4)],
    }
    for (batch, length), tunes in variants.items():
        data = _lab_inputs(batch, length, 0, k_cache, v_cache, gen)
        ref = run(data).float()
        kv_bytes = float(data["cache_seqlens"].sum()) * 256
        for tune in tunes:
            if lab.left() < 10:
                return
            try:
                err = float((run(data, tune).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: run(data, tune))
                ns = plan(data["config"], data["page_table"].shape[1], tune)[0]
                label = "/".join(map(str, tune)) if tune else "TUNE"
                _log(f"V F{'v' if length > 1 else 'd'}{batch} {label} n{ns}: {us:.1f}us "
                     f"{kv_bytes / us / 1e6:.2f}TB/s err {err:.0e}")
            except Exception as exc:
                _log(f"V F{batch}/{length} {tune}: FAILED {_why(exc)}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4321)
    rows = 32 * scored_max_pages(32, 8)
    k_cache = (torch.randn((rows, 1, 2, 64), generator=gen, device=dev) * 0.3).to(torch.float8_e4m3fn)
    v_cache = torch.randn((rows, 1, 2, 64), generator=gen, device=dev).to(torch.float8_e4m3fn)
    _log(f"L2 start: budget {_LAB_BUDGET_S:.0f}s, pool {rows} rows x 128 B per array "
         f"({rows * 256 / 1e6:.0f} MB K+V)")
    orders = _orders(rows, gen, dev)
    lab.section("G1", _g1, k_cache, v_cache, orders)
    lab.section("V", _v, k_cache, v_cache, gen)
    lab.section("G2", _g2, k_cache, v_cache, orders)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache, orders
    torch.cuda.empty_cache()


def kernel_fn(data):  # noqa: F811 (lab wrapper; the timed path is the submission's kernel_fn)
    if not _LAB and not torch.cuda.is_current_stream_capturing():
        _LAB.append(True)
        _kernel_fn_submitted(data)   # warm-up first, exactly as the submission does
        try:
            _lab_main()
        except Exception as exc:
            _log(f"FAILED {_why(exc)}")
    return _kernel_fn_submitted(data)
