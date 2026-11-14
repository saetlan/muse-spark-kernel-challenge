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
