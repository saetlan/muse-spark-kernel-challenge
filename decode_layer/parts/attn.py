# Embedded: the attention session's scored v3 (attention/scored/v3_submission.py, sha256 0ae3ecba90ad982c...;
# judged 1.281x on muse_attention_decode, AE12), every top-level name prefixed with 'att_' by
# decode_layer/local/make_attn.py. att_kernel_fn(data) computes our layer's attention segment.

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
"S1b: one CUDA kernel per decode layer (q_len 1): prologue + attention + merge + norm-gate.\nBuilds on A5b (cuda_dec2.py: whole-row loads, stacked heads, rows two chunks ahead) and adds:\n- A23 prologue fold: each CTA computes its 16 query heads (RMSNorm x qk factor, interleaved RoPE\n  on sliding layers) from qkv in registers. A lane's query values are bytes 32*t4 .. 32*t4+31 of\n  the pair = head hl = t4 >> 1, d = 32*(t4 & 1) + 0..31: RoPE pairs (2i, 2i+1) stay inside the\n  lane, the RMS sum needs one shuffle with lane t4 ^ 1. Queries are rounded to FP16 (finer than\n  the BF16/FP8 the reference and SGLang use).\n- The new token: the CTA whose slice holds position n-1 computes the pair's new key (RMSNorm,\n  RoPE on sliding layers, / k_scale) and value (/ v_scale), rounds them to E4M3 (satfinite, as the\n  Triton prologue does), stores them at cache row cache_loc[b], and patches them into its own\n  fragments for token n-1 (from shared memory). No other CTA reads row n-1, so nothing waits.\n- A25 merge: every CTA writes its partial (log2 max, sum, output); the last CTA of each (request,\n  pair), by an atomic ticket, merges all slices, applies RMSNorm x sigmoid(gate), writes the BF16\n  output and resets the ticket (graph-safe: tickets are back to 0 after every call).\nRoPE frequencies come in as a tensor computed by torch exactly as reference.py does; angles use\nfull-range sincosf.\n"
att_CUDA_FUSED_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_fused {\n\nconstexpr int CHUNK = 16;\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n// Two FP32 -> E4M3 (satfinite, round to nearest even): lo -> low byte, hi -> high byte of 16 bits.\n__device__ __forceinline__ uint32_t e4m3x2(float lo, float hi) {\n  uint16_t d;\n  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n__device__ __forceinline__ uint16_t f2bf16(float x) {\n  uint16_t d;\n  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(d) : "f"(x));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n__device__ __forceinline__ void set_word(uint4& v, int i, uint32_t x) {\n  if (i == 0) v.x = x; else if (i == 1) v.y = x; else if (i == 2) v.z = x; else v.w = x;\n}\n\nstruct Chunk {\n  uint4 k[4];\n  uint4 v[4];\n};\n\ntemplate <int WARPS>\n__global__ void __launch_bounds__(WARPS * 32) fused_dec_kernel(\n    const uint16_t* __restrict__ qkv, const long* __restrict__ positions, const long* __restrict__ cache_loc,\n    const float* __restrict__ inv_freq, uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const float* __restrict__ v_scale_p, const int* __restrict__ page_table,\n    const int* __restrict__ seqlens, float* __restrict__ o_part, float* __restrict__ m_part,\n    float* __restrict__ l_part, int* __restrict__ tickets, uint16_t* __restrict__ out,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, int rope, float qk_factor, float sm_scale) {\n  __shared__ float sm_m[WARPS][2][8];\n  __shared__ float sm_l[WARPS][2][8];\n  __shared__ float sm_o[WARPS][2][8][64];\n  __shared__ __align__(16) uint8_t new_k[128];\n  __shared__ __align__(16) uint8_t new_v[128];\n  __shared__ int last_flag;\n  __shared__ float rope_c[32], rope_s[32];\n\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int g = lane >> 2, t4 = lane & 3, hl = t4 >> 1;\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int pair = blockIdx.y % pairs;\n  const int kv0 = pair * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;\n  const long qkv_row = static_cast<long>(2 * hq + 2 * hkv) * 64;\n  const uint16_t* qkv_b = qkv + static_cast<long>(b) * qkv_row;\n  const float pos = static_cast<float>(positions[b]);\n\n  const int n = seqlens[b];\n  const int lo = window > 0 ? max(n - window, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const bool has_new = start <= n - 1 && n - 1 < end;      // this CTA\'s slice holds the new token\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // ---- RoPE table for this request\'s position: one full-range sincosf per frequency (the\n  // arguments reach ~1e5 rad, CUDA\'s slow reduction path), shared by every lane.\n  if (rope) {\n    if (warp == 0) sincosf(pos * inv_freq[lane], &rope_s[lane], &rope_c[lane]);\n    __syncthreads();\n  }\n\n  // ---- new key/value of the pair (only the CTA holding position n-1), warp 0: 4 values per lane.\n  if (has_new && warp == 0) {\n    const int h = lane >> 4, d0 = 4 * (lane & 15);          // head kv0 + h, dims d0 .. d0+3\n    const uint16_t* kp = qkv_b + static_cast<long>(hq) * 64 + (kv0 + h) * 64 + d0;\n    const uint16_t* vp = qkv_b + static_cast<long>(hq + hkv) * 64 + (kv0 + h) * 64 + d0;\n    float k4[4], v4[4];\n    float ss = 0.f;\n#pragma unroll\n    for (int j = 0; j < 4; ++j) {\n      k4[j] = bf16f(kp[j]);\n      v4[j] = bf16f(vp[j]);\n      ss += k4[j] * k4[j];\n    }\n    ss += __shfl_xor_sync(FULL, ss, 1);\n    ss += __shfl_xor_sync(FULL, ss, 2);\n    ss += __shfl_xor_sync(FULL, ss, 4);\n    ss += __shfl_xor_sync(FULL, ss, 8);                       // sum over the head\'s 16 lanes\n    const float r = rsqrtf(ss / 64.f + 1e-5f);\n#pragma unroll\n    for (int j = 0; j < 4; ++j) k4[j] *= r;\n    if (rope) {\n#pragma unroll\n      for (int p = 0; p < 2; ++p) {\n        const float sn = rope_s[d0 / 2 + p], cs = rope_c[d0 / 2 + p];\n        const float e = k4[2 * p], o = k4[2 * p + 1];\n        k4[2 * p] = e * cs - o * sn;\n        k4[2 * p + 1] = e * sn + o * cs;\n      }\n    }\n    const float kinv = 1.f / (*k_scale_p), vinv = 1.f / (*v_scale_p);\n    const uint32_t kw = e4m3x2(k4[0] * kinv, k4[1] * kinv) | (e4m3x2(k4[2] * kinv, k4[3] * kinv) << 16);\n    const uint32_t vw = e4m3x2(v4[0] * vinv, v4[1] * vinv) | (e4m3x2(v4[2] * vinv, v4[3] * vinv) << 16);\n    reinterpret_cast<uint32_t*>(new_k)[lane] = kw;            // bytes 4*lane .. = head h, dims d0 ..\n    reinterpret_cast<uint32_t*>(new_v)[lane] = vw;\n    const long row = cache_loc[b];\n    reinterpret_cast<uint32_t*>(kc + row * row_bytes + poff)[lane] = kw;\n    reinterpret_cast<uint32_t*>(vc + row * row_bytes + poff)[lane] = vw;\n  }\n\n  // ---- queries: lane = head hl, dims 32*(t4&1) .. +31 of query g; RMSNorm x factor, RoPE, FP16.\n  uint32_t qa[8][4];\n  {\n    const uint16_t* qp = qkv_b + static_cast<long>((kv0 + hl) * 8 + g) * 64 + 32 * (t4 & 1);\n    float x[32];\n    float ss = 0.f;\n#pragma unroll\n    for (int j = 0; j < 32; ++j) {\n      x[j] = bf16f(qp[j]);\n      ss += x[j] * x[j];\n    }\n    ss += __shfl_xor_sync(FULL, ss, 1);                       // the other half of the head (lane t4 ^ 1)\n    const float r = rsqrtf(ss / 64.f + 1e-5f) * qk_factor;\n#pragma unroll\n    for (int j = 0; j < 32; ++j) x[j] *= r;\n    if (rope) {\n#pragma unroll\n      for (int p = 0; p < 16; ++p) {\n        const float sn = rope_s[16 * (t4 & 1) + p], cs = rope_c[16 * (t4 & 1) + p];\n        const float e = x[2 * p], o = x[2 * p + 1];\n        x[2 * p] = e * cs - o * sn;\n        x[2 * p + 1] = e * sn + o * cs;\n      }\n    }\n#pragma unroll\n    for (int ks = 0; ks < 8; ++ks) {\n      const uint32_t lo2 = pack_f16x2(x[4 * ks + 0], x[4 * ks + 1]);\n      const uint32_t hi2 = pack_f16x2(x[4 * ks + 2], x[4 * ks + 3]);\n      qa[ks][0] = hl == 0 ? lo2 : 0u;\n      qa[ks][1] = hl == 1 ? lo2 : 0u;\n      qa[ks][2] = hl == 0 ? hi2 : 0u;\n      qa[ks][3] = hl == 1 ? hi2 : 0u;\n    }\n  }\n  if (has_new) __syncthreads();                               // new_k / new_v visible to every warp\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  float m_run[2] = {-1.0e30f, -1.0e30f};\n  float l_run[2] = {0.f, 0.f};\n  float acc[16][4];\n#pragma unroll\n  for (int j = 0; j < 16; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n\n  auto row_of = [&](int c) -> long {\n    const int s = min(start + c * CHUNK + (lane & 15), end - 1);\n    return static_cast<long>(pt[s / page]) * page + s % page;\n  };\n  auto load = [&](long my_row, Chunk& ch) {\n    const long r0 = __shfl_sync(FULL, my_row, g);\n    const long r1 = __shfl_sync(FULL, my_row, 8 + g);\n    const long v0 = __shfl_sync(FULL, my_row, 2 * t4);\n    const long v1 = __shfl_sync(FULL, my_row, 2 * t4 + 1);\n    const long v2 = __shfl_sync(FULL, my_row, 2 * t4 + 8);\n    const long v3 = __shfl_sync(FULL, my_row, 2 * t4 + 9);\n    const uint8_t* kb0 = kc + r0 * row_bytes + poff + 32 * t4;\n    const uint8_t* kb1 = kc + r1 * row_bytes + poff + 32 * t4;\n    ch.k[0] = __ldg(reinterpret_cast<const uint4*>(kb0));\n    ch.k[2] = __ldg(reinterpret_cast<const uint4*>(kb1));\n    ch.k[1] = __ldg(reinterpret_cast<const uint4*>(kb0 + 16));\n    ch.k[3] = __ldg(reinterpret_cast<const uint4*>(kb1 + 16));\n    ch.v[0] = __ldg(reinterpret_cast<const uint4*>(vc + v0 * row_bytes + poff + 16 * g));\n    ch.v[1] = __ldg(reinterpret_cast<const uint4*>(vc + v1 * row_bytes + poff + 16 * g));\n    ch.v[2] = __ldg(reinterpret_cast<const uint4*>(vc + v2 * row_bytes + poff + 16 * g));\n    ch.v[3] = __ldg(reinterpret_cast<const uint4*>(vc + v3 * row_bytes + poff + 16 * g));\n  };\n  // Token n-1 of this chunk takes the new key/value from shared memory (the cache row may not be\n  // written yet as far as this CTA\'s loads are concerned).\n  auto patch = [&](int s0, Chunk& ch) {\n    if (!has_new || n - 1 < s0 || n - 1 >= s0 + CHUNK) return;\n    const int tn = n - 1 - s0;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n      if (nt * 8 + g == tn) {\n        ch.k[nt * 2] = *reinterpret_cast<const uint4*>(new_k + 32 * t4);\n        ch.k[nt * 2 + 1] = *reinterpret_cast<const uint4*>(new_k + 32 * t4 + 16);\n      }\n    const int slot[4] = {2 * t4, 2 * t4 + 1, 2 * t4 + 8, 2 * t4 + 9};\n#pragma unroll\n    for (int i = 0; i < 4; ++i)\n      if (slot[i] == tn) ch.v[i] = *reinterpret_cast<const uint4*>(new_v + 16 * g);\n  };\n\n  Chunk cur, nxt;\n  long row_next = 0, row_next2 = 0;\n  int c = warp;\n  if (c < nchunks) {\n    load(row_of(c), cur);\n    if (c + WARPS < nchunks) row_next = row_of(c + WARPS);\n  }\n  for (; c < nchunks; c += WARPS) {\n    if (c + 2 * WARPS < nchunks) row_next2 = row_of(c + 2 * WARPS);\n    if (c + WARPS < nchunks) load(row_next, nxt);\n    const int s0 = start + c * CHUNK;\n    patch(s0, cur);\n\n    float sc[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n      for (int ks = 0; ks < 8; ++ks) {\n        const uint32_t w = word(cur.k[nt * 2 + (ks >> 2)], ks & 3);\n        mma16816(sc[nt], qa[ks][0], qa[ks][1], qa[ks][2], qa[ks][3], f16x2_lo(w), f16x2_hi(w));\n      }\n    }\n    float mx0 = -INFINITY, mx1 = -INFINITY;\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n      for (int jj = 0; jj < 2; ++jj) {\n        const bool ok = s0 + nt * 8 + 2 * t4 + jj < end;\n        sc[nt][jj] = ok ? sc[nt][jj] * qk_scale : -INFINITY;\n        sc[nt][2 + jj] = ok ? sc[nt][2 + jj] * qk_scale : -INFINITY;\n        mx0 = fmaxf(mx0, sc[nt][jj]);\n        mx1 = fmaxf(mx1, sc[nt][2 + jj]);\n      }\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 1));\n    mx0 = fmaxf(mx0, __shfl_xor_sync(FULL, mx0, 2));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 1));\n    mx1 = fmaxf(mx1, __shfl_xor_sync(FULL, mx1, 2));\n    const float mn0 = fmaxf(m_run[0], mx0), mn1 = fmaxf(m_run[1], mx1);\n    const float al0 = ex2(m_run[0] - mn0), al1 = ex2(m_run[1] - mn1);\n    float p[2][4];\n#pragma unroll\n    for (int nt = 0; nt < 2; ++nt) {\n      p[nt][0] = ex2(sc[nt][0] - mn0);\n      p[nt][1] = ex2(sc[nt][1] - mn0);\n      p[nt][2] = ex2(sc[nt][2] - mn1);\n      p[nt][3] = ex2(sc[nt][3] - mn1);\n    }\n    l_run[0] = l_run[0] * al0 + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n    l_run[1] = l_run[1] * al1 + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n    m_run[0] = mn0;\n    m_run[1] = mn1;\n    const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);\n    const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);\n    const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);\n    const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);\n#pragma unroll\n    for (int j = 0; j < 16; ++j) {\n      acc[j][0] *= al0;\n      acc[j][1] *= al0;\n      acc[j][2] *= al1;\n      acc[j][3] *= al1;\n      const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n      const uint32_t b0 = f16x2_lo(__byte_perm(word(cur.v[0], j >> 2), word(cur.v[1], j >> 2), sel));\n      const uint32_t b1 = f16x2_lo(__byte_perm(word(cur.v[2], j >> 2), word(cur.v[3], j >> 2), sel));\n      mma16816(acc[j], pa0, pa1, pa2, pa3, b0, b1);\n    }\n    cur = nxt;\n    row_next = row_next2;\n  }\n\n  // ---- CTA merge of the warps (shared memory), then this slice\'s partial.\n  float l0 = l_run[0], l1 = l_run[1];\n  l0 += __shfl_xor_sync(FULL, l0, 1);\n  l0 += __shfl_xor_sync(FULL, l0, 2);\n  l1 += __shfl_xor_sync(FULL, l1, 1);\n  l1 += __shfl_xor_sync(FULL, l1, 2);\n  if (t4 == 0) {\n    sm_m[warp][0][g] = m_run[0];\n    sm_m[warp][1][g] = m_run[1];\n    sm_l[warp][0][g] = l0;\n    sm_l[warp][1][g] = l1;\n  }\n#pragma unroll\n  for (int j = 0; j < 16; ++j)\n#pragma unroll\n    for (int jj = 0; jj < 2; ++jj) sm_o[warp][hl][g][32 * (t4 & 1) + 16 * jj + j] = (hl ? acc[j][2 + jj] : acc[j][jj])   /* static indices: keeps acc in registers */;\n  __syncthreads();\n  const long part0 = (static_cast<long>(b) * hq + kv0 * 8) * nsplit;   // head kv0*8 + 8h + row, slice split\n  for (int idx = threadIdx.x; idx < 2 * 8 * 64; idx += WARPS * 32) {\n    const int h = idx >> 9, row = (idx >> 6) & 7, d = idx & 63;\n    float mm = -1.0e30f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) mm = fmaxf(mm, sm_m[w][h][row]);\n    float o = 0.f, l = 0.f;\n#pragma unroll\n    for (int w = 0; w < WARPS; ++w) {\n      const float wt = ex2(sm_m[w][h][row] - mm);\n      o += wt * sm_o[w][h][row][d];\n      l += wt * sm_l[w][h][row];\n    }\n    const long part = part0 + static_cast<long>(h * 8 + row) * nsplit + split;\n    o_part[part * 64 + d] = o;\n    if (d == 0) {\n      m_part[part] = mm;\n      l_part[part] = l;\n    }\n  }\n\n  // ---- A25: the last slice of this (request, pair) merges every slice and writes the output.\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x == 0) {\n    const int ticket = atomicAdd(tickets + blockIdx.y, 1);\n    last_flag = ticket == nsplit - 1;\n  }\n  __syncthreads();\n  if (!last_flag) return;\n  __threadfence();\n  // CTA-wide merge of nsplit (<= 128, host-enforced) slices for the pair\'s 16 (head, row)s,\n  // reusing sm_o: wts[16][128] slice weights, then outs[16][64]. Partials are read through L2\n  // (ld.cg): other CTAs wrote them.\n  float* wts = &sm_o[0][0][0][0];\n  float* outs = wts + 16 * 128;\n  __shared__ float row_l[16];\n  for (int hr = warp; hr < 16; hr += WARPS) {\n    const long pbase = part0 + static_cast<long>(hr) * nsplit;\n    float mm = -1.0e30f;\n    for (int sp = lane; sp < nsplit; sp += 32) mm = fmaxf(mm, __ldcg(m_part + pbase + sp));\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) mm = fmaxf(mm, __shfl_xor_sync(FULL, mm, o));\n    float l = 0.f;\n    for (int sp = lane; sp < nsplit; sp += 32) {\n      const float wt = ex2(__ldcg(m_part + pbase + sp) - mm);\n      wts[hr * 128 + sp] = wt;\n      l += wt * __ldcg(l_part + pbase + sp);\n    }\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) l += __shfl_xor_sync(FULL, l, o);\n    if (lane == 0) row_l[hr] = l;\n  }\n  __syncthreads();\n  constexpr int PER = 1024 / (WARPS * 32);                    // output elements per thread\n  float acc_o[PER];\n#pragma unroll\n  for (int k = 0; k < PER; ++k) acc_o[k] = 0.f;\n#pragma unroll 2\n  for (int sp = 0; sp < nsplit; ++sp) {\n#pragma unroll\n    for (int k = 0; k < PER; ++k) {\n      const int e = threadIdx.x + k * WARPS * 32, hr = e >> 6, d = e & 63;\n      acc_o[k] += wts[hr * 128 + sp] * __ldcg(o_part + (part0 + static_cast<long>(hr) * nsplit + sp) * 64 + d);\n    }\n  }\n#pragma unroll\n  for (int k = 0; k < PER; ++k) outs[threadIdx.x + k * WARPS * 32] = acc_o[k];\n  __syncthreads();\n  const float vsc = *v_scale_p;\n  for (int hr = warp; hr < 16; hr += WARPS) {                 // RMSNorm x sigmoid(gate), one warp per row\n    const float scale = vsc / row_l[hr];\n    const float o0 = outs[hr * 64 + lane] * scale, o1 = outs[hr * 64 + 32 + lane] * scale;\n    float ss = o0 * o0 + o1 * o1;\n#pragma unroll\n    for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(FULL, ss, o);\n    const float r = rsqrtf(ss / 64.f + 1e-5f);\n    const int head = kv0 * 8 + hr;\n    const uint16_t* gp = qkv_b + static_cast<long>(hq + 2 * hkv) * 64 + head * 64;\n    const float g0 = bf16f(gp[lane]), g1 = bf16f(gp[32 + lane]);\n    uint16_t* op = out + static_cast<long>(b) * hq * 64 + head * 64;\n    op[lane] = f2bf16(o0 * r / (1.f + __expf(-g0)));\n    op[32 + lane] = f2bf16(o1 * r / (1.f + __expf(-g1)));\n  }\n  if (threadIdx.x == 0) tickets[blockIdx.y] = 0;              // ready for the next call (and replay)\n}\n\n}  // namespace okc_fused\n\nvoid fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq,\n               torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale,\n               torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n               torch::Tensor l_part, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq,\n               int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope,\n               double qk_factor, double sm_scale, int64_t warps) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n#define OKC_FUSED_ARGS                                                                                        \\\n  reinterpret_cast<const uint16_t*>(qkv.data_ptr()), reinterpret_cast<const long*>(positions.data_ptr()),    \\\n      reinterpret_cast<const long*>(cache_loc.data_ptr()), inv_freq.data_ptr<float>(),                      \\\n      reinterpret_cast<uint8_t*>(kc.data_ptr()), reinterpret_cast<uint8_t*>(vc.data_ptr()),                 \\\n      k_scale.data_ptr<float>(), v_scale.data_ptr<float>(), page_table.data_ptr<int>(),                     \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      tickets.data_ptr<int>(), reinterpret_cast<uint16_t*>(out.data_ptr()), static_cast<int>(hq),           \\\n      static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit), static_cast<int>(page), \\\n      static_cast<int>(window), static_cast<int>(rope), static_cast<float>(qk_factor), static_cast<float>(sm_scale)\n  if (warps == 8) okc_fused::fused_dec_kernel<8><<<grid, 256, 0, stream>>>(OKC_FUSED_ARGS);\n  else okc_fused::fused_dec_kernel<4><<<grid, 128, 0, stream>>>(OKC_FUSED_ARGS);\n#undef OKC_FUSED_ARGS\n}\n\nstd::vector<int64_t> fused_dec_attrs(int64_t warps) {\n  cudaFuncAttributes a;\n  if (warps == 8) cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<8>);\n  else cudaFuncGetAttributes(&a, okc_fused::fused_dec_kernel<4>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_FUSED_DECL = 'void fused_dec(torch::Tensor qkv, torch::Tensor positions, torch::Tensor cache_loc, torch::Tensor inv_freq, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor v_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, torch::Tensor tickets, torch::Tensor out, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, int64_t rope, double qk_factor, double sm_scale, int64_t warps);\nstd::vector<int64_t> fused_dec_attrs(int64_t warps);'
"S2a (A28): CUDA split-context attention for 8-token verify blocks (q_len 8, causal inside the\nblock), between the Triton prologue (q scratch, new cache rows) and the Triton _combine.\n\nOne CTA = (request b, KV-head pair, slice of keys), 4 warps. Each 16-key chunk's whole 128-byte\nK and V rows are copied once into a STAGES-deep shared-memory ring with 16-byte cp.async (one K\nand one V copy per thread per chunk; rows fetched one chunk ahead of their copies). Warp w owns\nhead kv0 + (w >> 1) and m-tiles 2*(w&1), 2*(w&1)+1: an m-tile's rows g / g+8 are query head g of\nblock tokens 2*mt / 2*mt+1, so all 64 rows per head are real (no stacking: verify can't afford\nthe 2x MMA work, AF17) and warps own disjoint rows (no intra-CTA merge).\nFragments come from shared memory with A5a's per-head byte permutation:\n- K (B of QK, m16n8k16 f16): lane (g, t4) reads 16 bytes at 16*t4 of token g (tile 0) / 8+g\n  (tile 1) in its head; logical k 2*t4+{0,1} (+8) of step ks <-> byte 16*t4 + 4*ks + {0,1} ({2,3}).\n- V (B of PV): lane (g, t4) reads 8 bytes at 8*g of tokens 2*t4, 2*t4+1, 2*t4+8, 2*t4+9;\n  logical column g of d-tile j <-> byte 8*g + j. Output column 2*t4+jj of tile j -> d = 8*(2*t4+jj) + j.\nRow pitches make both the 16 B K reads (8 lanes per phase: 2 tokens x 4 chunks) and the 8 B V\nreads (16 lanes per phase) bank-conflict free: K pitch 192 B (consecutive tokens shift by 4\n16-byte units), V pitch 144 B (by 2 8-byte units); the 16 B cp.async writes of a row stay\nconflict-free under both. Causal mask: key <= n - 8 + token; sliding window: key > that - window.\n"
att_CUDA_VER_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cstdint>\n#include <cmath>\n#include <vector>\n\nnamespace okc_ver {\n\nconstexpr int CHUNK = 16;\nconstexpr int L = 8;\nconstexpr int KP = 192;   // K row pitch in shared memory (bytes)\nconstexpr int VP = 144;   // V row pitch\nconstexpr unsigned FULL = 0xffffffffu;\n\n__device__ __forceinline__ uint32_t f16x2_lo(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, l;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t f16x2_hi(uint32_t w) {\n  uint32_t d;\n  asm("{\\n\\t.reg .b16 l, h;\\n\\tmov.b32 {l, h}, %1;\\n\\tcvt.rn.f16x2.e4m3x2 %0, h;\\n\\t}" : "=r"(d) : "r"(w));\n  return d;\n}\n__device__ __forceinline__ uint32_t pack_f16x2(float lo, float hi) {\n  uint32_t d;\n  asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(hi), "f"(lo));\n  return d;\n}\n__device__ __forceinline__ float ex2(float x) {\n  float y;\n  asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));\n  return y;\n}\n__device__ __forceinline__ float bf16f(uint16_t v) { return __uint_as_float(static_cast<uint32_t>(v) << 16); }\n__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile(\n      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\\n"\n      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void cp16(void* dst, const void* src) {\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));\n}\n__device__ __forceinline__ uint32_t w4(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }\n\ntemplate <int STAGES>\n__global__ void __launch_bounds__(128) attend_ver_kernel(\n    const uint16_t* __restrict__ q, const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,\n    const float* __restrict__ k_scale_p, const int* __restrict__ page_table, const int* __restrict__ seqlens,\n    float* __restrict__ o_part, float* __restrict__ m_part, float* __restrict__ l_part,\n    int hq, int hkv, int max_pages, int nsplit, int page, int window, float sm_scale) {\n  __shared__ __align__(16) uint8_t sk[STAGES][CHUNK * KP];\n  __shared__ __align__(16) uint8_t sv[STAGES][CHUNK * VP];\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t4 = lane & 3;\n  const int h = warp >> 1, mt0 = 2 * (warp & 1);\n  const int pairs = hkv >> 1;\n  const int b = blockIdx.y / pairs;\n  const int kv0 = (blockIdx.y % pairs) * 2;\n  const int split = blockIdx.x;\n  const long row_bytes = static_cast<long>(hkv) * 64;\n  const long poff = static_cast<long>(kv0) * 64;\n\n  const int n = seqlens[b];                         // keys 0 .. n-1; block token l sits at n - L + l\n  const int lo = window > 0 ? max(n - L - window + 1, 0) : 0;\n  int per = (n - lo + nsplit - 1) / nsplit;\n  per = (per + CHUNK - 1) / CHUNK * CHUNK;\n  const int start = lo + split * per;\n  const int end = min(start + per, n);\n  const int nchunks = end > start ? (end - start + CHUNK - 1) / CHUNK : 0;\n  const float qk_scale = sm_scale * (*k_scale_p) * 1.4426950408889634f;\n\n  // Query fragments: m-tile mt = mt0 + mi, rows g -> token 2*mt, g+8 -> token 2*mt+1, head (kv0+h)*8+g.\n  uint32_t qa[2][4][4];\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi) {\n    const int ta = 2 * (mt0 + mi);\n    const uint16_t* qA = q + ((static_cast<long>(b) * L + ta) * hq + (kv0 + h) * 8 + g) * 64 + 16 * t4;\n    const uint16_t* qB = qA + static_cast<long>(hq) * 64;               // token ta + 1\n#pragma unroll\n    for (int ks = 0; ks < 4; ++ks) {\n      qa[mi][ks][0] = pack_f16x2(bf16f(qA[4 * ks + 0]), bf16f(qA[4 * ks + 1]));\n      qa[mi][ks][1] = pack_f16x2(bf16f(qB[4 * ks + 0]), bf16f(qB[4 * ks + 1]));\n      qa[mi][ks][2] = pack_f16x2(bf16f(qA[4 * ks + 2]), bf16f(qA[4 * ks + 3]));\n      qa[mi][ks][3] = pack_f16x2(bf16f(qB[4 * ks + 2]), bf16f(qB[4 * ks + 3]));\n    }\n  }\n\n  const int* pt = page_table + static_cast<long>(b) * max_pages;\n  const int ctok = tid >> 3, cch = tid & 7;          // this thread\'s copy: token ctok, 16 B chunk cch\n  auto row_of = [&](int c) -> long {\n    const int s = min(start + c * CHUNK + ctok, end - 1);\n    return static_cast<long>(pt[s / page]) * page + s % page;\n  };\n  auto issue = [&](int c, long row) {\n    const int st = c % STAGES;\n    cp16(&sk[st][ctok * KP + 16 * cch], kc + row * row_bytes + poff + 16 * cch);\n    cp16(&sv[st][ctok * VP + 16 * cch], vc + row * row_bytes + poff + 16 * cch);\n  };\n\n  float m_run[2][2], l_run[2][2], acc[2][8][4];\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi) {\n    m_run[mi][0] = m_run[mi][1] = -1.0e30f;\n    l_run[mi][0] = l_run[mi][1] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) acc[mi][j][0] = acc[mi][j][1] = acc[mi][j][2] = acc[mi][j][3] = 0.f;\n  }\n\n#pragma unroll\n  for (int i = 0; i < STAGES - 1; ++i) {\n    if (i < nchunks) issue(i, row_of(i));\n    asm volatile("cp.async.commit_group;");\n  }\n  long rnext = STAGES - 1 < nchunks ? row_of(STAGES - 1) : 0;\n\n  for (int c = 0; c < nchunks; ++c) {\n    asm volatile("cp.async.wait_group %0;" ::"n"(STAGES - 2));\n    __syncthreads();\n    if (c + STAGES - 1 < nchunks) issue(c + STAGES - 1, rnext);\n    asm volatile("cp.async.commit_group;");\n    if (c + STAGES < nchunks) rnext = row_of(c + STAGES);\n\n    const int st = c % STAGES;\n    const int s0 = start + c * CHUNK;\n    const uint4 kb0 = *reinterpret_cast<const uint4*>(&sk[st][g * KP + 64 * h + 16 * t4]);\n    const uint4 kb1 = *reinterpret_cast<const uint4*>(&sk[st][(8 + g) * KP + 64 * h + 16 * t4]);\n    const uint2 vb0 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4) * VP + 64 * h + 8 * g]);\n    const uint2 vb1 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 1) * VP + 64 * h + 8 * g]);\n    const uint2 vb2 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 8) * VP + 64 * h + 8 * g]);\n    const uint2 vb3 = *reinterpret_cast<const uint2*>(&sv[st][(2 * t4 + 9) * VP + 64 * h + 8 * g]);\n    // a chunk needs masks only at the slice tail or where it reaches the block\'s own keys\n    const bool masked = s0 + CHUNK > end || s0 + CHUNK > n - L || window > 0;\n\n#pragma unroll\n    for (int mi = 0; mi < 2; ++mi) {\n      float sc[2][4];\n#pragma unroll\n      for (int nt = 0; nt < 2; ++nt) {\n        const uint4& kb = nt == 0 ? kb0 : kb1;\n        sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;\n#pragma unroll\n        for (int ks = 0; ks < 4; ++ks) {\n          const uint32_t w = w4(kb, ks);\n          mma16816(sc[nt], qa[mi][ks][0], qa[mi][ks][1], qa[mi][ks][2], qa[mi][ks][3], f16x2_lo(w), f16x2_hi(w));\n        }\n      }\n      const int posA = n - L + 2 * (mt0 + mi), posB = posA + 1;     // rows g / g+8\n      float mxA = -INFINITY, mxB = -INFINITY;\n#pragma unroll\n      for (int nt = 0; nt < 2; ++nt)\n#pragma unroll\n        for (int jj = 0; jj < 2; ++jj) {\n          const int key = s0 + nt * 8 + 2 * t4 + jj;\n          float xa = sc[nt][jj] * qk_scale, xb = sc[nt][2 + jj] * qk_scale;\n          if (masked) {\n            const bool ok = key < end;\n            const bool okA = ok && key <= posA && (window <= 0 || key > posA - window);\n            const bool okB = ok && key <= posB && (window <= 0 || key > posB - window);\n            xa = okA ? xa : -INFINITY;\n            xb = okB ? xb : -INFINITY;\n          }\n          sc[nt][jj] = xa;\n          sc[nt][2 + jj] = xb;\n          mxA = fmaxf(mxA, xa);\n          mxB = fmaxf(mxB, xb);\n        }\n      mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 1));\n      mxA = fmaxf(mxA, __shfl_xor_sync(FULL, mxA, 2));\n      mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 1));\n      mxB = fmaxf(mxB, __shfl_xor_sync(FULL, mxB, 2));\n      const float mnA = fmaxf(m_run[mi][0], mxA), mnB = fmaxf(m_run[mi][1], mxB);\n      const float alA = ex2(m_run[mi][0] - mnA), alB = ex2(m_run[mi][1] - mnB);\n      float p[2][4];\n#pragma unroll\n      for (int nt = 0; nt < 2; ++nt) {\n        p[nt][0] = ex2(sc[nt][0] - mnA);\n        p[nt][1] = ex2(sc[nt][1] - mnA);\n        p[nt][2] = ex2(sc[nt][2] - mnB);\n        p[nt][3] = ex2(sc[nt][3] - mnB);\n      }\n      l_run[mi][0] = l_run[mi][0] * alA + (p[0][0] + p[0][1] + p[1][0] + p[1][1]);\n      l_run[mi][1] = l_run[mi][1] * alB + (p[0][2] + p[0][3] + p[1][2] + p[1][3]);\n      m_run[mi][0] = mnA;\n      m_run[mi][1] = mnB;\n      const uint32_t pa0 = pack_f16x2(p[0][0], p[0][1]);\n      const uint32_t pa1 = pack_f16x2(p[0][2], p[0][3]);\n      const uint32_t pa2 = pack_f16x2(p[1][0], p[1][1]);\n      const uint32_t pa3 = pack_f16x2(p[1][2], p[1][3]);\n#pragma unroll\n      for (int j = 0; j < 8; ++j) {\n        acc[mi][j][0] *= alA;\n        acc[mi][j][1] *= alA;\n        acc[mi][j][2] *= alB;\n        acc[mi][j][3] *= alB;\n        const uint32_t sel = static_cast<uint32_t>((j & 3) | ((4 + (j & 3)) << 4));\n        const uint32_t b0 = f16x2_lo(__byte_perm(j < 4 ? vb0.x : vb0.y, j < 4 ? vb1.x : vb1.y, sel));\n        const uint32_t b1 = f16x2_lo(__byte_perm(j < 4 ? vb2.x : vb2.y, j < 4 ? vb3.x : vb3.y, sel));\n        mma16816(acc[mi][j], pa0, pa1, pa2, pa3, b0, b1);\n      }\n    }\n  }\n  asm volatile("cp.async.wait_all;");\n\n  // Partials: rows (token 2*mt + r, head (kv0+h)*8 + g), d = 8*(2*t4+jj) + j.\n#pragma unroll\n  for (int mi = 0; mi < 2; ++mi)\n#pragma unroll\n    for (int r = 0; r < 2; ++r) {\n      float l = l_run[mi][r];\n      l += __shfl_xor_sync(FULL, l, 1);\n      l += __shfl_xor_sync(FULL, l, 2);\n      const int tok = 2 * (mt0 + mi) + r;\n      const long part = ((static_cast<long>(b) * L + tok) * hq + (kv0 + h) * 8 + g) * nsplit + split;\n#pragma unroll\n      for (int j = 0; j < 8; ++j)\n#pragma unroll\n        for (int jj = 0; jj < 2; ++jj) o_part[part * 64 + 8 * (2 * t4 + jj) + j] = acc[mi][j][2 * r + jj];\n      if (t4 == 0) {\n        m_part[part] = m_run[mi][r];\n        l_part[part] = l;\n      }\n    }\n}\n\n}  // namespace okc_ver\n\nvoid attend_ver(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale,\n                torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part,\n                torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit,\n                int64_t page, int64_t window, double sm_scale, int64_t stages) {\n  auto stream = at::cuda::getCurrentCUDAStream();\n  dim3 grid(static_cast<unsigned>(nsplit), static_cast<unsigned>(batch * (hkv / 2)));\n#define OKC_VER_ARGS                                                                                         \\\n  reinterpret_cast<const uint16_t*>(q.data_ptr()), reinterpret_cast<const uint8_t*>(kc.data_ptr()),          \\\n      reinterpret_cast<const uint8_t*>(vc.data_ptr()), k_scale.data_ptr<float>(), page_table.data_ptr<int>(), \\\n      seqlens.data_ptr<int>(), o_part.data_ptr<float>(), m_part.data_ptr<float>(), l_part.data_ptr<float>(), \\\n      static_cast<int>(hq), static_cast<int>(hkv), static_cast<int>(max_pages), static_cast<int>(nsplit),     \\\n      static_cast<int>(page), static_cast<int>(window), static_cast<float>(sm_scale)\n  if (stages == 4) okc_ver::attend_ver_kernel<4><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);\n  else if (stages == 6) okc_ver::attend_ver_kernel<6><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);\n  else okc_ver::attend_ver_kernel<3><<<grid, 128, 0, stream>>>(OKC_VER_ARGS);\n#undef OKC_VER_ARGS\n}\n\nstd::vector<int64_t> attend_ver_attrs(int64_t stages) {\n  cudaFuncAttributes a;\n  if (stages == 4) cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<4>);\n  else if (stages == 6) cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<6>);\n  else cudaFuncGetAttributes(&a, okc_ver::attend_ver_kernel<3>);\n  return {a.numRegs, static_cast<int64_t>(a.localSizeBytes), static_cast<int64_t>(a.sharedSizeBytes)};\n}\n'
att_CUDA_VER_DECL = 'void attend_ver(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, torch::Tensor k_scale, torch::Tensor page_table, torch::Tensor seqlens, torch::Tensor o_part, torch::Tensor m_part, torch::Tensor l_part, int64_t batch, int64_t hq, int64_t hkv, int64_t max_pages, int64_t nsplit, int64_t page, int64_t window, double sm_scale, int64_t stages);\nstd::vector<int64_t> attend_ver_attrs(int64_t stages);'
import tempfile as _tempfile
att__CUDA = {}
att__TICKETS = []
att_CUDA_ROUTE = {(True, False, 16): ('dec2', 4, 4), (True, False, 32): ('dec2', 8, 4)}

def att__cuda_build():
    from torch.utils.cpp_extension import load_inline
    src = att_CUDA_DEC2_SRC + '\n' + att_CUDA_FUSED_SRC + '\n' + att_CUDA_VER_SRC
    decl = att_CUDA_DEC2_DECL + '\n' + att_CUDA_FUSED_DECL + '\n' + att_CUDA_VER_DECL
    try:
        att__CUDA['mod'] = load_inline(name='okc_attention_v3', cpp_sources=decl, cuda_sources=src, functions=['attend_dec2', 'attend_dec2_attrs', 'fused_dec', 'fused_dec_attrs', 'attend_ver', 'attend_ver_attrs'], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a'], build_directory=_tempfile.mkdtemp(prefix='okc_attention_v3_'), verbose=False)
    except Exception as exc:
        print(f'v3: CUDA build failed, staying on v2.2 paths: {type(exc).__name__}: {str(exc)[-300:]}')

def att__nsplit(cfg, max_pages, ctas_per_sm, min_keys):
    tokens = max_pages * cfg['page_size']
    if cfg['window']:
        tokens = min(tokens, cfg['window'] + cfg['q_len'])
    programs = cfg['batch'] * (cfg['num_kv_heads'] // 2)
    return max(1, min(math.ceil(ctas_per_sm * att_NUM_SMS / programs), tokens // min_keys, att_MAX_SPLITS))

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
    kind, cps, knob = route
    cfg = data['config']
    batch, length, hq, hkv = (cfg['batch'], cfg['q_len'], cfg['num_q_heads'], cfg['num_kv_heads'])
    device = data['qkv'].device
    max_pages = data['page_table'].shape[1]
    mod = att__CUDA['mod']
    if kind == 'fused':
        nsplit = att__nsplit(cfg, max_pages, cps, 32 * knob)
        o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        out = torch.empty((batch, 1, hq * 64), dtype=torch.bfloat16, device=device)
        rope = bool(cfg['rope_theta'])
        mod.fused_dec(data['qkv'], data['positions'], data['cache_loc'], att__inv_freq(cfg['rope_theta'] if rope else 1.0, device), data['k_cache'], data['v_cache'], data['k_scale'], data['v_scale'], data['page_table'], data['cache_seqlens'], o_part, m_part, l_part, att__TICKETS[0], out, batch, hq, hkv, max_pages, nsplit, cfg['page_size'], cfg['window'], int(rope), cfg['qk_scale_factor'], cfg['sm_scale'], knob)
        return out
    q = att__prologue_q(data)
    if kind == 'dec2':
        nsplit = att__nsplit(cfg, max_pages, cps, 32 * knob)
        o_part = torch.empty((batch * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((batch * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        mod.attend_dec2(q, data['k_cache'], data['v_cache'], data['k_scale'], data['page_table'], data['cache_seqlens'], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg['page_size'], cfg['window'], cfg['sm_scale'], knob)
    else:
        nsplit = att__nsplit(cfg, max_pages, cps, 64)
        tokens = batch * length
        o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=device)
        m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=device)
        l_part = torch.empty_like(m_part)
        mod.attend_ver(q, data['k_cache'], data['v_cache'], data['k_scale'], data['page_table'], data['cache_seqlens'], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg['page_size'], cfg['window'], cfg['sm_scale'], knob)
    return att__merge(data, o_part, m_part, l_part, nsplit)

def att__cuda_ok(cfg, route):
    """The CUDA kernels need groups of 8 query heads, paired KV heads, q_len 1 (fused/dec2) or 8
    (ver), and the fused merge at most 128 slices."""
    kind, cps, knob = route
    if cfg['num_q_heads'] // cfg['num_kv_heads'] != 8 or cfg['num_kv_heads'] % 2:
        return False
    if kind in ('fused', 'dec2') and cfg['q_len'] != 1:
        return False
    if kind == 'ver' and cfg['q_len'] != 8:
        return False
    return True
att__kernel_fn_v22 = att_kernel_fn
att__warm_up_v22 = att__warm_up

def att__warm_up():
    att__warm_up_v22()
    att__cuda_build()
    if 'mod' not in att__CUDA:
        return
    dev = torch.device('cuda')
    att__TICKETS.append(torch.zeros((4096,), dtype=torch.int32, device=dev))
    for cls, route in att_CUDA_ROUTE.items():
        full, verify, bclass = cls
        length = 8 if verify else 1
        data = att_dummy_inputs(bclass, length, 0 if full else 2048, att_scored_max_pages(bclass, length), dev)
        att__cuda_run(data, route)

def att_kernel_fn(data):
    if not att__WARM and (not torch.cuda.is_current_stream_capturing()):
        att__warm_up()
    cfg = data['config']
    route = att_CUDA_ROUTE.get(att_shape_class(cfg))
    if route is not None and 'mod' in att__CUDA and att__cuda_ok(cfg, route):
        return att__cuda_run(data, route)
    return att__kernel_fn_v22(data)
