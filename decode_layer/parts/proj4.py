# Embedded: the bf16_proj session's scored v4 (proj/scored/submission_v4.py, sha256 4048d4cec26d1688...;
# scored 1.247x on muse_bf16_proj, job 0d1dc429eebf455f), every top-level name prefixed with 'pj_' by
# decode_layer/local/make_attn.py. pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m); P9 needs pj_p9_setup(state).

"""Dense projections: per-shape dispatch between SGLang's path and our own kernels.

CHOICE maps (layer, m) to a kernel; shapes not in it run SGLang's path exactly as baseline.py
does (cuBLAS; the router FP32 up to 8 tokens, BF16 hi + lo above). Our kernels:

  _swap_kernel  m <= 32: weights as the MMA's wide side, tokens padded to BM (16 or 32) as the
                narrow side; split-K over grid axis 1 with FP32 atomics when SPLIT > 1.
  _gemm_kernel  m >= 128: tokens as M, weights as N, split-K over grid axis 2 likewise.

  _fma_kernel   small m: CUDA-core GEMV. Each program loads its [BN, K / split] weight slice
                into registers at once, then for each token multiplies and row-sums in FP32
                (exact FP32 for the router's FP32 weights); split-K adds with FP32 atomics.

MODE 0: BF16 weights. MODE 1: router, FP32 weights split in registers into a BF16 hi + lo pair,
both accumulated into one FP32 accumulator (only above 8 tokens, as SGLang does). MODE 2:
router, exact FP32 (ieee) dot.
"""
from typing import TypedDict
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.srt.layers.quantization.unquant import _bf16_gemm_dispatch_impl

class pj_Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor

class pj_Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str

class pj_Inputs(TypedDict):
    x: torch.Tensor
    config: pj_Config
pj_ROUTER_FP32_MAX_TOKENS = 8
pj_CHOICE = {('o', 1): {'kind': 'fma', 'BN': 4, 'split': 1, 'warps': 8}, ('pooler', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('unpooler', 1): {'kind': 'tiny', 'max_m': 16}, ('pooler', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 8): {'kind': 'p9', 'NT': 1, 'RTW': 1, 'RG': 2, 'KSW': 8, 'D': 2, 'CS': 8}, ('unpooler', 8): {'kind': 'fi_mm', 'backend': 'tinygemm'}, ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('pooler', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 16): {'kind': 'p9', 'NT': 2, 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'CS': 8}, ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 128): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 256): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('qkv_gate', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('router', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('qkv_gate', 8): {'kind': 'fi_mm', 'backend': 'auto'}, ('pooler', 128): {'kind': 'dg', 'native': True}, ('qkv_gate', 128): {'kind': 'fi_mm', 'backend': 'cublaslt'}}
pj_P9_ENABLED = False

@triton.jit(do_not_specialize=['M'])
def pj__swap_kernel(w_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    m_mask = offs_m < M
    k0 = pid_k * k_per_split
    w_ptrs = w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        w = tl.load(w_ptrs + kk, mask=n_mask[:, None], other=0.0, eviction_policy='evict_first')
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        if MODE == 0:
            acc = tl.dot(w, xt, acc)
        elif MODE == 1:
            hi = w.to(tl.bfloat16)
            lo = (w - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(hi, xt, acc)
            acc = tl.dot(lo, xt, acc)
        else:
            acc = tl.dot(w, xt.to(tl.float32), acc, input_precision='ieee')
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem='relaxed')
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)

@triton.jit(do_not_specialize=['M'])
def pj__gemm_kernel(x_ptr, w_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * k_per_split
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        x = tl.load(x_ptrs + kk, mask=m_mask[:, None], other=0.0)
        wt = tl.load(w_ptrs + kk, mask=n_mask[None, :], other=0.0)
        if MODE == 0:
            acc = tl.dot(x, wt, acc)
        elif MODE == 1:
            hi = wt.to(tl.bfloat16)
            lo = (wt - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(x, hi, acc)
            acc = tl.dot(x, lo, acc)
        else:
            acc = tl.dot(x.to(tl.float32), wt, acc, input_precision='ieee')
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem='relaxed')
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)

@triton.jit
def pj__fma_kernel(w_ptr, x_ptr, out_ptr, N, stride_wn, stride_xm, stride_om, M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)
    n_mask = offs_n < N
    w = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0.0, eviction_policy='evict_first').to(tl.float32)
    for j in tl.static_range(M):
        xj = tl.load(x_ptr + j * stride_xm + offs_k).to(tl.float32)
        r = tl.sum(w * xj[None, :], axis=1)
        o_ptrs = out_ptr + j * stride_om + offs_n
        if ATOMIC:
            tl.atomic_add(o_ptrs, r, mask=n_mask, sem='relaxed')
        elif OUT_BF16:
            tl.store(o_ptrs, r.to(tl.bfloat16), mask=n_mask)
        else:
            tl.store(o_ptrs, r, mask=n_mask)

def pj_run_fma(c, x, w, n):
    """_fma_kernel per config c: BN rows per program, split-K, warps."""
    m, k = x.shape
    split = c['split']
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    pj__fma_kernel[triton.cdiv(n, c['BN']), split](w, x, out, n, w.stride(0), x.stride(0), out.stride(0), M=m, BN=c['BN'], BK=k // split, ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'])
    if atomic and (not fp32_out):
        out = out.to(torch.bfloat16)
    return out

def pj_run_triton(c, x, w, n):
    """One of our kernels per config c: kind, BM, BN, BK, split, mode, warps, stages."""
    m, k = x.shape
    split = c['split']
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    kps = k // split
    bm, bn, bk = (c['BM'], c['BN'], c['BK'])
    if c['kind'] == 'swap':
        pj__swap_kernel[triton.cdiv(n, bn), split](w, x, out, m, n, kps, w.stride(0), x.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, MODE=c['mode'], ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'], num_stages=c['stages'])
    else:
        pj__gemm_kernel[triton.cdiv(m, bm), triton.cdiv(n, bn), split](x, w, out, m, n, kps, x.stride(0), w.stride(0), out.stride(0), BM=bm, BN=bn, BK=bk, MODE=c['mode'], ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c['warps'], num_stages=c['stages'])
    if atomic and (not fp32_out):
        out = out.to(torch.bfloat16)
    return out

def pj_run_library(c, x, w, n):
    """BF16 GEMMs from libraries in the judge image (FlashInfer 0.7.0.post1, DeepGEMM), each with
    its own handle or kernels; none changes torch's or cuBLAS's global settings."""
    kind = c['kind']
    if kind == 'fi_mm':
        import flashinfer
        return flashinfer.mm_bf16(x, w.T, out_dtype=torch.bfloat16, backend=c['backend'])
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    if kind == 'fi_wsk':
        from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import WarpSplitKTactic, run_warp_splitk_dense
        return run_warp_splitk_dense(x, w.T, out, False, WarpSplitKTactic(*c['tactic']))
    if kind == 'fi_direct':
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense
        return run_direct_dense(x, w.T, out, False, DirectTactic(*c['tactic']))
    if kind == 'fi_tiny':
        from flashinfer.gemm.routergemm import tinygemm_bf16
        tinygemm_bf16(x, w, out)
        return out
    import deep_gemm
    deep_gemm.use_deterministic_algorithms(c['native'])
    deep_gemm.bf16_gemm_nt(x, w, out)
    return out

def pj_run_choice(c, x, layer, n, state):
    if c['kind'] in ('swap', 'gemm'):
        return pj_run_triton(c, x, state[layer][:n], n)
    if c['kind'] == 'fma':
        return pj_run_fma(c, x, state[layer][:n], n)
    if c['kind'] == 'p9':
        return pj_run_p9(c, x, layer, n, state)
    if c['kind'] in ('fi_mm', 'fi_wsk', 'fi_direct', 'fi_tiny', 'dg'):
        return pj_run_library(c, x, state[layer][:n], n)
    if c['kind'] == 'sgl_router':
        from sglang.kernels.ops.muse_spark_v12.router_gemm import router_gemm
        return router_gemm(x.float(), state['router_t'])
    if c['kind'] == 'gemv':
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        return hopper_bf16_gemv(x, state[layer][:n])
    if c['kind'] == 'tiny':
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
        return tiny_gemm_bf16(x, state[layer][:n], max_m=c['max_m'])
    raise ValueError(f"unknown kernel kind {c['kind']}")

def pj_sglang_path(x, layer, n, state):
    """Exactly baseline.py's kernel_fn."""
    if layer != 'router':
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= pj_ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state['router'])
    acc = torch.mm(x, state['router_hi_t'], out_dtype=torch.float32)
    acc += torch.mm(x, state['router_lo_t'], out_dtype=torch.float32)
    return acc

def pj__warm(state):
    """Run every chosen kernel once on zeros, so compiling happens here and not in a capture."""
    for (layer, m), c in pj_CHOICE.items():
        n = 1024 if layer == 'unpooler' and m <= 8 else state[layer].shape[0]
        x = torch.zeros((m, state[layer].shape[1]), dtype=torch.bfloat16, device=state[layer].device)
        if c['kind'] == 'fi_mm':
            import flashinfer
            with flashinfer.autotune(True):
                pj_run_choice(c, x, layer, n, state)
        pj_run_choice(c, x, layer, n, state)
    torch.cuda.synchronize()

def pj_prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights['router']
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = lo.t().contiguous()
    state['router_t'] = router.t().contiguous()
    if pj_P9_ENABLED:
        pj_p9_setup(state)
    pj__warm(state)
    return state

def pj_kernel_fn(data, state):
    x, cfg = (data['x'], data['config'])
    layer, n = (cfg['layer'], cfg['n'])
    c = pj_CHOICE.get((layer, x.shape[0]))
    if c is None:
        return pj_sglang_path(x, layer, n, state)
    return pj_run_choice(c, x, layer, n, state)
import tempfile

def pj_k_perm(k):
    """Column order of the packed weights: inside every 64-wide K chunk c, packed column 16 j + col
    (MMA step j = 0..3, A column col = 0..15) holds W column 64 c + 16 t + 4 j + 2 h + p, where
    col = 8 h + 2 t + p. Then lane (g, t)'s B values for the chunk's 4 MMAs are x[64 c + 16 t ..
    64 c + 16 t + 16): two 16-byte loads, b0 of step j = 32-bit word 2 j, b1 = word 2 j + 1."""
    idx = []
    for c in range(k // 64):
        for j in range(4):
            for col in range(16):
                h, t, p = (col // 8, col % 8 // 2, col % 2)
                idx.append(64 * c + 16 * t + 4 * j + 2 * h + p)
    return torch.tensor(idx, dtype=torch.long)

def pj_p9_pack(w):
    """W [N, K] bf16 -> [N/16, K/16, 32, 8] bf16 in A-fragment order (one lane's uint4 = 8 bf16),
    columns permuted by k_perm inside each 64-wide chunk."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and (k % 64 == 0), (w.shape, w.dtype)
    w = w[:, pj_k_perm(k).to(w.device)]
    v = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)
    v = v.permute(0, 3, 2, 5, 4, 1, 6)
    return v.reshape(n // 16, k // 16, 32, 8).contiguous()

def pj_p9_split_router(w):
    """FP32 router -> (hi, lo) bf16, exactly SGLang's MuseSpark1xRouter._build_split2."""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return (hi, lo)

def pj_p9_pack_hilo(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] bf16: per k16 step the hi fragment, then the lo one."""
    hi, lo = pj_p9_split_router(w)
    return torch.stack((pj_p9_pack(hi), pj_p9_pack(lo)), dim=2).contiguous()

def pj_a_tile_from_fragments(frag):
    """Inverse view used by the checks: [32, 8] bf16 (one 16x16 tile, all lanes) -> the 16x16 A tile,
    read with the PTX formulas (row = g + 8 (j & 1), col = 2 t + 8 (j >> 1) + p)."""
    a = torch.empty(16, 16, dtype=frag.dtype)
    for lane in range(32):
        g, t = (lane >> 2, lane & 3)
        for j in range(4):
            for p in range(2):
                a[g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + p] = frag[lane, 2 * j + p]
    return a
pj_ESC = 15
pj_M32 = 4294967295

def pj_choose_top(e_tile):
    """e_tile: int64 exponents of one 16-row tile -> top of the best 15-exponent window (>= 15)."""
    hist = torch.bincount(e_tile.flatten(), minlength=256).double()
    win = torch.nn.functional.conv1d(hist.view(1, 1, -1), torch.ones(1, 1, 15, dtype=torch.float64, device=hist.device)).flatten()
    start = int(win.argmax())
    return max(start + 14, 15)

def pj_p22_encode(frag, k_perm_idx):
    """frag: p9_pack output [RT, KS, 32, 8] bf16 (K % 64 == 0). Returns dict of int32 planes, tile tops,
    and the escape list in original (unpermuted) K, plus counts."""
    rt, ks = (frag.shape[0], frag.shape[1])
    assert ks % 4 == 0
    b = frag.contiguous().view(torch.int16).to(torch.int64) & 65535
    e = b >> 7 & 255
    r = b >> 15 << 7 | b & 127
    dev = frag.device
    tops = torch.tensor([pj_choose_top(e[i]) for i in range(rt)], dtype=torch.int64, device=dev)
    c = tops.view(rt, 1, 1, 1) - e
    esc = (c < 0) | (c > 14)
    c = torch.where(esc, torch.full_like(c, pj_ESC), c)
    r = torch.where(esc, torch.zeros_like(r), r)
    q = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        q |= c[..., i] << 8 * i
        q |= c[..., 4 + i] << 8 * i + 4
    ra = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    rb = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        ra |= r[..., i] << 8 * i
        rb |= r[..., 4 + i] << 8 * i
    chunks = ks // 4
    qp = q.view(rt, chunks, 4, 32).permute(0, 1, 3, 2)
    rr = torch.stack((ra, rb), dim=-1).view(rt, chunks, 2, 2, 32, 2)
    rp = rr.permute(0, 1, 2, 4, 3, 5).reshape(rt, chunks, 2, 32, 4)
    ti, si, li, ei = esc.nonzero(as_tuple=True)
    g, t = (li >> 2, li & 3)
    reg = ei >> 1
    row = 16 * ti + g + 8 * (reg & 1)
    col = 2 * t + 8 * (reg >> 1) + (ei & 1)
    korig = k_perm_idx.to(dev)[16 * si + col]
    true_bits = b[ti, si, li, ei]
    to32 = lambda v: (v & pj_M32).to(torch.int64)
    return {'q': to32(qp), 'r': to32(rp), 'tops': tops, 'esc_row': row, 'esc_k': korig, 'esc_bits': true_bits, 'n_esc': int(esc.sum()), 'n': int(esc.numel())}

def pj_prmt(a, b, sel):
    """PTX prmt.b32 default mode for selector nibbles 0..7: byte n = byte sel_n of {b:a}."""
    src = b << 32 | a
    out = torch.zeros_like(a)
    for n in range(4):
        idx = sel >> 4 * n & 7
        out |= (src >> 8 * idx & 255) << 8 * n
    return out

def pj_p22_decode_step(q, ra, rb, top):
    """The kernel's decode of one step for tensors of words (int64 holding uint32). Returns 4 registers."""
    top4 = top * 16843009 & pj_M32
    ca, cb = (q & 252645135, q >> 4 & 252645135)
    ea, eb = (top4 - ca & pj_M32, top4 - cb & pj_M32)
    hia = ra & 2155905152 | ea >> 1 & 2139062143
    loa = ea << 7 & 2155905152 | ra & 2139062143
    hib = rb & 2155905152 | eb >> 1 & 2139062143
    lob = eb << 7 & 2155905152 | rb & 2139062143
    return (pj_prmt(loa, hia, 20800), pj_prmt(loa, hia, 29538), pj_prmt(lob, hib, 20800), pj_prmt(lob, hib, 29538))

def pj_p22_decode_all(enc):
    """Decode every fragment from the planes exactly as the kernel indexes them -> [RT, KS, 32, 8] bits."""
    q, r, tops = (enc['q'], enc['r'], enc['tops'])
    rt, chunks = (q.shape[0], q.shape[1])
    out = torch.zeros(rt, chunks * 4, 32, 8, dtype=torch.int64)
    for j in range(4):
        qw = q[:, :, :, j]
        h, w0 = (j >> 1, 2 * (j & 1))
        ra, rb = (r[:, :, h, :, w0], r[:, :, h, :, w0 + 1])
        regs = pj_p22_decode_step(qw, ra, rb, tops.view(rt, 1, 1))
        for reg, v in enumerate(regs):
            out[:, j::4, :, 2 * reg] = v & 65535
            out[:, j::4, :, 2 * reg + 1] = v >> 16 & 65535
    return out

def pj_p22_dec_bits(top):
    """Bits an escape decodes to: exponent top - 15, sign 0, mantissa 0."""
    return top - 15 << 7
pj_P22_MAX_ESCAPE_RATE = 0.005

def pj_p22_device_planes(w, n_rows):
    """BF16 W [N, K] -> kernel tensors on W's device: q [RT, K/64, 32, 4] int32, r [RT, K/64, 2, 32, 4]
    int32, tops [RT] uint8, CSR erow [N+1] int32 / ecol int32 / ebits int16 (sorted by row), or None when
    escapes exceed P22_MAX_ESCAPE_RATE (that layer then stays on uncompressed kernels)."""
    n, k = w.shape
    enc = pj_p22_encode(pj_p9_pack(w), pj_k_perm(k))
    if enc['n_esc'] > pj_P22_MAX_ESCAPE_RATE * enc['n']:
        return None
    order = torch.argsort(enc['esc_row'] * k + enc['esc_k'])
    rows, cols, bits = (enc['esc_row'][order], enc['esc_k'][order], enc['esc_bits'][order])
    erow = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    erow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    as32 = lambda t: t.to(torch.int64).where(t < 2 ** 31, t - 2 ** 32).to(torch.int32).contiguous()
    return {'q': as32(enc['q']), 'r': as32(enc['r']), 'tops': enc['tops'].to(torch.uint8).contiguous(), 'erow': erow.to(torch.int32).contiguous(), 'ecol': cols.to(torch.int32).contiguous(), 'ebits': bits.where(bits < 2 ** 15, bits - 2 ** 16).to(torch.int16).contiguous(), 'rate': enc['n_esc'] / enc['n']}
pj_P9_SRC = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace p9 {\n\n__device__ __forceinline__ void ld_frag(const uint4* p, uint4& v) {\n  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"\n               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n}\n\n__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));\n}\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n\ntemplate <int CS>\n__device__ __forceinline__ void cluster_sync() {\n  if constexpr (CS > 1) {\n    asm volatile("barrier.cluster.arrive.release.aligned;\\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");\n  } else {\n    __syncthreads();\n  }\n}\n\ntemplate <int CS>\n__device__ __forceinline__ uint32_t cluster_rank() {\n  if constexpr (CS > 1) {\n    uint32_t r;\n    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));\n    return r;\n  } else {\n    return 0u;\n  }\n}\n\n__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {\n  uint32_t remote;\n  float v;\n  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));\n  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");\n  return v;\n}\n\n__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {\n  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n}\n\n__device__ __forceinline__ uint32_t word(const uint4& v, int i) {\n  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;\n}\n\n// P22 (F12) decode of one k16 step into the four A registers (proj/p9/p22_layout.py documents the\n// format; check_p22.py proves it exact on all 65,536 BF16 patterns with these same operations).\n__device__ __forceinline__ uint4 f12_decode(uint32_t q, uint32_t ra, uint32_t rb, uint32_t top4) {\n  const uint32_t ca = q & 0x0F0F0F0Fu, cb = (q >> 4) & 0x0F0F0F0Fu;\n  const uint32_t ea = top4 - ca, eb = top4 - cb;\n  const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x7F7F7F7Fu);\n  const uint32_t loa = ((ea << 7) & 0x80808080u) | (ra & 0x7F7F7F7Fu);\n  const uint32_t hib = (rb & 0x80808080u) | ((eb >> 1) & 0x7F7F7F7Fu);\n  const uint32_t lob = ((eb << 7) & 0x80808080u) | (rb & 0x7F7F7F7Fu);\n  return make_uint4(__byte_perm(loa, hia, 0x5140u), __byte_perm(loa, hia, 0x7362u),\n                    __byte_perm(lob, hib, 0x5140u), __byte_perm(lob, hib, 0x7362u));\n}\n\n// Shared memory: part[KSW][RT*16][8*NT] f32 only. K is permuted inside each 64-wide chunk (k_perm),\n// so lane (g, t)\'s B words for the chunk\'s 4 MMAs are x[row][64 c + 16 t .. + 16): two 16-B loads.\ntemplate <int NT, int RTW, int RG, int KSW, int D, int CS, bool HILO, bool OUT_F32, bool F12>\n__global__ void __launch_bounds__(RG * KSW * 32)\np9_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ qp, const uint4* __restrict__ rp,\n          const uint8_t* __restrict__ tops, const int* __restrict__ erow, const int* __restrict__ ecol,\n          const unsigned short* __restrict__ ebits, const uint4* __restrict__ x, void* __restrict__ out,\n          int m, int n_rows, int k) {\n  constexpr int RT = RG * RTW;          // 16-row tiles per CTA\n  constexpr int MT = 8 * NT;            // token rows covered\n  constexpr int AS = HILO ? 2 : 1;      // A fragments per tile per step\n  constexpr int SLOTS = F12 ? 3 : 4 * AS;   // uint4 per tile per chunk: Q + 2 R, or 4 steps x AS\n  static_assert(!(F12 && HILO), "the router stays hi + lo");\n  extern __shared__ __align__(16) unsigned char smem[];\n  float* part = reinterpret_cast<float*>(smem);\n  const int kc = k / CS;                // this CTA\'s K slice\n  const int kw = kc / KSW;              // one warp\'s K part\n  const int chunks = kw / 64;           // 64-wide chunks (4 MMA steps each)\n\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, t = lane & 3;\n  const int rg = warp / KSW, kpart = warp % KSW;\n  const uint32_t rank = cluster_rank<CS>();\n  const int rowblock = blockIdx.x / CS;\n  const int c0 = (static_cast<int>(rank) * kc + kpart * kw) / 64;   // first chunk of this warp\n  const int ksteps_total = k / 16;\n\n  const int chunks_total = k / 64;\n  const uint4* wbase[RTW];\n  const uint4* qbase[RTW];\n  const uint4* rbase[RTW];\n  uint32_t top4[RTW];\n#pragma unroll\n  for (int i = 0; i < RTW; ++i) {\n    const long long tile = static_cast<long long>(rowblock) * RT + rg * RTW + i;\n    if constexpr (F12) {\n      qbase[i] = qp + (tile * chunks_total + c0) * 32 + lane;            // Q plane [tile][chunk][lane]\n      rbase[i] = rp + (tile * chunks_total + c0) * 64 + lane;            // R plane [tile][chunk][2][lane]\n      top4[i] = static_cast<uint32_t>(tops[tile]) * 0x01010101u;\n    } else {\n      wbase[i] = wp + (tile * ksteps_total + 4LL * c0) * (32 * AS) + lane;\n    }\n  }\n  const uint4* xb[NT];\n  bool bok[NT];\n#pragma unroll\n  for (int n = 0; n < NT; ++n) {\n    const int row = 8 * n + g;\n    bok[n] = row < m;\n    xb[n] = x + static_cast<long long>(bok[n] ? row : 0) * (k / 8) + (64LL * c0 + 16 * t) / 8;\n  }\n\n  uint4 ra[D][RTW][SLOTS];\n  uint4 rb[D][NT][2];\n  auto load = [&](int slot, int cc) {\n#pragma unroll\n    for (int i = 0; i < RTW; ++i) {\n      if constexpr (F12) {\n        ld_frag(qbase[i] + 32LL * cc, ra[slot][i][0]);\n        ld_frag(rbase[i] + 64LL * cc, ra[slot][i][1]);\n        ld_frag(rbase[i] + 64LL * cc + 32, ra[slot][i][2]);\n      } else {\n#pragma unroll\n        for (int j = 0; j < 4; ++j)\n#pragma unroll\n          for (int h = 0; h < AS; ++h) ld_frag(wbase[i] + (4LL * cc + j) * (32 * AS) + h * 32, ra[slot][i][j * AS + h]);\n      }\n    }\n#pragma unroll\n    for (int n = 0; n < NT; ++n) {\n      if (bok[n]) {\n        ld_x(xb[n] + 8LL * cc, rb[slot][n][0]);\n        ld_x(xb[n] + 8LL * cc + 1, rb[slot][n][1]);\n      } else {\n        rb[slot][n][0] = make_uint4(0u, 0u, 0u, 0u);\n        rb[slot][n][1] = make_uint4(0u, 0u, 0u, 0u);\n      }\n    }\n  };\n#pragma unroll\n  for (int s = 0; s < D; ++s)\n    if (s < chunks) load(s, s);\n\n  float acc[RTW][NT][4];\n#pragma unroll\n  for (int i = 0; i < RTW; ++i)\n#pragma unroll\n    for (int n = 0; n < NT; ++n)\n#pragma unroll\n      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;\n\n  for (int cc0 = 0; cc0 < chunks; cc0 += D) {\n#pragma unroll\n    for (int s = 0; s < D; ++s) {\n      const int cc = cc0 + s;\n      if (cc < chunks) {\n#pragma unroll\n        for (int j = 0; j < 4; ++j) {\n#pragma unroll\n          for (int n = 0; n < NT; ++n) {\n            const uint32_t b0 = word(rb[s][n][j >> 1], (2 * j) & 3);\n            const uint32_t b1 = word(rb[s][n][j >> 1], (2 * j + 1) & 3);\n#pragma unroll\n            for (int i = 0; i < RTW; ++i) {\n              if constexpr (F12) {\n                const uint4& rv = ra[s][i][1 + (j >> 1)];\n                const uint4 a = f12_decode(word(ra[s][i][0], j), word(rv, 2 * (j & 1)), word(rv, 2 * (j & 1) + 1), top4[i]);\n                mma(acc[i][n], a, b0, b1);\n              } else {\n#pragma unroll\n                for (int h = 0; h < AS; ++h) mma(acc[i][n], ra[s][i][j * AS + h], b0, b1);\n              }\n            }\n          }\n        }\n        if (cc + D < chunks) load(s, cc + D);\n      }\n    }\n  }\n\n  // C fragment: row g + 8 (r >> 1) of the 16-row tile, token 8 n + 2 t + (r & 1).\n  constexpr int ROWS = RT * 16;\n#pragma unroll\n  for (int i = 0; i < RTW; ++i)\n#pragma unroll\n    for (int n = 0; n < NT; ++n)\n#pragma unroll\n      for (int r = 0; r < 4; ++r) {\n        const int row = (rg * RTW + i) * 16 + g + 8 * (r >> 1);\n        const int tok = 8 * n + 2 * t + (r & 1);\n        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];\n      }\n  cluster_sync<CS>();\n\n  // CTA rank r reduces rows [r * ROWS/CS, (r+1) * ROWS/CS) over peers and K parts in fixed order.\n  constexpr int SHARE = ROWS / CS;\n  const uint32_t part_addr = smem_u32(part);\n  const int m_out = m < MT ? m : MT;\n  for (int idx = tid; idx < SHARE * m_out; idx += blockDim.x) {\n    const int tok = idx / SHARE;\n    const int row = static_cast<int>(rank) * SHARE + idx % SHARE;\n    float sum = 0.0f;\n#pragma unroll\n    for (int peer = 0; peer < CS; ++peer)\n#pragma unroll\n      for (int kp = 0; kp < KSW; ++kp) {\n        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;\n        if constexpr (CS > 1) {\n          sum += ld_peer_f32(part_addr + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(peer));\n        } else {\n          sum += part[off];\n        }\n      }\n    const long long orow = static_cast<long long>(rowblock) * ROWS + row;\n    if constexpr (F12) {  // escapes: + x * w_true - x * w_dec (w_dec: exponent top - 15, sign/mantissa 0)\n      if (orow < n_rows) {\n        const unsigned short* xs = reinterpret_cast<const unsigned short*>(x) + static_cast<long long>(tok) * k;\n        const float wdec = __uint_as_float((static_cast<uint32_t>(tops[orow >> 4]) - 15u) << 23);\n        for (int e = erow[orow]; e < erow[orow + 1]; ++e) {\n          const float xv = __uint_as_float(static_cast<uint32_t>(xs[ecol[e]]) << 16);\n          sum = fmaf(xv, __uint_as_float(static_cast<uint32_t>(ebits[e]) << 16), sum);\n          sum = fmaf(-xv, wdec, sum);\n        }\n      }\n    }\n    if (orow < n_rows) {\n      if constexpr (OUT_F32) {\n        reinterpret_cast<float*>(out)[static_cast<long long>(tok) * n_rows + orow] = sum;\n      } else {\n        reinterpret_cast<__nv_bfloat16*>(out)[static_cast<long long>(tok) * n_rows + orow] = __float2bfloat16_rn(sum);\n      }\n    }\n  }\n  if constexpr (CS > 1) cluster_sync<CS>();  // peers may still read this CTA\'s part[]\n}\n\n}  // namespace p9\n\n@@INSTANCES@@\n'
pj_P9_INSTANCES = ((1, 1, 2, 8, 2, 8, True, True), (2, 1, 1, 8, 2, 8, True, True))
pj_P9_CHECK_INSTANCES = ((1, 1, 8, 1, 4, 4, False, False), (4, 2, 2, 1, 1, 1, False, False), (1, 1, 1, 8, 2, 8, True, True), (2, 1, 4, 2, 2, 2, False, False), (1, 2, 4, 2, 4, 4, False, False, True), (4, 1, 8, 1, 2, 1, False, False, True))

def pj_p9_smem_bytes(nt, rtw, rg, ksw, cs, k):
    return ksw * rg * rtw * 16 * 8 * nt * 4

def pj_p9_name(inst):
    nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
    f12 = inst[8] if len(inst) > 8 else False
    return f'p9_run_{nt}_{rtw}_{rg}_{ksw}_{d}_{cs}_{int(hilo)}_{int(f32)}' + ('_f12' if f12 else '')

def pj_p9_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
        f12 = inst[8] if len(inst) > 8 else False
        tmpl = f"{nt}, {rtw}, {rg}, {ksw}, {d}, {cs}, {('true' if hilo else 'false')}, {('true' if f32 else 'false')}, {('true' if f12 else 'false')}"
        wrappers.append(f'\nvoid {pj_p9_name(inst)}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow,\n                     torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out,\n                     int64_t m, int64_t n_rows, int64_t k, int64_t smem) {{\n  cudaStream_t stream = at::cuda::getCurrentCUDAStream();\n  const unsigned ctas = static_cast<unsigned>((n_rows / ({rg} * {rtw} * 16)) * {cs});\n  const uint4* wptr = reinterpret_cast<const uint4*>(wp.data_ptr());\n  const uint4* qptr = reinterpret_cast<const uint4*>(qp.data_ptr());\n  const uint4* rptr = reinterpret_cast<const uint4*>(rp.data_ptr());\n  const uint8_t* tptr = reinterpret_cast<const uint8_t*>(tops.data_ptr());\n  const int* erp = reinterpret_cast<const int*>(erow.data_ptr());\n  const int* ecp = reinterpret_cast<const int*>(ecol.data_ptr());\n  const unsigned short* ebp = reinterpret_cast<const unsigned short*>(ebits.data_ptr());\n  const uint4* xptr = reinterpret_cast<const uint4*>(x.data_ptr());\n  void* optr = out.data_ptr();\n  const int mi = static_cast<int>(m), ni = static_cast<int>(n_rows), ki = static_cast<int>(k);\n  if ({cs} == 1) {{\n    p9::p9_kernel<{tmpl}><<<ctas, {rg} * {ksw} * 32, static_cast<size_t>(smem), stream>>>(wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n  }} else {{\n    cudaLaunchConfig_t cfg = {{}};\n    cudaLaunchAttribute attr[1];\n    attr[0].id = cudaLaunchAttributeClusterDimension;\n    attr[0].val.clusterDim.x = {cs};\n    attr[0].val.clusterDim.y = 1;\n    attr[0].val.clusterDim.z = 1;\n    cfg.gridDim = dim3(ctas, 1, 1);\n    cfg.blockDim = dim3({rg} * {ksw} * 32, 1, 1);\n    cfg.dynamicSmemBytes = static_cast<size_t>(smem);\n    cfg.stream = stream;\n    cfg.attrs = attr;\n    cfg.numAttrs = 1;\n    const cudaError_t err = cudaLaunchKernelEx(&cfg, p9::p9_kernel<{tmpl}>, wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);\n    TORCH_CHECK(err == cudaSuccess, "p9 cluster launch failed: ", cudaGetErrorString(err));\n    C10_CUDA_KERNEL_LAUNCH_CHECK();\n  }}\n}}\nvoid {pj_p9_name(inst)}_attr(int64_t smem) {{\n  cudaFuncSetAttribute(p9::p9_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));\n}}\nint64_t {pj_p9_name(inst)}_info() {{\n  cudaFuncAttributes a;\n  cudaFuncGetAttributes(&a, p9::p9_kernel<{tmpl}>);\n  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);\n}}')
    return pj_P9_SRC.replace('@@INSTANCES@@', '\n'.join(wrappers))

def pj_p9_decls(instances):
    out = []
    for inst in instances:
        name = pj_p9_name(inst)
        out.append(f'void {name}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow, torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows, int64_t k, int64_t smem);')
        out.append(f'void {name}_attr(int64_t smem);')
        out.append(f'int64_t {name}_info();')
    return out

def pj_p9_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = pj_p9_decls(instances)
    return load_inline(name='okc_proj_p9', cpp_sources='\n'.join(decls), cuda_sources=pj_p9_source(instances), functions=[d.split('(')[0].split()[-1] for d in decls], extra_cuda_cflags=['-O3', '-gencode=arch=compute_90a,code=sm_90a', '-lineinfo'], build_directory=tempfile.mkdtemp(prefix='okc_proj_p9_'), verbose=False)
pj_P9_ENABLED = True
pj_P9_MOD = None
pj_P9_BF16_LAYERS = ('qkv_gate', 'o', 'pooler', 'unpooler')
pj_P9_MAX_SMEM = 227 * 1024

def pj_p9_setup(state):
    """Pack the weights into fragment order (lossless, same bytes; router as SGLang's hi + lo), build
    every P9 instance, allow the full shared memory, and launch each instance once on zeros so no
    compile or module load lands inside a capture."""
    global pj_P9_MOD
    state['p9'] = {name: pj_p9_pack(state[name]) for name in pj_P9_BF16_LAYERS}
    state['p9']['router'] = pj_p9_pack_hilo(state['router'])
    dev = state['router'].device
    state['p9_dummy'] = torch.zeros(16, dtype=torch.int32, device=dev)
    state['p22'] = {}
    if any((len(inst) > 8 and inst[8] for inst in pj_P9_INSTANCES)):
        for name in pj_P9_BF16_LAYERS:
            planes = pj_p22_device_planes(state[name], state[name].shape[0])
            if planes is not None:
                state['p22'][name] = planes
    pj_P9_MOD = pj_p9_build(pj_P9_INSTANCES)
    for inst in pj_P9_INSTANCES:
        getattr(pj_P9_MOD, pj_p9_name(inst) + '_attr')(pj_P9_MAX_SMEM)

def pj_run_p9(c, x, layer, n, state):
    """c: dict(kind="p9", NT, RTW, RG, KSW, D, CS[, F12])."""
    hilo = layer == 'router'
    f12 = bool(c.get('F12', False))
    inst = (c['NT'], c['RTW'], c['RG'], c['KSW'], c['D'], c['CS'], hilo, hilo) + ((True,) if f12 else ())
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32 if hilo else torch.bfloat16, device=x.device)
    smem = pj_p9_smem_bytes(c['NT'], c['RTW'], c['RG'], c['KSW'], c['CS'], k)
    dmy = state['p9_dummy']
    if f12:
        pl = state['p22'][layer]
        args = (dmy, pl['q'], pl['r'], pl['tops'], pl['erow'], pl['ecol'], pl['ebits'])
    else:
        args = (state['p9'][layer], dmy, dmy, dmy, dmy, dmy, dmy)
    getattr(pj_P9_MOD, pj_p9_name(inst))(*args, x.contiguous(), out, m, n, k, smem)
    return out
