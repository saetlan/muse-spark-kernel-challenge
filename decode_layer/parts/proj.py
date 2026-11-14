# Embedded: the bf16_proj session's scored v2 (proj/scored/submission_v2.py, sha256 da1b76b358072dee...;
# scored 1.192x on muse_bf16_proj, job aae716920c1d403f), every top-level name prefixed with 'pj_' by
# decode_layer/local/make_attn.py. pj_run_choice / pj_sglang_path compute our dense projections; pj_CHOICE is keyed by (layer, m).

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
pj_CHOICE = {('o', 1): {'kind': 'fma', 'BN': 4, 'split': 1, 'warps': 8}, ('pooler', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('unpooler', 1): {'kind': 'tiny', 'max_m': 16}, ('pooler', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 8): {'kind': 'fma', 'BN': 16, 'split': 64, 'warps': 8}, ('unpooler', 8): {'kind': 'tiny', 'max_m': 16}, ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('pooler', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4}, ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}, ('router', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 128): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('router', 256): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4}, ('qkv_gate', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}, ('router', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}}

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

def pj_run_choice(c, x, layer, n, state):
    if c['kind'] in ('swap', 'gemm'):
        return pj_run_triton(c, x, state[layer][:n], n)
    if c['kind'] == 'fma':
        return pj_run_fma(c, x, state[layer][:n], n)
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
        pj_run_choice(c, x, layer, n, state)
    torch.cuda.synchronize()

def pj_prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights['router']
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state['router_hi_t'] = hi.t().contiguous()
    state['router_lo_t'] = lo.t().contiguous()
    pj__warm(state)
    return state

def pj_kernel_fn(data, state):
    x, cfg = (data['x'], data['config'])
    layer, n = (cfg['layer'], cfg['n'])
    c = pj_CHOICE.get((layer, x.shape[0]))
    if c is None:
        return pj_sglang_path(x, layer, n, state)
    return pj_run_choice(c, x, layer, n, state)
