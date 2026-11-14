# okc: problem=muse_bf16_proj
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


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int      # output width; the unpooler's is 1024 up to 8 tokens (this GPU's slice), else 8192
    k: int
    layer: str  # which weight: "qkv_gate", "o", "pooler", "unpooler" or "router"


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


# MuseSpark1xRouter switches to the split BF16 GEMMs above this many tokens.
ROUTER_FP32_MAX_TOKENS = 8

# (layer, m) -> kernel config, from the PE1-PE3, PE6 and PE8-PE11 labs. Other shapes run
# SGLang's path.
CHOICE = {
    ('o', 1): {'kind': 'fma', 'BN': 4, 'split': 1, 'warps': 8},
    ('pooler', 1): {'kind': 'f12v', 'R': 4, 'RG': 4, 'KSW': 2, 'D': 1, 'fallback': {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}},
    ('unpooler', 1): {'kind': 'tiny', 'max_m': 16},
    ('pooler', 8): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 1, 'fallback': {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}},
    ('router', 8): {'kind': 'p9', 'NT': 1, 'RTW': 1, 'RG': 2, 'KSW': 8, 'D': 2, 'CS': 8},
    ('unpooler', 8): {'kind': 'fi_mm', 'backend': 'tinygemm'},
    ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('pooler', 16): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 2, 'fallback': {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4}},
    ('router', 16): {'kind': 'p9', 'NT': 2, 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'CS': 8},
    ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 32): {'kind': 'gemm', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 4},
    ('router', 128): {'kind': 'gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'mode': 1, 'warps': 8, 'stages': 3},
    ('router', 256): {'kind': 'gemm', 'BM': 256, 'BN': 64, 'BK': 64, 'split': 32, 'mode': 1, 'warps': 8, 'stages': 3},
    ('qkv_gate', 1): {'kind': 'f12v', 'R': 3, 'RG': 3, 'KSW': 2, 'D': 2, 'fallback': {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8}},
    ('router', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8},
    ('qkv_gate', 8): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 4, 'F12': True, 'NT': 1, 'fallback': {'kind': 'fi_mm', 'backend': 'auto'}},
    ('pooler', 128): {'kind': 'dg', 'native': True},
    ('qkv_gate', 128): {'kind': 'fi_mm', 'backend': 'cublaslt'},
    ('o', 8): {'kind': 'f12m', 'RTW': 2, 'RG': 1, 'KSW': 8, 'D': 1, 'F12': True, 'NT': 1, 'fallback': None},
    ('qkv_gate', 16): {'kind': 'f12m', 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 2, 'F12': True, 'NT': 2, 'fallback': None},
    ('unpooler', 16): {'kind': 'f12m', 'NT': 2, 'RTW': 1, 'RG': 1, 'KSW': 8, 'D': 8, 'F12': True, 'XD': 1, 'fallback': None},
}
# Set to True by an appended P9 section (proj/local/make_p9.py), whose p9_setup() packs the weights
# and builds the P9 kernels in prepare().
P9_ENABLED = False
# Set to True by an appended F12V section (proj/local/make_f12v.py), whose f12v_setup() packs the BF16
# weights into the 12-bit lossless format and builds the m = 1 GEMV kernels in prepare().
F12V_ENABLED = False
# Set to True by an appended F12M section (proj/local/make_f12m.py), whose f12m_setup() packs the BF16
# weights into the 12-bit format in mma fragment order and builds the m = 2..32 kernels in prepare().
F12M_ENABLED = False
# Set to True by an appended WG section (proj/local/make_wg.py), whose wg_setup() builds the m = 128/256
# wgmma kernels in prepare().
WG_ENABLED = False
# Set to True by an appended RG section (proj/local/make_rg.py): the router hi + lo mma.sync kernel.
RG_ENABLED = False
# Set to True by an appended SX section (proj/local/make_sx.py): the m = 32/128/256 12-bit mma.sync GEMM.
SX_ENABLED = False


@triton.jit(do_not_specialize=["M"])
def _swap_kernel(w_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr,
                 ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
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
        w = tl.load(w_ptrs + kk, mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        if MODE == 0:
            acc = tl.dot(w, xt, acc)
        elif MODE == 1:
            hi = w.to(tl.bfloat16)
            lo = (w - hi.to(tl.float32)).to(tl.bfloat16)
            acc = tl.dot(hi, xt, acc)
            acc = tl.dot(lo, xt, acc)
        else:
            acc = tl.dot(w, xt.to(tl.float32), acc, input_precision="ieee")
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)


@triton.jit(do_not_specialize=["M"])
def _gemm_kernel(x_ptr, w_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr,
                 ATOMIC: tl.constexpr, OUT_BF16: tl.constexpr):
    # Axis 0 walks M fastest, so the CTAs sharing a weight tile run together and share it in L2.
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
            acc = tl.dot(x.to(tl.float32), wt, acc, input_precision="ieee")
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    elif OUT_BF16:
        tl.store(out_ptrs, acc.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptrs, acc, mask=mask)


@triton.jit
def _fma_kernel(w_ptr, x_ptr, out_ptr, N, stride_wn, stride_xm, stride_om,
                M: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr,
                OUT_BF16: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = pid_k * BK + tl.arange(0, BK)
    n_mask = offs_n < N
    # The whole slice at once: every weight byte of this program is in flight together.
    w = tl.load(w_ptr + offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :], mask=n_mask[:, None],
                other=0.0, eviction_policy="evict_first").to(tl.float32)
    for j in tl.static_range(M):
        xj = tl.load(x_ptr + j * stride_xm + offs_k).to(tl.float32)
        r = tl.sum(w * xj[None, :], axis=1)
        o_ptrs = out_ptr + j * stride_om + offs_n
        if ATOMIC:
            tl.atomic_add(o_ptrs, r, mask=n_mask, sem="relaxed")
        elif OUT_BF16:
            tl.store(o_ptrs, r.to(tl.bfloat16), mask=n_mask)
        else:
            tl.store(o_ptrs, r, mask=n_mask)


def run_fma(c, x, w, n):
    """_fma_kernel per config c: BN rows per program, split-K, warps."""
    m, k = x.shape
    split = c["split"]
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    _fma_kernel[(triton.cdiv(n, c["BN"]), split)](
        w, x, out, n, w.stride(0), x.stride(0), out.stride(0),
        M=m, BN=c["BN"], BK=k // split, ATOMIC=atomic, OUT_BF16=not fp32_out, num_warps=c["warps"])
    if atomic and not fp32_out:
        out = out.to(torch.bfloat16)
    return out


def run_triton(c, x, w, n):
    """One of our kernels per config c: kind, BM, BN, BK, split, mode, warps, stages."""
    m, k = x.shape
    split = c["split"]
    fp32_out = w.dtype == torch.float32
    atomic = split > 1
    if atomic:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    else:
        out = torch.empty((m, n), dtype=torch.float32 if fp32_out else torch.bfloat16, device=x.device)
    kps = k // split
    bm, bn, bk = c["BM"], c["BN"], c["BK"]
    if c["kind"] == "swap":
        _swap_kernel[(triton.cdiv(n, bn), split)](
            w, x, out, m, n, kps, w.stride(0), x.stride(0), out.stride(0),
            BM=bm, BN=bn, BK=bk, MODE=c["mode"], ATOMIC=atomic, OUT_BF16=not fp32_out,
            num_warps=c["warps"], num_stages=c["stages"])
    else:
        _gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
            x, w, out, m, n, kps, x.stride(0), w.stride(0), out.stride(0),
            BM=bm, BN=bn, BK=bk, MODE=c["mode"], ATOMIC=atomic, OUT_BF16=not fp32_out,
            num_warps=c["warps"], num_stages=c["stages"])
    if atomic and not fp32_out:
        out = out.to(torch.bfloat16)
    return out


def run_library(c, x, w, n):
    """BF16 GEMMs from libraries in the judge image (FlashInfer 0.7.0.post1, DeepGEMM), each with
    its own handle or kernels; none changes torch's or cuBLAS's global settings."""
    kind = c["kind"]
    if kind == "fi_mm":
        # mm_bf16 takes b as (k, n) column-major: W.T. Tactics come from a prior
        # flashinfer.autotune() pass in prepare() (in-memory cache keyed by shape).
        import flashinfer
        return flashinfer.mm_bf16(x, w.T, out_dtype=torch.bfloat16, backend=c["backend"])
    out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=x.device)
    if kind == "fi_wsk":
        from flashinfer.gemm.kernels.dense_bf16_gemm_warp_splitk import WarpSplitKTactic, run_warp_splitk_dense
        return run_warp_splitk_dense(x, w.T, out, False, WarpSplitKTactic(*c["tactic"]))
    if kind == "fi_direct":
        from flashinfer.gemm.kernels.dense_bf16_gemm_direct import DirectTactic, run_direct_dense
        return run_direct_dense(x, w.T, out, False, DirectTactic(*c["tactic"]))
    if kind == "fi_tiny":
        from flashinfer.gemm.routergemm import tinygemm_bf16
        tinygemm_bf16(x, w, out)
        return out
    # DeepGEMM: native sm90 kernel with deterministic algorithms on, its own cuBLASLt call otherwise
    # (a host-side DeepGEMM setting, read at each call).
    import deep_gemm
    deep_gemm.use_deterministic_algorithms(c["native"])
    deep_gemm.bf16_gemm_nt(x, w, out)
    return out


def run_choice(c, x, layer, n, state):
    if c["kind"] in ("swap", "gemm"):
        return run_triton(c, x, state[layer][:n], n)
    if c["kind"] == "fma":
        return run_fma(c, x, state[layer][:n], n)
    if c["kind"] == "p9":
        return run_p9(c, x, layer, n, state)
    if c["kind"] == "f12v":
        return run_f12v(c, x, layer, n, state)
    if c["kind"] == "f12m":
        return run_f12m(c, x, layer, n, state)
    if c["kind"] == "wg":
        return run_wg(c, x, layer, n, state)
    if c["kind"] == "wgf":
        return run_wgf(c, x, layer, n, state)
    if c["kind"] == "rg":
        return run_rg(c, x, layer, n, state)
    if c["kind"] == "sx":
        return run_sx(c, x, layer, n, state)
    if c["kind"] in ("fi_mm", "fi_wsk", "fi_direct", "fi_tiny", "dg"):
        return run_library(c, x, state[layer][:n], n)
    if c["kind"] == "sgl_router":
        # SGLang's Muse Spark FP32 router kernel (T <= 8): FP32 x, weight as [K, N].
        from sglang.kernels.ops.muse_spark_v12.router_gemm import router_gemm
        return router_gemm(x.float(), state["router_t"])
    if c["kind"] == "gemv":
        from sglang.kernels.ops.gemm.hopper_bf16_gemv import hopper_bf16_gemv
        return hopper_bf16_gemv(x, state[layer][:n])
    if c["kind"] == "tiny":
        from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
        return tiny_gemm_bf16(x, state[layer][:n], max_m=c["max_m"])
    raise ValueError(f"unknown kernel kind {c['kind']}")


def sglang_path(x, layer, n, state):
    """Exactly baseline.py's kernel_fn."""
    if layer != "router":
        return _bf16_gemm_dispatch_impl(x, state[layer][:n], None)
    if x.shape[0] <= ROUTER_FP32_MAX_TOKENS:
        return F.linear(x.float(), state["router"])
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _warm(state):
    """Run every chosen kernel once on zeros, so compiling happens here and not in a capture."""
    for (layer, m), c in CHOICE.items():
        n = 1024 if layer == "unpooler" and m <= 8 else state[layer].shape[0]
        x = torch.zeros((m, state[layer].shape[1]), dtype=torch.bfloat16, device=state[layer].device)
        if c["kind"] == "fi_mm":  # FlashInfer picks its tactic for this shape now (synthetic input)
            import flashinfer
            with flashinfer.autotune(True):
                run_choice(c, x, layer, n, state)
        run_choice(c, x, layer, n, state)
    torch.cuda.synchronize()


def _submission_prepare(weights):
    state = {name: w.contiguous() for name, w in weights.items()}
    router = weights["router"]
    hi = router.to(torch.bfloat16)
    lo = (router - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_t"] = router.t().contiguous()  # [K, N] FP32 for SGLang's router kernel
    if F12V_ENABLED:
        f12v_setup(state)
    if F12M_ENABLED:
        f12m_setup(state)
    if WG_ENABLED:
        wg_setup(state)
    if RG_ENABLED:
        rg_setup(state)
    if SX_ENABLED:
        sx_setup(state)
    if P9_ENABLED:
        p9_setup(state)
    _warm(state)
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    c = CHOICE.get((layer, x.shape[0]))
    if c is None:
        return sglang_path(x, layer, n, state)
    return run_choice(c, x, layer, n, state)


# ====================================================================== P9 (proj/p9): swap-AB mma.sync kernel for m <= 32

import tempfile

def k_perm(k):
    """Column order of the packed weights: inside every 64-wide K chunk c, packed column 16 j + col
    (MMA step j = 0..3, A column col = 0..15) holds W column 64 c + 16 t + 4 j + 2 h + p, where
    col = 8 h + 2 t + p. Then lane (g, t)'s B values for the chunk's 4 MMAs are x[64 c + 16 t ..
    64 c + 16 t + 16): two 16-byte loads, b0 of step j = 32-bit word 2 j, b1 = word 2 j + 1."""
    idx = []
    for c in range(k // 64):
        for j in range(4):
            for col in range(16):
                h, t, p = col // 8, (col % 8) // 2, col % 2
                idx.append(64 * c + 16 * t + 4 * j + 2 * h + p)
    return torch.tensor(idx, dtype=torch.long)


def p9_pack(w):
    """W [N, K] bf16 -> [N/16, K/16, 32, 8] bf16 in A-fragment order (one lane's uint4 = 8 bf16),
    columns permuted by k_perm inside each 64-wide chunk."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0, (w.shape, w.dtype)
    w = w[:, k_perm(k).to(w.device)]
    # W[16 rt + 8 h + g][16 ks + 8 c + 2 t + p] -> [rt, ks, g, t, c, h, p]; register j = h + 2 c.
    v = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2)          # rt, h, g, ks, c, t, p
    v = v.permute(0, 3, 2, 5, 4, 1, 6)                      # rt, ks, g, t, c, h, p
    return v.reshape(n // 16, k // 16, 32, 8).contiguous()


def p9_split_router(w):
    """FP32 router -> (hi, lo) bf16, exactly SGLang's MuseSpark1xRouter._build_split2."""
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return hi, lo


def p9_pack_hilo(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] bf16: per k16 step the hi fragment, then the lo one."""
    hi, lo = p9_split_router(w)
    return torch.stack((p9_pack(hi), p9_pack(lo)), dim=2).contiguous()


def a_tile_from_fragments(frag):
    """Inverse view used by the checks: [32, 8] bf16 (one 16x16 tile, all lanes) -> the 16x16 A tile,
    read with the PTX formulas (row = g + 8 (j & 1), col = 2 t + 8 (j >> 1) + p)."""
    a = torch.empty(16, 16, dtype=frag.dtype)
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for j in range(4):
            for p in range(2):
                a[g + 8 * (j & 1), 2 * t + 8 * (j >> 1) + p] = frag[lane, 2 * j + p]
    return a


ESC = 15
M32 = 0xFFFFFFFF


def choose_top(e_tile):
    """e_tile: int64 exponents of one 16-row tile -> top of the best 15-exponent window (>= 15)."""
    hist = torch.bincount(e_tile.flatten(), minlength=256).double()
    win = torch.nn.functional.conv1d(hist.view(1, 1, -1), torch.ones(1, 1, 15, dtype=torch.float64, device=hist.device)).flatten()
    start = int(win.argmax())          # window [start, start + 14]
    return max(start + 14, 15)


def p22_encode(frag, k_perm_idx):
    """frag: p9_pack output [RT, KS, 32, 8] bf16 (K % 64 == 0). Returns dict of int32 planes, tile tops,
    and the escape list in original (unpermuted) K, plus counts."""
    rt, ks = frag.shape[0], frag.shape[1]
    assert ks % 4 == 0
    b = frag.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF            # [RT, KS, 32, 8]
    e = (b >> 7) & 0xFF
    r = ((b >> 15) << 7) | (b & 0x7F)
    dev = frag.device
    tops = torch.tensor([choose_top(e[i]) for i in range(rt)], dtype=torch.int64, device=dev)
    c = tops.view(rt, 1, 1, 1) - e
    esc = (c < 0) | (c > 14)
    c = torch.where(esc, torch.full_like(c, ESC), c)
    r = torch.where(esc, torch.zeros_like(r), r)
    # Q word per (tile, step, lane): low nibbles c0..c3, high nibbles c4..c7.
    q = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        q |= c[..., i] << (8 * i)
        q |= c[..., 4 + i] << (8 * i + 4)
    ra = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    rb = torch.zeros(rt, ks, 32, dtype=torch.int64, device=dev)
    for i in range(4):
        ra |= r[..., i] << (8 * i)
        rb |= r[..., 4 + i] << (8 * i)
    chunks = ks // 4
    # Q plane [RT, chunks, 32, 4]: word j = step 4 chunk + j.
    qp = q.view(rt, chunks, 4, 32).permute(0, 1, 3, 2)
    # R plane [RT, chunks, 2, 32, 4]: uint4 h = [RA(2h), RB(2h), RA(2h+1), RB(2h+1)].
    rr = torch.stack((ra, rb), dim=-1).view(rt, chunks, 2, 2, 32, 2)          # rt, ch, h, step-in-h, lane, A/B
    rp = rr.permute(0, 1, 2, 4, 3, 5).reshape(rt, chunks, 2, 32, 4)
    # Escape list (row, original k, true bits).
    ti, si, li, ei = esc.nonzero(as_tuple=True)
    g, t = li >> 2, li & 3
    reg = ei >> 1
    row = 16 * ti + g + 8 * (reg & 1)
    col = 2 * t + 8 * (reg >> 1) + (ei & 1)
    korig = k_perm_idx.to(dev)[16 * si + col]
    true_bits = b[ti, si, li, ei]
    to32 = lambda v: (v & M32).to(torch.int64)
    return {"q": to32(qp), "r": to32(rp), "tops": tops, "esc_row": row, "esc_k": korig, "esc_bits": true_bits,
            "n_esc": int(esc.sum()), "n": int(esc.numel())}


def prmt(a, b, sel):
    """PTX prmt.b32 default mode for selector nibbles 0..7: byte n = byte sel_n of {b:a}."""
    src = (b << 32) | a
    out = torch.zeros_like(a)
    for n in range(4):
        idx = (sel >> (4 * n)) & 0x7
        out |= ((src >> (8 * idx)) & 0xFF) << (8 * n)
    return out


def p22_decode_step(q, ra, rb, top):
    """The kernel's decode of one step for tensors of words (int64 holding uint32). Returns 4 registers."""
    top4 = (top * 0x01010101) & M32
    ca, cb = q & 0x0F0F0F0F, (q >> 4) & 0x0F0F0F0F
    ea, eb = (top4 - ca) & M32, (top4 - cb) & M32
    hia = (ra & 0x80808080) | ((ea >> 1) & 0x7F7F7F7F)
    loa = ((ea << 7) & 0x80808080) | (ra & 0x7F7F7F7F)
    hib = (rb & 0x80808080) | ((eb >> 1) & 0x7F7F7F7F)
    lob = ((eb << 7) & 0x80808080) | (rb & 0x7F7F7F7F)
    return prmt(loa, hia, 0x5140), prmt(loa, hia, 0x7362), prmt(lob, hib, 0x5140), prmt(lob, hib, 0x7362)


def p22_decode_all(enc):
    """Decode every fragment from the planes exactly as the kernel indexes them -> [RT, KS, 32, 8] bits."""
    q, r, tops = enc["q"], enc["r"], enc["tops"]
    rt, chunks = q.shape[0], q.shape[1]
    out = torch.zeros(rt, chunks * 4, 32, 8, dtype=torch.int64)
    for j in range(4):
        qw = q[:, :, :, j]                                   # Q uint4 word j
        h, w0 = j >> 1, 2 * (j & 1)
        ra, rb = r[:, :, h, :, w0], r[:, :, h, :, w0 + 1]    # R uint4 h, words w0 and w0 + 1
        regs = p22_decode_step(qw, ra, rb, tops.view(rt, 1, 1))
        for reg, v in enumerate(regs):
            out[:, j::4, :, 2 * reg] = v & 0xFFFF
            out[:, j::4, :, 2 * reg + 1] = (v >> 16) & 0xFFFF
    return out


def p22_dec_bits(top):
    """Bits an escape decodes to: exponent top - 15, sign 0, mantissa 0."""
    return (top - 15) << 7


P22_MAX_ESCAPE_RATE = 0.005


def p22_device_planes(w, n_rows):
    """BF16 W [N, K] -> kernel tensors on W's device: q [RT, K/64, 32, 4] int32, r [RT, K/64, 2, 32, 4]
    int32, tops [RT] uint8, CSR erow [N+1] int32 / ecol int32 / ebits int16 (sorted by row), or None when
    escapes exceed P22_MAX_ESCAPE_RATE (that layer then stays on uncompressed kernels)."""
    n, k = w.shape
    enc = p22_encode(p9_pack(w), k_perm(k))
    if enc["n_esc"] > P22_MAX_ESCAPE_RATE * enc["n"]:
        return None
    order = torch.argsort(enc["esc_row"] * k + enc["esc_k"])
    rows, cols, bits = enc["esc_row"][order], enc["esc_k"][order], enc["esc_bits"][order]
    erow = torch.zeros(n + 1, dtype=torch.int64, device=w.device)
    erow[1:] = torch.cumsum(torch.bincount(rows, minlength=n), 0)
    as32 = lambda t: t.to(torch.int64).where(t < 2**31, t - 2**32).to(torch.int32).contiguous()
    return {"q": as32(enc["q"]), "r": as32(enc["r"]), "tops": enc["tops"].to(torch.uint8).contiguous(),
            "erow": erow.to(torch.int32).contiguous(), "ecol": cols.to(torch.int32).contiguous(),
            "ebits": bits.where(bits < 2**15, bits - 2**16).to(torch.int16).contiguous(),
            "rate": enc["n_esc"] / enc["n"]}


P9_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace p9 {

__device__ __forceinline__ void ld_frag(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

template <int CS>
__device__ __forceinline__ void cluster_sync() {
  if constexpr (CS > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
    __syncthreads();
  }
}

template <int CS>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (CS > 1) {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
  } else {
    return 0u;
  }
}

__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// P22 (F12) decode of one k16 step into the four A registers (proj/p9/p22_layout.py documents the
// format; check_p22.py proves it exact on all 65,536 BF16 patterns with these same operations).
__device__ __forceinline__ uint4 f12_decode(uint32_t q, uint32_t ra, uint32_t rb, uint32_t top4) {
  const uint32_t ca = q & 0x0F0F0F0Fu, cb = (q >> 4) & 0x0F0F0F0Fu;
  const uint32_t ea = top4 - ca, eb = top4 - cb;
  const uint32_t hia = (ra & 0x80808080u) | ((ea >> 1) & 0x7F7F7F7Fu);
  const uint32_t loa = ((ea << 7) & 0x80808080u) | (ra & 0x7F7F7F7Fu);
  const uint32_t hib = (rb & 0x80808080u) | ((eb >> 1) & 0x7F7F7F7Fu);
  const uint32_t lob = ((eb << 7) & 0x80808080u) | (rb & 0x7F7F7F7Fu);
  return make_uint4(__byte_perm(loa, hia, 0x5140u), __byte_perm(loa, hia, 0x7362u),
                    __byte_perm(lob, hib, 0x5140u), __byte_perm(lob, hib, 0x7362u));
}

// Shared memory: part[KSW][RT*16][8*NT] f32 only. K is permuted inside each 64-wide chunk (k_perm),
// so lane (g, t)'s B words for the chunk's 4 MMAs are x[row][64 c + 16 t .. + 16): two 16-B loads.
template <int NT, int RTW, int RG, int KSW, int D, int CS, bool HILO, bool OUT_F32, bool F12>
__global__ void __launch_bounds__(RG * KSW * 32)
p9_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ qp, const uint4* __restrict__ rp,
          const uint8_t* __restrict__ tops, const int* __restrict__ erow, const int* __restrict__ ecol,
          const unsigned short* __restrict__ ebits, const uint4* __restrict__ x, void* __restrict__ out,
          int m, int n_rows, int k) {
  constexpr int RT = RG * RTW;          // 16-row tiles per CTA
  constexpr int MT = 8 * NT;            // token rows covered
  constexpr int AS = HILO ? 2 : 1;      // A fragments per tile per step
  constexpr int SLOTS = F12 ? 3 : 4 * AS;   // uint4 per tile per chunk: Q + 2 R, or 4 steps x AS
  static_assert(!(F12 && HILO), "the router stays hi + lo");
  extern __shared__ __align__(16) unsigned char smem[];
  float* part = reinterpret_cast<float*>(smem);
  const int kc = k / CS;                // this CTA's K slice
  const int kw = kc / KSW;              // one warp's K part
  const int chunks = kw / 64;           // 64-wide chunks (4 MMA steps each)

  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rg = warp / KSW, kpart = warp % KSW;
  const uint32_t rank = cluster_rank<CS>();
  const int rowblock = blockIdx.x / CS;
  const int c0 = (static_cast<int>(rank) * kc + kpart * kw) / 64;   // first chunk of this warp
  const int ksteps_total = k / 16;

  const int chunks_total = k / 64;
  const uint4* wbase[RTW];
  const uint4* qbase[RTW];
  const uint4* rbase[RTW];
  uint32_t top4[RTW];
#pragma unroll
  for (int i = 0; i < RTW; ++i) {
    const long long tile = static_cast<long long>(rowblock) * RT + rg * RTW + i;
    if constexpr (F12) {
      qbase[i] = qp + (tile * chunks_total + c0) * 32 + lane;            // Q plane [tile][chunk][lane]
      rbase[i] = rp + (tile * chunks_total + c0) * 64 + lane;            // R plane [tile][chunk][2][lane]
      top4[i] = static_cast<uint32_t>(tops[tile]) * 0x01010101u;
    } else {
      wbase[i] = wp + (tile * ksteps_total + 4LL * c0) * (32 * AS) + lane;
    }
  }
  const uint4* xb[NT];
  bool bok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int row = 8 * n + g;
    bok[n] = row < m;
    xb[n] = x + static_cast<long long>(bok[n] ? row : 0) * (k / 8) + (64LL * c0 + 16 * t) / 8;
  }

  uint4 ra[D][RTW][SLOTS];
  uint4 rb[D][NT][2];
  auto load = [&](int slot, int cc) {
#pragma unroll
    for (int i = 0; i < RTW; ++i) {
      if constexpr (F12) {
        ld_frag(qbase[i] + 32LL * cc, ra[slot][i][0]);
        ld_frag(rbase[i] + 64LL * cc, ra[slot][i][1]);
        ld_frag(rbase[i] + 64LL * cc + 32, ra[slot][i][2]);
      } else {
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int h = 0; h < AS; ++h) ld_frag(wbase[i] + (4LL * cc + j) * (32 * AS) + h * 32, ra[slot][i][j * AS + h]);
      }
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (bok[n]) {
        ld_x(xb[n] + 8LL * cc, rb[slot][n][0]);
        ld_x(xb[n] + 8LL * cc + 1, rb[slot][n][1]);
      } else {
        rb[slot][n][0] = make_uint4(0u, 0u, 0u, 0u);
        rb[slot][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s)
    if (s < chunks) load(s, s);

  float acc[RTW][NT][4];
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

  for (int cc0 = 0; cc0 < chunks; cc0 += D) {
#pragma unroll
    for (int s = 0; s < D; ++s) {
      const int cc = cc0 + s;
      if (cc < chunks) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
          for (int n = 0; n < NT; ++n) {
            const uint32_t b0 = word(rb[s][n][j >> 1], (2 * j) & 3);
            const uint32_t b1 = word(rb[s][n][j >> 1], (2 * j + 1) & 3);
#pragma unroll
            for (int i = 0; i < RTW; ++i) {
              if constexpr (F12) {
                const uint4& rv = ra[s][i][1 + (j >> 1)];
                const uint4 a = f12_decode(word(ra[s][i][0], j), word(rv, 2 * (j & 1)), word(rv, 2 * (j & 1) + 1), top4[i]);
                mma(acc[i][n], a, b0, b1);
              } else {
#pragma unroll
                for (int h = 0; h < AS; ++h) mma(acc[i][n], ra[s][i][j * AS + h], b0, b1);
              }
            }
          }
        }
        if (cc + D < chunks) load(s, cc + D);
      }
    }
  }

  // C fragment: row g + 8 (r >> 1) of the 16-row tile, token 8 n + 2 t + (r & 1).
  constexpr int ROWS = RT * 16;
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const int row = (rg * RTW + i) * 16 + g + 8 * (r >> 1);
        const int tok = 8 * n + 2 * t + (r & 1);
        part[(static_cast<size_t>(kpart) * ROWS + row) * MT + tok] = acc[i][n][r];
      }
  cluster_sync<CS>();

  // CTA rank r reduces rows [r * ROWS/CS, (r+1) * ROWS/CS) over peers and K parts in fixed order.
  constexpr int SHARE = ROWS / CS;
  const uint32_t part_addr = smem_u32(part);
  const int m_out = m < MT ? m : MT;
  for (int idx = tid; idx < SHARE * m_out; idx += blockDim.x) {
    const int tok = idx / SHARE;
    const int row = static_cast<int>(rank) * SHARE + idx % SHARE;
    float sum = 0.0f;
#pragma unroll
    for (int peer = 0; peer < CS; ++peer)
#pragma unroll
      for (int kp = 0; kp < KSW; ++kp) {
        const size_t off = (static_cast<size_t>(kp) * ROWS + row) * MT + tok;
        if constexpr (CS > 1) {
          sum += ld_peer_f32(part_addr + 4u * static_cast<uint32_t>(off), static_cast<uint32_t>(peer));
        } else {
          sum += part[off];
        }
      }
    const long long orow = static_cast<long long>(rowblock) * ROWS + row;
    if constexpr (F12) {  // escapes: + x * w_true - x * w_dec (w_dec: exponent top - 15, sign/mantissa 0)
      if (orow < n_rows) {
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(x) + static_cast<long long>(tok) * k;
        const float wdec = __uint_as_float((static_cast<uint32_t>(tops[orow >> 4]) - 15u) << 23);
        for (int e = erow[orow]; e < erow[orow + 1]; ++e) {
          const float xv = __uint_as_float(static_cast<uint32_t>(xs[ecol[e]]) << 16);
          sum = fmaf(xv, __uint_as_float(static_cast<uint32_t>(ebits[e]) << 16), sum);
          sum = fmaf(-xv, wdec, sum);
        }
      }
    }
    if (orow < n_rows) {
      if constexpr (OUT_F32) {
        reinterpret_cast<float*>(out)[static_cast<long long>(tok) * n_rows + orow] = sum;
      } else {
        reinterpret_cast<__nv_bfloat16*>(out)[static_cast<long long>(tok) * n_rows + orow] = __float2bfloat16_rn(sum);
      }
    }
  }
  if constexpr (CS > 1) cluster_sync<CS>();  // peers may still read this CTA's part[]
}

}  // namespace p9

@@INSTANCES@@
"""

# (NT, RTW, RG, KSW, D, CS, HILO, OUT_F32) instances compiled into the module.
P9_INSTANCES = ((1, 1, 2, 8, 2, 8, True, True), (2, 1, 1, 8, 2, 8, True, True))
# Instances rendered by the local clang syntax check (one per template path: CS 1 and > 1, HILO, F32).
P9_CHECK_INSTANCES = ((1, 1, 8, 1, 4, 4, False, False), (4, 2, 2, 1, 1, 1, False, False),
                      (1, 1, 1, 8, 2, 8, True, True), (2, 1, 4, 2, 2, 2, False, False),
                      (1, 2, 4, 2, 4, 4, False, False, True), (4, 1, 8, 1, 2, 1, False, False, True))


def p9_smem_bytes(nt, rtw, rg, ksw, cs, k):
    return ksw * rg * rtw * 16 * 8 * nt * 4  # part[KSW][RT*16][8*NT] f32


def p9_name(inst):
    nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
    f12 = inst[8] if len(inst) > 8 else False
    return f"p9_run_{nt}_{rtw}_{rg}_{ksw}_{d}_{cs}_{int(hilo)}_{int(f32)}" + ("_f12" if f12 else "")


def p9_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, d, cs, hilo, f32 = inst[:8]
        f12 = inst[8] if len(inst) > 8 else False
        tmpl = (f"{nt}, {rtw}, {rg}, {ksw}, {d}, {cs}, {'true' if hilo else 'false'}, {'true' if f32 else 'false'}, "
                f"{'true' if f12 else 'false'}")
        wrappers.append(f"""
void {p9_name(inst)}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow,
                     torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out,
                     int64_t m, int64_t n_rows, int64_t k, int64_t smem) {{
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const unsigned ctas = static_cast<unsigned>((n_rows / ({rg} * {rtw} * 16)) * {cs});
  const uint4* wptr = reinterpret_cast<const uint4*>(wp.data_ptr());
  const uint4* qptr = reinterpret_cast<const uint4*>(qp.data_ptr());
  const uint4* rptr = reinterpret_cast<const uint4*>(rp.data_ptr());
  const uint8_t* tptr = reinterpret_cast<const uint8_t*>(tops.data_ptr());
  const int* erp = reinterpret_cast<const int*>(erow.data_ptr());
  const int* ecp = reinterpret_cast<const int*>(ecol.data_ptr());
  const unsigned short* ebp = reinterpret_cast<const unsigned short*>(ebits.data_ptr());
  const uint4* xptr = reinterpret_cast<const uint4*>(x.data_ptr());
  void* optr = out.data_ptr();
  const int mi = static_cast<int>(m), ni = static_cast<int>(n_rows), ki = static_cast<int>(k);
  if ({cs} == 1) {{
    p9::p9_kernel<{tmpl}><<<ctas, {rg} * {ksw} * 32, static_cast<size_t>(smem), stream>>>(wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }} else {{
    cudaLaunchConfig_t cfg = {{}};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = {cs};
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(ctas, 1, 1);
    cfg.blockDim = dim3({rg} * {ksw} * 32, 1, 1);
    cfg.dynamicSmemBytes = static_cast<size_t>(smem);
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    const cudaError_t err = cudaLaunchKernelEx(&cfg, p9::p9_kernel<{tmpl}>, wptr, qptr, rptr, tptr, erp, ecp, ebp, xptr, optr, mi, ni, ki);
    TORCH_CHECK(err == cudaSuccess, "p9 cluster launch failed: ", cudaGetErrorString(err));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }}
}}
void {p9_name(inst)}_attr(int64_t smem) {{
  cudaFuncSetAttribute(p9::p9_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
}}
int64_t {p9_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, p9::p9_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return P9_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def p9_decls(instances):
    out = []
    for inst in instances:
        name = p9_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor qp, torch::Tensor rp, torch::Tensor tops, torch::Tensor erow, "
                   f"torch::Tensor ecol, torch::Tensor ebits, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows, "
                   f"int64_t k, int64_t smem);")
        out.append(f"void {name}_attr(int64_t smem);")
        out.append(f"int64_t {name}_info();")
    return out


def p9_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = p9_decls(instances)
    return load_inline(
        name="okc_proj_p9", cpp_sources="\n".join(decls), cuda_sources=p9_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_p9_"), verbose=False)

P9_ENABLED = True
P9_MOD = None
P9_BF16_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")
P9_MAX_SMEM = 227 * 1024


def p9_setup(state):
    """Pack the weights into fragment order (lossless, same bytes; router as SGLang's hi + lo), build
    every P9 instance, allow the full shared memory, and launch each instance once on zeros so no
    compile or module load lands inside a capture."""
    global P9_MOD
    state["p9"] = {name: p9_pack(state[name]) for name in P9_BF16_LAYERS}
    state["p9"]["router"] = p9_pack_hilo(state["router"])
    dev = state["router"].device
    state["p9_dummy"] = torch.zeros(16, dtype=torch.int32, device=dev)
    state["p22"] = {}
    if any(len(inst) > 8 and inst[8] for inst in P9_INSTANCES):
        for name in P9_BF16_LAYERS:
            planes = p22_device_planes(state[name], state[name].shape[0])
            if planes is not None:
                state["p22"][name] = planes
    P9_MOD = p9_build(P9_INSTANCES)
    for inst in P9_INSTANCES:
        getattr(P9_MOD, p9_name(inst) + "_attr")(P9_MAX_SMEM)


def run_p9(c, x, layer, n, state):
    """c: dict(kind="p9", NT, RTW, RG, KSW, D, CS[, F12])."""
    hilo = layer == "router"
    f12 = bool(c.get("F12", False))
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], c["D"], c["CS"], hilo, hilo) + ((True,) if f12 else ())
    m, k = x.shape
    out = torch.empty((m, n), dtype=torch.float32 if hilo else torch.bfloat16, device=x.device)
    smem = p9_smem_bytes(c["NT"], c["RTW"], c["RG"], c["KSW"], c["CS"], k)
    dmy = state["p9_dummy"]
    if f12:
        pl = state["p22"][layer]
        args = (dmy, pl["q"], pl["r"], pl["tops"], pl["erow"], pl["ecol"], pl["ebits"])
    else:
        args = (state["p9"][layer], dmy, dmy, dmy, dmy, dmy, dmy)
    getattr(P9_MOD, p9_name(inst))(*args, x.contiguous(), out, m, n, k, smem)
    return out


# ====================================================================== F12V (proj/f12v): 12-bit lossless weights, m = 1 GEMV

F12V_SLOTS = 7          # escapes per row in the record
F12V_KBLOCK = 1024      # k per lane-block: 32 lanes x 32 weights
M32 = 0xFFFFFFFF


def f12v_bases(e):
    """e: [N, K] int64 exponents -> [N] int64 base of the 16-exponent window holding the most weights
    (first such window), within [1, 239]."""
    n = e.shape[0]
    hist = torch.zeros(n, 256, dtype=torch.int64, device=e.device)
    hist.scatter_add_(1, e, torch.ones_like(e))
    pad = torch.zeros(n, 257, dtype=torch.int64, device=e.device)
    pad[:, 1:] = hist.cumsum(1)                       # pad[:, j] = count of e < j
    win = pad[:, 16:257] - pad[:, 0:241]              # win[:, b] = count of e in [b, b + 15], b = 0..240
    win[:, 0] = -1
    win[:, 240] = -1
    return win.argmax(1)


def f12v_fields(w):
    """BF16 W [N, K] -> int64 fields (s, c, m, esc, bits, base)."""
    b = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    s, e, m = (b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F
    base = f12v_bases(e)
    c = e - base[:, None]
    esc = (c < 0) | (c > 15)
    zero = torch.zeros_like(c)
    return (torch.where(esc, zero, s), torch.where(esc, zero, c), torch.where(esc, zero, m), esc, b, base)


def f12v_words(s, c, m):
    """Fields [..., 8] of one group of 8 weights -> (Q, L0, L1) int64 words."""
    h = c >> 1
    q = torch.zeros(s.shape[:-1], dtype=torch.int64, device=s.device)
    for i in range(4):
        q |= (s[..., i] << (8 * i + 7)) | (h[..., i] << (8 * i))
        j = i
        q |= (s[..., 4 + j] << (8 * j + 3)) | (h[..., 4 + j] << ((8 * j - 4) % 32))
    lb = ((c & 1) << 7) | m
    l0 = lb[..., 0] | (lb[..., 1] << 8) | (lb[..., 2] << 16) | (lb[..., 3] << 24)
    l1 = lb[..., 4] | (lb[..., 5] << 8) | (lb[..., 6] << 16) | (lb[..., 7] << 24)
    return q, l0, l1


def f12v_encode(w):
    """BF16 W [N, K] (K % 1024 == 0) -> dict: P int64 words [N, K/1024, 3, 32, 4], base [N], and the
    escapes (row, k, true bits) sorted by (row, k)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % F12V_KBLOCK == 0, (w.shape, w.dtype)
    s, c, m, esc, bits, base = f12v_fields(w)
    kb = k // F12V_KBLOCK
    shape = (n, kb, 32, 4, 8)                         # row, kb, lane, group, i
    q, l0, l1 = f12v_words(s.view(shape), c.view(shape), m.view(shape))   # [n, kb, 32, 4]
    slot0 = q
    slot1 = torch.stack((l0[..., 0], l1[..., 0], l0[..., 1], l1[..., 1]), dim=-1)
    slot2 = torch.stack((l0[..., 2], l1[..., 2], l0[..., 3], l1[..., 3]), dim=-1)
    p = torch.stack((slot0, slot1, slot2), dim=2)     # [n, kb, 3, 32, 4]
    rows, cols = esc.nonzero(as_tuple=True)           # row-major order: sorted by (row, k)
    return {"p": p, "base": base, "esc_row": rows, "esc_k": cols, "esc_bits": bits[rows, cols],
            "n_esc": int(esc.sum()), "n": n * k}


def as_int32(t):
    """int64 holding uint32 values -> int32 with the same bits."""
    t = t & M32
    return torch.where(t >= 2**31, t - 2**32, t).to(torch.int32)


def f12v_records(enc, n):
    """[N, 8] int64 record words, or None if a row has more than F12V_SLOTS escapes."""
    rows, cols, bits = enc["esc_row"], enc["esc_k"], enc["esc_bits"]
    dev = enc["base"].device
    count = torch.bincount(rows, minlength=n) if len(rows) else torch.zeros(n, dtype=torch.int64, device=dev)
    if len(rows) and int(count.max()) > F12V_SLOTS:
        return None
    rec = torch.zeros(n, 1 + F12V_SLOTS, dtype=torch.int64, device=dev)
    rec[:, 0] = enc["base"] | (count << 8)
    if len(rows):
        start = torch.zeros(n, dtype=torch.int64, device=dev)
        start[1:] = count.cumsum(0)[:-1]
        rank = torch.arange(len(rows), device=dev) - start[rows]
        rec[rows, 1 + rank] = cols | (bits << 16)
    return rec


def f12v_device_planes(w):
    """BF16 W [N, K] -> {"p": int32 [N, K/1024, 3, 32, 4], "rec": int32 [N, 8], "rate", "max_row"} on W's
    device, or None when a row has more than F12V_SLOTS escapes (that layer keeps its other kernels)."""
    n = w.shape[0]
    enc = f12v_encode(w)
    rec = f12v_records(enc, n)
    if rec is None:
        return None
    rows = enc["esc_row"]
    max_row = int(torch.bincount(rows, minlength=n).max()) if len(rows) else 0
    return {"p": as_int32(enc["p"]).contiguous(), "rec": as_int32(rec).contiguous(),
            "rate": enc["n_esc"] / enc["n"], "max_row": max_row}


F12V_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace f12v {

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// Index helpers (check_f12v.py evaluates their return expressions): row of warp row-group rg, row r
// in CTA bx; k block of warp part ks, step s; uint4 index of a weight slot, an x group, a record half;
// the slot and word holding group g's L words (h = 0: L0, h = 1: L1).
__device__ __forceinline__ long long f12v_row(long long bx, int rg, int r, int rb, int rr) { return bx * rb + rg * rr + r; }
__device__ __forceinline__ int f12v_kb(int ks, int s, int ns) { return ks * ns + s; }
__device__ __forceinline__ long long f12v_widx(long long row, int kbs, int kb, int slot, int lane) { return ((row * kbs + kb) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ int f12v_xidx(int kb, int lane, int g) { return kb * 128 + lane * 4 + g; }
__device__ __forceinline__ long long f12v_ridx(long long row, int h) { return row * 2 + h; }
__device__ __forceinline__ int f12v_lslot(int g) { return 1 + (g >> 1); }
__device__ __forceinline__ int f12v_lword(int g, int h) { return 2 * (g & 1) + h; }

// Decode of one group of 8 weights to FP32 scaled by 2^(112 - base) (format in f12v_layout.py).
__device__ __forceinline__ void f12v_decode8(uint32_t q, uint32_t l0, uint32_t l1, float (&f)[8]) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t h0 = (q & 0x00870087u) | 0x00380038u;
  const uint32_t h1 = (q & 0x87008700u) | 0x38003800u;
  const uint32_t h2 = (r4 & 0x00870087u) | 0x00380038u;
  const uint32_t h3 = (r4 & 0x87008700u) | 0x38003800u;
  f[0] = __uint_as_float(__byte_perm(l0, h0, 0x4055u));
  f[1] = __uint_as_float(__byte_perm(l0, h1, 0x5144u));
  f[2] = __uint_as_float(__byte_perm(l0, h0, 0x6277u));
  f[3] = __uint_as_float(__byte_perm(l0, h1, 0x7366u));
  f[4] = __uint_as_float(__byte_perm(l1, h2, 0x4055u));
  f[5] = __uint_as_float(__byte_perm(l1, h3, 0x5144u));
  f[6] = __uint_as_float(__byte_perm(l1, h2, 0x6277u));
  f[7] = __uint_as_float(__byte_perm(l1, h3, 0x7366u));
}

// Two BF16 x values (low half first) to FP32.
__device__ __forceinline__ void f12v_xpair(uint32_t w, float& lo, float& hi) {
  lo = __uint_as_float(w << 16);
  hi = __uint_as_float(w & 0xFFFF0000u);
}

// 2^(base - 112): undoes the decode's scale. 2^(base - 127): what an escape's stored pattern decodes to.
__device__ __forceinline__ float f12v_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12v_wdec(uint32_t base) { return __uint_as_float(base << 23); }

template <int R, int RG, int KSW, int NS, int D>
__global__ void __launch_bounds__(RG * KSW * 32)
f12v_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint4* __restrict__ x,
            __nv_bfloat16* __restrict__ out) {
  static_assert(D >= 1 && D <= NS && R >= 1 && R <= 32, "f12v config");
  constexpr int RB = R * RG;            // rows per CTA
  constexpr int KBS = NS * KSW;         // 1024-wide k blocks per row
  __shared__ float red[KSW][RB];
  __shared__ uint4 xs[KBS * 128];       // x (BF16), for the escapes
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int rg = warp / KSW, ks = warp % KSW;
  const long long bx = blockIdx.x;

  uint4 wv[D][R][3];
  uint4 xv[D][4];
  auto load = [&](int slot, int s) {
    const int kb = f12v_kb(ks, s, NS);
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const long long row = f12v_row(bx, rg, r, RB, R);
#pragma unroll
      for (int j = 0; j < 3; ++j) ld_w(wp + f12v_widx(row, KBS, kb, j, lane), wv[slot][r][j]);
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) ld_x(x + f12v_xidx(kb, lane, g), xv[slot][g]);
  };
#pragma unroll
  for (int s = 0; s < D; ++s) load(s, s);

  // Row records for the finishing threads: issued behind the first weight loads, used after the reduction.
  const bool fin = tid < RB;
  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);
  if (fin) {
    const long long row = f12v_row(bx, 0, tid, RB, R);
    ld_x(rec + f12v_ridx(row, 0), rc0);
    ld_x(rec + f12v_ridx(row, 1), rc1);
  }

  float acc[R][4];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int g = 0; g < 4; ++g) acc[r][g] = 0.0f;

#pragma unroll
  for (int s = 0; s < NS; ++s) {
    const int slot = s % D;
    if (rg == 0) {
#pragma unroll
      for (int g = 0; g < 4; ++g) xs[f12v_xidx(f12v_kb(ks, s, NS), lane, g)] = xv[slot][g];
    }
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      float xf[8];
#pragma unroll
      for (int t = 0; t < 4; ++t) f12v_xpair(word(xv[slot][g], t), xf[2 * t], xf[2 * t + 1]);
#pragma unroll
      for (int r = 0; r < R; ++r) {
        const uint4& lv = wv[slot][r][f12v_lslot(g)];
        float f[8];
        f12v_decode8(word(wv[slot][r][0], g), word(lv, f12v_lword(g, 0)), word(lv, f12v_lword(g, 1)), f);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][g] = fmaf(xf[i], f[i], acc[r][g]);
      }
    }
    if (s + D < NS) load(slot, s + D);
  }

#pragma unroll
  for (int r = 0; r < R; ++r) {
    float v = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    if (lane == 0) red[ks][rg * R + r] = v;
  }
  __syncthreads();
  if (fin) {
    float sum = 0.0f;
#pragma unroll
    for (int p = 0; p < KSW; ++p) sum += red[p][tid];
    const uint32_t base = rc0.x & 0xFFu, count = (rc0.x >> 8) & 0xFFu;
    sum *= f12v_scale(base);
    const float wdec = f12v_wdec(base);
    const uint32_t esc[7] = {rc0.y, rc0.z, rc0.w, rc1.x, rc1.y, rc1.z, rc1.w};
    const unsigned short* xh = reinterpret_cast<const unsigned short*>(xs);
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      if (j < count) {
        const float xk = __uint_as_float(static_cast<uint32_t>(xh[esc[j] & 0xFFFFu]) << 16);
        sum = fmaf(-xk, wdec, sum);
        sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);
      }
    }
    out[f12v_row(bx, 0, tid, RB, R)] = __float2bfloat16_rn(sum);
  }
}

}  // namespace f12v

@@INSTANCES@@
"""

# (R, RG, KSW, NS, D) instances compiled into the module.
F12V_INSTANCES = ((4, 4, 2, 4, 1), (3, 3, 2, 4, 2))
# Rendered by the local clang syntax check: ring depth 1, 2 and 4, one k block, several warps along K.
F12V_CHECK_INSTANCES = ((1, 4, 8, 1, 1), (2, 2, 2, 4, 2), (4, 4, 1, 8, 4), (2, 8, 1, 1, 1))


def f12v_name(inst):
    return "f12v_run_" + "_".join(str(int(v)) for v in inst)


def f12v_source(instances):
    wrappers = []
    for inst in instances:
        r, rg, ksw, ns, d = inst
        rb, kbs, threads = r * rg, ns * ksw, 32 * rg * ksw
        wrappers.append(f"""
void {f12v_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  TORCH_CHECK(n_rows > 0 && n_rows % {rb} == 0, "f12v: rows not a multiple of {rb}");
  TORCH_CHECK(x.numel() == {1024 * kbs} && out.numel() == n_rows, "f12v: x or out has the wrong size");
  TORCH_CHECK(wp.numel() >= n_rows * {kbs * 384} && rec.numel() >= n_rows * 8, "f12v: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "f12v: unaligned");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  f12v::f12v_kernel<{r}, {rg}, {ksw}, {ns}, {d}><<<static_cast<unsigned>(n_rows / {rb}), {threads}, 0, stream>>>(
      reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()),
      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
int64_t {f12v_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f12v::f12v_kernel<{r}, {rg}, {ksw}, {ns}, {d}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return F12V_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def f12v_decls(instances):
    out = []
    for inst in instances:
        name = f12v_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"int64_t {name}_info();")
    return out


def f12v_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12v_decls(instances)
    return load_inline(
        name="okc_proj_f12v", cpp_sources="\n".join(decls), cuda_sources=f12v_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_f12v_"), verbose=False)

F12V_ENABLED = True
F12V_MOD = None
F12V_LAYERS = ("qkv_gate", "o", "pooler", "unpooler")


def f12v_setup(state):
    """Pack the BF16 weights into the 12-bit format (lossless: per-row exponent window plus exact
    escapes; a layer with a row of more than 7 escapes is left out and its shapes keep their other
    kernels), then build every F12V instance."""
    global F12V_MOD
    state["f12v"] = {}
    for name in F12V_LAYERS:
        planes = f12v_device_planes(state[name])
        if planes is not None:
            state["f12v"][name] = planes
    try:
        F12V_MOD = f12v_build(F12V_INSTANCES)
    except Exception as exc:  # no module: every F12V shape runs its fallback (slower, still correct)
        print(f"f12v build FAILED, F12V shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        state["f12v"] = {}


def run_f12v(c, x, layer, n, state):
    """c: dict(kind="f12v", R, RG, KSW, D, fallback). m = 1 only; without planes for this layer (more
    than 7 escapes in a row) it runs c["fallback"], the kernel the shape had before, or SGLang's path."""
    pl = state["f12v"].get(layer)
    if pl is None or x.shape[0] != 1:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    k = x.shape[1]
    inst = (c["R"], c["RG"], c["KSW"], k // 1024 // c["KSW"], c["D"])
    x = x.contiguous()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12V_MOD, f12v_name(inst))(pl["p"], pl["rec"], x, out, n)
    return out


# ====================================================================== F12M (proj/f12m): 12-bit lossless weights, mma.sync at m = 2..32

def f12m_encode(w, natural=False):
    """BF16 W [N, K] (N % 16 == 0, K % 64 == 0) -> dict: P int64 words [RT, K/64, 3, 32, 4], base [N],
    escapes (row, original k, true bits) sorted by (row, k)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and n % 16 == 0 and k % 64 == 0, (w.shape, w.dtype)
    bits_w = w.contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    base = f12v_bases((bits_w >> 7) & 0xFF)                    # [N], from the unpermuted rows
    if natural:   # same fragment order without k_perm (B from shared memory reads natural k: WGF)
        frag = w.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 32, 8)
    else:
        frag = p9_pack(w)                                      # [RT, KS, 32, 8]
    rt_n, ks_n = frag.shape[0], frag.shape[1]
    fb = frag.view(torch.int16).to(torch.int64) & 0xFFFF
    dev = w.device
    lane = torch.arange(32, device=dev).view(1, 1, 32, 1)
    el = torch.arange(8, device=dev).view(1, 1, 1, 8)
    g, t, j = lane >> 2, lane & 3, el >> 1
    row = 16 * torch.arange(rt_n, device=dev).view(-1, 1, 1, 1) + g + 8 * (j & 1)      # [RT, 1, 32, 8]
    col = 16 * torch.arange(ks_n, device=dev).view(1, -1, 1, 1) + 2 * t + 8 * (j >> 1) + (el & 1)
    s, e, m = (fb >> 15) & 1, (fb >> 7) & 0xFF, fb & 0x7F
    c = e - base[row]
    esc = (c < 0) | (c > 15)
    zero = torch.zeros_like(c)
    s, c, m = torch.where(esc, zero, s), torch.where(esc, zero, c), torch.where(esc, zero, m)
    q, l0, l1 = f12v_words(s, c, m)                            # [RT, KS, 32]
    ch = ks_n // 4
    qv, l0v, l1v = (v.view(rt_n, ch, 4, 32) for v in (q, l0, l1))   # rt, chunk, step, lane
    slot0 = qv.permute(0, 1, 3, 2)
    slot1 = torch.stack((l0v[:, :, 0], l1v[:, :, 0], l0v[:, :, 1], l1v[:, :, 1]), dim=-1)
    slot2 = torch.stack((l0v[:, :, 2], l1v[:, :, 2], l0v[:, :, 3], l1v[:, :, 3]), dim=-1)
    p = torch.stack((slot0, slot1, slot2), dim=2)              # [RT, ch, 3, 32, 4]
    kp = torch.arange(k, device=dev) if natural else k_perm(k).to(dev)
    rows = row.expand_as(esc)[esc]
    ks_ = kp[col.expand_as(esc)[esc]]
    tbits = fb[esc]
    order = torch.argsort(rows * k + ks_)
    return {"p": p, "base": base, "esc_row": rows[order], "esc_k": ks_[order], "esc_bits": tbits[order],
            "n_esc": int(esc.sum()), "n": n * k}


def f12m_device_planes(w, natural=False):
    """BF16 W [N, K] -> {"p": int32 [RT, K/64, 3, 32, 4], "rec": int32 [N, 8], "rate", "max_row"}, or None
    when a row has more than 7 escapes."""
    n = w.shape[0]
    enc = f12m_encode(w, natural)
    rec = f12v_records(enc, n)
    if rec is None:
        return None
    rows = enc["esc_row"]
    max_row = int(torch.bincount(rows, minlength=n).max()) if len(rows) else 0
    return {"p": as_int32(enc["p"]).contiguous(), "rec": as_int32(rec).contiguous(),
            "rate": enc["n_esc"] / enc["n"], "max_row": max_row}


F12M_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace f12m {

__device__ __forceinline__ void ld_w(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void ld_x(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ uint32_t word(const uint4& v, int i) {
  return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int CS>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (CS > 1) {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
  } else {
    return 0u;
  }
}

__device__ __forceinline__ void cluster_sync_all() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

__device__ __forceinline__ float ld_peer_f32(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
}

// Index helpers (check_f12m.py evaluates their return expressions): tile i of warp row group rg in CTA
// bx; chunk of cluster rank rank, warp part ks, step s; uint4 index of an F12 slot, a BF16 fragment (P9 layout), an x
// half-chunk of token tok (k8 = K / 8); slot and word of step j's L words; C fragment row and token;
// record half.
__device__ __forceinline__ long long f12m_tile(long long bx, int rg, int i, int rt, int rtw) { return bx * rt + rg * rtw + i; }
__device__ __forceinline__ int f12m_chunk(int rank, int ks, int s, int ns, int ksw) { return (rank * ksw + ks) * ns + s; }
__device__ __forceinline__ long long f12m_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long f12m_bidx(long long tile, int kss, int c, int j, int lane) { return (tile * kss + 4 * c + j) * 32 + lane; }
__device__ __forceinline__ long long f12m_xidx(long long tok, int k8, int c, int t, int h) { return tok * k8 + 8 * c + 2 * t + h; }
__device__ __forceinline__ int f12m_lslot(int j) { return 1 + (j >> 1); }
__device__ __forceinline__ int f12m_lword(int j, int h) { return 2 * (j & 1) + h; }
__device__ __forceinline__ int f12m_crow(int rg, int rtw, int i, int g, int r) { return 16 * (rg * rtw + i) + g + 8 * (r >> 1); }
__device__ __forceinline__ int f12m_ctok(int n, int t, int r) { return 8 * n + 2 * t + (r & 1); }
__device__ __forceinline__ long long f12m_ridx(long long row, int h) { return row * 2 + h; }

// Decode of one lane's A fragment for one k16 step (format in f12m_layout.py): four BF16x2 registers
// scaled by 2^(112 - base) per row.
__device__ __forceinline__ uint4 f12m_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }

template <int NT, int RTW, int RG, int KSW, int NS, int D, bool F12, int CS, int XD>
__global__ void __launch_bounds__(RG * KSW * 32)
f12m_kernel(const uint4* __restrict__ wp, const uint4* __restrict__ rec, const uint4* __restrict__ x,
            __nv_bfloat16* __restrict__ out, int m) {
  static_assert(D >= 1 && D <= NS && NS % D == 0 && CS >= 1 && CS <= 8, "f12m ring depth or cluster");
  static_assert(XD >= 1 && XD <= D && D % XD == 0, "f12m x ring depth");
  constexpr int RT = RG * RTW;          // 16-row tiles per CTA
  constexpr int RB = 16 * RT;           // rows per CTA
  constexpr int MT = 8 * NT;            // token slots
  constexpr int CHS = NS * KSW * CS;    // 64-wide chunks per row
  constexpr int KSS = 4 * CHS;          // k16 steps per row
  constexpr int K8 = 8 * CHS;           // K / 8: uint4 per x row
  constexpr int SL = F12 ? 3 : 4;       // uint4 per tile per chunk
  __shared__ float part[KSW][RB][MT];
  __shared__ uint4 recs[F12 ? 2 * RB : 1];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int g = lane >> 2, t = lane & 3;
  const int rg = warp / KSW, ks = warp % KSW;
  const uint32_t rank = cluster_rank<CS>();
  const long long bx = blockIdx.x / CS;

  bool tok_ok[NT];
#pragma unroll
  for (int n = 0; n < NT; ++n) tok_ok[n] = 8 * n + g < m;
  uint4 wv[D][RTW][SL];
  uint4 xv[XD][NT][2];
  auto loadw = [&](int slot, int s) {
    const int c = f12m_chunk(static_cast<int>(rank), ks, s, NS, KSW);
#pragma unroll
    for (int i = 0; i < RTW; ++i) {
      const long long tile = f12m_tile(bx, rg, i, RT, RTW);
#pragma unroll
      for (int j = 0; j < SL; ++j) {
        if constexpr (F12) {
          ld_w(wp + f12m_widx(tile, CHS, c, j, lane), wv[slot][i][j]);
        } else {
          ld_w(wp + f12m_bidx(tile, KSS, c, j, lane), wv[slot][i][j]);
        }
      }
    }
  };
  auto loadx = [&](int xslot, int s) {
    const int c = f12m_chunk(static_cast<int>(rank), ks, s, NS, KSW);
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      if (tok_ok[n]) {
        ld_x(x + f12m_xidx(8 * n + g, K8, c, t, 0), xv[xslot][n][0]);
        ld_x(x + f12m_xidx(8 * n + g, K8, c, t, 1), xv[xslot][n][1]);
      } else {
        xv[xslot][n][0] = make_uint4(0u, 0u, 0u, 0u);
        xv[xslot][n][1] = make_uint4(0u, 0u, 0u, 0u);
      }
    }
  };
#pragma unroll
  for (int s = 0; s < D; ++s) {
    loadw(s, s);
    if (s < XD) loadx(s, s);
  }

  // Row records (F12): issued behind the first weight loads, kept in registers, written to shared
  // memory after the main loop.
  uint4 rc0 = make_uint4(0u, 0u, 0u, 0u), rc1 = make_uint4(0u, 0u, 0u, 0u);
  if (F12 && tid < RB) {
    ld_x(rec + f12m_ridx(bx * RB + tid, 0), rc0);
    ld_x(rec + f12m_ridx(bx * RB + tid, 1), rc1);
  }

  float acc[RTW][NT][4];
#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) acc[i][n][r] = 0.0f;

#pragma unroll 1
  for (int s0 = 0; s0 < NS; s0 += D) {
#pragma unroll
  for (int slot = 0; slot < D; ++slot) {
    const int s = s0 + slot;
    const int xs = slot % XD;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      uint4 a[RTW];
#pragma unroll
      for (int i = 0; i < RTW; ++i) {
        if constexpr (F12) {
          const uint4& lv = wv[slot][i][f12m_lslot(j)];
          a[i] = f12m_decode(word(wv[slot][i][0], j), word(lv, f12m_lword(j, 0)), word(lv, f12m_lword(j, 1)));
        } else {
          a[i] = wv[slot][i][j];
        }
      }
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        const uint32_t b0 = word(xv[xs][n][j >> 1], (2 * j) & 3);
        const uint32_t b1 = word(xv[xs][n][j >> 1], (2 * j + 1) & 3);
#pragma unroll
        for (int i = 0; i < RTW; ++i) mma(acc[i][n], a[i], b0, b1);
      }
    }
    if (s + D < NS) loadw(slot, s + D);
    if (s + XD < NS) loadx(xs, s + XD);
  }
  }

#pragma unroll
  for (int i = 0; i < RTW; ++i)
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
      for (int r = 0; r < 4; ++r) part[ks][f12m_crow(rg, RTW, i, g, r)][f12m_ctok(n, t, r)] = acc[i][n][r];
  if (F12 && tid < RB) {
    recs[2 * tid] = rc0;
    recs[2 * tid + 1] = rc1;
  }
  __syncthreads();

  const int mo = m < MT ? m : MT;
  const long long n_rows = static_cast<long long>(gridDim.x / CS) * RB;
  const unsigned short* xh = reinterpret_cast<const unsigned short*>(x);
  if constexpr (CS > 1) {
    // this CTA's KSW parts summed in place (one thread per item), then every rank's sum is visible
    for (int idx = tid; idx < RB * mo; idx += RG * KSW * 32) {
      const int tok = idx / RB, row = idx % RB;
      float own = 0.0f;
#pragma unroll
      for (int p = 0; p < KSW; ++p) own += part[p][row][tok];
      part[0][row][tok] = own;
    }
    cluster_sync_all();
  }
  for (int idx = tid * CS + static_cast<int>(rank); idx < RB * mo; idx += RG * KSW * 32 * CS) {
    const int tok = idx / RB, row = idx % RB;
    float sum = 0.0f;
    if constexpr (CS > 1) {
      const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(&part[0][row][tok]));
#pragma unroll
      for (int p = 0; p < CS; ++p) sum += ld_peer_f32(a, static_cast<uint32_t>(p));
    } else {
#pragma unroll
      for (int p = 0; p < KSW; ++p) sum += part[p][row][tok];
    }
    if constexpr (F12) {
      const uint4 q0 = recs[2 * row], q1 = recs[2 * row + 1];
      const uint32_t base = q0.x & 0xFFu, count = (q0.x >> 8) & 0xFFu;
      sum *= f12m_scale(base);
      const float wdec = f12m_wdec(base);
      const uint32_t esc[7] = {q0.y, q0.z, q0.w, q1.x, q1.y, q1.z, q1.w};
#pragma unroll
      for (int j = 0; j < 7; ++j) {
        if (j < count) {
          const float xk = __uint_as_float(static_cast<uint32_t>(xh[static_cast<long long>(tok) * (8 * K8) + (esc[j] & 0xFFFFu)]) << 16);
          sum = fmaf(-xk, wdec, sum);
          sum = fmaf(xk, __uint_as_float(esc[j] & 0xFFFF0000u), sum);
        }
      }
    }
    out[static_cast<long long>(tok) * n_rows + bx * RB + row] = __float2bfloat16_rn(sum);
  }
  if constexpr (CS > 1) cluster_sync_all();   // peers may still read this CTA's sums
}

}  // namespace f12m

@@INSTANCES@@
"""

# (NT, RTW, RG, KSW, NS, D, F12[, CS]) instances compiled into the module; CS (cluster split-K) defaults to 1.
F12M_INSTANCES = ((1, 1, 1, 8, 16, 2, True), (2, 1, 1, 8, 16, 2, True), (1, 1, 1, 8, 16, 4, True), (1, 2, 1, 8, 2, 1, True), (2, 1, 1, 8, 8, 8, True, 1, 1))
# Rendered by the local clang syntax check: F12 and BF16, ring depths 1/2/4, 1-4 token tiles, clusters of 2/4/8.
F12M_CHECK_INSTANCES = ((1, 2, 2, 4, 32, 2, True), (2, 1, 4, 2, 64, 4, True), (4, 2, 2, 2, 8, 1, True),
                        (1, 1, 4, 4, 32, 2, False), (4, 4, 1, 4, 4, 4, False), (2, 1, 1, 8, 8, 2, True, 2),
                        (1, 1, 1, 8, 2, 2, True, 4), (1, 2, 1, 4, 2, 2, True, 8), (2, 1, 1, 8, 16, 4, True, 1, 1),
                        (1, 1, 1, 8, 16, 8, True, 1, 2), (2, 2, 1, 8, 8, 4, True, 1, 2))


def f12m_unpack(inst):
    """(NT, RTW, RG, KSW, NS, D, F12, CS, XD) from a 7-, 8- or 9-tuple (CS defaults to 1, XD to D)."""
    inst = tuple(inst)
    if len(inst) == 7:
        inst += (1,)
    if len(inst) == 8:
        inst += (inst[5],)
    return inst


def f12m_name(inst):
    nt, rtw, rg, ksw, ns, d, f12, cs, xd = f12m_unpack(inst)
    return (f"f12m_run_{nt}_{rtw}_{rg}_{ksw}_{ns}_{d}_{int(f12)}" + (f"_c{cs}" if cs > 1 else "")
            + (f"_x{xd}" if xd != d else ""))


def f12m_smem(inst):
    nt, rtw, rg, ksw, ns, d, f12, cs, xd = f12m_unpack(inst)
    rb = 16 * rg * rtw
    return ksw * rb * 8 * nt * 4 + (2 * rb if f12 else 1) * 16


def f12m_launch(tmpl, rb, threads, cs):
    """Launch statement: <<<>>> for CS = 1 (unchanged), cudaLaunchKernelEx with a cluster of CS otherwise."""
    args = ("reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()),\n"
            "      reinterpret_cast<const uint4*>(x.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(m)")
    if cs == 1:
        return (f"f12m::f12m_kernel<{tmpl}><<<static_cast<unsigned>(n_rows / {rb}), {threads}, 0, stream>>>(\n      {args});")
    return f"""cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((n_rows / {rb}) * {cs}), 1, 1);
  cfg.blockDim = dim3({threads}, 1, 1);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, f12m::f12m_kernel<{tmpl}>, {args});
  TORCH_CHECK(err == cudaSuccess, "f12m cluster launch failed: ", cudaGetErrorString(err));"""


def f12m_source(instances):
    wrappers = []
    for inst in instances:
        nt, rtw, rg, ksw, ns, d, f12, cs, xd = f12m_unpack(inst)
        rb, chs, threads = 16 * rg * rtw, ns * ksw * cs, 32 * rg * ksw
        k = 64 * chs
        tmpl = f"{nt}, {rtw}, {rg}, {ksw}, {ns}, {d}, {'true' if f12 else 'false'}, {cs}, {xd}"
        tile_bytes = chs * 1536 if f12 else chs * 2048   # bytes per 16-row tile: 3 or 4 uint4 x 32 lanes per chunk
        wrappers.append(f"""
void {f12m_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows) {{
  TORCH_CHECK(n_rows > 0 && n_rows % {rb} == 0 && m >= 1 && m <= {8 * nt}, "f12m: rows or tokens out of range");
  TORCH_CHECK(x.numel() == m * {k} && out.numel() == m * n_rows, "f12m: x or out has the wrong size");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * {tile_bytes}LL, "f12m: weight planes too small");
  TORCH_CHECK(!{'true' if f12 else 'false'} || rec.numel() >= n_rows * 8, "f12m: records too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "f12m: unaligned");
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  {f12m_launch(tmpl, rb, threads, cs)}
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
int64_t {f12m_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, f12m::f12m_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return F12M_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def f12m_decls(instances):
    out = []
    for inst in instances:
        name = f12m_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t m, int64_t n_rows);")
        out.append(f"int64_t {name}_info();")
    return out


def f12m_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = f12m_decls(instances)
    return load_inline(
        name="okc_proj_f12m", cpp_sources="\n".join(decls), cuda_sources=f12m_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_f12m_"), verbose=False)

F12M_ENABLED = True
F12M_MOD = None


def f12m_setup(state):
    """Pack the BF16 weights into F12M planes (the F12V format in mma A-fragment order; a layer with a
    row of more than 7 escapes is left out), then build every F12M instance. A failed build leaves no
    module, so every F12M shape runs its fallback (slower, still correct)."""
    global F12M_MOD
    state["f12m"] = {}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        planes = f12m_device_planes(state[name])
        if planes is not None:
            state["f12m"][name] = planes
    try:
        F12M_MOD = f12m_build(F12M_INSTANCES)
    except Exception as exc:
        print(f"f12m build FAILED, F12M shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        F12M_MOD = None
        state["f12m"] = {}


def run_f12m(c, x, layer, n, state):
    """c: dict(kind="f12m", NT, RTW, RG, KSW, D, F12[, CS][, XD], fallback). F12 = False runs the same kernel on
    P9's BF16 fragments; CS > 1 splits K across a cluster of CS CTAs; XD < D keeps a shallower x ring. Without a module, planes or fragments
    for this layer, or with more tokens than 8 NT, it runs c["fallback"] (the shape's previous kernel) or
    SGLang's path."""
    m, k = x.shape
    pl = state.get("f12m", {}).get(layer)
    wp = pl["p"] if c["F12"] and pl is not None else (state.get("p9", {}).get(layer) if not c["F12"] else None)
    if F12M_MOD is None or wp is None or m > 8 * c["NT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    rec = pl["rec"] if c["F12"] else state["p9_dummy"]
    cs, xd = c.get("CS", 1), c.get("XD", c["D"])
    inst = (c["NT"], c["RTW"], c["RG"], c["KSW"], k // 64 // c["KSW"] // cs, c["D"], c["F12"])
    inst += (cs, xd) if xd != c["D"] else ((cs,) if cs > 1 else ())
    x = x.contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(F12M_MOD, f12m_name(inst))(wp, rec, x, out, m, n)
    return out


# ====================================================================== HL (proj/lab/e18.py): router with pre-split BF16 hi/lo in Triton

@triton.jit(do_not_specialize=["M"])
def _hl_gemm_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr):
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
    w_off = offs_n[None, :].to(tl.int64) * stride_wn + offs_k[:, None]
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        x = tl.load(x_ptrs + kk, mask=m_mask[:, None], other=0.0)
        hi = tl.load(hi_ptr + w_off + kk, mask=n_mask[None, :], other=0.0, eviction_policy="evict_first")
        lo = tl.load(lo_ptr + w_off + kk, mask=n_mask[None, :], other=0.0, eviction_policy="evict_first")
        acc = tl.dot(x, hi, acc)
        acc = tl.dot(x, lo, acc)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    else:
        tl.store(out_ptrs, acc, mask=mask)


@triton.jit(do_not_specialize=["M"])
def _hl_swap_kernel(hi_ptr, lo_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ATOMIC: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_m = tl.arange(0, BM)
    offs_k = tl.arange(0, BK)
    n_mask = offs_n < N
    m_mask = offs_m < M
    k0 = pid_k * k_per_split
    w_off = offs_n[:, None].to(tl.int64) * stride_wn + offs_k[None, :]
    x_ptrs = x_ptr + offs_m[None, :] * stride_xm + offs_k[:, None]
    acc = tl.zeros([BN, BM], dtype=tl.float32)
    for kk in range(k0, k0 + k_per_split, BK):
        hi = tl.load(hi_ptr + w_off + kk, mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        lo = tl.load(lo_ptr + w_off + kk, mask=n_mask[:, None], other=0.0, eviction_policy="evict_first")
        xt = tl.load(x_ptrs + kk, mask=m_mask[None, :], other=0.0)
        acc = tl.dot(hi, xt, acc)
        acc = tl.dot(lo, xt, acc)
    out_ptrs = out_ptr + offs_m[None, :] * stride_om + offs_n[:, None]
    mask = n_mask[:, None] & m_mask[None, :]
    if ATOMIC:
        tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")
    else:
        tl.store(out_ptrs, acc, mask=mask)


def hl_setup(state):
    """The router's BF16 hi + lo, contiguous [N, K] (SGLang's split; state keeps the transposed copies too)."""
    router = state["router"]
    hi = router.to(torch.bfloat16)
    state["router_hi"] = hi.contiguous()
    state["router_lo"] = (router - hi.float()).to(torch.bfloat16).contiguous()


def run_hl(c, x, layer, n, state):
    """c: dict(kind="hl_gemm" | "hl_swap", BM, BN, BK, split, warps, stages). FP32 out; split-K adds with FP32
    atomics into zeros. Without the hi/lo copies (or for another layer) it runs SGLang's path."""
    hi, lo = state.get("router_hi"), state.get("router_lo")
    m, k = x.shape
    if layer != "router" or hi is None or (c["kind"] == "hl_swap" and m > c["BM"]):
        return sglang_path(x, layer, n, state)
    split = c["split"]
    atomic = split > 1
    out = (torch.zeros if atomic else torch.empty)((m, n), dtype=torch.float32, device=x.device)
    kps = k // split
    x = x.contiguous()
    if c["kind"] == "hl_swap":
        _hl_swap_kernel[(triton.cdiv(n, c["BN"]), split)](
            hi, lo, x, out, m, n, kps, hi.stride(0), x.stride(0), out.stride(0),
            BM=c["BM"], BN=c["BN"], BK=c["BK"], ATOMIC=atomic, num_warps=c["warps"], num_stages=c["stages"])
    else:
        _hl_gemm_kernel[(triton.cdiv(m, c["BM"]), triton.cdiv(n, c["BN"]), split)](
            x, hi, lo, out, m, n, kps, x.stride(0), hi.stride(0), out.stride(0),
            BM=c["BM"], BN=c["BN"], BK=c["BK"], ATOMIC=atomic, num_warps=c["warps"], num_stages=c["stages"])
    return out


_run_choice_base = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] in ("hl_gemm", "hl_swap"):
        return run_hl(c, x, layer, n, state)
    return _run_choice_base(c, x, layer, n, state)


HL_CHOICE = {('router', 16): {'kind': 'hl_gemm', 'BM': 16, 'BN': 32, 'BK': 128, 'split': 32, 'warps': 2, 'stages': 3}, ('router', 32): {'kind': 'hl_swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'warps': 4, 'stages': 4}, ('router', 128): {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 128, 'split': 16, 'warps': 8, 'stages': 3}, ('router', 256): {'kind': 'hl_gemm', 'BM': 128, 'BN': 32, 'BK': 64, 'split': 16, 'warps': 8, 'stages': 3}}


for _key, _c in HL_CHOICE.items():
    CHOICE[_key] = dict(_c)

_warm_before_hl = _warm


def _warm(state):
    """prepare() ends with _warm(state): the router's hi/lo copies first, then every chosen kernel once, so the hl
    kernels compile here, before any capture (prepare itself stays the base's, untouched)."""
    hl_setup(state)
    _warm_before_hl(state)


# ====================================================================== lab (common): check runs only

"""Lab helpers shared by every check-run lab file (appended after submission.py by make_lab.py).
Everything here runs on synthetic inputs inside prepare(); none of it touches the timed calls."""

import statistics
import time

LAB_FLUSH_BYTES = 512 << 20  # larger than the H200's 60 MB L2, like the judge's cold-L2 read


def log(msg):
    print(f"[lab] {msg}", flush=True)


def _error_text(exc, keep=500):
    """The end of an exception's message, where Triton puts the actual reason."""
    text = " ".join(str(exc).split())
    return text if len(text) <= keep else "... " + text[-keep:]


class Lab:
    def __init__(self, budget_s):
        self.t0 = time.monotonic()
        self.budget_s = budget_s
        self.flush = None

    def left(self):
        return self.budget_s - (time.monotonic() - self.t0)

    def section(self, name, fn, *args, min_left=5.0):
        if self.left() < min_left:
            log(f"{name}: skipped, {self.left():.0f}s of budget left")
            return
        t = time.monotonic()
        try:
            fn(self, *args)
        except Exception as exc:  # a lab failure must never fail the run
            log(f"{name}: FAILED {type(exc).__name__}: {_error_text(exc)}")
        torch.cuda.synchronize()
        log(f"{name}: {time.monotonic() - t:.1f}s")

    def cold_l2(self):
        if self.flush is None:
            self.flush = torch.empty(LAB_FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
        return self.flush.sum()

    def time_graph(self, fn, reps=20, cold=True):
        """Median microseconds of a CUDA graph replay of fn, each after an L2 flush if cold."""
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
            if cold:
                self.cold_l2()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000.0)
        del graph
        return statistics.median(times)


# ====================================================================== lab (_dg12_section): check runs only

import os

def dg12_pack(w, bn_list=(32, 48, 64, 80, 96, 112, 128, 256)):
    """BF16 W [N, K] (K % 64 == 0) -> dict(q int32 [N, K/8], l int32 [N, K/8, 2], scale f32 [N],
    esc: {BN: (offsets int32 [N/BN + 1], entries int32 [E, 4] = (n_local, k, f32 bits of w_true scaled, 0))}),
    or None when an escape's scaled value is not an exact normal FP32 (never for real weights)."""
    n, k = w.shape
    assert w.dtype == torch.bfloat16 and k % 64 == 0, (w.shape, w.dtype)
    s, c, m, esc, bits, base = f12v_fields(w)
    shape = (n, k // 8, 8)
    q, l0, l1 = f12v_words(s.view(shape), c.view(shape), m.view(shape))     # [n, k/8] int64
    rows, cols = esc.nonzero(as_tuple=True)
    true = bits[rows, cols]
    e_true = (true >> 7) & 0xFF
    scaled_e = e_true - base[rows] + 112
    zero = (true & 0x7FFF) == 0
    if bool(((~zero) & ((e_true == 0) | (scaled_e < 1) | (scaled_e > 254))).any()):
        return None
    val = torch.where(zero, true & 0x8000, ((true >> 15) << 15) | (scaled_e.clamp(1, 254) << 7) | (true & 0x7F))
    f32 = (val << 16) & 0xFFFFFFFF                                          # BF16 bits -> FP32 bits
    q32 = as_int32(q).contiguous()
    l32 = as_int32(torch.stack((l0, l1), dim=-1)).contiguous()
    out = {"q": q32, "l": l32,
           # the kernel's planes, chunk-major (PF42): [K/64][N][8 units] (32 B per row) and [K/64][N][8][2] (64 B per row),
           # so a stage's BN rows are one contiguous block
           "qc": q32.view(n, k // 64, 8).permute(1, 0, 2).contiguous(),
           "lc": l32.view(n, k // 64, 8, 2).permute(1, 0, 2, 3).contiguous(),
           "scale": torch.pow(2.0, (base - 112).double()).float().contiguous(), "esc": {}, "n_esc": int(rows.numel()),
           "base": base}
    order = torch.argsort(rows * k + cols)
    rows, cols, f32 = rows[order], cols[order], f32[order]
    # the kernel's table: entries (global row, k, f32 bits, 0) sorted by (row, k), offsets per 8-row group (= the
    # epilogue's n8 accumulator group): entries of group g are ent8[offs8[g]:offs8[g + 1]]
    g8 = rows // 8
    counts8 = torch.bincount(g8, minlength=n // 8)
    offs8 = torch.zeros(n // 8 + 1, dtype=torch.int64, device=w.device)
    offs8[1:] = counts8.cumsum(0)
    out["offs8"] = as_int32(offs8).contiguous()
    out["ent8"] = as_int32(torch.stack((rows, cols, f32, torch.zeros_like(cols)), dim=-1)).contiguous().view(-1, 4)
    # per-BN tables for the kernel's epilogue (PF45): entries of n-block b and lane class t = ((row % 8) >> 1) are
    # et[bn][1][offs[4 b + t]:offs[4 b + t + 1]] = (row in block, k, f32 bits, 0), sorted by (block, t, row, k)
    out["et"] = {}
    for bn in bn_list:
        if n % bn:
            continue
        blk = rows // bn
        rib = rows - blk * bn
        tcls = (rib & 7) >> 1
        key = blk * 4 + tcls
        order_t = torch.argsort(key * (n * k) + rib * k + cols)
        cnt = torch.bincount(key[order_t], minlength=(n // bn) * 4)
        offs_t = torch.zeros((n // bn) * 4 + 1, dtype=torch.int64, device=w.device)
        offs_t[1:] = cnt.cumsum(0)
        ent_t = torch.stack((rib[order_t], cols[order_t], f32[order_t], torch.zeros_like(cols)), dim=-1)
        out["et"][bn] = (as_int32(offs_t).contiguous(), as_int32(ent_t).contiguous().view(-1, 4))
    for bn in bn_list:
        if n % bn:
            continue
        blk = rows // bn
        counts = torch.bincount(blk, minlength=n // bn)
        offs = torch.zeros(n // bn + 1, dtype=torch.int64, device=w.device)
        offs[1:] = counts.cumsum(0)
        ent = torch.stack((rows - blk * bn, cols, f32, torch.zeros_like(cols)), dim=-1)
        out["esc"][bn] = (as_int32(offs).contiguous(), as_int32(ent).contiguous().view(-1, 4))
    return out


def dg12_unpack_reference(pack, n, k, decode):
    """FP64 W [N, K] rebuilt from the planes with a decode function (parsed from the CUDA text by the checks):
    placeholder 2^-15 at escapes in the scaled domain, then the escape corrections, then the per-row scale."""
    q = pack["q"].to(torch.int64) & 0xFFFFFFFF
    l = pack["l"].to(torch.int64) & 0xFFFFFFFF
    regs = decode(q, l[..., 0], l[..., 1])                                 # 4 x [n, k/8] uint32
    el = torch.stack([(regs[e >> 1].to(torch.int64) >> (16 * (e & 1))) & 0xFFFF for e in range(8)], dim=-1)
    f = ((el.view(n, k) << 16) & 0xFFFFFFFF).to(torch.int32).view(torch.float32).double()
    return f


DG12_CU = r"""
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <stdint.h>
#include <deep_gemm/impls/sm90_bf16_gemm.cuh>

namespace deep_gemm {

// The F12 7-op decode: 3 words -> 8 BF16 (element e = k0 + e), scaled by 2^(112 - base).
__device__ __forceinline__ uint4 dg12_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

// Index helpers (check_dg12.py evaluates their return expressions): byte offsets of (row r, 8-k unit u) in a raw Q
// stage, a raw L stage and the SW128 B tile; the decoder's (row, unit) for pass row r0 and lane; the accumulator index
// of n8 group j, row half h, column parity c in wave l; the row in the block of wave l, warpgroup g, warp w, lane row
// gr, half h.
__device__ __forceinline__ uint32_t dg12_smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t dg12_lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.u32 %0, [%1];" : "=r"(v) : "r"(a));
  return v;
}
__device__ __forceinline__ void dg12_lds64(uint32_t a, uint32_t& x, uint32_t& y) {
  asm volatile("ld.shared.v2.u32 {%0, %1}, [%2];" : "=r"(x), "=r"(y) : "r"(a));
}
__device__ __forceinline__ void dg12_sts128(uint32_t a, const uint4& v) {
  asm volatile("st.shared.v4.u32 [%0], {%1, %2, %3, %4};" :: "r"(a), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ uint32_t dg12_qoff(uint32_t r, uint32_t u) { return r * 32 + u * 4; }
__device__ __forceinline__ uint32_t dg12_loff(uint32_t r, uint32_t u) { return r * 64 + u * 8; }
__device__ __forceinline__ uint32_t dg12_boff(uint32_t r, uint32_t u) { return r * 128 + ((u ^ (r & 7)) << 4); }
__device__ __forceinline__ uint32_t dg12_drow(uint32_t r0, uint32_t lane) { return r0 + (lane >> 3); }
__device__ __forceinline__ uint32_t dg12_dunit(uint32_t lane) { return lane & 7; }
__device__ __forceinline__ uint32_t dg12_acc(uint32_t na, uint32_t l, uint32_t j, uint32_t h, uint32_t c) { return na * l + 4 * j + 2 * h + c; }
__device__ __forceinline__ uint32_t dg12_row(uint32_t wave, uint32_t l, uint32_t g, uint32_t wm, uint32_t w, uint32_t gr, uint32_t h) { return l * wave + g * wm + 16 * w + gr + 8 * h; }

template <uint32_t SHAPE_N, uint32_t SHAPE_K, uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t kSwizzleDMode,
          uint32_t kNumStages, uint32_t kNumMathThreads, uint32_t kNumSMs, uint32_t kDiag>
CUTLASS_GLOBAL __launch_bounds__(128 + kNumMathThreads, 1) void
dg12_gemm_impl(uint32_t shape_m,
               const __grid_constant__ cute::TmaDescriptor tensor_map_a,
               const __grid_constant__ cute::TmaDescriptor tensor_map_q,
               const __grid_constant__ cute::TmaDescriptor tensor_map_l,
               const __grid_constant__ cute::TmaDescriptor tensor_map_cd,
               const float* __restrict__ col_scale, const int* __restrict__ esc_offs,
               const int4* __restrict__ esc_ent, const __nv_bfloat16* __restrict__ x) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900))
    constexpr uint32_t BLOCK_K = 64;
    constexpr uint32_t kNumTMAThreads = 128;
    constexpr uint32_t kNumDecodeWarps = 3;
    using WGMMA = typename mma::sm90::BF16MMASelector<BLOCK_N, cute::UMMA::Major::K, cute::UMMA::Major::K>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");
    DG_STATIC_ASSERT(SHAPE_N % BLOCK_N == 0 and SHAPE_K % BLOCK_K == 0 and BLOCK_N % 8 == 0 and BLOCK_N / 2 <= 256, "dg12 shape");
    const uint32_t shape_n = SHAPE_N, shape_k = SHAPE_K;

    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(cutlass::bfloat16_t)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_bfloat16);
    static constexpr uint32_t SMEM_Q_SIZE_PER_STAGE = BLOCK_N * 32;
    static constexpr uint32_t SMEM_L_SIZE_PER_STAGE = BLOCK_N * 64;
    DG_STATIC_ASSERT(SMEM_A_SIZE_PER_STAGE % 1024 == 0 and SMEM_B_SIZE_PER_STAGE % 1024 == 0 and SMEM_Q_SIZE_PER_STAGE % 128 == 0,
                     "dg12 smem alignment");

    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_q);
        cute::prefetch_tma_descriptor(&tensor_map_l);
        cute::prefetch_tma_descriptor(&tensor_map_cd);
    }
    __syncwarp();

    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    auto smem_d = reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<cutlass::bfloat16_t*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    uint8_t* const smem_raw = smem_buffer + SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_q = utils::PatternVisitor([&](const uint32_t& i) { return smem_raw + i * SMEM_Q_SIZE_PER_STAGE; });
    auto smem_l = utils::PatternVisitor([&](const uint32_t& i) {
        return smem_raw + kNumStages * SMEM_Q_SIZE_PER_STAGE + i * SMEM_L_SIZE_PER_STAGE;
    });
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(smem_raw + kNumStages * (SMEM_Q_SIZE_PER_STAGE + SMEM_L_SIZE_PER_STAGE));
    auto full_barriers  = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (i); });
    auto empty_barriers = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (kNumStages + i); });
    auto raw_barriers   = utils::PatternVisitor([=](const uint32_t& i) { return barrier_start_ptr + (2 * kNumStages + i); });

    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            full_barriers[i]->init(1 + kNumDecodeWarps);
            empty_barriers[i]->init(kNumMathThreads / 32);
            raw_barriers[i]->init(1);
        }
        cutlass::arch::fence_barrier_init();
    }
    __syncthreads();

    constexpr uint32_t kNumTMARegisters = kNumMathThreads == 128 ? 64 : 56;
    constexpr uint32_t kNumMathRegisters = kNumMathThreads == 128 ? 248 : 224;

    cudaGridDependencySynchronize();

    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<GemmType::Normal, BLOCK_M, BLOCK_N, 1, 1, false, kNumSMs>(shape_m, shape_n, shape_k, nullptr);

    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };
    constexpr uint32_t kNumKBlocks = SHAPE_K / BLOCK_K;

    if (warp_idx >= kNumMathThreads / 32) {
        cutlass::arch::warpgroup_reg_dealloc<kNumTMARegisters>();
        const uint32_t tma_warp = warp_idx - kNumMathThreads / 32;
        if (tma_warp == 2) {
            // Producer: one elected thread issues A and the raw planes for every stage
            if (cute::elect_one_sync()) {
                while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                    for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                        empty_barriers[stage_idx]->wait(phase ^ 1);
                        const uint32_t m_idx = m_block_idx * BLOCK_M, n_idx = n_block_idx * BLOCK_N;
                        auto& full_barrier = *full_barriers[stage_idx];
                        tma::copy<BLOCK_K, BLOCK_M, 128, cutlass::bfloat16_t>(
                            &tensor_map_a, &full_barrier, smem_a[stage_idx], k_block_idx * BLOCK_K, m_idx);
                        full_barrier.arrive_and_expect_tx(SMEM_A_SIZE_PER_STAGE);
                        auto& raw_barrier = *raw_barriers[stage_idx];
                        if constexpr ((kDiag & 2) == 0) {
                            tma::copy<128, BLOCK_N / 4, 0, uint8_t>(&tensor_map_q, &raw_barrier, smem_q[stage_idx], 0,
                                                                    (k_block_idx * SHAPE_N + n_idx) / 4);
                            tma::copy<128, BLOCK_N / 2, 0, uint8_t>(&tensor_map_l, &raw_barrier, smem_l[stage_idx], 0,
                                                                    (k_block_idx * SHAPE_N + n_idx) / 2);
                            raw_barrier.arrive_and_expect_tx(SMEM_Q_SIZE_PER_STAGE + SMEM_L_SIZE_PER_STAGE);
                        } else {
                            raw_barrier.arrive();             // DIAG 2: no raw planes (decoders read stale bytes)
                        }
                    }
                }
            }
        } else {
            // Decoders (PF43): raw stage -> SW128 B tile, then arrive on the full barrier. Per batch of kBatch passes the
            // warp issues all its shared loads (ld.shared, shared-window addresses) before any decode and store, so the
            // load latency overlaps instead of serializing behind the previous pass's store.
            constexpr uint32_t kRowsPerPass = kNumDecodeWarps * 4;
            constexpr uint32_t kPasses = (BLOCK_N + kRowsPerPass - 1) / kRowsPerPass;
            constexpr uint32_t kBatch = kPasses < 6 ? kPasses : 6;
            const uint32_t dw = tma_warp == 3 ? 2 : tma_warp;
            const uint32_t u = dg12_dunit(lane_idx);
            while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
                for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                    raw_barriers[stage_idx]->wait(phase);
                    const uint32_t q_base = dg12_smem_u32(smem_q[stage_idx]);
                    const uint32_t l_base = dg12_smem_u32(smem_l[stage_idx]);
                    const uint32_t b_base = dg12_smem_u32(smem_b[stage_idx]);
                    #pragma unroll
                    for (uint32_t p0 = 0; p0 < kPasses and (kDiag & 1) == 0; p0 += kBatch) {   // DIAG 1: no decode
                        uint32_t qv[kBatch], l0v[kBatch], l1v[kBatch];
                        #pragma unroll
                        for (uint32_t i = 0; i < kBatch; ++ i) {
                            const uint32_t r = dg12_drow(dw * 4 + (p0 + i) * kRowsPerPass, lane_idx);
                            if (p0 + i < kPasses and r < BLOCK_N) {
                                qv[i] = dg12_lds32(q_base + dg12_qoff(r, u));
                                dg12_lds64(l_base + dg12_loff(r, u), l0v[i], l1v[i]);
                            }
                        }
                        #pragma unroll
                        for (uint32_t i = 0; i < kBatch; ++ i) {
                            const uint32_t r = dg12_drow(dw * 4 + (p0 + i) * kRowsPerPass, lane_idx);
                            if (p0 + i < kPasses and r < BLOCK_N)
                                dg12_sts128(b_base + dg12_boff(r, u), dg12_decode(qv[i], l0v[i], l1v[i]));
                        }
                    }
                    if constexpr ((kDiag & 4) == 0)
                        cutlass::arch::fence_view_async_shared();   // DIAG 4: no proxy fence (measures its cost)
                    __syncwarp();
                    if (lane_idx == 0)
                        full_barriers[stage_idx]->arrive();
                }
            }
        }
    } else {
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        auto a_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128>(smem_a[0], math_wg_idx * WGMMA::M, 0);
        auto b_desc = mma::sm90::make_gmma_desc<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128>(smem_b[0], 0, 0);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            constexpr uint32_t kNumWaves = BLOCK_M / WAVE_BLOCK_M;
            float accum[WGMMA::kNumAccum * kNumWaves] = {0};
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");

            // 12-bit epilogue data (PF45), loaded now so the latency hides behind the k-loop: this thread's escapes
            // (n-block, lane class tq) with x at its rows, and its column scales
            constexpr uint32_t kRows = 2 * kNumWaves;
            constexpr uint32_t kMaxEsc = (BLOCK_N / 8) * kNumWaves >= 16 ? 6 : 12;
            const uint32_t gr = lane_idx >> 2, tq = lane_idx & 3, wq = warp_idx % 4;
            int e_lo = 0, e_hi = 0;
            int4 ent[kMaxEsc];
            float xe[kMaxEsc][kRows];
            float2 sc[BLOCK_N / 8];
            #pragma unroll
            for (uint32_t j = 0; j < BLOCK_N / 8; ++ j)
                sc[j] = __ldg(reinterpret_cast<const float2*>(col_scale + n_block_idx * BLOCK_N + 8 * j + 2 * tq));
            if constexpr ((kDiag & 8) == 0) {
                e_lo = __ldg(esc_offs + n_block_idx * 4 + tq);
                e_hi = __ldg(esc_offs + n_block_idx * 4 + tq + 1);
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i)
                    ent[i] = static_cast<int>(i) < e_hi - e_lo ? __ldg(esc_ent + e_lo + i) : make_int4(0, 0, 0, 0);
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i) {
                    #pragma unroll
                    for (uint32_t rr = 0; rr < kRows; ++ rr) {
                        const uint32_t row = dg12_row(WAVE_BLOCK_M, rr >> 1, math_wg_idx, WGMMA::M, wq, gr, rr & 1);
                        const uint32_t mg = m_block_idx * BLOCK_M + row;
                        xe[i][rr] = (static_cast<int>(i) < e_hi - e_lo and row < BLOCK_M and mg < shape_m)
                            ? __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(ent[i].y)]) : 0.0f;
                    }
                }
            }
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= 64 or warp_idx < kNumWGMMAStoreThreads / 32;

            for (uint32_t k_block_idx = 0; k_block_idx < kNumKBlocks; advance_pipeline(k_block_idx)) {
                const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);
                full_barriers[stage_idx]->wait(phase);
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumWaves; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_arrive();
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < kNumWaves; ++ local_idx) {
                    auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                        a_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_M, BLOCK_K, 128, nv_bfloat16>(
                            a_desc_base_lo, local_idx * WAVE_BLOCK_M, k * WGMMA::K, 0);
                        b_desc.reg32_[0] = mma::sm90::advance_gmma_desc_lo<cute::UMMA::Major::K, BLOCK_N, BLOCK_K, 128, nv_bfloat16>(
                            b_desc_base_lo, 0, k * WGMMA::K, 0);
                        WGMMA::wgmma(a_desc, b_desc, shifted_accum, 1);
                    }
                }
                ptx::warpgroup_commit_batch();
                #pragma unroll
                for (uint32_t i = 0; i < WGMMA::kNumAccum * kNumWaves; ++ i)
                    ptx::warpgroup_fence_operand(accum[i]);
                ptx::warpgroup_wait<0>();
                if (lane_idx == 0)
                    empty_barriers[stage_idx]->arrive();
            }

            // 12-bit epilogue: escape corrections in the scaled domain (acc -= x_k 2^-15; acc += x_k w_true_scaled), from
            // registers for the first kMaxEsc entries and the old serial path for any overflow, then the column scale
            if constexpr ((kDiag & 8) == 0) {
                #pragma unroll
                for (uint32_t i = 0; i < kMaxEsc; ++ i) {
                    if (static_cast<int>(i) < e_hi - e_lo) {
                        const uint32_t nl = static_cast<uint32_t>(ent[i].x), c = nl & 1u;
                        const float val = __int_as_float(ent[i].z);
                        #pragma unroll
                        for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                            if ((nl >> 3) == j) {
                                #pragma unroll
                                for (uint32_t rr = 0; rr < kRows; ++ rr) {
                                    float& a0 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)];
                                    float& a1 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)];
                                    if (c == 0) {
                                        a0 = fmaf(-xe[i][rr], 3.0517578125e-05f, a0);
                                        a0 = fmaf(xe[i][rr], val, a0);
                                    } else {
                                        a1 = fmaf(-xe[i][rr], 3.0517578125e-05f, a1);
                                        a1 = fmaf(xe[i][rr], val, a1);
                                    }
                                }
                            }
                        }
                    }
                }
                for (int e = e_lo + static_cast<int>(kMaxEsc); e < e_hi; ++ e) {
                    const int4 en = __ldg(esc_ent + e);
                    const uint32_t nl = static_cast<uint32_t>(en.x), c = nl & 1u;
                    const float val = __int_as_float(en.z);
                    #pragma unroll
                    for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                        if ((nl >> 3) == j) {
                            #pragma unroll
                            for (uint32_t rr = 0; rr < kRows; ++ rr) {
                                const uint32_t row = dg12_row(WAVE_BLOCK_M, rr >> 1, math_wg_idx, WGMMA::M, wq, gr, rr & 1);
                                const uint32_t mg = m_block_idx * BLOCK_M + row;
                                if (row < BLOCK_M and mg < shape_m) {
                                    const float xk = __bfloat162float(x[static_cast<uint64_t>(mg) * SHAPE_K + static_cast<uint32_t>(en.y)]);
                                    float& a0 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)];
                                    float& a1 = accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)];
                                    if (c == 0) {
                                        a0 = fmaf(-xk, 3.0517578125e-05f, a0);
                                        a0 = fmaf(xk, val, a0);
                                    } else {
                                        a1 = fmaf(-xk, 3.0517578125e-05f, a1);
                                        a1 = fmaf(xk, val, a1);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            #pragma unroll
            for (uint32_t j = 0; j < BLOCK_N / 8; ++ j) {
                #pragma unroll
                for (uint32_t rr = 0; rr < kRows; ++ rr) {
                    accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 0)] *= sc[j].x;
                    accum[dg12_acc(WGMMA::kNumAccum, rr >> 1, j, rr & 1, 1)] *= sc[j].y;
                }
            }

            // DeepGEMM's epilogue (BF16 D): STSM into the swizzled D tile, TMA store
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
            if (not do_wgmma_store)
                continue;
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
            DG_STATIC_ASSERT(kSwizzleDMode > 0, "Invalid swizzling type");
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < kNumWaves; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    constexpr uint32_t kNumBankGroupBytes = 16;
                    auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);
                    auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);
                    constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                    auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                    auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                    col ^= row % (kSwizzleDMode / 16);
                    uint8_t* smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +
                        warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +
                        m_offset * kSwizzleDMode +
                        atom_offset * BLOCK_M * kSwizzleDMode +
                        row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes;
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr);
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 0);
            const auto m_idx = m_block_idx * BLOCK_M;
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                cute::SM90_TMA_STORE_2D::copy(&tensor_map_cd, smem_ptr, n_block_idx * BLOCK_N + in_block_n_offset, m_idx);
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

}  // namespace deep_gemm

namespace dg12_@@NAME@@ {

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                   const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                   CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn encode_fn() {
  static EncodeTiledFn fn = nullptr;
  if (fn == nullptr) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) fn = reinterpret_cast<EncodeTiledFn>(dlsym(h, "cuTensorMapEncodeTiled"));
  }
  return fn;
}

static CUtensorMapSwizzle swizzle_of(int mode) {
  switch (mode) {
    case 32: return CU_TENSOR_MAP_SWIZZLE_32B;
    case 64: return CU_TENSOR_MAP_SWIZZLE_64B;
    case 128: return CU_TENSOR_MAP_SWIZZLE_128B;
    default: return CU_TENSOR_MAP_SWIZZLE_NONE;
  }
}

// gmem [outer][inner] elements of elem_bytes (row stride outer_stride_bytes); with a swizzle the smem inner box is
// swizzle / elem_bytes elements (DeepGEMM's make_tma_2d_desc).
static int make_2d(CUtensorMap* map, CUtensorMapDataType dtype, int elem_bytes, void* ptr, int inner, int outer,
                   int smem_inner, int smem_outer, long long outer_stride_bytes, int swizzle) {
  EncodeTiledFn fn = encode_fn();
  if (fn == nullptr) return 9001;
  if (swizzle != 0) smem_inner = swizzle / elem_bytes;
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(inner), static_cast<cuuint64_t>(outer)};
  const cuuint32_t sdims[2] = {static_cast<cuuint32_t>(smem_inner), static_cast<cuuint32_t>(smem_outer)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(outer_stride_bytes)};
  const cuuint32_t estrides[2] = {1, 1};
  const CUresult r = fn(map, dtype, 2, ptr, gdims, gstrides, sdims, estrides, CU_TENSOR_MAP_INTERLEAVE_NONE,
                        swizzle_of(swizzle), CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  return r == CUDA_SUCCESS ? 0 : 10000 + static_cast<int>(r);
}

constexpr uint32_t BM = @@BM@@, BN = @@BN@@, ST = @@ST@@, SWD = @@SWD@@, NMATH = @@NMATH@@;
constexpr uint32_t N = @@N@@, K = @@K@@, SMEM = @@SMEM@@, NUM_SMS = 132, DIAG = @@DIAG@@;

static auto kernel = &deep_gemm::dg12_gemm_impl<N, K, BM, BN, SWD, ST, NMATH, NUM_SMS, DIAG>;

}  // namespace dg12_@@NAME@@

int dg12_launch_@@NAME@@(const void* x, const void* q, const void* l, void* out, const void* scale, const void* offs8,
                         const void* ent, int m, void* stream) {
  using namespace dg12_@@NAME@@;
  CUtensorMap ta, tq, tl, td;
  int e = make_2d(&ta, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), K, m, 64, BM, 2LL * K, 128);
  // chunk-major planes: Q = (K/64)*N rows of 32 B viewed as rows of 128 B; L = rows of 64 B viewed as rows of 128 B
  if (e == 0) e = make_2d(&tq, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(q), 128, (K / 64) * N / 4, 128, BN / 4, 128, 0);
  if (e == 0) e = make_2d(&tl, CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, const_cast<void*>(l), 128, (K / 64) * N / 2, 128, BN / 2, 128, 0);
  if (e == 0) e = make_2d(&td, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, out, N, m, BN, BM, 2LL * N, SWD);
  if (e != 0) return e;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(NUM_SMS, 1, 1);
  cfg.blockDim = dim3(128 + NMATH, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = static_cast<cudaStream_t>(stream);
  cfg.attrs = nullptr;
  cfg.numAttrs = 0;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, kernel, static_cast<uint32_t>(m), ta, tq, tl, td,
                                             static_cast<const float*>(scale), static_cast<const int*>(offs8),
                                             static_cast<const int4*>(ent), static_cast<const __nv_bfloat16*>(x));
  return static_cast<int>(err);
}

int dg12_attr_@@NAME@@() {
  return static_cast<int>(cudaFuncSetAttribute(dg12_@@NAME@@::kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                               static_cast<int>(dg12_@@NAME@@::SMEM)));
}

long long dg12_info_@@NAME@@() {
  cudaFuncAttributes a;
  if (cudaFuncGetAttributes(&a, dg12_@@NAME@@::kernel) != cudaSuccess) return -1;
  return static_cast<long long>(a.numRegs) * 1000000 + static_cast<long long>(a.localSizeBytes);
}
"""

# Instances compiled into the module, (BM, BN, ST, SWD, NMATH, N, K).
DG12_INSTANCES = ()
SMEM_CAPACITY = 232448


def dg12_unpack(inst):
    return tuple(inst) + ((0,) if len(inst) == 7 else ())


def dg12_name(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    return f"dg12_{bm}_{bn}_{st}_{swd}_{nmath}_{n}_{k}" + (f"_d{diag}" if diag else "")


def dg12_smem(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    smem_d = -(-(bm * bn * 2) // 1024) * 1024
    return smem_d + st * (bm * 128 + bn * 128 + bn * 96) + 3 * st * 8


def dg12_stages(bm, bn, max_stages=16):
    smem_d = -(-(bm * bn * 2) // 1024) * 1024
    return min(max_stages, (SMEM_CAPACITY - smem_d) // (bm * 128 + bn * 128 + bn * 96 + 24))


def dg12_cu(inst):
    bm, bn, st, swd, nmath, n, k, diag = dg12_unpack(inst)
    src = DG12_CU
    for key, val in (("NAME", dg12_name(inst)), ("BM", bm), ("BN", bn), ("ST", st), ("SWD", swd), ("NMATH", nmath),
                     ("N", n), ("K", k), ("SMEM", dg12_smem(inst)), ("DIAG", diag)):
        src = src.replace(f"@@{key}@@", str(val))
    return src


def dg12_cpp(inst):
    """C++ side of one instance's module: torch wrapper (shape checks, current stream), attr and info."""
    name, n, k = dg12_name(inst), inst[5], inst[6]
    return f"""#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
int dg12_launch_{name}(const void* x, const void* q, const void* l, void* out, const void* scale, const void* offs8,
                       const void* ent, int m, void* stream);
int dg12_attr_{name}();
long long dg12_info_{name}();
void {name}(torch::Tensor x, torch::Tensor q, torch::Tensor l, torch::Tensor scale, torch::Tensor offs8, torch::Tensor ent,
            torch::Tensor out) {{
  TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16, "dg12: x/out dtype");
  TORCH_CHECK(q.scalar_type() == at::kInt && l.scalar_type() == at::kInt && offs8.scalar_type() == at::kInt
              && ent.scalar_type() == at::kInt && scale.scalar_type() == at::kFloat, "dg12: plane dtypes");
  TORCH_CHECK(x.is_contiguous() && q.is_contiguous() && l.is_contiguous() && out.is_contiguous() && scale.is_contiguous()
              && offs8.is_contiguous() && ent.is_contiguous(), "dg12: layouts");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == {k} && x.size(0) >= 1 && out.size(0) == x.size(0) && out.size(1) == {n}, "dg12: x/out shape");
  TORCH_CHECK(q.numel() >= {n}LL * {k // 8} && l.numel() >= {n}LL * {k // 4} && scale.numel() >= {n}
              && offs8.numel() >= {(n // inst[1]) * 4 + 1}, "dg12: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(q.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(l.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(scale.data_ptr()) % 8 == 0 && reinterpret_cast<uintptr_t>(ent.data_ptr()) % 16 == 0,
              "dg12: unaligned");
  const int e = dg12_launch_{name}(x.data_ptr(), q.data_ptr(), l.data_ptr(), out.data_ptr(), scale.data_ptr(),
                                    offs8.data_ptr(), ent.data_ptr(), static_cast<int>(x.size(0)),
                                    static_cast<void*>(at::cuda::getCurrentCUDAStream().stream()));
  TORCH_CHECK(e == 0, "dg12 launch failed: ", e);
}}
void {name}_attr() {{
  const int e = dg12_attr_{name}();
  TORCH_CHECK(e == 0, "dg12 attr failed: ", e);
}}
int64_t {name}_info() {{ return static_cast<int64_t>(dg12_info_{name}()); }}
"""


def dg12_include_dirs():
    import deep_gemm
    return [os.path.join(os.path.dirname(deep_gemm.__file__), "include")]


def dg12_build_one(inst, include_dirs):
    from torch.utils.cpp_extension import load_inline
    name = dg12_name(inst)
    return load_inline(
        name="okc_proj_" + name, cpp_sources=dg12_cpp(inst), cuda_sources=dg12_cu(inst),
        functions=[name, name + "_attr", name + "_info"], extra_include_paths=list(include_dirs),
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr",
                           "--expt-extended-lambda", "--ptxas-options=--register-usage-level=10"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_" + name + "_"), verbose=False, no_implicit_headers=True)


def dg12_build(instances, include_dirs=None, timeout=None):
    """Every instance's module built from a thread pool; waits at most `timeout` seconds. Returns (namespace of the
    built functions, {name: error})."""
    import concurrent.futures
    import types
    inc = include_dirs or dg12_include_dirs()
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(instances)))
    futures = {pool.submit(dg12_build_one, inst, inc): inst for inst in instances}
    done, _ = concurrent.futures.wait(list(futures), timeout=timeout)
    pool.shutdown(wait=False)
    fns, errors = {}, {}
    for fut, inst in futures.items():
        name = dg12_name(inst)
        if fut not in done:
            errors[name] = TimeoutError(f"build not finished within {timeout:.0f}s")
            continue
        try:
            mod = fut.result()
            for fn in (name, name + "_attr", name + "_info"):
                fns[fn] = getattr(mod, fn)
        except Exception as exc:
            errors[name] = exc
    return types.SimpleNamespace(**fns), errors


DG12_TARGETS = {('pooler', 128): [{'BM': 64, 'BN': 64, 'ST': 9, 'SWD': 128, 'NMATH': 128}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 32, 'ST': 9, 'SWD': 64, 'NMATH': 256}], ('unpooler', 128): [{'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 128, 'ST': 4, 'SWD': 128, 'NMATH': 256}], ('qkv_gate', 128): [{'BM': 64, 'BN': 48, 'ST': 11, 'SWD': 32, 'NMATH': 128}, {'BM': 128, 'BN': 48, 'ST': 8, 'SWD': 32, 'NMATH': 256}], ('o', 128): [{'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 128, 'ST': 4, 'SWD': 128, 'NMATH': 256}], ('pooler', 256): [{'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 32, 'ST': 9, 'SWD': 64, 'NMATH': 256}, {'BM': 128, 'BN': 128, 'ST': 4, 'SWD': 128, 'NMATH': 256}], ('unpooler', 256): [{'BM': 128, 'BN': 128, 'ST': 4, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}], ('qkv_gate', 256): [{'BM': 128, 'BN': 48, 'ST': 8, 'SWD': 32, 'NMATH': 256}, {'BM': 64, 'BN': 96, 'ST': 7, 'SWD': 64, 'NMATH': 128}], ('o', 256): [{'BM': 128, 'BN': 128, 'ST': 4, 'SWD': 128, 'NMATH': 256}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256}], ('pooler', 32): [{'BM': 32, 'BN': 32, 'ST': 16, 'SWD': 64, 'NMATH': 128}], ('unpooler', 32): [{'BM': 32, 'BN': 64, 'ST': 12, 'SWD': 128, 'NMATH': 128}], ('qkv_gate', 32): [{'BM': 32, 'BN': 32, 'ST': 16, 'SWD': 64, 'NMATH': 128}], ('o', 32): [{'BM': 32, 'BN': 64, 'ST': 12, 'SWD': 128, 'NMATH': 128}]}
DG12_DIAG = {('unpooler', 128): [{'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256, 'DIAG': 0}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256, 'DIAG': 1}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256, 'DIAG': 8}, {'BM': 128, 'BN': 64, 'ST': 7, 'SWD': 128, 'NMATH': 256, 'DIAG': 9}], ('pooler', 128): [{'BM': 64, 'BN': 64, 'ST': 9, 'SWD': 128, 'NMATH': 128, 'DIAG': 0}, {'BM': 64, 'BN': 64, 'ST': 9, 'SWD': 128, 'NMATH': 128, 'DIAG': 8}, {'BM': 64, 'BN': 64, 'ST': 9, 'SWD': 128, 'NMATH': 128, 'DIAG': 9}]}


# ====================================================================== lab (sweep): check runs only

"""Sweep helpers shared by the lab sections from PE2 on (PE1's lab/e1.py keeps its own copy).

A Sweep times, per (layer, m), SGLang's path and the base submission's current CHOICE entry as
references, then each candidate that is correct against FP64 (half the judge's bounds, probe rows
included). commit() re-checks each winner on outlier inputs, re-times it next to the references
and puts it in CHOICE only if it beats the faster reference by MIN_GAIN, so the run's timed
kernel_fn uses it."""

import math

MIN_GAIN = 1.03
FULL = {"qkv_gate": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096),
        "router": (256, 8192)}


def lab_n(layer, m):
    return 1024 if layer == "unpooler" and m <= 8 else FULL[layer][0]


def lab_input(m, k, seed, dist="normal", probe_all=False):
    """Like the judge's generate_input: standard normal, ~1 row in 8 a single power of two
    (every row with probe_all, so the probe check runs at any m)."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device="cuda")
    if dist == "outliers":
        idx = torch.randperm(k, generator=gen, device="cuda")[: max(1, k // 100)]
        x[:, idx] *= 30
    probes = torch.rand((m,), generator=gen, device="cuda") < 1 / 8
    if m >= 8:
        probes[torch.randint(m, (1,), generator=gen, device="cuda")] = True
    if probe_all:
        probes[:] = True
    cols = torch.randint(k, (m,), generator=gen, device="cuda")
    signs = torch.randint(2, (m,), generator=gen, device="cuda") * 2 - 1
    powers = torch.randint(3, 8, (m,), generator=gen, device="cuda")
    spike = torch.zeros_like(x).scatter_(1, cols[:, None], (signs * 2.0 ** powers).float()[:, None])
    return torch.where(probes[:, None], spike, x).to(torch.bfloat16)


def lab_errors(out, x, w):
    """(relative L2, worst row relative to max(own norm, 0.1 x mean), worst probe row) vs FP64."""
    want = torch.float32 if w.dtype == torch.float32 else torch.bfloat16
    ref = (x.double() @ w.double().T).to(want).double()
    if not isinstance(out, torch.Tensor) or out.shape != ref.shape or out.dtype != want:
        return math.inf, math.inf, math.inf
    o = out.double()
    if not torch.isfinite(o).all():
        return math.inf, math.inf, math.inf
    diff = o - ref
    rel = float(diff.norm() / ref.norm().clamp(min=1e-12))
    rows_ref = ref.norm(dim=-1)
    row = float((diff.norm(dim=-1) / torch.maximum(rows_ref, 0.1 * rows_ref.mean())).max())
    probe = ((x != 0).sum(-1) == 1).nonzero().flatten()
    pr = float((diff[probe].norm(dim=-1) / rows_ref[probe].clamp(min=1e-30)).max()) if len(probe) else 0.0
    return rel, row, pr


def lab_ok(errs, layer):
    tol = 1e-4 if layer == "router" else 1e-2
    rel, row, pr = errs
    return rel <= 0.5 * tol and row <= 2.5 * tol and pr <= 0.5e-3


def cfg_name(c):
    if c is None:
        return "sglang"
    if c["kind"] in ("swap", "gemm"):
        return (f"{c['kind']} BM{c['BM']} BN{c['BN']} BK{c['BK']} s{c['split']} m{c['mode']} "
                f"w{c['warps']} st{c['stages']}")
    if c["kind"] == "fma":
        return f"fma BN{c['BN']} s{c['split']} w{c['warps']}"
    if c["kind"] == "fi_mm":
        return f"fi_mm {c['backend']}"
    if c["kind"] == "dg":
        return "dg native" if c["native"] else "dg cublaslt"
    if c["kind"] == "p9":
        return f"p9 NT{c['NT']} RTW{c['RTW']} RG{c['RG']} KSW{c['KSW']} D{c['D']} CS{c['CS']}"
    if c["kind"] == "rg":
        return f"rg RTC{c['RTC']} NTW{c['NTW']} CS{c['CS']} XS{c['XS']} D{c['D']}"
    if c["kind"] == "sx" and c.get("W"):
        return f"sxw NB{c['NB']} WGS{c['WGS']} KSW{c['KSW']} NS{c['NS']} P{c['P']} CS{c['CS']}"
    if c["kind"] == "sx":
        return (f"sx NT8{c['NT8']} TG{c['TG']} RTW{c['RTW']} RGN{c['RGN']} KSW{c['KSW']} NS{c['NS']} D{c['D']} "
                f"XS{c['XS']} CS{c['CS']}")
    if c["kind"] == "wgf":
        return f"wgf KS{c['KS']} ST{c['ST']} TT{c['TT']} D{c['D']}"
    if c["kind"] == "wg":
        return f"wg KS{c['KS']} ST{c['ST']} TT{c['TT']}"
    if c["kind"] == "f12m":
        return (f"f12m NT{c['NT']} RTW{c['RTW']} RG{c['RG']} KSW{c['KSW']} D{c['D']}"
                + (f" CS{c['CS']}" if c.get("CS", 1) > 1 else "") + ("" if c["F12"] else " bf16"))
    if c["kind"] == "f12v":
        return f"f12v R{c['R']} RG{c['RG']} KSW{c['KSW']} D{c['D']}"
    if c["kind"] in ("fi_wsk", "fi_direct"):
        return f"{c['kind']} {c['tactic']}"
    return c["kind"] + (f" max_m{c['max_m']}" if "max_m" in c else "")


class Sweep:
    def __init__(self, state):
        self.state = state
        self.results = {}
        self.logged = set()

    def note(self, key, msg):
        if key not in self.logged and len(self.logged) < 8:
            self.logged.add(key)
            log(msg)

    def shape(self, lab, layer, m, cands):
        n, k = lab_n(layer, m), FULL[layer][1]
        st = self.state
        x = lab_input(m, k, 1000 + m)
        x_probe = lab_input(m, k, 1500 + m, probe_all=True)
        w = st[layer][:n]
        rec = self.results.get((layer, m))
        if rec is None:
            cur = CHOICE.get((layer, m))
            rec = {"best": None, "n": 0, "fails": 0, "wrong": 0, "current": cur,
                   "sglang": lab.time_graph(lambda: sglang_path(x, layer, n, st))}
            rec["cur_us"] = lab.time_graph(lambda: run_choice(cur, x, layer, n, st)) if cur else rec["sglang"]
            self.results[(layer, m)] = rec
        for c in cands:
            if lab.left() < 8:
                log(f"{layer} m{m}: out of budget after {rec['n']} candidates")
                break
            try:
                errs = lab_errors(run_choice(c, x, layer, n, st), x, w)
                errs_p = lab_errors(run_choice(c, x_probe, layer, n, st), x_probe, w)
                errs = (max(errs[0], errs_p[0]), max(errs[1], errs_p[1]), max(errs[2], errs_p[2]))
            except Exception as exc:
                rec["fails"] += 1
                text = _error_text(exc, 300)
                self.note(("fail", c["kind"], text[-120:]), f"{layer} m{m} {cfg_name(c)}: FAILED {type(exc).__name__}: {text}")
                continue
            if not lab_ok(errs, layer):
                rec["wrong"] += 1
                self.note(("wrong", layer, c["kind"]),
                          f"{layer} m{m} {cfg_name(c)}: WRONG rel {errs[0]:.2e} row {errs[1]:.2e} probe {errs[2]:.2e}")
                continue
            us = lab.time_graph(lambda: run_choice(c, x, layer, n, st))
            rec["n"] += 1
            rec.setdefault("all", []).append((us, c))
            if rec["best"] is None or us < rec["best"][0]:
                rec["best"] = (us, c)

    def commit(self, lab, top=None):
        st = self.state
        for (layer, m), rec in sorted(self.results.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            n, k = lab_n(layer, m), FULL[layer][1]
            cur = rec["current"]
            line = (f"{layer:9s} m{m:<4d} sglang {rec['sglang']:6.2f} | v1 {cfg_name(cur)} {rec['cur_us']:6.2f}"
                    f" | {rec['n']} ok")
            if rec["fails"] or rec["wrong"]:
                line += f", {rec['fails']} failed, {rec['wrong']} wrong"
            if rec["best"] is None:
                log(line + " -> keep v1")
                continue
            c = rec["best"][1]
            x = lab_input(m, k, 2000 + m)
            x_out = lab_input(m, k, 3000 + m, dist="outliers")
            ok = lab_ok(lab_errors(run_choice(c, x_out, layer, n, st), x_out, st[layer][:n]), layer)
            t_s = min(rec["sglang"], lab.time_graph(lambda: sglang_path(x, layer, n, st)))
            t_v = t_s if cur is None else min(rec["cur_us"], lab.time_graph(lambda: run_choice(cur, x, layer, n, st)))
            t_c = min(rec["best"][0], lab.time_graph(lambda: run_choice(c, x, layer, n, st)))
            ref = min(t_s, t_v)
            gbs = (n * k * st[layer].element_size()) / (t_c * 1e-6) / 1e12
            line += f" -> best {cfg_name(c)} {t_c:.2f} us ({ref / t_c:.3f}x of the faster reference, {gbs:.2f} TB/s)"
            if ok and ref / t_c >= MIN_GAIN:
                CHOICE[(layer, m)] = c
                log(line + " CHOSEN")
            else:
                log(line + (" (outliers WRONG)" if not ok else " (below MIN_GAIN)") + " -> keep v1")


# ====================================================================== lab (e28): check runs only

"""PE28 (P40 step 2d): DG12 with the escape epilogue prefetched per thread (PF45), batched decoder (PF43), BM 128
variants and chunk-major raw planes; plus DIAG timings (1 no decode, 8 no escape corrections, 9 both) to show what is
left (PF42: a stage's raw data = one contiguous block, TMA'd as 128-B
rows), otherwise as PE24: DeepGEMM's sm90 BF16 GEMM with the 12-bit weight load (proj/dg/dg12.py; planes from
proj/dg/dg12_layout.py packed in prepare()), plain launch (PF41), on all four BF16 layers at m = 32/128/256 with
DeepGEMM's block shapes (stages recomputed for the raw planes) plus a smaller-BN alternative at m = 128/256; timed
next to the base (v10: cuBLAS / DeepGEMM / Triton). Winners >= 3% enter CHOICE with the base kernel as fallback,
printed as "V22 CHOICE" lines."""

import time

PREPARE_CAP_S = 280.0
LAB_MAX_S = 200.0
DG12_BUILD_MAX_START_S = 160.0
DG12_BUILD_DEADLINE_S = 245.0
DG12_MOD = None
DG12_OK = set()


def dg12_inst(c, n, k):
    return (c["BM"], c["BN"], c["ST"], c["SWD"], c["NMATH"], n, k, c.get("DIAG", 0))


def dg12_setup_lab(state, t_start):
    global DG12_MOD
    state["dg12"] = {}
    t0 = time.monotonic()
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        try:
            pk = dg12_pack(state[layer])
        except Exception as exc:
            log(f"dg12 pack {layer} FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-200:]}")
            pk = None
        if pk is not None:
            state["dg12"][layer] = {"qc": pk["qc"], "lc": pk["lc"], "scale": pk["scale"], "et": pk["et"], "n_esc": pk["n_esc"]}
    log(f"dg12 pack: {sorted(state['dg12'])} in {time.monotonic() - t0:.1f}s; escapes "
        + ", ".join(f"{k} {v['n_esc']}" for k, v in state["dg12"].items()))
    if time.monotonic() - t_start > DG12_BUILD_MAX_START_S:
        log(f"dg12 build skipped: {time.monotonic() - t_start:.0f}s already used")
        return
    insts = tuple(dict.fromkeys([dg12_inst(c, FULL[layer][0], FULL[layer][1]) for (layer, m), cs in DG12_TARGETS.items()
                                 for c in cs] + [dg12_inst(c, FULL[layer][0], FULL[layer][1])
                                                 for (layer, m), cs in DG12_DIAG.items() for c in cs]))
    t0 = time.monotonic()
    try:
        mod, errors = dg12_build(insts, timeout=max(10.0, DG12_BUILD_DEADLINE_S - (time.monotonic() - t_start)))
    except Exception as exc:
        log(f"dg12 build FAILED to start ({type(exc).__name__}): {' '.join(str(exc).split())[-500:]}")
        return
    shown = 0
    for name, exc in errors.items():
        text = str(exc)
        if shown < 2:
            errs = [ln.strip() for ln in text.splitlines() if "error" in ln.lower()][:5]
            log(f"dg12 {name} FAILED ({type(exc).__name__}): " + " | ".join(e[-300:] for e in errs)
                + " || tail: " + " ".join(text.split())[-500:])
        shown += 1
    ok = [inst for inst in insts if dg12_name(inst) not in errors]
    regs = []
    for inst in ok:
        try:
            getattr(mod, dg12_name(inst) + "_attr")()
            v = int(getattr(mod, dg12_name(inst) + "_info")())
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}:{v // 1000000}r" + (f"+{v % 1000000}B" if v % 1000000 else ""))
        except Exception as exc:
            regs.append(f"{inst[0]}x{inst[1]}s{inst[2]}: attr FAILED {' '.join(str(exc).split())[-120:]}")
    log(f"dg12 build: {len(ok)}/{len(insts)} instances ok in {time.monotonic() - t0:.1f}s; regs {' '.join(regs)}")
    if ok:
        DG12_MOD = mod
        DG12_OK.update(dg12_name(inst) for inst in ok)


def run_dg12(c, x, layer, n, state):
    m, k = x.shape
    pk = state.get("dg12", {}).get(layer)
    if DG12_MOD is None or pk is None or dg12_name(dg12_inst(c, n, k)) not in DG12_OK or n != FULL[layer][0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    offs, ent = pk["et"][c["BN"]]
    getattr(DG12_MOD, dg12_name(dg12_inst(c, n, k)))(x.contiguous(), pk["qc"], pk["lc"], pk["scale"], offs, ent, out)
    return out


_run_choice_pre_dg12 = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dg12":
        return run_dg12(c, x, layer, n, state)
    return _run_choice_pre_dg12(c, x, layer, n, state)


def sec_dg12(lab, sweep, layer, m):
    if DG12_MOD is None or layer not in sweep.state.get("dg12", {}):
        return
    n, k = FULL[layer]
    cands = [dict(kind="dg12", fallback=CHOICE.get((layer, m)), **c) for c in DG12_TARGETS[(layer, m)]
             if dg12_name(dg12_inst(c, n, k)) in DG12_OK]
    if not cands:
        return
    sweep.shape(lab, layer, m, cands)
    rec = sweep.results[(layer, m)]
    nb = n * k * 1.5
    times = " ".join(f"{c['BM']}x{c['BN']}s{c['ST']}:{us:.2f}({nb / us / 1e6:.2f})"
                     for us, c in sorted(rec.get("all", []), key=lambda e: e[0]))
    log(f"dg12 {layer} m{m} (sglang {rec['sglang']:.2f}, base {cfg_name(rec['current'])} {rec['cur_us']:.2f} us; "
        f"{rec['n']} ok, {rec['fails']} failed, {rec['wrong']} wrong) BMxBN s stages:us(12-bit TB/s) {times}")


def sec_diag(lab, state, layer, m):
    n, k = FULL[layer]
    pk = state.get("dg12", {}).get(layer)
    if DG12_MOD is None or pk is None:
        return
    x = lab_input(m, k, 4200 + m)
    w = state[layer][:n]
    parts = []
    for c in DG12_DIAG[(layer, m)]:
        name = dg12_name(dg12_inst(c, n, k))
        if name not in DG12_OK:
            parts.append(f"d{c['DIAG']}:FAILED")
            continue
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        fn = getattr(DG12_MOD, name)
        offs, ent = pk["et"][c["BN"]]

        def run(fn=fn, out=out, offs=offs, ent=ent):
            fn(x, pk["qc"], pk["lc"], pk["scale"], offs, ent, out)
            return out
        try:
            run()
            us = lab.time_graph(run)
            tag = ""
            if c["DIAG"] == 0:
                errs = lab_errors(run(), x, w)
                tag = "(ok)" if lab_ok(errs, layer) else f"(WRONG {errs[0]:.1e})"
            parts.append(f"d{c['DIAG']}:{us:.2f}{tag}")
        except Exception as exc:
            parts.append(f"d{c['DIAG']}:ERR {type(exc).__name__}")
    c0 = DG12_DIAG[(layer, m)][0]
    log(f"diag {layer} m{m} {c0['BM']}x{c0['BN']}s{c0['ST']}: " + " ".join(parts)
        + "  [bits: 1 no decode, 2 no raw TMA, 4 no fence, 8 no escape corrections]")


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    log(f"base prepare (v10) took {time.monotonic() - t0:.1f}s")
    dg12_setup_lab(state, t0)
    lab = Lab(min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0)))
    sweep = Sweep(state)
    for layer, m in DG12_TARGETS:
        lab.section(f"dg12 {layer} m{m}", sec_dg12, sweep, layer, m, min_left=6.0)
    for layer, m in DG12_DIAG:
        lab.section(f"diag {layer} m{m}", sec_diag, state, layer, m, min_left=6.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    for (layer, m), c in sorted(CHOICE.items()):
        if c.get("kind") == "dg12":
            log(f"V22 CHOICE {layer} m{m}: " + repr({k: v for k, v in c.items() if k != "fallback"}))
    log(f"prepare took {time.monotonic() - t0:.0f}s")
    return state
