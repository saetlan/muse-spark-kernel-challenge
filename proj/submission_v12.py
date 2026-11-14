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


def prepare(weights):
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
  __shared__ float part[KSW][RB][MT + 3];   // row stride MT + 3: <= 2-way banks (PE45, PF62)
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


# ====================================================================== DG12-RS (shared/gemm/dg12rs, P41): 12-bit weights into register-A wgmma, m >= 32

from torch.utils.cpp_extension import load_inline

DGRS_INSTANCES = ((32, 1, 0, 2), (64, 1, 0, 2), (128, 1, 0, 2))  # (BM, CPS, DIAG, MATH) compiled by default
DGRS_W_CHUNK = 8 * 3 * 32 * 16  # weight bytes per 128-row x 64-k stage


def _wgmma_rs(n):
    regs = n // 2
    d = ", ".join(f"%{i}" for i in range(regs))
    a = ", ".join(f"%{regs + i}" for i in range(4))
    asm = (r"{\n.reg .pred p;\nsetp.ne.b32 p, %" + str(regs + 5) + r", 0;\n"
           f"wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{d}}}, {{{a}}}, %{regs + 4}, p, 1, 1, 0;"
           + r"\n}\n")
    outs = ", ".join(f'"+f"(d[{i}])' for i in range(regs))
    return (f"__device__ __forceinline__ void wgmma{n}(float (&d)[{regs}], uint32_t a0, uint32_t a1, uint32_t a2, "
            f"uint32_t a3, uint64_t desc, int sc) {{\n  asm volatile(\"{asm}\"\n      : {outs}\n"
            f"      : \"r\"(a0), \"r\"(a1), \"r\"(a2), \"r\"(a3), \"l\"(desc), \"r\"(sc));\n}}\n")


_DGRS_CUDA = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <dlfcn.h>
#include <stdint.h>

namespace dgrs {

constexpr int W_CHUNK = 12288;          // 8 row tiles x 3 slots x 32 lanes x 16 B
constexpr int SMEM_DYN1 = 226 * 1024;  // dynamic shared memory per CTA at 1 CTA/SM (227 KB opt-in max, F47)
constexpr int SMEM_DYN2 = 112 * 1024;  // per CTA at 2 CTAs/SM (228 KB per SM, 1 KB reserved per CTA)
constexpr int PRE_WARPS = 3;           // producer warps 1-3 precompute the finalize rows (pre_corr)
constexpr int ESC_CAP = 304;           // escapes listed per warp: ceil(128 / 3) rows x 7 = 301
constexpr int GATHER = 512;            // pre_corr gather buffer (u32 words): GATHER / BM list entries x BM tokens
constexpr int SMAX = 12;                // most ring slots
constexpr long long SPIN_LIMIT = 1ll << 22;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("{\n.reg .b64 st;\nmbarrier.arrive.shared::cta.b64 st, [%0];\n}\n" ::"r"(bar) : "memory");
}
// Bounded parity wait: a protocol fault flags err and bails instead of hanging the run.
__device__ __forceinline__ bool mbar_wait(uint32_t bar, uint32_t parity, int* err) {
  for (long long i = 0; i < SPIN_LIMIT; ++i) {
    uint32_t done;
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}\n"
                 : "=r"(done) : "r"(bar), "r"(parity) : "memory");
    if (done) return true;
    if ((i & 1023) == 1023 && *reinterpret_cast<volatile int*>(err) != 0) return false;
  }
  atomicAdd(err, 1);
  return false;
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               ::"r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}
// TMA 2-D tile load (inner coordinate c0 in elements, outer c1 in rows) into this CTA's shared memory.
__device__ __forceinline__ void tma_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               ::"r"(dst), "l"(reinterpret_cast<uint64_t>(map)), "r"(c0), "r"(c1), "r"(bar) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
// fp32 at this CTA's shared offset `addr` in cluster rank `rank`'s shared memory (DSMEM)
__device__ __forceinline__ float ld_peer_f32(uint32_t addr, uint32_t rank) {
  uint32_t ra;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(ra) : "memory");
  return v;
}
__device__ __forceinline__ void wg_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ float ld_cg_f32(const float* p) {
  float v;
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(v) : "l"(p) : "memory");
  return v;
}
template <int R>
__device__ __forceinline__ void setmaxnreg_inc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(R)); }
template <int R>
__device__ __forceinline__ void setmaxnreg_dec() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(R)); }
__device__ __forceinline__ void keep(uint32_t& r) { asm volatile("" : "+r"(r)::"memory"); }
__device__ __forceinline__ void keepf(float& r) { asm volatile("" : "+f"(r)::"memory"); }
__device__ __forceinline__ void named_sync(int id, int threads) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ uint32_t word(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

// K-major 128B-swizzled B tile: start address, LBO 1 (unused when swizzled), SBO 1024 B (8 rows x 128 B).
__device__ __forceinline__ uint64_t make_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1) << 16) |
         (static_cast<uint64_t>(1024 >> 4) << 32) | (static_cast<uint64_t>(1) << 62);
}

// F12M's decode (bf16_proj, proj/f12m/f12m.py): one lane's A fragment of one k16 step, value * 2^(112 - base).
__device__ __forceinline__ uint4 f12m_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}
__device__ __forceinline__ float f12m_scale(uint32_t base) { return __uint_as_float((base + 15u) << 23); }
__device__ __forceinline__ float f12m_wdec(uint32_t base) { return __uint_as_float(base << 23); }

@@WGMMA@@

template <int N>
__device__ __forceinline__ void mma_k64(float (&acc)[N / 2], const uint32_t (&a)[16], uint64_t desc, int first);
#define DGRS_MMA_K64(NN)                                                                            \
  template <>                                                                                       \
  __device__ __forceinline__ void mma_k64<NN>(float (&acc)[NN / 2], const uint32_t (&a)[16], uint64_t desc, \
                                              int first) {                                          \
    wgmma##NN(acc, a[0], a[1], a[2], a[3], desc, first ? 0 : 1);                                    \
    wgmma##NN(acc, a[4], a[5], a[6], a[7], desc + 2, 1);                                            \
    wgmma##NN(acc, a[8], a[9], a[10], a[11], desc + 4, 1);                                          \
    wgmma##NN(acc, a[12], a[13], a[14], a[15], desc + 6, 1);                                        \
  }
DGRS_MMA_K64(32)
DGRS_MMA_K64(64)
DGRS_MMA_K64(128)

// This lane's 16 A registers for one chunk: step j = 4 registers, Q word j of slot 0, L words of slot 1 + (j >> 1).
// NODEC (DIAG 2, lab): the raw words as A (wrong values, same loads).
template <bool NODEC>
__device__ __forceinline__ void decode_frag(const uint8_t* sw, int woff, uint32_t (&a)[16]) {
  const uint4 q = *reinterpret_cast<const uint4*>(sw + woff);
  const uint4 l01 = *reinterpret_cast<const uint4*>(sw + woff + 512);
  const uint4 l23 = *reinterpret_cast<const uint4*>(sw + woff + 1024);
  if constexpr (NODEC) {
    const uint32_t raw[16] = {q.x, q.y, q.z, q.w, l01.x, l01.y, l01.z, l01.w, l23.x, l23.y, l23.z, l23.w, q.x, l01.y, l23.z, q.w};
#pragma unroll
    for (int r = 0; r < 16; ++r) a[r] = raw[r];
    return;
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint4& l = j < 2 ? l01 : l23;
    const uint4 r = f12m_decode(word(q, j), word(l, 2 * (j & 1)), word(l, 2 * (j & 1) + 1));
    a[4 * j] = r.x;
    a[4 * j + 1] = r.y;
    a[4 * j + 2] = r.z;
    a[4 * j + 3] = r.w;
  }
}

__device__ __forceinline__ int ring_next(int s, int S) { return s + 1 == S ? 0 : s + 1; }

// Non-aligned CTA barriers (PTX barrier.sync / barrier.arrive without .aligned): the producer warpgroup and the math
// warpgroups reach them at different instructions (PTX ISA 9.4, barrier{.cta}: arrive releases this thread's prior
// shared-memory writes to the threads that sync on the same barrier).
__device__ __forceinline__ void bar_sync_na(int id, int threads) {
  asm volatile("barrier.sync %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
__device__ __forceinline__ void bar_arrive_na(int id, int threads) {
  asm volatile("barrier.arrive %0, %1;" ::"r"(id), "r"(threads) : "memory");
}
// Split cluster barrier, non-aligned (red 2): arrive (release, or relaxed at kernel start) and wait (acquire).
__device__ __forceinline__ void cl_arrive_relaxed() { asm volatile("barrier.cluster.arrive.relaxed;" ::: "memory"); }
__device__ __forceinline__ void cl_arrive() { asm volatile("barrier.cluster.arrive.release;" ::: "memory"); }
__device__ __forceinline__ void cl_wait() { asm volatile("barrier.cluster.wait.acquire;" ::: "memory"); }
// fp32 store into cluster shared memory (an address from mapa)
__device__ __forceinline__ uint32_t mapa_u32(uint32_t addr, uint32_t rank) {
  uint32_t ra;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(ra) : "r"(addr), "r"(rank));
  return ra;
}
__device__ __forceinline__ void st_cluster_f32(uint32_t addr, float v) {
  asm volatile("st.shared::cluster.f32 [%0], %1;" ::"r"(addr), "f"(v) : "memory");
}
// predicated: the instruction is executed (and its code fetched) either way, the store happens only when `on`
__device__ __forceinline__ void st_cluster_f32_if(uint32_t addr, float v, bool on) {
  asm volatile("{\n.reg .pred q;\nsetp.ne.b32 q, %2, 0;\n@q st.shared::cluster.f32 [%0], %1;\n}\n"
               ::"r"(addr), "f"(v), "r"(static_cast<int>(on)) : "memory");
}
// Lab timers (DIAG 64): %globaltimer (ns, comparable across SMs), %clock (this SM's cycles), %smid
__device__ __forceinline__ uint32_t gtimer_lo() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return static_cast<uint32_t>(t);
}
__device__ __forceinline__ uint32_t clk32() {
  uint32_t c;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(c));
  return c;
}
__device__ __forceinline__ uint32_t smid32() {
  uint32_t v;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(v));
  return v;
}
// 4-byte cp.async (zero-filled when src_bytes = 0), per-thread groups
__device__ __forceinline__ void cp_async4(uint32_t dst, const void* src, uint32_t src_bytes) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::"r"(dst), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

// Shared-memory map (bytes from the 1024-aligned base), the same in every CTA of a launch (DSMEM uses the same offsets
// in the peers): sX [S][BM * 128] | sW [S][12288] | mbarriers 16 * SMAX | sScale [128] f32 | sEsc [PRE_WARPS][ESC_CAP]
// uint2 | sStage [PRE_WARPS][2][GATHER] u32 (DIAG 512: bucketed escapes + bucket ends) | sCorr [BM][CS] f32, CS = RS + 1
// (the rank's rows, odd stride; DIAG 512: 129, all rows) |
// red 2 (push) only: sRecv [splits * NP][BM][RSP] f32, RSP = RS rounded up to 4 mod 8 (conflict-free pushes and
// reads). S = the ring stages that fit.
struct DgrsLayout {
  int S, rs, cs, rsp, push, bars, scale, esc, stage, corr, recv, end;
};
__host__ __device__ inline DgrsLayout dgrs_layout(int bm, int cps, int math, int splits, int red, int smax, int ring = 0) {
  DgrsLayout L;
  const int nr = (red == 1 && splits > 1) ? 1 : splits;
  L.push = red == 2 && splits > 1;
  L.rs = (128 + nr - 1) / nr;
  L.cs = ring ? 129 : L.rs + 1;  // ring corrections (DIAG 512) cover all 128 rows of the slice
  L.rsp = L.rs + ((4 - L.rs) & 7);
  const int stage = bm * 128 + W_CHUNK;
  const int fixed = 16 * SMAX + 128 * 4 + PRE_WARPS * ESC_CAP * 8 + PRE_WARPS * 2 * GATHER * 4;
  const int corr = (bm * L.cs * 4 + 15) / 16 * 16;
  const int recv = L.push ? splits * (math / 2) * bm * L.rsp * 4 : 0;
  const int avail = (cps == 1 ? SMEM_DYN1 : SMEM_DYN2) - 1024 - fixed - corr - recv;
  int S = avail / stage;
  S = S < smax ? S : smax;
  S = S < SMAX ? S : SMAX;
  if (math == 4) S &= ~1;  // even: slot s only ever holds chunks of parity s % 2, i.e. of pair s % 2
  L.S = S;
  L.bars = S * stage;
  L.scale = L.bars + 16 * SMAX;
  L.esc = L.scale + 128 * 4;
  L.stage = L.esc + PRE_WARPS * ESC_CAP * 8;
  L.corr = L.stage + PRE_WARPS * 2 * GATHER * 4;
  L.recv = L.corr + corr;
  L.end = L.recv + recv;
  return L;
}
// launchable: >= 4 ring stages, and (no push) the epilogue's math / 2 fp32 tiles [BM][128] fit in the ring
__host__ __device__ inline bool dgrs_layout_ok(const DgrsLayout& L, int bm, int math) {
  return L.S >= 4 && (L.push || (math / 2) * bm * 128 * 4 <= L.bars);
}

// Producer warps 1-3 (pw = 0..2), during the main loop: scales and escape corrections of this CTA's finalize rows
// [row0, row0 + rn) of out. Warp pw takes rows [pw * RW, ...), RW = ceil(rn / 3): zeroes their corrections, writes
// their scales, lists their escapes in row then escape order (warp scan of the counts). Then (v6) lane = token gathers
// x with cp.async, G = GATHER / BM list entries per batch, double-buffered (no registers held while loads are in
// flight): the 4-byte word holding x[t][k] (k & ~1) of entry e, token t lands at stage[(buf G + e - b) BM + t]; then
// corr[t][ri] = fma(x, true, fma(-x, wdec, corr[t][ri])) per escape of row ri, in list order.
template <int BM>
__device__ __forceinline__ void pre_corr(const uint4* __restrict__ rec, const uint16_t* __restrict__ xh,
                                         float* sScale, uint2* sEsc, uint32_t* sStage, float* sCorr, int row0, int rn,
                                         int cs, int tok0, int M, int K, int pw, int lane) {
  constexpr int G = GATHER / BM;
  const int RW = (rn + PRE_WARPS - 1) / PRE_WARPS;
  const int wlo = pw * RW;
  const int whi = wlo + RW < rn ? wlo + RW : rn;
  uint2* list = sEsc + pw * ESC_CAP;
#pragma unroll 1
  for (int t = 0; t < BM; ++t)
    for (int ri = wlo + lane; ri < whi; ri += 32) sCorr[t * cs + ri] = 0.f;
  int ne = 0;
  for (int r0 = wlo; r0 < whi; r0 += 32) {
    const int ri = r0 + lane;
    const bool v = ri < whi;
    const uint4 q0 = v ? rec[2 * (row0 + ri)] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t base = q0.x & 0xFFu;
    uint32_t cnt = (q0.x >> 8) & 0xFFu;
    cnt = cnt < 7u ? cnt : 7u;
    const uint4 q1 = cnt > 3u ? rec[2 * (row0 + ri) + 1] : make_uint4(0u, 0u, 0u, 0u);
    if (v) sScale[ri] = f12m_scale(base);
    int inc = static_cast<int>(cnt);
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int nb = __shfl_up_sync(0xFFFFFFFFu, inc, o);
      if (lane >= o) inc += nb;
    }
    const int off = ne + inc - static_cast<int>(cnt);
    const uint32_t tag = static_cast<uint32_t>(ri) | (base << 8);
#pragma unroll
    for (int e = 0; e < 7; ++e)
      if (e < static_cast<int>(cnt)) list[off + e] = make_uint2(e < 3 ? word(q0, e + 1) : word(q1, e - 3), tag);
    ne += __shfl_sync(0xFFFFFFFFu, inc, 31);
  }
  __syncwarp();
  uint32_t* stg = sStage + pw * (2 * GATHER);
  const uint32_t stg0 = smem_u32(stg);
  auto issue = [&](int b, int buf) {
    const int be = b + G < ne ? b + G : ne;
#pragma unroll 1
    for (int e = b; e < be; ++e) {
      const uint32_t k2 = list[e].x & 0xFFFEu;  // the 4-byte word holding element k
#pragma unroll
      for (int v2 = 0; v2 < BM / 32; ++v2) {
        const int t = lane + 32 * v2;
        const bool live = tok0 + t < M;
        cp_async4(stg0 + static_cast<uint32_t>(((buf * G + e - b) * BM + t) * 4),
                  xh + static_cast<size_t>(live ? tok0 + t : 0) * K + k2, live ? 4u : 0u);
      }
    }
    cp_async_commit();
  };
  if (ne > 0) issue(0, 0);
  int buf = 0;
#pragma unroll 1
  for (int b = 0; b < ne; b += G) {
    if (b + G < ne) {
      issue(b + G, buf ^ 1);
      cp_async_wait<1>();  // batch b landed (this lane's own words)
    } else {
      cp_async_wait<0>();
    }
    const int be = b + G < ne ? b + G : ne;
#pragma unroll 1
    for (int e = b; e < be; ++e) {
      const uint2 d = list[e];
      const float wdec = f12m_wdec(d.y >> 8), wtrue = __uint_as_float(d.x & 0xFFFF0000u);
#pragma unroll
      for (int v2 = 0; v2 < BM / 32; ++v2) {
        const int t = lane + 32 * v2;
        const uint32_t wv = stg[(buf * G + e - b) * BM + t];
        const float xv = __uint_as_float((d.x & 1u) ? (wv & 0xFFFF0000u) : (wv << 16));
        float* cp = sCorr + t * cs + (d.y & 0xFFu);
        float c = *cp;
        c = fmaf(-xv, wdec, c);
        c = fmaf(xv, wtrue, c);
        *cp = c;
      }
    }
    buf ^= 1;
  }
}

// v9 (DIAG 512): producer warps 1-3 as a third ring consumer. Warp pw takes rows [43 pw, ...) of all 128 rows of the
// row block: scales (absolute row index), and the escapes whose k falls in this slice's chunks [kc0, kc0 + nk), listed
// in row then record order, then bucketed by chunk (deterministic: warp match_any ranks keep list order) into
// buck[]; ends[c] = end of bucket c. Then for each chunk i, in order: wait for slot i % S to land, read x[t][k] for the
// bucket's escapes straight from the swizzled x tile (token t, element kk: t * 128 + (((kk >> 3) ^ (t & 7)) << 4) +
// (kk & 7) * 2; rows >= M are TMA zero-fill), corr[t][row] = fma(x, true, fma(-x, wdec, corr)), and release the slot
// (empty count 2 + PRE_WARPS). No global gathers. The math warps fold corr * 2^(112 - base) into the partials.
template <int BM>
__device__ __forceinline__ bool pre_ring(const uint4* __restrict__ rec, float* sScale, uint2* sEsc, uint32_t* sStage,
                                         float* sCorr, const uint8_t* sX, int row0, int cs, int kc0, int nk, int S,
                                         uint32_t full0, uint32_t empty0, int* err, int pw, int lane) {
  constexpr int RWR = (128 + PRE_WARPS - 1) / PRE_WARPS;  // 43
  constexpr int X_BYTES = BM * 128;
  const int wlo = pw * RWR;
  const int whi = wlo + RWR < 128 ? wlo + RWR : 128;
  uint2* list = sEsc + pw * ESC_CAP;
  uint2* buck = reinterpret_cast<uint2*>(sStage + pw * (2 * GATHER));
  int* ends = reinterpret_cast<int*>(buck + ESC_CAP);
  bool ok = true;
#pragma unroll 1
  for (int t = 0; t < BM; ++t)
    for (int ri = wlo + lane; ri < whi; ri += 32) sCorr[t * cs + ri] = 0.f;
  for (int c = lane; c < nk; c += 32) ends[c] = 0;
  int ne = 0;
  for (int r0 = wlo; r0 < whi; r0 += 32) {
    const int ri = r0 + lane;
    const bool v = ri < whi;
    const uint4 q0 = v ? rec[2 * (row0 + ri)] : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t base = q0.x & 0xFFu;
    uint32_t cnt = (q0.x >> 8) & 0xFFu;
    cnt = cnt < 7u ? cnt : 7u;
    const uint4 q1 = cnt > 3u ? rec[2 * (row0 + ri) + 1] : make_uint4(0u, 0u, 0u, 0u);
    if (v) sScale[ri] = f12m_scale(base);
    int m = 0;
#pragma unroll
    for (int e = 0; e < 7; ++e) {
      const int ch = static_cast<int>(((e < 3 ? word(q0, e + 1) : word(q1, e - 3)) & 0xFFFFu) >> 6) - kc0;
      if (e < static_cast<int>(cnt) && ch >= 0 && ch < nk) ++m;
    }
    int inc = m;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int nb = __shfl_up_sync(0xFFFFFFFFu, inc, o);
      if (lane >= o) inc += nb;
    }
    int off = ne + inc - m;
    const uint32_t tag = static_cast<uint32_t>(ri) | (base << 8);
#pragma unroll
    for (int e = 0; e < 7; ++e) {
      const uint32_t wd = e < 3 ? word(q0, e + 1) : word(q1, e - 3);
      const int ch = static_cast<int>((wd & 0xFFFFu) >> 6) - kc0;
      if (e < static_cast<int>(cnt) && ch >= 0 && ch < nk) list[off++] = make_uint2(wd, tag);
    }
    ne += __shfl_sync(0xFFFFFFFFu, inc, 31);
  }
  __syncwarp();
  // bucket counts (match_any groups of equal chunk; the group's lowest lane adds the group size)
  for (int b = 0; b < ne; b += 32) {
    const int e = b + lane;
    const int ch = e < ne ? static_cast<int>((list[e].x & 0xFFFFu) >> 6) - kc0 : -1;
    const unsigned grp = __match_any_sync(0xFFFFFFFFu, ch);
    if (ch >= 0 && lane == __ffs(grp) - 1) ends[ch] += __popc(grp);
    __syncwarp();
  }
  // exclusive scan of the counts in place (lane: 8 consecutive chunks; nk <= 256, host-checked)
  {
    int vals[8];
    int tot = 0;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int c = 8 * lane + u;
      vals[u] = c < nk ? ends[c] : 0;
      tot += vals[u];
    }
    int inc = tot;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const int nb = __shfl_up_sync(0xFFFFFFFFu, inc, o);
      if (lane >= o) inc += nb;
    }
    int run = inc - tot;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const int c = 8 * lane + u;
      if (c < nk) ends[c] = run;  // start of bucket c (the scatter turns it into the end)
      run += vals[u];
    }
  }
  __syncwarp();
  // scatter in list order: rank within the chunk group = lanes below with the same chunk
  for (int b = 0; b < ne; b += 32) {
    const int e = b + lane;
    const uint2 d = e < ne ? list[e] : make_uint2(0u, 0u);
    const int ch = e < ne ? static_cast<int>((d.x & 0xFFFFu) >> 6) - kc0 : -1;
    const unsigned grp = __match_any_sync(0xFFFFFFFFu, ch);
    const int pos = ch >= 0 ? ends[ch] + __popc(grp & ((1u << lane) - 1u)) : 0;
    __syncwarp();
    if (ch >= 0) {
      buck[pos] = d;
      if (lane == __ffs(grp) - 1) ends[ch] += __popc(grp);
    }
    __syncwarp();
  }
  // per chunk, in ring order
#pragma unroll 1
  for (int i = 0; i < nk; ++i) {
    const int s = i % S;
    ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);
    const uint8_t* xs = sX + s * X_BYTES;
    const int lo = i > 0 ? ends[i - 1] : 0, hi = ends[i];
    const int k0 = (kc0 + i) * 64;
#pragma unroll 1
    for (int e = lo; e < hi; ++e) {
      const uint2 d = buck[e];
      const int kk = static_cast<int>(d.x & 0xFFFFu) - k0;
      const float wdec = f12m_wdec(d.y >> 8), wtrue = __uint_as_float(d.x & 0xFFFF0000u);
#pragma unroll
      for (int v2 = 0; v2 < BM / 32; ++v2) {
        const int t = lane + 32 * v2;
        const uint16_t hx = *reinterpret_cast<const uint16_t*>(xs + t * 128 + ((((kk >> 3) ^ (t & 7)) & 7) << 4) + (kk & 7) * 2);
        const float xv = __uint_as_float(static_cast<uint32_t>(hx) << 16);
        float* cp = sCorr + t * cs + (d.y & 0xFFu);
        float c = *cp;
        c = fmaf(-xv, wdec, c);
        c = fmaf(xv, wtrue, c);
        *cp = c;
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty0 + 8 * s);  // this warp is done with chunk i's slot
  }
  return ok;
}

// grid (N / 128, token tiles, splits), 128 + 128 * MATH threads, cluster (1, 1, splits) when splits > 1 and red 0 / 2.
// wp: chunk-major planes [K/64][N/16][3][32][4] int32; rec [N][8] int32 (base | count << 8, then k | true bits << 16);
// xmap: x [M][K] bf16 (box 64 x BM, SWIZZLE_128B); x itself for the escapes; out [M][N] bf16.
// MATH = 2 or 4 math warpgroups; with 4, two pairs (warpgroups 2p + 1, 2p + 2 = pair p) take alternate chunks of the
// same 128 rows (pair p: chunks p, p + 2, ...; even ring, so a slot always serves the same pair), each into its own
// fp32 tile, summed (pair 0 + pair 1) before the finalize. setmaxnreg: producer 24, math 112 (F57: 24 * 128 + 112 * 512
// <= 96 * 640 = the CTA's launch allocation).
// v5: the producer warpgroup never runs the epilogue (it may hold only 24 registers): thread 0 issues the copies,
// warps 1-3 precompute the finalize rows' scales and escape corrections (pre_corr) while the math runs; the math
// warpgroups alone write the tiles, reduce split-K and finalize = sum of slices * scale + corr from shared memory.
// RED >= 0 / SPL > 0 (v9): an instance specialized for one reduction mode / split count; the constants fold the
// epilogue's control flow and index math (row share, thread map, slot loop) and drop the other modes' code.
template <int BM, int CPS, int DIAG, int MATH, int RED = -1, int SPL = 0, int SMX = 0, int MM = 0, int NN = 0, int KK = 0>
__global__ void __launch_bounds__(128 + 128 * MATH, CPS) rs_kernel(const __grid_constant__ CUtensorMap xmap,
                                                      const uint8_t* __restrict__ wp, const uint4* __restrict__ rec,
                                                      const __nv_bfloat16* __restrict__ x,
                                                      __nv_bfloat16* __restrict__ out, float* __restrict__ part,
                                                      int* __restrict__ cnt, int* __restrict__ err, int M, int N, int K,
                                                      int splits, int smax, int red) {
  static_assert(CPS == 1 || BM <= 64, "2 CTAs per SM only for BM <= 64");
  static_assert(MATH == 2 || (MATH == 4 && CPS == 1 && BM <= 64), "4 math warpgroups: 1 CTA per SM, BM <= 64");
  static_assert(MATH == 2 || 24 * 128 + 112 * 512 <= 96 * 640, "setmaxnreg split must fit the launch allocation (F57)");
  static_assert(RED >= -1 && RED <= 2 && SPL >= 0 && SPL <= 8, "specialization: RED -1 / 0..2, SPL 0 / 1..8");
  static_assert(SMX >= 0 && MM >= 0 && NN % 128 == 0 && KK % 64 == 0, "specialization: SMX, MM, NN, KK >= 0 (0 = runtime)");
  if constexpr (SPL > 0) splits = SPL;  // dgrs_gemm picks this instance only when splits == SPL
  if constexpr (RED >= 0) red = RED;    // ... and red == RED
  if constexpr (SMX > 0) smax = SMX;    // v10: ring depth, so S and every i % S, i / S fold
  if constexpr (MM > 0) M = MM;         // v10: the layer's shape (dgrs_gemm matches M, N, K exactly)
  if constexpr (NN > 0) N = NN;
  if constexpr (KK > 0) K = KK;
  constexpr int THREADS = 128 + 128 * MATH;
  constexpr int TM = 128 * MATH;   // math threads: the epilogue is theirs alone
  constexpr int NP = MATH / 2;     // warpgroup pairs taking alternate chunks
  constexpr int FU = 16 / MATH;    // tokens per finalize step
  const int rb = blockIdx.x, tt = blockIdx.y, sl = blockIdx.z;
  const int KC = K / 64;
  const int per = (KC + splits - 1) / splits;
  const int kc0 = sl * per;
  const int nk = min(per, KC - kc0);  // >= 1 (host check: (splits - 1) * per < KC)
  constexpr int X_BYTES = BM * 128;
  constexpr uint32_t TX = ((DIAG & 8) ? 0u : W_CHUNK) + ((DIAG & 4) ? 0u : X_BYTES);
  constexpr bool RING = (DIAG & 512) != 0;  // v9: escape corrections from the ring's x tiles (pre_ring), folded into acc
  static_assert(!RING || (DIAG & 32) == 0, "DIAG 512 needs the producer warps as ring consumers");
  // splits > 1: red 0 = cluster, peers' tiles pulled over DSMEM; red 1 = plain launch, global fp32 partials +
  // last-arriver counter; red 2 = cluster, each slice pushes its accumulators into the owner rank's receive area
  const bool glob = red == 1 && splits > 1;
  const DgrsLayout L = dgrs_layout(BM, CPS, MATH, splits, red, smax, RING ? 1 : 0);
  const bool push = L.push;
  const int S = L.S;
  // this CTA's finalize rows [rlo, rlo + rn) of the row block: rank sl of nr (red 1: all 128, by the last arriver)
  const int rank = glob ? 0 : sl;
  const int rlo = rank * L.rs;
  const int rhi = rlo + L.rs < 128 ? rlo + L.rs : 128;
  const int rn = (128 % L.rs == 0) ? L.rs : (rhi > rlo ? rhi - rlo : 0);  // equal shares fold to a constant
  extern __shared__ uint8_t smem_raw[];
  const uint32_t raw = smem_u32(smem_raw);
  uint8_t* smem = smem_raw + ((1024u - (raw & 1023u)) & 1023u);
  uint8_t* sX = smem;                  // S x BM*128 (1024-aligned: BM % 8 == 0)
  uint8_t* sW = smem + S * X_BYTES;    // S x 12288
  uint64_t* bars = reinterpret_cast<uint64_t*>(smem + L.bars);
  float* sScale = reinterpret_cast<float*>(smem + L.scale);
  uint2* sEsc = reinterpret_cast<uint2*>(smem + L.esc);
  uint32_t* sStage = reinterpret_cast<uint32_t*>(smem + L.stage);
  float* sCorr = reinterpret_cast<float*>(smem + L.corr);
  float* sRecv = reinterpret_cast<float*>(smem + L.recv);
  const uint32_t full0 = smem_u32(bars), empty0 = smem_u32(bars + SMAX);
  const int tid = threadIdx.x;
  // DIAG 64 (lab): 16 u32 per CTA in `part` (red 0 / 2 only): 0 globaltimer at entry, 1 clock at entry, 2 producer's
  // last copy issued, 3 pre_corr done, 4 math loop end (pair 0), 5 reduced (after push / pull barrier), 6 bar 2 passed,
  // 7 finalize done, 8 exit, 9 globaltimer at exit, 10 SM id, 11 math loop end (pair 1, MATH 4), 12 dry pass done
  // (DIAG 256), 13 pushes issued (red 2), 14 finalize loads summed (first step); clocks are this SM's
  constexpr bool STAMP = (DIAG & 64) != 0;
  constexpr bool WARM = (DIAG & 256) != 0;  // dry pass of the epilogue code before the main loop (red 2 only)
  static_assert(!WARM || (DIAG & 17) == 0, "DIAG 256 warms the epilogue: needs the finalize");
  uint32_t* stamps = reinterpret_cast<uint32_t*>(part) +
                     16 * ((static_cast<size_t>(blockIdx.z) * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x);
  if (STAMP && tid == 0) {
    stamps[0] = gtimer_lo();
    stamps[1] = clk32();
    stamps[10] = smid32();
  }
  if (!(DIAG & 128) && tid == 0)  // the x tensor map's descriptor, before the first TMA (CUTLASS does the same)
    asm volatile("prefetch.tensormap [%0];" ::"l"(reinterpret_cast<uint64_t>(&xmap)) : "memory");
  if (!dgrs_layout_ok(L, BM, MATH)) {  // dgrs_gemm rejects these launches; never taken
    if (tid == 0) atomicAdd(err, 1);
    return;
  }
  if (tid == 0) {
    for (int s = 0; s < S; ++s) {
      mbar_init(full0 + 8 * s, 1);  // thread 0's expect_tx arrive (bytes: the weight copy + the x TMA)
      mbar_init(empty0 + 8 * s, RING ? 2 + PRE_WARPS : 2);  // both math warpgroups (+ the producer warps 1-3, DIAG 512)
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if ((DIAG & 17) == 0 && push) cl_arrive_relaxed();  // red 2 phase A: waited before the first push (CTAs started)
  bool ok = true;
  float* tile = reinterpret_cast<float*>(smem);  // epilogue: NP fp32 tiles [BM][128] in the ring (free by then)
  if (tid < 128) {
    if (MATH == 4) setmaxnreg_dec<24>();
    // ---- producer: thread 0 issues the copies; warps 1-3 precompute the finalize rows' scales and corrections
    if (tid == 0) {
      const uint8_t* wsrc = wp + (static_cast<size_t>(kc0) * (N / 16) + static_cast<size_t>(rb) * 8) * 1536;
      const size_t wstride = static_cast<size_t>(N / 16) * 1536;  // one chunk of all rows
      int s = 0;
      uint32_t round = 0;
      for (int i = 0; i < nk; ++i) {
        if (round > 0) ok &= mbar_wait(empty0 + 8 * s, (round - 1) & 1u, err);
        mbar_expect_tx(full0 + 8 * s, TX);
        if (!(DIAG & 8))
          bulk_g2s(smem_u32(sW + s * W_CHUNK), wsrc + static_cast<size_t>(i) * wstride, W_CHUNK, full0 + 8 * s);
        if (!(DIAG & 4)) tma_2d(smem_u32(sX + s * X_BYTES), &xmap, (kc0 + i) * 64, tt * BM, full0 + 8 * s);
        if (++s == S) {
          s = 0;
          ++round;
        }
      }
      if (STAMP) stamps[2] = clk32();
    } else if (RING && tid >= 32) {
      ok &= pre_ring<BM>(rec, sScale, sEsc, sStage, sCorr, sX, rb * 128, L.cs, kc0, nk, S, full0, empty0, err,
                         (tid >> 5) - 1, tid & 31);
      if (STAMP && tid == 32) stamps[3] = clk32();
    } else if (tid >= 32 && (DIAG & 49) == 0) {
      pre_corr<BM>(rec, reinterpret_cast<const uint16_t*>(x), sScale, sEsc, sStage, sCorr, rb * 128 + rlo, rn, L.cs,
                   tt * BM, M, K, (tid >> 5) - 1, tid & 31);
      if (STAMP && tid == 32) stamps[3] = clk32();
    }
    if constexpr ((DIAG & 17) == 0) {
      __syncwarp();
      bar_arrive_na(2, THREADS);  // this CTA's scales and corrections are in shared memory
      if (push) {  // red 2: phase A, then the math warpgroups' push barrier B
        cl_wait();
        cl_arrive();
        cl_wait();
      } else if (splits > 1 && !glob) {  // red 0: the math warpgroups' two cluster barriers
        cluster_sync();
        cluster_sync();
      }
    }
  } else {
    if (MATH == 4) setmaxnreg_inc<112>();
    // ---- math warpgroup mw = pair p (chunks p, p + NP, ...), row half c; warp w owns row tile 4c + w of the CTA's 8
    const int mw = (tid >> 7) - 1, c = mw & 1, p = mw >> 1;
    const int w = (tid >> 5) & 3, lane = tid & 31, g = lane >> 2, q = lane & 3;
    const int woff = ((4 * c + w) * 3 * 32 + lane) * 16;  // slot 0 of this lane; slots 1, 2 at +512, +1024
    float acc[BM / 2];
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) acc[r] = 0.f;
    uint32_t fa[16], fb[16], fc[16];
    const uint32_t sX0 = smem_u32(sX);
    // chunk i sits in slot i % S, parity (i / S) & 1; at this pair's step for chunk i, after wait_group<1>, its
    // previous chunk i - NP is done: release that slot
    auto load = [&](int i, uint32_t (&f)[16]) {
      const int s = i % S;
      ok &= mbar_wait(full0 + 8 * s, static_cast<uint32_t>((i / S) & 1), err);
      decode_frag<(DIAG & 2) != 0>(sW + s * W_CHUNK, woff, f);
    };
    auto step = [&](int i, const uint32_t (&cur)[16], uint32_t (&nxt)[16]) {
      wg_fence();
      mma_k64<BM>(acc, cur, make_desc(sX0 + (i % S) * X_BYTES), i == p);
      wg_commit();
      wg_wait<1>();  // this pair's chunk i - NP is done: its slot and its fragment buffer are free
#pragma unroll
      for (int r = 0; r < 16; ++r) keep(nxt[r]);
      if (i >= NP && (tid & 127) == 0) mbar_arrive(empty0 + 8 * ((i - NP) % S));
      if (i + NP < nk) load(i + NP, nxt);
    };
    __shared__ int s_last;
    // The body runs once (real). DIAG 256 (v8, red 2 only): first a dry pass before the main loop: no math, DSMEM
    // stores predicated off, no finalize stores, no barriers; it executes the push and finalize code once while the
    // ring's first copies are in flight, so their instructions are cached when the real pass reaches them (PE35:
    // finalize 1.55 us for ~24 loads, shape-independent).
    auto body = [&](const bool real) {
    if (real && p < nk) {
      load(p, fa);
      int i = p;
      for (; i + 2 * NP < nk; i += 3 * NP) {
        step(i, fa, fb);
        step(i + NP, fb, fc);
        step(i + 2 * NP, fc, fa);
      }
      if (i < nk) step(i, fa, fb);
      if (i + NP < nk) step(i + NP, fb, fc);
      wg_wait<0>();
    }
#pragma unroll
    for (int r = 0; r < BM / 2; ++r) keepf(acc[r]);
    if (STAMP && real && (tid == 128 || tid == 384)) stamps[tid == 128 ? 4 : 11] = clk32();
    if constexpr (RING && (DIAG & 17) == 0) {
      if (real) {
        bar_sync_na(2, THREADS);  // DIAG 512: the producer warps' corrections of this slice's escapes (all 128 rows)
        if (p == 0) {             // fold once per CTA: acc += corr * 2^(112 - base) (exact power-of-two reciprocal)
          const int rf = 64 * c + 16 * w + g;
          const float i0 = __frcp_rn(sScale[rf]), i8 = __frcp_rn(sScale[rf + 8]);
#pragma unroll
          for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
              const int t = 8 * j + 2 * q + h;
              acc[4 * j + h] = fmaf(sCorr[t * L.cs + rf], i0, acc[4 * j + h]);
              acc[4 * j + 2 + h] = fmaf(sCorr[t * L.cs + rf + 8], i8, acc[4 * j + 2 + h]);
            }
          }
        }
      }
    }
    if constexpr ((DIAG & 16) != 0) {
      // lab: keep the math (a store ptxas can't drop), skip the finalize
      float sacc = 0.f;
#pragma unroll
      for (int r = 0; r < BM / 2; ++r) sacc += acc[r];
      if (__float_as_uint(sacc) == 0x7F8FEDCBu) out[tid] = __float2bfloat16_rn(sacc);
    } else if constexpr ((DIAG & 1) == 0) {
      // ---- epilogue (math threads only): split-K reduction, then finalize
      const int mt = tid - 128;
      const int r0 = 64 * c + 16 * w + g;  // rows r0, r0 + 8; tokens 8j + 2q (+1)
      const int Mp = gridDim.y * BM;
      if (push) {
        // red 2: this slice's accumulators go straight into the owner rank's (row / RS) receive slot sl * NP + p;
        // one cluster barrier; every rank then finalizes from its own shared memory (no pulls, no exit barrier)
        if (real) cl_wait();  // phase A: every CTA of the cluster is running (its receive area exists)
        const int ra = r0 / L.rs, rb8 = (r0 + 8) / L.rs;
        const int slot = (sl * NP + p) * BM * L.rsp;
        const uint32_t recv0 = smem_u32(sRecv);
        const uint32_t ga = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 - ra * L.rs) * 4), static_cast<uint32_t>(ra));
        const uint32_t gb = mapa_u32(recv0 + static_cast<uint32_t>((slot + r0 + 8 - rb8 * L.rs) * 4), static_cast<uint32_t>(rb8));
#pragma unroll
        for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t t = static_cast<uint32_t>(8 * j + 2 * q + h);
            st_cluster_f32_if(ga + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + h], real);
            st_cluster_f32_if(gb + t * static_cast<uint32_t>(L.rsp * 4), acc[4 * j + 2 + h], real);
          }
        }
        if (real) {
          if (STAMP && tid == 128) stamps[13] = clk32();
          cl_arrive();  // phase B: release this CTA's pushes ...
          cl_wait();    // ... and acquire every slice's
          if (STAMP && tid == 128) stamps[5] = clk32();
        }
      } else if (real) {
        // red 0 / 1 / S = 1: the CTA's fp32 tile [BM][128] in the (free) ring, pairs summed in order
        named_sync(1, TM);  // every math warpgroup is done with the ring (all chunks landed and were read)
        float* mtile = tile + p * (BM * 128);
#pragma unroll
        for (int j = 0; j < BM / 8; ++j) {
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int t = 8 * j + 2 * q + h;
            mtile[t * 128 + r0] = acc[4 * j + h];
            mtile[t * 128 + r0 + 8] = acc[4 * j + 2 + h];
          }
        }
        named_sync(1, TM);
        if (MATH == 4) {
          for (int idx = mt; idx < BM * 128; idx += TM) tile[idx] += tile[BM * 128 + idx];
          named_sync(1, TM);
        }
        if (glob) {
          for (int idx = mt; idx < BM * 128; idx += TM)
            part[(static_cast<size_t>(sl) * Mp + tt * BM + (idx >> 7)) * N + rb * 128 + (idx & 127)] = tile[idx];
          __threadfence();  // release: this thread's partials before the counter
          named_sync(1, TM);
          if (mt == 0) {
            __threadfence();
            const int last = atomicAdd(cnt + tt * gridDim.x + rb, 1) == splits - 1;
            if (last) {
              __threadfence();  // acquire: every slice's partials
              atomicExch(cnt + tt * gridDim.x + rb, 0);  // self-reset: no other CTA of this call touches it again
            }
            s_last = last;
          }
          named_sync(1, TM);
        } else if (splits > 1) {
          cluster_sync();  // every slice's tile is written (release / acquire across the cluster)
        }
        if (STAMP && tid == 128) stamps[5] = clk32();
      }
      if (real && !RING) {
        bar_sync_na(2, THREADS);  // the producer warps' scales and corrections
        if (STAMP && tid == 128) stamps[6] = clk32();
      }
      // thread -> (row ri, token lane tl): sum the slices in order, * scale + corr, BF16 (rows contiguous)
      const int TL = rn > 0 ? TM / rn : 0;
      if (rn > 0 && mt < TL * rn && (!glob || s_last)) {
        const int ri = mt % rn, tl = mt / rn;
        const int row = rb * 128 + rlo + ri;
        const float scale = sScale[RING ? rlo + ri : ri];
        const uint32_t tile0 = smem_u32(tile);
        for (int t0 = tl; t0 < BM; t0 += TL * FU) {
          int tok[FU];
          bool live[FU];
          float sum[FU];
#pragma unroll
          for (int u = 0; u < FU; ++u) {
            const int t = t0 + TL * u;
            tok[u] = tt * BM + t;
            live[u] = t < BM && tok[u] < M;
            sum[u] = 0.f;
          }
          const int ns = push ? splits * NP : splits;
          for (int s2 = 0; s2 < ns; ++s2) {  // slice (then pair) order: deterministic
            float v[FU];
#pragma unroll
            for (int u = 0; u < FU; ++u) {
              const uint32_t a = tile0 + static_cast<uint32_t>(((t0 + TL * u) * 128 + rlo + ri) * 4);
              v[u] = !live[u] ? 0.f
                     : push ? sRecv[(s2 * BM + t0 + TL * u) * L.rsp + ri]
                     : glob ? ld_cg_f32(part + (static_cast<size_t>(s2) * Mp + tok[u]) * N + row)
                     : splits > 1 ? ld_peer_f32(a, static_cast<uint32_t>(s2))
                                  : tile[(t0 + TL * u) * 128 + rlo + ri];
            }
#pragma unroll
            for (int u = 0; u < FU; ++u) sum[u] += v[u];
          }
          if (STAMP && real && tid == 128 && t0 == tl) stamps[14] = clk32();
#pragma unroll
          for (int u = 0; u < FU; ++u)
            if (live[u] && real)
              out[static_cast<size_t>(tok[u]) * N + row] =
                  __float2bfloat16_rn(fmaf(sum[u], scale, RING ? 0.f : sCorr[(t0 + TL * u) * L.cs + ri]));
        }
      }
      if (real) {
        if (STAMP && tid == 128) stamps[7] = clk32();
        if (splits > 1 && red == 0) cluster_sync();  // peers may still read this CTA's tile
      }
    }
    };
    if constexpr (WARM) {
#pragma unroll 1
      for (int pass = 0; pass < 2; ++pass) {
        asm volatile("" : "+r"(pass));  // opaque: one copy of the body, executed twice
        body(pass == 1);
        if (STAMP && pass == 0 && tid == 128) stamps[12] = clk32();
      }
    } else {
      body(true);
    }
  }
  if (!ok) atomicAdd(err, 1);
  if (STAMP && tid == 128) {
    stamps[8] = clk32();
    stamps[9] = gtimer_lo();
  }
}

}  // namespace dgrs

typedef CUresult (*DgrsEncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static DgrsEncodeFn dgrs_encode_fn() {  // the driver's cuTensorMapEncodeTiled, without linking libcuda (as dg12.py)
  static DgrsEncodeFn fn = nullptr;
  if (fn == nullptr) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (h != nullptr) fn = reinterpret_cast<DgrsEncodeFn>(dlsym(h, "cuTensorMapEncodeTiled"));
  }
  return fn;
}

// x [M][K] bf16 -> tensor map with box 64 (inner, 128 B) x BM rows, 128B swizzle; rows >= M read as zeros.
static void dgrs_xmap(CUtensorMap* map, const void* x, int M, int K, int bm) {
  DgrsEncodeFn fn = dgrs_encode_fn();
  TORCH_CHECK(fn != nullptr, "dgrs: cuTensorMapEncodeTiled not found");
  const cuuint64_t gdims[2] = {static_cast<cuuint64_t>(K), static_cast<cuuint64_t>(M)};
  const cuuint64_t gstrides[1] = {static_cast<cuuint64_t>(K) * 2};
  const cuuint32_t box[2] = {64, static_cast<cuuint32_t>(bm)};
  const cuuint32_t estr[2] = {1, 1};
  const CUresult r = fn(map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(x), gdims, gstrides, box, estr,
                        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "dgrs: cuTensorMapEncodeTiled failed: ", static_cast<int>(r));
}

template <int BM, int CPS, int DIAG, int MATH, int RED, int SPL, int SMX, int MM, int NN, int KK>
static void dgrs_launch(dim3 grid, int splits, int red, const CUtensorMap& xmap, const uint8_t* w, const uint4* rc,
                        const __nv_bfloat16* xb, __nv_bfloat16* o, float* part, int* cnt, int* err, int M, int N, int K,
                        int smax) {
  auto kern = dgrs::rs_kernel<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>;
  const size_t smem = static_cast<size_t>(CPS == 1 ? dgrs::SMEM_DYN1 : dgrs::SMEM_DYN2);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(128 + 128 * MATH, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = 1;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = static_cast<unsigned>(splits);
  cfg.attrs = attr;
  cfg.numAttrs = (splits > 1 && red != 1) ? 1 : 0;  // a real cluster only for DSMEM split-K (PF41: size 1 costs ~30%)
  const cudaError_t e = cudaLaunchKernelEx(&cfg, kern, xmap, w, rc, xb, o, part, cnt, err, M, N, K, splits, smax, red);
  if (e != cudaSuccess) (void)cudaGetLastError();
  TORCH_CHECK(e == cudaSuccess, "dgrs launch failed: ", cudaGetErrorString(e));
}

void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red, int64_t math) {
  const int KC = static_cast<int>(K / 64);
  const int per = static_cast<int>((KC + splits - 1) / splits);
  const int tiles = static_cast<int>((M + bm - 1) / bm);
  TORCH_CHECK(K % 64 == 0 && N % 128 == 0 && splits >= 1 && splits <= 8 && (splits - 1) * per < KC,
              "dgrs_gemm: shapes / splits (1..8, every slice non-empty)");
  TORCH_CHECK(red >= 0 && red <= 2, "dgrs_gemm: red 0 (pull), 1 (global), 2 (push)");
  TORCH_CHECK(!(diag & 512) || (per <= 256 && !(diag & 32)), "dgrs_gemm: DIAG 512 needs <= 256 chunks per slice, no DIAG 32");
  TORCH_CHECK(!(diag & 256) || (red == 2 && splits > 1), "dgrs_gemm: DIAG 256 (epilogue warm-up) needs red 2, splits > 1");
  TORCH_CHECK(!(diag & 64) || (!(red == 1 && splits > 1) && part.numel() >= 16 * (N / 128) * tiles * splits),
              "dgrs_gemm: DIAG 64 stamps go to part (red 0 / 2, 16 words per CTA)");
  TORCH_CHECK(red != 1 || splits == 1 || (part.numel() >= splits * static_cast<int64_t>(tiles) * bm * N &&
                                           cnt.numel() >= (N / 128) * tiles), "dgrs_gemm: red 1 needs part / cnt");
  const dgrs::DgrsLayout L = dgrs::dgrs_layout(static_cast<int>(bm), static_cast<int>(cps), static_cast<int>(math),
                                               static_cast<int>(splits), static_cast<int>(red), static_cast<int>(smax),
                                               (diag & 512) ? 1 : 0);
  TORCH_CHECK(dgrs::dgrs_layout_ok(L, static_cast<int>(bm), static_cast<int>(math)), "dgrs_gemm: shared memory gives ",
              L.S, " ring stages (need >= 4 and room for the epilogue tiles)");
  const dim3 grid(static_cast<unsigned>(N / 128), static_cast<unsigned>(tiles), static_cast<unsigned>(splits));
  CUtensorMap xmap;
  dgrs_xmap(&xmap, x.data_ptr(), static_cast<int>(M), static_cast<int>(K), static_cast<int>(bm));
  const uint8_t* w = reinterpret_cast<const uint8_t*>(wp.data_ptr());
  const uint4* rc = reinterpret_cast<const uint4*>(rec.data_ptr());
  const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* o = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
#define DGRS_CASE(BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK)                                              \
  if (bm == BM && cps == CPS && diag == DIAG && math == MATH && (RED < 0 || red == RED) &&                    \
      (SPL == 0 || splits == SPL) && (SMX == 0 || smax == SMX) && (MM == 0 || M == MM) &&                     \
      (NN == 0 || N == NN) && (KK == 0 || K == KK)) {                                                         \
    dgrs_launch<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>(grid, static_cast<int>(splits), static_cast<int>(red), xmap, w, rc, xb, o,      \
                               part.data_ptr<float>(), cnt.data_ptr<int>(), err.data_ptr<int>(), static_cast<int>(M),  \
                               static_cast<int>(N), static_cast<int>(K), static_cast<int>(smax));                       \
    return;                                                                                                   \
  }
  @@CASES@@
#undef DGRS_CASE
  TORCH_CHECK(false, "dgrs_gemm: no instance for bm=", bm, " cps=", cps, " diag=", diag, " math=", math);
}

torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag, int64_t math) {
  cudaFuncAttributes a;
#define DGRS_ATTR(BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK) if (bm == BM && cps == CPS && diag == DIAG && math == MATH) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>); } else
  @@ATTRS@@
  { TORCH_CHECK(false, "dgrs_attrs: no instance"); }
#undef DGRS_ATTR
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}

// (registers, stack bytes) of one exact instance, specializations included (red -1 / spl 0 = generic)
torch::Tensor dgrs_attrs_s(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t red, int64_t spl) {
  cudaFuncAttributes a;
#define DGRS_ATTRS(BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK) if (bm == BM && cps == CPS && diag == DIAG && math == MATH && red == RED && spl == SPL) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>); } else
  @@ATTRS6@@
  { TORCH_CHECK(false, "dgrs_attrs_s: no instance"); }
#undef DGRS_ATTRS
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}

// (registers, stack bytes) of one exact v10 instance (0 = runtime for smx / m / n / k)
torch::Tensor dgrs_attrs_x(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t red, int64_t spl, int64_t smx,
                           int64_t mm, int64_t nn, int64_t kk) {
  cudaFuncAttributes a;
#define DGRS_ATTRX(BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK) if (bm == BM && cps == CPS && diag == DIAG && math == MATH && red == RED && spl == SPL && smx == SMX && mm == MM && nn == NN && kk == KK) { cudaFuncGetAttributes(&a, dgrs::rs_kernel<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>); } else
  @@ATTRSX@@
  { TORCH_CHECK(false, "dgrs_attrs_x: no instance"); }
#undef DGRS_ATTRX
  return torch::tensor({static_cast<int64_t>(a.numRegs), static_cast<int64_t>(a.localSizeBytes)});
}

// Ring stages of a launch (0 if dgrs_gemm would reject it), for lab logs.
int64_t dgrs_stages(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax) {
  const dgrs::DgrsLayout L = dgrs::dgrs_layout(static_cast<int>(bm), static_cast<int>(cps), static_cast<int>(math),
                                               static_cast<int>(splits), static_cast<int>(red), static_cast<int>(smax));
  return dgrs::dgrs_layout_ok(L, static_cast<int>(bm), static_cast<int>(math)) ? L.S : 0;
}

// dgrs_stages for a given DIAG (512 changes the correction area)
int64_t dgrs_stages_d(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax, int64_t diag) {
  const dgrs::DgrsLayout L = dgrs::dgrs_layout(static_cast<int>(bm), static_cast<int>(cps), static_cast<int>(math),
                                               static_cast<int>(splits), static_cast<int>(red), static_cast<int>(smax),
                                               (diag & 512) ? 1 : 0);
  return dgrs::dgrs_layout_ok(L, static_cast<int>(bm), static_cast<int>(math)) ? L.S : 0;
}

// Clusters of `splits` CTAs that can be co-resident (cudaOccupancyMaxActiveClusters) for an instance; 0 if splits < 2.
int64_t dgrs_max_clusters(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t splits) {
  if (splits < 2) return 0;
  int n = -1;
#define DGRS_OCC(BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK)                                               \
  if (bm == BM && cps == CPS && diag == DIAG && math == MATH) {                                               \
    auto kern = dgrs::rs_kernel<BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK>;                              \
    const size_t smem = static_cast<size_t>(CPS == 1 ? dgrs::SMEM_DYN1 : dgrs::SMEM_DYN2);                 \
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));           \
    cudaLaunchConfig_t cfg = {};                                                                              \
    cfg.gridDim = dim3(1, 1, static_cast<unsigned>(splits));                                                 \
    cfg.blockDim = dim3(128 + 128 * MATH, 1, 1);                                                              \
    cfg.dynamicSmemBytes = smem;                                                                              \
    cudaLaunchAttribute attr[1];                                                                              \
    attr[0].id = cudaLaunchAttributeClusterDimension;                                                         \
    attr[0].val.clusterDim.x = 1;                                                                             \
    attr[0].val.clusterDim.y = 1;                                                                             \
    attr[0].val.clusterDim.z = static_cast<unsigned>(splits);                                                 \
    cfg.attrs = attr;                                                                                         \
    cfg.numAttrs = 1;                                                                                         \
    if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) n = -1;                               \
    (void)cudaGetLastError();                                                                                 \
  } else
  @@OCCS@@
  { TORCH_CHECK(false, "dgrs_max_clusters: no instance"); }
#undef DGRS_OCC
  return n;
}
"""

_DGRS_CPP = """
void dgrs_gemm(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, torch::Tensor part,
               torch::Tensor cnt, torch::Tensor err, int64_t M, int64_t N, int64_t K, int64_t bm, int64_t splits,
               int64_t smax, int64_t cps, int64_t diag, int64_t red, int64_t math);
torch::Tensor dgrs_attrs(int64_t bm, int64_t cps, int64_t diag, int64_t math);
torch::Tensor dgrs_attrs_s(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t red, int64_t spl);
torch::Tensor dgrs_attrs_x(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t red, int64_t spl, int64_t smx,
                           int64_t mm, int64_t nn, int64_t kk);
int64_t dgrs_max_clusters(int64_t bm, int64_t cps, int64_t diag, int64_t math, int64_t splits);
int64_t dgrs_stages(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax);
int64_t dgrs_stages_d(int64_t bm, int64_t cps, int64_t math, int64_t splits, int64_t red, int64_t smax, int64_t diag);
"""


def _norm_instances(instances):
    """-> (BM, CPS, DIAG, MATH, RED, SPL, SMX, MM, NN, KK): RED -1 / others 0 = runtime; v9 / v10 specializations fix
    red / splits / smax / M / N / K at compile time (dgrs_gemm prefers the most specialized instance that matches)."""
    out = []
    for inst in (DGRS_INSTANCES if instances is None else instances):
        t = (inst, 1, 0) if isinstance(inst, int) else tuple(inst)
        t = t + (2, -1, 0, 0, 0, 0, 0)[len(t) - 3:]
        assert len(t) == 10, t
        out.append(t)
    return tuple(out)


def _n_spec(t):
    return int(t[4] >= 0) + sum(int(v > 0) for v in t[5:])


def dgrs_source(instances=None):
    insts = _norm_instances(instances)
    spec_first = sorted(insts, key=lambda t: -_n_spec(t))   # dispatch: most specialized first (first match wins)
    gen_first = sorted(insts, key=_n_spec)                   # attrs / occupancy: generic first
    fmt = lambda t: ", ".join(str(v) for v in t)
    return (_DGRS_CUDA.replace("@@WGMMA@@", "".join(_wgmma_rs(n) for n in (32, 64, 128)))
            .replace("@@CASES@@", "\n  ".join(f"DGRS_CASE({fmt(t)})" for t in spec_first))
            .replace("@@ATTRS@@", "\n  ".join(f"DGRS_ATTR({fmt(t)})" for t in gen_first))
            .replace("@@ATTRS6@@", "\n  ".join(f"DGRS_ATTRS({fmt(t)})" for t in insts))
            .replace("@@ATTRSX@@", "\n  ".join(f"DGRS_ATTRX({fmt(t)})" for t in insts))
            .replace("@@OCCS@@", "\n  ".join(f"DGRS_OCC({fmt(t)})" for t in gen_first)))


def dgrs_build(instances=None):
    """One load_inline module; instances = (BM, CPS, DIAG[, MATH[, RED[, SPL[, SMX[, M[, N[, K]]]]]]]) tuples (or plain
    BM ints = (BM, 1, 0, 2)); RED / SPL (v9) and SMX / M / N / K (v10) specialize an instance; sm_90a."""
    return load_inline(
        name="dgrs_ext", cpp_sources=_DGRS_CPP, cuda_sources=dgrs_source(instances),
        functions=["dgrs_gemm", "dgrs_attrs", "dgrs_attrs_s", "dgrs_attrs_x", "dgrs_max_clusters", "dgrs_stages", "dgrs_stages_d"], extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="dgrs_"), verbose=False)


def dgrs_pack(p):
    """bf16_proj's f12m_device_planes(w, natural=True)["p"] int32 [N/16][K/64][3][32][4] -> chunk-major
    [K/64][N/16][3][32][4] (a stage's 128 rows = one contiguous 12,288-byte block). Lossless reordering."""
    return p.permute(1, 0, 2, 3, 4).contiguous()


def dgrs_buffers(m, n, bm, splits, device):
    """Per-layer scratch: int32 error flag; for red 1 (global split-K reduction) fp32 partials [splits][tiles * bm][n]
    and int32 counters [(n / 128) * tiles] zeroed once (zero again after every call). red 0 (cluster DSMEM) ignores
    part / cnt."""
    tiles = -(-m // bm)
    return {"part": torch.empty(splits * tiles * bm * n, dtype=torch.float32, device=device),
            "cnt": torch.zeros((n // 128) * tiles, dtype=torch.int32, device=device),
            "err": torch.zeros(1, dtype=torch.int32, device=device)}


def dgrs_run(ext, x, wp_cm, rec, out, bufs, bm, splits, smax=8, cps=1, diag=0, red=0, math=2):
    """out [m, n] bf16 = x [m, k] bf16 (contiguous) @ W^T, W from the chunk-major 12-bit planes and rec. (bm, cps, diag)
    must be a built instance; splits: K-chunk slices (grid z), 1..8, every slice non-empty; red (splits > 1): 0 = a
    cluster (1, 1, splits) pulling peers' tiles over DSMEM, 1 = plain launch, global fp32 partials + last-arriver
    counter, 2 = a cluster where each slice pushes into the owner rank's receive area (v6; ext.dgrs_stages(...) = 0
    for configs that don't fit, e.g. BM 128 at S 2)."""
    m, k = x.shape
    n = out.shape[1]
    ext.dgrs_gemm(wp_cm, rec, x, out, bufs["part"], bufs["cnt"], bufs["err"], m, n, k, bm, splits, smax, cps, diag, red,
                  math)
    return out


# proj/dg/pinned/dgrs_v10.py (shared/gemm/dg12rs) sha256 f4ff43fafffd5af2
DGRS_CHOICE = {('unpooler', 32): {'kind': 'dgrs', 'BM': 32, 'S': 2, 'CPS': 1, 'RED': 2, 'MATH': 4, 'DIAG': 0, 'MOD': 'full', 'M': 32, 'N': 8192, 'K': 4096}, ('pooler', 32): {'kind': 'dgrs', 'BM': 32, 'S': 4, 'CPS': 1, 'RED': 1, 'MATH': 4, 'DIAG': 0, 'MOD': 'full', 'M': 32, 'N': 4096, 'K': 8192}}


DGRS_EXT = None
DGRS_STATE = {}


def dgrs_key(c, layer, m):
    return (layer, m, c["BM"], c["S"], c.get("CPS", 1), c.get("RED", 0), c.get("MATH", 2), c.get("DIAG", 0))


def dgrs_inst(c, layer, m):
    t = (c["BM"], c.get("CPS", 1), c.get("DIAG", 0), c.get("MATH", 2))
    if c.get("MOD") == "full":
        return t + (c.get("RED", 0), c["S"], 8, c["M"], c["N"], c["K"])
    if c.get("SPEC") or c.get("MOD") == "spec":
        return t + (c.get("RED", 0), c["S"])
    return t + (-1, 0)


def dgrs_setup(state):
    """Planes of every layer with a DGRS shape, one build with the chosen BMs, scratch per chosen (layer, m)."""
    global DGRS_EXT
    planes = {}
    for layer in sorted({layer for (layer, m) in DGRS_CHOICE}):
        try:
            pl = f12m_device_planes(state[layer], natural=True)
        except Exception as exc:
            print(f"dgrs planes {layer} FAILED: {type(exc).__name__}", flush=True)
            pl = None
        if pl is not None:
            planes[layer] = {"wp": dgrs_pack(pl["p"]), "rec": pl["rec"]}
    if not planes:
        return
    try:
        ext = dgrs_build(tuple(sorted({dgrs_inst(c, layer, m) for (layer, m), c in DGRS_CHOICE.items()})))
    except Exception as exc:
        print(f"dgrs build FAILED: {type(exc).__name__}: {' '.join(str(exc).split())[-300:]}", flush=True)
        return
    bufs = {}
    for (layer, m), c in DGRS_CHOICE.items():
        if layer in planes:
            bufs[dgrs_key(c, layer, m)] = dgrs_buffers(m, state[layer].shape[0], c["BM"], c["S"], state[layer].device)
    DGRS_STATE["planes"] = planes
    DGRS_STATE["bufs"] = bufs
    DGRS_EXT = ext


def run_dgrs(c, x, layer, n, state):
    m, k = x.shape
    pl = DGRS_STATE.get("planes", {}).get(layer)
    bufs = DGRS_STATE.get("bufs", {}).get(dgrs_key(c, layer, m))
    if DGRS_EXT is None or pl is None or bufs is None or n != state[layer].shape[0]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    return dgrs_run(DGRS_EXT, x.contiguous(), pl["wp"], pl["rec"], out, bufs, c["BM"], c["S"],
                    cps=c.get("CPS", 1), red=c.get("RED", 0), math=c.get("MATH", 2), diag=c.get("DIAG", 0))


_run_choice_pre_dgrs = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] == "dgrs":
        return run_dgrs(c, x, layer, n, state)
    return _run_choice_pre_dgrs(c, x, layer, n, state)


for _key, _c in DGRS_CHOICE.items():
    CHOICE[_key] = dict(_c, fallback=CHOICE.get(_key))

_warm_before_dgrs = _warm


def _warm(state):
    """prepare() ends with _warm(state): DG12-RS's planes, build and scratch first, then every chosen kernel once."""
    dgrs_setup(state)
    _warm_before_dgrs(state)
