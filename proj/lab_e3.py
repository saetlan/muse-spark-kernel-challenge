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

# (layer, m) -> kernel config, from the PE1 and PE2 labs (jobs eb62bbc5e6d147fa, 1d761e3043c74f50). Other shapes run
# SGLang's path.
CHOICE = {
    ('o', 1): {'kind': 'fma', 'BN': 4, 'split': 1, 'warps': 8},
    ('pooler', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8},
    ('unpooler', 1): {'kind': 'tiny', 'max_m': 16},
    ('pooler', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 8): {'kind': 'fma', 'BN': 16, 'split': 64, 'warps': 8},
    ('unpooler', 8): {'kind': 'tiny', 'max_m': 16},
    ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('pooler', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4},
    ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4},
    ('router', 128): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4},
    ('router', 256): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4},
    ('qkv_gate', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8},
    ('router', 1): {'kind': 'fma', 'BN': 2, 'split': 1, 'warps': 8},
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
    if c["kind"] == "sx":
        return (f"sx NT8{c['NT8']} TG{c['TG']} RTW{c['RTW']} RGN{c['RGN']} KSW{c['KSW']} NS{c['NS']} D{c['D']} "
                f"XS{c['XS']} CS{c['CS']}")
    if c["kind"] == "wgf":
        return f"wgf KS{c['KS']} ST{c['ST']} TT{c['TT']} D{c['D']}"
    if c["kind"] == "wg":
        return f"wg KS{c['KS']} ST{c['ST']} TT{c['TT']}"
    if c["kind"] == "f12m":
        return (f"f12m NT{c['NT']} RTW{c['RTW']} RG{c['RG']} KSW{c['KSW']} D{c['D']}"
                + ("" if c["F12"] else " bf16"))
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


# ====================================================================== lab (e3): check runs only

"""PE3 recon: v2 timed by the judge, plus in prepare():
  env      versions and availability (DeepGEMM, FlashInfer, cutlass-dsl, HPC-Ops, Gluon)
  mb       CUDA microbenchmarks (load_inline, sm_90a), graphs + cold L2:
             ldg     one-shot / looped 16-B ld.global.nc.L1::no_allocate.L2::256B streams per size
             bulk    cp.async.bulk 1D global->smem rings (mbarrier expect_tx) per size
             pf      cp.async.bulk.prefetch.L2 of the whole slice, then ldg
             empty   an empty kernel (the fixed cost every call pays)
             cluster fixed-order DSMEM reduction across 2/4/8-CTA clusters (mapa + ld.shared::cluster)
             prmt    PRMT and LOP3 issue rates (for the F12 decode budget)
  libs     library A/B on all 30 scored shapes through lab/sweep.py's gate and commit:
             FlashInfer mm_bf16 (cublaslt / cudnn / tinygemm / auto, autotuned under its own
             cold-L2 graph replay), DeepGEMM bf16_gemm_nt (its cuBLASLt, then its native sm90 kernel at m >= 128),
             SGLang's Muse FP32 router kernel (m <= 8).
FlashInfer's CuTe-DSL warp split-K / direct kernels are gated to SM100 by FlashInfer; calling them
on sm90 risks a sticky CUDA fault, so they get their own check (PE3b), not this one.
Winners go into CHOICE only where correct and >= MIN_GAIN faster than both SGLang and v2."""

import time

PREPARE_CAP_S = 270.0
LAB_MAX_S = 240.0
MB_SIZES = (("8.4MB", 1024 * 4096 * 2), ("16.8MB", 8192 * 1024 * 2), ("37.7MB", 2304 * 8192 * 2),
            ("67.1MB", 4096 * 8192 * 2))

MB_SRC = {
    "core": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

// Each CTA streams its contiguous slice; D 16-B loads in flight per thread per step.
template <int D>
__global__ void __launch_bounds__(512) mb_ldg(const uint4* __restrict__ w, long long per_cta, unsigned* sink, int prefetch) {
    const uint4* base = w + (long long)blockIdx.x * per_cta;
    if (prefetch && threadIdx.x == 0) {
        long long bytes = per_cta * 16;
        for (long long off = 0; off < bytes; off += 32768) {
            long long sz = bytes - off < 32768 ? bytes - off : 32768;
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"((const char*)base + off), "r"((unsigned)sz) : "memory");
        }
    }
    unsigned acc = 0;
    const long long step = (long long)blockDim.x * D;
    for (long long i = threadIdx.x; i < per_cta; i += step) {
        unsigned r[D][4];
#pragma unroll
        for (int d = 0; d < D; ++d) {
            long long j = i + (long long)d * blockDim.x;
            if (j < per_cta) {
                ld_nc(base + j, r[d]);
            } else {
                r[d][0] = r[d][1] = r[d][2] = r[d][3] = 0u;
            }
        }
#pragma unroll
        for (int d = 0; d < D; ++d) acc ^= r[d][0] ^ r[d][1] ^ r[d][2] ^ r[d][3];
    }
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

__global__ void mb_empty(unsigned* sink) {
    if (threadIdx.x == 100000u) sink[0] = 1u;
}

// 8 independent chains per thread of PRMT (op 0) or LOP3 (op 1).
__global__ void __launch_bounds__(256) mb_int(unsigned* sink, int iters, int op) {
    unsigned a0 = threadIdx.x, a1 = a0 * 3u + 1u, a2 = a0 ^ 0x5a5a5a5au, a3 = a0 + 7u;
    unsigned a4 = a0 * 5u, a5 = ~a0, a6 = a0 << 3, a7 = a0 >> 1;
    const unsigned k = 0x76543210u ^ (unsigned)iters;
    const unsigned k2 = 0x80808080u ^ (unsigned)(iters >> 20);
    if (op == 0) {
        for (int i = 0; i < iters; ++i) {
            asm("prmt.b32 %0, %0, %8, 0x5140;\nprmt.b32 %1, %1, %8, 0x7362;\nprmt.b32 %2, %2, %8, 0x5140;\nprmt.b32 %3, %3, %8, 0x7362;\n"
                "prmt.b32 %4, %4, %8, 0x5140;\nprmt.b32 %5, %5, %8, 0x7362;\nprmt.b32 %6, %6, %8, 0x5140;\nprmt.b32 %7, %7, %8, 0x7362;"
                : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7) : "r"(k), "r"(k2));
        }
    } else {
        for (int i = 0; i < iters; ++i) {
            asm("lop3.b32 %0, %0, %8, %9, 0xEA;\nlop3.b32 %1, %1, %8, %9, 0xEA;\nlop3.b32 %2, %2, %8, %9, 0xEA;\n"
                "lop3.b32 %3, %3, %8, %9, 0xEA;\nlop3.b32 %4, %4, %8, %9, 0xEA;\nlop3.b32 %5, %5, %8, %9, 0xEA;\n"
                "lop3.b32 %6, %6, %8, %9, 0xEA;\nlop3.b32 %7, %7, %8, %9, 0xEA;"
                : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(a3), "+r"(a4), "+r"(a5), "+r"(a6), "+r"(a7) : "r"(k), "r"(k2));
        }
    }
    unsigned acc = a0 ^ a1 ^ a2 ^ a3 ^ a4 ^ a5 ^ a6 ^ a7;
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

void mb_run_ldg(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t threads, int64_t d, int64_t prefetch, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* p = reinterpret_cast<const uint4*>(w.data_ptr());
    unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr());
    const int pf = (int)prefetch;
    switch (d) {
        case 2: mb_ldg<2><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 4: mb_ldg<4><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 8: mb_ldg<8><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        case 16: mb_ldg<16><<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(p, per_cta, s, pf); break;
        default: TORCH_CHECK(false, "unsupported D");
    }
}

void mb_run_empty(int64_t ctas, int64_t threads, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    mb_empty<<<(unsigned)ctas, (unsigned)threads, 0, stream>>>(reinterpret_cast<unsigned*>(sink.data_ptr()));
}

void mb_run_int(int64_t ctas, int64_t iters, int64_t op, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    mb_int<<<(unsigned)ctas, 256, 0, stream>>>(reinterpret_cast<unsigned*>(sink.data_ptr()), (int)iters, (int)op);
}
""",
    "bulk": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_issue(unsigned bar, unsigned dst, const char* src, unsigned bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
                 :: "r"(bar), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// cp.async.bulk ring: S stages of CHUNK bytes; one thread issues, all threads touch each stage.
template <int S>
__global__ void __launch_bounds__(256) mb_bulk(const uint4* __restrict__ w, long long per_cta, int chunk, unsigned* sink) {
    extern __shared__ __align__(128) unsigned char ring[];
    __shared__ __align__(8) unsigned long long bars[S];
    const char* base = (const char*)(w + (long long)blockIdx.x * per_cta);
    const int nchunks = (int)(per_cta * 16 / chunk);
    const unsigned bar0 = smem_addr(bars);
    const unsigned ring0 = smem_addr(ring);
    if (threadIdx.x == 0) {
        for (int s = 0; s < S; ++s) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0 + 8u * s) : "memory");
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int c = 0; c < S && c < nchunks; ++c)
            bulk_issue(bar0 + 8u * c, ring0 + (unsigned)(c * chunk), base + (long long)c * chunk, (unsigned)chunk);
    }
    unsigned acc = 0;
    for (int c = 0; c < nchunks; ++c) {
        const int s = c % S;
        mbar_wait(bar0 + 8u * s, (unsigned)((c / S) & 1));
        const uint4* st = (const uint4*)(ring + (size_t)s * chunk);
        for (int t = threadIdx.x * 8; t < chunk / 16; t += blockDim.x * 8) acc ^= st[t].x;
        __syncthreads();
        if (threadIdx.x == 0 && c + S < nchunks)
            bulk_issue(bar0 + 8u * s, ring0 + (unsigned)(s * chunk), base + (long long)(c + S) * chunk, (unsigned)chunk);
    }
    if (acc == 0x9e3779b9u) sink[blockIdx.x] = acc;
}

void mb_run_bulk(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t stages, int64_t chunk, torch::Tensor sink) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const uint4* p = reinterpret_cast<const uint4*>(w.data_ptr());
    unsigned* s = reinterpret_cast<unsigned*>(sink.data_ptr());
    const size_t smem = (size_t)stages * (size_t)chunk;
    TORCH_CHECK(per_cta * 16 % chunk == 0 && chunk % 16 == 0 && smem <= 200 * 1024, "bad bulk config");
    if (stages == 4) {
        mb_bulk<4><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    } else if (stages == 6) {
        mb_bulk<6><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    } else {
        TORCH_CHECK(stages == 8, "unsupported stages");
        mb_bulk<8><<<(unsigned)ctas, 256, smem, stream>>>(p, per_cta, (int)chunk, s);
    }
}

void mb_bulk_attrs() {
    const int big = 200 * 1024;
    cudaFuncSetAttribute(mb_bulk<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
    cudaFuncSetAttribute(mb_bulk<6>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
    cudaFuncSetAttribute(mb_bulk<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, big);
}
""",
    "cluster": r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

__device__ __forceinline__ void ld_nc(const uint4* p, unsigned (&r)[4]) {
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "l"(p));
}

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void mbar_wait(unsigned bar, unsigned parity) {
    asm volatile("{\n .reg .pred p;\n WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n @!p bra WAIT_%=;\n}\n"
                 :: "r"(bar), "r"(parity) : "memory");
}

__device__ __forceinline__ void bulk_issue(unsigned bar, unsigned dst, const char* src, unsigned bytes) {
    asm volatile("{\n .reg .b64 st;\n mbarrier.arrive.expect_tx.shared::cta.b64 st, [%0], %1;\n}\n"
                 :: "r"(bar), "r"(bytes) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                 :: "r"(dst), "l"(src), "r"(bytes), "r"(bar) : "memory");
}

// Each CTA holds an n_per-float partial; CTA r of a C-CTA cluster sums rows [r*n/C, (r+1)*n/C)
// over all peers in rank order (deterministic) and writes them out.
template <int C>
__global__ void __cluster_dims__(C, 1, 1) __launch_bounds__(256) mb_cluster(const float* __restrict__ in, float* __restrict__ out, int n_per) {
    extern __shared__ __align__(16) float part[];
    for (int i = threadIdx.x; i < n_per; i += blockDim.x) part[i] = in[(long long)blockIdx.x * n_per + i];
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    const int share = n_per / C;
    const unsigned local = smem_addr(part);
    for (int i = (int)rank * share + threadIdx.x; i < ((int)rank + 1) * share; i += blockDim.x) {
        float s = 0.f;
#pragma unroll
        for (int r = 0; r < C; ++r) {
            unsigned remote;
            float v;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local + 4u * (unsigned)i), "r"(r));
            asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote) : "memory");
            s += v;
        }
        out[(long long)(blockIdx.x / C) * n_per + i] = s;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}

void mb_run_cluster(torch::Tensor in, torch::Tensor out, int64_t ctas, int64_t c, int64_t n_per) {
    auto stream = at::cuda::getCurrentCUDAStream();
    const float* i = reinterpret_cast<const float*>(in.data_ptr());
    float* o = reinterpret_cast<float*>(out.data_ptr());
    const size_t smem = (size_t)n_per * 4;
    TORCH_CHECK(ctas % c == 0 && n_per % c == 0 && smem <= 48 * 1024, "bad cluster config");
    if (c == 2) mb_cluster<2><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per);
    else if (c == 4) mb_cluster<4><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per);
    else { TORCH_CHECK(c == 8, "unsupported cluster"); mb_cluster<8><<<(unsigned)ctas, 256, smem, stream>>>(i, o, (int)n_per); }
}

""",
}

MB_CPP = {
    "core": ['void mb_run_ldg(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t threads, int64_t d, int64_t prefetch, torch::Tensor sink);', 'void mb_run_empty(int64_t ctas, int64_t threads, torch::Tensor sink);', 'void mb_run_int(int64_t ctas, int64_t iters, int64_t op, torch::Tensor sink);'],
    "bulk": ['void mb_run_bulk(torch::Tensor w, int64_t per_cta, int64_t ctas, int64_t stages, int64_t chunk, torch::Tensor sink);', 'void mb_bulk_attrs();'],
    "cluster": ['void mb_run_cluster(torch::Tensor in, torch::Tensor out, int64_t ctas, int64_t c, int64_t n_per);'],
}


def mb_build(part):
    import tempfile
    from torch.utils.cpp_extension import load_inline
    decls = MB_CPP[part]
    names = [d.split("(")[0].split()[-1] for d in decls]
    return load_inline(
        name=f"okc_proj_mb_{part}", cpp_sources="\n".join(decls), cuda_sources=MB_SRC[part], functions=names,
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix=f"okc_proj_mb_{part}_"), verbose=False)


def sec_env(lab):
    import importlib.metadata as md
    import importlib.util as iu
    parts = [f"torch {torch.__version__}", f"triton {triton.__version__}"]
    for dist in ("flashinfer-python", "flashinfer", "deep_gemm", "deep-gemm", "nvidia-cutlass-dsl", "tilelang",
                 "cuda-python", "nvidia-cudnn-frontend", "hpc-ops"):
        try:
            parts.append(f"{dist} {md.version(dist)}")
        except Exception:
            pass
    for mod in ("hpc", "deep_gemm", "flashinfer", "cudnn", "triton.experimental.gluon",
                "triton.experimental.gluon.language.nvidia.hopper", "cutlass", "quack"):
        try:
            parts.append(f"{mod}:{'y' if iu.find_spec(mod) else 'n'}")
        except Exception as exc:
            parts.append(f"{mod}:{type(exc).__name__}")
    props = torch.cuda.get_device_properties(0)
    parts.append(f"L2 {getattr(props, 'L2_cache_size', -1) / 2**20:.0f} MiB, sms {props.multi_processor_count}")
    log("env: " + ", ".join(parts))
    try:
        import deep_gemm
        log(f"deep_gemm: bf16_gemm_nt {hasattr(deep_gemm, 'bf16_gemm_nt')}, "
            f"deterministic {deep_gemm.get_deterministic_algorithms()}")
    except Exception as exc:
        log(f"deep_gemm: {type(exc).__name__}: {_error_text(exc, 200)}")


class _MB:
    def __init__(self):
        self.mods = {}
        self.empty_us = None


def sec_mb_build(lab, mb, part):
    t = time.monotonic()
    mod = mb_build(part)
    if part == "bulk":
        mod.mb_bulk_attrs()
    mb.mods[part] = mod
    log(f"mb: nvcc build {part} {time.monotonic() - t:.1f}s")


def sec_mb_basics(lab, mb):
    m = mb.mods["core"]
    sink = torch.zeros(4096, dtype=torch.int32, device="cuda")
    e1 = lab.time_graph(lambda: m.mb_run_empty(1, 32, sink))
    e132 = lab.time_graph(lambda: m.mb_run_empty(132, 512, sink))
    e132w = lab.time_graph(lambda: m.mb_run_empty(132, 512, sink), cold=False)
    two = lab.time_graph(lambda: (m.mb_run_empty(132, 512, sink), m.mb_run_empty(132, 512, sink)))
    mb.empty_us = e132
    log(f"mb empty kernel (graph replay, raw): 1 CTA {e1:.2f} us, 132 CTAs {e132:.2f} cold / {e132w:.2f} warm, "
        f"2 kernels {two:.2f} us")
    for op, name in ((0, "prmt"), (1, "lop3")):
        iters = 4096
        us = lab.time_graph(lambda: m.mb_run_int(132 * 8, iters, op, sink), cold=False)
        ops = 132 * 8 * 256 * iters * 8
        per_clk_sm = ops / (max(us - e132w, 1e-3) * 1e-6) / 132 / 1.98e9
        log(f"mb {name}: {per_clk_sm:.1f} thread-ops per clock per SM ({us:.1f} us)")


def sec_mb_stream(lab, mb):
    m = mb.mods["core"]
    mbulk = mb.mods.get("bulk")
    sink = torch.zeros(4096, dtype=torch.int32, device="cuda")
    buf = torch.randint(-2**31, 2**31 - 1, (MB_SIZES[-1][1] // 4,), dtype=torch.int32, device="cuda")
    base = mb.empty_us or 0.0
    for name, nbytes in MB_SIZES:
        rows = []
        best = (1e9, "")
        for ctas in (132, 264, 528):
            per = nbytes // 16 // ctas
            for threads in (256, 512):
                for d in (2, 4, 8, 16):
                    if threads * d > 4 * per + threads:  # mostly idle threads
                        continue
                    us = lab.time_graph(lambda: m.mb_run_ldg(buf, per, ctas, threads, d, 0, sink))
                    rows.append(f"c{ctas}t{threads}d{d}:{us:.2f}")
                    if us < best[0]:
                        best = (us, f"ldg c{ctas} t{threads} d{d}")
        for ctas in (132, 264):
            per = nbytes // 16 // ctas
            for threads, d in ((512, 4), (512, 8)):
                us = lab.time_graph(lambda: m.mb_run_ldg(buf, per, ctas, threads, d, 1, sink))
                rows.append(f"pf c{ctas}t{threads}d{d}:{us:.2f}")
                if us < best[0]:
                    best = (us, f"pf+ldg c{ctas} t{threads} d{d}")
        for ctas in ((132, 264) if mbulk is not None else ()):
            for stages, chunk in ((4, 16384), (8, 16384), (4, 32768), (6, 32768)):
                per = (nbytes // ctas // chunk) * chunk // 16
                if per == 0:
                    continue
                us = lab.time_graph(lambda: mbulk.mb_run_bulk(buf, per, ctas, stages, chunk, sink))
                eff_bytes = per * 16 * ctas
                rows.append(f"bulk c{ctas}s{stages}k{chunk // 1024}:{us:.2f}")
                if us * nbytes / eff_bytes < best[0]:
                    best = (us * nbytes / eff_bytes, f"bulk c{ctas} s{stages} {chunk // 1024}K")
        tb = nbytes / ((best[0] - base + 1.1) * 1e-6) / 1e12 if best[0] > base else 0.0
        log(f"mb read {name}: best {best[1]} {best[0]:.2f} us raw (~{best[0] - base + 1.1:.2f} us judge-equivalent, "
            f"{tb:.2f} TB/s); " + " ".join(rows))


def sec_mb_cluster(lab, mb):
    m = mb.mods["cluster"]
    n_per = 4096
    for c in (2, 4, 8):
        ctas = c * (132 // c)
        src = torch.randn(ctas * n_per, device="cuda")
        out = torch.empty((ctas // c) * n_per, device="cuda")
        m.mb_run_cluster(src, out, ctas, c, n_per)
        torch.cuda.synchronize()
        ref = src.view(ctas // c, c, n_per).sum(1).flatten()
        err = float((out - ref).abs().max())
        us = lab.time_graph(lambda: m.mb_run_cluster(src, out, ctas, c, n_per))
        log(f"mb cluster {c}: {ctas} CTAs, 16 KB partial each, max err {err:.1e}, {us:.2f} us raw")


def lib_candidates(layer, m, n, k):
    cands = []
    if layer == "router":
        if m <= 8:
            cands.append({"kind": "sgl_router"})
        return cands
    for backend in ("cublaslt", "cudnn", "tinygemm", "auto"):
        if backend == "tinygemm" and m > 32:
            continue
        cands.append({"kind": "fi_mm", "backend": backend})
    cands.append({"kind": "dg", "native": False})
    return cands


def _autotune_fi(lab, state, layer, m, cands):
    """Profile mm_bf16 for this shape once inside flashinfer.autotune(), so later calls (timed or
    lab) reuse the in-memory tactic."""
    fi = [c for c in cands if c["kind"] == "fi_mm"]
    if not fi:
        return
    import flashinfer
    n, k = lab_n(layer, m), FULL[layer][1]
    x = lab_input(m, k, 4000 + m)
    for c in fi:
        if lab.left() < 30:
            log(f"autotune {layer} m{m}: skipped, budget")
            return
        try:
            with flashinfer.autotune(True):
                run_choice(c, x, layer, n, state)
        except Exception as exc:
            log(f"autotune {layer} m{m} {c['backend']}: {type(exc).__name__}: {_error_text(exc, 160)}")


def sec_libs(lab, sweep, ms, layers, kinds):
    for m in ms:
        for layer in layers:
            n, k = lab_n(layer, m), FULL[layer][1]
            cands = [c for c in lib_candidates(layer, m, n, k) if c["kind"] in kinds]
            if not cands:
                continue
            if "fi_mm" in kinds:
                _autotune_fi(lab, sweep.state, layer, m, cands)
            sweep.shape(lab, layer, m, cands)


def sec_dg_native(lab, sweep):
    for m in (128, 256):
        for layer in ("unpooler", "o", "pooler", "qkv_gate"):
            if lab.left() < 30:
                log("dg native: stopped, budget")
                return
            sweep.shape(lab, layer, m, [{"kind": "dg", "native": True}])


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    log(f"v2 prepare took {time.monotonic() - t0:.1f}s; lab budget {budget:.0f}s")
    lab = Lab(budget)
    mb = _MB()
    sweep = Sweep(state)
    lab.section("env", sec_env)
    for part in ("core", "bulk", "cluster"):
        lab.section(f"mb build {part}", sec_mb_build, mb, part, min_left=60.0)
    if "core" in mb.mods:
        lab.section("mb basics", sec_mb_basics, mb)
        lab.section("mb stream", sec_mb_stream, mb, min_left=60.0)
    if "cluster" in mb.mods:
        lab.section("mb cluster", sec_mb_cluster, mb)
    small, large = (1, 8, 16, 32), (128, 256)
    bf16 = ("unpooler", "o", "qkv_gate", "pooler")
    lab.section("libs router", sec_libs, sweep, (1, 8), ("router",), ("sgl_router",), min_left=20.0)
    lab.section("libs fi_mm large", sec_libs, sweep, large, bf16, ("fi_mm",), min_left=40.0)
    lab.section("libs fi_mm small", sec_libs, sweep, small, bf16, ("fi_mm",), min_left=40.0)
    lab.section("libs dg cublaslt", sec_libs, sweep, small + large, bf16, ("dg",), min_left=20.0)
    lab.section("libs dg native", sec_dg_native, sweep, min_left=40.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    try:
        import deep_gemm
        if not any(c.get("kind") == "dg" and c.get("native") for c in CHOICE.values()):
            deep_gemm.use_deterministic_algorithms(False)
    except Exception:
        pass
    log(f"CHOICE now has {len(CHOICE)} shapes; prepare took {time.monotonic() - t0:.0f}s")
    return state
