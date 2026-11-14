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

# (layer, m) -> kernel config, from the PE1 lab (job eb62bbc5e6d147fa). Other shapes run
# SGLang's path.
CHOICE = {
    ('o', 1): {'kind': 'gemv'},
    ('pooler', 1): {'kind': 'gemv'},
    ('unpooler', 1): {'kind': 'tiny', 'max_m': 16},
    ('pooler', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 8): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 64, 'split': 32, 'mode': 2, 'warps': 4, 'stages': 4},
    ('unpooler', 8): {'kind': 'tiny', 'max_m': 16},
    ('o', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('pooler', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 16): {'kind': 'swap', 'BM': 16, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4},
    ('pooler', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 256, 'split': 1, 'mode': 0, 'warps': 4, 'stages': 4},
    ('router', 32): {'kind': 'swap', 'BM': 32, 'BN': 64, 'BK': 128, 'split': 32, 'mode': 1, 'warps': 4, 'stages': 4},
    ('router', 128): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4},
    ('router', 256): {'kind': 'gemm', 'BM': 64, 'BN': 64, 'BK': 64, 'split': 16, 'mode': 1, 'warps': 4, 'stages': 4},
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


# ====================================================================== lab (e2): check runs only

"""PE2: v1 timed by the judge, plus in prepare() a sweep of _fma_kernel (CUDA-core GEMV, exact
FP32 for the router), first on the router at m 1, 8, 16, 32, then on the BF16 layers at m 1 and 8.
A config replaces v1's choice only if it is correct and >= MIN_GAIN faster than both SGLang and
v1's current kernel for that shape (Sweep.commit).

prepare() has 300 s in all; v1's own prepare (image-kernel JIT warm-up) took 20 s in the scored
run but its JIT took ~90 s in PE1, so the lab gets whatever is left of PREPARE_CAP_S."""

import time

PREPARE_CAP_S = 265.0
LAB_MAX_S = 200.0


def fma_candidates(layer, m, n, k):
    wbytes = n * k * (4 if layer == "router" else 2)
    cands = []
    for bn in (2, 4, 8, 16, 32, 64):
        tiles = -(-n // bn)
        for split in (1, 2, 4, 8, 16, 32, 64):
            bk = k // split
            if bk < 128 or k % split:
                continue
            ctas = tiles * split
            if not (96 <= ctas <= 2048):
                continue
            if tiles * m * k * 2 > 2 * wbytes:  # x re-read through L2 by every row tile
                continue
            for warps in (4, 8):
                elems = bn * bk // (32 * warps)  # weights held per thread
                if 8 <= elems <= 128:
                    cands.append({"kind": "fma", "BN": bn, "split": split, "warps": warps})
    return cands


def sec_fma(lab, sweep, layers, ms):
    for m in ms:
        for layer in layers:
            n, k = lab_n(layer, m), FULL[layer][1]
            sweep.shape(lab, layer, m, fma_candidates(layer, m, n, k))


def prepare(weights):
    t0 = time.monotonic()
    state = _submission_prepare(weights)
    budget = min(LAB_MAX_S, PREPARE_CAP_S - (time.monotonic() - t0))
    log(f"v1 prepare took {time.monotonic() - t0:.1f}s; lab budget {budget:.0f}s")
    lab = Lab(budget)
    sweep = Sweep(state)
    lab.section("fma router", sec_fma, sweep, ("router",), (1, 8, 16, 32), min_left=10.0)
    lab.section("fma bf16 m1", sec_fma, sweep, ("unpooler", "o", "qkv_gate", "pooler"), (1,), min_left=10.0)
    lab.section("fma bf16 m8", sec_fma, sweep, ("unpooler", "o", "qkv_gate", "pooler"), (8,), min_left=10.0)
    lab.section("commit", lambda lab_: sweep.commit(lab_), min_left=-1e9)
    log(f"CHOICE now has {len(CHOICE)} shapes; prepare took {time.monotonic() - t0:.0f}s")
    return state
