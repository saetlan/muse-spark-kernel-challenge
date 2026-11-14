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

# (layer, m) -> kernel config. Empty: every shape runs SGLang's path.
CHOICE = {}
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
    if P9_ENABLED:
        p9_setup(state)
    return state


def kernel_fn(data, state):
    x, cfg = data["x"], data["config"]
    layer, n = cfg["layer"], cfg["n"]
    c = CHOICE.get((layer, x.shape[0]))
    if c is None:
        return sglang_path(x, layer, n, state)
    return run_choice(c, x, layer, n, state)


# ====================================================================== RG (proj/rg): router hi + lo mma.sync kernel, m = 32/128/256

import tempfile

def rg_frag(v):
    """BF16 [N, K] -> [N/16, K/16, 32, 8] in A-fragment order, natural k."""
    n, k = v.shape
    assert n % 16 == 0 and k % 16 == 0
    return v.reshape(n // 16, 2, 8, k // 16, 2, 4, 2).permute(0, 3, 2, 5, 4, 1, 6).reshape(n // 16, k // 16, 32, 8)


def rg_pack(w):
    """FP32 router [N, K] -> [N/16, K/16, 2, 32, 8] BF16 (hi fragment, then lo), contiguous."""
    assert w.dtype == torch.float32
    hi = w.to(torch.bfloat16)
    lo = (w - hi.to(torch.float32)).to(torch.bfloat16)
    return torch.stack((rg_frag(hi), rg_frag(lo)), dim=2).contiguous()


RG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace rg {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ld_a(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src, uint32_t bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(dst), "l"(src), "r"(bytes) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); }

__device__ __forceinline__ void ldm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void ldm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void mma(float (&c)[4], const uint4& a, uint32_t b0, uint32_t b1) {
  asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1));
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

__device__ __forceinline__ float4 ld_peer_v4(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float4 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
  return v;
}

// Index helpers (check_rg.py evaluates their return expressions): first chunk of rank r; byte offset of
// 16-B piece j of token row r in an x stage; uint4 index of tile `tile`'s step s hi (h = 0) / lo (h = 1)
// fragment of lane `lane` (KSS k16 steps per row); a warp's row tile and token group; the token a lane
// addresses for ldmatrix matrix mi of n8-tile pair p (group base tb); C fragment row (in the CTA) and
// token; first token of rank r's share, item -> token offset / 4-row group.
__device__ __forceinline__ int rg_c0(int r, int chunks, int cs) { return r * chunks / cs; }
__device__ __forceinline__ int rg_xoff(int tok, int j) { return tok * 128 + ((j ^ (tok & 7)) << 4); }
__device__ __forceinline__ long long rg_aidx(long long tile, int kss, int s, int h, int lane) { return ((tile * kss + s) * 2 + h) * 32 + lane; }
__device__ __forceinline__ int rg_wrt(int warp, int tg) { return warp / tg; }
__device__ __forceinline__ int rg_wtg(int warp, int tg) { return warp % tg; }
__device__ __forceinline__ int rg_ltok(int tb, int p, int lane) { return tb + 16 * p + (lane & 7) + 8 * ((lane >> 3) >> 1); }
__device__ __forceinline__ int rg_lkch(int j, int lane) { return 2 * j + ((lane >> 3) & 1); }
__device__ __forceinline__ int rg_crow(int rt, int g, int r) { return 16 * rt + g + 8 * (r >> 1); }
__device__ __forceinline__ int rg_ctok(int tb, int n, int t, int r) { return tb + 8 * n + 2 * t + (r & 1); }
__device__ __forceinline__ int rg_etok(int r, int share, int idx, int groups) { return r * share + idx / groups; }
__device__ __forceinline__ int rg_erow(int idx, int groups) { return 4 * (idx % groups); }

template <int RTC, int NTW, int CS, int XS, int D>
__global__ void __launch_bounds__(128)
rg_kernel(const uint4* __restrict__ wp, const __nv_bfloat16* __restrict__ x, float* __restrict__ out, int m) {
  constexpr int TG = 4 / RTC;                 // token groups (warps per row tile)
  constexpr int MT = TG * 8 * NTW;            // tokens covered (padded)
  constexpr int K = 8192, CHUNKS = K / 64, KSS = K / 16, ROWS = 16 * RTC;
  static_assert(RTC == 1 || RTC == 2, "row tiles per CTA");
  static_assert(NTW == 1 || NTW % 2 == 0, "n8 tiles per warp come in pairs (ldmatrix.x4)");
  static_assert(XS >= 2 && D >= 1 && MT % CS == 0, "rg config");
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
  const int rt = rg_wrt(warp, TG), tgi = rg_wtg(warp, TG);
  const uint32_t rank = cluster_rank<CS>();
  const long long cl = blockIdx.x / CS;
  const long long tile = cl * RTC + rt;
  const int c0 = rg_c0(static_cast<int>(rank), CHUNKS, CS), c1 = rg_c0(static_cast<int>(rank) + 1, CHUNKS, CS);
  const int nch = c1 - c0;
  const uint32_t sbase = smem_u32(smem);

  auto copy_x = [&](int c, int s) {
    for (int p = tid; p < MT * 8; p += 128) {
      const int tok = p >> 3, j = p & 7;
      const bool ok = tok < m;
      const __nv_bfloat16* src = x + static_cast<long long>(ok ? tok : 0) * K + 64 * c + 8 * j;
      cp16(sbase + static_cast<uint32_t>(s * MT * 128 + rg_xoff(tok, j)), src, ok ? 16u : 0u);
    }
  };
  uint4 wa[D][4][2];
  auto load_a = [&](int sl, int c) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int h = 0; h < 2; ++h) ld_a(wp + rg_aidx(tile, KSS, 4 * c + j, h, lane), wa[sl][j][h]);
  };

#pragma unroll
  for (int sl = 0; sl < D; ++sl)
    if (sl < nch) load_a(sl, c0 + sl);
#pragma unroll
  for (int s = 0; s < XS - 1; ++s) {
    if (s < nch) copy_x(c0 + s, s);
    cp_commit();
  }

  float acc[NTW][4];
#pragma unroll
  for (int n = 0; n < NTW; ++n)
#pragma unroll
    for (int r = 0; r < 4; ++r) acc[n][r] = 0.0f;
  const int tb = tgi * 8 * NTW;

  for (int i0 = 0; i0 < nch; i0 += D) {
#pragma unroll
    for (int sl = 0; sl < D; ++sl) {
      const int i = i0 + sl;
      if (i < nch) {
        cp_wait<XS - 2>();
        __syncthreads();                                   // chunk i landed; every warp done with chunk i - 1
        if (i + XS - 1 < nch) copy_x(c0 + i + XS - 1, (i + XS - 1) % XS);
        cp_commit();
        const uint32_t st = sbase + static_cast<uint32_t>((i % XS) * MT * 128);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          if constexpr (NTW == 1) {
            uint32_t b0, b1;
            ldm_x2(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, 0, lane & 15), rg_lkch(j, lane & 15))), b0, b1);
            mma(acc[0], wa[sl][j][0], b0, b1);
            mma(acc[0], wa[sl][j][1], b0, b1);
          } else {
#pragma unroll
            for (int p = 0; p < NTW / 2; ++p) {
              uint32_t b[4];
              ldm_x4(st + static_cast<uint32_t>(rg_xoff(rg_ltok(tb, p, lane), rg_lkch(j, lane))), b[0], b[1], b[2], b[3]);
              mma(acc[2 * p], wa[sl][j][0], b[0], b[1]);
              mma(acc[2 * p], wa[sl][j][1], b[0], b[1]);
              mma(acc[2 * p + 1], wa[sl][j][0], b[2], b[3]);
              mma(acc[2 * p + 1], wa[sl][j][1], b[2], b[3]);
            }
          }
        }
        if (i + D < nch) load_a(sl, c0 + i + D);
      }
    }
  }
  cp_wait<0>();
  __syncthreads();                                         // stages idle: reuse them for the partial tile

  float* part = reinterpret_cast<float*>(smem);             // [MT tokens][ROWS rows]
#pragma unroll
  for (int n = 0; n < NTW; ++n)
#pragma unroll
    for (int r = 0; r < 4; ++r) part[rg_ctok(tb, n, t, r) * ROWS + rg_crow(rt, g, r)] = acc[n][r];
  cluster_sync<CS>();
  constexpr int SHARE = MT / CS, GROUPS = ROWS / 4;
  const uint32_t paddr = sbase;
  for (int idx = tid; idx < SHARE * GROUPS; idx += 128) {
    const int tok = rg_etok(static_cast<int>(rank), SHARE, idx, GROUPS), row = rg_erow(idx, GROUPS);
    float4 v[CS];
#pragma unroll
    for (int pr = 0; pr < CS; ++pr) {
      if constexpr (CS > 1) {
        v[pr] = ld_peer_v4(paddr + 4u * static_cast<uint32_t>(tok * ROWS + row), static_cast<uint32_t>(pr));
      } else {
        v[pr] = *reinterpret_cast<const float4*>(part + tok * ROWS + row);
      }
    }
    float4 s = v[0];
#pragma unroll
    for (int pr = 1; pr < CS; ++pr) {
      s.x += v[pr].x; s.y += v[pr].y; s.z += v[pr].z; s.w += v[pr].w;
    }
    if (tok < m) *reinterpret_cast<float4*>(out + static_cast<long long>(tok) * 256 + cl * ROWS + row) = s;
  }
  if constexpr (CS > 1) cluster_sync<CS>();                // peers may still read this CTA's tile
}

}  // namespace rg

@@INSTANCES@@
"""

# (RTC, NTW, CS, XS, D) instances compiled into the module.
RG_INSTANCES = ((1, 4, 16, 3, 2),)
RG_CHECK_INSTANCES = ((1, 1, 8, 3, 1), (2, 2, 16, 4, 2), (2, 16, 8, 3, 1), (1, 8, 16, 3, 2))


def rg_name(inst):
    return "rg_run_" + "_".join(str(int(v)) for v in inst)


def rg_mt(inst):
    rtc, ntw = inst[0], inst[1]
    return (4 // rtc) * 8 * ntw


def rg_smem(inst):
    rtc, ntw, cs, xs, d = inst
    return max(xs * rg_mt(inst) * 128, rg_mt(inst) * 16 * rtc * 4)


def rg_source(instances):
    wrappers = []
    for inst in instances:
        rtc, ntw, cs, xs, d = inst
        tmpl = f"{rtc}, {ntw}, {cs}, {xs}, {d}"
        mt = rg_mt(inst)
        wrappers.append(f"""
void {rg_name(inst)}(torch::Tensor wp, torch::Tensor x, torch::Tensor out) {{
  const int64_t m = x.size(0);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::Float, "rg: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && x.size(1) == 8192, "rg: layouts");
  TORCH_CHECK(m >= 1 && m <= {mt} && out.size(0) == m && out.size(1) == 256, "rg: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= 256LL * 8192 * 4, "rg: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0
              && reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0, "rg: unaligned");
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {cs};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(static_cast<unsigned>((16 / {rtc}) * {cs}), 1, 1);
  cfg.blockDim = dim3(128, 1, 1);
  cfg.dynamicSmemBytes = {rg_smem(inst)};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, rg::rg_kernel<{tmpl}>, reinterpret_cast<const uint4*>(wp.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()), reinterpret_cast<float*>(out.data_ptr()), static_cast<int>(m));
  TORCH_CHECK(err == cudaSuccess, "rg launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}}
void {rg_name(inst)}_attr() {{
  cudaFuncSetAttribute(rg::rg_kernel<{tmpl}>, cudaFuncAttributeMaxDynamicSharedMemorySize, {rg_smem(inst)});
  if ({cs} > 8) cudaFuncSetAttribute(rg::rg_kernel<{tmpl}>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
}}
int64_t {rg_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, rg::rg_kernel<{tmpl}>);
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return RG_SRC.replace("@@INSTANCES@@", "\n".join(wrappers))


def rg_decls(instances):
    out = []
    for inst in instances:
        name = rg_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor x, torch::Tensor out);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    return out


def rg_build(instances):
    from torch.utils.cpp_extension import load_inline
    decls = rg_decls(instances)
    return load_inline(
        name="okc_proj_rg", cpp_sources="\n".join(decls), cuda_sources=rg_source(instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_rg_"), verbose=False)

RG_ENABLED = True
RG_MOD = None


def rg_setup(state):
    """Pack the FP32 router as SGLang's BF16 hi + lo pair in A-fragment order (natural k), build every RG
    instance; a failed build leaves no module, so every RG shape runs its fallback."""
    global RG_MOD
    state["rg"] = rg_pack(state["router"])
    try:
        RG_MOD = rg_build(RG_INSTANCES)
        for inst in RG_INSTANCES:
            getattr(RG_MOD, rg_name(inst) + "_attr")()
    except Exception as exc:
        print(f"rg build FAILED, RG shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        RG_MOD = None


def run_rg(c, x, layer, n, state):
    """c: dict(kind="rg", RTC, NTW, CS, XS, D, fallback). Router only; m <= the instance's token count."""
    inst = (c["RTC"], c["NTW"], c["CS"], c["XS"], c["D"])
    m = x.shape[0]
    if RG_MOD is None or layer != "router" or m > rg_mt(inst):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, 256), dtype=torch.float32, device=x.device)
    getattr(RG_MOD, rg_name(inst))(state["rg"], x.contiguous(), out)
    return out


# ====================================================================== WG (proj/wg): wgmma GEMM with in-cluster split-K, m = 128/256

def wgmma_asm(n=128):
    """The inline asm of one wgmma.m64n{n}k16 (FP32 accumulate, A and B from descriptors, scale-d by a register)."""
    nd = n // 2
    regs = ", ".join(f"%{i}" for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 2}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, %{nd}, %{nd + 1}, p, 1, 1, 0, 0;\\n}}"\n'
            f'      : ' + ", ".join(f'"+f"(d[{i}])' for i in range(nd)) +
            f'\n      : "l"(da), "l"(db), "r"(scale_d));')


def wgmma_rs_asm(n=128):
    """One wgmma.m64n{n}k16 with A from four registers (RS form), B from a descriptor, scale-d by a register."""
    nd = n // 2
    regs = ", ".join(f"%{i}" for i in range(nd))
    return (f'asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{nd + 5}, 0;\\n'
            f'wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{regs}}}, {{%{nd}, %{nd + 1}, %{nd + 2}, %{nd + 3}}}, '
            f'%{nd + 4}, p, 1, 1, 0;\\n}}"\n'
            f'      : ' + ", ".join(f'"+f"(d[{i}])' for i in range(nd)) +
            f'\n      : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "l"(db), "r"(scale_d));')


WG_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace wg {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint64_t* bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(bar)) : "memory");
}

// Bounded wait for the phase of the given parity; after a failed wait every later one returns at once
// (flag in shared memory), so a protocol error costs at most ~one bounded spin, never a hang.
__device__ __forceinline__ void mbar_wait(uint64_t* bar, uint32_t parity, volatile uint32_t* failed) {
  uint32_t done = 0;
  for (int it = 0; it < (1 << 20) && !done && !*failed; ++it) {
    asm volatile("{\n.reg .pred p;\nmbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\nselp.u32 %0, 1, 0, p;\n}"
                 : "=r"(done) : "r"(smem_u32(bar)), "r"(parity) : "memory");
  }
  if (!done) *failed = 1u;
}

__device__ __forceinline__ void tma_load_2d(void* dst, const CUtensorMap* map, uint64_t* bar, int c0, int c1) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%3, %4}], [%2];"
               :: "r"(smem_u32(dst)), "l"(reinterpret_cast<uint64_t>(map)), "r"(smem_u32(bar)), "r"(c0), "r"(c1)
               : "memory");
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit() { asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wgmma_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;" :: "n"(N) : "memory"); }

// K-major, 128B-swizzle shared memory descriptor (PTX ISA 9.4 sec. 9.7.17.5.1.2.2): start address,
// LBO 1 (unused for swizzled K-major), SBO 1024 B (next 8 rows), base offset 0 (1024-B aligned tiles),
// swizzle mode 1 (128B).
__device__ __forceinline__ uint64_t wg_desc(uint32_t saddr) {
  return static_cast<uint64_t>((saddr & 0x3FFFFu) >> 4) | (static_cast<uint64_t>(1u) << 16) |
         (static_cast<uint64_t>(1024u >> 4) << 32) | (static_cast<uint64_t>(1u) << 62);
}

__device__ __forceinline__ void wgmma_128(float (&d)[64], uint64_t da, uint64_t db, uint32_t scale_d) {
  @@WGMMA128@@
}

template <int KS>
__device__ __forceinline__ void cluster_sync() {
  if constexpr (KS > 1) {
    asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  } else {
    __syncthreads();
  }
}

template <int KS>
__device__ __forceinline__ uint32_t cluster_rank() {
  if constexpr (KS > 1) {
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

// Index helpers (check_wg.py evaluates their return expressions): first and end chunk of rank r of ks;
// accumulator element e of consumer warpgroup w's thread (warp q, lane g/t) -> tile row and token;
// rows rank r reduces; byte offset of the k16 step j inside a stage's A rows of warpgroup w / B.
__device__ __forceinline__ int wg_c0(int r, int chunks, int ks) { return r * chunks / ks; }
__device__ __forceinline__ int wg_row(int w, int q, int g, int e) { return 64 * w + 16 * q + g + 8 * ((e >> 1) & 1); }
__device__ __forceinline__ int wg_tok(int t, int e) { return 8 * (e >> 2) + 2 * t + (e & 1); }
__device__ __forceinline__ int wg_aoff(int w, int j) { return 8192 * w + 32 * j; }
__device__ __forceinline__ int wg_boff(int j) { return 32 * j; }

__device__ __forceinline__ float4 ld_peer_v4(uint32_t local_addr, uint32_t rank) {
  uint32_t remote;
  float4 v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local_addr), "r"(rank));
  asm volatile("ld.shared::cluster.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(remote) : "memory");
  return v;
}

__device__ __forceinline__ uint32_t bf16x2_bits(float lo, float hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (static_cast<uint32_t>(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}

// Epilogue items (check_wg.py): item idx of rank r covers token wg_etok and rows wg_erow .. + 7.
__device__ __forceinline__ int wg_etok(int idx, int groups) { return idx / groups; }
__device__ __forceinline__ int wg_erow(int idx, int groups, int r, int share) { return r * share + (idx % groups) * 8; }

// Shared epilogue (all 384 threads): the consumers' accumulators go to shared memory as [token][row] FP32
// over the idle stages; after a cluster barrier rank r sums rows [r * 128/KS, ...) over its KS peers, 8
// consecutive rows of one token per item (two 16-B loads per peer, all issued before the sum), scales each
// row by 2^(base - 112) for WGF, and stores the 8 BF16 as one 16-B vector (coalesced along the token row).
template <int KS, bool SCALED>
__device__ __forceinline__ void wg_epilogue(const float (&d)[64], uint8_t* smem, const uint32_t* bases, int wgi, int tid,
                                            uint32_t rank, __nv_bfloat16* out, long long n_rows, long long row0,
                                            long long tok0, int m) {
  const int lane = tid & 31, q = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
  float* red = reinterpret_cast<float*>(smem);
  if (wgi > 0) {
    asm volatile("bar.sync 1, 256;" ::: "memory");          // both consumer warpgroups done with the stages
#pragma unroll
    for (int e = 0; e < 64; ++e) red[wg_tok(t, e) * 128 + wg_row(wgi - 1, q, g, e)] = d[e];
  }
  cluster_sync<KS>();
  constexpr int SHARE = 128 / KS, GROUPS = SHARE / 8;
  const uint32_t red_addr = smem_u32(red);
  for (int idx = tid; idx < GROUPS * 128; idx += 384) {
    const int tk = wg_etok(idx, GROUPS), row = wg_erow(idx, GROUPS, static_cast<int>(rank), SHARE);
    float4 lo[KS], hi[KS];
#pragma unroll
    for (int p = 0; p < KS; ++p) {
      if constexpr (KS > 1) {
        const uint32_t a = red_addr + 4u * static_cast<uint32_t>(tk * 128 + row);
        lo[p] = ld_peer_v4(a, static_cast<uint32_t>(p));
        hi[p] = ld_peer_v4(a + 16u, static_cast<uint32_t>(p));
      } else {
        lo[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row);
        hi[p] = *reinterpret_cast<const float4*>(red + tk * 128 + row + 4);
      }
    }
    float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < KS; ++p) {
      v[0] += lo[p].x; v[1] += lo[p].y; v[2] += lo[p].z; v[3] += lo[p].w;
      v[4] += hi[p].x; v[5] += hi[p].y; v[6] += hi[p].z; v[7] += hi[p].w;
    }
    if constexpr (SCALED) {
#pragma unroll
      for (int i = 0; i < 8; ++i) v[i] *= __uint_as_float((bases[row + i] + 15u) << 23);   // 2^(base - 112)
    }
    const long long tok = tok0 + tk;
    if (tok < m) {
      *reinterpret_cast<uint4*>(out + tok * n_rows + row0 + row) =
          make_uint4(bf16x2_bits(v[0], v[1]), bf16x2_bits(v[2], v[3]), bf16x2_bits(v[4], v[5]), bf16x2_bits(v[6], v[7]));
    }
  }
  if constexpr (KS > 1) cluster_sync<KS>();                // peers may still read this CTA's tile
}

constexpr int STAGE_BYTES = 32768;     // A 16 KB + B 16 KB
constexpr int A_BYTES = 16384;

template <int KS, int ST, int TT>
__global__ void __launch_bounds__(384, 1)
wg_kernel(const __grid_constant__ CUtensorMap tw, const __grid_constant__ CUtensorMap tx,
          __nv_bfloat16* __restrict__ out, int n_rows, int chunks, int m) {
  static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "row shares need KS | 128");
  static_assert(ST >= 2 && ST * STAGE_BYTES >= 128 * 128 * 4, "the split-K tile reuses the stage memory");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * STAGE_BYTES);
  uint64_t* empty = full + ST;
  volatile uint32_t* failed = reinterpret_cast<volatile uint32_t*>(empty + ST);
  const int tid = threadIdx.x, wgi = tid >> 7;
  const uint32_t rank = cluster_rank<KS>();
  const int tile = blockIdx.x / KS;
  const int rowtile = tile / TT, tt = tile % TT;
  const int c0 = wg_c0(static_cast<int>(rank), chunks, KS), c1 = wg_c0(static_cast<int>(rank) + 1, chunks, KS);
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1u);
      mbar_init(&empty[s], 256u);
    }
    *failed = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();

  float d[64];
  if (wgi == 0) {
    if (tid == 0) {
      for (int c = c0, i = 0; c < c1; ++c, ++i) {
        const int s = i % ST;
        mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);
        mbar_expect_tx(&full[s], static_cast<uint32_t>(STAGE_BYTES));
        tma_load_2d(smem + s * STAGE_BYTES, &tw, &full[s], 64 * c, 128 * rowtile);
        tma_load_2d(smem + s * STAGE_BYTES + A_BYTES, &tx, &full[s], 64 * c, 128 * tt);
      }
    }
  } else {
    const int w = wgi - 1;
#pragma unroll
    for (int e = 0; e < 64; ++e) d[e] = 0.0f;
    for (int c = c0, i = 0; c < c1; ++c, ++i) {
      const int s = i % ST;
      mbar_wait(&full[s], static_cast<uint32_t>((i / ST) & 1), failed);
      const uint32_t a0 = smem_u32(smem + s * STAGE_BYTES), b0 = smem_u32(smem + s * STAGE_BYTES + A_BYTES);
      wgmma_fence();
#pragma unroll
      for (int j = 0; j < 4; ++j) wgmma_128(d, wg_desc(a0 + wg_aoff(w, j)), wg_desc(b0 + wg_boff(j)), 1u);
      wgmma_commit();
      wgmma_wait<1>();
      if (i > 0) mbar_arrive(&empty[(i - 1) % ST]);
    }
    wgmma_wait<0>();
  }

  wg_epilogue<KS, false>(d, smem, nullptr, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);
}

// ---------------- WGF: the same pipeline with 12-bit weights in registers (RS wgmma, P31) ----------------
// A = the warp's 16-row tile of F12M planes in natural k order (f12m_layout natural=True), decoded per k16
// step in 7 integer ops (scaled by 2^(112 - base) per row); B = x from the TMA stage. Escapes are applied
// in the loop, after each chunk's MMAs completed, from the x values already in shared memory; each row's
// 2^(base - 112) is applied once after the split-K sum. Exact, no tail loads.
__device__ __forceinline__ uint32_t wword(const uint4& v, int i) { return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w; }

__device__ __forceinline__ uint4 wgf_decode(uint32_t q, uint32_t l0, uint32_t l1) {
  const uint32_t r4 = __funnelshift_l(q, q, 4);
  const uint32_t ha = (q & 0x87878787u) | 0x38383838u;
  const uint32_t hb = (r4 & 0x87878787u) | 0x38383838u;
  return make_uint4(__byte_perm(l0, ha, 0x5140u), __byte_perm(l0, ha, 0x7362u),
                    __byte_perm(l1, hb, 0x5140u), __byte_perm(l1, hb, 0x7362u));
}

__device__ __forceinline__ void wgmma_rs_128(float (&d)[64], const uint4& a, uint64_t db, uint32_t scale_d) {
  @@WGMMARS128@@
}

__device__ __forceinline__ void ld_wf(const uint4* p, uint4& v) {
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
}

// Index helpers (check_wg.py): uint4 index of plane slot j of 16-row tile `tile`, chunk c (chs chunks per
// row); the tile of warp q of consumer warpgroup w in row tile rt; byte offset of (token, kk) in a 128B-
// swizzled 128 x 64 BF16 stage.
__device__ __forceinline__ long long wgf_widx(long long tile, int chs, int c, int slot, int lane) { return ((tile * chs + c) * 3 + slot) * 32 + lane; }
__device__ __forceinline__ long long wgf_tile(long long rt, int w, int q) { return 8 * rt + 4 * w + q; }
__device__ __forceinline__ int wgf_xoff(int tok, int kk) { return tok * 128 + ((((2 * kk) >> 4) ^ (tok & 7)) << 4) + ((2 * kk) & 15); }

constexpr int XB = 16384;   // one x stage: 128 tokens x 64 k BF16

template <int KS, int ST, int TT, int D>
__global__ void __launch_bounds__(384, 1)
wgf_kernel(const __grid_constant__ CUtensorMap tx, const uint4* __restrict__ wp, const uint4* __restrict__ rec,
           __nv_bfloat16* __restrict__ out, int n_rows, int chunks, int m) {
  static_assert(KS == 1 || KS == 2 || KS == 4 || KS == 8, "row shares need KS | 128");
  static_assert(ST >= 4 && D >= 1, "the 64 KB split-K tile reuses the x stages");
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~static_cast<uintptr_t>(1023));
  uint64_t* full = reinterpret_cast<uint64_t*>(smem + ST * XB);
  uint64_t* empty = full + ST;
  volatile uint32_t* failed = reinterpret_cast<volatile uint32_t*>(empty + ST);
  uint32_t* bases = reinterpret_cast<uint32_t*>(smem + ST * XB + 16 * ST + 16);   // [128] base per tile row
  const int tid = threadIdx.x, wgi = tid >> 7;
  const uint32_t rank = cluster_rank<KS>();
  const int tile = blockIdx.x / KS;
  const int rowtile = tile / TT, tt = tile % TT;
  const int c0 = wg_c0(static_cast<int>(rank), chunks, KS), c1 = wg_c0(static_cast<int>(rank) + 1, chunks, KS);
  if (tid == 0) {
    for (int s = 0; s < ST; ++s) {
      mbar_init(&full[s], 1u);
      mbar_init(&empty[s], 256u);
    }
    *failed = 0u;
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  if (wgi == 0) bases[tid] = rec[2LL * (128LL * rowtile + tid)].x & 0xFFu;   // row bases for the epilogue
  __syncthreads();

  float d[64];
  if (wgi == 0) {
    if (tid == 0) {
      for (int c = c0, i = 0; c < c1; ++c, ++i) {
        const int s = i % ST;
        mbar_wait(&empty[s], static_cast<uint32_t>(((i / ST) & 1) ^ 1), failed);
        mbar_expect_tx(&full[s], static_cast<uint32_t>(XB));
        tma_load_2d(smem + s * XB, &tx, &full[s], 64 * c, 128 * tt);
      }
    }
  } else {
    const int w = wgi - 1, lane = tid & 31, q = (tid >> 5) & 3, g = lane >> 2, t = lane & 3;
    const long long mytile = wgf_tile(rowtile, w, q);
    const int chs = chunks;
    // the thread's two rows (g, g + 8 of its tile): records (base | count << 8, escapes k | bits << 16)
    uint4 rc[2][2];
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const long long row = 16 * mytile + g + 8 * h;
      rc[h][0] = rec[2 * row];
      rc[h][1] = rec[2 * row + 1];
    }
    uint4 wv[D][3];
    const int nch = c1 - c0;
#pragma unroll
    for (int sl = 0; sl < D; ++sl)
      if (sl < nch)
#pragma unroll
        for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c0 + sl, j, lane), wv[sl][j]);
#pragma unroll
    for (int e = 0; e < 64; ++e) d[e] = 0.0f;
    for (int i0 = 0; i0 < nch; i0 += D) {
#pragma unroll
      for (int sl = 0; sl < D; ++sl) {
        const int i = i0 + sl;
        if (i < nch) {
          const int c = c0 + i, s = i % ST;
          uint4 a[4];
#pragma unroll
          for (int j = 0; j < 4; ++j) {
            const uint4& lv = wv[sl][1 + (j >> 1)];
            a[j] = wgf_decode(wword(wv[sl][0], j), wword(lv, 2 * (j & 1)), wword(lv, 2 * (j & 1) + 1));
          }
          if (i + D < nch) {
#pragma unroll
            for (int j = 0; j < 3; ++j) ld_wf(wp + wgf_widx(mytile, chs, c + D, j, lane), wv[sl][j]);
          }
          mbar_wait(&full[s], static_cast<uint32_t>((i / ST) & 1), failed);
          const uint32_t b0 = smem_u32(smem + s * XB);
          wgmma_fence();
#pragma unroll
          for (int j = 0; j < 4; ++j) wgmma_rs_128(d, a[j], wg_desc(b0 + wg_boff(j)), 1u);
          wgmma_commit();
          wgmma_wait<0>();
          // escapes with k in this chunk: d += x * (w_true - w_dec), scaled by 2^(112 - base); x from the stage
          const unsigned short* xs = reinterpret_cast<const unsigned short*>(smem + s * XB);
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const uint32_t base = rc[h][0].x & 0xFFu, count = (rc[h][0].x >> 8) & 0xFFu;
            const uint32_t esc[7] = {rc[h][0].y, rc[h][0].z, rc[h][0].w, rc[h][1].x, rc[h][1].y, rc[h][1].z, rc[h][1].w};
#pragma unroll
            for (int j = 0; j < 7; ++j) {
              const int kk = static_cast<int>(esc[j] & 0xFFFFu) - 64 * c;
              if (j < count && kk >= 0 && kk < 64) {
                const float wt = ldexpf(__uint_as_float(esc[j] & 0xFFFF0000u), 112 - static_cast<int>(base));
#pragma unroll
                for (int e = 0; e < 64; ++e) {
                  if (((e >> 1) & 1) == h) {
                    const float xk = __uint_as_float(static_cast<uint32_t>(xs[wgf_xoff(wg_tok(t, e), kk) >> 1]) << 16);
                    d[e] = fmaf(-xk, 3.0517578125e-05f, d[e]);   // 2^-15: what the stored escape decodes to
                    d[e] = fmaf(xk, wt, d[e]);
                  }
                }
              }
            }
          }
          mbar_arrive(&empty[s]);
        }
      }
    }
  }

  wg_epilogue<KS, true>(d, smem, bases, wgi, tid, rank, out, n_rows, 128LL * rowtile, 128LL * tt, m);
}

}  // namespace wg

// cuTensorMapEncodeTiled's signature (cuda.h, CUDA 12.0+), fetched through the runtime so no libcuda link is needed.
typedef CUresult (*wg_encode_t)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                                const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                                CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static wg_encode_t wg_encode_fn() {
  static wg_encode_t fn = nullptr;
  if (fn == nullptr) {
    void* p = nullptr;
    cudaDriverEntryPointQueryResult q;
    TORCH_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q) == cudaSuccess
                && q == cudaDriverEntryPointSuccess && p != nullptr, "wg: cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<wg_encode_t>(p);
  }
  return fn;
}

// 2D BF16 tensor [rows][k] (row-major), box 64 k x 128 rows, 128B swizzle, zero fill out of bounds.
static CUtensorMap wg_map(const void* ptr, int64_t rows, int64_t k) {
  CUtensorMap map;
  cuuint64_t dims[2] = {static_cast<cuuint64_t>(k), static_cast<cuuint64_t>(rows)};
  cuuint64_t strides[1] = {static_cast<cuuint64_t>(k * 2)};
  cuuint32_t box[2] = {64u, 128u};
  cuuint32_t estr[2] = {1u, 1u};
  const CUresult r = wg_encode_fn()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, const_cast<void*>(ptr), dims, strides, box, estr,
                                    CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                    CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "wg: cuTensorMapEncodeTiled failed (", static_cast<int>(r), ")");
  return map;
}

@@INSTANCES@@
"""

# (KS, ST, TT) WG instances and (KS, ST, TT, D) WGF instances compiled into the module.
WG_INSTANCES = ((2, 4, 1),)
WGF_INSTANCES = ()
WG_CHECK_INSTANCES = ((1, 4, 1), (4, 4, 2), (8, 3, 1))
WGF_CHECK_INSTANCES = ((1, 4, 1, 2), (4, 4, 2, 1), (8, 5, 1, 4))


def wg_name(inst):
    return "wg_run_" + "_".join(str(int(v)) for v in inst)


def wgf_name(inst):
    return "wgf_run_" + "_".join(str(int(v)) for v in inst)


def wg_smem(inst):
    ks, st, tt = inst
    return st * 32768 + 1024 + 2 * st * 8 + 16


def wgf_smem(inst):
    ks, st, tt, d = inst
    return st * 16384 + 1024 + 2 * st * 8 + 16 + 512


def _launch(kernel, smem, ks, tt, args):
    return f"""  const unsigned ctas = static_cast<unsigned>((n_rows / 128) * {tt} * {ks});
  cudaLaunchConfig_t cfg = {{}};
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = {ks};
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.gridDim = dim3(ctas, 1, 1);
  cfg.blockDim = dim3(384, 1, 1);
  cfg.dynamicSmemBytes = {smem};
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  const cudaError_t err = cudaLaunchKernelEx(&cfg, {kernel}, {args});
  TORCH_CHECK(err == cudaSuccess, "wg launch failed: ", cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();"""


def wg_source(instances, wgf_instances=()):
    wrappers = []
    for inst in instances:
        ks, st, tt = inst
        kern = f"wg::wg_kernel<{ks}, {st}, {tt}>"
        wrappers.append(f"""
void {wg_name(inst)}(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(w.scalar_type() == at::ScalarType::BFloat16 && x.scalar_type() == at::ScalarType::BFloat16
              && out.scalar_type() == at::ScalarType::BFloat16, "wg: dtypes");
  TORCH_CHECK(w.is_contiguous() && x.is_contiguous() && out.is_contiguous() && w.size(1) == k && w.size(0) >= n_rows, "wg: layouts");
  TORCH_CHECK(n_rows % 128 == 0 && k % 64 == 0 && k / 64 >= {ks} && m >= 1 && m <= 128 * {tt} && out.numel() == m * n_rows, "wg: shape");
  const CUtensorMap tw = wg_map(w.data_ptr(), n_rows, k);
  const CUtensorMap tx = wg_map(x.data_ptr(), m, k);
{_launch(kern, wg_smem(inst), ks, tt, "tw, tx, reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_rows), static_cast<int>(k / 64), static_cast<int>(m)")}
}}
void {wg_name(inst)}_attr() {{
  cudaFuncSetAttribute({kern}, cudaFuncAttributeMaxDynamicSharedMemorySize, {wg_smem(inst)});
}}
int64_t {wg_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, {kern});
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    for inst in wgf_instances:
        ks, st, tt, d = inst
        kern = f"wg::wgf_kernel<{ks}, {st}, {tt}, {d}>"
        wrappers.append(f"""
void {wgf_name(inst)}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows) {{
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.scalar_type() == at::ScalarType::BFloat16 && out.scalar_type() == at::ScalarType::BFloat16, "wgf: dtypes");
  TORCH_CHECK(x.is_contiguous() && out.is_contiguous() && wp.is_contiguous() && rec.is_contiguous(), "wgf: layouts");
  TORCH_CHECK(n_rows % 128 == 0 && k % 64 == 0 && k / 64 >= {ks} && m >= 1 && m <= 128 * {tt} && out.numel() == m * n_rows, "wgf: shape");
  TORCH_CHECK(wp.numel() * wp.element_size() >= (n_rows / 16) * (k / 64) * 1536 && rec.numel() >= n_rows * 8, "wgf: planes too small");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(wp.data_ptr()) % 16 == 0 && reinterpret_cast<uintptr_t>(rec.data_ptr()) % 16 == 0, "wgf: unaligned");
  const CUtensorMap tx = wg_map(x.data_ptr(), m, k);
{_launch(kern, wgf_smem(inst), ks, tt, "tx, reinterpret_cast<const uint4*>(wp.data_ptr()), reinterpret_cast<const uint4*>(rec.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), static_cast<int>(n_rows), static_cast<int>(k / 64), static_cast<int>(m)")}
}}
void {wgf_name(inst)}_attr() {{
  cudaFuncSetAttribute({kern}, cudaFuncAttributeMaxDynamicSharedMemorySize, {wgf_smem(inst)});
}}
int64_t {wgf_name(inst)}_info() {{
  cudaFuncAttributes a;
  cudaFuncGetAttributes(&a, {kern});
  return static_cast<int64_t>(a.numRegs) * 1000000 + static_cast<int64_t>(a.localSizeBytes);
}}""")
    return (WG_SRC.replace("@@WGMMA128@@", wgmma_asm(128)).replace("@@WGMMARS128@@", wgmma_rs_asm(128))
            .replace("@@INSTANCES@@", "\n".join(wrappers)))


def wg_decls(instances, wgf_instances=()):
    out = []
    for inst in instances:
        name = wg_name(inst)
        out.append(f"void {name}(torch::Tensor w, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    for inst in wgf_instances:
        name = wgf_name(inst)
        out.append(f"void {name}(torch::Tensor wp, torch::Tensor rec, torch::Tensor x, torch::Tensor out, int64_t n_rows);")
        out.append(f"void {name}_attr();")
        out.append(f"int64_t {name}_info();")
    return out


def wg_build(instances, wgf_instances=()):
    from torch.utils.cpp_extension import load_inline
    decls = wg_decls(instances, wgf_instances)
    return load_inline(
        name="okc_proj_wg", cpp_sources="\n".join(decls), cuda_sources=wg_source(instances, wgf_instances),
        functions=[d.split("(")[0].split()[-1] for d in decls],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
        build_directory=tempfile.mkdtemp(prefix="okc_proj_wg_"), verbose=False)

WG_ENABLED = True
WG_MOD = None


def wg_setup(state):
    """Build every WG/WGF instance and allow its dynamic shared memory; WGF also packs the BF16 layers into
    F12M planes in natural k order. A failed build leaves no module, so every WG/WGF shape runs its
    fallback (slower, still correct)."""
    global WG_MOD
    state["wgf"] = {}
    if WGF_INSTANCES:
        for name in ("qkv_gate", "o", "pooler", "unpooler"):
            planes = f12m_device_planes(state[name], natural=True)
            if planes is not None:
                state["wgf"][name] = planes
    try:
        WG_MOD = wg_build(WG_INSTANCES, WGF_INSTANCES)
        for inst in WG_INSTANCES:
            getattr(WG_MOD, wg_name(inst) + "_attr")()
        for inst in WGF_INSTANCES:
            getattr(WG_MOD, wgf_name(inst) + "_attr")()
    except Exception as exc:
        print(f"wg build FAILED, WG shapes fall back: {type(exc).__name__}: {' '.join(str(exc).split())[-600:]}", flush=True)
        WG_MOD = None


def run_wg(c, x, layer, n, state):
    """c: dict(kind="wg", KS, ST, TT, fallback). BF16 layers, m <= 128 TT."""
    m, k = x.shape
    if WG_MOD is None or m > 128 * c["TT"] or state[layer].dtype != torch.bfloat16:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(WG_MOD, wg_name((c["KS"], c["ST"], c["TT"])))(state[layer], x.contiguous(), out, n)
    return out


def run_wgf(c, x, layer, n, state):
    """c: dict(kind="wgf", KS, ST, TT, D, fallback). 12-bit planes (natural k order), m <= 128 TT."""
    m, k = x.shape
    pl = state.get("wgf", {}).get(layer)
    if WG_MOD is None or pl is None or m > 128 * c["TT"]:
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    getattr(WG_MOD, wgf_name((c["KS"], c["ST"], c["TT"], c["D"])))(pl["p"], pl["rec"], x.contiguous(), out, n)
    return out
