# okc: problem=muse_moe_w4a16
"""E1 lab, for an `okc check` run only (never a scored run).

kernel_fn is v0 (submission.py): SGLang's Marlin path with the tile height sized for this GPU's
share of the routes, so the timed part reports v0 against SGLang on every scored shape.

prepare() also measures, on synthetic inputs only, and prints the results:
  env      versions of the libraries in the image, GPU properties
  sweep    Marlin's time at every tile height for each scored token count
  bw       HBM read bandwidth of a plain streaming Triton kernel
  graph    the cost of one more kernel inside a CUDA graph
  decode   the Marlin E2M1 -> BF16 bit trick on the GPU: exactness, the subnormal multiply, speed
Each section is time-boxed and wrapped so a failure only prints and moves on."""

from typing import TypedDict

import importlib.metadata
import os
import statistics
import time
import types

import torch
import triton
import triton.language as tl
from sgl_kernel.scalar_type import scalar_types

from nvfp4 import dequantize, random_weights, unpack_e2m1
from sglang.kernels.ops.activation.activation import silu_and_mul
from sglang.kernels.ops.moe.moe_wna16_marlin import moe_wna16_marlin_gemm
from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
    fused_post_expert_norm_weighted_sum,
)
from sglang.srt.layers.moe.fused_moe_triton import moe_align_block_size
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace
from sglang.srt.layers.quantization.marlin_utils_fp4 import prepare_moe_nvfp4_layer_for_marlin


class Weights(TypedDict):
    w13: torch.Tensor          # uint8 [256, 2I, H/2], gate rows [0, I) then up rows [I, 2I), packed E2M1
    w13_sf: torch.Tensor       # float8_e4m3fn [256, 2I, H/16]
    w13_gs: torch.Tensor       # float32 [256]
    w2: torch.Tensor           # uint8 [256, H, I/2]
    w2_sf: torch.Tensor        # float8_e4m3fn [256, H, I/16]
    w2_gs: torch.Tensor        # float32 [256]
    post_norm_w: torch.Tensor  # bfloat16 [H], the post-expert RMSNorm's weight (applied as 1 + w)


class Config(TypedDict):
    bs: int           # tokens, M
    d_hidden: int     # H = 4096
    d_expert: int     # I = 4096
    n_experts: int    # 256
    local_start: int  # this GPU computes experts [local_start, local_start + local_count)
    local_count: int  # 32
    top_k: int        # 8
    sharding: str     # "ep8"


class Inputs(TypedDict):
    hidden_states: torch.Tensor  # bfloat16 [M, H]
    topk_ids: torch.Tensor       # int32 [M, 8], global expert ids (0-255)
    topk_weights: torch.Tensor   # float32 [M, 8]
    config: Config


N_EXPERTS = 256
TOP_K = 8
BLOCK_SIZE_M_CANDIDATES = (8, 16, 32, 48, 64)
POST_NORM_EPS = 1e-8
BLOCK_SIZE_M_BY_TOKENS: dict[int, int] = {}

SCORED_TOKENS = (1, 8, 16, 32, 128, 256)
LAB_BUDGET_S = 170.0
FLUSH_BYTES = 512 << 20  # larger than H200's 50 MB L2, as the judge's cold-L2 read


# ---------------------------------------------------------------- v0 (same as submission.py)

def block_size_m(num_tokens: int) -> int:
    if num_tokens in BLOCK_SIZE_M_BY_TOKENS:
        return BLOCK_SIZE_M_BY_TOKENS[num_tokens]
    per_expert = num_tokens * TOP_K / N_EXPERTS
    for size in BLOCK_SIZE_M_CANDIDATES:
        if per_expert / size < 0.9:
            return size
    return BLOCK_SIZE_M_CANDIDATES[-1]


def sglang_block_size_m(num_tokens: int, local_count: int) -> int:
    """MuseSpark1xFp4Experts' own rule, for comparison."""
    for size in BLOCK_SIZE_M_CANDIDATES:
        if num_tokens * TOP_K / local_count / size < 0.9:
            return size
    return BLOCK_SIZE_M_CANDIDATES[-1]


def _marlin_layer(weights, start, count):
    sl = slice(start, start + count)
    layer = types.SimpleNamespace(
        quant_config=types.SimpleNamespace(group_size=16),
        moe_runner_config=types.SimpleNamespace(is_gated=True),
        intermediate_size_per_partition=weights["w13"].shape[1] // 2,
        params_dtype=torch.bfloat16,
        w13_weight=weights["w13"][sl].contiguous(),
        w2_weight=weights["w2"][sl].contiguous(),
        w13_weight_scale=weights["w13_sf"][sl].contiguous(),
        w2_weight_scale=weights["w2_sf"][sl].contiguous(),
        w13_weight_scale_2=weights["w13_gs"][sl].contiguous(),
        w2_weight_scale_2=weights["w2_gs"][sl].contiguous(),
    )
    prepare_moe_nvfp4_layer_for_marlin(layer)
    return layer


def run_experts(x, topk_ids, topk_weights, state, bank, moe_block):
    num_tokens, size_k = x.shape
    size_n = bank["size_n"]
    gemm1_n = 2 * size_n
    expert_map = bank["expert_map"]

    local_ids = topk_ids.to(torch.int32)
    if expert_map is not None:
        local_ids = expert_map[local_ids]
    local_ids = local_ids.contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()

    if num_tokens == 1 and expert_map is None:
        from sglang.kernels.ops.moe.moe_align_single_token import moe_align_single_token

        sorted_ids, block_experts, n_padded = moe_align_single_token(local_ids, moe_block)
    else:
        sorted_ids, block_experts, n_padded = moe_align_block_size(
            local_ids, moe_block, N_EXPERTS if expert_map is not None else bank["count"])

    cache2 = torch.empty((num_tokens * TOP_K, size_n), device=x.device, dtype=x.dtype)
    cache13 = torch.empty((num_tokens * TOP_K * max(gemm1_n, size_k),), device=x.device, dtype=x.dtype)
    cache1 = cache13[: num_tokens * TOP_K * gemm1_n].view(-1, gemm1_n)
    cache3 = cache13[: num_tokens * TOP_K * size_k].view(-1, size_k)
    common = dict(moe_block_size=moe_block, mul_topk_weights=False, is_ep=expert_map is not None,
                  b_q_type=scalar_types.float4_e2m1f, is_k_full=True, use_atomic_add=False,
                  use_fp32_reduce=True, is_zp_float=False)
    ws = state["workspace"]
    cache1 = moe_wna16_marlin_gemm(
        x, cache1, bank["w13"], None, bank["w13_s"], bank["w13_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=TOP_K,
        size_m=num_tokens, size_n=gemm1_n, size_k=size_k, **common)
    silu_and_mul(cache1.view(-1, gemm1_n), cache2, expert_ids=local_ids.view(-1), expert_step=1)
    y = moe_wna16_marlin_gemm(
        cache2, cache3, bank["w2"], None, bank["w2_s"], bank["w2_gs"], None, None, None, ws,
        sorted_ids, block_experts, n_padded, topk_weights, top_k=1,
        size_m=num_tokens * TOP_K, size_n=size_k, size_k=size_n, **common).view(num_tokens, TOP_K, size_k)
    return fused_post_expert_norm_weighted_sum(
        y, topk_weights, norm_weight=state["norm_w"], eps=POST_NORM_EPS, weight_first=True,
        slot_expert_ids=local_ids)


def prepare(weights):
    device = weights["w13"].device
    banks = {}
    for start, count in ((0, 32),):
        layer = _marlin_layer(weights, start, count)
        expert_map = None
        if count != N_EXPERTS:
            expert_map = torch.full((N_EXPERTS,), -1, dtype=torch.int32, device=device)
            expert_map[start:start + count] = torch.arange(count, dtype=torch.int32, device=device)
        banks[(start, count)] = {
            "w13": layer.w13_weight.data, "w13_s": layer.w13_weight_scale.data, "w13_gs": layer.w13_weight_scale_2.data,
            "w2": layer.w2_weight.data, "w2_s": layer.w2_weight_scale.data, "w2_gs": layer.w2_weight_scale_2.data,
            "expert_map": expert_map, "count": count, "size_n": layer.intermediate_size_per_partition,
        }
    torch.cuda.synchronize()
    state = {
        "banks": banks,
        "workspace": marlin_make_workspace(device, max_blocks_per_sm=4),
        "norm_w": (weights["post_norm_w"].float() + 1.0).to(torch.bfloat16),
    }
    run_lab(weights, state)
    return state


def kernel_fn(data, state):
    cfg = data["config"]
    bank = state["banks"][(cfg["local_start"], cfg["local_count"])]
    x = data["hidden_states"].contiguous()
    return run_experts(x, data["topk_ids"], data["topk_weights"], state, bank, block_size_m(x.shape[0]))


# ---------------------------------------------------------------- lab helpers

def log(msg):
    print(f"[lab] {msg}", flush=True)


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
            log(f"{name}: FAILED {type(exc).__name__}: {str(exc)[:300]}")
        torch.cuda.synchronize()
        log(f"{name}: {time.monotonic() - t:.1f}s")

    def cold_l2(self):
        if self.flush is None:
            self.flush = torch.empty(FLUSH_BYTES // 8, dtype=torch.int64, device="cuda").fill_(1)
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


def synthetic_routing(bs, seed, device="cuda", local_count=32, popularity=0.03):
    """Top-8 of uniform scores plus a heavy-tailed per-expert popularity, rotated so experts
    0..local_count-1 are the busiest GPU's, like the judge's default routing."""
    gen = torch.Generator(device=device).manual_seed(seed)
    scores = torch.rand((bs, N_EXPERTS), generator=gen, device=device)
    z = torch.randn((N_EXPERTS,), generator=gen, device=device)
    u = torch.rand((N_EXPERTS,), generator=gen, device=device).clamp(min=1e-12)
    scores = scores + popularity * z / (-u.log()).sqrt()
    vals, ids = scores.topk(TOP_K, dim=-1)
    pairs = torch.bincount((ids // local_count).flatten(), minlength=N_EXPERTS // local_count)
    ids = (ids - local_count * pairs.argmax()) % N_EXPERTS
    return ids.to(torch.int32), (vals / vals.sum(-1, keepdim=True)).float()


def routing_stats(ids, local_count=32):
    local = ids[ids < local_count].long()
    counts = torch.bincount(local, minlength=local_count)
    return int(local.numel()), int((counts > 0).sum()), int(counts.max()) if local.numel() else 0


# ---------------------------------------------------------------- lab sections

def sec_env(lab):
    p = torch.cuda.get_device_properties(0)
    log(f"gpu {p.name} sms={p.multi_processor_count} mem={p.total_memory >> 30}GB "
        f"l2={getattr(p, 'L2_cache_size', 0) >> 20}MB cc={p.major}.{p.minor}")
    log(f"torch {torch.__version__} cuda {torch.version.cuda} triton {triton.__version__}")
    found = []
    for name in ("sglang", "sgl-kernel", "flashinfer-python", "flashinfer", "tilelang", "helion",
                 "nvidia-cutlass-dsl", "nvidia-cutlass", "cutlass", "deep-gemm", "deep_gemm",
                 "cuda-python", "nvidia-cuda-nvcc-cu12", "nvidia-cuda-runtime-cu12", "ninja", "numpy"):
        try:
            found.append(f"{name}={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    log("pkgs " + " ".join(found))
    from torch.utils import cpp_extension
    log(f"CUDA_HOME={cpp_extension.CUDA_HOME}")
    for mod in ("flashinfer", "tilelang", "cutlass", "deep_gemm", "helion"):
        try:
            m = importlib.import_module(mod)
            log(f"import {mod}: ok at {os.path.dirname(getattr(m, '__file__', '') or '')}")
        except Exception as exc:
            log(f"import {mod}: {type(exc).__name__}")


def sec_sweep(lab, weights, state):
    bank = state["banks"][(0, 32)]
    for bs in SCORED_TOKENS:
        for seed in (11, 12):
            if lab.left() < 20:
                log("sweep: out of budget")
                return
            ids, w = synthetic_routing(bs, 1000 * bs + seed)
            x = torch.randn((bs, 4096), device="cuda").to(torch.bfloat16)
            pairs, distinct, busiest = routing_stats(ids)
            ref = run_experts(x, ids, w, state, bank, 8).float()
            cells = []
            for size in BLOCK_SIZE_M_CANDIDATES:
                out = run_experts(x, ids, w, state, bank, size).float()
                err = float((out - ref).norm() / ref.norm().clamp(min=1e-12))
                us = lab.time_graph(lambda s=size: run_experts(x, ids, w, state, bank, s))
                cells.append((us, size, err))
            best = min(cells)
            row = " ".join(f"{s}:{us:.1f}" + ("!" if err > 1e-3 else "") for us, s, err in cells)
            log(f"sweep bs={bs} seed={seed} pairs={pairs} experts={distinct} max/expert={busiest} "
                f"sglang={sglang_block_size_m(bs, 32)} v0={block_size_m(bs)} best={best[1]} | us {row}")


@triton.jit
def _stream_read(ptr, out_ptr, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        acc ^= tl.load(ptr + offs, mask=offs < n, other=0)
    tl.store(out_ptr + pid, tl.reduce(acc, 0, _xor))


@triton.jit
def _xor(a, b):
    return a ^ b


def sec_bw(lab):
    nbytes = 1 << 30
    buf = torch.randint(0, 2**31 - 1, (nbytes // 4,), dtype=torch.int32, device="cuda")
    n = buf.numel()
    results = []
    for chunk in (1 << 14, 1 << 16, 1 << 18):
        for block in (1024, 4096):
            for warps in (4, 8):
                if lab.left() < 10:
                    break
                grid = (triton.cdiv(n, chunk),)
                out = torch.empty(grid[0], dtype=torch.int32, device="cuda")
                fn = lambda c=chunk, b=block, w=warps, o=out: _stream_read[grid](buf, o, n, CHUNK=c, BLOCK=b, num_warps=w)
                us = lab.time_graph(fn, reps=10, cold=False)
                results.append((nbytes / us / 1e6, chunk, block, warps))
    results.sort(reverse=True)
    log("bw TB/s (chunk,block,warps): " + " ".join(f"{r[0]:.2f}({r[1]},{r[2]},{r[3]})" for r in results[:8]))
    us = lab.time_graph(lambda: buf.sum(), reps=10, cold=False)
    log(f"bw torch int32 sum: {nbytes / us / 1e6:.2f} TB/s")


def sec_graph(lab):
    tiny = torch.zeros(1, device="cuda")
    rows = []
    for k in (1, 2, 4, 8, 16):
        def fn(k=k):
            for _ in range(k):
                tiny.add_(1.0)
        rows.append((k, lab.time_graph(fn, reps=30, cold=False)))
    slope = (rows[-1][1] - rows[0][1]) / (rows[-1][0] - rows[0][0])
    log("graph us per replay: " + " ".join(f"{k}:{us:.2f}" for k, us in rows) + f" | per extra kernel {slope:.2f} us")


# The Marlin E2M1 -> BF16 step on one 32-bit word (8 codes): four bf16x2 words, from the top
# nibble of each 16-bit half down. Each value is e2m1 * 2^-126 (subnormal for code 0.5).
_DECODE_ASM = """
{
.reg .b32 t, s, m;
and.b32 s, $4, 0x80008000;
and.b32 m, $4, 0x70007000;
shr.b32 m, m, 6;
or.b32 $0, s, m;
shl.b32 t, $4, 4;
and.b32 s, t, 0x80008000;
and.b32 m, t, 0x70007000;
shr.b32 m, m, 6;
or.b32 $1, s, m;
shl.b32 t, t, 4;
and.b32 s, t, 0x80008000;
and.b32 m, t, 0x70007000;
shr.b32 m, m, 6;
or.b32 $2, s, m;
shl.b32 t, t, 4;
and.b32 s, t, 0x80008000;
and.b32 m, t, 0x70007000;
shr.b32 m, m, 6;
or.b32 $3, s, m;
}
"""


@triton.jit
def _decode_words(w_ptr, out_ptr, n_words, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_words
    q = tl.load(w_ptr + offs, mask=mask, other=0)
    r0, r1, r2, r3 = tl.inline_asm_elementwise(
        asm=_DECODE_ASM, constraints="=r,=r,=r,=r,r", args=[q],
        dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=True, pack=1)
    tl.store(out_ptr + offs * 4 + 0, r0, mask=mask)
    tl.store(out_ptr + offs * 4 + 1, r1, mask=mask)
    tl.store(out_ptr + offs * 4 + 2, r2, mask=mask)
    tl.store(out_ptr + offs * 4 + 3, r3, mask=mask)


@triton.jit
def _mul_bf16x2(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    b = tl.load(b_ptr + offs, mask=mask, other=0)
    c = tl.inline_asm_elementwise(
        asm="mul.rn.bf16x2 $0, $1, $2;", constraints="=r,r,r", args=[a, b],
        dtype=tl.int32, is_pure=True, pack=1)
    tl.store(out_ptr + offs, c, mask=mask)


@triton.jit
def _mul_f32_triton(a_ptr, s_ptr, out_ptr, n, BLOCK: tl.constexpr):
    """Triton's own fp32 multiply on bf16 values widened by bit shift: does it flush subnormals?"""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask, other=0)
    lo = (a << 16).to(tl.float32, bitcast=True)
    s = tl.load(s_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, lo * s, mask=mask)


@triton.jit
def _decode_stream(w_ptr, out_ptr, n_words, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    """Load + decode every word of a large buffer; xor-reduce so nothing is optimized away."""
    pid = tl.program_id(0)
    base = pid.to(tl.int64) * CHUNK
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for i in range(0, CHUNK, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        q = tl.load(w_ptr + offs, mask=offs < n_words, other=0)
        r0, r1, r2, r3 = tl.inline_asm_elementwise(
            asm=_DECODE_ASM, constraints="=r,=r,=r,=r,r", args=[q],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=True, pack=1)
        acc ^= r0 ^ r1 ^ r2 ^ r3
    tl.store(out_ptr + pid, tl.reduce(acc, 0, _xor))


def _bf16_halves(words):
    """int32 bf16x2 words -> float32 [..., 2] (low half, high half), exact for subnormals."""
    return words.contiguous().view(torch.int16).view(torch.bfloat16).float().view(*words.shape, 2)


def sec_decode(lab):
    gen = torch.Generator(device="cuda").manual_seed(7)
    packed, sf, gs = random_weights(256, 4096, generator=gen, device="cuda")  # [256, 2048] uint8
    words = packed.contiguous().view(torch.int32)  # [256, 512], 8 codes per word
    n = words.numel()
    out = torch.empty((n, 4), dtype=torch.int32, device="cuda")
    _decode_words[(triton.cdiv(n, 1024),)](words, out, n, BLOCK=1024)
    halves = _bf16_halves(out)  # [n, 4, 2]: (round r, half h) -> element 8c + (3 - r) + 4h
    order = torch.tensor([[3, 7], [2, 6], [1, 5], [0, 4]], device="cuda")
    vals = torch.empty((n, 8), dtype=torch.float32, device="cuda")
    vals[:, order.flatten()] = halves.reshape(n, 8)
    want = unpack_e2m1(packed).reshape(n, 8)
    exact = torch.equal(vals * 2.0**126, want)
    log(f"decode bits: {'EXACT' if exact else 'MISMATCH'} vs nvfp4.unpack_e2m1 "
        f"(max diff {float((vals * 2.0**126 - want).abs().max()):.3g})")

    # Scale multiply on the GPU: (code * 2^-126) * (scale * 2^64) in bf16 must be exact.
    shift = 64
    scale = (sf.float() * 2.0**shift).to(torch.bfloat16)  # [256, 256], one per 16 values
    col_of_elem = torch.arange(4096, device="cuda") // 16
    flat_vals = vals.reshape(-1)
    # Pair consecutive decoded values into bf16x2 words for mul.rn.bf16x2.
    vb = flat_vals.to(torch.bfloat16).view(torch.int16).view(-1, 2)
    a_words = vb.contiguous().view(torch.int32).view(-1)
    s_full = scale.reshape(256, 256)[:, col_of_elem].reshape(-1)  # per element
    s_words = s_full.view(torch.int16).view(-1, 2).contiguous().view(torch.int32).view(-1)
    prod = torch.empty_like(a_words)
    m = a_words.numel()
    _mul_bf16x2[(triton.cdiv(m, 1024),)](a_words, s_words, prod, m, BLOCK=1024)
    got = _bf16_halves(prod).reshape(-1) * 2.0 ** (126 - shift)
    want_w = want.reshape(-1) * sf.float().reshape(256, 256)[:, col_of_elem].reshape(-1)
    log(f"decode mul.rn.bf16x2 with scale*2^{shift}: "
        f"{'EXACT' if torch.equal(got, want_w) else 'MISMATCH'} "
        f"(max rel {float(((got - want_w).abs() / want_w.abs().clamp(min=1e-30)).max()):.3g})")
    ref = dequantize(packed, sf, gs).reshape(-1)
    full = got * gs
    log(f"decode end-to-end vs nvfp4.dequantize: max abs diff {float((full - ref).abs().max()):.3g} "
        f"(max |w| {float(ref.abs().max()):.3g})")

    # Does Triton's own fp32 multiply keep subnormal inputs? (code 0.5 * 2^-126 = 2^-127)
    sub = torch.tensor([0x00000040, 0x00008040], dtype=torch.int32, device="cuda")  # bf16 bits 2^-127 (+/-)
    s = torch.full((2,), 2.0**126, device="cuda")
    o = torch.empty(2, device="cuda")
    _mul_f32_triton[(1,)](sub, s, o, 2, BLOCK=2)
    log(f"decode triton fp32 mul of subnormal: {o.tolist()} (want [0.5, -0.5]; 0 means flushed)")

    # Speed: load + decode a 1 GB buffer vs the plain streaming read in sec_bw.
    big = torch.randint(0, 2**31 - 1, ((1 << 30) // 4,), dtype=torch.int32, device="cuda")
    nb = big.numel()
    res = []
    for chunk in (1 << 16, 1 << 18):
        for block in (1024, 2048):
            for warps in (4, 8):
                if lab.left() < 8:
                    break
                grid = (triton.cdiv(nb, chunk),)
                o2 = torch.empty(grid[0], dtype=torch.int32, device="cuda")
                fn = lambda c=chunk, b=block, w=warps, o2=o2: _decode_stream[grid](big, o2, nb, CHUNK=c, BLOCK=b, num_warps=w)
                us = lab.time_graph(fn, reps=10, cold=False)
                res.append(((1 << 30) / us / 1e6, chunk, block, warps))
    res.sort(reverse=True)
    log("decode+load TB/s of packed weights (chunk,block,warps): "
        + " ".join(f"{r[0]:.2f}({r[1]},{r[2]},{r[3]})" for r in res[:6]))


def run_lab(weights, state):
    lab = Lab(LAB_BUDGET_S)
    log(f"start, budget {LAB_BUDGET_S:.0f}s")
    lab.section("env", sec_env)
    lab.section("graph", sec_graph)
    lab.section("sweep", sec_sweep, weights, state, min_left=30)
    lab.section("bw", sec_bw)
    lab.section("decode", sec_decode)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s")
