"""E1b: v1 timed by the judge, plus in prepare():
  postnorm  SGLang's small-batch post-norm path vs the fused kernel v1 always uses
  decode    the Marlin E2M1 -> BF16 trick on the GPU (fixed: the asm is an inline literal)
  headers   where CUTLASS / CuTe headers live in the image, for load_inline kernels
  nvcc      load_inline compile time for sm_90a, and __hmul2 on BF16 subnormals in CUDA"""

import importlib.util
import os
import tempfile

import triton.language as tl

from nvfp4 import dequantize, random_weights, unpack_e2m1

LAB_BUDGET_S = 200.0


def sec_postnorm(lab, state):
    from sglang.kernels.ops.muse_spark_v12.fused_post_expert_norm_weighted_sum_triton import (
        fused_post_expert_norm_weighted_sum,
    )
    norm_w = state["norm_w"]
    for bs in (1, 8, 16, 32):
        ids, w = synthetic_routing(bs, 77 + bs)
        local = torch.where(ids < 32, ids, torch.full_like(ids, -1)).contiguous()
        y = torch.randn((bs, TOP_K, 4096), device="cuda").to(torch.bfloat16)

        def sglang_path():
            return fused_post_expert_norm_weighted_sum(
                y, w, norm_weight=norm_w, eps=POST_NORM_EPS, weight_first=True, slot_expert_ids=local)

        def fused_path():
            return post_expert_norm_weighted_sum(y, w, norm_w, local)

        ref, out = sglang_path().float(), fused_path().float()
        err = float((out - ref).norm() / ref.norm().clamp(min=1e-12))
        log(f"postnorm bs={bs}: sglang {lab.time_graph(sglang_path):.1f} us, fused {lab.time_graph(fused_path):.1f} us, "
            f"rel diff {err:.2e}")


@triton.jit
def _decode_words(w_ptr, out_ptr, n_words, BLOCK: tl.constexpr):
    """Marlin's E2M1 -> BF16 step: each int32 word (8 codes) -> four bf16x2 words, from the top
    nibble of each 16-bit half down; values are e2m1 * 2^-126."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_words
    q = tl.load(w_ptr + offs, mask=mask, other=0)
    r0, r1, r2, r3 = tl.inline_asm_elementwise(
        asm="""
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
        """,
        constraints="=r,=r,=r,=r,r", args=[q],
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
def _xor_combine(a, b):
    return a ^ b


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
            asm="""
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
            """,
            constraints="=r,=r,=r,=r,r", args=[q],
            dtype=(tl.int32, tl.int32, tl.int32, tl.int32), is_pure=True, pack=1)
        acc ^= r0 ^ r1 ^ r2 ^ r3
    tl.store(out_ptr + pid, tl.reduce(acc, 0, _xor_combine))


def _bf16_halves(words):
    """int32 bf16x2 words -> float32 [..., 2] (low half, high half), exact for subnormals."""
    return words.contiguous().view(torch.int16).view(torch.bfloat16).float().view(*words.shape, 2)


def _all_e4m3_scales(shape, device):
    """Every finite non-negative E4M3 value (0x00-0x7E, subnormals included), cycled over shape.
    random_weights clamps its scales to [8, 448]; this covers the rest."""
    n = 1
    for d in shape:
        n *= d
    return (torch.arange(n, device=device) % 127).to(torch.uint8).view(torch.float8_e4m3fn).view(shape)


def sec_decode(lab):
    gen = torch.Generator(device="cuda").manual_seed(7)
    packed, _, gs = random_weights(256, 4096, generator=gen, device="cuda")  # [256, 2048] uint8
    sf = _all_e4m3_scales((256, 256), "cuda")  # each scale ~516 times, each with all 16 codes
    words = packed.contiguous().view(torch.int32)  # [256, 512], 8 codes per word
    n = words.numel()
    out = torch.empty((n, 4), dtype=torch.int32, device="cuda")
    _decode_words[(triton.cdiv(n, 1024),)](words, out, n, BLOCK=1024)
    halves = _bf16_halves(out)  # [n, 4, 2]: round r, half h -> element 8c + (3 - r) + 4h
    order = torch.tensor([[3, 7], [2, 6], [1, 5], [0, 4]], device="cuda")
    vals = torch.empty((n, 8), dtype=torch.float32, device="cuda")
    vals[:, order.flatten()] = halves.reshape(n, 8)
    want = unpack_e2m1(packed).reshape(n, 8)
    log(f"decode bits: {'EXACT' if torch.equal(vals * 2.0**126, want) else 'MISMATCH'} vs nvfp4.unpack_e2m1 "
        f"(max diff {float((vals * 2.0**126 - want).abs().max()):.3g})")

    # (code * 2^-126) * (scale * 2^64) with mul.rn.bf16x2 must be exact and never flush.
    shift = 64
    scale = (sf.float() * 2.0**shift).to(torch.bfloat16)  # [256, 256], one per 16 values
    col_of_elem = torch.arange(4096, device="cuda") // 16
    a_words = vals.reshape(-1).to(torch.bfloat16).view(torch.int32)
    s_words = scale[:, col_of_elem].reshape(-1).contiguous().view(torch.int32)
    prod = torch.empty_like(a_words)
    m = a_words.numel()
    _mul_bf16x2[(triton.cdiv(m, 1024),)](a_words, s_words, prod, m, BLOCK=1024)
    got = _bf16_halves(prod).reshape(-1) * 2.0 ** (126 - shift)
    want_w = want.reshape(-1) * sf.float()[:, col_of_elem].reshape(-1)
    rel = ((got - want_w).abs() / want_w.abs().clamp(min=1e-30)).max()
    log(f"decode mul.rn.bf16x2 with scale*2^{shift}: {'EXACT' if torch.equal(got, want_w) else 'MISMATCH'} "
        f"(max rel {float(rel):.3g})")
    ref = dequantize(packed, sf, gs).reshape(-1)
    log(f"decode end-to-end vs nvfp4.dequantize: max abs diff {float((got * gs - ref).abs().max()):.3g} "
        f"(max |w| {float(ref.abs().max()):.3g})")

    sub = torch.tensor([0x00000040, 0x00008040], dtype=torch.int32, device="cuda")  # bf16 +/-2^-127
    s = torch.full((2,), 2.0**126, device="cuda")
    o = torch.empty(2, device="cuda")
    _mul_f32_triton[(1,)](sub, s, o, 2, BLOCK=2)
    log(f"decode triton fp32 mul of a subnormal: {o.tolist()} (want [0.5, -0.5]; 0 means flushed)")

    big = torch.randint(0, 2**31 - 1, ((1 << 30) // 4,), dtype=torch.int32, device="cuda")
    nb = big.numel()
    res = []
    for chunk in (1 << 14, 1 << 16):
        for block in (1024, 2048):
            for warps in (4, 8):
                if lab.left() < 8:
                    break
                grid = (triton.cdiv(nb, chunk),)
                o2 = torch.empty(grid[0], dtype=torch.int32, device="cuda")
                fn = lambda c=chunk, b=block, w=warps, o2=o2, g=grid: _decode_stream[g](big, o2, nb, CHUNK=c, BLOCK=b, num_warps=w)
                us = lab.time_graph(fn, reps=10, cold=False)
                res.append(((1 << 30) / us / 1e6, chunk, block, warps))
    res.sort(reverse=True)
    log("decode+load TB/s of packed weights (chunk,block,warps): "
        + " ".join(f"{r[0]:.2f}({r[1]},{r[2]},{r[3]})" for r in res[:6]) + "  [plain load: 4.49]")


def sec_headers(lab):
    # find_spec locates a top-level package without running it, and its search locations also
    # cover namespace packages, whose __file__ is None.
    roots = []
    for mod in ("flashinfer", "deep_gemm", "tilelang", "sgl_kernel", "nvidia_cutlass_dsl", "cutlass_library",
                "nvidia", "torch"):
        try:
            spec = importlib.util.find_spec(mod)
        except Exception as exc:
            log(f"headers: find_spec {mod}: {type(exc).__name__}")
            continue
        if spec is None or not spec.submodule_search_locations:
            log(f"headers: {mod} not installed as a package")
            continue
        roots.extend(spec.submodule_search_locations)
    roots.append("/usr/local/cuda/include")
    hits = {"cutlass": [], "cute": []}
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            if dirpath[len(root):].count(os.sep) >= 7:
                dirnames[:] = []
                continue
            if dirpath.endswith(os.sep + "cutlass") and "cutlass.h" in filenames:
                hits["cutlass"].append(os.path.dirname(dirpath))
            if dirpath.endswith(os.sep + "cute") and "tensor.hpp" in filenames:
                hits["cute"].append(os.path.dirname(dirpath))
    for kind, dirs in hits.items():
        log(f"headers {kind}: {len(dirs)} include dir(s)")
        for d in dirs[:6]:
            log(f"   {d}")


# Launches on PyTorch's current stream, as any kernel inside kernel_fn must for CUDA graph capture.
_NVCC_SRC = r"""
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>

__global__ void bf16x2_mul_kernel(const __nv_bfloat162* a, const __nv_bfloat162* b, __nv_bfloat162* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = __hmul2(a[i], b[i]);
}

torch::Tensor bf16x2_mul(torch::Tensor a, torch::Tensor b) {
  auto c = torch::empty_like(a);
  int n = static_cast<int>(a.numel() / 2);
  bf16x2_mul_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat162*>(a.data_ptr()),
      reinterpret_cast<const __nv_bfloat162*>(b.data_ptr()),
      reinterpret_cast<__nv_bfloat162*>(c.data_ptr()), n);
  return c;
}
"""


def sec_nvcc(lab):
    from torch.utils.cpp_extension import load_inline
    t = time.monotonic()
    # An explicit -gencode makes PyTorch add no arch flags of its own (_get_cuda_arch_flags), so
    # this builds for sm_90a only; its default would be plain sm_90, which lacks wgmma.
    mod = load_inline(
        name="lab_e1b_bf16x2", cpp_sources="torch::Tensor bf16x2_mul(torch::Tensor a, torch::Tensor b);",
        cuda_sources=_NVCC_SRC, functions=["bf16x2_mul"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
        build_directory=tempfile.mkdtemp(prefix="lab_e1b_"), verbose=False)
    log(f"nvcc load_inline (sm_90a, torch extension): compiled in {time.monotonic() - t:.1f}s")
    a = (torch.tensor([0.5, -0.5, 1.0, 6.0]) * 2.0**-126).to(torch.bfloat16).cuda()  # on CPU: exact subnormals
    b = torch.full_like(a, 2.0**64)
    c = mod.bf16x2_mul(a, b).float() * 2.0**62
    log(f"nvcc __hmul2 on bf16 subnormals: {c.tolist()} (want [0.5, -0.5, 1.0, 6.0])")


def prepare(weights):
    state = _submission_prepare(weights)
    lab = Lab(LAB_BUDGET_S)
    log(f"E1b start, budget {LAB_BUDGET_S:.0f}s")
    lab.section("postnorm", sec_postnorm, state)
    lab.section("decode", sec_decode)
    lab.section("headers", sec_headers)
    lab.section("nvcc", sec_nvcc, min_left=90)
    lab.flush = None
    torch.cuda.empty_cache()
    log(f"done in {time.monotonic() - lab.t0:.0f}s")
    return state
