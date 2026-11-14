"""HLP (P42): the router's hi/lo split-K Triton kernels (v10's HL, PE18-PE20) without paying for a second kernel in the
graph. v10 runs torch.zeros (a fill kernel, ~0.7 us) and then the split-K kernel with FP32 atomics; on the judge a
2-kernel shape costs ~1.2 us more launch overhead than a 1-kernel one (PE29-PE31 profiles), so the fill costs ~1.9 us
per router call at m = 16/32/128/256. Programmatic dependent launch (Triton 3.8: launch_pdl=True; extra.cuda.gdc_*):
  zero kernel  calls gdc_launch_dependents() first, then zeroes out (so the GEMM may launch at once);
  GEMM kernel  launched with launch_pdl=True: its main loop overlaps the fill; gdc_wait() right before its atomic adds
               (griddepcontrol.wait: every write of the prior grid is visible after it), so the adds never race the fill.
Same tiles, same atomics, same numerics as v10's HL. Variant "hlpt": v10's torch.zeros + the PDL GEMM (implicit
trigger at the fill's exit), in case gdc_launch_dependents misbehaves."""

import triton.language.extra.cuda as tlc   # a module: Triton 3.8 global_lookup accepts it (F24)

HLP_BLOCK = 1024


@triton.jit
def _hlp_zero_kernel(out_ptr, numel, BLOCK: tl.constexpr):
    tlc.gdc_launch_dependents()
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, tl.zeros([BLOCK], dtype=tl.float32), mask=offs < numel)


@triton.jit(do_not_specialize=["M"])
def _hlp_gemm_kernel(x_ptr, hi_ptr, lo_ptr, out_ptr, M, N, k_per_split, stride_xm, stride_wn, stride_om,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    tlc.gdc_wait()
    tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")


@triton.jit(do_not_specialize=["M"])
def _hlp_swap_kernel(hi_ptr, lo_ptr, x_ptr, out_ptr, M, N, k_per_split, stride_wn, stride_xm, stride_om,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    tlc.gdc_wait()
    tl.atomic_add(out_ptrs, acc, mask=mask, sem="relaxed")


def run_hlp(c, x, layer, n, state):
    """c: dict(kind="hlp" | "hlpt", base=v10-style HL config with split > 1). Without the hi/lo copies (or for another
    layer, or a swap config with too many tokens) it runs c["fallback"] or SGLang's path."""
    hi, lo = state.get("router_hi"), state.get("router_lo")
    b = c["base"]
    m, k = x.shape
    if layer != "router" or hi is None or b["split"] < 2 or (b["kind"] == "hl_swap" and m > b["BM"]):
        fb = c.get("fallback")
        return run_choice(fb, x, layer, n, state) if fb else sglang_path(x, layer, n, state)
    kps = k // b["split"]
    x = x.contiguous()
    if c["kind"] == "hlp":
        out = torch.empty((m, n), dtype=torch.float32, device=x.device)
        _hlp_zero_kernel[(triton.cdiv(m * n, HLP_BLOCK),)](out, m * n, BLOCK=HLP_BLOCK, num_warps=4)
    else:
        out = torch.zeros((m, n), dtype=torch.float32, device=x.device)
    if b["kind"] == "hl_swap":
        _hlp_swap_kernel[(triton.cdiv(n, b["BN"]), b["split"])](
            hi, lo, x, out, m, n, kps, hi.stride(0), x.stride(0), out.stride(0),
            BM=b["BM"], BN=b["BN"], BK=b["BK"], num_warps=b["warps"], num_stages=b["stages"], launch_pdl=True)
    else:
        _hlp_gemm_kernel[(triton.cdiv(m, b["BM"]), triton.cdiv(n, b["BN"]), b["split"])](
            x, hi, lo, out, m, n, kps, x.stride(0), hi.stride(0), out.stride(0),
            BM=b["BM"], BN=b["BN"], BK=b["BK"], num_warps=b["warps"], num_stages=b["stages"], launch_pdl=True)
    return out


_run_choice_pre_hlp = run_choice


def run_choice(c, x, layer, n, state):
    if c["kind"] in ("hlp", "hlpt"):
        return run_hlp(c, x, layer, n, state)
    return _run_choice_pre_hlp(c, x, layer, n, state)
