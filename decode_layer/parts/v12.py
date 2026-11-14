# v12 = v11 + SGLang's own multi-CTA cluster kernel for the layer's two boundaries (post-norm + gated residual +
# pre-norm, FusedPostNormResidualPreNorm). SGLang's fused_norm_residual (sglang/kernels/ops/muse_spark_v12/
# norm_residual.py at the pinned commit a55ab7d5) dispatches its cluster kernel (one row split over C CTAs, partial
# RMS moments exchanged through distributed shared memory: mapa + st.async + mbarrier complete_tx, all sm_90 PTX) only
# when the device's major capability is >= 10, so on the H200 every boundary runs the one-CTA kernel (~4 us at
# 1-32 tokens, latency-bound: DE12 profiles). Its cluster_size argument selects the cluster kernel explicitly; the
# math is SGLang's (the same moments; the pre-norm sum is formed from them instead of a second pass). A lab picks C
# per token class (V12_BOUNDARY); missing classes keep SGLang's dispatch (v11's path).

# Token-count bound (fewer than this many tokens) -> cluster size. Filled from a lab's printed V12_BOUNDARY.
V12_BOUNDARY = {}
V12_SIZES = (2, 4, 8, 16)
V12_KEYS = ("attn_boundary", "ffn_boundary")


def v12_cluster_for(picks, tokens):
    for below, size in picks:
        if tokens < below:
            return size
    return 1


def v12_cluster_call(mod, stream, branch, router_dtype, size):
    """FusedPostNormResidualPreNorm.forward_jit's call, with SGLang's cluster kernel of `size` CTAs per row."""
    from sglang.kernels.ops.muse_spark_v12.norm_residual import fused_norm_residual as fn
    mod._ensure_cached_weights(torch.bfloat16)
    return fn(branch, stream, mod._post_norm.eps, mod._pre_norm.eps,
              post_norm_is_input_scale=mod._post_norm_is_input_scale,
              post_weight=mod._cc_post_weight if mod._post_norm.has_weight else None,
              pre_weight=mod._cc_pre_weight if mod._pre_norm.has_weight else None,
              alpha=mod._cc_alpha if mod._use_residual_gate else None,
              beta=mod._cc_beta if mod._use_residual_gate else None,
              want_router=router_dtype is not None, pre_norm_bf16=False, cluster_size=size)


class V12Boundary:
    """state[key] stand-in: SGLang's module for classes without a pick, its cluster kernel for the others."""

    def __init__(self, mod, picks):
        self.mod, self.picks = mod, tuple(sorted(picks.items()))

    def __call__(self, stream, branch, branch_dtype=torch.bfloat16, router_dtype=None):
        size = v12_cluster_for(self.picks, branch.shape[0])
        if size == 1 or branch_dtype is not torch.bfloat16 or router_dtype not in (None, torch.float32):
            return self.mod(stream, branch, branch_dtype=branch_dtype, router_dtype=router_dtype)
        return v12_cluster_call(self.mod, stream, branch, router_dtype, size)


def v12_build(sizes, may_start=None, tokens=1):
    """Compile SGLang's cluster modules for `sizes`, in order, one at a time (no background threads: the judge
    rejects submissions that start processes, and no judged file has run threads). may_start(seconds of the last
    build) -> bool stops before a build that might not fit the caller's budget. Returns {size: seconds or error}."""
    import time as _v12_time
    from sglang.kernels.ops.muse_spark_v12 import norm_residual as nr
    out, last = {}, 0.0
    for size in sizes:
        if may_start is not None and not may_start(last):
            out[size] = "skipped (time)"
            continue
        t0 = _v12_time.monotonic()
        try:
            threads, prefetch = nr._cluster_launch_config(HIDDEN, tokens, size)
            nr._jit_module(HIDDEN, threads, prefetch, size)
            out[size] = last = _v12_time.monotonic() - t0
        except Exception as exc:
            out[size] = f"{type(exc).__name__}: {str(exc)[-160:]}"
            last = _v12_time.monotonic() - t0
    return out


def v12_inputs(tokens, device, gen):
    branch = (torch.randn((tokens, HIDDEN), generator=gen, device=device) * 0.7).to(torch.bfloat16)
    stream = torch.randn((tokens, HIDDEN), generator=gen, device=device) * 3.0
    return branch, stream


def v12_compare(mod, tokens, size, device, gen):
    """Worst relative error of the cluster kernel vs SGLang's dispatch, over output / new stream / router."""
    branch, stream = v12_inputs(tokens, device, gen)
    router = torch.float32 if tokens <= ROUTER_FP32_MAX_TOKENS else None
    ref = mod(stream, branch, branch_dtype=torch.bfloat16, router_dtype=router)
    got = v12_cluster_call(mod, stream, branch, router, size)
    return max(_v3_rel(g, r) for g, r in zip(got, ref) if r is not None)


def v12_install(state, picks):
    base = state.setdefault("v12_base", {key: state[key] for key in V12_KEYS})
    for key in V12_KEYS:
        state[key] = V12Boundary(base[key], picks) if picks else base[key]
    state["v12_boundary"] = dict(picks)


def v12_setup(state, weights):
    """Build the picked cluster modules, check each pick against SGLang's dispatch at its class (< 1e-2) on both
    boundaries (this is also each module's first launch, outside any capture), install. Failures keep v11."""
    state["v12_boundary"] = {}
    picks = {below: size for below, size in V12_BOUNDARY.items() if size in V12_SIZES}
    if not picks:
        print("[v12] boundaries: SGLang's dispatch (no cluster picks)")
        return
    device = weights["qkv_gate"].device
    gen = torch.Generator(device=device).manual_seed(53)
    try:
        built = v12_build(sorted(set(picks.values())))
        bad = {s: v for s, v in built.items() if isinstance(v, str)}
        if bad:
            raise RuntimeError(f"cluster build failed: {bad}")
        worst = 0.0
        for below, size in sorted(picks.items()):
            for key in V12_KEYS:
                for tokens in sorted({1, below - 1}):
                    worst = max(worst, v12_compare(state[key], tokens, size, device, gen))
        torch.cuda.synchronize()
        if not worst < 1e-2:
            raise RuntimeError(f"cluster boundary differs from SGLang's: rel err {worst:.1e}")
        v12_install(state, picks)
        print(f"[v12] boundaries: cluster {picks} (build s {', '.join(f'{s}: {v:.1f}' for s, v in built.items())}; "
              f"worst rel err vs SGLang {worst:.1e})")
    except Exception as exc:
        v12_install(state, {})
        print(f"[v12] boundaries: SGLang's dispatch ({type(exc).__name__}: {str(exc)[-200:]})")
