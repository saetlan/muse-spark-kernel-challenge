

# ---------------------------------------------------------------------------------------------
# Lab section "xu" (check runs only): instruction throughput of W1's softmax ops (cuda_xu.py), as ops / clk /
# SM relative to FFMA = 128. One launch per op (4 CTAs of 256 per SM), best of 3, CUDA events.
# ---------------------------------------------------------------------------------------------
_XU_ITERS = 2048


class _XUHost:   # host-side timing (lab only: eager, never under capture)
    @staticmethod
    def bench(lab):
        from torch.utils.cpp_extension import load_inline
        import tempfile
        mod = load_inline(name="okc_xu", cpp_sources=CUDA_XU_DECL, cuda_sources=CUDA_XU_SRC, functions=["xu_run"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_xu_"), verbose=False)
        blocks = 4 * NUM_SMS
        out = torch.empty(blocks * 256, dtype=torch.float32, device="cuda")
        times = []
        for op in range(len(XU_OPS)):
            if lab.left() < 15:
                _log("xu: out of budget")
                break
            mod.xu_run(out, op, 16, blocks)
            torch.cuda.synchronize()
            best = float("inf")
            for _ in range(3):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record()
                mod.xu_run(out, op, _XU_ITERS, blocks)
                e1.record()
                torch.cuda.synchronize()
                best = min(best, e0.elapsed_time(e1) * 1e3)
            times.append(best)
        if not times:
            return
        per = blocks * 256 * 8 * _XU_ITERS                  # inner steps
        ffma = per / times[0]                                # FFMA ops per us = 128 / clk / SM
        parts = []
        for (name, n_ops, vals), us in zip(XU_OPS, times):
            rate = 128.0 * (per * n_ops / us) / ffma
            parts.append(f"{name} {rate:.1f}" + (f" ({rate * vals:.0f} values)" if vals > 1 else ""))
        _log(f"xu ops/clk/SM (FFMA = 128; {times[0]:.0f}us per FFMA launch): " + "; ".join(parts))
