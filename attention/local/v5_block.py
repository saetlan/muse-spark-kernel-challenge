

# =============================================================================================
# v5: v4 + programmatic dependent launch on the Triton chain (pdl_block.py, make_pdl.py) for the
# classes where L13 (AE18) measured it >= 3% faster than their v4 path, with the late trigger
# (producers trigger after their last store: TRACKER.md checklist 8). Classes: Sd16 20.7 -> 19.7
# us, Sd32 28.3 -> 27.0, Fd1 32.1 -> 31.1, Sv1 19.6 -> 18.5, Sv16 29.1 -> 28.2, Sv32 (SGLang 42.8
# in L11) -> 40.7. The PDL kernels compile at warm-up and are checked against the Triton path on
# synthetic inputs; any failure drops PDL for every class (v4 paths stay).
# =============================================================================================
PDL_ROUTE = {(False, False, 16), (False, False, 32), (True, False, 1),
             (False, True, 1), (False, True, 16), (False, True, 32)}
_kernel_fn_v4 = kernel_fn
_warm_up_v4 = _warm_up


def _warm_up():  # noqa: F811 (v5: v4's warm-up, then compile and check the PDL chain per class)
    _warm_up_v4()
    dev = torch.device(_WARM_DEVICE)
    try:
        for full, verify, bclass in sorted(PDL_ROUTE):
            data = _self_check_inputs(bclass, 8 if verify else 1, full, dev)
            ours = run_pdl(data, early=False).float()
            want = run(data).float()
            err = float((ours - want).norm() / want.norm().clamp_min(1e-6))
            if not err < SELF_CHECK_TOL:
                raise RuntimeError(f"class {(full, verify, bclass)} differs from the Triton path: {err:.2e}")
    except Exception as exc:
        print(f"v5: PDL chain dropped: {type(exc).__name__}: {str(exc)[-200:]}")
        PDL_ROUTE.clear()


def kernel_fn(data):  # noqa: F811
    if not _WARM and not torch.cuda.is_current_stream_capturing():
        _warm_up()
    cfg = data["config"]
    cls = shape_class(cfg)
    if cls not in CUDA_ROUTE and cls in PDL_ROUTE:
        return run_pdl(data, early=False)
    return _kernel_fn_v4(data)
