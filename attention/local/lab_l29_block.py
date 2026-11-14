

# ---------------------------------------------------------------------------------------------
# Lab L29 (check runs only), on v10 (the timed path = v10 byte for byte: v9 + W2c on Fv1 / Sv1):
# where does the full-verify stage time go? (L28: overlapping W2c's wgmmas with its softmax gained nothing on
# Fv16 / Fv32, so the consumer chain is not the limit.) On Fv16, Fv32, Sv32 lab inputs, cold L2, CUDA graphs:
# v10's path (W1-N), W2c (4 stages, ping-pong) and its ablations (cuda_w2.py attend_w2abl): 1 consumers skip all
# compute (wait full, release: copier + transposers + memory only), 2 the copier skips the K/V copies (no K/V
# memory traffic), 3 the transposers skip the transpose. Whichever ablation leaves the time unchanged is not on
# the critical path.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L29_CASES = [("Fv16", 16, 0), ("Fv32", 32, 0), ("Sv32", 32, 2048)]


def _l29_run(mod, data, abl):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"], cfg["sm_scale"])
    if abl == 0:
        mod.attend_w2(*args, 4, 1)
    else:
        mod.attend_w2abl(*args, abl)
    return _merge(data, o_part, m_part, l_part, nsplit)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2929)
    _log(f"L29: timed path v10: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        mod = load_inline(name="okc_l29_w2", cpp_sources=CUDA_W2_DECL, cuda_sources=CUDA_W2_SRC,
                          functions=["attend_w2", "attend_w2_setup", "attend_w2abl", "attend_w2abl_setup"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_l29_"), verbose=False)
        if mod.attend_w2_setup() or mod.attend_w2abl_setup():
            raise RuntimeError("setup refused")
    except Exception as exc:
        _log(f"w2: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    for name, bclass, window in _L29_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq = data["cache_seqlens"] if window == 0 else data["cache_seqlens"].clamp(max=2048 + 8)
            kv = float(seq.sum()) * 256
            res = [f"v10 path {base:.1f}us"]
            for abl, label in ((0, "w2c"), (1, "abl1 no-consumer"), (2, "abl2 no-K/V-copy"), (3, "abl3 no-transpose")):
                try:
                    if abl == 0:
                        err = float((_l29_run(mod, data, 0).float() - ref).norm() / ref.norm())
                    us = lab.time_graph(lambda: _l29_run(mod, data, abl))
                    res.append(f"{label} {us:.1f}us ({kv / us / 1e6:.2f}TB/s)" + (f" err {err:.1e}" if abl == 0 else ""))
                except Exception as exc:
                    res.append(f"{label} FAILED {_why(exc)}")
            _log(f"{name}: " + "; ".join(res))
        except Exception as exc:
            _log(f"{name}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
