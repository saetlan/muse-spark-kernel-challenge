

# ---------------------------------------------------------------------------------------------
# Lab L24 (check runs only), on v8 (submission_v8.py): W1 fixed (AF46: each wgmma pair with its fence, commit
# and wait in one asm statement, cuda_w1.py) in (ctas per SM, stages, Vt stride, WF) = (1, 3/4/6, 528, 0), plus two
# fallbacks with the L22/L23 debug kernel's layout: W1-N (cuda_w1n.py, dump guarded off by a null pointer) and
# W1-D (cuda_w1dbg_l22.py, the exact kernel L23 found right, with a scratch dump buffer), on all six verify classes:
# error vs the FP32 reference and time vs the current path; a class moves to the fastest correct variant >= 3%
# faster (the lab's _cuda_ok / _cuda_run accept "w1", "w1n", "w1d"). Prints ptxas -v for W1 and ROUTES = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W1 = {}
_L24_VARIANTS = [("w1", 1, 4, 528, 0), ("w1", 1, 3, 528, 0), ("w1", 1, 6, 528, 0), ("w1n", 1, 4), ("w1d", 1, 4)]
_L24_CLASSES = [(True, True, 16), (True, True, 32), (False, True, 16), (False, True, 32), (True, True, 1), (False, True, 1)]
_L24_ERR = 0.12


def _l24_build(name, src, decl, funcs, verbose=False):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + (["-Xptxas", "-v"] if verbose else [])
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs, extra_cuda_cflags=flags,
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=verbose)


def _w1_run(data, route):
    kind, cps, stages = route[0], route[1], route[2]
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
            o_part, m_part, l_part)
    tail = (batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"], cfg["sm_scale"])
    if kind == "w1":
        _W1["w1"].attend_w1(*args, *tail, stages, route[3], route[4])
    elif kind == "w1n":
        _W1["w1n"].attend_w1n(*args, *tail, stages)
    else:
        _W1["w1d"].attend_w1dbg(*args, _W1["scratch"], *tail)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l24 = _cuda_ok
_cuda_run_l24 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W1 routes run on the lab's modules)
    if route[0] in ("w1", "w1n", "w1d"):
        return (route[0] in _W1 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l24(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] in ("w1", "w1n", "w1d"):
        return _w1_run(data, route)
    return _cuda_run_l24(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2424)
    _log(f"L24: timed path until a class wins: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    try:
        _W1["w1"] = _l24_build("okc_l24_w1", CUDA_W1_SRC, CUDA_W1_DECL, ["attend_w1", "attend_w1_setup", "attend_w1_attrs"],
                               verbose=True)
        err = _W1["w1"].attend_w1_setup()
        if err:
            raise RuntimeError(f"attend_w1_setup returned {err}")
        for s in (3, 4, 6):
            a = _W1["w1"].attend_w1_attrs(s, 528, 0)
            _log(f"w1 s{s} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _W1.pop("w1", None)
        _log(f"w1: build FAILED {_why(exc)}")
    try:
        _W1["w1n"] = _l24_build("okc_l24_w1n", CUDA_W1N_SRC, CUDA_W1N_DECL, ["attend_w1n", "attend_w1n_setup"])
        if _W1["w1n"].attend_w1n_setup():
            raise RuntimeError("attend_w1n_setup refused")
    except Exception as exc:
        _W1.pop("w1n", None)
        _log(f"w1n: build FAILED {_why(exc)}")
    try:
        _W1["w1d"] = _l24_build("okc_l24_w1d", CUDA_W1DBG_L22_SRC, CUDA_W1DBG_L22_DECL, ["attend_w1dbg"])
        _W1["scratch"] = torch.zeros((W1DBG_L22_FLOATS,), dtype=torch.float32, device=dev)
    except Exception as exc:
        _W1.pop("w1d", None)
        _log(f"w1d: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for cls in _L24_CLASSES:
        full, _, bclass = cls
        name = f"{'F' if full else 'S'}v{bclass}"
        if lab.left() < 25:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, 0 if full else 2048, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq = data["cache_seqlens"] if full else data["cache_seqlens"].clamp(max=2048 + 8)
            kv = float(seq.sum()) * 256
            cur = CUDA_ROUTE.get(cls) or (("fa3t",) + FA3T_ROUTE[cls] if cls in FA3T_ROUTE else "v5")
            _log(f"{name} current {cur}: {base:.1f}us {kv / base / 1e6:.2f}TB/s")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for route in _L24_VARIANTS:
            if route[0] not in _W1:
                continue
            try:
                err = float((_w1_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w1_run(data, route))
                _log(f"{name} {route}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                if err < _L24_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} {route}: FAILED {_why(exc)}")
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            FA3T_ROUTE.pop(cls, None)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
