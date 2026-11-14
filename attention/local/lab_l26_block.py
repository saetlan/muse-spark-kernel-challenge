

# ---------------------------------------------------------------------------------------------
# Lab L26 (check runs only), on v9 (the timed path = v9 byte for byte):
#   1. wgmma unit test CHAIN 0-4 (cuda_w1unit.py; CHAIN 4 = W2's SS form: A from a shared-memory tile through its
#      descriptor, W2's async issue + bound wait);
#   2. W2 (cuda_w2.py: W1's copier / transposers, Q and P in shared memory, SS wgmma pairs, Q.K(i+1) overlapped with
#      softmax(i), P.V(i) overlapped with the next stage's first half) in (stages, db) = (4, 0), (3, 0), (6, 0), (4, 1)
#      on all six verify classes: error vs the FP32 reference and time vs v9's path; a class moves to the fastest
#      correct variant >= 3% faster (the lab's _cuda_ok / _cuda_run accept ("w2", ctas per SM, stages, db)).
#   Prints W2's registers / local memory and ROUTES = {...}.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W2 = {}
_L26_VARIANTS = [("w2", 1, 4, 0), ("w2", 1, 3, 0), ("w2", 1, 6, 0), ("w2", 1, 4, 1)]
_L26_CLASSES = [(True, True, 16), (True, True, 32), (False, True, 16), (False, True, 32), (True, True, 1), (False, True, 1)]
_L26_ERR = 0.12


def _l26_build(name, src, decl, funcs, verbose=False):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    flags = ["-O3", "-gencode=arch=compute_90a,code=sm_90a"] + (["-Xptxas", "-v"] if verbose else [])
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs, extra_cuda_cflags=flags,
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=verbose)


def _w2_run(data, route):
    _, cps, stages, db = route
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W2["w2"].attend_w2(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"],
                        o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"],
                        cfg["sm_scale"], stages, db)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l26 = _cuda_ok
_cuda_run_l26 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W1 routes run on the lab's modules)
    if route[0] == "w2":
        return ("w2" in _W2 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l26(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w2":
        return _w2_run(data, route)
    return _cuda_run_l26(data, route)


class _L26Host:   # host-side reads (lab only: eager calls, never under capture)
    @staticmethod
    def unit(mod, gen):
        dev = torch.device("cuda")
        for ch in (0, 1, 2, 3, 4):
            kd = 64 if ch else 32
            A = (torch.randn((64, kd), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            B = (torch.randn((64, kd), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            Vv = (torch.randn((64, 64), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            D0 = torch.randn((64, 64), generator=gen, device=dev)
            S = A.float() @ B.float().T
            ref = ((S / 16).to(torch.float8_e4m3fn).float() @ Vv.float()) if ch == 3 else S + (D0 if ch == 2 else 0)
            D = D0.clone() if ch == 2 else torch.full((64, 64), float("nan"), device=dev)
            try:
                mod.w1u(A.view(torch.uint8), B.view(torch.uint8), Vv.view(torch.uint8), D, 512, 0, 0, 0, ch, 1.0 / 16)
                torch.cuda.synchronize()
                err = float((torch.nan_to_num(D, nan=1e9) - ref).norm() / ref.norm())
                _log(f"unit CHAIN {ch}: rel err {err:.2e}")
            except Exception as exc:
                _log(f"unit CHAIN {ch}: FAILED {_why(exc)}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2626)
    _log(f"L26: timed path until a class wins: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    try:
        unit = _l26_build("okc_l26_unit", CUDA_W1U_SRC, CUDA_W1U_DECL, ["w1u"])
        _L26Host.unit(unit, gen)
    except Exception as exc:
        _log(f"unit: build/run FAILED {_why(exc)}")
    try:
        _W2["w2"] = _l26_build("okc_l26_w2", CUDA_W2_SRC, CUDA_W2_DECL, ["attend_w2", "attend_w2_setup", "attend_w2_attrs"],
                               verbose=True)
        err = _W2["w2"].attend_w2_setup()
        if err:
            raise RuntimeError(f"attend_w2_setup returned {err}")
        for s, d in ((3, 0), (4, 0), (6, 0), (4, 1)):
            a = _W2["w2"].attend_w2_attrs(s, d)
            _log(f"w2 s{s} db{d} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    except Exception as exc:
        _W2.pop("w2", None)
        _log(f"w2: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for cls in _L26_CLASSES:
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
        for route in _L26_VARIANTS:
            if route[0] not in _W2:
                continue
            try:
                err = float((_w2_run(data, route).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w2_run(data, route))
                _log(f"{name} {route}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                if err < _L26_ERR and (best is None or us < best[0]):
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
