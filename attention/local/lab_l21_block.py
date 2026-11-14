

# ---------------------------------------------------------------------------------------------
# Lab L21 (check runs only), on v8 (submission_v8.py), to find why W1 is fast but wrong (AF42):
#   1. wgmma unit test (cuda_w1unit.py: one warpgroup, D = A . B^T with W1's helpers): 7 variants
#      (B group stride 512 / 528, descriptor fields swapped, B by plain stores / cp.async + consumer
#      fence / cp.async + writer wait + fence, the other A register order) vs torch; for the base
#      variant also which reference row / column each output row / column matches best;
#   2. W1 (cuda_w1.py, templated) in four variants: Vt stride 528 / 512 x copier noinc arrive / writer
#      wait + fence (WF), on all six verify classes, error vs the FP32 reference and time vs the
#      current path; a class moves to a correct variant >= 3% faster (the lab's _cuda_ok / _cuda_run
#      accept ("w1", ctas per SM, stages, Vt stride, WF)).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W1 = {}
_L21_UNIT = [(512, 0, 0, 0), (528, 0, 0, 0), (512, 1, 0, 0), (512, 0, 1, 0), (512, 0, 2, 0), (528, 0, 1, 0), (512, 0, 0, 1)]
_L21_W1 = [(1, 4, 528, 0), (1, 4, 512, 0), (1, 4, 528, 1), (1, 4, 512, 1)]
_L21_CLASSES = [(True, True, 16), (True, True, 32), (False, True, 16), (False, True, 32), (True, True, 1), (False, True, 1)]
_L21_ERR = 0.12


def _l21_build(name, src, decl, funcs):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=False)


class _L21Host:   # host-side reads (lab only: first eager call, never under capture)
    @staticmethod
    def unit(mod, gen):
        dev = torch.device("cuda")
        A = (torch.randn((64, 32), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
        B = (torch.randn((64, 32), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
        ref = A.float() @ B.float().T
        for sbo, swap, cp, am in _L21_UNIT:
            D = torch.full((64, 64), float("nan"), device=dev)
            try:
                mod.w1u(A.view(torch.uint8), B.view(torch.uint8), D, sbo, swap, cp, am)
                torch.cuda.synchronize()
                err = float((D - ref).norm() / ref.norm())
                _log(f"unit SBO {sbo} SWAP {swap} CPMODE {cp} AMODE {am}: rel err {err:.2e}")
                if (sbo, swap, cp, am) == (512, 0, 0, 0) and not err < 1e-3:
                    Dn = torch.nan_to_num(D)
                    rows = [int((torch.nn.functional.cosine_similarity(Dn[r:r + 1], ref, dim=1)).argmax()) for r in range(64)]
                    cols = [int((torch.nn.functional.cosine_similarity(Dn[:, c:c + 1], ref, dim=0)).argmax()) for c in range(64)]
                    rmax = [round(float(torch.nn.functional.cosine_similarity(Dn[r:r + 1], ref, dim=1).max()), 2) for r in range(8)]
                    _log(f"unit base: out row -> ref row {rows[:32]}; out col -> ref col {cols[:32]}; best cos rows 0-7 {rmax}")
                    _log(f"unit base: D[0,:8] {[round(float(x), 2) for x in Dn[0, :8]]} ref[0,:8] {[round(float(x), 2) for x in ref[0, :8]]}")
            except Exception as exc:
                _log(f"unit SBO {sbo} SWAP {swap} CPMODE {cp} AMODE {am}: FAILED {_why(exc)}")


def _w1_run(data, cps, stages, sbov, wf):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W1["mod"].attend_w1(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, sbov, wf)
    return _merge(data, o_part, m_part, l_part, nsplit)


_cuda_ok_l21 = _cuda_ok
_cuda_run_l21 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W1 routes run on the lab's module)
    if route[0] == "w1":
        return ("mod" in _W1 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l21(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w1":
        return _w1_run(data, *route[1:])
    return _cuda_run_l21(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2121)
    _log(f"L21: timed path until a class wins: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    try:
        unit = _l21_build("okc_l21_unit", CUDA_W1U_SRC, CUDA_W1U_DECL, ["w1u"])
        _L21Host.unit(unit, gen)
    except Exception as exc:
        _log(f"unit: build/run FAILED {_why(exc)}")
    try:
        _W1["mod"] = _l21_build("okc_l21_w1", CUDA_W1_SRC, CUDA_W1_DECL, ["attend_w1", "attend_w1_setup", "attend_w1_attrs"])
        err = _W1["mod"].attend_w1_setup()
        if err:
            raise RuntimeError(f"attend_w1_setup returned {err}")
    except Exception as exc:
        _W1.pop("mod", None)
        _log(f"W1: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for cls in _L21_CLASSES if "mod" in _W1 else ():
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
        for cps, s, sbov, wf in _L21_W1:
            route = ("w1", cps, s, sbov, wf)
            try:
                err = float((_w1_run(data, cps, s, sbov, wf).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w1_run(data, cps, s, sbov, wf))
                _log(f"{name} w1 c{cps} s{s} sbov{sbov} wf{wf}: {us:.1f}us {kv / us / 1e6:.2f}TB/s err {err:.1e}")
                if err < _L21_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                _log(f"{name} w1 c{cps} s{s} sbov{sbov} wf{wf}: FAILED {_why(exc)}")
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
