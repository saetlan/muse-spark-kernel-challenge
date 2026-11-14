


# ---------------------------------------------------------------------------------------------
# Lab L31 (check runs only), on v11 (the timed path = v11 byte for byte), B5's two first steps:
# (1) W3 (cuda_w3.py, A66: W2c with P.V into O in place, two copiers, 2 or 4 consumer warpgroups with setmaxnreg) on
#     Fv16, Fv32, Fv1, Sv32: attributes (registers, local bytes, launchable), error vs the FP32 reference, time vs
#     v11's path; a class moves to the fastest correct variant >= 3% faster (the lab's _cuda_ok / _cuda_run accept
#     ("w3", stages, pp, nwg, rc)).
# (2) sliding floor probes (cuda_probe.py, A65): the bare chain of a one-shot sliding kernel (seqlens -> page entries
#     -> every K / V row at once -> read -> DSMEM merge -> output) and its parts, vs v11's sliding paths on the same
#     inputs, graph times with a cold L2 (raw: the judge subtracts ~3.9 us).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3 = {}
_L31_W3_CASES = [("Fv16", (True, True, 16), 0), ("Fv32", (True, True, 32), 0), ("Fv1", (True, True, 1), 0),
                 ("Sv32", (False, True, 32), 2048)]
_L31_ERR = 0.12
_L31_PROBES = [(1, [(16, 128), (16, 256), (8, 256)]), (16, [(8, 256), (4, 256)]), (32, [(4, 256)])]


def _w3_run(data, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    nparts = nsplit * (nwg // 2)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nparts, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nparts,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W3["mod"].attend_w3(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


_cuda_ok_l31 = _cuda_ok
_cuda_run_l31 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W3 routes run on the lab's module)
    if route[0] == "w3":
        return ("mod" in _W3 and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l31(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3":
        return _w3_run(data, *route[1:])
    return _cuda_run_l31(data, route)


def _l31_w3(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        _W3["mod"] = load_inline(name="okc_l31_w3", cpp_sources=CUDA_W3_DECL, cuda_sources=CUDA_W3_SRC,
                                 functions=["attend_w3", "attend_w3_setup", "attend_w3_attrs"],
                                 extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                 build_directory=tempfile.mkdtemp(prefix="okc_l31_w3_"), verbose=False)
        refused = _W3["mod"].attend_w3_setup()
        ok_vars = []
        for i, (s, p, nw, rc, rp) in enumerate(W3_VARIANTS):
            a = _W3["mod"].attend_w3_attrs(s, p, nw, rc)
            ok = not (refused >> i) & 1
            _log(f"w3 s{s} pp{p} nwg{nw} rc{rc}/{rp} attrs (regs, local B, static smem B, pool ok): {a}"
                 f"{'  <-- LOCAL MEMORY' if a[1] else ''}{'' if ok else '  REFUSED'}")
            if ok:
                ok_vars.append((s, p, nw, rc))
    except Exception as exc:
        _W3.pop("mod", None)
        _log(f"w3: build FAILED {_why(exc)}")
        return {}
    chosen = {}
    for name, cls, window in _L31_W3_CASES:
        if lab.left() < 120:
            _log("w3: out of budget")
            break
        bclass = cls[2]
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for v in ok_vars:
            try:
                err = float((_w3_run(data, *v).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w3_run(data, *v))
                res.append(f"{v} {us:.1f}us err {err:.1e}")
                if err < _L31_ERR and (best is None or us < best[0]):
                    best = (us, ("w3",) + v)
            except Exception as exc:
                res.append(f"{v} FAILED {_why(exc)[:120]}")
        _log(f"{name} current {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    return chosen


def _l31_probes(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        mod = load_inline(name="okc_l31_probe", cpp_sources=CUDA_PROBE_DECL, cuda_sources=CUDA_PROBE_SRC,
                          functions=["probe", "probe_empty", "probe_setup"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_l31_probe_"), verbose=False)
        err = mod.probe_setup()
        if err:
            raise RuntimeError(f"probe_setup returned {err}")
    except Exception as exc:
        _log(f"probes: build FAILED {_why(exc)}")
        return
    out = torch.empty((32 * 1024,), dtype=torch.bfloat16, device="cuda")
    try:
        e0 = lab.time_graph(lambda: mod.probe_empty(out, 0))
        e1 = lab.time_graph(lambda: mod.probe_empty(out, 1))
        _log(f"probe: empty kernel {e0:.2f}us, empty cluster-16 kernel {e1:.2f}us (raw graph times)")
    except Exception as exc:
        _log(f"probe empty FAILED {_why(exc)}")
    for bclass, cfgs in _L31_PROBES:
        if lab.left() < 25:
            _log("probes: out of budget")
            break
        try:
            res = []
            for length, tag in ((1, "Sd"), (8, "Sv")):
                data = _lab_inputs(bclass, length, 2048, k_cache, v_cache, gen)
                _kernel_fn_submitted(data)
                res.append(f"{tag}{bclass} v11 {lab.time_graph(lambda: _kernel_fn_submitted(data)):.2f}")
            seq, pt = data["cache_seqlens"], data["page_table"]
            kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
            for cs, nt in cfgs:
                parts = []
                for mode in (0, 2, 1, 3, 4):
                    us = lab.time_graph(lambda: mod.probe(seq, pt, kc, vc, out, bclass, pt.shape[1], 2048, cs, nt, mode))
                    parts.append(f"m{mode} {us:.2f}")
                res.append(f"cs{cs}x{nt}: " + " ".join(parts))
            _log(f"probe bs{bclass} (us raw; m0 chain, m2 no page entries, m1 no K/V, m3 neither, m4 no merge): " + "; ".join(res))
        except Exception as exc:
            _log(f"probe bs{bclass}: FAILED {_why(exc)}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3131)
    _log(f"L31: timed path v11: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = _l31_w3(lab, k_cache, v_cache, gen)
    _log(f"ROUTES = {chosen!r}")
    _l31_probes(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
