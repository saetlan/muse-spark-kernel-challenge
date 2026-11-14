


# ---------------------------------------------------------------------------------------------
# Lab L33 (check runs only), on v13 (the timed path = v13 byte for byte: v12 + W3b on Fv16 / Fv32):
# (1) W3c (cuda_w3c.py, AF58): W3b at 2 CTAs per SM (NWG 2, setmaxnreg, 3 stages each) on Fv16, Fv32, Fv1, Sv32 vs
#     v13's path; routed when correct (< 0.12 vs FP32) and >= 3% faster (("w3c", stages, pp, nwg, rc)).
# (2) UD3 anatomy (lab_anat_ud3_block.py) on Sd1 (cs16 w4 s3) and Sd16 (cs8 w4 s3), v13's routes.
# (3) Floor probes with whole-row copies (cuda_probe.py mode 8): the chain with K/V at whole-row bandwidth, with and
#     without page entries, next to v13's Sd / Sv times (raw graph µs).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3C = {}
_L33_ERR = 0.12
_L33_CASES = [("Fv16", (True, True, 16), 0), ("Fv32", (True, True, 32), 0), ("Fv1", (True, True, 1), 0),
              ("Sv32", (False, True, 32), 2048)]
_L33_PROBES = [(1, (16, 128)), (16, (8, 256)), (32, (4, 256))]


def _w3c_run(data, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 2, 64)
    nparts = nsplit * (nwg // 2)
    rows = batch * length * hq * nparts
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((rows,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W3C["mod"].attend_w3c(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                           cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    return _merge(data, o_part, m_part, l_part, nparts)


_cuda_ok_l33 = _cuda_ok
_cuda_run_l33 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W3c routes run on the lab's module)
    if route[0] == "w3c":
        return ("mod" in _W3C and cfg["q_len"] == 8 and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"]
                and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l33(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3c":
        return _w3c_run(data, *route[1:])
    return _cuda_run_l33(data, route)


def _l33_w3c(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        _W3C["mod"] = load_inline(name="okc_l33_w3c", cpp_sources=CUDA_W3C_DECL, cuda_sources=CUDA_W3C_SRC,
                                  functions=["attend_w3c", "attend_w3c_setup", "attend_w3c_attrs"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l33_w3c_"), verbose=False)
        refused = _W3C["mod"].attend_w3c_setup()
        ok_vars = []
        for i, (s, p, nw, rc, rp) in enumerate(W3C_VARIANTS):
            a = _W3C["mod"].attend_w3c_attrs(s, p, nw, rc)
            ok = not (refused >> i) & 1
            _log(f"w3c s{s} pp{p} nwg{nw} rc{rc}/{rp} attrs (regs, local B, static smem B, pool ok): {a}"
                 f"{'  <-- LOCAL MEMORY' if a[1] else ''}{'' if ok else '  REFUSED'}")
            if ok:
                ok_vars.append((s, p, nw, rc))
    except Exception as exc:
        _W3C.pop("mod", None)
        _log(f"w3c: build FAILED {_why(exc)}")
        return {}
    chosen = {}
    for name, cls, window in _L33_CASES:
        if lab.left() < 150:
            _log("w3c: out of budget")
            break
        try:
            data = _lab_inputs(cls[2], 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for v in ok_vars:
            try:
                err = float((_w3c_run(data, *v).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _w3c_run(data, *v))
                res.append(f"{v} {us:.1f}us err {err:.1e}")
                if err < _L33_ERR and (best is None or us < best[0]):
                    best = (us, ("w3c",) + v)
            except Exception as exc:
                res.append(f"{v} FAILED {_why(exc)[:120]}")
        _log(f"{name} v13 {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    return chosen


def _l33_probes(lab, k_cache, v_cache, gen):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    try:
        mod = load_inline(name="okc_l33_probe", cpp_sources=CUDA_PROBE_DECL, cuda_sources=CUDA_PROBE_SRC,
                          functions=["probe", "probe_empty", "probe_setup"],
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                          build_directory=tempfile.mkdtemp(prefix="okc_l33_probe_"), verbose=False)
        if mod.probe_setup():
            raise RuntimeError("probe_setup refused")
    except Exception as exc:
        _log(f"probes: build FAILED {_why(exc)}")
        return
    out = torch.empty((32 * 1024,), dtype=torch.bfloat16, device="cuda")
    for bclass, (cs, nt) in _L33_PROBES:
        if lab.left() < 15:
            _log("probes: out of budget")
            break
        try:
            data = _lab_inputs(bclass, 1, 2048, k_cache, v_cache, gen)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            seq, pt = data["cache_seqlens"], data["page_table"]
            kc, vc = data["k_cache"].view(torch.uint8), data["v_cache"].view(torch.uint8)
            parts = []
            for mode in (8, 10, 0, 1):
                us = lab.time_graph(lambda: mod.probe(seq, pt, kc, vc, out, bclass, pt.shape[1], 2048, cs, nt, mode))
                parts.append(f"m{mode} {us:.2f}")
            _log(f"probe bs{bclass} cs{cs}x{nt} (us raw; m8 whole-row chain, m10 whole rows no page entries, m0 16 B "
                 f"pieces, m1 no K/V): Sd{bclass} v13 {base:.2f}; " + " ".join(parts))
        except Exception as exc:
            _log(f"probe bs{bclass}: FAILED {_why(exc)}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(3333)
    _log(f"L33: timed path v13: {sorted(CUDA_ROUTE.items())}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = _l33_w3c(lab, k_cache, v_cache, gen)
    _log(f"ROUTES = {chosen!r}")
    _anat_ud3(lab, k_cache, v_cache, gen)
    _l33_probes(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
