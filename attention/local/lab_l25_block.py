

# ---------------------------------------------------------------------------------------------
# Lab L25 (check runs only), on v9 (the timed path = v9 byte for byte: W1-N on five verify classes):
#   1. AF46 bisection (make_w1m.py): W1-M with only dump branch k (MASK 1/2/4/8), none (0) or all four (15 = W1-N),
#      all guarded off by a null pointer, and W1-C (W1 + "memory" clobbers on its cp.async waits) on Sv16 and Fv1:
#      error vs the lab reference (right ~0.07, wrong ~1.4); W1-C also timed on Fv16 / Sv32 vs the v9 path;
#   2. W1 anatomy (A64, cuda_anatw1.py = W1-C with %clock64 stamps) on Fv16, Fv32, Sv32, Fv1 lab inputs, cold L2:
#      cycles per 64-key stage per role (consumers: wait on full, Q.K pair, softmax + pack, P.V pair, O update;
#      copier: wait on empty / on its cp.async groups; transposers: wait on land), CTA span.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L25_CASES = [("Fv16", 16, 0), ("Fv32", 32, 0), ("Sv32", 32, 2048), ("Fv1", 1, 0)]


class _L25Host:   # host-side reads (lab only: eager calls, never under capture)
    @staticmethod
    def anat(mod, data, name, stages):
        cfg = data["config"]
        batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
        max_pages = data["page_table"].shape[1]
        q = _prologue_q(data)
        nsplit = _nsplit_w(cfg, max_pages, 1, 64)
        R = batch * length * hq * nsplit
        o = torch.empty((R, 64), dtype=torch.float32, device=q.device)
        m = torch.empty((R,), dtype=torch.float32, device=q.device)
        l_ = torch.empty_like(m)
        ctas = nsplit * batch * (hkv // 2)
        buf = torch.zeros((ctas * 64,), dtype=torch.int64, device=q.device)
        args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"], o, m, l_,
                batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, 528, 0)
        mod.an_w1(*args, buf)                                   # warm (module load, L2 state)
        flush = torch.empty((512 << 20) // 8, dtype=torch.int64, device=q.device)
        flush.fill_(1)
        buf.zero_()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        mod.an_w1(*args, buf)
        e1.record()
        torch.cuda.synchronize()
        us = e0.elapsed_time(e1) * 1e3
        b = buf.view(ctas, 64).cpu().double()
        nst = b[:, 59].clamp(min=1)
        active = b[:, 59] > 0
        b, nst = b[active], nst[active]
        cons = b[:, :48].view(-1, 8, 6)                         # [cta, warp, slot]
        per = lambda x: float((x / nst[:, None]).mean())        # noqa: E731  cycles per stage, mean over CTAs and warps
        wait, qk, sm, pv, ou = (per(cons[:, :, k]) for k in range(5))
        span = float((b[:, 57] - b[:, 56]).mean())
        cyc_per_us = float((b[:, 57].max() - b[:, 56].min()) / us)
        _log(f"{name} anat s{stages} nsplit {nsplit}: kernel {us:.1f}us (~{cyc_per_us:.0f} cyc/us), stages/CTA {float(nst.mean()):.1f}, "
             f"CTA span {span:.0f} cyc = {span / float(nst.mean()):.0f} cyc/stage; consumer per stage: wait full {wait:.0f}, "
             f"QK {qk:.0f}, softmax+pack {sm:.0f}, PV {pv:.0f}, O upd {ou:.0f}; copier per stage: wait empty "
             f"{float((b[:, 48] / nst).mean()):.0f}, wait groups {float((b[:, 49] / nst).mean()):.0f}; transposers wait land "
             f"{float((b[:, 52] / nst).mean()):.0f} / {float((b[:, 54] / nst).mean()):.0f}")
        del flush


def _l25_build(name, src, decl, funcs):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=False)


def _l25_run(mods, data, kind, arg):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit_w(cfg, max_pages, 1, 64)
    R = batch * length * hq * nsplit
    o = torch.empty((R, 64), dtype=torch.float32, device=q.device)
    m = torch.empty((R,), dtype=torch.float32, device=q.device)
    l_ = torch.empty_like(m)
    args = (q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"], o, m, l_,
            batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"], cfg["sm_scale"])
    if kind == "w1m":
        mods["w1m"].attend_w1m(*args, 4, arg)
    else:
        mods["w1c"].attend_w1c(*args, 4, 528, 0)
    return _merge(data, o, m, l_, nsplit)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2525)
    _log(f"L25: timed path v9: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    mods = {}
    for key, name, src, decl, funcs, setup in (
            ("w1m", "okc_l25_w1m", CUDA_W1M_SRC, CUDA_W1M_DECL, ["attend_w1m", "attend_w1m_setup"], "attend_w1m_setup"),
            ("w1c", "okc_l25_w1c", CUDA_W1C_SRC, CUDA_W1C_DECL, ["attend_w1c", "attend_w1c_setup", "attend_w1c_attrs"], "attend_w1c_setup"),
            ("aw1", "okc_l25_aw1", CUDA_AW1_SRC, CUDA_AW1_DECL, ["an_w1", "an_w1_setup", "an_w1_attrs"], "an_w1_setup")):
        try:
            mods[key] = _l25_build(name, src, decl, funcs)
            if getattr(mods[key], setup)():
                raise RuntimeError(f"{setup} refused")
        except Exception as exc:
            mods.pop(key, None)
            _log(f"{key}: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    # 1. bisection
    for name, bclass, window in (("Sv16", 16, 2048), ("Fv1", 1, 0)):
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            res = []
            for kind, arg in [("w1m", mk) for mk in (0, 1, 2, 4, 8, 15)] + [("w1c", None)]:
                if kind not in mods:
                    continue
                try:
                    err = float((_l25_run(mods, data, kind, arg).float() - ref).norm() / ref.norm())
                    res.append(f"{kind}{'' if arg is None else arg} {err:.2f}")
                except Exception as exc:
                    res.append(f"{kind}{'' if arg is None else arg} FAILED {_why(exc)}")
            _log(f"{name} bisection err: " + ", ".join(res))
        except Exception as exc:
            _log(f"{name} bisection: FAILED {_why(exc)}")
    if "w1c" in mods:
        for name, bclass, window in (("Fv16", 16, 0), ("Sv32", 32, 2048)):
            try:
                data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
                _kernel_fn_submitted(data)
                ref = _lab_reference(data)
                base = lab.time_graph(lambda: _kernel_fn_submitted(data))
                err = float((_l25_run(mods, data, "w1c", None).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _l25_run(mods, data, "w1c", None))
                _log(f"{name} v9 path {base:.1f}us; w1c {us:.1f}us err {err:.1e}")
            except Exception as exc:
                _log(f"{name} w1c timing: FAILED {_why(exc)}")
    # 2. anatomy
    if "aw1" in mods:
        _log(f"aw1 s4 attrs (regs, local B, static smem B): {mods['aw1'].an_w1_attrs(4, 528, 0)}")
        for name, bclass, window in _L25_CASES:
            if lab.left() < 30:
                _log("out of budget")
                break
            cls = (window == 0, True, bclass)
            route = CUDA_ROUTE.get(cls)
            stages = route[2] if route is not None and route[0] in ("w1", "w1n") else 4
            try:
                data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
                _kernel_fn_submitted(data)
                _L25Host.anat(mods["aw1"], data, name, stages)
            except Exception as exc:
                _log(f"{name}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
