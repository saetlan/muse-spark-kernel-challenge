

# ---------------------------------------------------------------------------------------------
# Lab L19 (check runs only): anatomy of V5 (cuda_anat5.py = cuda_ver5.py + stamps, make_anat5.py)
# on the base's timed routes (v8: L16-L18 winners; nothing is re-routed). Built with ptxas -v so the
# log shows each variant's registers, stack frame and spill bytes (V5 minb1: 168 regs + 32 B local,
# minb2: 96 regs + 120 B local in L18). Per case: every CTA's consumer warps report cycles per stage
# waiting on the full barrier / Q.K+mask+max / exp+pack+V fragments / P.V+rescale, the producer its
# empty-barrier and page-entry waits; the report gives means per stage, CTA spans and CTAs per SM.
# Then V5t (cuda_ver5t.py: V re-laid once per stage by the producer, A56) against V5 at the same
# settings and against the class's current path on all six verify classes; a class moves only when
# correct and >= 3% faster (the base carries a V5t guess on Fv32 so the kind is built and checked).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L19_CASES = [("Fv16", 16, 0, [(4, 3, 2), (4, 3, 1)]), ("Fv1", 1, 0, [(1, 4, 1), (1, 3, 2)]),
              ("Sv16", 16, 2048, [(1, 3, 2), (1, 3, 1)])]                     # (name, batch, window, [(cps, s, minb)])
_AN19 = {}
_L19_V5T = {(True, True, 16): [(4, 3, 2), (4, 3, 1), (4, 4, 1), (6, 3, 2)], (True, True, 32): [(8, 3, 2), (8, 3, 1), (8, 4, 1)],
            (True, True, 1): [(1, 4, 1), (1, 3, 1), (1, 3, 2)], (False, True, 16): [(1, 3, 2), (1, 3, 1)],
            (False, True, 32): [(1, 3, 2), (1, 3, 1)], (False, True, 1): [(1, 3, 1), (1, 3, 2)]}   # (cps, stages, minb)
_L19_ERR = 0.12


def _l19_build():
    from torch.utils.cpp_extension import load_inline
    import tempfile
    _AN19["mod"] = load_inline(
        name="okc_l19_an5", cpp_sources=CUDA_AN5_DECL, cuda_sources=CUDA_AN5_SRC,
        functions=["an_ver5", "an_ver5_setup", "an_ver5_attrs"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-Xptxas", "-v"],
        build_directory=tempfile.mkdtemp(prefix="okc_l19_"), verbose=True)
    _log(f"an5 setup {_AN19['mod'].an_ver5_setup()}; attrs s3m2 {_AN19['mod'].an_ver5_attrs(3, 2)} "
         f"s3m1 {_AN19['mod'].an_ver5_attrs(3, 1)} s4m1 {_AN19['mod'].an_ver5_attrs(4, 1)}")


def _l19_run(data, cps, stages, minb, buf):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    nsplit = _nsplit(cfg, max_pages, cps, 64)
    tokens = batch * length
    o_part = torch.empty((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _AN19["mod"].an_ver5(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, minb, buf)
    return _merge(data, o_part, m_part, l_part, nsplit), nsplit


class _L19Host:   # host-side reads of the stamp buffer (lab only: first eager call, never under capture)
    @staticmethod
    def sync():
        torch.cuda.synchronize()

    @staticmethod
    def report(tag, buf, nctas, us):
        b = buf[:nctas * 64].view(nctas, 64).cpu().double()
        w = b[:, :48].view(nctas, 8, 6)                         # per warp: wait, qk, softmax+V, pv, stages, span
        st = w[:, :, 4].clamp(min=1)
        per = lambda k: float((w[:, :, k] / st).mean())        # noqa: E731  mean cycles per stage
        span = b[:, 57] - b[:, 56]
        loop = w[:, :, 5].mean(1)
        nst = float(w[:, 0, 4].mean())
        smid = b[:, 58].long()
        per_sm = torch.bincount(smid, minlength=132).float()
        prod_e, prod_p, prod_l = (b[:, 48] / b[:, 50].clamp(min=1)).mean(), (b[:, 49] / b[:, 50].clamp(min=1)).mean(), b[:, 50].mean()
        _log(f"{tag}: {us:.1f}us, {nctas} CTAs (up to {per_sm.max():.0f} on one SM over the run), {nst:.1f} stages/CTA; per stage per warp: "
             f"wait {per(0):.0f} qk {per(1):.0f} soft+V {per(2):.0f} pv {per(3):.0f} = {per(0) + per(1) + per(2) + per(3):.0f} cyc; "
             f"loop/stage {float((loop / w[:, 0, 4].clamp(min=1)).mean()):.0f} cyc; CTA span mean {float(span.mean()):.0f} max {float(span.max()):.0f}; "
             f"producer: empty-wait {100 * float(prod_e):.0f}% page-wait {100 * float(prod_p):.0f}% of {float(prod_l):.0f} cyc")


def _l19_compare(lab, k_cache, v_cache, gen):
    """V5t vs V5 (same settings) vs the current path, per verify class; routes winners (>= 3%)."""
    chosen = {}
    for cls, cfgs in _L19_V5T.items():
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
            cur = CUDA_ROUTE.get(cls) or (("fa3t",) + FA3T_ROUTE[cls] if cls in FA3T_ROUTE else "v5")
            _log(f"{name} current {cur}: {base:.1f}us")
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best = None
        for cps, s, mb in cfgs:
            for kind in ("ver5", "ver5t"):
                route = (kind, cps, s, mb)
                if lab.left() < 15:
                    break
                try:
                    err = float((_cuda_run(data, route).float() - ref).norm() / ref.norm())
                    us = lab.time_graph(lambda: _cuda_run(data, route))
                    _log(f"{name} {kind} c{cps} s{s} m{mb}: {us:.1f}us err {err:.1e}")
                    if err < _L19_ERR and (best is None or us < best[0]):
                        best = (us, route)
                except Exception as exc:
                    _log(f"{name} {kind} c{cps} s{s} m{mb}: FAILED {_why(exc)}")
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            FA3T_ROUTE.pop(cls, None)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs current {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP current path on {cls}" + (f" (best {best[1]} {best[0]:.1f}us vs {base:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1919)
    guesses = {c: r for c, r in CUDA_ROUTE.items() if r[0] == "ver5t"}
    for c in guesses:
        CUDA_ROUTE.pop(c)
    v5t_ok = bool(guesses) and "mod" in _CUDA
    _log(f"L19: ver5t guesses kept at warm-up {sorted(guesses)}; timed path until a class wins: {sorted(CUDA_ROUTE.items())}")
    if v5t_ok:
        for s, mb in ((3, 2), (3, 1), (4, 1)):
            a = _CUDA["mod"].attend_ver5t_attrs(s, mb)
            _log(f"ver5t s{s} minb{mb} attrs (regs, local B, static smem B): {a}{'  <-- LOCAL MEMORY' if a[1] else ''}")
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    try:
        _l19_build()
    except Exception as exc:
        _log(f"L19: anatomy build FAILED {_why(exc)}")
    buf = torch.zeros((4096 * 64,), dtype=torch.int64, device=dev)
    for name, batch, window, cfgs in _L19_CASES if "mod" in _AN19 else ():
        if lab.left() < 200:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(batch, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            for cps, s, mb in cfgs:
                out, ns = _l19_run(data, cps, s, mb, buf)
                err = float((out.float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _l19_run(data, cps, s, mb, buf))
                buf.zero_()
                _l19_run(data, cps, s, mb, buf)
                _L19Host.sync()
                _L19Host.report(f"{name} an5 c{cps} s{s} m{mb} n{ns} err {err:.1e}", buf, ns * batch, us)
                if name == "Fv16":
                    hot = dict(data)
                    hot["page_table"] = data["page_table"] % 8192
                    buf.zero_()
                    _l19_run(hot, cps, s, mb, buf)
                    _L19Host.sync()
                    _L19Host.report(f"{name} an5 c{cps} s{s} m{mb} L2-hot", buf, ns * batch, float("nan"))
        except Exception as exc:
            _log(f"{name}: FAILED {_why(exc)}")
    if v5t_ok:
        _l19_compare(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
