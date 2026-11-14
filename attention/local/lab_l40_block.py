


# ---------------------------------------------------------------------------------------------
# Lab L40 (check runs only), on v14 (the timed path = v14 byte for byte): W3s (cuda_w3s.py, make_w3s.py, AF65): W3b with a
# length-aware split that spans requests (the flattened 64-key stage list cut into G = 132 equal ranges, one per CTA;
# per-segment Q tile / partial; _combine_s merges each request's slots, its count recomputed from seqlens). On Fv16, Fv32,
# Fv1 vs v14's path: attrs, error vs FP32, watchdog, time; routed when correct and >= 3% faster
# (("w3s", stages, pp, nwg, rc)). Then the tail test again (Fv: W3s if routed) to see the tail gone.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_W3S = {}
_L40_ERR = 0.12
_L40_G = NUM_SMS
_L40_CASES = [("Fv16", (True, True, 16)), ("Fv32", (True, True, 32)), ("Fv1", (True, True, 1))]


@triton.jit(do_not_specialize=["batch"])
def _combine_s(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, seqlens_ptr, batch,
               HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr, BP: tl.constexpr,
               G: tl.constexpr, NPAIR: tl.constexpr, QL: tl.constexpr):
    """_combine with W3s's slots: request b's rows read (last - first + 1) * NPAIR partials, first / last the CTAs that
    cover its 64-key stages in the flattened list cut into G ranges of ceil(S_total / G) stages (make_w3s.py)."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    th = t.to(tl.int64) * HQ + h
    b = t // QL
    i = tl.arange(0, BP)
    nb = tl.load(seqlens_ptr + i, mask=i < batch, other=0)
    sb = (nb + 63) // 64
    stot = tl.sum(sb, 0)
    qs = tl.maximum((stot + G - 1) // G, 1)
    pb = tl.sum(tl.where(i < b, sb, 0), 0)
    sbb = tl.sum(tl.where(i == b, sb, 0), 0)
    first = pb // qs
    last = (pb + sbb - 1) // qs
    nsl = (last - first + 1) * NPAIR
    stride = G * NPAIR
    j = tl.arange(0, CB)
    d = tl.arange(0, 64)
    big = tl.max(tl.full([CB], -1.0e30, tl.float32), 0)
    den = tl.sum(tl.zeros([CB], tl.float32), 0)
    num = tl.zeros([64], tl.float32)
    for s0 in range(0, nsl, CB):
        ok = s0 + j < nsl
        idx = th * stride + s0 + j
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1.0e30)
        lsum = tl.load(l_part_ptr + idx, mask=ok, other=0.0)
        o = tl.load(o_part_ptr + idx[:, None] * 64 + d[None, :], mask=ok[:, None], other=0.0)
        new = tl.maximum(big, tl.max(m, 0))
        scale = tl.exp2(big - new)
        w = tl.exp2(m - new)
        den = den * scale + tl.sum(w * lsum, 0)
        num = num * scale + tl.sum(w[:, None] * o, 0)
        big = new
    a = num / den * tl.load(v_scale_ptr)
    a = a * tl.rsqrt(tl.sum(a * a, 0) / 64.0 + EPS)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    gt = tl.load(qkv_ptr + t.to(tl.int64) * ROW + (HQ + 2 * HKV) * 64 + h * 64 + d).to(tl.float32)
    a = a / (1.0 + tl.exp(-gt))
    tl.store(out_ptr + t.to(tl.int64) * (HQ * 64) + h * 64 + d, a.to(tl.bfloat16))


def _w3s_run(data, stages, pp, nwg, rc):
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    max_pages = data["page_table"].shape[1]
    q = _prologue_q(data)
    G, npair = _L40_G, nwg // 2
    tokens = batch * length
    rows = tokens * hq * G * npair
    o_part = torch.empty((rows, 64), dtype=torch.float32, device=q.device)
    m_part = torch.empty((rows,), dtype=torch.float32, device=q.device)
    l_part = torch.empty_like(m_part)
    _W3S["mod"].attend_w3s(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                           data["cache_seqlens"], o_part, m_part, l_part, batch, hq, hkv, max_pages, G,
                           cfg["page_size"], cfg["window"], cfg["sm_scale"], stages, pp, nwg, rc)
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=q.device)
    _combine_s[(tokens, hq)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, data["cache_seqlens"], batch,
                             HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=16, BP=max(2, triton.next_power_of_2(batch)), G=G,
                             NPAIR=npair, QL=length, num_warps=4)
    return out


_cuda_ok_l40 = _cuda_ok
_cuda_run_l40 = _cuda_run


def _cuda_ok(cfg, route):  # noqa: F811 (lab: W3s routes run on the lab's module)
    if route[0] == "w3s":
        return ("mod" in _W3S and cfg["q_len"] == 8 and cfg["window"] == 0 and cfg["batch"] <= 32
                and cfg["num_q_heads"] == 8 * cfg["num_kv_heads"] and cfg["num_kv_heads"] % 2 == 0)
    return _cuda_ok_l40(cfg, route)


def _cuda_run(data, route):  # noqa: F811
    if route[0] == "w3s":
        return _w3s_run(data, *route[1:])
    return _cuda_run_l40(data, route)


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(4040)
    _log(f"L40: timed path v14: {sorted(CUDA_ROUTE.items())}")
    try:
        from torch.utils.cpp_extension import load_inline
        import tempfile
        _W3S["mod"] = load_inline(name="okc_l40_w3s", cpp_sources=CUDA_W3S_DECL, cuda_sources=CUDA_W3S_SRC,
                                  functions=["attend_w3s", "attend_w3s_setup", "attend_w3s_attrs", "attend_w3s_hang"],
                                  extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                                  build_directory=tempfile.mkdtemp(prefix="okc_l40_"), verbose=False)
        refused = _W3S["mod"].attend_w3s_setup()
        variants = []
        for i, (s, p, nw, rc, rp) in enumerate(W3S_VARIANTS):
            a = _W3S["mod"].attend_w3s_attrs(s, p, nw, rc)
            ok = not (refused >> i) & 1
            _log(f"w3s s{s} pp{p} nwg{nw} rc{rc}/{rp} attrs (regs, local B, static smem B, pool ok): {a}"
                 f"{'  <-- LOCAL MEMORY' if a[1] else ''}{'' if ok else '  REFUSED'}")
            if ok:
                variants.append(("w3s", s, p, nw, rc))
    except Exception as exc:
        _W3S.pop("mod", None)
        _log(f"w3s: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    chosen = {}
    for name, cls in _L40_CASES:
        if "mod" not in _W3S or lab.left() < 60:
            break
        try:
            data = _lab_inputs(cls[2], 8, 0, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
        except Exception as exc:
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        best, res = None, []
        for route in variants:
            try:
                err = float((_w3s_run(data, *route[1:]).float() - ref).norm() / ref.norm())   # float(): finished
                if _W3S["mod"].attend_w3s_hang():
                    res.append(f"{route} HANG (watchdog)")
                    _W3S.pop("mod", None)
                    break
                us = lab.time_graph(lambda: _w3s_run(data, *route[1:]))
                if _W3S["mod"].attend_w3s_hang():
                    res.append(f"{route} HANG in timing (watchdog)")
                    _W3S.pop("mod", None)
                    break
                res.append(f"{route} {us:.1f}us err {err:.1e}")
                if err < _L40_ERR and (best is None or us < best[0]):
                    best = (us, route)
            except Exception as exc:
                res.append(f"{route} FAILED {_why(exc)[:160]}")
        _log(f"{name} v14 {CUDA_ROUTE.get(cls)} {base:.1f}us; " + "; ".join(res))
        if best is not None and best[0] < 0.97 * base:
            CUDA_ROUTE[cls] = best[1]
            PDL_ROUTE.discard(cls)
            chosen[cls] = best[1]
            _log(f"ROUTE {cls} -> {best[1]}: {best[0]:.1f}us vs {base:.1f}us ({base / best[0]:.2f}x)")
        else:
            _log(f"KEEP {cls}" + (f" (best {best[1]} {best[0]:.1f}us)" if best else ""))
    _log(f"ROUTES = {chosen!r}")
    _lab_tail(lab, k_cache, v_cache, gen)
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
