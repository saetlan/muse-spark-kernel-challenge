

# ---------------------------------------------------------------------------------------------
# Lab L22 (check runs only), on v8 (submission_v8.py), to find where W1 goes wrong (AF44: the one-wgmma
# unit test is right on the GPU, W1 is not):
#   1. wgmma unit test (cuda_w1unit.py) with three more variants: two chained k32 wgmmas (W1's Q.K),
#      the same on a preloaded accumulator (scale-d 1: D0 + A.B^T), and W1's whole consumer step in one
#      warpgroup (Q.K -> W1's P pack -> P.V on a Vt laid out as W1's transposers do);
#   2. W1 with a stage-0 dump (cuda_w1dbg.py) on Fv16, Sv16 and Fv1 lab inputs: every region the kernel
#      dumped (cache rows, V staging, K tile, Vt, Q bytes, raw Q.K, P, P.V) compared with torch, plus
#      split 0's normalized partial vs FP32, so the first wrong step is named.
# Nothing is routed: the timed path stays v8.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L22_UNIT = [(512, 0, 0, 0, 0), (512, 0, 0, 0, 1), (512, 0, 0, 0, 2), (512, 0, 0, 0, 3)]
_L22_PSCALE = 1.0 / 16


def _l22_build(name, src, decl, funcs):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=False)


def _l22_dec(u8):
    return u8.contiguous().view(torch.float8_e4m3fn).float()


def _l22_e4m3(x):
    return x.float().to(torch.float8_e4m3fn).view(torch.uint8)


def _l22_rel(a, b):
    a, b = torch.nan_to_num(a.float(), nan=1e9), b.float()
    return float((a - b).norm() / max(float(b.norm()), 1e-30))


def _l22_perm(got, ref, dim):
    """For each slice of got along dim (first 16), the index of the best-matching slice of ref (cosine)."""
    g, r = torch.nan_to_num(got.float()), ref.float()
    if dim == 1:
        g, r = g.T, r.T
    gn = g / g.norm(dim=1, keepdim=True).clamp(min=1e-30)
    rn = r / r.norm(dim=1, keepdim=True).clamp(min=1e-30)
    return [int(x) for x in (gn[:16] @ rn.T).argmax(1)]


def _l22_analyze(data, q, dbg, nsplit, o_part, m_part, l_part, sbov=528, pt_orig=None):
    """Compares the stage-0 dump of block (split 0, request 0 x KV pair 0) with torch. CPU tensors in (data may
    be compacted to request 0's rows: pt_orig = its original page-table row, for the copier's row check);
    returns compact log lines."""
    out = []
    cfg = data["config"]
    hq, hkv, page, window, L = cfg["num_q_heads"], cfg["num_kv_heads"], cfg["page_size"], cfg["window"], cfg["q_len"]
    dbg = dbg.float()
    n = int(data["cache_seqlens"][0])
    lo = max(n - L - window + 1, 0) if window > 0 else 0
    per = (n - lo + nsplit - 1) // nsplit
    per = (per + 15) // 16 * 16
    start, end = lo, min(lo + per, n)
    keys = torch.clamp(start + torch.arange(64), max=end - 1)
    pt = data["page_table"][0].long()
    rows = pt[keys // page] * page + keys % page
    got_rows = dbg[58624:58688].round().long()
    rows_chk = rows if pt_orig is None else pt_orig.long()[keys // page] * page + keys % page
    kc = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    vc = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)
    K, V = kc[rows], vc[rows]                                                   # [64 keys, 128] (heads 0, 1)
    vs_got = dbg[49408:49408 + 64 * 144].round().view(64, 144)[:, :128]
    out.append(f"n {n} start {start} end {end} nsplit {nsplit}: rows differ {int((got_rows != rows_chk).sum())}/64, "
               f"V staging bytes differ {int((vs_got != V.float()).sum())}/8192")
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * 1.4426950408889634
    r_ = torch.arange(64)
    tok, gq = 2 * (r_ // 16) + (r_ % 16) // 8, r_ % 8
    col = torch.arange(64)
    nn_, dd_ = col[:, None].expand(64, 64), col[None, :].expand(64, 64)       # (key / dim n, k)
    kap = torch.arange(64)
    sig = 16 * (kap // 16) + 8 * ((kap % 16 % 4) // 2) + 2 * (kap % 16 // 4) + (kap % 2)   # Vt k-position -> key
    kpos = start + col                                                          # stage-0 key of each column
    pos = n - L + tok
    ok = (kpos[None, :] < end) & (kpos[None, :] <= pos[:, None])
    if window > 0:
        ok = ok & (kpos[None, :] > pos[:, None] - window)
    for h in range(2):
        Kh, Vh = K[:, 64 * h:64 * h + 64], V[:, 64 * h:64 * h + 64]
        toff = (nn_ % 8) * 16 + (nn_ // 8) * 512 + (dd_ % 16) + (dd_ // 16) * 128   # K tile offset of (key, dim)
        kt = dbg[32768 + h * 4096:32768 + h * 4096 + 4096].round()
        kt_kd = kt[toff]                                                         # the dumped tile as [key, dim]
        voff = (nn_ % 8) * 16 + (nn_ // 8) * sbov + (dd_ % 16) + (dd_ // 16) * 128  # Vt offset of (dim, k-position)
        vt = dbg[40960 + h * 4224:40960 + h * 4224 + 8 * sbov].round()
        vt_dk = vt[voff]                                                         # [dim, k-position]
        vt_kd = torch.empty(64, 64)
        vt_kd[sig] = vt_dk.T                                                     # back to [key, dim]
        qb_exp = _l22_e4m3(q[tok, (2 * 0 + h) * 8 + gq])                         # request 0, kv0 = 0
        qb = dbg[24576 + h * 4096:24576 + h * 4096 + 4096].round().view(64, 64)
        S = dbg[h * 4096:h * 4096 + 4096].view(64, 64)
        S_dump = _l22_dec(qb.to(torch.uint8)) @ _l22_dec(kt_kd.to(torch.uint8)).T
        S_true = _l22_dec(qb_exp) @ _l22_dec(Kh).T
        P = dbg[8192 + h * 4096:8192 + h * 4096 + 4096].view(64, 64)
        z = torch.where(ok, S * qk_scale, torch.tensor(float("-inf")))
        P_exp = torch.where(ok, torch.exp2(z - z.amax(1, keepdim=True) + 8.0), torch.tensor(0.0))
        O = dbg[16384 + h * 4096:16384 + h * 4096 + 4096].view(64, 64)
        Pq = _l22_dec(_l22_e4m3(P))
        O_dump = Pq @ _l22_dec(vt_kd.to(torch.uint8))
        O_true = Pq @ _l22_dec(Vh)
        line = (f"h{h}: Ktile {int((kt_kd != Kh.float()).sum())}/4096 differ, Vt {int((vt_kd != Vh.float()).sum())}/4096, "
                f"Q bytes {int((qb != qb_exp.float()).sum())}/4096; S vs dump operands {_l22_rel(S, S_dump):.1e}, "
                f"vs true {_l22_rel(S, S_true):.1e}; P {_l22_rel(P, P_exp):.1e}; "
                f"PV vs dump Vt {_l22_rel(O, O_dump):.1e}, vs true V {_l22_rel(O, O_true):.1e}")
        out.append(line)
        if _l22_rel(S, S_dump) > 1e-3:
            out.append(f"h{h} S col -> dump col {_l22_perm(S, S_dump, 1)}; row -> {_l22_perm(S, S_dump, 0)}")
        if _l22_rel(O, O_dump) > 2e-2:
            out.append(f"h{h} PV col -> dump col {_l22_perm(O, O_dump, 1)}; row -> {_l22_perm(O, O_dump, 0)}")
        # split 0's whole normalized partial vs FP32 over its key range
        ks = torch.arange(start, end)
        rws = pt[ks // page] * page + ks % page
        Kf = _l22_dec(kc[rws][:, 64 * h:64 * h + 64])
        Vf = _l22_dec(vc[rws][:, 64 * h:64 * h + 64])
        okf = ks[None, :] <= pos[:, None]
        if window > 0:
            okf = okf & (ks[None, :] > pos[:, None] - window)
        zf = torch.where(okf, (_l22_dec(qb_exp) @ Kf.T) * qk_scale, torch.tensor(float("-inf")))
        pf = torch.exp2(zf - zf.amax(1, keepdim=True))
        ref_o = (pf @ Vf) / pf.sum(1, keepdim=True)
        part = ((0 * L + tok) * hq + (2 * 0 + h) * 8 + gq) * nsplit + 0
        got_o = o_part[part] / l_part[part][:, None]
        out.append(f"h{h} split-0 partial (normalized) vs FP32: {_l22_rel(got_o, ref_o):.1e}; "
                   f"m {[round(float(x), 2) for x in m_part[part][:4]]}")
    return out


class _L22Host:   # host-side reads (lab only: eager calls, never under capture)
    @staticmethod
    def unit(mod, gen):
        dev = torch.device("cuda")
        for sbo, swap, cp, am, ch in _L22_UNIT:
            kd = 64 if ch else 32
            A = (torch.randn((64, kd), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            B = (torch.randn((64, kd), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            Vv = (torch.randn((64, 64), generator=gen, device=dev) * 2).to(torch.float8_e4m3fn)
            D0 = torch.randn((64, 64), generator=gen, device=dev)
            S = A.float() @ B.float().T
            if ch == 3:
                ref = (S * _L22_PSCALE).to(torch.float8_e4m3fn).float() @ Vv.float()
            else:
                ref = S + (D0 if ch == 2 else 0)
            D = D0.clone() if ch == 2 else torch.full((64, 64), float("nan"), device=dev)
            try:
                mod.w1u(A.view(torch.uint8), B.view(torch.uint8), Vv.view(torch.uint8), D, sbo, swap, cp, am, ch, _L22_PSCALE)
                torch.cuda.synchronize()
                err = _l22_rel(D.cpu(), ref.cpu())
                _log(f"unit SBO {sbo} CHAIN {ch}: rel err {err:.2e}")
                if not err < 2e-2:
                    _log(f"unit CHAIN {ch}: col -> ref col {_l22_perm(D.cpu(), ref.cpu(), 1)}; row -> {_l22_perm(D.cpu(), ref.cpu(), 0)}")
            except Exception as exc:
                _log(f"unit SBO {sbo} CHAIN {ch}: FAILED {_why(exc)}")

    @staticmethod
    def compact(data):
        """Request 0's cache rows only (page size 1: row = page-table entry), on the CPU, page table = identity."""
        cfg = data["config"]
        assert cfg["page_size"] == 1
        hkv = cfg["num_kv_heads"]
        n0 = int(data["cache_seqlens"][0])
        pt0 = data["page_table"][0, :n0].long()
        small = dict(data)
        small["k_cache"] = data["k_cache"].view(torch.uint8).reshape(-1, hkv * 64)[pt0].cpu()
        small["v_cache"] = data["v_cache"].view(torch.uint8).reshape(-1, hkv * 64)[pt0].cpu()
        small["page_table"] = torch.arange(n0, dtype=torch.int32)[None, :]
        small["cache_seqlens"] = data["cache_seqlens"][:1].cpu()
        small["k_scale"] = data["k_scale"].cpu()
        return small, pt0.cpu()

    @staticmethod
    def dump(mod, data, name):
        cfg = data["config"]
        batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
        max_pages = data["page_table"].shape[1]
        q = _prologue_q(data)
        nsplit = _nsplit_w(cfg, max_pages, 1, 64)
        tokens = batch * length
        o_part = torch.zeros((tokens * hq * nsplit, 64), dtype=torch.float32, device=q.device)
        m_part = torch.zeros((tokens * hq * nsplit,), dtype=torch.float32, device=q.device)
        l_part = torch.ones_like(m_part)
        dbg = torch.full((W1DBG_FLOATS,), float("nan"), dtype=torch.float32, device=q.device)
        mod.attend_w1dbg(q, data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"],
                         data["cache_seqlens"], o_part, m_part, l_part, dbg, batch, hq, hkv, max_pages, nsplit,
                         cfg["page_size"], cfg["window"], cfg["sm_scale"])
        torch.cuda.synchronize()
        small, pt0 = _L22Host.compact(data)
        o_c, m_c, l_c = o_part.cpu(), m_part.cpu(), l_part.cpu()
        for line in _l22_analyze(small, q.float().cpu(), dbg.cpu(), nsplit, o_c, m_c, l_c, pt_orig=pt0):
            _log(f"{name} {line}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2222)
    _log(f"L22: timed path v8: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    try:
        unit = _l22_build("okc_l22_unit", CUDA_W1U_SRC, CUDA_W1U_DECL, ["w1u"])
        _L22Host.unit(unit, gen)
    except Exception as exc:
        _log(f"unit: build/run FAILED {_why(exc)}")
    try:
        mod = _l22_build("okc_l22_w1dbg", CUDA_W1DBG_SRC, CUDA_W1DBG_DECL, ["attend_w1dbg"])
    except Exception as exc:
        _log(f"w1dbg: build FAILED {_why(exc)}")
        return
    k_cache, v_cache = _lab_soft_cache(gen, 16 * scored_max_pages(16, 8))
    for name, bclass, window in (("Fv16", 16, 0), ("Sv16", 16, 2048), ("Fv1", 1, 0)):
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _L22Host.dump(mod, data, name)
        except Exception as exc:
            _log(f"{name}: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
