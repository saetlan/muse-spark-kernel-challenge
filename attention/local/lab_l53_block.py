


# ---------------------------------------------------------------------------------------------
# Lab L53 (check runs only), on v15 (the timed path = v15 byte for byte): a multi-row combine. v15's `_combine` runs one
# 4-warp program per (token, head) row: 4,096 tiny programs at Sv32 / Fv32 (judge profiles: 5.5 / 7.1 us), each paying
# its loads, then a dependent gate load. `_combine_r`: R rows per program ([R, CB, 64] tiles), the gate rows loaded
# before the split loop. Per class, the whole call with v15's merge vs `_combine_r` at R = 4 / 8 / 16 (error vs FP32);
# logged only (the merge is shared by every split route: v16 picks R from these numbers).
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L53_CASES = [("Sv32", (False, True, 32), 8, 2048), ("Fv32", (True, True, 32), 8, 0), ("Fv16", (True, True, 16), 8, 0),
              ("Fd16", (True, False, 16), 1, 0), ("Fd32", (True, False, 32), 1, 0), ("Fv1", (True, True, 1), 8, 0)]


@triton.jit(do_not_specialize=["nsplit", "n_rows"])
def _combine_r(o_part_ptr, m_part_ptr, l_part_ptr, qkv_ptr, v_scale_ptr, out_ptr, nsplit, n_rows,
               HQ: tl.constexpr, HKV: tl.constexpr, EPS: tl.constexpr, CB: tl.constexpr, R: tl.constexpr):
    r = tl.program_id(0) * R + tl.arange(0, R)                 # row = token * HQ + head (v15's partial order)
    rok = r < n_rows
    t = r // HQ
    h = r % HQ
    d = tl.arange(0, 64)
    j = tl.arange(0, CB)
    ROW: tl.constexpr = (2 * HQ + 2 * HKV) * 64
    g = tl.load(qkv_ptr + t.to(tl.int64)[:, None] * ROW + (HQ + 2 * HKV) * 64 + h[:, None] * 64 + d[None, :],
                mask=rok[:, None], other=0.0).to(tl.float32)   # gate rows first: independent of the partials
    big = tl.full([R], -1.0e30, tl.float32)
    den = tl.zeros([R], tl.float32)
    num = tl.zeros([R, 64], tl.float32)
    for s0 in range(0, nsplit, CB):
        ok = rok[:, None] & (s0 + j[None, :] < nsplit)
        idx = r.to(tl.int64)[:, None] * nsplit + s0 + j[None, :]
        m = tl.load(m_part_ptr + idx, mask=ok, other=-1.0e30)
        lsum = tl.load(l_part_ptr + idx, mask=ok, other=0.0)
        o = tl.load(o_part_ptr + idx[:, :, None] * 64 + d[None, None, :], mask=ok[:, :, None], other=0.0)
        new = tl.maximum(big, tl.max(m, 1))
        scale = tl.exp2(big - new)
        w = tl.exp2(m - new[:, None])
        den = den * scale + tl.sum(w * lsum, 1)
        num = num * scale[:, None] + tl.sum(w[:, :, None] * o, 1)
        big = new
    a = num / den[:, None] * tl.load(v_scale_ptr)
    a = a * tl.rsqrt(tl.sum(a * a, 1) / 64.0 + EPS)[:, None]
    a = a / (1.0 + tl.exp(-g))
    tl.store(out_ptr + t.to(tl.int64)[:, None] * (HQ * 64) + h[:, None] * 64 + d[None, :], a.to(tl.bfloat16),
             mask=rok[:, None])


def _merge_r_factory(R0):
    def merge_r(data, o_part, m_part, l_part, nsplit):
        cfg = data["config"]
        batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
        out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
        cb = min(64, triton.next_power_of_2(nsplit))
        R = R0
        while R > 1 and R * cb * 64 > 16384:
            R //= 2
        n_rows = batch * length * hq
        _combine_r[(triton.cdiv(n_rows, R),)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit, n_rows,
                                              HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, R=R, num_warps=4)
        return out
    return merge_r


def _lab_main():
    global _merge
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(5353)
    _log(f"L53: timed path v15: {sorted(CUDA_ROUTE.items())}")
    merge_v15 = _merge
    k_cache, v_cache = _lab_soft_cache(gen, 32 * scored_max_pages(32, 8))
    for name, cls, length, window in _L53_CASES:
        if lab.left() < 30:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(cls[2], length, window, k_cache, v_cache, gen)
            _merge = merge_v15
            _kernel_fn_submitted(data)
            ref = _lab_reference(data)
            base = lab.time_graph(lambda: _kernel_fn_submitted(data))
            berr = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
        except Exception as exc:
            _merge = merge_v15
            _log(f"{name}: setup FAILED {_why(exc)}")
            continue
        res = []
        for R in (4, 8, 16):
            try:
                _merge = _merge_r_factory(R)
                err = float((_kernel_fn_submitted(data).float() - ref).norm() / ref.norm())
                us = lab.time_graph(lambda: _kernel_fn_submitted(data))
                res.append(f"R{R} {us:.1f}us err {err:.1e} ({(us / base - 1) * 100:+.1f}%)")
            except Exception as exc:
                res.append(f"R{R} FAILED {_why(exc)[:120]}")
            finally:
                _merge = merge_v15
        _log(f"{name} v15 {CUDA_ROUTE.get(cls) or 'triton'} {base:.1f}us err {berr:.1e}; " + "; ".join(res))
    _merge = merge_v15
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
    del k_cache, v_cache
    torch.cuda.empty_cache()
