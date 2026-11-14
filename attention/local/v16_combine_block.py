# =============================================================================================
# v16: multi-row combine (L53): R rows per program ([R, CB, 64] tiles, gate rows loaded before the split loop). L53 on
# the judge's H200, whole calls: Sv32 (4 splits) -10.3% at R = 16; Fv16 / Fv32 / Fd16 / Fd32 within +-1%; Fv1 (132+
# splits) +7%. So `_merge` uses it only for calls with at most 4 splits; every other call keeps v15's `_combine`.
# =============================================================================================
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


_merge_v15 = _merge


def _merge(data, o_part, m_part, l_part, nsplit):  # noqa: F811
    if nsplit > 4:
        return _merge_v15(data, o_part, m_part, l_part, nsplit)
    cfg = data["config"]
    batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
    out = torch.empty((batch, length, hq * 64), dtype=torch.bfloat16, device=data["qkv"].device)
    cb = triton.next_power_of_2(nsplit)
    n_rows = batch * length * hq
    _combine_r[(triton.cdiv(n_rows, 16),)](o_part, m_part, l_part, data["qkv"], data["v_scale"], out, nsplit, n_rows,
                                           HQ=hq, HKV=hkv, EPS=RMS_EPS, CB=cb, R=16, num_warps=4)
    return out
