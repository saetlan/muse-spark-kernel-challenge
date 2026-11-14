

# ---------------------------------------------------------------------------------------------
# Lab L23 (check runs only), on v8 (submission_v8.py): differential test of W1 (L22: the debug copy's
# stage 0 and split 0 of block 0 are right, yet W1's merged output was uncorrelated in L20/L21).
# On the same inputs (Fv16, Sv16, Fv1, Sv1) and the same whole-wave split count:
#   real W1 (cuda_w1.py, 4 stages, Vt 528, WF 0), the debug W1 (cuda_w1dbg.py), V5 (v8's module, 4 stages,
#   minb 1), each into NaN-filled o / m / l buffers; per kernel: unwritten rows, error of every (row, split)
#   normalized partial vs FP32 (bad rows, which splits / requests), m and l vs FP32, the merged output vs the
#   lab reference; FP32 partials in the kernels' convention through _merge (tests the merge itself).
# Then the instruction-throughput microbenchmark (lab_xu_block.py). Nothing is routed.
# ---------------------------------------------------------------------------------------------
_LAB_BUDGET_S = 450.0
_L23_LOG2E = 1.4426950408889634


def _l23_build(name, src, decl, funcs):
    from torch.utils.cpp_extension import load_inline
    import tempfile
    return load_inline(name=name, cpp_sources=decl, cuda_sources=src, functions=funcs,
                       extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"],
                       build_directory=tempfile.mkdtemp(prefix=name + "_"), verbose=False)


def _l23_rel(a, b):
    a, b = torch.nan_to_num(a.float(), nan=1e9), b.float()
    return float((a - b).norm() / max(float(b.norm()), 1e-30))


def _l23_split_ref(data, q, nsplit):
    """FP32 split partials in the kernels' convention: rows ((b*L + l)*hq + head)*nsplit + split; o and l scaled by
    2^(z - m + 8) as the kernels do (z = score x sm_scale x k_scale x log2 e, m = the split's row max), plus a mask of
    rows with at least one key."""
    cfg = data["config"]
    B, L, hq, hkv, window, page = (cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"], cfg["window"],
                                   cfg["page_size"])
    G = hq // hkv
    dev = q.device
    qk_scale = cfg["sm_scale"] * float(data["k_scale"]) * _L23_LOG2E
    q8 = q.float().to(torch.float8_e4m3fn).float().view(B * L, hq, 64)
    kf = data["k_cache"].view(-1, hkv, 64)
    vf = data["v_cache"].view(-1, hkv, 64)
    R = B * L * hq * nsplit
    o_ref = torch.zeros((R, 64), device=dev)
    m_ref = torch.full((R,), -1.0e30, device=dev)
    l_ref = torch.zeros((R,), device=dev)
    valid = torch.zeros((R,), dtype=torch.bool, device=dev)
    la, ga = torch.arange(L, device=dev), torch.arange(G, device=dev)
    for b in range(B):
        n = int(data["cache_seqlens"][b])
        lo = max(n - L - window + 1, 0) if window > 0 else 0
        per = (n - lo + nsplit - 1) // nsplit
        per = (per + 15) // 16 * 16
        keys = torch.arange(lo, n, device=dev)
        rows = data["page_table"][b, keys // page].long() * page + keys % page
        k, v = kf[rows].float(), vf[rows].float()                       # [nk, hkv, 64]
        pos = n - L + la
        ok = keys[None, :] <= pos[:, None]
        if window > 0:
            ok = ok & (keys[None, :] > pos[:, None] - window)
        for h in range(hkv):
            z = torch.einsum("lgd,kd->lgk", q8[b * L:(b + 1) * L, h * G:(h + 1) * G], k[:, h]) * qk_scale
            z = z.masked_fill(~ok[:, None, :], float("-inf"))
            for s in range(nsplit):
                s0, s1 = lo + s * per, min(lo + s * per + per, n)
                if s1 <= s0:
                    continue
                zs = z[:, :, s0 - lo:s1 - lo]
                m = zs.amax(-1, keepdim=True)
                good = torch.isfinite(m[..., 0])
                p = torch.exp2(zs - torch.where(good[..., None], m, torch.zeros_like(m)) + 8.0)
                idx = (((b * L + la[:, None]) * hq + h * G + ga[None, :]) * nsplit + s).reshape(-1)
                o_ref[idx] = (p @ v[s0 - lo:s1 - lo, h]).reshape(-1, 64)
                l_ref[idx] = p.sum(-1).reshape(-1)
                m_ref[idx] = torch.where(good, m[..., 0], torch.full_like(m[..., 0], -1.0e30)).reshape(-1)
                valid[idx] = good.reshape(-1)
    return o_ref, m_ref, l_ref, valid


class _L23Host:   # host-side reads (lab only: eager calls, never under capture)
    @staticmethod
    def diag(mods, data, name):
        cfg = data["config"]
        batch, length, hq, hkv = cfg["batch"], cfg["q_len"], cfg["num_q_heads"], cfg["num_kv_heads"]
        max_pages = data["page_table"].shape[1]
        q = _prologue_q(data)
        nsplit = _nsplit_w(cfg, max_pages, 1, 64)
        R = batch * length * hq * nsplit
        nan = float("nan")
        args = (data["k_cache"], data["v_cache"], data["k_scale"], data["page_table"], data["cache_seqlens"])
        tail = (batch, hq, hkv, max_pages, nsplit, cfg["page_size"], cfg["window"], cfg["sm_scale"])
        parts = {}
        for kname in ("w1", "w1dbg", "v5"):
            o = torch.full((R, 64), nan, device=q.device)
            m = torch.full((R,), nan, device=q.device)
            l_ = torch.full((R,), nan, device=q.device)
            try:
                if kname == "w1":
                    mods["w1"].attend_w1(q, *args, o, m, l_, *tail, 4, 528, 0)
                elif kname == "w1dbg":
                    dbg = torch.full((W1DBG_FLOATS,), nan, device=q.device)
                    mods["w1dbg"].attend_w1dbg(q, *args, o, m, l_, dbg, *tail)
                else:
                    _CUDA["mod"].attend_ver5(q, *args, o, m, l_, *tail, 4, 1)
                torch.cuda.synchronize()
                parts[kname] = (o, m, l_)
            except Exception as exc:
                _log(f"{name} {kname}: FAILED {_why(exc)}")
        o_ref, m_ref, l_ref, valid = _l23_split_ref(data, q, nsplit)
        ref = _lab_reference(data)
        vr = valid.nonzero()[:, 0]
        _log(f"{name}: nsplit {nsplit}, rows {R}, valid {int(valid.sum())}; FP32 partials merged vs reference "
             f"{_l23_rel(_merge(data, o_ref, m_ref, l_ref, nsplit).float(), ref):.1e}")
        norm_ref = o_ref[vr] / l_ref[vr][:, None]
        for kname, (o, m, l_) in parts.items():
            unw = (int(torch.isnan(o).any(-1).sum()), int(torch.isnan(m).sum()), int(torch.isnan(l_).sum()))
            no = o[vr] / l_[vr][:, None]
            rerr = (no - norm_ref).norm(dim=-1) / norm_ref.norm(dim=-1).clamp(min=1e-30)
            rerr = torch.nan_to_num(rerr, nan=1e9)
            bad = vr[rerr > 0.1]
            bad_split = sorted({int(x) for x in (bad % nsplit)[:4096].tolist()})[:12]
            bad_b = sorted({int(x) for x in (bad // (nsplit * hq * length))[:4096].tolist()})[:12]
            dm = float(torch.nan_to_num((m[vr] - m_ref[vr]).abs(), nan=1e9).max())
            dl = float(torch.nan_to_num(((l_[vr] - l_ref[vr]) / l_ref[vr]).abs(), nan=1e9).max())
            try:
                merged = _l23_rel(_merge(data, torch.nan_to_num(o), torch.nan_to_num(m, nan=-1.0e30),
                                         torch.nan_to_num(l_), nsplit).float(), ref)
            except Exception as exc:
                merged = f"FAILED {_why(exc)}"
            _log(f"{name} {kname}: unwritten o/m/l rows {unw}; partials vs FP32 {_l23_rel(no, norm_ref):.1e}, "
                 f"bad rows {int(bad.numel())}/{int(vr.numel())} (splits {bad_split}, requests {bad_b}); "
                 f"max |dm| {dm:.2e}, max l rel {dl:.2e}; merged vs ref {merged if isinstance(merged, str) else f'{merged:.1e}'}")
        if "w1" in parts and "w1dbg" in parts:
            (o1, m1, l1), (o2, m2, l2) = parts["w1"], parts["w1dbg"]
            same = bool(torch.equal(torch.nan_to_num(o1), torch.nan_to_num(o2)) and torch.equal(torch.nan_to_num(m1), torch.nan_to_num(m2)))
            _log(f"{name} w1 vs w1dbg: bitwise equal {same}; o rel {_l23_rel(o1, o2):.1e}")


def _lab_main():
    lab = _Lab()
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(2323)
    _log(f"L23: timed path v8: {sorted(CUDA_ROUTE.items())} fa3t {FA3T_ROUTE}")
    mods = {}
    try:
        mods["w1"] = _l23_build("okc_l23_w1", CUDA_W1_SRC, CUDA_W1_DECL, ["attend_w1", "attend_w1_setup", "attend_w1_attrs"])
        err = mods["w1"].attend_w1_setup()
        if err:
            _log(f"w1: attend_w1_setup returned {err}")
    except Exception as exc:
        _log(f"w1: build FAILED {_why(exc)}")
    try:
        mods["w1dbg"] = _l23_build("okc_l23_w1dbg", CUDA_W1DBG_SRC, CUDA_W1DBG_DECL, ["attend_w1dbg"])
    except Exception as exc:
        _log(f"w1dbg: build FAILED {_why(exc)}")
    k_cache, v_cache = _lab_soft_cache(gen, 16 * scored_max_pages(16, 8))
    for name, bclass, window in (("Fv16", 16, 0), ("Sv16", 16, 2048), ("Fv1", 1, 0), ("Sv1", 1, 2048)):
        if lab.left() < 90:
            _log("out of budget")
            break
        try:
            data = _lab_inputs(bclass, 8, window, k_cache, v_cache, gen)
            _kernel_fn_submitted(data)
            _L23Host.diag(mods, data, name)
        except Exception as exc:
            _log(f"{name}: FAILED {_why(exc)}")
    del k_cache, v_cache
    torch.cuda.empty_cache()
    if lab.left() > 40:
        try:
            _XUHost.bench(lab)
        except Exception as exc:
            _log(f"xu: FAILED {_why(exc)}")
    _log(f"done in {_time.monotonic() - lab.t0:.0f}s")
