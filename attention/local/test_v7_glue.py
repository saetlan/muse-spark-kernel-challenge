"""CPU test of a v7 file's glue (default attention/submission_v7_lab.py): the compiled module is a mock
whose uv4 runs emulate_cuda_uv4.uv4 from the exact positional arguments v7 passes (a wrong order,
flag or buffer shape shows up as a wrong result), and whose ver4 / dec4 / ud3 run their emulators
as in test_v6_glue.py. Checks: 1. the warm-up keeps every correct route (uv4 at the FP8 tolerance);
2. each sliding verify class (and a full verify class routed by hand) runs uv4 with its route's
cluster size / stages / m-tiles and passes the judge's check_implementation (output and new cache
entries); 3. uv4 takes verify only;
4. a uv4 kernel returning a wrong result is dropped at warm-up and its class falls back to v5's path;
5. v7's own self-check inputs (not mocked) give every request disjoint cache rows.
Run: .venv/bin/python attention/local/test_v7_glue.py [file]"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, PROB)
import emulate  # noqa: E402
import emulate_cuda_uv4 as emu_uv4  # noqa: E402
import reference as ref  # noqa: E402
from emulate_cuda_dec3 import attend_dec3  # noqa: E402
from emulate_cuda_ud import run_ud  # noqa: E402
from emulate_cuda_ver4 import attend_ver4  # noqa: E402
from emulate_cuda_ver5 import attend_ver5  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(PROB, "submission_v7_lab.py")
v7 = emulate.load_submission(path)
calls = []
corrupt = set()


def as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, q_len):
    cfg = dict(batch=batch, q_len=q_len, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
               sm_scale=sm_scale)
    return dict(k_cache=kc, v_cache=vc, k_scale=k_scale, v_scale=k_scale, page_table=pt, cache_seqlens=seqlens,
                config=cfg)


def full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window, rope,
              qk_factor, sm_scale, q_len):
    return dict(qkv=qkv, positions=positions, cache_loc=cache_loc, k_cache=kc, v_cache=vc, k_scale=k_scale,
                v_scale=v_scale, page_table=pt, cache_seqlens=seqlens,
                config=dict(batch=batch, q_len=q_len, num_q_heads=hq, num_kv_heads=hkv, page_size=page, window=window,
                            rope_theta=500000.0 if rope else 0.0, qk_scale_factor=qk_factor, sm_scale=sm_scale))


class Mod:
    def attend_ver4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, stages, mt, minb):
        calls.append(("ver4", nsplit, stages, mt, minb))
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = attend_ver4(d, q, nsplit, stages, mt)
        o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)

    def attend_ver5(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, stages, minb):
        calls.append(("ver5", nsplit, stages, minb))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = attend_ver5(d, q, nsplit, stages)
        o_part.copy_(o.flip(-1) if "ver5" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_ver5t(self, *a):   # V5t: V5's math with the V fragments through the producer's re-laid tile
        import emulate_cuda_ver5 as e5
        import emulate_cuda_ver5t as e5t
        e5.VFRAG = e5t.vt_frags
        try:
            self.attend_ver5(*a)
            calls[-1] = ("ver5t",) + calls[-1][1:]
        finally:
            e5.VFRAG = None

    def attend_w1(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                  nsplit, page, window, sm_scale, stages, sbov, wf):   # v9: W1 through its emulator (CUTLASS layouts)
        import emulate_cuda_w1 as ew1
        calls.append(("w1", nsplit, stages, sbov, wf))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, sbov)
        o_part.copy_(o.flip(-1) if "w1" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_w1n(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                   nsplit, page, window, sm_scale, stages):   # v9: W1-N (W1's math, Vt stride 528) through W1's emulator
        import emulate_cuda_w1 as ew1
        assert stages in (3, 4, 6)
        calls.append(("w1n", nsplit, stages))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, 528)
        o_part.copy_(o.flip(-1) if "w1n" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_w2(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                  nsplit, page, window, sm_scale, stages, pp):   # v10: W2c (Q / P tiles, SS wgmma) through W1's emulator in W2 mode
        import emulate_cuda_w1 as ew1
        assert (stages, pp) in ((6, 0), (6, 1), (8, 1), (4, 1))
        calls.append(("w2", nsplit, stages, pp))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, 528, w2=True)
        o_part.copy_(o.flip(-1) if "w2" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_w3(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                  nsplit, page, window, sm_scale, stages, pp, nwg, rc):   # v12: W3 through W1's emulator in W3 mode
        import emulate_cuda_w1 as ew1
        import cuda_w3
        assert (stages, pp, nwg, rc) in {tuple(v[:4]) for v in cuda_w3.W3_VARIANTS}
        calls.append(("w3", nsplit, stages, pp, nwg, rc))
        npar = nwg // 2
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit * npar, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, 528, npar=npar if nwg == 4 else 1)
        o_part.copy_(o.flip(-1) if "w3" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_w3b(self, *a):   # v13: W3b = W3 with whole-row copies (same tile contents: make_wholerow.py coverage)
        self.attend_w3(*a)
        calls[-1] = ("w3b",) + calls[-1][1:]
        if ("w3b" in corrupt) != ("w3" in corrupt):   # attend_w3 flipped iff "w3" is corrupted: flipped iff "w3b" is
            a[6].copy_(a[6].flip(-1))

    def attend_w3d(self, *a):   # v15: W3d = W3's tile contents (make_w3d.py coverage), W3 emulator
        self.attend_w3(*a)
        calls[-1] = ("w3d",) + calls[-1][1:]
        if ("w3d" in corrupt) != ("w3" in corrupt):
            a[6].copy_(a[6].flip(-1))

    def attend_w3t(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                   nsplit, page, window, sm_scale, stages, pp, nwg, rc):   # v15: W3t = W3's tile contents, W3 emulator
        import emulate_cuda_w1 as ew1
        import cuda_w3t
        assert (stages, pp, nwg, rc) in {tuple(v[:4]) for v in cuda_w3t.W3T_VARIANTS}
        calls.append(("w3t", nsplit, stages, pp, nwg, rc))
        npar = nwg // 2
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit * npar, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, 528, npar=npar if nwg == 4 else 1)
        o_part.copy_(o.flip(-1) if "w3t" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_w3e(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                   nsplit, page, window, sm_scale, stages, pp, nwg, rc):   # v15: W3e = W3b's tiles, W3 emulator
        import emulate_cuda_w1 as ew1
        import cuda_w3e
        assert (stages, pp, nwg, rc) in {tuple(v[:4]) for v in cuda_w3e.W3E_VARIANTS}
        calls.append(("w3e", nsplit, stages, pp, nwg, rc))
        npar = nwg // 2
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit * npar, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, stages, 528, npar=npar if nwg == 4 else 1)
        o_part.copy_(o.flip(-1) if "w3e" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def uw2e(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
             max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs):   # v15: UW2e = UW2's math
        import emulate_cuda_uw2 as emu_uw2
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)) and window > 0
        calls.append(("uw2e", stages, pp, cs))
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages)
        out.copy_((res.flip(-1) if "uw2e" in corrupt else res).view_as(out))

    def attend_w4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                  nsplit, page, window, sm_scale, stages, npair, minb):   # W4: per pair W2c's math on split j (W2 emulator)
        import emulate_cuda_w1 as ew1
        import cuda_w4
        assert (stages, npair, minb) in {tuple(v) for v in cuda_w4.W4_VARIANTS} and nsplit % npair == 0
        calls.append(("w4", nsplit, stages, npair, minb))
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * nsplit, 64)
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 8)
        o, m, l_ = ew1.attend_w1(d, q, nsplit, 4, 528, w2=True)
        o_part.copy_(o.flip(-1) if "w4" in corrupt else o), m_part.copy_(m), l_part.copy_(l_)

    def attend_dec4(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                    nsplit, page, window, sm_scale, warps, stages):
        calls.append(("dec4", nsplit, warps, stages))
        d = as_data(kc, vc, k_scale, pt, seqlens, batch, hq, hkv, page, window, sm_scale, 1)
        o, m, l_ = attend_dec3(d, q, nsplit, warps, stages)
        o_part.copy_(o), m_part.copy_(m), l_part.copy_(l_)

    def attend_dec4p(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                     nsplit, page, window, sm_scale, warps, stages, early):   # v15: D3p = D3's math (make_pdl.py)
        assert early in (0, 1)
        self.attend_dec4(q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                         nsplit, page, window, sm_scale, warps, stages)
        calls[-1] = ("dec4p",) + calls[-1][1:] + (early,)
        if "dec4p" in corrupt:
            o_part.copy_(o_part.flip(-1))

    def attend_w3s(self, q, kc, vc, k_scale, pt, seqlens, o_part, m_part, l_part, batch, hq, hkv, max_pages,
                   g, page, window, sm_scale, stages, pp, nwg, rc):   # v15: glue only (args, buffer sizes); numerics:
        import cuda_w3s                                                 # emulate_w3s.py + the lab; _merge_s mock below
        assert (stages, pp, nwg, rc) in {tuple(v[:4]) for v in cuda_w3s.W3S_VARIANTS} and window == 0 and batch <= 32
        assert pt.shape[1] == max_pages and o_part.shape == (batch * 8 * hq * g * (nwg // 2), 64) and m_part.numel() == o_part.shape[0]
        calls.append(("w3s", g, stages, pp, nwg, rc))

    def ud3(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs):
        calls.append(("ud3", cs, warps, stages))
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 1)
        out.copy_(run_ud(d, v7, cs, 1, warps, stages).view_as(out))

    def ud4(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs):   # v14: UD4 = UD3's math (sim_ud4_merge.py)
        assert (cs, warps, stages) in ((16, 4, 3), (16, 8, 2), (8, 4, 3), (8, 8, 2)) and window > 0
        calls.append(("ud4", cs, warps, stages))
        assert pt.shape[1] == max_pages and out.shape == (batch, 1, hq * 64)
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 1)
        res = run_ud(d, v7, cs, 1, warps, stages)
        out.copy_((res.flip(-1) if "ud4" in corrupt else res).view_as(out))

    def _ud4_like(self, kind, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq,
                  hkv, max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs):   # v15: UD5 / UD4c = UD4's math
        import make_ud4c
        import make_ud8
        import make_ud9
        variants = {"ud9": make_ud9.VARIANTS, "ud8": make_ud8.VARIANTS}.get(kind, make_ud4c.VARIANTS)
        assert (warps, stages, cs) in variants and window > 0 and 16 // cs <= warps, (kind, warps, stages, cs)
        calls.append((kind, cs, warps, stages))
        assert pt.shape[1] == max_pages and out.shape == (batch, 1, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 1)
        res = run_ud(d, v7, cs, 1, warps, stages)
        out.copy_((res.flip(-1) if kind in corrupt else res).view_as(out))

    def ud5(self, *a):
        self._ud4_like("ud5", *a)

    def ud8(self, *a):   # v17: UD8 = UD5's math (make_ud8.py merge-split model)
        self._ud4_like("ud8", *a)

    def ud9(self, *a):   # v18: UD9 = UD8's math (make_ud9.py slot model)
        self._ud4_like("ud9", *a)

    def ud4c(self, *a):
        self._ud4_like("ud4c", *a)

    def ud6(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs, groups, gpart, tickets):
        # v16: UD6 = UD5's math over groups * cs slices (second level: make_ud6.py's model); buffers checked
        import make_ud6
        import torch
        assert (warps, stages, cs) in make_ud6.VARIANTS and groups >= 1 and 16 // cs <= warps
        bp = batch * (hkv // 2)
        assert groups == 1 or gpart.numel() >= bp * groups * 16 * 68
        assert tickets.dtype == torch.int32 and tickets.numel() >= bp * cs and int(tickets.abs().sum()) == 0
        calls.append(("ud6", cs, warps, stages, groups))
        assert pt.shape[1] == max_pages and out.shape == (batch, 1, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 1)
        res = run_ud(d, v7, cs, 1, warps, stages)
        out.copy_((res.flip(-1) if "ud6" in corrupt else res).view_as(out))

    def ud7(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, warps, stages, cs, groups, gpart, tickets):
        # v16: UD7 = UD6 with multi-row owners (make_ud7.py's ownership model); UD3's math through its emulator
        import make_ud7
        import torch
        assert (warps, stages, cs) in make_ud7.VARIANTS and groups >= 1
        bp = batch * (hkv // 2)
        assert groups == 1 or gpart.numel() >= bp * groups * 16 * 68
        assert tickets.dtype == torch.int32 and tickets.numel() >= bp * cs and int(tickets.abs().sum()) == 0
        calls.append(("ud7", cs, warps, stages, groups))
        assert pt.shape[1] == max_pages and out.shape == (batch, 1, hq * 64)
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 1)
        res = run_ud(d, v7, 16, 1, 4, 3)
        out.copy_((res.flip(-1) if "ud7" in corrupt else res).view_as(out))

    def uw2g(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
             max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs, ncl, gpart, tickets):   # v16: UW2's math
        import emulate_cuda_uw2 as emu_uw2
        import torch
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16), (6, 0, 8), (6, 1, 8), (6, 1, 16)) and ncl >= 1
        bp = batch * (hkv // 2)
        assert ncl == 1 or gpart.numel() >= bp * ncl * 128 * 68
        assert tickets.dtype == torch.int32 and tickets.numel() >= bp * cs and int(tickets.abs().sum()) == 0
        calls.append(("uw2g", stages, pp, cs, ncl))
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages)
        out.copy_((res.flip(-1) if "uw2g" in corrupt else res).view_as(out))

    def uw2h(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
             max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs):   # v16: UW2h = UW2's math (make_uw2h.py model)
        import emulate_cuda_uw2 as emu_uw2
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)) and window > 0
        calls.append(("uw2h", stages, pp, cs))
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages)
        out.copy_((res.flip(-1) if "uw2h" in corrupt else res).view_as(out))

    def uw2v(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
             max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs):   # v15: UW2v = UW2's math (make_uw2v.py model)
        import emulate_cuda_uw2 as emu_uw2
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)) and window > 0
        calls.append(("uw2v", stages, pp, cs))
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages)
        out.copy_((res.flip(-1) if "uw2v" in corrupt else res).view_as(out))

    def uw3(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs):   # v15: UW3 through UW2's emulator (npar 2)
        import emulate_cuda_uw2 as emu_uw2
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (6, 0, 16), (4, 1, 16)) and window > 0
        calls.append(("uw3", stages, pp, cs))
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages, npar=2)
        out.copy_((res.flip(-1) if "uw3" in corrupt else res).view_as(out))

    def uv4(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, stages, mt, cs, ncl, gpart, tickets):
        calls.append(("uv4", cs, stages, mt) + ((ncl,) if ncl > 1 else ()))
        assert ncl == 1 or (gpart.numel() >= batch * (hkv // 2) * ncl * 128 * 68 and tickets.numel() >= batch * cs)
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uv4.uv4(d, cs, stages, mt, ncl=ncl)
        out.copy_((res.flip(-1) if "uv4" in corrupt else res).view_as(out))


    def uw2(self, qkv, positions, cache_loc, inv_freq, kc, vc, k_scale, v_scale, pt, seqlens, out, batch, hq, hkv,
            max_pages, page, window, rope, qk_factor, sm_scale, stages, pp, cs):   # v11: UW2 through its emulator
        import emulate_cuda_uw2 as emu_uw2
        assert (stages, pp, cs) in ((4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)) and window > 0
        calls.append(("uw2", stages, pp, cs))
        assert pt.shape[1] == max_pages and out.shape == (batch, 8, hq * 64)
        assert (not rope) or abs(float(inv_freq[1]) / (500000.0 ** (-2 / 64)) - 1) < 1e-6, "inv_freq"
        d = full_data(qkv, positions, cache_loc, kc, vc, k_scale, v_scale, pt, seqlens, batch, hq, hkv, page, window,
                      rope, qk_factor, sm_scale, 8)
        res = emu_uw2.uw2(d, cs, stages)
        out.copy_((res.flip(-1) if "uw2" in corrupt else res).view_as(out))


v7.torch.cuda.is_current_stream_capturing = lambda: False
real_self_check_inputs = v7._self_check_inputs
v7._prologue_q = lambda data: emulate.prologue(data, v7)
if hasattr(v7, "_merge_s"):   # v15: W3s's merge -> the CPU emulator's full path (glue test: routing, args, fallback)
    v7._merge_s = lambda data, o, m, l_, g, npair: (emulate.run(data, v7).flip(-1) if "w3s" in corrupt else emulate.run(data, v7))
if hasattr(v7, "_prologue_q_pdl"):   # v15: the PDL chain's Triton kernels through the same emulators
    v7._prologue_q_pdl = lambda data, early: emulate.prologue(data, v7)
    v7._merge_pdl = lambda data, o, m, l_, ns: emulate.combine(data, o, m, l_, ns, min(64, 1 << max(0, (ns - 1).bit_length())), v7)
v7._merge = lambda data, o, m, l_, ns: emulate.combine(data, o, m, l_, ns, min(64, 1 << max(0, (ns - 1).bit_length())), v7)
v7.run = lambda data, tune=None: emulate.run(data, v7)
v7._kernel_fn_v22 = lambda data: (calls.append(("v22",)), emulate.run(data, v7))[1]
v7.run_pdl = lambda data, tune=None, early=False: (calls.append(("pdl",)), emulate.run(data, v7))[1]
v7._warm_up_v22 = lambda: v7._WARM.append(True)
v7._WARM_DEVICE = "cpu"
class StrictMod:   # v17 lesson: like load_inline's module, expose only the functions of the routed (compiled) kinds
    def __init__(self, kinds):
        self._inner = Mod()
        self._fns = {f for k in kinds for f in v7._KINDS[k][2]}

    def __getattr__(self, name):   # kinds routed after the build (the test's hand-set routes) count as compiled
        live = {f for r in v7.CUDA_ROUTE.values() for f in v7._KINDS.get(r[0], (None, None, []))[2]}
        if name not in self._fns | live:
            raise AttributeError(f"module has no attribute {name!r} (kinds compiled: {sorted(self._fns)})")
        return getattr(self._inner, name)


v7._cuda_build = lambda: v7._CUDA.update(mod=StrictMod({r[0] for r in v7.CUDA_ROUTE.values()}), refused=set())
if hasattr(v7, "run_fa3t"):   # FA3 / Triton do not run on the CPU: the route's glue and fallback are what is tested
    v7.run_fa3t = lambda data, ns=0, pg=None: (calls.append(("fa3t", ns, pg)), emulate.run(data, v7))[1]
v7._self_check_inputs = lambda bclass, length, full, device: ref.generate_input(
    seed=91 + bclass, device="cpu", batch=1, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
    q_len=length, page=1, dist="soft")
ROUTES = dict(v7.CUDA_ROUTE)
UV4 = {c: r for c, r in ROUTES.items() if r[0] in ("uv4", "uw2", "ud4", "uw3", "dec4p", "uw2e", "ud5", "ud4c", "uw2v", "ud6", "uw2g", "ud7", "uw2h", "ud8", "ud9")}
VER5 = {c: r for c, r in ROUTES.items() if r[0] in ("ver5", "ver5t", "w1", "w1n", "w2", "w3", "w3b", "w3d", "w3s", "w3t", "w3e", "w4")}
bad = 0


def run_class(cls, seed, batch=1):
    full, verify, bclass = cls
    case = dict(batch=batch, ctx=700 if full else 2300, heads="tp8", layer="full" if full else "sliding",
                q_len=8 if verify else 1, page=1, dist="soft")
    data = ref.generate_input(seed=seed, device="cpu", **case)
    calls.clear()
    sc = v7.shape_class
    v7.shape_class = lambda cfg, c=cls: c
    out = v7.kernel_fn(data)
    v7.shape_class = sc
    good, msg = ref.check_implementation(data, ref.collect(data, out))
    return good, list(calls), msg


# 1. warm-up keeps every correct route
v7._WARM.clear()
v7._warm_up()
ok = v7.CUDA_ROUTE == ROUTES and all(c in v7.CUDA_ROUTE for c in v7.V7_ROUTES) and \
    all(c in v7.CUDA_ROUTE for c in getattr(v7, "V8_ROUTES", {}))
print(("PASS " if ok else "FAIL ") + f"warm-up keeps all {len(ROUTES)} routes ({len(UV4)} uv4): {sorted(v7.CUDA_ROUTE)}")
bad += not ok
# 2. each uv4 class runs its route (batch 2: two clusters per layer, or 2 x ncl with a second level)
for i, (cls, route) in enumerate(sorted(UV4.items())):
    good, cl, msg = run_class(cls, 600 + i, batch=2)
    ok = good and cl and (cl[0] == route or (route[0] == "dec4p" and cl[0][0] == "dec4p" and cl[0][2:] == route[2:]))
    print(("PASS " if ok else "FAIL ") + f"{cls} -> {route}: {cl} {msg[:70]}")
    bad += not ok
for i, (cls, route) in enumerate(sorted(VER5.items())):    # v8+: V5 verify routes
    good, cl, msg = run_class(cls, 620 + i, batch=2)
    ok = good and cl and cl[0][0] == route[0] and cl[0][2:] == route[2:] and v7._self_check_tol(route) == 0.15
    ok = ok if route[0] not in ("w1", "w1n", "w2", "w3", "w3b", "w3d", "w3s", "w3t", "w3e", "w4") else (good and cl and cl[0][0] == route[0] and cl[0][2:] == route[2:])
    ok = ok if route[0] not in ("w3", "w3b", "w3d", "w3s", "w3t", "w3e", "w4") else (good and cl and cl[0][0] == route[0] and cl[0][2:] == route[1:5])   # w3 routes: no ctas-per-SM field (v15: 6th = cps)
    print(("PASS " if ok else "FAIL ") + f"{cls} -> {route}: {cl} {msg[:70]}")
    bad += not ok
for cls, (ns, pg) in sorted(getattr(v7, "FA3T_ROUTE", {}).items()):   # v8+: fa3t routes
    good, cl, msg = run_class(cls, 630, batch=2)
    ok = good and cl and cl[0] == ("fa3t", ns, pg)
    print(("PASS " if ok else "FAIL ") + f"{cls} -> fa3t {(ns, pg)}: {cl[:1]} {msg[:60]}")
    bad += not ok
# 3. refusals and tolerance
ok = (v7._cuda_ok(dict(num_q_heads=16, num_kv_heads=2, q_len=8, window=0), ("uv4", 8, 3, 1))
      and not v7._cuda_ok(dict(num_q_heads=16, num_kv_heads=2, q_len=1, window=2048), ("uv4", 8, 3, 1))
      and not v7._cuda_ok(dict(num_q_heads=64, num_kv_heads=4, q_len=8, window=2048), ("uv4", 8, 3, 1))
      and v7._cuda_ok(dict(num_q_heads=16, num_kv_heads=2, q_len=8, window=2048), ("uv4", 8, 3, 1))
      and v7._self_check_tol(("uv4", 8, 3, 1)) == 0.15)
print(("PASS " if ok else "FAIL ") + "uv4 takes verify (sliding and full), refuses decode and groups != 8; FP8 tolerance 0.15")
bad += not ok
# full verify through uv4 (route set by hand: the lab file has none there)
v7.CUDA_ROUTE[(True, True, 16)] = ("uv4", 4, 3, 1)
good, cl, msg = run_class((True, True, 16), 640, batch=2)
ok = good and cl and cl[0] == ("uv4", 4, 3, 1)
print(("PASS " if ok else "FAIL ") + f"(True, True, 16) -> uv4 cs4: {cl} {msg[:70]}")
bad += not ok
if (True, True, 16) in ROUTES:
    v7.CUDA_ROUTE[(True, True, 16)] = ROUTES[(True, True, 16)]
else:
    v7.CUDA_ROUTE.pop((True, True, 16))
# 4. a wrong uv4 result is dropped at warm-up; its class falls back to v5's path (PDL chain)
corrupt.update({"uv4", "ver5", "ver5t", "w1", "w1n", "w2", "uw2", "w3", "w3b", "ud4", "w3d", "uw3", "dec4p", "w3s", "w3t", "w3e", "uw2e", "w4", "ud5", "ud4c", "uw2v", "ud6", "uw2g", "ud7", "uw2h", "ud8", "ud9"})
v7.CUDA_ROUTE.clear()
v7.CUDA_ROUTE.update(ROUTES)
v7._WARM.clear()
v7._warm_up()
ok = all(c not in v7.CUDA_ROUTE for c in list(UV4) + list(VER5)) and \
    all(v7.CUDA_ROUTE.get(c) == r for c, r in ROUTES.items() if c not in UV4 and c not in VER5)
print(("PASS " if ok else "FAIL ") + f"corrupted uv4 dropped at warm-up (left: {sorted(v7.CUDA_ROUTE)})")
bad += not ok
corrupt.clear()
for cls in sorted(list(UV4) + list(VER5)):   # v5's path: the PDL chain on sliding verify and Fv1's Triton, SGLang on Fv16/32
    good, cl, msg = run_class(cls, 650)
    ok = good and cl and cl[0][0] in ("pdl", "v22")
    print(("PASS " if ok else "FAIL ") + f"{cls} after drop -> {cl[:1]} {msg[:40]}")
    bad += not ok
# 5. v7's own self-check inputs give every request disjoint cache rows (keys 0 .. n-1 and the new slots)
for bclass, length, full in ((16, 1, False), (32, 8, False), (16, 8, True)):
    d = real_self_check_inputs(bclass, length, full, "cpu")
    n = d["cache_seqlens"].long()
    owned = [set(d["page_table"][b, :int(n[b])].tolist()) for b in range(bclass)]
    disjoint = sum(len(o) for o in owned) == len(set().union(*owned)) and all(len(o) == int(n[b]) for b, o in enumerate(owned))
    slots = all(int(d["cache_loc"][b * length + j]) in owned[b] for b in range(bclass) for j in range(length))
    in_range = int(d["page_table"].max()) < d["k_cache"].shape[0]
    ok = disjoint and slots and in_range
    print(("PASS " if ok else "FAIL ") + f"self-check inputs (batch {bclass}, q_len {length}, full {full}): disjoint rows {disjoint}, "
          f"new slots owned {slots}, rows in range {in_range}")
    bad += not ok
print("test_v7_glue:", "all passed" if not bad else f"{bad} FAILED")
sys.exit(1 if bad else 0)
