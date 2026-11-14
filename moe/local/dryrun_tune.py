"""CPU dry run of submission_m13t.prepare() (the self-tuning prepare): real M13/M14 Python glue and the
tuner, with nvcc and CUDA launches replaced by FP32 stand-ins that check argument lists and shapes; a
broken M13 candidate (NaN: GEMM2 with PDL + start-up prefetch) and an M14 stage that leaves its scratch dirty are planted;
time_graph is a fake timer under which M13/M14 beat the incumbents.

Run: .venv/bin/python moe/local/dryrun_tune.py
"""

import importlib.util
import re
import sys
import types
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_m2  # noqa: E402
import check_v0_lab as base  # noqa: E402


def load_cpu_sub():
    src = (base.MOE / "submission_m13t.py").read_text().replace('device="cuda"', 'device="cpu"').replace('"cuda"', '"cpu"')
    path = Path("/tmp") / "submission_m13t_cpu.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("submission_m13t_cpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    check_m2.install_stubs()
    ext_mod = types.ModuleType("torch.utils.cpp_extension")
    ext_mod.load_inline = None
    sys.modules["torch.utils.cpp_extension"] = ext_mod
    torch.cuda.synchronize = lambda *a, **k: None
    torch.cuda.empty_cache = lambda *a, **k: None
    torch.cuda.get_device_properties = lambda *a, **k: types.SimpleNamespace(multi_processor_count=132)
    lab = load_cpu_sub()
    from nvfp4 import random_weights
    gen = torch.Generator().manual_seed(1)
    w13, w13_sf, gs13 = random_weights(8192, 4096, generator=gen, device="cpu", leading=(32,))
    w2, w2_sf, gs2 = random_weights(4096, 4096, generator=gen, device="cpu", leading=(32,))
    weights = {"w13": w13, "w13_sf": w13_sf, "w13_gs": gs13.expand(32).contiguous(), "w2": w2, "w2_sf": w2_sf,
               "w2_gs": gs2.expand(32).contiguous(),
               "post_norm_w": (0.1 * torch.randn((4096,), generator=gen)).to(torch.bfloat16)}
    class DryRef(lab._Ref):
        def act(self, x, ids):
            flat = ids.reshape(-1).long()
            out = torch.zeros((flat.numel(), 4096))
            for e in [e for e in torch.unique(flat).tolist() if e < 32]:
                p = (flat == e).nonzero().flatten()
                h = x[p // 8].float() @ self.expert(e)[0].T
                out[p] = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
            return out

    ref = DryRef(weights)
    calls = {"m13_g1": 0, "m13_g2": 0, "m11": 0, "post": 0, "v1": 0, "m13_warm": 0, "timed_m13": 0}
    state_box = {}

    def local_pairs(ids):
        flat = ids.reshape(-1).long()
        return flat, [e for e in torch.unique(flat).tolist() if e < 32]

    class Ext:
        def m11f_gemm2_post(self, w2, w2_s, c2, ids, act, y, topk_w, norm_w, out, ctr, n_pairs, local_start, rt_w2,
                            hidden, tokens, eps, tt, nt, depth, ks, grid, pdl):  # E23: G2P = GEMM2 + post-norm
            assert (tt, nt, depth, ks, pdl) in lab.M11F_INSTANCES, (tt, nt, depth, ks, pdl)
            assert ctr.dtype == torch.int32 and ctr.numel() >= 32 + tokens and not bool(ctr.ne(0).any()), "counters"
            assert n_pairs == tokens * 8 and out.shape == (tokens, 4096) and grid % 132 == 0
            calls["m11f"] = calls.get("m11f", 0) + 1
            flat, experts = local_pairs(ids)
            for e in experts:
                p = (flat == e).nonzero().flatten()
                y[p] = (act[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
            return self.m13_post_norm(y, norm_w, topk_w, ids, out, tokens, local_start, eps, pdl)

        def m11b_gemm(self, w, sf, c, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated, tt, nt,
                      depth, ks, grid, pdl, sel):  # E24: M11 on the M13 bank; E26: sel = thr | hot side << 16
            thr, side = sel & 0xFFFF, (sel >> 16) & 1
            dyn = (depth >> 11) & 1  # E42
            if dyn:
                assert (gated, tt, nt, depth & 15, ks) in lab.M11B_DYN_INSTANCES and not (depth & 0x7F0), (gated, tt, nt, depth, ks)
                assert (sel >> 20) & 3 == (0 if gated else 2) + ((sel >> 16) & 1), ("dyn slot", sel, gated)
                calls.setdefault("m11b_dyn", set()).add(gated)
                depth &= ~(1 << 11)
            sel &= ~(3 << 20)
            nu = (depth >> 10) & 1  # E34
            pf, early, occ, depth = (depth >> 9) & 1, (depth >> 8) & 1, (depth >> 4) & 15, depth & 15  # E33 pf << 9
            if nu:
                assert (gated, tt, nt, depth, ks) in lab.M11B_NU_INSTANCES and not (pf or early or occ), (gated, tt, nt, depth, ks)
                assert not (gated and pdl), "no gated PDL rolled build"
                calls.setdefault("m11b_nu", set()).add((gated, tt, ks))
            elif pf:
                assert (gated, tt, nt, depth, ks) in lab.M11B_PF_INSTANCES and pdl == 1 and not early and not occ, \
                    (gated, tt, nt, depth, ks, pdl)
                calls.setdefault("m11b_pf", set()).add(tt)
            elif early:
                assert (gated, tt, nt, depth, ks, occ or 1) in lab.M11B_EARLY_INSTANCES, (gated, tt, nt, depth, ks, occ)
                calls.setdefault("m11b_early", set()).add((gated, occ or 1))
            elif occ:
                assert (gated, tt, nt, depth, ks, occ) in lab.M11B_OCC_INSTANCES and grid // 132 in (occ, 2 * occ), \
                    (gated, tt, nt, depth, ks, occ, grid)
                calls.setdefault("m11b_occ", set()).add((gated, occ))
            else:
                assert (gated, tt, nt, depth, ks) in lab.M11B_INSTANCES, (gated, tt, nt, depth, ks)
            assert pdl in (0, 1), pdl
            assert not (gated and pdl) or (nt == 2 and side == 1) or (dyn and side == 0 and (sel >> 17) & 1), \
                "gated PDL M11b: the 16-pair hot launch, or (E47) the dynamic cold launch after a hot-first launch"
            if (sel >> 18) & 1:  # E36: lad only on a cold GEMM2 with PDL
                assert not gated and pdl == 1 and side == 0 and thr > 0, (sel, gated, pdl)
                calls["lad"] = calls.get("lad", 0) + 1
                sel &= ~(1 << 18)
            if (sel >> 17) & 1:  # E36: late wait only on a hot launch with PDL
                assert pdl == 1 and (side == 1 or dyn), (sel, gated, pdl)
                calls["m11b_late"] = calls.get("m11b_late", 0) + 1
                sel &= ~(1 << 17)
            assert sel >> 17 == 0 and (side == 0 or thr > 0) and grid % 132 == 0 and 1 <= grid // 132 <= 24, (sel, grid)
            bank = state_box["s"]["m13"]
            assert w.data_ptr() in (bank["w13"].data_ptr(), bank["w2"].data_ptr()) and sf.dtype == torch.int16
            assert (gated and rt_total == 512 and a_per_pair == 0) or (not gated and rt_total == 256 and a_per_pair == 1)
            calls["m11b"] = calls.get("m11b", 0) + 1
            flat, experts = local_pairs(ids)
            cnt = {e: int((flat == e).sum()) for e in experts}
            if thr:  # as the M14 mock: this side only, and cold + hot cover every local expert once
                mine = [e for e in experts if (cnt[e] > thr if side else cnt[e] <= thr)]
                cover = calls.setdefault("cover", {})
                key = (out.data_ptr(), bool(gated))
                if key not in cover:  # E47: either side may come first (hot-first order)
                    cover[key] = (side, set(mine))
                else:
                    s0, seen = cover.pop(key)
                    assert s0 != side, "one side ran twice on one output"
                    assert not (seen & set(mine)) and seen | set(mine) == set(experts), (seen, mine, experts)
                    calls["hot"] = calls.get("hot", 0) + 1
                    if s0 == 1:
                        calls["hot_first"] = calls.get("hot_first", 0) + 1
                calls["m11b_split"] = calls.get("m11b_split", 0) + 1
                calls.setdefault("m11b_sides", set()).add(side)
                experts = mine
            # the real kernel lists at most MAXP = M11_MAX_TOKENS pairs per expert and drops the rest
            assert all(cnt[e] <= lab.M11_MAX_TOKENS for e in experts), ("M11b would truncate", sel, cnt)
            if gated:
                act = ref.act(a, ids.view(-1, 8))
                rows = torch.isin(flat, torch.tensor(experts, dtype=flat.dtype)).nonzero().flatten()
                out[rows] = act[rows].to(torch.bfloat16)
            else:
                for e in experts:
                    p = (flat == e).nonzero().flatten()
                    out[p] = (a[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
            return out

        def m11b_attrs(self, gated, tt, nt, depth, ks):  # E27 log line (E30: early builds, depth bit 8)
            mb, early = (depth >> 4) & 15, (depth >> 8) & 1
            if early:
                assert (gated, tt, nt, depth & 15, ks, mb or 1) in lab.M11B_EARLY_INSTANCES, (gated, tt, nt, depth, ks)
            else:
                assert ((gated, tt, nt, depth & 15, ks, mb) in lab.M11B_OCC_INSTANCES if mb
                        else (gated, tt, nt, depth, ks) in lab.M11B_INSTANCES), (gated, tt, nt, depth, ks)
            calls["m11b_attrs"] = calls.get("m11b_attrs", 0) + 1
            return torch.tensor([96 if mb == 5 else 80 if mb == 6 else 128, 0, 8448, mb or 4])

        def m11b_recon(self, w, sf, c, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated, tt, nt,
                       depth, ks, grid, sel, mode):  # E29: timing-only recon modes
            assert (gated, tt, nt, depth, ks) in lab.M11B_RECON_SHAPES and mode in (1, 2, 3), (gated, tt, nt, depth, ks, mode)
            assert (gated, tt, nt, depth, ks) in lab.M11B_INSTANCES and sel in (0, 16) and grid % 132 == 0
            bank = state_box["s"]["m13"]
            assert w.data_ptr() in (bank["w13"].data_ptr(), bank["w2"].data_ptr()) and out.shape == (n_pairs, out_cols)
            calls.setdefault("m11b_recon", set()).add((gated, mode))
            return out

        def m11_sr_setup(self):  # E22b: raises the SR kernels' dynamic limits in prepare
            calls["m11_sr_setup"] = calls.get("m11_sr_setup", 0) + 1
            return torch.zeros(1)

        def m11_attrs(self, gated, tt, nt, depth, ks):  # E22 log line
            calls["m11_attrs"] = calls.get("m11_attrs", 0) + 1
            return torch.tensor([128 if depth & 15 == 4 else 80, 0, 8000])

        def m13_attrs(self, gated, tt, nt):
            return torch.tensor([128 if nt == 1 else 168, 0, 7000, 4 if nt == 1 else 3])

        def m14_setup(self):
            return torch.tensor([[g, tt, nt, pa, 160, 0, 21000, (pa + 1) * 8 * nt * 128, 3]
                                 for g, tt, nt, pa in lab.M14_INSTANCES])

        def m13_recon(self, *args):
            assert len(args) == 18 and args[13] == 1 and args[17] in (1, 2, 3) and args[16] == 4 * 132, args[13:]
            calls["recon"] = calls.get("recon", 0) + 1
            return args[5]

        def m14_recon(self, *args):
            assert len(args) == 19 and args[13:16] == (1, 2, 2) and args[17] == 4 and args[18] in (1, 2, 3), args[13:]
            assert args[16] == 3 * 132 and args[8] == args[3].numel() <= 8 * lab.M14_MAX_TOKENS
            calls["recon14"] = calls.get("recon14", 0) + 1
            return args[5]

        def m13_stream(self, base, experts, len16, sink, unroll, grid):
            assert base.dtype == torch.int32 and experts.dtype == torch.int32 and len16 == base[0].numel() // 4
            assert unroll in (4, 8, 16) and grid % 132 == 0
            calls["stream"] = calls.get("stream", 0) + 1
            return sink

        def m14_gemm(self, *args):
            return self.m13_gemm(*args, kind="m14")

        def m19_gemm(self, *args):
            calls["m19"] = calls.get("m19", 0) + 1
            return self.m13_gemm(*args, kind="m19")

        def m19_setup(self):
            return torch.tensor([[g, tt, nt, pa | (mb << 4), 128 if mb == 4 else 160, 0, 21000,
                                  (pa + 1) * 8 * nt * 128 + 4 * 5 * tt * 512, mb] for g, tt, nt, pa, mb in lab.M19_INSTANCES])

        def m15_setup(self):
            return torch.tensor([[g, tt, nt, pa, 150, 0, 21000, 80000, 2] for g, tt, nt, r, pa in lab.M15_INSTANCES])

        def m16_setup(self):
            return torch.tensor([[g, tt, nt, pa | (db << 4), 168, 0, 21000, 1024 + (pa + 1 + db) * 8 * nt * 128, 2]
                                 for g, tt, nt, pa, db in lab.M16_INSTANCES])

        def m16_recon(self, *args):
            assert len(args) == 13 and args[12] in (1, 2) and args[11] == 2 * 132, args[8:]
            assert args[0].data_ptr() == state_box["s"]["m13"]["w13n"].data_ptr()
            calls["recon16"] = calls.get("recon16", 0) + 1
            return args[5]

        def m16_gemm(self, *args):
            calls["m16"] = calls.get("m16", 0) + 1
            return self.m13_gemm(*args, kind="m16")

        def m18_setup(self):
            return torch.tensor([[g, tt, nt, pa, 160, 0, 21000, 70000, 3] for g, tt, nt, pa in lab.M18_INSTANCES])

        def m18_recon(self, *args):
            assert len(args) == 13 and args[12] in (1, 2, 3, 4) and args[11] == 3 * 132, args[8:]
            assert args[0].data_ptr() == state_box["s"]["m13"]["w13r"].data_ptr()
            calls["recon18"] = calls.get("recon18", 0) + 1
            if args[12] == 2:  # planted: a refused launch (E17's failure); only this number may be lost
                raise RuntimeError("m13 launch failed: invalid argument (planted)")
            return args[5]

        def m14_recon(self, *args):
            assert len(args) == 19 and args[13:16] == (1, 2, 2) and args[17] == 4 and args[18] == 1, args[13:]
            calls["recon14"] = calls.get("recon14", 0) + 1
            return args[5]

        def m18_gemm(self, *args):
            calls["m18"] = calls.get("m18", 0) + 1
            return self.m13_gemm(*args, kind="m18")

        def m17_setup(self):
            return torch.tensor([[g, tt, nt, pa | (db << 4), 160, 0, 21000, 80000, 2] for g, tt, nt, r, pa, db in lab.M17_INSTANCES])

        def m17_gemm(self, *args):
            calls["m17"] = calls.get("m17", 0) + 1
            return self.m13_gemm(*args, kind="m17")

        def m15_gemm(self, *args):
            calls["m15"] = calls.get("m15", 0) + 1
            return self.m13_gemm(*args, kind="m15")

        def m13_gemm(self, *args, kind="m13"):
            assert len(args) == (24 if kind == "m13" else 19), (kind, len(args))
            (w, sf, c, ids, a, out, scr, ctr, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated, tt, nt,
             grid, pdl, pf) = args[:19]
            if kind == "m13":  # next-stage prefetch: GEMM1 only, pointing at GEMM2's arrays in M13's bank
                nxw, nxs, nxwb, nxsb, nxwarps = args[19:]
                bank = state_box["s"]["m13"]
                if (pf >> 24) & 7:
                    calls["nx"] = calls.get("nx", 0) + 1
                    assert gated == 1 and (pf >> 24) in (2, 4), pf
                    assert (nxw, nxs) == (bank["w2"].data_ptr(), bank["w2_s"].data_ptr()), "wrong next-stage arrays"
                    assert nxwb == bank["w2"][0].numel() * 4 and nxsb == bank["w2_s"][0].numel() * 2, (nxwb, nxsb)
                    assert nxwarps > 0 and nxwarps % (4 * 132) == 0, nxwarps
                else:
                    assert (nxw, nxs, nxwb, nxsb, nxwarps) == (0, 0, 0, 0, 0), args[19:]
                assert (pf & 0xFF) in (0, 255), pf
                if pf & 0xFF:
                    calls["npre"] = calls.get("npre", 0) + 1
            else:
                thr, side = (pf >> 16) & 0xFF, (pf >> 30) & 1  # E18 hot/cold split (M14/M18/M19 only)
                if (pf >> 29) & 1:  # E35/E36: late wait only on M19's hot launches with PDL
                    assert kind in ("m19", "m18") and side == 1 and pdl == 1, (kind, gated, side, pdl)
                    calls["late" if gated else "late2"] = calls.get("late" if gated else "late2", 0) + 1
                    if kind == "m18":
                        calls["late_m18"] = calls.get("late_m18", 0) + 1
                    pf &= ~(1 << 29)
                if (pf >> 15) & 1:  # E36: lad only on M19's cold GEMM2 with PDL
                    assert kind in ("m19", "m18") and not gated and pdl == 1 and thr > 0, (kind, gated, side, pdl)
                    if side == 1:
                        calls["hot_first2"] = calls.get("hot_first2", 0) + 1
                    calls["lad"] = calls.get("lad", 0) + 1
                    pf &= ~(1 << 15)
                if kind in ("m14", "m18", "m19"):
                    pf &= ~((0xFF << 16) | (1 << 30))
                    assert side == 0 or thr > 0, (kind, thr, side)
                else:
                    thr = side = 0
                pa, db = (pf >> 24) & 15, (pf >> 28) & 1
                assert (pf & 0xFF) == 0 and (pf >> 29) == 0 and (db == 0 or kind in ("m16", "m17", "m19")), (kind, pf)
                tt2 = 2 if gated else tt
                if kind == "m14":  # the activation lead names a compiled instance
                    assert (gated, tt2, nt, pa) in lab.M14_INSTANCES, (gated, tt, nt, pa)
                    calls.setdefault("m14_p", set()).add(pa)
                elif kind == "m16":
                    assert (gated, tt2, nt, pa, db) in lab.M16_INSTANCES, (gated, tt, nt, pa, db)
                    calls.setdefault("m16_db", set()).add(db)
                elif kind == "m18":
                    assert (gated, tt2, nt, pa) in lab.M18_INSTANCES and db == 0, (gated, tt, nt, pa, db)
                elif kind == "m19":  # pf bit 28 = MB 4
                    assert (gated, tt2, nt, pa, 4 if db else 3) in lab.M19_INSTANCES, (gated, tt, nt, pa, db)
                    calls.setdefault("m19_mb", set()).add(4 if db else 3)
                elif kind == "m15":
                    assert (gated, tt2, nt, pa) in {(i[0], i[1], i[2], i[4]) for i in lab.M15_INSTANCES}, (gated, tt, nt, pa)
                else:
                    assert (gated, tt2, nt, pa, db) in {(i[0], i[1], i[2], i[4], i[5]) for i in lab.M17_INSTANCES}, \
                        (gated, tt, nt, pa, db)
            pf &= ~(1 << 28)
            npre = pf & 0xFF
            pf &= 0x00FFFF00  # checked above; the asserts below see the older bits only
            assert w.dtype == torch.int32 and w.shape[2:] == (64, 32, 4)
            if kind == "m18":  # row-chunk bank: M13's shapes
                assert sf.dtype == torch.int16 and sf.shape[2:] == (16, 32, 4), (sf.dtype, sf.shape)
                bank = state_box["s"]["m13"]
                assert w.data_ptr() in (bank["w13r"].data_ptr(), bank["w2r"].data_ptr()), "M18 must read its own bank"
            elif kind in ("m16", "m17"):  # natural-K bank: per-chunk uint2 scales for the 8 row pairs
                assert sf.dtype == torch.int32 and sf.shape[2:] == (64, 8, 2), (sf.dtype, sf.shape)
                bank = state_box["s"]["m13"]
                assert w.data_ptr() in (bank["w13n"].data_ptr(), bank["w2n"].data_ptr()), "M16 must read its own bank"
            else:
                assert sf.dtype == torch.int16 and sf.shape[2:] == (16, 32, 4)
            limit = lab.M13_MAX_TOKENS if kind == "m13" else lab.M14_MAX_TOKENS
            assert ids.dtype == torch.int32 and ids.dim() == 1 and n_pairs == ids.numel() <= 8 * limit, (kind, n_pairs)
            assert out.shape == (n_pairs, out_cols) and out.dtype == torch.bfloat16
            assert scr.shape[1] == 16 * rt_total and ctr.dtype == torch.int32
            occ = 5 if (gated, tt, nt) in ((0, 1, 1), (0, 1, 4)) else 4
            assert grid % 132 == 0 and 1 <= grid // 132 <= occ, (grid, occ)
            if kind == "m13":
                thr = side = 0
            assert pdl in (0, 1) and pf >= 0 and (gated == 0 or pdl == 0 or side == 1)
            assert ((pf >> 8) & 0xFF) in (0, 2, 4) and (pf >> 16) in ((0, 2, 4) if kind == "m17" else (0, 4)), (kind, pf)
            assert kind in ("m15", "m17") or (pf >> 16) == 0, (kind, pf)
            flat, experts = local_pairs(ids)
            if thr:  # only this launch's side of the split; the two launches must cover every local expert once
                cnt = {e: min(int((flat == e).sum()), lab.M14_MAX_TOKENS) for e in experts}
                mine = [e for e in experts if (cnt[e] > thr if side else cnt[e] <= thr)]
                cover = calls.setdefault("cover", {})
                key = (out.data_ptr(), bool(gated))
                if key not in cover:  # E47: either side may come first (hot-first order)
                    cover[key] = (side, set(mine))
                else:
                    s0, seen = cover.pop(key)
                    assert s0 != side, "one side ran twice on one output"
                    assert not (seen & set(mine)) and seen | set(mine) == set(experts), (seen, mine, experts)
                    calls["hot"] = calls.get("hot", 0) + 1
                    if s0 == 1:
                        calls["hot_first"] = calls.get("hot_first", 0) + 1
                experts = mine
            if gated:
                assert a_per_pair == 0 and rt_total == 512
                calls["m13_g1"] += 1
                if not thr:
                    out.copy_(torch.zeros_like(out))
                act = ref.act(a, ids.view(-1, 8))
                rows = torch.isin(flat, torch.tensor(experts, dtype=flat.dtype)).nonzero().flatten()
                out[rows] = act[rows].to(torch.bfloat16)
            else:
                assert a_per_pair == 1 and rt_total == 256
                calls["m13_g2"] += 1
                for e in experts:
                    p = (flat == e).nonzero().flatten()
                    out[p] = (a[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
                if pdl == 1 and kind == "m13":
                    out.fill_(float("nan"))  # planted: a broken candidate (M13 GEMM2 with PDL)
                    calls["bad"] = calls.get("bad", 0) + 1
                if kind == "m14" and nt == 4 and grid == 2 * 132:
                    scr[0, 0] = 1.0  # planted: an M14 config that leaves its scratch dirty
            return out

        def m11p_gemm(self, *args):  # M11's GEMM2 with PDL
            assert len(args) == 17 and args[11] == 0, args[6:]
            tt, nt, depth, ks = args[12:16]
            insts = lab.M13_M11_CR_INSTANCES if depth & 16 else lab.M13_M11_SR_INSTANCES if depth & 32 else lab.M13_M11_INSTANCES
            assert (0, tt, nt, depth & 15, ks) in insts and not (depth & 16 and depth & 32), args[12:16]
            if depth & 32:
                calls["m11sr"] = calls.get("m11sr", 0) + 1
            if depth & 16:
                calls["m11cr"] = calls.get("m11cr", 0) + 1
            calls["m11p"] = calls.get("m11p", 0) + 1
            return self.m11_gemm(*args)

        def m11_gemm(self, *args):
            assert len(args) == 17, len(args)
            w, sf, c, ids, a, out, n_pairs, local_start, rt_total, a_per_pair, out_cols, gated = args[:12]
            tt, nt, depth, ks = args[12:16]
            if depth & 16:  # E21: the continuous-ring kernel must name a compiled CR instance
                assert (gated, tt, nt, depth & 15, ks) in lab.M13_M11_CR_INSTANCES, args[11:16]
                calls["m11cr"] = calls.get("m11cr", 0) + 1
            if depth & 32:  # E22: the shared-memory-ring kernel
                assert (gated, tt, nt, depth & 15, ks) in lab.M13_M11_SR_INSTANCES and not depth & 16, args[11:16]
                assert calls.get("m11_sr_setup") == 1, "SR launched before m11_sr_setup"
                calls["m11sr"] = calls.get("m11sr", 0) + 1
                if gated and args[16] == 5 * 132:  # planted (E22b): a refused launch; only this candidate may be lost
                    calls["sr_refused"] = calls.get("sr_refused", 0) + 1
                    raise RuntimeError("m11 sr launch failed: invalid argument (planted)")
            calls["m11"] += 1
            flat, experts = local_pairs(ids)
            if gated:
                act = ref.act(a, ids.view(-1, 8))
                rows = (flat < 32).nonzero().flatten()
                out[rows] = act[rows].to(torch.bfloat16)
            else:
                for e in experts:
                    p = (flat == e).nonzero().flatten()
                    out[p] = (a[p].float() @ ref.expert(e)[1].T).to(torch.bfloat16)
            return out

        def m13_post_norm(self, y, nw, tw, ids, out, tokens, local_start, eps, pdl):
            calls["post"] += 1
            assert out.shape == (tokens, 4096) and pdl in (0, 1) and nw.dtype == torch.bfloat16
            flat = ids.reshape(-1).long()
            res = torch.zeros((tokens, 4096))
            for p in range(flat.numel()):
                e = int(flat[p]) - local_start
                if 0 <= e < 32:
                    v = y[p].float() * nw.float()
                    v = v * torch.rsqrt(v.pow(2).mean() + eps)
                    res[p // 8] += tw.reshape(-1)[p] * v
            out.copy_(res.to(torch.bfloat16))
            return out

    def fake_bank(weights, start, count):
        bank = {"start": start, "count": count, "inter": 4096, "hidden": 4096,
                "c13": torch.ones(count), "c2": torch.ones(count), "rt_w13": 512, "rt_w2": 256}
        for name in ("w13", "w2"):
            bank[name] = torch.zeros((count, 2, 64, 32, 4), dtype=torch.int32)
            bank[name + "_s"] = torch.zeros((count, 2, 64, 32), dtype=torch.int16)
        return bank

    def v1_kernel_fn(data, state):
        calls["v1"] += 1
        return ref.full(data["hidden_states"], data["topk_ids"], data["topk_weights"]).to(torch.bfloat16)

    def m11_post_norm(y, ids, tw, state, tokens):
        out = torch.empty((tokens, 4096), dtype=torch.bfloat16)
        return Ext().m13_post_norm(y, state["norm_w"], tw, ids, out, tokens, 0, lab.POST_NORM_EPS, 0)

    real_m13_warm = lab.m13_warm

    def m13_warm(state, cfg, tokens=8):
        calls["m13_warm"] += 1
        return real_m13_warm(state, cfg, tokens)

    def fake_time(self, fn, reps=10):
        before = calls["m13_g1"] + calls["m13_g2"]
        bad = calls.get("bad", 0)
        hot = calls.get("hot", 0)
        sp11 = calls.get("m11b_split", 0)
        fn()
        used = calls["m13_g1"] + calls["m13_g2"] > before
        calls["timed_m13"] += used
        if calls.get("bad", 0) > bad:
            return 10.0  # the broken candidate looks fastest: the FP32 check must reject it
        if calls.get("m11b_split", 0) > sp11:
            return 14.0  # E26: M11b splits win where offered, so kernel_fn and validation run them
        if calls.get("hot", 0) > hot:
            return 15.0  # E18: split candidates win from 128 tokens, so kernel_fn and validation run the split path
        return (20.0 if used else 40.0) + (hash(fn.__code__.co_firstlineno) % 3)

    def v1_prepare(w):
        return {"norm_w": (w["post_norm_w"].float() + 1.0).to(torch.bfloat16)}

    lab.m11_post_norm = m11_post_norm
    lab.m13_build = lambda: Ext()
    lab.m11_bank = fake_bank
    lab._v1_kernel_fn, lab._v1_prepare = v1_kernel_fn, v1_prepare
    lab.m13_warm = m13_warm
    lab._Tuner.time_graph = fake_time
    # the CPU stand-ins are slow and timing-dependent; this run checks the logic, not the GPU budget
    lab.TUNE_BUDGET_S, lab.TUNE_PREPARE_LIMIT_S = 10.0**6, 10.0**6
    lab.TUNE_PRUNE = False  # E46: the dry run keeps every family (pruning is checked on its own below)
    m11b_base = dict(kind="m11b", nt=1, d=4, ks=2, c=8)
    assert lab._pruned(dict(s1=dict(m11b_base, early=1), s2=dict(kind="m11b", tt=2)), 8)
    assert lab._pruned(dict(s1=dict(m11b_base, thr=8, hot=dict(kind="m11b")), s2=dict(kind="m11b", tt=2)), 32)
    assert not lab._pruned(dict(s1=dict(m11b_base, dyn=1, c=4), s2=dict(kind="m11b", tt=2, dyn=1)), 32)
    assert not lab._pruned(dict(s1=dict(m11b_base, thr=16, hot=dict(kind="m19", lw=1)), s2=dict(kind="m11b", tt=2, thr=16,
                                hot=dict(kind="m18", lw=1))), 128)
    assert not lab._pruned(dict(s1=dict(kind="m11b", nt=1, d=4, ks=4, c=8), s2=dict(kind="m11b", tt=1, nt=1)), 1)
    real_setup = lab.m13_setup

    def m13_setup(state, w):
        real_setup(state, w)
        state_box["s"] = state  # the mock checks the next-stage pointers against this state's bank

    lab.m13_setup = m13_setup
    real_recon = lab._m11b_recon

    def m11b_recon(tuner, state):  # E29: bs128's cold-only timing launches have no hot partner (not a split)
        real_recon(tuner, state)
        left = calls.pop("cover", {})
        assert len(left) == 2 and all(k[0] for k in [(bool(v),) for v in left.values()]), left
        calls["recon_cold"] = calls.get("recon_cold", 0) + 1

    lab._m11b_recon = m11b_recon

    def fake_m16_bank(weights, bank, start, count):  # the real repack is checked in check_m16.py
        for name, rt in (("w13", 2), ("w2", 2)):
            bank[name + "n"] = torch.zeros((count, rt, 64, 32, 4), dtype=torch.int32)
            bank[name + "n_s"] = torch.zeros((count, rt, 64, 8, 2), dtype=torch.int32)
    lab.m16_bank = fake_m16_bank

    def fake_m18_bank(weights, bank, start, count):  # the real repack is checked in check_m18.py
        for name in ("w13", "w2"):
            bank[name + "r"] = torch.zeros((count, 2, 64, 32, 4), dtype=torch.int32)
            bank[name + "r_s"] = torch.zeros((count, 2, 16, 32, 4), dtype=torch.int16)
    lab.m18_bank = fake_m18_bank
    lab.m13_probe_build = lambda: calls.__setitem__("probe", calls.get("probe", 0) + 1)  # needs nvcc
    import contextlib
    import io
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        state = lab.prepare(weights)
    text = log.getvalue()
    sys.stdout.write(text[-1500:] if "--verbose" in sys.argv else "")
    for bs in (128, 256):  # the planted mode-2 failure is logged, the other skeleton numbers still come out
        assert f"skeleton bs{bs} m18 mode 2 FAILED RuntimeError" in text, text[-3000:]
        line = next(l for l in text.splitlines() if l.startswith(f"[tune] skeleton bs{bs} GEMM1"))
        assert "decode+store nan" in line and "loads+stages+barrier nan" not in line and "M14 n2c3 nan" not in line, line
    assert "tune FAILED" not in text and "bw recon FAILED" not in text, text[-3000:]
    assert calls.get("sr_refused", 0) > 0 and "candidate(s) failed and were skipped" in text, "planted SR refusal unexercised"
    assert not re.search(r"class \[\d+,\d+\) FAILED", text), [l for l in text.splitlines() if " FAILED" in l][:3]
    assert "m11 attrs: g2.1d4k4 128r" in text and "m11 attrs FAILED" not in text and "gS2.1d4k4" in text, \
        [l for l in text.splitlines() if "m11 attrs" in l]
    cfgs = state["m13_configs"]
    assert [c[0] for c in cfgs] == [5, 9, 17, 33, 65, 192, 321], cfgs
    for below, cfg in cfgs:
        assert cfg is not None, (below, cfg)
        assert not (cfg["s2"]["kind"] == "m13" and cfg["s2"].get("pdl") == 1), "broken candidate chosen"
        assert not (cfg["s1"]["kind"] == "m14" and cfg["s1"]["nt"] == 4 and cfg["s1"]["c"] == 2), "dirty M14 config chosen"
        assert cfg["s1"].get("kind") in (None, "m13", "m15", "m11b") or (cfg["s1"].get("kind") == "m19" and below > 9) or below > 65, (below, cfg)
    assert calls.get("bad", 0) > 0, "the broken candidate was never exercised"
    for below, cfg in cfgs:  # E18/E26: the split wins (fake timer) and passes validation only with full, disjoint coverage
        split = bool(cfg["s1"].get("thr"))
        assert split == (below in (17, 33, 192, 321)) == bool(cfg["s2"].get("thr")), (below, cfg)
        if split:  # E26: M11b splits win where offered: both sides at 9-32 tokens, the cold side from 65
            assert cfg["s1"]["kind"] == cfg["s2"]["kind"] == "m11b", (below, cfg)
            hk = {cfg["s1"]["hot"]["kind"], cfg["s2"]["hot"]["kind"]}
            assert hk == {"m11b"} if below <= 33 else not (hk & {"m11b"}), (below, cfg)
            assert cfg["s1"]["thr"] == (8 if below <= 33 else cfg["s1"]["thr"]) <= lab.M11_MAX_TOKENS, (below, cfg)
    assert calls.get("m11b_sides") == {0, 1}, calls.get("m11b_sides")
    assert calls.get("m11b_pf") == {1, 2}, calls.get("m11b_pf")
    assert calls.get("m11b_dyn") == {0, 1}, calls.get("m11b_dyn")  # E42 ran
    assert calls.get("hot_first", 0) > 0, "E47 hot-first candidates never ran"
    assert calls.get("hot_first2", 0) > 0, "E48 GEMM2 hot-first candidates never ran"
    assert calls.get("late", 0) > 0, "E35 late-wait candidates never ran"
    assert calls.get("late2", 0) > 0 and calls.get("lad", 0) > 0, "E36 GEMM2 late-wait candidates never ran"
    assert calls.get("m11b_late", 0) > 0, "E36 M11b late-wait candidates never ran"
    assert calls.get("late_m18", 0) > 0, "E37 M18 late-wait candidates never ran"
    assert calls.get("m11b_nu") == {(1, 2, 4), (1, 2, 2), (0, 1, 4), (0, 2, 4)}, calls.get("m11b_nu")  # E34 ran  # E33 prefetch builds ran (bs1 t1, others t2)
    assert calls.get("m11b_recon") == {(g, m) for g in (0, 1) for m in (1, 2, 3)}, calls.get("m11b_recon")
    recon = [l for l in text.splitlines() if l.startswith("[tune] m11b recon")]
    assert len(recon) == 4 and all("FAILED" not in l and "no-decode" in l for l in recon), recon
    assert calls.get("m11b_occ") == {(g, m) for g in (0, 1) for m in (4, 5, 6)}, calls.get("m11b_occ")  # E27/E28 candidates ran
    assert calls.get("m11b_attrs") == len(lab._m11b_all()), calls.get("m11b_attrs")
    assert calls.get("m11b_early") == {(1, 1), (1, 4)}, calls.get("m11b_early")  # E30/E31 (GEMM1 only) ran
    assert "m11b attrs: g2.1d4k4 128r 8K 4c" in text and "o5 96r 8K 5c" in text and "m11b attrs FAILED" not in text, \
        [l for l in text.splitlines() if "m11b attrs" in l]
    assert calls.get("hot", 0) > 0 and not calls.get("cover"), calls.get("cover")
    assert calls.get("m16_db") is None, calls.get("m16_db")
    assert calls.get("probe", 0) == 0, "the probe build is off since E15"
    assert calls.get("m11p", 0) > 0, "the M11 PDL candidates were never run"
    assert calls.get("m11cr", 0) > 0, "the M11 continuous-ring candidates (E21) were never run"
    assert calls.get("m11sr", 0) > 0, "the M11 shared-memory-ring candidates (E22) were never run"
    assert calls.get("m11f", 0) > 0, "the G2P candidates (E23) were never run"
    assert calls.get("m11b", 0) > 0, "the M11b candidates (E24) were never run"
    assert calls.get("m14_p") == {2, 4, 6}, calls.get("m14_p")
    assert calls.get("m15", 0) > 0, "M15 was never searched"
    assert calls.get("m18", 0) > 0, "M18 was never searched"
    assert calls.get("m19", 0) > 0 and calls.get("m19_mb") == {3, 4}, ("M19 (both MB) must be searched", calls.get("m19_mb"))
    assert calls.get("m16", 0) == 0 and calls.get("m17", 0) == 0, "M16/M17 left the search in E16"
    assert calls.get("stream", 0) > 0 and calls.get("recon18", 0) == 8 and calls.get("recon14", 0) == 2, calls
    assert calls["m13_warm"] >= 7, calls
    for bs in (8, 128):
        x, ids, w = lab._case(bs, 5)
        data = {"hidden_states": x, "topk_ids": ids, "topk_weights": w, "config": {"bs": bs, "local_start": 0, "local_count": 32}}
        out = lab.kernel_fn(data, state)
        err, row = lab._errors(out, ref.full(x, ids, w))
        assert out.shape == (bs, 4096) and err < 1e-2, (bs, err)
    calls["m14_p"] = sorted(calls["m14_p"])  # (after the kernel_fn calls above, which may add to it)
    calls.pop("m16_db", None)
    print(f"ok: submission_m13t.prepare dry run on the CPU; calls {calls}; classes {[c[0] for c in cfgs]}; "
          f"kernel_fn bs8 (M13) and bs128 (M14) err < 1e-2")


if __name__ == "__main__":
    main()
