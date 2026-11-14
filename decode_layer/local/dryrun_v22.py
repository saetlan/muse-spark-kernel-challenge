"""CPU dry run of v22 (parts/moe36.py + parts/v22.py: MoE E36's path, with its hot / cold split at 128 / 256 tokens) and
the DE25 lab.

The moe36 extension is an argument-checking stand-in (load_inline name "moe36_ext"): m11b_gemm with E36's 19 arguments
(sel = thr | hot side << 16 | late wait << 17 | lad << 18), m19_gemm with 19 (pf: lad bit 15, P << 24, MB4 bit 28,
late wait bit 29, hot bit 30, thr << 16),
m13_post_norm; it returns zeros like dryrun_v17's (Marlin's and M11's stand-ins return zeros too). Each launch is
recorded as a tag, so a run's sequence shows which config ran. The judge's weights have normal E4M3 scales (0x38 here).

lab modes (fake timings: a layer call 500 us, 20 us less when a split ran ("gain"), 2 us less ("nogain")):
gain      both splits installed (4% > 2%): V17_CHOICE printed = state = v20b's + {192: 's128', 321: 's256'}; kernel_fn
          runs the s128 launch sequence at 128 tokens, s256's at 256, M11b at 1-16, M19 at 32 / 65.
nogain    0.4%: v20b's choice kept (plain M19 at 128 / 256).
badscale  a zero E4M3 scale: the E35 path is OFF; the file's V17_CHOICE is printed unchanged.
clean     submission_v22.py with V17_CHOICE from DRYRUN_V22_LOG (or the file's): installed, kernel_fn follows.
Also: no E25 function is reached (every v17 hook is v22's), no lab failures, real routing restored.

Run: .venv/bin/python decode_layer/local/dryrun_v22.py [gain | nogain | badscale | clean]
"""

import ast
import os
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
CLASSES = ((5, 1), (9, 5), (17, 9), (33, 17), (192, 65), (321, 192))
V20B = {5: "m11b", 9: "m11b", 17: "m11b", 33: "m19", 192: "m19", 321: "m19"}
SEQ = {  # the launch tags each config's run records, in order
    "m11b": ["m11b g1 sel0 pdl0", "m11b g2 sel0 pdl1", "post"],
    "m19": ["m19 g1 cold", "m19 g2 cold pdl1", "post"],
    "s128": ["m11b g1 sel16 pdl0", "m19 g1 hot thr16 pdl1", "m11b g2 sel16 lad pdl1", "m19 g2 hot thr16 lw pdl1", "post"],
    "s256": ["m19 g1 cold thr16", "m19 g1 hot thr16 lw pdl1", "m19 g2 cold thr16 lad pdl1", "m19 g2 hot thr16 lw pdl1", "post"],
}
LOG = []


class E35Ext:
    def m19_setup(self):
        return torch.zeros(4, dtype=torch.int32)

    def m19_gemm(self, *a):
        assert len(a) == 19, len(a)
        out, n_pairs, ids, nt, pdl, pf, cols, gated, tt = a[5], a[8], a[3], a[15], a[17], a[18], a[12], a[13], a[14]
        assert ids.dtype == torch.int32 and n_pairs == ids.numel() and a[9] == 0 and out.shape == (n_pairs, cols)
        hot, lw, thr, p, mb4 = (pf >> 30) & 1, (pf >> 29) & 1, (pf >> 16) & 0xFF, (pf >> 24) & 0xF, (pf >> 28) & 1
        lad = (pf >> 15) & 1
        inst = (gated, tt, nt, p, 4 if mb4 else 3)
        assert inst in ((1, 2, 2, 2, 4), (0, 2, 2, 4, 4), (1, 2, 4, 2, 3), (0, 2, 4, 2, 4)), inst
        assert not lw or (hot and pdl), "late wait: hot M19 launches with PDL only"
        assert not lad or (not gated and not hot and pdl and thr), "lad: GEMM2's cold launch of a split, PDL"
        assert not hot or pdl, "hot launches run with PDL"
        tag = (f"m19 g{1 if gated else 2} {'hot' if hot else 'cold'}" + (f" thr{thr}" if thr else "")
               + (" lad" if lad else "") + (" lw" if lw else "") + (" pdl1" if pdl else ""))
        LOG.append(tag)
        base.hit("e35 m19")
        return out.zero_()

    def m11b_gemm(self, *a):
        assert len(a) == 19, len(a)
        w, sf, c, ids, x, out, n_pairs, start, rt, a_per_pair, cols, gated, tt, nt, d, ks, grid, pdl, sel = a
        assert ids.dtype == torch.int32 and n_pairs == ids.numel() and start == 0 and grid > 0 and cols == 4096
        assert x.dtype == torch.bfloat16 and out.shape == (n_pairs, cols) and (sel >> 19) == 0 and (sel & 0xFFFF) <= 64
        thr, hot, lw, lad = sel & 0xFFFF, (sel >> 16) & 1, (sel >> 17) & 1, (sel >> 18) & 1
        assert not hot and not lw, "v22 has no M11b hot side"
        assert not lad or (not gated and pdl and thr), "lad: GEMM2's cold M11b launch of a split, PDL"
        inst = (gated, tt, nt, d, ks)
        assert inst in ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4)), inst
        assert not (gated and pdl), "no PDL GEMM1 M11b instance"
        LOG.append(f"m11b g{1 if gated else 2} sel{thr}" + (" lad" if lad else "") + f" pdl{pdl}")
        base.hit("e35 m11b")
        return out.zero_()

    def m13_post_norm(self, y, nw, tw, ids, out, tokens, start, eps, pdl):
        assert pdl == 1 and start == 0 and tw.dtype == torch.float32 and out.shape == (tokens, 4096)
        LOG.append("post")
        return out.zero_()


def install(lab):
    import torch.utils.cpp_extension as ce
    real = ce.load_inline

    def load(**kw):
        if kw["name"] == "moe36_ext":
            base.hit("nvcc moe36_ext")
            src = kw["cuda_sources"]
            assert "-DM13_NO_WGMMA" in kw["extra_cuda_cflags"] and "m11b_gemm" in kw["functions"]
            for args in ("true, 2, 2, 2, 4", "false, 2, 2, 4, 4", "true, 2, 4, 2, 3", "false, 2, 4, 2, 4"):
                assert f"M19_CASE({args})" in src, args
            assert len(re.findall(r"M19_CASE\((true|false), ", src)) == 4, "exactly v22's four M19 instances"
            for g, tt, nt, d, ks in ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4)):
                assert f"if (gated == {g} && tt == {tt} && nt == {nt} && depth == {d} && ks == {ks}) {{" in src, (g, tt, nt, d, ks)
            assert not re.search(r"depth == \d+ \| ", src) and "nu_kernel<" not in src.split("@@")[0][:0] + ""  # no OCC / EARLY / NU builds
            assert not re.search(r"M11_(CR|SR)_CASE\((true|false)", src) and not re.search(r"M11F_CASE\(\d", src)
            return E35Ext()
        return real(**kw)

    ce.load_inline = load
    lab.load_inline = load
    for name in dir(lab):  # no E25 function may be reached: make any call to one fail loudly
        if name.startswith("m25_") and callable(getattr(lab, name)):
            setattr(lab, name, lambda *a, _n=name, **k: (_ for _ in ()).throw(AssertionError(f"{_n} reached")))


def want(choice, tokens):
    for below, low in CLASSES:
        if low <= tokens < below:
            return choice.get(below)
    return None


def paths(lab, state):
    got = {}
    for tokens, (b, q, layer) in {1: (1, 1, "sliding"), 16: (16, 1, "sliding"), 32: (32, 1, "full"),
                                  65: (65, 1, "full"), 128: (16, 8, "sliding"), 256: (32, 8, "full")}.items():
        LOG.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
        got[tokens] = list(LOG)
    return got


def main():
    mode = (sys.argv[1:] or ["gain"])[0]
    clean = mode == "clean"
    lab = dryrun_v4.load(DECODE / ("submission_v22.py" if clean else "lab_de25.py"))
    install(lab)
    w = base.weights()
    for name in ("w13_sf", "w2_sf"):
        w[name] = torch.full_like(w[name], 0x38)
    if mode == "badscale":
        w["w13_sf"].view(torch.uint8)[5, 0, 0] = 0
    lines, printed = [], []
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            LOG.clear()
            fn()
            split = any("thr16" in t or "sel16" in t for t in LOG)
            ran_layer = "post" in LOG or any(t.startswith("m1") for t in LOG)
            return 100.0 + (500.0 - ((20.0 if mode == "gain" else 2.0) if split else 0.0) if ran_layer else 0.0)

        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
        lab.Lab.empty_graph_us = lambda self: 100.0
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(w)
    builtins.print = real_print
    text = "\n".join(lines)
    choice = state["v17_choice"]
    got = paths(lab, state)
    follows = all(got[t] == SEQ[want(choice, t)] for t in got if want(choice, t))
    v22 = [p for p in printed if p.startswith("[v22]")]
    v20 = [p for p in printed if p.startswith("[v20]")]
    routes = {"Sd1": "ud8", "Sd16": "ud8", "Sd32": "ud9", "Sv1": "uw2h", "Sv16": "uw2h", "Sv32": "w1n", "Fd1": "dec4",
              "Fd16": "dec4p", "Fd32": "dec4p", "Fv1": "w2", "Fv16": "w3b", "Fv32": "w3b"}
    att_ok = (len(v20) == 1 and "attention v18" in v20[0] and str(routes) in v20[0]
              and lab.att_CUDA_ROUTE[(True, False, 1)] == ("dec4", 1, 4, 3, "w")
              and lab.att_CUDA_ROUTE[(False, False, 32)] == ("ud9", 8, 4, 2)
              and lab.att_CUDA_ROUTE[(True, False, 32)] == ("dec4p", 6, 4, 3, 1)
              and "ud9" in lab.att__KINDS and "dec4p" in lab.att__KINDS
              and lab.att_CUDA_ROUTE[(False, True, 1)] == ("uw2h", 4, 1, 16)
              and "ud8" in lab.att__KINDS and "uw2h" in lab.att__KINDS and "uw2h" in lab.att_FP8_KINDS
              and lab.V20_ROUTED_KINDS == sorted(set(routes.values()))
              and all(lab.att__KINDS[k][0] for k in lab.V20_ROUTED_KINDS)
              and all(not lab.att__KINDS[k][0] for k in ("ud4", "ud5", "ud4c", "uw2v", "ud6", "uw2g", "uw2", "w3", "ud3")))
    if clean:
        log_name = os.environ.get("DRYRUN_V22_LOG")
        want_c = dict(lab.V17_CHOICE)
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V17_CHOICE = " in x][-1]
            want_c = ast.literal_eval(ln.split("V17_CHOICE = ", 1)[1])
        checks = {
            "V17_CHOICE = the log's": dict(lab.V17_CHOICE) == want_c,
            "installed = the file's (built and checked)": choice == want_c and any("installed" in v for v in v22),
            "kernel_fn follows (1-256 tokens: launch sequences)": follows,
            "no lab code in the file": not hasattr(lab, "sec_split") and not hasattr(lab, "Lab"),
            "attention v18: routes, UD8 / UD9 / UW2h / D3p registered, routed sources kept, unrouted emptied": att_ok,
            "F12M reduction buffer padded (bf16_proj PE45), built source": lab.pj_f12m_source(lab.pj_F12M_INSTANCES).count(
                "__shared__ float part[KSW][RB][MT + 3];") == 1 and "part[KSW][RB][MT];" not in lab.pj_F12M_SRC,
        }
    else:
        want_c = {} if mode == "badscale" else ({**V20B, 192: "s128", 321: "s256"} if mode == "gain" else dict(V20B))
        printed_c = [ln for ln in lines if ln.startswith("V17_CHOICE = ")]
        want_printed = V20B if mode == "badscale" else want_c
        checks = {
            "path built and checked (8 cases incl. both splits)" if mode != "badscale" else "path OFF on a bad scale":
                any(("OFF" in v) == (mode == "badscale") and "MoE E36 path" in v for v in v22),
            "e35 choice": choice == want_c,
            "V17_CHOICE printed": bool(printed_c) and printed_c[-1] == f"V17_CHOICE = {dict(sorted(want_printed.items()))!r}",
            "kernel_fn follows (1-256 tokens: launch sequences)": follows,
            "split timed" if mode != "badscale" else "nothing timed": ("split (weighted us" in text) == (mode != "badscale"),
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
            "attention v18: routes, UD8 / UD9 / UW2h / D3p registered, routed sources kept, unrouted emptied": att_ok,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        for t, g in got.items():
            print(t, want(choice, t), g)
        raise SystemExit(1)
    print(f"dry run v22 ({mode}): all checks passed (choice {choice})")


if __name__ == "__main__":
    main()
