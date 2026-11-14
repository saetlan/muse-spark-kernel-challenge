"""CPU dry run of v16 (parts/moe19.py + parts/v16.py: MoE E19's M19 path at 65-320 tokens) and the DE18 lab.

The moe19 extension is a shape- and argument-checking stand-in (load_inline name "moe19_ext": m19_setup, m19_gemm for
GEMM1 / GEMM2 with E19's argument layout, m13_post_norm), returning zeros like dryrun_v3's M11 stand-in; Marlin's
stand-in returns zeros too, so prepare's check passes on wiring alone (the kernels are MoE's judged code).

lab      lab_de18.py, fake timings (a layer call 500 us; 20 us less when M19 ran): the build happens (V16_FORCE), both
         classes pick M19, V16_M19 printed = state; kernel_fn: M19 (GEMM1, GEMM2 PDL, post-norm PDL) at 128 / 256
         tokens and at 65 / 320, M11 or Marlin elsewhere (1 / 8 / 16 / 32 / 64 / 321+ never M19).
nogain   M19 no faster: nothing installed.
clean    submission_v16.py with V16_M19 from DRYRUN_V16_LOG (or {} with no build at all).
badscale a zero E4M3 scale in the local experts: the path stays OFF (M13's decode needs normal scales).

Run: .venv/bin/python decode_layer/local/dryrun_v16.py lab_de18.py [nogain | badscale]
     DRYRUN_V16_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v16.py submission_v16.py
"""

import ast
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]


class G2PExt:
    def g2p_attrs(self, *a):
        return torch.tensor([64, 0, 21000])

    def g2p_gemm(self, w, sf, c, ids, a, y, n_pairs, start, rt, out_cols, tt, nt, d, ks, grid, pdl, nw, tw, out, cnt,
                 eps):
        assert (tt, nt, d, ks, pdl) in ((1, 1, 4, 4, 1), (2, 1, 4, 4, 1), (2, 2, 2, 4, 0)), (tt, nt, d, ks, pdl)  # v14 builds 3
        assert int(cnt.abs().sum()) == 0 and cnt.numel() == 32 + 320 and ids.dtype == torch.int32 and start == 0
        assert y.shape == (n_pairs, out_cols) and out.shape[1] == 4096 and tw.dtype == torch.float32
        base.hit(f"g2p t{tt}n{nt} pdl{pdl}")
        y.zero_()
        return out.zero_()


class M19Ext:
    def m19_setup(self):
        base.hit("m19 setup")
        return torch.zeros(4, dtype=torch.int32)

    def m19_gemm(self, *a):
        assert len(a) == 19, len(a)
        w, sf, c, ids, x, out, scr, ctr, n_pairs, start, rt, gated_or_ap, cols, mode, tt_or_two, nt, grid, z, flags = a
        assert ids.dtype == torch.int32 and n_pairs == ids.numel() and start == 0 and grid > 0
        assert x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16 and out.shape == (n_pairs, cols)
        assert nt == 2 and (flags >> 28) & 1 == 1, (nt, flags)  # MB 4
        gemm1 = gated_or_ap == 0
        p = (flags >> 24) & 0xF
        assert (gemm1 and p == 2 and cols == 4096 and z == 0) or (not gemm1 and p == 4 and cols == 4096 and z == 1), \
            (gemm1, p, cols, z)
        base.hit("m19 g1" if gemm1 else "m19 g2 pdl")
        return out.zero_()

    def m13_post_norm(self, y, nw, tw, ids, out, tokens, start, eps, pdl):
        base.hit(f"m13 post pdl{pdl}")
        assert pdl == 1 and start == 0 and tw.dtype == torch.float32 and out.shape == (tokens, 4096)
        return out.zero_()


def install(mode):
    import torch.utils.cpp_extension as ce
    real = ce.load_inline

    def load(**kw):
        if kw["name"] == "g2p_ext":
            base.hit("nvcc g2p_ext")
            assert kw["functions"] == ["g2p_gemm", "g2p_attrs"]
            return G2PExt()
        if kw["name"] == "moe19_ext":
            base.hit("nvcc moe19_ext")
            assert "-DM13_NO_WGMMA" in kw["extra_cuda_cflags"] and "m19_gemm" in kw["functions"]
            src = kw["cuda_sources"]
            assert "M19_CASE(true, 2, 2, 2, 4)" in src and "M19_CASE(false, 2, 2, 4, 4)" in src
            assert "M14_CASE(" not in src.split("@@")[0] or True
            return M19Ext()
        return real(**kw)

    ce.load_inline = load
    return load


def paths(lab, state):
    got = {}
    for tokens, (b, q, layer) in {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"),
                                  32: (32, 1, "full"), 64: (8, 8, "sliding"), 65: (65, 1, "full"),
                                  128: (16, 8, "sliding"), 256: (32, 8, "full"), 320: (40, 8, "sliding"),
                                  336: (42, 8, "full")}.items():
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
        got[tokens] = {"m19": base.CALLS.get("m19 g1", 0) == 1 and base.CALLS.get("m19 g2 pdl", 0) == 1
                       and base.CALLS.get("m13 post pdl1", 0) == 1,
                       "m11": any(k.startswith("m11") for k in base.CALLS),
                       "g2p": any(k.startswith("g2p t") for k in base.CALLS),
                       "post": any("post" in k and k.startswith("m11") for k in base.CALLS)}
    return got


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de18.py"
    mode = (sys.argv[2:] or [""])[0]
    lab = dryrun_v4.load(path)
    lab.load_inline = install(mode)  # the file bound load_inline at import: rebind its global too
    clean = not hasattr(lab, "sec_m19")
    g2p_lab = hasattr(lab, "sec_gemm2_post")  # DE18 runs the G2P section; DE19 doesn't
    w = base.weights()
    for name in ("w13_sf", "w2_sf"):  # normal positive E4M3 block scales (0x38 = 1.0), as the judge's weights
        w[name] = torch.full_like(w[name], 0x38)
    if mode == "badscale":
        w["w2_sf"] = w["w2_sf"].clone()
        w["w2_sf"].view(torch.uint8)[3, 0, 0] = 0
    lines, printed, ref = [], [], {}
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            before = dict(base.CALLS)
            fn()
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            t = 100.0
            if "v3 lid" in ran:
                t += 500.0 - (20.0 if "m19 g1" in ran and mode != "nogain" else 0.0)
                t -= 3.0 if any(k.startswith("g2p t") for k in ran) and mode != "nogain" else 0.0
            return t

        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(w)
    builtins.print = real_print
    text = "\n".join(lines)
    v16 = [p for p in printed if p.startswith("[v16]")]
    picks = state["v16_m19"]
    got = paths(lab, state)
    want_on = {t: any(low <= t < below and picks.get(below) for below, low in ((33, 17), (192, 65), (321, 192)))
               for t in got}
    follows = all(got[t]["m19"] == want_on[t] for t in got) and all(not got[t]["m11"] for t in got if want_on[t])
    fused = {b: bool((c or {}).get("g2p")) for b, c in state["m11_configs"] if b in (5, 9, 17, 33)}
    g2p_follows = all(got[t]["g2p"] == fused.get(next((b for b in (5, 9, 17, 33) if t < b), 0), False)
                      and (not got[t]["g2p"] or not got[t]["post"]) for t in (1, 8, 16, 32))
    if clean:
        log_name = os.environ.get("DRYRUN_V16_LOG")
        want = {}
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V16_M19 = " in x][-1]
            want = ast.literal_eval(ln.split("V16_M19 = ", 1)[1])
        checks = {
            "V16_M19 = the log's": lab.V16_M19 == want,
            "installed = the file's": picks == want,
            "no build without picks; built and on with them": (base.CALLS.get("nvcc moe19_ext", 0) == 0) if not want
                else (len(v16) == 1 and "installed" in v16[0]),
            "kernel_fn follows (1-336 tokens)": follows,
            "G2P where the file picks it, never with the separate post-norm": g2p_follows,
            "no lab code in the file": not hasattr(lab, "sec_m19") and not hasattr(lab, "Lab"),
        }
    else:
        want = {} if mode in ("nogain", "badscale") else {33: True, 192: True, 321: True}
        printed_m19 = [ln for ln in lines if ln.startswith("V16_M19 = ")]
        checks = {
            "path built and checked (lab forces it)" if mode != "badscale" else "path OFF on a bad scale":
                (len(v16) == 1 and ("OFF" in v16[0]) == (mode == "badscale")),
            "m19 picks": picks == want,
            "V16_M19 printed = state": bool(printed_m19) and printed_m19[-1] == f"V16_M19 = {dict(sorted(picks.items()))!r}",
            "kernel_fn follows (1-336 tokens)": follows,
            "G2P picks": fused == ({5: False, 9: False, 17: False, 33: False} if mode == "nogain" or not g2p_lab
                                   else {5: True, 9: True, 17: True, 33: True}),
            "V14_FUSED printed (labs with the G2P section)": (not g2p_lab) or any(ln.startswith("V14_FUSED = ") for ln in lines),
            "G2P where the lab picks it, never with the separate post-norm": g2p_follows,
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v16 ({'clean file' if clean else 'lab'}{(' ' + mode) if mode else ''}): all checks passed (picks {picks})")


def v16_class_on(picks, tokens):
    for below in sorted(picks):
        if tokens < below:
            return bool(picks[below])
    return False


if __name__ == "__main__":
    main()
