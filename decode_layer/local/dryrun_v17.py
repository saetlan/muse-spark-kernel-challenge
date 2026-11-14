"""CPU dry run of v17 (parts/moe25.py + parts/v17.py: MoE E25's M11b at 1-32 tokens and M19 at 65-191) and the DE20 lab.

The moe25 extension is an argument-checking stand-in (load_inline name "moe25_ext": m19_setup, m19_gemm with E19/E25's
19-argument layout, m11b_gemm with E25's 18-argument layout, m13_post_norm), returning zeros like dryrun_v3's M11
stand-in (Marlin's and M11's stand-ins return zeros too, so prepare's check passes on wiring alone; the kernels are
MoE's judged code). The judge's weights have normal E4M3 scales (0x38 here).

lab      lab_de20.py on v17 with V17_CHOICE {33: m19, 192: m19}; fake timings (a layer call 500 us, 15 us less when M11b
         ran): every small class picks M11b, 192 keeps M19; V17_CHOICE printed = state; kernel_fn: M11b (GEMM1, GEMM2
         PDL, M13 post-norm PDL) at 1 / 8 / 16 / 32 tokens and never our M11 there, M19 at 65 / 128, Marlin at 64 / 256.
nogain   M11b no faster: the file's choice stays.
badscale a zero E4M3 scale: the E25 path stays OFF, kernel_fn is v15's (our M11, Marlin).
clean    submission_v17.py with V17_CHOICE from DRYRUN_V17_LOG (or the file's): installed, kernel_fn follows.

Run: .venv/bin/python decode_layer/local/dryrun_v17.py lab_de20.py [nogain | badscale]
     DRYRUN_V17_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v17.py submission_v17.py
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
CLASSES = ((5, 1), (9, 5), (17, 9), (33, 17), (192, 65), (321, 192))


class E25Ext:
    def m19_setup(self):
        base.hit("e25 m19 setup")
        return torch.zeros(4, dtype=torch.int32)

    def m19_gemm(self, *a):
        assert len(a) == 19, len(a)
        out, n_pairs, ids, nt, flags, cols, z = a[5], a[8], a[3], a[15], a[18], a[12], a[17]
        assert ids.dtype == torch.int32 and n_pairs == ids.numel() and a[9] == 0 and nt == 2 and (flags >> 28) & 1 == 1
        gemm1 = a[11] == 0
        assert out.shape == (n_pairs, cols) and ((gemm1 and z == 0) or (not gemm1 and z == 1))
        base.hit("e25 m19 g1" if gemm1 else "e25 m19 g2 pdl")
        return out.zero_()

    def m11b_gemm(self, *a):
        assert len(a) == 18, len(a)
        w, sf, c, ids, x, out, n_pairs, start, rt, a_per_pair, cols, gated, tt, nt, d, ks, grid, pdl = a
        assert ids.dtype == torch.int32 and n_pairs == ids.numel() and start == 0 and grid > 0
        assert x.dtype == torch.bfloat16 and out.shape == (n_pairs, cols) and cols == 4096
        if gated:
            assert (tt, nt, d, ks) in ((2, 1, 4, 4), (2, 1, 4, 2)) and pdl == 0 and a_per_pair == 0, (tt, nt, d, ks, pdl)
        else:
            assert (tt, nt, d, ks) in ((1, 1, 4, 4), (2, 1, 4, 4)) and pdl == 1 and a_per_pair == 1, (tt, nt, d, ks, pdl)
        base.hit("e25 m11b g1" if gated else "e25 m11b g2 pdl")
        return out.zero_()

    def m13_post_norm(self, y, nw, tw, ids, out, tokens, start, eps, pdl):
        base.hit(f"e25 post pdl{pdl}")
        assert pdl == 1 and start == 0 and tw.dtype == torch.float32 and out.shape == (tokens, 4096)
        return out.zero_()


def install():
    import torch.utils.cpp_extension as ce
    real = ce.load_inline

    def load(**kw):
        if kw["name"] == "moe25_ext":
            base.hit("nvcc moe25_ext")
            src = kw["cuda_sources"]
            assert "-DM13_NO_WGMMA" in kw["extra_cuda_cflags"] and "m11b_gemm" in kw["functions"]
            assert "M19_CASE(true, 2, 2, 2, 4)" in src and "M19_CASE(false, 2, 2, 4, 4)" in src
            for g, tt, nt, d, ks in ((1, 2, 1, 4, 4), (1, 2, 1, 4, 2), (0, 1, 1, 4, 4), (0, 2, 1, 4, 4)):
                assert f"if (gated == {g} && tt == {tt} && nt == {nt} && depth == {d} && ks == {ks})" in src, (g, tt, nt, d, ks)
            assert "gated == 1 && tt == 2 && nt == 2" not in src  # E25's 16-pair M11b shapes are not built
            import re  # no CR / SR / G2P instantiations (their macro definitions stay in the text)
            assert not re.search(r"M11_(CR|SR)_CASE\((true|false)", src) and not re.search(r"M11F_CASE\(\d", src)
            return E25Ext()
        return real(**kw)

    ce.load_inline = load
    return load


def paths(lab, state):
    got = {}
    for tokens, (b, q, layer) in {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"), 32: (32, 1, "full"),
                                  64: (8, 8, "sliding"), 65: (65, 1, "full"), 128: (16, 8, "sliding"),
                                  256: (32, 8, "full")}.items():
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
        c = base.CALLS
        got[tokens] = {"m11b": c.get("e25 m11b g1", 0) == 1 and c.get("e25 m11b g2 pdl", 0) == 1 and c.get("e25 post pdl1", 0) == 1,
                       "m19": c.get("e25 m19 g1", 0) == 1 and c.get("e25 m19 g2 pdl", 0) == 1 and c.get("e25 post pdl1", 0) == 1,
                       "m11": any(k.startswith("m11") for k in c)}
    return got


def want(choice, tokens):
    for below, low in CLASSES:
        if low <= tokens < below:
            return choice.get(below)
    return None


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de20.py"
    mode = (sys.argv[2:] or [""])[0]
    lab = dryrun_v4.load(path)
    lab.load_inline = install()
    clean = not hasattr(lab, "sec_e25")
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
            before = dict(base.CALLS)
            fn()
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            t = 100.0
            if "v3 lid" in ran:
                t += 500.0 - (15.0 if "e25 m11b g1" in ran and mode != "nogain" else 0.0)
            return t

        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(w)
    builtins.print = real_print
    text = "\n".join(lines)
    v17 = [p for p in printed if p.startswith("[v17]")]
    choice = state["v17_choice"]
    got = paths(lab, state)
    follows = True
    for t, g in got.items():
        k = want(choice, t)
        follows &= g["m11b"] == (k == "m11b") and g["m19"] == (k == "m19") and (not g["m11"] or k is None)
        if t <= 32 and k is None:
            follows &= g["m11"]
    if clean:
        log_name = os.environ.get("DRYRUN_V17_LOG")
        want_c = dict(lab.V17_CHOICE)
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V17_CHOICE = " in x][-1]
            want_c = ast.literal_eval(ln.split("V17_CHOICE = ", 1)[1])
        checks = {
            "V17_CHOICE = the log's": dict(lab.V17_CHOICE) == want_c,
            "installed = the file's (built and checked when it has a choice)": choice == want_c and (
                not want_c or (len(v17) == 1 and "installed" in v17[0])),
            "kernel_fn follows (1-256 tokens)": follows,
            "no lab code in the file": not hasattr(lab, "sec_e25") and not hasattr(lab, "Lab"),
        }
    else:
        base_c = {33: "m19", 192: "m19"}
        want_c = {} if mode == "badscale" else (base_c if mode == "nogain" else {5: "m11b", 9: "m11b", 17: "m11b", 33: "m11b", 192: "m19"})
        printed_c = [ln for ln in lines if ln.startswith("V17_CHOICE = ")]
        checks = {
            "path built and checked" if mode != "badscale" else "path OFF on a bad scale":
                len(v17) == 1 and ("OFF" in v17[0]) == (mode == "badscale"),
            "e25 choice": choice == want_c,
            "V17_CHOICE printed": bool(printed_c) and (mode == "badscale" or printed_c[-1] == f"V17_CHOICE = {dict(sorted(choice.items()))!r}"),
            "kernel_fn follows (1-256 tokens)": follows,
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v17 ({'clean file' if clean else 'lab'}{(' ' + mode) if mode else ''}): all checks passed (choice {choice})")


if __name__ == "__main__":
    main()
