"""CPU dry run of v21 (parts/dgrs.py + parts/v21.py: bf16_proj v11b's DG12-RS for the pooler / unpooler at 32 tokens)
and the DE24 lab.

encoder   the real embedded encoder, pb_f12m_device_planes(natural=True) + pb_dgrs_pack, on randn / sqrt(k) BF16 weights
          of the real pooler (4096 x 8192) and unpooler (8192 x 4096) sizes: planes fit, shapes [K/64, N/16, 3, 32, 4]
          int32 and rec [N, 8] int32 (what dgrs_gemm reads).
Then the stubbed world (dryrun_v4 + dryrun_v17's E25 stand-in); the dgrs extension is an argument-checking stand-in
(load_inline name "dgrs_ext": dgrs_gemm with v11b's 17 arguments, the fully specialized (M, N, K, BM, S, red, math)
of pb_DGRS_CHOICE; it computes x @ W^T from the "planes", which the dry run sets to W itself).

lab modes (fake timings: one projection 20 us, DG12-RS 16 us; a layer call 500 us minus 4 per DG12-RS launch):
gain      both shapes pass op and layer: V21_DGRS = both; kernel_fn runs DG12-RS for pooler and unpooler at 32 tokens only.
slowpool  the pooler's DG12-RS is 21 us (slower): only the unpooler passes; V21_DGRS = unpooler.
nobuild   the dgrs build fails: [v21] OFF, V21_DGRS = {}, kernel_fn is v20's.
clean     submission_v21.py with V21_DGRS from DRYRUN_V21_LOG (default: the file's): installed, kernel_fn follows.

Run: .venv/bin/python decode_layer/local/dryrun_v21.py [gain | slowpool | nobuild]
     DRYRUN_V21_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v21.py clean
"""

import ast
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402
import dryrun_v17  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
MODE = [""]


class DgrsExt:
    def __init__(self, cases):
        self.cases = cases

    def dgrs_gemm(self, wp, rec, x, out, part, cnt, err, M, N, K, bm, splits, smax, cps, diag, red, math):
        assert x.dtype == torch.bfloat16 and x.is_contiguous() and x.shape == (M, K) and out.shape == (M, N)
        assert out.dtype == torch.bfloat16 and err.dtype == torch.int32 and cnt.dtype == torch.int32
        assert (M, N, K, bm, splits, smax, cps, diag, red, math) in self.cases, (M, N, K, bm, splits, smax, cps, diag, red, math)
        assert wp.shape == (N, K), "dry run: planes are W itself"
        base.hit(f"dgrs {'pooler' if N == 4096 else 'unpooler'}")
        out.copy_((x.float() @ wp.float().T).to(torch.bfloat16))
        return out


def install(lab):
    import torch.utils.cpp_extension as ce
    inner = dryrun_v17.install()  # moe25_ext stand-in (and the real load_inline for anything else)

    def load(**kw):
        if kw["name"] == "dgrs_ext":
            base.hit("nvcc dgrs_ext")
            if MODE[0] == "nobuild":
                raise RuntimeError("nvcc failed (dry run)")
            src = kw["cuda_sources"]
            cases = set()
            for key, c in lab.pb_DGRS_CHOICE.items():
                t = lab.pb_dgrs_inst(c, *key)  # (BM, CPS, DIAG, MATH, RED, S, SMX, M, N, K)
                assert "DGRS_CASE(" + ", ".join(str(v) for v in t) + ")" in src, t
                cases.add((t[7], t[8], t[9], t[0], t[5], t[6], t[1], t[2], t[4], t[3]))
            assert "-gencode=arch=compute_90a,code=sm_90a" in kw["extra_cuda_cflags"] and "dgrs_gemm" in kw["functions"]
            return DgrsExt(cases)
        return inner(**kw)

    ce.load_inline = load
    lab.load_inline = load
    lab.pb_f12m_device_planes = lambda w, natural=False: {"p": w, "rec": torch.zeros((w.shape[0], 8), dtype=torch.int32)}
    lab.pb_dgrs_pack = lambda p: p


def encoder_check(lab):
    g = torch.Generator().manual_seed(5)
    ok = True
    for n, k in ((4096, 8192), (8192, 4096)):
        w = (torch.randn((n, k), generator=g) / k ** 0.5).to(torch.bfloat16)
        pl = lab.pb_f12m_device_planes(w, natural=True)
        ok &= pl is not None
        if pl is None:
            continue
        wp = lab.pb_dgrs_pack(pl["p"])
        ok &= (wp.dtype == torch.int32 and tuple(wp.shape) == (k // 64, n // 16, 3, 32, 4) and wp.is_contiguous()
               and pl["rec"].dtype == torch.int32 and tuple(pl["rec"].shape) == (n, 8) and pl["max_row"] <= 7)
        print(f"encoder {n}x{k}: planes {tuple(wp.shape)} rec {tuple(pl['rec'].shape)} escape rate {pl['rate']:.2e} "
              f"max per row {pl['max_row']}")
    return ok


def paths(lab, state):
    got = {}
    for tokens, (b, q, layer) in {16: (16, 1, "sliding"), 32: (32, 1, "sliding"), 33: (32, 1, "full"),
                                  128: (16, 8, "sliding")}.items():
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
        got[tokens] = {"pooler": base.CALLS.get("dgrs pooler", 0), "unpooler": base.CALLS.get("dgrs unpooler", 0)}
    return got


def main():
    MODE[0] = (sys.argv[1:] or ["gain"])[0]
    clean = MODE[0] == "clean"
    lab = dryrun_v4.load(DECODE / ("submission_v21.py" if clean else "lab_de24.py"))
    enc_ok = encoder_check(lab) if MODE[0] in ("gain", "clean") else True
    install(lab)
    w = base.weights()
    for name in ("w13_sf", "w2_sf"):
        w[name] = torch.full_like(w[name], 0x38)
    lines, printed = [], []
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            before = dict(base.CALLS)
            fn()
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            if "v3 lid" in ran:  # a whole layer
                return 100.0 + 500.0 - 4.0 * sum(1 for k in ran if k.startswith("dgrs "))
            if "dgrs pooler" in ran:
                return 100.0 + (21.0 if MODE[0] == "slowpool" else 16.0)
            if "dgrs unpooler" in ran:
                return 100.0 + 16.0
            return 100.0 + 20.0  # one projection on v20's kernel (cuBLAS records no call here)

        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
        lab.Lab.empty_graph_us = lambda self: 100.0
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(w)
    builtins.print = real_print
    text = "\n".join(lines)
    on = state.get("v21_on") or set()
    got = paths(lab, state)
    follows = all(got[t]["pooler"] == int(("pooler", 32) in on and t in (32, 33))
                  and got[t]["unpooler"] == int(("unpooler", 32) in on and t in (32, 33)) for t in got)
    v21 = [p for p in printed if p.startswith("[v21]")]
    if clean:
        want = dict(lab.V21_DGRS)
        log_name = os.environ.get("DRYRUN_V21_LOG")
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V21_DGRS = " in x][-1]
            want = ast.literal_eval(ln.split("V21_DGRS = ", 1)[1])
        checks = {
            "encoder (real, CPU, full size)": enc_ok,
            "V21_DGRS = the log's": dict(lab.V21_DGRS) == want,
            "installed = the file's picks": on == {k for k, v in want.items() if v},
            "[v21] line": len(v21) == 1 and ("not used" in v21[0] if not want else "serving" in v21[0]),
            "kernel_fn follows (16 / 32 / 128 tokens)": follows,
            "no lab code in the file": not hasattr(lab, "sec_dgrs") and not hasattr(lab, "Lab"),
        }
    else:
        want = {"gain": {("pooler", 32), ("unpooler", 32)}, "slowpool": {("unpooler", 32)}, "nobuild": set()}[MODE[0]]
        printed_c = [ln for ln in lines if ln.startswith("V21_DGRS = ")]
        checks = {
            "encoder (real, CPU, full size)": enc_ok,
            "[v21] built and checked" if MODE[0] != "nobuild" else "[v21] OFF on a failed build":
                len(v21) == 1 and (("OFF" in v21[0]) == (MODE[0] == "nobuild")),
            "dgrs choice": on == want,
            "V21_DGRS printed = state": bool(printed_c) and printed_c[-1] == f"V21_DGRS = {dict(sorted((k, True) for k in on))!r}",
            "kernel_fn follows (16 / 32 / 128 tokens)": follows,
            "op and layer timings logged" if MODE[0] != "nobuild" else "nothing timed":
                ("dgrs op (" in text and "dgrs layer (" in text) == (MODE[0] != "nobuild"),
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v21 ({MODE[0]}): all checks passed (serving {sorted(on)})")


if __name__ == "__main__":
    main()
