"""CPU dry run of v10's PDL path (parts/m11_pdl.py + parts/v10.py) and its DE12 lab section.

lab   lab_de12.py under fake timings: a layer whose expert stage ran GEMM2 PDL-launched is 3 us faster, a CUDA
      post-norm with PDL 1 us faster, but only at 1 token (8 tokens: both 1 us slower). The pdl section must
      install {5: GEMM2 + post PDL} only, print it as V10_PDL, and the timed kernel_fn must then launch GEMM2
      through m11p_gemm and the CUDA post-norm with pdl 1 at 1 token, and v9's path (m11_gemm, the Triton
      post-norm) at 8 tokens. The same lab with equal times must install nothing (the bar is sign-safe).
      small: GEMM2 PDL only 1 us faster at 1 token, so GEMM2 + post-norm PDL gains 2 of 200 us (1%): installed
      under a bar below 1% (DE13's PDL_BAR 0.5%), not under DE12's 1.5%.
clean submission_v10.py with V10_PDL from a lab log (DRYRUN_PDL_LOG) or the default {}: kernel_fn follows it.

Run: .venv/bin/python decode_layer/local/dryrun_v10.py lab_de12.py [equal | small]
     DRYRUN_PDL_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v10.py submission_v10.py
"""

import ast
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]


def expert_calls(lab, state, tokens, layer):
    b, q = {1: (1, 1), 8: (1, 8)}[tokens]
    base.CALLS.clear()
    lab.kernel_fn(lab.lab_input(b, q, layer, seed=tokens, device="cpu"), state)
    return {k for k in base.CALLS if k.startswith("m11")}


def follows(lab, state, picks):
    ok = True
    for tokens, below in ((1, 5), (8, 9)):
        for layer in ("sliding", "full"):
            got = expert_calls(lab, state, tokens, layer)
            p = picks.get(below)
            want_m11p = bool(p and p.get("s2"))
            want_post = None if not p or p.get("post") is None else f"m11 post cuda pdl{p['post']}"
            ok &= ("m11p g0" in got) == want_m11p and ("m11 g0" in got) == (not want_m11p) and "m11 g1" in got
            # the Triton post-norm's stand-in records "m11 post" (dryrun_v2)
            ok &= (want_post in got and "m11 post" not in got) if want_post else ("m11 post" in got and not any("post cuda" in c for c in got))
    return ok


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de12.py"
    equal = "equal" in sys.argv[2:]
    small = "small" in sys.argv[2:]
    lab = dryrun_v4.load(path)
    clean = not hasattr(lab, "sec_pdl")
    lines = []
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            before = dict(base.CALLS)
            fn()
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            t = 100.0 + (200.0 if ran else 0.0)
            if not equal and "v3 lid" in ran:  # a layer call (the empty graph runs no lid kernel)
                one = before.get("m11 g1", 0) < base.CALLS.get("m11 g1", 0) and self._tokens == 1
                if "m11p g0" in ran:
                    t += (-1.0 if small else -3.0) if one else 1.0
                if "m11 post cuda pdl1" in ran:
                    t += -1.0 if one else 1.0
            return t

        real_kernel_fn = lab.kernel_fn

        def kernel_fn(data, state):  # remembers the token count for fake_time
            lab.Lab._tokens = data["config"]["tokens"]
            return real_kernel_fn(data, state)

        lab.kernel_fn = kernel_fn
        lab.Lab._tokens = 0
        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    if not clean:
        lab.kernel_fn = real_kernel_fn
    text = "\n".join(lines)
    v10 = [p for p in printed if p.startswith("[v10]")]
    picks = state["v10_pdl"]
    if clean:
        log_name = os.environ.get("DRYRUN_PDL_LOG")
        want = {}
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "V10_PDL = " in x][-1]
            want = ast.literal_eval(ln.split("V10_PDL = ", 1)[1])
        checks = {
            "v10 check on": len(v10) == 1 and "PDL path: on" in v10[0],
            "installed = the file's V10_PDL": picks == lab.V10_PDL and picks is not lab.V10_PDL,
            "V10_PDL = the lab's printed picks": lab.V10_PDL == want,
            "kernel_fn follows it (1 and 8 tokens, sliding and full)": follows(lab, state, picks),
            "no lab code in the file": not hasattr(lab, "sec_pdl") and not hasattr(lab, "Lab"),
        }
    else:
        want = {} if equal or (small and not getattr(lab, "PDL_BAR", 0.015) < 0.01) else {5: {"s2": 1, "post": 1}}
        printed_picks = [ln for ln in lines if ln.startswith("V10_PDL = ")]
        checks = {
            "v10 check on": len(v10) == 1 and "PDL path: on" in v10[0],
            "pdl picks (fake timings)": picks == want,
            "V10_PDL printed = state": bool(printed_picks) and printed_picks[-1] == "V10_PDL = " + repr(dict(sorted(picks.items()))),
            "kernel_fn follows the picks (1 and 8 tokens, sliding and full)": follows(lab, state, picks),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v10 ({'clean file' if clean else 'lab' + (' equal' if equal else ' small' if small else '')}): "
          f"all checks passed" + ('' if clean else f" (bar {getattr(lab, 'PDL_BAR', 0.015):.1%}, picks {picks})"))


if __name__ == "__main__":
    main()
