"""CPU dry run of v15 (parts/v15.py: local ids on the side stream, pre-expert norm before the join; PDL classes + 9-16
tokens) and the DE17 lab.

lab      lab_de17.py under fake timings (a layer call 400 us, 1.5 us less with v15 on): side installs v15 and prints
         V15_SIDE True; the pdl section covers classes 5 / 9 / 17 (no PDL gain modelled: nothing installed).
nogain   the same with no v15 gain: V15_SIDE False.
Both: every layer call runs the local-ids kernel and the pre-expert norm exactly once with v15 off and on (no double
norm, none skipped), the layer outputs are identical, v10's prepare check covers 16 tokens, the real routing is restored.
clean    submission_v15.py with V15_SIDE / V10_PDL from DRYRUN_V15_LOG: installed as printed; with PDL at class 17,
         kernel_fn at 16 tokens launches GEMM2 through m11p and the CUDA post-norm with PDL.

Run: .venv/bin/python decode_layer/local/dryrun_v15.py lab_de17.py [nogain]
     DRYRUN_V15_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v15.py submission_v15.py
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


def calls_once(lab, state):
    ok, outs = True, {}
    for on in (False, True):
        state["v15_side"] = on
        for b, q, layer in ((1, 1, "sliding"), (1, 8, "full"), (16, 1, "sliding"), (32, 8, "full")):
            base.CALLS.clear()
            out = lab.kernel_fn(lab.lab_input(b, q, layer, seed=b * q, device="cpu"), state)
            ok &= base.CALLS.get("v3 lid", 0) == 1 and base.CALLS.get("v3 norm", 0) == 1
            outs[(on, b, q)] = out[0]
    same = all(torch.equal(outs[(False, b, q)], outs[(True, b, q)]) for (_o, b, q) in outs)
    return ok, same


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de17.py"
    nogain = "nogain" in sys.argv[2:]
    lab = dryrun_v4.load(path)
    clean = not hasattr(lab, "sec_side")
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
                t += 400.0 - (1.5 if ref["s"].get("v15_side") and not nogain else 0.0)
            return t

        real_kernel_fn = lab.kernel_fn

        def kernel_fn(data, state):
            ref["s"] = state
            return real_kernel_fn(data, state)

        lab.kernel_fn = kernel_fn
        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    if not clean:
        lab.kernel_fn = real_kernel_fn
    text = "\n".join(lines)
    v10 = [p for p in printed if p.startswith("[v10]")]
    side = state["v15_side"]
    once, same = calls_once(lab, state)
    state["v15_side"] = side
    if clean:
        log_name = os.environ.get("DRYRUN_V15_LOG")
        want_side, want_pdl = lab.V15_SIDE, dict(lab.V10_PDL)
        if log_name:
            lt = (DECODE / "profiles" / log_name).read_text().splitlines()
            want_side = ast.literal_eval([x for x in lt if "V15_SIDE = " in x][-1].split("V15_SIDE = ", 1)[1])
            want_pdl = ast.literal_eval([x for x in lt if "V10_PDL = " in x][-1].split("V10_PDL = ", 1)[1])
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(16, 1, "sliding", seed=16, device="cpu"), state)
        at16 = set(base.CALLS)
        pdl17 = want_pdl.get(17)
        checks = {
            "V15_SIDE / V10_PDL = the log's": lab.V15_SIDE == want_side and lab.V10_PDL == want_pdl,
            "installed = the file's": side == want_side and state["v10_pdl"] == want_pdl,
            "v10 check covers 4 / 8 / 16 tokens": len(v10) == 1 and "T4/T8/T16 checked" in v10[0],
            "16 tokens follow the class-17 PDL pick": (("m11p g0" in at16) == bool(pdl17 and pdl17.get("s2"))) and (
                ("m11 post cuda pdl1" in at16) == bool(pdl17 and pdl17.get("post") == 1)),
            "lid and norm once per layer call, off and on": once,
            "outputs identical off / on": same,
            "no lab code in the file": not hasattr(lab, "sec_side") and not hasattr(lab, "Lab"),
        }
    else:
        printed_side = [ln for ln in lines if ln.startswith("V15_SIDE = ")]
        checks = {
            "side decision": side == (not nogain),
            "V15_SIDE printed = state": bool(printed_side) and printed_side[-1] == f"V15_SIDE = {side!r}",
            "pdl covers classes 5 / 9 / 17": "<17 " in text and "<9 " in text and "<5 " in text,
            "v10 check covers 4 / 8 / 16 tokens": len(v10) == 1 and "T4/T8/T16 checked" in v10[0],
            "lid and norm once per layer call, off and on": once,
            "outputs identical off / on": same,
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v15 ({'clean file' if clean else 'lab'}{' nogain' if nogain else ''}): all checks passed "
          f"(side {side}, PDL {state['v10_pdl']})")


if __name__ == "__main__":
    main()
