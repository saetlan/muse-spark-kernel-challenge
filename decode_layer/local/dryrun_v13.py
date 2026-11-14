"""CPU dry run of v13 (parts/v13.py: one concat instead of F.pad at <= 8 tokens) and the DE16 lab.

lab      lab_de16.py under fake timings (dryrun_v12's boundary costs, but C16 at 17-64 tokens 3.0 us per call; a layer
         whose expert stage ran with the concat installed 1.5 us faster): pad installs the concat, b32 keeps C16 at
         decode 32; V13_CAT / V12_BOUNDARY printed = state; kernel_fn: the concat path at 1 / 8 tokens (outputs equal
         F.pad's), C16 at 32 tokens, the file's other boundary picks unchanged.
nogain   the same lab with no concat gain and C16 at 32 slower: nothing new installed, the file's picks unchanged.
clean    submission_v13.py with V13_CAT / V12_BOUNDARY from DRYRUN_V13_LOG (or the file's defaults): prepare's check
         prints the concat line iff V13_CAT, kernel_fn follows.

Run: .venv/bin/python decode_layer/local/dryrun_v13.py lab_de16.py [nogain]
     DRYRUN_V13_LOG=<log> .venv/bin/python decode_layer/local/dryrun_v13.py submission_v13.py
"""

import ast
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402
import dryrun_v12  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de16.py"
    nogain = "nogain" in sys.argv[2:]
    real_cost = dryrun_v12.cluster_cost
    dryrun_v12.cluster_cost = lambda t, s: (6.0 if nogain else 3.0) if (17 <= t <= 64 and s == 16) else real_cost(t, s)
    lab = dryrun_v4.load(path)
    clean = not hasattr(lab, "sec_pad")
    file_bounds = dict(lab.V12_BOUNDARY)
    lines, printed = [], []
    state_ref = {}
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

        def fake_time(self, fn, reps=15, cold=True):
            before = dict(base.CALLS)
            fn()
            ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
            t = 100.0 + dryrun_v12.boundary_cost(before, base.CALLS)
            if "v3 lid" in ran:
                t += 300.0
                if not nogain and state_ref["s"].get("v13_zeros") is not None and self._tokens <= 8:
                    t -= 1.5
            return t

        real_kernel_fn = lab.kernel_fn

        def kernel_fn(data, state):
            state_ref["s"] = state
            lab.Lab._tokens = data["config"]["tokens"]
            return real_kernel_fn(data, state)

        lab.kernel_fn = kernel_fn
        lab.Lab._tokens = 0
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
    v13 = [p for p in printed if p.startswith("[v13]")]
    cat_on = state.get("v13_zeros") is not None
    paths = dryrun_v12.kernel_paths(lab, state)
    pads_equal = True
    for tokens in (1, 5, 8):
        x = torch.randn((tokens, lab.UNPOOL_SLICE)).to(torch.bfloat16)
        pads_equal &= bool(torch.equal(lab.v13_pad(x, state), F.pad(x, (0, lab.HIDDEN - lab.UNPOOL_SLICE))))
    if clean:
        log_name = os.environ.get("DRYRUN_V13_LOG")
        want_cat, want_b = lab.V13_CAT, file_bounds
        if log_name:
            log_text = (DECODE / "profiles" / log_name).read_text().splitlines()
            want_cat = ast.literal_eval([x for x in log_text if "V13_CAT = " in x][-1].split("V13_CAT = ", 1)[1])
            want_b = ast.literal_eval([x for x in log_text if "V12_BOUNDARY = " in x][-1].split("V12_BOUNDARY = ", 1)[1])
        checks = {
            "V13_CAT / V12_BOUNDARY = the log's": lab.V13_CAT == want_cat and file_bounds == want_b,
            "v13 line: concat iff V13_CAT": len(v13) == 1 and ("one concat" in v13[0]) == want_cat,
            "installed = the file's": cat_on == want_cat and state["v12_boundary"] == want_b,
            "the pad equals F.pad (1 / 5 / 8 tokens)": pads_equal,
            "kernel_fn's boundaries follow the file": paths == dryrun_v12.want_paths(want_b),
            "no lab code in the file": not hasattr(lab, "sec_pad") and not hasattr(lab, "Lab"),
        }
    else:
        want_b = {**file_bounds, **({} if nogain else {33: 16})}
        printed_cat = [ln for ln in lines if ln.startswith("V13_CAT = ")]
        printed_b = [ln for ln in lines if ln.startswith("V12_BOUNDARY = ")]
        checks = {
            "pad decision": cat_on == (not nogain) and state.get("v13_cat") == (not nogain),
            "V13_CAT printed = state": bool(printed_cat) and printed_cat[-1] == f"V13_CAT = {cat_on!r}",
            "b32 decision; the file's other picks kept": state["v12_boundary"] == want_b,
            "V12_BOUNDARY printed = state": bool(printed_b)
                and printed_b[-1] == "V12_BOUNDARY = " + repr(dict(sorted(state["v12_boundary"].items()))),
            "the pad equals F.pad (1 / 5 / 8 tokens)": pads_equal,
            "kernel_fn's boundaries follow the picks": paths == dryrun_v12.want_paths(want_b),
            "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
            "no failures": "failures 0" in text and "FAILED" not in text,
        }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v13 ({'clean file' if clean else 'lab'}{' nogain' if nogain else ''}): all checks passed "
          f"(concat {cat_on}, boundaries {state['v12_boundary']})")


if __name__ == "__main__":
    main()
