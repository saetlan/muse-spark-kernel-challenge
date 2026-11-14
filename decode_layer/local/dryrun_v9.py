"""CPU dry run of the DE12 lab (lab_de12.py = submission_v9 + lab/common.py + lab/de6.py + lab/de12.py) when the
modes section runs out of time: a fake clock advances 1 s per timed graph and the lab budget is 22 s, so the
section (after DE12's pdl section) reaches only the first few cases (1 s for the empty graph + 4 s per case; a
case starts while elapsed <= the deadline). The cases reached are read from the lab's "modes stopped before"
entries; at least one must be reached and one not. Checks: the reached cases get the lab's decision (fake
timings: attention on, projections off), every case not reached keeps the file's V4_MODE (DE9's picks), the
final V4_MODE line printed equals the installed state, and the timed kernel_fn follows it.

Run: .venv/bin/python decode_layer/local/dryrun_v9.py [lab file]
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de12.py"
    lab = dryrun_v4.load(path)
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    clock = [0.0]

    def fake_time(self, fn, reps=15, cold=True):
        before = dict(base.CALLS)
        fn()
        clock[0] += 1.0
        ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
        return 100.0 - (10 if "att" in ran else 0) + (3 if any(k.startswith("pj__") for k in ran) else 0)

    real_init = lab.Lab.__init__
    lab.Lab.__init__ = lambda self, budget_s: real_init(self, float(os.environ.get("DRYRUN_V9_BUDGET", "22")))
    lab.Lab.elapsed = lambda self: clock[0]
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    file_modes = {k: dict(v) for k, v in lab.V4_MODE.items()}
    state = lab.prepare(base.weights())
    text = "\n".join(lines)
    mode = state["v4_mode"]
    stopped = {e[len("modes stopped before "):] for e in lab.LAB_SUMMARY if e.startswith("modes stopped before ")}
    names = [n for n, *_ in lab.SCORED]
    n_reached = len(names) - len(stopped)
    order = [(b * q, layer == "sliding") for _n, b, q, layer, _w in lab.SCORED]
    reached, rest = order[:n_reached], order[n_reached:]
    print(f"modes reached {n_reached} of {len(names)} cases")
    printed = [ln for ln in lines if ln.startswith("V4_MODE = ")]
    paths = {}
    for t, (b, q, layer) in {1: (1, 1, "sliding"), 32: (32, 1, "sliding"), 256: (32, 8, "full")}.items():
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, layer, seed=t, device="cpu"), state)
        paths[(t, layer == "sliding")] = set(base.CALLS)
    checks = {
        "some cases reached, some not, in order": 0 < n_reached < len(names) and stopped == set(names[n_reached:]),
        "file's picks differ from the fake-timing decision somewhere unreached":
            any(file_modes[k] != {"att": True, "pj": False} for k in rest),
        "reached cases: the lab's decision": all(mode[k] == {"att": True, "pj": False} for k in reached),
        "unreached cases: the file's picks kept": all(mode[k] == file_modes[k] for k in rest),
        "the unreached cases are named in the log": f"file's picks kept: {', '.join(names[n_reached:])}" in text,
        "last V4_MODE printed = state": bool(printed) and printed[-1] == "V4_MODE = " + repr(dict(sorted(mode.items()))),
        "kernel_fn follows the state (1, 32, 256 tokens)": all(
            ("att" in p) == mode[k]["att"] and any(c.startswith("pj__") for c in p) == mode[k]["pj"] for k, p in paths.items()),
        "no failures": "failures 0" in text and "FAILED" not in text,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v9 (modes cut short): all checks passed")


if __name__ == "__main__":
    main()
