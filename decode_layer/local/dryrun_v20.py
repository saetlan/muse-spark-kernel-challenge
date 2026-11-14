"""CPU dry run of the DE23 lab (lab_de23.py = submission_v20 + common + de6 + de12 + de14 + de16 + lab/de23.py).

Uses dryrun_v17's moe25 stand-in (argument-checking, zeros) and dryrun_v4's stubbed world. Fake timings: a layer call
500 us, minus 30 us when E25's M19 ran ("gain") or 3 us ("nogain"); the attention stand-in 15 us less, projections
3 us more (the modes then pick att1 pj0 everywhere: 3% > 2%).

gain      m19_256 installs M19 at 192-320 tokens (6% > 3%): V17_CHOICE printed = state = DE20's + {321: 'm19'};
          kernel_fn runs M19 at 256 tokens, M11b at 1-16, M19 at 32 / 65 / 128, Marlin at 64.
nogain    0.6%: Marlin kept, V17_CHOICE = DE20's.
badscale  a zero E4M3 scale: the E25 path is OFF; the file's V17_CHOICE is printed unchanged.
Also: modes ran (V4_MODE printed), no lab failures, real routing restored, the [v20] line names v14's routes.

Run: .venv/bin/python decode_layer/local/dryrun_v20.py [gain | nogain | badscale]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402
import dryrun_v17  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
FILE_CHOICE = {5: "m11b", 9: "m11b", 17: "m11b", 33: "m19", 192: "m19"}
ROUTES = {"Sd1": "ud4", "Sd16": "ud4", "Sd32": "ud4", "Sv1": "uw2", "Sv16": "uw2", "Sv32": "w1n", "Fd1": "dec4",
          "Fd16": "dec4", "Fd32": "dec4", "Fv1": "w2", "Fv16": "w3b", "Fv32": "w3b"}


def main():
    mode = (sys.argv[1:] or ["gain"])[0]
    lab = dryrun_v4.load(DECODE / "lab_de23.py")
    lab.load_inline = dryrun_v17.install()
    w = base.weights()
    for name in ("w13_sf", "w2_sf"):
        w[name] = torch.full_like(w[name], 0x38)
    if mode == "badscale":
        w["w13_sf"].view(torch.uint8)[5, 0, 0] = 0
    lines, printed = [], []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]

    def fake_time(self, fn, reps=15, cold=True):
        before = dict(base.CALLS)
        fn()
        ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
        t = 100.0
        if "v3 lid" in ran:
            t += 500.0 - ((30.0 if mode == "gain" else 3.0) if "e25 m19 g1" in ran else 0.0)
            if "att" in ran:
                t -= 15.0
            if any(k.startswith("pj__") for k in ran):
                t += 3.0
        return t

    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(w)
    builtins.print = real_print
    text = "\n".join(lines)
    choice = state["v17_choice"]
    got = dryrun_v17.paths(lab, state)
    follows = True
    for t, g in got.items():
        k = dryrun_v17.want(choice, t)
        follows &= g["m11b"] == (k == "m11b") and g["m19"] == (k == "m19") and (not g["m11"] or k is None)
    want_c = {} if mode == "badscale" else ({**FILE_CHOICE, 321: "m19"} if mode == "gain" else dict(FILE_CHOICE))
    printed_c = [ln for ln in lines if ln.startswith("V17_CHOICE = ")]
    want_printed = FILE_CHOICE if mode == "badscale" else want_c
    v20 = [p for p in printed if p.startswith("[v20]")]
    checks = {
        "e25 choice after the lab": choice == want_c,
        "V17_CHOICE printed (the clean file's)": bool(printed_c) and printed_c[-1] == f"V17_CHOICE = {dict(sorted(want_printed.items()))!r}",
        "kernel_fn follows (1-256 tokens)": follows,
        "M19 at 256 tokens iff installed": got[256]["m19"] == (mode == "gain"),
        "modes ran (V4_MODE printed)": any(ln.startswith("V4_MODE = ") for ln in lines) and "modes, routing locked" in text,
        "m19_256 ran" if mode != "badscale" else "m19_256 saw the path OFF":
            ("m19_256 (weighted us" in text) == (mode != "badscale"),
        "real routing restored": not isinstance(state["routing"], lab.DE14Lock),
        "no failures": "failures 0" in text and "FAILED" not in text,
        "[v20] names v14's routes": len(v20) == 1 and str(ROUTES) in v20[0],
        "routed kinds keep their sources": lab.V20_ROUTED_KINDS == sorted(set(ROUTES.values())) and all(
            lab.att__KINDS[k][0] for k in lab.V20_ROUTED_KINDS),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v20 lab ({mode}): all checks passed (choice {choice})")


if __name__ == "__main__":
    main()
