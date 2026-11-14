"""CPU dry run of the DE9 lab (lab_de9.py = submission_v7 + lab/common.py + lab/de6.py + lab/de9.py): v7's
prepare (v6's self-checks and L9's setup on a stub extension that validates and records every staging call),
de6's modes section, then de9's staging section under fake timings (attention -10 us, projections +3; staging
o -3, e13 at 8/16 -2/-1, e2 +1, the slice -2.5). Checks the lab's exact picks, that they are installed and
printed, and that the timed kernel_fn issues exactly the installed staging (o, e13 with GEMM1's ks, slice
only at <= 8 tokens, nothing for Marlin classes).

Run: .venv/bin/python decode_layer/local/dryrun_v7.py [lab file]
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de9.py"
    lab = dryrun_v4.load(path)
    calls = dryrun_v4.L9_CALLS
    clean = not hasattr(lab, "sec_stage")
    if clean:  # a clean v7: borrow the lab's input generator; no lab sections run
        lab.lab_input = base.load_lab(DECODE / "lab_de9.py").lab_input
    lines = []
    if not clean:
        real_log = lab.log
        lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    holder = {}

    def fake_time(self, fn, reps=15, cold=True):
        before, n0 = dict(base.CALLS), len(calls)
        fn()
        ran = {k for k, v in base.CALLS.items() if v > before.get(k, 0)}
        t = 100.0 + (200.0 if ran else 0.0)  # positive layer times above the empty graph's, as on the GPU
        t += -(10 if "att" in ran else 0) + (3 if any(k.startswith("pj__") for k in ran) else 0)
        st = holder.get("state")
        for c in calls[n0:]:
            if c[0] == "ranges" and st is not None and c[1] == st["dense"]["o"].data_ptr():
                t -= 3
            elif c[0] == "ranges" and st is not None and c[1] == st["dense"]["pooler"].data_ptr():
                t -= {4: 2.0, 8: 2.5}[round(16 * c[2] / (st["dense"]["pooler"].numel() * 2))]
            elif c[0] == "ranges" and st is not None and c[2] != 1024 * st["dense"]["unpooler"].shape[1] * 2:
                t -= 2.5  # the full unpooler's prefix (> 8 tokens): above the 1% step rule
            elif c[0] == "ranges":
                t -= 2.5
            elif c[0] == "experts" and c[2] == st["m11"]["rt_w13"]:
                t -= {8: 2, 16: 1}.get(c[4], 0)
            elif c[0] == "experts":
                t += 1
        return t
    if not clean:
        lab.Lab.time_graph = fake_time
        lab.Lab.cold_l2 = lambda self: None
    real_setup = lab.l9_setup

    def setup(state, weights):
        holder["state"] = state
        return real_setup(state, weights)
    lab.l9_setup = setup
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    text = "\n".join(lines)
    l9 = state["l9_mode"]
    print("L9 picks:", dict(sorted(l9.items())))

    v8 = hasattr(lab, "l10_setup")

    def want(key):
        tokens = key[0]
        m11 = lab.m11_config_for(state, tokens) is not None
        if v8:  # o -3, e13 8 -2 (M11 classes), slice -2.5 (<= 8), pool 8 -2.5, unp -2.5 (> 8; ties pick 4)
            w = {"pm": 0, "o": 1, "pool": 8}
            if m11:
                w["e13"] = 8
            if tokens <= 8:
                w["slice"] = 1
            else:
                w["unp"] = 4
            return w
        if not m11:  # Marlin classes: only o stages (-3 of 190 us = 1.6% < the 2% rule): no pick
            return {}
        w = {"pm": 0, "o": 1, "e13": 8}
        if tokens <= 8:
            w["slice"] = 1
        return w

    launched = {}
    for t, (b, q, layer) in {1: (1, 1, "sliding"), 8: (1, 8, "full"), 16: (16, 1, "sliding"), 32: (32, 1, "sliding"),
                             256: (32, 8, "full")}.items():
        n0 = len(calls)
        data = lab.lab_input(b, q, layer, seed=t, device="cpu")
        lab.kernel_fn(data, state)
        launched[(t, layer == "sliding")] = calls[n0:]

    def follows(key, got):
        m = l9.get(key, {})
        exp = []
        if m.get("o"):
            exp.append(("ranges", state["dense"]["o"].data_ptr(), state["dense"]["o"].numel() * 2, m.get("pm", 0)))
        cfg = lab.m11_config_for(state, key[0])
        if m.get("e13") and cfg is not None:
            exp.append(("experts", state["m11"]["w13"].data_ptr(), state["m11"]["rt_w13"], cfg["s1"]["ks"], m["e13"], m.get("pm", 0)))
        if m.get("pool"):  # L10: issued right after o, in the same attention window
            w = state["dense"]["pooler"]
            exp.insert(1 if m.get("o") else 0, ("ranges", w.data_ptr(), (w.numel() * 2 * m["pool"] // 16) & ~15, m.get("pm", 0)))
        if m.get("slice") and key[0] <= 8:
            w = state["dense"]["unpooler"]
            exp.append(("ranges", w.data_ptr(), 1024 * w.shape[1] * 2, m.get("pm", 0)))
        if m.get("unp") and key[0] > 8:
            w = state["dense"]["unpooler"]
            exp.append(("ranges", w.data_ptr(), (w.numel() * 2 * m["unp"] // 16) & ~15, m.get("pm", 0)))
        return got == exp

    if clean:
        import ast
        import os
        log_name = os.environ.get("DRYRUN_STAGE_LOG")
        printed_l9 = None
        if log_name:
            ln = [x for x in (DECODE / "profiles" / log_name).read_text().splitlines() if "L9_MODE = " in x][-1]
            printed_l9 = ast.literal_eval(ln.split("L9_MODE = ", 1)[1])
        checks = {
            "l9 set up": any(p.startswith("[l9] staging kernels: on") for p in printed),
            "state = the file's L9_MODE": {k: v for k, v in l9.items() if v} == lab.L9_MODE and l9 is not lab.L9_MODE,
            "L9_MODE = the lab's printed picks": printed_l9 is None or lab.L9_MODE == printed_l9,
            "kernel_fn issues the installed staging (1, 8, 16, 32, 256 tokens)": all(follows(k, g) for k, g in launched.items()),
            "no lab code in the file": not hasattr(lab, "sec_modes") and not hasattr(lab, "Lab"),
        }
        for k, v in checks.items():
            print(("PASS " if v else "FAIL ") + k)
        if not all(checks.values()):
            raise SystemExit(1)
        print("dry run v7 (clean file): all checks passed")
        return
    checks = {
        "l9 set up": any(p.startswith("[l9] staging kernels: on") for p in printed),
        "modes: att only (fake timings)": all(m == {"att": True, "pj": False} for m in state["v4_mode"].values()),
        "stage: the exact picks per case": len(l9) == 12 and all(l9[k] == want(k) for k in l9),
        "L9_MODE printed = state": ("L9_MODE = " + repr(dict(sorted((k, v) for k, v in l9.items() if v)))) in text,
        "kernel_fn issues the installed staging (1, 8, 16, 32, 256 tokens)": all(follows(k, g) for k, g in launched.items()),
        "no failures": "failures 0" in text and "FAILED" not in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("L9_MODE"),
        "log under 12K": len(text) < 12000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v7: all checks passed")


if __name__ == "__main__":
    main()
