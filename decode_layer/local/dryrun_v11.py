"""CPU dry run of the DE11 lab (lab_de11.py = clean v7 + lab/common.py + lab/de7.py + lab/de11.py): the v7 world of
dryrun_v4.load (sibling stand-ins, L9 stub), the M11 stub behind a proxy that logs every launch's config, fake
timings (each M11 launch 20 us + |CTAs per SM - 6| (tiny: 0.1 x that); Marlin +60). Checks the exact table the
lab must pick and install (normal: the first instance of each stage at 6 CTAs per SM, M11 at 128 / 256;
tiny: small classes keep their base), that it is printed, and that kernel_fn launches exactly those configs.

Run: .venv/bin/python decode_layer/local/dryrun_v11.py [lab file] [normal|tiny]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dryrun_d0 as base  # noqa: E402
import dryrun_v4  # noqa: E402

DECODE = Path(__file__).resolve().parents[1]
LOG = []


class Proxy:
    def __init__(self, inner):
        self.inner = inner

    def m11_gemm(self, *a):
        gated, tt, nt, depth, ks, grid, pf = a[11:18]
        LOG.append((gated, tt, nt, depth, ks, grid, pf))
        return self.inner.m11_gemm(*a)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de11.py"
    scenario = sys.argv[2] if len(sys.argv) > 2 else "normal"
    lab = dryrun_v4.load(path)
    real_inline = lab.load_inline
    lab.load_inline = lambda **kw: Proxy(real_inline(**kw)) if kw["name"] == "m11_ext" else real_inline(**kw)
    if not hasattr(lab, "sec_pf"):  # a clean file with an appended M11 table (make_v4.py --table)
        return clean(lab)
    lines = []
    real_log = lab.log
    lab.log = lambda msg: (lines.append(msg), real_log(msg))[1]
    holder = {}
    scale = 0.1 if scenario == "tiny" else 1.0

    def fake_time(self, fn, reps=15, cold=True):
        n0, before = len(LOG), dict(base.CALLS)
        fn()
        sms = holder["state"]["sms"]
        t = 100.0 + sum(20.0 + scale * abs(grid / sms - 6) for *_, grid, pf in LOG[n0:])
        if any(k.startswith("marlin") and v > before.get(k, 0) for k, v in base.CALLS.items()):
            t += 60.0
        return t
    lab.Lab.time_graph = fake_time
    lab.Lab.cold_l2 = lambda self: None
    real_v3 = lab.v3_m11_setup

    def v3_setup(state, weights):
        holder["state"] = state
        return real_v3(state, weights)
    lab.v3_m11_setup = v3_setup
    state = lab.prepare(base.weights())
    text = "\n".join(lines)
    table, old = dict(state["m11_configs"]), dict(state["m11_configs_old"])
    g1 = [dict(nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in lab.M11_INSTANCES if g == 1]
    g2 = [dict(tt=tt, nt=nt, d=d, ks=ks) for g, tt, nt, d, ks in lab.M11_INSTANCES if g == 0]
    g1l, g2l = [x for x in g1 if x["nt"] >= 2], [x for x in g2 if x["nt"] >= 2]
    large = {"s1": dict(g1l[0], c=6, pf=0), "s2": dict(g2l[0], c=6, pf=0)}
    if scenario == "tiny":
        want_small = {b: {k: dict(v, pf=0) for k, v in old[b].items()} for b in (5, 9, 17, 33)}
    else:
        want_small = {b: {"s1": dict(g1[0], c=6, pf=0), "s2": dict(g2[0], c=6, pf=0)} for b in (5, 9, 17, 33)}
    print("table:", table)
    launched = {}
    for t, (b, q) in {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1), 40: (5, 8), 128: (16, 8), 256: (32, 8)}.items():
        n0 = len(LOG)
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, "sliding", seed=t, device="cpu"), state)
        launched[t] = (LOG[n0:], any(k.startswith("marlin") for k in base.CALLS))

    def follows(t):
        cfg = lab.m11_config_for(state, t)
        calls, marlin = launched[t]
        if cfg is None:
            return marlin and not calls
        s1, s2 = cfg["s1"], cfg["s2"]
        sms = state["sms"]
        return calls == [(1, 2, s1["nt"], s1["d"], s1["ks"], s1["c"] * sms, s1.get("pf", 0)),
                         (0, s2["tt"], s2["nt"], s2["d"], s2["ks"], s2["c"] * sms, s2.get("pf", 0))] and not marlin

    checks = {
        f"{scenario}: small classes picked as expected": all(table[b] == want_small[b] for b in want_small),
        "128 / 256 on M11 at 6 CTAs per SM (beats Marlin)": table[192] == large and table[321] == large,
        "33-64 tokens on Marlin": table[65] is None,
        "kernel_fn follows the table (1, 8, 16, 32, 40, 128, 256 tokens)": all(follows(t) for t in launched),
        "V5_M11_CONFIGS printed = state": ("V5_M11_CONFIGS = " + repr(state["m11_configs"])) in text,
        "no failures": "failures 0" in text and "FAILED" not in text,
        "summary last": text.rfind("SUMMARY") > text.rfind("V5_M11_CONFIGS"),
        "log under 12K": len(text) < 12000,
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print(f"dry run v11 ({scenario}): all checks passed")


def clean(lab):
    """The appended table is what prepare installs (stub errors are 0, so classes above 32 tokens stay), the
    [table] re-check ran, and kernel_fn launches the installed configs (Marlin for 33-64 tokens)."""
    lab.lab_input = base.load_lab(DECODE / "lab_de9.py").lab_input
    printed = []
    import builtins
    real_print = builtins.print
    builtins.print = lambda *a, **k: (printed.append(" ".join(map(str, a))), real_print(*a, **k))[1]
    state = lab.prepare(base.weights())
    builtins.print = real_print
    file_table = dict(lab.V3_M11_CONFIGS)
    got = dict(state["m11_configs"])
    launched = {}
    for t, (b, q) in {1: (1, 1), 8: (1, 8), 16: (16, 1), 32: (32, 1), 40: (5, 8), 128: (16, 8), 256: (32, 8)}.items():
        n0 = len(LOG)
        base.CALLS.clear()
        lab.kernel_fn(lab.lab_input(b, q, "sliding", seed=t, device="cpu"), state)
        launched[t] = (LOG[n0:], any(k.startswith("marlin") for k in base.CALLS))

    def follows(t):
        cfg = lab.m11_config_for(state, t)
        calls, marlin = launched[t]
        if cfg is None:
            return marlin and not calls
        s1, s2, sms = cfg["s1"], cfg["s2"], state["sms"]
        return calls == [(1, 2, s1["nt"], s1["d"], s1["ks"], s1["c"] * sms, s1.get("pf", 0)),
                         (0, s2["tt"], s2["nt"], s2["d"], s2["ks"], s2["c"] * sms, s2.get("pf", 0))] and not marlin

    checks = {
        "table re-check printed": any(p.startswith("[table] M11 classes above 32 tokens vs Marlin") for p in printed),
        "installed table = the file's": got == file_table,
        "v3 self-checks on (M11 included)": all(": on (" in p for p in printed if p.startswith("[v3]")),
        "kernel_fn follows the table (1, 8, 16, 32, 40, 128, 256 tokens)": all(follows(t) for t in launched),
    }
    for k, v in checks.items():
        print(("PASS " if v else "FAIL ") + k)
    if not all(checks.values()):
        raise SystemExit(1)
    print("dry run v11 (clean file): all checks passed")


if __name__ == "__main__":
    main()
