"""Mutation test of dryrun_v5: plant bugs in a copy of lab_de7.py, each run under the scenario where it
matters; the dry run must fail on each. The unmutated file passes every scenario first.

Run: .venv/bin/python decode_layer/local/mutate_v5.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
DRY = DECODE / "local" / "dryrun_v5.py"
SCENARIOS = ("normal", "dirty", "err", "tiny", "slowlayer")
MUTANTS = {  # name: (scenario, old, new)
    "exactness ignored": ("dirty", "            ok = ex1 and ex2\n", "            ok = True\n"),
    "128/256 skip the error check vs Marlin": (
        "err", "use = ok and err < 2e-2 and new_us < 0.97 * ref_us", "use = ok and new_us < 0.97 * ref_us"),
    "3% rule bypassed": ("tiny", "if not (ok and st1.time(s1) < 0.97 * st1.time(ref1)):", "if not ok:"),
    "layer revert skipped": ("slowlayer", "        keep_new = below not in sums or t_new <= 1.01 * t_old\n",
                             "        keep_new = True\n"),
    "final table not installed": ("slowlayer", '    state["m11_configs"] = tuple(final)\n', ""),
    "pf not passed to the kernel": ("normal", 's1["c"] * state["sms"], s1.get("pf", 0))', 's1["c"] * state["sms"], 0)'),
    "class boundaries shifted (33-64 to M11)": ("normal", "(33, None), (65, None), (192, None)", "(33, None), (192, None)"),
}


def run(path, scenario):
    return subprocess.run([sys.executable, str(DRY), str(path), scenario], capture_output=True, text=True)


def main():
    lab = DECODE / "lab_de7.py"
    for sc in SCENARIOS:
        r = run(lab, sc)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f"unmutated lab_de7.py fails scenario {sc}")
    src = lab.read_text()
    caught = 0
    for name, (sc, old, new) in MUTANTS.items():
        if src.count(old) < 1:
            print(f"MUTANT NOT APPLIED: {name}")
            continue
        path = Path("/tmp") / "lab_de7_mutant.py"
        path.write_text(src.replace(old, new, 1))
        ok = run(path, sc).returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + f"{name} ({sc})")
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    if caught != len(MUTANTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
