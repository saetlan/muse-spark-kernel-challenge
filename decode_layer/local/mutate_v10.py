"""Mutation test of the v10 checks (dryrun_v10.py): plant bugs in copies of lab_de12.py (lab scenarios) and
submission_v10.py (clean file with DE12-style picks pasted in); each must fail its dry run.

Run: .venv/bin/python decode_layer/local/mutate_v10.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
OUT = DECODE / "local" / "_mutants"
DRY = DECODE / "local" / "dryrun_v10.py"
# (file, scenario args) -> {name: (old, new)}
MUTANTS = {
    ("lab_de12.py", ()): {
        "GEMM2's PDL option ignored": ('    elif s2.get("pdl"):', '    elif False:'),
        "m11_run drops the config's post-norm option": ('state, x.shape[0], cfg.get("post"))\n', 'state, x.shape[0], None)\n'),
        "picks applied to every class": ("v10_with(cfg, picks[below]) if cfg is not None and below in picks else cfg",
                                         "v10_with(cfg, next(iter(picks.values()))) if cfg is not None and picks else cfg"),
        "class 5-8 shapes timed for the 1-4 class": ("if s[1] * s[2] < below and (below == 5 or s[1] * s[2] >= 5)]",
                                                     "if s[1] * s[2] < 9 and s[1] * s[2] >= 5]"),
    },
    ("lab_de12.py", ("equal",)): {
        "1.5% rule bypassed": ("totals[best] < totals[-1] - 0.015 * abs(totals[-1])", "totals[best] <= totals[-1]"),
    },
}


def run(path, args):
    return subprocess.run([sys.executable, str(DRY), str(path), *args], capture_output=True, text=True)


def main():
    OUT.mkdir(exist_ok=True)
    caught = total = 0
    for (name, args), mutants in MUTANTS.items():
        src = (DECODE / name).read_text()
        if run(DECODE / name, args).returncode != 0:
            raise SystemExit(f"unmutated {name} fails dryrun_v10 {args}")
        for label, (old, new) in mutants.items():
            total += 1
            if src.count(old) != 1:
                print(f"MUTANT NOT APPLIED ({src.count(old)} matches): {label}")
                continue
            path = OUT / f"mutant10_{name}"
            path.write_text(src.replace(old, new, 1))
            ok = run(path, args).returncode != 0
            caught += ok
            print(("caught   " if ok else "MISSED   ") + f"{label} ({name} {' '.join(args)})")
    print(f"{caught}/{total} mutants caught")
    if caught != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
