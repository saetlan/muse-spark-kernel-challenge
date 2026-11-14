"""Mutation test of dryrun_v11: plant bugs in a copy of lab_de11.py, each under the scenario where it matters.

Run: .venv/bin/python decode_layer/local/mutate_v11.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
DRY = DECODE / "local" / "dryrun_v11.py"
MUTANTS = {
    "3% rule bypassed": ("tiny", "if not (ok and st1.time(s1) < 0.97 * st1.time(ref1)):", "if not ok:"),
    "CTA search back to 4 / 8": ("normal", "DE11_C = (3, 4, 5, 6, 7, 8)\n", "DE11_C = (4, 8)\n"),
    "small search drops the other instances": ("normal", '    cands = [dict(inst, c=c, pf=0) for inst in _stage_instances("tt" not in base) for c in DE11_C]\n',
                                               '    cands = [dict(base, c=c, pf=0) for c in DE11_C]\n'),
}


def main():
    lab = DECODE / "lab_de11.py"
    for sc in ("normal", "tiny"):
        if subprocess.run([sys.executable, str(DRY), str(lab), sc], capture_output=True).returncode:
            raise SystemExit(f"unmutated lab_de11.py fails scenario {sc}")
    src = lab.read_text()
    caught = 0
    for name, (sc, old, new) in MUTANTS.items():
        if src.count(old) != 1:
            print(f"MUTANT NOT APPLIED ({src.count(old)}): {name}")
            continue
        path = Path("/tmp") / "lab_de11_mutant.py"
        path.write_text(src.replace(old, new, 1))
        ok = subprocess.run([sys.executable, str(DRY), str(path), sc], capture_output=True).returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + f"{name} ({sc})")
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    if caught != len(MUTANTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
