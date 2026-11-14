"""Mutation test of dryrun_v7: plant bugs in a copy of lab_de9.py; the dry run must fail on each.

Run: .venv/bin/python decode_layer/local/mutate_v7.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "stage pick not installed": ('            state["l9_mode"][key] = pick\n', "            pass\n"),
    "w13 staged with GEMM2's ks": ('("e13", bank["w13"], bank["w13_s"], bank["rt_w13"], mcfg["s1"]["ks"])',
                                   '("e13", bank["w13"], bank["w13_s"], bank["rt_w13"], mcfg["s2"]["ks"])'),
    "o staged whatever the mode": ('    m = l9_mode(cfg, state)\n    if not m.get("o"):\n        return\n', '    m = l9_mode(cfg, state)\n'),
    "2% rule bypassed": ("            pick = best_mode if best < base - 0.02 * abs(base) else {}\n", "            pick = best_mode\n"),
    "staging mode keyed by batch": ('state.get("l9_mode", {}).get((cfg["tokens"], bool(cfg["window"])), {})',
                                    'state.get("l9_mode", {}).get((cfg["batch"], bool(cfg["window"])), {})'),
    "experts staged on the w2 bank for e13": ('("e13", bank["w13"], bank["w13_s"], bank["rt_w13"],', '("e13", bank["w2"], bank["w2_s"], bank["rt_w2"],'),
}


def main():
    src = (DECODE / "lab_de9.py").read_text()
    r = subprocess.run([sys.executable, str(DECODE / "local" / "dryrun_v7.py"), str(DECODE / "lab_de9.py")], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("unmutated lab_de9.py fails its dry run")
    caught = 0
    for name, (old, new) in MUTANTS.items():
        if src.count(old) != 1:
            print(f"MUTANT NOT APPLIED ({src.count(old)} matches): {name}")
            continue
        path = Path("/tmp") / "lab_de9_mutant.py"
        path.write_text(src.replace(old, new, 1))
        ok = subprocess.run([sys.executable, str(DECODE / "local" / "dryrun_v7.py"), str(path)], capture_output=True, text=True).returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + name)
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    if caught != len(MUTANTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
