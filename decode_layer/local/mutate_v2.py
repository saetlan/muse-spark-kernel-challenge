"""Mutation test of dryrun_v2: plant bugs in a copy of lab_de1.py; the dry run must fail on each.

Run: .venv/bin/python decode_layer/local/mutate_v2.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "router kernel never used": ('    if not state.get("v2_router"):\n        return F.linear', '    if True:\n        return F.linear'),
    "M11 table cut at 9 tokens": ("    (17, dict(nt=2, d1=4, c1=4, d2=4, c2=4)),\n", ""),
    "tiny_gemm never used": ('    if state.get("v2_tiny") is None:\n', '    if True:\n'),
    "M11 at every size (no 33 cap)": ("    (33, None),\n", "    (1 << 30, dict(nt=4, d1=2, c1=4, d2=2, c2=4)),\n"),
    "summary line dropped": ('    log("SUMMARY " + "; ".join(LAB_SUMMARY))\n', ""),
    "experts swap reverted": ('    expert_output = v2_experts(', '    expert_output = _experts('),
}


def main():
    src = (DECODE / "lab_de1.py").read_text()
    caught = 0
    for name, (old, new) in MUTANTS.items():
        if src.count(old) < 1:
            print(f"MUTANT NOT APPLIED: {name}")
            continue
        path = Path("/tmp") / "lab_de1_mutant.py"
        path.write_text(src.replace(old, new, 1) if name != "experts swap reverted" else
                        src[::-1].replace(old[::-1], new[::-1], 1)[::-1])  # the last occurrence: v2's kernel_fn
        r = subprocess.run([sys.executable, str(DECODE / "local" / "dryrun_v2.py"), str(path)], capture_output=True, text=True)
        ok = r.returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + name)
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    if caught != len(MUTANTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
