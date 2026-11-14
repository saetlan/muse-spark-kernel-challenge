"""Mutation test of the v9 checks: plant bugs in copies of lab_de12.py (cut-short modes scenario, dryrun_v9.py)
and submission_v9.py (clean-file dry run, dryrun_v4.py with DE9's picks); each must fail its dry run.

Run: .venv/bin/python decode_layer/local/mutate_v9.py
"""

import os
import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
OUT = DECODE / "local" / "_mutants"
ENV = dict(os.environ, DRYRUN_PICKS_LOG="lab_de9.6d043f8d93604684.log")
# (file, dry run) -> {name: (old, new)}
MUTANTS = {
    ("lab_de12.py", "dryrun_v9.py"): {
        "unreached case set to all off": ('            state["v4_mode"][key] = dict(keep.get(key, {"att": False, "pj": False}))\n',
                                          '            state["v4_mode"][key] = {"att": False, "pj": False}\n'),
        "file's picks restored for reached cases too": ("    if kept:\n        log(f\"modes not reached",
                                                        "    state[\"v4_mode\"].update({k: dict(v) for k, v in keep.items()})\n"
                                                        "    if kept:\n        log(f\"modes not reached"),
        "final V4_MODE not printed": ("        log(f\"modes not reached, file's picks kept: {', '.join(kept)}\")\n"
                                      "    log(f\"V4_MODE = {dict(sorted(state['v4_mode'].items()))!r}\")\n",
                                      "        log(f\"modes not reached, file's picks kept: {', '.join(kept)}\")\n"),
    },
    ("submission_v9.py", "dryrun_v4.py"): {
        "F12V setup skipped": ("        if pj_F12V_ENABLED:\n            pj_f12v_setup(st)\n", ""),
        "F12M setup skipped": ("        if pj_F12M_ENABLED:\n            pj_f12m_setup(st)\n", ""),
        "router at 8 tokens left on P9 (BF16 input)":
            ("pj_CHOICE[(\"router\", 8)] = {'kind': 'fma', 'BN': 16, 'split': 64, 'warps': 8}\n", ""),
        "unpooler slice around bf16_proj's choice": ("        unpooled = v6_unpool_slice(expert_output, cfg, state)\n",
                                                     "        unpooled = v3_unpool_slice(expert_output, state)\n"),
        "projections keyed by batch": ('        c = pj_CHOICE.get((layer, x.shape[0]))\n        if c is not None:\n            return pj_run_choice(c, x, layer, n, state["pj_state"])\n    w = state["dense"]',
                                       '        c = pj_CHOICE.get((layer, cfg["batch"]))\n        if c is not None:\n            return pj_run_choice(c, x, layer, n, state["pj_state"])\n    w = state["dense"]'),
    },
}


def run(dry, path):
    return subprocess.run([sys.executable, str(DECODE / "local" / dry), str(path)], capture_output=True, text=True, env=ENV)


def main():
    OUT.mkdir(exist_ok=True)
    caught = total = 0
    for (name, dry), mutants in MUTANTS.items():
        src = (DECODE / name).read_text()
        if run(dry, DECODE / name).returncode != 0:
            raise SystemExit(f"unmutated {name} fails {dry}")
        for label, (old, new) in mutants.items():
            total += 1
            if src.count(old) != 1:
                print(f"MUTANT NOT APPLIED ({src.count(old)} matches): {label}")
                continue
            path = OUT / f"mutant_{name}"
            path.write_text(src.replace(old, new, 1))
            ok = run(dry, path).returncode != 0
            caught += ok
            print(("caught   " if ok else "MISSED   ") + f"{label} ({name})")
    print(f"{caught}/{total} mutants caught")
    if caught != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
