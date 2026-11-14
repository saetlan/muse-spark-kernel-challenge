"""Mutation test of dryrun_v4: plant bugs in a copy of lab_de6.py; the dry run must fail on each.

Run: .venv/bin/python decode_layer/local/mutate_v4.py [lab file, default lab_de6.py]
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
MUTANTS = {
    "attention swap reverted": ("    o = v4_attention(data, qkv, cfg, state)\n", "    o = _v4_attention_d1(data, qkv, state)\n"),
    "router fed the BF16 input at <= 8 tokens": (
        "return pj_run_choice(c, ffn_input if split else router_input, \"router\", N_EXPERTS, state[\"pj_state\"])",
        "return pj_run_choice(c, ffn_input, \"router\", N_EXPERTS, state[\"pj_state\"])"),
    "mode keyed by batch instead of tokens": ('state["v4_mode"].get((cfg["tokens"], bool(cfg["window"])), {})',
                                              'state["v4_mode"].get((cfg["batch"], bool(cfg["window"])), {})'),
    "lab mode not installed": ('            state["v4_mode"][key] = {"att": pick[0], "pj": pick[1]}\n', ""),
    "pj ignores the mode": ('    if v4_mode(cfg, state, "pj") and "pj_state" in state:\n        c = pj_CHOICE.get((layer, x.shape[0]))',
                            '    if "pj_state" in state:\n        c = pj_CHOICE.get((layer, x.shape[0]))'),
}

MUTANTS_V6 = {  # only for files with bf16_proj v4 (parts/v6.py)
    "(router, 8) back on P9 (BF16 input)": ('pj_CHOICE[("router", 8)] = {\'kind\': \'fma\', \'BN\': 16, \'split\': 64, \'warps\': 8}\n', ""),
    "P9 not set up": ("            pj_p9_setup(st)", "            pass"),
}


def main():
    src = (Path(sys.argv[1]) if len(sys.argv) > 1 else DECODE / "lab_de6.py").read_text()
    caught = 0
    mutants = dict(MUTANTS, **(MUTANTS_V6 if "pj_P9_ENABLED" in src else {}))
    for name, (old, new) in mutants.items():
        if src.count(old) < 1:
            print(f"MUTANT NOT APPLIED: {name}")
            continue
        path = Path("/tmp") / "lab_de6_mutant.py"
        path.write_text(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, str(DECODE / "local" / "dryrun_v4.py"), str(path)], capture_output=True, text=True)
        ok = r.returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + name)
    print(f"{caught}/{len(mutants)} mutants caught")
    if caught != len(mutants):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
