"""Mutation test of dryrun_v3: plant bugs in a copy of lab_de5.py; the dry run must fail on each.
(Bugs inside Triton kernel bodies can't show on the CPU, where stand-ins compute the answer; prepare()'s
GPU self-checks against the replaced path cover those and switch the kernel off.)

Run: .venv/bin/python decode_layer/local/mutate_v3.py
"""

import subprocess
import sys
from pathlib import Path

DECODE = Path(__file__).resolve().parents[1]
LAST = "last"  # replace the last occurrence (v3's kernel_fn) rather than the first (D1's)
MUTANTS = {
    "experts swap reverted": ("    expert_output = v3_experts(expert_input, topk_weights, local_ids, state)\n",
                              "    expert_output = _experts(state[\"pre_expert_norm\"](expert_input), topk_weights, local_ids, state)\n", None),
    "router swap reverted": ("            logits = v3_router_logits(router_input, state)\n",
                             "            logits = F.linear(router_input, state[\"router\"])\n", None),
    "norm uses the raw weight": ('state["v3_pre_expert_w"] = state["pre_expert_norm"]._get_effective_weight()',
                                 'state["v3_pre_expert_w"] = state["pre_expert_norm"].weight', None),
    "lab table not installed": ('    state["m11_configs"] = tuple(table)\n', "", None),
    "prefetch ignores the lab": ('    return state["v3_prefetch"].get((cfg["tokens"], bool(cfg["window"])), 0)\n',
                                 '    return 16 << 20\n', None),
}


def main():
    src = (DECODE / "lab_de5.py").read_text()
    caught = 0
    for name, (old, new, _) in MUTANTS.items():
        if src.count(old) < 1:
            print(f"MUTANT NOT APPLIED: {name}")
            continue
        path = Path("/tmp") / "lab_de5_mutant.py"
        path.write_text(src.replace(old, new, 1))
        r = subprocess.run([sys.executable, str(DECODE / "local" / "dryrun_v3.py"), str(path)], capture_output=True, text=True)
        ok = r.returncode != 0
        caught += ok
        print(("caught   " if ok else "MISSED   ") + name)
    print(f"{caught}/{len(MUTANTS)} mutants caught")
    if caught != len(MUTANTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
