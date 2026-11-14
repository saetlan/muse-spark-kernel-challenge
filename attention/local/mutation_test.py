"""Mutation test for the local checks: plants one bug at a time in the emulator's copy of the
kernels' math and requires emulate.py's case set (through the judge's check_implementation)
to fail. A mutation that survives means the checks can't see that class of bug.

Run: .venv/bin/python attention/local/mutation_test.py
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate.py")).read()
MUTATIONS = {
    "window edge off by one": ("vis &= s[None, :] > row_pos[:, None] - window", "vis &= s[None, :] >= row_pos[:, None] - window"),
    "window start (lo) one key early, mask off by one": ("vis &= s[None, :] > row_pos[:, None] - window", "vis &= s[None, :] > row_pos[:, None] - window - 1"),
    "causal mask lets one future key in": ("(s[None, :] <= row_pos[:, None])", "(s[None, :] <= row_pos[:, None] + 1)"),
    "query heads swapped between the pair": ("heads = [kv0 * G + r_g, kv0 * G + r_g + G]", "heads = [kv0 * G + r_g + G, kv0 * G + r_g]"),
    "RoPE rotation sign": ("q_e, q_o = q_e * c - q_o * s, q_e * s + q_o * c", "q_e, q_o = q_e * c + q_o * s, q_e * s + q_o * c"),
    "last key of each request skipped": ("end = min(start + per, n)", "end = min(start + per, n - 1)"),
    "merge forgets to rescale the running output": ("num = num * scale[:, None] + (w[..., None] * oc).sum(1)", "num = num + (w[..., None] * oc).sum(1)"),
    "new keys stored without the cache scale": ("kf = torch.stack([k_e, k_o], -1).reshape(t, hkv, 64) / data[\"k_scale\"]", "kf = torch.stack([k_e, k_o], -1).reshape(t, hkv, 64)"),
}
# Not mutated: dropping v_scale, or forgetting to rescale the merge's denominator, only scales each
# (token, head) output by a constant, which the RMSNorm after attention cancels: equivalent
# mutations by construction, not coverage gaps (both survived when tried, Nov 3).
survived = []
for name, (old, new) in MUTATIONS.items():
    assert SRC.count(old) == 1, f"mutation anchor not found exactly once: {name}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=HERE, delete=False) as f:
        f.write(SRC.replace(old, new))
        path = f.name
    try:
        res = subprocess.run([sys.executable, path, "--quick"], capture_output=True, text=True, timeout=600)
    finally:
        os.unlink(path)
    caught = res.returncode != 0
    fails = sum(line.startswith("FAIL") for line in res.stdout.splitlines())
    print(f"{'caught  ' if caught else 'SURVIVED'} {name} ({fails} failing cases)")
    if not caught:
        survived.append(name)
print("mutation_test:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
