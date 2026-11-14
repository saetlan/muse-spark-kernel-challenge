"""Mutation test for emulate_cuda_ver.py (S2a verify kernel).
Run: .venv/bin/python attention/local/mutation_test_cuda_ver.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_ver.py")).read()
MUTATIONS = {
    "V read with the K pitch": ("vb = [sv[st][(vt * VP", "vb = [sv[st][(vt * KP"),
    "compute reads the wrong ring stage": ("st = c % stages\n                    s0 = start", "st = (c + 1) % stages\n                    s0 = start"),
    "causal mask lets one future key in": ("okA = ok & (key <= posA)", "okA = ok & (key <= posA + 1)"),
    "m-tile rows' two tokens swapped": ("vals(qA, 0), vals(qA, 1), vals(qB, 0), vals(qB, 1)", "vals(qB, 0), vals(qB, 1), vals(qA, 0), vals(qA, 1)"),
    "rows prefetched for the wrong chunk": ("rnext = row_of(c + stages)", "rnext = row_of(c + stages - 1)"),
    "masks skipped where the chunk reaches the block's own keys": ("masked = s0 + CHUNK > end or s0 + CHUNK > n - L or window > 0", "masked = s0 + CHUNK > end or window > 0"),
    "partial written to the wrong token": ("tok = 2 * (mt0 + mi) + r\n", "tok = 2 * (mt0 + mi)\n"),
    "K tile 1 reads tile 0's tokens": ("for g in (G_OF, 8 + G_OF)]", "for g in (G_OF, G_OF)]"),
}
survived = []
for name, (old, new) in MUTATIONS.items():
    assert SRC.count(old) == 1, f"anchor not found once: {name}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=HERE, delete=False) as f:
        f.write(SRC.replace(old, new))
        path = f.name
    try:
        res = subprocess.run([sys.executable, path, "--quick"], capture_output=True, text=True, timeout=1200)
    finally:
        os.unlink(path)
    caught = res.returncode != 0
    print(f"{'caught  ' if caught else 'SURVIVED'} {name}")
    if not caught:
        survived.append(name)
print("mutation_test_cuda_ver:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
