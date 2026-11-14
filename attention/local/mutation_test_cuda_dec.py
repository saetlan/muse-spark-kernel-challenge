"""Mutation test for emulate_cuda_dec.py: plants register-mapping bugs one at a time (the kind
a CUDA fragment kernel gets wrong) and requires the quick case set to fail.
Run: .venv/bin/python attention/local/mutation_test_cuda_dec.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_dec.py")).read()
MUTATIONS = {
    "V byte_perm takes both bytes from the first token": ("sel = (j & 3) | ((4 + (j & 3)) << 4)", "sel = (j & 3) | ((j & 3) << 4)"),
    "K word halves swapped": ("bf = torch.stack([b_lo[:, 0], b_lo[:, 1], b_hi[:, 0], b_hi[:, 1]], 1)", "bf = torch.stack([b_hi[:, 0], b_hi[:, 1], b_lo[:, 0], b_lo[:, 1]], 1)"),
    "output d un-permuted wrongly": ("8 * (2 * T_OF + jj) + j] = acc", "8 * j + 2 * T_OF + jj] = acc"),
    "V uses the .x word for every d-tile": ("half = 0 if j < 4 else 4", "half = 0"),
    "second V token row taken from the first": ("my_row[2 * T_OF], my_row[2 * T_OF + 1],", "my_row[2 * T_OF], my_row[2 * T_OF],"),
    "K tile 1 rows from tile 0": ("rk = [my_row[G_OF], my_row[8 + G_OF]]", "rk = [my_row[G_OF], my_row[G_OF]]"),
    "Q pairs packed in the wrong order": ("qa[h, ks] = [f16(rows[LANES, base + i]) for i in range(4)]", "qa[h, ks] = [f16(rows[LANES, base + (i ^ 2)]) for i in range(4)]"),
    "last chunk's tail not masked": ("torch.where(tok < end, sc[nt][:, jj] * qk_scale", "torch.where(tok < end + 8, sc[nt][:, jj] * qk_scale"),
}
survived = []
for name, (old, new) in MUTATIONS.items():
    assert SRC.count(old) == 1, f"anchor not found once: {name}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=HERE, delete=False) as f:
        f.write(SRC.replace(old, new))
        path = f.name
    try:
        res = subprocess.run([sys.executable, path, "--quick"], capture_output=True, text=True, timeout=900)
    finally:
        os.unlink(path)
    caught = res.returncode != 0
    print(f"{'caught  ' if caught else 'SURVIVED'} {name}")
    if not caught:
        survived.append(name)
print("mutation_test_cuda_dec:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
