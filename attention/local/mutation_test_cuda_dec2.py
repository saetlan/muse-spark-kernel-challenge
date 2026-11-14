"""Mutation test for emulate_cuda_dec2.py (A5b): plants stacked-layout mapping bugs one at a time
and requires the quick case set to fail.
Run: .venv/bin/python attention/local/mutation_test_cuda_dec2.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_dec2.py")).read()
MUTATIONS = {
    "queries leak across the block diagonal": (
        "torch.where(~h0, vals[0], zero), torch.where(~h0, vals[1], zero),\n                                       torch.where(h0, vals[2], zero)",
        "vals[0], vals[1],\n                                       torch.where(h0, vals[2], zero)"),
    "second K half read at the first half's bytes": ("+ 16 * half + torch.arange(16)", "+ 0 * half + torch.arange(16)"),
    "V bytes at 8*g instead of 16*g": ("poff + 16 * G_OF[:, None] + torch.arange(16)", "poff + 8 * G_OF[:, None] + torch.arange(16)"),
    "output d un-permuted wrongly": ("d = 32 * (T_OF & 1) + 16 * jj + j", "d = 32 * (T_OF & 1) + jj + 2 * j"),
    "output takes the other head's accumulators": ("torch.where(HL == 0, acc[j][:, jj], acc[j][:, 2 + jj])", "torch.where(HL == 0, acc[j][:, 2 + jj], acc[j][:, jj])"),
    "head-1 accumulators rescaled with head 0's factor": ("acc[j][:, 2] *= al1", "acc[j][:, 2] *= al0"),
    "P rows of the two heads swapped": ("a = torch.stack([f16(p[0][:, 0]), f16(p[0][:, 1]), f16(p[0][:, 2]), f16(p[0][:, 3]),", "a = torch.stack([f16(p[0][:, 2]), f16(p[0][:, 3]), f16(p[0][:, 0]), f16(p[0][:, 1]),"),
    "tail past `end` not masked": ("ok = tok < end\n", "ok = tok < end + 8\n"),
    "query base offset by head instead of by half": ("base = 32 * (T_OF & 1)", "base = 32 * (T_OF >> 1)"),
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
print("mutation_test_cuda_dec2:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
