"""Mutation test for emulate_cuda_dec3.py (D2 ring). Run: .venv/bin/python attention/local/mutation_test_cuda_dec3.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_dec3.py")).read()
MUTATIONS = {
    "V tile written over the K tile": ("o = CHUNK * PITCH + tok * PITCH + 16 * ch", "o = tok * PITCH + 16 * ch"),
    "fragments read with pitch 128": ("k.append(sk[(gg * PITCH + 32 * T_OF", "k.append(sk[(gg * 128 + 32 * T_OF"),
    "rows prefetched for the wrong chunk": ("row_next = row_of(i + stages)", "row_next = row_of(i + stages - 1)"),
    "compute reads the wrong ring stage": ("st = ring[i % stages]", "st = ring[(i + 1) % stages]"),
    "copy token mapping by 4 lanes instead of 8": ("tok, ch = qq * 4 + (ln >> 3), ln & 7", "tok, ch = qq * 4 + ((ln >> 2) & 3), ln & 7"),
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
print("mutation_test_cuda_dec3:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
