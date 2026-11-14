"""Mutation test for emulate_cuda_ver2.py (S2b staging). Run: .venv/bin/python attention/local/mutation_test_cuda_ver2.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_ver2.py")).read()
MUTATIONS = {
    "producer rows: second half taken from r0": ("row = int((r0 if qq < 8 else r1)[tq & 31])", "row = int((r0 if qq < 12 else r1)[tq & 31])"),
    "fragments ignore the sub-chunk offset": ("sk[((16 * scx + g) * KP", "sk[((0 * scx + g) * KP"),
    "V tile based at the V pitch": ("o = st * STAGE_BYTES + SKEYS * KP + tq * VP + 16 * ch", "o = st * STAGE_BYTES + SKEYS * VP + tq * VP + 16 * ch"),
    "rows prefetched for the same stage": ("nxt = rows_of(i + 1) if i + 1 < nst else None", "nxt = rows_of(i) if i + 1 < nst else None"),
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
print("mutation_test_cuda_ver2:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
