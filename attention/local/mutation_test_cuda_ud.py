"""Mutation test for emulate_cuda_ud.py's hierarchical merge. Run: .venv/bin/python attention/local/mutation_test_cuda_ud.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_ud.py")).read()
MUTATIONS = {
    # Not mutated: regrouping the splits into clusters (e.g. strided) is equivalent by construction
    # (merging is associative: any partition gives the same output). The kernel's grouping is the
    # hardware's (consecutive x-blocks form a 1-D cluster), and each cluster writes its own slot
    # ci = split / CS; that pairing is reviewed in the source, not testable here.
    "leader merges only half its ranks": ("sl = slice(ci * cs, ci * cs + cs)", "sl = slice(ci * cs, ci * cs + cs // 2)"),
    "second level adds cluster partials without rescaling": (
        "mm, ll, oo = merge(torch.stack(cm), torch.stack(cl), torch.stack(co))",
        "mm, ll, oo = cm[0], sum(cl), sum(co)"),
    "second level keeps only the last cluster": (
        "mm, ll, oo = merge(torch.stack(cm), torch.stack(cl), torch.stack(co))",
        "mm, ll, oo = cm[-1], cl[-1], co[-1]"),
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
print("mutation_test_cuda_ud:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
