"""Mutation test for emulate_cuda_fused.py (S1b additions): query fold and new-row handling.
Run: .venv/bin/python attention/local/mutation_test_cuda_fused.py"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = open(os.path.join(HERE, "emulate_cuda_fused.py")).read()
MUTATIONS = {
    "query RoPE rotation sign": ("q = torch.stack([e * c - o * s, e * s + o * c], -1).flatten(-2)\n    return q",
                                 "q = torch.stack([e * c + o * s, e * s + o * c], -1).flatten(-2)\n    return q"),
    "qk factor missing": ("+ EPS) * cfg[\"qk_scale_factor\"]", "+ EPS)"),
    "new rows never written (patch skipped)": ("    new_rows(data, sub)\n    o, m", "    o, m"),
    "new key not normalized": ("    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) / 64.0 + EPS)\n", ""),
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
print("mutation_test_cuda_fused:", "every mutation caught" if not survived else f"{len(survived)} SURVIVED")
sys.exit(1 if survived else 0)
