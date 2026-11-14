"""Builds a v11 file: v10 (make_v10.py with v9's and v10's routes) + cuda_uw2.py + `V11_ROUTES` + v11_block.py.
Usage: .venv/bin/python attention/local/make_v11.py <output> <v9 routes> <v10 routes> <v11 routes: {class: ("uw2", stages, pp, cs)}>"""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
UW2_VARIANTS = {(4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)}
out_name, v9r, v10r, v11 = sys.argv[1], sys.argv[2], sys.argv[3], ast.literal_eval(sys.argv[4])
assert all(c[1] and not c[0] and r[0] == "uw2" and tuple(r[1:]) in UW2_VARIANTS for c, r in v11.items()), v11
subprocess.run([sys.executable, os.path.join(HERE, "make_v10.py"), out_name, v9r, v10r], check=True, stdout=subprocess.DEVNULL)
tmp = os.path.join(PROB, out_name)
src = open(tmp).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_uw2.py =====\n" + open(os.path.join(HERE, "cuda_uw2.py")).read(),
         "\n\n# ===== v11 routes (make_v11.py) =====\nV11_ROUTES = " + repr(dict(sorted(v11.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v11_block.py =====\n" + open(os.path.join(HERE, "v11_block.py")).read()]
open(tmp, "w").write("".join(parts))
print(f"wrote {tmp}: V11 {v11}")
