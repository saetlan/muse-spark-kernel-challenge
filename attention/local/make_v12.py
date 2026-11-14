"""Builds a v12 file: submission_v11.py (the scored v11, byte for byte) + cuda_w3.py + `V12_ROUTES` + v12_block.py.
Usage: .venv/bin/python attention/local/make_v12.py <output> <v12 routes: {class: ("w3", stages, pp, nwg, rc)}>"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
ns = {}
exec(open(os.path.join(HERE, "cuda_w3.py")).read(), ns)
W3_VARIANTS = {tuple(v[:4]) for v in ns["W3_VARIANTS"]}
out_name, v12 = sys.argv[1], ast.literal_eval(sys.argv[2])
assert all(c[1] and r[0] == "w3" and tuple(r[1:]) in W3_VARIANTS for c, r in v12.items()), v12
src = open(os.path.join(PROB, "submission_v11.py")).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_w3.py =====\n" + open(os.path.join(HERE, "cuda_w3.py")).read(),
         "\n\n# ===== v12 routes (make_v12.py) =====\nV12_ROUTES = " + repr(dict(sorted(v12.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v12_block.py =====\n" + open(os.path.join(HERE, "v12_block.py")).read()]
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V12 {v12}")
