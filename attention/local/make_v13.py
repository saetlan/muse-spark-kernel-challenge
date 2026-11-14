"""Builds a v13 file: submission_v12.py (the scored v12, byte for byte) + cuda_w3b.py + `V13_ROUTES` + v13_block.py.
Usage: .venv/bin/python attention/local/make_v13.py <output> <v13 routes: {class: ("w3b", stages, pp, nwg, rc)}>"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
ns = {}
exec(open(os.path.join(HERE, "cuda_w3b.py")).read(), ns)
W3B_VARIANTS = {tuple(v[:4]) for v in ns["W3B_VARIANTS"]}
out_name, v13 = sys.argv[1], ast.literal_eval(sys.argv[2])
assert all(c[1] and r[0] == "w3b" and tuple(r[1:]) in W3B_VARIANTS for c, r in v13.items()), v13
src = open(os.path.join(PROB, "submission_v12.py")).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_w3b.py =====\n" + open(os.path.join(HERE, "cuda_w3b.py")).read(),
         "\n\n# ===== v13 routes (make_v13.py) =====\nV13_ROUTES = " + repr(dict(sorted(v13.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v13_block.py =====\n" + open(os.path.join(HERE, "v13_block.py")).read()]
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V13 {v13}")
