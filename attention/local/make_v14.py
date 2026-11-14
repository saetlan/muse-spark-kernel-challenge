"""Builds a v14 file: submission_v13.py (byte for byte) + cuda_ud4.py + `V14_ROUTES` + v14_block.py.
Usage: .venv/bin/python attention/local/make_v14.py <output> <v14 routes: {class: ("ud4", cs, warps, stages)}>"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
UD4_VARIANTS = {(16, 4, 3), (16, 8, 2), (8, 4, 3), (8, 8, 2)}
out_name, v14 = sys.argv[1], ast.literal_eval(sys.argv[2])
assert all(not c[0] and not c[1] and r[0] == "ud4" and tuple(r[1:]) in UD4_VARIANTS for c, r in v14.items()), v14
src = open(os.path.join(PROB, "submission_v13.py")).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_ud4.py =====\n" + open(os.path.join(HERE, "cuda_ud4.py")).read(),
         "\n\n# ===== v14 routes (make_v14.py) =====\nV14_ROUTES = " + repr(dict(sorted(v14.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v14_block.py =====\n" + open(os.path.join(HERE, "v14_block.py")).read()]
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V14 {v14}")
