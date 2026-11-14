"""Builds a v15 file with W4: submission_v14.py (the scored v14, byte for byte) + cuda_w4.py + `V15_ROUTES` + v16w4_block.py.
Usage: .venv/bin/python attention/local/make_v15w4.py <output> <routes: {class: ("w4", stages, npair, minb)}>"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
ns = {}
exec(open(os.path.join(HERE, "cuda_w4.py")).read(), ns)
W4 = {tuple(v) for v in ns["W4_VARIANTS"]}
out_name, routes = sys.argv[1], ast.literal_eval(sys.argv[2])
assert all(c[1] and r[0] == "w4" and tuple(r[1:]) in W4 for c, r in routes.items()), routes
src = open(os.path.join(PROB, "submission_v14.py")).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_w4.py =====\n" + open(os.path.join(HERE, "cuda_w4.py")).read(),
         "\n\n# ===== v15 routes (make_v15w4.py) =====\nV15_ROUTES = " + repr(dict(sorted(routes.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v16w4_block.py =====\n" + open(os.path.join(HERE, "v16w4_block.py")).read()]
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V15 {routes}")
