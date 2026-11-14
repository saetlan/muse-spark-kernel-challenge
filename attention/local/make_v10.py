"""Builds a v10 file: v9 (make_v9.py with v9's routes) + cuda_w2.py + `V10_ROUTES` + v10_block.py.
Usage: .venv/bin/python attention/local/make_v10.py <output> <v9 routes dict literal> <v10 routes dict literal>
  v10 routes: {class: ("w2", ctas per SM, stages, pp)} with (stages, pp) in cuda_w2.py's OKC_W2_VARIANTS."""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
W2_VARIANTS = {(6, 0), (6, 1), (8, 1), (4, 1)}
out_name, v9_routes, v10 = sys.argv[1], sys.argv[2], ast.literal_eval(sys.argv[3])
assert all(c[1] and r[0] == "w2" and len(r) == 4 and r[1] >= 1 and tuple(r[2:]) in W2_VARIANTS for c, r in v10.items()), v10
subprocess.run([sys.executable, os.path.join(HERE, "make_v9.py"), out_name, v9_routes], check=True, stdout=subprocess.DEVNULL)
tmp = os.path.join(PROB, out_name)
src = open(tmp).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_w2.py =====\n" + open(os.path.join(HERE, "cuda_w2.py")).read(),
         "\n\n# ===== v10 routes (make_v10.py) =====\nV10_ROUTES = " + repr(dict(sorted(v10.items()))) + "\n",
         "\n\n# ===== appended: attention/local/v10_block.py =====\n" + open(os.path.join(HERE, "v10_block.py")).read()]
open(tmp, "w").write("".join(parts))
print(f"wrote {tmp}: V10 {v10}")
