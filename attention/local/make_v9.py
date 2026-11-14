"""Builds a v9 file: v8's parts (make_v8.py) + cuda_w1n.py + `V9_ROUTES` + TUNE_UPDATE + SGLANG_FALLBACK +
v9_block.py.
Usage: .venv/bin/python attention/local/make_v9.py <output> <routes dict literal> [<tune dict literal> [<fallback set literal>]]
  routes: as make_v8.py, plus ("w1n", ctas per SM, stages) -> V9_ROUTES (W1-N, AF46); tune: {class: (per_sm, min_keys,
  block, warps, stages)} from L20's TUNE_UPDATE; fallback: L20's SGLANG_FALLBACK."""
import ast
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
W1N_STAGES = {3, 4, 6}   # cuda_w1n.py's OKC_W1N_VARIANTS
out_name = sys.argv[1]
routes = ast.literal_eval(sys.argv[2])
tune = ast.literal_eval(sys.argv[3]) if len(sys.argv) > 3 else {}
fallback = ast.literal_eval(sys.argv[4]) if len(sys.argv) > 4 else set()
v9r = {c: r for c, r in routes.items() if r[0] == "w1n"}
rest = {c: r for c, r in routes.items() if r[0] != "w1n"}
assert all(c[1] and len(r) == 3 and r[2] in W1N_STAGES and r[1] >= 1 for c, r in v9r.items()), v9r
assert all(len(t) == 5 and c[1] for c, t in tune.items()) and all(c[1] and not c[0] for c in fallback), (tune, fallback)
tmp = os.path.join(PROB, out_name)
subprocess.run([sys.executable, os.path.join(HERE, "make_v8.py"), out_name, repr(rest)], check=True, stdout=subprocess.DEVNULL)
src = open(tmp).read()
parts = [src,
         "\n\n# ===== appended: attention/local/cuda_w1n.py =====\n" + open(os.path.join(HERE, "cuda_w1n.py")).read(),
         "\n\n# ===== v9 routes and L20 tune results (make_v9.py) =====\nV9_ROUTES = " + repr(dict(sorted(v9r.items()))) +
         "\nTUNE_UPDATE = " + repr(dict(sorted(tune.items()))) + "\nSGLANG_FALLBACK = " + repr(set(fallback)) + "\n",
         "\n\n# ===== appended: attention/local/v9_block.py =====\n" + open(os.path.join(HERE, "v9_block.py")).read()]
open(tmp, "w").write("".join(parts))
print(f"wrote {tmp}: V9 {v9r}, TUNE_UPDATE {tune}, SGLANG_FALLBACK {fallback}")
