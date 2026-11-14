"""Builds a v15 file: submission_v14.py (the scored v14, byte for byte) + cuda_w3e.py + cuda_uw2e.py (only the kinds the
routes use) + `V15_ROUTES` + v15_block.py.
Usage: .venv/bin/python attention/local/make_v15.py <output> <v15 routes: {class: ("w3e", s, pp, nwg, rc) | ("uw2e", s, pp, cs)}>"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROB = os.path.dirname(HERE)
ns = {}
exec(open(os.path.join(HERE, "cuda_w3e.py")).read(), ns)
W3E = {tuple(v[:4]) for v in ns["W3E_VARIANTS"]}
UW2E = {(4, 1, 4), (4, 1, 8), (4, 1, 16), (6, 0, 16)}
out_name, v15 = sys.argv[1], ast.literal_eval(sys.argv[2])
assert all(c[1] and ((r[0] == "w3e" and tuple(r[1:]) in W3E) or (r[0] == "uw2e" and not c[0] and tuple(r[1:]) in UW2E))
           for c, r in v15.items()), v15
src = open(os.path.join(PROB, "submission_v14.py")).read()
parts = [src]
for kind, f in (("w3e", "cuda_w3e.py"), ("uw2e", "cuda_uw2e.py")):   # both sources always (the block names both kinds)
    parts.append(f"\n\n# ===== appended: attention/local/{f} =====\n" + open(os.path.join(HERE, f)).read())
parts += ["\n\n# ===== v15 routes (make_v15.py) =====\nV15_ROUTES = " + repr(dict(sorted(v15.items()))) + "\n",
          "\n\n# ===== appended: attention/local/v15_block.py =====\n" + open(os.path.join(HERE, "v15_block.py")).read()]
out = os.path.join(PROB, out_name)
open(out, "w").write("".join(parts))
print(f"wrote {out}: V15 {v15}")
